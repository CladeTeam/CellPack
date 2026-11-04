"""Runs fail with a reason, instead of hanging, when they cannot finish;
shared regions reach other processes even without room in /dev/shm."""

import mmap
import multiprocessing as mp
import os
import threading

import numpy as np
import pytest

from pycellpack import cellpack, loader


def fixture(tmp_path, *, cells=12):
    x = np.arange(cells * 4, dtype=np.float32).reshape(cells, 4)
    source = loader.DataSource([(cellpack.write(tmp_path / "x.cpk", x, chunk_rows=3), np.arange(4))], 4)
    loading = loader.Loader(source, workers=2, backend="pread", slot_rows=2)
    return x, loading, np.arange(cells, dtype=np.uint32)


def test_join_before_consuming_fails_instead_of_hanging(tmp_path):
    x, loading, accesses = fixture(tmp_path)
    with loading.start(accesses, slots=2) as run, pytest.raises(  # 6 batches, 2 slots
        RuntimeError, match="consume the run before join"
    ):
        run.join()
    # Batches that all fit in the ring can be joined first and read after.
    with loading.start(accesses[:4], slots=2) as run:
        assert run.join().accesses == 4
        np.testing.assert_array_equal(np.concatenate([b.data.copy() for b in run]), x[:4])
    # Another thread consuming keeps join() waiting normally.
    out = []
    with loading.start(accesses, slots=2) as run:
        worker = threading.Thread(target=lambda: out.extend(b.data.copy() for b in run))
        worker.start()
        assert run.join().accesses == len(accesses)
        worker.join()
    np.testing.assert_array_equal(np.stack(out).reshape(-1, 4), x)


LAYOUT = loader.RingLayout(2, 2, [("X", np.float32, 16)])


def _consume(control, data, mode, queue):
    ring = loader.Ring(LAYOUT, data=data, control=control, initialize=False)
    session = ring.consumer(0)
    rows = []
    for batch in session:
        rows.append(batch["X"].copy())
        if mode == "early" and len(rows) == 1:
            break
        if mode == "crash" and len(rows) == 1:
            os._exit(3)  # dies holding its registration, without leaving
    session.close()
    queue.put(len(rows))


def _shared():
    maps = [mmap.mmap(-1, n) for n in (LAYOUT.control_bytes, LAYOUT.data_bytes)]
    arrays = [np.ndarray((len(m),), np.uint8, buffer=m) for m in maps]
    return maps, arrays


@pytest.mark.parametrize("mode", ["all", "early", "crash"])
def test_registered_consumers_in_other_processes(tmp_path, mode):
    maps, (control, data) = _shared()
    ring = loader.Ring(LAYOUT, data=data, control=control)
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    # Fork before the loader starts threads; the consumer also attaches
    # before the run starts, as separate processes may.
    child = ctx.Process(target=_consume, args=(control, data, mode, queue))
    child.start()
    _, loading, accesses = fixture(tmp_path)
    with loading.start(accesses, ring=ring) as run:
        if mode == "all":
            assert run.join().accesses == len(accesses)
            assert queue.get(timeout=10) == 6
        else:
            match = "left before taking" if mode == "early" else "exited without finishing"
            with pytest.raises(RuntimeError, match=match):
                run.join()
        child.join(timeout=10)
    assert child.exitcode == (3 if mode == "crash" else 0)
    del ring, run, control, data
    for m in maps:
        m.close()


def _produce_and_exit(tmp_path, control, data):
    ring = loader.Ring(LAYOUT, data=data, control=control, initialize=False)
    _, loading, accesses = fixture(tmp_path)
    run = loading.start(accesses, ring=ring)
    os._exit(0 if run else 1)  # dies registered as the producer, mid-run


def test_consumer_reports_a_dead_producer(tmp_path):
    maps, (control, data) = _shared()
    ring = loader.Ring(LAYOUT, data=data, control=control)
    child = mp.get_context("fork").Process(target=_produce_and_exit, args=(tmp_path, control, data))
    child.start()
    child.join(timeout=10)
    # Batches the producer wrote before it died are taken; the next wait fails.
    with ring.consumer(0) as session, pytest.raises(RuntimeError, match="producer .* exited"):
        for _ in session:
            pass
    del ring, control, data
    for m in maps:
        m.close()


@pytest.mark.parametrize("transport", ["shm", "memfd", "file"])
def test_transports_share_one_mapping(tmp_path, monkeypatch, transport):
    from pycellpack.loader import _transport as t

    monkeypatch.setenv("PYCELLPACK_SHM_DIR", str(tmp_path))
    chosen, regions = t.choose({"a": 4096, "b": 8192}, "test" + transport, transport)
    assert chosen == transport
    server = None
    try:
        fds = {}
        if transport == "memfd":
            server = t.FdServer({k: r.fd for k, r in regions.items()})
            fds = t.receive_fds(server.address)
        peer = t.Region(regions["b"].descriptor(), fd=fds.get("b"))
        peer.buffer[:5] = b"hello"
        assert regions["b"].buffer[:5] == b"hello"
        peer.close()
        for fd in fds.values():
            os.close(fd)
    finally:
        if server is not None:
            server.stop()
        for region in regions.values():
            region.close()
            region.unlink()
    assert not list(tmp_path.glob("pycellpack-*"))


def test_auto_transport_falls_back_when_dev_shm_is_full(monkeypatch):
    from pycellpack.loader import _transport as t

    monkeypatch.setattr(t, "_free", lambda path: 0 if str(path) == "/dev/shm" else 1 << 40)
    chosen, regions = t.choose({"a": 4096}, "fallback", "auto")
    try:
        assert chosen == "memfd"
    finally:
        for region in regions.values():
            region.close()
            region.unlink()
