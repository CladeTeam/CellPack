"""Shared rings and the distributed producer release their resources on
every path, and one rank's run works end to end without torch."""

import io
import os
import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.loader import _producer

LAYOUT = loader.RingLayout(2, 2, [("X", np.float32, 12)])


def test_shared_resources_close_when_run_close_fails(monkeypatch):
    closed = []

    def fail():
        raise RuntimeError("run close failed")

    run = SimpleNamespace(join=lambda: SimpleNamespace(to_dict=dict), close=fail)
    shared = SimpleNamespace(owner=False, ring=SimpleNamespace(abort=lambda: None), close=lambda: closed.append(True))
    stub = SimpleNamespace(start=lambda *a, **kw: run)
    monkeypatch.setattr(_producer, "_follow_parent", lambda: None)
    monkeypatch.setattr(_producer.SharedRing, "attach", staticmethod(lambda descriptor: shared))
    monkeypatch.setattr(_producer, "_Status", lambda shared: SimpleNamespace(publish=lambda *a: None))
    request = {"loader": stub, "shared": {}, "accesses": [], "access_count": None, "plan": None, "plan_path": None}
    monkeypatch.setattr(_producer.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(_dump(request))))
    with pytest.raises(RuntimeError, match="run close failed"):
        _producer.main()
    assert closed == [True]


def _dump(request):
    # The stub loader is not picklable; substitute a reference resolved at load.
    _producer._test_stub = request.pop("loader")
    request["loader"] = _StubRef()
    return pickle.dumps(request)


class _StubRef:
    def __reduce__(self):
        return _stub, ()


def _stub():
    return _producer._test_stub


def test_unicode_failure_fits_shared_status(tmp_path, monkeypatch):
    from pycellpack.loader._distributed import _ERROR, _STATUS_BYTES, _Status

    monkeypatch.setenv("PYCELLPACK_SHM_DIR", str(tmp_path))
    with loader.SharedRing.create(LAYOUT, transport="file", extra={"status": _STATUS_BYTES}) as shared:
        status = _Status(shared)
        assert status.publish(_ERROR, {"message": "\U0001f600" * 6000})
        state, payload = status.read()
        assert state == _ERROR
        assert payload["message"] and set(payload["message"]) == {"\U0001f600"}
    assert not list(tmp_path.glob("pycellpack-*"))


def test_descriptor_server_start_failure_closes_regions(monkeypatch):
    from pycellpack.loader import _shared, _transport

    descriptors = []
    choose = _transport.choose

    def record(*args):
        result = choose(*args)
        descriptors.extend(r.fd for r in result[1].values())
        return result

    def fail(*args):
        raise OSError("descriptor server failed")

    monkeypatch.setattr(_shared, "choose", record)
    monkeypatch.setattr(_shared, "FdServer", fail)
    try:
        with pytest.raises(OSError, match="descriptor server failed"):
            loader.SharedRing.create(LAYOUT, transport="memfd")
        assert descriptors
        for fd in descriptors:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        for fd in descriptors:
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.parametrize("failure", ["unlink", "server"])
def test_shared_cleanup_attempts_all_regions(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("PYCELLPACK_SHM_DIR", str(tmp_path))
    shared = loader.SharedRing.create(LAYOUT, transport="file", extra={"status": 64})
    regions = list(shared._regions.values())

    def fail():
        raise OSError("injected cleanup failure")

    try:
        with monkeypatch.context() as patch:
            if failure == "unlink":
                patch.setattr(regions[0], "unlink", fail)
            else:
                patch.setattr(shared, "_server", SimpleNamespace(stop=fail))
            with pytest.raises(OSError, match="injected cleanup failure"):
                shared.close()
            assert all(region.fd == -1 for region in regions)
            assert all(not region.path.exists() for region in regions[1:])
    finally:
        shared.close()


def test_shared_ring_attaches_by_its_descriptor(tmp_path, monkeypatch):
    monkeypatch.setenv("PYCELLPACK_SHM_DIR", str(tmp_path))
    with loader.SharedRing.create(LAYOUT, transport="file") as shared:
        peer = loader.SharedRing.attach(pickle.loads(pickle.dumps(shared.descriptor)))
        assert peer.layout == LAYOUT and peer.ring.generation == shared.ring.generation
        peer.close()


def test_shared_extras_keep_ring_storage_separate():
    for name in ("control", "data"):
        with pytest.raises(ValueError, match="extra region names"):
            loader.SharedRing.create(LAYOUT, extra={name: 1})


def test_oversized_success_reports_a_producer_error():
    from pycellpack.loader._distributed import _ERROR, _SUCCESS, _Status

    with loader.SharedRing.create(LAYOUT, transport="memfd", extra={"status": 256}) as shared:
        status = _Status(shared)
        assert status.publish(_SUCCESS, {"stats": {"large": "x" * 1000}})
        state, payload = status.read()
        assert state == _ERROR
        assert "statistics exceed" in payload["message"]


def _publish_status(shared, connection):
    from pycellpack.loader._distributed import _ERROR, _Status

    status = _Status(shared)
    connection.send("ready")
    status.publish(_ERROR, {"message": "child"})
    connection.send("published")
    connection.close()


def test_inherited_memfd_status_serializes_processes():
    import fcntl
    import multiprocessing as mp

    from pycellpack.loader._distributed import _ERROR, _Status

    context = mp.get_context("fork")
    with loader.SharedRing.create(LAYOUT, transport="memfd", extra={"status": 1024}) as shared:
        shared.stop_sharing()
        status = _Status(shared)
        parent, child = context.Pipe()
        process = context.Process(target=_publish_status, args=(shared, child))
        fcntl.lockf(status.fd, fcntl.LOCK_EX)
        try:
            process.start()
            child.close()
            assert parent.poll(5) and parent.recv() == "ready"
            assert not parent.poll(0.1)
            assert status.publish(_ERROR, {"message": "parent"})
        finally:
            fcntl.lockf(status.fd, fcntl.LOCK_UN)
            if process.pid is not None:
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
            parent.close()
        assert process.exitcode == 0
        assert status.read() == (_ERROR, {"message": "parent"})


@pytest.fixture
def one_rank(monkeypatch):
    """A one-rank torch.distributed stand-in: collectives return the local
    values."""
    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: True
    dist.is_initialized = lambda: True
    dist.get_rank = lambda group=None: 0
    dist.get_world_size = lambda group=None: 1
    dist.get_global_rank = lambda group, rank: rank

    def all_gather_object(out, value, group=None):
        out[0] = value

    dist.all_gather_object = all_gather_object
    dist.broadcast_object_list = lambda values, src=0, group=None: None
    torch = types.ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


@pytest.mark.parametrize("kind", ["array", "computed"])
def test_producer_process_feeds_a_rank(tmp_path, one_rank, kind):
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    path = cellpack.write(tmp_path / "x.cpk", x, chunk_rows=3)
    schema = loader.OutputSchema([3, 0], np.float64, name="pair")
    loading = loader.Loader(loader.DataSource([(path, np.arange(4))], 4), schema, workers=2, backend="pread", slot_rows=2)
    accesses = np.array([8, 3, 2, 3, 1, 0, 9], np.uint32) if kind == "array" else loader.Accesses.shuffle(10, 2, seed=5)
    cells = accesses if kind == "array" else accesses.fill()
    with loader.DistributedLoader(loading).start(accesses, plan=4096 * 4) as run:
        assert run.producer_pid != os.getpid() and len(run) == -(-len(cells) // 2)
        rows = [batch["pair"].copy() for batch in run]
        stats = run.join()
    np.testing.assert_array_equal(np.concatenate(rows), x[cells][:, [3, 0]].astype(np.float64))
    assert stats.accesses == len(cells) and stats.online.accesses == len(cells)


def test_producer_failure_reaches_the_rank(tmp_path, one_rank):
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    path = cellpack.write(tmp_path / "x.cpk", x)
    loading = loader.Loader(loader.DataSource([(path, np.arange(4))], 4), workers=1, backend="pread", slot_rows=2)
    with pytest.raises(RuntimeError, match="outside the data source"), loader.DistributedLoader(loading).start(np.array([0, 99, 1, 2], np.uint32)) as run:
        for batch in run:
            batch.release()
    with pytest.raises(RuntimeError, match="startup failed: OSError"):
        loader.DistributedLoader(lambda: loader.Loader(loader.DataSource([(tmp_path / "missing.cpk", [0])], 1))).start(np.zeros(2, np.uint32))


def test_producer_reports_large_named_outputs(tmp_path, one_rank):
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = cellpack.write(tmp_path / "x.cpk", x)
    name = "\U0001f600" * 6000
    loading = loader.Loader(loader.DataSource([(path, np.arange(4))], 4),
                            {name: loader.OutputSchema([3, 0], np.float64)},
                            workers=1, backend="pread", slot_rows=2, profile=True)
    with loader.DistributedLoader(loading).start(np.arange(4, dtype=np.uint32)) as run:
        rows = [batch[name].copy() for batch in run]
        stats = run.join()
    np.testing.assert_array_equal(np.concatenate(rows), x[:, [3, 0]])
    assert stats.outputs[name].accesses == 4
    assert stats.profile is not None
