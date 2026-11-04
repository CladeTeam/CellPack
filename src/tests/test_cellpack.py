"""File, buffer ownership and bounded producer/consumer contracts."""

import gc
import mmap
import multiprocessing as mp
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pycellpack import cellpack as store
from pycellpack import loader as cellpack


def write_dense(path, matrix, chunk_rows=3):
    with store.Writer(
        path, matrix.shape[1], matrix.dtype, format="dense", chunk_rows=chunk_rows
    ) as writer:
        writer.append(matrix)


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    ],
)
def test_store_dense_and_csr_into(tmp_path, dtype):
    a = np.array(
        [[0, 3, 0, 2], [1, 0, 4, 0], [0, 0, 0, 0], [0, 2, 1, 0], [4, 0, 0, 3]],
        dtype=dtype,
    )
    path = tmp_path / "m.cpk"
    write_dense(path, a)
    file = store.File(path)
    selection = file.prepare_read(slice(1, 5))
    del file
    out = np.full((4, 4), 99, dtype=dtype)
    assert selection.read_dense(out=out) is out
    np.testing.assert_array_equal(out, a[1:])
    out.fill(17)
    selection.read_dense(out=out)
    np.testing.assert_array_equal(out, a[1:])
    for integer in (np.int32, np.int64):
        csr = selection.read_csr(index_dtype=integer)
        assert csr.indices.dtype == integer
        assert csr.indptr.dtype == integer
        np.testing.assert_array_equal(csr.to_scipy().toarray(), a[1:])
        assert np.shares_memory(csr.to_scipy().data, csr.data)
    empty = store.Store(path)[2:2].read_csr()
    assert empty.shape == (0, 4)
    np.testing.assert_array_equal(empty.indptr, [0])


@pytest.mark.parametrize("integer", [np.int32, np.int64, np.uint32, np.uint64])
def test_csr_inputs_borrow_integer_arrays(tmp_path, integer):
    ip = np.array([0, 2, 2, 3], dtype=integer)
    ix = np.array([0, 3, 2], dtype=integer)
    vx = np.array([5, 7, 9], dtype=np.float32)
    path = tmp_path / "s.cpk"
    with store.Writer(path, 4, np.float32, chunk_rows=2) as w:
        w.append(store.CSR(ip, ix, vx, (3, 4)))
    got = store.Store(path).read_csr()
    np.testing.assert_array_equal(got.indptr, ip)
    np.testing.assert_array_equal(got.indices, ix)
    np.testing.assert_array_equal(got.data, vx)


def test_output_validation_precedes_writes(tmp_path):
    path = tmp_path / "v.cpk"
    write_dense(path, np.eye(3, dtype=np.float32))
    s = store.File(path).prepare_read()
    for out in (
        np.full((3, 3), 7, dtype=np.float64),
        np.full((3, 6), 7, dtype=np.float32)[:, ::2],
        np.full((2, 3), 7, dtype=np.float32),
        np.full((3, 3), 7, dtype=">f4"),
    ):
        before = out.copy()
        with pytest.raises(ValueError):
            s.read_dense(out=out)
        np.testing.assert_array_equal(out, before)
    out = np.full((3, 3), 7, dtype=np.float32)
    out.flags.writeable = False
    with pytest.raises(ValueError):
        s.read_dense(out=out)
    shared = np.full(30, 7, dtype=np.int32)
    before = shared.copy()
    with pytest.raises(ValueError, match="overlap"):
        s.read_csr(out=(shared[:4], shared[1:4], shared[2:5].view(np.float32)))
    np.testing.assert_array_equal(shared, before)
    with pytest.raises(IndexError):
        store.File(path).prepare_read([0, 4])


@pytest.mark.parametrize("bad", [[-1, 1], [0, 2**32], [1, 0], [0, 4]])
def test_invalid_csr_indices_are_rejected(tmp_path, bad):
    with (
        pytest.raises(ValueError),
        store.Writer(tmp_path / "bad.cpk", 4, np.float32, chunk_rows=1) as w,
    ):
        w.append(store.CSR(
            np.array([0, 2], np.int64), np.array(bad, np.int64), np.ones(2, np.float32), (1, 4)
        ))


def test_scipy_append_keeps_csr_layout(tmp_path):
    scipy = pytest.importorskip("scipy.sparse")
    a = scipy.csr_array(np.eye(5, dtype=np.uint16))
    path = tmp_path / "scipy.cpk"
    with store.Writer(path, 5, a.dtype, chunk_rows=2) as w:
        w.append(a)
    np.testing.assert_array_equal(store.Store(path).toarray(), a.toarray())


def fixture_loader(tmp_path, *, slots=2, batch_rows=2, world=1, plan=False):
    a = np.arange(1, 41, dtype=np.float32).reshape(10, 4)
    a[::2, 1] = 0
    path = tmp_path / "load.cpk"
    write_dense(path, a)
    source = cellpack.DataSource([(path, np.arange(4))], 4)
    accesses = np.array([8, 3, 2, 3, 1, 0, 9, 7, 4], np.uint32)
    loader = cellpack.Loader(source, workers=2, inflight=2, backend="pread", slot_rows=batch_rows)
    p = (
        loader.compile(accesses, options=cellpack.PlanOptions(cache_bytes=4096 * 4, objective="misses"))
        if plan
        else None
    )
    layout = loader.ring_layout(slots=slots, world=world)
    data = np.empty(layout.data_bytes, np.uint8)
    ring = cellpack.Ring(layout, data=data)
    return a, accesses, loader, ring, (data, layout.slot_stride), p


@pytest.mark.parametrize("planned", [False, True])
def test_loader_into_numpy_ring(tmp_path, planned):
    a, accesses, loader, ring, (data, stride), plan = fixture_loader(tmp_path, plan=planned)
    if plan:
        plan.write(tmp_path / "plan.bin")
        restored = cellpack.Plan.open(tmp_path / "plan.bin")
        assert restored.distance == plan.distance
        assert loader.replay(restored, accesses) == plan.summary
    with loader.start(accesses, ring=ring, plan=plan) as run:
        for batch in run:
            with batch:
                x = batch.data
                assert np.shares_memory(x, data)
                assert x.ctypes.data == data[(batch.index % ring.slots) * stride :].ctypes.data
                expected = a[accesses[batch.first_position : batch.first_position + batch.rows]]
                np.testing.assert_array_equal(x, expected)
                del x
        stats = run.join()
        assert stats.accesses == len(accesses)
        assert run.finished
    # A ring is reused by a new run (a new generation).
    with loader.start(accesses, ring=ring, plan=plan) as again:
        assert sum(b.rows for b in again) == len(accesses)
        assert again.join().accesses == len(accesses)


def test_exported_view_retains_slot_and_memory(tmp_path):
    a, accesses, loader, ring, (data, _), _ = fixture_loader(tmp_path, slots=1, batch_rows=1)
    data_ref = weakref.ref(data)
    with loader.start(accesses, ring=ring) as run:
        batch = next(run)
        x = batch.data
        subview = x[:, 1:]
        first = x.copy()
        batch.release()
        with pytest.raises(RuntimeError, match="released"):
            _ = batch.data
        del data, x
        gc.collect()
        assert data_ref() is not None
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(next, run)
            time.sleep(0.1)
            assert not future.done()
            np.testing.assert_array_equal(subview, first[:, 1:])
            del subview
            gc.collect()
            second = future.result(timeout=5)
            np.testing.assert_array_equal(second.data, a[accesses[1:2]])
            second.release()
    del ring, run, batch, second
    gc.collect()
    assert data_ref() is None


def test_tensor_retains_numpy_slot(tmp_path):
    torch = pytest.importorskip("torch")
    a, accesses, loader, ring, _data, _ = fixture_loader(tmp_path, slots=1, batch_rows=1)
    with loader.start(accesses, ring=ring) as run:
        batch = next(run)
        array = batch.data
        tensor = torch.from_numpy(array)
        assert tensor.data_ptr() == array.ctypes.data
        batch.release()
        del array
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(next, run)
            time.sleep(0.1)
            assert not future.done()
            np.testing.assert_array_equal(tensor.numpy(), a[accesses[:1]])
            del tensor
            gc.collect()
            future.result(timeout=5).release()


def test_cancel_with_live_views_and_invalid_inputs(tmp_path):
    _a, accesses, loader, ring, _data, _ = fixture_loader(tmp_path, slots=1)
    with pytest.raises(TypeError):
        loader.start(accesses.astype(np.float64), ring=ring)
    with pytest.raises(ValueError):
        loader.start(np.array([999], np.uint32), ring=ring)
    run = loader.start(accesses, ring=ring)
    batch = next(run)
    x = batch.data
    copy = x.copy()
    run.close()
    np.testing.assert_array_equal(x, copy)
    batch.release()
    del batch, x
    with pytest.raises(RuntimeError, match="aborted"):
        run.join()


def test_world_consumers_and_explicit_ids(tmp_path):
    a = np.arange(24, dtype=np.float32).reshape(6, 4)
    path = tmp_path / "map.cpk"
    write_dense(path, a)
    mapping = np.array([2, 0, 0xFFFFFFFF, 1], np.uint32)
    source = cellpack.DataSource([(path, mapping)], 3)
    loader = cellpack.Loader(source, workers=2, inflight=2, backend="pread", slot_rows=1)
    accesses = np.arange(6, dtype=np.uint32)
    ring = loader.ring(slots=2, world=2)
    with loader.start(accesses, ring=ring) as run:

        def consume(rank):
            out = []
            with run.consumer(rank) as consumer:
                for batch in consumer:
                    out.append((batch.first_position, batch["X"].copy()))
            return out

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(consume, r) for r in range(2)]
            actual = sorted([row for f in futures for row in f.result(timeout=5)], key=lambda r: r[0])
        np.testing.assert_array_equal(np.concatenate([x for _, x in actual]), a[:, [1, 3, 0]])
        assert run.join().accesses == 6


def ipc_consumer(layout, control, data, queue):
    try:
        ring = cellpack.Ring(layout, data=data, control=control, initialize=False)
        with ring.consumer(0) as consumer:
            actual = [batch["X"].copy() for batch in consumer]
        queue.put(np.concatenate(actual))
    except (RuntimeError, ValueError, OSError) as e:
        queue.put(str(e))


def test_shared_regions_across_processes(tmp_path):
    ctx = mp.get_context("fork")
    a = np.arange(24, dtype=np.float32).reshape(6, 4)
    path = tmp_path / "ipc.cpk"
    write_dense(path, a)
    loader = cellpack.Loader(cellpack.DataSource([(path, np.arange(4))], 4), workers=2, inflight=2, backend="pread", slot_rows=2)
    layout = loader.ring_layout(slots=2)
    with mmap.mmap(-1, layout.control_bytes) as cm, mmap.mmap(-1, layout.data_bytes) as dm:
        control = np.ndarray((layout.control_bytes,), np.uint8, buffer=cm)
        data = np.ndarray((layout.data_bytes,), np.uint8, buffer=dm)
        ring = cellpack.Ring(layout, data=data, control=control)
        queue = ctx.Queue()
        process = ctx.Process(target=ipc_consumer, args=(layout, control, data, queue))
        process.start()
        with loader.start(np.arange(6, dtype=np.uint32), ring=ring) as run:
            actual = queue.get(timeout=10)
            process.join(timeout=5)
            assert process.exitcode == 0
            np.testing.assert_array_equal(actual, a)
            run.join()
        del ring, data, control, run
        gc.collect()


def test_ring_wait_observes_python_interrupt(python_sigint):
    import _thread

    ring = cellpack.Ring(cellpack.RingLayout(1, 1, [("x", None, 4)]))
    timer = threading.Timer(0.1, _thread.interrupt_main)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            ring.consumer(0).wait(0)
    finally:
        timer.join()
