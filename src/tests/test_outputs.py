"""Several schemas over one data source: aligned slots, typed outputs, plans,
generations and lease ownership."""

import numpy as np
import pytest

from pycellpack import cellpack, loader


def graph_fixture(tmp_path):
    x = np.arange(64, dtype=np.float32).reshape(16, 4)
    path = tmp_path / "source.cpk"
    cellpack.write(path, x, chunk_rows=4)
    source = loader.DataSource([(path, np.arange(4))], 4)
    schemas = {
        "a": loader.OutputSchema.dense(np.arange(4), id=100),
        "b": loader.OutputSchema.dense(np.arange(4), np.float64, id=200),
        "mapped": loader.OutputSchema.dense([1, 0], id=300),
    }
    loading = loader.Loader(
        source,
        schemas,
        config=loader.LoaderConfig(workers=2, backend="pread", inflight=2, profile=True, slot_rows=2),
    )
    return x, loading


def test_schemas_share_slots_and_reads(tmp_path):
    x, loading = graph_fixture(tmp_path)
    cells = np.array([8, 3, 0, 8, 15, 2, 7, 4, 9], np.uint32)
    accesses = loader.Accesses.replicate(cells, 3)
    ring = loading.ring(slots=2)
    actual = {name: [] for name in loading.schemas}
    with loading.start(accesses, ring=ring) as run:
        generation = run.generation
        for batch in run:
            with batch:
                assert batch.generation == generation
                for name in batch:
                    view = batch[name]
                    actual[name].append(view.copy())
                    del view
        stats = run.join()
    np.testing.assert_array_equal(np.concatenate(actual["a"]), x[cells])
    b = np.concatenate(actual["b"])
    assert b.dtype == np.float64
    np.testing.assert_array_equal(b, x[cells])
    np.testing.assert_array_equal(np.concatenate(actual["mapped"]), x[cells][:, [1, 0]])
    assert all(stats.outputs[name].accesses == len(cells) for name in loading.schemas)
    # A position's schemas share one read and one decode.
    assert stats.page_uses == stats.page_reads and stats.decoded_rows == len(cells)
    assert stats.profile.site("worker").time >= 0
    # A ring is reused by a new run (a new generation).
    with loading.start(accesses, ring=ring) as second:
        assert second.generation > generation
        for batch in second:
            batch.release()
        assert second.join().accesses == 3 * len(cells)


def test_rows_cached_for_every_schema(tmp_path):
    x, loading = graph_fixture(tmp_path)
    cells = np.tile(np.arange(16, dtype=np.uint32), 4)
    accesses = loader.Accesses.replicate(cells, 3).fill(0, 3 * len(cells))
    options = loading.plan_options(4096 * 20)
    assert set(options.cost.outputs) == {"a", "b", "mapped"}
    options = loader.PlanOptions(cache_bytes=options.cache_bytes, gap=options.gap, rows=True, cost=loader.CostModel(**{**vars(options.cost), "input": loader.InputCost(decode=1e-3)}))
    plan = loading.compile(accesses, options=options)
    assert plan.summary.accesses == len(accesses)
    assert plan.summary.row_hits > 0
    plan.write(tmp_path / "plan.bin")
    restored = loader.Plan.open(tmp_path / "plan.bin")
    assert loading.replay(restored, accesses) == plan.summary
    assert restored.summary == plan.summary
    actual = []
    with loading.start(accesses, plan=restored, slots=2) as run:
        for batch in run:
            with batch:
                actual.append(batch["mapped"].copy())
        stats = run.join()
    assert stats.accesses == len(accesses) and stats.row_hits > 0
    np.testing.assert_array_equal(np.concatenate(actual), x[cells][:, [1, 0]])


def test_cancel_keeps_exported_views_valid(tmp_path):
    x, loading = graph_fixture(tmp_path)
    accesses = loader.Accesses.replicate(np.arange(16, dtype=np.uint32), 3)
    run = loading.start(accesses, slots=1)
    batch = next(run)
    view = batch["a"]
    batch.release()
    run.close()
    np.testing.assert_array_equal(view, x[:2])
    with pytest.raises(RuntimeError):
        run.join()
    # A fresh ring can run after cancellation.
    with loading.start(accesses, slots=2) as fresh:
        for batch in fresh:
            batch.release()
        assert fresh.join().accesses == 48


def test_ring_layout_access_count_and_plan_identity_errors(tmp_path):
    _, loading = graph_fixture(tmp_path)
    cells = np.arange(8, dtype=np.uint32)
    accesses = loader.Accesses.replicate(cells, 3).fill(0, 24)
    other = loader.Ring(loader.RingLayout(2, 2, [("a", None, 16), ("b", None, 16), ("c", None, 8)]))
    with pytest.raises(ValueError, match="ring"):
        loading.start(accesses, ring=other)
    # Every position needs an access of every schema.
    with pytest.raises(ValueError, match="multiple"):
        loading.start(cells)
    plan = loading.compile(accesses, 4096 * 20)
    with pytest.raises(ValueError):
        loading.replay(plan, accesses[::-1].copy())


def test_world_ranks_and_python_interrupt(tmp_path, python_sigint):
    import _thread
    import threading
    from concurrent.futures import ThreadPoolExecutor

    x, loading = graph_fixture(tmp_path)
    cells = np.arange(16, dtype=np.uint32)
    accesses = loader.Accesses.replicate(cells, 3)
    with loading.start(accesses, slots=2, world=2) as run:
        pool = ThreadPoolExecutor(2)
        try:

            def consume(rank):
                with run.consumer(rank) as consumer:
                    return [(batch.first_position, batch["mapped"].copy()) for batch in consumer]

            futures = [pool.submit(consume, r) for r in range(2)]
            rows = sorted((row for f in futures for row in f.result(timeout=10)), key=lambda r: r[0])
            assert run.join().accesses == 48
        except BaseException:
            run.cancel()
            raise
        finally:
            pool.shutdown()
    np.testing.assert_array_equal(np.concatenate([r[1] for r in rows]), x[:, [1, 0]])
    run = loading.start(accesses, slots=1)
    first = next(run)
    held = first["a"]
    first.release()
    timer = threading.Timer(0.1, _thread.interrupt_main)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            next(run)
    finally:
        timer.join()
        run.close()
    np.testing.assert_array_equal(held, x[:2])
