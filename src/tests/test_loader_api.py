"""Python-first loader declarations: one data source, typed output schemas,
paired accesses and aligned slots."""

import numpy as np
import pytest

from pycellpack import cellpack, loader


def test_computed_access_errors_keep_their_exception_type(tmp_path):
    _, _, file, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(file, np.arange(4))], 4),
                            workers=1, backend="pread", inflight=1, slot_rows=1)
    invalid = loader.Accesses.offset(np.array([0], dtype=np.uint32), [100])
    with loading.start(invalid) as run:
        with pytest.raises(ValueError, match="outside the data source"):
            run.wait(0)
        with pytest.raises(ValueError, match="outside the data source"):
            run.join()
    # Closing a failed run releases the loader for subsequent work.
    with loading.start(np.array([0], dtype=np.uint32)) as run:
        with run.wait(0) as batch:
            np.testing.assert_array_equal(batch.data, np.arange(4, dtype=np.float32)[None])
        run.join()


def test_native_run_polling_uses_bounded_timeouts(tmp_path):
    _, _, file, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(file, np.arange(4))], 4),
                            workers=1, backend="pread", inflight=1, slot_rows=1)
    with loading.start(np.arange(12, dtype=np.uint32), slots=1) as run:
        with run.wait(0):
            # A leased slot keeps the producer unfinished without relying
            # on the relative speeds of IO and the polling caller.
            assert run.wait_finished(0) is False
            with pytest.raises(ValueError, match="1000 ms"):
                run.wait_finished(1001)
        run.cancel()


def matrices(tmp_path):
    x = np.arange(48, dtype=np.float32).reshape(12, 4)
    y = np.arange(60, dtype=np.float32).reshape(12, 5)
    a = cellpack.write(tmp_path / "a.cpk", x, chunk_rows=3)
    b = cellpack.write(tmp_path / "b.cpk", y, chunk_rows=4)
    return x, y, a, b


@pytest.mark.parametrize("keep_view", [False, True])
def test_close_releases_iteration_batch_but_preserves_exported_arrays(tmp_path, keep_view):
    x, _, file, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(file, np.arange(4))], 4),
                            workers=1, backend="pread", inflight=1, slot_rows=1)
    ring = loading.ring(slots=1)
    with loading.start(np.array([0], np.uint32), ring=ring) as run:
        batch = next(run)
        view = batch.data if keep_view else None
        assert run.wait_finished(1000)
    assert run.succeeded
    assert batch.held == keep_view
    if view is not None:
        np.testing.assert_array_equal(view, x[:1])
        with pytest.raises(ValueError, match="unconsumed"):
            loading.start(np.array([0], np.uint32), ring=ring)
        del view
    with loading.start(np.array([0], np.uint32), ring=ring) as fresh:
        with next(fresh):
            pass
        fresh.join()


def test_consumption_progress_reclaims_completed_prefixes(tmp_path):
    x, _, file, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(file, np.arange(4))], 4),
                            workers=1, backend="pread", inflight=1, slot_rows=1)
    accesses = np.arange(300, dtype=np.uint32) % len(x)
    with loading.start(accesses, slots=4, world=2) as run:
        consumers = [run.consumer(rank) for rank in range(2)]
        for offset in range(0, len(accesses), 4):
            for index in [offset + 2, offset, offset + 3, offset + 1]:
                with consumers[index % 2].wait(index) as batch:
                    np.testing.assert_array_equal(batch.data, x[accesses[index:index + 1]])
        assert run.join().accesses == len(accesses)


def test_failed_run_close_does_not_report_success(tmp_path):
    _, _, file, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(file, np.arange(4))], 4), workers=1, backend="pread")
    with loading.start(loader.Accesses.offset(np.array([0], np.uint32), [100])) as run, pytest.raises(ValueError, match="outside the data source"):
        run.wait(0)
    assert run.finished and not run.succeeded


def test_access_fill_validates_the_range_before_allocating():
    accesses = loader.Accesses.replicate(np.arange(3, dtype=np.uint32), 2)
    np.testing.assert_array_equal(accesses.fill(1, 4), [0, 1, 1, 2])
    assert len(accesses.fill(6, 0)) == 0
    for start, count in [(7, 0), (5, 2), (2**64 - 1, 2), (0, 2**62)]:
        with pytest.raises(ValueError, match="access range"):
            accesses.fill(start, count)


def test_memory_estimate_uses_planning_mode_and_dimensions(tmp_path):
    from dataclasses import replace

    _, _, file, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(file, np.arange(4))], 4),
                            workers=1, backend="pread", inflight=1, slot_rows=1)
    options = loader.PlanOptions(cache_bytes=0, gap=4, chunk=1, threads=1)
    empty = loading.memory_estimate()
    zero = loading.memory_estimate(plan=options)
    assert empty.plan_ring_bytes == 0 < zero.plan_ring_bytes
    assert zero.cache_frame_bytes == 0
    assert loading.memory_estimate(plan=replace(options, chunk=100)).plan_ring_bytes > zero.plan_ring_bytes
    assert loading.memory_estimate(plan=replace(options, gap=100)).plan_ring_bytes > zero.plan_ring_bytes
    rows = replace(loading.plan_options(4096 * 20), rows=True)
    assert loading.memory_estimate(plan=rows).worker_scratch_bytes > loading.memory_estimate(plan=replace(rows, rows=False)).worker_scratch_bytes
    assert loading.memory_estimate(plan=replace(rows, objective="misses")).worker_scratch_bytes == empty.worker_scratch_bytes
    plan = loading.compile(np.arange(12, dtype=np.uint32), options=options)
    assert loading.memory_estimate(plan=plan).plan_ring_bytes > 0
    assert loading.memory_estimate(4096 * 4).cache_frame_bytes == 4096 * 4
    with pytest.raises(ValueError, match="frame count"):
        loading.memory_estimate(plan=replace(options, cache_bytes=(2**32) * 4096))


def test_unplanned_run_does_not_reserve_a_plan_ring(tmp_path):
    x, _, file, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(file, np.arange(4))], 4),
                            config=loader.LoaderConfig(workers=1, backend="pread", inflight=1,
                                                       slot_rows=1, window=2**64 - 1))
    assert loading.memory_estimate().plan_ring_bytes == 0
    with loading.start(np.array([0], np.uint32)) as run:
        with next(run) as batch:
            np.testing.assert_array_equal(batch.data, x[:1])
        run.join()


def test_config_schema_and_profile(tmp_path):
    x, _, a, _ = matrices(tmp_path)
    source = loader.DataSource([(a, np.arange(4))], 4)
    schema = loader.OutputSchema.dense(np.arange(4), id=23, name="expression")
    options = {
        "workers": 2,
        "backend": loader.IoBackend.PREAD,
        "inflight": 2,
        "claim_batch": 1,
        "slot_rows": 3,
        "window": 4,
        "profile": True,
    }
    config = loader.LoaderConfig(**options)
    assert config.profile == ("all",)
    loading = loader.Loader(source, schema, config=config)
    assert loading.accesses_per_slot == 3 and loading.window == 4
    accesses = np.array([9, 0, 3, 1, 9, 2, 7], np.uint32)
    out = []
    with loading.start(accesses, slots=2) as run:
        assert run.batches == 3
        for batch in run:
            with batch:
                assert set(batch) == {"expression"}
                out.append(batch.data.copy())
        stats = run.join()
    np.testing.assert_array_equal(np.concatenate(out), x[accesses])
    assert stats.accesses == len(accesses)
    assert stats.io.requests == stats.page_reads == stats.page_uses
    assert stats.written_bytes == len(accesses) * 4 * 4
    assert stats.outputs["expression"].accesses == len(accesses)
    assert set(stats.profile.sites) == set(loader.profile_sites())
    assert stats.profile.site("decode").calls > 0 and stats.profile.site("worker").time > 0
    assert sum(calls for _, calls, _ in stats.profile.site("map").details) == len(accesses)
    with pytest.raises(ValueError):
        loader.LoaderConfig(workers=0)
    with pytest.raises(ValueError):
        loader.LoaderConfig(slot_rows=0)


def test_profile_selects_sites_and_groups(tmp_path):
    x, _, a, _ = matrices(tmp_path)
    source = loader.DataSource([(a, np.arange(4))], 4)
    groups = loader.profile_groups()
    assert {"io", "cpu", "wait", "worker", "planner", "calibrate"} <= set(groups)
    assert loader.LoaderConfig(profile=False).profile is None
    assert loader.LoaderConfig(profile="decode").profile == ("decode",)
    with pytest.raises(ValueError, match="unknown profile"):
        loader.LoaderConfig(profile=["decode", "nope"])
    accesses = np.tile(np.arange(12, dtype=np.uint32), 3)
    for profile, sites in [(None, None), (("io", "decode"), {*groups["io"], "decode"}), ((), set())]:
        loading = loader.Loader(source, workers=2, backend="pread", inflight=2, slot_rows=4, profile=profile)
        with loading.start(accesses, slots=2) as run:
            got = [batch.data.copy() for batch in run]
            live = run.snapshot()
            stats = run.join()
        np.testing.assert_array_equal(np.concatenate(got), x[accesses])
        assert live.accesses <= stats.accesses == len(accesses)
        if sites is None:
            assert stats.profile is None and live.profile is None
        else:
            assert set(stats.profile.sites) == sites


def test_typed_schemas_defaults_and_partial_panels(tmp_path):
    counts = (np.arange(48, dtype=np.uint16).reshape(12, 4) % 5) * 1000
    path = cellpack.write(tmp_path / "counts.cpk", counts, chunk_rows=5)
    source = loader.DataSource([(path, np.arange(4))], 6)
    schemas = [
        loader.OutputSchema.dense([0, 1, 2, 3], np.float32, id=0, name="f32"),
        loader.OutputSchema.dense([3, 1, 5], np.uint32, id=1, name="u32", default=7),
        loader.OutputSchema.dense([2, 0], np.float64, id=2, name="f64", default=-1),
    ]
    loading = loader.Loader(source, schemas, workers=2, inflight=2, backend="pread", slot_rows=4)
    cells = np.array([11, 0, 5, 5, 3, 8, 1], np.uint32)
    got = {s.name: [] for s in schemas}
    with loading.start(loader.Accesses.replicate(cells, 3), slots=2) as run:
        for batch in run:
            with batch:
                for name in batch:
                    got[name].append(batch[name].copy())
        stats = run.join()
    f32, u32, f64 = (np.concatenate(got[k]) for k in ("f32", "u32", "f64"))
    assert f32.dtype == np.float32 and u32.dtype == np.uint32 and f64.dtype == np.float64
    np.testing.assert_array_equal(f32, counts[cells])
    # Genes the matrix has keep their values, unstored zeros included; only
    # id 5, which the matrix has no column for, takes the default.
    np.testing.assert_array_equal(u32, np.column_stack([counts[cells][:, [3, 1]], np.full(len(cells), 7, np.uint32)]))
    np.testing.assert_array_equal(f64, counts[cells][:, [2, 0]].astype(np.float64))
    assert stats.accesses == 3 * len(cells)
    # f32 values cannot become integers; u16 cannot narrow to u8.
    floats = loader.DataSource([(cellpack.write(tmp_path / "f.cpk", np.ones((3, 2), np.float32)), np.arange(2))], 2)
    with pytest.raises(ValueError, match="cannot be output"):
        loader.Loader(floats, loader.OutputSchema.dense([0, 1], np.int32))
    with pytest.raises(ValueError, match="cannot be output"):
        loader.Loader(source, loader.OutputSchema.dense([0], np.uint8))


def test_paired_accesses_share_one_frontier(tmp_path):
    x, y, a, b = matrices(tmp_path)
    # Counts take standard ids 0..3, the embedding 4..8.
    source = loader.DataSource([(a, np.arange(4)), (b, np.arange(4, 9))], 9)
    config = loader.LoaderConfig(workers=2, backend="pread", inflight=2, slot_rows=2)
    counts = loader.OutputSchema.dense(np.arange(4), id=1, name="counts")
    embedding = loader.OutputSchema.dense(np.arange(4, 9), id=2, name="embedding")
    paired = loader.Loader(source, [counts, embedding], config=config)
    sequence = np.array([11, 1, 9, 1, 0, 7, 3], np.uint32)
    out_x, out_y = [], []
    retained = None
    with paired.start(loader.Accesses.offset(sequence, [0, 12]), slots=2) as run:
        for batch in run:
            with batch:
                assert set(batch) == {"counts", "embedding"}
                out_x.append(batch["counts"].copy())
                out_y.append(batch["embedding"].copy())
                if batch.first_position == 6:
                    retained = batch["counts"]
        stats = run.join()
        assert stats.accesses == 2 * len(sequence)
        assert stats.outputs["counts"].decoded_rows == stats.outputs["embedding"].decoded_rows == len(sequence)
    np.testing.assert_array_equal(np.concatenate(out_x), x[sequence])
    np.testing.assert_array_equal(np.concatenate(out_y), y[sequence])
    np.testing.assert_array_equal(retained, x[sequence[-1:]])


def cost(names):
    return loader.CostModel(
        cores=2,
        iops=1000,
        bandwidth=4e6,
        access=1e-6,
        request=1e-5,
        bucket=8,
        input=loader.InputCost(decode=1e-4, decode_element=1e-8),
        rows=loader.RowCost(gather=1e-10, admit=1e-10),
        outputs={name: loader.OutputCost(map_element=1e-9) for name in names},
    )


def test_costed_plan_options(tmp_path):
    _, _, a, _ = matrices(tmp_path)
    sequence = np.tile(np.arange(12, dtype=np.uint32), 4)
    config = loader.LoaderConfig(workers=2, backend="pread", inflight=2, slot_rows=4)
    loading = loader.Loader(loader.DataSource([(a, np.arange(4))], 4), config=config)
    options = loader.PlanOptions(cache_bytes=4096 * 12, cost=cost(["X"]), rows=True)
    plan = loading.compile(sequence, options=options)
    assert plan.distance == loading.window * loading.accesses_per_slot
    assert plan.summary.accesses == len(sequence)
    assert loading.plan_options(4096).cost.outputs.keys() == {"X"}
    assert len(loading.plan_options(4096).cost.convert) == len(loader.CONVERSIONS)
    assert loader.access_fingerprint(sequence) == loader.access_fingerprint(sequence.copy())
    with loading.start(sequence, slots=2, plan=plan) as run:
        for batch in run:
            batch.release()
        stats = run.join()
    # A run serves exactly what the plan's replay counts.
    s = plan.summary
    assert stats.accesses == len(sequence)
    assert (stats.page_uses, stats.page_hits, stats.row_hits, stats.rows_admitted) == (s.page_uses, s.page_hits, s.row_hits, s.row_residencies)
    r = plan.residencies()
    assert len(r["kind"]) == s.page_residencies + s.row_residencies and len(plan) == len(sequence)
    assert (r["frames"] < plan.cache_frames).all() and (r["start"] <= r["end"]).all()


def test_online_plans_over_computed_sequences(tmp_path):
    import ctypes

    x, _, a, _ = matrices(tmp_path)
    config = loader.LoaderConfig(workers=2, backend="pread", inflight=2, slot_rows=4)
    loading = loader.Loader(loader.DataSource([(a, np.arange(4))], 4), config=config)
    options = loader.PlanOptions(cache_bytes=4096 * 12, cost=cost(["X"]), rows=True, threads=2)
    shuffle = loader.Accesses.shuffle(12, 6, seed=3, block=4, group=2)
    assert len(shuffle) == 72
    cells = shuffle.fill(0, 72)
    for e in range(6):
        assert sorted(cells[12 * e : 12 * (e + 1)]) == list(range(12))
    np.testing.assert_array_equal(shuffle.fill(30, 5), cells[30:35])

    # A native function: cell = (position * 5) % 12, from any thread.
    @ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32))
    def strided(context, start, count, out):
        for i in range(count):
            out[i] = (start + i) * 5 % 12
        return 0

    function = loader.Accesses.function(ctypes.cast(strided, ctypes.c_void_p).value, 0, 60)
    expected = np.arange(60, dtype=np.uint64) * 5 % 12
    np.testing.assert_array_equal(function.fill(0, 60), expected)

    for sequence, want in [(shuffle, cells), (function, expected)]:
        out = []
        with loading.start(sequence, slots=2, plan=options) as run:
            for batch in run:
                with batch:
                    out.append(batch.data.copy())
            stats = run.join()
        np.testing.assert_array_equal(np.concatenate(out), x[want])
        assert stats.online.accesses == len(want)
        assert stats.page_hits == stats.online.page_hits and stats.row_hits == stats.online.row_hits

    plan = loading.compile(cells, options, online=True)
    assert plan.summary.accesses == len(cells) and plan.online_stats.accesses == len(cells)
    # With the whole sequence as horizon, the online compiler is the
    # offline one, whatever its chunk.
    full = loader.PlanOptions(cache_bytes=4096 * 12, cost=cost(["X"]), rows=True, threads=2, horizon=len(cells), chunk=5)
    assert loading.compile(cells, full, online=True).summary == loading.compile(cells, full).summary
    assert loading.compile_stats(shuffle, options).accesses == len(cells)


def test_page_reference_and_source_regions(tmp_path):
    import mmap
    import os

    x, _, a, _ = matrices(tmp_path)
    source = loader.DataSource([(a, np.arange(4))], 4)
    trace = loader.PageTrace.from_pages([0, 1, 0, 1], universe=2)
    assert trace.page_uses == 4 and trace.distinct_pages == 2
    assert trace.optimum(2).hits == 2
    assert trace.optimum(0).misses == 4
    accesses = np.arange(12, dtype=np.uint32)
    loading = loader.Loader(source, workers=2, inflight=2)
    plan = loading.compile(accesses, options=loader.PlanOptions(cache_bytes=4096 * 8, objective="misses"))
    assert loading.replay(plan, accesses).page_uses == loader.PageTrace.from_source(source, accesses).page_uses
    assert source.n_matrices == 1 and source.vocab == 4
    assert source.matrix_dtypes == (np.dtype(np.float32),)
    assert source.page_size == 4096
    assert source.max_cell_pages >= 1
    assert isinstance(source.mapping_fingerprint, int)
    fd, offset, length = source.cell_bytes(-1)
    page_fd, page_offset, pages = source.cell_pages(-1)
    assert fd == page_fd and page_offset % 4096 == 0 and pages >= 1
    from pycellpack.cellpack import scans

    # The source descriptor may use O_DIRECT, which requires aligned buffers and offsets.
    with mmap.mmap(-1, pages * source.page_size) as buffer:
        size = os.preadv(fd, [buffer], page_offset)
        start = offset - page_offset
        assert start + length <= size
        assert scans.record_nnz(buffer[start:start + length]) == np.count_nonzero(x[-1])


def test_execution_config_conflicts(tmp_path):
    _, _, a, _ = matrices(tmp_path)
    source = loader.DataSource([(a, np.arange(4))], 4)
    for kwargs in ({"workers": 2}, {"profile": False}, {"cpus": []}, {"slot_rows": 2}):
        with pytest.raises(TypeError, match="config"):
            loader.Loader(source, config=loader.LoaderConfig(), **kwargs)
    with pytest.raises(TypeError):
        loader.Loader([a])


def test_access_callback_owner_survives_native_run(tmp_path):
    import ctypes
    import gc
    import weakref

    x, _, a, _ = matrices(tmp_path)

    class Owner:
        pass

    owner = Owner()

    @ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32))
    def callback(context, start, count, out):
        for i in range(count):
            out[i] = (start + i) % 12
        return 0

    owner.callback = callback
    retained = weakref.ref(owner)
    sequence = loader.Accesses.function(ctypes.cast(callback, ctypes.c_void_p).value, 0, 48, owner=owner)
    loading = loader.Loader(loader.DataSource([(a, np.arange(4))], 4), workers=2, backend="pread", inflight=2, slot_rows=4)
    run = loading.start(sequence, slots=2)
    del owner, callback, sequence
    gc.collect()
    assert retained() is not None
    outputs = []
    with run:
        for batch in run:
            with batch:
                outputs.append(batch.data.copy())
        assert run.join().accesses == 48
    np.testing.assert_array_equal(np.concatenate(outputs), x[np.arange(48) % 12])
    del run, batch
    gc.collect()
    assert retained() is None


def test_store_views_as_loader_sources(tmp_path):
    x, y, a, b = matrices(tmp_path)
    stored = cellpack.write(tmp_path / "n.cpk", x, chunk_rows=5)
    # A Store contributes the cells it selects; the ids map its stored columns.
    view = cellpack.Store(stored)[[10, 2, 7]]
    whole = cellpack.Store(b)
    source = loader.DataSource([(view, [2, -1, -1, 0]), (whole, [1, -1, 2, -1, 0])], 3)
    assert source.vocab == 3 and source.n_cells == 3 + 12
    assert source.matrix_cells(1) == (3, 12)
    first = np.zeros((3, 3), np.float32)
    first[:, [2, 0]] = x[[10, 2, 7]][:, [0, 3]]
    second = np.zeros((12, 3), np.float32)
    second[:, [1, 2, 0]] = y[:, [0, 2, 4]]
    expected = np.concatenate([first, second])
    accesses = np.array([0, 14, 2, 3, 1, 9], np.uint32)
    loading = loader.Loader(source, workers=2, backend="pread", slot_rows=2)
    with loading.start(accesses, slots=2) as run:
        actual = np.concatenate([batch.data.copy() for batch in run])
        run.join()
    np.testing.assert_array_equal(actual, expected[accesses])
    # Computed sequences are translated too, and plans compile from them.
    shuffled = loader.Accesses.shuffle(source.n_cells, 1, 7, 1, 1)
    plan = loading.compile(shuffled, 1 << 20)
    with loading.start(shuffled, plan=plan) as run:
        got = np.concatenate([batch.data.copy() for batch in run])
        run.join()
    np.testing.assert_array_equal(got, expected[shuffled.fill(0, len(shuffled))])
    np.testing.assert_array_equal(loader.load(source, [2, 0], workers=1), expected[[2, 0]])
    with pytest.raises(TypeError, match="cells only"):
        loader.DataSource([(cellpack.Store(a)[:, [1, 0]], np.arange(4))], 4)
    with pytest.raises(TypeError, match="pair"):
        loader.DataSource([a], 4)
    with pytest.raises(TypeError, match="vocab"):
        loader.DataSource([(a, np.arange(4))], None)


def test_schema_ids_slot_guard_and_validation(tmp_path):
    x, _, a, _ = matrices(tmp_path)
    source = loader.DataSource([(a, np.arange(4))], 4)
    first = loader.OutputSchema.dense(np.arange(4), name="all")
    second = loader.OutputSchema.dense([3, 1], name="pair")
    loading = loader.Loader(source, [first, second], workers=1, backend="pread", slot_rows=2)
    assert sorted(s.id for s in loading.schemas.values()) == [0, 1]
    accesses = np.tile(np.arange(6, dtype=np.uint32), 2)  # 6 positions, 3 batches
    with loading.start(accesses, slots=2) as run:
        # Iteration releases the previous batch without a `with` block.
        rows = [batch["all"].copy() for batch in run]
        run.join()
    # Positions take one access per schema.
    np.testing.assert_array_equal(np.concatenate(rows), x[accesses[::2]])
    import threading

    # Batches dropped without release free their slots (nothing else keeps them).
    with loading.start(accesses, slots=2) as run:
        dropped = [run.wait(b)["all"].copy() for b in range(run.batches)]
        run.join()
    np.testing.assert_array_equal(np.concatenate(dropped), x[accesses[::2]])
    with loading.start(accesses, slots=2) as run:
        held = [run.wait(0)["all"], run.wait(1)["all"]]
        threading.Timer(0.2, held.clear).start()
        with pytest.warns(RuntimeWarning, match="still holds"):
            third = run.wait(2)
        np.testing.assert_array_equal(third["all"], x[accesses[8:12:2]])
        third.release()
        run.cancel()
    with pytest.raises(ValueError, match="distinct"):
        loader.Loader(source, [loader.OutputSchema.dense([0], id=3), loader.OutputSchema.dense([1], id=3)])
    with pytest.raises(ValueError, match="uint32"):
        loader.OutputSchema.dense(np.array([2**32 + 3]))
    with pytest.raises(ValueError, match="uint32"):
        loader.DataSource([(a, np.array([0, -5, 1, 2]))], 4)
    with pytest.raises(TypeError):
        loader.DataSource([(a, np.array([0, 1.5, 1, 2]))], 4)
    big = cellpack.write(tmp_path / "big.cpk", x, page_size=8192)
    grown = loader.DataSource([], 1)
    grown.add(big, [0, -1, -1, -1])
    assert grown.page_size == 8192


@pytest.mark.parametrize("output", [("other", np.float32, 16), ("X", np.float64, 16), ("X", None, 16)])
def test_ring_views_match_the_loader_before_start(tmp_path, output):
    x, _, a, _ = matrices(tmp_path)
    loading = loader.Loader(loader.DataSource([(a, np.arange(4))], 4),
                            workers=1, backend="pread", slot_rows=2)
    ring = loader.Ring(loader.RingLayout(2, 2, [output]))
    with pytest.raises(ValueError, match="ring layout"):
        loading.start(np.array([0, 1], np.uint32), ring=ring)
    assert ring.generation == 0
    with loading.start(np.array([0, 1], np.uint32)) as run:
        batch = next(run)
        np.testing.assert_array_equal(batch.data, x[:2])
        batch.release()
        run.join()


def test_ring_dtype_preserves_native_byte_order():
    dtype = np.dtype(np.float32).newbyteorder("S")
    with pytest.raises(ValueError, match="native byte order"):
        loader.RingLayout(1, 2, [("X", dtype, 16)])
