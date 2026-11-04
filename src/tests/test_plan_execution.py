"""Plan execution across schema groups, frame sizes and oversized inputs."""

import threading
from dataclasses import replace

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.loader._config import uring_supported


def consume(loading, accesses, plan=None):
    with loading.start(accesses, plan=plan, slots=2) as run:
        watchdog = threading.Timer(10, run.cancel)
        watchdog.start()
        try:
            result = {name: [] for name in loading.schemas}
            for batch in run:
                with batch:
                    for name, parts in result.items():
                        parts.append(batch[name].copy())
            stats = run.join()
            return {name: np.concatenate(parts) for name, parts in result.items()}, stats
        finally:
            watchdog.cancel()
            watchdog.join()


@pytest.mark.parametrize("backend", ["pread", "uring"])
@pytest.mark.parametrize("online,chunk", [(False, None), (True, 1), (True, 5), (True, 16384)])
def test_nonadjacent_schemas_use_rows_admitted_by_their_task(tmp_path, backend, online, chunk):
    if backend == "uring" and not uring_supported():
        pytest.skip("io_uring unavailable")
    values = np.arange(32, dtype=np.uint16).reshape(2, 16) + 1
    file = cellpack.write(tmp_path / "rows.cpk", values)
    source = loader.DataSource([(file, np.arange(16))], 16)
    schemas = [loader.OutputSchema(np.arange(16), np.uint16, id=i) for i in range(3)]
    loading = loader.Loader(source, schemas, workers=2, inflight=2, slot_rows=1, backend=backend)
    accesses = np.tile(np.array([0, 1, 0], np.uint32), 20)
    options = replace(loading.plan_options(1 << 20), rows=True, threads=2, chunk=chunk)
    options = replace(options, cost=replace(options.cost, input=replace(options.cost.input, decode=1.0)))
    plan = options if online else loading.compile(accesses, options=options)
    result, stats = consume(loading, accesses, plan)
    assert stats.row_hits > 0
    for i, output in enumerate(result.values()):
        np.testing.assert_array_equal(output, values[accesses[i::3]])


@pytest.mark.parametrize("page_size", [512, 4096, 8192])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("online", [False, True])
def test_cached_rows_follow_source_frame_size(tmp_path, page_size, sparse, online):
    values = np.random.default_rng(43).integers(1, 1000, (4, 1700), dtype=np.uint16)
    stored = values
    if sparse:
        values[:, 1::2] = 0
        row, col = np.nonzero(values)
        stored = cellpack.CSR(np.arange(5, dtype=np.uint64) * 850, col.astype(np.uint32), values[row, col], values.shape)
    file = cellpack.write(tmp_path / "rows.cpk", stored, page_size=page_size)
    loading = loader.Loader(loader.DataSource([(file, np.arange(1700))], 1700),
                            loader.OutputSchema(np.arange(1700), np.uint16),
                            workers=2, inflight=2, slot_rows=2, backend="pread")
    accesses = np.tile(np.arange(4, dtype=np.uint32), 20)
    options = replace(loading.plan_options(1 << 20), rows=True, threads=1, chunk=5)
    options = replace(options, cost=replace(options.cost, input=replace(options.cost.input, decode=1.0)))
    plan = options if online else loading.compile(accesses, options=options)
    result, stats = consume(loading, accesses, plan)
    assert stats.row_hits > 0
    np.testing.assert_array_equal(next(iter(result.values())), values[accesses])


@pytest.mark.parametrize("backend", ["pread", "uring"])
def test_one_position_can_read_multiple_outlier_records(tmp_path, backend):
    if backend == "uring" and not uring_supported():
        pytest.skip("io_uring unavailable")
    small, columns = 4000, 32768
    large = np.random.default_rng(4).integers(1, 65535, (3, columns), dtype=np.uint16)
    indptr = np.concatenate((np.arange(small + 1, dtype=np.uint64), small + np.arange(1, 4, dtype=np.uint64) * columns))
    indices = np.concatenate((np.zeros(small, np.uint32), np.tile(np.arange(columns, dtype=np.uint32), 3)))
    data = np.concatenate((np.ones(small, np.uint16), large.ravel()))
    file = cellpack.write(tmp_path / "outliers.cpk", cellpack.CSR(indptr, indices, data, (small + 3, columns)))
    source = loader.DataSource([(file, np.arange(columns))], columns)
    schemas = [loader.OutputSchema([0, 1, columns - 1], np.uint16, id=i) for i in range(3)]
    loading = loader.Loader(source, schemas, workers=1, inflight=1, slot_rows=1, backend=backend)
    result, stats = consume(loading, np.arange(small, small + 3, dtype=np.uint32))
    assert stats.accesses == 3
    assert stats.private_bytes <= loading.memory_estimate().private_frame_bytes
    for i, output in enumerate(result.values()):
        np.testing.assert_array_equal(output, large[i:i + 1, [0, 1, columns - 1]])


def test_small_plan_ring_holds_a_wide_decompressed_row(tmp_path):
    values = np.ones((2, 32768), dtype=np.uint16)
    file = cellpack.write(tmp_path / "wide.cpk", values, page_size=512)
    loading = loader.Loader(loader.DataSource([(file, np.arange(32768))], 32768),
                            loader.OutputSchema([0, 32767], np.uint16),
                            config=loader.LoaderConfig(workers=1, backend="pread", inflight=1,
                                                       slot_rows=1, window=1))
    accesses = np.tile(np.arange(2, dtype=np.uint32), 10)
    options = replace(loading.plan_options(1 << 20), rows=True, threads=1, chunk=1)
    options = replace(options, cost=replace(options.cost, input=replace(options.cost.input, decode=1.0)))
    result, stats = consume(loading, accesses, options)
    assert stats.row_hits > 0
    np.testing.assert_array_equal(next(iter(result.values())), values[accesses][:, [0, 32767]])
