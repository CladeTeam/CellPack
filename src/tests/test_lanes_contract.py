"""Symbolic panels, exact defaults, and slot-based planning contracts."""

from dataclasses import replace

import numpy as np
import pytest

from pycellpack import cellpack, loader


def test_explicit_ids_and_schema_panels(tmp_path):
    assert loader.allowed(np.uint16, np.float64)
    assert not loader.allowed(np.float64, np.float32)
    counts = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    embedding = np.array([[0.5], [0.75]], dtype=np.float32)
    a = cellpack.write(tmp_path / "counts.cpk", counts)
    b = cellpack.write(tmp_path / "embedding.cpk", embedding)
    # Standard ids: 0 and 1 for the counts' columns 1 and 0, 2 unused, 3 the embedding.
    source = loader.DataSource([], 4)
    assert source.add(a, [1, 0]) == 0
    assert source.add(b, [3]) == 2
    with pytest.raises(TypeError):
        source.add(a)
    schema = loader.OutputSchema([0, 2, 1], np.uint64, default=2**63 + 19)
    loading = loader.Loader(source, schema, backend="pread", workers=1, inflight=1, slot_rows=2)
    with loading.start(np.arange(4, dtype=np.uint32)) as run:
        result = np.concatenate([batch.data.copy() for batch in run])
        run.join()
    fill = 2**63 + 19
    np.testing.assert_array_equal(result, np.array([[2, fill, 1], [4, fill, 3], [fill, fill, fill], [fill, fill, fill]], dtype=np.uint64))
    with pytest.raises(ValueError, match="uint32"):
        loader.OutputSchema(["a"])


def test_online_non_power_of_two_slot_and_schema_counts(tmp_path):
    values = np.arange(45, dtype=np.uint16).reshape(15, 3)
    file = cellpack.write(tmp_path / "rows.cpk", values)
    source = loader.DataSource([(file, np.arange(3))], 3)
    schemas = [loader.OutputSchema([0, 1, 2], dtype, id=i) for i, dtype in enumerate([np.uint16, np.float32, np.float64])]
    loading = loader.Loader(source, schemas, workers=2, inflight=2, backend="pread", slot_rows=7)
    cells = np.tile(np.arange(15, dtype=np.uint32), 40)
    accesses = loader.Accesses.replicate(cells, 3)
    options = replace(loading.plan_options(1 << 20), horizon=loading.window)
    with loading.start(accesses, plan=options) as run:
        actual = []
        for batch in run:
            with batch:
                actual.append(batch["0"].copy())
                np.testing.assert_array_equal(batch["0"], batch["1"])
                np.testing.assert_array_equal(batch["0"], batch["2"])
        run.join()
    np.testing.assert_array_equal(np.concatenate(actual), values[cells])
    with pytest.raises(ValueError, match="horizon"):
        loading.start(accesses, plan=replace(options, gap=3, horizon=2))


def test_plan_units_and_external_ring_layout(tmp_path):
    import os
    x = np.arange(30, dtype=np.uint16).reshape(10, 3)
    file = cellpack.write(tmp_path / "ring.cpk", x)
    loading = loader.Loader(loader.DataSource([(file, np.arange(3))], 3), loader.OutputSchema([2, 0], np.uint16),
                            workers=1, inflight=1, backend="pread", slot_rows=3)
    layout = loading.ring_layout(slots=2)
    assert layout.slot_stride == 4096
    assert layout.data_min_bytes == 4096 + 3 * 4
    storage = np.empty(layout.data_min_bytes + 64, dtype=np.uint8)
    start = (-storage.ctypes.data) % 64 + 8
    data = storage[start:start + layout.data_min_bytes]
    assert data.ctypes.data % 8 == 0 and data.ctypes.data % 64 != 0
    ring = loader.Ring(layout, data=data)
    cells = np.arange(10, dtype=np.uint32)
    plan = loading.compile(cells, options=loading.plan_options(4096))
    original_cpus = os.sched_getaffinity(0)
    pinned = replace(loading.plan_options(4096), threads=1, cpus=(min(original_cpus),))
    assert loading.compile(cells, options=pinned).summary == plan.summary
    assert os.sched_getaffinity(0) == original_cpus
    plan.write(tmp_path / "plan.bin")
    plan = loader.Plan.open(tmp_path / "plan.bin")
    original = loading.replay(plan, cells)
    model = loading.plan_options(4096).cost
    priced = loading.replay(plan, cells, cost=replace(model, access=model.access + 1))
    assert priced.requests == original.requests
    assert priced.worker_seconds == pytest.approx(original.worker_seconds + len(cells), abs=1e-6)
    assert plan.summary.worker_seconds == original.worker_seconds
    assert plan.schemas == 1 and plan.slot_rows == 3
    assert plan.distance == plan.reuse_gap * plan.slot_rows * plan.schemas
    assert sum(plan.summary.quarter_requests) == plan.summary.requests
    assert len(plan.summary.quarter_service_seconds) == 4
    with loading.start(cells, plan=plan, ring=ring) as run:
        actual = np.concatenate([b.data.copy() for b in run])
        run.join()
    np.testing.assert_array_equal(actual, x[:, [2, 0]])
    with loading.start(np.empty(0, dtype=np.uint32), plan=loading.plan_options(0)) as run:
        assert list(run) == []
        stats = run.join()
        assert stats.accesses == 0 and stats.online.accesses == 0
