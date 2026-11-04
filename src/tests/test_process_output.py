"""Fixed output layouts, field views and their ownership of ring leases."""

import gc
import pickle

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.loader.process_fn import OutputLayout


def test_output_layout_preserves_c_padding_and_strided_views():
    layout = OutputLayout([("ids", "i8", (3,)), ("length", "u4")])
    assert layout.row_bytes == 32 and layout.offsets == {"ids": 0, "length": 24}
    raw = np.zeros((6, layout.row_bytes), np.uint8)
    fields = layout.unpack(raw)
    fields["ids"][:] = np.arange(18).reshape(6, 3)
    fields["length"][:] = np.arange(6)
    for sliced in (raw, raw[::2], raw[::-1], raw[0], raw[:0]):
        got = layout.unpack(sliced)
        copied = layout.unpack(sliced, copy=True)
        for name, field in got.items():
            np.testing.assert_array_equal(copied[name], field)
            if field.size:
                assert np.shares_memory(field, sliced)
            assert copied[name].flags.c_contiguous and copied[name].flags.aligned
            assert not np.shares_memory(copied[name], sliced)
    assert layout.unpack(raw[0])["length"].shape == ()
    assert layout.unpack(raw[:0])["ids"].shape == (0, 3)
    restored = pickle.loads(pickle.dumps(layout))
    assert restored.dtype == layout.dtype


def test_output_layout_supports_explicit_packed_fields_and_rejects_invalid_buffers():
    layout = OutputLayout([("tag", "u1"), ("value", "f8")], align=False)
    assert layout.row_bytes == 9 and layout.offsets["value"] == 1
    raw = np.zeros((3, 9), np.uint8)
    layout.unpack(raw)["value"][:] = 2.5
    np.testing.assert_array_equal(layout.unpack(raw, copy=True)["value"], 2.5)
    for fields in ([ ("pointer", object) ], [("text", "S4")], [("empty", "f4", (0,))]):
        with pytest.raises(ValueError):
            OutputLayout(fields)
    for raw, error in ((np.zeros((3, 9), np.int8), TypeError), (np.zeros((3, 8), np.uint8), ValueError),
                        (np.zeros((3, 18), np.uint8)[:, ::2], ValueError)):
        with pytest.raises(error):
            layout.unpack(raw)


def test_custom_process_output_schema_pickle_and_lease_lifetime(tmp_path):
    code = r'''
#include "process_fn_cabi.h"
#include <cstring>
extern "C" int32_t emit(const PyCellPackDenseRow *r) {
    uint64_t id = r->context.cell;
    std::memcpy(r->out, &id, sizeof id);
    return 0;
}
'''
    process = loader.ProcessFn.from_source(code=code, dense="emit", output=[("cell", "u8")], cache_dir=tmp_path)
    process = pickle.loads(pickle.dumps(process))
    file = cellpack.write(tmp_path / "cells.cpk", np.ones((2, 3), np.float32))
    loading = loader.Loader(loader.DataSource([(file, np.arange(3))], 3), process.schema(np.arange(3), name="id"),
                            workers=1, backend="pread", slot_rows=2)
    ring = loading.ring(slots=1)
    with loading.start([1, 0], ring=ring) as run:
        batch = next(run)
        observer = batch.observer()
        fields = process.unpack(batch.region("id"))
        copied = process.unpack(batch.region("id"), copy=True)
        batch.release()
        run.join()
    del batch
    gc.collect()
    assert observer.held
    np.testing.assert_array_equal(fields["cell"], [1, 0])
    del fields
    gc.collect()
    assert not observer.held
    with loading.start([0, 1], ring=ring) as run:
        for batch in run:
            pass
        run.join()
    np.testing.assert_array_equal(copied["cell"], [1, 0])
