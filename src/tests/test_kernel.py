"""The native process-function ABI: typed rows, partial rows and failures."""

import ctypes

import numpy as np
import pytest

from pycellpack import cellpack, loader

_CAPSULE_NAME = b"pycellpack.process.v1"


class Table(ctypes.Structure):
    _fields_ = [
        ("abi", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("csr", ctypes.c_void_p),
        ("dense", ctypes.c_void_p),
        ("context", ctypes.c_void_p),
    ]


PROVIDER = r"""
#include "process_fn_cabi.h"
#include <stddef.h>
#include <string.h>
/* Writes (sum of values, position or cell, kind) as int64; mode 1 keys by
 * position, mode 3 fails at position 2. Values are float64 (dtype 9). */
static int32_t finish(double sum, const PyCellPackContext *c, uint8_t *out, int64_t kind) {
    uintptr_t mode=(uintptr_t)c->user_data;
    if(c->dtype!=9) return -9;
    if(mode==3 && c->position==2) return -37;
    int64_t data[3]={(int64_t)sum,(int64_t)(mode==1?c->position:c->cell),kind};
    memcpy(out,data,sizeof data);
    return 0;
}
int32_t csr_fn(const PyCellPackCsrRow *c) {
    const double *values=c->values;
    int64_t kind=0;
    if(c->nnz && !values) {
        uintptr_t start=((uintptr_t)c->scratch+7)&~(uintptr_t)7;
        uint64_t skip=start-(uintptr_t)c->scratch;
        if(skip>c->scratch_bytes || c->nnz>(c->scratch_bytes-skip)/12 || !c->resolve) return -12;
        double *v=(double*)start;
        uint32_t *columns=(uint32_t*)(v+c->nnz);
        if(c->resolve(c->row,NULL,c->nnz,columns,v)) return -13;
        values=v;
        kind=2;
    }
    double sum=0;
    for(uint64_t i=0;i<c->nnz;++i) sum+=values[i];
    return finish(sum,&c->context,c->out,kind);
}
int32_t dense_fn(const PyCellPackDenseRow *c) {
    const double *values=c->values;
    double sum=0;
    for(uint64_t i=0;i<c->out_dim;++i) sum+=values[i];
    return finish(sum,&c->context,c->out,1);
}
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """The provider compiled with pycellpack's own build helper (path)."""
    return loader.compile_library(code=PROVIDER, language="c", name="provider",
                                  cache_dir=tmp_path_factory.mktemp("process-provider"), optimize="-O2")


def process(library, *, mode=0, partial_rows=False):
    # The mode travels as the user_data pointer value itself.
    return loader.ProcessFn.from_library(library, csr="csr_fn", dense="dense_fn", user_data=mode, key=mode,
                                         partial_rows=partial_rows)


def capsule_process(library, *, abi=5):
    """The same functions bound through a hand-made vtable capsule."""
    lib = ctypes.CDLL(str(library))
    table = Table(
        abi,
        ctypes.sizeof(Table),
        ctypes.cast(lib.csr_fn, ctypes.c_void_p).value,
        ctypes.cast(lib.dense_fn, ctypes.c_void_p).value,
        None,
    )
    make = ctypes.pythonapi.PyCapsule_New
    make.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    make.restype = ctypes.py_object
    capsule = make(ctypes.addressof(table), _CAPSULE_NAME, None)
    return loader.ProcessFn(capsule, identity=17000, owner=(lib, table))


def prepare(tmp_path, library, mode=0, layout="auto", partial_rows=False):
    x = np.arange(1, 121, dtype=np.uint16).reshape(12, 10)
    x[::2, 1:] = 0
    file = cellpack.write(tmp_path / "m.cpk", x, chunk_rows=4, options=cellpack.scans.EncodeOptions(layout=layout))
    source = loader.DataSource([(file, np.arange(10))], 10)
    schema = loader.OutputSchema(
        np.arange(10), np.float64, id=100, name="model", process=process(library, mode=mode, partial_rows=partial_rows), row_bytes=24, scratch_bytes=256
    )
    loading = loader.Loader(source, {"model": schema}, config=loader.LoaderConfig(workers=2, backend="pread", inflight=2, slot_rows=2))
    return x, loading


@pytest.mark.parametrize("layout,partial", [("sparse", False), ("sparse", True), ("dense", False)])
def test_typed_rows_reach_the_function_their_record_needs(tmp_path, library, layout, partial):
    x, loading = prepare(tmp_path, library, layout=layout, partial_rows=partial)
    accesses = np.array([8, 1, 8, 0, 9, 4, 2], np.uint32)
    actual = []
    with loading.start(accesses, slots=2) as run:
        for batch in run:
            with batch:
                data = batch["model"]
                assert data.dtype == np.uint8
                assert data.shape == (batch.rows, 24)
                actual.append(data.view(np.int64).copy())
                del data
        stats = run.join()
        assert stats.outputs["model"].process_calls == len(accesses)
        assert stats.written_bytes == len(accesses) * 24
    actual = np.concatenate(actual)
    np.testing.assert_array_equal(actual[:, 0], x[accesses].sum(axis=1).astype(np.int64))
    np.testing.assert_array_equal(actual[:, 1], accesses)
    assert set(actual[:, 2]) == {1 if layout == "dense" else 2 if partial else 0}


def test_position_seeds_and_function_failures(tmp_path, library):
    _, loading = prepare(tmp_path, library, mode=1)
    accesses = np.array([5, 5, 5, 3], np.uint32)
    with loading.start(accesses, slots=2) as run:
        positions = np.concatenate([b["model"].view(np.int64)[:, 1].copy() for b in run])
        run.join()
    np.testing.assert_array_equal(positions, np.arange(4))
    _, failing = prepare(tmp_path, library, mode=3)
    with pytest.raises(RuntimeError, match="-37"), failing.start(np.arange(12, dtype=np.uint32), slots=2) as run:
        for batch in run:
            batch.release()
    with pytest.raises(ValueError, match="ABI"):
        capsule_process(library, abi=99)


def test_process_cost_bands_and_profile_round_trip(tmp_path, library):
    from dataclasses import replace

    _, base = prepare(tmp_path, library)
    loading = loader.Loader(base.source, base.schemas, config=replace(base.config, profile=["process"]))
    cells = np.arange(12, dtype=np.uint32)
    options = loading.plan_options(4096 * 20)
    baseline = loading.compile(cells, options=options)
    output = options.cost.outputs["model"]
    curve = (loader.ProcessCost(128, 100e-6, 1e-6), loader.ProcessCost(2**32 - 1, 200e-6, 2e-6))
    cost = replace(options.cost, outputs={"model": replace(output, curve=curve)})
    planned = loading.compile(cells, options=replace(options, cost=cost))
    assert planned.summary.worker_seconds > baseline.summary.worker_seconds
    planned.write(tmp_path / "curve.plan")
    restored = loader.Plan.open(tmp_path / "curve.plan")
    assert loading.replay(restored, cells) == planned.summary
    with loading.start(cells, plan=restored) as run:
        for batch in run:
            batch.release()
        stats = run.join()
    model = list(loading.schemas).index("model")
    bands = stats.profile.process_bands(model)
    assert [b["max_elements"] for b in bands] == loader.PROCESS_BOUNDS
    assert sum(b["full_calls"] + b["partial_calls"] for b in bands) == len(cells) == stats.outputs["model"].process_calls
    bad = replace(cost, outputs={"model": replace(output, curve=curve[:1])})
    with pytest.raises(ValueError, match="bands"):
        loading.compile(cells, options=replace(options, cost=bad))
    # A process function's schema declares its cell bytes.
    with pytest.raises(ValueError, match="row_bytes"):
        loader.OutputSchema(np.arange(10), np.float64, process=process(library))
