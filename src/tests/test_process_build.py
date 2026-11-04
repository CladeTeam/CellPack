"""Building and binding process functions: compile_library, from_source,
from_library, and their error messages."""

import ctypes

import numpy as np
import pytest

from pycellpack import cellpack, loader


def test_compile_flags_iterable_is_used_for_compilation_and_cache(tmp_path):
    code = "int value(void) { return EXPECTED_VALUE; }"
    flags = ["-DEXPECTED_VALUE=37"]
    path = loader.compile_library(code=code, language="c", flags=iter(flags), cache_dir=tmp_path)
    assert ctypes.CDLL(str(path)).value() == 37
    assert loader.compile_library(code=code, language="c", flags=flags, cache_dir=tmp_path) == path

# Writes the row's value sum plus a per-schema offset read from user_data.
SOURCE = r"""
#include "process_fn_cabi.h"
#include <string.h>
struct Params { double offset; };
static int32_t finish(double sum, const PyCellPackContext *c, uint8_t *out) {
    const struct Params *p = (const struct Params *)c->user_data;
    double value = sum + (p ? p->offset : 0);
    memcpy(out, &value, sizeof value);
    return 0;
}
extern "C" int32_t sum_csr(const PyCellPackCsrRow *r) {
    const float *v = (const float *)r->values;
    double s = 0;
    for (uint64_t i = 0; i < r->nnz; ++i) s += v[i];
    return finish(s, &r->context, r->out);
}
extern "C" int32_t sum_dense(const PyCellPackDenseRow *r) {
    const float *v = (const float *)r->values;
    double s = 0;
    for (uint64_t i = 0; i < r->out_dim; ++i) s += v[i];
    return finish(s, &r->context, r->out);
}
"""


class Params(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_double)]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("process-cache")


def run(tmp_path, process):
    x = np.arange(1, 41, dtype=np.uint16).reshape(8, 5)
    x[::2, 1:] = 0
    file = cellpack.write(tmp_path / "m.cpk", x, chunk_rows=4)
    schema = loader.OutputSchema(np.arange(5), np.float32, name="sum", process=process, row_bytes=8)
    loading = loader.Loader(loader.DataSource([(file, np.arange(5))], 5), {"sum": schema},
                            config=loader.LoaderConfig(workers=2, backend="pread", slot_rows=4))
    cells = np.arange(8, dtype=np.uint32)
    with loading.start(cells) as r:
        out = np.concatenate([b["sum"].view(np.float64).ravel().copy() for b in r])
        r.join()
    return x.sum(axis=1).astype(np.float64), out


def test_inline_source_is_compiled_once_and_bound(tmp_path, cache):
    path = loader.compile_library(code=SOURCE, name="sum", cache_dir=cache)
    assert path.is_file() and path.name.startswith("sum-")
    assert loader.compile_library(code=SOURCE, name="sum", cache_dir=cache) == path
    assert loader.compile_library(code=SOURCE, name="sum", cache_dir=cache, defines={"X": 1}) != path
    params = Params(2.5)
    process = loader.ProcessFn.from_library(path, csr="sum_csr", dense="sum_dense", user_data=params,
                                            key=params.offset)
    assert process.row_format == "auto"
    want, got = run(tmp_path, process)
    np.testing.assert_array_equal(got, want + 2.5)


def test_from_source_files_and_identity(tmp_path, cache):
    src = tmp_path / "sum.cpp"
    src.write_text(SOURCE)
    a = loader.ProcessFn.from_source(src, csr="sum_csr", cache_dir=cache)
    b = loader.ProcessFn.from_source(src, csr="sum_csr", cache_dir=cache)
    assert a.row_format == "csr" and a.identity == b.identity
    c = loader.ProcessFn.from_source(src, csr="sum_csr", user_data=Params(1), key=1.0, cache_dir=cache)
    d = loader.ProcessFn.from_source(src, csr="sum_csr", user_data=Params(2), key=2.0, cache_dir=cache)
    assert len({a.identity, c.identity, d.identity}) == 3
    want, got = run(tmp_path, c)
    np.testing.assert_array_equal(got, want + 1)
    assert loader.ProcessFn.from_source(src, dense="sum_dense", cache_dir=cache).row_format == "dense"
    e = loader.ProcessFn.from_source(src, csr="sum_csr", identity=7, cache_dir=cache)
    assert e.identity == 7


def test_compile_errors_carry_the_command_and_messages(tmp_path, cache):
    with pytest.raises(loader.ProcessBuildError) as info:
        loader.compile_library(code="int broken( {", name="broken", cache_dir=cache)
    text = str(info.value)
    assert "compiling broken failed" in text and "\n$ " in text and "error" in text.lower()
    with pytest.raises(loader.CompilerNotFoundError, match="not a working GCC or Clang"):
        loader.compile_library(code=SOURCE, compiler="no-such-compiler-xyz", cache_dir=cache)
    (tmp_path / "a.c").write_text("int a(void) { return 1; }")
    (tmp_path / "b.cpp").write_text("int b() { return 2; }")
    with pytest.raises(ValueError, match="cannot be built together"):
        loader.compile_library([tmp_path / "a.c", tmp_path / "b.cpp"], cache_dir=cache)
    with pytest.raises(FileNotFoundError, match="process source not found"):
        loader.compile_library("missing.cpp", cache_dir=cache)
    with pytest.raises(ValueError, match="sources or code"):
        loader.compile_library(cache_dir=cache)
    with pytest.raises(ValueError, match="language must be"):
        loader.compile_library(code=SOURCE, language="rust", cache_dir=cache)


def test_binding_errors_explain_the_fix(cache):
    path = loader.compile_library(code=SOURCE, name="sum", cache_dir=cache)
    with pytest.raises(loader.ProcessLibraryError, match=r'extern "C"'):
        loader.ProcessFn.from_library(path, csr="sum_csr_typo")
    with pytest.raises(ValueError, match="csr= and/or dense="):
        loader.ProcessFn.from_library(path)
    with pytest.raises(ValueError, match="needs dense="):
        loader.ProcessFn.from_library(path, csr="sum_csr", row_format="auto")
    with pytest.raises(ValueError, match="needs a CSR function"):
        loader.ProcessFn.from_library(path, dense="sum_dense", partial_rows=True)
    with pytest.raises(ValueError, match="without key= or identity="):
        loader.ProcessFn.from_library(path, csr="sum_csr", user_data=Params(1))
    with pytest.raises(TypeError, match="user_data must be"):
        loader.ProcessFn.from_library(path, csr="sum_csr", user_data="text", identity=1)
    with pytest.raises(TypeError, match="key values must be"):
        loader.ProcessFn.from_library(path, csr="sum_csr", key=object())
    with pytest.raises(ValueError, match="not both"):
        loader.ProcessFn.from_library(path, csr="sum_csr", key=1, identity=1)
    with pytest.raises(FileNotFoundError, match="process library not found"):
        loader.ProcessFn.from_library(path.with_name("absent.so"), csr="sum_csr")
    with pytest.raises(ValueError, match=r"identity must be an integer"):
        loader.ProcessFn.from_library(path, csr="sum_csr", identity=-1)
    lib = ctypes.CDLL(str(path))
    bound = loader.ProcessFn.from_library(lib, csr=lib.sum_csr, identity=3)
    assert bound.identity == 3 and bound.row_format == "csr"
    with pytest.raises(ValueError, match="explicit identity"):
        loader.ProcessFn.from_library(path, csr=ctypes.cast(lib.sum_csr, ctypes.c_void_p).value)


LAYOUT = r"""
#include "process_fn_cabi.h"
#include <stddef.h>
#include <stdint.h>
#define F(T, f) (uint64_t)offsetof(T, f)
static const uint64_t layout[] = {
    sizeof(PyCellPackContext), F(PyCellPackContext, position), F(PyCellPackContext, cell),
    F(PyCellPackContext, schema), F(PyCellPackContext, user_data),
    sizeof(PyCellPackCsrRow), F(PyCellPackCsrRow, out_dim), F(PyCellPackCsrRow, nnz), F(PyCellPackCsrRow, columns),
    F(PyCellPackCsrRow, values), F(PyCellPackCsrRow, resolve), F(PyCellPackCsrRow, out),
    F(PyCellPackCsrRow, scratch_bytes),
    sizeof(PyCellPackDenseRow), F(PyCellPackDenseRow, out_dim), F(PyCellPackDenseRow, values),
    F(PyCellPackDenseRow, out), F(PyCellPackDenseRow, scratch_bytes),
};
const uint64_t *row_layout(void) { return layout; }
"""


def test_numba_row_types_match_the_header(cache):
    pytest.importorskip("numba")
    pn = loader.process_fn.numba_bindings()
    lib = ctypes.CDLL(str(loader.compile_library(code=LAYOUT, language="c", name="layout", cache_dir=cache)))
    lib.row_layout.restype = ctypes.POINTER(ctypes.c_uint64)
    got = lib.row_layout()[:18]
    c, r, d = pn.CONTEXT_DTYPE, pn.CSR_ROW_DTYPE, pn.DENSE_ROW_DTYPE
    want = [c.itemsize, *(c.fields[f][1] for f in ("position", "cell", "schema", "user_data")),
            r.itemsize, *(r.fields[f][1] for f in ("out_dim", "nnz", "columns", "values", "resolve", "out",
                                                     "scratch_bytes")),
            d.itemsize, *(d.fields[f][1] for f in ("out_dim", "values", "out", "scratch_bytes"))]
    assert got == want


def test_numba_cfuncs_and_python_functions(tmp_path):
    numba = pytest.importorskip("numba")
    pn = loader.process_fn.numba_bindings()

    @numba.cfunc(pn.CSR_SIGNATURE)
    def csr(call):
        row = pn.row(call)
        values = pn.array_at(row.values, row.nnz, np.float32)
        offset = pn.array_at(row.context.user_data, 1, np.float64)[0]
        pn.array_at(row.out, 1, np.float64)[0] = values.astype(np.float64).sum() + offset
        return 0

    def dense(call):
        row = pn.row(call)
        values = pn.array_at(row.values, row.out_dim, np.float32)
        offset = pn.array_at(row.context.user_data, 1, np.float64)[0]
        pn.array_at(row.out, 1, np.float64)[0] = values.astype(np.float64).sum() + offset
        return 0

    offset = np.array([4.0])
    both = loader.ProcessFn.from_cfunc(csr=csr, dense=dense, user_data=offset, key=offset)
    assert both.row_format == "auto"
    want, got = run(tmp_path, both)
    np.testing.assert_array_equal(got, want + 4)
    only = loader.ProcessFn.from_cfunc(csr=csr, user_data=offset, key=offset)
    assert only.row_format == "csr" and only.identity != both.identity
    assert loader.ProcessFn.from_cfunc(csr=csr, user_data=offset, key=offset).identity == only.identity
    assert loader.ProcessFn.from_cfunc(csr=csr, user_data=offset, key=offset + 1).identity != only.identity

    @numba.cfunc("int32(int64)")
    def wrong(x):
        return 0

    with pytest.raises(TypeError, match="CSR_SIGNATURE"):
        loader.ProcessFn.from_cfunc(csr=wrong)

    def broken(call):
        return call[0].no_such_field

    with pytest.raises(loader.ProcessBuildError, match="numba could not compile"):
        loader.ProcessFn.from_cfunc(dense=broken)
    with pytest.raises(TypeError, match="numba @cfunc or a Python function"):
        loader.ProcessFn.from_cfunc(csr=3)
    with pytest.raises(ValueError, match="without key= or identity="):
        loader.ProcessFn.from_cfunc(csr=csr, user_data=offset)
