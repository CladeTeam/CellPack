"""Model presets under process_fn/include/template: each compiles as a template
instance, its CSR and dense entry points agree whatever the stored column
order, its outputs keep the documented layout, and a preset runs as a loader
schema."""

import ctypes
import pickle
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.loader.process_fn import presets

GENES = 64
I64P, U32P, F32P = ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float)

# Calls a preset's entry points with rows built here, so the test needs no
# copy of the row structs.
DRIVER = r"""
#include "{header}"
struct Config : pycellpack::preset::{defaults} {{ {params} }};
using Model = pycellpack::preset::{model}<Config>;
PYCELLPACK_EXPORT_PROCESS_FN(preset, Model)
static PyCellPackContext context(uint64_t position, uint64_t cell, const void *tables) {{
    PyCellPackContext c{{}};
    c.abi_version = PYCELLPACK_PROCESS_ABI_VERSION; c.struct_size = sizeof c;
    c.position = position; c.cell = cell; c.dtype = 8; c.user_data = tables;
    return c;
}}
extern "C" uint64_t out_bytes() {{ return Model::out_bytes; }}
extern "C" int32_t call_csr(uint64_t position, uint64_t cell, uint64_t out_dim, uint64_t nnz,
                            const uint32_t *columns, const float *values, const void *tables, uint8_t *out) {{
    PyCellPackCsrRow r{{}};
    r.context = context(position, cell, tables); r.out_dim = out_dim; r.nnz = nnz;
    r.columns = columns; r.values = values; r.out = out;
    return preset_csr(&r);
}}
extern "C" int32_t call_dense(uint64_t position, uint64_t cell, uint64_t out_dim, const float *values,
                              const void *tables, uint8_t *out) {{
    PyCellPackDenseRow r{{}};
    r.context = context(position, cell, tables); r.out_dim = out_dim; r.values = values; r.out = out;
    return preset_dense(&r);
}}
"""


class ScGptTables(ctypes.Structure):
    _fields_ = [("tokens", I64P), ("n_genes", ctypes.c_uint64), ("pad_token", ctypes.c_int64),
                ("cls_token", ctypes.c_int64), ("seed", ctypes.c_uint64)]


class UceTables(ctypes.Structure):
    _fields_ = [("tokens", I64P), ("chroms", I64P), ("starts", I64P), ("valid", U32P), ("local", U32P),
                ("n_genes", ctypes.c_uint64), ("n_valid", ctypes.c_uint64), ("pad_token", ctypes.c_int64),
                ("cls_token", ctypes.c_int64), ("chrom_offset", ctypes.c_int64),
                ("chrom_close_token", ctypes.c_int64), ("seed", ctypes.c_uint64)]


class StateSeTables(ctypes.Structure):
    _fields_ = [("tokens", I64P), ("valid", U32P), ("local", U32P), ("reverse", I64P),
                ("n_genes", ctypes.c_uint64), ("n_valid", ctypes.c_uint64), ("global_size", ctypes.c_uint64),
                ("seed", ctypes.c_uint64), ("stream", ctypes.c_uint32)]


class GeneformerTables(ctypes.Structure):
    _fields_ = [("tokens", I64P), ("medians", F32P), ("n_genes", ctypes.c_uint64), ("pad_token", ctypes.c_int64),
                ("cls_token", ctypes.c_int64), ("eos_token", ctypes.c_int64)]


def _ptr(array, kind):
    return array.ctypes.data_as(kind)


def _tokens():
    tokens = np.arange(100, 100 + GENES, dtype=np.int64)
    tokens[::7] = -1  # genes without a token
    return tokens


def _panel(tokens):
    valid = np.flatnonzero(tokens >= 0).astype(np.uint32)
    local = np.full(GENES, 0xFFFFFFFF, np.uint32)
    local[valid] = np.arange(len(valid), dtype=np.uint32)
    return valid, local


@dataclass
class Preset:
    name: str
    header: str
    defaults: str
    model: str
    params: str
    words: dict  # field -> (dtype, count) of the output, in order

    def tables(self):
        tokens = _tokens()
        keep = [tokens]
        if self.name == "scgpt":
            t = ScGptTables(_ptr(tokens, I64P), GENES, 0, 1, 7)
        elif self.name == "uce":
            valid, local = _panel(tokens)
            chroms = (np.arange(GENES) % 5).astype(np.int64)
            starts = np.arange(GENES, dtype=np.int64)[::-1].copy()
            keep += [valid, local, chroms, starts]
            t = UceTables(_ptr(tokens, I64P), _ptr(chroms, I64P), _ptr(starts, I64P), _ptr(valid, U32P),
                          _ptr(local, U32P), GENES, len(valid), 0, 3, 1000, 2, 7)
        elif self.name == "state_se":
            valid, local = _panel(tokens)
            reverse = np.full(200, -1, np.int64)
            reverse[tokens[valid]] = valid
            keep += [valid, local, reverse]
            t = StateSeTables(_ptr(tokens, I64P), _ptr(valid, U32P), _ptr(local, U32P), _ptr(reverse, I64P),
                              GENES, len(valid), 200, 7, 1)
        else:
            medians = np.linspace(0.5, 3.0, GENES, dtype=np.float32)
            keep += [medians]
            t = GeneformerTables(_ptr(tokens, I64P), _ptr(medians, F32P), GENES, 0, 2, 3)
        return t, keep

    def decode(self, raw: bytes) -> dict:
        out, at = {}, 0
        for field, (dtype, count) in self.words.items():
            dtype = np.dtype(dtype)
            at = -(-at // dtype.alignment) * dtype.alignment
            out[field] = np.frombuffer(raw, dtype, count, at)
            at += dtype.itemsize * count
        return out


PRESETS = [
    Preset("scgpt", "template/scgpt.hpp", "ScGptDefaults", "ScGpt",
           "static constexpr uint32_t length = 24; static constexpr double mask_ratio = 0.25;",
           {"genes": ("i8", 24), "values": ("f4", 24), "targets": ("f4", 24)}),
    Preset("uce", "template/uce.hpp", "UceDefaults", "Uce",
           "static constexpr uint32_t length = 64; static constexpr uint32_t sample = 32;"
           "static constexpr uint32_t positive = 6; static constexpr uint32_t negative = 6;",
           {"ids": ("i8", 64), "loss_ids": ("i8", 12), "labels": ("f4", 12), "length": ("u4", 1)}),
    Preset("state_se", "template/state_se.hpp", "StateSeDefaults", "StateSe",
           "static constexpr uint32_t length = 32; static constexpr uint32_t positive = 4;"
           "static constexpr uint32_t negative = 4; static constexpr uint32_t shared = 4;"
           "static constexpr uint32_t batch = 4;",
           {"ids": ("i8", 32), "tasks": ("i8", 12), "targets": ("f4", 12), "counts": ("f4", 32), "total": ("f4", 1)}),
    Preset("geneformer", "template/geneformer.hpp", "GeneformerDefaults", "Geneformer",
           "static constexpr uint32_t length = 16;",
           {"ids": ("i8", 16), "length": ("u4", 1)}),
]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("preset-cache")


def _library(preset, cache):
    code = DRIVER.format(header=preset.header, defaults=preset.defaults, model=preset.model, params=preset.params)
    lib = ctypes.CDLL(str(loader.compile_library(code=code, name=f"preset_{preset.name}", cache_dir=cache,
                                                 flags=["-Wall", "-Wextra", "-Werror"])))
    lib.out_bytes.restype = ctypes.c_uint64
    lib.call_csr.restype = lib.call_dense.restype = ctypes.c_int32
    return lib, code


def _rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.poisson(0.8, size=(n, GENES)).astype(np.float32)
    x[rng.random((n, GENES)) < 0.4] = 0
    x[:, 1] += 1  # every cell expresses a gene with a token
    return x


def _call(lib, preset, tables, position, cell, row, *, columns=None):
    size = lib.out_bytes()
    out = np.zeros(size, np.uint8)
    t = ctypes.byref(tables)
    if columns is None:
        status = lib.call_dense(ctypes.c_uint64(position), ctypes.c_uint64(cell), ctypes.c_uint64(GENES),
                                _ptr(row, F32P), t, _ptr(out, ctypes.POINTER(ctypes.c_uint8)))
    else:
        values = np.ascontiguousarray(row[columns])
        columns = np.ascontiguousarray(columns, dtype=np.uint32)
        status = lib.call_csr(ctypes.c_uint64(position), ctypes.c_uint64(cell), ctypes.c_uint64(GENES),
                              ctypes.c_uint64(len(columns)), _ptr(columns, U32P), _ptr(values, F32P), t,
                              _ptr(out, ctypes.POINTER(ctypes.c_uint8)))
    assert status == 0, f"{preset.name} returned {status}"
    return out.tobytes()


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_preset_entry_points_agree_and_keep_the_layout(preset, cache):
    lib, _ = _library(preset, cache)
    tables, keep = preset.tables()
    assert lib.out_bytes() >= sum(np.dtype(d).itemsize * n for d, n in preset.words.values())
    tokens = keep[0]
    rng = np.random.default_rng(1)
    for cell, row in enumerate(_rows()):
        nz = np.flatnonzero(row).astype(np.uint32)
        dense = _call(lib, preset, tables, cell, cell, row)
        assert _call(lib, preset, tables, cell, cell, row, columns=nz) == dense
        assert _call(lib, preset, tables, cell, cell, row, columns=rng.permutation(nz)) == dense
        out = preset.decode(dense)
        expressed = [g for g in nz if tokens[g] >= 0]
        if preset.name == "scgpt":
            n = min(len(expressed), 23)
            assert out["genes"][0] == 1 and np.all(out["genes"][n + 1:] == 0)
            assert set(out["genes"][1 : n + 1]) <= set(tokens[expressed])
            kept = out["targets"][1 : n + 1]
            assert np.all((kept >= 1) & (kept <= 50))
            assert np.all((out["values"][1 : n + 1] == kept) | (out["values"][1 : n + 1] == -1))
        elif preset.name == "uce":
            length = int(out["length"][0])
            assert out["ids"][0] == 3 and 1 < length <= 64 and np.all(out["ids"][length:] == 0)
            assert set(out["labels"]) <= {0.0, 1.0}
        elif preset.name == "state_se":
            np.testing.assert_allclose(out["total"][0], out["targets"].astype(np.float64).sum(), rtol=1e-6)
            assert set(out["ids"]) <= set(tokens[tokens >= 0])
        else:
            medians = np.linspace(0.5, 3.0, GENES, dtype=np.float32)
            order = sorted(expressed, key=lambda g: (-(row[g] / medians[g]), g))[:14]
            want = [2, *tokens[order], 3]
            length = int(out["length"][0])
            assert length == len(want)
            assert list(out["ids"][:length]) == want and np.all(out["ids"][length:] == 0)


def test_preset_runs_as_a_loader_schema(tmp_path, cache):
    preset = PRESETS[0]
    lib, code = _library(preset, cache)
    tables, keep = preset.tables()
    x = _rows(24, seed=3)
    file = cellpack.write(tmp_path / "cells.cpk", x, format="csr", chunk_rows=8)
    process = loader.ProcessFn.from_source(code=code, name="preset_scgpt", cache_dir=cache,
                                           flags=["-Wall", "-Wextra", "-Werror"], csr="preset_csr",
                                           dense="preset_dense", user_data=tables, key=(keep, tables.seed),
                                           owner=keep)
    schema = loader.OutputSchema(np.arange(GENES), np.float32, name="scgpt", process=process,
                                 row_bytes=lib.out_bytes())
    loading = loader.Loader(loader.DataSource([(file, np.arange(GENES))], GENES), {"scgpt": schema},
                            config=loader.LoaderConfig(workers=2, backend="pread", slot_rows=8))
    cells = np.random.default_rng(4).permutation(len(x)).astype(np.uint32)
    with loading.start(cells) as run:
        got = [bytes(b["scgpt"][i]) for b in run for i in range(b.rows)]
        run.join()
    want = [_call(lib, preset, tables, position, int(cell), x[cell]) for position, cell in enumerate(cells)]
    assert got == want


def _python_preset(preset, cache):
    common = {"tokens": _tokens(), "cache_dir": cache, "flags": ["-Wall", "-Wextra", "-Werror"]}
    if preset.name == "scgpt":
        return presets.scgpt(**common, pad_token=0, cls_token=1, seed=7, length=24, mask_ratio=0.25)
    if preset.name == "geneformer":
        return presets.geneformer(**common, medians=np.linspace(0.5, 3, GENES, dtype=np.float32),
                                   pad_token=0, cls_token=2, eos_token=3, length=16)
    if preset.name == "uce":
        return presets.uce(**common, chroms=np.arange(GENES) % 5, starts=np.arange(GENES)[::-1],
                            pad_token=0, cls_token=3, chrom_offset=1000, chrom_close_token=2, seed=7,
                            length=64, sample=32, positive=6, negative=6)
    return presets.state_se(**common, global_size=200, seed=7, stream=1, length=32,
                             positive=4, negative=4, shared=4, batch=4)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
@pytest.mark.parametrize("format", ["csr", "dense"])
def test_python_presets_match_independent_native_bindings(tmp_path, cache, preset, format):
    lib, _ = _library(preset, cache)
    tables, _keep = preset.tables()
    original = _python_preset(preset, cache)
    process = pickle.loads(pickle.dumps(original))
    assert process.identity == original.identity
    assert process.row_bytes == lib.out_bytes()
    assert process.out_dim == GENES and process.input_dtype == np.dtype(np.float32)
    x = _rows(17, seed=3)
    file = cellpack.write(tmp_path / "cells.cpk", x, format=format, chunk_rows=8)
    schema = process.schema(np.arange(GENES), name=preset.name)
    source = loader.DataSource([(file, np.arange(GENES))], GENES)
    loading = loader.Loader(source, schema, workers=2, backend="pread", slot_rows=4)
    cells = np.random.default_rng(4).permutation(len(x)).astype(np.uint32)
    got = []
    with loading.start(cells, slots=2) as run:
        for batch in run:
            got.append(process.unpack(batch.region(preset.name), copy=True))
        run.join()
    for name in process.output.names:
        actual = np.concatenate([item[name] for item in got])
        want = []
        for position, cell in enumerate(cells):
            value = preset.decode(_call(lib, preset, tables, position, int(cell), x[cell]))[name]
            want.append(value[0] if name in ("length", "total") else value)
        np.testing.assert_array_equal(actual, np.asarray(want))


def test_preset_snapshots_tables_and_rebuilds_in_a_fresh_process(tmp_path, cache):
    tokens = _tokens()
    process = presets.scgpt(tokens, pad_token=0, cls_token=1, seed=7, length=24, cache_dir=cache)
    tokens[:] = -1
    rebuilt = presets.scgpt(_tokens(), pad_token=0, cls_token=1, seed=7, length=24, cache_dir=cache)
    assert process.identity == rebuilt.identity
    changed = presets.scgpt(_tokens(), pad_token=0, cls_token=1, seed=8, length=24, cache_dir=cache)
    assert changed.identity != process.identity
    file = cellpack.write(tmp_path / "cells.cpk", _rows(3), format="csr")
    loading = loader.Loader(loader.DataSource([(file, np.arange(GENES))], GENES),
                            process.schema(np.arange(GENES), name="scgpt"), workers=1, backend="pread")
    with loading.start([0, 1, 2]) as run:
        with next(run) as batch:
            want = process.unpack(batch.region("scgpt"), copy=True)
        run.join()
    script = """
import pickle, sys
import numpy as np
from pycellpack import DataSource, Loader
process, path = pickle.load(sys.stdin.buffer)
loading = Loader(DataSource([(path, np.arange(64))], 64),
                 process.schema(np.arange(64), name="scgpt"), workers=1, backend="pread")
with loading.start([0, 1, 2]) as run:
    with next(run) as batch:
        result = process.unpack(batch.region("scgpt"), copy=True)
    run.join()
sys.stdout.buffer.write(pickle.dumps((process.identity, result)))
"""
    child = subprocess.run([sys.executable, "-c", script], input=pickle.dumps((process, str(file.file.path))),
                           capture_output=True, timeout=60, check=True)
    identity, got = pickle.loads(child.stdout)
    assert identity == process.identity
    for name in want:
        np.testing.assert_array_equal(got[name], want[name])


def test_preset_rejects_incompatible_schema_and_tables(cache):
    process = presets.scgpt(_tokens(), pad_token=0, cls_token=1, length=24, cache_dir=cache)
    with pytest.raises(ValueError, match="requires input dtype"):
        process.schema(np.arange(GENES), np.float64)
    with pytest.raises(ValueError, match="requires 64 out_ids"):
        process.schema(np.arange(3))
    with pytest.raises(ValueError, match="row_bytes must match"):
        loader.OutputSchema(np.arange(GENES), process=process, row_bytes=1)
    with pytest.raises(ValueError, match="medians.*64 entries"):
        presets.geneformer(_tokens(), [1], pad_token=0, cls_token=1, eos_token=2)
    with pytest.raises(ValueError, match="below global_size"):
        presets.state_se(_tokens(), global_size=10)
    with pytest.raises(ValueError, match="mask_ratio"):
        presets.scgpt(_tokens(), pad_token=0, cls_token=1, mask_ratio=float("nan"))
