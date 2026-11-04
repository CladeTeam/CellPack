"""Python bindings for the native scGPT, Geneformer, UCE and STATE SE templates.

Each factory snapshots its tables, compiles a cached template instance and
returns a ProcessFn with a named output layout. Tokens and auxiliary tables
address the schema's output dims, in exactly the order of its out_ids.
Pickling stores values and rebuilds pointers in the receiving process.
"""

from __future__ import annotations

import ctypes as C
import math
import operator
from typing import Any

import numpy as np

from ._output import OutputLayout
from ._process_fn import ProcessFn

_PTR = C.c_void_p


class _ScGptTables(C.Structure):
    _fields_ = [("tokens", _PTR), ("n_genes", C.c_uint64), ("pad_token", C.c_int64),
                ("cls_token", C.c_int64), ("seed", C.c_uint64)]


class _GeneformerTables(C.Structure):
    _fields_ = [("tokens", _PTR), ("medians", _PTR), ("n_genes", C.c_uint64),
                ("pad_token", C.c_int64), ("cls_token", C.c_int64), ("eos_token", C.c_int64)]


class _UceTables(C.Structure):
    _fields_ = [("tokens", _PTR), ("chroms", _PTR), ("starts", _PTR), ("valid", _PTR), ("local", _PTR),
                ("n_genes", C.c_uint64), ("n_valid", C.c_uint64), ("pad_token", C.c_int64),
                ("cls_token", C.c_int64), ("chrom_offset", C.c_int64), ("chrom_close_token", C.c_int64),
                ("seed", C.c_uint64)]


class _StateSeTables(C.Structure):
    _fields_ = [("tokens", _PTR), ("valid", _PTR), ("local", _PTR), ("reverse", _PTR),
                ("n_genes", C.c_uint64), ("n_valid", C.c_uint64), ("global_size", C.c_uint64),
                ("seed", C.c_uint64), ("stream", C.c_uint32)]


def _integer(value, name, *, minimum=0, maximum=(1 << 32) - 1):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _i64(value, name):
    return _integer(value, name, minimum=-(1 << 63), maximum=(1 << 63) - 1)


def _seed(value):
    return _integer(value, "seed", maximum=(1 << 64) - 1)


def _ratio(value, name, *, closed=True):
    value = float(value)
    if not math.isfinite(value) or not (0 <= value <= 1 if closed else 0 <= value < 1):
        raise ValueError(f"{name} must be finite and in [0, 1{']' if closed else ')'}")
    return value


def _vector(value, name, *, dtype: Any = np.int64, size=None):
    array = np.asarray(value)
    if array.ndim != 1 or (size is not None and len(array) != size):
        raise ValueError(f"{name} must be one-dimensional" + (f" with {size} entries" if size is not None else ""))
    if np.dtype(dtype).kind == "i":
        if array.dtype.kind not in "iu":
            raise TypeError(f"{name} must contain integers")
        if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
            raise ValueError(f"{name} values must fit int64")
    elif array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    array = np.array(array, dtype=dtype, copy=True, order="C")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


def _tokens(value):
    tokens = _vector(value, "tokens")
    if not 0 < len(tokens) < (1 << 32):
        raise ValueError("tokens must contain between 1 and 2**32-1 entries")
    if not np.any(tokens >= 0):
        raise ValueError("tokens must include at least one nonnegative token")
    return tokens


def _panel(tokens):
    valid = np.flatnonzero(tokens >= 0).astype(np.uint32)
    local = np.full(len(tokens), 0xFFFFFFFF, np.uint32)
    local[valid] = np.arange(len(valid), dtype=np.uint32)
    return valid, local


def _bind(name, model, params, tables, arrays, fields, kwargs, row_format, build):
    output = OutputLayout(fields)
    declarations = []
    for key, value in params.items():
        kind = "bool" if isinstance(value, bool) else "double" if isinstance(value, float) else "uint32_t"
        literal = str(value).lower() if isinstance(value, bool) else repr(value)
        declarations.append(f"static constexpr {kind} {key} = {literal};")
    checks = [f'static_assert(sizeof(Model::Output) == {output.row_bytes}, "output record size");',
              f'static_assert(sizeof(Model::Tables) == {C.sizeof(tables)}, "table size");']
    for field in output.names:
        checks.append(f'static_assert(offsetof(Model::Output, {field}) == {output.offsets[field]}, "{field} offset");')
    for field, _ in tables._fields_:
        checks.append(f'static_assert(offsetof(Model::Tables, {field}) == {getattr(type(tables), field).offset}, "{field} table offset");')
    code = '\n'.join([
        '#include <cstddef>', f'#include "template/{name}.hpp"',
        f'struct Config : pycellpack::preset::{model}Defaults {{ {" ".join(declarations)} }};',
        f'using Model = pycellpack::preset::{model}<Config>;', *checks,
        'PYCELLPACK_EXPORT_PROCESS_FN(preset, Model)',
    ])
    fn = ProcessFn.from_source(code=code, csr="preset_csr", dense="preset_dense", user_data=tables,
                               key=kwargs, owner=arrays, output=output, row_format=row_format, **build)
    fn._input_dtype = np.dtype(np.float32)
    fn._out_dim = len(arrays[0])
    fn._template = name
    fn._recipe = (f"preset:{name}", dict(kwargs, row_format=row_format, **build))
    return fn


def scgpt(tokens, *, pad_token, cls_token, length=1024, n_bins=51, mask_ratio=0.4,
          pad_value=-2.0, mask_value=-1.0, cls_value=-2.0, seed=0, row_format="auto", **build) -> ProcessFn:
    """Quantile binning, sampling and masking using ``template/scgpt.hpp``.

    Output fields: genes int64[length], values and targets float32[length].
    Random draws depend on seed and global cell id. Negative tokens exclude
    genes. Special token ids and value sentinels are supplied explicitly.
    """
    tokens = _tokens(tokens)
    length = _integer(length, "length", minimum=2)
    n_bins = _integer(n_bins, "n_bins", minimum=3)
    mask_ratio = _ratio(mask_ratio, "mask_ratio")
    pad_token, cls_token, seed = _i64(pad_token, "pad_token"), _i64(cls_token, "cls_token"), _seed(seed)
    values = [float(v) for v in (pad_value, mask_value, cls_value)]
    if not all(math.isfinite(v) and abs(v) <= np.finfo(np.float32).max for v in values):
        raise ValueError("pad_value, mask_value and cls_value must be finite float32 values")
    pad_value, mask_value, cls_value = values
    params = {"length": length, "n_bins": n_bins, "mask_ratio": mask_ratio, "pad_value": pad_value,
                  "mask_value": mask_value, "cls_value": cls_value}
    kwargs = dict(tokens=tokens, pad_token=pad_token, cls_token=cls_token, seed=seed, **params)
    tables = _ScGptTables(tokens.ctypes.data, len(tokens), pad_token, cls_token, seed)
    return _bind("scgpt", "ScGpt", params, tables, [tokens],
                 [("genes", "i8", (length,)), ("values", "f4", (length,)), ("targets", "f4", (length,))],
                 kwargs, row_format, build)


def geneformer(tokens, medians, *, pad_token, cls_token, eos_token, length=4096,
               special_tokens=True, row_format="auto", **build) -> ProcessFn:
    """Median-normalized ranking using ``template/geneformer.hpp``.

    Nonpositive medians and negative tokens exclude genes. Output fields:
    ids int64[length] and length uint32 (the used token count). Set
    special_tokens=False for sentences without CLS/EOS (V1).
    """
    tokens = _tokens(tokens)
    medians = _vector(medians, "medians", dtype=np.float32, size=len(tokens))
    if not isinstance(special_tokens, (bool, np.bool_)):
        raise TypeError("special_tokens must be a bool")
    special_tokens = bool(special_tokens)
    length = _integer(length, "length", minimum=3 if special_tokens else 1)
    pad_token, cls_token, eos_token = (_i64(v, n) for v, n in
                                      ((pad_token, "pad_token"), (cls_token, "cls_token"), (eos_token, "eos_token")))
    params = {"length": length, "special_tokens": special_tokens}
    kwargs = dict(tokens=tokens, medians=medians, pad_token=pad_token, cls_token=cls_token,
                  eos_token=eos_token, **params)
    tables = _GeneformerTables(tokens.ctypes.data, medians.ctypes.data, len(tokens), pad_token, cls_token, eos_token)
    return _bind("geneformer", "Geneformer", params, tables, [tokens, medians],
                 [("ids", "i8", (length,)), ("length", "u4")], kwargs, row_format, build)


def uce(tokens, chroms, starts, *, pad_token, cls_token, chrom_offset, chrom_close_token,
        length=1536, sample=1024, positive=512, negative=512, mask_ratio=0.2, seed=0,
        row_format="auto", **build) -> ProcessFn:
    """Chromosome-grouped sampling using ``template/uce.hpp``.

    chroms and starts address every output dim. Output fields: ids
    int64[length], loss_ids int64[positive+negative], labels
    float32[positive+negative] and length uint32. The template reports an
    error for cells with fewer than two eligible expressed genes, all genes
    masked, or a sentence exceeding length.
    """
    tokens = _tokens(tokens)
    chroms, starts = (_vector(v, n, size=len(tokens)) for v, n in ((chroms, "chroms"), (starts, "starts")))
    length, sample = _integer(length, "length", minimum=2), _integer(sample, "sample", minimum=1)
    positive, negative = _integer(positive, "positive"), _integer(negative, "negative")
    q = _integer(positive + negative, "positive + negative", minimum=1)
    mask_ratio, seed = _ratio(mask_ratio, "mask_ratio", closed=False), _seed(seed)
    pad_token, cls_token, chrom_offset, chrom_close_token = (
        _i64(v, n) for v, n in ((pad_token, "pad_token"), (cls_token, "cls_token"),
                                (chrom_offset, "chrom_offset"), (chrom_close_token, "chrom_close_token")))
    valid, local = _panel(tokens)
    if len(valid) < 2:
        raise ValueError("UCE needs at least two nonnegative tokens")
    eligible_chroms = chroms[valid]
    if any(not -(1 << 63) <= int(c) + chrom_offset < (1 << 63) for c in np.unique(eligible_chroms)):
        raise ValueError("chrom_offset + chromosome ids must fit int64")
    params = {"length": length, "sample": sample, "positive": positive, "negative": negative, "mask_ratio": mask_ratio}
    kwargs = dict(tokens=tokens, chroms=chroms, starts=starts, pad_token=pad_token, cls_token=cls_token,
                  chrom_offset=chrom_offset, chrom_close_token=chrom_close_token, seed=seed, **params)
    tables = _UceTables(tokens.ctypes.data, chroms.ctypes.data, starts.ctypes.data, valid.ctypes.data,
                        local.ctypes.data, len(tokens), len(valid), pad_token, cls_token, chrom_offset,
                        chrom_close_token, seed)
    return _bind("uce", "Uce", params, tables, [tokens, chroms, starts, valid, local],
                 [("ids", "i8", (length,)), ("loss_ids", "i8", (q,)), ("labels", "f4", (q,)), ("length", "u4")],
                 kwargs, row_format, build)


def state_se(tokens, *, global_size, length=2048, positive=512, negative=512, shared=512,
             batch=128, raw_counts=0, cls_index=3, seed=0, stream=0, row_format="auto", **build) -> ProcessFn:
    """Cell sentences and expression queries using ``template/state_se.hpp``.

    global_size is the protein token vocabulary size. raw_counts is 0 for
    detection, 1 for raw counts, 2 for log values. batch sets the number of
    consecutive access positions sharing query tokens; match it to the
    training microbatch. Output: ids int64[length], tasks int64[Q], targets
    float32[Q], counts float32[length], total float32, Q=positive+negative+shared.
    """
    tokens = _tokens(tokens)
    global_size = _integer(global_size, "global_size", minimum=1)
    if tokens.max() >= global_size:
        raise ValueError("nonnegative tokens must be below global_size")
    length = _integer(length, "length", minimum=2)
    positive, negative, shared = (_integer(v, n) for v, n in
                                  ((positive, "positive"), (negative, "negative"), (shared, "shared")))
    q = _integer(positive + negative + shared, "positive + negative + shared", minimum=1)
    batch = _integer(batch, "batch", minimum=1)
    raw_counts = _integer(raw_counts, "raw_counts", maximum=2)
    cls_index, seed, stream = _integer(cls_index, "cls_index"), _seed(seed), _integer(stream, "stream")
    valid, local = _panel(tokens)
    if cls_index >= len(valid):
        raise ValueError("cls_index must be below the number of nonnegative tokens")
    reverse = np.full(global_size, -1, np.int64)
    # Repeated protein tokens resolve to the last panel dim, as the template expects.
    for gene in valid:
        reverse[tokens[gene]] = gene
    params = {"length": length, "positive": positive, "negative": negative, "shared": shared,
                  "batch": batch, "raw_counts": raw_counts, "cls_index": cls_index}
    kwargs = dict(tokens=tokens, global_size=global_size, seed=seed, stream=stream, **params)
    tables = _StateSeTables(tokens.ctypes.data, valid.ctypes.data, local.ctypes.data, reverse.ctypes.data,
                            len(tokens), len(valid), global_size, seed, stream)
    return _bind("state_se", "StateSe", params, tables, [tokens, valid, local, reverse],
                 [("ids", "i8", (length,)), ("tasks", "i8", (q,)), ("targets", "f4", (q,)),
                  ("counts", "f4", (length,)), ("total", "f4")], kwargs, row_format, build)


__all__ = ["geneformer", "scgpt", "state_se", "uce"]
