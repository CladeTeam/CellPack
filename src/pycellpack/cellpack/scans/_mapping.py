"""Native value conversion and direct mapping of decoded parts.

Gene maps are :class:`~pycellpack.cellpack.GeneMap` objects indexed by the
ids of the parts' space (coded or source genes); integer vectors and
mappings are accepted and validated once per call. Compose a chunk's map
once with :func:`compose_mapping` and pass the result to every record.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from ... import _cellpack
from ..._cellpack import GeneMap
from .._arrays import checked_dtype
from ..selection import as_gene_map, normalize_rows
from ._types import DecodeTables, Fill, Parts, RowSelector, SparseRow, output_dtype

_SparseOut = SparseRow | tuple[np.ndarray, np.ndarray] | None


def compose_mapping(tables: DecodeTables, gene_map, *, n_genes: int | None = None) -> GeneMap:
    """Compose a source gene → panel map with the chunk's coded gene order,
    giving the coded gene → panel map that coded-space parts use."""
    if not isinstance(tables, DecodeTables):
        raise TypeError("compose_mapping requires DecodeTables")
    return tables._handle.compose(_panel(gene_map, tables.meta.n_dims, n_genes))


def cast_values(
    values: np.ndarray,
    dtype: DTypeLike | None = None,
    *,
    source_dtype: DTypeLike | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert values (or raw bytes read as ``source_dtype``) natively."""
    values = np.asarray(values)
    source = checked_dtype(values.dtype if source_dtype is None else source_dtype)
    if values.nbytes % source.itemsize:
        raise ValueError("source bytes do not contain complete values")
    count = values.nbytes // source.itemsize
    target = output_dtype(dtype, out, np.float32)
    if out is None:
        out = np.empty(values.shape if source_dtype is None else (count,), target)
    if out.size != count or (source_dtype is None and out.shape != values.shape):
        raise ValueError("cast output shape differs from the input")
    _cellpack.scans_cast(values, source.name, out)
    return out


def _panel(gene_map, n_source: int, n_genes: int | None) -> GeneMap:
    result = as_gene_map(gene_map, n_source, n_genes)
    if result is None:
        result = GeneMap(np.arange(n_source, dtype=np.uint32))
    return result


def _sparse_out(out: _SparseOut, panel: GeneMap, n_genes: int | None, size: int, target) -> SparseRow:
    if out is None:
        return SparseRow(np.empty(size, np.uint32), np.empty(size, target), panel.n_genes)
    if not isinstance(out, SparseRow):
        out = SparseRow(out[0], out[1], panel.n_genes)
    if n_genes is not None and out.n_genes != n_genes:
        raise ValueError("sparse output width differs from n_genes")
    if out.n_genes != panel.n_genes:
        raise ValueError("sparse output width differs from the gene map's panel width")
    return out


def map_parts(
    parts: Parts,
    gene_map,
    *,
    dtype: DTypeLike | None = None,
    out: _SparseOut = None,
    dictionary: np.ndarray | None = None,
    modes: np.ndarray | None = None,
    n_genes: int | None = None,
) -> SparseRow:
    """Map a row's elements to panel columns, dropping unmapped genes.

    Columns keep the row's id order. Dense-layout records keep their
    non-zero values. ``dictionary`` and ``modes`` reuse converted components.
    """
    panel = _panel(gene_map, parts.meta.n_dims, n_genes if n_genes is not None else _width(out))
    target = output_dtype(dtype, None if out is None else _values(out), parts.dtype)
    out = _sparse_out(out, panel, n_genes, parts.elements, target)
    n = parts._handle.map_into(
        parts.dtype.name, panel, out.indices, out.data, None, dictionary, modes, True
    )
    return SparseRow(out.indices[:n], out.data[:n], out.n_genes)


def map_parts_at(
    parts: Parts,
    positions: RowSelector,
    gene_map,
    *,
    dtype: DTypeLike | None = None,
    out: _SparseOut = None,
    dictionary: np.ndarray | None = None,
    modes: np.ndarray | None = None,
    n_genes: int | None = None,
) -> SparseRow:
    """Map the elements at ``positions``, one output entry per position.

    Entries stay aligned with ``positions``: an element whose gene is
    dropped by ``gene_map`` keeps index ``UNMAPPED``, so filter on it before
    treating the result as a sparse row.
    """
    panel = _panel(gene_map, parts.meta.n_dims, n_genes if n_genes is not None else _width(out))
    positions = normalize_rows(positions, parts.elements).astype(np.uint32)
    target = output_dtype(dtype, None if out is None else _values(out), parts.dtype)
    out = _sparse_out(out, panel, n_genes, len(positions), target)
    n = parts._handle.map_into(
        parts.dtype.name, panel, out.indices, out.data, positions, dictionary, modes, True
    )
    return SparseRow(out.indices[:n], out.data[:n], out.n_genes)


def map_dense(
    parts: Parts,
    gene_map,
    *,
    dtype: DTypeLike | None = None,
    out: np.ndarray | None = None,
    dictionary: np.ndarray | None = None,
    modes: np.ndarray | None = None,
    n_genes: int | None = None,
    fill: Fill | str = Fill.ZERO,
) -> np.ndarray:
    """Scatter a row into a dense panel row of ``n_genes`` columns."""
    target = output_dtype(dtype, out, parts.dtype)
    if out is not None and (out.ndim != 1 or (n_genes is not None and out.shape != (n_genes,))):
        raise ValueError("dense output shape differs from n_genes")
    panel = _panel(gene_map, parts.meta.n_dims, n_genes if out is None else out.shape[0])
    if out is None:
        out = (np.zeros if Fill(fill) is Fill.ASSUME_ZEROED else np.empty)(panel.n_genes, target)
    elif out.shape != (panel.n_genes,):
        raise ValueError("dense output shape differs from the gene map's panel width")
    parts._handle.map_into(
        parts.dtype.name, panel, None, out, None, dictionary, modes, Fill(fill) is Fill.ZERO
    )
    return out


def _values(out):
    return out.data if isinstance(out, SparseRow) else out[1]


def _width(out) -> int | None:
    return out.n_genes if isinstance(out, SparseRow) else None
