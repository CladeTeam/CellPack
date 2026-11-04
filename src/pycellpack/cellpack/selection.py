"""Shared public normalization of cell selectors and source-to-target gene maps."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import SupportsIndex, cast

import numpy as np

from .._cellpack import GeneMap
from ._types import UNMAPPED

__all__ = ["as_gene_map", "normalize_rows"]

_UNNAMED = "columns are selected by positions; cellpack keeps no gene names"


def normalize_rows(rows, size: int) -> np.ndarray:
    """Normalize positional selectors to contiguous uint64 IDs, preserving order and repeats."""
    if rows is None:
        return np.arange(size, dtype=np.uint64)
    if isinstance(rows, slice):
        return (
            np.arange(*rows.indices(size), dtype=np.uint64)
            if (rows.step or 1) > 0
            else np.arange(*rows.indices(size), dtype=np.int64).astype(np.uint64)
        )
    if np.isscalar(rows):
        if isinstance(rows, str):
            raise TypeError(_UNNAMED)
        rows = [operator.index(cast(SupportsIndex, rows))]
    a = np.asarray(rows)
    if a.ndim == 0 and a.dtype.kind in "iu":
        a = a.reshape(1)
    if a.ndim != 1:
        raise IndexError("row indices must be one-dimensional")
    if a.dtype.kind == "b":
        if len(a) != size:
            raise IndexError("boolean mask length differs from cell count")
        return np.flatnonzero(a).astype(np.uint64)
    if not len(a):
        return np.empty(0, dtype=np.uint64)
    if a.dtype.kind in "USO" and any(isinstance(n, str) for n in a.flat):
        raise TypeError(_UNNAMED)
    if a.dtype.kind not in "iu":
        raise TypeError("row indices must be integers or booleans")
    if a.dtype.kind == "i":
        a = a.astype(np.int64, copy=False)
        a = np.where(a < 0, a + size, a)
        if np.any(a < 0):
            raise IndexError("row index out of bounds")
    if np.any(a >= size):
        raise IndexError("row index out of bounds")
    return np.ascontiguousarray(a, dtype=np.uint64)


def as_gene_map(gene_map, n_source: int, n_genes: int | None = None) -> GeneMap | None:
    """Resolve ``gene_map`` over ``n_source`` columns into a :class:`GeneMap`.

    Accepts a ``GeneMap`` (used as is unless ``n_genes`` sets another
    width), an integer vector with one target per
    source column, or a mapping from source positions to targets; -1 and
    ``UNMAPPED`` drop a column. ``n_genes`` sets the panel width. Without a
    map, a width other than ``n_source`` places every column in a wider
    panel; otherwise the result is None (every column in place).
    """
    if isinstance(gene_map, GeneMap):
        if len(gene_map) != n_source:
            raise ValueError("gene map must cover every source column")
        if n_genes is None or n_genes == gene_map.n_genes:
            return gene_map
        # Another panel width revalidates the targets against it.
        gene_map = gene_map.targets
    if gene_map is None:
        if n_genes is None or n_genes == n_source:
            return None
        return GeneMap(np.arange(n_source, dtype=np.uint32), n_genes)
    if isinstance(gene_map, Mapping):
        targets = np.full(n_source, UNMAPPED, np.uint32)
        for source, target in gene_map.items():
            if isinstance(source, str):
                raise TypeError(_UNNAMED)
            target = operator.index(target)
            if not -1 <= target <= UNMAPPED:
                raise ValueError("gene map targets need uint32 positions, -1 or UNMAPPED")
            targets[int(normalize_rows([source], n_source)[0])] = UNMAPPED if target == -1 else target
        gene_map = targets
    result = GeneMap(gene_map, n_genes)
    if len(result) != n_source:
        raise ValueError("gene map must cover every source column")
    return result
