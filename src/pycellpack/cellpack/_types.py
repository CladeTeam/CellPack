"""Shared NumPy data containers, independent of codecs and file I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Drops a gene in a source-to-panel map; the largest uint32, as in Rust.
UNMAPPED = int(np.iinfo(np.uint32).max)


@dataclass
class CSR:
    """Canonical CSR arrays, all sharing one decoded dtype.

    ``indptr`` holds ``shape[0] + 1`` row starts, ``indices`` and ``data`` are
    parallel and column ids ascend within each row. ``data`` may be longer
    than ``nnz``: callers that reserve a capacity slice the views they export.
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        """Stored element count: the last ``indptr`` entry."""
        return int(self.indptr[-1])

    def to_scipy(self):
        """Build a scipy.sparse.csr_array over these arrays, asking for no copy.

        SciPy normally shares the buffers, so mutating them later changes the
        matrix too; indices must already be canonical for SciPy's own
        operations.
        """
        from scipy.sparse import csr_array

        return csr_array(
            (self.data, self.indices, self.indptr), shape=self.shape, copy=False
        )


@dataclass
class SparseRow:
    """One row of stored elements over ``n_genes`` columns.

    ``indices`` are gene ids in the row's id space (source genes, or coded
    genes for "coded" decodes) and ``data`` holds the matching values. A
    dense record expands to one entry per gene, zeros included.
    """
    indices: np.ndarray
    data: np.ndarray
    n_genes: int

    @property
    def nnz(self) -> int:
        """Number of entries, i.e. ``len(data)``."""
        return len(self.data)

    @property
    def shape(self) -> tuple[int]:
        """Row shape as a tuple, as NumPy-style consumers expect."""
        return (self.n_genes,)

    def to_csr(self) -> CSR:
        """Wrap the row as a one-row CSR matrix; ``indices`` and ``data`` are shared."""
        return CSR(
            np.array([0, self.nnz], dtype=np.int64),
            self.indices,
            self.data,
            (1, self.n_genes),
        )
