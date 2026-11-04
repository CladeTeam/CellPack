"""Buffer adapters shared by storage and scans.

Native calls borrow C-contiguous, native-endian buffers and canonical CSR
(strictly ascending column ids per row). Inputs are normalized here so callers
can pass any NumPy layout, any SciPy sparse format or the CellPack containers;
dtypes and byte buffers are validated here once for every subsystem.
"""

from __future__ import annotations

import numpy as np

from ._types import CSR, SparseRow

__all__ = ["byte_buffer", "checked_dtype", "input_matrix"]


def _native(array: np.ndarray) -> np.ndarray:
    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    return np.ascontiguousarray(array)


def _scipy_csr(matrix):
    # Duplicates are summed and ids sorted, as SciPy defines canonical CSR.
    csr = matrix.tocsr()
    if not csr.has_canonical_format:
        csr = csr.copy() if csr is matrix else csr
        csr.sum_duplicates()
    return csr


def input_matrix(matrix):
    """Return (data, indptr, indices, n_genes, n_rows); indptr is None if dense."""
    if isinstance(matrix, SparseRow):
        matrix = matrix.to_csr()
    if isinstance(matrix, CSR):
        return (
            _native(matrix.data),
            _native(matrix.indptr),
            _native(matrix.indices),
            matrix.shape[1],
            matrix.shape[0],
        )
    if isinstance(matrix, np.ndarray):
        if np.ma.isMaskedArray(matrix) and np.ma.getmaskarray(matrix).any():
            raise ValueError("fill masked values explicitly before encoding")
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("dense matrix must have one or two dimensions")
        return _native(matrix), None, None, matrix.shape[1], matrix.shape[0]
    if hasattr(matrix, "tocsr") and getattr(matrix, "ndim", 2) == 2:
        csr = _scipy_csr(matrix)
        return (
            _native(csr.data),
            _native(csr.indptr),
            _native(csr.indices),
            csr.shape[1],
            csr.shape[0],
        )
    raise TypeError("expected a NumPy array, a SciPy sparse matrix, CSR or SparseRow")


def byte_buffer(buffer) -> np.ndarray:
    """A contiguous uint8 view of encoded bytes (NumPy or buffer protocol)."""
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(buffer, dtype=np.uint8)
    if (
        not isinstance(buffer, np.ndarray)
        or buffer.dtype != np.uint8
        or buffer.ndim != 1
        or not buffer.flags.c_contiguous
    ):
        raise TypeError("encoded buffers require a contiguous uint8 vector or byte buffer")
    return buffer


def checked_dtype(dtype, width: int | None = None) -> np.dtype:
    """A native numeric dtype CellPack stores, optionally of a given width."""
    result = np.dtype(dtype)
    if (
        result.kind not in "uif"
        or result.itemsize not in (1, 2, 4, 8)
        or (result.kind == "f" and result.itemsize not in (4, 8))
        or not result.isnative
    ):
        raise TypeError("a native numeric dtype supported by CellPack is required")
    if width is not None and result.itemsize != width:
        raise ValueError("dtype width differs from metadata")
    return result
