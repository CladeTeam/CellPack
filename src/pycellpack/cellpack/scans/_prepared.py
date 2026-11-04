"""Prepared codec objects; algorithms and reusable scratch remain native."""

from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy.typing import DTypeLike

from ... import _cellpack
from ..._cellpack import GeneMap
from ..._workers import resolve_workers
from .._arrays import byte_buffer, checked_dtype
from .._types import CSR
from ..selection import as_gene_map
from ._codec import encode_arguments, selection_nnz
from ._types import Bytes, EncodeOptions, Fill, Meta, RowSelector, chunk_rows

_INDEX_DTYPES = tuple(map(np.dtype, ("int32", "int64", "uint32", "uint64")))


class PreparedEncoder:
    """Snapshot an input and build its statistical model once.

    Rust PreparedEncoder borrows immutable input; this Python adapter takes
    private copies so edits to caller arrays cannot invalidate the model.
    Each encode owns its output/workspace and may run concurrently.
    """

    def __init__(self, matrix, rows: RowSelector = None, *, options: EncodeOptions | None = None):
        args = encode_arguments(matrix, rows, options)
        self.dtype = np.dtype(args[0].dtype)
        self._handle = _cellpack.ScansEncoder(*args)
        self.output_bytes, self.workspace_words = self._handle.sizes()

    @cached_property
    def meta(self) -> Meta:
        """Model metadata for independent record encoding, before record placement."""
        return Meta(self._handle.metadata(), source_dtype=self.dtype)

    def encode(self, *, out: np.ndarray | None = None, workspace: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.output_bytes, np.uint8)
        if workspace is None:
            workspace = np.empty(self.workspace_words, np.uint64)
        n = self._handle.encode_into(out, workspace)
        return out[:n]


class Decoder:
    """Decode records of one chunk straight into a panel of ``dtype``.

    ``gene_map`` (a :class:`~pycellpack.cellpack.GeneMap`, integer vector or
    mapping over the chunk's source genes; default every gene in place) is
    composed with the chunk's gene order once. Tables and scratch are
    reused; no owned Parts is created. One decoder serves one call at a
    time (which may use ``workers`` threads); concurrent consumers use
    separate decoders.
    """

    def __init__(self, meta: Meta, *, gene_map=None, n_genes: int | None = None,
                 dtype: DTypeLike | None = None):
        self.meta = meta
        self.dtype = checked_dtype(meta.dtype if dtype is None else dtype)
        mapping = as_gene_map(gene_map, meta.n_dims, n_genes)
        if mapping is None:
            mapping = GeneMap(np.arange(meta.n_dims, dtype=np.uint32))
        self.gene_map = mapping
        self.n_genes = mapping.n_genes
        self._handle = _cellpack.ScansDecoder(meta._handle, mapping, meta.dtype.name, self.dtype.name)

    def _records(self, records: Bytes | None) -> np.ndarray:
        return self.meta.records() if records is None else byte_buffer(records)

    def decode_dense(
        self,
        *,
        records: Bytes | None = None,
        rows: RowSelector = None,
        out: np.ndarray | None = None,
        fill: Fill | str = Fill.ZERO,
        workers: int | None = None,
    ) -> np.ndarray:
        selection, count = chunk_rows(rows, self.meta.n_cells)
        zero = Fill(fill) is Fill.ZERO
        if out is None:
            out = (np.empty if zero else np.zeros)((count, self.n_genes), self.dtype)
        self._handle.decode_into(self._records(records), selection, out, zero=zero,
                                 workers=resolve_workers(workers))
        return out

    def decode_csr(
        self,
        *,
        records: Bytes | None = None,
        rows: RowSelector = None,
        out: CSR | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        index_dtype: DTypeLike = np.int64,
        workers: int | None = None,
    ) -> CSR:
        """Canonical CSR in panel columns; ``out`` buffers need
        :func:`selection_nnz` entries (dropped genes reduce the result)."""
        selection, count = chunk_rows(rows, self.meta.n_cells)
        shape = (count, self.n_genes)
        if out is None:
            index_dtype = np.dtype(index_dtype)
            if index_dtype not in _INDEX_DTYPES:
                raise ValueError("index_dtype must be int32, int64, uint32 or uint64")
            capacity = selection_nnz(self.meta, rows)
            out = CSR(np.empty(count + 1, index_dtype), np.empty(capacity, index_dtype),
                      np.empty(capacity, self.dtype), shape)
        elif not isinstance(out, CSR):
            out = CSR(*out, shape)
        if out.shape != shape:
            raise ValueError("CSR output shape differs from the selection")
        n = self._handle.decode_csr_into(self._records(records), selection, out.indptr,
                                         out.indices, out.data, workers=resolve_workers(workers))
        return CSR(out.indptr, out.indices[:n], out.data[:n], shape)

    def decode_record(
        self,
        record: Bytes | None = None,
        *,
        cell: int | None = None,
        out: np.ndarray | None = None,
        fill: Fill | str = Fill.ZERO,
    ) -> np.ndarray:
        if cell is not None:
            if record is not None:
                raise TypeError("provide record or cell, not both")
            record = self.meta.record(cell)
        if record is None:
            raise TypeError("record or cell is required")
        zero = Fill(fill) is Fill.ZERO
        if out is None:
            out = (np.empty if zero else np.zeros)(self.n_genes, self.dtype)
        if out.shape != (self.n_genes,) or out.dtype != self.dtype:
            raise ValueError("output shape/dtype mismatch")
        self._handle.decode_into(byte_buffer(record), None, out.reshape(1, -1), single=True, zero=zero)
        return out
