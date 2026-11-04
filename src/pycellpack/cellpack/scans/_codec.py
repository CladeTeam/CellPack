"""Encoding, resident-buffer decoding and output sizing."""

from __future__ import annotations

from typing import Literal, NamedTuple, overload

import numpy as np
from numpy.typing import DTypeLike

from ... import _cellpack
from ..._workers import resolve_workers
from .._arrays import byte_buffer, input_matrix
from .._types import CSR
from ._types import (
    Bytes,
    DecodeTables,
    EncodeOptions,
    EncodeTables,
    Fill,
    IdSpace,
    Meta,
    PagedRecord,
    Parts,
    RowSelector,
    SparseRow,
    chunk_rows,
    coded,
)

_CSROut = CSR | tuple[np.ndarray, np.ndarray, np.ndarray] | None
_SparseOut = SparseRow | tuple[np.ndarray, np.ndarray] | None


class ChunkSizes(NamedTuple):
    """Capacities one chunk encoding needs."""

    output_bytes: int
    workspace_words: int


def _tables(meta: Meta, tables: DecodeTables | None) -> DecodeTables:
    tables = DecodeTables(meta) if tables is None else tables
    if not isinstance(tables, DecodeTables):
        raise TypeError("decoding requires DecodeTables")
    tables._handle.check(meta._handle)
    return tables


def _workspace(meta: Meta, workspace: np.ndarray | None) -> np.ndarray:
    return np.empty(meta.decode_workspace_words, np.uint64) if workspace is None else workspace


def selection_nnz(meta: Meta, rows: RowSelector = None) -> int:
    """Stored entries of the selected cells (the CSR capacity)."""
    return meta._handle.selection_sizes(chunk_rows(rows, meta.n_cells)[0])[0]


def dense_bytes(meta: Meta, rows: RowSelector = None) -> int:
    """Bytes of the selected cells' dense rows in the stored dtype."""
    return meta._handle.selection_sizes(chunk_rows(rows, meta.n_cells)[0])[1]


def decode_workspace_bound(n_genes: int, value_width: int) -> int:
    """Decode workspace words sufficient for any chunk of these dimensions."""
    return _cellpack.scans_workspace("decode", n_genes, 0, value_width)


def encode_workspace_words(n_genes: int, max_elems: int) -> int:
    """Workspace words :func:`encode_record` needs for rows of this size."""
    return _cellpack.scans_workspace("encode", n_genes, max_elems, 1)


def record_bound(n_genes: int, value_width: int, nnz: int, dense: bool = False) -> int:
    """Upper bound of one encoded record's bytes."""
    return _cellpack.scans_record_bound(n_genes, value_width, nnz, dense)


def _options(options: EncodeOptions | None) -> dict:
    return (EncodeOptions() if options is None else options)._asdict()


def encode_arguments(matrix, rows: RowSelector, options: EncodeOptions | None):
    """Native encoder arguments: data, indptr, indices, genes, rows, options."""
    data, ip, ix, genes, count = input_matrix(matrix)
    return data, ip, ix, genes, chunk_rows(rows, count)[0], _options(options)


def chunk_sizes(matrix, rows: RowSelector = None, *, options: EncodeOptions | None = None) -> ChunkSizes:
    """Output bytes and uint64 workspace words :func:`encode_chunk` needs."""
    return ChunkSizes(*_cellpack.scans_encode_sizes(*encode_arguments(matrix, rows, options)))


def encode_chunk(
    matrix,
    rows: RowSelector = None,
    *,
    options: EncodeOptions | None = None,
    out: np.ndarray | None = None,
    workspace: np.ndarray | None = None,
) -> np.ndarray:
    """Encode selected rows as one chunk; missing buffers are sized first."""
    args = encode_arguments(matrix, rows, options)
    if out is None or workspace is None:
        bound, words = _cellpack.scans_encode_sizes(
            *args, output=out is None, workspace=workspace is None
        )
        if out is None:
            out = np.empty(bound, np.uint8)
        if workspace is None:
            workspace = np.empty(words, np.uint64)
    size = _cellpack.scans_encode_chunk(*args, out, workspace)
    return out[:size]


def encode_record(
    meta: Meta,
    row,
    *,
    tables: EncodeTables | None = None,
    options: EncodeOptions | None = None,
    out: np.ndarray | None = None,
    workspace: np.ndarray | None = None,
) -> np.ndarray:
    """Encode one row against an existing chunk model."""
    tables = EncodeTables(meta) if tables is None else tables
    if not isinstance(tables, EncodeTables):
        raise TypeError("encode_record requires EncodeTables")
    tables._handle.check(meta._handle)
    data, ip, ix, genes, count = input_matrix(row)
    if genes != meta.n_dims or count != 1 or data.dtype != meta.dtype:
        raise ValueError("record row shape/dtype differs from metadata")
    if out is None:
        out = np.empty(record_bound(meta.n_dims, meta.value_width, meta.max_nnz, dense=True), np.uint8)
    if workspace is None:
        workspace = np.empty(encode_workspace_words(meta.n_dims, meta.max_elems), np.uint64)
    size = _cellpack.scans_encode_record(
        meta._handle, tables._handle, data, ip, ix, _options(options), out, workspace
    )
    return out[:size]


def packed_len(chunk: Bytes, page_size: int = 4096) -> int:
    """Bytes of the chunk packed to ``page_size`` pages."""
    return _cellpack.scans_pack(byte_buffer(chunk), page_size, None)


def pack_chunk(chunk: Bytes, page_size: int = 4096, *, out: np.ndarray | None = None) -> np.ndarray:
    """Place a chunk's records so each reads its minimum number of pages."""
    chunk = byte_buffer(chunk)
    if out is None:
        out = np.empty(packed_len(chunk, page_size), np.uint8)
    size = _cellpack.scans_pack(chunk, page_size, out)
    return out[:size]


def record_nnz(record: Bytes | PagedRecord) -> int:
    """The non-zero count a record declares."""
    if isinstance(record, PagedRecord):
        # Only the bounded record header is needed for the declared nnz.
        header = np.empty(min(11, record.length), np.uint8)
        written, skip = 0, record.start
        for page in record.pages:
            n = min(len(header) - written, max(0, len(page) - skip))
            header[written : written + n] = page[skip : skip + n]
            written += n
            skip = 0
            if written == len(header):
                break
        if written != len(header):
            raise ValueError("pages do not contain the record header")
        record = header
    return _cellpack.scans_record_nnz(byte_buffer(record))


def _mapped(gene_map, n_genes, dtype, space) -> bool:
    if gene_map is None and n_genes is None and dtype is None:
        return False
    if IdSpace(space) is not IdSpace.SOURCE:
        raise ValueError("gene maps are indexed by source genes")
    return True


def decode_dense(
    meta: Meta,
    *,
    records: Bytes | None = None,
    rows: RowSelector = None,
    tables: DecodeTables | None = None,
    out: np.ndarray | None = None,
    workspace: np.ndarray | None = None,
    space: IdSpace | str = IdSpace.SOURCE,
    fill: Fill | str = Fill.ZERO,
    workers: int | None = None,
    gene_map=None,
    n_genes: int | None = None,
    dtype: DTypeLike | None = None,
) -> np.ndarray:
    """Decode selected cells densely; many rows decode on ``workers`` threads.

    ``gene_map`` (a :class:`~pycellpack.cellpack.GeneMap`, an integer
    vector or a mapping of source genes to panel columns), ``n_genes`` and
    ``dtype`` decode straight into a panel through :class:`Decoder`.
    ``workspace`` serves the first thread; others allocate their own.
    """
    if _mapped(gene_map, n_genes, dtype, space):
        from ._prepared import Decoder

        return Decoder(meta, gene_map=gene_map, n_genes=n_genes, dtype=dtype).decode_dense(
            records=records, rows=rows, out=out, fill=fill, workers=workers
        )
    tables = _tables(meta, tables)
    selection, count = chunk_rows(rows, meta.n_cells)
    if out is None:
        out = (np.empty if Fill(fill) is Fill.ZERO else np.zeros)((count, meta.n_dims), meta.dtype)
    if out.dtype != meta.dtype or out.shape != (count, meta.n_dims):
        raise ValueError("dense output dtype/shape differs from selection")
    tables._handle.decode_dense_into(
        meta._handle,
        meta.records() if records is None else byte_buffer(records),
        selection,
        out,
        _workspace(meta, workspace),
        coded(space),
        Fill(fill) is Fill.ZERO,
        resolve_workers(workers),
    )
    return out


def decode_csr(
    meta: Meta,
    *,
    records: Bytes | None = None,
    rows: RowSelector = None,
    tables: DecodeTables | None = None,
    out: _CSROut = None,
    workspace: np.ndarray | None = None,
    space: IdSpace | str = IdSpace.SOURCE,
    index_dtype: DTypeLike = np.int64,
    workers: int | None = None,
    gene_map=None,
    n_genes: int | None = None,
    dtype: DTypeLike | None = None,
) -> CSR:
    """Decode selected cells as CSR; many rows decode on ``workers`` threads.

    With ``gene_map``/``n_genes``/``dtype`` the result is in the panel:
    dropped genes are removed and each row's columns ascend. ``out``
    buffers need :func:`selection_nnz` entries.
    """
    if _mapped(gene_map, n_genes, dtype, space):
        from ._prepared import Decoder

        return Decoder(meta, gene_map=gene_map, n_genes=n_genes, dtype=dtype).decode_csr(
            records=records, rows=rows, out=out, index_dtype=index_dtype, workers=workers
        )
    tables = _tables(meta, tables)
    selection, count = chunk_rows(rows, meta.n_cells)
    if out is None:
        nnz = meta._handle.selection_sizes(selection)[0]
        out = CSR(
            np.empty(count + 1, index_dtype),
            np.empty(nnz, index_dtype),
            np.empty(nnz, meta.dtype),
            (count, meta.n_dims),
        )
    elif not isinstance(out, CSR):
        out = CSR(*out, (count, meta.n_dims))
    if out.shape != (count, meta.n_dims) or out.data.dtype != meta.dtype:
        raise ValueError("CSR output shape/dtype mismatch")
    n = tables._handle.decode_csr_into(
        meta._handle,
        meta.records() if records is None else byte_buffer(records),
        selection,
        out.indptr,
        out.indices,
        out.data,
        _workspace(meta, workspace),
        coded(space),
        resolve_workers(workers),
    )
    return CSR(out.indptr, out.indices[:n], out.data[:n], out.shape)


def _record(meta: Meta, record, cell):
    if cell is not None:
        if record is not None:
            raise TypeError("provide record or cell, not both")
        return meta.record(cell)
    if record is None:
        raise TypeError("a record buffer or cell index is required")
    return record


def decode_record(
    meta: Meta,
    record: Bytes | PagedRecord | None = None,
    *,
    cell: int | None = None,
    tables: DecodeTables | None = None,
    out: _SparseOut = None,
    workspace: np.ndarray | None = None,
    space: IdSpace | str = IdSpace.SOURCE,
    index_dtype: DTypeLike = np.int64,
) -> SparseRow:
    """Decode one record (bytes, pages or ``cell=``) as a sparse row."""
    record = _record(meta, record, cell)
    tables = _tables(meta, tables)
    nnz = record_nnz(record)
    if nnz > meta.n_dims:
        raise ValueError("corrupt record: nnz exceeds the gene count")
    if out is None:
        out = SparseRow(np.empty(nnz, index_dtype), np.empty(nnz, meta.dtype), meta.n_dims)
    elif not isinstance(out, SparseRow):
        out = SparseRow(*out, meta.n_dims)
    if out.data.dtype != meta.dtype:
        raise ValueError("value dtype differs from metadata")
    workspace = _workspace(meta, workspace)
    if isinstance(record, PagedRecord):
        n = tables._handle.decode_pages_into(
            meta._handle, record.pages, record.page_size, record.start, record.length,
            out.indices, out.data, workspace, coded(space), True,
        )
    else:
        n = tables._handle.decode_record_into(
            meta._handle, byte_buffer(record), out.indices, out.data, workspace, coded(space), True
        )
    return SparseRow(out.indices[:n], out.data[:n], meta.n_dims)


def decode_record_dense(
    meta: Meta,
    record: Bytes | PagedRecord | None = None,
    *,
    cell: int | None = None,
    tables: DecodeTables | None = None,
    out: np.ndarray | None = None,
    workspace: np.ndarray | None = None,
    space: IdSpace | str = IdSpace.SOURCE,
    fill: Fill | str = Fill.ZERO,
) -> np.ndarray:
    """Decode one record (bytes, pages or ``cell=``) as a dense row."""
    record = _record(meta, record, cell)
    tables = _tables(meta, tables)
    zero = Fill(fill) is Fill.ZERO
    if out is None:
        out = (np.empty if zero else np.zeros)(meta.n_dims, meta.dtype)
    if out.dtype != meta.dtype or out.shape != (meta.n_dims,):
        raise ValueError("dense row dtype/shape mismatch")
    workspace = _workspace(meta, workspace)
    if isinstance(record, PagedRecord):
        tables._handle.decode_pages_into(
            meta._handle, record.pages, record.page_size, record.start, record.length,
            None, out, workspace, coded(space), zero,
        )
    else:
        tables._handle.decode_record_into(
            meta._handle, byte_buffer(record), None, out, workspace, coded(space), zero
        )
    return out


def decode_parts(
    meta: Meta,
    record: Bytes | PagedRecord | None = None,
    *,
    cell: int | None = None,
    tables: DecodeTables | None = None,
    space: IdSpace | str = IdSpace.CODED,
) -> Parts:
    """Decode one record (bytes, pages or ``cell=``) into owned parts."""
    record = _record(meta, record, cell)
    tables = _tables(meta, tables)
    if isinstance(record, PagedRecord):
        handle = _cellpack.ScansParts.from_pages(
            meta._handle, tables._handle, record.pages, record.page_size,
            record.start, record.length, coded(space),
        )
    else:
        handle = _cellpack.ScansParts(meta._handle, tables._handle, byte_buffer(record), coded(space))
    return Parts(handle, meta, space)


@overload
def decode_chunk(
    buffer: Bytes | Meta,
    *,
    source_dtype: DTypeLike | None = ...,
    format: Literal["dense"] = ...,
    rows: RowSelector = ...,
    out: np.ndarray | None = ...,
    workspace: np.ndarray | None = ...,
    tables: DecodeTables | None = ...,
    space: IdSpace | str = ...,
    workers: int | None = ...,
    gene_map=...,
    n_genes: int | None = ...,
    dtype: DTypeLike | None = ...,
) -> np.ndarray: ...
@overload
def decode_chunk(
    buffer: Bytes | Meta,
    *,
    source_dtype: DTypeLike | None = ...,
    format: Literal["csr"],
    rows: RowSelector = ...,
    out: _CSROut = ...,
    workspace: np.ndarray | None = ...,
    tables: DecodeTables | None = ...,
    space: IdSpace | str = ...,
    workers: int | None = ...,
    gene_map=...,
    n_genes: int | None = ...,
    dtype: DTypeLike | None = ...,
) -> CSR: ...
def decode_chunk(
    buffer,
    *,
    source_dtype=None,
    format="dense",
    rows=None,
    out=None,
    workspace=None,
    tables=None,
    space: IdSpace | str = IdSpace.SOURCE,
    workers=None,
    gene_map=None,
    n_genes=None,
    dtype=None,
):
    """Decode an encoded chunk: bytes read as ``source_dtype``, or a parsed
    :class:`Meta`, into ``format`` (``"dense"`` or ``"csr"``).

    Chunks record value width but not signedness or float-ness, so raw bytes
    need the ``source_dtype`` they were encoded from. ``dtype`` and the gene
    map arguments select the output as in :func:`decode_dense`.
    """
    if isinstance(buffer, Meta):
        if source_dtype is not None and np.dtype(source_dtype) != buffer.dtype:
            raise ValueError("source_dtype differs from the parsed metadata")
        meta = buffer
    elif source_dtype is None:
        raise TypeError("decoding raw chunk bytes requires source_dtype=")
    else:
        meta = Meta(buffer, source_dtype=source_dtype)
    common = {"rows": rows, "out": out, "workspace": workspace, "tables": tables, "space": space,
              "workers": workers, "gene_map": gene_map, "n_genes": n_genes, "dtype": dtype}
    if format == "dense":
        return decode_dense(meta, **common)
    if format == "csr":
        return decode_csr(meta, **common)
    raise ValueError("format must be dense or csr")
