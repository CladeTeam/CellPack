"""User-facing ``.cpk`` file tools.

Thin entry points over :mod:`pycellpack.cellpack`: open a stream lazily
(:func:`open_cpk`), decode a selection (:func:`read_cpk`) or row batches
(:func:`iter_cpk`), inspect a file without decoding (:func:`cpk_info`), and
write one from a whole matrix, a Store or a stream of row batches
(:func:`write_cpk`). Sources are paths, Files, Stores or encoded bytes; a
stream holds the matrix only, so names and alignment stay with the caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..cellpack import CSR, File, Store, Writer, write
from ..cellpack._arrays import input_matrix
from ..cellpack._types import SparseRow
from ..cellpack.format import SUPERBLOCK_LEN, parse_superblock
from ..cellpack.scans import EncodeOptions

__all__ = ["CpkInfo", "cpk_info", "iter_cpk", "open_cpk", "read_cpk", "write_cpk"]

#: Rows per batch when a Store is re-encoded.
_COPY_ROWS = 65536


def _file(source, offset: int = 0, length: int | None = None) -> File:
    if isinstance(source, File):
        return source
    if isinstance(source, Store):
        return source.file
    if isinstance(source, (bytes, bytearray, memoryview)):
        return File.from_bytes(source)
    return File(source, offset=offset, length=length)


def open_cpk(source, *, offset: int = 0, length: int | None = None) -> Store:
    """A lazy :class:`~pycellpack.cellpack.Store` over a stream.

    ``source`` is a path (``offset``/``length`` bound a stream embedded in a
    larger file), a :class:`~pycellpack.cellpack.File`, a Store (returned as
    is) or encoded bytes. Nothing is decoded until the Store is read.
    """
    if isinstance(source, Store):
        return source
    return Store(_file(source, offset, length))


def read_cpk(source, cells=None, genes=None, *, format: str | None = None, dtype=None, **kwargs):
    """Decode selected cells and genes of a stream.

    ``cells`` and ``genes`` take what Store indexing takes (slices, integer
    arrays with repeats, boolean masks; None for all). ``format`` is
    ``"dense"`` (a NumPy array), ``"csr"`` (a :class:`~pycellpack.cellpack.CSR`)
    or None for the stored layout; ``dtype`` converts values. Other keywords
    go to :meth:`Store.read` (``workers``, ``out``, ``workspace``, ...).
    """
    store = open_cpk(source)
    if cells is not None or genes is not None:
        store = store[slice(None) if cells is None else cells, slice(None) if genes is None else genes]
    if dtype is not None:
        store = store.astype(dtype)
    return store.read(format, **kwargs)


def iter_cpk(source, batch_size: int = 4096, *, format: str | None = None, cells=None, dtype=None,
             **kwargs) -> Iterator[Any]:
    """Decode successive batches of ``batch_size`` cells (of ``cells`` when
    given), each as :func:`read_cpk` returns them."""
    store = open_cpk(source)
    if cells is not None:
        store = store[cells]
    if dtype is not None:
        store = store.astype(dtype)
    return store.iter_batches(batch_size, format=format, **kwargs)


@dataclass(frozen=True)
class CpkInfo:
    """A stream's layout, read from its superblock and directory alone."""

    path: Path | None
    offset: int
    shape: tuple[int, int]
    dtype: np.dtype
    format: str
    page_size: int
    chunk_rows: int
    n_chunks: int
    nbytes: int

    @property
    def bytes_per_cell(self) -> float:
        """Encoded bytes per stored cell (the whole stream, metadata included)."""
        return self.nbytes / self.shape[0] if self.shape[0] else 0.0


def cpk_info(source, *, offset: int = 0, length: int | None = None) -> CpkInfo:
    """Shape, dtype, layout, page and chunk sizes of a stream, without
    decoding any record."""
    file = _file(source, offset, length)
    with open(file.path, "rb") as stream:
        stream.seek(file.offset)
        header = stream.read(SUPERBLOCK_LEN)
    block = parse_superblock(np.frombuffer(header, np.uint8))
    return CpkInfo(
        path=None if file.in_memory else file.path,
        offset=file.offset,
        shape=(int(block["n_rows"]), int(block["n_cols"])),
        dtype=np.dtype(block["dtype"]),
        format=str(block["format"]),
        page_size=int(block["page_size"]),
        chunk_rows=int(block["chunk_rows"]),
        n_chunks=int(block["n_chunks"]),
        nbytes=int(block["file_len"]),
    )


def _is_matrix(value) -> bool:
    return isinstance(value, (np.ndarray, CSR, SparseRow)) or hasattr(value, "tocsr")


def write_cpk(
    path: str | PathLike,
    matrix,
    *,
    format: str | None = None,
    workers: int | None = None,
    chunk_rows: int = 4096,
    page_size: int = 4096,
    options: EncodeOptions | None = None,
    atomic: bool = False,
) -> Store:
    """Write one ``.cpk`` stream and return a Store over it.

    ``matrix`` is a whole matrix (NumPy, SciPy sparse, CSR or SparseRow), a
    Store (re-encoded batch by batch, so it may be larger than memory, and
    may read the file being replaced when ``atomic=True``) or an iterable of
    row batches with one column count and dtype (streamed; nothing but the
    current batch is held). ``format`` is the stored layout, ``"dense"`` or
    ``"csr"``, by default the input's own (a Store's stored layout, the first
    batch's). The remaining arguments are :class:`~pycellpack.cellpack.Writer`'s.
    """
    if _is_matrix(matrix) and not isinstance(matrix, Store):
        return write(path, matrix, format=format, workers=workers, chunk_rows=chunk_rows, page_size=page_size,
                     options=options, atomic=atomic)
    if isinstance(matrix, Store):
        if matrix.ndim != 2:
            raise ValueError("write_cpk writes two-dimensional Stores")
        layout = format or matrix.format
        batches: Iterable = matrix.iter_batches(max(chunk_rows, _COPY_ROWS), format=layout)
    elif isinstance(matrix, Iterable) and not isinstance(matrix, (str, bytes)):
        layout = format
        batches = matrix
    else:
        raise TypeError("expected a matrix, a Store or an iterable of row batches")
    iterator = iter(batches)
    first = next(iterator, None)
    if first is None:
        raise ValueError("write_cpk needs at least one row batch")
    data, indptr, _, n_cols, _ = input_matrix(first)
    layout = layout or ("dense" if indptr is None else "csr")
    with Writer(path, n_cols, data.dtype, format=layout, chunk_rows=chunk_rows, page_size=page_size,
                options=options, workers=workers, atomic=atomic) as writer:
        writer.append(first)
        for batch in iterator:
            writer.append(batch)
    return Store(path)
