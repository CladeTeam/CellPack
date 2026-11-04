"""Lazy matrix views over CellPack streams: the array API.

A Store records which cells of a stream it shows and where each output column
comes from; indexing and gene mapping only compose those records. Decoding
happens when data are requested, so a large stream (on disk, inside an H5CP
file or in memory) can be indexed piecewise without being read.
"""

from __future__ import annotations

import math
import operator
from os import PathLike
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .._cellpack import GeneMap
from ._arrays import checked_dtype
from ._file import File, ReadOptions, ReadSelection, encode
from ._types import CSR, UNMAPPED
from .scans import EncodeOptions
from .selection import as_gene_map, normalize_rows

if TYPE_CHECKING:
    from typing_extensions import Self

    from .._cellpack import ReadWorkspace

__all__ = ["Store", "compress", "read", "write"]

_Rows = slice | np.ndarray
_ABSENT = -1


class _Plan(NamedTuple):
    """How a view's columns are decoded: the source → decoded column map
    (None: every column in place), and for repeated source columns the
    copy from decoded to output columns (``inverse``, ``present``)."""

    gene_map: GeneMap | None
    expand: tuple[np.ndarray, np.ndarray] | None


class Store:
    """An immutable, lazily indexed view of one CellPack stream.

    Rows are cells. Each output column is a stored column or an absent one,
    which reads as zeros, so a Store can present a stream in a panel the
    caller lays out. Columns are addressed by position only. Views come from:

    - ``store[cells, genes]``: orthogonal selection, as in AnnData or h5py.
      Integers keep their axis. Both axes take slices, integers, integer
      arrays (any order, repeats) and boolean masks.
    - :meth:`map_genes`: place columns by a caller-given column → panel map.
    - :meth:`astype`: convert values while decoding.

    None of these read data. :meth:`read`, :meth:`read_dense` and
    :meth:`read_csr` decode into NumPy / :class:`~pycellpack.cellpack.CSR`
    (into caller buffers with ``out=``); :meth:`prepare` returns a
    :class:`ReadSelection` for repeated reads; :meth:`to_memory`,
    :meth:`toarray`, :meth:`tocsr` and ``np.asarray`` return NumPy / SciPy
    objects. Higher-rank dense arrays stored with flattened trailing axes
    take their logical ``shape`` and select cells only.
    """

    __slots__ = ("_cols", "_dtype", "_file", "_plan", "_rows", "_shape")

    def __init__(
        self,
        source: File | Store | str | PathLike,
        *,
        shape: tuple[int, ...] | None = None,
        dtype=None,
    ):
        if isinstance(source, Store):
            if shape is not None and tuple(shape) != source.shape:
                raise ValueError("a Store keeps the shape of its selection")
            self._init(source._file, source._rows, source._cols, source._shape,
                       source._dtype if dtype is None else checked_dtype(dtype), source._plan)
            return
        file = source if isinstance(source, File) else File(source)
        n_rows, n_flat = file.shape
        logical = (n_rows, n_flat) if shape is None else tuple(int(n) for n in shape)
        if len(logical) < 2 or logical[0] != n_rows or math.prod(logical[1:]) != n_flat:
            raise ValueError(
                f"shape {logical} is incompatible with stream shape {file.shape}"
            )
        if len(logical) > 2 and file.format != "dense":
            raise ValueError("only dense streams can hold higher-rank arrays")
        self._init(file, slice(0, n_rows), None, logical,
                   file.dtype if dtype is None else checked_dtype(dtype), None)

    def _init(self, file, rows, cols, shape, dtype, plan):
        if isinstance(rows, np.ndarray):
            rows.flags.writeable = False
        if cols is not None:
            cols.flags.writeable = False
        self._file = file
        self._rows = rows
        self._cols = cols
        self._shape = shape
        self._dtype = dtype
        self._plan = plan

    def _derive(self, rows=None, cols=None, dtype=None, *, keep_cols=True) -> Store:
        rows = self._rows if rows is None else rows
        plan = None
        if keep_cols:
            cols, plan = self._cols, self._plan
        new = Store.__new__(Store)
        n_rows = _length(rows)
        if self.ndim > 2:
            shape = (n_rows, *self._shape[1:])
        else:
            shape = (n_rows, self._file.shape[1] if cols is None else len(cols))
        new._init(self._file, rows, cols, shape,
                  self._dtype if dtype is None else dtype, plan)
        return new

    # -- array metadata ----------------------------------------------------

    @property
    def file(self) -> File:
        """The underlying stream; views never modify it."""
        return self._file

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return math.prod(self._shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def format(self) -> str:
        """Natural output format of the stream: ``"dense"`` or ``"csr"``."""
        return self._file.format

    @property
    def cells(self) -> np.ndarray:
        """Read-only source cell ids in output order."""
        if isinstance(self._rows, slice):
            rows = np.arange(self._rows.start, self._rows.stop, dtype=np.uint64)
            rows.flags.writeable = False
            return rows.view()
        return self._rows.view()

    @property
    def genes(self) -> np.ndarray | None:
        """Read-only source ids per output column (-1 absent), or None for all."""
        return None if self._cols is None else self._cols.view()

    @property
    def selects_all_cells(self) -> bool:
        """True when the view shows every stored cell once, in order."""
        return isinstance(self._rows, slice) and self._rows == slice(0, self._file.shape[0])

    @property
    def is_identity(self) -> bool:
        """True when the view is the whole stream in its stored dtype."""
        return self.selects_all_cells and self._cols is None and self._dtype == self._file.dtype

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        where = "memory" if self._file.in_memory else str(self._file.path)
        view = "" if self.is_identity else ", view"
        return (
            f"Store(shape={self._shape}, dtype={self._dtype}, format={self.format!r}, "
            f"file={where!r}{view})"
        )

    # -- lazy views --------------------------------------------------------

    def __getitem__(self, key) -> Store:
        rows_key, cols_key = _split_key(key, self.ndim)
        rows = None if _is_full(rows_key) else _compose_rows(
            self._rows, rows_key, self._shape[0]
        )
        if _is_full(cols_key):
            return self if rows is None else self._derive(rows)
        if self.ndim > 2:
            raise IndexError("higher-rank stores can only select cells")
        cols = self._take_columns(_positions(cols_key, self._shape[1]).astype(np.int64))
        return self._derive(rows, cols, keep_cols=False)

    def map_genes(self, gene_map, n_genes: int | None = None) -> Store:
        """Place this view's columns into a panel of width ``n_genes``.

        ``gene_map`` gives each current column's panel position: a
        :class:`GeneMap`, an integer array with -1 or ``UNMAPPED`` for
        dropped columns, or a mapping from column positions to panel
        positions. Positions must be distinct; unfilled panel positions read
        as zeros.
        """
        if self.ndim > 2:
            raise ValueError("higher-rank stores have no gene axis")
        mapping = as_gene_map(gene_map, self._shape[1], n_genes)
        if mapping is None:
            return self
        targets = mapping.targets
        kept = targets != UNMAPPED
        cols = np.full(mapping.n_genes, _ABSENT, np.int64)
        cols[targets[kept]] = self._current_cols()[kept]
        return self._derive(cols=_canonical_cols(cols, self._file.shape[1]), keep_cols=False)

    def astype(self, dtype, copy: bool = True) -> Store:
        """Convert values to another CellPack numeric dtype while decoding."""
        del copy
        dtype = checked_dtype(dtype)
        if dtype == self._dtype:
            return self
        return self._derive(dtype=dtype)

    def copy(self) -> Self:
        """Return this view; Stores are immutable, so sharing is a copy."""
        return self

    def __reduce__(self):
        return (_restore_store, (self._file, self._rows, self._cols, self._shape, self._dtype))

    def __setitem__(self, key, value):
        raise TypeError("Store is read-only; materialize it with to_memory() first")

    def _current_cols(self) -> np.ndarray:
        if self._cols is None:
            return np.arange(self._file.shape[1], dtype=np.int64)
        return self._cols

    def _take_columns(self, positions: np.ndarray):
        # Positions index the current columns; -1 selects an absent column.
        keep = positions >= 0
        cols = np.full(len(positions), _ABSENT, np.int64)
        cols[keep] = self._current_cols()[positions[keep]]
        return _canonical_cols(cols, self._file.shape[1])

    # -- decoding ----------------------------------------------------------

    def _column_plan(self) -> _Plan:
        """The decode plan of this view's columns, resolved once per view."""
        if self._plan is not None:
            return self._plan
        cols = self._cols
        if cols is None:
            plan = _Plan(None, None)
        else:
            present = np.flatnonzero(cols >= 0)
            sources = cols[present]
            unique, inverse = np.unique(sources, return_inverse=True)
            mapping = np.full(self._file.shape[1], UNMAPPED, np.uint32)
            if len(unique) == len(sources):
                # Distinct sources decode straight into their output columns.
                mapping[sources] = present
                plan = _Plan(GeneMap(mapping, len(cols)), None)
            else:
                # Repeated sources decode once and are copied out afterwards.
                mapping[unique] = np.arange(len(unique))
                plan = _Plan(GeneMap(mapping, len(unique)), (inverse.reshape(-1), present))
        self._plan = plan
        return plan

    def _selection(self, plan: _Plan, read_options: ReadOptions | None) -> ReadSelection:
        return self._file.prepare_read(self._rows, gene_map=plan.gene_map, dtype=self._dtype,
                                       read_options=read_options)

    def prepare(self, *, read_options: ReadOptions | None = None) -> ReadSelection:
        """Prepare reusable read metadata (views without repeated genes)."""
        plan = self._column_plan()
        if plan.expand is not None:
            raise ValueError("views with repeated genes cannot be prepared; use read()")
        return self._selection(plan, read_options)

    def read(
        self,
        format: str | None = None,
        *,
        out=None,
        index_dtype=np.int64,
        workspace: ReadWorkspace | None = None,
        read_options: ReadOptions | None = None,
        workers: int | None = None,
    ):
        """Decode as NumPy (logical shape) or :class:`CSR`.

        ``format`` defaults to the stream's own. Dense ``out`` must be
        C-contiguous with the view's size and dtype; CSR ``out`` follows
        :meth:`ReadSelection.read_csr`. Chunks decode on ``workers`` threads
        (default: one per CPU, at most 32).
        """
        format = self.format if format is None else format
        common = {"workspace": workspace, "read_options": read_options, "workers": workers}
        if format == "dense":
            return self.read_dense(out=out, **common)
        if format == "csr":
            return self.read_csr(out=out, index_dtype=index_dtype, **common)
        raise ValueError("format must be dense or csr")

    def read_dense(self, *, out=None, workspace=None, read_options=None,
                   workers=None) -> np.ndarray:
        flat = _flat_out(out, self)
        plan = self._column_plan()
        options = {"workspace": workspace, "workers": workers}
        if plan.expand is None:
            dense = self._selection(plan, read_options).read_dense(out=flat, **options)
        else:
            inverse, present = plan.expand
            decoded = self._selection(plan, read_options).read_dense(**options)
            dense = flat if flat is not None else np.empty(
                (self._shape[0], self._shape[1]), self._dtype
            )
            dense[...] = 0
            dense[:, present] = decoded[:, inverse]
        return out if out is not None else dense.reshape(self._shape)

    def read_csr(self, *, out=None, index_dtype=np.int64, workspace=None,
                 read_options=None, workers=None) -> CSR:
        if self.ndim > 2:
            raise ValueError("higher-rank stores can only be read densely")
        plan = self._column_plan()
        if plan.expand is not None and out is not None:
            raise ValueError("CSR output buffers cannot receive repeated genes")
        decoded = self._selection(plan, read_options).read_csr(
            out=out, index_dtype=index_dtype, workspace=workspace, workers=workers
        )
        return decoded if plan.expand is None else _expand_csr(decoded, *plan.expand, self._shape)

    def to_memory(self):
        """Decode in the stream's format: NumPy, or a SciPy ``csr_array``."""
        result = self.read()
        return result.to_scipy() if isinstance(result, CSR) else result

    def toarray(self) -> np.ndarray:
        return self.read_dense()

    def tocsr(self):
        """Decode as a SciPy ``csr_array``."""
        return self.read_csr().to_scipy()

    def __array__(self, dtype=None, copy=None):
        del copy
        dense = self.read_dense()
        return dense if dtype is None else dense.astype(dtype, copy=False)

    def iter_batches(self, batch_size: int = 4096, *, format: str | None = None, **kwargs):
        """Yield ``read()`` results for consecutive cell batches of this view."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self._shape[0], batch_size):
            yield self[start : start + batch_size].read(format, **kwargs)


def read(path: str | PathLike | File, cells=None, genes=None, *, format: str | None = None,
         dtype=None, **kwargs):
    """Decode selected cells and genes of a stream; see :meth:`Store.read`."""
    store = Store(path)
    if cells is not None or genes is not None:
        store = store[slice(None) if cells is None else cells, slice(None) if genes is None else genes]
    if dtype is not None:
        store = store.astype(dtype)
    return store.read(format, **kwargs)


def write(
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
    """Write a matrix to one ``.cpk`` stream and return a Store over it.

    ``matrix`` may be NumPy (any layout), SciPy sparse (any format), CSR or
    SparseRow; sparse input is canonicalized. ``format`` picks the stored
    layout (``"dense"`` or ``"csr"``) and defaults to the input's own.
    Chunks are encoded by ``workers`` threads (default: one per CPU, at most
    32). ``atomic=True`` preserves the destination on failure; see
    :class:`Writer`.
    """
    return Store(encode(matrix, path, format=format, workers=workers, chunk_rows=chunk_rows,
                        page_size=page_size, options=options, atomic=atomic))


def compress(
    matrix,
    *,
    format: str | None = None,
    workers: int | None = None,
    chunk_rows: int = 4096,
    page_size: int = 4096,
    options: EncodeOptions | None = None,
) -> Store:
    """Encode a matrix into an in-memory stream and return a Store over it.

    Arguments follow :func:`write`. The encoded bytes stay in this process
    (pickling copies them); reads decode only what they select.
    """
    return Store(encode(matrix, None, format=format, workers=workers, chunk_rows=chunk_rows,
                        page_size=page_size, options=options))


def _restore_store(file, rows, cols, shape, dtype):
    store = Store.__new__(Store)
    store._init(file, rows, cols, shape, dtype, None)
    return store


def _length(rows: _Rows) -> int:
    return rows.stop - rows.start if isinstance(rows, slice) else len(rows)


def _is_full(key) -> bool:
    return isinstance(key, slice) and key == slice(None)


def _split_key(key, ndim: int):
    if not isinstance(key, tuple):
        key = (key,)
    if any(k is Ellipsis for k in key):
        if sum(k is Ellipsis for k in key) > 1:
            raise IndexError("an index can only have a single ellipsis")
        at = next(i for i, k in enumerate(key) if k is Ellipsis)
        fill = (slice(None),) * max(0, 2 - (len(key) - 1))
        key = key[:at] + fill + key[at + 1 :]
    if ndim > 2 and len(key) > 2 and all(_is_full(k) for k in key[2:]):
        key = key[:2]
    if len(key) > 2:
        raise IndexError("a Store takes at most a cell and a gene selector")
    if any(k is None for k in key):
        raise IndexError("a Store cannot insert new axes")
    return (*key, *(slice(None),) * (2 - len(key)))


def _positions(key, size: int) -> np.ndarray:
    if hasattr(key, "to_numpy") and not isinstance(key, np.ndarray):
        key = key.to_numpy()
    if isinstance(key, (int, np.integer)):
        key = operator.index(key)
    return normalize_rows(key, size)


def _contiguous(rows: np.ndarray) -> _Rows:
    # Ascending runs keep the contiguous read path and identity detection.
    if len(rows) and np.all(np.diff(rows.astype(np.int64)) == 1):
        return slice(int(rows[0]), int(rows[-1]) + 1)
    return rows


def _compose_rows(current: _Rows, key, size: int) -> _Rows:
    if isinstance(current, slice) and isinstance(key, slice):
        start, stop, step = key.indices(size)
        if step == 1:
            stop = max(start, stop)
            return slice(current.start + start, current.start + stop)
    positions = _positions(key, size)
    if isinstance(current, slice):
        rows = positions + np.uint64(current.start)
    else:
        rows = current[positions]
    return _contiguous(rows)


def _canonical_cols(cols: np.ndarray, n_source: int) -> np.ndarray | None:
    if len(cols) == n_source and np.array_equal(cols, np.arange(n_source)):
        return None
    return cols


def _expand_csr(decoded: CSR, inverse, present, shape) -> CSR:
    """Copy decoded unique-gene columns to every output column that uses them."""
    nnz = decoded.nnz
    indptr = decoded.indptr.astype(np.int64)
    indices, data = decoded.indices[:nnz], decoded.data[:nnz]
    counts = np.bincount(inverse)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    repeat = counts[indices]
    ends = np.concatenate(([0], np.cumsum(repeat)))
    first = np.repeat(ends[:-1], repeat)
    slot = np.repeat(starts[indices], repeat) + (np.arange(ends[-1]) - first)
    out_cols = present[order[slot]]
    out_data = np.repeat(data, repeat)
    out_indptr = ends[indptr]
    row_ids = np.repeat(np.arange(shape[0]), np.diff(out_indptr))
    perm = np.lexsort((out_cols, row_ids))
    index_dtype = decoded.indices.dtype
    return CSR(out_indptr.astype(index_dtype), out_cols[perm].astype(index_dtype),
               out_data[perm], shape)


def _flat_out(out, store: Store):
    if out is None:
        return None
    flat = (store._shape[0], math.prod(store._shape[1:]))
    if (
        not isinstance(out, np.ndarray)
        or out.size != math.prod(flat)
        or not out.flags.c_contiguous
        or not out.flags.aligned
        or not out.flags.writeable
        or out.dtype != store.dtype
    ):
        raise ValueError("output must be writable, C-contiguous and aligned with the selection's dtype and size")
    return out.reshape(flat)
