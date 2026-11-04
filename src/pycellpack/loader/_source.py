"""The data source: every input matrix with its explicit column -> standard id map.

Global cell ids number the matrices' cells in registration order, and standard
ids name the caller's panel; every matrix carries its own `ids`, one per stored
column. The source owns paths, regions and those maps, never cell data: it
reads metadata only, and it infers, renames and aligns nothing. Native loaders
and plans keep the source they were built with, so `add` swaps this object's
state for a new snapshot instead of mutating the old one.
"""

import mmap
import os

import numpy as np

from .. import _cellpack

NONE = 0xFFFFFFFF


def _std_map(ids) -> np.ndarray:
    """A matrix's column -> standard id map as uint32; -1 or 0xFFFFFFFF
    drops the column."""
    values = np.asarray(ids)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise TypeError("ids must be a one-dimensional integer array, one id per column")
    # Compare signed, where -1 and 0xFFFFFFFF are one range check and int64 holds
    # both drop markers; a uint64 map cannot be negative and is checked as it is.
    values = values.astype(np.int64, copy=False) if values.dtype != np.uint64 else values
    if values.size and (values.min() < -1 or values.max() > NONE):
        raise ValueError("ids must be uint32 standard ids or -1")
    return np.where(values == -1, NONE, values).astype(np.uint32)


def _entries(matrices) -> list[tuple]:
    """The `(source, ids)` pairs of a source list."""
    entries = list(matrices)
    for entry in entries:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise TypeError(
                "each matrix is a (source, ids) pair: ids[c] is the standard id "
                "of its stored column c"
            )
    return entries


def _store_cells(store):
    """A Store source's file and selected cells (None for all). Its matrix's
    ids address the stored columns, so a Store contributes cells only."""
    if store.ndim > 2:
        raise ValueError("loader sources need two-dimensional stores")
    if store.genes is not None:
        raise TypeError("a Store source selects cells only; its ids map the stored columns")
    return store.file, None if store.selects_all_cells else np.ascontiguousarray(store.cells, dtype=np.uint64)


def direct_io_supported(paths, page_size) -> bool:
    """Whether every file's filesystem serves O_DIRECT reads of one aligned
    page of `page_size` bytes, the loader's read unit (one probe per
    filesystem). Files that cannot be inspected are left to the native open,
    which reports them."""
    flag = getattr(os, "O_DIRECT", None)
    if flag is None:
        return False
    probed = set()
    page = mmap.mmap(-1, page_size)  # page-aligned, as the loader's frames
    try:
        for path in paths:
            try:
                device = os.stat(path).st_dev
            except OSError:
                continue
            if device in probed:
                continue
            probed.add(device)
            try:
                fd = os.open(path, os.O_RDONLY | flag)
            except OSError:
                return False
            try:
                os.preadv(fd, [page], 0)
            except OSError:
                return False
            finally:
                os.close(fd)
    finally:
        page.close()
    return True


class DataSource:
    """Every input matrix, numbered by global cell id in registration order.

    `matrices` is a sequence of `(source, ids)` pairs. A source is a path, a
    `cellpack.File` region or a `cellpack.Store` selecting cells; `ids[c]` is
    the standard id (below `vocab`) of the matrix's stored column `c`, and -1
    or `0xFFFFFFFF` drops the column. The ids belong to the matrix and are
    always given by the caller: the source infers, names and aligns nothing.
    A Store's cell selection defines which cells it contributes: global cell
    ids then number the selected cells, and runs, plans and traces translate
    them to stored cells natively as they read the sequence. Output schemas
    name the standard ids of their output dims.

    Data reads use O_DIRECT unless `direct_io=False`; with `None` (the
    default) the source falls back to buffered reads when a file's
    filesystem does not support it (`direct_io` reports the choice).
    Metadata reads stay buffered. Sources pickle (in-memory streams do not).
    `paths` lists the files of the registered matrices, in registration order.
    """

    def __init__(self, matrices, vocab, *, page_size=None, direct_io=None):
        """Registers `matrices` in order and builds the native catalog. A
        `Store` source is reduced to its file and selected cells, a `File` to
        its region (path, offset, length) and anything else must be a path.
        The page size defaults to the first matrix's. With `direct_io=None`
        the choice follows `direct_io_supported` over the filesystems of the
        source files. Raises TypeError for an unusable source or ids and
        ValueError for ids outside `[-1, 0xFFFFFFFF]`."""
        from ..cellpack import File, Store

        matrices = _entries(matrices)
        if isinstance(vocab, bool) or not isinstance(vocab, (int, np.integer)) or vocab <= 0:
            raise TypeError("vocab must be the positive number of standard ids")
        regions, cells = [], []
        for source, ids in matrices:
            selected = None
            if isinstance(source, Store):
                source, selected = _store_cells(source)
            if isinstance(source, File):
                if source.in_memory:
                    raise TypeError("loader sources are files; an in-memory stream cannot be loaded")
                region = (str(source.path), source.offset, source.length)
            else:
                region = (os.fspath(source), 0, None)
            regions.append((*region, _std_map(ids)))
            cells.append(selected)
        self._matrices, self._page_size, self._direct_io = matrices, page_size, direct_io
        if page_size is None:
            first = matrices[0][0] if matrices else None
            first = first.file if isinstance(first, Store) else first
            page_size = 4096 if first is None else (first if isinstance(first, File) else File(first)).page_size
        paths = [r[0] for r in regions]
        self._direct = direct_io_supported(paths, page_size) if direct_io is None else bool(direct_io)
        self._handle = _cellpack.DataSource(regions, int(vocab), page_size=page_size, direct_io=self._direct,
                                            cells=cells if any(c is not None for c in cells) else None)
        self.paths = tuple(paths)

    def __reduce__(self):
        """Pickles the (source, ids) pairs and the resolved page size and
        direct-I/O choice; the catalog is rebuilt on the other side."""
        return _source, (self._matrices, self.vocab, self._page_size, self._direct_io)

    def __repr__(self) -> str:
        return (f"DataSource(n_matrices={self.n_matrices}, n_cells={self.n_cells}, vocab={self.vocab}, "
                f"page_size={self.page_size}, direct_io={self.direct_io})")

    def add(self, source, ids) -> int:
        """Register a matrix with its column -> standard id map and return its
        first global cell id. Existing loaders keep the source they were
        built with."""
        first = self.n_cells
        updated = DataSource(self._matrices + [(source, ids)], self.vocab,
                             page_size=self._page_size, direct_io=self._direct_io)
        self.__dict__.update(updated.__dict__)
        return first

    @property
    def n_cells(self) -> int:
        """Global cells: every stored cell, or the cells Store views select."""
        return self._handle.n_cells

    @property
    def vocab(self) -> int:
        """Size of the standard id space: ids are ``0..vocab``."""
        return self._handle.vocab

    @property
    def n_pages(self) -> int:
        """Pages of every file together, in the global page numbering plans use."""
        return self._handle.n_pages

    @property
    def fingerprint(self) -> int:
        """Stable identity of the page layout and cell numbering; offline plans
        are bound to it, and output column maps do not take part."""
        return self._handle.fingerprint

    @property
    def mapping_fingerprint(self) -> int:
        """Identity including the standard id maps and the chunk metadata."""
        return self._handle.mapping_fingerprint

    def matrix_cells(self, matrix: int):
        """The first global cell of a matrix and its (selected) cell count, or
        None for a matrix the source does not have."""
        return self._handle.matrix_cells(matrix)

    def _row_counts(self) -> tuple[int, ...]:
        """Dataset view lengths from this registration snapshot."""
        counts = []
        for matrix in range(self.n_matrices):
            span = self.matrix_cells(matrix)
            assert span is not None
            counts.append(span[1])
        return tuple(counts)

    @property
    def matrix_dtypes(self):
        """Stored element type of each matrix, in registration order."""
        return tuple(np.dtype(d) for d in self._handle.matrix_dtypes)

    @property
    def n_matrices(self):
        """Matrices registered, each a contiguous range of global cell ids."""
        return self._handle.n_matrices

    @property
    def page_size(self):
        """Page size shared by every file of the source, in bytes."""
        return self._handle.page_size

    @property
    def direct_io(self) -> bool:
        """Whether data reads use O_DIRECT."""
        return self._direct

    @property
    def max_cell_pages(self):
        """Largest number of pages one cell spans."""
        return self._handle.max_cell_pages

    def stored_cells(self, accesses) -> np.ndarray:
        """The stored cells of global cell ids (an array or `Accesses`)."""
        from ._accesses import sequence

        return self._handle.stored_cells(sequence(accesses, counts=self._row_counts()))

    def cell_pages(self, cell):
        """Borrowed fd, aligned byte offset and page count of a global cell
        (negative cell ids count from the end); retain this source."""
        return self._handle.cell_pages(self._cell(cell))

    def cell_bytes(self, cell):
        """Borrowed fd, exact offset and byte length of a global cell
        (negative cell ids count from the end); retain this source."""
        return self._handle.cell_bytes(self._cell(cell))

    def _cell(self, cell) -> int:
        """A global cell id as an unsigned index, accepting negative ids from
        the end. Raises IndexError outside the source."""
        cell = int(cell)
        if not -self.n_cells <= cell < self.n_cells:
            raise IndexError("cell outside the data source")
        return cell % self.n_cells


def _source(matrices, vocab, page_size, direct_io):
    """The unpickling half of `DataSource.__reduce__`."""
    return DataSource(matrices, vocab, page_size=page_size, direct_io=direct_io)
