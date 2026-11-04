"""Matrix streams: the low-level file handle, prepared reads and the writer.

Writers normalize NumPy/SciPy inputs to the contiguous, canonical layouts the
native encoder borrows; output buffers passed as ``out=`` must already match.
Arrays borrowed by native calls must not be mutated or resized until the call
returns. Output arrays must not be accessed concurrently with their native
writes.
"""

from __future__ import annotations

import errno
import operator
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from .. import _cellpack
from .._workers import resolve_workers
from ._arrays import checked_dtype, input_matrix
from ._types import CSR
from .format import ChunkEntry
from .scans import EncodeOptions, Meta
from .selection import as_gene_map, normalize_rows

if TYPE_CHECKING:
    from typing_extensions import Self

    from .._cellpack import GeneMap, ReadWorkspace

__all__ = ["File", "ReadOptions", "ReadSelection", "ReadSummary", "Writer", "encode"]

_INDEX_DTYPES = tuple(map(np.dtype, ("int32", "int64", "uint32", "uint64")))


@dataclass(frozen=True)
class ReadOptions:
    """Physical read policy; request_cost_bytes is an explicit Auto heuristic."""

    strategy: str = "auto"
    max_gap_bytes: int = 0
    request_cost_bytes: int = 4096


@dataclass(frozen=True)
class ReadSummary:
    """Payload I/O planned for one execution, excluding metadata preparation."""

    record_bytes: int
    read_bytes: int
    requests: int
    whole_chunk_bytes: int


class ReadSelection:
    """Read metadata of a cell selection, reusable across output arrays.

    ``dtype`` is the output dtype values are converted to. ``capacity_nnz``
    sizes CSR outputs; dropped genes can reduce the nnz a read returns.
    Reads split the selected chunks over ``workers`` threads (default: one
    per CPU, at most 32); a ``workspace`` keeps their scratch between reads.
    """

    __slots__ = ("_dtype", "_handle")

    def __init__(self, handle: _cellpack.ReadSelection, dtype: np.dtype):
        self._handle = handle
        self._dtype = dtype

    @property
    def io_summary(self) -> ReadSummary:
        return ReadSummary(**self._handle.io_summary())

    @property
    def shape(self) -> tuple[int, int]:
        return self._handle.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def capacity_nnz(self) -> int:
        return self._handle.capacity_nnz

    def read_dense(
        self,
        *,
        out: np.ndarray | None = None,
        workspace: ReadWorkspace | None = None,
        workers: int | None = None,
    ) -> np.ndarray:
        if out is None:
            out = np.empty(self.shape, self._dtype)
        elif not isinstance(out, np.ndarray) or out.dtype != self._dtype or out.shape != self.shape:
            raise ValueError("output dtype/shape differs from the selection")
        self._handle.read_dense_into(out, workspace, resolve_workers(workers))
        return out

    def read_csr(
        self,
        *,
        out: CSR | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        index_dtype=np.int64,
        workspace: ReadWorkspace | None = None,
        workers: int | None = None,
    ) -> CSR:
        """Canonical CSR; caller ``out`` buffers need ``capacity_nnz`` entries.

        indptr may be int32/int64/uint32/uint64 and indices also uint16.
        """
        if out is None:
            index_dtype = np.dtype(index_dtype)
            if index_dtype not in _INDEX_DTYPES:
                raise ValueError("index_dtype must be int32, int64, uint32 or uint64")
            indptr = np.empty(self.shape[0] + 1, index_dtype)
            indices = np.empty(self.capacity_nnz, index_dtype)
            data = np.empty(self.capacity_nnz, self._dtype)
        else:
            if isinstance(out, CSR):
                if out.shape != self.shape:
                    raise ValueError("CSR output shape differs from the selection")
                out = (out.indptr, out.indices, out.data)
            indptr, indices, data = out
            if not isinstance(data, np.ndarray) or data.dtype != self._dtype:
                raise ValueError("output dtype differs from the selection")
        nnz = self._handle.read_csr_into(indptr, indices, data, workspace, resolve_workers(workers))
        return CSR(indptr, indices[:nnz], data[:nnz], self.shape)


def _anonymous_fd() -> int:
    create = getattr(os, "memfd_create", None)
    if create is not None:
        try:
            return create("cellpack", getattr(os, "MFD_CLOEXEC", 1))
        except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EPERM, errno.EACCES):
                raise
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "memfd_create"):
        fd = libc.memfd_create(b"cellpack", 1)  # MFD_CLOEXEC
        if fd >= 0:
            return fd
        error = ctypes.get_errno()
        if error not in (errno.ENOSYS, errno.EPERM, errno.EACCES):
            raise OSError(error, os.strerror(error))
    # Unnamed files behave the same where memfd is unavailable.
    for directory in ("/dev/shm", tempfile.gettempdir()):
        try:
            return os.open(directory, os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC, 0o600)
        except OSError:
            continue
    raise OSError("no anonymous file is available for an in-memory stream")


class _Memory:
    """An anonymous in-memory file (Linux memfd) holding one encoded stream.

    Its path names the descriptor through this process's /proc entry, so the
    native reader and loader open it like any file, while other processes
    fail to open it instead of reaching an unrelated descriptor.
    """

    def __init__(self):
        self.fd = _anonymous_fd()
        self.path = Path(f"/proc/{os.getpid()}/fd/{self.fd}")

    def __del__(self):
        fd = getattr(self, "fd", None)
        if fd is not None:
            os.close(fd)


def _rows(rows, size: int):
    """Contiguous selections stay a native range; others become uint64 ids."""
    if rows is None:
        return (0, size)
    if isinstance(rows, slice):
        start, stop, step = rows.indices(size)
        if step == 1:
            return (start, max(start, stop))
    return normalize_rows(rows, size)


def _options(read_options: ReadOptions | None):
    return None if read_options is None else asdict(read_options)


class File:
    """The low-level handle of one matrix stream: a ``.cpk`` file, a bounded
    region of another file (such as an H5CP dataset), or an in-memory stream
    (:meth:`from_bytes`).

    A File exposes the stream's layout, chunk I/O and prepared reads
    (:meth:`prepare_read`); :class:`~pycellpack.cellpack.Store` is the array
    API over it. A stream holds the matrix only; columns are selected by
    position and cellpack keeps no names.
    """

    def __init__(
        self,
        path: str | PathLike,
        *,
        offset: int = 0,
        length: int | None = None,
        _memory: _Memory | None = None,
    ):
        self._memory = _memory
        self.path = _memory.path if _memory is not None else Path(path).resolve()
        self.offset = offset
        self.length = length
        self._handle = _cellpack.File(self.path, offset, length)
        self._identity = self._handle.identity

    @classmethod
    def from_bytes(cls, buffer) -> File:
        """Open an encoded stream held in memory (copied into an anonymous file)."""
        memory = _Memory()
        with open(memory.path, "wb") as stream:
            stream.write(memoryview(buffer).cast("B"))
        return cls(memory.path, _memory=memory)

    @property
    def identity(self) -> tuple[int, int]:
        """Device and inode of the opened descriptor, fixed at opening."""
        return self._identity

    @property
    def in_memory(self) -> bool:
        """True for streams held in anonymous memory rather than on disk."""
        return self._memory is not None

    def to_bytes(self) -> bytes:
        """Return the whole region; reject a path replaced since opening."""
        with open(self.path, "rb") as stream:
            status = os.fstat(stream.fileno())
            if (status.st_dev, status.st_ino) != self._identity:
                raise OSError("stream path was replaced since opening")
            length = self.length
            if length is None:
                length = status.st_size - self.offset
            if length < 0:
                raise OSError("stream ended before its region")
            stream.seek(self.offset)
            data = stream.read(length)
        if len(data) != length:
            raise OSError("stream ended before its region")
        return data

    def __repr__(self) -> str:
        where = "memory" if self.in_memory else str(self.path)
        region = "" if self.offset == 0 and self.length is None else (
            f", offset={self.offset}, length={self.length}"
        )
        return (
            f"File({where!r}{region}, shape={self.shape}, dtype={self.dtype}, "
            f"format={self.format!r})"
        )

    def is_current(self) -> bool:
        """Whether ``path`` still names the file this stream was opened from.

        Reads always use the opened file; a path replaced since then (for
        example by an atomic rewrite) no longer refers to the same bytes.
        """
        try:
            status = self.path.stat()
        except FileNotFoundError:
            return False
        return (status.st_dev, status.st_ino) == self._identity

    def __reduce__(self):
        # The native handle is per process; a copy reopens the same region,
        # or carries the bytes of an in-memory stream.
        if self.in_memory:
            return (File.from_bytes, (self.to_bytes(),))
        return (_reopen, (str(self.path), self.offset, self.length, self._identity))

    @property
    def shape(self) -> tuple[int, int]:
        return self._handle.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._handle.dtype)

    @property
    def format(self) -> str:
        """Stored layout the matrix was written from: ``"dense"`` or ``"csr"``."""
        return self._handle.format

    @property
    def page_size(self) -> int:
        return self._handle.page_size

    def prepare_read(
        self,
        rows=None,
        *,
        gene_map: GeneMap | np.ndarray | dict | None = None,
        n_genes: int | None = None,
        dtype=None,
        read_options: ReadOptions | None = None,
    ) -> ReadSelection:
        """Prepare a reusable read of ``rows`` (a slice, integer ids in any
        order with repeats, or a boolean mask; default all cells).

        ``gene_map`` places source columns into a panel of ``n_genes``
        columns (see :class:`~pycellpack.cellpack.GeneMap`); ``dtype`` is
        the output dtype (default: stored). Only metadata is read here.
        """
        mapping = as_gene_map(gene_map, self.shape[1], n_genes)
        target = self.dtype if dtype is None else checked_dtype(dtype)
        handle = self._handle.prepare_read(
            _rows(rows, self.shape[0]), mapping, _options(read_options)
        )
        return ReadSelection(handle, target)

    @property
    def chunks(self) -> list[ChunkEntry]:
        """The chunk directory, one entry per chunk in row order."""
        return [ChunkEntry(*row) for row in self._handle.chunk_info()]

    def read_chunk(self, chunk: int, *, out: np.ndarray | None = None,
                   metadata_only: bool = False) -> np.ndarray:
        """Read one chunk's bytes (or only its metadata) into ``out``."""
        chunk = operator.index(chunk)
        count = self._handle.n_chunks
        chunk = chunk + count if chunk < 0 else chunk
        if not 0 <= chunk < count:
            raise IndexError("chunk out of range")
        entry = ChunkEntry(*self._handle.chunk_info_one(chunk))
        size = entry.meta_length if metadata_only else entry.length
        if out is None:
            out = np.empty(size, np.uint8)
        self._handle.read_chunk_into(chunk, out, metadata_only)
        return out

    def chunk_meta(self, chunk: int) -> Meta:
        """Parsed metadata of one chunk, read in the stored dtype."""
        return Meta(self.read_chunk(chunk, metadata_only=True), source_dtype=self.dtype)


def _reopen(path, offset, length, identity=None):
    file = File(path, offset=offset, length=length)
    if identity is not None and file.identity != identity:
        raise OSError("stream path was replaced since pickling")
    return file


class Writer:
    """Streaming stream writer. A stream becomes readable after finish().

    ``append`` takes dense or sparse rows whatever the stored ``format``;
    the native writer converts them. Whole chunks within one ``append`` are
    encoded by ``workers`` threads (default: one per CPU, at most 32); the
    file is identical to a serial write. Append many rows at a time to let
    chunks encode concurrently.

    With ``atomic=True``, a sibling temporary file is published only after
    finish succeeds; abort leaves the destination untouched. This requires
    a regular filesystem path and does not guarantee power-loss durability.
    New atomic files are private (mode 0600); existing permission bits are kept.
    """

    def __init__(
        self,
        path: str | PathLike,
        n_cols: int,
        dtype,
        *,
        format: str = "csr",
        chunk_rows: int = 4096,
        page_size: int = 4096,
        options: EncodeOptions | None = None,
        workers: int | None = None,
        atomic: bool = False,
    ):
        dtype = np.dtype(dtype)
        if not dtype.isnative:
            raise ValueError("native little-endian dtype required")
        self._dtype = dtype
        self._n_cols = n_cols
        self._closed = True
        self._owner_pid = os.getpid()
        self._temporary = None
        self._destination = None
        self._publish_mode = None
        try:
            if atomic:
                self._destination = Path(path).resolve()
                try:
                    status = self._destination.stat()
                except FileNotFoundError:
                    status = None
                if status is not None and not stat.S_ISREG(status.st_mode):
                    raise ValueError("atomic writes require a regular file destination")
                fd, name = tempfile.mkstemp(
                    prefix=f".{self._destination.name}.", suffix=".tmp",
                    dir=self._destination.parent,
                )
                self._temporary = Path(name)
                os.close(fd)
                if status is not None:
                    self._publish_mode = stat.S_IMODE(status.st_mode)
                path = self._temporary
            self._handle = _cellpack.Writer(
                path, n_cols, dtype.name, format, chunk_rows, page_size,
                None if options is None else options._asdict(), resolve_workers(workers),
            )
            self._closed = False
        except BaseException:
            self._discard_temporary()
            raise

    def append(self, matrix) -> None:
        """Append rows from NumPy, SciPy sparse, CSR or SparseRow input.

        The dtype and column count must match the writer's.
        """
        self._check_process()
        data, indptr, indices, n_cols, _ = input_matrix(matrix)
        if n_cols != self._n_cols:
            raise ValueError(f"input has {n_cols} columns but the writer has {self._n_cols}")
        if data.dtype != self._dtype:
            raise ValueError(f"input dtype {data.dtype} differs from writer dtype {self._dtype}")
        if indptr is None:
            self._handle.append_dense(data)
        else:
            self._handle.append_csr(indptr, cast(np.ndarray, indices), data)

    def end_chunk(self) -> None:
        """End the current chunk early; the next row starts a new one."""
        self._check_process()
        self._handle.end_chunk()

    def finish(self) -> None:
        self._check_process()
        if not self._closed:
            try:
                self._handle.finish()
                if self._temporary is not None:
                    if self._publish_mode is not None:
                        self._temporary.chmod(self._publish_mode)
                    os.replace(self._temporary, cast(Path, self._destination))
                    self._temporary = None
            finally:
                self._closed = True
                try:
                    self._handle.abort()
                finally:
                    self._discard_temporary()

    def abort(self) -> None:
        try:
            if not self._closed:
                self._handle.abort()
        finally:
            self._closed = True
            self._discard_temporary()

    def _discard_temporary(self) -> None:
        if self._temporary is not None:
            if os.getpid() == self._owner_pid:
                self._temporary.unlink(missing_ok=True)
            self._temporary = None

    def _check_process(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("writer must be used in the process that created it")

    def __del__(self):
        if getattr(self, "_temporary", None) is not None:
            self.abort()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish()
        else:
            self.abort()
        return False


def encode(
    matrix,
    path: str | PathLike | None = None,
    *,
    format: str | None = None,
    workers: int | None = None,
    chunk_rows: int = 4096,
    page_size: int = 4096,
    options: EncodeOptions | None = None,
    atomic: bool = False,
) -> File:
    """Encode a whole matrix to ``path`` (or anonymous memory) and open it.

    ``format`` defaults to the input's own layout.
    """
    data, indptr, indices, n_cols, n_rows = input_matrix(matrix)
    if format is None:
        format = "dense" if indptr is None else "csr"
    if format not in ("dense", "csr"):
        raise ValueError("format must be dense or csr")
    memory = None
    if path is None:
        if atomic:
            raise ValueError("atomic replacement applies to disk files, not in-memory streams")
        memory = _Memory()
        path = memory.path
    with Writer(path, n_cols, data.dtype, format=format, chunk_rows=chunk_rows,
                page_size=page_size, options=options, workers=workers,
                atomic=atomic) as writer:
        writer.append(data if indptr is None else CSR(indptr, cast(np.ndarray, indices), data, (n_rows, n_cols)))
    return File(path, _memory=memory)
