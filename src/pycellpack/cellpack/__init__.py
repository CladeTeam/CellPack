"""CellPack storage and composable codecs, independent of loader orchestration.

:class:`Store` is the array API: lazy indexing, gene maps and dtype
conversion, decoded on request. :class:`File` is the low-level stream handle
(prepared reads, chunk I/O, bytes); :class:`Writer` streams rows into a file.
"""

from .._cellpack import GeneMap, ReadWorkspace
from .._workers import default_workers
from . import format, scans, selection
from ._file import File, ReadOptions, ReadSelection, ReadSummary, Writer
from ._store import Store, compress, read, write
from ._types import CSR

__all__ = [
    "CSR",
    "File",
    "GeneMap",
    "ReadOptions",
    "ReadSelection",
    "ReadSummary",
    "ReadWorkspace",
    "Store",
    "Writer",
    "compress",
    "default_workers",
    "format",
    "read",
    "scans",
    "selection",
    "write",
]
