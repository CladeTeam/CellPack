"""CellPack superblock and chunk directory parsing."""

from __future__ import annotations

from typing import NamedTuple

from .. import _cellpack
from ._arrays import byte_buffer

__all__ = ["ENTRY_LEN", "SUPERBLOCK_LEN", "ChunkEntry", "parse_directory", "parse_superblock"]

SUPERBLOCK_LEN: int = _cellpack.SUPERBLOCK_LEN
ENTRY_LEN: int = _cellpack.ENTRY_LEN


class ChunkEntry(NamedTuple):
    """One chunk of the directory: its first row, byte offset in the
    stream, byte length, metadata length and row count."""

    first_row: int
    offset: int
    length: int
    meta_length: int
    n_rows: int


def parse_superblock(buffer) -> dict:
    """Fields of a stream's superblock (its first ``SUPERBLOCK_LEN`` bytes)."""
    return _cellpack.scans_parse_superblock(byte_buffer(buffer))


def parse_directory(superblock, directory) -> list[ChunkEntry]:
    """The chunk directory: ``n_chunks`` entries of ``ENTRY_LEN`` bytes."""
    rows = _cellpack.scans_parse_directory(byte_buffer(superblock), byte_buffer(directory))
    return [ChunkEntry(*row) for row in rows]
