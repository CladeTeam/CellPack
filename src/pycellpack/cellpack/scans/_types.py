"""Python value objects for native scans metadata, tables and decoded parts."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

import numpy as np
from numpy.typing import DTypeLike

from ... import _cellpack
from .._arrays import byte_buffer, checked_dtype
from .._types import UNMAPPED as UNMAPPED  # noqa: PLC0414 - public scans re-export
from .._types import SparseRow as SparseRow  # noqa: PLC0414 - public scans re-export
from ..selection import normalize_rows

Bytes = np.ndarray | bytes | bytearray | memoryview
"""Encoded bytes: a contiguous uint8 vector or a buffer-protocol object."""
RowSelector = int | slice | Sequence[int] | Sequence[bool] | np.ndarray | None
"""Cells of a chunk: an index, slice, integer ids (any order, repeats) or mask."""

DEFAULT_MU_INDICES = DEFAULT_MU_VALUES = 0.6
DEFAULT_MAX_LANES = DEFAULT_LANE_MIN = 32
DEFAULT_MODE_SHARE = 0.5


class IdSpace(str, Enum):
    SOURCE = "source"
    CODED = "coded"


class RecordLayout(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"


class LayoutChoice(str, Enum):
    AUTO = "auto"
    SPARSE = "sparse"
    DENSE = "dense"


class ValueChoice(str, Enum):
    AUTO = "auto"
    RAW = "raw"
    DICT = "dict"
    MODES = "modes"


class Fill(str, Enum):
    ZERO = "zero"
    ASSUME_ZEROED = "assume_zeroed"


@dataclass(frozen=True)
class Thresholds:
    indices: float = DEFAULT_MU_INDICES
    values: float = DEFAULT_MU_VALUES


@dataclass(frozen=True)
class EncodeOptions:
    renumber: bool = True
    thresholds: Thresholds = field(default_factory=Thresholds)
    max_lanes: int = DEFAULT_MAX_LANES
    lane_min: int = DEFAULT_LANE_MIN
    column_modes: float | None = DEFAULT_MODE_SHARE
    layout: LayoutChoice | str = LayoutChoice.AUTO
    values: ValueChoice | str = ValueChoice.AUTO

    def _asdict(self):
        return {
            "renumber": self.renumber,
            "indices_threshold": self.thresholds.indices,
            "values_threshold": self.thresholds.values,
            "max_lanes": self.max_lanes,
            "lane_min": self.lane_min,
            "column_modes": self.column_modes,
            "layout": LayoutChoice(self.layout).value,
            "values": ValueChoice(self.values).value,
        }


def coded(space) -> bool:
    return IdSpace(space) is IdSpace.CODED


def chunk_rows(rows, size: int):
    """A chunk row selection at the native boundary, and its row count:
    None for all rows, ``(start, stop)`` for unit-step slices, else uint32
    ids (order and repeats kept)."""
    if rows is None:
        return None, size
    if isinstance(rows, np.ndarray) and rows.dtype == np.uint32 and rows.ndim == 1:
        if np.any(rows >= size):
            raise IndexError("row index out of bounds")
        return np.ascontiguousarray(rows), len(rows)
    if isinstance(rows, slice):
        start, stop, step = rows.indices(size)
        if step == 1:
            stop = max(start, stop)
            return (start, stop), stop - start
    indices = normalize_rows(rows, size)
    if np.any(indices > np.iinfo(np.uint32).max):
        raise IndexError("chunk row indices must fit uint32")
    return np.ascontiguousarray(indices, dtype=np.uint32), len(indices)


def output_dtype(dtype, out, default) -> np.dtype:
    """An explicit dtype must agree with caller-owned output storage."""
    if out is not None and not isinstance(out, np.ndarray):
        raise TypeError("out must be a NumPy array")
    if dtype is None:
        target = checked_dtype(default if out is None else out.dtype)
    else:
        target = checked_dtype(dtype)
    if out is not None and out.dtype != target:
        raise ValueError("explicit dtype differs from out.dtype")
    return target


def readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class PagedRecord:
    pages: list[np.ndarray]
    page_size: int
    start: int
    length: int

    def __post_init__(self):
        if (
            self.page_size <= 0
            or self.start < 0
            or self.start >= self.page_size
            or self.length < 0
        ):
            raise ValueError("invalid page size, start or record length")
        for page in self.pages:
            byte_buffer(page)


class Meta:
    """Validated metadata snapshot; records remain byte views over caller input.

    Chunk metadata stores value width, not signedness: ``source_dtype``
    says how stored values read (default: unsigned integers of the width)
    and becomes :attr:`dtype`, the dtype decoding yields by default.
    """

    def __init__(self, buffer: Bytes, *, source_dtype: DTypeLike | None = None):
        self.buffer = byte_buffer(buffer)
        self._handle = _cellpack.ScansMeta(self.buffer)
        self._info = self._handle.info()
        self.dtype = checked_dtype(
            source_dtype if source_dtype is not None else f"u{self.value_width}", self.value_width
        )

    @property
    def n_cells(self) -> int:
        return self._info["n_cells"]

    @property
    def n_dims(self) -> int:
        return self._info["n_dims"]

    @property
    def value_width(self) -> int:
        return self._info["value_width"]

    @property
    def index_width(self) -> int:
        return self._info["index_width"]

    @property
    def renumbered(self) -> bool:
        return self._info["renumbered"]

    @property
    def max_nnz(self) -> int:
        return self._info["max_nnz"]

    @property
    def max_elems(self) -> int:
        return self._info["max_elems"]

    @property
    def max_k(self) -> int:
        return self._info["max_k"]

    @property
    def nnz_total(self) -> int:
        return self._info["nnz_total"]

    @property
    def meta_len(self) -> int:
        return self._info["meta_len"]

    @property
    def records_start(self) -> int:
        return self._info["records_start"]

    @property
    def records_len(self) -> int:
        return self._info["records_len"]

    @property
    def chunk_len(self) -> int:
        return self._info["chunk_len"]

    @property
    def page_size(self) -> int:
        return self._info["page_size"]

    @property
    def decode_workspace_words(self) -> int:
        """uint64 workspace words decoding any record of this chunk needs."""
        return self._info["decode_workspace_words"]

    @property
    def has_modes(self) -> bool:
        return self._info["has_modes"]

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_cells, self.n_dims

    def _cell(self, cell):
        cell = operator.index(cell)
        if cell < 0:
            cell += self.n_cells
        if not 0 <= cell < self.n_cells:
            raise IndexError("cell out of bounds")
        return self._handle.cell(cell)

    def cell_nnz(self, cell) -> int:
        return self._cell(cell)[0]

    def cell_layout(self, cell) -> RecordLayout:
        return RecordLayout(self._cell(cell)[1])

    def cell_kind(self, cell) -> int:
        return self._cell(cell)[2]

    def record_range(self, cell) -> slice:
        return slice(*self._cell(cell)[3:])

    def records(self, buffer: Bytes | None = None) -> np.ndarray:
        b = self.buffer if buffer is None else byte_buffer(buffer)
        end = self.records_start + self.records_len
        if len(b) < end:
            raise ValueError("buffer does not contain the full records region")
        return b[self.records_start : end]

    def record(self, cell, *, records=None) -> np.ndarray:
        records = self.records() if records is None else byte_buffer(records)
        region = self.record_range(cell)
        if region.stop > len(records):
            raise ValueError("record extends beyond supplied records")
        return records[region]

    @cached_property
    def column_modes(self) -> np.ndarray | None:
        if not self._info["has_modes"]:
            return None
        result = np.empty(self.n_dims, dtype=self.dtype)
        self._handle.copy_modes(result)
        return readonly(result)

    def __repr__(self):
        return (
            f"Meta(shape={self.shape}, dtype={self.dtype}, page_size={self.page_size})"
        )


class DecodeTables:
    """Decoding lookup tables for one metadata model."""

    def __init__(self, meta: Meta):
        self.meta = meta
        self._handle = _cellpack.ScansDecodeTables(meta._handle)

    @cached_property
    def _maps(self):
        if not self.meta.renumbered:
            return None, None
        source = np.empty(self.meta.n_dims, np.uint32)
        coded = np.empty(self.meta.n_dims, np.uint32)
        self._handle.gene_maps(source, coded)
        return readonly(source), readonly(coded)

    @property
    def source_of_coded(self) -> np.ndarray | None:
        return self._maps[0]

    @property
    def coded_of_source(self) -> np.ndarray | None:
        return self._maps[1]


class EncodeTables:
    """Encoding lookup tables for one metadata model."""

    def __init__(self, meta: Meta):
        self.meta = meta
        self._handle = _cellpack.ScansEncodeTables(meta._handle)

    @cached_property
    def coded_of_source(self) -> np.ndarray | None:
        if not self.meta.renumbered:
            return None
        result = np.empty(self.meta.n_dims, np.uint32)
        self._handle.gene_map(result)
        return readonly(result)


@dataclass(frozen=True)
class Ranks:
    kind: str
    data: np.ndarray | None


class Dict:
    def __init__(self, parts: Parts):
        self._parts = parts
        self.k = parts._info[4]
        self.value_width = parts.meta.value_width

    @cached_property
    def entries(self) -> np.ndarray:
        result = np.empty(self.k, self._parts.dtype)
        self._parts._handle.copy_component("entries", result)
        return readonly(result)

    @cached_property
    def ranks(self) -> Ranks:
        kind = self._parts._info[5]
        if kind == "one":
            return Ranks(kind, None)
        result = np.empty(self._parts.elements, dtype=kind)
        self._parts._handle.copy_component("ranks", result)
        return Ranks(kind, readonly(result))

    def cast_into(self, dtype: DTypeLike | None = None, *, out: np.ndarray | None = None) -> np.ndarray:
        target = output_dtype(dtype, out, np.float32)
        if out is None:
            out = np.empty(self.k, dtype=target)
        if out.shape != (self.k,):
            raise ValueError("dictionary output shape mismatch")
        self._parts._handle.cast_dictionary(self._parts.dtype.name, out)
        return out


class ModeDict(Dict):
    @property
    def modes(self) -> np.ndarray | None:
        return self._parts.meta.column_modes


@dataclass(frozen=True)
class Values:
    kind: str
    raw: np.ndarray | None = None
    dictionary: Dict | None = None


class Parts:
    """An owned decoded object with stable private record and scratch storage.

    Component properties return read-only NumPy snapshots. Mapping reads the
    private native data. expand_values(out=) writes into caller-owned storage.
    """

    def __init__(self, handle: _cellpack.ScansParts, meta: Meta, space: IdSpace | str):
        self._handle = handle
        self.meta = meta
        self.dtype = meta.dtype
        self.space = IdSpace(space)
        self._info = handle.info()

    @property
    def nnz(self) -> int:
        return self._info[0]

    @property
    def elements(self) -> int:
        return self._info[1]

    @property
    def layout(self) -> RecordLayout:
        return RecordLayout(self._info[2])

    @cached_property
    def ids(self) -> np.ndarray:
        result = np.empty(self.elements, np.uint32)
        self._handle.copy_component("ids", result)
        return readonly(result)

    @cached_property
    def values(self) -> Values:
        kind = self._info[3]
        if kind == "raw":
            return Values(kind, raw=readonly(self.expand_values()))
        return Values(kind, dictionary=(ModeDict if kind == "modes" else Dict)(self))

    def expand_values(self, *, out=None) -> np.ndarray:
        if out is None:
            out = np.empty(self.elements, self.dtype)
        if out.dtype != self.dtype or out.shape != (self.elements,):
            raise ValueError("value output dtype/shape mismatch")
        self._handle.copy_component("values", out)
        return out
