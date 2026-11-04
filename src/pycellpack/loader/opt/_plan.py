"""Compiled cache plans and the serial page reference.

A `Plan` is what the compiler decided before a run: which pages and rows to
hold, where each residency starts and ends, and which cache frames it takes, so
a run follows it exactly instead of deciding anything. `PageTrace` computes the
serial page optimum of a page sequence or of a data source's accesses, the
reference a plan's hits are compared with.
"""

from __future__ import annotations

import numpy as np

from ... import _cellpack
from ._stats import CacheSummary, OnlineStats, PlanSummary


class Plan:
    """A compiled cache plan (see `Loader.compile`), bound to its loader's
    data source, schemas and slots, and to its access sequence. Pass it to
    `Loader.start`. A plan pickles (by its file bytes); `online_stats`
    holds the compilation's statistics when it was compiled as an online
    run would."""

    def __init__(self, handle, online_stats: OnlineStats | None = None):
        """Wraps a native plan; `online_stats` is kept only for plans compiled
        as an online run would."""
        self._handle = handle
        self.online_stats = online_stats

    @classmethod
    def open(cls, path) -> Plan:
        """A plan read from a `CPKPLAN1` file; its `summary` is counted when a
        loader first verifies it."""
        return cls(_cellpack.Plan.open(path))

    def write(self, path) -> None:
        """Writes the plan file (the `CPKPLAN1` v1 form, with its cost model
        and fingerprints)."""
        self._handle.write(path)

    @classmethod
    def from_bytes(cls, data) -> Plan:
        """A plan read from the bytes of a plan file."""
        return cls(_cellpack.Plan.from_bytes(bytes(data)))

    def to_bytes(self) -> bytes:
        """The plan file's bytes, for another process or for storage."""
        return self._handle.to_bytes()

    @property
    def summary(self) -> PlanSummary:
        """The serial replay's counts; for a plan read from a file or bytes,
        counted when a loader first verifies it (start or replay)."""
        return PlanSummary.from_dict(self._handle.summary)

    @property
    def distance(self) -> int:
        """The reuse gap in accesses."""
        return self._handle.distance

    @property
    def reuse_gap(self) -> int:
        """The reuse gap in slots, independent of the runtime window."""
        return self._handle.reuse_gap

    @property
    def cache_bytes(self) -> int:
        """The cache budget the plan was compiled for, in bytes."""
        return self._handle.cache_bytes

    @property
    def cache_frames(self) -> int:
        """Cache frames the plan's budget buys (frame ids lie below it)."""
        return self._handle.cache_frames

    @property
    def schemas(self) -> int:
        """Schemas per position the plan was compiled for (the loader's)."""
        return self._handle.schemas

    @property
    def slot_rows(self) -> int:
        """Positions per slot the plan's sequences are expressed in."""
        return self._handle.slot_rows

    def __len__(self) -> int:
        """Accesses of the sequence the plan is bound to."""
        return len(self._handle)

    def __repr__(self) -> str:
        return (f"Plan(accesses={len(self)}, schemas={self.schemas}, slot_rows={self.slot_rows}, "
                f"cache_bytes={self.cache_bytes}, reuse_gap={self.reuse_gap})")

    def residencies(self) -> dict[str, np.ndarray]:
        """The residencies in admission order as arrays: `kind` (0 a page, 1
        a row), `key` (global page id, or cell), `start` (the admitting
        access), `end` (the last use), and the frames of residency `i` as
        `frames[offsets[i]:offsets[i + 1]]`."""
        return self._handle.residencies()


class PageTrace:
    """A page trace over a page sequence or a data source's accesses;
    `optimum` gives what a cache of a given size could hit at best."""

    def __init__(self, handle):
        """Wraps a native trace over an already expanded page sequence."""
        self._handle = handle

    @classmethod
    def from_pages(cls, pages, *, universe=None) -> PageTrace:
        """A trace over a stored page sequence. `universe` is the page id
        space's size (default: the largest id plus one); ids must fit uint32.
        Raises ValueError for a sequence that is not one-dimensional or holds
        a negative or too large id."""
        a = np.asarray(pages)
        if a.ndim != 1 or (a.size and (a.dtype.kind not in "iu" or a.min() < 0 or a.max() > 0xFFFFFFFF)):
            raise ValueError("pages must be nonnegative uint32 ids")
        a = np.ascontiguousarray(a, dtype=np.uint32)
        universe = (int(a.max()) + 1 if a.size else 0) if universe is None else universe
        return cls(_cellpack.PageTrace(a, universe))

    @classmethod
    def from_source(cls, source, accesses) -> PageTrace:
        """The pages `accesses` (global cell ids of `source`) read."""
        from .._accesses import sequence

        return cls(_cellpack.PageTrace.from_source(source._handle, sequence(accesses, counts=source._row_counts())))

    @property
    def page_uses(self) -> int:
        """Pages the sequence touches, repeats included."""
        return self._handle.page_uses

    @property
    def distinct_pages(self) -> int:
        """Pages of the id space the sequence reads at least once."""
        return self._handle.distinct_pages

    def optimum(self, capacity_pages) -> CacheSummary:
        """The hits and misses of a page cache of `capacity_pages` under the
        optimal offline policy (Belady)."""
        return CacheSummary(**self._handle.optimum(capacity_pages))
