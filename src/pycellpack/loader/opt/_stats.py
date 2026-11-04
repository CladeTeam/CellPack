"""Plan summaries, online compilation statistics and the serial page
optimum: frozen records built from the native counts, field for field.

Seconds are modeled from a cost model unless a record names a wall time.
`PlanSummary` replays a plan serially and splits the work into quarters, so
steady-state fitting can leave out the first quarter's initial reads and
admissions; `OnlineStats` adds what the compiler itself did and cost;
`CacheSummary` is the page-only optimum a plan's hits are compared with.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PlanSummary:
    """A plan's serial replay: page uses and hits; read requests and bytes;
    residencies and extra admission reads (`block_reads`); decoded records,
    their elements and those in a cold chunk; row hits and admissions with
    their elements and payload bytes; per schema (in loader order) the
    elements mapped, rows filled, process and partial calls; elements per
    conversion class; the most cache bytes held (serially, and reserved from
    each admission's reuse gap); modeled core-seconds (`worker_seconds`).
    Per quarter of the sequence (one epoch of a 4-epoch run): page uses,
    hits, requests, modeled seconds and each resource's service seconds
    (requests, bytes, cores, memory: the largest binds). Seconds are zero
    without a cost model."""

    accesses: int
    page_uses: int
    page_hits: int
    requests: int
    read_bytes: int
    page_residencies: int
    row_residencies: int
    block_reads: int
    decoded_rows: int
    decoded_elements: int
    cold_rows: int
    row_hits: int
    row_elements: int
    row_bytes: int
    admitted_elements: int
    admitted_bytes: int
    mapped_elements: tuple[int, ...]
    filled_elements: tuple[int, ...]
    converted_elements: tuple[int, ...]
    process_calls: tuple[int, ...]
    partial_calls: tuple[int, ...]
    partial_elements: tuple[int, ...]
    peak_bytes: int
    concurrent_peak_bytes: int
    worker_seconds: float
    quarter_uses: tuple[int, ...]
    quarter_hits: tuple[int, ...]
    quarter_requests: tuple[int, ...]
    quarter_seconds: tuple[float, ...]
    quarter_service_seconds: tuple[tuple[float, ...], ...]

    @classmethod
    def from_dict(cls, d) -> PlanSummary:
        """A summary from the native counts, restoring its per-quarter and
        per-schema vectors as tuples."""
        values: dict[str, Any] = {}
        for k, v in d.items():
            values[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
        return cls(**values)

    @property
    def seconds(self) -> float:
        """Modeled seconds of the whole sequence."""
        return sum(self.quarter_seconds)

    def quarter_accesses(self) -> tuple[int, ...]:
        """Accesses per quarter of the sequence."""
        n = self.accesses
        return tuple(-(-n * (q + 1) // 4) - -(-n * q // 4) for q in range(4))

    def to_dict(self) -> dict:
        """The summary as nested plain values (JSON or another process)."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class Prices:
    """Marginal seconds of modeled time per read request, byte read,
    core-second of worker work and byte of cached-row memory traffic: which
    resource binds (the cost model's gradient)."""

    request: float
    byte: float
    core: float
    memory: float


@dataclass(frozen=True, slots=True)
class OnlineStats:
    """An online compilation: accesses, chunks and the chunk size, horizon
    and reuse gap (`distance`) in accesses, shards; page uses, hits and
    admissions, row hits and admissions, block reads and read requests of
    the published plan; row candidates seen and kept and page intervals
    refused; the last capacity price (`row_price`, seconds per
    frame-position) and resource `prices`; modeled seconds and core-seconds;
    wall seconds compiling and waiting for the loader."""

    accesses: int
    chunks: int
    chunk: int
    horizon: int
    distance: int
    shards: int
    page_uses: int
    page_hits: int
    page_admissions: int
    row_admissions: int
    row_hits: int
    block_reads: int
    requests: int
    row_candidates: int
    row_kept: int
    page_refused: int
    row_price: float
    prices: Prices
    modeled_seconds: float
    worker_seconds: float
    compile_seconds: float
    waiting_seconds: float

    @classmethod
    def from_dict(cls, d) -> OnlineStats:
        """An online compilation's statistics from the native record, its
        resource prices rebuilt as `Prices`."""
        return cls(**{**d, "prices": Prices(**d["prices"])})

    def to_dict(self) -> dict:
        """The statistics as nested plain values (JSON or another process)."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class CacheSummary:
    """The serial page optimum at a capacity: page uses, distinct pages,
    hits and misses (row and output hits are zero for pages)."""

    page_uses: int
    distinct_pages: int
    hits: int
    misses: int
    capacity_pages: int
    row_hits: int
    output_hits: int
