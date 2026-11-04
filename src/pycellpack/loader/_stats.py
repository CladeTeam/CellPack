"""Run statistics and memory reservations: frozen records, built from the
native counts (field for field) and convertible to and from plain dicts
for JSON or other processes.

Counts are cumulative over one run, except the profile report, which holds the
sites a profiled run recorded, and `MemoryEstimate`, which is computed before a
run starts. `from_dict` is also the converter the native side calls to hand a
finished run back as Python objects, so its field names are the native ones.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .. import _cellpack
from .opt._stats import OnlineStats


def _asdict(record) -> dict:
    """A record as nested plain values (for JSON and `RunStats.from_dict`)."""
    return dataclasses.asdict(record)


@dataclass(frozen=True, slots=True)
class IoStats:
    """Read requests, bytes requested and read, and requests by size in
    pages (1, 2, 3-4, 5-8, 9-16, 17-32, 33+).

    `requested_bytes` counts what the issued requests asked for (pages per
    request times the page size), `read_bytes` what the device returned;
    `sizes` has one count per band, summing to `requests`."""

    requests: int
    requested_bytes: int
    read_bytes: int
    sizes: tuple[int, ...]

    @classmethod
    def from_dict(cls, d) -> IoStats:
        """A record from its dict form (`sizes` back as a tuple)."""
        return cls(**{**d, "sizes": tuple(d["sizes"])})


@dataclass(frozen=True, slots=True)
class OutputStats:
    """One output's access counts: accesses of that schema, records decoded
    for it (a cell shared by two outputs decodes once, so outputs of a
    position can differ), hits served by a cached row, and process function
    calls (0 for the built-in output)."""

    accesses: int
    decoded_rows: int
    row_hits: int
    process_calls: int


@dataclass(frozen=True, slots=True)
class SiteReport:
    """One profiled site: calls, seconds inside it (`time`, including child
    sites) and without its recorded children (`self_time`); `details` as
    `(key, calls, seconds)` per schema (`map`) or per schema, size band and
    row kind (`process`, see `ProfileReport.process_key`); `hist` as
    `(lower bound in seconds, count)` per log2 bucket for histogram sites."""

    calls: int
    time: float
    self_time: float
    details: tuple[tuple[int, int, float], ...]
    hist: tuple[tuple[float, int], ...] | None

    @classmethod
    def from_dict(cls, d) -> SiteReport:
        """A record from its dict form, restoring the tuple fields and an
        absent `hist` as None."""
        hist = d.get("hist")
        return cls(d["calls"], d["time"], d["self_time"], tuple(tuple(x) for x in d["details"]),
                   None if hist is None else tuple(tuple(x) for x in hist))


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """The sites a profiled run recorded: `schemas` the outputs the sites
    were laid out for, `bands` the process size bounds (`PROCESS_BOUNDS`),
    and `sites` the reports by site name."""

    schemas: int
    bands: tuple[int, ...]
    sites: dict[str, SiteReport]

    @classmethod
    def from_dict(cls, d) -> ProfileReport:
        """A report from its dict form, rebuilding each site's record."""
        return cls(d["schemas"], tuple(d["bands"]), {n: SiteReport.from_dict(s) for n, s in d["sites"].items()})

    def site(self, name) -> SiteReport | None:
        """The site's report, or None when it was not recorded."""
        return self.sites.get(name)

    def time(self, *names) -> float:
        """Summed seconds of the named sites (0 for unrecorded ones)."""
        return sum(s.time for n in names if (s := self.sites.get(n)) is not None)

    def process_key(self, schema: int, band: int, partial: bool) -> int:
        """The `process` detail key of a schema, size band and row kind."""
        return (schema * len(self.bands) + band) * 2 + int(partial)

    def process_bands(self, schema: int) -> list[dict]:
        """Per size band of `schema`: full and partial calls and seconds."""
        n = len(self.bands)
        out = [{"max_elements": b, "full_calls": 0, "full_time": 0.0, "partial_calls": 0, "partial_time": 0.0} for b in self.bands]
        site = self.site("process")
        for key, calls, seconds in site.details if site is not None else ():
            # Keys are `(schema * bands + band) * 2 + partial`; the divmods undo
            # `process_key`.
            s, rest = divmod(key, 2 * n)
            if s != schema:
                continue
            band, partial = divmod(rest, 2)
            kind = "partial" if partial else "full"
            out[band][f"{kind}_calls"] += calls
            out[band][f"{kind}_time"] += seconds
        return out

    def map_time(self, schema: int) -> float:
        """Seconds mapping `schema`'s rows."""
        site = self.site("map")
        return sum(t for key, _, t in (site.details if site is not None else ()) if key == schema)


@dataclass(frozen=True, slots=True)
class RunStats:
    """A run's counts (always kept) and, for profiled runs, its sites.

    Counts: accesses, decoded and mapped rows, process calls, bytes written;
    record pages used, served from cache frames (`page_hits`) and read;
    pages admitted into cache frames and the extra admission reads among
    them (`block_reads`); accesses served by a cached row and the row bytes
    gathered for them, rows built and their bytes; tasks that waited for
    a frame another access admits; claims held back by the window, the plan
    and the output ring; tasks whose reads waited for private frames
    (`starved`). `elapsed` is the workers' and plan thread's wall time after
    `setup_elapsed`; `peak_span` the most accesses claimed past the frontier;
    `cache_bytes` the plan's cache frames (`cache frames × page size`) and
    `private_bytes` the workers' private frames. `outputs` maps output names
    to their counts; `online` is the online compiler's statistics
    (`OnlineStats.accesses` counts accesses, not positions)."""

    accesses: int
    decoded_rows: int
    mapped_rows: int
    process_calls: int
    written_bytes: int
    page_uses: int
    page_hits: int
    page_reads: int
    admitted: int
    block_reads: int
    row_hits: int
    row_gather_bytes: int
    rows_admitted: int
    row_admit_bytes: int
    waited: int
    window_stalls: int
    plan_stalls: int
    output_stalls: int
    starved: int
    elapsed: float
    setup_elapsed: float
    peak_span: int
    cache_bytes: int
    private_bytes: int
    io: IoStats
    outputs: dict[str, OutputStats]
    profile: ProfileReport | None
    online: OnlineStats | None

    @classmethod
    def from_dict(cls, d) -> RunStats:
        """A record from `to_dict` (or the native counts)."""
        return cls(**{
            **d,
            "io": IoStats.from_dict(d["io"]),
            "outputs": {name: OutputStats(**o) for name, o in d["outputs"].items()},
            "profile": None if d.get("profile") is None else ProfileReport.from_dict(d["profile"]),
            "online": None if d.get("online") is None else OnlineStats.from_dict(d["online"]),
        })

    def to_dict(self) -> dict:
        """The record as nested plain values, ready for JSON or another
        process (`DistributedRun` passes it through the status region)."""
        return _asdict(self)


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """A run's known reservations in bytes: cache frames and their
    generation words, private frames, worker scratch, the plan ring,
    metadata and the output ring, and their sum. Compiler state, allocator
    overhead and the OS cache are not included; this is not RSS."""

    cache_frame_bytes: int
    generation_bytes: int
    private_frame_bytes: int
    worker_scratch_bytes: int
    plan_ring_bytes: int
    metadata_bytes: int
    output_ring_bytes: int
    known_reservation_bytes: int

    def to_dict(self) -> dict:
        """The estimate as a plain dict, keyed like the native fields."""
        return _asdict(self)


# The native side builds RunStats through this converter, so `Run.join` and
# `Run.snapshot` return records rather than dicts.
_cellpack.register_run_stats(RunStats.from_dict)
