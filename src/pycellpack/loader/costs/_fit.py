"""Cost constants from steady windows of profiled runs.

Each window is the difference of two run snapshots: worker seconds per
profiling site (summed over workers, see `pycellpack.loader.profile_sites`)
and event counts. Counts the runtime does not keep (stored elements per
decoded row and per cached row, and the share of decoded rows in a cold
chunk) come from the serial summary of the window's plan, whose windows lie
in stationary epochs of the sequence. Each constant comes from the sites
that do its work: decoding, gathering and admitting rows, mapping and
processing are direct ratios; the per-access bookkeeping is fitted by
nonnegative least squares over the windows. All are scaled by the profile's
timer overhead and from worker seconds to core-seconds (`scale`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..._cellpack import PROCESS_BOUNDS


def delta(after, before):
    """`after - before` for nested counters (dicts, lists, numbers)."""
    if isinstance(after, dict):
        return type(after)({k: delta(v, before.get(k, 0) if isinstance(before, dict) else 0) for k, v in after.items()})
    if isinstance(after, list):
        before = before if isinstance(before, list) else []
        return [delta(a, before[i] if i < len(before) else 0) for i, a in enumerate(after)]
    if isinstance(after, bool) or not isinstance(after, (int, float)):
        return after
    return after - (before if isinstance(before, (int, float)) else 0)


def plain_profile(profile) -> dict | None:
    """A profile report as nested numbers a snapshot difference can take:
    per site its calls and times, and its details by key."""
    if profile is None:
        return None
    sites = {}
    for name, s in profile.sites.items():
        sites[name] = {
            "calls": s.calls,
            "time": s.time,
            "self_time": s.self_time,
            "details": {str(key): {"calls": calls, "time": t} for key, calls, t in s.details},
        }
    return {"sites": sites}


def site_time(stats, name, field="time") -> float:
    """Seconds of a site in plain stats (0 when it was not recorded)."""
    profile = stats.get("profile")
    site = profile["sites"].get(name) if profile else None
    return site[field] if site else 0.0


def detail(stats, name, key) -> dict:
    """Calls and seconds of one detail key of a site in plain stats."""
    profile = stats.get("profile")
    site = profile["sites"].get(name) if profile else None
    return site["details"].get(str(key), {"calls": 0, "time": 0.0}) if site else {"calls": 0, "time": 0.0}


def process_bands(stats, schema: int) -> tuple[list, float]:
    """A schema's process calls and seconds by size band (full and partial)
    in plain stats, and their total seconds."""
    n = len(PROCESS_BOUNDS)
    bands, total = [], 0.0
    for band, bound in enumerate(PROCESS_BOUNDS):
        # The detail key `ProfileReport.process_key` builds: schema, band, kind.
        full = detail(stats, "process", (schema * n + band) * 2)
        partial = detail(stats, "process", (schema * n + band) * 2 + 1)
        bands.append({"max_elements": bound, "full_calls": full["calls"], "full_time": full["time"],
                      "partial_calls": partial["calls"], "partial_time": partial["time"]})
        total += full["time"] + partial["time"]
    return bands, total


def nnls(columns, target):
    """Least squares with nonnegative coefficients: columns that come out
    negative are dropped and the rest refitted."""
    a = np.asarray(columns, dtype=float).T
    b = np.asarray(target, dtype=float)
    keep = [j for j in range(a.shape[1]) if np.any(a[:, j])]
    coef = np.zeros(a.shape[1])
    while keep:
        x, *_ = np.linalg.lstsq(a[:, keep], b, rcond=None)
        if (x >= 0).all():
            coef[keep] = x
            break
        keep = [j for j, v in zip(keep, x) if v >= 0]
    return coef


@dataclass
class Shape:
    """Per-row element counts of a sequence under a plan (serial summary).

    The runtime counts rows and bytes, not elements, so the fits take per-row
    sizes from the plan's serial replay: stored elements per decoded row, per
    access served by a cached row, and the share of decoded rows in a cold
    chunk (one outside the few most recently decoded).
    """

    decoded: float  # stored elements per decoded row
    cached: float  # per access served by a cached row
    cold: float  # share of decoded rows in a cold chunk

    @classmethod
    def of(cls, summary, base: Shape | None = None) -> Shape:
        """The shapes a plan's serial summary implies. A count of zero gives no
        ratio, so that field keeps `base`'s (the per-row sizes of the related
        runs) and, for cached rows, falls back to the decoded row size."""
        def ratio(a, b, fallback):
            """`a / b`, or `fallback` when `b` is zero."""
            return a / b if b else fallback

        base = base or cls(0.0, 0.0, 0.0)
        return cls(
            decoded=ratio(summary.decoded_elements, summary.decoded_rows, base.decoded),
            cached=ratio(summary.row_elements, summary.row_hits, base.cached or base.decoded),
            cold=ratio(summary.cold_rows, summary.decoded_rows, base.cold),
        )


@dataclass
class Measured:
    """One steady window: `stats` the snapshot difference over `seconds`.

    `profiled` says whether the run timed its sites, `steady` whether the
    detector accepted the window and `cv` its spread, which orders the fallback
    window when none is steady. `workers` closes the per-access phases with the
    time no site covered.
    """

    name: str
    stats: dict
    seconds: float
    shape: Shape
    profiled: bool
    steady: bool
    cv: float
    #: Worker threads of the run (0: unknown, no unattributed time).
    workers: int = 0

    @property
    def accesses(self):
        """Accesses in the window, at least one so per-access ratios stay
        finite."""
        return max(1, self.stats["accesses"])

    @property
    def rate(self):
        """Accesses per second over the window."""
        return self.stats["accesses"] / self.seconds if self.seconds > 0 else 0.0

    def per(self, value):
        """`value` per access of the window."""
        return value / self.accesses

    def phase(self, *names):
        """Seconds of the named sites; `claim` counts without the issuing
        and waiting it contains."""
        return sum(site_time(self.stats, n, "self_time" if n == "claim" else "time") for n in names)

    @property
    def wait_share(self):
        """The share of worker time spent on reads rather than work:
        submitting them (filesystems without asynchronous direct reads, such
        as GPFS, complete them inside the submission), waiting for them, or
        idle."""
        waiting = self.phase(*SUBMIT, *WAITS)
        total = self.phase(*BUSY) + self.phase(*WAITS)
        return waiting / total if total > 0 else 0.0

    def other(self):
        """Worker seconds no site timed (the loop between them), when the
        worker count is known."""
        if not self.workers:
            return 0.0
        return max(0.0, self.workers * self.seconds - self.phase(*BUSY, *WAITS))

    def bookkeeping(self):
        """Worker seconds of per-access work: the bookkeeping sites and the
        unattributed rest."""
        return self.phase(*BOOKKEEPING) + self.other()


#: Per-access work: claiming, building reads, completing, the frontier.
BOOKKEEPING = ("claim", "issue", "complete", "frontier")
#: Submitting and harvesting reads.
SUBMIT = ("io.submit", "io.reap")
#: Waiting: for completions, with nothing to do, held back by the window or the plan.
WAITS = ("io.wait", "idle", "wait.plan", "wait.window")
#: Bookkeeping constants by how well windows identify them (indices into
#: `bookkeeping_columns`): access, cold (random against block sequences),
#: decode.
PRIORITY = (0, 2, 1)
#: Relative spread below which a count is treated as the same in every window.
FLAT = 0.05


def _flat(values) -> bool:
    """Whether a per-window count varies by at most `FLAT` of its size, and so
    cannot be told apart from a per-access constant."""
    lo, hi = min(values), max(values)
    return hi - lo <= FLAT * max(abs(hi), abs(lo), 1e-30)
BUSY = (*BOOKKEEPING, "decode", "map", "row.gather", "row.admit", "process", *SUBMIT)


def bookkeeping_columns(m: Measured):
    """One window's bookkeeping-fit columns, per access: the constant, decoded
    rows and cold decoded rows, whose coefficients are `access`, `decode` and
    `cold`."""
    s = m.stats
    return [1.0, m.per(s["decoded_rows"]), m.per(s["decoded_rows"] * m.shape.cold)]


def _ratio(runs: list[Measured], site: str, count: str) -> float | None:
    """Seconds of a site per unit of a count, over the windows that have it."""
    time = sum(m.phase(site) for m in runs)
    n = sum(m.stats.get(count, 0) for m in runs)
    return time / n if n > 0 else None


def fit_host(runs: list[Measured], scale: float, base: dict) -> tuple[dict, dict]:
    """Worker constants from profiled windows; `base` supplies what the
    windows cannot separate or did not exercise."""
    prof = [m for m in runs if m.profiled]
    values = copy.deepcopy(base)
    inp: dict = values["input"]
    rows: dict = values["rows"]
    report = {}

    # Stored elements decoded in the profiled windows: the runtime counts rows,
    # not elements, so the per-row size comes from each plan's summary.
    elements = sum(m.stats["decoded_rows"] * m.shape.decoded for m in prof)
    if elements > 0:
        inp["decode_element"] = scale * sum(m.phase("decode") for m in prof) / elements
    for key, site, count in (("gather", "row.gather", "row_gather_bytes"), ("admit", "row.admit", "row_admit_bytes")):
        r = _ratio(prof, site, count)
        if r is not None:
            rows[key] = scale * r

    # Bookkeeping: per access, decoded record and cold record.
    cols = [bookkeeping_columns(m) for m in prof]
    ys = [m.per(m.bookkeeping()) for m in prof]
    # A count that barely varies across the windows cannot be told apart
    # from the per-access constant, and fewer windows than unknowns cannot
    # separate them all: such constants keep their base values, taken out of
    # the target, lowest priority first.
    names = ("access", "decode", "cold")
    base_coef = [0.0, inp["decode"], inp["cold"]]
    fixed = [j for j in range(1, len(names)) if _flat([c[j] for c in cols])]
    free = [j for j in PRIORITY if j not in fixed]
    while len(free) > 1 and (len(free) >= len(cols)
                             or np.linalg.matrix_rank(np.array([[c[j] for j in free] for c in cols])) < len(free)):
        fixed.append(free.pop())
    # Take the fixed constants' contribution out of each target, then fit the
    # free columns only; fixed constants keep their base values.
    targets = [y - sum(base_coef[j] / scale * c[j] for j in fixed) for y, c in zip(ys, cols)]
    fit_cols = [[0.0 if j in fixed else v for j, v in enumerate(c)] for c in cols]
    coef = [base_coef[j] / scale if j in fixed else v for j, v in enumerate(nnls(list(zip(*fit_cols)), targets))]
    report["fixed"] = [names[j] for j in fixed]
    values["access"] = scale * coef[0]
    inp["decode"] = scale * coef[1]
    inp["cold"] = scale * coef[2]
    predicted = [float(np.dot(c, coef)) for c in cols]
    report["bookkeeping_residuals"] = {m.name: (p - y) / y if y else 0.0 for m, p, y in zip(prof, predicted, ys)}
    report["profile_scale"] = scale
    return values, report


def map_element(runs: list[Measured], index: int, scale: float) -> float | None:
    """One output's mapping cost per element, from its map time."""
    time = elements = 0.0
    for m in runs:
        if not m.profiled:
            continue
        o = m.stats["outputs"][index]
        time += detail(m.stats, "map", index)["time"]
        elements += o["decoded_rows"] * m.shape.decoded + o["row_hits"] * m.shape.cached
    return scale * time / elements if elements > 0 else None


def process_values(profile: list, process_time: float, calls: int, scale: float) -> tuple[dict, list]:
    """A process function's flat and banded costs from its size profile;
    bands without calls take their nearest measured neighbour's. Returns the
    values and the calls per band (their weights when merged later)."""
    bands = []
    for b in profile:
        full = b["full_time"] / b["full_calls"] if b["full_calls"] else None
        partial = b["partial_time"] / b["partial_calls"] if b["partial_calls"] else None
        bands.append((full, partial, b["full_calls"] + b["partial_calls"]))
    measured = [i for i, (f, _, _) in enumerate(bands) if f is not None]
    curve, weights = [], []
    for i, bound in enumerate(PROCESS_BOUNDS):
        if not measured:
            break
        j = min(measured, key=lambda m: abs(m - i))
        full, partial, _ = bands[j]
        resolve = max(0.0, partial - full) if partial is not None else 0.0
        curve.append({"max_elements": int(bound), "process": scale * full, "resolve": scale * resolve})
        weights.append(bands[i][2] if i < len(bands) else 0)
    flat = scale * process_time / calls if calls else None
    values = {"process": flat, "curve": curve}
    if curve:
        total = sum(weights) or 1
        values["resolve"] = sum(c["resolve"] * w for c, w in zip(curve, weights)) / total
    return values, weights


def fit_storage(device: Measured, plain: Measured | None, tier_defaults: dict, scale: float, cores: float) -> tuple[dict, dict]:
    """Device rates and the overlap share from the cache-off windows: the
    measured request and byte rates when the workers spent at least a quarter
    of their time waiting on the device; otherwise (worker-bound) at least the
    tier's defaults. `scale` turns profiled worker seconds into core-seconds;
    `cores` is the workers' CPU capacity."""
    timed = plain or device
    io = timed.stats["io"]
    iops = io["requests"] / timed.seconds
    bandwidth = io["read_bytes"] / timed.seconds
    busy = device.phase(*BUSY)
    wait_share = device.wait_share
    bound = "device" if wait_share >= 0.25 else "worker"
    if bound == "worker":
        iops = max(iops, tier_defaults["iops"])
        bandwidth = max(bandwidth, tier_defaults["bandwidth"])
    requests = device.stats["io"]["requests"]
    request = scale * device.phase(*SUBMIT) / requests if requests else tier_defaults["request"]
    # Overlap: the measured time per access against the longer of device and
    # core time (the profiled busy time, without the timers' overhead).
    a = timed.accesses
    dev = max(io["requests"] / a / iops, io["read_bytes"] / a / bandwidth)
    cpu = scale * device.per(busy) / cores
    measured = timed.seconds / a
    shorter, longer = min(dev, cpu), max(dev, cpu)
    serial = min(1.0, max(0.0, (measured - longer) / shorter)) if shorter > 0 else 0.0
    values = {"iops": iops, "bandwidth": bandwidth, "request": request, "serial": serial}
    return values, {"bound": bound, "wait_share": wait_share}
