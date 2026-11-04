"""Measuring a loader's cost model from profiled runs.

`calibrate` runs a loader over a random subset of cells in seven
configurations (no cache timed and untimed, a page plan, a row plan timed and
untimed, a block shuffle, an online plan), holds each to a steady state
(`pycellpack.loader.costs.Steady`) and fits the host's worker constants, the storage's
device rates and each process function's costs from the snapshot differences.
Worker seconds are converted to core-seconds, and the per-row element counts a
window needs come from the serial summary of the plan it ran. Each part is
stored only when the windows it comes from are sound.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import replace

import numpy as np

from ..opt import OutputCost, PlanOptions
from . import _defaults
from ._fit import (
    BOOKKEEPING,
    SUBMIT,
    Measured,
    Shape,
    delta,
    fit_host,
    fit_storage,
    map_element,
    plain_profile,
    process_bands,
    process_values,
)
from ._resolve import compose, keys
from ._steady import Steady
from ._store import CostStore

#: Seconds one epoch of the calibration subset should take (short epochs
#: warm the cache plans quickly).
EPOCH_SECONDS = 4.0
#: Seconds between run snapshots.
INTERVAL = 0.5
#: Seconds the pilot run measures the uncached rate over.
PILOT_SECONDS = 3.0
#: Most worker wait share of a window fitting the host's worker constants.
WAIT_LIMIT = 0.25
#: Most bookkeeping misfit (relative, any window) a saved calibration may have.
MAX_RESIDUAL = 0.25
#: Bytes per access a calibration sequence and its plan take, roughly.
BYTES_PER_ACCESS = 32


def _plain(stats) -> dict:
    """Snapshot statistics as plain nested numbers a difference can take:
    outputs in order, profile details by key."""
    d = stats.to_dict()
    d["outputs"] = list(d["outputs"].values())
    d["profile"] = plain_profile(stats.profile)
    return d


def _measure(loading, accesses, plan, steady: Steady):
    """Runs until `steady` has a window. Returns the snapshots bounding it
    (None without one) and the run's accesses per second after its first
    sample."""
    samples = []
    with loading.start(accesses, plan=plan, slots=8) as run:
        due = 0.0
        for _ in run:
            now = time.monotonic()
            if now < due:
                continue
            due = now + INTERVAL
            s = run.snapshot()
            samples.append(s)
            if steady.push(s.elapsed, s.accesses, s.page_reads):
                break
        else:
            if steady.result is None:
                steady.result = steady.best()
        run.cancel()
    rate = 0.0
    if len(samples) > 1 and samples[-1].elapsed > samples[0].elapsed:
        rate = (samples[-1].accesses - samples[0].accesses) / (samples[-1].elapsed - samples[0].elapsed)
    w = steady.result
    if w is None:
        return None, rate
    return (_plain(samples[w.first]), _plain(samples[w.last]), w), rate


_BAND_SUMS = ("full_calls", "full_time", "partial_calls", "partial_time")


def _sum_bands(a: list, b: list) -> list:
    """Process size profiles added band by band."""
    if not a:
        return [dict(x) for x in b]
    return [{**x, **{k: x[k] + y[k] for k in _BAND_SUMS}} for x, y in zip(a, b)]

#: Profile group a calibration turns on: every site the fits read.
SITES = ("calibrate",)


def _pilot(loading, cells, schemas, seconds=PILOT_SECONDS):
    """Positions per second without a cache, over repeats of `cells`."""
    sequence = np.repeat(np.tile(cells, max(1, (1 << 22) // cells.size)), schemas)
    with loading.start(sequence, slots=8) as run:
        start = None
        for _ in run:
            s = run.snapshot()
            if start is None and s.accesses > 0:
                start = s
            elif start is not None and s.elapsed - start.elapsed >= seconds:
                break
        end = run.snapshot()
        run.cancel()
    if start is None or end.elapsed <= start.elapsed:
        return 1e5
    return max(1.0, (end.accesses - start.accesses) / (end.elapsed - start.elapsed) / schemas)


class _Log:
    """Progress lines on stderr, written only when verbose."""

    def __init__(self, verbose):
        """Keeps whether lines are written."""
        self.verbose = verbose

    def __call__(self, message):
        """Writes one line, when verbose."""
        if self.verbose:
            print(f"[pycellpack.loader.costs] {message}", file=sys.stderr, flush=True)


def calibrate(loading, cells=None, *, steady_seconds=30.0, max_warmup=None, budget=None, save=True,
              force=False, store: CostStore | None = None, seed=0, verbose=True):
    """Measures `loading`'s cost model on this machine and stores it.

    Seven configurations over a random subset of `cells` (default: every
    cell) each hold a steady state for `steady_seconds` (warm-up excluded;
    see `pycellpack.loader.costs.Steady`): no cache, and its untimed twin; a page
    plan of `budget` bytes (default: a quarter of the subset's page
    footprint, at least 64 pages and at most a quarter of the available
    memory); a row plan of `budget` bytes and its untimed twin; a page plan
    over a block shuffle; an online page plan. Worker seconds become
    core-seconds where workers outnumber the physical cores they run on
    (`Loader.cores`). Their snapshot differences fit the host's worker
    constants, the storage's device rates and each process function's costs,
    saved as three records (see `CostStore`) unless `save` is false. Each
    part is saved only when its own windows are sound (unless `force`): the
    worker constants when windows that keep the workers busy fit them within
    `MAX_RESIDUAL`, the device rates when the uncached windows are steady,
    process costs when the profiled windows are; a part not saved keeps its
    stored record or defaults. Shared storage that fluctuates may thus yield
    device rates only.

    Returns the measured model (`provenance` says what was measured).
    """
    from ... import loader as pl

    log = _Log(verbose)
    store = CostStore() if store is None else store
    source, schemas = loading.source, dict(loading.schemas)
    if not source.direct_io:
        warnings.warn("the data source reads through the page cache: device rates measure the cache, not the storage",
                      RuntimeWarning, stacklevel=2)
    profiled = pl.Loader(source, schemas, config=replace(loading.config, profile=SITES))
    untimed = pl.Loader(source, schemas, config=replace(loading.config, profile=None))
    S = len(schemas)
    rng = np.random.default_rng(seed)
    pool = np.arange(source.n_cells, dtype=np.uint32) if cells is None else np.unique(np.asarray(cells, dtype=np.uint32))
    if pool.size == 0:
        raise ValueError("calibration needs cells")

    # Pilot: positions per second without a cache, which sizes the subset and
    # each run's epochs.
    rate = _pilot(profiled, rng.permutation(pool)[: min(pool.size, 1 << 20)], S)
    from .._config import available_memory

    memory = available_memory() or (8 << 30)
    # Longest sequence one run may take: at most 2**27 accesses and half the
    # memory available at BYTES_PER_ACCESS per access, in positions (`S` per
    # access).
    cap = max(1 << 16, int(min(1 << 27, memory * 0.5 / BYTES_PER_ACCESS) // S))
    # One epoch of the subset should last about EPOCH_SECONDS, so a plan warms
    # its cache well inside a run.
    subset_size = int(min(pool.size, max(256, rate * EPOCH_SECONDS)))
    subset = np.sort(rng.choice(pool, size=subset_size, replace=False))
    warm = (max_warmup if max_warmup is not None else 10.0 + 3 * EPOCH_SECONDS)
    log(f"pilot: {rate:,.0f} positions/s; subset of {subset_size:,} cells; {steady_seconds:g} s steady per run")

    def epochs_for(positions_per_second):
        """Epochs of the subset a run needs to last its warm-up and steady time
        at this rate (with headroom), capped by `cap` accesses."""
        need = positions_per_second * (warm + steady_seconds) * 1.3
        return max(2, math.ceil(min(need, cap) / subset_size))

    def random_sequence(epochs):
        """`epochs` random permutations of the subset, each cell repeated once
        per schema (position-major access order)."""
        return np.repeat(np.concatenate([rng.permutation(subset) for _ in range(epochs)]), S)

    def block_sequence(epochs, block=256):
        """`epochs` shuffles of `block`-cell blocks: cells of a block stay
        adjacent, so records and pages are reused, as a spatially ordered access
        pattern would reuse them."""
        n = subset_size // block * block
        blocks = subset[:n].reshape(-1, block) if n else subset.reshape(1, -1)
        return np.repeat(np.concatenate([blocks[rng.permutation(len(blocks))].reshape(-1) for _ in range(epochs)]), S)

    sample = rng.choice(subset, size=min(subset_size, 2000), replace=False)
    pages = float(np.mean([source.cell_pages(int(c))[2] for c in sample]))
    footprint = int(pages * subset_size * source.page_size)
    cache = int(budget if budget is not None else max(source.page_size * 64, footprint // 4))
    cache = int(min(cache, memory * 0.25))
    log(f"subset footprint {footprint / 2**20:,.0f} MiB; cache budget {cache / 2**20:,.0f} MiB")

    base_cost = compose(loading, _defaults.HOST, _defaults.storage("unknown"), {
        n: OutputCost(**(_defaults.PROCESS if s.process is not None else _defaults.HOST["builtin"]["*"])) for n, s in schemas.items()})
    configs = [
        # name, loader, sequence kind, speed guess, plan kind
        ("off", profiled, "random", 1.0, None),
        ("off-untimed", untimed, "random", 1.0, None),
        ("pages", profiled, "random", 2.0, "pages"),
        ("rows", profiled, "random", 3.0, "rows"),
        ("rows-untimed", untimed, "random", 3.0, "rows"),
        ("block", profiled, "block", 3.0, "pages"),
        ("online", profiled, "random", 2.0, "online"),
    ]
    sequences, plans, shapes, measured = {}, {}, {}, {}
    fastest = 0.0  # the highest positions per second measured so far
    for name, ld, kind, speed, planning in configs:
        found = None
        speed_rate = max(rate * speed, fastest)
        for attempt in range(2):
            epochs = epochs_for(speed_rate)
            if (kind, epochs) not in sequences:
                sequences[(kind, epochs)] = (random_sequence if kind == "random" else block_sequence)(epochs)
            accesses = sequences[(kind, epochs)]
            key = (kind, planning, epochs)
            plan = None
            if planning in ("pages", "rows"):
                if key not in plans:
                    options = {
                        "pages": PlanOptions(cache_bytes=cache),
                        # Rows first, so that rows are cached whatever the
                        # default costs would choose.
                        "rows": PlanOptions(cache_bytes=cache, cost=base_cost, rows=True, row_policy="global"),
                    }[planning]
                    plans[key] = profiled.compile(accesses, options=options)
                plan = plans[key]
                shapes[name] = Shape.of(plan.summary)
            elif planning == "online":
                plan = PlanOptions(cache_bytes=cache)
            steady = Steady(steady_seconds, max_warmup=max_warmup)
            found, observed = _measure(ld, accesses, plan, steady)
            fastest = max(fastest, observed / S)
            if found is not None and (found[2].steady or attempt == 1):
                break
            # The retry's length follows this run's measured rate.
            speed_rate = max(speed_rate, 1.5 * observed / S)
            log(f"{name}: no {steady_seconds:g}-second steady window before the sequence ended; retrying with a longer one")
        if found is None:
            raise RuntimeError(f"calibration run {name!r} ended before a window of {steady_seconds:g} s")
        first, last, w = found
        measured[name] = (delta(last, first), w)
        log(f"{name}: {(last['accesses'] - first['accesses']) / w.seconds:,.0f} accesses/s over {w.seconds:.1f} s "
            f"({'steady' if w.steady else 'unsteady'}, cv {w.cv:.3f}, {w.restarts} restarts)")

    # Every window's shape: its own plan's, else the page plan's (cache-off
    # and online runs decode every access of the same random sequence).
    reference = shapes["pages"]
    runs = []
    for name, ld, kind, _, planning in configs:
        stats, w = measured[name]
        shape = shapes.get(name) or (shapes["rows"] if name == "rows-untimed" else reference)
        runs.append(Measured(name, stats, w.seconds, shape, ld is profiled, w.steady, w.cv, loading.config.workers))
    by = {m.name: m for m in runs}
    # Profiling overhead: the timed and untimed runs of the same sequence differ
    # by the timers' cost; their throughput ratio is the scale applied to every
    # site time, clamped so a noisy window cannot rescale the whole model.
    ratios = [by[a].rate / by[b].rate for a, b in (("off", "off-untimed"), ("rows", "rows-untimed")) if by[b].rate > 0]
    profile_scale = float(min(1.0, max(0.5, np.mean(ratios)))) if ratios else 1.0
    # Worker seconds to core-seconds: workers sharing a core through SMT
    # each count its time.
    cores = float(loading.cores)
    scale = profile_scale * min(1.0, cores / loading.config.workers)

    k = keys(loading)
    host_k, host_id = k["host"]
    old_host = store.get("host", host_k)
    host_base = old_host["values"] if old_host else _defaults.HOST
    fit_runs = [m for m in runs if m.name != "online"]
    # Worker constants come from windows the workers kept busy: on a slow
    # device the small bookkeeping phases drown in its variation.
    bound = [m for m in fit_runs if m.profiled and m.wait_share < WAIT_LIMIT]
    for m in runs:
        if m.profiled:
            phases = " ".join(f"{p} {1e6 * m.per(m.phase(p)):.3f}"
                              for p in (*BOOKKEEPING, *SUBMIT, "decode", "map", "row.gather", "row.admit", "process", "io.wait", "idle"))
            phases += f" other {1e6 * m.per(m.other()):.3f}"
            log(f"{m.name}: workers wait {m.wait_share:.0%} of the time; us per access: {phases}")
    fit_host_constants = len(bound) >= 2
    if fit_host_constants:
        host, report = fit_host(bound, scale, host_base)
        log("bookkeeping residuals: " + ", ".join(f"{n} {v:+.1%}" for n, v in report["bookkeeping_residuals"].items()))
    else:
        host, report = dict(host_base), {"bookkeeping_residuals": {}}
        log(f"host constants kept ({'stored' if old_host else 'defaults'}): fewer than two windows keep the workers busy")
    online, pages = by["online"], by["pages"]
    if online.wait_share < WAIT_LIMIT and pages.wait_share < WAIT_LIMIT:
        host["compile"] = scale * max(0.0, online.per(online.bookkeeping()) - pages.per(pages.bookkeeping()))
    builtin = dict(host_base.get("builtin", {}))
    outputs = {}
    process_records = []
    for i, (name, schema) in enumerate(schemas.items()):
        mapped = map_element(fit_runs, i, scale)
        if schema.process is None:
            entry = dict(builtin.get(schema.dtype.name) or _defaults.HOST["builtin"]["*"])
            if mapped is not None:
                entry["map_element"] = mapped
            builtin[schema.dtype.name] = entry
            outputs[name] = OutputCost(**entry)
            continue
        profile: list = []
        time_sum = calls = 0
        for m in fit_runs:
            if not m.profiled:
                continue
            o = m.stats["outputs"][i]
            bands, seconds = process_bands(m.stats, i)
            profile = _sum_bands(profile, bands)
            time_sum += seconds
            calls += o["process_calls"]
        values, weights = process_values(profile, time_sum, calls, scale)
        record = {**_defaults.PROCESS, **{kk: v for kk, v in values.items() if v is not None}}
        if mapped is not None:
            record["map_element"] = mapped
        outputs[name] = OutputCost(**record)
        process_records.append((name, k["outputs"][name], record, weights))
    host["builtin"] = builtin

    storage_k, storage_id = k["storage"]
    storage, device = fit_storage(by["off"], by["off-untimed"], _defaults.storage(k["tier"]), scale, cores)

    windows = {m.name: {"seconds": m.seconds, "rate": m.rate, "steady": m.steady, "cv": m.cv, "wait_share": m.wait_share} for m in runs}
    residuals = report["bookkeeping_residuals"]
    worst = max((abs(v) for v in residuals.values()), default=0.0)
    # Each part is judged on the windows it comes from: the host's worker
    # constants on the bookkeeping fit, the device rates on the uncached
    # windows, process costs on the profiled windows (timed per call).
    problems = {}
    if not fit_host_constants:
        problems["host"] = "no windows keep the workers busy"
    elif worst > MAX_RESIDUAL or any(not m.steady for m in bound):
        problems["host"] = f"bookkeeping misfit {worst:.0%}" if worst > MAX_RESIDUAL else "unsteady windows"
    if not (by["off"].steady and by["off-untimed"].steady):
        problems["storage"] = "unsteady uncached windows"
    if process_records and any(not m.steady for m in fit_runs if m.profiled):
        problems["output"] = "unsteady profiled windows"
    if "host" in problems:
        host = {**host_base, "builtin": builtin}
    keep = {part: save and (force or part not in problems) for part in ("host", "storage", "output")}
    for part, why in problems.items():
        if save and not keep[part]:
            fallback = ("the stored record" if old_host else "the defaults") if part == "host" else "the stored record or defaults"
            warnings.warn(f"{part} costs not saved ({why}); keeping {fallback} (pass force=True to save)",
                          RuntimeWarning, stacklevel=2)
    common = {"workers": loading.config.workers, "cores": cores, "steady_seconds": steady_seconds, "windows": windows,
              "cache_bytes": cache, "subset_cells": subset_size}
    saved = []
    if keep["host"]:
        store.put("host", host_k, host_id, host, {**common, "profile_scale": profile_scale, "residuals": residuals})
        saved.append(f"host {host_k}")
    if keep["storage"]:
        store.put("storage", storage_k, storage_id, storage, {**common, **device, "tier": k["tier"]})
        saved.append(f"storage {storage_k}")
    if keep["output"]:
        for name, (output_k, output_id), record, weights in process_records:
            store.put("output", output_k, output_id, record, {**common, "calls": weights, "runs": 1})
            saved.append(f"output {output_k}")
    if saved:
        log(f"saved {', '.join(saved)} to {store.root}")
    host_source = (f"store:host/{host_k}" if keep["host"] or old_host else "default") if "host" in problems else \
        (f"store:host/{host_k}" if keep["host"] else "calibrated")
    provenance = {
        "host": host_source,
        "storage": f"store:storage/{storage_k}" if keep["storage"] else "calibrated",
        "outputs": {n: (host_source if k["outputs"][n] is None else
                        f"store:output/{k['outputs'][n][0]}" if keep["output"] else "calibrated") for n in schemas},
    }
    return compose(loading, host, storage, outputs, provenance)
