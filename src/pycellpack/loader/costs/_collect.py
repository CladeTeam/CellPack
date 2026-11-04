"""Process-function costs gathered from profiled runs as they finish.

A run that profiles the `process` site records every call with its schema, size
band and row kind. `record_run` turns one finished run's process calls into the
per-band curve of each process function it called, converts worker seconds to
core-seconds and merges them into that function's stored record, where earlier
samples decay. Recording only ever warns: a run never fails because its costs
could not be saved.
"""

from __future__ import annotations

import warnings

from . import _defaults
from ._fit import plain_profile, process_bands, process_values
from ._keys import host_key
from ._resolve import keys
from ._store import CostStore

#: Weight left to earlier samples each time a run is merged in.
DECAY = 0.8


def _mean(old, new, w_old, w_new):
    """A weighted mean that falls back to the new value without weights."""
    return (old * w_old + new * w_new) / (w_old + w_new) if w_old + w_new else new


def _merge(old: dict | None, values: dict, weights: list):
    """Band-wise weighted means of the stored and new curves (earlier calls
    decayed by `DECAY`); the flat costs weighted by total calls."""
    fresh = {**_defaults.PROCESS, **{k: v for k, v in values.items() if v is not None}}
    if old is None:
        return fresh, {"calls": weights, "runs": 1}
    before, measured = old["values"], old.get("measured", {})
    prior = [DECAY * w for w in measured.get("calls", [])]
    if len(prior) != len(weights) or len(before.get("curve", ())) != len(fresh["curve"]):
        return fresh, {"calls": weights, "runs": 1}
    curve = []
    for a, b, wa, wb in zip(before["curve"], fresh["curve"], prior, weights):
        curve.append({"max_elements": b["max_elements"], "process": _mean(a["process"], b["process"], wa, wb),
                      "resolve": _mean(a["resolve"], b["resolve"], wa, wb)})
    wa, wb = sum(prior), sum(weights)
    merged = dict(before)
    merged["curve"] = curve
    for name in ("process", "resolve"):
        if values.get(name) is not None and wa + wb:
            merged[name] = (before.get(name, values[name]) * wa + values[name] * wb) / (wa + wb)
    return merged, {"calls": [a + b for a, b in zip(prior, weights)], "runs": measured.get("runs", 0) + 1}


def record_run(loading, stats, *, store: CostStore | None = None) -> int:
    """Merges the size profile (the `process` site) of each process function
    a profiled run called into its stored costs (scaled by the host's measured profile
    overhead, if any). Returns the records updated; failures only warn."""
    store = CostStore() if store is None else store
    try:
        k = keys(loading)
        host = store.get("host", host_key())
        scale = host["measured"].get("profile_scale", 1.0) if host else 1.0
        # Worker seconds to core-seconds where workers share cores.
        scale *= min(1.0, loading.cores / max(1, loading.config.workers))
        updated = 0
        plain = {"profile": plain_profile(stats.profile)}
        names = list(stats.outputs)
        for name, entry in k["outputs"].items():
            o = stats.outputs.get(name)
            if entry is None or o is None or not o.process_calls:
                continue
            bands, seconds = process_bands(plain, names.index(name))
            if not any(b["full_calls"] + b["partial_calls"] for b in bands):
                continue
            values, weights = process_values(bands, seconds, o.process_calls, scale)
            output_k, identity = entry
            store.update("output", output_k, identity, lambda old, v=values, w=weights: _merge(old, v, w))
            updated += 1
        return updated
    except Exception as exc:  # noqa: BLE001 - recording must never fail a run
        warnings.warn(f"process costs not recorded: {exc}", RuntimeWarning, stacklevel=3)
        return 0
