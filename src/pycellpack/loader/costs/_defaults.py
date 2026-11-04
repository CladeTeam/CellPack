"""Costs used where nothing was measured: the crate's table
(`cellpack::opt::defaults`), which `Loader.plan_options` in Rust and the
bench examples read too. Worker costs are core-seconds measured on a Xeon
Platinum 8558 (local NVMe through io_uring with O_DIRECT, the benchmark's
`sample` process function over cellxgene); device rates per storage tier are
typical sustained random-read rates of their class.
"""

from __future__ import annotations

from ..._cellpack import cost_defaults

_TABLE = cost_defaults()

#: Storage values per tier: device rates, the per-request worker cost and
#: the non-overlapping share of device and worker time.
TIERS: dict = {name: dict(values) for name, values in _TABLE["tiers"].items()}
#: Tier assumed where the storage is not recognized.
UNKNOWN_TIER = "nvme"


def storage(tier: str) -> dict:
    """Storage values of a tier (NVMe's for unknown tiers)."""
    return dict(TIERS.get(tier, TIERS[UNKNOWN_TIER]))


_BUILTIN = {k: _TABLE["builtin"][k] for k in ("map_element", "fill_element")}

#: Host values: worker costs per access, record, element and cached-row
#: byte, memory bandwidth, and the built-in output's costs by dtype (`*`
#: where none was measured; the shipped table holds only `*`).
HOST: dict = {**_TABLE["host"], "builtin": {"*": _BUILTIN}}

#: A process function nothing was measured for.
PROCESS: dict = dict(_TABLE["process"])

#: Accesses per modeled bucket.
BUCKET: int = _TABLE["bucket"]
