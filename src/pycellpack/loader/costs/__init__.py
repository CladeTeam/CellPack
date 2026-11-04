"""Measured cost models: calibration, storage and lookup.

`resolve(loader)` assembles the loader's cost model from records under
`$CELLPACK_HOME` (default `~/.cellpack`): the host's worker constants (CPU
and build), the storage's device rates (filesystem, backend, O_DIRECT, page
size) and each process function's costs (its code, identity and output
shape), falling back to defaults with a warning where nothing was measured.
`Loader.plan_options` uses it. `calibrate(loader)` measures all three from
profiled runs; `record_run` merges the process-call profile of a finished run
that recorded the `process` site into the process functions' records.
`python -m pycellpack.loader.costs` lists, shows, removes, exports and imports
records and runs calibrations.
"""

from ._calibrate import calibrate
from ._code import signature
from ._collect import record_run
from ._keys import host_identity, host_key, storage_identity, storage_tier
from ._resolve import compose, keys, resolve
from ._steady import Steady, Window
from ._store import CostStore, home

__all__ = [
    "CostStore",
    "Steady",
    "Window",
    "calibrate",
    "compose",
    "home",
    "host_identity",
    "host_key",
    "keys",
    "record_run",
    "resolve",
    "signature",
    "storage_identity",
    "storage_tier",
]
