"""A loader's cost model, assembled from stored records and defaults.

`keys` derives the three record identities a model is looked up by: the host's
worker constants, the storage's device rates, and each process function's code
and output shape. `resolve` fetches the matching records and replaces every
missing part with `pycellpack.loader.costs` defaults, warning once per missing record;
`compose` turns the three parts plus the loader's core count into a
`CostModel`. `Loader.plan_options(cost="auto")` resolves this way.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import fields, replace

from ..opt import CostModel, InputCost, OutputCost, RowCost
from . import _defaults
from ._code import Mappings
from ._keys import digest, host_identity, host_key, output_identity, storage_identity
from ._store import CostStore

_warned: set = set()


def _warn_once(tag, message):
    """Warns once per tag in this process: a run with several unmeasured parts
    must not repeat each warning."""
    if tag not in _warned:
        _warned.add(tag)
        warnings.warn(message, RuntimeWarning, stacklevel=4)


def keys(loading, mappings: Mappings | None = None) -> dict:
    """The record keys a loader's cost model is looked up by, with their
    identities and the storage tier."""
    source, config = loading.source, loading.config
    storage, tier = storage_identity(source.paths, backend=config.backend.value, direct_io=source.direct_io, page_size=source.page_size)
    mappings = Mappings() if mappings is None else mappings
    outputs = {}
    for name, schema in loading.schemas.items():
        identity = output_identity(schema, mappings)
        outputs[name] = None if identity is None else (digest(identity), identity)
    return {
        "host": (host_key(), host_identity()),
        "storage": (digest(storage), storage),
        "tier": tier,
        "outputs": outputs,
    }


def _builtin(host_values, dtype):
    """The built-in output's costs for `dtype`, else the `*` entry (the one
    measured for whatever dtype calibration exercised)."""
    table = host_values.get("builtin", {})
    return table.get(dtype) or table.get("*") or _defaults.HOST["builtin"]["*"]


def _cost(kind, values):
    """Read a cost record's known fields; unrelated fields do not change it."""
    names = {f.name for f in fields(kind)}
    return kind(**{k: v for k, v in values.items() if k in names})


def resolve(loading, *, store: CostStore | None = None) -> CostModel:
    """The cost model of `loading` on this machine.

    `$CELLPACK_COST_MODEL` names a saved model that replaces everything when
    its outputs match the loader's schemas (the same names, or the same
    count); when it does not, that model is ignored with a warning. Otherwise
    each part comes from the store when a record matches (the host's worker
    constants, the storage's device rates, each process function's costs) and
    from `pycellpack.loader.costs` defaults when none does, which warns once per
    missing record. `cores` is the loader's (physical cores its workers run
    on).
    `provenance` names each part's source.
    """
    names = list(loading.schemas)
    path = os.environ.get("CELLPACK_COST_MODEL")
    if path:
        model = CostModel.load(path)
        if list(model.outputs) == names or len(model.outputs) == len(names):
            outputs = dict(zip(names, model.outputs.values()))
            return replace(model, cores=float(loading.cores), outputs=outputs, provenance={"model": f"file:{path}"})
        _warn_once(("file", path), f"ignoring CELLPACK_COST_MODEL={path}: its outputs do not match the loader's schemas")
    store = CostStore() if store is None else store
    k = keys(loading)
    provenance = {}

    host_k, _ = k["host"]
    record = store.get("host", host_k)
    host = record["values"] if record else _defaults.HOST
    provenance["host"] = f"store:host/{host_k}" if record else "default"
    if not record:
        _warn_once(("host", host_k), "no measured host costs for this CPU and build; using defaults "
                   "(run pycellpack.loader.costs.calibrate to measure them)")

    storage_k, storage_id = k["storage"]
    record = store.get("storage", storage_k)
    storage = record["values"] if record else _defaults.storage(k["tier"])
    provenance["storage"] = f"store:storage/{storage_k}" if record else f"default:{k['tier']}"
    if not record:
        _warn_once(("storage", storage_k), f"no measured device costs for {storage_id['fs'] or 'this'} storage "
                   f"({storage_id['source'] or 'unknown source'}); using {k['tier']} defaults")

    outputs, sources = {}, {}
    for name, schema in loading.schemas.items():
        entry = k["outputs"][name]
        if entry is None:
            outputs[name] = _cost(OutputCost, _builtin(host, schema.dtype.name))
            sources[name] = provenance["host"]
            continue
        output_k, _ = entry
        record = store.get("output", output_k)
        values = record["values"] if record else _defaults.PROCESS
        outputs[name] = _cost(OutputCost, values)
        sources[name] = f"store:output/{output_k}" if record else "default"
        if not record:
            _warn_once(("output", output_k), f"no measured costs for the process function of output {name!r}; using defaults")
    provenance["outputs"] = sources
    return compose(loading, host, storage, outputs, provenance)


def compose(loading, host: dict, storage: dict, outputs: dict, provenance: dict | None = None) -> CostModel:
    """A cost model from host values, storage values and per-output costs;
    values a record lacks are the defaults'."""
    d = _defaults.HOST
    host = {**d, **host, "input": {**d["input"], **host.get("input", {})}, "rows": {**d["rows"], **host.get("rows", {})}}
    storage = {**_defaults.storage(_defaults.UNKNOWN_TIER), **storage}
    return CostModel(
        cores=float(loading.cores),
        iops=storage["iops"],
        bandwidth=storage["bandwidth"],
        memory_bandwidth=host["memory_bandwidth"],
        serial=storage["serial"],
        bucket=_defaults.BUCKET,
        access=host["access"],
        request=storage["request"],
        compile=host["compile"],
        input=_cost(InputCost, host["input"]),
        rows=_cost(RowCost, host["rows"]),
        outputs=outputs,
        convert=tuple(host["convert"]),
        provenance=provenance or {},
    )
