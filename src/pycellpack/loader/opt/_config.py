"""Cache planning options and measured cost models.

`PlanOptions` is what a plan is compiled for: the cache budget, the reuse gap
and execution window, whether rows are cached and how they are chosen, block
reads, the online horizon and chunk, compiler threads and affinity.
`CostModel` says what one machine's resources are and what a unit of work
costs, with one `OutputCost` per output; its field set is the native model's
(`cellpack::opt::model`), and its JSON form (`COST_FORMAT`) is what the Rust
crate reads. Both are frozen: a loader resolves unset fields with
`dataclasses.replace`, never by mutation.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from ..._cellpack import cost_defaults

_DEFAULTS = cost_defaults()

#: Conversion classes priced by `CostModel.convert`, in order.
CONVERSIONS = ("same", "widen", "reinterpret", "int_to_f32", "int_to_f64", "f32_to_f64")


@dataclass(frozen=True)
class InputCost:
    """Core-seconds of decoding records: per decoded record, per stored
    element decoded, and per decoded record in a chunk not among the few
    most recently used (cold decode tables and maps)."""

    decode: float = 0.0
    decode_element: float = 0.0
    cold: float = 0.0


@dataclass(frozen=True)
class RowCost:
    """Core-seconds of cached rows, per payload byte: gathering a row from
    its frames for a hit, and building and writing it on admission."""

    gather: float = 0.0
    admit: float = 0.0


@dataclass(frozen=True)
class ProcessCost:
    """Per-call costs through this inclusive element bound: the stored
    elements a CSR row carries, or the output dims a dense row fills. A
    curve's bounds increase and its last covers ``2**32 - 1``."""

    max_elements: int
    process: float
    resolve: float = 0.0


@dataclass(frozen=True)
class OutputCost:
    """Core-seconds of one output: its process function per call (none for
    the built-in output), per element mapped to its output dims and written,
    and per output dim filled with the default (dense rows). A partial call
    adds `resolve` per call and `resolve_element` per stored element; a
    non-empty `curve` replaces the fixed process and resolver costs with
    size bands."""

    process: float = 0.0
    map_element: float = 0.0
    fill_element: float = 0.0
    resolve: float = 0.0
    resolve_element: float = 0.0
    curve: tuple[ProcessCost, ...] = ()

    def __post_init__(self):
        """Normalizes `curve` entries: `ProcessCost` objects stay, mappings and
        sequences become them."""
        object.__setattr__(self, "curve", tuple(
            b if isinstance(b, ProcessCost) else ProcessCost(**b) if isinstance(b, Mapping) else ProcessCost(*b)
            for b in cast(tuple[Any, ...], self.curve)))


@dataclass(frozen=True)
class CostModel:
    """The loader's resources on one machine and the work it does (see
    `cellpack::opt::model`).

    Four resources serve a run: the device's read requests (`iops`) and
    bytes (`bandwidth`), the workers' physical cores (`cores`) and main
    memory traffic of cached rows (`memory_bandwidth`). Work is in
    core-seconds: per access (`access`), per read request (`request`), per
    access the online compiler takes from the workers' cores (`compile`,
    online plans only), decoding (`input`), cached rows (`rows`) and each
    output (`outputs`, keyed by the loader's output names, or by position;
    `convert` adds a per-element cost for each conversion class,
    `CONVERSIONS`). A window of `bucket` accesses takes its device time (the
    longer of requests and bytes) or its core time, whichever is longer, plus
    `serial` of the shorter; a longer memory time overrides that sum.
    `provenance` records where each part came from (see
    `pycellpack.loader.costs.resolve`); it is not compared. The rates are positive
    and finite, costs finite and nonnegative, `serial` in `[0, 1]`, `bucket`
    at least 1, and `outputs` has one entry per loader output; compilation
    and replay reject a model that breaks this.

    `to_dict`/`save` write JSON in the native field names (outputs as a list
    in mapping order, each with its `name`), which the Rust crate reads with
    its `serde` feature.
    """

    cores: float
    iops: float
    bandwidth: float
    memory_bandwidth: float = _DEFAULTS["host"]["memory_bandwidth"]
    serial: float = 0.0
    bucket: int = _DEFAULTS["bucket"]
    access: float = 0.0
    request: float = 0.0
    compile: float = 0.0
    input: InputCost = field(default_factory=InputCost)
    rows: RowCost = field(default_factory=RowCost)
    outputs: Mapping = field(default_factory=dict)
    convert: tuple = (0.0,) * len(CONVERSIONS)
    provenance: Mapping = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        """Copies the mappings into read-only views, builds `InputCost` and
        `RowCost` from their mapping forms, and checks that one conversion cost
        is given per class. The dataclass is frozen, so fields are replaced
        through `object.__setattr__`."""
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        if isinstance(self.input, Mapping):
            object.__setattr__(self, "input", InputCost(**self.input))
        if isinstance(self.rows, Mapping):
            object.__setattr__(self, "rows", RowCost(**self.rows))
        if isinstance(self.convert, Mapping):
            object.__setattr__(self, "convert", tuple(float(self.convert.get(n, 0.0)) for n in CONVERSIONS))
        object.__setattr__(self, "convert", tuple(self.convert))
        if len(self.convert) != len(CONVERSIONS):
            raise ValueError("one conversion cost per class")

    def to_dict(self) -> dict:
        """JSON-ready fields (see the class documentation)."""
        rates = {f.name: getattr(self, f.name) for f in fields(self)
                 if f.name not in ("input", "rows", "outputs", "convert", "provenance")}
        # Rates first, then the nested costs; outputs stay a list so the file
        # keeps the mapping's order.
        return {
            "format": COST_FORMAT,
            **rates,
            "input": {f.name: getattr(self.input, f.name) for f in fields(InputCost)},
            "rows": {f.name: getattr(self.rows, f.name) for f in fields(RowCost)},
            "outputs": [{"name": name, **_output_dict(cost)} for name, cost in self.outputs.items()],
            "convert": list(self.convert),
        }

    @classmethod
    def from_dict(cls, d) -> CostModel:
        """Reads `to_dict`; outputs without a name are keyed by position, and
        fields this version does not know are ignored."""
        if d.get("format", COST_FORMAT) != COST_FORMAT:
            raise ValueError(f"unsupported cost model format {d.get('format')}")
        names = {f.name for f in fields(cls)} - {"input", "rows", "outputs", "convert", "provenance"}
        outputs = {}
        for i, o in enumerate(d.get("outputs", ())):
            o = dict(o)
            outputs[o.pop("name", i)] = OutputCost(**{k: v for k, v in o.items() if k in _OUTPUT_FIELDS})
        known = lambda kind, v: {k: x for k, x in v.items() if k in {f.name for f in fields(kind)}}
        return cls(
            **{k: d[k] for k in names if k in d},
            input=InputCost(**known(InputCost, d.get("input", {}))),
            rows=RowCost(**known(RowCost, d.get("rows", {}))),
            outputs=outputs,
            convert=tuple(d.get("convert", (0.0,) * len(CONVERSIONS))),
        )

    def save(self, path) -> Path:
        """Writes `to_dict` as JSON, atomically (a temporary file renamed)."""
        path = Path(path)
        _atomic_json(path, self.to_dict())
        return path

    @classmethod
    def load(cls, path) -> CostModel:
        """A model read from a JSON file written by `save`.
        `$CELLPACK_COST_MODEL` names such a file for `pycellpack.loader.costs.resolve`."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


#: Version of the cost model's JSON form.
COST_FORMAT = 1

_OUTPUT_FIELDS = ("process", "map_element", "fill_element", "resolve", "resolve_element", "curve")


def _output_dict(cost: OutputCost) -> dict:
    """One output's costs in the native field names, its curve as a list of
    band records."""
    d = {k: getattr(cost, k) for k in _OUTPUT_FIELDS if k != "curve"}
    d["curve"] = [{"max_elements": b.max_elements, "process": b.process, "resolve": b.resolve} for b in cost.curve]
    return d


def _atomic_json(path: Path, value) -> None:
    """Writes JSON through a sibling temporary file renamed into place, so a
    reader never sees a partial model or record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=1, sort_keys=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class PlanOptions:
    """What a plan is compiled for. `gap` is the reuse gap in slots (default:
    the loader's window); runs of the plan execute within a window of at most
    it, and every residency reserves its frames from that far before its
    admission. `rows` caches rows too (page+row; needs a cost model), chosen
    by `row_policy`: "incremental" (each reuse decided as it arrives, against
    a shadow price of capacity learned from the pages) or "global" (rows
    first over the whole stored sequence; offline only). `block_pages`
    merges adjacent admissions into reads of 2 to 128 pages (needs a cost
    model). `horizon` bounds, in slots, the reuse intervals an online plan
    considers (at least `gap`); `chunk` is how many accesses the online
    compiler advances, freezes and publishes at a time. `objective` minimizes
    the modeled time ("time", the default) or page misses ("misses"); `cost`
    prices the decisions and is required for `rows` and `block_pages`.

    `threads=None` resolves when a loader uses the options: offline, one
    per usable CPU (`cpus` if pinned); online, a quarter of the loader's
    workers; both at most 16. `cpus` pins the compiler threads (empty
    inherits the calling thread's CPUs)."""

    cache_bytes: int
    gap: int | None = None
    threads: int | None = None
    objective: str = "time"
    cost: CostModel | None = None
    rows: bool = False
    block_pages: int | None = None
    row_policy: str = "incremental"
    horizon: int | None = None
    chunk: int | None = None
    cpus: tuple[int, ...] = ()

    def __post_init__(self):
        """Copies `cpus` to a tuple and validates the enumerations, the
        positive counts and the CPU indices; the native layer checks the rest
        against the loader (window, horizon, cache size)."""
        object.__setattr__(self, "cpus", tuple(self.cpus))
        if any(cpu < 0 for cpu in self.cpus):
            raise ValueError("compiler CPU indices must be nonnegative")
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be positive")
        if self.objective not in {"time", "misses"}:
            raise ValueError("objective must be time or misses")
        if self.row_policy not in {"incremental", "global"}:
            raise ValueError("row_policy must be incremental or global")
        if self.chunk is not None and self.chunk < 1:
            raise ValueError("chunk must be positive")
