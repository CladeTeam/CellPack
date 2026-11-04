"""Native loading: one data source, named output schemas and runs.

`Loader` binds a `DataSource`, the schemas every position fills and a
`LoaderConfig`, then compiles plans for an access sequence (offline, or as an
online run would), replays a compiled plan, sizes the output ring and starts
`Run`s; `load` is the one-shot dense convenience over the same machinery.
Decoding stays native: a run fills ring slots and Python only chooses
allocations and sequences, so nothing here touches cell data.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .. import _cellpack
from ._accesses import sequence
from ._config import IoBackend, LoaderConfig, available_memory, compile_threads
from ._ring import Ring, RingLayout
from ._schema import OutputSchema
from ._source import DataSource
from ._stats import MemoryEstimate
from .opt import CostModel, OnlineStats, Plan, PlanOptions, PlanSummary

#: A started run (native): iterate it, take other ranks' batches through
#: `consumer(rank)`, then `join()` for its `RunStats`.
Run = _cellpack.Run


class Loader:
    """One native execution frontier over a data source and named outputs.

    `outputs` is an OutputSchema, a sequence of schemas (named by their
    `name`, else their id) or a mapping from names to schemas; without one,
    a dense float32 row over every standard id named "X". `config` is a
    `LoaderConfig`, or give its fields as keywords (`workers`, `inflight`,
    `slot_rows`, ...).

    A run executes one access per output per position (in the order
    `p0s0 p0s1 ... p1s0`), each loading one cell: pass a uint32 array of the
    accesses' cells or an `Accesses` (`Accesses.replicate` gives every
    output the position's cell, `Accesses.offset` shifts each output's
    cell, e.g. to a paired matrix). Outputs land in one ring whose slots
    hold every output's rows of `slot_rows` consecutive positions. A run
    caches only as a plan says (an offline `Plan`, or `PlanOptions` compiled
    online while it runs); cached rows keep their stored type, shared by
    every output. Loaders pickle (to build the same loader in another
    process) when their source and outputs do.

    `source` and `config` are what the loader was built from, `schemas` maps
    each output name to its bound `OutputSchema`, and `names` lists the names
    in output order.
    """

    def __init__(self, source, outputs=None, *, config=None, **settings):
        """Validates the outputs and builds the native loader. An output's name
        is its mapping key, else its `name`, else its assigned id; explicit ids
        are kept and the rest take the smallest free ones. Raises TypeError
        for a source that is not a `DataSource` or a config given together
        with its fields, ValueError for no outputs, duplicate names or ids."""
        if config is not None and settings:
            raise TypeError("give a config or its fields as keywords, not both")
        if not isinstance(source, DataSource):
            raise TypeError("a loader reads one DataSource")
        if outputs is None:
            outputs = OutputSchema.dense(np.arange(source.vocab, dtype=np.uint32), name="X")
        if isinstance(outputs, OutputSchema):
            outputs = [outputs]
        pairs = list(outputs.items()) if isinstance(outputs, Mapping) else [(None, s) for s in outputs]
        if not pairs:
            raise ValueError("a loader needs at least one output schema")
        ids = _assign_ids([s.id for _, s in pairs])
        names = [n if n is not None else (s.name or str(i)) for (n, s), i in zip(pairs, ids)]
        if len(set(names)) != len(names):
            raise ValueError("output names must be distinct")
        self.schemas = MappingProxyType({n: s._bind(i, n) for (_, s), i, n in zip(pairs, ids, names)})
        self.names = tuple(names)
        self.source = source
        self._source_n_cells = source.n_cells
        self._source_direct_io = source.direct_io
        self._source_counts = source._row_counts()
        self.config = config if config is not None else LoaderConfig(**settings)
        self._handle = _cellpack.Loader(source._handle, [s._handle for s in self.schemas.values()], self.config)

    def __reduce__(self):
        """Pickles as (source, named schemas, config); the native loader is
        rebuilt from them on the other side."""
        return _loader, (self.source, dict(self.schemas), self.config)

    def __repr__(self) -> str:
        from reprlib import repr as short_repr

        outputs = {n: (f"{s.dtype}[{s.out_dim}]" if s.is_builtin else f"process[{s.row_bytes} bytes]")
                   for n, s in self.schemas.items()}
        return (f"Loader(n_cells={self._source_n_cells}, outputs={short_repr(outputs)}, "
                f"slot_rows={self.config.slot_rows}, workers={self.config.workers}, "
                f"backend={IoBackend(self.config.backend).value!r}, direct_io={self._source_direct_io})")

    @property
    def window(self) -> int:
        """The execution window in slots of runs without a plan."""
        return self._handle.window

    @property
    def cores(self) -> float:
        """Physical cores the workers run on (among the pinned CPUs, else the
        allowed ones), at most one per worker: a cost model's CPU capacity."""
        return self._handle.cores

    @property
    def accesses_per_slot(self) -> int:
        """Accesses a slot holds: `slot_rows` positions of every output."""
        return self._handle.accesses_per_slot

    def ring_layout(self, *, slots=4, world=1) -> RingLayout:
        """The layout of this loader's ring: `slots` slots of `slot_rows`
        positions shared by `world` consumer ranks (batch `b` is rank
        `b % world`'s). It sizes caller-owned or shared memory and pickles to
        other processes."""
        # (name, view dtype, bytes per row) per output, in schema order.
        outputs = [(n, s.view_dtype, s.row_bytes) for n, s in self.schemas.items()]
        return RingLayout(slots, self.config.slot_rows, outputs, world)

    def ring(self, *, slots=4, world=1) -> Ring:
        """A natively allocated ring for this loader's outputs."""
        return Ring(self.ring_layout(slots=slots, world=world))

    def plan_options(self, cache_bytes, *, cost="auto") -> PlanOptions:
        """Plan options for `cache_bytes` of cache, with this loader's window
        as the reuse gap and `cost`: "auto" for this machine's cost model
        (`pycellpack.loader.costs.resolve`: measured records where they exist,
        defaults with a warning elsewhere), a `CostModel`, or None (every page
        hit counts the same)."""
        if isinstance(cost, str):
            if cost != "auto":
                raise ValueError("cost is 'auto', a CostModel or None")
            from .costs import resolve

            cost = resolve(self)
        return PlanOptions(cache_bytes=int(cache_bytes), gap=self.window, cost=cost)

    def _options(self, options, *, online) -> PlanOptions:
        """Options with the gap and compiler threads resolved for this loader."""
        if not isinstance(options, PlanOptions):
            options = self.plan_options(options)
        workers = int(self.config.workers or 1)
        # An unset thread count follows this loader: one per usable CPU offline,
        # a quarter of the workers online, where compilers share worker cores.
        threads = options.threads if options.threads is not None else compile_threads(workers, online=online, cpus=options.cpus)
        return dataclasses.replace(options, gap=self.window if options.gap is None else options.gap, threads=threads)

    def compile(self, accesses, options, *, online=False) -> Plan:
        """Compiles a cache plan for a stored sequence (an array or an
        `Accesses`, materialized): offline (the whole sequence is the
        horizon), or as an online run would (a finite horizon, frozen and
        published chunk by chunk; `plan.online_stats` holds its statistics).
        `options` are `PlanOptions` or a cache budget in bytes."""
        handle, stats = self._handle.compile(sequence(accesses, counts=self._source_counts), self._options(options, online=online), online=online)
        return Plan(handle, None if stats is None else OnlineStats.from_dict(stats))

    def compile_stats(self, accesses, options) -> OnlineStats:
        """The statistics of compiling `accesses` (an array or an `Accesses`)
        online, without loading them or keeping the plan: what an online run
        would follow, its modeled time among them."""
        return OnlineStats.from_dict(self._handle.compile_stats(sequence(accesses, counts=self._source_counts), self._options(options, online=True)))

    def replay(self, plan, accesses, *, cost: CostModel | None = None) -> PlanSummary:
        """Replays a plan's actions serially, priced by its own cost model or
        `cost`; the plan and its stored summary are unchanged."""
        return PlanSummary.from_dict(self._handle.replay(plan._handle, sequence(accesses, counts=self._source_counts), cost=cost))

    def memory_estimate(self, plan=None, *, slots=4, world=1) -> MemoryEstimate:
        """Known reservations of a run with `plan`: a `Plan`, `PlanOptions`
        (with their gap, chunk and row mode), a cache budget in bytes for
        default page planning, or None for no plan."""
        if isinstance(plan, Plan):
            d = self._handle.memory_estimate(slots, world, plan=plan._handle)
        elif isinstance(plan, PlanOptions):
            d = self._handle.memory_estimate(slots, world, options=self._options(plan, online=True))
        else:
            d = self._handle.memory_estimate(slots, world, cache_bytes=int(plan or 0))
        return MemoryEstimate(**d)

    def start(self, accesses, *, plan=None, ring=None, slots=None, world=None, rank=0) -> Run:
        """Starts loading `accesses` (a uint32 array of the accesses' cells,
        or an `Accesses`; the native side keeps its own copy of an array).

        `plan` is an offline `Plan` (for the sequence it was compiled for),
        `PlanOptions` or a cache budget in bytes to compile the plan online
        while the run goes, or None (no cache). Outputs go to `ring`, or to a
        new ring of `slots` slots (default 4) shared by `world` ranks
        (default 1); this process consumes rank `rank` by iterating the run.

        Warns (RuntimeWarning) when the run's known reservations
        (`memory_estimate`) exceed the memory still available to the process.
        """
        if ring is not None and (slots is not None or world is not None):
            raise TypeError("slots and world describe a new ring; a given ring has its own")
        own = ring is not None
        if ring is None:
            ring = self.ring(slots=4 if slots is None else slots, world=1 if world is None else world)
        if not isinstance(ring, Ring):
            raise TypeError("ring must be a Ring")
        if ring.layout != self.ring_layout(slots=ring.slots, world=ring.world):
            raise ValueError("ring layout must match the loader's output names, dtypes and row sizes")
        options = None
        if plan is not None and not isinstance(plan, Plan):
            # Anything but a compiled plan describes a plan to compile online
            # while the run goes.
            options, plan = self._options(plan, online=True), None
        self._check_memory(plan if plan is not None else options, ring.slots, ring.world, own)
        return self._handle.start(sequence(accesses, counts=self._source_counts), ring, rank=rank, plan=None if plan is None else plan._handle, options=options)

    def _check_memory(self, plan, slots, world, own_ring):
        """Warns when a run's known reservations exceed the memory this process
        may still allocate. A caller's ring is already allocated, so its bytes
        are left out of the estimate compared with what is available."""
        available = available_memory()
        if available is None:
            return
        e = self.memory_estimate(plan, slots=slots, world=world)
        # A caller's ring is already allocated.
        need = e.known_reservation_bytes - (e.output_ring_bytes if own_ring else 0)
        if need > available:
            gib = 1 << 30
            cache = 0 if plan is None else plan.cache_bytes
            warnings.warn(
                f"this run reserves about {need / gib:.2f} GiB (cache budget {cache / gib:.2f} GiB, "
                f"private frames {e.private_frame_bytes / gib:.2f} GiB) but only {available / gib:.2f} GiB "
                "of memory is available; lower the cache budget, workers or inflight",
                RuntimeWarning,
                stacklevel=3,
            )


def _loader(source, outputs, config):
    """The unpickling half of `Loader.__reduce__`."""
    return Loader(source, outputs, config=config)


def _assign_ids(ids):
    """Keep explicit schema ids and give the others the smallest free ones."""
    explicit = [i for i in ids if i is not None]
    if len(set(explicit)) != len(explicit):
        raise ValueError("output schema ids must be distinct")
    taken, free, result = set(explicit), 0, []
    for i in ids:
        if i is None:
            while free in taken:
                free += 1
            i = free
            taken.add(i)
        result.append(i)
    return result


def load(source, cells=None, *, out=None, config=None, **settings) -> np.ndarray:
    """Loads a DataSource's cells (default: all) into one dense float32 array
    over its standard ids, through the native concurrent loader; cell order
    and repeats are preserved. `out` (C-contiguous float32, 8-byte aligned)
    receives them; `config` sets the loader, or pass its fields as keywords
    when `config` is None."""
    from ..cellpack.selection import normalize_rows

    if not isinstance(source, DataSource):
        raise TypeError("load reads a DataSource: (source, ids) pairs with their vocab")
    if config is not None and settings:
        raise TypeError("give a config or its fields as keywords, not both")
    selected = normalize_rows(cells, source.n_cells).astype(np.uint32)
    shape = (len(selected), source.vocab)
    if out is None:
        out = np.empty(shape, np.float32)
    if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != shape or not out.flags.c_contiguous or not out.flags.writeable:
        raise ValueError("out requires a writable C-contiguous float32 array with selected shape")
    if out.ctypes.data % 8:
        raise ValueError("out must be 8-byte aligned (the ring's data alignment)")
    if not len(selected):
        return out
    if config is None:
        config = LoaderConfig(**{**settings, "slot_rows": len(selected)})
    else:
        config = dataclasses.replace(config, slot_rows=len(selected))
    # One slot holds every row: the output array is the ring's data. `out` is
    # C-contiguous and 8-byte aligned and holds `slot_rows` positions of the
    # one schema, so its bytes are exactly one slot.
    loading = Loader(source, config=config)
    ring = Ring(loading.ring_layout(slots=1), data=out.reshape(-1).view(np.uint8))
    with loading.start(selected, ring=ring) as run:
        with next(run):
            pass
        run.join()
    return out
