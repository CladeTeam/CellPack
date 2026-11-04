# Statistics and cost records API

Use run statistics to check delivered work, selected timing sites to investigate
execution, and cost records to price future plans. These are separate observations:
a cache hit count or a modeled cost is not an end-to-end throughput measurement.

```python
from pycellpack.loader import LoaderConfig, RunStats, MemoryEstimate, profile_sites, profile_groups
from pycellpack.loader import costs
```

<DocChoices title="Choose the information you need" :items="[
  { label: 'RunStats and ProfileReport', when: 'Check what a run did and where it spent time.', detail: 'Read final counters first; enable only the timing sites you need.', href: '#read-a-run-result' },
  { label: 'MemoryEstimate', when: 'Size known native reservations before starting.', detail: 'Include output memory and private scratch alongside the planned cache.', href: '#estimate-memory' },
  { label: 'costs.resolve / calibrate', when: 'Supply costs for cache decisions.', detail: 'Resolve stored values or defaults; run actual measurements only when requested.', href: '#resolve-a-cost-model' },
]" />

## Read a run result

`run.snapshot()` returns the currently published RunStats without waiting for
completion; it requires an open Run. Workers publish periodically, so small or
recent runs can still show zero counters. The snapshot is not a barrier across
threads. Consume output before `run.join()`, which waits for completion and
returns final statistics or propagates failure. See [run lifecycle](rings.md).

| Record / fields | Meaning and unit |
| --- | --- |
| `RunStats.accesses` | Schema deliveries: positions × schemas, not unique cells or batches |
| `decoded_rows`, `mapped_rows`, `row_hits`, `process_calls` | Work and reuse counters; a shared input can serve several outputs |
| `io: IoStats` | Issued requests, requested/returned bytes and request counts by page-size band |
| `outputs: dict[str, OutputStats]` | Access, decode, row-hit and process counts keyed by output name |
| `elapsed`, `setup_elapsed` | Execution wall time after setup, and setup time, in seconds |
| `cache_bytes`, `private_bytes` | Cache-frame and worker-private-frame reservations, in bytes |
| `profile`, `online` | Optional timing report and online compiler statistics |

Python exposes counters directly on RunStats, rather than under a `counters`
attribute. `to_dict()` produces nested JSON-ready values; `RunStats.from_dict(d)`
restores typed I/O, output, profile and online records. These dataclasses are frozen;
use the returned records rather than changing counters to represent another run.

I/O bytes count application reads, including reads served by the OS page cache;
they are not physical-device bytes. `io.sizes` counts requests in bands of
`1`, `2`, `3–4`, `5–8`, `9–16`, `17–32`, and `33+` pages. Its entries sum to
`io.requests`. The [planning guide](/python/docs/planning.md) shows a complete run
and how to compare its values and final statistics.

## Select and interpret timing

| `LoaderConfig.profile` | Behavior |
| --- | --- |
| `None` or `False` | Untimed execution; event counters remain available |
| `True` or `"all"` | Time every available site |
| Site/group string or sequence of names | Time the selected sites; unknown names raise ValueError |
| Empty sequence | Timed implementation with no selected sites; use None for an untimed baseline |

`profile_sites()` maps site names to descriptions. `profile_groups()` maps group
names to member names, which may include other groups. Names are case-sensitive;
the native parser prefers a site over a same-named group. In particular,
`profile="worker"` selects the worker-lifetime site. Expand the group explicitly
when you want its stage coverage:

```python
config = LoaderConfig(profile=profile_groups()["worker"])
```

ProfileReport contains `schemas`, process-size `bands`, and `sites` keyed by name.
Every selected site can appear even when its call count is zero. A missing site
is not evidence that the operation took no time.

| Query | Result |
| --- | --- |
| `report.site(name)` | SiteReport or None when unrecorded/unknown |
| `report.time(*names)` | Sum of named site seconds; absent sites contribute zero |
| `report.map_time(schema)` | Mapping seconds for a zero-based schema index |
| `report.process_bands(schema)` | Full/partial calls and seconds for each element-count band |
| `report.process_key(schema, band, partial)` | Integer detail key; band is an index into `report.bands` |

All Python profile durations are **seconds**, including SiteReport `time`,
`self_time`, detail tuples `(key, calls, seconds)`, and histogram lower bounds
`(seconds, count)`. Process-size bounds count elements, not time. Schema indices
follow loader output order; they are not application schema IDs or output names.

`time` includes child sites; `self_time` subtracts only recorded direct children,
clamped at zero. If a child was not timed, its work remains in the parent's self
time. Durations aggregate across threads and can exceed wall time. Neither
summing nested site times nor treating `io.latency` as device-only latency gives
an end-to-end elapsed time. Compare profiled work with the same untimed workload
when assessing instrumentation overhead.

## Estimate memory

`loading.memory_estimate(plan=..., slots=..., world=...)` returns MemoryEstimate
for the selected planning mode and output layout. `to_dict()` reports byte
reservations for cache frames, generation words, worker-private frames, scratch,
PlanRing, known metadata and the output ring; `known_reservation_bytes` is their
sum. Match the plan, slot count and world size to the intended run.

Compiler state, allocator overhead and OS page cache are excluded. This is neither
measured RSS nor a hard process-memory limit. Cache capacity alone does not bound
the loader's complete memory use.

## Resolve a cost model

<DocDiagram name="cost-workflow" />

`costs.resolve(loading, store=None)` returns a CostModel without running a
calibration. Its `provenance` identifies each source; inspect it before treating
the model as measured for the current workload.

1. If `CELLPACK_COST_MODEL` names a saved model, load it. Matching output names,
   **or merely matching output count**, admit that model; outputs are then assigned
   to loader names in order and `cores` is replaced by `loading.cores`. Verify the
   order yourself. A count mismatch warns and continues with record lookup;
   explicit file read/parse errors propagate.
2. Otherwise, fetch matching host, storage and process-output records from the
   CostStore. Each missing part uses package defaults and warns once per record
   in this process. Built-in output costs come from the host's dtype entry,
   falling back to its generic entry.
3. Assemble costs in loader output order, with the loader's physical core count.

`costs.keys(loading)` returns `host`, `storage`, `tier` and named `outputs` lookup
identities. Host identity covers CPU/build/kernel context; storage identity covers
the largest source file's mount, backend, direct-I/O mode and page size. A process
output identity includes callback code, declared identity and output shape.
Built-in outputs have no separate output-record key. `costs.compose(...)` assembles
an explicit model from supplied records/defaults; it does not measure them.

Cost rates use requests/second and bytes/second. Work coefficients use
core-seconds per stated operation, element or byte. See [planning](planning.md)
for the cost-model fields and modeled-time contract.

## Record observations or calibrate

| Operation | Work and side effect | Result |
| --- | --- | --- |
| `costs.record_run(loading, stats, store=None)` | Merge eligible `process` timing observations into process-output records | Number of records updated; warns and returns 0 on failure |
| `costs.calibrate(loading, cells=None, ...)` | Pilot plus repeated timed/untimed loader runs, plan compilation and fitting; saves accepted record parts by default | CostModel with provenance |

`record_run` needs a run that selected the `process` site and actually called a
process function. It does not save all RunStats or calibrate host/storage rates.
It converts worker seconds to core-seconds, applies measured profiling correction
when available, and merges per-size-band samples with decaying earlier weights.
Without eligible observations, it returns zero. Recording failures do not fail the
completed loading run; updates across several records are not one transaction.

Calibration uses a subset of supplied global cells, or the source's full cell
range when `cells=None`, and measures seven configurations. Its principal options
are:

| Option | Default | Contract |
| --- | --- | --- |
| `steady_seconds` | `30.0` | Required steady measurement window per configuration, excluding warm-up |
| `max_warmup` | None | Steady-state detector's warm-up allowance |
| `budget` | None | Cache bytes; derived from subset footprint when unset, capped at one quarter of available memory |
| `save` / `force` | True / False | Write accepted parts; force permits saving parts whose quality checks failed |
| `store` | None | Use the default CostStore |
| `seed` / `verbose` | `0` / True | Subset/order seed and stderr progress |

Runtime includes the pilot, warm-up, retries, planning and measurement windows;
`steady_seconds=30` is not a 30-second total-runtime bound. Saving is judged per
host, storage and output part, so one calibration can retain existing values for
some parts. Buffered I/O can measure page-cache behavior rather than storage.
Choose representative cells and resource limits before deliberately starting it.

## CostStore persistence

`CostStore(root=None)` treats root as a **home directory** and appends
`costs/v1`. The default home is `CELLPACK_HOME`, or `~/.cellpack`. Record kinds
are `host`, `storage` and `output`; files contain identity, values, measurement
metadata and timestamps.

`get(kind, key)` returns a record or None; unreadable or mismatched records warn
and count as absent. `records()` iterates readable records by kind then key.
`put` replaces one record, `update` performs a locked read-modify-write, and
`remove` deletes one record, returning False when absent. Writes use an atomic
file replacement under a store lock. These mutation methods change local cost
records, not matrix files.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/statistics-members.md-->

:::
