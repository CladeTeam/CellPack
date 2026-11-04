# Plans and costs: opt

`cellpack::opt` represents cache residency, compiler options, machine costs,
modeled loads and serial reference policies. The application's bound
`Loader` supplies the graph and high-level compilation entry points.
The runtime executes the resulting plan. Online compilation makes decisions
ahead of the workers, rather than on each worker cache lookup. For a complete
program and mode selection, use the [cache planning guide](planning.md).

<DocChoices title="Look up the contract you need" :items="[
  { label: 'PlanOptions', when: 'Configure compilation', detail: 'Budget, objective, reuse gap, horizon and compiler resources.', href: '#planoptions' },
  { label: 'Plan and Residency', when: 'Inspect or persist a fixed schedule', detail: 'Bound identity, frame assignments and compatibility checks.', href: '#plan-and-residency' },
  { label: 'CostModel', when: 'Price work for a machine', detail: 'Resource capacities and per-operation core-second costs.', href: '#costmodel-and-resource-accounting' },
  { label: 'PageTrace', when: 'Compare with a serial page reference', detail: 'Page-only optimum with bypass and its limited scope.', href: '#serial-page-reference' },
]" />

## Compilation and replay

| Loader method | Input | Return / effect |
|---|---|---|
| `plan_options` | Cache bytes | Options with loader-derived gap and default costs |
| `compile` | Stored `&[CellId]`, options | Owned offline `Plan` |
| `replay` | Plan and exact stored sequence | `PlanSummary` without record loading |
| `replay_with_cost` | Same plus another cost model | Repriced summary, unchanged decisions |
| `compile_online` | Stored sequence, finite-horizon options | Collected plan and `OnlineStats` |
| `plan_online` | General `Accesses`, options | Online compiler statistics without loading |
| `online` | Validated options | `Planning::Online` for a run |

Offline plans require a stored sequence. Its length must be divisible by the
schema count; each value is a global `CellId`. Online planning accepts computed
`Accesses` with the same ordering and determinism requirements. A collected online
Plan still binds to the exact stored sequence supplied to `compile_online`.
Compiler thread count affects sharding and can affect decisions; scheduling order
of the same compilation does not define the policy.

## PlanOptions

`PlanOptions::new(cache_bytes, gap)` uses 16 compiler threads, the `Time`
objective, no cost model, no rows or block merging, `Incremental` row policy,
and no explicit horizon or chunk. `Loader::plan_options` additionally derives the
gap from the loader and supplies an NVMe default model with one output-cost
entry per schema. It is the usual starting point for a loader.

### Units and validation

| Field | Unit and accepted values |
| --- | --- |
| `cache_bytes` | Bytes; usable frames are `cache_bytes / page_size`, at most `u32::MAX` |
| `gap` | Positive slots; `gap × slot_cells × schemas` must be below `u32::MAX` accesses |
| `horizon` | Optional slots; when explicit, at least `gap` |
| `chunk` | Optional positive accesses; None uses `DEFAULT_CHUNK = 16384` |
| `threads` | Compiler count; the engine treats zero as one |
| `cpus` | Optional logical CPU IDs; empty inherits affinity, invalid or clamped CPU sets fail |
| `block_pages` | None or 2–128 pages; requires a cost model |

The execution window is in slots and cannot exceed the reuse gap. Unlike the
options, `Plan::distance()` and `OnlineStats::{distance, horizon, chunk}` use
accesses. A slot contains `slot_cells × schemas` accesses.

Offline compilation uses the whole sequence as its horizon. Online compilation
starts from the configured horizon, or twice the source cell count times the
schema count in accesses. It raises that value to at least `distance + chunk`
and clips it to the sequence length (`max(1)` for an empty sequence). The returned
OnlineStats reports this effective horizon.

### Objectives and retained objects

| Choice | Contract |
|---|---|
| `PlanObjective::Misses` | Page-only miss objective; decoded-row caching is off |
| `PlanObjective::Time` | Minimize modeled time under declared costs |
| `rows = true` | Requires a cost model; row selection takes effect under `Time` with nonzero frame capacity |
| `RowPolicy::Incremental` | Online/offline reuse-interval decisions |
| `RowPolicy::Global` | Whole-sequence row reference; active row selection requires a stored slice and full-sequence horizon |
| `block_pages` | Optional merged admission size; requires costs |
| `cache_bytes` | Planned frame budget, not total RSS |

Under `Misses`, setting `rows = true` does not enable row caching. Without a cost
model, setting rows or block merging is an error. Enabling row candidates does
not guarantee row admission: they compete with pages for the same frame budget.

Rows retain stored dtype and standard ids. Their size includes decompressed
payload and page rounding; a row can occupy noncontiguous frames. Row hits still
gather, map and process the row. Process outputs are not cached.

## Plan and Residency

`Plan` contains immutable residency and frame assignments plus compatibility
identities. `Residency` describes an `Object::Page` or `Object::Row`, its admission
and last use, and assigned frame extent. `frames_of` resolves its frame list.

| Plan query | Meaning |
|---|---|
| `len / schemas / slot_cells` | Access count / schema count / positions per slot |
| `reuse_gap / distance` | Reservation gap in slots / accesses |
| `cache_bytes / cache_frames` | Declared byte budget / usable whole-page frames |
| `cost / summary` | Optional model and compiled summary |
| `residencies / frames_of` | Immutable planned contents |
| `write / open` | Binary file persistence |
| `to_bytes / from_bytes` | Owned byte representation |

`CPKPLAN1` plans bind to inputs, mappings, schemas, process identities, slot size
and exact access contents. `prepare` and `replay` check compatibility against the
actual loader and sequence. `open` and `from_bytes` read the plan representation;
successful deserialization alone does not establish graph compatibility.
A serialized plan is not a portable promise of identical throughput on different
hardware. Keep process identities synchronized with callback/configuration changes.

## CostModel and resource accounting

| Component | Describes |
|---|---|
| Capacity | Physical worker cores, device IOPS/bandwidth, memory bandwidth, overlap share |
| `InputCost` | Per-record, per-element and cold-chunk decode work |
| `RowCost` | Per-payload-byte gather and row-admission work |
| `OutputCost` | Mapping, filling, processing and partial-row costs per schema |
| `ProcessCost` | Process-size bands ending at declared maximum elements |
| `convert` | Per-element cost per loader `Conversion` class |
| `Loads` | Requests, device bytes, core-seconds and memory traffic |
| `Prices` | Marginal values used to compare reuse candidates |

Operation costs use core-seconds; capacity converts them into modeled wall time.
Output-cost order must match schema order. `seconds`, `service` and `prices`
operate on a load vector; their outputs are model results, not measurements.

For loads `R` requests, `B` device bytes, `C` core-seconds and `M` memory bytes:

```text
device = max(R / iops, B / bandwidth)
workers = C / cores
seconds = max(max(device, workers) + serial * min(device, workers),
              M / memory_bandwidth)
```

`service` returns the four independent service times in request, device-byte,
core and memory order. `prices` assigns marginal seconds per load unit near the
supplied workload; `Prices::value(loads)` values a change at those prices.

Compilation validates positive finite capacities, finite nonnegative operation
costs, `serial` in `[0, 1]`, a positive `bucket`, and exactly one output entry per
schema. Nonempty process-cost curves have increasing inclusive `max_elements`
bounds ending at `u32::MAX`. Loading a standalone cost-model byte representation
checks its encoding; graph-specific validation happens at compilation/replay.

`CostModel::defaults` provides hardware-tier defaults; `opt::defaults::model`
and `Tier` select those defaults explicitly. Calibration records are managed by
the application. Binary `to_bytes/from_bytes` is available without Serde; optional
`serde` adds structured serialization derives.

### Read summaries without mixing evidence

| Result | Meaning |
| --- | --- |
| `PlanSummary` counters | Serial action counts, including requests, reads, decode work and hits |
| `peak_bytes` | Maximum bytes retained in serial access order |
| `concurrent_peak_bytes` | Peak with each residency reserved from `start − distance` |
| `PlanSummary::seconds()` | Sum of modeled time over four sequence quarters; zero without costs |
| `OnlineStats::modeled_seconds` | Modeled work of the published accesses; zero without costs |
| `OnlineStats::compile_seconds / waiting_seconds` | Measured compilation wall time excluding / measuring wait for plan-ring space |

Use [statistics and profiling](profiling.md) for measured run counters and timing.
Repricing a saved plan does not recalibrate it or change its decisions.

## Serial page reference

`PageTrace::from_source` extracts page uses for a stored sequence;
`from_pages` constructs a synthetic trace. `page_uses` and `distinct_pages`
describe it; `optimum(capacity_pages)` returns a page-only `CacheSummary`.

This optimal reference has serial page semantics and bypass. It does not include
decoded rows, concurrency reservation gaps, sharding, or finite lookahead.
Do not describe every production page-and-row plan as globally optimal.

## Complete interface index

<!--@include: ./_catalog/opt.md-->

## Default-model API

<!--@include: ./_catalog/opt-defaults.md-->
