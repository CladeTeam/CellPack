# Planning API

Use this reference for mode selection, option units and plan identity. The
[planning guide](/python/docs/planning.md) compiles, saves and executes a small
plan while checking its decoded result.

```python
from pycellpack.loader.opt import PlanOptions, Plan, PageTrace, CostModel
```

<DocChoices title="Select a planning interface" :items="[
  { label: 'Loader.start', when: 'Deliver batches under a selected mode', href: '#execution-and-compilation-modes', detail: 'None disables planning; Plan executes fixed decisions; options compile online.' },
  { label: 'Loader.compile / compile_stats', when: 'Inspect compilation without loading values', href: '#execution-and-compilation-modes', detail: 'Retain a Plan or collect online compiler observations.' },
  { label: 'Plan / PageTrace', when: 'Inspect or compare cache decisions', href: '#plan-identity-and-serialization', detail: 'Use bound-plan replay for actions, and the serial page optimum for a narrower reference.' },
]" />

## Execution and compilation modes

| Call | Planning behavior | Returns |
| --- | --- | --- |
| `start(accesses, plan=None)` | No planned cache | Run |
| `start(accesses, plan=plan)` | Execute a compatible fixed Plan | Run |
| `start(accesses, plan=options)` | Compile online ahead of executing workers | Run |
| `start(accesses, plan=byte_budget)` | Resolve default options/costs and compile online | Run |
| `compile(accesses, options, online=False)` | Whole-sequence compilation | Plan |
| `compile(accesses, options, online=True)` | Finite-horizon compilation into a retained plan | Plan with OnlineStats |
| `compile_stats(accesses, options)` | Online compilation without output delivery or retained decisions | OnlineStats |
| `replay(plan, accesses, cost=None)` | Verify binding and simulate serial actions | PlanSummary |

`compile` accepts PlanOptions or a byte budget and materializes computed Accesses
into a stored sequence. `compile_stats` and online `start` can generate intervals
without retaining all cell IDs. Access counts are position-first, schema-second
and must be divisible by the loader's schema count. See
[access sequences](access-functions.md) for composition and deterministic callbacks.

`plan_options(cache_bytes, cost="auto")` returns frozen PlanOptions with the
loader's window as `gap`. Automatic cost resolution consults saved records and
defaults; it does not calibrate hardware. Pass a CostModel explicitly or
`cost=None` for equal-valued page hits. Constructing PlanOptions directly does
not resolve costs automatically. The loader fills unset gap/thread values in a
copy, leaving your object unchanged.

## PlanOptions parameters

| Field | Default | Contract |
| --- | --- | --- |
| `cache_bytes` | Required | Nonnegative frame budget in bytes; rounded down to whole source pages; frame count must fit uint32 |
| `gap` | None | Reuse reservation in slots; resolves to loader window; positive, with its access distance below `2**32 - 1` |
| `threads` | None | Positive compiler thread count; automatic offline/online policies described below |
| `objective` | `"time"` | Minimize modeled time or `"misses"` for page misses |
| `cost` | None | CostModel for this loader; required when rows or block candidates are requested |
| `rows` | False | Consider native-precision decoded rows; effective only with `"time"` and at least one frame |
| `block_pages` | None | Optional candidate read width in 2–128 pages; requires costs |
| `row_policy` | `"incremental"` | Incremental decisions, or `"global"` as a complete-stored-sequence reference |
| `horizon` | None | Configured online lookahead in slots, at least gap; offline compilation uses the whole sequence |
| `chunk` | None | Positive advance in accesses; native default 16384 |
| `cpus` | `()` | Compiler CPU affinity; empty inherits caller affinity |

Automatic `threads` is one per usable CPU offline, or a quarter of executor
workers online, bounded to 1–16 and the usable CPU count. An explicit positive
value bypasses that automatic cap. `cpus` influences the usable count and pins
compiler threads; it does not configure executor workers.

`rows=True` with `objective="misses"` is disabled by the engine, rather than
selecting a mixed row/page miss objective. The requested row mode still requires
a cost model during validation. A zero-byte plan still compiles and uses a
PlanRing; choose `plan=None` to disable planning. Global row selection requires
effective row caching over a fully available stored sequence and full horizon;
use the incremental policy for finite-horizon or generated streams.

### Slots versus accesses

One slot represents `S × slot_rows` accesses, where `S` is the number of loader
schemas. The gap reserves frames before their admission; loader `window`
controls execution lead and may not exceed that gap. Neither setting is the
output ring's `slots` capacity.

Configured `horizon` is converted to accesses. With no explicit horizon, online
lookahead starts at twice the source cell count per schema. The engine raises
it to at least `distance + chunk`, then caps it at sequence length (at least one
for an empty sequence). Consequently OnlineStats reports the **effective**
horizon, which can differ from the configured request.

| Reported field | Unit |
| --- | --- |
| `Plan.reuse_gap` | Slots |
| `Plan.distance`, `OnlineStats.distance` | Accesses |
| `OnlineStats.horizon`, `OnlineStats.chunk` | Accesses |
| `Plan.slot_rows` | Positions per slot |
| `len(plan)` | Accesses |

## Plan identity and serialization

A Plan binds graph/source identity, column mappings, schema/process identities,
schema order, `slot_rows`, and the exact access sequence. Reordered accesses or a
different schema require recompilation. Ring slot count and consumer world size
are not plan fingerprints; execution must still satisfy the loader's layout,
window and ring contracts. Changing source files in place is outside the immutable
input contract.

`Plan.write(path)` and `Plan.open(path)` use CPKPLAN1 files. `to_bytes` and
`from_bytes` use the same format; a Plan also pickles through those bytes. The
file stores fingerprints, residency decisions and the optional cost model.
Opening parses the file, while `start` or `replay` checks compatibility with an
actual loader and sequence.

For a freshly compiled Plan, `summary` contains serial replay counts. A restored
Plan's summary is initially empty and is populated when a loader first verifies
it through `start` or `replay`; inspect it after that step. `online_stats` belongs
to a `compile(..., online=True)` result and is not retained by the plan file or
pickle. Keep those observations separately if you need them.

`residencies()` returns arrays in admission order: `kind` (0 page, 1 row), `key`
(global page ID or cell ID), `start`, `end`, `offsets` and `frames`. Residency
`i` uses `frames[offsets[i]:offsets[i + 1]]`; start/end address accesses, not
positions or batches. `cache_bytes` is the original budget and `cache_frames`
is its whole-page capacity, not measured resident memory.

`replay(..., cost=other_model)` prices the same actions with another compatible
model without changing the Plan or its stored summary. PlanSummary seconds are
modeled; OnlineStats separates modeled seconds from compiler/wait wall seconds.
Use executed [RunStats](statistics.md) for run observations.

## Costs and reference comparisons

CostModel prices request capacity, byte bandwidth, worker physical cores,
cached-row memory traffic and operation costs in core-seconds. Rates must be
positive and finite, costs finite and nonnegative, `serial` in `[0, 1]`, and
`bucket >= 1`. Output records must match the loader's outputs. Compilation and
replay check model compatibility; constructing a dataclass alone is not that
validation.

`to_dict` / `from_dict` use the versioned native JSON field names; `save` writes
JSON by sibling-file replacement, and `load` reads it. Required resource rates
remain required; omitted optional fields use current defaults. Neither operation
measures hardware. For resolution, record provenance and calibration, see
[statistics and cost records](statistics.md).

`PageTrace.from_pages(pages, universe=None)` normalizes a one-dimensional
nonnegative uint32-range page sequence; the default universe is maximum ID + 1,
or zero for an empty sequence. `from_source(source, accesses)` expands the pages
needed by those global cells. `optimum(capacity_pages)` returns CacheSummary for
serial Belady replacement. Its capacity counts pages, and its model excludes
reuse-gap reservation, execution concurrency, decoded rows and management costs.
It is not an optimal mixed page/row plan.

## Validation and declarations

Invalid Python enum/count declarations raise ValueError; native checks also
validate the resolved gap, horizon, frame count, CPU indices, cost model and
plan binding. I/O errors opening or writing a plan raise OSError. Keep outputs
and records from failed runs out of downstream results; lifecycle recovery is
covered by [rings and runs](rings.md).

::: details Complete declarations and members

<!--@include: ../../_catalog/python/planning-members.md-->

:::
