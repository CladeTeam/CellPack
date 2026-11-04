# Cache planning and diagnostics

Validate values and output order without a cache first. A plan then decides which
compressed pages and decoded input rows remain in memory between accesses. It
does not cache mapped outputs or choose your sampling order.

<DocChoices title="Choose when to plan" :items="[
  { label: 'plan=None', when: 'Establish correctness or expect little reuse', detail: 'Use private read buffers and the output ring without a planned cache.' },
  { label: 'Compile a Plan', when: 'Retain and reuse one exact sequence', href: '#compile-save-and-run-a-fixed-plan', detail: 'Inspect or save residency decisions, then execute compatible runs.' },
  { label: 'Start with PlanOptions', when: 'Compile ahead of an executing run', href: '#compile-during-execution', detail: 'Use a finite lookahead and a bounded PlanRing, including for computed accesses.' },
]" />

<DocDiagram name="python-planning" />

## Compile, save, and run a fixed plan

The following blocks continue with the single-output `loading` and `source`
created in [native loading](native-loader.md#register-sources-and-outputs). Run
them in the same writable directory. They require NumPy and the core package;
they create `training.plan`.

```python
import numpy as np
from pycellpack.loader.opt import Plan

cells = np.array([5, 1, 5, 0], dtype=np.uint32)
options = loading.plan_options(8 << 20, cost=None)
plan = loading.compile(cells, options)
plan.write("training.plan")
restored = Plan.open("training.plan")
summary = loading.replay(restored, cells)

parts = []
with loading.start(cells, plan=restored, slots=2) as run:
    for batch in run:
        parts.append(batch["pair"].copy())
    stats = run.join()
expected = np.array([[23, 20], [7, 4], [23, 20], [3, 0]], np.float64)
np.testing.assert_array_equal(np.concatenate(parts), expected)
assert summary.accesses == stats.accesses == len(cells)
assert restored.distance == restored.reuse_gap * restored.slot_rows * restored.schemas
print("fixed plan: 4 accesses; requested values and order verified")
```

The two batches contain two positions each. Repeated cell 5 appears twice, and
the output columns remain `[3, 0]`. A cache hit changes input work, not the
delivered values or order. Copying retained arrays releases their native slots
for reuse.

`cost=None` makes this a page-only example without resolving cost records. A
small fixture establishes correctness; its hit count is not a performance
claim. `replay` checks and counts serial plan actions without delivering output.
The subsequent run checks the actual decoded result.

A fixed plan binds the source graph, schema/process identities, positions per
slot, and exact access sequence. Recompile after changing those inputs. Output
ring capacity `slots` is a separate buffering choice, subject to the loader's
layout and execution contracts.

## Compile during execution

Pass options to `start` to compile ahead of workers instead of executing a
retained Plan. This continues with the same `cells`, `options` and `expected`:

```python
parts = []
with loading.start(cells, plan=options, slots=2) as run:
    for batch in run:
        parts.append(batch["pair"].copy())
    online_stats = run.join()
np.testing.assert_array_equal(np.concatenate(parts), expected)
assert online_stats.online is not None
assert online_stats.accesses == len(cells)
print("online plan: the same 4 accesses verified")
```

Passing an integer byte budget also selects online compilation, with costs
resolved automatically. A zero-byte plan still runs planning machinery; use
`plan=None` to turn planned caching off.

These three operations answer different questions:

| Operation | What it does | Result |
| --- | --- | --- |
| `loading.start(cells, plan=options)` | Compile ahead of workers and deliver batches | Run, then RunStats |
| `loading.compile(cells, options, online=True)` | Simulate finite-horizon compilation and retain its decisions | Plan with `online_stats` |
| `loading.compile_stats(cells, options)` | Compile for observations without delivering data or retaining a plan | OnlineStats |

`compile` materializes computed access sequences. For a long native generator,
use online `start` to retain interval-based generation. A finite-horizon compile
is useful for inspecting decisions; it does not execute a loader run.

## Keep units and budgets separate

| Setting | Unit | What it controls |
| --- | --- | --- |
| `cache_bytes` | Bytes | Planned page and native-precision row frames |
| `slot_rows` | Positions | Logical output rows per slot |
| `slots` | Slots | Output ring capacity |
| `gap`, configured `horizon`, loader `window` | Slots | Reuse reservation, lookahead and execution lead |
| `chunk` | Accesses | Compiler advance and publication granularity |
| `threads`, `workers` | Threads | Compiler and executor parallelism, respectively |

With `S` schemas and `R` positions per slot, one slot represents `S × R`
accesses. A gap of 4 slots with 2 schemas and 256 positions per slot is 2048
accesses. `Plan.distance`, `OnlineStats.distance` and `OnlineStats.horizon`
report **accesses**, while `Plan.reuse_gap` reports slots.

The configured online horizon is a request: the compiler raises it to cover at
least the reuse distance plus one compilation chunk, then limits it to the
sequence length. A short sequence can therefore report fewer accesses than the
requested horizon. The execution `window` must not exceed the plan's gap.

<DocDiagram name="memory-budgets" />

Use `loading.memory_estimate(plan=options, slots=2)` for the actual mode. Known
reservations include cache frames, private buffers, worker scratch, PlanRing,
metadata and the output ring. Compiler state, allocator overhead and OS page
cache remain additional. This estimate is neither measured RSS nor a hard
process memory limit. Cache budget and output prefetch capacity are independent.

## Add rows and measured costs deliberately

`loading.plan_options(cache_bytes)` defaults to `cost="auto"`: it resolves
compatible saved records and fills missing parts with defaults. This does not
run calibration. A directly constructed `PlanOptions(cache_bytes=...)` instead
has `cost=None` until you supply one.

Page hits avoid reads but still need decoding. Row hits reuse decoded input in
its stored dtype; mapping and process functions still run. Consider row caching
when repeated decoding justifies the larger frame footprint. `rows=True` and
`block_pages` require costs even when another option would make them ineffective.
Rows take effect with the `"time"` objective and a nonempty cache; `"misses"`
disables row caching. Block candidates trade fewer requests against more retained
frames; they do not force every read to a fixed block size.

Use the incremental row policy for streaming compilation. The global policy is
an offline reference over a complete stored sequence. Exact defaults, option
ranges and plan binding rules are in the [planning API](/python/api/planning.md).

## Interpret observations

Use final RunStats after draining output. `summary` and `replay` provide modeled
serial work, whereas RunStats records the executed run. Modeled seconds are not
measured throughput. Timing sites, cost calibration and reservation fields are
explained in [statistics and cost records](/python/api/statistics.md).

`PageTrace.optimum` supplies a serial Belady page-cache reference. It excludes
decoded-row choices, the execution window, reuse-gap reservation and management
costs, so it is not an optimal mixed page/row loading plan.
