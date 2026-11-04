# Cache planning

A cache plan decides which compressed pages and decoded input rows stay in
memory between accesses. It assigns their frames before workers use them; workers
then read, decode and wait on those assignments. Start after checking the values
and ordering of an [uncached loading run](loading.md).

## Choose a planning mode

<DocChoices title="Choose when to compile" :items="[
  { label: 'Planning::None', when: 'Establish correctness or expect little reuse', detail: 'Use worker-private read buffers and the output ring without a planned cache.' },
  { label: 'Planning::Fixed', when: 'Keep an exact access sequence in memory', detail: 'Compile once, inspect or save the Plan, then execute compatible runs.', href: '#compile-save-replay-and-run' },
  { label: 'loader.online(options)', when: 'Generate a long or computed sequence', detail: 'Compile ahead of workers with bounded lookahead; consume the run as usual.', href: '#set-the-lookahead-and-execution-window' },
]" />

Fixed and online runs use the same worker and output-ring contracts. A zero-byte
plan still performs planning and uses a PlanRing; choose `Planning::None` to
disable that work. The cache budget excludes worker buffers, output and compiler
state; see [memory reservations](performance.md#memory-reservations).

## Compile, save, replay, and run

The example creates a four-row fixture, requests those rows repeatedly, and
executes four variants. Its checks establish these results:

| Stage | What it checks |
| --- | --- |
| Uncached run | Every delivered row equals the requested input row |
| Fixed page plan | Delivered values match; page hits, read bytes and requests equal serial replay |
| Decoded-row candidates | Delivered values match with row caching enabled |
| Online run | Delivered values match and every access completes |

::: details Complete runnable example

<<< @/examples/planning.rs

:::

In the application from [installation](installation.md), save
[planning.rs](/downloads/rust/planning.rs) as `src/main.rs` and
[common.rs](/downloads/rust/common.rs) as `src/common.rs`, then run:

```sh
cargo run -- output/planning
```

The final line starts with `page reads:` and reports the observed counters and
known reservations. The tiny fixture and its counters are functional checks,
not performance measurements; no particular hit count is needed for correctness.

The decoded-row variant uses the loader's default cost model for demonstration.
It enables row candidates rather than forcing their admission; use measured
costs to decide which candidates are worthwhile on your machine.

For your own loader, the core fixed-plan sequence is:

```rust
let options = loader.plan_options(cache_bytes);
let plan = loader.compile(&accesses, &options)?;
let expected = loader.replay(&plan, &accesses)?;
plan.write("cache.plan")?;
// Pass Planning::Fixed(Arc::new(plan)) to loader.prepare, then consume the run.
```

`accesses` contains global cell IDs in position-first, schema-second order.
Compilation and replay inspect source descriptors and plan actions; they do not
deliver batches. Execute the plan to check actual decoded values and I/O behavior.

## Choose what to retain

<DocDiagram name="cache-paths" />

A page hit avoids storage reads but still requires decoding. A row hit gathers a
decoded input row from cached frames; mapping and processing still run. Rows keep
their stored dtype and can be shared by schemas. Mapped output and process results
are not cached.

Begin with pages. Enable `rows` when repeated decoding is expensive enough to
repay decoded-row admission and memory traffic. Rows occupy page-rounded frames
and can be much wider than compressed records. Current row hits gather
noncontiguous frames into worker scratch.

`block_pages` considers reading adjacent pages together to save requests, at the
cost of retaining frames longer. It does not force every read to that size.
Use measured costs for these decisions rather than treating more hits as the
sole objective.

## Set the lookahead and execution window

Start with `Loader::plan_options(cache_bytes)`: it derives a reuse gap from the
loader and installs the default NVMe cost model in schema order. Replace those
costs with measurements for your target machine when evaluating performance.

| Field | Unit or meaning |
|---|---|
| `cache_bytes` | Cache-frame byte budget, rounded down to whole pages |
| `gap` | Reuse reservation distance in output slots |
| `threads`, `cpus` | Compiler parallelism and optional CPU affinity, separate from worker settings |
| `objective` | Page `Misses` or modeled `Time` |
| `cost` | Hardware and operation costs; optional for page-only planning |
| `rows` | Consider decoded rows; requires costs and takes effect with `Time` and at least one frame |
| `row_policy` | `Incremental` for ordinary runs; `Global` for a full-sequence stored reference |
| `block_pages` | None, or a merged admission limit of 2–128 pages; requires costs |
| `horizon` | Online lookahead in output slots; an explicit value must be at least `gap` |
| `chunk` | Compiler advance in accesses; default `DEFAULT_CHUNK` (16384) |

One slot spans `slot_cells × schemas` accesses. For example, with 256 positions
per slot and two schemas, `gap = 4` reserves 2048 accesses. A `chunk = 16384`
advances 32 such slots. `Plan::reuse_gap()` reports slots, whereas
`Plan::distance()` and `OnlineStats::{distance, horizon, chunk}` report accesses.

`LoaderConfig::window` is also in slots. With a plan it defaults to the plan's
gap and cannot exceed it. Reservation distance prevents frame reuse while an
earlier in-flight access still reads that frame.

Offline compilation sees the stored sequence. Online compilation limits reuse
search to its effective horizon. If unset, the starting lookahead is twice the
source's cell count times the schema count, in accesses. The compiler raises it
to at least `distance + chunk`, then clips it to the sequence length; inspect
`OnlineStats.horizon` when the effective value matters. Longer horizons retain
more compiler state.
Decisions can depend on compiler thread count through sharding, so keep it fixed
for comparisons.

## Plan identity

A fixed plan binds to input identity, graph and schemas, process identities, slot
size, and exact access contents. `prepare` verifies this binding and frame reuse.
`Plan::write` and `Plan::open` use the binary `CPKPLAN1` format.
Recompile after changing those semantics or replacing inputs.

A custom process schema's `identity` versions its callbacks and captured settings.
Callback addresses do not describe semantics.

Opening a plan reads its serialized form. Compatibility with the actual loader
and exact sequence is established when replaying or preparing it, not merely by
successfully calling `Plan::open`.

## Replay and optimal references

| Question | Use | Interpret the result |
| --- | --- | --- |
| Which work does this fixed plan imply? | `replay` | Counts actions without loading record payloads |
| How do other machine costs price those actions? | `replay_with_cost` | Changes prices, not cache decisions; recompile to choose new decisions |
| What does finite-horizon compilation retain? | `compile_online` | Returns a collected Plan and OnlineStats; does not start a run |
| Can I inspect a computed sequence's planning? | `plan_online` | Returns compiler statistics without delivering batches |

`PlanSummary::seconds()` and `OnlineStats::modeled_seconds` are model predictions.
`OnlineStats::compile_seconds` measures compilation wall time, excluding its wait
for PlanRing space; `waiting_seconds` reports that wait separately. Measure a
real run's elapsed time and declared workload to assess throughput.

`PageTrace::optimum` is the serial page-only optimal reference with bypass.
It does not model decoded rows, concurrent frame reservations, sharding, or finite
lookahead. Production page-and-row planning uses constrained, cost-guided
decisions; it is not a globally optimal end-to-end scheduler.

API: [PlanOptions](/generated-opt-struct-PlanOptions),
[Plan](/generated-opt-struct-Plan),
[PlanSummary](/generated-opt-struct-PlanSummary),
[PageTrace](/generated-opt-struct-PageTrace).
