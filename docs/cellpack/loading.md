<script setup>
import RustLoaderLifecycle from './.vitepress/theme/components/RustLoaderLifecycle.vue'
</script>

# Loading and output ownership

A `Loader` binds a data source, output schemas, and execution settings. Each
`prepare` binds an access sequence, output ring, and planning mode to one run.
Workers may finish out of order; outputs occupy their requested positions.

<DocDiagram name="rust-loading" />

## Choose a consumption model

<DocChoices title="Who drives the consumer?" :items="[
  { label: 'Run::for_each_batch', when: 'Process batches locally in order', href: '#a-complete-multi-output-run', detail: 'Starts a scoped producer and releases each lease after your callback. Begin here.' },
  { label: 'Run::run + Run::wait', when: 'Integrate your own threads or asynchronous readers', href: '#separate-producer-and-consumers', detail: 'Run production concurrently with every rank consumer and retain owned leases until readers finish.' },
]" />

Before delivery, register immutable files with [source-to-standard maps](data-sources.md)
and choose [an access sequence](accesses.md). Loader construction does not choose
sample order or create a training loop.

## Bind one run

| Step | What it fixes | What remains yours |
|---|---|---|
| `Loader::new(source, schemas, config)` | Output order, mapping, conversion and worker settings | Source files remain immutable |
| `loader.ring_layout(slots, world)` | Layout for these schemas and `slot_cells` | Choose buffering and consumer count |
| `Ring::allocate(layout)` | Backing storage and slot control | Keep every reader within its lease |
| `loader.prepare(&accesses, ring, planning)` | Exact sequence, ring and run generation | Start production and consumption |
| `run.for_each_batch(consume)` | Concurrent producer and ordered local consumption | Read or copy each batch in `consume` |

**`prepare` does not start workers.** Use one execution path per prepared run.
For the first correctness run, use `Planning::None`, buffered reads and explicit
`IoBackend::Pread`; add [planned reuse](planning.md) after checking values and order.

## A complete multi-output run

This example emits a three-column `f32` panel and a two-column `f64` panel for each
position. The first panel includes a missing column marked with NaN. Repeated
cells and a partial final batch are checked against the source matrix.
Selected I/O, CPU, and wait timing sites are enabled through `SiteMask`.
Five requested positions become ten accesses and three batches of 2, 2 and 1
rows. Each batch has both output regions; schema count does not multiply its
`rows()`.

::: details Complete program: source, schemas, accesses, consumption and checks
<<< @/examples/loading.rs
:::

In the application from [installation](installation.md), save
[loading.rs](/downloads/rust/loading.rs) as `src/main.rs` and
[common.rs](/downloads/rust/common.rs) as `src/common.rs`, then run:

```sh
cargo run -- output/loading
```

The example uses buffered `Pread` reads, two workers, and tiny slots. These settings
make the example easy to run; they are not throughput recommendations.

Expected output:

```text
loaded 5 positions into two schemas, including a partial last batch
```

## Output schemas

`OutputSchema::dense(id, out_ids, dtype)` produces `out_ids.len()` dense values
per cell. Schema ids must be distinct within a loader; at most `MAX_SCHEMAS` (64)
schemas are supported. Labels set by `named` are for reports, outside plan identity.
The schema's position in the supplied vector selects `batch.region(s)` and
`ProcessContext.schema`; its application `id` is a separate identity.

The schema dtype controls conversion before delivery. `loader::allowed` describes
accepted pairs; construction rejects unsupported conversions. Missing columns use
the default, while present zeros remain zero. See [column mapping](data-sources.md).
`OutputSchema::process` reserves a custom fixed-size record for a
[native callback](process-functions.md).

## Layout and leases

`loader.ring_layout(slots, world)` derives a `RingLayout` with the correct
`slot_cells` and per-schema `cell_bytes`. Allocate it with `Ring::allocate`.
Each slot has one compact region per schema, with 64-byte-aligned region starts
and a 4096-byte-aligned slot stride.

<RustLoaderLifecycle />

| Batch method | Meaning |
|---|---|
| `index()` | Batch index in this run |
| `generation()` | Run generation to which this lease belongs |
| `first_position()` | First logical position in the batch |
| `rows()` | Actual row count, including a possibly short tail |
| `region(s)` | `rows × cell_bytes(s)` valid bytes for schema `s` |
| `data()` | Full slot data area, including padding and unused tail space |

Use `region(s)` and `rows()` to access meaningful output. The examples use
`from_ne_bytes` rather than assuming arbitrary bytes form an aligned typed slice.
Output values use native representation.

`Run::for_each_batch` starts a scoped producer thread and consumes batches in
order. Its callback borrows a `BatchLease`; the lease is released after the callback
returns. References obtained from it must not outlive the lease.

For a custom asynchronous consumer, retain an owned `BatchLease` until every
reader, including a device copy, finishes reading its bytes. Holding every slot
prevents further production. Copy into independent storage for long-lived output.

## Separate producer and consumers

`Run::run()` executes the producer on the calling thread. Run it concurrently
with consumers calling `Run::wait(rank, batch)`. Running the producer to completion
before consuming a sequence longer than the ring capacity can block indefinitely.

Batch `b` uses slot `b % slots` and belongs to rank `b % world`. `world` assigns
batch ownership; it does not create a distributed runtime or processes.
Each assigned rank consumes its batches in increasing order, releasing them to
allow reuse. It is not a broadcast: two independent readers must not acquire the
same batch. The convenience `for_each_batch` interface consumes all ranks locally.

`Ring::acquire_timeout` returns `Acquisition::{Batch, Pending, Done}` for bounded
waiting. For waits tied to a prepared run, `Run::wait` validates rank, batch and
generation. On error, propagate the original failure and stop that run.

`Run::cancel` aborts its ring and wakes participants. Cancellation is not a pause:
this ring cannot be reused. A consumer callback returning an error also cancels
the run; `for_each_batch` waits for the producer before returning the failure.

## Reuse and external memory

One loader serves one run at a time. Drop the completed `Run` before preparing
another. A ring can serve successive runs after all prior slots are consumed and
all leases are released. An aborted ring cannot start another run; allocate a
new one for recovery. Dropping a prepared run that was never started also cancels
it and aborts its ring. Completing the producer alone does not release a retained
run handle or consumer lease.

External buffers use unsafe `Ring::init`, `attach`, `init_regions`,
`attach_regions`, or `with_buffer`. Follow their alignment, size, ownership,
and synchronization contracts. Backing storage must remain stable through every
lease. Shared control blocks use offsets and shared futexes; the application
supplies shared mappings and manages participant lifetimes.

API: [Loader](/generated-loader-struct-Loader),
[Run](/generated-loader-struct-Run),
[RingLayout](/generated-loader-struct-RingLayout),
[BatchLease](/generated-loader-struct-BatchLease).
