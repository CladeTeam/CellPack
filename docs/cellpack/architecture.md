# Architecture

Read this page after [loading and ownership](loading.md) if you need to connect
public objects to their execution path. For a first working program, start with
the [quickstart](quickstart.md). The architecture explains the current layering;
the [module references](api.md#choose-a-module) specify the callable contracts.

## Public layers

| Module | Responsibility |
|---|---|
| [`store`](reference-store.md) | Matrix streams, prepared selections, borrowed and allocating reads |
| [`scans`](reference-scans.md) | Chunk/record encoding, decoding, mapping, packing, and table construction |
| [`format`](reference-format.md) | Byte layouts and metadata views |
| [`loader`](reference-loader.md) | Sources, schemas, accesses, workers, rings, and run lifecycle |
| [`opt`](reference-opt.md) | Offline/online cache planning, model costs, residency files, serial references |
| [`prof`](reference-prof.md) | Timing sites, masks, and reports |
| [`runtime`](reference-runtime.md) | Process-wide kernel binding and CPU topology queries |

The core crate is independent of Python. Language bindings adapt these interfaces
and own their language-specific arrays and lifetimes.

## Input preparation

A data source reads chunk metadata and builds decode tables. Loader construction
composes source, coded, standard, and output column mappings, checks conversion
routes, and associates the immutable graph with execution settings.

`prepare` checks file identity, sequence shape, ring compatibility, and plan
binding, then opens a new run generation. Inputs and outputs are bound before
worker execution begins.

## Record path

<DocDiagram name="record-path" />

A task owns an entire position. When schemas at that position reference the same
cell, input work can be shared. Results still go to their separate output regions.
The output ring publishes a slot only after all its required schema outputs are
complete.

## Frame grid and generations

A run reserves a grid of page-sized frames. Planned frames hold cached pages or
decoded input rows; worker-private frames hold misses. A generation word marks
each cached frame's current content. Admission reads or row construction publish
that generation after the content is ready. Dependent accesses wait for it.

A plan reserves frames ahead of admission by the reuse gap. The execution window
cannot exceed that gap. This constrains overwrites without requiring runtime
eviction decisions or a cache-directory lookup.

Decoded rows can span noncontiguous frames. Current row hits gather them into
scratch before mapping and processing. The process ABI receives CSR or dense
rows, not a direct borrowed paged-row representation.

## Compiler and plan ring

Offline compilation builds residency intervals for a stored sequence. A fixed-plan
producer expands them into per-access entries. Online compilation discovers
reuse, selects rows and pages, freezes frame assignments, and publishes entries
ahead of workers through a bounded PlanRing.

Entries describe misses, cached frames to wait for or admit, extra block reads,
and decoded rows to build or use. `Planning::None` has no PlanRing.

The compiler advances in chunks and uses a lookahead horizon. Published entries
may be split into smaller ranges; whole-chunk atomic publication is not a public
guarantee. Row selection is incremental by default, with an offline global
reference. See [planning](planning.md) for the actual option units and constraints.

## I/O and worker specialization

Each worker uses io_uring or blocking preadv. io_uring queues and harvests
requests; files are registered when supported. Failure to create a ring is an
error, so restricted environments must select `Pread` explicitly.

Workers are specialized by profiling and cache mode. Kernel pointers are bound
during context construction rather than selected per element. Per-record codec
components operate on caller buffers; some loader task and I/O containers can
still grow on demand. The complete runtime is not promised to be allocation-free.

## Completion and failure

Output slots carry run generations and ownership state. A lease delays reuse
until all its readers finish. Cancellation wakes blocked parties; worker errors
cancel the run and workers are joined before the producer returns.
A new successful run requires a released previous run and compatible unleased
slots. An aborted ring must be replaced.

Use counters, replay, and actual decoded values to inspect these behaviors.
Historical benchmark results and design goals are not current guarantees for a
different workload.
