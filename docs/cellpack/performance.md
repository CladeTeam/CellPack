# Memory and performance

Tune a representative workload only after checking decoded values, output order,
and ownership. Cache hit counts, modeled time, and small examples describe different
parts of a run; none alone establishes end-to-end training throughput.

## Memory reservations

`Loader::memory_estimate(cache_bytes, slots, world)` estimates a default
page-only configuration; zero bytes selects no planning. Use
`memory_estimate_for(&planning, slots, world)` for an actual fixed or online plan,
including row caching and zero-budget plans.

<DocDiagram name="memory-budgets" />

| `MemoryEstimate` field | Reservation |
|---|---|
| `cache_frame_bytes` | Planned page and row frames |
| `generation_bytes` | A generation word per cache frame |
| `private_frame_bytes` | Worker-private miss buffers |
| `worker_scratch_bytes` | Decode, conversion, mapping, and process scratch |
| `plan_ring_bytes` | Published plan entries and their variable-length data |
| `metadata_bytes` | Chunk metadata, decode tables, column and panel maps |
| `output_ring_bytes` | Output data and ring control |
| `known_reservation_bytes` | Sum of the known reservations |

These values are not measured RSS and are not a total process-memory limit.
Allocator overhead, compiler state, and the OS page cache are outside the estimate.
Reserved virtual memory need not all be resident.

Private buffers depend on worker count, in-flight depth, schema count, record-page
sizes, and a reserve for oversized records. Row-cache and plan-ring sizing must
also account for wide decoded rows, rather than only compressed page counts.
The planner's horizon retains additional state, which grows with the number of
accesses it covers.

Use `RingLayout::bytes()`, `control_bytes()`, and `data_bytes()` for actual aligned
ring sizes. A rough payload expression,
`slots × slot_cells × sum(cell_bytes)`, excludes padding and control metadata.

## Execution settings

| `LoaderConfig` field | Default | Tuning role |
|---|---|---|
| `workers` | 16 | Parallel reads, decoding, mapping, and callbacks |
| `backend` | `Uring` | Asynchronous io_uring or blocking `Pread` |
| `max_inflight_per_worker` | 64 | Outstanding positions per worker |
| `claim_batch` | 8 | Accesses per claim, rounded and bounded to positions and slots |
| `slot_cells` | 256 | Positions in one output slot |
| `window` | Derived | Maximum execution lead in slots |
| `profile` | `None` | Optional timing mask |
| `cpus` | Empty | Optional worker CPU affinity |

Start with uncached execution and a validated data source. Increase workers and
in-flight depth while observing CPU, I/O, memory reservations, and stalls.
Blocking `Pread` does not issue asynchronous reads merely because in-flight depth
is large. More ring slots help absorb consumer variation, but cannot remove a
persistent consumer bottleneck.

Add page caching next, then evaluate decoded rows with measured costs.
Keep input files, exact accesses, schema semantics, batch size, CPU placement,
and timing boundaries fixed when comparing modes.

## Runtime kernels

The runtime probes CPU support and binds a kernel set once per process.
On x86-64 it selects the best supported scalar, AVX2, or AVX-512 implementation.
Other architectures use scalar codec kernels; this does not imply that the entire
Linux loader supports every platform.

`CELLPACK_KERNELS=scalar|avx2|avx512` caps the selected ISA. A cap does not force
unsupported instructions. Set it before launching the process; the first runtime
or codec context fixes the binding. Invalid values fail environment-based
initialization. Changing the variable afterward cannot rebind existing contexts.

For example, set up the [quickstart](quickstart.md) with the crates.io dependency
and run that small correctness fixture with scalar kernels:

```sh
CELLPACK_KERNELS=scalar cargo run -- output/scalar
```

Compare caps in separate fresh processes while keeping input, workload and
measurement scope fixed. This fixture checks decoded values; it does not measure
representative throughput. Inspect `runtime::get()?.isa()` for the actual selected
ISA, rather than reporting the requested cap as the implementation used.

For explicit startup control, initialization errors, and physical-core queries,
see the [runtime API](reference-runtime.md). Kernel selection, physical-core
counting, worker count, and worker affinity are separate controls.

## Interpret I/O carefully

`IoStats` counts application requests and returned bytes. With buffered I/O,
those reads can be served by the OS page cache. Direct I/O can make device traffic
easier to interpret but still needs filesystem and storage verification.
Do not label application bytes as physical device bytes.

Report cache state, direct-I/O setting, backend, CPU placement, input shape and
dtype, access distribution, schemas, consumer work, and elapsed-time scope with
any throughput result. See [profiling](profiling.md) to find the limiting stage.
