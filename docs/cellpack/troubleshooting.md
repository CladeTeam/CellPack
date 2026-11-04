# Troubleshooting

## A run stops making progress

Check whether the consumer still holds a batch lease or has stopped consuming
an assigned rank. A full output ring blocks the producer. Finish readers and drop
leases, or copy outputs into independent storage.

Do not call the producer synchronously and wait for it to finish before starting
consumption when the sequence can fill the ring. Use `for_each_batch`, or run
producer and consumers concurrently. Inspect `wait.ring`, `wait.plan`,
`wait.window`, and `wait.gen` when profiling is enabled.

## io_uring initialization fails

Some kernels or container policies disable io_uring. Select
`LoaderConfig { backend: IoBackend::Pread, .. }` explicitly. Backend creation
errors are not an automatic request to change the backend.
Check [performance](performance.md) when comparing asynchronous and blocking reads.

## Direct I/O returns an error

Check filesystem support, stream offset alignment, and page size.
Use `DataSourceBuilder::direct_io(false)` to establish buffered correctness.
An embedded stream's logical page alignment alone does not establish the outer
file's physical offset alignment.

## Configuration or plan is rejected

| Symptom | Check |
|---|---|
| Source page-size mismatch | All streams use the builder's declared page size |
| Invalid column map | One entry per source column; distinct in-range targets or `UNMAPPED` |
| Invalid schema | Distinct schema ids and standard ids; compatible conversion and nonzero sizes |
| Access count mismatch | Length is divisible by the number of schemas |
| Cell outside the source | Global registration ranges and paired offsets |
| Ring mismatch | Derive layout with this loader's `ring_layout` |
| Plan mismatch | Inputs, exact access sequence, schema/process identities, and slot size |
| Window too large | Execution window does not exceed plan reuse gap |
| Invalid online options | Horizon is at least the gap; row and block policies have their required costs |

## `Busy` or an unusable ring

Only one run may hold a loader and one producer may hold a ring.
Drop the previous `Run`, consume ready output, and release every lease before
reuse. An aborted ring remains aborted and must be replaced.

After a run fails, report the first non-`Aborted` error when available:
cancellation can cause other participants to report `Aborted` as a consequence.
Treat that run as finished with failure and discard incomplete output.

## Unexpected zeros or defaults

A present matrix column with an omitted sparse value is zero. A column absent
from that matrix receives the schema default. Verify source-to-standard and
output-to-standard maps. A default does not replace present zeros.
See [data sources](data-sources.md).

## CSR output has unused capacity

`ReadSpec::nnz` is source-based capacity, which can exceed actual output nnz
after dropping columns. Use the count returned by `read_csr_into` and the final
row offsets. Bytes beyond actual nnz are not output data.

## Memory exceeds the cache budget

`cache_bytes` limits cache frames, not process RSS. Private read buffers, metadata,
scratch, output rings, compiler state, allocator overhead, and OS page cache
are additional. Use the matching planning mode with `memory_estimate_for` and
measure resident memory separately.

## Malformed data or a failed decode

Keep chunks, records, decode tables, and value widths matched. The codec trusts
inner records; release decoding is not a sanitizer for arbitrary untrusted bytes.
After any failing read or decode, discard potentially partial output.

## Documentation build fails

- Use the pinned documentation toolchain from [documentation development](contributing.md).
- Run `uv sync --locked --project docs/cellpack` for the isolated documentation environment.
- A rustdoc warning is an error; check the reported Rust source link.
- Install the frontend with `npm ci --prefix docs/cellpack`.
- Missing Markdown links fail the VitePress build; fix the reported target.
- Prepare the API with the build script before starting the frontend directly.
- On Read the Docs, select `docs/cellpack/.readthedocs.yaml` as the build configuration.
- Follow your compute environment's policy for compilation and example execution.
