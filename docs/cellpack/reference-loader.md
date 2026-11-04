# Training data delivery: loader

`cellpack::loader` binds immutable matrix sources to output schemas, evaluates
the application's access sequence, and produces ordered batches into a bounded
ring. It owns read/decode/map/process scheduling. It does not own sampling policy,
distributed process creation, model execution, or GPU transfer completion.

## Find the contract for your task

<DocChoices title="Follow one boundary at a time" :items="[
  { label: 'Sources and schemas', when: 'Bind files to the model input panel', href: '#source-registration', detail: 'Registration maps source columns to standard IDs. Schemas choose output order and conversion.' },
  { label: 'Accesses and callbacks', when: 'Define which row each output requests', href: '#access-sequences-and-native-access-abi', detail: 'Sequences use global cell IDs in position-major, schema-minor order.' },
  { label: 'Runs and rings', when: 'Start work and read bounded output', href: '#loader-and-run-lifecycle', detail: 'Preparation reserves a generation. Production and consumption must overlap.' },
  { label: 'Process ABI', when: 'Transform mapped rows inside workers', href: '#process-function-abi', detail: 'Input is call-local; output is a fixed record whose size and meaning you declare.' },
]" />

For an end-to-end program, use [loading and ownership](loading.md). This page
collects argument, identity and lifecycle rules; the interface index at the end
provides exact declarations and member links.

## Public object model

| Object | Responsibility | Input → output |
|---|---|---|
| `DataSourceBuilder` | Register streams and compose column identities | Files + source-to-standard maps → source |
| `DataSource` | Immutable catalog and metadata shared by planner/workers | Global cell → input location and shape metadata |
| `OutputSchema` | Declare one delivery's panel, dtype, default and record size | Mapped input → dense vector or process record |
| `Accesses` | Define deterministic cell selection | Access index → global `CellId` |
| `Loader` | Bind source, schemas and execution configuration | Accesses + ring + planning → prepared run |
| `Run` | Own the producer lifecycle | Concurrent production/consumption → final statistics |
| `RingLayout / Ring` | Describe/own shared bounded output | Slot layout → reusable leased storage |
| `Batch / BatchLease` | Prevent slot overwrite while reading | Completed slot → valid per-schema byte regions |
| `ProcessFn` | Transform one mapped row | CSR/dense ABI row → fixed-size native record |

## Source registration

`DataSourceBuilder::new(vocab, page_size)` requires a positive standard vocabulary
below `u32::MAX` and a supported power-of-two page size.
`direct_io(enabled)` configures record-file opening.
`add_cpk(path, to_std)` and `add_cpk_region(path, offset, length, map)` return a
matrix index. Every map entry is a distinct in-range standard id or `UNMAPPED`.
The map length equals the stored column count, and all streams use the same page size.
`build` consumes the builder and returns a source.

| DataSource query | Return and units |
|---|---|
| `n_cells / n_pages / n_matrices / vocab` | Counts in their respective global spaces |
| `matrix_cells(matrix)` | Optional first global `CellId` and `u64` row count |
| `matrix_dtypes` | Stored dtype per registered matrix |
| `page_size / max_cell_pages` | Bytes per page / maximum pages covering a record |
| `resident_metadata_bytes` | Known resident metadata allocation estimate |
| `fingerprint / mapping_fingerprint` | Compatibility identities, not cryptographic integrity proofs |
| `cell_pages` | Borrowed file descriptor, aligned byte offset and page count |
| `cell_bytes` | Borrowed file descriptor, exact record byte offset and length |

Descriptors remain owned by the source; do not close them. Keep the source and
input files alive and immutable while using returned locations.

## Schemas and conversion

`OutputSchema::dense(id, out_ids, dtype)` fixes dense output width to
`out_ids.len() × dtype.width()`.
`OutputSchema::process(id, out_ids, dtype, cell_bytes, process, identity, scratch_bytes)`
fixes a custom record size independent of input-panel width.

| Declaration | Contract |
|---|---|
| `id` | Unique schema id within a loader |
| `out_ids` | Nonempty distinct standard ids, in output order |
| `dtype` | Input is converted to this type before built-in delivery or callback |
| `cell_bytes` | Positive fixed bytes per cell; dense built-in size is derived |
| `identity` | Versions custom callback behavior and retained configuration |
| `scratch_bytes` | Private scratch available to a process call |
| `with_default / with_default_bytes` | Sets missing dimensions only; present omitted values remain zero |
| `named` | Human-readable reporting label, outside plan identity |

Schema accessors expose these settings and `is_builtin / row_format / process_fn`.
The loader checks source-to-output conversion with `allowed`; its policy differs
from unrestricted low-level casts. See [dtype compatibility](contracts.md).

## Access sequences and native access ABI

| Interface | Input | Result and constraints |
|---|---|---|
| `Vec / slice / array / Box / Arc` implementations | Stored global ids | Borrowed or owned fixed sequence |
| `Shuffle::new / subset` | Cell range or subset, epochs, seed | Random-access epoch permutations |
| `Shuffle::blocks` | Block size, group size | Adds locality to generated ordering |
| `Replicate::new` | Per-position sequence, schema count | Same cell for every schema |
| `Offset::new` | Base sequence, per-schema offsets | Paired global cells; checked addition |
| `SliceAccesses::new` | `Arc<dyn Accesses>`, start, length | Bounded interval without copying values |
| Unsafe `ExternAccesses::new` | `AccessFn`, context, length | Native generator of global ids |
| Unsafe `ExternCorpusAccesses::new` | `CorpusAccessFn`, context, length, dataset counts | Native dataset/local-row pairs translated into global ids |

`Accesses::fill(start, out)` must fill exactly the requested range. Calls can be
concurrent, out of order, and repeated; values depend only on the index and
immutable configuration. Length must be divisible by the schema count.

For `S` schemas and `N` positions, the sequence contains `N × S` IDs. Access
`k` supplies schema **index** `k % S` at position `k / S`. `OutputSchema::id`
does not change this order. For two schemas, `[3, 8, 0, 5]` delivers cells
`[3, 0]` to region 0 and `[8, 5]` to region 1. With `slot_cells=2`, that is
one batch with two rows and two output regions.

`AccessFn(context, start, count, out_cells)` writes `count` global `u32` ids.
`CorpusAccessFn(context, start, count, out_datasets, out_rows)` writes `u32`
dataset indices and `u64` local rows. Dataset counts must match registration order.
Callbacks return zero on success; nonzero becomes `Error::Accesses`.
Code and raw context outlive the sequence, tolerate concurrency, and retain no
output pointers. Rust wrappers do not automatically own a foreign context.

## Loader and run lifecycle

| Method | Inputs | Return and behavior |
|---|---|---|
| `Loader::new` | `Arc<DataSource>`, schemas, `LoaderConfig` | Immutable bound loader; validates graph and conversions |
| `ring_layout` | Slot count, world size | Layout derived from schema byte sizes and slot positions |
| `prepare` | Borrowed `Accesses`, `Arc<Ring>`, `Planning` | Borrowing `Run`; checks files, layout, count and plan binding |
| `Run::run` | Prepared run | Producer execution on calling thread; returns `RunStats` |
| `Run::wait` | Rank and batch index | `Result<Option<BatchLease>>`; none past sequence end |
| `Run::for_each_batch` | Borrowed-batch callback | Scoped producer plus ordered local consumption of all ranks |
| `Run::snapshot` | Active run | Published, potentially lagging statistics |
| `Run::cancel` | Active/prepared run | Wakes parties and aborts further production |
| `batches / generation / ring` | Run | Batch count, run identity and ring handle |

A run borrows the loader and sequence. One run holds a loader at a time; drop it
before preparing another, even after `run()` succeeds. `prepare` does not launch
workers. `Run::run` may execute once; subsequent attempts return `Busy`.

Production and consumption must overlap once a sequence can fill the ring.
`for_each_batch` provides this overlap and joins its producer on success or
failure. For a custom consumer, call `wait(rank, b)` only for batches owned by
that rank, in increasing order, while another thread executes `run()`.

`cancel()` permanently aborts this run's ring; it cannot be reused afterward.
Dropping an unstarted run cancels it too. A failed run's output is incomplete
and cannot be treated as success. Reuse after success additionally requires all
ready batches consumed and all leases released.

Planning-related methods `plan_options, compile, replay, replay_with_cost,
compile_online, plan_online, online` are described in [opt](reference-opt.md).
`memory_estimate_for` describes a particular planning mode; a cache budget alone
does not bound total process memory.

## Ring layout and ownership

| Constructor/family | Memory model | Caller obligation |
|---|---|---|
| `Ring::allocate` | Rust-owned control and data | Retain ring through consumers |
| Unsafe `init / attach` | External contiguous storage | Stable owner, alignment, capacity and protocol |
| Unsafe `init_regions / attach_regions` | Separate control and data | Correct independent regions, no overlap |
| Unsafe `with_buffer` | External data with private control | Stable data owner and compatible layout |

`RingLayout` stores slots, positions per slot, world, and per-schema cell bytes.
`bytes, control_bytes, data_bytes, data_min_bytes, slot_stride, regions` report
checked sizes/offsets. Control alignment is 64 bytes; region starts are 64-byte
aligned; slot stride and contiguous storage use 4096-byte alignment.

Batch `b` occupies slot `b % slots` and belongs to rank `b % world`.
All schemas and positions of a slot must finish before it is ready.
`Ring::wait` yields a borrowed `Batch`; `acquire / acquire_generation` yield an
owned `BatchLease`; `acquire_timeout` returns `Batch, Pending,` or `Done`.

`region(s)` exposes exactly `rows() × cell_bytes(s)` bytes.
`data()` can include padding and unused tail space.
Drop/release the lease only after all readers finish. A ring can serve another
run after its previous slots and leases are released; an aborted ring cannot.
Participant methods and headers are advanced shared-process protocol interfaces,
not process-launching utilities.

## Process function ABI

`ProcessFn::new` binds both CSR and dense callbacks in `Auto`.
`with_format` requires only callbacks needed by forced `Csr` or `Dense`.
Both auto callbacks must compute equivalent outputs. Each sees already mapped
columns and values converted to the schema dtype.

`ProcessContext` identifies ABI version, row struct size, logical position, global
cell, schema index, dtype code and user data. `CsrRow` has element count, columns,
values, output and scratch; columns are not necessarily sorted after mapping.
`DenseRow` supplies a full panel including missing-dimension defaults.

Callbacks initialize the fixed output record, write only owned output/scratch,
are concurrent-safe, retain no input pointer, and never unwind across the C ABI.
Nonzero status becomes `Error::Process`.
Unsafe `with_partial_rows` additionally requires handling null value/column pointers
through a call-local resolver and valid element positions.
See [ProcessFn details](process-functions.md) for the complete callback contract.

## Configuration, statistics, and failures

[Performance](performance.md) lists every `LoaderConfig` default and its units.
`IoBackend::Uring` requires io_uring support; choose `Pread` explicitly otherwise.
`Counters` are always on; `RunStats` combines event, I/O, per-schema and optional
profile/online reports. The generated pages below enumerate every field.

Configuration errors, invalid cells, I/O, incompatible plans, callback statuses,
worker panics, busy state, and cancellation have distinct [error variants](reference-errors.md).
A secondary `Aborted` can be the consequence of another participant's original error.

## Complete interface index

::: details Declarations and member links by interface
<!--@include: ./_catalog/loader.md-->
:::
