# Errors and crate-root exports

The crate has two error domains. Storage, codecs, metadata and runtime use
`cellpack::Error` / `cellpack::Result<T>`. Loading and planning use
`cellpack::loader::Error` / `loader::Result<T>`.
The root also re-exports `CpkFile` and `CpkWriter` for convenience.

<DocChoices title="Start with the failing operation" :items="[
  { label: 'Read, write or decode', when: 'A synchronous storage or codec call returned Err', href: '#format-codec-and-storage-errors', detail: 'Inspect cellpack::Error. Failed output may be partial; correct the cause before reading again.' },
  { label: 'Prepare or execute a run', when: 'A source, plan, worker or consumer failed', href: '#loader-and-planner-errors', detail: 'Inspect loader::Error and distinguish rejected preparation from cancellation of an active run.' },
]" />

This page explains what to report and what can be reused. For a working consumption
pattern, see [loading and ownership](loading.md); for the exact validated/trusted
boundary, see [data contracts](data-contracts.md).

## Format, codec, and storage errors

| Variant | Meaning | Recovery scope |
|---|---|---|
| `InvalidInput / InvalidOptions` | An argument or declaration violates the API | Correct inputs/options before retrying |
| `OutputTooSmall { need, have }` | Output extent differs from the required size | Use the operation's sizing contract; exact-size calls can also reject oversized buffers |
| `WorkspaceTooSmall` | Scratch is undersized | Allocate the requested number of `u64` words |
| `OutputTypeTooNarrow` | Index/offset type cannot hold a required value | Use a wider output integer type |
| `RowOutOfRange` | Requested row/cell/chunk position is invalid | Correct the selector and id space |
| `TablesMismatch` | Tables belong to different metadata | Rebuild/bind the correct tables |
| `CorruptMeta / CorruptRecord` | An explicitly checked consistency rule failed | Reject the affected input/output |
| `Io` | File operation failed | Inspect preserved error kind and message |

These errors do not imply every malformed record is checked.
Trusted inner-record corruption can violate optimized decoder assumptions.
After a failed read/decode, discard potentially partial output.

`OutputTooSmall` sizes follow the particular operation's output unit; they are
not universally byte counts. `WorkspaceTooSmall` always reports `u64` words.
Use the [shape and capacity reference](contracts.md#shapes-and-capacity) rather
than inferring allocation units from the variant name.

`Error::Io { kind, message }` preserves the OS error kind and formatted message,
not the original `std::io::Error` object. Match `kind` for programmatic decisions
and retain `message` for diagnostics. `CorruptRecord` reports an explicit failed
comparison; it is not evidence that decoding arbitrary damaged records is safe.

## Loader and planner errors

| Variant | Meaning | Recovery scope |
|---|---|---|
| `Config(String)` | Invalid declaration or protocol use | Correct configuration or call order |
| `Access { position, cell }` | Sequence names a cell outside the source | `position` is the access index `k`; check registration ranges and schema expansion |
| `Io` | OS I/O failure or short read | Inspect the OS error and input/storage state |
| `CellPack` | Wrapped format/codec/storage error | Preserve and inspect its underlying cause |
| `Plan(String)` | Incompatible plan, costs or planning options | Correct options or recompile for the actual graph |
| `Process(status)` | Native process callback failed | Interpret application-defined status |
| `Accesses(status)` | Native sampling callback failed | Interpret application-defined status |
| `WorkerPanic` | A worker or compiler thread panicked | Fail the run and inspect diagnostics |
| `Busy` | A loader/run or ring producer is already occupied | Finish the existing work and drop its run handle before retrying |
| `Aborted` | Cancellation was observed | Stop this run; identify its originating cause |

For `S` schemas, access index `k` belongs to logical position `k / S` and schema
index `k % S`. The field name `position` in `Error::Access` does **not** refer to
that logical position. A wrong cell may originate in a native sampler, a stored
sequence, or an incorrect per-schema offset.

### Preserve the cause

`loader::Error::Io` retains its `std::io::Error`; `CellPack` retains the underlying
`cellpack::Error`. Both participate in `std::error::Error::source()`. Report the
operation context with the error chain, rather than replacing it with a generic
"loading failed" message. Callback statuses are application-defined nonzero
codes, not portable OS error numbers.

A producer or worker failure cancels other participants, which can then report
`Aborted`. The producer joins its threads and prefers a non-`Aborted` cause over
secondary cancellation. `for_each_batch` also preserves a consumer callback's
error; if consumption only observed cancellation, it returns the producer's
result instead. With several concurrent failures, this is not a guarantee of
which failure happened first in wall-clock time.

### Decide what to reuse

| Failure point | Meaning for the next attempt |
|---|---|
| Source/schema/plan construction or rejected `prepare` | Correct the declaration or file state. No successful prepared run was returned |
| `Busy` | Another owner is active; it is not a request to abort that owner's work |
| `Config` saying the ring has ready/leased slots | Consume remaining ready output and release leases after successful production, or use a fresh ring |
| Failure during producer execution, or a `Run::wait` error | The run is cancelled; its ring is permanently aborted |
| Explicit `Run::cancel` or dropping a prepared but unstarted run | The ring is permanently aborted |
| Successful run | Drain output, release every lease, and drop the old run handle before reusing the loader and ring |

An aborted ring cannot be reset or prepared again. Allocate a fresh ring for a
new attempt after fixing the originating cause. Keep old backing storage alive
while any lease still references it, and discard incomplete output. A wrong-rank
`wait` is a protocol error that cancels the run even though its variant is
`Config`; the variant alone does not tell you whether a run remains usable.

If progress stops without an error, check [backpressure and held
leases](troubleshooting.md#a-run-stops-making-progress) before treating the wait
as an I/O failure.

## Root interface index

The crate root exports `Error`, `Result`, `CpkFile` and `CpkWriter`. Loading and
planning use their own `loader::Result`, so qualify the result type when combining
the two domains in one module. Other interfaces live in their named modules.

<!--@include: ./_catalog/root.md-->
