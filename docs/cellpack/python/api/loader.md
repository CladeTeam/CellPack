# Native loader API

```python
from pycellpack.loader import DataSource, OutputSchema, LoaderConfig, Loader, load, allowed
```

Use the native loader for explicit column maps, output schemas and NumPy batch
delivery. For complete inputs and checked results, follow
[Native NumPy loading](/python/docs/native-loader.md). For dictionaries of Tensor
fields, use [ScDataLoader](dataloader.md).

<DocChoices title="Choose the result you need" :items="[
  { label: 'load(source, cells)', when: 'Read one full-vocabulary float32 matrix', detail: 'Returns an independent array, or writes your exact out buffer.', href: '#one-shot-load' },
  { label: 'Loader → Run', when: 'Stream named or processed outputs', detail: 'Returns batches whose NumPy views retain output-slot leases.', href: '#start-a-run' },
]" />

## Register a DataSource

`DataSource(matrices, vocab, *, page_size=None, direct_io=None)` reads metadata and
registers `(source, ids)` pairs in order. It does not decode matrix values or infer
gene names. `vocab` is a positive standard-ID-space size; kept IDs lie in
`0 <= id < vocab`.

| Input | Contract |
| --- | --- |
| Path | A finalized, immutable CellPack matrix stream |
| `File` | A file-backed stream or bounded embedded stream; in-memory streams are rejected |
| `Store` | A two-dimensional, row-only view; a gene-selected view is rejected |
| `ids` | One integer per **stored column**, mapping source column → standard ID; `-1` or `uint32.max` drops a column; kept IDs are distinct |
| `page_size=None` | Resolve from the first stream, or 4096 for an empty source; all registered streams must agree |
| `direct_io=None` | Probe filesystem support and select direct or buffered data reads; metadata stays buffered |
| `direct_io=False / True` | Force buffered reads / request direct I/O; `True` does not promise the filesystem accepts every read |

Global cell IDs number the contributed rows consecutively. A Store selecting
stored rows `[5, 2]` contributes two global cells in that order. Accesses address
these contributed rows, rather than the original file row numbers.
Registration uses that Store's underlying file dtype; choose conversion on the
OutputSchema rather than on a lazy Store dtype view.

| Operation | Returns / ownership |
| --- | --- |
| `add(source, ids)` | First global cell ID of the new matrix; rebuilds this DataSource snapshot |
| `matrix_cells(matrix)` | `(first_global_cell, contributed_count)`, or `None` for an absent matrix |
| `stored_cells(accesses)` | Original stored-row IDs for the supplied global cells |
| `cell_pages(cell)` | Borrowed file descriptor, aligned offset and page count |
| `cell_bytes(cell)` | Borrowed file descriptor, exact offset and byte length |

Keep the source alive while using its borrowed descriptors; do not close those
file descriptors. The two descriptor methods accept negative cells from the end.
Run access arrays instead require nonnegative global IDs.

An existing Loader retains the native catalog and row-count snapshot from its
construction. `source.add(...)` does not extend that Loader; construct a new
Loader to use the added rows. Keep registered files immutable for all active
sources, plans and runs.

## Describe an OutputSchema

| Parameter | Default | Contract |
| --- | --- | --- |
| `out_ids` | Required | Nonempty, distinct standard IDs in desired output-column order; loader binding requires every ID below `vocab` |
| `dtype` | float32 | Native-endian numeric dtype for mapped input and built-in output |
| `id`, `name` | None | Plan identity and batch output name; the Loader binds unset values |
| `default` | 0 | Scalar converted to `dtype` at construction, used for source-absent columns |
| `process` | None | Native callback; None selects built-in dense rows |
| `row_bytes` | None | Fixed processed bytes per position; inferred from a declared ProcessFn layout, otherwise required for process output |
| `out_dtype` | None | Optional process-output element view; `row_bytes` must be divisible by its item size; illegal for built-in output |
| `scratch_bytes` | 0 | Callback scratch per worker, in bytes |

For built-in output, `batch[name]` has shape `(batch.rows, len(out_ids))` and dtype
`dtype`. For process output it is raw uint8 rows, or an `out_dtype` view of those
rows. `batch.region(name)` always returns uint8 `(batch.rows, row_bytes)` storage.
Field interpretation belongs to the [process-function contract](process-functions.md).

A present source column with no sparse entry contributes **zero**. A standard ID
absent from that source contributes `default`; defaults do not replace existing
zeros. Source and schema maps use standard IDs, rather than output positions.

`allowed(source_dtype, target_dtype)` tests loader conversion support. A schema
dtype is checked against every source route when the Loader is built. Codec or
NumPy conversion support does not imply loader support.

## Bind a Loader

`Loader(source, outputs=None, *, config=None, **settings)` accepts one schema, a
schema sequence, or an ordered name → schema mapping. At most 64 schemas are
supported. `outputs=None` creates float32 full-vocabulary output named `X`.

Mapping keys name outputs; otherwise a schema's name is used, falling back to its
assigned ID as a string. Names and IDs must be distinct. Explicit IDs are kept;
unset IDs receive the smallest unused values. `names` preserves output order and
`schemas` is a read-only mapping of the bound schemas.

Supply a LoaderConfig **or** its fields, never both. LoaderConfig is frozen;
unset defaults resolve when it is constructed.

| Field | Default | Unit / behavior |
| --- | --- | --- |
| `workers` | Usable CPUs, at most 16 automatically | Native worker threads; an explicit value is not capped at 16 by this default |
| `backend` | Probe | `"uring"` where supported, otherwise `"pread"` |
| `inflight` | 64 | Positions in flight per worker |
| `claim_batch` | 8 | Accesses per cursor claim, rounded to positions and bounded by slot/inflight sizes |
| `slot_rows` | 256 | Positions per output slot, shared by every schema |
| `window` | Derived | Execution lead in slots; defaults depend on the planning mode |
| `profile` | None | None/False untimed; True all sites; site/group names select timing |
| `cpus` | `()` | Logical CPU IDs; worker `i` is pinned to `cpus[i % len(cpus)]`; empty inherits affinity |

Worker, inflight, claim and slot counts must be positive, as must an explicit
window. Use [statistics](statistics.md) to select profiling sites and interpret
reservations and timings.

## Start a run

`loading.start(accesses, *, plan=None, ring=None, slots=None, world=None, rank=0)`
starts native production and returns a Run. It accepts native `Accesses`, AccessFn,
or a one-dimensional integer array; array contents are copied natively. Values
must be uint32-range global cell IDs within this Loader's source snapshot.

With `S` schemas, length counts **accesses** and must be divisible by `S`:
access `k` belongs to position `k // S`, schema `k % S`. A bare `[5, 1]` with two
schemas describes one position with different cells. Use `Accesses.replicate`
for the same cell in each schema, or `Accesses.offset` for paired cells.
See [access sequences](access-functions.md).

| Keyword | Meaning |
| --- | --- |
| `plan=None` | No planned cache |
| `plan=Plan` | Execute a plan bound to this sequence and compatible loader |
| `plan=PlanOptions` or byte budget | Compile ahead of workers during execution |
| `ring=None` | Allocate an output ring; `slots` defaults to 4 and `world` to 1 |
| `ring=Ring` | Use its existing slots/world/layout; do not also supply `slots` or `world` |
| `rank=0` | Rank consumed through this Run's iterator; batch `b` belongs to `b % world` |

A caller-supplied ring must match output names, view dtypes, row byte sizes and
slot layout. `ring_layout` describes memory; `ring` allocates it. These methods
default to `slots=4, world=1`.

Consume batches before `run.join()`. A NumPy output view retains its slot lease;
copy data that must outlive immediate consumption. `join()` returns final RunStats,
and context exit closes the Run. See [rings and lifecycle](rings.md) for precise
lease release, cancellation, manual consumers and reuse rules.

`start` can issue a RuntimeWarning when estimated known reservations exceed
available process memory. This is a diagnostic estimate, not a hard memory limit.
It excludes an already allocated caller-supplied ring from the additional-memory
comparison. See [planning](planning.md) for `compile`, `compile_stats`, `replay`
and `memory_estimate`.

## One-shot load

`load(source, cells=None, *, out=None, config=None, **settings)` returns float32
shape `(selected_rows, source.vocab)`, preserving row order and repetitions.
`cells=None` selects all rows; this convenience normalizes positional selectors,
including slices, masks and negative row indices.

`out` must be a writable, C-contiguous NumPy float32 array of the **exact** result
shape, aligned to 8 bytes. It is returned unchanged by identity. Without `out`,
the function allocates it. An empty selection returns a correctly shaped empty
array without starting a run. The convenience sets its slot size to the entire
selection, so use streaming Loader output for bounded batch memory.

## Errors and result validity

Python type/protocol mistakes raise TypeError; invalid IDs, layouts, schema
declarations or configuration generally raise ValueError. Source descriptor
indexing can raise IndexError. I/O failures raise OSError; busy, cancellation and
worker failures raise RuntimeError. Asynchronous failures may surface during
iteration or `join()`. Discard incomplete output after a failed run.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/loader-members.md-->

:::
