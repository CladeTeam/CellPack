# Storage and selection reads

<script setup>
import RustStoreSelection from './.vitepress/theme/components/RustStoreSelection.vue';
</script>

Use `cellpack::store` when you want to write one matrix or synchronously read a
selection into your own memory. Start with a finalized `.cpk` file and keep it
unchanged while reading. For background batch delivery, continue to
[loading and ownership](loading.md).

<DocChoices title="Choose your read path" :items="[
  { label: 'A contiguous range, stored dtype', when: 'Read a few rows with allocation handled for you.', detail: 'CpkFile::read_dense / read_csr return stored value bytes; retain the file dtype to interpret them.', href: '#allocating-convenience-methods' },
  { label: 'Typed, mapped, or repeated reads', when: 'Reorder rows or columns, convert values, or reuse buffers.', detail: 'Prepare a ReadSelection, size outputs from its spec, then reuse a ReadWorkspace.', href: '#select-rows-map-columns-and-convert-values' }
]" />

## Write a row stream

<DocDiagram name="rust-storage" />

`CpkWriter::create(path, n_cols, dtype, source_layout, options)` creates or
truncates a file. Calls to `append` need not align with chunk boundaries; the
writer stages incomplete chunks. `end_chunk` ends a chunk early, and `finish`
flushes the final chunk and writes the directory.

- `DenseRef` borrows row-major value bytes and declares the column count.
- `CsrRef` borrows `u64` row offsets, column indices, and value bytes.
- `CsrView` accepts row offsets in other integer types through `Offsets`.

CSR columns must be strictly increasing, distinct, and in range within each row.
Each value occupies 1, 2, 4, or 8 little-endian bytes, interpreted by the file dtype.
A writer accepts input in a different form from its declared source layout and
converts that form before encoding.

The writer borrows input only for `append`; you can reuse that input once the call
returns. Appends preserve row order. A new append adds rows to the same matrix;
it does not create another named matrix or update previously written rows.

::: info
Successful `finish` completes encoding and file writes. It does not implement an
atomic file replacement or power-loss durability protocol. Applications needing
those guarantees must manage temporary paths, synchronization, and publication.
:::

## Select rows, map columns, and convert values

`RowSelection::List(&[3, 0, 3])` preserves order and repeated rows. The storage
`gene_map` maps **each source column to an output column**; `UNMAPPED` drops it.
Mapped targets must be distinct. Output columns without a source are zero-filled.
This map differs from a loader's standard-vocabulary map.

For the [shared four-row fixture](quickstart.md), the map
`[1, UNMAPPED, 0, UNMAPPED]` keeps source columns 0 and 2, swaps their output
positions, and drops the other columns:

<RustStoreSelection />

The sequence is **prepare → inspect `spec` → allocate → read**. Preparation reads
chunk metadata and keeps an open file descriptor; each read loads record bytes.
It is useful to prepare once when reading the same selection several times, but
preparation does not preload the selected values.

`read_dense_into::<f64>` can convert stored `f32` values into `f64`. Consult
[OutputElement](/generated-store-trait-OutputElement) for supported output types.
Conversion changes numeric representation; narrowing can lose precision.

### Complete runnable example

This program checks both dense and CSR outputs above, then writes and reads a
second `u16` CSR matrix. It prints
`selection, conversion and CSR round-trip passed` when its assertions succeed.

<<< @/examples/storage.rs

::: details Shared fixture and encoding options
`common::fixture()` creates `matrix.cpk` in the supplied output directory. It
uses two rows per chunk so the row request exercises multiple chunks.

<<< @/examples/common.rs
:::

Run from the repository root on a suitable build worker. Use a disposable output
directory: the example recreates `matrix.cpk` and `counts.cpk` there.

```sh
cargo run --locked --manifest-path docs/cellpack/examples/Cargo.toml \
  --bin storage -- tmp/cellpack-guide/storage
```

## Capacity and actual nonzero count

`ReadSelection::spec()` reports row count, output columns, stored dtype, and CSR
capacity:

- A dense output must contain exactly `spec.dense_len()` elements.
- CSR `indptr` has exactly `spec.rows + 1` entries. Indices and values each have
  capacity for at least `spec.nnz` elements.
- `spec.nnz` is a capacity computed from source records. Dropping columns can
  reduce the actual nnz returned by `read_csr_into`.
- Use indices and values only through the returned nnz. The remaining capacity
  has unspecified contents.

Reuse `ReadWorkspace` between reads; it retains scratch, not outputs. A workspace
serves one read at a time. `clear()` releases retained scratch. A prepared selection
can serve repeated reads, provided the underlying stream remains unchanged.

An I/O or codec error can leave partially written output. Discard that invocation's
output when it fails.

::: tip Reuse the allocation, keep the result
After a successful read, output belongs to you and remains valid while the
workspace serves another read. Concurrent reads require separate outputs and
separate mutable workspaces. The selection itself owns its metadata and file
descriptor, so it does not borrow the temporary row list or map passed during
preparation.
:::

## Allocating convenience methods

`CpkFile::read_dense(range)` returns stored little-endian value bytes in `Vec<u8>`,
rather than a typed `Vec<f32>`. `read_csr(range)` returns a `Csr` with `u64`
`indptr`, `u32` `indices`, and stored-dtype bytes in `data`.
Use prepared reads for typed output or buffer reuse.

## Read policies

`ReadOptions` controls record-range merging within a chunk:

| Setting | Purpose |
|---|---|
| `ReadStrategy::Auto` | Select ranges or the whole records region using byte and request costs |
| `ReadStrategy::Ranges` | Read selected record ranges |
| `ReadStrategy::WholeChunk` | Read the entire records region |
| `max_gap_bytes` | Maximum gap crossed while merging ranges |
| `request_cost_bytes` | Equivalent byte cost assigned to one request |

`ReadSelection::io_summary()` reports the prepared record bytes, read bytes,
request count, and whole-region bytes. These describe application reads, not
measured block-device traffic.

Read worker counts are clamped to at least one and at most the number of selected
chunks. `set_threads(n)` on the writer likewise parallelizes complete chunks,
defaulting to one thread; serial and parallel writes are byte-identical. Both
forms of parallelism require additional scratch. Tune them after the basic
workflow is correct; they are independent of the asynchronous loader's workers.

## Embedded streams

`CpkFile::open_region(path, offset, Some(length))` opens a `.cpk` stream inside
another file. Directory offsets are relative to the stream start. The application
must supply the correct region; this Rust reader does not parse HDF5 or AnnData
metadata to locate it.

API: [CpkWriter](/generated-store-struct-CpkWriter),
[CpkFile](/generated-store-struct-CpkFile),
[ReadSelection](/generated-store-struct-ReadSelection).
