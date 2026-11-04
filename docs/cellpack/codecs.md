# Codec components

Use `cellpack::scans` when you own the container, allocation policy, or record-read
strategy. It turns borrowed matrix rows into encoded chunks and decodes records
into your buffers. You do not need a loader or a `.cpk` file. If you want to open,
write, or select rows from complete matrix files, start with [storage](storage.md).

<DocChoices
  title="Choose what to decode"
  :items="[
    { label: 'A resident chunk', when: 'The whole chunk is already in memory.', detail: 'Use decode_csr or decode_dense with a chunk-local Rows selection.' },
    { label: 'One fetched record', when: 'Your container fetches individual rows.', detail: 'Keep metadata and tables resident; use Decoder with caller-owned scratch.' },
    { label: 'A row to map or transform', when: 'You will filter columns, cast values, or scatter into a panel.', detail: 'Decode Parts and consume its values before reusing the workspace.' }
  ]"
/>

## First round trip

This complete example encodes two CSR rows, packs the records onto 4096-byte
pages, then reconstructs the CSR in source column order. Input values are
little-endian `u16` bytes; compression does not convert their numeric type.

<DocDiagram name="codec-path" />

<<< @/examples/codec.rs

In the application from [installation](installation.md), save
[codec.rs](/downloads/rust/codec.rs) as `src/main.rs` and
[common.rs](/downloads/rust/common.rs) as `src/common.rs`, then run:

```sh
cargo run
```

The assertions verify the result:

| Array | Recovered contents |
|---|---|
| Row offsets | `[0, 3, 5]` |
| Column indices | `[0, 2, 5, 1, 2]` |
| Values interpreted as `u16` | `[1, 1, 3, 2, 1]` |

The printed packed length depends on the encoding choices; it is not the dense
matrix size. The example uses [the same encoding options as file writers](storage.md).

### 1. Size, encode, and keep the written prefix

`chunk_sizes` supplies **output bytes** and **workspace `u64` words** for the exact
input selection and options. Allocate them as `Vec<u8>` and `Vec<u64>` respectively.
`encode_chunk` returns the number of meaningful output bytes: truncate to that
length before handing the chunk to another operation.

`Rows::List` selects chunk-local input rows in the listed order, including repeats.
These are not global loader cell IDs or file-level row selectors. CSR inputs must
have sorted, unique indices within each row; see [matrix inputs](contracts.md#matrix-inputs).

### 2. Pack only when the container needs page-oriented reads

`packed_len` sizes a separate output buffer; `pack_chunk` rewrites record offsets
while retaining logical row identities. Packing is optional for in-memory use.
To obtain its page-read guarantee on disk, place the chunk at a page-aligned
position and read with the same page size. Packing accepts powers of two from
512 through 65536 bytes.

Always parse the metadata of the **packed result** after packing. Offsets from the
unpacked chunk no longer describe that result.

### 3. Parse once, build tables once, reuse scratch

`parse_meta` borrows the chunk's metadata bytes. `decode_tables` builds the lookup
tables for that metadata. `meta.records_of(&packed)` identifies the records region;
chunk decoding takes that region, rather than the whole packed buffer.

For a loop over independently fetched records, bind a `Decoder` once per chunk.
The following continues from the example's `meta`, `tables`, and `records`:

```rust
let decoder = Decoder::new(&meta, &tables)?;
let mut workspace = vec![0u64; decoder.workspace_words()];
for row in 0..meta.n_cells() {
    let record = &records[meta.record_range(row)?];
    let parts = decoder.parts(record, IdSpace::Coded, &mut workspace)?;
    // Consume or copy parts here, before the next decode reuses workspace.
    assert!(parts.nnz() <= parts.elements());
}
```

Metadata and tables are reusable across this chunk's records. Workspace is
exclusive to each active decode; give concurrent decoders separate workspace.
`Parts` can borrow record bytes, metadata, and scratch. It is a temporary view,
not an independently owned decoded row.

## Choose an output representation

- **CSR arrays:** use `decode_csr` or `decode_record`. Size entries from
  `selection_nnz` or `Meta::cell_nnz`; values need that count multiplied by
  `value_width` bytes. Keep only the returned entry count.
- **Dense bytes:** use `decode_dense` or `decode_record_dense`.
  `dense_len` supplies the selection's byte extent. Use `Fill::Zero` unless the
  destination is already all zero. `Fill::AssumeZeroed` is a caller promise;
  reusing a previous row without clearing it can leave stale values.
- **Parts:** use `decode_parts` or `Decoder::parts` to leave dictionary/rank or
  column-mode values unexpanded. Use `expand_values` only if you need every stored
  value materialized. Dense-record `elements()` includes zero entries, so it can
  exceed `nnz()`.

The [API reference](reference-scans.md#decoding-buffers-and-results) gives exact
buffer extents and return contracts.

## Map columns without restoring source order

`IdSpace::Source` restores original columns with ascending indices.
`IdSpace::Coded` keeps the chunk's compression order. If your next step maps into
a panel, compose that permutation once per chunk instead of restoring order for
every record.

For example, suppose coded columns correspond to source columns `[2, 0, 1]`, and
your source-to-output map is `[1, UNMAPPED, 0]`. The coded-to-output map is
`[0, 1, UNMAPPED]`: `coded_to_output[c] = source_to_output[meta.source_dim(c)]`.
Build the map for each chunk; column permutations can differ between chunks.

`map_parts` writes compact columns and values and returns the number kept.
`map_dense` scatters directly into a dense row. Convert dictionary entries and
column modes to the target numeric type once for the relevant view/table, then
pass them with the matching `Parts`. Use `Mapper<S, T>` to reuse dispatch for a
fixed source/target type pair.

::: tip Output order follows the map
Mapped sparse columns are not necessarily sorted. If the next consumer requires
canonical CSR, sort each mapped row together with its values. `UNMAPPED` drops an
element in `map_parts` and `map_dense`; `map_parts_at` instead returns the requested
element positions and writes the sentinel unchanged.
:::

## Records fetched as pages

Use the `_pages` variants when a record spans separate page buffers.
`PagedRecord` describes pages in file order, the page size, the starting offset
inside the first page, and the exact record length. All but the last page must
cover the full page size.

These calls avoid an **application-owned concatenation buffer**. A record wholly
inside one page is borrowed directly; a cross-page record is copied into staging
space inside the caller's decode workspace. The same output, ordering, and
lifetime rules apply.

## Before integrating a custom container

Keep each record paired with its encoder-produced chunk metadata and tables.
Metadata parsing and table checks do not validate arbitrary untrusted record
internals: violating the trusted-record contract can cause undefined behavior in
optimized decoding. The low-level codec is not a hostile-input validation layer.

For container layouts, continue to [the file format](format.md). For exact sizing,
encoding options, mapping rules, and declarations, use the
[scans API reference](reference-scans.md).
