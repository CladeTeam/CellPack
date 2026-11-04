# Metadata and representation: format

Use `cellpack::format` to inspect **bytes you already hold**. It parses matrix-stream
descriptors and chunk metadata; it does not open files or decode matrix values.
For the wire layout and offset examples, read [File and record format](format.md).

<DocChoices title="Choose the metadata lifetime" :items="[
  { label: 'Borrow for this operation', when: 'The input bytes stay alive and immutable.', detail: 'parse_meta returns a cheap, copyable Meta view borrowing the metadata prefix.' },
  { label: 'Retain metadata independently', when: 'The input buffer will be released or reused.', detail: 'OwnedMeta keeps only the validated metadata prefix. Call view() to borrow it without reparsing.' }
]" />

For ordinary matrix reads, start with [store](reference-store.md). For encoding,
decoding and mapping rows in memory, use [scans](reference-scans.md).

## Chunk parsing and ownership

| Entry point | Required input | Result |
|---|---|---|
| `parse_meta(bytes)` | Bytes beginning at a chunk's `CPAK` header, covering the entire declared metadata prefix | `Result<Meta<'_>>`; sections borrow `bytes` |
| `OwnedMeta::new(bytes)` | The same prefix, or a complete chunk | Validates and copies only `meta_len` bytes |
| `OwnedMeta::from_vec(bytes)` | An owned vector with that prefix | Validates, takes ownership and discards bytes after `meta_len` |
| `OwnedMeta::bytes()` | A retained owner | Borrowed metadata bytes, exactly `meta_len` long |
| `OwnedMeta::view()` | A retained owner | A `Meta` tied to the borrow of the owner, without reparsing |

The records region need not be present. `parse_meta` checks the header, metadata
section sizes, model frequencies, aggregate bounds, and the endpoints of the nnz
table. It does **not** walk every row's table entry or validate record payloads.
The owned form applies the same validation; owning bytes adds no stronger record
guarantee.

### Query shape and allocation bounds

| Query | Meaning and units |
|---|---|
| `n_cells / n_dims` | Local chunk row count / column count |
| `value_width / index_width` | Bytes per stored value / raw column ID; widths do not identify the numeric dtype |
| `max_nnz` | Largest declared row nnz: stored entries for sparse rows, nonzero values for dense rows |
| `max_elems` | Largest stored element count; a dense row stores `n_dims` elements |
| `max_k / nnz_total` | Largest row-local dictionary / sum of declared row nnz |
| `renumbered / column_modes` | Whether coded IDs were permuted / optional raw mode values in coded order |

A four-byte value could be `U32`, `I32` or `F32`. A CPK stream's `Dtype` supplies
that interpretation; a standalone chunk only declares the byte width. Use codec
sizing functions for decode buffers rather than treating `max_nnz` as a dense
output length.

### Locate bytes and rows

| Query | Coordinate system or buffer requirement |
|---|---|
| `meta_len()` | Metadata-prefix byte length, excluding padding before records |
| `records_start()` | Byte offset **from the chunk start**, including any page-alignment padding |
| `records_len() / chunk_len()` | Records-region bytes, including gaps / complete chunk bytes |
| `page_size()` | `Some(bytes)` for packed records, `None` for an unpacked chunk |
| `record_range(cell)` | Byte range **from the records-region start**; `cell` is a zero-based local chunk row |
| `records_of(chunk)` | Borrows the records region from the matching complete chunk buffer |
| `record(records, cell)` | Borrows one row's encoded bytes from the complete records-region buffer |
| `cell_nnz / cell_layout / cell_kind` | Declared count / sparse-or-dense layout / raw flags for one local row |

`record_range` and `cell_nnz` validate the requested row's extent or nnz entries.
Out-of-range local rows return `Error::RowOutOfRange`; invalid table entries or
short chunk/records buffers return `Error::CorruptMeta`. An empty row may have an
empty record slice.

`records_of` and `record` check extents, not the identity of a supplied buffer.
Keep each `Meta`, records buffer and decode tables associated with the same
encoded chunk. Returned record slices borrow the supplied records bytes, not
the `Meta` value.

### Read a record count without decoding

`record_nnz(record)` returns the count in the record header and checks every byte
it reads. An empty record returns zero; a missing, truncated or overflowing count
returns `Error::CorruptRecord`. This helper accepts arbitrary bytes, but it only
checks that count: it does not validate flags, index/value sections, or agreement
with `Meta::cell_nnz`.

## Matrix-stream parsing: format::cpk

Use these functions when implementing a container or reader. A normal file
reader can obtain the parsed descriptors from `CpkFile` instead.

| Step | Input and result | What is checked |
|---|---|---|
| `parse_superblock(bytes)` | At least `SUPERBLOCK_LEN` bytes → owned `Superblock` | Magic, version, reserved bytes, dtype/source codes, page size, shape and directory extent |
| `superblock.directory_len()` | Parsed descriptor → `n_chunks * ENTRY_LEN` bytes | Size query only |
| `parse_directory(super_bytes, &superblock, directory_bytes)` | The **same** complete superblock bytes and descriptor, plus at least `directory_len()` directory bytes → owned `Vec<ChunkEntry>` | Header/directory hash, row coverage and aligned, ascending, non-overlapping chunk extents |

Read the directory at `superblock.directory_offset`, measured from the CPK stream
start. `parse_directory` does not fetch it, parse individual chunk metadata, or
check that payload bytes actually exist in the backing file. Keep the original
superblock buffer until directory parsing finishes; the returned descriptors
contain values and do not borrow either input.

| Type or constant | Contract |
|---|---|
| `Superblock` | Matrix shape, numeric dtype, original input layout, page size, chunk count/target rows, directory offset and declared stream length |
| `ChunkEntry` | Matrix `first_row`, local row count, stream-relative byte `offset`, total chunk `len` and metadata `meta_len` |
| `Dtype` | Ten numeric types; `width()` is bytes per value; `f32(bytes)` converts the first stored value |
| `SourceLayout` | The matrix arrived as CSR or dense; not a per-record coding decision |
| `RecordLayout` | One record stores sparse entries or every column's value |
| `SUPERBLOCK_LEN / ENTRY_LEN` | 4096 / 40 bytes; exported under `format::cpk` |

`Dtype::f32(bytes)` requires at least `width()` bytes; a short slice can panic.
Integer values beyond exact `f32` representation and `F64` values can round.

::: warning Validation boundary
The superblock/directory hash is non-cryptographic and excludes record payloads.
Successful metadata parsing does not make arbitrary inner records safe to
decode. Follow the [trusted-data contract](contracts.md#failure-and-validation-boundary).
:::

## Complete chunk API

<!--@include: ./_catalog/format.md-->

## Complete matrix-stream API

<!--@include: ./_catalog/format-cpk.md-->
