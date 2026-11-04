<script setup>
import RustFormatOffsets from './.vitepress/theme/components/RustFormatOffsets.vue';
</script>

# File and record format

CellPack has three nested byte layouts: a **CPK stream** describes one matrix,
each **CPAK chunk** shares compression metadata across rows, and each **record**
holds one independently decodable row. This page describes version 1; multi-byte
integers and stored values are little-endian.

<DocChoices title="Choose the layer you need" :items="[
  { label: 'Read or write a matrix', when: 'You need selections, numeric conversion or a .cpk file.', href: '/rust/docs/storage.html', detail: 'Use store. It owns file I/O and combines the stream and chunk offsets for you.' },
  { label: 'Inspect bytes or embed a stream', when: 'You are writing a container, file inspector or reader.', href: '/rust/api/reference-format.html', detail: 'Use format parsers and their typed descriptors. Keep stream, chunk and record coordinates distinct.' },
  { label: 'Encode or decode rows in memory', when: 'You already manage the buffers and container.', href: '/rust/docs/codecs.html', detail: 'Use scans with matching metadata, records and coding tables.' }
]" />

## Inspect a chunk without loading its records

Given an existing, finalized `matrix.cpk`, this program opens its directory and
reads only the first chunk's metadata. An empty matrix has no first chunk.

```rust
use cellpack::{CpkFile, Result};
use cellpack::format::parse_meta;

fn main() -> Result<()> {
    let file = CpkFile::open("matrix.cpk")?;
    println!("matrix: {} x {}, {:?}", file.n_rows(), file.n_cols(), file.dtype());

    if let Some(entry) = file.chunks().first() {
        let bytes = file.read_meta(0)?;
        let meta = parse_meta(&bytes)?;
        println!("chunk starts at matrix row {}", entry.first_row);
        println!("metadata: {} bytes; records start: {} bytes into chunk",
                 meta.meta_len(), meta.records_start());
        println!("chunk: {} rows x {} columns", meta.n_cells(), meta.n_dims());
    }
    Ok(())
}
```

`meta` borrows `bytes`; keep the buffer immutable and alive while using the view.
Use `OwnedMeta::from_vec(bytes)` if the metadata must outlive that input buffer.
Metadata-only input supports shape, row counts and record-range queries, but
`records_of` needs a complete chunk buffer.

## Keep byte coordinates separate

`ChunkEntry::offset` and `Superblock::directory_offset` are **stream-relative**.
`Meta::records_start()` is **chunk-relative**. `Meta::record_range(row)` is
**records-region-relative**; its row argument is local to the chunk.

<RustFormatOffsets />

For an embedded stream, add the outer-file start exactly once. `CpkFile::open_region`
accepts that outer offset and handles the translation. To find the local row of
a directory entry, subtract `entry.first_row` from the matrix row ID.

## Matrix stream: CPK

A `.cpk` stream has a 4096-byte superblock, page-aligned chunks, and a final chunk
directory. Each directory entry locates a consecutive range of matrix rows.

<DocDiagram name="stream-layout" />

The writer starts with an invalid superblock and writes the valid one last.
Call `CpkWriter::finish` before opening the result. The header/directory hash
detects damage or mismatches in those two regions; it is neither cryptographic
nor a whole-payload integrity check.

::: details Superblock byte map

Offsets below are from the stream start.

| Offset | Field | Width |
|---|---|---|
| 0 | Magic `CPK\0` | 4 bytes |
| 4 | Version, currently 1 | `u16` |
| 6 | Flags, currently zero | `u16` |
| 8 | Page size | `u32` |
| 12 | Dtype code | `u8` |
| 13 | Source layout code | `u8` |
| 14 | Reserved, zero | `u16` |
| 16 | Matrix row count | `u64` |
| 24 | Column count | `u32` |
| 28 | Chunk count | `u32` |
| 32 | Target rows per chunk | `u32` |
| 36 | Reserved, zero | `u32` |
| 40 | Directory offset | `u64` |
| 48 | Stream length | `u64` |
| 56 | Non-cryptographic hash of bytes `0..56` and the directory | `u64` |
| 64–4095 | Reserved, zero | 4032 bytes |

:::

Dtype codes 0–9 are `U8, U16, U32, U64, I8, I16, I32, I64, F32, F64`.
Source layout codes are CSR (0) and dense (1). Neither determines an individual
record's sparse/dense coding choice. The chunk stores only a value **width**;
the stream supplies the numeric interpretation.

::: details Directory entry byte map

Each entry is 40 bytes. Offsets here are relative to the entry start.

| Offset | Field | Width |
|---|---|---|
| 0 | First matrix row | `u64` |
| 8 | Chunk offset, relative to the stream | `u64` |
| 16 | Total chunk length, including metadata and padding | `u64` |
| 24 | Metadata-prefix length | `u64` |
| 32 | Chunk row count | `u32` |
| 36 | Reserved, zero | `u32` |

Entries tile all matrix rows in order; chunks are aligned, ascending and
non-overlapping. The directory ends the declared stream exactly.

:::

## Chunk: CPAK

A chunk starts with a 64-byte header using magic `CPAK` and version 1. It declares
the row/column counts, value width, rANS lane settings, packed page size, row
bounds, total nnz, metadata length and records-region length. Two flags indicate
an optional column permutation and per-column mode values.

The metadata sections follow in this order:

1. Three 256-entry `u16` rANS frequency tables: gaps, ranks and dictionaries.
   Each used model sums to `2^11`; an unused model sums to zero.
2. An optional **coded-to-source** column permutation, then optional per-column
   mode values in coded order.
3. Per-record offsets (`u64`), lengths (`u32`) and kind flags (`u8`).
4. A cumulative `u64` nnz table with `n_cells + 1` entries.

Raw column IDs and permutation entries use two bytes through 65536 columns and
four bytes above that. Offset and cumulative nnz tables begin at 8-byte-aligned
positions; their section locations follow from the declared chunk shape.

`meta_len` ends after the nnz table. With `page_log2 = 0`, records begin there and
are contiguous in row order. Packed chunks insert padding until `records_start`
is page-aligned; records may then appear in a different physical order with gaps.
Use the per-row extent table in both cases. Packed page sizes are powers of two
from 512 to 65536 bytes.

## Record: one row

A nonempty record starts with flags and an nnz varint, followed by its index and
value sections. A sparse record stores entries in coded column order; an explicit
stored zero is still an entry. A dense record omits the index section and stores
one value per column, including zeros.

<DocDiagram name="record-layout" />

| Count | Sparse record | Dense record |
|---|---|---|
| `nnz` | Stored entry count, including explicit zeros | Nonzero value count |
| Stored elements | `nnz` | `n_dims` |

Empty rows can have a zero-length record. This is why `max_nnz` alone is not a
bound for every dense decode buffer: use `max_elems` or the codec sizing API.

::: details Record flag encoding and value coding

| Flag bits | Encoding |
|---|---|
| 0–1 | Index mode: raw (0), gap-coded (1), dense (2) |
| 2–3 | Value mode: raw (0), dictionary-coded (1), column-mode-coded (2) |
| 4–7 | Reserved, zero |

Dictionary coding stores distinct value bit patterns and per-element ranks.
Column-mode coding uses rank zero for that column's mode and positive ranks for
exceptions in a record-local dictionary. Leading escape or pool bytes can be
carried in rANS states; use the decoder rather than treating the stored sections
as independent byte streams.

:::

## What parsing guarantees

`parse_superblock` checks matrix-level fields. `parse_directory` then ties the
directory to that header and checks row coverage and chunk extents. Neither reads
chunk metadata or payload bytes.

`parse_meta` validates a chunk's metadata structure and returns borrowed views.
Some per-row table checks happen only when queried; it does not validate every
inner record. Decoding retains the [trusted-data contract](data-contracts.md).
The [format API reference](reference-format.md) gives input lengths, lifetimes
and error behavior for each parser.
