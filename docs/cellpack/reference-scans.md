# Encoding, decoding, and mapping: scans

`cellpack::scans` operates on application-owned bytes and buffers. Use this page
for capacities, units, ordering, and borrowing rules. For a complete round trip
and an explanation of how the pieces fit, read [Codec components](codecs.md).
Use [store](reference-store.md) when you need complete matrix files.

## Select an entry point

<DocChoices
  title="Start at the granularity you own"
  :items="[
    { label: 'Matrix selection → chunk', when: 'Build a model and encode selected matrix rows.', detail: 'chunk_sizes → encode_chunk; PreparedEncoder reuses a fixed borrowed input and model.', href: '#encoding-buffers-and-results' },
    { label: 'Chunk or record → arrays', when: 'Materialize CSR or dense output.', detail: 'decode_csr / decode_dense for a resident chunk; Decoder for repeated records.', href: '#decoding-buffers-and-results' },
    { label: 'Record → panel', when: 'Keep values encoded until mapping or casting.', detail: 'decode_parts → map_parts / map_dense; consume the borrowed view before scratch reuse.', href: '#parts-and-mapping' }
  ]"
/>

All codec `Rows` selections use **chunk-local row IDs**. `Rows::All` and half-open
`Rows::Range` preserve natural order; `Rows::List(&[u32])` preserves the listed
order and repetitions. Inputs and output types follow [data contracts](contracts.md).

## Encoding buffers and results

| Operation | Sizing / preparation | Result and ownership |
|---|---|---|
| `encode_chunk` | `chunk_sizes(src, rows, options)` | Encoded byte count; chunk occupies `out[..count]` |
| `PreparedEncoder::encode_into` | `PreparedEncoder::new(source, rows, options)`, then `sizes()` | Encoded byte count; repeated calls write the same bytes for the bound input |
| `encode_record` | `record_bound(n_genes, value_width, nnz, dense)` and `encode_workspace_words(n_genes, max_elems)` | Encoded byte count for one row under the supplied chunk model |
| `pack_chunk` | `packed_len(chunk, page_size)` | Packed byte count into a separate output buffer |

`ChunkSizes::output_bytes` is a byte capacity; `workspace_words` counts `u64`
elements. Pass matching inputs, selection and options to sizing and encoding.
The individual `chunk_bound` and `chunk_workspace_words` queries are available
when only one capacity is needed. Do not treat the allocation's unused suffix as
encoded data.

`PreparedEncoder` borrows immutable matrix input and its row selection.
Construction builds the model and owns its metadata/tables; keep the borrowed
inputs alive and unchanged. `metadata()` and `metadata_bytes()` describe the
model **before record placement**. They are useful for independent
`encode_record` calls, but do not constitute a completed chunk with record extents.
`encode_into` completes those extents in its output.

Independent record encoding requires a row with the model's dimensions and value
width, matching encoding tables, and the intended encoding options. An empty
record can have length zero. Preserve the model and record identity together in
your container.

`pack_chunk` accepts page sizes that are powers of two in `[512, 65536]` bytes.
It can change physical record order and metadata extents, while retaining logical
row order. Each nonempty record reads `ceil(record_length / page_size)` pages
when the packed chunk is stored at a page-aligned position. Parse the resulting
chunk again; old offsets do not describe the new layout.

### EncodeOptions

| Field | Meaning / valid domain |
|---|---|
| `renumber` | Build a frequency-based chunk column permutation |
| `thresholds` | Maximum coded/raw size ratios for index and value sections, each in `[0, 1]`; zero retains raw sections |
| `max_lanes` | rANS lane cap: `1, 2, 4, 8, 16, 32` |
| `lane_min` | Minimum symbols per lane; power of two in `[1, 65536]` |
| `column_modes` | Optional candidate share in `[0, 1]`; `None` disables column modes |
| `layout` | `Auto`, forced sparse, or forced dense layout |
| `values` | Automatic, raw, dictionary, or column-mode value coding |

Forced value coding ignores the value-size threshold. Forced mode coding requires
`column_modes: Some(...)`; enabling a column-mode candidate in automatic mode
does not guarantee that every row uses it. `CpkWriter` normally suppresses column
modes for nonfloating types unless modes are forced. `DEFAULT_*` constants expose
suggested values; the generated field and enum entries below give exact variants.

::: warning Forced dense layout and explicit sparse zeros
Automatic layout selection preserves explicit sparse zeros by retaining a sparse
record when needed. Forcing dense layout drops those explicit entries. Distinguish
numeric equality of a dense matrix from exact recovery of a CSR representation.
:::

## Decoding buffers and results

For a resident chunk, parse its `Meta`, build matching `DecodeTables`, and pass
`meta.records_of(chunk)` as the records region. This region differs from the full
chunk, which also contains metadata and possible page-alignment padding.

Let `R` be selected rows, `N` selected nnz, `C = meta.n_dims()`, and
`W = meta.value_width()` bytes.

| Call | Caller buffers | Result |
|---|---|---|
| `decode_csr` | `R + 1` row offsets, capacity for `N` indices and `N × W` value bytes | Written nnz |
| `decode_dense` | Exactly `R × C × W` output bytes | `Result<()>` |
| `decode_record` | Capacity for the record's nnz indices and `nnz × W` value bytes | Written nnz |
| `decode_record_dense` | Exactly `C × W` output bytes | `Result<()>` |
| `decode_parts` | Borrowed record and mutable workspace | `Parts` borrowing record, metadata and workspace dependencies |

All calls need `decode_workspace_words(meta)` `u64` words. Use
`decode_workspace_bound(n_genes, value_width)` when allocating for a maximum shape
before metadata is available. This is a workspace bound, not a compressed-record
or output-array bound.

`selection_nnz(meta, rows)` returns `u64` nnz for sparse capacity sizing;
`Meta::cell_nnz` sizes one record. `dense_len(meta, rows)` returns **bytes**.
Output indices must represent the chunk's entire dimension range, even when the
selected rows happen to use only small IDs. CSR offset types must hold the
resulting cumulative nnz. See the `IndexOut` and `IndptrOut` implementations below.

### Column order and zero filling

`IdSpace::Source` restores ascending source columns with their values.
`IdSpace::Coded` produces ascending chunk-coded IDs, avoiding source-order
restoration; the two spaces coincide when renumbering is disabled. This ordering
promise applies to codec sparse decode, not to subsequent panel mapping.

| Fill policy | Required initial state | Behavior |
|---|---|---|
| `Fill::Zero` | Any initialized output buffer | Clear each dense row before writing values |
| `Fill::AssumeZeroed` | Caller guarantees the destination is already zero | Skip clearing; omitted sparse values rely on those zeros |

`AssumeZeroed` is not a request to merge into arbitrary existing data. Clear a
reused sparse-row destination before decoding unless using `Zero`.

### Repeated and paged records

`Decoder::new(&meta, &tables)` checks the binding once and retains the chunk's
kernel dispatch and workspace layout. Its `sparse`, `dense`, `parts`, and
`parts_pages` methods reuse that binding. It does not own an output buffer or
workspace: each active operation still needs exclusive scratch.

`PagedRecord { pages, page_size, start, len }` supplies pages in file order.
`start` is a byte offset inside the first page and `len` is the exact compressed
record length. All but the last page cover `page_size`; the last may be shorter
if the record ends inside it. Use the `_pages` record functions for the same
output contract without assembling a contiguous application buffer. A single-page
record is borrowed; a cross-page record is copied into the supplied workspace's
staging region.

## Parts and mapping

`Parts` exposes `ids()`, `values()`, `nnz()`, `elements()`, and `layout()`.
`Values` distinguishes direct bytes, dictionary/rank coding, and column-mode
coding. A dense record has one element per dimension, so `elements()` can exceed
`nnz()`. Size materialized values for the element count. `expand_values` restores
stored value bytes without panel mapping or numeric conversion.

Consume or copy a `Parts` view before reusing its workspace or changing backing
record/metadata storage. The view is not an owned row. `OwnedMeta` owns metadata
when retaining the original chunk buffer is inconvenient; it does not own records.

### Mapping contracts

| Call | Selection and destination | Result |
|---|---|---|
| `map_parts` | Map every part element, dropping `UNMAPPED`; columns and values need room for every element | Kept count; use only written prefixes |
| `map_parts_at` | Select element positions in the supplied order; repeats allowed; output capacity for those positions | `Result<()>`; writes `UNMAPPED` unchanged rather than dropping it |
| `map_dense` | Scatter mapped elements into the panel; kept target columns must fit the destination | `Result<()>`; honors `Fill` |

`map[id]` is an **ID-to-output-column** lookup in the `Parts` ID space. With
`IdSpace::Coded`, compose source-to-output mapping with `Meta::source_dim` once
per chunk. `positions` addresses elements within the row; it does not contain
source columns, output columns, or row IDs. Mapped compact columns retain the
part traversal order and are sorted only if the map is monotonic.

For `Mapper<S, T>` and the free mapping functions, `S` must describe the stored
value representation and `T` the output element type. Provide dictionary entries
converted by `Dict::cast_into` and, for mode-coded values, a converted chunk-mode
table indexed by coded ID. Empty dictionary/mode slices are appropriate only
when the representation does not use them. Reuse `Mapper` for repeated calls
with a fixed type pair; the free functions resolve a route for each call.

`UNMAPPED = u32::MAX` is a drop sentinel, not a valid panel column.
SIMD-addressed maps, rows, dictionaries and mode tables cannot exceed
`i32::MAX` elements. Numeric casts follow Rust `as` semantics; the loader's
admitted conversion pairs are a separate contract. See [conversion rules](contracts.md#numeric-types-and-conversion).

## Allocation and failure boundary

Chunk model building, table construction, and page-packing planning can allocate.
Record encode/decode and mapping use supplied output and workspace memory. These
per-record properties do not imply allocation-free file reading or loader runs.

Entry points check their documented capacities, table binding, options, and
selected shape constraints. Record internals and per-cell metadata are trusted;
parsing metadata does not establish that arbitrary record bytes are safe to
decode. Keep encoder-produced records with their matching model. Invalid trusted
inputs can cause undefined behavior in optimized decoding. See the
[validation boundary](contracts.md#failure-and-validation-boundary).

## Complete interface index

The families below are useful when looking beyond the main workflow. Exact
signatures, variants, and source links follow in the generated index.

::: details Interface families
| Family | Interfaces |
|---|---|
| Borrowed inputs | `DenseRef`, `CsrRef`, `CsrView`, `MatrixRef`, `CellRef`, `RowRef` |
| Selection and output policies | `Rows`, `IdSpace`, `Fill`, `IndexOut`, `IndptrOut` |
| Tables and runtime binding | `DecodeTables`, `EncodeTables`, `decode_tables`, `encode_tables`, `*_for` variants |
| Numeric conversion | `Element`, `CastInto`, `cast_values`, `Dict::cast_into`, `ModeDict::cast_into` |
| Runtime helpers | `Isa`, `kernel_set`, `prefetch` |
:::

<!--@include: ./_catalog/scans.md-->
