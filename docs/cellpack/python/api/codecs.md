# Codec components API

`pycellpack.cellpack.scans` encodes chunks and decodes records over NumPy buffers.
Use it when your application owns the container or record-read strategy. Use
[Store](storage.md) for ordinary matrix files, or follow the
[codec guide](/python/docs/codecs.md) for a complete round trip.

```python
from pycellpack.cellpack import scans
```

<DocChoices title="Choose the interface you need" :items="[
  { label: 'Chunk functions', when: 'Encode a matrix selection or decode resident rows.', href: '#chunk-and-record-operations', detail: 'Allocate results automatically, or supply exact output and workspace buffers.' },
  { label: 'PreparedEncoder / Decoder', when: 'Repeat work with one model or panel.', href: '#reusable-objects-and-concurrency', detail: 'Snapshot encoder input; reuse decoder mapping, tables and scratch.' },
  { label: 'Parts and mapping', when: 'Inspect components or transform one decoded row.', href: '#column-spaces-and-parts', detail: 'Keep source, coded and output positions distinct.' },
]" />

## Numeric interpretation and metadata

Raw chunks store value **width**, not signedness or floating-point type.
`Meta(buffer, source_dtype=...)` declares how those bits are read; its dtype must
be a supported native numeric dtype of that width. Omitting it on `Meta` selects
an unsigned integer dtype, which is usually wrong for float or signed input.

`decode_chunk(raw_bytes, source_dtype=...)` requires that argument explicitly.
`decode_chunk(meta)` reuses the parsed interpretation; supplying a conflicting
`source_dtype` raises ValueError. `dtype=` instead requests a numeric **output
conversion**. It does not repair an incorrectly interpreted source.
`File.chunk_meta` already supplies the stream's stored dtype.

| Object / method | Meaning and ownership |
| --- | --- |
| `Meta` | Validated private metadata snapshot; `buffer` retains the supplied bytes |
| `shape`, `dtype`, `value_width` | Chunk rows/columns and declared stored interpretation |
| `records()` | View of the complete records region in `buffer`; raises if it is absent |
| `record_range(cell)` | A slice relative to the records region, not the chunk start |
| `record(cell, records=...)` | View of that record; supplied `records` is already the records region |
| `DecodeTables(meta)` / `EncodeTables(meta)` | Lookup tables bound to that metadata model |
| `column_modes`, table permutation arrays | Read-only NumPy snapshots; no permutation array means identity order |

Metadata-only bytes are sufficient for model inspection and table construction.
Pass `records=` to chunk decoders when records are held separately. Derive
metadata, tables and records from the same encoded chunk. After `pack_chunk`,
parse its returned buffer again: packing changes offsets and byte extents.

## Chunk and record operations

<DocDiagram name="python-codec" />

| Operation | Input | Result |
| --- | --- | --- |
| `chunk_sizes` | Matrix, rows and encoding options | `ChunkSizes(output_bytes, workspace_words)` |
| `encode_chunk` | Same input; optional `out` and `workspace` | Written uint8 prefix, which can share `out` |
| `decode_chunk` | Raw bytes plus source dtype, or Meta; `format="dense"` / `"csr"` | NumPy ndarray / package CSR |
| `decode_dense` / `decode_csr` | Meta and optional rows, records, tables, buffers or mapping | Selected dense rows / CSR, in requested row order |
| `encode_record` | One row and matching Meta/EncodeTables/options | Encoded uint8 prefix under that chunk model |
| `decode_record` | Meta plus record bytes, PagedRecord or `cell=` | SparseRow with trimmed indices and values |
| `decode_record_dense` | Same record inputs | One dense row of shape `(meta.n_dims,)` |
| `decode_parts` | Same record inputs; coded space by default | Owned Parts |
| `packed_len` / `pack_chunk` | Encoded chunk; page size default 4096 bytes | Packed byte count / packed uint8 prefix |
| `record_nnz` | Record bytes or PagedRecord | Declared header count, not record integrity validation |

`rows` addresses **chunk-local rows**. None selects all; integers, slices, masks
and integer lists preserve the selected order and repetitions. Record calls
accept either `record` or `cell`, never both. `cell` indexes the metadata's own
record buffer. It is not a loader global cell ID.

`encode_record` requires one row with the model's dimensions and dtype, matching
encoding tables and compatible encoding options. It does not rebuild a model
for an unrelated row population. Chunk encoders accept NumPy, SciPy sparse,
CSR and SparseRow inputs; package CSR must already have ascending distinct
column indices per row. SciPy inputs are canonicalized before encoding.

### Encoding options

`EncodeOptions()` defaults to renumbering, thresholds of 0.6, `max_lanes=32`,
`lane_min=32`, `column_modes=0.5`, and automatic layout/value coding.
Thresholds and column-mode share are in `[0, 1]`; None disables column modes.
Lane caps are `1, 2, 4, 8, 16, 32`; lane minimum is a power of two through 65536.
Forced dense layout can remove explicit sparse-zero entries, even when the dense
numeric matrix is unchanged. Keep automatic/sparse layout when CSR structure
itself must round-trip.

## Buffers, capacities, and fill

| Buffer / sizing query | Unit and required extent |
| --- | --- |
| Encoding `out`, `record_bound`, `packed_len` | uint8 storage / bytes; only the returned prefix is meaningful |
| Encoding or decoding `workspace` | Writable contiguous uint64 **words**, not bytes |
| `meta.decode_workspace_words`, `decode_workspace_bound` | Scratch word count for one active decode |
| `dense_bytes(meta, rows)` | Dense selection bytes in the **stored dtype**, before panel/dtype changes |
| Dense chunk output | Exact `(selected_rows, columns)` shape and requested dtype |
| Dense record output | Exact one-dimensional row width and dtype |
| CSR output | `rows + 1` offsets and capacity for `selection_nnz(meta, rows)` indices/values |

Output arrays must be writable, aligned, native-endian and C-contiguous. CSR
accepts the package CSR or `(indptr, indices, data)`; allocating calls default
to int64 indices. Mapped CSR can return fewer entries than its source-based
capacity: use the returned trimmed arrays and final row offset.

`Fill.ZERO` / `"zero"` initializes absent dense values. With a supplied output,
`Fill.ASSUME_ZEROED` / `"assume_zeroed"` promises it is **already all zero**.
It does not preserve arbitrary previous content correctly; clear a reused row
before relying on that promise. Allocating calls provide zeroed storage for it.

Chunk decoders may split rows over `workers` threads. On the direct decode path,
caller workspace serves the first thread; other threads allocate their own.
Supplying `gene_map`, `n_genes` or `dtype` selects the Decoder panel path, whose
tables and scratch are managed internally rather than reusing those standalone
arguments. Construct a Decoder explicitly to reuse that state across calls.

Keep input/output/scratch regions disjoint as required by the call. Do not
resize or mutate borrowed inputs while native work is active, and do not share
one output or workspace between concurrent calls. Discard potentially partial
output after a failed operation.

## Column spaces and Parts

| Space | Meaning | Mapping rule |
| --- | --- | --- |
| Source | Original input column position | Chunk decode `gene_map` uses source → output |
| Coded | This chunk's compression permutation | Compose the source map with `compose_mapping(tables, map)` |
| Output | Desired panel position | `n_genes` is panel width; unfilled positions are zero |

Most decoders default to `IdSpace.SOURCE`; `decode_parts` defaults to CODED.
For chunk decoding, `gene_map`, `n_genes` or `dtype` require source space.
For Parts mapping, the map is indexed by **that Parts object's space**.
`-1` or `UNMAPPED` drops a source dimension; kept targets must be distinct and
inside the output panel. A source-to-output map is not a loader vocabulary map.

| Operation | Result and ordering |
| --- | --- |
| `compose_mapping` | Coded → output GeneMap composed from DecodeTables and a source → output map |
| `map_dense` | Dense panel row, with optional dtype conversion and fill policy |
| `map_parts` | SparseRow, dropping unmapped genes; columns follow input element order and need not be sorted |
| `map_parts_at` | One entry per selected element position, preserving order/repeats; dropped dimensions keep `UNMAPPED` |
| `cast_values` / `Dict.cast_into` | Converted values / dictionary; supplied `out` must agree with explicit dtype |

Filter `UNMAPPED` entries from `map_parts_at` before treating the result as a
normal sparse row. Element positions address `parts.elements`, not gene IDs.
Dense-layout Parts can have more elements than nonzeros. `map_parts` omits
dense zero values; mapping selected positions retains the selected entries.
Optional converted `dictionary` and `modes` arrays must match this row's
components and target dtype; a mode vector addresses coded dimensions.

Python Parts owns private metadata, copied record/page bytes and scratch.
Its `ids`, values, dictionary and rank properties expose read-only snapshots;
mapping reads the stable private representation. `expand_values(out=...)`
materializes all elements. These are owned Python objects, unlike Rust's
borrowed Parts, and can outlive the original caller buffers.

## Reusable objects and concurrency

`PreparedEncoder(matrix, rows=None, options=...)` snapshots input arrays and
selection, then builds the model once. `encode(out=..., workspace=...)` can be
called concurrently with independent output/scratch. Its `meta` describes the
model **before record placement**: use it for encoding tables or independent
record encoding, not cell-indexed decoding. Parse the encoded result for that.

`Decoder(meta, gene_map=..., n_genes=..., dtype=...)` composes the mapping and
retains reusable native tables/scratch. It serves one call at a time, even when
that call uses several workers; concurrent use raises RuntimeError. Give
independent consumers separate Decoders. Its `decode_record` returns a **dense
panel row**, whereas the standalone function returns SparseRow.

`PagedRecord(pages, page_size, start, length)` describes separately read pages
in file order and the record's offset within the first page. Record/Parts
decoders accept it. Pages must cover the declared record; all preceding pages
are full-size. Cross-page decoding may stage bytes in scratch, so this does
not promise zero-copy decoding.

`kernel_set()` reports the process binding; `supported_kernel_sets()` lists CPU
capabilities. A supported ISA need not be the bound one. The Python extension
binds the runtime when imported; set `CELLPACK_KERNELS` before importing
PyCellPack, preferably in the environment used to launch the interpreter.

## Errors and trusted records

Invalid dtype/buffer declarations, incompatible tables, out-of-range selectors
or insufficient capacities raise TypeError, IndexError or ValueError as
appropriate. Decoder concurrent use raises RuntimeError. Metadata checks and a
successful `record_nnz` do not validate arbitrary compressed record internals.
Only decode records produced for their matching chunk model; violating that
trusted-record contract can cause undefined behavior in optimized native paths.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/codecs-members.md-->

:::
