# Data contracts

Use this reference before connecting external arrays, sampling callbacks, or
shared output buffers. Shapes, units, id spaces, and ownership are part of the
API contract; a successful compilation alone does not establish them.

## Which layer accepts what?

| Layer | Input | Output | Allocation and ownership |
|---|---|---|---|
| `scans` | Borrowed matrix/record bytes, metadata and caller workspace | Encoded bytes, decoded arrays, or borrowed `Parts` | Caller owns outputs and scratch; borrowed views retain dependencies |
| `store` writer | CSR/dense row batches or packed chunks | A finalized `.cpk` stream | Writer owns staging and encoding buffers |
| `store` prepared reader | A file region, ordered row selection, optional column map | Caller-owned typed dense/CSR arrays | Selection owns prepared metadata; workspace owns scratch |
| `loader` | Immutable registered files, schemas, deterministic accesses | Leased native output bytes per schema and batch | Producer owns ring writes; consumer lease delays slot reuse |
| `opt` | Loader graph, accesses, frame budget and costs | Residency plan or online compiler state | Plan binds to input, schema and sequence identity |

## Matrix inputs

Let `R` be rows, `C` columns, `E` stored entries, and `W` bytes per stored value.

| Type | Required representation |
|---|---|
| `DenseRef` | Row-major `data`; a whole number of rows of `C × W` bytes |
| `CsrRef` | `R + 1` `u64` row offsets; index and value arrays covering every row interval |
| `CsrView` | The same row model with borrowed integer offsets through `Offsets` |
| `CellRef` | One sparse row, strictly ascending unique column ids; `indices.len() × W` value bytes |
| `RowRef` | One sparse or dense row, interpreted with its declared dimensions and value width |

Values are little-endian bytes, with `W ∈ {1, 2, 4, 8}`. Chunk encoders require
`1 <= C <= i32::MAX`. Within every CSR row, indices must be nonnegative,
strictly increasing, distinct, and below `C`. Row offsets must describe readable
index/value intervals and must not decrease. A complete standalone CSR normally
starts at offset zero; views can describe intervals into larger backing arrays.

`Indices` accepts `u16`, `u32`, `i32`, `i64`, or `u64` storage.
Signed negatives and ids outside the supported range are invalid.
`Offsets` accepts `u32`, `u64`, `i32`, or `i64`; signed offsets must be nonnegative.
Wider host storage does not expand the codec's column-id space.

Dense and CSR source layouts do not force each encoded record's layout. With
automatic layout selection, individual records can be sparse or dense.
Explicit sparse zeros are stored entries. In loader semantics, an omitted
value of a present column is zero, rather than a missing dimension.

## Numeric types and conversion

File dtypes are `U8/U16/U32/U64/I8/I16/I32/I64/F32/F64`.
Codec compression preserves stored value bit patterns. It does not normalize,
quantize, replace NaNs, or attach biological meanings to values.

Typed storage reads and `scans::CastInto` use Rust `as` semantics for their
supported numeric element types. The loader admits a narrower set of pairs:

| Stored → output | Loader support |
|---|---|
| Same dtype | Yes |
| Unsigned → wider unsigned or wider signed | Yes |
| Signed → wider signed | Yes |
| Signed → same-width or wider unsigned | Yes; negative values are not checked before casting |
| Any integer → `f32` or `f64` | Yes; large integers can lose precision |
| `f32` → `f64` | Yes |
| Other narrowing, float → integer, `f64` → `f32` | Rejected by loader construction |

Use `loader::allowed(source, output)` for the actual pair check.
The low-level cast API and loader admission rules are distinct.
Native loader output is typed host representation; file/codec byte streams are
little-endian representation.

## Shapes and capacity

| Operation | Output extent |
|---|---|
| Storage dense read | Exactly `ReadSpec.rows × ReadSpec.columns` elements |
| Storage CSR read | Exactly `rows + 1` offsets; index/value capacity at least `ReadSpec.nnz` |
| Codec chunk CSR decode | `selected_rows + 1` offsets and sufficient entry/value capacity |
| Codec dense decode | Byte count from `scans::dense_len` |
| Dense loader schema | `out_ids.len() × dtype.width()` bytes per cell |
| Process schema | Exactly declared `cell_bytes` per cell |
| Leased schema region | `batch.rows() × cell_bytes(schema)` bytes |

`ReadSpec.nnz` is source-based capacity. Column filtering can reduce the returned
actual nnz. Ignore unused index/value capacity. A tail batch can have fewer than
`slot_cells` positions; use `rows()`, not the configured slot size.

Encoding and decoding sizing APIs distinguish byte capacity from `u64` workspace
words. Do not multiply or divide units implicitly.

## Column and row identities

| Name | Meaning | Important limit |
|---|---|---|
| Source column | Position in one input matrix | Codec dimension range |
| Coded column | Chunk-local compression order | Requires the matching chunk permutation |
| Standard column | Application-owned vocabulary id | Loader builder's `vocab` |
| Output column | Position in one schema's panel | `out_ids.len()` |
| `CellId` | Global source row, assigned in registration order | `u32`; constructors check representable counts |
| Position | Logical output row in the sequence | Can repeat the same cell |
| Access | One schema delivery at a position | `k / S` is position; `k % S` is schema |

Storage `RowSelection::List` uses `u64` file rows. Codec `Rows::List` uses
`u32` chunk-local rows. Loader `Accesses` produces global `CellId` values.
These selectors are not interchangeable.

A storage gene map is `source → output`. A data-source map is
`source → standard`. A schema lists standard ids in output order.
Maps must cover their required input domain, and kept targets must be distinct.
`UNMAPPED = u32::MAX` means dropped, not a valid output or standard id.

## Missing dimensions

A source matrix either has a standard column or does not. A present column whose
sparse record omits a value produces zero. A dimension absent from that matrix
produces the schema default. Defaults never replace present zeros.
CSR process rows cannot distinguish these two omitted cases; dense process rows
can distinguish them when the default is a sentinel no valid value takes.

## File and page identity

A stream is valid after `CpkWriter::finish` succeeds. A writer may truncate an
existing path. Its success is not an atomic publication or fsync guarantee.
Keep files immutable while prepared selections, sources, loaders, and plans use
them. Loader construction/preparation verifies recorded file identity.

Packed page sizes are powers of two in `[512, 65536]` bytes.
All matrices of one data source use the builder's page size. Embedded stream
offsets and lengths are supplied by the application. `O_DIRECT` additionally
requires filesystem and physical-offset alignment support.

## Borrowing and concurrency

| Resource | Lifetime rule |
|---|---|
| Borrowed encoder input | Keep immutable through the call or the entire `PreparedEncoder` borrow |
| `Meta` | Keep its backing metadata bytes alive |
| Decode/encode tables | Use only with their matching metadata |
| `Parts` | Borrows workspace, record, and possibly metadata; consume before reusing them |
| `ReadWorkspace` | Exclusive to one read at a time; can be reused afterward |
| Loader run | Borrows loader and sequence; one run holds a loader at a time |
| Batch region | Valid only while its lease is held and backing memory remains alive |
| Native callback | Code/context must outlive every call, be thread-safe, and retain no input pointers |

External ring constructors are unsafe: backing storage must be stable, correctly
aligned, sufficiently large, and governed by the slot protocol. An asynchronous
device copy is still a reader; release its source lease only after completion.

## Failure and validation boundary

Entry points check declared shapes, capacities, option ranges, and metadata
compatibility where documented. Inner record bytes are trusted.
Malformed trusted data can cause undefined behavior in optimized paths; the
decoder is not an untrusted-data validator.

A read/decode/process failure can leave partial output. Discard that operation's
output. Loader failure cancels the run and wakes participants. An aborted ring
cannot start another run. Report the original non-`Aborted` error when available.

See [errors](reference-errors.md) for exact variants and recovery scope.
