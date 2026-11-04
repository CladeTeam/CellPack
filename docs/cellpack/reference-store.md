# Matrix storage: store

`cellpack::store` owns the file-level workflow: streaming matrix writes,
prepared reads, physical read planning, and decoding into caller-owned outputs.
It does not own sampling, a shared gene vocabulary, or an asynchronous training
queue. For those tasks, use [loader](reference-loader.md).

<DocChoices title="Start at the level you need" :items="[
  { label: 'Use the file workflow', when: 'Write rows or read a matrix selection.', detail: 'CpkWriter, CpkFile and ReadSelection cover the normal path. The guide includes a complete asserted round trip.', href: '/rust/docs/storage.html' },
  { label: 'Integrate your own record I/O', when: 'You already manage chunk bytes and buffers.', detail: 'RecordReadPlan and PanelRows expose physical reads and row mapping; keep metadata, tables and records matched.', href: '#physical-reads-and-panel-helpers' }
]" />

## Input and output model

| Operation family | Takes | Produces |
|---|---|---|
| Writer | Declared matrix dtype/layout and row batches | One finalized `.cpk` matrix stream |
| File inspection | Path or embedded region | Dimensions, dtype, superblock, directory and chunk bytes |
| Prepared selection | File rows, optional source-to-output map, read policy | `ReadSelection` with shape, capacity and I/O plan |
| Borrowed-output read | Selection, typed buffers, workspace, worker count | Dense values or canonical CSR plus actual nnz |
| Panel-row helper | Chunk metadata/tables, panel and mapper | Mapped rows in caller buffers |

See the shared [matrix, conversion, and lifetime contracts](contracts.md).

## CpkWriter

| Method | Inputs | Return and postcondition |
|---|---|---|
| `create` | `path, n_cols, dtype, source_layout, CpkOptions` | `Result<CpkWriter>`; creates/truncates the path, which is not yet a valid stream |
| `append` | Borrowed `MatrixRef`-convertible rows | `Result<()>`; appends in order, buffering a partial chunk |
| `set_threads` | Desired thread count | `()`; counts below one become one; complete chunks can encode concurrently |
| `end_chunk` | Writer's staged rows | `Result<()>`; flushes them so the next append starts a new chunk |
| `append_packed` | One encoded and page-packed chunk | `Result<()>`; appends without re-encoding after compatibility checks |
| `finish` | Consumes writer | `Result<()>`; writes tail, directory and final superblock |

`append` requires matching column count and value width. If the incoming form
differs from the declared source layout, the writer converts CSR/dense form.
CSR columns remain canonical. `append_packed` requires no staged rows and a chunk
with matching dimensions, value width, page size and total byte length.

`source_layout` chooses the form staged by the writer. It does not promise that
every encoded record will be dense or sparse; chunk encoding options decide that.
The supplied `dtype` interprets input bytes: matching byte width is not numeric
conversion. For example, writing four-byte integer values with `Dtype::F32`
declares them to be float bit patterns. Convert values before appending if needed.
Borrowed append inputs may be reused after the call returns.

`CpkOptions::new(encode)` defaults to 4096 rows per chunk and 4096-byte pages.
`chunk_rows` is a target; early chunk endings and appended packed chunks may
produce different chunk lengths. Codec choices live in [EncodeOptions](reference-scans.md).

## CpkFile

| Method | Inputs | Return |
|---|---|---|
| `open` | Path | Opened matrix handle |
| `open_region` | Path, byte offset, optional bounded byte length | Handle to an embedded stream |
| `n_rows / n_cols / dtype` | Handle | Matrix shape and stored type |
| `superblock / chunks` | Handle | Borrowed stream metadata and directory entries |
| `byte_offset` | Handle | Stream's offset in its outer file |
| `identity` | Open descriptor | Device and inode, independent of later path replacement |
| `read_meta / read_chunk` | Chunk index | Owned metadata bytes or complete chunk bytes |
| `read_chunk_into` | Chunk index, metadata-only flag, exact-size byte buffer | Writes chunk bytes into the supplied buffer |
| `read_dense / read_csr` | Half-open `Range<u64>` | Allocated stored-dtype bytes or `Csr` |
| `prepare_read` | Half-open row range | Prepared selection of every source column |
| `prepare_selection` | `RowSelection`, optional map, output width, `ReadOptions` | Prepared ordered selection |

`open_region` interprets format offsets relative to the stream.
`read_dense` returns `Vec<u8>`, not `Vec<f32>`.
Allocated `Csr` stores `u64` offsets, `u32` indices and value bytes. Obtain its
value dtype and source column count from the file; `Csr` has no dtype/shape fields.
Prepared typed reads are the route for conversion or buffer reuse.

Opening reads the superblock and directory; chunk metadata and record payloads
are read later as requested. A successfully opened handle is not a full-file
record verification. `identity()` describes the opened descriptor, so replacing
the path does not retarget an existing handle.

## ReadSelection and ReadWorkspace

| Method | Inputs | Output and mutation |
|---|---|---|
| `spec` | Prepared selection | `ReadSpec { rows, columns, nnz, dtype }` |
| `io_summary` | Prepared physical reads | Planned payload/read bytes and request counts |
| `read_dense_into<T>` | Exact `rows × columns` buffer, exclusive workspace, workers | `Result<()>`; fills every output element |
| `read_csr_into<P,I,T>` | Exact row-offset array, index/value capacities, workspace, workers | `Result<usize>`; returns actual nnz and canonical output indices |
| `ReadWorkspace::clear` | Workspace | Releases retained scratch |

`RowSelection::Range` uses half-open file rows. `List` preserves arbitrary order
and repetitions. The optional map covers every source column, maps kept columns
to distinct output positions, and uses `UNMAPPED` for drops. Missing output
positions are zero. Without a map, output width must equal file width.

`ReadSpec::dense_len` is an element count; `dense_bytes` and `value_bytes`
use the stored dtype width, not a requested converted output type.
Allocate typed output using element counts and `size_of::<T>()`.
`nnz` is capacity, not post-filter actual nnz. Narrow index types must hold the
largest output column and every needed row offset.

Preparing a selection reads metadata. Each read invocation performs record I/O.
The selection owns its prepared state; output is owned by the caller.
Concurrent reads need separate mutable workspaces and outputs.
Failures can leave partial writes.

### Validation and failure scope

| Boundary | Checked contract | On failure |
|---|---|---|
| `create` / `append` | Encoding options, page/chunk sizes; input column count and value width | Returns an error; an unfinished file must not be published |
| `open` / `open_region` | Stream bounds, superblock and directory | No usable handle is returned |
| `prepare_selection` | Row bounds, full source map, distinct/in-range targets, metadata and addressable capacities | No usable selection is returned |
| Typed output entry | Exact dense length; CSR lengths/capacities and index widths | Output dimensions are rejected before decoding |
| Record I/O / decoding | I/O and supported record checks | Treat that invocation's output as incomplete |

Canonical CSR inputs and chunk/record consistency remain part of the shared
[trusted-data contract](contracts.md#failure-and-validation-boundary). These
interfaces are not an untrusted-file sanitizer. After a writer error, do not
assume that retrying an append resumes at a transactional checkpoint.

## Physical reads and panel helpers

`RecordReadPlan::new(meta, cells, options)` plans one chunk's records from metadata.
`ranges` returns disjoint read ranges; `summary` returns byte/request totals.
`read_into(file, base, out)` resizes a byte vector and reads each range relative
to the records-region `base`. `record(out, position)` returns a borrowed record
slice for the corresponding requested position, including repeated/empty records.

`ReadOptions` defaults to `Auto`, zero merge gap, and 4096 equivalent bytes per
request. `WholeChunk` means the complete records region, excluding metadata.
`ReadSummary` describes application reads, not physical device traffic.

`compose_panel` combines a chunk permutation and an optional source map.
`PanelRows<S,T>` binds metadata, tables, a panel and a mapper; `bind` prepares
`RowScratch<T>`; `dense` and `csr` decode/map individual records. These helpers
share the codec's trusted-record contract. Returned CSR column slices borrow
scratch and must be consumed before its reuse.

## Complete interface index

Each entry opens a searchable page with its declaration, parameter/return types,
fields, methods, and source documentation.

<!--@include: ./_catalog/store.md-->
