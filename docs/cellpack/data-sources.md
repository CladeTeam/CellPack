# Data sources and column mapping

Before constructing a loader, give every matrix a common integer vocabulary.
The resulting `DataSource` describes where rows live and what their columns mean;
an `OutputSchema` chooses the columns and dtype delivered to the consumer.
If you only need one file's selected rows, the simpler
[storage selection](storage.md) uses a direct source-to-output map.

## Register matrices

`DataSourceBuilder::new(vocab, page_size)` declares the shared vocabulary size
and I/O page size. `add_cpk(path, to_std)` registers a complete matrix stream;
`add_cpk_region(path, offset, length, to_std)` registers an embedded stream.
`build()` returns an immutable `DataSource`, normally shared with `Arc`.

`to_std` must cover every source column. Entries are distinct standard ids in
`[0, vocab)`, or `UNMAPPED` for dropped columns. All registered streams must use
the builder's page size: a power of two from 512 through 65536 bytes.

Registration reads stream directories and chunk metadata, not every row's
values. It is a setup step: register all inputs, call `build`, then reuse the
immutable source when creating loaders and plans.

## The application defines alignment

Suppose matrices have these column names:

<DocDiagram name="rust-alignment" />

A schema with `out_ids = [1, 0]` requests `[CD3D, GATA1]`. CellPack composes integer
maps. It does not read names, compute a union, or infer alignment.

Keep these three spaces separate:

| Space | Example | Who defines it? |
|---|---|---|
| Source column | Matrix B column 0 is `MS4A1` | Each file's stored column order |
| Standard id | `MS4A1` has id 2 | Your shared vocabulary and each matrix's `to_std` |
| Output position | Position 0 requests standard id 1 (`CD3D`) | `OutputSchema::out_ids` order |

For the diagram, register A with `to_std = [0, 1, 2]` and B with
`to_std = [2, 0]`. Then request `out_ids = [1, 0]`. B's first stored column
is dropped from that panel; its second goes to output position 1. Output
position 0 uses the schema default because B has no `CD3D` column.

`CD3D` is missing from matrix B. In matrix A, it is present even if a particular
CSR row omits it, which represents a zero value.

| Situation | Output |
|---|---|
| Source matrix has the column and the row stores a value | Value converted to the schema dtype |
| Source matrix has the column but the sparse row omits it | Zero |
| Source matrix has no column for that standard id | Schema default |
| Source column is outside the output panel | Dropped |

Use `with_default(f64::NAN)` in floating-point dense outputs to distinguish
missing dimensions from present zeros. `with_default_bytes` sets an exact
little-endian default, including large integers.

::: tip A zero value is not a missing column
The map declares whether a dimension exists for the entire matrix. An omitted
CSR entry is a zero in an existing dimension. Choose a nonzero default such as
`NaN` only when downstream code needs to distinguish the two situations.
:::

## Global cell ids

Matrices receive consecutive `CellId` ranges in registration order.
`DataSource::matrix_cells(index)` returns the matrix's first global id and row
count. Sampling must produce ids in this global space.

`CellId` is `u32`. The `.cpk` format's `u64` row count does not imply that one loader
data source accepts arbitrary `u64` cell ids.

For two registered matrices with 4 and 3 rows, their ranges are `[0, 4)` and
`[4, 7)`. Matrix B local row 1 is global cell 5, regardless of the requested
output panel. Obtain the actual base through `matrix_cells`; do not use a
matrix index as a cell id. [Access sequences](accesses.md) explains how global
cell ids become requested positions.

## Input identity and direct I/O

Registration records each file's device, inode, size, and modification time.
`Loader::new` and `Loader::prepare` check that identity to reject changed inputs.
Keep input files immutable during a run as well.

`direct_io(true)` opens record data with `O_DIRECT`; metadata reads remain
buffered. Enable it only when the filesystem and embedded-stream alignment
support the required reads. Guide examples use `direct_io(false)` for portable
small-file execution.

### Check registration failures first

| Failure | Check before changing the loader |
|---|---|
| Map rejected | Exactly one entry per source column; kept ids distinct and below `vocab`; `UNMAPPED` may repeat |
| Page-size mismatch | Builder page size matches the file superblock and every chunk |
| Changed input identity | Rebuild the source and any plans after replacing or modifying input files |
| Direct-I/O open or read failure | Filesystem support and stream alignment; use buffered I/O for a portable baseline |

See [loading](loading.md) for a complete example that requests an absent column
with a `NaN` default, alongside present columns and repeated cells.
API: [DataSourceBuilder](/generated-loader-struct-DataSourceBuilder),
[DataSource](/generated-loader-struct-DataSource).
