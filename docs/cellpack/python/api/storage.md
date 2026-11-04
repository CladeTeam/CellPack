# Storage API

Use `Store` for lazy matrix access, `Writer` for incremental encoding, and `File`
when you need embedded regions or chunk metadata. For an executable selection
and streaming-write workflow, see [matrix access](/python/docs/storage.md).

```python
from pycellpack import CSR, File, Store, Writer, read, write, compress
from pycellpack.cellpack import GeneMap, ReadOptions, ReadSelection, ReadWorkspace
from pycellpack import io
```

<DocChoices title="Select an interface" :items="[
  { label: 'Store / read', when: 'Select positions and decode matrix values.', href: '#views-and-selections', detail: 'Store retains a lazy view; read returns an array immediately.' },
  { label: 'ReadSelection', when: 'Reuse metadata and caller-owned output.', href: '#prepared-reads-and-buffer-extents', detail: 'Choose exact dense extents or capacity-based CSR buffers.' },
  { label: 'Writer / io.write_cpk', when: 'Encode a complete matrix or row stream.', href: '#writing-and-publication', detail: 'Writer is incremental; write and compress return Stores.' }
]" />

## Entry points and results

| Interface | Input | Result |
| --- | --- | --- |
| `Store(source, *, shape=None, dtype=None)` | Path, File or Store | Immutable lazy matrix view |
| `File(path, *, offset=0, length=None)` | Path and optional bounded stream region | Stream handle; no matrix decoding |
| `File.from_bytes(buffer)` | Encoded stream bytes | File backed by an owned anonymous copy |
| `read(path, cells=None, genes=None, ...)` | Path/File and positional selectors | NumPy ndarray or CSR |
| `write(path, matrix, ...)` | NumPy, SciPy sparse, CSR or SparseRow | File-backed Store |
| `compress(matrix, ...)` | Same matrix inputs | Store over an anonymous in-memory stream |
| `Writer(path, n_cols, dtype, ...)` | Declared width/type, then `append` calls | Incremental writer; `finish()` returns None |
| `io.open_cpk(source, ...)` | Path, File, Store or encoded bytes | Store; an existing Store is returned unchanged |
| `io.read_cpk` / `io.iter_cpk` | Stream or Store and selectors | One decoded result / successive decoded row batches |
| `io.write_cpk(path, matrix, ...)` | Whole matrix, 2-D Store or iterable of batches | File-backed Store; streamed inputs need at least one batch |
| `io.cpk_info(source, ...)` | Stream source | CpkInfo: stream shape, dtype, format, sizes and chunk count |

`cpk_info` describes the underlying stream, even when passed a selected Store;
it does not describe that view's selected dimensions. `CpkInfo.bytes_per_cell`
includes metadata and returns zero for an empty matrix. Stream inspection does
not decode record values.

## Views and selections

| Operation | Contract |
| --- | --- |
| `store[cells, genes]` | Orthogonal, axis-preserving selection; both axes support slices, integers, integer arrays and boolean masks |
| `store.astype(dtype)` | Choose decode dtype without decoding; `copy` does not force materialization |
| `store.copy()` | Return the same immutable view |
| `store.map_genes(gene_map, n_genes=None)` | Map each current column to a distinct output position; unfilled output positions read as zero |
| `store.prepare(read_options=None)` | Prepare reusable metadata; repeated columns are rejected |
| `store.read(format=None, ...)` | Decode in stored format by default; `"dense"` returns ndarray, `"csr"` returns CSR |
| `store.iter_batches(batch_size=4096, ...)` | Decode successive view rows; batch size must be positive |

Order and repeated positions are preserved. Negative integer positions and
negative-step slices follow positional indexing. Boolean mask lengths must
match their current axis. Selection arrays exposed by `cells` and `genes` are
read-only; `genes=None` means every source column in order. Strings are not gene
or cell selectors: the stream stores no names.

For gene mapping, `GeneMap` owns the validated source-to-output map.
`cellpack.selection.as_gene_map` also accepts an integer vector or a dict from
current source positions to targets. The map covers every current column;
`-1` or `scans.UNMAPPED` drops a column, kept targets are distinct and below
`n_genes`. This map assigns output positions, whereas a DataSource map assigns
standard vocabulary IDs; see [gene alignment](/python/docs/alignment.md).
`normalize_rows(rows, size)` produces contiguous uint64 positional IDs.

`Store(file, shape=...)` requires its first dimension to equal stream rows and
the product of trailing dimensions to equal stream columns. Only dense streams
support more than two dimensions; those views allow row selection and dense
reads only. Stores are read-only: materialize before assigning values.

### Dtypes and materialized results

Supported value dtypes are uint8/16/32/64, int8/16/32/64, float32 and float64 in
native byte order. A view/selection dtype requests conversion during decoding;
compression alone preserves stored value bits. Conversion can lose precision.

`read_dense`, `toarray` and `np.asarray(store)` produce dense NumPy arrays.
`read_csr` produces the package's NumPy-backed CSR without requiring SciPy.
`to_memory` converts sparse output to a SciPy `csr_array`; `tocsr` always returns
that SciPy type. `CSR.to_scipy()` requests shared storage, so later mutation of
shared buffers can affect both representations.

## Prepared reads and buffer extents

`File.prepare_read(rows=None, *, gene_map=None, n_genes=None, dtype=None,
read_options=None)` reads selection metadata and returns ReadSelection.
`shape`, `dtype`, `capacity_nnz` and `io_summary` describe the prepared result.
Omitting output storage allocates arrays; supplied buffers follow these rules:

| Read | Required output | Returned storage |
| --- | --- | --- |
| `selection.read_dense(out=...)` | ndarray with exact `selection.shape` and `selection.dtype` | The supplied array itself |
| `selection.read_csr(out=...)` | CSR with matching shape, or `(indptr, indices, data)` | CSR sharing supplied buffers; indices/data trimmed to actual entries |
| CSR `indptr` | Exactly `rows + 1` elements | Final value gives actual stored-entry count |
| CSR `indices`, `data` | Capacity for `selection.capacity_nnz`; data dtype matches selection | Unused trailing capacity is not output data |

`capacity_nnz` is source-based capacity, not an exact mapped count: column
filtering can reduce the returned `CSR.nnz`. Explicit sparse zeros remain stored
entries. Allocated CSR indices/offsets use `index_dtype=np.int64`; int32, uint32
and uint64 are also supported. Caller-supplied indices additionally accept
uint16, while row offsets require one of the four allocation types. The chosen
integer widths must hold the resulting indices and offsets.

Outputs must be native-endian, aligned, writable and C-contiguous. Inputs,
outputs and scratch must not overlap where native calls reject overlapping
regions. Do not mutate or resize arrays borrowed by a native call, or read an
output concurrently with its write. A ReadWorkspace retains scratch and permits
one active read; use separate workspaces concurrently. After any failed read,
discard potentially partial output.

Allocating Store reads can expand repeated selected columns. Such a view cannot
`prepare()` or write CSR into caller-supplied buffers; dense `out=` remains
supported. See the [worked buffer example](/python/docs/storage.md#reuse-metadata-and-buffers).

### Payload read policy

| ReadOptions field | Default | Meaning |
| --- | --- | --- |
| `strategy` | `"auto"` | Compare range/whole-region read costs; `"ranges"` or `"whole"` forces a policy |
| `max_gap_bytes` | 0 | Maximum gap joined while merging selected ranges |
| `request_cost_bytes` | 4096 | Per-request byte-equivalent cost in the auto heuristic |

`ReadSummary` reports `record_bytes`, `read_bytes`, `requests` and
`whole_chunk_bytes` for one payload execution, excluding metadata preparation.
These are application read estimates, not physical device traffic or calibration.

## Writing and publication

Writer defaults are `format="csr"`, `chunk_rows=4096`, `page_size=4096`,
`options=None`, `workers=None`, `atomic=False`. `write`, `compress` and
`io.write_cpk` instead default stored format to their input layout. Page size is
a power of two from 512 through 65536 bytes; `chunk_rows` counts encoding rows.
`options` accepts `scans.EncodeOptions`.

Every `append(matrix)` must match declared dtype and width. Dense or sparse
batches can feed either stored layout. NumPy buffers are normalized; SciPy sparse
inputs are canonicalized by sorting and summing duplicates. Package CSR/SparseRow
inputs must already have sorted, distinct column indices per row. Explicit
masked values must be filled before encoding.

| Action | Lifecycle |
| --- | --- |
| `append` | Encode more rows in order; stream is unfinished |
| `end_chunk` | End the current encoding chunk early |
| `finish` / successful context exit | Flush/finalize, then publish an atomic destination; returns None |
| `abort` / exceptional context exit | Close unfinished work and discard atomic staging |

Writers belong to their creating process. Without atomic mode an existing path
can be truncated. `atomic=True` stages beside a regular destination and replaces
it only after success; abort preserves the destination. Existing permission bits
are preserved; a new atomic file uses mode 0600. Atomic publication does not
promise persistence across power loss. Keep a completed stream immutable while
readers or loaders use it.

Encoding and decoding release the GIL and use native worker threads.
`workers=None` uses available CPUs, capped at 32 by default; the nonempty
`PYCELLPACK_WORKERS` environment value overrides that default. Explicit counts
and environment overrides must be positive and are capped by available CPU
capacity, including affinity/cgroup quota. They are not fixed to the default
32-worker ceiling.

## File regions, chunk access and identity

`offset` and `length` bound an embedded encoded stream. `File.chunks` returns
its directory. `read_chunk(chunk, out=None, metadata_only=False)` reads encoded
bytes, accepting negative chunk indices; `chunk_meta(chunk)` parses metadata
using the stream dtype. Chunk APIs expose encoded components rather than
materialized matrices; see [codecs](codecs.md) for record operations.

A File retains the opened descriptor and recorded device/inode identity. After
path replacement, existing native reads use the original file; `is_current`
reports whether the path still names it. `to_bytes()` rejects a replacement, and
unpickling rejects a changed identity. `File.from_bytes` copies into anonymous
storage; such Files pickle by copying their encoded bytes. Stores pickle their
file and composed view. Exporting the whole region can require a large buffer.

Invalid types/selectors/buffer declarations raise TypeError, IndexError or
ValueError as appropriate; file failures raise OSError. Those checked errors do
not validate arbitrary trusted record bytes. Keep encoded data paired with its
metadata, and do not modify files in place during reads.

## Complete declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/storage-members.md-->

:::
