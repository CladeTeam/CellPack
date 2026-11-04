# Matrix access and streaming writes

A `Store` describes an immutable matrix view over a `File`. Selecting rows,
selecting columns and changing the requested dtype compose that view; they do
not decode matrix values. A read produces ordinary NumPy or CSR arrays.

<DocChoices title="Choose your storage path" :items="[
  { label: 'Select and read', when: 'You need an array from an existing matrix.', href: '#compose-selections', detail: 'Compose a lazy view, then choose dense or CSR output.' },
  { label: 'Prepare and reuse', when: 'You will read the same selection into reusable buffers.', href: '#reuse-metadata-and-buffers', detail: 'Prepare metadata once; keep one workspace per active read.' },
  { label: 'Append row batches', when: 'The complete input does not fit in one array.', href: '#write-a-matrix-in-row-batches', detail: 'Declare the dtype and width, append batches, then finish the stream.' }
]" />

<DocDiagram name="python-storage" />

## Compose selections

Run the following blocks in order in a writable directory. The first block
creates or replaces `selection.cpk`.

```python
import numpy as np
import pycellpack as cp

x = np.arange(24, dtype=np.float32).reshape(6, 4)
matrix = cp.write("selection.cpk", x, workers=1, atomic=True)
view = matrix[[5, 1, 5], [3, 0, 3]].astype(np.float64)
result = view.read_dense(workers=1)
np.testing.assert_array_equal(result, x[np.ix_([5, 1, 5], [3, 0, 3])])
assert result.dtype == np.float64
print(result)
```

```text
[[23. 20. 23.]
 [ 7.  4.  7.]
 [23. 20. 23.]]
```

The row and column selectors are **orthogonal**: three row IDs and three column
IDs produce a 3 × 3 result, rather than three paired elements. Order and repeats
are preserved. An integer keeps its axis: `matrix[1, 2].shape` is `(1, 1)`.
Slices, including negative steps, integer arrays and boolean masks work on both
axes. Positions refer to the current view, not necessarily the original file.

`astype` chooses the decode dtype. `copy()` can return the same immutable Store;
it does not materialize values. Streams contain positional matrix data, not
cell names or gene names. For an explicit column-to-panel map, continue with
[gene alignment](alignment.md#mapping-a-store-without-a-loader).

## Choose a result type

| Call | Result | Optional dependency |
| --- | --- | --- |
| `view.read_dense()` / `view.toarray()` | NumPy ndarray in the view's logical shape | None |
| `view.read_csr()` | NumPy-backed `CSR(indptr, indices, data, shape)` | None |
| `view.read()` | Dense ndarray or CSR according to the stored format | None |
| `view.tocsr()` | SciPy `csr_array` | SciPy |
| `view.to_memory()` | Dense ndarray or SciPy `csr_array` | SciPy for sparse output |

These arrays belong to the read result; retaining them does not lease a loader
ring slot. `iter_batches(batch_size=4096, format=...)` decodes successive rows of
the current view, preserving its order and repeated rows.

Higher-rank dense views use `Store(file, shape=...)`; trailing dimensions must
flatten to the stream width. They support row selection and dense reads only.

## Reuse metadata and buffers

Use a selection without repeated columns for a prepared read:

```python
from pycellpack.cellpack import ReadWorkspace

selection = matrix[[5, 1], [3, 0]].prepare()
workspace = ReadWorkspace()
out = np.empty(selection.shape, dtype=selection.dtype)
returned = selection.read_dense(out=out, workspace=workspace, workers=1)
assert returned is out
np.testing.assert_array_equal(out, [[23, 20], [7, 4]])
workspace.clear()
```

Preparation reads and retains metadata. The dense output must have the exact
shape and dtype, native byte order, suitable alignment and writable C-contiguous
storage. The workspace retains scratch between reads; use separate workspaces
for concurrent calls. Keep borrowed arrays unchanged until a call returns, and
discard partially written output after a failed read.

For CSR output, capacity and actual stored-entry count are different:

```python
indptr = np.empty(selection.shape[0] + 1, np.int64)
indices = np.empty(selection.capacity_nnz, np.int64)
data = np.empty(selection.capacity_nnz, selection.dtype)
csr = selection.read_csr(out=(indptr, indices, data), workspace=workspace, workers=1)
assert csr.shape == (2, 2)
assert csr.nnz == 4
assert csr.indices.size == csr.data.size == csr.nnz
```

The returned CSR uses your buffers and trims indices/data to the actual count.
Dropped columns can make that count smaller than `capacity_nnz`. Allocating
reads support repeated columns, as in the first example; those views cannot be
prepared or read into caller-supplied CSR buffers.

::: details Choose physical read ranges
`File.prepare_read(..., read_options=ReadOptions(...))`, or
`view.prepare(read_options=...)`, selects the payload read policy.
`strategy="ranges"` coalesces selected ranges, `"whole"` reads whole record
regions and `"auto"` compares bytes plus a per-request cost.
`max_gap_bytes=0` and `request_cost_bytes=4096` are heuristic defaults, not device
calibration. Inspect `selection.io_summary` for planned bytes and requests;
it excludes metadata preparation.
:::

## Write a matrix in row batches

This continues with the same `x` and creates or replaces `streamed.cpk`:

```python
with cp.Writer("streamed.cpk", n_cols=4, dtype=np.float32,
               format="dense", chunk_rows=3, workers=1, atomic=True) as writer:
    writer.append(x[:2])
    writer.append(x[2:])
np.testing.assert_array_equal(cp.read("streamed.cpk", workers=1), x)
```

Every appended batch must have the declared dtype and column count. Successful
context exit finishes the stream; exceptional exit aborts. Manual use requires
`finish()` or `abort()`. `finish()` returns `None`; reopen the path to read it.
`chunk_rows` controls encoding chunks and `page_size` controls page alignment;
neither is a training batch size.

`write` and `compress` take NumPy, SciPy sparse, `CSR` and `SparseRow` inputs.
SciPy sparse inputs are canonicalized, including duplicate summation. Package
CSR inputs must already have sorted, distinct column indices in each row.
`compress` returns a Store over an anonymous in-memory stream rather than a
file-backed training source. `io.write_cpk` also streams a Store or an iterable
of row batches without retaining the full matrix.

`atomic=True` stages a sibling file and replaces the destination only after
success. Aborting discards staging and preserves the existing destination.
Without atomic mode, opening a writer can truncate an existing path. Atomic
publication does not promise power-loss durability.

## File identity and embedded streams

`File(path, offset=0, length=None)` opens a stream or bounded region inside another
file. `File.from_bytes(buffer)` copies an encoded stream into anonymous storage;
`file.to_bytes()` exports the complete region and can allocate a large buffer.
Use `io.cpk_info(source)` to inspect shape, dtype, chunks and encoded size without
decoding records.

An opened File keeps reading its original descriptor if the path is replaced.
`is_current` reports whether the path still names that file; `to_bytes()` and
unpickling reject a replacement. Reopen explicitly before constructing a new
dataset from a newly published stream. Keep stream contents immutable while
prepared selections or loaders use them.

See the [storage API](/python/api/storage.md) for exact buffer, mapping and
publication contracts.
