# AnnData, H5AD, and H5CP

Use H5CP when an annotated dataset should retain its AnnData structure while
selected matrix slots remain compressed, lazy CellPack Stores. Install the
`anndata` extra with Python 3.12 or newer; the integration follows the pinned
AnnData version. The core package does not load these optional dependencies
until an integration is used.

<DocChoices title="Choose what to keep" :items="[
  { label: 'H5CP', when: 'Keep annotations and lazily read embedded matrices', detail: 'Encode numeric matrix slots or embed existing Stores, then read with lazy=True.', href: '#encode-an-annotated-dataset' },
  { label: 'Plain H5AD', when: 'Exchange ordinary AnnData matrices', detail: 'Write decoded matrices with write_h5ad. A plain H5AD does not become a Store through lazy=True.', href: '#select-before-materializing' },
  { label: 'One .cpk matrix', when: 'Prepare a standalone matrix for a loader', detail: 'Export a matrix slot; keep its row names, gene names and annotations separately.', href: '#open-or-export-one-matrix' },
]" />

<DocDiagram name="anndata" />

## Encode an annotated dataset

Run the following blocks in order in a writable directory. They create or
replace the named example files. This first block stores both `X` and a layer
as embedded streams, then checks values and annotation order:

```python
import numpy as np
from pycellpack.anndata import AnnData
from pycellpack import io

x = np.array([[0, 2, 0], [1, 0, 3]], np.float32)
adata = AnnData(X=x)
adata.obs_names = ["cell-a", "cell-b"]
adata.var_names = ["ACTB", "GAPDH", "CD3D"]
adata.layers["counts"] = x + 1
written = io.write_h5cp("example.h5cp", adata, encode=True, workers=1)
assert written.name == "example.h5cp"
lazy = io.read_h5(written, lazy=True)
assert lazy.is_lazy
assert lazy.obs_names.tolist() == ["cell-a", "cell-b"]
assert lazy.var_names.tolist() == ["ACTB", "GAPDH", "CD3D"]
np.testing.assert_array_equal(lazy.X.read_dense(workers=1), x)
np.testing.assert_array_equal(lazy.layers["counts"].read_dense(workers=1), x + 1)
print(lazy.shape, lazy.is_lazy)
```

```text
(2, 3) True
```

`encode=True` is intentional: the default `encode=None` embeds existing Stores
but leaves ordinary NumPy/SciPy matrices in their usual AnnData encoding. The
`.h5cp` suffix alone does not compress a matrix.

An embedded matrix is a Store over a bounded region of the HDF5 file. It owns
its file descriptor after the temporary HDF5 metadata handle closes. Annotations
and ordinary matrix elements are read into memory; `lazy=True` does not make the
whole AnnData object lazy or turn a plain H5AD matrix into a Store.

## Select before materializing

Slicing the adapted AnnData composes Store selections. Decode only the portion
you need, or explicitly export all matrices to plain H5AD:

```python
selected = lazy[[1, 0], [2, 0]]
assert selected.is_lazy
in_memory = selected.to_memory()
assert not in_memory.is_lazy
np.testing.assert_array_equal(in_memory.X, [[3, 1], [0, 0]])

io.write_h5ad("decoded.h5ad", lazy)
ordinary = io.read_h5("decoded.h5ad", lazy=True)
assert not ordinary.is_lazy
np.testing.assert_array_equal(ordinary.X, x)
```

`to_memory()` decodes every remaining Store, including layers and raw slots.
`copy=False` does not avoid allocating decoded matrix buffers. `copy()` alone
keeps immutable Stores shared. `transpose()` materializes Stores, while `to_df`
materializes the selected matrix and produces a dense data frame. Check
`is_lazy` and subset before operations whose decoded result could be large.

For a file containing streams, `backed="r"` also preserves embedded Stores;
other elements still enter memory. This differs from ordinary AnnData backing.
For plain H5AD, `backed` follows AnnData and backed results are returned unchanged.

## Open or export one matrix

Use slot paths such as `X`, `layers/counts` or `raw/X`, rather than a path to
the internal `data` dataset:

```python
view = io.open_matrix("example.h5cp", key="layers/counts")
assert view.shape == (2, 3)
np.testing.assert_array_equal(view.read_dense(workers=1), x + 1)
exported = io.h5ad_to_cpk(
    "example.h5cp", "extracted.cpk", key="layers/counts", workers=1,
)
np.testing.assert_array_equal(exported.read_dense(workers=1), x + 1)
```

`open_matrix` requires an existing embedded stream and returns a lazy Store;
it does not encode an ordinary H5AD matrix. `h5ad_to_cpk` also supports ordinary
2-D dense datasets and CSR groups, encoding them in row batches. An embedded
stream is copied as encoded bytes, so encoding and format options do not
transform that branch. Export to a separate destination path.

A `.cpk` file keeps matrix positions and values. Preserve the original `obs`
and `var` metadata when you need to interpret its rows or columns.

## Build a training dataset from a Store slot

The adapter does not infer gene identity or compress arrays implicitly:

```python
from pycellpack import ScDataset

dataset = ScDataset.from_anndata(
    lazy, ids=np.array([10, 20, 30], np.uint32), layer="counts",
)
assert dataset.shape == (2, 3)
np.testing.assert_array_equal(dataset.ids, [10, 20, 30])
```

`layer=None` selects `X`; a layer name selects `adata.layers[layer]`. The chosen
slot must already hold a two-dimensional file-backed Store. Supply one standard
ID per visible column, in its current order. Continue with [gene alignment](alignment.md)
and the [DataLoader guide](dataloader.md) for sample order and Tensor batches.

## Choose how a write handles Stores

Whole, unchanged Stores with current source paths are copied as encoded bytes;
selected or converted views are re-encoded. Both preserve the represented
matrix. `chunk_rows` and `page_size` control new encoding, not the layout of an
unchanged copied stream. H5CP embeds streams in contiguous, unfiltered uint8
regions; compression keywords apply to ordinary AnnData datasets instead.

The stream-writing paths stage beside the destination and replace it after
success, including when the input reads that destination. Calls that simply
forward an object without Stores to AnnData follow AnnData's writing behavior.
See the [API reference](/python/api/anndata.md#writing-and-publication) for the
three `encode` policies, exact return values and publication boundaries.
