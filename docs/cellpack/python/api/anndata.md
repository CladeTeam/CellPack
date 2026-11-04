# AnnData and H5CP API

Use this reference for file/slot arguments, storage policy and materialization.
The [AnnData guide](/python/docs/anndata.md) provides a complete annotated matrix
round trip with checked values.

```python
from pycellpack import read_h5, write_h5ad, write_h5cp, open_matrix
from pycellpack.io import h5ad_to_cpk
from pycellpack.anndata import AnnData
```

These calls load the optional `anndata` integration. The package pins its
AnnData version and requires Python >=3.12 for that extra. Importing the core
package does not eagerly import AnnData, h5py, SciPy, pandas or Zarr.

<DocChoices title="Select the object you need" :items="[
  { label: 'read_h5 → AnnData', when: 'Read matrices together with annotations', detail: 'Choose eager decoding or preserve embedded streams as Stores.', href: '#reading-and-backed-behavior' },
  { label: 'open_matrix → Store', when: 'Read a single embedded matrix', detail: 'Keep its file region lazy without loading the full annotated dataset.', href: '#matrix-slots-and-export' },
  { label: 'write_h5cp / write_h5ad', when: 'Publish an annotated dataset', detail: 'Select embedded or decoded matrix storage explicitly.', href: '#writing-and-publication' },
]" />

## Reading and backed behavior

`read_h5(filename, backed=None, *, lazy=False, **kwargs)` selects behavior from
file contents, not the filename suffix:

| Contents / options | Result and allocation |
| --- | --- |
| Embedded streams, defaults | Adapted AnnData; streams decoded to ordinary arrays |
| Embedded streams, `lazy=True` or `backed="r"` | Adapted AnnData; embedded matrices remain Stores; other elements read into memory |
| No embedded streams, `backed=None` | AnnData's normal eager read, adapted to this subclass |
| No embedded streams, backed read | AnnData's backed object returned unchanged |

For files containing streams, `backed` accepts only None, False or `"r"`;
additional AnnData read kwargs are rejected. In particular, `backed="r+"` is not
supported. For ordinary H5AD, `backed` and kwargs are forwarded to AnnData;
`lazy=True` does not add CellPack encoding or make its arrays Stores.

Embedded Stores keep independent file descriptors after the metadata handle
closes. They provide immutable matrix access, rather than writable HDF5-backed
arrays. Preserve the source file while derived Stores and datasets use it.

## Writing and publication

The module functions take **filepath first, adata second**:

| Function | Result | Storage behavior |
| --- | --- | --- |
| `write_h5cp(filepath, adata, ...)` | Resolved absolute `Path` | Policy selected by `encode` below |
| `write_h5ad(filepath, adata, ...)` | None | Plain H5AD; existing Stores decoded in row batches |
| `adata.write_h5cp(filename=None, ...)` | None | Instance convenience using the same H5CP policy |
| `adata.write_h5ad(filename=None, ...)` | None | Instance convenience for plain H5AD |

A missing instance filename uses the AnnData backing filename; an unbacked
object must receive an explicit filename.

| `write_h5cp` policy | Existing Store slots | Ordinary supported numeric matrices |
| --- | --- | --- |
| `encode=None` (default) | Embed compressed streams | Keep normal AnnData encoding |
| `encode=True` | Embed compressed streams | Encode matrix slots as CellPack streams |
| `encode=False` | Decode to ordinary datasets | Keep normal AnnData encoding |

Supported numeric NumPy/SciPy matrices have at least two dimensions and a
positive product of trailing dimensions, using supported integer or float32/64
dtypes. Dense higher-rank matrices retain their logical shape, with flattened
columns in the embedded stream. Annotation values use AnnData encodings.

### Encoding parameters

| Parameter | Default | Contract |
| --- | --- | --- |
| `chunk_rows` | 4096 | New-encoding rows per chunk, integer in `[1, 2**32)` |
| `page_size` | 4096 | New-encoding page bytes; power of two in `[512, 65536]` |
| `workers` | None | Encoding/decoding threads; storage worker resolution applies |
| `convert_strings_to_categoricals` | True | Convert copied annotation frames on the stream-writing path |
| `dataset_kwargs`, `**kwargs` | Empty | Options for ordinary AnnData datasets, not embedded stream compression |
| `as_dense` | `()` | AnnData dense-export selection; nonempty values rejected on the H5CP stream-writing path |

Whole identity Stores whose source paths are still current are copied byte for
byte. Slices, column maps and dtype-converting views are decoded and re-encoded.
`chunk_rows` and `page_size` affect those new encodings, not copied streams.
Embedded payloads are contiguous, unfiltered uint8 HDF5 datasets. Their largest
page size determines HDF5 region alignment; no HDF5 filter wraps those payloads.

When writing streams or decoding existing Stores, the implementation stages in
a sibling temporary directory and replaces the destination only on success.
It can therefore write an annotated object over the file its Stores read from.
When no Stores exist and `encode` is None or False, `write_h5cp` delegates directly
to AnnData; `write_h5ad` also delegates directly when no Stores exist. These
branches follow upstream publication behavior. The staging paths do not promise
power-loss durability.

## Matrix slots and export

Both functions accept AnnData slot paths: `X`, `layers/<name>`, `obsm/<name>`,
`obsp/<name>`, `varm/<name>`, `varp/<name>`, `raw/X` and `raw/varm/<name>`.
Leading/trailing slashes are stripped; individual keys must not contain path
separators, NUL, `.` or `..`. These paths identify matrix groups, not their
internal `data` datasets.

| Function | Input | Result / important constraint |
| --- | --- | --- |
| `open_matrix(filename, key="X")` | Supported embedded stream group | Lazy Store with logical shape; no annotations loaded |
| `h5ad_to_cpk(filename, path, key="X", ...)` | Embedded stream, 2-D dense dataset or CSR group | File-backed Store over exported `.cpk`; annotations stay in the source |

`h5ad_to_cpk` uses `batch_rows=65536` for ordinary matrix inputs; it must be
positive. `format=None`, `workers=None`, `chunk_rows=4096`, `page_size=4096`,
`options=None` and `atomic=False` configure their row-batched encoding through
`io.write_cpk`. Ordinary CSC and unsupported element encodings are not converted
by this row-export helper.

For an already embedded stream, export copies encoded bytes directly. That
branch does not apply requested format/encoding options or `atomic=True`; use a
separate destination path. `open_matrix` also requires an embedded stream: it
does not lazily wrap an ordinary H5AD dense/CSR matrix. Its File owns a bounded
region after the HDF5 metadata handle has closed.

## AnnData materialization and datasets

| Interface | Store behavior |
| --- | --- |
| Constructor / `X`, `layers`, `obsm/varm`, `obsp/varp`, raw | Accept Stores alongside ordinary AnnData arrays; aligned axes are validated |
| Slicing | Compose Store selections lazily |
| `copy()` | Copy annotations while sharing immutable Stores |
| `is_lazy` | True if any matrix slot, including raw, contains a Store |
| `to_memory(copy=False)` | Return a new AnnData with every Store and backed array materialized; Store decoding allocates regardless of copy |
| `transpose()` / `T` | Materialize remaining Stores before transposing |
| `to_df(layer=None)` | Materialize the selected matrix as a dense data frame |
| `chunked_X(chunk_size=None)` | Yield decoded chunks with start/end row positions |
| `chunk_X(select=1000, replace=True)` | Materialize only selected/sampled X rows as dense output |

`ScDataset.from_anndata(adata, ids=..., layer=None)` selects X, or the named
layer. It requires an existing 2-D file-backed Store in its stored dtype and one
explicit standard ID per visible column. It does not encode ordinary arrays or
match gene names. See [dataset contracts](dataloader.md#dataset-and-sample-declarations).

## Errors and boundaries

Missing optional packages raise ModuleNotFoundError with installation guidance.
Invalid shapes, layout declarations, slot paths and unsupported embedded stream
layouts raise ValueError. Missing HDF5 slots raise KeyError; unsupported read
modes, kwargs, dense-writing requests or row-export encodings raise
NotImplementedError. Filesystem/HDF5 and native I/O errors propagate.

On the stream-writing path, metadata mappings are checked for unsafe keys and
recursive mappings before publication. These file-structure checks do not make
trusted CellPack record bytes a validation boundary. Keep source streams
immutable during reads and exports; after replacing a source path, reopen the
new stream before constructing a new training source.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/anndata-members.md-->

:::
