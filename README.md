<p align="center">
  <img src="assets/logo.svg" alt="CellPack logo" width="128" height="128">
</p>

<h1 align="center">CellPack</h1>

<p align="center">
  <strong>Compact single-cell storage. Fast random access. Ready for AnnData.</strong>
</p>

<p align="center">
  <a href="https://cladeteam.github.io/CellPack/">Website</a> ·
  <a href="https://github.com/CladeTeam/CellPack/issues">Issues</a> ·
  <a href="LICENSE">MIT License</a>
</p>

<p align="center">
  <img alt="Python 3.10–3.15" src="https://img.shields.io/badge/python-3.10–3.15-1E7A68?logo=python&logoColor=white">
  <img alt="Rust core" src="https://img.shields.io/badge/core-Rust-1E7A68?logo=rust">
  <img alt="Platform: Linux" src="https://img.shields.io/badge/platform-Linux-1E7A68?logo=linux&logoColor=white">
</p>

CellPack stores expression matrices as compact, independently decodable cell
records. Read individual cells or shuffled batches without decoding their
neighbours, work with datasets larger than memory, and keep your AnnData
annotations alongside the compressed matrices.

The storage engine and loader are written in Rust. **`pycellpack`** brings them
to Python, with NumPy, SciPy, AnnData and PyTorch integrations.

## What you can do

- **Compress without losing values.** Shared statistical models capture sparse
  matrix structure while each cell remains independently decodable.
- **Read just the cells and genes you need.** Lazy matrix views support slicing,
  masks and reordered selections; materialize the selected data as dense or CSR.
- **Keep annotated datasets together.** H5CP embeds CellPack matrices in one
  HDF5 file alongside AnnData annotations, layers, embeddings and metadata.
- **Load shuffled data beyond RAM.** Asynchronous I/O overlaps reads and
  decoding; bounded buffers deliver batches in the requested sampling order.
- **Make reuse count.** A planner uses the supplied access sequence to allocate
  a cache budget between compressed pages and decoded rows.
- **Feed model inputs directly.** Align genes across datasets, apply native
  preprocessing and deliver PyTorch batches on CPU or CUDA. Presets cover
  scGPT, Geneformer, UCE and STATE SE.

## AnnData, with compressed matrices

Start with an ordinary AnnData object, including one from an existing Scanpy
workflow. Write it once, then select cells and genes before decoding:

```python
import anndata as ad
import numpy as np
import pycellpack as cp

adata = ad.AnnData(X=np.array([[0, 2, 0], [1, 0, 3]], dtype=np.float32))
adata.obs_names = ["cell-a", "cell-b"]
adata.var_names = ["ACTB", "GAPDH", "CD3D"]
adata.layers["counts"] = adata.X.copy()

# One portable file: compressed matrices and their annotations.
cp.write_h5cp("cells.h5cp", adata, encode=True)
cells = cp.read_h5("cells.h5cp", lazy=True)

# AnnData slicing keeps the embedded matrices lazy.
subset = cells[["cell-b"], ["CD3D", "ACTB"]]
small = subset.to_memory()
print(small.X)  # [[3. 1.]]

# Export an ordinary H5AD for downstream tools.
cp.write_h5ad("subset.h5ad", small)
```

`encode=True` compresses numeric matrix slots; `lazy=True` keeps embedded
matrices on disk while annotations are loaded into memory. Decode a subset
with `to_memory()` before passing it to tools that expect NumPy or SciPy
matrices. H5CP is read through `pycellpack`; `write_h5ad()` exports standard H5AD.

For training, `ScDataset.from_anndata(cells, ids=gene_ids)` connects a lazy
matrix to `ScDataLoader`. Supply one vocabulary ID per gene and your own sample
order; the loader handles alignment, preprocessing and batch delivery.

## Performance

The figures below are newly drawn from frozen experimental results. They
describe the experimental implementation used for the technical report,
rather than benchmarks of the current commit.

### Compact storage

![Compression ratios for CellPack and selected general-purpose codecs; larger is better.](assets/performance-compression.svg)

CellPack reaches **5.89× compression** while retaining per-cell decoding.
Ratios are means across raw and normalized matrix blocks: 734 for CellPack and
724 for the other codecs. These are corpus summaries, not a paired-input
comparison or a ranking of decoding speed.

### Random-order loading

![Global random-order loading on one NVMe SSD with 16 workers; a logarithmic axis shows cells per second.](assets/performance-loading.svg)

On one NVMe SSD with 16 workers, the report gives **1.165M cells/s** for
CellPack without a data cache. Baselines use a **20 GB cache** and their best
variant, averaged over three seeds. CellPack's point is the report's
author-selected value for a 5 s post-warm-up check, without a variance estimate;
baselines use 60 s measurement windows. CellPack emits a fixed 2,048-gene panel;
baselines use their sample-output paths. Only global random-order methods are
shown.

The [figure data and provenance](assets/performance.json) record the exact
values and source hashes. Both figures can be regenerated with
`uv run scripts/plot_readme.py`.

## Install

From a repository checkout, with a Rust toolchain installed:

```sh
# Python 3.12+ for AnnData integration (pinned to AnnData 0.13.4).
uv add '/path/to/CellPack[anndata]'

# Or install the matrix storage and loader core (Python 3.10–3.15).
uv add /path/to/CellPack
```

CellPack targets Linux. The default loader uses `io_uring`; a `pread` backend
is available where `io_uring` is disabled. Install PyTorch separately for
`ScDataLoader`, choosing the build appropriate for your device.

## License

[MIT](LICENSE).
