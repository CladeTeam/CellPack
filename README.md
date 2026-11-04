<p align="center">
  <img src="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/logo.svg" alt="CellPack logo" width="128" height="128">
</p>

<h1 align="center">CellPack</h1>

<p align="center">
  <strong>Compact single-cell storage. Million-cell-per-second loading. Built for AnnData.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycellpack/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pycellpack?color=1E7A68&label=pypi"></a>
  <a href="https://crates.io/crates/cellpack"><img alt="crates.io" src="https://img.shields.io/crates/v/cellpack?color=1E7A68"></a>
  <img alt="Python 3.10–3.15" src="https://img.shields.io/badge/python-3.10–3.15-1E7A68?logo=python&logoColor=white">
  <a href="https://github.com/CladeTeam/CellPack/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-1E7A68"></a>
</p>

<p align="center">
  <a href="https://cladeteam.github.io/CellPack/">Website</a> ·
  <a href="https://cladeteam.github.io/CellPack/python/">Python docs</a> ·
  <a href="https://docs.rs/cellpack">Rust docs</a> ·
  <a href="https://cladeteam.github.io/CellPack/report.html">Technical report</a>
</p>

CellPack is a storage format and data loader for single-cell expression
matrices. Every cell is a compressed record that can be read on its own, so
shuffled training batches stream straight from disk at over a million cells
per second. Your AnnData files stay AnnData files, at a fraction of the size.

## Install

```sh
pip install pycellpack            # Python
pip install "pycellpack[anndata]" # with AnnData integration
uv add pycellpack                 # or with uv
cargo add cellpack                # Rust
```

Start with the [Python quickstart](https://cladeteam.github.io/CellPack/python/docs/quickstart.html).

## 14.7× faster random-order loading

Training wants shuffled cells. CellPack reads, decodes and batches them in any
order you choose, from a single SSD and with no data cache.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-loading-dark.svg">
  <img src="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-loading.svg" alt="Global random-order loading from one NVMe SSD with 16 workers. CellPack: 1,165,000 cells/s, 14.7× the next-fastest loader, Lance at 79,135 cells/s." width="1120">
</picture>

## Keep your AnnData workflow

An H5CP file is an H5AD file whose expression matrix `X` is a CellPack matrix.
Everything else keeps its AnnData encoding, so `obs`, `var`, embeddings and
metadata work as before.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/anndata-integration-dark.svg">
  <img src="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/anndata-integration.svg" alt="Swap only X. In H5AD, X is stored as CSR arrays. In H5CP, X is a CellPack matrix, 9.74× smaller. obs, var, obsm, obsp, varm, varp, layers and uns keep their place in the AnnData layout." width="1120">
</picture>

```python
import pycellpack as cp

cp.write_h5cp("cells.h5cp", adata, encode=True)
cells = cp.read_h5("cells.h5cp", lazy=True)
subset = cells[cell_ids, gene_ids].to_memory()  # an AnnData object for Scanpy
```

Slices stay on disk until you materialize them, and the same file feeds the
PyTorch loader. See the
[AnnData guide](https://cladeteam.github.io/CellPack/python/docs/anndata.html).

## A fraction of the disk

A 124M-cell CELLxGENE corpus takes 4.87 TB as H5AD and **611 GB** as H5CP. Its
expression matrices shrink from 2.22 TB to 227.7 GB.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-storage-dark.svg">
  <img src="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-storage.svg" alt="A 124M-cell CELLxGENE corpus. Whole files: 4.87 TB as H5AD, 611 GB as H5CP, 7.96× smaller. Expression matrix X: 2.22 TB versus 227.7 GB, 9.74× smaller." width="1120">
</picture>

## Built for single-cell data

General-purpose codecs compress well only when many cells share one block,
and lose most of their ratio when each cell must be readable alone. CellPack
models the structure of expression data and compresses best of all while
keeping every cell independently decodable.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-compression-dark.svg">
  <img src="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-compression.svg" alt="Compression ratio on 734 CELLxGENE matrices. CellPack 5.89× with every cell independently decodable. One cell per unit versus 2,048-cell blocks: PCodec 4.88× and 4.81×, XZ 2.11× and 4.79×, Zstd 22 1.95× and 4.34×, Zstd 3 1.85× and 3.09×, LZ4 1.42× and 1.69×." width="1120">
</picture>

## One billion cells

CellPack delivers 5 billion cells from a 1B-cell corpus in 25.7 minutes, with
a fresh global shuffle every epoch and every cell decoded for the model.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-billion-dark.svg">
  <img src="https://raw.githubusercontent.com/CladeTeam/CellPack/main/assets/performance-billion.svg" alt="Five fresh global shuffles of a 1B-cell corpus: 3.24M cells/s including startup, 3.67M cells/s in steady state, 5B cells in 25.7 minutes." width="1120">
</picture>

## Learn more

- [Website](https://cladeteam.github.io/CellPack/): how CellPack works, with
  interactive explanations
- [Python documentation](https://cladeteam.github.io/CellPack/python/):
  storage, AnnData, the training loader and the full API
- [Rust documentation](https://docs.rs/cellpack)
- [Technical report](https://cladeteam.github.io/CellPack/report.html):
  methods and full benchmarks. Figure data and sources are in
  [performance.json](https://github.com/CladeTeam/CellPack/blob/main/assets/performance.json).

## License

[MIT](https://github.com/CladeTeam/CellPack/blob/main/LICENSE)
