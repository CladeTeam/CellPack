# Installation

Use the `pycellpack` distribution and import package. From a repository checkout:

```sh
uv add /absolute/path/to/CellPack
```

This installs from source through maturin and requires a compatible Rust toolchain
and native build environment. Use an authorized build worker where compilation is
restricted. An already built wheel can be installed through `uv add /path/to/wheel.whl`.

## Dependencies and integrations

| Capability | Dependencies |
|---|---|
| Core package | Python, NumPy and typing-extensions, as declared in `pyproject.toml` |
| SciPy conveniences | The `scipy` extra |
| AnnData / H5AD / H5CP | The `anndata` extra and its declared Python-version constraints |
| Training Tensor batches | Install PyTorch separately for the target environment |
| Native process/access functions | A compatible native compiler or a prepared shared library; Numba where used |

The current package declares Python `>=3.10,<3.16`. The AnnData extra is pinned
separately and requires Python 3.12 or newer. Importing `pycellpack` does not import
SciPy, AnnData, h5py, pandas, PyTorch, or Numba; integrations load dependencies when used.

For an existing project, select the needed extras when adding the checkout:

```sh
uv add '/absolute/path/to/CellPack[scipy,anndata]'
```

Continue with the [matrix quickstart](quickstart.md) or [training examples](/python/training/).
