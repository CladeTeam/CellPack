# Installation

Install the [published `pycellpack` package from PyPI](https://pypi.org/project/pycellpack/)
in your Python project:

```sh
uv add pycellpack
```

For a new project, run `uv init cellpack-demo` and `cd cellpack-demo` first.
The distribution name and import name are both `pycellpack`.

Version 0.1.0 provides Linux wheels for CPython 3.10–3.15 on x86-64 and AArch64
with glibc 2.28 or newer. Installing a matching wheel requires no Rust toolchain
or CellPack checkout. The source distribution requires a compatible Rust/native
build environment if no wheel matches your platform.

Verify the installation in the project environment:

```sh
uv run python -c "import pycellpack; from importlib.metadata import version; print(version('pycellpack'))"
```

## Dependencies and integrations

| Capability | Dependencies |
|---|---|
| Core package | Python, NumPy and typing-extensions; installed automatically |
| SciPy conveniences | The `scipy` extra |
| AnnData / H5AD / H5CP | The `anndata` extra and its declared Python-version constraints |
| Training Tensor batches | Install PyTorch separately for the target environment |
| Native process/access functions | A compatible native compiler or a prepared shared library; Numba where used |

The current package declares Python `>=3.10,<3.16`. The AnnData extra is pinned
separately and requires Python 3.12 or newer. Importing `pycellpack` does not import
SciPy, AnnData, h5py, pandas, PyTorch, or Numba; integrations load dependencies when used.

Select the needed extras from PyPI:

```sh
uv add 'pycellpack[scipy]'
uv add 'pycellpack[anndata]'
```

The `anndata` extra already includes SciPy and h5py. PyTorch is installed
separately for your CPU or CUDA environment; it is not a PyCellPack extra.

Continue with the [matrix quickstart](quickstart.md) or [training examples](/python/training/).
