"""AnnData with native CellPack Store support.

:class:`AnnData` subclasses :class:`anndata.AnnData`: any matrix slot can hold
a lazy :class:`~pycellpack.cellpack.Store`, slicing keeps it lazy, and
``to_memory()`` decodes it. This module holds the subclass definitions only;
files are read and written by :mod:`pycellpack.io` (``read_h5``,
``write_h5cp``, ``write_h5ad``). Importing it also registers the CellPack
encoding with AnnData's own reader and writer.
"""

try:
    # Register the CellPack encoding after the matrix classes and dispatchers exist.
    from ..io import _anndata as _anndata
    from . import _dispatch as _dispatch
    from ._core import AnnData, Raw, as_store, iter_stores
except ModuleNotFoundError as exc:
    if exc.name not in {"anndata", "h5py", "pandas", "scipy", "zarr"}:
        raise
    raise ModuleNotFoundError(
        f"AnnData integration requires the optional dependency {exc.name!r}. "
        'Install with: uv add "pycellpack[anndata]" (Python >=3.12).',
        name=exc.name,
    ) from exc

__all__ = ["AnnData", "Raw", "as_store", "iter_stores"]
