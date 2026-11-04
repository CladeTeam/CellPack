"""Reading and writing matrices: ``.cpk`` streams, H5AD and H5CP files.

``.cpk`` tools need no optional dependency: :func:`open_cpk` (a lazy Store),
:func:`read_cpk`, :func:`iter_cpk`, :func:`cpk_info` and :func:`write_cpk`
(from a matrix, a Store or row batches).

H5CP files are H5AD files whose matrices are embedded CellPack streams:
:func:`read_h5` reads H5CP and H5AD, :func:`write_h5cp` and
:func:`write_h5ad` write them, :func:`open_matrix` opens one embedded matrix
lazily and :func:`h5ad_to_cpk` exports a matrix slot to a ``.cpk`` file.
These live in :mod:`pycellpack.io._anndata` and load AnnData, h5py and SciPy
on first use (``pycellpack[anndata]``), so importing this module does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._cpk import CpkInfo, cpk_info, iter_cpk, open_cpk, read_cpk, write_cpk

__all__ = [
    "CpkInfo",
    "cpk_info",
    "h5ad_to_cpk",
    "iter_cpk",
    "open_cpk",
    "open_matrix",
    "read_cpk",
    "read_h5",
    "write_cpk",
    "write_h5ad",
    "write_h5cp",
]

if TYPE_CHECKING:
    from ._anndata import h5ad_to_cpk, open_matrix, read_h5, write_h5ad, write_h5cp
else:

    def _forward(name: str):
        def entry(*args, **kwargs):
            # The AnnData integration reports missing optional dependencies.
            from .. import anndata as _integration  # noqa: F401
            from . import _anndata

            return getattr(_anndata, name)(*args, **kwargs)

        entry.__name__ = entry.__qualname__ = name
        entry.__module__ = __name__
        entry.__doc__ = f"See :func:`pycellpack.io._anndata.{name}` (loads the AnnData integration)."
        return entry

    h5ad_to_cpk = _forward("h5ad_to_cpk")
    open_matrix = _forward("open_matrix")
    read_h5 = _forward("read_h5")
    write_h5ad = _forward("write_h5ad")
    write_h5cp = _forward("write_h5cp")
