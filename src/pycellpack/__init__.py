"""Common matrix storage, loading and H5CP tools, plus their subsystem APIs.

The package root re-exports the storage entry points (:class:`File`,
:class:`Store`, :class:`Writer`, ``read``, ``write``, ``compress``), the
corpus ``Accesses`` dataclass and loader objects (``DataSource``, ``Loader``,
``LoaderConfig``, ``OutputSchema``, ``ProcessFn``, ``DistributedLoader``,
``load``, ``compile_library``), the planning objects and the H5CP functions;
``cellpack``, ``loader`` (with ``loader.opt``, ``loader.costs`` and
``loader.process_fn``), ``io`` and ``anndata`` are importable submodules.
AnnData integration lives in :mod:`pycellpack.anndata`, so importing this
package loads no optional dependency.
"""

from . import cellpack, dataloader, io, loader
from .cellpack import CSR, File, Store, Writer, compress, read, write
from .dataloader import Accesses, ScDataLoader, ScDataset
from .io import open_matrix, read_h5, write_h5ad, write_h5cp
from .loader import (
    AccessFn,
    DataSource,
    DistributedLoader,
    Loader,
    LoaderConfig,
    OutputSchema,
    Plan,
    PlanOptions,
    ProcessFn,
    compile_library,
    load,
)

__all__ = [
    "CSR",
    "AccessFn",
    "Accesses",
    "DataSource",
    "DistributedLoader",
    "File",
    "Loader",
    "LoaderConfig",
    "OutputSchema",
    "Plan",
    "PlanOptions",
    "ProcessFn",
    "ScDataLoader",
    "ScDataset",
    "Store",
    "Writer",
    "cellpack",
    "compile_library",
    "compress",
    "dataloader",
    "io",
    "load",
    "loader",
    "open_matrix",
    "read",
    "read_h5",
    "write",
    "write_h5ad",
    "write_h5cp",
]
