"""Teach AnnData's array dispatchers to keep Stores lazy.

Subsetting composes the Store's selection, views share it (Stores are
immutable), and ``to_memory`` decodes it.
"""

from typing import Any, cast

from anndata._core.file_backing import to_memory
from anndata._core.index import _ensure_numpy_idx, _subset
from anndata._core.views import as_view

from ..cellpack import Store


@_subset.register(Store)
@_ensure_numpy_idx
def _subset_store(store: Store, subset_idx) -> Store:
    """Subset a Store by AnnData's index tuple; the result composes the views."""
    return store[tuple(subset_idx)]


@cast(Any, as_view).register(Store)
def _view_store(store: Store, view_args) -> Store:
    """Views share the Store: it is immutable, so no view layer is needed."""
    del view_args
    return store


@to_memory.register(Store)
def _store_to_memory(store: Store, *, copy: bool = False):
    """Decode a Store for AnnData's ``to_memory``; decoding always allocates."""
    del copy
    return store.to_memory()
