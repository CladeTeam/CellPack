"""Single-cell datasets: lazy Store views with explicit standard column ids."""

from __future__ import annotations

import numpy as np

from ..cellpack import Store
from ..loader._source import NONE, _std_map


class ScDataset:
    """A two-dimensional Store and one explicit standard id per visible dim.

    ids must be a NumPy integer vector. -1/uint32.max drops a dim; other ids
    must be distinct. Rows address this Store's current view. Ordinary column
    subsets/reorderings stay lazy. Multiple mapped dims referring to the same
    stored column and dtype-converting Store views are rejected explicitly.
    """

    def __init__(self, store: Store, *, ids: np.ndarray):
        if not isinstance(store, Store):
            raise TypeError("ScDataset needs a CellPack Store; use adata.X or adata.layers[layer]")
        if store.ndim != 2:
            raise ValueError("ScDataset needs a two-dimensional Store")
        if store.file.in_memory:
            raise ValueError("ScDataset needs a file-backed Store; write the in-memory stream to a .cpk first")
        if not isinstance(ids, np.ndarray):
            raise TypeError("ids must be an explicit NumPy array of standard ids")
        mapping = _std_map(ids)
        if len(mapping) != store.shape[1]:
            raise ValueError(f"ids needs {store.shape[1]} entries for this Store, got {len(mapping)}")
        mapped = mapping[mapping != NONE]
        if len(np.unique(mapped)) != len(mapped):
            raise ValueError("mapped standard ids must be distinct within a dataset")
        if store.dtype != store.file.dtype:
            raise ValueError("ScDataset needs the stored dtype; choose the mapped dtype on ScDataLoader instead of Store.astype")
        genes = store.genes
        if genes is not None:
            selected = genes[(genes >= 0) & (mapping != NONE)]
            if len(np.unique(selected)) != len(selected):
                raise ValueError("a stored column cannot contribute multiple mapped dataset dims")
        mapping.setflags(write=False)
        self._store, self._ids = store, mapping

    @classmethod
    def from_anndata(cls, adata, *, ids: np.ndarray, layer=None):
        """Select X (layer=None) or a layer already holding a Store.

        This convenience imports no AnnData and never compresses a matrix
        implicitly. H5CP's read_h5(..., lazy=True) preserves these Stores.
        """
        return cls(adata.X if layer is None else adata.layers[layer], ids=ids)

    @property
    def store(self) -> Store:
        return self._store

    @property
    def ids(self) -> np.ndarray:
        return self._ids.view()

    @property
    def shape(self):
        return self.store.shape

    @property
    def dtype(self):
        return self.store.dtype

    def __len__(self):
        return self.shape[0]

    def __reduce__(self):
        return _rebuild, (self.store, self._ids)

    def __repr__(self):
        return (f"ScDataset(shape={self.shape}, dtype={self.dtype}, "
                f"mapped_dims={np.count_nonzero(self._ids != NONE)}, format={self.store.format!r})")

    def _source_entry(self):
        file = self.store.file
        if not file.is_current():
            raise OSError("the dataset Store's file was replaced; open the current stream explicitly")
        source = Store(file)
        if not self.store.selects_all_cells:
            source = source[self.store.cells]
        genes = self.store.genes
        if genes is None:
            mapping = self._ids
        else:
            mapping = np.full(file.shape[1], NONE, dtype=np.uint32)
            keep = (genes >= 0) & (self._ids != NONE)
            mapping[genes[keep]] = self._ids[keep]
        return source, mapping


__all__ = ["ScDataset"]


def _rebuild(store, ids):
    return ScDataset(store, ids=ids)
