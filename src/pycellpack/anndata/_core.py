"""AnnData with CellPack stores as native, lazily read matrix values.

The subclass keeps AnnData's in-memory semantics and only redirects the paths
that would otherwise coerce a matrix to NumPy or rebuild a plain AnnData: value
validation in the aligned mappings, ``X``, views, copies, ``raw`` and the
conversions that need decoded arrays. Stores are immutable, so views and copies
share them; ``to_memory()`` decodes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from anndata._core import aligned_mapping as _am
from anndata._core.file_backing import to_memory as _to_memory
from anndata._core.index import _subset
from anndata._core.raw import Raw as _Raw
from anndata._core.raw import _RawViewHack
from anndata._core.storage import _non_2d_message
from anndata._warnings import ImplicitModificationWarning, warn
from anndata.utils import iter_outer

from ..cellpack import File, Store

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["AnnData", "Raw", "as_store", "iter_stores"]

_NO_X = "no_set_X"


def as_store(value):
    """Wrap an opened CellPack stream as a Store; other values pass through."""
    return Store(value) if isinstance(value, File) else value


# AnnData's mapping base classes validate values with their own shape rules;
# _StoreValues extends those rules to Stores.
_MappingBase = _am.AlignedMappingBase if TYPE_CHECKING else object


class _StoreValues(_MappingBase):
    """Accept Stores in an aligned mapping after checking their aligned axes."""

    def _validate_value(self, val, key) -> Any:
        """Check a Store's axes against the mapping; defer other values."""
        # A Store is checked against the mapping's axes only; AnnData's own
        # validation would first coerce it to an array.
        val = as_store(val)
        if not isinstance(val, Store):
            return super()._validate_value(val, key)
        for i, axis in enumerate(self.axes):
            if val.shape[i] != self.parent.shape[axis]:
                dims = tuple(("obs", "var")[a] for a in self.axes)
                expected = tuple(self.parent.shape[a] for a in self.axes)
                msg = (
                    f"Value passed for key {key!r} is of incorrect shape. "
                    f"Values of {self.attrname} must match dimensions {dims} of "
                    f"parent. Value had shape {val.shape[: len(self.axes)]} while "
                    f"it should have had {expected}."
                )
                raise ValueError(msg)
        if self.attrname == "layers" and key is not None and (
            msg := _non_2d_message(val, name=f"Layer {key!r}")
        ):
            warn(msg, UserWarning)
        return val


class Layers(_StoreValues, _am.Layers):
    """``layers`` mapping whose values may be Stores."""


class LayersView(_StoreValues, _am.LayersView):
    """View of a :class:`Layers` mapping; Stores stay shared."""


class AxisArrays(_StoreValues, _am.AxisArrays):
    """``obsm``/``varm`` mapping whose values may be Stores."""


class AxisArraysView(_StoreValues, _am.AxisArraysView):
    """View of an :class:`AxisArrays` mapping; Stores stay shared."""


class PairwiseArrays(_StoreValues, _am.PairwiseArrays):
    """``obsp``/``varp`` mapping whose values may be Stores."""


class PairwiseArraysView(_StoreValues, _am.PairwiseArraysView):
    """View of a :class:`PairwiseArrays` mapping; Stores stay shared."""


# AnnData dispatches on the actual/view distinction, which needs each view
# class to name its actual class even when both accept Stores.
for _actual, _view in (
    (Layers, LayersView),
    (AxisArrays, AxisArraysView),
    (PairwiseArrays, PairwiseArraysView),
):
    for _cls in (_actual, _view):
        _cls._actual_class = _actual
        _cls._view_class = _view


class Raw(_Raw):
    """Raw whose ``X`` and ``varm`` may hold Stores."""

    # Upstream descriptors constrain values to AnnData's registered array union;
    # this adapter admits Store values through its own shape validator.
    varm: Any = _am.AlignedMappingProperty(AxisArrays, 1)
    _adata: Any
    _varm: Any

    def __init__(self, adata, X: Any = None, var: Any = None, varm: Any = None):
        super().__init__(adata, X=X, var=var, varm=varm)

    def __getitem__(self, index) -> Any:
        """Subset raw data; a Store ``X`` composes the selection instead of reading."""
        from anndata.acc import AdRef

        if isinstance(index, AdRef):
            return super().__getitem__(index)
        if (
            isinstance(index, tuple)
            and len(index) == 2
            and isinstance(index[0], ad.AnnData)
        ):
            adata, index = index
            oidx, vidx = self._normalize_indices(index)
        else:
            oidx, vidx = self._normalize_indices(index)
            adata = self._adata[oidx]
        if isinstance(vidx, (int, np.integer)):
            vidx = slice(vidx, vidx + 1, 1)
        if isinstance(oidx, (int, np.integer)):
            oidx = slice(oidx, oidx + 1, 1)
        X = _subset(self.X, (oidx, vidx)) if not self._adata.isbacked else None
        new = Raw(adata, X=X, var=cast(pd.DataFrame, self._var).iloc[vidx])
        if self.varm is not None:
            new.varm = self.varm._view(_RawViewHack(self, cast(Any, vidx)), (vidx,)).copy()
        return new

    def copy(self) -> Raw:
        """Copy the annotations; a Store ``X`` is shared, since it is immutable."""
        return Raw(
            self._adata,
            X=cast(Any, self.X).copy(),
            var=self.var.copy(),
            varm=None if self._varm is None else self._varm.copy(),
        )

    def to_adata(self) -> AnnData:
        """Materialize raw data as an AnnData; Stores stay lazy."""
        return AnnData(
            X=None if self.X is None else cast(Any, self.X).copy(),
            var=self.var.copy(),
            varm=None if self._varm is None else self._varm.copy(),
            obs=self._adata.obs.copy(),
            obsm=self._adata.obsm.copy(),
            obsp=self._adata.obsp.copy(),
            uns=self._adata.uns.copy(),
        )


def _adopt_raw(raw):
    """Retag the base ``Raw`` AnnData built, so it gains this module's API."""
    # Backed AnnData builds the base Raw by name; the subclass adds only
    # methods and a descriptor, so retagging it is enough.
    if raw is not None and type(raw) is _Raw:
        raw.__class__ = Raw
    return raw


def _build_raw(adata: AnnData, raw) -> Raw | None:
    """Build this module's Raw from an AnnData, a mapping of fields, or a Raw."""
    if not raw:
        return None
    if isinstance(raw, Mapping):
        return Raw(adata, **raw)
    return Raw(adata, raw.X, raw.var, raw.varm)


class AnnData(ad.AnnData):
    """AnnData whose matrix slots can hold lazy CellPack Stores.

    ``X``, ``layers``, ``obsm``, ``varm``, ``obsp`` and ``varp`` accept
    :class:`~pycellpack.cellpack.Store` (or an opened
    :class:`~pycellpack.cellpack.File`) next to the usual AnnData arrays;
    ``raw`` takes the Store form. Slicing the object composes the stores'
    selections without reading; :meth:`to_memory` decodes them. H5CP files
    written from Stores embed the streams, and lazy reads return Stores over
    the embedded streams.
    """

    layers: Any = _am.AlignedMappingProperty(Layers)
    obsm: Any = _am.AlignedMappingProperty(AxisArrays, 0)
    varm: Any = _am.AlignedMappingProperty(AxisArrays, 1)
    obsp: Any = _am.AlignedMappingProperty(PairwiseArrays, 0)
    varp: Any = _am.AlignedMappingProperty(PairwiseArrays, 1)

    def _init_as_actual(
        self, X=None, *, layers=None, shape=None, raw=None, filename=None, **kwargs
    ):
        """Initialize from matrices and annotations, accepting a Store as ``X``.

        The base initializer takes ``X`` as ``layers[None]``; a Store is set
        afterwards, with its own shape, because the base would need a
        materialized matrix. ``raw`` is built here as this module's Raw.
        """
        if filename is not None:
            super()._init_as_actual(
                X, layers=layers, shape=shape, raw=raw, filename=filename, **kwargs
            )
            return
        if isinstance(X, ad.AnnData):
            if any(kwargs.get(k) is not None for k in
                   ("obs", "var", "uns", "obsm", "varm", "obsp", "varp")):
                msg = "If `X` is a dict no further arguments must be provided."
                raise ValueError(msg)
            source = X
            kwargs.update(
                obs=source.obs, var=source.var, uns=source.uns,
                obsm=source.obsm, varm=source.varm,
                obsp=source.obsp, varp=source.varp,
            )
            # X travels as layers[None]; views build it anew on each access.
            layers, X, raw = source.layers, None, source.raw
        if layers is not None and None in layers:
            if X is not None and X is not layers[None]:
                msg = "If you provide `layers[None]` and `X`, they must be identical."
                raise ValueError(msg)
            X = layers[None]
            layers = {k: v for k, v in layers.items() if k is not None}
        store = as_store(X)
        if isinstance(store, Store):
            if shape is not None:
                msg = "`shape` needs to be `None` if `X` is not `None`."
                raise ValueError(msg)
            X, shape = None, store.shape[:2]
        # The base initializer validates raw with the base Raw; build it here.
        super()._init_as_actual(X, layers=layers, shape=shape, **kwargs)
        if isinstance(store, Store):
            self.X = store
        self._raw = _build_raw(self, raw)

    @property
    def X(self):
        """Data matrix of shape ``n_obs × n_vars``; may be a lazy Store."""
        return cast(Any, ad.AnnData.X).fget(self)

    @X.setter
    def X(self, value) -> None:
        """Assign the data matrix; a Store stays lazy and is shape-checked."""
        value = as_store(value)
        if not isinstance(value, Store):
            cast(Any, ad.AnnData.X).fset(self, value)
            return
        if value.shape[:2] != self.shape:
            msg = f"Data matrix has wrong shape {value.shape}, need to be {self.shape}."
            raise ValueError(msg)
        if (msg := _non_2d_message(value, name="X")) is not None:
            warn(msg, UserWarning)
        if self.is_view:
            msg = "Setting element `.X` of view, initializing view as actual."
            warn(msg, ImplicitModificationWarning)
            self._init_as_actual(self._copy(X=value))
            return
        self.layers[None] = value

    @X.deleter
    def X(self) -> None:
        """Clear the data matrix."""
        self.X = None

    # AnnData's getter annotation omits the valid unset (None) state.
    @property
    def raw(self) -> Raw | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Raw data, or ``None``; a base-class Raw is retagged on access."""
        return _adopt_raw(self._raw)

    @raw.setter
    def raw(self, value) -> None:
        """Set raw data from an AnnData; a view becomes actual first."""
        if value is None:
            del self.raw
            return
        if not isinstance(value, ad.AnnData):
            msg = "Can only init raw attribute with an AnnData object."
            raise ValueError(msg)  # noqa: TRY004 - matches anndata
        raw = Raw(self, X=value.X, var=value.var, varm=value.varm)
        if self.is_view:
            self._init_as_actual(self.copy())
        self._raw = raw

    @raw.deleter
    def raw(self) -> None:
        """Clear raw data."""
        cast(Any, ad.AnnData.raw).fdel(self)

    def __getitem__(self, index) -> Any:
        """Slice into a view; Store slots compose their selections."""
        from anndata.acc import AdRef, MapAcc, RefAcc

        if isinstance(index, (AdRef, MapAcc, RefAcc)):
            return super().__getitem__(index)
        oidx, vidx = self._normalize_indices(index)
        return type(self)(self, oidx=oidx, vidx=vidx, asview=True)

    def _copy(self, *, X=_NO_X) -> AnnData:
        """Copy annotations and matrices; Stores are shared, not decoded.

        ``X`` replaces the ``layers[None]`` matrix; the string sentinel keeps
        the current one.
        """
        new: dict[str, Any] = {}
        for key, elem in iter_outer(self):
            new[key] = None if elem is None else cast(Any, elem).copy()
            if key == "layers" and not (isinstance(X, str) and X == _NO_X):
                new[key][None] = X
        new["uns"] = deepcopy(self._uns)
        if self.raw is not None:
            new["raw"] = self.raw.copy()
        return type(self)(**new)

    def to_memory(self, *, copy: bool = False) -> AnnData:
        """Return an AnnData with every Store (and backed array) decoded."""
        new: dict[str, Any] = {}
        for name, elem in iter_outer(self):
            if elem is None:
                continue
            if name == "raw":
                elem = cast(Raw, elem)
                new["raw"] = {
                    "X": _to_memory(elem.X, copy=copy),
                    "var": _to_memory(elem.var, copy=copy),
                    "varm": _to_memory(elem.varm, copy=copy),
                }
            else:
                new[name] = _to_memory(elem, copy=copy)
        return type(self)(**new)

    @property
    def is_lazy(self) -> bool:
        """True when any matrix slot (including raw) still holds a Store."""
        return next(iter_stores(self), None) is not None

    def transpose(self) -> AnnData:
        """Transpose, decoding Stores first since streams are cell-major."""
        if self.is_lazy:
            return self.to_memory().transpose()
        return type(self)(super().transpose())

    T = property(transpose)

    def to_df(self, layer: str | None = None) -> pd.DataFrame:
        """Data frame of ``X`` (or a layer); a Store is decoded first."""
        matrix = self.layers[layer] if layer is not None else self.X
        if isinstance(matrix, Store):
            return pd.DataFrame(
                matrix.toarray(), index=self.obs_names, columns=self.var_names
            )
        return super().to_df(layer)

    def chunked_X(self, chunk_size: int | None = None):
        """Yield ``(chunk, start, end)`` for ``X``, decoding Store chunks."""
        for chunk, start, end in super().chunked_X(chunk_size):
            yield (chunk.to_memory() if isinstance(chunk, Store) else chunk), start, end

    def chunk_X(self, select=1000, *, replace: bool = True):
        """Dense sample of ``X``; a Store contributes only the sampled cells."""
        if not isinstance(self.X, Store):
            return super().chunk_X(select, replace=replace)
        if isinstance(select, int):
            select = np.random.choice(self.n_obs, min(select, self.n_obs), replace)
        return self.X[select].toarray()

    def write_h5cp(self, filename: PathLike[str] | str | None = None, **kwargs: Any) -> None:
        """Write H5CP, embedding Stores as CellPack streams.

        Arguments follow :func:`pycellpack.io.write_h5cp`; ``filename``
        defaults to the backing file.
        """
        from ..io._anndata import write_h5cp

        write_h5cp(self._target(filename), self, **kwargs)
        self._retarget(filename)

    def write_h5ad(self, filename: PathLike[str] | str | None = None, **kwargs: Any) -> None:
        """Write a plain H5AD as :meth:`anndata.AnnData.write_h5ad` does,
        decoding Stores (arguments follow :func:`pycellpack.io.write_h5ad`)."""
        from ..io._anndata import write_h5ad

        write_h5ad(self._target(filename), self, **kwargs)
        self._retarget(filename)

    def _target(self, filename) -> Path:
        if filename is None and not self.isbacked:
            msg = "Provide a filename!"
            raise ValueError(msg)
        return Path(cast(PathLike[str] | str, self.filename if filename is None else filename))

    def _retarget(self, filename) -> None:
        # A backed object now points at the complete new copy.
        if filename is not None and self.isbacked and not self.is_view:
            self.file.filename = filename


def iter_stores(adata: ad.AnnData) -> Iterator[tuple[str, Store]]:
    """Yield ``(element path, Store)`` for every Store-valued matrix slot.

    Paths use H5AD names: ``X``, ``layers/<key>``, ``obsm/<key>``,
    ``raw/X``, ``raw/varm/<key>`` and so on, matching what the writers and
    :func:`pycellpack.io.open_matrix` address.
    """
    if (x := adata.X) is not None and isinstance(x, Store):
        yield "X", x
    for attr in ("layers", "obsm", "varm", "obsp", "varp"):
        for key, value in getattr(adata, attr).items():
            if key is not None and isinstance(value, Store):
                yield f"{attr}/{key}", value
    raw = cast(Raw | None, adata.raw)
    if raw is not None:
        if isinstance(raw.X, Store):
            yield "raw/X", raw.X
        for key, value in raw.varm.items():
            if isinstance(value, Store):
                yield f"raw/varm/{key}", value

