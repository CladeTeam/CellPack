"""H5CP files: H5AD files whose matrices are embedded, unfiltered CellPack
streams.

A matrix is stored as a group with ``encoding-type`` ``cellpack`` holding one
contiguous uint8 dataset, the stream itself, so the native reader and loader
address it in place. Annotations keep their standard AnnData encodings; streams
hold the matrix only. The layout is H5AD's, so ``.h5cp`` names the files that
embed streams.

:func:`read_h5` reads H5CP and plain H5AD files; :func:`write_h5cp` writes
H5CP. Both mirror :mod:`anndata.io` and defer to AnnData for everything
else; :func:`open_matrix` opens one embedded matrix lazily and
:func:`h5ad_to_cpk` exports one matrix slot to a ``.cpk`` file. Importing this
module also registers the encoding with AnnData's element registry, so
``anndata.read_h5ad`` decodes these files and AnnData's writers accept
:class:`~pycellpack.cellpack.Store` values.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from contextvars import ContextVar
from copy import copy
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import h5py
import numpy as np
import zarr
from anndata._io.specs import IOSpec
from anndata._io.specs.methods import write_anndata, write_raw
from anndata._io.specs.registry import _REGISTRY
from anndata.experimental import read_dispatched, write_dispatched
from scipy import sparse

from ..anndata._core import AnnData, Raw, iter_stores
from ..cellpack import CSR, File, Store, Writer, default_workers
from ..cellpack.scans import EncodeOptions
from ._cpk import write_cpk

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

__all__ = ["h5ad_to_cpk", "open_matrix", "read_h5", "write_h5ad", "write_h5cp"]

_SPEC = IOSpec("cellpack", "0.1.0")
_DTYPES = frozenset(
    np.dtype(t)
    for t in (
        "uint8", "uint16", "uint32", "uint64",
        "int8", "int16", "int32", "int64",
        "float32", "float64",
    )
)
# Set while write_h5ad runs: Stores are decoded instead of embedded. Holds the
# slots written dense (``as_dense``).
_DECODING: ContextVar[frozenset[str] | None] = ContextVar("_DECODING", default=None)
_MATRIX_SLOTS = frozenset({"layers", "obsm", "obsp", "varm", "varp"})
_COPY_BYTES = 8 * 1024 * 1024


# -- element paths and validation -------------------------------------------


def _matrix_slot(name: str) -> bool:
    parts = name.strip("/").split("/")
    return (
        parts == ["X"]
        or parts == ["raw", "X"]
        or (len(parts) == 2 and parts[0] in _MATRIX_SLOTS)
        or (len(parts) == 3 and parts[:2] == ["raw", "varm"])
    )


def _encodable(value) -> bool:
    return (
        (isinstance(value, np.ndarray) or sparse.issparse(value))
        and value.ndim >= 2
        and value.dtype in _DTYPES
        and math.prod(value.shape[1:]) > 0
    )


def _safe_key(key) -> bool:
    return (
        isinstance(key, str)
        and key not in {"", ".", ".."}
        and not any(c in key for c in "/\\\0")
    )


def _validate_keys(mapping, active=None):
    active = set() if active is None else active
    if id(mapping) in active:
        raise ValueError("recursive AnnData metadata mapping")
    active.add(id(mapping))
    try:
        for key, value in mapping.items():
            if not _safe_key(key):
                raise ValueError(f"invalid AnnData mapping key: {key!r}")
            if isinstance(value, Mapping):
                _validate_keys(value, active)
    finally:
        active.remove(id(mapping))


def _preflight(adata):
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData")
    for name in _MATRIX_SLOTS:
        values = getattr(adata, name)
        _validate_keys({k: v for k, v in values.items() if k is not None})
    _validate_keys(adata.uns)
    if adata.raw is not None:
        _validate_keys(cast(Raw, adata.raw).varm)


def _annotation_copy(adata):
    # Categorical conversion mutates frames, so only frames are copied while
    # matrix payloads, including Stores, stay shared.
    working = copy(adata)
    working._obs = adata.obs.copy()
    working._var = adata.var.copy()
    frames = [working.obs, working.var]
    if adata.raw is not None:
        working._raw = copy(adata.raw)
        working._raw._var = adata.raw.var.copy()
        frames.append(working._raw._var)
    for frame in frames:
        working.strings_to_categoricals(frame)
    return working


def _check_layout(chunk_rows: int, page_size: int) -> None:
    if chunk_rows <= 0 or chunk_rows > np.iinfo(np.uint32).max:
        raise ValueError("chunk_rows must lie in [1, 2**32)")
    if page_size < 512 or page_size > 65536 or page_size & (page_size - 1):
        raise ValueError("page_size must be a power of two in [512, 65536]")


# -- stream groups ----------------------------------------------------------


def _embed(parent, key, path: Path, offset: int, length: int, shape, format: str, *, identity=None):
    group = parent.create_group(key)
    group.attrs.update(
        {
            "encoding-type": _SPEC.encoding_type,
            "encoding-version": _SPEC.encoding_version,
            "shape": tuple(int(n) for n in shape),
            "source": format,
        }
    )
    data = group.create_dataset(
        "data", shape=(length,), dtype=np.uint8, chunks=None, compression=None
    )
    # Bounded copies keep the stream out of Python's matrix-sized allocations.
    with open(path, "rb") as stream:
        if identity is not None:
            status = os.fstat(stream.fileno())
            if (status.st_dev, status.st_ino) != identity:
                raise OSError("CellPack source path was replaced before export")
        stream.seek(offset)
        position = 0
        while position < length:
            block = stream.read(min(_COPY_BYTES, length - position))
            if not block:
                raise OSError("CellPack stream ended before its declared size")
            data[position : position + len(block)] = np.frombuffer(block, np.uint8)
            position += len(block)


def _encode_batches(path: Path, batches, *, width, dtype, format, chunk_rows,
                    page_size, workers=None):
    with Writer(path, width, dtype, format=format, chunk_rows=chunk_rows,
                page_size=page_size, workers=workers) as writer:
        for batch in batches:
            writer.append(batch)


def _write_array(parent, key, matrix, scratch: Path, chunk_rows: int, page_size: int,
                 workers=None):
    shape = tuple(int(n) for n in matrix.shape)
    if sparse.issparse(matrix):
        format, payload = "csr", matrix
    else:
        format = "dense"
        payload = matrix.reshape(shape[0], math.prod(shape[1:]))
    encoded = scratch / "matrix.cpk"
    # One append lets the writer encode all whole chunks concurrently.
    _encode_batches(
        encoded, [payload], width=math.prod(shape[1:]), dtype=matrix.dtype,
        format=format, chunk_rows=chunk_rows, page_size=page_size, workers=workers,
    )
    _embed(parent, key, encoded, 0, encoded.stat().st_size, shape, format)
    encoded.unlink()


def _store_batches(store: Store, rows: int, workers):
    for batch in store.iter_batches(rows, workers=workers):
        yield batch.reshape(batch.shape[0], -1) if isinstance(batch, np.ndarray) else batch


def _write_store(parent, key, store: Store, *, chunk_rows=4096, page_size=4096,
                 scratch: Path | None = None, workers=None):
    # Whole streams are copied by path, so the path must still name the
    # opened file; otherwise the view is decoded through its own handle.
    if store.is_identity and store.file.is_current():
        file = store.file
        length = file.length
        if length is None:
            length = file.path.stat().st_size - file.offset
        _embed(parent, key, file.path, file.offset, length, store.shape, store.format,
               identity=file.identity)
        return
    if math.prod(store.shape[1:]) == 0:
        from anndata.io import write_elem

        write_elem(parent, key, store.to_memory())
        return
    workers = default_workers() if workers is None else workers
    with tempfile.TemporaryDirectory(
        prefix=".pycellpack-", dir=scratch or Path(parent.file.filename).parent
    ) as directory:
        encoded = Path(directory) / "selection.cpk"
        # Batches of one chunk per worker keep decode and encode both parallel.
        _encode_batches(
            encoded, _store_batches(store, chunk_rows * workers, workers),
            width=math.prod(store.shape[1:]), dtype=store.dtype, format=store.format,
            chunk_rows=chunk_rows, page_size=page_size, workers=workers,
        )
        _embed(parent, key, encoded, 0, encoded.stat().st_size, store.shape, store.format)


_INT32_MAX = np.iinfo(np.int32).max


def _scipy_index_dtype(store: Store, batches) -> np.dtype:
    """The index dtype SciPy gives the decoded matrix: int32 unless a
    dimension or the stored count exceeds it."""
    if max(store.shape) > _INT32_MAX:
        return np.dtype(np.int64)
    try:
        bound = store.prepare().capacity_nnz  # exact without a column selection
    except ValueError:  # repeated columns cannot be prepared
        bound = None
    if bound is None or (bound > _INT32_MAX and store.genes is not None):
        bound = sum(len(batch.data) for batch in batches())
    return np.dtype(np.int32 if bound <= _INT32_MAX else np.int64)


def _min_index_dtype(minor: int) -> np.dtype:
    """``settings.write_csr_csc_indices_with_min_possible_dtype``, as in AnnData."""
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if minor <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError("column count exceeds uint64")


def _write_decoded(parent, key, store: Store, *, dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
                   as_dense: bool = False, rows: int = 4096, workers=None):
    """Write a Store as AnnData writes the decoded matrix, one row batch at a time.

    The result matches ``write_elem`` on the SciPy/NumPy matrix the Store
    decodes to, with SciPy's default index dtype; ``as_dense`` follows
    ``write_h5ad(as_dense=...)``.
    """
    from anndata import settings
    from anndata.io import write_elem

    csr = store.format == "csr" and not as_dense
    if math.prod(store.shape) == 0:
        empty = (sparse.csr_array(store.shape, dtype=store.dtype) if csr
                 else np.zeros(store.shape, store.dtype))
        write_elem(parent, key, empty, dataset_kwargs=dataset_kwargs)
        return
    workers = default_workers() if workers is None else workers

    def batches(format=None):
        return store.iter_batches(rows * workers, format=format, workers=workers)

    options = dict(dataset_kwargs)
    if not csr:
        array = parent.create_dataset(key, shape=store.shape, dtype=store.dtype, **options)
        array.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
        start = 0
        for batch in batches("dense"):
            batch = cast(np.ndarray, batch)
            array[start : start + len(batch)] = batch
            start += len(batch)
        return
    n_rows, n_cols = store.shape
    index = _scipy_index_dtype(store, batches)
    indptr_dtype = np.dtype(options.pop("indptr_dtype", index))
    if settings.write_csr_csc_indices_with_min_possible_dtype:
        index = _min_index_dtype(n_cols)
    group = parent.require_group(key)
    group.attrs["shape"] = (n_rows, n_cols)
    group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0"})
    # AnnData writes resizable datasets; the stored count is known only after
    # decoding, so values grow in fixed chunks unless the caller sets them.
    options = {"maxshape": (None,), **options}
    growing = {"chunks": (1 << 20,), **options}
    data = group.create_dataset("data", shape=(0,), dtype=store.dtype, **growing)
    indices = group.create_dataset("indices", shape=(0,), dtype=index, **growing)
    indptr = np.zeros(n_rows + 1, np.int64)
    row = nnz = 0
    for batch in batches():
        batch = cast(CSR, batch)
        count = len(batch.data)
        for target, values in ((data, batch.data), (indices, batch.indices)):
            target.resize((nnz + count,))
            target[nnz : nnz + count] = values
        indptr[row + 1 : row + 1 + batch.shape[0]] = nnz + np.asarray(batch.indptr[1:])
        row += batch.shape[0]
        nnz += count
    group.create_dataset("indptr", data=indptr.astype(indptr_dtype, copy=False), **options)


def _open_group(group) -> Store:
    if (
        group.attrs.get("encoding-type") != _SPEC.encoding_type
        or group.attrs.get("encoding-version") != _SPEC.encoding_version
    ):
        raise ValueError("not a supported CellPack H5CP matrix")
    if set(group) != {"data"}:
        raise ValueError("CellPack matrix group must contain only its data stream")
    data = group["data"]
    if (
        not isinstance(data, h5py.Dataset)
        or data.dtype != np.dtype("uint8")
        or data.ndim != 1
        or data.chunks is not None
        or data.id.get_create_plist().get_nfilters()
        or data.is_virtual
        or data.external
    ):
        raise ValueError("CellPack streams require contiguous, unfiltered uint8 datasets")
    shape_attr = group.attrs.get("shape")
    if (
        not isinstance(shape_attr, np.ndarray)
        or shape_attr.ndim != 1
        or len(shape_attr) < 2
    ):
        raise ValueError("invalid original matrix shape")
    if shape_attr.dtype.kind not in "iu" or np.any(shape_attr < 0):
        raise ValueError("matrix dimensions must be nonnegative integers")
    shape = tuple(int(n) for n in shape_attr)
    offset = data.id.get_offset()
    if offset is None:
        raise ValueError("matrix stream has no allocated file region")
    file = File(group.file.filename, offset=offset, length=data.size)
    if (
        file.shape != (shape[0], math.prod(shape[1:]))
        or file.format != group.attrs.get("source")
        or (len(shape) > 2 and file.format != "dense")
    ):
        raise ValueError("CellPack matrix metadata differs from embedded stream")
    return Store(file, shape=shape)


# -- AnnData registry -------------------------------------------------------


@_REGISTRY.register_read(h5py.Group, _SPEC)
def _read_matrix(group, *, _reader):
    del _reader
    return _open_group(group).to_memory()


@_REGISTRY.register_write(h5py.Group, Store, _SPEC)
def _write_store_elem(f, k, elem: Store, *, _writer, dataset_kwargs: Mapping[str, Any] = MappingProxyType({})):
    del _writer
    as_dense = _DECODING.get()
    if as_dense is None:
        _write_store(f, k, elem)
    else:
        name = f"{f.name.rstrip('/')}/{k}".lstrip("/")
        _write_decoded(f, k, elem, dataset_kwargs=dataset_kwargs, as_dense=name in as_dense)


@_REGISTRY.register_write(zarr.Group, Store, _SPEC)
def _write_store_zarr(f, k, elem: Store, *, _writer, dataset_kwargs: Mapping[str, Any] = MappingProxyType({})):
    # Zarr has no contiguous byte region to address, so the store is decoded.
    _writer.write_elem(f, k, elem.to_memory(), dataset_kwargs=dataset_kwargs)


for _group in (h5py.Group, zarr.Group):
    _REGISTRY.register_write(cast(Any, _group), AnnData, IOSpec("anndata", "0.1.0"))(write_anndata)
    _REGISTRY.register_write(cast(Any, _group), Raw, IOSpec("raw", "0.1.0"))(write_raw)


# -- public entry points ----------------------------------------------------


def _embedded_slots(root: h5py.File):
    def cellpack(node) -> bool:
        return node.attrs.get("encoding-type") == _SPEC.encoding_type

    for name in ("X", "raw/X"):
        if name in root and cellpack(root[name]):
            yield name
    for parent in (*sorted(_MATRIX_SLOTS), "raw/varm"):
        group = root.get(parent)
        if isinstance(group, h5py.Group):
            for key, node in group.items():
                if cellpack(node):
                    yield f"{parent}/{key}"


def read_h5(filename: PathLike[str] | str, backed=None, *, lazy: bool = False, **kwargs):
    """Read an H5CP or H5AD file as a :class:`pycellpack.anndata.AnnData`.

    Files without CellPack streams (plain H5AD) are read by :func:`anndata.io.read_h5ad`
    with ``backed`` and ``kwargs`` and converted (backed objects are returned
    unchanged). Otherwise matrices stored as CellPack streams are decoded, or
    with ``lazy=True`` (or ``backed="r"``) returned as Stores over the
    embedded streams; other elements are read into memory. Stores keep their
    own file descriptors, so the file is closed on return.
    """
    path = Path(filename)
    with h5py.File(path, "r") as root:
        if next(_embedded_slots(root), None) is not None:
            if backed not in (None, False, "r"):
                raise NotImplementedError("CellPack streams are read-only; use lazy=True")
            if kwargs:
                raise NotImplementedError(
                    f"{sorted(kwargs)} are unsupported for files with CellPack streams"
                )
            return _read_embedded(root, lazy or backed == "r")
    adata = ad.io.read_h5ad(path, backed=backed, **kwargs)
    return adata if adata.isbacked else AnnData(adata)


def _read_embedded(root: h5py.File, lazy: bool) -> AnnData:
    def callback(read_func, elem_name: str, elem, iospec: IOSpec):
        if iospec.encoding_type == _SPEC.encoding_type:
            store = _open_group(elem)
            return store if lazy else store.to_memory()
        if iospec.encoding_type in ("anndata", "raw"):
            parts: dict[str, Any] = {
                k: read_dispatched(elem[k], callback)
                for k in elem
                if not k.startswith("raw.")
            }
            return parts if iospec.encoding_type == "raw" else AnnData(**parts)
        return read_func(elem)

    return cast(AnnData, read_dispatched(root, callback))


def write_h5ad(
    filepath: PathLike[str] | str,
    adata: ad.AnnData,
    *,
    as_dense: Sequence[str] = (),
    convert_strings_to_categoricals: bool = True,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs,
) -> None:
    """Write a plain H5AD exactly as :func:`anndata.io.write_h5ad` does.

    Arguments and behavior are AnnData's. Stores are written as AnnData writes
    the matrices they decode to (SciPy's default index dtype), one row batch
    at a time, and the file is staged next to the target so a Store may be
    written over the file it reads from.
    """
    if not any(True for _ in iter_stores(adata)):
        ad.io.write_h5ad(
            filepath, adata, as_dense=as_dense,
            convert_strings_to_categoricals=convert_strings_to_categoricals,
            dataset_kwargs=dataset_kwargs, **kwargs,
        )
        return
    dense = [as_dense] if isinstance(as_dense, str) else list(as_dense)
    dense = frozenset("raw/X" if name == "raw.X" else name for name in dense)
    target = Path(filepath).resolve()
    with tempfile.TemporaryDirectory(prefix=".pycellpack-", dir=target.parent) as scratch:
        staged = Path(scratch) / "data.h5ad"
        token = _DECODING.set(dense)
        try:
            ad.io.write_h5ad(
                staged, adata, as_dense=as_dense,
                convert_strings_to_categoricals=convert_strings_to_categoricals,
                dataset_kwargs=dataset_kwargs, **kwargs,
            )
        finally:
            _DECODING.reset(token)
        os.replace(staged, target)


def write_h5cp(
    filepath: PathLike[str] | str,
    adata: ad.AnnData,
    *,
    as_dense: Sequence[str] = (),
    convert_strings_to_categoricals: bool = True,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
    encode: bool | None = None,
    chunk_rows: int = 4096,
    page_size: int = 4096,
    workers: int | None = None,
    **kwargs,
) -> Path:
    """Write H5CP, embedding CellPack streams; argument order follows AnnData.

    ``encode`` selects how matrices in ``X``, ``layers``, ``obs[mp]``,
    ``var[mp]`` and ``raw`` are stored:

    - ``None``: Stores are embedded as CellPack streams and other values use
      AnnData's encodings. Objects without Stores are written by
      :func:`anndata.io.write_h5ad` unchanged.
    - ``True``: numeric NumPy/SciPy matrices are additionally encoded.
    - ``False``: Stores are decoded as in :func:`write_h5ad`, producing a plain H5AD.

    Whole Stores are copied byte for byte; sliced Stores are re-encoded with
    ``chunk_rows`` and ``page_size``, as are encoded in-memory matrices,
    using ``workers`` threads (default: one per CPU, at most 32).
    Dense/CSR form, dtype and higher-rank shapes are retained. ``kwargs`` and
    ``dataset_kwargs`` (e.g. ``compression``) apply to AnnData's datasets only.
    The file is staged next to the target and replaces it only on success, so
    a Store may be written over the file it reads from.
    """
    target = Path(filepath).resolve()
    stores = list(iter_stores(adata))
    if not stores and not encode:
        ad.io.write_h5ad(
            target, adata, as_dense=as_dense,
            convert_strings_to_categoricals=convert_strings_to_categoricals,
            dataset_kwargs=dataset_kwargs, **kwargs,
        )
        return target
    if as_dense:
        raise NotImplementedError("as_dense is unsupported when writing CellPack streams")
    _check_layout(chunk_rows, page_size)
    _preflight(adata)
    working = _annotation_copy(adata) if convert_strings_to_categoricals else adata
    options = {
        k: v for k, v in {**dataset_kwargs, **kwargs}.items() if v is not None
    }
    # Align streams at their largest page so direct I/O can address them.
    alignment = max([page_size, *(s.file.page_size for _, s in stores)])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pycellpack-", dir=target.parent) as scratch:
        scratch = Path(scratch)
        staged = scratch / "data.h5cp"
        with h5py.File(
            staged, "w", alignment_threshold=4096, alignment_interval=alignment
        ) as root:

            def callback(write_func, parent, key, elem, *, dataset_kwargs, iospec):
                del iospec
                name = f"{parent.name.rstrip('/')}/{key.lstrip('/')}"
                if isinstance(elem, Store):
                    if encode is False:
                        _write_decoded(parent, key, elem, dataset_kwargs=dataset_kwargs,
                                       rows=chunk_rows, workers=workers)
                    else:
                        _write_store(parent, key, elem, chunk_rows=chunk_rows,
                                     page_size=page_size, scratch=scratch,
                                     workers=workers)
                elif encode and _matrix_slot(name) and _encodable(elem):
                    _write_array(parent, key, elem, scratch, chunk_rows, page_size,
                                 workers)
                else:
                    write_func(parent, key, elem, dataset_kwargs=dataset_kwargs)

            write_dispatched(root, "/", working, callback=callback,
                             dataset_kwargs=options)
        os.replace(staged, target)
    return target


def h5ad_to_cpk(
    filename: PathLike[str] | str,
    path: PathLike[str] | str,
    key: str = "X",
    *,
    batch_rows: int = 65536,
    format: str | None = None,
    workers: int | None = None,
    chunk_rows: int = 4096,
    page_size: int = 4096,
    options: EncodeOptions | None = None,
    atomic: bool = False,
) -> Store:
    """Export one matrix slot of an H5AD or H5CP file to a ``.cpk`` file.

    ``key`` names an AnnData matrix slot (``"X"``, ``"layers/counts"``,
    ``"raw/X"``, ...). A dense dataset or a CSR group is read and encoded
    ``batch_rows`` rows at a time, so the matrix need not fit in memory; the
    remaining arguments are :func:`~pycellpack.io.write_cpk`'s. An embedded
    CellPack stream is copied byte for byte. Returns a Store over the new
    file; the file keeps no annotations, so cell and gene names stay in the
    source.
    """
    parts = key.strip("/").split("/")
    if not _matrix_slot(key) or not all(_safe_key(p) for p in parts):
        raise ValueError("key must name an AnnData matrix slot")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    with h5py.File(filename, "r") as root:
        node = root["/".join(parts)]
        encoding = node.attrs.get("encoding-type")
        if encoding == _SPEC.encoding_type:
            return _copy_stream(_open_group(node).file, Path(path))
        if isinstance(node, h5py.Dataset):
            if node.ndim != 2:
                raise ValueError("only two-dimensional matrices export to .cpk")
            rows = node.shape[0]
            batches = (node[start : start + batch_rows] for start in range(0, rows, batch_rows))
        elif encoding == "csr_matrix":
            matrix = ad.io.sparse_dataset(cast(h5py.Group, node))
            rows = matrix.shape[0]
            batches = (matrix[start : start + batch_rows] for start in range(0, rows, batch_rows))
        else:
            raise NotImplementedError(f"cannot export {encoding or 'this'} element by rows; "
                                      "dense datasets and CSR matrices are supported")
        return write_cpk(path, batches, format=format, workers=workers, chunk_rows=chunk_rows,
                         page_size=page_size, options=options, atomic=atomic)


def _copy_stream(file: File, path: Path) -> Store:
    """Copy a stream region to its own file, ``_COPY_BYTES`` at a time."""
    remaining = file.length
    with open(file.path, "rb") as source, open(path, "wb") as target:
        source.seek(file.offset)
        while remaining is None or remaining > 0:
            block = source.read(_COPY_BYTES if remaining is None else min(_COPY_BYTES, remaining))
            if not block:
                break
            target.write(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining:
        raise OSError("stream ended before its region")
    return Store(path)


def open_matrix(filename: PathLike[str] | str, key: str = "X") -> Store:
    """Open one embedded matrix lazily, for reads or as a loader source.

    Higher-rank arrays keep their logical shape (their stream stores
    flattened columns). The Store keeps its own file descriptor after the
    HDF5 metadata handle closes.
    """
    parts = key.strip("/").split("/")
    if not _matrix_slot(key) or not all(_safe_key(p) for p in parts):
        raise ValueError("key must name an AnnData matrix slot")
    with h5py.File(filename, "r") as root:
        return _open_group(root["/".join(parts)])
