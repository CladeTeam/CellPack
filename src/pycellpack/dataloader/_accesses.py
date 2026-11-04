"""Fixed corpus accesses and conversion to native cell ids."""

from dataclasses import dataclass

import numpy as np

from .. import _cellpack
from ..loader.access_fn import AccessFn


@dataclass
class Accesses:
    """Parallel arrays of corpus dataset indices and dataset-local rows.

    The caller owns ordering, sampling and epoch policy. Arrays are borrowed
    by this value object and snapshotted when a loader iterator starts.
    """

    dataset_ids: np.ndarray
    rows: np.ndarray

    def __len__(self):
        return len(self.rows)


def corpus_counts(corpus):
    counts = tuple(len(dataset) for dataset in corpus)
    if sum(counts) > 0xFFFFFFFF:
        raise ValueError("the corpus exceeds the native uint32 CellId range")
    return counts


def _integer_array(values, name, maximum, dtype):
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy integer array")
    if values.ndim != 1 or values.dtype.kind not in "iu" or (values.size and (values.min() < 0 or values.max() > maximum)):
        raise ValueError(f"{name} must be a one-dimensional integer array in [0, {maximum}]")
    return np.array(values, dtype=dtype, copy=True, order="C")


def bind_accesses(value, counts):
    if isinstance(value, AccessFn):
        return value.bind(counts)
    datasets = _integer_array(value.dataset_ids, "dataset_ids", 0xFFFFFFFF, np.uint32)
    rows = _integer_array(value.rows, "rows", (1 << 64) - 1, np.uint64)
    if datasets.shape != rows.shape:
        raise ValueError("dataset_ids and rows must have the same shape")
    if not len(rows):
        return _cellpack.Accesses.array(np.empty(0, np.uint32))
    counts = np.asarray(counts, dtype=np.uint64)
    if np.any(datasets >= len(counts)):
        raise ValueError("dataset_ids contains an index outside corpus")
    if np.any(rows >= counts[datasets]):
        raise ValueError("rows contains an index outside its dataset view")
    offsets = np.concatenate((np.zeros(1, np.uint64), counts.cumsum()))
    return _cellpack.Accesses.array((offsets[datasets] + rows).astype(np.uint32))


__all__ = ["Accesses"]
