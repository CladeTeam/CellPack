"""Access sequences as the native loader takes them.

One access is one (position, schema) pair and loads one cell, so the loader
needs a cell id per access rather than a list of cells. Callers give either an
`Accesses` (computed or stored, possibly far larger than memory) or a plain
sequence of global cell ids. `sequence` is the single conversion point: every
entry point that takes accesses (`Loader.start`, `compile`, `compile_stats`,
`replay`, `DataSource.stored_cells`, `PageTrace.from_source`) normalizes
through it, so dtype, shape and range are checked once, before a native call.
"""

from __future__ import annotations

import numpy as np

from .. import _cellpack


def sequence(accesses, *, counts=None):
    """An `Accesses` as is, or a one-dimensional integer sequence of cells
    as a contiguous uint32 array (the native side copies it).

    Order and repeats are kept. An array of any integer dtype is accepted
    and cast to uint32, so cell ids must lie in ``[0, 2**32)``. Raises
    TypeError for anything but an `Accesses` or a one-dimensional integer
    array, ValueError for a cell id outside uint32.
    """
    from .access_fn import AccessFn

    if isinstance(accesses, AccessFn):
        return accesses.bind(counts)
    if isinstance(accesses, _cellpack.Accesses):
        return accesses
    values = np.asarray(accesses)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise TypeError("accesses are an Accesses or a one-dimensional integer array of cells")
    if values.dtype == np.uint32:
        return np.ascontiguousarray(values)
    if values.size and (values.min() < 0 or values.max() > 0xFFFFFFFF):
        raise ValueError("access cells must be uint32 values")
    return values.astype(np.uint32)
