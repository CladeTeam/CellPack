"""Process functions written in Python and compiled with numba's ``@cfunc``.

The row structs of ``process_fn_cabi.h`` as numba record types, the
signatures a process function's ``@cfunc`` takes, and helpers that view the
row's pointers as arrays inside the compiled function::

    from numba import cfunc
    from pycellpack.loader.process_fn import ProcessFn, numba_bindings

    pn = numba_bindings()

    @cfunc(pn.CSR_SIGNATURE)
    def total(call):
        row = pn.row(call)
        values = pn.array_at(row.values, row.nnz, np.float32)
        out = pn.array_at(row.out, 1, np.float64)
        out[0] = values.sum()
        return 0

    fn = ProcessFn.from_cfunc(csr=total)

Pointer fields (``columns``, ``values``, ``out``, ``scratch``,
``context.user_data``) are addresses (uint64); `row` reads the struct a
function's argument points to and `array_at` turns an address into an array
of `count` elements of `dtype`. Only calling :func:`bindings` needs numba.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from types import ModuleType
from typing import Any, cast

import numpy as np

_U4, _U8 = np.uint32, np.uint64
# The header's row layouts as NumPy dtypes: `align=True` keeps the C offsets,
# and pointer fields are plain uint64 addresses, which numba records can hold
# where C pointers cannot be. Field order is the header's.
CONTEXT_DTYPE = np.dtype([("abi_version", _U4), ("struct_size", _U4), ("position", _U8), ("cell", _U8),
                          ("schema", _U4), ("dtype", _U4), ("user_data", _U8)], align=True)
CSR_ROW_DTYPE = np.dtype([("context", CONTEXT_DTYPE), ("out_dim", _U8), ("nnz", _U8), ("columns", _U8),
                          ("values", _U8), ("row", _U8), ("resolve", _U8), ("out", _U8), ("scratch", _U8),
                          ("scratch_bytes", _U8)], align=True)
DENSE_ROW_DTYPE = np.dtype([("context", CONTEXT_DTYPE), ("out_dim", _U8), ("values", _U8), ("out", _U8),
                            ("scratch", _U8), ("scratch_bytes", _U8)], align=True)


class Bindings(ModuleType):
    """Numba types and helpers in a module namespace usable by compiled functions."""

    CONTEXT_DTYPE: np.dtype
    CSR_ROW_DTYPE: np.dtype
    DENSE_ROW_DTYPE: np.dtype
    CSR_ROW: Any
    DENSE_ROW: Any
    CSR_SIGNATURE: Any
    DENSE_SIGNATURE: Any
    SIGNATURES: dict[str, Any]
    row: Callable[[Any], Any]
    array_at: Callable[[int, int, Any], np.ndarray]


@lru_cache(maxsize=1)
def bindings() -> Bindings:
    """Load optional numba and return cached row types, signatures and JIT helpers.

    Use ``pn = bindings()`` before defining a ``@numba.cfunc`` that accesses
    ``pn.row`` or ``pn.array_at``. Install numba with ``uv pip install numba``.
    """
    try:
        import numba
        from numba import types
        from numba.extending import intrinsic
        from numba.np.numpy_support import carray
    except ModuleNotFoundError as exc:
        if exc.name != "numba":
            raise
        raise ModuleNotFoundError(
            "Numba process functions require numba. Install with: uv pip install numba",
            name=exc.name,
        ) from exc

    @intrinsic
    def pointer(typingctx, address, dtype):
        """Convert an integer address to a pointer to dtype elements."""
        if not isinstance(address, types.Integer):
            return None
        if isinstance(dtype, types.DType):
            element = dtype.dtype
        elif isinstance(dtype, types.NumberClass):
            element = dtype.instance_type
        else:
            return None
        result = types.CPointer(element)

        def codegen(context, builder, signature, args):
            """LLVM IR: reinterpret the address argument as the pointer type."""
            return builder.inttoptr(args[0], context.get_value_type(result))

        return result(address, dtype), codegen

    # Numba supplies the typing context; compiled callers pass only address and dtype.
    address_pointer = cast(Callable[[int, Any], Any], pointer)

    @numba.njit(inline="always")
    def row(call):
        """View the row without copying its nested context record."""
        return carray(call, 1)[0]

    @numba.njit(inline="always")
    def array_at(address, count, dtype):
        """View count elements of dtype at address without copying."""
        return carray(address_pointer(address, dtype), count)

    result = Bindings(f"{__name__}._bindings")
    result.CONTEXT_DTYPE = CONTEXT_DTYPE
    result.CSR_ROW_DTYPE = CSR_ROW_DTYPE
    result.DENSE_ROW_DTYPE = DENSE_ROW_DTYPE
    result.CSR_ROW = numba.from_dtype(CSR_ROW_DTYPE)
    result.DENSE_ROW = numba.from_dtype(DENSE_ROW_DTYPE)
    result.CSR_SIGNATURE = types.int32(types.CPointer(result.CSR_ROW))
    result.DENSE_SIGNATURE = types.int32(types.CPointer(result.DENSE_ROW))
    result.SIGNATURES = {"csr": result.CSR_SIGNATURE, "dense": result.DENSE_SIGNATURE}
    result.row = row
    result.array_at = array_at
    return result


__all__ = ["CONTEXT_DTYPE", "CSR_ROW_DTYPE", "DENSE_ROW_DTYPE", "Bindings", "bindings"]
