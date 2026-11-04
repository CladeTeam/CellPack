"""Lazy Numba signatures for the access generator C ABI."""

from functools import lru_cache
from types import ModuleType
from typing import Any


class Bindings(ModuleType):
    GLOBAL_SIGNATURE: Any
    CORPUS_SIGNATURE: Any


@lru_cache(maxsize=1)
def numba_bindings() -> Bindings:
    try:
        from numba import types
    except ModuleNotFoundError as exc:
        if exc.name != "numba":
            raise
        raise ModuleNotFoundError("Numba access generators require numba; install it separately", name="numba") from exc
    result = Bindings("pycellpack_access_numba")
    common = (types.voidptr, types.uint64, types.uint64, types.CPointer(types.uint32))
    result.GLOBAL_SIGNATURE = types.int32(*common)
    result.CORPUS_SIGNATURE = types.int32(*common, types.CPointer(types.uint64))
    return result
