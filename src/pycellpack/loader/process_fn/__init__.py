"""Native process functions: the loader's per-row transform of a schema.

A process function turns one cell's row, mapped to the schema's output dims
and converted to its dtype, into the cell's fixed-size output. The ABI is
``include/process_fn_cabi.h``; :class:`ProcessFn` binds functions compiled
from C/C++ (:meth:`ProcessFn.from_source`, :func:`compile_library`), taken
from a library built elsewhere (:meth:`ProcessFn.from_library`) or written in
Python with numba (:meth:`ProcessFn.from_cfunc`, whose signatures and row
helpers :func:`numba_bindings` returns).

``include/template/<model>.hpp`` holds model presets (scGPT, Geneformer,
UCE, STATE SE): C++ templates over a parameter struct, since every training
configuration needs its own statically compiled functions;
``template/common.hpp`` adapts them to the C ABI. Sources compiled with
:func:`compile_library` have ``include/`` on their include path, so they
``#include "template/scgpt.hpp"``.
"""

from pathlib import Path

from . import presets
from ._build import (
    CAPSULE_NAME,
    Compiler,
    CompilerNotFoundError,
    ProcessBuildError,
    ProcessLibraryError,
    compile_library,
    find_compiler,
)
from ._numba import bindings as numba_bindings
from ._output import OutputLayout
from ._process_fn import ROW_FORMATS, ProcessFn


def include_dir() -> Path:
    """The directory holding ``process_fn_cabi.h`` and ``template/``."""
    return Path(__file__).resolve().parent / "include"


__all__ = [
    "CAPSULE_NAME",
    "ROW_FORMATS",
    "Compiler",
    "CompilerNotFoundError",
    "OutputLayout",
    "ProcessBuildError",
    "ProcessFn",
    "ProcessLibraryError",
    "compile_library",
    "find_compiler",
    "include_dir",
    "numba_bindings",
    "presets",
]
