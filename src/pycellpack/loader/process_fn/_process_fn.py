"""Native process functions with explicit identity and ownership.

`ProcessFn` wraps one ``pycellpack.process.v1`` vtable capsule: a CSR and/or a
dense function pointer, the user-data address and the declared identity that
versions them. The library, the code it points at and the user data must
outlive every loader that uses the function, so the binding retains them; the
identity is what tells two configurations of the same code apart in plans and
cost records. The `from_*` constructors keep a recipe, which is what makes a
bound function picklable to another process.
"""

from typing import Any

import numpy as np

from ... import _cellpack
from ._output import OutputLayout

#: Row formats a vtable may declare (see `ProcessFn.row_format`).
ROW_FORMATS = ("auto", "csr", "dense")


class ProcessFn:
    """A provider-owned ``pycellpack.process.v1`` capsule (see
    ``process_fn_cabi.h``): a CSR and/or a dense function turning one
    cell's row, mapped to the schema's output dims and converted to its
    dtype, into the cell's fixed-size output.

    Four ways to build one: ``ProcessFn.from_source`` compiles C/C++ code
    and binds it; ``ProcessFn.from_cfunc`` binds numba ``@cfunc`` functions
    (or compiles Python functions as such); ``ProcessFn.from_library`` binds
    the functions of a library built elsewhere (an MSVC DLL, say); the
    constructor takes a vtable capsule made by hand.

    Row semantics: an output dim the cell's matrix has a column for is
    present (its stored value, or 0 where the record stores none: CSR drops
    zeros, so they are expression values); an output dim with no matching
    column is missing; a column no output dim takes is dropped. CSR rows list
    stored elements only. Dense rows hold 0 at present zeros and the schema's
    `default` at missing dims, so a default no value takes (NaN, the dtype's
    maximum) lets a dense function tell missing dims apart; a CSR function
    can instead keep each dataset's genes in its user data and find the
    cell's dataset from the context's global cell id.

    `row_format` picks the rows delivered: ``"auto"`` (both functions) gives
    CSR records as CSR rows; dense records use internal throughput thresholds
    based on dimensions, dtype conversion, value coding and cache state.
    Cached dense records may compact to CSR once at admission.
    ``"csr"`` or ``"dense"`` always delivers that format and needs
    only that function (the other vtable entry may be NULL).

    The functions are native and thread-safe, write only their cell's output
    and their scratch (nothing is checked), and must compute the same output
    from the same row. `identity` versions the functions and all retained
    configuration; `owner` can retain a shared library, vtable or parameters.
    `partial_rows` declares that the CSR function resolves rows given without
    columns and values.

    A function built by `from_source`, by `from_library` from a library path
    and exported names, or by `from_cfunc` from picklable functions pickles
    as that recipe (the other process rebuilds it, from the build cache for
    sources); one bound from a hand-made capsule or raw addresses names this
    process's memory and does not.
    """

    __slots__ = (
        "__weakref__",
        "_handle",
        "_identity",
        "_input_dtype",
        "_out_dim",
        "_output",
        "_partial_rows",
        "_recipe",
        "_row_format",
        "_template",
    )

    def __init__(self, capsule, *, identity, owner=None, partial_rows=False, row_format="auto", output=None):
        """Binds a hand-made vtable capsule. `identity` must be in
        ``[0, 2**64)`` and versions the function for plans and cost records;
        `owner` retains whatever the capsule points at; `partial_rows` is kept
        only for CSR rows, since only they can arrive without columns and
        values. Optional `output` declares an OutputLayout (or its NumPy
        dtype specification), enabling schema inference and named unpacking."""
        if row_format not in ROW_FORMATS:
            raise ValueError(f"row_format must be one of {ROW_FORMATS}, not {row_format!r}")
        if isinstance(identity, bool) or not isinstance(identity, int) or not 0 <= identity < 1 << 64:
            raise ValueError(f"identity must be an integer in [0, 2**64), not {identity!r}")
        self._identity = identity
        self._partial_rows = partial_rows and row_format != "dense"
        self._row_format = row_format
        if output is not None and not isinstance(output, OutputLayout):
            output = OutputLayout(output)
        self._output = output
        self._input_dtype: np.dtype | None = None
        self._out_dim: int | None = None
        self._template: str | None = None
        self._recipe: tuple[str, dict[str, Any]] | None = None
        self._handle = _cellpack.ProcessFn(capsule, identity, owner, partial_rows, row_format)

    def __repr__(self) -> str:
        from reprlib import repr as short_repr

        template = "" if self._template is None else f"template={self._template!r}, "
        output = "" if self.output is None else (
            f", row_bytes={self.output.row_bytes}, fields={short_repr(self.output.names)}"
        )
        return (f"ProcessFn({template}row_format={self.row_format!r}{output}, "
                f"identity=0x{self.identity:016x})")

    @property
    def output(self) -> OutputLayout | None:
        """Declared output layout, or None for a function with opaque output."""
        return self._output

    @property
    def row_bytes(self) -> int | None:
        """Declared output record size; None when no layout was supplied."""
        return None if self.output is None else self.output.row_bytes

    @property
    def input_dtype(self):
        """Required input dtype for a model template, otherwise None."""
        return self._input_dtype

    @property
    def out_dim(self) -> int | None:
        """Required mapped input width for a model template, otherwise None."""
        return self._out_dim

    def schema(self, out_ids, dtype=None, *, id=None, name=None, default=0, scratch_bytes=0):
        """Build an OutputSchema using this function's declared record size.

        Model templates also bind the input dtype and width of their tables.
        Custom functions default to float32 input. Supply ``output=`` when
        constructing a custom function to use this convenience.
        """
        from .._schema import OutputSchema

        if self.output is None:
            raise ValueError("schema() needs a declared output layout; pass output= when binding the ProcessFn")
        dtype = (self.input_dtype if self.input_dtype is not None else np.float32) if dtype is None else dtype
        return OutputSchema(out_ids, dtype, id=id, name=name, default=default, process=self,
                            scratch_bytes=scratch_bytes)

    def unpack(self, raw, *, copy=False):
        """Named output arrays; views retain the batch lease unless copy=True.

        ``raw`` is a uint8 array, normally ``batch.region(name)``. See
        ``OutputLayout.unpack`` for shapes, strides and ownership.
        For GPU training, transfer the packed array once and construct
        fields on the device; copy=True is for retaining CPU arrays.
        """
        if self.output is None:
            raise ValueError("unpack() needs a declared output layout; pass output= when binding the ProcessFn")
        return self.output.unpack(raw, copy=copy)

    def __reduce__(self):
        """Pickles as its recipe, so the other process rebuilds the function
        from the build cache or from the given callables. A function bound from
        a capsule or raw addresses names this process's memory and raises
        TypeError."""
        if self._recipe is None:
            raise TypeError("this ProcessFn names native memory of this process; build it with from_source, "
                            "from_library (a path and exported names) or from_cfunc to send it to another process")
        factory, kwargs = self._recipe
        return _rebuild, (factory, kwargs)

    @property
    def identity(self):
        """The declared identity: a 64-bit version of the functions and every
        piece of configuration they read."""
        return self._identity

    @property
    def partial_rows(self):
        """Whether the CSR function resolves rows given without columns and
        values (always false for dense rows)."""
        return self._partial_rows

    @property
    def row_format(self):
        """The row format the loader delivers: "auto", "csr" or "dense"."""
        return self._row_format

    @classmethod
    def from_library(cls, library, *, csr=None, dense=None, user_data=None, key=None, identity=None,
                     row_format=None, partial_rows=False, owner=None, output=None):
        """Binds a shared library's process functions, without a hand-made
        vtable or capsule.

        `library` is a path or a loaded `ctypes.CDLL`; `csr` and `dense` name
        the exported functions (or give ctypes functions or addresses). The
        row format defaults to what they cover: both ``"auto"``, one alone
        that format. `user_data` is passed to every call: a ctypes object
        (a Structure, an array) or a numpy array, both kept alive, or a raw
        address the caller keeps valid; `owner` retains anything else it
        points to.

        The identity defaults to a hash of the library file, the function
        names, the row format, `partial_rows` and `key`, which must describe
        what `user_data` holds (numbers, strings, bytes, numpy arrays, ctypes
        objects, or lists/tuples/dicts of them); give `identity` instead to
        set it directly. Raises ProcessLibraryError when the library cannot be
        loaded or lacks a function.

        `output` is an OutputLayout or structured dtype specification for
        the function's fixed output record. It enables `schema()` and
        `unpack()` and must agree with the function's actual writes.
        """
        import ctypes
        import os

        from ._build import bind_library

        capsule, identity, retained, bound_format = bind_library(
            library, csr=csr, dense=dense, user_data=user_data, key=key, identity=identity, row_format=row_format,
            partial_rows=partial_rows, owner=owner)
        fn = cls(capsule, identity=identity, owner=retained, partial_rows=partial_rows, row_format=bound_format,
                 output=output)
        path = library._name if isinstance(library, ctypes.CDLL) else library
        named = all(f is None or isinstance(f, str) for f in (csr, dense))
        if isinstance(path, (str, os.PathLike)) and named and not isinstance(user_data, int) and owner is None:
            fn._recipe = ("from_library", {"library": os.fspath(path), "csr": csr, "dense": dense,
                                          "user_data": user_data, "key": key, "identity": identity,
                                          "row_format": row_format, "partial_rows": partial_rows, "output": fn.output})
        return fn

    @classmethod
    def from_source(cls, sources=(), *, code=None, csr=None, dense=None, user_data=None, key=None, identity=None,
                    row_format=None, partial_rows=False, owner=None, output=None, **build):
        """Compiles C/C++ sources (`sources` files and/or inline `code`) with
        `compile_library` (the remaining keywords: language, name, compiler,
        std, optimize, flags, include_dirs, defines, depends, cache_dir,
        force) and binds them as `from_library` does."""
        from ._build import compile_library

        library = compile_library(sources, code=code, **build)
        fn = cls.from_library(library, csr=csr, dense=dense, user_data=user_data, key=key, identity=identity,
                              row_format=row_format, partial_rows=partial_rows, owner=owner, output=output)
        if fn._recipe is not None:
            fn._recipe = ("from_source", dict(sources=sources, code=code, csr=csr, dense=dense, user_data=user_data, key=key,
                                              identity=identity, row_format=row_format, partial_rows=partial_rows,
                                              output=fn.output, **build))
        return fn

    @classmethod
    def from_cfunc(cls, *, csr=None, dense=None, user_data=None, key=None, identity=None, row_format=None,
                   partial_rows=False, owner=None, output=None):
        """Binds process functions written in Python with numba.

        `csr` and `dense` are numba ``@cfunc`` functions with the signatures
        returned by `pycellpack.loader.process_fn.numba_bindings()` (``CSR_SIGNATURE``,
        ``DENSE_SIGNATURE``; ``int32(voidptr)`` is accepted too), or plain
        Python functions, which are compiled with those signatures here.
        `user_data`, `key`, `identity`, `row_format`, `partial_rows` and
        `owner` are as for `from_library`; the default identity hashes the
        functions' source (or bytecode), the numba version, the row format,
        `partial_rows` and `key`, which must also cover globals or closures
        the functions read. Raises ProcessBuildError when numba cannot
        compile a function, TypeError for a cfunc of another signature.
        """
        from ._build import bind_cfuncs

        capsule, identity, retained, bound_format = bind_cfuncs(
            csr=csr, dense=dense, user_data=user_data, key=key, identity=identity, row_format=row_format,
            partial_rows=partial_rows, owner=owner)
        fn = cls(capsule, identity=identity, owner=retained, partial_rows=partial_rows, row_format=bound_format,
                 output=output)
        if not isinstance(user_data, int) and owner is None:
            fn._recipe = ("from_cfunc", {"csr": csr, "dense": dense, "user_data": user_data, "key": key,
                                        "identity": identity, "row_format": row_format, "partial_rows": partial_rows,
                                        "output": fn.output})
        return fn

    @staticmethod
    def include_dir():
        """The directory holding `process_fn_cabi.h` and the model templates,
        which compiled sources include; `compile_library` puts it first on the
        include path."""
        from pathlib import Path

        return Path(__file__).resolve().parent / "include"


def _rebuild(factory, kwargs):
    """A ProcessFn from its recipe (unpickling)."""
    if factory.startswith("preset:"):
        from . import presets

        return getattr(presets, factory.removeprefix("preset:"))(**kwargs)
    return getattr(ProcessFn, factory)(**kwargs)
