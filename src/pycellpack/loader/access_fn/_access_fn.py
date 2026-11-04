"""Owned native access functions, rebuilt from values when pickled."""

from __future__ import annotations

import hashlib
import operator
import os
from pathlib import Path
from typing import Any

from ... import _cellpack
from ..process_fn._build import (
    _feed,
    _function,
    _load,
    _python_code,
    _user_data,
    compile_library,
)


def include_dir() -> Path:
    """Directory containing access_fn_cabi.h."""
    return Path(__file__).resolve().parent / "include"


def _length(value):
    if isinstance(value, bool):
        raise TypeError("length must be an integer")
    value = operator.index(value)
    if not 0 <= value < 1 << 64:
        raise ValueError("length must be in [0, 2**64)")
    return value


def _identity(code, kind, key, identity, user_data):
    if identity is not None:
        if key is not None:
            raise ValueError("give identity= or key=, not both")
        return _length(identity)
    if user_data is not None and key is None:
        raise ValueError("user_data requires key= or identity= covering its configuration")
    digest = hashlib.sha256()
    _feed(digest, (code, kind, key))
    return int.from_bytes(digest.digest()[:8], "little")


class AccessFn:
    """A deterministic range function with global or corpus addressing.

    ``kind='global'`` implements CellPackAccessFn; ``kind='corpus'``
    implements CellPackCorpusAccessFn and fills dataset ids and local rows.
    Functions may be called concurrently, repeatedly and out of order by
    online compilation and loading. A corpus function is bound to row
    counts by ScDataLoader, or explicitly with bind(counts).
    """

    def __init__(self, address, context, length, *, kind, identity, owner=None):
        if kind not in ("global", "corpus"):
            raise ValueError("kind must be 'global' or 'corpus'")
        self._address = operator.index(address)
        self._context = operator.index(context)
        if self._address <= 0 or self._context < 0:
            raise ValueError("a generator needs a positive function address and nonnegative context")
        self._length = _length(length)
        self._kind = kind
        self._identity = _length(identity)
        self._owner = owner
        self._recipe: tuple[str, dict[str, Any]] | None = None

    def __len__(self):
        return self._length

    @property
    def kind(self):
        return self._kind

    @property
    def identity(self):
        return self._identity

    def __repr__(self):
        return f"AccessFn(kind={self.kind!r}, length={self._length}, identity=0x{self.identity:016x})"

    def bind(self, counts=None):
        """Build a native Accesses retaining code and parameters for its run.

        Corpus results are translated to the registration order of these
        dataset views. Sampling policy belongs to the supplied function.
        """
        if self.kind == "global":
            return _cellpack.Accesses.function(self._address, self._context, self._length, owner=self._owner)
        if counts is None:
            raise ValueError("a corpus AccessFn needs dataset row counts; use bind(counts) or ScDataLoader")
        counts = tuple(operator.index(n) for n in counts)
        return _cellpack.Accesses.corpus_function(self._address, self._context, self._length, list(counts), owner=self._owner)

    def fill(self, start=0, count=None, *, counts=None, out=None):
        """Generate global cell ids for an interval; primarily for inspection."""
        return self.bind(counts).fill(start, count, out=out)

    @classmethod
    def from_library(cls, library, *, function="generate_accesses", length, kind="global", user_data=None,
                     key=None, identity=None, owner=None):
        """Bind an exported symbol, retaining the library and parameter owners.

        user_data accepts the same ctypes/NumPy values as ProcessFn. key
        must describe pointed-to values, excluding process-local addresses.
        A path, symbol name and picklable values form a reconstruction recipe.
        """
        lib, path = _load(library)
        address = _function(lib, path, "access", function)
        if address is None:
            raise ValueError("function must name an access generator")
        context, retained = _user_data(user_data)
        if identity is None and (path is None or not isinstance(function, str)):
            raise ValueError("binding by address needs an explicit identity=")
        code = None
        if identity is None:
            assert path is not None
            code = (hashlib.sha256(path.read_bytes()).hexdigest(), function)
        identity = _identity(code, kind, key, identity, user_data)
        result = cls(address, context or 0, length, kind=kind, identity=identity, owner=(lib, retained, owner))
        if path is not None and isinstance(function, str) and not isinstance(user_data, int) and owner is None:
            result._recipe = ("from_library", {"library": str(path), "function": function, "length": length,
                                               "kind": kind, "user_data": user_data, "identity": identity})
        return result

    @classmethod
    def from_source(cls, sources=(), *, code=None, function="generate_accesses", length, kind="global",
                    user_data=None, key=None, identity=None, owner=None, **build):
        """Compile C/C++ with the access headers, cache and bind the library."""
        # Recipes keep concrete iterables so another process sees the same build.
        sources = os.fspath(sources) if isinstance(sources, (str, os.PathLike)) else tuple(sources)
        for name in ("flags", "include_dirs", "depends"):
            if name in build:
                value = build[name]
                build[name] = (value,) if isinstance(value, (str, os.PathLike)) else tuple(value)
        headers = tuple(sorted(p for p in include_dir().rglob("*") if p.is_file()))
        options = {**build, "include_dirs": (include_dir(), *build.get("include_dirs", ())),
                   "depends": (*headers, *build.get("depends", ()))}
        library = compile_library(sources, code=code, **options)
        result = cls.from_library(library, function=function, length=length, kind=kind, user_data=user_data,
                                  key=key, identity=identity, owner=owner)
        if result._recipe is not None:
            result._recipe = ("from_source", dict(sources=sources, code=code, function=function, length=length,
                                                  kind=kind, user_data=user_data, identity=result.identity, **build))
        return result

    @classmethod
    def from_cfunc(cls, function, *, length, kind="global", user_data=None, key=None, identity=None, owner=None):
        """Bind a Numba cfunc or compile a Python function with the access ABI."""
        from ._numba import numba_bindings

        bindings = numba_bindings()
        import numba
        from numba.core.ccallback import CFunc

        if kind not in ("global", "corpus"):
            raise ValueError("kind must be 'global' or 'corpus'")
        signature = bindings.GLOBAL_SIGNATURE if kind == "global" else bindings.CORPUS_SIGNATURE
        compiled = function if isinstance(function, CFunc) else numba.cfunc(signature)(function)
        if compiled._sig != signature:
            raise TypeError(f"the {kind} access generator requires {signature}, got {compiled._sig}")
        source = compiled._pyfunc
        context, retained = _user_data(user_data)
        identity = _identity((_python_code(source), numba.__version__), kind, key, identity, user_data)
        result = cls(compiled.address, context or 0, length, kind=kind, identity=identity,
                     owner=(compiled, retained, owner))
        if not isinstance(user_data, int) and owner is None:
            result._recipe = ("from_cfunc", {"function": source, "length": length, "kind": kind,
                                             "user_data": user_data, "identity": identity})
        return result

    @staticmethod
    def include_dir():
        return include_dir()

    def __reduce__(self):
        if self._recipe is None:
            raise TypeError("this AccessFn retains process-local owners; rebuild it from picklable configuration externally")
        return _rebuild, self._recipe


def _rebuild(factory, kwargs):
    return getattr(AccessFn, factory)(**kwargs)
