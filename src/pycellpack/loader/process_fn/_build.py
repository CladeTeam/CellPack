"""Building and binding native process functions.

`compile_library` turns C or C++ sources (files or inline code) into a shared
library with the platform's GCC or Clang driver, cached by content;
`bind_library` turns a library's exported functions, and `bind_cfuncs`
numba ``@cfunc`` functions, into the vtable capsule `ProcessFn` takes. MSVC is not driven automatically (its environment comes
from its own activation scripts): build the DLL there and bind it with
`ProcessFn.from_library`.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ... import _cellpack

#: Capsule name of the vtable `ProcessFn` takes; the native side checks it.
CAPSULE_NAME = b"pycellpack.process.v1"
_C_SUFFIXES = {".c"}
_CXX_SUFFIXES = {".cc", ".cpp", ".cxx", ".c++", ".C"}
_STD = {"c": "c11", "c++": "c++17"}
# Compiler driver names tried in order, per platform and language.
_CANDIDATES = {
    "linux": {"c++": ("c++", "g++", "clang++"), "c": ("cc", "gcc", "clang")},
    "darwin": {"c++": ("clang++", "c++", "g++"), "c": ("clang", "cc", "gcc")},
    "win32": {"c++": ("clang++", "g++"), "c": ("clang", "gcc")},
}
# Compiler output kept at the end of a build error: the messages are there.
_STDERR_TAIL = 8000


class ProcessBuildError(RuntimeError):
    """A process library could not be compiled."""


class CompilerNotFoundError(ProcessBuildError):
    """No usable GCC or Clang driver was found."""


class ProcessLibraryError(RuntimeError):
    """A process library could not be loaded or lacks a function."""


@dataclass(frozen=True)
class Compiler:
    """A GCC or Clang compiler driver: `path`, `kind` ("gcc", "clang" or
    "unknown", which is driven like GCC) and the first line of its version
    output."""

    path: str
    kind: str
    version: str


def _platform() -> str:
    """The platform key of `_CANDIDATES` and of the shared library suffix."""
    return "darwin" if sys.platform == "darwin" else "win32" if sys.platform.startswith("win") else "linux"


def _probe(name: str) -> Compiler | None:
    """The driver `name` if PATH has it and it answers `--version`; its kind
    comes from that banner, the only place a driver names itself."""
    path = shutil.which(name)
    if path is None:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr or "").strip()
    if out.returncode != 0 or not text:
        return None
    lower = text.lower()
    kind = "clang" if "clang" in lower else "gcc" if ("gcc" in lower or "free software foundation" in lower) else "unknown"
    return Compiler(path, kind, text.splitlines()[0])


def find_compiler(language: str = "c++", compiler: str | os.PathLike | None = None) -> Compiler:
    """The compiler driver for `language` ("c" or "c++"): `compiler` if
    given, else `$CXX` / `$CC`, else the platform's usual GCC/Clang names."""
    if language not in _STD:
        raise ValueError(f"language must be 'c' or 'c++', not {language!r}")
    env = "CXX" if language == "c++" else "CC"
    if compiler is not None or os.environ.get(env):
        name = str(compiler) if compiler is not None else os.environ[env]
        # $CXX may carry flags ("ccache g++"): its last word is the driver.
        words = shlex.split(name) if compiler is None else [name]
        found = _probe(words[-1]) if words else None
        if found is None:
            origin = "compiler=" if compiler is not None else f"${env}"
            raise CompilerNotFoundError(f"{origin} {name!r} is not a working GCC or Clang driver "
                                        f"(not on PATH, or `--version` failed)")
        if Path(found.path).stem.lower() in ("cl", "clang-cl"):
            raise CompilerNotFoundError(_msvc_message(found.path))
        return found
    tried = _CANDIDATES[_platform()][language]
    for name in tried:
        found = _probe(name)
        if found is not None:
            return found
    if _platform() == "win32" and shutil.which("cl"):
        raise CompilerNotFoundError(_msvc_message(shutil.which("cl")))
    raise CompilerNotFoundError(f"no {language} compiler found (tried ${env}, then {', '.join(tried)} on PATH); "
                                f"install GCC or Clang, or pass compiler=")


def _msvc_message(path) -> str:
    """Why a path that resolves to MSVC's driver cannot be used, and the two
    ways out."""
    return (f"MSVC ({path}) is not driven automatically: build the DLL in an MSVC developer shell "
            "(exporting the functions with __declspec(dllexport)) and bind it with ProcessFn.from_library, "
            "or install Clang or MinGW GCC and set CXX/CC")


def _language(paths, code_language):
    """The one language the sources and inline code share ("c" or "c++").
    Mixing them is refused: one driver invocation compiles all inputs as the
    same language."""
    kinds = set()
    for p in paths:
        if p.suffix in _C_SUFFIXES:
            kinds.add("c")
        elif p.suffix in _CXX_SUFFIXES:
            kinds.add("c++")
        else:
            raise ValueError(f"{p}: unknown source suffix {p.suffix!r} (use .c for C; .cc, .cpp or .cxx for C++)")
    if code_language is not None:
        kinds.add(code_language)
    if len(kinds) > 1:
        raise ValueError("C and C++ sources cannot be built together; compile the C parts as C++ "
                         "(rename them .cpp) or build the library yourself")
    return kinds.pop()


def _library_suffix() -> str:
    """The shared library suffix of this platform."""
    return {"darwin": ".dylib", "win32": ".dll"}.get(_platform(), ".so")


def _cache_dir(cache_dir) -> Path:
    """Where compiled libraries are kept: `cache_dir`, else
    `$CELLPACK_HOME/process-libraries` (`~/.cellpack` when that is unset)."""
    if cache_dir is not None:
        return Path(cache_dir)
    home = Path(os.environ.get("CELLPACK_HOME") or Path.home() / ".cellpack")
    return home / "process-libraries"


def _command(compiler: Compiler, language, std, optimize, include_dirs, defines, flags, sources, output):
    """The driver's command line: one shared library (ELF `-shared`, Mach-O
    `-dynamiclib`) from `sources`, with `-fPIC` where it is not the default,
    then the standard, the optimization flag, include directories, defines
    and flags. `language` only reached the standard picked by the caller."""
    cmd = [compiler.path, "-shared"]
    if _platform() == "darwin":
        cmd = [compiler.path, "-dynamiclib"]
    elif _platform() != "win32":
        cmd.append("-fPIC")
    cmd.append(f"-std={std}")
    if optimize:
        cmd.append(optimize)
    for d in include_dirs:
        cmd += ["-I", str(d)]
    for name, value in (defines or {}).items():
        cmd.append(f"-D{name}" if value is None else f"-D{name}={value}")
    cmd += [str(f) for f in flags]
    cmd += [str(s) for s in sources]
    cmd += ["-o", str(output)]
    return cmd


def compile_library(sources=(), *, code=None, language=None, name=None, compiler=None, std=None, optimize="-O3",
                    flags=(), include_dirs=(), defines=None, depends=(), cache_dir=None, force=False) -> Path:
    """Compiles process function sources into a shared library; returns its path.

    `sources` are C or C++ files (by suffix); `code` adds inline source text
    in `language`, C++ unless `language="c"` (one driver invocation compiles
    every input as one language, so inline code joins C source files only
    when it is declared C). The process headers (`process_fn_cabi.h` and
    the model templates under `template/`) and every source file's directory are on the include
    path, before `include_dirs`.
    `flags` are passed to the compiler as is (say ``["-march=native"]``);
    `defines` maps macro names to values (None: defined without a value).

    The library is cached under `cache_dir` (default
    ``$CELLPACK_HOME/process-libraries``) by everything that shapes it: the
    compiler and its version, the command options, the sources, the
    process headers and the files in `depends` (headers or included sources
    not listed in `sources`). A cached library is reused unless `force`;
    concurrent builds of the same library are safe. Raises
    CompilerNotFoundError when no GCC/Clang driver is usable and
    ProcessBuildError (with the command and the compiler's messages) when
    compilation fails.
    """
    flags = tuple(str(flag) for flag in flags)
    paths = [Path(s) for s in ([sources] if isinstance(sources, (str, os.PathLike)) else list(sources))]
    if not paths and code is None:
        raise ValueError("compile_library needs sources or code")
    if code is not None and not isinstance(code, str):
        raise TypeError("code must be source text (str)")
    if language is not None and language not in _STD:
        raise ValueError(f"language must be 'c' or 'c++', not {language!r}")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"process source not found: {p}")
    deps = [Path(d) for d in ([depends] if isinstance(depends, (str, os.PathLike)) else list(depends))]
    for d in deps:
        if not d.is_file():
            raise FileNotFoundError(f"process dependency not found: {d}")
    code_language = None if code is None else (language or "c++")
    lang = _language(paths, code_language) if paths else (code_language or "c++")
    if language is not None and lang != language:
        raise ValueError(f"language={language!r} but the sources are {lang}")
    driver = find_compiler(lang, compiler)
    std = std or _STD[lang]
    headers = sorted(f for f in _include_dir().rglob("*") if f.is_file())
    includes = [_include_dir(), *dict.fromkeys(p.resolve().parent for p in paths), *map(Path, include_dirs)]

    # Everything that shapes the library enters the digest: the driver and its
    # version, the options, the header, the sources and the declared
    # dependencies. Its first 16 hex digits name the cached file.
    h = hashlib.sha256()
    h.update(json.dumps({"compiler": [driver.path, driver.kind, driver.version], "platform": _platform(),
                         "language": lang, "std": std, "optimize": optimize, "flags": [str(f) for f in flags],
                         "includes": [str(i) for i in includes], "defines": defines or {}}, sort_keys=True,
                        default=str).encode())
    for f in [*headers, *paths, *deps]:
        h.update(str(f.name).encode())
        h.update(f.read_bytes())
    if code is not None:
        h.update(code.encode())
    digest = h.hexdigest()[:16]
    stem = name or (paths[0].stem if paths else "inline")
    root = _cache_dir(cache_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ProcessBuildError(f"cannot create the process library cache {root}: {e}") from e
    target = root / f"{stem}-{digest}{_library_suffix()}"
    if target.exists() and not force:
        return target

    with tempfile.TemporaryDirectory(dir=root, prefix=f".{stem}-") as work:
        work = Path(work)
        units = list(paths)
        if code is not None:
            inline = work / ("inline.c" if lang == "c" else "inline.cpp")
            inline.write_text(code)
            units.append(inline)
        output = work / target.name
        cmd = _command(driver, lang, std, optimize, includes, defines, flags, units, output)
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as e:
            raise ProcessBuildError(f"cannot run {driver.path}: {e}") from e
        if out.returncode != 0 or not output.is_file():
            messages = (out.stderr or "") + (out.stdout or "")
            if len(messages) > _STDERR_TAIL:
                messages = "...\n" + messages[-_STDERR_TAIL:]
            raise ProcessBuildError(f"compiling {stem} failed (exit {out.returncode}) with {driver.kind} "
                                    f"{driver.version}\n$ {shlex.join(cmd)}\n{messages.rstrip()}")
        os.replace(output, target)
    return target


def _include_dir() -> Path:
    """The directory holding `process_fn_cabi.h` and the model templates."""
    return Path(__file__).resolve().parent / "include"


# Binding.


_FUNCTION_POINTER = getattr(ctypes, "_CFuncPtr")  # noqa: B009 - runtime-only ctypes base absent from stubs


class _VTable(ctypes.Structure):
    """`PyCellPackProcessVTable` from `process_fn_cabi.h`, field for field;
    `ctypes` uses the platform's C layout, so the capsule can point the native
    side at this structure."""

    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("csr", ctypes.c_void_p), ("dense", ctypes.c_void_p), ("user_data", ctypes.c_void_p)]


def _load(library):
    """A loaded library and its path (None for a `CDLL` without a file, such
    as the process itself). Raises FileNotFoundError for a missing file and
    ProcessLibraryError when the dynamic loader refuses it."""
    if isinstance(library, ctypes.CDLL):
        return library, Path(library._name) if library._name else None
    path = Path(library)
    if not path.is_file():
        raise FileNotFoundError(f"process library not found: {path}")
    try:
        return ctypes.CDLL(str(path.resolve())), path.resolve()
    except OSError as e:
        raise ProcessLibraryError(f"cannot load process library {path}: {e}") from e


def _exports(path) -> list[str]:
    """Best-effort exported symbol names (for error messages)."""
    nm = shutil.which("nm")
    if nm is None or path is None:
        return []
    flags = ["-gU"] if _platform() == "darwin" else ["-D", "--defined-only"]
    try:
        out = subprocess.run([nm, *flags, str(path)], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = [line.split()[-1].lstrip("_" if _platform() == "darwin" else "") for line in out.stdout.splitlines()
             if len(line.split()) >= 3 and line.split()[1] in ("T", "t")]
    return sorted(set(names))


def _function(lib, path, role, entry):
    """The address of `entry` (a symbol name, a ctypes function or an address)."""
    if entry is None:
        return None
    if isinstance(entry, int) and not isinstance(entry, bool):
        if entry <= 0:
            raise ValueError(f"{role} address must be positive")
        return entry
    if isinstance(entry, str):
        try:
            function = getattr(lib, entry)
        except AttributeError:
            exported = _exports(path)
            near = [n for n in exported if entry.lower() in n.lower() or n.lower() in entry.lower()]
            hint = f"; exported functions like it: {', '.join(near[:8])}" if near else (
                f"; it exports: {', '.join(exported[:12])}{' ...' if len(exported) > 12 else ''}" if exported else "")
            raise ProcessLibraryError(
                f"{path or lib} has no function {entry!r} for {role}: process functions must be exported with C "
                f"linkage (extern \"C\" in C++, which otherwise mangles names; __declspec(dllexport) on Windows)"
                f"{hint}") from None
        return ctypes.cast(function, ctypes.c_void_p).value
    if isinstance(entry, _FUNCTION_POINTER):
        return ctypes.cast(entry, ctypes.c_void_p).value
    raise TypeError(f"{role} must be a symbol name, a ctypes function or an address, not {type(entry).__name__}")


def _user_data(user_data):
    """(address, object to retain) of `user_data`."""
    if user_data is None:
        return None, None
    if isinstance(user_data, int) and not isinstance(user_data, bool):
        if user_data < 0:
            raise ValueError("user_data address must be non-negative")
        return user_data or None, None
    if isinstance(user_data, np.ndarray):
        if not (user_data.flags.c_contiguous or user_data.flags.f_contiguous):
            raise ValueError("a user_data array must be contiguous")
        return user_data.ctypes.data, user_data
    if isinstance(user_data, (ctypes.Structure, ctypes.Union, ctypes.Array, ctypes._SimpleCData)):
        return ctypes.addressof(user_data), user_data
    raise TypeError(f"user_data must be None, an address, a ctypes object or a numpy array, "
                    f"not {type(user_data).__name__}")


def _feed(h, value, depth=0):
    """Hashes a key value: numbers, strings, bytes, numpy arrays, ctypes
    objects (their bytes) and lists, tuples or dicts of them."""
    if depth > 32:
        raise ValueError("key is nested too deeply")
    if value is None or isinstance(value, (bool, int, float, str)):
        h.update(json.dumps(value).encode())
    elif isinstance(value, (bytes, bytearray, memoryview)):
        h.update(b"b")
        h.update(bytes(value))
    elif isinstance(value, np.ndarray):
        h.update(f"a{value.dtype.str}{value.shape}".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.generic):
        h.update(f"s{value.dtype.str}".encode())
        h.update(value.tobytes())
    elif isinstance(value, (ctypes.Structure, ctypes.Union, ctypes.Array, ctypes._SimpleCData)):
        h.update(f"c{type(value).__name__}".encode())
        h.update(bytes(value))
    elif isinstance(value, (list, tuple)):
        h.update(f"l{len(value)}".encode())
        for v in value:
            _feed(h, v, depth + 1)
    elif isinstance(value, dict):
        h.update(f"d{len(value)}".encode())
        for k in sorted(value, key=str):
            _feed(h, str(k), depth + 1)
            _feed(h, value[k], depth + 1)
    else:
        raise TypeError(f"key values must be numbers, strings, bytes, numpy arrays, ctypes objects or "
                        f"lists/tuples/dicts of them, not {type(value).__name__}")


def _formats(csr, dense, row_format, partial_rows):
    """The row format (inferred from the functions given) after checking it."""
    if csr is None and dense is None:
        raise ValueError("give csr= and/or dense=: the CSR and dense process functions")
    if row_format is None:
        row_format = "auto" if csr is not None and dense is not None else "csr" if csr is not None else "dense"
    if row_format not in ("auto", "csr", "dense"):
        raise ValueError(f"row_format must be 'auto', 'csr' or 'dense', not {row_format!r}")
    need = {"auto": ("csr", "dense"), "csr": ("csr",), "dense": ("dense",)}[row_format]
    missing = [r for r in need if {"csr": csr, "dense": dense}[r] is None]
    if missing:
        raise ValueError(f"row_format={row_format!r} needs {' and '.join(f'{m}=' for m in missing)}")
    if partial_rows and csr is None:
        raise ValueError("partial_rows needs a CSR function (csr=)")
    if partial_rows and row_format == "dense":
        raise ValueError("partial_rows applies to CSR rows; row_format='dense' never delivers them")
    return row_format


def _capsule(csr, dense, user_data, key, identity, row_format, partial_rows, owner, code):
    """(capsule, identity, owner) of resolved function addresses; `code`
    (hashable key material, or None) identifies the functions when no
    identity is given."""
    address, retained = _user_data(user_data)
    if identity is None:
        if code is None:
            raise ValueError("functions given by address need an explicit identity=")
        if user_data is not None and key is None:
            raise ValueError("user_data given without key= or identity=: the identity must cover the "
                             "configuration user_data points to (pass key= with the values it is built from)")
        h = hashlib.sha256()
        _feed(h, code)
        _feed(h, {"row_format": row_format, "partial_rows": bool(partial_rows)})
        _feed(h, key)
        identity = int.from_bytes(h.digest()[:8], "little")
    elif key is not None:
        raise ValueError("give identity= or key=, not both")
    # The capsule carries only the vtable's address, so the table and whatever
    # its pointers name stay referenced by the returned owner.
    table = _VTable(_cellpack.PROCESS_ABI_VERSION, ctypes.sizeof(_VTable), csr, dense, address)
    make = ctypes.pythonapi.PyCapsule_New
    make.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    make.restype = ctypes.py_object
    capsule = make(ctypes.addressof(table), CAPSULE_NAME, None)
    return capsule, identity, (table, retained, owner)


def bind_library(library, *, csr=None, dense=None, user_data=None, key=None, identity=None, row_format=None,
                 partial_rows=False, owner=None):
    """(capsule, identity, owner, row_format) of a library's process
    functions; see `ProcessFn.from_library`."""
    row_format = _formats(csr, dense, row_format, partial_rows)
    lib, path = _load(library)
    entries = {"csr": _function(lib, path, "csr", csr), "dense": _function(lib, path, "dense", dense)}
    by_address = any(isinstance(e, int) and not isinstance(e, bool) for e in (csr, dense) if e is not None)
    code = None
    if not by_address and identity is None:
        if path is None:
            raise ValueError("a library without a file path needs an explicit identity=")
        code = {"library": hashlib.sha256(path.read_bytes()).hexdigest(),
                "csr": csr if isinstance(csr, str) else None, "dense": dense if isinstance(dense, str) else None}
    capsule, identity, retained = _capsule(entries["csr"], entries["dense"], user_data, key, identity, row_format,
                                           partial_rows, owner, code)
    return capsule, identity, (lib, retained), row_format


def _python_code(function) -> dict:
    """Key material for a Python function's code: its source when available,
    else its bytecode and constants."""
    import inspect

    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = None
    code = getattr(function, "__code__", None)
    return {"name": getattr(function, "__qualname__", repr(function)), "source": source,
            "bytecode": None if source is not None or code is None else code.co_code.hex(),
            "consts": None if source is not None or code is None else repr(code.co_consts)}


def _cfunc(role, entry):
    """(address, compiled object, key material) of a numba @cfunc or a
    Python function compiled as one."""
    try:
        import numba
        from numba.core.ccallback import CFunc
    except ModuleNotFoundError as e:
        if e.name != "numba":
            raise
        raise ModuleNotFoundError(
            "ProcessFn.from_cfunc needs numba. Install with: uv pip install numba", name=e.name,
        ) from e
    from ._numba import bindings

    expected = bindings().SIGNATURES[role]
    if isinstance(entry, CFunc):
        compiled = entry
        sig = compiled._sig
        arg = sig.args[0] if len(sig.args) == 1 else None
        if (sig.return_type != numba.types.int32 or arg is None
                or not (arg == expected.args[0] or arg == numba.types.voidptr)):
            raise TypeError(f"the {role} cfunc {compiled._pyfunc.__qualname__} has signature {sig}; process "
                            f"functions take {expected} (pycellpack.loader.process_fn.numba_bindings()."
                            f"{role.upper()}_SIGNATURE) or int32(voidptr)")
        function = compiled._pyfunc
    elif callable(entry):
        function = entry
        try:
            compiled = numba.cfunc(expected)(entry)
        except Exception as e:  # numba reports typing and lowering errors with its own types
            raise ProcessBuildError(f"numba could not compile {getattr(entry, '__qualname__', entry)!r} as a "
                                    f"{role} process function {expected}:\n{e}") from e
    else:
        raise TypeError(f"{role} must be a numba @cfunc or a Python function, not {type(entry).__name__}")
    return compiled.address, compiled, {"role": role, "numba": numba.__version__, **_python_code(function)}


def bind_cfuncs(*, csr=None, dense=None, user_data=None, key=None, identity=None, row_format=None,
                partial_rows=False, owner=None):
    """(capsule, identity, owner, row_format) of numba process functions; see
    `ProcessFn.from_cfunc`."""
    row_format = _formats(csr, dense, row_format, partial_rows)
    bound = {role: _cfunc(role, entry) for role, entry in (("csr", csr), ("dense", dense)) if entry is not None}
    address = {role: bound[role][0] if role in bound else None for role in ("csr", "dense")}
    code = [bound[role][2] for role in sorted(bound)]
    capsule, identity, retained = _capsule(address["csr"], address["dense"], user_data, key, identity, row_format,
                                           partial_rows, owner, code)
    return capsule, identity, ({role: b[1] for role, b in bound.items()}, retained), row_format


__all__ = ["Compiler", "CompilerNotFoundError", "ProcessBuildError", "ProcessLibraryError", "bind_cfuncs",
           "bind_library", "compile_library", "find_compiler"]
