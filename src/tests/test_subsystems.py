"""Core APIs work without optional dependencies; H5AD loads its integration on use."""

import subprocess
import sys

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.cellpack.selection import normalize_rows


def test_core_without_optional_integrations(tmp_path):
    program = r"""
import importlib.abc
import sys
from pathlib import Path

class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("torch", "anndata", "scipy", "h5py", "pandas", "zarr", "numba")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError(f"unavailable subsystem: {fullname}", name=fullname)
        return None

sys.meta_path.insert(0, CoreOnly())
import pycellpack
from pycellpack import open_matrix, read, read_h5, write, write_h5cp
for name in ("cellpack", "io", "loader"):
    assert getattr(pycellpack, name) is sys.modules[f"pycellpack.{name}"]
for name in ("opt", "costs", "process_fn"):
    assert getattr(pycellpack.loader, name) is sys.modules[f"pycellpack.loader.{name}"]
assert "pycellpack.anndata" not in sys.modules

from pycellpack import cellpack
from pycellpack.cellpack import scans
from pycellpack.loader.process_fn import _numba, numba_bindings
import numpy as np

assert _numba.CSR_ROW_DTYPE.itemsize > 0
for entry in (numba_bindings, lambda: pycellpack.ProcessFn.from_cfunc(csr=lambda call: 0)):
    try:
        entry()
    except ModuleNotFoundError as exc:
        assert exc.name == "numba"
        assert exc.__cause__.name == "numba"
        assert "uv pip install numba" in str(exc)
    else:
        raise AssertionError("missing numba was ignored")

x = np.arange(12, dtype=np.float32).reshape(3, 4)
encoded = scans.encode_chunk(x)
np.testing.assert_array_equal(scans.decode_chunk(encoded, source_dtype=x.dtype), x)
row = scans.decode_record(scans.Meta(encoded, source_dtype=x.dtype), cell=1)
csr = row.to_csr()
np.testing.assert_array_equal(csr.data, x[1])
path = Path(sys.argv[1]) / "standalone.cpk"
file = write(path, x)
np.testing.assert_array_equal(file.read_dense(), x)
np.testing.assert_array_equal(read(path), x)
assert all(name not in sys.modules for name in ("torch", "anndata", "scipy", "h5py", "pandas", "zarr", "numba"))
"""
    subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_optional_dependency_errors():
    program = r'''
import importlib
import importlib.abc
import sys

from pycellpack import io

class MissingDependency(importlib.abc.MetaPathFinder):
    error = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "anndata":
            raise self.error

finder = MissingDependency()
sys.meta_path.insert(0, finder)
entries = (
    lambda: importlib.import_module("pycellpack.anndata"),
    lambda: io.read_h5("unused.h5cp"),
    lambda: io.write_h5cp("unused.h5cp", None),
    lambda: io.open_matrix("unused.h5cp"),
)
for name in ("anndata", "h5py", "pandas", "scipy", "zarr"):
    for entry in entries:
        finder.error = ModuleNotFoundError(f"No module named {name!r}", name=name)
        try:
            entry()
        except ModuleNotFoundError as exc:
            assert exc.name == name
            assert exc.__cause__ is finder.error
            assert 'uv add "pycellpack[anndata]"' in str(exc)
        else:
            raise AssertionError("missing dependency was ignored")

for error in (
    ModuleNotFoundError("missing internal module", name="anndata._core.missing"),
    ModuleNotFoundError("missing unrelated dependency", name="unrelated"),
    ImportError("incompatible AnnData API"),
    OSError("native library load failed"),
):
    for entry in entries:
        finder.error = error
        try:
            entry()
        except Exception as exc:
            assert exc is error
        else:
            raise AssertionError("integration error was ignored")
'''
    subprocess.run(
        [sys.executable, "-c", program], check=True,
        capture_output=True, text=True, timeout=30,
    )


def test_h5ad_first_use_registers_integration(tmp_path):
    pytest.importorskip("anndata")
    program = r'''
import sys
from pathlib import Path

import numpy as np
from pycellpack import io

assert "anndata" not in sys.modules
assert "pycellpack.anndata" not in sys.modules
path = Path(sys.argv[1]) / "cold.h5cp"
try:
    io.read_h5(path)
except FileNotFoundError:
    pass

from pycellpack.anndata import AnnData
from pycellpack.io import read_h5
import anndata as ad

x = np.arange(6, dtype=np.float32).reshape(2, 3)
io.write_h5cp(path, AnnData(x), encode=True, chunk_rows=1, workers=1)
assert isinstance(io.read_h5(path), AnnData)
np.testing.assert_array_equal(read_h5(path).X, x)
np.testing.assert_array_equal(ad.read_h5ad(path).X, x)
np.testing.assert_array_equal(io.open_matrix(path).read_dense(), x)
'''
    subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)], check=True,
        capture_output=True, text=True, timeout=60,
    )


def test_public_selection_composes_storage_and_loader(tmp_path):
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    store = cellpack.write(tmp_path / "source.cpk", x)
    cells = normalize_rows([3, -1, 3], store.shape[0])
    view = store[cells]
    # Stored columns 0 and 2 take standard ids 1 and 0: the loader's panel
    # reads them in id order, as the Store's gene map does in panel order.
    ids = np.array([1, -1, 0, -1])
    catalog = loader.DataSource([(view, ids)], 2)
    loaded = loader.load(catalog, workers=1, backend="pread")
    np.testing.assert_array_equal(loaded, x[[3, 4, 3]][:, [2, 0]])
    np.testing.assert_array_equal(
        view.map_genes({0: 1, 2: 0}, n_genes=2).read_dense(), loaded
    )
