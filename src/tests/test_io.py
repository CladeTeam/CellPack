"""User-facing .cpk tools in pycellpack.io and the H5AD export."""

import numpy as np
import pytest

from pycellpack import cellpack, io


def _matrix():
    x = np.arange(60, dtype=np.float32).reshape(10, 6)
    x[::3, 1:4] = 0
    return x


def test_write_cpk_accepts_matrices_stores_and_batches(tmp_path):
    x = _matrix()
    whole = io.write_cpk(tmp_path / "whole.cpk", x, chunk_rows=4)
    np.testing.assert_array_equal(whole.read_dense(), x)

    batches = (x[i : i + 3] for i in range(0, 10, 3))
    streamed = io.write_cpk(tmp_path / "dense.cpk", batches, chunk_rows=4)
    assert streamed.format == "dense"
    np.testing.assert_array_equal(streamed.read_dense(), x)

    csr = io.write_cpk(tmp_path / "csr.cpk", (cellpack.read(whole.file, slice(i, i + 4), format="csr")
                                              for i in range(0, 10, 4)))
    assert csr.format == "csr"
    np.testing.assert_array_equal(csr.read_dense(), x)

    view = whole[[9, 0, 4], [5, 1]]
    copy = io.write_cpk(tmp_path / "view.cpk", view, format="csr")
    assert copy.format == "csr"
    np.testing.assert_array_equal(copy.read_dense(), x[[9, 0, 4]][:, [5, 1]])

    with pytest.raises(ValueError, match="at least one row batch"):
        io.write_cpk(tmp_path / "empty.cpk", iter(()))
    with pytest.raises(TypeError):
        io.write_cpk(tmp_path / "bad.cpk", 3)


def test_read_iter_open_and_info(tmp_path):
    x = _matrix()
    path = tmp_path / "m.cpk"
    io.write_cpk(path, x, chunk_rows=4, page_size=512)
    np.testing.assert_array_equal(io.read_cpk(path, [7, 2], [0, 5]), x[[7, 2]][:, [0, 5]])
    csr = io.read_cpk(path, dtype=np.float64, format="csr")
    assert csr.data.dtype == np.float64 and csr.shape == x.shape and csr.nnz == np.count_nonzero(x)
    batches = list(io.iter_cpk(path, 4))
    assert [len(b) for b in batches] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate(batches), x)

    info = io.cpk_info(path)
    assert info.shape == (10, 6) and info.dtype == np.float32 and info.format == "dense"
    assert info.page_size == 512 and info.chunk_rows == 4 and info.n_chunks == 3
    assert info.nbytes == path.stat().st_size and info.path == path.resolve()

    memory = io.open_cpk(path.read_bytes())
    np.testing.assert_array_equal(memory.read_dense(), x)
    assert io.cpk_info(memory).path is None
    assert io.open_cpk(memory) is memory


def test_h5ad_to_cpk_exports_matrix_slots(tmp_path):
    ad = pytest.importorskip("anndata")
    sparse = pytest.importorskip("scipy.sparse")
    x = _matrix()
    a = ad.AnnData(x, layers={"counts": sparse.csr_matrix(x * 2)})
    plain = tmp_path / "plain.h5ad"
    a.write_h5ad(plain)
    dense = io.h5ad_to_cpk(plain, tmp_path / "x.cpk", batch_rows=3)
    np.testing.assert_array_equal(dense.read_dense(), x)
    counts = io.h5ad_to_cpk(plain, tmp_path / "counts.cpk", "layers/counts", batch_rows=4)
    assert counts.format == "csr"
    np.testing.assert_array_equal(counts.read_dense(), x * 2)

    embedded = tmp_path / "cells.h5cp"
    io.write_h5cp(embedded, a, encode=True, chunk_rows=4)
    copied = io.h5ad_to_cpk(embedded, tmp_path / "copied.cpk")
    np.testing.assert_array_equal(copied.read_dense(), x)
    assert (tmp_path / "copied.cpk").read_bytes() == io.open_matrix(embedded).file.to_bytes()

    with pytest.raises(ValueError, match="matrix slot"):
        io.h5ad_to_cpk(plain, tmp_path / "obs.cpk", "obs")
