"""Caller-owned CSR buffers have the same contract at every worker count."""

import numpy as np
import pytest

from pycellpack import cellpack


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("invalid", ["dtype", "indices", "shape", "readonly", "strided", "overlap"])
def test_invalid_csr_output_is_rejected_before_writes(tmp_path, workers, invalid):
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3) + 0.5
    file = cellpack.write(tmp_path / "matrix.cpk", matrix, chunk_rows=2)
    ip = np.full(5, -7, np.int64)
    ix = np.full(12, -7, np.int64)
    data = np.full(12, -7, np.float32)
    shape = matrix.shape
    if invalid == "dtype":
        data = data.astype(np.int32)
    elif invalid == "indices":
        ix = ix.astype(np.float32)
    elif invalid == "shape":
        shape = (4, 4)
    elif invalid == "readonly":
        data.flags.writeable = False
    elif invalid == "strided":
        data = np.full(24, -7, np.float32)[::2]
    elif invalid == "overlap":
        ip = ix[:5]
    out = cellpack.CSR(ip, ix, data, shape)
    before = [a.copy() for a in (ip, ix, data)]
    with pytest.raises((ValueError, TypeError)):
        file.read_csr(out=out, workers=workers)
    for actual, expected in zip((ip, ix, data), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_parallel_csr_output(tmp_path, index_dtype):
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3)
    file = cellpack.write(tmp_path / "matrix.cpk", matrix, chunk_rows=2)
    out = cellpack.CSR(np.empty(5, index_dtype), np.empty(12, index_dtype),
                       np.empty(12, np.float32), matrix.shape)
    actual = file.read_csr(out=out, workers=2)
    assert actual.indptr is out.indptr
    np.testing.assert_array_equal(actual.to_scipy().toarray(), matrix)


@pytest.mark.parametrize("workers", [1, 2])
def test_csr_column_ids_must_fit_output_dtype(tmp_path, workers):
    from scipy.sparse import csr_array

    matrix = csr_array((np.ones(4, np.float32), np.full(4, 69999),
                        np.arange(5)), shape=(4, 70000))
    file = cellpack.write(tmp_path / "wide.cpk", matrix, chunk_rows=2)
    out = cellpack.CSR(np.full(5, -7, np.int64), np.zeros(4, np.uint16),
                       np.full(4, -7, np.float32), matrix.shape)
    with pytest.raises(ValueError):
        file.read_csr(out=out, workers=workers)
    np.testing.assert_array_equal(out.indptr, -7)
    np.testing.assert_array_equal(out.data, -7)


def test_export_rejects_replaced_path(tmp_path):
    original = np.ones((4, 3), np.float32)
    file = cellpack.write(tmp_path / "matrix.cpk", original)
    cellpack.write(tmp_path / "new.cpk", original * 2)
    (tmp_path / "new.cpk").replace(tmp_path / "matrix.cpk")
    np.testing.assert_array_equal(file.read_dense(), original)
    with pytest.raises(OSError, match="replaced"):
        file.file.to_bytes()
