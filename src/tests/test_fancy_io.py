"""Arbitrary cell gathers and composed gene mapping directly into outputs."""

import numpy as np
import pytest
from scipy import sparse

from pycellpack import cellpack, io, loader


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    ],
)
@pytest.mark.parametrize("source", ["dense", "csr"])
def test_fancy_cells_genes_and_buffers(tmp_path, dtype, source):
    x = np.arange(13 * 17, dtype=np.uint16).reshape(13, 17).astype(dtype)
    x[::2, 1::3] = 0
    store = cellpack.write(
        tmp_path / "m.cpk",
        x if source == "dense" else sparse.csr_array(x),
        chunk_rows=3,
    )
    rows = [11, 0, 11, 5, -1, 2]
    np.testing.assert_array_equal(store[rows].read_dense(), x[rows])
    np.testing.assert_array_equal(
        store[rows].read("csr").to_scipy().toarray(), x[rows]
    )
    np.testing.assert_array_equal(store[::-2].toarray(), x[::-2])
    np.testing.assert_array_equal(
        store[[11, 0, 5]][:, [13, 2, 7]].toarray(), x[[11, 0, 5]][:, [13, 2, 7]]
    )
    np.testing.assert_array_equal(store[3].toarray(), x[[3]])
    assert store[3, -1].toarray()[0, 0] == x[3, -1]
    mask = np.arange(13) % 3 == 0
    np.testing.assert_array_equal(store[mask].toarray(), x[mask])
    expected = np.zeros((len(rows), 5), np.float32)
    expected[:, [4, 0, 2]] = x[rows][:, [2, 13, 7]]
    mapping = np.full(17, -1, np.int64)
    mapping[[2, 13, 7]] = [4, 0, 2]
    out = np.full(expected.shape, 999, np.float32)
    selection = store[rows].map_genes(mapping, n_genes=5).astype(np.float32).prepare()
    assert selection.read_dense(out=out) is out
    np.testing.assert_array_equal(out, expected)
    csr = selection.read_csr(index_dtype=np.int32)
    np.testing.assert_array_equal(csr.to_scipy().toarray(), expected)
    assert all(
        np.all(np.diff(csr.indices[csr.indptr[i] : csr.indptr[i + 1]]) > 0)
        for i in range(len(rows))
    )
    # Columns are positions; cellpack keeps no gene names.
    with pytest.raises(TypeError, match="position"):
        store[rows][:, ["g13", "g2", "g7"]]
    with pytest.raises(TypeError, match="position"):
        store[rows].map_genes({"g2": 1, "g13": 0})
    np.testing.assert_array_equal(
        store[rows].map_genes({2: 1, 13: 0}).read_dense(), x[rows][:, [13, 2]]
    )
    assert store[[]][:, [1, 2]].read_dense().shape == (0, 2)
    assert store[rows][:, []].read_dense().shape == (len(rows), 0)
    assert store[rows][:, []].read_csr().nnz == 0
    batches = list(store[rows][:, [13, 7]].iter_batches(2, format="dense"))
    np.testing.assert_array_equal(np.concatenate(batches), x[rows][:, [13, 7]])
    expected_panel = np.column_stack(
        (x[rows, 13], np.zeros(len(rows), x.dtype), x[rows, 2])
    )
    np.testing.assert_array_equal(
        store[rows].map_genes({13: 0, 2: 2}, n_genes=3).toarray(), expected_panel
    )


def test_native_loader_one_shot_and_embedded_sources(tmp_path):
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    source = cellpack.write(tmp_path / "m.cpk", x)
    rows = [8, 3, 8, 0, 1]
    out = np.empty((len(rows), 2), np.float32)
    # Columns 3 and 0 take standard ids 0 and 1; the others are dropped.
    panel = loader.DataSource([(source, [1, -1, -1, 0])], 2)
    assert loader.load(panel, rows, out=out, workers=2, inflight=2) is out
    np.testing.assert_array_equal(out, x[rows][:, [3, 0]])
    with pytest.raises(TypeError, match="DataSource"):
        loader.load(source, rows)
    data_source = loader.DataSource([(source, np.arange(4))], 4)
    np.testing.assert_array_equal(loader.load(data_source, rows), x[rows])
    assert loader.load(data_source, []).shape == (0, 4)
    import anndata as ad

    io.write_h5cp(tmp_path / "cells.h5cp", ad.AnnData(x), encode=True)
    embedded = io.open_matrix(tmp_path / "cells.h5cp")
    np.testing.assert_array_equal(
        embedded[rows][:, [2, 0]].read_dense(), x[rows][:, [2, 0]]
    )
    np.testing.assert_array_equal(
        loader.load(loader.DataSource([(embedded, [1, -1, 0, -1])], 2), rows), x[rows][:, [2, 0]]
    )


def test_format_parsing_and_invalid_selections(tmp_path):
    path = tmp_path / "m.cpk"
    store = cellpack.write(path, np.eye(8, dtype=np.float32), chunk_rows=3)
    file = store.file
    from pycellpack.cellpack import format as fmt
    from pycellpack.cellpack import scans

    raw = np.fromfile(path, dtype=np.uint8)
    sb = fmt.parse_superblock(raw[: fmt.SUPERBLOCK_LEN])
    assert sb["n_rows"] == 8
    entries = fmt.parse_directory(
        raw[:4096],
        raw[
            sb["directory_offset"] : sb["directory_offset"]
            + sb["n_chunks"] * fmt.ENTRY_LEN
        ],
    )
    assert entries == file.chunks
    # The directory ends the file.
    assert sb["chunk_rows"] == 3
    assert sb["file_len"] == len(raw)
    assert sb["directory_offset"] + sb["n_chunks"] * fmt.ENTRY_LEN == len(raw)
    chunk = file.read_chunk(1)
    out = np.empty_like(chunk)
    assert file.read_chunk(1, out=out) is out
    np.testing.assert_array_equal(chunk, out)
    np.testing.assert_array_equal(
        scans.decode_dense(scans.Meta(chunk, source_dtype=store.dtype)), np.eye(8)[3:6]
    )
    assert file.chunk_meta(1).shape == (3, 8)
    assert file.chunk_meta(-1).shape == (2, 8)
    np.testing.assert_array_equal(
        np.concatenate(list(store.iter_batches(3, format="dense"))), np.eye(8)
    )
    np.testing.assert_array_equal(store[...].toarray(), np.eye(8))
    with pytest.raises((ValueError, IndexError)):
        store[[8]].read_dense()
    with pytest.raises(IndexError):
        store[np.array([True, False])]
    # Repeated columns are a view: the same stored column fills both.
    np.testing.assert_array_equal(
        store[[0]][:, [1, 1]].read_dense(), np.eye(8)[[0]][:, [1, 1]]
    )
    with pytest.raises(ValueError):
        store[[0]].map_genes([0] * 8)


def test_mapped_csr_retains_explicit_zero_and_float_bits(tmp_path):
    ip = np.array([0, 3, 4], np.int64)
    ix = np.array([0, 2, 3, 1], np.int64)
    values = np.array([0x7FC12345, 0, 0x80000000, 0x3F800000], np.uint32).view(
        np.float32
    )
    store = cellpack.write(
        tmp_path / "bits.cpk", cellpack.CSR(ip, ix, values, (2, 4)), chunk_rows=1
    )
    result = store[[0, 1, 0]][:, [3, 2, 0, 1]].read_csr()
    assert result.nnz == 7
    np.testing.assert_array_equal(result.indptr, [0, 3, 4, 7])
    np.testing.assert_array_equal(
        result.data.view(np.uint32), values[[2, 1, 0, 3, 2, 1, 0]].view(np.uint32)
    )
    dense = store[[0, 1, 0]][:, [3, 2, 0, 1]].read_dense()
    assert dense.view(np.uint32)[0, 0] == 0x80000000
    assert dense.view(np.uint32)[0, 2] == 0x7FC12345


def test_self_describing_input_pages(tmp_path):
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    store = cellpack.write(tmp_path / "small-pages.cpk", x, page_size=512, chunk_rows=3)
    assert store.file.page_size == 512
    data_source = loader.DataSource([(store, np.arange(4))], 4)
    assert data_source.page_size == 512
    np.testing.assert_array_equal(
        loader.load(data_source, [4, 0, 4], workers=2, inflight=2), x[[4, 0, 4]]
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"])
def test_read_workspace_reuse_across_selections_and_clear(tmp_path, dtype):
    x = np.arange(48, dtype=np.uint16).reshape(8, 6)
    store = cellpack.write(tmp_path / "workspace.cpk", x, chunk_rows=3)
    workspace = cellpack.ReadWorkspace()
    selection = store[[6, 1, 6]][:, [4, 0, 2]].astype(dtype).prepare()
    expected = x[[6, 1, 6]][:, [4, 0, 2]].astype(dtype)
    out = np.empty(selection.shape, dtype)
    for _ in range(2):
        assert selection.read_dense(out=out, workspace=workspace) is out
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(selection.read_csr(workspace=workspace).to_scipy().toarray(), expected)
        np.testing.assert_array_equal(selection.read_dense(), expected)
    # A workspace can also serve the contiguous reader and a different target dtype.
    np.testing.assert_array_equal(store[1:4].read_dense(workspace=workspace), x[1:4])
    np.testing.assert_array_equal(store[[2, 0]].astype(np.float32).read_csr(workspace=workspace).to_scipy().toarray(), x[[2, 0]])
    workspace.clear()
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(selection.read_dense(workspace=workspace), expected)


def test_concurrent_reads_use_independent_workspaces(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    x = np.arange(512, dtype=np.float32).reshape(64, 8)
    store = cellpack.write(tmp_path / "parallel.cpk", x, chunk_rows=7)
    rows = np.array([61, 5, 12, 61, 0])
    selection = store[rows][:, [7, 2, 0]].astype(np.float64).prepare()
    expected = x[rows][:, [7, 2, 0]]

    def read(workspace):
        out = np.empty(selection.shape, selection.dtype)
        for _ in range(3):
            selection.read_dense(out=out, workspace=workspace)
            np.testing.assert_array_equal(out, expected)
        return out

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(read, [cellpack.ReadWorkspace(), cellpack.ReadWorkspace()]))
    for result in results:
        np.testing.assert_array_equal(result, expected)
