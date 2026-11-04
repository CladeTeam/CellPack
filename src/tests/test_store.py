"""Lazy Store views compose selections and decode only on request."""

import pickle

import numpy as np
import pytest
from scipy import sparse

from pycellpack import cellpack


@pytest.fixture(params=["dense", "csr"])
def stored(request, tmp_path):
    x = np.arange(11 * 7, dtype=np.int32).reshape(11, 7)
    x[::2, 1::3] = 0
    payload = x if request.param == "dense" else sparse.csr_array(x)
    file = cellpack.write(tmp_path / "m.cpk", payload, chunk_rows=3)
    return cellpack.Store(file), x


def test_selections_compose_lazily(stored):
    store, x = stored
    assert store.is_identity and store.shape == x.shape and len(store) == 11
    view = store[::-2][[4, 0, 4, -1]][:, [6, 1, 6]]
    assert isinstance(view, cellpack.Store) and not view.is_identity
    expected = x[::-2][[4, 0, 4, -1]][:, [6, 1, 6]]
    assert view.shape == expected.shape
    np.testing.assert_array_equal(view.toarray(), expected)
    np.testing.assert_array_equal(view.tocsr().toarray(), expected)
    np.testing.assert_array_equal(view.read('csr').to_scipy().toarray(), expected)
    np.testing.assert_array_equal(np.asarray(view), expected)
    # Orthogonal selection: integers keep their axis, arrays select a sub-grid.
    np.testing.assert_array_equal(store[3, [0, 2]].toarray(), x[[3]][:, [0, 2]])
    np.testing.assert_array_equal(store[..., 5].toarray(), x[:, [5]])
    mask = np.arange(11) % 3 == 0
    np.testing.assert_array_equal(store[mask, mask[:7]].toarray(), x[mask][:, mask[:7]])
    np.testing.assert_array_equal(store[2:9][1:4].cells, [3, 4, 5])
    assert store[2:9][1:4].read().shape == (3, 7)


def test_identity_is_recovered_and_selection_is_reusable(stored):
    store, x = stored
    assert store[:, :].is_identity
    assert store[np.arange(11)].is_identity
    assert store[:, np.arange(7)].genes is None
    selection = store[[5, 1]][:, [3, 0]].prepare()
    out = np.empty(selection.shape, selection.dtype)
    assert selection.read_dense(out=out) is out
    np.testing.assert_array_equal(out, x[[5, 1]][:, [3, 0]])
    with pytest.raises(ValueError):
        store[:, [1, 1]].prepare()


def test_dtype_batches_buffers_and_errors(stored):
    store, x = stored
    as_float = store[[1, 2]].astype(np.float64)
    assert as_float.dtype == np.float64 and store.dtype == np.int32
    np.testing.assert_array_equal(as_float.toarray(), x[[1, 2]].astype(np.float64))
    out = np.empty((3, 7), np.int32)
    assert store[4:7].read("dense", out=out) is out
    np.testing.assert_array_equal(out, x[4:7])
    batches = list(store[[9, 0, 3, 3, 8]].iter_batches(2, format="dense"))
    np.testing.assert_array_equal(np.concatenate(batches), x[[9, 0, 3, 3, 8]])
    assert store[[]].toarray().shape == (0, 7)
    assert store[:, []].toarray().shape == (11, 0)
    assert store.copy() is store
    restored = pickle.loads(pickle.dumps(store[[3, 1]][:, [2]]))
    np.testing.assert_array_equal(restored.toarray(), x[[3, 1]][:, [2]])
    with pytest.raises(TypeError):
        store[0] = 1
    with pytest.raises(IndexError):
        store[[11]]
    with pytest.raises(TypeError):
        store[:, ["missing"]]  # columns are positions
    with pytest.raises(IndexError):
        store[0, 0, 0]


def test_higher_rank_dense_store(tmp_path):
    x = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    file = cellpack.write(tmp_path / "t.cpk", x.reshape(5, 6))
    store = cellpack.Store(file.file, shape=(5, 2, 3))
    assert store.ndim == 3 and store.size == x.size
    np.testing.assert_array_equal(store[[4, 1]].toarray(), x[[4, 1]])
    np.testing.assert_array_equal(store[1:3, :, :].to_memory(), x[1:3])
    with pytest.raises(IndexError):
        store[:, 0]
    with pytest.raises(ValueError):
        cellpack.Store(file.file, shape=(5, 4))


def test_gene_mapped_views_match_file_reads(stored):
    store, x = stored
    rows = [9, 2, 2, 0]
    # Repeated columns by selection, an absent one by an explicit panel map.
    view = store[rows][:, [4, 0, 4, 6]].map_genes([0, 2, 3, 4], n_genes=5)
    expected = np.zeros((4, 5), x.dtype)
    expected[:, [0, 2, 3, 4]] = x[rows][:, [4, 0, 4, 6]]
    assert view.shape == (4, 5)
    np.testing.assert_array_equal(view.toarray(), expected)
    csr = view.read_csr(index_dtype=np.int32)
    assert csr.indices.dtype == np.int32
    assert all(np.all(np.diff(csr.indices[csr.indptr[i]:csr.indptr[i + 1]]) > 0) for i in range(4))
    np.testing.assert_array_equal(csr.to_scipy().toarray(), expected)
    # Views compose after mapping, and absent columns stay absent.
    np.testing.assert_array_equal(view[1:, [1, 4]].toarray(), expected[1:][:, [1, 4]])

    gene_map = np.full(7, -1)
    gene_map[[2, 5]] = [3, 0]
    mapped = store.map_genes(gene_map, n_genes=4)
    expected_mapped = np.zeros((4, 4), x.dtype)
    expected_mapped[:, [3, 0]] = x[rows][:, [2, 5]]
    np.testing.assert_array_equal(mapped[rows].toarray(), expected_mapped)
    by_position = store.map_genes({2: 3, 5: 0}, n_genes=4)
    np.testing.assert_array_equal(by_position.toarray(), mapped.toarray())
    with pytest.raises(ValueError, match="distinct"):
        store.map_genes({2: 1, 5: 1})
    with pytest.raises(ValueError, match="inside"):
        store.map_genes({2: 5}, n_genes=3)
    with pytest.raises(TypeError, match="positions"):
        store.map_genes({"g2": 1})
    with pytest.raises(TypeError):
        store[:, ["g2"]]
    assert store.map_genes(np.arange(7)).is_identity


def test_in_memory_stream_and_writer_normalization(tmp_path):
    x = np.arange(30, dtype=np.float32).reshape(5, 6)
    x[:, ::2] = 0
    memory = cellpack.compress(sparse.coo_array(x), chunk_rows=2)
    assert memory.file.in_memory and memory.format == "csr"
    np.testing.assert_array_equal(memory[[4, 1], [5, 1]].toarray(), x[[4, 1]][:, [5, 1]])
    copied = pickle.loads(pickle.dumps(memory[::2]))
    np.testing.assert_array_equal(copied.toarray(), x[::2])
    assert cellpack.File.from_bytes(memory.file.to_bytes()).shape == (5, 6)
    # Any NumPy layout or SciPy format is accepted, in either stored layout.
    fortran = cellpack.write(tmp_path / "f.cpk", np.asfortranarray(x), format="csr")
    assert fortran.format == "csr"
    np.testing.assert_array_equal(fortran.read_dense(), x)
    unsorted = sparse.csr_array((np.array([1, 2], np.float32), np.array([3, 1]), np.array([0, 2])), shape=(1, 6))
    np.testing.assert_array_equal(cellpack.write(tmp_path / "u.cpk", unsorted).read_dense(), unsorted.toarray())
    with cellpack.Writer(tmp_path / "w.cpk", 6, np.float32, format="dense") as writer:
        writer.append(sparse.csr_array(x[:2]))
        writer.append(x[2:, :])
    np.testing.assert_array_equal(cellpack.Store(tmp_path / "w.cpk").read_dense(), x)
    with pytest.raises(ValueError, match="dtype"):
        cellpack.Writer(tmp_path / "v.cpk", 6, np.float64).append(x)
    np.testing.assert_array_equal(cellpack.Store(tmp_path / "w.cpk")[np.array(3)].read_dense(), x[[3]])


def test_parallel_encode_and_decode_match_serial(tmp_path):
    from pycellpack import _workers
    from pycellpack.cellpack import scans

    rng = np.random.default_rng(0)
    x = rng.poisson(0.3, size=(6000, 40)).astype(np.float32)
    serial_path, threaded_path = tmp_path / "s.cpk", tmp_path / "p.cpk"
    serial = cellpack.write(serial_path, sparse.csr_array(x), chunk_rows=300, workers=1)
    threaded = cellpack.write(threaded_path, sparse.csr_array(x), chunk_rows=300, workers=8)
    assert serial_path.read_bytes() == threaded_path.read_bytes()
    rows = rng.integers(0, 6000, 5000)
    mapping = np.full(40, -1)
    mapping[[3, 17, 30]] = [2, 0, 1]
    for view in (
        serial,
        threaded,
        serial[150:5900],
        serial[rows],
        serial[rows].map_genes(mapping, n_genes=4),
        serial[rows][:, [9, 1]].astype(np.float64),
    ):
        expected = view.read_dense(workers=1)
        np.testing.assert_array_equal(view.read_dense(workers=8), expected)
        csr = view.read_csr(workers=8)
        np.testing.assert_array_equal(csr.to_scipy().toarray(), expected)
    view = cellpack.Store(serial)[rows][:, [5, 5, 0]].map_genes([0, 2, 3], n_genes=4)
    np.testing.assert_array_equal(view.read_dense(workers=8), view.read_dense(workers=1))
    np.testing.assert_array_equal(view.tocsr().toarray(), view.read_dense(workers=1))

    chunk = scans.encode_chunk(x)
    meta = scans.Meta(chunk, source_dtype=x.dtype)
    np.testing.assert_array_equal(scans.decode_dense(meta, rows=rows, workers=8), x[rows])
    np.testing.assert_array_equal(scans.decode_csr(meta, workers=8).to_scipy().toarray(), x)
    with pytest.raises(TypeError, match="dtype"):
        scans.decode_chunk(chunk)

    assert 1 <= _workers.default_workers() <= _workers.available_cpus()
    assert _workers.resolve_workers(10**6) == _workers.available_cpus()
    with pytest.raises(ValueError):
        _workers.resolve_workers(0)
