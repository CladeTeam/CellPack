"""AnnData holds CellPack Stores natively; H5AD embeds them as streams."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")

from pycellpack import anndata as cpad
from pycellpack import cellpack, io, loader

_SLOTS = (
    "X",
    "layers/counts",
    "layers/dense",
    "obsm/embedding",
    "obsm/tensor",
    "obsp/neighbors",
    "varm/loadings",
    "varp/correlation",
    "raw/X",
    "raw/varm/loadings",
)


def test_export_rejects_source_replaced_during_copy(tmp_path, monkeypatch):
    from pycellpack.io import _anndata

    matrix = np.ones((4, 3), np.float32)
    source = cellpack.write(tmp_path / "source.cpk", matrix)
    replacement = cellpack.write(tmp_path / "next.cpk", matrix * 2)
    destination = tmp_path / "result.h5cp"
    io.write_h5cp(destination, cpad.AnnData(matrix))
    before = destination.read_bytes()
    embed = _anndata._embed

    def replace_before_open(*args, **kwargs):
        replacement.file.path.replace(source.file.path)
        return embed(*args, **kwargs)

    monkeypatch.setattr(_anndata, "_embed", replace_before_open)
    with pytest.raises(OSError, match="replaced"):
        io.write_h5cp(destination, cpad.AnnData(cellpack.Store(source)))
    assert destination.read_bytes() == before


def example(dense):
    x = np.array(
        [[0, 1, 0, 2], [3, 0, 0, 4], [0, 0, 0, 0], [0, 5, 1, 0], [2, 0, 1, 0]],
        np.float32,
    )
    a = ad.AnnData(
        x if dense else sparse.csr_array(x),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(["a", "a", "b", "c", "c"]),
                "batch": [0, 0, 1, 1, 1],
            },
            index=[f"c{i}" for i in range(5)],
        ),
        var=pd.DataFrame(
            {"gene": ["a", "b", "c", "d"]}, index=[f"g{i}" for i in range(4)]
        ),
    )
    a.layers["counts"] = sparse.csr_array(x.astype(np.uint16))
    a.layers["dense"] = x.astype(np.float64)
    a.obsm["embedding"] = np.arange(15, dtype=np.float32).reshape(5, 3)
    a.obsm["tensor"] = np.arange(30, dtype=np.int32).reshape(5, 2, 3)
    a.obsm["table"] = pd.DataFrame({"x": range(5)}, index=a.obs_names)
    a.obsp["neighbors"] = sparse.eye(5, format="csr", dtype=np.float32)
    a.varm["loadings"] = np.arange(8, dtype=np.float64).reshape(4, 2)
    a.varp["correlation"] = np.eye(4, dtype=np.float32)
    a.uns["config"] = {"label": "sample", "values": np.arange(3)}
    a.raw = a.copy()
    return a, x


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.mark.parametrize("is_dense", [False, True])
def test_encoded_round_trip_and_native_annotations(tmp_path, is_dense):
    a, x = example(is_dense)
    path = tmp_path / "cells.h5cp"
    before_var = a.var.copy()
    io.write_h5cp(path, a, encode=True, chunk_rows=2, compression="gzip", compression_opts=1)
    pd.testing.assert_frame_equal(a.var, before_var)
    with h5py.File(path, "r") as f:
        assert f.attrs["encoding-type"] == "anndata"
        for slot in _SLOTS:
            node = f[slot]
            assert node.attrs["encoding-type"] == "cellpack"
            assert set(node) == {"data"}
            payload = node["data"]
            assert payload.dtype == np.uint8
            assert payload.chunks is None
            assert payload.id.get_create_plist().get_nfilters() == 0
            assert payload.id.get_offset() % 4096 == 0
            assert bytes(payload[:4]) == b"CPK\0"
        assert f["obs"].attrs["encoding-type"] == "dataframe"
        assert f["obsm/table"].attrs["encoding-type"] == "dataframe"
        assert f["uns/config/values"].compression == "gzip"
    # The registered element reader lets plain AnnData decode the file too.
    for b in (ad.read_h5ad(path), io.read_h5(path)):
        assert sparse.issparse(b.X) != is_dense
        np.testing.assert_array_equal(dense(b.X), x)
        pd.testing.assert_frame_equal(a.obs, b.obs)
        np.testing.assert_array_equal(a.var_names, b.var_names)
        for name in a.layers:
            left, right = a.layers[name], b.layers[name]
            assert sparse.issparse(left) == sparse.issparse(right)
            assert left.dtype == right.dtype
            np.testing.assert_array_equal(dense(left), dense(right))
        for name in ("embedding", "tensor"):
            np.testing.assert_array_equal(a.obsm[name], b.obsm[name])
        pd.testing.assert_frame_equal(a.obsm["table"], b.obsm["table"])
        np.testing.assert_array_equal(a.varm["loadings"], b.varm["loadings"])
        np.testing.assert_array_equal(a.varp["correlation"], b.varp["correlation"])
        np.testing.assert_array_equal(dense(b.raw.X), x)
        np.testing.assert_array_equal(a.raw.varm["loadings"], b.raw.varm["loadings"])
        assert b.uns["config"]["label"] == "sample"
    assert isinstance(io.read_h5(path), cpad.AnnData)


def test_lazy_read_keeps_stores_through_views_and_copies(tmp_path):
    a, x = example(False)
    path = tmp_path / "cells.h5cp"
    io.write_h5cp(path, a, encode=True, chunk_rows=2)
    b = io.read_h5(path, lazy=True)
    assert isinstance(b, cpad.AnnData) and b.is_lazy
    assert isinstance(b.X, cellpack.Store) and b.X.file.path == path.resolve()
    assert isinstance(b.layers["counts"], cellpack.Store)
    assert b.obsm["tensor"].shape == (5, 2, 3)
    assert isinstance(b.raw, cpad.Raw) and isinstance(b.raw.X, cellpack.Store)
    assert isinstance(b.obsm["table"], pd.DataFrame)

    rows = b.obs["cell_type"].isin(["a", "c"]).to_numpy()
    view = b[rows, ["g3", "g0"]]
    assert isinstance(view, cpad.AnnData) and view.is_view
    for matrix in (view.X, view.layers["counts"], view.obsm["tensor"], view.raw.X):
        assert isinstance(matrix, cellpack.Store)
    np.testing.assert_array_equal(view.X.toarray(), x[rows][:, [3, 0]])
    np.testing.assert_array_equal(view.obsm["tensor"].to_memory(), a.obsm["tensor"][rows])
    np.testing.assert_array_equal(view.obsp["neighbors"].toarray(), np.eye(5)[rows][:, rows])
    np.testing.assert_array_equal(view.raw.X.toarray(), x[rows])
    np.testing.assert_array_equal(view[[2, 0]].X.toarray(), x[rows][[2, 0]][:, [3, 0]])

    actual = view.copy()
    assert not actual.is_view and isinstance(actual.X, cellpack.Store)
    rebuilt = cpad.AnnData(view)
    assert not rebuilt.is_view and rebuilt.X.shape == (4, 2)
    assert isinstance(rebuilt.raw.X, cellpack.Store)
    memory = view.to_memory()
    assert isinstance(memory, cpad.AnnData) and not memory.is_lazy
    assert sparse.issparse(memory.X) and memory.layers["dense"].dtype == np.float64
    np.testing.assert_array_equal(memory.X.toarray(), x[rows][:, [3, 0]])
    np.testing.assert_array_equal(memory.raw.varm["loadings"], a.raw.varm["loadings"])
    assert b.to_df().shape == (5, 4)
    assert b.T.shape == (4, 5) and not b.T.is_lazy
    np.testing.assert_array_equal(
        np.concatenate([c.toarray() for c, _, _ in b.chunked_X(2)]), x
    )


def test_assigned_stores_are_embedded_byte_for_byte(tmp_path):
    x = np.arange(48, dtype=np.float32).reshape(8, 6)
    x[::2, ::3] = 0
    source = cellpack.write(
        tmp_path / "x.cpk", sparse.csr_array(x), chunk_rows=3, page_size=512
    )
    counts = cellpack.write(tmp_path / "counts.cpk", x.astype(np.uint16))
    a = cpad.AnnData(
        cellpack.Store(source),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )
    a.layers["counts"] = counts  # An opened File is wrapped as a Store.
    assert isinstance(a.layers["counts"], cellpack.Store)
    with pytest.raises(ValueError, match="shape"):
        a.layers["wrong"] = cellpack.Store(source)[:4]
    a.obs["group"] = pd.Categorical(list("aabbaabb"))
    path = tmp_path / "assembled.h5cp"
    a.write_h5cp(path)
    with h5py.File(path, "r") as f:
        assert bytes(f["X/data"][()]) == source.file.path.read_bytes()
        assert bytes(f["layers/counts/data"][()]) == counts.file.path.read_bytes()
        assert f["X/data"].id.get_offset() % 4096 == 0
    b = io.read_h5(path, lazy=True)
    np.testing.assert_array_equal(b.X.toarray(), x)
    np.testing.assert_array_equal(b.layers["counts"].toarray(), x.astype(np.uint16))
    # The caller maps var_names to standard ids; cellpack takes the ids.
    panel = {"g4": 0, "g1": 2}
    ids = [panel.get(g, -1) for g in b.var_names]
    loaded = loader.load(loader.DataSource([(b.X, ids)], 3), np.arange(8), workers=1)
    np.testing.assert_array_equal(loaded[:, [0, 2]], x[:, [4, 1]])
    assert not loaded[:, 1].any()
    assert not hasattr(b.X, "column_names")

    # Sliced stores are re-encoded; writing over the file being read is safe.
    view = b[b.obs["group"] == "b", [5, 1]]
    view.write_h5cp(path)
    c = io.read_h5(path, lazy=True)
    np.testing.assert_array_equal(c.X.toarray(), x[[2, 3, 6, 7]][:, [5, 1]])
    assert c.X.format == "csr" and c.layers["counts"].format == "dense"
    assert list(c.var_names) == ["g5", "g1"]
    # b still reads the replaced file, so its streams are re-encoded, not copied.
    assert not b.X.file.is_current()
    b.write_h5cp(tmp_path / "again.h5cp")
    np.testing.assert_array_equal(io.read_h5(tmp_path / "again.h5cp").X.toarray(), x)


def test_store_slots_on_existing_anndata_and_plain_output(tmp_path):
    x = np.arange(20, dtype=np.float64).reshape(5, 4)
    file = cellpack.write(tmp_path / "x.cpk", x)
    a = cpad.AnnData(np.zeros((5, 4)))
    a.X = file
    a.obsm["pca"] = cellpack.Store(file)[:, :2]
    view = a[1:3]
    with pytest.warns(Warning, match="initializing view as actual"):
        view.X = cellpack.Store(file)[1:3]
    assert not view.is_view and isinstance(view.X, cellpack.Store)
    path = tmp_path / "plain.h5ad"
    io.write_h5cp(path, a, encode=False)
    with h5py.File(path, "r") as f:
        assert f["X"].attrs["encoding-type"] == "array"
        assert f["obsm/pca"].attrs["encoding-type"] == "array"
    np.testing.assert_array_equal(ad.read_h5ad(path).obsm["pca"], x[:, :2])
    # The AnnData's own write_h5ad writes plain H5AD; write_h5cp embeds streams.
    a.write_h5ad(tmp_path / "method.h5ad")
    with h5py.File(tmp_path / "method.h5ad", "r") as f:
        assert f["X"].attrs["encoding-type"] == "array"
    np.testing.assert_array_equal(ad.read_h5ad(tmp_path / "method.h5ad").X, x)
    with pytest.raises(TypeError):
        a.write_h5ad(tmp_path / "method.h5ad", encode=True)
    a.write_h5cp(tmp_path / "method.h5cp")
    with h5py.File(tmp_path / "method.h5cp", "r") as f:
        assert f["X"].attrs["encoding-type"] == "cellpack"
    # Without Stores, writing defers to AnnData unchanged.
    io.write_h5cp(path, cpad.AnnData(x))
    with h5py.File(path, "r") as f:
        assert f["X"].attrs["encoding-type"] == "array"
    assert type(io.read_h5(path)) is cpad.AnnData
    assert io.read_h5(path, lazy=True).X.dtype == np.float64


@pytest.mark.parametrize("backend", ["pread", "uring"])
def test_embedded_matrix_direct_reads_and_loader(tmp_path, backend):
    a, x = example(False)
    scratch_parent = Path(__file__).resolve().parents[2] / "tmp"
    scratch_parent.mkdir(exist_ok=True)
    scratch = tempfile.TemporaryDirectory(prefix="pycellpack-direct-", dir=scratch_parent)
    path = Path(scratch.name) / "embedded.h5cp"
    io.write_h5cp(path, a, encode=True, chunk_rows=2)
    source = io.open_matrix(path)
    assert source.file.offset > 0
    assert source.file.length > 0
    np.testing.assert_array_equal(source[1:4].read_dense(), x[1:4])
    lazy = io.read_h5(path, lazy=True)
    assert lazy.X.file.offset == source.file.offset
    data_source = loader.DataSource([(lazy.X.file, np.arange(4))], 4, direct_io=True)
    accesses = np.array([4, 1, 2, 0, 3], np.uint32)
    loading = loader.Loader(data_source, workers=2, inflight=2, backend=backend, slot_rows=2)
    ring = loading.ring(slots=2)
    actual = []
    with loading.start(accesses, ring=ring) as run:
        for batch in run:
            with batch:
                actual.append(batch.data.copy())
        assert run.join().accesses == 5
    np.testing.assert_array_equal(np.concatenate(actual), x[accesses])
    scratch.cleanup()


def test_plain_h5ad_and_empty_matrices(tmp_path):
    normal = ad.AnnData(np.eye(3, dtype=np.float32))
    normal.write_h5ad(tmp_path / "normal.h5ad")
    read = io.read_h5(tmp_path / "normal.h5ad")
    assert isinstance(read, cpad.AnnData)
    np.testing.assert_array_equal(read.X, normal.X)
    assert io.read_h5(tmp_path / "normal.h5ad", backed="r").isbacked
    for shape in ((0, 3), (3, 0)):
        a = ad.AnnData(np.empty(shape, np.float32))
        path = tmp_path / f"empty{shape[0]}.h5cp"
        io.write_h5cp(path, a, encode=True)
        assert io.read_h5(path).shape == shape


def test_failed_write_preserves_destination_and_rejects_invalid_payload(tmp_path):
    from anndata._io.specs.registry import IORegistryError

    a, _ = example(True)
    path = tmp_path / "protected.h5cp"
    path.write_bytes(b"existing")
    a.uns["unsupported"] = object()
    with pytest.raises(IORegistryError):
        io.write_h5cp(path, a, encode=True)
    assert path.read_bytes() == b"existing"
    del a.uns["unsupported"]
    io.write_h5cp(path, a, encode=True)
    with h5py.File(path, "r+") as f:
        f["X"].attrs["shape"] = [100, 4]
    with pytest.raises(ValueError, match="metadata"):
        io.read_h5(path)
    with pytest.raises(NotImplementedError):
        io.read_h5(path, backed="r+")


def _h5_tree(path):
    """Every group and dataset: attributes, dtype and values (layout aside)."""
    tree = {}
    with h5py.File(path, "r") as f:
        def visit(name, node):
            attrs = {k: np.asarray(v).tolist() for k, v in node.attrs.items()}
            if isinstance(node, h5py.Dataset):
                tree[name] = (attrs, node.dtype, node.shape, node[()])
            else:
                tree[name] = (attrs,)
        f.visititems(visit)
        tree["/"] = ({k: np.asarray(v).tolist() for k, v in f.attrs.items()},)
    return tree


def _scipy_default(matrix):
    return sparse.csr_array((matrix.data, matrix.indices.astype(np.int32),
                             matrix.indptr.astype(np.int32)), shape=matrix.shape)


def _assert_same_h5ad(ours, reference):
    a, b = _h5_tree(ours), _h5_tree(reference)
    assert sorted(a) == sorted(b)
    for name in a:
        assert a[name][0] == b[name][0], name
        if len(a[name]) > 1:
            assert a[name][1:3] == b[name][1:3], name
            np.testing.assert_array_equal(a[name][3], b[name][3], err_msg=name)


@pytest.mark.parametrize("options", [
    {},
    {"as_dense": ["X"]},
    {"compression": "gzip"},
    {"min_indices": True},
])
def test_write_h5ad_writes_what_anndata_writes_for_decoded_stores(tmp_path, options):
    options = dict(options)
    rng = np.random.default_rng(1)
    counts = sparse.random(37, 9, density=0.3, random_state=rng, format="csr")
    counts.data = rng.integers(1, 50, counts.nnz)
    counts = sparse.csr_array(counts.astype(np.uint16))
    counts.data[::5] = 0  # explicit zeros survive the round trip
    tensor = rng.normal(size=(37, 2, 3)).astype(np.float32)
    obs = pd.DataFrame({"kind": list("ab" * 18 + "a")}, index=[f"c{i}" for i in range(37)])
    # Higher-rank Stores come from embedded streams.
    embedded = tmp_path / "t.h5cp"
    io.write_h5cp(embedded, ad.AnnData(np.zeros((37, 1)), obsm={"t": tensor}),
                  encode=True, chunk_rows=4)
    store = cellpack.Store(cellpack.write(tmp_path / "x.cpk", counts, chunk_rows=4))
    rows, cols = np.array([30, 2, 7, 7, 11]), np.array([8, 0, 3])
    tensor_store = io.read_h5(embedded, lazy=True).obsm["t"]

    order = rng.permutation(9)

    def build(x, view, part, t):
        return {"X": x, "obs": obs.copy(), "layers": {"view": view}, "obsm": {"part": part, "t": t}}

    ours = cpad.AnnData(**build(store, store[:, order], store[:, cols], tensor_store))
    # The reference holds the matrices the Stores decode to, in SciPy's index dtype.
    # AnnData applies as_dense to csr_matrix only; Stores follow its documented behavior.
    reference = ad.AnnData(**build(sparse.csr_matrix(counts),
                                   _scipy_default(store[:, order].to_memory()),
                                   _scipy_default(store[:, cols].to_memory()), tensor))
    min_indices = options.pop("min_indices", False)
    with ad.settings.override(write_csr_csc_indices_with_min_possible_dtype=min_indices):
        io.write_h5ad(tmp_path / "ours.h5ad", ours, **options)
        ad.io.write_h5ad(tmp_path / "ref.h5ad", reference, **options)
    _assert_same_h5ad(tmp_path / "ours.h5ad", tmp_path / "ref.h5ad")
    # As in AnnData, strings in obs become categoricals on the object itself.
    assert isinstance(ours.obs["kind"].dtype, pd.CategoricalDtype)
    # Views decode only their selection.
    io.write_h5ad(tmp_path / "rows.h5ad", cpad.AnnData(store[rows][:, cols]))
    np.testing.assert_array_equal(ad.read_h5ad(tmp_path / "rows.h5ad").X.toarray(),
                                  counts[rows][:, cols].toarray())


def test_write_h5ad_over_its_own_source(tmp_path):
    x = sparse.csr_array(np.arange(12, dtype=np.float32).reshape(4, 3))
    path = tmp_path / "data.h5cp"
    io.write_h5cp(path, ad.AnnData(x), encode=True)
    lazy = io.read_h5(path, lazy=True)
    lazy.write_h5ad(path)
    with h5py.File(path, "r") as f:
        assert f["X"].attrs["encoding-type"] == "csr_matrix"
        assert f["X/indices"].dtype == np.int32
    np.testing.assert_array_equal(ad.read_h5ad(path).X.toarray(), x.toarray())
