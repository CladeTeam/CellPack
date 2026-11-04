"""Native component composition over independent NumPy-owned buffers."""

import gc

import numpy as np
import pytest

from pycellpack.cellpack import CSR
from pycellpack.cellpack import scans as s


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
@pytest.mark.parametrize("layout", ["sparse", "dense", "auto"])
def test_chunk_components_round_trip(dtype, layout):
    x = np.zeros((7, 24), dtype=dtype)
    x[0, [2, 8, 19]] = [3, 2, 4]
    x[2, :] = np.arange(24) % 5
    x[3, [1, 6]] = [1, 7]
    x[5] = x[0]
    options = s.EncodeOptions(layout=layout)
    sizes = s.chunk_sizes(x, options=options)
    bound, words = sizes.output_bytes, sizes.workspace_words
    ws = np.empty(words, np.uint64)
    out = np.empty(bound + 23, np.uint8)
    encoded = s.encode_chunk(x, options=options, out=out, workspace=ws)
    assert np.shares_memory(encoded, out)
    assert len(encoded) <= bound
    meta = s.Meta(encoded, source_dtype=x.dtype)
    assert meta.shape == x.shape
    assert meta.value_width == x.dtype.itemsize
    assert meta.nnz_total == np.count_nonzero(x)
    assert s.dense_bytes(meta) == x.nbytes
    assert s.selection_nnz(meta, [5, 0, 5, -1]) == np.count_nonzero(x[[5, 0, 5, -1]])
    tables = s.DecodeTables(meta)
    workspace = np.empty(meta.decode_workspace_words, np.uint64)
    decoded = np.full(x.shape, 99, x.dtype)
    assert (
        s.decode_dense(meta, tables=tables, out=decoded, workspace=workspace) is decoded
    )
    np.testing.assert_array_equal(decoded, x)
    np.testing.assert_array_equal(s.decode_chunk(encoded, source_dtype=x.dtype), x)
    np.testing.assert_array_equal(s.decode_dense(meta, fill=s.Fill.ASSUME_ZEROED), x)
    np.testing.assert_array_equal(
        s.decode_dense(meta, rows=[5, 0, 5, -1]), x[[5, 0, 5, -1]]
    )
    sparse = s.decode_csr(meta, rows=[2, 0, 2], tables=tables)
    np.testing.assert_array_equal(sparse.to_scipy().toarray(), x[[2, 0, 2]])
    for cell in range(len(x)):
        record = meta.record(cell)
        assert np.shares_memory(record, encoded) or not len(record)
        assert s.record_nnz(record) == meta.cell_nnz(cell)
        np.testing.assert_array_equal(
            s.decode_record_dense(meta, cell=cell, tables=tables), x[cell]
        )
        row = s.decode_record(meta, record, tables=tables)
        np.testing.assert_array_equal(
            row.to_csr().to_scipy().toarray(), x[cell : cell + 1]
        )
        parts = s.decode_parts(meta, cell=cell, tables=tables, space="source")
        np.testing.assert_array_equal(parts.expand_values(), x[cell, parts.ids])
        assert parts.nnz == np.count_nonzero(x[cell])
    packed = s.pack_chunk(encoded, page_size=512)
    assert len(packed) == s.packed_len(encoded, 512)
    packed_meta = s.Meta(packed, source_dtype=x.dtype)
    np.testing.assert_array_equal(s.decode_dense(packed_meta), x)


@pytest.mark.parametrize("values", ["raw", "dict", "modes"])
def test_parts_dictionary_modes_and_composed_mapping(values):
    x = np.zeros((12, 64), np.uint16)
    x[:, [1, 5, 12, 19, 21, 38, 60]] = [1, 2, 3, 2, 1, 2, 3]
    x[3, 5] = 7
    options = s.EncodeOptions(layout="sparse", values=values, column_modes=0.0)
    encoded = s.encode_chunk(x, options=options)
    meta = s.Meta(encoded, source_dtype=x.dtype)
    tables = s.DecodeTables(meta)
    source_map = np.full(x.shape[1], -1, np.int64)
    source_map[[38, 5, 19]] = [0, 1, 3]
    panel = s.compose_mapping(tables, source_map)
    np.testing.assert_array_equal(
        panel, source_map.astype(np.uint32)[tables.source_of_coded]
    )
    parts = s.decode_parts(meta, cell=3, tables=tables)
    np.testing.assert_array_equal(
        s.map_dense(parts, panel, dtype=np.float32, n_genes=5),
        np.array([2, 7, 0, 2, 0], np.float32),
    )
    mapped = s.map_parts(parts, panel, dtype=np.float64, n_genes=5)
    dense = np.zeros(5)
    dense[mapped.indices] = mapped.data
    np.testing.assert_array_equal(dense, [2, 7, 0, 2, 0])
    positions = np.array([0, 2, 0], np.uint32)
    subset = s.map_parts_at(parts, positions, panel, dtype=np.float32)
    np.testing.assert_array_equal(subset.indices, np.asarray(panel)[parts.ids[positions]])
    np.testing.assert_array_equal(
        subset.data, parts.expand_values()[positions].astype(np.float32)
    )
    if parts.values.dictionary is not None:
        dictionary = parts.values.dictionary
        np.testing.assert_array_equal(
            dictionary.cast_into(np.float64), dictionary.entries.astype(np.float64)
        )
        assert dictionary.ranks.kind in ("one", "uint8", "uint32")
        if parts.values.kind == "modes":
            np.testing.assert_array_equal(dictionary.modes, meta.column_modes)
    with pytest.raises(ValueError, match="read-only"):
        parts.ids[:] = 0
    editable_ids = parts.ids.copy()
    editable_ids[:] = 0
    del encoded, meta, tables
    gc.collect()
    np.testing.assert_array_equal(
        s.map_dense(parts, panel, dtype=np.float32, n_genes=5), [2, 7, 0, 2, 0]
    )


def test_record_encoding_and_unrenumbered_tables():
    x = np.zeros((6, 48), np.float32)
    x[:, [2, 8, 27]] = [1, 3, 2]
    options = s.EncodeOptions(renumber=False, layout="sparse", values="dict")
    meta = s.Meta(s.encode_chunk(x, options=options), source_dtype=x.dtype)
    tables = s.EncodeTables(meta)
    assert tables.coded_of_source is None
    decoding = s.DecodeTables(meta)
    assert decoding.source_of_coded is None
    np.testing.assert_array_equal(
        s.compose_mapping(decoding, np.arange(48, dtype=np.uint32)), np.arange(48)
    )
    encoded = s.encode_record(meta, x[1], tables=tables, options=options)
    np.testing.assert_array_equal(s.decode_record_dense(meta, encoded), x[1])
    ip = np.array([0, 3], np.int32)
    ix = np.array([2, 8, 27], np.int32)
    row = CSR(ip, ix, x[1, ix], (1, 48))
    encoded = s.encode_record(meta, row, tables=tables, options=options)
    np.testing.assert_array_equal(s.decode_record_dense(meta, encoded), x[1])


def test_paged_record_variants_cross_a_page_boundary():
    x = (np.arange(4 * 2048, dtype=np.float32).reshape(4, 2048) + 1) / 7
    meta = s.Meta(
        s.encode_chunk(x, options=s.EncodeOptions(layout="dense", values="raw")),
        source_dtype=x.dtype,
    )
    record = meta.record(2)
    storage = np.full(47 + len(record), 0, np.uint8)
    storage[47:] = record
    pages = [storage[i : i + 512] for i in range(0, len(storage), 512)]
    paged = s.PagedRecord(pages, 512, 47, len(record))
    assert s.record_nnz(paged) == s.record_nnz(record)
    np.testing.assert_array_equal(s.decode_record_dense(meta, paged), x[2])
    np.testing.assert_array_equal(
        s.decode_record(meta, paged).to_csr().to_scipy().toarray(), x[2:3]
    )
    parts = s.decode_parts(meta, paged, space="source")
    np.testing.assert_array_equal(parts.expand_values(), x[2])
    del paged, pages, storage, record, meta
    gc.collect()
    np.testing.assert_array_equal(parts.expand_values(), x[2])


def test_cast_combinations_and_output_validation():
    for source in (
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ):
        data = np.array([0, 1, 3, 7], source)
        for target in (
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
        ):
            out = np.empty(4, target)
            assert s.cast_values(data, out=out) is out
            np.testing.assert_array_equal(out, data.astype(target))
    x = np.eye(5, dtype=np.float32)
    encoded = s.encode_chunk(x)
    meta = s.Meta(encoded, source_dtype=x.dtype)
    before = np.full_like(x, 7)
    with pytest.raises(ValueError):
        s.decode_dense(meta, out=before, workspace=before.ravel()[:24].view(np.uint64))
    np.testing.assert_array_equal(before, 7)
    bad = encoded.copy()
    bad[0] ^= 255
    with pytest.raises(ValueError):
        s.Meta(bad)
    wrong_meta = s.Meta(s.encode_chunk(np.ones((6, 9), np.float32)), source_dtype=np.float32)
    with pytest.raises(ValueError):
        s.decode_dense(meta, tables=s.DecodeTables(wrong_meta))
    assert s.kernel_set() in s.supported_kernel_sets()
    assert s.decode_workspace_bound(5, 4) >= meta.decode_workspace_words


def test_large_rank_dictionary_and_dense_parts_mapping():
    x = np.tile(np.arange(1, 401, dtype=np.uint32), (4, 1))
    meta = s.Meta(
        s.encode_chunk(
            x,
            options=s.EncodeOptions(layout="sparse", values="dict", column_modes=None),
        ),
        source_dtype=x.dtype,
    )
    parts = s.decode_parts(meta, cell=0, space="source")
    assert parts.values.dictionary.k == 400
    assert parts.values.dictionary.ranks.kind == "uint32"
    np.testing.assert_array_equal(
        parts.values.dictionary.entries[parts.values.dictionary.ranks.data],
        parts.expand_values(),
    )
    dense_meta = s.Meta(
        s.encode_chunk(x, options=s.EncodeOptions(layout="dense", values="dict")),
        source_dtype=x.dtype,
    )
    dense_parts = s.decode_parts(dense_meta, cell=2)
    panel = s.compose_mapping(
        s.DecodeTables(dense_meta), np.arange(400, dtype=np.uint32)[::-1]
    )
    np.testing.assert_array_equal(
        s.map_dense(dense_parts, panel, dtype=np.float64), x[2, ::-1]
    )


@pytest.mark.parametrize("rows", [None, slice(2, 6), slice(None, None, -2), [5, 1, 5], np.array([True, False, True, False, False, True])])
def test_selection_composition(rows):
    x = np.arange(36, dtype=np.uint16).reshape(6, 6)
    expected = x if rows is None else x[rows]
    meta = s.Meta(s.encode_chunk(x), source_dtype=x.dtype)
    tables = s.DecodeTables(meta)
    np.testing.assert_array_equal(s.decode_dense(meta, rows=rows, tables=tables), expected)
    np.testing.assert_array_equal(s.decode_csr(meta, rows=rows, tables=tables).to_scipy().toarray(), expected)
    np.testing.assert_array_equal(s.decode_chunk(s.encode_chunk(x, rows=rows), source_dtype=x.dtype), expected)


def test_mapping_output_dtype_and_shape_contracts():
    x = np.array([[0, 2, 4, 0]], np.uint16)
    meta = s.Meta(s.encode_chunk(x), source_dtype=x.dtype)
    parts = s.decode_parts(meta, cell=0, space="source")
    panel = np.arange(4, dtype=np.uint32)
    out = np.full(4, 17, np.float32)
    with pytest.raises(ValueError, match="dtype"):
        s.map_dense(parts, panel, dtype=np.float64, out=out)
    with pytest.raises(ValueError, match="shape"):
        s.map_dense(parts, panel, n_genes=3, out=out)
    np.testing.assert_array_equal(out, 17)
    assert s.map_dense(parts, panel, out=out) is out
    np.testing.assert_array_equal(out, x[0])
    with pytest.raises(ValueError, match="dtype"):
        s.cast_values(x[0], dtype=np.float64, out=out)
    sparse = s.SparseRow(np.empty(4, np.uint32), np.full(4, 17, np.float32), 4)
    for function, args in [(s.map_parts, (parts, panel)), (s.map_parts_at, (parts, [0], panel))]:
        with pytest.raises(ValueError, match="dtype"):
            function(*args, dtype=np.float64, out=sparse)
    np.testing.assert_array_equal(sparse.data, 17)
    with pytest.raises(ValueError, match="width"):
        s.map_parts(parts, panel, n_genes=1)


def test_gather_and_output_must_not_overlap():
    x = np.eye(4, dtype=np.uint32)
    meta = s.Meta(s.encode_chunk(x), source_dtype=x.dtype)
    output = np.zeros_like(x)
    rows = output.ravel()[:4]
    rows[:] = np.arange(4, dtype=np.uint32)
    before = output.copy()
    with pytest.raises(ValueError, match="overlap"):
        s.decode_dense(meta, rows=rows, out=output)
    np.testing.assert_array_equal(output, before)


@pytest.mark.parametrize("values", ["raw", "dict", "modes"])
def test_metadata_snapshot_and_explicit_mapping_components(values):
    x = np.tile(np.array([0, 2, 0, 4, 2, 0, 3, 0], np.uint16), (8, 1))
    x[3, 1] = 7
    encoded = s.encode_chunk(x, options=s.EncodeOptions(layout="sparse", values=values, column_modes=0.0))
    meta = s.Meta(encoded, source_dtype=x.dtype)
    tables = s.DecodeTables(meta)
    panel = s.compose_mapping(tables, np.arange(8, dtype=np.uint32))
    expected_nnz = meta.cell_nnz(3)
    encoded[:meta.meta_len] = 0
    assert meta.cell_nnz(3) == expected_nnz
    np.testing.assert_array_equal(s.decode_dense(meta, tables=tables), x)
    parts = s.decode_parts(meta, cell=3, tables=tables)
    dictionary = parts.values.dictionary
    converted = None if dictionary is None else dictionary.cast_into(np.float64)
    modes = None if parts.values.kind != "modes" else meta.column_modes.astype(np.float64)
    out = np.empty(8, np.float64)
    s.map_dense(parts, panel, out=out, dictionary=converted, modes=modes)
    np.testing.assert_array_equal(out, x[3])
    del meta, tables, encoded
    gc.collect()
    s.map_dense(parts, panel, out=out, dictionary=converted, modes=modes)
    np.testing.assert_array_equal(out, x[3])


@pytest.mark.parametrize("provided", ["out", "workspace", "both"])
def test_encoding_with_caller_buffers(provided):
    x = np.arange(60, dtype=np.float32).reshape(10, 6)
    rows = [7, 1, 7, 4]
    options = s.EncodeOptions(layout="auto")
    capacity, words = s.chunk_sizes(x, rows, options=options)
    kwargs = {}
    if provided in ("out", "both"):
        kwargs["out"] = np.empty(capacity, np.uint8)
    if provided in ("workspace", "both"):
        kwargs["workspace"] = np.empty(words, np.uint64)
    result = s.encode_chunk(x, rows, options=options, **kwargs)
    expected = s.encode_chunk(x, rows, options=options)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(s.decode_chunk(result, source_dtype=x.dtype), x[rows])


def test_gene_mapped_chunk_decoding_matches_file_reads(tmp_path):
    from pycellpack import cellpack

    rng = np.random.default_rng(3)
    x = rng.poisson(0.4, size=(1500, 30)).astype(np.uint16)
    x[7] = np.arange(1, 31)  # a full row is stored densely
    meta = s.Meta(s.encode_chunk(x), source_dtype=x.dtype)
    file = cellpack.write(tmp_path / "x.cpk", x, chunk_rows=1500)
    gene_map = np.full(30, -1)
    gene_map[[29, 3, 11, 7]] = [0, 4, 2, 1]
    rows = rng.integers(0, 1500, 900)
    expected = file[rows].map_genes(gene_map, n_genes=6).astype(np.float32).read_dense()
    np.testing.assert_array_equal(
        s.decode_dense(meta, rows=rows, gene_map=gene_map, n_genes=6, dtype=np.float32, workers=4),
        expected,
    )
    csr = s.decode_csr(meta, rows=rows, gene_map=gene_map, n_genes=6, dtype=np.float32, workers=4)
    assert csr.shape == (900, 6) and csr.data.dtype == np.float32
    assert all(np.all(np.diff(csr.indices[csr.indptr[i]:csr.indptr[i + 1]]) > 0) for i in range(900))
    np.testing.assert_array_equal(csr.to_scipy().toarray(), expected)
    # Dense-layout records map like sparse ones.
    tables = s.DecodeTables(meta)
    parts = s.decode_parts(meta, cell=7, tables=tables, space=s.IdSpace.SOURCE)
    assert parts.layout is s.RecordLayout.DENSE
    row = s.map_parts(parts, gene_map, dtype=np.float32, n_genes=6)
    mapped = np.zeros(6, np.float32)
    mapped[row.indices] = row.data
    np.testing.assert_array_equal(mapped, file[[7]].map_genes(gene_map, n_genes=6).astype(np.float32).read_dense()[0])
    at = s.map_parts_at(parts, [29, 0], gene_map, dtype=np.float32, n_genes=6)
    np.testing.assert_array_equal(at.indices, [0, s.UNMAPPED])
    np.testing.assert_array_equal(at.data, [30, 1])
