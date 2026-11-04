"""Native access ABI, deterministic generation, ownership and online execution."""

import gc
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.loader.access_fn import AccessFn, numba_bindings

GLOBAL = r'''
#include "access_fn_cabi.h"
CELLPACK_ACCESS_EXTERN CELLPACK_ACCESS_EXPORT int32_t generate_accesses(
    void *context, uint64_t start, uint64_t count, uint32_t *out) {
    const uint32_t offset = context ? *(const uint32_t *)context : 0;
    for (uint64_t i = 0; i < count; ++i) out[i] = (uint32_t)((start + i) % 7) + offset;
    return 0;
}
static CellPackAccessFn const check_signature = generate_accesses;
'''

CORPUS = r'''
#include "access_fn_cabi.h"
CELLPACK_ACCESS_EXTERN CELLPACK_ACCESS_EXPORT int32_t generate_accesses(
    void *context, uint64_t start, uint64_t count, uint32_t *datasets, uint64_t *rows) {
    (void)context;
    for (uint64_t i = 0; i < count; ++i) {
        datasets[i] = (uint32_t)((start + i) % 2);
        rows[i] = (start + i) % 3;
    }
    return 0;
}
static CellPackCorpusAccessFn const check_signature = generate_accesses;
'''


@pytest.mark.parametrize("language", ["c", "c++"])
def test_global_generator_binds_c_abi_owns_parameters_and_pickles(tmp_path, language):
    params = np.array([13], np.uint32)
    generator = AccessFn.from_source(code=GLOBAL, length=100, user_data=params, key=13,
                                            language=language, cache_dir=tmp_path)
    identity = generator.identity
    del params
    gc.collect()
    want = np.arange(100, dtype=np.uint32) % 7 + 13
    np.testing.assert_array_equal(generator.fill(), want)
    restored = pickle.loads(pickle.dumps(generator))
    assert restored.identity == identity
    intervals = [(31, 20), (0, 40), (30, 20), (99, 1), (100, 0)] * 3
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda span: restored.fill(*span), intervals))
    for (start, count), values in zip(intervals, results):
        np.testing.assert_array_equal(values, want[start:start + count])
    with pytest.raises(ValueError, match="access range"):
        restored.fill(99, 2)


def test_library_binding_and_corpus_adapter(tmp_path):
    built = AccessFn.from_source(code=CORPUS, length=21, kind="corpus", language="c", cache_dir=tmp_path)
    library = next(tmp_path.glob("*.so"))
    direct = AccessFn.from_library(library, length=21, kind="corpus")
    rows = np.arange(21, dtype=np.uint32) % 3
    want = rows + (np.arange(21, dtype=np.uint32) % 2) * 3
    for generator in (built, pickle.loads(pickle.dumps(direct))):
        np.testing.assert_array_equal(generator.fill(counts=[3, 3]), want)
    with pytest.raises(ValueError, match="dataset row counts"):
        built.bind()
    with pytest.raises(ValueError, match="outside the corpus"):
        built.fill(counts=[1, 1])




def test_generated_accesses_drive_online_planning_and_native_slices(tmp_path):
    x = np.arange(7 * 5, dtype=np.float32).reshape(7, 5)
    store = cellpack.write(tmp_path / "x.cpk", x, workers=1)
    loading = loader.Loader(loader.DataSource([(store, np.arange(5))], 5), workers=2, backend="pread", slot_rows=3)
    generator = AccessFn.from_source(code=GLOBAL, length=31, cache_dir=tmp_path)
    options = loader.PlanOptions(cache_bytes=4096 * 8, cost=None, threads=2, chunk=5)
    with loading.start(generator, plan=options, slots=2) as run:
        actual = np.concatenate([batch.data.copy() for batch in run])
        stats = run.join()
    np.testing.assert_array_equal(actual, x[np.arange(31) % 7])
    assert stats.online is not None
    native = generator.bind().slice(5, 12)
    np.testing.assert_array_equal(native.fill(), np.arange(5, 17) % 7)
    sequential = loader.Accesses.array(np.tile(np.arange(7, dtype=np.uint32), 2)).slice(3, 9)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(sequential)).fill(), np.arange(3, 12) % 7)


def test_numba_global_and_corpus_signatures(tmp_path):
    numba = pytest.importorskip("numba")
    bindings = numba_bindings()

    @numba.cfunc(bindings.GLOBAL_SIGNATURE)
    def global_fn(context, start, count, cells):
        for i in range(count):
            cells[i] = (start + i) % 7
        return 0

    def corpus_fn(context, start, count, datasets, rows):
        for i in range(count):
            datasets[i] = (start + i) % 2
            rows[i] = (start + i) % 3
        return 0

    generator = AccessFn.from_cfunc(global_fn, length=30)
    np.testing.assert_array_equal(generator.fill(), np.arange(30) % 7)
    corpus = AccessFn.from_cfunc(corpus_fn, length=30, kind="corpus")
    np.testing.assert_array_equal(corpus.fill(counts=[3, 3]), np.arange(30) % 3 + (np.arange(30) % 2) * 3)
    with pytest.raises(TypeError, match="requires"):
        AccessFn.from_cfunc(global_fn, length=30, kind="corpus")


def test_corpus_generator_binds_loader_source_snapshot_and_row_views(tmp_path):
    x = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    a = cellpack.write(tmp_path / "a.cpk", x, workers=1)[[5, 2, 4]]
    b = cellpack.write(tmp_path / "b.cpk", x + 100, workers=1)[[1, 3, 0]]
    source = loader.DataSource([(a, np.arange(4)), (b, np.arange(4))], 4)
    loading = loader.Loader(source, workers=2, backend="pread", slot_rows=3)
    gen = AccessFn.from_source(code=CORPUS, kind="corpus", length=21, cache_dir=tmp_path)
    want = np.stack([(x[[5, 2, 4]] if i % 2 == 0 else (x + 100)[[1, 3, 0]])[i % 3] for i in range(21)])
    options = loader.PlanOptions(cache_bytes=4096 * 8, cost=None, threads=2, chunk=5)
    plan = loading.compile(gen, options)
    assert loading.replay(plan, gen).accesses == 21
    assert loader.PageTrace.from_source(source, gen).page_uses > 0
    assert len(source.stored_cells(gen)) == 21
    # Existing loaders bind the registration snapshot even if the source grows.
    source.add(b, np.arange(4))
    for planning in (None, options, plan):
        with loading.start(gen, plan=planning, slots=2) as run:
            got = np.concatenate([batch.data.copy() for batch in run])
            run.join()
        np.testing.assert_array_equal(got, want)
