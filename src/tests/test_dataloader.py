"""Corpus addressing, Store views, ordinary Tensor batches and run cleanup."""

import gc
import pickle
import subprocess
import sys
import weakref

import numpy as np
import pytest

from pycellpack import cellpack
from pycellpack.dataloader import (
    Accesses,
    AccessFn,
    OutputLayout,
    ProcessFn,
    ScDataLoader,
    ScDataset,
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def dataset(tmp_path, name="x", rows=11):
    x = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    return x, ScDataset(cellpack.write(tmp_path / f"{name}.cpk", x, workers=1), ids=np.arange(4))


def visits(corpus):
    counts = [len(d) for d in corpus]
    return Accesses(np.repeat(np.arange(len(corpus), dtype=np.uint32), counts),
                    np.concatenate([np.arange(n, dtype=np.uint64) for n in counts]))


def test_import_construction_and_length_load_no_torch(tmp_path):
    _, data = dataset(tmp_path)
    script = '''
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('blocked torch', name='torch')
sys.meta_path.insert(0, BlockTorch())
import numpy as np
import pycellpack
from pycellpack.dataloader import Accesses, ScDataset, ScDataLoader
store = pycellpack.Store(sys.argv[1])
visits = Accesses(np.zeros(store.shape[0], np.uint32), np.arange(store.shape[0], dtype=np.uint64))
loading = ScDataLoader([ScDataset(store, ids=np.arange(4))], accesses=visits, workers=1, batch_size=3)
assert len(loading) == 4
assert 'torch' not in sys.modules
try:
    iter(loading)
except ModuleNotFoundError as error:
    assert error.name == 'torch' and 'ScDataLoader' in str(error)
else:
    raise AssertionError('iteration should require Torch')
print('PASS lazy Torch')
'''
    out = subprocess.run([sys.executable, "-c", script, str(data.store.file.path)], capture_output=True, text=True,
                         check=True, timeout=30)
    assert "PASS lazy Torch" in out.stdout


def test_dataset_maps_visible_columns_and_snapshots_ids(tmp_path, torch):
    a = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    b = np.arange(3 * 2, dtype=np.float32).reshape(3, 2) + 100
    sa = cellpack.write(tmp_path / "a.cpk", a, workers=1)[[4, 1, 4], [3, 0]]
    sb = cellpack.write(tmp_path / "b.cpk", b, workers=1)
    ids = np.array([30, 10])
    da = ScDataset(sa, ids=ids)
    ids[:] = -1
    db = ScDataset(sb, ids=np.array([10, 30]))
    corpus = [da, db]
    dsids, rows = np.array([0, 1, 0, 1, 0]), np.array([2, 1, 0, 0, 1])
    loading = ScDataLoader(corpus, accesses=(dsids, rows), out_ids=np.array([30, 10, 20]),
                           batch_size=2, prefetch=1, workers=2, backend="pread")
    saved = list(loading)
    got = torch.cat([batch["X"] for batch in saved]).numpy()
    expected_a = np.column_stack((a[[4, 1, 4], 3], a[[4, 1, 4], 0], np.zeros(3)))
    expected_b = np.column_stack((b[:, 1], b[:, 0], np.zeros(3)))
    want = np.stack([(expected_a if d == 0 else expected_b)[r] for d, r in zip(dsids, rows)])
    np.testing.assert_array_equal(got, want)
    again = list(loading)
    np.testing.assert_array_equal(torch.cat([b["X"] for b in again]).numpy(), got)
    np.testing.assert_array_equal(saved[0]["X"].numpy(), want[:2])
    assert loading.stats.accesses == 5


@pytest.mark.parametrize("contiguous", [False, True, ("values",)])
def test_process_fields_share_one_owned_packed_batch(tmp_path, torch, contiguous):
    x, data = dataset(tmp_path)
    code = r'''
#include "process_fn_cabi.h"
struct Out { int64_t ids[2]; float values[2]; uint32_t length; };
extern "C" int32_t emit(const PyCellPackDenseRow *r) {
    auto &o = *reinterpret_cast<Out *>(r->out);
    const float *x = static_cast<const float *>(r->values);
    o.ids[0] = int64_t(r->context.position); o.ids[1] = 99;
    o.values[0] = x[0]; o.values[1] = x[3]; o.length = 2;
    return 0;
}
'''
    output = OutputLayout([("ids", "i8", (2,)), ("values", "f4", (2,)), ("length", "u4")])
    process = ProcessFn.from_source(code=code, dense="emit", output=output, cache_dir=tmp_path)
    loading = ScDataLoader([data], accesses=visits([data]), process=process, batch_size=3, prefetch=1, workers=2,
                           contiguous=contiguous, backend="pread")
    saved = list(loading)
    np.testing.assert_array_equal(torch.cat([b["values"] for b in saved]).numpy(), x[:, [0, 3]])
    np.testing.assert_array_equal(torch.cat([b["ids"][:, 0] for b in saved]).numpy(), np.arange(len(x)))
    first = saved[0]
    if contiguous is False:
        assert first["ids"].untyped_storage().data_ptr() == first["values"].untyped_storage().data_ptr()
        assert not first["values"].is_contiguous()
    else:
        assert first["values"].is_contiguous()
    assert all(b["length"].shape == (len(b["ids"]),) for b in saved)


def test_unaligned_packed_fields_are_copied_after_packed_transfer(tmp_path, torch):
    _, data = dataset(tmp_path)
    code = r'''
#include "process_fn_cabi.h"
#include <cstring>
extern "C" int32_t emit(const PyCellPackDenseRow *r) {
    r->out[0] = 7;
    double value = double(r->context.position) + 0.5;
    std::memcpy(r->out + 1, &value, sizeof value);
    return 0;
}
'''
    process = ProcessFn.from_source(code=code, dense="emit", output=OutputLayout([("tag", "u1"), ("value", "f8")], align=False),
                                    cache_dir=tmp_path)
    for size in (1, 3):
        batches = list(ScDataLoader([data], accesses=visits([data]), process=process, batch_size=size, workers=1, prefetch=1))
        np.testing.assert_array_equal(torch.cat([b["value"] for b in batches]).numpy(), np.arange(len(data)) + 0.5)



def test_external_order_drop_last_and_early_close(tmp_path, torch):
    x, data = dataset(tmp_path)
    order = np.random.default_rng(17).permutation(len(data))
    loading = ScDataLoader([data], accesses=Accesses(np.zeros(len(data), np.uint32), order), batch_size=3,
                           drop_last=True, prefetch=1, workers=2, backend="pread")
    assert len(loading) == 3
    one = torch.cat([b["X"] for b in loading]).numpy()
    np.testing.assert_array_equal(one, x[order[:9]])
    np.testing.assert_array_equal(torch.cat([b["X"] for b in loading]).numpy(), one)
    next_order = np.random.default_rng(18).permutation(len(data))
    loading.accesses = Accesses(np.zeros(len(data), np.uint32), next_order)
    two = torch.cat([b["X"] for b in loading]).numpy()
    np.testing.assert_array_equal(two, x[next_order[:9]])
    with iter(loading) as iterator:
        first = next(iterator)
        transfer = weakref.ref(iterator.transfer)
        with pytest.raises(RuntimeError, match="active iterator"):
            iter(loading)
    assert transfer() is None
    assert len(list(loading)) == 3
    assert first["X"].shape == (3, 4)
    iterator = iter(loading)
    next(iterator)
    del iterator
    gc.collect()
    assert len(list(loading)) == 3



def test_external_access_fn_can_be_replaced_between_runs(tmp_path, torch):
    x, data = dataset(tmp_path)
    code = r'''
#include "access_fn_cabi.h"
extern "C" int32_t generate_accesses(void *context, uint64_t start, uint64_t count, uint32_t *d, uint64_t *r) {
    const uint32_t offset = *static_cast<const uint32_t *>(context);
    for (uint64_t i = 0; i < count; ++i) { d[i] = 0; r[i] = (start + i + offset) % 11; }
    return 0;
}
'''
    def make_fn(offset):
        return AccessFn.from_source(code=code, kind="corpus", length=11,
                                    user_data=np.array([offset], np.uint32), key=offset, cache_dir=tmp_path)
    loading = ScDataLoader([data], access_fn=make_fn(0), batch_size=4, workers=1)
    assert len(loading) == 3
    np.testing.assert_array_equal(torch.cat([b["X"] for b in loading]).numpy(), x)
    loading.access_fn = make_fn(3)
    np.testing.assert_array_equal(torch.cat([b["X"] for b in loading]).numpy(), x[(np.arange(11) + 3) % 11])


def test_native_corpus_generator_online_and_empty_sequence(tmp_path, torch):
    x, data = dataset(tmp_path)
    code = r'''
#include "access_fn_cabi.h"
extern "C" int32_t generate_accesses(void *, uint64_t start, uint64_t count, uint32_t *d, uint64_t *r) {
    for (uint64_t i = 0; i < count; ++i) { d[i] = 0; r[i] = (start + i) % 11; }
    return 0;
}
'''
    gen = AccessFn.from_source(code=code, length=29, kind="corpus", cache_dir=tmp_path)
    loading = ScDataLoader([data], access_fn=gen, cache_bytes=4096 * 8, batch_size=4, prefetch=1,
                           workers=2, backend="pread")
    got = torch.cat([b["X"] for b in loading]).numpy()
    np.testing.assert_array_equal(got, x[np.arange(29) % 11])
    assert loading.stats.online is not None
    empty = ScDataLoader([data], accesses=(np.empty(0, np.uint32), np.empty(0, np.uint64)), workers=1)
    assert len(empty) == 0 and list(empty) == []


def test_anndata_store_convenience_and_input_errors(tmp_path, torch):
    from pycellpack import io
    from pycellpack.anndata import AnnData

    x, data = dataset(tmp_path)
    adata = AnnData(data.store)
    adata.layers["counts"] = data.store
    io.write_h5cp(tmp_path / "cells.h5cp", adata)
    lazy = io.read_h5(tmp_path / "cells.h5cp", lazy=True)
    for layer in (None, "counts"):
        ds = ScDataset.from_anndata(lazy, ids=np.arange(4), layer=layer)
        got = torch.cat([b["X"] for b in ScDataLoader([ds], accesses=visits([ds]), workers=1)]).numpy()
        np.testing.assert_array_equal(got, x)
    with pytest.raises(TypeError, match="explicit NumPy"):
        ScDataset(data.store, ids=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="distinct"):
        ScDataset(data.store, ids=np.zeros(4, np.int32))
    with pytest.raises(ValueError, match="stored dtype"):
        ScDataset(data.store.astype(np.float64), ids=np.arange(4))
    with pytest.raises(ValueError, match="multiple mapped"):
        ScDataset(data.store[:, [0, 0]], ids=np.arange(2))
    invalid = ScDataLoader([data], accesses=(np.array([1]), np.array([0])), workers=1)
    with pytest.raises(ValueError, match="outside corpus"):
        iter(invalid)
    invalid = ScDataLoader([data], accesses=(np.array([0]), np.array([11])), workers=1)
    with pytest.raises(ValueError, match="outside its dataset"):
        iter(invalid)



def test_pickle_rebuilds_corpus_and_native_access_fn(tmp_path, torch):
    x, data = dataset(tmp_path)
    data = pickle.loads(pickle.dumps(data))
    with pytest.raises(ValueError):
        data.ids.setflags(write=True)
    arrays = pickle.loads(pickle.dumps(Accesses(np.array([0, 0]), np.array([3, 1]))))
    np.testing.assert_array_equal(arrays.rows, [3, 1])
    code = r'''
#include "access_fn_cabi.h"
extern "C" int32_t generate_accesses(void *, uint64_t start, uint64_t count, uint32_t *d, uint64_t *r) {
    for (uint64_t i = 0; i < count; ++i) { d[i] = 0; r[i] = start + i; }
    return 0;
}
'''
    fn = AccessFn.from_source(code=code, length=len(data), kind="corpus", cache_dir=tmp_path)
    loading = ScDataLoader([data], access_fn=fn, workers=1, prefetch=1, batch_size=4)
    list(loading)
    rebuilt = pickle.loads(pickle.dumps(loading))
    np.testing.assert_array_equal(torch.cat([b["X"] for b in rebuilt]).numpy(), x)
    with iter(loading) as iterator:
        next(iterator)
        with pytest.raises(TypeError, match="active"):
            pickle.dumps(loading)


@pytest.mark.parametrize("contiguous", [False, True])
def test_cuda_retires_host_views_before_reuse_and_preserves_device_fields(tmp_path, torch, contiguous):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x, data = dataset(tmp_path)
    loading = ScDataLoader([data], accesses=visits([data]), workers=1, prefetch=1, batch_size=2, device="cuda:0", contiguous=contiguous)
    with iter(loading) as iterator:
        batch = next(iterator.run)
        observer = batch.observer()
        packed = iterator.transfer.copy_batch(batch)
        assert not observer.held
        first = iterator.transfer.unpack(packed)
        others = list(iterator)
    np.testing.assert_array_equal(torch.cat([first["X"], *(b["X"] for b in others)]).cpu().numpy(), x)



def test_access_fn_failure_closes_run_and_external_replacement_can_start(tmp_path, torch):
    x, data = dataset(tmp_path)
    code = r'''
#include "access_fn_cabi.h"
extern "C" int32_t generate_accesses(void *, uint64_t, uint64_t count, uint32_t *d, uint64_t *r) {
    for (uint64_t i = 0; i < count; ++i) { d[i] = 0; r[i] = 999; }
    return 0;
}
'''
    bad = AccessFn.from_source(code=code, length=11, kind="corpus", cache_dir=tmp_path)
    loading = ScDataLoader([data], access_fn=bad, workers=1, batch_size=2, prefetch=1)
    with pytest.raises(ValueError, match="outside the corpus"):
        next(iter(loading))
    loading.access_fn = None
    loading.accesses = visits([data])
    np.testing.assert_array_equal(torch.cat([b["X"] for b in loading]).numpy(), x)


def test_cuda_process_field_streams_match_cpu_after_iterator_close(tmp_path, torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pycellpack.dataloader import process_presets

    _, data = dataset(tmp_path)
    process = process_presets.scgpt(np.arange(10, 14), pad_token=0, cls_token=1, length=8, cache_dir=tmp_path)
    cpu = list(ScDataLoader([data], accesses=visits([data]), process=process, workers=1, batch_size=3))
    gpu = list(ScDataLoader([data], accesses=visits([data]), process=process, workers=1, batch_size=3, prefetch=1,
                           device="cuda:0", contiguous=True))
    for host, device in zip(cpu, gpu):
        for name in host:
            assert device[name].is_contiguous()
            np.testing.assert_array_equal(device[name].cpu().numpy(), host[name].numpy())


def test_sparse_standard_ids_map_to_the_requested_panel(tmp_path, torch):
    x, data = dataset(tmp_path)
    ds = ScDataset(data.store, ids=np.array([4_000_000_000, 7, 1_000_000_000, 5], np.uint32))
    loading = ScDataLoader([ds], accesses=visits([ds]), out_ids=np.array([5, 4_000_000_000, 123], np.uint32), workers=1, default=-3)
    got = torch.cat([b["X"] for b in loading]).numpy()
    np.testing.assert_array_equal(got, np.column_stack((x[:, 3], x[:, 0], np.full(len(x), -3))))


def test_custom_process_receives_declared_worker_scratch(tmp_path, torch):
    _, data = dataset(tmp_path)
    code = r'''
#include "process_fn_cabi.h"
#include <cstring>
extern "C" int32_t emit(const PyCellPackDenseRow *r) {
    if (r->scratch_bytes < 8) return -7;
    double value = double(r->context.position) + 0.5;
    std::memcpy(r->scratch, &value, 8);
    std::memcpy(r->out, r->scratch, 8);
    return 0;
}
'''
    process = ProcessFn.from_source(code=code, dense="emit", output=[("value", "f8")], cache_dir=tmp_path)
    loading = ScDataLoader([data], accesses=visits([data]), process=process, scratch_bytes=8, workers=2, batch_size=3)
    np.testing.assert_array_equal(torch.cat([b["value"] for b in loading]).numpy(), np.arange(len(data)) + 0.5)


def test_fixed_accesses_snapshot_each_iterator_and_accept_external_reordering(tmp_path, torch):
    x, data = dataset(tmp_path)
    access = visits([data])
    loading = ScDataLoader([data], accesses=access, workers=1, batch_size=2, prefetch=1)
    with iter(loading) as iterator:
        first = next(iterator)
        access.rows[:] = access.rows[::-1]
        current = [first, *iterator]
    np.testing.assert_array_equal(torch.cat([b["X"] for b in current]).numpy(), x)
    np.testing.assert_array_equal(torch.cat([b["X"] for b in loading]).numpy(), x[::-1])


def test_loader_requires_one_explicit_access_input(tmp_path):
    _, data = dataset(tmp_path)
    with pytest.raises(TypeError, match="exactly one"):
        ScDataLoader([data])
    with pytest.raises(TypeError, match="exactly one"):
        ScDataLoader([data], accesses=visits([data]), access_fn=object())
    with pytest.raises(TypeError, match="must be an AccessFn"):
        ScDataLoader([data], access_fn=lambda: visits([data]))
