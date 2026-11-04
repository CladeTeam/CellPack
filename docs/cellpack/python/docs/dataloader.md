# Training batches

`ScDataLoader` turns an explicit sample order over file-backed matrices into
dictionaries of PyTorch Tensors. Native threads read, map and optionally process
rows; the returned batches own their storage. Install PyTorch separately for
your CPU or CUDA environment. Construction and `len()` do not import it.

<DocChoices title="Choose your batch interface" :items="[
  { label: 'ScDataLoader', when: 'Your training step consumes Tensors.', detail: 'Supply corpus-local sample IDs and retain returned batches normally.', href: '#check-an-ordered-cpu-run' },
  { label: 'Native Loader', when: 'You need NumPy views, multiple schemas or manual ring memory.', detail: 'Use global cell IDs and manage output leases explicitly.', href: '/python/docs/native-loader.html' },
]" />

<DocDiagram name="python-training" />

## Check an ordered CPU run

Run this complete example in a writable directory. It writes `training.cpk`,
requests five samples including a repeated row, and returns columns in model
order `[3, 0]`. The built-in dense output needs no process compiler.

```python
import numpy as np
import pycellpack as cp

x = np.arange(24, dtype=np.float32).reshape(6, 4)
store = cp.write("training.cpk", x, workers=1, atomic=True)
corpus = [cp.ScDataset(store, ids=np.arange(4, dtype=np.uint32))]
rows = np.array([5, 1, 5, 0, 2], dtype=np.uint64)
accesses = cp.Accesses(np.zeros(len(rows), dtype=np.uint32), rows)
loading = cp.ScDataLoader(
    corpus, accesses, out_ids=np.array([3, 0], dtype=np.uint32),
    batch_size=2, device="cpu", workers=1, prefetch=2,
    backend="pread", direct_io=False,
)
with loading:
    batches = list(loading)

assert len(loading) == 3
assert [tuple(batch["X"].shape) for batch in batches] == [(2, 2), (2, 2), (1, 2)]
actual = np.concatenate([batch["X"].numpy() for batch in batches])
np.testing.assert_array_equal(actual, x[rows][:, [3, 0]])
assert loading.stats is not None and loading.stats.accesses == 5
print(actual)
```

```text
[[23. 20.]
 [ 7.  4.]
 [23. 20.]
 [ 3.  0.]
 [11.  8.]]
```

The three batches contain 2, 2 and 1 samples. CPU delivery copies each packed
slot before returning it, so collecting this small result in a list is safe.
For a real training loop, consume each batch as it arrives to keep application
memory bounded.

## Keep sample order separate from column identity

| Input | What its indices mean |
| --- | --- |
| `corpus[d]` | Dataset `d`, in registration order |
| `Accesses.dataset_ids[k]` | Which dataset supplies sample position `k` |
| `Accesses.rows[k]` | Row within that dataset's current Store view |
| `ScDataset.ids[c]` | Standard identity of visible source column `c` |
| `out_ids[j]` | Standard identity at model input column `j` |

Sample repeats implement oversampling; their order is preserved. The root
`pycellpack.Accesses` holds dataset/local-row arrays. The different
`pycellpack.loader.Accesses` uses global native cell IDs and is not accepted here.
See [gene alignment](alignment.md) when matrices have different column panels.

## Choose each epoch explicitly

Every new iterator starts a new run with the currently assigned order. It does
not advance an internal epoch counter or reshuffle automatically. For the corpus
above, replace the order between runs:

```python
for epoch in range(2):
    epoch_rows = np.random.default_rng(epoch).permutation(len(store)).astype(np.uint64)
    loading.accesses = cp.Accesses(np.zeros(len(epoch_rows), np.uint32), epoch_rows)
    with loading:
        for batch in loading:
            features = batch["X"]  # Pass to the model here.
    assert loading.stats.accesses == len(epoch_rows)
```

The arrays are copied and validated when the iterator starts. Changes to the
original arrays after that point do not change the active run. Assign the next
order only after closing the previous iterator. To resume within a sample order,
slice both corpus arrays yourself before the next run.

`drop_last=True` removes the incomplete final batch: the first example would
deliver four samples in two batches. `len(loading)` counts batches for the
current order; it does not verify dataset or row bounds. Those checks happen
when iteration starts.

For long computed orders, supply [AccessFn](access-functions.md) through
`access_fn=` instead of materialized `accesses=`. Supply exactly one of the two.

## Outputs and device transfers

Without `process`, batches contain `X` of shape
`(actual_batch_rows, len(out_ids))`, with float32 values by default. Choose
`dtype` on the loader; ScDataset rejects dtype-converting Store views. Different
dataset maps require explicit `out_ids`; identical maps can use their common
non-dropped IDs. A [ProcessFn](process-functions.md) with a declared output layout
returns that layout's field names and shapes instead of built-in `X`.

<DocDiagram name="python-ownership" />

| Target | Returned storage | Transfer behavior |
| --- | --- | --- |
| `device="cpu"` | Fields view an independent packed Tensor clone | Native ring slots can be reused after delivery |
| `device="cuda:0"` | Fields view a packed device allocation | One H2D copy from pinned host memory; its copy stream completes before the host lease is released |

CUDA field views can be strided. `contiguous=True` requests contiguous copies of
all fields; `contiguous=("genes", "values")` selects fields. Field-copy stream
dependencies are connected to the current CUDA stream before the batch is
returned. If your application hands Tensors to another stream, manage that
stream's dependencies and storage lifetime as usual.

## Bound work and close iterators

`prefetch` counts host output slots, `batch_size` counts samples per slot, and
`workers` counts native threads. A retained Tensor batch uses additional
application memory. `cache_bytes` budgets online input caching separately from
output prefetch; evaluate it after correctness with [planning](planning.md).

Only one iterator may be active. Wrap a loop in `with loading:` when it may
break or raise; context exit closes unfinished work. `close()` leaves the loader
reusable and retains a progress snapshot in `stats`. Exhausting the iterator
records final RunStats. Starting another iterator resets `stats` to `None`.

Use the [API reference](/python/api/dataloader.md) for parameter and error
contracts, or [the complete training example](/python/training/dataloader.md)
for a multi-dataset process preset and native sampler.
