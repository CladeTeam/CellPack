# Training DataLoader API

```python
from pycellpack import Accesses, ScDataset, ScDataLoader
```

These interfaces produce owned Tensor batches from corpus-local sample orders.
For a complete program and checked output, read [training batches](/python/docs/dataloader.md).
For NumPy ring views and global cell IDs, use the separate [native Loader](loader.md).

## Dataset and sample declarations

### ScDataset

`ScDataset(store, *, ids)` pairs a **2-D file-backed Store view** with one standard
ID per visible column. It keeps the Store and copies the ID map to read-only
uint32 storage. Keep the backing file immutable while using the loader.

| Input | Requirement |
| --- | --- |
| `store` | Its decode dtype equals its file dtype; memory streams are rejected |
| `ids` | One-dimensional NumPy integer array of length `store.shape[1]` |
| Kept IDs | Distinct integers in `[0, uint32.max)` within this dataset |
| Dropped columns | `-1` or `uint32.max` |
| Store views | Row selection and ordinary column subsets/reordering remain lazy; one stored column cannot contribute multiple kept dimensions |

Rows in sample orders address this view. For `store[[5, 2]]`, local row `0`
means stored row `5`. Choose output conversion on ScDataLoader rather than with
`Store.astype`. `ScDataset.from_anndata(adata, ids=..., layer=None)` selects an
existing Store from `X` or a named layer; it does not encode arrays or match names.

### Corpus Accesses

`Accesses(dataset_ids, rows)` borrows two parallel 1-D NumPy integer arrays.
The value object does not validate their contents. Starting an iterator copies
them to uint32 dataset IDs and uint64 local rows, checks equal lengths and
nonnegative ranges, and validates each pair against the corpus. Repeats and
order are retained. The corpus's total contributed rows must fit uint32 CellIds.

`len(accesses)` counts **samples**. This is the package-root Accesses, distinct
from [native `loader.Accesses`](access-functions.md), whose values are global
cell IDs. ScDataLoader also accepts the tuple `(dataset_ids, rows)`.

## Construct a loader

### Sampling and output

| Parameter | Default | Contract |
| --- | --- | --- |
| `corpus` | Required | Nonempty sequence of ScDataset objects; captured as a tuple |
| `accesses` | `None` | Corpus Accesses or its two-array tuple |
| `access_fn` | `None` | Native AccessFn; corpus counts are bound automatically; supply exactly one of accesses/access_fn |
| `out_ids` | `None` | Nonempty distinct standard IDs below `uint32.max`, in output order; omission requires identical dataset maps and selects their kept IDs |
| `process` | `None` | ProcessFn with declared OutputLayout; omission produces dense `X` |
| `dtype` | `None` | float32, or the ProcessFn's required input dtype; must be a supported native loader conversion |
| `default` | `0` | Typed fill for dimensions absent from a source; does not replace present sparse zeros |
| `drop_last` | `False` | Trim incomplete final batch before execution |

Preset token/configuration tables follow `out_ids` order. They do not use standard
IDs as table offsets. Output fields with a process are declared by its
[OutputLayout](process-functions.md).

### Execution and memory

| Parameter | Default | Unit / contract |
| --- | --- | --- |
| `batch_size` | `256` | Positive samples per output slot; bool is rejected |
| `prefetch` | `4` | Positive host output slots; bool is rejected |
| `workers` | `None` | Native threads; automatic value is `min(16, usable CPUs)`; explicit values bypass this automatic cap |
| `cache_bytes` | `0` | Nonnegative online input-cache budget; zero disables planning |
| `scratch_bytes` | `0` | Process scratch bytes per worker |
| `backend` | `None` | Probe io_uring, else pread; explicit `"uring"` or `"pread"` |
| `direct_io` | `None` | Probe O_DIRECT; `False` forces buffered input I/O |
| `**settings` | Empty | Individual LoaderConfig fields such as inflight, claim_batch, window, profile and cpus; config and slot_rows are rejected |

Output prefetch, input cache and returned Tensor allocations are separate
budgets. ScDataLoader derives native `slot_rows` from `batch_size`.

### Tensor delivery

| Parameter | Default | Contract |
| --- | --- | --- |
| `device` | `"cpu"` | CPU or CUDA PyTorch device, for example `"cuda:0"` |
| `contiguous` | `False` | True for all output fields, a field name or a sequence for selected fields; unknown names are rejected |

Iteration returns `dict[str, torch.Tensor]`. Built-in `X` has shape
`(actual_batch_rows, len(out_ids))`; process field shapes have batch rows prefixed.
CPU outputs own a packed clone and may be retained without holding native ring
slots. CUDA outputs own device storage after one packed H2D transfer. The copy
stream completes before host views are released. Field views may be strided;
selected fields or misaligned field layouts can require copies. Device field-copy
streams establish dependencies with the current CUDA stream before return.

## Iteration, epochs and completion

| Operation | Behavior |
| --- | --- |
| `len(loading)` | Batch count for the current order; does not start workers, import Torch or validate sample bounds |
| `iter(loading)` | Copies/validates an array order or binds its AccessFn, imports Torch, creates a fresh ring/run and resets stats to None |
| `loading.out_ids` | Read-only output-panel view |
| `loading.stats` | None before the first run or during a new run; final RunStats on exhaustion, progress snapshot after early close |
| `loading.close()` | Closes the active iterator; another run can be started afterward |
| `with loading:` | Closes the active iterator on exit, including break and exception paths |

Only one iterator may be active. The caller owns epoch policy: replace
`loading.accesses` or `loading.access_fn` **between** runs. Every iterator starts
from the beginning of that supplied order; there is no implicit reshuffle or
resume offset. Use a sliced order to resume. Do not mutate corpus or output
configuration after construction.

## Failures and validation timing

| Failure | Where to look |
| --- | --- |
| `TypeError` / `ValueError` during construction | Corpus, Store, ID maps, panel, process output declaration or execution settings |
| `TypeError` / `ValueError` on iterator start | Sample array shapes/types, dataset indices or dataset-local rows |
| `RuntimeError` for a second iterator | Exhaust or close the active iterator first |
| `ModuleNotFoundError` naming torch | Install PyTorch separately before iteration |
| I/O or native worker error during iteration | Source file state, backend, process callback and native error cause |

Completed Tensor batches retain independent storage after early close. A progress
snapshot is not a success receipt for the full order; only normal exhaustion
collects the final statistics.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/dataloader-members.md-->

:::
