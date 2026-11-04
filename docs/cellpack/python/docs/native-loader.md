# Native NumPy loading

Use `loader.Loader` for named NumPy outputs, explicit memory layouts or reusable
plans. You provide immutable matrix files, column identity and sample order;
native workers decode, map and fill output slots.

<DocChoices title="Choose an entry point" :items="[
  { label: 'loader.load', when: 'You need one dense full-vocabulary array', detail: 'A one-shot float32 result; optionally reuse an exact output allocation.', href: '#read-one-array' },
  { label: 'loader.Loader', when: 'You need ordered batches or multiple schemas', detail: 'Stream views from a bounded ring and copy the results you retain.', href: '#register-sources-and-outputs' },
  { label: 'ScDataLoader', when: 'You need Tensor dictionaries for training', detail: 'Use the training wrapper and its CPU/CUDA transfer path.', href: '/python/docs/dataloader.html' },
]" />

<DocDiagram name="python-native" />

## Register sources and outputs

Run this complete example in a writable directory. It needs the core package and
NumPy, creates `native.cpk`, and deliberately uses buffered `pread` for a small
correctness check.

```python
import numpy as np
import pycellpack as cp
from pycellpack import loader

x = np.arange(24, dtype=np.float32).reshape(6, 4)
cp.write("native.cpk", x, workers=1)
source = loader.DataSource(
    [("native.cpk", np.arange(4, dtype=np.uint32))],
    vocab=5, direct_io=False,
)
outputs = {"pair": loader.OutputSchema([3, 0], np.float64)}
loading = loader.Loader(source, outputs, workers=1, backend="pread", slot_rows=2)
requested = np.array([5, 1, 5, 0], np.uint32)
retained = []
with loading.start(requested, slots=2) as run:
    for batch in run:
        retained.append(batch["pair"].copy())
    stats = run.join()
actual = np.concatenate(retained)
np.testing.assert_array_equal(actual, x[requested][:, [3, 0]])
assert [part.shape for part in retained] == [(2, 2), (2, 2)]
print(actual)
```

```text
[[23. 20.]
 [ 7.  4.]
 [23. 20.]
 [ 3.  0.]]
```

The source map says stored column `c` has standard ID `c`. The schema asks for
standard IDs `[3, 0]` in that output order, converted to float64. Cell `5` is
requested twice and therefore appears twice in the result. The fifth vocabulary
ID exists in the declared ID space but has no column in this matrix.

Sources can register a path, a file-backed File region, or a row-only Store view.
Each registered matrix has its own stored-column → standard-ID map. Standard IDs
are application identities; PyCellPack does not infer them from gene names.

Global cell IDs count contributed source rows in registration order. For a Store
view selecting stored rows `[5, 2]`, global cells `0` and `1` refer to those two
rows. Adding another matrix does not change an already constructed Loader's native
source snapshot; construct a new Loader to include it.

## Missing columns differ from zero values

`default` fills columns absent from a source, while a present sparse column without
an entry remains zero. For the first source row `[0, 1, 2, 3]`:

| Schema | Result | Why |
| --- | --- | --- |
| `OutputSchema([4, 0], default=-1)` | `[-1, 0]` | ID 4 is absent; ID 0 is present with value zero |
| `OutputSchema([3, 0])` | `[3, 0]` | Existing values follow requested output order |

With `outputs=None`, the loader uses float32 over the full vocabulary and names
the result `X`. An explicit mapping assigns names, while a schema sequence uses
schema names or assigned IDs. Choose [process functions](process-functions.md)
when each position must produce a custom fixed-size record instead of a dense row.

## Multiple outputs count as multiple accesses

Every position executes all schemas. With two schemas, accesses are ordered
`position0/schema0`, `position0/schema1`, `position1/schema0`, `position1/schema1`.

<DocDiagram name="access-order" />

This continues from the same `source`:

```python
schemas = {
    "left": loader.OutputSchema([0, 1]),
    "right": loader.OutputSchema([3, 2]),
}
multi = loader.Loader(source, schemas, workers=1, backend="pread", slot_rows=2)
cells = loader.Accesses.replicate(np.array([5, 1], np.uint32), schemas=2)
assert cells.fill().tolist() == [5, 5, 1, 1]
with multi.start(cells) as run:
    for batch in run:
        np.testing.assert_array_equal(batch["left"], [[20, 21], [4, 5]])
        np.testing.assert_array_equal(batch["right"], [[23, 22], [7, 6]])
    run.join()
```

Four accesses produce **two positions in one batch**. Every named output has two
rows. In contrast, passing the bare array `[5, 1]` describes one position: `left`
loads cell 5 and `right` loads cell 1. It does not imply replication. Use `offset`
for paired cells and [access functions](access-functions.md) for computed orders.

## Slot leases and backpressure

<DocDiagram name="slot-lifecycle" />

A Batch exposes views into reusable native ring memory. Iteration releases its
previous Batch reference, but each exported NumPy view or slice keeps that slot
leased. Keeping uncopied output from every slot can stop production.

Read a view immediately, or copy before retention as the first example does.
`batch.release()` releases the Batch's own reference; it does not revoke exported
arrays. A `with batch:` block releases that Batch reference on exit.

A nonblocking device copy is still a reader. Retain its host array and any host
Tensor until the copy completes. See [rings and lifecycle](/python/api/rings.md)
for caller-owned memory and manual consumers; ScDataLoader manages its own
Tensor transfer path.

## Completion and failure

<DocDiagram name="run-completion" />

Consume batches before `join()`, unless another consumer drains them or every
output fits in the ring. Waiting for the producer while retaining the slots it
needs can prevent completion. `join()` returns final RunStats after native work
finishes; `snapshot()` observes progress without waiting.

Context exit calls `close()`, cancelling unfinished work and waiting for cleanup.
`finished` includes success, failure and cancellation; use `succeeded` to identify
success. Existing exported arrays remain valid after close. Validation, I/O and
execution errors propagate during setup, iteration or joining; discard incomplete
output after failure. The [lifecycle reference](/python/api/rings.md) explains
which rings can be reused.

## Read one array

For a full-vocabulary float32 result, this continues from `source` without
requiring a schema or an explicit run:

```python
destination = np.empty((2, source.vocab), np.float32)
result = loader.load(
    source, cells=[5, 0], out=destination, workers=1, backend="pread",
)
assert result is destination
np.testing.assert_array_equal(result, [[20, 21, 22, 23, 0], [0, 1, 2, 3, 0]])
```

The fifth column takes the built-in zero default because it is absent from the
matrix. `out` must have the exact shape, be writable C-contiguous float32, and be
8-byte aligned. This convenience allocates one slot covering the whole selection;
use streaming Loader batches when output memory must stay bounded.

Continue with [planning](planning.md) after checking values and order. The
[native loader API](/python/api/loader.md) specifies configuration defaults,
accepted source views, `start` arguments and errors.
