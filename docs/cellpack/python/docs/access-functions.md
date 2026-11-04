# Access sequences and native samplers

Choose sample order before starting a run. A native sequence answers “which global
cell does access `k` load?” without requiring the entire order to be stored.
For a training corpus that already has explicit dataset/row pairs, use
[training Accesses](dataloader.md) instead.

<DocChoices title="Choose how to describe the order" :items="[
  { label: 'Stored Accesses', when: 'You already have an explicit order.', detail: 'Snapshot a uint32 vector. Repeats and requested order are preserved.', href: '#inspect-and-compose-a-stored-sequence' },
  { label: 'shuffle or subset', when: 'You need deterministic shuffled epochs.', detail: 'Generate a seeded order and optionally trade locality against mixing.', href: '#generate-shuffled-epochs' },
  { label: 'AccessFn', when: 'Your policy is native or too large to materialize.', detail: 'Bind a range callback with immutable seed, epoch and policy configuration.', href: '#bind-a-native-range-function' },
]" />

## Inspect and compose a stored sequence

This program needs NumPy and the core package, with no compiler or PyTorch:

```python
import numpy as np
from pycellpack.loader import Accesses

positions = Accesses.array(np.array([5, 1, 5], dtype=np.uint32))
np.testing.assert_array_equal(positions.fill(), [5, 1, 5])
np.testing.assert_array_equal(positions.slice(1, 2).fill(), [1, 5])

paired = Accesses.replicate(positions, schemas=2)
np.testing.assert_array_equal(paired.fill(), [5, 5, 1, 1, 5, 5])
assert len(paired) == 6

out = np.empty(2, dtype=np.uint32)
assert positions.fill(start=1, out=out) is out
np.testing.assert_array_equal(out, [1, 5])
```

The three positions become six accesses for a two-schema Loader. Accesses are
ordered **position first, schema second**: `k // 2` chooses the position and
`k % 2` the schema. Pass `paired` to a two-schema Loader backed by at least six
source cells. It does not infer replication from a three-element input.

Use `Accesses.offset(positions, offsets=[0, 6])` for paired registrations: the
same positions then produce `[5, 11, 1, 7, 5, 11]`. This example requires a source
with at least twelve global cells. Offsets are cell-ID offsets, not column maps.

`Accesses.array` snapshots a native-endian, aligned, contiguous **uint32 ndarray**.
In contrast, passing an ordinary integer array directly to `Loader.start` lets
the Python loader normalize its dtype and check the uint32 range first.
`slice` remains lazy; `fill` allocates an inspected interval, so avoid calling
`fill()` over a sequence too large for memory.

## Generate shuffled epochs

```python
shuffled = Accesses.shuffle(cells=6, epochs=2, seed=7)
assert len(shuffled) == 12
for epoch in range(2):
    np.testing.assert_array_equal(
        np.sort(shuffled.fill(start=epoch * 6, count=6)), np.arange(6),
    )

subset = Accesses.subset(np.array([5, 1, 5], np.uint32), epochs=2, seed=7)
np.testing.assert_array_equal(np.sort(subset.fill(count=3)), [1, 5, 5])
```

Each epoch visits every input entry once; a subset can contain repeated cell IDs.
The assertions check membership rather than prescribing a particular shuffled
order. Reusing the same seed and options reproduces the order. `block` and
`group` control locality: blocks of consecutive entries are shuffled, then entries
within groups of those shuffled blocks are mixed. The [API](../api/access-functions.md#stored-and-computed-sequences)
gives the valid ranges.

## Bind a native range function

This self-contained example compiles C++ and inspects a generated interval. It
requires a compatible native compiler; PyTorch and Numba are unnecessary.

```python
from pycellpack import AccessFn

code = r'''
#include "access_fn_cabi.h"
CELLPACK_ACCESS_EXTERN CELLPACK_ACCESS_EXPORT int32_t generate_accesses(
    void *, uint64_t start, uint64_t count, uint32_t *cells) {
    for (uint64_t i = 0; i < count; ++i)
        cells[i] = (start + i) % 6;
    return 0;
}
'''
sampling = AccessFn.from_source(code=code, length=12, kind="global")
np.testing.assert_array_equal(sampling.fill(start=4, count=4), [4, 5, 0, 1])
np.testing.assert_array_equal(sampling.fill(start=4, count=4), [4, 5, 0, 1])
```

The sequence repeats global cells 0 through 5 twice. A single-schema native
Loader with at least six registered cells accepts `loading.start(sampling)`.
For several schemas, bind and then compose the native sequence with `replicate`
or generate one ID per schema directly. Planning with a retained fixed Plan may
materialize a computed sequence to check its exact values; online planning keeps
range generation ahead of execution.

## Choose global or corpus addressing

| Callback kind | Callback writes | How it becomes a global cell |
| --- | --- | --- |
| `global` | One uint32 cell ID per access | Already addresses concatenated registered rows |
| `corpus` | uint32 dataset IDs and uint64 local rows | Bind dataset view row counts; add each dataset's first global cell |

ScDataLoader accepts `access_fn=sampling` instead of corpus `accesses`, and binds
its dataset counts automatically. Native Loader and PageTrace entry points also
bind counts from their source. For explicit inspection of a corpus callback,
use `sampling.fill(counts=[...])` or `sampling.bind(counts=[...])`.

For example, counts `[2, 3]` assign dataset 0 global cells `[0, 1]` and dataset 1
`[2, 3, 4]`. Dataset 1/local row 2 therefore means global cell 4. Rows refer to
the current Store view, rather than the unsliced file.

Supply exactly one of `accesses` or `access_fn` to ScDataLoader. Seeds, epoch
numbers and weights belong to your sampler configuration. Iterating a Loader
again does not silently advance a callback's epoch; construct the next configured
sampler explicitly.

## Keep generation independent of call order

<DocDiagram name="access-callback" />

Fill exactly `count` results for `[start, start + count)`, return zero on success,
and support concurrent, repeated and out-of-order calls. Determine each result
from its access index and immutable configuration. A mutable random generator
advanced once per call cannot satisfy this contract.

Keep callback code and context alive; never retain output pointers or let an
exception cross the C boundary. `user_data` passes context, `owner` retains
additional storage, and `key` or `identity` must describe every configuration
value affecting results. Keep those values immutable while a sequence or plan
uses them. Pointer addresses do not identify configuration values.

Use `from_library` for an existing exported symbol, `from_source` for cached
compilation, or `from_cfunc` for optional Numba integration. Reconstruction recipes
can cross processes when their configuration is picklable; raw addresses and
process-local owners require reconstruction by the application. See the
[API contract](../api/access-functions.md#callback-abi-and-lifetime) for exact
signatures, return statuses and ownership, or the
[sample-order program](/python/training/dataloader.md) for training integration.
