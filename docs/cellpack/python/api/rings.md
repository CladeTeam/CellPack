<script setup>
import PythonRingLease from '../../.vitepress/theme/components/PythonRingLease.vue';
</script>

# Rings, batches, and runs API

A ring is reusable output storage. A Batch is a lease on one completed slot;
NumPy arrays exported from it also retain that lease. A Run starts production
and provides one rank's consumer. Read the [native loading guide](/python/docs/native-loader.md)
for a complete multi-output workflow.

```python
from pycellpack.loader import RingLayout, Ring, Batch, Consumer, Run
```

<DocChoices title="Find the contract you need" :items="[
  { label: 'Batch and exported views', when: 'Retain results or diagnose a full ring', detail: 'Release the Batch reference and finish every reader of its arrays.', href: '#batch-views-and-slot-leases' },
  { label: 'Run and Consumer', when: 'Wait for batches, statistics or cleanup', detail: 'Consume assigned slots before joining; distinguish completion from success.', href: '#consume-by-iteration-or-explicit-wait' },
  { label: 'RingLayout and Ring', when: 'Allocate pinned or shared output memory', detail: 'Derive the layout from the loader and keep data and control storage stable.', href: '#allocate-or-wrap-ring-memory' },
]" />

## Batch views and slot leases

Batch `b` occupies slot `b % slots` and belongs to consumer rank `b % world`.
All output schemas share that slot. `batch.rows` counts positions in this batch,
including a potentially short tail; it does not multiply by the schema count.

| Interface | Result / contract |
| --- | --- |
| `batch[name]` or `batch[index]` | Typed NumPy view, shape `(rows, row_bytes / itemsize)`; raw bytes if the output has no dtype |
| `batch.data` | The same view for a single-output batch; raises TypeError for multiple outputs |
| `batch.region(name)` | Raw uint8 view, shape `(rows, row_bytes)` |
| `batch.arrays()` | Dictionary of named views; every value retains the slot |
| `index`, `first_position`, `rows`, `generation` | Batch index, first sequence position, actual row count and run generation |
| `names`, `keys()`, iteration | Output names in layout order; `len(batch)` counts outputs |
| `release()` | Drop this Batch's own reference to the lease |
| `held` / `observer().held` | Whether this Batch or any exported array still retains the lease |

An exported array remains valid after `batch.release()` or `run.close()`.
Derived NumPy views and Tensors that share that array's storage also keep its
ownership chain alive. The released Batch itself can no longer export arrays
or expose lease-dependent fields such as `index` and `rows`; those calls raise
RuntimeError. `held`, `observer()` and output names remain available.

<PythonRingLease />

Copy a result before retaining it independently. Keeping `batch`, `batch["X"]`,
a slice, or a storage-sharing Tensor can prevent the slot from being reused.
When every slot is held, production waits. Increasing slot count only postpones
that limit; it does not make an unbounded list of views safe.

### Observe the difference between release and copy

This complete CPU example creates a tiny file and one output slot. It requires
NumPy and the core package, and creates or replaces `lease.cpk`:

```python
import numpy as np
import pycellpack as cp
from pycellpack import loader

x = np.array([[1, 2], [3, 4]], np.float32)
store = cp.write("lease.cpk", x, workers=1)
source = loader.DataSource([(store, [0, 1])], vocab=2, direct_io=False)
loading = loader.Loader(source, workers=1, backend="pread", slot_rows=2)
with loading.start([1, 0], slots=1) as run:
    batch = run.wait(0)
    assert batch is not None
    observed = batch.observer()
    view = batch["X"]
    saved = view.copy()
    batch.release()
    assert observed.held                    # view still retains the slot
    np.testing.assert_array_equal(view, [[3, 4], [1, 2]])
    del view
    assert not observed.held                # saved owns independent storage
    assert run.wait(1) is None              # no second batch
    stats = run.join()
np.testing.assert_array_equal(saved, x[[1, 0]])
```

The `LeaseObserver` returned by `batch.observer()` holds a weak reference;
obtain it from a Batch rather than importing or constructing it directly.
Observing a lease does not prolong it.
For an asynchronous device transfer, retain the host array and its sharing
Tensor until the device has finished reading. Submission alone is not completion.
The [training DataLoader](dataloader.md) manages its own transfer path.

## Consume by iteration or explicit wait

| Consumption path | Ownership on advance / completion |
| --- | --- |
| `for batch in run` | Consumes `run.rank`; advancing releases the preceding Batch reference |
| `for batch in consumer` | Consumes that rank; advancing releases its preceding Batch reference |
| `run.wait(b)` / `consumer.wait(b)` | Returns one Batch or None past the end; caller releases or drops each returned Batch |
| `with batch:` | Releases that Batch reference on context exit |
| `with consumer:` | Closes the consumer on exit and records whether its assigned sequence was taken |

Iteration releases the previous **Batch**, including when you stored the Batch
object in a list. It does not release arrays already exported from it. Use one
consumption path per rank and consume batches in increasing order; `world` is
rank partitioning, not broadcast to several independent readers of one batch.
For `world=2`, rank 0 takes batches `0, 2, 4, …`, while rank 1 takes `1, 3, 5, …`.

`run.consumer(rank, idle=None)` opens another rank's consumer tied to this run's
generation and failure state. `ring.consumer(rank=0, idle=None,
watch_producer=True)` joins the current or next run, including from another
process with properly shared storage. `idle()` runs between blocked-wait polls
and may raise. Producer watching detects a departed producer; it is not a wait
deadline. Closing a consumer releases its iteration-held Batch and records
completion; exported arrays still hold their slots. An early departure can make
the producer fail because the assigned output has no remaining consumer.

## Finish, cancel, and reuse

Drain output before `join()`, unless another thread/process drains it or the
entire run fits in available slots. Waiting for production while retaining the
slots it needs can prevent completion. The binding detects impossible local
join stalls and failed/departed shared consumers, cancels the run, and raises
RuntimeError; this does not replace correct consumption order.

| Operation / state | Meaning |
| --- | --- |
| `snapshot()` | Current published RunStats; requires an open Run |
| `join()` | Wait for the producer, return final RunStats or propagate its failure; also available after close |
| `finished` | Producer stopped through success, failure or cancellation; True after close |
| `succeeded` | Producer finished successfully; remains queryable after close |
| `wait_finished(timeout_ms=50)` | Poll completion without consuming; timeout is 0–1000 ms, with Python signal checks |
| `cancel()` | Request cancellation; the ring is permanently aborted |
| `close()` / Run context exit | Release the iteration-held Batch, cancel unfinished work, wait for native cleanup without holding the GIL |
| `ring.abort()` | Permanently abort the ring for every participant; blocked waits fail |

`close()` preserves exported arrays and caches the outcome for later `join()`.
It does not itself return statistics or substitute for inspecting failure.
After close, active operations including `snapshot`, `wait`, `consumer`,
`generation` and `batches` require an open run; iteration stops. `ring`, `rank`,
`world`, `finished` and `succeeded` remain available.

A successful ring can serve another run once all ready output has been consumed,
every lease has been released, the previous Run is closed/dropped, and Consumers
tied to that Run have been dropped. `join()` alone does not release the loader
reservation; even a closed Consumer can retain its run state. A cancelled or aborted ring
cannot be reset: allocate a new ring. Likewise, closing unfinished work cancels
it rather than preparing its ring for reuse.

Invalid layouts, ranks, sequences and plans raise ValueError; I/O failures raise
OSError; cancellation, busy state and worker failures raise RuntimeError. Errors
can surface during iteration, explicit wait or join. Discard incomplete results
from a failing run and retain its original cause.

## Allocate or wrap ring memory

Prefer `loading.ring_layout(slots=4, world=1)` and `loading.ring(...)`: they derive
the output names, dtypes and record sizes from this loader. A supplied ring must
match those declarations and the loader's `slot_rows`. With `start(ring=ring)`,
the ring determines slots and world; do not specify them again.

`RingLayout(slots, slot_rows, outputs, world=1)` is a picklable description,
without storage. Counts must be positive; `outputs` contains distinct nonempty
names and `(name, dtype, row_bytes)` tuples in output order. Numeric dtypes are
native-endian and must divide `row_bytes`; None exposes raw bytes.

| Layout field | Unit / purpose |
| --- | --- |
| `slot_rows`, `slots`, `world` | Positions per slot, buffering capacity and consumer-rank count |
| `row_bytes`, `offsets` | Bytes per position for each output; output offsets within a slot |
| `slot_stride` | Bytes between slot starts, aligned to 4096 bytes |
| `data_bytes` | Data storage for all slots, including final padding |
| `data_min_bytes` | Minimum data storage when final-slot padding is omitted |
| `control_bytes` | Control-storage bytes, including slot headers and participant records |

### External and shared storage

`Ring(layout, data=None, control=None, initialize=True)` allocates native storage
when data is omitted. With external storage:

| Argument | Requirements |
| --- | --- |
| `data` | Writable C-contiguous 1-D uint8 NumPy array; 8-byte-aligned; length between `data_min_bytes` and `data_bytes`, inclusive |
| `control` | Writable C-contiguous 1-D uint8 array; 64-byte-aligned; at least `control_bytes` bytes; disjoint from data; requires data |
| `initialize=True` | Initialize fresh control storage; never reinitialize an active ring |
| `initialize=False` | Attach supplied data and control to an already initialized compatible ring |

The ring retains the Python array owners. Keep their addresses and extents stable
and let the slot protocol govern all access; do not resize, free shared mappings,
or independently overwrite their contents while used. `Ring.shared` reports
that caller-owned control was supplied. It does **not** establish that the arrays
are shared between processes. Use [SharedRing](distributed.md) for the supported
shared-memory construction and descriptor workflow.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/rings-members.md-->

:::
