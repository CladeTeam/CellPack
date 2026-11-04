# Shared and distributed loading

Use `DistributedLoader` when several ranks on **one host** should share one
native CPU producer. Each rank receives different batches from a shared ring.
The producer rebuilds a lower-level Loader; it does not wrap ScDataLoader or
provide a multi-host transport.

<DocChoices title="Choose who manages the processes" :items="[
  { label: 'DistributedLoader', when: 'You already have a torch.distributed group on one host.', detail: 'Start collectively; the wrapper launches one CPU producer and attaches every rank.', href: '#run-a-two-rank-cpu-example' },
  { label: 'SharedRing', when: 'You own producer and consumer process orchestration.', detail: 'Exchange a ring descriptor and attach consumers explicitly.', href: '/python/api/distributed.html#sharedring-and-descriptor-lifetime' },
]" />

<DocDiagram name="distributed" />

## Run a two-rank CPU example

Install [PyCellPack from PyPI](installation.md) and PyTorch in the same project.
Save the program below as `shared-loader.py` in a writable directory, then launch
**two ranks on the same host**:

```sh
uv run torchrun --standalone --nproc-per-node=2 shared-loader.py
```

It uses a CPU `gloo` process group and creates or replaces `shared-example.cpk`.
Rank zero constructs the file and Loader inside the startup factory, so setup
errors can follow the collective startup path. All ranks need the same package
environment, and the producer must be able to reopen that file.

::: details Complete two-rank CPU program

```python
import numpy as np
import torch.distributed as dist
import pycellpack as cp
from pycellpack import loader


def make_loader():
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    store = cp.write("shared-example.cpk", x, workers=1, atomic=True)
    source = loader.DataSource([(store, np.arange(4, dtype=np.uint32))],
                               vocab=4, direct_io=False)
    return loader.Loader(source, workers=1, backend="pread", slot_rows=2)


def main():
    dist.init_process_group("gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 2
        requested = np.array([3, 1, 0, 2, 3, 0, 1, 2, 0, 3], np.uint32)
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        shared = loader.DistributedLoader(make_loader if rank == 0 else None)
        with shared.start(requested if rank == 0 else None,
                          slots_per_rank=2, drop_last=True) as run:
            assert len(run) == 2
            for step, batch in enumerate(run):
                global_batch = rank + step * world
                assert batch.index == global_batch
                local = batch["X"].copy()
                ids = requested[global_batch * 2:(global_batch + 1) * 2]
                np.testing.assert_array_equal(local, expected[ids])
                assert local.shape == (2, 4)
            stats = run.join()
            assert stats.accesses == 8
        print(f"rank {rank}: 2 batches, 4 rows")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

:::

The two result lines may appear in either order:

```text
rank 0: 2 batches, 4 rows
rank 1: 2 batches, 4 rows
```

A factory runs on **group rank zero**, then its returned Loader is pickled to
one dedicated CPU producer. The producer initializes neither CUDA nor the
training group. Keep matrix paths accessible and callback configuration
reconstructible there; see [process-function reconstruction](/python/api/process-functions.md#compilation-and-reconstruction).

## Understand batch partitioning

Batch `b` belongs to group rank `b % world`. This is partitioning, not a
broadcast. With one schema and two rows per slot, the example delivers:

| Global batch | Requested cells | Consumer |
| --- | --- | --- |
| 0 | `[3, 1]` | Rank 0, step 0 |
| 1 | `[0, 2]` | Rank 1, step 0 |
| 2 | `[3, 0]` | Rank 0, step 1 |
| 3 | `[1, 2]` | Rank 1, step 1 |

`drop_last=True` trims the final `[0, 3]`: a complete round needs
`slot_rows × world × schemas` accesses. Without trimming, that tail becomes a
fifth batch on rank zero, giving ranks different step counts. Keep training
collectives consistent with the chosen tail policy. Trimming accepts ordinary
stored arrays; native Accesses require an explicit length matching your rounds.

For multiple schemas, expand positions into [position-first, schema-second
accesses](access-functions.md#inspect-and-compose-a-stored-sequence) before
starting. All named outputs share a batch's rows and slot lifetime.

## Finish readers before joining

Batches follow the [native lease rules](/python/api/rings.md#batch-views-and-slot-leases).
Iteration releases its preceding Batch reference; exported arrays and sharing
Tensors keep the slot leased. Copy retained data, and keep host storage alive
until asynchronous device reads finish.

Drain every rank's assigned batches before `join()`. It waits for the producer
and returns **whole-run** statistics, not just this rank's counts. A rank that
stops consuming can block production while another rank waits in join.
Normal context exit after complete consumption waits and closes local resources.
An exception or early exit cancels the active run for all ranks. Producer failures
reach peers through shared status; preserve the reported cause. Save statistics
before leaving the context.

## Select a transport when needed

Default `transport="auto"` tries `/dev/shm`, then anonymous memfd memory, then
mapped files. `PYCELLPACK_SHM` overrides that automatic choice; an explicit
`"shm"`, `"memfd"` or `"file"` takes precedence and tries only that transport.
`PYCELLPACK_SHM_DIR` selects an existing writable directory for file-backed
regions. memfd avoids the `/dev/shm` mount size limit while remaining subject
to memory limits.

For custom orchestration, create SharedRing, exchange its descriptor, attach
peers while its memory is available, and close consumers before mappings.
Keep memfd descriptor sharing open until all peers attach. See the
[distributed API](/python/api/distributed.md) for exact startup arguments,
manual wait indices, descriptor lifetime and failure behavior.
