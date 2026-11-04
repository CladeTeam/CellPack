# Shared and distributed API

These interfaces share native output memory between processes on **one host**.
Use [the distributed guide](/python/docs/distributed.md) for a two-rank example
with expected batches. Output views obey the separate [ring lease contract](rings.md).

```python
from pycellpack.loader import SharedRing, DistributedLoader, DistributedRun
```

<DocChoices title="Choose the orchestration layer" :items="[
  { label: 'DistributedLoader.start', when: 'An initialized torch.distributed group owns the ranks.', detail: 'Collective startup creates one CPU producer and one consumer per group rank.', href: '#collective-startup' },
  { label: 'SharedRing.create / attach', when: 'Your application owns process startup and descriptor exchange.', detail: 'Share a ring layout and its backing regions; launch a Loader producer separately.', href: '#sharedring-and-descriptor-lifetime' },
]" />

## Collective startup

`DistributedLoader(loader=None, *, transport="auto")` stores a Loader or a
zero-argument factory on **group rank zero**. Other ranks normally pass nothing.
The factory runs on rank zero during `start`; the resulting Loader is pickled
and rebuilt in the producer. The factory itself is not sent there. Matrix paths,
ProcessFn libraries/configuration and access-function owners must be usable or
reconstructible in that process.

| `start` argument | Default | Contract |
| --- | --- | --- |
| `accesses` | None | Rank zero supplies a 1-D integer array/list or native Accesses; global cell IDs, one per schema per position |
| `plan` | None | Rank zero supplies a Plan, plan-file path, PlanOptions or cache byte budget; None disables planned caching |
| `slots_per_rank` | 2 | Positive output depth per rank; total slots are `slots_per_rank × world` |
| `drop_last` | False | Trim an ordinary stored array to complete rounds of every rank's full batch |
| `group` | None | Initialized torch.distributed group; None uses the default group |

All members call `start` in matching collective order. It gathers hostnames,
rejects a group spanning hosts, launches the producer on group rank zero,
broadcasts descriptors and gathers attachment failures before returning a
DistributedRun. Subgroup rank zero may differ from global rank zero.
Only group rank zero's loader, accesses, plan and ring sizing are used. Peers
should supply no loader, accesses or plan. The producer uses the launching Python
interpreter and initializes neither CUDA nor a torch.distributed group.

Arrays normalize to contiguous uint32 IDs, retaining order and repeats. Access
length must be divisible by schema count. With `S` schemas, `drop_last=True`
keeps `floor(length / (slot_rows × world × S)) × (slot_rows × world × S)`
accesses. It rejects native Accesses, including `Accesses.array(...)`, because
that object is not the ordinary array trimming path. A fixed plan must match
the sequence actually executed, including trimming.

## DistributedRun results and lifecycle

<DocDiagram name="distributed" />

| Interface | Result / behavior |
| --- | --- |
| `rank`, `world` | Group rank and consumer count |
| `producer_pid` | Dedicated producer's process ID |
| `len(run)` | Batches assigned to this rank, including a partial tail it owns |
| Iteration | Native Batches in this rank's order; advancing releases the preceding Batch reference |
| `wait(b)` | Acquire **global batch index** b; require `b % world == rank`; caller releases/drops each returned Batch |
| `finished` | True only after producer success; False while pending; failure/cancellation raises RuntimeError |
| `join()` | Wait without consuming and return the producer's complete RunStats; failure/cancellation raises RuntimeError |
| `cancel()` | Cancel the active run for all ranks and abort its shared ring |
| `close(cancel=True)` | Close consumer and local resources; default cancels, False waits for success/failure |

For local step `j`, the manual wait index is `rank + j × world`, not `j`.
Consume each assigned global batch once, in increasing order. `wait` past the
end raises RuntimeError rather than returning None; bound manual loops with
`range(len(run))`. For `B` total batches, a rank has
`max(0, ceil((B - rank) / world))` batches.

`join` neither drains output nor acts as a group barrier. It may wait for other
ranks to release slots. Statistics describe the whole producer run: do not sum
the same counts returned to every rank. `finished` differs from native
`Run.finished`, which also counts failure and cancellation as completion.

Successful context exit after all assigned batches were taken calls
`close(cancel=False)`. Early exit or an exception cancels the run. `close` is
idempotent; all ranks close mappings, and rank zero waits for or stops the
producer. After close, iteration stops and `join`, `wait` and `finished` require
an open run. Capture statistics before closing. Already exported arrays retain
their mappings and their leases while production is active.

## SharedRing and descriptor lifetime

| Interface | Contract |
| --- | --- |
| `create(layout, *, transport="auto", extra=None)` | Create and initialize a ring; derive layout from the intended Loader |
| `extra` | Additional named byte sizes, clamped to at least one; cannot replace `control` or `data` |
| `descriptor` | Picklable layout/transport/region descriptors, not copied output bytes |
| `attach(descriptor, *, fds=None)` | Map existing regions without reinitializing control; attaching process does not own/unlink names |
| `inheritable()` | Return `(descriptor, pass_fds)` for child subprocess.Popen with memfd regions |
| `sharing` / `stop_sharing()` | Whether peers can fetch memfd descriptors / stop serving after attachment |
| `ring`, `layout` | Native Ring and unchanged layout |
| `consumer(rank=0, **kwargs)` | Open that rank's current/next ring consumer; kwargs follow Ring.consumer |
| `region(name)` | Memoryview over a named region's bytes |
| `region_fd(name)` | OS descriptor, valid while this SharedRing owns it |
| `close()` / context exit | Stop sharing, close FDs, unlink named regions only when this process created them |

Keep the creator open until peers attach. memfd peers without inherited FDs
receive duplicates over a same-user Unix socket. `DistributedLoader.start`
manages this handshake and stops serving after every peer attached or failed.
A saved descriptor does not preserve closed/unlinked regions as persistent data.

Close consumers and finish active runs before releasing resources. SharedRing
close is memory cleanup, not cancellation or completion signaling. Existing
NumPy arrays retain their mappings; names may already be unlinked and new
attachment unavailable. Do not close `region_fd` yourself or retain it past the
SharedRing's lifetime. See [rings](rings.md) for slot leases and consumer cleanup.

## Transport selection and failures

| Transport | Allocation / attachment |
| --- | --- |
| `shm` | Named mmap files under `/dev/shm`, limited by mount capacity |
| `memfd` | Anonymous memory files; FD inheritance or same-user transfer; subject to memory/cgroup limits |
| `file` | Mapped files in `PYCELLPACK_SHM_DIR` or system temporary directory; dirty pages may be written back |

`transport="auto"` consults `PYCELLPACK_SHM`, then tries shm → memfd → file if
still automatic. Explicit transport names take precedence over the environment
and try only that backend. Automatic selection leaves filesystem headroom;
creation reserves regions before mapping. For file transport,
`PYCELLPACK_SHM_DIR` must name an existing writable directory.

Unavailable/uninitialized process groups raise RuntimeError; cross-host groups
raise ValueError. Rank-zero startup errors are broadcast as RuntimeError, and
attachment failures are gathered before ranks leave startup. Producer exceptions
or unexpected exits publish a cause through shared status. A departed consumer
makes the producer fail rather than wait indefinitely. These mechanisms do not
replace matching training collectives or correct lease release. Direct SharedRing
calls can raise ValueError for invalid declarations/descriptors and OSError for
allocation or attachment failures.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/distributed-members.md-->

:::
