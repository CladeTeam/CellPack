"""One host's ranks fed by a dedicated producer process: any `Loader` runs
in it over a `SharedRing`, and every rank consumes its share.

`DistributedLoader.start` is collective. Rank zero builds the ring, launches
`pycellpack.loader._producer` and broadcasts the ring's descriptor and the
producer's pid; the other ranks attach and register as their rank's consumer.
One status region (a state, a length and a JSON payload under an advisory
lock) carries the producer's outcome, so every rank raises with the same
reason, and a rank that dies or leaves early makes the producer stop instead
of waiting for it.
"""

from __future__ import annotations

import fcntl
import json
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._shared import SharedRing

if TYPE_CHECKING:
    from typing_extensions import Self

# Producer states, the fixed size of the status region and its header: a
# state and the byte length of the JSON payload that follows it.
_RUNNING, _SUCCESS, _ERROR, _CANCELLED = range(4)
_STATUS_BYTES = 64 * 1024
_STATUS_HEADER = struct.Struct("<II")


class _Status:
    """The producer's outcome, shared by every rank: a state and a JSON
    payload (its statistics or its error), protected by an advisory lock."""

    def __init__(self, shared: SharedRing):
        """The shared status region; every access takes an advisory lock on
        its descriptor (shared for reads, exclusive for writes), and threads
        of one process also share `_lock`."""
        self.buffer = shared.region("status")
        self.fd = shared.region_fd("status")
        self._lock = shared._status_lock
        if len(self.buffer) < _STATUS_HEADER.size + len(json.dumps({"message": ""})):
            raise ValueError("producer status region is too small")

    def read(self):
        """The recorded state and its payload (`_RUNNING` and an empty dict
        while the producer is still going). Raises RuntimeError when the region
        holds no valid state or length."""
        with self._lock:
            fcntl.lockf(self.fd, fcntl.LOCK_SH)
            try:
                state, length = _STATUS_HEADER.unpack_from(self.buffer)
                if state not in (_RUNNING, _SUCCESS, _ERROR, _CANCELLED) or length > len(self.buffer) - _STATUS_HEADER.size:
                    raise RuntimeError("invalid producer status")
                start = _STATUS_HEADER.size
                return state, json.loads(bytes(self.buffer[start:start + length])) if length else {}
            finally:
                fcntl.lockf(self.fd, fcntl.LOCK_UN)

    def publish(self, state, payload) -> bool:
        """Records the outcome unless one is recorded already."""
        encoded = json.dumps(payload, ensure_ascii=True).encode()
        room = len(self.buffer) - _STATUS_HEADER.size
        if len(encoded) > room:
            if state == _SUCCESS:
                state = _ERROR
                payload = {"message": "producer statistics exceed the shared status capacity"}
            message = str(payload.get("message", "producer failed"))
            lo, hi = 0, len(message)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(json.dumps({"message": message[:mid]}, ensure_ascii=True).encode()) <= room:
                    lo = mid
                else:
                    hi = mid - 1
            encoded = json.dumps({"message": message[:lo]}, ensure_ascii=True).encode()
        with self._lock:
            # POSIX record locks are per process, including inherited or
            # SCM_RIGHTS memfd descriptors; flock would share one lock owner.
            fcntl.lockf(self.fd, fcntl.LOCK_EX)
            try:
                if _STATUS_HEADER.unpack_from(self.buffer)[0] != _RUNNING:
                    return False
                self.buffer[_STATUS_HEADER.size:_STATUS_HEADER.size + len(encoded)] = encoded
                _STATUS_HEADER.pack_into(self.buffer, 0, state, len(encoded))
                return True
            finally:
                fcntl.lockf(self.fd, fcntl.LOCK_UN)


class DistributedLoader:
    """A `Loader` run by a dedicated producer process for every rank of a
    torch.distributed group on one host.

    Group rank zero gives the loader (any data source, outputs, process
    functions and configuration; it is pickled to the producer, which
    rebuilds it), or a function returning it, called in `start` so that its
    failures reach every rank; the other ranks give nothing. `start` is collective: rank
    zero launches the producer over a `SharedRing` of `slots_per_rank`
    slots per rank, and each rank iterates its batches (`b % world == rank`)
    as typed `Batch`es. No CUDA work runs in the producer. `transport`
    picks how the ring's memory is shared (see `SharedRing`).
    """

    def __init__(self, loader=None, *, transport="auto"):
        """`loader` is used, or called in `start`, on group rank zero only; the
        other ranks pass nothing. `transport` picks how the ring's memory is
        shared (see `SharedRing`)."""
        self.loader = loader
        self.transport = transport

    def __repr__(self) -> str:
        from reprlib import repr as short_repr

        source = "factory" if callable(self.loader) else "None" if self.loader is None else repr(self.loader)
        return f"DistributedLoader(loader={source}, transport={short_repr(self.transport)})"

    def start(self, accesses=None, *, plan=None, slots_per_rank=2, drop_last=False, group=None) -> DistributedRun:
        """Starts the producer (on rank zero, which gives `accesses`: an array
        of cells or an `Accesses`; and `plan`: a `Plan`, a plan file path,
        `PlanOptions` or a cache budget in bytes, or None) and attaches every
        rank. `drop_last` trims an array to whole rounds of every rank's
        slot."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DistributedLoader requires an initialized torch.distributed process group")
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        hosts = [None] * world
        dist.all_gather_object(hosts, socket.gethostname(), group=group)
        if len(set(hosts)) != 1:
            raise ValueError("DistributedLoader requires ranks on one host (shared memory)")
        shared: SharedRing | None = None
        process = monitor = None
        payload: list[Any] = [None]
        if rank == 0:
            try:
                shared, process, monitor = self._launch(accesses, plan, world, slots_per_rank, drop_last)
                payload[0] = {"shared": shared.descriptor, "producer_pid": process.pid}
            except BaseException as exc:  # noqa: BLE001 - startup failures reach every rank
                if shared is not None:
                    _stop(shared, process, monitor)
                payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
        # The broadcast comes from group rank zero, which need not be global
        # rank zero.
        source_rank = dist.get_global_rank(group, 0) if group is not None else 0
        dist.broadcast_object_list(payload, src=source_rank, group=group)
        message = cast(dict[str, Any], payload[0])
        if "error" in message:
            raise RuntimeError(f"data loader startup failed: {message['error']}")
        failure = None
        if rank != 0:
            try:
                shared = SharedRing.attach(message["shared"])
            except BaseException as exc:  # noqa: BLE001 - peers leave the same collective
                failure = f"rank {rank}: {type(exc).__name__}: {exc}"
        failures = [None] * world
        dist.all_gather_object(failures, failure, group=group)
        if rank == 0 and shared is not None:
            shared.stop_sharing()  # every peer has attached or failed
        if any(failures) or shared is None:
            if shared is not None:
                _Status(shared).publish(_ERROR, {"message": "; ".join(f for f in failures if f)})
                shared.ring.abort()
                _stop(shared, process, monitor)
            raise RuntimeError(f"shared-memory attachment failed: {failures}")
        return DistributedRun(shared, rank, message["producer_pid"], process, monitor)

    def _launch(self, accesses, plan, world, slots_per_rank, drop_last) -> tuple[SharedRing, subprocess.Popen, threading.Thread]:
        """Rank zero's half of `start`: sizes the ring and its extra regions
        (status, and the accesses when they are a stored array), copies the
        accesses there, pickles the run request and starts the producer with
        the descriptors it must inherit."""
        from ._accesses import sequence
        from ._loader import Loader

        if self.loader is None:
            raise ValueError("group rank zero must give the loader")
        if slots_per_rank <= 0:
            raise ValueError("slots_per_rank must be positive")
        loader = self.loader if isinstance(self.loader, Loader) else self.loader()
        accesses = sequence(accesses)
        stored = isinstance(accesses, np.ndarray)
        if drop_last:
            if not stored:
                raise ValueError("drop_last trims an array of accesses")
            unit = loader.config.slot_rows * world * len(loader.names)
            accesses = accesses[: len(accesses) // unit * unit]
        # Batches alternate between ranks (`b % world`), so equal slots per rank
        # give every rank the same depth.
        layout = loader.ring_layout(slots=slots_per_rank * world, world=world)
        # Output names and per-schema profile details grow with the loader.
        # Reserve their JSON representation as well as the fixed counters.
        from .._cellpack import PROCESS_BOUNDS

        status_bytes = _STATUS_BYTES + len(json.dumps(loader.names, ensure_ascii=True).encode())
        if loader.config.profile is not None:
            status_bytes += len(loader.names) * (2 * len(PROCESS_BOUNDS) + 1) * 128
        extra = {"status": status_bytes, **({"accesses": max(4, accesses.nbytes)} if isinstance(accesses, np.ndarray) else {})}
        shared = SharedRing.create(layout, transport=self.transport, extra=extra)
        try:
            if isinstance(accesses, np.ndarray):
                np.frombuffer(shared.region("accesses"), np.uint32, len(accesses))[:] = accesses
            plan_path = os.fspath(Path(plan).resolve()) if isinstance(plan, (str, os.PathLike)) else None
            # Accesses are position-major, one per schema, and a batch holds
            # whole positions.
            positions = len(accesses) // len(loader.names)
            descriptor, inherited = shared.inheritable()
            request = pickle.dumps({
                "loader": loader,
                "shared": descriptor,
                "accesses": None if stored else accesses,
                "access_count": len(accesses) if stored else None,
                "plan": None if plan_path is not None else plan,
                "plan_path": plan_path,
            })
            process = subprocess.Popen(
                [sys.executable, "-m", "pycellpack.loader._producer"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                env={**os.environ, "PYCELLPACK_PARENT_PID": str(os.getpid())}, pass_fds=inherited,
            )
        except BaseException:
            shared.close()
            raise
        # Whole slots of the run: a last, partly filled slot is still a batch.
        shared.descriptor = {**shared.descriptor, "batches": -(-positions // layout.slot_rows)}
        monitor = threading.Thread(target=_supervise, args=(process, request, shared), name="pycellpack-producer-monitor", daemon=True)
        monitor.start()
        return shared, process, monitor


def _supervise(process, request, shared):
    """Rank zero's watcher thread: feeds the request to the producer, then
    records an error status (with the tail of its stderr) if it exited without
    publishing an outcome, and aborts the ring so blocked ranks stop."""
    status = _Status(shared)
    try:
        _, stderr = process.communicate(request)
        state, _ = status.read()
        if state == _RUNNING:
            message = f"data loader process {process.pid} exited with code {process.returncode}"
            if stderr:
                message += ": " + stderr.decode(errors="replace")[-6000:]
            status.publish(_ERROR, {"message": message})
            shared.ring.abort()
    except (OSError, ValueError, RuntimeError) as exc:
        status.publish(_ERROR, {"message": f"producer supervision failed: {exc}"})
        shared.ring.abort()


def _stop(shared, process, monitor):
    """Waits for (or stops) the producer and releases the shared memory."""
    if monitor is not None:
        monitor.join(timeout=5)
        if monitor.is_alive():
            process.terminate()
            monitor.join(timeout=2)
        if monitor.is_alive():
            process.kill()
            monitor.join()
    elif process is not None:
        process.kill()
        process.communicate()
    shared.close()


class DistributedRun:
    """This rank's share of a producer's run: iterate its batches (advancing
    releases the previous one; exported arrays keep their slot just as in
    the local loader), then `join` for the producer's `RunStats`.
    Successful context exit waits for the producer; early exit or an
    exception cancels it for every rank. `rank` and `world` are this rank
    and the ring's rank count; `producer_pid` is the producer's process id."""

    def __init__(self, shared: SharedRing, rank, producer_pid, process, monitor):
        """Takes this rank's consumer of `shared` and remembers the producer
        handle (None outside rank zero, which owns the process)."""
        self._shared, self._process, self._monitor = shared, process, monitor
        self._status = _Status(shared)
        self.rank, self.world = rank, shared.layout.world
        self.producer_pid = producer_pid
        self._batches = shared.descriptor["batches"]
        self._closed = False
        self._taken = 0
        # Registered: the producer fails rather than waiting on this rank if
        # this process dies or leaves early; its failures arrive through the
        # status region with their cause.
        self._consumer = shared.consumer(rank, idle=self._check, watch_producer=False)

    def __len__(self):
        """Batches of this rank: those of the producer's run with
        `b % world == rank`."""
        return max(0, (self._batches - self.rank + self.world - 1) // self.world)

    def __repr__(self) -> str:
        return (f"DistributedRun(rank={self.rank}, world={self.world}, batches={len(self)}, "
                f"taken={self._taken}, closed={self._closed}, producer_pid={self.producer_pid})")

    def __iter__(self):
        """The run is its own iterator."""
        return self

    def __next__(self):
        """This rank's next batch. Advancing releases the previous one; a
        producer failure raises with the reason the status region holds."""
        if self._closed:
            raise StopIteration
        try:
            batch = next(self._consumer)
        except RuntimeError:
            self._check()  # the producer's published reason is the better error
            raise
        self._taken += 1
        return batch

    def _check(self):
        """The producer's state and payload; raises when it failed or the run
        was cancelled, and when this rank closed its run."""
        if self._closed:
            raise RuntimeError("distributed run is closed")
        state, payload = self._status.read()
        if state in (_ERROR, _CANCELLED):
            raise RuntimeError(payload.get("message", "data loader cancelled"))
        return state, payload

    def wait(self, batch):
        """Batch `batch` of this rank."""
        self._check()
        result = self._consumer.wait(batch)
        if result is None:
            raise RuntimeError("producer ended before the expected batch")
        self._taken += 1
        return result

    @property
    def finished(self) -> bool:
        """Whether the producer succeeded (a failure raises)."""
        return self._check()[0] == _SUCCESS

    def join(self):
        """Waits for the producer's outcome and returns its `RunStats`; polls
        the status region every ten milliseconds and raises its failure."""
        from ._stats import RunStats

        while True:
            state, payload = self._check()
            if state == _SUCCESS:
                return RunStats.from_dict(payload["stats"])
            time.sleep(0.01)

    def cancel(self):
        """Cancels the run for every rank: the status names this rank and the
        ring is aborted, so blocked consumers and the producer stop."""
        if not self._closed:
            self._status.publish(_CANCELLED, {"message": f"data loading cancelled by rank {self.rank}"})
            self._shared.ring.abort()

    def close(self, *, cancel=True):
        """Releases this rank's consumer and stops the producer: cancelled
        when `cancel` is true, otherwise waited for (a failed producer raises
        through `join`). Idempotent."""
        if self._closed:
            return
        if cancel:
            self.cancel()
        try:
            # Records whether this rank took all of its batches.
            self._consumer.close()
            if not cancel:
                self.join()
        finally:
            self._closed = True
            if self._process is not None:
                _stop(self._shared, self._process, self._monitor)
            else:
                self._shared.close()

    def __enter__(self) -> Self:
        """The run itself; leaving the block closes it."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Closes the run, cancelling the producer when the block raised or
        this rank did not take all of its batches."""
        self.close(cancel=exc_type is not None or self._taken < len(self))
        return False

    def __del__(self):
        """Best-effort close of an abandoned run; finalization must not raise."""
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except (AttributeError, OSError, RuntimeError, TypeError):
                pass
