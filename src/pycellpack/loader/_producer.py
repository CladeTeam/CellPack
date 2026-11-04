"""The dedicated producer process of a `DistributedLoader`; launched only by
group rank zero, which sends it the pickled loader and run.

The producer attaches the shared ring, runs the loader into it, publishes the
outcome in the status region and releases the shared memory. A parent-death
signal stops it when rank zero dies, so an abandoned ring is not left waiting
and named regions are unlinked by their last user.
"""

from __future__ import annotations

import os
import pickle
import signal
import sys

import numpy as np

from ._distributed import _ERROR, _SUCCESS, _Status
from ._shared import SharedRing
from .opt import Plan


class _ParentExited(BaseException):
    """Rank zero died; the producer must stop and release the shared names."""


def _follow_parent() -> None:
    """Installs the watch that ends this process with rank zero."""
    # The kernel signals this process when its parent dies; the pid check
    # covers a parent that died before the request took effect.
    def stop(signum, frame):
        """Turns the termination signal into an exception, so the shared names
        are released by the normal cleanup path."""
        raise _ParentExited

    signal.signal(signal.SIGTERM, stop)
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        return
    parent = int(os.environ.get("PYCELLPACK_PARENT_PID", "0"))
    if parent and os.getppid() != parent:
        raise _ParentExited


def main():
    """The producer's entry point: reads the request from stdin, attaches the
    ring, runs the loader into it and publishes the outcome (statistics or
    error) for the ranks."""
    _follow_parent()
    request = pickle.load(sys.stdin.buffer)
    shared = SharedRing.attach(request["shared"])
    status = _Status(shared)
    run = None
    orphaned = False
    try:
        loader = request["loader"]
        accesses = request["accesses"]
        if accesses is None:
            accesses = np.frombuffer(shared.region("accesses"), np.uint32, request["access_count"])
        plan = Plan.open(request["plan_path"]) if request["plan_path"] is not None else request["plan"]
        run = loader.start(accesses, ring=shared.ring, plan=plan)
        stats = run.join()
        status.publish(_SUCCESS, {"stats": stats.to_dict(), "pid": os.getpid()})
        return 0
    except BaseException as exc:  # noqa: BLE001 - every failure reaches the ranks
        orphaned = isinstance(exc, _ParentExited)
        status.publish(_ERROR, {"message": f"{type(exc).__name__}: {exc}"[:6000], "pid": os.getpid()})
        shared.ring.abort()
        return 1
    finally:
        try:
            if run is not None:
                run.close()
        finally:
            # Without rank zero nobody else unlinks the shared names.
            shared.owner = shared.owner or orphaned
            shared.close()


if __name__ == "__main__":
    raise SystemExit(main())
