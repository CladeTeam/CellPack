"""Worker counts for encoding and decoding outside the loader.

Native encode and decode calls release the GIL and split their work over
scoped threads, so counts scale with cores. ``workers=None`` resolves to
:func:`default_workers`: one worker per available CPU (affinity and cgroup
quota), at most :data:`MAX_WORKERS`, unless ``PYCELLPACK_WORKERS`` sets it.
Explicit counts are capped at the available CPUs.
"""

from __future__ import annotations

import math
import os

__all__ = ["MAX_WORKERS", "available_cpus", "default_workers", "resolve_workers"]

MAX_WORKERS = 32


def available_cpus() -> int:
    """CPUs this process may use: affinity, reduced by any cgroup CPU quota."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    quota = _cgroup_quota()
    return max(1, min(cpus, quota) if quota else cpus)


def _cgroup_quota() -> int | None:
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            limit, period = f.read().split()[:2]
        if limit != "max":
            return math.ceil(int(limit) / int(period))
        return None
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            limit = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return math.ceil(limit / period) if limit > 0 else None
    except (OSError, ValueError):
        return None


def default_workers() -> int:
    """``PYCELLPACK_WORKERS`` if set, else min(available CPUs, MAX_WORKERS)."""
    override = os.environ.get("PYCELLPACK_WORKERS")
    if override:
        return resolve_workers(int(override))
    return min(available_cpus(), MAX_WORKERS)


def resolve_workers(workers: int | None) -> int:
    """Resolve a ``workers`` argument: None is the default; counts are capped."""
    if workers is None:
        return default_workers()
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    return min(workers, available_cpus())
