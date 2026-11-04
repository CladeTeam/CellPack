"""Immutable execution configuration for native loader runs.

`LoaderConfig` holds what a run is built from: worker threads, the I/O
backend, inflight and claim sizes, slot size, execution window, CPU pinning
and profiling sites. Unset fields resolve here, once, from the CPUs this
process may use (affinity and cgroup quota) and from what the kernel allows
(io_uring), so everything downstream reads one frozen object. `available_memory`
reads the same kind of limits for memory, which the loader compares a run's
reservations against.
"""

from __future__ import annotations

import ctypes
import functools
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .._cellpack import build_info
from .._workers import available_cpus

# Defaults scale with the CPUs this process may use, up to the reported
# 16-worker configuration (the native crate's fixed default).
MAX_WORKERS = 16
MAX_COMPILE_THREADS = 16
# Online compilation outpaces loading by far: one compiler thread per this
# many loader workers keeps ahead.
ONLINE_WORKERS_PER_THREAD = 4


def _cpus(pinned) -> int:
    """Usable CPUs: the pinned ones, else the CPUs this process may run on."""
    return len(pinned) if pinned else available_cpus()


def default_workers(cpus=()) -> int:
    """Loader workers when unset: one per usable CPU (`cpus` if pinned), at
    most `MAX_WORKERS`."""
    return min(MAX_WORKERS, _cpus(cpus))


def compile_threads(workers: int, *, online: bool, cpus=()) -> int:
    """Compiler threads when unset. Offline: one per usable CPU (`cpus` if
    pinned). Online: a quarter of the loader's workers. Both at most
    `MAX_COMPILE_THREADS` and the usable CPUs."""
    usable = _cpus(cpus)
    threads = workers // ONLINE_WORKERS_PER_THREAD if online else usable
    return max(1, min(MAX_COMPILE_THREADS, usable, threads))


def _read_int(path):
    """The first number of a cgroup file; None where it is unreadable or
    reads `max` (cgroup v2's "no limit")."""
    try:
        with open(path) as f:
            text = f.read().split()[0]
        return None if text == "max" else int(text)
    except (OSError, ValueError, IndexError):
        return None


def _stat(path, key) -> int:
    """One field of a cgroup `memory.stat` in bytes; 0 where it is missing."""
    try:
        with open(path) as f:
            for line in f:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return 0


def available_memory() -> int | None:
    """Bytes this process may still allocate: the smaller of the system's
    MemAvailable and the cgroup's headroom (v2 or v1); None if neither is
    known."""
    known = []
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    known.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    for root, limit, usage, inactive in (
        ("/sys/fs/cgroup", "memory.max", "memory.current", "inactive_file"),
        ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file"),
    ):
        cap, used = _read_int(f"{root}/{limit}"), _read_int(f"{root}/{usage}")
        # cgroup v1 reports no limit as a huge page-rounded value.
        if cap is not None and used is not None and cap < 1 << 60:
            # Usage counts page cache; its inactive part is reclaimed first
            # (the working-set convention).
            known.append(max(0, cap - used + _stat(f"{root}/memory.stat", inactive)))
            break
    return min(known) if known else None


class IoBackend(str, Enum):
    """How the loader issues reads. `PREAD` completes each read inside the
    call (`preadv`); `URING` queues a worker's reads on its own io_uring and
    harvests completions. The value is the name the native side takes, so a
    member compares equal to its string."""

    PREAD = "pread"
    URING = "uring"


# io_uring_setup has one number on every Linux architecture (the unified
# table since 5.1); its parameter block is 120 bytes.
_IO_URING_SETUP = 425
_IO_URING_PARAMS = 120


@functools.cache
def uring_supported() -> bool:
    """Whether this process can create an io_uring: Linux, and a probe ring
    that the kernel, its `io_uring_disabled` setting and any seccomp filter
    allow."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        params = ctypes.create_string_buffer(_IO_URING_PARAMS)
        fd = libc.syscall(ctypes.c_long(_IO_URING_SETUP), ctypes.c_uint(8), params)
    except (OSError, AttributeError):
        return False
    if fd < 0:
        return False
    os.close(fd)
    return True


def default_backend() -> IoBackend:
    """io_uring where supported, else pread."""
    return IoBackend.URING if uring_supported() else IoBackend.PREAD


def profile_sites() -> dict[str, str]:
    """Every profiling site this build carries: name -> what it times."""
    return dict(build_info()["profile_sites"])


def profile_groups() -> dict[str, tuple[str, ...]]:
    """Named site sets: group -> its member sites and groups (a group may
    name another); `profile` also takes `all` for every site."""
    return {name: tuple(sites) for name, sites in build_info()["profile_groups"]}


def _profile(value) -> tuple[str, ...] | None:
    """`profile` as site and group names, or None for no profiling."""
    if value is None or value is False:
        return None
    if value is True:
        return ("all",)
    names = (value,) if isinstance(value, str) else tuple(value) if isinstance(value, Iterable) else None
    if names is None or not all(isinstance(n, str) for n in names):
        raise TypeError("profile takes a bool, a site or group name, or names of them")
    known = set(profile_sites()) | set(profile_groups()) | {"all"}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown profile sites or groups {unknown}; see profile_sites() and profile_groups()")
    return names


@dataclass(frozen=True)
class LoaderConfig:
    """How a loader runs: `workers` threads (default `default_workers(cpus)`),
    each with `inflight` positions in flight; I/O through `backend`
    (default io_uring where supported, else pread); `claim_batch` accesses
    claimed per cursor update; `slot_rows` positions per output slot; the
    execution `window` in slots (default: a plan's reuse gap, or enough slots
    for the in-flight accesses); `cpus` to pin worker `i` to `cpus[i % len]`.

    `profile` selects the timed sites: False or None runs the untimed
    instance, True every site, or names of sites and groups (see
    `profile_sites()` and `profile_groups()`); it is kept as a tuple of
    names, or None. Every build carries both instances."""

    workers: int | None = None
    backend: IoBackend | str | None = None
    inflight: int = 64
    claim_batch: int = 8
    slot_rows: int = 256
    window: int | None = None
    profile: bool | str | tuple[str, ...] | None = None
    cpus: tuple[int, ...] = ()

    def __post_init__(self):
        """Resolves unset fields and validates the rest. The dataclass is
        frozen, so resolved values are set through `object.__setattr__`; a
        `profile` value is kept as a tuple of site and group names or None."""
        object.__setattr__(self, "backend", default_backend() if self.backend is None else IoBackend(self.backend))
        object.__setattr__(self, "cpus", tuple(self.cpus))
        object.__setattr__(self, "profile", _profile(self.profile))
        workers = default_workers(self.cpus) if self.workers is None else self.workers
        object.__setattr__(self, "workers", workers)
        if min(workers, self.inflight, self.claim_batch, self.slot_rows) <= 0:
            raise ValueError("workers, inflight, claim_batch and slot_rows must be positive")
        if self.window is not None and self.window <= 0:
            raise ValueError("the window (slots) must be positive")
        if any(cpu < 0 for cpu in self.cpus):
            raise ValueError("CPU ids must be nonnegative")
