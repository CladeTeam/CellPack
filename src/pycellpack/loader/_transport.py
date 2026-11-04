"""How processes on one host share the loader's regions.

The regions (a ring's control and data plus the extra status and accesses
regions) keep their sizes and layouts; only the way processes reach the same
memory differs:

- ``shm``: named files in ``/dev/shm``, opened by name.
- ``memfd``: anonymous memory files. They are not limited by the size of the
  ``/dev/shm`` mount (only by the memory cgroup), and the kernel frees them
  when the last process closes them, so nothing can leak. The producer
  inherits them; peer ranks receive them over a Unix socket (``SCM_RIGHTS``)
  from rank zero, which accepts only processes of the same user.
- ``file``: files in a directory (``PYCELLPACK_SHM_DIR`` or the temporary
  directory) mapped shared. Use it where neither of the above is possible;
  dirty pages may be written back to that file system.

``auto`` takes ``shm`` when ``/dev/shm`` has room, then ``memfd``, then
``file``. ``PYCELLPACK_SHM`` or the ``transport`` argument picks one.
"""

from __future__ import annotations

import json
import mmap
import os
import socket
import struct
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

__all__ = ["TRANSPORTS", "FdServer", "Region", "choose", "receive_fds"]

TRANSPORTS = ("shm", "memfd", "file")
_SHM = Path("/dev/shm")
# Room kept free on a file system that is not ours alone.
_HEADROOM = 1.05


def _check_name(name):
    """Rejects names that could escape the region directory or collide with
    another program's shared memory."""
    if (
        not isinstance(name, str)
        or not name.startswith("pycellpack-")
        or "/" in name
        or "\\" in name
        or "\0" in name
    ):
        raise ValueError("invalid shared-memory region name")


def _memfd(name: str) -> int:
    """An anonymous memory file, closed on exec. libc is the fallback where
    `os.memfd_create` is not exposed."""
    create = getattr(os, "memfd_create", None)
    if create is not None:
        return create(name, os.MFD_CLOEXEC)
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "memfd_create"):
        raise OSError("memfd_create is unavailable")
    fd = libc.memfd_create(name.encode(), 1)  # MFD_CLOEXEC
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return fd


def _directory() -> Path:
    """The directory `file` regions live in: `PYCELLPACK_SHM_DIR`, else the
    temporary directory."""
    return Path(os.environ.get("PYCELLPACK_SHM_DIR") or tempfile.gettempdir())


def _free(path: Path) -> int:
    """Bytes available to this user on the file system of `path`."""
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


class Region:
    """One shared mapping, created here or attached to from its spec."""

    def __init__(self, spec, *, create=False, fd=None):
        """Creates the region `spec` names, or attaches to its existing
        memory when `create` is false. Created regions reserve their size with
        `posix_fallocate`, so an over-full file system or cgroup fails here
        rather than as SIGBUS on the first touched page; attached regions must
        match the descriptor's size. memfd regions need `fd` unless created
        here. The mapping is shared, which is what makes the memory common."""
        name, size = spec["name"], spec["size"]
        _check_name(name)
        if not isinstance(size, int) or size <= 0:
            raise ValueError("shared-memory size must be positive")
        self.spec = dict(spec)
        self.name, self.size = name, size
        self.transport = spec.get("transport", "shm")
        self.owner = create
        self.fd = -1
        self.buffer = None
        path = self.path
        if self.transport == "memfd":
            if create:
                self.fd = _memfd(name)
            elif fd is None:
                raise ValueError("attaching a memfd region needs its descriptor")
            else:
                self.fd = os.dup(fd)
        elif path is not None:
            flags = os.O_RDWR | os.O_CLOEXEC | (os.O_CREAT | os.O_EXCL if create else 0)
            self.fd = os.open(path, flags, 0o600)
        else:
            raise ValueError(f"unknown shared-memory transport {self.transport!r}")
        try:
            if create:
                # Reservation reports ENOSPC/ENOMEM here, never as SIGBUS later.
                os.posix_fallocate(self.fd, 0, size)
            elif os.fstat(self.fd).st_size != size:
                raise ValueError("shared-memory region size differs from descriptor")
            self.buffer = mmap.mmap(self.fd, size)
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            if create:
                self.unlink()
            raise

    @property
    def path(self) -> Path | None:
        """The region's file, or None for memfd regions."""
        if self.spec.get("transport", "shm") == "shm":
            return _SHM / self.spec["name"]
        if self.spec["transport"] == "file":
            return Path(self.spec["directory"]) / self.spec["name"]
        return None

    def descriptor(self):
        """The spec another process passes to `Region` to attach to this
        region (a plain dict, so it pickles)."""
        return dict(self.spec)

    def unlink(self):
        """Removes the region's file; memfd regions have none and are freed
        when their last user closes them."""
        path = self.path
        if path is not None:
            path.unlink(missing_ok=True)

    def close(self):
        """Closes the descriptor and drops this reference to the mapping.
        Arrays exported from the region hold the mapping itself, so they stay
        valid after close."""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        # Exported arrays hold the mmap itself, so dropping this reference never
        # unmaps storage still used by a Batch, NumPy slice or Tensor.
        self.buffer = None


def _close_regions(regions, *, unlink):
    """Attempt every release even if an unlink or close fails."""
    with ExitStack() as cleanup:
        for region in regions:
            cleanup.callback(region.close)
            if unlink:
                cleanup.callback(region.unlink)


def choose(sizes: dict[str, int], identity: str, transport: str = "auto"):
    """Create every region with the first transport that can hold them.

    Returns ``(transport, regions)``; raises with each transport's reason
    when none can.
    """
    if transport == "auto":
        transport = os.environ.get("PYCELLPACK_SHM") or "auto"
    order = TRANSPORTS if transport == "auto" else (transport,)
    if any(t not in TRANSPORTS for t in order):
        raise ValueError(f"transport must be one of {('auto', *TRANSPORTS)}")
    total = sum(sizes.values())
    reasons = []
    for candidate in order:
        extra = {}
        if candidate == "shm":
            if not _SHM.is_dir():
                reasons.append("shm: /dev/shm is missing")
                continue
            if transport == "auto" and _free(_SHM) < total * _HEADROOM:
                reasons.append(f"shm: /dev/shm has {_free(_SHM)} bytes free, {total} needed")
                continue
        if candidate == "file":
            directory = _directory()
            if transport == "auto" and _free(directory) < total * _HEADROOM:
                reasons.append(f"file: {directory} has {_free(directory)} bytes free, {total} needed")
                continue
            extra["directory"] = str(directory)
        regions = {}
        try:
            for key, size in sizes.items():
                spec = {"name": f"pycellpack-{identity}-{key}", "size": size,
                        "transport": candidate, **extra}
                regions[key] = Region(spec, create=True)
            return candidate, regions
        except BaseException as exc:
            _close_regions(regions.values(), unlink=True)
            if not isinstance(exc, OSError):
                raise
            reasons.append(f"{candidate}: {exc}")
    raise OSError("no shared-memory transport can hold the regions: " + "; ".join(reasons))


class FdServer:
    """Hands rank zero's memfd regions to peer ranks of the same user."""

    def __init__(self, fds: dict[str, int]):
        """Listens on an abstract Unix socket (a name starting with NUL, which
        leaves no file behind) and serves `fds`, keyed by region name, from a
        daemon thread."""
        self._fds = dict(fds)
        self.address = f"\0pycellpack-{uuid.uuid4().hex}"
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        try:
            self._socket.bind(self.address)
            self._socket.listen(64)
            self._socket.settimeout(0.2)
            self._stopped = threading.Event()
            self._thread = threading.Thread(target=self._serve, name="pycellpack-fd-server", daemon=True)
            self._thread.start()
        except BaseException:
            self._socket.close()
            raise

    def _serve(self):
        """Sends every region's descriptor to each accepted peer of the same
        user, one message per connection, until stopped."""
        keys = list(self._fds)
        message = json.dumps(keys).encode()
        while not self._stopped.is_set():
            try:
                conn, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                try:
                    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
                    _, uid, _ = struct.unpack("3i", creds)
                    if uid != os.getuid():
                        continue
                    socket.send_fds(conn, [message], [self._fds[k] for k in keys])
                except OSError:
                    # A disconnected peer must not prevent other ranks attaching.
                    continue

    def stop(self):
        """Stops accepting connections and joins the serving thread (at most
        five seconds)."""
        self._stopped.set()
        self._socket.close()
        self._thread.join(timeout=5)


def receive_fds(address: str, *, timeout: float = 30.0) -> dict[str, int]:
    """Receive every region's descriptor from rank zero's server."""
    fds = []
    try:
        deadline = time.monotonic() + timeout
        message = bytearray()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC) as conn:
            conn.settimeout(timeout)
            conn.connect(address)
            # The server closes each connection after sending; stream reads may split JSON.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out receiving shared-memory descriptors")
                conn.settimeout(remaining)
                part, received, flags, _ = socket.recv_fds(conn, 64 * 1024, 64)
                fds.extend(received)
                for fd in received:
                    os.set_inheritable(fd, False)
                if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
                    raise OSError("truncated shared-memory descriptors")
                message.extend(part)
                if len(message) > 64 * 1024 or len(fds) > 64:
                    raise OSError("shared-memory descriptor message is too large")
                if not part:
                    break
        keys = json.loads(message)
        if (not isinstance(keys, list) or not all(isinstance(key, str) for key in keys)
                or len(keys) != len(fds) or len(set(keys)) != len(keys)):
            raise OSError("invalid shared-memory descriptor names or count")
        return dict(zip(keys, fds))
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
