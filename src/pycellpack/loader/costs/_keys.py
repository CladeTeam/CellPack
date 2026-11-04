"""What a stored cost record applies to: the host, the storage under a data
source, and each process function's code.

An identity is a JSON object and `digest` reduces it to the record's file name.
The host's covers the CPU model, architecture, kernel set, package version and
row policy; the storage's covers the mount of the source's largest file, how
the loader reads it and the page size; an output's covers both entry code
signatures, the declared identity and the output shape. `storage_tier` maps a
mount to the class of device whose default rates apply.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
from pathlib import Path

from ._code import Mappings, signature

#: Filesystems served over a network or by a parallel/user-space service.
NETWORK_FS = frozenset({
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "gpfs", "beegfs", "cephfs", "ceph",
    "glusterfs", "9p", "afs", "orangefs", "pvfs2", "juicefs", "s3fs",
})
#: Filesystems that are memory, not a device.
MEMORY_FS = frozenset({"tmpfs", "ramfs", "devtmpfs"})


def digest(identity) -> str:
    """A short stable key for a JSON-able identity."""
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode(), digest_size=10).hexdigest()


@functools.cache
def cpu_model() -> str:
    """The CPU's model name, read once from `/proc/cpuinfo` (other platforms
    take the process's processor description)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Hardware", "Processor"):
                    return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version() -> str:
    """The package's major.minor version; worker constants measured for another
    minor version are not reused."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("pycellpack")
    except PackageNotFoundError:
        v = "0.0"
    return ".".join(v.split(".")[:2])


@functools.cache
def host_identity() -> dict:
    """The CPU model, machine architecture, kernel set, package version and
    native row policy."""
    from ..._cellpack import kernel_set

    return {"cpu": cpu_model(), "machine": platform.machine(), "kernel_set": kernel_set(), "version": _version(), "row_policy": 1}


def host_key() -> str:
    """The host record key: the digest of `host_identity`."""
    return digest(host_identity())


def _mount(device: int) -> tuple[str, str]:
    """Filesystem type and mount source of the mount with this st_dev."""
    major, minor = os.major(device), os.minor(device)
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                left, _, right = line.partition(" - ")
                fields = left.split()
                if len(fields) >= 3 and fields[2] == f"{major}:{minor}":
                    rest = right.split()
                    return (rest[0], rest[1] if len(rest) > 1 else "")
    except OSError:
        pass
    return ("", "")


def _block(major: int, minor: int, depth: int = 0) -> tuple[str, bool | None]:
    """A block device's kernel name and whether it rotates (its first
    underlying device for device-mapper and md stacks)."""
    base = Path(f"/sys/dev/block/{major}:{minor}")
    try:
        real = base.resolve()
    except OSError:
        return ("", None)
    queue = real / "queue"
    if not queue.exists() and (real.parent / "queue").exists():
        queue = real.parent / "queue"  # a partition
    slaves = real / "slaves"
    if depth < 3 and slaves.is_dir():
        for slave in sorted(slaves.iterdir()):
            try:
                major_minor = (slave / "dev").read_text().strip().split(":")
                return _block(int(major_minor[0]), int(major_minor[1]), depth + 1)
            except (OSError, ValueError, IndexError):
                break
    try:
        rotational = (queue / "rotational").read_text().strip() == "1"
    except OSError:
        rotational = None
    name = real.name if (real / "queue").exists() else real.parent.name
    return (name, rotational)


def storage_tier(fs: str, device: int | None) -> str:
    """A coarse class of the storage, for default device rates: memory,
    network, nvme, ssd, hdd, or unknown."""
    if fs in MEMORY_FS:
        return "memory"
    if fs in NETWORK_FS or fs.startswith("fuse"):
        return "network"
    if device is None:
        return "unknown"
    name, rotational = _block(os.major(device), os.minor(device))
    if name.startswith("nvme"):
        return "nvme"
    if rotational is True:
        return "hdd"
    if rotational is False:
        return "ssd"
    return "unknown"


def _dominant(paths) -> str | None:
    """The data source file with the most bytes."""
    best, size = None, -1
    for p in paths:
        try:
            n = os.stat(p).st_size
        except OSError:
            continue
        if n > size:
            best, size = p, n
    return best


def storage_identity(paths, *, backend: str, direct_io: bool, page_size: int) -> tuple[dict, str]:
    """The storage behind most of a data source's bytes, how it is read, and
    its tier."""
    path = _dominant(paths)
    fs, source, device = "", "", None
    if path is not None:
        device = os.stat(path).st_dev
        fs, source = _mount(device)
    identity = {"host": host_key(), "fs": fs, "source": source, "backend": str(backend),
                "direct_io": bool(direct_io), "page_size": int(page_size)}
    return identity, storage_tier(fs, device)


def output_identity(schema, mappings: Mappings | None = None) -> dict | None:
    """A process function's code, declared identity and output shape on this
    host; None for the built-in output (priced per dtype with the host)."""
    process = schema.process
    if process is None:
        return None
    mappings = Mappings() if mappings is None else mappings
    csr, dense = process._handle.addresses
    return {
        "host": host_key(),
        "code": [signature(csr, mappings), signature(dense, mappings)],
        "identity": int(process.identity),
        "partial_rows": bool(process.partial_rows),
        "row_format": str(process.row_format),
        "dtype": schema.dtype.name,
        "cell_bytes": int(schema.row_bytes),
        "scratch_bytes": int(schema.scratch_bytes),
    }
