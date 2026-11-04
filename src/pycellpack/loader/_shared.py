"""Rings shared by processes of one host.

`SharedRing` owns the memory of one ring (control, data and any extra named
regions), the descriptor by which other processes attach, and the socket that
hands memfd descriptors to them. Region sizes and the layout never change, so
producer and consumers can start in any order once they have the descriptor;
creation unlinks named memory on close, attaching never does.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import TYPE_CHECKING

import numpy as np

from ._ring import Consumer, Ring, RingLayout
from ._transport import FdServer, Region, _close_regions, choose, receive_fds

if TYPE_CHECKING:
    from typing_extensions import Self


class SharedRing:
    """A ring in memory that processes of this host share: create it in one
    process, send its `descriptor` (a picklable dict) to others and
    `attach` there. A loader in any of them produces into `ring`
    (`Loader.start(..., ring=shared.ring)`); each consumer process takes its
    rank's batches through `consumer(rank)`. Consumers and the producer
    register in the ring's control, so a blocked wait fails with the reason
    when a participant died or left early.

    `transport` picks how the memory is shared: named files in /dev/shm,
    anonymous memfd files (handed to attaching processes of the same user over
    a Unix socket while `sharing`), or files in a directory; "auto" (or
    `PYCELLPACK_SHM`) takes the first that has room.
    `extra` adds named byte regions of the given sizes in the same memory
    (`region(name)`). The creating process unlinks named memory on `close`;
    memfd memory is freed with its last user. Arrays exported from batches
    keep their mappings alive after close.
    """

    def __init__(self, layout: RingLayout, regions: dict, descriptor: dict, *, owner: bool, server=None):
        """Views the regions as one ring. The creating process (`owner`)
        initializes the control block; an attaching one must find it already
        initialized. A failure here releases the regions again."""
        self.layout, self.descriptor, self.owner = layout, descriptor, owner
        self._regions, self._server = regions, server
        self._status_lock = threading.Lock()
        try:
            # The native ring takes the two regions as byte arrays over their
            # mappings; only the creating process initializes the control block.
            control, data = (np.ndarray((regions[k].size,), np.uint8, buffer=regions[k].buffer) for k in ("control", "data"))
            self.ring = Ring(layout, data=data, control=control, initialize=owner)
        except BaseException:
            self.close()
            raise

    @classmethod
    def create(cls, layout: RingLayout, *, transport="auto", extra=None) -> SharedRing:
        """A new shared ring of `layout` (and `extra` regions)."""
        if {"control", "data"}.intersection(extra or {}):
            raise ValueError("extra region names cannot replace control or data")
        # A zero-sized region cannot be mapped, so `extra` sizes are at least
        # one byte.
        sizes = {"control": layout.control_bytes, "data": layout.data_bytes, **{k: max(1, int(v)) for k, v in (extra or {}).items()}}
        chosen, regions = choose(sizes, uuid.uuid4().hex, transport)
        descriptor = {"layout": layout, "transport": chosen, "regions": {k: r.descriptor() for k, r in regions.items()}}
        server = None
        try:
            if chosen == "memfd":
                server = FdServer({k: r.fd for k, r in regions.items()})
                descriptor["fd_socket"] = server.address
        except BaseException:
            _close_regions(regions.values(), unlink=True)
            raise
        return cls(layout, regions, descriptor, owner=True, server=server)

    @classmethod
    def attach(cls, descriptor: dict, *, fds: dict | None = None) -> SharedRing:
        """The shared ring a `descriptor` names. memfd memory arrives from
        the creator's socket, or as inherited `fds` (see `inheritable`)."""
        fds = dict(fds or descriptor.get("fds") or {})
        received = {}
        if descriptor["transport"] == "memfd" and not fds:
            fds = received = receive_fds(descriptor["fd_socket"])
        regions = {}
        try:
            for key, spec in descriptor["regions"].items():
                regions[key] = Region(spec, fd=fds.get(key))
        except BaseException:
            _close_regions(regions.values(), unlink=False)
            raise
        finally:
            for fd in received.values():
                os.close(fd)  # regions hold their own duplicates
        return cls(descriptor["layout"], regions, descriptor, owner=False)

    def inheritable(self) -> tuple[dict, tuple[int, ...]]:
        """A descriptor for a child process and the file descriptors it
        must inherit (`subprocess.Popen(pass_fds=...)`), for memfd memory."""
        if self.descriptor["transport"] != "memfd":
            return self.descriptor, ()
        fds = {k: r.fd for k, r in self._regions.items()}
        return {**self.descriptor, "fds": fds}, tuple(fds.values())

    @property
    def sharing(self) -> bool:
        """Whether attaching processes can still fetch memfd descriptors."""
        return self._server is not None

    def stop_sharing(self) -> None:
        """Stops handing out memfd descriptors (every peer has attached)."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def region(self, name: str) -> memoryview:
        """An `extra` region's bytes."""
        return memoryview(self._regions[name].buffer)

    def region_fd(self, name: str) -> int:
        """An `extra` region's descriptor (the status region is locked through
        it)."""
        return self._regions[name].fd

    def consumer(self, rank: int = 0, **kwargs) -> Consumer:
        """Rank `rank`'s consumer of the ring's current (or next) run (see
        `Ring.consumer`)."""
        return self.ring.consumer(rank, **kwargs)

    def close(self) -> None:
        """Stops handing out descriptors and releases every region, unlinking
        named memory when this process created it."""
        try:
            self.stop_sharing()
        finally:
            # The ring stays usable for batches already taken: their arrays
            # keep the mappings alive.
            _close_regions(self._regions.values(), unlink=self.owner)
            self._regions = {}

    def __enter__(self) -> Self:
        """The ring itself; leaving the block closes it."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Closes the ring, whether the block succeeded or not."""
        self.close()
        return False
