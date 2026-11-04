"""Output rings (native): a picklable `RingLayout` naming each output's view,
`Ring`s over owned, caller or shared memory, typed `Batch` slot leases and
per-rank `Consumer`s.

A ring holds `slots` slots of `slot_rows` positions; batch `b` fills slot
`b % slots` and is released once the run has advanced past it and every array
taken from it is dropped. `LeaseObserver` exposes that release to code holding
derived arrays. The classes are the extension's; this module only re-exports
them so `pycellpack.loader` has one place for ring objects.
"""

from .._cellpack import Batch, Consumer, LeaseObserver, Ring, RingLayout

__all__ = ["Batch", "Consumer", "LeaseObserver", "Ring", "RingLayout"]
