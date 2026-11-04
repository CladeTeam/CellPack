"""Signatures of native function code, from a bare function pointer.

A function's extent is unknown from its address, so its code is read from
the entry up to its first return instruction, at least `MIN_BYTES` and at
most `MAX_BYTES` and the end of its mapping. Leading x86-64 jumps (PLT stubs
and other trampolines) are followed. Eight-byte values that point into this
process's mappings are zeroed first: JIT compilers embed absolute addresses,
which differ between processes. This is a heuristic: code reached through
calls is not covered, and the first return may be an early exit.
"""

from __future__ import annotations

import bisect
import hashlib
import platform

MIN_BYTES = 64
MAX_BYTES = 4096
#: Bytes taken where no return instruction is recognized.
FALLBACK_BYTES = 256
#: Jumps followed from an entry before its code is read.
MAX_JUMPS = 2


class Mappings:
    """This process's memory mappings, from `/proc/self/maps`."""

    def __init__(self, text=None):
        """Reads `/proc/self/maps`, or `text` as that file's content, into
        sorted address spans."""
        if text is None:
            try:
                with open("/proc/self/maps") as f:
                    text = f.read()
            except OSError:
                text = ""
        spans = []
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            spans.append((lo, hi))
        spans.sort()
        self._lo = [lo for lo, _ in spans]
        self._hi = [hi for _, hi in spans]

    def end(self, address: int) -> int | None:
        """The end of the mapping holding `address`, or None."""
        i = bisect.bisect_right(self._lo, address) - 1
        return self._hi[i] if i >= 0 and address < self._hi[i] else None

    def __contains__(self, address: int) -> bool:
        """Whether `address` lies in one of this process's mappings."""
        return self.end(address) is not None


def _read(address: int, length: int) -> bytes:
    """Memory of this process; empty where it is not readable."""
    if length <= 0:
        return b""
    try:
        with open("/proc/self/mem", "rb", buffering=0) as f:
            f.seek(address)
            return f.read(length)
    except (OSError, OverflowError, ValueError):
        return b""


def _jump_target(address: int, code: bytes) -> int | None:
    """Where an x86-64 entry that only jumps goes (after an `endbr64`)."""
    at = 4 if code[:4] == b"\xf3\x0f\x1e\xfa" else 0
    op = code[at:at + 1]
    if op == b"\xe9" and len(code) >= at + 5:
        return address + at + 5 + int.from_bytes(code[at + 1:at + 5], "little", signed=True)
    if op == b"\xeb" and len(code) >= at + 2:
        return address + at + 2 + int.from_bytes(code[at + 1:at + 2], "little", signed=True)
    if code[at:at + 2] == b"\xff\x25" and len(code) >= at + 6:
        slot = address + at + 6 + int.from_bytes(code[at + 2:at + 6], "little", signed=True)
        target = _read(slot, 8)
        return int.from_bytes(target, "little") if len(target) == 8 else None
    return None


def body_length(code: bytes, machine: str) -> int:
    """Bytes through the first return instruction, by architecture."""
    if machine in ("x86_64", "amd64", "i386", "i686"):
        for i, b in enumerate(code):
            if b == 0xC3:
                return i + 1
            if b == 0xC2 and i + 3 <= len(code):
                return i + 3
        return len(code)
    if machine in ("aarch64", "arm64"):
        for i in range(0, len(code) - 3, 4):
            if int.from_bytes(code[i:i + 4], "little") & 0xFFFFFC1F == 0xD65F0000:
                return i + 4
        return len(code)
    return min(len(code), FALLBACK_BYTES)


def mask_pointers(code: bytes, mappings: Mappings) -> bytes:
    """`code` with every eight-byte value pointing into a mapping zeroed."""
    out = bytearray(code)
    i = 0
    while i + 8 <= len(out):
        if int.from_bytes(out[i:i + 8], "little") in mappings:
            out[i:i + 8] = bytes(8)
            i += 8
        else:
            i += 1
    return bytes(out)


def function_code(address: int, mappings: Mappings | None = None, machine: str | None = None) -> bytes:
    """The masked code a function pointer's signature covers."""
    mappings = Mappings() if mappings is None else mappings
    machine = platform.machine().lower() if machine is None else machine
    code = b""
    for _ in range(MAX_JUMPS + 1):
        end = mappings.end(address)
        if end is None:
            return b""
        code = _read(address, min(MAX_BYTES, end - address))
        if machine not in ("x86_64", "amd64"):
            break
        target = _jump_target(address, code)
        if target is None or target == address:
            break
        address = target
    n = max(body_length(code, machine), min(MIN_BYTES, len(code)))
    return mask_pointers(code[:n], mappings)


def signature(address: int, mappings: Mappings | None = None) -> str:
    """Hex digest of `function_code`; stable across processes for code that
    is loaded from the same file or compiled the same way."""
    machine = platform.machine().lower()
    code = function_code(address, mappings, machine)
    return hashlib.blake2b(machine.encode() + b"\0" + code, digest_size=16).hexdigest()
