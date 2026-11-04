"""Measured cost records under `$CELLPACK_HOME` (default `~/.cellpack`).

Records live in `costs/v1/<kind>/<key>.json`, one kind per lifetime:
`host` (worker constants of a CPU and build), `storage` (device rates of a
filesystem read one way) and `output` (one process function's code). Each
holds `format`, `kind`, `key`, the `identity` its key digests, `values`,
`measured` (how they were obtained) and timestamps. Writes are atomic and
serialized by a lock file; unreadable records are skipped with a warning.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import warnings
from pathlib import Path

from ..opt._config import _atomic_json

FORMAT = 1
KINDS = ("host", "storage", "output")


def home() -> Path:
    """`$CELLPACK_HOME`, else `~/.cellpack`."""
    return Path(os.environ.get("CELLPACK_HOME") or Path.home() / ".cellpack")


class CostStore:
    """The cost records of one home directory."""

    def __init__(self, root=None):
        """The records under `root` (a home directory, `home()` when None), in
        the `costs/v1` layout."""
        self.root = (Path(root) if root is not None else home()) / "costs" / f"v{FORMAT}"

    def path(self, kind: str, key: str) -> Path:
        """The file holding one record; raises ValueError for an unknown
        kind."""
        if kind not in KINDS:
            raise ValueError(f"record kind must be one of {KINDS}")
        return self.root / kind / f"{key}.json"

    def get(self, kind: str, key: str) -> dict | None:
        """A record by kind and key, or None when there is none. A record that
        cannot be read, or whose format, kind or key disagree with its position,
        warns and counts as absent."""
        path = self.path(kind, key)
        try:
            with open(path) as f:
                record = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            warnings.warn(f"ignoring unreadable cost record {path}: {exc}", RuntimeWarning, stacklevel=2)
            return None
        if record.get("format") != FORMAT or record.get("kind") != kind or record.get("key") != key:
            warnings.warn(f"ignoring cost record {path} of another format", RuntimeWarning, stacklevel=2)
            return None
        return record

    @contextlib.contextmanager
    def _locked(self):
        """Serializes writers of this store through a lock file, so concurrent
        processes cannot interleave a read-modify-write."""
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def put(self, kind: str, key: str, identity: dict, values: dict, measured: dict | None = None) -> dict:
        """Writes a record, replacing any with this key."""
        with self._locked():
            return self._write(kind, key, identity, values, measured, self.get(kind, key))

    def update(self, kind: str, key: str, identity: dict, merge) -> dict:
        """Read-modify-write under the lock: `merge(old record or None)`
        returns `(values, measured)`."""
        with self._locked():
            old = self.get(kind, key)
            values, measured = merge(old)
            return self._write(kind, key, identity, values, measured, old)

    def _write(self, kind, key, identity, values, measured, old):
        """Writes one record atomically, keeping the creation time of the record
        it replaces."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        record = {
            "format": FORMAT,
            "kind": kind,
            "key": key,
            "identity": identity,
            "created": old["created"] if old else now,
            "updated": now,
            "values": values,
            "measured": measured or {},
        }
        _atomic_json(self.path(kind, key), record)
        return record

    def records(self):
        """Every readable record, by kind then key."""
        for kind in KINDS:
            folder = self.root / kind
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.json")):
                record = self.get(kind, path.stem)
                if record is not None:
                    yield record

    def remove(self, kind: str, key: str) -> bool:
        """Deletes one record; False when it was not there."""
        with self._locked():
            try:
                self.path(kind, key).unlink()
                return True
            except FileNotFoundError:
                return False
