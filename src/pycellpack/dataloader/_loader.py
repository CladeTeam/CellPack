"""Friendly training iteration over a corpus of lazy single-cell datasets."""

from __future__ import annotations

import operator
import weakref
from collections.abc import Sequence

import numpy as np

from ..loader import AccessFn, DataSource, Loader, OutputSchema, ProcessFn, Run
from ..loader._source import NONE
from ._accesses import Accesses, bind_accesses, corpus_counts
from ._dataset import ScDataset
from ._transfer import PackedTransfer


def _positive(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = operator.index(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class ScDataLoader:
    """Iterate named Tensors while native workers supply a bounded packed ring.

    corpus is an ordered list of ScDatasets; dataset ids are its indices.
    accesses accepts Accesses or (dataset_ids, rows). Alternatively access_fn
    takes a native AccessFn. Supply exactly one; the caller owns sampling,
    ordering and epoch policy.

    out_ids defines the common mapped panel, in ProcessFn table order. It
    can be omitted for datasets with identical standard ids. process outputs
    require a declared OutputLayout; without process the output is dense X.
    Torch is imported only by iteration, never by construction or len().

    prefetch is the number of host output slots; workers are native threads.
    CUDA uses one packed H2D and returns field views; contiguous=True or a
    sequence of field names requests field copies on independent CUDA streams.
    CPU batches own their storage and may be retained without blocking a slot.
    """

    def __init__(self, corpus: Sequence[ScDataset], accesses=None, *, access_fn: AccessFn | None = None,
                 out_ids: np.ndarray | None = None, process: ProcessFn | None = None,
                 batch_size=256, device="cpu", prefetch=4, cache_bytes=0, workers=None,
                 dtype=None, default=0, scratch_bytes=0, drop_last=False, contiguous=False, backend=None,
                 direct_io=None, **settings):
        self.corpus = tuple(corpus)
        if not self.corpus or not all(isinstance(d, ScDataset) for d in self.corpus):
            raise TypeError("corpus must be a nonempty sequence of ScDatasets")
        self._counts = corpus_counts(self.corpus)
        self.batch_size = _positive(batch_size, "batch_size")
        self.prefetch = _positive(prefetch, "prefetch")
        self.cache_bytes = operator.index(cache_bytes)
        if self.cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.accesses = accesses
        self.access_fn = access_fn
        self._access_input()
        if process is not None and (not isinstance(process, ProcessFn) or process.output is None):
            raise TypeError("process must be a ProcessFn with a declared output= layout")
        self.process = process
        if out_ids is None:
            ids = self.corpus[0].ids
            if any(not np.array_equal(ids, d.ids) for d in self.corpus[1:]):
                raise ValueError("datasets have different column maps; give the common output panel as out_ids")
            out_ids = ids[ids != NONE]
        values = np.asarray(out_ids)
        if (values.ndim != 1 or not values.size or values.dtype.kind not in "iu"
                or values.min() < 0 or values.max() >= NONE or len(np.unique(values)) != len(values)):
            raise ValueError("out_ids must be a nonempty vector of distinct standard ids below uint32.max")
        self._out_ids = np.array(values, dtype=np.uint32, copy=True)
        self._out_ids.setflags(write=False)
        if dtype is None:
            dtype = np.float32 if process is None or process.input_dtype is None else process.input_dtype
        self.dtype = np.dtype(dtype)
        self.device = str(device)
        self.drop_last = bool(drop_last)
        self.stats = None
        self._active: weakref.ReferenceType | None = None
        output = None if process is None else process.output
        names = ("X",) if output is None else output.names
        if isinstance(contiguous, bool):
            contiguous = names if contiguous else ()
        elif isinstance(contiguous, str):
            contiguous = (contiguous,)
        self._contiguous = frozenset(contiguous)
        if not self._contiguous <= set(names):
            raise ValueError(f"contiguous fields must be among {names}")
        # The private standard-id space is the output panel. Its size depends
        # on the requested dims, independently of sparse external id values.
        order = np.argsort(self._out_ids)
        sorted_ids = self._out_ids[order]
        entries = []
        for dataset in self.corpus:
            store, ids = dataset._source_entry()
            positions = np.searchsorted(sorted_ids, ids)
            within = np.flatnonzero((ids != NONE) & (positions < len(sorted_ids)))
            matched = within[sorted_ids[positions[within]] == ids[within]]
            mapping = np.full(ids.shape, NONE, np.uint32)
            mapping[matched] = order[positions[matched]]
            entries.append((store, mapping))
        source = DataSource(entries, len(self._out_ids), direct_io=direct_io)
        schema = OutputSchema(np.arange(len(self._out_ids), dtype=np.uint32), self.dtype,
                              name="X", default=default, process=process, scratch_bytes=scratch_bytes)
        if "slot_rows" in settings or "config" in settings:
            raise TypeError("ScDataLoader configures slot_rows from batch_size; pass individual worker settings")
        self._loading = Loader(source, schema, workers=workers, backend=backend, slot_rows=self.batch_size, **settings)

    @property
    def out_ids(self):
        return self._out_ids.view()

    def __repr__(self):
        return (f"ScDataLoader(datasets={len(self.corpus)}, cells={sum(self._counts)}, batch_size={self.batch_size}, "
                f"device={self.device!r}, prefetch={self.prefetch}, cache_bytes={self.cache_bytes})")

    def __getstate__(self):
        if self._active is not None and self._active() is not None:
            raise TypeError("close the active ScDataLoader iterator before pickling")
        state = self.__dict__.copy()
        state["_active"] = None
        return state

    def _access_input(self):
        if (self.accesses is None) == (self.access_fn is None):
            raise TypeError("give exactly one of accesses or access_fn")
        if self.access_fn is not None:
            if not isinstance(self.access_fn, AccessFn):
                raise TypeError("access_fn must be an AccessFn; construct sampling policy externally")
            return self.access_fn
        if isinstance(self.accesses, tuple) and len(self.accesses) == 2:
            return Accesses(*self.accesses)
        if not isinstance(self.accesses, Accesses):
            raise TypeError("accesses must be Accesses or (dataset_ids, rows)")
        return self.accesses

    def _sequence(self):
        native = bind_accesses(self._access_input(), self._counts)
        if self.drop_last:
            native = native.slice(0, len(native) // self.batch_size * self.batch_size)
        return native

    def __len__(self):
        n = len(self._access_input())
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self._active is not None and self._active() is not None:
            raise RuntimeError("ScDataLoader already has an active iterator; close it before starting another")
        iterator = _Iterator(self)
        self._active = weakref.ref(iterator)
        return iterator

    def close(self):
        """Close this loader's active iterator; the loader can be reused."""
        active = None if self._active is None else self._active()
        if active is not None:
            active.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _Iterator:
    def __init__(self, loading):
        self.loading = loading
        self.closed = True
        sequence = loading._sequence()
        self.batches = (len(sequence) + loading.batch_size - 1) // loading.batch_size
        self.transfer: PackedTransfer | None = PackedTransfer(
            loading.device, loading._loading.schemas["X"],
            None if loading.process is None else loading.process.output,
            loading._contiguous,
        )
        ring = self.transfer.ring(loading._loading, loading.prefetch)
        self.run: Run | None = loading._loading.start(sequence, ring=ring, plan=loading.cache_bytes or None)
        loading.stats = None
        self.closed = False

    def __len__(self):
        return self.batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        run = self.run
        assert run is not None
        try:
            batch = next(run)
            transfer = self.transfer
            assert transfer is not None
            packed = transfer.copy_batch(batch)
            return transfer.unpack(packed)
        except StopIteration:
            try:
                self.loading.stats = run.join()
            finally:
                self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        run = self.run
        assert run is not None
        try:
            if self.loading.stats is None:
                self.loading.stats = run.snapshot()
        finally:
            try:
                run.close()
            finally:
                self.run = None
                self.transfer = None
                self.loading._active = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        if not getattr(self, "closed", True):
            try:
                self.close()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass


__all__ = ["ScDataLoader"]
