"""Output schemas: output dims as standard ids, a dtype, and the built-in dense
row or a process function.

A schema describes one output of every position: how many dims it has
(`out_dim`), which standard id each dim takes values from, the dtype built-in
rows are converted to, and either nothing (a dense row) or the `ProcessFn`
that writes a fixed-size record per cell. The schema owns no memory; the
loader binds it to an id and a name, and a run's batches view its ring region.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from .. import _cellpack


def allowed(source, target) -> bool:
    """Whether the loader permits a stored source dtype to an output dtype."""
    return _cellpack.allowed(np.dtype(source).name, np.dtype(target).name)


class OutputSchema:
    """One output of every position: `out_ids[j]` is the standard id of output
    dim `j`; rows are mapped to these dims and converted to `dtype` (values
    stored as other types follow the conversion contract: the same type,
    lossless widening, signed to unsigned of the same width with negatives
    unchecked, signed to a wider unsigned, and any integer to a float).

    Without `process`, the output is the cell's dense row. An output dim whose
    standard id the cell's matrix has is present: the stored value, or 0 where
    the record stores none (CSR drops zeros, so they are expression values).
    An output dim the matrix has no column for is missing and takes
    `default`; a column no output dim takes is dropped. With a `ProcessFn`,
    each cell's output is `row_bytes` written by the function (its
    `row_format` picks CSR or dense rows, with the same semantics), viewed in
    batches as `out_dtype` elements (default: raw bytes); `scratch_bytes`
    per worker.

    `name` names the output in batches and statistics (a loader given a
    mapping of names uses those); `id` identifies the schema for plans
    (unset ids get the smallest free ones). Schemas pickle when their
    process function does.
    """

    def __init__(self, out_ids, dtype: DTypeLike = np.float32, *, id=None, name=None, default=0, process=None,
                 row_bytes=None, out_dtype: DTypeLike | None = None, scratch_bytes=0):
        """Validates and copies the schema: `out_ids` become a read-only uint32
        vector, `dtype` must have native byte order, `out_dtype` is only legal
        for a process output and must leave `row_bytes` a whole number of its
        elements. `default` is converted to `dtype` here, so its bits are what
        dense rows carry. The native handle built here only validates; the
        loader binds the final id and name."""
        values = np.asarray(out_ids)
        if values.ndim != 1 or (values.size and (
            values.dtype.kind not in "iu" or values.min() < 0 or values.max() > 0xFFFFFFFF
        )):
            raise ValueError("out_ids must be uint32 standard ids")
        self.out_ids = np.array(values, dtype=np.uint32, copy=True)
        self.out_ids.setflags(write=False)
        self.dtype = np.dtype(dtype)
        if not self.dtype.isnative:
            raise ValueError("output dtype must have native byte order")
        if process is None and out_dtype is not None:
            raise ValueError("out_dtype views a process function's output; the built-in row has `dtype`")
        self.out_dtype = None if out_dtype is None else np.dtype(out_dtype)
        self.id, self.name, self.default, self.process, self.scratch_bytes = id, name, default, process, scratch_bytes
        if process is not None:
            if process.row_bytes is not None:
                if row_bytes is not None and row_bytes != process.row_bytes:
                    raise ValueError(f"row_bytes must match the ProcessFn output layout ({process.row_bytes})")
                row_bytes = process.row_bytes
            if process.input_dtype is not None and self.dtype != process.input_dtype:
                raise ValueError(f"the ProcessFn template requires input dtype {process.input_dtype}, got {self.dtype}")
            if process.out_dim is not None and len(self.out_ids) != process.out_dim:
                raise ValueError(f"the ProcessFn template requires {process.out_dim} out_ids, got {len(self.out_ids)}")
        # The loader binds the final id and name; these only validate.
        self._handle = _cellpack.OutputSchema(
            0 if id is None else id, self.out_ids, self.dtype.name, np.asarray(default, dtype=self.dtype).tobytes(),
            row_bytes=row_bytes, process=None if process is None else process._handle,
            scratch_bytes=scratch_bytes, name=name,
        )
        self.row_bytes = self._handle.row_bytes
        if self.out_dtype is not None and self.row_bytes % self.out_dtype.itemsize:
            raise ValueError("row_bytes must be a whole number of out_dtype elements")

    def __repr__(self) -> str:
        from reprlib import repr as short_repr

        process = "builtin" if self.process is None else f"0x{self.process.identity:016x}"
        return (f"OutputSchema(name={short_repr(self.name)}, id={self.id}, out_dim={self.out_dim}, "
                f"dtype={self.dtype}, row_bytes={self.row_bytes}, process={process})")

    def _bind(self, id: int, name: str) -> OutputSchema:
        """This schema with the id and output name a loader gives it."""
        return OutputSchema(self.out_ids, self.dtype, id=id, name=name, default=self.default, process=self.process,
                            row_bytes=self.row_bytes if self.process is not None else None, out_dtype=self.out_dtype,
                            scratch_bytes=self.scratch_bytes)

    def __reduce__(self):
        """Pickles by the constructor's arguments, not the native handle, so a
        schema can be rebuilt in another process."""
        return _schema, (self.out_ids, self.dtype.str, self.id, self.name, self.default, self.process,
                         self.row_bytes if self.process is not None else None,
                         None if self.out_dtype is None else self.out_dtype.str, self.scratch_bytes)

    @classmethod
    def dense(cls, out_ids, dtype: DTypeLike = np.float32, *, id=None, name=None, default=0) -> OutputSchema:
        """The built-in output: each cell as a dense row of `dtype`."""
        return cls(out_ids, dtype, id=id, name=name, default=default)

    @property
    def out_dim(self) -> int:
        """Output dims: the values in every dense row."""
        return len(self.out_ids)

    @property
    def is_builtin(self) -> bool:
        """Whether the loader writes the dense row itself (no process function)."""
        return self.process is None

    @property
    def view_dtype(self) -> np.dtype | None:
        """The dtype batches view this output's rows as (None: raw bytes)."""
        return self.dtype if self.process is None else self.out_dtype


def _schema(out_ids, dtype, id, name, default, process, row_bytes, out_dtype, scratch_bytes):
    """The unpickling half of `OutputSchema.__reduce__`."""
    return OutputSchema(out_ids, dtype, id=id, name=name, default=default, process=process, row_bytes=row_bytes,
                        out_dtype=out_dtype, scratch_bytes=scratch_bytes)
