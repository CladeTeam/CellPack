"""Named views of a process function's fixed-size output records."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


class OutputLayout:
    """A NumPy structured dtype describing one native output record.

    ``fields`` takes NumPy's structured dtype specifications, for example
    ``[("ids", "i8", (1024,)), ("length", "u4")]``. C struct alignment
    (including tail padding) is used by default. Explicit offsets and an
    itemsize can instead be supplied through a NumPy dtype. The declaration
    must agree with the native function; it does not change the function.
    """

    __slots__ = ("_dtype",)

    def __init__(self, fields, *, align=True):
        dtype = np.dtype(fields, align=align)
        if not dtype.names or dtype.itemsize <= 0:
            raise ValueError("output layout needs named fields and a positive record size")
        fields_by_name = cast(dict[str, tuple[Any, ...]], dtype.fields)
        for name in dtype.names:
            field, offset = fields_by_name[name][:2]
            base = field.base
            if base.kind not in "biufc" or not base.isnative or field.itemsize <= 0:
                raise ValueError(f"output field {name!r} needs native numeric values and a positive size")
            if offset < 0 or offset + field.itemsize > dtype.itemsize:
                raise ValueError(f"output field {name!r} extends beyond its record")
        self._dtype = dtype

    @property
    def dtype(self) -> np.dtype:
        """The structured record dtype, including offsets and padding."""
        return self._dtype

    @property
    def names(self) -> tuple[str, ...]:
        """Field names in declaration order."""
        return cast(tuple[str, ...], self._dtype.names)

    @property
    def offsets(self) -> dict[str, int]:
        """Byte offsets of the named fields within one record."""
        fields = cast(dict[str, tuple[Any, ...]], self.dtype.fields)
        return {name: fields[name][1] for name in self.names}

    @property
    def row_bytes(self) -> int:
        """Bytes of one complete record, including tail padding."""
        return self._dtype.itemsize

    def __repr__(self) -> str:
        from reprlib import repr as short_repr

        fields = {}
        for name in self.names:
            field = self.dtype[name]
            shape = "" if not field.shape else f"[{', '.join(map(str, field.shape))}]"
            fields[name] = f"{field.base.name}{shape}"
        return f"OutputLayout(fields={short_repr(fields)}, row_bytes={self.row_bytes})"

    def unpack(self, raw: np.ndarray, *, copy=False) -> dict[str, np.ndarray]:
        """View ``uint8[..., row_bytes]`` as named arrays without copying.

        The last axis must be contiguous; preceding axes may be strided.
        Views retain their source, including a batch's slot lease. Fields
        may be strided or unaligned. ``copy=True`` returns independent,
        aligned C-contiguous arrays, releasing that dependency.
        GPU training can instead transfer the complete packed buffer and
        construct device field views after the H2D copy has completed.
        """
        if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8:
            raise TypeError("process output must be a NumPy uint8 array; use batch.region(name)")
        if raw.ndim < 1 or raw.shape[-1] != self.row_bytes:
            raise ValueError(f"process output needs a last axis of {self.row_bytes} bytes, got {raw.shape}")
        if raw.size and raw.strides[-1] != 1:
            raise ValueError("process output's byte axis must be contiguous")
        records = np.squeeze(raw.view(self.dtype), axis=-1)
        return {name: records[name].copy(order="C") if copy else records[name] for name in self.names}
