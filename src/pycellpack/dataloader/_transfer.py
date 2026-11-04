"""One packed transfer, followed by typed views and optional field copies."""

from __future__ import annotations

from typing import Any

import numpy as np


def torch_module() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ModuleNotFoundError(
            "ScDataLoader iteration requires PyTorch; install a build compatible with your CPU/CUDA environment",
            name="torch",
        ) from exc
    return torch


class PackedTransfer:
    """Copies a complete batch once and retires its host lease before return.

    CPU output owns one packed clone. CUDA output uses one H2D from pinned
    storage, waits only for its copy stream, and releases all host views.
    Device field processing can continue after the producer reuses the slot.
    """

    def __init__(self, device, schema, output, contiguous):
        torch = self.torch = torch_module()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("ScDataLoader supports CPU and CUDA devices")
        self.schema = schema
        self.row_bytes = schema.row_bytes
        self.contiguous = contiguous
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.field_streams: dict[str, Any] = {}
        if output is None:
            specs = [("X", np.dtype(schema.dtype), (schema.out_dim,), 0)]
        else:
            specs = [(name, output.dtype[name].base, output.dtype[name].shape, output.offsets[name])
                     for name in output.names]
        self.fields = []
        for name, dtype, shape, offset in specs:
            tensor_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            width = dtype.itemsize * int(np.prod(shape)) if shape else dtype.itemsize
            self.fields.append((name, tensor_dtype, shape, offset, width, dtype.itemsize))

    def ring(self, loading, prefetch):
        from ..loader import Ring

        layout = loading.ring_layout(slots=prefetch)
        if self.copy_stream is None:
            return Ring(layout)
        storage = self.torch.empty(layout.data_bytes, dtype=self.torch.uint8, pin_memory=True)
        return Ring(layout, data=storage.numpy())

    def copy_batch(self, batch):
        torch = self.torch
        raw = batch.region(self.schema.name)
        host = torch.from_numpy(raw)
        try:
            if self.copy_stream is None:
                return host.clone()
            with torch.cuda.stream(self.copy_stream):
                return host.to(self.device, non_blocking=True)
        finally:
            try:
                if self.copy_stream is not None:
                    # Host views remain live through this wait, including error paths.
                    self.copy_stream.synchronize()
            finally:
                batch.release()

    def unpack(self, packed):
        torch = self.torch
        current = torch.cuda.current_stream(self.device) if self.copy_stream is not None else None
        result = {}
        for name, dtype, shape, offset, width, itemsize in self.fields:
            aligned = offset % itemsize == 0 and self.row_bytes % itemsize == 0
            needs_copy = name in self.contiguous or not aligned
            stream = None
            if current is not None:
                packed.record_stream(current)
                if needs_copy:
                    if name not in self.field_streams:
                        self.field_streams[name] = torch.cuda.Stream(device=self.device)
                    stream = self.field_streams[name]
            if stream is None:
                result[name] = self._field(packed, dtype, shape, offset, width, aligned, needs_copy)
            else:
                assert current is not None
                with torch.cuda.stream(stream):
                    packed.record_stream(stream)
                    result[name] = self._field(packed, dtype, shape, offset, width, aligned, needs_copy)
                current.wait_stream(stream)
                result[name].record_stream(current)
        return result

    def _field(self, packed, dtype, shape, offset, width, aligned, contiguous):
        raw = packed[:, offset:offset + width]
        # Byte-aligned records can require a device-side copy before a typed view.
        if not aligned:
            raw = raw.clone(memory_format=self.torch.contiguous_format)
        field = raw.view(dtype).reshape(packed.shape[0], *shape)
        return field.contiguous() if contiguous else field
