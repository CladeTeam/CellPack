"""Pinned output views keep their slot alive through asynchronous CUDA copies."""

import gc

import numpy as np
import pytest

from pycellpack import cellpack, loader

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("backend", ["pread", "uring"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pinned_output_survives_close_until_cuda_copy_finishes(tmp_path, backend, dtype):
    if backend == "uring" and not loader._config.uring_supported():
        pytest.skip("io_uring unavailable")
    x = np.arange(7 * 64, dtype=np.float32).reshape(7, 64)
    path = cellpack.write(tmp_path / "x.cpk", x)
    source = loader.DataSource([(path, np.arange(64))], 64)
    loading = loader.Loader(source, loader.OutputSchema(np.arange(64), dtype),
                            workers=2, backend=backend, slot_rows=7)
    layout = loading.ring_layout(slots=1)
    storage = torch.empty(layout.data_bytes, dtype=torch.uint8, pin_memory=True)
    ring = loader.Ring(layout, data=storage.numpy())
    cells = np.array([6, 2, 0, 2, 4, 1, 5], np.uint32)
    stream = torch.cuda.Stream(device=0)
    run = loading.start(cells, ring=ring)
    try:
        batch = next(run)
        observer = batch.observer()
        view = batch.data
        host = torch.from_numpy(view)
        assert host.is_pinned()
        with torch.cuda.stream(stream):
            device = host.to("cuda:0", non_blocking=True)
        batch.release()
        run.join()
    finally:
        run.close()
        stream.synchronize()
    assert observer.held
    np.testing.assert_array_equal(device.cpu().numpy(), x[cells].astype(dtype))
    del host, view
    gc.collect()
    assert not observer.held
