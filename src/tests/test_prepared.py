"""Prepared components, physical read policies and run ownership."""

import subprocess
import sys

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.cellpack import scans


@pytest.mark.parametrize("dtype", ["uint16", "float32", "float64"])
@pytest.mark.parametrize("values", ["raw", "dict", "modes"])
def test_prepared_codec_snapshot_and_fused_mapping(dtype, values):
    x = np.tile(np.arange(32, dtype=dtype) % 5, (12, 1))
    x[3, 5] = 17
    opts = scans.EncodeOptions(values=values, layout="sparse", column_modes=0.0)
    expected = scans.encode_chunk(x, options=opts)
    prepared = scans.PreparedEncoder(x, options=opts)
    x[:] = 0
    out = np.empty(prepared.output_bytes, np.uint8)
    ws = np.empty(prepared.workspace_words, np.uint64)
    for _ in range(2):
        encoded = prepared.encode(out=out, workspace=ws)
        np.testing.assert_array_equal(encoded, expected)
    meta = scans.Meta(encoded, source_dtype=dtype)
    panel = np.full(32, -1, np.int64)
    panel[[5, 2, 7]] = [0, 2, 3]
    decoder = scans.Decoder(meta, gene_map=panel, n_genes=5, dtype=np.float64)
    panel[:] = -1
    original = scans.decode_chunk(expected, source_dtype=dtype)
    want = np.zeros((12, 5))
    want[:, [0, 2, 3]] = original[:, [5, 2, 7]]
    np.testing.assert_array_equal(decoder.decode_dense(), want)
    selected = np.array([8, 3, 8], np.uint32)
    target = np.empty((3, 5), np.float64)
    assert decoder.decode_dense(rows=selected, out=target) is target
    np.testing.assert_array_equal(target, want[selected])
    row = np.empty(5, np.float64)
    assert decoder.decode_record(meta.record(3), out=row) is row
    np.testing.assert_array_equal(row, want[3])
    with pytest.raises(ValueError):
        decoder.decode_dense(out=np.empty((12, 5), np.float32))


@pytest.mark.parametrize("strategy", ["auto", "ranges", "whole"])
def test_read_plan_regions_duplicates_and_summary(tmp_path, strategy):
    x = np.arange(256 * 48, dtype=np.float32).reshape(256, 48)
    file = cellpack.write(tmp_path / "plan.cpk", x, chunk_rows=256)
    options = cellpack.ReadOptions(strategy=strategy)
    select = file[[200, 1, 200]][:, [6, 2, 0]].prepare(read_options=options)
    np.testing.assert_array_equal(select.read_dense(), x[[200, 1, 200]][:, [6, 2, 0]])
    np.testing.assert_array_equal(
        select.read_csr().to_scipy().toarray(), x[[200, 1, 200]][:, [6, 2, 0]]
    )
    summary = select.io_summary
    assert summary.read_bytes >= summary.record_bytes
    if strategy == "ranges":
        assert summary.read_bytes < summary.whole_chunk_bytes
    if strategy == "whole":
        assert summary.read_bytes == summary.whole_chunk_bytes
    np.testing.assert_array_equal(
        file[2:5].read_dense(read_options=options), x[2:5]
    )


@pytest.mark.parametrize("layout", ["dense", "sparse"])
def test_dense_output_and_resources(tmp_path, layout):
    x = np.arange(160, dtype=np.float32).reshape(20, 8)
    file = cellpack.write(tmp_path / "direct.cpk", x, options=scans.EncodeOptions(layout=layout))
    loading = loader.Loader(
        loader.DataSource([(file, np.arange(8))], 8),
        config=loader.LoaderConfig(workers=1, backend="pread", slot_rows=4),
    )
    estimate = loading.memory_estimate(slots=2)
    assert estimate.output_ring_bytes >= 2 * 4 * 8 * 4
    assert estimate.known_reservation_bytes >= estimate.private_frame_bytes > 0 and estimate.cache_frame_bytes == 0
    outputs = []
    with loading.start(np.arange(20, dtype=np.uint32), slots=2) as run:
        for batch in run:
            with batch:
                outputs.append(batch.data.copy())
        stats = run.join()
    np.testing.assert_array_equal(np.concatenate(outputs), x)
    assert stats.written_bytes == x.nbytes
    assert stats.process_calls == 0


@pytest.mark.parametrize("mode", ["close", "drop", "exception", "exit"])
def test_shutdown_during_python_access_callback(tmp_path, mode):
    program = r"""
import ctypes,threading,time,sys
from pathlib import Path
import numpy as np
from pycellpack import cellpack,loader
f=cellpack.write(Path(sys.argv[1])/'shutdown.cpk',np.arange(16,dtype=np.float32).reshape(4,4))
loading=loader.Loader(loader.DataSource([(f, np.arange(4))], 4),workers=1,inflight=1,backend='pread',slot_rows=4)
event=threading.Event()
@ctypes.CFUNCTYPE(ctypes.c_int32,ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint32))
def callback(ctx,start,count,out):
 event.set();time.sleep(.05)
 for i in range(count):out[i]=(start+i)%4
 return 0
seq=loader.Accesses.function(ctypes.cast(callback,ctypes.c_void_p).value,0,128,owner=callback)
run=loading.start(seq,slots=2)
assert event.wait(5)
if sys.argv[2]=='close':run.close()
elif sys.argv[2]=='drop':del run
elif sys.argv[2]=='exception':
 try:
  with run: raise ValueError('consumer failed')
 except ValueError: pass
print('closed',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "closed" in result.stdout


def test_prepared_model_encodes_independent_records():
    x = np.tile(np.array([0, 3, 2, 0, 1, 0], np.uint16), (5, 1))
    encoder = scans.PreparedEncoder(x, options=scans.EncodeOptions(layout="sparse"))
    meta = encoder.meta
    tables = scans.EncodeTables(meta)
    record = scans.encode_record(
        meta, x[2], tables=tables, options=scans.EncodeOptions(layout="sparse")
    )
    np.testing.assert_array_equal(scans.decode_record_dense(meta, record), x[2])


def test_read_plan_preserves_embedded_region(tmp_path):
    x = np.arange(4096, dtype=np.uint16).reshape(128, 32)
    cellpack.write(tmp_path / "inner.cpk", x, chunk_rows=128)
    payload = (tmp_path / "inner.cpk").read_bytes()
    outer = tmp_path / "outer.bin"
    outer.write_bytes(b"x" * 8192 + payload + b"y" * 8192)
    embedded = cellpack.File(outer, offset=8192, length=len(payload))
    for strategy in ("ranges", "whole", "auto"):
        selection = embedded.prepare_read(
            [120, 0, 120], read_options=cellpack.ReadOptions(strategy=strategy)
        )
        np.testing.assert_array_equal(selection.read_dense(), x[[120, 0, 120]])
