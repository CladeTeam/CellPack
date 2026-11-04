# Compose chunk and record codecs

Use `scans` when your application owns the container, buffer allocation or
record reads. A chunk contains compression metadata plus independently decodable
rows; it is not a complete `.cpk` matrix file. For file I/O and lazy selections,
start with [matrix access](storage.md).

<DocChoices title="Choose how much control you need" :items="[
  { label: 'Round-trip a chunk', when: 'You already have a matrix in memory.', href: '#encode-and-check-a-small-matrix', detail: 'Encode once, then check values, row ordering and repeated selections.' },
  { label: 'Reuse codec state', when: 'Decode many rows from the same chunk.', href: '#reuse-buffers-and-models', detail: 'Reuse parsed metadata, tables and scratch, or bind a Decoder for a fixed panel.' },
  { label: 'Map individual records', when: 'Read one row into a custom panel.', href: '#map-one-record-into-a-panel', detail: 'Distinguish source and coded columns before composing the map.' },
]" />

## Encode and check a small matrix

Run these blocks in order. They need only NumPy and the core package, create no
files, and use one decoding worker for a small correctness check.

```python
import numpy as np
from pycellpack.cellpack import scans

x = np.array([[0, 2, 0, 4], [1, 0, 3, 0], [0, 0, 0, 0]], np.float32)
chunk = scans.encode_chunk(x)
meta = scans.Meta(chunk, source_dtype=x.dtype)
assert meta.shape == (3, 4)
selected = scans.decode_chunk(meta, rows=[1, 0, 1], workers=1)
np.testing.assert_array_equal(selected, x[[1, 0, 1]])
print(selected)
```

```text
[[1. 0. 3. 0.]
 [0. 2. 0. 4.]
 [1. 0. 3. 0.]]
```

The repeated row remains repeated. `rows` uses positions within this chunk,
rather than global loader cell IDs. Raw chunks encode value width without
recording whether those bits are floats or signed integers. Keep the original
dtype with your container: `source_dtype=x.dtype` supplies that interpretation.
An output `dtype=` instead converts interpreted numeric values.

Use `format="csr"` on `decode_chunk`, or call `decode_csr(meta)`, when you need
the package's NumPy-backed CSR instead of a dense ndarray. No SciPy is required.

## Reuse buffers and models

This continues with the same chunk. Dense output has an exact shape/dtype;
workspace capacity counts uint64 words:

```python
tables = scans.DecodeTables(meta)
workspace = np.empty(meta.decode_workspace_words, np.uint64)
out = np.empty((2, 4), np.float32)
returned = scans.decode_dense(
    meta, rows=[1, 0], tables=tables, out=out, workspace=workspace, workers=1,
)
assert returned is out
np.testing.assert_array_equal(out, x[[1, 0]])

input_copy = x.copy()
encoder = scans.PreparedEncoder(input_copy)
input_copy.fill(-1)
encoded_again = encoder.encode()
np.testing.assert_array_equal(
    scans.decode_chunk(encoded_again, source_dtype=x.dtype, workers=1), x,
)
```

PreparedEncoder snapshots input before constructing its model. Its pre-encoding
`meta` is for model/table reuse, not for locating finished records; parse the
returned chunk before decoding cells. Each concurrent encode needs independent
output and workspace buffers.

`Fill.ZERO` is the default dense fill. Only choose `"assume_zeroed"` when a
supplied output is already all zero. Reusing a previous result without clearing
it can leave stale values. For mapped rows or output conversion, bind
`Decoder(meta, gene_map=..., dtype=...)`; it reuses its own tables and scratch.
Use one Decoder per concurrent consumer.

## Map one record into a panel

The source-to-output map below keeps source column 2 at output 0 and source
column 0 at output 1. Output column 2 has no source and stays zero:

```python
source_to_panel = np.array([1, -1, 0, -1], np.int64)
parts = scans.decode_parts(meta, cell=1, space=scans.IdSpace.SOURCE)
panel = scans.map_dense(parts, source_to_panel, n_genes=3, dtype=np.float64)
np.testing.assert_array_equal(panel, [3, 1, 0])
assert panel.dtype == np.float64

coded_parts = scans.decode_parts(meta, cell=1)  # Default space is CODED.
coded_to_panel = scans.compose_mapping(tables, source_to_panel, n_genes=3)
np.testing.assert_array_equal(scans.map_dense(coded_parts, coded_to_panel), panel)

decoder = scans.Decoder(meta, gene_map=source_to_panel, n_genes=3, dtype=np.float64)
np.testing.assert_array_equal(decoder.decode_record(cell=1), panel)
```

Each chunk may have a different compression permutation. Compose that chunk's
map once with its DecodeTables; do not apply an original source-column map
directly to coded-space Parts. This map addresses panel positions, unlike a
DataSource's shared vocabulary map.

Python Parts owns copied record data and scratch. Its component properties are
read-only snapshots; retaining Parts does not retain a loader slot or borrow
your original input record. Use a Decoder when you only need mapped output
and can avoid creating those inspection objects.

## Integrate a custom container

Keep each record with its matching metadata, coding tables and numeric dtype.
You can retain metadata while reading records separately, then pass record bytes
or `PagedRecord` to the record decoders. `pack_chunk` supports page-oriented
placement; parse the packed result again because offsets change.

Metadata validation is not a sanitizer for arbitrary compressed record bytes.
Use encoder-produced immutable data for the matching model. The
[codec API](/python/api/codecs.md) specifies buffer capacity, mapping order,
paged input, fill, ownership and failure rules.
