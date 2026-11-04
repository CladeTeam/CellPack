# Native process functions

A ProcessFn transforms one mapped row into a fixed-size output record inside a
native worker. Column mapping and conversion to the schema's input dtype happen
before the callback. Use it for tokenization or preprocessing that should run
without a Python callback for every row.

<DocChoices title="Choose a processing path" :items="[
  { label: 'Use a model preset', when: 'The shipped template matches your preprocessing.', href: '#inspect-a-preset-on-the-cpu', detail: 'Provide token tables in output-panel order and inspect named NumPy fields first.' },
  { label: 'Bind a custom callback', when: 'You own the native transform or record layout.', href: '#declare-a-custom-output-record', detail: 'Declare the exact output bytes and bind C/C++ or Numba functions.' },
]" />

<DocDiagram name="process-path" />

## Inspect a preset on the CPU

This complete example needs PyCellPack, NumPy and a C++17 compiler. It does not
require PyTorch. Run it in a writable directory; it creates or replaces
`tokens.cpk`. It deliberately changes the output panel order before processing.

```python
import numpy as np
import pycellpack as cp
from pycellpack.loader.process_fn import presets

store = cp.write("tokens.cpk", np.array([[1, 2, 0], [0, 3, 4]], np.float32),
                 format="csr", workers=1)
source_ids = np.array([10, 20, 30], np.uint32)
out_ids = np.array([30, 10, 20], np.uint32)
source = cp.DataSource([(store, source_ids)], vocab=31)
process = presets.scgpt(
    tokens=np.array([12, 10, 11], np.int64),  # One token per out_ids entry.
    pad_token=0, cls_token=1, length=8, mask_ratio=0, seed=7,
)
loading = cp.Loader(source, process.schema(out_ids, name="tokens"),
                    workers=1, backend="pread", slot_rows=2)
retained = []
with loading.start(np.array([1, 0], np.uint32), slots=2) as run:
    for batch in run:
        retained.append(process.unpack(batch.region("tokens"), copy=True))
    run.join()
fields = retained[0]
assert fields["genes"].shape == fields["values"].shape == fields["targets"].shape == (2, 8)
assert fields["genes"].dtype == np.int64
assert fields["values"].dtype == fields["targets"].dtype == np.float32
np.testing.assert_array_equal(fields["genes"][:, 0], [1, 1])
print({name: array.shape for name, array in fields.items()})
```

```text
{'genes': (2, 8), 'values': (2, 8), 'targets': (2, 8)}
```

`source_ids` maps stored columns to standard IDs. `out_ids` sets their order at
callback entry. Consequently `tokens=[12, 10, 11]` addresses standard IDs
`[30, 10, 20]`, rather than stored-column order or indices into a length-31
vocabulary. Apply the same reordering to every auxiliary table.

The copies in `retained` own their storage. With the default `copy=False`, named
fields are views that retain the native slot lease; finish using them before
requesting enough batches to fill the ring, or copy them before keeping them.
See [rings and lifecycle](/python/api/rings.md).

## Choose and configure a preset

| Factory | Transform and required configuration | Named output |
| --- | --- | --- |
| `scgpt` | Quantile binning and masking; tokens, pad and CLS IDs | `genes`, `values`, `targets` |
| `geneformer` | Median-normalized ranking; tokens, medians, pad/CLS/EOS IDs | `ids`, `length` |
| `uce` | Chromosome-grouped sampling; tokens, chromosomes, starts and special IDs | `ids`, `loss_ids`, `labels`, `length` |
| `state_se` | Sentences and expression queries; tokens and global token-vocabulary size | `ids`, `tasks`, `targets`, `counts`, `total` |

Factories snapshot tables and return a ProcessFn with an OutputLayout. Input is
float32, and the mapped panel width must equal the token-table width. Negative
tokens exclude dimensions. Defaults implement the shipped templates; they are
not inferred from a model checkpoint. Supply the model's actual vocabulary,
sampling settings and special tokens.

Geneformer excludes nonpositive medians. UCE also imposes per-cell eligibility
and sentence-length conditions, reported through native status codes. STATE SE's
`batch` groups consecutive access positions for shared query draws; match it to
the intended training microbatch. See the [preset contracts](/python/api/process-functions.md#model-presets)
for output shapes and configuration limits.

For Tensor training, pass a declared ProcessFn to `ScDataLoader(process=...,
out_ids=...)`; its batches expose the named fields directly. That path requires
PyTorch. The [DataLoader guide](dataloader.md) covers iteration and transfer,
while the [runnable process program](/python/training/process-functions.md)
demonstrates direct NumPy inspection and explicit packed CUDA transfer.

## Declare a custom output record

The layout describes bytes written by your callback; it does not generate or
validate that callback's writes. This declaration mirrors a native record with
128 int64 IDs followed by a uint32 length:

```python
from pycellpack.loader.process_fn import OutputLayout

layout = OutputLayout([("ids", "i8", (128,)), ("length", "u4")])
assert layout.offsets == {"ids": 0, "length": 1024}
assert layout.row_bytes == layout.dtype.itemsize == 1032
```

The last four bytes are C struct tail padding. The callback must match field
offsets, numeric types and the complete record size. Use `align=False` only for
an explicitly matching packed record; explicit dtype offsets and itemsize are
also supported.

Choose `ProcessFn.from_source(..., output=layout)` to compile C/C++,
`from_library` for exported functions in an existing library, or `from_cfunc`
for Numba callbacks. `process.schema(out_ids, name=...)` then infers the record
size. An opaque function without an OutputLayout needs explicit
`OutputSchema(row_bytes=..., process=...)` and has no named `unpack()` support.
The [API reference](/python/api/process-functions.md#construction-and-identity)
lists the binding and compiler options.

## Respect the input and callback contract

Use `row_format="csr"` or `"dense"` for one implementation. `"auto"` requires
both callbacks to produce equivalent output; delivery may depend on stored
layout and execution state. A CSR row's mapped columns need not be sorted.
Dense rows contain zero at present-but-zero columns and the schema default at
source-absent columns. CSR alone cannot distinguish those omitted cases.

Callbacks must be thread-safe, write only their output and declared scratch,
retain no borrowed pointers, and return a nonzero status on failure instead of
unwinding across the ABI. `partial_rows=True` lets a CSR callback receive null
column/value pointers and resolve requested elements; opt in only when that
implementation handles the resolver protocol.

`owner` keeps native objects alive. `key` or `identity` must separately cover
all configuration affecting the result. Recipe-based bindings can reconstruct
picklable state in another process; raw addresses and hand-made capsules cannot.
For the full ABI and build failures, see the [process API](/python/api/process-functions.md).
