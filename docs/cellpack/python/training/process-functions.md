# Native process functions

Run this program to inspect a scGPT preset's named outputs, then optionally move
one packed batch to CUDA. It creates its own two-row CSR fixture in a temporary
directory and needs no model checkpoint or training data.

## Choose a run

In your Python project, install PyCellPack from PyPI and save the
[downloadable program](/downloads/examples/processfn-example.py) as
`processfn-example.py`. With a C++17 compiler available, run:

```sh
uv add pycellpack
uv run processfn-example.py
uv run processfn-example.py --device cuda
```

| Mode | Requirements | What you inspect |
| --- | --- | --- |
| Default CPU | PyCellPack, NumPy and a C++17 compiler | Named NumPy field views; PyTorch is not needed |
| `--device cuda` | The same, plus compatible PyTorch/CUDA and a GPU | One packed H2D copy, then named contiguous GPU Tensor fields |

The preset uses a length of 8 and requests cells `[1, 0]` in one batch. Both modes
produce `genes` as int64 and `values`/`targets` as float32, each shaped `(2, 8)`.
The first gene token is CLS (`1`); the padding token is `0`. Negative table entries
exclude dimensions. Input values are quantile-binned and masked by the preset,
so do not expect the `values` field to reproduce raw counts.

`process.output` describes a 128-byte row: 64 bytes of genes, 32 of values and
32 of targets. The CPU path unpacks that record without copying. The CUDA path
uploads the complete record before forming field views, avoiding separate host
field rearrangement and transfers.

## Follow the storage lifetime

<DocDiagram name="process-path" />

1. The native workers write a leased output slot.
2. CPU inspection finishes using its field views before returning. Retaining
   them would keep the slot leased; use `unpack(copy=True)` to keep CPU results.
3. CUDA upload retains the raw NumPy array and its zero-copy host Tensor until
   the copy stream finishes reading them. `non_blocking=True` alone is not a
   completion signal.
4. Once host readers are gone, the slot is reusable. Device fields retain GPU
   storage, and `wait_stream` / `record_stream` govern their separate consumers.

The example explicitly synchronizes the copy stream before releasing host
storage; it does not yield an unfinished host-to-device transfer to the caller.
Pinned output memory is allocated only in CUDA mode.

## Complete source

::: details Complete program: preset, CPU inspection and packed CUDA transfer

<<< @/../../example/processfn-example.py

:::

[Download the source](/downloads/examples/processfn-example.py).
For a smaller CPU example with assertions, use the
[process guide](/python/docs/process-functions.md#inspect-a-preset-on-the-cpu).
For factory arguments, field-view shapes and callback rules, use the
[process API](/python/api/process-functions.md).
