# Python examples

| Example | Purpose | Requirements |
|---|---|---|
| [`dataloader-example.py`](dataloader-example.py) | ScDataset corpus, ordinary Tensor iteration, external sample ordering and optional native AccessFn | Installed `pycellpack`, Torch and a C++17 compiler; optional CUDA |
| [`processfn-example.py`](processfn-example.py) | Runnable scGPT preset: CPU field views or one packed H2D transfer followed by GPU field processing | Installed `pycellpack` and a C++17 compiler; CUDA mode also needs PyTorch/CUDA |
| [`loader-example.py`](loader-example.py) | Full scGPT integration: exported `.so` functions, handwritten C++ ABI, independent Numba callbacks, pinned output ring, GPU transfer and training | Prepared data and vocabulary, PyTorch/CUDA, Numba, official `scgpt`, and a C++17 compiler or compiled provider |

For training iteration, use the DataLoader example on an existing worker:

```sh
uv run example/dataloader-example.py
uv run example/dataloader-example.py --access-fn
uv run example/dataloader-example.py --device cuda
```

It combines datasets with different panels, supplies their standard ids,
and returns named Tensors with no explicit ring or release operations.
The example constructs its order with NumPy outside the loader. `--access-fn`
binds caller-provided C++ code reading those arrays through the corpus C ABI.
Each epoch replaces the input externally; the loader manages no sampling policy.
CUDA mode transfers one packed buffer and retires host leases before GPU
field processing. Torch remains optional at package import time.

Run the self-contained ProcessFn CPU example from the repository root on an existing
worker:

```sh
uv run example/processfn-example.py
```

It writes a small CSR `.cpk` in a temporary directory, creates a data source with
an explicit column map, constructs a schema with `process.schema(...)`, and
prints zero-copy fields from `process.unpack(...)`. The inspection helper
drops its field views before releasing the batch. Set `TMPDIR` to a
shared-storage scratch directory when local temporary storage is constrained.

For the recommended GPU training data path, run on an existing GPU worker:

```sh
uv run example/processfn-example.py --device cuda
```

Workers write one packed output into a pinned ring. A NumPy slot view and a
zero-copy host Tensor retain that slot through one complete H2D transfer.
Once the copy stream finishes, the batch is released and those host views
are dropped. Field views are constructed on the GPU; each field's contiguous
copy uses a separate CUDA stream, with dependencies and allocation lifetimes
recorded for the consuming stream. GPU field processing can continue while
the producer reuses the host slot. `copy=True` is an explicit option for
retaining independent CPU arrays.

`loader-example.py` is intentionally illustrative rather than directly runnable
or importable: fill its data/vocabulary/access placeholders and choose one of
the three ProcessFn bindings. Its explicit native configuration, output offsets
and GPU training loop show the full integration. The pinned-ring source stays
leased until its copy stream has finished reading it.

The [Python API](../docs/python-api.md) describes the current loader and
ProcessFn interfaces. [Rust tools](../crates/cellpack/examples/README.md) cover
codec, planning and loader experiments; [Rust API tutorials](../docs/cellpack/index.md)
use the core crate directly.
