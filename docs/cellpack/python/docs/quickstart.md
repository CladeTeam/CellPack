# Matrix quickstart

This program needs only the core package and NumPy. In the Python project
created during [installation](installation.md), add the published package:

```sh
uv add pycellpack
```

Save the Python blocks below in `quickstart.py` and run `uv run quickstart.py`.
Use a writable directory; the program creates `matrix.cpk`.

<DocDiagram name="python-storage" />

## Write, reopen, and select

```python
import numpy as np
import pycellpack as cp

x = np.array([[0, 2, 0], [1, 0, 3]], dtype=np.float32)
stored = cp.write("matrix.cpk", x, workers=1, atomic=True)
assert stored.shape == (2, 3)

matrix = cp.Store("matrix.cpk")
view = matrix[[1, 0], [2, 0]]
actual = view.read_dense(workers=1)
np.testing.assert_array_equal(actual, [[3, 1], [0, 0]])
print(actual)
```

```text
[[3. 1.]
 [0. 0.]]
```

`write` encodes the matrix and returns a file-backed Store. `atomic=True` publishes the destination after success; it may replace an existing file. `Store(path)` opens an existing stream without decoding the matrix.

Selections are **orthogonal**: two row IDs and two column IDs produce a 2 × 2 matrix. Integers keep their axis, so `matrix[1, 2].shape` is `(1, 1)`. Order and repeated indices are preserved.

## Read dense or sparse output

```python
dense = matrix.read_dense(workers=1)  # NumPy ndarray
csr = matrix.read_csr(workers=1)      # cp.CSR, no SciPy required
assert csr.shape == x.shape
np.testing.assert_array_equal(dense, x)
# With the scipy extra installed:
# scipy_matrix = csr.to_scipy()
```

`read()` chooses the stored layout; use `read_dense()` or `read_csr()` when the result type matters. Decoding allocates output unless you supply buffers.

## Keep compression in memory

```python
memory = cp.compress(x, workers=1)
np.testing.assert_array_equal(memory.read_dense(workers=1), x)
assert memory.file.in_memory
```

This is convenient for matrix access. Training datasets require file-backed streams, so use the Store returned by `write` for training.

## Iterate training batches

Install PyTorch separately before running this section.

```python
from pycellpack import Accesses, ScDataset, ScDataLoader

dataset = ScDataset(matrix, ids=np.array([10, 20, 30], dtype=np.uint32))
order = Accesses(
    dataset_ids=np.array([0, 0, 0], dtype=np.uint32),
    rows=np.array([1, 0, 1], dtype=np.uint64),
)
loading = ScDataLoader(
    [dataset], order, out_ids=np.array([30, 10], dtype=np.uint32),
    batch_size=2, workers=1, device="cpu", backend="pread",
)
with loading:
    batches = list(loading)
assert [tuple(b["X"].shape) for b in batches] == [(2, 2), (1, 2)]
np.testing.assert_array_equal(batches[0]["X"].numpy(), [[3, 1], [0, 0]])
```

CPU Tensor batches own their storage and can be retained. Sample IDs refer to rows of the dataset view. Continue with [matrix access](storage.md), [gene alignment](alignment.md), or the full [training guide](dataloader.md).
