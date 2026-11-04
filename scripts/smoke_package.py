"""Exercise an installed wheel or sdist without optional frameworks."""

import sys
import tempfile
from pathlib import Path

import numpy as np

import pycellpack as cp

assert 'torch' not in sys.modules
with tempfile.TemporaryDirectory() as directory:
    for dtype in [np.uint16, np.float32]:
        x = np.array([[0, 2, 0], [1, 0, 3], [0, 4, 5]], dtype=dtype)
        matrix = cp.write(str(Path(directory) / f'{dtype.__name__}.cpk'), x)
        np.testing.assert_array_equal(matrix.read_dense(), x)
        np.testing.assert_array_equal(matrix[[2, 0], [2, 1]].read_dense(), x[np.ix_([2, 0], [2, 1])])
        assert matrix.read_csr().shape == x.shape
print('Installed package: native import, encoding, dense/CSR reads and lazy optional dependencies passed')
