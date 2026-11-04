"""Lazy views preserve their selections and caller-owned output contracts."""

import pickle

import numpy as np
import pytest

from pycellpack import cellpack


@pytest.mark.parametrize("invalid", ["dtype", "readonly", "unaligned"])
def test_repeated_columns_validate_output_before_writing(tmp_path, invalid):
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3) + 0.25
    view = cellpack.Store(cellpack.write(tmp_path / "data.cpk", matrix))[:, [2, 2]]
    out = np.full(view.shape, -7, np.float32)
    if invalid == "dtype":
        out = out.astype(np.int32)
    elif invalid == "readonly":
        out.flags.writeable = False
    else:
        out = np.ndarray(view.shape, np.float32, buffer=bytearray(out.nbytes + 1), offset=1)
        out.fill(-7)
    before = out.copy()
    with pytest.raises(ValueError):
        view.read_dense(out=out)
    np.testing.assert_array_equal(out, before)


@pytest.mark.parametrize("axis", ["cells", "genes"])
@pytest.mark.parametrize("serialized", [False, True])
def test_selection_metadata_does_not_allow_mutating_shared_views(tmp_path, axis, serialized):
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3)
    view = cellpack.Store(cellpack.write(tmp_path / "data.cpk", matrix))[[3, 1], [2, 0]]
    if serialized:
        view = pickle.loads(pickle.dumps(view))
    alias = view.copy()
    selection = getattr(view, axis)
    with pytest.raises(ValueError):
        selection[0] = 0
    with pytest.raises(ValueError):
        selection.setflags(write=True)
    np.testing.assert_array_equal(alias.toarray(), matrix[[3, 1]][:, [2, 0]])
