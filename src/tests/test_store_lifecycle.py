"""File publication, descriptor identity, and failed writer cleanup."""

import errno
import gc
import multiprocessing
import pickle
import stat

import numpy as np
import pytest

from pycellpack import cellpack
from pycellpack.cellpack import _file as store


@pytest.mark.parametrize("mode", [0o640, 0o444])
def test_atomic_writer_publishes_only_on_finish(tmp_path, mode):
    path = tmp_path / "matrix.cpk"
    matrix = np.ones((4, 3), np.float32)
    old = cellpack.write(path, matrix)
    path.chmod(mode)
    with cellpack.Writer(path, 3, np.float32, format="dense", atomic=True) as writer:
        writer.append(matrix * 2)
        np.testing.assert_array_equal(cellpack.read(path), matrix)
    np.testing.assert_array_equal(cellpack.read(path), matrix * 2)
    np.testing.assert_array_equal(old.read_dense(), matrix)
    assert not old.file.is_current()
    assert stat.S_IMODE(path.stat().st_mode) == mode
    assert list(tmp_path.iterdir()) == [path]


def _child_writer(writer, action, connection):
    try:
        getattr(writer, action)()
        connection.send("ok")
    except RuntimeError as exc:
        connection.send(str(exc))
    finally:
        connection.close()


@pytest.mark.parametrize("action", ["abort", "finish"])
def test_forked_writer_does_not_publish_or_remove_parent_work(tmp_path, action):
    matrix = np.ones((4, 3), np.float32)
    path = tmp_path / "matrix.cpk"
    cellpack.write(path, matrix)
    with cellpack.Writer(path, 3, np.float32, format="dense", atomic=True) as writer:
        writer.append(matrix * 2)
        context = multiprocessing.get_context("fork")
        parent, child = context.Pipe()
        process = context.Process(target=_child_writer, args=(writer, action, child))
        process.start()
        child.close()
        try:
            assert parent.poll(5)
            message = parent.recv()
            process.join(5)
            assert process.exitcode == 0
            assert message == "ok" if action == "abort" else "process" in message
            np.testing.assert_array_equal(cellpack.read(path), matrix)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()
    np.testing.assert_array_equal(cellpack.read(path), matrix * 2)


@pytest.mark.parametrize("failure", ["construct", "append", "finish", "publish", "abort", "abandon"])
def test_atomic_failure_preserves_destination(tmp_path, monkeypatch, failure):
    path = tmp_path / "matrix.cpk"
    matrix = np.ones((4, 3), np.float32)
    cellpack.write(path, matrix)
    before = path.read_bytes()
    if failure == "construct":
        with pytest.raises(ValueError):
            cellpack.Writer(path, 3, np.float32, atomic=True, page_size=3)
    else:
        writer = cellpack.Writer(path, 3, np.float32, format="dense", atomic=True)
        writer.append(matrix * 2)
        if failure == "append":
            with pytest.raises(ValueError), writer:
                writer.append(matrix.astype(np.int32))
        elif failure in ("finish", "publish"):
            def fail(*args):
                raise OSError("injected write failure")

            if failure == "publish":
                monkeypatch.setattr(store.os, "replace", fail)
            else:
                native = writer._handle

                class FailedFinish:
                    finish = staticmethod(fail)
                    abort = native.abort

                writer._handle = FailedFinish()
            with pytest.raises(OSError, match="injected"), writer:
                pass
        elif failure == "abort":
            writer.abort()
            writer.abort()
        else:
            del writer
            gc.collect()
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_new_file_and_memory_roundtrip(tmp_path):
    matrix = np.ones((4, 3), np.float32)
    file = cellpack.write(tmp_path / "matrix.cpk", matrix, atomic=True)
    np.testing.assert_array_equal(file.read_dense(), matrix)
    memory = cellpack.compress(matrix)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(memory)).read_dense(), matrix)


def test_pickle_rejects_replaced_file(tmp_path):
    matrix = np.ones((4, 3), np.float32)
    file = cellpack.write(tmp_path / "matrix.cpk", matrix)
    payload = pickle.dumps(file)
    np.testing.assert_array_equal(pickle.loads(payload).read_dense(), matrix)
    cellpack.write(tmp_path / "matrix.cpk", matrix * 2, atomic=True)
    with pytest.raises(OSError, match="replaced"):
        pickle.loads(payload)


def test_identity_comes_from_opened_descriptor(tmp_path, monkeypatch):
    matrix = np.ones((4, 3), np.float32)
    path = tmp_path / "matrix.cpk"
    cellpack.write(path, matrix)
    replacement_path = tmp_path / "next.cpk"
    cellpack.write(replacement_path, matrix * 2)
    native = store._cellpack.File

    def replace_before_open(*args):
        replacement_path.replace(path)
        return native(*args)

    monkeypatch.setattr(store._cellpack, "File", replace_before_open)
    file = cellpack.File(path)
    assert file.is_current()
    assert file.to_bytes() == path.read_bytes()


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EPERM, errno.EMFILE])
def test_memfd_fallback_does_not_hide_exhaustion(monkeypatch, error):
    def fail(*args):
        raise OSError(error, "injected memfd failure")

    monkeypatch.setattr(store.os, "memfd_create", fail, raising=False)
    if error == errno.EMFILE:
        with pytest.raises(OSError) as caught:
            store._anonymous_fd()
        assert caught.value.errno == errno.EMFILE
    else:
        import ctypes

        monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: object())
        fd = store._anonymous_fd()
        store.os.close(fd)
