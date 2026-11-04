"""Descriptor service failures must not strand peers or leak sockets."""

import os
import socket
import threading
import uuid

import pytest

from pycellpack.loader import _transport as transport


def test_failed_peer_does_not_stop_descriptor_service(tmp_path, monkeypatch):
    fd = os.open(tmp_path / "region", os.O_CREAT | os.O_RDWR, 0o600)
    send = transport.socket.send_fds
    attempts = 0

    def send_to_peer(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise BrokenPipeError("peer disconnected")
        return send(*args, **kwargs)

    monkeypatch.setattr(transport.socket, "send_fds", send_to_peer)
    server = transport.FdServer({"region": fd})
    try:
        with pytest.raises(ValueError):
            transport.receive_fds(server.address, timeout=1)
        received = transport.receive_fds(server.address, timeout=1)
        try:
            assert os.fstat(received["region"]).st_ino == os.fstat(fd).st_ino
        finally:
            for descriptor in received.values():
                os.close(descriptor)
    finally:
        server.stop()
        os.close(fd)


def test_server_start_failure_closes_socket(monkeypatch):
    sockets = []
    socket = transport.socket.socket

    def record(*args, **kwargs):
        result = socket(*args, **kwargs)
        sockets.append(result)
        return result

    def fail(*args):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(transport.socket, "socket", record)
    monkeypatch.setattr(transport.threading.Thread, "start", fail)
    try:
        with pytest.raises(RuntimeError, match="cannot start thread"):
            transport.FdServer({})
        assert sockets and all(s.fileno() == -1 for s in sockets)
    finally:
        for s in sockets:
            s.close()


@pytest.mark.parametrize("parts", [(b'["reg', b'ion"]'), (b'{', b''), (b'["same","same"]', b'')])
def test_descriptor_receive_handles_fragmentation_and_closes_on_error(tmp_path, monkeypatch, parts):
    fd = os.open(tmp_path / "region", os.O_CREAT | os.O_RDWR, 0o600)
    address = "\0pycellpack-test-" + uuid.uuid4().hex
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(address)
    listener.listen(1)
    listener.settimeout(5)
    first_read = threading.Event()
    received = []
    recv = socket.recv_fds
    valid = parts[1] != b''

    def record(*args, **kwargs):
        result = recv(*args, **kwargs)
        received.extend(result[1])
        first_read.set()
        return result

    def serve():
        with listener.accept()[0] as peer:
            socket.send_fds(peer, [parts[0]], [fd] if valid or parts[0] == b'{' else [fd, fd])
            if parts[1]:
                assert first_read.wait(5)
                try:
                    peer.sendall(parts[1])
                except BrokenPipeError:
                    pass

    monkeypatch.setattr(socket, "recv_fds", record)
    thread = threading.Thread(target=serve)
    thread.start()
    try:
        if valid:
            result = transport.receive_fds(address, timeout=2)
            assert set(result) == {"region"}
            assert not os.get_inheritable(result["region"])
        else:
            with pytest.raises((OSError, ValueError)):
                transport.receive_fds(address, timeout=2)
            for descriptor in received:
                with pytest.raises(OSError):
                    os.fstat(descriptor)
    finally:
        first_read.set()
        thread.join(5)
        listener.close()
        os.close(fd)
        for descriptor in received:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("error", [KeyboardInterrupt, ValueError])
def test_region_creation_failure_cleans_earlier_regions(tmp_path, monkeypatch, error):
    region = transport.Region
    allocated = []

    def create(*args, **kwargs):
        if allocated:
            raise error("interrupted allocation")
        result = region(*args, **kwargs)
        allocated.append(result)
        return result

    monkeypatch.setenv("PYCELLPACK_SHM_DIR", str(tmp_path))
    monkeypatch.setattr(transport, "Region", create)
    try:
        with pytest.raises(error, match="interrupted allocation"):
            transport.choose({"a": 4096, "b": 4096}, uuid.uuid4().hex, "file")
        assert allocated[0].fd == -1
        assert not allocated[0].path.exists()
    finally:
        for r in allocated:
            r.close()
            r.unlink()
