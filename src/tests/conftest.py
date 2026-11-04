import signal

import pytest


@pytest.fixture
def python_sigint():
    """Python's KeyboardInterrupt handler on SIGINT for the test.

    `_thread.interrupt_main` does nothing while SIGINT is ignored, which is
    how a non-interactive shell starts a background job; tests that rely on
    it install the handler themselves.
    """
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


@pytest.fixture(autouse=True)
def cellpack_home(tmp_path_factory, monkeypatch):
    """An empty `CELLPACK_HOME` per test: cost lookups never see this
    machine's records."""
    home = tmp_path_factory.mktemp("cellpack-home")
    monkeypatch.setenv("CELLPACK_HOME", str(home))
    monkeypatch.delenv("CELLPACK_COST_MODEL", raising=False)
    return home
