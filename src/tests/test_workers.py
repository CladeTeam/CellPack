"""Worker counts follow this process's CPU allowance."""

import pytest

from pycellpack import _workers


def test_default_worker_count_is_within_the_allowance():
    cpus = _workers.available_cpus()
    assert cpus >= 1
    assert 1 <= _workers.default_workers() <= cpus


def test_explicit_worker_count_is_capped_and_validated():
    cpus = _workers.available_cpus()
    assert _workers.resolve_workers(None) == _workers.default_workers()
    assert _workers.resolve_workers(1) == 1
    assert _workers.resolve_workers(cpus + 100) == cpus
    with pytest.raises(ValueError):
        _workers.resolve_workers(0)
    with pytest.raises(ValueError):
        _workers.resolve_workers(-1)


def test_cgroup_quota_is_a_positive_count_or_unknown():
    quota = _workers._cgroup_quota()
    assert quota is None or quota >= 1
