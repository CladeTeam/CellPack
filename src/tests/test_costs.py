import ctypes
import json
import subprocess
import sys
import warnings

import numpy as np
import pytest

from pycellpack import cellpack, loader
from pycellpack.loader import costs
from pycellpack.loader.costs import _code, _collect, _fit, _resolve
from pycellpack.loader.opt import CostModel, InputCost, OutputCost, ProcessCost, RowCost


def model():
    return CostModel(
        cores=16, iops=8e5, bandwidth=3e9, memory_bandwidth=5e10, access=9.5e-7, request=2.5e-6,
        compile=1.2e-7, serial=0.05, input=InputCost(decode_element=1.6e-9, cold=1e-6), rows=RowCost(gather=4e-11, admit=3e-10),
        outputs={"X": OutputCost(map_element=7.8e-10), "fn": OutputCost(
            process=2e-6, resolve=1e-6, curve=(ProcessCost(512, 1.5e-6, 1e-7), ProcessCost(2**32 - 1, 3e-6, 2e-7)))},
        convert=(0.0, 5e-11, 0.0, 5e-11, 1e-11, 0.0),
        provenance={"host": "default"},
    )


def test_cost_model_json_round_trip(tmp_path):
    m = model()
    d = m.to_dict()
    assert d["outputs"][1]["name"] == "fn" and d["outputs"][1]["curve"][0] == {"max_elements": 512, "process": 1.5e-6, "resolve": 1e-7}
    assert CostModel.from_dict(json.loads(json.dumps(d))) == m
    assert CostModel.load(m.save(tmp_path / "m.json")) == m
    unnamed = dict(d, outputs=[{k: v for k, v in o.items() if k != "name"} for o in d["outputs"]])
    assert list(CostModel.from_dict(unnamed).outputs) == [0, 1]
    with pytest.raises(ValueError, match="format"):
        CostModel.from_dict(dict(d, format=99))
    # Fields this version does not know are ignored.
    assert CostModel.from_dict(dict(d, retired_rate=1.0)) == m


def test_return_scans_by_architecture():
    assert _code.body_length(bytes([0x55, 0x01, 0xC3, 0x90]), "x86_64") == 3
    assert _code.body_length(bytes([0x90, 0xC2, 0x08, 0x00, 0x90]), "x86_64") == 4
    ret = (0xD65F03C0).to_bytes(4, "little")
    assert _code.body_length(bytes(4) + ret + bytes(4), "aarch64") == 8
    assert _code.body_length(bytes(1000), "riscv64") == _code.FALLBACK_BYTES


def test_pointer_masking_zeroes_addresses_into_mappings():
    maps = _code.Mappings("7f0000000000-7f0000001000 r-xp 00000000 00:00 0\n")
    inside = (0x7F0000000010).to_bytes(8, "little")
    outside = (0x1234).to_bytes(8, "little")
    code = b"\x48\xb8" + inside + b"\x48\xb8" + outside
    masked = _code.mask_pointers(code, maps)
    assert masked == b"\x48\xb8" + bytes(8) + b"\x48\xb8" + outside


def _libc(name):
    return ctypes.cast(getattr(ctypes.CDLL(None), name), ctypes.c_void_p).value


def test_code_signatures_are_stable_across_processes_and_distinguish_functions():
    here = costs.signature(_libc("strlen"))
    assert here == costs.signature(_libc("strlen"))
    assert here != costs.signature(_libc("memcmp"))
    script = ("import ctypes; from pycellpack.loader import costs; "
              "print(costs.signature(ctypes.cast(ctypes.CDLL(None).strlen, ctypes.c_void_p).value))")
    there = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip()
    assert there == here


def _feed(steady, rates, reads_per_access=0.0, dt=0.5, jitter=0.0):
    t = acc = reads = 0.0
    rng = np.random.default_rng(1)
    for r in rates:
        step = dt + jitter * rng.uniform(-1, 1)
        t += step
        acc += r * step
        reads += r * step * reads_per_access
        if steady.push(t, acc, reads):
            return steady.result
    return steady.result


def test_steady_window_excludes_warmup():
    rates = [1000 * min(1.0, 0.1 + i / 40) for i in range(40)] + [1000.0] * 200
    w = _feed(costs.Steady(10, window=4, min_warmup=1), rates)
    assert w.steady and w.restarts == 0
    assert w.seconds >= 10 and w.first >= 40  # after the ramp (index 40 is t = 20.5 s)


def test_steady_restarts_after_a_step_and_falls_back_when_never_quiet():
    rates = [1000.0] * 30 + [600.0] * 200
    w = _feed(costs.Steady(20, window=4, min_warmup=1), rates)
    assert w.steady and w.restarts >= 1 and w.first > 30
    noisy = ([1000.0] * 4 + [50.0] * 4) * 50  # swings beyond the noise ceiling
    for jitter in (0.0, 0.03):  # sample times off the half-second grid
        w = _feed(costs.Steady(10, window=4, min_warmup=1, max_warmup=20), noisy, jitter=jitter)
        assert not w.steady and w.seconds >= 10


def test_nnls_recovers_coefficients_and_drops_negative_ones():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, (8, 3))
    b = a @ np.array([2.0, 0.0, 5.0])
    np.testing.assert_allclose(_fit.nnls(a.T, b), [2.0, 0.0, 5.0], atol=1e-9)
    assert (_fit.nnls(a.T, a @ np.array([1.0, -3.0, 1.0])) >= 0).all()


def _sites(**times):
    """Plain profile sites with the given seconds (claim's as self time)."""
    return {"sites": {name: {"calls": 1, "time": t, "self_time": t, "details": {}} for name, t in times.items()}}


def _window(name, accesses, decoded, gathered, admitted, shape, per):
    """A profiled window: `gathered` and `admitted` are row payload bytes."""
    profile = _sites(
        claim=accesses * per["access"] + decoded * per["decode"] + decoded * shape.cold * per["cold"],
        **{"row.admit": admitted * per["admit"], "row.gather": gathered * per["gather"]},
        decode=decoded * shape.decoded * per["decode_element"],
    )
    stats = {"accesses": accesses, "decoded_rows": decoded, "row_hits": accesses - decoded,
             "row_gather_bytes": gathered, "row_admit_bytes": admitted, "profile": profile, "outputs": []}
    return _fit.Measured(name, stats, 1.0, shape, True, True, 0.01)


PER = {"access": 1e-6, "decode": 2e-7, "cold": 1e-6, "decode_element": 2e-9, "gather": 4e-11, "admit": 3e-10}


def test_host_fit_recovers_constants():
    per = PER
    windows = [
        _window("a", 1e6, 1e6, 0, 0, _fit.Shape(800, 0, 0.0), per),
        _window("b", 1e6, 9e5, 4e8, 1e8, _fit.Shape(800, 900, 0.3), per),
        _window("c", 1e6, 6e5, 1e9, 2e8, _fit.Shape(700, 900, 0.1), per),
        # Fitting the three bookkeeping constants takes more windows than unknowns.
        _window("d", 1e6, 8e5, 5e8, 1e8, _fit.Shape(750, 900, 0.6), per),
    ]
    values, report = _fit.fit_host(windows, 0.5, costs._defaults.HOST)
    assert not report["fixed"]
    assert values["access"] == pytest.approx(0.5 * per["access"], rel=1e-6)
    for k in ("decode", "cold", "decode_element"):
        assert values["input"][k] == pytest.approx(0.5 * per[k], rel=1e-6), k
    for k in ("gather", "admit"):
        assert values["rows"][k] == pytest.approx(0.5 * per[k], rel=1e-6), k
    assert max(abs(v) for v in report["bookkeeping_residuals"].values()) < 1e-6


def test_process_bands_fill_gaps_and_merge_by_calls():
    bands = [{"max_elements": b, "full_calls": 0, "full_time": 0.0, "partial_calls": 0, "partial_time": 0.0}
             for b in loader.PROCESS_BOUNDS]
    bands[2].update(full_calls=100, full_time=1e-4)
    bands[5].update(full_calls=100, full_time=4e-4, partial_calls=10, partial_time=5e-5)
    values, weights = _fit.process_values(bands, 5e-4, 200, 1.0)
    curve = values["curve"]
    assert [c["max_elements"] for c in curve] == list(loader.PROCESS_BOUNDS)
    assert curve[0]["process"] == pytest.approx(1e-6) and curve[-1]["process"] == pytest.approx(4e-6)
    assert curve[5]["resolve"] == pytest.approx(1e-6)
    merged, measured = _collect._merge(None, values, weights)
    again, measured = _collect._merge({"values": merged, "measured": measured}, values, weights)
    assert again["curve"][2]["process"] == pytest.approx(1e-6) and measured["runs"] == 2
    assert measured["calls"][2] == pytest.approx(100 * _collect.DECAY + 100)


@pytest.fixture
def loading(tmp_path):
    x = np.arange(48, dtype=np.float32).reshape(12, 4)
    return loader.Loader(loader.DataSource([(cellpack.write(tmp_path / "a.cpk", x), np.arange(4))], 4), workers=2, backend="pread")


def test_resolve_defaults_then_stored_records(loading, cellpack_home):
    _resolve._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = costs.resolve(loading)
    messages = [str(w.message) for w in caught]
    assert any("no measured host costs" in s for s in messages)
    assert any("no measured device costs" in s for s in messages)
    assert m.provenance["host"] == "default" and m.provenance["storage"].startswith("default:")
    assert m.cores == loading.cores and 1 <= m.cores <= 2 and list(m.outputs) == ["X"]
    k = costs.keys(loading)
    store = costs.CostStore()
    assert store.root.is_relative_to(cellpack_home)
    host = dict(costs._defaults.HOST, access=1.5e-6)
    store.put("host", k["host"][0], k["host"][1], host)
    store.put("storage", k["storage"][0], k["storage"][1], {"iops": 1e5, "bandwidth": 1e9, "request": 3e-6, "serial": 0.1})
    m = costs.resolve(loading)
    assert m.access == 1.5e-6 and m.iops == 1e5
    assert m.provenance["host"] == f"store:host/{k['host'][0]}"
    assert loading.plan_options(4096).cost == m


def test_cost_model_file_overrides_everything(loading, tmp_path, monkeypatch):
    path = model().save(tmp_path / "m.json")
    d = json.loads(path.read_text())
    d["outputs"] = d["outputs"][:1]
    path.write_text(json.dumps(d))
    monkeypatch.setenv("CELLPACK_COST_MODEL", str(path))
    m = costs.resolve(loading)
    assert m.iops == 8e5 and m.cores == loading.cores and list(m.outputs) == ["X"]
    assert m.provenance["model"] == f"file:{path}"


def test_partial_cost_records_compose_known_fields_and_defaults(loading, cellpack_home):
    keys = costs.keys(loading)
    store = costs.CostStore()
    values = {"access": 1.5e-6, "action": 2e-6,
              "input": {"decode_element": 2.5e-9, "row_element": 1e-10, "row_copy": 2e-9},
              "builtin": {"*": {"map_element": 3e-9, "unused": 42}}}
    store.put("host", *keys["host"], values)
    m = costs.resolve(loading)
    assert m.access == 1.5e-6 and m.input.decode_element == 2.5e-9
    assert m.input.cold == costs._defaults.HOST["input"]["cold"]
    assert m.rows == RowCost(**costs._defaults.HOST["rows"])
    assert m.compile == costs._defaults.HOST["compile"]
    assert m.outputs["X"].map_element == 3e-9
    assert store.get("host", keys["host"][0])["values"] == values


def test_compilers_read_models_by_output_name(loading):
    from dataclasses import replace

    options = loading.plan_options(1 << 20)
    cells = np.arange(12, dtype=np.uint32)
    assert loading.compile(cells, options).summary.accesses == 12
    # Outputs are found by name, or by position.
    positional = replace(options.cost, outputs={0: options.cost.outputs["X"]})
    assert loading.compile(cells, replace(options, cost=positional)).summary.accesses == 12
    with pytest.raises(ValueError, match="no costs for output"):
        loading.compile(cells, replace(options, cost=replace(options.cost, outputs={"Y": OutputCost()})))


def test_storage_tiers():
    assert costs.storage_tier("tmpfs", None) == "memory"
    assert costs.storage_tier("nfs4", None) == "network"
    assert costs.storage_tier("fuse.juicefs", None) == "network"
    assert costs.storage_tier("ext4", None) == "unknown"


def test_profiled_runs_record_process_bands(tmp_path):
    x = np.arange(48, dtype=np.float32).reshape(12, 4)
    source = loader.DataSource([(cellpack.write(tmp_path / "a.cpk", x), np.arange(4))], 4)
    loading = loader.Loader(source, workers=2, backend="pread", profile=["process", "map"])
    with loading.start(np.tile(np.arange(12, dtype=np.uint32), 4)) as run:
        for _ in run:
            pass
        stats = run.join()
    plain = {"profile": _fit.plain_profile(stats.profile)}
    # The built-in output maps every access; no process function ran.
    assert _fit.detail(plain, "map", 0)["calls"] == 48
    bands, seconds = _fit.process_bands(plain, 0)
    assert seconds == 0 and not any(b["full_calls"] for b in bands)
    assert stats.profile.map_time(0) == pytest.approx(_fit.detail(plain, "map", 0)["time"])


def test_resolve_helper_names_builtin_dtype_rows():
    host = dict(costs._defaults.HOST, builtin={"float32": {"map_element": 1e-9, "fill_element": 0.0}})
    assert _resolve._builtin(host, "float32")["map_element"] == 1e-9
    assert _resolve._builtin(host, "uint8") == costs._defaults.HOST["builtin"]["*"]


def test_counts_equal_in_every_window_keep_their_base_values():
    per = PER
    # Every access decodes: the decode count cannot be told from the access constant.
    windows = [_window(n, 1e6, 1e6, 0, 0, _fit.Shape(800, 0, c), per) for n, c in (("a", 0.0), ("b", 0.3), ("c", 0.6))]
    base = dict(costs._defaults.HOST, input=dict(costs._defaults.HOST["input"], decode=2e-7))
    values, report = _fit.fit_host(windows, 1.0, base)
    assert report["fixed"] == ["decode"] and values["input"]["decode"] == 2e-7
    assert values["access"] == pytest.approx(per["access"], rel=1e-6)
    assert values["input"]["cold"] == pytest.approx(per["cold"], rel=1e-6)
    # Windows without rows keep the rows' base costs.
    assert values["rows"] == base["rows"]


def test_fewer_windows_than_unknowns_fix_the_least_identified_constants():
    per = PER
    windows = [
        _window("a", 1e6, 1e6, 0, 0, _fit.Shape(800, 0, 0.0), per),
        _window("b", 1e6, 1e6, 0, 0, _fit.Shape(800, 0, 0.3), per),
        _window("c", 1e6, 6e5, 0, 0, _fit.Shape(700, 0, 0.1), per),
    ]
    base = dict(costs._defaults.HOST, input=dict(costs._defaults.HOST["input"], decode=2e-7, cold=1e-6))
    values, report = _fit.fit_host(windows, 1.0, base)
    # Three windows fit two constants (access, then cold); decode keeps its base.
    assert report["fixed"] == ["decode"]
    assert values["input"]["decode"] == 2e-7 and values["access"] == pytest.approx(per["access"], rel=1e-6)
    assert values["input"]["cold"] == pytest.approx(per["cold"], rel=1e-6)


def test_steady_tolerates_noisy_storage_but_not_trends():
    rng = np.random.default_rng(2)
    noisy = list(1000.0 * (1 + 0.2 * rng.standard_normal(300)))  # network storage
    w = _feed(costs.Steady(20, window=10, min_warmup=1), noisy)
    assert w.steady
    ramp = [1000.0 * (1 + 0.01 * i) * (1 + 0.2 * rng.standard_normal()) for i in range(300)]
    w = _feed(costs.Steady(20, window=10, min_warmup=1, max_warmup=60), ramp)
    assert not w.steady  # +2%/s: a trend, not noise


def test_device_rates_follow_the_reads_when_workers_block_on_them():
    def window(submit, requests, read_bytes):
        profile = _sites(decode=1.0, map=1.0, **{"io.submit": submit})
        stats = {"accesses": 1e6, "profile": profile, "io": {"requests": requests, "read_bytes": read_bytes}}
        return _fit.Measured("off", stats, 10.0, _fit.Shape(0, 0, 0), True, True, 0.1)

    tier = costs._defaults.storage("network")
    # GPFS completes direct reads inside the submission: device-bound.
    values, device = _fit.fit_storage(window(20.0, 2e6, 8e9), None, tier, 1.0, 16)
    assert device["bound"] == "device" and values["bandwidth"] == pytest.approx(8e8) and values["iops"] == pytest.approx(2e5)
    # Busy workers: the measured rates are lower bounds, raised to the tier's.
    values, device = _fit.fit_storage(window(0.1, 2e6, 8e9), None, tier, 1.0, 16)
    assert device["bound"] == "worker" and values["bandwidth"] == tier["bandwidth"]
