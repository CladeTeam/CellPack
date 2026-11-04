//! Run statistics for Python: native `RunStats` (counters, I/O, per-output
//! counts, the profile and online compilation) as dicts, wrapped by the
//! factory `pycellpack.loader` registers into its typed `RunStats`.

use cellpack::loader as core;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use crate::opt::plan::online_dict;

/// Converts native run statistics (set by `pycellpack.loader` to its
/// `RunStats`); plain dicts until it is.
static STATS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Registers the factory that wraps the statistics dicts `Run.snapshot` and
/// `Run.join` return; the first registration wins, and until one happens they
/// are plain dicts.
#[pyfunction]
fn register_run_stats(py: Python<'_>, factory: Py<PyAny>) {
    let _ = STATS.set(py, factory);
}

/// A run's statistics as the registered factory's object (a plain dict
/// until one is registered).
pub(super) fn run_stats<'py>(py: Python<'py>, s: &core::RunStats, names: &[String]) -> PyResult<Bound<'py, PyAny>> {
    let d = stats_dict(py, s, names)?;
    match STATS.get(py) {
        Some(f) => f.bind(py).call1((d,)),
        None => Ok(d.into_any()),
    }
}

/// A profile report: per recorded site, its calls and times in seconds,
/// details by key and histogram buckets (lower bound in seconds, count).
fn profile_dict<'py>(py: Python<'py>, p: &cellpack::prof::ProfileReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("schemas", p.schemas)?;
    d.set_item("bands", core::PROCESS_BOUNDS.to_vec())?;
    let sites = PyDict::new(py);
    for s in &p.sites {
        let e = PyDict::new(py);
        e.set_item("calls", s.calls)?;
        e.set_item("time", s.time.as_secs_f64())?;
        e.set_item("self_time", s.self_time.as_secs_f64())?;
        e.set_item("details", s.details.iter().map(|x| (x.key, x.calls, x.time.as_secs_f64())).collect::<Vec<_>>())?;
        e.set_item("hist", s.hist.as_ref().map(|h| h.iter().map(|&(lo, n)| (lo as f64 * 1e-9, n)).collect::<Vec<_>>()))?;
        sites.set_item(s.name, e)?;
    }
    d.set_item("sites", sites)?;
    Ok(d)
}

/// Native run statistics as a dict, with counters, I/O, per-output counts,
/// profile and online-compilation statistics.
fn stats_dict<'py>(py: Python<'py>, s: &core::RunStats, names: &[String]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("elapsed", s.elapsed.as_secs_f64())?;
    d.set_item("setup_elapsed", s.setup_elapsed.as_secs_f64())?;
    d.set_item("peak_span", s.peak_span)?;
    d.set_item("cache_bytes", s.cache_bytes)?;
    d.set_item("private_bytes", s.private_bytes)?;
    for (name, v) in core::Counters::NAMES.iter().zip(s.counters.words()) {
        d.set_item(*name, v)?;
    }
    let io = PyDict::new(py);
    io.set_item("requests", s.io.requests)?;
    io.set_item("requested_bytes", s.io.requested_bytes)?;
    io.set_item("read_bytes", s.io.read_bytes)?;
    io.set_item("sizes", s.io.sizes.to_vec())?;
    d.set_item("io", io)?;
    let outputs = PyDict::new(py);
    for (o, name) in s.outputs.iter().zip(names) {
        let q = PyDict::new(py);
        q.set_item("accesses", o.accesses)?;
        q.set_item("decoded_rows", o.decoded_rows)?;
        q.set_item("row_hits", o.row_hits)?;
        q.set_item("process_calls", o.process_calls)?;
        outputs.set_item(name, q)?;
    }
    d.set_item("outputs", outputs)?;
    d.set_item("profile", s.profile.as_ref().map(|p| profile_dict(py, p)).transpose()?)?;
    d.set_item("online", s.online.as_ref().map(|o| online_dict(py, o)).transpose()?)?;
    Ok(d)
}

/// Adds the statistics factory registration to the module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_run_stats, m)?)
}
