//! Cache plans for Python: compiled plans (files, bytes, pickles), their
//! serial summaries and online compilation statistics, and the serial page
//! optimum plans are measured against.

use std::path::PathBuf;
use std::sync::Arc;

use cellpack::opt::{OnlineStats, Plan as CorePlan, PlanSummary};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::errors::loader_error;
use crate::loader::accesses::snapshot;
use crate::loader::source::DataSource;

/// A compiled cache plan: the residencies a run follows, its reuse gap,
/// frames and the summary of its serial replay. Plans are immutable and
/// shareable between runs and processes.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "Plan")]
pub(crate) struct Plan {
    pub inner: Arc<CorePlan>,
}

#[pymethods]
impl Plan {
    /// Reads a plan file at `path`.
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Ok(Self { inner: Arc::new(py.detach(|| CorePlan::open(path)).map_err(loader_error)?) })
    }
    /// Writes the plan file to `path`.
    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.inner.write(path)).map_err(loader_error)
    }
    /// The plan file's bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = py.detach(|| self.inner.to_bytes());
        PyBytes::new(py, &bytes)
    }
    /// Reads a plan from the bytes of a plan file.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        let data = data.to_vec();
        Ok(Self { inner: Arc::new(py.detach(|| CorePlan::from_bytes(&data)).map_err(loader_error)?) })
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((slf.get_type().getattr("from_bytes")?, (slf.get().to_bytes(py),).into_pyobject(py)?))
    }
    /// The serial replay's counts (all zero for a plan read from bytes until
    /// a loader verifies it).
    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        summary_dict(py, self.inner.summary())
    }
    /// The reuse gap in accesses.
    #[getter]
    fn distance(&self) -> u64 {
        self.inner.distance()
    }
    /// The number of output schemas the plan was compiled for.
    #[getter]
    fn schemas(&self) -> u64 {
        self.inner.schemas()
    }
    /// The number of positions a slot holds.
    #[getter]
    fn slot_rows(&self) -> u64 {
        self.inner.slot_cells()
    }
    /// The reuse gap in slots.
    #[getter]
    fn reuse_gap(&self) -> u64 {
        self.inner.reuse_gap()
    }
    /// The cache budget in bytes.
    #[getter]
    fn cache_bytes(&self) -> usize {
        self.inner.cache_bytes()
    }
    /// The number of cache frames the budget buys.
    #[getter]
    fn cache_frames(&self) -> u32 {
        self.inner.cache_frames()
    }
    /// Accesses in the sequence the plan is bound to.
    fn __len__(&self) -> usize {
        self.inner.len() as usize
    }
    /// The residencies in admission order, as arrays: `kind` (0 page, 1
    /// row), `key` (global page id, or cell), `start`, `end`, and each
    /// residency's frames as `frames[offsets[i]:offsets[i + 1]]`.
    fn residencies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use cellpack::opt::Object;
        let plan = self.inner.clone();
        let (kind, key, start, end, offsets, frames) = py.detach(|| {
            let rs = plan.residencies();
            let (mut kind, mut key, mut start, mut end, mut offsets, mut frames) = (Vec::with_capacity(rs.len()), Vec::with_capacity(rs.len()), Vec::with_capacity(rs.len()), Vec::with_capacity(rs.len()), Vec::with_capacity(rs.len() + 1), Vec::new());
            offsets.push(0u64);
            for r in rs {
                let (k, v) = match r.object {
                    Object::Page(p) => (0u8, p),
                    Object::Row(c) => (1u8, c as u64),
                };
                kind.push(k);
                key.push(v);
                start.push(r.start);
                end.push(r.end);
                frames.extend_from_slice(plan.frames_of(r));
                offsets.push(frames.len() as u64);
            }
            (kind, key, start, end, offsets, frames)
        });
        let d = PyDict::new(py);
        d.set_item("kind", PyArray1::from_vec(py, kind))?;
        d.set_item("key", PyArray1::from_vec(py, key))?;
        d.set_item("start", PyArray1::from_vec(py, start))?;
        d.set_item("end", PyArray1::from_vec(py, end))?;
        d.set_item("offsets", PyArray1::from_vec(py, offsets))?;
        d.set_item("frames", PyArray1::from_vec(py, frames))?;
        Ok(d)
    }
}

/// The serial replay's counts as a dict, with the quarter series separate.
pub(crate) fn summary_dict<'py>(py: Python<'py>, s: &PlanSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    macro_rules! fields{($($f:ident),*)=>{$(d.set_item(stringify!($f), &s.$f)?;)*};}
    fields!(
        accesses,
        page_uses,
        page_hits,
        requests,
        read_bytes,
        page_residencies,
        row_residencies,
        block_reads,
        decoded_rows,
        decoded_elements,
        cold_rows,
        row_hits,
        row_elements,
        row_bytes,
        admitted_elements,
        admitted_bytes,
        mapped_elements,
        filled_elements,
        converted_elements,
        process_calls,
        partial_calls,
        partial_elements,
        peak_bytes,
        concurrent_peak_bytes,
        worker_seconds
    );
    d.set_item("quarter_uses", s.quarter_uses.to_vec())?;
    d.set_item("quarter_hits", s.quarter_hits.to_vec())?;
    d.set_item("quarter_requests", s.quarter_requests.to_vec())?;
    d.set_item("quarter_seconds", s.quarter_seconds.to_vec())?;
    d.set_item("quarter_service_seconds", s.quarter_service_seconds.iter().map(|q| q.to_vec()).collect::<Vec<_>>())?;
    Ok(d)
}

/// Online compilation statistics as a dict, including the prices the
/// compiler learned.
pub(crate) fn online_dict<'py>(py: Python<'py>, s: &OnlineStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    macro_rules! fields{($($f:ident),*)=>{$(d.set_item(stringify!($f), s.$f)?;)*};}
    fields!(accesses, chunks, chunk, horizon, distance, shards, page_uses, page_hits, page_admissions, row_admissions, row_hits, block_reads, requests, row_candidates, row_kept, page_refused, row_price, modeled_seconds, worker_seconds, compile_seconds, waiting_seconds);
    let p = &s.prices;
    let prices = PyDict::new(py);
    prices.set_item("request", p.request)?;
    prices.set_item("byte", p.byte)?;
    prices.set_item("core", p.core)?;
    prices.set_item("memory", p.memory)?;
    d.set_item("prices", prices)?;
    Ok(d)
}

/// The fingerprint a plan binds its stored access sequence by.
#[pyfunction]
pub(crate) fn access_fingerprint(py: Python<'_>, accesses: &Bound<'_, PyAny>) -> PyResult<u64> {
    let cells = snapshot(py, accesses)?;
    Ok(py.detach(|| cellpack::opt::access_fingerprint(&cells)))
}

/// The page sequence a set of accesses reads, and the serial Belady optimum
/// it is measured against. Building a trace reads only metadata.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "PageTrace")]
pub(crate) struct Trace {
    inner: cellpack::opt::PageTrace,
}
#[pymethods]
impl Trace {
    /// A trace of page ids below `universe`.
    #[new]
    fn new(py: Python<'_>, pages: &Bound<'_, PyAny>, universe: usize) -> PyResult<Self> {
        let pages = snapshot(py, pages)?;
        let inner = py.detach(|| cellpack::opt::PageTrace::from_pages(&pages, universe)).map_err(loader_error)?;
        Ok(Self { inner })
    }
    /// The pages a sequence of a source's cells reads, in order.
    #[staticmethod]
    fn from_source(py: Python<'_>, source: &Bound<'_, DataSource>, accesses: &Bound<'_, PyAny>) -> PyResult<Self> {
        let cells = source.get().stored(py, accesses)?;
        let src = source.get().inner.clone();
        let inner = py.detach(|| cellpack::opt::PageTrace::from_source(&src, &cells)).map_err(loader_error)?;
        Ok(Self { inner })
    }
    /// The number of page reads in the traced sequence.
    #[getter]
    fn page_uses(&self) -> usize {
        self.inner.page_uses()
    }
    /// The number of distinct pages the sequence touches.
    #[getter]
    fn distinct_pages(&self) -> usize {
        self.inner.distinct_pages()
    }
    /// The serial optimum's counts with `capacity_pages` pages of cache.
    fn optimum<'py>(&self, py: Python<'py>, capacity_pages: usize) -> PyResult<Bound<'py, PyDict>> {
        let s = py.detach(|| self.inner.optimum(capacity_pages));
        let d = PyDict::new(py);
        macro_rules! f{($($name:ident),*)=>{$(d.set_item(stringify!($name), s.$name)?;)*};}
        f!(page_uses, distinct_pages, hits, misses, capacity_pages, row_hits, output_hits);
        Ok(d)
    }
}
