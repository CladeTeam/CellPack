//! The native loader for Python: one data source, its output schemas and
//! settings. It compiles and replays plans, estimates memory and starts
//! [`Run`]s over a caller's ring.

use std::sync::Arc;

use cellpack::loader as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::config::LoaderConfigArg;
use super::ring::Ring;
use super::run::Run;
use super::schema::Schema;
use super::source::DataSource;
use crate::errors::loader_error;
use crate::opt::args::{CostModelArg, PlanOptionsArg};
use crate::opt::plan::{Plan, online_dict, summary_dict};

/// The native loader: one data source, its output schemas and settings. It
/// compiles plans, estimates memory, and starts runs over a caller's ring.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "Loader")]
pub(crate) struct Loader {
    inner: Arc<core::Loader>,
    source: Py<DataSource>,
    names: Arc<[String]>,
}

impl Loader {
    fn options(&self, options: PlanOptionsArg<'_>) -> PyResult<cellpack::opt::PlanOptions> {
        options.native(&self.names)
    }
}

#[pymethods]
impl Loader {
    /// `schemas` carry their output names; `config` is a `LoaderConfig`.
    #[new]
    fn new(py: Python<'_>, source: Bound<'_, DataSource>, schemas: Vec<Py<Schema>>, config: LoaderConfigArg) -> PyResult<Self> {
        let names = schemas.iter().map(|s| s.get().name().ok_or_else(|| PyValueError::new_err("loader schemas need output names"))).collect::<PyResult<Vec<_>>>()?;
        let schemas = schemas.iter().map(|s| s.get().materialize()).collect::<core::Result<Vec<_>>>().map_err(loader_error)?;
        let config = config.native()?;
        let src = source.get().inner.clone();
        let inner = py.detach(|| core::Loader::new(src, schemas, config)).map_err(loader_error)?;
        Ok(Self { inner: Arc::new(inner), source: source.unbind(), names: names.into() })
    }
    /// The execution window in slots of runs without a plan.
    #[getter]
    fn window(&self) -> u64 {
        self.inner.window()
    }
    /// Accesses a slot holds: `slot_rows` positions of every schema.
    #[getter]
    fn accesses_per_slot(&self) -> u64 {
        self.inner.accesses_per_slot()
    }
    /// Physical cores the workers run on (a cost model's CPU capacity).
    #[getter]
    fn cores(&self) -> f64 {
        self.inner.cores()
    }
    /// Known reservations of a run: with an offline `plan` or online
    /// `options`, theirs; else default page planning of `cache_bytes`.
    #[pyo3(signature = (slots, world, *, plan = None, options = None, cache_bytes = 0))]
    fn memory_estimate<'py>(&self, py: Python<'py>, slots: u32, world: u32, plan: Option<PyRef<'_, Plan>>, options: Option<PlanOptionsArg<'_>>, cache_bytes: usize) -> PyResult<Bound<'py, PyDict>> {
        let planning = planning(plan, options.map(|o| self.options(o)).transpose()?)?;
        let specified = !matches!(planning, core::Planning::None);
        if specified && cache_bytes != 0 {
            return Err(PyValueError::new_err("cache_bytes cannot be combined with a plan or plan options"));
        }
        let s = py
            .detach(|| if specified { self.inner.memory_estimate_for(&planning, slots, world) } else { self.inner.memory_estimate(cache_bytes, slots, world) })
            .map_err(loader_error)?;
        let d = PyDict::new(py);
        macro_rules! fields {($($f:ident),*) => {$(d.set_item(stringify!($f), s.$f)?;)*};}
        fields!(cache_frame_bytes, generation_bytes, private_frame_bytes, worker_scratch_bytes, plan_ring_bytes, metadata_bytes, output_ring_bytes, known_reservation_bytes);
        Ok(d)
    }
    /// Compiles a plan for a stored sequence: offline (the whole sequence is
    /// the horizon) or as an online run would, which also returns the
    /// compilation's statistics.
    #[pyo3(signature = (accesses, options, *, online = false))]
    fn compile<'py>(&self, py: Python<'py>, accesses: &Bound<'py, PyAny>, options: PlanOptionsArg<'_>, online: bool) -> PyResult<(Plan, Option<Bound<'py, PyDict>>)> {
        let cells = self.source.get().stored(py, accesses)?;
        let options = self.options(options)?;
        if online {
            let (plan, stats) = py.detach(|| self.inner.compile_online(&cells, &options)).map_err(loader_error)?;
            Ok((Plan { inner: Arc::new(plan) }, Some(online_dict(py, &stats)?)))
        } else {
            let plan = py.detach(|| self.inner.compile(&cells, &options)).map_err(loader_error)?;
            Ok((Plan { inner: Arc::new(plan) }, None))
        }
    }
    /// The statistics of compiling a sequence online, without loading it or
    /// keeping its plan.
    fn compile_stats<'py>(&self, py: Python<'py>, accesses: &Bound<'py, PyAny>, options: PlanOptionsArg<'_>) -> PyResult<Bound<'py, PyDict>> {
        let seq = self.source.get().translate(py, accesses)?;
        let options = self.options(options)?;
        let stats = py.detach(|| self.inner.plan_online(&*seq, &options)).map_err(loader_error)?;
        online_dict(py, &stats)
    }
    /// Replays a plan serially, priced by its own cost model or `cost`.
    #[pyo3(signature = (plan, accesses, *, cost = None))]
    fn replay<'py>(&self, py: Python<'py>, plan: PyRef<'_, Plan>, accesses: &Bound<'py, PyAny>, cost: Option<CostModelArg<'_>>) -> PyResult<Bound<'py, PyDict>> {
        let plan = plan.inner.clone();
        let cells = self.source.get().stored(py, accesses)?;
        let cost = cost.map(|c| c.native(&self.names)).transpose()?;
        let summary = py
            .detach(|| match &cost {
                Some(cost) => self.inner.replay_with_cost(&plan, &cells, cost),
                None => self.inner.replay(&plan, &cells),
            })
            .map_err(loader_error)?;
        summary_dict(py, &summary)
    }
    /// Starts loading `accesses` into `ring`, consumed as rank `rank`, with
    /// an offline `plan` or online `options` (or neither: no cache).
    #[pyo3(signature = (accesses, ring, *, rank = 0, plan = None, options = None))]
    fn start(&self, py: Python<'_>, accesses: &Bound<'_, PyAny>, ring: Bound<'_, Ring>, rank: u32, plan: Option<PyRef<'_, Plan>>, options: Option<PlanOptionsArg<'_>>) -> PyResult<Run> {
        let world = ring.get().layout.get().world;
        if rank >= world {
            return Err(PyValueError::new_err("rank outside the ring's world"));
        }
        let access = self.source.get().translate(py, accesses)?;
        let planning = planning(plan, options.map(|o| self.options(o)).transpose()?)?;
        Run::start(py, self.inner.clone(), access, planning, ring, rank, self.names.clone())
    }
}

fn planning(plan: Option<PyRef<'_, Plan>>, options: Option<cellpack::opt::PlanOptions>) -> PyResult<core::Planning> {
    match (plan, options) {
        (Some(_), Some(_)) => Err(PyValueError::new_err("a run follows an offline plan or online options, not both")),
        (Some(p), None) => Ok(core::Planning::Fixed(p.inner.clone())),
        (None, Some(o)) => Ok(core::Planning::Online(Box::new(o))),
        (None, None) => Ok(core::Planning::None),
    }
}
