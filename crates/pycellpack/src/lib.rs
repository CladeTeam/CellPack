//! Python adapters over the public Rust buffer, storage, scans and loader
//! interfaces (`pycellpack._cellpack`).
//!
//! Native code keeps the caller's data as validated snapshots and
//! caller-configured state; Python supplies buffers and choices, Rust
//! validates and executes them. Error kinds map onto Python exceptions
//! (see [`errors`]), and long native calls release the GIL.
//!
//! The modules follow the core crate's: [`store`] and [`format`] for `.cpk`
//! files, [`scans`] for the codec components, [`loader`] for sources,
//! schemas, accesses, rings and runs, and [`opt`] for plans and costs.
//! Every class registers in the one flat extension module.

mod buffers;
mod errors;
mod format;
mod loader;
mod opt;
mod scans;
mod store;

use pyo3::prelude::*;

/// The private extension behind `pycellpack`: `.cpk` storage, the scans
/// codecs, access sequences, the loader with its rings and runs, plans and
/// cost defaults.
#[pymodule]
fn _cellpack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bind kernels and environment settings at import, so an invalid
    // setting fails here rather than at the first decode.
    cellpack::runtime::get().map_err(errors::error)?;
    register(m)
}

/// Adds every class, function and constant to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kernel_set, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(physical_cores, m)?)?;
    store::register(m)?;
    format::register(m)?;
    scans::register(m)?;
    loader::register(m)?;
    opt::register(m)?;
    Ok(())
}

/// The bound kernel set: `"scalar"`, `"avx2"` or `"avx512"`.
#[pyfunction]
fn kernel_set() -> &'static str {
    cellpack::scans::kernel_set()
}

/// Build capabilities: the bound kernel set, and the profiling sites (name,
/// description) and groups (name, sites) every build carries.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("kernel_set", cellpack::scans::kernel_set())?;
    d.set_item("profile_sites", cellpack::prof::SITES.iter().map(|s| (s.name, s.doc)).collect::<Vec<_>>())?;
    d.set_item("profile_groups", cellpack::prof::GROUPS.iter().map(|(g, m)| (*g, m.to_vec())).collect::<Vec<_>>())?;
    Ok(d)
}

/// Physical cores among `cpus` (or this thread's allowed CPUs).
#[pyfunction]
#[pyo3(signature = (cpus=Vec::new()))]
fn physical_cores(cpus: Vec<usize>) -> usize {
    cellpack::runtime::physical_cores(&cpus)
}
