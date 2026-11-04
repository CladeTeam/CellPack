//! The loader for Python, module by module as in `cellpack::loader`: the
//! data source ([`source`]), output schemas ([`schema`]), access sequences
//! ([`accesses`]), process functions ([`process`]), output rings and their
//! consumers ([`ring`]), the loader ([`loader`]) and its runs ([`run`]),
//! the settings Python passes ([`config`]) and run statistics ([`stats`]).

pub(crate) mod accesses;
mod config;
#[allow(clippy::module_inception)]
mod loader;
mod process;
mod ring;
mod run;
mod schema;
pub(crate) mod source;
mod stats;

use pyo3::prelude::*;

/// Adds the loader's classes, functions and constants to the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PROCESS_ABI_VERSION", cellpack::loader::PROCESS_ABI_VERSION)?;
    m.add("PROCESS_BOUNDS", cellpack::loader::PROCESS_BOUNDS.to_vec())?;
    m.add_class::<source::DataSource>()?;
    m.add_class::<process::ProcessFn>()?;
    schema::register(m)?;
    accesses::register(m)?;
    ring::register(m)?;
    m.add_class::<loader::Loader>()?;
    m.add_class::<run::Run>()?;
    stats::register(m)?;
    Ok(())
}
