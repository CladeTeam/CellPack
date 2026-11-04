//! Cache plans and their costs for Python: compiled plans and page traces
//! ([`plan`]), the option and cost-model values Python passes ([`args`]),
//! and the crate's default costs ([`defaults`]).

pub(crate) mod args;
mod defaults;
pub(crate) mod plan;

use pyo3::prelude::*;

/// Adds the plan classes and cost functions to the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<plan::Plan>()?;
    m.add_class::<plan::Trace>()?;
    m.add_function(wrap_pyfunction!(plan::access_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(defaults::cost_defaults, m)?)?;
    Ok(())
}
