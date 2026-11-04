//! Output schemas: output dims as standard ids, an element type, a default,
//! and the built-in dense row or a process function.

use super::process::ProcessFn;
use crate::buffers::{Buffer, dtype_from_name, dtype_name};
use crate::errors::loader_error;
use cellpack::format::Dtype;
use cellpack::loader as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// The binding-side copy of one output schema. It validates the pieces the
/// core needs (uint32 output ids, native dtype, row width) and materializes
/// a native schema when a loader is built.
#[pyclass(module = "pycellpack._cellpack", name = "OutputSchema", frozen)]
pub(crate) struct Schema {
    id: u64,
    name: Option<String>,
    out_ids: Arc<[u32]>,
    dtype: Dtype,
    /// The default's exact bits in the output dtype.
    default: Vec<u8>,
    row_bytes: usize,
    process: Option<(core::ProcessFn, u64)>,
    scratch_bytes: usize,
}
impl Schema {
    /// The native schema this describes (its id and name are the caller's).
    pub(crate) fn materialize(&self) -> core::Result<core::OutputSchema> {
        let s = match &self.process {
            None => core::OutputSchema::dense(self.id, self.out_ids.clone(), self.dtype)?,
            Some((p, identity)) => core::OutputSchema::process(self.id, self.out_ids.clone(), self.dtype, self.row_bytes, p.clone(), *identity, self.scratch_bytes)?,
        };
        let s = s.with_default_bytes(&self.default)?;
        Ok(match &self.name {
            Some(n) => s.named(n),
            None => s,
        })
    }
    /// The schema's name, if set.
    pub(crate) fn name(&self) -> Option<String> {
        self.name.clone()
    }
}
#[pymethods]
impl Schema {
    /// `default`: the default's bytes in `dtype`; `row_bytes`: a process
    /// function's output bytes per row (the built-in row's are implied).
    #[new]
    #[pyo3(signature=(id, out_ids, dtype, default, *, row_bytes=None, process=None, scratch_bytes=0, name=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(id: u64, out_ids: &Bound<'_, PyAny>, dtype: &str, default: Vec<u8>, row_bytes: Option<usize>, process: Option<PyRef<'_, ProcessFn>>, scratch_bytes: usize, name: Option<String>) -> PyResult<Self> {
        let ids = Buffer::get(out_ids, false)?;
        if ids.dtype != Dtype::U32 || ids.shape.len() != 1 {
            return Err(PyValueError::new_err("output dims require a 1D uint32 array of standard ids"));
        }
        // SAFETY: validated uint32 vector, copied here.
        let out_ids: Arc<[u32]> = unsafe { ids.slice::<u32>() }.into();
        let dtype = dtype_from_name(dtype)?;
        let row_bytes = match (&process, row_bytes) {
            (None, None) => out_ids.len() * dtype.width(),
            (None, Some(n)) if n == out_ids.len() * dtype.width() => n,
            (None, Some(_)) => return Err(PyValueError::new_err("the built-in output is a dense row of the schema's dtype")),
            (Some(_), Some(n)) => n,
            (Some(_), None) => return Err(PyValueError::new_err("a process function's schema needs row_bytes")),
        };
        let s = Self { id, name, out_ids, dtype, default, row_bytes, process: process.map(|p| (p.inner.clone(), p.identity)), scratch_bytes };
        s.materialize().map_err(loader_error)?;
        Ok(s)
    }
    /// The schema's id: the loader's output id, or 0 before a loader binds it.
    #[getter]
    fn id(&self) -> u64 {
        self.id
    }
    /// The output dtype name.
    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.dtype)
    }
    /// Bytes one output row occupies in a ring slot.
    #[getter]
    fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    /// The number of output dims.
    #[getter]
    fn out_dim(&self) -> usize {
        self.out_ids.len()
    }
}

/// Adds the schema class and constants to the module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Schema>()?;
    m.add_function(wrap_pyfunction!(allowed, m)?)?;
    m.add("MAX_SCHEMAS", core::MAX_SCHEMAS)?;
    Ok(())
}

/// Whether the loader can convert stored `source` values to `target` while
/// filling rows.
#[pyfunction]
fn allowed(source: &str, target: &str) -> PyResult<bool> {
    Ok(core::allowed(dtype_from_name(source)?, dtype_from_name(target)?))
}
