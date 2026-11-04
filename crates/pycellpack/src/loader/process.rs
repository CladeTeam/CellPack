//! Versioned native process-function capsules; functions never call the
//! Python runtime.

use cellpack::loader as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::c_void;
use std::sync::Arc;

/// The leading fields every process capsule shares: the ABI version it was
/// built for and the size of its vtable.
#[repr(C)]
struct Header {
    abi_version: u32,
    struct_size: u32,
}

/// The v1 process-function vtable: the optional CSR and dense entry points,
/// both taking the provider's context.
#[repr(C)]
struct VTable {
    abi_version: u32,
    struct_size: u32,
    csr: Option<core::CsrFn>,
    dense: Option<core::DenseFn>,
    user_data: *const c_void,
}

/// A validated v1 process function. The capsule and optional owner stay
/// retained, so the entry points and context outlive every run that uses
/// them.
#[pyclass(module = "pycellpack._cellpack", name = "ProcessFn", frozen)]
pub(crate) struct ProcessFn {
    pub(crate) inner: core::ProcessFn,
    pub(crate) identity: u64,
    /// Entry addresses of the CSR and dense functions.
    entries: (usize, usize),
}
#[pymethods]
impl ProcessFn {
    /// Validates the `pycellpack.process.v1` capsule and selects a row format.
    ///
    /// `identity` versions the functions for plan identities and cost records;
    /// `partial_rows` declares the CSR entry point able to emit partial rows
    /// through the resolver; `owner` keeps provider state alive.
    #[new]
    #[pyo3(signature=(capsule,identity,owner=None,partial_rows=false,row_format="auto"))]
    fn new(capsule: &Bound<'_, PyCapsule>, identity: u64, owner: Option<Py<PyAny>>, partial_rows: bool, row_format: &str) -> PyResult<Self> {
        let format = match row_format {
            "auto" => core::RowFormat::Auto,
            "csr" => core::RowFormat::Csr,
            "dense" => core::RowFormat::Dense,
            _ => return Err(PyValueError::new_err("row_format must be 'auto', 'csr' or 'dense'")),
        };
        let pointer = capsule.pointer_checked(Some(c"pycellpack.process.v1"))?;
        if !(pointer.as_ptr() as usize).is_multiple_of(std::mem::align_of::<VTable>()) {
            return Err(PyValueError::new_err("misaligned process capsule"));
        }
        // SAFETY: the named capsule protocol requires a valid VTable retained
        // by the provider until both capsule and owner have been released.
        let header = unsafe { &*pointer.as_ptr().cast::<Header>() };
        if header.abi_version != core::PROCESS_ABI_VERSION || header.struct_size < std::mem::size_of::<VTable>() as u32 {
            return Err(PyValueError::new_err("process ABI or vtable size mismatch"));
        }
        // SAFETY: the provider's validated header declares the full vtable size.
        let table = unsafe { &*pointer.as_ptr().cast::<VTable>() };
        let retained: Arc<dyn Send + Sync> = Arc::new((capsule.clone().unbind(), owner));
        // SAFETY: providers promise thread-safe native functions, stable
        // context, writes only within their cell and scratch, and no
        // unwinding. The owner pins all code and data.
        let mut inner = unsafe { core::ProcessFn::with_format(format, table.csr, table.dense, table.user_data, retained) }
            .map_err(|_| PyValueError::new_err(format!("row_format {row_format:?} needs {}", match format {
                core::RowFormat::Auto => "both the CSR and the dense function",
                core::RowFormat::Csr => "the CSR function",
                core::RowFormat::Dense => "the dense function",
            })))?;
        if partial_rows {
            // SAFETY: the provider declares resolver-capable CSR functions.
            inner = unsafe { inner.with_partial_rows() };
        }
        let entries = inner.addresses();
        Ok(Self { inner, identity, entries })
    }

    /// Entry addresses of the CSR and dense functions (for identifying their
    /// code); 0 for one the row format does not use.
    #[getter]
    fn addresses(&self) -> (usize, usize) {
        self.entries
    }
}
