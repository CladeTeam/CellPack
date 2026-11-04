//! `LoaderConfig` as Python passes it (a frozen dataclass in
//! `pycellpack.loader`), read by attribute and converted field by field.
//! Enum strings and profile sites are checked here, so a bad setting fails
//! at the call that supplied it.

use cellpack::loader as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::loader_error;

/// `LoaderConfig` as Python passes it: worker count, backend, inflight
/// bound, claim batch, slot rows, optional window and profile sites.
#[derive(FromPyObject)]
pub(crate) struct LoaderConfigArg {
    workers: usize,
    backend: String,
    inflight: usize,
    claim_batch: usize,
    slot_rows: u64,
    window: Option<u64>,
    profile: Option<Vec<String>>,
    cpus: Vec<usize>,
}

impl LoaderConfigArg {
    /// The native config; rejects unknown backends and profile sites.
    pub(crate) fn native(self) -> PyResult<core::LoaderConfig> {
        Ok(core::LoaderConfig {
            workers: self.workers,
            backend: match self.backend.as_str() {
                "pread" => core::IoBackend::Pread,
                "uring" => core::IoBackend::Uring,
                _ => return Err(PyValueError::new_err("backend must be 'pread' or 'uring'")),
            },
            max_inflight_per_worker: self.inflight,
            claim_batch: self.claim_batch,
            slot_cells: self.slot_rows,
            window: self.window,
            profile: self.profile.map(|names| cellpack::prof::SiteMask::parse(&names)).transpose().map_err(loader_error)?,
            cpus: self.cpus,
        })
    }
}
