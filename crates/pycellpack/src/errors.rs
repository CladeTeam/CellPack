//! How core errors become Python exceptions: `OSError` for I/O, `ValueError`
//! for rejected arguments, configurations, plans and data, `RuntimeError`
//! for a run that failed.

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Maps a core storage/codec error to a Python exception.
pub(crate) fn error(e: cellpack::Error) -> PyErr {
    match e {
        cellpack::Error::Io { .. } => PyOSError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// Maps a loader error to a Python exception.
pub(crate) fn loader_error(e: cellpack::loader::Error) -> PyErr {
    loader_exception(&e).error(e.to_string())
}

/// The Python exception class a loader error becomes, kept separate from the
/// message so a run can publish a failure without creating a traceback.
#[derive(Clone, Copy)]
pub(crate) enum Exception {
    Io,
    Value,
    Runtime,
}

impl Exception {
    /// Builds the Python exception with `message`.
    pub(crate) fn error(self, message: String) -> PyErr {
        match self {
            Self::Io => PyOSError::new_err(message),
            Self::Value => PyValueError::new_err(message),
            Self::Runtime => PyRuntimeError::new_err(message),
        }
    }
}

/// I/O failures are `OSError`; configuration, access, plan and non-I/O
/// format errors are `ValueError`; everything else (a process or access
/// function returning nonzero, a panicking worker, a busy loader, an aborted
/// run) is a `RuntimeError`.
pub(crate) fn loader_exception(e: &cellpack::loader::Error) -> Exception {
    use cellpack::loader::Error;
    match e {
        Error::Io(_) | Error::CellPack(cellpack::Error::Io { .. }) => Exception::Io,
        Error::Config(_) | Error::Access { .. } | Error::Plan(_) | Error::CellPack(_) => Exception::Value,
        _ => Exception::Runtime,
    }
}
