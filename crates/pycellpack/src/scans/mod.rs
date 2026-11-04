//! NumPy-buffer adapters for the scans components, as in `cellpack::scans`:
//! chunk metadata and tables ([`tables`]), decoded parts and mapping
//! ([`parts`]), encoding ([`encode`]) and prepared decoding into a panel
//! ([`decoder`]). Shared here: row selections and their split over threads.

mod decoder;
pub(crate) mod encode;
mod parts;
mod tables;

use cellpack::format::{self, Dtype};
use cellpack::scans as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::buffers::{Buffer, disjoint};
use crate::errors::error;

/// Rows of a chunk as Python passes them: None (all), a `(start, end)`
/// range or a uint32 vector of cells.
pub(super) enum Selection {
    All,
    Range { start: usize, end: usize },
    List(Buffer),
}
impl Selection {
    pub(super) fn get(rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        if rows.is_none() {
            Ok(Self::All)
        } else if rows.is_instance_of::<pyo3::types::PyTuple>() {
            let (start, end) = rows.extract::<(usize, usize)>()?;
            Ok(Self::Range { start, end })
        } else {
            let indices = Buffer::get(rows, false)?;
            indices.expect(Dtype::U32, &[indices.len / 4])?;
            Ok(Self::List(indices))
        }
    }
    pub(super) fn rows(&self) -> core::Rows<'_> {
        match self {
            Self::All => core::Rows::All,
            Self::Range { start, end } => core::Rows::Range { start: *start, end: *end },
            Self::List(indices) => {
                // SAFETY: validated u32 storage is retained by this selection;
                // callers must not mutate borrowed inputs during native calls.
                core::Rows::List(unsafe { indices.slice::<u32>() })
            }
        }
    }
    pub(super) fn check_outputs(&self, outputs: &[&Buffer]) -> PyResult<()> {
        if let Self::List(indices) = self {
            for out in outputs {
                disjoint(&[indices, out])?;
            }
        }
        Ok(())
    }
    /// Consecutive pieces of the selection for up to `workers` threads, each
    /// with at least [`MIN_PIECE_ROWS`] rows; one piece when not worth it.
    pub(super) fn pieces(&self, n_cells: usize, workers: usize) -> PyResult<Vec<core::Rows<'_>>> {
        let rows = self.rows();
        let count = rows.count(n_cells).map_err(error)?;
        let n = workers.min(count / MIN_PIECE_ROWS).max(1);
        if n == 1 {
            return Ok(vec![rows]);
        }
        let step = count.div_ceil(n);
        Ok((0..count)
            .step_by(step)
            .map(|lo| {
                let hi = (lo + step).min(count);
                match rows {
                    core::Rows::List(list) => core::Rows::List(&list[lo..hi]),
                    _ => core::Rows::Range { start: rows.get(lo), end: rows.get(lo) + (hi - lo) },
                }
            })
            .collect())
    }
}

/// Rows below which a decode is not split over threads.
const MIN_PIECE_ROWS: usize = 256;

/// Runs one task per piece on scoped threads (inline for a single piece).
pub(super) fn scoped<R: Send>(tasks: Vec<impl FnOnce() -> R + Send>) -> Vec<R> {
    if tasks.len() == 1 {
        return tasks.into_iter().map(|t| t()).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = tasks.into_iter().map(|t| scope.spawn(t)).collect();
        handles.into_iter().map(|h| h.join().expect("decode worker panicked")).collect()
    })
}

/// Parsed chunk metadata owned by the Python objects that share it.
pub(super) type Metadata = format::OwnedMeta;

/// The id space `coded` selects.
pub(super) fn space(coded: bool) -> core::IdSpace {
    if coded {
        core::IdSpace::Coded
    } else {
        core::IdSpace::Source
    }
}
/// Whether dense outputs are cleared first.
pub(super) fn fill(zero: bool) -> core::Fill {
    if zero {
        core::Fill::Zero
    } else {
        core::Fill::AssumeZeroed
    }
}
/// A writable uint64 workspace vector.
pub(super) fn workspace_buffer(obj: &Bound<'_, PyAny>) -> PyResult<Buffer> {
    let b = Buffer::get(obj, true)?;
    if b.dtype != Dtype::U64 || b.shape.len() != 1 {
        return Err(PyValueError::new_err(
            "workspace requires a writable uint64 vector",
        ));
    }
    Ok(b)
}

/// Kernel sets this CPU supports.
#[pyfunction]
fn scans_kernel_sets() -> Vec<&'static str> {
    core::supported_kernel_sets()
}

/// Adds the scans classes and functions to the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    tables::register(m)?;
    parts::register(m)?;
    encode::register(m)?;
    decoder::register(m)?;
    m.add_function(wrap_pyfunction!(scans_kernel_sets, m)?)
}
