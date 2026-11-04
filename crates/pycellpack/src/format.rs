//! `.cpk` byte layouts for Python, as in `cellpack::format`: the superblock,
//! the chunk directory and record element counts.

use cellpack::format;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::buffers::{bytes, dtype_name};
use crate::errors::error;

/// A directory entry: first row, offset, length, metadata length, rows.
type DirectoryEntry = (u64, u64, u64, u64, u32);

/// The stored elements of a record.
#[pyfunction]
fn scans_record_nnz(record: &Bound<'_, PyAny>) -> PyResult<usize> {
    let r = bytes(record)?;
    // SAFETY: validated immutable bytes retained during parsing.
    format::record_nnz(unsafe { r.slice() }).map_err(error)
}

/// A `.cpk` superblock's fields.
#[pyfunction]
fn scans_parse_superblock<'py>(
    py: Python<'py>,
    bytes_: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let b = bytes(bytes_)?;
    // SAFETY: validated immutable header bytes.
    let s = format::cpk::parse_superblock(unsafe { b.slice() }).map_err(error)?;
    let d = PyDict::new(py);
    macro_rules! fields{($($f:ident),*)=>{$(d.set_item(stringify!($f),s.$f)?;)*};}
    fields!(
        page_size,
        n_rows,
        n_cols,
        n_chunks,
        chunk_rows,
        directory_offset,
        file_len
    );
    d.set_item("dtype", dtype_name(s.dtype))?;
    d.set_item(
        "format",
        if s.source == format::SourceLayout::Dense {
            "dense"
        } else {
            "csr"
        },
    )?;
    Ok(d)
}
/// The chunk directory a superblock describes.
#[pyfunction]
fn scans_parse_directory(
    header: &Bound<'_, PyAny>,
    directory: &Bound<'_, PyAny>,
) -> PyResult<Vec<DirectoryEntry>> {
    let (h, d) = (bytes(header)?, bytes(directory)?);
    // SAFETY: retained immutable byte inputs.
    unsafe {
        let h = h.slice();
        let s = format::cpk::parse_superblock(h).map_err(error)?;
        Ok(format::cpk::parse_directory(h, &s, d.slice())
            .map_err(error)?
            .iter()
            .map(|e| (e.first_row, e.offset, e.len, e.meta_len, e.n_rows))
            .collect())
    }
}

/// Adds the format constants and parsers to the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SUPERBLOCK_LEN", format::cpk::SUPERBLOCK_LEN)?;
    m.add("ENTRY_LEN", format::cpk::ENTRY_LEN)?;
    m.add_function(wrap_pyfunction!(scans_record_nnz, m)?)?;
    m.add_function(wrap_pyfunction!(scans_parse_superblock, m)?)?;
    m.add_function(wrap_pyfunction!(scans_parse_directory, m)?)?;
    Ok(())
}
