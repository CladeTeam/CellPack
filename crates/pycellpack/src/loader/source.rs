//! The data source for Python: every input matrix with its column map, and
//! optionally the cells each matrix contributes. Global cell ids number the
//! contributed cells; a run reads them through a lazy remap, while offline
//! plans and traces take a stored sequence of the translated cells.

use std::path::PathBuf;
use std::sync::Arc;

use cellpack::format::Dtype;
use cellpack::loader as core;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use super::accesses::{Remapped, sequence};
use crate::buffers::Buffer;
use crate::errors::loader_error;

/// A registered set of matrices, numbered by global cell id. Python keeps
/// the original `(source, ids)` pairs so the source can be rebuilt and
/// pickled; the native source stays immutable.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "DataSource")]
pub(crate) struct DataSource {
    pub inner: Arc<core::DataSource>,
    /// Global cell id -> stored cell id, when some matrix contributes only
    /// selected cells; each matrix's contributed count.
    cells: Option<(Arc<[u32]>, Vec<u64>)>,
}

impl DataSource {
    /// The stored-cell sequence of a sequence of global cell ids.
    pub(crate) fn translate(&self, py: Python<'_>, accesses: &Bound<'_, PyAny>) -> PyResult<Arc<dyn core::Accesses>> {
        let seq = sequence(py, accesses)?;
        Ok(match &self.cells {
            None => seq,
            Some((table, _)) => Arc::new(Remapped { inner: seq, table: table.clone() }),
        })
    }
    /// A stored-cell sequence held in memory (offline plans bind to it).
    pub(crate) fn stored(&self, py: Python<'_>, accesses: &Bound<'_, PyAny>) -> PyResult<Arc<[u32]>> {
        let seq = self.translate(py, accesses)?;
        py.detach(|| {
            if let Some(s) = seq.as_slice() {
                return Ok(Arc::from(s));
            }
            let n = usize::try_from(seq.len()).map_err(|_| core::Error::Config("sequence too long to store".into()))?;
            let mut v = vec![0u32; n];
            seq.fill(0, &mut v)?;
            Ok(v.into())
        })
        .map_err(loader_error)
    }
    fn stored_cell(&self, cell: u64) -> PyResult<u32> {
        match &self.cells {
            None if cell < self.inner.n_cells() => Ok(cell as u32),
            Some((table, _)) if (cell as usize) < table.len() => Ok(table[cell as usize]),
            _ => Err(PyIndexError::new_err("cell outside the data source")),
        }
    }
}

#[pymethods]
impl DataSource {
    /// `matrices`: `(path, offset, length, ids)` per matrix, `ids` a uint32
    /// vector naming each stored column's standard id (`0xFFFFFFFF` drops
    /// it); `cells`: per matrix, None for all its cells or a uint64 vector
    /// of the stored cells it contributes (omit when every matrix gives all).
    #[new]
    #[pyo3(signature = (matrices, vocab, *, page_size = 4096, direct_io = false, cells = None))]
    fn new(py: Python<'_>, matrices: Vec<(PathBuf, u64, Option<u64>, Bound<'_, PyAny>)>, vocab: usize, page_size: usize, direct_io: bool, cells: Option<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Self> {
        let maps = matrices
            .iter()
            .map(|(_, _, _, ids)| {
                let b = Buffer::get(ids, false)?;
                if b.dtype != Dtype::U32 || b.shape.len() != 1 {
                    return Err(PyValueError::new_err("standard id maps require 1D uint32 arrays"));
                }
                Ok(b)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let regions: Vec<(PathBuf, u64, Option<u64>)> = matrices.iter().map(|(p, o, l, _)| (p.clone(), *o, *l)).collect();
        let inner = py
            .detach(|| {
                let mut builder = core::DataSourceBuilder::new(vocab, page_size)?.direct_io(direct_io);
                for ((path, offset, length), map) in regions.iter().zip(&maps) {
                    // SAFETY: the validated map is retained by `maps` and read here.
                    builder.add_cpk_region(path, *offset, *length, unsafe { map.slice::<u32>() })?;
                }
                builder.build()
            })
            .map_err(loader_error)?;
        let cells = match cells {
            Some(selected) => {
                if selected.len() != matrices.len() {
                    return Err(PyValueError::new_err("one cell selection per matrix"));
                }
                if !selected.iter().any(Option::is_some) {
                    None
                } else {
                    let mut table = Vec::new();
                    let mut counts = Vec::with_capacity(selected.len());
                    for (m, s) in selected.iter().enumerate() {
                        let (first, count) = inner.matrix_cells(m as u32).expect("matrices are registered in order");
                        let before = table.len();
                        match s {
                            None => table.extend((0..count).map(|c| first + c as u32)),
                            Some(a) => {
                                let b = Buffer::get(a, false)?;
                                if b.dtype != Dtype::U64 || b.shape.len() != 1 {
                                    return Err(PyValueError::new_err("cell selections require 1D uint64 arrays"));
                                }
                                // SAFETY: validated uint64 vector, read here.
                                for &c in unsafe { b.slice::<u64>() } {
                                    if c >= count {
                                        return Err(PyIndexError::new_err("selected cell outside its matrix"));
                                    }
                                    table.push(first + c as u32);
                                }
                            }
                        }
                        counts.push((table.len() - before) as u64);
                    }
                    Some((table.into(), counts))
                }
            }
            None => None,
        };
        Ok(Self { inner: Arc::new(inner), cells })
    }
    /// Global cells: every stored cell, or the cells the matrices contribute.
    #[getter]
    fn n_cells(&self) -> u64 {
        self.cells.as_ref().map_or_else(|| self.inner.n_cells(), |(t, _)| t.len() as u64)
    }
    /// The standard-id vocabulary size.
    #[getter]
    fn vocab(&self) -> usize {
        self.inner.vocab()
    }
    /// Pages of all files together, in the global page numbering plans use.
    #[getter]
    fn n_pages(&self) -> u64 {
        self.inner.n_pages()
    }
    /// The number of registered matrices.
    #[getter]
    fn n_matrices(&self) -> usize {
        self.inner.n_matrices()
    }
    /// The common record page size in bytes.
    #[getter]
    fn page_size(&self) -> usize {
        self.inner.page_size()
    }
    /// The largest page count of one cell.
    #[getter]
    fn max_cell_pages(&self) -> usize {
        self.inner.max_cell_pages()
    }
    /// A fingerprint of the page layout and cell numbering: page size, file
    /// lengths, cell counts and the chunk directory.
    #[getter]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }
    /// A fingerprint including the standard id maps and chunk metadata on
    /// top of the page layout.
    #[getter]
    fn mapping_fingerprint(&self) -> u64 {
        self.inner.mapping_fingerprint()
    }
    /// The stored dtype names, in registration order.
    #[getter]
    fn matrix_dtypes(&self) -> Vec<&'static str> {
        self.inner.matrix_dtypes().into_iter().map(crate::buffers::dtype_name).collect()
    }
    /// The first global cell of a matrix and its (contributed) cell count.
    fn matrix_cells(&self, matrix: u32) -> Option<(u64, u64)> {
        match &self.cells {
            None => self.inner.matrix_cells(matrix).map(|(f, n)| (f as u64, n)),
            Some((_, counts)) => counts.get(matrix as usize).map(|&n| (counts[..matrix as usize].iter().sum(), n)),
        }
    }
    /// The page run holding a cell: borrowed fd, aligned offset and page count.
    fn cell_pages(&self, cell: u64) -> PyResult<(i32, u64, usize)> {
        self.inner.cell_pages(self.stored_cell(cell)?).map_err(loader_error)
    }
    /// A cell's exact bytes: borrowed fd, offset and length.
    fn cell_bytes(&self, cell: u64) -> PyResult<(i32, u64, usize)> {
        self.inner.cell_bytes(self.stored_cell(cell)?).map_err(loader_error)
    }
    /// The stored cells of a sequence of global cell ids, as a uint32 vector.
    fn stored_cells<'py>(&self, py: Python<'py>, accesses: &Bound<'py, PyAny>) -> PyResult<Bound<'py, numpy::PyArray1<u32>>> {
        let cells = self.stored(py, accesses)?;
        Ok(numpy::PyArray1::from_slice(py, &cells))
    }
}
