//! End-to-end `.cpk` files for Python, as in `cellpack::store`: the file
//! handle with prepared reads into caller arrays, their reusable scratch,
//! the streaming writer and validated gene maps.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use cellpack::format::cpk::{Dtype, SourceLayout};
use cellpack::scans::{CsrView, DenseRef, UNMAPPED};
use cellpack::store::{CpkFile, CpkOptions, CpkWriter, RowSelection};
use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::buffers::{Buffer, disjoint, dtype_from_name, dtype_name, with_dtype};
use crate::errors::error;
use crate::scans::encode;

/// A validated source → panel column map: one target per source column,
/// distinct and below the panel width, or [`UNMAPPED`] to drop the column.
/// Built once, it is reused by every read and decode without revalidation.
#[pyclass(module = "pycellpack._cellpack", name = "GeneMap", frozen)]
pub(crate) struct GeneMap {
    pub(crate) map: Box<[u32]>,
    pub(crate) columns: usize,
}

impl GeneMap {
    fn targets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let array = PyArray1::from_slice(py, &self.map);
        array.getattr("flags")?.setattr("writeable", false)?;
        Ok(array)
    }
}

#[pymethods]
impl GeneMap {
    /// `targets` is an integer vector with one panel column per source
    /// column; -1 and `UNMAPPED` drop a column. `n_genes` is the panel
    /// width (default: one past the largest target).
    #[new]
    #[pyo3(signature = (targets, n_genes=None))]
    fn new(py: Python<'_>, targets: &Bound<'_, PyAny>, n_genes: Option<usize>) -> PyResult<Self> {
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("ascontiguousarray", (targets,))?;
        let a = array.cast::<PyUntypedArray>()?;
        if a.ndim() != 1 {
            return Err(PyTypeError::new_err("gene map requires a one-dimensional integer array"));
        }
        let kind = a.dtype().kind();
        let map: Vec<u32> = if a.len() == 0 {
            Vec::new()
        } else if kind == b'i' {
            let wide = numpy.call_method1("asarray", (&array, "int64"))?;
            let b = Buffer::get(&wide, false)?;
            // SAFETY: a retained contiguous int64 vector.
            unsafe { b.slice::<i64>() }
                .iter()
                .map(|&v| match v {
                    -1 => Ok(UNMAPPED),
                    v if v < -1 => Err(PyValueError::new_err("only -1 is accepted as an unmapped sentinel")),
                    v => u32::try_from(v).map_err(|_| PyValueError::new_err("gene map exceeds uint32")),
                })
                .collect::<PyResult<_>>()?
        } else if kind == b'u' {
            let wide = numpy.call_method1("asarray", (&array, "uint64"))?;
            let b = Buffer::get(&wide, false)?;
            // SAFETY: a retained contiguous uint64 vector.
            unsafe { b.slice::<u64>() }
                .iter()
                .map(|&v| u32::try_from(v).map_err(|_| PyValueError::new_err("gene map exceeds uint32")))
                .collect::<PyResult<_>>()?
        } else {
            return Err(PyTypeError::new_err("gene map requires a one-dimensional integer array"));
        };
        let needed = map.iter().filter(|&&c| c != UNMAPPED).map(|&c| c as usize + 1).max().unwrap_or(0);
        let columns = n_genes.unwrap_or(needed);
        if columns > u32::MAX as usize {
            return Err(PyValueError::new_err("panel width exceeds uint32"));
        }
        let mut seen = vec![0u64; columns.div_ceil(64)];
        for &c in map.iter().filter(|&&c| c != UNMAPPED) {
            let c = c as usize;
            if c >= columns {
                return Err(PyValueError::new_err("gene map targets must lie inside the panel width"));
            }
            if seen[c / 64] & (1 << (c % 64)) != 0 {
                return Err(PyValueError::new_err("gene map targets must be distinct"));
            }
            seen[c / 64] |= 1 << (c % 64);
        }
        Ok(Self { map: map.into(), columns })
    }

    /// Panel width.
    #[getter]
    fn n_genes(&self) -> usize {
        self.columns
    }

    /// Read-only uint32 targets, `UNMAPPED` for dropped columns.
    #[getter(targets)]
    fn get_targets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        self.targets(py)
    }

    /// Whether every source column keeps its position in a panel of the
    /// same width.
    #[getter]
    fn is_identity(&self) -> bool {
        self.columns == self.map.len() && self.map.iter().enumerate().all(|(i, &c)| c as usize == i)
    }

    fn __len__(&self) -> usize {
        self.map.len()
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(&self, py: Python<'py>, dtype: Option<&Bound<'py, PyAny>>, copy: Option<bool>) -> PyResult<Bound<'py, PyAny>> {
        let _ = copy;
        let targets = self.targets(py)?.into_any();
        match dtype {
            Some(dtype) => targets.call_method1("astype", (dtype,)),
            None => Ok(targets),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        PyTuple::new(py, [slf.get_type().into_any(), PyTuple::new(py, [this.targets(py)?.into_any(), this.columns.into_pyobject(py)?.into_any()])?.into_any()])
    }

    fn __repr__(&self) -> String {
        let kept = self.map.iter().filter(|&&c| c != UNMAPPED).count();
        format!("GeneMap(n_source={}, n_genes={}, mapped={kept})", self.map.len(), self.columns)
    }
}

/// Scratch ownership is independent of selections so callers can choose
/// sequential reuse or one workspace per concurrent reader. A read with
/// several workers keeps one scratch set per thread here.
#[pyclass(module = "pycellpack._cellpack", name = "ReadWorkspace")]
#[derive(Default)]
pub(crate) struct Workspace {
    scratch: Mutex<cellpack::store::ReadWorkspace>,
}

impl Workspace {
    fn lock(&self) -> PyResult<MutexGuard<'_, cellpack::store::ReadWorkspace>> {
        self.scratch.try_lock().map_err(|_| PyRuntimeError::new_err("read workspace is already in use"))
    }
}

#[pymethods]
impl Workspace {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Releases retained scratch; selections and outputs stay valid.
    fn clear(&self) -> PyResult<()> {
        self.lock()?.clear();
        Ok(())
    }
}

fn read_options(settings: Option<&Bound<'_, PyDict>>) -> PyResult<cellpack::store::ReadOptions> {
    let mut o = cellpack::store::ReadOptions::default();
    if let Some(d) = settings {
        if let Some(s) = d.get_item("strategy")? {
            o.strategy = match s.extract::<String>()?.as_str() {
                "auto" => cellpack::store::ReadStrategy::Auto,
                "ranges" => cellpack::store::ReadStrategy::Ranges,
                "whole" => cellpack::store::ReadStrategy::WholeChunk,
                _ => return Err(PyValueError::new_err("read strategy must be auto, ranges or whole")),
            };
        }
        if let Some(n) = d.get_item("max_gap_bytes")? {
            o.max_gap_bytes = n.extract()?;
        }
        if let Some(n) = d.get_item("request_cost_bytes")? {
            o.request_cost_bytes = n.extract()?;
        }
    }
    Ok(o)
}

fn read_summary<'py>(py: Python<'py>, s: cellpack::store::ReadSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("record_bytes", s.record_bytes)?;
    d.set_item("read_bytes", s.read_bytes)?;
    d.set_item("requests", s.requests)?;
    d.set_item("whole_chunk_bytes", s.whole_chunk_bytes)?;
    Ok(d)
}

fn format_name(layout: SourceLayout) -> &'static str {
    match layout {
        SourceLayout::Csr => "csr",
        SourceLayout::Dense => "dense",
    }
}

#[pyclass(module = "pycellpack._cellpack", name = "File")]
pub(crate) struct File {
    inner: CpkFile,
}

#[pymethods]
impl File {
    #[new]
    #[pyo3(signature = (path, offset=0, length=None))]
    fn new(py: Python<'_>, path: PathBuf, offset: u64, length: Option<u64>) -> PyResult<Self> {
        Ok(Self { inner: py.detach(|| CpkFile::open_region(path, offset, length)).map_err(error)? })
    }

    /// Device and inode of the opened descriptor.
    #[getter]
    fn identity(&self) -> PyResult<(u64, u64)> {
        self.inner.identity().map_err(error)
    }

    #[getter]
    fn shape(&self) -> (u64, u32) {
        (self.inner.n_rows(), self.inner.n_cols())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.dtype())
    }

    #[getter]
    fn format(&self) -> &'static str {
        format_name(self.inner.superblock().source)
    }

    #[getter]
    fn page_size(&self) -> u32 {
        self.inner.superblock().page_size
    }

    /// `rows` is `(start, stop)` for a contiguous range or a uint64 vector
    /// of cell ids in output order; `gene_map` defaults to every column in
    /// place.
    #[pyo3(signature = (rows, gene_map=None, options=None))]
    fn prepare_read(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        gene_map: Option<PyRef<'_, GeneMap>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<ReadSelection> {
        let options = read_options(options)?;
        let list = if rows.is_instance_of::<PyTuple>() {
            None
        } else {
            let b = Buffer::get(rows, false)?;
            if b.dtype != Dtype::U64 || b.shape.len() != 1 {
                return Err(PyTypeError::new_err("row selections require a uint64 vector"));
            }
            Some(b)
        };
        let range = if list.is_none() { Some(rows.extract::<(u64, u64)>()?) } else { None };
        let (map, columns) = match &gene_map {
            Some(m) => (Some(&m.map[..]), m.columns),
            None => (None, self.inner.n_cols() as usize),
        };
        let inner = py
            .detach(|| {
                let selection = match (&list, range) {
                    // SAFETY: a retained, validated uint64 vector the caller keeps immutable.
                    (Some(b), _) => RowSelection::List(unsafe { b.slice::<u64>() }),
                    (None, Some((start, stop))) => RowSelection::Range(start..stop.max(start)),
                    (None, None) => unreachable!("rows are a range or a list"),
                };
                self.inner.prepare_selection(selection, map, columns, options)
            })
            .map_err(error)?;
        Ok(ReadSelection { inner, workspace: Workspace::default() })
    }

    fn chunk_info(&self) -> Vec<(u64, u64, u64, u64, u32)> {
        self.inner.chunks().iter().map(|e| (e.first_row, e.offset, e.len, e.meta_len, e.n_rows)).collect()
    }

    #[getter]
    fn n_chunks(&self) -> usize {
        self.inner.chunks().len()
    }

    fn chunk_info_one(&self, chunk: usize) -> PyResult<(u64, u64, u64, u64, u32)> {
        let e = self.inner.chunks().get(chunk).ok_or_else(|| PyValueError::new_err("chunk out of range"))?;
        Ok((e.first_row, e.offset, e.len, e.meta_len, e.n_rows))
    }

    fn read_chunk_into(&self, py: Python<'_>, chunk: usize, out: &Bound<'_, PyAny>, metadata: bool) -> PyResult<()> {
        let mut out = Buffer::get(out, true)?;
        let e = self.inner.chunks().get(chunk).ok_or_else(|| PyValueError::new_err("chunk out of range"))?;
        let len = if metadata { e.meta_len } else { e.len } as usize;
        out.expect(Dtype::U8, &[len])?;
        py.detach(|| {
            // SAFETY: exact contiguous writable byte output; owner retained for the call.
            self.inner.read_chunk_into(chunk, metadata, unsafe { out.slice_mut() }).map_err(error)
        })
    }
}

/// One selection for ranges and gathers alike; outputs of any numeric
/// dtype receive converted values.
#[pyclass(module = "pycellpack._cellpack", name = "ReadSelection")]
pub(crate) struct ReadSelection {
    inner: cellpack::store::ReadSelection,
    workspace: Workspace,
}

#[pymethods]
impl ReadSelection {
    fn io_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        read_summary(py, self.inner.io_summary())
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        let s = self.inner.spec();
        (s.rows, s.columns)
    }

    /// Stored dtype of the source values.
    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.spec().dtype)
    }

    /// Required sparse output capacity; dropped genes reduce the result nnz.
    #[getter]
    fn capacity_nnz(&self) -> usize {
        self.inner.spec().nnz
    }

    #[pyo3(signature = (out, workspace=None, workers=1))]
    fn read_dense_into(&self, py: Python<'_>, out: &Bound<'_, PyAny>, workspace: Option<PyRef<'_, Workspace>>, workers: usize) -> PyResult<()> {
        let workspace = workspace.as_deref().unwrap_or(&self.workspace);
        let mut out = Buffer::get(out, true)?;
        let s = self.inner.spec();
        if out.shape != [s.rows, s.columns] {
            return Err(PyValueError::new_err("output shape differs from selection"));
        }
        py.detach(|| {
            let mut scratch = workspace.lock()?;
            with_dtype!(out.dtype, T, {
                // SAFETY: validated writable output of exactly the selection's shape.
                unsafe { self.inner.read_dense_into::<T>(out.slice_mut(), &mut scratch, workers) }.map_err(error)
            })
        })
    }

    #[pyo3(signature = (indptr, indices, data, workspace=None, workers=1))]
    fn read_csr_into(
        &self,
        py: Python<'_>,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        data: &Bound<'_, PyAny>,
        workspace: Option<PyRef<'_, Workspace>>,
        workers: usize,
    ) -> PyResult<usize> {
        let workspace = workspace.as_deref().unwrap_or(&self.workspace);
        let (mut ip, mut ix, mut vx) = (Buffer::get(indptr, true)?, Buffer::get(indices, true)?, Buffer::get(data, true)?);
        let s = self.inner.spec();
        if ip.shape != [s.rows + 1] || ix.shape.len() != 1 || vx.shape.len() != 1 || ix.shape[0] < s.nnz || vx.shape[0] < s.nnz {
            return Err(PyValueError::new_err("CSR output buffers differ from the selection's rows or capacity"));
        }
        if !matches!(ip.dtype, Dtype::I32 | Dtype::I64 | Dtype::U32 | Dtype::U64) {
            return Err(PyTypeError::new_err("unsupported CSR indptr dtype"));
        }
        if !matches!(ix.dtype, Dtype::I32 | Dtype::I64 | Dtype::U16 | Dtype::U32 | Dtype::U64) {
            return Err(PyTypeError::new_err("unsupported CSR index dtype"));
        }
        disjoint(&[&ip, &ix, &vx])?;
        py.detach(|| {
            let mut scratch = workspace.lock()?;
            macro_rules! read {
                ($P:ty, $I:ty) => {
                    with_dtype!(vx.dtype, T, {
                        // SAFETY: dtype dispatch matches each array; outputs were checked disjoint.
                        unsafe { self.inner.read_csr_into::<$P, $I, T>(ip.slice_mut(), ix.slice_mut(), vx.slice_mut(), &mut scratch, workers) }
                            .map_err(error)
                    })
                };
            }
            macro_rules! indices {
                ($P:ty) => {
                    match ix.dtype {
                        Dtype::I32 => read!($P, i32),
                        Dtype::I64 => read!($P, i64),
                        Dtype::U16 => read!($P, u16),
                        Dtype::U32 => read!($P, u32),
                        _ => read!($P, u64),
                    }
                };
            }
            match ip.dtype {
                Dtype::I32 => indices!(i32),
                Dtype::I64 => indices!(i64),
                Dtype::U32 => indices!(u32),
                _ => indices!(u64),
            }
        })
    }
}

#[pyclass(module = "pycellpack._cellpack", name = "Writer")]
pub(crate) struct Writer {
    inner: Option<CpkWriter>,
    dtype: Dtype,
    columns: u32,
}

impl Writer {
    fn open(&mut self) -> PyResult<&mut CpkWriter> {
        self.inner.as_mut().ok_or_else(|| PyRuntimeError::new_err("writer is closed"))
    }
}

#[pymethods]
impl Writer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, n_cols, dtype, format="csr", chunk_rows=4096, page_size=4096, options=None, threads=1))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        n_cols: u32,
        dtype: &str,
        format: &str,
        chunk_rows: usize,
        page_size: u32,
        options: Option<&Bound<'_, PyDict>>,
        threads: usize,
    ) -> PyResult<Self> {
        let dtype = dtype_from_name(dtype)?;
        let source = match format {
            "csr" => SourceLayout::Csr,
            "dense" => SourceLayout::Dense,
            _ => return Err(PyValueError::new_err("format must be csr or dense")),
        };
        let encode = match options {
            Some(d) => encode::options(d)?,
            None => encode::default_options(),
        };
        let options = CpkOptions { encode, chunk_rows, page_size };
        let mut inner = py.detach(|| CpkWriter::create(path, n_cols, dtype, source, options)).map_err(error)?;
        inner.set_threads(threads);
        Ok(Self { inner: Some(inner), dtype, columns: n_cols })
    }

    /// Appends `(rows, n_cols)` dense rows in either stored format.
    fn append_dense(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = Buffer::get(data, false)?;
        if data.dtype != self.dtype || data.shape.len() != 2 || data.shape[1] != self.columns as usize {
            return Err(PyValueError::new_err("dense input differs from the writer's dtype or columns"));
        }
        let columns = self.columns;
        let inner = self.open()?;
        py.detach(|| {
            inner
                .append(DenseRef {
                    n_genes: columns,
                    value_width: data.dtype.width(),
                    // SAFETY: the retained input stays immutable during append.
                    data: unsafe { data.slice() },
                })
                .map_err(error)
        })
    }

    /// Appends canonical CSR rows in either stored format.
    fn append_csr(&mut self, py: Python<'_>, indptr: &Bound<'_, PyAny>, indices: &Bound<'_, PyAny>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let (ip, ix, vx) = (Buffer::get(indptr, false)?, Buffer::get(indices, false)?, Buffer::get(data, false)?);
        if vx.dtype != self.dtype || ip.shape.len() != 1 || ix.shape.len() != 1 || vx.shape != ix.shape || ip.shape[0] == 0 {
            return Err(PyValueError::new_err("CSR input differs from the writer's dtype or dimensions"));
        }
        // SAFETY: all views match their validated integer dtypes and live through this call.
        let (offsets, genes) = unsafe { (ip.offsets()?, ix.indices()?) };
        if offsets.get(0).map_err(error)? != 0 || offsets.get(offsets.len() - 1).map_err(error)? != ix.shape[0] as u64 {
            return Err(PyValueError::new_err("CSR indptr must start at zero and end at nnz"));
        }
        let columns = self.columns;
        let inner = self.open()?;
        py.detach(|| {
            let input = CsrView {
                n_genes: columns,
                value_width: vx.dtype.width(),
                indptr: offsets,
                indices: genes,
                // SAFETY: numeric arrays stay retained and immutable throughout append.
                data: unsafe { vx.slice() },
            };
            inner.append(input).map_err(error)
        })
    }

    fn end_chunk(&mut self, py: Python<'_>) -> PyResult<()> {
        let w = self.open()?;
        py.detach(|| w.end_chunk()).map_err(error)
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        let w = self.inner.take().ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;
        py.detach(|| w.finish()).map_err(error)
    }

    fn abort(&mut self) {
        self.inner.take();
    }
}

/// Adds the storage classes to the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<File>()?;
    m.add_class::<ReadSelection>()?;
    m.add_class::<Workspace>()?;
    m.add_class::<Writer>()?;
    m.add_class::<GeneMap>()?;
    Ok(())
}
