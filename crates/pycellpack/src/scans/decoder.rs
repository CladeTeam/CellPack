//! Prepared decoding (`ScansDecoder`): records of one chunk straight into a
//! panel of the target dtype, with the coded → panel map composed once and
//! per-thread scratch reused across calls.

use std::sync::Arc;

use cellpack::format::Dtype;
use cellpack::scans as core;
use cellpack::store::{OutputElement, PanelRows, RowScratch};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::tables::MetaView;
use super::{Metadata, Selection, fill, scoped};
use crate::buffers::{Buffer, bytes, disjoint, dtype_from_name, with_dtype};
use crate::errors::error;
use crate::store::GeneMap;

/// Per-call scratch of a [`PreparedDecoder`]: one row scratch per thread
/// and the row bookkeeping of CSR decodes.
struct DecoderScratch<T> {
    threads: Vec<RowScratch<T>>,
    cols: Vec<u32>,
    starts: Vec<usize>,
    counts: Vec<usize>,
}

impl<T> Default for DecoderScratch<T> {
    fn default() -> Self {
        Self { threads: Vec::new(), cols: Vec::new(), starts: Vec::new(), counts: Vec::new() }
    }
}

/// Decodes records of one chunk straight into a panel of the target dtype:
/// the coded → panel map is composed once and scratch is reused per call.
#[pyclass(module = "pycellpack._cellpack", name = "ScansDecoder")]
struct PreparedDecoder {
    meta: Arc<Metadata>,
    tables: core::DecodeTables,
    panel: Vec<u32>,
    columns: usize,
    source: Dtype,
    target: Dtype,
    scratch: std::sync::Mutex<Box<dyn std::any::Any + Send>>,
}

impl PreparedDecoder {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Box<dyn std::any::Any + Send>>> {
        self.scratch.try_lock().map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("decoder is already in use"))
    }

    /// Scatters `pieces` of rows into consecutive dense output rows.
    fn dense<S: core::CastInto<T>, T: OutputElement>(&self, records: &[u8], pieces: &[core::Rows], out: &mut [T], zero: bool, scratch: &mut DecoderScratch<T>) -> PyResult<()> {
        let m = self.meta.view();
        let mapper = core::Mapper::<S, T>::new().map_err(error)?;
        let rows = PanelRows::new(&m, &self.tables, &self.panel, &mapper).map_err(error)?;
        if scratch.threads.len() < pieces.len() {
            scratch.threads.resize_with(pieces.len(), RowScratch::default);
        }
        let mut rest = out;
        let tasks: Vec<_> = pieces
            .iter()
            .zip(scratch.threads.iter_mut())
            .map(|(&piece, s)| {
                let n = piece.count(m.n_cells()).unwrap_or(0);
                let (part, tail) = std::mem::take(&mut rest).split_at_mut(n * self.columns);
                rest = tail;
                let (m, rows) = (&m, &rows);
                move || -> cellpack::Result<()> {
                    rows.bind(s)?;
                    for (i, row) in part.chunks_exact_mut(self.columns.max(1)).enumerate().take(n) {
                        let cell = piece.get(i);
                        rows.dense(m.record(records, cell)?, Some(m.cell_nnz(cell)?), row, fill(zero), s)?;
                    }
                    Ok(())
                }
            })
            .collect();
        scoped(tasks).into_iter().collect::<Result<Vec<_>, _>>().map(drop).map_err(error)
    }

    /// Decodes `pieces` of rows as CSR at their capacity offsets, then
    /// compacts the values in place and the columns into `scratch.cols`.
    /// Returns the row ends of the compacted entries.
    fn csr<S: core::CastInto<T>, T: OutputElement>(&self, records: &[u8], pieces: &[core::Rows], values: &mut [T], scratch: &mut DecoderScratch<T>) -> PyResult<Vec<u64>> {
        let m = self.meta.view();
        let mapper = core::Mapper::<S, T>::new().map_err(error)?;
        let rows = PanelRows::new(&m, &self.tables, &self.panel, &mapper).map_err(error)?;
        let DecoderScratch { threads, cols, starts, counts } = scratch;
        starts.clear();
        starts.push(0);
        for piece in pieces {
            for i in 0..piece.count(m.n_cells()).map_err(error)? {
                starts.push(starts.last().unwrap() + m.cell_nnz(piece.get(i)).map_err(error)?);
            }
        }
        let n_rows = starts.len() - 1;
        let capacity = starts[n_rows];
        if values.len() < capacity {
            return Err(PyValueError::new_err("CSR output buffers are smaller than the selection's capacity"));
        }
        cols.resize(capacity, 0);
        counts.clear();
        counts.resize(n_rows, 0);
        if threads.len() < pieces.len() {
            threads.resize_with(pieces.len(), RowScratch::default);
        }
        let (mut vrest, mut crest, mut nrest) = (&mut values[..capacity], &mut cols[..], &mut counts[..]);
        let mut first = 0;
        let mut tasks = Vec::with_capacity(pieces.len());
        for (&piece, s) in pieces.iter().zip(threads.iter_mut()) {
            let n = piece.count(m.n_cells()).map_err(error)?;
            let span = starts[first + n] - starts[first];
            let (v, tail) = std::mem::take(&mut vrest).split_at_mut(span);
            vrest = tail;
            let (c, tail) = std::mem::take(&mut crest).split_at_mut(span);
            crest = tail;
            let (k, tail) = std::mem::take(&mut nrest).split_at_mut(n);
            nrest = tail;
            let (m, rows, starts, base) = (&m, &rows, &starts[first..=first + n], starts[first]);
            first += n;
            tasks.push(move || -> cellpack::Result<()> {
                rows.bind(s)?;
                for i in 0..n {
                    let cell = piece.get(i);
                    let (lo, hi) = (starts[i] - base, starts[i + 1] - base);
                    let kept = rows.csr(m.record(records, cell)?, Some(hi - lo), &mut v[lo..hi], s)?;
                    c[lo..lo + kept.len()].copy_from_slice(kept);
                    k[i] = kept.len();
                }
                Ok(())
            });
        }
        scoped(tasks).into_iter().collect::<Result<Vec<_>, _>>().map_err(error)?;
        let mut ends = Vec::with_capacity(n_rows);
        let mut cursor = 0;
        for row in 0..n_rows {
            let (start, n) = (starts[row], counts[row]);
            if start != cursor {
                values.copy_within(start..start + n, cursor);
                cols.copy_within(start..start + n, cursor);
            }
            cursor += n;
            ends.push(cursor as u64);
        }
        Ok(ends)
    }
}

fn write_indptr<P: core::IndptrOut>(out: &mut [P], ends: &[u64]) {
    out[0] = P::from_offset(0);
    for (o, &e) in out[1..].iter_mut().zip(ends) {
        *o = P::from_offset(e);
    }
}

fn write_indices<I: core::IndexOut>(out: &mut [I], cols: &[u32]) {
    for (o, &c) in out.iter_mut().zip(cols) {
        *o = I::from_id(c);
    }
}

#[pymethods]
impl PreparedDecoder {
    /// `gene_map` maps the chunk's source genes; `source` is the stored
    /// dtype and `target` the output dtype.
    #[new]
    fn new(meta: PyRef<'_, MetaView>, gene_map: PyRef<'_, GeneMap>, source: &str, target: &str) -> PyResult<Self> {
        let meta = meta.inner.clone();
        let m = meta.view();
        let (source, target) = (dtype_from_name(source)?, dtype_from_name(target)?);
        if source.width() != m.value_width() {
            return Err(PyValueError::new_err("source width differs from metadata"));
        }
        if gene_map.map.len() != m.n_dims() as usize {
            return Err(PyValueError::new_err("gene map length differs from the chunk's gene count"));
        }
        let tables = core::decode_tables(&m).map_err(error)?;
        let mut panel = Vec::new();
        cellpack::store::compose_panel(&tables, Some(&gene_map.map), m.n_dims() as usize, &mut panel);
        let scratch: Box<dyn std::any::Any + Send> = with_dtype!(target, T, Box::new(DecoderScratch::<T>::default()));
        Ok(Self { meta, tables, panel, columns: gene_map.columns, source, target, scratch: std::sync::Mutex::new(scratch) })
    }

    /// Dense rows of the selected cells (or of the single `record` passed
    /// as `records` with `single`), on up to `workers` threads.
    #[pyo3(signature = (records, rows, out, *, single=false, zero=true, workers=1))]
    #[allow(clippy::too_many_arguments)]
    fn decode_into(&self, py: Python<'_>, records: &Bound<'_, PyAny>, rows: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>, single: bool, zero: bool, workers: usize) -> PyResult<()> {
        let rows = Selection::get(rows)?;
        let rec = bytes(records)?;
        let mut out = Buffer::get(out, true)?;
        let m = self.meta.view();
        let count = if single { 1 } else { rows.rows().count(m.n_cells()).map_err(error)? };
        out.expect(self.target, &[count, self.columns])?;
        disjoint(&[&rec, &out])?;
        rows.check_outputs(&[&out])?;
        let pieces = if single { Vec::new() } else { rows.pieces(m.n_cells(), workers)? };
        py.detach(|| {
            let mut owned = self.lock()?;
            with_dtype!(self.source, S, with_dtype!(self.target, T, {
                let scratch = owned.downcast_mut::<DecoderScratch<T>>().expect("scratch matches target dtype");
                // SAFETY: inputs and exact writable output have checked dtype, alignment and disjointness.
                let (records, output) = unsafe { (rec.slice::<u8>(), out.slice_mut::<T>()) };
                if single {
                    let mapper = core::Mapper::<S, T>::new().map_err(error)?;
                    let rows = PanelRows::new(&m, &self.tables, &self.panel, &mapper).map_err(error)?;
                    if scratch.threads.is_empty() {
                        scratch.threads.push(RowScratch::default());
                    }
                    let s = &mut scratch.threads[0];
                    rows.bind(s).map_err(error)?;
                    return rows.dense(records, None, output, fill(zero), s).map_err(error);
                }
                self.dense::<S, T>(records, &pieces, output, zero, scratch)
            }))
        })
    }

    /// Canonical CSR of the selected cells in panel columns: rows are
    /// decoded at their capacity offsets and compacted in place. Returns
    /// the nnz.
    #[pyo3(signature = (records, rows, indptr, indices, values, workers=1))]
    #[allow(clippy::too_many_arguments)]
    fn decode_csr_into(&self, py: Python<'_>, records: &Bound<'_, PyAny>, rows: &Bound<'_, PyAny>, indptr: &Bound<'_, PyAny>, indices: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>, workers: usize) -> PyResult<usize> {
        let rows = Selection::get(rows)?;
        let rec = bytes(records)?;
        let (mut ip, mut ix, mut vx) = (Buffer::get(indptr, true)?, Buffer::get(indices, true)?, Buffer::get(values, true)?);
        let m = self.meta.view();
        let count = rows.rows().count(m.n_cells()).map_err(error)?;
        let capacity = core::selection_nnz(&m, rows.rows()).map_err(error)? as usize;
        if ip.shape != [count + 1] || ix.shape.len() != 1 || vx.shape.len() != 1 || ix.shape[0] < capacity || vx.shape[0] < capacity {
            return Err(PyValueError::new_err("CSR output buffers differ from the selection's rows or capacity"));
        }
        if vx.dtype != self.target {
            return Err(PyValueError::new_err("CSR values differ from the decoder's dtype"));
        }
        let largest = self.columns.saturating_sub(1) as u64;
        let fits = |dtype: Dtype, max: u64| match dtype {
            Dtype::U16 => Ok(max <= u16::MAX as u64),
            Dtype::U32 => Ok(max <= u32::MAX as u64),
            Dtype::I32 => Ok(max <= i32::MAX as u64),
            Dtype::U64 | Dtype::I64 => Ok(true),
            _ => Err(PyTypeError::new_err("unsupported CSR index dtype")),
        };
        if ip.dtype == Dtype::U16 {
            return Err(PyTypeError::new_err("unsupported CSR indptr dtype"));
        }
        if !fits(ip.dtype, capacity as u64)? || !fits(ix.dtype, largest)? {
            return Err(error(cellpack::Error::OutputTypeTooNarrow((capacity as u64).max(largest))));
        }
        disjoint(&[&rec, &ip, &ix, &vx])?;
        rows.check_outputs(&[&ip, &ix, &vx])?;
        let pieces = rows.pieces(m.n_cells(), workers)?;
        py.detach(|| {
            let mut owned = self.lock()?;
            // SAFETY: immutable retained records.
            let records = unsafe { rec.slice::<u8>() };
            let (ends, scratch_cols) = with_dtype!(self.source, S, with_dtype!(self.target, T, {
                let scratch = owned.downcast_mut::<DecoderScratch<T>>().expect("scratch matches target dtype");
                // SAFETY: validated writable values of the decoder's dtype, disjoint from inputs.
                let ends = self.csr::<S, T>(records, &pieces, unsafe { vx.slice_mut::<T>() }, scratch)?;
                (ends, std::mem::take(&mut scratch.cols))
            }));
            let nnz = ends.last().copied().unwrap_or(0) as usize;
            // SAFETY: dispatch matches each validated integer output; regions are disjoint.
            unsafe {
                match ip.dtype {
                    Dtype::I32 => write_indptr::<i32>(ip.slice_mut(), &ends),
                    Dtype::I64 => write_indptr::<i64>(ip.slice_mut(), &ends),
                    Dtype::U32 => write_indptr::<u32>(ip.slice_mut(), &ends),
                    _ => write_indptr::<u64>(ip.slice_mut(), &ends),
                }
                let cols = &scratch_cols[..nnz];
                match ix.dtype {
                    Dtype::U16 => write_indices::<u16>(ix.slice_mut(), cols),
                    Dtype::U32 => write_indices::<u32>(ix.slice_mut(), cols),
                    Dtype::I32 => write_indices::<i32>(ix.slice_mut(), cols),
                    Dtype::I64 => write_indices::<i64>(ix.slice_mut(), cols),
                    _ => write_indices::<u64>(ix.slice_mut(), cols),
                }
            }
            with_dtype!(self.target, T, {
                owned.downcast_mut::<DecoderScratch<T>>().expect("scratch matches target dtype").cols = scratch_cols;
            });
            Ok(nnz)
        })
    }

    /// Output columns.
    #[getter]
    fn n_genes(&self) -> usize {
        self.columns
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PreparedDecoder>()
}
