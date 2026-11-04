//! Chunk metadata (`ScansMeta`) and the encode and decode tables derived
//! from it, with whole-selection dense and CSR decoding over the tables.

use std::sync::Arc;

use cellpack::format::{self, Dtype, RecordLayout};
use cellpack::scans as core;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{Metadata, Selection, fill, scoped, space, workspace_buffer};
use crate::buffers::{Buffer, bytes, disjoint};
use crate::errors::error;
use crate::store::GeneMap;

#[pyclass(module = "pycellpack._cellpack", name = "ScansMeta")]
pub(crate) struct MetaView {
    pub(super) inner: Arc<Metadata>,
}
#[pymethods]
impl MetaView {
    #[new]
    fn new(bytes: &Bound<'_, PyAny>) -> PyResult<Self> {
        let b = Buffer::get(bytes, false)?;
        b.expect(Dtype::U8, &[b.len])?;
        // SAFETY: validated immutable byte input retained through the snapshot.
        let bytes = unsafe { b.slice::<u8>() };
        Ok(Self {
            inner: Arc::new(Metadata::new(bytes).map_err(error)?),
        })
    }
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.inner.view();
        let d = PyDict::new(py);
        macro_rules! field {($($f:ident),*)=>{$(d.set_item(stringify!($f),m.$f())?;)*};}
        field!(
            n_cells,
            n_dims,
            value_width,
            index_width,
            renumbered,
            max_nnz,
            max_elems,
            max_k,
            nnz_total,
            meta_len,
            records_start,
            records_len,
            chunk_len,
            page_size
        );
        d.set_item("decode_workspace_words", core::decode_workspace_words(&m))?;
        d.set_item("has_modes", m.column_modes().is_some())?;
        Ok(d)
    }
    fn cell(&self, row: usize) -> PyResult<(usize, &'static str, u8, usize, usize)> {
        let m = self.inner.view();
        let r = m.record_range(row).map_err(error)?;
        Ok((
            m.cell_nnz(row).map_err(error)?,
            if m.cell_layout(row).map_err(error)? == RecordLayout::Dense {
                "dense"
            } else {
                "sparse"
            },
            m.cell_kind(row).map_err(error)?,
            r.start,
            r.end,
        ))
    }
    fn selection_sizes(&self, rows: &Bound<'_, PyAny>) -> PyResult<(u64, usize)> {
        let rows = Selection::get(rows)?;
        let m = self.inner.view();
        Ok((
            core::selection_nnz(&m, rows.rows()).map_err(error)?,
            core::dense_len(&m, rows.rows()).map_err(error)?,
        ))
    }
    fn copy_modes(&self, out: &Bound<'_, PyAny>) -> PyResult<bool> {
        let m = self.inner.view();
        let Some(modes) = m.column_modes() else {
            return Ok(false);
        };
        let mut b = Buffer::get(out, true)?;
        if b.len != modes.len() {
            return Err(PyValueError::new_err("column mode output length mismatch"));
        }
        // SAFETY: writable numeric output has exact byte size; source is private immutable metadata.
        unsafe { b.slice_mut::<u8>() }.copy_from_slice(modes);
        Ok(true)
    }
}

#[pyclass(module = "pycellpack._cellpack", name = "ScansEncodeTables")]
pub(crate) struct EncodeTables {
    pub(super) encode: core::EncodeTables,
}
#[pymethods]
impl EncodeTables {
    #[new]
    fn new(meta: PyRef<'_, MetaView>) -> PyResult<Self> {
        Ok(Self { encode: core::encode_tables(&meta.inner.view()).map_err(error)? })
    }
    fn check(&self, meta: PyRef<'_, MetaView>) -> PyResult<()> {
        self.encode.check(&meta.inner.view()).map_err(error)
    }
    fn gene_map(&self, out: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(map) = self.encode.coded_of_source() else { return Ok(false) };
        let mut out = Buffer::get(out, true)?;
        out.expect(Dtype::U32, &[map.len()])?;
        // SAFETY: exact writable u32 output and immutable native permutation.
        unsafe { out.slice_mut::<u32>() }.copy_from_slice(map);
        Ok(true)
    }
}

#[pyclass(module = "pycellpack._cellpack", name = "ScansDecodeTables")]
pub(crate) struct DecodeTables {
    pub(super) decode: core::DecodeTables,
    genes: usize,
}
#[pymethods]
impl DecodeTables {
    #[new]
    fn new(meta: PyRef<'_, MetaView>) -> PyResult<Self> {
        let m = meta.inner.view();
        Ok(Self {
            decode: core::decode_tables(&m).map_err(error)?,
            genes: m.n_dims() as usize,
        })
    }
    fn check(&self, meta: PyRef<'_, MetaView>) -> PyResult<()> {
        self.decode.check(&meta.inner.view()).map_err(error)
    }
    /// The source → panel map composed with the chunk's coded gene order,
    /// indexed by coded id.
    fn compose(&self, gene_map: PyRef<'_, GeneMap>) -> PyResult<GeneMap> {
        if gene_map.map.len() != self.genes {
            return Err(PyValueError::new_err("gene map length differs from the chunk's gene count"));
        }
        let mut panel = Vec::new();
        cellpack::store::compose_panel(&self.decode, Some(&gene_map.map), self.genes, &mut panel);
        Ok(GeneMap { map: panel.into(), columns: gene_map.columns })
    }
    fn gene_maps(&self, source: &Bound<'_, PyAny>, coded: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(a) = self.decode.source_of_coded() else {
            return Ok(false);
        };
        let b = self.decode.coded_of_source().unwrap();
        let (mut source, mut coded) = (Buffer::get(source, true)?, Buffer::get(coded, true)?);
        source.expect(Dtype::U32, &[a.len()])?;
        coded.expect(Dtype::U32, &[b.len()])?;
        disjoint(&[&source, &coded])?;
        // SAFETY: exact validated disjoint u32 outputs; private table arrays are immutable.
        unsafe { source.slice_mut::<u32>() }.copy_from_slice(a);
        // SAFETY: as above for the second output.
        unsafe { coded.slice_mut::<u32>() }.copy_from_slice(b);
        Ok(true)
    }
    /// Decodes the selected cells densely; selections of many rows are
    /// split over up to `workers` threads. `workspace` serves the first
    /// thread; the others allocate their own decode scratch.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (meta, records, rows, out, workspace, coded, zero, workers=1))]
    fn decode_dense_into(
        &self,
        py: Python<'_>,
        meta: PyRef<'_, MetaView>,
        records: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
        out: &Bound<'_, PyAny>,
        workspace: &Bound<'_, PyAny>,
        coded: bool,
        zero: bool,
        workers: usize,
    ) -> PyResult<()> {
        let rows = Selection::get(rows)?;
        let m = meta.inner.clone();
        let (rec, mut out, mut ws) = (bytes(records)?, Buffer::get(out, true)?, workspace_buffer(workspace)?);
        if out.dtype.width() != m.view().value_width() {
            return Err(PyValueError::new_err("output dtype width differs from metadata"));
        }
        disjoint(&[&rec, &out, &ws])?;
        rows.check_outputs(&[&out, &ws])?;
        let view = m.view();
        let pieces = rows.pieces(view.n_cells(), workers)?;
        let row_bytes = view.n_dims() as usize * view.value_width();
        py.detach(|| {
            // SAFETY: validated distinct byte output, u64 scratch and immutable records.
            let (records, output, first) = unsafe { (rec.slice::<u8>(), out.slice_mut::<u8>(), ws.slice_mut::<u64>()) };
            let need = core::dense_len(&view, rows.rows()).map_err(error)?;
            if output.len() != need {
                return Err(error(cellpack::Error::OutputTooSmall { need, have: output.len() }));
            }
            let mut rest = output;
            let mut first = Some(first);
            let tasks: Vec<_> = pieces
                .iter()
                .map(|&piece| {
                    let n = piece.count(view.n_cells()).unwrap_or(0);
                    let (part, tail) = std::mem::take(&mut rest).split_at_mut(n * row_bytes);
                    rest = tail;
                    let scratch = first.take();
                    let (view, tables) = (&view, &self.decode);
                    move || {
                        let mut own = Vec::new();
                        let ws = match scratch {
                            Some(ws) => ws,
                            None => {
                                own.resize(core::decode_workspace_words(view), 0);
                                &mut own[..]
                            }
                        };
                        core::decode_dense(view, tables, records, piece, space(coded), part, fill(zero), ws)
                    }
                })
                .collect();
            scoped(tasks).into_iter().collect::<Result<Vec<_>, _>>().map(drop).map_err(error)
        })
    }
    /// Decodes the selected cells as CSR; many rows are split over up to
    /// `workers` threads, each writing its rows' entries in place.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (meta, records, rows, indptr, indices, values, workspace, coded, workers=1))]
    fn decode_csr_into(
        &self,
        py: Python<'_>,
        meta: PyRef<'_, MetaView>,
        records: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        workspace: &Bound<'_, PyAny>,
        coded: bool,
        workers: usize,
    ) -> PyResult<u64> {
        let rows = Selection::get(rows)?;
        let m = meta.inner.clone();
        let (rec, mut ip, mut ix, mut vx, mut ws) = (
            bytes(records)?,
            Buffer::get(indptr, true)?,
            Buffer::get(indices, true)?,
            Buffer::get(values, true)?,
            workspace_buffer(workspace)?,
        );
        if ip.shape.len() != 1 || ix.shape.len() != 1 || vx.shape.len() != 1 || vx.dtype.width() != m.view().value_width() {
            return Err(PyValueError::new_err("CSR outputs require vectors with matching value width"));
        }
        disjoint(&[&rec, &ip, &ix, &vx, &ws])?;
        rows.check_outputs(&[&ip, &ix, &vx, &ws])?;
        let view = m.view();
        let pieces = rows.pieces(view.n_cells(), workers)?;
        let space = space(coded);
        py.detach(|| {
            macro_rules! write {
                ($P:ty, $I:ty) => {{
                    let (records, indptr, indices, values, ws) =
                        // SAFETY: matching index type dispatch and checked disjoint output/scratch buffers.
                        unsafe { (rec.slice::<u8>(), ip.slice_mut::<$P>(), ix.slice_mut::<$I>(), vx.slice_mut::<u8>(), ws.slice_mut::<u64>()) };
                    if pieces.len() == 1 {
                        core::decode_csr(&view, &self.decode, records, rows.rows(), space, indptr, indices, values, ws).map_err(error)
                    } else {
                        csr_pieces(&view, &self.decode, records, &pieces, space, indptr, indices, values, ws)
                    }
                }};
            }
            macro_rules! indices {($P:ty)=>{match ix.dtype {Dtype::U16=>write!($P,u16),Dtype::U32=>write!($P,u32),Dtype::U64=>write!($P,u64),Dtype::I32=>write!($P,i32),Dtype::I64=>write!($P,i64),_=>Err(PyTypeError::new_err("unsupported index dtype"))}};}
            match ip.dtype {Dtype::U32=>indices!(u32),Dtype::U64=>indices!(u64),Dtype::I32=>indices!(i32),Dtype::I64=>indices!(i64),_=>Err(PyTypeError::new_err("unsupported indptr dtype"))}
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn decode_record_into(
        &self,
        py: Python<'_>,
        meta: PyRef<'_, MetaView>,
        record: &Bound<'_, PyAny>,
        indices: Option<&Bound<'_, PyAny>>,
        values: &Bound<'_, PyAny>,
        workspace: &Bound<'_, PyAny>,
        coded: bool,
        zero: bool,
    ) -> PyResult<usize> {
        let m = meta.inner.clone();
        let (rec, mut ix, mut vx, mut ws) = (
            bytes(record)?,
            indices.map(|b| Buffer::get(b, true)).transpose()?,
            Buffer::get(values, true)?,
            workspace_buffer(workspace)?,
        );
        if vx.dtype.width() != m.view().value_width() {
            return Err(PyValueError::new_err("value width mismatch"));
        }
        let mut regions = [&rec, &vx, &ws, &rec];
        let n = if let Some(i) = &ix { regions[3] = i; 4 } else { 3 };
        disjoint(&regions[..n])?;
        py.detach(||{
            if let Some(ix)=&mut ix {
                macro_rules! write {($I:ty)=>{{
                    // SAFETY: dispatch matches indices; numeric values and scratch are disjoint.
                    unsafe{core::decode_record(&m.view(),&self.decode,rec.slice(),space(coded),ix.slice_mut::<$I>(),vx.slice_mut(),ws.slice_mut())}.map_err(error)
                }};}
                match ix.dtype {Dtype::U16=>write!(u16),Dtype::U32=>write!(u32),Dtype::U64=>write!(u64),Dtype::I32=>write!(i32),Dtype::I64=>write!(i64),_=>Err(PyTypeError::new_err("unsupported index dtype"))}
            }else{
                // SAFETY: distinct byte output and u64 scratch, retained records.
                unsafe{core::decode_record_dense(&m.view(),&self.decode,rec.slice(),space(coded),vx.slice_mut(),fill(zero),ws.slice_mut())}.map_err(error)?;
                // SAFETY: immutable record bytes remain valid.
                unsafe{format::record_nnz(rec.slice())}.map_err(error)
            }
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn decode_pages_into(
        &self,
        py: Python<'_>,
        meta: PyRef<'_, MetaView>,
        pages: Vec<Py<PyAny>>,
        page_size: usize,
        start: usize,
        length: usize,
        indices: Option<&Bound<'_, PyAny>>,
        values: &Bound<'_, PyAny>,
        workspace: &Bound<'_, PyAny>,
        coded: bool,
        zero: bool,
    ) -> PyResult<usize> {
        let pages = pages
            .iter()
            .map(|p| bytes(p.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let (mut ix, mut vx, mut ws) = (
            indices.map(|b| Buffer::get(b, true)).transpose()?,
            Buffer::get(values, true)?,
            workspace_buffer(workspace)?,
        );
        let m = meta.inner.clone();
        if vx.dtype.width() != m.view().value_width() {
            return Err(PyValueError::new_err("value width mismatch"));
        }
        let mut regions: Vec<_> = pages.iter().collect();
        regions.extend([&vx, &ws]);
        if let Some(i) = &ix {
            regions.push(i)
        }
        disjoint(&regions)?;
        py.detach(|| {
            // SAFETY: immutable retained page buffers; mutable outputs checked disjoint.
            let slices: Vec<&[u8]> = pages.iter().map(|p| unsafe { p.slice() }).collect();
            let record = core::PagedRecord {
                pages: &slices,
                page_size,
                start,
                len: length,
            };
            if let Some(ix) = &mut ix {
                macro_rules! write {
                    ($I:ty) => {{
                        // SAFETY: matching typed indices, disjoint outputs and scratch.
                        unsafe {
                            core::decode_record_pages(
                                &m.view(),
                                &self.decode,
                                record,
                                space(coded),
                                ix.slice_mut::<$I>(),
                                vx.slice_mut(),
                                ws.slice_mut(),
                            )
                        }
                        .map_err(error)
                    }};
                }
                match ix.dtype {
                    Dtype::U16 => write!(u16),
                    Dtype::U32 => write!(u32),
                    Dtype::U64 => write!(u64),
                    Dtype::I32 => write!(i32),
                    Dtype::I64 => write!(i64),
                    _ => Err(PyTypeError::new_err("unsupported index dtype")),
                }
            } else {
                // SAFETY: validated output dtype and retained input pages.
                unsafe {
                    core::decode_record_dense_pages(
                        &m.view(),
                        &self.decode,
                        record,
                        space(coded),
                        vx.slice_mut(),
                        fill(zero),
                        ws.slice_mut(),
                    )
                }
                .map_err(error)?;
                Ok(0)
            }
        })
    }
}

/// [`core::decode_csr`] over consecutive pieces on scoped threads: each
/// piece writes its entries at its own offset, so no copy follows.
#[allow(clippy::too_many_arguments)]
fn csr_pieces<P: core::IndptrOut + Send, I: core::IndexOut + Send>(
    meta: &format::Meta,
    tables: &core::DecodeTables,
    records: &[u8],
    pieces: &[core::Rows],
    space: core::IdSpace,
    indptr: &mut [P],
    indices: &mut [I],
    values: &mut [u8],
    ws: &mut [u64],
) -> PyResult<u64> {
    let vw = meta.value_width();
    let counts = pieces.iter().map(|p| p.count(meta.n_cells())).collect::<Result<Vec<_>, _>>().map_err(error)?;
    let sizes = pieces.iter().map(|&p| core::selection_nnz(meta, p).map(|n| n as usize)).collect::<Result<Vec<_>, _>>().map_err(error)?;
    let (rows, total) = (counts.iter().sum::<usize>(), sizes.iter().sum::<usize>());
    if total as u64 > P::MAX {
        return Err(error(cellpack::Error::OutputTypeTooNarrow(total as u64)));
    }
    if indptr.len() != rows + 1 || indices.len() < total || values.len() < total * vw {
        return Err(error(cellpack::Error::OutputTooSmall { need: total, have: indices.len().min(values.len() / vw) }));
    }
    indptr[0] = P::from_offset(0);
    let (mut ip, mut ix, mut vx) = (&mut indptr[1..], indices, values);
    let mut first = Some(ws);
    let mut base = 0usize;
    let mut tasks = Vec::with_capacity(pieces.len());
    for ((&piece, &n), &size) in pieces.iter().zip(&counts).zip(&sizes) {
        let (p, rest) = std::mem::take(&mut ip).split_at_mut(n);
        ip = rest;
        let (i, rest) = std::mem::take(&mut ix).split_at_mut(size);
        ix = rest;
        let (v, rest) = std::mem::take(&mut vx).split_at_mut(size * vw);
        vx = rest;
        let scratch = first.take();
        let offset = base;
        base += size;
        tasks.push(move || -> cellpack::Result<()> {
            let mut own = Vec::new();
            let ws = match scratch {
                Some(ws) => ws,
                None => {
                    own.resize(core::decode_workspace_words(meta), 0);
                    &mut own[..]
                }
            };
            let mut local = vec![0u64; n + 1];
            core::decode_csr(meta, tables, records, piece, space, &mut local, i, v, ws)?;
            for (o, &end) in p.iter_mut().zip(&local[1..]) {
                *o = P::from_offset(offset as u64 + end);
            }
            Ok(())
        });
    }
    scoped(tasks).into_iter().collect::<Result<Vec<_>, _>>().map_err(error)?;
    Ok(total as u64)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MetaView>()?;
    m.add_class::<EncodeTables>()?;
    m.add_class::<DecodeTables>()?;
    Ok(())
}
