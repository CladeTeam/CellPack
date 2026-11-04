//! Decoded record parts (`ScansParts`): ids, dictionary and ranks before
//! values are materialized, with mapping onto a caller's panel and value
//! casts.

use std::sync::Arc;

use cellpack::format::{Dtype, RecordLayout};
use cellpack::scans as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::tables::{DecodeTables, MetaView};
use super::{Metadata, fill, space};
use crate::buffers::{Buffer, bytes, disjoint, dtype_from_name, with_dtype};
use crate::errors::error;
use crate::store::GeneMap;

/// The private boxes never resize; Parts borrows only these owned allocations.
#[pyclass(module = "pycellpack._cellpack", name = "ScansParts")]
pub(crate) struct Parts {
    parts: core::Parts<'static>,
    _metadata: Arc<Metadata>,
    _record: Box<[u8]>,
    _pages: Vec<Box<[u8]>>,
    _workspace: Box<[u64]>,
}
#[pymethods]
impl Parts {
    #[new]
    fn new(
        meta: PyRef<'_, MetaView>,
        tables: PyRef<'_, DecodeTables>,
        record: &Bound<'_, PyAny>,
        coded: bool,
    ) -> PyResult<Self> {
        let record = bytes(record)?;
        // SAFETY: input is retained until copied into the private immutable owner.
        let record: Box<[u8]> = unsafe { record.slice::<u8>() }.into();
        let metadata = meta.inner.clone();
        let m = metadata.view();
        let mut ws = vec![0; core::decode_workspace_words(&m)].into_boxed_slice();
        let p = core::decode_parts(&m, &tables.decode, &record, space(coded), &mut ws)
            .map_err(error)?;
        // SAFETY: all referents are private stable allocations retained in this
        // object, never mutated or resized. Returned references borrow self only.
        let p = unsafe { std::mem::transmute::<core::Parts<'_>, core::Parts<'static>>(p) };
        Ok(Self {
            parts: p,
            _metadata: metadata,
            _record: record,
            _pages: vec![],
            _workspace: ws,
        })
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_pages(
        py: Python<'_>,
        meta: PyRef<'_, MetaView>,
        tables: PyRef<'_, DecodeTables>,
        pages: Vec<Py<PyAny>>,
        page_size: usize,
        start: usize,
        length: usize,
        coded: bool,
    ) -> PyResult<Self> {
        let mut owned = Vec::<Box<[u8]>>::new();
        for p in pages {
            let p = bytes(p.bind(py))?;
            // SAFETY: snapshot retained input before making any decoded views.
            owned.push(unsafe { p.slice::<u8>() }.into());
        }
        let slices: Vec<&[u8]> = owned.iter().map(|p| p.as_ref()).collect();
        let metadata = meta.inner.clone();
        let m = metadata.view();
        let mut ws = vec![0; core::decode_workspace_words(&m)].into_boxed_slice();
        let p = core::decode_parts_pages(
            &m,
            &tables.decode,
            core::PagedRecord {
                pages: &slices,
                page_size,
                start,
                len: length,
            },
            space(coded),
            &mut ws,
        )
        .map_err(error)?;
        // SAFETY: private stable page, metadata and workspace boxes are retained;
        // moving their owners does not move allocations. None is exposed mutable.
        let p = unsafe { std::mem::transmute::<core::Parts<'_>, core::Parts<'static>>(p) };
        Ok(Self {
            parts: p,
            _metadata: metadata,
            _record: Box::new([]),
            _pages: owned,
            _workspace: ws,
        })
    }
    fn info(
        &self,
    ) -> (
        usize,
        usize,
        &'static str,
        &'static str,
        usize,
        &'static str,
    ) {
        let (kind, k, ranks) = match self.parts.values() {
            core::Values::Raw(_) => ("raw", 0, "none"),
            core::Values::Dict(d) => ("dict", d.k(), rank_kind(d.ranks())),
            core::Values::Modes(d) => ("modes", d.k(), rank_kind(d.ranks())),
        };
        (
            self.parts.nnz(),
            self.parts.elements(),
            if self.parts.layout() == RecordLayout::Dense {
                "dense"
            } else {
                "sparse"
            },
            kind,
            k,
            ranks,
        )
    }
    fn copy_component(&self, which: &str, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut out = Buffer::get(out, true)?;
        match which {
            "ids" => {
                out.expect(Dtype::U32, &[self.parts.elements()])?;
                // SAFETY: exact writable u32 output, immutable owned parts ids.
                unsafe { out.slice_mut::<u32>() }.copy_from_slice(self.parts.ids());
            }
            "values" => {
                if out.dtype.width() != self._metadata.view().value_width() {
                    return Err(PyValueError::new_err("value width mismatch"));
                }
                // SAFETY: numeric byte output, disjoint from private decoded storage.
                let output = unsafe { out.slice_mut() };
                self.parts.expand_values(output).map_err(error)?;
            }
            "entries" => {
                let b = match self.parts.values() {
                    core::Values::Dict(d) => d.entries(),
                    core::Values::Modes(d) => d.entries(),
                    _ => &[],
                };
                if out.len != b.len() {
                    return Err(PyValueError::new_err("dictionary output size mismatch"));
                }
                // SAFETY: exact byte output with private immutable dictionary input.
                unsafe { out.slice_mut::<u8>() }.copy_from_slice(b);
            }
            "modes" => {
                let b = self._metadata.view().column_modes().unwrap_or(&[]);
                if out.len != b.len() {
                    return Err(PyValueError::new_err("modes output size mismatch"));
                }
                // SAFETY: exact byte output distinct from private metadata.
                unsafe { out.slice_mut::<u8>() }.copy_from_slice(b);
            }
            "ranks" => {
                let r = match self.parts.values() {
                    core::Values::Dict(d) => d.ranks(),
                    core::Values::Modes(d) => d.ranks(),
                    _ => return Err(PyValueError::new_err("raw values have no ranks")),
                };
                match r {
                    core::Ranks::One => {
                        out.expect(Dtype::U8, &[self.parts.elements()])?;
                        // SAFETY: exact writable u8 output.
                        unsafe { out.slice_mut::<u8>() }.fill(0);
                    }
                    core::Ranks::U8(r) => {
                        out.expect(Dtype::U8, &[r.len()])?;
                        // SAFETY: exact u8 output with private ranks.
                        unsafe { out.slice_mut::<u8>() }.copy_from_slice(r);
                    }
                    core::Ranks::U32(r) => {
                        out.expect(Dtype::U32, &[r.len()])?;
                        // SAFETY: exact u32 output with private ranks.
                        unsafe { out.slice_mut::<u32>() }.copy_from_slice(r);
                    }
                }
            }
            _ => return Err(PyValueError::new_err("unknown component")),
        }
        Ok(())
    }
    fn cast_dictionary(&self, source: &str, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let s = dtype_from_name(source)?;
        let mut out = Buffer::get(out, true)?;
        with_dtype!(
            s,
            S,
            with_dtype!(out.dtype, T, {
                // SAFETY: numeric dtype dispatch matches validated output alignment.
                let out = unsafe { out.slice_mut::<T>() };
                match self.parts.values() {
                    core::Values::Dict(d) => d.cast_into::<S, T>(out),
                    core::Values::Modes(d) => d.cast_into::<S, T>(out),
                    _ => Ok(()),
                }
                .map_err(error)
            })
        )
    }
    /// Maps this row through `gene_map` (indexed by the row's id space):
    /// into `columns`/`values` (dropping unmapped genes, or one entry per
    /// position with `positions`), or scattered into a dense `values` row
    /// without `columns`. Dense-layout records keep only their non-zero
    /// values when mapped to columns.
    #[allow(clippy::too_many_arguments)]
    fn map_into(
        &self,
        py: Python<'_>,
        source: &str,
        gene_map: PyRef<'_, GeneMap>,
        columns: Option<&Bound<'_, PyAny>>,
        values: &Bound<'_, PyAny>,
        positions: Option<&Bound<'_, PyAny>>,
        dictionary: Option<&Bound<'_, PyAny>>,
        modes: Option<&Bound<'_, PyAny>>,
        zero: bool,
    ) -> PyResult<usize> {
        let source = dtype_from_name(source)?;
        let panel = &gene_map.map[..];
        if panel.len() != self._metadata.view().n_dims() as usize {
            return Err(PyValueError::new_err("gene map length differs from the chunk's gene count"));
        }
        let (mut columns, mut values, positions, dictionary, modes) = (
            columns.map(|a| Buffer::get(a, true)).transpose()?,
            Buffer::get(values, true)?,
            positions.map(|a| Buffer::get(a, false)).transpose()?,
            dictionary.map(|a| Buffer::get(a, false)).transpose()?,
            modes.map(|a| Buffer::get(a, false)).transpose()?,
        );
        if let Some(c) = &columns
            && (c.dtype != Dtype::U32 || c.shape.len() != 1)
        {
            return Err(PyValueError::new_err("columns require uint32 vector"));
        }
        if let Some(p) = &positions
            && (p.dtype != Dtype::U32 || p.shape.len() != 1)
        {
            return Err(PyValueError::new_err("positions require uint32 vector"));
        }
        for b in [&dictionary, &modes].into_iter().flatten() {
            if b.dtype != values.dtype || b.shape.len() != 1 {
                return Err(PyValueError::new_err("dictionary/modes dtype must equal output dtype"));
            }
        }
        let mut regions = vec![&values];
        regions.extend([columns.as_ref(), positions.as_ref(), dictionary.as_ref(), modes.as_ref()].into_iter().flatten());
        disjoint(&regions)?;
        let dense_layout = self.parts.layout() == RecordLayout::Dense;
        py.detach(|| {
            with_dtype!(
                source,
                S,
                with_dtype!(values.dtype, T, {
                    if dense_layout && let Some(cols) = &mut columns {
                        // SAFETY: matching numeric dtypes over retained, disjoint buffers.
                        let (cols, vals, positions) = unsafe { (cols.slice_mut::<u32>(), values.slice_mut::<T>(), positions.as_ref().map(|p| p.slice::<u32>())) };
                        return self.map_dense_layout::<S, T>(panel, positions, cols, vals);
                    }
                    let mut own_dict = if dictionary.is_none() { vec![T::default(); self.info().4] } else { Vec::new() };
                    let needs_modes = matches!(self.parts.values(), core::Values::Modes(_));
                    let mut own_modes = if needs_modes && modes.is_none() {
                        vec![T::default(); self._metadata.view().n_dims() as usize]
                    } else {
                        Vec::new()
                    };
                    if dictionary.is_none() {
                        match self.parts.values() {
                            core::Values::Dict(d) => d.cast_into::<S, T>(&mut own_dict).map_err(error)?,
                            core::Values::Modes(d) => d.cast_into::<S, T>(&mut own_dict).map_err(error)?,
                            _ => (),
                        }
                    }
                    if needs_modes
                        && modes.is_none()
                        && let Some(b) = self._metadata.view().column_modes()
                    {
                        core::cast_values::<S, T>(b, &mut own_modes).map_err(error)?;
                    }
                    // SAFETY: all numeric buffers have matching dtypes, stable owners and nonoverlapping regions.
                    unsafe {
                        let dict = dictionary.as_ref().map_or(&own_dict[..], |d| d.slice::<T>());
                        let modes = modes.as_ref().map_or(&own_modes[..], |m| m.slice::<T>());
                        if let Some(cols) = &mut columns {
                            if let Some(p) = &positions {
                                core::map_parts_at::<S, T>(&self.parts, p.slice(), panel, dict, modes, cols.slice_mut(), values.slice_mut()).map_err(error)?;
                                Ok(p.shape[0])
                            } else {
                                core::map_parts::<S, T>(&self.parts, panel, dict, modes, cols.slice_mut(), values.slice_mut()).map_err(error)
                            }
                        } else {
                            core::map_dense::<S, T>(&self.parts, panel, dict, modes, values.slice_mut(), fill(zero)).map_err(error)?;
                            Ok(values.shape.iter().product())
                        }
                    }
                })
            )
        })
    }
}

impl Parts {
    /// Columns and values of a dense-layout record, whose element `i` is
    /// gene `ids[i]`: every element at `positions`, aligned with them, or
    /// else the mapped non-zero elements in id order.
    fn map_dense_layout<S: core::CastInto<T>, T: core::Element>(&self, panel: &[u32], positions: Option<&[u32]>, cols: &mut [u32], vals: &mut [T]) -> PyResult<usize> {
        let ids = self.parts.ids();
        let mut raw = vec![0u8; ids.len() * S::WIDTH];
        self.parts.expand_values(&mut raw).map_err(error)?;
        let value = |i: usize| -> T { S::from_le(&raw[i * S::WIDTH..]).cast() };
        if let Some(positions) = positions {
            if cols.len() < positions.len() || vals.len() < positions.len() {
                return Err(PyValueError::new_err("mapped outputs are shorter than the positions"));
            }
            for (k, &p) in positions.iter().enumerate() {
                let p = p as usize;
                if p >= ids.len() {
                    return Err(PyValueError::new_err("position beyond the row"));
                }
                cols[k] = panel[ids[p] as usize];
                vals[k] = value(p);
            }
            return Ok(positions.len());
        }
        let mut n = 0;
        for (i, &id) in ids.iter().enumerate() {
            let column = panel[id as usize];
            if column == core::UNMAPPED || raw[i * S::WIDTH..(i + 1) * S::WIDTH].iter().all(|&b| b == 0) {
                continue;
            }
            if n >= cols.len().min(vals.len()) {
                return Err(PyValueError::new_err("mapped outputs are shorter than the row"));
            }
            cols[n] = column;
            vals[n] = value(i);
            n += 1;
        }
        Ok(n)
    }
}
fn rank_kind(r: core::Ranks<'_>) -> &'static str {
    match r {
        core::Ranks::One => "one",
        core::Ranks::U8(_) => "uint8",
        core::Ranks::U32(_) => "uint32",
    }
}

/// Converts `values` of dtype `source` into `out`'s dtype.
#[pyfunction]
fn scans_cast(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    source: &str,
    out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let source = dtype_from_name(source)?;
    let v = Buffer::get(values, false)?;
    let mut out = Buffer::get(out, true)?;
    disjoint(&[&v, &out])?;
    py.detach(|| {
        with_dtype!(
            source,
            S,
            with_dtype!(out.dtype, T, {
                // SAFETY: matching numeric output dispatch; immutable bytes are disjoint.
                unsafe { core::cast_values::<S, T>(v.slice(), out.slice_mut::<T>()) }.map_err(error)
            })
        )
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Parts>()?;
    m.add_function(wrap_pyfunction!(scans_cast, m)?)
}
