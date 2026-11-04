//! Rows of one chunk decoded into a caller's column panel and element type.
//!
//! File selections and resident-chunk decoders share this path: the chunk's
//! coded gene order is composed with the caller's source → panel map once
//! per chunk, and each record is decoded in coded order and scattered
//! straight into the output (dense) or compacted into ascending panel
//! columns (CSR).

use crate::format::{Meta, RecordLayout};
use crate::scans::{CastInto, DecodeTables, Decoder, Element, Fill, IdSpace, Mapper, UNMAPPED, Values, cast_values};
use crate::{Error, Result};

/// Private per-thread scratch for [`PanelRows`] and file selections. It
/// grows on demand, is reusable across chunks and never holds outputs.
pub struct RowScratch<T> {
    pub(crate) records: Vec<u8>,
    pub(crate) panel: Vec<u32>,
    pub(crate) decode: Vec<u64>,
    dict: Vec<T>,
    modes: Vec<T>,
    values: Vec<T>,
    raw: Vec<u8>,
    pairs: Vec<(u32, u32)>,
    cols: Vec<u32>,
}

impl<T> Default for RowScratch<T> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            panel: Vec::new(),
            decode: Vec::new(),
            dict: Vec::new(),
            modes: Vec::new(),
            values: Vec::new(),
            raw: Vec::new(),
            pairs: Vec::new(),
            cols: Vec::new(),
        }
    }
}

/// Composes a chunk's coded gene order with a source → panel map (identity
/// when `None`) into `out`, indexed by coded id.
pub fn compose_panel(tables: &DecodeTables, gene_map: Option<&[u32]>, n_genes: usize, out: &mut Vec<u32>) {
    out.clear();
    out.extend((0..n_genes).map(|coded| {
        let source = tables.source_of_coded().map_or(coded, |m| m[coded] as usize);
        gene_map.map_or(source as u32, |m| m[source])
    }));
}

/// Decodes records of one chunk into panel rows of `T`, reading stored
/// values as `S` (the chunk's value width).
pub struct PanelRows<'a, 'm, S, T> {
    meta: &'a Meta<'m>,
    tables: &'a DecodeTables,
    decoder: Decoder<'a, 'm>,
    panel: &'a [u32],
    mapper: &'a Mapper<S, T>,
}

impl<'a, 'm, S: CastInto<T>, T: Element + Default> PanelRows<'a, 'm, S, T> {
    /// `panel` maps every coded id of the chunk to an output column or
    /// [`UNMAPPED`] (see [`compose_panel`]).
    pub fn new(meta: &'a Meta<'m>, tables: &'a DecodeTables, panel: &'a [u32], mapper: &'a Mapper<S, T>) -> Result<Self> {
        if S::WIDTH != meta.value_width() {
            return Err(Error::InvalidInput("source element type does not match the value width"));
        }
        if panel.len() != meta.n_dims() as usize {
            return Err(Error::InvalidInput("panel map must cover every gene of the chunk"));
        }
        Ok(Self { meta, tables, decoder: Decoder::new(meta, tables)?, panel, mapper })
    }

    /// Sizes `scratch` for this chunk and converts its column modes once.
    pub fn bind(&self, scratch: &mut RowScratch<T>) -> Result<()> {
        scratch.decode.resize(self.decoder.workspace_words(), 0);
        scratch.dict.resize(self.meta.max_k(), T::default());
        scratch.modes.clear();
        if let Some(bytes) = self.meta.column_modes() {
            scratch.modes.resize(self.meta.n_dims() as usize, T::default());
            cast_values::<S, T>(bytes, &mut scratch.modes)?;
        }
        Ok(())
    }

    /// Scatters one record into a dense panel row; `nnz`, when given, is
    /// the cell's count from the chunk metadata and is checked.
    pub fn dense(&self, record: &[u8], nnz: Option<usize>, out: &mut [T], fill: Fill, scratch: &mut RowScratch<T>) -> Result<()> {
        let RowScratch { decode, dict, modes, .. } = scratch;
        let parts = self.decoder.parts(record, IdSpace::Coded, decode)?;
        if nnz.is_some_and(|n| n != parts.nnz()) {
            return Err(Error::CorruptRecord("record nnz differs from metadata"));
        }
        match parts.values() {
            Values::Dict(d) => d.cast_into::<S, T>(dict)?,
            Values::Modes(d) => d.cast_into::<S, T>(dict)?,
            Values::Raw(_) => (),
        }
        self.mapper.map_dense(&parts, self.panel, dict, modes, out, fill)
    }

    /// Writes one record's mapped elements in ascending panel columns:
    /// values to the front of `values` (which holds the record's nnz) and
    /// the columns to the returned slice. Unmapped genes and the zeros of a
    /// dense-layout record are dropped; stored sparse zeros are kept.
    pub fn csr<'s>(&self, record: &[u8], nnz: Option<usize>, values: &mut [T], scratch: &'s mut RowScratch<T>) -> Result<&'s [u32]> {
        let RowScratch { decode, values: cast, raw, pairs, cols, .. } = scratch;
        let parts = self.decoder.parts(record, IdSpace::Coded, decode)?;
        if nnz.is_some_and(|n| n != parts.nnz()) {
            return Err(Error::CorruptRecord("record nnz differs from metadata"));
        }
        let elements = parts.elements();
        raw.resize(elements * S::WIDTH, 0);
        parts.expand_values_with(self.tables, raw)?;
        cast.resize(elements, T::default());
        cast_values::<S, T>(raw, cast)?;
        let dense = parts.layout() == RecordLayout::Dense;
        pairs.clear();
        for (i, &id) in parts.ids().iter().enumerate() {
            let column = self.panel[id as usize];
            let zero = dense && raw[i * S::WIDTH..(i + 1) * S::WIDTH].iter().all(|&b| b == 0);
            if column != UNMAPPED && !zero {
                pairs.push((column, i as u32));
            }
        }
        if values.len() < pairs.len() {
            return Err(Error::OutputTooSmall { need: pairs.len(), have: values.len() });
        }
        pairs.sort_unstable_by_key(|&(column, _)| column);
        cols.clear();
        for (j, &(column, i)) in pairs.iter().enumerate() {
            cols.push(column);
            values[j] = cast[i as usize];
        }
        Ok(cols)
    }
}
