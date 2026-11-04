//! Encoding: chunk models from a selection of CSR or dense rows, then one
//! record per row in the smaller of its sparse and dense layouts, as the
//! options allow.
//!
//! A chunk takes three passes over its rows:
//! 1. **Scan** ([`model::scan`]): validation, record shapes, gene presence
//!    and column-mode votes, which fix the gene order and the mode
//!    candidates.
//! 2. **Prepare** ([`model::build`]): every symbol and escape bit of each
//!    row's sparse record (see [`prep`]), kept in the workspace, and the
//!    counts of the expected codings, which become the models.
//! 3. **Code** ([`record`]): each row's kept symbols are rANS-coded and the
//!    smallest candidate record is written.
//!
//! A single record ([`encode_record`]) prepares and codes in one go.

mod input;
mod model;
mod prep;
mod record;
mod scratch;

use self::input::{Index, any_zero, with_width, zeros};
use self::model::{Model, prepare_dense, prepare_row};
use self::prep::{Arena, SparsePrep, sparse_bound};
use self::record::Coder;
use self::scratch::{RowLayout, RowWs};
use crate::error::{Error, Result};
use crate::format::{self, Header, Layout, Meta, id_width};
use crate::scans::tables::{self, EncodeTables};
use crate::scans::types::{EncodeOptions, LayoutChoice, MatrixRef, RowRef, Rows};

/// Encode workspace in `u64` words for records of at most `max_elems`
/// stored elements (see [`crate::format::Meta::max_elems`]).
pub fn encode_workspace_words(n_genes: u32, max_elems: usize) -> usize {
    RowLayout::new(n_genes, max_elems, 8, true).total()
}

/// Largest record a row of at most `nnz` stored elements can produce;
/// `dense` when the row may be stored with all `n_genes` elements.
pub fn record_bound(n_genes: u32, value_width: usize, nnz: usize, dense: bool) -> usize {
    if nnz == 0 {
        return 0;
    }
    let sparse = nnz * (id_width(n_genes) + value_width);
    let all = if dense {
        n_genes as usize * value_width
    } else {
        0
    };
    1 + format::varint_len(nnz as u64) + sparse.max(all)
}

fn check_matrix(src: &MatrixRef) -> Result<()> {
    if src.n_genes() == 0 || src.n_genes() > i32::MAX as u32 {
        return Err(Error::InvalidInput("n_genes must lie in 1..2^31"));
    }
    if !matches!(src.value_width(), 1 | 2 | 4 | 8) {
        return Err(Error::InvalidInput("value width must be 1, 2, 4 or 8"));
    }
    match src {
        MatrixRef::Csr(m) if m.indptr.is_empty() => {
            Err(Error::InvalidInput("indptr must hold n_rows + 1 entries"))
        }
        MatrixRef::CsrView(m) if m.indptr.is_empty() => {
            Err(Error::InvalidInput("indptr must hold n_rows + 1 entries"))
        }
        MatrixRef::Dense(m) if m.data.len() % (m.n_genes as usize * m.value_width) != 0 => Err(
            Error::InvalidInput("dense data is not a whole number of rows"),
        ),
        _ => Ok(()),
    }
}

/// The first violation of a cell's index contract: an index outside the
/// `u32` range, indices not strictly ascending, or one beyond `n_genes`.
fn check_cell_exact<I: Index>(idx: &[I], n_genes: u32) -> Result<()> {
    let mut prev: i64 = -1;
    for &i in idx {
        let s = i.check()? as i64;
        if s <= prev {
            return Err(Error::InvalidInput("indices must be strictly ascending"));
        }
        prev = s;
    }
    if prev >= n_genes as i64 {
        return Err(Error::InvalidInput("index beyond n_genes"));
    }
    Ok(())
}

fn check_row(row: &RowRef, n_genes: u32, vw: usize) -> Result<()> {
    match row {
        RowRef::Sparse(c) => {
            if c.values.len() != c.indices.len() * vw {
                return Err(Error::InvalidInput("values length differs from nnz * value width"));
            }
            input::with_indices!(c.indices, |idx| check_cell_exact(idx, n_genes))
        }
        RowRef::Dense(v) if v.len() != n_genes as usize * vw => Err(Error::InvalidInput(
            "dense row length differs from n_genes * value width",
        )),
        RowRef::Dense(_) => Ok(()),
    }
}

/// Whether a row of `nnz` non-zeros gets a dense candidate. Measured on
/// CELLxGENE counts and normalized matrices: below 60% non-zero, dense
/// records are larger and decode slower; from 90%, a dense candidate keeps
/// the chunk smaller than sparse records alone and decodes faster (lower
/// thresholds trade size for speed, the models fitting both layouts worse).
fn dense_candidate(choice: LayoutChoice, nnz: usize, n_genes: u32) -> bool {
    match choice {
        LayoutChoice::Sparse => false,
        LayoutChoice::Dense => true,
        LayoutChoice::Auto => 10 * nnz >= 9 * n_genes as usize,
    }
}

/// A dense candidate at most this much larger (in 1/64ths) than the sparse
/// record is kept: it decodes faster.
const DENSE_SLACK_64THS: usize = 1;

/// A row's declared non-zeros and whether it may be stored dense. A dense
/// record cannot keep a sparse row's explicitly stored zeros, so such a row
/// stays sparse unless the dense layout is forced, which drops them.
fn row_shape(row: &RowRef, vw: usize, n_genes: u32, choice: LayoutChoice) -> (usize, bool) {
    match row {
        RowRef::Sparse(c) => {
            let stored = c.indices.len();
            match choice {
                LayoutChoice::Dense => {
                    let z = with_width!(vw, |W| zeros::<W>(c.values));
                    (stored - z, stored > z)
                }
                _ => (
                    stored,
                    dense_candidate(choice, stored, n_genes) && !with_width!(vw, |W| any_zero::<W>(c.values)),
                ),
            }
        }
        RowRef::Dense(v) => {
            let nnz = v.len() / vw - with_width!(vw, |W| zeros::<W>(v));
            (nnz, nnz > 0 && dense_candidate(choice, nnz, n_genes))
        }
    }
}

/// The layout a row is expected to take, to count symbols before records
/// are coded: sparse pays roughly `log2(n / nnz) + 2` bits per index, dense
/// pays the entropy of zero versus non-zero over every gene.
fn expect_dense(choice: LayoutChoice, allowed: bool, nnz: usize, n_genes: u32) -> bool {
    if !allowed {
        return false;
    }
    if choice == LayoutChoice::Dense {
        return true;
    }
    let (n, k) = (n_genes as f64, nnz as f64);
    let p = k / n;
    let zeros = if p < 1.0 {
        -(p * p.log2() + (1.0 - p) * (1.0 - p).log2()) * n
    } else {
        0.0
    };
    zeros < k * ((n / k).log2() + 2.0)
}

/// Workspace of a chunk: row scratch for its largest record and room for
/// every prepared sparse record.
struct Needs {
    n_genes: u32,
    vw: usize,
    /// Whether records are prepared against column modes.
    modes: bool,
    /// Whether rows may be stored sparse.
    sparse: bool,
    /// Elements of the largest record, the row scratch's sizing bound.
    max_elems: usize,
    /// Bytes of every prepared sparse record, kept after the row scratch.
    prepared: usize,
}

impl Needs {
    fn new(n_genes: u32, vw: usize, opt: &EncodeOptions) -> Self {
        Needs {
            n_genes,
            vw,
            modes: opt.column_modes.is_some(),
            sparse: opt.layout != LayoutChoice::Dense,
            max_elems: 0,
            prepared: 0,
        }
    }

    /// Accounts for a row storing `stored` elements in its sparse record,
    /// `nnz` non-zeros, `allowed` dense, from a dense input when `dense_in`.
    fn add(&mut self, stored: usize, nnz: usize, allowed: bool, dense_in: bool) {
        let g = self.n_genes as usize;
        self.max_elems = self.max_elems.max(if allowed { g } else { stored }).max(dense_in as usize * g);
        if self.sparse && nnz > 0 {
            self.prepared += sparse_bound(stored, self.n_genes, self.vw, self.modes);
        }
    }

    /// Row scratch for the largest record of the chunk.
    fn layout(&self) -> RowLayout {
        RowLayout::new(self.n_genes, self.max_elems, self.vw, self.modes)
    }

    /// Row scratch plus the prepared records, in `u64` words.
    fn words(&self) -> usize {
        self.layout().total() + self.prepared.div_ceil(8)
    }
}

/// Output bytes and workspace words of a chunk of the selected rows.
pub(crate) fn chunk_needs(src: &MatrixRef, rows: Rows, opt: &EncodeOptions) -> Result<(usize, usize)> {
    check_matrix(src)?;
    let (n_genes, vw) = (src.n_genes(), src.value_width());
    let n = rows.count(src.n_rows())?;
    let modes_len = if opt.column_modes.is_some() { n_genes as usize * vw } else { 0 };
    let layout = Layout::new(n, n_genes, opt.renumber, modes_len).ok_or(Error::InvalidInput("chunk too large"))?;
    let mut total = layout.meta_len;
    let mut needs = Needs::new(n_genes, vw, opt);
    for i in 0..n {
        let row = src.row(rows.get(i))?;
        let (nnz, dense) = row_shape(&row, vw, n_genes, opt.layout);
        let (stored, dense_in) = match row {
            RowRef::Sparse(c) => (c.indices.len(), false),
            RowRef::Dense(_) => (nnz, true),
        };
        let bound = record_bound(n_genes, vw, stored.max(dense as usize), dense);
        total = total.checked_add(bound).ok_or(Error::InvalidInput("chunk too large"))?;
        needs.add(stored, nnz, dense, dense_in);
    }
    Ok((total, needs.words()))
}

/// Bytes a chunk of the selected rows can take at most.
pub fn chunk_bound<'a>(src: impl Into<MatrixRef<'a>>, rows: Rows, opt: &EncodeOptions) -> Result<usize> {
    Ok(chunk_needs(&src.into(), rows, opt)?.0)
}

/// Encode workspace in `u64` words for [`encode_chunk`] on the selected rows.
pub fn chunk_workspace_words<'a>(src: impl Into<MatrixRef<'a>>, rows: Rows, opt: &EncodeOptions) -> Result<usize> {
    Ok(chunk_needs(&src.into(), rows, opt)?.1)
}

/// Encodes the selected rows as one chunk into `out` (at least
/// [`chunk_bound`] bytes) with `ws` of [`chunk_workspace_words`] words.
/// Returns the chunk length.
pub fn encode_chunk<'a>(
    src: impl Into<MatrixRef<'a>>,
    rows: Rows,
    opt: &EncodeOptions,
    out: &mut [u8],
    ws: &mut [u64],
) -> Result<usize> {
    let src = src.into();
    opt.check()?;
    check_matrix(&src)?;
    let (n_genes, vw) = (src.n_genes(), src.value_width());
    let n = rows.count(src.n_rows())?;
    u32::try_from(n).map_err(|_| Error::InvalidInput("more than 2^32 - 1 cells"))?;
    let scan = model::scan(&src, rows, n, opt)?;
    let mut needs = Needs::new(n_genes, vw, opt);
    let dense_in = matches!(src, MatrixRef::Dense(_));
    for r in &scan.rows {
        needs.add(r.stored, r.nnz, r.allowed, dense_in);
    }
    let need = needs.words();
    if ws.len() < need {
        return Err(Error::WorkspaceTooSmall { need, have: ws.len() });
    }
    let kernels = crate::runtime::get()?.kernels();
    let (mut rw, rest) = needs.layout().split(ws, kernels)?;
    let rest = &mut rest[..needs.prepared.div_ceil(8)];
    // SAFETY: `u8` has no alignment requirement and every bit pattern is
    // valid for both types; the length covers the same memory.
    let mut cache = Arena::new(unsafe { std::slice::from_raw_parts_mut(rest.as_mut_ptr().cast(), rest.len() * 8) });
    let model = model::build(&src, rows, opt, scan, &mut rw, Some(&mut cache))?;
    encode_modeled(src, rows, opt, &model, cache.contents(), &mut rw, out)
}

/// A fixed, immutably borrowed input and its reusable statistical model.
/// Construction performs model building and sizing; repeated writes preserve
/// the same encoding decisions as encode_chunk.
pub struct PreparedEncoder<'a> {
    source: MatrixRef<'a>,
    rows: Rows<'a>,
    options: EncodeOptions,
    metadata: format::OwnedMeta,
    tables: EncodeTables,
    layout: Layout,
    sizes: super::ChunkSizes,
}

impl<'a> PreparedEncoder<'a> {
    /// Scans and prepares the selection once, fixing the gene order, column
    /// modes and models; fails on an invalid input or option.
    pub fn new(source: impl Into<MatrixRef<'a>>, rows: Rows<'a>, options: EncodeOptions) -> Result<Self> {
        let source = source.into();
        options.check()?;
        check_matrix(&source)?;
        let (n_genes, vw) = (source.n_genes(), source.value_width());
        let n = rows.count(source.n_rows())?;
        u32::try_from(n).map_err(|_| Error::InvalidInput("more than 2^32 - 1 cells"))?;
        let scan = model::scan(&source, rows, n, &options)?;
        let layout = RowLayout::new(n_genes, scan.max_elems, vw, options.column_modes.is_some());
        let mut scratch = vec![0u64; layout.total()];
        let (mut rw, _) = layout.split(&mut scratch, crate::runtime::get()?.kernels())?;
        let model = model::build(&source, rows, &options, scan, &mut rw, None)?;
        let sizes = super::chunk_sizes(source, rows, &options)?;
        let modes_len = if model.modes.is_some() { n_genes as usize * vw } else { 0 };
        let len = Layout::new(n, n_genes, options.renumber, modes_len)
            .ok_or(Error::InvalidInput("chunk too large"))?
            .meta_len;
        let mut bytes = vec![0; len];
        let (layout, _) = model_metadata(n_genes, vw, &options, &model, &mut bytes)?;
        let metadata = format::OwnedMeta::from_vec(bytes)?;
        let tables = tables::encode_tables(&metadata.view())?;
        Ok(Self { source, rows, options, metadata, tables, layout, sizes })
    }

    /// Model metadata before record placement; use it for independent record encoding.
    pub fn metadata(&self) -> Meta<'_> {
        self.metadata.view()
    }

    /// Metadata bytes of the chunk before any record is written; records
    /// appended with their extents and kinds complete the chunk.
    pub fn metadata_bytes(&self) -> &[u8] {
        self.metadata.bytes()
    }

    /// The tables matching [`Self::metadata`], for [`encode_record`].
    pub fn tables(&self) -> &EncodeTables {
        &self.tables
    }

    /// Output and workspace sizes of one [`Self::encode_into`] call.
    pub fn sizes(&self) -> super::ChunkSizes {
        self.sizes
    }

    /// Writes the chunk (metadata, then records, then the record extents and
    /// kinds in the header) into `out`, which must hold
    /// [`ChunkSizes::output_bytes`](crate::scans::ChunkSizes::output_bytes)
    /// bytes, using `workspace` of [`encode_workspace_words`] words.
    /// Returns the chunk length; repeated calls write the same bytes.
    pub fn encode_into(&self, out: &mut [u8], workspace: &mut [u64]) -> Result<usize> {
        let n = self.metadata.bytes().len();
        if out.len() < n {
            return Err(Error::OutputTooSmall { need: n, have: out.len() });
        }
        let (header, records) = out.split_at_mut(n);
        header.copy_from_slice(self.metadata.bytes());
        let meta = self.metadata.view();
        let mut cursor = 0usize;
        for i in 0..meta.n_cells() {
            let row = self.source.row(self.rows.get(i))?;
            let len = encode_record(&meta, &self.tables, row, &self.options, &mut records[cursor..], workspace)?;
            let len32 = u32::try_from(len).map_err(|_| Error::InvalidInput("record longer than 4 GiB"))?;
            let offset_at = self.layout.offsets_off + i * 8;
            let length_at = self.layout.lengths_off + i * 4;
            header[offset_at..offset_at + 8].copy_from_slice(&(cursor as u64).to_le_bytes());
            header[length_at..length_at + 4].copy_from_slice(&len32.to_le_bytes());
            header[self.layout.kinds_off + i] = if len == 0 { 0 } else { records[cursor] };
            cursor += len;
        }
        Header::rewrite_records(header, 0, cursor as u64);
        Ok(n + cursor)
    }
}

/// Writes the metadata section of `model` (header, frequency tables, gene
/// order, column modes in coded order and the non-zero prefix sums) into the
/// start of `out`; returns the layout used and the header written.
fn model_metadata(n_genes: u32, vw: usize, opt: &EncodeOptions, model: &Model, out: &mut [u8]) -> Result<(Layout, Header)> {
    let n = model.rows.len();
    let modes_len = if model.modes.is_some() { n_genes as usize * vw } else { 0 };
    let layout = Layout::new(n, n_genes, opt.renumber, modes_len).ok_or(Error::InvalidInput("chunk too large"))?;
    if out.len() < layout.meta_len {
        return Err(Error::OutputTooSmall {
            need: layout.meta_len,
            have: out.len(),
        });
    }
    let (meta_out, _records_out) = out.split_at_mut(layout.meta_len);
    meta_out.fill(0);
    let header = Header {
        renumbered: opt.renumber,
        modes: model.modes.is_some(),
        n_cells: n as u32,
        n_dims: n_genes,
        value_width: vw as u8,
        max_lanes: opt.max_lanes,
        lane_min_log2: opt.lane_min.trailing_zeros() as u8,
        page_log2: 0,
        max_k: model.max_k as u32,
        max_nnz: model.max_nnz as u32,
        max_elems: model.max_elems as u32,
        nnz_total: model.nnz().sum(),
        meta_len: layout.meta_len as u64,
        records_len: 0,
    };
    header.write(meta_out);
    format::write_models(meta_out, &model.freqs);
    if let Some(soc) = &model.source_of_coded {
        let w = id_width(n_genes);
        for (c, &s) in soc.iter().enumerate() {
            let at = layout.dims_off + w * c;
            meta_out[at..at + w].copy_from_slice(&s.to_le_bytes()[..w]);
        }
    }
    if let Some(modes) = &model.modes {
        // Stored in coded order, the order records keep their values in.
        for c in 0..n_genes as usize {
            let s = model.source_of_coded.as_ref().map_or(c, |soc| soc[c] as usize);
            let at = layout.modes_off + vw * c;
            meta_out[at..at + vw].copy_from_slice(&modes[s].to_le_bytes()[..vw]);
        }
    }
    let mut acc = 0u64;
    for (i, k) in model.nnz().enumerate() {
        acc += k;
        let at = layout.nnz_off + 8 * (i + 1);
        meta_out[at..at + 8].copy_from_slice(&acc.to_le_bytes());
    }
    Ok((layout, header))
}

/// Writes the chunk of a prepared `model` into `out`: its metadata, then one
/// record per row (`prepared` holds the cached sparse candidates), then the
/// record extents, kinds and length in the header. Returns the chunk length.
fn encode_modeled(
    src: MatrixRef<'_>,
    rows: Rows<'_>,
    opt: &EncodeOptions,
    model: &Model,
    prepared: &[u8],
    rw: &mut RowWs,
    out: &mut [u8],
) -> Result<usize> {
    let (n_genes, vw) = (src.n_genes(), src.value_width());
    let n = model.rows.len();
    let (layout, mut header) = model_metadata(n_genes, vw, opt, model, out)?;
    let (meta_out, records_out) = out.split_at_mut(layout.meta_len);
    let mut extents = Vec::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    let mut cursor = 0usize;
    {
        let meta = format::parse_meta(meta_out)?;
        let tables = tables::encode_tables(&meta)?;
        let coder = Coder { meta: &meta, tables: &tables, opt };
        for (i, info) in model.rows.iter().enumerate() {
            let len = if info.nnz == 0 {
                0
            } else {
                let row = src.row(rows.get(i))?;
                let cached = model.preps.get(i).map(|p| (p, prepared));
                encode_row(&coder, row, info.nnz, info.allowed, cached, rw, &mut records_out[cursor..])?
            };
            let len32 = u32::try_from(len).map_err(|_| Error::InvalidInput("record longer than 4 GiB"))?;
            extents.push((cursor as u64, len32));
            kinds.push(if len > 0 { records_out[cursor] } else { 0 });
            cursor += len;
        }
    }
    format::write_extents(meta_out, &layout, &extents);
    format::write_kinds(meta_out, &layout, &kinds);
    header.records_len = cursor as u64;
    header.write(meta_out);
    Ok(layout.meta_len + cursor)
}

/// The record of a validated row of `nnz > 0` non-zeros (`allowed` when it
/// may be dense): the smaller of its sparse and dense candidates as the
/// options allow. The sparse candidate is `cached` or prepared here.
fn encode_row(
    c: &Coder,
    row: RowRef,
    nnz: usize,
    allowed: bool,
    cached: Option<(&SparsePrep, &[u8])>,
    rw: &mut RowWs,
    out: &mut [u8],
) -> Result<usize> {
    let meta = c.meta;
    let (n_genes, vw) = (meta.n_dims(), meta.value_width());
    let layout = c.opt.layout;
    let try_sparse = layout != LayoutChoice::Dense;
    let try_dense = allowed && meta.max_elems() == n_genes as usize;
    if layout == LayoutChoice::Dense && !try_dense {
        return Err(Error::InvalidInput("dense record in a chunk without room for dense records"));
    }
    let stored = match row {
        RowRef::Sparse(cell) => cell.indices.len(),
        RowRef::Dense(_) => nnz,
    };
    let nnz = if try_sparse { stored } else { nnz };
    if try_sparse && nnz > meta.max_nnz() {
        return Err(Error::InvalidInput("row nnz exceeds the chunk maximum"));
    }
    let need = record_bound(n_genes, vw, nnz, try_dense);
    if out.len() < need {
        return Err(Error::OutputTooSmall { need, have: out.len() });
    }
    let cos = c.tables.coded_of_source();
    let modes = meta.column_modes();
    let mut len = None;
    if try_sparse {
        let (prep, bytes) = match cached {
            Some((p, bytes)) => (*p, bytes),
            None => {
                rw.arena.clear();
                let p = prepare_row(row, vw, n_genes, cos, modes, &mut rw.prep, &mut rw.arena);
                (p, rw.arena.contents())
            }
        };
        len = Some(c.write_sparse(&prep, bytes, row, &mut rw.prep, &mut rw.code, out));
    }
    if try_dense {
        let vp = prepare_dense(row, vw, n_genes, cos, modes, &mut rw.prep.values, rw.dense, &mut rw.dense_arena);
        let values = &rw.dense[..n_genes as usize * vw];
        let bytes = rw.dense_arena.contents();
        len = Some(match len {
            None => c.write_dense(nnz, &vp, bytes, values, &mut rw.code, out),
            Some(l) => {
                let alt = c.write_dense(nnz, &vp, bytes, values, &mut rw.code, rw.alt);
                if alt * 64 <= l * (64 + DENSE_SLACK_64THS) {
                    out[..alt].copy_from_slice(&rw.alt[..alt]);
                    alt
                } else {
                    l
                }
            }
        });
    }
    Ok(len.unwrap_or(0))
}

/// Encodes one row with a chunk's metadata and tables into `out` (at least
/// [`record_bound`] bytes). `ws` needs [`encode_workspace_words`] for the
/// chunk's [`Meta::max_elems`]. Returns the record length; a row without
/// non-zeros has an empty record. The row takes the smaller of its sparse
/// and dense layouts as `opt.layout` allows; a dense candidate needs
/// `max_elems == n_genes`. Dictionaries beyond [`Meta::max_k`] are not
/// used, so the record fits the chunk's decode workspace.
pub fn encode_record<'a>(
    meta: &Meta,
    tables: &EncodeTables,
    row: impl Into<RowRef<'a>>,
    opt: &EncodeOptions,
    out: &mut [u8],
    ws: &mut [u64],
) -> Result<usize> {
    tables.check(meta)?;
    opt.check()?;
    let row = row.into();
    let (n_genes, vw) = (meta.n_dims(), meta.value_width());
    check_row(&row, n_genes, vw)?;
    let layout = RowLayout::new(n_genes, meta.max_elems(), vw, meta.column_modes().is_some());
    let (mut rw, _) = layout.split(ws, tables.kernels)?;
    let (nnz, allowed) = row_shape(&row, vw, n_genes, opt.layout);
    if nnz == 0 {
        return Ok(0);
    }
    let coder = Coder { meta, tables, opt };
    encode_row(&coder, row, nnz, allowed, None, &mut rw, out)
}
