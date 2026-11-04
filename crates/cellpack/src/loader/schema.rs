//! Output schemas, and the loader's graph: one data source, the schemas
//! that output its cells, and the tables composed from both when a loader
//! is built. IDs and process identities name declared semantics; callback
//! addresses never enter plan identity.
//!
//! A schema names, for each of its output dims, the standard id it takes
//! (its side of the panel); each matrix names the standard id of each of
//! its columns (the data side). Composing the two gives, per chunk, the
//! output column of every coded gene, so a record maps with one lookup per
//! element; cached CSR rows hold standard ids and map through the schema's
//! standard-id table instead.
//!
//! Missing values. An output dim is present for a cell when its matrix has
//! a column with that standard id, and missing otherwise; a column no output
//! dim takes is dropped. A present dim the record stores no value for holds
//! 0 (CSR is built by dropping zeros); only missing dims take the schema's
//! default.

use std::sync::Arc;

use super::catalog::{CellId, DataSource, Location};
use super::convert::{DTYPES, RowOps, dtype_index, value_bits};
use super::error::config;
use super::memory::HugeBuf;
use super::process::{ProcessFn, RowFormat};
use super::{Dtype, Result};
use crate::scans::UNMAPPED;
use crate::scans::map::SparseMap;

/// A map keeps its sparse form ([`SparseMap`]) when at most this fraction
/// of its ids is mapped: past it, the full map's one lookup per element
/// beats a bit test plus a rank per kept element.
const SPARSE_MAPPED: (usize, usize) = (1, 4);

/// Whether `kept` mapped ids of `ids` are few enough for the sparse form.
fn sparse_enough(kept: usize, ids: usize) -> bool {
    kept * SPARSE_MAPPED.1 <= ids * SPARSE_MAPPED.0
}

/// Where [`SparseMap::append`] put one map in an arena.
#[derive(Clone, Copy)]
pub(crate) struct SparseRange {
    /// First `u32` of the map's bit/count word pairs in the arena.
    at: u32,
    /// Word pairs of the table (`append`'s first return value), 32 ids each.
    words: u32,
    /// Mapped ids in the map, i.e. `append`'s second return value and
    /// [`SparseMap::at`]'s `kept`.
    kept: u32,
    /// Whether columns are stored as `u32` (some column exceeds `u16::MAX`).
    wide: bool,
}

impl SparseRange {
    /// The map over the arena it was carved from.
    fn map<'a>(&self, arena: &'a [u32]) -> SparseMap<'a> {
        SparseMap::at(arena, self.at as usize, self.words as usize, self.kept as usize, self.wide)
    }
}

/// A caller-chosen schema id; ids must be distinct within a loader and enter
/// the plan fingerprint.
pub type SchemaId = u64;

/// Most schemas one loader outputs.
pub const MAX_SCHEMAS: usize = 64;

/// One output: its dims as standard ids, the element type and size of a cell,
/// and either the built-in dense row or a process function.
///
/// The schema owns nothing borrowed: `out_ids` is shared by `Arc`, and a
/// process function keeps its callbacks and `user_data` alive through its own
/// owner. Building the loader composes the schema's side of the panel with each
/// matrix's side, so changing a schema needs a new loader and, for plans, a new
/// `identity`.
pub struct OutputSchema {
    id: SchemaId,
    name: Option<String>,
    out_ids: Arc<[u32]>,
    dtype: Dtype,
    default: u64,
    cell_bytes: usize,
    process: Option<ProcessFn>,
    identity: u64,
    scratch_bytes: usize,
}

impl OutputSchema {
    /// The built-in output: each cell as a dense row of `out_ids.len()`
    /// values of `dtype` (CSR records are scattered into it).
    pub fn dense(id: SchemaId, out_ids: impl Into<Arc<[u32]>>, dtype: Dtype) -> Result<Self> {
        let out_ids = out_ids.into();
        let cell_bytes = out_ids.len().checked_mul(dtype.width()).ok_or_else(|| config("output size overflow"))?;
        Self::build(id, out_ids, dtype, cell_bytes, None, 0, 0)
    }

    /// A process function's output of `cell_bytes` per cell, with
    /// `scratch_bytes` of private scratch per call. `identity` is a stable
    /// caller-chosen version of both callbacks and every setting their owner
    /// retains; changing semantics requires changing it.
    pub fn process(id: SchemaId, out_ids: impl Into<Arc<[u32]>>, dtype: Dtype, cell_bytes: usize, process: ProcessFn, identity: u64, scratch_bytes: usize) -> Result<Self> {
        Self::build(id, out_ids.into(), dtype, cell_bytes, Some(process), identity, scratch_bytes)
    }

    fn build(id: SchemaId, out_ids: Arc<[u32]>, dtype: Dtype, cell_bytes: usize, process: Option<ProcessFn>, identity: u64, scratch_bytes: usize) -> Result<Self> {
        if out_ids.is_empty() || out_ids.len() >= u32::MAX as usize {
            return Err(config("a schema needs between 1 and 2^32 - 1 output dims"));
        }
        if cell_bytes == 0 || cell_bytes > u32::MAX as usize || scratch_bytes > isize::MAX as usize {
            return Err(config("cell bytes must lie in 1..2^32 and scratch must be addressable"));
        }
        let mut seen = std::collections::HashSet::new();
        if out_ids.iter().any(|&s| s == UNMAPPED || !seen.insert(s)) {
            return Err(config("output dims name distinct standard ids"));
        }
        Ok(Self { id, name: None, out_ids, dtype, default: 0, cell_bytes, process, identity, scratch_bytes })
    }

    /// Sets the default value of missing output dims in dense rows: dims
    /// whose standard id the cell's matrix has no column for. Present dims
    /// the record stores no value for are 0, whatever the default. Zero
    /// unless set; converted to the schema's dtype with `as`.
    pub fn with_default(mut self, value: f64) -> Self {
        self.default = value_bits(self.dtype, value);
        self
    }

    /// Sets the exact little-endian value, including integers above 2^53.
    pub fn with_default_bytes(mut self, value: &[u8]) -> Result<Self> {
        if value.len() != self.dtype.width() {
            return Err(config("default value width differs from the schema dtype"));
        }
        let mut bytes = [0u8; 8];
        bytes[..value.len()].copy_from_slice(value);
        self.default = u64::from_le_bytes(bytes);
        Ok(self)
    }

    /// Attaches a label, for reports and debugging only: names never enter a
    /// fingerprint and are not resolved against anything.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// The schema's id.
    pub fn id(&self) -> SchemaId {
        self.id
    }
    /// The label from [`Self::named`], if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    /// The standard id of every output dim, in output-column order.
    pub fn out_ids(&self) -> &[u32] {
        &self.out_ids
    }
    /// Output dims, i.e. the length of a built-in dense row.
    pub fn out_dim(&self) -> usize {
        self.out_ids.len()
    }
    /// The element type of the output.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Little-endian bytes of the default value in a word.
    pub fn default_bits(&self) -> u64 {
        self.default
    }
    /// Bytes of one cell's output.
    pub fn cell_bytes(&self) -> usize {
        self.cell_bytes
    }
    /// Scratch bytes a process call of this schema may use, per call.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
    /// The process function, or none for the built-in dense row.
    pub fn process_fn(&self) -> Option<&ProcessFn> {
        self.process.as_ref()
    }
    /// Whether the loader writes this schema's output itself.
    pub fn is_builtin(&self) -> bool {
        self.process.is_none()
    }
    /// Whether the schema's CSR callback accepts rows behind `resolve`.
    pub(crate) fn partial_rows(&self) -> bool {
        self.process.as_ref().is_some_and(|p| p.partial_rows())
    }
    /// The row format the schema's process function receives (the built-in
    /// output writes dense rows).
    pub fn row_format(&self) -> RowFormat {
        self.process.as_ref().map_or(RowFormat::Dense, |p| p.row_format())
    }
    /// The identity a plan binds to: every declared semantic of the schema,
    /// including the caller's process `identity` but never a callback address.
    pub(crate) fn fingerprint(&self) -> u64 {
        let mut h = mix(mix(self.id, self.dtype.code() as u64), self.default);
        h = mix(mix(h, self.cell_bytes as u64), self.scratch_bytes as u64);
        h = mix(mix(h, self.process.is_some() as u64), self.identity);
        h = mix(h, self.partial_rows() as u64);
        h = mix(h, self.row_format().code() as u64);
        for &s in self.out_ids.iter() {
            h = mix(h, s as u64);
        }
        h
    }
}

/// Folds one identity value into a fingerprint. The catalog mixes with a
/// different rotation, so a graph fingerprint is not a renaming of a source
/// fingerprint over the same values.
pub(crate) fn mix(h: u64, n: u64) -> u64 {
    (h ^ n).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(31)
}

/// One input file as the graph needs it: the descriptor, the global id of its
/// first page, its length and the identity recorded at registration.
pub(crate) struct PhysicalFile {
    /// The same descriptor the workers read through.
    pub file: Arc<std::fs::File>,
    /// Global id of its first page.
    pub first_page: u64,
    /// Length in bytes, as recorded at registration.
    pub len: u64,
    /// Device, inode, length, mtime and mtime nanoseconds at registration;
    /// `prepare` rechecks it to reject a changed input.
    pub identity: (u64, u64, u64, i64, i64),
}

/// A range of an arena.
#[derive(Clone, Copy, Default)]
pub(crate) struct Range32 {
    /// First entry of the arena.
    pub at: u32,
    /// Entries in the range.
    pub len: u32,
}

/// Per schema: what its accesses need of each chunk, composed at build.
pub(crate) struct SchemaTables {
    /// Standard id -> output column ([`UNMAPPED`]: not output).
    pub std_to_out: Box<[u32]>,
    /// Standard id `s` maps to output column `s` for every id.
    pub std_identity: bool,
    /// The output panel (shared by schemas with equal output dims).
    pub panel: usize,
    /// Kernels by source dtype ([`dtype_index`]); present for every dtype
    /// of a chunk the schema can output.
    pub ops: Vec<Option<RowOps>>,
    /// Per chunk: its column modes converted to the schema's dtype, in
    /// `mode_words` (empty without modes).
    pub modes: Vec<Range32>,
    /// Converted modes of every chunk, in 8-byte words.
    pub mode_words: Vec<u64>,
    /// The sparse form of `std_to_out`, for narrow outputs (arena, range).
    pub std_sparse: Option<(Vec<u32>, SparseRange)>,
}

impl SchemaTables {
    /// The narrow-output form of `std_to_out`, when the panel keeps one.
    #[inline]
    pub fn std_sparse(&self) -> Option<SparseMap<'_>> {
        self.std_sparse.as_ref().map(|(a, r)| r.map(a))
    }
}

/// Per distinct output panel: each chunk's coded gene -> output column, or
/// none for chunks with no gene in the panel (every output dim of their
/// cells is missing).
///
/// Schemas with equal `out_ids` share one panel, so a chunk's map is composed
/// and stored once per distinct output rather than once per schema. `identity`
/// marks a chunk whose coded gene order already is panel order, which lets a
/// mapped row be delivered without a gather.
pub(crate) struct Panel {
    /// Per chunk: the span of its map in the arena, or none when no gene of it
    /// is in the panel.
    pub chunks: Vec<Option<Range32>>,
    /// Every coded gene of the chunk has an output column.
    pub total: Vec<bool>,
    /// Per chunk: it has an output column and coded gene `g` maps to output
    /// column `g` for every gene.
    pub identity: Vec<bool>,
    /// Coded genes of the chunk with an output column: its cells' present
    /// output dims.
    pub present: Vec<u32>,
    /// The maps, in huge pages: workers gather from them at random, across
    /// many chunks.
    arena: HugeBuf,
    /// The arena's length in `u32` entries.
    pub len: usize,
    /// Per chunk: the sparse form of its map, when narrow enough.
    sparse: Vec<Option<SparseRange>>,
    sparse_arena: HugeBuf,
    sparse_len: usize,
}

impl Panel {
    /// The chunk's coded gene -> output column map, or none when the chunk has
    /// no gene in the panel.
    #[inline]
    pub fn map(&self, chunk: u32) -> Option<&[u32]> {
        debug_assert!((chunk as usize) < self.chunks.len());
        // SAFETY: chunk ids index the panel's chunk table; each range lies in
        // the arena it was carved from.
        unsafe { self.chunks.get_unchecked(chunk as usize).map(|r| self.arena.u32s(self.len).get_unchecked(r.at as usize..(r.at + r.len) as usize)) }
    }

    /// The chunk's map in sparse form, if it keeps one.
    #[inline]
    pub fn sparse(&self, chunk: u32) -> Option<SparseMap<'_>> {
        debug_assert!((chunk as usize) < self.sparse.len());
        // SAFETY: chunk ids index the panel's chunk table.
        unsafe { self.sparse.get_unchecked(chunk as usize) }.map(|r| r.map(self.sparse_arena.u32s(self.sparse_len)))
    }
}

/// A loader's immutable tables: the source, its schemas, and everything
/// composed from both at build time.
///
/// `tables[k]` is schema `k`'s conversion view, `panels` are the distinct
/// output panels those tables share, and `fingerprint` names the whole
/// arrangement (inputs, schemas, slot size) for plans. `files` mirrors the
/// source's files with their registration identity, which `Loader::prepare`
/// rechecks.
pub(crate) struct Graph {
    /// The inputs; the graph borrows nothing from them beyond `Arc`s.
    pub source: Arc<DataSource>,
    /// Output schemas, in loader order.
    pub schemas: Vec<OutputSchema>,
    /// Per schema: its maps and kernels.
    pub tables: Vec<SchemaTables>,
    /// Distinct output panels, shared by schemas with equal `out_ids`.
    pub panels: Vec<Panel>,
    /// Per chunk: some schema's dense callback can take the chunk's native
    /// image unchanged.
    borrow_dense: Vec<bool>,
    /// Kernels keeping each stored type (rows admitted to the cache), by
    /// [`dtype_index`].
    pub admit_ops: Vec<Option<RowOps>>,
    /// The inputs, by file id.
    pub files: Vec<PhysicalFile>,
    /// Pages of all inputs, in the global page numbering.
    pub pages: u64,
    /// Page size of every input, in bytes.
    pub page_size: usize,
    /// Positions per output slot.
    pub slot_cells: u64,
    /// Identity of the whole arrangement, which a plan binds to.
    pub fingerprint: u64,
}

impl Graph {
    /// Composes the tables: validates the schemas against the source, records
    /// the inputs' identities, builds one panel per distinct output, resolves
    /// each schema's kernels per stored type and fingerprints the result.
    pub fn new(source: Arc<DataSource>, schemas: Vec<OutputSchema>, slot_cells: u64) -> Result<Self> {
        use std::os::unix::fs::MetadataExt;
        if schemas.is_empty() || schemas.len() > MAX_SCHEMAS {
            return Err(config(format!("a loader outputs 1 to {MAX_SCHEMAS} schemas")));
        }
        if slot_cells == 0 || slot_cells > u32::MAX as u64 {
            return Err(config("slots hold 1 to 2^32 - 1 positions"));
        }
        let mut seen = std::collections::HashSet::new();
        for s in &schemas {
            if !seen.insert(s.id()) {
                return Err(config("duplicate output schema id"));
            }
            if s.out_ids.iter().any(|&x| x as usize >= source.vocab) {
                return Err(config("output dims name standard ids outside the vocabulary"));
            }
            if s.is_builtin() && s.cell_bytes != s.out_dim() * s.dtype.width() {
                return Err(config("the built-in output is a dense row of the schema's dtype"));
            }
        }
        let mut files = Vec::new();
        let mut fingerprint = mix(0x5343_4845_4d41_0003, source.mapping_fingerprint());
        for f in &source.files {
            let m = f.file.metadata()?;
            let key = (m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec());
            if key != f.identity {
                return Err(config("input file changed after data source construction"));
            }
            for n in [key.0, key.1, key.2, key.3 as u64, key.4 as u64] {
                fingerprint = mix(fingerprint, n);
            }
            files.push(PhysicalFile { file: f.file.clone(), first_page: f.page_base, len: f.len, identity: key });
        }
        fingerprint = mix(fingerprint, slot_cells);
        for s in &schemas {
            fingerprint = mix(fingerprint, s.fingerprint());
        }
        let mut panels: Vec<Panel> = Vec::new();
        let mut panel_ids: Vec<Arc<[u32]>> = Vec::new();
        let mut tables = Vec::with_capacity(schemas.len());
        for s in &schemas {
            let mut std_to_out = vec![UNMAPPED; source.vocab].into_boxed_slice();
            for (j, &id) in s.out_ids.iter().enumerate() {
                std_to_out[id as usize] = j as u32;
            }
            let panel = match panel_ids.iter().position(|p| **p == *s.out_ids) {
                Some(p) => p,
                None => {
                    panels.push(compose(&source, &std_to_out)?);
                    panel_ids.push(s.out_ids.clone());
                    panels.len() - 1
                }
            };
            let mut ops: Vec<Option<RowOps>> = DTYPES.iter().map(|_| None).collect();
            let mut modes = vec![Range32::default(); source.chunks.len()];
            let mut mode_words = Vec::new();
            for (c, chunk) in source.chunks.iter().enumerate() {
                if panels[panel].chunks[c].is_none() {
                    continue;
                }
                let k = dtype_index(chunk.dtype);
                if ops[k].is_none() {
                    ops[k] = Some(RowOps::new(chunk.dtype, s.dtype).map_err(|e| config(format!("schema {}: {e}", s.id)))?);
                }
                if let Some(span) = chunk.modes {
                    let bytes = source.modes(span);
                    let n = bytes.len() / chunk.dtype.width();
                    let at = mode_words.len();
                    mode_words.resize(at + (n * s.dtype.width()).div_ceil(8), 0u64);
                    (ops[k].as_ref().unwrap().values)(bytes, mode_words[at..].as_mut_ptr().cast(), n)?;
                    modes[c] = Range32 { at: at as u32, len: (mode_words.len() - at) as u32 };
                }
            }
            let std_identity = std_to_out.iter().enumerate().all(|(i, &c)| c as usize == i);
            let kept = std_to_out.iter().filter(|&&c| c != UNMAPPED).count();
            let std_sparse = sparse_enough(kept, std_to_out.len()).then(|| {
                let mut arena = Vec::new();
                let (words, kept, wide) = SparseMap::append(&std_to_out, &mut arena);
                (arena, SparseRange { at: 0, words: words as u32, kept: kept as u32, wide })
            });
            tables.push(SchemaTables { std_to_out, std_identity, panel, ops, modes, mode_words, std_sparse });
        }
        let mut admit_ops: Vec<Option<RowOps>> = DTYPES.iter().map(|_| None).collect();
        for chunk in &source.chunks {
            let k = dtype_index(chunk.dtype);
            if admit_ops[k].is_none() {
                admit_ops[k] = Some(RowOps::new(chunk.dtype, chunk.dtype)?);
            }
        }
        let borrow_dense = source.chunks.iter().enumerate().map(|(chunk, c)| {
            schemas.iter().enumerate().any(|(k, s)| {
                s.process_fn().is_some() && s.row_format() == RowFormat::Dense && s.dtype() == c.dtype
                    && s.out_dim() == c.meta().n_dims() as usize && panels[tables[k].panel].identity[chunk]
            })
        }).collect();
        Ok(Self { pages: source.pages, page_size: source.page_size, source, schemas, tables, panels, borrow_dense, admit_ops, files, slot_cells, fingerprint })
    }

    /// Accesses per slot: every schema of every position.
    #[inline]
    pub fn lanes_per_slot(&self) -> u64 {
        self.slot_cells * self.schemas.len() as u64
    }

    /// Where `cell` lives: its location, physical file and first page.
    pub fn locate(&self, position: u64, cell: CellId) -> Result<(Location<'_>, usize, u64)> {
        let loc = self.source.locate(position, cell)?;
        Ok((loc, loc.file.id as usize, loc.first_page))
    }

    /// Whether schema `k` receives a record of `chunk` as a dense row (else
    /// as a CSR row). `dense_record` is the record's stored layout, `encoding`
    /// its value coding and `whole` a cached image (whose values are raw).
    /// The runtime and the planner's pricing both follow this.
    #[inline]
    pub fn dense_delivery(&self, k: usize, chunk: u32, dense_record: bool, encoding: u8, whole: bool) -> bool {
        match self.schemas[k].row_format() {
            RowFormat::Dense => true,
            RowFormat::Csr => false,
            RowFormat::Auto => {
                if !dense_record { return false; }
                let schema = &self.schemas[k];
                let panel = &self.panels[self.tables[k].panel];
                if panel.present[chunk as usize] == 0 { return false; }
                let source = &self.source.chunks[chunk as usize];
                let genes = source.meta().n_dims() as usize;
                // Borrowing a native cached row avoids both mapping and compaction.
                if whole && source.dtype == schema.dtype() && panel.identity[chunk as usize] && genes == schema.out_dim() {
                    return true;
                }
                auto_dense(source.dtype, schema.dtype(), genes, schema.out_dim(), if whole { crate::format::VAL_RAW } else { encoding })
            }
        }
    }

    /// One image serves every schema; preserve an explicit dense callback's
    /// native borrow before choosing the smaller allocated representation.
    pub fn cached_dense(&self, chunk: u32, dense_record: bool, nnz: usize) -> bool {
        let c = &self.source.chunks[chunk as usize];
        dense_record && (self.borrow_dense[chunk as usize]
            || super::rows::keep_dense(c.meta().n_dims() as usize, nnz, c.dtype.width()))
    }

    /// A schema's converted column modes of a chunk.
    #[inline]
    pub fn modes(&self, schema: usize, chunk: u32) -> &[u64] {
        debug_assert!(schema < self.tables.len());
        // SAFETY: schema and chunk ids index the tables built for every
        // schema and chunk; each range lies in its schema's mode words.
        unsafe {
            let t = self.tables.get_unchecked(schema);
            let r = *t.modes.get_unchecked(chunk as usize);
            t.mode_words.get_unchecked(r.at as usize..(r.at + r.len) as usize)
        }
    }
}

/// Each chunk's coded gene -> output column through its standard ids.
fn compose(source: &DataSource, std_to_out: &[u32]) -> Result<Panel> {
    let mut arena = Vec::new();
    let mut chunks = Vec::with_capacity(source.chunks.len());
    let mut total = Vec::with_capacity(source.chunks.len());
    let mut identity = Vec::with_capacity(source.chunks.len());
    let mut present = Vec::with_capacity(source.chunks.len());
    let mut sparse = Vec::with_capacity(source.chunks.len());
    let mut sparse_arena = Vec::new();
    for chunk in &source.chunks {
        let at = arena.len();
        let mut any = false;
        let mut all = true;
        let mut count = 0u32;
        for &s in source.columns(chunk.columns) {
            let out = if s == UNMAPPED { UNMAPPED } else { std_to_out[s as usize] };
            any |= out != UNMAPPED;
            all &= out != UNMAPPED;
            count += (out != UNMAPPED) as u32;
            arena.push(out);
        }
        present.push(count);
        sparse.push((any && sparse_enough(count as usize, arena.len() - at)).then(|| {
            let start = sparse_arena.len();
            let (words, kept, wide) = SparseMap::append(&arena[at..], &mut sparse_arena);
            SparseRange { at: start as u32, words: words as u32, kept: kept as u32, wide }
        }));
        identity.push(any && arena[at..].iter().enumerate().all(|(i, &c)| c as usize == i));
        if any {
            chunks.push(Some(Range32 { at: at as u32, len: (arena.len() - at) as u32 }));
        } else {
            arena.truncate(at);
            chunks.push(None);
        }
        total.push(any && all);
    }
    Ok(Panel { chunks, total, identity, present, arena: HugeBuf::from_u32s(&arena)?, len: arena.len(), sparse, sparse_arena: HugeBuf::from_u32s(&sparse_arena)?, sparse_len: sparse_arena.len() })
}

/// Whether a dense record is delivered as a dense row rather than compacted
/// to CSR, decided by a quantized crossover of the two costs. Scanning a dense
/// record costs its source width even when only a few columns are mapped.
/// Coded values must first expand before CSR compaction, while dense scatter
/// can consume their dictionary or column modes directly.
fn auto_dense(source: Dtype, target: Dtype, genes: usize, out_dim: usize, encoding: u8) -> bool {
    use super::convert::Conversion;
    // Ratios are in halves of the source dimension; integer comparisons keep
    // this decision independent of rounding and floating-point state.
    let mut halves = match Conversion::of(source, target) {
        Conversion::Same if matches!(source, Dtype::F32 | Dtype::F64) => 1,
        Conversion::IntToF32 if matches!(source, Dtype::U8 | Dtype::I8) => 1,
        Conversion::IntToF32 | Conversion::IntToF64 => 4,
        _ => 2,
    };
    halves = match encoding {
        crate::format::VAL_MODE => halves.max(12),
        crate::format::VAL_DICT => halves.max(2),
        _ => halves,
    };
    2 * out_dim <= halves * genes
}
