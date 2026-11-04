//! Prepared reads of any cell selection: contiguous ranges and gathers
//! (any order, repeats) share one path. Cells are grouped by chunk at
//! preparation; reads decode each selected chunk once into caller-owned
//! dense or CSR outputs of any element type, splitting the chunks over
//! scoped worker threads.

use super::panel::{PanelRows, RowScratch, compose_panel};
use super::{CpkFile, ReadOptions, ReadSpec, ReadSummary, RecordReadPlan};
use crate::format::{Dtype, OwnedMeta, record_nnz};
use crate::scans::{
    CastInto, DecodeTables, Decoder, Element, Fill, IdSpace, IndexOut, IndptrOut, Mapper, Rows,
    UNMAPPED, decode_tables, selection_nnz,
};
use crate::{Error, Result};
use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Cells to read: a contiguous range, or a gather in output order whose
/// ids may repeat.
#[derive(Clone, Debug)]
pub enum RowSelection<'a> {
    Range(Range<u64>),
    List(&'a [u64]),
}

/// A chunk's selected cells with their output rows.
enum ChunkRows {
    /// Cells `cells` land on consecutive output rows from `position`.
    Run { position: usize, cells: Range<usize> },
    /// `(output row, cell)` pairs in request order.
    Gather(Vec<(usize, usize)>),
}

impl ChunkRows {
    fn len(&self) -> usize {
        match self {
            Self::Run { cells, .. } => cells.len(),
            Self::Gather(rows) => rows.len(),
        }
    }

    /// The output row and chunk cell of the `i`-th record.
    #[inline]
    fn get(&self, i: usize) -> (usize, usize) {
        match self {
            Self::Run { position, cells } => (position + i, cells.start + i),
            Self::Gather(rows) => rows[i],
        }
    }

    fn cells(&self) -> Vec<usize> {
        match self {
            Self::Run { cells, .. } => cells.clone().collect(),
            Self::Gather(rows) => rows.iter().map(|&(_, cell)| cell).collect(),
        }
    }
}

struct Chunk {
    /// Absolute file offset of the chunk's records region.
    records: u64,
    reads: RecordReadPlan,
    metadata: OwnedMeta,
    tables: DecodeTables,
    rows: ChunkRows,
}

/// Metadata prepared once for repeated reads into independently owned
/// outputs. `spec().nnz` is the CSR capacity: the stored entries of the
/// selected cells, which dropped genes can reduce in the result.
pub struct ReadSelection {
    file: Arc<File>,
    chunks: Vec<Chunk>,
    /// Source column → output column; `None` reads every column in place.
    gene_map: Option<Box<[u32]>>,
    spec: ReadSpec,
}

/// Reusable private scratch: per-thread decode buffers by output element
/// type and the row bookkeeping of CSR reads. It never holds outputs; one
/// workspace serves one read at a time.
#[derive(Default)]
pub struct ReadWorkspace {
    threads: HashMap<TypeId, Box<dyn Any + Send>>,
    starts: Vec<usize>,
    counts: Vec<usize>,
}

impl ReadWorkspace {
    /// Releases retained scratch.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn scratch<T: Send + 'static>(&mut self, n: usize) -> &mut [RowScratch<T>] {
        let all = self
            .threads
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::<Vec<RowScratch<T>>>::default())
            .downcast_mut::<Vec<RowScratch<T>>>()
            .expect("scratch keyed by element type");
        if all.len() < n {
            all.resize_with(n, RowScratch::default);
        }
        &mut all[..n]
    }
}

impl CpkFile {
    /// Prepares a row range of every column in the stored dtype, with the
    /// default read policy.
    pub fn prepare_read(&self, rows: Range<u64>) -> Result<ReadSelection> {
        self.prepare_selection(RowSelection::Range(rows), None, self.n_cols() as usize, ReadOptions::default())
    }

    /// Prepares `rows` from metadata only; record bytes are read by each
    /// `*_into` call. `gene_map` sends each source column to an output
    /// column below `columns` (distinct targets) or drops it with
    /// [`UNMAPPED`]; without it, `columns` is the file's column count.
    pub fn prepare_selection(
        &self,
        rows: RowSelection<'_>,
        gene_map: Option<&[u32]>,
        columns: usize,
        options: ReadOptions,
    ) -> Result<ReadSelection> {
        let n_cols = self.n_cols() as usize;
        if columns > u32::MAX as usize {
            return Err(Error::InvalidInput("panel exceeds u32 range"));
        }
        if let Some(map) = gene_map {
            if map.len() != n_cols {
                return Err(Error::InvalidInput("gene map must cover every source column"));
            }
            let mut seen = vec![0u64; columns.div_ceil(64)];
            for &c in map.iter().filter(|&&c| c != UNMAPPED) {
                let c = c as usize;
                if c >= columns || seen[c / 64] & (1 << (c % 64)) != 0 {
                    return Err(Error::InvalidInput("mapped columns must be distinct and inside the panel"));
                }
                seen[c / 64] |= 1 << (c % 64);
            }
        } else if columns != n_cols {
            return Err(Error::InvalidInput("identity map requires source column count"));
        }
        let identity = gene_map.is_none_or(|m| columns == n_cols && m.iter().enumerate().all(|(i, &c)| c as usize == i));
        let groups = match rows {
            RowSelection::Range(range) => {
                let mut position = 0;
                self.spans(&range)?
                    .into_iter()
                    .map(|(chunk, cells)| {
                        let run = ChunkRows::Run { position, cells: cells.clone() };
                        position += cells.len();
                        (chunk, run)
                    })
                    .collect::<Vec<_>>()
            }
            RowSelection::List(list) => {
                let mut groups = BTreeMap::<usize, Vec<(usize, usize)>>::new();
                for (position, &row) in list.iter().enumerate() {
                    if row >= self.n_rows() {
                        return Err(Error::RowOutOfRange { row: row as usize, n_rows: self.n_rows() as usize });
                    }
                    let chunk = self.chunks().partition_point(|e| e.first_row <= row) - 1;
                    groups
                        .entry(chunk)
                        .or_default()
                        .push((position, (row - self.chunks()[chunk].first_row) as usize));
                }
                groups.into_iter().map(|(chunk, rows)| (chunk, ChunkRows::Gather(rows))).collect()
            }
        };
        let mut n_rows = 0usize;
        let mut nnz = 0usize;
        let mut chunks = Vec::with_capacity(groups.len());
        for (id, rows) in groups {
            let entry = self.chunks()[id];
            let metadata = OwnedMeta::from_vec(self.read_meta(id)?)?;
            let meta = metadata.view();
            if meta.chunk_len() != entry.len
                || meta.n_dims() != self.n_cols()
                || meta.value_width() != self.dtype().width()
                || meta.n_cells() != entry.n_rows as usize
            {
                return Err(Error::CorruptMeta("chunk differs from file directory"));
            }
            let stored = match &rows {
                ChunkRows::Run { cells, .. } => {
                    selection_nnz(&meta, Rows::Range { start: cells.start, end: cells.end })? as usize
                }
                ChunkRows::Gather(list) => {
                    let mut total = 0usize;
                    for &(_, cell) in list {
                        total += meta.cell_nnz(cell)?;
                    }
                    total
                }
            };
            nnz = nnz.checked_add(stored).ok_or(Error::InvalidInput("selection too large"))?;
            n_rows += rows.len();
            chunks.push(Chunk {
                records: self.byte_offset() + entry.offset + meta.records_start() as u64,
                reads: RecordReadPlan::new(&meta, &rows.cells(), options)?,
                tables: decode_tables(&meta)?,
                metadata,
                rows,
            });
        }
        let spec = ReadSpec { rows: n_rows, columns, nnz, dtype: self.dtype() };
        spec.dense_len()?;
        spec.value_bytes()?;
        if n_rows.checked_add(1).and_then(|n| n.checked_mul(8)).is_none_or(|n| n > isize::MAX as usize) {
            return Err(Error::InvalidInput("selection row offsets exceed address space"));
        }
        Ok(ReadSelection {
            file: self.file.clone(),
            chunks,
            gene_map: if identity { None } else { gene_map.map(Into::into) },
            spec,
        })
    }
}

/// A caller output shared by threads that write disjoint rows of it.
struct Shared<T>(*mut T, usize);

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}
// SAFETY: threads only form references to disjoint regions (see `slice`).
unsafe impl<T: Send> Send for Shared<T> {}
// SAFETY: as above; the pointer itself is never mutated.
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Shared<T> {
    fn new(s: &mut [T]) -> Self {
        Self(s.as_mut_ptr(), s.len())
    }

    /// # Safety
    /// No other live reference overlaps `start..start + len`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        assert!(start.checked_add(len).is_some_and(|end| end <= self.1), "output region outside the buffer");
        // SAFETY: in bounds of the caller's buffer; exclusivity per contract.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// Index outputs with the element type resolved once per read rather than
/// per instantiation of the decode loops.
#[derive(Clone, Copy)]
enum IndexSink {
    U16(Shared<u16>),
    U32(Shared<u32>),
    U64(Shared<u64>),
    I32(Shared<i32>),
    I64(Shared<i64>),
}

macro_rules! by_sink {
    ($sink:expr, $p:ident => $body:expr) => {
        match $sink {
            IndexSink::U16($p) => $body,
            IndexSink::U32($p) => $body,
            IndexSink::U64($p) => $body,
            IndexSink::I32($p) => $body,
            IndexSink::I64($p) => $body,
        }
    };
}

impl IndexSink {
    fn new<I: IndexOut + 'static>(s: &mut [I]) -> Self {
        let (p, n) = (s.as_mut_ptr().cast::<u8>(), s.len());
        let id = TypeId::of::<I>();
        if id == TypeId::of::<u16>() {
            Self::U16(Shared(p.cast(), n))
        } else if id == TypeId::of::<u32>() {
            Self::U32(Shared(p.cast(), n))
        } else if id == TypeId::of::<u64>() {
            Self::U64(Shared(p.cast(), n))
        } else if id == TypeId::of::<i32>() {
            Self::I32(Shared(p.cast(), n))
        } else {
            assert_eq!(id, TypeId::of::<i64>(), "IndexOut is sealed to u16, u32, u64, i32 and i64");
            Self::I64(Shared(p.cast(), n))
        }
    }

    /// # Safety
    /// As [`Shared::slice`] for `at..at + cols.len()`.
    unsafe fn put(&self, at: usize, cols: &[u32]) {
        by_sink!(self, p => {
            // SAFETY: exclusive region per contract.
            for (o, &c) in unsafe { p.slice(at, cols.len()) }.iter_mut().zip(cols) {
                *o = IndexOut::from_id(c);
            }
        })
    }

    /// Decodes a record's source-order CSR entries into `at..at + nnz`.
    ///
    /// # Safety
    /// As [`Shared::slice`] for `at..at + nnz`.
    unsafe fn decode(&self, decoder: &Decoder, record: &[u8], at: usize, nnz: usize, values: &mut [u8], ws: &mut [u64]) -> Result<usize> {
        // SAFETY: exclusive region per contract.
        by_sink!(self, p => decoder.sparse(record, IdSpace::Source, unsafe { p.slice(at, nnz) }, values, ws))
    }
}

/// The dtype an element type reads and writes.
fn dtype_of<T: Element>() -> Dtype {
    match (T::KIND, T::WIDTH) {
        (0, 1) => Dtype::U8,
        (0, 2) => Dtype::U16,
        (0, 4) => Dtype::U32,
        (0, 8) => Dtype::U64,
        (1, 1) => Dtype::I8,
        (1, 2) => Dtype::I16,
        (1, 4) => Dtype::I32,
        (1, 8) => Dtype::I64,
        (2, 4) => Dtype::F32,
        _ => Dtype::F64,
    }
}

/// Little-endian bytes of numeric elements (the only supported targets).
fn bytes_mut<T: Element>(s: &mut [T]) -> &mut [u8] {
    // SAFETY: element types are padding-free numerics, valid for any bytes.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast(), std::mem::size_of_val(s)) }
}

macro_rules! with_element {
    ($dtype:expr, $S:ident, $body:expr) => {
        match $dtype {
            Dtype::U8 => { type $S = u8; $body }
            Dtype::U16 => { type $S = u16; $body }
            Dtype::U32 => { type $S = u32; $body }
            Dtype::U64 => { type $S = u64; $body }
            Dtype::I8 => { type $S = i8; $body }
            Dtype::I16 => { type $S = i16; $body }
            Dtype::I32 => { type $S = i32; $body }
            Dtype::I64 => { type $S = i64; $body }
            Dtype::F32 => { type $S = f32; $body }
            Dtype::F64 => { type $S = f64; $body }
        }
    };
}

/// Element types a selection reads into, from any stored dtype.
pub trait OutputElement: Element + Default + Send + Sync + 'static {
    /// Calls `visit` with the element type of `dtype` as the source.
    #[doc(hidden)]
    fn visit<V: Visit<Self>>(dtype: Dtype, visit: V) -> V::Out;
}

/// A computation over a stored element type `S` converted into `T`.
#[doc(hidden)]
pub trait Visit<T: Element> {
    type Out;
    fn visit<S: CastInto<T>>(self) -> Self::Out;
}

macro_rules! outputs {
    ($($t:ty),*) => {$(
        impl OutputElement for $t {
            fn visit<V: Visit<Self>>(dtype: Dtype, visit: V) -> V::Out {
                with_element!(dtype, S, visit.visit::<S>())
            }
        }
    )*};
}
outputs!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

struct Dense<'a, T> {
    selection: &'a ReadSelection,
    out: Shared<T>,
    direct: bool,
    scratch: &'a mut [RowScratch<T>],
}

impl<T: OutputElement> Visit<T> for Dense<'_, T> {
    type Out = Result<()>;
    fn visit<S: CastInto<T>>(self) -> Result<()> {
        let mapper = Mapper::<S, T>::new()?;
        let Self { selection, out, direct, scratch } = self;
        selection.run(scratch, |chunk, s| selection.dense_chunk(chunk, out, direct, &mapper, s))
    }
}

struct Csr<'a, T> {
    selection: &'a ReadSelection,
    sink: IndexSink,
    out: Shared<T>,
    kept: Shared<usize>,
    starts: &'a [usize],
    direct: bool,
    scratch: &'a mut [RowScratch<T>],
}

impl<T: OutputElement> Visit<T> for Csr<'_, T> {
    type Out = Result<()>;
    fn visit<S: CastInto<T>>(self) -> Result<()> {
        let mapper = Mapper::<S, T>::new()?;
        let Self { selection, sink, out, kept, starts, direct, scratch } = self;
        selection.run(scratch, |chunk, s| selection.csr_chunk(chunk, sink, out, kept, starts, direct, &mapper, s))
    }
}

impl ReadSelection {
    /// Aggregates this selection's planned application read bytes and requests.
    /// These values exclude metadata reads and do not measure device traffic.
    pub fn io_summary(&self) -> ReadSummary {
        let mut summary = ReadSummary::default();
        for chunk in &self.chunks {
            summary.add(chunk.reads.summary());
        }
        summary
    }

    /// Output rows and columns, the CSR capacity and the stored dtype.
    pub fn spec(&self) -> ReadSpec {
        self.spec
    }

    /// Whether records decode straight into `T` in source column order.
    fn direct<T: Element>(&self) -> bool {
        self.gene_map.is_none() && dtype_of::<T>() == self.spec.dtype
    }

    /// Runs `task` on every chunk, on up to `scratch.len()` threads that
    /// take chunks in order; the first error stops the others.
    fn run<T: Send>(&self, scratch: &mut [RowScratch<T>], task: impl Fn(&Chunk, &mut RowScratch<T>) -> Result<()> + Sync) -> Result<()> {
        if let [only] = scratch {
            return self.chunks.iter().try_for_each(|chunk| task(chunk, only));
        }
        let (next, failed, task) = (&AtomicUsize::new(0), &AtomicBool::new(false), &task);
        std::thread::scope(|scope| {
            let workers: Vec<_> = scratch
                .iter_mut()
                .map(|s| {
                    scope.spawn(move || {
                        while !failed.load(Ordering::Relaxed) {
                            let Some(chunk) = self.chunks.get(next.fetch_add(1, Ordering::Relaxed)) else {
                                break;
                            };
                            if let Err(e) = task(chunk, s) {
                                failed.store(true, Ordering::Relaxed);
                                return Err(e);
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().unwrap_or(Err(Error::InvalidInput("read worker panicked"))))
                .collect::<Result<Vec<()>>>()
                .map(drop)
        })
    }

    fn threads(&self, workers: usize) -> usize {
        workers.clamp(1, self.chunks.len().max(1))
    }

    /// Writes `rows × columns` values of `T`, converting from the stored
    /// dtype. Dimensions are validated before any I/O; codec or I/O
    /// failures may leave the output partially written. Chunks are split
    /// over up to `workers` threads, each with its own workspace scratch.
    pub fn read_dense_into<T: OutputElement>(
        &self,
        out: &mut [T],
        ws: &mut ReadWorkspace,
        workers: usize,
    ) -> Result<()> {
        let need = self.spec.dense_len()?;
        if out.len() != need {
            return Err(Error::OutputTooSmall { need, have: out.len() });
        }
        let direct = self.direct::<T>();
        let out = Shared::new(out);
        let scratch = ws.scratch::<T>(self.threads(workers));
        T::visit(self.spec.dtype, Dense { selection: self, out, direct, scratch })
    }

    fn dense_chunk<S: CastInto<T>, T: Element + Default>(
        &self,
        chunk: &Chunk,
        out: Shared<T>,
        direct: bool,
        mapper: &Mapper<S, T>,
        s: &mut RowScratch<T>,
    ) -> Result<()> {
        let meta = chunk.metadata.view();
        let width = self.spec.columns;
        chunk.reads.read_into(&self.file, chunk.records, &mut s.records)?;
        let records = std::mem::take(&mut s.records);
        let mut panel = std::mem::take(&mut s.panel);
        let result = (|| {
            if direct {
                let decoder = Decoder::new(&meta, &chunk.tables)?;
                s.decode.resize(decoder.workspace_words(), 0);
                for i in 0..chunk.rows.len() {
                    let (position, cell) = chunk.rows.get(i);
                    let record = chunk.reads.record(&records, i)?;
                    if record_nnz(record)? != meta.cell_nnz(cell)? {
                        return Err(Error::CorruptRecord("record nnz differs from metadata"));
                    }
                    // SAFETY: every output row belongs to exactly one selected cell.
                    let row = unsafe { out.slice(position * width, width) };
                    decoder.dense(record, IdSpace::Source, bytes_mut(row), Fill::Zero, &mut s.decode)?;
                }
                return Ok(());
            }
            compose_panel(&chunk.tables, self.gene_map.as_deref(), meta.n_dims() as usize, &mut panel);
            let rows = PanelRows::new(&meta, &chunk.tables, &panel, mapper)?;
            rows.bind(s)?;
            for i in 0..chunk.rows.len() {
                let (position, cell) = chunk.rows.get(i);
                let record = chunk.reads.record(&records, i)?;
                // SAFETY: every output row belongs to exactly one selected cell.
                let row = unsafe { out.slice(position * width, width) };
                rows.dense(record, Some(meta.cell_nnz(cell)?), row, Fill::Zero, s)?;
            }
            Ok(())
        })();
        s.records = records;
        s.panel = panel;
        result
    }

    /// Writes canonical CSR (ascending output columns per row) into caller
    /// arrays: `indptr` of `rows + 1` entries, `indices` and `values` of at
    /// least [`ReadSpec::nnz`] entries. Rows are decoded at their capacity
    /// offsets, then compacted in place when genes were dropped; capacity
    /// past the returned nnz is unspecified. Dimensions and index widths
    /// are validated before any write. Returns the nnz.
    pub fn read_csr_into<P: IndptrOut, I: IndexOut + 'static, T: OutputElement>(
        &self,
        indptr: &mut [P],
        indices: &mut [I],
        values: &mut [T],
        ws: &mut ReadWorkspace,
        workers: usize,
    ) -> Result<usize> {
        let s = self.spec;
        let largest = s.columns.saturating_sub(1) as u64;
        if s.nnz as u64 > P::MAX || largest > I::MAX {
            return Err(Error::OutputTypeTooNarrow((s.nnz as u64).max(largest)));
        }
        if indptr.len() != s.rows + 1 || indices.len() < s.nnz || values.len() < s.nnz {
            return Err(Error::InvalidInput("CSR output capacity differs from selection"));
        }
        let mut starts = std::mem::take(&mut ws.starts);
        let mut counts = std::mem::take(&mut ws.counts);
        let result = self.csr(indices, values, ws, workers, &mut starts, &mut counts);
        ws.starts = starts;
        ws.counts = counts;
        let ReadWorkspace { starts, counts, .. } = ws;
        result?;
        indptr[0] = P::from_offset(0);
        let mut cursor = 0;
        for row in 0..s.rows {
            let (start, n) = (starts[row], counts[row]);
            if start != cursor {
                indices.copy_within(start..start + n, cursor);
                values.copy_within(start..start + n, cursor);
            }
            cursor += n;
            indptr[row + 1] = P::from_offset(cursor as u64);
        }
        Ok(cursor)
    }

    /// Decodes every row at its capacity offset (`starts`) and records the
    /// entries it kept (`counts`).
    fn csr<I: IndexOut + 'static, T: OutputElement>(
        &self,
        indices: &mut [I],
        values: &mut [T],
        ws: &mut ReadWorkspace,
        workers: usize,
        starts: &mut Vec<usize>,
        counts: &mut Vec<usize>,
    ) -> Result<()> {
        starts.clear();
        starts.resize(self.spec.rows + 1, 0);
        for chunk in &self.chunks {
            let meta = chunk.metadata.view();
            for i in 0..chunk.rows.len() {
                let (position, cell) = chunk.rows.get(i);
                starts[position + 1] = meta.cell_nnz(cell)?;
            }
        }
        for row in 0..self.spec.rows {
            starts[row + 1] += starts[row];
        }
        counts.clear();
        counts.resize(self.spec.rows, 0);
        let direct = self.direct::<T>();
        let scratch = ws.scratch::<T>(self.threads(workers));
        T::visit(self.spec.dtype, Csr {
            selection: self,
            sink: IndexSink::new(indices),
            out: Shared::new(values),
            kept: Shared::new(counts),
            starts: starts.as_slice(),
            direct,
            scratch,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn csr_chunk<S: CastInto<T>, T: Element + Default>(
        &self,
        chunk: &Chunk,
        sink: IndexSink,
        out: Shared<T>,
        kept: Shared<usize>,
        starts: &[usize],
        direct: bool,
        mapper: &Mapper<S, T>,
        s: &mut RowScratch<T>,
    ) -> Result<()> {
        let meta = chunk.metadata.view();
        chunk.reads.read_into(&self.file, chunk.records, &mut s.records)?;
        let records = std::mem::take(&mut s.records);
        let mut panel = std::mem::take(&mut s.panel);
        let result = (|| {
            let decoder;
            let rows;
            if direct {
                decoder = Some(Decoder::new(&meta, &chunk.tables)?);
                s.decode.resize(decoder.as_ref().map_or(0, Decoder::workspace_words), 0);
                rows = None;
            } else {
                decoder = None;
                compose_panel(&chunk.tables, self.gene_map.as_deref(), meta.n_dims() as usize, &mut panel);
                let panel_rows = PanelRows::new(&meta, &chunk.tables, &panel, mapper)?;
                panel_rows.bind(s)?;
                rows = Some(panel_rows);
            }
            for i in 0..chunk.rows.len() {
                let (position, _) = chunk.rows.get(i);
                let record = chunk.reads.record(&records, i)?;
                let (start, nnz) = (starts[position], starts[position + 1] - starts[position]);
                // SAFETY: each output row owns its capacity region and count.
                let (values, count) = unsafe { (out.slice(start, nnz), &mut kept.slice(position, 1)[0]) };
                *count = match (&decoder, &rows) {
                    (Some(decoder), _) => {
                        // SAFETY: the row's capacity region, as above.
                        let got = unsafe { sink.decode(decoder, record, start, nnz, bytes_mut(values), &mut s.decode)? };
                        if got != nnz {
                            return Err(Error::CorruptRecord("record nnz differs from metadata"));
                        }
                        nnz
                    }
                    (None, Some(rows)) => {
                        let cols = rows.csr(record, Some(nnz), values, s)?;
                        // SAFETY: the row's capacity region, as above.
                        unsafe { sink.put(start, cols) };
                        cols.len()
                    }
                    (None, None) => unreachable!("a chunk decodes directly or through its panel"),
                };
            }
            Ok(())
        })();
        s.records = records;
        s.panel = panel;
        result
    }
}
