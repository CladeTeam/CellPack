//! The input side: every registered matrix as chunks of metadata plus one map
//! from its stored columns to standard ids, and the global cell numbering
//! that concatenates them.
//!
//! Cells are numbered in registration order: a matrix owns the next run of
//! global ids and a chunk a contiguous run of them, so a cell's
//! chunk is one partition search over `chunk_start`. Per-chunk
//! coded-gene -> standard-id maps live in one huge-page arena and column
//! modes in 8-byte words: a worker gathers from them once per cell, across
//! many chunks.
//!
//! The source holds no names and infers nothing: the caller states each
//! matrix's map, and one matrix may not name a standard id twice.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{OpenOptionsExt,MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::store::CpkFile;
use crate::format::{Dtype, Meta};
use crate::scans::{DecodeTables, UNMAPPED, decode_tables};

use crate::loader::error::{Error, Result, config};
use crate::loader::memory::HugeBuf;

/// Global cell identity: matrices occupy consecutive ranges in registration order.
pub type CellId = u32;

/// One opened input file, its global page numbering and the identity it had
/// when it was registered.
pub(crate) struct FileSource {
    /// The descriptor, opened with the builder's `O_DIRECT` flag; workers read
    /// through it.
    pub file: Arc<File>,
    /// Position in [`DataSource::files`]: the file id that plan entries and
    /// the workers' descriptor tables name.
    pub id: u32,
    /// Length in bytes, which bounds a read's `need`.
    pub len: u64,
    /// Global id of the file's first page; every file is paged.
    pub page_base: u64,
    /// Device, inode, length, mtime and mtime nanoseconds at registration;
    /// `prepare` rechecks it to reject a changed input.
    pub identity: (u64,u64,u64,i64,i64),
}

/// One chunk of a matrix: its metadata, decode tables and the maps composed
/// from the caller's column map, all resolved once at registration.
pub(crate) struct Chunk {
    metadata: crate::format::OwnedMeta,
    /// Decode tables of the chunk's record format, built once.
    pub tables: DecodeTables,
    /// Coded gene id -> standard id ([`UNMAPPED`] where the matrix's
    /// column has none), composed once with the chunk's gene order; a span
    /// of the column arena.
    pub columns: Span,
    /// The chunk's column modes in its stored type, by coded id: a span of
    /// the source's mode words, in bytes.
    pub modes: Option<Span>,
    /// File id of the input holding the chunk.
    pub file: u32,
    /// File offset of the chunk's records area; for a registered region it
    /// includes the region's offset.
    pub offset: u64,
    /// Element type the chunk stores.
    pub dtype: Dtype,
}

impl Chunk {
    /// Shortens the internal borrow to the catalog reference's lifetime.
    pub fn meta(&self) -> Meta<'_> {
        self.metadata.view()
    }
}

/// A range of the catalog's column arena or of its mode bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Span {
    at: usize,
    len: usize,
}

/// Every input matrix, numbered by global cell id, with each matrix's map
/// from its columns to the standard ids (the data side of the panel).
/// Shared by planner and workers.
pub struct DataSource {
    pub(crate) fingerprint: u64,
    pub(crate) chunks: Vec<Chunk>,
    /// First cell of each chunk, followed by the total cell count.
    chunk_start: Vec<u64>,
    /// First cell, cell count and element type of each matrix.
    matrices: Vec<(u64, u64, Dtype)>,
    pub(crate) files: Vec<FileSource>,
    pub(crate) page_size: usize,
    /// Standard ids are `0..vocab`.
    pub(crate) vocab: usize,
    /// Most genes of one chunk (the length of a dense record).
    pub(crate) max_genes: usize,
    /// Pages of all files together, in the global page numbering.
    pub(crate) pages: u64,
    /// Most stored elements of one record.
    pub(crate) max_nnz: usize,
    /// Most pages one record spans.
    pub(crate) max_pages: usize,
    /// Cells by the pages their record spans, capped at 64.
    page_histogram: [u64; 65],
    /// Largest decode workspace of a chunk, in words.
    pub(crate) workspace_words: usize,
    /// Every column map, in one huge-page arena: maps are gathered at random
    /// for every cell, across many chunks.
    columns: HugeBuf,
    column_len: usize,
    /// Column modes, 8-byte aligned.
    modes: Vec<u64>,
}

/// Registers matrices one at a time, then composes them into a [`DataSource`].
///
/// Registration accumulates each chunk's column map and modes in plain
/// vectors; [`Self::build`] moves the maps into a huge-page arena and the
/// modes into words, and nothing is shared before then, so the builder is
/// `&mut self` throughout. Files are opened once per canonical path, so
/// registering one `.cpk` twice reuses its descriptor and page numbering.
pub struct DataSourceBuilder {
    catalog: DataSource,
    columns: Vec<u32>,
    modes: Vec<u8>,
    paths: HashMap<PathBuf, u32>,
    direct: bool,
}

/// Where one cell's bytes live: a run of whole pages of one file.
#[derive(Clone, Copy)]
pub(crate) struct Location<'a> {
    /// The chunk the cell belongs to.
    pub chunk: &'a Chunk,
    /// Index of the chunk in the catalog.
    pub chunk_index: u32,
    /// The cell's true non-zeros, from the chunk metadata.
    pub nnz: usize,
    /// The record is stored dense (every gene an element).
    pub dense: bool,
    /// Value coding from the metadata; cached images contain raw values.
    pub value_encoding: u8,
    /// The input holding the chunk.
    pub file: &'a FileSource,
    /// Global id of the first page.
    pub first_page: u64,
    /// File offset of the first page.
    pub page_offset: u64,
    /// Pages the record spans, [`Self::start_in_page`] and [`Self::len`]
    /// included; 0 for a record with no bytes.
    pub page_count: usize,
    /// Offset of the record's first byte within the first page.
    pub start_in_page: usize,
    /// Bytes of the record: what it occupies inside the page run.
    pub len: usize,
}

/// Folds one identity value into a fingerprint (a 64-bit mix, so that
/// neighbouring inputs land far apart).
fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29)
}

impl DataSourceBuilder {
    /// `vocab`: the number of standard ids the matrices map their columns to
    /// (ids are `0..vocab`), at most `u32::MAX - 1`; `page_size`: the page
    /// size of every file registered later, a power of two in
    /// `[512, 65536]`.
    pub fn new(vocab: usize, page_size: usize) -> Result<Self> {
        if vocab == 0
            || vocab >= u32::MAX as usize
            || !page_size.is_power_of_two()
            || !(512..=65536).contains(&page_size)
        {
            return Err(config(
                "a positive vocabulary below 2^32 and a power-of-two page size in [512, 65536] required",
            ));
        }
        Ok(Self {
            catalog: DataSource {
                fingerprint: mix(page_size as u64, 0x005c_10ad),
                chunks: Vec::new(),
                chunk_start: vec![0],
                matrices: Vec::new(),
                files: Vec::new(),
                page_size,
                vocab,
                max_genes: 0,
                pages: 0,
                max_nnz: 0,
                max_pages: 0,
                page_histogram: [0; 65],
                workspace_words: 0,
                columns: HugeBuf::zeroed(0)?,
                column_len: 0,
                modes: Vec::new(),
            },
            columns: Vec::new(),
            modes: Vec::new(),
            paths: HashMap::new(),
            direct: false,
        })
    }

    /// Data reads use O_DIRECT; metadata reads remain buffered.
    pub fn direct_io(mut self, enabled: bool) -> Self {
        self.direct = enabled;
        self
    }

    /// Opens `path` once (by canonical path), records its identity and
    /// extends the global page numbering. A file registered again by another
    /// matrix or region keeps its descriptor and page base, so identity
    /// checks and fixed-file tables see it once.
    fn file(&mut self, path: &Path) -> Result<u32> {
        let path = path.canonicalize()?;
        if let Some(&id) = self.paths.get(&path) {
            return Ok(id);
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(if self.direct { libc::O_DIRECT } else { 0 })
            .open(&path)?;
        let metadata = file.metadata()?;
        let len = metadata.len();
        let identity = (metadata.dev(),metadata.ino(),len,metadata.mtime(),metadata.mtime_nsec());
        let id = u32::try_from(self.catalog.files.len()).map_err(|_| config("too many files"))?;
        let c = &mut self.catalog;
        c.fingerprint = mix(c.fingerprint, len);
        c.files.push(FileSource {
            file: Arc::new(file),
            id,
            len,
            page_base: c.pages,
            identity,
        });
        c.pages = c
            .pages
            .checked_add(len.div_ceil(c.page_size as u64))
            .ok_or_else(|| config("page count overflow"))?;
        self.paths.insert(path, id);
        Ok(id)
    }

    /// One matrix's map must name one column per stored column, every mapped
    /// id inside the vocabulary and distinct; [`UNMAPPED`] drops a column and
    /// may repeat.
    fn check_map(&self, map: &[u32], n: usize) -> Result<()> {
        if map.len() != n {
            return Err(config("a standard id map must cover every column of its matrix"));
        }
        let mut seen = HashSet::new();
        for &c in map.iter().filter(|&&c| c != UNMAPPED) {
            if c as usize >= self.catalog.vocab || !seen.insert(c) {
                return Err(config(
                    "standard ids of one matrix must be distinct and inside the vocabulary",
                ));
            }
        }
        Ok(())
    }

    /// Ends the chunk being registered: its `rows` cells extend the global
    /// numbering. `chunk_start` therefore has one entry per chunk plus the
    /// total, which the last entry holds.
    fn push(&mut self, rows: u64) -> Result<()> {
        let c = &mut self.catalog;
        let end = c.chunk_start.last().unwrap() + rows;
        if end > CellId::MAX as u64 {
            return Err(config("cell count exceeds the CellId range"));
        }
        c.fingerprint = mix(c.fingerprint, rows);
        c.chunk_start.push(end);
        Ok(())
    }

    /// Registers a `.cpk` file as one matrix. `to_std` is the matrix's
    /// standard id of every stored column ([`UNMAPPED`]: none), given by the
    /// caller for this matrix: the source neither infers nor aligns columns.
    /// The file's page size must be the data source's.
    pub fn add_cpk(&mut self, path: impl AsRef<Path>, to_std: &[u32]) -> Result<u32> {
        self.add_cpk_region(path, 0, None, to_std)
    }

    /// Registers an embedded CellPack stream without extracting its bytes:
    /// the file image lies at `offset` of `path` and is `length` bytes long
    /// (`None`: to the end of the file). `map` is the matrix's standard id
    /// per stored column, as for [`Self::add_cpk`]. The file's superblock and
    /// every chunk must use the source's page size; each chunk's metadata is
    /// parsed and its per-cell page counts recorded for the sizing estimates.
    pub fn add_cpk_region(&mut self, path: impl AsRef<Path>, offset: u64,
        length: Option<u64>, map: &[u32]) -> Result<u32> {
        let path = path.as_ref();
        let cpk = CpkFile::open_region(path, offset, length)?;
        let sb = *cpk.superblock();
        let page = self.catalog.page_size;
        if sb.page_size as usize != page {
            return Err(config("file page size differs from the data source's"));
        }
        self.check_map(map, sb.n_cols as usize)?;
        let first = *self.catalog.chunk_start.last().unwrap();
        let file = self.file(path)?;
        for (i, entry) in cpk.chunks().iter().enumerate() {
            let metadata = crate::format::OwnedMeta::from_vec(cpk.read_meta(i)?)?;
            let meta = metadata.view();
            if meta.page_size() != Some(page as u32) || meta.value_width() != sb.dtype.width() || meta.n_dims() != sb.n_cols {
                return Err(config("chunk page size, value width or columns differ from its file"));
            }
            if meta.chunk_len() != entry.len || meta.n_cells() != entry.n_rows as usize {
                return Err(crate::Error::CorruptMeta("chunk differs from file directory").into());
            }
            let tables = decode_tables(&meta)?;
            let at = self.columns.len();
            match tables.source_of_coded() {
                Some(soc) => self.columns.extend(soc.iter().map(|&s| map[s as usize])),
                None => self.columns.extend_from_slice(map),
            }
            let columns = Span { at, len: self.columns.len() - at };
            let modes = meta.column_modes().map(|bytes| {
                let at = self.modes.len().next_multiple_of(8);
                self.modes.resize(at, 0);
                self.modes.extend_from_slice(bytes);
                Span { at, len: bytes.len() }
            });
            let chunk_offset = offset + entry.offset;
            let records = usize::try_from(chunk_offset).map_err(|_| config("chunk offset exceeds address space"))?
                .checked_add(meta.records_start()).ok_or_else(|| config("chunk offset overflow"))?;
            let mut max_pages = 0;
            let mut histogram = [0u64; 65];
            // Workers read these per-cell tables without checks, so every
            // entry is validated once here.
            for row in 0..meta.n_cells() {
                meta.cell_nnz(row)?;
                let r = meta.record_range(row)?;
                let mut pages = 0;
                if !r.is_empty() {
                    let start = (records + r.start) % page;
                    pages = (start + r.len()).div_ceil(page);
                    max_pages = max_pages.max(pages);
                }
                histogram[pages.min(64)] += 1;
            }
            let c = &mut self.catalog;
            for (total, count) in c.page_histogram.iter_mut().zip(histogram) {
                *total += count;
            }
            c.max_pages = c.max_pages.max(max_pages);
            c.max_nnz = c.max_nnz.max(meta.max_elems());
            c.max_genes = c.max_genes.max(meta.n_dims() as usize);
            c.workspace_words = c.workspace_words.max(crate::scans::decode_workspace_words(&meta));
            c.fingerprint = mix(mix(c.fingerprint, chunk_offset), meta.meta_len() as u64);
            let rows = meta.n_cells() as u64;
            c.chunks.push(Chunk { metadata, tables, columns, modes, file, offset: chunk_offset, dtype: sb.dtype });
            self.push(rows)?;
        }
        self.matrix(first, sb.dtype)
    }

    /// Records the matrix the chunks registered since the last matrix form:
    /// `first` is the global id of its first cell, and the element type
    /// applies to all of them.
    fn matrix(&mut self, first: u64, dtype: Dtype) -> Result<u32> {
        let id =
            u32::try_from(self.catalog.matrices.len()).map_err(|_| config("too many matrices"))?;
        let end = *self.catalog.chunk_start.last().unwrap();
        self.catalog.matrices.push((first, end - first, dtype));
        Ok(id)
    }

    /// Copies the accumulated column maps into a huge-page arena and the
    /// column modes into 8-byte words, then returns the source. Metadata and
    /// decode tables stay owned by their chunks.
    pub fn build(mut self) -> Result<DataSource> {
        let arena = HugeBuf::zeroed(self.columns.len() * 4)?;
        // SAFETY: the arena holds at least this many u32 and is exclusively ours.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.columns.as_ptr(),
                arena.ptr().cast(),
                self.columns.len(),
            )
        };
        self.catalog.columns = arena;
        self.catalog.column_len = self.columns.len();
        self.catalog.modes = self.modes.chunks(8).map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        }).collect();
        Ok(self.catalog)
    }
}

impl DataSource {
    /// Record-page p99.9, capped at 64; oversized tasks have a separate reserve.
    pub(crate) fn normal_pages(&self) -> usize {
        let target = self.n_cells() - self.n_cells() / 1000;
        let mut count = 0;
        for (pages, &n) in self.page_histogram.iter().enumerate() {
            count += n;
            if count >= target {
                return pages.max(1);
            }
        }
        64
    }

    /// Bytes of metadata and maps this source retains for its whole lifetime:
    /// chunk metadata, decode tables, the column arena, the mode words and
    /// the chunk and matrix tables. Counted by `memory_estimate`, not by RSS.
    pub fn resident_metadata_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.metadata.bytes().len() + c.tables.allocated_bytes()).sum::<usize>()
            + self.column_len * 4 + self.modes.len() * 8 + self.chunk_start.len() * 8 + self.matrices.len() * 16
    }

    /// A chunk's coded gene -> standard id map, the span it occupies in the
    /// column arena.
    pub(crate) fn columns(&self, span: Span) -> &[u32] {
        &self.columns.u32s(self.column_len)[span.at..span.at + span.len]
    }
    /// A chunk's column modes: little-endian values of its dtype, starting
    /// on an 8-byte boundary.
    pub(crate) fn modes(&self, span: Span) -> &[u8] {
        // SAFETY: the words are initialized plain data covering the span.
        let bytes = unsafe { std::slice::from_raw_parts(self.modes.as_ptr().cast::<u8>(), self.modes.len() * 8) };
        &bytes[span.at..span.at + span.len]
    }
    /// The same modes as whole words (the tail of the last word is zero).
    pub(crate) fn mode_words(&self, span: Span) -> &[u64] {
        &self.modes[span.at / 8..(span.at + span.len).div_ceil(8)]
    }

    /// Total cells of every matrix: the size of the global cell id space.
    pub fn n_cells(&self) -> u64 {
        *self.chunk_start.last().unwrap()
    }
    /// Pages of all files together, in the global page numbering.
    pub fn n_pages(&self) -> u64 {
        self.pages
    }
    /// Size of the standard id space: ids are `0..vocab`.
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    /// Element types of the matrices, in registration order.
    pub fn matrix_dtypes(&self) -> Vec<Dtype> {
        self.matrices.iter().map(|m| m.2).collect()
    }
    /// Matrices registered, each a contiguous range of cell ids.
    pub fn n_matrices(&self) -> usize {
        self.matrices.len()
    }
    /// Page size shared by every file of the source, in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }
    /// Largest number of pages one cell spans.
    pub fn max_cell_pages(&self) -> usize {
        self.max_pages
    }
    /// Stable identity of the page layout and cell numbering. Offline plans are
    /// bound to it; output column maps do not participate.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
    /// Identity including the standard id maps and the chunk metadata.
    pub fn mapping_fingerprint(&self) -> u64 {
        let mut h = mix(self.fingerprint, self.vocab as u64);
        for &value in self.columns.u32s(self.column_len) { h = mix(h, value as u64); }
        for chunk in &self.chunks {for &byte in chunk.metadata.bytes().iter(){h=mix(h,byte as u64);}}
        h
    }
    /// The first cell id of a matrix and its row count.
    pub fn matrix_cells(&self, matrix: u32) -> Option<(CellId, u64)> {
        self.matrices
            .get(matrix as usize)
            .map(|&(first, rows, _)| (first as CellId, rows))
    }

    /// The page run holding one cell: file descriptor (opened with the
    /// catalog's I/O flags), offset of the first page, and page count. Meant
    /// for I/O tools measuring the device on the loader's exact reads.
    pub fn cell_pages(&self, cell: CellId) -> Result<(std::os::fd::RawFd, u64, usize)> {
        use std::os::fd::AsRawFd;
        let loc = self.locate(0, cell)?;
        Ok((loc.file.file.as_raw_fd(), loc.page_offset, loc.page_count))
    }

    /// The exact byte range of one cell: file descriptor, offset and length.
    pub fn cell_bytes(&self, cell: CellId) -> Result<(std::os::fd::RawFd, u64, usize)> {
        use std::os::fd::AsRawFd;
        let loc = self.locate(0, cell)?;
        Ok((
            loc.file.file.as_raw_fd(),
            loc.page_offset + loc.start_in_page as u64,
            loc.len,
        ))
    }

    /// Where one cell's record lies: its chunk, the whole pages covering it,
    /// the byte offset and length inside them, and whether the record is
    /// stored dense. `position` names the access only for the error.
    ///
    /// The pages start at the page-aligned offset below the record, so a
    /// record of `len` bytes at `start_in_page` spans
    /// `ceil((start_in_page + len) / page_size)` pages (0 when the record is
    /// empty), and `page_count` is what the plan counts and the worker reads.
    pub(crate) fn locate(&self, position: u64, cell: CellId) -> Result<Location<'_>> {
        let cell64 = cell as u64;
        if cell64 >= self.n_cells() {
            return Err(Error::Access {
                position,
                cell: cell64,
            });
        }
        // `chunk_start` begins at 0 and holds every chunk's first cell, so the
        // chunk holding the cell is the last start at or below it.
        let s = self.chunk_start.partition_point(|&v| v <= cell64) - 1;
        debug_assert!(s < self.chunks.len());
        // SAFETY: `s` indexes the chunks (and their starts); the row lies in
        // its chunk; chunk file ids index the files. Chunk metadata, its
        // record extents and its non-zero table were validated for every cell
        // when the source was built.
        let (chunk, meta, (r, nnz, kind), f) = unsafe {
            let chunk = self.chunks.get_unchecked(s);
            let meta = chunk.meta();
            let row = (cell64 - *self.chunk_start.get_unchecked(s)) as usize;
            (chunk, meta, meta.cell_unchecked(row), self.files.get_unchecked(chunk.file as usize))
        };
        let page = self.page_size as u64;
        let (offset, len) = (chunk.offset + (meta.records_start() + r.start) as u64, r.len());
        let start_in_page = (offset % page) as usize;
        Ok(Location {
            chunk,
            chunk_index: s as u32,
            nnz,
            dense: kind & 3 == crate::format::IDX_DENSE,
            value_encoding: (kind >> 2) & 3,
            file: f,
            first_page: f.page_base + offset / page,
            page_offset: offset - start_in_page as u64,
            page_count: if len == 0 {
                0
            } else {
                (start_in_page + len).div_ceil(self.page_size)
            },
            start_in_page,
            len,
        })
    }
}
