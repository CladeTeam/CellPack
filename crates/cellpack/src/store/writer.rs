//! Streaming CSR and dense writing with reusable private encoding scratch.

use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::error::{Error, Result};
use crate::format::cpk::{self, ChunkEntry, Dtype, SUPERBLOCK_LEN, SourceLayout, Superblock};
use crate::format::parse_meta;
use crate::scans::{
    CsrRef, DenseRef, EncodeOptions, Indices, MatrixRef, Rows, chunk_sizes, encode_chunk, pack_chunk,
    packed_len,
};

/// How a file is chunked and coded.
#[derive(Clone, Copy, Debug)]
pub struct CpkOptions {
    /// Per-chunk column ordering, layout candidates, value coding and rANS settings.
    pub encode: EncodeOptions,
    /// Rows per chunk; each chunk has its own models and column order.
    pub chunk_rows: usize,
    /// Records are packed so each reads its minimum number of these pages.
    pub page_size: u32,
}

impl CpkOptions {
    /// 4096-row chunks packed to 4 KiB pages.
    pub fn new(encode: EncodeOptions) -> Self {
        Self {
            encode,
            chunk_rows: 4096,
            page_size: 4096,
        }
    }
}

/// Rows appended but not yet forming a whole chunk.
enum Staged {
    Csr {
        indptr: Vec<u64>,
        indices: Vec<u32>,
        data: Vec<u8>,
    },
    Dense {
        data: Vec<u8>,
    },
}

/// Writes one matrix to a `.cpk` file. Rows are appended in order; the file
/// is valid only after [`CpkWriter::finish`].
pub struct CpkWriter {
    file: File,
    options: CpkOptions,
    dtype: Dtype,
    source: SourceLayout,
    n_cols: u32,
    offset: u64,
    entries: Vec<ChunkEntry>,
    staged: Staged,
    buffers: Buffers,
    threads: usize,
}

impl CpkWriter {
    /// Creates (or truncates) `path` for a matrix of `n_cols` columns of
    /// `dtype` values arriving as `source`.
    pub fn create(
        path: impl AsRef<Path>,
        n_cols: u32,
        dtype: Dtype,
        source: SourceLayout,
        mut options: CpkOptions,
    ) -> Result<Self> {
        options.encode.check()?;
        // Column modes pay off for float values (a gene's typical
        // normalized level); integer counts code smaller and map faster with
        // per-cell dictionaries, so they get modes only when forced.
        if !matches!(dtype, Dtype::F32 | Dtype::F64) && options.encode.values != crate::scans::ValueChoice::Modes {
            options.encode.column_modes = None;
        }
        if n_cols == 0 {
            return Err(Error::InvalidInput("a matrix needs columns"));
        }
        if options.chunk_rows == 0 || options.chunk_rows > u32::MAX as usize {
            return Err(Error::InvalidOptions("chunk_rows must be in [1, 2^32)"));
        }
        if !options.page_size.is_power_of_two() || !(512..=65536).contains(&options.page_size) {
            return Err(Error::InvalidOptions(
                "page size must be a power of two in [512, 65536]",
            ));
        }
        let file = File::create(path)?;
        file.write_all_at(&[0u8; SUPERBLOCK_LEN], 0)?;
        let staged = match source {
            SourceLayout::Csr => Staged::Csr {
                indptr: vec![0],
                indices: Vec::new(),
                data: Vec::new(),
            },
            SourceLayout::Dense => Staged::Dense { data: Vec::new() },
        };
        Ok(Self {
            file,
            options,
            dtype,
            source,
            n_cols,
            offset: SUPERBLOCK_LEN.next_multiple_of(options.page_size as usize) as u64,
            entries: Vec::new(),
            staged,
            buffers: Buffers::default(),
            threads: 1,
        })
    }

    /// Encodes up to `threads` whole chunks of one append concurrently.
    /// Chunks are independent, so the file is identical to a serial write.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    fn rows_written(&self) -> u64 {
        self.entries
            .last()
            .map_or(0, |e| e.first_row + e.n_rows as u64)
    }

    fn staged_rows(&self) -> usize {
        match &self.staged {
            Staged::Csr { indptr, .. } => indptr.len() - 1,
            Staged::Dense { data } => data.len() / (self.n_cols as usize * self.dtype.width()),
        }
    }

    /// Appends dense or CSR rows. Rows in the other form than the writer's
    /// source layout are converted once: dense rows keep their non-zero
    /// values (a value is zero when all its bytes are), CSR rows fill
    /// zeroed dense rows.
    pub fn append<'a>(&mut self, rows: impl Into<MatrixRef<'a>>) -> Result<()> {
        let m = rows.into();
        let form = match m {
            MatrixRef::Csr(_) | MatrixRef::CsrView(_) => SourceLayout::Csr,
            MatrixRef::Dense(_) => SourceLayout::Dense,
        };
        if m.n_genes() != self.n_cols || m.value_width() != self.dtype.width() {
            return Err(Error::InvalidInput("rows differ from the writer's columns or value width"));
        }
        if form != self.source {
            return self.append_converted(m);
        }
        let (n, chunk) = (m.n_rows(), self.options.chunk_rows);
        let mut i = 0;
        while i < n {
            if self.staged_rows() == 0 && n - i >= chunk {
                let whole = (n - i) / chunk;
                if self.threads > 1 && whole > 1 {
                    self.write_chunks(m, i, whole)?;
                    i += whole * chunk;
                } else {
                    self.write_chunk(
                        m,
                        Rows::Range {
                            start: i,
                            end: i + chunk,
                        },
                    )?;
                    i += chunk;
                }
                continue;
            }
            let take = (chunk - self.staged_rows()).min(n - i);
            self.stage(m, i..i + take)?;
            i += take;
            if self.staged_rows() == chunk {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn append_converted(&mut self, m: MatrixRef) -> Result<()> {
        let (n_genes, w) = (self.n_cols, self.dtype.width());
        let width = n_genes as usize * w;
        match m {
            MatrixRef::Dense(d) => {
                let mut indptr = Vec::with_capacity(d.n_rows() + 1);
                indptr.push(0u64);
                let (mut indices, mut data) = (Vec::new(), Vec::new());
                for row in d.data.chunks_exact(width) {
                    for (column, value) in row.chunks_exact(w).enumerate() {
                        if value.iter().any(|&b| b != 0) {
                            indices.push(column as u32);
                            data.extend_from_slice(value);
                        }
                    }
                    indptr.push(indices.len() as u64);
                }
                self.append(CsrRef { n_genes, value_width: w, indptr: &indptr, indices: Indices::U32(&indices), data: &data })
            }
            csr => {
                let n = csr.n_rows();
                let mut data = vec![0u8; n.checked_mul(width).ok_or(Error::InvalidInput("dense rows exceed the address space"))?];
                for (r, row) in data.chunks_exact_mut(width).enumerate() {
                    let crate::scans::RowRef::Sparse(cell) = csr.row(r)? else {
                        unreachable!("CSR rows are sparse")
                    };
                    for t in 0..cell.indices.len() {
                        let column = cell.indices.checked(t)? as usize;
                        if column >= n_genes as usize {
                            return Err(Error::InvalidInput("gene index outside the matrix"));
                        }
                        row[column * w..(column + 1) * w].copy_from_slice(&cell.values[t * w..(t + 1) * w]);
                    }
                }
                self.append(DenseRef { n_genes, value_width: w, data: &data })
            }
        }
    }

    fn stage(&mut self, m: MatrixRef, rows: Range<usize>) -> Result<()> {
        let w = self.dtype.width();
        match (&mut self.staged, m) {
            (
                Staged::Csr {
                    indptr,
                    indices,
                    data,
                },
                m @ (MatrixRef::Csr(_) | MatrixRef::CsrView(_)),
            ) => {
                for r in rows {
                    let crate::scans::RowRef::Sparse(c) = m.row(r)? else {
                        unreachable!()
                    };
                    for t in 0..c.indices.len() {
                        indices.push(c.indices.checked(t)?);
                    }
                    data.extend_from_slice(c.values);
                    indptr.push(indices.len() as u64);
                }
            }
            (Staged::Dense { data }, MatrixRef::Dense(d)) => {
                let width = self.n_cols as usize * w;
                data.extend_from_slice(&d.data[rows.start * width..rows.end * width]);
            }
            _ => unreachable!("append checked the form"),
        }
        Ok(())
    }

    /// Ends the current chunk early, so the next row starts a new one (to
    /// keep chunks aligned with blocks of the source).
    pub fn end_chunk(&mut self) -> Result<()> {
        self.flush()
    }

    /// Appends one chunk already coded with [`encode_chunk`] and packed with
    /// [`pack_chunk`] at this writer's page size, so callers can encode
    /// chunks in parallel and keep only the ordered writes here. Rows staged
    /// by [`CpkWriter::append`] must be ended with [`CpkWriter::end_chunk`]
    /// first.
    pub fn append_packed(&mut self, packed: &[u8]) -> Result<()> {
        if self.staged_rows() != 0 {
            return Err(Error::InvalidInput("staged rows precede a packed chunk"));
        }
        let meta = parse_meta(packed)?;
        if meta.n_dims() != self.n_cols
            || meta.value_width() != self.dtype.width()
            || meta.page_size() != Some(self.options.page_size)
            || meta.chunk_len() != packed.len() as u64
        {
            return Err(Error::InvalidInput(
                "packed chunk differs from the writer's columns, value width or page size",
            ));
        }
        self.write_packed(packed)
    }

    /// Writes one packed chunk at the next page-aligned offset.
    fn write_packed(&mut self, packed: &[u8]) -> Result<()> {
        let meta = parse_meta(packed)?;
        let entry = ChunkEntry {
            first_row: self.rows_written(),
            offset: self.offset,
            len: packed.len() as u64,
            meta_len: meta.meta_len() as u64,
            n_rows: meta.n_cells() as u32,
        };
        self.file.write_all_at(packed, self.offset)?;
        self.offset = (self.offset + packed.len() as u64).next_multiple_of(self.options.page_size as u64);
        self.entries.push(entry);
        Ok(())
    }

    /// Encodes `count` whole chunks from row `start` on scoped threads, at
    /// most `threads` at a time, and writes them in row order.
    fn write_chunks(&mut self, m: MatrixRef, start: usize, count: usize) -> Result<()> {
        let (chunk, page, opt) = (self.options.chunk_rows, self.options.page_size, self.options.encode);
        // Each worker slot keeps its buffers across batches.
        let mut slots: Vec<Buffers> = (0..self.threads.min(count)).map(|_| Buffers::default()).collect();
        let mut done = 0;
        while done < count {
            let batch = (count - done).min(self.threads);
            let lens: Vec<Result<usize>> = std::thread::scope(|scope| {
                let workers: Vec<_> = slots[..batch]
                    .iter_mut()
                    .enumerate()
                    .map(|(k, buf)| {
                        let lo = start + (done + k) * chunk;
                        let rows = Rows::Range { start: lo, end: lo + chunk };
                        scope.spawn(move || buf.encode_packed(m, rows, &opt, page))
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|w| w.join().unwrap_or(Err(Error::InvalidInput("chunk encoder panicked"))))
                    .collect()
            });
            for (buf, len) in slots.iter().zip(lens) {
                self.write_packed(&buf.packed[..len?])?;
            }
            done += batch;
        }
        Ok(())
    }

    /// Encodes the staged rows as one chunk.
    fn flush(&mut self) -> Result<()> {
        if self.staged_rows() == 0 {
            return Ok(());
        }
        let empty = match self.source {
            SourceLayout::Csr => Staged::Csr {
                indptr: vec![0],
                indices: Vec::new(),
                data: Vec::new(),
            },
            SourceLayout::Dense => Staged::Dense { data: Vec::new() },
        };
        let staged = std::mem::replace(&mut self.staged, empty);
        let (n_genes, value_width) = (self.n_cols, self.dtype.width());
        match &staged {
            Staged::Csr {
                indptr,
                indices,
                data,
            } => {
                let m = CsrRef {
                    n_genes,
                    value_width,
                    indptr,
                    indices: Indices::U32(indices),
                    data,
                };
                self.write_chunk(MatrixRef::Csr(m), Rows::All)?;
            }
            Staged::Dense { data } => {
                self.write_chunk(
                    MatrixRef::Dense(DenseRef {
                        n_genes,
                        value_width,
                        data,
                    }),
                    Rows::All,
                )?;
            }
        }
        Ok(())
    }

    fn write_chunk(&mut self, m: MatrixRef, rows: Rows) -> Result<()> {
        let mut buf = std::mem::take(&mut self.buffers);
        let written = buf
            .encode_packed(m, rows, &self.options.encode, self.options.page_size)
            .and_then(|len| self.write_packed(&buf.packed[..len]));
        self.buffers = buf;
        written
    }

    /// Encodes the remaining rows, writes the directory, which ends the
    /// file, and finally the superblock.
    pub fn finish(mut self) -> Result<()> {
        self.flush()?;
        let mut directory = Vec::with_capacity(self.entries.len() * cpk::ENTRY_LEN);
        for e in &self.entries {
            cpk::write_entry(e, &mut directory);
        }
        let directory_offset = self
            .entries
            .last()
            .map_or(self.offset, |e| e.offset + e.len);
        let file_len = directory_offset + directory.len() as u64;
        self.file.write_all_at(&directory, directory_offset)?;
        self.file.set_len(file_len)?;
        let n_chunks = u32::try_from(self.entries.len())
            .map_err(|_| Error::InvalidInput("more than 2^32 - 1 chunks"))?;
        let sb = Superblock {
            page_size: self.options.page_size,
            dtype: self.dtype,
            source: self.source,
            n_rows: self.rows_written(),
            n_cols: self.n_cols,
            n_chunks,
            chunk_rows: self.options.chunk_rows as u32,
            directory_offset,
            file_len,
        };
        self.file.write_all_at(&sb.write(&directory), 0)?;
        Ok(())
    }
}

/// Encoding buffers reused from chunk to chunk.
#[derive(Default)]
struct Buffers {
    encoded: Vec<u8>,
    ws: Vec<u64>,
    packed: Vec<u8>,
}

/// `v` with at least `n` elements; a grown buffer is allocated afresh
/// rather than resized, which leaves its pages untouched until used.
fn room<T: Clone + Default>(v: &mut Vec<T>, n: usize) {
    if v.len() < n {
        *v = vec![T::default(); n];
    }
}

impl Buffers {
    /// Encodes and packs one chunk into `self.packed`; returns its length.
    fn encode_packed(&mut self, m: MatrixRef, rows: Rows, opt: &EncodeOptions, page: u32) -> Result<usize> {
        let sizes = chunk_sizes(m, rows, opt)?;
        room(&mut self.encoded, sizes.output_bytes);
        room(&mut self.ws, sizes.workspace_words);
        let len = encode_chunk(m, rows, opt, &mut self.encoded, &mut self.ws)?;
        let need = packed_len(&self.encoded[..len], page)?;
        room(&mut self.packed, need);
        pack_chunk(&self.encoded[..len], page, &mut self.packed)
    }
}
