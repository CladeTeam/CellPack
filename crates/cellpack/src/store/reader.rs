//! Opened `.cpk` files: superblock, chunk directory and chunk bytes, plus
//! allocating convenience reads over [`ReadSelection`](super::ReadSelection).

use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::Path;
use std::sync::Arc;

use crate::format::{
    cpk::{self, ChunkEntry, Dtype, SUPERBLOCK_LEN, Superblock},
};
use super::ReadWorkspace;
use crate::{Error, Result};

/// Owned CSR rows with ascending source column ids within each row.
/// Values are little-endian bytes interpreted with [`CpkFile::dtype`];
/// this object does not carry a separate dtype or shape descriptor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csr {
    /// Row offsets starting at zero; one more entry than the output row count.
    pub indptr: Vec<u64>,
    /// Source column ids, one per stored output value.
    pub indices: Vec<u32>,
    /// Value bytes: `indices.len() * file.dtype().width()` bytes.
    pub data: Vec<u8>,
}

/// An opened `.cpk` file.
pub struct CpkFile {
    pub(super) file: Arc<File>,
    offset: u64,
    superblock: Superblock,
    entries: Vec<ChunkEntry>,
}

impl CpkFile {
    /// Opens a complete `.cpk` stream and reads its superblock and directory.
    /// Record payloads remain on disk until a read is requested.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_region(path, 0, None)
    }

    /// Opens a CellPack stream embedded in a bounded region of another file.
    /// `offset` and `length` are outer-file bytes; `None` permits the region
    /// through the end of the file. The region must cover the declared stream.
    /// The caller locates the region; this method does not parse its container.
    pub fn open_region(path: impl AsRef<Path>, offset: u64, length: Option<u64>) -> Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let available = file_len
            .checked_sub(offset)
            .ok_or(Error::InvalidInput("store starts past file end"))?;
        let length = length.unwrap_or(available);
        if length > available || length < SUPERBLOCK_LEN as u64 {
            return Err(Error::InvalidInput(
                "store region outside file or shorter than superblock",
            ));
        }
        let mut head = vec![0u8; SUPERBLOCK_LEN];
        file.read_exact_at(&mut head, offset)?;
        let superblock = cpk::parse_superblock(&head)?;
        if length < superblock.file_len {
            return Err(Error::CorruptMeta("file shorter than its superblock says"));
        }
        let mut dir = vec![0u8; superblock.directory_len()];
        file.read_exact_at(&mut dir, offset + superblock.directory_offset)?;
        let entries = cpk::parse_directory(&head, &superblock, &dir)?;
        Ok(Self {
            file: Arc::new(file),
            offset,
            superblock,
            entries,
        })
    }

    /// Borrows the parsed matrix-stream descriptor without additional I/O.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }
    /// Device and inode of the opened descriptor, independent of path replacement.
    pub fn identity(&self) -> Result<(u64, u64)> {
        let metadata = self.file.metadata()?;
        Ok((metadata.dev(), metadata.ino()))
    }
    /// Start of this stream in its outer file, in bytes.
    pub fn byte_offset(&self) -> u64 {
        self.offset
    }
    /// Number of rows in the matrix, independent of any prepared selection.
    pub fn n_rows(&self) -> u64 {
        self.superblock.n_rows
    }
    /// Number of source columns in every matrix row.
    pub fn n_cols(&self) -> u32 {
        self.superblock.n_cols
    }
    /// Stored numeric type used to interpret allocated read value bytes.
    pub fn dtype(&self) -> Dtype {
        self.superblock.dtype
    }
    /// Borrows the matrix directory in row order; offsets are stream-relative.
    pub fn chunks(&self) -> &[ChunkEntry] {
        &self.entries
    }
    /// A chunk's metadata bytes.
    pub fn read_meta(&self, chunk: usize) -> Result<Vec<u8>> {
        let e = self.entries.get(chunk).ok_or(Error::RowOutOfRange {
            row: chunk,
            n_rows: self.entries.len(),
        })?;
        let mut b = vec![0u8; e.meta_len as usize];
        self.file.read_exact_at(&mut b, self.offset + e.offset)?;
        Ok(b)
    }

    /// A whole chunk's bytes (metadata and records).
    pub fn read_chunk(&self, chunk: usize) -> Result<Vec<u8>> {
        let e = self.entries.get(chunk).ok_or(Error::RowOutOfRange {
            row: chunk,
            n_rows: self.entries.len(),
        })?;
        let mut b = vec![0u8; e.len as usize];
        self.file.read_exact_at(&mut b, self.offset + e.offset)?;
        Ok(b)
    }
    /// Reads a chunk into an exact-size caller byte buffer.
    /// `metadata = true` requires that directory entry's `meta_len` bytes;
    /// `false` requires its full `len`. A failed read can leave partial output.
    pub fn read_chunk_into(&self, chunk: usize, metadata: bool, out: &mut [u8]) -> Result<()> {
        let e = self.entries.get(chunk).ok_or(Error::RowOutOfRange {
            row: chunk,
            n_rows: self.entries.len(),
        })?;
        let need = if metadata { e.meta_len } else { e.len } as usize;
        if out.len() != need {
            return Err(Error::OutputTooSmall {
                need,
                have: out.len(),
            });
        }
        self.file.read_exact_at(out, self.offset + e.offset)?;
        Ok(())
    }

    /// The chunks overlapping `rows`, each with its local row range.
    pub(super) fn spans(&self, rows: &Range<u64>) -> Result<Vec<(usize, Range<usize>)>> {
        if rows.start > rows.end || rows.end > self.n_rows() {
            return Err(Error::RowOutOfRange {
                row: rows.end as usize,
                n_rows: self.n_rows() as usize,
            });
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let first = self
            .entries
            .partition_point(|e| e.first_row + e.n_rows as u64 <= rows.start);
        let mut out = Vec::new();
        for (i, e) in self.entries.iter().enumerate().skip(first) {
            if e.first_row >= rows.end {
                break;
            }
            let a = rows.start.max(e.first_row) - e.first_row;
            let b = rows.end.min(e.first_row + e.n_rows as u64) - e.first_row;
            out.push((i, a as usize..b as usize));
        }
        Ok(out)
    }

    /// Rows as CSR in source column order, using the borrowed-output reader.
    pub fn read_csr(&self, rows: Range<u64>) -> Result<Csr> {
        let selection = self.prepare_read(rows)?;
        let spec = selection.spec();
        let mut indptr = vec![0u64; spec.rows + 1];
        let mut indices = vec![0u32; spec.nnz];
        let data = by_dtype!(spec.dtype, T, {
            let mut values = vec![T::default(); spec.nnz];
            selection.read_csr_into(&mut indptr, &mut indices, &mut values, &mut ReadWorkspace::default(), 1)?;
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        });
        Ok(Csr { indptr, indices, data })
    }

    /// Rows as dense `n_cols` values each, row-major.
    pub fn read_dense(&self, rows: Range<u64>) -> Result<Vec<u8>> {
        let selection = self.prepare_read(rows)?;
        let spec = selection.spec();
        Ok(by_dtype!(spec.dtype, T, {
            let mut values = vec![T::default(); spec.dense_len()?];
            selection.read_dense_into(&mut values, &mut ReadWorkspace::default(), 1)?;
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        }))
    }
}

macro_rules! by_dtype {
    ($dtype:expr, $T:ident, $body:expr) => {
        match $dtype {
            Dtype::U8 => { type $T = u8; $body }
            Dtype::U16 => { type $T = u16; $body }
            Dtype::U32 => { type $T = u32; $body }
            Dtype::U64 => { type $T = u64; $body }
            Dtype::I8 => { type $T = i8; $body }
            Dtype::I16 => { type $T = i16; $body }
            Dtype::I32 => { type $T = i32; $body }
            Dtype::I64 => { type $T = i64; $body }
            Dtype::F32 => { type $T = f32; $body }
            Dtype::F64 => { type $T = f64; $body }
        }
    };
}
use by_dtype;

/// Output dimensions of a prepared selection: `nnz` is the CSR capacity
/// and `dtype` the stored dtype.
#[derive(Clone, Copy, Debug)]
pub struct ReadSpec {
    /// Selected output rows, including repeats in a gather.
    pub rows: usize,
    /// Output panel width after the optional column map.
    pub columns: usize,
    /// Source-based CSR entry capacity, before mapped columns are dropped.
    pub nnz: usize,
    /// Stored dtype, not the generic output type of a converted read.
    pub dtype: Dtype,
}

impl ReadSpec {
    /// Elements of the dense output.
    pub fn dense_len(&self) -> Result<usize> {
        self.rows
            .checked_mul(self.columns)
            .filter(|&n| n.checked_mul(8).is_some_and(|b| b <= isize::MAX as usize))
            .ok_or(Error::InvalidInput("dense output too large"))
    }
    /// Dense byte count at the stored dtype width, with overflow checks.
    /// For converted typed output, use [`Self::dense_len`] elements of `T`.
    pub fn dense_bytes(&self) -> Result<usize> {
        self.rows
            .checked_mul(self.columns)
            .and_then(|n| n.checked_mul(self.dtype.width()))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(Error::InvalidInput("dense output too large"))
    }
    /// CSR value capacity in bytes at the stored dtype width.
    /// Actual mapped nnz may be smaller, and converted `T` may have another width.
    pub fn value_bytes(&self) -> Result<usize> {
        self.nnz
            .checked_mul(self.dtype.width())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(Error::InvalidInput("CSR output too large"))
    }
}

