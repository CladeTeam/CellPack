//! The `.cpk` file: one matrix, stored as a superblock followed by
//! page-aligned chunks of consecutive rows and the chunk directory.
//!
//! ```text
//! superblock  SUPERBLOCK_LEN bytes at offset 0
//!   0   magic "CPK\0" | version u16 | flags u16 (0) | page_size u32
//!   12  dtype u8 | source u8 | 0 u16
//!   16  n_rows u64 | n_cols u32 | n_chunks u32 | chunk_rows u32 | 0 u32
//!   40  directory_offset u64 | file_len u64
//!   56  hash u64 of bytes 0..56 and the directory
//! chunks      packed chunks (see [`crate::format`]), rows in order, each at a
//!             page-aligned offset
//! directory   n_chunks × { first_row u64 | offset u64 | len u64 | meta_len u64 | n_rows u32 | 0 u32 }
//! ```
//!
//! The file holds the matrix only: row and column labels and other
//! annotations belong to the container or caller (e.g. AnnData `obs`/`var`).
//! `dtype` is the numeric type of the stored values and `source` records
//! whether the matrix arrived as CSR or dense; neither decides how rows are
//! stored (every record picks its own layout). The directory ends the file.
//! The superblock is written last, so a file whose writer did not finish has
//! no valid magic.

use crate::error::{Error, Result};

/// File magic, written only by a finished superblock.
pub(crate) const MAGIC: [u8; 4] = *b"CPK\0";
/// Superblock layout version.
pub(crate) const VERSION: u16 = 1;
/// Bytes reserved for the superblock at the start of the file.
pub const SUPERBLOCK_LEN: usize = 4096;
const HEAD: usize = 56;
/// Bytes of one directory entry.
pub const ENTRY_LEN: usize = 40;

/// Numeric type of a matrix's values; its width is the chunks' value width.
///
/// The discriminant is the code stored in the superblock's `dtype` byte, so
/// the declaration order is part of the file format and cannot be rearranged
/// without changing the version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// Unsigned, 1 byte (code 0).
    U8,
    /// Unsigned, 2 bytes (code 1).
    U16,
    /// Unsigned, 4 bytes (code 2).
    U32,
    /// Unsigned, 8 bytes (code 3).
    U64,
    /// Signed, 1 byte (code 4).
    I8,
    /// Signed, 2 bytes (code 5).
    I16,
    /// Signed, 4 bytes (code 6).
    I32,
    /// Signed, 8 bytes (code 7).
    I64,
    /// IEEE binary32, 4 bytes (code 8).
    F32,
    /// IEEE binary64, 8 bytes (code 9).
    F64,
}

impl Dtype {
    /// Bytes of one stored value: 1, 2, 4 or 8.
    pub fn width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// The superblock `dtype` byte: the variant's discriminant.
    pub(crate) fn code(self) -> u8 {
        self as u8
    }

    /// The type a superblock `dtype` byte names, or `None` for an unknown code.
    pub(crate) fn from_code(c: u8) -> Option<Self> {
        Some(match c {
            0 => Self::U8,
            1 => Self::U16,
            2 => Self::U32,
            3 => Self::U64,
            4 => Self::I8,
            5 => Self::I16,
            6 => Self::I32,
            7 => Self::I64,
            8 => Self::F32,
            9 => Self::F64,
            _ => return None,
        })
    }

    /// One little-endian value as `f32`.
    ///
    /// Reads the first [`Dtype::width`] bytes of `b` and converts, so integers
    /// lose precision above 2^24 and `F64` rounds; `b` must hold that many
    /// bytes.
    pub fn f32(self, b: &[u8]) -> f32 {
        macro_rules! read {
            ($t:ty) => {
                <$t>::from_le_bytes(b[..size_of::<$t>()].try_into().unwrap()) as f32
            };
        }
        match self {
            Self::U8 => b[0] as f32,
            Self::I8 => b[0] as i8 as f32,
            Self::U16 => read!(u16),
            Self::U32 => read!(u32),
            Self::U64 => read!(u64),
            Self::I16 => read!(i16),
            Self::I32 => read!(i32),
            Self::I64 => read!(i64),
            Self::F32 => read!(f32),
            Self::F64 => read!(f64),
        }
    }
}

/// The form a matrix arrived in.
///
/// Recorded for the caller's benefit; every record still picks its own layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceLayout {
    /// CSR input: indices and values per row, sorted by column.
    Csr,
    /// Dense input: `n_cols` values per row.
    Dense,
}

/// Decoded superblock fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Superblock {
    /// Page size records were packed to, in bytes; a power of two in
    /// `[512, 65536]`, which also aligns every chunk offset.
    pub page_size: u32,
    /// Numeric type of the stored values.
    pub dtype: Dtype,
    /// Layout the matrix arrived in; records choose their own layout.
    pub source: SourceLayout,
    /// Rows of the whole matrix; the chunks' row counts add up to it.
    pub n_rows: u64,
    /// Columns of the whole matrix, i.e. the dims of every chunk.
    pub n_cols: u32,
    /// Directory entries, one per chunk.
    pub n_chunks: u32,
    /// Rows per chunk the writer aimed for; early-ended and pre-packed chunks
    /// can have other row counts.
    pub chunk_rows: u32,
    /// Byte offset of the directory from the start of the file.
    pub directory_offset: u64,
    /// End of the directory, which ends the file.
    pub file_len: u64,
}

/// One chunk: rows `first_row..first_row + n_rows` at `offset`, `len` bytes
/// of which the first `meta_len` are its metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkEntry {
    /// First row of the chunk in the matrix; chunks tile `0..n_rows` in order.
    pub first_row: u64,
    /// Byte offset of the chunk from the start of the file, a multiple of the
    /// page size.
    pub offset: u64,
    /// Bytes of the whole chunk: metadata, padding and records.
    pub len: u64,
    /// Leading bytes of the chunk that are its metadata (`len` includes them).
    pub meta_len: u64,
    /// Rows in this chunk, `n_cells` of its metadata.
    pub n_rows: u32,
}

/// Word-at-a-time 64-bit hash for detecting damaged or mismatched files;
/// not cryptographic.
///
/// Folds 8-byte little-endian words as xor, multiply and rotate, and folds the
/// part's length into its final (zero-padded) word. Cheap enough to run over a
/// directory on open.
pub(crate) fn hash(parts: &[&[u8]]) -> u64 {
    let mut h = 0x243f_6a88_85a3_08d3u64;
    for part in parts {
        let mut words = part.chunks_exact(8);
        for w in &mut words {
            h = (h ^ u64::from_le_bytes(w.try_into().unwrap()))
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(29);
        }
        let mut tail = [0u8; 8];
        let rest = words.remainder();
        tail[..rest.len()].copy_from_slice(rest);
        h = (h ^ u64::from_le_bytes(tail) ^ (part.len() as u64) << 56)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(29);
    }
    h
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

impl Superblock {
    /// The superblock bytes, with the hash over them and `directory`.
    ///
    /// Fields occupy bytes 0..56, the hash 56..64, the reserved rest zero. The
    /// hash covers the fields and the whole directory, which ties the two
    /// halves of the file together: [`parse_directory`] recomputes it.
    pub(crate) fn write(&self, directory: &[u8]) -> [u8; SUPERBLOCK_LEN] {
        let mut b = [0u8; SUPERBLOCK_LEN];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..6].copy_from_slice(&VERSION.to_le_bytes());
        b[8..12].copy_from_slice(&self.page_size.to_le_bytes());
        b[12] = self.dtype.code();
        b[13] = self.source as u8;
        b[16..24].copy_from_slice(&self.n_rows.to_le_bytes());
        b[24..28].copy_from_slice(&self.n_cols.to_le_bytes());
        b[28..32].copy_from_slice(&self.n_chunks.to_le_bytes());
        b[32..36].copy_from_slice(&self.chunk_rows.to_le_bytes());
        b[40..48].copy_from_slice(&self.directory_offset.to_le_bytes());
        b[48..56].copy_from_slice(&self.file_len.to_le_bytes());
        let h = hash(&[&b[..HEAD], directory]);
        b[HEAD..HEAD + 8].copy_from_slice(&h.to_le_bytes());
        b
    }

    /// Bytes of the directory this superblock describes:
    /// `n_chunks` entries of [`ENTRY_LEN`] bytes each.
    pub fn directory_len(&self) -> usize {
        self.n_chunks as usize * ENTRY_LEN
    }
}

/// Parses a superblock (at least [`SUPERBLOCK_LEN`] bytes) and checks its
/// fields against each other; the hash is checked by [`parse_directory`].
///
/// Checks magic, version, the reserved bytes, a known dtype and source code, a
/// power-of-two page size in `[512, 65536]`, a non-zero column count, and that
/// the directory ends the file exactly. A file whose writer stopped before the
/// superblock has no valid magic, which is how a truncated write is detected.
pub fn parse_superblock(b: &[u8]) -> Result<Superblock> {
    let b = b
        .get(..SUPERBLOCK_LEN)
        .ok_or(Error::CorruptMeta("file shorter than the superblock"))?;
    if b[0..4] != MAGIC {
        return Err(Error::CorruptMeta("not a cpk file (or an unfinished one)"));
    }
    if u16_at(b, 4) != VERSION {
        return Err(Error::CorruptMeta("unsupported cpk version"));
    }
    if u16_at(b, 6) != 0
        || u16_at(b, 14) != 0
        || u32_at(b, 36) != 0
        || b[HEAD + 8..].iter().any(|&x| x != 0)
    {
        return Err(Error::CorruptMeta("reserved superblock bytes set"));
    }
    let dtype = Dtype::from_code(b[12]).ok_or(Error::CorruptMeta("unknown dtype"))?;
    let source = match b[13] {
        0 => SourceLayout::Csr,
        1 => SourceLayout::Dense,
        _ => return Err(Error::CorruptMeta("unknown source layout")),
    };
    let s = Superblock {
        page_size: u32_at(b, 8),
        dtype,
        source,
        n_rows: u64_at(b, 16),
        n_cols: u32_at(b, 24),
        n_chunks: u32_at(b, 28),
        chunk_rows: u32_at(b, 32),
        directory_offset: u64_at(b, 40),
        file_len: u64_at(b, 48),
    };
    if !s.page_size.is_power_of_two() || !(512..=65536).contains(&s.page_size) {
        return Err(Error::CorruptMeta("bad page size"));
    }
    if s.n_cols == 0 {
        return Err(Error::CorruptMeta("no columns"));
    }
    if s.directory_offset.checked_add(s.directory_len() as u64) != Some(s.file_len) {
        return Err(Error::CorruptMeta("the directory does not end the file"));
    }
    Ok(s)
}

/// Parses and checks the directory against its superblock: the hash, rows
/// that tile `0..n_rows` in order, and page-aligned, ascending,
/// non-overlapping chunks after the superblock and before the directory.
///
/// `super_bytes` is the whole superblock and `dir` the directory bytes (at
/// least `s.directory_len()`). Entries come back in row order, which is also
/// file order. The checks are inter-field, so a directory never needs to be
/// trusted beyond what the returned entries say.
pub fn parse_directory(super_bytes: &[u8], s: &Superblock, dir: &[u8]) -> Result<Vec<ChunkEntry>> {
    let dir = dir
        .get(..s.directory_len())
        .ok_or(Error::CorruptMeta("truncated directory"))?;
    if hash(&[&super_bytes[..HEAD], dir]) != u64_at(super_bytes, HEAD) {
        return Err(Error::CorruptMeta("superblock or directory damaged"));
    }
    let mut out = Vec::with_capacity(s.n_chunks as usize);
    let (mut row, mut end) = (0u64, SUPERBLOCK_LEN as u64);
    for e in dir.chunks_exact(ENTRY_LEN) {
        let c = ChunkEntry {
            first_row: u64_at(e, 0),
            offset: u64_at(e, 8),
            len: u64_at(e, 16),
            meta_len: u64_at(e, 24),
            n_rows: u32_at(e, 32),
        };
        if u32_at(e, 36) != 0 {
            return Err(Error::CorruptMeta("reserved directory bytes set"));
        }
        let stop = c.offset.checked_add(c.len);
        if c.first_row != row
            || !c.offset.is_multiple_of(s.page_size as u64)
            || c.offset < end
            || stop.is_none_or(|x| x > s.directory_offset)
            || c.meta_len > c.len
        {
            return Err(Error::CorruptMeta("bad chunk directory entry"));
        }
        row += c.n_rows as u64;
        end = stop.unwrap();
        out.push(c);
    }
    if row != s.n_rows {
        return Err(Error::CorruptMeta("chunks do not cover the rows"));
    }
    Ok(out)
}

/// Appends the 40-byte directory entry of `c` to `out`: five fields and the
/// zero reserved word that [`parse_directory`] requires.
pub(crate) fn write_entry(c: &ChunkEntry, out: &mut Vec<u8>) {
    out.extend_from_slice(&c.first_row.to_le_bytes());
    out.extend_from_slice(&c.offset.to_le_bytes());
    out.extend_from_slice(&c.len.to_le_bytes());
    out.extend_from_slice(&c.meta_len.to_le_bytes());
    out.extend_from_slice(&c.n_rows.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
}
