//! Chunk and record layout, and the zero-copy metadata view.
//!
//! A chunk is its metadata followed by its records:
//!
//! ```text
//! header   64 B   magic "CPAK" | version u16 | flags u16 (bit 0: renumbered,
//!                 bit 1: column modes)
//!                 n_cells u32 | n_dims u32 | value_width u8 | max_lanes u8
//!                 lane_min_log2 u8 | page_log2 u8 | max_k u32 | max_nnz u32 | max_elems u32
//!                 nnz_total u64 | meta_len u64 | records_len u64 | 0 u64
//! models   3 × 256 × u16    symbol frequencies (sum 2^11, or 0 when unused): gap, rank, dict
//! dims    n_dims × u16|u32 source dim (column) of each coded id (renumbered only)
//! modes    n_dims × value_width  most frequent value of each coded id (with
//!                               column modes only)
//! offsets  n_cells × u64        record start within the records region
//! lengths  n_cells × u32        record length (0 for an empty cell)
//! kinds    n_cells × u8         each record's flags byte (0 for an empty cell)
//! nnz      (n_cells + 1) × u64  cumulative nnz (the CSR indptr of the
//!                               decoded sparse rows): stored entries for a
//!                               sparse record, non-zeros for a dense one
//! [pad]                         zeros up to a page boundary when paged
//! records  one per cell, in any order, possibly with gaps between them
//! ```
//!
//! With `page_log2 = 0` records are contiguous in cell order; otherwise the
//! records region starts on a page boundary and records are placed so each
//! reads its minimum number of pages (see [`crate::scans::pack_chunk`]). The
//! offset and nnz tables start 8-aligned. All integers are little-endian.
//! Dim (column) ids, the dim order and raw record indices use 2 bytes when
//! `n_dims <= 65536`, else 4.
//!
//! A record is `flags u8 | nnz varint | index section | value section`. It
//! stores its *elements*: the stored entries of a sparse layout (an explicit
//! zero is still a stored entry) or every dim of a dense layout (index mode 2,
//! no index section; zeros are values). `nnz` counts the stored entries of a
//! sparse record and the non-zero values of a dense one; a dense record's
//! element count is `n_dims`. `max_elems` bounds the elements of any record in
//! the chunk.
//!
//! ```text
//! flags    bits 0-1 index mode (0 raw, 1 gap-coded, 2 dense), bits 2-3 value
//!          mode (0 raw, 1 dictionary-coded, 2 mode-coded), bits 4-7 zero
//! index    raw:   nnz coded ids, ascending (source ids without renumbering)
//!          coded: rans_len varint | bits_len varint | gap rans | bits tail
//!          dense: absent
//! value    raw:   one value per element, in coded order
//!          coded: k varint | dict_rans_len | dict_bits_len | [k > 1: rank_rans_len | rank_bits_len | rank rans]
//!                 | pool tail
//!          mode:  k varint | [k > 0: dict_rans_len | dict_bits_len | rank_rans_len | rank_bits_len | rank rans
//!                 | pool tail]
//! ```
//!
//! A mode-coded record stores only the values that differ from their dim's
//! column mode: they form the dictionary (`k` distinct values, possibly 0),
//! and rank 0 stands for the column mode while rank `r > 0` is dictionary
//! entry `r - 1`. With `k = 0` every value is its column mode.
//!
//! The leading bytes of the gap escape bits, and of the value pool (dict
//! rans, dict escape bits, rank escape bits, concatenated), ride in the
//! initial states of the gap and rank streams: as many as their capacity
//! (2 bytes per lane) allows. Only the remainder is stored as the tail. A
//! dictionary-coded record with a single entry has no rank stream to carry
//! into, so it stores its pool whole; a mode-coded record with `k = 0` has no
//! pool at all.
//!
//! `meta_len` covers the header, models and every table up to the end of the
//! nnz table; the padding left by the 8-byte alignment of the offset and
//! nnz tables belongs to no section and is written as zeros, unchecked.
//! Section offsets are a pure function of the chunk shape, so a reader
//! recomputes them from the header instead of storing them.

use std::hash::{Hash, Hasher};

pub mod cpk;
pub use cpk::{ChunkEntry, Dtype, SourceLayout, Superblock};

use crate::error::{Error, Result};
use crate::scans::rans::payload_capacity;
use crate::scans::symbol::MAX_SYMBOL;

// Chunk magic and version, the fixed 64-byte header, and the three rANS models
// (gap, rank, dict) in the order the metadata stores them.
pub(crate) const MAGIC: [u8; 4] = *b"CPAK";
pub(crate) const VERSION: u16 = 1;
pub(crate) const HEADER_LEN: usize = 64;
pub(crate) const N_MODELS: usize = 3;
pub(crate) const GAP: usize = 0;
pub(crate) const RANK: usize = 1;
pub(crate) const DICT: usize = 2;
const MODEL_BYTES: usize = 512;
const FLAG_RENUMBERED: u16 = 1;
const FLAG_MODES: u16 = 2;

// Record flag encodings: bits 0-1 index mode, bits 2-3 value mode.
pub(crate) const IDX_RAW: u8 = 0;
pub(crate) const IDX_GAP: u8 = 1;
pub(crate) const IDX_DENSE: u8 = 2;
// Page-size bounds of the format: 512 B to 64 KiB.
pub(crate) const MIN_PAGE_LOG2: u32 = 9;
pub(crate) const MAX_PAGE_LOG2: u32 = 16;

pub(crate) const VAL_RAW: u8 = 0;
pub(crate) const VAL_DICT: u8 = 1;
pub(crate) const VAL_MODE: u8 = 2;

/// Byte width of stored dim ids (2 or 4) for a dim count: ids are
/// `0..n_dims`, so 2 bytes suffice through 65536 dims.
#[inline]
pub(crate) fn id_width(n_dims: u32) -> usize {
    if n_dims <= 1 << 16 { 2 } else { 4 }
}

/// Section offsets of the metadata, a function of the chunk shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub(crate) dims_off: usize,
    pub(crate) modes_off: usize,
    pub(crate) offsets_off: usize,
    pub(crate) lengths_off: usize,
    pub(crate) kinds_off: usize,
    pub(crate) nnz_off: usize,
    pub(crate) meta_len: usize,
}

impl Layout {
    /// `modes_len` is the column-mode table's size (0 without one). Returns
    /// `None` when any section offset or `meta_len` overflows `usize`, which a
    /// header claiming an enormous dim or cell count can cause.
    pub(crate) fn new(
        n_cells: usize,
        n_dims: u32,
        renumbered: bool,
        modes_len: usize,
    ) -> Option<Layout> {
        let dims_off = HEADER_LEN + N_MODELS * MODEL_BYTES;
        let dims_len = if renumbered {
            (n_dims as usize).checked_mul(id_width(n_dims))?
        } else {
            0
        };
        let modes_off = dims_off.checked_add(dims_len)?;
        let offsets_off = modes_off
            .checked_add(modes_len)?
            .checked_next_multiple_of(8)?;
        let lengths_off = offsets_off.checked_add(n_cells.checked_mul(8)?)?;
        let kinds_off = lengths_off.checked_add(n_cells.checked_mul(4)?)?;
        let nnz_off = kinds_off
            .checked_add(n_cells)?
            .checked_next_multiple_of(8)?;
        let meta_len = nnz_off.checked_add(n_cells.checked_add(1)?.checked_mul(8)?)?;
        Some(Layout {
            dims_off,
            modes_off,
            offsets_off,
            lengths_off,
            kinds_off,
            nnz_off,
            meta_len,
        })
    }
}

/// Start of the records region in a chunk.
pub(crate) fn records_start(meta_len: usize, page_log2: u32) -> Option<usize> {
    if page_log2 == 0 {
        Some(meta_len)
    } else {
        meta_len.checked_next_multiple_of(1 << page_log2)
    }
}

/// Header fields as written by the encoder.
pub(crate) struct Header {
    pub(crate) renumbered: bool,
    pub(crate) modes: bool,
    pub(crate) n_cells: u32,
    pub(crate) n_dims: u32,
    pub(crate) value_width: u8,
    pub(crate) max_lanes: u8,
    pub(crate) lane_min_log2: u8,
    pub(crate) page_log2: u8,
    pub(crate) max_k: u32,
    pub(crate) max_nnz: u32,
    pub(crate) max_elems: u32,
    pub(crate) nnz_total: u64,
    pub(crate) meta_len: u64,
    pub(crate) records_len: u64,
}

impl Header {
    /// Updates the page size and records length of written metadata, at the
    /// header bytes the layout fixes: `page_log2` is byte 19, `records_len`
    /// bytes 48..56. Packing calls it once the records region is placed, since
    /// both fields depend on the final placement.
    pub(crate) fn rewrite_records(meta: &mut [u8], page_log2: u8, records_len: u64) {
        meta[19] = page_log2;
        meta[48..56].copy_from_slice(&records_len.to_le_bytes());
    }

    /// Writes the 64-byte header, zeroing the reserved bytes that
    /// [`parse_meta`] requires to be zero.
    pub(crate) fn write(&self, out: &mut [u8]) {
        let h = &mut out[..HEADER_LEN];
        h.fill(0);
        h[0..4].copy_from_slice(&MAGIC);
        h[4..6].copy_from_slice(&VERSION.to_le_bytes());
        let flags = if self.renumbered { FLAG_RENUMBERED } else { 0 }
            | if self.modes { FLAG_MODES } else { 0 };
        h[6..8].copy_from_slice(&flags.to_le_bytes());
        h[8..12].copy_from_slice(&self.n_cells.to_le_bytes());
        h[12..16].copy_from_slice(&self.n_dims.to_le_bytes());
        h[16] = self.value_width;
        h[17] = self.max_lanes;
        h[18] = self.lane_min_log2;
        h[19] = self.page_log2;
        h[20..24].copy_from_slice(&self.max_k.to_le_bytes());
        h[24..28].copy_from_slice(&self.max_nnz.to_le_bytes());
        h[28..32].copy_from_slice(&self.max_elems.to_le_bytes());
        h[32..40].copy_from_slice(&self.nnz_total.to_le_bytes());
        h[40..48].copy_from_slice(&self.meta_len.to_le_bytes());
        h[48..56].copy_from_slice(&self.records_len.to_le_bytes());
    }
}

#[inline(always)]
fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

#[inline(always)]
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// An unaligned little-endian `u64` at `at`; panics if the range lies outside `b`.
#[inline(always)]
pub(crate) fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Interpretation of a chunk's metadata bytes. Holds no data of its own:
/// header fields are decoded once, sections are borrowed.
#[derive(Clone, Copy, Debug)]
pub struct Meta<'a> {
    model_key: u64,
    renumbered: bool,
    n_cells: u32,
    n_dims: u32,
    value_width: usize,
    max_lanes: u32,
    lane_min: u32,
    page_log2: u32,
    max_k: u32,
    max_nnz: u32,
    max_elems: u32,
    nnz_total: u64,
    meta_len: usize,
    records_start: usize,
    records_len: u64,
    models: &'a [u8],
    dims: &'a [u8],
    modes: Option<&'a [u8]>,
    offsets: &'a [u8],
    lengths: &'a [u8],
    kinds: &'a [u8],
    nnz: &'a [u8],
}

/// Immutable metadata bytes and a validated descriptor. Views borrow only this
/// owner; section offsets avoid self-referential pointers and lifetime extension.
#[derive(Debug)]
pub struct OwnedMeta {
    bytes: Box<[u8]>,
    descriptor: Meta<'static>,
    sections: [std::ops::Range<usize>; 7],
}
impl OwnedMeta {
    /// Copies the metadata out of `input` (the metadata alone or a whole
    /// chunk), keeping only the `meta_len` bytes the header declares.
    pub fn new(input: &[u8]) -> Result<Self> {
        let n = parse_meta(input)?.meta_len();
        Self::from_vec(input[..n].to_vec())
    }
    /// Validates `bytes` and takes ownership of the metadata prefix, dropping
    /// anything past `meta_len`.
    ///
    /// The section ranges recorded here are byte offsets into those same
    /// bytes, computed while the slices returned by [`parse_meta`] are still
    /// borrowed; truncation cannot invalidate them because every section lies
    /// below `meta_len`.
    pub fn from_vec(mut bytes: Vec<u8>) -> Result<Self> {
        let m = parse_meta(&bytes)?;
        let extent = |s: &[u8]| {
            let start = s.as_ptr() as usize - bytes.as_ptr() as usize;
            start..start + s.len()
        };
        let sections = [m.models, m.dims, m.modes.unwrap_or(&bytes[..0]), m.offsets, m.lengths, m.kinds, m.nnz].map(extent);
        let descriptor = Meta {
            model_key: m.model_key,
            renumbered: m.renumbered,
            n_cells: m.n_cells,
            n_dims: m.n_dims,
            value_width: m.value_width,
            max_lanes: m.max_lanes,
            lane_min: m.lane_min,
            page_log2: m.page_log2,
            max_k: m.max_k,
            max_nnz: m.max_nnz,
            max_elems: m.max_elems,
            nnz_total: m.nnz_total,
            meta_len: m.meta_len,
            records_start: m.records_start,
            records_len: m.records_len,
            models: &[],
            dims: &[],
            modes: m.modes.map(|_| &[][..]),
            offsets: &[],
            lengths: &[],
            kinds: &[],
            nnz: &[],
        };
        bytes.truncate(m.meta_len());
        Ok(Self { bytes: bytes.into_boxed_slice(), descriptor, sections })
    }
    /// The metadata bytes, exactly `meta_len` long.
    pub fn bytes(&self) -> &[u8] { &self.bytes }
    /// A descriptor borrowing `self`; the section ranges stored at
    /// construction become slices of the same bytes, so repeated calls are
    /// free and never re-parse.
    ///
    /// The returned view lives as long as the borrow of `self`, not as long as
    /// the owning `OwnedMeta`.
    #[inline]
    pub fn view(&self) -> Meta<'_> {
        // SAFETY: the sections were taken from these bytes when they were
        // validated and all lie below `meta_len`, the length these bytes were
        // truncated to; `bytes` is immutable, so the ranges stay in bounds for
        // as long as the returned slices borrow it.
        let section = |i: usize| unsafe { self.bytes.get_unchecked(self.sections.get_unchecked(i).clone()) };
        Meta {
            model_key: self.descriptor.model_key,
            renumbered: self.descriptor.renumbered,
            n_cells: self.descriptor.n_cells,
            n_dims: self.descriptor.n_dims,
            value_width: self.descriptor.value_width,
            max_lanes: self.descriptor.max_lanes,
            lane_min: self.descriptor.lane_min,
            page_log2: self.descriptor.page_log2,
            max_k: self.descriptor.max_k,
            max_nnz: self.descriptor.max_nnz,
            max_elems: self.descriptor.max_elems,
            nnz_total: self.descriptor.nnz_total,
            meta_len: self.descriptor.meta_len,
            records_start: self.descriptor.records_start,
            records_len: self.descriptor.records_len,
            models: section(0),
            dims: section(1),
            modes: self.descriptor.modes.map(|_| section(2)),
            offsets: section(3),
            lengths: section(4),
            kinds: section(5),
            nnz: section(6),
        }
    }
}

/// Parses metadata from the start of `bytes` (the metadata alone or a whole
/// chunk). Checks the header and section sizes; per-cell table entries are
/// checked where they are read.
///
/// One pass of O(1) checks: magic, version, reserved bits, the dim count, value
/// width and lane and page parameters, the per-dimension and per-cell maxima
/// against each other, the model frequencies (each sums to 0 or to the rANS
/// total, and no symbol above `MAX_SYMBOL` is used), the layout against
/// `meta_len`, the chunk extent against the address space, and the non-zero
/// table (first entry 0, last `nnz_total`). Record bytes are not read, so
/// metadata alone can be validated before any record is touched.
///
/// The returned view also carries `model_key`, a hash of the coding inputs
/// (`n_dims`, renumbering, models, dim sources, column modes) that lets the
/// decode tables check they were built for this chunk.
pub fn parse_meta(bytes: &[u8]) -> Result<Meta<'_>> {
    let h = bytes
        .get(..HEADER_LEN)
        .ok_or(Error::CorruptMeta("shorter than the header"))?;
    if h[0..4] != MAGIC {
        return Err(Error::CorruptMeta("bad magic"));
    }
    if u16_at(h, 4) != VERSION {
        return Err(Error::CorruptMeta("unsupported version"));
    }
    let flags = u16_at(h, 6);
    if flags & !(FLAG_RENUMBERED | FLAG_MODES) != 0 || u64_at(h, 56) != 0 {
        return Err(Error::CorruptMeta("reserved bits set"));
    }
    let renumbered = flags & FLAG_RENUMBERED != 0;
    let has_modes = flags & FLAG_MODES != 0;
    let (n_cells, n_dims) = (u32_at(h, 8), u32_at(h, 12));
    let value_width = h[16] as usize;
    let (max_lanes, lane_min_log2, page_log2) = (h[17] as u32, h[18] as u32, h[19] as u32);
    let (max_k, max_nnz, max_elems) = (u32_at(h, 20), u32_at(h, 24), u32_at(h, 28));
    let (nnz_total, meta_len, records_len) = (u64_at(h, 32), u64_at(h, 40), u64_at(h, 48));
    if n_dims == 0 {
        return Err(Error::CorruptMeta("no dims"));
    }
    // Ids, ranks and dictionary sizes are below `n_dims`; SIMD gathers take
    // them as signed 32-bit indices.
    if n_dims > i32::MAX as u32 {
        return Err(Error::CorruptMeta("more than 2^31 - 1 dims"));
    }
    if !matches!(value_width, 1 | 2 | 4 | 8) {
        return Err(Error::CorruptMeta("value width must be 1, 2, 4 or 8"));
    }
    if !max_lanes.is_power_of_two() || max_lanes > 32 || lane_min_log2 > 16 {
        return Err(Error::CorruptMeta("bad lane parameters"));
    }
    if page_log2 != 0 && !(MIN_PAGE_LOG2..=MAX_PAGE_LOG2).contains(&page_log2) {
        return Err(Error::CorruptMeta("bad page size"));
    }
    if max_nnz > n_dims || max_elems > n_dims || max_nnz > max_elems || max_k > max_elems {
        return Err(Error::CorruptMeta("bad per-cell maxima"));
    }
    let modes_len = if has_modes {
        n_dims as usize * value_width
    } else {
        0
    };
    let layout = Layout::new(n_cells as usize, n_dims, renumbered, modes_len)
        .ok_or(Error::CorruptMeta("chunk too large"))?;
    if meta_len != layout.meta_len as u64 {
        return Err(Error::CorruptMeta(
            "metadata length does not match the layout",
        ));
    }
    let meta = bytes
        .get(..layout.meta_len)
        .ok_or(Error::CorruptMeta("truncated metadata"))?;
    let models = &meta[HEADER_LEN..layout.dims_off];
    for m in 0..N_MODELS {
        let freq = &models[m * MODEL_BYTES..(m + 1) * MODEL_BYTES];
        let (mut sum, mut beyond) = (0u32, false);
        for s in 0..256 {
            let f = u16_at(freq, 2 * s) as u32;
            sum += f;
            beyond |= s > MAX_SYMBOL && f != 0;
        }
        if (sum != 0 && sum != crate::scans::rans::M) || beyond {
            return Err(Error::CorruptMeta("bad symbol frequencies"));
        }
    }
    let n = n_cells as usize;
    let records_start =
        records_start(layout.meta_len, page_log2).ok_or(Error::CorruptMeta("chunk too large"))?;
    if (records_start as u64)
        .checked_add(records_len)
        .and_then(|end| usize::try_from(end).ok())
        .is_none()
    {
        return Err(Error::CorruptMeta("chunk extent exceeds the address space"));
    }
    if nnz_total > n_cells as u64 * max_nnz as u64 {
        return Err(Error::CorruptMeta(
            "non-zero total exceeds the per-cell maximum",
        ));
    }
    let dims = &meta[layout.dims_off..layout.modes_off];
    let modes = has_modes.then(|| &meta[layout.modes_off..layout.modes_off + modes_len]);
    // Table association is computed once while parsing, not over the dim
    // permutation on every record. Packing and row extents do not change it.
    let mut key = std::collections::hash_map::DefaultHasher::new();
    (n_dims, renumbered, models, dims, modes).hash(&mut key);
    let m = Meta {
        model_key: key.finish(),
        renumbered,
        n_cells,
        n_dims,
        value_width,
        max_lanes,
        lane_min: 1 << lane_min_log2,
        page_log2,
        max_k,
        max_nnz,
        max_elems,
        nnz_total,
        meta_len: layout.meta_len,
        records_start,
        records_len,
        models,
        dims,
        modes,
        offsets: &meta[layout.offsets_off..layout.offsets_off + 8 * n],
        lengths: &meta[layout.lengths_off..layout.lengths_off + 4 * n],
        kinds: &meta[layout.kinds_off..layout.kinds_off + n],
        nnz: &meta[layout.nnz_off..layout.nnz_off + 8 * (n + 1)],
    };
    if u64_at(m.nnz, 0) != 0 || u64_at(m.nnz, 8 * n) != nnz_total {
        return Err(Error::CorruptMeta("non-zero table does not span nnz_total"));
    }
    Ok(m)
}

impl<'a> Meta<'a> {
    /// Hash of the coding inputs a chunk's models and dim order depend on;
    /// tables built from one chunk's metadata must match this to be used on
    /// another (see [`Error::TablesMismatch`]).
    pub(crate) fn model_key(&self) -> u64 {
        self.model_key
    }

    /// Cells (rows) in the chunk.
    pub fn n_cells(&self) -> usize {
        self.n_cells as usize
    }
    /// Dims (columns) of the chunk: its coded alphabet size.
    pub fn n_dims(&self) -> u32 {
        self.n_dims
    }
    /// Bytes per stored value (1, 2, 4 or 8), also the record value stride.
    pub fn value_width(&self) -> usize {
        self.value_width
    }
    /// Byte width of raw record indices.
    pub fn index_width(&self) -> usize {
        id_width(self.n_dims)
    }
    /// Whether coded ids are a permutation of the source ids (`source_dim`
    /// maps them back); without renumbering they are the source ids.
    pub fn renumbered(&self) -> bool {
        self.renumbered
    }
    /// The column-mode table, `value_width` bytes per dim in coded order,
    /// when the chunk has one (see mode-coded records).
    pub fn column_modes(&self) -> Option<&'a [u8]> {
        self.modes
    }
    /// Most nnz any record in the chunk declares: stored entries for a sparse
    /// record (explicit zeros included), non-zeros for a dense one.
    pub fn max_nnz(&self) -> usize {
        self.max_nnz as usize
    }
    /// Most elements any record stores (`n_dims` when a record is dense).
    pub fn max_elems(&self) -> usize {
        self.max_elems as usize
    }
    /// Largest per-cell dictionary in the chunk.
    pub fn max_k(&self) -> usize {
        self.max_k as usize
    }
    /// The chunk's declared nnz: stored entries summed over sparse records,
    /// non-zeros over dense ones.
    pub fn nnz_total(&self) -> u64 {
        self.nnz_total
    }
    /// Bytes of metadata.
    pub fn meta_len(&self) -> usize {
        self.meta_len
    }
    /// Where the records region starts in a chunk (page-aligned when paged).
    pub fn records_start(&self) -> usize {
        self.records_start
    }
    /// The page size records are packed to, if any.
    pub fn page_size(&self) -> Option<u32> {
        (self.page_log2 != 0).then(|| 1 << self.page_log2)
    }
    /// Bytes of the records region: from [`Meta::records_start`] through the
    /// last record, gaps between records included.
    pub fn records_len(&self) -> u64 {
        self.records_len
    }
    /// Upper bound the encoder set on the lanes of any rANS stream, as stored
    /// in the header; a record's streams never use more lanes than their
    /// element count allows (`rans::lanes_for`).
    pub(crate) fn max_lanes(&self) -> u32 {
        self.max_lanes
    }
    /// Fewest symbols per lane the encoder used, so a stream of `n` symbols
    /// gets the same lane count the encoder chose (`rans::lanes_for`).
    pub(crate) fn lane_min(&self) -> u32 {
        self.lane_min
    }

    /// The records region of a chunk whose metadata this is.
    pub fn records_of<'c>(&self, chunk: &'c [u8]) -> Result<&'c [u8]> {
        let end = (self.records_start as u64)
            .checked_add(self.records_len)
            .and_then(|e| usize::try_from(e).ok());
        end.and_then(|e| chunk.get(self.records_start..e))
            .ok_or(Error::CorruptMeta("chunk shorter than its records"))
    }

    /// Bytes of the whole chunk.
    pub fn chunk_len(&self) -> u64 {
        self.records_start as u64 + self.records_len
    }

    fn check_cell(&self, cell: usize) -> Result<()> {
        if cell >= self.n_cells as usize {
            return Err(Error::RowOutOfRange {
                row: cell,
                n_rows: self.n_cells as usize,
            });
        }
        Ok(())
    }

    /// Declared nnz of a cell (stored entries for sparse, non-zeros for
    /// dense), as the difference of two entries of the cumulative table,
    /// checked to be non-decreasing up to `nnz_total` and within `max_nnz`.
    pub fn cell_nnz(&self, cell: usize) -> Result<usize> {
        self.check_cell(cell)?;
        let (a, b) = (u64_at(self.nnz, 8 * cell), u64_at(self.nnz, 8 * cell + 8));
        if a > b || b > self.nnz_total || b - a > self.max_nnz as u64 {
            return Err(Error::CorruptMeta("bad non-zero table entry"));
        }
        Ok((b - a) as usize)
    }

    /// Layout of a cell's record from the metadata alone (an empty cell is
    /// sparse).
    pub fn cell_layout(&self, cell: usize) -> Result<RecordLayout> {
        self.check_cell(cell)?;
        Ok(if self.kinds[cell] & 3 == IDX_DENSE {
            RecordLayout::Dense
        } else {
            RecordLayout::Sparse
        })
    }

    /// A cell's record flags byte (0 for an empty cell): its layout and
    /// index and value codings, for callers that model per-record costs.
    pub fn cell_kind(&self, cell: usize) -> Result<u8> {
        self.check_cell(cell)?;
        Ok(self.kinds[cell])
    }

    /// Byte range of a cell's record within the records region, relative to
    /// [`Meta::records_start`]; an empty cell gives an empty range.
    pub fn record_range(&self, cell: usize) -> Result<std::ops::Range<usize>> {
        self.check_cell(cell)?;
        let a = u64_at(self.offsets, 8 * cell);
        let len = u32_at(self.lengths, 4 * cell) as u64;
        match a.checked_add(len) {
            Some(b) if b <= self.records_len => Ok(a as usize..b as usize),
            _ => Err(Error::CorruptMeta("bad record extent")),
        }
    }

    /// The record range, element count and flags of a cell, from
    /// the per-cell tables without validating them: metadata of a chunk the
    /// caller trusts.
    ///
    /// Also skips the arithmetic checks of [`Meta::record_range`] and
    /// [`Meta::cell_nnz`]: the range is not compared against `records_len` and
    /// the nnz difference not against `max_nnz`. Debug builds compare the
    /// range with `record_range` and assert the cell bound.
    ///
    /// # Safety
    /// `cell < n_cells`.
    #[inline(always)]
    pub(crate) unsafe fn cell_unchecked(&self, cell: usize) -> (std::ops::Range<usize>, usize, u8) {
        debug_assert!(cell < self.n_cells as usize);
        // SAFETY: the parser checked that every per-cell table holds
        // `n_cells` entries (and the nnz table one more), so the reads at
        // `8 * cell`, `4 * cell` and `cell` stay inside their sections; the
        // caller guarantees `cell < n_cells`.
        unsafe {
            let read8 = |b: &[u8], at: usize| u64::from_le_bytes(b.as_ptr().add(at).cast::<[u8; 8]>().read_unaligned());
            let a = read8(self.offsets, 8 * cell) as usize;
            let len = u32::from_le_bytes(self.lengths.as_ptr().add(4 * cell).cast::<[u8; 4]>().read_unaligned()) as usize;
            let nnz = (read8(self.nnz, 8 * cell + 8) - read8(self.nnz, 8 * cell)) as usize;
            debug_assert_eq!(Ok(a..a + len), self.record_range(cell));
            (a..a + len, nnz, *self.kinds.get_unchecked(cell))
        }
    }

    /// A cell's record within the records region, i.e. `records[record_range]`
    /// for a region that holds the whole chunk's records.
    pub fn record<'r>(&self, records: &'r [u8], cell: usize) -> Result<&'r [u8]> {
        let r = self.record_range(cell)?;
        records.get(r).ok_or(Error::CorruptMeta(
            "records region shorter than the offset table",
        ))
    }

    /// Symbol frequencies of model `model` ([`GAP`], [`RANK`] or [`DICT`]) as
    /// a full 256-entry alphabet, for building the decode and encode tables.
    pub(crate) fn freqs(&self, model: usize) -> [u16; 256] {
        let b = &self.models[model * MODEL_BYTES..(model + 1) * MODEL_BYTES];
        std::array::from_fn(|s| u16_at(b, 2 * s))
    }

    /// Source dim of coded id `c`; only for renumbered chunks.
    pub(crate) fn source_dim(&self, c: usize) -> u32 {
        if id_width(self.n_dims) == 2 {
            u16_at(self.dims, 2 * c) as u32
        } else {
            u32_at(self.dims, 4 * c)
        }
    }
}

/// How a record stores its elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordLayout {
    /// The stored entries: coded ids and their values, explicit zeros included.
    Sparse,
    /// Every dim's value in coded order, zeros included.
    Dense,
}

/// Writes the offset and length tables, one row per extent: `off` counts from
/// the records region start, `len` is the record's byte length (0 for an empty
/// cell). `meta` must hold at least `layout.meta_len` bytes.
pub(crate) fn write_extents(meta: &mut [u8], layout: &Layout, extents: &[(u64, u32)]) {
    for (i, &(off, len)) in extents.iter().enumerate() {
        meta[layout.offsets_off + 8 * i..layout.offsets_off + 8 * i + 8]
            .copy_from_slice(&off.to_le_bytes());
        meta[layout.lengths_off + 4 * i..layout.lengths_off + 4 * i + 4]
            .copy_from_slice(&len.to_le_bytes());
    }
}

/// Writes the record kinds table.
pub(crate) fn write_kinds(meta: &mut [u8], layout: &Layout, kinds: &[u8]) {
    meta[layout.kinds_off..layout.kinds_off + kinds.len()].copy_from_slice(kinds);
}

/// Writes the three frequency tables at their fixed offsets after the header;
/// `out` must hold at least `HEADER_LEN + N_MODELS * MODEL_BYTES` bytes.
pub(crate) fn write_models(out: &mut [u8], freqs: &[[u16; 256]; N_MODELS]) {
    for (m, f) in freqs.iter().enumerate() {
        for (s, v) in f.iter().enumerate() {
            let at = HEADER_LEN + m * MODEL_BYTES + 2 * s;
            out[at..at + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
}

// ---- records ----

/// Bytes [`put_varint`] writes for `v`: seven payload bits per byte.
pub(crate) fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// Writes a varint at `out[*at..]`; the caller has reserved room.
pub(crate) fn put_varint(out: &mut [u8], at: &mut usize, mut v: u64) {
    while v >= 0x80 {
        out[*at] = v as u8 | 0x80;
        *at += 1;
        v >>= 7;
    }
    out[*at] = v as u8;
    *at += 1;
}

/// Reads a trusted record: varints and sections lie inside it (a damaged
/// record is undefined behavior; debug builds check every read).
struct Cursor<'r> {
    b: &'r [u8],
    at: usize,
}

impl<'r> Cursor<'r> {
    /// Decodes a varint, low 7 bits first, most significant group last; the
    /// high bit of a byte continues the encoding.
    #[inline(always)]
    fn varint(&mut self) -> u64 {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            debug_assert!(self.at < self.b.len(), "truncated varint");
            // SAFETY: a well-formed record holds every byte of its varints.
            let byte = unsafe { *self.b.get_unchecked(self.at) };
            self.at += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return v;
            }
            shift += 7;
            debug_assert!(shift < 64, "varint too long");
        }
    }

    /// A section length: a varint narrowed to `usize`.
    #[inline(always)]
    fn len(&mut self) -> usize {
        self.varint() as usize
    }

    /// The next `n` bytes of the record, borrowed from it; the cursor moves
    /// past them.
    #[inline(always)]
    fn take(&mut self, n: usize) -> &'r [u8] {
        debug_assert!(self.at + n <= self.b.len(), "section past the record end");
        // SAFETY: a well-formed record holds every section it declares.
        let s = unsafe { self.b.get_unchecked(self.at..self.at + n) };
        self.at += n;
        s
    }
}

/// A record's index section as [`parse_record`] splits it.
pub(crate) enum IdxPart<'r> {
    /// One coded id per element, `index_width` bytes each, ascending.
    Raw(&'r [u8]),
    /// Every dim in coded order; no index section.
    Dense,
    /// `bits_len` escape bytes: the first `carried` ride in the gap
    /// stream's initial states, the rest are `tail`. `rans` is the gap code
    /// stream itself; `carried <= bits_len`, and `carried` is at most that
    /// stream's payload capacity.
    Gap {
        rans: &'r [u8],
        bits_len: usize,
        carried: usize,
        tail: &'r [u8],
    },
}

/// A record's value section as [`parse_record`] splits it.
pub(crate) enum ValPart<'r> {
    /// One value per element, `value_width` bytes each.
    Raw(&'r [u8]),
    /// The pool (dict rans, dict escape bits, rank escape bits) of
    /// `dict_len + dict_bits_len + rank_bits_len` bytes: the first `carried`
    /// ride in the rank stream's initial states, the rest are `tail`. With
    /// `modes`, rank 0 is the column mode and rank `r` dictionary entry `r - 1`.
    ///
    /// `k` is the distinct value count of the dictionary, 0 only for a
    /// mode-coded record whose values are all their column mode. `rank` is the
    /// rank stream: present for a mode-coded record whenever `k >= 1`, and for
    /// a dictionary-coded one only when `k > 1`, since a single entry needs no
    /// ranks. With no rank stream nothing is carried and the whole pool is
    /// `tail`.
    Dict {
        modes: bool,
        k: usize,
        dict_len: usize,
        dict_bits_len: usize,
        rank: Option<&'r [u8]>,
        rank_bits_len: usize,
        carried: usize,
        tail: &'r [u8],
    },
}

impl ValPart<'_> {
    /// Bytes of the dictionary pool once its carried prefix is rejoined with
    /// `tail`: `dict_len + dict_bits_len + rank_bits_len`; 0 for raw values.
    pub(crate) fn pool_len(&self) -> usize {
        match *self {
            ValPart::Dict {
                dict_len,
                dict_bits_len,
                rank_bits_len,
                ..
            } => dict_len + dict_bits_len + rank_bits_len,
            ValPart::Raw(_) => 0,
        }
    }
}

/// A record split into its sections, with the element counts needed to
/// interpret them.
pub(crate) struct RecordView<'r> {
    /// Element count: stored entries for sparse, non-zeros for dense.
    pub(crate) nnz: usize,
    /// Stored elements: `nnz`, or `n_dims` for a dense record.
    pub(crate) elems: usize,
    /// Whether the record stores every dim (index mode dense).
    pub(crate) dense: bool,
    /// Index section.
    pub(crate) idx: IdxPart<'r>,
    /// Value section.
    pub(crate) val: ValPart<'r>,
}

/// Element count of a record from its header alone: the varint after the
/// flags byte (0 for an empty record). For a sparse record it counts stored
/// entries (explicit zeros included); for a dense record, non-zero values.
/// A caller can compare it with metadata before committing to a decode.
///
/// Unlike the record decoders, this reads only the header and checks every
/// byte it reads, so it is safe on arbitrary bytes: a truncated or overlong
/// count is [`Error::CorruptRecord`].
pub fn record_nnz(rec: &[u8]) -> Result<usize> {
    let Some(header) = rec.get(1..) else {
        return Ok(0);
    };
    let mut v = 0u64;
    for (i, &byte) in header.iter().take(10).enumerate() {
        let group = (byte & 0x7f) as u64;
        if i == 9 && group > 1 {
            return Err(Error::CorruptRecord("record element count overflows u64"));
        }
        v |= group << (7 * i);
        if byte & 0x80 == 0 {
            return usize::try_from(v).map_err(|_| Error::CorruptRecord("record element count overflows usize"));
        }
    }
    Err(Error::CorruptRecord("truncated record element count"))
}

/// Splits a non-empty record into its sections. Records are trusted: the
/// structure is only checked in debug builds. `lanes` gives the lane count
/// of a stream of `n` symbols.
///
/// Every section length is a varint, and every section is taken in the order
/// the layout lists it, so the cursor ends exactly at the record's end (a
/// debug assertion otherwise). `lanes(elems)` sizes the carried prefix of the
/// escape bits and of the value pool: carrying is a property of the encoder's
/// stream geometry, not of the record, so the caller supplies the same lane
/// count the encoder used.
#[inline]
pub(crate) fn parse_record(
    rec: &[u8],
    n_dims: u32,
    iw: usize,
    vw: usize,
    lanes: impl Fn(usize) -> usize,
) -> Result<RecordView<'_>> {
    debug_assert!(!rec.is_empty());
    // SAFETY: the record is non-empty.
    let flags = unsafe { *rec.get_unchecked(0) };
    debug_assert_eq!(flags & 0xf0, 0, "reserved flag bits set");
    let mut c = Cursor { b: rec, at: 1 };
    let nnz = c.len();
    debug_assert!(nnz > 0, "non-empty record with no non-zeros");
    let dense = flags & 3 == IDX_DENSE;
    let elems = if dense { n_dims as usize } else { nnz };
    debug_assert!(nnz <= elems, "more non-zeros than dims");
    let capacity = payload_capacity(lanes(elems));
    let idx = match flags & 3 {
        IDX_DENSE => IdxPart::Dense,
        IDX_RAW => IdxPart::Raw(c.take(elems * iw)),
        _ => {
            debug_assert_eq!(flags & 3, IDX_GAP, "unknown index mode");
            let (r, bits_len) = (c.len(), c.len());
            let carried = bits_len.min(capacity);
            IdxPart::Gap {
                rans: c.take(r),
                bits_len,
                carried,
                tail: c.take(bits_len - carried),
            }
        }
    };
    let val = match (flags >> 2) & 3 {
        VAL_RAW => ValPart::Raw(c.take(elems * vw)),
        VAL_DICT => {
            let k = c.len();
            debug_assert!(k > 0 && k <= elems, "bad dictionary size");
            let (dict_len, dict_bits_len) = (c.len(), c.len());
            let (rank, rank_bits_len) = if k > 1 {
                let (rr, rb) = (c.len(), c.len());
                (Some(c.take(rr)), rb)
            } else {
                (None, 0)
            };
            let pool = dict_len + dict_bits_len + rank_bits_len;
            let carried = if rank.is_some() { pool.min(capacity) } else { 0 };
            ValPart::Dict {
                modes: false,
                k,
                dict_len,
                dict_bits_len,
                rank,
                rank_bits_len,
                carried,
                tail: c.take(pool - carried),
            }
        }
        _ => {
            debug_assert_eq!((flags >> 2) & 3, VAL_MODE, "unknown value mode");
            let k = c.len();
            debug_assert!(k <= elems, "bad dictionary size");
            if k == 0 {
                ValPart::Dict {
                    modes: true,
                    k,
                    dict_len: 0,
                    dict_bits_len: 0,
                    rank: None,
                    rank_bits_len: 0,
                    carried: 0,
                    tail: &[],
                }
            } else {
                let (dict_len, dict_bits_len) = (c.len(), c.len());
                let (rr, rank_bits_len) = (c.len(), c.len());
                let rank = Some(c.take(rr));
                let pool = dict_len + dict_bits_len + rank_bits_len;
                let carried = pool.min(capacity);
                ValPart::Dict {
                    modes: true,
                    k,
                    dict_len,
                    dict_bits_len,
                    rank,
                    rank_bits_len,
                    carried,
                    tail: c.take(pool - carried),
                }
            }
        }
    };
    debug_assert_eq!(c.at, rec.len(), "trailing bytes");
    Ok(RecordView {
        nnz,
        elems,
        dense,
        idx,
        val,
    })
}
