//! Borrowed inputs, selections and option values shared by encode and decode.

use std::ops::Range;

use crate::error::{Error, Result};

/// Suggested threshold for [`Thresholds::indices`].
pub const DEFAULT_MU_INDICES: f32 = 0.6;
/// Suggested threshold for [`Thresholds::values`].
pub const DEFAULT_MU_VALUES: f32 = 0.6;
/// Suggested [`EncodeOptions::max_lanes`].
pub const DEFAULT_MAX_LANES: u8 = 32;
/// Suggested [`EncodeOptions::lane_min`].
pub const DEFAULT_LANE_MIN: u32 = 32;
/// Suggested [`EncodeOptions::column_modes`].
pub const DEFAULT_MODE_SHARE: f32 = 0.5;

/// Gene indices of a matrix or a cell, in their stored width.
#[derive(Clone, Copy, Debug)]
pub enum Indices<'a> {
    /// 16-bit indices; every entry is in the `u32` range.
    U16(&'a [u16]),
    /// 32-bit indices.
    U32(&'a [u32]),
    /// Signed 32-bit indices; negative entries are rejected.
    I32(&'a [i32]),
    /// Signed 64-bit indices; entries outside the `u32` range are rejected.
    I64(&'a [i64]),
    /// Unsigned 64-bit indices; entries outside the `u32` range are rejected.
    U64(&'a [u64]),
}

impl<'a> Indices<'a> {
    /// Number of indices.
    pub fn len(&self) -> usize {
        match self {
            Indices::U16(v) => v.len(),
            Indices::U32(v) => v.len(),
            Indices::I32(v) => v.len(),
            Indices::I64(v) => v.len(),
            Indices::U64(v) => v.len(),
        }
    }

    /// Whether the selection holds no index.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sub-range `r`, or `None` when it leaves the indices.
    pub(crate) fn slice(&self, r: Range<usize>) -> Option<Indices<'a>> {
        Some(match self {
            Indices::U16(v) => Indices::U16(v.get(r)?),
            Indices::U32(v) => Indices::U32(v.get(r)?),
            Indices::I32(v) => Indices::I32(v.get(r)?),
            Indices::I64(v) => Indices::I64(v.get(r)?),
            Indices::U64(v) => Indices::U64(v.get(r)?),
        })
    }

    /// Index `i` as a gene id; negative or out-of-range values are an error.
    pub(crate) fn checked(&self, i: usize) -> Result<u32> {
        let value = match self {
            Self::U16(v) => return Ok(v[i] as u32),
            Self::U32(v) => return Ok(v[i]),
            Self::I32(v) => u64::try_from(v[i]),
            Self::I64(v) => u64::try_from(v[i]),
            Self::U64(v) => {
                return u32::try_from(v[i])
                    .map_err(|_| Error::InvalidInput("index outside u32 range"));
            }
        }
        .map_err(|_| Error::InvalidInput("negative gene index"))?;
        u32::try_from(value).map_err(|_| Error::InvalidInput("index outside u32 range"))
    }
}

/// Borrowed CSR row offsets; signed inputs are checked before indexing.
#[derive(Clone, Copy, Debug)]
pub enum Offsets<'a> {
    /// 32-bit offsets.
    U32(&'a [u32]),
    /// 64-bit offsets.
    U64(&'a [u64]),
    /// Signed 32-bit offsets; negative entries are rejected.
    I32(&'a [i32]),
    /// Signed 64-bit offsets; negative entries are rejected.
    I64(&'a [i64]),
}

impl Offsets<'_> {
    /// Number of offsets, one more than the rows they describe.
    pub fn len(&self) -> usize {
        match self {
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
        }
    }
    /// Whether there is no offset, i.e. no row.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Offset `i` as an unsigned value; a negative entry is an error.
    pub fn get(&self, i: usize) -> Result<u64> {
        match self {
            Self::U32(v) => Ok(v[i] as u64),
            Self::U64(v) => Ok(v[i]),
            Self::I32(v) => {
                u64::try_from(v[i]).map_err(|_| Error::InvalidInput("negative CSR offset"))
            }
            Self::I64(v) => {
                u64::try_from(v[i]).map_err(|_| Error::InvalidInput("negative CSR offset"))
            }
        }
    }
}

/// Canonical CSR with row offsets borrowed in the caller's integer type.
#[derive(Clone, Copy, Debug)]
pub struct CsrView<'a> {
    /// Genes each row may name (ids lie in `0..n_genes`).
    pub n_genes: u32,
    /// Bytes per value: 1, 2, 4 or 8.
    pub value_width: usize,
    /// `n_rows + 1` row offsets, in any integer type.
    pub indptr: Offsets<'a>,
    /// Gene indices of every row, in row order.
    pub indices: Indices<'a>,
    /// Value bytes, `value_width` per index.
    pub data: &'a [u8],
}

impl<'a> CsrView<'a> {
    /// Rows the offsets describe; an empty `indptr` has none.
    pub fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }
    /// Row `row` as a cell, checking the offsets against the indices and data.
    pub(crate) fn cell(&self, row: usize) -> Result<CellRef<'a>> {
        if row >= self.n_rows() {
            return Err(Error::RowOutOfRange {
                row,
                n_rows: self.n_rows(),
            });
        }
        let a = usize::try_from(self.indptr.get(row)?)
            .map_err(|_| Error::InvalidInput("CSR offset too large"))?;
        let b = usize::try_from(self.indptr.get(row + 1)?)
            .map_err(|_| Error::InvalidInput("CSR offset too large"))?;
        if a > b {
            return Err(Error::InvalidInput("indptr decreases"));
        }
        let indices = self
            .indices
            .slice(a..b)
            .ok_or(Error::InvalidInput("indptr beyond indices"))?;
        let start = a
            .checked_mul(self.value_width)
            .ok_or(Error::InvalidInput("CSR offset too large"))?;
        let end = b
            .checked_mul(self.value_width)
            .ok_or(Error::InvalidInput("CSR offset too large"))?;
        let values = self
            .data
            .get(start..end)
            .ok_or(Error::InvalidInput("indptr beyond data"))?;
        Ok(CellRef { indices, values })
    }
}

/// A canonical CSR matrix: every row's indices strictly ascending and below
/// `n_genes`; `data` holds `value_width` little-endian bytes per element.
/// Values are opaque bit patterns, so any 1/2/4/8-byte type round-trips.
#[derive(Clone, Copy, Debug)]
pub struct CsrRef<'a> {
    /// Genes each row may name (ids lie in `0..n_genes`).
    pub n_genes: u32,
    /// Bytes per value: 1, 2, 4 or 8.
    pub value_width: usize,
    /// `n_rows + 1` row offsets.
    pub indptr: &'a [u64],
    /// Gene indices of every row, in row order.
    pub indices: Indices<'a>,
    /// Value bytes, `value_width` per index.
    pub data: &'a [u8],
}

impl<'a> CsrRef<'a> {
    /// Rows the offsets describe; an empty `indptr` has none.
    pub fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Row `row` as a cell, checking the offsets against the indices and data.
    pub(crate) fn cell(&self, row: usize) -> Result<CellRef<'a>> {
        let n_rows = self.n_rows();
        if row >= n_rows {
            return Err(Error::RowOutOfRange { row, n_rows });
        }
        let (a, b) = (self.indptr[row], self.indptr[row + 1]);
        if a > b {
            return Err(Error::InvalidInput("indptr decreases"));
        }
        let (a, b) = (usize::try_from(a), usize::try_from(b));
        let (Ok(a), Ok(b)) = (a, b) else {
            return Err(Error::InvalidInput("indptr beyond address space"));
        };
        let indices = self
            .indices
            .slice(a..b)
            .ok_or(Error::InvalidInput("indptr beyond indices"))?;
        let w = self.value_width;
        let values = self
            .data
            .get(a * w..b * w)
            .ok_or(Error::InvalidInput("indptr beyond data"))?;
        Ok(CellRef { indices, values })
    }
}

/// One cell: strictly ascending gene indices and their values
/// (`indices.len() * value_width` bytes).
#[derive(Clone, Copy, Debug)]
pub struct CellRef<'a> {
    /// Strictly ascending gene ids.
    pub indices: Indices<'a>,
    /// One value per index, `indices.len() * value_width` bytes.
    pub values: &'a [u8],
}

/// A row-major dense matrix: `n_genes` values of `value_width` little-endian
/// bytes per row. A value whose bytes are all zero is a zero.
#[derive(Clone, Copy, Debug)]
pub struct DenseRef<'a> {
    /// Values per row.
    pub n_genes: u32,
    /// Bytes per value: 1, 2, 4 or 8.
    pub value_width: usize,
    /// `n_rows * n_genes * value_width` bytes.
    pub data: &'a [u8],
}

impl<'a> DenseRef<'a> {
    /// Whole rows in `data`; a partial trailing row is not counted.
    pub fn n_rows(&self) -> usize {
        self.data.len() / (self.n_genes as usize * self.value_width).max(1)
    }

    /// Row `row`, `n_genes * value_width` bytes.
    pub(crate) fn row(&self, row: usize) -> Result<&'a [u8]> {
        let w = self.n_genes as usize * self.value_width;
        let n_rows = self.n_rows();
        if row >= n_rows {
            return Err(Error::RowOutOfRange { row, n_rows });
        }
        Ok(&self.data[row * w..(row + 1) * w])
    }
}

/// An input matrix. Its form does not decide how rows are stored: every row
/// is stored sparse or dense by its own size (see [`LayoutChoice`]).
#[derive(Clone, Copy, Debug)]
pub enum MatrixRef<'a> {
    /// CSR with `u64` offsets.
    Csr(CsrRef<'a>),
    /// CSR with offsets in the caller's integer type.
    CsrView(CsrView<'a>),
    /// Row-major dense.
    Dense(DenseRef<'a>),
}

impl<'a> From<CsrView<'a>> for MatrixRef<'a> {
    fn from(m: CsrView<'a>) -> Self {
        Self::CsrView(m)
    }
}

impl<'a> From<CsrRef<'a>> for MatrixRef<'a> {
    fn from(m: CsrRef<'a>) -> Self {
        MatrixRef::Csr(m)
    }
}
impl<'a> From<&CsrRef<'a>> for MatrixRef<'a> {
    fn from(m: &CsrRef<'a>) -> Self {
        MatrixRef::Csr(*m)
    }
}
impl<'a> From<DenseRef<'a>> for MatrixRef<'a> {
    fn from(m: DenseRef<'a>) -> Self {
        MatrixRef::Dense(m)
    }
}
impl<'a> From<&DenseRef<'a>> for MatrixRef<'a> {
    fn from(m: &DenseRef<'a>) -> Self {
        MatrixRef::Dense(*m)
    }
}

impl<'a> MatrixRef<'a> {
    /// Values per row of the matrix.
    pub fn n_genes(&self) -> u32 {
        match self {
            MatrixRef::Csr(m) => m.n_genes,
            MatrixRef::CsrView(m) => m.n_genes,
            MatrixRef::Dense(m) => m.n_genes,
        }
    }
    /// Bytes per value.
    pub fn value_width(&self) -> usize {
        match self {
            MatrixRef::Csr(m) => m.value_width,
            MatrixRef::CsrView(m) => m.value_width,
            MatrixRef::Dense(m) => m.value_width,
        }
    }
    /// Rows the matrix claims, not counting a partial dense row.
    pub fn n_rows(&self) -> usize {
        match self {
            MatrixRef::Csr(m) => m.n_rows(),
            MatrixRef::CsrView(m) => m.n_rows(),
            MatrixRef::Dense(m) => m.n_rows(),
        }
    }
    /// Row `row`, checked against the matrix's own bounds.
    pub fn row(&self, row: usize) -> Result<RowRef<'a>> {
        Ok(match self {
            MatrixRef::Csr(m) => RowRef::Sparse(m.cell(row)?),
            MatrixRef::CsrView(m) => RowRef::Sparse(m.cell(row)?),
            MatrixRef::Dense(m) => RowRef::Dense(m.row(row)?),
        })
    }
}

/// One input row: sparse (a cell) or dense (`n_genes` values).
#[derive(Clone, Copy, Debug)]
pub enum RowRef<'a> {
    /// Stored elements, normally the non-zeros, with ascending gene ids.
    Sparse(CellRef<'a>),
    /// Every gene, in gene order.
    Dense(&'a [u8]),
}

impl<'a> From<CellRef<'a>> for RowRef<'a> {
    fn from(c: CellRef<'a>) -> Self {
        RowRef::Sparse(c)
    }
}
impl<'a> From<&CellRef<'a>> for RowRef<'a> {
    fn from(c: &CellRef<'a>) -> Self {
        RowRef::Sparse(*c)
    }
}

/// How records lay out their elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutChoice {
    /// Per row, the smaller of the sparse and dense layouts.
    Auto,
    /// Every row stores its non-zeros.
    Sparse,
    /// Every row stores all genes.
    Dense,
}

/// How records code their values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueChoice {
    /// Per row, the smallest coding that beats raw by the value threshold.
    Auto,
    /// Every row stores raw values.
    Raw,
    /// Every row codes a per-row dictionary.
    Dict,
    /// Every row codes against the column modes; the chunk always builds the
    /// table.
    Modes,
}

/// Which rows of a matrix or chunk an operation covers, in the given order.
#[derive(Clone, Copy, Debug)]
pub enum Rows<'a> {
    /// Every row, in order.
    All,
    /// Rows `start..end`.
    Range {
        start: usize,
        end: usize,
    },
    /// The listed rows, in the listed order (repeats allowed).
    List(&'a [u32]),
}

impl Rows<'_> {
    /// Number of selected rows, after checking the selection against `n_rows`.
    pub fn count(&self, n_rows: usize) -> Result<usize> {
        match self {
            Rows::All => Ok(n_rows),
            &Rows::Range { start, end } => {
                if start > end || end > n_rows {
                    return Err(Error::RowOutOfRange {
                        row: end.max(start),
                        n_rows,
                    });
                }
                Ok(end - start)
            }
            Rows::List(l) => {
                if let Some(&row) = l.iter().find(|&&r| r as usize >= n_rows) {
                    return Err(Error::RowOutOfRange {
                        row: row as usize,
                        n_rows,
                    });
                }
                Ok(l.len())
            }
        }
    }

    /// The `i`-th selected row; `i` below a checked [`Rows::count`].
    #[inline(always)]
    pub fn get(&self, i: usize) -> usize {
        match self {
            Rows::All => i,
            Rows::Range { start, .. } => start + i,
            Rows::List(l) => l[i] as usize,
        }
    }
}

/// Gene numbering of decoded indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdSpace {
    /// The input's gene ids, ascending within each cell, with the input's
    /// values: decoding reproduces the input exactly.
    Source,
    /// The chunk's internal ids (frequency rank when renumbered), ascending,
    /// with their values. Records are stored in this order, so nothing is
    /// reordered; callers holding their own gene mapping compose it with the
    /// chunk's gene order once per chunk. Equal to `Source` for chunks
    /// without renumbering.
    Coded,
}

/// A record held in caller pages: `len` bytes starting `start` bytes into
/// the first of `pages`, each `page_size` bytes long (the last may be
/// shorter if the record ends inside it).
#[derive(Clone, Copy, Debug)]
pub struct PagedRecord<'a> {
    /// Pages the record spans, in file order.
    pub pages: &'a [&'a [u8]],
    /// Page size the record was packed for; only the last page may be shorter.
    pub page_size: usize,
    /// Byte offset of the record's first byte in the first page.
    pub start: usize,
    /// Bytes from `start` on, across the pages.
    pub len: usize,
}

/// Whether dense output rows are cleared before the non-zeros are written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// Clear the row first: every gene the record does not store reads as 0.
    Zero,
    /// The caller guarantees the rows are already zero.
    AssumeZeroed,
}

/// Per-cell coding thresholds: a section is coded only when its coded size
/// is at most `mu` times its raw size; otherwise it is stored raw. Each lies
/// in `[0, 1]`; 0 stores everything raw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// Bound on the coded index section over the raw ids.
    pub indices: f32,
    /// Bound on the coded value section over the raw values.
    pub values: f32,
}

impl Thresholds {
    /// Rejects a threshold outside `[0, 1]`.
    pub(crate) fn check(&self) -> Result<()> {
        let ok = |m: f32| (0.0..=1.0).contains(&m);
        if ok(self.indices) && ok(self.values) {
            Ok(())
        } else {
            Err(Error::InvalidOptions("thresholds must lie in [0, 1]"))
        }
    }
}

/// Chunk-level encoding choices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodeOptions {
    /// Number genes by descending presence in the chunk, which concentrates gaps.
    pub renumber: bool,
    /// Size bounds a coded section must not exceed to be kept.
    pub thresholds: Thresholds,
    /// Upper bound on interleaved rANS states per stream: 1, 2, 4, 8, 16 or 32.
    pub max_lanes: u8,
    /// A stream of `n` symbols uses the most lanes `l <= max_lanes` with
    /// `l * lane_min <= n`, one lane at least, so small cells do not pay for
    /// unused states. A power of two in `[1, 65536]`.
    pub lane_min: u32,
    /// Column modes: the chunk stores each gene's most frequent value and
    /// records may code only the values that differ from it. The table is
    /// built when at least this share of the chunk's values equal their
    /// gene's mode and the estimated saving clearly exceeds the table;
    /// otherwise records keep per-cell dictionaries. `None` never builds it.
    /// Each record still chooses its smallest coding. `CpkWriter` builds it
    /// only for float matrices (integer counts code smaller and map faster
    /// without it) unless mode coding is forced.
    pub column_modes: Option<f32>,
    /// Record layout; anything but `Auto` applies to every row.
    pub layout: LayoutChoice,
    /// Value coding; anything but `Auto` applies to every row and ignores
    /// the value threshold.
    pub values: ValueChoice,
}

impl EncodeOptions {
    /// Rejects lane counts, thresholds and mode settings the encoder cannot
    /// honor, so a chunk never depends on undefined options.
    pub(crate) fn check(&self) -> Result<()> {
        self.thresholds.check()?;
        if !self.max_lanes.is_power_of_two() || self.max_lanes > 32 {
            return Err(Error::InvalidOptions(
                "max_lanes must be 1, 2, 4, 8, 16 or 32",
            ));
        }
        if !self.lane_min.is_power_of_two() || self.lane_min > 1 << 16 {
            return Err(Error::InvalidOptions(
                "lane_min must be a power of two in [1, 65536]",
            ));
        }
        if self.column_modes.is_some_and(|s| !(0.0..=1.0).contains(&s)) {
            return Err(Error::InvalidOptions(
                "column mode share must lie in [0, 1]",
            ));
        }
        if self.values == ValueChoice::Modes && self.column_modes.is_none() {
            return Err(Error::InvalidOptions(
                "forced mode coding needs column modes",
            ));
        }
        Ok(())
    }
}

mod sealed {
    /// Seals `IndexOut` to the integer types the decoder writes.
    pub trait Sealed {}
}

/// Integer types decoded gene indices can be written as.
pub trait IndexOut: Copy + sealed::Sealed {
    /// Largest id the type holds; decoding checks the chunk's `n_dims`.
    const MAX: u64;
    /// `v <= MAX`, checked by the caller once per call.
    fn from_id(v: u32) -> Self;
}

/// Integer types a decoded CSR `indptr` can be written as.
pub trait IndptrOut: Copy + sealed::Sealed {
    /// Largest offset the type holds; decoding checks the selected nnz.
    const MAX: u64;
    /// `v <= MAX`, checked by the caller once per call.
    fn from_offset(v: u64) -> Self;
}

macro_rules! out_types {
    ($trait:ident, $f:ident, $arg:ty: $($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl $trait for $t {
            const MAX: u64 = <$t>::MAX as u64;
            #[inline(always)]
            fn $f(v: $arg) -> Self {
                v as $t
            }
        }
    )*};
}
out_types!(IndexOut, from_id, u32: u16, u32, i32, i64, u64);

macro_rules! indptr_types {
    ($($t:ty),*) => {$(
        impl IndptrOut for $t {
            const MAX: u64 = <$t>::MAX as u64;
            #[inline(always)]
            fn from_offset(v: u64) -> Self {
                v as $t
            }
        }
    )*};
}
indptr_types!(u32, u64, i32, i64);
