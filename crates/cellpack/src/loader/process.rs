//! Process functions: a schema's native callbacks turning one cell's row,
//! already mapped to the schema's output dims and converted to its element
//! type, into the cell's fixed-size output. How a function uses the row
//! (copying it, sampling it, ...) is its own business.
//!
//! Row semantics. An output dim whose standard id the cell's matrix has is
//! present: its value is the stored value, and 0 where the record stores
//! none (CSR is built by dropping zeros, so an unstored element is a zero
//! expression value, not a missing one). An output dim the matrix has no
//! column for is missing. A matrix column no output dim takes is dropped.
//! A [`CsrRow`] lists only stored elements, so it cannot tell a present zero
//! from a missing dim; a [`DenseRow`] holds 0 at present zeros and the
//! schema's default at missing dims, so a default no value can take (say
//! NaN, or the type's maximum) marks missing dims for a function that must
//! skip them.
//!
//! [`RowFormat`] picks the row a function receives. `Auto` follows the
//! record: CSR records arrive as CSR rows; dense records use measured
//! dimension, conversion and value-coding crossovers, delivering either a
//! dense row or their non-zeros as CSR. Cached images can compact dense
//! records once at admission. `Csr` and `Dense` force one format
//! and need only that callback.
//!
//! A function writes only its cell's `cell_bytes` at `out` and its scratch;
//! the loader checks nothing it writes. A nonzero return stops the run with
//! that code.

use std::ffi::c_void;
use std::sync::Arc;

use crate::format::Dtype;
use crate::loader::error::{Error, Result};

/// The process ABI the loader implements: the layout of [`ProcessContext`],
/// [`CsrRow`] and [`DenseRow`]. A callback built against another version must
/// not be passed to [`ProcessFn`].
pub const PROCESS_ABI_VERSION: u32 = 1;

/// What every process call learns about itself: which cell, which schema, and
/// the caller's `user_data`.
///
/// `struct_size` lets a callback detect a newer, longer row struct; the loader
/// sets it to the size of the struct it passes, not of this one on its own.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProcessContext {
    /// [`PROCESS_ABI_VERSION`] of the call.
    pub abi_version: u32,
    /// Size of the row struct the callback receives, in bytes.
    pub struct_size: u32,
    /// The access's position in the sequence (its position index, not the
    /// access index: every schema of one position sees the same value).
    pub position: u64,
    /// The global cell loaded.
    pub cell: u64,
    /// The schema's index in the loader.
    pub schema: u32,
    /// Element type code of `values` (the schema's dtype).
    pub dtype: u32,
    /// The pointer passed to [`ProcessFn::new`].
    pub user_data: *const c_void,
}

/// A CSR row: `nnz` stored elements with output columns in the chunk's
/// coded order (not sorted) and values of the schema's dtype. Every other
/// output dim is a zero or missing (see the module notes). A dense record
/// delivered as a CSR row lists its non-zeros.
///
/// With partial rows (see [`ProcessFn::with_partial_rows`]) `columns` and
/// `values` may be null: the function then maps the elements it needs with
/// `resolve(row, positions, count, columns, values)`, which writes the
/// column and value of each requested element (its position must lie below
/// `nnz`; repeats allowed), or of every element in row order when `positions`
/// is null and `count == nnz`. It returns 0, or nonzero for invalid
/// arguments, including a position at or beyond `nnz`. The `columns` and
/// `values` buffers are trusted to hold `count` entries. Partial rows are
/// offered only when every element of the row has an output column.
#[repr(C)]
pub struct CsrRow {
    /// ABI version, struct size, position, cell, schema and user data.
    pub context: ProcessContext,
    /// Output dims of the schema: the upper bound of a column.
    pub out_dim: u64,
    /// Stored elements listed; every other output dim is 0 or missing.
    pub nnz: u64,
    /// Output columns of the `nnz` elements, or null with partial rows.
    pub columns: *const u32,
    /// The `nnz` values, of the schema's dtype, or null with partial rows.
    pub values: *const c_void,
    /// The loader's decoded row, the argument `resolve` expects; null unless
    /// rows are partial.
    pub row: *const c_void,
    /// Maps elements of `row` on demand; null unless rows are partial.
    pub resolve: Option<ResolveFn>,
    /// The cell's output: exactly the schema's `cell_bytes`.
    pub out: *mut u8,
    /// This call's private scratch, at least the schema's `scratch_bytes`
    /// long.
    pub scratch: *mut u8,
    /// Bytes available at `scratch`.
    pub scratch_bytes: u64,
}

/// A dense row: `out_dim` values of the schema's dtype: the stored value or
/// 0 at present dims, the schema's default at missing dims (see the module
/// notes).
#[repr(C)]
pub struct DenseRow {
    /// ABI version, struct size, position, cell, schema and user data.
    pub context: ProcessContext,
    /// Values in `values`.
    pub out_dim: u64,
    /// The mapped and converted values, one per output dim.
    pub values: *const c_void,
    /// The cell's output: exactly the schema's `cell_bytes`.
    pub out: *mut u8,
    /// This call's private scratch, at least the schema's `scratch_bytes`
    /// long.
    pub scratch: *mut u8,
    /// Bytes available at `scratch`.
    pub scratch_bytes: u64,
}

/// The partial-row resolver a [`CsrRow`] carries: `(row, positions, count,
/// columns, values) -> status`, writing `count` column/value pairs.
pub type ResolveFn = unsafe extern "C" fn(*const c_void, *const u32, u64, *mut u32, *mut c_void) -> i32;
/// A CSR callback: returns 0 on success, any other value stops the run.
pub type CsrFn = unsafe extern "C" fn(*const CsrRow) -> i32;
/// A dense callback: returns 0 on success, any other value stops the run.
pub type DenseFn = unsafe extern "C" fn(*const DenseRow) -> i32;

/// The row handed to a process function, as the runtime holds it before
/// building the ABI struct.
#[derive(Clone, Copy)]
pub(crate) enum Row {
    /// A CSR image: `nnz` columns and values.
    Csr { nnz: usize, columns: *const u32, values: *const u8 },
    /// Elements resolved on demand behind `resolve` (partial rows).
    Partial { nnz: usize, row: *const c_void, resolve: ResolveFn },
    /// A dense image of the schema's dtype.
    Dense(*const u8),
}

/// The row format a process function receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RowFormat {
    /// CSR records as CSR rows; dense records use internal throughput
    /// crossovers. A compacted cached image reaches this format as CSR.
    Auto = 0,
    /// Every record as a CSR row (dense records: their non-zeros).
    Csr = 1,
    /// Every record as a dense row.
    Dense = 2,
}

impl RowFormat {
    /// The code stored in profiles and plan identities.
    pub fn code(self) -> u8 {
        self as u8
    }
    /// The format of `code`, or none for a value this version does not know.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Auto),
            1 => Some(Self::Csr),
            2 => Some(Self::Dense),
            _ => None,
        }
    }
}

/// A schema's callbacks: both for [`RowFormat::Auto`], one for a forced
/// format.
#[derive(Clone)]
pub struct ProcessFn {
    csr: Option<CsrFn>,
    dense: Option<DenseFn>,
    format: RowFormat,
    user_data: *const c_void,
    _owner: Arc<dyn Send + Sync>,
    partial: bool,
}

// SAFETY: construction requires thread-safe callbacks and context.
unsafe impl Send for ProcessFn {}
// SAFETY: the same contract permits simultaneous calls.
unsafe impl Sync for ProcessFn {}

impl ProcessFn {
    /// Binds a pair of callbacks ([`RowFormat::Auto`]). Both must compute the
    /// same output from the same row; which one runs depends on how the
    /// record is stored and how much of the output it covers.
    ///
    /// # Safety
    /// Callbacks and context stay valid through `owner`, tolerate concurrent
    /// calls, never unwind across the ABI, write only their cell's output and
    /// their scratch, and retain no pointer after returning.
    pub unsafe fn new(csr: CsrFn, dense: DenseFn, user_data: *const c_void, owner: Arc<dyn Send + Sync>) -> Self {
        Self { csr: Some(csr), dense: Some(dense), format: RowFormat::Auto, user_data, _owner: owner, partial: false }
    }

    /// Binds the callbacks a row format needs: both for `Auto`, `csr` for
    /// `Csr`, `dense` for `Dense`; an unneeded one may be absent and is
    /// never called.
    ///
    /// # Safety
    /// As for [`Self::new`].
    pub unsafe fn with_format(format: RowFormat, csr: Option<CsrFn>, dense: Option<DenseFn>, user_data: *const c_void, owner: Arc<dyn Send + Sync>) -> Result<Self> {
        let ok = match format {
            RowFormat::Auto => csr.is_some() && dense.is_some(),
            RowFormat::Csr => csr.is_some(),
            RowFormat::Dense => dense.is_some(),
        };
        if !ok {
            return Err(crate::loader::error::config(format!("row format {format:?} needs its callbacks")));
        }
        let (csr, dense) = match format {
            RowFormat::Auto => (csr, dense),
            RowFormat::Csr => (csr, None),
            RowFormat::Dense => (None, dense),
        };
        Ok(Self { csr, dense, format, user_data, _owner: owner, partial: false })
    }

    /// Lets `csr` receive partial rows (see [`CsrRow`]); ignored when only
    /// dense rows are delivered.
    ///
    /// # Safety
    /// The CSR callback handles null `columns` and `values` by calling
    /// `resolve`, and calls it only during its own invocation.
    pub unsafe fn with_partial_rows(mut self) -> Self {
        self.partial = self.csr.is_some();
        self
    }

    /// Whether the CSR callback takes partial rows.
    pub fn partial_rows(&self) -> bool {
        self.partial
    }

    /// The format the schema delivers, which the planner prices rows with.
    pub fn row_format(&self) -> RowFormat {
        self.format
    }

    /// Entry addresses of the CSR and dense callbacks (0 when absent).
    pub fn addresses(&self) -> (usize, usize) {
        (self.csr.map_or(0, |f| f as usize), self.dense.map_or(0, |f| f as usize))
    }

    /// Makes one call: builds the ABI struct for the row's kind, points it at
    /// the cell's output and the worker's scratch, and turns a nonzero status
    /// into [`Error::Process`].
    ///
    /// A row whose kind has no bound callback reports status -1 rather than
    /// calling anything; the runtime never asks for a format it did not bind.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn call(&self, position: u64, cell: u64, schema: u32, dtype: Dtype, out_dim: usize, row: Row, out: *mut u8, scratch: &mut [u8]) -> Result<()> {
        let context = ProcessContext { abi_version: PROCESS_ABI_VERSION, struct_size: 0, position, cell, schema, dtype: dtype.code() as u32, user_data: self.user_data };
        let status = match row {
            Row::Dense(values) => {
                let call = DenseRow { context: ProcessContext { struct_size: size_of::<DenseRow>() as u32, ..context }, out_dim: out_dim as u64, values: values.cast(), out, scratch: scratch.as_mut_ptr(), scratch_bytes: scratch.len() as u64 };
                // The runtime delivers dense rows only to formats with a dense callback.
                let dense = self.dense.ok_or(Error::Process(-1))?;
                // SAFETY: construction binds this ABI; buffers are live and
                // exclusive for the call.
                unsafe { dense(&call) }
            }
            Row::Csr { .. } | Row::Partial { .. } => {
                let (nnz, columns, values, row, resolve) = match row {
                    Row::Csr { nnz, columns, values } => (nnz, columns, values.cast(), std::ptr::null(), None),
                    Row::Partial { nnz, row, resolve } => (nnz, std::ptr::null(), std::ptr::null(), row, Some(resolve)),
                    Row::Dense(_) => unreachable!(),
                };
                let call = CsrRow { context: ProcessContext { struct_size: size_of::<CsrRow>() as u32, ..context }, out_dim: out_dim as u64, nnz: nnz as u64, columns, values, row, resolve, out, scratch: scratch.as_mut_ptr(), scratch_bytes: scratch.len() as u64 };
                let csr = self.csr.ok_or(Error::Process(-1))?;
                // SAFETY: as above.
                unsafe { csr(&call) }
            }
        };
        if status != 0 {
            return Err(Error::Process(status));
        }
        Ok(())
    }
}
