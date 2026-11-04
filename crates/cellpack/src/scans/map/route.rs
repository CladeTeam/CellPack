//! Which kernel maps each (element types, value encoding, output shape),
//! decided once per element-type pair, and the validation every row passes
//! before its kernel runs.
//!
//! Routes are chosen in one place ([`csr_route`], [`at_route`],
//! [`dense_route`]) from the kernel set's ISA, the measured selection policy
//! ([`mapping_policy`]) and which kernels exist for the pair. A row's
//! encoding is the only per-row input: it indexes the resolved table.

use super::{CastInto, Element, SparseMap, UNMAPPED};
use super::slots;
use crate::error::{Error, Result};
use crate::scans::decode::{Parts, Ranks, Values};
use crate::scans::kernels::{Conv, Isa, Kernels, Src, mapping_policy, out_slot};
use crate::scans::types::Fill;

/// Value encodings, in the order the selection policy uses. The code is the
/// index into a route table and the position of its bit in the policy masks.
pub(crate) const RAW: u8 = 0;
/// A one-entry dictionary: every value is the same.
pub(crate) const CONST: u8 = 1;
/// A dictionary of at most 192 entries, ranked by byte symbols.
pub(crate) const DICT8: u8 = 2;
/// A dictionary with 32-bit ranks.
pub(crate) const DICT32: u8 = 3;
/// Mode coding with no exceptions: every value is its column mode.
pub(crate) const MODE: u8 = 4;
/// Mode coding with byte ranks.
pub(crate) const MODES8: u8 = 5;
/// Mode coding with 32-bit ranks.
pub(crate) const MODES32: u8 = 6;
/// Number of value encodings, the length of every route table.
pub(crate) const ENCODINGS: usize = 7;

/// Output shapes, in the order the selection policy uses.
const COMPACT: u8 = 0;
const INDEXED: u8 = 1;
const SCATTER: u8 = 2;

/// Largest index the SIMD gathers take.
const GATHER_MAX: usize = i32::MAX as usize;

/// The encoding of a row's values and its dictionary size.
#[inline(always)]
pub(crate) fn encoding(p: &Parts) -> (u8, usize) {
    match p.values {
        Values::Raw(_) => (RAW, 0),
        Values::Dict(d) => (match d.ranks() { Ranks::One => CONST, Ranks::U8(_) => DICT8, Ranks::U32(_) => DICT32 }, d.k()),
        Values::Modes(d) => (match d.ranks() { Ranks::One => MODE, Ranks::U8(_) => MODES8, Ranks::U32(_) => MODES32 }, d.k()),
    }
}

/// A slice of converted elements, type-erased: `len` elements of the
/// target type at `ptr`.
#[derive(Clone, Copy)]
pub(crate) struct Elems {
    /// First element of the target type.
    pub ptr: *const u8,
    /// Number of elements.
    pub len: usize,
}

impl Elems {
    /// A converted slice of the target type.
    #[inline(always)]
    pub(crate) fn of<T>(v: &[T]) -> Self {
        Self { ptr: v.as_ptr().cast(), len: v.len() }
    }
}

/// Validated inputs of one row for a route kernel.
pub(crate) struct Args<'a> {
    /// The row's decoded view.
    pub parts: &'a Parts<'a>,
    /// Column map, indexed by gene id.
    pub map: &'a [u32],
    /// Converted dictionary entries (exactly `k`) and column modes.
    pub dict: Elems,
    /// Converted column modes, indexed by coded id.
    pub modes: Elems,
    /// Elements to map, for the indexed shape.
    pub positions: &'a [u32],
    /// Room for every output column (compact, indexed).
    pub cols: *mut u32,
    /// Output elements of the target type: the row's elements (compact),
    /// one per position (indexed), or the dense row (scatter).
    pub vals: *mut u8,
    /// Entries of the dense output row, for the scatter shape.
    pub out_len: usize,
}

/// Mapped CSR row; returns the elements kept.
pub(crate) type CsrSlot = unsafe fn(&Kernels, &Args) -> usize;
/// Elements at positions.
pub(crate) type AtSlot = unsafe fn(&Kernels, &Args);
/// Dense scatter; false when a mapped column lies outside the row.
pub(crate) type DenseSlot = unsafe fn(&Kernels, &Args) -> bool;
/// Dense scatter through a [`SparseMap`]; false as for [`DenseSlot`].
pub(crate) type SparseSlot = unsafe fn(&Kernels, &Args, &SparseMap) -> bool;

/// The routes of one element-type pair on one kernel set.
///
/// Everything is fixed at binding time: the kernel set, the element widths
/// and, per encoding, the function of each shape, so a row costs one check,
/// one table read and one indirect call.
#[derive(Clone, Copy)]
pub(crate) struct RouteTable {
    k: &'static Kernels,
    s_width: usize,
    t_width: usize,
    csr: [CsrSlot; ENCODINGS],
    at: [AtSlot; ENCODINGS],
    dense: [DenseSlot; ENCODINGS],
    sparse: [SparseSlot; ENCODINGS],
}

impl RouteTable {
    /// Resolves the selected route of every encoding and shape for the
    /// element-type pair on `k`.
    pub(crate) fn new<S: CastInto<T>, T: Element>(k: &'static Kernels) -> Self {
        Self {
            k,
            s_width: S::WIDTH,
            t_width: T::WIDTH,
            csr: std::array::from_fn(|e| csr_route::<S, T>(k, e as u8)),
            at: std::array::from_fn(|e| at_route::<S, T>(k, e as u8)),
            dense: std::array::from_fn(|e| dense_route::<S, T>(k, e as u8)),
            sparse: std::array::from_fn(|e| sparse_route::<S, T>(e as u8)),
        }
    }

    /// Every route a kernel exists for, ignoring the measured policy, for
    /// the equivalence tests and dispatch benchmarks.
    #[cfg(test)]
    pub(crate) fn candidates<S: CastInto<T>, T: Element>(k: &'static Kernels) -> Self {
        Self {
            k,
            s_width: S::WIDTH,
            t_width: T::WIDTH,
            csr: std::array::from_fn(|e| pick_csr::<S, T>(k, e as u8, true)),
            at: std::array::from_fn(|e| pick_at::<S, T>(k, e as u8, true)),
            dense: std::array::from_fn(|e| pick_dense::<S, T>(k, e as u8, true)),
            sparse: std::array::from_fn(|e| sparse_route::<S, T>(e as u8)),
        }
    }

    /// Compact route: see [`super::map_parts`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn csr(&self, p: &Parts, map: &[u32], dict: Elems, modes: Elems, cols: &mut [u32], vals: *mut u8, vals_len: usize) -> Result<usize> {
        csr_with(self.k, self.s_width, |e| self.csr[e as usize], p, map, dict, modes, cols, vals, vals_len)
    }

    /// Indexed route: see [`super::map_parts_at`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn at(&self, p: &Parts, positions: &[u32], map: &[u32], dict: Elems, modes: Elems, cols: &mut [u32], vals: *mut u8, vals_len: usize) -> Result<()> {
        at_with(self.k, self.s_width, |e| self.at[e as usize], p, positions, map, dict, modes, cols, vals, vals_len)
    }

    /// Dense scatter: see [`super::map_dense`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense(&self, p: &Parts, map: &[u32], dict: Elems, modes: Elems, out: *mut u8, out_len: usize, fill: Fill) -> Result<()> {
        dense_with(self.k, self.s_width, self.t_width, |e| self.dense[e as usize], p, map, dict, modes, out, out_len, fill)
    }

    /// As [`Self::dense`], through a sparse map of the row's ids.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense_sparse(&self, p: &Parts, map: &SparseMap, dict: Elems, modes: Elems, out: *mut u8, out_len: usize, fill: Fill) -> Result<()> {
        let (enc, dict_k) = check(p, p.ids.last().copied(), map.ids(), dict, modes, self.s_width)?;
        if fill == Fill::Zero {
            // SAFETY: as in `dense_with`.
            unsafe { std::ptr::write_bytes(out, 0, out_len * self.t_width) };
        }
        let args = Args { parts: p, map: &[], dict: Elems { len: dict_k, ..dict }, modes, positions: &[], cols: std::ptr::null_mut(), vals: out, out_len };
        // SAFETY: as in `dense_with`: `check` bounds every id by the map's
        // bits, and the kernel checks mapped columns against `out_len`.
        if unsafe { self.sparse[enc as usize](self.k, &args, map) } {
            Ok(())
        } else {
            Err(Error::InvalidInput("mapped column outside the output row"))
        }
    }
}

/// Checks the inputs every shape shares and returns the row's encoding
/// and dictionary size. `last` is the row's largest id, if any.
#[inline]
fn check(p: &Parts, last: Option<u32>, map_len: usize, dict: Elems, modes: Elems, s_width: usize) -> Result<(u8, usize)> {
    let (enc, k) = encoding(p);
    // Decoded ids ascend, so the last bounds them all. Cached rows the
    // loader builds hold standard ids, all below the vocabulary its maps
    // cover.
    debug_assert!(p.ids.iter().all(|&id| (id as usize) < map_len) || last.is_some_and(|l| l as usize >= map_len));
    if let Some(last) = last
        && last as usize >= map_len
    {
        return Err(Error::InvalidInput("column map shorter than the row's ids"));
    }
    if dict.len < k {
        return Err(Error::InvalidInput("converted dictionary shorter than k"));
    }
    if matches!(p.values, Values::Modes(_))
        && last.is_some_and(|l| l as usize >= modes.len)
    {
        return Err(Error::InvalidInput("converted column modes shorter than the row's ids"));
    }
    if let Values::Raw(b) = p.values
        && b.len() != p.ids.len() * s_width
    {
        return Err(Error::InvalidInput("source element type does not match the value width"));
    }
    if map_len > GATHER_MAX || modes.len > GATHER_MAX || p.ids.len() > GATHER_MAX || k > GATHER_MAX + 1 {
        return Err(Error::InvalidInput("column map, column modes, row or dictionary beyond signed 32-bit indexing"));
    }
    Ok((enc, k))
}

/// Validates one row and runs the compact route `slot` picks for its
/// encoding; returns the elements kept.
///
/// A dense record is rejected: its elements are every gene, which only the
/// dense shape writes.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn csr_with(k: &Kernels, s_width: usize, slot: impl FnOnce(u8) -> CsrSlot, p: &Parts, map: &[u32], dict: Elems, modes: Elems, cols: &mut [u32], vals: *mut u8, vals_len: usize) -> Result<usize> {
    if p.dense {
        return Err(Error::InvalidInput("a dense record maps to a dense row (see map_dense)"));
    }
    let n = p.elements();
    if cols.len() < n || vals_len < n {
        return Err(Error::OutputTooSmall { need: n, have: cols.len().min(vals_len) });
    }
    let Some(&last) = p.ids.last() else {
        return Ok(0);
    };
    let (enc, dict_k) = check(p, Some(last), map.len(), dict, modes, s_width)?;
    let args = Args { parts: p, map, dict: Elems { len: dict_k, ..dict }, modes, positions: &[], cols: cols.as_mut_ptr(), vals, out_len: n };
    // SAFETY: `check` and the capacity test establish the route contract:
    // ids index the map (and the modes), the dictionary holds `k` entries,
    // raw values cover the row, every index fits a signed gather, and the
    // outputs hold every element. Decoded ranks are within `k`.
    Ok(unsafe { slot(enc)(k, &args) })
}

/// Validates one row and runs the indexed route `slot` picks for its
/// encoding; a position at or beyond the row length is an error.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn at_with(k: &Kernels, s_width: usize, slot: impl FnOnce(u8) -> AtSlot, p: &Parts, positions: &[u32], map: &[u32], dict: Elems, modes: Elems, cols: &mut [u32], vals: *mut u8, vals_len: usize) -> Result<()> {
    if p.dense {
        return Err(Error::InvalidInput("a dense record maps to a dense row (see map_dense)"));
    }
    let (n, m) = (p.elements(), positions.len());
    if cols.len() < m || vals_len < m {
        return Err(Error::OutputTooSmall { need: m, have: cols.len().min(vals_len) });
    }
    if m == 0 {
        return Ok(());
    }
    // Positions come from the caller, not from the record: one pass bounds
    // them all before any kernel gathers through them.
    if positions.iter().copied().max().is_some_and(|q| q as usize >= n) {
        return Err(Error::InvalidInput("element position beyond the row"));
    }
    let (enc, dict_k) = check(p, p.ids.last().copied(), map.len(), dict, modes, s_width)?;
    let args = Args { parts: p, map, dict: Elems { len: dict_k, ..dict }, modes, positions, cols: cols.as_mut_ptr(), vals, out_len: m };
    // SAFETY: as in `csr_with`, with every position below the row length
    // and outputs holding one entry per position.
    unsafe { slot(enc)(k, &args) };
    Ok(())
}

/// Validates one row, clears the dense row when `fill` asks for it, and runs
/// the scatter route `slot` picks; a mapped column outside `out` is an error.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn dense_with(k: &Kernels, s_width: usize, t_width: usize, slot: impl FnOnce(u8) -> DenseSlot, p: &Parts, map: &[u32], dict: Elems, modes: Elems, out: *mut u8, out_len: usize, fill: Fill) -> Result<()> {
    let (enc, dict_k) = check(p, p.ids.last().copied(), map.len(), dict, modes, s_width)?;
    if fill == Fill::Zero {
        // SAFETY: `out` holds `out_len` elements of `t_width` bytes; zero
        // bytes are zero in every element type.
        unsafe { std::ptr::write_bytes(out, 0, out_len * t_width) };
    }
    let args = Args { parts: p, map, dict: Elems { len: dict_k, ..dict }, modes, positions: &[], cols: std::ptr::null_mut(), vals: out, out_len };
    // SAFETY: as in `csr_with`; scatter kernels check every mapped column
    // against `out_len` before storing.
    if unsafe { slot(enc)(k, &args) } {
        Ok(())
    } else {
        Err(Error::InvalidInput("mapped column outside the output row"))
    }
}

/// Selection-policy code of an element type: unsigned widths 1, 2, 4 and 8
/// bytes are 0..4, signed ones 4..8, `f32` 8 and `f64` 9.
const fn code(kind: u8, width: usize) -> usize {
    match kind {
        0 => width.trailing_zeros() as usize,
        1 => 4 + width.trailing_zeros() as usize,
        _ => 8 + (width == 8) as usize,
    }
}

/// Whether the measured policy selects the kernel route of an encoding and
/// shape on this ISA over the scalar loop: bit `3 * enc + shape` of the
/// pair's mask, where the three shape bits are compact, indexed and scatter.
fn selected<S: Element, T: Element>(isa: Isa, enc: u8, shape: u8) -> bool {
    let masks = match isa {
        Isa::Avx512 => &mapping_policy::MASKS_AVX512,
        Isa::Avx2 => &mapping_policy::MASKS_AVX2,
        Isa::Scalar => return false,
    };
    masks[code(S::KIND, S::WIDTH)][code(T::KIND, T::WIDTH)] & (1 << (3 * enc as usize + shape as usize)) != 0
}

/// How lane kernels read `S` and produce `T`, where they cover the pair:
/// sources of at most 4 bytes, outputs of at most 4 bytes, and no float
/// narrowing or float-to-integer conversion.
pub(crate) fn lanes<S: Element, T: Element>() -> Option<(Src, Conv)> {
    let src = match (S::WIDTH, S::KIND) {
        (1, 0) => Src::U8,
        (1, 1) => Src::I8,
        (2, 0) => Src::U16,
        (2, 1) => Src::I16,
        (4, _) => Src::W32,
        _ => return None,
    };
    let conv = match (S::KIND, T::KIND, T::WIDTH) {
        (_, _, w) if w > 4 => return None,
        (2, 2, 4) if S::WIDTH == 4 => Conv::Bits,
        (0, 2, 4) => Conv::U32ToF32,
        (1, 2, 4) => Conv::I32ToF32,
        (2, _, _) | (_, 2, _) => return None,
        _ => Conv::Bits,
    };
    Some((src, conv))
}

/// Whether the eight-lane typed kernels take this encoding and shape:
/// they cover wide and narrow outputs, while 4-byte outputs keep the
/// sixteen-lane kernels except for raw sources those do not read.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn typed<S: Element, T: Element>(k: &Kernels, enc: u8, shape: u8) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // A byte-ranked narrow dictionary fits the sixteen-lane widened table.
        if shape == COMPACT && T::WIDTH < 4 && enc == DICT8 {
            return false;
        }
        k.isa == Isa::Avx512
            && (T::WIDTH != 4 || (enc == RAW && (S::WIDTH == 8 || (shape == INDEXED && S::WIDTH < 4))))
            && crate::scans::kernels::typed::supported::<S, T>()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (k, enc, shape);
        false
    }
}

/// Encodings the sixteen-lane 32-bit kernels read for this pair: every
/// dictionary or mode encoding, and raw sources lane kernels convert.
fn wide<S: Element, T: Element>(enc: u8) -> bool {
    T::WIDTH == 4 && (enc != RAW || lanes::<S, T>().is_some())
}

/// Builds a route table by monomorphizing `$f` for every encoding: the
/// encoding becomes a constant parameter, so a route kernel carries no
/// per-element dispatch.
macro_rules! by_encoding {
    ($e:expr, $f:ident::<$s:ident, $t:ident> as $ty:ty) => {
        match $e {
            RAW => slots::$f::<$s, $t, RAW> as $ty,
            CONST => slots::$f::<$s, $t, CONST> as $ty,
            DICT8 => slots::$f::<$s, $t, DICT8> as $ty,
            DICT32 => slots::$f::<$s, $t, DICT32> as $ty,
            MODE => slots::$f::<$s, $t, MODE> as $ty,
            MODES8 => slots::$f::<$s, $t, MODES8> as $ty,
            _ => slots::$f::<$s, $t, MODES32> as $ty,
        }
    };
}

/// Compact route of one encoding: the kernel the policy selects, else the
/// scalar loop.
pub(crate) fn csr_route<S: CastInto<T>, T: Element>(k: &Kernels, enc: u8) -> CsrSlot {
    pick_csr::<S, T>(k, enc, selected::<S, T>(k.isa, enc, COMPACT))
}

/// Indexed route of one encoding.
pub(crate) fn at_route<S: CastInto<T>, T: Element>(k: &Kernels, enc: u8) -> AtSlot {
    pick_at::<S, T>(k, enc, selected::<S, T>(k.isa, enc, INDEXED))
}

/// Dense route of one encoding.
pub(crate) fn dense_route<S: CastInto<T>, T: Element>(k: &Kernels, enc: u8) -> DenseSlot {
    pick_dense::<S, T>(k, enc, selected::<S, T>(k.isa, enc, SCATTER))
}

/// The compact route: a kernel of this set when `use_kernels`, else the
/// scalar loop.
///
/// Candidates are tried in order of measured speed: the eight-lane typed
/// kernels, then raw lane kernels (or the 4-byte copy when source and target
/// types agree), then the sixteen-lane 32-bit kernels for dictionary and
/// mode encodings, then the byte-ranked dictionary-lane kernel, then scalar.
fn pick_csr<S: CastInto<T>, T: Element>(k: &Kernels, enc: u8, use_kernels: bool) -> CsrSlot {
    if !use_kernels || k.isa == Isa::Scalar {
        return by_encoding!(enc, csr_scalar::<S, T> as CsrSlot);
    }
    #[cfg(target_arch = "x86_64")]
    if typed::<S, T>(k, enc, COMPACT) {
        return slots::csr_typed::<S, T>;
    }
    match enc {
        RAW => {
            if let Some((src, conv)) = lanes::<S, T>()
                && k.map_lanes[src as usize][conv as usize][out_slot(T::WIDTH)].is_some()
            {
                slots::csr_lanes::<S, T> as CsrSlot
            } else if S::WIDTH == 4 && T::WIDTH == 4 && std::any::TypeId::of::<S>() == std::any::TypeId::of::<T>() {
                slots::csr_raw32::<T> as CsrSlot
            } else {
                slots::csr_scalar::<S, T, RAW> as CsrSlot
            }
        }
        _ if T::WIDTH == 4 => by_encoding!(enc, csr_wide::<S, T> as CsrSlot),
        DICT8 if T::WIDTH < 4 && k.map_dict8_lanes[out_slot(T::WIDTH)].is_some() => slots::csr_dict8_lanes::<T> as CsrSlot,
        _ => by_encoding!(enc, csr_scalar::<S, T> as CsrSlot),
    }
}

/// The indexed route: the typed kernels, else the 4-byte kernels for the
/// pairs they read, else scalar. No dictionary-lane kernel takes positions.
fn pick_at<S: CastInto<T>, T: Element>(k: &Kernels, enc: u8, use_kernels: bool) -> AtSlot {
    if !use_kernels || k.isa == Isa::Scalar {
        return by_encoding!(enc, at_scalar::<S, T> as AtSlot);
    }
    #[cfg(target_arch = "x86_64")]
    if typed::<S, T>(k, enc, INDEXED) {
        return slots::at_typed::<S, T>;
    }
    if wide::<S, T>(enc) {
        by_encoding!(enc, at_wide::<S, T> as AtSlot)
    } else {
        by_encoding!(enc, at_scalar::<S, T> as AtSlot)
    }
}

/// The dense route through a [`SparseMap`]: one kernel for every route, since
/// the map's bit test replaces the gather a kernel would speed up.
fn sparse_route<S: CastInto<T>, T: Element>(enc: u8) -> SparseSlot {
    by_encoding!(enc, dense_sparse::<S, T> as SparseSlot)
}

/// The dense route: the typed kernels, else the 4-byte scatter for the pairs
/// the wide kernels read, else scalar.
fn pick_dense<S: CastInto<T>, T: Element>(k: &Kernels, enc: u8, use_kernels: bool) -> DenseSlot {
    if !use_kernels || k.isa == Isa::Scalar {
        return by_encoding!(enc, dense_scalar::<S, T> as DenseSlot);
    }
    #[cfg(target_arch = "x86_64")]
    if typed::<S, T>(k, enc, SCATTER) {
        return slots::dense_typed::<S, T>;
    }
    if wide::<S, T>(enc) {
        by_encoding!(enc, dense_wide::<S, T> as DenseSlot)
    } else {
        by_encoding!(enc, dense_scalar::<S, T> as DenseSlot)
    }
}

const _: () = assert!(UNMAPPED == u32::MAX);
