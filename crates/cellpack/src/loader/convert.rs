//! The precision conversion stage: a source row (a decoded record, or a
//! cached row) in its stored element type `S`, mapped to a schema's output
//! dims and converted to the schema's element type `T`, as a CSR row or a
//! dense row. Caches hold rows in `S`; only this stage converts.
//!
//! A dense row holds 0 at present dims the record stores no value for and
//! the schema's default at missing dims (see [`crate::loader::schema`]). A
//! dense record converted to a CSR row keeps its non-zeros (bitwise, as the
//! encoder counts them), the elements its CSR form would store.
//!
//! Conversions follow `as` semantics without range checks. Which pairs are
//! allowed is a declaration-time contract ([`allowed`]); kernels are chosen
//! once per (S, T) pair when a loader is built ([`RowOps`]).

use crate::format::Dtype;
use crate::loader::error::{Result, config};
use crate::runtime::Runtime;
use crate::scans::map::{Elems, RouteTable, SparseMap};
use crate::scans::{CastInto, Element, Fill, Parts, UNMAPPED, Values, cast_values};

/// Integer signedness or float: what decides which conversions exist.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Unsigned,
    Signed,
    Float,
}

/// The kind of `d`.
fn kind(d: Dtype) -> Kind {
    match d {
        Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => Kind::Unsigned,
        Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => Kind::Signed,
        Dtype::F32 | Dtype::F64 => Kind::Float,
    }
}

/// Whether values stored as `s` may be output as `t`: the same type,
/// lossless widening, same-width signed to unsigned (negatives are not
/// checked), signed to wider unsigned, and any integer to a float.
pub fn allowed(s: Dtype, t: Dtype) -> bool {
    let (sw, tw) = (s.width(), t.width());
    s == t
        || match (kind(s), kind(t)) {
            (Kind::Unsigned, Kind::Unsigned | Kind::Signed) | (Kind::Signed, Kind::Signed) => tw > sw,
            (Kind::Signed, Kind::Unsigned) => tw >= sw,
            (Kind::Unsigned | Kind::Signed, Kind::Float) => true,
            (Kind::Float, Kind::Float) => s == Dtype::F32 && t == Dtype::F64,
            (Kind::Float, _) => false,
        }
}

/// Cost classes of conversions, as the cost model prices them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Conversion {
    /// No conversion: the same element type.
    Same = 0,
    /// Integer widening (zero or sign extension).
    Widen = 1,
    /// Same-width signed to unsigned.
    Reinterpret = 2,
    /// Any integer to `f32` (exact up to 2^24).
    IntToF32 = 3,
    /// Any integer to `f64`.
    IntToF64 = 4,
    /// `f32` to `f64`.
    F32ToF64 = 5,
}

/// Classes in [`Conversion`]'s numbering; the cost model's table size.
pub const CONVERSIONS: usize = 6;

impl Conversion {
    /// The class of the pair `s -> t`, for every pair [`allowed`] accepts.
    /// Same-width signed-to-unsigned pairs are reinterpretations, wider pairs
    /// widenings.
    pub fn of(s: Dtype, t: Dtype) -> Self {
        match (kind(s), t) {
            _ if s == t => Self::Same,
            (Kind::Float, _) => Self::F32ToF64,
            (_, Dtype::F32) => Self::IntToF32,
            (_, Dtype::F64) => Self::IntToF64,
            _ if s.width() == t.width() => Self::Reinterpret,
            _ => Self::Widen,
        }
    }
}

/// Every dtype, in code order.
pub(crate) const DTYPES: [Dtype; 10] = [Dtype::U8, Dtype::U16, Dtype::U32, Dtype::U64, Dtype::I8, Dtype::I16, Dtype::I32, Dtype::I64, Dtype::F32, Dtype::F64];

/// Dtype code of `d`: the index of every per-dtype table in the loader.
pub(crate) fn dtype_index(d: Dtype) -> usize {
    d.code() as usize
}

/// `n` elements of `T` from 8-byte aligned words.
#[inline(always)]
fn typed_mut<T: Element>(words: &mut [u64], n: usize) -> &mut [T] {
    debug_assert!(n * T::WIDTH <= words.len() * 8);
    // SAFETY: numeric elements of at most 8 bytes have alignment at most 8,
    // every bit pattern is valid, and the words cover `n` of them.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), n) }
}

/// One (S, T) pair's conversion: dictionary and value casts, and the
/// mapping routes resolved for the bound kernel set. Byte buffers of `T`
/// are aligned to `T`; `dict` and `modes` hold `T` values in 8-byte words.
pub(crate) struct RowOps {
    /// A record's dictionary entries (dictionary or mode coded) as `T`.
    pub cast_dict: fn(&Parts<'_>, &mut [u64]) -> Result<()>,
    /// Converts `n` stored values (little-endian `S`) to `T`.
    pub values: fn(&[u8], *mut u8, usize) -> Result<()>,
    /// A dense record's values (little-endian `S`, coded order) to the CSR
    /// row of its mapped non-zeros; returns the count.
    dense_csr: fn(&[u8], &[u32], &mut [u32], *mut u8) -> usize,
    /// Fills `n` elements of `T` with the value whose little-endian bytes
    /// are the word.
    fill: fn(*mut u8, usize, u64),
    routes: RouteTable,
    t_width: usize,
    /// The bound kernel set has 512-bit streaming stores.
    pub streams: bool,
}

fn dict_k(parts: &Parts<'_>) -> usize {
    match parts.values() {
        Values::Dict(d) => d.k(),
        Values::Modes(d) => d.k(),
        Values::Raw(_) => 0,
    }
}

fn cast_dict<S: CastInto<T>, T: Element>(parts: &Parts<'_>, out: &mut [u64]) -> Result<()> {
    let k = dict_k(parts);
    let out = typed_mut::<T>(out, k);
    match parts.values() {
        Values::Dict(d) => d.cast_into::<S, T>(out)?,
        Values::Modes(d) => d.cast_into::<S, T>(out)?,
        Values::Raw(_) => {}
    }
    Ok(())
}

fn fill_ptr<T: Element>(out: *mut u8, n: usize, bits: u64) {
    // SAFETY: the caller passes `n` elements aligned to `T`.
    fill_value(unsafe { std::slice::from_raw_parts_mut(out.cast::<T>(), n) }, bits);
}

/// Fills `out` with the value whose little-endian bytes are `bits`; zero
/// (the usual default) is a memset, so large rows do not read the lines they
/// overwrite.
#[inline]
pub(crate) fn fill_value<T: Element>(out: &mut [T], bits: u64) {
    if bits == 0 {
        // SAFETY: zero bytes are a valid value of every numeric element.
        unsafe { std::ptr::write_bytes(out.as_mut_ptr(), 0, out.len()) };
    } else {
        out.fill(T::from_le(&bits.to_le_bytes()));
    }
}

fn values<S: CastInto<T>, T: Element>(bytes: &[u8], out: *mut u8, n: usize) -> Result<()> {
    // SAFETY: the caller passes `n` elements aligned to `T`.
    let out = unsafe { std::slice::from_raw_parts_mut(out.cast::<T>(), n) };
    Ok(cast_values::<S, T>(&bytes[..n * S::WIDTH], out)?)
}

/// Whether the `W`-byte little-endian value at `p` is zero bits.
///
/// # Safety
/// `p` points to `W` readable bytes.
#[inline(always)]
unsafe fn zero_bits<const W: usize>(p: *const u8) -> bool {
    // SAFETY: the caller guarantees `W` readable bytes.
    unsafe {
        match W {
            1 => *p == 0,
            2 => p.cast::<u16>().read_unaligned() == 0,
            4 => p.cast::<u32>().read_unaligned() == 0,
            _ => p.cast::<u64>().read_unaligned() == 0,
        }
    }
}

/// A dense record's values as the CSR row of its stored elements, the way the
/// encoder's zero rule defines them: an element is kept when its column maps
/// and its bits are not all zero.
///
/// Each element is written at the current count and the count advances only for
/// a kept one, so a dropped element's write is overwritten by the next kept
/// element, and a trailing one lies past the returned count.
fn dense_csr<S: CastInto<T>, T: Element>(bytes: &[u8], map: &[u32], cols: &mut [u32], vals: *mut u8) -> usize {
    let n = (bytes.len() / S::WIDTH).min(map.len());
    debug_assert!(cols.len() >= n);
    let vals = vals.cast::<T>();
    let mut k = 0;
    for (i, &col) in map[..n].iter().enumerate() {
        // SAFETY: `i < n` values lie in `bytes`; `cols` and `vals` hold an
        // element for each of the `n` genes and `k <= i`.
        unsafe {
            let p = bytes.as_ptr().add(i * S::WIDTH);
            let keep = col != UNMAPPED && !match S::WIDTH {
                1 => zero_bits::<1>(p),
                2 => zero_bits::<2>(p),
                4 => zero_bits::<4>(p),
                _ => zero_bits::<8>(p),
            };
            *cols.get_unchecked_mut(k) = col;
            vals.add(k).write(S::read_le(p).cast());
            k += keep as usize;
        }
    }
    k
}

/// Writes 0 at every mapped column of `map` in a dense row of `width`-byte
/// elements (a sparse record's present dims, before its values land).
fn zero_present(out: *mut u8, out_dim: usize, width: usize, map: &[u32]) {
    macro_rules! zero {
        ($t:ty) => {{
            let row = out.cast::<$t>();
            for &col in map {
                if col != UNMAPPED {
                    debug_assert!((col as usize) < out_dim);
                    // SAFETY: mapped columns lie inside the row of `out_dim`.
                    unsafe { row.add(col as usize).write_unaligned(0) };
                }
            }
        }};
    }
    match width {
        1 => zero!(u8),
        2 => zero!(u16),
        4 => zero!(u32),
        _ => zero!(u64),
    }
}

fn ops<S: CastInto<T>, T: Element>(rt: &Runtime) -> RowOps {
    RowOps {
        cast_dict: cast_dict::<S, T>,
        values: values::<S, T>,
        dense_csr: dense_csr::<S, T>,
        fill: fill_ptr::<T>,
        routes: RouteTable::new::<S, T>(rt.kernels()),
        t_width: T::WIDTH,
        streams: rt.isa() == crate::scans::Isa::Avx512,
    }
}

impl RowOps {
    /// The record's dictionary words and the chunk's mode words as `T`.
    #[inline(always)]
    fn elems(&self, parts: &Parts<'_>, dict: &[u64], modes: &[u64]) -> (Elems, Elems) {
        (
            Elems { ptr: dict.as_ptr().cast(), len: dict_k(parts).min(dict.len() * 8 / self.t_width) },
            Elems { ptr: modes.as_ptr().cast(), len: modes.len() * 8 / self.t_width },
        )
    }

    /// Mapped CSR row: kept elements' columns and `T` values in order;
    /// returns the count. `vals` has room for every element.
    #[inline]
    pub fn csr(&self, parts: &Parts<'_>, map: &[u32], dict: &[u64], modes: &[u64], cols: &mut [u32], vals: *mut u8) -> Result<usize> {
        let (d, m) = self.elems(parts, dict, modes);
        Ok(self.routes.csr(parts, map, d, m, cols, vals, parts.elements())?)
    }

    /// Dense row of `out_dim` values of `T`: the row's elements, 0 at the
    /// other present dims and `fill` at missing ones. `present` is the
    /// chunk's coded gene -> output column map; a dense record writes every
    /// present dim itself, so it is consulted only for a sparse row with a
    /// nonzero fill.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn dense(&self, parts: &Parts<'_>, map: &[u32], dict: &[u64], modes: &[u64], fill: u64, present: Option<&[u32]>, out: *mut u8, out_dim: usize) -> Result<()> {
        self.missing(fill, present, !parts.dense, out, out_dim);
        let (d, m) = self.elems(parts, dict, modes);
        Ok(self.routes.dense(parts, map, d, m, out, out_dim, Fill::AssumeZeroed)?)
    }

    /// [`Self::dense`] through the sparse form of `map`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn dense_sparse(&self, parts: &Parts<'_>, map: &SparseMap, dict: &[u64], modes: &[u64], fill: u64, present: Option<&[u32]>, out: *mut u8, out_dim: usize) -> Result<()> {
        self.missing(fill, present, !parts.dense, out, out_dim);
        let (d, m) = self.elems(parts, dict, modes);
        Ok(self.routes.dense_sparse(parts, map, d, m, out, out_dim, Fill::AssumeZeroed)?)
    }

    /// A dense row with no stored element: 0 at present dims, `fill` at
    /// missing ones.
    #[inline]
    pub fn empty_dense(&self, fill: u64, present: Option<&[u32]>, out: *mut u8, out_dim: usize) {
        self.missing(fill, present, true, out, out_dim);
    }

    #[inline]
    fn missing(&self, fill: u64, present: Option<&[u32]>, sparse: bool, out: *mut u8, out_dim: usize) {
        (self.fill)(out, out_dim, fill);
        if fill != 0 && sparse && let Some(map) = present {
            zero_present(out, out_dim, self.t_width, map);
        }
    }

    /// A dense record's values (`S`, coded order, every gene) as the CSR
    /// row of its non-zeros mapped through `map`; returns the count. `cols`
    /// and `vals` hold an element per gene.
    #[inline]
    pub fn dense_as_csr(&self, values: &[u8], map: &[u32], cols: &mut [u32], vals: *mut u8) -> usize {
        (self.dense_csr)(values, map, cols, vals)
    }

    /// Elements at `positions` of a row every element of which is mapped.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn at(&self, parts: &Parts<'_>, positions: &[u32], map: &[u32], dict: &[u64], modes: &[u64], cols: &mut [u32], vals: *mut u8) -> Result<()> {
        let (d, m) = self.elems(parts, dict, modes);
        Ok(self.routes.at(parts, positions, map, d, m, cols, vals, positions.len())?)
    }
}

macro_rules! by_type {
    ($d:expr, $t:ident => $e:expr) => {
        match $d {
            Dtype::U8 => { type $t = u8; $e }
            Dtype::U16 => { type $t = u16; $e }
            Dtype::U32 => { type $t = u32; $e }
            Dtype::U64 => { type $t = u64; $e }
            Dtype::I8 => { type $t = i8; $e }
            Dtype::I16 => { type $t = i16; $e }
            Dtype::I32 => { type $t = i32; $e }
            Dtype::I64 => { type $t = i64; $e }
            Dtype::F32 => { type $t = f32; $e }
            Dtype::F64 => { type $t = f64; $e }
        }
    };
}

impl RowOps {
    /// The kernels converting `source` values to `target`, if allowed, resolved
    /// against the process-wide kernel set.
    pub fn new(source: Dtype, target: Dtype) -> Result<Self> {
        if !allowed(source, target) {
            return Err(config(format!("values stored as {source:?} cannot be output as {target:?}")));
        }
        let rt = crate::runtime::get()?;
        Ok(by_type!(source, S => by_type!(target, T => ops::<S, T>(rt))))
    }
}

/// A value of `dtype` given as f64 (exact for every integer up to 2^53 and
/// every float), as little-endian bytes in a word.
pub(crate) fn value_bits(dtype: Dtype, v: f64) -> u64 {
    let mut out = [0u8; 8];
    macro_rules! put {
        ($t:ty) => {
            out[..size_of::<$t>()].copy_from_slice(&(v as $t).to_le_bytes())
        };
    }
    by_type!(dtype, T => put!(T));
    u64::from_le_bytes(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_conversions_follow_the_contract() {
        use Dtype::*;
        for (s, t, ok) in [
            (U16, U16, true), (U16, U32, true), (U8, I16, true), (I8, I32, true), (I32, U32, true), (I16, U32, true),
            (U32, F32, true), (I64, F64, true), (F32, F64, true),
            (U32, U16, false), (I64, I32, false), (F64, F32, false), (F32, I32, false), (U32, I32, false), (I16, U8, false), (U8, I8, false),
        ] {
            assert_eq!(allowed(s, t), ok, "{s:?} -> {t:?}");
        }
        assert_eq!(Conversion::of(U16, U32), Conversion::Widen);
        assert_eq!(Conversion::of(I32, U32), Conversion::Reinterpret);
        assert_eq!(Conversion::of(U32, F32), Conversion::IntToF32);
        assert_eq!(Conversion::of(F32, F64), Conversion::F32ToF64);
        assert_eq!(Conversion::of(F32, F32), Conversion::Same);
    }

    #[test]
    fn every_allowed_pair_builds_kernels() {
        for s in DTYPES {
            for t in DTYPES {
                assert_eq!(RowOps::new(s, t).is_ok(), allowed(s, t));
            }
        }
    }
}
