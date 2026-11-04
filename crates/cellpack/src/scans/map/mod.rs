//! From a [`Parts`] view to a mapped sparse row: gene ids through a caller
//! column map, values converted to the caller's element type. This is the
//! step between decoding and a caller's own sampling or scatter.
//!
//! The usual column map is indexed by coded id and composed once per chunk
//! from [`DecodeTables::source_of_coded`](crate::scans::DecodeTables::source_of_coded)
//! and the caller's gene panel, so a row maps with one lookup per value. A
//! dictionary converts its `k` entries once ([`Dict::cast_into`]); values
//! then come from the converted entries by rank.
//!
//! A [`Mapper`] resolves, once per element-type pair, which kernel maps
//! each value encoding in each output shape (compact, indexed, scatter) on
//! the bound kernel set. Mapping a row then validates the inputs, reads the
//! row's encoding and makes one indirect call. The free functions
//! ([`map_parts`], [`map_parts_at`], [`map_dense`]) resolve only the route a
//! row needs, for one-off calls.
//!
//! Inputs addressed by the SIMD gathers use signed 32-bit indices: column
//! maps, column modes, rows and dictionaries longer than `i32::MAX` are
//! rejected.

mod route;
mod slots;

use std::marker::PhantomData;

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::scans::decode::{Dict, ModeDict, Parts};
use crate::scans::types::Fill;
use crate::unchecked;

/// Route resolution and its validated per-row inputs.
pub(crate) use route::{Elems, RouteTable};

/// Column-map entry for genes the caller drops.
pub const UNMAPPED: u32 = u32::MAX;

/// A column map that keeps few of its ids, for outputs much narrower than
/// the rows they read. Per 32 ids, `words[2w]` holds their bits (bit
/// `i % 32` marks id `i` as mapped) and `words[2w + 1]` the count of mapped
/// ids below them, side by side so a kept element reads one line for both;
/// the `r`-th mapped id maps to column `r` of `cols` (u16 when every column
/// fits, else u32). It replaces a lookup per element in a full map (one
/// cache line per 16 ids) with a bit test in a table 16 times smaller.
#[derive(Clone, Copy)]
pub(crate) struct SparseMap<'a> {
    /// Bit and count word pairs, 32 ids each (see the struct docs).
    pub words: &'a [u32],
    cols: *const u8,
    wide: bool,
    kept: usize,
    _cols: std::marker::PhantomData<&'a [u32]>,
}

impl SparseMap<'_> {
    /// Ids the map covers.
    #[inline]
    pub fn ids(&self) -> usize {
        self.words.len() / 2 * 32
    }

    /// Output column of the `r`-th mapped id.
    ///
    /// # Safety
    /// `r` is below the number of mapped ids.
    #[inline(always)]
    pub unsafe fn col(&self, r: usize) -> u32 {
        debug_assert!(r < self.kept);
        // SAFETY: as documented; `cols` holds `kept` entries of its width.
        unsafe { if self.wide { *self.cols.cast::<u32>().add(r) } else { *self.cols.cast::<u16>().add(r) as u32 } }
    }

    /// Appends the sparse form of a full map (`UNMAPPED`: dropped) to
    /// `arena` as paired words then columns; returns the word pairs, the
    /// kept count and whether columns are u32.
    pub fn append(map: &[u32], arena: &mut Vec<u32>) -> (usize, usize, bool) {
        let words = map.len().div_ceil(32);
        let at = arena.len();
        arena.resize(at + 2 * words, 0);
        let mut cols = Vec::new();
        for w in 0..words {
            arena[at + 2 * w + 1] = cols.len() as u32;
            let mut bits = 0u32;
            for (b, &col) in map[w * 32..map.len().min(w * 32 + 32)].iter().enumerate() {
                if col != UNMAPPED {
                    bits |= 1 << b;
                    cols.push(col);
                }
            }
            arena[at + 2 * w] = bits;
        }
        let wide = cols.iter().any(|&c| c > u16::MAX as u32);
        if wide {
            arena.extend_from_slice(&cols);
        } else {
            arena.extend(cols.chunks(2).map(|p| p[0] | p.get(1).map_or(0, |&c| c << 16)));
        }
        (words, cols.len(), wide)
    }

    /// The map `append` wrote at `at`, given what it returned.
    #[inline]
    pub fn at(arena: &[u32], at: usize, words: usize, kept: usize, wide: bool) -> SparseMap<'_> {
        let pairs = &arena[at..at + 2 * words];
        let len = if wide { kept } else { kept.div_ceil(2) };
        let cols = &arena[at + 2 * words..at + 2 * words + len];
        SparseMap { words: pairs, cols: cols.as_ptr().cast(), wide, kept, _cols: std::marker::PhantomData }
    }
}

mod sealed {
    /// Seals `Element` to the numeric types this crate supports.
    pub trait Sealed {}
}

/// Numeric element types values can be read as or converted to.
pub trait Element: Copy + sealed::Sealed + 'static {
    /// Bytes per element.
    const WIDTH: usize;
    /// 0 unsigned, 1 signed, 2 float.
    #[doc(hidden)]
    const KIND: u8;
    /// Reads one element from `WIDTH` little-endian bytes.
    fn from_le(b: &[u8]) -> Self;
    /// Reads one element from `WIDTH` little-endian bytes at `p`.
    ///
    /// # Safety
    /// `p` points to `WIDTH` readable bytes.
    #[doc(hidden)]
    unsafe fn read_le(p: *const u8) -> Self;
}

/// Conversion between element types with `as` semantics.
pub trait CastInto<T: Element>: Element {
    /// `self` converted as `as` would convert it, which every route matches
    /// bit for bit.
    fn cast(self) -> T;
}

macro_rules! elements {
    ($($t:ty => $kind:literal),*) => {$(
        impl sealed::Sealed for $t {}
        impl Element for $t {
            const WIDTH: usize = std::mem::size_of::<$t>();
            const KIND: u8 = $kind;
            #[inline(always)]
            fn from_le(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b[..Self::WIDTH].try_into().unwrap())
            }
            #[inline(always)]
            unsafe fn read_le(p: *const u8) -> Self {
                // SAFETY: the caller guarantees `WIDTH` readable bytes.
                <$t>::from_le_bytes(unsafe { p.cast::<[u8; std::mem::size_of::<$t>()]>().read_unaligned() })
            }
        }
    )*};
}
elements!(u8 => 0, u16 => 0, u32 => 0, u64 => 0, i8 => 1, i16 => 1, i32 => 1, i64 => 1, f32 => 2, f64 => 2);

macro_rules! casts {
    ($s:ty => $($t:ty),*) => {$(
        impl CastInto<$t> for $s {
            #[inline(always)]
            fn cast(self) -> $t {
                self as $t
            }
        }
    )*};
}
macro_rules! all_casts {
    ($($s:ty),*) => {$( casts!($s => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64); )*};
}
all_casts!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl Dict<'_> {
    /// The `k` entries read as `S` and converted to `T` into `out[..k]`.
    /// `S` must have the chunk's value width.
    pub fn cast_into<S: CastInto<T>, T: Element>(&self, out: &mut [T]) -> Result<()> {
        if S::WIDTH != self.value_width() {
            return Err(Error::InvalidInput(
                "source element type does not match the value width",
            ));
        }
        let k = self.k();
        let have = out.len();
        let out = out
            .get_mut(..k)
            .ok_or(Error::OutputTooSmall { need: k, have })?;
        for (o, e) in out.iter_mut().zip(self.entries().chunks_exact(S::WIDTH)) {
            *o = S::from_le(e).cast();
        }
        Ok(())
    }
}

impl ModeDict<'_> {
    /// The `k` entries read as `S` and converted to `T` into `out[..k]`.
    pub fn cast_into<S: CastInto<T>, T: Element>(&self, out: &mut [T]) -> Result<()> {
        cast_values::<S, T>(self.entries(), out)
    }
}

/// Values of `S::WIDTH` little-endian bytes each, read as `S` and converted
/// to `T` into `out` (at least as many entries). For converting a chunk's
/// [`Meta::column_modes`](crate::format::Meta::column_modes) once.
pub fn cast_values<S: CastInto<T>, T: Element>(bytes: &[u8], out: &mut [T]) -> Result<()> {
    if !bytes.len().is_multiple_of(S::WIDTH) {
        return Err(Error::InvalidInput(
            "source element type does not match the value width",
        ));
    }
    let n = bytes.len() / S::WIDTH;
    let have = out.len();
    let out = out
        .get_mut(..n)
        .ok_or(Error::OutputTooSmall { need: n, have })?;
    for (o, e) in out.iter_mut().zip(bytes.chunks_exact(S::WIDTH)) {
        *o = S::from_le(e).cast();
    }
    Ok(())
}

/// Maps decoded rows of element type `S` to outputs of `T`, with the
/// kernel for every value encoding and output shape chosen once, for the
/// kernel set the mapper was bound to. Build one per element-type pair and
/// reuse it for every row; the methods behave as [`map_parts`],
/// [`map_parts_at`] and [`map_dense`].
pub struct Mapper<S, T> {
    table: RouteTable,
    _types: PhantomData<fn(S) -> T>,
}

impl<S: CastInto<T>, T: Element> Mapper<S, T> {
    /// A mapper on the process kernel set ([`crate::runtime::get`]).
    pub fn new() -> Result<Self> {
        Ok(Self::for_runtime(crate::runtime::get()?))
    }

    /// A mapper on the kernel set of `rt`.
    pub fn for_runtime(rt: &Runtime) -> Self {
        Self { table: RouteTable::new::<S, T>(rt.kernels()), _types: PhantomData }
    }

    /// Every route a kernel exists for, ignoring the measured selection
    /// policy (equivalence tests and dispatch benchmarks).
    #[cfg(test)]
    pub(crate) fn candidates(rt: &Runtime) -> Self {
        Self { table: RouteTable::candidates::<S, T>(rt.kernels()), _types: PhantomData }
    }

    /// See [`map_parts`].
    #[inline]
    pub fn map_parts(&self, parts: &Parts, map: &[u32], dict: &[T], modes: &[T], cols: &mut [u32], vals: &mut [T]) -> Result<usize> {
        self.table.csr(parts, map, Elems::of(dict), Elems::of(modes), cols, vals.as_mut_ptr().cast(), vals.len())
    }

    /// See [`map_parts_at`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn map_parts_at(&self, parts: &Parts, positions: &[u32], map: &[u32], dict: &[T], modes: &[T], cols: &mut [u32], vals: &mut [T]) -> Result<()> {
        self.table.at(parts, positions, map, Elems::of(dict), Elems::of(modes), cols, vals.as_mut_ptr().cast(), vals.len())
    }

    /// See [`map_dense`].
    #[inline]
    pub fn map_dense(&self, parts: &Parts, map: &[u32], dict: &[T], modes: &[T], out: &mut [T], fill: Fill) -> Result<()> {
        self.table.dense(parts, map, Elems::of(dict), Elems::of(modes), out.as_mut_ptr().cast(), out.len(), fill)
    }
}

/// Maps a decoded row: for each element, `col = map[id]`; elements whose
/// column is [`UNMAPPED`] are dropped, the rest are written in order to
/// `cols` and `vals`. Values are `dict[rank]` for a dictionary (`dict`
/// holds its entries converted by [`Dict::cast_into`]) or the raw values
/// read as `S` and converted. For mode-coded values, `modes` holds the
/// chunk's column modes converted to `T` (see [`cast_values`]), indexed by
/// coded id; it may be empty for chunks without column modes. `cols` and
/// `vals` need room for every element of the row; `map` must cover the
/// row's largest id. Returns the number of elements kept. Columns keep the
/// row's id order, so they are ascending only if the map is monotonic.
pub fn map_parts<S: CastInto<T>, T: Element>(
    parts: &Parts,
    map: &[u32],
    dict: &[T],
    modes: &[T],
    cols: &mut [u32],
    vals: &mut [T],
) -> Result<usize> {
    let k = crate::runtime::get()?.kernels();
    route::csr_with(k, S::WIDTH, |e| route::csr_route::<S, T>(k, e), parts, map, Elems::of(dict), Elems::of(modes), cols, vals.as_mut_ptr().cast(), vals.len())
}

/// Maps the elements of a decoded row at `positions` (each below the row's
/// nnz, in any order, repeats allowed): `cols[i] = map[id]` and `vals[i]`
/// the value of element `positions[i]`, with `dict` and `modes` as in
/// [`map_parts`]. Nothing is dropped: an [`UNMAPPED`] column is written as
/// it is. For callers that need only some elements of a row, such as a
/// sampler choosing positions first. A position at or beyond the row's nnz
/// is [`Error::InvalidInput`](crate::Error::InvalidInput).
pub fn map_parts_at<S: CastInto<T>, T: Element>(
    parts: &Parts,
    positions: &[u32],
    map: &[u32],
    dict: &[T],
    modes: &[T],
    cols: &mut [u32],
    vals: &mut [T],
) -> Result<()> {
    let k = crate::runtime::get()?.kernels();
    route::at_with(k, S::WIDTH, |e| route::at_route::<S, T>(k, e), parts, positions, map, Elems::of(dict), Elems::of(modes), cols, vals.as_mut_ptr().cast(), vals.len())
}

/// Scatters a decoded row into a dense output row: `out[map[id]]` gets each
/// element's value, with `dict` and `modes` as in [`map_parts`]; elements
/// whose column is [`UNMAPPED`] are skipped. With [`Fill::Zero`] the row is
/// cleared first. Every mapped column must lie inside `out`.
pub fn map_dense<S: CastInto<T>, T: Element>(
    parts: &Parts,
    map: &[u32],
    dict: &[T],
    modes: &[T],
    out: &mut [T],
    fill: Fill,
) -> Result<()> {
    let k = crate::runtime::get()?.kernels();
    route::dense_with(k, S::WIDTH, T::WIDTH, |e| route::dense_route::<S, T>(k, e), parts, map, Elems::of(dict), Elems::of(modes), out.as_mut_ptr().cast(), out.len(), fill)
}

/// Branch-free compaction: every element is written, and kept by advancing
/// only when its column is mapped.
///
/// # Safety
/// Every id indexes `map`; `value` accepts every position below
/// `ids.len()`; `cols` and `vals` hold at least `ids.len()` entries.
#[inline(always)]
pub(crate) unsafe fn compact<T: Copy>(
    ids: &[u32],
    map: &[u32],
    value: impl Fn(usize) -> T,
    cols: &mut [u32],
    vals: &mut [T],
) -> usize {
    debug_assert!(cols.len() >= ids.len() && vals.len() >= ids.len());
    let mut j = 0;
    for (t, &id) in ids.iter().enumerate() {
        // SAFETY: ids index `map`, and `j <= t < ids.len()` bounds both
        // outputs, per the contract.
        unsafe {
            let col = unchecked::get(map, id as usize);
            unchecked::set(cols, j, col);
            unchecked::set(vals, j, value(t));
            j += (col != UNMAPPED) as usize;
        }
    }
    j
}
