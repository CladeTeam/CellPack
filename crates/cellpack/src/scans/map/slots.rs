//! Route kernels: one monomorphic function per element-type pair, value
//! encoding and route, over inputs [`route::check`](super::route) validated.
//! The encoding is a constant parameter, so a row's views are extracted
//! once and inner loops carry no per-element dispatch. Indices are bounded
//! by the route contract and read unchecked.
//!
//! The contract every function relies on: `ENC` is the encoding of
//! `a.parts`; every id indexes `a.map` (and `a.modes` for mode encodings);
//! `a.dict` holds exactly the row's `k` entries; ranks are below `k`
//! (dictionaries) or at most `k` (mode encodings), which decoding
//! establishes for well-formed records; raw values cover every element;
//! positions lie below the row length; every index fits a signed 32-bit
//! gather; and the outputs hold the elements each shape writes.

use std::hint::unreachable_unchecked;
use std::slice;

use super::route::{Args, CONST, DICT8, DICT32, Elems, MODE, MODES8, MODES32, RAW, lanes};
use super::{CastInto, Element, SparseMap, UNMAPPED, compact};
use crate::scans::decode::{Ranks, Values};
use crate::scans::kernels::{At, Kernels, out_slot};
use crate::unchecked::{get, set};

#[inline(always)]
unsafe fn elems<'a, T>(e: Elems) -> &'a [T] {
    // SAFETY: `Elems` describe `len` live, aligned elements of `T`.
    unsafe { slice::from_raw_parts(e.ptr.cast(), e.len) }
}

#[inline(always)]
unsafe fn cols<'a>(a: &Args<'a>, n: usize) -> &'a mut [u32] {
    // SAFETY: the shape's validation reserved `n` output columns.
    unsafe { slice::from_raw_parts_mut(a.cols, n) }
}

#[inline(always)]
unsafe fn vals<'a, T>(a: &Args<'a>, n: usize) -> &'a mut [T] {
    // SAFETY: the shape's validation reserved `n` aligned output elements.
    unsafe { slice::from_raw_parts_mut(a.vals.cast(), n) }
}

/// The views of one row in encoding `ENC`, extracted once.
#[derive(Clone, Copy)]
struct Row<'a, T> {
    ids: &'a [u32],
    raw: &'a [u8],
    r8: &'a [u8],
    r32: &'a [u32],
    dict: &'a [T],
    modes: &'a [T],
}

#[inline(always)]
fn ranks<'a>(v: Values<'a>) -> Ranks<'a> {
    match v {
        Values::Dict(d) => d.ranks(),
        Values::Modes(d) => d.ranks(),
        Values::Raw(_) => Ranks::One,
    }
}

impl<'a, T: Element> Row<'a, T> {
    /// # Safety
    /// The route contract, with `T` the width of the converted values.
    #[inline(always)]
    unsafe fn new<const ENC: u8>(a: &Args<'a>) -> Self {
        let v = a.parts.values;
        // SAFETY: `ENC` is the row's encoding, so each view below exists.
        unsafe {
            Self {
                ids: a.parts.ids,
                raw: if ENC == RAW { match v { Values::Raw(b) => b, _ => unreachable_unchecked() } } else { &[] },
                r8: if ENC == DICT8 || ENC == MODES8 { match ranks(v) { Ranks::U8(r) => r, _ => unreachable_unchecked() } } else { &[] },
                r32: if ENC == DICT32 || ENC == MODES32 { match ranks(v) { Ranks::U32(r) => r, _ => unreachable_unchecked() } } else { &[] },
                dict: elems(a.dict),
                modes: elems(a.modes),
            }
        }
    }

    /// Value of element `t`: the column mode for rank 0, else a dictionary
    /// entry. Both are read and one is selected, without a branch.
    #[inline(always)]
    unsafe fn mode(&self, t: usize, rank: usize) -> T {
        // SAFETY: ids index the modes. Rows with mode ranks have `k >= 1`
        // (decoding gives a `k = 0` row no ranks), so rank 0 reads entry 0
        // (discarded) and a rank `r <= k` reads entry `r - 1`.
        unsafe {
            debug_assert!(!self.dict.is_empty());
            let m = get(self.modes, get(self.ids, t) as usize);
            let d = get(self.dict, rank.max(1) - 1);
            if rank == 0 { m } else { d }
        }
    }

    /// Value of element `t`, read as `S` and converted for raw values.
    #[inline(always)]
    unsafe fn value<S: CastInto<T>, const ENC: u8>(&self, t: usize) -> T {
        // SAFETY: `t` is below the row length; the route contract bounds
        // the raw bytes, ranks, dictionary and modes it indexes.
        unsafe {
            match ENC {
                RAW => S::read_le(self.raw.as_ptr().add(t * S::WIDTH)).cast(),
                CONST => get(self.dict, 0),
                DICT8 => get(self.dict, get(self.r8, t) as usize),
                DICT32 => get(self.dict, get(self.r32, t) as usize),
                MODE => get(self.modes, get(self.ids, t) as usize),
                MODES8 => self.mode(t, get(self.r8, t) as usize),
                MODES32 => self.mode(t, get(self.r32, t) as usize),
                _ => unreachable_unchecked(),
            }
        }
    }
}

/// The bits of an element of at most 4 bytes, zero-extended.
#[inline(always)]
fn lane_bits<T: Element>(v: T) -> u32 {
    let mut b = [0u8; 4];
    // SAFETY: elements of at most 4 bytes are plain numbers; the copy
    // reads exactly their bytes.
    unsafe { std::ptr::copy_nonoverlapping((&v as *const T).cast::<u8>(), b.as_mut_ptr(), T::WIDTH.min(4)) };
    u32::from_le_bytes(b)
}

// ---- Scalar loops: every pair, encoding and shape. ----

/// Compact mapping of encoding `ENC` in portable code: [`compact`] over the
/// row's values.
pub(super) unsafe fn csr_scalar<S: CastInto<T>, T: Element, const ENC: u8>(_: &Kernels, a: &Args) -> usize {
    let n = a.parts.ids.len();
    // SAFETY: the route contract; outputs hold the row's elements.
    unsafe {
        let row = Row::<T>::new::<ENC>(a);
        compact(row.ids, a.map, |t| row.value::<S, ENC>(t), cols(a, n), vals::<T>(a, n))
    }
}

/// Indexed mapping of encoding `ENC` in portable code, one element per
/// position.
pub(super) unsafe fn at_scalar<S: CastInto<T>, T: Element, const ENC: u8>(_: &Kernels, a: &Args) {
    let m = a.positions.len();
    // SAFETY: the route contract; positions lie below the row length and
    // the outputs hold one entry per position.
    unsafe {
        let row = Row::<T>::new::<ENC>(a);
        let (cols, vals) = (cols(a, m), vals::<T>(a, m));
        for (i, &p) in a.positions.iter().enumerate() {
            let p = p as usize;
            set(cols, i, get(a.map, get(row.ids, p) as usize));
            set(vals, i, row.value::<S, ENC>(p));
        }
    }
}

/// Dense scatter of encoding `ENC` in portable code; false when a mapped
/// column lies outside the row.
pub(super) unsafe fn dense_scalar<S: CastInto<T>, T: Element, const ENC: u8>(_: &Kernels, a: &Args) -> bool {
    // SAFETY: the route contract; every store's column is checked against
    // the row length first.
    unsafe {
        let row = Row::<T>::new::<ENC>(a);
        let out = vals::<T>(a, a.out_len);
        for (t, &id) in row.ids.iter().enumerate() {
            let col = get(a.map, id as usize);
            if col != UNMAPPED {
                if col as usize >= out.len() {
                    return false;
                }
                set(out, col as usize, row.value::<S, ENC>(t));
            }
        }
    }
    true
}

/// Dense scatter through a [`SparseMap`]: a block of ids is filtered to
/// the positions of its mapped elements (sixteen bit tests per step on
/// AVX-512), then only those elements are ranked, read and stored.
pub(super) unsafe fn dense_sparse<S: CastInto<T>, T: Element, const ENC: u8>(k: &Kernels, a: &Args, map: &SparseMap) -> bool {
    const BLOCK: usize = 256;
    // SAFETY: the route contract with ids bounded by the map's bits; ranks
    // of marked ids index `cols` by construction; every store's column is
    // checked against the row length first.
    unsafe {
        let row = Row::<T>::new::<ENC>(a);
        let out = vals::<T>(a, a.out_len);
        let ids = row.ids;
        let mut hits = [0u32; BLOCK + 16];
        let mut start = 0;
        while start < ids.len() {
            let block = &ids[start..ids.len().min(start + BLOCK)];
            let found = filter(k, block, map.words, &mut hits);
            for &h in &hits[..found] {
                let t = start + h as usize;
                let id = get(ids, t);
                let w = 2 * (id >> 5) as usize;
                let rank = get(map.words, w + 1) + (get(map.words, w) & ((1u32 << (id & 31)) - 1)).count_ones();
                let col = map.col(rank as usize);
                if col as usize >= out.len() {
                    return false;
                }
                set(out, col as usize, row.value::<S, ENC>(t));
            }
            start += BLOCK;
        }
    }
    true
}

/// Writes the positions in `ids` (at most 256) whose bit is set to `hits`
/// (room for 16 more), in order; returns their count.
///
/// # Safety
/// Every id indexes `words` (bit `id % 32` of word `2 * (id / 32)`).
#[inline(always)]
unsafe fn filter(k: &Kernels, ids: &[u32], bits: &[u32], hits: &mut [u32]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if k.isa == crate::scans::kernels::Isa::Avx512 {
        // SAFETY: the kernel set is AVX-512; as documented.
        return unsafe { filter_avx512(ids, bits, hits) };
    }
    let _ = k;
    let mut n = 0;
    for (i, &id) in ids.iter().enumerate() {
        // SAFETY: as documented; `n <= i` stays inside `hits`.
        unsafe {
            set(hits, n, i as u32);
            n += ((get(bits, 2 * (id >> 5) as usize) >> (id & 31)) & 1) as usize;
        }
    }
    n
}

/// [`filter`] 16 ids at a time: gather each id's word, test its bit and
/// compress the positions of the marked ids.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn filter_avx512(ids: &[u32], bits: &[u32], hits: &mut [u32]) -> usize {
    use std::arch::x86_64::*;
    let n = ids.len();
    let lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let (low, one) = (_mm512_set1_epi32(31), _mm512_set1_epi32(1));
    let mut found = 0;
    let mut i = 0;
    while i < n {
        let tail = if n - i >= 16 { 0xffff } else { ((1u32 << (n - i)) - 1) as u16 };
        // SAFETY: masked lanes stay in the block; every id's word lies in
        // `bits`; `hits` has 16 entries of room past the block's count.
        unsafe {
            let v = _mm512_maskz_loadu_epi32(tail, ids.as_ptr().add(i).cast());
            let w = _mm512_mask_i32gather_epi32::<8>(_mm512_setzero_si512(), tail, _mm512_srli_epi32::<5>(v), bits.as_ptr().cast());
            let hit = _mm512_mask_test_epi32_mask(tail, _mm512_srlv_epi32(w, _mm512_and_si512(v, low)), one);
            let at = _mm512_maskz_compress_epi32(hit, _mm512_add_epi32(lanes, _mm512_set1_epi32(i as i32)));
            _mm512_storeu_si512(hits.as_mut_ptr().add(found).cast(), at);
            found += hit.count_ones() as usize;
        }
        i += 16;
    }
    found
}

// ---- Eight-lane typed kernels (AVX-512): wide and narrow outputs. ----

/// Compact mapping through the eight-lane typed kernel.
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn csr_typed<S: CastInto<T>, T: Element>(_: &Kernels, a: &Args) -> usize {
    let n = a.parts.ids.len();
    // SAFETY: the route contract; the kernel set is AVX-512.
    unsafe {
        crate::scans::kernels::typed::map::<S, T, 0>(a.parts, &[], a.map, elems(a.dict), elems(a.modes), cols(a, n), vals::<T>(a, n))
            .unwrap_unchecked()
    }
}

/// Indexed mapping through the eight-lane typed kernel.
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn at_typed<S: CastInto<T>, T: Element>(_: &Kernels, a: &Args) {
    let m = a.positions.len();
    // SAFETY: as in `csr_typed`, with outputs of one entry per position.
    unsafe {
        crate::scans::kernels::typed::map::<S, T, 1>(a.parts, a.positions, a.map, elems(a.dict), elems(a.modes), cols(a, m), vals::<T>(a, m));
    }
}

/// Dense scatter through the eight-lane typed kernel.
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn dense_typed<S: CastInto<T>, T: Element>(_: &Kernels, a: &Args) -> bool {
    // SAFETY: as in `csr_typed`; the kernel checks mapped columns against
    // the dense row before storing.
    unsafe {
        crate::scans::kernels::typed::map::<S, T, 2>(a.parts, &[], a.map, elems(a.dict), elems(a.modes), &mut [], vals::<T>(a, a.out_len))
            .is_some()
    }
}

// ---- Sixteen-lane 32-bit kernels: 4-byte outputs, lane conversions. ----

/// 4-byte outputs (bit patterns of `u32`, `i32`, `f32`) from dictionary
/// and mode encodings.
// `S` keeps the signature uniform across the routes `by_encoding!` builds.
#[allow(clippy::extra_unused_type_parameters)]
pub(super) unsafe fn csr_wide<S: CastInto<T>, T: Element, const ENC: u8>(k: &Kernels, a: &Args) -> usize {
    let n = a.parts.ids.len();
    // SAFETY: the route contract; `T` has 4 bytes, so its converted values
    // are read and written as their bits.
    unsafe {
        let row = Row::<u32>::new::<ENC>(a);
        let (cols, vals) = (cols(a, n), vals::<u32>(a, n));
        match ENC {
            CONST => {
                let c = get(row.dict, 0);
                compact(row.ids, a.map, |_| c, cols, vals)
            }
            DICT8 => (k.map_u8)(row.ids, a.map, row.r8, row.dict, cols, vals),
            DICT32 => (k.map_u32)(row.ids, a.map, row.r32, row.dict, cols, vals),
            // SAFETY: ids index the modes.
            MODE => compact(row.ids, a.map, |t| get(row.modes, get(row.ids, t) as usize), cols, vals),
            MODES8 => (k.map_modes_u8)(row.ids, a.map, row.r8, row.dict, row.modes, cols, vals),
            MODES32 => (k.map_modes_u32)(row.ids, a.map, row.r32, row.dict, row.modes, cols, vals),
            _ => unreachable_unchecked(),
        }
    }
}

/// Raw values converted by a lane kernel of the set.
pub(super) unsafe fn csr_lanes<S: CastInto<T>, T: Element>(k: &Kernels, a: &Args) -> usize {
    let n = a.parts.ids.len();
    // SAFETY: the route only exists where the pair has a lane conversion
    // and the set a kernel for it; the route contract covers the call.
    unsafe {
        let (src, conv) = lanes::<S, T>().unwrap_unchecked();
        let f = k.map_lanes[src as usize][conv as usize][out_slot(T::WIDTH)].unwrap_unchecked();
        let row = Row::<T>::new::<RAW>(a);
        f(row.ids, a.map, row.raw, cols(a, n), a.vals)
    }
}

/// Raw 4-byte values of the output's own type, copied unchanged.
pub(super) unsafe fn csr_raw32<T: Element>(k: &Kernels, a: &Args) -> usize {
    let n = a.parts.ids.len();
    // SAFETY: the route contract, with 4-byte values on both sides.
    unsafe {
        let row = Row::<T>::new::<RAW>(a);
        (k.map_raw32)(row.ids, a.map, row.raw, cols(a, n), vals::<u32>(a, n))
    }
}

/// Byte-ranked dictionaries to 1- or 2-byte outputs through the entries
/// widened to 32-bit lanes.
pub(super) unsafe fn csr_dict8_lanes<T: Element>(k: &Kernels, a: &Args) -> usize {
    let n = a.parts.ids.len();
    // SAFETY: byte ranks name at most 192 entries, which the widened table
    // holds; the route exists only where the set has the kernel.
    unsafe {
        let row = Row::<T>::new::<DICT8>(a);
        let mut wide = [0u32; 256];
        for (w, &e) in wide.iter_mut().zip(row.dict) {
            *w = lane_bits(e);
        }
        let f = k.map_dict8_lanes[out_slot(T::WIDTH)].unwrap_unchecked();
        f(row.ids, a.map, row.r8, &wide[..row.dict.len()], cols(a, n), a.vals)
    }
}

/// The row's values as 4-byte lanes for the position and scatter kernels.
#[inline(always)]
unsafe fn lane_values<'a, S: CastInto<T>, T: Element, const ENC: u8>(a: &Args<'a>) -> At<'a> {
    // SAFETY: the route contract; raw routes exist only for pairs with a
    // lane conversion.
    unsafe {
        let row = Row::<u32>::new::<ENC>(a);
        match ENC {
            RAW => {
                let (src, conv) = lanes::<S, T>().unwrap_unchecked();
                At::Raw(row.raw, src, conv)
            }
            CONST => At::Const(get(row.dict, 0)),
            DICT8 => At::Dict8(row.r8, row.dict),
            DICT32 => At::Dict32(row.r32, row.dict),
            MODE => At::ModesOne(row.modes),
            MODES8 => At::Modes8(row.r8, row.dict, row.modes),
            MODES32 => At::Modes32(row.r32, row.dict, row.modes),
            _ => unreachable_unchecked(),
        }
    }
}

/// Indexed mapping through the 4-byte position kernel.
pub(super) unsafe fn at_wide<S: CastInto<T>, T: Element, const ENC: u8>(k: &Kernels, a: &Args) {
    let m = a.positions.len();
    // SAFETY: the route contract, with 4-byte outputs, one per position.
    unsafe { (k.map_at)(a.parts.ids, a.positions, a.map, lane_values::<S, T, ENC>(a), cols(a, m), vals::<u32>(a, m)) }
}

/// Dense scatter through the 4-byte scatter kernel.
pub(super) unsafe fn dense_wide<S: CastInto<T>, T: Element, const ENC: u8>(k: &Kernels, a: &Args) -> bool {
    // SAFETY: the route contract, with a 4-byte dense row; the kernel
    // checks mapped columns before storing.
    unsafe { (k.scatter)(a.parts.ids, a.map, lane_values::<S, T, ENC>(a), vals::<u32>(a, a.out_len)) }
}
