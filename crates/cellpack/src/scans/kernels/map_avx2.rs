//! AVX2 mapping, field and running-sum kernels, 8 lanes per step.
//!
//! Columns come from a gather of the column map. AVX2 has no compress, so
//! the kept lanes are packed with a permutation looked up by the 8-bit
//! keep mask, then stored as full vectors at the packed position, which
//! never passes the current element. AVX2 has no scatter either: the
//! scatter kernel computes columns and values with vectors and stores the
//! kept lanes one by one. Every kernel has the contract of its function
//! type in [`super`] and matches the scalar kernel bit for bit.

use std::arch::x86_64::*;

use super::x86::BIT_POSITIONS;
use super::{At, Conv, LanesTable, Src, ascend_tail, at_value, conv_value, fields_tail, src_value};
use crate::scans::map::compact;
use crate::scans::symbol::{DIRECT, FIRST_BITS};
use crate::unchecked::{get, set};

/// Lanes not holding the unmapped column, as 8 mask bits.
#[target_feature(enable = "avx2")]
#[inline]
fn keep8(col: __m256i) -> u32 {
    let unmapped = _mm256_cmpeq_epi32(col, _mm256_set1_epi32(-1));
    !(_mm256_movemask_ps(_mm256_castsi256_ps(unmapped)) as u32) & 0xff
}

/// The permutation packing the lanes of an 8-bit mask to the front.
#[target_feature(enable = "avx2")]
#[inline]
fn packing(keep: u32) -> __m256i {
    // SAFETY: each table row holds 8 entries.
    unsafe { _mm256_loadu_si256(BIT_POSITIONS[(keep & 0xff) as usize].as_ptr().cast()) }
}

/// Stores the kept lanes packed at `cols` and `vals`; returns the count.
///
/// # Safety
/// `cols` and `vals` have 8 writable entries.
#[target_feature(enable = "avx2,popcnt")]
#[inline]
unsafe fn put8(keep: u32, col: __m256i, v: __m256i, cols: *mut u32, vals: *mut u32) -> usize {
    let idx = packing(keep);
    // SAFETY: as documented.
    unsafe {
        _mm256_storeu_si256(cols.cast(), _mm256_permutevar8x32_epi32(col, idx));
        _mm256_storeu_si256(vals.cast(), _mm256_permutevar8x32_epi32(v, idx));
    }
    keep.count_ones() as usize
}

/// Dictionary entries for 8 ranks: a register permute when the dictionary
/// fits one vector, else a gather.
struct Lookup8 {
    table: __m256i,
    small: bool,
}

impl Lookup8 {
    #[target_feature(enable = "avx2")]
    #[inline]
    fn new(dict: &[u32]) -> Self {
        let mut t = [0u32; 8];
        let small = dict.len() <= 8;
        if small {
            t[..dict.len()].copy_from_slice(dict);
        }
        // SAFETY: `t` holds 8 entries.
        Self { table: unsafe { _mm256_loadu_si256(t.as_ptr().cast()) }, small }
    }

    /// Entries at `rank`.
    ///
    /// # Safety
    /// Every rank is below the dictionary length.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn get(&self, rank: __m256i, dict: &[u32]) -> __m256i {
        if self.small {
            _mm256_permutevar8x32_epi32(self.table, rank)
        } else {
            // SAFETY: as documented.
            unsafe { _mm256_i32gather_epi32::<4>(dict.as_ptr().cast(), rank) }
        }
    }

    /// Entries at `rank` in the lanes of `mask` (all ones), `src` elsewhere.
    ///
    /// # Safety
    /// Ranks of the masked lanes are below the dictionary length.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn get_masked(&self, src: __m256i, mask: __m256i, rank: __m256i, dict: &[u32]) -> __m256i {
        if self.small {
            _mm256_blendv_epi8(src, _mm256_permutevar8x32_epi32(self.table, rank), mask)
        } else {
            // SAFETY: as documented; unmasked lanes read no memory.
            unsafe { _mm256_mask_i32gather_epi32::<4>(src, dict.as_ptr().cast(), rank, mask) }
        }
    }
}

/// Mode-coded values: rank 0 takes the column mode, rank `r` entry `r - 1`.
///
/// # Safety
/// Ids index `modes`; ranks are at most the dictionary length.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mode_values(id: __m256i, rank: __m256i, lookup: &Lookup8, dict: &[u32], modes: &[u32]) -> __m256i {
    let zero = _mm256_setzero_si256();
    let is_mode = _mm256_cmpeq_epi32(rank, zero);
    let exception = _mm256_xor_si256(is_mode, _mm256_set1_epi32(-1));
    // SAFETY: as documented; each gather reads only its own lanes.
    unsafe {
        let m = _mm256_mask_i32gather_epi32::<4>(zero, modes.as_ptr().cast(), id, is_mode);
        lookup.get_masked(m, exception, _mm256_sub_epi32(rank, _mm256_set1_epi32(1)), dict)
    }
}

/// Byte ranks of 8 elements: contiguous from element `i`.
///
/// # Safety
/// `ranks` holds elements `i..i + 8`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ranks8(ranks: &[u8], i: usize) -> __m256i {
    // SAFETY: as documented.
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(ranks.as_ptr().add(i).cast())) }
}

/// Byte ranks at positions, gathered as 4-byte words; lanes within 3 bytes
/// of the end are left out of the returned mask (rank 0) for the caller.
///
/// # Safety
/// Every position lies below `ranks.len()`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ranks_at(pos: __m256i, ranks: &[u8]) -> (__m256i, u32) {
    let safe = _mm256_cmpgt_epi32(_mm256_set1_epi32(ranks.len().saturating_sub(3) as i32), pos);
    // SAFETY: masked lanes read 4 bytes ending inside `ranks`.
    let w = unsafe { _mm256_mask_i32gather_epi32::<1>(_mm256_setzero_si256(), ranks.as_ptr().cast(), pos, safe) };
    (_mm256_and_si256(w, _mm256_set1_epi32(0xff)), _mm256_movemask_ps(_mm256_castsi256_ps(safe)) as u32)
}

/// The mask of `safe` lanes as a vector.
#[target_feature(enable = "avx2")]
#[inline]
fn lanes_of(mask: u32) -> __m256i {
    let bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(mask as i32), bit), bit)
}

/// The [`super::MapU8Fn`] contract, 8 elements per step.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn map_u8_avx2(ids: &[u32], map: &[u32], ranks: &[u8], dict: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: loads cover elements `i..i + 8` with `i + 8 <= n`; gathers
    // index `map` by ids and the dictionary by ranks per the `MapU8Fn`
    // contract; stores at `j <= i` stay inside the outputs of `n` entries.
    unsafe {
        let lookup = Lookup8::new(dict);
        while i + 8 <= n {
            let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), _mm256_loadu_si256(ids.as_ptr().add(i).cast()));
            let v = lookup.get(ranks8(ranks, i), dict);
            j += put8(keep8(col), col, v, cols.as_mut_ptr().add(j), vals.as_mut_ptr().add(j));
            i += 8;
        }
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| get(dict, get(r, t) as usize), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// The [`super::MapU32Fn`] contract, 8 elements per step.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn map_u32_avx2(ids: &[u32], map: &[u32], ranks: &[u32], dict: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_u8_avx2`, with 32-bit ranks.
    unsafe {
        let lookup = Lookup8::new(dict);
        while i + 8 <= n {
            let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), _mm256_loadu_si256(ids.as_ptr().add(i).cast()));
            let v = lookup.get(_mm256_loadu_si256(ranks.as_ptr().add(i).cast()), dict);
            j += put8(keep8(col), col, v, cols.as_mut_ptr().add(j), vals.as_mut_ptr().add(j));
            i += 8;
        }
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| get(dict, get(r, t) as usize), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// The [`super::MapModesU8Fn`] contract, 8 elements per step.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn map_modes_u8_avx2(ids: &[u32], map: &[u32], ranks: &[u8], dict: &[u32], modes: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_u8_avx2`, with the mode gathers of `mode_values`
    // per the `MapModesU8Fn` contract.
    unsafe {
        let lookup = Lookup8::new(dict);
        while i + 8 <= n {
            let id = _mm256_loadu_si256(ids.as_ptr().add(i).cast());
            let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), id);
            let v = mode_values(id, ranks8(ranks, i), &lookup, dict, modes);
            j += put8(keep8(col), col, v, cols.as_mut_ptr().add(j), vals.as_mut_ptr().add(j));
            i += 8;
        }
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| super::mode_value(get(ids, t), get(r, t) as usize, dict, modes), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// The [`super::MapModesU32Fn`] contract, 8 elements per step.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn map_modes_u32_avx2(ids: &[u32], map: &[u32], ranks: &[u32], dict: &[u32], modes: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_modes_u8_avx2`, with 32-bit ranks.
    unsafe {
        let lookup = Lookup8::new(dict);
        while i + 8 <= n {
            let id = _mm256_loadu_si256(ids.as_ptr().add(i).cast());
            let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), id);
            let v = mode_values(id, _mm256_loadu_si256(ranks.as_ptr().add(i).cast()), &lookup, dict, modes);
            j += put8(keep8(col), col, v, cols.as_mut_ptr().add(j), vals.as_mut_ptr().add(j));
            i += 8;
        }
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| super::mode_value(get(ids, t), get(r, t) as usize, dict, modes), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// The [`super::MapRaw32Fn`] contract, 8 elements per step.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn map_raw32_avx2(ids: &[u32], map: &[u32], raw: &[u8], cols: &mut [u32], vals: &mut [u32]) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_u8_avx2`, with `raw` holding one 4-byte value per id.
    unsafe {
        while i + 8 <= n {
            let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), _mm256_loadu_si256(ids.as_ptr().add(i).cast()));
            let v = _mm256_loadu_si256(raw.as_ptr().add(4 * i).cast());
            j += put8(keep8(col), col, v, cols.as_mut_ptr().add(j), vals.as_mut_ptr().add(j));
            i += 8;
        }
        let (ids, r) = (ids.get_unchecked(i..), raw.get_unchecked(4 * i..));
        j + compact(ids, map, |t| src_value(r, t, Src::W32), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// Raw values of elements `i..i + 8` as 32-bit lanes.
///
/// # Safety
/// The values hold elements `i..i + 8`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load8(raw: *const u8, i: usize, src: Src) -> __m256i {
    // SAFETY: as documented.
    unsafe {
        match src {
            Src::U8 => _mm256_cvtepu8_epi32(_mm_loadl_epi64(raw.add(i).cast())),
            Src::I8 => _mm256_cvtepi8_epi32(_mm_loadl_epi64(raw.add(i).cast())),
            Src::U16 => _mm256_cvtepu16_epi32(_mm_loadu_si128(raw.add(2 * i).cast())),
            Src::I16 => _mm256_cvtepi16_epi32(_mm_loadu_si128(raw.add(2 * i).cast())),
            Src::W32 => _mm256_loadu_si256(raw.add(4 * i).cast()),
        }
    }
}

/// Lane conversions with `as` semantics. AVX2 converts only signed
/// integers, so an unsigned lane is converted as its high and low 16 bits,
/// whose exact sum rounds once, as `u32 as f32` does.
#[target_feature(enable = "avx2")]
#[inline]
fn conv8(v: __m256i, conv: Conv) -> __m256i {
    match conv {
        Conv::Bits => v,
        Conv::I32ToF32 => _mm256_castps_si256(_mm256_cvtepi32_ps(v)),
        Conv::U32ToF32 => {
            let hi = _mm256_cvtepi32_ps(_mm256_srli_epi32::<16>(v));
            let lo = _mm256_cvtepi32_ps(_mm256_and_si256(v, _mm256_set1_epi32(0xffff)));
            _mm256_castps_si256(_mm256_add_ps(_mm256_mul_ps(hi, _mm256_set1_ps(65536.0)), lo))
        }
    }
}

/// Stores 8 lanes truncated to `OUT` bytes each.
///
/// # Safety
/// `vals` has room for 8 elements of `OUT` bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_narrow<const OUT: usize>(vals: *mut u8, v: __m256i) {
    // SAFETY: as documented.
    unsafe {
        match OUT {
            4 => _mm256_storeu_si256(vals.cast(), v),
            2 => {
                let low = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
                let p = _mm256_permute4x64_epi64::<0b1000>(_mm256_shuffle_epi8(v, low));
                _mm_storeu_si128(vals.cast(), _mm256_castsi256_si128(p));
            }
            _ => {
                let low = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                let p = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v, low), _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0));
                _mm_storel_epi64(vals.cast(), _mm256_castsi256_si128(p));
            }
        }
    }
}

/// Stores the kept lanes packed: columns, then values truncated to `OUT`
/// bytes; returns the count.
///
/// # Safety
/// `cols` has 8 writable entries and `vals` 8 elements of `OUT` bytes.
#[target_feature(enable = "avx2,popcnt")]
#[inline]
unsafe fn put_lanes<const OUT: usize>(keep: u32, col: __m256i, v: __m256i, cols: *mut u32, vals: *mut u8) -> usize {
    let idx = packing(keep);
    // SAFETY: as documented.
    unsafe {
        _mm256_storeu_si256(cols.cast(), _mm256_permutevar8x32_epi32(col, idx));
        store_narrow::<OUT>(vals, _mm256_permutevar8x32_epi32(v, idx));
    }
    keep.count_ones() as usize
}

/// Writes the low `OUT` bytes of a lane value.
///
/// # Safety
/// `vals` has room for element `j`.
#[inline(always)]
unsafe fn put_value<const OUT: usize>(vals: *mut u8, j: usize, v: u32) {
    // SAFETY: as documented.
    unsafe { std::ptr::copy_nonoverlapping(v.to_le_bytes().as_ptr(), vals.add(j * OUT), OUT) };
}

const fn src_of(code: u8) -> Src {
    match code {
        0 => Src::U8,
        1 => Src::I8,
        2 => Src::U16,
        3 => Src::I16,
        _ => Src::W32,
    }
}

const fn conv_of(code: u8) -> Conv {
    match code {
        0 => Conv::Bits,
        1 => Conv::U32ToF32,
        _ => Conv::I32ToF32,
    }
}

/// Raw values of source kind `SRC`, converted by `CONV`, mapped 8 at a time
/// to `OUT`-byte elements (see [`super::MapLanesFn`]).
#[target_feature(enable = "avx2,popcnt")]
unsafe fn map_lanes_avx2<const SRC: u8, const CONV: u8, const OUT: usize>(ids: &[u32], map: &[u32], raw: &[u8], cols: &mut [u32], vals: *mut u8) -> usize {
    let (src, conv) = (src_of(SRC), conv_of(CONV));
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: loads cover elements `i..i + 8` with `i + 8 <= n`; gathers
    // index `map` by ids in range per the caller's contract; stores write
    // entries `j..j + 8` with `j <= i`, inside the outputs of `n` entries.
    unsafe {
        while i + 8 <= n {
            let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), _mm256_loadu_si256(ids.as_ptr().add(i).cast()));
            let v = conv8(load8(raw.as_ptr(), i, src), conv);
            j += put_lanes::<OUT>(keep8(col), col, v, cols.as_mut_ptr().add(j), vals.add(j * OUT));
            i += 8;
        }
        for t in i..n {
            let col = get(map, get(ids, t) as usize);
            set(cols, j, col);
            put_value::<OUT>(vals, j, conv_value(src_value(raw, t, src), conv));
            j += (col != crate::scans::UNMAPPED) as usize;
        }
    }
    j
}

/// Byte-ranked dictionary values (entries as 32-bit lanes) mapped 8 at a
/// time to `OUT`-byte elements (see [`super::MapDictLanesFn`]).
#[target_feature(enable = "avx2,popcnt")]
unsafe fn map_dict8_lanes_avx2<const OUT: usize>(ids: &[u32], map: &[u32], ranks: &[u8], dict: &[u32], cols: &mut [u32], vals: *mut u8) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_lanes_avx2`, ranks indexing the dictionary per
    // the caller's contract.
    unsafe {
        let lookup = Lookup8::new(dict);
        while i + 8 <= n {
            let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), _mm256_loadu_si256(ids.as_ptr().add(i).cast()));
            let v = lookup.get(ranks8(ranks, i), dict);
            j += put_lanes::<OUT>(keep8(col), col, v, cols.as_mut_ptr().add(j), vals.add(j * OUT));
            i += 8;
        }
        for t in i..n {
            let col = get(map, get(ids, t) as usize);
            set(cols, j, col);
            put_value::<OUT>(vals, j, get(dict, get(ranks, t) as usize));
            j += (col != crate::scans::UNMAPPED) as usize;
        }
    }
    j
}

/// Lane kernels for every source kind, conversion and output width they
/// cover; the rest stay `None`.
pub(super) const LANES_AVX2: LanesTable = {
    let mut t: LanesTable = [[[None; 3]; 3]; 5];
    macro_rules! put {
        ($($s:literal $c:literal $o:literal),* $(,)?) => {
            $( t[$s][$c][super::out_slot($o)] = Some(map_lanes_avx2::<$s, $c, $o> as super::MapLanesFn); )*
        };
    }
    put!(
        0 0 1, 1 0 1, 0 0 2, 1 0 2, 2 0 2, 3 0 2,
        0 0 4, 1 0 4, 2 0 4, 3 0 4, 4 0 4,
        0 1 4, 2 1 4, 4 1 4,
        1 2 4, 3 2 4, 4 2 4,
    );
    t
};

/// Byte-ranked dictionaries to 1-, 2- and 4-byte outputs.
pub(super) const DICT8_LANES_AVX2: [Option<super::MapDictLanesFn>; 3] = [
    Some(map_dict8_lanes_avx2::<1>),
    Some(map_dict8_lanes_avx2::<2>),
    Some(map_dict8_lanes_avx2::<4>),
];

/// Gathers the mapped columns and values of 8 positions at a time, one loop
/// per value source. Byte ranks are gathered as 4-byte words, so lanes
/// within 3 bytes of the end of the ranks are filled one by one.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn map_at_avx2(ids: &[u32], positions: &[u32], map: &[u32], v: At, cols: &mut [u32], vals: &mut [u32]) {
    let n = positions.len();
    let mut i = 0;
    // SAFETY: loads cover `positions[i..i + 8]` with `i + 8 <= n`; gathers
    // index `ids` by positions, `map` and the modes by ids, the dictionary
    // by ranks and raw values by position, all in range per the `MapAtFn`
    // contract; stores cover `cols[i..i + 8]` and `vals[i..i + 8]`.
    unsafe {
        macro_rules! run {
            (|$pos:ident, $id:ident| $value:expr) => {
                while i + 8 <= n {
                    let $pos = _mm256_loadu_si256(positions.as_ptr().add(i).cast());
                    let $id = _mm256_i32gather_epi32::<4>(ids.as_ptr().cast(), $pos);
                    let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), $id);
                    _mm256_storeu_si256(cols.as_mut_ptr().add(i).cast(), col);
                    let (value, safe): (__m256i, u32) = $value;
                    _mm256_storeu_si256(vals.as_mut_ptr().add(i).cast(), value);
                    let mut rest = !safe & 0xff;
                    while rest != 0 {
                        let k = i + rest.trailing_zeros() as usize;
                        let p = get(positions, k) as usize;
                        set(vals, k, at_value(&v, p, get(ids, p)));
                        rest &= rest - 1;
                    }
                    i += 8;
                }
            };
        }
        match v {
            At::Const(c) => run!(|_pos, _id| (_mm256_set1_epi32(c as i32), 0xff)),
            At::Dict32(r, d) => {
                let lookup = Lookup8::new(d);
                run!(|pos, _id| (lookup.get(_mm256_i32gather_epi32::<4>(r.as_ptr().cast(), pos), d), 0xff))
            }
            At::Raw(b, Src::W32, Conv::Bits) => run!(|pos, _id| (_mm256_i32gather_epi32::<4>(b.as_ptr().cast(), pos), 0xff)),
            At::Raw(b, Src::W32, Conv::U32ToF32) => run!(|pos, _id| (conv8(_mm256_i32gather_epi32::<4>(b.as_ptr().cast(), pos), Conv::U32ToF32), 0xff)),
            At::Raw(b, Src::W32, Conv::I32ToF32) => run!(|pos, _id| (conv8(_mm256_i32gather_epi32::<4>(b.as_ptr().cast(), pos), Conv::I32ToF32), 0xff)),
            // Narrow raw values (cached rows are not partial) take the
            // scalar loop below.
            At::Raw(..) => {}
            At::ModesOne(m) => run!(|_pos, id| (_mm256_i32gather_epi32::<4>(m.as_ptr().cast(), id), 0xff)),
            At::Modes32(r, d, m) => {
                let lookup = Lookup8::new(d);
                run!(|pos, id| (mode_values(id, _mm256_i32gather_epi32::<4>(r.as_ptr().cast(), pos), &lookup, d, m), 0xff))
            }
            At::Dict8(r, d) => {
                let lookup = Lookup8::new(d);
                run!(|pos, _id| {
                    let (rank, safe) = ranks_at(pos, r);
                    (lookup.get_masked(_mm256_setzero_si256(), lanes_of(safe), rank, d), safe)
                })
            }
            At::Modes8(r, d, m) => {
                let lookup = Lookup8::new(d);
                run!(|pos, id| {
                    let (rank, safe) = ranks_at(pos, r);
                    (mode_values(id, rank, &lookup, d, m), safe)
                })
            }
        }
        for k in i..n {
            let p = get(positions, k) as usize;
            let id = get(ids, p);
            set(cols, k, get(map, id as usize));
            set(vals, k, at_value(&v, p, id));
        }
    }
}

/// Scatters 8 elements at a time: columns and values computed with vectors,
/// each vector's mapped columns checked against the row length first, then
/// the kept lanes stored one by one.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn scatter_avx2(ids: &[u32], map: &[u32], v: At, out: &mut [u32]) -> bool {
    let n = ids.len();
    let mut i = 0;
    // SAFETY: loads cover `ids[i..i + 8]` with `i + 8 <= n`; gathers index
    // `map`, the modes and the values in range per the `ScatterFn`
    // contract; stores write only kept lanes whose column was checked to be
    // below `out.len()`.
    unsafe {
        let sign = _mm256_set1_epi32(i32::MIN);
        let width = _mm256_xor_si256(_mm256_set1_epi32(out.len().min(i32::MAX as usize) as i32), sign);
        let iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        macro_rules! run {
            (|$pos:ident, $id:ident, $at:ident| $value:expr) => {
                while i + 8 <= n {
                    let $at = i;
                    let $pos = _mm256_add_epi32(_mm256_set1_epi32(i as i32), iota);
                    let $id = _mm256_loadu_si256(ids.as_ptr().add(i).cast());
                    let col = _mm256_i32gather_epi32::<4>(map.as_ptr().cast(), $id);
                    let keep = keep8(col);
                    let inside = _mm256_cmpgt_epi32(width, _mm256_xor_si256(col, sign));
                    if keep & !(_mm256_movemask_ps(_mm256_castsi256_ps(inside)) as u32) != 0 {
                        return false;
                    }
                    let (value, safe): (__m256i, u32) = $value;
                    let (mut c, mut w) = ([0u32; 8], [0u32; 8]);
                    _mm256_storeu_si256(c.as_mut_ptr().cast(), col);
                    _mm256_storeu_si256(w.as_mut_ptr().cast(), value);
                    let mut rest = keep;
                    while rest != 0 {
                        let l = rest.trailing_zeros() as usize;
                        let x = if safe & (1 << l) != 0 { w[l] } else { at_value(&v, i + l, get(ids, i + l)) };
                        set(out, c[l] as usize, x);
                        rest &= rest - 1;
                    }
                    i += 8;
                }
            };
        }
        macro_rules! raw {
            ($b:ident, $src:expr, $conv:ident) => {
                match $conv {
                    Conv::Bits => run!(|_pos, _id, at| (load8($b.as_ptr(), at, $src), 0xff)),
                    Conv::U32ToF32 => run!(|_pos, _id, at| (conv8(load8($b.as_ptr(), at, $src), Conv::U32ToF32), 0xff)),
                    Conv::I32ToF32 => run!(|_pos, _id, at| (conv8(load8($b.as_ptr(), at, $src), Conv::I32ToF32), 0xff)),
                }
            };
        }
        match v {
            At::Const(c) => run!(|_pos, _id, _at| (_mm256_set1_epi32(c as i32), 0xff)),
            At::Raw(b, Src::U8, conv) => raw!(b, Src::U8, conv),
            At::Raw(b, Src::I8, conv) => raw!(b, Src::I8, conv),
            At::Raw(b, Src::U16, conv) => raw!(b, Src::U16, conv),
            At::Raw(b, Src::I16, conv) => raw!(b, Src::I16, conv),
            At::Raw(b, Src::W32, conv) => raw!(b, Src::W32, conv),
            At::Dict32(r, d) => {
                let lookup = Lookup8::new(d);
                run!(|_pos, _id, at| (lookup.get(_mm256_loadu_si256(r.as_ptr().add(at).cast()), d), 0xff))
            }
            At::Dict8(r, d) => {
                let lookup = Lookup8::new(d);
                run!(|_pos, _id, at| (lookup.get(ranks8(r, at), d), 0xff))
            }
            At::ModesOne(m) => run!(|_pos, id, _at| (_mm256_i32gather_epi32::<4>(m.as_ptr().cast(), id), 0xff)),
            At::Modes32(r, d, m) => {
                let lookup = Lookup8::new(d);
                run!(|_pos, id, at| (mode_values(id, _mm256_loadu_si256(r.as_ptr().add(at).cast()), &lookup, d, m), 0xff))
            }
            At::Modes8(r, d, m) => {
                let lookup = Lookup8::new(d);
                run!(|_pos, id, at| (mode_values(id, ranks8(r, at), &lookup, d, m), 0xff))
            }
        }
        for t in i..n {
            let id = get(ids, t);
            let col = get(map, id as usize);
            if col != crate::scans::UNMAPPED {
                if col as usize >= out.len() {
                    return false;
                }
                set(out, col as usize, at_value(&v, t, id));
            }
        }
    }
    true
}

/// Inclusive prefix sums of 8 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn prefix8(p: __m256i) -> __m256i {
    let p = _mm256_add_epi32(p, _mm256_slli_si256::<4>(p));
    let p = _mm256_add_epi32(p, _mm256_slli_si256::<8>(p));
    // The low half's total, carried into every lane of the high half.
    let total = _mm256_shuffle_epi32::<0xff>(p);
    _mm256_add_epi32(p, _mm256_permute2x128_si256::<0x08>(total, total))
}

/// Field unpacking 8 symbols at a time, as the AVX-512 kernel does with 16:
/// a group without escapes is its symbols; otherwise each lane's bit offset
/// is the exclusive prefix sum of the escape lengths, and each lane loads
/// the 8 bytes holding its field with one gather. A group too close to the
/// end of `bits` for 8-byte loads runs the scalar loop.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn fields_avx2(syms: &[u8], bits: &[u8], out: &mut [u32]) -> u64 {
    let n = syms.len().min(out.len());
    let mut bit = 0u64;
    let mut i = 0;
    // SAFETY: symbol loads and stores cover `syms[i..i + 8]` and
    // `out[i..i + 8]` with `i + 8 <= n`; a group with escapes reads 8 bytes
    // at offsets below `bit / 8 + 8 * 31 / 8 + 1`, which the guard keeps
    // 8 bytes inside `bits`.
    unsafe {
        let one = _mm256_set1_epi64x(1);
        let low_words = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
        while i + 8 <= n {
            let idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(syms.as_ptr().add(i).cast()));
            let escaped = _mm256_cmpgt_epi32(idx, _mm256_set1_epi32(DIRECT as i32 - 1));
            if _mm256_movemask_ps(_mm256_castsi256_ps(escaped)) == 0 {
                _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), idx);
                i += 8;
                continue;
            }
            if (bit / 8) as usize + 40 > bits.len() {
                bit = fields_tail(syms.get_unchecked(i..i + 8), bits, out.get_unchecked_mut(i..i + 8), bit);
                i += 8;
                continue;
            }
            // EXTRA and BASE by arithmetic (see `symbol`).
            let above = _mm256_cmpgt_epi32(idx, _mm256_set1_epi32(DIRECT as i32));
            let first = _mm256_cmpeq_epi32(idx, _mm256_set1_epi32(DIRECT as i32));
            let len = _mm256_sub_epi32(idx, _mm256_set1_epi32(DIRECT as i32 - 7));
            let e = _mm256_or_si256(
                _mm256_and_si256(first, _mm256_set1_epi32(FIRST_BITS as i32)),
                _mm256_and_si256(above, len),
            );
            let b = _mm256_blendv_epi8(idx, _mm256_sllv_epi32(_mm256_set1_epi32(1), len), above);
            let p = prefix8(e);
            let pos = _mm256_add_epi32(_mm256_sub_epi32(p, e), _mm256_set1_epi32((bit & 7) as i32));
            let byte = _mm256_srli_epi32::<3>(pos);
            let sh = _mm256_and_si256(pos, _mm256_set1_epi32(7));
            let base = bits.as_ptr().add((bit / 8) as usize);
            let field = |half_byte: __m128i, half_sh: __m128i, half_e: __m128i| {
                let w = _mm256_i32gather_epi64::<1>(base.cast(), half_byte);
                let mask = _mm256_sub_epi64(_mm256_sllv_epi64(one, _mm256_cvtepu32_epi64(half_e)), one);
                let f = _mm256_and_si256(_mm256_srlv_epi64(w, _mm256_cvtepu32_epi64(half_sh)), mask);
                _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(f, low_words))
            };
            let lo = field(_mm256_castsi256_si128(byte), _mm256_castsi256_si128(sh), _mm256_castsi256_si128(e));
            let hi = field(_mm256_extracti128_si256::<1>(byte), _mm256_extracti128_si256::<1>(sh), _mm256_extracti128_si256::<1>(e));
            let f = _mm256_set_m128i(hi, lo);
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), _mm256_or_si256(b, f));
            bit += _mm256_extract_epi32::<7>(p) as u32 as u64;
            i += 8;
        }
        fields_tail(syms.get_unchecked(i..n), bits, out.get_unchecked_mut(i..n), bit)
    }
}

/// Running sums 8 lanes at a time; the exact total of each group comes from
/// a 64-bit reduction so that wrapping in the 32-bit lanes cannot hide an
/// out-of-range value.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn ascend_avx2(v: &mut [u32]) -> u64 {
    let n = v.len();
    let mut c = u64::MAX;
    let mut i = 0;
    // SAFETY: loads and stores cover `v[i..i + 8]` with `i + 8 <= n`.
    unsafe {
        let one = _mm256_set1_epi32(1);
        while i + 8 <= n {
            let raw = _mm256_loadu_si256(v.as_ptr().add(i).cast());
            let p = prefix8(_mm256_add_epi32(raw, one));
            _mm256_storeu_si256(v.as_mut_ptr().add(i).cast(), _mm256_add_epi32(p, _mm256_set1_epi32(c as u32 as i32)));
            let wide = _mm256_add_epi64(
                _mm256_cvtepu32_epi64(_mm256_castsi256_si128(raw)),
                _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(raw)),
            );
            let pair = _mm256_add_epi64(wide, _mm256_permute4x64_epi64::<0b0100_1110>(wide));
            let total = _mm256_add_epi64(pair, _mm256_shuffle_epi32::<0b0100_1110>(pair));
            c = c.wrapping_add(_mm256_extract_epi64::<0>(total) as u64 + 8);
            i += 8;
        }
        ascend_tail(v.get_unchecked_mut(i..), c)
    }
}
