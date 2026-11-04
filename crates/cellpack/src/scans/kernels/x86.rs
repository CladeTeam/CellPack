//! AVX2 and AVX-512 kernels.
//!
//! rANS: one vector holds 8 (AVX2) or 16 (AVX-512) states. Per step it
//! gathers the slot entries, updates every state, and refills the states
//! that dropped below `L` from consecutive stream words: AVX2 moves the
//! words into place with a permutation looked up by the refill mask,
//! AVX-512 with a masked expand. Near the end of a stream the words come
//! from a zero-padded copy, so the vector loop runs to the last full group;
//! the scalar decoder finishes the remainder from the same states, and both
//! read identical bytes. Pair kernels step two
//! independent streams alternately, overlapping their gather latencies;
//! with 4 (AVX2) or 8 (AVX-512) lanes per stream both streams share one
//! vector, its halves gathering from the two slot tables through one base.

use std::arch::x86_64::*;

use super::map_avx2;
use super::{At, Conv, Isa, Kernels, LanesTable, SCALAR, Src, ascend_tail, at_value, conv_value, fields_tail, src_value};
use crate::scans::map::compact;
use crate::unchecked::{get, set};
use crate::scans::rans::{
    L, M, SCALE, Slots, decode_pair_scalar, decode_scalar, finished, run_pair, run_scalar, states,
};
use crate::scans::symbol::{DIRECT, FIRST_BITS};

/// Whether this CPU runs [`AVX2`]: AVX2, popcnt and FMA (the encoders use
/// `fma` for the exact reciprocal).
pub(super) fn avx2_supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt") && is_x86_feature_detected!("fma")
}

/// Whether this CPU runs [`AVX512`], including the AVX2 requirements.
pub(super) fn avx512_supported() -> bool {
    avx2_supported()
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
}

/// The AVX-512 set, with the byte-compress scan where VBMI2 exists.
pub(super) fn avx512_set() -> &'static Kernels {
    if is_x86_feature_detected!("avx512vbmi2") { &AVX512_VBMI2 } else { &AVX512 }
}

/// [`AVX512`] with the byte-compress versions of the two bitmap scans.
const AVX512_VBMI2: Kernels = Kernels { scan: scan_vbmi2, scan_bits: super::encode_x86::scan_bits_vbmi2, ..AVX512 };

/// The AVX2 set: 8-lane rANS, 2/4/8-byte expansion, and the AVX2 mapping,
/// field and running-sum kernels. Single streams below 8 lanes decode
/// scalar; paired streams use the four-lane pair kernel from 4 lanes.
pub(super) const AVX2: Kernels = Kernels {
    isa: Isa::Avx2,
    rans: [
        decode_scalar::<1>,
        decode_scalar::<2>,
        decode_scalar::<4>,
        rans_avx2::<8>,
        rans_avx2::<16>,
        rans_avx2::<32>,
    ],
    rans_pair: [
        decode_pair_scalar::<1>,
        decode_pair_scalar::<2>,
        pair4_avx2,
        pair_avx2::<8>,
        pair_avx2::<16>,
        pair_avx2::<32>,
    ],
    expand: [SCALAR.expand[0], expand2_avx2, expand4_avx2, expand8_avx2],
    expand32: [
        SCALAR.expand32[0],
        expand32_2_avx2,
        expand32_4_avx2,
        expand32_8_avx2,
    ],
    scan: scan_avx2,
    rans_encode: super::encode_x86::rans_encode_avx2,
    rans_many: super::encode_x86::rans_many_avx2,
    scan_bits: super::encode_x86::scan_bits_avx2,
    gaps: super::encode_x86::gaps_avx2,
    fields: map_avx2::fields_avx2,
    ascend: map_avx2::ascend_avx2,
    map_u8: map_avx2::map_u8_avx2,
    map_u32: map_avx2::map_u32_avx2,
    map_modes_u8: map_avx2::map_modes_u8_avx2,
    map_modes_u32: map_avx2::map_modes_u32_avx2,
    map_raw32: map_avx2::map_raw32_avx2,
    map_at: map_avx2::map_at_avx2,
    scatter: map_avx2::scatter_avx2,
    map_lanes: map_avx2::LANES_AVX2,
    map_dict8_lanes: map_avx2::DICT8_LANES_AVX2,
};

/// The AVX-512 set: the AVX2 rANS kernel for 8 lanes, 16-lane vectors for
/// 16 and 32 lanes, and mapping kernels of 16 elements per step (2- and
/// 8-byte expansion stay on the AVX2 kernels).
pub(super) const AVX512: Kernels = Kernels {
    isa: Isa::Avx512,
    rans: [
        decode_scalar::<1>,
        decode_scalar::<2>,
        decode_scalar::<4>,
        rans_avx2::<8>,
        rans_avx512::<16>,
        rans_avx512::<32>,
    ],
    rans_pair: [
        decode_pair_scalar::<1>,
        decode_pair_scalar::<2>,
        pair4_avx2,
        pair8_avx512,
        pair_avx512::<16>,
        pair_avx512::<32>,
    ],
    expand: [SCALAR.expand[0], expand2_avx2, expand4_avx512, expand8_avx2],
    expand32: [
        SCALAR.expand32[0],
        expand32_2_avx2,
        expand32_4_avx512,
        expand32_8_avx2,
    ],
    scan: scan_avx512,
    rans_encode: super::encode_x86::rans_encode_avx512,
    rans_many: super::encode_x86::rans_many_avx512,
    scan_bits: super::encode_x86::scan_bits_avx512,
    gaps: super::encode_x86::gaps_avx512,
    fields: fields_avx512,
    ascend: ascend_avx512,
    map_u8: map_u8_avx512,
    map_u32: map_u32_avx512,
    map_modes_u8: map_modes_u8_avx512,
    map_modes_u32: map_modes_u32_avx512,
    map_raw32: map_raw32_avx512,
    map_at: map_at_avx512,
    scatter: scatter_avx512,
    map_lanes: LANES_AVX512,
    map_dict8_lanes: [
        Some(map_dict8_lanes_avx512::<1>),
        Some(map_dict8_lanes_avx512::<2>),
        Some(map_dict8_lanes_avx512::<4>),
    ],
};

/// For each 8-bit refill mask, the source word of each lane: lane `i` takes
/// the word whose index is the number of refilling lanes below it.
static REFILL_PERM: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut i) = (0u32, 0);
        while i < 8 {
            if m & (1 << i) != 0 {
                t[m][i] = k;
                k += 1;
            }
            i += 1;
        }
        m += 1;
    }
    t
};

/// As [`REFILL_PERM`] for two independent 4-lane halves: the low half's
/// words occupy source lanes 0..4 and the high half's 4..8.
static REFILL_PERM_HALVES: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let mut i = 0;
        while i < 8 {
            let first = i & 4;
            let below = m & ((1 << i) - 1) & !((1 << first) - 1);
            t[m][i] = first as u32 + (below as u32).count_ones();
            i += 1;
        }
        m += 1;
    }
    t
};

/// Entries from slot table `a` to slot table `b`, when one gather base can
/// reach both (the tables of a chunk share one allocation).
fn table_offset(a: &Slots, b: &Slots) -> Option<i32> {
    let d = (b.as_ptr() as isize - a.as_ptr() as isize) / 4;
    i32::try_from(d).ok().filter(|d| d.unsigned_abs() < 1 << 28)
}

/// Readable word windows into one stream. A group of vector steps reads up
/// to `need` bytes at the current position; near the end of the stream the
/// window comes from a zero-padded copy of the remaining whole words, which
/// reads exactly like the scalar decoder past the end. Only a corrupt
/// stream can outrun the padding, and then the group falls back to scalar.
struct Window<'a> {
    stream: &'a [u8],
    pad: [u8; 2 * WINDOW_MAX],
    base: usize,
    copied: bool,
}

/// Largest `need`: 32 lanes of 16-bit words.
const WINDOW_MAX: usize = 64;

impl<'a> Window<'a> {
    fn new(stream: &'a [u8]) -> Self {
        Window {
            stream,
            pad: [0; 2 * WINDOW_MAX],
            base: 0,
            copied: false,
        }
    }

    /// Pointer to `need <= WINDOW_MAX` readable bytes holding the words at `pos`.
    #[inline(always)]
    fn at(&mut self, pos: usize, need: usize) -> Option<*const u8> {
        if pos + need <= self.stream.len() {
            // SAFETY: `pos + need <= len`.
            return Some(unsafe { self.stream.as_ptr().add(pos) });
        }
        if !self.copied {
            self.copied = true;
            self.base = pos.min(self.stream.len());
            // Whole words only: the scalar decoder reads a final odd byte as zero.
            let rem = (self.stream.len() - self.base) & !1;
            self.pad[..rem].copy_from_slice(&self.stream[self.base..self.base + rem]);
        }
        let off = pos - self.base;
        (off + need <= self.pad.len()).then(|| self.pad[off..].as_ptr())
    }
}

/// One step of 8 states: emits 8 symbols at `out` and refills from `stream + *pos`.
///
/// # Safety
/// `out` has 8 writable bytes, `stream + *pos` 16 readable bytes, and
/// `slots` points to an `M`-entry slot table.
#[target_feature(enable = "avx2,popcnt")]
#[inline]
unsafe fn step8(
    x: __m256i,
    slots: *const i32,
    stream: *const u8,
    pos: &mut usize,
    out: *mut u8,
) -> __m256i {
    // SAFETY: gathers index `slots` with `x & (M - 1)`; the loads and stores
    // stay inside the ranges the caller guarantees; the permutation table
    // is indexed by an 8-bit mask.
    unsafe {
        let mask12 = _mm256_set1_epi32(0xfff);
        let slot = _mm256_and_si256(x, _mm256_set1_epi32(M as i32 - 1));
        let e = _mm256_i32gather_epi32::<4>(slots, slot);
        let f = _mm256_add_epi32(_mm256_and_si256(e, mask12), _mm256_set1_epi32(1));
        let c = _mm256_and_si256(_mm256_srli_epi32::<12>(e), mask12);
        let sym = _mm256_srli_epi32::<24>(e);
        let y = _mm256_add_epi32(
            _mm256_mullo_epi32(f, _mm256_srli_epi32::<{ SCALE as i32 }>(x)),
            _mm256_sub_epi32(slot, c),
        );
        let p16 = _mm256_packus_epi32(sym, sym);
        let p8 = _mm256_packus_epi16(p16, p16);
        out.cast::<i32>().write_unaligned(_mm256_cvtsi256_si32(p8));
        out.add(4)
            .cast::<i32>()
            .write_unaligned(_mm256_extract_epi32::<4>(p8));
        let need = _mm256_cmpeq_epi32(_mm256_srli_epi32::<16>(y), _mm256_setzero_si256());
        let m = _mm256_movemask_ps(_mm256_castsi256_ps(need)) as u32;
        let words = _mm256_cvtepu16_epi32(_mm_loadu_si128(stream.add(*pos).cast()));
        let perm = _mm256_loadu_si256(REFILL_PERM[m as usize].as_ptr().cast());
        let w = _mm256_permutevar8x32_epi32(words, perm);
        *pos += 2 * m.count_ones() as usize;
        _mm256_blendv_epi8(y, _mm256_or_si256(_mm256_slli_epi32::<16>(y), w), need)
    }
}

/// One step of 16 states; as [`step8`] with 16 output bytes and 32 readable stream bytes.
#[target_feature(enable = "avx512f,avx512bw,avx2,popcnt")]
#[inline]
unsafe fn step16(
    x: __m512i,
    slots: *const i32,
    stream: *const u8,
    pos: &mut usize,
    out: *mut u8,
) -> __m512i {
    // SAFETY: as in `step8`.
    unsafe {
        let mask12 = _mm512_set1_epi32(0xfff);
        let slot = _mm512_and_si512(x, _mm512_set1_epi32(M as i32 - 1));
        let e = _mm512_i32gather_epi32::<4>(slot, slots.cast());
        let f = _mm512_add_epi32(_mm512_and_si512(e, mask12), _mm512_set1_epi32(1));
        let c = _mm512_and_si512(_mm512_srli_epi32::<12>(e), mask12);
        let sym = _mm512_srli_epi32::<24>(e);
        let y = _mm512_add_epi32(
            _mm512_mullo_epi32(f, _mm512_srli_epi32::<SCALE>(x)),
            _mm512_sub_epi32(slot, c),
        );
        _mm_storeu_si128(out.cast(), _mm512_cvtepi32_epi8(sym));
        let need = _mm512_cmplt_epu32_mask(y, _mm512_set1_epi32(L as i32));
        let words = _mm512_cvtepu16_epi32(_mm256_loadu_si256(stream.add(*pos).cast()));
        let w = _mm512_maskz_expand_epi32(need, words);
        *pos += 2 * need.count_ones() as usize;
        _mm512_mask_mov_epi32(y, need, _mm512_or_si512(_mm512_slli_epi32::<16>(y), w))
    }
}

/// `N` lanes as `N / 8` AVX2 vectors.
#[target_feature(enable = "avx2,popcnt")]
unsafe fn rans_avx2<const N: usize>(
    slots: &Slots,
    stream: &[u8],
    out: &mut [u8],
    ends: &mut [u32],
) -> bool {
    const { assert!(N.is_multiple_of(8) && N <= 32) };
    let mut lanes = states::<N>(stream);
    let (v, n) = (N / 8, out.len());
    let (mut pos, mut j) = (4 * N, 0);
    // SAFETY: each vector step reads at most 16 bytes, so a group stays
    // inside the `16 * v` bytes its window guarantees; symbol stores cover
    // `out[j..j + N]` with `j + N <= n`; state loads and stores stay inside
    // `lanes`.
    unsafe {
        let mut x = [_mm256_setzero_si256(); 4];
        for (i, xv) in x.iter_mut().enumerate().take(v) {
            *xv = _mm256_loadu_si256(lanes.as_ptr().add(8 * i).cast());
        }
        let base = slots.as_ptr().cast::<i32>();
        let mut win = Window::new(stream);
        while j + N <= n {
            let Some(src) = win.at(pos, 16 * v) else {
                break;
            };
            let mut used = 0;
            for (i, xv) in x.iter_mut().enumerate().take(v) {
                *xv = step8(*xv, base, src, &mut used, out.as_mut_ptr().add(j + 8 * i));
            }
            pos += used;
            j += N;
        }
        for (i, xv) in x.iter().enumerate().take(v) {
            _mm256_storeu_si256(lanes.as_mut_ptr().add(8 * i).cast(), *xv);
        }
    }
    run_scalar::<N>(slots, &mut lanes, stream, &mut pos, out, j);
    finished(&lanes, ends, pos, stream.len())
}

/// Two equal-length streams of `N` lanes each, as `N / 8` AVX2 vectors per
/// stream, stepped alternately.
#[target_feature(enable = "avx2,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair_avx2<const N: usize>(
    sa: &Slots,
    a: &[u8],
    oa: &mut [u8],
    sb: &Slots,
    b: &[u8],
    ob: &mut [u8],
    ea: &mut [u32],
    eb: &mut [u32],
) -> bool {
    const { assert!(N.is_multiple_of(8) && N <= 32) };
    let (mut la, mut lb) = (states::<N>(a), states::<N>(b));
    debug_assert_eq!(oa.len(), ob.len());
    let (v, n) = (N / 8, oa.len());
    let (mut pa, mut pb, mut j) = (4 * N, 4 * N, 0);
    // SAFETY: as in `rans_avx2`, for both streams.
    unsafe {
        let (mut xa, mut xb) = ([_mm256_setzero_si256(); 4], [_mm256_setzero_si256(); 4]);
        for i in 0..v {
            xa[i] = _mm256_loadu_si256(la.as_ptr().add(8 * i).cast());
            xb[i] = _mm256_loadu_si256(lb.as_ptr().add(8 * i).cast());
        }
        let (ba, bb) = (sa.as_ptr().cast::<i32>(), sb.as_ptr().cast::<i32>());
        let (mut wa, mut wb) = (Window::new(a), Window::new(b));
        while j + N <= n {
            let (Some(ra), Some(rb)) = (wa.at(pa, 16 * v), wb.at(pb, 16 * v)) else {
                break;
            };
            let (mut ua, mut ub) = (0, 0);
            for i in 0..v {
                xa[i] = step8(xa[i], ba, ra, &mut ua, oa.as_mut_ptr().add(j + 8 * i));
                xb[i] = step8(xb[i], bb, rb, &mut ub, ob.as_mut_ptr().add(j + 8 * i));
            }
            pa += ua;
            pb += ub;
            j += N;
        }
        for i in 0..v {
            _mm256_storeu_si256(la.as_mut_ptr().add(8 * i).cast(), xa[i]);
            _mm256_storeu_si256(lb.as_mut_ptr().add(8 * i).cast(), xb[i]);
        }
    }
    run_pair::<N>(sa, &mut la, a, &mut pa, oa, sb, &mut lb, b, &mut pb, ob, j);
    finished(&la, ea, pa, a.len()) & finished(&lb, eb, pb, b.len())
}

/// Two equal-length streams of 4 lanes in one AVX2 vector: stream `a` in
/// the low half, `b` in the high half.
#[target_feature(enable = "avx2,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair4_avx2(
    sa: &Slots,
    a: &[u8],
    oa: &mut [u8],
    sb: &Slots,
    b: &[u8],
    ob: &mut [u8],
    ea: &mut [u32],
    eb: &mut [u32],
) -> bool {
    let Some(off) = table_offset(sa, sb) else {
        return decode_pair_scalar::<4>(sa, a, oa, sb, b, ob, ea, eb);
    };
    let (mut la, mut lb) = (states::<4>(a), states::<4>(b));
    debug_assert_eq!(oa.len(), ob.len());
    let n = oa.len();
    let (mut pa, mut pb, mut j) = (16, 16, 0);
    // SAFETY: gathers index the allocation holding both tables with
    // `slot` (low half) or `slot + off` (high half); each step reads 8
    // bytes of each window and stores 4 symbols at `oa[j..]` and `ob[j..]`
    // with `j + 4 <= n`; state loads and stores stay inside `la` and `lb`.
    unsafe {
        let mut x = _mm256_set_m128i(
            _mm_loadu_si128(lb.as_ptr().cast()),
            _mm_loadu_si128(la.as_ptr().cast()),
        );
        let base = sa.as_ptr().cast::<i32>();
        let shift = _mm256_setr_epi32(0, 0, 0, 0, off, off, off, off);
        let mask12 = _mm256_set1_epi32(0xfff);
        let (mut wa, mut wb) = (Window::new(a), Window::new(b));
        while j + 4 <= n {
            let (Some(ra), Some(rb)) = (wa.at(pa, 8), wb.at(pb, 8)) else {
                break;
            };
            let slot = _mm256_and_si256(x, _mm256_set1_epi32(M as i32 - 1));
            let e = _mm256_i32gather_epi32::<4>(base, _mm256_add_epi32(slot, shift));
            let f = _mm256_add_epi32(_mm256_and_si256(e, mask12), _mm256_set1_epi32(1));
            let c = _mm256_and_si256(_mm256_srli_epi32::<12>(e), mask12);
            let sym = _mm256_srli_epi32::<24>(e);
            let y = _mm256_add_epi32(
                _mm256_mullo_epi32(f, _mm256_srli_epi32::<{ SCALE as i32 }>(x)),
                _mm256_sub_epi32(slot, c),
            );
            let p16 = _mm256_packus_epi32(sym, sym);
            let p8 = _mm256_packus_epi16(p16, p16);
            oa.as_mut_ptr()
                .add(j)
                .cast::<i32>()
                .write_unaligned(_mm256_cvtsi256_si32(p8));
            ob.as_mut_ptr()
                .add(j)
                .cast::<i32>()
                .write_unaligned(_mm256_extract_epi32::<4>(p8));
            let need = _mm256_cmpeq_epi32(_mm256_srli_epi32::<16>(y), _mm256_setzero_si256());
            let m = _mm256_movemask_ps(_mm256_castsi256_ps(need)) as u32;
            let packed = _mm_set_epi64x(
                rb.cast::<i64>().read_unaligned(),
                ra.cast::<i64>().read_unaligned(),
            );
            let perm = _mm256_loadu_si256(REFILL_PERM_HALVES[m as usize].as_ptr().cast());
            let w = _mm256_permutevar8x32_epi32(_mm256_cvtepu16_epi32(packed), perm);
            pa += 2 * (m & 0xf).count_ones() as usize;
            pb += 2 * (m >> 4).count_ones() as usize;
            x = _mm256_blendv_epi8(y, _mm256_or_si256(_mm256_slli_epi32::<16>(y), w), need);
            j += 4;
        }
        _mm_storeu_si128(la.as_mut_ptr().cast(), _mm256_castsi256_si128(x));
        _mm_storeu_si128(lb.as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(x));
    }
    run_pair::<4>(sa, &mut la, a, &mut pa, oa, sb, &mut lb, b, &mut pb, ob, j);
    finished(&la, ea, pa, a.len()) & finished(&lb, eb, pb, b.len())
}

/// Two equal-length streams of 8 lanes in one AVX-512 vector, as
/// [`pair4_avx2`].
#[target_feature(enable = "avx512f,avx512bw,avx2,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair8_avx512(
    sa: &Slots,
    a: &[u8],
    oa: &mut [u8],
    sb: &Slots,
    b: &[u8],
    ob: &mut [u8],
    ea: &mut [u32],
    eb: &mut [u32],
) -> bool {
    let Some(off) = table_offset(sa, sb) else {
        // SAFETY: this CPU runs AVX-512, hence AVX2.
        return unsafe { pair_avx2::<8>(sa, a, oa, sb, b, ob, ea, eb) };
    };
    let (mut la, mut lb) = (states::<8>(a), states::<8>(b));
    debug_assert_eq!(oa.len(), ob.len());
    let n = oa.len();
    let (mut pa, mut pb, mut j) = (32, 32, 0);
    // SAFETY: as in `pair4_avx2`, with 16 bytes per window and 8 symbols
    // per stream and step.
    unsafe {
        let mut x = _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm256_loadu_si256(la.as_ptr().cast())),
            _mm256_loadu_si256(lb.as_ptr().cast()),
        );
        let base = sa.as_ptr().cast::<i32>();
        let shift = _mm512_maskz_set1_epi32(0xff00, off);
        let mask12 = _mm512_set1_epi32(0xfff);
        let (mut wa, mut wb) = (Window::new(a), Window::new(b));
        while j + 8 <= n {
            let (Some(ra), Some(rb)) = (wa.at(pa, 16), wb.at(pb, 16)) else {
                break;
            };
            let slot = _mm512_and_si512(x, _mm512_set1_epi32(M as i32 - 1));
            let e = _mm512_i32gather_epi32::<4>(_mm512_add_epi32(slot, shift), base.cast());
            let f = _mm512_add_epi32(_mm512_and_si512(e, mask12), _mm512_set1_epi32(1));
            let c = _mm512_and_si512(_mm512_srli_epi32::<12>(e), mask12);
            let sym = _mm512_cvtepi32_epi8(_mm512_srli_epi32::<24>(e));
            let y = _mm512_add_epi32(
                _mm512_mullo_epi32(f, _mm512_srli_epi32::<SCALE>(x)),
                _mm512_sub_epi32(slot, c),
            );
            _mm_storel_epi64(oa.as_mut_ptr().add(j).cast(), sym);
            _mm_storel_epi64(ob.as_mut_ptr().add(j).cast(), _mm_unpackhi_epi64(sym, sym));
            let need = _mm512_cmplt_epu32_mask(y, _mm512_set1_epi32(L as i32));
            let words_a = _mm512_cvtepu16_epi32(_mm256_castsi128_si256(_mm_loadu_si128(ra.cast())));
            let words_b = _mm512_cvtepu16_epi32(_mm256_castsi128_si256(_mm_loadu_si128(rb.cast())));
            let w = _mm512_or_si512(
                _mm512_maskz_expand_epi32(need & 0xff, words_a),
                _mm512_maskz_expand_epi32(need & 0xff00, words_b),
            );
            pa += 2 * (need & 0xff).count_ones() as usize;
            pb += 2 * (need >> 8).count_ones() as usize;
            x = _mm512_mask_mov_epi32(y, need, _mm512_or_si512(_mm512_slli_epi32::<16>(y), w));
            j += 8;
        }
        _mm256_storeu_si256(la.as_mut_ptr().cast(), _mm512_castsi512_si256(x));
        _mm256_storeu_si256(lb.as_mut_ptr().cast(), _mm512_extracti64x4_epi64::<1>(x));
    }
    run_pair::<8>(sa, &mut la, a, &mut pa, oa, sb, &mut lb, b, &mut pb, ob, j);
    finished(&la, ea, pa, a.len()) & finished(&lb, eb, pb, b.len())
}

/// `N` lanes as `N / 16` AVX-512 vectors.
#[target_feature(enable = "avx512f,avx512bw,avx2,popcnt")]
unsafe fn rans_avx512<const N: usize>(
    slots: &Slots,
    stream: &[u8],
    out: &mut [u8],
    ends: &mut [u32],
) -> bool {
    const { assert!(N.is_multiple_of(16) && N <= 32) };
    let mut lanes = states::<N>(stream);
    let (v, n) = (N / 16, out.len());
    let (mut pos, mut j) = (4 * N, 0);
    // SAFETY: as in `rans_avx2`, with at most 32 bytes per vector step and
    // `32 * v`-byte windows.
    unsafe {
        let mut x = [_mm512_setzero_si512(); 2];
        for (i, xv) in x.iter_mut().enumerate().take(v) {
            *xv = _mm512_loadu_si512(lanes.as_ptr().add(16 * i).cast());
        }
        let base = slots.as_ptr().cast::<i32>();
        let mut win = Window::new(stream);
        while j + N <= n {
            let Some(src) = win.at(pos, 32 * v) else {
                break;
            };
            let mut used = 0;
            for (i, xv) in x.iter_mut().enumerate().take(v) {
                *xv = step16(*xv, base, src, &mut used, out.as_mut_ptr().add(j + 16 * i));
            }
            pos += used;
            j += N;
        }
        for (i, xv) in x.iter().enumerate().take(v) {
            _mm512_storeu_si512(lanes.as_mut_ptr().add(16 * i).cast(), *xv);
        }
    }
    run_scalar::<N>(slots, &mut lanes, stream, &mut pos, out, j);
    finished(&lanes, ends, pos, stream.len())
}

/// Two equal-length streams with `N / 16` AVX-512 vectors each.
#[target_feature(enable = "avx512f,avx512bw,avx2,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair_avx512<const N: usize>(
    sa: &Slots,
    a: &[u8],
    oa: &mut [u8],
    sb: &Slots,
    b: &[u8],
    ob: &mut [u8],
    ea: &mut [u32],
    eb: &mut [u32],
) -> bool {
    const { assert!(N.is_multiple_of(16) && N <= 32) };
    let (mut la, mut lb) = (states::<N>(a), states::<N>(b));
    debug_assert_eq!(oa.len(), ob.len());
    let (v, n) = (N / 16, oa.len());
    let (mut pa, mut pb, mut j) = (4 * N, 4 * N, 0);
    // SAFETY: as in `rans_avx512`, for both streams.
    unsafe {
        let (mut xa, mut xb) = ([_mm512_setzero_si512(); 2], [_mm512_setzero_si512(); 2]);
        for i in 0..v {
            xa[i] = _mm512_loadu_si512(la.as_ptr().add(16 * i).cast());
            xb[i] = _mm512_loadu_si512(lb.as_ptr().add(16 * i).cast());
        }
        let (ba, bb) = (sa.as_ptr().cast::<i32>(), sb.as_ptr().cast::<i32>());
        let (mut wa, mut wb) = (Window::new(a), Window::new(b));
        while j + N <= n {
            let (Some(ra), Some(rb)) = (wa.at(pa, 32 * v), wb.at(pb, 32 * v)) else {
                break;
            };
            let (mut ua, mut ub) = (0, 0);
            for i in 0..v {
                xa[i] = step16(xa[i], ba, ra, &mut ua, oa.as_mut_ptr().add(j + 16 * i));
                xb[i] = step16(xb[i], bb, rb, &mut ub, ob.as_mut_ptr().add(j + 16 * i));
            }
            pa += ua;
            pb += ub;
            j += N;
        }
        for i in 0..v {
            _mm512_storeu_si512(la.as_mut_ptr().add(16 * i).cast(), xa[i]);
            _mm512_storeu_si512(lb.as_mut_ptr().add(16 * i).cast(), xb[i]);
        }
    }
    run_pair::<N>(sa, &mut la, a, &mut pa, oa, sb, &mut lb, b, &mut pb, ob, j);
    finished(&la, ea, pa, a.len()) & finished(&lb, eb, pb, b.len())
}

/// For each byte, the positions of its set bits packed to the front.
pub(super) static BIT_POSITIONS: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut i) = (0, 0);
        while i < 8 {
            if m & (1 << i) != 0 {
                t[m][k] = i as u32;
                k += 1;
            }
            i += 1;
        }
        m += 1;
    }
    t
};

/// Marks to positions via a 64-bit mask per block, one byte of the mask per step: positions come
/// from a table, so the loop has a fixed trip count and no data branches.
#[target_feature(enable = "avx2,popcnt")]
unsafe fn scan_avx2(marks: &[u8], out: &mut [u32]) -> usize {
    let mut j = 0usize;
    // SAFETY: mark loads read `marks[64 * w..64 * w + 64]`; each step stores
    // 8 entries at `out[j]`, where `j` never exceeds the final count, and
    // `out` holds the count plus `SCAN_SLACK` (>= 16) entries.
    unsafe {
        let o = out.as_mut_ptr();
        let zero = _mm256_setzero_si256();
        for w in 0..marks.len() / 64 {
            let p = marks.as_ptr().add(64 * w);
            let lo =
                !_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(p.cast()), zero)) as u32;
            let hi = !_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                _mm256_loadu_si256(p.add(32).cast()),
                zero,
            )) as u32;
            let word = lo as u64 | (hi as u64) << 32;
            for byte in 0..8 {
                let m = (word >> (8 * byte)) as u8 as usize;
                let base = _mm256_set1_epi32((w * 64 + byte * 8) as i32);
                let pos = _mm256_loadu_si256(BIT_POSITIONS[m].as_ptr().cast());
                _mm256_storeu_si256(o.add(j).cast(), _mm256_add_epi32(base, pos));
                j += m.count_ones() as usize;
            }
        }
    }
    j
}

/// Marks to positions via a 64-bit mask per block, 16 bits per step with a masked compress.
#[target_feature(enable = "avx512f,avx512bw,avx2,popcnt")]
unsafe fn scan_avx512(marks: &[u8], out: &mut [u32]) -> usize {
    let mut j = 0usize;
    // SAFETY: mark loads read `marks[64 * w..64 * w + 64]`; each step stores
    // 16 entries at `out[j]`, where `j` never exceeds the final count, and
    // `out` holds the count plus `SCAN_SLACK` (>= 16) entries.
    unsafe {
        let o = out.as_mut_ptr();
        let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        for w in 0..marks.len() / 64 {
            let v = _mm512_loadu_si512(marks.as_ptr().add(64 * w).cast());
            let word = _mm512_test_epi8_mask(v, v);
            for q in 0..4 {
                let m = (word >> (16 * q)) as u16;
                let pos = _mm512_add_epi32(iota, _mm512_set1_epi32((w * 64 + q * 16) as i32));
                _mm512_storeu_si512(o.add(j).cast(), _mm512_maskz_compress_epi32(m, pos));
                j += m.count_ones() as usize;
            }
        }
    }
    j
}

/// Marks to positions, one 64-byte block per step: a byte compress packs the
/// block's marked positions, of which the first 32 are always widened and
/// stored; denser words take a second, rarely used path.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,avx2,popcnt")]
unsafe fn scan_vbmi2(marks: &[u8], out: &mut [u32]) -> usize {
    let mut j = 0usize;
    // SAFETY: mark loads read `marks[64 * w..64 * w + 64]`; a block with `c`
    // marks stores `max(32, 16 * ceil(c / 16))` entries at `out[j]`; with
    // `j + c` never above the final count, every store ends within the
    // count plus 32 entries, which `out` holds.
    unsafe {
        let o = out.as_mut_ptr();
        let iota = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );
        for w in 0..marks.len() / 64 {
            let v = _mm512_loadu_si512(marks.as_ptr().add(64 * w).cast());
            let word = _mm512_test_epi8_mask(v, v);
            let packed = _mm512_maskz_compress_epi8(word, iota);
            let base = _mm512_set1_epi32((w * 64) as i32);
            let widen = |q: __m128i| _mm512_add_epi32(base, _mm512_cvtepu8_epi32(q));
            _mm512_storeu_si512(o.add(j).cast(), widen(_mm512_castsi512_si128(packed)));
            _mm512_storeu_si512(
                o.add(j + 16).cast(),
                widen(_mm512_extracti32x4_epi32::<1>(packed)),
            );
            let c = word.count_ones() as usize;
            if c > 32 {
                _mm512_storeu_si512(
                    o.add(j + 32).cast(),
                    widen(_mm512_extracti32x4_epi32::<2>(packed)),
                );
                _mm512_storeu_si512(
                    o.add(j + 48).cast(),
                    widen(_mm512_extracti32x4_epi32::<3>(packed)),
                );
            }
            j += c;
        }
    }
    j
}

/// # Safety
/// As [`super::expand_scalar`].
#[inline]
unsafe fn expand_tail<const W: usize>(dict: &[u8], syms: &[u8], out: &mut [u8]) {
    // SAFETY: as documented.
    unsafe { super::expand_scalar::<W>(dict, syms, out) }
}

/// [`super::ExpandFn`] for 4-byte entries, 8 symbols per gather.
#[target_feature(enable = "avx2")]
unsafe fn expand4_avx2(dict: &[u8], syms: &[u8], out: &mut [u8]) {
    let n = syms.len();
    let mut i = 0;
    // SAFETY: every symbol indexes a dictionary entry (caller contract), so
    // each 4-byte gather stays inside `dict`; loads and stores cover
    // `syms[i..i + 8]` and `out[4 * i..4 * i + 32]` with `i + 8 <= n`.
    unsafe {
        while i + 8 <= n {
            let idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(syms.as_ptr().add(i).cast()));
            let v = _mm256_i32gather_epi32::<4>(dict.as_ptr().cast(), idx);
            _mm256_storeu_si256(out.as_mut_ptr().add(4 * i).cast(), v);
            i += 8;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand_tail::<4>(dict, syms.get_unchecked(i..), out.get_unchecked_mut(4 * i..)) };
}

/// [`super::ExpandFn`] for 2-byte entries, 8 symbols per step: each gather
/// loads 4 bytes and keeps the low 2.
#[target_feature(enable = "avx2")]
unsafe fn expand2_avx2(dict: &[u8], syms: &[u8], out: &mut [u8]) {
    let n = syms.len();
    let mut i = 0;
    // SAFETY: each gather loads 4 bytes at `2 * s`, at most 2 bytes past the
    // last entry, within the dictionary slack; loads and stores cover
    // `syms[i..i + 8]` and `out[2 * i..2 * i + 16]` with `i + 8 <= n`.
    unsafe {
        let low16 = _mm256_set1_epi32(0xffff);
        while i + 8 <= n {
            let idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(syms.as_ptr().add(i).cast()));
            let v = _mm256_and_si256(
                _mm256_i32gather_epi32::<2>(dict.as_ptr().cast(), idx),
                low16,
            );
            let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi32(v, v));
            _mm_storeu_si128(
                out.as_mut_ptr().add(2 * i).cast(),
                _mm256_castsi256_si128(packed),
            );
            i += 8;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand_tail::<2>(dict, syms.get_unchecked(i..), out.get_unchecked_mut(2 * i..)) };
}

/// [`super::ExpandFn`] for 8-byte entries, 4 symbols per gather.
#[target_feature(enable = "avx2")]
unsafe fn expand8_avx2(dict: &[u8], syms: &[u8], out: &mut [u8]) {
    let n = syms.len();
    let mut i = 0;
    // SAFETY: each 8-byte gather reads one dictionary entry; loads and
    // stores cover `syms[i..i + 4]` and `out[8 * i..8 * i + 32]` with `i + 4 <= n`.
    unsafe {
        while i + 4 <= n {
            let idx = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
                syms.as_ptr().add(i).cast::<i32>().read_unaligned(),
            ));
            let v = _mm256_i32gather_epi64::<8>(dict.as_ptr().cast(), idx);
            _mm256_storeu_si256(out.as_mut_ptr().add(8 * i).cast(), v);
            i += 4;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand_tail::<8>(dict, syms.get_unchecked(i..), out.get_unchecked_mut(8 * i..)) };
}

/// [`super::ExpandFn`] for 4-byte entries, 16 symbols per gather.
#[target_feature(enable = "avx512f,avx2")]
unsafe fn expand4_avx512(dict: &[u8], syms: &[u8], out: &mut [u8]) {
    let n = syms.len();
    let mut i = 0;
    // SAFETY: as in `expand4_avx2`, 16 entries per step with `i + 16 <= n`.
    unsafe {
        while i + 16 <= n {
            let idx = _mm512_cvtepu8_epi32(_mm_loadu_si128(syms.as_ptr().add(i).cast()));
            let v = _mm512_i32gather_epi32::<4>(idx, dict.as_ptr().cast());
            _mm512_storeu_si512(out.as_mut_ptr().add(4 * i).cast(), v);
            i += 16;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand_tail::<4>(dict, syms.get_unchecked(i..), out.get_unchecked_mut(4 * i..)) };
}

/// # Safety
/// As [`super::expand32_scalar`].
#[inline]
unsafe fn expand32_tail<const W: usize>(dict: &[u8], idx: &[u32], out: &mut [u8]) {
    // SAFETY: as documented.
    unsafe { super::expand32_scalar::<W>(dict, idx, out) }
}

/// [`super::Expand32Fn`] for 4-byte entries, 8 indices per gather.
#[target_feature(enable = "avx2")]
unsafe fn expand32_4_avx2(dict: &[u8], idx: &[u32], out: &mut [u8]) {
    let n = idx.len();
    let mut i = 0;
    // SAFETY: every index names a dictionary entry (caller contract); loads
    // and stores cover `idx[i..i + 8]` and `out[4 * i..4 * i + 32]` with `i + 8 <= n`.
    unsafe {
        while i + 8 <= n {
            let v = _mm256_i32gather_epi32::<4>(
                dict.as_ptr().cast(),
                _mm256_loadu_si256(idx.as_ptr().add(i).cast()),
            );
            _mm256_storeu_si256(out.as_mut_ptr().add(4 * i).cast(), v);
            i += 8;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand32_tail::<4>(dict, idx.get_unchecked(i..), out.get_unchecked_mut(4 * i..)) };
}

/// [`super::Expand32Fn`] for 2-byte entries, 8 indices per step, as
/// [`expand2_avx2`] with the indices loaded from `idx`.
#[target_feature(enable = "avx2")]
unsafe fn expand32_2_avx2(dict: &[u8], idx: &[u32], out: &mut [u8]) {
    let n = idx.len();
    let mut i = 0;
    // SAFETY: as in `expand2_avx2`, with indices loaded from `idx[i..i + 8]`.
    unsafe {
        let low16 = _mm256_set1_epi32(0xffff);
        while i + 8 <= n {
            let at = _mm256_loadu_si256(idx.as_ptr().add(i).cast());
            let v = _mm256_and_si256(_mm256_i32gather_epi32::<2>(dict.as_ptr().cast(), at), low16);
            let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi32(v, v));
            _mm_storeu_si128(
                out.as_mut_ptr().add(2 * i).cast(),
                _mm256_castsi256_si128(packed),
            );
            i += 8;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand32_tail::<2>(dict, idx.get_unchecked(i..), out.get_unchecked_mut(2 * i..)) };
}

/// [`super::Expand32Fn`] for 8-byte entries, 4 indices per gather.
#[target_feature(enable = "avx2")]
unsafe fn expand32_8_avx2(dict: &[u8], idx: &[u32], out: &mut [u8]) {
    let n = idx.len();
    let mut i = 0;
    // SAFETY: as in `expand8_avx2`, with indices loaded from `idx[i..i + 4]`.
    unsafe {
        while i + 4 <= n {
            let v = _mm256_i32gather_epi64::<8>(
                dict.as_ptr().cast(),
                _mm_loadu_si128(idx.as_ptr().add(i).cast()),
            );
            _mm256_storeu_si256(out.as_mut_ptr().add(8 * i).cast(), v);
            i += 4;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand32_tail::<8>(dict, idx.get_unchecked(i..), out.get_unchecked_mut(8 * i..)) };
}

/// [`super::Expand32Fn`] for 4-byte entries, 16 indices per gather.
#[target_feature(enable = "avx512f,avx2")]
unsafe fn expand32_4_avx512(dict: &[u8], idx: &[u32], out: &mut [u8]) {
    let n = idx.len();
    let mut i = 0;
    // SAFETY: as in `expand32_4_avx2`, 16 entries per step with `i + 16 <= n`.
    unsafe {
        while i + 16 <= n {
            let v = _mm512_i32gather_epi32::<4>(
                _mm512_loadu_si512(idx.as_ptr().add(i).cast()),
                dict.as_ptr().cast(),
            );
            _mm512_storeu_si512(out.as_mut_ptr().add(4 * i).cast(), v);
            i += 16;
        }
    }
    // SAFETY: the tail covers the remaining elements, which `out` holds.
    unsafe { expand32_tail::<4>(dict, idx.get_unchecked(i..), out.get_unchecked_mut(4 * i..)) };
}

/// Inclusive prefix sums of 16 lanes.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn prefix16(p: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let p = _mm512_add_epi32(p, _mm512_alignr_epi32::<15>(p, zero));
    let p = _mm512_add_epi32(p, _mm512_alignr_epi32::<14>(p, zero));
    let p = _mm512_add_epi32(p, _mm512_alignr_epi32::<12>(p, zero));
    _mm512_add_epi32(p, _mm512_alignr_epi32::<8>(p, zero))
}

/// Field unpacking 16 symbols at a time. A group without escapes is its
/// symbols. Otherwise each lane's bit offset is the exclusive prefix sum of
/// the escape lengths, so every lane loads the 8 bytes holding its field
/// with one gather and extracts it with variable shifts: no lane waits for
/// another. A group too close to the end of `bits` for 8-byte loads runs
/// the scalar loop.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
unsafe fn fields_avx512(syms: &[u8], bits: &[u8], out: &mut [u32]) -> u64 {
    let n = syms.len().min(out.len());
    let mut bit = 0u64;
    let mut i = 0;
    // SAFETY: symbol loads and stores cover `syms[i..i + 16]` and
    // `out[i..i + 16]` with `i + 16 <= n`; a group with escapes reads 8 bytes at offsets below
    // `bit / 8 + 16 * 31 / 8 + 1`, which the guard keeps 8 bytes inside `bits`.
    unsafe {
        let direct = _mm512_set1_epi32(DIRECT as i32);
        let one = _mm512_set1_epi64(1);
        while i + 16 <= n {
            let s8 = _mm_loadu_si128(syms.as_ptr().add(i).cast());
            let idx = _mm512_cvtepu8_epi32(s8);
            if _mm512_cmpge_epu32_mask(idx, direct) == 0 {
                _mm512_storeu_si512(out.as_mut_ptr().add(i).cast(), idx);
                i += 16;
                continue;
            }
            if (bit / 8) as usize + 72 > bits.len() {
                // Too close to the end of the escape bits for 8-byte loads.
                bit = fields_tail(syms.get_unchecked(i..i + 16), bits, out.get_unchecked_mut(i..i + 16), bit);
                i += 16;
                continue;
            }
            // EXTRA and BASE by arithmetic (see `symbol`): the first bucket
            // has `FIRST_BITS` escape bits; above it, symbol `s` has
            // `s - (DIRECT - 7)` and base `1 << (s - (DIRECT - 7))`.
            let above = _mm512_cmpgt_epu32_mask(idx, _mm512_set1_epi32(DIRECT as i32));
            let first = _mm512_cmpeq_epu32_mask(idx, _mm512_set1_epi32(DIRECT as i32));
            let len = _mm512_sub_epi32(idx, _mm512_set1_epi32(DIRECT as i32 - 7));
            let e = _mm512_mask_mov_epi32(
                _mm512_maskz_mov_epi32(first, _mm512_set1_epi32(FIRST_BITS as i32)),
                above,
                len,
            );
            let b = _mm512_mask_mov_epi32(idx, above, _mm512_sllv_epi32(_mm512_set1_epi32(1), len));
            let p = prefix16(e);
            let pos = _mm512_add_epi32(_mm512_sub_epi32(p, e), _mm512_set1_epi32((bit & 7) as i32));
            let byte = _mm512_srli_epi32::<3>(pos);
            let sh = _mm512_and_si512(pos, _mm512_set1_epi32(7));
            let base = bits.as_ptr().add((bit / 8) as usize);
            let field = |half_byte: __m256i, half_sh: __m256i, half_e: __m256i| {
                let w = _mm512_i32gather_epi64::<1>(half_byte, base.cast());
                let mask =
                    _mm512_sub_epi64(_mm512_sllv_epi64(one, _mm512_cvtepu32_epi64(half_e)), one);
                _mm512_cvtepi64_epi32(_mm512_and_si512(
                    _mm512_srlv_epi64(w, _mm512_cvtepu32_epi64(half_sh)),
                    mask,
                ))
            };
            let lo = field(
                _mm512_castsi512_si256(byte),
                _mm512_castsi512_si256(sh),
                _mm512_castsi512_si256(e),
            );
            let hi = field(
                _mm512_extracti64x4_epi64::<1>(byte),
                _mm512_extracti64x4_epi64::<1>(sh),
                _mm512_extracti64x4_epi64::<1>(e),
            );
            let f = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
            _mm512_storeu_si512(out.as_mut_ptr().add(i).cast(), _mm512_or_si512(b, f));
            bit += _mm_extract_epi32::<3>(_mm512_extracti32x4_epi32::<3>(p)) as u32 as u64;
            i += 16;
        }
    }
    // SAFETY: `i <= n`, and both slices hold `n` elements.
    unsafe { fields_tail(syms.get_unchecked(i..n), bits, out.get_unchecked_mut(i..n), bit) }
}

/// Running sums 16 lanes at a time; the exact total of each group comes
/// from a 64-bit reduction so that wrapping in the 32-bit lanes cannot hide
/// an out-of-range value.
#[target_feature(enable = "avx512f,avx2")]
unsafe fn ascend_avx512(v: &mut [u32]) -> u64 {
    let n = v.len();
    let mut c = u64::MAX;
    let mut i = 0;
    // SAFETY: loads and stores cover `v[i..i + 16]` with `i + 16 <= n`.
    unsafe {
        let one = _mm512_set1_epi32(1);
        while i + 16 <= n {
            let raw = _mm512_loadu_si512(v.as_ptr().add(i).cast());
            let g = _mm512_add_epi32(raw, one);
            let p = prefix16(g);
            _mm512_storeu_si512(
                v.as_mut_ptr().add(i).cast(),
                _mm512_add_epi32(p, _mm512_set1_epi32(c as u32 as i32)),
            );
            let wide = _mm512_add_epi64(
                _mm512_cvtepu32_epi64(_mm512_castsi512_si256(raw)),
                _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(raw)),
            );
            c = c.wrapping_add(_mm512_reduce_add_epi64(wide) as u64 + 16);
            i += 16;
        }
    }
    // SAFETY: `i <= n`.
    ascend_tail(unsafe { v.get_unchecked_mut(i..) }, c)
}

/// Dictionary entries for 16 ranks: register permutes when the
/// dictionary fits one or two vectors, else a gather.
struct DictLookup {
    lo: __m512i,
    hi: __m512i,
    small: bool,
    wide: bool,
}

impl DictLookup {
    /// # Safety
    /// AVX-512F is available.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn new(dict: &[u32]) -> Self {
        let k = dict.len();
        // SAFETY: masked loads read only the first `min(k, 32)` entries.
        unsafe {
            let m = |c: usize| {
                if c >= 16 {
                    0xffffu16
                } else {
                    ((1u32 << c) - 1) as u16
                }
            };
            let lo = _mm512_maskz_loadu_epi32(m(k), dict.as_ptr().cast());
            let hi = if k > 16 {
                _mm512_maskz_loadu_epi32(m(k - 16), dict.as_ptr().add(16).cast())
            } else {
                _mm512_setzero_si512()
            };
            DictLookup {
                lo,
                hi,
                small: k <= 16,
                wide: k > 32,
            }
        }
    }

    /// Entries at `rank` for the lanes in `lanes` (zero elsewhere).
    ///
    /// # Safety
    /// Ranks in `lanes` are below the dictionary length.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn get(&self, lanes: __mmask16, rank: __m512i, dict: &[u32]) -> __m512i {
        // SAFETY: as documented; gathered lanes index `dict` in range.
        unsafe {
            if self.small {
                _mm512_maskz_permutexvar_epi32(lanes, rank, self.lo)
            } else if !self.wide {
                _mm512_maskz_permutex2var_epi32(lanes, self.lo, rank, self.hi)
            } else {
                _mm512_mask_i32gather_epi32::<4>(
                    _mm512_setzero_si512(),
                    lanes,
                    rank,
                    dict.as_ptr().cast(),
                )
            }
        }
    }
}

/// Stores the lanes of `keep` packed at `cols + j` and `vals + j` with
/// full-vector stores (an in-register compress is much cheaper than a
/// compressing store); returns the count.
///
/// # Safety
/// `cols` and `vals` have 16 writable entries at `j`.
#[target_feature(enable = "avx512f,popcnt")]
#[inline]
unsafe fn put16(
    keep: __mmask16,
    col: __m512i,
    v: __m512i,
    cols: *mut u32,
    vals: *mut u32,
) -> usize {
    // SAFETY: as documented.
    unsafe {
        _mm512_storeu_si512(cols.cast(), _mm512_maskz_compress_epi32(keep, col));
        _mm512_storeu_si512(vals.cast(), _mm512_maskz_compress_epi32(keep, v));
    }
    keep.count_ones() as usize
}

/// Row mapping 16 elements at a time: gather the columns, look up the
/// values, and pack the mapped ones. Outputs are written with full vectors
/// at the packed position, which never passes `i + 16 <= n`.
#[target_feature(enable = "avx512f,avx2,popcnt")]
unsafe fn map_u8_avx512(
    ids: &[u32],
    map: &[u32],
    ranks: &[u8],
    dict: &[u32],
    cols: &mut [u32],
    vals: &mut [u32],
) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: loads cover `ids[i..i + 16]` and `ranks[i..i + 16]` with
    // `i + 16 <= n`; gathers index `map` by ids and `dict` by ranks, both in
    // range by the caller's contract; stores write `cols` and `vals` at
    // `j..j + 16` with `j <= i`.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        let lookup = DictLookup::new(dict);
        while i + 16 <= n {
            let col = _mm512_i32gather_epi32::<4>(
                _mm512_loadu_si512(ids.as_ptr().add(i).cast()),
                map.as_ptr().cast(),
            );
            let rank = _mm512_cvtepu8_epi32(_mm_loadu_si128(ranks.as_ptr().add(i).cast()));
            let v = lookup.get(0xffff, rank, dict);
            let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
            j += put16(
                keep,
                col,
                v,
                cols.as_mut_ptr().add(j),
                vals.as_mut_ptr().add(j),
            );
            i += 16;
        }
    }
    // SAFETY: the tail keeps the kernel's contract, and `j <= i` leaves
    // room for the remaining elements in both outputs.
    unsafe {
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| get(dict, get(r, t) as usize), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// As [`map_u8_avx512`] with 32-bit ranks.
#[target_feature(enable = "avx512f,avx2,popcnt")]
unsafe fn map_u32_avx512(
    ids: &[u32],
    map: &[u32],
    ranks: &[u32],
    dict: &[u32],
    cols: &mut [u32],
    vals: &mut [u32],
) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_u8_avx512`.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        while i + 16 <= n {
            let col = _mm512_i32gather_epi32::<4>(
                _mm512_loadu_si512(ids.as_ptr().add(i).cast()),
                map.as_ptr().cast(),
            );
            let v = _mm512_i32gather_epi32::<4>(
                _mm512_loadu_si512(ranks.as_ptr().add(i).cast()),
                dict.as_ptr().cast(),
            );
            let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
            j += put16(
                keep,
                col,
                v,
                cols.as_mut_ptr().add(j),
                vals.as_mut_ptr().add(j),
            );
            i += 16;
        }
    }
    // SAFETY: the tail keeps the kernel's contract, and `j <= i` leaves
    // room for the remaining elements in both outputs.
    unsafe {
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| get(dict, get(r, t) as usize), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// As [`map_u8_avx512`] for raw 4-byte values copied as they are.
#[target_feature(enable = "avx512f,avx2,popcnt")]
unsafe fn map_raw32_avx512(
    ids: &[u32],
    map: &[u32],
    raw: &[u8],
    cols: &mut [u32],
    vals: &mut [u32],
) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_u8_avx512`, with the 64 bytes of values `i..i + 16` loaded.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        while i + 16 <= n {
            let col = _mm512_i32gather_epi32::<4>(
                _mm512_loadu_si512(ids.as_ptr().add(i).cast()),
                map.as_ptr().cast(),
            );
            let v = _mm512_loadu_si512(raw.as_ptr().add(4 * i).cast());
            let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
            j += put16(
                keep,
                col,
                v,
                cols.as_mut_ptr().add(j),
                vals.as_mut_ptr().add(j),
            );
            i += 16;
        }
    }
    // SAFETY: the tail keeps the kernel's contract, and `j <= i` leaves
    // room for the remaining elements in both outputs.
    unsafe {
        let (ids, r) = (ids.get_unchecked(i..), raw.get_unchecked(4 * i..));
        j + compact(ids, map, |t| src_value(r, t, Src::W32), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// Byte ranks at 16 positions, gathered as 4-byte words; lanes within 3
/// bytes of the end are not read (zero) and are reported clear in the mask.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn ranks8(pos: __m512i, r: &[u8]) -> (__m512i, __mmask16) {
    // SAFETY: only lanes with `pos + 4 <= r.len()` are gathered.
    unsafe {
        let safe =
            _mm512_cmplt_epu32_mask(pos, _mm512_set1_epi32(r.len().saturating_sub(3) as i32));
        let w =
            _mm512_mask_i32gather_epi32::<1>(_mm512_setzero_si512(), safe, pos, r.as_ptr().cast());
        (_mm512_and_si512(w, _mm512_set1_epi32(0xff)), safe)
    }
}

/// Gathers the mapped columns and values of 16 positions at a time. Byte
/// ranks are gathered as 4-byte words, so lanes within 3 bytes of the end
/// of the ranks are skipped by the gather and filled one by one.
#[target_feature(enable = "avx512f,avx2,popcnt")]
unsafe fn map_at_avx512(
    ids: &[u32],
    positions: &[u32],
    map: &[u32],
    v: At,
    cols: &mut [u32],
    vals: &mut [u32],
) {
    let n = positions.len();
    let mut i = 0;
    // SAFETY: loads cover `positions[i..i + 16]` with `i + 16 <= n`; gathers
    // index `ids` by positions, `map` and the modes by ids, the dictionary
    // by ranks and raw values by position, all in range per the caller's
    // contract; byte-rank gathers read 4 bytes only where `pos + 4 <= len`;
    // stores cover `cols[i..i + 16]` and `vals[i..i + 16]`.
    unsafe {
        macro_rules! run {
            (|$pos:ident, $id:ident| $value:expr) => {
                while i + 16 <= n {
                    let $pos = _mm512_loadu_si512(positions.as_ptr().add(i).cast());
                    let $id = _mm512_i32gather_epi32::<4>($pos, ids.as_ptr().cast());
                    let col = _mm512_i32gather_epi32::<4>($id, map.as_ptr().cast());
                    _mm512_storeu_si512(cols.as_mut_ptr().add(i).cast(), col);
                    _mm512_storeu_si512(vals.as_mut_ptr().add(i).cast(), $value);
                    i += 16;
                }
            };
        }
        let fix = |i: usize, safe: __mmask16, vals: &mut [u32]| {
            let mut rest = !safe;
            while rest != 0 {
                let l = rest.trailing_zeros() as usize;
                let p = positions[i + l] as usize;
                vals[i + l] = at_value(&v, p, get(ids, p));
                rest &= rest - 1;
            }
        };
        match v {
            At::Const(c) => run!(|_pos, _id| _mm512_set1_epi32(c as i32)),
            At::Dict32(r, d) => run!(|pos, _id| _mm512_i32gather_epi32::<4>(
                _mm512_i32gather_epi32::<4>(pos, r.as_ptr().cast()),
                d.as_ptr().cast()
            )),
            At::Raw(b, Src::W32, Conv::Bits) => run!(|pos, _id| _mm512_i32gather_epi32::<4>(pos, b.as_ptr().cast())),
            At::Raw(b, Src::W32, Conv::U32ToF32) => run!(|pos, _id| conv_lanes(_mm512_i32gather_epi32::<4>(pos, b.as_ptr().cast()), Conv::U32ToF32)),
            At::Raw(b, Src::W32, Conv::I32ToF32) => run!(|pos, _id| conv_lanes(_mm512_i32gather_epi32::<4>(pos, b.as_ptr().cast()), Conv::I32ToF32)),
            // Narrow raw values (rare here: cached rows are not partial)
            // take the scalar loop below.
            At::Raw(..) => {}
            At::ModesOne(m) => run!(|_pos, id| _mm512_i32gather_epi32::<4>(id, m.as_ptr().cast())),
            At::Modes32(r, d, m) => {
                let lookup = DictLookup::new(d);
                run!(|pos, id| mode_values(
                    id,
                    _mm512_i32gather_epi32::<4>(pos, r.as_ptr().cast()),
                    &lookup,
                    d,
                    m
                ))
            }
            At::Dict8(r, d) => {
                let lookup = DictLookup::new(d);
                while i + 16 <= n {
                    let pos = _mm512_loadu_si512(positions.as_ptr().add(i).cast());
                    let id = _mm512_i32gather_epi32::<4>(pos, ids.as_ptr().cast());
                    let col = _mm512_i32gather_epi32::<4>(id, map.as_ptr().cast());
                    let (rank, safe) = ranks8(pos, r);
                    _mm512_storeu_si512(cols.as_mut_ptr().add(i).cast(), col);
                    _mm512_storeu_si512(vals.as_mut_ptr().add(i).cast(), lookup.get(safe, rank, d));
                    fix(i, safe, vals);
                    i += 16;
                }
            }
            At::Modes8(r, d, m) => {
                let lookup = DictLookup::new(d);
                while i + 16 <= n {
                    let pos = _mm512_loadu_si512(positions.as_ptr().add(i).cast());
                    let id = _mm512_i32gather_epi32::<4>(pos, ids.as_ptr().cast());
                    let col = _mm512_i32gather_epi32::<4>(id, map.as_ptr().cast());
                    let (rank, safe) = ranks8(pos, r);
                    _mm512_storeu_si512(cols.as_mut_ptr().add(i).cast(), col);
                    _mm512_storeu_si512(
                        vals.as_mut_ptr().add(i).cast(),
                        mode_values(id, rank, &lookup, d, m),
                    );
                    fix(i, safe, vals);
                    i += 16;
                }
            }
        }
    }
    // SAFETY: the tail keeps the kernel's contract.
    unsafe {
        for k in i..n {
            let p = get(positions, k) as usize;
            let id = get(ids, p);
            set(cols, k, get(map, id as usize));
            set(vals, k, at_value(&v, p, id));
        }
    }
}

/// Scatters 16 elements at a time: gathered columns and values, masked to
/// mapped columns, each vector checked against the row length first. The
/// value source is resolved before the loop, one loop per kind, so the
/// loop body has no dispatch.
#[target_feature(enable = "avx512f,avx2,popcnt")]
unsafe fn scatter_avx512(ids: &[u32], map: &[u32], v: At, out: &mut [u32]) -> bool {
    let n = ids.len();
    let mut i = 0;
    // SAFETY: loads cover `ids[i..i + 16]` with `i + 16 <= n`; gathers index
    // `map`, the modes and the values in range per the caller's contract;
    // scatters write only mapped lanes whose column was checked to be below
    // `out.len()`; lanes a byte-rank gather skips are stored one by one.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        let width = _mm512_set1_epi32(out.len().min(i32::MAX as usize) as i32);
        let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        macro_rules! run {
            (|$pos:ident, $id:ident, $at:ident| $value:expr) => {
                while i + 16 <= n {
                    let $at = i;
                    let $pos = _mm512_add_epi32(_mm512_set1_epi32(i as i32), iota);
                    let $id = _mm512_loadu_si512(ids.as_ptr().add(i).cast());
                    let col = _mm512_i32gather_epi32::<4>($id, map.as_ptr().cast());
                    let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
                    if _mm512_mask_cmpge_epu32_mask(keep, col, width) != 0 {
                        return false;
                    }
                    let (val, fix): (__m512i, __mmask16) = $value;
                    _mm512_mask_i32scatter_epi32::<4>(out.as_mut_ptr().cast(), keep & !fix, col, val);
                    let mut rest = keep & fix;
                    while rest != 0 {
                        let t = i + rest.trailing_zeros() as usize;
                        let id = get(ids, t);
                        set(out, get(map, id as usize) as usize, at_value(&v, t, id));
                        rest &= rest - 1;
                    }
                    i += 16;
                }
            };
        }
        macro_rules! raw {
            ($b:ident, $src:expr, $conv:ident) => {
                match $conv {
                    Conv::Bits => run!(|_pos, _id, at| (conv_lanes(load_lanes($b.as_ptr(), at, $src), Conv::Bits), 0)),
                    Conv::U32ToF32 => run!(|_pos, _id, at| (conv_lanes(load_lanes($b.as_ptr(), at, $src), Conv::U32ToF32), 0)),
                    Conv::I32ToF32 => run!(|_pos, _id, at| (conv_lanes(load_lanes($b.as_ptr(), at, $src), Conv::I32ToF32), 0)),
                }
            };
        }
        match v {
            At::Const(c) => run!(|_pos, _id, _at| (_mm512_set1_epi32(c as i32), 0)),
            At::Raw(b, Src::U8, conv) => raw!(b, Src::U8, conv),
            At::Raw(b, Src::I8, conv) => raw!(b, Src::I8, conv),
            At::Raw(b, Src::U16, conv) => raw!(b, Src::U16, conv),
            At::Raw(b, Src::I16, conv) => raw!(b, Src::I16, conv),
            At::Raw(b, Src::W32, conv) => raw!(b, Src::W32, conv),
            At::Dict32(r, d) => {
                let lookup = DictLookup::new(d);
                run!(|_pos, _id, at| (lookup.get(0xffff, _mm512_loadu_si512(r.as_ptr().add(at).cast()), d), 0))
            }
            At::Dict8(r, d) => {
                let lookup = DictLookup::new(d);
                run!(|pos, _id, _at| {
                    let (rank, safe) = ranks8(pos, r);
                    (lookup.get(safe, rank, d), !safe)
                })
            }
            At::ModesOne(m) => run!(|_pos, id, _at| (_mm512_i32gather_epi32::<4>(id, m.as_ptr().cast()), 0)),
            At::Modes32(r, d, m) => {
                let lookup = DictLookup::new(d);
                run!(|_pos, id, at| (mode_values(id, _mm512_loadu_si512(r.as_ptr().add(at).cast()), &lookup, d, m), 0))
            }
            At::Modes8(r, d, m) => {
                let lookup = DictLookup::new(d);
                run!(|pos, id, _at| {
                    let (rank, safe) = ranks8(pos, r);
                    (mode_values(id, rank, &lookup, d, m), !safe)
                })
            }
        }
    }
    // SAFETY: the tail keeps the kernel's contract; each store's column is
    // checked against the row length first.
    unsafe {
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

/// Mode-coded values of 16 elements: lanes with rank 0 gather their column
/// mode, the others take dictionary entry `rank - 1`; the gather is masked
/// to its lanes, so no lane reads out of range.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn mode_values(
    id: __m512i,
    rank: __m512i,
    lookup: &DictLookup,
    dict: &[u32],
    modes: &[u32],
) -> __m512i {
    // SAFETY: masked lanes are not read; the others index `modes` by id and
    // `dict` by `rank - 1 < dict.len()`, per the caller's contract.
    unsafe {
        let zero = _mm512_cmpeq_epi32_mask(rank, _mm512_setzero_si512());
        let m = _mm512_mask_i32gather_epi32::<4>(
            _mm512_setzero_si512(),
            zero,
            id,
            modes.as_ptr().cast(),
        );
        let d = lookup.get(!zero, _mm512_sub_epi32(rank, _mm512_set1_epi32(1)), dict);
        _mm512_or_si512(m, d)
    }
}

/// As [`map_u8_avx512`] for mode-coded rows.
#[target_feature(enable = "avx512f,avx2,popcnt")]
unsafe fn map_modes_u8_avx512(
    ids: &[u32],
    map: &[u32],
    ranks: &[u8],
    dict: &[u32],
    modes: &[u32],
    cols: &mut [u32],
    vals: &mut [u32],
) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_u8_avx512`, with the mode gathers of `mode_values`.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        let lookup = DictLookup::new(dict);
        while i + 16 <= n {
            let id = _mm512_loadu_si512(ids.as_ptr().add(i).cast());
            let col = _mm512_i32gather_epi32::<4>(id, map.as_ptr().cast());
            let rank = _mm512_cvtepu8_epi32(_mm_loadu_si128(ranks.as_ptr().add(i).cast()));
            let v = mode_values(id, rank, &lookup, dict, modes);
            let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
            j += put16(
                keep,
                col,
                v,
                cols.as_mut_ptr().add(j),
                vals.as_mut_ptr().add(j),
            );
            i += 16;
        }
    }
    // SAFETY: the tail keeps the kernel's contract, and `j <= i` leaves
    // room for the remaining elements in both outputs.
    unsafe {
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| super::mode_value(get(ids, t), get(r, t) as usize, dict, modes), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// As [`map_modes_u8_avx512`] with 32-bit ranks.
#[target_feature(enable = "avx512f,avx2,popcnt")]
unsafe fn map_modes_u32_avx512(
    ids: &[u32],
    map: &[u32],
    ranks: &[u32],
    dict: &[u32],
    modes: &[u32],
    cols: &mut [u32],
    vals: &mut [u32],
) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_modes_u8_avx512`.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        let lookup = DictLookup::new(dict);
        while i + 16 <= n {
            let id = _mm512_loadu_si512(ids.as_ptr().add(i).cast());
            let col = _mm512_i32gather_epi32::<4>(id, map.as_ptr().cast());
            let rank = _mm512_loadu_si512(ranks.as_ptr().add(i).cast());
            let v = mode_values(id, rank, &lookup, dict, modes);
            let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
            j += put16(
                keep,
                col,
                v,
                cols.as_mut_ptr().add(j),
                vals.as_mut_ptr().add(j),
            );
            i += 16;
        }
    }
    // SAFETY: the tail keeps the kernel's contract, and `j <= i` leaves
    // room for the remaining elements in both outputs.
    unsafe {
        let (ids, r) = (ids.get_unchecked(i..), ranks.get_unchecked(i..));
        j + compact(ids, map, |t| super::mode_value(get(ids, t), get(r, t) as usize, dict, modes), cols.get_unchecked_mut(j..), vals.get_unchecked_mut(j..))
    }
}

/// Sixteen raw values from element `i` as 32-bit lanes.
///
/// # Safety
/// The values hold elements `i..i + 16`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn load_lanes(raw: *const u8, i: usize, src: Src) -> __m512i {
    // SAFETY: as documented.
    unsafe {
        match src {
            Src::U8 => _mm512_cvtepu8_epi32(_mm_loadu_si128(raw.add(i).cast())),
            Src::I8 => _mm512_cvtepi8_epi32(_mm_loadu_si128(raw.add(i).cast())),
            Src::U16 => _mm512_cvtepu16_epi32(_mm256_loadu_si256(raw.add(2 * i).cast())),
            Src::I16 => _mm512_cvtepi16_epi32(_mm256_loadu_si256(raw.add(2 * i).cast())),
            Src::W32 => _mm512_loadu_si512(raw.add(4 * i).cast()),
        }
    }
}

/// Sixteen lanes converted as [`Conv`] describes.
#[target_feature(enable = "avx512f")]
#[inline]
fn conv_lanes(v: __m512i, conv: Conv) -> __m512i {
    match conv {
        Conv::Bits => v,
        Conv::U32ToF32 => _mm512_castps_si512(_mm512_cvtepu32_ps(v)),
        Conv::I32ToF32 => _mm512_castps_si512(_mm512_cvtepi32_ps(v)),
    }
}

/// Stores the kept lanes packed: columns, then values truncated to `OUT`
/// bytes; returns the count.
///
/// # Safety
/// `cols` has 16 writable entries and `vals` 16 elements of `OUT` bytes.
#[target_feature(enable = "avx512f,avx512bw,popcnt")]
#[inline]
unsafe fn put_lanes<const OUT: usize>(keep: __mmask16, col: __m512i, v: __m512i, cols: *mut u32, vals: *mut u8) -> usize {
    // SAFETY: as documented.
    unsafe {
        _mm512_storeu_si512(cols.cast(), _mm512_maskz_compress_epi32(keep, col));
        let v = _mm512_maskz_compress_epi32(keep, v);
        match OUT {
            4 => _mm512_storeu_si512(vals.cast(), v),
            2 => _mm256_storeu_si256(vals.cast(), _mm512_cvtepi32_epi16(v)),
            _ => _mm_storeu_si128(vals.cast(), _mm512_cvtepi32_epi8(v)),
        }
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

/// [`Src`] of a lane kernel's `SRC` constant.
const fn src_of(code: u8) -> Src {
    match code {
        0 => Src::U8,
        1 => Src::I8,
        2 => Src::U16,
        3 => Src::I16,
        _ => Src::W32,
    }
}

/// [`Conv`] of a lane kernel's `CONV` constant.
const fn conv_of(code: u8) -> Conv {
    match code {
        0 => Conv::Bits,
        1 => Conv::U32ToF32,
        _ => Conv::I32ToF32,
    }
}

/// Raw values of source kind `SRC`, converted by `CONV`, mapped 16 at a
/// time to `OUT`-byte elements (see [`super::MapLanesFn`]).
#[target_feature(enable = "avx512f,avx512bw,avx2,popcnt")]
unsafe fn map_lanes_avx512<const SRC: u8, const CONV: u8, const OUT: usize>(ids: &[u32], map: &[u32], raw: &[u8], cols: &mut [u32], vals: *mut u8) -> usize {
    let (src, conv) = (src_of(SRC), conv_of(CONV));
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: loads cover elements `i..i + 16` with `i + 16 <= n`; gathers
    // index `map` by ids in range per the caller's contract; stores write
    // entries `j..j + 16` with `j <= i`, inside the outputs of `n` entries.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        while i + 16 <= n {
            let col = _mm512_i32gather_epi32::<4>(_mm512_loadu_si512(ids.as_ptr().add(i).cast()), map.as_ptr().cast());
            let v = conv_lanes(load_lanes(raw.as_ptr(), i, src), conv);
            let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
            j += put_lanes::<OUT>(keep, col, v, cols.as_mut_ptr().add(j), vals.add(j * OUT));
            i += 16;
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

/// Byte-ranked dictionary values (entries as 32-bit lanes) mapped 16 at a
/// time to `OUT`-byte elements (see [`super::MapDictLanesFn`]).
#[target_feature(enable = "avx512f,avx512bw,avx2,popcnt")]
unsafe fn map_dict8_lanes_avx512<const OUT: usize>(ids: &[u32], map: &[u32], ranks: &[u8], dict: &[u32], cols: &mut [u32], vals: *mut u8) -> usize {
    let n = ids.len();
    let (mut i, mut j) = (0, 0);
    // SAFETY: as in `map_lanes_avx512`, ranks indexing the dictionary per
    // the caller's contract.
    unsafe {
        let unmapped = _mm512_set1_epi32(-1);
        let lookup = DictLookup::new(dict);
        while i + 16 <= n {
            let col = _mm512_i32gather_epi32::<4>(_mm512_loadu_si512(ids.as_ptr().add(i).cast()), map.as_ptr().cast());
            let rank = _mm512_cvtepu8_epi32(_mm_loadu_si128(ranks.as_ptr().add(i).cast()));
            let v = lookup.get(0xffff, rank, dict);
            let keep = _mm512_cmpneq_epi32_mask(col, unmapped);
            j += put_lanes::<OUT>(keep, col, v, cols.as_mut_ptr().add(j), vals.add(j * OUT));
            i += 16;
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
const LANES_AVX512: LanesTable = {
    let mut t: LanesTable = [[[None; 3]; 3]; 5];
    macro_rules! put {
        ($($s:literal $c:literal $o:literal),* $(,)?) => {
            $( t[$s][$c][super::out_slot($o)] = Some(map_lanes_avx512::<$s, $c, $o> as super::MapLanesFn); )*
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
