//! Vector rANS encoders. A stream with at least 8 states codes one group of
//! `lanes` consecutive symbols per step, one vector of states per 8 (AVX2)
//! or 16 (AVX-512) lanes: each symbol belongs to its own lane, so the lanes
//! of a group are independent. Words leave a group in descending symbol
//! order, so in stream order they are the emitting lanes ascending; they are
//! stored backwards from the end of the scratch, vector by vector.
//!
//! Division by the frequency is exact in `f64`: for a state `x < 2^32`,
//! `f <= M` and `inv` within `2^-47` of `1 / f` (relative),
//! `floor(x * inv + 2^-15)` equals `floor(x / f)`, since the product's
//! error stays below `2^-15` while a non-integer quotient lies at least
//! `1 / f >= 2^-11` below the next integer. The AVX2 encoder gathers `1 / f`
//! rounded to nearest; the AVX-512 one refines a 14-bit estimate twice by
//! Newton's method, to within a few ulps.

use std::arch::x86_64::*;

use crate::scans::rans::{EMIT_SHIFT, EncModel, Job, SCALE, WORD_SLACK, encode_scalar, many_scalar};
use crate::scans::symbol::DIRECT;

/// `2^-15`, the bias that makes truncation of the quotient exact.
const BIAS: f64 = 1.0 / 32768.0;

/// Constants of the 16-lane encoder.
struct Wide {
    low16: __m512i,
    bias: __m512d,
    one: __m512d,
}

/// One step of 16-lane states: `active` lanes code symbols `syms[j0..]`,
/// storing their emitted words below `*pos`. False on a zero frequency.
///
/// # Safety
/// The active lanes' symbols exist; `*pos` stays at or above the number of
/// emitted words; the CPU has AVX-512 F, BW, VL, DQ and FMA.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,popcnt")]
unsafe fn wide_step(m: &EncModel, c: &Wide, syms: *const u8, wp: *mut u16, pos: &mut usize, x: &mut __m512i, active: u16) -> bool {
    // SAFETY: as documented; gathers index the 256-entry table with bytes.
    unsafe {
        let zero = _mm512_setzero_si512();
        let s = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(active, syms.cast()));
        let fc = _mm512_mask_i32gather_epi32::<4>(zero, active, s, m.fc.as_ptr().cast());
        let f = _mm512_and_si512(fc, c.low16);
        if _mm512_mask_cmpeq_epi32_mask(active, f, zero) != 0 {
            return false;
        }
        let v = *x;
        let emit = _mm512_mask_cmpge_epu32_mask(active, _mm512_srli_epi32::<EMIT_SHIFT>(v), f);
        let cnt = emit.count_ones() as usize;
        *pos -= cnt;
        let kept = _mm512_cvtepi32_epi16(_mm512_maskz_compress_epi32(emit, v));
        _mm256_mask_storeu_epi16(wp.add(*pos).cast(), ((1u32 << cnt) - 1) as u16, kept);
        let v = _mm512_mask_srli_epi32::<16>(v, emit, v);
        // `1 / f` from a 14-bit estimate refined twice by Newton's method,
        // within a few ulps (see the module docs for the margin); it depends
        // on the symbols only, off the chain through the states.
        let quotient = |vh: __m256i, fh: __m256i| {
            let fd = _mm512_cvtepu32_pd(fh);
            let r = _mm512_rcp14_pd(fd);
            let r = _mm512_fmadd_pd(r, _mm512_fnmadd_pd(fd, r, c.one), r);
            let inv = _mm512_fmadd_pd(r, _mm512_fnmadd_pd(fd, r, c.one), r);
            _mm512_cvttpd_epu32(_mm512_fmadd_pd(_mm512_cvtepu32_pd(vh), inv, c.bias))
        };
        let qlo = quotient(_mm512_castsi512_si256(v), _mm512_castsi512_si256(f));
        let qhi = quotient(_mm512_extracti64x4_epi64::<1>(v), _mm512_extracti64x4_epi64::<1>(f));
        let q = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qlo), qhi);
        // `x' = q * M + (v - q * f) + cum = v + cum + q * (M - f)`.
        let spare = _mm512_sub_epi32(_mm512_set1_epi32(1 << SCALE), f);
        let next = _mm512_add_epi32(_mm512_add_epi32(v, _mm512_srli_epi32::<16>(fc)), _mm512_mullo_epi32(q, spare));
        *x = _mm512_mask_mov_epi32(*x, active, next);
        true
    }
}

/// The [`super::RansEncodeFn`] contract with 16-lane vectors (8 lanes use
/// the low half of one).
///
/// # Safety
/// As [`encode_scalar`], on a CPU with AVX-512 F, BW, VL and DQ.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,popcnt")]
pub(super) unsafe fn rans_encode_avx512(m: &EncModel, syms: &[u8], lanes: usize, x: &mut [u32; 32], words: &mut [u16]) -> Option<usize> {
    if lanes < 8 {
        // SAFETY: as documented.
        return unsafe { encode_scalar(m, syms, lanes, x, words) };
    }
    debug_assert!(words.len() >= syms.len() + WORD_SLACK);
    let n = syms.len();
    let width = lanes.min(16);
    let full = ((1u32 << width) - 1) as u16;
    let groups = n / lanes;
    let rem = n - groups * lanes;
    let mut pos = words.len();
    // SAFETY: symbol loads are masked to the active lanes, below `n`; each
    // step stores at most one word per active lane below `pos`, which stays
    // at or above `words.len() - n`; the CPU features are enabled here.
    unsafe {
        let c = Wide { low16: _mm512_set1_epi32(0xffff), bias: _mm512_set1_pd(BIAS), one: _mm512_set1_pd(1.0) };
        let (sp, wp) = (syms.as_ptr(), words.as_mut_ptr());
        let mut x0 = _mm512_loadu_si512(x.as_ptr().cast());
        let mut x1 = _mm512_loadu_si512(x.as_ptr().add(16).cast());
        let live = |h: usize| ((1u32 << rem.saturating_sub(width * h).min(width)) - 1) as u16;
        let tail = groups * lanes;
        if lanes == 32 {
            if rem > 16 && !wide_step(m, &c, sp.add(tail + 16), wp, &mut pos, &mut x1, live(1)) {
                return None;
            }
            if rem > 0 && !wide_step(m, &c, sp.add(tail), wp, &mut pos, &mut x0, live(0)) {
                return None;
            }
            for g in (0..groups).rev() {
                let j0 = g * 32;
                if !wide_step(m, &c, sp.add(j0 + 16), wp, &mut pos, &mut x1, full) || !wide_step(m, &c, sp.add(j0), wp, &mut pos, &mut x0, full) {
                    return None;
                }
            }
        } else {
            if rem > 0 && !wide_step(m, &c, sp.add(tail), wp, &mut pos, &mut x0, live(0)) {
                return None;
            }
            for g in (0..groups).rev() {
                if !wide_step(m, &c, sp.add(g * lanes), wp, &mut pos, &mut x0, full) {
                    return None;
                }
            }
        }
        _mm512_storeu_si512(x.as_mut_ptr().cast(), x0);
        _mm512_storeu_si512(x.as_mut_ptr().add(16).cast(), x1);
    }
    Some(words.len() - pos)
}

/// Jobs interleaved group by group, `J` of them, 16-lane states in
/// registers; a job meeting a zero frequency stops.
///
/// # Safety
/// As [`rans_encode_avx512`] for every job, with `lanes >= 8` and every
/// job's symbols of one count.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,popcnt")]
unsafe fn many_wide<const J: usize>(jobs: &mut [Job], lanes: usize) {
    debug_assert!(jobs.len() == J && lanes >= 8);
    let n = jobs[0].syms.len();
    let width = lanes.min(16);
    let full = ((1u32 << width) - 1) as u16;
    let groups = n / lanes;
    let rem = n - groups * lanes;
    let tail = groups * lanes;
    let live = |h: usize| ((1u32 << rem.saturating_sub(width * h).min(width)) - 1) as u16;
    // SAFETY: as for `rans_encode_avx512`, per job.
    unsafe {
        let c = Wide { low16: _mm512_set1_epi32(0xffff), bias: _mm512_set1_pd(BIAS), one: _mm512_set1_pd(1.0) };
        let zero = _mm512_setzero_si512();
        let (mut x0, mut x1, mut pos, mut alive) = ([zero; J], [zero; J], [0usize; J], [true; J]);
        let mut sp = [std::ptr::null::<u8>(); J];
        let mut wp = [std::ptr::null_mut::<u16>(); J];
        for j in 0..J {
            x0[j] = _mm512_loadu_si512(jobs[j].x.as_ptr().cast());
            x1[j] = _mm512_loadu_si512(jobs[j].x.as_ptr().add(16).cast());
            pos[j] = jobs[j].words.len();
            sp[j] = jobs[j].syms.as_ptr();
            wp[j] = jobs[j].words.as_mut_ptr();
        }
        let models: [&EncModel; J] = std::array::from_fn(|j| jobs[j].model);
        macro_rules! step {
            ($x:ident, $at:expr, $mask:expr) => {
                for j in 0..J {
                    if alive[j] {
                        alive[j] = wide_step(models[j], &c, sp[j].add($at), wp[j], &mut pos[j], &mut $x[j], $mask);
                    }
                }
            };
        }
        if lanes == 32 {
            if rem > 16 {
                step!(x1, tail + 16, live(1));
            }
            if rem > 0 {
                step!(x0, tail, live(0));
            }
            for g in (0..groups).rev() {
                step!(x1, g * 32 + 16, full);
                step!(x0, g * 32, full);
            }
        } else {
            if rem > 0 {
                step!(x0, tail, live(0));
            }
            for g in (0..groups).rev() {
                step!(x0, g * lanes, full);
            }
        }
        for j in 0..J {
            _mm512_storeu_si512(jobs[j].x.as_mut_ptr().cast(), x0[j]);
            _mm512_storeu_si512(jobs[j].x.as_mut_ptr().add(16).cast(), x1[j]);
            jobs[j].nw = alive[j].then(|| jobs[j].words.len() - pos[j]);
        }
    }
}

/// The [`super::RansManyFn`] contract with 16-lane vectors, the jobs
/// interleaved.
///
/// # Safety
/// As [`rans_encode_avx512`] for every job.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,popcnt")]
pub(super) unsafe fn rans_many_avx512(jobs: &mut [Job], lanes: usize) {
    // SAFETY: as documented.
    unsafe {
        if lanes < 8 {
            return many_scalar(jobs, lanes);
        }
        match jobs.len() {
            0 => {}
            1 => many_wide::<1>(jobs, lanes),
            2 => many_wide::<2>(jobs, lanes),
            _ => many_wide::<3>(jobs, lanes),
        }
    }
}

/// The [`super::RansManyFn`] contract with the 8-lane encoder per job.
///
/// # Safety
/// As [`rans_encode_avx2`] for every job.
#[target_feature(enable = "avx2,fma,popcnt")]
pub(super) unsafe fn rans_many_avx2(jobs: &mut [Job], lanes: usize) {
    for j in jobs {
        // SAFETY: as documented.
        j.nw = unsafe { rans_encode_avx2(j.model, j.syms, lanes, &mut j.x, j.words) };
    }
}

/// For each 8-bit mask, the set lanes ascending in the top positions of a
/// permutation (the others pick lane 0).
static TOP: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let cnt = (m as u32).count_ones() as usize;
        let mut at = 8 - cnt;
        let mut lane = 0;
        while lane < 8 {
            if m >> lane & 1 == 1 {
                t[m][at] = lane as u32;
                at += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
};

/// The [`super::RansEncodeFn`] contract with 8-lane vectors.
///
/// # Safety
/// As [`encode_scalar`], on a CPU with AVX2 and FMA.
#[target_feature(enable = "avx2,fma,popcnt")]
pub(super) unsafe fn rans_encode_avx2(m: &EncModel, syms: &[u8], lanes: usize, x: &mut [u32; 32], words: &mut [u16]) -> Option<usize> {
    if lanes < 8 {
        // SAFETY: as documented.
        return unsafe { encode_scalar(m, syms, lanes, x, words) };
    }
    debug_assert!(words.len() >= syms.len() + WORD_SLACK);
    let n = syms.len();
    let vectors = lanes / 8;
    let mut pos = words.len();
    // SAFETY: symbol loads read 8 bytes at `j0 + 8 <= n` or a padded copy;
    // gathers index the 256-entry tables with byte symbols; each vector
    // stores 8 words ending at `pos`, of which the emitted ones (at most
    // one per active lane) are kept: `pos - 8` stays at or above
    // `words.len() - n - 8 >= 0` by the word slack.
    unsafe {
        let mut xv = [_mm256_setzero_si256(); 4];
        for (h, v) in xv.iter_mut().enumerate() {
            *v = _mm256_loadu_si256(x.as_ptr().add(8 * h).cast());
        }
        let zero = _mm256_setzero_si256();
        let low16 = _mm256_set1_epi32(0xffff);
        let sign = _mm256_set1_epi32(i32::MIN);
        let two31 = _mm256_set1_pd(2147483648.0);
        let bias = _mm256_set1_pd(BIAS);
        let iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let wp = words.as_mut_ptr();
        // One vector of states: lanes below `live` code symbols `j0 + lane`.
        let mut step = |xv: &mut __m256i, j0: usize, live: usize| -> bool {
            let bytes = if live == 8 {
                _mm_loadl_epi64(syms.as_ptr().add(j0).cast())
            } else {
                let mut pad = [0u8; 8];
                std::ptr::copy_nonoverlapping(syms.as_ptr().add(j0), pad.as_mut_ptr(), live);
                _mm_loadl_epi64(pad.as_ptr().cast())
            };
            let active = _mm256_cmpgt_epi32(_mm256_set1_epi32(live as i32), iota);
            let s = _mm256_cvtepu8_epi32(bytes);
            let fc = _mm256_mask_i32gather_epi32::<4>(zero, m.fc.as_ptr().cast(), s, active);
            let f = _mm256_and_si256(fc, low16);
            if _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(_mm256_cmpeq_epi32(f, zero), active))) != 0 {
                return false;
            }
            let v = *xv;
            // Both sides stay below 2^31, so the signed compare is exact.
            let quiet = _mm256_cmpgt_epi32(f, _mm256_srli_epi32::<{ EMIT_SHIFT as i32 }>(v));
            let emit_v = _mm256_andnot_si256(quiet, active);
            let emit = _mm256_movemask_ps(_mm256_castsi256_ps(emit_v)) as usize;
            let cnt = emit.count_ones() as usize;
            let top = _mm256_permutevar8x32_epi32(_mm256_and_si256(v, low16), _mm256_loadu_si256(TOP[emit].as_ptr().cast()));
            let packed = _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_packus_epi32(top, zero));
            _mm_storeu_si128(wp.add(pos - 8).cast(), _mm256_castsi256_si128(packed));
            pos -= cnt;
            let v = _mm256_blendv_epi8(v, _mm256_srli_epi32::<16>(v), emit_v);
            // Unsigned conversions through a 2^31 offset.
            let quotient = |vh: __m128i, sh: __m128i, mh: __m128i| {
                let inv = _mm256_mask_i32gather_pd::<8>(_mm256_setzero_pd(), m.inv.as_ptr().cast(), sh, _mm256_castsi256_pd(_mm256_cvtepi32_epi64(mh)));
                let vd = _mm256_add_pd(_mm256_cvtepi32_pd(_mm_xor_si128(vh, _mm256_castsi256_si128(sign))), two31);
                let q = _mm256_floor_pd(_mm256_fmadd_pd(vd, inv, bias));
                _mm_xor_si128(_mm256_cvttpd_epi32(_mm256_sub_pd(q, two31)), _mm256_castsi256_si128(sign))
            };
            let qlo = quotient(_mm256_castsi256_si128(v), _mm256_castsi256_si128(s), _mm256_castsi256_si128(active));
            let qhi = quotient(_mm256_extracti128_si256::<1>(v), _mm256_extracti128_si256::<1>(s), _mm256_extracti128_si256::<1>(active));
            let q = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(qlo), qhi);
            let next = _mm256_add_epi32(
                _mm256_sub_epi32(_mm256_add_epi32(_mm256_slli_epi32::<{ SCALE as i32 }>(q), v), _mm256_mullo_epi32(q, f)),
                _mm256_srli_epi32::<16>(fc),
            );
            *xv = _mm256_blendv_epi8(*xv, next, active);
            true
        };
        let groups = n / lanes;
        let rem = n - groups * lanes;
        if rem > 0 {
            let j0 = groups * lanes;
            for h in (0..vectors).rev() {
                let live = rem.saturating_sub(8 * h).min(8);
                if live > 0 && !step(&mut xv[h], j0 + 8 * h, live) {
                    return None;
                }
            }
        }
        for g in (0..groups).rev() {
            for h in (0..vectors).rev() {
                if !step(&mut xv[h], g * lanes + 8 * h, 8) {
                    return None;
                }
            }
        }
        for (h, v) in xv.iter().enumerate() {
            _mm256_storeu_si256(x.as_mut_ptr().add(8 * h).cast(), *v);
        }
    }
    Some(words.len() - pos)
}

// ---- coded order ----

/// `perm[j] = pos[coded[j]]` with 16-lane gathers.
///
/// # Safety
/// Every id indexes `pos`; `perm` holds `coded.len()` entries.
#[target_feature(enable = "avx512f")]
unsafe fn gather_avx512(pos: &[u32], coded: &[u32], perm: &mut [u32]) {
    let n = coded.len();
    let mut j = 0;
    // SAFETY: as documented; vectors cover `j + 16 <= n`.
    unsafe {
        while j + 16 <= n {
            let c = _mm512_loadu_si512(coded.as_ptr().add(j).cast());
            _mm512_storeu_si512(perm.as_mut_ptr().add(j).cast(), _mm512_i32gather_epi32::<4>(c, pos.as_ptr().cast()));
            j += 16;
        }
        while j < n {
            *perm.get_unchecked_mut(j) = *pos.get_unchecked(*coded.get_unchecked(j) as usize);
            j += 1;
        }
    }
}

/// `perm[j] = pos[coded[j]]` with 8-lane gathers.
///
/// # Safety
/// As [`gather_avx512`], on a CPU with AVX2.
#[target_feature(enable = "avx2")]
unsafe fn gather_avx2(pos: &[u32], coded: &[u32], perm: &mut [u32]) {
    let n = coded.len();
    let mut j = 0;
    // SAFETY: as documented; vectors cover `j + 8 <= n`.
    unsafe {
        while j + 8 <= n {
            let c = _mm256_loadu_si256(coded.as_ptr().add(j).cast());
            _mm256_storeu_si256(perm.as_mut_ptr().add(j).cast(), _mm256_i32gather_epi32::<4>(pos.as_ptr().cast(), c));
            j += 8;
        }
        while j < n {
            *perm.get_unchecked_mut(j) = *pos.get_unchecked(*coded.get_unchecked(j) as usize);
            j += 1;
        }
    }
}

/// The [`super::ScanBitsFn`] contract: a byte compress packs each word's
/// set positions, of which the first 16 are always widened and stored.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,avx512vl,popcnt")]
pub(super) unsafe fn scan_bits_vbmi2(words: &mut [u64], pos: &[u32], coded: &mut [u32], perm: &mut [u32]) -> usize {
    let mut j = 0usize;
    // SAFETY: a word with `c` set bits stores `16 * max(1, ceil(c / 16))`
    // entries at `coded[j]`, with `j + c` never above the final count, so
    // every store ends within the count plus 32 entries.
    unsafe {
        let o = coded.as_mut_ptr();
        let iota = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );
        for (w, word) in words.iter_mut().enumerate() {
            let b = std::mem::take(word);
            let packed = _mm512_maskz_compress_epi8(b, iota);
            let base = _mm512_set1_epi32((w * 64) as i32);
            let widen = |q: __m128i| _mm512_add_epi32(base, _mm512_cvtepu8_epi32(q));
            _mm512_storeu_si512(o.add(j).cast(), widen(_mm512_castsi512_si128(packed)));
            let cnt = b.count_ones() as usize;
            if cnt > 16 {
                _mm512_storeu_si512(o.add(j + 16).cast(), widen(_mm512_extracti32x4_epi32::<1>(packed)));
                if cnt > 32 {
                    _mm512_storeu_si512(o.add(j + 32).cast(), widen(_mm512_extracti32x4_epi32::<2>(packed)));
                    _mm512_storeu_si512(o.add(j + 48).cast(), widen(_mm512_extracti32x4_epi32::<3>(packed)));
                }
            }
            j += cnt;
        }
        gather_avx512(pos, coded.get_unchecked(..j), perm);
    }
    j
}

/// The [`super::ScanBitsFn`] contract: each 16-bit quarter of a word
/// compresses its positions with a masked 32-bit compress.
#[target_feature(enable = "avx512f,popcnt")]
pub(super) unsafe fn scan_bits_avx512(words: &mut [u64], pos: &[u32], coded: &mut [u32], perm: &mut [u32]) -> usize {
    let mut j = 0usize;
    // SAFETY: each quarter stores 16 entries at `coded[j]`, where `j` never
    // exceeds the final count, which `coded` holds plus 16 entries.
    unsafe {
        let o = coded.as_mut_ptr();
        let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        for (w, word) in words.iter_mut().enumerate() {
            let b = std::mem::take(word);
            for q in 0..4 {
                let m = (b >> (16 * q)) as u16;
                let at = _mm512_add_epi32(iota, _mm512_set1_epi32((w * 64 + q * 16) as i32));
                _mm512_storeu_si512(o.add(j).cast(), _mm512_maskz_compress_epi32(m, at));
                j += m.count_ones() as usize;
            }
        }
        gather_avx512(pos, coded.get_unchecked(..j), perm);
    }
    j
}

/// The [`super::ScanBitsFn`] contract: each byte of a word stores its
/// positions from a table.
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn scan_bits_avx2(words: &mut [u64], pos: &[u32], coded: &mut [u32], perm: &mut [u32]) -> usize {
    let mut j = 0usize;
    // SAFETY: each byte stores 8 entries at `coded[j]`, where `j` never
    // exceeds the final count, which `coded` holds plus 8 entries.
    unsafe {
        let o = coded.as_mut_ptr();
        for (w, word) in words.iter_mut().enumerate() {
            let b = std::mem::take(word);
            if b == 0 {
                continue;
            }
            for byte in 0..8 {
                let m = (b >> (8 * byte)) as u8 as usize;
                let at = _mm256_loadu_si256(super::x86::BIT_POSITIONS[m].as_ptr().cast());
                _mm256_storeu_si256(o.add(j).cast(), _mm256_add_epi32(at, _mm256_set1_epi32((w * 64 + byte * 8) as i32)));
                j += m.count_ones() as usize;
            }
        }
        gather_avx2(pos, coded.get_unchecked(..j), perm);
    }
    j
}

// ---- gap symbols ----

/// The [`super::GapsFn`] contract with 16-lane vectors. A gap `g >= 256`
/// has symbol `DIRECT + bitlen(g) - 8 = DIRECT + bitlen(g >> 8)`, and
/// `g >> 8 < 2^24` converts to `f32` exactly, whose exponent field is
/// `126 + bitlen`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,popcnt")]
pub(super) unsafe fn gaps_avx512(coded: &[u32], syms: &mut [u8], esc: &mut [u32]) -> usize {
    let n = coded.len();
    let (mut t, mut e) = (0usize, 0usize);
    // SAFETY: vectors cover `t + 16 <= n` ids, symbols and escapes (at most
    // one per id, compress-stored); the tail is the portable kernel's.
    unsafe {
        let mut prev = _mm512_set1_epi32(-1);
        let one = _mm512_set1_epi32(1);
        let direct = _mm512_set1_epi32(DIRECT as i32);
        let first = _mm512_set1_epi32(256);
        let bias = _mm512_set1_epi32(DIRECT as i32 - 126);
        while t + 16 <= n {
            let c = _mm512_loadu_si512(coded.as_ptr().add(t).cast());
            let g = _mm512_sub_epi32(_mm512_sub_epi32(c, _mm512_alignr_epi32::<15>(c, prev)), one);
            prev = c;
            let small = _mm512_cmplt_epu32_mask(g, direct);
            let low = _mm512_cmplt_epu32_mask(g, first);
            let exp = _mm512_srli_epi32::<23>(_mm512_castps_si512(_mm512_cvtepi32_ps(_mm512_srli_epi32::<8>(g))));
            let s = _mm512_mask_mov_epi32(_mm512_mask_mov_epi32(_mm512_add_epi32(exp, bias), low, direct), small, g);
            _mm_storeu_si128(syms.as_mut_ptr().add(t).cast(), _mm512_cvtepi32_epi8(s));
            let escapes = !small;
            _mm512_mask_compressstoreu_epi32(esc.as_mut_ptr().add(e).cast(), escapes, g);
            e += escapes.count_ones() as usize;
            t += 16;
        }
        if t < n {
            let before = if t > 0 { *coded.get_unchecked(t - 1) } else { u32::MAX };
            let mut p = before;
            for k in t..n {
                let c = *coded.get_unchecked(k);
                let g = c.wrapping_sub(p).wrapping_sub(1);
                p = c;
                *syms.get_unchecked_mut(k) = super::symbol32(g);
                *esc.get_unchecked_mut(e) = g;
                e += (g >= DIRECT as u32) as usize;
            }
        }
    }
    e
}

/// The [`super::GapsFn`] contract with 8-lane vectors (see [`gaps_avx512`]).
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn gaps_avx2(coded: &[u32], syms: &mut [u8], esc: &mut [u32]) -> usize {
    let n = coded.len();
    let (mut t, mut e) = (0usize, 0usize);
    // SAFETY: vectors cover `t + 8 <= n` ids and symbols; escapes are
    // stored one per escaping id; the tail is scalar.
    unsafe {
        let mut last = -1i32;
        let one = _mm256_set1_epi32(1);
        let direct = _mm256_set1_epi32(DIRECT as i32);
        let first = _mm256_set1_epi32(256);
        let bias = _mm256_set1_epi32(DIRECT as i32 - 126);
        let rotate = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
        let narrow = _mm256_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        while t + 8 <= n {
            let c = _mm256_loadu_si256(coded.as_ptr().add(t).cast());
            let shifted = _mm256_blend_epi32::<1>(_mm256_permutevar8x32_epi32(c, rotate), _mm256_set1_epi32(last));
            let g = _mm256_sub_epi32(_mm256_sub_epi32(c, shifted), one);
            last = *coded.get_unchecked(t + 7) as i32;
            // Gaps stay below 2^31, so signed compares are exact.
            let small = _mm256_cmpgt_epi32(direct, g);
            let low = _mm256_cmpgt_epi32(first, g);
            let exp = _mm256_srli_epi32::<23>(_mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_srli_epi32::<8>(g))));
            let s = _mm256_blendv_epi8(_mm256_blendv_epi8(_mm256_add_epi32(exp, bias), direct, low), g, small);
            let b = _mm256_shuffle_epi8(s, narrow);
            let packed = _mm_unpacklo_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256::<1>(b));
            _mm_storel_epi64(syms.as_mut_ptr().add(t).cast(), packed);
            let mut escapes = !_mm256_movemask_ps(_mm256_castsi256_ps(small)) as u32 & 0xff;
            if escapes != 0 {
                let mut lanes = [0u32; 8];
                _mm256_storeu_si256(lanes.as_mut_ptr().cast(), g);
                while escapes != 0 {
                    *esc.get_unchecked_mut(e) = lanes[escapes.trailing_zeros() as usize];
                    e += 1;
                    escapes &= escapes - 1;
                }
            }
            t += 8;
        }
        let mut p = last as u32;
        for k in t..n {
            let c = *coded.get_unchecked(k);
            let g = c.wrapping_sub(p).wrapping_sub(1);
            p = c;
            *syms.get_unchecked_mut(k) = super::symbol32(g);
            *esc.get_unchecked_mut(e) = g;
            e += (g >= DIRECT as u32) as usize;
        }
    }
    e
}
