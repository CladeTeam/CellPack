//! Typed mapping skeletons. Eight lanes cover every numeric width without
//! rounding integers through an intermediate float. Narrow gathers read
//! exactly one element; no dictionary padding is part of this contract.

use std::arch::x86_64::*;
use crate::scans::decode::{Parts, Ranks, Values};
use crate::scans::map::{CastInto, Element, UNMAPPED};

/// These kernels only convert pairs whose vector operation has Rust `as`
/// semantics. Other scans conversions continue through the scalar path.
pub(crate) fn supported<S: Element, T: Element>() -> bool {
    S::KIND != 2 || (T::KIND == 2 && T::WIDTH >= S::WIDTH)
}

#[inline]
fn bytes<T: Element>(v: &[T]) -> &[u8] {
    // SAFETY: numeric elements contain no padding and the borrow is retained.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// Eight values as zero-extended bit patterns. All indices must name live
/// elements. Small elements deliberately avoid overreading a slice's end.
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
#[inline]
unsafe fn gather<const W: usize>(b: &[u8], ix: __m256i, mask: u8) -> __m512i {
    // SAFETY: selected indices name live elements; masked lanes never read
    // memory. For narrow values, wider gathers exclude the last partial word.
    unsafe {
        if W == 8 {
            return _mm512_mask_i32gather_epi64::<8>(_mm512_setzero_si512(), mask, ix, b.as_ptr().cast());
        }
        let ix64 = _mm512_cvtepu32_epi64(ix);
        let safe = if W == 4 { mask } else if b.len() < 4 { 0 } else {
            mask & _mm512_cmple_epu64_mask(ix64, _mm512_set1_epi64(((b.len() - 4) / W) as i64))
        };
        let v = match W {
            1 => _mm512_mask_i64gather_epi32::<1>(_mm256_setzero_si256(), safe, ix64, b.as_ptr().cast()),
            2 => _mm512_mask_i64gather_epi32::<2>(_mm256_setzero_si256(), safe, ix64, b.as_ptr().cast()),
            4 => _mm512_mask_i64gather_epi32::<4>(_mm256_setzero_si256(), safe, ix64, b.as_ptr().cast()),
            _ => std::hint::unreachable_unchecked(),
        };
        let bits = if W == 4 { v } else { _mm256_and_si256(v, _mm256_set1_epi32(if W == 1 { 255 } else { 65535 })) };
        let mut v = _mm512_cvtepu32_epi64(bits);
        let tail = mask & !safe;
        if tail != 0 {
            let mut indices = [0u32; 8];
            let mut values = [0u64; 8];
            _mm256_storeu_si256(indices.as_mut_ptr().cast(), ix);
            for j in 0..8 {
                if tail & (1 << j) != 0 {
                    let p = b.as_ptr().add(indices[j] as usize * W);
                    values[j] = if W == 1 { *p as u64 } else { p.cast::<u16>().read_unaligned() as u64 };
                }
            }
            v = _mm512_or_si512(v, _mm512_loadu_si512(values.as_ptr().cast()));
        }
        v
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
#[inline]
unsafe fn gather_width(b: &[u8], ix: __m256i, width: usize, mask: u8) -> __m512i {
    // SAFETY: the width belongs to a numeric element and indices are valid.
    unsafe { match width { 1 => gather::<1>(b, ix, mask), 2 => gather::<2>(b, ix, mask), 4 => gather::<4>(b, ix, mask), 8 => gather::<8>(b, ix, mask), _ => std::hint::unreachable_unchecked() } }
}

/// Eight lanes from `S` to `T`: sign-extend signed sources narrower than 8
/// bytes, then convert to a float target, which takes integers of either
/// signedness or an `f32` widened to `f64`. Same-type pairs keep their bits.
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
#[inline]
unsafe fn convert<S: Element, T: Element>(mut v: __m512i) -> __m512i {
    if S::KIND == 1 && S::WIDTH < 8 {
        let shift = _mm512_set1_epi64((64 - 8 * S::WIDTH) as i64);
        v = _mm512_srav_epi64(_mm512_sllv_epi64(v, shift), shift);
    }
    if T::KIND != 2 || (S::KIND == 2 && S::WIDTH == T::WIDTH) {
        return v;
    }
    if T::WIDTH == 8 {
        let f = if S::KIND == 2 {
            _mm512_cvtps_pd(_mm256_castsi256_ps(_mm512_cvtepi64_epi32(v)))
        } else if S::KIND == 1 {
            _mm512_cvtepi64_pd(v)
        } else {
            _mm512_cvtepu64_pd(v)
        };
        _mm512_castpd_si512(f)
    } else {
        let f = if S::KIND == 1 { _mm512_cvtepi64_ps(v) } else { _mm512_cvtepu64_ps(v) };
        _mm512_cvtepu32_epi64(_mm256_castps_si256(f))
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
#[inline]
unsafe fn values<S: Element, T: Element>(parts: &Parts<'_>, ix: __m256i, ids: __m256i, dict: &[T], modes: &[T], contiguous: bool, at: usize) -> __m512i {
    // SAFETY: decoded Parts validate ranks; entry points validate ids and
    // positions, and dictionaries and modes have the corresponding lengths.
    unsafe {
        match parts.values {
            Values::Raw(b) => {
                let v = if contiguous {
                    let p = b.as_ptr().add(at * S::WIDTH);
                    match S::WIDTH {
                        1 => _mm512_cvtepu8_epi64(_mm_loadl_epi64(p.cast())),
                        2 => _mm512_cvtepu16_epi64(_mm_loadu_si128(p.cast())),
                        4 => _mm512_cvtepu32_epi64(_mm256_loadu_si256(p.cast())),
                        8 => _mm512_loadu_si512(p.cast()),
                        _ => std::hint::unreachable_unchecked(),
                    }
                } else { gather_width(b, ix, S::WIDTH, 255) };
                convert::<S,T>(v)
            }
            Values::Dict(d) => {
                let r = match d.ranks() {
                    Ranks::One => {
                        let mut word = [0u8; 8];
                        std::ptr::copy_nonoverlapping(dict.as_ptr().cast::<u8>(), word.as_mut_ptr(), T::WIDTH);
                        return _mm512_set1_epi64(u64::from_le_bytes(word) as i64);
                    }
                    Ranks::U8(r) => _mm512_cvtepi64_epi32(gather::<1>(r, ix, 255)),
                    Ranks::U32(r) => _mm256_i32gather_epi32::<4>(r.as_ptr().cast(), ix),
                };
                gather_width(bytes(dict), r, T::WIDTH, 255)
            }
            Values::Modes(d) => {
                let r = match d.ranks() {
                    Ranks::One => return gather_width(bytes(modes), ids, T::WIDTH, 255),
                    Ranks::U8(r) => _mm512_cvtepi64_epi32(gather::<1>(r, ix, 255)),
                    Ranks::U32(r) => _mm256_i32gather_epi32::<4>(r.as_ptr().cast(), ix),
                };
                let nonzero = _mm512_cmpneq_epi64_mask(_mm512_cvtepu32_epi64(r), _mm512_setzero_si512());
                let exceptions = gather_width(bytes(dict), _mm256_sub_epi32(r, _mm256_set1_epi32(1)), T::WIDTH, nonzero);
                let defaults = gather_width(bytes(modes), ids, T::WIDTH, !nonzero);
                _mm512_or_si512(exceptions, defaults)
            }
        }
    }
}

/// Value of element `t` converted to `T`.
///
/// # Safety
/// The mapping contract: `t` is an element of the row; ranks index `dict`
/// (mode ranks at most its length) and ids index `modes`.
#[inline(always)]
pub(crate) unsafe fn scalar<S: CastInto<T>, T: Element>(parts: &Parts<'_>, t: usize, dict: &[T], modes: &[T]) -> T {
    use crate::unchecked::get;
    // SAFETY: as documented.
    unsafe {
        match parts.values {
            Values::Raw(b) => S::read_le(b.as_ptr().add(t * S::WIDTH)).cast(),
            Values::Dict(d) => get(dict, match d.ranks() { Ranks::One => 0, Ranks::U8(r) => get(r, t) as usize, Ranks::U32(r) => get(r, t) as usize }),
            Values::Modes(d) => {
                let r = match d.ranks() { Ranks::One => 0, Ranks::U8(r) => get(r, t) as usize, Ranks::U32(r) => get(r, t) as usize };
                if r == 0 { get(modes, get(parts.ids, t) as usize) } else { get(dict, r - 1) }
            }
        }
    }
}

/// Stores exactly eight numeric elements, including unaligned byte outputs.
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
#[inline]
unsafe fn store<T: Element>(out: *mut T, v: __m512i) {
    // SAFETY: caller provides eight writable elements; unaligned stores
    // avoid adding a stronger alignment requirement than the numeric type.
    unsafe {
        match T::WIDTH {
            8 => _mm512_storeu_si512(out.cast(), v),
            4 => _mm256_storeu_si256(out.cast(), _mm512_cvtepi64_epi32(v)),
            2 => _mm_storeu_si128(out.cast(), _mm512_cvtepi64_epi16(v)),
            1 => _mm_storel_epi64(out.cast(), _mm512_cvtepi64_epi8(v)),
            _ => std::hint::unreachable_unchecked(),
        }
    }
}

/// SHAPE: 0 compact CSR, 1 indexed CSR, 2 dense scatter. All Parts and
/// buffer bounds are checked by the safe mapping entry points.
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
pub(crate) unsafe fn map<S: CastInto<T>, T: Element, const SHAPE: u8>(parts: &Parts<'_>, positions: &[u32], panel: &[u32], dict: &[T], modes: &[T], cols: &mut [u32], out: &mut [T]) -> Option<usize> {
    let n = if SHAPE == 1 { positions.len() } else { parts.ids.len() };
    let mut at = 0;
    let mut kept = 0;
    // SAFETY: entry-point contracts cover every load, index and output. A
    // dense target is checked before stores, and tails never read a vector.
    unsafe {
        while at + 8 <= n {
            let ix = if SHAPE == 1 { _mm256_loadu_si256(positions.as_ptr().add(at).cast()) } else { _mm256_add_epi32(_mm256_set1_epi32(at as i32), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)) };
            let ids = if SHAPE == 1 { _mm256_i32gather_epi32::<4>(parts.ids.as_ptr().cast(), ix) } else { _mm256_loadu_si256(parts.ids.as_ptr().add(at).cast()) };
            let c = _mm256_i32gather_epi32::<4>(panel.as_ptr().cast(), ids);
            let c64 = _mm512_cvtepu32_epi64(c);
            let keep = _mm512_cmpneq_epi64_mask(c64, _mm512_set1_epi64(UNMAPPED as i64));
            let v = values::<S, T>(parts, ix, ids, dict, modes, SHAPE != 1, at);
            if SHAPE == 0 {
                let count = keep.count_ones() as usize;
                _mm512_mask_compressstoreu_epi32(cols.as_mut_ptr().add(kept).cast(), keep as u16, _mm512_castsi256_si512(c));
                // Full stores are within the n-element output even when
                // only a subset of these lanes is retained.
                store(out.as_mut_ptr().add(kept), _mm512_maskz_compress_epi64(keep, v));
                kept += count;
            } else if SHAPE == 1 {
                _mm256_storeu_si256(cols.as_mut_ptr().add(at).cast(), c);
                store(out.as_mut_ptr().add(at), v);
            } else {
                if _mm512_mask_cmpge_epu64_mask(keep, c64, _mm512_set1_epi64(out.len() as i64)) != 0 { return None; }
                if T::WIDTH == 8 {
                    _mm512_mask_i32scatter_epi64::<8>(out.as_mut_ptr().cast(), keep, c, v);
                } else if T::WIDTH == 4 {
                    _mm512_mask_i64scatter_epi32::<4>(out.as_mut_ptr().cast(), keep, c64, _mm512_cvtepi64_epi32(v));
                } else {
                    let mut columns = [0u32; 8];
                    let mut vals = [0u64; 8];
                    _mm256_storeu_si256(columns.as_mut_ptr().cast(), c);
                    _mm512_storeu_si512(vals.as_mut_ptr().cast(), v);
                    for j in 0..8 {
                        // Mapped columns of this vector were checked against
                        // the row above.
                        if columns[j] != UNMAPPED { crate::unchecked::set(out, columns[j] as usize, T::from_le(&vals[j].to_le_bytes())); }
                    }
                }
            }
            at += 8;
        }
    }
    // SAFETY: the tail keeps the entry-point contracts: positions lie below
    // the row length, ids index the panel, compact outputs hold every
    // element (`kept <= i`), indexed outputs one per position, and a dense
    // store's column is checked against the row first.
    unsafe {
        use crate::unchecked::{get, set};
        for i in at..n {
            let p = if SHAPE == 1 { get(positions, i) as usize } else { i };
            let c = get(panel, get(parts.ids, p) as usize);
            let v = scalar::<S, T>(parts, p, dict, modes);
            if SHAPE == 0 {
                set(cols, kept, c);
                set(out, kept, v);
                kept += usize::from(c != UNMAPPED);
            } else if SHAPE == 1 {
                set(cols, i, c);
                set(out, i, v);
            } else if c != UNMAPPED {
                if c as usize >= out.len() {
                    return None;
                }
                set(out, c as usize, v);
            }
        }
    }
    Some(if SHAPE == 0 { kept } else { n })
}
