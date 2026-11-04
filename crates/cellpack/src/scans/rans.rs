//! Static rANS with interleaved states sharing one word stream.
//!
//! A stream of `n` symbols uses `lanes` states; symbol `j` belongs to state
//! `j % lanes`. The state lives in `[L, L << 16)`, so each symbol emits or
//! consumes at most one 16-bit word. Probabilities have 11 bits: an 8 KiB
//! slot table per model stays in L1 next to the others, at a negligible
//! cost in ratio. Words of all states are interleaved in
//! decode order in a single stream, so a SIMD decoder can refill a vector of
//! states with one load and the scalar decoder reads the same bytes.
//!
//! Stream layout: `lanes` final states (u32 LE), then the words (u16 LE).
//!
//! A state starting at `L` would spend 16 bits of every stored final state
//! on nothing. Instead each lane starts at `L + p` with a 16-bit payload
//! `p`: bytes the caller only needs after decoding (escape bits, another
//! stream), which therefore cost nothing extra. The decoder ends in exactly
//! the starting states and recovers the payload from them.
//!
//! Streams are trusted: they come from the encoder for this chunk, and the
//! decoders index without bounds checks under that assumption (a damaged
//! stream is undefined behavior). Debug builds assert exact consumption.

use crate::scans::kernels::Kernels;
use crate::unchecked::{get, set};

/// Probability precision: frequencies sum to [`M`], so a slot table is 8 KiB.
pub(crate) const SCALE: u32 = 11;
/// Total frequency of a model; a state's slot is its low `SCALE` bits.
pub(crate) const M: u32 = 1 << SCALE;
/// Lower bound of a state; a step whose result drops below it reads a word.
pub(crate) const L: u32 = 1 << 16;

/// Slot table of one model: `(freq - 1) | cum << 12 | symbol << 24` per
/// slot (12-bit fields hold the 11-bit values).
pub(crate) type Slots = [u32; M as usize];

/// Lanes of a stream of `n` symbols: the largest power of two of at most
/// `max_lanes` whose lanes hold at least `lane_min` symbols on average, and
/// one lane at least, so short streams do not pay for states that code
/// almost nothing.
#[inline]
pub(crate) fn lanes_for(n: usize, max_lanes: u32, lane_min: u32) -> usize {
    let mut l = 1usize;
    while (l as u32) < max_lanes && (2 * l as u64) * lane_min as u64 <= n as u64 {
        l *= 2;
    }
    l
}

/// Upper bound on the encoded size of `n` symbols with `lanes` states.
pub(crate) fn stream_bound(n: usize, lanes: usize) -> usize {
    4 * lanes + 2 * n
}

/// Payload bytes the initial states of `lanes` lanes can carry.
#[inline]
pub(crate) fn payload_capacity(lanes: usize) -> usize {
    2 * lanes
}

/// Right shift of a state that leaves the quotient compared against a
/// frequency to decide whether a word is emitted: a state `x` emits before
/// coding a symbol of frequency `f` when `x >= ((L >> SCALE) << 16) * f`.
pub(crate) const EMIT_SHIFT: u32 = L.trailing_zeros() - SCALE + 16;

/// `u16` entries of encoder word scratch beyond the symbol count: vector
/// encoders store whole vectors below the words they keep.
pub(crate) const WORD_SLACK: usize = 16;

/// Encoder constants of one model, per symbol.
#[derive(Clone, Debug)]
pub(crate) struct EncModel {
    /// `freq | cum << 16`; zero for an unused symbol.
    pub(crate) fc: [u32; 256],
    /// `u64::MAX / freq + 1`, so the quotient of a state below `2^32` is
    /// the high half of its product with this (exact for `2 <= freq <=
    /// M`); zero for frequency 1, whose quotient is the state.
    pub(crate) rcp: [u64; 256],
    /// `1 / freq` rounded to nearest; `floor(x * inv + 2^-15)` is the exact
    /// quotient of a state below `2^32` (vector encoders).
    pub(crate) inv: [f64; 256],
}

impl EncModel {
    /// Constants of the model with these frequencies (see [`EncModel`]).
    pub(crate) fn new(freq: &[u16; 256]) -> Self {
        let mut m = EncModel {
            fc: [0; 256],
            rcp: [0; 256],
            inv: [0.0; 256],
        };
        let mut cum = 0u32;
        for (s, &f) in freq.iter().enumerate() {
            let f = f as u32;
            if f > 0 {
                m.fc[s] = f | cum << 16;
                m.rcp[s] = if f == 1 { 0 } else { u64::MAX / f as u64 + 1 };
                m.inv[s] = 1.0 / f as f64;
            }
            cum += f;
        }
        m
    }
}

/// The [`crate::scans::kernels::RansEncodeFn`] contract, portably: codes
/// `syms` (symbol `j` with state `x[j % lanes]`) from the last symbol to
/// the first, storing the emitted words backwards so they end at
/// `words.len()` in stream order. Returns the word count, or `None` when a
/// symbol has zero frequency.
///
/// # Safety
/// `lanes` is a power of two up to 32 and `words` holds at least
/// `syms.len() + WORD_SLACK` entries.
pub(crate) unsafe fn encode_scalar(m: &EncModel, syms: &[u8], lanes: usize, x: &mut [u32; 32], words: &mut [u16]) -> Option<usize> {
    debug_assert!(lanes.is_power_of_two() && lanes <= 32 && words.len() >= syms.len() + WORD_SLACK);
    let mut pos = words.len();
    // SAFETY: `j < syms.len()`; `lane < lanes <= 32`; every symbol stores
    // at `pos - 1` and keeps at most one word, so `pos - 1` stays at or
    // above `words.len() - syms.len() - 1 >= 0`.
    unsafe {
        for j in (0..syms.len()).rev() {
            let s = get(syms, j) as usize;
            let fc = get(&m.fc, s);
            let f = fc & 0xffff;
            if f == 0 {
                return None;
            }
            let lane = j & (lanes - 1);
            let mut v = get(x, lane);
            // The word is always stored and kept only when emitted, which
            // avoids a data-dependent branch.
            let emit = ((v >> EMIT_SHIFT) >= f) as usize;
            set(words, pos - 1, v as u16);
            pos -= emit;
            v >>= 16 * emit as u32;
            let q = if f == 1 { v } else { ((v as u128 * get(&m.rcp, s) as u128) >> 64) as u32 };
            set(x, lane, (q << SCALE) + (v - q * f) + (fc >> 16));
        }
    }
    Some(words.len() - pos)
}

/// Encodes `syms` into `out` with the first `payload_capacity(lanes)` bytes
/// of `payload` in the initial states, using `words` (at least
/// `syms.len() + WORD_SLACK` long) as scratch and the encoder of `k`.
/// Returns the stream length, or `None` when a symbol has zero frequency
/// or a buffer is too small.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    k: &Kernels,
    m: &EncModel,
    syms: &[u8],
    lanes: usize,
    payload: &[u8],
    words: &mut [u16],
    out: &mut [u8],
) -> Option<usize> {
    let n = syms.len();
    if words.len() < n + WORD_SLACK || out.len() < stream_bound(n, lanes) {
        return None;
    }
    debug_assert!(lanes.is_power_of_two() && lanes <= 32);
    let mut x = [L; 32];
    for (j, v) in x[..lanes].iter_mut().enumerate() {
        let lo = payload.get(2 * j).copied().unwrap_or(0) as u32;
        let hi = payload.get(2 * j + 1).copied().unwrap_or(0) as u32;
        *v = L + (lo | hi << 8);
    }
    // SAFETY: lane counts are powers of two up to 32, the word scratch was
    // checked above, and the kernel set's ISA is supported.
    let nw = unsafe { (k.rans_encode)(m, syms, lanes, &mut x, words) }?;
    let head = 4 * lanes;
    // SAFETY: the output holds `stream_bound(n, lanes)` bytes (checked
    // above): the states and at most `n` words.
    unsafe {
        let o = out.as_mut_ptr();
        for (i, v) in x[..lanes].iter().enumerate() {
            o.add(4 * i).cast::<[u8; 4]>().write_unaligned(v.to_le_bytes());
        }
        for (i, w) in words.get_unchecked(words.len() - nw..).iter().enumerate() {
            o.add(head + 2 * i).cast::<[u8; 2]>().write_unaligned(w.to_le_bytes());
        }
    }
    Some(head + 2 * nw)
}

/// A stream of a run coded with [`encode_many`]: its initial states, its
/// word scratch and, after the run, its word count (`None` when a symbol
/// has zero frequency).
pub(crate) struct Job<'a> {
    /// The stream's model.
    pub(crate) model: &'a EncModel,
    /// One symbol per element.
    pub(crate) syms: &'a [u8],
    /// The stream's initial states, replaced in place by its final states.
    pub(crate) x: [u32; 32],
    /// Word scratch; emitted words end at its end, in stream order.
    pub(crate) words: &'a mut [u16],
    /// Emitted words after the run; `None` when a symbol has zero frequency.
    pub(crate) nw: Option<usize>,
}

/// The [`crate::scans::kernels::RansManyFn`] contract, portably: each job
/// coded on its own.
///
/// # Safety
/// As [`encode_scalar`] for every job.
pub(crate) unsafe fn many_scalar(jobs: &mut [Job], lanes: usize) {
    for j in jobs {
        // SAFETY: as documented.
        j.nw = unsafe { encode_scalar(j.model, j.syms, lanes, &mut j.x, j.words) };
    }
}

/// A stream to code with [`encode_many`].
pub(crate) struct Stream<'a> {
    /// Its model.
    pub(crate) model: &'a EncModel,
    /// One symbol per element.
    pub(crate) syms: &'a [u8],
    /// Bytes the initial states carry, at most [`payload_capacity`] of them.
    pub(crate) payload: &'a [u8],
    /// Word scratch of `syms.len() + WORD_SLACK` entries.
    pub(crate) words: &'a mut [u16],
    /// Room for [`stream_bound`] bytes of stream.
    pub(crate) out: &'a mut [u8],
}

/// Codes up to three streams of one symbol count with `lanes` states each,
/// interleaved so their dependency chains overlap. Each result, at its
/// stream's position, is that of [`encode`] on the stream; a stream of
/// another symbol count or one whose buffers are too small is skipped.
pub(crate) fn encode_many(k: &Kernels, lanes: usize, streams: [Option<Stream>; 3]) -> [Option<usize>; 3] {
    debug_assert!(lanes.is_power_of_two() && lanes <= 32);
    let mut results = [None; 3];
    let Some(first) = streams.iter().flatten().next() else { return results };
    let (n, model) = (first.syms.len(), first.model);
    let mut jobs: [Job; 3] = std::array::from_fn(|_| Job { model, syms: &[], x: [L; 32], words: &mut [], nw: None });
    let mut outs: [&mut [u8]; 3] = [&mut [], &mut [], &mut []];
    let mut index = [0usize; 3];
    let mut count = 0;
    for (i, s) in streams.into_iter().enumerate() {
        let Some(Stream { model, syms, payload, words, out }) = s else { continue };
        if syms.len() != n || words.len() < n + WORD_SLACK || out.len() < stream_bound(n, lanes) {
            continue;
        }
        let mut x = [L; 32];
        for (j, v) in x[..lanes].iter_mut().enumerate() {
            let lo = payload.get(2 * j).copied().unwrap_or(0) as u32;
            let hi = payload.get(2 * j + 1).copied().unwrap_or(0) as u32;
            *v = L + (lo | hi << 8);
        }
        jobs[count] = Job { model, syms, x, words, nw: None };
        outs[count] = out;
        index[count] = i;
        count += 1;
    }
    // SAFETY: lane counts are powers of two up to 32, every job's symbols
    // number `n` with the word scratch checked above, and the kernel set's
    // ISA is supported.
    unsafe { (k.rans_many)(&mut jobs[..count], lanes) };
    let head = 4 * lanes;
    for c in 0..count {
        let job = &jobs[c];
        let Some(nw) = job.nw else { continue };
        let out = &mut *outs[c];
        for (o, v) in out.chunks_exact_mut(4).zip(&job.x[..lanes]) {
            o.copy_from_slice(&v.to_le_bytes());
        }
        let words = &job.words[job.words.len() - nw..];
        for (o, w) in out[head..head + 2 * nw].chunks_exact_mut(2).zip(words) {
            o.copy_from_slice(&w.to_le_bytes());
        }
        results[index[c]] = Some(head + 2 * nw);
    }
    results
}

/// One decode step of one state. `last` is the stream length minus 2.
///
/// The refill word is read on every step without branching on the state;
/// its position is clamped to `last`, so a step that needs no word (such
/// as the final one, at the end of the stream) reads an earlier word and
/// discards it. A step that needs a word reads it at `pos`, which a
/// well-formed stream keeps at most `last`.
#[inline(always)]
fn step(x: &mut u32, slots: &Slots, stream: &[u8], last: usize, pos: &mut usize) -> u8 {
    let slot = *x & (M - 1);
    let e = slots[slot as usize];
    let (f, c) = ((e & 0xfff) + 1, (e >> 12) & 0xfff);
    let y = f
        .wrapping_mul(*x >> SCALE)
        .wrapping_add(slot)
        .wrapping_sub(c);
    debug_assert!(last + 2 <= stream.len());
    // SAFETY: `min(pos, last) + 2 <= stream.len()`.
    let w = u16::from_le_bytes(unsafe { stream.as_ptr().add((*pos).min(last)).cast::<[u8; 2]>().read_unaligned() }) as u32;
    let need = (y < L) as u32;
    debug_assert!(need == 0 || *pos <= last, "rANS refill past the stream");
    *x = (y << (need * 16)) | (w & need.wrapping_neg());
    *pos += 2 * need as usize;
    (e >> 24) as u8
}

/// Decodes symbols `start..` (`start` a multiple of `N`) with the states in `x`.
#[inline(always)]
pub(crate) fn run_scalar<const N: usize>(
    slots: &Slots,
    x: &mut [u32; N],
    stream: &[u8],
    pos: &mut usize,
    out: &mut [u8],
    start: usize,
) {
    let n = out.len();
    let last = stream.len() - 2;
    let mut j = start;
    // SAFETY: groups cover `out[j..j + N]` with `j + N <= n`; the tail
    // covers `out[j..n]` with `n - j < N` lanes.
    unsafe {
        while j + N <= n {
            for (lane, xl) in x.iter_mut().enumerate() {
                set(out, j + lane, step(xl, slots, stream, last, pos));
            }
            j += N;
        }
        for lane in 0..n - j {
            set(out, j + lane, step(x.get_unchecked_mut(lane), slots, stream, last, pos));
        }
    }
}

/// Reads the `N` initial states of a stream (which holds at least them).
#[inline(always)]
pub(crate) fn states<const N: usize>(stream: &[u8]) -> [u32; N] {
    debug_assert!(stream.len() >= 4 * N);
    // SAFETY: a stream starts with its `N` 4-byte states.
    std::array::from_fn(|i| u32::from_le_bytes(unsafe { stream.as_ptr().add(4 * i).cast::<[u8; 4]>().read_unaligned() }))
}

/// Hands out the final states (`ends` holds at least `N`).
#[inline(always)]
pub(crate) fn finished<const N: usize>(x: &[u32; N], ends: &mut [u32], pos: usize, len: usize) -> bool {
    debug_assert!(ends.len() >= N);
    debug_assert_eq!(pos, len, "rANS stream not consumed exactly");
    // SAFETY: `ends` holds at least `N` states.
    unsafe { ends.get_unchecked_mut(..N) }.copy_from_slice(x);
    true
}

/// Recovers `b` payload bytes from the final states of a stream's lanes into
/// `out` (at least `b` bytes). Debug builds check that every state is `L`
/// plus a 16-bit value and that payload beyond `b` bytes is zero.
#[inline]
pub(crate) fn payload(ends: &[u32], b: usize, out: &mut [u8]) -> bool {
    debug_assert!(out.len() >= b);
    for (j, &x) in ends.iter().enumerate() {
        let p = x.wrapping_sub(L);
        debug_assert!(x >= L && p <= 0xffff);
        let (lo, hi) = (2 * j, 2 * j + 1);
        // SAFETY: stores stay below `b <= out.len()`.
        unsafe {
            if hi < b {
                set(out, lo, p as u8);
                set(out, hi, (p >> 8) as u8);
            } else if lo < b {
                set(out, lo, p as u8);
                debug_assert_eq!(p >> 8, 0);
            } else {
                debug_assert_eq!(p, 0);
            }
        }
    }
    true
}

/// Portable decoder for `N` lanes: fills `out` and `ends` (the final
/// states) and reports exact consumption.
pub(crate) fn decode_scalar<const N: usize>(
    slots: &Slots,
    stream: &[u8],
    out: &mut [u8],
    ends: &mut [u32],
) -> bool {
    let mut x = states::<N>(stream);
    let mut pos = 4 * N;
    run_scalar::<N>(slots, &mut x, stream, &mut pos, out, 0);
    finished(&x, ends, pos, stream.len())
}

/// Decodes symbols `start..` of two equal-length streams, interleaving their
/// steps so the two dependency chains overlap.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pair<const N: usize>(
    sa: &Slots,
    xa: &mut [u32; N],
    a: &[u8],
    pa: &mut usize,
    oa: &mut [u8],
    sb: &Slots,
    xb: &mut [u32; N],
    b: &[u8],
    pb: &mut usize,
    ob: &mut [u8],
    start: usize,
) {
    let n = oa.len().min(ob.len());
    let (la, lb) = (a.len() - 2, b.len() - 2);
    let mut j = start;
    // SAFETY: as in `run_scalar`, for both outputs (each at least `n` long).
    unsafe {
        while j + N <= n {
            for lane in 0..N {
                set(oa, j + lane, step(&mut xa[lane], sa, a, la, pa));
                set(ob, j + lane, step(&mut xb[lane], sb, b, lb, pb));
            }
            j += N;
        }
        for lane in 0..n - j {
            set(oa, j + lane, step(xa.get_unchecked_mut(lane), sa, a, la, pa));
            set(ob, j + lane, step(xb.get_unchecked_mut(lane), sb, b, lb, pb));
        }
    }
}

/// Portable decoder of two streams of `out.len()` symbols each with `N`
/// lanes, as [`decode_scalar`] for both.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_pair_scalar<const N: usize>(
    sa: &Slots,
    a: &[u8],
    oa: &mut [u8],
    sb: &Slots,
    b: &[u8],
    ob: &mut [u8],
    ea: &mut [u32],
    eb: &mut [u32],
) -> bool {
    let (mut xa, mut xb) = (states::<N>(a), states::<N>(b));
    debug_assert_eq!(oa.len(), ob.len());
    let (mut pa, mut pb) = (4 * N, 4 * N);
    run_pair::<N>(sa, &mut xa, a, &mut pa, oa, sb, &mut xb, b, &mut pb, ob, 0);
    finished(&xa, ea, pa, a.len()) & finished(&xb, eb, pb, b.len())
}
