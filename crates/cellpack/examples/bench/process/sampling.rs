//! Example process functions used by the real-data benchmark.
use super::{CsrRow, DenseRow, Rng, SAMPLE};

/// The `sample` schema's candidates, its `user_data`: dataset d holds the
/// global cells `cells[d]..cells[d + 1]`, and its candidates are the standard
/// ids `genes[offsets[d]..offsets[d + 1]]`, its columns with an output dim in
/// column order. Standard ids must be below the schema's `out_dim`. They are
/// `id_bytes` wide: 4 (u32), or 2 (u16) when every id fits, which halves the
/// table a cell's sample gathers from.
#[repr(C)]
pub struct SampleTable {
    pub datasets: u64,
    pub cells: *const u64,
    pub offsets: *const u64,
    pub genes: *const u8,
    pub id_bytes: u64,
}

/// An owned [`SampleTable`]: build it, keep it alive with the schema's owner
/// and pass [`Candidates::table`] as `user_data`.
#[allow(dead_code)]
pub(super) struct Candidates {
    cells: Vec<u64>,
    offsets: Vec<u64>,
    genes: Vec<u32>,
    narrow: Vec<u16>,
    table: SampleTable,
}

// SAFETY: the table points into the struct's own vectors, never written after
// construction.
unsafe impl Send for Candidates {}
// SAFETY: as above; the data is read-only.
unsafe impl Sync for Candidates {}

#[allow(dead_code)]
impl Candidates {
    /// From each dataset's cell count and column -> standard id map
    /// (`unmapped` marks columns without an output dim).
    pub(super) fn new<'a>(datasets: impl IntoIterator<Item = (u64, &'a [u32])>, unmapped: u32) -> Self {
        let (mut cells, mut offsets, mut genes) = (vec![0u64], vec![0u64], Vec::new());
        for (n, map) in datasets {
            genes.extend(map.iter().copied().filter(|&s| s != unmapped));
            cells.push(cells.last().unwrap() + n);
            offsets.push(genes.len() as u64);
        }
        let narrow: Vec<u16> = if genes.iter().all(|&g| g <= u16::MAX as u32) { genes.iter().map(|&g| g as u16).collect() } else { Vec::new() };
        let (ptr, id_bytes) = if narrow.is_empty() && !genes.is_empty() { (genes.as_ptr().cast(), 4) } else { (narrow.as_ptr().cast(), 2) };
        let table = SampleTable { datasets: cells.len() as u64 - 1, cells: cells.as_ptr(), offsets: offsets.as_ptr(), genes: ptr, id_bytes };
        Self { cells, offsets, genes, narrow, table }
    }
    pub(super) fn table(&self) -> *const SampleTable {
        &self.table
    }
    /// The most candidates of one dataset.
    pub(super) fn max_genes(&self) -> usize {
        self.offsets.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0)
    }
}

/// scGPT-style sample of one cell over the gene vocabulary, from CSR rows
/// ([`cellpack::loader::RowFormat::Csr`]; dense records arrive as their
/// non-zeros). The candidates are the m genes of the cell's dataset (its
/// [`SampleTable`] entry), zeros included: CSR drops zeros, so a candidate the
/// row does not store is a zero expression value; only genes the dataset
/// lacks are missing, and they are never candidates. min(m, 2048) of them are
/// chosen without replacement, seeded by the global cell id, with Floyd's
/// algorithm over positions in [0, m) (splitmix64 seeded by `cell ^
/// 0x5ca1ab1e`, two 32-bit draws per output, each scaled by a multiply and
/// shift) and an m-bit membership map:
///
/// - m <= 2048: every candidate, in column order;
/// - 2048 < m < 4096: Floyd draws the m - 2048 dropped positions; the kept
///   ones are written in column order;
/// - m >= 4096: Floyd draws the 2048 kept positions, written in the order
///   drawn.
///
/// Row layout: `[u32; 2048]` standard ids then `[f32; 2048]` values, padded
/// with id u32::MAX and value 0. The choice depends only on m and the cell id,
/// so it matches the loader benchmark's `outputs.py` element by element; the
/// AVX-512 and scalar paths agree. A cell costs O(m / 64 + 2048 + stored
/// elements): the chosen genes get a bit map over the output dims and a slot
/// table, and each stored element tests the bit map.
pub(super) unsafe extern "C" fn sample(call: *const CsrRow) -> i32 {
    // SAFETY: the loader passes a valid row of f32 values and the schema's
    // user_data is a SampleTable; scratch is laid out by `Scratch`.
    unsafe {
        let c = &*call;
        let table = &*c.context.user_data.cast::<SampleTable>();
        let cells = std::slice::from_raw_parts(table.cells, table.datasets as usize + 1);
        let cell = c.context.cell;
        let d = cells.partition_point(|&start| start <= cell);
        if d == 0 || d > table.datasets as usize {
            return 1;
        }
        let (a, b) = (*table.offsets.add(d - 1) as usize, *table.offsets.add(d) as usize);
        let wide = table.id_bytes != 2;
        let genes = table.genes.add(a * if wide { 4 } else { 2 });
        let m = b - a;
        let f = c.out_dim as usize;
        let s = Scratch::new(c.scratch, f, m);
        let k = select(m, cell, s.words, s.draws, s.pos, genes, if wide { 4 } else { 2 });
        // The chosen genes: ids in output order, a bit map over the output
        // dims and each one's output slot.
        let ids = s.stage.cast::<u32>();
        let vals = s.stage.add(SAMPLE * 4).cast::<f32>();
        // The id gathers go first, so their cache misses overlap; the slot
        // table (L2 resident) is prefetched for writing a few genes ahead.
        if wide {
            for j in 0..k {
                *ids.add(j) = *genes.cast::<u32>().add(*s.pos.add(j) as usize);
            }
        } else {
            for j in 0..k {
                *ids.add(j) = *genes.cast::<u16>().add(*s.pos.add(j) as usize) as u32;
            }
        }
        std::ptr::write_bytes(s.chosen, 0, f / 64 + 1);
        for j in 0..k {
            if j + 16 < k {
                prefetch_write(s.slot.add(*ids.add(j + 16) as usize).cast());
            }
            let g = *ids.add(j);
            *s.slot.add(g as usize) = j as u16;
            *s.chosen.add(g as usize >> 6) |= 1u64 << (g & 63);
        }
        std::ptr::write_bytes(ids.add(k), 0xff, SAMPLE - k);
        std::ptr::write_bytes(vals, 0, SAMPLE);
        let n = c.nnz as usize;
        let columns = c.columns;
        let values = c.values.cast::<f32>();
        if std::arch::is_x86_feature_detected!("avx512f") {
            place_avx512(columns, values, n, s.chosen, s.slot, vals);
        } else {
            for e in 0..n {
                let col = *columns.add(e) as usize;
                if (*s.chosen.add(col >> 6) >> (col & 63)) & 1 != 0 {
                    *vals.add(*s.slot.add(col) as usize) = *values.add(e);
                }
            }
        }
        // The row is assembled in scratch (cache resident) and streamed to the
        // ring with non-temporal stores: the ring is far larger than cache,
        // and ordinary stores would first read every line they overwrite.
        if std::arch::is_x86_feature_detected!("avx512f") && (c.out as usize).is_multiple_of(64) {
            stream_avx512(s.stage, c.out, SAMPLE * 8);
        } else {
            std::ptr::copy_nonoverlapping(s.stage, c.out, SAMPLE * 8);
        }
    }
    0
}

/// Scratch bytes of [`sample`] for `out_dim` output dims and datasets of at
/// most `max_genes` candidates.
pub(super) const fn sample_scratch_bytes(out_dim: usize, max_genes: usize) -> usize {
    Scratch::bytes(out_dim, max_genes)
}

/// The scratch areas of one [`sample`] call, each 64-byte aligned relative to
/// the scratch start: the staged output row, the chosen dims' bit map
/// (out_dim bits) and slot table (u16 per dim), Floyd's membership map
/// (m bits), its draws and the chosen positions.
struct Scratch {
    stage: *mut u8,
    chosen: *mut u64,
    slot: *mut u16,
    words: *mut u64,
    draws: *mut u32,
    pos: *mut u32,
}

impl Scratch {
    const fn sizes(out_dim: usize, max_genes: usize) -> [usize; 6] {
        // The slot table has two bytes of slack for 32-bit gathers.
        [SAMPLE * 8, (out_dim / 64 + 1) * 8, out_dim * 2 + 2, (max_genes / 64 + 2) * 8, (SAMPLE + 16) * 4, (SAMPLE + 16) * 4]
    }
    const fn bytes(out_dim: usize, max_genes: usize) -> usize {
        let sizes = Self::sizes(out_dim, max_genes);
        let mut total = 64; // room to align the start
        let mut i = 0;
        while i < sizes.len() {
            total += sizes[i].next_multiple_of(64);
            i += 1;
        }
        total
    }
    /// # Safety
    /// `base` holds `bytes(out_dim, max_genes)` with `m <= max_genes`.
    unsafe fn new(base: *mut u8, out_dim: usize, m: usize) -> Self {
        let sizes = Self::sizes(out_dim, m);
        // SAFETY: every area lies inside the reserved scratch (see bytes; the
        // m-bit map is no larger than the max_genes one it was sized for).
        unsafe {
            let mut at = base.add(base.align_offset(64));
            let mut next = |size: usize| {
                let p = at;
                at = at.add(size.next_multiple_of(64));
                p
            };
            let stage = next(sizes[0]);
            let chosen = next(sizes[1]).cast();
            let slot = next(sizes[2]).cast();
            Self { stage, chosen, slot, words: next(sizes[3]).cast(), draws: next(sizes[4]).cast(), pos: next(sizes[5]).cast() }
        }
    }
}

/// Writes the positions chosen among `m` for `cell`, in output order (see
/// [`sample`]), to `pos`; returns their count, min(m, 2048). `words` holds
/// m / 64 + 2 words, `draws` and `pos` 2048 + 16 entries. While Floyd's
/// loop runs (bound by its own latency), the candidates' ids at the draws
/// ahead are prefetched: a draw is almost always the position kept, and the
/// id table of a random dataset is rarely cached.
///
/// # Safety
/// The buffers are as documented and disjoint; `genes` holds m ids of
/// `id_bytes` each.
unsafe fn select(m: usize, cell: u64, words: *mut u64, draws: *mut u32, pos: *mut u32, genes: *const u8, id_bytes: usize) -> usize {
    // SAFETY: as documented.
    unsafe {
        if m <= SAMPLE {
            for j in 0..m {
                *pos.add(j) = j as u32;
            }
            return m;
        }
        let nw = m.div_ceil(64);
        std::ptr::write_bytes(words, 0, nw);
        let keep = SAMPLE <= m - SAMPLE;
        let want = if keep { SAMPLE } else { m - SAMPLE };
        let first = m - want;
        let seed = cell ^ 0x5ca1_ab1e;
        if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512dq") {
            draws_avx512(seed, first, want, draws);
        } else {
            draws_scalar(seed, first, want, draws);
        }
        const AHEAD: usize = 32;
        for step in 0..if keep { want.min(AHEAD) } else { 0 } {
            prefetch_read(genes.add(*draws.add(step) as usize * id_bytes).cast());
        }
        for step in 0..want {
            if keep && step + AHEAD < want {
                prefetch_read(genes.add(*draws.add(step + AHEAD) as usize * id_bytes).cast());
            }
            let t = *draws.add(step) as usize;
            let j = first + step;
            let x = if (*words.add(t >> 6) >> (t & 63)) & 1 != 0 { j } else { t };
            *words.add(x >> 6) |= 1u64 << (x & 63);
            *pos.add(step) = x as u32;
        }
        if keep {
            return SAMPLE;
        }
        // Fewer than 4096 candidates: the unmarked positions, ascending.
        let mut k = 0;
        for w in 0..nw {
            let mut bits = !*words.add(w);
            if w == nw - 1 && !m.is_multiple_of(64) {
                bits &= (1u64 << (m % 64)) - 1;
            }
            while bits != 0 {
                *pos.add(k) = (w * 64 + bits.trailing_zeros() as usize) as u32;
                k += 1;
                bits &= bits - 1;
            }
        }
        debug_assert_eq!(k, SAMPLE);
        SAMPLE
    }
}

/// Floyd's draws for steps 0..want: step s draws t in [0, first + s] from the
/// low (s even) or high (s odd) half of splitmix64 output s / 2.
///
/// # Safety
/// `draws` holds `want` entries.
unsafe fn draws_scalar(seed: u64, first: usize, want: usize, draws: *mut u32) {
    let mut rng = Rng(seed);
    let mut r = 0u64;
    for step in 0..want {
        if step % 2 == 0 {
            r = rng.next();
        }
        let half = (r >> (32 * (step % 2))) as u32;
        // SAFETY: step < want.
        unsafe { *draws.add(step) = ((half as u64 * (first + step + 1) as u64) >> 32) as u32 };
    }
}

/// `draws_scalar` eight splitmix64 outputs (16 draws) at a time: output p
/// mixes state seed + (p + 1) * golden, so every lane is independent.
///
/// # Safety
/// `draws` holds `want` rounded up to 16 entries.
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn draws_avx512(seed: u64, first: usize, want: usize, draws: *mut u32) {
    use std::arch::x86_64::*;
    const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
    let lane = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
    let golden = _mm512_set1_epi64(GOLDEN as i64);
    let mut state = _mm512_add_epi64(_mm512_set1_epi64(seed as i64), _mm512_mullo_epi64(lane, golden));
    let step8 = _mm512_set1_epi64(GOLDEN.wrapping_mul(8) as i64);
    let mix1 = _mm512_set1_epi64(0xbf58_476d_1ce4_e5b9u64 as i64);
    let mix2 = _mm512_set1_epi64(0x94d0_49bb_1331_11ebu64 as i64);
    // Bounds j + 1 = first + s + 1 of the even (s = 2p) and odd steps.
    let pair = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let mut even = _mm512_add_epi64(_mm512_set1_epi64(first as i64 + 1), pair);
    let mut odd = _mm512_add_epi64(even, _mm512_set1_epi64(1));
    let sixteen = _mm512_set1_epi64(16);
    let mut s = 0;
    while s < want {
        let mut z = state;
        z = _mm512_mullo_epi64(_mm512_xor_si512(z, _mm512_srli_epi64(z, 30)), mix1);
        z = _mm512_mullo_epi64(_mm512_xor_si512(z, _mm512_srli_epi64(z, 27)), mix2);
        z = _mm512_xor_si512(z, _mm512_srli_epi64(z, 31));
        // mul_epu32 multiplies the low 32 bits of each 64-bit lane.
        let lo = _mm512_srli_epi64(_mm512_mul_epu32(z, even), 32);
        let hi = _mm512_srli_epi64(_mm512_mul_epu32(_mm512_srli_epi64(z, 32), odd), 32);
        // SAFETY: draws has room for whole groups of 16.
        unsafe { _mm512_storeu_si512(draws.add(s).cast(), _mm512_or_si512(lo, _mm512_slli_epi64(hi, 32))) };
        state = _mm512_add_epi64(state, step8);
        even = _mm512_add_epi64(even, sixteen);
        odd = _mm512_add_epi64(odd, sixteen);
        s += 16;
    }
}

/// Places the stored elements whose column is chosen, 16 at a time without
/// branches: their bit map words are gathered (the map is L1 resident) and
/// tested, and the hits gather their slots and scatter their values.
///
/// # Safety
/// `columns` and `values` hold n entries, every column below the map's dims;
/// the slot table has two bytes of slack.
#[target_feature(enable = "avx512f")]
unsafe fn place_avx512(columns: *const u32, values: *const f32, n: usize, chosen: *const u64, slot: *const u16, vals: *mut f32) {
    use std::arch::x86_64::*;
    let low = _mm512_set1_epi32(31);
    let one = _mm512_set1_epi32(1);
    let half = _mm512_set1_epi32(0xffff);
    let mut i = 0;
    while i < n {
        let tail = if n - i >= 16 { 0xffff } else { ((1u32 << (n - i)) - 1) as u16 };
        // SAFETY: masked lanes stay inside the row; gathered words lie in the
        // map and slots in the table (plus slack) because every column is
        // below its dims; hit slots are distinct and below 2048.
        unsafe {
            let cols = _mm512_maskz_loadu_epi32(tail, columns.add(i).cast());
            let word = _mm512_mask_i32gather_epi32::<4>(_mm512_setzero_si512(), tail, _mm512_srli_epi32(cols, 5), chosen.cast());
            let hit = _mm512_mask_test_epi32_mask(tail, _mm512_srlv_epi32(word, _mm512_and_si512(cols, low)), one);
            let at = _mm512_and_si512(_mm512_mask_i32gather_epi32::<2>(_mm512_setzero_si512(), hit, cols, slot.cast()), half);
            _mm512_mask_i32scatter_ps::<4>(vals.cast(), hit, at, _mm512_maskz_loadu_ps(hit, values.add(i)));
        }
        i += 16;
    }
}

/// Prefetches the cache line at `p` for reading.
#[inline(always)]
fn prefetch_read(p: *const i8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p)
    };
}

/// Prefetches the cache line at `p` for writing.
#[inline(always)]
fn prefetch_write(p: *const i8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_ET0 }>(p)
    };
}

/// Copies `len` bytes (a multiple of 64) to a 64-byte aligned destination with
/// streaming stores.
#[target_feature(enable = "avx512f")]
unsafe fn stream_avx512(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::*;
    for at in (0..len).step_by(64) {
        // SAFETY: both ranges hold len bytes; dst is 64-byte aligned.
        unsafe { _mm512_stream_si512(dst.add(at).cast(), _mm512_loadu_si512(src.add(at).cast())) };
    }
    _mm_sfence();
}

/// Stored non-zeros of a dense row.
#[allow(dead_code)]
pub(super) unsafe extern "C" fn count_dense(call: *const DenseRow) -> i32 {
    // SAFETY: the output holds 8 bytes; values hold out_dim floats.
    unsafe {
        let c = &*call;
        let row = std::slice::from_raw_parts(c.values.cast::<f32>(), c.out_dim as usize);
        c.out.cast::<u64>().write_unaligned(row.iter().filter(|&&v| v != 0.0).count() as u64);
    }
    0
}

#[allow(dead_code)]
pub(super) unsafe extern "C" fn count(call: *const CsrRow) -> i32 {
    // SAFETY: the output holds 8 bytes.
    unsafe { (*call).out.cast::<u64>().write_unaligned((*call).nnz) };
    0
}
