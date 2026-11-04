//! Partitioning of encode workspace: the scratch one row needs, sized by
//! the chunk's largest record. A chunk's prepared rows follow it (see
//! [`super::prep`] for their bound).

use super::prep::{Arena, Distinct, OrderWs, SMALL, ValuesWs, hash_slots, sparse_bound, values_bound};
use super::record_bound;
use crate::error::{Error, Result};
use crate::scans::kernels::{Kernels, SCAN_SLACK};
use crate::scans::rans::{WORD_SLACK, stream_bound};

#[inline]
fn words(bytes: usize) -> usize {
    bytes.div_ceil(8)
}

fn as_bytes(w: &mut [u64]) -> &mut [u8] {
    // SAFETY: `u8` has no alignment requirement and every bit pattern is
    // valid for both types; the length covers the same memory.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast(), w.len() * 8) }
}

fn as_u32(w: &mut [u64]) -> &mut [u32] {
    // SAFETY: `u64` alignment satisfies `u32`; every bit pattern is valid
    // for both types; the length covers the same memory.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast(), w.len() * 2) }
}

fn as_u16(w: &mut [u64]) -> &mut [u16] {
    // SAFETY: `u64` alignment satisfies `u16`; every bit pattern is valid
    // for both types; the length covers the same memory.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast(), w.len() * 4) }
}

/// The next `n` words of a workspace whose size was checked.
fn take<'w>(ws: &mut &'w mut [u64], n: usize) -> &'w mut [u64] {
    let (head, tail) = std::mem::take(ws).split_at_mut(n);
    *ws = tail;
    head
}

/// Row scratch regions, in words, for records of at most `max_elems`
/// elements of `vw`-byte values.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowLayout {
    bitmap: usize,
    pos: usize,
    elems32: usize,
    coded: usize,
    elems64: usize,
    hash_keys: usize,
    hash_ids: usize,
    rank_of: usize,
    words: usize,
    stream: usize,
    arena: usize,
    dense: usize,
    dense_arena: usize,
    alt: usize,
    sp_idx: usize,
    sp_val: usize,
    modes: bool,
}

/// Row scratch: coded order, value preparation, record coding, and the
/// dense candidate.
pub(crate) struct RowWs<'w> {
    /// Coded order and value preparation of the row.
    pub(crate) prep: PrepWs<'w>,
    /// rANS scratch and the streams of both value codings.
    pub(crate) code: CodeWs<'w>,
    /// Prepared symbols of a sparse record coded on its own.
    pub(crate) arena: Arena<'w>,
    /// Every gene's value of a row in coded order.
    pub(crate) dense: &'w mut [u8],
    /// Prepared symbols of the dense candidate.
    pub(crate) dense_arena: Arena<'w>,
    /// The dense candidate while the sparse record holds the output.
    pub(crate) alt: &'w mut [u8],
}

/// Scratch for preparing a sparse record.
pub(crate) struct PrepWs<'w> {
    /// Coded order of the row's cell.
    pub(crate) order: OrderWs<'w>,
    /// Value preparation of the row.
    pub(crate) values: ValuesWs<'w>,
    /// The non-zeros of a dense input row.
    pub(crate) sp_idx: &'w mut [u32],
    /// Values of those non-zeros.
    pub(crate) sp_val: &'w mut [u8],
}

/// rANS scratch and the streams of both value codings, kept side by side
/// so the smaller can be written.
pub(crate) struct CodeWs<'w> {
    /// Word scratch of the streams coded together.
    pub(crate) words_gap: &'w mut [u16],
    /// Word scratch of the dictionary's rank stream.
    pub(crate) words_rank: &'w mut [u16],
    /// Word scratch of the mode coding's rank stream.
    pub(crate) words_mode: &'w mut [u16],
    /// Gap stream of a sparse record.
    pub(crate) gap_stream: &'w mut [u8],
    /// Dictionary stream of the per-row dictionary coding.
    pub(crate) dict_stream: &'w mut [u8],
    /// Rank stream of the per-row dictionary coding.
    pub(crate) rank_stream: &'w mut [u8],
    /// Dictionary stream of the mode coding.
    pub(crate) mode_dict_stream: &'w mut [u8],
    /// Rank stream of the mode coding.
    pub(crate) mode_rank_stream: &'w mut [u8],
}

impl RowLayout {
    /// Regions for records of at most `max_elems` elements of `vw`-byte
    /// values; `modes` when rows are prepared against column modes, which
    /// needs one more word per element. Dense-candidate regions appear only
    /// when `max_elems` covers every gene.
    pub(crate) fn new(n_genes: u32, max_elems: usize, vw: usize, modes: bool) -> Self {
        let g = n_genes as usize;
        let e = max_elems.max(1);
        let cap = hash_slots(e);
        let dense = max_elems >= g;
        RowLayout {
            bitmap: g.div_ceil(64),
            pos: words(4 * g),
            elems32: words(4 * e),
            coded: words(4 * (e + SCAN_SLACK)),
            elems64: e,
            hash_keys: cap,
            hash_ids: words(4 * cap),
            rank_of: words(4 * (e + 1)),
            words: words(2 * (e + WORD_SLACK)),
            stream: words(stream_bound(e, 32)),
            arena: words(sparse_bound(e, n_genes, vw, modes)),
            dense: if dense { words(g * vw) } else { 0 },
            dense_arena: if dense { words(values_bound(g, vw, modes)) } else { 0 },
            alt: if dense { words(record_bound(n_genes, vw, g, true)) } else { 0 },
            sp_idx: words(4 * e.min(g)),
            sp_val: words(vw * e.min(g)),
            modes,
        }
    }

    /// Words the row scratch reserves: an upper bound on what
    /// [`RowLayout::split`] carves out.
    pub(crate) fn total(&self) -> usize {
        let exceptions = self.modes as usize;
        self.bitmap
            + self.pos
            + 4 * self.elems32
            + self.coded
            + (3 + exceptions) * self.elems64
            + self.hash_keys
            + self.hash_ids
            + words(4 * SMALL)
            + self.rank_of
            + 3 * self.words
            + 5 * self.stream
            + self.arena
            + self.dense
            + self.dense_arena
            + self.alt
            + self.sp_idx
            + self.sp_val
    }

    /// The row scratch at the start of `ws` and the rest of it, which a
    /// chunk uses for the arena of its prepared records.
    ///
    /// The bitmap and the direct table start clear; every other region is
    /// written before it is read.
    pub(crate) fn split<'w>(&self, ws: &'w mut [u64], kernels: &'static Kernels) -> Result<(RowWs<'w>, &'w mut [u64])> {
        if ws.len() < self.total() {
            return Err(Error::WorkspaceTooSmall {
                need: self.total(),
                have: ws.len(),
            });
        }
        let mut ws = ws;
        let ws = &mut ws;
        // The bitmap is kept clear between cells, not between calls.
        let bitmap = take(ws, self.bitmap);
        bitmap.fill(0);
        let order = OrderWs {
            bitmap,
            pos: as_u32(take(ws, self.pos)),
            coded: as_u32(take(ws, self.coded)),
            perm: as_u32(take(ws, self.elems32)),
            keys: take(ws, self.elems64),
            kernels,
        };
        let ids = as_u32(take(ws, self.elems32));
        let mids = as_u32(take(ws, self.elems32));
        let sort = take(ws, self.elems64);
        let hash_keys = take(ws, self.hash_keys);
        let hash_ids = as_u32(take(ws, self.hash_ids));
        // The direct table is kept clear between cells, not between calls.
        let small = as_u32(take(ws, words(4 * SMALL)));
        small.fill(0);
        let dict = Distinct::new(hash_keys, hash_ids, small, take(ws, self.elems64));
        let exc = take(ws, if self.modes { self.elems64 } else { 0 });
        let values = ValuesWs {
            esc: as_u32(take(ws, self.elems32)),
            ids,
            mids,
            dict,
            exc,
            sort,
            rank_of: as_u32(take(ws, self.rank_of)),
        };
        let code = CodeWs {
            words_gap: as_u16(take(ws, self.words)),
            words_rank: as_u16(take(ws, self.words)),
            words_mode: as_u16(take(ws, self.words)),
            gap_stream: as_bytes(take(ws, self.stream)),
            dict_stream: as_bytes(take(ws, self.stream)),
            rank_stream: as_bytes(take(ws, self.stream)),
            mode_dict_stream: as_bytes(take(ws, self.stream)),
            mode_rank_stream: as_bytes(take(ws, self.stream)),
        };
        let prep = PrepWs {
            order,
            values,
            sp_idx: as_u32(take(ws, self.sp_idx)),
            sp_val: as_bytes(take(ws, self.sp_val)),
        };
        let row = RowWs {
            prep,
            code,
            arena: Arena::new(as_bytes(take(ws, self.arena))),
            dense: as_bytes(take(ws, self.dense)),
            dense_arena: Arena::new(as_bytes(take(ws, self.dense_arena))),
            alt: as_bytes(take(ws, self.alt)),
        };
        Ok((row, std::mem::take(ws)))
    }
}
