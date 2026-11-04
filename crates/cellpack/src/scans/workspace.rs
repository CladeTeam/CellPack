//! Partitioning of caller-provided workspace.
//!
//! Workspace is a `&mut [u64]` whose size depends only on chunk-level
//! parameters, so one buffer serves every cell of a chunk (and of any chunk
//! with smaller parameters). Its contents carry nothing between calls.
//!
//! [`DecodeLayout`] derives every region from the gene count, the value
//! width, the largest dictionary and the largest record, so a caller sizes
//! its buffer from metadata (`decode_workspace_words`) or from bounds alone
//! (`decode_workspace_bound`).

use crate::error::{Error, Result};
use crate::format::id_width;
use crate::scans::kernels::{DICT_SLACK, SCAN_SLACK};
use crate::scans::rans::stream_bound;

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

/// The next `n` words. Callers check the total size first.
#[inline(always)]
fn take<'w>(ws: &mut &'w mut [u64], n: usize) -> &'w mut [u64] {
    debug_assert!(n <= ws.len());
    // SAFETY: `split` checked that the workspace holds every region.
    let (head, tail) = unsafe { std::mem::take(ws).split_at_mut_unchecked(n) };
    *ws = tail;
    head
}

/// Decode workspace regions, in words.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodeLayout {
    stage: usize,
    marks: usize,
    slot: usize,
    idx_bits: usize,
    val_pool: usize,
    dict: usize,
    syms: usize,
    syms_dict: usize,
    ids: usize,
    vstage: usize,
}

/// Regions of a decode workspace, sized by [`DecodeLayout`]. Every region is
/// written before it is read in the call that needs it.
pub(crate) struct DecodeWs<'w> {
    /// A paged record copied into contiguous memory.
    pub(crate) stage: &'w mut [u8],
    /// One byte per gene (a multiple of 64); restores source order after renumbering.
    pub(crate) marks: &'w mut [u8],
    /// Record position of each source gene while restoring source order.
    pub(crate) slot: &'w mut [u32],
    /// Gap escape bits, rejoined from the gap stream's states and the record.
    pub(crate) idx_bits: &'w mut [u8],
    /// Dict rans, dict escape bits and rank escape bits, rejoined likewise.
    pub(crate) val_pool: &'w mut [u8],
    /// The cell's dictionary in value width, plus gather slack.
    pub(crate) dict: &'w mut [u8],
    /// Gap symbols.
    pub(crate) syms_idx: &'w mut [u8],
    /// Rank symbols.
    pub(crate) syms_val: &'w mut [u8],
    /// Dictionary symbols.
    pub(crate) syms_dict: &'w mut [u8],
    /// Coded ids in record order, with scan slack.
    pub(crate) coded: &'w mut [u32],
    /// Source ids ascending, with scan slack.
    pub(crate) sorted: &'w mut [u32],
    /// Ranks of dictionaries beyond the direct symbol range; dictionary staging.
    pub(crate) ranks32: &'w mut [u32],
    /// Values reordered into source order.
    pub(crate) vstage: &'w mut [u8],
}

impl DecodeLayout {
    /// `max_nnz` bounds the elements of a record (non-zeros, or every gene
    /// of a dense record).
    pub(crate) fn new(
        n_genes: u32,
        renumbered: bool,
        vw: usize,
        max_k: usize,
        max_nnz: usize,
    ) -> Self {
        let record = 1 + 10 + max_nnz * (id_width(n_genes) + vw);
        DecodeLayout {
            stage: words(record),
            marks: if renumbered {
                (n_genes as usize).div_ceil(64) * 8
            } else {
                0
            },
            slot: if renumbered {
                words(n_genes as usize * 4)
            } else {
                0
            },
            // Gaps and ranks of integers below 2^32 carry at most 31 escape bits,
            // so four bytes per element cover them; the dictionary's gaps can
            // reach 63 bits and take eight bytes each in `val_pool`.
            idx_bits: words(4 * max_nnz + 8),
            val_pool: words(stream_bound(max_k, 32) + 8 * max_k + 4 * max_nnz + 8),
            dict: words(max_k * vw + DICT_SLACK),
            syms: words(max_nnz),
            syms_dict: words(max_k),
            ids: words((max_nnz + SCAN_SLACK) * 4),
            vstage: words(max_nnz * vw),
        }
    }

    /// Words all regions of [`DecodeLayout::split`] occupy.
    pub(crate) fn total(&self) -> usize {
        self.stage
            + self.marks
            + self.slot
            + self.idx_bits
            + self.val_pool
            + self.dict
            + 2 * self.syms
            + self.syms_dict
            + 3 * self.ids
            + self.vstage
    }

    /// Carves `ws` into the regions a decode borrows; `ws` must hold
    /// [`DecodeLayout::total`] words. Regions carry nothing between calls.
    pub(crate) fn split<'w>(&self, ws: &'w mut [u64]) -> Result<DecodeWs<'w>> {
        if ws.len() < self.total() {
            return Err(Error::WorkspaceTooSmall {
                need: self.total(),
                have: ws.len(),
            });
        }
        let mut ws = ws;
        Ok(DecodeWs {
            stage: as_bytes(take(&mut ws, self.stage)),
            marks: as_bytes(take(&mut ws, self.marks)),
            slot: as_u32(take(&mut ws, self.slot)),
            idx_bits: as_bytes(take(&mut ws, self.idx_bits)),
            val_pool: as_bytes(take(&mut ws, self.val_pool)),
            dict: as_bytes(take(&mut ws, self.dict)),
            syms_idx: as_bytes(take(&mut ws, self.syms)),
            syms_val: as_bytes(take(&mut ws, self.syms)),
            syms_dict: as_bytes(take(&mut ws, self.syms_dict)),
            coded: as_u32(take(&mut ws, self.ids)),
            sorted: as_u32(take(&mut ws, self.ids)),
            ranks32: as_u32(take(&mut ws, self.ids)),
            vstage: as_bytes(take(&mut ws, self.vstage)),
        })
    }
}
