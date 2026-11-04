//! Decoding at three granularities: one record, a chunk, or a row selection
//! of a resident chunk, each to CSR arrays or dense rows, plus a parts view
//! of one record for callers that map and scatter themselves. Records come
//! as contiguous bytes or as page arrays (copied into the workspace only
//! when they cross a page). All output and workspace memory comes from the
//! caller.

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "kernels/typed_tests.rs"]
mod typed_tests;

use crate::error::{Error, Result};
use crate::format::{self, DICT, GAP, IdxPart, Meta, RANK, RecordView, ValPart};
use crate::scans::bits::{BitReader, exact};
use crate::scans::kernels::Kernels;
use crate::scans::rans::{lanes_for, payload};
use crate::scans::tables::DecodeTables;
use crate::scans::types::{Fill, IdSpace, IndexOut, IndptrOut, PagedRecord, Rows};
use crate::scans::workspace::{DecodeLayout, DecodeWs};
use crate::unchecked::{get, set};
use crate::scans::{coding::value_max, symbol};

/// What every per-record step reads: the bound kernels and the chunk's metadata and tables.
#[derive(Clone, Copy)]
struct Ctx<'c, 'm> {
    k: &'c Kernels,
    meta: &'c Meta<'m>,
    t: &'c DecodeTables,
}

/// Workspace layout of a chunk: sized for its largest record, dictionary and
/// gene order.
fn layout(meta: &Meta) -> DecodeLayout {
    DecodeLayout::new(
        meta.n_dims(),
        meta.renumbered(),
        meta.value_width(),
        meta.max_k(),
        meta.max_elems(),
    )
}

/// Coded ids `0..n` of a dense record, shared by every decode.
///
/// One static table serves every dense record up to `2^20` genes; a longer
/// record gets `None` and its ids are written out instead.
pub(crate) fn iota(n: usize) -> Option<&'static [u32]> {
    static IOTA: std::sync::OnceLock<Box<[u32]>> = std::sync::OnceLock::new();
    IOTA.get_or_init(|| (0..1u32 << 20).collect()).get(..n)
}

/// Decode workspace in `u64` words for any record of the chunk.
pub fn decode_workspace_words(meta: &Meta) -> usize {
    layout(meta).total()
}

/// Decode workspace in `u64` words sufficient for every chunk with at most
/// these parameters, without reading any metadata.
pub fn decode_workspace_bound(n_genes: u32, value_width: usize) -> usize {
    DecodeLayout::new(
        n_genes,
        true,
        value_width,
        n_genes as usize,
        n_genes as usize,
    )
    .total()
}

/// Non-zeros of the selected cells.
pub fn selection_nnz(meta: &Meta, rows: Rows) -> Result<u64> {
    let n = rows.count(meta.n_cells())?;
    if matches!(rows, Rows::All) {
        return Ok(meta.nnz_total());
    }
    let mut total = 0u64;
    for i in 0..n {
        total += meta.cell_nnz(rows.get(i))? as u64;
    }
    Ok(total)
}

/// Bytes of dense output for the selected cells: rows × n_genes × value width.
pub fn dense_len(meta: &Meta, rows: Rows) -> Result<usize> {
    let n = rows.count(meta.n_cells())?;
    n.checked_mul(meta.n_dims() as usize)
        .and_then(|v| v.checked_mul(meta.value_width()))
        .ok_or(Error::InvalidInput(
            "dense output exceeds the address space",
        ))
}

fn check_index_type<I: IndexOut>(meta: &Meta) -> Result<()> {
    let largest = meta.n_dims() as u64 - 1;
    if largest > I::MAX {
        Err(Error::OutputTypeTooNarrow(largest))
    } else {
        Ok(())
    }
}

/// Decodes one stream with the lane count the encoder chose for `out.len()`
/// symbols; the lanes' final states go to `ends`.
fn rans(
    cx: Ctx,
    model: usize,
    stream: &[u8],
    out: &mut [u8],
    ends: &mut [u32; 32],
) -> Result<usize> {
    let Ctx { k, meta, t } = cx;
    let lanes = lanes_for(out.len(), meta.max_lanes(), meta.lane_min());
    // SAFETY: a well-formed record only has streams for models its chunk
    // uses; lane counts are powers of two up to 32; the bound kernel set
    // only contains kernels the CPU supports.
    unsafe {
        let slots = t.slots_unchecked(model);
        (k.rans.get_unchecked(lanes.trailing_zeros() as usize))(slots, stream, out, ends);
    }
    Ok(lanes)
}

/// Decodes the gap and rank streams of a record together.
fn rans_pair(
    cx: Ctx,
    gap: &[u8],
    rank: &[u8],
    ws: &mut DecodeWs,
    n: usize,
    eg: &mut [u32; 32],
    er: &mut [u32; 32],
) -> Result<usize> {
    let Ctx { k, meta, t } = cx;
    let lanes = lanes_for(n, meta.max_lanes(), meta.lane_min());
    // SAFETY: as in `rans`; the symbol buffers hold `max_elems >= n`.
    unsafe {
        (k.rans_pair.get_unchecked(lanes.trailing_zeros() as usize))(
            t.slots_unchecked(GAP),
            gap,
            ws.syms_idx.get_unchecked_mut(..n),
            t.slots_unchecked(RANK),
            rank,
            ws.syms_val.get_unchecked_mut(..n),
            eg,
            er,
        );
    }
    Ok(lanes)
}

/// Rebuilds bytes split between a stream's initial states (`carried`
/// leading bytes) and a stored `tail` into `dst`.
fn rejoin(ends: &[u32], carried: usize, tail: &[u8], dst: &mut [u8]) -> Result<()> {
    let total = carried + tail.len();
    debug_assert!(total <= dst.len(), "escape bits longer than the chunk allows");
    // SAFETY: the workspace holds the escape bits of any well-formed record
    // of the chunk.
    let dst = unsafe { dst.get_unchecked_mut(..total) };
    // SAFETY: `carried <= total = dst.len()`.
    let (head, rest) = unsafe { dst.split_at_mut_unchecked(carried) };
    payload(ends, carried, head);
    rest.copy_from_slice(tail);
    Ok(())
}

/// Decodes every symbol stream of a record into the workspace: gap and rank
/// symbols (paired when both exist), then the escape bits and value pool
/// they carry, then the dictionary symbols.
fn streams(cx: Ctx, view: &RecordView, ws: &mut DecodeWs) -> Result<()> {
    let n = view.elems;
    let (mut eg, mut er) = ([0u32; 32], [0u32; 32]);
    let gap = match view.idx {
        IdxPart::Gap { rans, .. } => Some(rans),
        IdxPart::Raw(_) | IdxPart::Dense => None,
    };
    let rank = match view.val {
        ValPart::Dict { k, rank, .. } => {
            debug_assert!(k <= cx.meta.max_k(), "dictionary larger than the chunk maximum");
            rank
        }
        ValPart::Raw(_) => None,
    };
    // Gap and rank streams have one symbol per element and so share the lane
    // count, which lets one kernel decode both with overlapping chains.
    let (lg, lr) = match (gap, rank) {
        (Some(g), Some(r)) => {
            let l = rans_pair(cx, g, r, ws, n, &mut eg, &mut er)?;
            (l, l)
        }
        // SAFETY: the symbol buffers hold `max_elems >= n`.
        (Some(g), None) => (rans(cx, GAP, g, unsafe { ws.syms_idx.get_unchecked_mut(..n) }, &mut eg)?, 0),
        // SAFETY: as above.
        (None, Some(r)) => (0, rans(cx, RANK, r, unsafe { ws.syms_val.get_unchecked_mut(..n) }, &mut er)?),
        (None, None) => (0, 0),
    };
    if let IdxPart::Gap { carried, tail, .. } = view.idx {
        rejoin(&eg[..lg], carried, tail, ws.idx_bits)?;
    }
    if let ValPart::Dict {
        k,
        dict_len,
        carried,
        tail,
        ..
    } = view.val
        && k > 0
    {
        rejoin(&er[..lr], carried, tail, ws.val_pool)?;
        let mut ed = [0u32; 32];
        // SAFETY: the pool holds `dict_len` rejoined bytes and the dictionary
        // symbols hold `max_k >= k`.
        let ld = unsafe {
            rans(cx, DICT, ws.val_pool.get_unchecked(..dict_len), ws.syms_dict.get_unchecked_mut(..k), &mut ed)?
        };
        payload(&ed[..ld], 0, &mut []);
    }
    Ok(())
}

// `debug_assert!` type-checks its condition in every profile.
#[cfg_attr(not(debug_assertions), allow(dead_code))]
fn max_symbol(syms: &[u8]) -> u8 {
    syms.iter().fold(0u8, |m, &s| m.max(s))
}

/// Integers from symbols into `out` (see [`crate::scans::kernels::FieldsFn`]).
#[inline]
fn fields(k: &Kernels, syms: &[u8], bits: &[u8], out: &mut [u32]) {
    debug_assert!(max_symbol(syms) <= symbol::MAX_SYMBOL_32, "integer wider than 32 bits");
    // SAFETY: a well-formed record's integers fit 32-bit buckets; the
    // kernel's ISA is supported.
    let used = unsafe { (k.fields)(syms, bits, out) };
    debug_assert!(exact(used, bits.len()), "escape bits do not decode exactly");
    let _ = used;
}

/// Strictly ascending values from gaps in place; returns the exact last value.
fn ascend(k: &Kernels, v: &mut [u32]) -> u64 {
    // SAFETY: the kernel's ISA is supported.
    unsafe { (k.ascend)(v) }
}

/// Coded ids from gap symbols into `out`.
#[inline]
fn gaps(k: &Kernels, syms: &[u8], bits: &[u8], n_genes: u64, out: &mut [u32]) {
    fields(k, syms, bits, out);
    // Ids strictly ascend, so the exact last id bounds all of them.
    let last = ascend(k, out);
    debug_assert!(last < n_genes || out.is_empty(), "gene ids out of range");
    let _ = (last, n_genes);
}

/// Coded ids of a record, ascending, into `out` (length nnz).
fn coded_ids(cx: Ctx, part: &IdxPart, syms: &[u8], bits: &[u8], out: &mut [u32]) {
    let n_genes = cx.meta.n_dims() as u64;
    match *part {
        IdxPart::Raw(bytes) => {
            fn read<const W: usize>(bytes: &[u8], out: &mut [u32]) {
                debug_assert!(bytes.len() >= out.len() * W);
                for (t, o) in out.iter_mut().enumerate() {
                    // SAFETY: a raw index section holds one `W`-byte id per
                    // element.
                    let p = unsafe { bytes.as_ptr().add(t * W) };
                    // SAFETY: as above.
                    *o = unsafe { if W == 2 { u16::from_le_bytes(p.cast::<[u8; 2]>().read_unaligned()) as u32 } else { u32::from_le_bytes(p.cast::<[u8; 4]>().read_unaligned()) } };
                }
            }
            if cx.meta.index_width() == 2 { read::<2>(bytes, out) } else { read::<4>(bytes, out) }
            debug_assert!(out.windows(2).all(|w| w[0] < w[1]) && out.last().is_none_or(|&c| (c as u64) < n_genes), "raw indices not ascending or out of range");
        }
        IdxPart::Gap { .. } => gaps(cx.k, syms, bits, n_genes, out),
        IdxPart::Dense => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = i as u32;
            }
        }
    }
}

/// Source order of a renumbered record: `sorted` gets its source genes
/// ascending and `slot[s]` the record position of source gene `s`. Each
/// coded id marks its source gene with one independent byte store; the
/// scan kernel reads the marks back in order.
fn restore(
    k: &Kernels,
    soc: &[u32],
    coded: &[u32],
    marks: &mut [u8],
    slot: &mut [u32],
    sorted: &mut [u32],
) -> Result<()> {
    marks.fill(0);
    for (t, &c) in coded.iter().enumerate() {
        // SAFETY: coded ids lie below `n_genes`, which bounds `soc`; source
        // genes index `marks` and `slot`, both sized for every gene.
        unsafe {
            let s = get(soc, c as usize) as usize;
            set(marks, s, 1);
            set(slot, s, t as u32);
        }
    }
    // SAFETY: `sorted` holds `max_nnz >= coded.len()` entries plus the scan
    // slack, `marks` is a multiple of 64 bytes, and the kernel's ISA is supported.
    let found = unsafe { (k.scan)(marks, sorted) };
    // Distinct coded ids under a permutation mark exactly as many genes.
    debug_assert_eq!(found, coded.len(), "gene ids repeat");
    let _ = found;
    Ok(())
}

/// Rebuilds a dictionary (ascending distinct values) from its gap symbols.
/// The values ascend, so checking the last against the value width bounds
/// all of them. Widths up to 4 bytes have gaps below 2^32 and use the
/// field and running-sum kernels, staged in `wide`.
fn dictionary(
    k: &Kernels,
    syms: &[u8],
    bits: &[u8],
    vw: usize,
    dict: &mut [u8],
    wide: &mut [u32],
) -> Result<()> {
    fn narrow<const W: usize>(vals: &[u32], dict: &mut [u8]) {
        for (d, v) in dict.chunks_exact_mut(W).zip(vals) {
            d.copy_from_slice(&v.to_le_bytes()[..W]);
        }
    }
    let kd = syms.len();
    if vw <= 4 {
        // SAFETY: the rank staging buffer holds `max_elems >= max_k >= kd`.
        let vals = unsafe { wide.get_unchecked_mut(..kd) };
        fields(k, syms, bits, vals);
        let last = ascend(k, vals);
        debug_assert!(last <= value_max(vw), "dictionary value overflows its width");
        let _ = last;
        match vw {
            1 => narrow::<1>(vals, dict),
            2 => narrow::<2>(vals, dict),
            _ => narrow::<4>(vals, dict),
        }
        return Ok(());
    }
    let mut reader = BitReader::new(bits);
    // Values ascend from the first gap: `prev` starts at -1.
    let mut prev = u64::MAX;
    for (d, &s) in dict.chunks_exact_mut(8).zip(syms) {
        let g = symbol::join(s, reader.get(symbol::extra_bits(s)));
        prev = prev.wrapping_add(1).wrapping_add(g);
        d.copy_from_slice(&prev.to_le_bytes());
    }
    debug_assert!(reader.finish_ok(), "dictionary escape bits do not decode exactly");
    Ok(())
}

/// Dictionary positions of a record's values.
#[derive(Clone, Copy, Debug)]
pub enum Ranks<'a> {
    /// A one-entry dictionary: every value is entry 0.
    One,
    /// Small dictionaries: a rank is its own symbol, so no escape bits.
    U8(&'a [u8]),
    /// Larger dictionaries: ranks are unpacked into `u32` first.
    U32(&'a [u32]),
}

/// A record's dictionary: `k` distinct values ascending, and each element's
/// position in it. Debug builds check every rank against `k`.
#[derive(Clone, Copy, Debug)]
pub struct Dict<'a> {
    /// The entries followed by gather slack.
    padded: &'a [u8],
    k: usize,
    vw: usize,
    ranks: Ranks<'a>,
}

impl<'a> Dict<'a> {
    /// Distinct values in the dictionary.
    pub fn k(&self) -> usize {
        self.k
    }
    /// Bytes per entry.
    pub fn value_width(&self) -> usize {
        self.vw
    }
    /// The `k` entries, `value_width` little-endian bytes each.
    pub fn entries(&self) -> &'a [u8] {
        &self.padded[..self.k * self.vw]
    }
    /// Each element's position in [`Self::entries`].
    pub fn ranks(&self) -> Ranks<'a> {
        self.ranks
    }
}

/// A mode-coded record's values: rank 0 is the element's column mode (the
/// chunk's [`Meta::column_modes`] entry for its coded id) and rank `r` is
/// dictionary entry `r - 1`. [`Ranks::One`] means every value is its
/// column mode (an empty dictionary). Debug builds check every rank against
/// `k`.
#[derive(Clone, Copy, Debug)]
pub struct ModeDict<'a> {
    padded: &'a [u8],
    k: usize,
    vw: usize,
    ranks: Ranks<'a>,
    modes: &'a [u8],
}

impl<'a> ModeDict<'a> {
    /// Values that differ from their column mode, distinct.
    pub fn k(&self) -> usize {
        self.k
    }
    /// Bytes per entry, and per column mode.
    pub fn value_width(&self) -> usize {
        self.vw
    }
    /// The `k` entries, `value_width` little-endian bytes each.
    pub fn entries(&self) -> &'a [u8] {
        &self.padded[..self.k * self.vw]
    }
    /// Each element's rank: 0 for its column mode, else entry `rank - 1`.
    pub fn ranks(&self) -> Ranks<'a> {
        self.ranks
    }
    /// The chunk's column modes, `value_width` bytes per coded id.
    pub fn modes(&self) -> &'a [u8] {
        self.modes
    }
    /// Value bytes of element `t` with coded id `id`.
    ///
    /// # Safety
    /// `t` is an element of the row and `id` its coded id.
    #[inline(always)]
    unsafe fn value(&self, t: usize, id: u32) -> &'a [u8] {
        let vw = self.vw;
        // SAFETY: ranks hold one entry per element, at most `k`; a rank `r > 0`
        // names dictionary entry `r - 1`, and coded ids index the modes.
        unsafe {
            let r = match self.ranks {
                Ranks::One => 0,
                Ranks::U8(r) => get(r, t) as usize,
                Ranks::U32(r) => get(r, t) as usize,
            };
            if r == 0 {
                self.modes.get_unchecked(id as usize * vw..id as usize * vw + vw)
            } else {
                self.padded.get_unchecked((r - 1) * vw..r * vw)
            }
        }
    }
}

/// A record's values, aligned with its ids.
#[derive(Clone, Copy, Debug)]
pub enum Values<'a> {
    /// `value_width` little-endian bytes per element.
    Raw(&'a [u8]),
    /// A dictionary and each element's position in it.
    Dict(Dict<'a>),
    /// Mode-coded values; only with [`IdSpace::Coded`] ids, where they index
    /// the column modes. Source-order views materialize them as `Raw`.
    Modes(ModeDict<'a>),
}

/// A decoded record before values are materialized: ids and values in the
/// same order, borrowed from the workspace (or the record). With
/// [`IdSpace::Coded`] the ids ascend in the chunk's coded numbering and no
/// order restoration runs; with [`IdSpace::Source`] they ascend in source
/// numbering. Callers mapping genes themselves can convert the `k`
/// dictionary entries once instead of every value.
/// Read-only access keeps the validated ids and rank lengths aligned.
#[derive(Clone, Copy, Debug)]
pub struct Parts<'a> {
    pub(crate) ids: &'a [u32],
    pub(crate) values: Values<'a>,
    /// Element count: stored entries for a sparse record, non-zeros for a
    /// dense one (fewer than its elements).
    pub(crate) nnz: usize,
    pub(crate) dense: bool,
}

impl<'a> Parts<'a> {
    /// Validated ascending ids, aligned with [`Self::values`].
    #[inline]
    pub fn ids(&self) -> &'a [u32] {
        self.ids
    }

    /// Values aligned with [`Self::ids`]; ranks remain tied to this row.
    #[inline]
    pub fn values(&self) -> Values<'a> {
        self.values
    }

    /// Declared element count of the record: stored entries for a sparse
    /// record, non-zeros for a dense one.
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Elements: the stored entries of a sparse record, every gene of a dense
    /// one.
    pub fn elements(&self) -> usize {
        self.ids.len()
    }

    /// How the record stores its elements. A dense record's elements are
    /// every gene in id order, zeros included.
    pub fn layout(&self) -> format::RecordLayout {
        if self.dense {
            format::RecordLayout::Dense
        } else {
            format::RecordLayout::Sparse
        }
    }

    /// Writes the element values in order to `out` (`elements × value_width` bytes).
    pub fn expand_values(&self, out: &mut [u8]) -> Result<()> {
        expand(crate::runtime::get()?.kernels(), &self.values, self.ids, out)
    }

    /// [`Self::expand_values`] with the kernel set of the chunk's tables.
    pub(crate) fn expand_values_with(&self, tables: &DecodeTables, out: &mut [u8]) -> Result<()> {
        expand(tables.kernels, &self.values, self.ids, out)
    }
}

/// Values of the elements with these ids into `out` from a parts view.
fn expand(k: &Kernels, values: &Values, ids: &[u32], out: &mut [u8]) -> Result<()> {
    let n = ids.len();
    match *values {
        Values::Modes(d) => {
            let vw = d.vw;
            let need = n * vw;
            let have = out.len();
            let out = out
                .get_mut(..need)
                .ok_or(Error::OutputTooSmall { need, have })?;
            for (t, (o, &id)) in out.chunks_exact_mut(vw).zip(ids).enumerate() {
                // SAFETY: `t` and `id` describe element `t` of the row.
                o.copy_from_slice(unsafe { d.value(t, id) });
            }
        }
        Values::Raw(bytes) => {
            let have = out.len();
            let dst = out.get_mut(..bytes.len()).ok_or(Error::OutputTooSmall {
                need: bytes.len(),
                have,
            })?;
            dst.copy_from_slice(bytes);
        }
        Values::Dict(d) => {
            let vw = d.vw;
            let need = n * vw;
            let have = out.len();
            let out = out
                .get_mut(..need)
                .ok_or(Error::OutputTooSmall { need, have })?;
            match d.ranks {
                Ranks::One => {
                    for o in out.chunks_exact_mut(vw) {
                        o.copy_from_slice(&d.padded[..vw]);
                    }
                }
                // SAFETY: the record is well-formed, so every rank is below
                // `k` (debug builds assert it when the view is built);
                // `padded` carries the gather slack, `out` holds exactly one
                // value per rank, and the kernel's ISA is supported.
                Ranks::U8(r) => unsafe {
                    (k.expand[vw.trailing_zeros() as usize])(d.padded, r, out)
                },
                // SAFETY: as above; ranks are below `k <= n_genes`, which the
                // metadata keeps below 2^31, so they fit a signed gather index.
                Ranks::U32(r) => unsafe { (k.expand32[vw.trailing_zeros() as usize])(d.padded, r, out) },
            }
        }
    }
    Ok(())
}

/// Workspace pieces a values view borrows from.
struct ValueBufs<'a, 'w> {
    /// Dictionary symbols, `k` of them.
    syms_dict: &'a [u8],
    /// The rejoined value pool: dictionary stream, dictionary escape bits,
    /// rank escape bits, in that order.
    val_pool: &'a [u8],
    /// Rank symbols, one per element.
    syms_val: &'w [u8],
    /// `k` entries in value width, with gather slack.
    dict: &'w mut [u8],
    /// Rank staging for dictionaries past the direct symbol range.
    ranks32: &'w mut [u32],
}

/// The values of a record in record order, as a view.
fn values<'w, 'm: 'w>(
    cx: Ctx<'_, 'm>,
    part: &ValPart<'w>,
    n: usize,
    b: ValueBufs<'_, 'w>,
) -> Result<Values<'w>> {
    let Ctx { k, meta, .. } = cx;
    let vw = meta.value_width();
    match *part {
        ValPart::Raw(bytes) => Ok(Values::Raw(bytes)),
        ValPart::Dict {
            modes: true,
            k: kd,
            dict_len,
            dict_bits_len,
            ..
        } => {
            debug_assert!(meta.column_modes().is_some(), "mode-coded record in a chunk without column modes");
            // SAFETY: mode-coded records only occur in chunks with column modes.
            let table = unsafe { meta.column_modes().unwrap_unchecked() };
            if kd == 0 {
                return Ok(Values::Modes(ModeDict {
                    padded: b.dict,
                    k: 0,
                    vw,
                    ranks: Ranks::One,
                    modes: table,
                }));
            }
            // SAFETY: the pool holds the rejoined `pool_len` bytes, whose
            // dictionary stream and escape bits come first; the symbol and
            // dictionary buffers hold `max_k >= kd` and `max_elems >= n`.
            let (dict_bits, rank_bits, syms_dict, dict, syms) = unsafe {
                let pool = b.val_pool.get_unchecked(..part.pool_len());
                let (dict_bits, rank_bits) = pool.get_unchecked(dict_len..).split_at_unchecked(dict_bits_len);
                (dict_bits, rank_bits, b.syms_dict.get_unchecked(..kd), b.dict.get_unchecked_mut(..kd * vw), b.syms_val.get_unchecked(..n))
            };
            dictionary(k, syms_dict, dict_bits, vw, dict, b.ranks32)?;
            // Ranks run from 0 (the column mode) to k, so all of them are
            // direct symbols only while `k < DIRECT`; otherwise the largest
            // rank falls into a length bucket and needs escape bits.
            let ranks = if kd < symbol::DIRECT as usize {
                debug_assert!(max_symbol(syms) as usize <= kd && rank_bits.is_empty(), "rank outside the dictionary");
                Ranks::U8(syms)
            } else {
                // SAFETY: the rank staging buffer holds `max_elems >= n`.
                let ranks = unsafe { b.ranks32.get_unchecked_mut(..n) };
                fields(k, syms, rank_bits, ranks);
                debug_assert!(ranks.iter().all(|&r| r as usize <= kd), "rank outside the dictionary");
                Ranks::U32(ranks)
            };
            Ok(Values::Modes(ModeDict {
                padded: b.dict,
                k: kd,
                vw,
                ranks,
                modes: table,
            }))
        }
        ValPart::Dict {
            k: kd,
            dict_len,
            dict_bits_len,
            rank,
            ..
        } => {
            // SAFETY: the pool holds the rejoined `pool_len` bytes, whose
            // dictionary stream and escape bits come first; the symbol and
            // dictionary buffers hold `max_k >= kd` and `max_elems >= n`.
            let (dict_bits, rank_bits, syms_dict, dict, syms) = unsafe {
                let pool = b.val_pool.get_unchecked(..part.pool_len());
                let (dict_bits, rank_bits) = pool.get_unchecked(dict_len..).split_at_unchecked(dict_bits_len);
                (dict_bits, rank_bits, b.syms_dict.get_unchecked(..kd), b.dict.get_unchecked_mut(..kd * vw), b.syms_val.get_unchecked(..n))
            };
            dictionary(k, syms_dict, dict_bits, vw, dict, b.ranks32)?;
            let ranks = match rank {
                None => Ranks::One,
                Some(_) if kd <= symbol::DIRECT as usize => {
                    // Every rank is its own symbol: no escapes.
                    debug_assert!((max_symbol(syms) as usize) < kd && rank_bits.is_empty(), "rank outside the dictionary");
                    Ranks::U8(syms)
                }
                Some(_) => {
                    // SAFETY: the rank staging buffer holds `max_elems >= n`.
                    let ranks = unsafe { b.ranks32.get_unchecked_mut(..n) };
                    fields(k, syms, rank_bits, ranks);
                    debug_assert!(ranks.iter().all(|&r| (r as usize) < kd), "rank outside the dictionary");
                    Ranks::U32(ranks)
                }
            };
            Ok(Values::Dict(Dict {
                padded: b.dict,
                k: kd,
                vw,
                ranks,
            }))
        }
    }
}

/// Parses a non-empty record and checks it fits the workspace.
fn view<'r>(meta: &Meta, rec: &'r [u8]) -> Result<RecordView<'r>> {
    let view = format::parse_record(
        rec,
        meta.n_dims(),
        meta.index_width(),
        meta.value_width(),
        |n| lanes_for(n, meta.max_lanes(), meta.lane_min()),
    )?;
    debug_assert!(view.nnz <= meta.max_nnz() && view.elems <= meta.max_elems(), "more elements than the chunk maximum");
    Ok(view)
}

/// Decodes a record into a parts view; the workspace is consumed so the
/// view can borrow from it for `'w`.
fn parts<'w, 'm: 'w>(
    cx: Ctx<'_, 'm>,
    rec: &'w [u8],
    space: IdSpace,
    mut ws: DecodeWs<'w>,
) -> Result<Parts<'w>> {
    if rec.is_empty() {
        return Ok(Parts {
            ids: &[],
            values: Values::Raw(&[]),
            nnz: 0,
            dense: false,
        });
    }
    let vw = cx.meta.value_width();
    let view = view(cx.meta, rec)?;
    let (n, nnz, dense) = (view.elems, view.nnz, view.dense);
    streams(cx, &view, &mut ws)?;
    let bits_len = match view.idx {
        IdxPart::Gap { bits_len, .. } => bits_len,
        IdxPart::Raw(_) | IdxPart::Dense => 0,
    };
    let DecodeWs {
        marks,
        slot,
        idx_bits,
        val_pool,
        dict,
        syms_idx,
        syms_val,
        syms_dict,
        coded,
        sorted,
        ranks32,
        vstage,
        ..
    } = ws;
    // A dense record's ids are `0..n`: shared, not written, when they fit.
    let shared = if dense { iota(n) } else { None };
    // SAFETY: every workspace buffer below is sized for `max_elems >= n`
    // elements and for the escape bits of any well-formed record.
    let coded: &'w mut [u32] = unsafe { coded.get_unchecked_mut(..n) };
    if shared.is_none() {
        // SAFETY: as above.
        unsafe { coded_ids(cx, &view.idx, syms_idx.get_unchecked(..n), idx_bits.get_unchecked(..bits_len), coded) };
    }
    let bufs = ValueBufs {
        syms_dict,
        val_pool,
        syms_val,
        dict,
        ranks32,
    };
    let values = values(cx, &view.val, n, bufs)?;
    let soc = match (space, cx.t.source_of_coded()) {
        (IdSpace::Source, Some(soc)) => soc,
        _ => {
            return Ok(Parts {
                ids: shared.unwrap_or(coded),
                values,
                nnz,
                dense,
            });
        }
    };
    restore(cx.k, soc, shared.unwrap_or(coded), marks, slot, sorted)?;
    // SAFETY: as above; the scan wrote `n` source ids.
    let (order, vstage) = unsafe { (sorted.get_unchecked(..n), vstage.get_unchecked_mut(..n * vw)) };
    // Values follow their genes into source order. `slot[s]` is the record
    // position of source gene `s`, so every lookup below is below `n`.
    let values = match values {
        Values::Raw(bytes) => {
            permute(vw, bytes, order, slot, vstage);
            Values::Raw(vstage)
        }
        Values::Modes(d) => {
            // Column modes are indexed by coded id: materialize in source order.
            for (o, &s) in vstage.chunks_exact_mut(vw).zip(order) {
                // SAFETY: source ids index `slot`; `t < n` is an element whose
                // coded id is `coded[t]` (or `t` for a dense record).
                unsafe {
                    let t = get(slot, s as usize) as usize;
                    o.copy_from_slice(d.value(t, shared.map_or_else(|| get(coded, t), |_| t as u32)));
                }
            }
            Values::Raw(vstage)
        }
        Values::Dict(d) => {
            let ranks = match d.ranks {
                Ranks::One => Ranks::One,
                Ranks::U8(r) => {
                    // SAFETY: as above.
                    let out = unsafe { syms_idx.get_unchecked_mut(..n) };
                    for (o, &s) in out.iter_mut().zip(order) {
                        // SAFETY: source ids index `slot`, whose entries are below `n`.
                        *o = unsafe { get(r, get(slot, s as usize) as usize) };
                    }
                    Ranks::U8(out)
                }
                Ranks::U32(r) => {
                    // The coded ids are no longer needed: they hold the ranks.
                    for (o, &s) in coded.iter_mut().zip(order) {
                        // SAFETY: as above.
                        *o = unsafe { get(r, get(slot, s as usize) as usize) };
                    }
                    Ranks::U32(coded)
                }
            };
            Values::Dict(Dict { ranks, ..d })
        }
    };
    Ok(Parts {
        ids: order,
        values,
        nnz,
        dense,
    })
}

/// Bytes of element `t` (id `id`) of a values view.
///
/// # Safety
/// `t` is an element of the row and `id` its coded id.
#[inline(always)]
unsafe fn element<'a>(values: &Values<'a>, t: usize, id: u32, vw: usize) -> &'a [u8] {
    // SAFETY: values hold one entry per element; ranks name dictionary entries.
    unsafe {
        match *values {
            Values::Raw(bytes) => bytes.get_unchecked(t * vw..t * vw + vw),
            Values::Dict(d) => {
                let r = match d.ranks {
                    Ranks::One => 0,
                    Ranks::U8(r) => get(r, t) as usize,
                    Ranks::U32(r) => get(r, t) as usize,
                };
                d.padded.get_unchecked(r * d.vw..(r + 1) * d.vw)
            }
            Values::Modes(d) => d.value(t, id),
        }
    }
}

/// `out[j] = values[slot[order[j]]]` for `vw`-byte values.
fn permute(vw: usize, values: &[u8], order: &[u32], slot: &[u32], out: &mut [u8]) {
    fn run<const W: usize>(values: &[u8], order: &[u32], slot: &[u32], out: &mut [u8]) {
        for (o, &s) in out.chunks_exact_mut(W).zip(order) {
            // SAFETY: source ids index `slot`, whose entries are record
            // positions below the element count `values` holds.
            unsafe {
                let at = get(slot, s as usize) as usize * W;
                o.copy_from_slice(values.get_unchecked(at..at + W));
            }
        }
    }
    match vw {
        1 => run::<1>(values, order, slot, out),
        2 => run::<2>(values, order, slot, out),
        4 => run::<4>(values, order, slot, out),
        _ => run::<8>(values, order, slot, out),
    }
}

/// One record into CSR slices of at least its nnz; returns the nnz.
fn record_csr<I: IndexOut>(
    cx: Ctx,
    rec: &[u8],
    space: IdSpace,
    idx: &mut [I],
    val: &mut [u8],
    ws: DecodeWs,
) -> Result<usize> {
    let vw = cx.meta.value_width();
    let p = parts(cx, rec, space, ws)?;
    let n = p.nnz();
    if idx.len() < n || val.len() < n * vw {
        return Err(Error::OutputTooSmall {
            need: n,
            have: idx.len().min(val.len() / vw),
        });
    }
    if p.dense {
        // A dense record's zeros are not CSR elements; it declares `n`
        // non-zeros, which the outputs hold.
        let mut j = 0;
        for (t, &c) in p.ids.iter().enumerate() {
            // SAFETY: `t` is an element with id `c`; `j < n` for a record
            // that declares its non-zeros truthfully.
            unsafe {
                let v = element(&p.values, t, c, vw);
                if v.iter().any(|&b| b != 0) {
                    debug_assert!(j < n, "dense record has more non-zeros than it declares");
                    set(idx, j, I::from_id(c));
                    val.get_unchecked_mut(j * vw..(j + 1) * vw).copy_from_slice(v);
                    j += 1;
                }
            }
        }
        debug_assert_eq!(j, n, "dense record has fewer non-zeros than it declares");
        return Ok(n);
    }
    for (o, &c) in idx.iter_mut().zip(p.ids) {
        *o = I::from_id(c);
    }
    // SAFETY: `val` holds `n * vw` bytes (checked above).
    expand(cx.k, &p.values, p.ids, unsafe { val.get_unchecked_mut(..n * vw) })?;
    Ok(n)
}

/// One record into a dense row of `n_genes` values. Order does not matter
/// for a scatter, so the record is never reordered: source columns come
/// straight from the gene order.
fn record_dense(
    cx: Ctx,
    rec: &[u8],
    space: IdSpace,
    row: &mut [u8],
    fill: Fill,
    ws: DecodeWs,
) -> Result<()> {
    if fill == Fill::Zero {
        row.fill(0);
    }
    let p = parts(cx, rec, IdSpace::Coded, ws)?;
    let soc = match space {
        IdSpace::Source => cx.t.source_of_coded(),
        IdSpace::Coded => None,
    };
    // SAFETY: coded ids are below `n_genes`, which bounds the gene order and
    // the row (`check_row` holds it to `n_genes` values).
    unsafe {
        match soc {
            Some(soc) => scatter(cx.meta.value_width(), &p, |c| get(soc, c as usize), row),
            None => scatter(cx.meta.value_width(), &p, |c| c, row),
        }
    }
    Ok(())
}

/// `row[col(id)] = value` over a parts view.
///
/// # Safety
/// Every column `col` gives lies inside the row of `vw`-byte values.
#[inline(always)]
unsafe fn scatter(vw: usize, p: &Parts, col: impl Fn(u32) -> u32, row: &mut [u8]) {
    #[inline(always)]
    unsafe fn run<const W: usize>(p: &Parts, col: impl Fn(u32) -> u32, row: &mut [u8]) {
        let base = row.as_mut_ptr();
        debug_assert!(p.ids.iter().all(|&c| (col(c) as usize + 1) * W <= row.len()));
        // SAFETY: every column lies inside the row (the caller's contract);
        // values hold one entry per element and ranks name entries.
        unsafe {
            let put = |c: u32, v: *const u8| std::ptr::copy_nonoverlapping(v, base.add(col(c) as usize * W), W);
            match p.values {
                Values::Raw(bytes) => p.ids.iter().enumerate().for_each(|(t, &c)| put(c, bytes.as_ptr().add(t * W))),
                Values::Modes(d) => p.ids.iter().enumerate().for_each(|(t, &c)| put(c, d.value(t, c).as_ptr())),
                Values::Dict(d) => {
                    let e = d.padded.as_ptr();
                    match d.ranks {
                        Ranks::One => p.ids.iter().for_each(|&c| put(c, e)),
                        Ranks::U8(r) => p.ids.iter().zip(r).for_each(|(&c, &k)| put(c, e.add(k as usize * W))),
                        Ranks::U32(r) => p.ids.iter().zip(r).for_each(|(&c, &k)| put(c, e.add(k as usize * W))),
                    }
                }
            }
        }
    }
    // SAFETY: as documented.
    unsafe {
        match vw {
            1 => run::<1>(p, col, row),
            2 => run::<2>(p, col, row),
            4 => run::<4>(p, col, row),
            _ => run::<8>(p, col, row),
        }
    }
}

/// A contiguous view of a paged record: the page itself when the record
/// lies in one page, else a copy into `stage`.
fn resolve<'w>(rec: PagedRecord<'w>, stage: &'w mut [u8]) -> Result<&'w [u8]> {
    let PagedRecord {
        pages,
        page_size,
        start,
        len,
    } = rec;
    if len == 0 {
        return Ok(&[]);
    }
    if start >= page_size {
        return Err(Error::InvalidInput(
            "record start must lie inside the first page",
        ));
    }
    let span = start
        .checked_add(len)
        .ok_or(Error::InvalidInput("record extent overflows"))?;
    if span.div_ceil(page_size) > pages.len() {
        return Err(Error::InvalidInput("record extends past its pages"));
    }
    // A record inside one page is borrowed; a longer one is copied into the
    // workspace's staging buffer, since only its pages are contiguous.
    let short = Error::InvalidInput("page shorter than the page size");
    if span <= page_size {
        return pages
            .first()
            .map_or(Ok(&[][..]), |p| p.get(start..span).ok_or(short));
    }
    let out = stage
        .get_mut(..len)
        .ok_or(Error::CorruptRecord("record longer than the chunk allows"))?;
    let (mut at, mut from) = (0, start);
    for page in pages {
        if at == len {
            break;
        }
        let take = (page_size - from).min(len - at);
        let piece = page.get(from..from + take).ok_or(short.clone())?;
        out[at..at + take].copy_from_slice(&piece[..take]);
        at += take;
        from = 0;
    }
    Ok(out)
}

fn context<'c, 'm>(meta: &'c Meta<'m>, tables: &'c DecodeTables) -> Result<Ctx<'c, 'm>> {
    tables.check(meta)?;
    Ok(bound(meta, tables))
}

/// The context of tables already checked against `meta`.
#[inline]
fn bound<'c, 'm>(meta: &'c Meta<'m>, tables: &'c DecodeTables) -> Ctx<'c, 'm> {
    Ctx { k: tables.kernels, meta, t: tables }
}

/// Bound metadata, tables, ISA dispatch and workspace layout for repeated records.
///
/// Construction checks the tables against `meta` once; every method then
/// decodes with the bound kernel set and the layout of that chunk, so a
/// caller looping over a chunk's records avoids re-deriving either.
pub struct Decoder<'c, 'm> { cx: Ctx<'c, 'm>, layout: DecodeLayout }
impl<'c, 'm> Decoder<'c, 'm> {
    /// Binds `meta` and its tables; fails when they do not belong together.
    pub fn new(meta: &'c Meta<'m>, tables: &'c DecodeTables) -> Result<Self> {
        Ok(Self { cx: context(meta, tables)?, layout: layout(meta) })
    }
    /// Workspace words every method needs ([`decode_workspace_words`]).
    pub fn workspace_words(&self) -> usize { self.layout.total() }
    /// [`decode_parts`] with this decoder's binding.
    pub fn parts<'w>(&self, record: &'w [u8], space: IdSpace, workspace: &'w mut [u64]) -> Result<Parts<'w>> where 'm: 'w {
        parts(self.cx, record, space, self.layout.split(workspace)?)
    }
    /// [`decode_parts_pages`] with this decoder's binding.
    pub fn parts_pages<'w>(&self, record: PagedRecord<'w>, space: IdSpace, workspace: &'w mut [u64]) -> Result<Parts<'w>> where 'm: 'w {
        let mut ws = self.layout.split(workspace)?;
        let record = resolve(record, std::mem::take(&mut ws.stage))?;
        parts(self.cx, record, space, ws)
    }
    /// [`decode_record_dense`] with this decoder's binding.
    pub fn dense(&self, record: &[u8], space: IdSpace, out: &mut [u8], fill: Fill, workspace: &mut [u64]) -> Result<()> {
        check_row(self.cx.meta, out)?;
        record_dense(self.cx, record, space, out, fill, self.layout.split(workspace)?)
    }
    /// [`decode_record`] with this decoder's binding; returns the nnz.
    pub fn sparse<I: IndexOut>(&self, record: &[u8], space: IdSpace, indices: &mut [I], values: &mut [u8], workspace: &mut [u64]) -> Result<usize> {
        check_index_type::<I>(self.cx.meta)?;
        record_csr(self.cx, record, space, indices, values, self.layout.split(workspace)?)
    }
}

/// Decodes one record (from a chunk with this metadata, read however the
/// caller likes) into a [`Parts`] view borrowing the workspace and record.
pub fn decode_parts<'w, 'm: 'w>(
    meta: &Meta<'m>,
    tables: &DecodeTables,
    rec: &'w [u8],
    space: IdSpace,
    ws: &'w mut [u64],
) -> Result<Parts<'w>> {
    let cx = context(meta, tables)?;
    parts(cx, rec, space, layout(meta).split(ws)?)
}

/// [`decode_parts`] for a record held in pages.
pub fn decode_parts_pages<'w, 'm: 'w>(
    meta: &Meta<'m>,
    tables: &DecodeTables,
    rec: PagedRecord<'w>,
    space: IdSpace,
    ws: &'w mut [u64],
) -> Result<Parts<'w>> {
    let cx = context(meta, tables)?;
    let mut dws = layout(meta).split(ws)?;
    let rec = resolve(rec, std::mem::take(&mut dws.stage))?;
    parts(cx, rec, space, dws)
}

/// [`decode_parts_pages`] for tables the caller checked against `meta`
/// when it bound them (one check per chunk instead of per record).
pub(crate) fn decode_parts_pages_prechecked<'w, 'm: 'w>(
    meta: &Meta<'m>,
    tables: &DecodeTables,
    rec: PagedRecord<'w>,
    space: IdSpace,
    ws: &'w mut [u64],
) -> Result<Parts<'w>> {
    debug_assert!(tables.check(meta).is_ok());
    let mut dws = layout(meta).split(ws)?;
    let rec = resolve(rec, std::mem::take(&mut dws.stage))?;
    parts(bound(meta, tables), rec, space, dws)
}

/// Decodes one record into `indices` and `values`, which must hold at
/// least its nnz entries (see [`Meta::cell_nnz`] or
/// [`format::record_nnz`]). Indices ascend in `space`. Returns the nnz.
pub fn decode_record<I: IndexOut>(
    meta: &Meta,
    tables: &DecodeTables,
    rec: &[u8],
    space: IdSpace,
    indices: &mut [I],
    values: &mut [u8],
    ws: &mut [u64],
) -> Result<usize> {
    let cx = context(meta, tables)?;
    check_index_type::<I>(meta)?;
    record_csr(cx, rec, space, indices, values, layout(meta).split(ws)?)
}

/// [`decode_record`] for a record held in pages.
#[allow(clippy::too_many_arguments)]
pub fn decode_record_pages<I: IndexOut>(
    meta: &Meta,
    tables: &DecodeTables,
    rec: PagedRecord,
    space: IdSpace,
    indices: &mut [I],
    values: &mut [u8],
    ws: &mut [u64],
) -> Result<usize> {
    let cx = context(meta, tables)?;
    check_index_type::<I>(meta)?;
    let mut dws = layout(meta).split(ws)?;
    let rec = resolve(rec, std::mem::take(&mut dws.stage))?;
    record_csr(cx, rec, space, indices, values, dws)
}

/// Decodes one record into a dense row of `n_genes × value_width` bytes;
/// columns follow `space`.
pub fn decode_record_dense(
    meta: &Meta,
    tables: &DecodeTables,
    rec: &[u8],
    space: IdSpace,
    row: &mut [u8],
    fill: Fill,
    ws: &mut [u64],
) -> Result<()> {
    let cx = context(meta, tables)?;
    check_row(meta, row)?;
    record_dense(cx, rec, space, row, fill, layout(meta).split(ws)?)
}

/// [`decode_record_dense`] for a record held in pages.
pub fn decode_record_dense_pages(
    meta: &Meta,
    tables: &DecodeTables,
    rec: PagedRecord,
    space: IdSpace,
    row: &mut [u8],
    fill: Fill,
    ws: &mut [u64],
) -> Result<()> {
    let cx = context(meta, tables)?;
    check_row(meta, row)?;
    let mut dws = layout(meta).split(ws)?;
    let rec = resolve(rec, std::mem::take(&mut dws.stage))?;
    record_dense(cx, rec, space, row, fill, dws)
}

fn check_row(meta: &Meta, row: &[u8]) -> Result<()> {
    let need = meta.n_dims() as usize * meta.value_width();
    if row.len() != need {
        Err(Error::OutputTooSmall {
            need,
            have: row.len(),
        })
    } else {
        Ok(())
    }
}

/// Decodes the selected cells of a resident chunk into CSR arrays:
/// `indptr` of `rows + 1` entries, `indices` and `values` of at least
/// [`selection_nnz`] entries. Indices ascend in `space` within each row.
/// Returns the nnz written.
#[allow(clippy::too_many_arguments)]
pub fn decode_csr<P: IndptrOut, I: IndexOut>(
    meta: &Meta,
    tables: &DecodeTables,
    records: &[u8],
    rows: Rows,
    space: IdSpace,
    indptr: &mut [P],
    indices: &mut [I],
    values: &mut [u8],
    ws: &mut [u64],
) -> Result<u64> {
    let cx = context(meta, tables)?;
    check_index_type::<I>(meta)?;
    let n = rows.count(meta.n_cells())?;
    let total = selection_nnz(meta, rows)?;
    if total > P::MAX {
        return Err(Error::OutputTypeTooNarrow(total));
    }
    let vw = meta.value_width();
    if indptr.len() != n + 1 {
        return Err(Error::OutputTooSmall {
            need: n + 1,
            have: indptr.len(),
        });
    }
    if (indices.len() as u64) < total || (values.len() as u64) < total * vw as u64 {
        return Err(Error::OutputTooSmall {
            need: total as usize,
            have: indices.len().min(values.len() / vw),
        });
    }
    let layout = layout(meta);
    layout.split(ws)?;
    indptr[0] = P::from_offset(0);
    let mut off = 0usize;
    for i in 0..n {
        let cell = rows.get(i);
        let nnz = meta.cell_nnz(cell)?;
        let rec = meta.record(records, cell)?;
        let got = record_csr(
            cx,
            rec,
            space,
            &mut indices[off..off + nnz],
            &mut values[off * vw..(off + nnz) * vw],
            layout.split(ws)?,
        )?;
        if got != nnz {
            return Err(Error::CorruptRecord(
                "record nnz differs from the non-zero table",
            ));
        }
        off += nnz;
        indptr[i + 1] = P::from_offset(off as u64);
    }
    Ok(off as u64)
}

/// Decodes the selected cells of a resident chunk into dense rows
/// (`rows × n_genes × value_width` bytes, see [`dense_len`]).
#[allow(clippy::too_many_arguments)]
pub fn decode_dense(
    meta: &Meta,
    tables: &DecodeTables,
    records: &[u8],
    rows: Rows,
    space: IdSpace,
    out: &mut [u8],
    fill: Fill,
    ws: &mut [u64],
) -> Result<()> {
    let cx = context(meta, tables)?;
    let need = dense_len(meta, rows)?;
    if out.len() != need {
        return Err(Error::OutputTooSmall {
            need,
            have: out.len(),
        });
    }
    let n = rows.count(meta.n_cells())?;
    let width = meta.n_dims() as usize * meta.value_width();
    let layout = layout(meta);
    layout.split(ws)?;
    for i in 0..n {
        let cell = rows.get(i);
        let nnz = meta.cell_nnz(cell)?;
        let rec = meta.record(records, cell)?;
        if format::record_nnz(rec)? != nnz {
            return Err(Error::CorruptRecord(
                "record nnz differs from the non-zero table",
            ));
        }
        record_dense(
            cx,
            rec,
            space,
            &mut out[i * width..(i + 1) * width],
            fill,
            layout.split(ws)?,
        )?;
    }
    Ok(())
}
