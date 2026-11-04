//! The plan ring: finalized per-access plan entries, written by one
//! producer (an online compiler, the replay of a plan file) and read by the
//! workers below its published frontier `P`.
//!
//! An entry is a head (where its words lie in the word pool, and flags)
//! followed by words:
//!
//! ```text
//! refs     one per page of the access's record, when REFS is set
//! extras   per extra admission read: (file << 32 | pages), offset, refs
//! row      when ROW_USE or ROW_ADMIT: (frames << 32 | generation), frames packed two per word
//! ```
//!
//! A ref is `frame | generation << 32 | admit << 63`, or [`MISS`]. An admit
//! ref is read by this access into its cache frame, which then publishes
//! the generation; a use ref waits for that generation. The slot of access
//! `a` and its words are reused once the execution frontier `F` passes `a`;
//! the producer writes a piece of a range, then advances `P` once.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};

use crossbeam_utils::CachePadded;

/// Page refs follow the head, one per record page.
pub(crate) const REFS: u32 = 1;
/// Same cell as the previous access of its position: the worker reuses that
/// access's input; the entry holds no page refs.
pub(crate) const SAME: u32 = 2;
/// Served by a cached row.
pub(crate) const ROW_USE: u32 = 4;
/// Builds the cell's cached row into its frames.
pub(crate) const ROW_ADMIT: u32 = 8;
/// Number of extra admission reads in the flags' remaining bits.
const EXTRAS_SHIFT: u32 = 8;

/// A page ref that reads nothing: the page is a miss into a private frame.
pub(crate) const MISS: u64 = u32::MAX as u64;
/// Bits of the generation field of a page ref, also the row generation's mask.
pub(crate) const GEN_MASK: u32 = 0x7fff_ffff;
/// Top bit of an admission ref: this access reads the page into the frame.
const ADMIT_BIT: u64 = 1 << 63;

/// Packs a page ref: frame id, the generation the admitter publishes, and the
/// admit bit. A use ref waits for `generation`; an admit ref publishes it.
#[inline(always)]
pub(crate) fn page_ref(frame: u32, generation: u32, admit: bool) -> u64 {
    frame as u64 | ((generation & GEN_MASK) as u64) << 32 | if admit { ADMIT_BIT } else { 0 }
}
/// The frame of a page ref.
#[inline(always)]
pub(crate) fn ref_frame(r: u64) -> u32 {
    r as u32
}
/// The generation of a page ref, in [`GEN_MASK`] bits: the value the admitting
/// access publishes into the frame's generation word, and the value a use
/// waits for.
#[inline(always)]
pub(crate) fn ref_gen(r: u64) -> u32 {
    (r >> 32) as u32 & GEN_MASK
}
/// Whether a page ref admits its page into the frame.
#[inline(always)]
pub(crate) fn ref_admit(r: u64) -> bool {
    r & ADMIT_BIT != 0
}

/// An entry's head: where its words are, and how to read them.
///
/// Nothing in an entry points at an address: `pool` is the producer buffer's
/// word index on the way in and the ring pool's monotonic word position on the
/// way out, so both sides address the pool modulo its length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Head {
    /// First word: an offset into the producer's buffer, then a monotonic
    /// word position in the ring's pool.
    pub pool: u64,
    /// Words of the entry, after the head.
    pub words: u32,
    /// [`REFS`], [`SAME`], [`ROW_USE`], [`ROW_ADMIT`], and the extra-read count
    /// in the bits above [`EXTRAS_SHIFT`].
    pub flags: u32,
    /// The cell the entry was planned for. Workers compare it with the cell
    /// their own evaluation of the sequence produced before following any
    /// frame id in the entry.
    pub cell: u32,
}

/// An extra admission read: consecutive pages of one file, none of them the
/// access's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extra {
    /// File id, the index the catalog gives each opened input file.
    pub file: u32,
    /// Byte offset of the first page.
    pub offset: u64,
    /// Pages read, one admission ref each.
    pub pages: u32,
}

/// Entries of consecutive accesses, as a producer builds them: heads whose
/// `pool` indexes the producer's own `words`, so a buffer's entries are
/// relocatable into the ring as a unit.
#[derive(Default)]
pub(crate) struct EntryBuf {
    /// Access index of the first entry.
    pub first: u64,
    /// One head per access, in access order.
    pub heads: Vec<Head>,
    /// Every entry's words, packed back to back; a head's `pool` indexes it.
    pub words: Vec<u64>,
}

/// One access's plan, as an [`EntryBuf`] takes it.
pub(crate) struct AccessPlan<'a> {
    /// The cell the access loads.
    pub cell: u32,
    /// The access loads the same cell as the previous access of its position.
    pub same: bool,
    /// One per record page when the access reads its record; empty for a
    /// same-cell access, a row hit, or a record whose pages are all misses.
    pub refs: &'a [u64],
    /// Extra admission reads with their refs, in order.
    pub extras: &'a [(Extra, &'a [u64])],
    /// `(admit, generation, frames)` of the cell's cached row.
    pub row: Option<(bool, u32, &'a [u32])>,
}

impl EntryBuf {
    /// Empties the buffer for entries from access `first`.
    pub fn clear(&mut self, first: u64) {
        self.first = first;
        self.heads.clear();
        self.words.clear();
    }

    /// Entries buffered.
    pub fn len(&self) -> u64 {
        self.heads.len() as u64
    }

    /// Appends one access's entry: the head's flags, then refs, extra reads
    /// with their refs, and the row's frames packed two per word.
    pub fn push(&mut self, a: &AccessPlan<'_>) {
        let at = self.words.len();
        let mut flags = 0;
        if a.same {
            flags |= SAME;
        } else if !a.refs.is_empty() {
            flags |= REFS;
            self.words.extend_from_slice(a.refs);
        }
        debug_assert!(a.extras.len() < 1 << (32 - EXTRAS_SHIFT));
        for (e, refs) in a.extras {
            debug_assert_eq!(refs.len(), e.pages as usize);
            self.words.push((e.file as u64) << 32 | e.pages as u64);
            self.words.push(e.offset);
            self.words.extend_from_slice(refs);
        }
        flags |= (a.extras.len() as u32) << EXTRAS_SHIFT;
        if let Some((admit, generation, frames)) = a.row {
            flags |= if admit { ROW_ADMIT } else { ROW_USE };
            self.words.push((frames.len() as u64) << 32 | (generation & GEN_MASK) as u64);
            for pair in frames.chunks(2) {
                self.words.push(pair[0] as u64 | (pair.get(1).copied().unwrap_or(0) as u64) << 32);
            }
        }
        self.heads.push(Head { pool: at as u64, words: (self.words.len() - at) as u32, flags, cell: a.cell });
    }

    /// Appends `o` (the entries right after this buffer's).
    pub fn append(&mut self, o: &EntryBuf) {
        debug_assert_eq!(self.first + self.len(), o.first);
        let base = self.words.len() as u64;
        self.heads.extend(o.heads.iter().map(|h| Head { pool: h.pool + base, ..*h }));
        self.words.extend_from_slice(&o.words);
    }
}

/// Extra reads with their refs, and the row `(admit, generation, frames)`,
/// of a decoded entry.
pub(crate) type Rest<'a> = (Extras<'a>, Option<(bool, u32, RowFrames<'a>)>);

/// An entry's extra admission reads with their refs, in entry order.
#[derive(Clone)]
pub(crate) struct Extras<'a> {
    words: &'a [u64],
    remaining: usize,
}
impl<'a> Iterator for Extras<'a> {
    type Item = (Extra, &'a [u64]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 { return None; }
        let h = self.words[0];
        let e = Extra { file: (h >> 32) as u32, pages: h as u32, offset: self.words[1] };
        let end = 2 + e.pages as usize;
        let refs = &self.words[2..end];
        self.words = &self.words[end..];
        self.remaining -= 1;
        Some((e, refs))
    }
}

/// Packed frame ids borrowed from an entry; the worker copies them into its
/// reusable task buffer.
pub(crate) struct RowFrames<'a> {
    words: &'a [u64],
    len: usize,
}
impl RowFrames<'_> {
    /// The frame ids in packed order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        (0..self.len).map(|i| (self.words[i / 2] >> (32 * (i % 2))) as u32)
    }
}

/// A decoded entry, borrowing a worker's buffer.
pub(crate) struct Entry<'a> {
    /// The head's flags.
    pub flags: u32,
    /// The entry's words, head excluded.
    pub words: &'a [u64],
}

impl<'a> Entry<'a> {
    /// Page refs of a record of `pages` pages (empty without REFS).
    pub fn refs(&self, pages: usize) -> &'a [u64] {
        if self.flags & REFS != 0 { &self.words[..pages] } else { &[] }
    }

    /// Extra reads and the row, after the refs of a `pages`-page record.
    pub fn rest(&self, pages: usize) -> Rest<'a> {
        let mut at = if self.flags & REFS != 0 { pages } else { 0 };
        let n = (self.flags >> EXTRAS_SHIFT) as usize;
        let extras = Extras { words: &self.words[at..], remaining: n };
        for (e, _) in extras.clone() {
            at += 2 + e.pages as usize;
        }
        let row = (self.flags & (ROW_USE | ROW_ADMIT) != 0).then(|| {
            let h = self.words[at];
            let n = (h >> 32) as usize;
            let frames = RowFrames { words: &self.words[at + 1..at + 1 + n.div_ceil(2)], len: n };
            (self.flags & ROW_ADMIT != 0, h as u32 & GEN_MASK, frames)
        });
        (extras, row)
    }
}

/// The bounded ring of published entries.
pub(crate) struct PlanRing {
    /// One head per access, indexed by the access index modulo its length.
    heads: Box<[UnsafeCell<Head>]>,
    /// The word pool, addressed modulo its length.
    pool: Box<[UnsafeCell<u64>]>,
    /// `P`: entries below are published.
    published: CachePadded<AtomicU64>,
    /// Writer-only: the pool's next word.
    pool_end: UnsafeCell<u64>,
    stopped: AtomicBool,
}

// SAFETY: one producer writes slots no reader may read (access indices at or
// above `P`, or below `F`); readers read only slots of access indices in
// `F..P`, ordered by the release store of `P`.
unsafe impl Sync for PlanRing {}
// SAFETY: as above.
unsafe impl Send for PlanRing {}

impl PlanRing {
    /// `entries` slots (at least 1) and `words` pool words (at least 64).
    pub fn new(entries: u64, words: u64) -> Self {
        let entries = entries.max(1) as usize;
        let words = words.max(64) as usize;
        Self {
            heads: (0..entries).map(|_| UnsafeCell::new(Head::default())).collect(),
            pool: (0..words).map(|_| UnsafeCell::new(0)).collect(),
            published: CachePadded::new(AtomicU64::new(0)),
            pool_end: UnsafeCell::new(0),
            stopped: AtomicBool::new(false),
        }
    }

    /// Entries a producer keeps published ahead of the frontier when it
    /// publishes on demand: half the ring, so a piece never waits for itself.
    pub fn capacity(&self) -> u64 {
        self.heads.len() as u64 / 2
    }

    /// `P`, the frontier below which entries are published: an access at or
    /// above it has no entry yet.
    #[inline(always)]
    pub fn ready(&self) -> u64 {
        self.published.load(Acquire)
    }

    /// Stops publication; `publish` then returns false and the run unwinds.
    pub fn stop(&self) {
        self.stopped.store(true, Release);
    }
    /// Whether the ring was stopped.
    pub fn stopped(&self) -> bool {
        self.stopped.load(Acquire)
    }

    /// The entry of the access at `position` (below [`Self::ready`], at or
    /// above the frontier), its words copied into `out`; returns its flags
    /// and the cell it was planned for.
    #[inline]
    pub fn read(&self, position: u64, out: &mut Vec<u64>) -> (u32, u32) {
        let n = self.heads.len() as u64;
        // SAFETY: the slot holds the published entry of `position` (see the
        // type's contract).
        let h = unsafe { *self.heads[(position % n) as usize].get() };
        out.clear();
        let len = self.pool.len() as u64;
        for i in 0..h.words as u64 {
            // SAFETY: as above, for the entry's words.
            out.push(unsafe { *self.pool[((h.pool + i) % len) as usize].get() });
        }
        (h.flags, h.cell)
    }

    /// Publishes `buf` (entries from `P`): waits for room while `frontier`
    /// (the execution frontier) lags, in pieces of at most half the slots and
    /// half the pool. Returns false if the run stopped.
    pub fn publish(&self, buf: &EntryBuf, frontier: &dyn Fn() -> u64, mut wait: impl FnMut()) -> bool {
        let n = self.heads.len() as u64;
        let len = self.pool.len() as u64;
        debug_assert_eq!(buf.first, self.published.load(Relaxed));
        let mut at = 0usize;
        while at < buf.heads.len() {
            // A piece fits half the ring, so it never waits for itself.
            let mut end = at;
            let mut words = 0u64;
            while end < buf.heads.len() && ((end - at) as u64) < n.div_ceil(2) && words + buf.heads[end].words as u64 <= len / 2 {
                words += buf.heads[end].words as u64;
                end += 1;
            }
            if end == at {
                // One entry larger than half the pool: the ring was sized
                // below the largest entry.
                end = at + 1;
                words = buf.heads[at].words as u64;
                assert!(words <= len, "plan entry larger than the plan ring pool");
            }
            let from = buf.first + at as u64;
            // SAFETY: only the producer touches `pool_end`.
            let pool_end = unsafe { *self.pool_end.get() };
            loop {
                if self.stopped() {
                    return false;
                }
                let f = frontier().min(from);
                // SAFETY: the slot of `f` is published and not yet reclaimed
                // (`f` is at or above the frontier).
                let pool_start = if f < from { unsafe { (*self.heads[(f % n) as usize].get()).pool } } else { pool_end };
                if from + (end - at) as u64 - f <= n && pool_end + words - pool_start <= len {
                    break;
                }
                wait();
            }
            let mut w = pool_end;
            for i in at..end {
                let h = buf.heads[i];
                let p = buf.first + i as u64;
                for k in 0..h.words as u64 {
                    // SAFETY: the word's previous entry is below the
                    // frontier (checked above), and no reader reads past `P`.
                    unsafe { *self.pool[((w + k) % len) as usize].get() = buf.words[(h.pool + k) as usize] };
                }
                // SAFETY: as above, for the slot.
                unsafe { *self.heads[(p % n) as usize].get() = Head { pool: w, ..h } };
                w += h.words as u64;
            }
            // SAFETY: producer-only.
            unsafe { *self.pool_end.get() = w };
            self.published.store(buf.first + end as u64, Release);
            at = end;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_round_trip_and_wrap() {
        let ring = PlanRing::new(4, 64);
        let f = std::sync::atomic::AtomicU64::new(0);
        let mut buf = EntryBuf::default();
        let mut out = Vec::new();
        for round in 0..10u64 {
            buf.clear(round * 3);
            for i in 0..3u64 {
                let refs = [page_ref(i as u32, 7, i == 0), MISS];
                let extra = [page_ref(9, 1, true)];
                let extras = [(Extra { file: 2, offset: 4096 * round, pages: 1 }, &extra[..])];
                let frames = [5u32, 6, 7];
                buf.push(&AccessPlan { cell: i as u32, same: false, refs: &refs, extras: if i == 1 { &extras } else { &[] }, row: (i == 2).then_some((true, 3, &frames[..])) });
            }
            f.store(round * 3, Relaxed);
            assert!(ring.publish(&buf, &|| f.load(Relaxed), || {}));
            assert_eq!(ring.ready(), round * 3 + 3);
            for i in 0..3u64 {
                let (flags, cell) = ring.read(round * 3 + i, &mut out);
                assert_eq!(cell, i as u32);
                let e = Entry { flags, words: &out };
                let refs = e.refs(2);
                assert_eq!(ref_frame(refs[0]), i as u32);
                assert_eq!(ref_gen(refs[0]), 7);
                assert_eq!(ref_admit(refs[0]), i == 0);
                assert_eq!(refs[1], MISS);
                let (extras, row) = e.rest(2);
                let extras: Vec<_> = extras.collect();
                assert_eq!(extras.len(), usize::from(i == 1));
                if i == 1 {
                    assert_eq!(extras[0].0, Extra { file: 2, offset: 4096 * round, pages: 1 });
                }
                assert_eq!(row.map(|(a, g, f)| (a, g, f.iter().collect::<Vec<_>>())), (i == 2).then(|| (true, 3, vec![5, 6, 7])));
            }
        }
    }
}
