//! Access sequences: which cell each access loads.
//!
//! A loader with `S` schemas runs `S` accesses per position, in the order
//! `p0s0 p0s1 … p1s0 …`: access `k` outputs schema `k % S` of position
//! `k / S`. A sequence names the cell of every access; [`Replicate`] gives
//! every schema of a position the same cell, [`Offset`] shifts each
//! schema's cell by its own offset (paired matrices registered in the same
//! order, say).
//!
//! A sequence is a pure function of the access, evaluated over ranges in
//! any order and concurrently: by the loader's workers for the accesses
//! they claim and by the online compiler ahead of them. Hundreds of billions
//! of accesses never need to be stored.

use std::ffi::c_void;
use std::cell::RefCell;
use std::sync::Arc;

use super::catalog::CellId;
use super::error::{Error, Result, config};

/// A sequence of cells: a pure function from access index to [`CellId`].
///
/// Implementations must be thread-safe and order-independent: `fill` is
/// called concurrently for different ranges, possibly more than once for the
/// same access, by workers and by the online compiler. The sequence is never
/// required to be materialized; implementations that hold one say so with
/// [`Accesses::as_slice`], which lets a plan bind to the values without a
/// copy.
pub trait Accesses: Send + Sync {
    /// Accesses in the sequence. With `S` schemas this must be a multiple of
    /// `S`: access `k` is schema `k % S` of position `k / S`.
    fn len(&self) -> u64;
    /// Whether the sequence has no accesses.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Writes the cells of accesses `start..start + out.len()`. Called
    /// concurrently, in any order, possibly more than once per access; the
    /// result must depend on the access only.
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()>;
    /// The whole sequence, when it is held in memory; a fixed plan is
    /// verified against the sequence's values, which `Loader::prepare`
    /// materializes when this is `None`.
    fn as_slice(&self) -> Option<&[CellId]> {
        None
    }
}

/// Stored sequences.
macro_rules! stored {
    ($($t:ty),*) => {$(
        impl Accesses for $t {
            fn len(&self) -> u64 {
                <[CellId]>::len(self) as u64
            }
            fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
                let n = <[CellId]>::len(out);
                let at = usize::try_from(start).ok().filter(|&a| a.checked_add(n).is_some_and(|e| e <= <[CellId]>::len(self))).ok_or_else(|| config("access range beyond the sequence"))?;
                out.copy_from_slice(&self[at..at + n]);
                Ok(())
            }
            fn as_slice(&self) -> Option<&[CellId]> {
                Some(self)
            }
        }
    )*};
}

stored!(Vec<CellId>, &[CellId], Box<[CellId]>, Arc<[CellId]>);

impl<const N: usize> Accesses for [CellId; N] {
    fn len(&self) -> u64 {
        N as u64
    }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        self.as_slice().fill(start, out)
    }
    fn as_slice(&self) -> Option<&[CellId]> {
        Some(self)
    }
}

/// Epochs of shuffled cells: each epoch visits every position (or every entry
/// of a subset) once, in its own pseudorandom order. With blocks, an epoch
/// shuffles blocks of consecutive cells, and shuffles cells within each
/// group of consecutive (shuffled) blocks: local reads, as from chunked
/// shuffle buffers.
#[derive(Clone)]
pub struct Shuffle {
    subset: Option<Arc<[CellId]>>,
    n: u64,
    epochs: u64,
    seed: u64,
    block: u64,
    group: u64,
}

impl Shuffle {
    /// Every cell of `0..cells`, `epochs` times.
    pub fn new(cells: u64, epochs: u64, seed: u64) -> Result<Self> {
        if cells == 0 || cells > CellId::MAX as u64 + 1 || epochs == 0 || cells.checked_mul(epochs).is_none() {
            return Err(config("a shuffle needs a positive cell count within u32 and a positive epoch count"));
        }
        Ok(Self { subset: None, n: cells, epochs, seed, block: 1, group: 1 })
    }

    /// The entries of `cells` (repeats allowed), `epochs` times.
    pub fn subset(cells: Vec<CellId>, epochs: u64, seed: u64) -> Result<Self> {
        let mut s = Self::new(cells.len() as u64, epochs, seed)?;
        s.subset = Some(cells.into());
        Ok(s)
    }

    /// Shuffles blocks of `block` consecutive entries, and entries within
    /// groups of `group` blocks; `block` is capped at the entry count.
    pub fn blocks(mut self, block: u64, group: u64) -> Result<Self> {
        if block == 0 || group == 0 || block.checked_mul(group).is_none() {
            return Err(config("blocks and groups must be positive"));
        }
        self.block = block.min(self.n);
        self.group = group;
        Ok(self)
    }

    /// Entry index of position `k` of `epoch`.
    fn entry(&self, epoch: u64, k: u64) -> u64 {
        let n = self.n;
        let key = mix(self.seed, epoch);
        if self.block == 1 {
            return permute(n, key, k);
        }
        let (block, group) = (self.block, self.group);
        let nb = n.div_ceil(block);
        let short = n % block;
        let full = group * block;
        // Layered permutation: the `nb` blocks are permuted, entries are
        // permuted across each group of `group` blocks (so a group reads a
        // contiguous span of cells), and a short last block travels with its
        // slot in the block order.
        // The short last block sits at slot `s` of the block order; groups
        // after its group start `deficit` entries earlier.
        let deficit = if short > 0 { block - short } else { 0 };
        let s = if short > 0 { unpermute(nb, mix(key, 1), nb - 1) } else { u64::MAX };
        let gs = s / group;
        // First entry of group `g`: every group but the last spans `full`
        // entries, and groups after the short block's group lose `deficit`.
        let start = |g: u64| (g * full).saturating_sub(if g > gs { deficit } else { 0 }).min(n);
        let g = if short > 0 && k >= start(gs + 1) { (k + deficit) / full } else { k / full };
        let (g0, g1) = (start(g), start(g + 1));
        let o = permute(g1 - g0, mix(key, 2 + g), k - g0);
        // Slot within the group and offset within its block.
        let si = if s / group == g { s - g * group } else { u64::MAX };
        let (q, off) = if si == u64::MAX || o < si * block {
            (o / block, o % block)
        } else if o < si * block + short {
            (si, o - si * block)
        } else {
            let r = o - si * block - short;
            (si + 1 + r / block, r % block)
        };
        permute(nb, mix(key, 1), g * group + q) * block + off
    }
}

impl Accesses for Shuffle {
    fn len(&self) -> u64 {
        self.n * self.epochs
    }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        if start.checked_add(out.len() as u64).is_none_or(|e| e > self.len()) {
            return Err(config("access range beyond the sequence"));
        }
        for (i, o) in out.iter_mut().enumerate() {
            let p = start + i as u64;
            let e = self.entry(p / self.n, p % self.n);
            *o = match &self.subset {
                Some(cells) => cells[e as usize],
                None => e as CellId,
            };
        }
        Ok(())
    }
}

/// `fn(context, start, count, out) -> status`: writes the cells of accesses
/// `start..start + count` to `out`; zero on success.
pub type AccessFn = unsafe extern "C" fn(*mut c_void, u64, u64, *mut CellId) -> i32;

/// Fills dataset indices and dataset-local rows for an access interval.
/// The output arrays have `count` entries and do not outlive the call.
pub type CorpusAccessFn = unsafe extern "C" fn(*mut c_void, u64, u64, *mut u32, *mut u64) -> i32;

/// A native corpus generator bound to the registered datasets' row counts.
pub struct ExternCorpusAccesses {
    f: CorpusAccessFn,
    context: *mut c_void,
    len: u64,
    offsets: Vec<u64>,
}

// SAFETY: the constructor requires concurrent, order-independent callbacks.
unsafe impl Send for ExternCorpusAccesses {}
// SAFETY: the same callback contract applies to shared references.
unsafe impl Sync for ExternCorpusAccesses {}

thread_local! {
    static CORPUS_ROWS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

impl ExternCorpusAccesses {
    /// Binds a corpus callback to dataset-local row counts.
    ///
    /// # Safety
    /// `f` and `context` must remain valid, be safe to call concurrently and
    /// write exactly `count` dataset indices and rows. Results depend only
    /// on the access position and immutable configuration.
    pub unsafe fn new(f: CorpusAccessFn, context: *mut c_void, len: u64, counts: &[u64]) -> Result<Self> {
        let mut offsets = vec![0u64];
        for &n in counts {
            let end = offsets.last().unwrap().checked_add(n).filter(|&n| n <= CellId::MAX as u64)
                .ok_or_else(|| config("corpus cell count exceeds the CellId range"))?;
            offsets.push(end);
        }
        Ok(Self { f, context, len, offsets })
    }

    fn fill_rows(&self, start: u64, out: &mut [CellId], rows: &mut Vec<u64>) -> Result<()> {
        rows.resize(out.len(), 0);
        // SAFETY: the constructor's callback contract covers these disjoint outputs.
        let status = unsafe { (self.f)(self.context, start, out.len() as u64, out.as_mut_ptr(), rows.as_mut_ptr()) };
        if status != 0 {
            return Err(Error::Accesses(status));
        }
        for (i, (dataset, &row)) in out.iter_mut().zip(rows.iter()).enumerate() {
            let d = *dataset as usize;
            let first = self.offsets.get(d).copied();
            let end = self.offsets.get(d + 1).copied();
            match (first, end) {
                (Some(first), Some(end)) if row < end - first => *dataset = (first + row) as CellId,
                _ => return Err(config(format!("access {} names dataset {} row {}, outside the corpus", start + i as u64, d, row))),
            }
        }
        Ok(())
    }
}

impl Accesses for ExternCorpusAccesses {
    fn len(&self) -> u64 { self.len }

    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        if start.checked_add(out.len() as u64).is_none_or(|end| end > self.len) {
            return Err(config("access range beyond the sequence"));
        }
        CORPUS_ROWS.with(|scratch| match scratch.try_borrow_mut() {
            Ok(mut rows) => self.fill_rows(start, out, &mut rows),
            Err(_) => self.fill_rows(start, out, &mut Vec::new()),
        })
    }
}

/// A bounded interval of another computed or stored access sequence.
pub struct SliceAccesses { inner: Arc<dyn Accesses>, start: u64, len: u64 }

impl SliceAccesses {
    pub fn new(inner: Arc<dyn Accesses>, start: u64, len: u64) -> Result<Self> {
        if start.checked_add(len).is_none_or(|end| end > inner.len()) {
            return Err(config("access range beyond the sequence"));
        }
        Ok(Self { inner, start, len })
    }
}

impl Accesses for SliceAccesses {
    fn len(&self) -> u64 { self.len }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        if start.checked_add(out.len() as u64).is_none_or(|end| end > self.len) {
            return Err(config("access range beyond the sequence"));
        }
        self.inner.fill(self.start + start, out)
    }
}

/// A sequence computed by a native function.
pub struct ExternAccesses {
    f: AccessFn,
    context: *mut c_void,
    len: u64,
}

// SAFETY: the creator of an `ExternAccesses` guarantees (see `new`) that the
// function may be called from any thread, concurrently.
unsafe impl Send for ExternAccesses {}
// SAFETY: as above.
unsafe impl Sync for ExternAccesses {}

impl ExternAccesses {
    /// Wraps a native function as an access sequence of `len` accesses.
    ///
    /// # Safety
    /// `f` must be safe to call with `context` from any thread, concurrently,
    /// for every range within `len`, writing exactly `count` cells, and
    /// `context` must outlive every use of the sequence.
    pub unsafe fn new(f: AccessFn, context: *mut c_void, len: u64) -> Self {
        Self { f, context, len }
    }
}

impl Accesses for ExternAccesses {
    fn len(&self) -> u64 {
        self.len
    }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        if start.checked_add(out.len() as u64).is_none_or(|e| e > self.len) {
            return Err(config("access range beyond the sequence"));
        }
        // SAFETY: the creator guarantees the function's contract (see `new`);
        // `out` holds exactly `count` cells.
        let status = unsafe { (self.f)(self.context, start, out.len() as u64, out.as_mut_ptr()) };
        if status != 0 {
            return Err(Error::Accesses(status));
        }
        Ok(())
    }
}

/// A keyed 64-bit mix (splitmix64's finalizer): the Feistel rounds and the
/// per-epoch keys use it, so a small change in a key or an epoch lands far
/// away in the permutation.
#[inline]
fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Rounds of the Feistel network: alternating halves, each mixed with a hash
/// of the other, so `ROUNDS` steps of the network over `bits` bits form a
/// keyed bijection of `0..2^bits`.
const ROUNDS: u64 = 4;

/// A keyed bijection of `0..2^bits`: a Feistel network over halves of
/// `bits / 2` and `bits - bits / 2` bits.
#[inline]
fn feistel(bits: u32, key: u64, x: u64, inverse: bool) -> u64 {
    let lb = bits / 2;
    let rb = bits - lb;
    let (lm, rm) = ((1u64 << lb) - 1, (1u64 << rb) - 1);
    let (mut l, mut r) = (x >> rb, x & rm);
    // Alternating rounds on unequal halves: each round mixes one half with
    // a hash of the other.
    if !inverse {
        for i in 0..ROUNDS {
            if i % 2 == 0 {
                l = (l ^ mix(key ^ i, r)) & lm;
            } else {
                r = (r ^ mix(key ^ i, l)) & rm;
            }
        }
    } else {
        for i in (0..ROUNDS).rev() {
            if i % 2 == 0 {
                l = (l ^ mix(key ^ i, r)) & lm;
            } else {
                r = (r ^ mix(key ^ i, l)) & rm;
            }
        }
    }
    l << rb | r
}

/// A keyed permutation of `0..n` (cycle walking over the next power of two).
#[inline]
fn permute(n: u64, key: u64, x: u64) -> u64 {
    if n <= 1 {
        return 0;
    }
    let bits = 64 - (n - 1).leading_zeros();
    let mut y = feistel(bits, key, x, false);
    while y >= n {
        y = feistel(bits, key, y, false);
    }
    y
}

/// The inverse of `permute`.
fn unpermute(n: u64, key: u64, y: u64) -> u64 {
    if n <= 1 {
        return 0;
    }
    let bits = 64 - (n - 1).leading_zeros();
    let mut x = feistel(bits, key, y, true);
    while x >= n {
        x = feistel(bits, key, x, true);
    }
    x
}

/// Every schema of a position loads the position's cell: `inner` names one
/// cell per position, the sequence has `schemas` accesses per position.
pub struct Replicate<A> {
    inner: A,
    schemas: u64,
}

impl<A: Accesses> Replicate<A> {
    /// `schemas` accesses per position, each the position's cell.
    pub fn new(inner: A, schemas: usize) -> Result<Self> {
        if schemas == 0 || inner.len().checked_mul(schemas as u64).is_none() {
            return Err(config("replicated sequences need a positive schema count and fit u64"));
        }
        Ok(Self { inner, schemas: schemas as u64 })
    }
}

impl<A: Accesses> Accesses for Replicate<A> {
    fn len(&self) -> u64 {
        self.inner.len() * self.schemas
    }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        fill_positions(&self.inner, self.schemas, start, out, |_, c| Ok(c))
    }
}

/// Schema `s` of a position loads the position's cell plus `offsets[s]`.
pub struct Offset<A> {
    inner: A,
    offsets: Vec<CellId>,
}

impl<A: Accesses> Offset<A> {
    /// One offset per schema, in schema order: schema `s` of a position loads
    /// the position's cell plus `offsets[s]` (wrapping is an error, not a
    /// silent overflow). Paired matrices registered in the same order keep the
    /// same pairing across positions.
    pub fn new(inner: A, offsets: Vec<CellId>) -> Result<Self> {
        if offsets.is_empty() || inner.len().checked_mul(offsets.len() as u64).is_none() {
            return Err(config("offset sequences need one offset per schema and fit u64"));
        }
        Ok(Self { inner, offsets })
    }
}

impl<A: Accesses> Accesses for Offset<A> {
    fn len(&self) -> u64 {
        self.inner.len() * self.offsets.len() as u64
    }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        // SAFETY: `fill_positions` passes schemas below `offsets.len()`.
        fill_positions(&self.inner, self.offsets.len() as u64, start, out, |s, c| c.checked_add(unsafe { *self.offsets.get_unchecked(s) }).ok_or_else(|| config("offset cell overflows the cell id range")))
    }
}

/// Accesses `start..` of a per-position sequence expanded `schemas` times,
/// the cell of schema `s` given by `cell(s, position's cell)`.
fn fill_positions(inner: &dyn Accesses, schemas: u64, start: u64, out: &mut [CellId], cell: impl Fn(usize, CellId) -> Result<CellId>) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let first = start / schemas;
    let last = (start + out.len() as u64 - 1) / schemas;
    let n = (last - first + 1) as usize;
    // Claims usually span few positions, so their cells fit on the stack;
    // larger spans use the heap.
    const STACK: usize = 64;
    let (mut stack, mut heap) = ([0; STACK], Vec::new());
    let cells = if n <= STACK {
        &mut stack[..n]
    } else {
        heap.resize(n, 0);
        &mut heap[..]
    };
    inner.fill(first, cells)?;
    let (mut p, mut s) = (0usize, (start % schemas) as usize);
    for o in out.iter_mut() {
        debug_assert!(p < n);
        // SAFETY: `out` spans positions `first..=last`, so `p` stays below
        // their count `n`.
        *o = cell(s, unsafe { *cells.get_unchecked(p) })?;
        s += 1;
        if s as u64 == schemas {
            s = 0;
            p += 1;
        }
    }
    Ok(())
}

impl<A: Accesses + ?Sized> Accesses for Arc<A> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        (**self).fill(start, out)
    }
    fn as_slice(&self) -> Option<&[CellId]> {
        (**self).as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_are_permutations() {
        for (n, block, group) in [(1u64, 1, 1), (2, 1, 1), (1000, 1, 1), (1000, 7, 3), (1024, 256, 16), (999, 256, 2), (5, 8, 1), (77, 10, 100)] {
            let s = Shuffle::new(n, 2, 11).unwrap().blocks(block, group).unwrap();
            let mut out = vec![0; (2 * n) as usize];
            s.fill(0, &mut out).unwrap();
            for e in 0..2 {
                let mut epoch: Vec<u32> = out[(e * n) as usize..((e + 1) * n) as usize].to_vec();
                epoch.sort_unstable();
                assert_eq!(epoch, (0..n as u32).collect::<Vec<_>>(), "n={n} block={block} group={group}");
            }
            let mut part = vec![0; 5.min(n as usize)];
            s.fill(n / 2, &mut part).unwrap();
            assert_eq!(part, out[(n / 2) as usize..(n / 2) as usize + part.len()]);
        }
    }

    #[test]
    fn blocks_keep_cells_local() {
        let s = Shuffle::new(4096, 1, 3).unwrap().blocks(256, 1).unwrap();
        let mut out = vec![0; 4096];
        s.fill(0, &mut out).unwrap();
        for w in out.chunks(256) {
            assert!(w.iter().all(|&c| c / 256 == w[0] / 256));
        }
    }
}

