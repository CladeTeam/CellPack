//! Occupancy counts over a window of positions: range add, range max.
//!
//! Positions are grouped into blocks of `1 << shift`; a residency occupies
//! every block it touches, which over-counts only at block edges, so a
//! budget checked against it still holds. Blocks map onto a ring of leaves,
//! so an unbounded sequence is followed through a sliding window: a block
//! enters when the compiler's frontier reaches it, with an initial value,
//! and its leaf is reused once the window has passed it. The unit is the
//! caller's; the compiler counts frames.

/// A range-add, range-max tree over a sliding window of positions. `t` holds
/// each node's maximum of the values below it plus its own tag; `d` holds the
/// tags of internal nodes, added once to their subtree until pushed down.
pub(crate) struct Occupancy {
    /// Leaves, a power of two.
    n: usize,
    /// Height of the tree: `log2(n)`.
    h: u32,
    /// `log2` of the positions one leaf covers.
    shift: u32,
    t: Vec<u32>,
    d: Vec<u32>,
}

/// Blocks a window above [`EXACT`] positions fits into: the block size grows
/// until the ring needs at most this many (its leaf count then rounds up to a
/// power of two, at most twice as many leaves). Few enough for a shallow tree
/// in a core's caches; blocks stay far shorter than the residencies they hold.
const LEAVES: u64 = 1 << 13;

/// Windows up to this many positions keep a leaf per position.
const EXACT: u64 = 1 << 16;

impl Occupancy {
    /// A tree whose ring covers at least `window` positions: exact (a leaf
    /// per position) for short windows, else at the finest block size that
    /// needs at most [`LEAVES`] blocks.
    pub fn new(window: u64) -> Self {
        let window = window.max(1);
        let mut shift = 0;
        while window > EXACT && (window - 1) >> shift >= LEAVES {
            shift += 1;
        }
        let n = (((window - 1) >> shift) + 1).next_power_of_two() as usize;
        Self { n, h: usize::BITS - n.leading_zeros(), shift, t: vec![0; 2 * n], d: vec![0; n] }
    }

    /// `log2` of the positions one block (leaf) covers; 0 means a leaf per
    /// position.
    #[inline]
    pub fn shift(&self) -> u32 {
        self.shift
    }

    // Node accessors. Node indices of the tree lie in `1..2n` and tagged
    // (internal) nodes in `1..n`: every index below comes from a leaf in
    // `n..2n` or a range of leaves, shifted toward the root, or from an
    // internal node's children.
    #[inline(always)]
    fn t(&self, p: usize) -> u32 {
        debug_assert!(p < 2 * self.n);
        // SAFETY: a node index (see above).
        unsafe { *self.t.get_unchecked(p) }
    }
    #[inline(always)]
    fn t_mut(&mut self, p: usize) -> &mut u32 {
        debug_assert!(p < 2 * self.n);
        // SAFETY: a node index (see above).
        unsafe { self.t.get_unchecked_mut(p) }
    }
    #[inline(always)]
    fn d(&self, p: usize) -> u32 {
        debug_assert!(p < self.n);
        // SAFETY: an internal node index (see above).
        unsafe { *self.d.get_unchecked(p) }
    }
    #[inline(always)]
    fn d_mut(&mut self, p: usize) -> &mut u32 {
        debug_assert!(p < self.n);
        // SAFETY: an internal node index (see above).
        unsafe { self.d.get_unchecked_mut(p) }
    }

    /// Adds `v` at node `p`: its aggregate, and its tag when it is internal.
    #[inline]
    fn apply(&mut self, p: usize, v: u32) {
        *self.t_mut(p) += v;
        if p < self.n {
            *self.d_mut(p) += v;
        }
    }
    /// Recomputes the aggregates from leaf `p` to the root.
    #[inline]
    fn build(&mut self, mut p: usize) {
        while p > 1 {
            p >>= 1;
            *self.t_mut(p) = self.t(2 * p).max(self.t(2 * p + 1)) + self.d(p);
        }
    }
    /// Pushes the tags above leaf `p` down to it, so the leaf's own value can
    /// be read or overwritten.
    #[inline]
    fn push(&mut self, p: usize) {
        for s in (1..=self.h).rev() {
            let i = p >> s;
            if i >= 1 && self.d(i) != 0 {
                let v = self.d(i);
                self.apply(2 * i, v);
                self.apply(2 * i + 1, v);
                *self.d_mut(i) = 0;
            }
        }
    }

    /// Most occupancy anywhere in the ring (at least the most in the window).
    #[inline]
    pub fn peak(&self) -> u32 {
        self.t(1)
    }

    /// Sets the occupancy of the block holding `position`, as the window
    /// reaches it: blocks enter in position order, each with the value they
    /// held before any residency was added to it.
    pub fn enter(&mut self, position: u64, value: u32) {
        let p = ((position >> self.shift) as usize & (self.n - 1)) + self.n;
        self.push(p);
        *self.t_mut(p) = value;
        self.build(p);
    }

    /// Leaf ranges `[l, r)` covering positions `lo..=hi`, split where the
    /// ring wraps.
    #[inline]
    fn ranges(&self, lo: u64, hi: u64) -> [(usize, usize); 2] {
        let (bl, bh) = (lo >> self.shift, hi >> self.shift);
        debug_assert!(bh - bl < self.n as u64, "range wider than the ring");
        let mask = self.n - 1;
        let (l, r) = (bl as usize & mask, bh as usize & mask);
        if l <= r { [(l, r + 1), (0, 0)] } else { [(l, self.n), (0, r + 1)] }
    }

    /// Adds `v` over positions `lo..=hi`; `lo > hi` adds nothing.
    pub fn add(&mut self, lo: u64, hi: u64, v: u32) {
        if lo > hi || v == 0 {
            return;
        }
        for (l, r) in self.ranges(lo, hi) {
            if l < r {
                self.add_leaves(l, r, v);
            }
        }
    }

    /// Removes `v` over positions `lo..=hi`, added before.
    #[cfg(test)]
    pub fn sub(&mut self, lo: u64, hi: u64, v: u32) {
        if lo > hi || v == 0 {
            return;
        }
        for (l, r) in self.ranges(lo, hi) {
            if l < r {
                self.sub_leaves(l, r, v);
            }
        }
    }

    /// Covers leaves `[l, r)` with `v`: the canonical decomposition into
    /// subtrees, tagged at the highest nodes that fit, then the aggregates
    /// of the two boundary paths rebuilt.
    fn add_leaves(&mut self, l: usize, r: usize, v: u32) {
        let (mut l, mut r) = (l + self.n, r + self.n);
        let (l0, r0) = (l, r);
        while l < r {
            if l & 1 == 1 {
                self.apply(l, v);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                self.apply(r, v);
            }
            l >>= 1;
            r >>= 1;
        }
        self.build(l0);
        self.build(r0 - 1);
    }

    #[cfg(test)]
    fn sub_leaves(&mut self, l: usize, r: usize, v: u32) {
        // Subtraction never goes below what was added, so pushing tags down
        // to the touched leaves keeps every tag nonnegative.
        let (l, r) = (l + self.n, r + self.n);
        self.push(l);
        self.push(r - 1);
        let (mut a, mut b) = (l, r);
        while a < b {
            if a & 1 == 1 {
                self.unapply(a, v);
                a += 1;
            }
            if b & 1 == 1 {
                b -= 1;
                self.unapply(b, v);
            }
            a >>= 1;
            b >>= 1;
        }
        self.build(l);
        self.build(r - 1);
    }

    #[cfg(test)]
    fn unapply(&mut self, p: usize, v: u32) {
        if p < self.n && self.d(p) < v {
            // The tag holds less than `v`: part of the addition was pushed
            // further down; descend.
            let tag = std::mem::take(self.d_mut(p));
            self.apply(2 * p, tag);
            self.apply(2 * p + 1, tag);
            self.unapply(2 * p, v);
            self.unapply(2 * p + 1, v);
            *self.t_mut(p) = self.t(2 * p).max(self.t(2 * p + 1));
            return;
        }
        *self.t_mut(p) -= v;
        if p < self.n {
            *self.d_mut(p) -= v;
        }
    }

    /// Most occupancy over positions `lo..=hi`.
    pub fn max(&mut self, lo: u64, hi: u64) -> u32 {
        if lo > hi {
            return 0;
        }
        let mut out = 0;
        for (l, r) in self.ranges(lo, hi) {
            if l < r {
                out = out.max(self.max_leaves(l, r));
            }
        }
        out
    }

    /// Most occupancy over leaves `[l, r)`; the tags above both boundary
    /// leaves are pushed first so every visited node's aggregate is complete.
    fn max_leaves(&mut self, l: usize, r: usize) -> u32 {
        let (mut l, mut r) = (l + self.n, r + self.n);
        self.push(l);
        self.push(r - 1);
        let mut out = 0;
        while l < r {
            if l & 1 == 1 {
                out = out.max(self.t(l));
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                out = out.max(self.t(r));
            }
            l >>= 1;
            r >>= 1;
        }
        out
    }

    /// Whether `v` more over `lo..=hi` keeps every block within `cap`, in
    /// the same unit `add` takes. The ring's peak is checked first because it
    /// bounds every block, and the range only when the peak does not fit.
    #[inline]
    pub fn fits(&mut self, lo: u64, hi: u64, v: u32, cap: u32) -> bool {
        lo > hi || self.peak() + v <= cap || self.max(lo, hi) + v <= cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_flat_array_across_ring_wraps() {
        let mut t = Occupancy::new(64);
        assert_eq!(t.shift(), 0);
        let mut flat = vec![0u32; 1000];
        let mut seed = 7u64;
        let mut next = |m: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        let mut added: Vec<(u64, u64, u32)> = Vec::new();
        for front in 0..1000u64 {
            // The window is the last 64 positions; blocks enter at zero.
            t.enter(front, 0);
            for _ in 0..3 {
                let lo = front.saturating_sub(next(63));
                let v = next(5) as u32 + 1;
                t.add(lo, front, v);
                added.push((lo, front, v));
                for f in &mut flat[lo as usize..=front as usize] {
                    *f += v;
                }
                if next(3) == 0 {
                    let (l, h, w) = added.swap_remove(next(added.len() as u64) as usize);
                    if l + 63 >= front {
                        t.sub(l, h, w);
                        for f in &mut flat[l as usize..=h as usize] {
                            *f -= w;
                        }
                    }
                }
            }
            added.retain(|&(l, _, _)| l + 63 > front);
            let lo = front.saturating_sub(next(64));
            assert_eq!(t.max(lo, front), *flat[lo as usize..=front as usize].iter().max().unwrap());
        }
    }
}
