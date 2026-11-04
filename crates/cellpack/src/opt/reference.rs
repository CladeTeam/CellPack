//! Serial equal-page caching with an empty initial cache and optional admission.
//! The read scratch is outside the retention budget, so a miss may bypass it.
//!
//! This is the reference plans are measured against, not a policy the loader
//! runs: one page per request, no concurrency, no management cost.

use crate::loader::catalog::{CellId, DataSource};
use crate::loader::error::{Error, Result};

/// A page id and a position standing for "never": both counts stay below it.
const NONE: u32 = u32::MAX;

/// Canonical page requests: cells in access order, each cell's pages in file order.
/// This serial reference has no execution window, reuse-gap reservation or management cost.
pub struct PageTrace {
    /// The requests, as page ids in `[0, universe)`.
    pub(crate) pages: Vec<u32>,
    /// Per request, the next request of the same page, or [`NONE`].
    pub(crate) next: Vec<u32>,
    /// Page ids the trace may name: the catalog's page count, not the
    /// distinct pages it uses.
    pub(crate) universe: usize,
    /// Distinct pages actually requested.
    distinct: usize,
}

/// What a serial run of a trace did: its page uses, and the hits and misses
/// a capacity produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheSummary {
    /// Requests in the trace.
    pub page_uses: u64,
    /// Distinct pages it names.
    pub distinct_pages: u64,
    /// Requests served by a retained page.
    pub hits: u64,
    /// Requests that read their page.
    pub misses: u64,
    /// Retention slots the run had, in pages.
    pub capacity_pages: usize,
    /// Accesses served by a cached mapped row; a page trace has none, so
    /// zero.
    pub row_hits: u64,
    /// Accesses whose output was already in the cache; a page trace has
    /// none, so zero.
    pub output_hits: u64,
}

impl PageTrace {
    /// Expands `accesses` into physical page requests through the catalog's
    /// locations; errors when page ids, accesses or page uses would not fit
    /// `u32`.
    pub fn from_source(catalog: &DataSource, accesses: &[CellId]) -> Result<Self> {
        if catalog.n_pages() >= NONE as u64 || accesses.len() >= NONE as usize {
            return Err(Error::Plan(
                "serial trace requires u32 pages and positions".into(),
            ));
        }
        let mut pages = Vec::new();
        for (at, &cell) in accesses.iter().enumerate() {
            let loc = catalog.locate(at as u64, cell)?;
            if pages.len().saturating_add(loc.page_count) >= NONE as usize {
                return Err(Error::Plan(
                    "serial trace requires fewer than u32::MAX page uses".into(),
                ));
            }
            for p in loc.first_page..loc.first_page + loc.page_count as u64 {
                pages.push(p as u32);
            }
        }
        Ok(Self::index(pages, catalog.n_pages() as usize))
    }

    /// Page ids are dense integers in `[0, universe)`; no hidden renumbering.
    pub fn from_pages(pages: &[u32], universe: usize) -> Result<Self> {
        if pages.len() >= NONE as usize
            || universe >= NONE as usize
            || pages.iter().any(|&p| p as usize >= universe)
        {
            return Err(Error::Plan("invalid serial page trace".into()));
        }
        Ok(Self::index(pages.to_vec(), universe))
    }

    /// Indexes the next-use table from the end, so `next[i]` is the first
    /// request after `i` of the same page; `last` then holds each page's
    /// first request, which counts the distinct pages.
    fn index(pages: Vec<u32>, universe: usize) -> Self {
        let mut last = vec![NONE; universe];
        let mut next = vec![NONE; pages.len()];
        for (i, &p) in pages.iter().enumerate().rev() {
            next[i] = std::mem::replace(&mut last[p as usize], i as u32);
        }
        let distinct = last.iter().filter(|&&x| x != NONE).count();
        Self {
            pages,
            next,
            universe,
            distinct,
        }
    }

    /// Requests in the trace.
    pub fn page_uses(&self) -> usize {
        self.pages.len()
    }
    /// Distinct pages it requests.
    pub fn distinct_pages(&self) -> usize {
        self.distinct
    }

    /// Belady with bypass: on a miss the requested page participates in the
    /// farthest-next-use comparison. Dead pages are never retained.
    pub fn optimum(&self, capacity_pages: usize) -> CacheSummary {
        // The two ends of the capacity curve are known without running it:
        // no retention misses everything, and enough retention for every
        // distinct page misses each of them once.
        if capacity_pages == 0 || capacity_pages >= self.distinct {
            let misses = if capacity_pages == 0 {
                self.pages.len()
            } else {
                self.distinct
            } as u64;
            return CacheSummary {
                page_uses: self.pages.len() as u64,
                distinct_pages: self.distinct as u64,
                hits: self.pages.len() as u64 - misses,
                misses,
                capacity_pages,
                ..Default::default()
            };
        }
        let mut cache = FutureCache::new(self.universe, capacity_pages.min(self.distinct));
        let mut hits = 0;
        for (&p, &next) in self.pages.iter().zip(&self.next) {
            let hit = cache.touch(p, next);
            hits += u64::from(hit);
        }
        CacheSummary {
            page_uses: self.pages.len() as u64,
            distinct_pages: self.distinct as u64,
            hits,
            misses: self.pages.len() as u64 - hits,
            capacity_pages,
            ..Default::default()
        }
    }
}

/// Indexed max-heap: one entry per resident page, with no stale queue entries.
struct FutureCache {
    heap: Vec<(u32, u32)>, // (next request, page)
    /// Index in `heap` of each page, or `usize::MAX` when it is not resident.
    slots: Vec<usize>,
    capacity: usize,
}

impl FutureCache {
    /// An empty cache of `capacity` pages that may name page ids below
    /// `pages`.
    fn new(pages: usize, capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            slots: vec![usize::MAX; pages],
            capacity,
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.slots[self.heap[a].1 as usize] = a;
        self.slots[self.heap[b].1 as usize] = b;
    }

    fn up(&mut self, mut at: usize) {
        while at > 0 {
            let parent = (at - 1) / 2;
            if self.heap[parent] >= self.heap[at] {
                break;
            }
            self.swap(parent, at);
            at = parent;
        }
    }

    fn down(&mut self, mut at: usize) {
        while 2 * at + 1 < self.heap.len() {
            let mut child = 2 * at + 1;
            if child + 1 < self.heap.len() && self.heap[child + 1] > self.heap[child] {
                child += 1;
            }
            if self.heap[at] >= self.heap[child] {
                break;
            }
            self.swap(at, child);
            at = child;
        }
    }

    fn remove(&mut self, at: usize) {
        let page = self.heap[at].1;
        let tail = self.heap.len() - 1;
        self.swap(at, tail);
        self.heap.pop();
        self.slots[page as usize] = usize::MAX;
        if at < self.heap.len() {
            if at > 0 && self.heap[at] > self.heap[(at - 1) / 2] {
                self.up(at);
            } else {
                self.down(at);
            }
        }
    }

    /// One request: whether it hits. A resident page's entry moves to its new
    /// next use, or leaves the heap once it is dead; a miss is admitted only
    /// when its next use is nearer than the farthest resident's, and a page
    /// that is never used again is bypassed instead.
    fn touch(&mut self, page: u32, next: u32) -> bool {
        let at = self.slots[page as usize];
        if at != usize::MAX {
            if next == NONE {
                self.remove(at);
            } else {
                self.heap[at].0 = next;
                self.up(at);
            }
            return true;
        }
        if self.capacity == 0 || next == NONE {
            return false;
        }
        if self.heap.len() == self.capacity {
            // The heap's root is the farthest next use; the tie on page id
            // keeps the choice deterministic when several pages share it.
            if (next, page) >= self.heap[0] {
                return false;
            }
            self.remove(0);
        }
        self.slots[page as usize] = self.heap.len();
        self.heap.push((next, page));
        self.up(self.heap.len() - 1);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exhaustive_misses(pages: &[u32], capacity: usize) -> u64 {
        let mut dp = [u64::MAX / 2; 8];
        dp[0] = 0;
        for &page in pages {
            let mut next = [u64::MAX / 2; 8];
            for (state, &cost) in dp.iter().enumerate() {
                let available = state | (1 << page);
                let miss = u64::from(state & (1 << page) == 0);
                for (kept, best) in next.iter_mut().enumerate() {
                    if kept & !available == 0 && kept.count_ones() as usize <= capacity {
                        *best = (*best).min(cost + miss);
                    }
                }
            }
            dp = next;
        }
        *dp.iter().min().unwrap()
    }

    #[test]
    fn optimum_matches_all_retention_choices() {
        for n in 0..=7 {
            for mut code in 0..3usize.pow(n) {
                let pages: Vec<u32> = (0..n)
                    .map(|_| {
                        let p = code % 3;
                        code /= 3;
                        p as u32
                    })
                    .collect();
                let trace = PageTrace::from_pages(&pages, 3).unwrap();
                let mut prior = pages.len() as u64;
                let mut prior_gain = u64::MAX;
                for c in 0..=4 {
                    let s = trace.optimum(c);
                    assert_eq!(s.misses, exhaustive_misses(&pages, c), "{pages:?}, C={c}");
                    assert!(s.misses <= prior);
                    assert!(s.misses >= s.distinct_pages);
                    if c > 0 {
                        let gain = prior - s.misses;
                        assert!(gain <= prior_gain);
                        prior_gain = gain;
                    }
                    prior = s.misses;
                }
            }
        }
    }

    #[test]
    fn bypass_preserves_a_more_valuable_page() {
        let t = PageTrace::from_pages(&[0, 1, 0, 2, 0], 3).unwrap();
        assert_eq!(t.optimum(1).misses, 3);
        assert_eq!(t.optimum(0).misses, 5);
    }
}
