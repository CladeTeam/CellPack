//! Claims and the execution frontier.
//!
//! Accesses are claimed in order through one counter, in whole positions, at
//! most `window` past the frontier, the smallest unfinished access. Each
//! worker keeps its claims as a queue of ranges with the count still
//! unfinished and publishes the start of the oldest (its low-water mark); the
//! frontier is the least of these marks and the counter, computed when a
//! claim needs it. A worker with nothing outstanding publishes its mark before
//! advancing the counter, so a concurrent scan never passes a claim being
//! made.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{
    AtomicBool, AtomicU64,
    Ordering::{Acquire, Relaxed, Release, SeqCst},
};

use crossbeam_utils::CachePadded;

/// The claim counter, the execution frontier and the workers' low-water marks.
///
/// Claims advance `next` and are always whole `unit`s (one position's
/// accesses), so a position is never split across workers; `window` bounds how
/// far past the frontier a claim may start. `minima[worker]` is the first
/// access of that worker's oldest unfinished claim, `u64::MAX` when it has
/// none outstanding.
pub(crate) struct Progress {
    /// The next access to claim.
    next: CachePadded<AtomicU64>,
    /// The smallest unfinished access, as last scanned.
    frontier: CachePadded<AtomicU64>,
    /// Per worker: the first access of its oldest unfinished claim.
    minima: Box<[CachePadded<AtomicU64>]>,
    stopped: CachePadded<AtomicBool>,
    /// Whoever wins this scan recomputes the frontier; the rest reuse it.
    scanning: CachePadded<AtomicBool>,
    len: u64,
    window: u64,
    unit: u64,
    /// Most accesses ever claimed past the frontier at one moment.
    pub peak_span: AtomicU64,
}

/// A worker's claimed ranges with their unfinished counts, oldest first. Each
/// entry is `(first access, accesses, unfinished)`.
#[derive(Default)]
pub(crate) struct Outstanding {
    ranges: VecDeque<(u64, u64, u64)>,
}

impl Outstanding {
    /// Whether the worker has finished everything it claimed.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

/// One claim attempt's outcome.
pub(crate) enum Claim {
    /// A range of accesses covering whole positions.
    Got(Range<u64>),
    /// The window or a gate is full; retry after progress.
    Blocked,
    /// The sequence is fully claimed.
    Exhausted,
}

impl Progress {
    /// Claims are whole multiples of `unit` (a position's accesses); the
    /// window is raised to at least one `unit`, and `unit` to at least 1.
    pub fn new(workers: usize, len: u64, window: u64, unit: u64) -> Self {
        Self {
            next: CachePadded::new(AtomicU64::new(0)),
            frontier: CachePadded::new(AtomicU64::new(0)),
            minima: (0..workers).map(|_| CachePadded::new(AtomicU64::new(u64::MAX))).collect(),
            stopped: CachePadded::new(AtomicBool::new(false)),
            scanning: CachePadded::new(AtomicBool::new(false)),
            len,
            window: window.max(unit),
            unit: unit.max(1),
            peak_span: AtomicU64::new(0),
        }
    }

    /// Claims up to `max` accesses (whole positions) inside `[.., gate)` and
    /// the window.
    pub fn claim(&self, worker: usize, local: &mut Outstanding, max: u64, gate: u64) -> Claim {
        loop {
            let n = self.next.load(SeqCst);
            if n >= self.len {
                return Claim::Exhausted;
            }
            let front = self.frontier.load(Acquire);
            let limit = self.len.min(front.saturating_add(self.window)).min(gate);
            if n >= limit {
                return Claim::Blocked;
            }
            let mut end = n + max.min(limit - n);
            if end < self.len {
                end -= end % self.unit;
            }
            if end <= n {
                return Claim::Blocked;
            }
            // A worker with nothing outstanding publishes the access it is
            // about to claim before advancing the counter, so a concurrent
            // scan cannot see the frontier pass a claim in progress.
            if local.is_empty() {
                self.minima[worker].store(n, SeqCst);
            }
            if self.next.compare_exchange_weak(n, end, SeqCst, Relaxed).is_ok() {
                local.ranges.push_back((n, end - n, end - n));
                self.peak_span.fetch_max(end - front, Relaxed);
                return Claim::Got(n..end);
            }
            if local.is_empty() {
                self.minima[worker].store(u64::MAX, Release);
            }
        }
    }

    /// Completes `count` claimed accesses from `position`.
    ///
    /// # Panics
    /// `position` is not inside one of this worker's outstanding claims.
    pub fn complete(&self, worker: usize, local: &mut Outstanding, position: u64, count: u64) {
        let r = local.ranges.iter_mut().find(|r| r.0 <= position && position < r.0 + r.1).expect("completed position was claimed by this worker");
        debug_assert!(r.2 >= count);
        r.2 -= count;
        if local.ranges.front().is_some_and(|r| r.2 == 0) {
            while local.ranges.front().is_some_and(|r| r.2 == 0) {
                local.ranges.pop_front();
            }
            self.minima[worker].store(local.ranges.front().map_or(u64::MAX, |r| r.0), Release);
        }
    }

    /// Recomputes the frontier: the least of the claim counter and every
    /// worker's mark. One worker scans at a time; the others return the
    /// published value. It only moves forward (`fetch_max`), so a scan that
    /// raced with a claim cannot pull it back.
    pub fn scan(&self) -> u64 {
        if self.scanning.compare_exchange(false, true, Acquire, Relaxed).is_err() {
            return self.frontier.load(Acquire);
        }
        let mut front = self.next.load(SeqCst);
        for m in self.minima.iter() {
            front = front.min(m.load(Acquire));
        }
        let old = self.frontier.load(Acquire);
        let front = if front > old { self.frontier.fetch_max(front, Release).max(front) } else { old };
        self.scanning.store(false, Release);
        front
    }

    /// The claim counter: the next access a worker could claim.
    pub fn cursor(&self) -> u64 {
        self.next.load(SeqCst)
    }
    /// Stops the run: every worker's drive loop observes it and returns
    /// [`crate::loader::Error::Aborted`].
    pub fn stop(&self) {
        self.stopped.store(true, Release);
    }
    /// Whether the run was stopped.
    pub fn stopped(&self) -> bool {
        self.stopped.load(Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holes_bound_the_window_until_the_oldest_position_completes() {
        let p = Progress::new(2, 20, 4, 1);
        let (mut a, mut b) = (Outstanding::default(), Outstanding::default());
        assert!(matches!(p.claim(0, &mut a, 1, u64::MAX), Claim::Got(r) if r == (0..1)));
        assert!(matches!(p.claim(1, &mut b, 8, u64::MAX), Claim::Got(r) if r == (1..4)));
        for i in (1..4).rev() {
            p.complete(1, &mut b, i, 1);
        }
        p.scan();
        assert!(matches!(p.claim(1, &mut b, 8, u64::MAX), Claim::Blocked));
        p.complete(0, &mut a, 0, 1);
        assert_eq!(p.scan(), 4);
        assert!(matches!(p.claim(1, &mut b, 8, 6), Claim::Got(r) if r == (4..6)));
    }

    #[test]
    fn claims_keep_whole_units() {
        let p = Progress::new(2, 18, 6, 3);
        let (mut a, mut b) = (Outstanding::default(), Outstanding::default());
        assert!(matches!(p.claim(0, &mut a, 5, 18), Claim::Got(r) if r == (0..3)));
        assert!(matches!(p.claim(1, &mut b, 9, 18), Claim::Got(r) if r == (3..6)));
        p.complete(0, &mut a, 0, 3);
        assert_eq!(p.scan(), 3);
        assert!(matches!(p.claim(0, &mut a, 9, 18), Claim::Got(r) if r == (6..9)));
        p.complete(1, &mut b, 3, 3);
        p.complete(0, &mut a, 6, 3);
        assert_eq!(p.scan(), 9);
        assert!(p.peak_span.load(Relaxed) <= 6);
    }
}
