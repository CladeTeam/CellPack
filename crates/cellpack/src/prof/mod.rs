//! Profiling as a type parameter.
//!
//! Timed code is generic over [`Profiler`]. [`Off`] is zero-sized with empty
//! inline methods, so its instances compile to the untimed code; [`On`]
//! first tests one bit of a run-time mask per site, so a disabled site costs
//! that test and no clock read. Both instances live in every build and a run
//! picks one when it starts.
//!
//! Every timing point is a [`Site`]. Coarser views (io, worker, cpu, wait,
//! planner, calibrate) are named masks over sites ([`GROUPS`]), not separate
//! mechanisms. Event counts (accesses, reads, hits, bytes) are not profiling:
//! they are plain local additions, always kept in the run statistics.

use std::sync::atomic::{AtomicU64, Ordering::*};
use std::time::{Duration, Instant};

/// Kinds of site records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Calls and total time.
    Span,
    /// Calls, total time and a log2 histogram of the durations.
    Hist,
}

/// The detail a site keeps per call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// One record for the site.
    None,
    /// One record per schema.
    Schema,
    /// One record per schema, size band ([`crate::loader::PROCESS_BOUNDS`])
    /// and row kind (full or partial).
    SchemaBand,
}

/// A site's static description.
#[derive(Clone, Copy, Debug)]
pub struct SiteInfo {
    /// The site's name, as configuration and reports spell it.
    pub name: &'static str,
    /// Spans or histograms.
    pub kind: Kind,
    /// What it keeps per call.
    pub detail: Detail,
    /// The site this one runs inside, for self time.
    pub parent: Option<Site>,
    /// What the site times, as reports and `pycellpack`'s `build_info()` show
    /// it.
    pub doc: &'static str,
}

macro_rules! sites {
    ($($v:ident = $name:literal, $kind:ident, $detail:ident, $parent:expr, $doc:literal;)*) => {
        /// A timing point.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum Site { $($v),* }
        /// Every site, in [`Site`] order.
        pub const SITES: &[SiteInfo] = &[$(SiteInfo { name: $name, kind: Kind::$kind, detail: Detail::$detail, parent: $parent, doc: $doc }),*];
        impl Site {
            /// Every site, in declaration order.
            pub const ALL: &[Site] = &[$(Site::$v),*];
        }
    };
}

sites! {
    Worker = "worker", Span, None, None, "a worker's whole lifetime";
    Claim = "claim", Span, None, Some(Site::Worker), "claiming positions and fetching their plan entries";
    Issue = "issue", Span, None, Some(Site::Claim), "building the reads of claimed accesses";
    IoSubmit = "io.submit", Span, None, Some(Site::Worker), "submitting queued reads to the ring (a pread read completes under issue)";
    IoReap = "io.reap", Span, None, Some(Site::Worker), "harvesting completions and publishing generations";
    IoWait = "io.wait", Span, None, Some(Site::Worker), "blocked waiting for a completion";
    IoLatency = "io.latency", Hist, None, None, "each read from queueing to harvest";
    WaitGen = "wait.gen", Hist, None, None, "a planned frame admitted by another access becoming ready";
    WaitPlan = "wait.plan", Span, None, Some(Site::Claim), "claims held back by the plan's published frontier";
    WaitWindow = "wait.window", Span, None, Some(Site::Claim), "claims held back by the execution window";
    WaitRing = "wait.ring", Span, None, Some(Site::Idle), "claims held back by the output ring";
    Idle = "idle", Span, None, Some(Site::Worker), "spinning, yielding or sleeping with nothing to do";
    Decode = "decode", Span, None, Some(Site::Worker), "decoding records";
    RowAdmit = "row.admit", Span, None, Some(Site::Worker), "writing admitted rows into their frames";
    RowGather = "row.gather", Span, None, Some(Site::Worker), "gathering a cached row from its frames for mapping";
    Map = "map", Span, Schema, Some(Site::Worker), "mapping rows to output dims and converting types";
    Process = "process", Span, SchemaBand, Some(Site::Worker), "process function calls";
    Complete = "complete", Span, None, Some(Site::Worker), "completing outputs and the local frontier";
    Frontier = "frontier", Span, None, None, "computing the global frontier";
    PlanAccess = "plan.access", Span, None, None, "planner: generating accesses and scattering their uses";
    PlanRow = "plan.row", Span, None, None, "planner: row selection";
    PlanPage = "plan.page", Span, None, None, "planner: page selection";
    PlanFreeze = "plan.freeze", Span, None, None, "planner: freezing, frame allocation and generations";
    PlanBlock = "plan.block", Span, None, Some(Site::PlanFreeze), "planner: merging block reads";
    PlanAssemble = "plan.assemble", Span, None, None, "planner: expanding frozen positions into entries";
    PlanSpace = "plan.space", Span, None, None, "planner: waiting for plan ring space";
    PlanPublish = "plan.publish", Span, None, None, "planner: writing entries and advancing the frontier";
    PlanIdle = "plan.idle", Span, None, None, "planner: threads with no ready task";
}

/// Most sites a mask holds.
pub const MAX_SITES: usize = 64;
const _: () = assert!(SITES.len() <= MAX_SITES);

/// A set of sites, one bit per site in [`Site`] order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SiteMask(pub u64);

/// Named site sets; [`SiteMask::parse`] accepts any of these names beside the
/// site names and `all`.
pub const GROUPS: &[(&str, &[&str])] = &[
    ("io", &["io.submit", "io.reap", "io.wait", "io.latency"]),
    ("wait", &["wait.gen", "wait.plan", "wait.window", "wait.ring", "idle"]),
    ("cpu", &["decode", "row.admit", "row.gather", "map", "process"]),
    ("worker", &["worker", "claim", "issue", "io.submit", "io.reap", "io.wait", "wait.plan", "wait.window", "wait.ring", "idle", "decode", "row.admit", "row.gather", "map", "process", "complete", "frontier"]),
    ("planner", &["plan.access", "plan.row", "plan.page", "plan.freeze", "plan.block", "plan.assemble", "plan.space", "plan.publish", "plan.idle"]),
    // What `pycellpack.loader.costs` fits a cost model from.
    ("calibrate", &["worker", "claim", "issue", "io.submit", "io.reap", "io.wait", "idle", "wait.gen", "wait.plan", "wait.window", "wait.ring", "decode", "row.admit", "row.gather", "map", "process", "complete", "frontier"]),
];

impl SiteMask {
    /// No site: every hook is a bit test that fails.
    pub const NONE: SiteMask = SiteMask(0);
    /// Every site: all 64 bits when there are that many, else one per site.
    pub const ALL: SiteMask = SiteMask(if SITES.len() == 64 { u64::MAX } else { (1u64 << SITES.len()) - 1 });

    /// Sites and groups by name; `all` is every site.
    pub fn parse<S: AsRef<str>>(names: &[S]) -> crate::loader::Result<Self> {
        let mut m = 0u64;
        for name in names {
            let name = name.as_ref();
            if name == "all" {
                m |= Self::ALL.0;
            } else if let Some(i) = SITES.iter().position(|s| s.name == name) {
                m |= 1 << i;
            } else if let Some((_, members)) = GROUPS.iter().find(|(g, _)| *g == name) {
                m |= Self::parse(members)?.0;
            } else {
                return Err(crate::loader::error::config(format!("unknown profile site or group {name:?}")));
            }
        }
        Ok(Self(m))
    }

    /// Whether `site` is enabled; a bit test of its [`Site`] index, so the
    /// timed instances can skip a disabled site at its call site.
    #[inline(always)]
    pub fn contains(self, site: Site) -> bool {
        self.0 >> site as u32 & 1 != 0
    }

    /// The names of the enabled sites, in [`Site`] order.
    pub fn names(self) -> Vec<&'static str> {
        Site::ALL.iter().filter(|&&s| self.contains(s)).map(|&s| SITES[s as usize].name).collect()
    }
}

/// Run-time clock for profiled runs: the TSC where it is invariant, else
/// nanoseconds since the run started. Readings are converted to time once,
/// when the run reports.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    base: Instant,
    tsc: bool,
    tsc0: u64,
}

impl Clock {
    /// A clock started now: the TSC where the CPU has an invariant one, else
    /// `Instant`.
    pub fn new() -> Self {
        let tsc = invariant_tsc();
        Self { base: Instant::now(), tsc, tsc0: if tsc { rdtsc() } else { 0 } }
    }

    /// A nonzero reading: the TSC or nanoseconds since the clock started, the
    /// low bit always set so zero can mean "no stamp".
    #[inline(always)]
    pub fn now(&self) -> u64 {
        if self.tsc { rdtsc() | 1 } else { self.base.elapsed().as_nanos() as u64 + 1 }
    }

    /// Nanoseconds per tick, measured over the clock's life so far.
    pub fn nanos_per_tick(&self) -> f64 {
        if !self.tsc {
            return 1.0;
        }
        let ns = self.base.elapsed().as_nanos() as f64;
        let ticks = rdtsc().wrapping_sub(self.tsc0) as f64;
        if ticks > 0.0 && ns > 1e6 { ns / ticks } else { calibrate_tsc() }
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn rdtsc() -> u64 {
    // SAFETY: RDTSC is available on every x86_64 CPU.
    unsafe { std::arch::x86_64::_rdtsc() }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn rdtsc() -> u64 {
    0
}

/// Whether the CPU's TSC ticks at a constant rate whatever the core's
/// frequency: CPUID leaf `0x8000_0007`, EDX bit 8.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
fn invariant_tsc() -> bool {
    // SAFETY: CPUID is available on every x86-64 CPU.
    let max = unsafe { std::arch::x86_64::__cpuid(0x8000_0000) }.eax;
    // SAFETY: CPUID is available, and the maximum extended leaf is checked.
    max >= 0x8000_0007 && unsafe { std::arch::x86_64::__cpuid(0x8000_0007) }.edx & (1 << 8) != 0
}

/// Off x86-64 there is no TSC to read, so `Clock` uses `Instant`.
#[cfg(not(target_arch = "x86_64"))]
fn invariant_tsc() -> bool {
    false
}

/// Nanoseconds per tick over a 5 ms sleep, for clocks too young to measure
/// their own rate.
fn calibrate_tsc() -> f64 {
    let (t, c) = (Instant::now(), rdtsc());
    std::thread::sleep(Duration::from_millis(5));
    t.elapsed().as_nanos() as f64 / rdtsc().wrapping_sub(c).max(1) as f64
}

/// Log2 buckets of a histogram site.
pub const BUCKETS: usize = 48;

/// Where each site's records lie in a [`Record`]: the offset of its detail
/// slots and of its histogram, and the record's total length.
#[derive(Clone, Debug)]
pub struct Layout {
    /// Per site, the offset of its first detail slot.
    offsets: Vec<u32>,
    /// Per site, the offset of its histogram's buckets.
    hists: Vec<u32>,
    /// Detail slots in all.
    len: usize,
    /// Histogram buckets in all.
    hist_len: usize,
    /// Schemas the slot keys are sized for.
    schemas: usize,
    /// Size bands those keys are sized for.
    bands: usize,
}

impl Layout {
    /// The layout of records for `schemas` schemas and `bands` size bands;
    /// either count is clamped to one so every detail slot exists.
    pub fn new(schemas: usize, bands: usize) -> Self {
        let (mut offsets, mut hists, mut len, mut hist_len) = (Vec::new(), Vec::new(), 0usize, 0usize);
        for s in SITES {
            offsets.push(len as u32);
            len += match s.detail {
                Detail::None => 1,
                Detail::Schema => schemas.max(1),
                Detail::SchemaBand => schemas.max(1) * bands.max(1) * 2,
            };
            hists.push(hist_len as u32);
            if s.kind == Kind::Hist {
                hist_len += BUCKETS;
            }
        }
        Self { offsets, hists, len, hist_len, schemas, bands }
    }

    /// Words of a record: calls and ticks per detail slot, then histograms.
    pub fn words(&self) -> usize {
        2 * self.len + self.hist_len
    }
}

/// One thread's site records, as plain words.
#[derive(Clone, Debug, Default)]
pub struct Record {
    /// Calls then ticks per detail slot, then the histograms ([`Layout::words`]).
    pub words: Vec<u64>,
}

impl Record {
    /// Zeroed words for `layout`.
    pub fn new(layout: &Layout) -> Self {
        Self { words: vec![0; layout.words()] }
    }
    /// Adds another record's words into this one, growing to its length.
    pub fn add(&mut self, o: &Record) {
        if self.words.len() < o.words.len() {
            self.words.resize(o.words.len(), 0);
        }
        for (a, b) in self.words.iter_mut().zip(&o.words) {
            *a += b;
        }
    }
}

/// Timing hooks; see the module documentation.
pub trait Profiler: Send + Sync {
    /// Whether this instantiation times anything; false lets a caller skip
    /// the hooks entirely, so the untimed code has no clock reads.
    const ENABLED: bool;
    /// What a site's start returns: a clock reading, or `()` for [`Off`].
    type Stamp: Copy;
    /// Whether `site` records; constant false for [`Off`].
    fn on(&self, site: Site) -> bool;
    /// The clock if `site` records, else an empty stamp.
    fn start(&self, site: Site) -> Self::Stamp;
    /// One call of `site` since `at`; nothing for an empty stamp.
    fn stop(&mut self, site: Site, at: Self::Stamp);
    /// As [`Self::stop`], into detail slot `key` of the site.
    fn stop_at(&mut self, site: Site, key: usize, at: Self::Stamp);
    /// Adds `ticks` already measured to `site` (one call).
    fn add_ticks(&mut self, site: Site, ticks: u64);
    /// The ticks from `at` to now (0 for an empty stamp).
    fn since(&self, at: Self::Stamp) -> u64;
    /// The thread's records, if it keeps any.
    fn record(&self) -> Option<&Record>;
    /// A raw clock reading if `site` records, else 0: a stamp to keep
    /// across calls (a read's submission, say).
    fn mark(&self, site: Site) -> u64;
    /// One call of `site` since a [`Self::mark`]; nothing for 0.
    fn since_mark(&mut self, site: Site, mark: u64);
    /// A fresh profiler with the same sites, for another thread.
    fn fork(&self) -> Self;
    /// What turns records into a report; none for [`Off`].
    fn setup(&self) -> Option<Setup>;
}

/// No profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct Off;

const _: () = assert!(std::mem::size_of::<Off>() == 0);

impl Profiler for Off {
    const ENABLED: bool = false;
    type Stamp = ();
    #[inline(always)]
    fn on(&self, _: Site) -> bool {
        false
    }
    #[inline(always)]
    fn start(&self, _: Site) {}
    #[inline(always)]
    fn stop(&mut self, _: Site, _: ()) {}
    #[inline(always)]
    fn stop_at(&mut self, _: Site, _: usize, _: ()) {}
    #[inline(always)]
    fn add_ticks(&mut self, _: Site, _: u64) {}
    #[inline(always)]
    fn since(&self, _: ()) -> u64 {
        0
    }
    #[inline(always)]
    fn record(&self) -> Option<&Record> {
        None
    }
    #[inline(always)]
    fn mark(&self, _: Site) -> u64 {
        0
    }
    #[inline(always)]
    fn since_mark(&mut self, _: Site, _: u64) {}
    fn fork(&self) -> Self {
        Off
    }
    fn setup(&self) -> Option<Setup> {
        None
    }
}

/// Profiling of the sites in a mask.
pub struct On {
    /// The enabled sites, as bits of [`Site`] discriminants.
    mask: u64,
    clock: Clock,
    layout: std::sync::Arc<Layout>,
    /// This thread's records, written only by it.
    rec: Record,
}

impl On {
    /// A profiler for the sites in `mask`, reading `clock` and laying its
    /// records out as `layout` does (shared with every fork).
    pub fn new(mask: SiteMask, clock: Clock, layout: std::sync::Arc<Layout>) -> Self {
        let rec = Record::new(&layout);
        Self { mask: mask.0, clock, layout, rec }
    }

    /// The detail slot of `site` and `key` in this thread's record.
    #[inline(always)]
    fn slot(&self, site: Site, key: usize) -> usize {
        // SAFETY: the layout has an offset for every site.
        unsafe { *self.layout.offsets.get_unchecked(site as usize) as usize + key }
    }

    /// One call of `site` in detail slot `key` taking `ticks`, and one bucket
    /// of its histogram when the site keeps one (bucket = log2 of the ticks,
    /// capped at the last).
    #[inline(always)]
    fn add(&mut self, site: Site, key: usize, ticks: u64) {
        let i = self.slot(site, key);
        let len = self.layout.len;
        debug_assert!(len + i < self.rec.words.len());
        // SAFETY: detail slots lie below the layout's length, and the record
        // holds calls then ticks for each.
        unsafe {
            *self.rec.words.get_unchecked_mut(i) += 1;
            *self.rec.words.get_unchecked_mut(len + i) += ticks;
        }
        if SITES[site as usize].kind == Kind::Hist {
            let b = (63 - ticks.max(1).leading_zeros() as usize).min(BUCKETS - 1);
            let h = 2 * len + self.layout.hists[site as usize] as usize + b;
            self.rec.words[h] += 1;
        }
    }
}

impl Profiler for On {
    const ENABLED: bool = true;
    type Stamp = u64;
    #[inline(always)]
    fn on(&self, site: Site) -> bool {
        self.mask >> site as u32 & 1 != 0
    }
    #[inline(always)]
    fn start(&self, site: Site) -> u64 {
        if self.on(site) { self.clock.now() } else { 0 }
    }
    #[inline(always)]
    fn stop(&mut self, site: Site, at: u64) {
        if at != 0 {
            let t = self.clock.now().saturating_sub(at);
            self.add(site, 0, t);
        }
    }
    #[inline(always)]
    fn stop_at(&mut self, site: Site, key: usize, at: u64) {
        if at != 0 {
            let t = self.clock.now().saturating_sub(at);
            self.add(site, key, t);
        }
    }
    #[inline(always)]
    fn add_ticks(&mut self, site: Site, ticks: u64) {
        if self.on(site) {
            self.add(site, 0, ticks);
        }
    }
    #[inline(always)]
    fn since(&self, at: u64) -> u64 {
        if at == 0 { 0 } else { self.clock.now().saturating_sub(at) }
    }
    fn record(&self) -> Option<&Record> {
        Some(&self.rec)
    }
    #[inline(always)]
    fn mark(&self, site: Site) -> u64 {
        self.start(site)
    }
    #[inline(always)]
    fn since_mark(&mut self, site: Site, mark: u64) {
        self.stop(site, mark);
    }
    fn fork(&self) -> Self {
        On::new(SiteMask(self.mask), self.clock, self.layout.clone())
    }
    fn setup(&self) -> Option<Setup> {
        Some(Setup { mask: SiteMask(self.mask), clock: self.clock, layout: self.layout.clone() })
    }
}

/// A run's profiling setup: the mask, the clock and the record layout.
#[derive(Clone)]
pub struct Setup {
    /// The sites the run records.
    pub mask: SiteMask,
    /// The run's clock, shared by every forked profiler.
    pub clock: Clock,
    /// The record layout of every thread's record.
    pub layout: std::sync::Arc<Layout>,
}

impl Setup {
    /// A setup for `mask` with a fresh clock and a layout for the run's
    /// schemas and size bands.
    pub fn new(mask: SiteMask, schemas: usize, bands: usize) -> Self {
        Self { mask, clock: Clock::new(), layout: std::sync::Arc::new(Layout::new(schemas, bands)) }
    }
    /// A profiler for one thread of the run.
    pub fn profiler(&self) -> On {
        On::new(self.mask, self.clock, self.layout.clone())
    }

    /// The merged records as a report.
    pub fn report(&self, rec: &Record) -> ProfileReport {
        let l = &self.layout;
        let ns = self.clock.nanos_per_tick();
        let mut words = rec.words.clone();
        words.resize(l.words(), 0);
        let mut sites: Vec<SiteReport> = Vec::new();
        for (i, info) in SITES.iter().enumerate() {
            if !self.mask.contains(Site::ALL[i]) {
                continue;
            }
            let n = match info.detail {
                Detail::None => 1,
                Detail::Schema => l.schemas.max(1),
                Detail::SchemaBand => l.schemas.max(1) * l.bands.max(1) * 2,
            };
            let at = l.offsets[i] as usize;
            let mut details = Vec::new();
            let (mut calls, mut ticks) = (0u64, 0u64);
            for k in 0..n {
                let (c, t) = (words[at + k], words[l.len + at + k]);
                calls += c;
                ticks += t;
                if info.detail != Detail::None && c > 0 {
                    details.push(DetailReport { key: k as u32, calls: c, time: Duration::from_nanos((t as f64 * ns) as u64) });
                }
            }
            let hist = (info.kind == Kind::Hist).then(|| {
                let h = 2 * l.len + l.hists[i] as usize;
                (0..BUCKETS).filter(|&b| words[h + b] > 0).map(|b| ((((1u64 << b) as f64) * ns) as u64, words[h + b])).collect()
            });
            sites.push(SiteReport { name: info.name, calls, time: Duration::from_nanos((ticks as f64 * ns) as u64), self_time: Duration::ZERO, details, hist });
        }
        // Self time: a site's time less its recorded children's.
        for i in 0..sites.len() {
            let name = sites[i].name;
            let child: Duration = SITES
                .iter()
                .zip(Site::ALL)
                .filter(|(info, _)| info.parent.is_some_and(|p| SITES[p as usize].name == name))
                .filter_map(|(info, _)| sites.iter().find(|s| s.name == info.name))
                .map(|s| s.time)
                .sum();
            sites[i].self_time = sites[i].time.saturating_sub(child);
        }
        ProfileReport { schemas: l.schemas, bands: l.bands, sites }
    }
}

/// One detail slot of a site.
#[derive(Clone, Debug, PartialEq)]
pub struct DetailReport {
    /// The slot: the schema, or `(schema * bands + band) * 2 + partial`.
    pub key: u32,
    /// Calls recorded in the slot.
    pub calls: u64,
    /// Time inside those calls.
    pub time: Duration,
}

/// One site's totals, as a report shows them.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteReport {
    /// The site's name.
    pub name: &'static str,
    /// Calls recorded.
    pub calls: u64,
    /// Time inside the site, including recorded child sites.
    pub time: Duration,
    /// Time less the recorded child sites.
    pub self_time: Duration,
    /// The site's detail slots that recorded calls.
    pub details: Vec<DetailReport>,
    /// Histogram sites: (lower bound in nanoseconds, count) per nonempty
    /// log2 bucket.
    pub hist: Option<Vec<(u64, u64)>>,
}

/// What a profiled run recorded, per enabled site.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProfileReport {
    /// Schemas the run's sites were laid out for.
    pub schemas: usize,
    /// Size bands the run's sites were laid out for.
    pub bands: usize,
    /// One entry per enabled site, in [`Site`] order.
    pub sites: Vec<SiteReport>,
}

impl ProfileReport {
    /// The report of the site named `name`, if it was recorded.
    pub fn site(&self, name: &str) -> Option<&SiteReport> {
        self.sites.iter().find(|s| s.name == name)
    }
    /// A site's total time, zero when it was not recorded.
    pub fn time(&self, name: &str) -> Duration {
        self.site(name).map_or(Duration::ZERO, |s| s.time)
    }
}

/// Snapshot slots: one per thread, each a seqlock over plain words, so a
/// reader never blocks a writer.
pub struct Snapshots {
    slots: Box<[Slot]>,
}

/// One thread's slot: an even `seq` means the words are stable, an odd one
/// that the thread is between `publish`'s two stores.
struct Slot {
    seq: AtomicU64,
    words: Box<[AtomicU64]>,
}

impl Snapshots {
    /// `threads` slots of `words` zeroed words each.
    pub fn new(threads: usize, words: usize) -> Self {
        Self { slots: (0..threads).map(|_| Slot { seq: AtomicU64::new(0), words: (0..words).map(|_| AtomicU64::new(0)).collect() }).collect() }
    }

    /// Publishes thread `i`'s words (only that thread writes its slot). The
    /// odd sequence between the two stores marks the words as in flux.
    pub fn publish(&self, i: usize, words: impl Iterator<Item = u64>) {
        let s = &self.slots[i];
        let seq = s.seq.load(Relaxed);
        s.seq.store(seq + 1, Relaxed);
        std::sync::atomic::fence(Release);
        for (a, w) in s.words.iter().zip(words) {
            a.store(w, Relaxed);
        }
        s.seq.store(seq + 2, Release);
    }

    /// The words of every slot, summed.
    pub fn sum(&self) -> Vec<u64> {
        let n = self.slots.first().map_or(0, |s| s.words.len());
        let mut out = vec![0u64; n];
        let mut tmp = vec![0u64; n];
        for s in self.slots.iter() {
            // Read the words, then retry unless the sequence was even and has
            // not moved: a publish in between invalidated the copy.
            loop {
                let a = s.seq.load(Acquire);
                if a & 1 == 1 {
                    std::hint::spin_loop();
                    continue;
                }
                for (t, w) in tmp.iter_mut().zip(s.words.iter()) {
                    *t = w.load(Relaxed);
                }
                std::sync::atomic::fence(Acquire);
                if s.seq.load(Relaxed) == a {
                    break;
                }
            }
            for (o, t) in out.iter_mut().zip(&tmp) {
                *o += t;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_compose_sites() {
        let io = SiteMask::parse(&["io"]).unwrap();
        assert!(io.contains(Site::IoSubmit) && io.contains(Site::IoLatency) && !io.contains(Site::Decode));
        let both = SiteMask::parse(&["io", "decode"]).unwrap();
        assert_eq!(both.0, io.0 | 1 << Site::Decode as u32);
        assert_eq!(SiteMask::parse(&["all"]).unwrap(), SiteMask::ALL);
        assert!(SiteMask::parse(&["nope"]).is_err());
    }

    #[test]
    fn disabled_sites_record_nothing() {
        let setup = Setup::new(SiteMask::parse(&["decode"]).unwrap(), 2, 3);
        let mut p = setup.profiler();
        let t = p.start(Site::Map);
        assert_eq!(t, 0);
        p.stop_at(Site::Map, 1, t);
        let t = p.start(Site::Decode);
        p.stop(Site::Decode, t);
        let r = setup.report(p.record().unwrap());
        assert_eq!(r.sites.len(), 1);
        assert_eq!(r.site("decode").unwrap().calls, 1);
    }

    #[test]
    fn histograms_and_self_time() {
        let setup = Setup::new(SiteMask::ALL, 1, 1);
        let mut p = setup.profiler();
        p.add_ticks(Site::IoLatency, 1000);
        p.add_ticks(Site::IoLatency, 3000);
        p.add_ticks(Site::Worker, 10_000);
        p.add_ticks(Site::Decode, 4000);
        let r = setup.report(p.record().unwrap());
        assert_eq!(r.site("io.latency").unwrap().hist.as_ref().unwrap().iter().map(|b| b.1).sum::<u64>(), 2);
        let w = r.site("worker").unwrap();
        assert!(w.self_time < w.time);
    }

    #[test]
    fn snapshots_sum_slots() {
        let s = Snapshots::new(2, 3);
        s.publish(0, [1, 2, 3].into_iter());
        s.publish(1, [10, 20, 30].into_iter());
        assert_eq!(s.sum(), vec![11, 22, 33]);
    }
}
