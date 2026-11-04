//! A run: one huge-page grid of frames, a plan ring, its producer and the
//! workers.
//!
//! The grid holds the plan's cache frames, then each worker's private
//! frames (its misses). Every cache frame has a generation word: the
//! access a plan admits a frame at publishes its generation once the frame
//! is written, and every other use waits for it. Frames are reused only as
//! the plan says, so no run-time state tracks what they hold.
//!
//! The plan reaches the workers through the plan ring: from the online
//! compiler (a thread pool of the run), or from a thread expanding a plan
//! file's residencies; a run without a plan reads every page privately.
//! The run picks its instance once: profiled or not, pages or pages and
//! rows.

pub(crate) mod io;
pub(crate) mod progress;
mod worker;

use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::*};
use std::time::{Duration, Instant};

use crate::loader::catalog::CellId;
use crate::loader::config::{PROCESS_BOUNDS, RunStats};
use crate::loader::error::{Error, Result, config};
use crate::loader::loader::{Planning, Run};
use crate::loader::memory::HugeBuf;
use crate::loader::planring::{EntryBuf, PlanRing};
use crate::loader::schema::Graph;
use crate::opt::engine::{self, Output, Params};
use crate::opt::plan::{Object, Sweep};
use crate::opt::records::Records;
use crate::prof::{Off, On, Profiler, Record, Snapshots};

use io::IoBackend;
use progress::Progress;
use worker::Worker;

/// Pages only, or pages and rows: the mode a worker instance is built for.
pub(crate) trait Mode: Send + Sync + 'static {
    /// Whether the run may cache rows, i.e. whether the plan has any.
    const ROWS: bool;
}
/// Worker instance without row caching: every access decodes its records.
pub(crate) struct Pages;
impl Mode for Pages {
    const ROWS: bool = false;
}
/// Worker instance with row caching: cached rows are gathered and admitted.
pub(crate) struct PagesRows;
impl Mode for PagesRows {
    const ROWS: bool = true;
}

/// The frames of a run: the plan's cache frames, then one region of private
/// frames per worker. Frames are `page` bytes each and all live in one
/// huge-page mapping, so a worker's frames are contiguous and its node-local
/// first touch decides where they land.
pub(crate) struct Grid {
    buf: HugeBuf,
    page: usize,
    /// Cache frames, addressed by plans as frame ids.
    cache: usize,
    /// Private frames per worker.
    private: usize,
}

impl Grid {
    /// `cache` frames shared by the plan, then `private` frames for each of
    /// `workers` workers.
    fn new(page: usize, cache: usize, workers: usize, private: usize) -> Result<Self> {
        let frames = cache.checked_add(workers.checked_mul(private).ok_or_else(|| config("frame count overflow"))?).ok_or_else(|| config("frame count overflow"))?;
        Ok(Self { buf: HugeBuf::zeroed(frames.checked_mul(page).ok_or_else(|| config("frame grid overflow"))?)?, page, cache, private })
    }
    /// Cache frame `f`.
    #[inline(always)]
    pub fn cache(&self, f: u32) -> *mut u8 {
        debug_assert!((f as usize) < self.cache);
        // SAFETY: plans name frames below the cache's frame count (checked
        // when a plan file binds, by construction for the compiler).
        unsafe { self.buf.ptr().add(f as usize * self.page) }
    }
    /// Worker `w`'s private frames.
    pub fn private(&self, w: usize) -> *mut u8 {
        // SAFETY: the grid holds every worker's region after the cache.
        unsafe { self.buf.ptr().add((self.cache + w * self.private) * self.page) }
    }
}

/// Where the workers find plan entries.
#[derive(Clone, Copy)]
pub(crate) enum Source<'a> {
    /// No plan: every page is read privately.
    Empty,
    /// A published plan ring; workers read entries below its frontier.
    Ring(&'a PlanRing),
}

/// Everything the run's threads share: the inputs' descriptors, the frame
/// grid and its generation words, the plan ring and the progress frontier.
///
/// `gens[f]` is cache frame `f`'s generation word: the access that admits into
/// the frame stores the generation an entry names (with release), and every use
/// of that residency waits for it (with acquire). A `Shared` is borrowed by
/// every worker thread of a scoped run and dropped with it; `poisoned` makes
/// the run leak that borrow instead, when a failed read may still be writing
/// into frames.
pub(crate) struct Shared<'a> {
    /// The run being produced: ring, generation and window.
    pub run: &'a Run<'a>,
    /// Inputs, schemas and the tables composed from both.
    pub graph: &'a Graph,
    /// Where plan entries come from.
    pub source: Source<'a>,
    /// Claims and the execution frontier.
    pub progress: Progress,
    /// The cache and private frames.
    pub grid: Grid,
    /// One generation word per cache frame.
    pub gens: Box<[AtomicU32]>,
    /// The inputs' descriptors, by file id (registered with the ring when
    /// possible).
    pub fds: Vec<i32>,
    /// How the workers issue reads, as configured.
    pub backend: IoBackend,
    /// Page size of every input, and of each frame.
    pub page: usize,
    /// Per schema: its region's offset in a slot.
    pub offsets: Vec<usize>,
    /// Per schema: bytes of one cell's output.
    pub cell_bytes: Vec<usize>,
    /// Schemas: accesses per position.
    pub schemas: u64,
    /// Positions per slot.
    pub slot_cells: u64,
    /// Accesses per claim, a whole number of positions.
    pub claim_batch: u64,
    /// Each worker's in-flight position limit.
    pub depth: usize,
    /// Private frames per worker: the in-flight depth times the p99.9 record,
    /// plus one largest record, times the schemas.
    pub private_frames: usize,
    /// Record pages at the 99.9th percentile, what a claim's private frames
    /// are sized by.
    pub normal_pages: usize,
    /// The per-worker statistics slots.
    pub snapshots: std::sync::Arc<Snapshots>,
    /// A read failed while its frames may still be owned: never unmap them.
    pub poisoned: AtomicBool,
    stop: AtomicBool,
}

impl Shared<'_> {
    /// Whether the run must unwind: the progress was stopped or the ring was
    /// aborted.
    pub fn cancelled(&self) -> bool {
        self.progress.stopped() || self.run.ring.aborted()
    }
    fn stop(&self) {
        self.progress.stop();
        self.run.cancel();
        self.stop.store(true, Release);
        if let Source::Ring(r) = self.source {
            r.stop();
        }
    }
}

/// Live statistics of a running run, for [`Run::snapshot`].
pub(crate) struct Live {
    /// Set when the execution sets up, before its threads start; empty
    /// before that, so a snapshot of an unstarted run reports zeros.
    pub state: std::sync::OnceLock<LiveState>,
}

/// What a run leaves for [`Run::snapshot`]: the publication slots, the start
/// time and, for a profiled run, the site setup.
pub(crate) struct LiveState {
    /// Per-worker published statistics words.
    pub snapshots: std::sync::Arc<Snapshots>,
    /// When the execution started; snapshots measure their elapsed time from
    /// here.
    pub started: Instant,
    /// The run's profiled sites, if it profiles.
    pub setup: Option<crate::prof::Setup>,
}

impl Live {
    /// A handle with nothing published yet.
    pub fn new() -> Self {
        Self { state: std::sync::OnceLock::new() }
    }

    /// The workers' statistics as last published.
    pub fn snapshot(&self, graph: &Graph) -> RunStats {
        let Some(st) = self.state.get() else { return RunStats::empty(graph) };
        let words = st.snapshots.sum();
        let n = RunStats::snapshot_words(graph.schemas.len());
        let mut s = RunStats::from_words(&words[..n.min(words.len())], graph.schemas.len());
        s.elapsed = st.started.elapsed();
        if let Some(setup) = &st.setup {
            s.profile = Some(setup.report(&Record { words: words[n..].to_vec() }));
        }
        s
    }
}

/// Whether a run under this planning may cache rows, which decides the worker
/// instance: a fixed plan that holds any row residency, or an online plan with
/// rows enabled, a cost model, the time objective and at least one cache frame.
pub(crate) fn uses_rows(planning: &Planning, page: usize) -> bool {
    match planning {
        Planning::Fixed(plan) => plan.residencies().iter().any(|r| matches!(r.object, Object::Row(_))),
        Planning::Online(o) => o.rows && o.cost.is_some() && o.objective == crate::opt::PlanObjective::Time && o.cache_bytes >= page,
        Planning::None => false,
    }
}

/// Executes a run: picks the worker instance and calls [`go`].
pub(crate) fn run(run: &Run<'_>) -> Result<RunStats> {
    let rows = uses_rows(&run.planning, run.loader.graph.page_size);
    match (run.loader.config.profile, rows) {
        (None, false) => go::<Off, Pages>(run, Off),
        (None, true) => go::<Off, PagesRows>(run, Off),
        (Some(mask), rows) => {
            let base = crate::prof::Setup::new(mask, run.loader.graph.schemas.len(), PROCESS_BOUNDS.len()).profiler();
            if rows { go::<On, PagesRows>(run, base) } else { go::<On, Pages>(run, base) }
        }
    }
}

/// Entry slots and pool words of a plan ring: slots for the window rounded up
/// to whole chunks plus two chunks (at least two positions), and words for two
/// whole positions' pages, extra reads and cached rows, or a per-entry
/// allowance, whichever is larger; workers cannot publish a schema group
/// piecemeal.
pub(crate) fn plan_ring_sizes(graph: &Graph, window: u64, chunk: u64, rows: bool, block_pages: u32) -> Result<(u64, u64)> {
    let overflow = || config("plan ring size overflow");
    let schemas = graph.schemas.len() as u64;
    let entries = window.div_ceil(chunk).checked_add(2)
        .and_then(|n| n.checked_mul(chunk)).ok_or_else(overflow)?.max(2 * schemas);
    let source = &graph.source;
    let row_words = if rows {
        let frames = super::rows::frames(false, source.max_nnz, 8, graph.page_size)
            .max(super::rows::frames(true, source.max_genes, 8, graph.page_size));
        1 + (frames as u64).div_ceil(2)
    } else { 0 };
    // Each record page can anchor one block, with headers for its extras.
    let page_words = (source.max_pages as u64).checked_mul(1 + 3 * block_pages.saturating_sub(1) as u64).ok_or_else(overflow)?;
    let position_words = page_words.checked_add(row_words)
        .and_then(|n| n.checked_mul(2 * schemas)).ok_or_else(overflow)?;
    let words = entries.checked_mul((source.max_pages as u64).clamp(4, 64) * 2)
        .ok_or_else(overflow)?.max(position_words).max(64);
    if entries > (isize::MAX as u64) / 16 || words > (isize::MAX as u64) / 8 {
        return Err(overflow());
    }
    Ok((entries, words))
}

/// Runs one execution: sizes the private frames and the plan ring, starts the
/// plan thread and the workers, merges their statistics and returns the first
/// failure that is not [`Error::Aborted`].
///
/// The workers are scoped threads borrowing `Shared`; a read that failed may
/// still be writing into a frame, so a poisoned run forgets the shared state
/// instead of unmapping frames under a live request.
fn go<P: Profiler, M: Mode>(run: &Run<'_>, base: P) -> Result<RunStats> {
    let setup_started = Instant::now();
    let graph = &run.loader.graph;
    let cfg = &run.loader.config;
    let len = run.accesses.len();
    let schemas = graph.schemas.len() as u64;
    if len == 0 {
        let mut stats = RunStats::empty(graph);
        if matches!(run.planning, Planning::Online(_)) {
            stats.online = Some(crate::opt::OnlineStats::default());
        }
        return Ok(stats);
    }
    let page = graph.page_size;
    let cache = match &run.planning {
        Planning::None => 0,
        Planning::Fixed(plan) => plan.cache_frames() as usize,
        Planning::Online(o) => o.cache_bytes / page,
    };
    let depth = cfg.max_inflight_per_worker;
    let normal = graph.source.normal_pages();
    let max_pages = graph.source.max_pages.max(1);
    // A task owns every schema's input until it can deliver them together.
    // Its oversized-record reserve must fit a whole position.
    let private = depth.checked_mul(normal)
        .and_then(|n| n.checked_add(max_pages))
        .and_then(|n| n.checked_mul(schemas as usize))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| config("private frame count exceeds u32"))?;
    let grid = Grid::new(page, cache, cfg.workers, private)?;
    // A claim covers whole positions (a multiple of `schemas`) and never more
    // than a worker may keep in flight: a task is one position, so a deeper
    // claim could not start.
    let claim = (cfg.claim_batch as u64).max(schemas).div_ceil(schemas) * schemas;
    let claim = claim.min((depth as u64) * schemas).max(schemas);
    // The plan ring holds the window and a chunk to spare on both sides.
    let chunk = match &run.planning {
        Planning::Online(o) => o.chunk.unwrap_or(crate::opt::DEFAULT_CHUNK),
        _ => crate::opt::DEFAULT_CHUNK,
    };
    let blocks = match &run.planning {
        Planning::Online(o) => o.block_pages.unwrap_or(1),
        Planning::Fixed(_) => crate::opt::plan::MAX_BLOCK_PAGES,
        Planning::None => 1,
    };
    let ring = if matches!(run.planning, Planning::None) { None } else {
        let (entries, words) = plan_ring_sizes(graph, run.window, chunk, M::ROWS, blocks)?;
        Some(PlanRing::new(entries, words))
    };
    let record_words = base.record().map_or(0, |r| r.words.len());
    let snapshots = std::sync::Arc::new(Snapshots::new(cfg.workers, RunStats::snapshot_words(graph.schemas.len()) + record_words));
    let _ = run.live.state.set(LiveState { snapshots: snapshots.clone(), started: Instant::now(), setup: base.setup() });
    let shared = Shared {
        run,
        graph,
        source: ring.as_ref().map_or(Source::Empty, Source::Ring),
        progress: Progress::new(cfg.workers, len, run.window, schemas),
        grid,
        gens: (0..cache).map(|_| AtomicU32::new(0)).collect(),
        fds: graph.files.iter().map(|f| f.file.as_raw_fd()).collect(),
        backend: cfg.backend,
        page,
        offsets: (0..graph.schemas.len()).map(|s| run.ring.region_offset(s)).collect(),
        cell_bytes: (0..graph.schemas.len()).map(|s| run.ring.cell_bytes(s)).collect(),
        schemas,
        slot_cells: graph.slot_cells,
        claim_batch: claim,
        depth,
        private_frames: private,
        normal_pages: normal,
        snapshots,
        poisoned: AtomicBool::new(false),
        stop: AtomicBool::new(false),
    };
    let setup_elapsed = setup_started.elapsed();
    let started = Instant::now();
    let (results, produced) = std::thread::scope(|scope| {
        let shared = &shared;
        let base = &base;
        let producer = match (&run.planning, shared.source) {
            (Planning::Online(options), Source::Ring(ring)) => Some(
                std::thread::Builder::new()
                    .name("cellpack-plan".into())
                    .spawn_scoped(scope, move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(Option<crate::opt::OnlineStats>, Option<Record>)> {
                            let _affinity = crate::opt::compile::Affinity::bind(&options.cpus)?;
                            let records = Records::build(graph, options.threads.max(1))?;
                            let p = Params::new(graph, options, len, true, true, records.cells);
                            let frontier = || shared.progress.scan();
                            let fork = || base.fork();
                            let (_, stats, rec) = engine::compile::<P>(graph, &records, run.accesses, options, p, Output::Ring { ring, frontier: &frontier }, &shared.stop, &fork)?;
                            Ok((Some(stats), rec))
                        }))
                        .unwrap_or(Err(Error::WorkerPanic));
                        if result.is_err() {
                            shared.stop();
                        }
                        result
                    })
                    .map_err(|e| { shared.stop(); Error::Io(e) }),
            ),
            (Planning::Fixed(plan), Source::Ring(ring)) => Some(
                std::thread::Builder::new()
                    .name("cellpack-plan".into())
                    .spawn_scoped(scope, move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(Option<crate::opt::OnlineStats>, Option<Record>)> {
                            replay(graph, plan, run.accesses, ring, chunk, &|| shared.progress.scan())?;
                            Ok((None, None))
                        }))
                        .unwrap_or(Err(Error::WorkerPanic));
                        if result.is_err() {
                            shared.stop();
                        }
                        result
                    })
                    .map_err(|e| { shared.stop(); Error::Io(e) }),
            ),
            _ => None,
        };
        let handles: Vec<_> = (0..cfg.workers)
            .map(|id| {
                std::thread::Builder::new()
                    .name(format!("cellpack-w{id}"))
                    .spawn_scoped(scope, move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if !cfg.cpus.is_empty() {
                                pin(cfg.cpus[id % cfg.cpus.len()])?;
                            }
                            Worker::<P, M>::new(id, shared, base.fork())?.run()
                        }))
                        .unwrap_or(Err(Error::WorkerPanic));
                        if result.is_err() {
                            shared.stop();
                        }
                        result
                    })
                    .map_err(|e| {
                        shared.stop();
                        Error::Io(e)
                    })
            })
            .collect();
        let results = handles.into_iter().map(|h| h.and_then(|h| h.join().unwrap_or(Err(Error::WorkerPanic)))).collect::<Vec<_>>();
        if let Source::Ring(r) = shared.source {
            r.stop();
        }
        shared.stop.store(true, Release);
        let produced = producer.map(|c| c.and_then(|h| h.join().unwrap_or(Err(Error::WorkerPanic))));
        (results, produced)
    });
    let mut stats = RunStats::empty(graph);
    stats.elapsed = started.elapsed();
    stats.setup_elapsed = setup_elapsed;
    stats.peak_span = shared.progress.peak_span.load(Relaxed);
    stats.cache_bytes = (cache * page) as u64;
    stats.private_bytes = (cfg.workers * private * page) as u64;
    // The root cause wins over the Aborted seen by the others.
    let mut failure = None;
    let mut merged: Option<Record> = None;
    let mut merge = |r: Option<&Record>| {
        if let Some(r) = r {
            match merged.as_mut() {
                Some(m) => m.add(r),
                None => merged = Some(r.clone()),
            }
        }
    };
    for result in results {
        match result {
            Ok((s, prof)) => {
                stats.add(&s);
                merge(prof.record());
            }
            Err(e) => {
                if failure.as_ref().is_none_or(|f| matches!(f, Error::Aborted)) {
                    failure = Some(e);
                }
            }
        }
    }
    match produced {
        Some(Ok((online, rec))) => {
            stats.online = online;
            merge(rec.as_ref());
        }
        Some(Err(e)) if failure.as_ref().is_none_or(|f| matches!(f, Error::Aborted)) => failure = Some(e),
        _ => {}
    }
    if let Some(setup) = base.setup() {
        stats.profile = Some(setup.report(&merged.unwrap_or_default()));
    }
    if shared.poisoned.load(Acquire) {
        // A failed read may still write into frames: never unmap them.
        std::mem::forget(shared);
    }
    failure.map_or(Ok(stats), Err)
}

/// Expands a plan file into the plan ring, a chunk at a time.
fn replay(graph: &Graph, plan: &crate::opt::Plan, accesses: &dyn crate::loader::Accesses, ring: &PlanRing, chunk: u64, frontier: &dyn Fn() -> u64) -> Result<()> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let records = Records::build(graph, threads)?;
    let mut sweep = Sweep::new(plan, graph, &records);
    let schemas = graph.schemas.len() as u64;
    let len = accesses.len();
    let mut buf = EntryBuf::default();
    let mut cells: Vec<CellId> = Vec::new();
    let mut prev: CellId = 0;
    let mut at = 0;
    while at < len {
        let n = (len - at).min(chunk);
        cells.resize(n as usize, 0);
        accesses.fill(at, &mut cells)?;
        buf.clear(at);
        for (i, &cell) in cells.iter().enumerate() {
            let a = at + i as u64;
            let same = !a.is_multiple_of(schemas) && prev == cell;
            sweep.step(a, cell, same, &mut buf)?;
            prev = cell;
        }
        if !ring.publish(&buf, frontier, || std::thread::sleep(Duration::from_micros(20))) {
            return Ok(());
        }
        at += n;
    }
    sweep.finish()
}

/// Pins the calling thread to `cpu`.
pub(crate) fn pin(cpu: usize) -> Result<()> {
    if cpu >= libc::CPU_SETSIZE as usize {
        return Err(config("worker CPU index exceeds CPU_SETSIZE"));
    }
    // SAFETY: cpu_set_t is initialized and the index is bounded; this call
    // changes only the current thread's affinity.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
