//! The compiler: residencies chosen over a horizon `H` of reuse intervals,
//! advanced, frozen and published in chunks of `C` accesses.
//!
//! An object's consecutive uses `(u, v)` form a reuse interval. Keeping the
//! object from `u` to `v` holds its frames over `[u − D, v]` (or over
//! `[u + 1, v]` when a residency already ends at `u`), and serves `v`.
//! Intervals are decided when their right end `v` arrives:
//!
//! - pages: kept whenever the frames fit (the interval form of Belady with
//!   bypass, exact for equal sizes and costs);
//! - rows (page+row): a row is kept when its value per frame-position beats
//!   a shadow price learned from the page layer (or, offline, by a global
//!   density order); a kept row removes the access's page uses.
//!
//! Interval lengths of at most `D` are decided like the others; the
//! horizon never filters them. With compiler frontier `X`, the frozen
//! prefix ends at `Z` (from `X − H`, or further ahead while a run's workers
//! wait for entries, never past `X − D − C`) and the published one at
//! `P = Z − D`:
//! decisions only touch intervals with `u >= Z`, so their spans start at
//! or after `P` and no published entry changes. Freezing `[Z, Z')` releases
//! the frames of residencies that ended there, allocates frames to those
//! admitted there (in admission order, from frames whose last residency
//! ended more than `D` before) and resolves every use to a frame and
//! generation.
//!
//! Work is split into shards by page group: a shard owns groups of
//! adjacent pages assigned by hashing the group index, and the rows of cells
//! whose records start in them, with a fixed share of the frames. Each
//! shard's steps run in order on one thread at a time, so results do not
//! depend on scheduling.

use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::loader::accesses::Accesses;
use crate::loader::catalog::CellId;
use crate::loader::error::{Error, Result};
use crate::loader::planring::{AccessPlan, EntryBuf, Extra, MISS, PlanRing, page_ref, ref_admit};
use crate::loader::schema::Graph;
use crate::opt::model::{Loads, PlanObjective, PlanOptions, Prices, RowPolicy};
use crate::opt::occupancy::Occupancy;
use crate::opt::plan::{Built, MAX_BLOCK_PAGES, Object};
use crate::opt::price::Pricing;
use crate::opt::records::{Record as CellRecord, Records, WARM_CHUNKS};
use crate::prof::{Profiler, Record, Site};

/// Pages per shard group.
pub(crate) const GROUP: u64 = 128;
/// Most shards: the cache is split further only while each shard keeps at
/// least [`MIN_SHARD_FRAMES`] frames.
const MAX_SHARDS: usize = 64;
/// Frames a shard should own at least.
const MIN_SHARD_FRAMES: u64 = 1 << 14;
const NIL: u32 = u32::MAX;
const NONE: u64 = u64::MAX;

/// Flags per access of a chunk.
const SAME: u8 = 1;
const HIT: u8 = 2;

/// What one compilation produces.
pub(crate) enum Output<'a> {
    /// Entries into a plan ring, as the loader's frontier frees room.
    Ring { ring: &'a PlanRing, frontier: &'a (dyn Fn() -> u64 + Sync) },
    /// The residencies, for a plan.
    Plan,
    /// Statistics only.
    Count,
}

/// What an online compilation did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OnlineStats {
    /// Accesses compiled.
    pub accesses: u64,
    /// Chunks the sequence was compiled in.
    pub chunks: u64,
    /// Accesses per chunk.
    pub chunk: u64,
    /// The horizon `H` in accesses (the whole sequence offline).
    pub horizon: u64,
    /// The reuse gap `D` in accesses.
    pub distance: u64,
    /// Shards the cache was split into.
    pub shards: usize,
    /// Page uses that read input.
    pub page_uses: u64,
    /// The uses a residency served.
    pub page_hits: u64,
    /// Page residencies admitted.
    pub page_admissions: u64,
    /// Row residencies admitted.
    pub row_admissions: u64,
    /// The accesses a cached row served.
    pub row_hits: u64,
    /// Extra admission reads (block reads), one per run of moved pages
    /// (`Site::PlanBlock`).
    pub block_reads: u64,
    /// Read requests the published accesses issue.
    pub requests: u64,
    /// Row candidates seen.
    pub row_candidates: u64,
    /// The row candidates kept.
    pub row_kept: u64,
    /// The page intervals refused for lack of frames.
    pub page_refused: u64,
    /// The last shadow price of cache capacity (seconds per frame-position),
    /// averaged over shards; zero while the page layer refuses nothing, since
    /// then capacity is slack.
    pub row_price: f64,
    /// The last resource prices (seconds per request, byte, core-second and
    /// memory byte), averaged over shards: which resource binds.
    pub prices: Prices,
    /// Modeled seconds of the published accesses (zero without a cost
    /// model).
    pub modeled_seconds: f64,
    /// The modeled core-seconds of the workers (zero without a cost model).
    pub worker_seconds: f64,
    /// Wall seconds compiling, the wait for plan ring space excluded.
    pub compile_seconds: f64,
    /// The seconds spent waiting for ring space.
    pub waiting_seconds: f64,
}

/// The shape of one compilation.
#[derive(Clone, Debug)]
pub(crate) struct Params {
    /// The reuse gap `D` in accesses (the slots' gap times the lanes).
    pub d: u64,
    /// The horizon `H` in accesses.
    pub horizon: u64,
    /// Accesses per chunk `C`.
    pub chunk: u64,
    /// Freeze as the plan ring's consumer needs (online runs).
    pub on_demand: bool,
    /// The compiler runs alongside the workers (online), on their cores.
    pub online: bool,
    /// Compiler threads (at least one).
    pub threads: usize,
}

impl Params {
    /// `online`: a finite horizon (else the whole sequence); `on_demand`:
    /// freeze further ahead while the plan ring's consumer would otherwise
    /// wait (a run starts early), else keep the whole horizon behind the
    /// compiler's frontier. The horizon is raised to at least `D + C` unless
    /// the sequence is shorter, and never passes its end.
    pub fn new(graph: &Graph, options: &PlanOptions, len: u64, online: bool, on_demand: bool, cells: u64) -> Self {
        let lanes = graph.lanes_per_slot();
        let d = options.gap.saturating_mul(lanes).max(1);
        let chunk = options.chunk.unwrap_or(crate::opt::model::DEFAULT_CHUNK).max(1);
        let horizon = if online {
            options.horizon.map_or(2 * cells * graph.schemas.len() as u64, |h| h.saturating_mul(lanes))
        } else {
            len
        };
        Self { d, horizon: horizon.max(d + chunk).min(len.max(1)), chunk, on_demand, online, threads: options.threads.max(1) }
    }
}

/// A residency during compilation.
#[derive(Clone, Copy)]
struct Res {
    start: u64,
    end: u64,
    /// Local key: a page of the shard, or a cell of the shard.
    key: u32,
    /// A page's frame, or a row's frame list.
    frame: u32,
    generation: u32,
    row: bool,
    live: bool,
    /// Admitted at `start` by a block read rather than its first use.
    moved: bool,
    /// Final: its frames return to the pool.
    rel: bool,
}

/// One shard's state.
struct Shard {
    j: usize,
    /// Frames the shard's share of the cache holds.
    cap: u32,
    /// Global id of the shard's first frame: its local frames are
    /// `frame_base + f`.
    frame_base: u32,
    occ: Occupancy,
    page_last: Vec<u64>,
    page_entry: Vec<u32>,
    page_res: Vec<u32>,
    row_last: Vec<u64>,
    row_entry: Vec<u32>,
    row_res: Vec<u32>,
    res: Vec<Res>,
    free: Vec<u32>,
    row_lists: Vec<Vec<u32>>,
    row_free: Vec<u32>,
    /// Per chunk on the ring: this shard's uses (offset, local key) with
    /// their decisions.
    page_uses: Vec<Vec<(u32, u32)>>,
    page_dec: Vec<Vec<u32>>,
    row_uses: Vec<Vec<(u32, u32)>>,
    row_dec: Vec<Vec<u32>>,
    by_start: Vec<Vec<u32>>,
    by_end: Vec<Vec<u32>>,
    /// Free frames by `last end + 1` (0: never used).
    heap: BinaryHeap<Reverse<(u64, u32)>>,
    /// Generation of each of the shard's frames.
    gens: Vec<u32>,
    /// The frozen frontier this shard has reached.
    z: u64,
    /// How far the occupancy blocks have been entered.
    entered: u64,
    lambda: Lambda,
    /// Resource prices of the chunk being decided, the value of a page hit
    /// at them, the chunk's share of cold decodes and the recent page hit
    /// rate; chunks priced so far.
    prices: Prices,
    page_value: f64,
    cold: f64,
    hit: f64,
    priced: u64,
    seen: Seen,
    /// Residencies that ended, when the caller collects a plan.
    out: Vec<Built>,
    /// Global ids of local keys.
    groups: Vec<u32>,
    cells: Vec<CellId>,
    stats: ShardStats,
}

/// This shard's recent decisions (decaying by half per chunk): page uses
/// and hits, and the loads its rows added (admissions less savings) over
/// its accesses. Shards see a hashed share of the pages and rows, so their
/// hit rates and row loads stand for the whole cache's.
#[derive(Default, Clone, Copy)]
struct Seen {
    uses: f64,
    hits: f64,
    rows: Loads,
    accesses: f64,
}

#[derive(Default, Clone, Copy)]
struct ShardStats {
    row_candidates: u64,
    row_kept: u64,
    page_refused: u64,
}

/// The quantile of refused lengths taken as the short ones.
const SHORT_REFUSED: f64 = 0.1;

/// The shadow price of capacity, per frame-position: what the page layer
/// would earn with a little more of it. More capacity first admits the
/// shortest of the page intervals it refuses, so the price is a page hit's
/// value over a short refused length (a low quantile of the recent refused
/// lengths, by power-of-two bins). While the page layer refuses nothing,
/// capacity is slack and costs nothing.
struct Lambda {
    refused: [f64; 64],
    n: f64,
    events: f64,
}

impl Lambda {
    fn record(&mut self) {
        self.events += 1.0;
        self.decay();
    }
    fn refuse(&mut self, len: u64) {
        self.refused[63 - len.max(1).leading_zeros() as usize] += 1.0;
        self.n += 1.0;
        self.record();
    }
    /// Halves the counts, and the threshold with them, once 65536 page
    /// interval decisions have accumulated, so old refusals fade.
    fn decay(&mut self) {
        if self.events >= 65536.0 {
            for h in &mut self.refused {
                *h *= 0.5;
            }
            self.n *= 0.5;
            self.events *= 0.5;
        }
    }
    /// A page hit's value over the shortest refused lengths that hold a
    /// [`SHORT_REFUSED`] share of the recent refusals, or zero when nothing
    /// was refused.
    fn price(&self, page_value: f64) -> f64 {
        if self.n < 1.0 {
            return 0.0;
        }
        let mut acc = 0.0;
        for (b, &h) in self.refused.iter().enumerate() {
            acc += h;
            if acc >= SHORT_REFUSED * self.n {
                return page_value / (1u64 << b) as f64 / std::f64::consts::SQRT_2;
            }
        }
        0.0
    }
}

/// Uses of one part and shard: (access offset, local key).
type Uses = Mutex<Vec<(u32, u32)>>;
/// Extra admissions of one shard: (access offset, page, ref).
type Extras = Mutex<Vec<(u32, u64, u64)>>;

/// One chunk's accesses, shared by the stages that touch it.
struct ChunkData {
    first: u64,
    n: usize,
    cells: Box<[AtomicU32]>,
    flags: Box<[AtomicU8]>,
    /// Per access: its first page ref slot (prefix within the chunk).
    slot: Box<[AtomicU32]>,
    part_slots: Box<[AtomicU32]>,
    refs: OnceLock<Box<[AtomicU64]>>,
    /// Uses scattered by access part and shard: `[part * shards + shard]`.
    page_uses: Box<[Uses]>,
    row_uses: Box<[Uses]>,
    /// Rows: generation with the admit bit (0: none), and where the frames
    /// are (shard << 32 | offset in that shard's list).
    row_gen: Box<[AtomicU32]>,
    row_at: Box<[AtomicU64]>,
    row_frames: Box<[Mutex<Vec<u32>>]>,
    /// Extra admissions per shard: (access offset, page, ref).
    extras: Box<[Extras]>,
    /// What the accesses take uncached, per generating part and in all.
    part_base: Box<[Mutex<Base>]>,
    base: OnceLock<Base>,
    gen_left: AtomicU32,
}

/// What a chunk's accesses take uncached, summed by the parts that generate
/// them and then in part order, so the sums do not depend on scheduling.
#[derive(Clone, Copy, Default)]
struct Base {
    accesses: u64,
    /// Records read (one by each access that does not repeat the previous
    /// access's cell), their pages, those decoded and those decoded in a cold
    /// chunk.
    reads: u64,
    pages: u64,
    decoded: u64,
    cold: u64,
    loads: Loads,
}

impl Base {
    fn add(&mut self, o: &Base) {
        self.accesses += o.accesses;
        self.reads += o.reads;
        self.pages += o.pages;
        self.decoded += o.decoded;
        self.cold += o.cold;
        self.loads.add(&o.loads);
    }
    /// The share of decodes in a cold chunk.
    fn cold_share(&self) -> f64 {
        if self.decoded == 0 { 0.0 } else { self.cold as f64 / self.decoded as f64 }
    }
    /// Requests per page of a missed read.
    fn rho(&self) -> f64 {
        if self.pages == 0 { 1.0 } else { self.reads as f64 / self.pages as f64 }
    }
}

impl ChunkData {
    fn new(first: u64, n: usize, parts: usize, shards: usize, rows: bool) -> Self {
        let z32 = |n: usize| (0..n).map(|_| AtomicU32::new(0)).collect::<Box<[_]>>();
        Self {
            first,
            n,
            cells: z32(n),
            flags: (0..n).map(|_| AtomicU8::new(0)).collect(),
            slot: z32(n + 1),
            part_slots: z32(parts),
            refs: OnceLock::new(),
            page_uses: (0..parts * shards).map(|_| Mutex::new(Vec::new())).collect(),
            row_uses: (0..if rows { parts * shards } else { 0 }).map(|_| Mutex::new(Vec::new())).collect(),
            row_gen: z32(if rows { n } else { 0 }),
            row_at: (0..if rows { n } else { 0 }).map(|_| AtomicU64::new(0)).collect(),
            row_frames: (0..if rows { shards } else { 0 }).map(|_| Mutex::new(Vec::new())).collect(),
            extras: (0..shards).map(|_| Mutex::new(Vec::new())).collect(),
            part_base: (0..parts).map(|_| Mutex::new(Base::default())).collect(),
            base: OnceLock::new(),
            gen_left: AtomicU32::new(parts as u32),
        }
    }
    #[inline]
    fn cell(&self, off: usize) -> CellId {
        self.cells[off].load(Relaxed)
    }
}

/// Keys of the shards.
struct Keys {
    shards: usize,
    group_shard: Vec<u8>,
    group_local: Vec<u32>,
    cell_shard: Vec<u8>,
    cell_local: Vec<u32>,
}

fn mix(x: u64) -> u64 {
    let mut x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The fixed part of a compilation.
struct Ctx<'a> {
    graph: &'a Graph,
    records: &'a Records,
    accesses: &'a dyn Accesses,
    pricing: Pricing<'a>,
    p: Params,
    len: u64,
    schemas: u64,
    rows: bool,
    block_pages: Option<u32>,
    parts: usize,
    keys: Keys,
    ring: u64,
    global: Vec<AtomicU8>,
}

impl Ctx<'_> {
    /// The frozen frontier once the compiler reaches `x`, from `z`: at least
    /// the whole horizon behind `x` and at least `z`, and further ahead while
    /// the plan's consumer needs entries sooner (`demand`: the published
    /// entries it can take), never past `x − D − C` (or the end of the
    /// sequence).
    fn target(&self, x: u64, z: u64, demand: Option<u64>) -> u64 {
        if x >= self.len {
            return self.len;
        }
        let (d, c) = (self.p.d, self.p.chunk);
        let full = x.saturating_sub(self.p.horizon);
        let near = x.saturating_sub(d + c);
        let wanted = demand.map_or(0, |e| near.min(e.saturating_add(d)));
        full.max(wanted).max(z)
    }

    /// The published frontier for frozen frontier `z`: `D` accesses behind it,
    /// or the end of the sequence.
    fn published(&self, z: u64) -> u64 {
        if z >= self.len { self.len } else { z.saturating_sub(self.p.d) }
    }

    /// The shard owning `page` and the shard's local key for it.
    #[inline]
    fn page_key(&self, page: u64) -> (usize, u32) {
        let g = (page / GROUP) as usize;
        (self.keys.group_shard[g] as usize, self.keys.group_local[g] + (page % GROUP) as u32)
    }
}

/// Compiles `accesses`; returns the residencies (for [`Output::Plan`]),
/// the statistics and the merged profile records.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile<P: Profiler>(graph: &Graph, records: &Records, accesses: &dyn Accesses, options: &PlanOptions, p: Params, output: Output<'_>, stop: &AtomicBool, make: &(dyn Fn() -> P + Sync)) -> Result<(Vec<Built>, OnlineStats, Option<Record>)> {
    let started = Instant::now();
    let len = accesses.len();
    let schemas = graph.schemas.len() as u64;
    let page = graph.page_size;
    let frames = (options.cache_bytes / page) as u64;
    let rows = options.rows && options.cost.is_some() && options.objective == PlanObjective::Time && frames > 0;
    let policy = options.row_policy;
    if rows && policy == RowPolicy::Global && (p.horizon < len || accesses.as_slice().is_none()) {
        return Err(Error::Plan("the global row policy compiles stored sequences offline".into()));
    }
    let threads = p.threads.max(1);
    // A shard owns a share of the cache big enough for its frame pool and
    // occupancy tree to matter, at most two per thread so every thread finds
    // work, and never more than MAX_SHARDS.
    let shards = ((frames / MIN_SHARD_FRAMES) as usize).clamp(1, (2 * threads).min(MAX_SHARDS));
    // A chunk is generated and assembled in this many pieces, so one chunk's
    // accesses spread over threads; the cap bounds the per-chunk bookkeeping.
    let parts = threads.clamp(1, 32);
    let mut stats = OnlineStats { accesses: len, chunk: p.chunk, horizon: p.horizon, distance: p.d, shards, ..Default::default() };
    if len == 0 {
        if let Output::Ring { ring, .. } = &output {
            ring.publish(&EntryBuf::default(), &|| 0, || {});
        }
        return Ok((Vec::new(), stats, None));
    }
    // Keys: groups of pages hashed to shards; a cell's row goes with the
    // group of its first page.
    let ngroups = graph.pages.div_ceil(GROUP) as usize;
    let mut group_shard = vec![0u8; ngroups];
    let mut group_local = vec![0u32; ngroups];
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); shards];
    for g in 0..ngroups {
        let s = (mix(g as u64) % shards as u64) as usize;
        group_shard[g] = s as u8;
        group_local[g] = (groups[s].len() as u64 * GROUP) as u32;
        groups[s].push(g as u32);
    }
    if groups.iter().any(|g| g.len() as u64 * GROUP >= u32::MAX as u64) {
        return Err(Error::Plan("a shard holds 2^32 pages or more".into()));
    }
    let (mut cell_shard, mut cell_local) = (Vec::new(), Vec::new());
    let mut cells: Vec<Vec<CellId>> = vec![Vec::new(); shards];
    if rows {
        cell_shard = vec![0u8; records.cells as usize];
        cell_local = vec![0u32; records.cells as usize];
        for c in 0..records.cells as usize {
            let s = group_shard[(records.of(c as CellId).first_page as u64 / GROUP) as usize] as usize;
            cell_shard[c] = s as u8;
            cell_local[c] = cells[s].len() as u32;
            cells[s].push(c as CellId);
        }
    }
    let chunks = len.div_ceil(p.chunk);
    // The ring of chunks spans the window from the published frontier to
    // the compiler's, with room to run ahead.
    let ring = ((p.horizon + p.d).div_ceil(p.chunk) + 4).min(chunks);
    let ctx = Ctx {
        graph,
        records,
        accesses,
        pricing: Pricing::new(graph, records, options.cost.as_ref(), p.online),
        p: p.clone(),
        len,
        schemas,
        rows,
        block_pages: if options.cost.is_some() { options.block_pages } else { None },
        parts,
        keys: Keys { shards, group_shard, group_local, cell_shard, cell_local },
        ring,
        global: Vec::new(),
    };
    // The occupancy window has to reach from the oldest frame a residency
    // still reserves (`frozen − D`) to the compiler's frontier: the ring of
    // chunks, the gap, and two chunks of slack.
    let window = ring * p.chunk + p.d + 2 * p.chunk;
    let mut shard_states: Vec<Mutex<Shard>> = Vec::with_capacity(shards);
    let mut base = 0u64;
    for (j, (g, c)) in groups.into_iter().zip(cells).enumerate() {
        let cap = frames / shards as u64 + u64::from((j as u64) < frames % shards as u64);
        let keys = g.len() * GROUP as usize;
        let rkeys = c.len();
        shard_states.push(Mutex::new(Shard {
            j,
            cap: cap as u32,
            frame_base: base as u32,
            occ: Occupancy::new(window),
            page_last: vec![NONE; keys],
            page_entry: vec![0; keys],
            page_res: vec![NIL; keys],
            row_last: vec![NONE; rkeys],
            row_entry: vec![0; rkeys],
            row_res: vec![NIL; rkeys],
            res: Vec::new(),
            free: Vec::new(),
            row_lists: Vec::new(),
            row_free: Vec::new(),
            page_uses: vec![Vec::new(); ring as usize],
            page_dec: vec![Vec::new(); ring as usize],
            row_uses: vec![Vec::new(); ring as usize],
            row_dec: vec![Vec::new(); ring as usize],
            by_start: vec![Vec::new(); ring as usize],
            by_end: vec![Vec::new(); ring as usize],
            heap: (0..cap as u32).map(|f| Reverse((0u64, f))).collect(),
            gens: vec![0; cap as usize],
            z: 0,
            entered: 0,
            lambda: Lambda { refused: [0.0; 64], n: 0.0, events: 0.0 },
            prices: Prices::default(),
            page_value: 1.0,
            cold: 0.0,
            hit: 0.0,
            priced: 0,
            seen: Seen::default(),
            out: Vec::new(),
            groups: g,
            cells: c,
            stats: ShardStats::default(),
        }));
        base += cap;
    }
    let mut ctx = ctx;
    if rows && policy == RowPolicy::Global {
        // Rows first, over the whole stored sequence.
        let slice = accesses.as_slice().unwrap();
        ctx.global = (0..len).map(|_| AtomicU8::new(0)).collect();
        global_rows(&ctx, slice, &mut shard_states)?;
    }
    let ctx = ctx;

    let chunk_slots: Vec<Mutex<Option<Arc<ChunkData>>>> = (0..ring).map(|_| Mutex::new(None)).collect();
    let sched = Mutex::new(Sched::new(chunks, shards, ring, parts, rows));
    let failure: Mutex<Option<Error>> = Mutex::new(None);
    let failed = AtomicBool::new(false);
    let staging: Vec<Mutex<Vec<(EntryBuf, Tally)>>> = (0..2).map(|_| Mutex::new(Vec::new())).collect();
    let tally = Mutex::new(Tally::default());
    let waiting = Mutex::new(Duration::ZERO);
    let records_out: Mutex<Option<Record>> = Mutex::new(None);
    let want_entries = matches!(output, Output::Ring { .. });
    // What the consumer can take: published entries up to its frontier
    // plus the plan ring's room.
    let demand = || match &output {
        Output::Ring { ring, frontier } if p.on_demand => Some(frontier().saturating_add(ring.capacity())),
        _ => None,
    };
    let want_plan = matches!(output, Output::Plan);
    let published = AtomicU64::new(0);

    let work = |_: usize| {
        let mut prof = make();
        let mut idle = 0u32;
        loop {
            if failed.load(Acquire) || stop.load(Acquire) || matches!(&output, Output::Ring { ring, .. } if ring.stopped()) {
                break;
            }
            let task = {
                let mut s = sched.lock().unwrap();
                if s.finished() {
                    break;
                }
                s.take(&ctx, published.load(Acquire), &demand)
            };
            let Some(task) = task else {
                let t = prof.start(Site::PlanIdle);
                idle += 1;
                if idle < 32 {
                    std::hint::spin_loop();
                } else if idle < 64 {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(Duration::from_micros(30));
                }
                prof.stop(Site::PlanIdle, t);
                continue;
            };
            idle = 0;
            let result = (|| -> Result<()> {
                match task {
                    Task::Gen(k, i) => {
                        let t = prof.start(Site::PlanAccess);
                        let chunk = {
                            let mut slot = chunk_slots[(k % ring) as usize].lock().unwrap();
                            if slot.as_ref().is_none_or(|c| c.first != k * ctx.p.chunk) {
                                let first = k * ctx.p.chunk;
                                let n = (ctx.len - first).min(ctx.p.chunk) as usize;
                                *slot = Some(Arc::new(ChunkData::new(first, n, ctx.parts, shards, ctx.rows)));
                            }
                            slot.as_ref().unwrap().clone()
                        };
                        generate(&ctx, &chunk, i)?;
                        prof.stop(Site::PlanAccess, t);
                    }
                    Task::Rows(k, j) => {
                        let t = prof.start(Site::PlanRow);
                        let chunk = chunk_slots[(k % ring) as usize].lock().unwrap().clone().unwrap();
                        let mut sh = shard_states[j].lock().unwrap();
                        rows_step(&ctx, &mut sh, &chunk, k)?;
                        prof.stop(Site::PlanRow, t);
                    }
                    Task::Step(k, j, z1) => {
                        let chunk = chunk_slots[(k % ring) as usize].lock().unwrap().clone().unwrap();
                        let mut sh = shard_states[j].lock().unwrap();
                        let t = prof.start(Site::PlanPage);
                        pages_step(&ctx, &mut sh, &chunk, k)?;
                        prof.stop(Site::PlanPage, t);
                        let t = prof.start(Site::PlanFreeze);
                        freeze(&ctx, &mut sh, &chunk_slots, z1, want_plan, &mut prof)?;
                        prof.stop(Site::PlanFreeze, t);
                    }
                    // A plan's statistics come from its replay: nothing to expand.
                    Task::Assemble(..) if want_plan => {}
                    Task::Assemble(k, i, p0, p1) => {
                        let t = prof.start(Site::PlanAssemble);
                        let span = (p1 - p0).div_ceil(ctx.parts as u64).max(1);
                        let (lo, hi) = ((p0 + span * i as u64).min(p1), (p0 + span * (i as u64 + 1)).min(p1));
                        let mut buf = EntryBuf::default();
                        buf.clear(lo);
                        let t2 = assemble(&ctx, &chunk_slots, lo, hi, want_entries.then_some(&mut buf))?;
                        staging[(k % 2) as usize].lock().unwrap().push((buf, t2));
                        prof.stop(Site::PlanAssemble, t);
                    }
                    Task::Publish(k, p0, p1) => {
                        let mut parts = std::mem::take(&mut *staging[(k % 2) as usize].lock().unwrap());
                        parts.sort_by_key(|(b, _)| b.first);
                        let mut all = EntryBuf::default();
                        all.clear(p0);
                        let mut sum = Tally::default();
                        for (b, t) in &parts {
                            if want_entries {
                                all.append(b);
                            }
                            sum.add(t);
                        }
                        tally.lock().unwrap().add(&sum);
                        if let Output::Ring { ring, frontier } = &output {
                            let t = prof.start(Site::PlanPublish);
                            let w = Instant::now();
                            let mut waited = false;
                            if !ring.publish(&all, *frontier, || {
                                waited = true;
                                std::thread::sleep(Duration::from_micros(20));
                            }) {
                                return Ok(());
                            }
                            if waited {
                                *waiting.lock().unwrap() += w.elapsed();
                                prof.add_ticks(Site::PlanSpace, prof.since(t));
                            }
                            prof.stop(Site::PlanPublish, t);
                        }
                        published.store(p1, Release);
                    }
                }
                Ok(())
            })();
            let mut s = sched.lock().unwrap();
            match result {
                Ok(()) => s.done(task),
                Err(e) => {
                    failure.lock().unwrap().get_or_insert(e);
                    failed.store(true, Release);
                }
            }
        }
        if let Some(r) = prof.record() {
            let mut m = records_out.lock().unwrap();
            match m.as_mut() {
                Some(a) => a.add(r),
                None => *m = Some(r.clone()),
            }
        }
    };
    crate::opt::compile::par(threads, work);
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }
    let mut built = Vec::new();
    let mut lambda = 0.0;
    let mut prices = Prices::default();
    for s in shard_states {
        let s = s.into_inner().unwrap();
        stats.row_candidates += s.stats.row_candidates;
        stats.row_kept += s.stats.row_kept;
        stats.page_refused += s.stats.page_refused;
        lambda += s.lambda.price(s.page_value) / shards as f64;
        let w = 1.0 / shards as f64;
        prices.request += s.prices.request * w;
        prices.byte += s.prices.byte * w;
        prices.core += s.prices.core * w;
        prices.memory += s.prices.memory * w;
        if want_plan {
            built.extend(s.out);
        }
    }
    let t = tally.into_inner().unwrap();
    stats.chunks = chunks;
    stats.page_uses = t.page_uses;
    stats.page_hits = t.page_hits;
    stats.page_admissions = t.page_admissions;
    stats.row_admissions = t.row_admissions;
    stats.row_hits = t.row_hits;
    stats.block_reads = t.block_reads;
    stats.requests = t.requests;
    stats.modeled_seconds = t.seconds;
    stats.worker_seconds = t.worker_seconds;
    stats.row_price = lambda;
    stats.prices = prices;
    stats.waiting_seconds = waiting.into_inner().unwrap().as_secs_f64();
    stats.compile_seconds = started.elapsed().as_secs_f64() - stats.waiting_seconds;
    Ok((built, stats, records_out.into_inner().unwrap()))
}

/// The tasks of one chunk round.
#[derive(Clone, Copy, Debug)]
enum Task {
    /// Chunk and part: fill the accesses and scatter their uses.
    Gen(u64, usize),
    /// Chunk and shard: choose the chunk's rows.
    Rows(u64, usize),
    /// Chunk, shard and the round's frozen frontier.
    Step(u64, usize, u64),
    /// Chunk, part and the round's publication range.
    Assemble(u64, usize, u64, u64),
    /// Chunk and its publication range `[p0, p1)`.
    Publish(u64, u64, u64),
}

/// Which tasks are issued and done. Each stage advances chunk by chunk;
/// per-chunk counts live on a ring.
struct Sched {
    chunks: u64,
    shards: usize,
    ring: u64,
    parts: usize,
    rows: bool,
    gen_next: u64,
    gen_part: usize,
    gen_done: Vec<usize>,
    gen_complete: u64,
    rows_next: Vec<u64>,
    rows_busy: Vec<bool>,
    rows_done: Vec<usize>,
    rows_complete: u64,
    step_next: Vec<u64>,
    step_busy: Vec<bool>,
    step_done: Vec<usize>,
    step_complete: u64,
    asm_next: u64,
    asm_part: usize,
    asm_done: Vec<usize>,
    asm_complete: u64,
    pub_busy: bool,
    pub_next: u64,
    /// Per round on the ring: its frozen frontier, fixed when its first
    /// shard step is issued; and the last one fixed.
    zs: Vec<u64>,
    z_last: u64,
    z_rounds: u64,
}

impl Sched {
    fn new(chunks: u64, shards: usize, ring: u64, parts: usize, rows: bool) -> Self {
        let r = ring as usize;
        Self {
            chunks,
            shards,
            ring,
            parts,
            rows,
            gen_next: 0,
            gen_part: 0,
            gen_done: vec![0; r],
            gen_complete: 0,
            rows_next: vec![0; shards],
            rows_busy: vec![false; shards],
            rows_done: vec![0; r],
            rows_complete: 0,
            step_next: vec![0; shards],
            step_busy: vec![false; shards],
            step_done: vec![0; r],
            step_complete: 0,
            asm_next: 0,
            asm_part: 0,
            asm_done: vec![0; r],
            asm_complete: 0,
            pub_busy: false,
            pub_next: 0,
            zs: vec![0; r],
            z_last: 0,
            z_rounds: 0,
        }
    }

    fn finished(&self) -> bool {
        self.pub_next >= self.chunks
    }

    /// Round `k`'s publication range.
    fn range(&self, ctx: &Ctx<'_>, k: u64) -> (u64, u64) {
        let z = |k: u64| self.zs[(k % self.ring) as usize];
        let p0 = if k == 0 { 0 } else { ctx.published(z(k - 1)) };
        (p0, ctx.published(z(k)))
    }

    fn take(&mut self, ctx: &Ctx<'_>, published: u64, demand: &dyn Fn() -> Option<u64>) -> Option<Task> {
        // Publication, in order.
        if !self.pub_busy && self.pub_next < self.asm_complete {
            self.pub_busy = true;
            let (p0, p1) = self.range(ctx, self.pub_next);
            return Some(Task::Publish(self.pub_next, p0, p1));
        }
        // Assembly of a round whose freeze is complete; at most one round
        // ahead of publication.
        if self.asm_next < self.step_complete && self.asm_next <= self.pub_next + 1 {
            let k = self.asm_next;
            let i = self.asm_part;
            self.asm_part += 1;
            if self.asm_part == self.parts {
                self.asm_part = 0;
                self.asm_next += 1;
            }
            let (p0, p1) = self.range(ctx, k);
            return Some(Task::Assemble(k, i, p0, p1));
        }
        // Shard steps: pages and freezing, in chunk order per shard.
        for j in 0..self.shards {
            let k = self.step_next[j];
            if !self.step_busy[j] && k < self.gen_complete && (!self.rows || k < self.rows_complete) && !self.rows_busy[j] {
                if k == self.z_rounds {
                    let x = ((k + 1) * ctx.p.chunk).min(ctx.len);
                    self.z_last = ctx.target(x, self.z_last, demand());
                    self.zs[(k % self.ring) as usize] = self.z_last;
                    self.z_rounds += 1;
                }
                self.step_busy[j] = true;
                return Some(Task::Step(k, j, self.zs[(k % self.ring) as usize]));
            }
        }
        if self.rows {
            for j in 0..self.shards {
                let k = self.rows_next[j];
                if !self.rows_busy[j] && !self.step_busy[j] && k < self.gen_complete && k == self.step_next[j] {
                    self.rows_busy[j] = true;
                    return Some(Task::Rows(k, j));
                }
            }
        }
        // Accesses of the next chunk, once its ring slot is free: every
        // access of the chunk it replaces is published.
        let k = self.gen_next;
        if k < self.chunks && (k < self.ring || published >= (k + 1 - self.ring) * ctx.p.chunk) {
            let i = self.gen_part;
            self.gen_part += 1;
            if self.gen_part == self.parts {
                self.gen_part = 0;
                self.gen_next += 1;
            }
            return Some(Task::Gen(k, i));
        }
        None
    }

    fn done(&mut self, task: Task) {
        let ring = self.ring;
        let r = move |k: u64| (k % ring) as usize;
        match task {
            Task::Gen(k, _) => {
                self.gen_done[r(k)] += 1;
                while self.gen_complete < self.chunks && self.gen_done[r(self.gen_complete)] == self.parts {
                    self.gen_done[r(self.gen_complete)] = 0;
                    self.gen_complete += 1;
                }
            }
            Task::Rows(k, j) => {
                self.rows_busy[j] = false;
                self.rows_next[j] = k + 1;
                self.rows_done[r(k)] += 1;
                while self.rows_complete < self.gen_complete && self.rows_done[r(self.rows_complete)] == self.shards {
                    self.rows_done[r(self.rows_complete)] = 0;
                    self.rows_complete += 1;
                }
            }
            Task::Step(k, j, _) => {
                self.step_busy[j] = false;
                self.step_next[j] = k + 1;
                self.step_done[r(k)] += 1;
                while self.step_complete < self.gen_complete && self.step_done[r(self.step_complete)] == self.shards {
                    self.step_done[r(self.step_complete)] = 0;
                    self.step_complete += 1;
                }
            }
            Task::Assemble(k, ..) => {
                self.asm_done[r(k)] += 1;
                while self.asm_complete < self.step_complete && self.asm_done[r(self.asm_complete)] == self.parts {
                    self.asm_done[r(self.asm_complete)] = 0;
                    self.asm_complete += 1;
                }
            }
            Task::Publish(k, ..) => {
                self.pub_busy = false;
                self.pub_next = k + 1;
            }
        }
    }
}

/// Fills accesses of part `i` of a chunk and scatters their uses.
fn generate(ctx: &Ctx<'_>, chunk: &ChunkData, i: usize) -> Result<()> {
    let span = chunk.n.div_ceil(ctx.parts);
    let (lo, hi) = ((i * span).min(chunk.n), ((i + 1) * span).min(chunk.n));
    let shards = ctx.keys.shards;
    let mut pages: Vec<Vec<(u32, u32)>> = vec![Vec::new(); shards];
    let mut rows: Vec<Vec<(u32, u32)>> = vec![Vec::new(); if ctx.rows { shards } else { 0 }];
    let mut slots = 0u32;
    let mut base = Base::default();
    // Chunks recently decoded within this part (the part's own history, so
    // the bits do not depend on scheduling).
    let mut recent: Vec<u32> = Vec::with_capacity(WARM_CHUNKS);
    if lo < hi {
        // One access before the part, for the dedup of a position's schemas.
        let from = chunk.first + lo as u64;
        let back = u64::from(from > 0 && !from.is_multiple_of(ctx.schemas));
        let mut buf = vec![0 as CellId; hi - lo + back as usize];
        ctx.accesses.fill(from - back, &mut buf)?;
        for (x, &c) in buf.iter().enumerate().skip(back as usize) {
            let a = from + (x - back as usize) as u64;
            if c as u64 >= ctx.records.cells {
                return Err(Error::Access { position: a, cell: c as u64 });
            }
            let off = (a - chunk.first) as usize;
            chunk.cells[off].store(c, Relaxed);
            chunk.slot[off].store(slots, Relaxed);
            let same = !a.is_multiple_of(ctx.schemas) && x > 0 && buf[x - 1] == c;
            let r = ctx.records.of(c);
            let k = (a % ctx.schemas) as usize;
            base.accesses += 1;
            if same {
                chunk.flags[off].store(SAME, Relaxed);
                base.loads.add(&ctx.pricing.uncached(k, r, false, 0.0));
                continue;
            }
            chunk.flags[off].store(0, Relaxed);
            let cold = touch(&mut recent, r.chunk);
            if r.pages > 0 {
                base.reads += 1;
                base.pages += r.pages as u64;
            }
            if !r.empty() {
                base.decoded += 1;
                base.cold += cold as u64;
            }
            base.loads.add(&ctx.pricing.uncached(k, r, true, cold as u8 as f64));
            slots += r.pages as u32;
            for p in r.page_range() {
                let (s, local) = ctx.page_key(p);
                pages[s].push((off as u32, local));
            }
            if ctx.rows && !r.empty() {
                let s = ctx.keys.cell_shard[c as usize] as usize;
                rows[s].push((off as u32, ctx.keys.cell_local[c as usize]));
            }
        }
    }
    chunk.part_slots[i].store(slots, Relaxed);
    *chunk.part_base[i].lock().unwrap() = base;
    for (s, v) in pages.into_iter().enumerate() {
        *chunk.page_uses[i * shards + s].lock().unwrap() = v;
    }
    for (s, v) in rows.into_iter().enumerate() {
        *chunk.row_uses[i * shards + s].lock().unwrap() = v;
    }
    // The last part makes the slots global within the chunk.
    if chunk.gen_left.fetch_sub(1, AcqRel) == 1 {
        let mut base = vec![0u32; ctx.parts + 1];
        for (p, slots) in chunk.part_slots.iter().enumerate() {
            base[p + 1] = base[p] + slots.load(Relaxed);
        }
        for (p, &b) in base[..ctx.parts].iter().enumerate() {
            let (lo, hi) = ((p * span).min(chunk.n), ((p + 1) * span).min(chunk.n));
            for off in lo..hi {
                chunk.slot[off].fetch_add(b, Relaxed);
            }
        }
        chunk.slot[chunk.n].store(base[ctx.parts], Relaxed);
        let _ = chunk.refs.set((0..base[ctx.parts]).map(|_| AtomicU64::new(MISS)).collect());
        let mut sum = Base::default();
        for b in chunk.part_base.iter() {
            sum.add(&b.lock().unwrap());
        }
        let _ = chunk.base.set(sum);
    }
    Ok(())
}

/// Marks `chunk` most recently used among `recent`; whether it was cold
/// (not among the [`WARM_CHUNKS`] most recently used).
fn touch(recent: &mut Vec<u32>, chunk: u32) -> bool {
    match recent.iter().position(|&c| c == chunk) {
        Some(k) => {
            recent[..=k].rotate_right(1);
            false
        }
        None => {
            if recent.len() == WARM_CHUNKS {
                recent.pop();
            }
            recent.insert(0, chunk);
            true
        }
    }
}

/// Prices chunk `k` for shard `sh` (once, before its first decision): the
/// chunk's uncached loads, less the reads this shard's recent page hits
/// remove, plus the loads its recent rows add, at the cost model.
fn reprice(ctx: &Ctx<'_>, sh: &mut Shard, chunk: &ChunkData, k: u64) {
    if sh.priced > k {
        return;
    }
    sh.priced = k + 1;
    let base = chunk.base.get().copied().unwrap_or_default();
    sh.cold = base.cold_share();
    let s = &mut sh.seen;
    let h = if s.uses > 0.0 { s.hits / s.uses } else { 0.0 };
    sh.hit = h;
    if let Some(cost) = ctx.pricing.cost {
        let mut l = base.loads;
        let reads = base.reads as f64 * h;
        l.requests -= reads;
        l.bytes -= base.pages as f64 * h * ctx.graph.page_size as f64;
        l.core_seconds -= reads * cost.request;
        if s.accesses > 0.0 {
            l.add(&s.rows.scaled(ctx.keys.shards as f64 * base.accesses as f64 / s.accesses));
        }
        let l = Loads { requests: l.requests.max(0.0), bytes: l.bytes.max(0.0), core_seconds: l.core_seconds.max(0.0), memory_bytes: l.memory_bytes.max(0.0) };
        sh.prices = cost.prices(&l);
        sh.page_value = sh.prices.value(&ctx.pricing.page_hit(base.rho()));
    }
    s.uses *= 0.5;
    s.hits *= 0.5;
    s.rows = s.rows.scaled(0.5);
    s.accesses = s.accesses * 0.5 + base.accesses as f64;
}

/// Enters the occupancy blocks up to `x` (nothing to do when the global row
/// order entered them all).
fn enter(ctx: &Ctx<'_>, sh: &mut Shard, x: u64) {
    if !ctx.global.is_empty() {
        return;
    }
    let shift = sh.occ.shift();
    let block = 1u64 << shift;
    while sh.entered < x {
        sh.occ.enter(sh.entered, 0);
        sh.entered = (sh.entered / block + 1) * block;
    }
}

/// Stores `r` under a residency id: a released id when there is one, else a
/// new one. An id is valid until its residency is released, and ids are what
/// the per-use decision tables hold.
fn new_res(sh: &mut Shard, r: Res) -> u32 {
    match sh.free.pop() {
        Some(id) => {
            sh.res[id as usize] = r;
            id
        }
        None => {
            sh.res.push(r);
            (sh.res.len() - 1) as u32
        }
    }
}

/// The ring slot of the chunk holding access `a`.
fn bucket(ctx: &Ctx<'_>, a: u64) -> usize {
    ((a / ctx.p.chunk) % ctx.ring) as usize
}

/// Gathers shard `j`'s uses from every part, in part order, so the order does
/// not depend on how the parts were scheduled.
fn take_uses(ctx: &Ctx<'_>, uses: &[Mutex<Vec<(u32, u32)>>], j: usize) -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    for p in 0..ctx.parts {
        v.append(&mut uses[p * ctx.keys.shards + j].lock().unwrap());
    }
    v
}

/// Row decisions of shard `sh` for chunk `k`.
fn rows_step(ctx: &Ctx<'_>, sh: &mut Shard, chunk: &ChunkData, k: u64) -> Result<()> {
    let b = (k % ctx.ring) as usize;
    let uses = take_uses(ctx, &chunk.row_uses, sh.j);
    let x = chunk.first + chunk.n as u64;
    enter(ctx, sh, x);
    reprice(ctx, sh, chunk, k);
    let mut dec = vec![NIL; uses.len()];
    let price = sh.lambda.price(sh.page_value);
    let d = ctx.p.d;
    for (e, &(off, c)) in uses.iter().enumerate() {
        let a = chunk.first + off as u64;
        let u = sh.row_last[c as usize];
        let ue = sh.row_entry[c as usize];
        sh.row_last[c as usize] = a;
        sh.row_entry[c as usize] = e as u32;
        if u == NONE || u < sh.z {
            sh.row_res[c as usize] = NIL;
            continue;
        }
        let cell = sh.cells[c as usize];
        let rec = ctx.records.of(cell);
        let r = sh.row_res[c as usize];
        let ext = r != NIL && sh.res[r as usize].live && sh.res[r as usize].end == u && sh.res[r as usize].row;
        sh.stats.row_candidates += 1;
        let w = rec.row_frames;
        let lo = if ext { u + 1 } else { u.saturating_sub(d) };
        let keep = if ctx.global.is_empty() {
            let value = ctx.pricing.row_value(&sh.prices, u, a, cell, ext, sh.cold, sh.hit);
            let density = value / (w as f64 * (a - lo + 1) as f64);
            value > 0.0 && density >= price && w <= sh.cap && sh.occ.fits(lo, a, w, sh.cap)
        } else {
            ctx.global[a as usize].load(Relaxed) != 0
        };
        if !keep {
            sh.row_res[c as usize] = NIL;
            continue;
        }
        if ctx.global.is_empty() {
            sh.occ.add(lo, a, w);
        }
        let id = if ext {
            sh.res[r as usize].end = a;
            r
        } else {
            let id = new_res(sh, Res { start: u, end: a, key: c, frame: NIL, generation: 0, row: true, live: true, moved: false, rel: false });
            let bs = bucket(ctx, u);
            sh.by_start[bs].push(id);
            // The admission's entry is in the chunk of `u`.
            if (u / ctx.p.chunk) == k {
                dec[ue as usize] = id;
            } else {
                sh.row_dec[bs][ue as usize] = id;
            }
            id
        };
        sh.by_end[b].push(id);
        dec[e] = id;
        sh.row_res[c as usize] = id;
        sh.stats.row_kept += 1;
        let schema = |x: u64| (x % ctx.schemas) as usize;
        let mut added = ctx.pricing.row_saving(schema(a), rec, sh.cold, sh.hit).scaled(-1.0);
        if !ext {
            added.add(&ctx.pricing.row_admission(schema(u), rec));
        }
        sh.seen.rows.add(&added);
        chunk.flags[off as usize].fetch_or(HIT, Relaxed);
    }
    sh.row_uses[b] = uses;
    sh.row_dec[b] = dec;
    Ok(())
}

/// Page decisions of shard `sh` for chunk `k`.
fn pages_step(ctx: &Ctx<'_>, sh: &mut Shard, chunk: &ChunkData, k: u64) -> Result<()> {
    let b = (k % ctx.ring) as usize;
    let uses = take_uses(ctx, &chunk.page_uses, sh.j);
    let x = chunk.first + chunk.n as u64;
    enter(ctx, sh, x);
    reprice(ctx, sh, chunk, k);
    let mut dec = vec![NIL; uses.len()];
    let d = ctx.p.d;
    for (e, &(off, p)) in uses.iter().enumerate() {
        if chunk.flags[off as usize].load(Relaxed) & HIT != 0 {
            continue;
        }
        sh.seen.uses += 1.0;
        let a = chunk.first + off as u64;
        let pi = p as usize;
        let u = sh.page_last[pi];
        let ue = sh.page_entry[pi];
        sh.page_last[pi] = a;
        sh.page_entry[pi] = e as u32;
        if u == NONE || u < sh.z {
            sh.page_res[pi] = NIL;
            continue;
        }
        let r = sh.page_res[pi];
        let ext = r != NIL && sh.res[r as usize].live && !sh.res[r as usize].row && sh.res[r as usize].end == u;
        let lo = if ext { u + 1 } else { u.saturating_sub(d) };
        if !sh.occ.fits(lo, a, 1, sh.cap) {
            sh.page_res[pi] = NIL;
            sh.stats.page_refused += 1;
            sh.lambda.refuse(a - lo + 1);
            continue;
        }
        sh.occ.add(lo, a, 1);
        sh.lambda.record();
        sh.seen.hits += 1.0;
        let id = if ext {
            sh.res[r as usize].end = a;
            r
        } else {
            let id = new_res(sh, Res { start: u, end: a, key: p, frame: NIL, generation: 0, row: false, live: true, moved: false, rel: false });
            let bs = bucket(ctx, u);
            sh.by_start[bs].push(id);
            if (u / ctx.p.chunk) == k {
                dec[ue as usize] = id;
            } else {
                sh.page_dec[bs][ue as usize] = id;
            }
            id
        };
        sh.by_end[b].push(id);
        dec[e] = id;
        sh.page_res[pi] = id;
    }
    sh.page_uses[b] = uses;
    sh.page_dec[b] = dec;
    Ok(())
}

impl Shard {
    /// The global page id of this shard's local page key: groups keep their
    /// pages adjacent in the key space, so a group's offset is a multiple of
    /// [`GROUP`].
    fn global_page(&self, local: u32) -> u64 {
        self.groups[(local as u64 / GROUP) as usize] as u64 * GROUP + local as u64 % GROUP
    }
    /// A residency's frames: its one frame for a page, its list for a row.
    fn frames_of<'s>(&'s self, r: &'s Res) -> &'s [u32] {
        if r.row { &self.row_lists[r.frame as usize] } else { std::slice::from_ref(&r.frame) }
    }
}

/// Freezes shard `sh` up to `z1`: releases, block merges, frame
/// allocation, and resolution of every use in `[z0, z1)`.
fn freeze<P: Profiler>(ctx: &Ctx<'_>, sh: &mut Shard, slots: &[Mutex<Option<Arc<ChunkData>>>], z1: u64, collect: bool, prof: &mut P) -> Result<()> {
    let z0 = sh.z;
    if z1 <= z0 {
        return Ok(());
    }
    let c = ctx.p.chunk;
    let d = ctx.p.d;
    let chunk_of = |a: u64| a / c;
    let (k0, k1) = (chunk_of(z0), chunk_of(z1 - 1));
    // Releases: residencies whose end lies in [z0, z1) are final.
    let mut released = Vec::new();
    for k in k0..=k1 {
        let b = (k % ctx.ring) as usize;
        let ids = std::mem::take(&mut sh.by_end[b]);
        let mut keep = Vec::new();
        for id in ids {
            let r = sh.res[id as usize];
            if !r.live || chunk_of(r.end) != k {
                continue;
            }
            if r.end < z0 {
                continue;
            }
            if r.end < z1 {
                if !r.rel {
                    sh.res[id as usize].rel = true;
                    released.push(id);
                }
            } else {
                keep.push(id);
            }
        }
        sh.by_end[b] = keep;
    }
    // Frames of final residencies return to the pool, keyed by when they
    // are free; those admitted in this range return once allocated below.
    for &id in &released {
        let r = sh.res[id as usize];
        if r.start < z0 {
            release_frames(sh, id);
        }
    }
    // Admissions in [z0, z1), in order.
    let mut starts = Vec::new();
    for k in k0..=k1 {
        let b = (k % ctx.ring) as usize;
        let ids = std::mem::take(&mut sh.by_start[b]);
        let mut keep = Vec::new();
        for id in ids {
            let r = sh.res[id as usize];
            if !r.live || chunk_of(r.start) != k {
                continue;
            }
            if r.start < z1 && r.start >= z0 {
                starts.push(id);
            } else if r.start >= z1 {
                keep.push(id);
            }
        }
        sh.by_start[b] = keep;
    }
    starts.sort_unstable_by_key(|&id| (sh.res[id as usize].start, id));
    if let Some(max) = ctx.block_pages {
        let t = prof.start(Site::PlanBlock);
        merge_blocks(ctx, sh, slots, &mut starts, max, z0);
        starts.sort_unstable_by_key(|&id| (sh.res[id as usize].start, id));
        prof.stop(Site::PlanBlock, t);
    }
    for &id in &starts {
        let r = sh.res[id as usize];
        let w = if r.row { ctx.records.of(sh.cells[r.key as usize]).row_frames as usize } else { 1 };
        let limit = r.start.saturating_sub(d);
        let mut fs = Vec::with_capacity(w);
        let mut g = 0u32;
        for _ in 0..w {
            match sh.heap.peek() {
                Some(&Reverse((key, f))) if key <= limit => {
                    sh.heap.pop();
                    g = g.max(sh.gens[f as usize]);
                    fs.push(f);
                }
                _ => return Err(Error::Plan(format!("no free frame for an admission at {} (shard {}): the occupancy check and the frame pool disagree", r.start, sh.j))),
            }
        }
        // One past the highest generation its frames carried, so a use that
        // waits on an earlier residency of those frames cannot accept this
        // one; the mask bounds it to a ref's 31 bits.
        let g = (g + 1) & crate::loader::planring::GEN_MASK;
        for &f in &fs {
            sh.gens[f as usize] = g;
        }
        let res = &mut sh.res[id as usize];
        res.generation = g;
        if r.row {
            let list: Vec<u32> = fs.iter().map(|&f| f + sh.frame_base).collect();
            let at = match sh.row_free.pop() {
                Some(at) => {
                    sh.row_lists[at as usize] = list;
                    at
                }
                None => {
                    sh.row_lists.push(list);
                    (sh.row_lists.len() - 1) as u32
                }
            };
            sh.res[id as usize].frame = at;
        } else {
            sh.res[id as usize].frame = fs[0] + sh.frame_base;
        }
        if sh.res[id as usize].rel {
            release_frames(sh, id);
        }
    }
    // Resolution of every use in [z0, z1).
    for k in k0..=k1 {
        let b = (k % ctx.ring) as usize;
        let Some(chunk) = slots[b].lock().unwrap().clone().filter(|ch| ch.first == k * c) else {
            continue;
        };
        let refs = chunk.refs.get().expect("chunk slots are set when its accesses are generated");
        let lo = z0.max(chunk.first);
        let hi = z1.min(chunk.first + chunk.n as u64);
        let uses = &sh.page_uses[b];
        let dec = &sh.page_dec[b];
        let from = uses.partition_point(|u| chunk.first + (u.0 as u64) < lo);
        let mut extras = Vec::new();
        for (e, &(off, p)) in uses.iter().enumerate().skip(from) {
            let a = chunk.first + off as u64;
            if a >= hi {
                break;
            }
            let id = dec[e];
            if id == NIL {
                continue;
            }
            let r = sh.res[id as usize];
            let cell = chunk.cell(off as usize);
            let rec = ctx.records.of(cell);
            let global = sh.global_page(p);
            let slot = chunk.slot[off as usize].load(Relaxed) + (global - rec.first_page as u64) as u32;
            refs[slot as usize].store(page_ref(r.frame, r.generation, r.start == a && !r.moved), Relaxed);
        }
        // Block-moved admissions read by their block's access.
        for &id in &starts {
            let r = sh.res[id as usize];
            if r.moved && r.start >= lo && r.start < hi {
                extras.push(((r.start - chunk.first) as u32, sh.global_page(r.key), page_ref(r.frame, r.generation, true)));
            }
        }
        if !extras.is_empty() {
            chunk.extras[sh.j].lock().unwrap().extend(extras);
        }
        if ctx.rows {
            let uses = &sh.row_uses[b];
            let dec = &sh.row_dec[b];
            let from = uses.partition_point(|u| chunk.first + (u.0 as u64) < lo);
            let mut list = chunk.row_frames[sh.j].lock().unwrap();
            for (e, &(off, _)) in uses.iter().enumerate().skip(from) {
                let a = chunk.first + off as u64;
                if a >= hi {
                    break;
                }
                let id = dec[e];
                if id == NIL {
                    continue;
                }
                let r = sh.res[id as usize];
                let admit = r.start == a;
                chunk.row_gen[off as usize].store(r.generation | if admit { 1 << 31 } else { 0 }, Relaxed);
                chunk.row_at[off as usize].store((sh.j as u64) << 32 | list.len() as u64, Relaxed);
                list.extend_from_slice(&sh.row_lists[r.frame as usize]);
            }
        }
    }
    // Released residencies leave the plan; their ids and lists are reused.
    for &id in &released {
        let r = sh.res[id as usize];
        if collect {
            let object = if r.row { Object::Row(sh.cells[r.key as usize]) } else { Object::Page(sh.global_page(r.key)) };
            sh.out.push(Built { object, start: r.start, end: r.end, frames: sh.frames_of(&r).to_vec() });
        }
        if r.row {
            sh.row_free.push(r.frame);
        }
        sh.res[id as usize].live = false;
        sh.free.push(id);
    }
    // Chunks wholly frozen drop this shard's uses.
    for k in k0..chunk_of(z1) {
        let b = (k % ctx.ring) as usize;
        sh.page_uses[b] = Vec::new();
        sh.page_dec[b] = Vec::new();
        sh.row_uses[b] = Vec::new();
        sh.row_dec[b] = Vec::new();
    }
    sh.z = z1;
    Ok(())
}

/// Returns a released residency's frames to the pool, available from the
/// access after its last use.
fn release_frames(sh: &mut Shard, id: u32) {
    let r = sh.res[id as usize];
    debug_assert!(r.frame != NIL, "released before its frames were allocated");
    let base = sh.frame_base;
    let n = if r.row { sh.row_lists[r.frame as usize].len() } else { 1 };
    for i in 0..n {
        let f = if r.row { sh.row_lists[r.frame as usize][i] } else { r.frame };
        sh.heap.push(Reverse((r.end + 1, f - base)));
    }
}

/// Moves admissions of adjacent pages to the earliest of them, read there
/// as one block, where that pays: each admitting access whose read the
/// block removes saves a request, each run of moved pages costs one, and
/// every moved page holds its frame longer, priced at the capacity's shadow
/// price. Pages the earliest access reads itself stay with its own read.
fn merge_blocks(ctx: &Ctx<'_>, sh: &mut Shard, slots: &[Mutex<Option<Arc<ChunkData>>>], starts: &mut [u32], max: u32, z0: u64) {
    let Some(cost) = ctx.pricing.cost else { return };
    let d = ctx.p.d;
    // Seconds of one request (the device's share and its worker work), and
    // of holding a frame one access longer.
    let request = sh.prices.request + sh.prices.core * cost.request;
    let lambda = sh.lambda.price(sh.page_value);
    let mut pages: Vec<u32> = starts.iter().copied().filter(|&id| !sh.res[id as usize].row).collect();
    pages.sort_unstable_by_key(|&id| sh.global_page(sh.res[id as usize].key));
    let page = ctx.graph.page_size as u64;
    let mut moved: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < pages.len() {
        let p0 = sh.global_page(sh.res[pages[i] as usize].key);
        let file = ctx.graph.files.partition_point(|f| f.first_page <= p0) - 1;
        let f = &ctx.graph.files[file];
        let file_end = f.first_page + f.len.div_ceil(page);
        let mut j = i + 1;
        while j < pages.len() && (j - i) < max as usize {
            let pj = sh.global_page(sh.res[pages[j] as usize].key);
            if pj != p0 + (j - i) as u64 || pj >= file_end {
                break;
            }
            j += 1;
        }
        let s0 = pages[i..j].iter().map(|&id| sh.res[id as usize].start).min().unwrap();
        if j - i < 2 || s0 < z0 {
            i = j;
            continue;
        }
        // The access at s0 reads its own record's pages itself.
        let own = {
            let b = ((s0 / ctx.p.chunk) % ctx.ring) as usize;
            slots[b].lock().unwrap().clone().filter(|ch| ch.first <= s0 && s0 < ch.first + ch.n as u64).map(|ch| ctx.records.of(ch.cell((s0 - ch.first) as usize)).page_range())
        };
        // Candidates: pages whose own read a block can replace, each worth
        // more than the frame time it adds.
        moved.clear();
        for &id in &pages[i..j] {
            let r = sh.res[id as usize];
            if r.start == s0 || r.start - s0 > d || own.as_ref().is_some_and(|o| o.contains(&sh.global_page(r.key))) {
                continue;
            }
            // Holding the frame from `s0` to the page's own admission costs
            // `lambda` per position; move only when the request saved costs
            // more.
            if lambda * ((r.start - s0) as f64) < request {
                moved.push(id);
            }
        }
        let mut admitters: Vec<u64> = moved.iter().map(|&id| sh.res[id as usize].start).collect();
        admitters.sort_unstable();
        admitters.dedup();
        let runs = 1 + moved.windows(2).filter(|w| sh.global_page(sh.res[w[1] as usize].key) != sh.global_page(sh.res[w[0] as usize].key) + 1).count();
        let held: u64 = moved.iter().map(|&id| sh.res[id as usize].start - s0).sum();
        if moved.is_empty() || request * (admitters.len() as f64 - runs as f64) <= lambda * held as f64 {
            i = j;
            continue;
        }
        for &id in &moved {
            let r = sh.res[id as usize];
            let (lo, hi) = (s0.saturating_sub(d), r.start.saturating_sub(d));
            if hi > lo && !sh.occ.fits(lo, hi - 1, 1, sh.cap) {
                continue;
            }
            if hi > lo {
                sh.occ.add(lo, hi - 1, 1);
            }
            let res = &mut sh.res[id as usize];
            res.start = s0;
            res.moved = true;
        }
        i = j;
    }
}

/// What the assembled accesses add up to.
#[derive(Clone, Copy, Debug, Default)]
struct Tally {
    page_uses: u64,
    page_hits: u64,
    page_admissions: u64,
    row_admissions: u64,
    row_hits: u64,
    block_reads: u64,
    requests: u64,
    seconds: f64,
    worker_seconds: f64,
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        self.page_uses += o.page_uses;
        self.page_hits += o.page_hits;
        self.page_admissions += o.page_admissions;
        self.row_admissions += o.row_admissions;
        self.row_hits += o.row_hits;
        self.block_reads += o.block_reads;
        self.requests += o.requests;
        self.seconds += o.seconds;
        self.worker_seconds += o.worker_seconds;
    }
}

/// Expands accesses `[lo, hi)` (frozen) into entries, and counts them.
fn assemble(ctx: &Ctx<'_>, slots: &[Mutex<Option<Arc<ChunkData>>>], lo: u64, hi: u64, mut buf: Option<&mut EntryBuf>) -> Result<Tally> {
    let mut t = Tally::default();
    let c = ctx.p.chunk;
    let page = ctx.graph.page_size as u64;
    let cost = ctx.pricing.cost;
    let mut loads = Loads::default();
    let mut a = lo;
    let mut refs: Vec<u64> = Vec::new();
    let mut frames: Vec<u32> = Vec::new();
    let mut extra_refs: Vec<u64> = Vec::new();
    let mut runs: Vec<(Extra, std::ops::Range<usize>)> = Vec::new();
    while a < hi {
        let k = a / c;
        let chunk = slots[(k % ctx.ring) as usize].lock().unwrap().clone().filter(|ch| ch.first == k * c).ok_or_else(|| Error::Plan("compiler chunk recycled before its publication".into()))?;
        let end = hi.min(chunk.first + chunk.n as u64);
        let all = chunk.refs.get().expect("chunk refs are set when its accesses are generated");
        let cold = chunk.base.get().map_or(0.0, |b| b.cold_share());
        // This chunk's extra admissions in range, by access.
        let mut extras: Vec<(u32, u64, u64)> = Vec::new();
        for e in chunk.extras.iter() {
            extras.extend(e.lock().unwrap().iter().copied().filter(|x| chunk.first + (x.0 as u64) >= a && chunk.first + (x.0 as u64) < end));
        }
        extras.sort_unstable_by_key(|x| (x.0, x.1));
        let mut xi = 0;
        for at in a..end {
            let off = (at - chunk.first) as usize;
            let cell = chunk.cell(off);
            let flags = chunk.flags[off].load(Relaxed);
            let same = flags & SAME != 0;
            let rec = *ctx.records.of(cell);
            let k = (at % ctx.schemas) as usize;
            refs.clear();
            let mut row = None;
            let mut row_hit = false;
            if ctx.rows && !same {
                let g = chunk.row_gen[off].load(Relaxed);
                if g != 0 {
                    let admit = g >> 31 != 0;
                    let w = chunk.row_at[off].load(Relaxed);
                    let list = chunk.row_frames[(w >> 32) as usize].lock().unwrap();
                    let o = (w as u32) as usize;
                    frames.clear();
                    frames.extend_from_slice(&list[o..o + rec.row_frames as usize]);
                    row = Some((admit, g & !(1 << 31)));
                    row_hit = !admit;
                    if admit {
                        t.row_admissions += 1;
                        loads.add(&ctx.pricing.admit(&rec));
                    } else {
                        t.row_hits += 1;
                        loads.add(&ctx.pricing.gather(&rec));
                    }
                }
            }
            if !same && !row_hit {
                let s = chunk.slot[off].load(Relaxed) as usize;
                let mut any = false;
                let mut run = false;
                for i in 0..rec.pages as usize {
                    let r = all[s + i].load(Relaxed);
                    t.page_uses += 1;
                    if r == MISS || ref_admit(r) {
                        if r != MISS {
                            any = true;
                            t.page_admissions += 1;
                        }
                        loads.bytes += page as f64;
                        if !run {
                            loads.requests += 1.0;
                        }
                        run = true;
                    } else {
                        any = true;
                        t.page_hits += 1;
                        run = false;
                    }
                    refs.push(r);
                }
                if !any {
                    refs.clear();
                }
            }
            // Extra admissions of this access, as runs of one file.
            runs.clear();
            extra_refs.clear();
            while xi < extras.len() && (extras[xi].0 as usize) < off {
                xi += 1;
            }
            while xi < extras.len() && extras[xi].0 as usize == off {
                let p0 = extras[xi].1;
                let f = ctx.graph.files.partition_point(|f| f.first_page <= p0) - 1;
                let file = &ctx.graph.files[f];
                let file_end = file.first_page + file.len.div_ceil(page);
                let lo = extra_refs.len();
                let mut p = p0;
                while xi < extras.len() && extras[xi].0 as usize == off && extras[xi].1 == p && p < file_end && p - p0 < MAX_BLOCK_PAGES as u64 {
                    extra_refs.push(extras[xi].2);
                    xi += 1;
                    p += 1;
                }
                t.page_admissions += p - p0;
                t.block_reads += 1;
                loads.requests += 1.0;
                loads.bytes += ((p - p0) * page) as f64;
                runs.push((Extra { file: f as u32, offset: (p0 - file.first_page) * page, pages: (p - p0) as u32 }, lo..extra_refs.len()));
            }
            loads.core_seconds += ctx.pricing.access() + ctx.pricing.deliver_cost(k, &rec, row.is_some());
            if !same && !row_hit {
                loads.core_seconds += ctx.pricing.decode(&rec, cold);
            }
            if let Some(b) = buf.as_deref_mut() {
                let ex: Vec<(Extra, &[u64])> = runs.iter().map(|(e, r)| (*e, &extra_refs[r.clone()])).collect();
                b.push(&AccessPlan { cell, same, refs: &refs, extras: &ex, row: row.map(|(admit, g)| (admit, g, &frames[..])) });
            }
        }
        a = end;
    }
    t.requests = loads.requests as u64;
    if let Some(cm) = cost {
        loads.core_seconds += loads.requests * cm.request;
        t.seconds = cm.seconds(&loads);
        t.worker_seconds = loads.core_seconds;
    }
    Ok(t)
}

/// Global row selection: every row interval of the stored sequence, by
/// value density, kept where its frames fit; the decisions are replayed
/// by the row steps.
fn global_rows(ctx: &Ctx<'_>, cells: &[CellId], shards: &mut [Mutex<Shard>]) -> Result<()> {
    let d = ctx.p.d;
    let Some(cost) = ctx.pricing.cost else { return Ok(()) };
    // Rows are valued at the prices of the uncached sequence.
    let mut cold = vec![0u8; cells.len()];
    crate::opt::records::cold_bits(ctx.records, |at| Some(cells[at as usize]), 0, cells, &mut cold);
    let (mut loads, mut decoded, mut colds) = (Loads::default(), 0u64, 0u64);
    for (a, &cell) in cells.iter().enumerate() {
        let r: &CellRecord = ctx.records.of(cell);
        let own = (a as u64).is_multiple_of(ctx.schemas) || cells[a - 1] != cell;
        if own && !r.empty() {
            decoded += 1;
            colds += cold[a] as u64;
        }
        loads.add(&ctx.pricing.uncached(a % ctx.schemas as usize, r, own, cold[a] as f64));
    }
    let prices = cost.prices(&loads);
    let cold_share = if decoded == 0 { 0.0 } else { colds as f64 / decoded as f64 };
    let mut last: Vec<u64> = vec![NONE; ctx.records.cells as usize];
    let mut cands: Vec<Vec<(f64, u64, u64)>> = vec![Vec::new(); shards.len()];
    for (a, &cell) in cells.iter().enumerate() {
        let a = a as u64;
        if !a.is_multiple_of(ctx.schemas) && cells[a as usize - 1] == cell {
            continue;
        }
        let rec = ctx.records.of(cell);
        if rec.empty() {
            continue;
        }
        let u = std::mem::replace(&mut last[cell as usize], a);
        if u == NONE {
            continue;
        }
        let value = ctx.pricing.row_value(&prices, u, a, cell, false, cold_share, 0.0);
        if value <= 0.0 {
            continue;
        }
        let density = value / (rec.row_frames as f64 * (a - u.saturating_sub(d) + 1) as f64);
        cands[ctx.keys.cell_shard[cell as usize] as usize].push((density, u, a));
    }
    for (j, c) in cands.iter_mut().enumerate() {
        let mut sh = shards[j].lock().unwrap();
        // Every block of the sequence enters once.
        let block = 1u64 << sh.occ.shift();
        let mut at = 0;
        while at < ctx.len + d {
            sh.occ.enter(at, 0);
            at += block;
        }
        sh.entered = u64::MAX;
        c.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.2.cmp(&y.2)));
        // Intervals kept, by their right end: an interval whose left end is
        // a kept right end extends that residency.
        let mut kept_end = std::collections::HashSet::new();
        for &(_, u, v) in c.iter() {
            let rec = ctx.records.of(cells[v as usize]);
            let w = rec.row_frames;
            let lo = if kept_end.contains(&u) { u + 1 } else { u.saturating_sub(d) };
            let cap = sh.cap;
            if w <= cap && sh.occ.fits(lo, v, w, cap) {
                sh.occ.add(lo, v, w);
                kept_end.insert(v);
                ctx.global[v as usize].store(1, Relaxed);
            }
        }
    }
    Ok(())
}
