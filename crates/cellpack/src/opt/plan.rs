//! Plans: the residencies a compilation chose, each with the cache frames
//! it holds. A plan is bound to the graph's inputs, schemas and slots, and
//! to the exact access sequence.
//!
//! A residency keeps one object (a page, or a cell's row) from the access
//! that admits it (`start`) through its last use (`end`), in frames the
//! plan assigns. Two residencies of one frame are more than the reuse gap
//! `D` apart (`start − D` follows the earlier `end`): the later one's span
//! starts at `start − D`, so with an execution window of at most `D` every
//! reader of the earlier one has finished before the later one's admission
//! is issued. A frame's generation counts its residencies; the admitting
//! access publishes it once its read (or row) is complete, and every other
//! use waits for it.
//!
//! The loader reads plans through the plan ring: a [`Sweep`] expands the
//! residencies into per-access entries (see `loader::planring`), and the
//! same sweep replays a plan serially to check it and count its summary.

use std::path::Path;
use std::sync::OnceLock;

use crate::loader::catalog::CellId;
use crate::loader::convert::CONVERSIONS;
use crate::loader::error::{Error, Result};
use crate::loader::planring::{AccessPlan, EntryBuf, Extra, MISS, page_ref};
use crate::loader::schema::Graph;
use crate::opt::model::{CostModel, Loads};
use crate::opt::price::Pricing;
use crate::opt::records::{Records, Trace};

/// A cached object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Object {
    /// A physical input page (global page id).
    Page(u64),
    /// A cell's row, in its stored type.
    Row(CellId),
}

/// One residency: an object held from its admission through its last use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency {
    /// The object it holds.
    pub object: Object,
    /// The admitting access.
    pub start: u64,
    /// The last use.
    pub end: u64,
    /// Index of its first frame id in the plan's frame list.
    pub(crate) frames_at: u64,
    /// How many frame ids it holds: 1 for a page, the row's frames for a row.
    pub(crate) frames: u32,
}

/// Most pages one extra admission read covers.
pub(crate) const MAX_BLOCK_PAGES: u32 = 128;

/// Default execution window in accesses for a worker count and in-flight
/// depth: four times the accesses in flight.
pub fn default_distance(workers: usize, inflight_per_worker: usize) -> u64 {
    4u64.saturating_mul(workers as u64).saturating_mul(inflight_per_worker as u64)
}

/// Identity of an access sequence, checked before a plan is used.
pub fn access_fingerprint(accesses: &[CellId]) -> u64 {
    let mut h = 0x243f_6a88_85a3_08d3u64 ^ accesses.len() as u64;
    for pair in accesses.chunks(2) {
        let v = pair[0] as u64 | (pair.get(1).copied().unwrap_or(0) as u64) << 32;
        h = (h ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(31);
    }
    h
}

/// The serial execution of a plan, as an independent replay counts it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanSummary {
    /// Accesses in the sequence.
    pub accesses: u64,
    /// Page uses of accesses that read input.
    pub page_uses: u64,
    /// The uses a residency serves.
    pub page_hits: u64,
    /// Read requests: runs of pages an access reads, and extra admission
    /// reads.
    pub requests: u64,
    /// The bytes those requests read.
    pub read_bytes: u64,
    /// Page residencies.
    pub page_residencies: u64,
    /// The row residencies.
    pub row_residencies: u64,
    /// Extra admission reads (block reads).
    pub block_reads: u64,
    /// Records decoded.
    pub decoded_rows: u64,
    /// Their stored elements.
    pub decoded_elements: u64,
    /// Those decoded in a cold chunk.
    pub cold_rows: u64,
    /// Accesses served by a cached row.
    pub row_hits: u64,
    /// The served rows' image elements.
    pub row_elements: u64,
    /// The served rows' payload bytes.
    pub row_bytes: u64,
    /// Elements of the rows admitted.
    pub admitted_elements: u64,
    /// Payload bytes of the rows admitted.
    pub admitted_bytes: u64,
    /// Elements each schema maps, per schema.
    pub mapped_elements: Vec<u64>,
    /// The dense default fills, per schema.
    pub filled_elements: Vec<u64>,
    /// Elements converted, per conversion class.
    pub converted_elements: [u64; CONVERSIONS],
    /// Per schema: process function calls.
    pub process_calls: Vec<u64>,
    /// Per schema: partial process calls.
    pub partial_calls: Vec<u64>,
    /// The stored elements those partial calls saw.
    pub partial_elements: Vec<u64>,
    /// Most cache bytes held at once in the serial order.
    pub peak_bytes: u64,
    /// The same, with every residency reserved from `start − D` (what the
    /// frames must hold).
    pub concurrent_peak_bytes: u64,
    /// Page uses per quarter of the sequence (one epoch of a 4-epoch run).
    pub quarter_uses: [u64; 4],
    /// Page hits per quarter.
    pub quarter_hits: [u64; 4],
    /// Read requests per quarter.
    pub quarter_requests: [u64; 4],
    /// Modeled seconds per quarter; zero without a cost model.
    pub quarter_seconds: [f64; 4],
    /// Modeled core-seconds of the workers per quarter.
    pub quarter_worker_seconds: [f64; 4],
    /// The modeled core-seconds of the workers over the sequence.
    pub worker_seconds: f64,
    /// Each resource's service time per quarter (requests, bytes, cores,
    /// memory; see [`CostModel::service`]): which one binds.
    pub quarter_service_seconds: [[f64; 4]; 4],
}

impl PlanSummary {
    /// Accesses in each quarter; the quarters split the sequence the same way
    /// the per-quarter counters do.
    pub fn quarter_accesses(&self) -> [u64; 4] {
        std::array::from_fn(|q| (self.accesses * (q as u64 + 1)).div_ceil(4) - (self.accesses * q as u64).div_ceil(4))
    }
    /// Modeled seconds of the whole sequence: the quarters summed.
    pub fn seconds(&self) -> f64 {
        self.quarter_seconds.iter().sum()
    }
}

/// A cache plan: the residencies a compilation chose, the frames they hold,
/// and what they were compiled for.
pub struct Plan {
    /// Schemas of the graph it was compiled for.
    schemas: u64,
    slot_cells: u64,
    /// The graph's fingerprint, checked before the plan runs.
    graph: u64,
    /// Fingerprint of the access sequence, likewise.
    access: u64,
    /// Accesses in the sequence.
    len: u64,
    /// The reuse gap `D` in accesses.
    distance: u64,
    /// Cache bytes the plan was compiled with.
    budget: usize,
    cache_frames: u32,
    residencies: Vec<Residency>,
    /// Frame ids, each residency's run of them in residency order.
    frames: Vec<u32>,
    /// Per residency, its frames' generation.
    gens: Vec<u32>,
    cost: Option<CostModel>,
    /// Counted by the replay that verifies the plan: when it is built, or for
    /// a plan read from a file, on its first verification.
    summary: OnceLock<PlanSummary>,
}

/// A residency as compilers hand it over.
#[derive(Clone, Debug)]
pub(crate) struct Built {
    /// The object the residency holds.
    pub object: Object,
    /// The admitting access.
    pub start: u64,
    /// The last use.
    pub end: u64,
    /// Global frame ids of the object's frames, in the order it holds them.
    pub frames: Vec<u32>,
}

impl Plan {
    /// Builds a plan from residencies with their frames, checks the frame
    /// rule and replays it to count its summary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble(graph: &Graph, records: &Records, trace: &Trace, distance: u64, budget: usize, cost: Option<CostModel>, mut built: Vec<Built>) -> Result<Self> {
        built.sort_unstable_by_key(|r| (r.start, r.object));
        let mut residencies = Vec::with_capacity(built.len());
        let mut frames = Vec::with_capacity(built.len());
        for b in built {
            residencies.push(Residency { object: b.object, start: b.start, end: b.end, frames_at: frames.len() as u64, frames: b.frames.len() as u32 });
            frames.extend_from_slice(&b.frames);
        }
        let mut plan = Self {
            schemas: graph.schemas.len() as u64,
            slot_cells: graph.slot_cells,
            graph: graph.fingerprint,
            access: access_fingerprint(&trace.cells),
            len: trace.len() as u64,
            distance,
            budget,
            cache_frames: (budget / graph.page_size) as u32,
            residencies,
            frames,
            gens: Vec::new(),
            cost,
            summary: OnceLock::new(),
        };
        plan.gens = plan.check_frames()?;
        let summary = plan.simulate(graph, records, trace, plan.cost.as_ref())?;
        let _ = plan.summary.set(summary);
        Ok(plan)
    }

    /// The serial replay's counts; all zero for a plan read from a file until
    /// a loader verifies it (`prepare` or `replay`).
    pub fn summary(&self) -> &PlanSummary {
        static EMPTY: OnceLock<PlanSummary> = OnceLock::new();
        self.summary.get().unwrap_or_else(|| EMPTY.get_or_init(PlanSummary::default))
    }
    /// The reuse gap `D` in accesses.
    pub fn distance(&self) -> u64 {
        self.distance
    }
    /// Schemas of the graph the plan was compiled for.
    pub fn schemas(&self) -> u64 {
        self.schemas
    }
    /// Positions per output slot of that graph.
    pub fn slot_cells(&self) -> u64 {
        self.slot_cells
    }
    /// Accesses in the sequence.
    pub fn len(&self) -> u64 {
        self.len
    }
    /// Whether the plan covers no access.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Cache bytes it was compiled with.
    pub fn cache_bytes(&self) -> usize {
        self.budget
    }
    /// Cache frames the plan was compiled with; frame ids lie below it.
    pub fn cache_frames(&self) -> u32 {
        self.cache_frames
    }
    /// The cost model it was compiled with, if any.
    pub fn cost(&self) -> Option<&CostModel> {
        self.cost.as_ref()
    }
    /// The reuse gap in slots.
    pub fn reuse_gap(&self) -> u64 {
        self.distance / (self.schemas * self.slot_cells)
    }
    /// Residencies in admission order.
    pub fn residencies(&self) -> &[Residency] {
        &self.residencies
    }
    /// The global frame ids a residency holds, in order.
    pub fn frames_of(&self, r: &Residency) -> &[u32] {
        &self.frames[r.frames_at as usize..r.frames_at as usize + r.frames as usize]
    }

    /// Checks every frame id and the gap between consecutive residencies of
    /// a frame; returns each residency's generation. A residency's generation
    /// is one past the highest its frames carried, so a use that waits for an
    /// earlier residency of the same frame cannot be satisfied by the later
    /// one; the mask keeps it a 31-bit value in refs.
    fn check_frames(&self) -> Result<Vec<u32>> {
        let n = self.cache_frames as usize;
        let mut last = vec![u64::MAX; n];
        let mut count = vec![0u32; n];
        let mut gens = Vec::with_capacity(self.residencies.len());
        for r in &self.residencies {
            if r.end < r.start || r.end >= self.len || r.frames == 0 {
                return Err(Error::Plan("residency outside the sequence or without frames".into()));
            }
            let mut g = 0;
            for &f in self.frames_of(r) {
                let f = f as usize;
                if f >= n {
                    return Err(Error::Plan("residency frame outside the plan's cache".into()));
                }
                if last[f] != u64::MAX && r.start <= last[f] + self.distance {
                    return Err(Error::Plan("residencies of one frame closer than the reuse gap".into()));
                }
                g = g.max(count[f]);
            }
            let g = (g + 1) & crate::loader::planring::GEN_MASK;
            for &f in self.frames_of(r) {
                count[f as usize] = g;
                last[f as usize] = r.end;
            }
            gens.push(g);
        }
        Ok(gens)
    }

    /// Checks that the plan was compiled for these inputs, schemas and
    /// accesses, and (once, for plans read from files) that it keeps every
    /// rule.
    pub(crate) fn verify(&self, graph: &Graph, accesses: &[CellId]) -> Result<()> {
        if self.schemas != graph.schemas.len() as u64 || self.slot_cells != graph.slot_cells || self.graph != graph.fingerprint || self.len != accesses.len() as u64 || self.access != access_fingerprint(accesses) {
            return Err(Error::Plan("plan was built for different inputs, schemas or accesses".into()));
        }
        if self.summary.get().is_none() {
            let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
            let records = Records::build(graph, threads)?;
            let trace = Trace::build(&records, accesses, threads)?;
            let summary = self.simulate(graph, &records, &trace, self.cost.as_ref())?;
            let _ = self.summary.set(summary);
        }
        Ok(())
    }

    /// Replays the plan serially, independently of the compiler.
    pub(crate) fn replay(&self, graph: &Graph, accesses: &[CellId], cost: Option<&CostModel>) -> Result<PlanSummary> {
        self.verify(graph, accesses)?;
        if let Some(c) = cost {
            c.check(graph)?;
        }
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let records = Records::build(graph, threads)?;
        let trace = Trace::build(&records, accesses, threads)?;
        self.simulate(graph, &records, &trace, cost.or(self.cost.as_ref()))
    }

    /// Serial replay: checks the uses of every residency and counts the
    /// summary.
    fn simulate(&self, graph: &Graph, records: &Records, trace: &Trace, cost: Option<&CostModel>) -> Result<PlanSummary> {
        let n = self.len as usize;
        let d = self.distance;
        if let Some(c) = cost {
            c.check(graph)?;
        }
        let schemas = graph.schemas.len();
        let mut s = PlanSummary {
            accesses: n as u64,
            process_calls: vec![0; schemas],
            partial_calls: vec![0; schemas],
            partial_elements: vec![0; schemas],
            mapped_elements: vec![0; schemas],
            filled_elements: vec![0; schemas],
            ..Default::default()
        };
        let page = graph.page_size as u64;
        // Occupancy in frames over accesses, serial and reserved: a residency
        // holds its frames from its last use back to `start − D`, which is
        // what the loader's frames must cover.
        let mut serial = vec![0i64; n + 1];
        let mut reserved = vec![0i64; n + 1];
        for r in &self.residencies {
            match r.object {
                Object::Page(_) => s.page_residencies += 1,
                Object::Row(_) => s.row_residencies += 1,
            }
            let w = r.frames as i64;
            serial[r.start as usize] += w;
            serial[r.end as usize + 1] -= w;
            reserved[r.start.saturating_sub(d) as usize] += w;
            reserved[r.end as usize + 1] -= w;
        }
        let peak = |v: &[i64]| v.iter().scan(0i64, |h, x| { *h += x; Some(*h) }).max().unwrap_or(0).max(0) as u64;
        s.peak_bytes = peak(&serial) * page;
        s.concurrent_peak_bytes = peak(&reserved) * page;
        if s.concurrent_peak_bytes > (self.cache_frames as u64) * page {
            return Err(Error::Plan("residencies hold more frames than the cache at once".into()));
        }
        drop((serial, reserved));

        let pricing = Pricing::new(graph, records, cost, false);
        let mut sweep = Sweep::new(self, graph, records);
        let mut buf = EntryBuf::default();
        let mut loads = Loads::default();
        let bucket = cost.map_or(u64::MAX, |c| c.bucket);
        let mut since = 0u64;
        let mut q = 0;
        // Quarters end at `ceil(n · (q + 1) / 4)`, the cut
        // `PlanSummary::quarter_accesses` reports.
        let mut quarter_end_at = (n as u64).div_ceil(4) as usize;
        for at in 0..n {
            while at >= quarter_end_at {
                q += 1;
                quarter_end_at = (n as u64 * (q as u64 + 1)).div_ceil(4) as usize;
            }
            let cell = trace.cells[at];
            let k = at % schemas;
            let same = k > 0 && trace.cells[at - 1] == cell;
            buf.clear(at as u64);
            let st = sweep.step(at as u64, cell, same, &mut buf)?;
            let record = *records.of(cell);
            if !graph.schemas[k].is_builtin() {
                s.process_calls[k] += 1;
            }
            let whole = st.row_hit || st.row_admit;
            let (mapped, filled) = pricing.delivery_work(k, &record, whole);
            let partial = pricing.uses_partial(k, &record, whole);
            s.partial_calls[k] += u64::from(partial);
            if partial {
                s.partial_elements[k] += record.elements as u64;
            }
            s.mapped_elements[k] += mapped;
            s.filled_elements[k] += filled;
            if mapped != 0 {
                let class = crate::loader::convert::Conversion::of(crate::loader::convert::DTYPES[record.dtype as usize], graph.schemas[k].dtype());
                s.converted_elements[class as usize] += mapped;
            }
            loads.core_seconds += pricing.access() + pricing.deliver_cost(k, &record, whole);
            loads.requests += st.requests as f64;
            loads.bytes += (st.read_pages * page) as f64;
            loads.core_seconds += st.requests as f64 * cost.map_or(0.0, |c| c.request);
            s.block_reads += st.extra_reads;
            if st.row_hit {
                s.row_hits += 1;
                s.row_elements += record.image_elements as u64;
                s.row_bytes += record.row_bytes as u64;
                loads.add(&pricing.gather(&record));
            } else if !same {
                let cold = trace.cold(at);
                if !record.empty() {
                    s.decoded_rows += 1;
                    s.decoded_elements += record.elements as u64;
                    s.cold_rows += cold as u64;
                    loads.core_seconds += pricing.decode(&record, cold as u8 as f64);
                }
                s.page_uses += record.pages as u64;
                s.quarter_uses[q] += record.pages as u64;
                s.page_hits += st.hits;
                s.quarter_hits[q] += st.hits;
            }
            if st.row_admit {
                s.admitted_elements += record.image_elements as u64;
                s.admitted_bytes += record.row_bytes as u64;
                loads.add(&pricing.admit(&record));
            }
            s.quarter_requests[q] += st.requests;
            since += 1;
            let quarter_end = quarter_end_at == at + 1;
            if since >= bucket || quarter_end {
                if let Some(c) = cost {
                    s.quarter_seconds[q] += c.seconds(&loads);
                    for (t, x) in s.quarter_service_seconds[q].iter_mut().zip(c.service(&loads)) {
                        *t += x;
                    }
                    s.quarter_worker_seconds[q] += loads.core_seconds;
                    s.worker_seconds += loads.core_seconds;
                }
                s.requests += loads.requests as u64;
                s.read_bytes += loads.bytes as u64;
                (loads, since) = (Loads::default(), 0);
            }
        }
        sweep.finish()?;
        Ok(s)
    }

    /// Writes the plan file at `path`.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }

    /// The plan file's bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Header (magic, version, identities, counts), the cost model behind a
        // present flag, the residencies, their frame ids, then the checksum.
        let mut w = Vec::with_capacity(128 + self.residencies.len() * 33 + self.frames.len() * 4);
        w.extend_from_slice(MAGIC);
        for v in [VERSION, self.graph, self.access, self.len, self.distance, self.budget as u64, self.cache_frames as u64, self.schemas, self.slot_cells, self.residencies.len() as u64, self.frames.len() as u64] {
            w.extend_from_slice(&v.to_le_bytes());
        }
        match &self.cost {
            None => w.push(0),
            Some(c) => {
                w.push(1);
                c.encode(&mut w);
            }
        }
        for r in &self.residencies {
            let (kind, key) = match r.object {
                Object::Page(p) => (0u8, p),
                Object::Row(c) => (1, c as u64),
            };
            w.push(kind);
            w.extend_from_slice(&key.to_le_bytes());
            w.extend_from_slice(&r.start.to_le_bytes());
            w.extend_from_slice(&r.end.to_le_bytes());
            w.extend_from_slice(&r.frames.to_le_bytes());
        }
        for &f in &self.frames {
            w.extend_from_slice(&f.to_le_bytes());
        }
        let sum = checksum(&w);
        w.extend_from_slice(&sum.to_le_bytes());
        w
    }

    /// Reads a plan file; it is checked against the graph and accesses it is
    /// run with when it is verified (`prepare` or `replay`).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Reads [`Plan::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let bad = || Error::Plan("not a valid plan file".into());
        if bytes.len() < MAGIC.len() + 8 || &bytes[..MAGIC.len()] != MAGIC {
            return Err(bad());
        }
        let (body, sum) = bytes.split_at(bytes.len() - 8);
        if checksum(body) != u64::from_le_bytes(sum.try_into().unwrap()) {
            return Err(Error::Plan("plan file checksum mismatch".into()));
        }
        let mut r = Reader { bytes: &body[MAGIC.len()..], at: 0 };
        if r.u64()? != VERSION {
            return Err(Error::Plan("unsupported plan version".into()));
        }
        let graph = r.u64()?;
        let access = r.u64()?;
        let len = r.u64()?;
        let distance = r.u64()?;
        let budget = r.u64()? as usize;
        let cache_frames = r.u64()?;
        let schemas = r.u64()?;
        let slot_cells = r.u64()?;
        let count = r.u64()? as usize;
        let nframes = r.u64()? as usize;
        if !(1..=crate::loader::schema::MAX_SCHEMAS as u64).contains(&schemas)
            || !(1..=u32::MAX as u64).contains(&slot_cells)
            || !len.is_multiple_of(schemas)
            || distance == 0
            || cache_frames > u32::MAX as u64
            || count > body.len() / 29
            || nframes > body.len() / 4
        {
            return Err(bad());
        }
        let cost = match r.u8()? {
            0 => None,
            1 => Some(CostModel::decode(&mut r)?),
            _ => return Err(bad()),
        };
        let mut residencies = Vec::with_capacity(count);
        let mut at = 0u64;
        for _ in 0..count {
            let kind = r.u8()?;
            let key = r.u64()?;
            let object = match kind {
                0 => Object::Page(key),
                1 if key <= u32::MAX as u64 => Object::Row(key as CellId),
                _ => return Err(bad()),
            };
            let (start, end, k) = (r.u64()?, r.u64()?, r.u32()?);
            residencies.push(Residency { object, start, end, frames_at: at, frames: k });
            at += k as u64;
        }
        if at != nframes as u64 {
            return Err(bad());
        }
        let mut frames = Vec::with_capacity(nframes);
        for _ in 0..nframes {
            frames.push(r.u32()?);
        }
        if r.at != r.bytes.len() || residencies.windows(2).any(|w| (w[0].start, w[0].object) > (w[1].start, w[1].object)) {
            return Err(bad());
        }
        let mut plan = Self { schemas, slot_cells, graph, access, len, distance, budget, cache_frames: cache_frames as u32, residencies, frames, gens: Vec::new(), cost, summary: OnceLock::new() };
        plan.gens = plan.check_frames()?;
        Ok(plan)
    }
}

/// What one access does under a plan, as a [`Sweep`] counts it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StepCounts {
    /// Page uses a residency serves.
    pub hits: u64,
    /// Read requests: own runs and extra admission reads.
    pub requests: u64,
    /// Pages read: the record's missing or admitted pages, and the pages of
    /// extra admission reads.
    pub read_pages: u64,
    /// Extra admission reads (block reads), each one request.
    pub extra_reads: u64,
    /// The access is served by a cached row.
    pub row_hit: bool,
    /// The access builds the cell's cached row.
    pub row_admit: bool,
}

/// Expands a plan into per-access entries in sequence order, resolving each
/// page and row use to its residency's frame and generation.
pub(crate) struct Sweep<'p> {
    plan: &'p Plan,
    graph: &'p Graph,
    records: &'p Records,
    /// The latest residency of each page.
    page_res: Vec<u32>,
    /// The latest residency of each cell's row (empty without row caching).
    row_res: Vec<u32>,
    /// Index of the next residency to admit, in `plan.residencies` order.
    next: usize,
    /// Page refs of the access being expanded, one per record page in file
    /// order, or empty when it reads no record page through a residency.
    refs: Vec<u64>,
    /// Pages admitted here that this access does not read itself (extra
    /// admission reads), with their refs, by page.
    extras: Vec<(u64, u64)>,
    /// Runs of consecutive pages among `extras`, and the span of each run's
    /// refs in `extra_refs`.
    runs: Vec<(Extra, std::ops::Range<usize>)>,
    /// The refs of `runs`, in run order.
    extra_refs: Vec<u64>,
    /// Frames of the cached row the access serves or admits.
    row_frames: Vec<u32>,
}

/// No residency: residency indices stay below it, so a table of them can use
/// it as a sentinel.
const NIL: u32 = u32::MAX;

impl<'p> Sweep<'p> {
    /// A sweep over `plan`; `graph` and `records` must be the plan's (it was
    /// verified before).
    pub fn new(plan: &'p Plan, graph: &'p Graph, records: &'p Records) -> Self {
        let rows = plan.residencies.iter().any(|r| matches!(r.object, Object::Row(_)));
        Self {
            plan,
            graph,
            records,
            page_res: vec![NIL; graph.pages as usize],
            row_res: vec![NIL; if rows { records.cells as usize } else { 0 }],
            next: 0,
            refs: Vec::new(),
            extras: Vec::new(),
            runs: Vec::new(),
            extra_refs: Vec::new(),
            row_frames: Vec::new(),
        }
    }

    fn reference(&self, i: u32, at: u64) -> u64 {
        let r = &self.plan.residencies[i as usize];
        page_ref(self.plan.frames[r.frames_at as usize], self.plan.gens[i as usize], r.start == at)
    }

    /// Access `at` (of `cell`; `same`: the previous access of its position
    /// loads the same cell): its entry into `buf`, and its counts.
    pub fn step(&mut self, at: u64, cell: CellId, same: bool, buf: &mut EntryBuf) -> Result<StepCounts> {
        let plan = self.plan;
        let record = *self.records.of(cell);
        let mut c = StepCounts::default();
        self.extras.clear();
        // Residencies admitted here.
        while self.next < plan.residencies.len() && plan.residencies[self.next].start == at {
            let i = self.next as u32;
            let r = &plan.residencies[self.next];
            match r.object {
                Object::Page(p) => {
                    if p >= self.graph.pages {
                        return Err(Error::Plan("page outside the inputs".into()));
                    }
                    self.page_res[p as usize] = i;
                    self.extras.push((p, i as u64));
                }
                Object::Row(rc) => {
                    if rc != cell || same || self.row_res.is_empty() {
                        return Err(Error::Plan(format!("row residency of cell {rc} admitted at access {at}, which does not load it")));
                    }
                    if r.frames != self.records.of(rc).row_frames {
                        return Err(Error::Plan("row residency with the wrong number of frames".into()));
                    }
                    self.row_res[rc as usize] = i;
                }
            }
            self.next += 1;
        }
        if self.next < plan.residencies.len() && plan.residencies[self.next].start < at {
            return Err(Error::Plan("residencies out of order".into()));
        }
        // A frame's residency table can name a residency that has already
        // ended, so a use only counts when the span covers the access.
        let covers = |i: u32| i != NIL && plan.residencies[i as usize].start <= at && at <= plan.residencies[i as usize].end;
        let mut row = None;
        if !same && !self.row_res.is_empty() {
            let i = self.row_res[cell as usize];
            if covers(i) {
                let r = &plan.residencies[i as usize];
                c.row_admit = r.start == at;
                c.row_hit = !c.row_admit;
                self.row_frames.clear();
                self.row_frames.extend_from_slice(plan.frames_of(r));
                row = Some((c.row_admit, plan.gens[i as usize], ()));
            }
        }
        // Pages admitted here that this access does not read itself are
        // extra admission reads.
        let reads_own = !same && !c.row_hit;
        let mut kept = 0;
        for k in 0..self.extras.len() {
            let (p, i) = self.extras[k];
            if !reads_own || !record.page_range().contains(&p) {
                self.extras[kept] = (p, self.reference(i as u32, at));
                kept += 1;
            }
        }
        self.extras.truncate(kept);
        self.refs.clear();
        if reads_own {
            // Pages in file order: a run of pages that are read (admitted or
            // missing) is one request, a hit ends it.
            let mut any = false;
            let mut run = false;
            for p in record.page_range() {
                let i = self.page_res[p as usize];
                if covers(i) {
                    let rf = self.reference(i, at);
                    any = true;
                    if crate::loader::planring::ref_admit(rf) {
                        c.read_pages += 1;
                        if !run {
                            c.requests += 1;
                        }
                        run = true;
                    } else {
                        c.hits += 1;
                        run = false;
                    }
                    self.refs.push(rf);
                } else {
                    c.read_pages += 1;
                    if !run {
                        c.requests += 1;
                    }
                    run = true;
                    self.refs.push(MISS);
                }
            }
            if !any {
                self.refs.clear();
            }
        }
        // Extra admissions as runs of consecutive pages of one file.
        self.runs.clear();
        self.extra_refs.clear();
        self.extras.sort_unstable_by_key(|e| e.0);
        let page = self.graph.page_size as u64;
        let mut k = 0;
        while k < self.extras.len() {
            let (p0, _) = self.extras[k];
            let f = self.graph.files.partition_point(|f| f.first_page <= p0) - 1;
            let file = &self.graph.files[f];
            let end_page = file.first_page + file.len.div_ceil(page);
            let lo = self.extra_refs.len();
            let mut p = p0;
            while k < self.extras.len() && self.extras[k].0 == p && p < end_page && (p - p0) < MAX_BLOCK_PAGES as u64 {
                self.extra_refs.push(self.extras[k].1);
                k += 1;
                p += 1;
            }
            self.runs.push((Extra { file: f as u32, offset: (p0 - file.first_page) * page, pages: (p - p0) as u32 }, lo..self.extra_refs.len()));
            c.requests += 1;
            c.extra_reads += 1;
            c.read_pages += p - p0;
        }
        let extras: Vec<(Extra, &[u64])> = self.runs.iter().map(|(e, r)| (*e, &self.extra_refs[r.clone()])).collect();
        buf.push(&AccessPlan { cell, same, refs: &self.refs, extras: &extras, row: row.map(|(a, g, ())| (a, g, &self.row_frames[..])) });
        Ok(c)
    }

    /// Checks the sweep reached every residency.
    pub fn finish(&self) -> Result<()> {
        if self.next != self.plan.residencies.len() {
            return Err(Error::Plan("residency admitted past the sequence".into()));
        }
        Ok(())
    }
}

// `CPKPLAN1` and the version every reader requires.
const MAGIC: &[u8; 8] = b"CPKPLAN1";
const VERSION: u64 = 1;

/// The 64-bit checksum that ends every plan and cost model file: a
/// word-at-a-time hash over the bytes before it.
pub(crate) fn checksum(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for chunk in bytes.chunks(8) {
        let mut v = [0u8; 8];
        v[..chunk.len()].copy_from_slice(chunk);
        h = (h ^ u64::from_le_bytes(v)).wrapping_mul(0x0100_0000_01b3).rotate_left(23);
    }
    h
}

/// Little-endian fields of a binary file, bounds-checked.
pub(crate) struct Reader<'a> {
    /// The buffer being read.
    pub(crate) bytes: &'a [u8],
    /// The read cursor; after a typed read it points past the field.
    pub(crate) at: usize,
}

impl<'a> Reader<'a> {
    /// A reader at the start of `bytes`.
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or_else(|| Error::Plan("file truncated".into()))?;
        let s = &self.bytes[self.at..end];
        self.at = end;
        Ok(s)
    }
    /// The next byte.
    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    /// The next little-endian `u32`.
    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    /// The next little-endian `u64`.
    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// The next little-endian `f64`.
    pub(crate) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// The next count: a `u64` that must fit `unit`-byte fields in what is
    /// left, which bounds every allocation a reader makes from the file.
    pub(crate) fn count(&mut self, unit: usize) -> Result<usize> {
        let n = self.u64()? as usize;
        if n > (self.bytes.len() - self.at) / unit {
            return Err(Error::Plan("file truncated".into()));
        }
        Ok(n)
    }
}
