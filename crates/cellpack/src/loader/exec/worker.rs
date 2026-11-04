//! A worker: claims positions within the window, issues every read its
//! plan entries name, harvests completions one by one, waits for frames
//! other accesses admit, decodes and delivers.
//!
//! A task is one position: every schema's access of it. Accesses of a
//! position that load the same cell share one input (one read, one
//! decode). An input's pages come from cache frames (a use waits for the
//! frame's generation; an admission reads into the frame and publishes it)
//! or from the worker's private frames (misses), which return to the
//! worker's free stack once the input is decoded. A task completes when its
//! outputs are written and every read it issued has finished.

use std::collections::VecDeque;
use std::sync::atomic::Ordering::*;
use std::time::Duration;

use crate::loader::catalog::{CellId, Location};
use crate::loader::config::{PROCESS_BOUNDS, RunStats};
use crate::loader::convert::{RowOps, dtype_index};
use crate::loader::error::{Error, Result, config};
use crate::loader::exec::io::{Done, WorkerIo};
use crate::loader::exec::progress::{Claim, Outstanding};
use crate::loader::exec::{Mode, Shared, Source};
use crate::loader::memory::HugeBuf;
use crate::loader::planring::{Entry, MISS, ROW_ADMIT, ROW_USE, ref_admit, ref_frame, ref_gen};
use crate::loader::process::{ResolveFn, Row};
use crate::loader::rows;
use crate::prof::{Profiler, Site};
use crate::scans::{IdSpace, PagedRecord, Parts, Values};

/// Claims per loop round. A claim starts all its positions (their
/// admission reads go out at once); more claims per round would read far
/// ahead of decoding, so pages would wait, and cool, in memory.
const START_BUDGET: usize = 1;

/// Queued reads per submission system call (see `WorkerIo::flush`).
const SUBMIT_BATCH: u32 = 4;

/// Completed positions between snapshot publications.
const PUBLISH_EVERY: u32 = 1024;

/// How a page of an input is served: from a cache frame another access filled,
/// from a cache frame this access fills itself, or from a private frame the
/// task reads and frees.
const USE: u8 = 0;
const ADMIT: u8 = 1;
const PRIVATE: u8 = 2;

/// One access of a task: where its output goes, and which input it reads.
struct Acc {
    /// Access index in the sequence.
    position: u64,
    cell: CellId,
    /// Schema index within the position.
    schema: u32,
    /// The cell's `cell_bytes` in the slot, exactly as long as the schema's
    /// output.
    out: *mut u8,
    /// Batch (slot) the output belongs to, the counter `Ring::complete` uses.
    batch: u64,
    /// Index into the task's `inputs`.
    input: u8,
}

/// One distinct cell of a task: the pages of its record, how each is served,
/// the cache frames it admits (with the generation to publish) and, for a
/// cached row, the frames the plan gave it.
///
/// The vectors are reused across tasks: a finished task's inputs go back to the
/// worker's `spare` list, so their capacity survives.
struct Input<'a> {
    /// The cell's location; always set once the input is started, and `None`
    /// only in a reused slot before it is filled again.
    loc: Option<Location<'a>>,
    /// Index into the run's descriptor table.
    file: usize,
    /// One pointer per record page: a cache or private frame.
    frames: Vec<*mut u8>,
    /// Per page, one of [`USE`], [`ADMIT`], [`PRIVATE`].
    kinds: Vec<u8>,
    /// Admitted pages: their cache frame and the generation to publish.
    admits: Vec<(u32, u32)>,
    /// Pages issued (or not needing a read).
    issued: Vec<bool>,
    /// Private frames of this input, by local index.
    private: Vec<u32>,
    /// The input is served from a cached row instead of decoding its record.
    row_hit: bool,
    /// The access builds the cell's cached row into `row_frames`.
    row_admit: bool,
    /// The generation a row admission publishes, and the one a row hit
    /// waits for.
    row_gen: u32,
    /// The cache frames of the row.
    row_frames: Vec<u32>,
}

/// One position: every schema's access of it, and the distinct cells they load.
///
/// A task outlives its reads: it counts its own reads (`own`) and its extra
/// admission reads (`admit`), and completes only when the outputs are written
/// and both counts reach zero, so no read can still write into a frame the plan
/// has released.
struct Task<'a> {
    /// First access index of the position.
    first: u64,
    accs: Vec<Acc>,
    inputs: Vec<Input<'a>>,
    /// Reads outstanding whose data the decode needs.
    own: u32,
    /// Extra admission reads outstanding.
    admit: u32,
    /// Frames another access admits, with their generations.
    waits: Vec<(u32, u32)>,
    /// Private pages still to read.
    starved: u32,
    /// Outputs are written; only extra admissions may still be in flight.
    delivered: bool,
    /// Reads are being issued: completions must not ready the task yet.
    issuing: bool,
    /// In the ready queue or the waiting list.
    queued: bool,
    waited: bool,
    wait_mark: u64,
}

/// A worker's reusable buffers, one set for the whole run: decoding needs a
/// workspace and the record's pages, reads hand the kernel a run of frame
/// pointers and the frames to publish, and delivery converts into `out`.
struct Scratch {
    /// Decode workspace, sized by the catalog's largest chunk.
    decode: Vec<u64>,
    /// The pages of the record being decoded.
    pages: Vec<&'static [u8]>,
    /// The current plan entry's words.
    words: Vec<u64>,
    /// Frames of the read being issued.
    run: Vec<*mut u8>,
    /// Cache frames that read publishes, with their generations.
    publish: Vec<(u32, u32)>,
    /// Cache frames of a row being gathered or written.
    row_ptrs: Vec<*mut u8>,
    out: OutScratch,
}

/// Scratch of the conversion to outputs: the converted CSR row and dense row,
/// the record's dictionary and the gathered image of a cached row.
struct OutScratch {
    /// Output columns of a CSR row, one per element.
    columns: Vec<u32>,
    /// Converted values of a CSR row (any element type), in 8-byte words.
    values: Vec<u64>,
    /// The record's dictionary entries as the schema's dtype, in 8-byte
    /// words.
    dict: Vec<u64>,
    /// A built-in dense row as the schema's dtype, in 8-byte words.
    dense: Vec<u64>,
    /// A dense record's expanded stored values, a word per gene.
    stored: Vec<u64>,
    /// Standard ids of a cached sparse row's elements.
    row_ids: Vec<u32>,
    /// Values of the cached row, in the stored type.
    row_values: Vec<u64>,
    /// The process-call scratch buffer, sized for the largest schema's
    /// `scratch_bytes`.
    process: HugeBuf,
}

/// One worker thread's state: its task slots and their queues, its private
/// frames, its reads (io_uring or preadv) and its scratch.
///
/// Task slots are a fixed pool of `depth`; `free` lists the unused ones, `ready`
/// holds tasks whose reads and waited frames are ready, `waiting` those still
/// waiting for a cache frame's generation, and `starved` those whose private
/// pages could not be issued for lack of frames.
pub(crate) struct Worker<'a, P: Profiler, M: Mode> {
    /// Worker index, which selects its affinity and its `minima` slot.
    id: usize,
    s: &'a Shared<'a>,
    prof: P,
    io: WorkerIo,
    tasks: Vec<Task<'a>>,
    /// Task slots not in use.
    free: Vec<u32>,
    ready: VecDeque<u32>,
    waiting: Vec<u32>,
    starved: VecDeque<u32>,
    /// Inputs of finished tasks, reused with their buffers.
    spare: Vec<Input<'a>>,
    /// Private frame ids, a stack: taken from the back and pushed back when a
    /// task's input is done with them.
    private: Vec<u32>,
    private_base: *mut u8,
    outstanding: Outstanding,
    /// Cells of the claim being started.
    cells: Vec<CellId>,
    done: Vec<Done>,
    scratch: Scratch,
    /// This worker's counts, published to `snapshots`.
    pub stats: RunStats,
    /// The claim counter is past the end of the sequence.
    exhausted: bool,
    /// Rounds without progress, which lengthen the backoff.
    idle: u32,
    completed: u32,
    _mode: std::marker::PhantomData<M>,
}

/// Element `i` of a table the graph sizes for every id it hands out.
macro_rules! at {
    ($v:expr, $i:expr) => {{
        let i = $i as usize;
        debug_assert!(i < $v.len());
        // SAFETY: the graph sizes the table for every id it hands out.
        unsafe { $v.get_unchecked(i) }
    }};
}

/// A decoded record behind a partial-row process call: the row, its mapping
/// and the conversion kernels `resolve` needs.
struct PartialRow<'a> {
    parts: Parts<'a>,
    ops: &'a RowOps,
    map: &'a [u32],
    dict: &'a [u64],
    modes: &'a [u64],
}

/// The `ResolveFn` a partial-row process call receives: maps the positions it
/// asks for from the live decoded row into its buffers.
unsafe extern "C" fn resolve(row: *const std::ffi::c_void, positions: *const u32, count: u64, columns: *mut u32, values: *mut std::ffi::c_void) -> i32 {
    // SAFETY: the loader passes its live row for the duration of the call;
    // the function passes outputs (and positions) of `count` entries, values
    // aligned to the schema's element type.
    let (r, cols) = unsafe { (&*row.cast::<PartialRow>(), std::slice::from_raw_parts_mut(columns, count as usize)) };
    let ok = if positions.is_null() {
        count as usize == r.parts.elements() && r.ops.csr(&r.parts, r.map, r.dict, r.modes, cols, values.cast()).is_ok()
    } else {
        // SAFETY: as above.
        let positions = unsafe { std::slice::from_raw_parts(positions, count as usize) };
        r.ops.at(&r.parts, positions, r.map, r.dict, r.modes, cols, values.cast()).is_ok()
    };
    if ok { 0 } else { -1 }
}

impl<'a, P: Profiler, M: Mode> Worker<'a, P, M> {
    /// Builds a worker: task slots, private frame stack, the read queue and
    /// scratch sized by the catalog's largest chunk and row.
    ///
    /// The private frames are touched once here, by their owner, so the pages
    /// they touch fault in on the worker's node.
    pub fn new(id: usize, s: &'a Shared<'a>, prof: P) -> Result<Self> {
        let g = s.graph;
        let source = &g.source;
        let depth = s.depth;
        let dense_words = g.schemas.iter().map(|x| (x.out_dim() * x.dtype().width()).div_ceil(8)).max().unwrap_or(0);
        let rowcap = source.max_nnz.max(source.max_genes) + 1;
        let private_base = s.grid.private(id);
        for f in 0..s.private_frames {
            // SAFETY: the worker's private region holds `private_frames` frames.
            unsafe { private_base.add(f * s.page).write_volatile(0) };
        }
        Ok(Self {
            id,
            s,
            prof,
            io: WorkerIo::new(s.backend, depth * s.schemas as usize * 4 + 64, &s.fds)?,
            tasks: (0..depth)
                .map(|_| Task { first: 0, accs: Vec::with_capacity(s.schemas as usize), inputs: Vec::with_capacity(s.schemas as usize), own: 0, admit: 0, waits: Vec::new(), starved: 0, delivered: false, issuing: false, queued: false, waited: false, wait_mark: 0 })
                .collect(),
            free: (0..depth as u32).rev().collect(),
            ready: VecDeque::new(),
            waiting: Vec::new(),
            starved: VecDeque::new(),
            spare: Vec::new(),
            private: (0..s.private_frames as u32).rev().collect(),
            private_base,
            outstanding: Outstanding::default(),
            cells: Vec::new(),
            done: Vec::new(),
            scratch: Scratch {
                decode: vec![0; source.workspace_words],
                pages: Vec::with_capacity(source.max_pages),
                words: Vec::new(),
                run: Vec::new(),
                publish: Vec::new(),
                row_ptrs: Vec::new(),
                out: OutScratch {
                columns: vec![0; source.max_nnz.max(source.max_genes)],
                values: vec![0; rowcap],
                dict: vec![0; source.max_nnz + 1],
                dense: vec![0; dense_words + 1],
                stored: vec![0; source.max_genes + 1],
                row_ids: if M::ROWS { vec![0; rowcap] } else { Vec::new() },
                row_values: if M::ROWS { vec![0; rowcap] } else { Vec::new() },
                process: HugeBuf::zeroed(g.schemas.iter().map(|s| s.scratch_bytes()).max().unwrap_or(0))?,
                },
            },
            stats: RunStats::empty(g),
            exhausted: false,
            idle: 0,
            completed: 0,
            _mode: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    fn cache_frame(&self, f: u32) -> *mut u8 {
        self.s.grid.cache(f)
    }

    #[inline(always)]
    fn private_frame(&self, i: u32) -> *mut u8 {
        // SAFETY: private frame ids lie below the region's frame count.
        unsafe { self.private_base.add(i as usize * self.s.page) }
    }

    /// Runs the worker until the sequence is done: returns its statistics and
    /// its profiler's record. Reads still in flight are drained on drop.
    pub fn run(mut self) -> Result<(RunStats, P)> {
        let t = self.prof.start(Site::Worker);
        let r = self.drive();
        self.prof.stop(Site::Worker, t);
        r?;
        self.stats.io = self.io.stats;
        self.publish_snapshot();
        let stats = std::mem::take(&mut self.stats);
        let fresh = self.prof.fork();
        let prof = std::mem::replace(&mut self.prof, fresh);
        Ok((stats, prof))
    }

    /// Publishes this worker's counters and profile words to its snapshot
    /// slot.
    fn publish_snapshot(&mut self) {
        self.stats.io = self.io.stats;
        let rec = self.prof.record().map_or(&[][..], |r| &r.words[..]);
        self.s.snapshots.publish(self.id, self.stats.words().chain(rec.iter().copied()));
    }

    /// One worker loop: harvest completions, move unblocked tasks to delivery,
    /// feed starved tasks, deliver, claim, submit queued reads, and idle with
    /// a growing backoff when nothing moved.
    fn drive(&mut self) -> Result<()> {
        loop {
            if self.s.cancelled() {
                return Err(Error::Aborted);
            }
            let mut progressed = false;
            if self.io.ready() {
                progressed |= self.reap()?;
            }
            if !self.waiting.is_empty() {
                progressed |= self.poll_waiting();
            }
            if !self.starved.is_empty() {
                progressed |= self.feed_starved()?;
            }
            // Decode soon after a read completes, while its pages are still
            // in the cache: completions are harvested between deliveries.
            while let Some(t) = self.ready.pop_front() {
                self.deliver(t)?;
                progressed = true;
                if self.io.ready() {
                    self.reap()?;
                }
            }
            let mut started = 0;
            while started < START_BUDGET && self.can_claim() {
                started += 1;
                let t = self.prof.start(Site::Claim);
                let got = self.claim()?;
                self.prof.stop(Site::Claim, t);
                if !got {
                    break;
                }
                progressed = true;
            }
            let t = self.prof.start(Site::IoSubmit);
            self.io.flush(SUBMIT_BATCH)?;
            self.prof.stop(Site::IoSubmit, t);
            while let Some(t) = self.ready.pop_front() {
                self.deliver(t)?;
                progressed = true;
            }
            if self.exhausted && self.free.len() == self.tasks.len() {
                return Ok(());
            }
            if progressed {
                self.idle = 0;
                continue;
            }
            self.idle += 1;
            // Never idle on unsubmitted reads: other workers may wait for
            // the frames they admit.
            self.io.flush(0)?;
            if self.io.ready() {
                continue;
            }
            if !self.waiting.is_empty() {
                // Frames admitted elsewhere publish without notice: poll.
                let t = self.prof.start(Site::Idle);
                if self.idle < 64 { std::hint::spin_loop() } else { std::thread::yield_now() }
                self.prof.stop(Site::Idle, t);
            } else if self.io.inflight > 0 {
                let t = self.prof.start(Site::IoWait);
                self.io.wait()?;
                self.prof.stop(Site::IoWait, t);
            } else {
                let t = self.prof.start(Site::Idle);
                self.backoff();
                self.prof.stop(Site::Idle, t);
            }
        }
    }

    /// Idles after a round without progress: recomputes the frontier, waits
    /// for the ring's next slot to open, and otherwise spins, yields or
    /// sleeps with a backoff that grows with `idle`.
    fn backoff(&mut self) {
        let s = self.s;
        let t = self.prof.start(Site::Frontier);
        s.progress.scan();
        self.prof.stop(Site::Frontier, t);
        let ring = &s.run.ring;
        let b = s.progress.cursor() / (s.slot_cells * s.schemas);
        if !ring.gate(b) {
            let t = self.prof.start(Site::WaitRing);
            ring.wait_gate(b);
            self.prof.stop(Site::WaitRing, t);
        } else if self.idle < 16 {
            std::hint::spin_loop();
        } else if self.idle < 64 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_micros(20));
        }
    }

    /// Room for a whole claim: task slots, and private frames for its
    /// records at the usual size. No claim while a task waits for private
    /// frames: it then depends only on earlier positions, whose tasks
    /// complete and return their frames.
    fn can_claim(&self) -> bool {
        let positions = (self.s.claim_batch / self.s.schemas) as usize;
        !self.exhausted && self.starved.is_empty() && self.free.len() >= positions && self.private.len() >= positions * self.s.schemas as usize * self.s.normal_pages
    }

    /// Claims and starts a batch of positions. Returns false if none could
    /// be claimed.
    fn claim(&mut self) -> Result<bool> {
        let s = self.s;
        let ring = &s.run.ring;
        let lanes = s.slot_cells * s.schemas;
        for attempt in 0..2 {
            let n = s.progress.cursor();
            let batch = n / lanes;
            let mut gate = if ring.gate(batch) { (batch + 1) * lanes } else { n };
            if gate == n {
                self.stats.counters.output_stalls += 1;
            }
            let planned = match s.source {
                Source::Empty => u64::MAX,
                Source::Ring(r) => r.ready(),
            };
            gate = gate.min(planned);
            match s.progress.claim(self.id, &mut self.outstanding, s.claim_batch, gate.max(n)) {
                Claim::Got(r) => {
                    self.cells.resize((r.end - r.start) as usize, 0);
                    s.run.accesses.fill(r.start, &mut self.cells)?;
                    let mut a = r.start;
                    while a < r.end {
                        self.start(a, ((a - r.start) as usize, s.schemas as usize))?;
                        a += s.schemas;
                    }
                    return Ok(true);
                }
                Claim::Exhausted => {
                    self.exhausted = true;
                    return Ok(false);
                }
                Claim::Blocked if attempt == 0 => {
                    let t = self.prof.start(Site::Frontier);
                    s.progress.scan();
                    self.prof.stop(Site::Frontier, t);
                }
                Claim::Blocked if n >= planned => {
                    self.stats.counters.plan_stalls += 1;
                    let t = self.prof.start(Site::WaitPlan);
                    std::hint::spin_loop();
                    self.prof.stop(Site::WaitPlan, t);
                }
                Claim::Blocked => {
                    self.stats.counters.window_stalls += 1;
                    let t = self.prof.start(Site::WaitWindow);
                    std::hint::spin_loop();
                    self.prof.stop(Site::WaitWindow, t);
                }
            }
        }
        Ok(false)
    }

    /// Starts the position whose first access is `first`; its cells are
    /// `self.cells[at.0..at.0 + at.1]`.
    fn start(&mut self, first: u64, at: (usize, usize)) -> Result<()> {
        let s = self.s;
        let g = s.graph;
        let ti = self.free.pop().expect("claims hold free task slots");
        let tit = self.prof.start(Site::Issue);
        {
            let task = &mut self.tasks[ti as usize];
            task.first = first;
            task.accs.clear();
            self.spare.append(&mut task.inputs);
            task.own = 0;
            task.admit = 0;
            task.waits.clear();
            task.starved = 0;
            task.delivered = false;
            task.issuing = true;
            task.queued = false;
            task.waited = false;
            task.wait_mark = 0;
        }
        let p = first / s.schemas;
        let (j, b) = (p % s.slot_cells, p / s.slot_cells);
        let base = s.run.ring.slot_base(b);
        let mut extras_buf: Vec<(crate::loader::planring::Extra, Vec<u64>)> = Vec::new();
        for k in 0..at.1 {
            let a = first + k as u64;
            let cell = self.cells[at.0 + k];
            // SAFETY: the offset lies inside the slot (the ring's validated layout).
            let out = unsafe { base.add(s.offsets[k] + j as usize * s.cell_bytes[k]) };
            let flags = match s.source {
                Source::Empty => {
                    self.scratch.words.clear();
                    0
                }
                Source::Ring(r) => {
                    let (flags, planned) = r.read(a, &mut self.scratch.words);
                    // The entry's page and frame layout belongs to the cell it
                    // was planned for; a sequence that answers differently now
                    // must stop here, before any frame id is followed.
                    if planned != cell {
                        return Err(crate::loader::Error::Plan(format!(
                            "access {a} loads cell {cell}, but its plan entry was built for cell {planned}: the access sequence is not deterministic"
                        )));
                    }
                    flags
                }
            };
            let same = k > 0 && self.cells[at.0 + k - 1] == cell;
            let task = &mut self.tasks[ti as usize];
            if same {
                let input = task.accs[k - 1].input;
                task.accs.push(Acc { position: a, cell, schema: k as u32, out, batch: b, input });
            } else {
                let input = task.inputs.len() as u8;
                task.accs.push(Acc { position: a, cell, schema: k as u32, out, batch: b, input });
            }
            let entry = Entry { flags, words: &self.scratch.words };
            let loc = if same { None } else { Some(g.source.locate(a, cell)?) };
            let pages = loc.map_or(0, |l| l.page_count);
            let (extras, row) = entry.rest(pages);
            extras_buf.extend(extras.map(|(e, r)| (e, r.to_vec())));
            if let Some(loc) = loc {
                let refs = entry.refs(pages);
                let mut input = self.spare.pop().unwrap_or_else(|| Input { loc: None, file: 0, frames: Vec::new(), kinds: Vec::new(), admits: Vec::new(), issued: Vec::new(), private: Vec::new(), row_hit: false, row_admit: false, row_gen: 0, row_frames: Vec::new() });
                input.loc = Some(loc);
                input.file = loc.file.id as usize;
                input.frames.clear();
                input.frames.resize(pages, std::ptr::null_mut());
                input.kinds.clear();
                input.kinds.resize(pages, PRIVATE);
                input.admits.clear();
                input.admits.resize(pages, (0, 0));
                input.issued.clear();
                input.issued.resize(pages, false);
                debug_assert!(input.private.is_empty());
                input.row_hit = false;
                input.row_admit = false;
                input.row_gen = 0;
                input.row_frames.clear();
                if M::ROWS && let Some((admit, gen_, frames)) = row {
                    debug_assert!(flags & (ROW_USE | ROW_ADMIT) != 0);
                    input.row_gen = gen_;
                    input.row_frames.extend(frames.iter());
                    input.row_admit = admit;
                    input.row_hit = !admit;
                }
                if input.row_hit {
                    input.issued.iter_mut().for_each(|i| *i = true);
                    let f = input.row_frames[0];
                    // Inputs are delivered in access order. A row admitted
                    // by an earlier input of this task becomes ready during
                    // delivery, so it must not gate that same delivery.
                    let local = task.inputs.iter().any(|prior| prior.row_admit
                        && prior.row_gen == input.row_gen
                        && prior.row_frames.first() == Some(&f));
                    if !local && s.gens[f as usize].load(Acquire) != input.row_gen {
                        task.waits.push((f, input.row_gen));
                    }
                } else {
                    for (i, &r) in refs.iter().enumerate() {
                        if r == MISS {
                            continue;
                        }
                        let f = ref_frame(r);
                        input.frames[i] = s.grid.cache(f);
                        if ref_admit(r) {
                            input.kinds[i] = ADMIT;
                            input.admits[i] = (f, ref_gen(r));
                        } else {
                            input.kinds[i] = USE;
                            input.issued[i] = true;
                            self.stats.counters.page_hits += 1;
                            if s.gens[f as usize].load(Acquire) != ref_gen(r) {
                                task.waits.push((f, ref_gen(r)));
                            }
                        }
                    }
                    self.stats.counters.page_uses += pages as u64;
                }
                task.inputs.push(input);
            }
        }
        // Admission reads are issued at once and unconditionally; misses
        // need private frames. With enough of them, one pass reads each run
        // of pages whatever frames it goes to.
        let task = &self.tasks[ti as usize];
        let needed: u32 = task.inputs.iter().map(|i| i.kinds.iter().zip(&i.issued).filter(|(k, d)| **k == PRIVATE && !**d).count() as u32).sum();
        // Behind a task waiting for private frames, later positions of the
        // same claim wait too, in order.
        let private = needed == 0 || (needed as usize <= self.private.len() && self.starved.is_empty());
        for i in 0..self.tasks[ti as usize].inputs.len() {
            self.issue(ti, i, private)?;
        }
        for (e, refs) in extras_buf {
            self.extra(ti, e, &refs)?;
        }
        if !private && needed > 0 {
            self.tasks[ti as usize].starved = needed;
            self.stats.counters.starved += 1;
            self.starved.push_back(ti);
        }
        self.prof.stop(Site::Issue, tit);
        self.tasks[ti as usize].issuing = false;
        self.settle(ti);
        Ok(())
    }

    /// Issues the reads of input `i` of task `t`: runs of consecutive pages
    /// still to read, cache frames always, private frames when `private`. A
    /// run is cut at 1024 pages, the read's iovec capacity.
    fn issue(&mut self, t: u32, i: usize, private: bool) -> Result<()> {
        let s = self.s;
        let page = s.page;
        let n = self.tasks[t as usize].inputs[i].frames.len();
        let mut k = 0;
        while k < n {
            let input = &self.tasks[t as usize].inputs[i];
            let can = |kind: u8, issued: bool| !issued && (kind == ADMIT || (kind == PRIVATE && private));
            if !can(input.kinds[k], input.issued[k]) {
                k += 1;
                continue;
            }
            let k0 = k;
            self.scratch.run.clear();
            self.scratch.publish.clear();
            while k < n && (k - k0) < 1024 {
                let input = &self.tasks[t as usize].inputs[i];
                if !can(input.kinds[k], input.issued[k]) {
                    break;
                }
                if input.kinds[k] == PRIVATE {
                    let Some(f) = self.private.pop() else { break };
                    let ptr = self.private_frame(f);
                    let input = &mut self.tasks[t as usize].inputs[i];
                    input.private.push(f);
                    input.frames[k] = ptr;
                } else {
                    self.scratch.publish.push(input.admits[k]);
                }
                let input = &mut self.tasks[t as usize].inputs[i];
                input.issued[k] = true;
                self.scratch.run.push(input.frames[k]);
                k += 1;
            }
            if self.scratch.run.is_empty() {
                break;
            }
            let input = &self.tasks[t as usize].inputs[i];
            let loc = input.loc.expect("inputs with pages are located");
            let offset = loc.page_offset + (k0 * page) as u64;
            let need = (self.scratch.run.len() * page).min(loc.file.len.saturating_sub(offset) as usize);
            self.stats.counters.page_reads += self.scratch.run.len() as u64;
            self.stats.counters.admitted += self.scratch.publish.len() as u64;
            let mark = self.prof.mark(Site::IoLatency);
            let run = std::mem::take(&mut self.scratch.run);
            let publish = std::mem::take(&mut self.scratch.publish);
            let (fd, file) = (s.fds[input.file], input.file as u32);
            self.tasks[t as usize].own += 1;
            let (slot, now) = self.io.read(fd, file, offset, &run, page, need, t, true, &publish, mark)?;
            self.scratch.run = run;
            self.scratch.publish = publish;
            if let Some(result) = now {
                self.on_done(Done { slot, result })?;
            }
        }
        Ok(())
    }

    /// Issues an extra admission read of task `t`.
    fn extra(&mut self, t: u32, e: crate::loader::planring::Extra, refs: &[u64]) -> Result<()> {
        let s = self.s;
        let page = s.page;
        self.scratch.run.clear();
        self.scratch.publish.clear();
        for &r in refs {
            self.scratch.run.push(self.cache_frame(ref_frame(r)));
            self.scratch.publish.push((ref_frame(r), ref_gen(r)));
        }
        let file = &s.graph.files[e.file as usize];
        let need = (refs.len() * page).min(file.len.saturating_sub(e.offset) as usize);
        self.stats.counters.page_reads += refs.len() as u64;
        self.stats.counters.admitted += refs.len() as u64;
        self.stats.counters.block_reads += 1;
        self.tasks[t as usize].admit += 1;
        let mark = self.prof.mark(Site::IoLatency);
        let run = std::mem::take(&mut self.scratch.run);
        let publish = std::mem::take(&mut self.scratch.publish);
        let (slot, now) = self.io.read(s.fds[e.file as usize], e.file, e.offset, &run, page, need, t, false, &publish, mark)?;
        self.scratch.run = run;
        self.scratch.publish = publish;
        if let Some(result) = now {
            self.on_done(Done { slot, result })?;
        }
        Ok(())
    }

    /// Harvests completions: per read, publish its frames and count it off
    /// its task.
    fn reap(&mut self) -> Result<bool> {
        let t = self.prof.start(Site::IoReap);
        self.done.clear();
        self.io.harvest(&mut self.done);
        let done = std::mem::take(&mut self.done);
        let any = !done.is_empty();
        for &d in &done {
            self.on_done(Done { slot: d.slot, result: d.result })?;
        }
        self.done = done;
        self.prof.stop(Site::IoReap, t);
        Ok(any)
    }

    /// Handles one completion: checks the read, publishes the frames it
    /// admits, and counts it off its task.
    fn on_done(&mut self, d: Done) -> Result<()> {
        let s = self.s;
        let slot = self.io.slot(d.slot);
        check_read(d.result, slot.need as usize)?;
        for &(f, g) in &slot.publish {
            s.gens[f as usize].store(g, Release);
        }
        let (t, own, mark) = (slot.task, slot.own, slot.issued);
        self.io.recycle(d.slot);
        self.prof.since_mark(Site::IoLatency, mark);
        let task = &mut self.tasks[t as usize];
        if own {
            task.own -= 1;
            if task.own == 0 {
                self.settle(t);
            }
        } else {
            task.admit -= 1;
            if task.admit == 0 && task.delivered {
                self.complete(t);
            }
        }
        Ok(())
    }

    /// Moves task `t` to the ready queue once its own reads are done, its
    /// misses issued and every frame it uses is ready; else to the waiting
    /// list if only frames are missing.
    fn settle(&mut self, t: u32) {
        let s = self.s;
        let task = &mut self.tasks[t as usize];
        if task.own > 0 || task.starved > 0 || task.delivered || task.issuing || task.queued {
            return;
        }
        task.queued = true;
        task.waits.retain(|&(f, g)| s.gens[f as usize].load(Acquire) != g);
        if task.waits.is_empty() {
            prefetch(task);
            self.ready.push_back(t);
        } else {
            task.waited = true;
            task.wait_mark = self.prof.mark(Site::WaitGen);
            self.stats.counters.waited += 1;
            self.waiting.push(t);
        }
    }

    /// Moves the tasks whose waited frames became ready to the ready queue;
    /// returns whether any moved.
    fn poll_waiting(&mut self) -> bool {
        let s = self.s;
        let mut moved = false;
        let mut i = 0;
        while i < self.waiting.len() {
            let t = self.waiting[i];
            let task = &mut self.tasks[t as usize];
            task.waits.retain(|&(f, g)| s.gens[f as usize].load(Acquire) != g);
            if task.waits.is_empty() {
                self.waiting.swap_remove(i);
                self.prof.since_mark(Site::WaitGen, task.wait_mark);
                prefetch(task);
                self.ready.push_back(t);
                moved = true;
            } else {
                i += 1;
            }
        }
        moved
    }

    /// Issues the private reads of starved tasks, in claim order, as private
    /// frames free up.
    fn feed_starved(&mut self) -> Result<bool> {
        let mut any = false;
        while let Some(&t) = self.starved.front() {
            let need = self.tasks[t as usize].starved as usize;
            if need > self.private.len() {
                break;
            }
            self.starved.pop_front();
            self.tasks[t as usize].starved = 0;
            self.tasks[t as usize].issuing = true;
            for i in 0..self.tasks[t as usize].inputs.len() {
                self.issue(t, i, true)?;
            }
            self.tasks[t as usize].issuing = false;
            self.settle(t);
            any = true;
        }
        Ok(any)
    }

    /// Completes task `t`: counts its accesses off the frontier, frees its
    /// slot and publishes statistics periodically.
    fn complete(&mut self, t: u32) {
        let s = self.s;
        let tm = self.prof.start(Site::Complete);
        let task = &self.tasks[t as usize];
        s.progress.complete(self.id, &mut self.outstanding, task.first, task.accs.len() as u64);
        self.free.push(t);
        self.completed += 1;
        self.prof.stop(Site::Complete, tm);
        if self.completed.is_multiple_of(PUBLISH_EVERY) {
            self.publish_snapshot();
        }
    }

    /// Decodes each input of task `t` and delivers its accesses.
    fn deliver(&mut self, t: u32) -> Result<()> {
        let mut task = std::mem::replace(&mut self.tasks[t as usize], Task { first: 0, accs: Vec::new(), inputs: Vec::new(), own: 0, admit: 0, waits: Vec::new(), starved: 0, delivered: false, issuing: false, queued: false, waited: false, wait_mark: 0 });
        let r = (|| -> Result<()> {
            for i in 0..task.inputs.len() {
                self.deliver_input(&task, i)?;
                let input = &mut task.inputs[i];
                for f in input.private.drain(..) {
                    self.private.push(f);
                }
            }
            Ok(())
        })();
        task.delivered = true;
        task.queued = false;
        let admit = task.admit;
        self.tasks[t as usize] = task;
        r?;
        if admit == 0 {
            self.complete(t);
        }
        Ok(())
    }

    /// One input: the row (cached, or decoded and possibly admitted), then
    /// every access of the task that loads it.
    fn deliver_input(&mut self, task: &Task<'a>, i: usize) -> Result<()> {
        let s = self.s;
        let g = s.graph;
        let source = &g.source;
        let input = &task.inputs[i];
        let loc = input.loc.expect("inputs are located");
        let chunk = loc.chunk_index;
        let stored = loc.chunk.dtype;
        let width = stored.width();
        let sc = &mut self.scratch;
        let stats = &mut self.stats;
        let prof = &mut self.prof;
        sc.pages.clear();
        // The cached row as (dense, elements), in `row_ids`/`row_values`.
        let mut image: Option<(bool, usize)> = None;
        let mut parts = None;
        let cached_dense = g.cached_dense(chunk, loc.dense, loc.nnz);
        if M::ROWS && input.row_hit {
            let tm = prof.start(Site::RowGather);
            sc.row_ptrs.clear();
            sc.row_ptrs.extend(input.row_frames.iter().map(|&f| s.grid.cache(f)));
            let n = if cached_dense {
                let n = loc.chunk.meta().n_dims() as usize;
                // SAFETY: the frames hold the dense row the plan admitted (its
                // generation was observed); the row scratch holds a value per
                // gene of any chunk.
                unsafe { rows::gather_dense(&sc.row_ptrs, n, sc.out.row_values.as_mut_ptr().cast(), width, g.page_size) };
                n
            } else {
                // SAFETY: as above, for a sparse row of at most `max_nnz`
                // elements, which the row scratch holds.
                unsafe { rows::gather_sparse(&sc.row_ptrs, sc.out.row_ids.as_mut_ptr(), sc.out.row_values.as_mut_ptr().cast(), width, g.page_size) }
            };
            image = Some((cached_dense, n));
            prof.stop(Site::RowGather, tm);
            stats.counters.row_hits += 1;
            stats.counters.row_gather_bytes += rows::payload(cached_dense, n, width) as u64;
        } else if loc.len > 0 {
            for &p in &input.frames {
                // SAFETY: every page is complete (read by this task or ready
                // by its generation) and stays so until the task completes;
                // the slices are dropped before it does.
                sc.pages.push(unsafe { std::slice::from_raw_parts(p, g.page_size) });
            }
            let tm = prof.start(Site::Decode);
            let record = PagedRecord { pages: &sc.pages, page_size: g.page_size, start: loc.start_in_page, len: loc.len };
            // The chunk's tables were derived from its own metadata.
            let p = crate::scans::decode::decode_parts_pages_prechecked(&loc.chunk.meta(), &loc.chunk.tables, record, IdSpace::Coded, &mut sc.decode)?;
            prof.stop(Site::Decode, tm);
            if p.nnz() != loc.nnz {
                return Err(crate::Error::CorruptRecord("record nnz differs from metadata").into());
            }
            stats.counters.decoded_rows += 1;
            if M::ROWS && input.row_admit {
                let tm = prof.start(Site::RowAdmit);
                let ops = {
                    let ops = at!(g.admit_ops, dtype_index(stored));
                    debug_assert!(ops.is_some());
                    // SAFETY: the graph builds admission kernels for every stored type.
                    unsafe { ops.as_ref().unwrap_unchecked() }
                };
                let n = if cached_dense {
                    let n = p.elements();
                    // SAFETY: the row scratch holds a word per gene of any chunk.
                    p.expand_values_with(&loc.chunk.tables, unsafe { std::slice::from_raw_parts_mut(sc.out.row_values.as_mut_ptr().cast::<u8>(), n * width) })?;
                    n
                } else {
                    (ops.cast_dict)(&p, &mut sc.out.dict)?;
                    let modes = loc.chunk.modes.map_or(&[][..], |m| source.mode_words(m));
                    if p.dense {
                        let raw = match p.values {
                            Values::Raw(v) => v,
                            _ => {
                                // SAFETY: scratch holds one word per source gene.
                                let buf = unsafe { std::slice::from_raw_parts_mut(sc.out.stored.as_mut_ptr().cast::<u8>(), sc.out.stored.len() * 8) };
                                p.expand_values_with(&loc.chunk.tables, buf)?;
                                &buf[..p.elements() * width]
                            }
                        };
                        ops.dense_as_csr(raw, source.columns(loc.chunk.columns), &mut sc.out.row_ids, sc.out.row_values.as_mut_ptr().cast())
                    } else {
                        ops.csr(&p, source.columns(loc.chunk.columns), &sc.out.dict, modes, &mut sc.out.row_ids, sc.out.row_values.as_mut_ptr().cast())?
                    }
                };
                sc.row_ptrs.clear();
                sc.row_ptrs.extend(input.row_frames.iter().map(|&f| s.grid.cache(f)));
                if sc.row_ptrs.len() != rows::frames(cached_dense, n, width, g.page_size) {
                    return Err(Error::Plan("a planned row's frames do not fit the row".into()));
                }
                // SAFETY: the plan gives this access the row's frames, which
                // no reader touches before the generation below.
                unsafe {
                    let vals = std::slice::from_raw_parts(sc.out.row_values.as_ptr().cast::<u8>(), n * width);
                    if cached_dense { rows::write_dense(&sc.row_ptrs, vals, width, g.page_size) } else { rows::write_sparse(&sc.row_ptrs, &sc.out.row_ids[..n], vals, width, g.page_size) }
                }
                for &f in &input.row_frames {
                    s.gens[f as usize].store(input.row_gen, Release);
                }
                stats.counters.rows_admitted += 1;
                stats.counters.row_admit_bytes += rows::payload(cached_dense, n, width) as u64;
                image = Some((cached_dense, n));
                prof.stop(Site::RowAdmit, tm);
            } else {
                parts = Some(p);
            }
        }
        for acc in task.accs.iter().filter(|a| a.input as usize == i) {
            deliver_access::<P>(s, g, &mut sc.out, stats, prof, input, acc, loc, image, parts.as_ref())?;
        }
        Ok(())
    }
}

/// Starts loading what decoding a ready task touches first: its chunks'
/// decode tables.
fn prefetch(task: &Task<'_>) {
    for input in &task.inputs {
        if let Some(loc) = input.loc
            && !input.row_hit
        {
            loc.chunk.tables.prefetch();
        }
    }
}

/// Converts one access's row (cached, or decoded) to its schema's output.
#[allow(clippy::too_many_arguments)]
fn deliver_access<P: Profiler>(s: &Shared<'_>, g: &crate::loader::schema::Graph, sc: &mut OutScratch, stats: &mut RunStats, prof: &mut P, input: &Input<'_>, acc: &Acc, loc: Location<'_>, image: Option<(bool, usize)>, parts: Option<&Parts<'_>>) -> Result<()> {
    let k = acc.schema as usize;
    let schema = at!(g.schemas, k);
    let tables = at!(g.tables, k);
    let panel = at!(g.panels, tables.panel);
    let chunk = loc.chunk_index;
    let stored = loc.chunk.dtype;
    let width = stored.width();
    stats.counters.accesses += 1;
    stats.outputs[k].accesses += 1;
    if image.is_some() && input.row_hit {
        stats.outputs[k].row_hits += 1;
    } else if parts.is_some() {
        stats.outputs[k].decoded_rows += 1;
    }
    let ops = at!(tables.ops, dtype_index(stored)).as_ref();
    let chunk_map = panel.map(chunk);
    let cached;
    let (row, map, dict, modes): (Option<&Parts<'_>>, &[u32], &[u64], &[u64]) = match (image, parts, ops) {
        (Some((dense, n)), _, Some(_)) => {
            // SAFETY: the row scratch holds the row's `n` elements of the
            // stored width (and its ids for a sparse row).
            let values = unsafe { std::slice::from_raw_parts(sc.row_values.as_ptr().cast::<u8>(), n * width) };
            let (ids, map): (&[u32], Option<&[u32]>) = if dense {
                (crate::scans::decode::iota(n).ok_or_else(|| config("dense row longer than the id table"))?, chunk_map)
            } else {
                (&sc.row_ids[..n], Some(&tables.std_to_out[..]))
            };
            match map {
                Some(map) => {
                    cached = Parts { ids, values: Values::Raw(values), nnz: n, dense };
                    (Some(&cached), map, &[][..], &[][..])
                }
                None => (None, &[][..], &[][..], &[][..]),
            }
        }
        (None, Some(p), Some(ops)) if chunk_map.is_some() => {
            (ops.cast_dict)(p, &mut sc.dict)?;
            // SAFETY: the arm's guard checked `chunk_map`.
            (Some(p), unsafe { chunk_map.unwrap_unchecked() }, &sc.dict[..], g.modes(k, chunk))
        }
        _ => (None, &[][..], &[][..], &[][..]),
    };
    // Narrow outputs scatter through the sparse form of the map (cached
    // sparse rows hold standard ids; decoded and dense rows coded ids).
    let sparse = match image {
        Some((false, _)) => tables.std_sparse(),
        _ => panel.sparse(chunk),
    };
    let out_dim = schema.out_dim();
    let default = schema.default_bits();
    let whole = image.is_some();
    let dense_record = row.map_or(loc.dense, |p| p.dense);
    let deliver_dense = g.dense_delivery(k, chunk, dense_record, loc.value_encoding, whole);
    match schema.process_fn() {
        None => {
            let tm = prof.start(Site::Map);
            // A row that fits the first-level cache is assembled in scratch
            // and streamed to the ring, whose lines are then written once
            // without being read.
            let bytes = out_dim * schema.dtype().width();
            let staged = ops.is_some_and(|o| o.streams) && streamable(acc.out, bytes);
            let target = if staged { sc.dense.as_mut_ptr().cast::<u8>() } else { acc.out };
            match (row, ops, sparse) {
                (Some(p), Some(ops), Some(sm)) => ops.dense_sparse(p, &sm, dict, modes, default, chunk_map, target, out_dim)?,
                (Some(p), Some(ops), None) => ops.dense(p, map, dict, modes, default, chunk_map, target, out_dim)?,
                (None, Some(ops), _) => ops.empty_dense(default, chunk_map, target, out_dim),
                _ => fill(target, out_dim, schema.dtype().width(), default),
            }
            if staged {
                // SAFETY: scratch holds the row; the ring cell is this
                // access's, 64-byte aligned and `bytes` long (`streamable`).
                unsafe { stream_copy(target, acc.out, bytes) };
            }
            stats.counters.mapped_rows += 1;
            prof.stop_at(Site::Map, k, tm);
        }
        Some(process) => {
            let tm = prof.start(Site::Map);
            let partial = !deliver_dense && !whole && process.partial_rows() && *at!(panel.total, chunk);
            let partial_row;
            let call = match (row, ops) {
                (Some(p), _) if whole && stored == schema.dtype() && deliver_dense == p.dense
                    && if p.dense { *at!(panel.identity, chunk) && p.elements() == out_dim } else { tables.std_identity } => {
                    // SAFETY: cached rows hold native raw values.
                    let Values::Raw(values) = p.values else { unsafe { std::hint::unreachable_unchecked() } };
                    if p.dense { Row::Dense(values.as_ptr()) } else { Row::Csr { nnz: p.elements(), columns: p.ids.as_ptr(), values: values.as_ptr() } }
                }
                (Some(p), Some(ops)) if deliver_dense => {
                    match sparse {
                        Some(sm) => ops.dense_sparse(p, &sm, dict, modes, default, chunk_map, sc.dense.as_mut_ptr().cast(), out_dim)?,
                        None => ops.dense(p, map, dict, modes, default, chunk_map, sc.dense.as_mut_ptr().cast(), out_dim)?,
                    }
                    stats.counters.mapped_rows += 1;
                    Row::Dense(sc.dense.as_ptr().cast())
                }
                (Some(p), Some(ops)) if p.dense => {
                    // A dense record as the CSR row of its non-zeros.
                    let values: &[u8] = match p.values {
                        Values::Raw(v) => v,
                        _ => {
                            // SAFETY: the stored scratch holds a value word per gene of any chunk.
                            let buf = unsafe { std::slice::from_raw_parts_mut(sc.stored.as_mut_ptr().cast::<u8>(), sc.stored.len() * 8) };
                            p.expand_values_with(&loc.chunk.tables, buf)?;
                            &buf[..p.elements() * width]
                        }
                    };
                    let n = ops.dense_as_csr(values, map, &mut sc.columns, sc.values.as_mut_ptr().cast());
                    stats.counters.mapped_rows += 1;
                    Row::Csr { nnz: n, columns: sc.columns.as_ptr(), values: sc.values.as_ptr().cast() }
                }
                (Some(p), Some(ops)) if partial => {
                    partial_row = PartialRow { parts: *p, ops, map, dict, modes };
                    Row::Partial { nnz: p.elements(), row: (&partial_row as *const PartialRow).cast(), resolve: resolve as ResolveFn }
                }
                (Some(p), Some(ops)) => {
                    let n = ops.csr(p, map, dict, modes, &mut sc.columns, sc.values.as_mut_ptr().cast())?;
                    stats.counters.mapped_rows += 1;
                    Row::Csr { nnz: n, columns: sc.columns.as_ptr(), values: sc.values.as_ptr().cast() }
                }
                (None, Some(ops)) if deliver_dense => {
                    ops.empty_dense(default, chunk_map, sc.dense.as_mut_ptr().cast(), out_dim);
                    Row::Dense(sc.dense.as_ptr().cast())
                }
                _ if deliver_dense => {
                    fill(sc.dense.as_mut_ptr().cast(), out_dim, schema.dtype().width(), default);
                    Row::Dense(sc.dense.as_ptr().cast())
                }
                _ => Row::Csr { nnz: 0, columns: sc.columns.as_ptr(), values: sc.values.as_ptr().cast() },
            };
            prof.stop_at(Site::Map, k, tm);
            let tm = prof.start(Site::Process);
            let called_partial = matches!(call, Row::Partial { .. });
            // SAFETY: this worker exclusively uses its scratch allocation.
            let scratch = unsafe { std::slice::from_raw_parts_mut(sc.process.ptr(), schema.scratch_bytes()) };
            process.call(acc.position / s.schemas, acc.cell as u64, k as u32, schema.dtype(), out_dim, call, acc.out, scratch)?;
            stats.counters.process_calls += 1;
            stats.outputs[k].process_calls += 1;
            if prof.on(Site::Process) {
                let elements = if loc.dense { out_dim } else { loc.nnz };
                let band = PROCESS_BOUNDS.partition_point(|&bound| (bound as usize) < elements).min(PROCESS_BOUNDS.len() - 1);
                prof.stop_at(Site::Process, (k * PROCESS_BOUNDS.len() + band) * 2 + called_partial as usize, tm);
            }
        }
    }
    stats.counters.written_bytes += s.cell_bytes[k] as u64;
    s.run.ring.complete(acc.batch, 1);
    Ok(())
}

impl<P: Profiler, M: Mode> Drop for Worker<'_, P, M> {
    fn drop(&mut self) {
        // A failed drain leaves reads that may still write into frames: the
        // run must not unmap them.
        if !self.io.drain() {
            self.s.poisoned.store(true, Release);
        }
    }
}

/// Writes `n` copies of a `width`-byte value.
fn fill(out: *mut u8, n: usize, width: usize, value: u64) {
    if value == 0 {
        // SAFETY: the caller passes room for `n` values.
        unsafe { std::ptr::write_bytes(out, 0, n * width) };
        return;
    }
    let bytes = value.to_le_bytes();
    for i in 0..n {
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(i * width), width) };
    }
}

/// Rows staged in scratch before the ring: up to this many bytes (they stay
/// in the first-level cache), in whole 64-byte lines.
const STAGED_ROW: usize = 32 << 10;

/// Whether a built-in row can be staged in scratch and streamed to the ring:
/// 256 B to [`STAGED_ROW`], whole cache lines, into a 64-byte aligned cell.
fn streamable(out: *mut u8, bytes: usize) -> bool {
    (256..=STAGED_ROW).contains(&bytes) && bytes.is_multiple_of(64) && (out as usize).is_multiple_of(64)
}

/// Copies `len` bytes (whole lines) to 64-byte aligned `dst` with
/// non-temporal stores.
///
/// # Safety
/// Both ranges hold `len` bytes; `streamable(dst, len)` holds and the CPU
/// has AVX-512 (`RowOps::streams`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unused_unsafe)]
unsafe fn stream_copy(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::*;
    let mut at = 0;
    while at < len {
        // SAFETY: as documented.
        unsafe { _mm512_stream_si512(dst.add(at).cast(), _mm512_loadu_si512(src.add(at).cast())) };
        at += 64;
    }
    // SAFETY: SSE2 is part of the x86-64 baseline.
    unsafe { _mm_sfence() };
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn stream_copy(src: *const u8, dst: *mut u8, len: usize) {
    // SAFETY: as documented.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
}

/// Turns a completion result into an error: a negative errno, or fewer bytes
/// than the read needs.
fn check_read(result: i32, need: usize) -> Result<()> {
    if result < 0 {
        return Err(std::io::Error::from_raw_os_error(-result).into());
    }
    if (result as usize) < need {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short input read").into());
    }
    Ok(())
}
