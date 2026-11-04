//! The loader: one data source and the schemas that output its cells, bound
//! per run to one access sequence, one progress frontier and one ring.
use super::accesses::Accesses;
use super::config::{LoaderConfig, RunStats};
use super::error::config;
use super::schema::{Graph, OutputSchema};
use super::{BatchLease, CellId, DataSource, Error, Result, Ring, RingLayout};
use crate::opt::{CostModel, Plan, PlanOptions, PlanSummary};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering::*},
};

/// One data source and the schemas that output its cells, ready to serve runs.
///
/// The tables of a loader are immutable once built: the per-chunk output maps
/// and the conversion kernels are composed then, so [`Loader::prepare`] only
/// binds an access sequence and a ring. It serves one run at a time (see
/// [`Error::Busy`]) and the run borrows it, so the run must drop before the
/// loader can prepare the next run.
pub struct Loader {
    pub(crate) graph: Graph,
    pub(crate) config: LoaderConfig,
    running: AtomicBool,
}
impl Loader {
    /// Binds `schemas` to `source`. Rejects a zero worker count, in-flight
    /// depth or claim batch, a window of zero, an in-flight depth past
    /// `u32::MAX / 4` and CPU indices at or beyond `CPU_SETSIZE`; the claim
    /// batch is clamped to one slot's accesses (`slot_cells × schemas`), and a
    /// claim never crosses into the next slot anyway, since the open slot's
    /// gate ends it. `Graph::new` also rejects an empty schema list, duplicate
    /// schema ids, output dims outside the vocabulary, a built-in cell size
    /// that differs from its dense row, `slot_cells` outside `1..2^32`, and an
    /// input that changed since registration.
    pub fn new(source: Arc<DataSource>, schemas: Vec<OutputSchema>, config: LoaderConfig) -> Result<Self> {
        if config.cpus.iter().any(|&cpu| cpu >= libc::CPU_SETSIZE as usize) {
            return Err(config_error());
        }
        if config.window == Some(0)
            || config.max_inflight_per_worker > u32::MAX as usize / 4
            || config.workers == 0
            || config.max_inflight_per_worker == 0
            || config.claim_batch == 0
        {
            return Err(config_error());
        }
        let graph = Graph::new(source, schemas, config.slot_cells)?;
        let mut config = config;
        config.claim_batch = config.claim_batch.min(graph.lanes_per_slot() as usize);
        Ok(Self { graph, config, running: AtomicBool::new(false) })
    }
    /// The inputs the loader reads: every matrix, its cell ids and page map.
    pub fn source(&self) -> &Arc<DataSource> {
        &self.graph.source
    }
    /// The output schemas, in loader order: schema `k` receives access
    /// `position × S + k` and writes region `k` of the slot.
    pub fn schemas(&self) -> &[OutputSchema] {
        &self.graph.schemas
    }
    /// Accesses per slot: every schema of every position.
    pub fn accesses_per_slot(&self) -> u64 {
        self.graph.lanes_per_slot()
    }
    /// The execution window in slots: the configured one, or enough slots for
    /// every access in flight. It is the default for runs without a plan and
    /// for the reuse gap of plans built by [`Self::plan_options`].
    pub fn window(&self) -> u64 {
        self.config.window.unwrap_or_else(|| crate::opt::default_distance(self.config.workers, self.config.max_inflight_per_worker).div_ceil(self.accesses_per_slot()).max(1))
    }
    /// A span of slots in accesses.
    pub(crate) fn gap_lanes(&self, slots: u64) -> u64 {
        slots.saturating_mul(self.accesses_per_slot())
    }
    /// The ring layout of this loader's outputs.
    pub fn ring_layout(&self, slots: u32, world: u32) -> RingLayout {
        RingLayout { slots, slot_cells: self.graph.slot_cells as u32, world, cell_bytes: self.graph.schemas.iter().map(|s| s.cell_bytes() as u32).collect() }
    }

    /// Physical cores the workers run on: the cores among the pinned CPUs
    /// (or the process's allowed CPUs), at most one per worker. A cost
    /// model's CPU capacity.
    pub fn cores(&self) -> f64 {
        crate::runtime::physical_cores(&self.config.cpus).min(self.config.workers).max(1) as f64
    }

    /// Plan options with this loader's window as the reuse gap and the
    /// default cost model ([`CostModel::defaults`] with
    /// [`crate::opt::defaults::Tier::Nvme`]) on its cores, to be replaced by
    /// measurements on the target machine (pycellpack's `costs`). The
    /// per-schema builtin flags follow schema order.
    pub fn plan_options(&self, cache_bytes: usize) -> PlanOptions {
        let mut o = PlanOptions::new(cache_bytes, self.window());
        let builtin: Vec<bool> = self.graph.schemas.iter().map(|s| s.is_builtin()).collect();
        o.cost = Some(CostModel::defaults(self.cores(), crate::opt::Tier::Nvme, &builtin));
        o
    }

    /// Compiles a cache plan for a stored sequence of accesses (see
    /// [`crate::opt`]).
    pub fn compile(&self, accesses: &[CellId], options: &PlanOptions) -> Result<Plan> {
        crate::opt::compile::compile(&self.graph, accesses, options)
    }

    /// Replays a plan serially against this loader's inputs and schemas.
    pub fn replay(&self, plan: &Plan, accesses: &[CellId]) -> Result<PlanSummary> {
        plan.replay(&self.graph, accesses, None)
    }

    /// Prices the same actions with measured costs, without changing the
    /// plan or recompiling its cache decisions.
    pub fn replay_with_cost(&self, plan: &Plan, accesses: &[CellId], cost: &CostModel) -> Result<PlanSummary> {
        plan.replay(&self.graph, accesses, Some(cost))
    }

    /// Compiles a stored sequence with the online compiler (a finite
    /// horizon, chunked freezing) into a plan verified like an offline one:
    /// to inspect or compare what online runs do.
    pub fn compile_online(&self, accesses: &[CellId], options: &PlanOptions) -> Result<(Plan, crate::opt::OnlineStats)> {
        crate::opt::compile::collect(&self.graph, accesses, options)
    }

    /// Compiles a sequence online without loading it, for the statistics of
    /// the plan an online run would follow (its modeled time among them).
    pub fn plan_online(&self, accesses: &dyn Accesses, options: &PlanOptions) -> Result<crate::opt::OnlineStats> {
        crate::opt::compile::dry(&self.graph, accesses, options)
    }

    /// Checks plan options for this loader — including that the configured
    /// window does not exceed their reuse gap — and returns them as the run's
    /// planning. The compiler's threads run beside the workers
    /// (`options.threads` is used as given).
    pub fn online(&self, options: PlanOptions) -> Result<Planning> {
        crate::opt::compile::check(&self.graph, &options)?;
        self.runtime_window(options.gap)?;
        Ok(Planning::Online(Box::new(options)))
    }

    /// The run's window in accesses under a plan's reuse gap (slots): the
    /// configured window, which must not exceed the gap, or the gap.
    pub(crate) fn runtime_window(&self, gap: u64) -> Result<u64> {
        let w = self.config.window.unwrap_or(gap);
        if w > gap {
            return Err(config("the execution window exceeds the plan's reuse gap"));
        }
        Ok(self.gap_lanes(w))
    }

    /// Validates the input files, the ring's layout, the access count and the
    /// plan binding, then opens the ring's slots under the next run generation
    /// and returns the run. Fails with [`Error::Busy`] while another run holds
    /// the loader or the ring has a producer, and rejects a ring that was
    /// aborted or still holds ready or leased slots.
    pub fn prepare<'a>(&'a self, accesses: &'a dyn Accesses, ring: Arc<Ring>, planning: Planning) -> Result<Run<'a>> {
        if self.running.compare_exchange(false, true, AcqRel, Acquire).is_err() {
            return Err(Error::Busy);
        }
        let result = (|| {
            use std::os::unix::fs::MetadataExt;
            for f in &self.graph.files {
                let m = f.file.metadata()?;
                if (m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec()) != f.identity {
                    return Err(config("input file changed after data source construction"));
                }
            }
            let layout = ring.layout();
            if layout.slot_cells as u64 != self.graph.slot_cells || layout.cell_bytes != self.ring_layout(1, 1).cell_bytes {
                return Err(config("the ring's slot cells or cell bytes differ from the loader's schemas"));
            }
            let schemas = self.graph.schemas.len() as u64;
            if !accesses.len().is_multiple_of(schemas) {
                return Err(config("the access count is not a multiple of the schema count"));
            }
            let window = match &planning {
                Planning::Fixed(plan) => {
                    let owned;
                    let slice = match accesses.as_slice() {
                        Some(s) => s,
                        None => {
                            let n = usize::try_from(accesses.len()).ok().filter(|&n| n < u32::MAX as usize).ok_or_else(|| config("offline plans cover fewer than 2^32 accesses"))?;
                            let mut v = vec![0; n];
                            accesses.fill(0, &mut v)?;
                            owned = v;
                            &owned[..]
                        }
                    };
                    plan.verify(&self.graph, slice)?;
                    let w = self.config.window.map_or(plan.distance(), |w| self.gap_lanes(w));
                    if w > plan.distance() {
                        return Err(config("the execution window exceeds the plan's reuse gap"));
                    }
                    w
                }
                Planning::Online(options) => {
                    crate::opt::compile::check(&self.graph, options)?;
                    self.runtime_window(options.gap)?
                }
                Planning::None => {
                    let cells = self.graph.source.n_cells();
                    if let Some(slice) = accesses.as_slice()
                        && let Some((at, &cell)) = slice.iter().enumerate().find(|(_, cell)| **cell as u64 >= cells)
                    {
                        return Err(Error::Access { position: at as u64, cell: cell as u64 });
                    }
                    self.gap_lanes(self.window())
                }
            };
            let generation = ring.generation().checked_add(1).ok_or_else(|| config("run generation exhausted"))?;
            ring.begin_generation(accesses.len() / schemas, generation)?;
            ring.join_producer(generation);
            Ok(Run {
                loader: self,
                accesses,
                ring,
                generation,
                window,
                planning,
                state: AtomicU32::new(0),
                live: crate::loader::exec::Live::new(),
            })
        })();
        if result.is_err() {
            self.running.store(false, Release);
        }
        result
    }
}
/// The single message for every rejected [`LoaderConfig`] value; the fields
/// are checked together because any of them makes a run unrunnable.
fn config_error() -> Error {
    config("loaders need positive workers, in-flight depth and claim batch, and a nonzero window")
}

/// How a run caches.
pub enum Planning {
    /// No plan: every access reads its pages privately.
    None,
    /// An offline plan, bound to the stored access sequence.
    Fixed(Arc<Plan>),
    /// Compiled online by a thread of the run, ahead of the workers.
    Online(Box<PlanOptions>),
}

impl From<Option<Arc<Plan>>> for Planning {
    fn from(plan: Option<Arc<Plan>>) -> Self {
        plan.map_or(Planning::None, Planning::Fixed)
    }
}

/// A prepared run: the loader bound to an access sequence, a ring and its
/// planning mode.
///
/// The run holds the ring's generation, the token its leases and waits
/// check, and keeps the loader's busy flag set. It is the producer handle:
/// [`Run::run`] executes on the calling thread while consumers take batches
/// through [`Run::wait`] from other threads. Dropping an unstarted run
/// cancels it; dropping any run frees the loader.
pub struct Run<'a> {
    pub(crate) loader: &'a Loader,
    pub(crate) accesses: &'a dyn Accesses,
    pub(crate) ring: Arc<Ring>,
    generation: u64,
    /// The execution window in accesses.
    pub(crate) window: u64,
    pub(crate) planning: Planning,
    state: AtomicU32,
    pub(crate) live: crate::loader::exec::Live,
}
impl Run<'_> {
    /// The ring generation this run opened its slots under; leases are tied
    /// to it, so batches of one run are never confused with another's.
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// The ring the run produces into.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// Slots of the run.
    pub fn batches(&self) -> u64 {
        (self.accesses.len() / self.loader.graph.schemas.len() as u64).div_ceil(self.loader.graph.slot_cells)
    }
    /// The workers' statistics so far, while the run executes: summed per
    /// worker as last published (every 1024 completed positions), with the
    /// elapsed time of the live execution; profiled runs add the sites they
    /// record.
    pub fn snapshot(&self) -> RunStats {
        self.live.snapshot(&self.loader.graph)
    }
    /// Cancels the run: wakes every waiting worker and consumer. A no-op once
    /// the ring belongs to a later run, so a stale handle cannot abort it.
    pub fn cancel(&self) {
        if self.ring.generation() == self.generation {
            self.ring.abort();
        }
    }
    /// Executes the producer on the calling thread and returns the workers'
    /// statistics.
    ///
    /// A panic in a worker or a plan thread is caught and reported as
    /// [`Error::WorkerPanic`]; any error cancels the run. The ring's producer
    /// is released before returning, so the ring can serve a later run once
    /// every slot of this one has been consumed.
    pub fn run(&self) -> Result<RunStats> {
        if self.state.compare_exchange(0, 1, AcqRel, Acquire).is_err() {
            return Err(Error::Busy);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.ring.aborted() {
                return Err(Error::Aborted);
            }
            super::exec::run(self)
        }))
        .unwrap_or(Err(Error::WorkerPanic));
        if result.is_err() {
            self.cancel();
        }
        if self.ring.generation() == self.generation {
            self.ring.leave_producer(result.is_ok());
            self.ring.end();
        }
        self.state.store(if result.is_ok() { 2 } else { 3 }, Release);
        result
    }
    /// The slot of batch `batch`, once every output in it is written; none
    /// past the end. Batch `b` belongs to rank `b % world`, and a rank may ask
    /// only for its own batches; an error cancels the run.
    pub fn wait(&self, rank: u32, batch: u64) -> Result<Option<BatchLease>> {
        let result = (|| {
            let world = self.ring.layout().world;
            if rank >= world || batch % world as u64 != rank as u64 {
                return Err(config("batch does not belong to this consumer rank"));
            }
            if self.ring.aborted() {
                return Err(Error::Aborted);
            }
            match self.ring.acquire_generation(batch, self.generation)? {
                Some(lease) if lease.generation() != self.generation || lease.index() != batch => Err(config("output lease disagrees on generation or batch")),
                Some(lease) => Ok(Some(lease)),
                None if batch >= self.batches() => Ok(None),
                None => Err(config("the output ring ended before the run")),
            }
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    /// Local consumption of every rank's batches in order. Raw rank
    /// consumers may instead call `wait(rank, batch)` while the run executes.
    pub fn for_each_batch<F>(&self, mut consume: F) -> Result<RunStats>
    where
        F: FnMut(&BatchLease) -> Result<()>,
    {
        struct Cancel<'a, 'b> {
            run: &'a Run<'b>,
            armed: bool,
        }
        impl Drop for Cancel<'_, '_> {
            fn drop(&mut self) {
                if self.armed {
                    self.run.cancel();
                }
            }
        }
        let world = self.ring.layout().world as u64;
        std::thread::scope(|scope| {
            let mut guard = Cancel { run: self, armed: true };
            let producer = scope.spawn(|| self.run());
            let consumed = (|| {
                for batch in 0..self.batches() {
                    let lease = self.wait((batch % world) as u32, batch)?.ok_or_else(|| config("output run ended early"))?;
                    consume(&lease)?;
                }
                Ok(())
            })();
            if consumed.is_err() {
                self.cancel();
            }
            let produced = producer.join().unwrap_or(Err(Error::WorkerPanic));
            guard.armed = false;
            match (consumed, produced) {
                (Err(Error::Aborted), Err(p)) => Err(p),
                (Err(e), _) => Err(e),
                (_, r) => r,
            }
        })
    }
}
impl Drop for Run<'_> {
    // A run that was never started has no producer thread to cancel it, so
    // dropping it is what releases the ring and the loader.
    fn drop(&mut self) {
        if self.state.load(Acquire) == 0 {
            self.cancel();
            if self.ring.generation() == self.generation {
                self.ring.leave_producer(false);
                self.ring.end();
            }
        }
        self.loader.running.store(false, Release);
    }
}
