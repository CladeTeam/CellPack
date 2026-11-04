//! Compilation entry points over the engine (see [`crate::opt::engine`]):
//! offline over a stored sequence with the whole sequence as its horizon,
//! online with a finite horizon (into a plan, or counted only), and the
//! checks they share.

use std::sync::atomic::AtomicBool;

use crate::loader::accesses::Accesses;
use crate::loader::catalog::CellId;
use crate::loader::error::{Error, Result};
use crate::loader::schema::Graph;
use crate::opt::engine::{self, OnlineStats, Output, Params};
use crate::opt::model::PlanOptions;
use crate::opt::plan::Plan;
use crate::opt::records::{Records, Trace};
use crate::prof::Off;

/// The compiler threads' saved CPU affinity, restored when it drops.
pub(crate) struct Affinity(Option<libc::cpu_set_t>);

impl Affinity {
    /// Pins the calling thread to `cpus` and returns the guard that restores
    /// its previous set; an empty `cpus` binds nothing. Errors when an index
    /// is past `CPU_SETSIZE`, when the call fails, or when the environment
    /// clamps the requested set (a container's cpuset, typically).
    pub fn bind(cpus: &[usize]) -> Result<Self> {
        if cpus.is_empty() { return Ok(Self(None)); }
        if cpus.iter().any(|&c| c >= libc::CPU_SETSIZE as usize) {
            return Err(Error::Plan("compiler CPU index exceeds CPU_SETSIZE".into()));
        }
        // SAFETY: initialized cpu_set_t buffers; CPU indices are bounded.
        unsafe {
            let mut old: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut old) != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            let mut requested: libc::cpu_set_t = std::mem::zeroed();
            for &cpu in cpus { libc::CPU_SET(cpu, &mut requested); }
            if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &requested) != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            let guard = Self(Some(old));
            let mut actual: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut actual) != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            if cpus.iter().any(|&cpu| !libc::CPU_ISSET(cpu, &actual)) {
                return Err(Error::Plan("compiler CPU set was clamped by the environment".into()));
            }
            Ok(guard)
        }
    }
}

impl Drop for Affinity {
    fn drop(&mut self) {
        if let Some(old) = &self.0 {
            // SAFETY: restore this thread's saved, initialized mask.
            unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), old); }
        }
    }
}

/// Checks options against a graph (shared by offline and online plans): the
/// ranges the engine indexes with (`u32` frames, CPU indices, a positive gap
/// below 2^32 accesses, a horizon at least the gap), the options that need a
/// cost model, and the block and chunk sizes.
pub(crate) fn check(graph: &Graph, options: &PlanOptions) -> Result<()> {
    if options.cache_bytes / graph.page_size > u32::MAX as usize {
        return Err(Error::Plan("cache frame count exceeds u32".into()));
    }
    if options.cpus.iter().any(|&cpu| cpu >= libc::CPU_SETSIZE as usize) {
        return Err(Error::Plan("compiler CPU index exceeds CPU_SETSIZE".into()));
    }
    if options.gap == 0 || options.gap.saturating_mul(graph.lanes_per_slot()) >= u32::MAX as u64 {
        return Err(Error::Plan("the reuse gap must be positive and below 2^32 accesses".into()));
    }
    if options.horizon.is_some_and(|h| h < options.gap) {
        return Err(Error::Plan("the horizon must be at least the reuse gap".into()));
    }
    if let Some(c) = &options.cost {
        c.check(graph)?;
    }
    if (options.rows || options.block_pages.is_some()) && options.cost.is_none() {
        return Err(Error::Plan("caching rows, and blocks, need a cost model".into()));
    }
    if options.block_pages.is_some_and(|n| !(2..=super::plan::MAX_BLOCK_PAGES).contains(&n)) {
        return Err(Error::Plan("blocks hold 2 to 128 pages".into()));
    }
    if options.chunk == Some(0) {
        return Err(Error::Plan("the compilation chunk must be positive".into()));
    }
    Ok(())
}

/// Runs `f(0..n)` on `n` threads (inline for one).
pub(crate) fn par(n: usize, f: impl Fn(usize) + Sync) {
    if n <= 1 {
        if n == 1 {
            f(0);
        }
        return;
    }
    std::thread::scope(|s| {
        for k in 1..n {
            let f = &f;
            s.spawn(move || f(k));
        }
        f(0);
    });
}

/// Compiles a plan for a stored access sequence over a graph: the whole
/// sequence is the horizon.
pub(crate) fn compile(graph: &Graph, accesses: &[CellId], options: &PlanOptions) -> Result<Plan> {
    Ok(stored(graph, accesses, options, false)?.0)
}

/// Compiles a stored sequence as an online run would (a finite horizon,
/// chunked freezing), into a plan verified like an offline one.
pub(crate) fn collect(graph: &Graph, accesses: &[CellId], options: &PlanOptions) -> Result<(Plan, OnlineStats)> {
    stored(graph, accesses, options, true)
}

/// Compiles a stored sequence, either as the whole horizon (`online`: false)
/// or with the online compiler's finite horizon and chunked freezing.
fn stored(graph: &Graph, accesses: &[CellId], options: &PlanOptions, online: bool) -> Result<(Plan, OnlineStats)> {
    check(graph, options)?;
    if !(accesses.len() as u64).is_multiple_of(graph.schemas.len() as u64) {
        return Err(Error::Plan("the access count must be a multiple of the schema count".into()));
    }
    let _affinity = Affinity::bind(&options.cpus)?;
    let threads = options.threads.max(1);
    let records = Records::build(graph, threads)?;
    let trace = Trace::build(&records, accesses, threads)?;
    let p = Params::new(graph, options, accesses.len() as u64, online, false, records.cells);
    let d = p.d;
    // Nothing consumes a ring here, so the compiler always freezes the whole
    // horizon behind its frontier (`on_demand`: false).
    let (built, stats, _) = engine::compile::<Off>(graph, &records, &accesses, options, p, Output::Plan, &AtomicBool::new(false), &|| Off)?;
    let plan = Plan::assemble(graph, &records, &trace, d, options.cache_bytes, options.cost.clone(), built)?;
    Ok((plan, stats))
}

/// Compiles a sequence online without loading it, for its statistics.
pub(crate) fn dry(graph: &Graph, accesses: &dyn Accesses, options: &PlanOptions) -> Result<OnlineStats> {
    check(graph, options)?;
    if !accesses.len().is_multiple_of(graph.schemas.len() as u64) {
        return Err(Error::Plan("the access count must be a multiple of the schema count".into()));
    }
    let _affinity = Affinity::bind(&options.cpus)?;
    let records = Records::build(graph, options.threads.max(1))?;
    // The loader runs this sequence as it is compiled, so the compiler prices
    // its own work alongside the workers (`online`). Nothing consumes a ring
    // here, so freezing still follows the horizon (`on_demand` finds no
    // consumer).
    let p = Params::new(graph, options, accesses.len(), true, true, records.cells);
    Ok(engine::compile::<Off>(graph, &records, accesses, options, p, Output::Count, &AtomicBool::new(false), &|| Off)?.1)
}
