//! A started run for Python. Its batches are taken through the same
//! `Consumer` as any ring's; `join` waits for the producer and fails,
//! cancelling the run, when no consumer can ever give back a slot the
//! producer needs.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cellpack::loader as core;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use super::ring::{Batch, Consumer, Ring, RunSide};
use super::stats::run_stats;
use crate::errors::{Exception, loader_error, loader_exception};

/// How long `join` lets a stall persist before failing, so consumer
/// threads started just before it can begin.
const STALL_GRACE: Duration = Duration::from_millis(500);

struct State {
    // Drop the borrowed run before its backing owners.
    run: core::Run<'static>,
    _loader: Arc<core::Loader>,
    _access: Arc<dyn core::Accesses>,
    result: Mutex<Option<core::Result<core::RunStats>>>,
    done: Condvar,
}

impl State {
    fn poll(&self, timeout: Duration) {
        let r = self.result.lock().unwrap_or_else(|e| e.into_inner());
        if r.is_none() {
            let _ = self.done.wait_timeout(r, timeout);
        }
    }
    fn finished(&self) -> bool {
        self.result.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    fn error(&self) -> Option<(Exception, String)> {
        match self.result.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            Some(Err(e)) => Some((loader_exception(e), e.to_string())),
            _ => None,
        }
    }
}

impl RunSide for State {
    fn generation(&self) -> u64 {
        self.run.generation()
    }
    fn batches(&self) -> u64 {
        self.run.batches()
    }
    fn cancel(&self) {
        self.run.cancel();
    }
    fn failure(&self) -> Option<PyErr> {
        self.error().map(|(kind, message)| kind.error(message))
    }
    fn fail(&self, py: Python<'_>, fallback: PyErr) -> PyErr {
        self.run.cancel();
        while !self.finished() {
            py.detach(|| self.poll(Duration::from_millis(50)));
        }
        self.failure().unwrap_or(fallback)
    }
}

/// A started run: iterate it for rank `rank`'s batches (advancing releases
/// the previous one; arrays taken from it keep its slot until dropped), or
/// take any rank's through `consumer(rank)`; then `join` for statistics.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "Run")]
pub(crate) struct Run {
    state: Mutex<Option<Arc<State>>>,
    producer: Mutex<Option<JoinHandle<()>>>,
    // The category and text, not a Python exception with a traceback that
    // could retain this run through a reference cycle.
    completed: Mutex<Option<Result<core::RunStats, (Exception, String)>>>,
    ring: Py<Ring>,
    rank: u32,
    world: u32,
    names: Arc<[String]>,
    own: Mutex<Option<Py<Consumer>>>,
}

impl Run {
    /// Prepares `loader`'s run of `access` over `ring` and starts its
    /// producer thread; the caller consumes it as rank `rank`.
    pub(super) fn start(py: Python<'_>, loader: Arc<core::Loader>, access: Arc<dyn core::Accesses>, planning: core::Planning, ring: Bound<'_, Ring>, rank: u32, names: Arc<[String]>) -> PyResult<Self> {
        let world = ring.get().layout.get().world;
        if rank >= world {
            return Err(PyValueError::new_err("rank outside the ring's world"));
        }
        let core_ring = ring.get().inner.clone();
        let run = py
            .detach(|| {
                let r = loader.prepare(&*access, core_ring, planning)?;
                // SAFETY: the run borrows the loader and the sequence, both
                // behind Arcs that State retains until the run drops.
                Ok::<_, core::Error>(unsafe { std::mem::transmute::<core::Run<'_>, core::Run<'static>>(r) })
            })
            .map_err(loader_error)?;
        let state = Arc::new(State { run, _loader: loader, _access: access, result: Mutex::new(None), done: Condvar::new() });
        let producer_state = state.clone();
        let producer = std::thread::Builder::new()
            .name("cellpack-outputs".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| producer_state.run.run())).unwrap_or(Err(core::Error::WorkerPanic));
                if result.is_err() {
                    producer_state.run.cancel();
                }
                *producer_state.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
                producer_state.done.notify_all();
            })
            .map_err(|e| {
                state.run.cancel();
                PyRuntimeError::new_err(e.to_string())
            })?;
        Ok(Self {
            state: Mutex::new(Some(state)),
            producer: Mutex::new(Some(producer)),
            completed: Mutex::new(None),
            ring: ring.unbind(),
            rank,
            world,
            names,
            own: Mutex::new(None),
        })
    }

    fn state(&self) -> PyResult<Arc<State>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| PyRuntimeError::new_err("the run is closed"))
    }
    fn join_producer(&self) {
        if let Some(h) = self.producer.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
    fn cache(&self, s: &State) {
        if let Some(r) = s.result.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            *self.completed.lock().unwrap_or_else(|e| e.into_inner()) = Some(match r {
                Ok(r) => Ok(r.clone()),
                Err(e) => Err((loader_exception(e), e.to_string())),
            });
        }
    }
    fn close_native(&self) {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(s) = s {
            if !s.finished() {
                s.run.cancel();
            }
            self.join_producer();
            self.cache(&s);
        }
    }
    /// This run's own consumer of its rank, opened on first use.
    fn own(&self, py: Python<'_>) -> PyResult<Py<Consumer>> {
        let mut own = self.own.lock().unwrap_or_else(|e| e.into_inner());
        if own.is_none() {
            *own = Some(Py::new(py, self.open(py, self.rank)?)?);
        }
        Ok(own.as_ref().unwrap().clone_ref(py))
    }
    fn open(&self, py: Python<'_>, rank: u32) -> PyResult<Consumer> {
        let s = self.state()?;
        let generation = s.run.generation();
        Consumer::open(self.ring.bind(py), rank, generation, None, true, Some(s))
    }
    /// Why the producer can never finish, and whether that is certain now.
    fn stall(&self, s: &State) -> (Option<String>, bool) {
        let ring = self.ring.get();
        if let Some(reason) = ring.inner.consumer_problem(s.run.generation()) {
            return (Some(reason), true);
        }
        if ring.shared {
            // Consumers in other processes may still attach.
            return (None, false);
        }
        let joiner = std::thread::current().id();
        for (batch, held) in ring.inner.blocking(s.run.batches()) {
            let rank = (batch % self.world as u64) as u32;
            if !ring.local.able(rank, joiner) {
                let what = if held { "is still held" } else { "was not taken" };
                return (
                    Some(format!(
                        "batch {batch} of rank {rank} {what}, so the producer cannot reuse its slot, and no other thread consumes rank {rank}; consume the run before join() (or from another thread), or cancel()"
                    )),
                    false,
                );
            }
        }
        (None, false)
    }
}

#[pymethods]
impl Run {
    fn __repr__(&self) -> String {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let (status, detail) = match state {
            Some(s) => {
                let result = s.result.lock().unwrap_or_else(|e| e.into_inner());
                let status = match result.as_ref() {
                    None => "running",
                    Some(Ok(_)) => "succeeded",
                    Some(Err(_)) => "failed",
                };
                (status, format!(", batches={}, generation={}", s.run.batches(), s.run.generation()))
            }
            None => ("closed", format!(", succeeded={}", if self.succeeded() { "True" } else { "False" })),
        };
        format!("Run(rank={}, world={}, state={status:?}{detail})", self.rank, self.world)
    }
    /// The ring generation this run belongs to.
    #[getter]
    fn generation(&self) -> PyResult<u64> {
        Ok(self.state()?.run.generation())
    }
    /// Slots the run fills (batches over every rank).
    #[getter]
    fn batches(&self) -> PyResult<u64> {
        Ok(self.state()?.run.batches())
    }
    /// The ring the run writes into.
    #[getter]
    fn ring(&self, py: Python<'_>) -> Py<Ring> {
        self.ring.clone_ref(py)
    }
    /// The rank this process consumes by iterating the run.
    #[getter]
    fn rank(&self) -> u32 {
        self.rank
    }
    /// Consumer ranks sharing the ring.
    #[getter]
    fn world(&self) -> u32 {
        self.world
    }
    /// Whether the producer stopped (successfully, failed or cancelled).
    #[getter]
    fn finished(&self) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_none_or(|s| s.finished())
    }
    /// Whether the producer finished successfully; also after close.
    #[getter]
    fn succeeded(&self) -> bool {
        if let Ok(s) = self.state() {
            return matches!(s.result.lock().unwrap_or_else(|e| e.into_inner()).as_ref(), Some(Ok(_)));
        }
        matches!(self.completed.lock().unwrap_or_else(|e| e.into_inner()).as_ref(), Some(Ok(_)))
    }
    /// Stops the producer at the next safe point; the run reports failure.
    fn cancel(&self) {
        if let Ok(s) = self.state() {
            s.run.cancel();
        }
    }
    /// Cancels an unfinished run and waits for its producer.
    fn close(&self, py: Python<'_>) {
        if let Some(own) = self.own.lock().unwrap_or_else(|e| e.into_inner()).take()
            && let Ok(mut c) = own.bind(py).try_borrow_mut()
        {
            c.finish(py);
        }
        py.detach(|| self.close_native());
    }
    /// Entering a run leaves it open; exiting closes (and cancels an
    /// unfinished run).
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __exit__(&self, py: Python<'_>, _ty: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _tb: &Bound<'_, PyAny>) -> bool {
        self.close(py);
        false
    }
    /// Iterates this process's rank's batches.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    /// Advances the run's own consumer; stops at the end or after close.
    fn __next__(&self, py: Python<'_>) -> PyResult<Py<Batch>> {
        if self.state.lock().unwrap_or_else(|e| e.into_inner()).is_none() {
            return Err(PyStopIteration::new_err(()));
        }
        let own = self.own(py)?;
        let batch = own.bind(py).try_borrow_mut()?.advance(py)?;
        batch.ok_or_else(|| PyStopIteration::new_err(()))
    }
    /// Batch `batch` of this run's rank, or None past the end.
    fn wait(&self, py: Python<'_>, batch: u64) -> PyResult<Option<Py<Batch>>> {
        let own = self.own(py)?;
        let mut c = own.bind(py).try_borrow_mut()?;
        c.take(py, batch)
    }
    /// A consumer of rank `rank`'s batches of this run (another thread's
    /// share of a multi-rank ring, say).
    #[pyo3(signature = (rank, *, idle = None))]
    fn consumer(&self, py: Python<'_>, rank: u32, idle: Option<Py<PyAny>>) -> PyResult<Consumer> {
        let s = self.state()?;
        let generation = s.run.generation();
        Consumer::open(self.ring.bind(py), rank, generation, idle, true, Some(s))
    }
    /// Whether the producer finished within `timeout_ms` (0 to 1000).
    #[pyo3(signature = (timeout_ms = 50))]
    fn wait_finished(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<bool> {
        if timeout_ms > 1000 {
            return Err(PyValueError::new_err("the timeout is at most 1000 ms"));
        }
        let Ok(s) = self.state() else { return Ok(true) };
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        while !s.finished() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            py.detach(|| s.poll(remaining));
            if let Err(signal) = py.check_signals() {
                s.run.cancel();
                return Err(signal);
            }
        }
        Ok(s.finished())
    }
    /// The workers' statistics so far, as last published (profiled runs add
    /// their sites).
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = self.state()?;
        let stats = py.detach(|| s.run.snapshot());
        run_stats(py, &stats, &self.names)
    }
    /// Waits for the producer and returns the run's statistics. Fails,
    /// cancelling the run, when it cannot finish: a rank's batch blocks a
    /// slot the producer needs and no other thread consumes that rank, or a
    /// consumer of a shared ring died or left early.
    fn join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(s) = self.state() {
            let mut stalled: Option<Instant> = None;
            while !s.finished() {
                py.detach(|| s.poll(Duration::from_millis(50)));
                if let Err(signal) = py.check_signals() {
                    s.run.cancel();
                    py.detach(|| self.join_producer());
                    self.cache(&s);
                    return Err(signal);
                }
                if s.finished() {
                    break;
                }
                match self.stall(&s) {
                    (None, _) => stalled = None,
                    (Some(reason), now) => {
                        let since = *stalled.get_or_insert_with(Instant::now);
                        if now || since.elapsed() >= STALL_GRACE {
                            s.run.cancel();
                            py.detach(|| self.join_producer());
                            self.cache(&s);
                            return Err(PyRuntimeError::new_err(format!("the run cannot finish: {reason}")));
                        }
                    }
                }
            }
            py.detach(|| self.join_producer());
            self.cache(&s);
        }
        let result = self.completed.lock().unwrap_or_else(|e| e.into_inner());
        match result.as_ref() {
            Some(Ok(stats)) => run_stats(py, stats, &self.names),
            Some(Err((kind, message))) => Err(kind.error(message.clone())),
            None => Err(PyRuntimeError::new_err("the run has no result")),
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        if let Ok(s) = self.state() {
            s.run.cancel();
        }
        // Functions may need the GIL to finish. During interpreter shutdown,
        // cancellation is best effort and join handles detach rather than wait.
        let _ = Python::try_attach(|py| py.detach(|| self.close_native()));
    }
}
