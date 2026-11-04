//! Output rings for Python: a picklable layout naming each output's view,
//! rings over owned or caller memory, typed slot leases and per-rank
//! consumers. Consumers register in the ring's participant rows, so a
//! blocked wait fails with a reason when the producer process is gone.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use cellpack::format::Dtype;
use cellpack::loader as core;
use numpy::{PyArray2, ndarray::ArrayView2};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyRuntimeWarning, PyStopIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::buffers::{Buffer, disjoint, dtype_from_name, dtype_name, with_dtype};
use crate::errors::loader_error;

/// How often a blocked wait rereads the producer's participant row.
const CHECK_INTERVAL: Duration = Duration::from_millis(250);
/// One bounded wait between signal and peer checks.
const POLL: Duration = Duration::from_millis(50);

/// A NumPy dtype from anything `numpy.dtype` accepts.
pub(crate) fn dtype_arg(value: &Bound<'_, PyAny>) -> PyResult<Dtype> {
    let dtype = value.py().import("numpy")?.getattr("dtype")?.call1((value,))?;
    if !dtype.getattr("isnative")?.extract::<bool>()? {
        return Err(PyValueError::new_err("output dtype must have native byte order"));
    }
    let name: String = dtype.getattr("name")?.extract()?;
    dtype_from_name(&name)
}

/// One output's place in a slot: its name, the dtype its rows are viewed
/// as (none: raw bytes) and the bytes of one row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OutputView {
    pub name: String,
    pub dtype: Option<Dtype>,
    pub row_bytes: u32,
}

/// Slots of `slot_rows` positions, each holding every output's rows.
///
/// The layout fixes the control and data sizes and the mapping from an
/// output name to its region; it carries no storage and pickles by value.
#[pyclass(frozen, eq, skip_from_py_object, module = "pycellpack._cellpack", name = "RingLayout")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RingLayout {
    pub slots: u32,
    pub slot_rows: u32,
    pub world: u32,
    pub outputs: Arc<[OutputView]>,
}

impl RingLayout {
    /// The native layout (sizes and per-output row bytes).
    pub(crate) fn native(&self) -> core::RingLayout {
        core::RingLayout { slots: self.slots, slot_cells: self.slot_rows, world: self.world, cell_bytes: self.outputs.iter().map(|o| o.row_bytes).collect() }
    }
    /// The output an index or name selects.
    pub(crate) fn index(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(name) = key.extract::<&str>() {
            return self.outputs.iter().position(|o| o.name == name).ok_or_else(|| PyKeyError::new_err(name.to_string()));
        }
        let i: isize = key.extract().map_err(|_| PyTypeError::new_err("outputs are selected by name or index"))?;
        let n = self.outputs.len() as isize;
        let j = if i < 0 { i + n } else { i };
        if !(0..n).contains(&j) {
            return Err(PyIndexError::new_err("output index out of range"));
        }
        Ok(j as usize)
    }
    fn tuples(&self) -> Vec<(String, Option<&'static str>, u32)> {
        self.outputs.iter().map(|o| (o.name.clone(), o.dtype.map(dtype_name), o.row_bytes)).collect()
    }
    fn output_summary(&self) -> String {
        let mut items: Vec<String> = self.outputs.iter().take(4).map(|o| {
            let name: String = o.name.chars().take(64).collect();
            match o.dtype {
                Some(d) => format!("{name:?}: {}[{}]", dtype_name(d), o.row_bytes as usize / d.width()),
                None => format!("{name:?}: bytes[{}]", o.row_bytes),
            }
        }).collect();
        if self.outputs.len() > 4 {
            items.push("...".into());
        }
        format!("{{{}}}", items.join(", "))
    }
}

#[pymethods]
impl RingLayout {
    /// `outputs`: `(name, dtype, row_bytes)` per output in loader order;
    /// `dtype` (anything `numpy.dtype` takes) views each row as
    /// `row_bytes / itemsize` elements, `None` as raw bytes.
    #[new]
    #[pyo3(signature = (slots, slot_rows, outputs, world = 1))]
    fn new(slots: u32, slot_rows: u32, outputs: Vec<(String, Option<Bound<'_, PyAny>>, u32)>, world: u32) -> PyResult<Self> {
        let mut views = Vec::with_capacity(outputs.len());
        for (name, dtype, row_bytes) in outputs {
            if name.is_empty() || views.iter().any(|v: &OutputView| v.name == name) {
                return Err(PyValueError::new_err("outputs need distinct, nonempty names"));
            }
            let dtype = dtype.map(|d| dtype_arg(&d)).transpose()?;
            if dtype.is_some_and(|d| !(row_bytes as usize).is_multiple_of(d.width())) {
                return Err(PyValueError::new_err(format!("output {name:?}: row bytes are not a whole number of its dtype")));
            }
            views.push(OutputView { name, dtype, row_bytes });
        }
        let layout = Self { slots, slot_rows, world, outputs: views.into() };
        layout.native().control_bytes().map_err(loader_error)?;
        Ok(layout)
    }
    /// The number of slots.
    #[getter]
    fn slots(&self) -> u32 {
        self.slots
    }
    /// Positions per slot.
    #[getter]
    fn slot_rows(&self) -> u32 {
        self.slot_rows
    }
    /// Consumer ranks sharing the ring.
    #[getter]
    fn world(&self) -> u32 {
        self.world
    }
    /// The output names, in slot order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.outputs.iter().map(|o| o.name.clone()).collect()
    }
    /// `(name, dtype name or None, row_bytes)` per output.
    #[getter]
    fn outputs(&self) -> Vec<(String, Option<&'static str>, u32)> {
        self.tuples()
    }
    /// Bytes of one row per output, in slot order.
    #[getter]
    fn row_bytes(&self) -> Vec<u32> {
        self.outputs.iter().map(|o| o.row_bytes).collect()
    }
    /// Bytes of the control storage: header, slot headers and participant rows.
    #[getter]
    fn control_bytes(&self) -> PyResult<usize> {
        self.native().control_bytes().map_err(loader_error)
    }
    /// Bytes of data storage for every slot, the last slot's padding included.
    #[getter]
    fn data_bytes(&self) -> PyResult<usize> {
        self.native().data_bytes().map_err(loader_error)
    }
    /// The least data storage: the last slot need not hold its padding.
    #[getter]
    fn data_min_bytes(&self) -> PyResult<usize> {
        self.native().data_min_bytes().map_err(loader_error)
    }
    /// Each output's offset within a slot.
    #[getter]
    fn offsets(&self) -> Vec<usize> {
        self.native().regions().map(|r| r.0).unwrap_or_default()
    }
    /// Bytes between successive slots (4 KiB aligned).
    #[getter]
    fn slot_stride(&self) -> PyResult<usize> {
        self.native().slot_stride().map_err(loader_error)
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let l = slf.get();
        Ok((slf.get_type().into_any(), (l.slots, l.slot_rows, l.tuples(), l.world).into_pyobject(slf.py())?))
    }
    fn __repr__(&self) -> String {
        format!("RingLayout(slots={}, slot_rows={}, outputs={}, world={})", self.slots, self.slot_rows, self.output_summary(), self.world)
    }
}

/// This process's open consumers of one ring, for telling a stalled run
/// from one whose consumers are still at work.
#[derive(Default)]
pub(crate) struct Local {
    next: AtomicU64,
    entries: Mutex<Vec<Entry>>,
}

struct Entry {
    id: u64,
    rank: u32,
    /// The thread that last waited on it.
    thread: ThreadId,
    finished: bool,
}

impl Local {
    fn open(&self, rank: u32) -> u64 {
        let id = self.next.fetch_add(1, Relaxed);
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).push(Entry { id, rank, thread: std::thread::current().id(), finished: false });
        id
    }
    fn update(&self, id: u64, finished: bool) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = entries.iter_mut().find(|e| e.id == id) {
            e.thread = std::thread::current().id();
            e.finished = finished;
        }
    }
    fn close(&self, id: u64) {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).retain(|e| e.id != id);
    }
    /// Whether an unfinished consumer of `rank` waits from a thread other
    /// than `excluding`.
    pub(crate) fn able(&self, rank: u32, excluding: ThreadId) -> bool {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|e| e.rank == rank && !e.finished && e.thread != excluding)
    }
}

/// One run's output storage: a control block (header, slot headers and
/// participant rows) plus `slots` slots of data, shared by one producer and
/// `world` consumers. Storage lives either natively, in caller arrays
/// (possibly shared memory), or in a shared mapping attached in another
/// process.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "Ring")]
pub(crate) struct Ring {
    pub inner: Arc<core::Ring>,
    pub layout: Py<RingLayout>,
    /// Control storage the caller provides (and may share with other
    /// processes).
    pub shared: bool,
    pub local: Arc<Local>,
}

#[pymethods]
impl Ring {
    fn __repr__(&self) -> String {
        let l = self.layout.get();
        format!("Ring(slots={}, slot_rows={}, outputs={}, world={}, shared={}, generation={})",
                l.slots, l.slot_rows, l.output_summary(), l.world, if self.shared { "True" } else { "False" }, self.inner.generation())
    }
    /// A ring of `layout`: allocated natively, or over a caller's uint8
    /// `data` array (`data_min_bytes` to `data_bytes` bytes, 8-byte aligned)
    /// and, to share it with other processes, a uint8 `control` array
    /// (`control_bytes`, 64-byte aligned) initialized here or, with
    /// `initialize=False`, attached to.
    #[new]
    #[pyo3(signature = (layout, *, data = None, control = None, initialize = true))]
    fn new(py: Python<'_>, layout: Bound<'_, RingLayout>, data: Option<&Bound<'_, PyAny>>, control: Option<&Bound<'_, PyAny>>, initialize: bool) -> PyResult<Self> {
        let native = layout.get().native();
        let shared = control.is_some();
        let Some(data) = data else {
            if control.is_some() {
                return Err(PyValueError::new_err("shared control needs shared data"));
            }
            let inner = py.detach(|| core::Ring::allocate(native)).map_err(loader_error)?;
            return Ok(Self { inner: Arc::new(inner), layout: layout.unbind(), shared, local: Arc::default() });
        };
        let data = Buffer::get(data, true)?;
        if data.dtype != Dtype::U8 || data.shape.len() != 1 {
            return Err(PyValueError::new_err("ring data requires a 1D uint8 array"));
        }
        let control = control.map(|a| Buffer::get(a, true)).transpose()?;
        if let Some(c) = &control {
            if c.dtype != Dtype::U8 || c.shape.len() != 1 {
                return Err(PyValueError::new_err("control requires a 1D uint8 array"));
            }
            disjoint(&[&data, c])?;
        } else if !initialize {
            return Err(PyValueError::new_err("attachment requires shared control storage"));
        }
        let (dp, dl) = (data.ptr, data.len);
        let external = control.as_ref().map(|c| (c.ptr, c.len));
        let owners: Arc<dyn Send + Sync> = Arc::new((data.owner, control.map(|c| c.owner)));
        // SAFETY: regions are validated and their Python owners are retained
        // by the core ring; consumers follow the slot lease protocol.
        let inner = unsafe {
            match external {
                Some((cp, cl)) if initialize => core::Ring::init_regions(cp as *mut u8, cl, dp as *mut u8, dl, native.clone(), owners),
                Some((cp, cl)) => core::Ring::attach_regions(cp as *mut u8, cl, dp as *mut u8, dl, owners),
                None => core::Ring::with_buffer(dp as *mut u8, dl, native.clone(), owners),
            }
        }
        .map_err(loader_error)?;
        if inner.layout() != native {
            return Err(PyValueError::new_err("the attached ring's layout differs from `layout`"));
        }
        Ok(Self { inner: Arc::new(inner), layout: layout.unbind(), shared, local: Arc::default() })
    }
    /// This ring's layout (the same object it was created with).
    #[getter]
    fn layout(&self, py: Python<'_>) -> Py<RingLayout> {
        self.layout.clone_ref(py)
    }
    /// The number of slots.
    #[getter]
    fn slots(&self) -> u32 {
        self.layout.get().slots
    }
    /// The number of consumer ranks.
    #[getter]
    fn world(&self) -> u32 {
        self.layout.get().world
    }
    /// Whether the control storage is the caller's (possibly shared with
    /// other processes).
    #[getter]
    fn shared(&self) -> bool {
        self.shared
    }
    /// The current run generation; leases belong to one generation.
    #[getter]
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
    /// Aborts the ring for every participant: blocked waits fail, and no later
    /// run can start on it.
    fn abort(&self) {
        self.inner.abort()
    }
    /// Joins the current (or next) run as rank `rank`'s consumer, possibly
    /// from another process: iterate it for batches `rank, rank + world,
    /// ...`. `idle()` runs between polls of a blocked wait and may raise to
    /// stop it; `watch_producer` fails a blocked wait whose producer process
    /// is gone.
    #[pyo3(signature = (rank = 0, *, idle = None, watch_producer = true))]
    fn consumer(slf: &Bound<'_, Self>, rank: u32, idle: Option<Py<PyAny>>, watch_producer: bool) -> PyResult<Consumer> {
        let generation = slf.get().inner.generation();
        Consumer::open(slf, rank, generation, idle, watch_producer, None)
    }
}

#[pyclass(frozen, module = "pycellpack._cellpack")]
struct ArrayLease {
    _lease: Arc<core::BatchLease>,
}

/// Observes a batch's slot lease without holding it.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "LeaseObserver")]
pub(crate) struct LeaseObserver {
    lease: Weak<core::BatchLease>,
}
#[pymethods]
impl LeaseObserver {
    /// Whether any batch or exported array still leases the slot.
    #[getter]
    fn held(&self) -> bool {
        self.lease.strong_count() > 0
    }
}

/// One slot: every output's rows of `rows` consecutive positions. Index it
/// by output name (or position) for that output's array; arrays keep the
/// slot leased until they are dropped, also after the batch is released.
#[pyclass(module = "pycellpack._cellpack", name = "Batch", mapping)]
pub(crate) struct Batch {
    lease: Option<Arc<core::BatchLease>>,
    // Observes the lease without keeping it: exported arrays hold strong
    // references after the batch itself is released.
    observer: Weak<core::BatchLease>,
    layout: Py<RingLayout>,
}

impl Batch {
    /// Wraps a native lease; the batch releases it on `release`, arrays
    /// observe it through the weak reference.
    pub(crate) fn new(lease: core::BatchLease, layout: Py<RingLayout>) -> Self {
        let lease = Arc::new(lease);
        Self { observer: Arc::downgrade(&lease), lease: Some(lease), layout }
    }
    fn lease(&self) -> PyResult<&Arc<core::BatchLease>> {
        self.lease.as_ref().ok_or_else(|| PyRuntimeError::new_err("batch is released"))
    }
    /// Output `i`'s rows, as its view's dtype unless `raw`.
    fn array<'py>(&self, py: Python<'py>, i: usize, raw: bool) -> PyResult<Bound<'py, PyAny>> {
        let lease = self.lease()?;
        let view = &self.layout.get().outputs[i];
        let owner = Bound::new(py, ArrayLease { _lease: lease.clone() })?.into_any();
        let region = lease.region(i);
        let rows = lease.rows();
        Ok(match view.dtype.filter(|_| !raw) {
            Some(dtype) => with_dtype!(dtype, T, {
                let cols = view.row_bytes as usize / std::mem::size_of::<T>();
                // SAFETY: the leased region holds `rows × row_bytes` ready
                // bytes, aligned for T (regions are 64-byte aligned, data at
                // least 8); the owner keeps storage and slot alive and no
                // producer writes the slot until every view drops.
                let array = unsafe { ArrayView2::from_shape_ptr((rows, cols), region.as_ptr().cast::<T>()) };
                // SAFETY: ArrayLease is the base; neither pointer nor storage moves.
                unsafe { PyArray2::borrow_from_array(&array, owner) }.into_any()
            }),
            None => {
                // SAFETY: as above, as bytes.
                let array = unsafe { ArrayView2::from_shape_ptr((rows, view.row_bytes as usize), region.as_ptr()) };
                // SAFETY: as above.
                unsafe { PyArray2::borrow_from_array(&array, owner) }.into_any()
            }
        })
    }
}

#[pymethods]
impl Batch {
    fn __repr__(&self) -> String {
        let status = match &self.lease {
            Some(lease) => format!("index={}, rows={}, generation={}, released=False", lease.index(), lease.rows(), lease.generation()),
            None => "released=True".into(),
        };
        format!("Batch({status}, held={}, outputs={})", if self.held() { "True" } else { "False" }, self.layout.get().output_summary())
    }
    /// The batch's index in the run.
    #[getter]
    fn index(&self) -> PyResult<u64> {
        Ok(self.lease()?.index())
    }
    /// The first access position the batch holds.
    #[getter]
    fn first_position(&self) -> PyResult<u64> {
        Ok(self.lease()?.first_position())
    }
    /// Positions (and therefore rows per output) in the batch.
    #[getter]
    fn rows(&self) -> PyResult<usize> {
        Ok(self.lease()?.rows())
    }
    /// The run generation the lease belongs to.
    #[getter]
    fn generation(&self) -> PyResult<u64> {
        Ok(self.lease()?.generation())
    }
    /// The output names, in slot order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.layout.get().names()
    }
    /// The output names, in slot order (mapping key view).
    fn keys(&self) -> Vec<String> {
        self.names()
    }
    /// The number of outputs.
    fn __len__(&self) -> usize {
        self.layout.get().outputs.len()
    }
    /// Iterates output names.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.names().into_pyobject(py)?.try_iter().map(Bound::into_any)
    }
    /// Whether the batch has an output of that name.
    fn __contains__(&self, name: &str) -> bool {
        self.layout.get().outputs.iter().any(|o| o.name == name)
    }
    /// An output's rows, by name or position: `(rows, row_bytes /
    /// itemsize)` of its dtype, or `(rows, row_bytes)` bytes.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let i = self.layout.get().index(key)?;
        self.array(py, i, false)
    }
    /// An output's raw `(rows, row_bytes)` bytes.
    fn region<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let i = self.layout.get().index(key)?;
        self.array(py, i, true)
    }
    /// Every output's array, by name.
    fn arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (i, o) in self.layout.get().outputs.iter().enumerate() {
            d.set_item(&o.name, self.array(py, i, false)?)?;
        }
        Ok(d)
    }
    /// The only output's array (loaders with one output).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.layout.get().outputs.len() != 1 {
            return Err(PyTypeError::new_err("data names the only output; index batches with several by name"));
        }
        self.array(py, 0, false)
    }
    /// Gives the slot back (once exported arrays are dropped too).
    fn release(&mut self) {
        self.lease.take();
    }
    /// Whether this batch or arrays exported from it still lease the slot.
    #[getter]
    fn held(&self) -> bool {
        self.observer.strong_count() > 0
    }
    /// A handle reporting `held` without keeping the slot leased itself.
    fn observer(&self) -> LeaseObserver {
        LeaseObserver { lease: self.observer.clone() }
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.lease()?;
        Ok(slf)
    }
    fn __exit__(&mut self, _ty: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _tb: &Bound<'_, PyAny>) -> bool {
        self.release();
        false
    }
}

/// The run a consumer takes part in, when it was opened from one: its
/// failures replace the ring's.
pub(crate) trait RunSide: Send + Sync {
    fn generation(&self) -> u64;
    fn batches(&self) -> u64;
    fn cancel(&self);
    /// The run's error once it failed, if it has.
    fn failure(&self) -> Option<PyErr>;
    /// Cancels the run and waits for its producer; its error, else `fallback`.
    fn fail(&self, py: Python<'_>, fallback: PyErr) -> PyErr;
}

/// One rank's consumer: iterate it for the rank's batches; advancing
/// releases the previous batch (its exported arrays stay valid). Closing
/// it (or leaving its `with` block) records whether it took every batch,
/// so a producer fails instead of waiting for a consumer that left.
#[pyclass(module = "pycellpack._cellpack", name = "Consumer")]
pub(crate) struct Consumer {
    ring: Py<Ring>,
    core: Arc<core::Ring>,
    local: Arc<Local>,
    entry: u64,
    rank: u32,
    world: u32,
    next: u64,
    current: Option<Py<Batch>>,
    finished: bool,
    closed: bool,
    adopted: bool,
    idle: Option<Py<PyAny>>,
    watch: bool,
    checked: Instant,
    /// Per slot: the batch this consumer last took from it, observed.
    held: Vec<Option<(u64, Weak<core::BatchLease>)>>,
    run: Option<Arc<dyn RunSide>>,
}

impl Consumer {
    /// Joins `ring` as `rank`'s consumer, recording this process's entry so
    /// a stalled producer can tell whether another thread could still take
    /// the batches.
    pub(crate) fn open(ring: &Bound<'_, Ring>, rank: u32, generation: u64, idle: Option<Py<PyAny>>, watch: bool, run: Option<Arc<dyn RunSide>>) -> PyResult<Self> {
        let r = ring.get();
        let world = r.layout.get().world;
        if rank >= world {
            return Err(PyValueError::new_err("rank outside the ring's world"));
        }
        r.inner.join_consumer(rank, generation).map_err(loader_error)?;
        Ok(Self {
            ring: ring.clone().unbind(),
            core: r.inner.clone(),
            local: r.local.clone(),
            entry: r.local.open(rank),
            rank,
            world,
            next: rank as u64,
            current: None,
            finished: false,
            closed: false,
            adopted: false,
            idle,
            watch,
            checked: Instant::now(),
            held: vec![None; r.layout.get().slots as usize],
            run,
        })
    }

    fn generation(&self) -> u64 {
        self.run.as_ref().map_or_else(|| self.core.generation(), |r| r.generation())
    }

    /// Warns when `batch` must wait for a slot this consumer's own earlier
    /// batch (or its arrays) still holds: only another thread can free it.
    fn check_slot(&self, py: Python<'_>, batch: u64) -> PyResult<()> {
        let slot = (batch % self.held.len() as u64) as usize;
        if let Some((previous, lease)) = &self.held[slot]
            && *previous != batch
            && lease.strong_count() > 0
        {
            let message = format!(
                "batch {batch} waits for ring slot {slot}, which batch {previous} still holds; this blocks until it is released and its arrays are dropped (copy arrays or use more than {} slots to keep them)",
                self.held.len()
            );
            PyErr::warn(py, &py.get_type::<PyRuntimeWarning>(), &std::ffi::CString::new(message).unwrap(), 1)?;
        }
        Ok(())
    }

    fn release_current(&mut self, py: Python<'_>) {
        if let Some(current) = self.current.take()
            && let Ok(mut b) = current.bind(py).try_borrow_mut()
        {
            b.release();
        }
    }

    /// Waits for `batch`, polling so Python signals, the run's failure and
    /// the producer's participant row stay observable; returns None at the
    /// end of the run.
    pub(crate) fn take(&mut self, py: Python<'_>, batch: u64) -> PyResult<Option<Py<Batch>>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("consumer is closed"));
        }
        if batch % self.world as u64 != self.rank as u64 {
            return Err(PyValueError::new_err("batch does not belong to this consumer's rank"));
        }
        self.check_slot(py, batch)?;
        self.local.update(self.entry, false);
        loop {
            let core = self.core.clone();
            match py.detach(|| core.acquire_timeout(batch, POLL)) {
                Ok(core::Acquisition::Batch(lease)) => {
                    if let Some(run) = &self.run
                        && lease.generation() != run.generation()
                    {
                        run.cancel();
                        return Err(PyRuntimeError::new_err("output lease from another run"));
                    }
                    if !self.adopted {
                        self.core.adopt_generation(self.rank, lease.generation());
                        self.adopted = true;
                    }
                    let b = Batch::new(lease, self.ring.get().layout.clone_ref(py));
                    let slot = (batch % self.held.len() as u64) as usize;
                    self.held[slot] = Some((batch, b.observer.clone()));
                    self.next = self.next.max(batch + self.world as u64);
                    return Ok(Some(Py::new(py, b)?));
                }
                Ok(core::Acquisition::Done) => {
                    if let Some(run) = &self.run
                        && batch < run.batches()
                    {
                        return Err(run.fail(py, PyRuntimeError::new_err("the output ring ended before the run")));
                    }
                    self.finished = true;
                    self.local.update(self.entry, true);
                    return Ok(None);
                }
                Ok(core::Acquisition::Pending) => {
                    if let Err(signal) = py.check_signals() {
                        match &self.run {
                            Some(run) => run.cancel(),
                            None => self.core.abort(),
                        }
                        return Err(signal);
                    }
                    if let Some(e) = self.run.as_ref().and_then(|r| r.failure()) {
                        return Err(e);
                    }
                    if let Some(idle) = &self.idle {
                        idle.call0(py)?;
                    }
                    if self.watch && self.checked.elapsed() >= CHECK_INTERVAL {
                        self.checked = Instant::now();
                        if let Some(reason) = self.core.producer_problem(self.generation()) {
                            return Err(PyRuntimeError::new_err(reason));
                        }
                    }
                }
                Err(e) => {
                    let e = loader_error(e);
                    return Err(match &self.run {
                        Some(run) => run.fail(py, e),
                        None => e,
                    });
                }
            }
        }
    }

    /// Releases the current batch, then takes the next one (None at the end).
    pub(crate) fn advance(&mut self, py: Python<'_>) -> PyResult<Option<Py<Batch>>> {
        self.release_current(py);
        if self.closed || self.finished {
            return Ok(None);
        }
        let batch = self.take(py, self.next)?;
        self.current = batch.as_ref().map(|b| b.clone_ref(py));
        Ok(batch)
    }

    /// Leaves the ring, recording whether this consumer took every batch.
    pub(crate) fn finish(&mut self, py: Python<'_>) {
        if self.closed {
            return;
        }
        self.release_current(py);
        if !self.finished {
            // Past the end, a zero-timeout probe reports the end at once.
            let core = self.core.clone();
            let next = self.next;
            if let Ok(core::Acquisition::Done) = py.detach(|| core.acquire_timeout(next, Duration::ZERO)) {
                self.finished = true;
            }
        }
        self.core.leave_consumer(self.rank, self.finished);
        self.local.close(self.entry);
        self.closed = true;
    }
}

#[pymethods]
impl Consumer {
    fn __repr__(&self) -> String {
        format!("Consumer(rank={}, next_batch={}, finished={}, closed={})", self.rank, self.next,
                if self.finished { "True" } else { "False" }, if self.closed { "True" } else { "False" })
    }
    /// The rank this consumer takes batches for.
    #[getter]
    fn rank(&self) -> u32 {
        self.rank
    }
    /// The next batch iteration takes.
    #[getter]
    fn next_batch(&self) -> u64 {
        self.next
    }
    /// Whether every batch of the rank was taken.
    #[getter]
    fn finished(&self) -> bool {
        self.finished
    }
    /// Iterator protocol: a consumer is its own iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    /// Advances to the rank's next batch, or stops at the end of the run.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<Batch>> {
        self.advance(py)?.ok_or_else(|| PyStopIteration::new_err(()))
    }
    /// Batch `batch` of this rank, or None past the end of the run.
    fn wait(&mut self, py: Python<'_>, batch: u64) -> PyResult<Option<Py<Batch>>> {
        let b = self.take(py, batch)?;
        if b.is_none() {
            self.local.update(self.entry, true);
        }
        Ok(b)
    }
    /// Leaves the ring as a consumer, recording whether the rank took every
    /// batch; a producer waiting for a missing batch then fails instead of
    /// blocking forever.
    fn close(&mut self, py: Python<'_>) {
        self.finish(py);
    }
    /// Entering a consumer leaves it open; exiting closes it.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __exit__(&mut self, py: Python<'_>, _ty: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _tb: &Bound<'_, PyAny>) -> bool {
        self.finish(py);
        false
    }
}

impl Drop for Consumer {
    fn drop(&mut self) {
        if !self.closed {
            self.core.leave_consumer(self.rank, self.finished);
            self.local.close(self.entry);
        }
    }
}

/// Adds the ring, layout, batch and consumer classes to the module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RingLayout>()?;
    m.add_class::<Ring>()?;
    m.add_class::<Batch>()?;
    m.add_class::<LeaseObserver>()?;
    m.add_class::<Consumer>()?;
    Ok(())
}
