//! Access sequences for Python: stored (snapshotted from a uint32 array),
//! computed (shuffles, native functions) and composed (replicated or offset
//! per schema). Each remembers how it was built, so sequences pickle to
//! other processes, except those naming native code of this one.

use std::sync::Arc;

use cellpack::format::Dtype;
use cellpack::loader as core;
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::buffers::Buffer;
use crate::errors::loader_error;

/// How a sequence was built: enough to rebuild it in another process, unless
/// it names native code of this one.
enum Recipe {
    Shuffle { cells: u64, epochs: u64, seed: u64, block: u64, group: u64 },
    Subset { cells: Arc<[u32]>, epochs: u64, seed: u64, block: u64, group: u64 },
    Array(Arc<[u32]>),
    Replicate(Py<Accesses>, usize),
    Offset(Py<Accesses>, Vec<u32>),
    Slice(Py<Accesses>, u64, u64),
    Function,
}

/// An access sequence: stored, computed or composed, plus the recipe that
/// pickles it. The native sequence may be shared with any number of runs;
/// the recipe only serves reconstruction.
#[pyclass(frozen, module = "pycellpack._cellpack", name = "Accesses")]
pub(crate) struct Accesses {
    pub inner: Arc<dyn core::Accesses>,
    recipe: Recipe,
}

/// A native sequence that keeps its Python owner alive: the owner holds the
/// callback's code and context for every run using the sequence.
struct FunctionAccesses {
    inner: Arc<dyn core::Accesses>,
    _owner: Option<Py<PyAny>>,
}

impl core::Accesses for FunctionAccesses {
    fn len(&self) -> u64 {
        self.inner.len()
    }
    fn fill(&self, start: u64, out: &mut [u32]) -> core::Result<()> {
        self.inner.fill(start, out)
    }
}

/// A copy of a uint32 vector, taken with the GIL released.
pub(crate) fn snapshot(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Arc<[u32]>> {
    let b = Buffer::get(a, false)?;
    if b.dtype != Dtype::U32 || b.shape.len() != 1 {
        return Err(PyTypeError::new_err("accesses are a 1D uint32 array or an Accesses"));
    }
    // SAFETY: the validated vector is retained by `b`; it is copied before
    // the call returns.
    Ok(py.detach(|| Arc::from(unsafe { b.slice::<u32>() })))
}

/// The sequence an `Accesses` or a uint32 array names.
pub(crate) fn sequence(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Arc<dyn core::Accesses>> {
    if let Ok(x) = a.cast::<Accesses>() {
        return Ok(x.get().inner.clone());
    }
    Ok(Arc::new(snapshot(py, a)?))
}

/// An `Accesses` from an `Accesses` or a uint32 array (to compose).
fn accesses(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<Accesses>> {
    if let Ok(x) = a.cast::<Accesses>() {
        return Ok(x.clone().unbind());
    }
    let cells = snapshot(py, a)?;
    Py::new(py, Accesses { inner: Arc::new(cells.clone()), recipe: Recipe::Array(cells) })
}

/// A sequence whose cells pass through `table` (a data source's global
/// cell ids to its stored cells), translated as they are filled.
pub(crate) struct Remapped {
    /// Global cell ids.
    pub inner: Arc<dyn core::Accesses>,
    /// Global cell id → stored cell id.
    pub table: Arc<[u32]>,
}

impl core::Accesses for Remapped {
    fn len(&self) -> u64 {
        self.inner.len()
    }
    fn fill(&self, start: u64, out: &mut [u32]) -> core::Result<()> {
        self.inner.fill(start, out)?;
        for (i, c) in out.iter_mut().enumerate() {
            *c = *self.table.get(*c as usize).ok_or(core::Error::Access { position: start + i as u64, cell: *c as u64 })?;
        }
        Ok(())
    }
}

#[pymethods]
impl Accesses {
    fn __repr__(&self) -> String {
        let kind = match &self.recipe {
            Recipe::Shuffle { cells, epochs, seed, .. } => format!("shuffle, cells={cells}, epochs={epochs}, seed={seed}"),
            Recipe::Subset { cells, epochs, seed, .. } => format!("subset, cells={}, epochs={epochs}, seed={seed}", cells.len()),
            Recipe::Array(_) => "array".into(),
            Recipe::Replicate(_, schemas) => format!("replicate, schemas={schemas}"),
            Recipe::Offset(_, offsets) => format!("offset, schemas={}", offsets.len()),
            Recipe::Slice(_, start, count) => format!("slice, start={start}, count={count}"),
            Recipe::Function => "function".into(),
        };
        format!("Accesses({kind}, length={})", self.inner.len())
    }
    /// `epochs` passes over cells `0..cells`, each in its own pseudorandom
    /// order; with `block > 1`, blocks of `block` consecutive cells are
    /// shuffled, and entries within each group of `group` blocks.
    #[staticmethod]
    #[pyo3(signature = (cells, epochs, seed = 0, block = 1, group = 1))]
    fn shuffle(cells: u64, epochs: u64, seed: u64, block: u64, group: u64) -> PyResult<Self> {
        let s = core::Shuffle::new(cells, epochs, seed).and_then(|s| s.blocks(block, group)).map_err(loader_error)?;
        Ok(Self { inner: Arc::new(s), recipe: Recipe::Shuffle { cells, epochs, seed, block, group } })
    }
    /// Like `shuffle`, over the entries of a uint32 array of cells.
    #[staticmethod]
    #[pyo3(signature = (cells, epochs, seed = 0, block = 1, group = 1))]
    fn subset(py: Python<'_>, cells: &Bound<'_, PyAny>, epochs: u64, seed: u64, block: u64, group: u64) -> PyResult<Self> {
        let cells = snapshot(py, cells)?;
        let s = core::Shuffle::subset(cells.to_vec(), epochs, seed).and_then(|s| s.blocks(block, group)).map_err(loader_error)?;
        Ok(Self { inner: Arc::new(s), recipe: Recipe::Subset { cells, epochs, seed, block, group } })
    }
    /// A sequence of `length` accesses computed by a native function at
    /// `address`: `int32_t f(void *context, uint64_t start, uint64_t count,
    /// uint32_t *out)`, returning zero on success. It must be callable from
    /// any thread, concurrently, and depend on the access only; `owner`
    /// keeps the code and `context` alive through every run using it. Such
    /// a sequence names this process's code and does not pickle.
    #[staticmethod]
    #[pyo3(signature = (address, context, length, *, owner = None))]
    fn function(address: usize, context: usize, length: u64, owner: Option<Py<PyAny>>) -> PyResult<Self> {
        if address == 0 {
            return Err(PyValueError::new_err("null access function"));
        }
        // SAFETY: the caller supplies a function of this signature and the
        // documented contract (thread-safe, pure, context alive).
        let f: core::AccessFn = unsafe { std::mem::transmute::<usize, core::AccessFn>(address) };
        // SAFETY: as above.
        let inner = unsafe { core::ExternAccesses::new(f, context as *mut std::ffi::c_void, length) };
        Ok(Self { inner: Arc::new(FunctionAccesses { inner: Arc::new(inner), _owner: owner }), recipe: Recipe::Function })
    }
    /// Bind a callback filling dataset indices (uint32) and local rows (uint64).
    #[staticmethod]
    #[pyo3(signature = (address, context, length, counts, *, owner = None))]
    fn corpus_function(address: usize, context: usize, length: u64, counts: Vec<u64>, owner: Option<Py<PyAny>>) -> PyResult<Self> {
        if address == 0 {
            return Err(PyValueError::new_err("null corpus access function"));
        }
        // SAFETY: the caller supplies a concurrent function with the documented C ABI.
        let f: core::CorpusAccessFn = unsafe { std::mem::transmute::<usize, core::CorpusAccessFn>(address) };
        // SAFETY: its callback and context are retained by owner for every use.
        let inner = unsafe { core::ExternCorpusAccesses::new(f, context as *mut std::ffi::c_void, length, &counts) }.map_err(loader_error)?;
        Ok(Self { inner: Arc::new(FunctionAccesses { inner: Arc::new(inner), _owner: owner }), recipe: Recipe::Function })
    }
    /// Restrict a sequence to a bounded interval, preserving native generation.
    #[pyo3(signature = (start = 0, count = None))]
    fn slice(slf: &Bound<'_, Self>, start: u64, count: Option<u64>) -> PyResult<Self> {
        let count = count.unwrap_or_else(|| slf.get().inner.len().saturating_sub(start));
        let inner = core::SliceAccesses::new(slf.get().inner.clone(), start, count).map_err(loader_error)?;
        Ok(Self { inner: Arc::new(inner), recipe: Recipe::Slice(slf.clone().unbind(), start, count) })
    }
    /// A stored sequence: a copy of a uint32 array.
    #[staticmethod]
    fn array(py: Python<'_>, cells: &Bound<'_, PyAny>) -> PyResult<Self> {
        let cells = snapshot(py, cells)?;
        Ok(Self { inner: Arc::new(cells.clone()), recipe: Recipe::Array(cells) })
    }
    /// Every one of `schemas` schemas of a position loads the position's
    /// cell (`positions`: an `Accesses` or a uint32 array, one cell per
    /// position).
    #[staticmethod]
    fn replicate(py: Python<'_>, positions: &Bound<'_, PyAny>, schemas: usize) -> PyResult<Self> {
        let positions = accesses(py, positions)?;
        let inner = core::Replicate::new(positions.get().inner.clone(), schemas).map_err(loader_error)?;
        Ok(Self { inner: Arc::new(inner), recipe: Recipe::Replicate(positions, schemas) })
    }
    /// Schema `s` of a position loads the position's cell plus `offsets[s]`.
    #[staticmethod]
    fn offset(py: Python<'_>, positions: &Bound<'_, PyAny>, offsets: Vec<u32>) -> PyResult<Self> {
        let positions = accesses(py, positions)?;
        let inner = core::Offset::new(positions.get().inner.clone(), offsets.clone()).map_err(loader_error)?;
        Ok(Self { inner: Arc::new(inner), recipe: Recipe::Offset(positions, offsets) })
    }
    /// The number of accesses.
    fn __len__(&self) -> usize {
        self.inner.len() as usize
    }
    /// The cells of accesses `start..start + count` (default: to the end),
    /// into `out` (a uint32 vector of `count`) when given.
    #[pyo3(signature = (start = 0, count = None, *, out = None))]
    fn fill<'py>(&self, py: Python<'py>, start: u64, count: Option<u64>, out: Option<Bound<'py, PyArray1<u32>>>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let len = self.inner.len();
        let count = match (count, &out) {
            (Some(n), _) => n,
            (None, Some(o)) => o.len() as u64,
            (None, None) => len.saturating_sub(start),
        };
        if start.checked_add(count).is_none_or(|end| end > len) {
            return Err(PyValueError::new_err("access range beyond the sequence"));
        }
        let out = match out {
            Some(o) if o.len() as u64 == count && o.is_c_contiguous() => o,
            Some(_) => return Err(PyValueError::new_err("out must be a C-contiguous uint32 vector of `count` accesses")),
            None => PyArray1::zeros(py, count as usize, false),
        };
        {
            let mut view = out.try_readwrite()?;
            let slice = view.as_slice_mut()?;
            let inner = self.inner.clone();
            py.detach(|| inner.fill(start, slice)).map_err(loader_error)?;
        }
        Ok(out)
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let cls = slf.get_type();
        Ok(match &slf.get().recipe {
            Recipe::Shuffle { cells, epochs, seed, block, group } => (cls.getattr("shuffle")?, (*cells, *epochs, *seed, *block, *group).into_pyobject(py)?),
            Recipe::Subset { cells, epochs, seed, block, group } => (cls.getattr("subset")?, (PyArray1::from_slice(py, cells), *epochs, *seed, *block, *group).into_pyobject(py)?),
            Recipe::Array(cells) => (cls.getattr("array")?, (PyArray1::from_slice(py, cells),).into_pyobject(py)?),
            Recipe::Replicate(inner, schemas) => (cls.getattr("replicate")?, (inner.clone_ref(py), *schemas).into_pyobject(py)?),
            Recipe::Offset(inner, offsets) => (cls.getattr("offset")?, (inner.clone_ref(py), offsets.clone()).into_pyobject(py)?),
            Recipe::Slice(inner, start, count) => (inner.bind(py).getattr("slice")?, (*start, *count).into_pyobject(py)?),
            Recipe::Function => return Err(PyTypeError::new_err("a function sequence names native code of this process and cannot be pickled")),
        })
    }
}

/// Adds the `Accesses` class to the module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Accesses>()
}
