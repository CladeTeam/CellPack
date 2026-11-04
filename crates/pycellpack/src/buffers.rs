//! Validation and strong owners for arrays borrowed by native work.
//!
//! Native code receives raw pointers and lengths, so every array is checked
//! once here: numeric native-endian dtype, C-contiguous layout, alignment,
//! writability for outputs, and a live owner that keeps the storage from
//! moving or being freed during the call. Shapes up to four dimensions are
//! stored inline to keep the validation path allocation-free.

use cellpack::format::Dtype;
use cellpack::scans::{Indices, Offsets};
use numpy::{PyUntypedArray, PyUntypedArrayMethods, PyArrayDescrMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A validated array's shape: inline for up to four axes, boxed otherwise.
#[derive(Debug)]
pub(crate) enum Shape {
    Inline { dims: [usize; 4], len: usize },
    Heap(Vec<usize>),
}

impl Shape {
    fn new(shape: &[usize]) -> Self {
        if shape.len() <= 4 {
            let mut dims = [0; 4];
            dims[..shape.len()].copy_from_slice(shape);
            Self::Inline { dims, len: shape.len() }
        } else {
            Self::Heap(shape.to_vec())
        }
    }
}

impl std::ops::Deref for Shape {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        match self {
            Self::Inline { dims, len } => &dims[..*len],
            Self::Heap(v) => v,
        }
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<const N: usize> PartialEq<[usize; N]> for Shape {
    fn eq(&self, other: &[usize; N]) -> bool {
        **self == other[..]
    }
}

impl PartialEq<&[usize]> for Shape {
    fn eq(&self, other: &&[usize]) -> bool {
        &**self == *other
    }
}

/// A NumPy array validated for one native call: its owner (which keeps the
/// storage alive), data pointer, byte length, shape and CellPack dtype.
pub(crate) struct Buffer {
    pub owner: Py<PyAny>,
    pub ptr: usize,
    pub len: usize,
    pub shape: Shape,
    pub dtype: Dtype,
}

impl Buffer {
    /// Validates `obj` as a native numeric array. `writable` requires the
    /// array to be writable; inputs reject implicit conversions so borrowed
    /// bytes always match the dtype the caller sees.
    pub fn get(obj: &Bound<'_, PyAny>, writable: bool) -> PyResult<Self> {
        let a = obj.cast::<PyUntypedArray>().map_err(|_| {
            PyTypeError::new_err("expected a numpy.ndarray; implicit conversion is disabled")
        })?;
        let d = a.dtype();
        let kind = d.kind();
        let width = d.itemsize();
        let dtype = match (kind, width) {
            (b'u', 1) => Dtype::U8,
            (b'u', 2) => Dtype::U16,
            (b'u', 4) => Dtype::U32,
            (b'u', 8) => Dtype::U64,
            (b'i', 1) => Dtype::I8,
            (b'i', 2) => Dtype::I16,
            (b'i', 4) => Dtype::I32,
            (b'i', 8) => Dtype::I64,
            (b'f', 4) => Dtype::F32,
            (b'f', 8) => Dtype::F64,
            _ => return Err(PyTypeError::new_err("unsupported numeric dtype")),
        };
        if !d.is_native_byteorder().unwrap_or(true) || !cfg!(target_endian = "little") {
            return Err(PyValueError::new_err(
                "native little-endian arrays required",
            ));
        }
        if !a.is_c_contiguous() || !a.is_aligned() {
            return Err(PyValueError::new_err(
                "C-contiguous, dtype-aligned arrays required",
            ));
        }
        // SAFETY: a is a live NumPy array; only immutable descriptor fields are read.
        let raw = unsafe { &*a.as_array_ptr() };
        if writable && raw.flags & numpy::npyffi::NPY_ARRAY_WRITEABLE == 0 {
            return Err(PyValueError::new_err("output array is not writable"));
        }
        let len = a
            .len()
            .checked_mul(width)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| PyValueError::new_err("array too large"))?;
        let ptr = raw.data as usize;
        if ptr == 0 || ptr.checked_add(len).is_none() {
            return Err(PyValueError::new_err("invalid array address"));
        }
        Ok(Self {
            owner: obj.clone().unbind(),
            ptr,
            len,
            shape: Shape::new(a.shape()),
            dtype,
        })
    }
    /// Requires an exact dtype and shape.
    pub fn expect(&self, dtype: Dtype, shape: &[usize]) -> PyResult<()> {
        if self.dtype != dtype || self.shape != shape {
            return Err(PyValueError::new_err(format!(
                "expected {dtype:?} array with shape {shape:?}"
            )));
        }
        Ok(())
    }
    /// The buffer as a `T` slice.
    ///
    /// # Safety
    /// `T` must be the buffer's dtype; no concurrent mutation; the owner and any exporter must remain stable.
    pub unsafe fn slice<T>(&self) -> &[T] {
        let n = self.len / std::mem::size_of::<T>();
        let p = if n == 0 {
            std::ptr::NonNull::<T>::dangling().as_ptr()
        } else {
            self.ptr as *const T
        };
        // SAFETY: caller chooses a matching type over the validated contiguous array.
        unsafe { std::slice::from_raw_parts(p, n) }
    }
    /// The buffer as a mutable `T` slice.
    ///
    /// # Safety
    /// `T` must be the buffer's dtype; the call owns exclusive access to this nonoverlapping output region.
    pub unsafe fn slice_mut<T>(&mut self) -> &mut [T] {
        let n = self.len / std::mem::size_of::<T>();
        let p = if n == 0 {
            std::ptr::NonNull::<T>::dangling().as_ptr()
        } else {
            self.ptr as *mut T
        };
        // SAFETY: caller chooses a matching type and ensures exclusive access.
        unsafe { std::slice::from_raw_parts_mut(p, n) }
    }
    /// The buffer as native CSR indices, dispatching on its integer dtype.
    ///
    /// # Safety
    /// The buffer remains immutable through the returned view's use.
    pub unsafe fn indices(&self) -> PyResult<Indices<'_>> {
        // SAFETY: every arm selects the array's validated integer type.
        Ok(unsafe {
            match self.dtype {
                Dtype::U16 => Indices::U16(self.slice()),
                Dtype::U32 => Indices::U32(self.slice()),
                Dtype::U64 => Indices::U64(self.slice()),
                Dtype::I32 => Indices::I32(self.slice()),
                Dtype::I64 => Indices::I64(self.slice()),
                _ => return Err(PyTypeError::new_err("indices require u16/u32/u64/i32/i64")),
            }
        })
    }
    /// The buffer as native CSR offsets, dispatching on its integer dtype.
    ///
    /// # Safety
    /// The buffer remains immutable through the returned view's use.
    pub unsafe fn offsets(&self) -> PyResult<Offsets<'_>> {
        // SAFETY: every arm selects the array's validated integer type.
        Ok(unsafe {
            match self.dtype {
                Dtype::U32 => Offsets::U32(self.slice()),
                Dtype::U64 => Offsets::U64(self.slice()),
                Dtype::I32 => Offsets::I32(self.slice()),
                Dtype::I64 => Offsets::I64(self.slice()),
                _ => return Err(PyTypeError::new_err("indptr requires u32/u64/i32/i64")),
            }
        })
    }
}

/// A validated byte vector (input only).
pub(crate) fn bytes(obj: &Bound<'_, PyAny>) -> PyResult<Buffer> {
    let b = Buffer::get(obj, false)?;
    b.expect(Dtype::U8, &[b.len])?;
    Ok(b)
}

/// Rejects any pair of byte regions that overlaps, so an output cannot
/// alias an input or another output of the same native call.
pub(crate) fn disjoint(buffers: &[&Buffer]) -> PyResult<()> {
    for (i, a) in buffers.iter().enumerate() {
        for b in &buffers[i + 1..] {
            if a.len != 0 && b.len != 0 && a.ptr < b.ptr + b.len && b.ptr < a.ptr + a.len {
                return Err(PyValueError::new_err("array memory regions overlap"));
            }
        }
    }
    Ok(())
}

/// The Python dtype name of a CellPack dtype.
pub(crate) fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::U8 => "uint8",
        Dtype::U16 => "uint16",
        Dtype::U32 => "uint32",
        Dtype::U64 => "uint64",
        Dtype::I8 => "int8",
        Dtype::I16 => "int16",
        Dtype::I32 => "int32",
        Dtype::I64 => "int64",
        Dtype::F32 => "float32",
        Dtype::F64 => "float64",
    }
}

/// Binds `$name` to the Rust type of `$dtype` for the duration of `$body`.
macro_rules! with_dtype {
    ($dtype:expr, $name:ident, $body:expr) => {{
        use cellpack::format::Dtype;
        match $dtype {
            Dtype::U8 => {
                type $name = u8;
                $body
            }
            Dtype::U16 => {
                type $name = u16;
                $body
            }
            Dtype::U32 => {
                type $name = u32;
                $body
            }
            Dtype::U64 => {
                type $name = u64;
                $body
            }
            Dtype::I8 => {
                type $name = i8;
                $body
            }
            Dtype::I16 => {
                type $name = i16;
                $body
            }
            Dtype::I32 => {
                type $name = i32;
                $body
            }
            Dtype::I64 => {
                type $name = i64;
                $body
            }
            Dtype::F32 => {
                type $name = f32;
                $body
            }
            Dtype::F64 => {
                type $name = f64;
                $body
            }
        }
    }};
}
pub(crate) use with_dtype;

/// Parses a Python dtype name into a CellPack dtype.
pub(crate) fn dtype_from_name(name: &str) -> PyResult<Dtype> {
    match name {
        "uint8" => Ok(Dtype::U8),
        "uint16" => Ok(Dtype::U16),
        "uint32" => Ok(Dtype::U32),
        "uint64" => Ok(Dtype::U64),
        "int8" => Ok(Dtype::I8),
        "int16" => Ok(Dtype::I16),
        "int32" => Ok(Dtype::I32),
        "int64" => Ok(Dtype::I64),
        "float32" => Ok(Dtype::F32),
        "float64" => Ok(Dtype::F64),
        _ => Err(PyTypeError::new_err("unsupported numeric dtype")),
    }
}
