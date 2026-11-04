//! Encoding: chunk and record encoders, their output and workspace sizes,
//! page packing, the reusable prepared encoder (`ScansEncoder`), and the
//! options Python passes as a dict.

use cellpack::format::Dtype;
use cellpack::scans as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::tables::{EncodeTables, MetaView};
use super::{Selection, workspace_buffer};
use crate::buffers::{Buffer, bytes, disjoint};
use crate::errors::error;

/// A matrix input as Python passes it: dense `(cells, n_genes)` values, or
/// CSR values with `indptr` and `indices`.
struct Input {
    data: Buffer,
    ip: Option<Buffer>,
    ix: Option<Buffer>,
    genes: u32,
}
impl Input {
    fn new(
        data: &Bound<'_, PyAny>,
        ip: Option<&Bound<'_, PyAny>>,
        ix: Option<&Bound<'_, PyAny>>,
        genes: u32,
    ) -> PyResult<Self> {
        let a = Self {
            data: Buffer::get(data, false)?,
            ip: ip.map(|a| Buffer::get(a, false)).transpose()?,
            ix: ix.map(|a| Buffer::get(a, false)).transpose()?,
            genes,
        };
        if genes == 0 {
            return Err(PyValueError::new_err("n_genes must be positive"));
        }
        if a.ip.is_some() != a.ix.is_some() {
            return Err(PyValueError::new_err(
                "indptr and indices must be supplied together",
            ));
        }
        if let (Some(ip), Some(ix)) = (&a.ip, &a.ix) {
            if ip.shape.len() != 1
                || ix.shape.len() != 1
                || a.data.shape != ix.shape
                || ip.shape[0] == 0
            {
                return Err(PyValueError::new_err("invalid CSR input dimensions"));
            }
            // SAFETY: buffer methods dispatch only matching validated integer dtypes.
            let offsets = unsafe { ip.offsets()? };
            if offsets.get(0).map_err(error)? != 0
                || offsets.get(offsets.len() - 1).map_err(error)? != ix.shape[0] as u64
            {
                return Err(PyValueError::new_err(
                    "CSR indptr must start at zero and end at nnz",
                ));
            }
        } else if a.data.shape.len() != 2 || a.data.shape[1] != genes as usize {
            return Err(PyValueError::new_err(
                "dense input requires shape (cells,n_genes)",
            ));
        }
        Ok(a)
    }
    fn matrix(&self) -> PyResult<core::MatrixRef<'_>> {
        // SAFETY: all buffers are validated and remain immutable during native work.
        unsafe {
            Ok(match (&self.ip, &self.ix) {
                (Some(ip), Some(ix)) => core::CsrView {
                    n_genes: self.genes,
                    value_width: self.data.dtype.width(),
                    indptr: ip.offsets()?,
                    indices: ix.indices()?,
                    data: self.data.slice(),
                }
                .into(),
                _ => core::DenseRef {
                    n_genes: self.genes,
                    value_width: self.data.dtype.width(),
                    data: self.data.slice(),
                }
                .into(),
            })
        }
    }
    fn check_outputs(&self, out: &Buffer, ws: &Buffer) -> PyResult<()> {
        let mut b = [&self.data, out, ws, &self.data, &self.data];
        let mut n = 3;
        if let Some(ip) = &self.ip {
            b[n] = ip; n += 1;
        }
        if let Some(ix) = &self.ix {
            b[n] = ix; n += 1;
        }
        disjoint(&b[..n])
    }
}
/// The encoding options every unset key takes.
pub(crate) fn default_options() -> core::EncodeOptions {
    core::EncodeOptions {
        renumber: true,
        thresholds: core::Thresholds { indices: core::DEFAULT_MU_INDICES, values: core::DEFAULT_MU_VALUES },
        max_lanes: core::DEFAULT_MAX_LANES,
        lane_min: core::DEFAULT_LANE_MIN,
        column_modes: Some(core::DEFAULT_MODE_SHARE),
        layout: core::LayoutChoice::Auto,
        values: core::ValueChoice::Auto,
    }
}

/// Encoding options from Python's `EncodeOptions` dict; missing keys take
/// [`default_options`].
pub(crate) fn options(d: &Bound<'_, PyDict>) -> PyResult<core::EncodeOptions> {
    fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T> {
        match d.get_item(key)? {
            Some(v) => v.extract::<T>().map_err(Into::into),
            None => Ok(default),
        }
    }
    let base = default_options();
    let layout = match d.get_item("layout")? {
        None => base.layout,
        Some(v) => match v.extract::<String>()?.as_str() {
            "auto" => core::LayoutChoice::Auto,
            "sparse" => core::LayoutChoice::Sparse,
            "dense" => core::LayoutChoice::Dense,
            _ => return Err(PyValueError::new_err("layout must be auto, sparse or dense")),
        },
    };
    let values = match d.get_item("values")? {
        None => base.values,
        Some(v) => match v.extract::<String>()?.as_str() {
            "auto" => core::ValueChoice::Auto,
            "raw" => core::ValueChoice::Raw,
            "dict" => core::ValueChoice::Dict,
            "modes" => core::ValueChoice::Modes,
            _ => return Err(PyValueError::new_err("values must be auto, raw, dict or modes")),
        },
    };
    Ok(core::EncodeOptions {
        renumber: get(d, "renumber", base.renumber)?,
        thresholds: core::Thresholds {
            indices: get(d, "indices_threshold", base.thresholds.indices)?,
            values: get(d, "values_threshold", base.thresholds.values)?,
        },
        max_lanes: get(d, "max_lanes", base.max_lanes)?,
        lane_min: get(d, "lane_min", base.lane_min)?,
        column_modes: get(d, "column_modes", base.column_modes)?,
        layout,
        values,
    })
}

#[pyfunction]
#[pyo3(signature = (data, indptr, indices, n_genes, rows, opts, *, output=true, workspace=true))]
#[allow(clippy::too_many_arguments)]
fn scans_encode_sizes(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indptr: Option<&Bound<'_, PyAny>>,
    indices: Option<&Bound<'_, PyAny>>,
    n_genes: u32,
    rows: &Bound<'_, PyAny>,
    opts: &Bound<'_, PyDict>,
    output: bool,
    workspace: bool,
) -> PyResult<(usize, usize)> {
    let rows = Selection::get(rows)?;
    let input = Input::new(data, indptr, indices, n_genes)?;
    let opt = options(opts)?;
    py.detach(|| {
        let m = input.matrix()?;
        if output && workspace {
            let sizes = core::chunk_sizes(m, rows.rows(), &opt).map_err(error)?;
            return Ok((sizes.output_bytes, sizes.workspace_words));
        }
        Ok((
            if output { core::chunk_bound(m, rows.rows(), &opt).map_err(error)? } else { 0 },
            if workspace { core::chunk_workspace_words(m, rows.rows(), &opt).map_err(error)? } else { 0 },
        ))
    })
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn scans_encode_chunk(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indptr: Option<&Bound<'_, PyAny>>,
    indices: Option<&Bound<'_, PyAny>>,
    n_genes: u32,
    rows: &Bound<'_, PyAny>,
    opts: &Bound<'_, PyDict>,
    out: &Bound<'_, PyAny>,
    workspace: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let rows = Selection::get(rows)?;
    let input = Input::new(data, indptr, indices, n_genes)?;
    let opt = options(opts)?;
    let (mut out, mut ws) = (Buffer::get(out, true)?, workspace_buffer(workspace)?);
    out.expect(Dtype::U8, &[out.len])?;
    input.check_outputs(&out, &ws)?;
    rows.check_outputs(&[&out, &ws])?;
    py.detach(|| {
        // SAFETY: matching byte output/u64 scratch and validated immutable matrix inputs.
        unsafe {
            core::encode_chunk(
                input.matrix()?,
                rows.rows(),
                &opt,
                out.slice_mut(),
                ws.slice_mut(),
            )
        }
        .map_err(error)
    })
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn scans_encode_record(
    py: Python<'_>,
    meta: PyRef<'_, MetaView>,
    tables: PyRef<'_, EncodeTables>,
    data: &Bound<'_, PyAny>,
    indptr: Option<&Bound<'_, PyAny>>,
    indices: Option<&Bound<'_, PyAny>>,
    opts: &Bound<'_, PyDict>,
    out: &Bound<'_, PyAny>,
    workspace: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let metadata = meta.inner.clone();
    let input = Input::new(data, indptr, indices, metadata.view().n_dims())?;
    let (mut out, mut ws) = (Buffer::get(out, true)?, workspace_buffer(workspace)?);
    out.expect(Dtype::U8, &[out.len])?;
    input.check_outputs(&out, &ws)?;
    let opt = options(opts)?;
    let encode = &tables.encode;
    py.detach(|| {
        let m = input.matrix()?;
        if m.n_rows() != 1 {
            return Err(PyValueError::new_err("record input must have one row"));
        }
        // SAFETY: disjoint writable byte output/u64 scratch and validated input row.
        unsafe {
            core::encode_record(
                &metadata.view(),
                encode,
                m.row(0).map_err(error)?,
                &opt,
                out.slice_mut(),
                ws.slice_mut(),
            )
        }
        .map_err(error)
    })
}
#[pyfunction]
fn scans_pack(
    py: Python<'_>,
    chunk: &Bound<'_, PyAny>,
    page_size: u32,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    let chunk = bytes(chunk)?;
    // SAFETY: immutable retained byte input.
    let b = unsafe { chunk.slice::<u8>() };
    let mut out = out.map(|a| Buffer::get(a, true)).transpose()?;
    if let Some(out) = &out {
        out.expect(Dtype::U8, &[out.len])?;
        disjoint(&[&chunk, out])?;
    }
    py.detach(|| {
        if let Some(out) = &mut out {
            // SAFETY: disjoint byte output checked above.
            core::pack_chunk(b, page_size, unsafe { out.slice_mut() }).map_err(error)
        } else {
            core::packed_len(b, page_size).map_err(error)
        }
    })
}
#[pyfunction]
fn scans_workspace(
    kind: &str,
    n_genes: u32,
    max_elems: usize,
    value_width: usize,
) -> PyResult<usize> {
    if n_genes == 0 || !matches!(value_width, 1 | 2 | 4 | 8) || max_elems > n_genes as usize {
        return Err(PyValueError::new_err(
            "invalid genes, elements or value width",
        ));
    }
    Ok(match kind {
        "encode" => core::encode_workspace_words(n_genes, max_elems),
        "decode" => core::decode_workspace_bound(n_genes, value_width),
        "record" => core::record_bound(n_genes, value_width, max_elems, true),
        _ => return Err(PyValueError::new_err("unknown size request")),
    })
}

#[pyfunction]
fn scans_record_bound(
    n_genes: u32,
    value_width: usize,
    nnz: usize,
    dense: bool,
) -> PyResult<usize> {
    if n_genes == 0 || nnz > n_genes as usize || !matches!(value_width, 1 | 2 | 4 | 8) {
        return Err(PyValueError::new_err("invalid genes, nnz or width"));
    }
    Ok(core::record_bound(n_genes, value_width, nnz, dense))
}

#[pyclass(module = "pycellpack._cellpack", name = "ScansEncoder")]
struct PreparedEncoder {
    inner: core::PreparedEncoder<'static>,
    _input: Box<Input>,
    _rows: Box<Selection>,
}
#[pymethods]
impl PreparedEncoder {
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PyAny>, indptr: Option<&Bound<'_, PyAny>>, indices: Option<&Bound<'_, PyAny>>, n_genes: u32, rows: &Bound<'_, PyAny>, opts: &Bound<'_, PyDict>) -> PyResult<Self> {
        let data = data.call_method0("copy")?;
        let ip = indptr.map(|x| x.call_method0("copy")).transpose()?;
        let ix = indices.map(|x| x.call_method0("copy")).transpose()?;
        let rows = if rows.is_none() || rows.is_instance_of::<pyo3::types::PyTuple>() { rows.clone() } else { rows.call_method0("copy")? };
        let input = Box::new(Input::new(&data, ip.as_ref(), ix.as_ref(), n_genes)?);
        let rows = Box::new(Selection::get(&rows)?);
        let opts = options(opts)?;
        let inner = py.detach(|| core::PreparedEncoder::new(input.matrix()?, rows.rows(), opts).map_err(error))?;
        // SAFETY: immutable private array snapshots and row storage are retained
        // after inner, never exposed or resized; all public calls borrow self.
        let inner = unsafe { std::mem::transmute::<core::PreparedEncoder<'_>, core::PreparedEncoder<'static>>(inner) };
        Ok(Self { inner, _input: input, _rows: rows })
    }
    fn metadata<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, self.inner.metadata_bytes())
    }
    fn sizes(&self) -> (usize, usize) { let s = self.inner.sizes(); (s.output_bytes, s.workspace_words) }
    fn encode_into(&self, py: Python<'_>, out: &Bound<'_, PyAny>, workspace: &Bound<'_, PyAny>) -> PyResult<usize> {
        let (mut out, mut ws) = (bytes(out)?, workspace_buffer(workspace)?);
        // bytes() validates dtype but output must also be writable.
        out = Buffer::get(out.owner.bind(py), true)?;
        disjoint(&[&out, &ws])?;
        py.detach(|| {
            // SAFETY: exact disjoint writable byte and u64 buffers; input is private.
            unsafe { self.inner.encode_into(out.slice_mut(), ws.slice_mut()) }.map_err(error)
        })
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PreparedEncoder>()?;
    m.add_function(wrap_pyfunction!(scans_encode_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(scans_encode_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(scans_encode_record, m)?)?;
    m.add_function(wrap_pyfunction!(scans_pack, m)?)?;
    m.add_function(wrap_pyfunction!(scans_workspace, m)?)?;
    m.add_function(wrap_pyfunction!(scans_record_bound, m)?)?;
    Ok(())
}
