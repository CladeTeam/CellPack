//! The crate's default costs (`cellpack::opt::defaults`) as the record
//! values Python's cost store reads.

use cellpack::opt::defaults as d;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The crate's default costs (`cellpack::opt::defaults`), as record values:
/// `host`, `tiers` (storage values by tier name), `builtin` and `process`
/// output costs, and `bucket`.
#[pyfunction]
pub(super) fn cost_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    let h = d::HOST;
    let host = PyDict::new(py);
    host.set_item("access", h.access)?;
    host.set_item("compile", h.compile)?;
    host.set_item("memory_bandwidth", h.memory_bandwidth)?;
    let input = PyDict::new(py);
    input.set_item("decode", h.input.decode)?;
    input.set_item("decode_element", h.input.decode_element)?;
    input.set_item("cold", h.input.cold)?;
    host.set_item("input", input)?;
    let rows = PyDict::new(py);
    rows.set_item("gather", h.rows.gather)?;
    rows.set_item("admit", h.rows.admit)?;
    host.set_item("rows", rows)?;
    host.set_item("convert", h.convert.to_vec())?;
    out.set_item("host", host)?;
    let tiers = PyDict::new(py);
    for t in d::Tier::ALL {
        let s = d::storage(t);
        let v = PyDict::new(py);
        v.set_item("iops", s.iops)?;
        v.set_item("bandwidth", s.bandwidth)?;
        v.set_item("request", s.request)?;
        v.set_item("serial", s.serial)?;
        tiers.set_item(t.name(), v)?;
    }
    out.set_item("tiers", tiers)?;
    let output = |o: cellpack::opt::OutputCost| -> PyResult<Bound<'_, PyDict>> {
        let v = PyDict::new(py);
        v.set_item("process", o.process)?;
        v.set_item("map_element", o.map_element)?;
        v.set_item("fill_element", o.fill_element)?;
        v.set_item("resolve", o.resolve)?;
        v.set_item("resolve_element", o.resolve_element)?;
        v.set_item("curve", Vec::<(u32, f64, f64)>::new())?;
        Ok(v)
    };
    out.set_item("builtin", output(d::builtin())?)?;
    out.set_item("process", output(d::process())?)?;
    out.set_item("bucket", d::BUCKET)?;
    Ok(out)
}
