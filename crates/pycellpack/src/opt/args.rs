//! `PlanOptions` and `CostModel` as Python passes them (frozen dataclasses
//! in `pycellpack.loader.opt`), read by attribute and converted field by field to
//! their native counterparts, with the same names on both sides.
//!
//! Parsing validates enum strings and output coverage here, so a bad option
//! fails at the Python call that supplied it instead of inside a compiler.

use cellpack::opt::{CostModel, InputCost, OutputCost, PlanObjective, PlanOptions, ProcessCost, RowCost, RowPolicy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyMapping;

/// Per-input decode costs of one cost model.
#[derive(FromPyObject)]
struct InputCostArg {
    decode: f64,
    decode_element: f64,
    cold: f64,
}

/// Row gather and admission costs.
#[derive(FromPyObject)]
struct RowCostArg {
    gather: f64,
    admit: f64,
}

/// One point of a process function's per-element cost curve.
#[derive(FromPyObject)]
struct ProcessCostArg {
    max_elements: u32,
    process: f64,
    resolve: f64,
}

/// Costs of one output: mapping, filling, process calls and their curve.
#[derive(FromPyObject)]
struct OutputCostArg {
    process: f64,
    map_element: f64,
    fill_element: f64,
    resolve: f64,
    resolve_element: f64,
    curve: Vec<ProcessCostArg>,
}

impl OutputCostArg {
    /// The native output cost.
    fn native(self) -> OutputCost {
        OutputCost {
            process: self.process,
            map_element: self.map_element,
            fill_element: self.fill_element,
            resolve: self.resolve,
            resolve_element: self.resolve_element,
            curve: self.curve.into_iter().map(|b| ProcessCost { max_elements: b.max_elements, process: b.process, resolve: b.resolve }).collect(),
        }
    }
}

/// A cost model as Python passes it, with outputs keyed by name or position.
#[derive(FromPyObject)]
pub(crate) struct CostModelArg<'py> {
    cores: f64,
    iops: f64,
    bandwidth: f64,
    memory_bandwidth: f64,
    serial: f64,
    bucket: u64,
    access: f64,
    request: f64,
    compile: f64,
    input: InputCostArg,
    rows: RowCostArg,
    outputs: Bound<'py, PyMapping>,
    convert: Vec<f64>,
}

impl CostModelArg<'_> {
    /// The native model, its outputs in the order of `names` (the loader's
    /// output names); a model keys each output by its name or position.
    pub(crate) fn native(self, names: &[String]) -> PyResult<CostModel> {
        let outputs = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let cost = match self.outputs.get_item(name) {
                    Ok(c) => c,
                    Err(_) => self.outputs.get_item(i).map_err(|_| PyValueError::new_err(format!("the cost model has no costs for output {name:?}")))?,
                };
                Ok(cost.extract::<OutputCostArg>()?.native())
            })
            .collect::<PyResult<Vec<_>>>()?;
        if self.outputs.len()? != names.len() {
            return Err(PyValueError::new_err("the cost model's outputs must be the loader's"));
        }
        let convert = self.convert.try_into().map_err(|_| PyValueError::new_err("one conversion cost per class"))?;
        Ok(CostModel {
            cores: self.cores,
            iops: self.iops,
            bandwidth: self.bandwidth,
            memory_bandwidth: self.memory_bandwidth,
            serial: self.serial,
            bucket: self.bucket,
            access: self.access,
            request: self.request,
            compile: self.compile,
            input: InputCost { decode: self.input.decode, decode_element: self.input.decode_element, cold: self.input.cold },
            rows: RowCost { gather: self.rows.gather, admit: self.rows.admit },
            outputs,
            convert,
        })
    }
}

/// `PlanOptions` whose `gap` and `threads` the loader has resolved.
#[derive(FromPyObject)]
pub(crate) struct PlanOptionsArg<'py> {
    cache_bytes: usize,
    gap: u64,
    threads: usize,
    cpus: Vec<usize>,
    objective: String,
    cost: Option<CostModelArg<'py>>,
    rows: bool,
    row_policy: String,
    block_pages: Option<u32>,
    horizon: Option<u64>,
    chunk: Option<u64>,
}

impl PlanOptionsArg<'_> {
    /// The native options, resolving the cost model against `names`.
    pub(crate) fn native(self, names: &[String]) -> PyResult<PlanOptions> {
        let mut o = PlanOptions::new(self.cache_bytes, self.gap);
        o.threads = self.threads;
        o.cpus = self.cpus;
        o.objective = match self.objective.as_str() {
            "misses" => PlanObjective::Misses,
            "time" => PlanObjective::Time,
            _ => return Err(PyValueError::new_err("objective must be 'misses' or 'time'")),
        };
        o.rows = self.rows;
        o.row_policy = match self.row_policy.as_str() {
            "incremental" => RowPolicy::Incremental,
            "global" => RowPolicy::Global,
            _ => return Err(PyValueError::new_err("row_policy must be 'incremental' or 'global'")),
        };
        o.block_pages = self.block_pages;
        o.horizon = self.horizon;
        o.chunk = self.chunk;
        o.cost = self.cost.map(|c| c.native(names)).transpose()?;
        Ok(o)
    }
}
