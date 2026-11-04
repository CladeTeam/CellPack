//! The loads an access puts on each resource, from a cost model: reading a
//! record, decoding it, delivering it to each schema, and gathering or
//! admitting a cached row. The compiler values a decision by the loads it
//! removes at the current [`Prices`]; replays add the loads up and time
//! them. The routes priced mirror the loader's worker (`loader::exec`).

use crate::loader::catalog::CellId;
use crate::loader::convert::{Conversion, DTYPES};
use crate::loader::schema::Graph;
use crate::opt::model::{CostModel, Loads, Prices};
use crate::opt::records::{Record, Records};

/// Per-schema prices of one compilation or replay.
pub(crate) struct Pricing<'a> {
    /// The graph the accesses are delivered under.
    pub graph: &'a Graph,
    /// The records of the cells it delivers.
    pub records: &'a Records,
    /// The model priced from; without one every price is zero.
    pub cost: Option<&'a CostModel>,
    /// Schemas of `graph`: a position's schema is `position % schemas`.
    pub schemas: u64,
    /// Bytes of one page.
    page: f64,
    /// Core-seconds of every access: its bookkeeping, and the online
    /// compiler's share where it runs alongside the workers.
    access: f64,
    /// Per schema and stored dtype index: core-seconds per element mapped to
    /// the schema's output dims, conversion included.
    elem: Vec<[f64; 10]>,
    /// Per schema: core-seconds of filling a dense row with the default,
    /// already multiplied by the output dims.
    fill: Vec<f64>,
    /// Per schema: core-seconds of one process call (zero for the built-in
    /// output).
    process: Vec<f64>,
    /// Per schema: core-seconds a partial process call adds.
    resolve: Vec<f64>,
    /// Per schema: the same, per stored element of the partial row.
    resolve_element: Vec<f64>,
    /// Per schema: size bands replacing `process` and `resolve`; empty for
    /// the built-in output and for models without bands.
    curves: Vec<&'a [crate::opt::ProcessCost]>,
    /// Per schema: whether it is the built-in output.
    builtin: Vec<bool>,
    /// Per schema: whether its process function maps rows partially.
    partial: Vec<bool>,
    /// Per schema and chunk of its panel: what the built-in output's dense
    /// delivery maps or zeroes; empty for a schema with a process function.
    dense_work: Vec<Vec<DenseWork>>,
    /// Per schema: output dims.
    out_dims: Vec<u64>,
}

/// A chunk's dense delivery, from the panel's column map.
struct DenseWork {
    /// The chunk has a column map for this schema: its stored elements are
    /// mapped to output dims.
    mapped: bool,
    /// Present dims a sparse row must zero when it is expanded into a dense
    /// row with a nonzero default.
    zeroed: u32,
}

impl<'a> Pricing<'a> {
    /// `online`: the compiler runs alongside the workers, so its per-access
    /// cost joins the access's. Without a cost model every price is zero.
    pub fn new(graph: &'a Graph, records: &'a Records, cost: Option<&'a CostModel>, online: bool) -> Self {
        let schemas = &graph.schemas;
        let (mut elem, mut fill, mut process) = (vec![[0.0; 10]; schemas.len()], vec![0.0; schemas.len()], vec![0.0; schemas.len()]);
        if let Some(c) = cost {
            for (k, schema) in schemas.iter().enumerate() {
                let o = &c.outputs[k];
                for (i, &d) in DTYPES.iter().enumerate() {
                    elem[k][i] = o.map_element + if crate::loader::allowed(d, schema.dtype()) { c.convert[Conversion::of(d, schema.dtype()) as usize] } else { 0.0 };
                }
                fill[k] = o.fill_element * schema.out_dim() as f64;
                process[k] = if schema.is_builtin() { 0.0 } else { o.process };
            }
        }
        Self {
            graph,
            records,
            cost,
            schemas: schemas.len() as u64,
            page: graph.page_size as f64,
            access: cost.map_or(0.0, |c| c.access + if online { c.compile } else { 0.0 }),
            elem,
            fill,
            process,
            resolve: schemas.iter().enumerate().map(|(k, _)| cost.map_or(0.0, |c| c.outputs[k].resolve)).collect(),
            resolve_element: schemas.iter().enumerate().map(|(k, _)| cost.map_or(0.0, |c| c.outputs[k].resolve_element)).collect(),
            curves: schemas.iter().enumerate().map(|(k, s)| if s.is_builtin() { &[][..] } else { cost.map_or(&[][..], |c| c.outputs[k].curve.as_slice()) }).collect(),
            builtin: schemas.iter().map(|s| s.is_builtin()).collect(),
            partial: schemas.iter().map(|s| s.partial_rows()).collect(),
            dense_work: schemas
                .iter()
                .enumerate()
                .map(|(k, schema)| {
                    if !schema.is_builtin() {
                        return Vec::new();
                    }
                    let panel = &graph.panels[graph.tables[k].panel];
                    panel.chunks.iter().enumerate().map(|(j, chunk)| DenseWork { mapped: chunk.is_some(), zeroed: if schema.default_bits() == 0 { 0 } else { panel.present[j] } }).collect()
                })
                .collect(),
            out_dims: schemas.iter().map(|s| s.out_dim() as u64).collect(),
        }
    }

    /// Core-seconds every access takes.
    #[inline]
    pub(crate) fn access(&self) -> f64 {
        self.access
    }

    /// One read request of `pages` pages.
    #[inline]
    pub(crate) fn read(&self, pages: u64) -> Loads {
        Loads { requests: 1.0, bytes: pages as f64 * self.page, core_seconds: self.cost.map_or(0.0, |c| c.request), memory_bytes: 0.0 }
    }

    /// Reading record `r` whole, as a miss does.
    #[inline]
    pub(crate) fn record_read(&self, r: &Record) -> Loads {
        if r.pages == 0 { Loads::default() } else { self.read(r.pages as u64) }
    }

    /// Core-seconds of decoding `r`; `cold`: the share of such decodes whose
    /// chunk is cold (1 or 0 for a known access).
    #[inline]
    pub(crate) fn decode(&self, r: &Record, cold: f64) -> f64 {
        let Some(c) = self.cost else { return 0.0 };
        if r.empty() { 0.0 } else { c.input.decode + cold * c.input.cold + r.elements as f64 * c.input.decode_element }
    }

    /// Gathering `r`'s cached row for a hit.
    #[inline]
    pub(crate) fn gather(&self, r: &Record) -> Loads {
        let b = r.row_bytes as f64;
        Loads { core_seconds: self.cost.map_or(0.0, |c| c.rows.gather) * b, memory_bytes: b, ..Default::default() }
    }

    /// Building `r`'s row and writing it into its frames.
    #[inline]
    pub(crate) fn admit(&self, r: &Record) -> Loads {
        let b = r.row_bytes as f64;
        Loads { core_seconds: self.cost.map_or(0.0, |c| c.rows.admit) * b, memory_bytes: b, ..Default::default() }
    }

    /// What one page hit saves: its bytes, and `rho` requests (requests per
    /// page among missed reads).
    pub(crate) fn page_hit(&self, rho: f64) -> Loads {
        let mut l = self.read(1);
        l.requests = rho;
        l.core_seconds *= rho;
        l
    }

    /// What a row hit of `cell` at an access of schema `k` saves: reading
    /// the record (where its pages would miss: `hit` is the page hit rate)
    /// and decoding it and mapping its decoded form, less gathering and
    /// mapping the row.
    pub(crate) fn row_saving(&self, k: usize, r: &Record, cold: f64, hit: f64) -> Loads {
        let mut l = self.record_read(r).scaled(1.0 - hit);
        l.core_seconds += self.decode(r, cold) + self.deliver_cost(k, r, false) - self.deliver_cost(k, r, true);
        let g = self.gather(r);
        l.core_seconds -= g.core_seconds;
        l.memory_bytes -= g.memory_bytes;
        l
    }

    /// What admitting `r`'s row at an access of schema `k` adds: building
    /// and writing it, and mapping the row instead of the decoded record.
    pub(crate) fn row_admission(&self, k: usize, r: &Record) -> Loads {
        let mut l = self.admit(r);
        l.core_seconds += self.deliver_cost(k, r, true) - self.deliver_cost(k, r, false);
        l
    }

    /// Seconds a row residency of `cell` over `[start, end]` saves at
    /// `end`, less admitting it at `start` unless it `extends` a residency
    /// that ends there (`cold`, `hit`: see [`Pricing::row_saving`]). Zero or
    /// less is never kept.
    #[allow(clippy::too_many_arguments)]
    pub fn row_value(&self, prices: &Prices, start: u64, end: u64, cell: CellId, extends: bool, cold: f64, hit: f64) -> f64 {
        let r = self.records.of(cell);
        if self.cost.is_none() || r.empty() {
            return 0.0;
        }
        let k = |a: u64| (a % self.schemas) as usize;
        prices.value(&self.row_saving(k(end), r, cold, hit)) - if extends { 0.0 } else { prices.value(&self.row_admission(k(start), r)) }
    }

    /// The loads of an access of schema `k` to `r` that reads and decodes
    /// it (`own`: this access does not repeat the previous access's cell; the
    /// others reuse that access's read and decode).
    pub(crate) fn uncached(&self, k: usize, r: &Record, own: bool, cold: f64) -> Loads {
        let mut l = if own { self.record_read(r) } else { Loads::default() };
        l.core_seconds += self.access + self.deliver_cost(k, r, false) + if own { self.decode(r, cold) } else { 0.0 };
        l
    }

    /// Whether schema `k` may map record `r` partially (only a decoded
    /// record delivered as a CSR row; a cached row is mapped whole).
    #[inline]
    pub(crate) fn partial(&self, k: usize, r: &Record) -> bool {
        self.partial[k] && !r.dense() && !r.empty() && self.graph.panels[self.graph.tables[k].panel].total[r.chunk as usize] && !self.graph.dense_delivery(k, r.chunk, false, r.value_encoding(), false)
    }

    /// Whether schema `k` receives record `r` as a dense row (the runtime's
    /// route, see `Graph::dense_delivery`).
    #[inline]
    fn dense_delivery(&self, k: usize, r: &Record, whole: bool) -> bool {
        self.builtin[k] || self.graph.dense_delivery(k, r.chunk, if whole { r.image_dense() } else { r.dense() }, r.value_encoding(), whole)
    }

    /// Whether schema `k` can pass record `r`'s cached row through without
    /// mapping or converting it: the stored type is the output type, the
    /// delivery shape matches the row's own, and the chunk's column map and
    /// gene order are the identity over the schema's output dims.
    fn zero_copy(&self, k: usize, r: &Record) -> bool {
        let schema = &self.graph.schemas[k];
        let tables = &self.graph.tables[k];
        !self.builtin[k]
            && DTYPES[r.dtype as usize] == schema.dtype()
            && self.dense_delivery(k, r, true) == r.image_dense()
            && if r.image_dense() { self.graph.panels[tables.panel].identity[r.chunk as usize] && r.image_elements as usize == schema.out_dim() } else { tables.std_identity }
    }

    /// Whether this delivery of `r` reaches a process function as a partial
    /// row (`whole`: the row is mapped whole, never partial).
    pub(crate) fn uses_partial(&self, k: usize, r: &Record, whole: bool) -> bool {
        !whole && self.partial(k, r)
    }

    /// Core-seconds of converting record `r`'s row to schema `k`'s output
    /// and running its process function. `whole`: the row is mapped whole
    /// (cached, or admitted to the cache).
    #[inline]
    pub(crate) fn deliver_cost(&self, k: usize, r: &Record, whole: bool) -> f64 {
        if self.cost.is_none() {
            return 0.0;
        }
        if self.builtin[k] {
            return self.elem[k][r.dtype as usize] * self.dense_elements(k, r, whole) as f64 + if self.out_dims[k] == 0 { 0.0 } else { self.fill[k] };
        }
        let (mapped, filled) = self.delivery_work(k, r, whole);
        let process = if self.curves[k].is_empty() {
            self.process[k] + if self.uses_partial(k, r, whole) { self.resolve[k] + r.elements as f64 * self.resolve_element[k] } else { 0.0 }
        } else {
            let elements = if self.dense_delivery(k, r, whole) { self.graph.schemas[k].out_dim() as u32 } else if whole { r.image_elements } else { r.elements };
            // The first band whose `max_elements` covers the row.
            let band = self.curves[k][self.curves[k].partition_point(|b| b.max_elements < elements)];
            band.process + if self.uses_partial(k, r, whole) { band.resolve } else { 0.0 }
        };
        self.elem[k][r.dtype as usize] * mapped as f64 + if filled != 0 { self.fill[k] } else { 0.0 } + process
    }

    /// Elements the built-in output's dense delivery touches: the row's own
    /// elements when the chunk's column map covers it, plus the zeros a
    /// sparse row leaves in the present dims of a dense one.
    #[inline]
    fn dense_elements(&self, k: usize, r: &Record, whole: bool) -> u64 {
        let work = &self.dense_work[k][r.chunk as usize];
        let elements = if whole { r.image_elements } else { r.elements };
        let dense = if whole { r.image_dense() } else { r.dense() };
        (if !r.empty() && work.mapped { elements as u64 } else { 0 }) + if dense { 0 } else { work.zeroed as u64 }
    }

    /// Element counts `(mapped, filled)` priced by the model, also exposed by
    /// replay for fitting.
    pub(crate) fn delivery_work(&self, k: usize, r: &Record, whole: bool) -> (u64, u64) {
        if self.builtin[k] {
            return (self.dense_elements(k, r, whole), self.out_dims[k]);
        }
        let schema = &self.graph.schemas[k];
        if whole && self.zero_copy(k, r) {
            return (0, 0);
        }
        if !whole && self.partial(k, r) {
            return (0, 0);
        }
        let panel = &self.graph.panels[self.graph.tables[k].panel];
        let dense = self.dense_delivery(k, r, whole);
        // A chunk without a column map stores nothing this schema maps, so
        // only a dense delivery's zeros are written.
        let mapped = !r.empty() && panel.chunks[r.chunk as usize].is_some();
        // A sparse row in a dense row with a nonzero default first zeroes
        // its present dims.
        let source_dense = if whole { r.image_dense() } else { r.dense() };
        let elements = if whole { r.image_elements } else { r.elements };
        let zeroed = if dense && !source_dense && schema.default_bits() != 0 { panel.present[r.chunk as usize] as u64 } else { 0 };
        (if mapped { elements as u64 } else { 0 } + zeroed, if dense { schema.out_dim() as u64 } else { 0 })
    }
}
