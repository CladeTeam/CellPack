//! What a plan is compiled for: the budget, the reuse gap and the cost
//! model of the loader on one machine.
//!
//! The cost model describes four resources and what each unit of work takes
//! from them. Over a window of `bucket` accesses with loads `R` read
//! requests, `B` bytes read, `C` core-seconds of worker work and `M` bytes
//! of cached-row memory traffic, the window takes
//!
//! ```text
//! d = max(R / iops, B / bandwidth)          device
//! c = C / cores                             workers
//! T = max(max(d, c) + serial · min(d, c),   M / memory_bandwidth)
//! ```
//!
//! `T` is convex in the loads, so the marginal time of one more unit of each
//! load at the expected loads ([`Prices`]) prices every decision the same
//! way: a request against a byte against a core-second. Where bandwidth
//! binds, a page hit is worth its bytes and the page plan is the serial
//! optimum of misses; where requests bind, merging reads pays; where the
//! workers bind, only work removed from them (rows) pays.

use crate::loader::convert::CONVERSIONS;
use crate::loader::error::{Error, Result};
use crate::loader::schema::Graph;
use crate::opt::plan::Reader;

/// Default compiler threads: a fixed count, not derived from the machine.
pub const DEFAULT_COMPILE_THREADS: usize = 16;

/// What a compilation optimizes for: the budget, the reuse gap, the
/// objective and the cost model of one loader on one machine.
#[derive(Clone, Debug)]
pub struct PlanOptions {
    /// Cache bytes for planned frames: `cache_bytes / page_size` frames, the
    /// budget the shards split.
    pub cache_bytes: usize,
    /// The reuse gap in slots: the loader's execution window may not exceed
    /// it, and every residency reserves its frames from the same gap in
    /// accesses (this many slots times a slot's accesses) before its
    /// admission.
    pub gap: u64,
    /// Compiler threads. Decisions depend on the count only through the shard
    /// and part split, not on the order the threads take work in.
    pub threads: usize,
    /// Optional compiler CPU set; empty inherits the calling thread's set.
    pub cpus: Vec<usize>,
    /// Whether plans minimize modeled time or page misses.
    pub objective: PlanObjective,
    /// The machine's costs; without one, every page hit counts the same and
    /// only pages are cached.
    pub cost: Option<CostModel>,
    /// Also cache rows (page+row mode; needs a cost model, and is off under
    /// [`PlanObjective::Misses`]).
    pub rows: bool,
    /// How page+row compilations choose rows.
    pub row_policy: RowPolicy,
    /// Merge admissions of adjacent pages into reads of at most this many
    /// pages where the requests saved are worth more than the frames held
    /// longer (needs a cost model).
    pub block_pages: Option<u32>,
    /// The longest reuse interval the compiler considers, in slots: how far
    /// past the frozen plan it optimizes, and with it the compiler's memory
    /// (about 30 bytes per access of the horizon). Defaults online to twice
    /// the source's cells in accesses (two epochs when an epoch visits every
    /// cell once), which sees every reuse of epoch shuffles. Offline
    /// compilations see the whole sequence.
    pub horizon: Option<u64>,
    /// Accesses the compiler advances, freezes and publishes at a time
    /// (default [`DEFAULT_CHUNK`]).
    pub chunk: Option<u64>,
}

/// Default compilation chunk in accesses.
pub const DEFAULT_CHUNK: u64 = 16384;

/// What a compilation minimizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanObjective {
    /// Page misses only: rows are off, and the page layer keeps every reuse
    /// interval its frames fit (the hit-maximizing plan under the reservation
    /// and shard rules).
    Misses,
    /// The shortest modeled time.
    Time,
}

/// Row selection of page+row compilations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowPolicy {
    /// Decide each row interval when its reuse arrives, against a shadow
    /// price of capacity learned from the page layer (online and offline).
    Incremental,
    /// Rows first over the whole sequence by value density, then pages
    /// (offline only: the reference the incremental policy is measured by).
    Global,
}

impl PlanOptions {
    /// Options for `cache_bytes` of cache and reuse gap `gap` slots, with
    /// the default thread count, the time objective, no cost model and page
    /// caching only.
    pub fn new(cache_bytes: usize, gap: u64) -> Self {
        Self {
            cache_bytes,
            gap,
            threads: DEFAULT_COMPILE_THREADS,
            cpus: Vec::new(),
            objective: PlanObjective::Time,
            cost: None,
            rows: false,
            row_policy: RowPolicy::Incremental,
            block_pages: None,
            horizon: None,
            chunk: None,
        }
    }
}

/// The loader's resources on one machine and the work it does per access,
/// record, element, request and cached-row byte. Work is in core-seconds:
/// seconds of one physical core (two workers sharing a core through SMT
/// take it together).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CostModel {
    /// Physical cores the workers run on (at most one per worker).
    pub cores: f64,
    /// Read requests per second the device sustains.
    pub iops: f64,
    /// Bytes per second the device sustains.
    pub bandwidth: f64,
    /// Bytes per second of cached-row traffic main memory sustains.
    pub memory_bandwidth: f64,
    /// Share of the shorter of device and worker time that does not overlap
    /// the longer, in `[0, 1]`.
    pub serial: f64,
    /// Accesses per priced window.
    pub bucket: u64,
    /// Every access: claiming, bookkeeping, completing, the ring.
    pub access: f64,
    /// One read request: building, submitting and harvesting it.
    pub request: f64,
    /// The online compiler, per access, where it shares the workers' cores.
    pub compile: f64,
    /// Decoding records.
    pub input: InputCost,
    /// Cached rows.
    pub rows: RowCost,
    /// Per output schema, in loader order.
    pub outputs: Vec<OutputCost>,
    /// Converting an element, per [`Conversion`](crate::loader::Conversion)
    /// class, on top of the schema's `map_element`.
    pub convert: [f64; CONVERSIONS],
}

/// Decoding records: core-seconds per record, per stored element and per
/// record whose chunk is cold.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InputCost {
    /// Per decoded record.
    pub decode: f64,
    /// Per stored element decoded.
    pub decode_element: f64,
    /// Per decoded record whose chunk is not among the few most recently
    /// used (its decode tables and column map are cold).
    pub cold: f64,
}

/// Cached rows: core-seconds of gathering one into a worker's scratch and of
/// building and writing one, both per payload byte.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RowCost {
    /// Gathering a cached row from its frames into a worker's scratch, per
    /// payload byte.
    pub gather: f64,
    /// Building a decoded record's row and writing it into its frames, per
    /// payload byte.
    pub admit: f64,
}

/// One band of a process function's size curve: the costs that hold for a
/// row of up to `max_elements` elements.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProcessCost {
    /// The band's largest row size, inclusive; the last band ends at
    /// `u32::MAX`.
    pub max_elements: u32,
    /// Core-seconds of one call in this band.
    pub process: f64,
    /// Core-seconds a partial call in this band adds.
    pub resolve: f64,
}

/// What one output schema's delivery costs, per element, per row and per
/// call.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OutputCost {
    /// The process function, per call (none for the built-in output).
    pub process: f64,
    /// Mapping an element to the schema's output dims and writing it, per
    /// element (the conversion's class adds its own cost).
    pub map_element: f64,
    /// Filling a dense row with the default, per output dim.
    pub fill_element: f64,
    /// Additional work inside a process callback when it receives a partial
    /// row and resolves values itself, per partial call.
    pub resolve: f64,
    /// Additional partial-callback work per stored element.
    pub resolve_element: f64,
    /// Optional size bands replacing the fixed process/resolver costs.
    pub curve: Vec<ProcessCost>,
}

/// What a window of accesses takes from each resource.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Loads {
    /// Read requests: one per run of pages read together, and one per extra
    /// admission (block) read.
    pub requests: f64,
    /// Bytes read.
    pub bytes: f64,
    /// Worker work in core-seconds.
    pub core_seconds: f64,
    /// Bytes of cached-row traffic through main memory.
    pub memory_bytes: f64,
}

impl Loads {
    /// Adds another window's loads.
    pub fn add(&mut self, o: &Loads) {
        self.requests += o.requests;
        self.bytes += o.bytes;
        self.core_seconds += o.core_seconds;
        self.memory_bytes += o.memory_bytes;
    }
    /// The same loads multiplied by `k`.
    pub fn scaled(&self, k: f64) -> Loads {
        Loads { requests: self.requests * k, bytes: self.bytes * k, core_seconds: self.core_seconds * k, memory_bytes: self.memory_bytes * k }
    }
}

/// Marginal seconds of a window's modeled time per unit of each load: the
/// gradient of a smoothed `T` at the loads it was taken at. Saving loads
/// `x` is worth `prices.value(x)` seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Prices {
    /// Seconds per read request.
    pub request: f64,
    /// Seconds per byte read.
    pub byte: f64,
    /// Seconds per core-second of worker work.
    pub core: f64,
    /// Seconds per byte of cached-row memory traffic.
    pub memory: f64,
}

impl Prices {
    /// Seconds saved by removing loads `x` at these prices, and the cost of
    /// adding them.
    #[inline]
    pub fn value(&self, x: &Loads) -> f64 {
        self.request * x.requests + self.byte * x.bytes + self.core * x.core_seconds + self.memory * x.memory_bytes
    }
}

/// Relative width of the smoothed maxima behind [`Prices`]: resources
/// within about this share of each other split the price; one clearly
/// ahead takes all of it.
const SOFT: f64 = 0.1;

/// The weight of `x` against `y` in a smooth maximum: one half when both
/// are zero, else a logistic in their difference relative to the larger.
fn share(x: f64, y: f64) -> f64 {
    let m = x.max(y);
    if m <= 0.0 {
        return 0.5;
    }
    1.0 / (1.0 + (-(x - y) / (SOFT * m)).exp())
}

/// Leads the stand-alone binary form of a cost model.
const COST_MAGIC: &[u8; 8] = b"CPKCOST1";

impl CostModel {
    /// Default costs (see [`crate::opt::defaults`]) for a loader whose
    /// workers run on `cores` physical cores, reading storage of `tier`, with
    /// `builtin[k]` telling whether schema `k` is the built-in output.
    pub fn defaults(cores: f64, tier: crate::opt::defaults::Tier, builtin: &[bool]) -> Self {
        crate::opt::defaults::model(cores, tier, builtin)
    }

    /// Modeled seconds of a window with these loads.
    pub fn seconds(&self, l: &Loads) -> f64 {
        let d = (l.requests / self.iops).max(l.bytes / self.bandwidth);
        let c = l.core_seconds / self.cores;
        (d.max(c) + self.serial * d.min(c)).max(l.memory_bytes / self.memory_bandwidth)
    }

    /// Each resource's service time for these loads: requests, bytes,
    /// cores and memory.
    pub fn service(&self, l: &Loads) -> [f64; 4] {
        [l.requests / self.iops, l.bytes / self.bandwidth, l.core_seconds / self.cores, l.memory_bytes / self.memory_bandwidth]
    }

    /// The prices of loads near `l`: the chain rule through the three maxima
    /// of `T`, with each maximum split smoothly by `share` so a resource that
    /// is about to bind already carries part of the price.
    pub fn prices(&self, l: &Loads) -> Prices {
        let [xi, xb, xc, xm] = self.service(l);
        let s = self.serial;
        // Device: `d = max(xi, xb)`, split by which of requests and bytes is
        // ahead; `wd` weights device time against worker time in
        // `max(d, xc) + s · min(d, xc)`, so `gd` and `gc` are its two partial
        // derivatives; `wdc` the same for the memory maximum outside them.
        let wi = share(xi, xb);
        let d = xi.max(xb);
        let wd = share(d, xc);
        let dc = d.max(xc) + s * d.min(xc);
        let wdc = share(dc, xm);
        let gd = wdc * (wd + s * (1.0 - wd));
        let gc = wdc * ((1.0 - wd) + s * wd);
        // Seconds per unit of each load: the service time's own derivative
        // is the reciprocal of the rate.
        Prices { request: gd * wi / self.iops, byte: gd * (1.0 - wi) / self.bandwidth, core: gc / self.cores, memory: (1.0 - wdc) / self.memory_bandwidth }
    }

    /// The cost model as bytes: a magic, its fields (as plan files embed
    /// them) and a checksum.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = COST_MAGIC.to_vec();
        self.encode(&mut w);
        let sum = crate::opt::plan::checksum(&w);
        w.extend_from_slice(&sum.to_le_bytes());
        w
    }

    /// Reads [`CostModel::to_bytes`]. The values are not checked against a
    /// graph; compilation does that.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let bad = || Error::Plan("not a valid cost model".into());
        if bytes.len() < COST_MAGIC.len() + 8 || &bytes[..COST_MAGIC.len()] != COST_MAGIC {
            return Err(bad());
        }
        let (body, sum) = bytes.split_at(bytes.len() - 8);
        if crate::opt::plan::checksum(body) != u64::from_le_bytes(sum.try_into().unwrap()) {
            return Err(Error::Plan("cost model checksum mismatch".into()));
        }
        let mut r = Reader::new(&body[COST_MAGIC.len()..]);
        let model = Self::decode(&mut r)?;
        if r.at != r.bytes.len() {
            return Err(bad());
        }
        Ok(model)
    }

    /// The model's scalar costs in the order [`CostModel::encode`] writes
    /// them; [`CostModel::check`] validates them from `access` on.
    fn scalars(&self) -> [f64; 13] {
        let (i, r) = (&self.input, &self.rows);
        [self.cores, self.iops, self.bandwidth, self.memory_bandwidth, self.serial, self.access, self.request, self.compile, i.decode, i.decode_element, i.cold, r.gather, r.admit]
    }

    /// Appends the fields, little-endian: the scalars in declaration order,
    /// the conversion classes, the bucket, then the number of outputs and
    /// each output's fields and bands.
    pub(crate) fn encode(&self, w: &mut Vec<u8>) {
        for v in self.scalars().into_iter().chain(self.convert) {
            w.extend_from_slice(&v.to_le_bytes());
        }
        w.extend_from_slice(&self.bucket.to_le_bytes());
        w.extend_from_slice(&(self.outputs.len() as u64).to_le_bytes());
        for o in &self.outputs {
            for v in [o.process, o.map_element, o.fill_element, o.resolve, o.resolve_element] {
                w.extend_from_slice(&v.to_le_bytes());
            }
            w.extend_from_slice(&(o.curve.len() as u64).to_le_bytes());
            for band in &o.curve {
                w.extend_from_slice(&band.max_elements.to_le_bytes());
                w.extend_from_slice(&band.process.to_le_bytes());
                w.extend_from_slice(&band.resolve.to_le_bytes());
            }
        }
    }

    /// Reads what [`CostModel::encode`] wrote.
    pub(crate) fn decode(r: &mut Reader<'_>) -> Result<Self> {
        let mut s = [0.0; 13];
        for v in &mut s {
            *v = r.f64()?;
        }
        let [cores, iops, bandwidth, memory_bandwidth, serial, access, request, compile, decode, decode_element, cold, gather, admit] = s;
        let mut convert = [0.0; CONVERSIONS];
        for c in &mut convert {
            *c = r.f64()?;
        }
        let bucket = r.u64()?;
        let outputs = (0..r.count(48)?)
            .map(|_| {
                let process = r.f64()?;
                let map_element = r.f64()?;
                let fill_element = r.f64()?;
                let resolve = r.f64()?;
                let resolve_element = r.f64()?;
                let curve = (0..r.count(20)?).map(|_| Ok(ProcessCost { max_elements: r.u32()?, process: r.f64()?, resolve: r.f64()? })).collect::<Result<Vec<_>>>()?;
                Ok(OutputCost { process, map_element, fill_element, resolve, resolve_element, curve })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cores,
            iops,
            bandwidth,
            memory_bandwidth,
            serial,
            bucket,
            access,
            request,
            compile,
            input: InputCost { decode, decode_element, cold },
            rows: RowCost { gather, admit },
            outputs,
            convert,
        })
    }

    /// Checks the model against a graph: one output entry per schema, a
    /// nonzero bucket, positive finite rates, finite nonnegative costs, a
    /// `serial` share in `[0, 1]`, and process bands that increase and end at
    /// `u32::MAX`.
    pub(crate) fn check(&self, graph: &Graph) -> Result<()> {
        let mut costs = self.scalars()[5..].to_vec();
        for c in &self.outputs {
            costs.extend([c.process, c.map_element, c.fill_element, c.resolve, c.resolve_element]);
            for band in &c.curve {
                costs.extend([band.process, band.resolve]);
            }
            if c.curve.windows(2).any(|w| w[0].max_elements >= w[1].max_elements) || c.curve.last().is_some_and(|b| b.max_elements != u32::MAX) {
                return Err(Error::Plan("process cost bands must increase and end at u32::MAX".into()));
            }
        }
        costs.extend(self.convert);
        if self.outputs.len() != graph.schemas.len()
            || self.bucket == 0
            || [self.cores, self.iops, self.bandwidth, self.memory_bandwidth].iter().any(|r| !(r.is_finite() && *r > 0.0))
            || costs.iter().any(|c| !(c.is_finite() && *c >= 0.0))
            || !(0.0..=1.0).contains(&self.serial)
        {
            return Err(Error::Plan("cost model needs positive finite rates, finite nonnegative costs, a bucket and one entry per schema".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CostModel {
        CostModel {
            cores: 16.0,
            iops: 8e5,
            bandwidth: 3e9,
            memory_bandwidth: 5e10,
            serial: 0.05,
            bucket: 16384,
            access: 9.5e-7,
            request: 2.5e-6,
            compile: 7e-8,
            input: InputCost { decode: 0.0, decode_element: 1.6e-9, cold: 1e-6 },
            rows: RowCost { gather: 4e-11, admit: 3e-10 },
            outputs: vec![
                OutputCost { process: 0.0, map_element: 7.8e-10, fill_element: 1e-10, resolve: 0.0, resolve_element: 0.0, curve: Vec::new() },
                OutputCost {
                    process: 2e-6,
                    map_element: 7.8e-10,
                    fill_element: 0.0,
                    resolve: 1e-6,
                    resolve_element: 4e-10,
                    curve: vec![ProcessCost { max_elements: 512, process: 1.5e-6, resolve: 1e-7 }, ProcessCost { max_elements: u32::MAX, process: 3e-6, resolve: 2e-7 }],
                },
            ],
            convert: [0.0, 5e-11, 0.0, 5e-11, 1e-11, 0.0],
        }
    }

    #[test]
    fn bytes_round_trip_and_reject_damage() {
        let c = sample();
        let bytes = c.to_bytes();
        assert_eq!(CostModel::from_bytes(&bytes).unwrap(), c);
        let mut damaged = bytes.clone();
        damaged[20] ^= 1;
        assert!(CostModel::from_bytes(&damaged).is_err());
        assert!(CostModel::from_bytes(&bytes[..bytes.len() - 9]).is_err());
    }

    #[test]
    fn prices_follow_the_binding_resource() {
        let c = CostModel { serial: 0.0, ..sample() };
        // Requests far ahead: a request is worth 1 / iops, a byte nothing.
        let iops_bound = Loads { requests: 1e4, bytes: 4096.0, core_seconds: 1e-3, memory_bytes: 0.0 };
        let p = c.prices(&iops_bound);
        assert!((p.request * c.iops - 1.0).abs() < 1e-3 && p.byte * c.bandwidth < 1e-3 && p.core * c.cores < 1e-3);
        // Bytes ahead: the reverse.
        let bw_bound = Loads { requests: 1.0, bytes: 3e7, core_seconds: 1e-3, memory_bytes: 0.0 };
        let p = c.prices(&bw_bound);
        assert!((p.byte * c.bandwidth - 1.0).abs() < 1e-3 && p.request * c.iops < 1e-3);
        // Workers ahead: device loads are free.
        let cpu_bound = Loads { requests: 1.0, bytes: 4096.0, core_seconds: 10.0, memory_bytes: 0.0 };
        let p = c.prices(&cpu_bound);
        assert!((p.core * c.cores - 1.0).abs() < 1e-3 && p.request * c.iops < 1e-3 && p.byte * c.bandwidth < 1e-3);
        // Balanced device: request and byte split the device's price.
        let even = Loads { requests: 1e4, bytes: 1e4 * c.bandwidth / c.iops, core_seconds: 1e-3, memory_bytes: 0.0 };
        let p = c.prices(&even);
        assert!((p.request * c.iops - 0.5).abs() < 1e-3 && (p.byte * c.bandwidth - 0.5).abs() < 1e-3);
    }

    #[test]
    fn seconds_take_the_longest_resource_with_the_serial_share() {
        let c = CostModel { serial: 0.5, ..sample() };
        let l = Loads { requests: 8e5, bytes: 0.0, core_seconds: 8.0, memory_bytes: 0.0 };
        // Device 1 s, workers 0.5 s: 1 + 0.5 · 0.5.
        assert!((c.seconds(&l) - 1.25).abs() < 1e-12);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_round_trip_ignores_unknown_fields() {
        let c = sample();
        let mut value = serde_json::to_value(&c).unwrap();
        value["format"] = serde_json::json!(1);
        value["outputs"][1]["name"] = serde_json::json!("sample");
        let back: CostModel = serde_json::from_value(value).unwrap();
        assert_eq!(back, c);
    }
}
