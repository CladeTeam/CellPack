//! Costs used where nothing was measured: the one table `Loader::plan_options`,
//! pycellpack's lookup and the bench examples all read.
//!
//! Worker costs are core-seconds measured on a Xeon Platinum 8558 with 32
//! workers (local NVMe through io_uring with O_DIRECT, cellxgene), rounded;
//! the process function's is the benchmark's `sample`. Device rates per
//! storage tier are typical sustained random-read rates of their class; the
//! NVMe tier's is what the loader reached on that machine.

use crate::loader::convert::CONVERSIONS;
use crate::opt::model::{CostModel, InputCost, OutputCost, RowCost};

/// A class of storage, for its default device rates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// A local NVMe device.
    Nvme,
    /// A local SATA or SAS SSD.
    Ssd,
    /// A spinning local disk.
    Hdd,
    /// A network filesystem.
    Network,
    /// Pages served from main memory.
    Memory,
}

impl Tier {
    /// Every tier, in declaration order.
    pub const ALL: [Tier; 5] = [Tier::Nvme, Tier::Ssd, Tier::Hdd, Tier::Network, Tier::Memory];

    /// The tier's name in configuration and reports.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Nvme => "nvme",
            Tier::Ssd => "ssd",
            Tier::Hdd => "hdd",
            Tier::Network => "network",
            Tier::Memory => "memory",
        }
    }

    /// The tier of a name; unknown names are NVMe, the tier assumed where
    /// storage is not recognized.
    pub fn named(name: &str) -> Tier {
        Tier::ALL.into_iter().find(|t| t.name() == name).unwrap_or(Tier::Nvme)
    }
}

/// Device rates of a tier and its per-request worker cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Storage {
    /// Read requests per second.
    pub iops: f64,
    /// Bytes per second.
    pub bandwidth: f64,
    /// Core-seconds of building, submitting and harvesting one request.
    pub request: f64,
    /// Share of the shorter of device and worker time that does not overlap
    /// the longer, in `[0, 1]`.
    pub serial: f64,
}

/// The stored rates of a tier: sustained random reads, and the loader's
/// per-request cost measured on the reference machine.
pub fn storage(tier: Tier) -> Storage {
    let (iops, bandwidth) = match tier {
        Tier::Nvme => (1.0e6, 6.1e9),
        Tier::Ssd => (8e4, 5e8),
        Tier::Hdd => (200.0, 1.5e8),
        Tier::Network => (2e4, 1e9),
        Tier::Memory => (5e6, 1e10),
    };
    Storage { iops, bandwidth, request: 3.5e-6, serial: 0.0 }
}

/// The host's worker costs and memory bandwidth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Host {
    /// Bytes per second of cached-row traffic main memory sustains.
    pub memory_bandwidth: f64,
    /// Core-seconds of one access's bookkeeping.
    pub access: f64,
    /// Core-seconds of the online compiler, per access.
    pub compile: f64,
    /// Decoding records.
    pub input: InputCost,
    /// Cached rows.
    pub rows: RowCost,
    /// Converting an element, per conversion class.
    pub convert: [f64; CONVERSIONS],
}

/// Worker costs of the reference machine, in core-seconds.
pub const HOST: Host = Host {
    memory_bandwidth: 5e10,
    access: 1.1e-6,
    compile: 7e-8,
    input: InputCost { decode: 0.0, decode_element: 2.4e-9, cold: 9.2e-7 },
    rows: RowCost { gather: 3.0e-10, admit: 7.9e-10 },
    // same, widen, reinterpret, int_to_f32, int_to_f64, f32_to_f64
    convert: [0.0, 5e-11, 0.0, 5e-11, 1e-11, 0.0],
};

/// The built-in dense output.
pub fn builtin() -> OutputCost {
    OutputCost { process: 0.0, map_element: 1.3e-9, fill_element: 1e-10, resolve: 0.0, resolve_element: 0.0, curve: Vec::new() }
}

/// A process function nothing was measured for.
pub fn process() -> OutputCost {
    OutputCost { process: 1.0e-5, map_element: 8.8e-10, fill_element: 0.0, resolve: 1.7e-6, resolve_element: 0.0, curve: Vec::new() }
}

/// Accesses per priced window.
pub const BUCKET: u64 = 16384;

/// The default model (see [`CostModel::defaults`]) for `cores` physical
/// cores and storage of `tier`; `builtin_outputs[k]` says whether schema `k`
/// is the built-in output.
pub(crate) fn model(cores: f64, tier: Tier, builtin_outputs: &[bool]) -> CostModel {
    let s = storage(tier);
    let h = HOST;
    CostModel {
        cores,
        iops: s.iops,
        bandwidth: s.bandwidth,
        memory_bandwidth: h.memory_bandwidth,
        serial: s.serial,
        bucket: BUCKET,
        access: h.access,
        request: s.request,
        compile: h.compile,
        input: h.input,
        rows: h.rows,
        outputs: builtin_outputs.iter().map(|&b| if b { builtin() } else { process() }).collect(),
        convert: h.convert,
    }
}
