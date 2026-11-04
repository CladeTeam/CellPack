//! Process-wide settings bound once: environment variables are read and
//! the CPU is probed together, before the first kernel runs, and every
//! function pointer of the chosen kernel set is fixed from then on.
//!
//! Contexts built from metadata ([`decode_tables`](crate::scans::decode_tables),
//! [`encode_tables`](crate::scans::encode_tables), [`Mapper`](crate::scans::Mapper),
//! a loader) take the bound kernels at construction and carry them, so the
//! per-cell paths of those contexts never consult this module.
//!
//! Environment variables (all read here, nowhere else):
//!
//! | Variable | Values | Effect |
//! |---|---|---|
//! | `CELLPACK_KERNELS` | `scalar`, `avx2`, `avx512` | Caps the kernel set (for ablation); the best supported set at or below the cap is used. Other values are an error. |
//!
//! CPU topology is probed here as well ([`physical_cores`]): it reads `/sys`
//! (and, when `cpus` is empty, the affinity set) on every call and caches
//! nothing.

use std::sync::OnceLock;

use crate::error::{Error, Result};
use crate::scans::kernels::{self, Isa, Kernels};

/// Name of the kernel-cap environment variable.
pub const KERNELS_ENV: &str = "CELLPACK_KERNELS";

/// The most capable kernel set a process may use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KernelCap {
    /// The best set this CPU supports.
    #[default]
    Auto,
    /// At most this set.
    At(Isa),
}

impl KernelCap {
    /// Parses `scalar`, `avx2` or `avx512`, ignoring surrounding whitespace.
    pub fn parse(s: &str) -> Result<Self> {
        Isa::ALL
            .into_iter()
            .find(|isa| isa.name() == s.trim())
            .map(KernelCap::At)
            .ok_or(Error::InvalidOptions("CELLPACK_KERNELS must be scalar, avx2 or avx512"))
    }
}

/// Settings the process binds once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeOptions {
    /// The kernel set the process may bind.
    pub kernels: KernelCap,
}

impl RuntimeOptions {
    /// Options from the environment (see the module table).
    pub fn from_env() -> Result<Self> {
        let kernels = match std::env::var(KERNELS_ENV) {
            Ok(v) => KernelCap::parse(&v)?,
            Err(std::env::VarError::NotPresent) => KernelCap::Auto,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(Error::InvalidOptions("CELLPACK_KERNELS is not valid UTF-8"));
            }
        };
        Ok(Self { kernels })
    }
}

/// The bound process settings.
///
/// Holds the kernel set for the process lifetime (`&'static`, so no handle
/// needs cloning) and the options it was bound with, which [`init`] uses to
/// reject a second, conflicting binding.
#[derive(Debug)]
pub struct Runtime {
    options: RuntimeOptions,
    kernels: &'static Kernels,
}

impl Runtime {
    /// Selects the best kernel set this CPU supports at or below the cap; an
    /// `Auto` cap means the highest ISA the crate knows, so `select` chooses
    /// from what is actually supported.
    fn bind(options: RuntimeOptions) -> Self {
        let cap = match options.kernels {
            KernelCap::Auto => Isa::Avx512,
            KernelCap::At(isa) => isa,
        };
        Self { options, kernels: kernels::select(cap) }
    }

    /// Kernel set of the best ISA at or below `isa` this CPU supports,
    /// independent of the process binding (equivalence tests, benchmarks).
    #[doc(hidden)]
    pub fn for_isa(isa: Isa) -> Self {
        Self::bind(RuntimeOptions { kernels: KernelCap::At(isa) })
    }

    /// The options this binding was created with (not the ISA it resolved to;
    /// see [`Runtime::isa`]).
    pub fn options(&self) -> RuntimeOptions {
        self.options
    }

    /// The ISA of the bound kernel set.
    pub fn isa(&self) -> Isa {
        self.kernels.isa
    }

    /// The kernel set itself, for callers that bind function pointers once
    /// and then keep them (decode/encode tables, `Mapper`, `RowOps`).
    #[inline(always)]
    pub(crate) fn kernels(&self) -> &'static Kernels {
        self.kernels
    }
}

/// The single process binding. The first [`init`] or [`get`] fixes it for the
/// lifetime of the process; later calls can only confirm it.
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Binds the process settings. Calling it again with the same options
/// returns the existing binding; different options are an error, as is
/// any call after another entry point bound the settings from the
/// environment with different options.
///
/// The binding is never replaced: a mismatched call reports
/// [`Error::InvalidOptions`] and leaves the first binding in force.
pub fn init(options: RuntimeOptions) -> Result<&'static Runtime> {
    let rt = RUNTIME.get_or_init(|| Runtime::bind(options));
    if rt.options == options {
        Ok(rt)
    } else {
        Err(Error::InvalidOptions("the runtime is already bound with different options"))
    }
}

/// The bound settings, binding them from the environment on first use.
///
/// A `CELLPACK_KERNELS` value that does not name a kernel set is an error
/// here, on the first call that has to read the environment.
pub fn get() -> Result<&'static Runtime> {
    match RUNTIME.get() {
        Some(rt) => Ok(rt),
        None => init(RuntimeOptions::from_env()?),
    }
}

/// Physical cores among `cpus` (logical CPU ids), or among the calling
/// thread's affinity set when `cpus` is empty: SMT siblings count once.
///
/// A logical CPU is identified by its `(physical_package_id, core_id)` pair;
/// where either file is missing or unreadable, the CPU id itself stands in, so
/// the result never exceeds the number of logical CPUs and is at least 1.
pub fn physical_cores(cpus: &[usize]) -> usize {
    let logical: Vec<usize> = if cpus.is_empty() { affinity() } else { cpus.to_vec() };
    let read = |cpu: usize, what: &str| -> Option<String> {
        std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{what}")).ok().map(|s| s.trim().to_string())
    };
    let mut cores = std::collections::HashSet::new();
    for &cpu in &logical {
        match (read(cpu, "physical_package_id"), read(cpu, "core_id")) {
            (Some(p), Some(c)) => cores.insert((p, c)),
            _ => cores.insert((String::new(), cpu.to_string())),
        };
    }
    cores.len().max(1)
}

/// The calling thread's allowed CPUs; the first `available_parallelism()` CPU
/// ids when the affinity mask cannot be read.
fn affinity() -> Vec<usize> {
    // SAFETY: `cpu_set_t` is plain old data, so zeroing it is a valid empty
    // set for `sched_getaffinity` to fill, and the size passed is the size of
    // the buffer it fills.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect();
        }
        (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).collect()
    }
}
