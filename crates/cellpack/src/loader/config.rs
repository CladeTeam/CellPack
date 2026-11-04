//! Loader configuration and run statistics.

use std::time::Duration;

use crate::loader::exec::io::{IoBackend, IoStats};
use crate::prof::{ProfileReport, SiteMask};

/// How a loader's runs execute: threads, I/O backend, windowing and profiling.
///
/// The window and the in-flight depth bound how far a worker may run ahead of
/// the oldest unfinished access; both matter to plans, which reserve frames a
/// whole reuse gap ahead.
#[derive(Clone, Debug)]
pub struct LoaderConfig {
    /// Loader worker threads; the default is 16.
    pub workers: usize,
    /// How reads are issued (default `Uring`); `Uring` registers the input
    /// files with the ring when the kernel permits and otherwise passes a
    /// descriptor with each read.
    pub backend: IoBackend,
    /// Positions a worker keeps in flight, reading, waiting or decoding;
    /// default 64.
    pub max_inflight_per_worker: usize,
    /// Accesses per claim (default 8), rounded up to whole positions; bounded
    /// by one slot's accesses and, at execution, by the in-flight depth.
    pub claim_batch: usize,
    /// Positions per output slot (default 256); every schema of a position
    /// shares it.
    pub slot_cells: u64,
    /// The execution window, in slots: a claim starts at most this far past
    /// the oldest unfinished access. It must not exceed a plan's reuse gap;
    /// unset, it is the plan's gap, or enough slots for the in-flight
    /// accesses without a plan.
    pub window: Option<u64>,
    /// Profiled sites ([`crate::prof`]); none runs the untimed instance.
    pub profile: Option<SiteMask>,
    /// Pins worker `i` to `cpus[i % len]`; empty leaves placement to the OS.
    pub cpus: Vec<usize>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self { workers: 16, backend: IoBackend::Uring, max_inflight_per_worker: 64, claim_batch: 8, slot_cells: 256, window: None, profile: None, cpus: Vec::new() }
    }
}

/// Declares a counter struct that can also be published as plain words, so a
/// worker's live statistics cross a `Snapshots` slot without an allocation and
/// a snapshot rebuilds the struct from them. `NAMES` fixes the word order.
macro_rules! counters {
    ($(#[$m:meta])* pub struct $name:ident { $($(#[$fm:meta])* pub $f:ident: u64,)* }) => {
        $(#[$m])*
        pub struct $name { $($(#[$fm])* pub $f: u64,)* }
        impl $name {
            /// Words one instance occupies in a published slot.
            pub(crate) const WORDS: usize = Self::NAMES.len();
            /// Adds another instance's counts into this one.
            pub(crate) fn add(&mut self, o: &Self) {
                $(self.$f += o.$f;)*
            }
            /// The counts in [`Self::NAMES`] order.
            pub fn words(&self) -> impl Iterator<Item = u64> {
                [$(self.$f),*].into_iter()
            }
            /// Reads a published instance; missing trailing words are 0.
            pub(crate) fn from_words(w: &[u64]) -> Self {
                let mut it = w.iter().copied();
                Self { $($f: it.next().unwrap_or(0),)* }
            }
            /// Field names, in word order.
            pub const NAMES: &'static [&'static str] = &[$(stringify!($f)),*];
        }
    };
}

counters! {
    /// Event counts every run keeps.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Counters {
        /// Accesses delivered: one per schema of every position.
        pub accesses: u64,
        /// Records decoded (each shared input decodes once).
        pub decoded_rows: u64,
        /// Rows mapped to an output, with or without a process call.
        pub mapped_rows: u64,
        /// Calls into a schema's process function.
        pub process_calls: u64,
        /// Bytes of outputs written: one schema's `cell_bytes` per access.
        pub written_bytes: u64,
        /// Pages of the records the accesses decoded (a row hit has none).
        pub page_uses: u64,
        /// Page uses served from a frame an earlier access admitted.
        pub page_hits: u64,
        /// Pages read, whatever frame they went to.
        pub page_reads: u64,
        /// Pages read into cache frames by admission reads.
        pub admitted: u64,
        /// Extra admission reads (block reads), one per request.
        pub block_reads: u64,
        /// Inputs served from a cached row (one per distinct cell of a
        /// position).
        pub row_hits: u64,
        /// Bytes gathered for row hits.
        pub row_gather_bytes: u64,
        /// Rows decoded into cache frames.
        pub rows_admitted: u64,
        /// Bytes written for row admissions.
        pub row_admit_bytes: u64,
        /// Tasks that waited for a frame another access admits.
        pub waited: u64,
        /// Claim attempts refused by the execution window or by a gate other
        /// than the plan frontier.
        pub window_stalls: u64,
        /// Claim attempts refused because the plan ring has no entry at the
        /// claim counter yet.
        pub plan_stalls: u64,
        /// Claim attempts refused because the next output slot was not open
        /// for writing yet.
        pub output_stalls: u64,
        /// Tasks whose private-page reads had to wait for free frames.
        pub starved: u64,
    }
}

/// Counts of one schema's outputs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OutputStats {
    /// Accesses that wrote this schema.
    pub accesses: u64,
    /// Accesses of this schema whose record was decoded (one decode counts
    /// for every schema that reads the cell).
    pub decoded_rows: u64,
    /// Accesses served by a cached row of this schema's panel.
    pub row_hits: u64,
    /// Process calls made for it.
    pub process_calls: u64,
}

/// Element-count boundaries of the `process` site's size bands; a call's
/// element count (output dims for a dense record, stored elements otherwise)
/// belongs to the band below the first boundary above it.
pub const PROCESS_BOUNDS: [u32; 11] = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, u32::MAX];

/// A run's statistics: event counts, I/O counts, per-schema counts, the bytes
/// it reserved and, if it profiled, the recorded sites.
#[derive(Clone, Debug, Default)]
pub struct RunStats {
    /// Wall time of the workers (and plan thread) after setup.
    pub elapsed: Duration,
    /// Time spent sizing the frame grid and plan ring and building the
    /// shared state, before the threads start.
    pub setup_elapsed: Duration,
    /// Event counts, summed over workers.
    pub counters: Counters,
    /// Read counts, summed over workers.
    pub io: IoStats,
    /// Most accesses claimed past the frontier.
    pub peak_span: u64,
    /// Bytes of the plan's cache frames (`cache frames × page size`).
    pub cache_bytes: u64,
    /// The workers' private frames, `workers × private frames × page`.
    pub private_bytes: u64,
    /// Per output schema, in loader order.
    pub outputs: Vec<OutputStats>,
    /// The sites the run profiled; none without profiling.
    pub profile: Option<ProfileReport>,
    /// The online compilation of the run's plan, if any.
    pub online: Option<crate::opt::OnlineStats>,
}

impl RunStats {
    /// Zeros, with one output entry per schema.
    pub(crate) fn empty(graph: &crate::loader::schema::Graph) -> Self {
        Self { outputs: vec![OutputStats::default(); graph.schemas.len()], ..Default::default() }
    }

    /// Adds a worker's statistics into this run's total. Times, reservation
    /// sizes, the peak span and profile reports are not additive and stay as
    /// they are.
    pub(crate) fn add(&mut self, o: &RunStats) {
        self.counters.add(&o.counters);
        self.io.add(&o.io);
        for (a, b) in self.outputs.iter_mut().zip(&o.outputs) {
            a.accesses += b.accesses;
            a.decoded_rows += b.decoded_rows;
            a.row_hits += b.row_hits;
            a.process_calls += b.process_calls;
        }
    }

    /// Words of a live snapshot: counters, io, then per output.
    pub(crate) fn words(&self) -> impl Iterator<Item = u64> + '_ {
        self.counters
            .words()
            .chain([self.io.requests, self.io.requested_bytes, self.io.read_bytes])
            .chain(self.io.sizes)
            .chain(self.outputs.iter().flat_map(|o| [o.accesses, o.decoded_rows, o.row_hits, o.process_calls]))
    }

    /// Words a published snapshot occupies: [`Self::words`]'s length, which
    /// [`Self::from_words`] reads in the same order.
    pub(crate) fn snapshot_words(schemas: usize) -> usize {
        Counters::WORDS + 3 + 7 + 4 * schemas
    }

    /// Rebuilds the published part of a snapshot: counters, the I/O counts and
    /// the seven size buckets, then four words per schema.
    pub(crate) fn from_words(w: &[u64], schemas: usize) -> Self {
        let c = Counters::WORDS;
        let mut s = Self { counters: Counters::from_words(&w[..c]), ..Default::default() };
        s.io.requests = w[c];
        s.io.requested_bytes = w[c + 1];
        s.io.read_bytes = w[c + 2];
        s.io.sizes.copy_from_slice(&w[c + 3..c + 10]);
        s.outputs = (0..schemas).map(|k| {
            let o = &w[c + 10 + 4 * k..c + 14 + 4 * k];
            OutputStats { accesses: o[0], decoded_rows: o[1], row_hits: o[2], process_calls: o[3] }
        }).collect();
        s
    }
}
