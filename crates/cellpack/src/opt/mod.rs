//! Cache plans: which pages and rows stay in which cache frames over an
//! access sequence, compiled offline over a stored sequence or online ahead
//! of the loader over a sequence that need not be stored (`engine` compiles,
//! `model` holds the cost model); the plan file form, and the serial page
//! optimum that plans are measured against.
//!
//! A compilation depends only on its options, its access sequence and, for a
//! live online run, how far the plan ring's consumer has got: the order the
//! threads take work in never changes a decision.

/// Entry points: offline and online compilations and the option checks.
pub(crate) mod compile;
/// Costs used where nothing was measured.
pub mod defaults;
/// The chunked compiler: reuse intervals, shards, freezing and publication.
pub(crate) mod engine;
/// What a plan is compiled for: budget, reuse gap, cost model and prices.
mod model;
/// Occupancy over a window of positions (range add, range max).
pub(crate) mod occupancy;
/// Plans: residencies and frames, their file form, and the serial sweep.
pub(crate) mod plan;
/// Loads and prices of the work an access does, as the worker routes do it.
pub(crate) mod price;
/// Per-cell records and a sequence's cold bits.
pub(crate) mod records;
/// The serial page optimum plans are measured against.
pub(crate) mod reference;

pub use engine::OnlineStats;
pub use defaults::Tier;
pub use model::{CostModel, DEFAULT_CHUNK, DEFAULT_COMPILE_THREADS, InputCost, Loads, OutputCost, PlanObjective, PlanOptions, Prices, ProcessCost, RowCost, RowPolicy};
pub use plan::{Object, Plan, PlanSummary, Residency, access_fingerprint, default_distance};
pub use reference::{CacheSummary, PageTrace};
