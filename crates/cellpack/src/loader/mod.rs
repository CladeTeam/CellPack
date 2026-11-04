//! Cell loading for training: a data source (`.cpk` matrices, each mapping
//! its columns to standard ids), output schemas (output dims as standard
//! ids, an element type, and the built-in dense row or a process function),
//! an access sequence (one cell per schema per position, stored or
//! computed) and an optional cache plan, compiled offline or online while
//! the run goes (see [`crate::opt`]).
//!
//! Workers claim whole positions in order within a bounded window and
//! execute the plan's entries: read pages into the cache frames the plan
//! assigns or into private frames (misses) with io_uring or preadv, wait for
//! frames other accesses admit, decode the record, convert it to the
//! schema's output and write it to its place in the output ring, whose slots
//! hold every schema's outputs of consecutive positions. Cached rows keep
//! their stored type, shared by every schema.

pub(crate) mod accesses;
pub(crate) mod catalog;
mod config;
pub(crate) mod convert;
pub(crate) mod error;
pub(crate) mod exec;
#[allow(clippy::module_inception)]
pub(crate) mod loader;
pub(crate) mod memory;
pub(crate) mod planring;
pub(crate) mod process;
pub(crate) mod ring;
pub(crate) mod rows;
pub(crate) mod schema;

pub use crate::format::Dtype;
pub use accesses::{AccessFn, Accesses, CorpusAccessFn, ExternAccesses, ExternCorpusAccesses, Offset, Replicate, Shuffle, SliceAccesses};
pub use catalog::{CellId, DataSource, DataSourceBuilder};
pub use config::{Counters, LoaderConfig, OutputStats, PROCESS_BOUNDS, RunStats};
pub use convert::{CONVERSIONS, Conversion, allowed};
pub use error::{Error, Result};
pub use exec::io::{IoBackend, IoStats};
pub use loader::{Loader, Planning, Run};
pub use process::{CsrFn, CsrRow, DenseFn, DenseRow, PROCESS_ABI_VERSION, ProcessContext, ProcessFn, ResolveFn, RowFormat};
pub use ring::{Acquisition, Batch, BatchLease, REGION_ALIGN, Ring, RingHeader, RingLayout, SLOT_ALIGN, SlotHeader};
pub use schema::{MAX_SCHEMAS, OutputSchema, SchemaId};

// `Loader`'s memory-estimate methods, a second impl block kept apart from the
// run protocol in `loader.rs`.
mod resources;
pub use resources::MemoryEstimate;
