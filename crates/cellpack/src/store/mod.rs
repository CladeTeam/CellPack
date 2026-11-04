//! End-to-end `.cpk` files with borrowed-buffer and allocating Rust interfaces.
//! Readers prepare selections from metadata before writing into caller memory.

mod panel;
mod reader;
mod reads;
mod selection;
mod writer;

pub use panel::{PanelRows, RowScratch, compose_panel};
pub use reader::{CpkFile, Csr, ReadSpec};
pub use reads::{ReadOptions, ReadRange, ReadStrategy, ReadSummary, RecordReadPlan};
pub use selection::{OutputElement, ReadSelection, ReadWorkspace, RowSelection, Visit};
pub use writer::{CpkOptions, CpkWriter};
