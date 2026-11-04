//! cellpack — per-cell compressed records for single-cell matrices, coded
//! with scans (single-cell ANS), and a loader that feeds them to training.
//!
//! A *chunk* encodes any selection of rows chosen by the caller. It holds one
//! model set (an optional gene renumbering and three rANS frequency tables)
//! and one independently decodable record per cell, placed on pages so that
//! each reads its minimum number of pages.
//!
//! - [`store`]: end-to-end `.cpk` files (one matrix each): write from CSR or
//!   dense input, open, decode row ranges.
//! - [`scans`]: modular components over caller-owned memory (encode, decode,
//!   map, pack, tables).
//! - [`mod@format`]: byte layouts and zero-copy metadata views.
//! - [`loader`]: concurrent cell loading into a caller-provided output ring.
//! - [`opt`]: the cache plan compiler (offline and online) and its serial reference.
//! - [`prof`]: profiling sites and reports.
//!
//! # Trusted data
//!
//! Records and the per-cell metadata tables of a chunk are trusted: they are
//! checked only in debug builds, and decoding data the encoder did not
//! produce for that chunk is undefined behavior. Chunk metadata is validated
//! once when parsed, and arguments (buffer sizes, tables matching metadata,
//! element types, cell indices, element positions) are always checked.
//!
//! ```
//! use cellpack::format::parse_meta;
//! use cellpack::scans::*;
//!
//! // Two cells over 6 genes, u16 values.
//! let indptr = [0u64, 3, 5];
//! let indices = [0u16, 2, 5, 1, 2];
//! let data: Vec<u8> = [1u16, 1, 3, 2, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
//! let src = CsrRef { n_genes: 6, value_width: 2, indptr: &indptr, indices: Indices::U16(&indices), data: &data };
//! let opt = EncodeOptions {
//!     renumber: true,
//!     thresholds: Thresholds { indices: DEFAULT_MU_INDICES, values: DEFAULT_MU_VALUES },
//!     max_lanes: DEFAULT_MAX_LANES,
//!     lane_min: DEFAULT_LANE_MIN,
//!     column_modes: Some(DEFAULT_MODE_SHARE),
//!     layout: LayoutChoice::Auto,
//!     values: ValueChoice::Auto,
//! };
//! let mut chunk = vec![0u8; chunk_bound(&src, Rows::All, &opt).unwrap()];
//! let mut ews = vec![0u64; chunk_workspace_words(&src, Rows::All, &opt).unwrap()];
//! let len = encode_chunk(&src, Rows::All, &opt, &mut chunk, &mut ews).unwrap();
//! let chunk = &chunk[..len];
//!
//! let meta = parse_meta(chunk).unwrap();
//! let tables = decode_tables(&meta).unwrap();
//! let records = meta.records_of(chunk).unwrap();
//! let mut ws = vec![0u64; decode_workspace_words(&meta)];
//! let nnz = selection_nnz(&meta, Rows::All).unwrap() as usize;
//! let (mut ip, mut ix, mut vx) = (vec![0u64; 3], vec![0u32; nnz], vec![0u8; nnz * 2]);
//! decode_csr(&meta, &tables, records, Rows::All, IdSpace::Source, &mut ip, &mut ix, &mut vx, &mut ws).unwrap();
//! assert_eq!(ip, indptr);
//! assert_eq!(ix, [0, 2, 5, 1, 2]);
//! assert_eq!(vx, data);
//! ```

pub mod store;
mod error;
pub mod format;
pub mod loader;
pub mod opt;
pub mod prof;
pub mod runtime;
mod unchecked;
pub mod scans;

pub use error::{Error, Result};
pub use store::{CpkFile, CpkWriter};
