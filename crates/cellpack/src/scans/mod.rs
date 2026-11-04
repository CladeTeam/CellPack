//! Modular components over caller-owned memory: chunk and record encoding,
//! decoding to CSR, dense rows or [`Parts`] views, mapping to a caller's
//! column panel, page packing, and the per-chunk lookup tables.
//!
//! - **Bytes** are the only source of truth: chunks, metadata and records
//!   live in caller buffers.
//! - **Views** such as [`Meta`](crate::format::Meta) and [`Parts`] interpret
//!   bytes without owning them.
//! - **Tables** ([`DecodeTables`], [`EncodeTables`]) are lookup structures
//!   derived from a `Meta` once per chunk. Besides them a chunk allocates
//!   only its encoding statistics and metadata (see [`encode_chunk`] and
//!   [`PreparedEncoder`]) and its packing plan (see [`pack_chunk`]); every
//!   per-cell call runs on caller buffers.
//! - **Per-cell work** never allocates: outputs and workspace are caller
//!   slices, sized by the `*_len` / `*_words` / `*_bound` functions.
//!
//! Decoding preserves the input exactly: indices come back ascending in
//! source numbering with their values. Records store elements in the
//! chunk's coded gene order, so [`IdSpace::Coded`] decodes with no
//! reordering at all, for callers that fold the chunk's gene order into
//! their own mapping. [`decode_parts`] stops before values are materialized
//! (a dictionary plus ranks), for callers that map and scatter themselves.
//! Every record-level entry point has a `_pages` twin taking the record as
//! an array of pages.

pub(crate) mod bits;
pub(crate) mod coding;
pub(crate) mod decode;
pub(crate) mod encode;
pub(crate) mod kernels;
pub(crate) mod map;
pub(crate) mod pack;
pub(crate) mod rans;
mod requirements;
pub(crate) mod symbol;
pub(crate) mod tables;
pub(crate) mod types;
pub(crate) mod workspace;

/// Record and chunk decoding: CSR arrays, dense rows and [`Parts`] views,
/// plus the workspace sizing those calls need.
pub use decode::{
    Decoder, Dict, ModeDict, Parts, Ranks, Values, decode_csr, decode_dense, decode_parts,
    decode_parts_pages, decode_record, decode_record_dense, decode_record_dense_pages,
    decode_record_pages, decode_workspace_bound, decode_workspace_words, dense_len, selection_nnz,
};
/// Chunk and record encoding, the reusable [`PreparedEncoder`], and the
/// output and workspace bounds of both.
pub use encode::{
    PreparedEncoder, chunk_bound, chunk_workspace_words, encode_chunk, encode_record, encode_workspace_words,
    record_bound,
};
/// The bound instruction set, by name or as [`Isa`].
pub use kernels::{Isa, kernel_set};
/// Kernel sets this CPU supports, for equivalence tests and benchmarks.
#[doc(hidden)]
pub use kernels::supported_kernel_sets;
/// Mapping decoded rows onto a caller column panel, and the element types
/// its conversions take.
pub use map::{CastInto, Element, Mapper, UNMAPPED, cast_values, map_dense, map_parts, map_parts_at};
/// Page packing of a chunk's records.
pub use pack::{pack_chunk, packed_len};
/// Both encoding buffer sizes, queried together.
pub use requirements::{ChunkSizes, chunk_sizes};
/// The per-chunk lookup tables, bound to the runtime or to a given one.
pub use tables::{DecodeTables, EncodeTables, decode_tables, decode_tables_for, encode_tables, encode_tables_for, prefetch};
/// Borrowed inputs, selections and encoding options.
pub use types::{
    CellRef, CsrRef, CsrView, Offsets, DEFAULT_LANE_MIN, DEFAULT_MAX_LANES, DEFAULT_MODE_SHARE, DEFAULT_MU_INDICES,
    DEFAULT_MU_VALUES, DenseRef, EncodeOptions, Fill, IdSpace, IndexOut, Indices, IndptrOut,
    LayoutChoice, MatrixRef, PagedRecord, RowRef, Rows, Thresholds, ValueChoice,
};
