//! Buffer requirements for caller-managed encoding allocations.

use crate::Result;
use super::{EncodeOptions, MatrixRef, Rows};

/// Output capacity in bytes and scratch capacity in u64 words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkSizes {
    /// Bytes [`crate::scans::encode_chunk`] can write for the selection.
    pub output_bytes: usize,
    /// `u64` words of workspace those calls need.
    pub workspace_words: usize,
}

/// Queries both encoding buffers using the same input and options. The
/// underlying sizing rules are those of the individual sizing functions.
pub fn chunk_sizes<'a>(
    src: impl Into<MatrixRef<'a>>,
    rows: Rows,
    options: &EncodeOptions,
) -> Result<ChunkSizes> {
    let (output_bytes, workspace_words) = super::encode::chunk_needs(&src.into(), rows, options)?;
    Ok(ChunkSizes {
        output_bytes,
        workspace_words,
    })
}
