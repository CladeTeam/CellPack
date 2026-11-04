//! The crate's error type: one [`Error`] enum covering caller contract
//! violations, damaged metadata or records, and I/O, plus a [`Result`] alias.
//!
//! Errors are reported, never recovered from: a failed decode or read may
//! have written part of its output. Entry points check their own arguments
//! (output lengths, table/metadata agreement, per-cell index ranges); checks
//! inside record loops are debug-only because records are trusted, so record
//! damage reaches the caller only through explicit comparisons against
//! metadata ([`Error::CorruptRecord`]); per-cell table entries that fail their
//! checks surface as [`Error::CorruptMeta`] from the cell accessors, and I/O
//! failures as [`Error::Io`].

use thiserror::Error;

/// Why an operation failed.
///
/// The variants split by fault: arguments and options of the caller
/// ([`Error::InvalidInput`], [`Error::InvalidOptions`], [`Error::OutputTooSmall`],
/// [`Error::WorkspaceTooSmall`], [`Error::OutputTypeTooNarrow`],
/// [`Error::RowOutOfRange`], [`Error::TablesMismatch`]), damaged bytes
/// ([`Error::CorruptMeta`], [`Error::CorruptRecord`]) and the environment
/// ([`Error::Io`]).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// An argument, a source matrix or a cell violates the input contract.
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    /// An option lies outside its accepted range or contradicts another.
    #[error("invalid options: {0}")]
    InvalidOptions(&'static str),
    /// A caller buffer does not have the size the call requires: smaller than
    /// the output, or not the exact length of an exact-size read.
    #[error("output buffer too small: need {need}, have {have}")]
    OutputTooSmall { need: usize, have: usize },
    /// A scratch workspace is smaller than the call requires. Sizes are in
    /// `u64` words.
    #[error("workspace too small: need {need} words, have {have}")]
    WorkspaceTooSmall { need: usize, have: usize },
    /// The requested output integer type cannot hold every value it would receive.
    #[error("output integer type cannot hold {0}")]
    OutputTypeTooNarrow(u64),
    /// A row, cell or chunk index is at or past the count it indexes.
    #[error("row {row} out of range for {n_rows} rows")]
    RowOutOfRange { row: usize, n_rows: usize },
    /// The tables were built for another chunk's metadata.
    #[error("tables were derived from different metadata")]
    TablesMismatch,
    /// File or chunk metadata contradicts itself or the layout; the message names the check.
    #[error("corrupt metadata: {0}")]
    CorruptMeta(&'static str),
    /// A record's bytes contradict its metadata; the message names the check.
    #[error("corrupt record: {0}")]
    CorruptRecord(&'static str),
    /// A file operation failed.
    #[error("I/O ({kind:?}): {message}")]
    Io {
        kind: std::io::ErrorKind,
        message: String,
    },
}

impl From<std::io::Error> for Error {
    /// Keeps the `ErrorKind` and the formatted message, so callers can match
    /// on the kind without giving up the text.
    fn from(e: std::io::Error) -> Self {
        Error::Io {
            kind: e.kind(),
            message: e.to_string(),
        }
    }
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;
