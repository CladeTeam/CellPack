//! Loader errors, and the classification a caller retries or reports on.
//!
//! [`Error::Config`] carries everything a caller can get wrong: a builder,
//! schema, layout, plan or config value that no I/O can make valid, and a
//! run or ring call made out of order. The other variants report failures of
//! the data, the environment, a callback or a thread. An error out of a run
//! is final for that run: the producer cancels the ring and every other
//! thread observes [`Error::Aborted`], so the first non-aborted error is the
//! one worth reporting.

use thiserror::Error;

/// A loader failure.
///
/// The variants classify the failure: caller input ([`Error::Config`]), an
/// unusable access ([`Error::Access`]), the environment ([`Error::Io`],
/// [`Error::CellPack`]), a plan binding or option ([`Error::Plan`]), a
/// callback's status ([`Error::Process`], [`Error::Accesses`]), a panicked
/// thread ([`Error::WorkerPanic`]), an occupied loader or ring
/// ([`Error::Busy`]) and cancellation ([`Error::Aborted`]).
#[derive(Debug, Error)]
pub enum Error {
    /// A declaration or a protocol use was rejected: a builder, schema,
    /// layout or plan value, a config field, or a run/ring call out of order.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// An access names a cell outside the data source; `position` is the
    /// access index (`k`), `cell` the id it produced.
    #[error("access {position} names cell {cell}, outside the data source")]
    Access { position: u64, cell: u64 },
    /// An OS error: opening or stating an input, a ring allocation, or a
    /// read that returned fewer bytes than the record needs.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    /// A format layer error: a corrupt chunk, metadata or record.
    #[error("CellPack: {0}")]
    CellPack(#[from] crate::Error),
    /// A plan does not bind to this loader, its inputs or the sequence, or a
    /// plan option cannot hold (a range, a cost model, a block size).
    #[error("plan: {0}")]
    Plan(String),
    /// A process function returned a nonzero status, given here.
    #[error("process function returned status {0}")]
    Process(i32),
    /// An external access function returned a nonzero status, given here.
    #[error("access function returned status {0}")]
    Accesses(i32),
    /// A worker or plan thread panicked; caught, not unwound.
    #[error("worker panicked")]
    WorkerPanic,
    /// The loader already serves a run, or the ring already has a producer.
    #[error("loader is already running")]
    Busy,
    /// The run was cancelled, by [`crate::loader::Run::cancel`], a ring
    /// abort, a failing thread or a dropped `Run`.
    #[error("run aborted")]
    Aborted,
}

/// The loader's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Builds an [`Error::Config`] from a message.
pub(crate) fn config(message: impl Into<String>) -> Error {
    Error::Config(message.into())
}
