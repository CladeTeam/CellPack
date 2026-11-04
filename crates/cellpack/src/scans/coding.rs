//! The per-cell coding.
//!
//! - **Indices**: the cell's genes in coded numbering, ascending, as gaps
//!   (`c0`, then `c[t] - c[t-1] - 1`) under the chunk's gap model.
//! - **Values**: in the same order as the indices (ascending coded id, which
//!   is the cell's own order without renumbering). The distinct bit patterns form a
//!   dictionary, sorted ascending as unsigned integers and stored as gaps
//!   under the dict model; each value is coded as its dictionary position
//!   under the rank model. Small values dominate single-cell data, so value
//!   order and frequency order nearly coincide and position 0 is the most
//!   common rank. A one-entry dictionary needs no ranks.
//!
//! Every integer goes through [`crate::scans::symbol::split`]; symbols are rANS-coded and
//! escape bits form a separate stream per sequence.

/// Largest value representable in `vw` bytes.
#[inline(always)]
pub(crate) fn value_max(vw: usize) -> u64 {
    if vw == 8 {
        u64::MAX
    } else {
        (1u64 << (8 * vw)) - 1
    }
}
