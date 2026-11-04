//! Typed encoder input: gene indices by their stored type and values by
//! their width, so per-element loops compile for each combination. Rows
//! dispatch once to the typed code with [`with_indices`] and [`with_width`].

use crate::error::{Error, Result};

/// A stored gene index type.
pub(crate) trait Index: Copy {
    /// Gene id of an index that passed [`Index::valid`].
    fn id(self) -> u32;
    /// Whether the index is a gene id in `u32` range.
    fn valid(self) -> bool;
    /// The gene id, or why the index is none.
    fn check(self) -> Result<u32>;
}

const NEGATIVE: Error = Error::InvalidInput("negative gene index");
const BEYOND: Error = Error::InvalidInput("index outside u32 range");

impl Index for u16 {
    #[inline(always)]
    fn id(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn valid(self) -> bool {
        true
    }
    fn check(self) -> Result<u32> {
        Ok(self as u32)
    }
}

impl Index for u32 {
    #[inline(always)]
    fn id(self) -> u32 {
        self
    }
    #[inline(always)]
    fn valid(self) -> bool {
        true
    }
    fn check(self) -> Result<u32> {
        Ok(self)
    }
}

impl Index for i32 {
    #[inline(always)]
    fn id(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn valid(self) -> bool {
        self >= 0
    }
    fn check(self) -> Result<u32> {
        u32::try_from(self).map_err(|_| NEGATIVE)
    }
}

impl Index for i64 {
    #[inline(always)]
    fn id(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn valid(self) -> bool {
        (self as u64) <= u32::MAX as u64
    }
    fn check(self) -> Result<u32> {
        let v = u64::try_from(self).map_err(|_| NEGATIVE)?;
        u32::try_from(v).map_err(|_| BEYOND)
    }
}

impl Index for u64 {
    #[inline(always)]
    fn id(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn valid(self) -> bool {
        self <= u32::MAX as u64
    }
    fn check(self) -> Result<u32> {
        u32::try_from(self).map_err(|_| BEYOND)
    }
}

/// Evaluates `$body` with `$s` bound to the typed slice of an
/// [`Indices`](crate::scans::types::Indices).
macro_rules! with_indices {
    ($idx:expr, |$s:ident| $body:expr) => {
        match $idx {
            $crate::scans::types::Indices::U16($s) => $body,
            $crate::scans::types::Indices::U32($s) => $body,
            $crate::scans::types::Indices::I32($s) => $body,
            $crate::scans::types::Indices::I64($s) => $body,
            $crate::scans::types::Indices::U64($s) => $body,
        }
    };
}

/// Evaluates `$body` with the constant `$w` set to a value width of 1, 2,
/// 4 or 8 bytes.
macro_rules! with_width {
    ($vw:expr, |$w:ident| $body:expr) => {
        match $vw {
            1 => {
                const $w: usize = 1;
                $body
            }
            2 => {
                const $w: usize = 2;
                $body
            }
            4 => {
                const $w: usize = 4;
                $body
            }
            _ => {
                const $w: usize = 8;
                $body
            }
        }
    };
}

/// The typed-dispatch macros, for callers in sibling modules.
pub(crate) use {with_indices, with_width};

/// Value `t` of little-endian `W`-byte values as an unsigned integer.
///
/// # Safety
/// `values` holds at least `t + 1` values.
#[inline(always)]
pub(crate) unsafe fn load<const W: usize>(values: &[u8], t: usize) -> u64 {
    debug_assert!((t + 1) * W <= values.len());
    // SAFETY: as documented.
    unsafe {
        let p = values.as_ptr().add(t * W);
        match W {
            1 => *p as u64,
            2 => u16::from_le_bytes(p.cast::<[u8; 2]>().read_unaligned()) as u64,
            4 => u32::from_le_bytes(p.cast::<[u8; 4]>().read_unaligned()) as u64,
            _ => u64::from_le_bytes(p.cast::<[u8; 8]>().read_unaligned()),
        }
    }
}

/// Stores `v` as value `t` of little-endian `W`-byte values.
///
/// # Safety
/// `values` holds at least `t + 1` values.
#[inline(always)]
pub(crate) unsafe fn store<const W: usize>(values: &mut [u8], t: usize, v: u64) {
    debug_assert!((t + 1) * W <= values.len());
    // SAFETY: as documented; `W <= 8`.
    unsafe { std::ptr::copy_nonoverlapping(v.to_le_bytes().as_ptr(), values.as_mut_ptr().add(t * W), W) };
}

/// Zero values among `W`-byte values.
#[inline]
pub(crate) fn zeros<const W: usize>(values: &[u8]) -> usize {
    let n = values.len() / W;
    // SAFETY: `t < n` values.
    (0..n).map(|t| (unsafe { load::<W>(values, t) } == 0) as usize).sum()
}

/// Whether any of the `W`-byte values is zero.
#[inline]
pub(crate) fn any_zero<const W: usize>(values: &[u8]) -> bool {
    zeros::<W>(values) > 0
}
