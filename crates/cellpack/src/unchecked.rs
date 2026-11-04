//! Unchecked slice access for inner loops whose indices an entry point has
//! already validated. Every call site states, in its `SAFETY` comment, which
//! validation bounds the index. Debug builds still assert the bound.
//!
//! A release build drops the bound check entirely, so the named validation is
//! the sole guarantee that a decoded record, its ids and its workspace stay
//! inside their buffers.

/// `s[i]` without a bounds check.
///
/// # Safety
/// `i < s.len()`.
#[inline(always)]
pub(crate) unsafe fn get<T: Copy>(s: &[T], i: usize) -> T {
    debug_assert!(i < s.len(), "index {i} out of {}", s.len());
    // SAFETY: the caller guarantees `i < s.len()`; the assertion above catches
    // a violated guarantee in debug builds.
    unsafe { *s.get_unchecked(i) }
}

/// `s[i] = v` without a bounds check.
///
/// Assignment drops the previous value, so `T` needs no `Copy` bound; the
/// exclusive borrow of `s` is what rules out aliases of the written element.
///
/// # Safety
/// `i < s.len()`.
#[inline(always)]
pub(crate) unsafe fn set<T>(s: &mut [T], i: usize, v: T) {
    debug_assert!(i < s.len(), "index {i} out of {}", s.len());
    // SAFETY: the caller guarantees `i < s.len()`; `&mut [T]` keeps the element
    // unaliased, and the assertion above catches the bound in debug builds.
    unsafe { *s.get_unchecked_mut(i) = v }
}
