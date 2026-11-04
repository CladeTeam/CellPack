//! Huge-page-backed anonymous memory: the loader's large arenas (column and
//! panel maps, the frame grid, the output ring) and a worker's process
//! scratch.
//!
//! A buffer is one private `MAP_NORESERVE` mapping, so it reserves address
//! space rather than memory; pages materialize on first touch. `MADV_HUGEPAGE`
//! keeps the random per-cell accesses from costing a 4 KiB TLB miss each.
//! Ownership is exclusive: the buffer hands out raw pointers and slices, and
//! the owner keeps them from outliving it.

use crate::loader::error::Result;

/// Anonymous private memory reserved lazily and advised for transparent huge
/// pages. Used for the large structures the workers read at random (column
/// maps, frames) or stream (the output ring), where 4 KiB pages would cost a
/// TLB miss on most accesses.
pub(crate) struct HugeBuf {
    /// Start of the mapping, page-aligned; never null.
    ptr: *mut u8,
    /// Bytes mapped, at least 1 (a zero-length request maps one byte).
    len: usize,
}

// SAFETY: the buffer is plain memory owned by this value; sharing follows
// Rust's borrow rules through the accessors.
unsafe impl Send for HugeBuf {}
// SAFETY: as above.
unsafe impl Sync for HugeBuf {}

impl HugeBuf {
    /// A buffer of `len` bytes, readable and writable. Zero-filled (pages are
    /// materialized on first touch), so no initialization pass is needed; a
    /// request of 0 bytes maps one byte.
    pub fn zeroed(len: usize) -> Result<Self> {
        let len = len.max(1);
        // SAFETY: anonymous private mapping with no address hint.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        // Advisory: without THP the mapping still works with small pages.
        // SAFETY: the range is exactly the mapping just created.
        unsafe { libc::madvise(ptr, len, libc::MADV_HUGEPAGE) };
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }
    /// The mapping's start. Valid until this buffer drops; the memory reads
    /// as zero until written.
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    /// Bytes mapped (the requested length, or 1 for a request of 0).
    pub fn len(&self) -> usize {
        self.len
    }
    /// A copy of `values`, as `values.len() * 4` bytes.
    pub fn from_u32s(values: &[u32]) -> Result<Self> {
        let buf = Self::zeroed(values.len() * 4)?;
        // SAFETY: the fresh mapping holds the values and is exclusively ours.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), buf.ptr.cast(), values.len()) };
        Ok(buf)
    }
    /// The first `n` u32 values.
    ///
    /// # Panics
    /// `n * 4` exceeds the mapped length.
    pub fn u32s(&self, n: usize) -> &[u32] {
        assert!(n * 4 <= self.len);
        // SAFETY: page-aligned, in bounds, initialized (zero or written).
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), n) }
    }
}

impl Drop for HugeBuf {
    fn drop(&mut self) {
        // SAFETY: mapped in `zeroed` with this length.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}
