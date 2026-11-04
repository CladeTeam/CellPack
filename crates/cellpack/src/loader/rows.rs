//! Cached rows in frames: a row is a list of whole frames, not necessarily
//! contiguous, and no frame holds parts of two rows.
//!
//! - A sparse row stores its elements in order, `per_frame` to a frame:
//!   each frame starts with its element count (`u32`, then 4 bytes of
//!   padding), then the elements' standard ids, then their values in the
//!   stored type.
//! - A dense row stores its values in order, `page / width` to a frame.
//!
//! A plan gives the frames; their number follows from the element count, so
//! the compiler and the workers agree on it.

/// Bytes before a sparse frame's ids: the element count and 4 bytes of padding;
/// with the even `per` of [`sparse_per_frame`] this puts the value run on an
/// 8-byte boundary.
const HEADER: usize = 8;

/// Elements of a sparse row per frame: ids, then values aligned to their
/// width. `per` is even so that `HEADER + 4 * per` is 8-byte aligned, which
/// lets a value of any stored width be read as one aligned word.
#[inline]
pub(crate) fn sparse_per_frame(width: usize, page: usize) -> usize {
    ((page - HEADER) / (4 + width)) & !1
}

/// Frames a row of `elements` takes; at least one, so every row has a frame to
/// name even when it holds no element.
#[inline]
pub(crate) fn frames(dense: bool, elements: usize, width: usize, page: usize) -> usize {
    let per = if dense { page / width } else { sparse_per_frame(width, page) };
    elements.div_ceil(per).max(1)
}

/// Payload bytes of a row (what reading it costs in memory traffic).
#[inline]
pub(crate) fn payload(dense: bool, elements: usize, width: usize) -> usize {
    elements * if dense { width } else { 4 + width }
}

/// Whether a dense record's row is kept dense: when that is no larger than
/// its non-zeros with ids.
#[inline]
pub(crate) fn keep_dense(genes: usize, nnz: usize, width: usize) -> bool {
    genes * width <= nnz * (4 + width)
}

/// Writes a sparse row into `frames` (pointers to whole frames).
///
/// # Safety
/// `frames` holds `frames(false, ids.len(), width, page)` distinct frames, each
/// writable, and `values` holds `ids.len()` values of `width` bytes.
pub(crate) unsafe fn write_sparse(frames: &[*mut u8], ids: &[u32], values: &[u8], width: usize, page: usize) {
    let per = sparse_per_frame(width, page);
    for (i, &f) in frames.iter().enumerate() {
        let lo = (i * per).min(ids.len());
        let hi = ((i + 1) * per).min(ids.len());
        let n = hi - lo;
        // SAFETY: as documented; the frame holds the header, `per` ids and
        // `per` values.
        unsafe {
            f.cast::<u32>().write(n as u32);
            std::ptr::copy_nonoverlapping(ids.as_ptr().add(lo), f.add(HEADER).cast::<u32>(), n);
            std::ptr::copy_nonoverlapping(values.as_ptr().add(lo * width), f.add(HEADER + 4 * per), n * width);
        }
    }
}

/// Writes a dense row of `values.len() / width` values into `frames`.
///
/// # Safety
/// As for [`write_sparse`], with `frames(true, n, width, page)` frames.
pub(crate) unsafe fn write_dense(frames: &[*mut u8], values: &[u8], width: usize, page: usize) {
    let per = page / width * width;
    for (i, &f) in frames.iter().enumerate() {
        let lo = (i * per).min(values.len());
        let hi = ((i + 1) * per).min(values.len());
        // SAFETY: as documented.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr().add(lo), f, hi - lo) };
    }
}

/// Gathers a sparse row from its frames into `ids` and `values`; returns
/// its element count.
///
/// # Safety
/// `frames` hold a sparse row written by [`write_sparse`] with this width and page size,
/// `ids` and `values` have room for the row.
pub(crate) unsafe fn gather_sparse(frames: &[*mut u8], ids: *mut u32, values: *mut u8, width: usize, page: usize) -> usize {
    let per = sparse_per_frame(width, page);
    let mut n = 0;
    for &f in frames {
        // SAFETY: as documented.
        unsafe {
            let k = f.cast::<u32>().read() as usize;
            debug_assert!(k <= per);
            std::ptr::copy_nonoverlapping(f.add(HEADER).cast::<u32>(), ids.add(n), k);
            std::ptr::copy_nonoverlapping(f.add(HEADER + 4 * per), values.add(n * width), k * width);
            n += k;
        }
    }
    n
}

/// Gathers a dense row of `n` values.
///
/// # Safety
/// As for [`gather_sparse`], for a row written by [`write_dense`].
pub(crate) unsafe fn gather_dense(frames: &[*mut u8], n: usize, values: *mut u8, width: usize, page: usize) {
    let per = page / width;
    for (i, &f) in frames.iter().enumerate() {
        let lo = (i * per).min(n);
        let hi = ((i + 1) * per).min(n);
        // SAFETY: as documented.
        unsafe { std::ptr::copy_nonoverlapping(f, values.add(lo * width), (hi - lo) * width) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_round_trip_through_frames() {
        for page in [512usize, 4096, 8192, 65536] {
            for width in [1usize, 2, 4, 8] {
                for n in [1usize, 5, sparse_per_frame(width, page), sparse_per_frame(width, page) + 1, 3 * sparse_per_frame(width, page) + 7] {
                    let ids: Vec<u32> = (0..n as u32).map(|i| i * 3 + 1).collect();
                    let values: Vec<u8> = (0..n * width).map(|i| (i * 7) as u8).collect();
                    let k = frames(false, n, width, page);
                    let mut mem = vec![0u8; k * page + page];
                    let base = mem.as_mut_ptr().wrapping_add(mem.as_ptr().align_offset(8));
                    // Frames in reverse order: they need not be contiguous.
                    let fs: Vec<*mut u8> = (0..k).rev().map(|i| base.wrapping_add(i * page)).collect();
                    // SAFETY: `fs` are distinct whole frames of `mem`, enough for the row.
                    unsafe { write_sparse(&fs, &ids, &values, width, page) };
                    let (mut gi, mut gv) = (vec![0u32; n], vec![0u8; n * width]);
                    // SAFETY: the frames hold the row just written; the outputs hold it.
                    assert_eq!(unsafe { gather_sparse(&fs, gi.as_mut_ptr(), gv.as_mut_ptr(), width, page) }, n);
                    assert_eq!((gi, gv), (ids, values.clone()));
                    let k = frames(true, n, width, page);
                    let fs: Vec<*mut u8> = (0..k).map(|i| base.wrapping_add(i * page)).collect();
                    // SAFETY: as above, for the dense row.
                    unsafe { write_dense(&fs, &values, width, page) };
                    let mut gv = vec![0u8; n * width];
                    // SAFETY: as above.
                    unsafe { gather_dense(&fs, n, gv.as_mut_ptr(), width, page) };
                    assert_eq!(gv, values);
                }
            }
        }
    }
}
