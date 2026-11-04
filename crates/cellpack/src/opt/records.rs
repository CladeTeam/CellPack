//! What the compiler knows about accesses: the record of every cell (fixed
//! by the inputs), and the locality of a sequence (which accesses meet cold
//! chunks).

use crate::loader::catalog::CellId;
use crate::loader::convert::dtype_index;
use crate::loader::error::{Error, Result};
use crate::loader::rows;
use crate::loader::schema::Graph;

/// One cell's record.
#[derive(Clone, Copy, Default)]
pub(crate) struct Record {
    /// Physical id of the record's first page.
    pub first_page: u32,
    /// Pages the record spans; every one of them is read to decode it.
    pub pages: u16,
    /// [`EMPTY`], [`DENSE`], [`IMAGE_DENSE`] and the 2-bit value encoding.
    flags: u8,
    /// The stored type's [`dtype_index`].
    pub dtype: u8,
    /// The record's chunk.
    pub chunk: u32,
    /// Stored elements (non-zeros, or every gene of a dense record).
    pub elements: u32,
    /// Elements of the cached image (dense genes or compacted non-zeros).
    pub image_elements: u32,
    /// Payload bytes of the cell's cached row.
    pub row_bytes: u32,
    /// Frames of the cell's cached row.
    pub row_frames: u32,
}

// Flag bits: 0 empty, 1 dense, 2-3 the value encoding, 4 dense image.
const EMPTY: u8 = 1;
const DENSE: u8 = 2;
const IMAGE_DENSE: u8 = 16;

impl Record {
    /// The record holds no element (nothing to decode).
    #[inline]
    pub fn empty(&self) -> bool {
        self.flags & EMPTY != 0
    }
    /// The record stores every gene of its chunk densely, not a sorted list
    /// of non-zeros.
    #[inline]
    pub fn dense(&self) -> bool {
        self.flags & DENSE != 0
    }
    /// The record's value encoding, as the delivery route takes it (0 is
    /// raw).
    #[inline]
    pub fn value_encoding(&self) -> u8 {
        (self.flags >> 2) & 3
    }
    /// The cached row is the dense image of the record (every gene) rather
    /// than its compacted non-zeros.
    #[inline]
    pub fn image_dense(&self) -> bool {
        self.flags & IMAGE_DENSE != 0
    }
    /// The record's physical pages, in file order.
    #[inline]
    pub fn page_range(&self) -> std::ops::Range<u64> {
        self.first_page as u64..self.first_page as u64 + self.pages as u64
    }
}

/// Chunks whose decode tables and column maps a worker's caches keep warm.
pub(crate) const WARM_CHUNKS: usize = 32;

/// Every cell's record.
pub(crate) struct Records {
    /// Cells of the graph's source; access sequences are checked against it.
    pub cells: u64,
    table: Vec<Record>,
}

impl Records {
    /// Reads the graph's catalog and fills one record per cell, split over
    /// `threads`; errors on inputs plans cannot index (2^32 pages, records
    /// over 65535 pages or rows over 4 GiB).
    pub fn build(graph: &Graph, threads: usize) -> Result<Self> {
        let cells = graph.source.n_cells();
        if graph.pages >= u32::MAX as u64 {
            return Err(Error::Plan("plans support fewer than 2^32 physical pages".into()));
        }
        let mut table = vec![Record::default(); cells as usize];
        let per = (cells as usize).div_ceil(threads.max(1)).max(1);
        std::thread::scope(|s| {
            let handles: Vec<_> = table
                .chunks_mut(per)
                .enumerate()
                .map(|(k, out)| {
                    s.spawn(move || -> Result<()> {
                        for (i, row) in out.iter_mut().enumerate() {
                            let cell = (k * per + i) as CellId;
                            let (loc, _, first_page) = graph.locate(0, cell)?;
                            let pages = u16::try_from(loc.page_count).map_err(|_| Error::Plan("records span at most 65535 pages".into()))?;
                            let mut flags = loc.value_encoding << 2;
                            if loc.len == 0 {
                                flags |= EMPTY;
                            }
                            if loc.dense {
                                flags |= DENSE;
                            }
                            let elements = if loc.dense { loc.chunk.meta().n_dims() as usize } else { loc.nnz };
                            let dense = graph.cached_dense(loc.chunk_index, loc.dense, loc.nnz);
                            if dense { flags |= IMAGE_DENSE; }
                            let image_elements = if dense { elements } else { loc.nnz };
                            let width = loc.chunk.dtype.width();
                            let row_bytes = u32::try_from(rows::payload(dense, image_elements, width)).map_err(|_| Error::Plan("cached row exceeds 4 GiB".into()))?;
                            let row_frames = rows::frames(dense, image_elements, width, graph.page_size) as u32;
                            *row = Record { first_page: first_page as u32, pages, flags, dtype: dtype_index(loc.chunk.dtype) as u8, chunk: loc.chunk_index, elements: elements as u32, image_elements: image_elements as u32, row_bytes, row_frames };
                        }
                        Ok(())
                    })
                })
                .collect();
            handles.into_iter().try_for_each(|h| h.join().unwrap_or(Err(Error::WorkerPanic)))
        })?;
        Ok(Self { cells, table })
    }

    /// The record of a cell; access sequences are checked against the cell
    /// count before they reach the compiler.
    #[inline(always)]
    pub fn of(&self, cell: CellId) -> &Record {
        debug_assert!((cell as usize) < self.table.len());
        // SAFETY: cell ids reach the compiler only after `Trace::build` or
        // `generate` has rejected those past the source's cell count.
        unsafe { self.table.get_unchecked(cell as usize) }
    }
}

/// Cold bits of accesses `start..start + cells.len()` (one byte each):
/// whether the record's chunk is not among the `WARM_CHUNKS` most recently
/// used chunks, in sequence order. `history` holds the cells before `start`
/// (most recent last); only enough of it to find the warm chunks is read.
pub(crate) fn cold_bits(records: &Records, history: impl Fn(u64) -> Option<CellId>, start: u64, cells: &[CellId], out: &mut [u8]) {
    debug_assert_eq!(out.len(), cells.len());
    // The most recent distinct chunks before `start`, most recent first.
    let mut recent: Vec<u32> = Vec::with_capacity(WARM_CHUNKS);
    let mut at = start;
    while recent.len() < WARM_CHUNKS && at > 0 && start - at < WARM_SCAN {
        at -= 1;
        let Some(cell) = history(at) else { break };
        let chunk = records.of(cell).chunk;
        if !recent.contains(&chunk) {
            recent.push(chunk);
        }
    }
    for (o, &cell) in out.iter_mut().zip(cells) {
        let chunk = records.of(cell).chunk;
        *o = match recent.iter().position(|&c| c == chunk) {
            Some(k) => {
                // Move the chunk to the most recent end of the list.
                recent[..=k].rotate_right(1);
                0
            }
            None => {
                if recent.len() == WARM_CHUNKS {
                    recent.pop();
                }
                recent.insert(0, chunk);
                1
            }
        };
    }
}

/// Accesses searched back for warm chunks; a sequence that names fewer
/// distinct chunks over this span keeps them all warm.
const WARM_SCAN: u64 = 1 << 16;

/// Accesses whose cold bits one task computes; spans start at multiples of
/// it, so the bits are the same however they are scheduled.
pub(crate) const COLD_SPAN: usize = 1 << 16;

/// A finite access sequence with its cold bits.
pub(crate) struct Trace {
    /// The cells in access order.
    pub cells: Vec<CellId>,
    /// Per access: nonzero when its chunk is cold (see [`cold_bits`]).
    cold: Vec<u8>,
}

impl Trace {
    /// Copies the sequence and computes its cold bits over `threads`; errors
    /// on a cell index past the source's cell count.
    pub fn build(records: &Records, accesses: &[CellId], threads: usize) -> Result<Self> {
        if let Some((at, &cell)) = accesses.iter().enumerate().find(|(_, c)| **c as u64 >= records.cells) {
            return Err(Error::Access { position: at as u64, cell: cell as u64 });
        }
        let mut cold = vec![0u8; accesses.len()];
        // Fixed spans, so the bits do not depend on the thread count.
        let spans: Vec<(usize, &mut [u8])> = cold.chunks_mut(COLD_SPAN).enumerate().collect();
        let next = std::sync::Mutex::new(spans.into_iter());
        std::thread::scope(|s| {
            for _ in 0..threads.clamp(1, accesses.len().div_ceil(COLD_SPAN).max(1)) {
                s.spawn(|| {
                    while let Some((k, out)) = { next.lock().unwrap().next() } {
                        let start = k * COLD_SPAN;
                        let cells = &accesses[start..(start + COLD_SPAN).min(accesses.len())];
                        cold_bits(records, |at| Some(accesses[at as usize]), start as u64, cells, out);
                    }
                });
            }
        });
        Ok(Self { cells: accesses.to_vec(), cold })
    }

    /// Accesses in the sequence.
    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the access at `at` decodes a cold chunk.
    #[inline]
    pub fn cold(&self, at: usize) -> bool {
        self.cold[at] != 0
    }
}
