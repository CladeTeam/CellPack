//! Inspectable reservation estimates; these are not RSS or a total memory limit.
use super::{Loader, Planning, Result, error::config};

/// Bytes a run reserves, by purpose. Each term is a known reservation, not a
/// measured footprint: allocator overhead, the compiler's own state and the OS
/// page cache are outside it, and reserved bytes need not be resident.
#[derive(Clone, Debug, Default)]
pub struct MemoryEstimate {
    /// The plan's cache frames, `frames × page_size`.
    pub cache_frame_bytes: usize,
    /// One generation word per cache frame.
    pub generation_bytes: usize,
    /// Every worker's private frames (its misses).
    pub private_frame_bytes: usize,
    /// Conversion, decode and row scratch of every worker, including each
    /// worker's process scratch.
    pub worker_scratch_bytes: usize,
    /// The plan ring between the plan's producer and the workers.
    pub plan_ring_bytes: usize,
    /// Chunk metadata, decode tables, the column arena, the mode words, panel
    /// maps and schema tables.
    pub metadata_bytes: usize,
    /// The output ring, control and data.
    pub output_ring_bytes: usize,
    /// The sum of the terms above.
    pub known_reservation_bytes: usize,
}
/// Checked `a + b`, reported as a configuration error instead of wrapping.
fn sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| config("memory estimate overflow"))
}
/// Checked `a × b`, reported as a configuration error instead of wrapping.
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| config("memory estimate overflow"))
}
impl Loader {
    /// Estimates a page-only run with default planning options and this
    /// cache budget (0: no plan). Use `memory_estimate_for` for a specific
    /// plan, including zero-budget plans.
    pub fn memory_estimate(&self, cache_bytes: usize, slots: u32, world: u32) -> Result<MemoryEstimate> {
        let planning = if cache_bytes == 0 { Planning::None } else { Planning::Online(Box::new(self.plan_options(cache_bytes))) };
        self.memory_estimate_for(&planning, slots, world)
    }

    /// Known reservations for one run under this planning, with the window
    /// and chunk size the loader's config and the options resolve to. This
    /// does not bind a fixed plan to an access sequence.
    pub fn memory_estimate_for(&self, planning: &Planning, slots: u32, world: u32) -> Result<MemoryEstimate> {
        let g = &self.graph;
        let c = &self.config;
        let s = &g.source;
        let (frames, window, chunk, blocks) = match planning {
            Planning::None => (0, self.gap_lanes(self.window()), crate::opt::DEFAULT_CHUNK, 1),
            Planning::Fixed(p) => {
                let window = c.window.map_or(p.distance(), |w| self.gap_lanes(w));
                if window > p.distance() { return Err(config("the execution window exceeds the plan's reuse gap")); }
                (p.cache_frames() as usize, window, crate::opt::DEFAULT_CHUNK, crate::opt::plan::MAX_BLOCK_PAGES)
            }
            Planning::Online(o) => {
                crate::opt::compile::check(g, o)?;
                (o.cache_bytes / g.page_size, self.runtime_window(o.gap)?, o.chunk.unwrap_or(crate::opt::DEFAULT_CHUNK), o.block_pages.unwrap_or(1))
            }
        };
        let rows = super::exec::uses_rows(planning, g.page_size);
        let cache_frame_bytes = mul(frames, g.page_size)?;
        let generation_bytes = mul(frames, 4)?;
        let schemas = g.schemas.len();
        // Private frames are sized by the records' 99.9th percentile in
        // pages, plus one largest record per schema (a worst-case position).
        let private = mul(sum(mul(c.max_inflight_per_worker, s.normal_pages())?, s.max_pages.max(1))?, schemas)?;
        let private_frame_bytes = mul(mul(c.workers, private)?, g.page_size)?;
        let elements = s.max_nnz.max(s.max_genes);
        let dense = g.schemas.iter().map(|x| x.out_dim() * x.dtype().width()).max().unwrap_or(0);
        let process = g.schemas.iter().map(|x| x.scratch_bytes()).max().unwrap_or(0);
        let rowcap = sum(elements, 1)?;
        // Per worker: decode workspace, output columns, converted values,
        // dictionary, expanded stored values, one dense row, process scratch,
        // a slice per record page, and with rows the cached row's ids and
        // values (4 + 8 bytes per element).
        let worker_scratch_bytes = mul(c.workers, [
            mul(s.workspace_words, 8)?, mul(elements, 4)?, mul(rowcap, 8)?,
            mul(sum(s.max_nnz, 1)?, 8)?, mul(sum(s.max_genes, 1)?, 8)?,
            mul(sum(dense.div_ceil(8), 1)?, 8)?, process.max(1),
            mul(s.max_pages, std::mem::size_of::<&[u8]>())?,
            if rows { mul(rowcap, 12)? } else { 0 },
        ].into_iter().try_fold(0, sum)?)?;
        let plan_ring_bytes = if matches!(planning, Planning::None) { 0 } else {
            let (entries, words) = super::exec::plan_ring_sizes(g, window, chunk, rows, blocks)?;
            sum(mul(entries as usize, 16)?, mul(words as usize, 8)?)?
        };
        let tables = g.panels.iter().map(|p| p.len * 4).sum::<usize>() + g.tables.iter().map(|t| t.std_to_out.len() * 4 + t.mode_words.len() * 8).sum::<usize>();
        let metadata_bytes = sum(s.resident_metadata_bytes(), tables)?;
        let output_ring_bytes = self.ring_layout(slots, world).bytes()?;
        let known_reservation_bytes = [cache_frame_bytes, generation_bytes, private_frame_bytes, worker_scratch_bytes, plan_ring_bytes, metadata_bytes, output_ring_bytes].into_iter().try_fold(0, sum)?;
        Ok(MemoryEstimate { cache_frame_bytes, generation_bytes, private_frame_bytes, worker_scratch_bytes, plan_ring_bytes, metadata_bytes, output_ring_bytes, known_reservation_bytes })
    }
}
