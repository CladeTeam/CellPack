//! Lookup structures derived from metadata. A chunk builds them once and
//! every record decode or encode borrows them; everything per cell runs on
//! caller buffers.

use crate::error::{Error, Result};
use crate::format::{Meta, N_MODELS};
use crate::runtime::Runtime;
use crate::scans::kernels::Kernels;
use crate::scans::rans::{EncModel, M, Slots};

/// Normalizes symbol counts to frequencies summing to `M`. Every observed
/// symbol keeps a non-zero frequency; unobserved ones get none, so the
/// probability mass goes only to symbols the chunk uses. All-zero counts
/// give an all-zero (unused) model.
pub(crate) fn normalize(counts: &[u64; 256]) -> [u16; 256] {
    let total: u64 = counts.iter().sum();
    let mut f = [0u32; 256];
    if total == 0 {
        return [0; 256];
    }
    let mut sum = 0u32;
    for s in 0..256 {
        if counts[s] > 0 {
            f[s] = ((counts[s] as u128 * M as u128 / total as u128) as u32).max(1);
            sum += f[s];
        }
    }
    // Rounding up rare symbols can overshoot; take the excess from the
    // largest frequencies, which loses the least.
    while sum > M {
        let s = (0..256).max_by_key(|&s| f[s]).unwrap();
        f[s] -= 1;
        sum -= 1;
    }
    let top = (0..256).max_by_key(|&s| counts[s]).unwrap();
    f[top] += M - sum;
    f.map(|v| v as u16)
}

/// Decode-side lookup tables of one chunk.
///
/// Built from a [`Meta`] once per chunk and borrowed by every record decode.
/// `model_key`, `n_genes` and `renumbered` identify the metadata the tables
/// came from, so a caller rebinding them to another chunk is rejected in
/// constant time (see [`DecodeTables::check`]).
pub struct DecodeTables {
    model_key: u64,
    n_genes: u32,
    renumbered: bool,
    used: [bool; N_MODELS],
    slots: Box<[Slots]>,
    /// Source gene of each coded id; empty unless renumbered.
    source_of_coded: Box<[u32]>,
    /// Coded id of each source gene; empty unless renumbered.
    coded_of_source: Box<[u32]>,
    /// The kernel set records of this chunk decode with.
    pub(crate) kernels: &'static Kernels,
}

/// `(source_of_coded, coded_of_source)`; both empty unless renumbered.
type GeneMaps = (Box<[u32]>, Box<[u32]>);

/// Both directions of the gene order, checking that it is a permutation.
fn gene_maps(meta: &Meta) -> Result<GeneMaps> {
    if !meta.renumbered() {
        return Ok((Box::default(), Box::default()));
    }
    let n = meta.n_dims() as usize;
    let soc: Box<[u32]> = (0..n).map(|c| meta.source_dim(c)).collect();
    let mut cos = vec![u32::MAX; n].into_boxed_slice();
    for (c, &s) in soc.iter().enumerate() {
        let slot = cos
            .get_mut(s as usize)
            .ok_or(Error::CorruptMeta("gene order entry out of range"))?;
        if *slot != u32::MAX {
            return Err(Error::CorruptMeta("gene order is not a permutation"));
        }
        *slot = c as u32;
    }
    Ok((soc, cos))
}

/// Slot table of frequencies summing to `M`.
///
/// Slot `c` of a symbol that owns `[cum, cum + f)` holds `f - 1`, `cum` and
/// the symbol, which is all a decode step needs after masking the state.
pub(crate) fn fill_slots(freq: &[u16; 256], slots: &mut Slots) {
    let mut cum = 0u32;
    for (s, &f) in freq.iter().enumerate() {
        let f = f as u32;
        if f > 0 {
            let e = (f - 1) | (cum << 12) | ((s as u32) << 24);
            slots[cum as usize..(cum + f) as usize].fill(e);
        }
        cum += f;
    }
}

/// Start of each symbol's frequency interval, as a reference encoder needs.
#[cfg(test)]
pub(crate) fn cumulative(freq: &[u16; 256]) -> [u16; 256] {
    let mut c = [0u16; 256];
    let mut acc = 0u32;
    for s in 0..256 {
        c[s] = acc as u16;
        acc += freq[s] as u32;
    }
    c
}

/// Decode tables of a chunk, bound to the process kernel set
/// ([`crate::runtime::get`]).
pub fn decode_tables(meta: &Meta) -> Result<DecodeTables> {
    decode_tables_for(meta, crate::runtime::get()?)
}

/// [`decode_tables`] bound to the kernel set of `rt` (ablation and
/// equivalence checks across ISAs).
pub fn decode_tables_for(meta: &Meta, rt: &Runtime) -> Result<DecodeTables> {
    let mut slots = vec![[0u32; M as usize]; N_MODELS].into_boxed_slice();
    let mut used = [false; N_MODELS];
    for m in 0..N_MODELS {
        let freq = meta.freqs(m);
        used[m] = freq.iter().any(|&f| f != 0);
        if used[m] {
            fill_slots(&freq, &mut slots[m]);
        }
    }
    let (source_of_coded, coded_of_source) = gene_maps(meta)?;
    Ok(DecodeTables {
        model_key: meta.model_key(),
        n_genes: meta.n_dims(),
        renumbered: meta.renumbered(),
        used,
        slots,
        source_of_coded,
        coded_of_source,
        kernels: rt.kernels(),
    })
}

impl DecodeTables {
    /// Heap payload retained by the derived tables, excluding allocator overhead.
    pub fn allocated_bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<Slots>() + (self.source_of_coded.len() + self.coded_of_source.len()) * 4
    }

    /// Rejects tables derived from other metadata (another chunk, or the
    /// same chunk re-parsed with different models).
    pub fn check(&self, meta: &Meta) -> Result<()> {
        if self.model_key != meta.model_key()
            || self.n_genes != meta.n_dims()
            || self.renumbered != meta.renumbered()
        {
            return Err(Error::TablesMismatch);
        }
        Ok(())
    }

    /// Slot table of a model the chunk uses.
    ///
    /// # Safety
    /// `model < N_MODELS` and the chunk uses it (its records have a stream
    /// for it).
    #[inline(always)]
    pub(crate) unsafe fn slots_unchecked(&self, model: usize) -> &Slots {
        debug_assert!(self.used[model], "stream for a model the chunk does not use");
        // SAFETY: as documented.
        unsafe { self.slots.get_unchecked(model) }
    }

    /// Asks the processor to start loading the slot tables a record decode
    /// reads, so a caller can overlap the loads with other work.
    pub fn prefetch(&self) {
        for (m, slots) in self.slots.iter().enumerate() {
            if self.used[m] {
                prefetch(slots.as_ptr().cast(), size_of::<Slots>());
            }
        }
    }

    /// Source gene of each coded id, or `None` without renumbering (coded
    /// ids are source ids). Callers decoding in [`IdSpace::Coded`](crate::scans::IdSpace)
    /// compose their own gene mapping with this.
    pub fn source_of_coded(&self) -> Option<&[u32]> {
        self.renumbered.then_some(&self.source_of_coded[..])
    }

    /// Coded id of each source gene, or `None` without renumbering.
    pub fn coded_of_source(&self) -> Option<&[u32]> {
        self.renumbered.then_some(&self.coded_of_source[..])
    }
}

/// Encode-side lookup tables of one chunk: an `EncModel` per model and the
/// gene order records are written in. As with [`DecodeTables`], the metadata
/// key lets [`EncodeTables::check`] reject reuse across chunks.
pub struct EncodeTables {
    model_key: u64,
    n_genes: u32,
    renumbered: bool,
    /// Encoder constants of each model, in model order.
    pub(crate) models: Box<[EncModel; N_MODELS]>,
    coded_of_source: Box<[u32]>,
    /// The kernel set records of this chunk encode with.
    pub(crate) kernels: &'static Kernels,
}

/// Encode tables of a chunk, bound to the process kernel set.
pub fn encode_tables(meta: &Meta) -> Result<EncodeTables> {
    encode_tables_for(meta, crate::runtime::get()?)
}

/// [`encode_tables`] bound to the kernel set of `rt`.
pub fn encode_tables_for(meta: &Meta, rt: &Runtime) -> Result<EncodeTables> {
    let models = Box::new(std::array::from_fn(|m| EncModel::new(&meta.freqs(m))));
    let (_, coded_of_source) = gene_maps(meta)?;
    Ok(EncodeTables {
        model_key: meta.model_key(),
        n_genes: meta.n_dims(),
        renumbered: meta.renumbered(),
        models,
        coded_of_source,
        kernels: rt.kernels(),
    })
}

impl EncodeTables {
    /// Rejects tables derived from other metadata, as
    /// [`DecodeTables::check`] does for the decode side.
    pub fn check(&self, meta: &Meta) -> Result<()> {
        if self.model_key != meta.model_key()
            || self.n_genes != meta.n_dims()
            || self.renumbered != meta.renumbered()
        {
            return Err(Error::TablesMismatch);
        }
        Ok(())
    }

    /// Coded id of each source gene; `None` denotes the identity order.
    pub fn coded_of_source(&self) -> Option<&[u32]> {
        self.renumbered.then_some(&self.coded_of_source[..])
    }
}

/// Hints the processor to load `len` bytes from `p` into its caches.
#[inline]
pub fn prefetch(p: *const u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    for off in (0..len).step_by(64) {
        // SAFETY: prefetching has no architectural effect, even for addresses
        // outside the allocation.
        unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.wrapping_add(off).cast()) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (p, len);
}
