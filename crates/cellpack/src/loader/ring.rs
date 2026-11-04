//! The output ring: bounded slots over owned or caller-provided memory. A
//! slot holds `slot_cells` consecutive positions of every schema: one
//! compact `[slot_cells, cell_bytes]` region per schema at a [`REGION_ALIGN`]
//! multiple within the slot, and a slot stride that is a multiple of
//! [`SLOT_ALIGN`], so a page-aligned data region makes every slot and region
//! aligned. A slot is ready once every schema's output of every position in
//! it is written; a ready slot is leased to one consumer and reused after
//! its guard drops. Control and data can live in separate shared mappings;
//! pointers are local to each attachment and are never stored in shared
//! headers.
//!
//! The control also holds one participant row per consumer rank and one
//! for the producer (process, its start time, state and run generation).
//! Rows change when a participant joins or leaves a run, and once when a
//! consumer adopts the run's generation, never per batch; they are read only
//! by waits already blocked, so a wait fails with the reason instead of
//! hanging when a participant died or left early.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::*};
use std::time::{Duration, Instant};

use crate::loader::error::{Error, Result, config};
use crate::loader::memory::HugeBuf;
use crate::loader::schema::MAX_SCHEMAS;

const MAGIC: u64 = u64::from_le_bytes(*b"SCLRING1");
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 4096;
// Slot states. FILLING while outputs are still being written, READY once the
// last one is, LEASED while one consumer holds it. DONE marks a slot opened
// for a batch with no positions (the tail of a run), which consumers of that
// batch observe as the end.
const FILLING: u32 = 0;
const READY: u32 = 1;
const DONE: u32 = 2;
const LEASED: u32 = 3;
/// Stored in the header's `reserved` word: the control and data regions were
/// initialized separately (and may be attached by another process).
const EXTERNAL: u32 = 1;
/// Alignment of each schema's region within a slot.
pub const REGION_ALIGN: usize = 64;
/// Alignment of slots.
pub const SLOT_ALIGN: usize = 4096;

/// The ring's header, at the start of control memory: geometry, the current
/// run, and per-schema region offsets and cell sizes.
///
/// Fields hold offsets and counts only, never pointers, so the same mapping
/// is valid in every process that attaches it. Non-atomic fields are written
/// once, before any participant joins.
#[repr(C)]
pub struct RingHeader {
    /// `MAGIC`, so an unrelated mapping is rejected on attach.
    pub magic: u64,
    /// `VERSION` of this layout.
    pub version: u32,
    /// Slots, and batch `b`'s slot number is `b % slots`.
    pub slots: u32,
    /// Positions per slot.
    pub slot_cells: u32,
    /// Consumer ranks.
    pub world: u32,
    /// Schemas, one region each per slot.
    pub schemas: u32,
    /// `EXTERNAL` when control and data were initialized as separate regions;
    /// 0 for one contiguous mapping.
    pub reserved: u32,
    /// Bytes from one slot's start to the next.
    pub slot_stride: u64,
    /// Offset of the first slot header from the start of control memory.
    pub slot_offset: u64,
    /// Offset of the first slot's data from the start of the mapping.
    pub data_offset: u64,
    /// Bytes of control and data together, as one mapping.
    pub total_bytes: u64,
    /// Positions of the current run.
    pub positions: AtomicU64,
    /// Nonzero once any participant aborted the ring; never cleared.
    pub abort: AtomicU32,
    /// 1 while a producer holds the ring, 0 otherwise.
    pub producer: AtomicU32,
    /// The current or last run's generation.
    pub generation: AtomicU64,
    /// Per schema: its region's offset within a slot.
    pub offsets: [u64; MAX_SCHEMAS],
    /// Per schema: bytes of one cell's output.
    pub cell_bytes: [u32; MAX_SCHEMAS],
}

/// One slot's header, on its own cache line: which batch it holds, its state
/// and how many of its outputs are still unwritten.
#[repr(C, align(64))]
pub struct SlotHeader {
    /// The batch the slot was opened for, [`u64::MAX`] before any run.
    pub seq: AtomicU64,
    /// One of the slot states: filling, ready, done or leased.
    pub state: AtomicU32,
    /// Positions in the slot.
    pub rows: AtomicU32,
    /// Outputs (positions × schemas) not yet written.
    pub remaining: AtomicU32,
    /// The run generation the slot was opened under.
    pub generation: AtomicU64,
}

/// Slots of positions, each holding every schema's region. A slot's remaining
/// output counter is a `u32`, so `slot_cells × schemas` must fit in one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingLayout {
    /// Slot count; batch `b` uses slot `b % slots`.
    pub slots: u32,
    /// Positions per slot: one compact region per schema in each slot.
    pub slot_cells: u32,
    /// Consumer ranks; rank `r` consumes the batches `b` with `b % world == r`.
    pub world: u32,
    /// Bytes of one cell's output, per schema in loader order.
    pub cell_bytes: Vec<u32>,
}

impl RingLayout {
    /// Region offsets within a slot, and the slot stride: each schema's
    /// `slot_cells × cell_bytes` region starts at a [`REGION_ALIGN`] multiple
    /// within the slot, and the stride (all regions plus the tail padding) is
    /// a multiple of [`SLOT_ALIGN`], so slots of a page-aligned data region
    /// never share a page.
    pub fn regions(&self) -> Option<(Vec<usize>, usize)> {
        let mut offsets = Vec::with_capacity(self.cell_bytes.len());
        let mut at = 0usize;
        for &b in &self.cell_bytes {
            offsets.push(at);
            at = at.checked_add((self.slot_cells as usize).checked_mul(b as usize)?)?.checked_next_multiple_of(REGION_ALIGN)?;
        }
        Some((offsets, at.checked_next_multiple_of(SLOT_ALIGN)?))
    }
    /// Placement of the parts as (slot header offset, data offset, total
    /// bytes): a [`HEADER_BYTES`] header, one 64-byte slot header per slot,
    /// one 64-byte participant row per consumer rank plus the producer's, then
    /// the slots, whose start is [`SLOT_ALIGN`]-aligned. `None` on overflow or
    /// an unaddressable total.
    fn offsets(&self) -> Option<(usize, usize, usize)> {
        let (_, stride) = self.regions()?;
        let participants = HEADER_BYTES.checked_add((self.slots as usize).checked_mul(64)?)?;
        let data = participants.checked_add((self.world as usize).checked_add(1)?.checked_mul(64)?)?.checked_next_multiple_of(SLOT_ALIGN)?;
        let total = data.checked_add((self.slots as usize).checked_mul(stride)?)?;
        (total <= isize::MAX as usize).then_some((HEADER_BYTES, data, total))
    }
    /// Bytes from one slot's start to the next.
    pub fn slot_stride(&self) -> Result<usize> {
        self.validate()?;
        Ok(self.regions().unwrap().1)
    }
    /// Bytes of one contiguous mapping: control and data together.
    pub fn bytes(&self) -> Result<usize> {
        self.validate()?;
        Ok(self.offsets().unwrap().2)
    }
    /// Bytes of control memory: header, slot headers and participant rows.
    pub fn control_bytes(&self) -> Result<usize> {
        self.validate()?;
        Ok(self.offsets().unwrap().1)
    }
    /// Bytes of data storage, padding of the last slot included.
    pub fn data_bytes(&self) -> Result<usize> {
        Ok(self.slots as usize * self.slot_stride()?)
    }
    /// The least data storage: the last slot need not hold its padding.
    pub fn data_min_bytes(&self) -> Result<usize> {
        let stride = self.slot_stride()?;
        let (offsets, _) = self.regions().unwrap();
        let last = self.cell_bytes.len() - 1;
        Ok((self.slots as usize - 1) * stride + offsets[last] + self.slot_cells as usize * self.cell_bytes[last] as usize)
    }
    fn validate(&self) -> Result<()> {
        if self.slots == 0 || self.slot_cells == 0 || self.world == 0 || self.cell_bytes.is_empty() || self.cell_bytes.len() > MAX_SCHEMAS || self.cell_bytes.contains(&0) {
            return Err(config(format!("ring needs positive slots, slot cells and world, and 1 to {MAX_SCHEMAS} schemas of positive cell bytes")));
        }
        if self.offsets().is_none() || (self.slot_cells as u64) * self.cell_bytes.len() as u64 > u32::MAX as u64 {
            return Err(config("ring size overflow"));
        }
        Ok(())
    }
}

/// A bounded ring over owned or caller-provided memory: the slot protocol and
/// the participant bookkeeping that keep one producer and `world` consumer
/// ranks apart.
///
/// Construction either allocates (`allocate`, `with_buffer`) or takes
/// caller-provided memory (`init`, `init_regions`, `attach`,
/// `attach_regions`); the latter take raw pointers unsafely, and the
/// constructors that accept an `owner` keep it alive through an `Arc`.
/// `_owned` holds memory this value allocated, `_owner` retains a caller's;
/// both drop after every lease, since a lease keeps the `Arc<Ring>` it came
/// from.
pub struct Ring {
    base: NonNull<u8>,
    data: NonNull<u8>,
    /// Bytes of data storage (the last slot may lack its padding).
    data_len: usize,
    _owned: Option<HugeBuf>,
    _owner: Option<Arc<dyn Send + Sync>>,
}

// SAFETY: outputs are written by one producer access each, then published
// through atomics; only one consumer can lease a slot, and reuse waits for
// its guard.
unsafe impl Send for Ring {}
// SAFETY: shared access follows the same slot protocol; storage is retained.
unsafe impl Sync for Ring {}

/// Waits on `word` until it differs from `expected` or `timeout` passes. A
/// shared futex, so waiters in another process over the same mapping wake too.
fn futex_wait(word: &AtomicU32, expected: u32, timeout: Duration) {
    let ts = libc::timespec { tv_sec: timeout.as_secs() as libc::time_t, tv_nsec: timeout.subsec_nanos() as libc::c_long };
    // SAFETY: word is an aligned live atomic in a shared-capable mapping.
    unsafe {
        libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAIT, expected, &ts as *const libc::timespec, std::ptr::null::<u32>(), 0);
    }
}
/// Wakes every waiter on `word`.
fn futex_wake(word: &AtomicU32) {
    // SAFETY: word is an aligned live atomic.
    unsafe {
        libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAKE, i32::MAX, std::ptr::null::<libc::timespec>(), std::ptr::null::<u32>(), 0);
    }
}

/// A region's bounds as (start, end), checked for null, alignment and
/// addressable size.
fn region(ptr: *mut u8, len: usize, alignment: usize) -> Result<(usize, usize)> {
    let start = ptr as usize;
    if start == 0 || !start.is_multiple_of(alignment) || len > isize::MAX as usize {
        return Err(config("null, misaligned or oversized ring region"));
    }
    Ok((start, start.checked_add(len).ok_or_else(|| config("ring address overflow"))?))
}
/// Whether two half-open address ranges share no byte.
fn disjoint(a: (usize, usize), b: (usize, usize)) -> bool {
    a.1 <= b.0 || b.1 <= a.0
}

impl Ring {
    /// Allocates control and data together in huge pages.
    pub fn allocate(layout: RingLayout) -> Result<Self> {
        let memory = HugeBuf::zeroed(layout.bytes()?)?;
        // SAFETY: fresh exclusive page-aligned mapping, retained below.
        let mut ring = unsafe { Self::init(memory.ptr(), memory.len(), layout)? };
        ring._owned = Some(memory);
        Ok(ring)
    }

    /// Initializes contiguous storage.
    /// # Safety
    /// Memory is exclusive during initialization, remains valid for this ring
    /// and its batches, and is accessed only through the slot protocol.
    pub unsafe fn init(ptr: *mut u8, len: usize, layout: RingLayout) -> Result<Self> {
        layout.validate()?;
        region(ptr, len, SLOT_ALIGN)?;
        let (_, data, total) = layout.offsets().unwrap();
        if len < total {
            return Err(config("ring memory too small"));
        }
        // SAFETY: bounds and alignment checked; caller provides exclusivity.
        unsafe {
            Self::write_header(ptr, &layout, 0);
            Ok(Self::from_regions(ptr, ptr.add(data), total - data, None))
        }
    }

    /// Allocates control metadata and uses an external data array of at least
    /// [`RingLayout::data_min_bytes`] and at most [`RingLayout::data_bytes`],
    /// aligned to at least 8 bytes. Regions start at [`REGION_ALIGN`]
    /// multiples within the array, so a 64-byte aligned array gives 64-byte
    /// aligned regions.
    /// # Safety
    /// The array remains allocated without reallocation while `owner` is
    /// alive. No aliases may access filling slots; ready slots follow the
    /// lease protocol.
    pub unsafe fn with_buffer(data: *mut u8, data_len: usize, layout: RingLayout, owner: Arc<dyn Send + Sync>) -> Result<Self> {
        let memory = HugeBuf::zeroed(layout.control_bytes()?)?;
        // SAFETY: fresh control mapping; the external region follows the
        // caller's contract.
        let mut ring = unsafe { Self::init_regions(memory.ptr(), memory.len(), data, data_len, layout, owner)? };
        ring._owned = Some(memory);
        Ok(ring)
    }

    /// Initializes separate control and data (also suitable for IPC).
    /// # Safety
    /// Regions are writable, exclusive during initialization, and retained by
    /// `owner`. All processes must follow this ring's slot protocol.
    pub unsafe fn init_regions(control: *mut u8, control_len: usize, data: *mut u8, data_len: usize, layout: RingLayout, owner: Arc<dyn Send + Sync>) -> Result<Self> {
        Self::check_regions(control, control_len, data, data_len, &layout)?;
        // SAFETY: validated nonoverlapping regions; caller guarantees
        // exclusive initialization.
        unsafe { Self::write_header(control, &layout, EXTERNAL) };
        Ok(Self::from_regions(control, data, data_len, Some(owner)))
    }

    fn check_regions(control: *mut u8, control_len: usize, data: *mut u8, data_len: usize, layout: &RingLayout) -> Result<()> {
        layout.validate()?;
        let a = region(control, control_len, 64)?;
        let b = region(data, data_len, 8)?;
        if control_len < layout.control_bytes()? || data_len < layout.data_min_bytes()? || data_len > layout.data_bytes()? || !disjoint(a, b) {
            return Err(config("ring regions have incorrect sizes or overlap"));
        }
        Ok(())
    }

    unsafe fn write_header(ptr: *mut u8, layout: &RingLayout, external: u32) {
        let (slot_offset, data_offset, total) = layout.offsets().unwrap();
        let (regions, stride) = layout.regions().unwrap();
        let mut offsets = [0u64; MAX_SCHEMAS];
        let mut cell_bytes = [0u32; MAX_SCHEMAS];
        for (s, (&o, &b)) in regions.iter().zip(&layout.cell_bytes).enumerate() {
            offsets[s] = o as u64;
            cell_bytes[s] = b;
        }
        // SAFETY: callers validated the exclusive control region and alignment.
        unsafe {
            std::ptr::write_bytes(ptr, 0, slot_offset + (layout.slots as usize + layout.world as usize + 1) * 64);
            ptr.cast::<RingHeader>().write(RingHeader {
                magic: MAGIC,
                version: VERSION,
                slots: layout.slots,
                slot_cells: layout.slot_cells,
                world: layout.world,
                schemas: layout.cell_bytes.len() as u32,
                reserved: external,
                slot_stride: stride as u64,
                slot_offset: slot_offset as u64,
                data_offset: data_offset as u64,
                total_bytes: total as u64,
                positions: AtomicU64::new(0),
                abort: AtomicU32::new(0),
                producer: AtomicU32::new(0),
                generation: AtomicU64::new(0),
                offsets,
                cell_bytes,
            });
            for s in 0..layout.slots as usize {
                ptr.add(slot_offset + s * 64).cast::<SlotHeader>().write(SlotHeader {
                    seq: AtomicU64::new(u64::MAX),
                    state: AtomicU32::new(FILLING),
                    rows: AtomicU32::new(0),
                    remaining: AtomicU32::new(0),
                    generation: AtomicU64::new(0),
                });
            }
        }
    }

    fn from_regions(base: *mut u8, data: *mut u8, data_len: usize, owner: Option<Arc<dyn Send + Sync>>) -> Self {
        Self { base: NonNull::new(base).unwrap(), data: NonNull::new(data).unwrap(), data_len, _owned: None, _owner: owner }
    }

    fn attached_layout(ptr: *mut u8, len: usize, external: u32) -> Result<RingLayout> {
        region(ptr, len, 64)?;
        if len < HEADER_BYTES {
            return Err(config("ring control memory too small"));
        }
        // SAFETY: aligned header-sized region; attachment requires
        // initialized metadata.
        let h = unsafe { &*ptr.cast::<RingHeader>() };
        if h.magic != MAGIC || h.version != VERSION || h.reserved != external || h.schemas as usize > MAX_SCHEMAS {
            return Err(config("not a compatible ring"));
        }
        let layout = RingLayout { slots: h.slots, slot_cells: h.slot_cells, world: h.world, cell_bytes: h.cell_bytes[..h.schemas as usize].to_vec() };
        layout.validate()?;
        let (slots, data, total) = layout.offsets().unwrap();
        let (regions, stride) = layout.regions().unwrap();
        if h.slot_offset != slots as u64
            || h.data_offset != data as u64
            || h.total_bytes != total as u64
            || h.slot_stride != stride as u64
            || regions.iter().enumerate().any(|(s, &o)| h.offsets[s] != o as u64)
            || len < if external == 0 { total } else { data }
        {
            return Err(config("invalid ring offsets or length"));
        }
        Ok(layout)
    }

    /// Attaches to initialized contiguous shared storage.
    /// # Safety
    /// Storage remains valid and unmodified except through the slot protocol.
    pub unsafe fn attach(ptr: *mut u8, len: usize) -> Result<Self> {
        let layout = Self::attached_layout(ptr, len, 0)?;
        let (_, data, total) = layout.offsets().unwrap();
        // SAFETY: validated region bounds.
        Ok(unsafe { Self::from_regions(ptr, ptr.add(data), total - data, None) })
    }

    /// Attaches process-local pointers to initialized separate shared regions.
    /// # Safety
    /// Regions remain valid through `owner` and follow the slot protocol.
    pub unsafe fn attach_regions(control: *mut u8, control_len: usize, data: *mut u8, data_len: usize, owner: Arc<dyn Send + Sync>) -> Result<Self> {
        let layout = Self::attached_layout(control, control_len, EXTERNAL)?;
        Self::check_regions(control, control_len, data, data_len, &layout)?;
        Ok(Self::from_regions(control, data, data_len, Some(owner)))
    }

    /// The initialized header. Its non-atomic fields never change after
    /// initialization, so a shared reference is enough.
    fn header(&self) -> &RingHeader {
        // SAFETY: retained initialized header; non-atomic fields stay immutable.
        unsafe { &*self.base.as_ptr().cast::<RingHeader>() }
    }
    /// Batch `batch`'s slot header: the batch number folded into the slot
    /// count, so slot `b` is reused by batch `b + slots`.
    fn slot(&self, batch: u64) -> &SlotHeader {
        let h = self.header();
        // SAFETY: ring construction rejects zero slots.
        unsafe { std::hint::assert_unchecked(h.slots != 0) };
        // SAFETY: validated header offsets; index is within slots.
        unsafe { &*self.base.as_ptr().add(h.slot_offset as usize + (batch % h.slots as u64) as usize * 64).cast::<SlotHeader>() }
    }
    /// The layout this ring was initialized or attached with.
    pub fn layout(&self) -> RingLayout {
        let h = self.header();
        RingLayout { slots: h.slots, slot_cells: h.slot_cells, world: h.world, cell_bytes: h.cell_bytes[..h.schemas as usize].to_vec() }
    }
    /// Positions per slot.
    pub fn slot_cells(&self) -> u64 {
        self.header().slot_cells as u64
    }
    /// Schemas, one region each per slot.
    pub fn schemas(&self) -> usize {
        self.header().schemas as usize
    }
    /// Bytes between consecutive slots.
    pub fn slot_stride(&self) -> usize {
        self.header().slot_stride as usize
    }
    /// Offset of schema `s`'s region within a slot.
    pub fn region_offset(&self, s: usize) -> usize {
        self.header().offsets[s] as usize
    }
    /// Bytes of one cell's output for schema `s`.
    pub fn cell_bytes(&self, s: usize) -> usize {
        self.header().cell_bytes[s] as usize
    }
    /// Positions in batch `batch`'s slot: `slot_cells`, less the tail of the
    /// run's last slot.
    fn rows_of(&self, batch: u64) -> u32 {
        let h = self.header();
        h.positions.load(Acquire).saturating_sub(batch.saturating_mul(h.slot_cells as u64)).min(h.slot_cells as u64) as u32
    }
    /// Batches of the current run: its positions in slots of `slot_cells`.
    fn batches(&self) -> u64 {
        self.header().positions.load(Acquire).div_ceil(self.slot_cells())
    }
    /// Opens a slot for `batch`: its position count and remaining output
    /// count, the run's generation, and DONE when the batch has no positions,
    /// then the sequence number that publishes it. Storing `seq` last is what
    /// makes the other fields visible to a consumer that observes the new
    /// batch.
    fn open(&self, batch: u64) {
        let s = self.slot(batch);
        let rows = self.rows_of(batch);
        s.rows.store(rows, Relaxed);
        s.remaining.store(rows * self.header().schemas, Relaxed);
        s.generation.store(self.generation(), Relaxed);
        s.state.store(if rows == 0 { DONE } else { FILLING }, Relaxed);
        s.seq.store(batch, Release);
        futex_wake(&s.state);
    }
    /// Test helper: starts a run of `positions` under the next generation.
    #[cfg(test)]
    pub(crate) fn begin(&self, positions: u64) -> Result<()> {
        let generation = self.generation().checked_add(1).ok_or_else(|| config("ring generation exhausted"))?;
        self.begin_generation(positions, generation)
    }
    /// The generation of the current or last run; 0 before the first run.
    pub fn generation(&self) -> u64 {
        self.header().generation.load(Acquire)
    }
    /// Opens every slot for a run of `positions` positions under
    /// `generation`, which must be newer than the ring's. Fails with
    /// [`Error::Busy`] when another producer holds the ring, and rejects a
    /// ring that was aborted or still holds a ready or leased slot.
    pub(crate) fn begin_generation(&self, positions: u64, generation: u64) -> Result<()> {
        if self.header().producer.compare_exchange(0, 1, AcqRel, Acquire).is_err() {
            return Err(Error::Busy);
        }
        if generation <= self.generation() || self.aborted() || (0..self.header().slots as u64).any(|b| matches!(self.slot(b).state.load(Acquire), READY | LEASED)) {
            self.header().producer.store(0, Release);
            return Err(config("ring aborted or still contains unconsumed batches; use a fresh ring"));
        }
        self.header().positions.store(positions, Release);
        self.header().generation.store(generation, Release);
        for b in 0..self.header().slots as u64 {
            self.open(b);
        }
        Ok(())
    }
    /// Releases the producer on success or failure, so a later run can take
    /// the ring.
    pub(crate) fn end(&self) {
        self.header().producer.store(0, Release);
    }
    /// Whether batch `batch`'s slot is open for writing.
    pub(crate) fn gate(&self, batch: u64) -> bool {
        let slot = self.slot(batch);
        slot.seq.load(Acquire) == batch && slot.generation.load(Acquire) == self.generation()
    }
    /// Blocks up to 200 µs until batch `batch`'s slot carries its number. A
    /// bounded wait because a wake may belong to another batch, so the caller
    /// re-checks instead of trusting one wake.
    pub(crate) fn wait_gate(&self, batch: u64) {
        let s = self.slot(batch);
        let state = s.state.load(Acquire);
        if s.seq.load(Acquire) != batch {
            futex_wait(&s.state, state, Duration::from_micros(200));
        }
    }
    /// Start of batch `batch`'s slot.
    #[inline]
    pub(crate) fn slot_base(&self, batch: u64) -> *mut u8 {
        let h = self.header();
        // SAFETY: the index is bounded by the validated layout.
        unsafe { self.data.as_ptr().add((batch % h.slots as u64) as usize * h.slot_stride as usize) }
    }
    /// Marks `n` outputs of batch `batch` written.
    #[inline]
    pub(crate) fn complete(&self, batch: u64, n: u32) {
        let s = self.slot(batch);
        if s.remaining.fetch_sub(n, AcqRel) == n {
            s.state.store(READY, Release);
            futex_wake(&s.state);
        }
    }
    /// Aborts the ring: every wait fails with [`Error::Aborted`] and no later
    /// run can start on this ring.
    pub fn abort(&self) {
        self.header().abort.store(1, Release);
        for b in 0..self.header().slots as u64 {
            futex_wake(&self.slot(b).state);
        }
    }
    /// Whether the ring was aborted; the flag is never cleared.
    pub fn aborted(&self) -> bool {
        self.header().abort.load(Acquire) != 0
    }

    /// The acquisition loop shared by every consumer entry point: spin, then
    /// futex-wait on the slot's state. `deadline` bounds the wait and
    /// `generation` rejects a batch of another run. The state word is both
    /// the futex word and the lease flag, so taking the lease is a compare and
    /// exchange of `READY` to `LEASED`; the sequence and generation are
    /// re-read after it to detect a slot reopened in between.
    fn wait_ready(&self, batch: u64, deadline: Option<Instant>, generation: Option<u64>) -> Result<Ready> {
        let s = self.slot(batch);
        let mut spins = 0;
        loop {
            if generation.is_some_and(|g| g != self.generation()) {
                return Err(config("ring generation differs from the requested run"));
            }
            if self.aborted() {
                return Err(Error::Aborted);
            }
            let seq = s.seq.load(Acquire);
            let observed_generation = s.generation.load(Acquire);
            let state = s.state.load(Acquire);
            if seq == batch {
                match state {
                    READY if s.state.compare_exchange(READY, LEASED, AcqRel, Acquire).is_ok() => {
                        if s.seq.load(Acquire) != batch || s.generation.load(Acquire) != observed_generation {
                            s.state.store(READY, Release);
                            futex_wake(&s.state);
                            continue;
                        }
                        return Ok(Ready::Batch(s.rows.load(Relaxed) as usize));
                    }
                    LEASED => return Err(config("batch already leased")),
                    DONE => return Ok(Ready::Done),
                    _ => (),
                }
            } else if seq != u64::MAX && (batch >= self.batches() || seq > batch) {
                if seq > batch && batch < self.batches() {
                    return Err(config("batch already consumed"));
                }
                return Ok(Ready::Done);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(Ready::Pending);
            }
            spins += 1;
            if spins < 4096 {
                std::hint::spin_loop();
            } else {
                futex_wait(&s.state, state, Duration::from_millis(1));
            }
        }
    }
    /// Acquires one batch; dropping the guard returns its slot. `None` when
    /// the batch is past the end of the run or holds no position.
    pub fn wait(&self, batch: u64) -> Result<Option<Batch<'_>>> {
        match self.wait_ready(batch, None, None)? {
            Ready::Batch(rows) => Ok(Some(Batch { ring: self, index: batch, rows, generation: self.generation() })),
            _ => Ok(None),
        }
    }
    /// Acquires a batch whose storage and slot remain valid through an Arc;
    /// `None` as for [`Self::wait`].
    pub fn acquire(self: &Arc<Self>, batch: u64) -> Result<Option<BatchLease>> {
        match self.acquire_timeout_inner(batch, None, None)? {
            Acquisition::Batch(batch) => Ok(Some(batch)),
            _ => Ok(None),
        }
    }
    /// Acquires a batch only from the specified run generation; `None` as for
    /// [`Self::wait`].
    pub fn acquire_generation(self: &Arc<Self>, batch: u64, generation: u64) -> Result<Option<BatchLease>> {
        match self.acquire_timeout_inner(batch, None, Some(generation))? {
            Acquisition::Batch(batch) => Ok(Some(batch)),
            _ => Ok(None),
        }
    }
    /// Bounded wait for runtimes that need to poll cancellation or signals.
    pub fn acquire_timeout(self: &Arc<Self>, batch: u64, timeout: Duration) -> Result<Acquisition> {
        self.acquire_timeout_inner(batch, Some(Instant::now() + timeout), None)
    }
    /// [`Self::wait_ready`] for an owned lease: the lease clones the ring's
    /// `Arc`, so storage outlives the caller's handle to it.
    fn acquire_timeout_inner(self: &Arc<Self>, batch: u64, deadline: Option<Instant>, generation: Option<u64>) -> Result<Acquisition> {
        Ok(match self.wait_ready(batch, deadline, generation)? {
            Ready::Batch(rows) => Acquisition::Batch(BatchLease { ring: self.clone(), index: batch, rows, generation: self.generation() }),
            Ready::Done => Acquisition::Done,
            Ready::Pending => Acquisition::Pending,
        })
    }
    /// Returns a leased slot, opening it for the batch one ring further on so
    /// that producers and consumers of a long run stay `slots` batches apart.
    /// The sequence and generation check keeps a stale lease from opening a
    /// slot that a later run already owns.
    fn release(&self, batch: u64, generation: u64) {
        let s = self.slot(batch);
        if s.seq.load(Acquire) == batch && s.generation.load(Acquire) == generation && s.state.load(Acquire) == LEASED {
            self.open(batch + self.header().slots as u64);
        }
    }
    /// The slot's storage: the stride from its start, cut to the data actually
    /// mapped (`data_min_bytes` need not reach the last slot's padding).
    ///
    /// Only the live lease's guard calls this: the slot must be leased and
    /// published, or the range is not the caller's to read.
    fn slot_of(&self, batch: u64) -> &[u8] {
        let start = (batch % self.header().slots as u64) as usize * self.slot_stride();
        // SAFETY: called only by the unique live batch guard after
        // publication; the slot lies inside the data storage.
        unsafe { std::slice::from_raw_parts(self.slot_base(batch), self.slot_stride().min(self.data_len - start)) }
    }
    /// Schema `s`'s compact outputs in a leased slot: `rows × cell_bytes(s)`.
    fn region_of(&self, batch: u64, rows: usize, s: usize) -> &[u8] {
        let slot = self.slot_of(batch);
        let at = self.region_offset(s);
        &slot[at..at + rows * self.cell_bytes(s)]
    }
}

/// A participant row: who takes part in a run and in what state.
#[repr(C, align(64))]
struct Participant {
    /// `EMPTY` while unused or being rewritten.
    state: AtomicU32,
    pid: AtomicU32,
    /// The process's start time (clock ticks since boot), so that a
    /// recycled pid is never taken for it.
    start: AtomicU64,
    /// The run the participant joined.
    generation: AtomicU64,
}

// Participant states. EMPTY is also the "being rewritten" marker, so a reader
// never combines a new pid with an old start time.
const EMPTY: u32 = 0;
const ACTIVE: u32 = 1;
const FINISHED: u32 = 2;
const LEFT: u32 = 3;

/// A participant row as read.
#[derive(Clone, Copy)]
struct Row {
    state: u32,
    pid: u32,
    start: u64,
    generation: u64,
}

/// The start time of process `pid`, or None once it has exited (zombies
/// included).
fn start_time(pid: u32) -> Option<u64> {
    let stat = std::fs::read(format!("/proc/{pid}/stat")).ok()?;
    // The command name (field 2) may contain spaces and parentheses.
    let rest = &stat[stat.iter().rposition(|&b| b == b')')? + 2..];
    let mut fields = rest.split(|&b| b == b' ');
    let state = fields.next()?;
    if state == b"Z" || state == b"X" {
        return None;
    }
    // Field 22 (starttime) is the 19th after the state.
    std::str::from_utf8(fields.nth(18)?).ok()?.trim().parse().ok()
}

/// This process's pid and start time, the pair that identifies a participant
/// (0 when `/proc` is unreadable, which then never matches a live row).
fn me() -> (u32, u64) {
    let pid = std::process::id();
    (pid, start_time(pid).unwrap_or(0))
}

/// Whether the row's process still runs with the start time the row recorded,
/// so that a recycled pid is not mistaken for the participant.
fn alive(row: &Row) -> bool {
    start_time(row.pid) == Some(row.start)
}

impl Ring {
    fn participant(&self, row: usize) -> &Participant {
        let h = self.header();
        // SAFETY: rows follow the slot headers inside the validated control
        // region: `world` consumer rows and the producer's.
        unsafe { &*self.base.as_ptr().add(h.slot_offset as usize + (h.slots as usize + row) * 64).cast::<Participant>() }
    }
    /// The producer's row, after every consumer rank's.
    fn producer_row(&self) -> usize {
        self.header().world as usize
    }
    /// A participant row read consistently: the state word is re-read after the
    /// fields and the whole read retried if it moved, so a row being rewritten
    /// is never observed half done.
    fn read_row(&self, row: usize) -> Row {
        let p = self.participant(row);
        loop {
            let state = p.state.load(Acquire);
            let r = Row { state, pid: p.pid.load(Relaxed), start: p.start.load(Relaxed), generation: p.generation.load(Relaxed) };
            if p.state.load(Acquire) == state {
                return r;
            }
        }
    }
    /// Claims a participant row for this process. The row is emptied first, so
    /// a concurrent reader retries instead of pairing a new pid with an old
    /// start time.
    fn write_row(&self, row: usize, state: u32, generation: u64) {
        let p = self.participant(row);
        let (pid, start) = me();
        // Readers skip the row while it is rewritten.
        p.state.store(EMPTY, Release);
        p.pid.store(pid, Relaxed);
        p.start.store(start, Relaxed);
        p.generation.store(generation, Relaxed);
        p.state.store(state, Release);
    }
    /// Whether this process registered the row (pid alone: a row is rewritten
    /// only by its own process).
    fn own(&self, row: usize) -> bool {
        self.participant(row).pid.load(Acquire) == std::process::id()
    }

    /// Registers this process as rank `rank`'s consumer of run `generation`
    /// (or of the next run). Fails for a rank outside `world` and when another
    /// live process consumes it.
    pub fn join_consumer(&self, rank: u32, generation: u64) -> Result<()> {
        if rank >= self.header().world {
            return Err(config("rank outside the ring's world"));
        }
        let r = self.read_row(rank as usize);
        if r.state == ACTIVE && r.pid != std::process::id() && alive(&r) {
            return Err(config(format!("rank {rank} already has a consumer (pid {})", r.pid)));
        }
        self.write_row(rank as usize, ACTIVE, generation);
        Ok(())
    }
    /// Records the generation of the run a consumer's first batch belongs to.
    pub fn adopt_generation(&self, rank: u32, generation: u64) {
        let p = self.participant(rank as usize);
        if self.own(rank as usize) && p.generation.load(Acquire) != generation {
            p.generation.store(generation, Release);
        }
    }
    /// This process leaves rank `rank`, having taken all of its batches or not.
    pub fn leave_consumer(&self, rank: u32, finished: bool) {
        if rank < self.header().world && self.own(rank as usize) {
            self.participant(rank as usize).state.store(if finished { FINISHED } else { LEFT }, Release);
        }
    }
    /// Registers this process as the producer of run `generation`; rows
    /// that ended an earlier run describe nobody in this one.
    pub fn join_producer(&self, generation: u64) {
        for rank in 0..self.header().world as usize {
            let r = self.read_row(rank);
            if matches!(r.state, FINISHED | LEFT) && r.generation < generation {
                self.participant(rank).state.store(EMPTY, Release);
            }
        }
        self.write_row(self.producer_row(), ACTIVE, generation);
    }
    /// This process stops producing, successfully or not.
    pub fn leave_producer(&self, ok: bool) {
        let row = self.producer_row();
        if self.own(row) {
            self.participant(row).state.store(if ok { FINISHED } else { LEFT }, Release);
        }
    }
    /// Why run `generation`'s producer can no longer finish it.
    pub fn producer_problem(&self, generation: u64) -> Option<String> {
        let r = self.read_row(self.producer_row());
        (r.generation == generation && r.state == ACTIVE && !alive(&r)).then(|| format!("the producer (pid {}) exited during the run", r.pid))
    }
    /// Why a consumer of run `generation` can no longer take its batches.
    pub fn consumer_problem(&self, generation: u64) -> Option<String> {
        for rank in 0..self.header().world as usize {
            let r = self.read_row(rank);
            // Consumers may join before the run starts (one generation back).
            if r.generation.saturating_add(1) < generation {
                continue;
            }
            if r.state == LEFT {
                return Some(format!("the consumer of rank {rank} (pid {}) left before taking all of its batches", r.pid));
            }
            if r.state == ACTIVE && !alive(&r) {
                return Some(format!("the consumer of rank {rank} (pid {}) exited without finishing", r.pid));
            }
        }
        None
    }
    /// Whether a live process other than this one consumes rank `rank` of
    /// run `generation`.
    pub fn consumed_elsewhere(&self, rank: u32, generation: u64) -> bool {
        let r = self.read_row(rank as usize);
        r.state == ACTIVE && r.pid != std::process::id() && r.generation.saturating_add(1) >= generation && alive(&r)
    }
    /// Batches whose slots the producer of a run of `batches` batches still
    /// needs but that their consumers have not given back: ready and not
    /// taken (`false`), or still leased (`true`).
    pub fn blocking(&self, batches: u64) -> Vec<(u64, bool)> {
        let slots = self.header().slots as u64;
        (0..slots)
            .filter_map(|b| {
                let s = self.slot(b);
                let seq = s.seq.load(Acquire);
                if seq == u64::MAX || seq.saturating_add(slots) >= batches || s.generation.load(Acquire) != self.generation() {
                    return None;
                }
                match s.state.load(Acquire) {
                    READY => Some((seq, false)),
                    LEASED => Some((seq, true)),
                    _ => None,
                }
            })
            .collect()
    }
}

macro_rules! lease_accessors {
    () => {
        /// The run generation the lease was taken under.
        pub fn generation(&self) -> u64 {
            self.generation
        }
        /// The batch number.
        pub fn index(&self) -> u64 {
            self.index
        }
        /// The first position in the slot.
        pub fn first_position(&self) -> u64 {
            self.index * self.ring.slot_cells()
        }
        /// Positions in the slot.
        pub fn rows(&self) -> usize {
            self.rows
        }
        /// The whole slot: every schema's region plus padding, cut to the
        /// data actually mapped.
        pub fn data(&self) -> &[u8] {
            self.ring.slot_of(self.index)
        }
        /// Schema `s`'s outputs: `rows × cell_bytes(s)` compact bytes.
        pub fn region(&self, s: usize) -> &[u8] {
            self.ring.region_of(self.index, self.rows, s)
        }
        /// Schemas in the slot, one region each.
        pub fn schemas(&self) -> usize {
            self.ring.schemas()
        }
        /// Bytes of one cell's output for schema `s`.
        pub fn cell_bytes(&self, s: usize) -> usize {
            self.ring.cell_bytes(s)
        }
        /// Offset of schema `s`'s region within the slot (for consumers that
        /// compute their own addresses).
        pub fn region_offset(&self, s: usize) -> usize {
            self.ring.region_offset(s)
        }
    };
}

/// Borrowed slot lease. Output references cannot outlive this guard.
pub struct Batch<'a> {
    ring: &'a Ring,
    index: u64,
    rows: usize,
    generation: u64,
}
impl Batch<'_> {
    lease_accessors!();
    /// Returns the slot now instead of at the end of the guard's scope.
    pub fn release(self) {}
}
impl Drop for Batch<'_> {
    fn drop(&mut self) {
        self.ring.release(self.index, self.generation);
    }
}

/// Owned slot lease, suitable for foreign-runtime array base objects.
pub struct BatchLease {
    ring: Arc<Ring>,
    index: u64,
    rows: usize,
    generation: u64,
}
impl BatchLease {
    lease_accessors!();
}
impl Drop for BatchLease {
    fn drop(&mut self) {
        self.ring.release(self.index, self.generation);
    }
}

/// One acquisition attempt's outcome.
enum Ready {
    /// The slot's position count, once its lease is taken.
    Batch(usize),
    /// The deadline passed before the slot was ready.
    Pending,
    /// The batch is past the end of the run, or has no positions.
    Done,
}

/// Outcome of a bounded owned-lease acquisition.
pub enum Acquisition {
    /// The batch is ready and leased to the caller.
    Batch(BatchLease),
    /// The timeout passed first; the batch may still become ready.
    Pending,
    /// The batch is past the end of the run, or has no positions.
    Done,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_are_compact_and_aligned() {
        let layout = RingLayout { slots: 2, slot_cells: 3, world: 1, cell_bytes: vec![6, 64, 1] };
        let (offsets, stride) = layout.regions().unwrap();
        assert_eq!(offsets, vec![0, 64, 256]);
        assert_eq!(stride, 4096);
    }

    #[test]
    fn owned_lease_holds_external_storage_and_returns_only_its_slot() {
        let layout = RingLayout { slots: 1, slot_cells: 1, world: 1, cell_bytes: vec![16, 4] };
        let bytes = layout.data_bytes().unwrap();
        let mut data = vec![0u64; bytes / 8 + 8];
        // SAFETY: the offset stays inside the vector.
        let dp = unsafe { data.as_mut_ptr().add(data.as_ptr().align_offset(REGION_ALIGN)) };
        let owner = Arc::new(data);
        let weak = Arc::downgrade(&owner);
        // SAFETY: the array is moved into owner, never accessed directly, and
        // retained by the ring until every owned lease has been released.
        let ring = Arc::new(unsafe { Ring::with_buffer(dp.cast(), bytes, layout, owner).unwrap() });
        ring.begin(2).unwrap();
        let write = |batch: u64, v: u8| {
            let base = ring.slot_base(batch);
            // SAFETY: the position exclusively owns its outputs in the open slot.
            unsafe {
                std::ptr::write_bytes(base.add(ring.region_offset(0)), v, 16);
                std::ptr::write_bytes(base.add(ring.region_offset(1)), v + 1, 4);
            }
            ring.complete(batch, 1);
            ring.complete(batch, 1);
        };
        write(0, 7);
        let lease = ring.acquire(0).unwrap().unwrap();
        assert_eq!(lease.region(0), &[7; 16]);
        assert_eq!(lease.region(1), &[8; 4]);
        assert!(ring.acquire(0).is_err());
        assert!(!ring.gate(1));
        drop(lease);
        assert!(ring.gate(1));
        write(1, 9);
        let lease = ring.acquire(1).unwrap().unwrap();
        ring.end();
        drop(ring);
        assert!(weak.upgrade().is_some());
        assert_eq!(lease.region(0), &[9; 16]);
        assert_eq!(lease.first_position(), 1);
        drop(lease);
        assert!(weak.upgrade().is_none());
    }
}
