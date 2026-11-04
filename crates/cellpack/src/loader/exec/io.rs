//! A worker's reads: io_uring requests queued, submitted in batches and
//! harvested per completion, or blocking preadv.
//!
//! Every read takes a slot, which keeps its iovec array stable until the
//! completion, the task it belongs to, whether that task needs it before
//! decoding, and the cache frames it admits (with their generations, which
//! are published when the completion is handled). Slots grow on demand: an
//! admission read is never refused for lack of one. The catalog's files are
//! registered with the ring when possible.

use std::io;
use std::os::fd::RawFd;

use io_uring::{IoUring, opcode, types};

use crate::loader::error::{Result, config};

/// How a worker reads: the platform's asynchronous ring, or blocking preadv.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoBackend {
    /// Blocking `preadv`; a read completes inside `WorkerIo::read`.
    Pread,
    /// io_uring: reads are queued, submitted in batches and harvested from the
    /// completion queue. Creating the ring fails on a kernel without io_uring
    /// support.
    Uring,
}

/// Read statistics of one worker, summed over workers in `RunStats`.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoStats {
    /// Reads issued, including extra admission reads.
    pub requests: u64,
    /// Bytes asked for (pages × page size per request).
    pub requested_bytes: u64,
    /// Bytes returned; a short read ends the run.
    pub read_bytes: u64,
    /// Reads by size in pages: 1, 2, 3-4, 5-8, 9-16, 17-32, 33+.
    pub sizes: [u64; 7],
}

impl IoStats {
    /// Adds another worker's counts to these.
    pub(crate) fn add(&mut self, o: &IoStats) {
        self.requests += o.requests;
        self.requested_bytes += o.requested_bytes;
        self.read_bytes += o.read_bytes;
        for (a, b) in self.sizes.iter_mut().zip(o.sizes) {
            *a += b;
        }
    }
}

/// One read in flight: the iovecs handed to the kernel, the task it reports
/// to, and the frames it publishes on completion.
pub(crate) struct Slot {
    /// One iovec per page read; kept here so its storage is stable until the
    /// completion arrives.
    iov: Vec<libc::iovec>,
    /// Index of the claiming task, the completion's `user_data`.
    pub task: u32,
    /// The task decodes from this read.
    pub own: bool,
    /// Bytes the read must return.
    pub need: u32,
    /// Cache frames the read fills, with the generation each publishes.
    pub publish: Vec<(u32, u32)>,
    /// Profiling stamp of the submission (0 when not timed).
    pub issued: u64,
}

/// A completed read: its slot (valid until [`WorkerIo::recycle`]) and result.
#[derive(Clone, Copy)]
pub(crate) struct Done {
    /// The slot whose iovecs the read filled.
    pub slot: u32,
    /// Bytes read, or the negated errno.
    pub result: i32,
}

/// A worker's reads: the ring (or none, for preadv), the reusable slots and
/// the counts.
///
/// Slots grow on demand and are recycled after their completion is handled, so
/// an admission read is never refused for lack of one; `free` holds the
/// recycled ones.
pub(crate) struct WorkerIo {
    ring: Option<IoUring>,
    fixed_files: bool,
    slots: Vec<Slot>,
    free: Vec<u32>,
    /// Reads pushed to the submission queue and not yet submitted.
    queued: u32,
    /// The previous flush left its queued reads unsubmitted, so the next one
    /// submits.
    held: bool,
    /// Reads submitted and not yet harvested.
    pub inflight: u32,
    /// This worker's counters, copied into its [`crate::loader::RunStats`].
    pub stats: IoStats,
}

impl WorkerIo {
    /// `entries` sizes the submission queue, raised to a power of two and
    /// clamped to `2..=32768`; `files` are the catalog's descriptors by file
    /// id, registered with the ring when the kernel permits.
    pub fn new(backend: IoBackend, entries: usize, files: &[RawFd]) -> Result<Self> {
        let mut fixed_files = false;
        let ring = if backend == IoBackend::Uring {
            let entries = u32::try_from(entries.max(2)).ok().and_then(|d| d.checked_next_power_of_two()).ok_or_else(|| config("uring depth too large"))?;
            let ring = IoUring::new(entries.min(32768))?;
            fixed_files = !files.is_empty() && ring.submitter().register_files(files).is_ok();
            Some(ring)
        } else {
            None
        };
        Ok(Self { ring, fixed_files, slots: Vec::new(), free: Vec::new(), queued: 0, held: false, inflight: 0, stats: IoStats::default() })
    }

    /// The slot of a harvested completion, until [`Self::recycle`].
    pub fn slot(&mut self, i: u32) -> &mut Slot {
        &mut self.slots[i as usize]
    }

    /// Returns a harvested read's slot.
    pub fn recycle(&mut self, i: u32) {
        let s = &mut self.slots[i as usize];
        s.publish.clear();
        self.free.push(i);
    }

    /// Whether completions wait to be harvested: one load of the ring's tail.
    #[inline(always)]
    pub fn ready(&mut self) -> bool {
        self.ring.as_mut().is_some_and(|r| !r.completion().is_empty())
    }

    /// Reads `frames.len()` consecutive pages of file `file` (descriptor
    /// `fd`) at `offset` into `frames`. Returns the slot; pread completes at
    /// once and reports the byte count with it.
    #[allow(clippy::too_many_arguments)]
    pub fn read(&mut self, fd: RawFd, file: u32, offset: u64, frames: &[*mut u8], page: usize, need: usize, task: u32, own: bool, publish: &[(u32, u32)], issued: u64) -> Result<(u32, Option<i32>)> {
        let i = match self.free.pop() {
            Some(i) => i,
            None => {
                self.slots.push(Slot { iov: Vec::new(), task: 0, own: false, need: 0, publish: Vec::new(), issued: 0 });
                (self.slots.len() - 1) as u32
            }
        };
        let s = &mut self.slots[i as usize];
        s.iov.clear();
        s.iov.extend(frames.iter().map(|&f| libc::iovec { iov_base: f.cast(), iov_len: page }));
        s.task = task;
        s.own = own;
        s.need = need as u32;
        s.publish.clear();
        s.publish.extend_from_slice(publish);
        s.issued = issued;
        self.stats.requests += 1;
        self.stats.requested_bytes += (frames.len() * page) as u64;
        let bucket = (usize::BITS - (frames.len().max(1) - 1).leading_zeros()) as usize;
        self.stats.sizes[bucket.min(6)] += 1;
        let Some(ring) = &mut self.ring else {
            let n = loop {
                // SAFETY: the iovecs describe live frames owned by the caller.
                let n = unsafe { libc::preadv(fd, s.iov.as_ptr(), s.iov.len() as i32, offset as i64) };
                if n < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                break n;
            };
            if n < 0 {
                return Err(io::Error::last_os_error().into());
            }
            self.stats.read_bytes += n as u64;
            return Ok((i, Some(n as i32)));
        };
        // One page reads into its frame directly: no iovec to walk.
        let entry = match (self.fixed_files, frames.len()) {
            (true, 1) => opcode::Read::new(types::Fixed(file), frames[0], page as u32).offset(offset).build(),
            (false, 1) => opcode::Read::new(types::Fd(fd), frames[0], page as u32).offset(offset).build(),
            (true, _) => opcode::Readv::new(types::Fixed(file), s.iov.as_ptr(), s.iov.len() as u32).offset(offset).build(),
            (false, _) => opcode::Readv::new(types::Fd(fd), s.iov.as_ptr(), s.iov.len() as u32).offset(offset).build(),
        }
        .user_data(i as u64);
        // SAFETY: the iovec array and its frames stay untouched until the CQE.
        if unsafe { ring.submission().push(&entry) }.is_err() {
            ring.submit()?;
            self.queued = 0;
            // SAFETY: as above.
            unsafe { ring.submission().push(&entry) }.map_err(|_| config("uring submission queue remains full"))?;
        }
        self.queued += 1;
        self.inflight += 1;
        Ok((i, None))
    }

    /// Submits queued reads once enough have accumulated to amortize the
    /// system call, at once if the device would otherwise run short, and in
    /// any case on the next call.
    pub fn flush(&mut self, batch: u32) -> Result<()> {
        if let Some(ring) = &self.ring
            && self.queued > 0
        {
            if self.queued >= batch || self.inflight - self.queued < batch || self.held {
                ring.submit()?;
                self.queued = 0;
                self.held = false;
            } else {
                self.held = true;
            }
        }
        Ok(())
    }

    /// Collects available completions; never blocks.
    pub fn harvest(&mut self, out: &mut Vec<Done>) {
        let Some(ring) = &mut self.ring else {
            return;
        };
        for cqe in ring.completion() {
            let slot = cqe.user_data() as u32;
            if cqe.result() > 0 {
                self.stats.read_bytes += cqe.result() as u64;
            }
            out.push(Done { slot, result: cqe.result() });
            self.inflight -= 1;
        }
    }

    /// Blocks until at least one in-flight read completes.
    pub fn wait(&mut self) -> Result<()> {
        if let Some(ring) = &self.ring
            && self.inflight > 0
        {
            loop {
                match ring.submit_and_wait(1) {
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    r => {
                        r?;
                        break;
                    }
                }
            }
            self.queued = 0;
        }
        Ok(())
    }

    /// Waits for every in-flight read, so its frames may be reused. Returns
    /// false if the ring failed; those frames must then never be reused.
    pub fn drain(&mut self) -> bool {
        let mut sink = Vec::new();
        if self.flush(0).is_err() {
            return false;
        }
        while self.inflight > 0 {
            if self.wait().is_err() {
                return false;
            }
            sink.clear();
            self.harvest(&mut sink);
        }
        true
    }
}
