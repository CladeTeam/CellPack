//! Reusable physical reads for selected records.
//!
//! A plan turns the record extents of a list of cells into file reads:
//! duplicate and empty extents are dropped, ranges separated by at most
//! `max_gap_bytes` are merged, and the plan collapses to one read of the whole
//! records region when the strategy asks for it or when the merged bytes plus
//! one request per range no longer beat that single request. Ranges are
//! relative to the records region, so one plan serves a chunk at any file
//! offset: the caller passes the base position to `read_into`.
//!
//! The plan also maps each request position back into the buffer the reads
//! filled, so decoding walks the requested cells in order with no offset
//! arithmetic per record.
use std::{fs::File, ops::Range, os::unix::fs::FileExt};
use crate::{Result, Error, format::Meta};
/// When to read the whole records region instead of the selected ranges.
///
/// `Ranges` reads the planned ranges, `WholeChunk` reads the whole records
/// region in one request (an empty region reads nothing), and `Auto` compares
/// the two by the estimate in [`ReadOptions`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReadStrategy { #[default] Auto, Ranges, WholeChunk }
/// Read planning: strategy, merge threshold and the estimated cost of a read.
///
/// `strategy` picks range reads or a whole-region read, `max_gap_bytes` merges
/// two ranges separated by at most that many bytes, and `request_cost_bytes`
/// is the cost of issuing one read (in bytes of I/O it is worth), which only
/// `Auto` uses.
#[derive(Clone, Copy, Debug)]
pub struct ReadOptions { pub strategy: ReadStrategy, pub max_gap_bytes: usize, pub request_cost_bytes: usize }
// Only adjacent ranges merge, a request is worth 4 KiB, and the strategy is chosen by cost.
impl Default for ReadOptions { fn default() -> Self { Self { strategy: ReadStrategy::Auto, max_gap_bytes: 0, request_cost_bytes: 4096 } } }
/// I/O a plan performs, counted in bytes and requests.
///
/// `record_bytes` is the distinct record payload the plan needs, `read_bytes`
/// the buffer the ranges fill (payload plus merged gaps), `requests` the number
/// of reads, and `whole_chunk_bytes` the records region's length — what
/// [`ReadStrategy::WholeChunk`] would read, and not the chunk with its metadata.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReadSummary { pub record_bytes: u64, pub read_bytes: u64, pub requests: u64, pub whole_chunk_bytes: u64 }
// Accumulates one chunk's summary into a selection's total.
impl ReadSummary { pub(crate) fn add(&mut self, b: Self) { self.record_bytes += b.record_bytes; self.read_bytes += b.read_bytes; self.requests += b.requests; self.whole_chunk_bytes += b.whole_chunk_bytes; } }
/// One read: `length` bytes at `offset` (from the records region start) land at
/// `buffer_offset` in the read buffer.
///
/// Ranges never overlap and their buffer offsets tile `0..read_bytes`.
#[derive(Clone, Debug)]
pub struct ReadRange { pub offset: u64, pub length: usize, pub buffer_offset: usize }
/// Reads for a list of cells and the buffer slice each cell's record occupies.
///
/// Built from metadata alone, so it is reusable for every read of the same
/// cells; the ranges are relative to the records region, not to file offsets.
#[derive(Debug)]
pub struct RecordReadPlan { ranges: Vec<ReadRange>, records: Vec<Range<usize>>, summary: ReadSummary }
impl RecordReadPlan {
 /// Plans reads for `cells` (indices within one chunk's metadata) under `o`.
 ///
 /// Each cell's record extent is collected, empty ones are dropped, the rest
 /// are sorted and deduplicated (two cells sharing a range read it once but
 /// slice it at their own positions), then merged across gaps of at most
 /// `max_gap_bytes`. `Auto` replaces the merged list with a single
 /// `0..records_len` range when the merged bytes plus one `request_cost_bytes`
 /// per request reach the region length plus one request, i.e. when the ranges
 /// no longer save I/O.
 pub fn new(meta: &Meta<'_>, cells: &[usize], o: ReadOptions) -> Result<Self> {
  let extents = cells.iter().map(|&c| meta.record_range(c)).collect::<Result<Vec<_>>>()?;
  // Empty records need no bytes, and a range two cells share is read once.
  let mut unique: Vec<_> = extents.iter().filter(|r| !r.is_empty()).cloned().collect();
  unique.sort_unstable_by_key(|r| (r.start,r.end)); unique.dedup();
  let record_bytes=unique.iter().map(|r|r.len() as u64).sum();
  // Merge across gaps of at most the threshold; `saturating_sub` keeps
  // adjacent ranges (gap 0) mergeable at the default threshold.
  let mut merged: Vec<Range<usize>>=Vec::new();
  for r in unique { if let Some(last)=merged.last_mut() && r.start.saturating_sub(last.end)<=o.max_gap_bytes {last.end=last.end.max(r.end);} else {merged.push(r);} }
  let bytes:u128=merged.iter().map(|r|r.len() as u128).sum();
  // One contiguous read costs one request and the region's bytes; the ranges
  // cost their own bytes plus one request each.
  if (o.strategy==ReadStrategy::WholeChunk || (o.strategy==ReadStrategy::Auto && bytes+merged.len() as u128*o.request_cost_bytes as u128>=meta.records_len() as u128+o.request_cost_bytes as u128)) && meta.records_len()>0 {
   merged.clear();
   merged.push(0..usize::try_from(meta.records_len()).map_err(|_|Error::InvalidInput("records exceed address space"))?);
  }
  let mut offset=0usize; let mut ranges=Vec::with_capacity(merged.len());
  for r in &merged { ranges.push(ReadRange{offset:r.start as u64,length:r.len(),buffer_offset:offset}); offset=offset.checked_add(r.len()).filter(|&n|n<=isize::MAX as usize).ok_or(Error::InvalidInput("read buffer exceeds address space"))?; }
  // A record's slice is its merged range's buffer position plus the record's
  // distance from that range's start; the last range starting at or before the
  // record is the one that holds it.
  let records=extents.into_iter().map(|r|{ if r.is_empty(){return 0..0;} let i=merged.partition_point(|m|m.start<=r.start)-1; let start=ranges[i].buffer_offset+r.start-merged[i].start; start..start+r.len() }).collect();
  let summary=ReadSummary{record_bytes,read_bytes:offset as u64,requests:ranges.len() as u64,whole_chunk_bytes:meta.records_len()};
  Ok(Self{ranges,records,summary})
 }
 /// The reads to issue, in buffer order.
 pub fn ranges(&self)->&[ReadRange]{&self.ranges}
 /// What this plan reads, for cost accounting and tests.
 pub fn summary(&self)->ReadSummary{self.summary}
 /// The `position`-th requested cell's record, sliced from a buffer
 /// [`RecordReadPlan::read_into`] filled; empty records give an empty slice.
 /// A position past the plan, or a buffer that does not cover the record's
 /// slice, is an error.
 pub fn record<'a>(&self, bytes:&'a [u8], position:usize)->Result<&'a [u8]>{let r=self.records.get(position).ok_or(Error::InvalidInput("record position outside plan"))?; bytes.get(r.clone()).ok_or(Error::InvalidInput("read buffer too short"))}
 /// Reads every range into `out`, whose byte `n` is the plan's buffer byte
 /// `n`, at file position `base + range.offset`.
 ///
 /// `out` is resized to the plan's `read_bytes`; since the ranges tile that
 /// buffer, no stale byte survives into a record.
 pub fn read_into(&self,file:&File,base:u64,out:&mut Vec<u8>)->Result<()>{out.resize(self.summary.read_bytes as usize,0);for r in &self.ranges {let offset=base.checked_add(r.offset).ok_or(Error::InvalidInput("file offset overflow"))?;file.read_exact_at(&mut out[r.buffer_offset..r.buffer_offset+r.length],offset)?;}Ok(())}
}
