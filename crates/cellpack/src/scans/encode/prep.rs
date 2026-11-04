//! Per-row preparation: the symbols and escape bits of a row's record
//! candidates, which do not depend on the chunk's frequency tables. A chunk
//! prepares each row once, counts the symbols for its models and codes the
//! kept symbols afterwards; a single record prepares and codes in one go.
//!
//! - **Coded order** ([`order`]): with a gene order, the cell's coded ids
//!   are marked in a bitmap and read back ascending (short cells sort
//!   instead), each with its input position.
//! - **Gaps**: symbols and escape bits of the ascending coded ids.
//! - **Values** ([`prepare_values`]): one pass gives every value its
//!   dictionary id and, against column modes, its exception id; sorting the
//!   distinct values gives the dictionary symbols and every id's rank.
//!
//! Everything lands in an [`Arena`] as (symbols, escape bits) pairs; the
//! `*_bound` functions bound what a row can write.

use super::input::{Index, load};
use crate::scans::kernels::Kernels;
use crate::scans::symbol;
use crate::unchecked::{get, set};

/// A region of an [`Arena`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Span {
    at: usize,
    len: usize,
}

impl Span {
    /// The span's bytes in the contents of the arena it came from.
    #[inline]
    pub(crate) fn of(self, bytes: &[u8]) -> &[u8] {
        &bytes[self.at..self.at + self.len]
    }
}

/// Symbols and their escape bits.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Pair {
    /// One symbol per element (or per dictionary entry).
    pub(crate) syms: Span,
    /// The escape bits of those symbols, in order.
    pub(crate) bits: Span,
}

/// Bump allocation over a byte buffer.
///
/// Each [`Arena::pair`] takes `n` symbol bytes and the escape bytes they
/// bound, so the contents are a sequence of (symbols, bits) regions that
/// [`Span`]s address.
pub(crate) struct Arena<'a> {
    buf: &'a mut [u8],
    used: usize,
}

impl<'a> Arena<'a> {
    /// An empty arena over `buf`.
    pub(crate) fn new(buf: &'a mut [u8]) -> Self {
        Arena { buf, used: 0 }
    }

    /// Drops every region, keeping the buffer.
    pub(crate) fn clear(&mut self) {
        self.used = 0;
    }

    /// The regions written since the last [`Arena::clear`].
    pub(crate) fn contents(&self) -> &[u8] {
        &self.buf[..self.used]
    }

    /// `n` symbols and at most `bound` escape bytes written by `f`.
    fn pair(&mut self, n: usize, bound: usize, f: impl FnOnce(&mut [u8], &mut Bits)) -> Pair {
        let (syms, rest) = self.buf[self.used..].split_at_mut(n);
        let mut bits = Bits::new(&mut rest[..bound]);
        f(syms, &mut bits);
        let len = bits.finish();
        let pair = Pair {
            syms: Span { at: self.used, len: n },
            bits: Span { at: self.used + n, len },
        };
        self.used += n + len;
        pair
    }
}

/// LSB-first escape bits, the layout of [`crate::scans::bits::BitWriter`],
/// stored four bytes at a time.
pub(crate) struct Bits<'a> {
    buf: &'a mut [u8],
    len: usize,
    acc: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Bits { buf, len: 0, acc: 0, n: 0 }
    }

    /// Appends the low `nb` bits of `v` (`nb <= 32`, `v < 2^nb`).
    #[inline(always)]
    fn put32(&mut self, v: u64, nb: u32) {
        self.acc |= v << self.n;
        self.n += nb;
        if self.n >= 32 {
            self.buf[self.len..self.len + 4].copy_from_slice(&(self.acc as u32).to_le_bytes());
            self.len += 4;
            self.acc >>= 32;
            self.n -= 32;
        }
    }

    /// Appends the low `nb` bits of `v` (`nb <= 64`, `v < 2^nb`).
    #[inline(always)]
    fn put(&mut self, v: u64, nb: u32) {
        if nb > 32 {
            self.put32(v & 0xffff_ffff, 32);
            self.put32(v >> 32, nb - 32);
        } else {
            self.put32(v, nb);
        }
    }

    /// Flushes the partial bytes; returns the stream length.
    fn finish(self) -> usize {
        let tail = self.n.div_ceil(8) as usize;
        self.buf[self.len..self.len + tail].copy_from_slice(&self.acc.to_le_bytes()[..tail]);
        self.len + tail
    }
}

/// Symbol `t` and the escape bits of an integer below `2^32`.
///
/// # Safety
/// `t < syms.len()`.
#[inline(always)]
unsafe fn emit32(v: u32, syms: &mut [u8], t: usize, bits: &mut Bits) {
    let (s, raw, nb) = symbol::split(v as u64);
    // SAFETY: as documented.
    unsafe { set(syms, t, s) };
    bits.put32(raw, nb);
}

// ---- bounds ----

/// Escape bytes of `count` integers whose values plus one sum to at most
/// `total`. An integer `g` has at most `log2(g + 1) + 1` escape bits (none
/// below the direct range), a concave bound, so the sum is at most
/// `count * (log2(total / count) + 1)` bits. The ratio is floored at 1.5,
/// which keeps the bound increasing in `count`.
fn escape_bound(count: usize, total: f64) -> usize {
    if count == 0 {
        return 0;
    }
    let c = count as f64;
    ((c * ((total / c).max(1.5).log2() + 1.0)) / 8.0) as usize + 8
}

/// Escape bytes of `n` ranks or mode ranks, each at most `n`.
fn rank_escape_bound(n: usize) -> usize {
    if n < symbol::DIRECT as usize {
        0
    } else {
        n * (usize::BITS - n.leading_zeros()) as usize / 8 + 8
    }
}

/// Bytes of one value coding of `n` `vw`-byte values: dictionary symbols
/// and escape bits (gaps between distinct values, which sum below
/// `2^(8 vw)`), and rank symbols and escape bits.
fn coding_bound(n: usize, vw: usize) -> usize {
    let span = 2f64.powi(8 * vw as i32);
    let k = (n as f64).min(span) as usize;
    2 * n + escape_bound(k, span) + rank_escape_bound(n)
}

/// Bytes [`prepare_values`] writes for `n` values, with or without the mode coding.
pub(crate) fn values_bound(n: usize, vw: usize, modes: bool) -> usize {
    coding_bound(n, vw) * (1 + modes as usize)
}

/// Bytes [`prepare_sparse`] writes for a cell of `n` elements.
pub(crate) fn sparse_bound(n: usize, n_genes: u32, vw: usize, modes: bool) -> usize {
    if n == 0 {
        return 0;
    }
    n + escape_bound(n, n_genes as f64) + values_bound(n, vw, modes)
}

// ---- coded order ----

/// Scratch for the coded order of a cell.
pub(crate) struct OrderWs<'w> {
    /// One bit per gene, clear between cells.
    pub(crate) bitmap: &'w mut [u64],
    /// Input position of each marked coded id; one entry per gene.
    pub(crate) pos: &'w mut [u32],
    /// The cell's coded ids, ascending, with scan slack.
    pub(crate) coded: &'w mut [u32],
    /// Input position of each element in coded order.
    pub(crate) perm: &'w mut [u32],
    /// Sort keys of short cells.
    pub(crate) keys: &'w mut [u64],
    /// The bound kernel set, whose `scan_bits` expands the bitmap.
    pub(crate) kernels: &'static Kernels,
}

/// Cells up to this long sort their coded ids instead of scanning the bitmap.
const SORT_MAX: usize = 32;

/// The coded ids of a validated cell into `ws.coded`, ascending, and with
/// a gene order (`cos`) each one's input position into `ws.perm`.
pub(crate) fn order<I: Index>(idx: &[I], cos: Option<&[u32]>, ws: &mut OrderWs) {
    let n = idx.len();
    debug_assert!(ws.coded.len() >= n && ws.perm.len() >= n);
    let Some(cos) = cos else {
        for (c, &i) in ws.coded.iter_mut().zip(idx) {
            *c = i.id();
        }
        return;
    };
    // SAFETY: indices of a validated cell are distinct gene ids, and the
    // gene order is a permutation of them, so coded ids are distinct and
    // below `n_genes`: within `pos` (one entry per gene) and the bitmap
    // (one bit per gene); `t`, `j < n`, within the element buffers, and
    // `coded` holds the scan slack; the kernel set's ISA is supported.
    unsafe {
        if n <= SORT_MAX {
            for (t, &i) in idx.iter().enumerate() {
                set(ws.keys, t, (get(cos, i.id() as usize) as u64) << 32 | t as u64);
            }
            let keys = ws.keys.get_unchecked_mut(..n);
            keys.sort_unstable();
            for (j, &key) in keys.iter().enumerate() {
                set(ws.coded, j, (key >> 32) as u32);
                set(ws.perm, j, key as u32);
            }
            return;
        }
        let mut top = 0;
        for (t, &i) in idx.iter().enumerate() {
            let c = get(cos, i.id() as usize) as usize;
            *ws.bitmap.get_unchecked_mut(c >> 6) |= 1 << (c & 63);
            set(ws.pos, c, t as u32);
            top = top.max(c);
        }
        let words = ws.bitmap.get_unchecked_mut(..=top >> 6);
        let j = (ws.kernels.scan_bits)(words, ws.pos, ws.coded, ws.perm);
        debug_assert_eq!(j, n);
    }
}

// ---- values ----

/// Hash table size for `n` values: a power of two at least twice `n`.
pub(crate) fn hash_slots(n: usize) -> usize {
    (2 * n).next_power_of_two().max(16)
}

/// Smallest table a dictionary starts with.
const TABLE_MIN: usize = 256;

/// Entries of the direct table for small values: integers below
/// [`SMALL_INT`], then floats holding such an integer.
pub(crate) const SMALL: usize = 2 * SMALL_INT;
const SMALL_INT: usize = 2048;

/// The direct-table entry of a `W`-byte value: an integer below
/// [`SMALL_INT`], or (for 4 and 8 bytes) the bits of an `f32` / `f64`
/// holding such an integer, which counts stored as floats are. A value's
/// bits alone decide its entry, so no value has two.
#[inline(always)]
fn small_entry<const W: usize>(v: u64) -> Option<usize> {
    if v < SMALL_INT as u64 {
        return Some(v as usize);
    }
    let i = match W {
        4 => {
            let i = f32::from_bits(v as u32) as u32;
            ((i as f32).to_bits() as u64 == v).then_some(i)
        }
        8 => {
            let i = f64::from_bits(v) as u32;
            ((i as f64).to_bits() == v).then_some(i)
        }
        _ => None,
    }?;
    ((i as usize) < SMALL_INT).then_some(SMALL_INT + i as usize)
}

/// Distinct values of a sequence with ids in first-seen order. Small
/// values ([`small_entry`]) index a direct table; the others go to an open
/// addressing table that starts small and doubles, staying at most half
/// full, so the few values of most cells stay in a few cache lines.
pub(crate) struct Distinct<'w> {
    keys: &'w mut [u64],
    ids: &'w mut [u32],
    /// Id plus one of each small value; zero between cells.
    small: &'w mut [u32],
    /// Distinct values by id.
    values: &'w mut [u64],
    k: usize,
}

impl<'w> Distinct<'w> {
    /// `small` must be zero.
    pub(crate) fn new(keys: &'w mut [u64], ids: &'w mut [u32], small: &'w mut [u32], values: &'w mut [u64]) -> Self {
        assert!(small.len() >= SMALL);
        Distinct { keys, ids, small, values, k: 0 }
    }

    /// An empty dictionary for a sequence of `n` values, as a cursor that
    /// lives in registers while it inserts; [`Distinct::close`] takes back
    /// its count and clears the direct table.
    fn open<const W: usize>(&mut self, n: usize) -> Cursor {
        let limit = hash_slots(n);
        assert!(limit <= self.keys.len() && limit <= self.ids.len() && n <= self.values.len());
        let c = Cursor {
            keys: self.keys.as_mut_ptr(),
            ids: self.ids.as_mut_ptr(),
            small: self.small.as_mut_ptr(),
            values: self.values.as_mut_ptr(),
            k: 0,
            hashed: 0,
            left: 0,
            mask: 0,
            shift: 0,
            limit,
        };
        // SAFETY: the tables hold `limit` slots (checked above).
        unsafe { c.resize::<W>(limit.min(TABLE_MIN)) }
    }

    fn close<const W: usize>(&mut self, c: Cursor) {
        self.k = c.k;
        for &v in &self.values[..c.k] {
            if let Some(e) = small_entry::<W>(v) {
                self.small[e] = 0;
            }
        }
    }
}

/// An open [`Distinct`]: its tables' addresses and state.
#[derive(Clone, Copy)]
struct Cursor {
    keys: *mut u64,
    ids: *mut u32,
    small: *mut u32,
    values: *mut u64,
    k: usize,
    /// Values in the hash table, inserts left before it must grow, and its
    /// largest size.
    hashed: usize,
    left: usize,
    limit: usize,
    mask: usize,
    shift: u32,
}

impl Cursor {
    #[inline(always)]
    fn slot(&self, v: u64) -> usize {
        (v.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> self.shift) as usize
    }

    /// The cursor with an empty hash table of `cap` slots holding its
    /// values that are not small.
    ///
    /// # Safety
    /// `cap` is a power of two at most `limit`, above twice the hashed count.
    #[cold]
    #[inline(never)]
    unsafe fn resize<const W: usize>(mut self, cap: usize) -> Self {
        // SAFETY: as documented, the slots exist; ids below `k` have values.
        unsafe {
            std::slice::from_raw_parts_mut(self.ids, cap).fill(u32::MAX);
            self.mask = cap - 1;
            self.shift = 64 - cap.trailing_zeros();
            self.left = cap / 2 - self.hashed;
            for id in 0..self.k {
                let v = *self.values.add(id);
                if small_entry::<W>(v).is_some() {
                    continue;
                }
                let mut h = self.slot(v);
                while *self.ids.add(h) != u32::MAX {
                    h = (h + 1) & self.mask;
                }
                *self.ids.add(h) = id as u32;
                *self.keys.add(h) = v;
            }
        }
        self
    }

    /// The id of `v`, inserted if new.
    ///
    /// # Safety
    /// At most `n` values are inserted into a dictionary opened for `n`.
    #[inline(always)]
    unsafe fn insert<const W: usize>(&mut self, v: u64) -> u32 {
        // SAFETY: direct entries are below `SMALL`, which the direct table
        // holds; hash slots are below the table size, at most `limit`, which
        // the tables hold; new ids stay below `n`, within `values`.
        unsafe {
            if let Some(e) = small_entry::<W>(v) {
                let held = *self.small.add(e);
                if held != 0 {
                    return held - 1;
                }
                let id = self.k;
                *self.small.add(e) = id as u32 + 1;
                *self.values.add(id) = v;
                self.k += 1;
                return id as u32;
            }
            let mut h = self.slot(v);
            loop {
                let id = *self.ids.add(h);
                if id == u32::MAX {
                    let id = self.k;
                    *self.ids.add(h) = id as u32;
                    *self.keys.add(h) = v;
                    *self.values.add(id) = v;
                    self.k += 1;
                    self.hashed += 1;
                    self.left -= 1;
                    if self.left == 0 && self.mask + 1 < self.limit {
                        *self = self.resize::<W>(2 * (self.mask + 1));
                    }
                    return id as u32;
                }
                if *self.keys.add(h) == v {
                    return id;
                }
                h = (h + 1) & self.mask;
            }
        }
    }
}

/// Scratch for value preparation.
pub(crate) struct ValuesWs<'w> {
    /// Gaps with escape bits.
    pub(crate) esc: &'w mut [u32],
    /// Dictionary id of each element.
    pub(crate) ids: &'w mut [u32],
    /// Exception id plus one of each element; 0 for its gene's mode.
    pub(crate) mids: &'w mut [u32],
    /// Distinct values with ids in first-seen order.
    pub(crate) dict: Distinct<'w>,
    /// Distinct exceptions by exception id.
    pub(crate) exc: &'w mut [u64],
    /// Sort keys of a dictionary, shared by every coding of a row.
    pub(crate) sort: &'w mut [u64],
    /// Rank of each id, offset like the ids.
    pub(crate) rank_of: &'w mut [u32],
}

/// The elements of a record in record order.
pub(crate) trait Elements {
    /// Elements of the record.
    fn len(&self) -> usize;
    /// Value of element `t`.
    ///
    /// # Safety
    /// `t < self.len()`.
    unsafe fn value(&self, t: usize) -> u64;
    /// Coded gene of element `t`.
    ///
    /// # Safety
    /// `t < self.len()`.
    unsafe fn gene(&self, t: usize) -> usize;
}

/// Values in record order, element `t` at coded gene `genes[t]`.
pub(crate) struct InOrder<'a, const W: usize> {
    /// Values of the row, element `t` at `t`.
    pub(crate) values: &'a [u8],
    /// Coded gene of element `t`.
    pub(crate) genes: &'a [u32],
}

/// Values reordered by `perm`, element `t` at coded gene `genes[t]`.
pub(crate) struct Gathered<'a, const W: usize> {
    /// Values of the row, element `t` at `perm[t]`.
    pub(crate) values: &'a [u8],
    /// Input position of element `t`.
    pub(crate) perm: &'a [u32],
    /// Coded gene of element `t`.
    pub(crate) genes: &'a [u32],
}

/// Every gene's value in coded order.
pub(crate) struct Dense<'a, const W: usize> {
    /// Values of the row, element `t` at gene `t`.
    pub(crate) values: &'a [u8],
}

impl<const W: usize> Elements for InOrder<'_, W> {
    fn len(&self) -> usize {
        self.genes.len()
    }
    #[inline(always)]
    unsafe fn value(&self, t: usize) -> u64 {
        // SAFETY: one value per element.
        unsafe { load::<W>(self.values, t) }
    }
    #[inline(always)]
    unsafe fn gene(&self, t: usize) -> usize {
        // SAFETY: `t` is an element.
        unsafe { get(self.genes, t) as usize }
    }
}

impl<const W: usize> Elements for Gathered<'_, W> {
    fn len(&self) -> usize {
        self.genes.len()
    }
    #[inline(always)]
    unsafe fn value(&self, t: usize) -> u64 {
        // SAFETY: `perm` is a permutation of the elements.
        unsafe { load::<W>(self.values, get(self.perm, t) as usize) }
    }
    #[inline(always)]
    unsafe fn gene(&self, t: usize) -> usize {
        // SAFETY: `t` is an element.
        unsafe { get(self.genes, t) as usize }
    }
}

impl<const W: usize> Elements for Dense<'_, W> {
    fn len(&self) -> usize {
        self.values.len() / W
    }
    #[inline(always)]
    unsafe fn value(&self, t: usize) -> u64 {
        // SAFETY: `t` is an element.
        unsafe { load::<W>(self.values, t) }
    }
    #[inline(always)]
    unsafe fn gene(&self, t: usize) -> usize {
        t
    }
}

/// A prepared value coding: the dictionary's gap symbols and, when ranked,
/// every element's rank symbol.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CodingPrep {
    /// Distinct values (exceptions, for a mode coding).
    pub(crate) k: usize,
    /// Gap symbols of the distinct values, and their escape bits.
    pub(crate) dict: Pair,
    /// One rank symbol per element; `None` when every element is entry 0.
    pub(crate) ranks: Option<Pair>,
}

/// A prepared mode coding and its exception count.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ModePrep {
    /// The exceptions' dictionary and ranks.
    pub(crate) coding: CodingPrep,
    /// Elements that differ from their column mode.
    pub(crate) exceptions: usize,
}

/// The value codings of a record's elements.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ValuesPrep {
    /// Elements of the record.
    pub(crate) n: usize,
    /// The per-row dictionary coding.
    pub(crate) dict: CodingPrep,
    /// Against column modes, when the row is prepared with them.
    pub(crate) mode: Option<ModePrep>,
}

/// The `k` distinct values (by id) ascending, in place, with
/// `rank_of[base + id] = base + rank` (and `rank_of[0] = 0` for base 1).
fn rank_distinct<const W: usize>(values: &mut [u64], sort: &mut [u64], rank_of: &mut [u32], base: usize) {
    let k = values.len();
    let sort = &mut sort[..k];
    let rank_of = &mut rank_of[..k + base];
    if base == 1 {
        rank_of[0] = 0;
    }
    if W <= 4 {
        // Values below 2^32 sort with their ids as one key.
        for (id, (key, &v)) in sort.iter_mut().zip(values.iter()).enumerate() {
            *key = v << 32 | id as u64;
        }
        sort.sort_unstable();
        for (r, &key) in sort.iter().enumerate() {
            rank_of[base + (key as u32) as usize] = (base + r) as u32;
            values[r] = key >> 32;
        }
    } else {
        for (id, key) in sort.iter_mut().enumerate() {
            *key = id as u64;
        }
        sort.sort_unstable_by_key(|&id| values[id as usize]);
        for (r, key) in sort.iter_mut().enumerate() {
            let id = *key as usize;
            rank_of[base + id] = (base + r) as u32;
            *key = values[id];
        }
        values.copy_from_slice(sort);
    }
}

/// Codes a ranked dictionary: its gap symbols and, when `ranked`, the rank
/// symbol of every element's id (`ids`, offset by `base`).
fn prepare_coding<const W: usize>(
    values: &mut [u64],
    ids: &[u32],
    base: usize,
    ranked: bool,
    sort: &mut [u64],
    rank_of: &mut [u32],
    arena: &mut Arena,
) -> CodingPrep {
    let k = values.len();
    rank_distinct::<W>(values, sort, rank_of, base);
    let span = 2f64.powi(8 * W as i32);
    let values = &*values;
    let dict = arena.pair(k, escape_bound(k, span), |syms, bits| {
        let mut prev = u64::MAX;
        for (s, &v) in syms.iter_mut().zip(values) {
            let (sym, raw, nb) = symbol::split(v.wrapping_sub(prev).wrapping_sub(1));
            *s = sym;
            bits.put(raw, nb);
            prev = v;
        }
    });
    let ranks = ranked.then(|| {
        let n = ids.len();
        let top = k - 1 + base;
        arena.pair(n, rank_escape_bound(n), |syms, bits| {
            if top < symbol::DIRECT as usize {
                // Every rank is its own symbol.
                let mut table = [0u8; 256];
                for (t, &r) in table.iter_mut().zip(&rank_of[..=top]) {
                    *t = r as u8;
                }
                for (s, &id) in syms.iter_mut().zip(ids) {
                    // SAFETY: ids are at most `top`, below 256.
                    *s = unsafe { get(&table, id as usize) };
                }
            } else {
                for (t, &id) in ids.iter().enumerate() {
                    // SAFETY: ids are at most `top`, within `rank_of`; `t < n`.
                    unsafe { emit32(get(rank_of, id as usize), syms, t, bits) };
                }
            }
        })
    });
    CodingPrep { k, dict, ranks }
}

/// Prepares the value codings of `src`: its dictionary and, against the
/// coded-order column modes `modes` (`W` bytes per gene), its mode coding.
pub(crate) fn prepare_values<const W: usize, E: Elements>(
    src: &E,
    modes: Option<&[u8]>,
    ws: &mut ValuesWs,
    arena: &mut Arena,
) -> ValuesPrep {
    let n = src.len();
    let (ids, mids, rank_of) = (&mut *ws.ids, &mut *ws.mids, &mut *ws.rank_of);
    assert!(ids.len() >= n && mids.len() >= n && rank_of.len() >= n && (modes.is_none() || ws.exc.len() >= n));
    let mut dict = ws.dict.open::<W>(n);
    let mut ke = 0;
    let mut e = 0;
    // SAFETY: `t < n`, within the elements and every per-element buffer;
    // dictionary ids, and so exception ids, stay below `n` (checked
    // above); the dictionary takes at most `n` values; the mode table
    // holds a value per coded gene.
    unsafe {
        match modes {
            None => {
                for t in 0..n {
                    set(ids, t, dict.insert::<W>(src.value(t)));
                }
            }
            Some(table) => {
                debug_assert!(table.len() >= W * (0..n).map(|t| src.gene(t) + 1).max().unwrap_or(0));
                // The distinct exceptions are distinct dictionary values, so
                // each dictionary id maps to its exception id (`rank_of`
                // holds the map until the dictionaries are ranked). Whether
                // a value is an exception is data, so only selections use it.
                let exc = &mut *ws.exc;
                rank_of[..n].fill(u32::MAX);
                for t in 0..n {
                    let v = src.value(t);
                    let id = dict.insert::<W>(v);
                    set(ids, t, id);
                    let exception = v != load::<W>(table, src.gene(t));
                    let mut eid = get(rank_of, id as usize);
                    // A value's first exception is rare: only it writes the
                    // map, so repeats of one value do not chain through it.
                    if exception && eid == u32::MAX {
                        eid = ke as u32;
                        set(rank_of, id as usize, eid);
                        set(exc, ke, v);
                        ke += 1;
                    }
                    set(mids, t, if exception { eid + 1 } else { 0 });
                    e += exception as usize;
                }
            }
        }
    }
    ws.dict.close::<W>(dict);
    let kd = ws.dict.k;
    let dict = prepare_coding::<W>(&mut ws.dict.values[..kd], &ws.ids[..n], 0, kd > 1, ws.sort, ws.rank_of, arena);
    let mode = modes.map(|_| ModePrep {
        coding: prepare_coding::<W>(&mut ws.exc[..ke], &ws.mids[..n], 1, ke > 0, ws.sort, ws.rank_of, arena),
        exceptions: e,
    });
    ValuesPrep { n, dict, mode }
}

/// A prepared sparse record: gap symbols of the coded ids and the values.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SparsePrep {
    /// Gap symbols of the coded ids, and their escape bits.
    pub(crate) gaps: Pair,
    /// The values, in the order of the coded ids.
    pub(crate) values: ValuesPrep,
}

/// Prepares the sparse record of a cell of `n` elements whose coded order
/// [`order`] left in `ows` (with input positions when `renumbered`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_sparse<const W: usize>(
    values: &[u8],
    n: usize,
    renumbered: bool,
    modes: Option<&[u8]>,
    n_genes: u32,
    ows: &OrderWs,
    vws: &mut ValuesWs,
    arena: &mut Arena,
) -> SparsePrep {
    let coded = &ows.coded[..n];
    let esc = &mut vws.esc[..n];
    let gaps = arena.pair(n, escape_bound(n, n_genes as f64), |syms, bits| {
        // SAFETY: one symbol and at most one escape per id; the kernel
        // set's ISA is supported.
        let e = unsafe { (ows.kernels.gaps)(coded, syms, esc) };
        for &g in &esc[..e] {
            let (_, raw, nb) = symbol::split(g as u64);
            bits.put32(raw, nb);
        }
    });
    let values = if renumbered {
        let src = Gathered::<W> { values, perm: &ows.perm[..n], genes: coded };
        prepare_values::<W, _>(&src, modes, vws, arena)
    } else {
        let src = InOrder::<W> { values, genes: coded };
        prepare_values::<W, _>(&src, modes, vws, arena)
    };
    SparsePrep { gaps, values }
}

/// Symbol counts in four interleaved tables, so runs of one symbol do not
/// chain their increments through memory.
pub(crate) struct Hist {
    lanes: [[u32; 256]; 4],
    pending: u64,
    total: [u64; 256],
}

impl Hist {
    /// Empty counts.
    pub(crate) fn new() -> Self {
        Hist { lanes: [[0; 256]; 4], pending: 0, total: [0; 256] }
    }

    /// Counts the symbols of one prepared stream.
    pub(crate) fn add(&mut self, syms: &[u8]) {
        if self.pending + syms.len() as u64 > u32::MAX as u64 {
            self.flush();
        }
        let mut quads = syms.chunks_exact(4);
        let [a, b, c, d] = &mut self.lanes;
        for q in &mut quads {
            a[q[0] as usize] += 1;
            b[q[1] as usize] += 1;
            c[q[2] as usize] += 1;
            d[q[3] as usize] += 1;
        }
        for &s in quads.remainder() {
            a[s as usize] += 1;
        }
        self.pending += syms.len() as u64;
    }

    fn flush(&mut self) {
        for (s, t) in self.total.iter_mut().enumerate() {
            *t += self.lanes.iter().map(|l| l[s] as u64).sum::<u64>();
        }
        self.lanes = [[0; 256]; 4];
        self.pending = 0;
    }

    /// Total count of each symbol, the four lane tables merged.
    pub(crate) fn finish(mut self) -> [u64; 256] {
        self.flush();
        self.total
    }
}
