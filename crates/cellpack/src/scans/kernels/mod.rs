//! Hot inner loops with per-ISA implementations, bound once per process.
//!
//! Every implementation is compiled into every build. [`crate::runtime`]
//! probes the CPU once and picks one [`Kernels`] table, whose entries are all
//! plain function pointers (or arrays of them); contexts built from
//! metadata carry the table, so a kernel call is one indirect call with no
//! dispatch logic in front of it. Platforms other than x86-64 use the scalar
//! set.

/// Which mapping kernels the measured selection policy prefers, per ISA.
pub(crate) mod mapping_policy;

use crate::scans::bits::BitReader;
use crate::scans::map::compact;
use crate::unchecked::{get, set};
use crate::scans::rans::{EncModel, Job, Slots, decode_pair_scalar, decode_scalar, encode_scalar, many_scalar};
use crate::scans::symbol::{BASE, DIRECT, EXTRA};

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "x86_64")]
mod map_avx2;

#[cfg(target_arch = "x86_64")]
mod encode_x86;

/// Eight-lane typed mapping kernels for AVX-512 (wide and narrow outputs).
#[cfg(target_arch = "x86_64")]
pub(crate) mod typed;

#[cfg(test)]
mod boundaries;

/// Decodes `out.len()` symbols of one stream, writes the lanes' final
/// states to the last argument, and reports exact consumption: every
/// implementation asserts it in debug builds and returns true otherwise.
/// Callers guarantee a well-formed stream of the chunk, `out` holding the
/// symbol count, `ends` at least `lanes` entries, and the ISA the set
/// requires.
pub(crate) type RansFn = unsafe fn(&Slots, &[u8], &mut [u8], &mut [u32]) -> bool;

/// Decodes two streams of equal symbol count and lane count at once, as
/// [`RansFn`] for both; the flag carries the same debug-only exactness check.
pub(crate) type RansPairFn =
    unsafe fn(&Slots, &[u8], &mut [u8], &Slots, &[u8], &mut [u8], &mut [u32], &mut [u32]) -> bool;

/// `out[i] = dict[syms[i]]` for one value width. Callers guarantee every
/// symbol indexes a dictionary entry, that `dict` extends at least
/// [`DICT_SLACK`] bytes past the last entry, and that
/// `out.len() == syms.len() * width`.
pub(crate) type ExpandFn = unsafe fn(&[u8], &[u8], &mut [u8]);

/// `out[i] = dict[idx[i]]` with 32-bit indices, under the same contract
/// as [`ExpandFn`]. SIMD callers also guarantee indices fit in `i32`.
pub(crate) type Expand32Fn = unsafe fn(&[u8], &[u32], &mut [u8]);

/// Integers from symbols: `out[t] = BASE[s_t] | (the next EXTRA[s_t] escape
/// bits)`, for symbols up to `MAX_SYMBOL_32`. Escape bits past the end of
/// `bits` read as zero. Returns the escape bits consumed; callers check it
/// against the stream length.
pub(crate) type FieldsFn = unsafe fn(&[u8], &[u8], &mut [u32]) -> u64;

/// In place, `v[t] = v[t-1] + 1 + v[t]` from `v[-1] = -1` (gaps to strictly
/// ascending values), low 32 bits; returns the exact last value as u64, which
/// bounds all of them.
pub(crate) type AscendFn = unsafe fn(&mut [u32]) -> u64;

/// Maps a row of 4-byte values from a dictionary: `col = map[ids[t]]`,
/// value `dict[ranks[t]]`; drops `UNMAPPED` columns and returns the count
/// kept. Callers guarantee every id indexes `map` (and fits a signed 32-bit
/// index), every rank indexes `dict`, and `cols`/`vals` hold `ids.len()`.
pub(crate) type MapU8Fn = unsafe fn(&[u32], &[u32], &[u8], &[u32], &mut [u32], &mut [u32]) -> usize;
/// [`MapU8Fn`] with 32-bit ranks (dictionaries past 192 entries).
pub(crate) type MapU32Fn =
    unsafe fn(&[u32], &[u32], &[u32], &[u32], &mut [u32], &mut [u32]) -> usize;

/// As [`MapU8Fn`] for a mode-coded row (ids, map, ranks, dict, modes, cols,
/// vals): rank 0 takes `modes[id]`, rank `r` takes `dict[r - 1]`. Every id
/// also indexes `modes`, and every rank is at most `dict.len()`.
pub(crate) type MapModesU8Fn =
    unsafe fn(&[u32], &[u32], &[u8], &[u32], &[u32], &mut [u32], &mut [u32]) -> usize;
/// [`MapModesU8Fn`] with 32-bit ranks.
pub(crate) type MapModesU32Fn =
    unsafe fn(&[u32], &[u32], &[u32], &[u32], &[u32], &mut [u32], &mut [u32]) -> usize;

/// As [`MapU8Fn`] for raw 4-byte little-endian values copied unchanged
/// (ids, map, raw bytes, cols, vals); `raw` holds one value per id.
pub(crate) type MapRaw32Fn = unsafe fn(&[u32], &[u32], &[u8], &mut [u32], &mut [u32]) -> usize;

/// Raw stored values as 32-bit lanes: zero- or sign-extended 1- and 2-byte
/// values, or 4-byte values as they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Src {
    /// 1 byte, zero-extended.
    U8 = 0,
    /// 1 byte, sign-extended.
    I8 = 1,
    /// 2 bytes little-endian, zero-extended.
    U16 = 2,
    /// 2 bytes little-endian, sign-extended.
    I16 = 3,
    /// 4 bytes little-endian, as they are.
    W32 = 4,
}

/// The conversion of a 32-bit lane to the output's element: its bits (an
/// integer kept, widened or reinterpreted, truncated to the output's width
/// when stored), or an unsigned or signed integer to f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Conv {
    /// The lane's bits, as the output element's bits.
    Bits = 0,
    /// An unsigned integer to the nearest f32 (`as` semantics).
    U32ToF32 = 1,
    /// A signed integer to the nearest f32 (`as` semantics).
    I32ToF32 = 2,
}

/// Lane `t` of raw values.
///
/// # Safety
/// `b` holds element `t` of the source width.
#[inline(always)]
pub(crate) unsafe fn src_value(b: &[u8], t: usize, src: Src) -> u32 {
    debug_assert!((t + 1) * [1, 1, 2, 2, 4][src as usize] <= b.len());
    let p = b.as_ptr();
    // SAFETY: the caller guarantees the element's bytes are in `b`.
    unsafe {
        match src {
            Src::U8 => *p.add(t) as u32,
            Src::I8 => *p.add(t) as i8 as i32 as u32,
            Src::U16 => u16::from_le_bytes(p.add(2 * t).cast::<[u8; 2]>().read_unaligned()) as u32,
            Src::I16 => i16::from_le_bytes(p.add(2 * t).cast::<[u8; 2]>().read_unaligned()) as i32 as u32,
            Src::W32 => u32::from_le_bytes(p.add(4 * t).cast::<[u8; 4]>().read_unaligned()),
        }
    }
}

/// The scalar form of [`Conv`]: a lane converted into the output's bits.
#[inline(always)]
pub(crate) fn conv_value(v: u32, conv: Conv) -> u32 {
    match conv {
        Conv::Bits => v,
        Conv::U32ToF32 => (v as f32).to_bits(),
        Conv::I32ToF32 => (v as i32 as f32).to_bits(),
    }
}

/// Maps a row of raw values to `out`-byte elements (ids, map, raw, cols,
/// vals): as [`MapU8Fn`], with `vals` holding room for every element.
pub(crate) type MapLanesFn = unsafe fn(&[u32], &[u32], &[u8], &mut [u32], *mut u8) -> usize;

/// Maps a row of byte-ranked dictionary values (entries as 32-bit lanes)
/// to `out`-byte elements (ids, map, ranks, dict, cols, vals).
pub(crate) type MapDictLanesFn = unsafe fn(&[u32], &[u32], &[u8], &[u32], &mut [u32], *mut u8) -> usize;

/// Values of a decoded row read at chosen positions, as 4-byte patterns.
#[derive(Clone, Copy)]
pub(crate) enum At<'a> {
    /// Every element has this value.
    Const(u32),
    /// `dict[ranks[t]]`.
    Dict8(&'a [u8], &'a [u32]),
    /// [`At::Dict8`] with 32-bit ranks.
    Dict32(&'a [u32], &'a [u32]),
    /// Mode-coded (ranks, dict, modes): rank 0 is `modes[id]`, rank `r` is
    /// `dict[r - 1]`.
    Modes8(&'a [u8], &'a [u32], &'a [u32]),
    /// [`At::Modes8`] with 32-bit ranks.
    Modes32(&'a [u32], &'a [u32], &'a [u32]),
    /// Every element takes its column mode.
    ModesOne(&'a [u32]),
    /// Raw little-endian values, converted.
    Raw(&'a [u8], Src, Conv),
}

/// Value of element `t` (gene `id`) of an [`At`] row.
///
/// # Safety
/// `t` is an element of the row, `id` indexes the modes, and its rank
/// indexes the dictionary as [`MapU8Fn`] and [`MapModesU8Fn`] require.
#[inline(always)]
pub(crate) unsafe fn at_value(v: &At, t: usize, id: u32) -> u32 {
    // SAFETY: as documented.
    unsafe {
        match *v {
            At::Const(c) => c,
            At::Dict8(r, d) => get(d, get(r, t) as usize),
            At::Dict32(r, d) => get(d, get(r, t) as usize),
            At::Modes8(r, d, m) => mode_value(id, get(r, t) as usize, d, m),
            At::Modes32(r, d, m) => mode_value(id, get(r, t) as usize, d, m),
            At::ModesOne(m) => get(m, id as usize),
            At::Raw(b, src, conv) => conv_value(src_value(b, t, src), conv),
        }
    }
}

/// Mapped columns and values of the elements at `positions`, in that
/// order (ids, positions, map, values, cols, vals). Callers guarantee every
/// position is below `ids.len()` (and the length of every per-element
/// slice of the values), every id indexes `map` (and the modes), ranks
/// index the dictionary as in [`MapU8Fn`] and [`MapModesU8Fn`], every index
/// fits a signed 32-bit gather, and `cols`/`vals` hold `positions.len()`.
pub(crate) type MapAtFn = unsafe fn(&[u32], &[u32], &[u32], At, &mut [u32], &mut [u32]);

/// Scatters every element of a row into a dense output row (ids, map,
/// values, out): `out[map[id]] = value` for columns other than
/// [`UNMAPPED`](crate::scans::UNMAPPED). Returns false, possibly after
/// writing some elements, when a mapped column lies outside `out`. Callers
/// guarantee ids index `map` (and the modes), ranks index the dictionary, and
/// per-element values cover every id, with indices fitting a signed gather.
pub(crate) type ScatterFn = unsafe fn(&[u32], &[u32], At, &mut [u32]) -> bool;

/// Writes the positions of the non-zero bytes of `marks` (a multiple of 64
/// long), ascending, to `out` and returns their count. `out` must hold the
/// count plus [`SCAN_SLACK`].
pub(crate) type ScanFn = unsafe fn(&[u8], &mut [u32]) -> usize;

/// Positions of the set bits of `words` (bit `b` of word `w` is position
/// `64 w + b`), ascending, to the first argument after `pos`, clearing the
/// words, and `perm[j] = pos[coded[j]]`; returns their count. The positions
/// buffer holds the count plus [`SCAN_SLACK`]; `pos` covers every position.
pub(crate) type ScanBitsFn = unsafe fn(&mut [u64], &[u32], &mut [u32], &mut [u32]) -> usize;

/// Symbols of the gaps of ascending ids (`c[t] - c[t-1] - 1`, the first gap
/// `c[0]`, every gap below `2^31`) to `syms` (one per id); the gaps whose
/// symbols carry escape bits go, in order, to `esc` (one entry per id).
/// Returns their count.
pub(crate) type GapsFn = unsafe fn(&[u32], &mut [u8], &mut [u32]) -> usize;

/// rANS-codes symbols with a model and initial states (see
/// [`crate::scans::rans::encode_scalar`] for the contract). Unsafe through
/// that contract and the ISA it requires.
pub(crate) type RansEncodeFn = unsafe fn(&EncModel, &[u8], usize, &mut [u32; 32], &mut [u16]) -> Option<usize>;

/// rANS-codes up to three jobs of one symbol count with one lane count,
/// each as [`RansEncodeFn`] (see [`crate::scans::rans::Job`]). Unsafe
/// through that contract and the ISA it requires.
pub(crate) type RansManyFn = unsafe fn(&mut [Job], usize);

/// Entries past the last position a scan kernel may overwrite.
pub(crate) const SCAN_SLACK: usize = 32;

/// Readable bytes past the dictionary's last entry, for gathers that load
/// wider than the value.
pub(crate) const DICT_SLACK: usize = 32;

/// Instruction set of a kernel set, ordered by capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Isa {
    /// Portable loops, the only set off x86-64.
    Scalar,
    /// 8-lane vectors (AVX2, popcnt, FMA).
    Avx2,
    /// 16-lane vectors (AVX-512 F, BW, DQ and VL on top of AVX2).
    Avx512,
}

impl Isa {
    /// Every set, by ascending capability.
    pub const ALL: [Isa; 3] = [Isa::Scalar, Isa::Avx2, Isa::Avx512];

    /// The name `CELLPACK_KERNELS` accepts and reports show.
    pub const fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        }
    }

    /// Whether this CPU runs the set.
    pub fn supported(self) -> bool {
        select(self).isa == self
    }
}

impl std::fmt::Display for Isa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Lane kernels by `[Src][Conv][log2(output width)]`, output widths 1, 2, 4.
pub(crate) type LanesTable = [[[Option<MapLanesFn>; 3]; 3]; 5];

/// Index of an output width of 1, 2 or 4 bytes in the lane tables.
#[inline(always)]
pub(crate) const fn out_slot(width: usize) -> usize {
    width.trailing_zeros() as usize
}

/// The function pointers of one instruction set.
///
/// Every field is bound once (see [`select`]); callers pass the table around
/// instead of reading a global, so a kernel call is one indirect call. A
/// kernel's contract is that of its function type, and every kernel bound
/// works on any input its contract admits, so tables differ only in speed,
/// in the ISA they require, and in which optional lane kernels they carry
/// (`map_lanes`, `map_dict8_lanes`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Kernels {
    /// The set these kernels belong to.
    pub(crate) isa: Isa,
    /// Indexed by `log2(lanes)`, lanes 1..=32.
    pub(crate) rans: [RansFn; 6],
    /// Indexed by `log2(lanes)`.
    pub(crate) rans_pair: [RansPairFn; 6],
    /// Indexed by `log2(value width)`, widths 1..=8.
    pub(crate) expand: [ExpandFn; 4],
    /// Indexed like `expand`.
    pub(crate) expand32: [Expand32Fn; 4],
    /// Marks (one byte per gene) to the positions of the marked genes.
    pub(crate) scan: ScanFn,
    /// rANS encoding of one stream.
    pub(crate) rans_encode: RansEncodeFn,
    /// rANS encoding of up to three streams, interleaved.
    pub(crate) rans_many: RansManyFn,
    /// Set bits of the coded-order bitmap to ascending coded ids and their
    /// input positions.
    pub(crate) scan_bits: ScanBitsFn,
    /// Ascending ids to gap symbols and the gap values that escape.
    pub(crate) gaps: GapsFn,
    /// Symbols and escape bits back to integers.
    pub(crate) fields: FieldsFn,
    /// Gaps to strictly ascending values.
    pub(crate) ascend: AscendFn,
    /// Compact row mapping with byte ranks.
    pub(crate) map_u8: MapU8Fn,
    /// Compact row mapping with 32-bit ranks.
    pub(crate) map_u32: MapU32Fn,
    /// Compact mode-coded row mapping with byte ranks.
    pub(crate) map_modes_u8: MapModesU8Fn,
    /// Compact mode-coded row mapping with 32-bit ranks.
    pub(crate) map_modes_u32: MapModesU32Fn,
    /// Compact row mapping of raw 4-byte values.
    pub(crate) map_raw32: MapRaw32Fn,
    /// Row mapping at chosen positions.
    pub(crate) map_at: MapAtFn,
    /// Dense scatter of a row.
    pub(crate) scatter: ScatterFn,
    /// Raw values of a source kind, converted, to elements of a width.
    pub(crate) map_lanes: LanesTable,
    /// Byte-ranked dictionary values to elements of 1, 2 or 4 bytes, by
    /// [`out_slot`].
    pub(crate) map_dict8_lanes: [Option<MapDictLanesFn>; 3],
}

/// Name of the bound kernel set (binding the runtime from the environment
/// if nothing has yet); `"unbound"` when the environment is invalid.
pub fn kernel_set() -> &'static str {
    crate::runtime::get().map_or("unbound", |rt| rt.isa().name())
}

/// `out[i] = dict[syms[i]]` for `W`-byte entries (the [`ExpandFn`] contract).
///
/// # Safety
/// Every symbol indexes a `W`-byte entry of `dict`.
#[inline]
pub(crate) unsafe fn expand_scalar<const W: usize>(dict: &[u8], syms: &[u8], out: &mut [u8]) {
    for (o, &s) in out.chunks_exact_mut(W).zip(syms) {
        // SAFETY: as documented.
        unsafe { std::ptr::copy_nonoverlapping(dict.as_ptr().add(s as usize * W), o.as_mut_ptr(), W) };
    }
}

/// `out[i] = dict[idx[i]]` for `W`-byte entries (the [`Expand32Fn`] contract).
///
/// # Safety
/// Every index names a `W`-byte entry of `dict`.
#[inline]
pub(crate) unsafe fn expand32_scalar<const W: usize>(dict: &[u8], idx: &[u32], out: &mut [u8]) {
    for (o, &s) in out.chunks_exact_mut(W).zip(idx) {
        // SAFETY: as documented.
        unsafe { std::ptr::copy_nonoverlapping(dict.as_ptr().add(s as usize * W), o.as_mut_ptr(), W) };
    }
}

/// Scalar field unpacking of `syms` from escape bit `bit`; returns the bits consumed in total.
#[inline(always)]
pub(crate) fn fields_tail(syms: &[u8], bits: &[u8], out: &mut [u32], bit: u64) -> u64 {
    let mut reader = BitReader::at(bits, bit);
    for (o, &s) in out.iter_mut().zip(syms) {
        *o = (BASE[s as usize] | reader.take(EXTRA[s as usize] as u32)) as u32;
    }
    reader.consumed()
}

/// The [`FieldsFn`] contract from the first escape bit.
fn fields_scalar(syms: &[u8], bits: &[u8], out: &mut [u32]) -> u64 {
    fields_tail(syms, bits, out, 0)
}

/// Scalar running sums of `v` continuing from exact value `c`.
#[inline(always)]
pub(crate) fn ascend_tail(v: &mut [u32], mut c: u64) -> u64 {
    for x in v.iter_mut() {
        c = c.wrapping_add(1 + *x as u64);
        *x = c as u32;
    }
    c
}

/// The [`MapU8Fn`] contract.
unsafe fn map_u8_scalar(ids: &[u32], map: &[u32], ranks: &[u8], dict: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    // SAFETY: the `MapU8Fn` contract bounds every index.
    unsafe { compact(ids, map, |t| get(dict, get(ranks, t) as usize), cols, vals) }
}

/// The [`MapU32Fn`] contract.
unsafe fn map_u32_scalar(ids: &[u32], map: &[u32], ranks: &[u32], dict: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    // SAFETY: the `MapU32Fn` contract bounds every index.
    unsafe { compact(ids, map, |t| get(dict, get(ranks, t) as usize), cols, vals) }
}

/// Value of a mode-coded element: the column mode for rank 0, else a
/// dictionary entry. Both are read and one is selected, without a branch.
///
/// # Safety
/// `id` indexes `modes`, `rank <= dict.len()`, and `dict` is not empty
/// (rows with mode ranks have at least one exception).
#[inline(always)]
pub(crate) unsafe fn mode_value(id: u32, rank: usize, dict: &[u32], modes: &[u32]) -> u32 {
    // SAFETY: as documented; rank 0 reads entry 0 and discards it.
    unsafe {
        let m = get(modes, id as usize);
        let d = get(dict, rank.max(1) - 1);
        if rank == 0 { m } else { d }
    }
}

/// The [`MapModesU8Fn`] contract.
unsafe fn map_modes_u8_scalar(ids: &[u32], map: &[u32], ranks: &[u8], dict: &[u32], modes: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    // SAFETY: the `MapModesU8Fn` contract bounds every index.
    unsafe { compact(ids, map, |t| mode_value(get(ids, t), get(ranks, t) as usize, dict, modes), cols, vals) }
}

/// The [`MapModesU32Fn`] contract.
unsafe fn map_modes_u32_scalar(ids: &[u32], map: &[u32], ranks: &[u32], dict: &[u32], modes: &[u32], cols: &mut [u32], vals: &mut [u32]) -> usize {
    // SAFETY: the `MapModesU32Fn` contract bounds every index.
    unsafe { compact(ids, map, |t| mode_value(get(ids, t), get(ranks, t) as usize, dict, modes), cols, vals) }
}

/// The [`MapRaw32Fn`] contract.
unsafe fn map_raw32_scalar(ids: &[u32], map: &[u32], raw: &[u8], cols: &mut [u32], vals: &mut [u32]) -> usize {
    // SAFETY: the `MapRaw32Fn` contract: `raw` holds one value per id.
    unsafe { compact(ids, map, |t| src_value(raw, t, Src::W32), cols, vals) }
}

/// The [`MapAtFn`] contract.
unsafe fn map_at_scalar(ids: &[u32], positions: &[u32], map: &[u32], v: At, cols: &mut [u32], vals: &mut [u32]) {
    // SAFETY: the `MapAtFn` contract bounds positions, ids, values and
    // the outputs.
    unsafe {
        for (i, &p) in positions.iter().enumerate() {
            let id = get(ids, p as usize);
            set(cols, i, get(map, id as usize));
            set(vals, i, at_value(&v, p as usize, id));
        }
    }
}

/// The [`ScatterFn`] contract.
unsafe fn scatter_scalar(ids: &[u32], map: &[u32], v: At, out: &mut [u32]) -> bool {
    // SAFETY: the `ScatterFn` contract bounds ids and values; each store's
    // column is checked against `out` first.
    unsafe {
        for (t, &id) in ids.iter().enumerate() {
            let col = get(map, id as usize);
            if col != crate::scans::UNMAPPED {
                if col as usize >= out.len() {
                    return false;
                }
                set(out, col as usize, at_value(&v, t, id));
            }
        }
    }
    true
}

/// The [`AscendFn`] contract; gaps start below the first id, so the running
/// sum starts at `u64::MAX` as -1.
fn ascend_scalar(v: &mut [u32]) -> u64 {
    ascend_tail(v, u64::MAX)
}

/// One bit per byte of a 64-byte block of 0/1 marks.
#[inline(always)]
fn block_mask(block: &[u8]) -> u64 {
    let mut m = 0u64;
    for (q, eight) in block.chunks_exact(8).enumerate() {
        let x = u64::from_le_bytes(eight.try_into().unwrap());
        m |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * q);
    }
    m
}

/// The [`ScanFn`] contract: `out` holds the count plus [`SCAN_SLACK`].
unsafe fn scan_scalar(marks: &[u8], out: &mut [u32]) -> usize {
    let mut j = 0;
    for (w, block) in marks.chunks_exact(64).enumerate() {
        let mut b = block_mask(block);
        while b != 0 {
            // SAFETY: `j` stays below the count, which `out` holds.
            unsafe { set(out, j, ((w as u32) << 6) | b.trailing_zeros()) };
            j += 1;
            b &= b - 1;
        }
    }
    j
}

/// The [`ScanBitsFn`] contract, portably.
unsafe fn scan_bits_scalar(words: &mut [u64], pos: &[u32], coded: &mut [u32], perm: &mut [u32]) -> usize {
    let mut j = 0;
    for (w, word) in words.iter_mut().enumerate() {
        let mut b = std::mem::take(word);
        while b != 0 {
            let c = w * 64 + b.trailing_zeros() as usize;
            // SAFETY: `j` stays below the count, which both outputs hold;
            // `pos` covers every position.
            unsafe {
                set(coded, j, c as u32);
                set(perm, j, get(pos, c));
            }
            j += 1;
            b &= b - 1;
        }
    }
    j
}

/// Symbol of an integer below `2^32` (see [`crate::scans::symbol::split`]).
#[inline(always)]
pub(crate) fn symbol32(v: u32) -> u8 {
    if v < DIRECT as u32 {
        v as u8
    } else if v < 256 {
        DIRECT as u8
    } else {
        (DIRECT as u32 + 24 - v.leading_zeros()) as u8
    }
}

/// The [`GapsFn`] contract, portably.
unsafe fn gaps_scalar(coded: &[u32], syms: &mut [u8], esc: &mut [u32]) -> usize {
    let (mut prev, mut e) = (u32::MAX, 0);
    for (t, &c) in coded.iter().enumerate() {
        let g = c.wrapping_sub(prev).wrapping_sub(1);
        prev = c;
        // SAFETY: `e <= t < coded.len()`, which both outputs hold.
        unsafe {
            set(syms, t, symbol32(g));
            set(esc, e, g);
        }
        e += (g >= DIRECT as u32) as usize;
    }
    e
}

/// The portable kernel set: no lane kernels, no dictionary-lane kernels.
pub(crate) const SCALAR: Kernels = Kernels {
    isa: Isa::Scalar,
    rans: [
        decode_scalar::<1>,
        decode_scalar::<2>,
        decode_scalar::<4>,
        decode_scalar::<8>,
        decode_scalar::<16>,
        decode_scalar::<32>,
    ],
    rans_pair: [
        decode_pair_scalar::<1>,
        decode_pair_scalar::<2>,
        decode_pair_scalar::<4>,
        decode_pair_scalar::<8>,
        decode_pair_scalar::<16>,
        decode_pair_scalar::<32>,
    ],
    expand: [
        expand_scalar::<1>,
        expand_scalar::<2>,
        expand_scalar::<4>,
        expand_scalar::<8>,
    ],
    expand32: [
        expand32_scalar::<1>,
        expand32_scalar::<2>,
        expand32_scalar::<4>,
        expand32_scalar::<8>,
    ],
    scan: scan_scalar,
    rans_encode: encode_scalar,
    rans_many: many_scalar,
    scan_bits: scan_bits_scalar,
    gaps: gaps_scalar,
    fields: fields_scalar,
    ascend: ascend_scalar,
    map_u8: map_u8_scalar,
    map_u32: map_u32_scalar,
    map_modes_u8: map_modes_u8_scalar,
    map_modes_u32: map_modes_u32_scalar,
    map_raw32: map_raw32_scalar,
    map_at: map_at_scalar,
    scatter: scatter_scalar,
    map_lanes: [[[None; 3]; 3]; 5],
    map_dict8_lanes: [None; 3],
};

/// The best kernel set at or below `cap` this CPU supports.
pub(crate) fn select(cap: Isa) -> &'static Kernels {
    #[cfg(target_arch = "x86_64")]
    {
        if cap >= Isa::Avx512 && x86::avx512_supported() {
            return x86::avx512_set();
        }
        if cap >= Isa::Avx2 && x86::avx2_supported() {
            return &x86::AVX2;
        }
    }
    let _ = cap;
    &SCALAR
}

/// Every kernel set this CPU supports, for equivalence tests and benchmarks.
#[doc(hidden)]
pub fn supported_kernel_sets() -> Vec<&'static str> {
    Isa::ALL.into_iter().filter(|isa| isa.supported()).map(Isa::name).collect()
}

/// Every set this CPU runs, for the equivalence tests. The AVX-512 set
/// appears twice on a VBMI2 CPU: plain and with the byte-compress scan.
#[cfg(test)]
pub(crate) fn all_supported() -> Vec<Kernels> {
    let mut v = vec![SCALAR];
    #[cfg(target_arch = "x86_64")]
    {
        if x86::avx2_supported() {
            v.push(x86::AVX2);
        }
        if x86::avx512_supported() {
            v.push(x86::AVX512);
            v.push(*x86::avx512_set());
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scans::rans::{self, M};
    use crate::scans::tables::{cumulative, fill_slots, normalize};
    use crate::scans::rans::EncModel;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Skewed symbols over `alphabet` letters, with occasional rare ones.
    fn symbols(rng: &mut Rng, n: usize, alphabet: u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                let r = rng.next();
                let v = if r.is_multiple_of(50) {
                    (r >> 8) % alphabet
                } else {
                    (r >> 8) & (r >> 20) & 3
                };
                v.min(alphabet - 1) as u8
            })
            .collect()
    }

    /// Interleaved rANS encoding by division, one symbol at a time.
    fn reference_encode(freq: &[u16; 256], cum: &[u16; 256], syms: &[u8], lanes: usize, payload: &[u8]) -> Vec<u8> {
        let l = rans::L;
        let mut x: Vec<u32> = (0..lanes)
            .map(|j| {
                let lo = payload.get(2 * j).copied().unwrap_or(0) as u32;
                let hi = payload.get(2 * j + 1).copied().unwrap_or(0) as u32;
                l + (lo | hi << 8)
            })
            .collect();
        let mut words = Vec::new();
        for (j, &s) in syms.iter().enumerate().rev() {
            let (f, c) = (freq[s as usize] as u64, cum[s as usize] as u32);
            let v = &mut x[j % lanes];
            if *v as u64 >= (((l >> rans::SCALE) as u64) << 16) * f {
                words.push(*v as u16);
                *v >>= 16;
            }
            let f = f as u32;
            *v = ((*v / f) << rans::SCALE) + *v % f + c;
        }
        let mut out: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.extend(words.iter().rev().flat_map(|w| w.to_le_bytes()));
        out
    }

    #[test]
    fn every_kernel_set_decodes_every_lane_count_identically() {
        let sets = all_supported();
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for &n in &[1usize, 7, 8, 31, 32, 33, 100, 257, 1000, 4099] {
            for alphabet in [1u64, 3, 185] {
                let syms = symbols(&mut rng, n, alphabet);
                let mut counts = [0u64; 256];
                syms.iter().for_each(|&s| counts[s as usize] += 1);
                let freq = normalize(&counts);
                assert_eq!(freq.iter().map(|&f| f as u32).sum::<u32>(), M);
                let cum = cumulative(&freq);
                // Two models in one allocation, as a chunk's tables are: the
                // second stream uses its own model.
                let mut counts2 = counts;
                counts2
                    .iter_mut()
                    .take(alphabet as usize + 3)
                    .enumerate()
                    .for_each(|(s, c)| *c = *c * (s as u64 % 3 + 1) + 1);
                let (freq2, mut tables) = (normalize(&counts2), vec![[0u32; M as usize]; 2]);
                let cum2 = cumulative(&freq2);
                fill_slots(&freq, &mut tables[0]);
                fill_slots(&freq2, &mut tables[1]);
                let (slots, slots2) = (&tables[0], &tables[1]);
                for log in 0..6 {
                    let lanes = 1usize << log;
                    let mut words = vec![0u16; n + rans::WORD_SLACK];
                    let pay: Vec<u8> = (0..2 * lanes).map(|i| (i * 37 + n) as u8).collect();
                    let pay = &pay[..(n % (2 * lanes + 1)).min(2 * lanes)];
                    let (m1, m2) = (EncModel::new(&freq), EncModel::new(&freq2));
                    let stream = reference_encode(&freq, &cum, &syms, lanes, pay);
                    // A second stream of the same length: the reversed symbols.
                    let rev: Vec<u8> = syms.iter().rev().copied().collect();
                    let stream2 = reference_encode(&freq2, &cum2, &rev, lanes, &[]);
                    for set in &sets {
                        for (m, sy, p, want) in [(&m1, &syms, pay, &stream), (&m2, &rev, &[][..], &stream2)] {
                            let mut got = vec![0u8; rans::stream_bound(n, lanes)];
                            let len = rans::encode(set, m, sy, lanes, p, &mut words, &mut got).unwrap();
                            assert_eq!(&got[..len], &want[..], "{} encode lanes={lanes} n={n}", set.isa);
                        }
                    }
                    // Interleaved runs match single streams; a stream with a
                    // symbol of zero frequency fails alone.
                    let mut fz = freq;
                    fz[syms[0] as usize] = 0;
                    let mz = EncModel::new(&fz);
                    for set in &sets {
                        let bound = rans::stream_bound(n, lanes);
                        let (mut wa, mut wb, mut wc) = (vec![0u16; n + rans::WORD_SLACK], vec![0u16; n + rans::WORD_SLACK], vec![0u16; n + rans::WORD_SLACK]);
                        let (mut oa, mut ob, mut oc) = (vec![0u8; bound], vec![0u8; bound], vec![0u8; bound]);
                        let got = rans::encode_many(
                            set,
                            lanes,
                            [
                                Some(rans::Stream { model: &m1, syms: &syms, payload: pay, words: &mut wa, out: &mut oa }),
                                Some(rans::Stream { model: &m2, syms: &rev, payload: &[], words: &mut wb, out: &mut ob }),
                                Some(rans::Stream { model: &mz, syms: &syms, payload: &[], words: &mut wc, out: &mut oc }),
                            ],
                        );
                        assert_eq!(got[2], None, "{} zero frequency lanes={lanes} n={n}", set.isa);
                        assert_eq!(got[0].map(|l| &oa[..l]), Some(&stream[..]), "{} many lanes={lanes} n={n}", set.isa);
                        assert_eq!(got[1].map(|l| &ob[..l]), Some(&stream2[..]), "{} many lanes={lanes} n={n}", set.isa);
                    }
                    let (stream, stream2) = (&stream[..], &stream2[..]);
                    for set in &sets {
                        let (mut o1, mut o2) = (vec![0u8; n], vec![0u8; n]);
                        let (mut e1, mut e2) = ([0u32; 32], [0u32; 32]);
                        // SAFETY: `all_supported` lists only sets this CPU runs.
                        assert!(
                            // SAFETY: supported CPU set and correctly sized stream/output buffers.
                            unsafe {
                                (set.rans_pair[log])(
                                    slots, stream, &mut o1, slots2, stream2, &mut o2, &mut e1,
                                    &mut e2,
                                )
                            },
                            "{} pair lanes={lanes}",
                            set.isa
                        );
                        assert_eq!(
                            (&o1, &o2),
                            (&syms, &rev),
                            "{} pair lanes={lanes} n={n}",
                            set.isa
                        );
                        let mut got = vec![0u8; pay.len()];
                        assert!(
                            rans::payload(&e1[..lanes], pay.len(), &mut got) && got == pay,
                            "{} payload",
                            set.isa
                        );
                        assert!(
                            rans::payload(&e2[..lanes], 0, &mut []),
                            "{} empty payload",
                            set.isa
                        );
                    }
                    for set in &sets {
                        let mut out = vec![0u8; n];
                        let mut ends = [0u32; 32];
                        // SAFETY: `all_supported` lists only sets this CPU runs.
                        assert!(
                            // SAFETY: supported CPU set and correctly sized stream/output buffers.
                            unsafe { (set.rans[log])(slots, stream, &mut out, &mut ends) },
                            "{} lanes={lanes} n={n}",
                            set.isa
                        );
                        assert_eq!(out, syms, "{} lanes={lanes} n={n}", set.isa);
                        let mut got = vec![0u8; pay.len()];
                        assert!(
                            rans::payload(&ends[..lanes], pay.len(), &mut got) && got == pay,
                            "{} payload",
                            set.isa
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_kernel_set_scans_marks_identically() {
        let mut rng = Rng(5);
        for blocks in [0usize, 1, 3, 17, 422] {
            for density in [0u64, 1, 8, 32, 64] {
                let marks: Vec<u8> = (0..blocks * 64)
                    .map(|_| ((rng.next() % 64) < density) as u8)
                    .collect();
                let count = marks.iter().filter(|&&m| m != 0).count();
                let want: Vec<u32> = (0..marks.len() as u32)
                    .filter(|&p| marks[p as usize] != 0)
                    .collect();
                for set in all_supported() {
                    let mut out = vec![u32::MAX; count + SCAN_SLACK];
                    // SAFETY: `out` holds the count plus the slack, on a supported set.
                    assert_eq!(
                        // SAFETY: supported CPU set; output includes SCAN_SLACK.
                        unsafe { (set.scan)(&marks, &mut out) },
                        count,
                        "{}",
                        set.isa
                    );
                    assert_eq!(
                        &out[..count],
                        &want[..],
                        "{} blocks {blocks} density {density}",
                        set.isa
                    );
                }
            }
        }
    }

    #[test]
    fn every_kernel_set_unpacks_fields_and_ascends_identically() {
        use crate::scans::bits::{BitWriter, exact};
        use crate::scans::symbol::split;
        let mut rng = Rng(31);
        for &n in &[0usize, 1, 15, 16, 17, 64, 100, 1000] {
            for escape_rate in [0u64, 1, 10, 100] {
                let vals: Vec<u64> = (0..n)
                    .map(|_| {
                        if rng.next() % 100 < escape_rate {
                            (rng.next() >> (32 + rng.next() % 32)).max(192)
                        } else {
                            rng.next() % 40
                        }
                    })
                    .collect();
                let mut syms = vec![0u8; n];
                let mut buf = vec![0u8; 8 * n + 8];
                let mut w = BitWriter::new(&mut buf);
                for (s, &v) in syms.iter_mut().zip(&vals) {
                    let (sym, raw, nb) = split(v);
                    *s = sym;
                    w.put(raw, nb);
                }
                let len = w.finish().unwrap();
                let want: Vec<u32> = vals.iter().map(|&v| v as u32).collect();
                let asc: Vec<u32> = vals
                    .iter()
                    .scan(u64::MAX, |c, &g| {
                        *c = c.wrapping_add(1 + g);
                        Some(*c as u32)
                    })
                    .collect();
                let last = vals.iter().fold(u64::MAX, |c, &g| c.wrapping_add(1 + g));
                for set in all_supported() {
                    let mut out = vec![0u32; n];
                    // SAFETY: a supported set; symbols stay within 32-bit buckets.
                    let used = unsafe { (set.fields)(&syms, &buf[..len], &mut out) };
                    assert_eq!(out, want, "{} n={n} rate={escape_rate}", set.isa);
                    assert!(
                        exact(used, len),
                        "{} consumed {used} of {len} bytes",
                        set.isa
                    );
                    // SAFETY: as above.
                    assert_eq!(unsafe { (set.ascend)(&mut out) }, last, "{}", set.isa);
                    assert_eq!(out, asc, "{} ascend n={n}", set.isa);
                    // Truncated escape bits: every set reads the same zeros.
                    let cut = len / 2;
                    let (mut a, mut b) = (vec![0u32; n], vec![0u32; n]);
                    // SAFETY: as above.
                    let ua = unsafe { (set.fields)(&syms, &buf[..cut], &mut a) };
                    // SAFETY: the scalar set always runs.
                    let ub = unsafe { (SCALAR.fields)(&syms, &buf[..cut], &mut b) };
                    assert_eq!((ua, a), (ub, b), "{} truncated", set.isa);
                }
            }
        }
    }

    #[test]
    fn every_kernel_set_maps_rows_identically() {
        use crate::scans::map::UNMAPPED;
        let mut rng = Rng(11);
        for (case, &n) in [0usize, 1, 15, 16, 17, 100, 1000, 999, 333]
            .iter()
            .enumerate()
        {
            let genes = 3000usize;
            let map: Vec<u32> = (0..genes)
                .map(|_| {
                    if rng.next().is_multiple_of(5) {
                        UNMAPPED
                    } else {
                        (rng.next() % 20000) as u32
                    }
                })
                .collect();
            let mut ids: Vec<u32> = (0..n).map(|_| (rng.next() % genes as u64) as u32).collect();
            ids.sort();
            // Dictionaries in one vector, two vectors, and beyond.
            let k = [3usize, 16, 29, 300, 1, 32, 200, 17, 33][case];
            let dict: Vec<u32> = (0..k).map(|_| rng.next() as u32).collect();
            let r8: Vec<u8> = (0..n)
                .map(|_| (rng.next() % k.min(256) as u64) as u8)
                .collect();
            let r32: Vec<u32> = (0..n).map(|_| (rng.next() % k as u64) as u32).collect();
            let modes: Vec<u32> = (0..genes).map(|_| rng.next() as u32).collect();
            // Mode ranks: 0 (the column mode) or 1..=k.
            let m8: Vec<u8> = (0..n)
                .map(|_| {
                    if rng.next().is_multiple_of(3) {
                        (1 + rng.next() % k.min(255) as u64) as u8
                    } else {
                        0
                    }
                })
                .collect();
            let m32: Vec<u32> = (0..n)
                .map(|_| {
                    if rng.next().is_multiple_of(3) {
                        (1 + rng.next() % k as u64) as u32
                    } else {
                        0
                    }
                })
                .collect();
            let run = |set: &Kernels| {
                let (mut c8, mut v8, mut c32, mut v32) =
                    (vec![0u32; n], vec![0u32; n], vec![0u32; n], vec![0u32; n]);
                // SAFETY: ids index `map`, ranks index `dict`, outputs hold n; a supported set.
                let a = unsafe { (set.map_u8)(&ids, &map, &r8, &dict, &mut c8, &mut v8) };
                // SAFETY: as above.
                let b = unsafe { (set.map_u32)(&ids, &map, &r32, &dict, &mut c32, &mut v32) };
                let (mut d8, mut w8, mut d32, mut w32) =
                    (vec![0u32; n], vec![0u32; n], vec![0u32; n], vec![0u32; n]);
                // SAFETY: ids index `map` and `modes`, ranks are at most k; outputs hold n.
                let c =
                    unsafe { (set.map_modes_u8)(&ids, &map, &m8, &dict, &modes, &mut d8, &mut w8) };
                // SAFETY: as above.
                let d = unsafe {
                    (set.map_modes_u32)(&ids, &map, &m32, &dict, &modes, &mut d32, &mut w32)
                };
                let (mut cr, mut vr) = (vec![0u32; n], vec![0u32; n]);
                // Misaligned on purpose: raw values sit anywhere in a record.
                let raw: Vec<u8> = std::iter::once(0u8)
                    .chain(r32.iter().flat_map(|v| v.to_le_bytes()))
                    .collect();
                // SAFETY: as above, with one raw value per id.
                let e = unsafe { (set.map_raw32)(&ids, &map, &raw[1..], &mut cr, &mut vr) };
                let modes_out = (
                    d8[..c].to_vec(),
                    w8[..c].to_vec(),
                    d32[..d].to_vec(),
                    w32[..d].to_vec(),
                );
                (
                    c8[..a].to_vec(),
                    v8[..a].to_vec(),
                    c32[..b].to_vec(),
                    v32[..b].to_vec(),
                    modes_out,
                    (cr[..e].to_vec(), vr[..e].to_vec()),
                )
            };
            let want = run(&SCALAR);
            assert!(want.0.iter().all(|&c| c != UNMAPPED));
            // Chosen positions, including the last few (byte ranks near the end).
            let mut pos: Vec<u32> = (0..n as u32)
                .filter(|_| rng.next().is_multiple_of(2))
                .collect();
            if n > 0 {
                pos.extend([n as u32 - 1, (n as u32).saturating_sub(2), 0]);
            }
            let raw: Vec<u8> = std::iter::once(0u8)
                .chain(r32.iter().flat_map(|v| v.to_le_bytes()))
                .collect();
            let variants = [
                At::Const(dict[0]),
                At::Dict8(&r8, &dict),
                At::Dict32(&r32, &dict),
                At::Modes8(&m8, &dict, &modes),
                At::Modes32(&m32, &dict, &modes),
                At::ModesOne(&modes),
                At::Raw(&raw[1..], Src::W32, Conv::Bits),
            ];
            for v in variants {
                let at = |set: &Kernels| {
                    let (mut c, mut w) = (vec![0u32; pos.len()], vec![0u32; pos.len()]);
                    // SAFETY: positions are below n, ids index `map` and `modes`, ranks fit `dict`.
                    unsafe { (set.map_at)(&ids, &pos, &map, v, &mut c, &mut w) };
                    (c, w)
                };
                let want = at(&SCALAR);
                for set in all_supported() {
                    assert_eq!(at(&set), want, "{} map_at n={n}", set.isa);
                }
                // Scatter through an injective map, then with a row too short.
                let injective: Vec<u32> = (0..genes as u32).map(|g| if g % 5 == 0 { UNMAPPED } else { g * 7919 % 20000 }).collect();
                let scatter = |set: &Kernels, width: usize| {
                    let mut row = vec![0xdead_beefu32; width];
                    // SAFETY: ids index the map and modes, ranks fit `dict`, values cover every id.
                    let ok = unsafe { (set.scatter)(&ids, &injective, v, &mut row) };
                    (ok, if ok { row } else { Vec::new() })
                };
                let want = scatter(&SCALAR, 20000);
                assert!(want.0);
                for set in all_supported() {
                    assert_eq!(scatter(&set, 20000), want, "{} scatter n={n}", set.isa);
                    let mapped = ids.iter().any(|&id| injective[id as usize] != UNMAPPED && injective[id as usize] >= 100);
                    assert_eq!(scatter(&set, 100).0, !mapped, "{} short scatter n={n}", set.isa);
                }
            }
            for set in all_supported() {
                assert_eq!(run(&set), want, "{} n={n}", set.isa);
            }
        }
    }

    #[test]
    fn every_kernel_set_expands_every_width_identically() {
        let mut rng = Rng(77);
        for (log, w) in [1usize, 2, 4, 8].into_iter().enumerate() {
            for &n in &[0usize, 1, 7, 8, 9, 15, 16, 17, 100] {
                let k = 1 + (rng.next() % 256) as usize;
                let dict: Vec<u8> = (0..k * w + DICT_SLACK).map(|_| rng.next() as u8).collect();
                let syms: Vec<u8> = (0..n).map(|_| (rng.next() % k as u64) as u8).collect();
                let mut want = vec![0u8; n * w];
                // SAFETY: symbols index the dictionary, which carries the slack.
                unsafe { (SCALAR.expand[log])(&dict, &syms, &mut want) };
                let wide: Vec<u32> = syms.iter().map(|&s| s as u32).collect();
                for set in all_supported() {
                    let mut got = vec![0u8; n * w];
                    // SAFETY: as above, on a supported set.
                    unsafe { (set.expand[log])(&dict, &syms, &mut got) };
                    assert_eq!(got, want, "{} width {w} n {n}", set.isa);
                    let mut got = vec![0u8; n * w];
                    // SAFETY: as above.
                    unsafe { (set.expand32[log])(&dict, &wide, &mut got) };
                    assert_eq!(got, want, "{} wide width {w} n {n}", set.isa);
                }
            }
        }
    }
}
