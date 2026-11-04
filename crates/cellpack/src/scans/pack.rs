//! Page packing: records placed so each reads its minimum number of pages,
//! with as little padding as possible.
//!
//! A record of `s` bytes needs at least `k = ceil(s / P)` pages, and reads
//! exactly `k` when its start offset `o` satisfies `o mod P <= k * P - s`.
//! Writing `s = (k - 1) * P + f` with `0 < f <= P`, the whole pages never
//! affect that condition; only the remainders `f` do. Records laid end to end
//! from a page boundary all read their minimum as long as their remainders
//! sum to at most `P`. Packing is therefore bin packing of the remainders
//! into bins of size `P`, each bin starting on a page boundary: fewer bins
//! means less padding. Best-fit decreasing over byte-sized remainders runs
//! in `O(n + P)` with buckets and a bitmap of non-empty bucket sizes.

use crate::error::{Error, Result};
use crate::format::{self, Header, Layout, MAX_PAGE_LOG2, MIN_PAGE_LOG2, Meta, parse_meta};

/// Record offsets within the records region and its length, for records of
/// the given lengths packed to `page` bytes.
fn plan(lengths: &[u32], page: usize) -> Result<(Vec<u64>, u64)> {
    let n = lengths.len();
    let rem = |len: u32| (len as usize - 1) % page + 1;
    // Records by remainder, largest first (counting sort).
    let mut count = vec![0u32; page + 2];
    for &len in lengths.iter().filter(|&&l| l > 0) {
        count[rem(len)] += 1;
    }
    let mut start = vec![0u32; page + 2];
    let mut acc = 0;
    for f in (1..=page).rev() {
        start[f] = acc;
        acc += count[f];
    }
    let mut order = vec![0u32; acc as usize];
    for (i, &len) in lengths.iter().enumerate().filter(|(_, l)| **l > 0) {
        let f = rem(len);
        order[start[f] as usize] = i as u32;
        start[f] += 1;
    }

    // Best fit: the bin with the least room that still fits. Bins are kept
    // in stacks by remaining room, with a bitmap of non-empty rooms.
    let mut by_room: Vec<Vec<u32>> = vec![Vec::new(); page];
    let mut nonempty = vec![0u64; page.div_ceil(64)];
    let mut bins: Vec<Vec<u32>> = Vec::new();
    let mut rooms: Vec<usize> = Vec::new();
    for &i in &order {
        let f = rem(lengths[i as usize]);
        let fit = {
            let (mut w, mut found) = (f / 64, None);
            let mut word = nonempty.get(w).map_or(0, |&x| x & (!0u64 << (f % 64)));
            while w < nonempty.len() {
                if word != 0 {
                    found = Some(w * 64 + word.trailing_zeros() as usize);
                    break;
                }
                w += 1;
                word = nonempty.get(w).copied().unwrap_or(0);
            }
            found
        };
        let b = match fit {
            Some(room) => {
                let b = by_room[room].pop().unwrap();
                if by_room[room].is_empty() {
                    nonempty[room / 64] &= !(1 << (room % 64));
                }
                b as usize
            }
            None => {
                bins.push(Vec::new());
                rooms.push(page);
                bins.len() - 1
            }
        };
        bins[b].push(i);
        rooms[b] -= f;
        let room = rooms[b];
        if room > 0 {
            by_room[room].push(b as u32);
            nonempty[room / 64] |= 1 << (room % 64);
        }
    }

    let mut offsets = vec![0u64; n];
    let mut cursor = 0u64;
    for bin in &bins {
        cursor = cursor
            .checked_next_multiple_of(page as u64)
            .ok_or(Error::InvalidInput("packed chunk too large"))?;
        for &i in bin {
            offsets[i as usize] = cursor;
            cursor = cursor
                .checked_add(lengths[i as usize] as u64)
                .ok_or(Error::InvalidInput("packed chunk too large"))?;
        }
    }
    Ok((offsets, cursor))
}

/// `log2` of a page size, checked against the supported powers of two.
fn page_log2(page: u32) -> Result<u32> {
    let log = page.trailing_zeros();
    if !page.is_power_of_two() || !(MIN_PAGE_LOG2..=MAX_PAGE_LOG2).contains(&log) {
        return Err(Error::InvalidOptions(
            "page size must be a power of two in [512, 65536]",
        ));
    }
    Ok(log)
}

/// Record length of each cell, which packing orders by remainder.
fn lengths(meta: &Meta) -> Result<Vec<u32>> {
    (0..meta.n_cells())
        .map(|i| meta.record_range(i).map(|r| r.len() as u32))
        .collect()
}

/// Bytes of `chunk` once packed to `page`-byte pages.
pub fn packed_len(chunk: &[u8], page: u32) -> Result<usize> {
    let log = page_log2(page)?;
    let meta = parse_meta(chunk)?;
    let (_, records_len) = plan(&lengths(&meta)?, page as usize)?;
    let start = format::records_start(meta.meta_len(), log)
        .ok_or(Error::InvalidInput("chunk too large"))?;
    (start as u64)
        .checked_add(records_len)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(Error::InvalidInput("chunk too large"))
}

/// Rewrites `chunk` with its records packed to `page`-byte pages (see the
/// module docs) into `out`, which must hold [`packed_len`] bytes. Every
/// record then reads `ceil(len / page)` pages once the chunk is stored at a
/// page-aligned position. Decoding is unchanged. Returns the packed length.
pub fn pack_chunk(chunk: &[u8], page: u32, out: &mut [u8]) -> Result<usize> {
    let log = page_log2(page)?;
    let meta = parse_meta(chunk)?;
    let records = meta.records_of(chunk)?;
    let lens = lengths(&meta)?;
    let (offsets, records_len) = plan(&lens, page as usize)?;
    let start = format::records_start(meta.meta_len(), log)
        .ok_or(Error::InvalidInput("chunk too large"))?;
    let total = (start as u64)
        .checked_add(records_len)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(Error::InvalidInput("chunk too large"))?;
    if out.len() < total {
        return Err(Error::OutputTooSmall {
            need: total,
            have: out.len(),
        });
    }
    let meta_len = meta.meta_len();
    out[..meta_len].copy_from_slice(&chunk[..meta_len]);
    out[meta_len..total].fill(0);
    for (i, &off) in offsets.iter().enumerate() {
        let src = &records[meta.record_range(i)?];
        let at = start + off as usize;
        out[at..at + src.len()].copy_from_slice(src);
    }
    let modes_len = meta.column_modes().map_or(0, <[u8]>::len);
    let layout = Layout::new(meta.n_cells(), meta.n_dims(), meta.renumbered(), modes_len)
        .ok_or(Error::CorruptMeta("chunk too large"))?;
    let extents: Vec<(u64, u32)> = offsets.iter().zip(&lens).map(|(&o, &l)| (o, l)).collect();
    format::write_extents(&mut out[..meta_len], &layout, &extents);
    Header::rewrite_records(&mut out[..meta_len], log as u8, records_len);
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::plan;

    #[test]
    fn every_record_reads_its_minimum_pages_with_little_padding() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for page in [512usize, 4096] {
            let lengths: Vec<u32> = (0..5000)
                .map(|i| {
                    if i % 13 == 0 {
                        0
                    } else {
                        1 + (next() % (3 * page as u64)) as u32
                    }
                })
                .collect();
            let (offsets, total) = plan(&lengths, page).unwrap();
            let mut spans: Vec<(u64, u64)> = Vec::new();
            for (&o, &l) in offsets.iter().zip(&lengths) {
                if l == 0 {
                    continue;
                }
                let (first, last) = (o / page as u64, (o + l as u64 - 1) / page as u64);
                assert_eq!(
                    last - first + 1,
                    (l as u64).div_ceil(page as u64),
                    "record reads extra pages"
                );
                assert!(o + l as u64 <= total);
                spans.push((o, o + l as u64));
            }
            spans.sort();
            assert!(
                spans.windows(2).all(|w| w[0].1 <= w[1].0),
                "records overlap"
            );
            let used: u64 = lengths.iter().map(|&l| l as u64).sum();
            let whole: u64 = lengths
                .iter()
                .filter(|&&l| l > 0)
                .map(|&l| ((l as u64 - 1) / page as u64) * page as u64)
                .sum();
            let lower = whole + (used - whole).div_ceil(page as u64) * page as u64;
            assert!(
                total <= lower + lower / 50 + 2 * page as u64,
                "padding {} over the bin lower bound {lower}",
                total - lower
            );
        }
    }
}
