//! Integers to byte symbols.
//!
//! Values below [`DIRECT`] are their own symbol. Values in `[DIRECT, 256)`
//! share symbol `DIRECT` with their offset in 6 escape bits. A value with bit
//! length `b` in 9..=64 becomes symbol `DIRECT + b - 8`, and its `b - 1`
//! bits below the leading one go to escape bits. Every value is
//! `BASE[s] | escape bits`. Symbols above [`MAX_SYMBOL`] never occur, which
//! metadata validation enforces on the tables. `256 - DIRECT` is a power of
//! two, so the first bucket's offsets take whole bits, and dictionaries of
//! up to 192 entries rank without escapes.

/// Values below this are their own symbol; wider ones use a bucket.
pub(crate) const DIRECT: u64 = 192;
/// Last symbol a value of any width can take (bit length 64).
pub(crate) const MAX_SYMBOL: usize = DIRECT as usize + 56;
/// Largest symbol of an integer below 2^32 (gaps and ranks): buckets up to
/// bit length 32.
pub(crate) const MAX_SYMBOL_32: u8 = DIRECT as u8 + 24;
/// Escape bits of the first bucket, `[DIRECT, 256)`.
pub(crate) const FIRST_BITS: u32 = (256 - DIRECT).trailing_zeros();
const _: () = assert!((256 - DIRECT).is_power_of_two());

/// `(symbol, escape bits, escape bit count)`.
#[inline(always)]
pub(crate) fn split(v: u64) -> (u8, u64, u32) {
    if v < DIRECT {
        (v as u8, 0, 0)
    } else if v < 256 {
        (DIRECT as u8, v - DIRECT, FIRST_BITS)
    } else {
        // Bucket `len`: its base is the leading bit, and the `len - 1` bits
        // below it are the escape bits. Lengths 9..=64 map to symbols past
        // the first bucket.
        let len = 64 - v.leading_zeros();
        (
            (DIRECT as u32 + len - 8) as u8,
            v ^ (1 << (len - 1)),
            len - 1,
        )
    }
}

/// Per symbol, its escape bit count; zero above [`MAX_SYMBOL`].
pub(crate) static EXTRA: [u8; 256] = {
    let mut t = [0u8; 256];
    t[DIRECT as usize] = FIRST_BITS as u8;
    let mut s = DIRECT as usize + 1;
    while s <= MAX_SYMBOL {
        t[s] = (s - DIRECT as usize + 7) as u8;
        s += 1;
    }
    t
};

/// Per symbol, its value with zero escape bits; zero above [`MAX_SYMBOL`].
pub(crate) static BASE: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut s = 0;
    while s <= MAX_SYMBOL {
        t[s] = if s <= DIRECT as usize {
            s as u64
        } else {
            1 << (s - DIRECT as usize + 7)
        };
        s += 1;
    }
    t
};

/// Escape bit count of a symbol.
#[inline(always)]
pub(crate) fn extra_bits(sym: u8) -> u32 {
    EXTRA[sym as usize] as u32
}

/// Inverse of [`split`]; `bits` holds `extra_bits(sym)` bits.
#[inline(always)]
pub(crate) fn join(sym: u8, bits: u64) -> u64 {
    BASE[sym as usize] | bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_join_cover_every_bucket_edge() {
        let mut values = vec![
            0,
            1,
            DIRECT - 1,
            DIRECT,
            255,
            256,
            257,
            511,
            512,
            u32::MAX as u64,
            1 << 32,
            u64::MAX,
        ];
        values.extend((0..64).map(|b| 1u64 << b).flat_map(|v| [v - 1, v, v + 1]));
        for v in values {
            let (s, bits, nb) = split(v);
            assert!(s as usize <= MAX_SYMBOL);
            assert_eq!(nb, extra_bits(s));
            assert!(nb == 64 || bits >> nb == 0);
            assert_eq!(join(s, bits), v, "{v}");
            if v <= u32::MAX as u64 {
                assert!(s <= MAX_SYMBOL_32);
            }
        }
    }
}
