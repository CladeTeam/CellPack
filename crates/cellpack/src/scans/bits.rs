//! LSB-first escape bit streams.
//!
//! A symbol's escape bits follow each other in stream order, least
//! significant bit first, with no alignment between them. The reader keeps
//! an accumulator topped up to at least 56 bits, above the 32 one read
//! takes: it loads 8 bytes at once while they lie inside the buffer and
//! otherwise reads bytes one at a time, past the end as zero, which valid
//! streams do because readers refill ahead of need. [`exact`] is the
//! consumption check: a stream that decoded correctly leaves fewer than 8
//! unconsumed bits (the padding of its final byte).
//!
//! The writer serves tests only; the encoders write the same layout four
//! bytes at a time (see the escape-bit writer of `encode::prep`).

/// Writes into a caller buffer; running out of room is reported by `finish`.
#[cfg(test)]
pub(crate) struct BitWriter<'a> {
    buf: &'a mut [u8],
    len: usize,
    acc: u64,
    n: u32,
}

#[cfg(test)]
impl<'a> BitWriter<'a> {
    /// A writer filling `buf` from its first byte.
    pub(crate) fn new(buf: &'a mut [u8]) -> Self {
        BitWriter {
            buf,
            len: 0,
            acc: 0,
            n: 0,
        }
    }

    /// Appends the low `nb` bits of `v` (`nb <= 64`, `v < 2^nb`).
    #[inline(always)]
    pub(crate) fn put(&mut self, v: u64, nb: u32) {
        if nb > 32 {
            self.put32(v & 0xffff_ffff, 32);
            self.put32(v >> 32, nb - 32);
        } else if nb > 0 {
            self.put32(v, nb);
        }
    }

    #[inline(always)]
    fn put32(&mut self, v: u64, nb: u32) {
        self.acc |= v << self.n;
        self.n += nb;
        while self.n >= 8 {
            self.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    #[inline(always)]
    fn push(&mut self, b: u8) {
        if let Some(slot) = self.buf.get_mut(self.len) {
            *slot = b;
        }
        self.len += 1;
    }

    /// Byte length of the stream, or `None` if it did not fit.
    pub(crate) fn finish(mut self) -> Option<usize> {
        if self.n > 0 {
            self.push(self.acc as u8);
        }
        (self.len <= self.buf.len()).then_some(self.len)
    }
}

/// Reads one stream of escape bits, LSB-first (see the module docs). Reads
/// past the end yield zero bits, so a reader may refill ahead of need;
/// `finish_ok` checks that the stream was consumed exactly.
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl<'a> BitReader<'a> {
    /// A reader at the first bit of `buf`.
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        BitReader {
            buf,
            pos: 0,
            acc: 0,
            n: 0,
        }
    }

    /// A reader positioned `bit` bits into `buf`.
    pub(crate) fn at(buf: &'a [u8], bit: u64) -> Self {
        let mut r = BitReader {
            buf,
            pos: (bit / 8) as usize,
            acc: 0,
            n: 0,
        };
        r.take((bit % 8) as u32);
        r
    }

    /// Bits consumed so far.
    pub(crate) fn consumed(&self) -> u64 {
        self.pos as u64 * 8 - self.n as u64
    }

    /// Tops the accumulator up to at least 56 bits. The final bytes of a
    /// stream are read one at a time, and bytes past its end read as zero:
    /// readers refill ahead of need, so this happens for valid streams too.
    #[inline(always)]
    fn refill(&mut self) {
        let len = self.buf.len();
        if self.pos + 8 <= len {
            // SAFETY: 8 bytes at `pos` lie inside the buffer.
            let w = u64::from_le_bytes(unsafe { self.buf.as_ptr().add(self.pos).cast::<[u8; 8]>().read_unaligned() });
            self.acc |= w << self.n;
            let take = (63 - self.n) >> 3;
            self.pos += take as usize;
            self.n += take * 8;
        } else {
            while self.n <= 56 {
                // SAFETY: the read happens only below the buffer length.
                let b = if self.pos < len { unsafe { *self.buf.get_unchecked(self.pos) } } else { 0 };
                self.acc |= (b as u64) << self.n;
                self.pos += 1;
                self.n += 8;
            }
        }
    }

    /// Next `nb` bits, `nb <= 32`.
    #[inline(always)]
    fn get32(&mut self, nb: u32) -> u64 {
        if self.n < nb {
            self.refill();
        }
        let v = self.acc & ((1u64 << nb) - 1);
        self.acc >>= nb;
        self.n -= nb;
        v
    }

    /// Next `nb` bits, `nb <= 32`; refills on a fixed threshold rather than
    /// on `nb`, so a run of escape-free symbols does not branch on their
    /// widths.
    #[inline(always)]
    pub(crate) fn take(&mut self, nb: u32) -> u64 {
        if self.n < 32 {
            self.refill();
        }
        let v = self.acc & ((1u64 << nb) - 1);
        self.acc >>= nb;
        self.n -= nb;
        v
    }

    /// Next `nb` bits, `nb <= 64`.
    #[inline(always)]
    pub(crate) fn get(&mut self, nb: u32) -> u64 {
        if nb > 32 {
            let lo = self.get32(32);
            lo | (self.get32(nb - 32) << 32)
        } else {
            self.get32(nb)
        }
    }

    /// Whether exactly the stream was consumed, up to the final byte's padding.
    pub(crate) fn finish_ok(&self) -> bool {
        exact(self.consumed(), self.buf.len())
    }
}

/// Whether `consumed` bits cover a `len`-byte stream up to its final byte's padding.
#[inline]
pub(crate) fn exact(consumed: u64, len: usize) -> bool {
    let total = len as u64 * 8;
    consumed <= total && total - consumed < 8
}
