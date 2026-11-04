//! Record coding: the rANS streams of a prepared row and its record.
//!
//! The index section is the gap stream when it beats the raw ids by the
//! index threshold; the value section is the smaller of the dictionary and
//! mode codings when it beats the raw values by the value threshold (or the
//! forced coding when no larger than raw). Each coding's pool (dictionary
//! stream, dictionary escape bits, rank escape bits) is complete before its
//! rank stream is coded, so the pool's leading bytes ride in the rank
//! stream's initial states, as the gap escape bits ride in the gap stream's.

use super::input::{Index, load, store, with_indices, with_width};
use super::model::cell_of;
use super::prep::{CodingPrep, SparsePrep, ValuesPrep, order};
use super::scratch::{CodeWs, PrepWs};
use crate::format::{self, DICT, GAP, IDX_DENSE, IDX_GAP, IDX_RAW, Meta, RANK, VAL_DICT, VAL_MODE, VAL_RAW};
use crate::scans::rans;
use crate::scans::tables::EncodeTables;
use crate::scans::types::{EncodeOptions, Indices, RowRef, ValueChoice};

/// A chunk's metadata, tables and options, for coding its records.
pub(super) struct Coder<'a> {
    /// Metadata of the chunk being written.
    pub(super) meta: &'a Meta<'a>,
    /// Its tables: the models and the gene order records follow.
    pub(super) tables: &'a EncodeTables,
    /// The options records are coded under.
    pub(super) opt: &'a EncodeOptions,
}

#[inline]
fn worth(coded: usize, raw: usize, mu: f32) -> bool {
    coded as f64 <= mu as f64 * raw as f64
}

/// A coded value section held in the workspace: the dictionary stream and
/// escape bits, the rank escape bits and rank stream, and the pool bytes
/// carried by the rank stream.
#[derive(Clone, Copy)]
struct Coded {
    k: usize,
    dr: usize,
    db: usize,
    rb: usize,
    rr: usize,
    carried: usize,
    ranked: bool,
    size: usize,
}

/// A value coding whose dictionary stream is coded, with the head of its
/// pool for the rank stream.
#[derive(Clone, Copy)]
struct Begun {
    k: usize,
    dr: usize,
    db: usize,
    rb: usize,
    carried: usize,
    head: [u8; 64],
    ranked: bool,
}

impl Begun {
    /// The coded section once its rank stream (if ranked) has `rr` bytes.
    fn finish(&self, rr: usize) -> Coded {
        let lens = if self.ranked {
            format::varint_len(rr as u64) + format::varint_len(self.rb as u64) + rr
        } else {
            0
        };
        let size = format::varint_len(self.k as u64)
            + format::varint_len(self.dr as u64)
            + format::varint_len(self.db as u64)
            + lens
            + (self.dr + self.db + self.rb - self.carried);
        Coded { k: self.k, dr: self.dr, db: self.db, rb: self.rb, rr, carried: self.carried, ranked: self.ranked, size }
    }
}

/// The coded streams of a record: the gap stream's length and both value
/// codings (each `None` when unavailable).
struct Streams {
    gap: Option<usize>,
    dict: Option<Coded>,
    mode: Option<Coded>,
    mode_prep: Option<CodingPrep>,
}

/// Appends the concatenation of `parts` from byte `skip` on at `out[*at..]`.
fn put_parts(out: &mut [u8], at: &mut usize, parts: &[&[u8]], mut skip: usize) {
    for part in parts {
        let from = skip.min(part.len());
        skip -= from;
        let rest = &part[from..];
        out[*at..*at + rest.len()].copy_from_slice(rest);
        *at += rest.len();
    }
}

/// The first `n` (at most 64) bytes of the concatenation of `parts`.
fn head(parts: &[&[u8]], n: usize) -> [u8; 64] {
    let mut h = [0u8; 64];
    let mut got = 0;
    for part in parts {
        let take = (n - got).min(part.len());
        h[got..got + take].copy_from_slice(&part[..take]);
        got += take;
    }
    h
}

/// Gene ids as `iw`-byte little-endian integers.
fn write_raw_indices(idx: Indices, iw: usize, out: &mut [u8]) {
    with_indices!(idx, |s| {
        for (o, &i) in out.chunks_exact_mut(iw).zip(s) {
            o.copy_from_slice(&i.id().to_le_bytes()[..iw]);
        }
    })
}

impl Coder<'_> {
    /// Lanes of a stream of `count` symbols for this chunk's options.
    fn lanes(&self, count: usize) -> usize {
        rans::lanes_for(count, self.meta.max_lanes(), self.meta.lane_min())
    }

    /// Codes the dictionary stream of a prepared coding of `n` elements and
    /// the head of its pool, for its rank stream. None when the dictionary
    /// exceeds the chunk maximum or has a symbol outside the model.
    fn begin(&self, c: &CodingPrep, bytes: &[u8], n: usize, words: &mut [u16], dict_stream: &mut [u8]) -> Option<Begun> {
        let k = c.k;
        if k > self.meta.max_k() {
            return None;
        }
        let dict_bits = c.dict.bits.of(bytes);
        let db = dict_bits.len();
        let dr = rans::encode(self.tables.kernels, &self.tables.models[DICT], c.dict.syms.of(bytes), self.lanes(k), &[], words, dict_stream)?;
        let (rb, carried, head) = match c.ranks {
            Some(r) => {
                let rank_bits = r.bits.of(bytes);
                let rb = rank_bits.len();
                let carried = (dr + db + rb).min(rans::payload_capacity(self.lanes(n)));
                (rb, carried, head(&[&dict_stream[..dr], dict_bits, rank_bits], carried))
            }
            None => (0, 0, [0; 64]),
        };
        Some(Begun { k, dr, db, rb, carried, head, ranked: c.ranks.is_some() })
    }

    /// Codes every stream of a prepared record: the dictionary streams of
    /// its value codings, then the gap stream (`gap`: symbols and payload)
    /// and both rank streams, which have one symbol per element, together.
    fn code(&self, vp: &ValuesPrep, bytes: &[u8], gap: Option<(&[u8], &[u8])>, cws: &mut CodeWs) -> Streams {
        let n = vp.n;
        let values = self.opt.values;
        let models = &self.tables.models;
        let dict = matches!(values, ValueChoice::Auto | ValueChoice::Dict)
            .then(|| self.begin(&vp.dict, bytes, n, cws.words_rank, cws.dict_stream))
            .flatten();
        let mode = self
            .meta
            .column_modes()
            .filter(|_| matches!(values, ValueChoice::Auto | ValueChoice::Modes))
            .map(|_| vp.mode.expect("records are prepared against the chunk's column modes").coding);
        let mode_begun = mode.filter(|m| m.k > 0).and_then(|m| self.begin(&m, bytes, n, cws.words_mode, cws.mode_dict_stream));
        let rank_syms = |c: &CodingPrep| c.ranks.map_or(&[][..], |r| r.syms.of(bytes));
        let gap_job = gap.map(|(syms, payload)| rans::Stream {
            model: &models[GAP],
            syms,
            payload,
            words: &mut *cws.words_gap,
            out: &mut *cws.gap_stream,
        });
        let dict_job = dict.as_ref().filter(|b| b.ranked).map(|b| rans::Stream {
            model: &models[RANK],
            syms: rank_syms(&vp.dict),
            payload: &b.head[..b.carried],
            words: &mut *cws.words_rank,
            out: &mut *cws.rank_stream,
        });
        let mode_job = mode_begun.as_ref().zip(mode).map(|(b, m)| rans::Stream {
            model: &models[RANK],
            syms: rank_syms(&m),
            payload: &b.head[..b.carried],
            words: &mut *cws.words_mode,
            out: &mut *cws.mode_rank_stream,
        });
        let [gap, rank, mode_rank] = rans::encode_many(self.tables.kernels, self.lanes(n), [gap_job, dict_job, mode_job]);
        let dict = dict.and_then(|b| if b.ranked { rank.map(|rr| b.finish(rr)) } else { Some(b.finish(0)) });
        let mode_coded = mode.and_then(|m| {
            if m.k == 0 {
                Some(Coded { k: 0, dr: 0, db: 0, rb: 0, rr: 0, carried: 0, ranked: false, size: format::varint_len(0) })
            } else {
                mode_begun.zip(mode_rank).map(|(b, rr)| b.finish(rr))
            }
        });
        Streams { gap, dict, mode: mode_coded, mode_prep: mode }
    }

    /// Writes the value section of a coded record at `out[*at..]` and
    /// returns the value mode; `raw` writes the raw values in record order.
    #[allow(clippy::too_many_arguments)]
    fn write_values(
        &self,
        vp: &ValuesPrep,
        bytes: &[u8],
        st: &Streams,
        raw: impl FnOnce(&mut [u8]),
        cws: &CodeWs,
        out: &mut [u8],
        at: &mut usize,
    ) -> u8 {
        let raw_len = vp.n * self.meta.value_width();
        let mu = if self.opt.values == ValueChoice::Auto { self.opt.thresholds.values } else { 1.0 };
        let dict_ok = st.dict.filter(|c| worth(c.size, raw_len, mu));
        let mode_ok = st.mode.filter(|c| worth(c.size, raw_len, mu));
        let put = |out: &mut [u8], at: &mut usize, c: &Coded, prep: &CodingPrep, dict_stream: &[u8], rank_stream: &[u8]| {
            let rank_bits = prep.ranks.map_or(&[][..], |r| r.bits.of(bytes));
            format::put_varint(out, at, c.dr as u64);
            format::put_varint(out, at, c.db as u64);
            if c.ranked {
                format::put_varint(out, at, c.rr as u64);
                format::put_varint(out, at, c.rb as u64);
                put_parts(out, at, &[&rank_stream[..c.rr]], 0);
            }
            put_parts(out, at, &[&dict_stream[..c.dr], prep.dict.bits.of(bytes), rank_bits], c.carried);
        };
        match (dict_ok, mode_ok) {
            (_, Some(m)) if dict_ok.is_none_or(|d| m.size < d.size) => {
                format::put_varint(out, at, m.k as u64);
                if m.k > 0 {
                    let prep = st.mode_prep.expect("a mode coding has its preparation");
                    put(out, at, &m, &prep, cws.mode_dict_stream, cws.mode_rank_stream);
                }
                VAL_MODE
            }
            (Some(d), _) => {
                format::put_varint(out, at, d.k as u64);
                put(out, at, &d, &vp.dict, cws.dict_stream, cws.rank_stream);
                VAL_DICT
            }
            _ => {
                raw(&mut out[*at..*at + raw_len]);
                *at += raw_len;
                VAL_RAW
            }
        }
    }

    /// A sparse record of a prepared row into `out`; returns its length.
    /// Raw sections take the row's elements again from `row`.
    pub(super) fn write_sparse(&self, prep: &SparsePrep, bytes: &[u8], row: RowRef, ws: &mut PrepWs, cws: &mut CodeWs, out: &mut [u8]) -> usize {
        let (vw, iw) = (self.meta.value_width(), self.meta.index_width());
        let n = prep.values.n;
        let cos = self.tables.coded_of_source();
        // Byte 0 holds the two section modes; the element count follows it as
        // a varint, and the sections follow in the order they are written.
        let mut at = 1;
        format::put_varint(out, &mut at, n as u64);

        // As much of the gap escape bits as the stream states can carry rides
        // in them; the rest is stored after the stream.
        let gap_bits = prep.gaps.bits.of(bytes);
        let gb = gap_bits.len();
        let carried = gb.min(rans::payload_capacity(self.lanes(n)));
        let st = self.code(&prep.values, bytes, Some((prep.gaps.syms.of(bytes), &gap_bits[..carried])), cws);
        let idx_coded = st.gap.filter(|&r| {
            let size = format::varint_len(r as u64) + format::varint_len(gb as u64) + r + (gb - carried);
            worth(size, n * iw, self.opt.thresholds.indices)
        });
        let idx_mode = match idx_coded {
            Some(r) => {
                format::put_varint(out, &mut at, r as u64);
                format::put_varint(out, &mut at, gb as u64);
                put_parts(out, &mut at, &[&cws.gap_stream[..r]], 0);
                put_parts(out, &mut at, &[gap_bits], carried);
                IDX_GAP
            }
            None => {
                let cell = cell_of(row, vw, ws.sp_idx, ws.sp_val);
                let dst = &mut out[at..at + n * iw];
                match cos {
                    Some(_) => {
                        with_indices!(cell.indices, |idx| order(idx, cos, &mut ws.order));
                        write_raw_indices(Indices::U32(&ws.order.coded[..n]), iw, dst);
                    }
                    None => write_raw_indices(cell.indices, iw, dst),
                }
                at += n * iw;
                IDX_RAW
            }
        };
        // Raw values in coded order.
        let raw = |dst: &mut [u8]| {
            let cell = cell_of(row, vw, ws.sp_idx, ws.sp_val);
            match cos {
                Some(_) => {
                    with_indices!(cell.indices, |idx| order(idx, cos, &mut ws.order));
                    let perm = &ws.order.perm[..n];
                    with_width!(vw, |W| {
                        for (t, &p) in perm.iter().enumerate() {
                            // SAFETY: `perm` is a permutation of the `n`
                            // elements, each with a value in both buffers.
                            unsafe { store::<W>(dst, t, load::<W>(cell.values, p as usize)) };
                        }
                    })
                }
                None => dst.copy_from_slice(cell.values),
            }
        };
        let val_mode = self.write_values(&prep.values, bytes, &st, raw, cws, out, &mut at);
        out[0] = idx_mode | (val_mode << 2);
        at
    }

    /// A dense record (every gene, `values` in coded order, prepared as
    /// `vp`) of a row with `nnz` non-zeros into `out`; returns its length.
    pub(super) fn write_dense(&self, nnz: usize, vp: &ValuesPrep, bytes: &[u8], values: &[u8], cws: &mut CodeWs, out: &mut [u8]) -> usize {
        // Byte 0 holds the section modes; every gene follows as a value, so
        // only the non-zero count is stored with it.
        let mut at = 1;
        format::put_varint(out, &mut at, nnz as u64);
        let st = self.code(vp, bytes, None, cws);
        let val_mode = self.write_values(vp, bytes, &st, |dst| dst.copy_from_slice(values), cws, out, &mut at);
        out[0] = IDX_DENSE | (val_mode << 2);
        at
    }
}
