//! Ratio and single-thread speed on a raw CSR stream.
//!
//! Reads `[n_rows, n_cols, index_width, value_code, nnz]` (u64 LE), indptr
//! (u64 LE), indices, values from stdin. Arguments, all optional:
//! `chunk_rows max_lanes lane_min renumber(0|1) mu_indices mu_values mode_share`.
//! `value_code` selects the byte width: 0/3 -> 2, 1/4/6 -> 4, 2/5/7 -> 8.
//! These input tags are separate from the ProcessFn ABI's dtype codes.
//! Encodes the rows into chunks of `chunk_rows` and reports the ratio
//! (raw CSR bytes over chunk bytes) and per-cell encode and decode times for
//! each decode path, with the kernel set chosen by `CELLPACK_KERNELS`.

use std::io::Read;
use std::time::Instant;

use cellpack::format::*;
use cellpack::scans::*;

fn arg<T: std::str::FromStr>(i: usize, default: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|a| a.parse().ok())
        .unwrap_or(default)
}

/// Mean seconds per call of `f`, repeated for at least 0.3 s.
fn time(mut f: impl FnMut()) -> f64 {
    f();
    let (t, mut reps) = (Instant::now(), 0u32);
    while reps == 0 || t.elapsed().as_secs_f64() < 0.3 {
        f();
        reps += 1;
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    if std::env::args().skip(1).any(|a| a == "--help" || a == "-h") {
        println!("Usage: codec [chunk_rows max_lanes lane_min renumber mu_indices mu_values mode_share] < input.csr\n\
            Input: five little-endian u64 words (rows, cols, index_width, value_code, nnz),\n\
            followed by u64 indptr, unsigned 2/4-byte indices and value bytes.\n\
            value_code widths: 0/3=2, 1/4/6=4, 2/5/7=8 bytes. mode_share=0 disables modes.\n\
            See crates/cellpack/examples/README.md for the stream format.");
        return;
    }
    let chunk_rows: usize = arg(1, 1 << 30);
    assert!(chunk_rows > 0, "chunk_rows must be positive");
    let opt = EncodeOptions {
        max_lanes: arg(2, DEFAULT_MAX_LANES),
        lane_min: arg(3, DEFAULT_LANE_MIN),
        renumber: arg::<u8>(4, 1) == 1,
        thresholds: Thresholds {
            indices: arg(5, DEFAULT_MU_INDICES),
            values: arg(6, DEFAULT_MU_VALUES),
        },
        column_modes: Some(arg(7, DEFAULT_MODE_SHARE)).filter(|&s| s > 0.0),
        layout: LayoutChoice::Auto,
        values: ValueChoice::Auto,
    };
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();
    assert!(input.len() >= 40, "expected a raw CSR stream on stdin; use --help for its format");
    let word = |i: usize| u64::from_le_bytes(input[8 * i..8 * i + 8].try_into().unwrap());
    let (rows, cols, iw, code, nnz) = (
        word(0) as usize,
        word(1) as u32,
        word(2) as usize,
        word(3),
        word(4) as usize,
    );
    let cols_n = cols;
    assert!(matches!(iw, 2 | 4), "index_width must be 2 or 4 bytes");
    assert!(code < 8, "value_code must be in 0..8");
    let vw = [2usize, 4, 8, 2, 4, 8, 4, 8][code as usize];
    let mut at = 40;
    let indptr: Vec<u64> = (0..=rows).map(|i| word(5 + i)).collect();
    at += 8 * (rows + 1);
    let ib = &input[at..at + nnz * iw];
    let data = &input[at + nnz * iw..at + nnz * (iw + vw)];
    let idx16: Vec<u16> = if iw == 2 {
        ib.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    } else {
        Vec::new()
    };
    let idx32: Vec<u32> = if iw == 4 {
        ib.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        Vec::new()
    };
    let src = CsrRef {
        n_genes: cols,
        value_width: vw,
        indptr: &indptr,
        indices: if iw == 2 {
            Indices::U16(&idx16)
        } else {
            Indices::U32(&idx32)
        },
        data,
    };
    let raw = nnz * (iw + vw);
    let cells = (0..rows)
        .filter(|&r| indptr[r + 1] > indptr[r])
        .count()
        .max(1) as f64;

    let mut chunks = Vec::new();
    let mut enc_s = 0.0;
    let mut r0 = 0;
    while r0 < rows {
        let rsel = Rows::Range {
            start: r0,
            end: (r0 + chunk_rows).min(rows),
        };
        let mut out = vec![0u8; chunk_bound(src, rsel, &opt).unwrap()];
        let mut ws = vec![0u64; chunk_workspace_words(src, rsel, &opt).unwrap()];
        let t = Instant::now();
        let len = encode_chunk(src, rsel, &opt, &mut out, &mut ws).unwrap();
        enc_s += t.elapsed().as_secs_f64();
        out.truncate(len);
        chunks.push(out);
        r0 = (r0 + chunk_rows).min(rows);
    }
    let total: usize = chunks.iter().map(Vec::len).sum();
    let meta_bytes: usize = chunks
        .iter()
        .map(|c| parse_meta(c).unwrap().meta_len())
        .sum();
    let (mut idx_coded, mut val_coded, mut nonempty) = (0, 0, 0);
    for c in &chunks {
        let meta = parse_meta(c).unwrap();
        let rec = meta.records_of(c).unwrap();
        for i in 0..meta.n_cells() {
            if let Some(&f) = meta.record(rec, i).unwrap().first() {
                nonempty += 1;
                idx_coded += (f & 3 == 1) as usize;
                val_coded += (f >> 2 == 1) as usize;
            }
        }
    }

    // Page reads per non-empty cell: records as encoded (contiguous, region
    // start taken as page-aligned) versus packed to 4 KiB pages.
    let (mut pages_flat, mut pages_packed, mut packed_total) = (0u64, 0u64, 0usize);
    for c in &chunks {
        let meta = parse_meta(c).unwrap();
        let base = meta.records_start() as u64;
        for i in 0..meta.n_cells() {
            let r = meta.record_range(i).unwrap();
            if !r.is_empty() {
                let (a, b) = (base + r.start as u64, base + r.end as u64 - 1);
                pages_flat += b / 4096 - a / 4096 + 1;
                pages_packed += (r.len() as u64).div_ceil(4096);
            }
        }
        packed_total += packed_len(c, 4096).unwrap();
    }
    let cells_ne = nonempty.max(1) as f64;

    let parsed: Vec<_> = chunks.iter().map(|c| (parse_meta(c).unwrap(), c)).collect();
    let tables: Vec<_> = parsed
        .iter()
        .map(|(m, _)| decode_tables(m).unwrap())
        .collect();
    let words = parsed
        .iter()
        .map(|(m, _)| decode_workspace_words(m))
        .max()
        .unwrap_or(0);
    let mut ws = vec![0u64; words];
    let most = parsed.iter().map(|(m, _)| m.max_nnz()).max().unwrap_or(0);
    let (mut ix, mut vx) = (vec![0u32; most], vec![0u8; most * vw]);
    let mut record_pass = |space: IdSpace| {
        time(|| {
            for ((meta, c), t) in parsed.iter().zip(&tables) {
                let rec = meta.records_of(c).unwrap();
                for i in 0..meta.n_cells() {
                    decode_record(
                        meta,
                        t,
                        meta.record(rec, i).unwrap(),
                        space,
                        &mut ix,
                        &mut vx,
                        &mut ws,
                    )
                    .unwrap();
                }
            }
        })
    };
    let rec_source = record_pass(IdSpace::Source);
    let rec_coded = record_pass(IdSpace::Coded);
    let (mut ip, mut cx, mut cv) = (vec![0u64; rows + 1], vec![0u32; nnz], vec![0u8; nnz * vw]);
    let chunk_csr = time(|| {
        for ((meta, c), t) in parsed.iter().zip(&tables) {
            let n = meta.n_cells();
            let len = meta.nnz_total() as usize;
            decode_csr(
                meta,
                t,
                meta.records_of(c).unwrap(),
                Rows::All,
                IdSpace::Source,
                &mut ip[..n + 1],
                &mut cx[..len],
                &mut cv[..len * vw],
                &mut ws,
            )
            .unwrap();
        }
    });
    // Parts in coded space: what a loader worker consumes.
    let parts = time(|| {
        for ((meta, c), t) in parsed.iter().zip(&tables) {
            let rec = meta.records_of(c).unwrap();
            for i in 0..meta.n_cells() {
                let p = decode_parts(
                    meta,
                    t,
                    meta.record(rec, i).unwrap(),
                    IdSpace::Coded,
                    &mut ws,
                )
                .unwrap();
                std::hint::black_box(p.nnz());
            }
        }
    });
    // Parts mapped to f32 through a gene panel composed with each chunk's gene
    // order (every tenth gene dropped): the loader's per-cell work before sampling.
    let mut mcols = vec![0u32; most];
    let mut fvals = vec![0f32; most];
    let mut dict_f = vec![
        0f32;
        parsed
            .iter()
            .map(|(m, _)| m.max_k())
            .max()
            .unwrap_or(0)
            .max(1)
    ];
    let maps: Vec<Vec<u32>> = tables
        .iter()
        .map(|t| {
            let panel = |g: u32| {
                if g % 10 == 7 {
                    UNMAPPED
                } else {
                    g.wrapping_mul(2_654_435_761) % (2 * cols_n)
                }
            };
            match t.source_of_coded() {
                Some(soc) => soc.iter().map(|&s| panel(s)).collect(),
                None => (0..cols_n).map(panel).collect(),
            }
        })
        .collect();
    // Column modes converted once per chunk, like the gene panel.
    let modes_f: Vec<Vec<f32>> = parsed
        .iter()
        .map(|(m, _)| {
            let mut out = vec![0f32; m.column_modes().map_or(0, |b| b.len() / vw)];
            if let Some(b) = m.column_modes() {
                if vw == 2 {
                    cast_values::<u16, f32>(b, &mut out).unwrap()
                } else {
                    cast_values::<f32, f32>(b, &mut out).unwrap()
                }
            }
            out
        })
        .collect();
    let mapped = time(|| {
        for ((((meta, c), t), map), modes) in parsed.iter().zip(&tables).zip(&maps).zip(&modes_f) {
            let rec = meta.records_of(c).unwrap();
            for i in 0..meta.n_cells() {
                let p = decode_parts(
                    meta,
                    t,
                    meta.record(rec, i).unwrap(),
                    IdSpace::Coded,
                    &mut ws,
                )
                .unwrap();
                match p.values() {
                    Values::Dict(d) => {
                        if vw == 2 {
                            d.cast_into::<u16, f32>(&mut dict_f).unwrap()
                        } else {
                            d.cast_into::<f32, f32>(&mut dict_f).unwrap()
                        }
                    }
                    Values::Modes(d) => {
                        if vw == 2 {
                            d.cast_into::<u16, f32>(&mut dict_f).unwrap()
                        } else {
                            d.cast_into::<f32, f32>(&mut dict_f).unwrap()
                        }
                    }
                    Values::Raw(_) => {}
                }
                let kept = if vw == 2 {
                    map_parts::<u16, f32>(&p, map, &dict_f, modes, &mut mcols, &mut fvals).unwrap()
                } else {
                    map_parts::<f32, f32>(&p, map, &dict_f, modes, &mut mcols, &mut fvals).unwrap()
                };
                std::hint::black_box(kept);
            }
        }
    });
    // Paged input: each chunk packed to 4 KiB pages, its pages scattered in
    // memory; records decoded from page arrays versus the packed chunk directly.
    let packed: Vec<Vec<u8>> = chunks
        .iter()
        .map(|c| {
            let mut out = vec![0u8; packed_len(c, 4096).unwrap()];
            pack_chunk(c, 4096, &mut out).unwrap();
            out
        })
        .collect();
    let mut flat_s = 0.0;
    let mut paged_s = 0.0;
    for (pc, t) in packed.iter().zip(&tables) {
        let meta = parse_meta(pc).unwrap();
        let base = meta.records_start();
        let n_pages = (pc.len() - base).div_ceil(4096);
        let mut region = pc[base..].to_vec();
        region.resize(n_pages * 4096, 0);
        // Frame f holds logical page perm[f]; frames are spaced apart.
        let mut perm: Vec<usize> = (0..n_pages).collect();
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for i in (1..n_pages).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            perm.swap(i, (x as usize) % (i + 1));
        }
        let mut frames = vec![0u8; 2 * 4096 * n_pages];
        let mut where_ = vec![0usize; n_pages];
        for (f, &lp) in perm.iter().enumerate() {
            frames[2 * 4096 * f..2 * 4096 * f + 4096]
                .copy_from_slice(&region[lp * 4096..lp * 4096 + 4096]);
            where_[lp] = 2 * 4096 * f;
        }
        let page_refs: Vec<&[u8]> = (0..n_pages)
            .map(|lp| &frames[where_[lp]..where_[lp] + 4096])
            .collect();
        let records = meta.records_of(pc).unwrap();
        let spans: Vec<(usize, usize, usize)> = (0..meta.n_cells())
            .map(|i| {
                let r = meta.record_range(i).unwrap();
                (r.start / 4096, r.start % 4096, r.len())
            })
            .collect();
        flat_s += time(|| {
            for i in 0..meta.n_cells() {
                decode_record(
                    &meta,
                    t,
                    meta.record(records, i).unwrap(),
                    IdSpace::Coded,
                    &mut ix,
                    &mut vx,
                    &mut ws,
                )
                .unwrap();
            }
        });
        paged_s += time(|| {
            for &(first, start, len) in &spans {
                let last = (first * 4096 + start + len.max(1) - 1) / 4096;
                let rec = PagedRecord {
                    pages: &page_refs[first..=last.max(first)],
                    page_size: 4096,
                    start,
                    len,
                };
                decode_record_pages(&meta, t, rec, IdSpace::Coded, &mut ix, &mut vx, &mut ws)
                    .unwrap();
            }
        });
    }
    let mut row = vec![0u8; cols as usize * vw];
    let dense = time(|| {
        for ((meta, c), t) in parsed.iter().zip(&tables) {
            let rec = meta.records_of(c).unwrap();
            for i in 0..meta.n_cells() {
                decode_record_dense(
                    meta,
                    t,
                    meta.record(rec, i).unwrap(),
                    IdSpace::Source,
                    &mut row,
                    Fill::Zero,
                    &mut ws,
                )
                .unwrap();
            }
        }
    });
    let us = |s: f64| s / cells * 1e6;
    println!(
        "kernels={} rows={rows} genes={cols} vw={vw} nnz/cell={:.0} chunk={} lanes={}/{} renumber={} mu={}/{} | ratio={:.3} meta={:.2}% \
         idx_coded={:.1}% val_coded={:.1}% | KB/cell={:.2} pages/cell flat={:.3} packed={:.3} packed_size=+{:.2}% | enc={:.2} rec_src={:.2} rec_coded={:.2} chunk_csr={:.2} dense={:.2} parts={:.2} mapped={:.2} flat={:.2} paged={:.2} us/cell",
        kernel_set(),
        nnz as f64 / cells,
        chunk_rows.min(rows),
        opt.max_lanes,
        opt.lane_min,
        opt.renumber as u8,
        opt.thresholds.indices,
        opt.thresholds.values,
        raw as f64 / total as f64,
        100.0 * meta_bytes as f64 / total as f64,
        100.0 * idx_coded as f64 / nonempty.max(1) as f64,
        100.0 * val_coded as f64 / nonempty.max(1) as f64,
        total as f64 / cells_ne / 1024.0,
        pages_flat as f64 / cells_ne,
        pages_packed as f64 / cells_ne,
        100.0 * (packed_total as f64 / total as f64 - 1.0),
        us(enc_s),
        us(rec_source),
        us(rec_coded),
        us(chunk_csr),
        us(dense),
        us(parts),
        us(mapped),
        us(flat_s),
        us(paged_s),
    );
}
