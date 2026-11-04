use cellpack::Error;
use cellpack::format::*;
use cellpack::scans::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

#[derive(Clone, Copy, Debug)]
enum Values {
    /// Small counts: few distinct values per cell.
    Counts,
    /// log1p-normalized counts as f32 bit patterns.
    Normalized,
    /// Every value distinct.
    Continuous,
    /// Many distinct counts per cell, past the direct rank range.
    Wide,
}

/// An owned CSR matrix with skewed gene popularity and varied cell sizes,
/// including empty and fully dense cells.
struct Matrix {
    n_genes: u32,
    vw: usize,
    indptr: Vec<u64>,
    idx16: Vec<u16>,
    idx32: Vec<u32>,
    data: Vec<u8>,
}

impl Matrix {
    fn generate(seed: u64, rows: usize, n_genes: u32, vw: usize, kind: Values) -> Self {
        let mut rng = Rng(seed | 1);
        let mut indptr = vec![0u64];
        let (mut idx32, mut data) = (Vec::new(), Vec::new());
        let mask = if vw == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * vw)) - 1
        };
        for r in 0..rows {
            let density = match r % 9 {
                0 => 0.0,
                1 => 1.0,
                2 => 0.5,
                _ => rng.below(300) as f64 / 1000.0,
            };
            let scale = 1.0 + rng.below(50) as f32 / 10.0;
            for g in 0..n_genes {
                // Low genes are popular, as in real panels.
                let p = (density * (1.0 + 4.0 / (1.0 + g as f64 / 50.0))).min(1.0);
                if density == 1.0 || (rng.below(1_000_000) as f64) < p * 1_000_000.0 {
                    idx32.push(g);
                    let count = 1
                        + rng.below(3) * rng.below(3)
                        + if rng.below(40) == 0 {
                            rng.below(3000)
                        } else {
                            0
                        };
                    let v = match kind {
                        Values::Counts => count,
                        Values::Wide => 1 + rng.below(5000),
                        Values::Normalized => (count as f32 * scale).ln_1p().to_bits() as u64,
                        Values::Continuous => rng.next(),
                    } & mask;
                    data.extend_from_slice(&v.to_le_bytes()[..vw]);
                }
            }
            indptr.push(idx32.len() as u64);
        }
        let idx16 = if n_genes <= 1 << 16 {
            idx32.iter().map(|&g| g as u16).collect()
        } else {
            Vec::new()
        };
        Matrix {
            n_genes,
            vw,
            indptr,
            idx16,
            idx32,
            data,
        }
    }

    fn csr(&self, wide: bool) -> CsrRef<'_> {
        CsrRef {
            n_genes: self.n_genes,
            value_width: self.vw,
            indptr: &self.indptr,
            indices: if wide || self.idx16.is_empty() {
                Indices::U32(&self.idx32)
            } else {
                Indices::U16(&self.idx16)
            },
            data: &self.data,
        }
    }

    fn row(&self, r: usize) -> (&[u32], &[u8]) {
        let (a, b) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
        (&self.idx32[a..b], &self.data[a * self.vw..b * self.vw])
    }
}

fn options(renumber: bool, mu: f32, max_lanes: u8, lane_min: u32) -> EncodeOptions {
    EncodeOptions {
        renumber,
        thresholds: Thresholds {
            indices: mu,
            values: mu,
        },
        max_lanes,
        lane_min,
        column_modes: None,
        // These checks exercise explicit sparse ids and sparse value streams.
        layout: LayoutChoice::Sparse,
        values: ValueChoice::Auto,
    }
}

fn encode(m: &Matrix, rows: Rows, opt: &EncodeOptions) -> Vec<u8> {
    let src = m.csr(false);
    let mut out = vec![0u8; chunk_bound(src, rows, opt).unwrap()];
    let mut ws = vec![0u64; chunk_workspace_words(src, rows, opt).unwrap()];
    let len = encode_chunk(src, rows, opt, &mut out, &mut ws).unwrap();
    out.truncate(len);
    out
}

/// Checks every decode path of a chunk built from `selection` rows of `m`.
fn check_chunk(m: &Matrix, chunk: &[u8], selection: &[usize]) {
    let meta = parse_meta(chunk).unwrap();
    let tables = decode_tables(&meta).unwrap();
    let records = meta.records_of(chunk).unwrap();
    let vw = m.vw;
    let mut ws = vec![0u64; decode_workspace_words(&meta)];
    assert_eq!(meta.n_cells(), selection.len());
    let gene = |c: u32| tables.source_of_coded().map_or(c, |soc| soc[c as usize]);

    // Coded output: ids ascend in coded numbering, each with its value; as
    // (source gene, value) pairs it equals the input.
    let as_source = |ids: &[u32], vals: &[u8]| -> Vec<(u32, Vec<u8>)> {
        let mut v: Vec<(u32, Vec<u8>)> = ids
            .iter()
            .zip(vals.chunks_exact(vw))
            .map(|(&c, x)| (gene(c), x.to_vec()))
            .collect();
        v.sort();
        v
    };
    let want = |r: usize| -> Vec<(u32, Vec<u8>)> {
        let (gi, gv) = m.row(r);
        gi.iter()
            .zip(gv.chunks_exact(vw))
            .map(|(&g, x)| (g, x.to_vec()))
            .collect()
    };

    // Whole chunk to CSR, both id spaces.
    for space in [IdSpace::Source, IdSpace::Coded] {
        let nnz = selection_nnz(&meta, Rows::All).unwrap() as usize;
        let (mut ip, mut ix, mut vx) = (
            vec![0i64; selection.len() + 1],
            vec![0u32; nnz],
            vec![0u8; nnz * vw],
        );
        assert_eq!(
            decode_csr(
                &meta,
                &tables,
                records,
                Rows::All,
                space,
                &mut ip,
                &mut ix,
                &mut vx,
                &mut ws
            )
            .unwrap(),
            nnz as u64
        );
        for (i, &r) in selection.iter().enumerate() {
            let (gi, gv) = m.row(r);
            let (a, b) = (ip[i] as usize, ip[i + 1] as usize);
            assert!(ix[a..b].windows(2).all(|w| w[0] < w[1]), "ids must ascend");
            match space {
                IdSpace::Source => {
                    assert_eq!(&ix[a..b], gi, "cell {i}");
                    assert_eq!(&vx[a * vw..b * vw], gv, "cell {i}");
                }
                IdSpace::Coded => assert_eq!(
                    as_source(&ix[a..b], &vx[a * vw..b * vw]),
                    want(r),
                    "cell {i} coded"
                ),
            }
        }
    }

    // A shuffled selection with repeats, into i32 indptr and u64 indices.
    let n = selection.len();
    let pick: Vec<u32> = (0..2 * n)
        .map(|i| ((i * 7 + 3) % n.max(1)) as u32)
        .filter(|_| n > 0)
        .collect();
    let rows = Rows::List(&pick);
    let nnz = selection_nnz(&meta, rows).unwrap() as usize;
    let (mut ip, mut ix, mut vx) = (
        vec![0i32; pick.len() + 1],
        vec![0u64; nnz],
        vec![0u8; nnz * vw],
    );
    decode_csr(
        &meta,
        &tables,
        records,
        rows,
        IdSpace::Source,
        &mut ip,
        &mut ix,
        &mut vx,
        &mut ws,
    )
    .unwrap();
    for (i, &p) in pick.iter().enumerate() {
        let (gi, gv) = m.row(selection[p as usize]);
        let (a, b) = (ip[i] as usize, ip[i + 1] as usize);
        assert!(ix[a..b].iter().map(|&x| x as u32).eq(gi.iter().copied()));
        assert_eq!(&vx[a * vw..b * vw], gv);
    }

    // A range, densely.
    let (lo, hi) = (n / 3, n - n / 4);
    let rows = Rows::Range { start: lo, end: hi };
    let width = m.n_genes as usize * vw;
    let mut dense = vec![0xffu8; dense_len(&meta, rows).unwrap()];
    decode_dense(
        &meta,
        &tables,
        records,
        rows,
        IdSpace::Source,
        &mut dense,
        Fill::Zero,
        &mut ws,
    )
    .unwrap();
    for (i, &r) in selection[lo..hi].iter().enumerate() {
        let mut want = vec![0u8; width];
        let (gi, gv) = m.row(r);
        for (j, &g) in gi.iter().enumerate() {
            want[g as usize * vw..(g as usize + 1) * vw].copy_from_slice(&gv[j * vw..(j + 1) * vw]);
        }
        assert_eq!(
            &dense[i * width..(i + 1) * width],
            &want[..],
            "dense row {i}"
        );
    }

    // Record by record, as a loader holding only the metadata would.
    let meta_only = parse_meta(&chunk[..meta.meta_len()]).unwrap();
    let mut row = vec![0u8; width];
    for (i, &r) in selection.iter().enumerate() {
        let rec = meta_only.record(records, i).unwrap();
        let nnz = record_nnz(rec).unwrap();
        assert_eq!(nnz, meta_only.cell_nnz(i).unwrap());
        let (mut ix, mut vx) = (vec![0u16; nnz.max(1)], vec![0u8; nnz * vw]);
        let (gi, gv) = m.row(r);
        if m.n_genes <= 1 << 16 {
            assert_eq!(
                decode_record(
                    &meta_only,
                    &tables,
                    rec,
                    IdSpace::Coded,
                    &mut ix,
                    &mut vx,
                    &mut ws
                )
                .unwrap(),
                nnz
            );
            let ids: Vec<u32> = ix[..nnz].iter().map(|&c| c as u32).collect();
            assert_eq!(as_source(&ids, &vx), want(r));
        }
        // Parts agree with the materialized decode in both id spaces.
        for space in [IdSpace::Source, IdSpace::Coded] {
            let (mut ix, mut vx) = (vec![0u32; nnz], vec![0u8; nnz * vw]);
            decode_record(&meta_only, &tables, rec, space, &mut ix, &mut vx, &mut ws).unwrap();
            let p = decode_parts(&meta_only, &tables, rec, space, &mut ws).unwrap();
            assert_eq!(p.ids(), &ix[..]);
            let mut ex = vec![0u8; nnz * vw];
            p.expand_values(&mut ex).unwrap();
            assert_eq!(ex, vx);
            if let cellpack::scans::Values::Dict(d) = p.values() {
                assert!(
                    d.entries()
                        .chunks_exact(vw)
                        .collect::<Vec<_>>()
                        .windows(2)
                        .all(|w| w[0] != w[1])
                );
                let k = d.k();
                match d.ranks() {
                    Ranks::One => assert_eq!(k, 1),
                    Ranks::U8(r) => assert!(r.iter().all(|&x| (x as usize) < k)),
                    Ranks::U32(r) => assert!(r.iter().all(|&x| (x as usize) < k)),
                }
            }
        }
        decode_record_dense(
            &meta_only,
            &tables,
            rec,
            IdSpace::Coded,
            &mut row,
            Fill::Zero,
            &mut ws,
        )
        .unwrap();
        for (j, &g) in gi.iter().enumerate() {
            let c = tables.coded_of_source().map_or(g, |cos| cos[g as usize]) as usize;
            assert_eq!(&row[c * vw..(c + 1) * vw], &gv[j * vw..(j + 1) * vw]);
        }
    }
}

#[test]
fn every_configuration_roundtrips_exactly() {
    let shapes = [
        (300u32, 2usize, Values::Counts),
        (300, 4, Values::Normalized),
        (200, 1, Values::Counts),
        (250, 8, Values::Wide),
        (300, 4, Values::Continuous),
        (400, 2, Values::Wide),
    ];
    for (seed, &(genes, vw, kind)) in shapes.iter().enumerate() {
        let m = Matrix::generate(seed as u64 + 11, 60, genes, vw, kind);
        let all: Vec<usize> = (0..60).collect();
        for renumber in [false, true] {
            for mu in [0.0, DEFAULT_MU_VALUES, 1.0] {
                for (lanes, lane_min) in [
                    (1u8, 1u32),
                    (32, 1),
                    (32, 16),
                    (DEFAULT_MAX_LANES, DEFAULT_LANE_MIN),
                    (8, 4),
                ] {
                    let opt = options(renumber, mu, lanes, lane_min);
                    check_chunk(&m, &encode(&m, Rows::All, &opt), &all);
                }
            }
        }
    }
}

#[test]
fn chunks_take_any_row_selection() {
    let m = Matrix::generate(5, 80, 500, 4, Values::Counts);
    let opt = options(true, DEFAULT_MU_INDICES, DEFAULT_MAX_LANES, 8);
    let pick: Vec<u32> = vec![70, 3, 3, 41, 0, 79, 12];
    let chunk = encode(&m, Rows::List(&pick), &opt);
    check_chunk(
        &m,
        &chunk,
        &pick.iter().map(|&r| r as usize).collect::<Vec<_>>(),
    );
    let chunk = encode(&m, Rows::Range { start: 20, end: 50 }, &opt);
    check_chunk(&m, &chunk, &(20..50).collect::<Vec<_>>());
    let chunk = encode(&m, Rows::Range { start: 7, end: 7 }, &opt);
    check_chunk(&m, &chunk, &[]);
}

#[test]
fn wide_gene_spaces_use_four_byte_ids() {
    let m = Matrix::generate(9, 12, 70_000, 2, Values::Counts);
    let all: Vec<usize> = (0..12).collect();
    for renumber in [false, true] {
        for mu in [0.0, 1.0] {
            let chunk = encode(&m, Rows::All, &options(renumber, mu, 32, 32));
            check_chunk(&m, &chunk, &all);
            let meta = parse_meta(&chunk).unwrap();
            assert_eq!(meta.index_width(), 4);
            let tables = decode_tables(&meta).unwrap();
            let rec = meta.record(meta.records_of(&chunk).unwrap(), 1).unwrap();
            let mut ws = vec![0u64; decode_workspace_words(&meta)];
            let (mut ix, mut vx) = (vec![0u16; 70_000], vec![0u8; 140_000]);
            assert_eq!(
                decode_record(
                    &meta,
                    &tables,
                    rec,
                    IdSpace::Source,
                    &mut ix,
                    &mut vx,
                    &mut ws
                ),
                Err(Error::OutputTypeTooNarrow(69_999))
            );
        }
    }
}

#[test]
fn thresholds_choose_sections_independently() {
    let mode = |chunk: &[u8]| -> Vec<u8> {
        let meta = parse_meta(chunk).unwrap();
        let rec = meta.records_of(chunk).unwrap();
        (0..meta.n_cells())
            .filter_map(|i| meta.record(rec, i).unwrap().first().copied())
            .collect()
    };
    let counts = Matrix::generate(3, 40, 300, 4, Values::Counts);
    let all_raw = mode(&encode(&counts, Rows::All, &options(true, 0.0, 32, 16)));
    assert!(all_raw.iter().all(|&f| f == 0));
    // Tiny cells can stay raw even at mu = 1 (stream headers outweigh them).
    let coded = mode(&encode(&counts, Rows::All, &options(true, 1.0, 32, 16)));
    assert!(
        coded.iter().filter(|&&f| f == 0b0101).count() * 5 >= coded.len() * 4,
        "{coded:?}"
    );
    let split = EncodeOptions {
        thresholds: Thresholds {
            indices: 1.0,
            values: 0.0,
        },
        ..options(true, 1.0, 32, 16)
    };
    let modes = mode(&encode(&counts, Rows::All, &split));
    assert!(modes.iter().all(|&f| f >> 2 == 0));
    assert!(modes.iter().filter(|&&f| f == 0b0001).count() * 5 >= modes.len() * 4);
    // Distinct 4-byte values cost more than 0.6 of raw: values stay raw, indices still code.
    let cont = Matrix::generate(4, 40, 300, 4, Values::Continuous);
    let modes = mode(&encode(
        &cont,
        Rows::All,
        &options(true, DEFAULT_MU_VALUES, 32, 16),
    ));
    assert!(modes.iter().all(|&f| f >> 2 == 0));
    assert!(modes.iter().filter(|&&f| f & 3 == 1).count() > modes.len() / 2);
}

#[test]
fn coded_counts_compress_well() {
    let m = Matrix::generate(21, 200, 2000, 2, Values::Counts);
    let chunk = encode(
        &m,
        Rows::All,
        &options(true, DEFAULT_MU_VALUES, DEFAULT_MAX_LANES, DEFAULT_LANE_MIN),
    );
    let raw = m.idx32.len() * (2 + 2);
    assert!(chunk.len() * 3 < raw, "chunk {} vs raw {raw}", chunk.len());
}

#[test]
fn contract_violations_are_reported() {
    let m = Matrix::generate(2, 10, 100, 2, Values::Counts);
    let opt = options(true, DEFAULT_MU_VALUES, 32, 16);
    let src = m.csr(false);
    let bound = chunk_bound(src, Rows::All, &opt).unwrap();
    let words = chunk_workspace_words(src, Rows::All, &opt).unwrap();
    let mut out = vec![0u8; bound];
    let mut short = vec![0u64; words - 1];
    assert!(matches!(
        encode_chunk(src, Rows::All, &opt, &mut out, &mut short),
        Err(Error::WorkspaceTooSmall { .. })
    ));
    let mut ws = vec![0u64; words];
    let bad = EncodeOptions {
        max_lanes: 3,
        ..opt
    };
    assert!(matches!(
        encode_chunk(src, Rows::All, &bad, &mut out, &mut ws),
        Err(Error::InvalidOptions(_))
    ));
    let bad = EncodeOptions {
        thresholds: Thresholds {
            indices: 1.5,
            values: 0.6,
        },
        ..opt
    };
    assert!(matches!(
        encode_chunk(src, Rows::All, &bad, &mut out, &mut ws),
        Err(Error::InvalidOptions(_))
    ));
    assert!(matches!(
        encode_chunk(src, Rows::List(&[10]), &opt, &mut out, &mut ws),
        Err(Error::RowOutOfRange { .. })
    ));

    let unsorted = [3u16, 1];
    let vals = [1u8, 0, 1, 0];
    let ip = [0u64, 2];
    let bad = CsrRef {
        n_genes: 5,
        value_width: 2,
        indptr: &ip,
        indices: Indices::U16(&unsorted),
        data: &vals,
    };
    assert!(matches!(
        encode_chunk(bad, Rows::All, &opt, &mut out, &mut ws),
        Err(Error::InvalidInput(_))
    ));
    let beyond = [1u16, 5];
    let bad = CsrRef {
        indices: Indices::U16(&beyond),
        ..bad
    };
    assert!(matches!(
        encode_chunk(bad, Rows::All, &opt, &mut out, &mut ws),
        Err(Error::InvalidInput(_))
    ));

    let len = encode_chunk(src, Rows::All, &opt, &mut out, &mut ws).unwrap();
    let chunk = &out[..len];
    let meta = parse_meta(chunk).unwrap();
    let tables = decode_tables(&meta).unwrap();
    let records = meta.records_of(chunk).unwrap();
    let mut dws = vec![0u64; decode_workspace_words(&meta) - 1];
    let nnz = selection_nnz(&meta, Rows::All).unwrap() as usize;
    let (mut ip, mut ix, mut vx) = (vec![0u64; 11], vec![0u32; nnz], vec![0u8; nnz * 2]);
    assert!(matches!(
        decode_csr(
            &meta,
            &tables,
            records,
            Rows::All,
            IdSpace::Source,
            &mut ip,
            &mut ix,
            &mut vx,
            &mut dws
        ),
        Err(Error::WorkspaceTooSmall { .. })
    ));
    let mut dws = vec![0u64; decode_workspace_bound(100, 2)];
    let mut small = vec![0u32; nnz - 1];
    assert!(matches!(
        decode_csr(
            &meta,
            &tables,
            records,
            Rows::All,
            IdSpace::Source,
            &mut ip,
            &mut small,
            &mut vx,
            &mut dws
        ),
        Err(Error::OutputTooSmall { .. })
    ));
    let other = Matrix::generate(3, 4, 120, 2, Values::Counts);
    let other_chunk = encode(&other, Rows::All, &opt);
    let other_tables = decode_tables(&parse_meta(&other_chunk).unwrap()).unwrap();
    assert!(matches!(
        decode_csr(
            &meta,
            &other_tables,
            records,
            Rows::All,
            IdSpace::Source,
            &mut ip,
            &mut ix,
            &mut vx,
            &mut dws
        ),
        Err(Error::TablesMismatch)
    ));
}

#[test]
fn packed_chunks_decode_identically_and_read_minimal_pages() {
    for (seed, genes, vw, kind) in [
        (31u64, 400u32, 2usize, Values::Counts),
        (32, 300, 4, Values::Normalized),
        (33, 250, 4, Values::Continuous),
    ] {
        let m = Matrix::generate(seed, 70, genes, vw, kind);
        let all: Vec<usize> = (0..70).collect();
        for renumber in [false, true] {
            let chunk = encode(
                &m,
                Rows::All,
                &options(
                    renumber,
                    DEFAULT_MU_VALUES,
                    DEFAULT_MAX_LANES,
                    DEFAULT_LANE_MIN,
                ),
            );
            for page in [512u32, 4096] {
                let len = packed_len(&chunk, page).unwrap();
                let mut packed = vec![0xaau8; len + 3];
                assert_eq!(pack_chunk(&chunk, page, &mut packed).unwrap(), len);
                packed.truncate(len);
                let meta = parse_meta(&packed).unwrap();
                assert_eq!(meta.page_size(), Some(page));
                assert_eq!(meta.records_start() % page as usize, 0);
                assert_eq!(meta.chunk_len(), len as u64);
                for i in 0..meta.n_cells() {
                    let r = meta.record_range(i).unwrap();
                    if !r.is_empty() {
                        let (a, b) = (
                            meta.records_start() + r.start,
                            meta.records_start() + r.end - 1,
                        );
                        assert_eq!(
                            b / page as usize - a / page as usize + 1,
                            r.len().div_ceil(page as usize)
                        );
                    }
                }
                check_chunk(&m, &packed, &all);
                // Packing a packed chunk is stable in size.
                assert_eq!(packed_len(&packed, page).unwrap(), len);
            }
            assert!(packed_len(&chunk, 1000).is_err());
        }
    }
}

/// Splits `rec` over scattered pages starting `start` bytes into the first.
fn scatter_pages(rec: &[u8], page: usize, start: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut logical = vec![0xeeu8; start];
    logical.extend_from_slice(rec);
    let n = logical.len().div_ceil(page).max(1);
    logical.resize(n * page, 0xdd);
    // Physical frames in a shuffled order; `order[i]` is the frame of logical page i.
    let mut order: Vec<usize> = (0..n).collect();
    let mut x = seed | 1;
    for i in (1..n).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        order.swap(i, (x as usize) % (i + 1));
    }
    let mut frames = vec![Vec::new(); n];
    for (i, &f) in order.iter().enumerate() {
        frames[f] = logical[i * page..(i + 1) * page].to_vec();
    }
    (frames, order)
}

#[test]
fn paged_records_decode_like_contiguous_ones() {
    for (seed, genes, vw, kind) in [
        (41u64, 3000u32, 4usize, Values::Normalized),
        (42, 2500, 2, Values::Counts),
        (43, 600, 4, Values::Continuous),
    ] {
        let m = Matrix::generate(seed, 40, genes, vw, kind);
        for renumber in [false, true] {
            let chunk = encode(
                &m,
                Rows::All,
                &options(
                    renumber,
                    DEFAULT_MU_VALUES,
                    DEFAULT_MAX_LANES,
                    DEFAULT_LANE_MIN,
                ),
            );
            let meta = parse_meta(&chunk).unwrap();
            let tables = decode_tables(&meta).unwrap();
            let records = meta.records_of(&chunk).unwrap();
            let mut ws = vec![0u64; decode_workspace_words(&meta)];
            let mut ws2 = vec![0u64; decode_workspace_words(&meta)];
            let width = genes as usize * vw;
            for i in 0..meta.n_cells() {
                let rec = meta.record(records, i).unwrap();
                let nnz = meta.cell_nnz(i).unwrap();
                for (page, start) in [(512usize, 0usize), (512, 300), (4096, 4000), (4096, 17)] {
                    let (frames, order) = scatter_pages(rec, page, start, seed + i as u64);
                    let pages: Vec<&[u8]> = order.iter().map(|&f| &frames[f][..]).collect();
                    let paged = PagedRecord {
                        pages: &pages,
                        page_size: page,
                        start,
                        len: rec.len(),
                    };
                    for space in [IdSpace::Source, IdSpace::Coded] {
                        let (mut a, mut av) = (vec![0u32; nnz], vec![0u8; nnz * vw]);
                        let (mut b, mut bv) = (vec![0u32; nnz], vec![0u8; nnz * vw]);
                        decode_record(&meta, &tables, rec, space, &mut a, &mut av, &mut ws)
                            .unwrap();
                        decode_record_pages(
                            &meta, &tables, paged, space, &mut b, &mut bv, &mut ws2,
                        )
                        .unwrap();
                        assert_eq!((a, av), (b, bv), "cell {i} page {page} start {start}");
                        let (mut ra, mut rb) = (vec![0u8; width], vec![0xffu8; width]);
                        decode_record_dense(
                            &meta,
                            &tables,
                            rec,
                            space,
                            &mut ra,
                            Fill::Zero,
                            &mut ws,
                        )
                        .unwrap();
                        decode_record_dense_pages(
                            &meta,
                            &tables,
                            paged,
                            space,
                            &mut rb,
                            Fill::Zero,
                            &mut ws2,
                        )
                        .unwrap();
                        assert_eq!(ra, rb);
                        let pa =
                            decode_parts_pages(&meta, &tables, paged, space, &mut ws2).unwrap();
                        assert_eq!(pa.nnz(), nnz);
                    }
                    // Too few pages, or a start outside the first page, is an error.
                    if !rec.is_empty() {
                        let short = PagedRecord {
                            pages: &pages[..pages.len() - 1],
                            ..paged
                        };
                        let mut ix = vec![0u32; nnz];
                        let mut vx = vec![0u8; nnz * vw];
                        assert!(
                            decode_record_pages(
                                &meta,
                                &tables,
                                short,
                                IdSpace::Source,
                                &mut ix,
                                &mut vx,
                                &mut ws2
                            )
                            .is_err()
                        );
                        let outside = PagedRecord {
                            start: page,
                            ..paged
                        };
                        assert!(
                            decode_record_pages(
                                &meta,
                                &tables,
                                outside,
                                IdSpace::Source,
                                &mut ix,
                                &mut vx,
                                &mut ws2
                            )
                            .is_err()
                        );
                    }
                }
            }
        }
    }
}

/// Expected mapped row from the source-order decode: (column, value) pairs,
/// unmapped genes dropped, sorted by column.
fn mapped_pairs<S: CastInto<T>, T: Element + PartialOrd>(
    ids: &[u32],
    vals: &[u8],
    col_of_source: impl Fn(u32) -> u32,
) -> Vec<(u32, T)> {
    let mut v: Vec<(u32, T)> = ids
        .iter()
        .zip(vals.chunks_exact(S::WIDTH))
        .map(|(&g, b)| (col_of_source(g), S::from_le(b).cast()))
        .filter(|&(c, _)| c != UNMAPPED)
        .collect();
    v.sort_by_key(|a| a.0);
    v
}

fn check_mapping<S: CastInto<T>, T: Element + PartialOrd + std::fmt::Debug>(
    m: &Matrix,
    renumber: bool,
) {
    let chunk = encode(
        m,
        Rows::All,
        &options(
            renumber,
            DEFAULT_MU_VALUES,
            DEFAULT_MAX_LANES,
            DEFAULT_LANE_MIN,
        ),
    );
    let meta = parse_meta(&chunk).unwrap();
    let tables = decode_tables(&meta).unwrap();
    let records = meta.records_of(&chunk).unwrap();
    let mut ws = vec![0u64; decode_workspace_words(&meta)];
    let genes = m.n_genes as usize;
    // An injective panel: every fifth gene dropped, the rest scattered over a wider panel.
    let panel: Vec<u32> = (0..genes as u32)
        .map(|g| {
            if g % 5 == 3 {
                UNMAPPED
            } else {
                (g * 7919) % (3 * genes as u32)
            }
        })
        .collect();
    // The caller composes the panel with the chunk's gene order once.
    let coded_map: Vec<u32> = match tables.source_of_coded() {
        Some(soc) => soc.iter().map(|&s| panel[s as usize]).collect(),
        None => panel.clone(),
    };
    let mut dict_t: Vec<T> = Vec::new();
    for i in 0..meta.n_cells() {
        let rec = meta.record(records, i).unwrap();
        let nnz = meta.cell_nnz(i).unwrap();
        let (mut sx, mut sv) = (vec![0u32; nnz], vec![0u8; nnz * m.vw]);
        decode_record(
            &meta,
            &tables,
            rec,
            IdSpace::Source,
            &mut sx,
            &mut sv,
            &mut ws,
        )
        .unwrap();
        let want = mapped_pairs::<S, T>(&sx, &sv, |g| panel[g as usize]);
        let p = decode_parts(&meta, &tables, rec, IdSpace::Coded, &mut ws).unwrap();
        if let cellpack::scans::Values::Dict(d) = p.values() {
            dict_t.resize(d.k(), S::from_le(&[0u8; 8]).cast());
            d.cast_into::<S, T>(&mut dict_t).unwrap();
        }
        let (mut cols, mut vals) = (vec![0u32; nnz], vec![S::from_le(&[0u8; 8]).cast(); nnz]);
        let kept = map_parts::<S, T>(&p, &coded_map, &dict_t, &[], &mut cols, &mut vals).unwrap();
        let mut got: Vec<(u32, T)> = cols[..kept]
            .iter()
            .copied()
            .zip(vals[..kept].iter().copied())
            .collect();
        got.sort_by_key(|a| a.0);
        assert_eq!(got, want, "cell {i} renumber {renumber}");
        // By position: every other element, kept where mapped.
        let positions: Vec<u32> = (0..nnz as u32).step_by(2).collect();
        let (mut pc, mut pv) = (
            vec![0u32; positions.len()],
            vec![S::from_le(&[0u8; 8]).cast(); positions.len()],
        );
        map_parts_at::<S, T>(&p, &positions, &coded_map, &dict_t, &[], &mut pc, &mut pv).unwrap();
        let mut sub: Vec<(u32, T)> = pc
            .iter()
            .copied()
            .zip(pv.iter().copied())
            .filter(|a| a.0 != UNMAPPED)
            .collect();
        sub.sort_by_key(|a| a.0);
        assert!(
            sub.iter().all(|e| want.contains(e)),
            "cell {i} positions renumber {renumber}"
        );
        // Scattered into a dense row over the wider panel.
        let zero: T = S::from_le(&[0u8; 8]).cast();
        let mut row = vec![zero; 3 * genes];
        map_dense::<S, T>(&p, &coded_map, &dict_t, &[], &mut row, Fill::Zero).unwrap();
        let mut dense_want = vec![zero; 3 * genes];
        for &(c, v) in &want {
            dense_want[c as usize] = v;
        }
        assert_eq!(row, dense_want, "cell {i} dense renumber {renumber}");
    }
}

/// Every allowed conversion of the loader's contract, over dictionary,
/// raw and dense records, against the scalar source-order decode.
#[test]
fn every_allowed_conversion_maps_like_the_scalar_reference() {
    macro_rules! pairs {
        ($($s:ty, $w:literal => $($t:ty),*;)*) => {$($(
            for kind in [Values::Counts, Values::Wide, Values::Continuous] {
                check_mapping::<$s, $t>(&Matrix::generate(61 + $w as u64, 40, 700, $w, kind), true);
            }
        )*)*};
    }
    pairs! {
        u8, 1 => u8, u16, u32, u64, i16, i32, i64, f32, f64;
        i8, 1 => i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
        u16, 2 => u16, u32, u64, i32, i64, f32, f64;
        i16, 2 => i16, u16, i32, u32, i64, u64, f32, f64;
        u32, 4 => u32, u64, i64, f32, f64;
        i32, 4 => i32, u32, i64, u64, f32, f64;
        u64, 8 => u64, f32, f64;
        i64, 8 => i64, u64, f32, f64;
    }
    check_mapping::<f32, f32>(&Matrix::generate(71, 40, 700, 4, Values::Normalized), true);
    check_mapping::<f32, f64>(&Matrix::generate(72, 40, 700, 4, Values::Normalized), true);
}

#[test]
fn parts_map_to_caller_columns_and_types() {
    for renumber in [false, true] {
        check_mapping::<u16, f32>(&Matrix::generate(51, 50, 2000, 2, Values::Counts), renumber);
        check_mapping::<f32, f32>(
            &Matrix::generate(52, 50, 2000, 4, Values::Normalized),
            renumber,
        );
        check_mapping::<u32, f64>(&Matrix::generate(53, 50, 1500, 4, Values::Wide), renumber);
        check_mapping::<u16, u16>(&Matrix::generate(54, 50, 1500, 2, Values::Wide), renumber);
        check_mapping::<u32, i32>(
            &Matrix::generate(55, 50, 800, 4, Values::Continuous),
            renumber,
        );
    }
}

#[test]
fn workspace_contents_carry_nothing_between_chunks() {
    // Chunks of different shapes encoded in one dirty workspace match
    // their encodings in fresh ones.
    let chunks = [
        (Matrix::generate(11, 300, 5000, 2, Values::Counts), true),
        (Matrix::generate(12, 200, 700, 4, Values::Normalized), true),
        (Matrix::generate(13, 150, 3000, 2, Values::Counts), false),
        (Matrix::generate(14, 100, 400, 4, Values::Continuous), true),
    ];
    let mut ws = vec![0xa5a5_a5a5_a5a5_a5a5u64; 1 << 20];
    for (m, renumber) in &chunks {
        let mut opt = options(*renumber, DEFAULT_MU_VALUES, DEFAULT_MAX_LANES, DEFAULT_LANE_MIN);
        opt.column_modes = Some(DEFAULT_MODE_SHARE);
        let src = m.csr(false);
        let words = chunk_workspace_words(src, Rows::All, &opt).unwrap();
        assert!(words <= ws.len());
        let mut out = vec![0u8; chunk_bound(src, Rows::All, &opt).unwrap()];
        let len = encode_chunk(src, Rows::All, &opt, &mut out, &mut ws).unwrap();
        assert_eq!(&out[..len], &encode(m, Rows::All, &opt)[..]);
    }
}
