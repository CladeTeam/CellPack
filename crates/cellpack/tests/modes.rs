//! Column modes: chunks whose values repeat along genes rather than within
//! cells store each gene's mode and code only the values that differ.

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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Nearly dense f32 rows: each gene has a baseline value most cells
    /// share, plus distinct values elsewhere (cell-flat, gene-structured).
    Baseline,
    /// Small counts: per-cell dictionaries already capture them.
    Counts,
    /// Every value distinct.
    Unique,
}

struct Matrix {
    n_genes: u32,
    vw: usize,
    indptr: Vec<u64>,
    idx: Vec<u32>,
    data: Vec<u8>,
}

impl Matrix {
    fn generate(seed: u64, rows: usize, n_genes: u32, kind: Kind) -> Self {
        let mut rng = Rng(seed | 1);
        let base: Vec<u32> = (0..n_genes)
            .map(|_| (-(rng.below(1000) as f32) / 997.0).to_bits())
            .collect();
        let (mut indptr, mut idx, mut data) = (vec![0u64], Vec::new(), Vec::new());
        let vw = if kind == Kind::Counts { 2 } else { 4 };
        for r in 0..rows {
            let density = match r % 7 {
                0 => 0.0,
                1 => 1.0,
                _ => 0.6 + rng.below(400) as f64 / 1000.0,
            };
            for g in 0..n_genes {
                if density < 1.0 && (rng.below(1000) as f64) >= density * 1000.0 {
                    continue;
                }
                idx.push(g);
                match kind {
                    Kind::Baseline => {
                        let v = if rng.below(100) < 93 {
                            base[g as usize]
                        } else {
                            (rng.next() as u32 >> 2) | 0x3f00_0000
                        };
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                    Kind::Counts => {
                        data.extend_from_slice(&(1 + rng.below(5) as u16).to_le_bytes())
                    }
                    Kind::Unique => data.extend_from_slice(&(rng.next() as u32).to_le_bytes()),
                }
            }
            indptr.push(idx.len() as u64);
        }
        Matrix {
            n_genes,
            vw,
            indptr,
            idx,
            data,
        }
    }

    fn csr(&self) -> CsrRef<'_> {
        CsrRef {
            n_genes: self.n_genes,
            value_width: self.vw,
            indptr: &self.indptr,
            indices: Indices::U32(&self.idx),
            data: &self.data,
        }
    }

    fn row(&self, r: usize) -> (&[u32], &[u8]) {
        let (a, b) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
        (&self.idx[a..b], &self.data[a * self.vw..b * self.vw])
    }
}

fn options(renumber: bool, modes: Option<f32>) -> EncodeOptions {
    EncodeOptions {
        renumber,
        thresholds: Thresholds {
            indices: DEFAULT_MU_INDICES,
            values: DEFAULT_MU_VALUES,
        },
        max_lanes: DEFAULT_MAX_LANES,
        lane_min: DEFAULT_LANE_MIN,
        column_modes: modes,
        layout: LayoutChoice::Auto,
        values: ValueChoice::Auto,
    }
}

fn encode(m: &Matrix, opt: &EncodeOptions) -> Vec<u8> {
    let src = m.csr();
    let mut out = vec![0u8; chunk_bound(src, Rows::All, opt).unwrap()];
    let mut ws = vec![0u64; chunk_workspace_words(src, Rows::All, opt).unwrap()];
    let n = encode_chunk(src, Rows::All, opt, &mut out, &mut ws).unwrap();
    out.truncate(n);
    out
}

/// Every decode path reproduces the matrix exactly.
fn check_roundtrip(m: &Matrix, chunk: &[u8]) {
    let meta = parse_meta(chunk).unwrap();
    let tables = decode_tables(&meta).unwrap();
    let records = meta.records_of(chunk).unwrap();
    let mut ws = vec![0u64; decode_workspace_words(&meta)];
    let vw = m.vw;
    let rows = m.indptr.len() - 1;
    let total = meta.nnz_total() as usize;
    let (mut ip, mut ix, mut vx) = (
        vec![0u64; rows + 1],
        vec![0u32; total],
        vec![0u8; total * vw],
    );
    decode_csr(
        &meta,
        &tables,
        records,
        Rows::All,
        IdSpace::Source,
        &mut ip,
        &mut ix,
        &mut vx,
        &mut ws,
    )
    .unwrap();
    assert_eq!(ip, m.indptr);
    assert_eq!(ix, m.idx);
    assert_eq!(vx, m.data);

    let width = m.n_genes as usize * vw;
    let mut dense = vec![0u8; rows * width];
    decode_dense(
        &meta,
        &tables,
        records,
        Rows::All,
        IdSpace::Source,
        &mut dense,
        Fill::Zero,
        &mut ws,
    )
    .unwrap();
    let mut packed = vec![0u8; packed_len(chunk, 4096).unwrap()];
    pack_chunk(chunk, 4096, &mut packed).unwrap();
    let pmeta = parse_meta(&packed).unwrap();
    let pages: Vec<&[u8]> = packed[pmeta.records_start()..].chunks(4096).collect();
    let (mut rix, mut rvx) = (vec![0u32; meta.max_nnz()], vec![0u8; meta.max_nnz() * vw]);
    for r in 0..rows {
        let (want_ix, want_v) = m.row(r);
        let mut want_row = vec![0u8; width];
        for (t, &g) in want_ix.iter().enumerate() {
            want_row[g as usize * vw..(g as usize + 1) * vw]
                .copy_from_slice(&want_v[t * vw..(t + 1) * vw]);
        }
        assert_eq!(
            &dense[r * width..(r + 1) * width],
            &want_row[..],
            "dense row {r}"
        );

        // Coded-space parts, expanded and mapped through an identity panel
        // composed with the gene order; source columns recover the row.
        let rec = meta.record(records, r).unwrap();
        let p = decode_parts(&meta, &tables, rec, IdSpace::Coded, &mut ws).unwrap();
        let mut expanded = vec![0u8; p.elements() * vw];
        p.expand_values(&mut expanded).unwrap();
        let soc: Vec<u32> = tables
            .source_of_coded()
            .map_or_else(|| (0..m.n_genes).collect(), <[u32]>::to_vec);
        let mut got_row = vec![0u8; width];
        for (t, &c) in p.ids().iter().enumerate() {
            let g = soc[c as usize] as usize;
            got_row[g * vw..(g + 1) * vw].copy_from_slice(&expanded[t * vw..(t + 1) * vw]);
        }
        assert_eq!(got_row, want_row, "parts row {r}");
        if vw == 4 {
            let mut dict = vec![0u32; meta.max_k().max(1)];
            match p.values() {
                cellpack::scans::Values::Dict(d) => d.cast_into::<u32, u32>(&mut dict).unwrap(),
                cellpack::scans::Values::Modes(d) => d.cast_into::<u32, u32>(&mut dict).unwrap(),
                cellpack::scans::Values::Raw(_) => {}
            }
            let mut modes = vec![0u32; meta.column_modes().map_or(0, |b| b.len() / 4)];
            if let Some(b) = meta.column_modes() {
                cast_values::<u32, u32>(b, &mut modes).unwrap();
            }
            // Any record scatters into a dense row; a dense record maps
            // only that way.
            let mut scattered = vec![0u32; m.n_genes as usize];
            map_dense::<u32, u32>(&p, &soc, &dict, &modes, &mut scattered, Fill::Zero).unwrap();
            let scattered: Vec<u8> = scattered.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(scattered, want_row, "dense-mapped row {r}");
            if p.layout() == RecordLayout::Dense {
                let (mut cols, mut vals) = (vec![0u32; p.elements()], vec![0u32; p.elements()]);
                assert!(
                    map_parts::<u32, u32>(&p, &soc, &dict, &modes, &mut cols, &mut vals).is_err()
                );
                assert!(
                    map_parts_at::<u32, u32>(&p, &[0], &soc, &dict, &modes, &mut cols, &mut vals)
                        .is_err()
                );
            } else {
                let (mut cols, mut vals) = (vec![0u32; p.nnz()], vec![0u32; p.nnz()]);
                let kept =
                    map_parts::<u32, u32>(&p, &soc, &dict, &modes, &mut cols, &mut vals).unwrap();
                let mut mapped_row = vec![0u8; width];
                for (&c, &v) in cols[..kept].iter().zip(&vals[..kept]) {
                    mapped_row[c as usize * 4..c as usize * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
                assert_eq!(mapped_row, want_row, "mapped row {r}");
                // Every element by position, reversed, matches the mapped row.
                let positions: Vec<u32> = (0..p.nnz() as u32).rev().collect();
                let (mut pc, mut pv) = (vec![0u32; p.nnz()], vec![0u32; p.nnz()]);
                map_parts_at::<u32, u32>(&p, &positions, &soc, &dict, &modes, &mut pc, &mut pv)
                    .unwrap();
                let mut at_row = vec![0u8; width];
                for (&c, &v) in pc.iter().zip(&pv) {
                    at_row[c as usize * 4..c as usize * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                assert_eq!(at_row, want_row, "row {r} by position");
                let beyond = [p.nnz() as u32];
                assert!(
                    map_parts_at::<u32, u32>(&p, &beyond, &soc, &dict, &modes, &mut pc, &mut pv)
                        .is_err()
                );
            }
        }

        // A packed record read from pages.
        let range = pmeta.record_range(r).unwrap();
        let first = range.start / 4096;
        let paged = PagedRecord {
            pages: &pages[first..],
            page_size: 4096,
            start: range.start % 4096,
            len: range.len(),
        };
        let n = decode_record_pages(
            &pmeta,
            &tables,
            paged,
            IdSpace::Source,
            &mut rix,
            &mut rvx,
            &mut ws,
        )
        .unwrap();
        assert_eq!(
            (&rix[..n], &rvx[..n * vw]),
            (want_ix, want_v),
            "paged row {r}"
        );
    }
}

#[test]
fn gene_structured_floats_take_column_modes_and_shrink() {
    for renumber in [true, false] {
        let m = Matrix::generate(3, 300, 400, Kind::Baseline);
        let with = encode(&m, &options(renumber, Some(DEFAULT_MODE_SHARE)));
        let without = encode(&m, &options(renumber, None));
        assert!(parse_meta(&with).unwrap().column_modes().is_some());
        assert!(parse_meta(&without).unwrap().column_modes().is_none());
        assert!(
            with.len() * 4 < without.len(),
            "{} vs {}",
            with.len(),
            without.len()
        );
        check_roundtrip(&m, &with);
        check_roundtrip(&m, &without);
    }
}

#[test]
fn counts_and_unique_floats_keep_per_cell_coding() {
    for kind in [Kind::Counts, Kind::Unique] {
        let m = Matrix::generate(5, 200, 300, kind);
        let chunk = encode(&m, &options(true, Some(DEFAULT_MODE_SHARE)));
        assert!(
            parse_meta(&chunk).unwrap().column_modes().is_none(),
            "{kind:?}"
        );
        check_roundtrip(&m, &chunk);
    }
}

#[test]
fn a_zero_share_threshold_still_requires_a_saving() {
    let m = Matrix::generate(9, 200, 300, Kind::Counts);
    let chunk = encode(&m, &options(false, Some(0.0)));
    check_roundtrip(&m, &chunk);
    let m = Matrix::generate(9, 200, 300, Kind::Baseline);
    let chunk = encode(&m, &options(false, Some(0.0)));
    assert!(parse_meta(&chunk).unwrap().column_modes().is_some());
    check_roundtrip(&m, &chunk);
}
