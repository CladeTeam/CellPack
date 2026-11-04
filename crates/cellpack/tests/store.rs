//! `.cpk` files: CSR and dense inputs round-trip through any chunking and
//! append pattern, rows pick their own layout, and damage is detected.

use std::path::PathBuf;

use cellpack::format::{Dtype, RecordLayout, SourceLayout, parse_meta};
use cellpack::scans::*;
use cellpack::store::{CpkFile, CpkOptions, CpkWriter};

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

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cellpack-cpk-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("m.cpk")
}

fn options(layout: LayoutChoice, values: ValueChoice, chunk_rows: usize) -> CpkOptions {
    CpkOptions {
        encode: EncodeOptions {
            renumber: true,
            thresholds: Thresholds {
                indices: DEFAULT_MU_INDICES,
                values: DEFAULT_MU_VALUES,
            },
            max_lanes: DEFAULT_MAX_LANES,
            lane_min: DEFAULT_LANE_MIN,
            column_modes: Some(DEFAULT_MODE_SHARE),
            layout,
            values,
        },
        chunk_rows,
        page_size: 4096,
    }
}

/// Dense rows: `kind` 0 sparse counts, 1 nearly full small integers,
/// 2 embeddings (every value distinct, no zeros); f32 bit patterns.
fn dense_rows(seed: u64, rows: usize, cols: usize, kind: u8) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::with_capacity(rows * cols * 4);
    for _ in 0..rows {
        for _ in 0..cols {
            let v: f32 = match kind {
                0 if rng.below(10) == 0 => (1 + rng.below(4)) as f32,
                0 => 0.0,
                1 => rng.below(6) as f32,
                _ => (rng.next() as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0 + 1e-3,
            };
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// The dense rows as CSR (non-zero values only).
fn to_csr(data: &[u8], cols: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let (mut idx, mut val) = (Vec::new(), Vec::new());
    for row in data.chunks_exact(cols * 4) {
        for (c, v) in row.chunks_exact(4).enumerate() {
            if v != [0; 4] {
                idx.push(c as u32);
                val.extend_from_slice(v);
            }
        }
        indptr.push(idx.len() as u64);
    }
    (indptr, idx, val)
}

fn check(path: &PathBuf, dense: &[u8], cols: usize) {
    let f = CpkFile::open(path).unwrap();
    let rows = (dense.len() / (cols * 4)) as u64;
    assert_eq!(f.n_rows(), rows);
    let (ip, ix, vx) = to_csr(dense, cols);
    let all = f.read_csr(0..rows).unwrap();
    assert_eq!(
        (all.indptr, all.indices, all.data),
        (ip.clone(), ix.clone(), vx.clone())
    );
    assert_eq!(f.read_dense(0..rows).unwrap(), dense);
    // Ranges inside and across chunks.
    for (a, b) in [
        (0, 1),
        (3, 17),
        (rows / 3, rows - 1),
        (rows - 1, rows),
        (5, 5),
    ] {
        let part = f.read_csr(a..b).unwrap();
        let base = ip[a as usize];
        let want: Vec<u64> = ip[a as usize..=b as usize]
            .iter()
            .map(|&x| x - base)
            .collect();
        assert_eq!(part.indptr, want, "{a}..{b}");
        assert_eq!(part.indices, &ix[base as usize..ip[b as usize] as usize]);
        assert_eq!(
            f.read_dense(a..b).unwrap(),
            &dense[a as usize * cols * 4..b as usize * cols * 4]
        );
    }
}

fn layouts(path: &PathBuf) -> (usize, usize) {
    let f = CpkFile::open(path).unwrap();
    let (mut sparse, mut dense) = (0, 0);
    for i in 0..f.chunks().len() {
        let chunk = f.read_chunk(i).unwrap();
        let meta = parse_meta(&chunk).unwrap();
        for c in 0..meta.n_cells() {
            if meta.cell_nnz(c).unwrap() == 0 {
                continue;
            }
            match meta.cell_layout(c).unwrap() {
                RecordLayout::Sparse => sparse += 1,
                RecordLayout::Dense => dense += 1,
            }
        }
    }
    (sparse, dense)
}

#[test]
fn dense_and_csr_inputs_round_trip_in_pieces() {
    let cols = 300;
    for kind in 0..3u8 {
        let dense = dense_rows(kind as u64 + 7, 101, cols, kind);
        let (ip, ix, vx) = to_csr(&dense, cols);
        for chunk_rows in [1usize, 16, 64, 4096] {
            // Dense input, appended in uneven pieces.
            let path = temp(&format!("d{kind}-{chunk_rows}"));
            let mut w = CpkWriter::create(
                &path,
                cols as u32,
                Dtype::F32,
                SourceLayout::Dense,
                options(LayoutChoice::Auto, ValueChoice::Auto, chunk_rows),
            )
            .unwrap();
            let width = cols * 4;
            for (a, b) in [(0, 5), (5, 40), (40, 41), (41, 101)] {
                w.append(DenseRef {
                    n_genes: cols as u32,
                    value_width: 4,
                    data: &dense[a * width..b * width],
                })
                .unwrap();
            }
            w.finish().unwrap();
            check(&path, &dense, cols);

            // The same matrix as CSR, in pieces that cut chunks differently.
            let path = temp(&format!("c{kind}-{chunk_rows}"));
            let mut w = CpkWriter::create(
                &path,
                cols as u32,
                Dtype::F32,
                SourceLayout::Csr,
                options(LayoutChoice::Auto, ValueChoice::Auto, chunk_rows),
            )
            .unwrap();
            for (a, b) in [(0usize, 33usize), (33, 101)] {
                let base = ip[a];
                let sub: Vec<u64> = ip[a..=b].iter().map(|&x| x - base).collect();
                let (s, e) = (ip[a] as usize, ip[b] as usize);
                w.append(CsrRef {
                    n_genes: cols as u32,
                    value_width: 4,
                    indptr: &sub,
                    indices: Indices::U32(&ix[s..e]),
                    data: &vx[s * 4..e * 4],
                })
                .unwrap();
            }
            w.finish().unwrap();
            check(&path, &dense, cols);
        }
    }
}

#[test]
fn rows_take_the_layout_their_data_needs() {
    let cols = 256;
    let path = temp("layouts");
    let mut rows = dense_rows(1, 40, cols, 0);
    rows.extend(dense_rows(2, 40, cols, 2));
    let mut w = CpkWriter::create(
        &path,
        cols as u32,
        Dtype::F32,
        SourceLayout::Dense,
        options(LayoutChoice::Auto, ValueChoice::Auto, 4096),
    )
    .unwrap();
    w.append(DenseRef {
        n_genes: cols as u32,
        value_width: 4,
        data: &rows,
    })
    .unwrap();
    w.finish().unwrap();
    check(&path, &rows, cols);
    // Sparse counts stay sparse; embeddings (no zeros) are stored dense.
    assert_eq!(layouts(&path), (40, 40));

    for (layout, want) in [
        (LayoutChoice::Sparse, (80, 0)),
        (LayoutChoice::Dense, (0, 80)),
    ] {
        for values in [
            ValueChoice::Auto,
            ValueChoice::Raw,
            ValueChoice::Dict,
            ValueChoice::Modes,
        ] {
            let path = temp(&format!("forced-{layout:?}-{values:?}"));
            let mut w = CpkWriter::create(
                &path,
                cols as u32,
                Dtype::F32,
                SourceLayout::Dense,
                options(layout, values, 32),
            )
            .unwrap();
            w.append(DenseRef {
                n_genes: cols as u32,
                value_width: 4,
                data: &rows,
            })
            .unwrap();
            w.finish().unwrap();
            check(&path, &rows, cols);
            assert_eq!(layouts(&path), want, "{layout:?} {values:?}");
        }
    }
}

#[test]
fn explicit_zeros_in_csr_survive_unless_dense_is_forced() {
    // Every row stores a zero explicitly among otherwise full columns.
    let cols = 8u32;
    let indptr = [0u64, 8, 16];
    let indices: Vec<u32> = (0..16).map(|i| i % 8).collect();
    let data: Vec<u8> = (0..16u16)
        .flat_map(|i| if i % 8 == 3 { 0u16 } else { i + 1 }.to_le_bytes())
        .collect();
    let path = temp("zeros");
    let mut w = CpkWriter::create(
        &path,
        cols,
        Dtype::U16,
        SourceLayout::Csr,
        options(LayoutChoice::Auto, ValueChoice::Auto, 4096),
    )
    .unwrap();
    w.append(CsrRef {
        n_genes: cols,
        value_width: 2,
        indptr: &indptr,
        indices: Indices::U32(&indices),
        data: &data,
    })
    .unwrap();
    w.finish().unwrap();
    let f = CpkFile::open(&path).unwrap();
    let got = f.read_csr(0..2).unwrap();
    assert_eq!(
        (got.indptr, got.indices, got.data),
        (indptr.to_vec(), indices.clone(), data.clone())
    );

    let path = temp("zeros-dense");
    let mut w = CpkWriter::create(
        &path,
        cols,
        Dtype::U16,
        SourceLayout::Csr,
        options(LayoutChoice::Dense, ValueChoice::Auto, 4096),
    )
    .unwrap();
    w.append(CsrRef {
        n_genes: cols,
        value_width: 2,
        indptr: &indptr,
        indices: Indices::U32(&indices),
        data: &data,
    })
    .unwrap();
    w.finish().unwrap();
    let got = CpkFile::open(&path).unwrap().read_csr(0..2).unwrap();
    assert_eq!(got.indptr, [0, 7, 14]);
    assert!(got.indices.iter().all(|&c| c != 3));
}

#[test]
fn layout_and_damage() {
    let cols = 4u32;
    let path = temp("layout");
    let mut w = CpkWriter::create(
        &path,
        cols,
        Dtype::U16,
        SourceLayout::Dense,
        options(LayoutChoice::Auto, ValueChoice::Auto, 2),
    )
    .unwrap();
    let data: Vec<u8> = (0..20u16).flat_map(|i| (i % 3).to_le_bytes()).collect();
    w.append(DenseRef {
        n_genes: cols,
        value_width: 2,
        data: &data,
    })
    .unwrap();
    w.finish().unwrap();
    let f = CpkFile::open(&path).unwrap();
    assert_eq!(f.chunks().len(), 3);
    assert_eq!(f.superblock().chunk_rows, 2);
    assert_eq!(f.read_dense(0..5).unwrap(), data);

    // The directory ends the file.
    let bytes = std::fs::read(&path).unwrap();
    let sb = f.superblock();
    assert_eq!(sb.file_len, bytes.len() as u64);
    assert_eq!(sb.directory_offset + 3 * cellpack::format::cpk::ENTRY_LEN as u64, sb.file_len);
    // A flipped directory byte, a truncated file, and an unfinished one.
    let mut bad = bytes.clone();
    let dir = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
    bad[dir + 9] ^= 1;
    for (name, b) in [
        ("flip", bad),
        ("cut", bytes[..bytes.len() - 3].to_vec()),
        ("unfinished", {
            let mut u = bytes.clone();
            u[..4].fill(0);
            u
        }),
    ] {
        let p = temp(&format!("bad-{name}"));
        std::fs::write(&p, b).unwrap();
        assert!(CpkFile::open(&p).is_err(), "{name}");
    }
}

#[test]
fn prepared_reads_reuse_external_outputs_and_retain_the_file() {
    use cellpack::store::ReadWorkspace;
    let path = temp("prepared");
    let dense = dense_rows(107, 17, 13, 0);
    let mut writer = CpkWriter::create(
        &path,
        13,
        Dtype::F32,
        SourceLayout::Dense,
        options(LayoutChoice::Auto, ValueChoice::Auto, 4),
    )
    .unwrap();
    writer
        .append(DenseRef {
            n_genes: 13,
            value_width: 4,
            data: &dense,
        })
        .unwrap();
    writer.finish().unwrap();
    let file = CpkFile::open(&path).unwrap();
    let selection = file.prepare_read(2..15).unwrap();
    drop(file);
    let spec = selection.spec();
    let floats = |bytes: &[u8]| bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    let want = floats(&dense[2 * 13 * 4..15 * 13 * 4]);
    let mut ws = ReadWorkspace::default();
    for workers in [1, 3] {
        let mut out = vec![f32::NAN; spec.dense_len().unwrap()];
        selection.read_dense_into(&mut out, &mut ws, workers).unwrap();
        assert_eq!(out, want);
        out.fill(7.0);
        selection.read_dense_into(&mut out, &mut ws, workers).unwrap();
        assert_eq!(out, want);
        // Conversion follows the output element type.
        let mut wide = vec![0f64; spec.dense_len().unwrap()];
        selection.read_dense_into(&mut wide, &mut ws, workers).unwrap();
        assert_eq!(wide, want.iter().map(|&v| v as f64).collect::<Vec<_>>());
    }
    let (want_ip, want_ix, want_vx) = to_csr(&dense[2 * 13 * 4..15 * 13 * 4], 13);
    let want_vx = floats(&want_vx);
    for workers in [1, 3] {
        let mut ip = vec![-1i64; spec.rows + 1];
        let mut ix = vec![-1i32; spec.nnz + 3];
        let mut vx = vec![f32::NAN; spec.nnz + 3];
        assert_eq!(selection.read_csr_into(&mut ip, &mut ix, &mut vx, &mut ws, workers).unwrap(), spec.nnz);
        assert_eq!(ip, want_ip.iter().map(|&v| v as i64).collect::<Vec<_>>());
        assert_eq!(ix[..spec.nnz], want_ix.iter().map(|&v| v as i32).collect::<Vec<_>>());
        assert_eq!(vx[..spec.nnz], want_vx);
        assert_eq!(&ix[spec.nnz..], &[-1; 3]);
        let before = ip.clone();
        assert!(selection.read_csr_into::<i64, i32, f32>(&mut ip, &mut [], &mut vx, &mut ws, workers).is_err());
        assert_eq!(ip, before);
    }
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn gathers_map_genes_and_match_ranges() {
    use cellpack::store::{ReadOptions, ReadWorkspace, RowSelection};
    let path = temp("gather");
    let (rows, cols) = (23, 9);
    let dense = dense_rows(5, rows, cols, 1);
    let mut writer = CpkWriter::create(&path, cols as u32, Dtype::F32, SourceLayout::Csr, options(LayoutChoice::Auto, ValueChoice::Auto, 4)).unwrap();
    // Dense input converts to the CSR source layout.
    writer.append(DenseRef { n_genes: cols as u32, value_width: 4, data: &dense }).unwrap();
    writer.finish().unwrap();
    let file = CpkFile::open(&path).unwrap();
    let value = |r: usize, c: usize| f32::from_le_bytes(dense[(r * cols + c) * 4..][..4].try_into().unwrap());
    let cells = [17u64, 2, 17, 0, 22, 9];
    let mut map = vec![u32::MAX; cols];
    map[7] = 0;
    map[2] = 3;
    map[4] = 1;
    let selection = file.prepare_selection(RowSelection::List(&cells), Some(&map), 5, ReadOptions::default()).unwrap();
    let mut want = vec![0f64; cells.len() * 5];
    for (i, &cell) in cells.iter().enumerate() {
        for (source, &target) in map.iter().enumerate() {
            if target != u32::MAX {
                want[i * 5 + target as usize] = value(cell as usize, source) as f64;
            }
        }
    }
    let mut ws = ReadWorkspace::default();
    for workers in [1, 4] {
        let mut out = vec![f64::NAN; cells.len() * 5];
        selection.read_dense_into(&mut out, &mut ws, workers).unwrap();
        assert_eq!(out, want);
        let spec = selection.spec();
        let (mut ip, mut ix, mut vx) = (vec![0u32; cells.len() + 1], vec![0u16; spec.nnz], vec![0f64; spec.nnz]);
        let nnz = selection.read_csr_into(&mut ip, &mut ix, &mut vx, &mut ws, workers).unwrap();
        let mut rebuilt = vec![0f64; cells.len() * 5];
        for r in 0..cells.len() {
            let span = ip[r] as usize..ip[r + 1] as usize;
            assert!(ix[span.clone()].windows(2).all(|w| w[0] < w[1]));
            for k in span {
                rebuilt[r * 5 + ix[k] as usize] = vx[k];
            }
        }
        assert_eq!(ip[cells.len()] as usize, nnz);
        assert_eq!(rebuilt, want);
    }
    let range = file.prepare_selection(RowSelection::Range(3..20), None, cols, ReadOptions::default()).unwrap();
    let gather: Vec<u64> = (3..20).collect();
    let listed = file.prepare_selection(RowSelection::List(&gather), None, cols, ReadOptions::default()).unwrap();
    let (mut a, mut b) = (vec![0i32; 17 * cols], vec![0i32; 17 * cols]);
    range.read_dense_into(&mut a, &mut ws, 3).unwrap();
    listed.read_dense_into(&mut b, &mut ws, 2).unwrap();
    assert_eq!(a, b);
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn signed_csr_views_validate_before_narrowing() {
    let path = temp("signed");
    let values: Vec<u8> = [2f32, 3., 7.]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut writer = CpkWriter::create(
        &path,
        5,
        Dtype::F32,
        SourceLayout::Csr,
        options(LayoutChoice::Auto, ValueChoice::Auto, 2),
    )
    .unwrap();
    writer
        .append(CsrView {
            n_genes: 5,
            value_width: 4,
            indptr: Offsets::I32(&[0, 2, 2, 3]),
            indices: Indices::I64(&[0, 4, 2]),
            data: &values,
        })
        .unwrap();
    writer.finish().unwrap();
    let file = CpkFile::open(&path).unwrap();
    let csr = file.read_csr(0..3).unwrap();
    assert_eq!(csr.indptr, [0, 2, 2, 3]);
    assert_eq!(csr.indices, [0, 4, 2]);
    assert_eq!(csr.data, values);
    let opt = options(LayoutChoice::Auto, ValueChoice::Auto, 2).encode;
    for bad in [-1i64, u32::MAX as i64 + 1] {
        let ids = [bad];
        let src = CsrView {
            n_genes: 5,
            value_width: 4,
            indptr: Offsets::I64(&[0, 1]),
            indices: Indices::I64(&ids),
            data: &values[..4],
        };
        let mut out = vec![0; chunk_bound(src, Rows::All, &opt).unwrap()];
        let mut ws = vec![0; chunk_workspace_words(src, Rows::All, &opt).unwrap()];
        assert!(encode_chunk(src, Rows::All, &opt, &mut out, &mut ws).is_err());
    }
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn embedded_stores_obey_their_file_region() {
    let path = temp("embedded");
    let dense = dense_rows(43, 9, 7, 0);
    let mut writer = CpkWriter::create(
        &path,
        7,
        Dtype::F32,
        SourceLayout::Dense,
        options(LayoutChoice::Auto, ValueChoice::Auto, 3),
    )
    .unwrap();
    writer
        .append(DenseRef {
            n_genes: 7,
            value_width: 4,
            data: &dense,
        })
        .unwrap();
    writer.finish().unwrap();
    let encoded = std::fs::read(&path).unwrap();
    let outer = path.with_extension("outer");
    let mut bytes = vec![0x55; 137];
    bytes.extend_from_slice(&encoded);
    bytes.extend_from_slice(&[0xaa; 1024]);
    std::fs::write(&outer, bytes).unwrap();
    let file = CpkFile::open_region(&outer, 137, Some(encoded.len() as u64)).unwrap();
    assert_eq!(file.read_dense(1..8).unwrap(), dense[7 * 4..8 * 7 * 4]);
    assert!(CpkFile::open_region(&outer, 137, Some(encoded.len() as u64 - 1)).is_err());
    assert!(CpkFile::open_region(&outer, u64::MAX, None).is_err());
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}
