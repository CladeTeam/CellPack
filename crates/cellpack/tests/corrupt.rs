//! Damaged metadata must produce errors or a usable view, never a panic or
//! an out-of-bounds access: metadata is validated once per chunk. Records are
//! trusted (decoding a damaged record is undefined behavior), so they are not
//! damaged here.

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
}

/// With `modes`, rows are nearly dense with a per-gene baseline value, so the
/// chunk keeps column modes and most records are mode-coded.
fn chunk(seed: u64, n_genes: u32, vw: usize, renumber: bool, modes: bool) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut indptr = vec![0u64];
    let (mut idx, mut data) = (Vec::new(), Vec::new());
    for r in 0..40 {
        for g in 0..n_genes {
            if modes && rng.next() % 100 < 90 {
                idx.push(g);
                let v = if rng.next() % 100 < 90 {
                    (g as u64).wrapping_mul(0x9e37_79b9)
                } else {
                    rng.next()
                };
                data.extend_from_slice(&v.to_le_bytes()[..vw]);
            } else if !modes && rng.next() % 100 < (r % 5) * 10 + (g < 20) as u64 * 30 {
                idx.push(g);
                let v = if r.is_multiple_of(7) {
                    rng.next()
                } else {
                    1 + rng.next() % 4 + rng.next().is_multiple_of(30) as u64 * (rng.next() % 1000)
                };
                data.extend_from_slice(&v.to_le_bytes()[..vw]);
            }
        }
        indptr.push(idx.len() as u64);
    }
    let src = CsrRef {
        n_genes,
        value_width: vw,
        indptr: &indptr,
        indices: Indices::U32(&idx),
        data: &data,
    };
    let opt = EncodeOptions {
        renumber,
        thresholds: Thresholds {
            indices: 1.0,
            values: 1.0,
        },
        max_lanes: 32,
        lane_min: 4,
        column_modes: modes.then_some(0.5),
        layout: LayoutChoice::Auto,
        values: ValueChoice::Auto,
    };
    let mut out = vec![0u8; chunk_bound(src, Rows::All, &opt).unwrap()];
    let mut ws = vec![0u64; chunk_workspace_words(src, Rows::All, &opt).unwrap()];
    let len = encode_chunk(src, Rows::All, &opt, &mut out, &mut ws).unwrap();
    out.truncate(len);
    out
}

fn damage(rng: &mut Rng, bytes: &mut Vec<u8>) {
    match rng.next() % 4 {
        0 | 1 => {
            let i = (rng.next() as usize) % bytes.len();
            bytes[i] ^= 1 << (rng.next() % 8);
        }
        2 => bytes.truncate((rng.next() as usize) % bytes.len()),
        _ => bytes.push(rng.next() as u8),
    }
}

#[test]
fn record_headers_are_read_with_bounds() {
    assert_eq!(record_nnz(&[]), Ok(0));
    assert_eq!(record_nnz(&[1, 5]), Ok(5));
    assert_eq!(record_nnz(&[1, 0x80, 0x01]), Ok(128));
    assert!(record_nnz(&[1]).is_err());
    assert!(record_nnz(&[1, 0x80]).is_err());
    assert!(record_nnz(&[1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f]).is_err());
    assert!(record_nnz(&[1; 1].iter().chain(&[0xff; 16]).copied().collect::<Vec<_>>()).is_err());
}

#[test]
fn damaged_metadata_never_panics() {
    let mut rng = Rng(7);
    for c in [chunk(5, 250, 4, true, false), chunk(8, 250, 4, true, true)] {
        damage_metadata(&mut rng, &c);
    }
}

/// Every metadata-level entry point on damaged metadata: parsing, tables,
/// the records region and the per-cell tables.
fn damage_metadata(rng: &mut Rng, c: &[u8]) {
    let meta_len = parse_meta(c).unwrap().meta_len();
    let mut rejected = 0;
    for _ in 0..3000 {
        let mut bad = c.to_vec();
        let i = (rng.next() as usize) % meta_len.min(bad.len());
        bad[i] ^= 1 << (rng.next() % 8);
        if rng.next().is_multiple_of(5) {
            damage(rng, &mut bad);
        }
        let Ok(meta) = parse_meta(&bad) else {
            rejected += 1;
            continue;
        };
        let _ = decode_tables(&meta);
        let records = meta.records_of(&bad);
        let _ = selection_nnz(&meta, Rows::All);
        for cell in 0..meta.n_cells().min(64) {
            let _ = meta.cell_nnz(cell);
            let _ = meta.cell_layout(cell);
            if let Ok(records) = records {
                let _ = meta.record(records, cell);
            }
        }
    }
    assert!(rejected > 0);
}
