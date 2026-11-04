//! Single-thread chunk encoding throughput on raw CSR datasets.
//!
//! Usage: `encode <csr_root> <n_datasets> <values> [chunk_rows]`. Each
//! dataset directory under `csr_root` holds `indptr.i64`, `indices.i32` and
//! `data.f32`. `values` sets the value type: `u8`, `u16`, `u32`, `u64`
//! (counts cast to integers), `f32`, `f64`. Environment options:
//!
//! - `NORM=1`: values become `log1p(1e4 * x / row sum)` (normalized floats).
//! - `MODES=0|1`: column modes (default on for floats, as `CpkWriter` does).
//! - `LAYOUT=auto|sparse|dense`, `VALUES=auto|dict|modes|raw`, `RENUMBER=0|1`,
//!   `LANES=<max lanes>`, `LANE_MIN=<n>`.
//! - `DENSE=<rows>`: encode the first rows of each dataset from a dense matrix.
//! - `SKIP=<n>`: skip the first datasets.
//! - `REPEAT=<n>`: encode each chunk `n` times (profiling).
//! - `ROWS=<n>`: the first rows of each dataset only.
//! - `GENES=<k>`: keep the first `k` genes only.
//! - `DENSIFY=<p>`: genes a row lacks get count 1 where `(7 gene + row) % 100
//!   < p`, so rows reach at least `p`% non-zeros.
//!
//! Reports MB/s of raw CSR input (`nnz * (4 + value width)` bytes) for
//! sizing, encoding and packing, the packed ratio, and a digest of the
//! packed chunks.

use std::path::Path;
use std::time::Instant;

use cellpack::scans::*;

fn read<T: Copy>(path: &Path) -> Vec<T> {
    let bytes = std::fs::read(path).unwrap();
    let n = bytes.len() / size_of::<T>();
    let mut v = Vec::<T>::with_capacity(n);
    // SAFETY: plain-old-data element types; the length covers the bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr().cast::<u8>(), n * size_of::<T>());
        v.set_len(n);
    }
    v
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "Usage: encode <csr_root> <n_datasets> <u8|u16|u32|u64|f32|f64> [chunk_rows]\n\
        Each dataset directory contains indptr.i64, indices.i32 and data.f32.\n\
        Environment: NORM, MODES, LAYOUT, VALUES, RENUMBER, LANES, LANE_MIN,\n\
        DENSE, SKIP, REPEAT, ROWS, GENES, DENSIFY.\n\
        See crates/cellpack/examples/README.md and the source for details.";
    if args.iter().skip(1).any(|a| a == "--help" || a == "-h") {
        println!("{usage}");
        return;
    }
    if args.len() < 4 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let root = Path::new(&args[1]);
    let count: usize = args[2].parse().unwrap();
    let kind = args[3].as_str();
    let chunk_rows: usize = args.get(4).map_or(4096, |a| a.parse().unwrap());
    assert!(chunk_rows > 0, "chunk_rows must be positive");
    assert!(matches!(kind, "u8" | "u16" | "u32" | "u64" | "f32" | "f64"), "unsupported value type: {kind}");
    let float = kind.starts_with('f');
    let norm = env("NORM").is_some_and(|v| v == "1");
    let skip: usize = env("SKIP").map_or(0, |v| v.parse().unwrap());
    let dense_rows: Option<usize> = env("DENSE").map(|v| v.parse().unwrap());
    let repeat: usize = env("REPEAT").map_or(1, |v| v.parse().unwrap());
    assert!(repeat > 0, "REPEAT must be positive");
    let opt = EncodeOptions {
        renumber: env("RENUMBER").is_none_or(|v| v == "1"),
        thresholds: Thresholds { indices: DEFAULT_MU_INDICES, values: DEFAULT_MU_VALUES },
        max_lanes: env("LANES").map_or(DEFAULT_MAX_LANES, |v| v.parse().unwrap()),
        lane_min: env("LANE_MIN").map_or(DEFAULT_LANE_MIN, |v| v.parse().unwrap()),
        column_modes: env("MODES").map_or(float, |v| v == "1").then_some(DEFAULT_MODE_SHARE),
        layout: match env("LAYOUT").as_deref() {
            Some("sparse") => LayoutChoice::Sparse,
            Some("dense") => LayoutChoice::Dense,
            _ => LayoutChoice::Auto,
        },
        values: match env("VALUES").as_deref() {
            Some("dict") => ValueChoice::Dict,
            Some("modes") => ValueChoice::Modes,
            Some("raw") => ValueChoice::Raw,
            _ => ValueChoice::Auto,
        },
    };
    let mut dirs: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("indptr.i64").exists())
        .collect();
    dirs.sort();
    let (mut t_size, mut t_enc, mut t_pack) = (0.0, 0.0, 0.0);
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let (mut raw, mut packed_total, mut cells) = (0usize, 0usize, 0usize);
    let (mut out, mut ws, mut packed): (Vec<u8>, Vec<u64>, Vec<u8>) = Default::default();
    for dir in dirs.iter().skip(skip).take(count) {
        let indptr: Vec<i64> = read(&dir.join("indptr.i64"));
        let mut indptr: Vec<u64> = indptr.iter().map(|&x| x as u64).collect();
        let mut indices: Vec<i32> = read(&dir.join("indices.i32"));
        let mut data: Vec<f32> = read(&dir.join("data.f32"));
        let mut n_genes = (indices.iter().copied().max().unwrap_or(0) as u32 + 1).max(1);
        let genes: Option<u32> = env("GENES").map(|v| v.parse().unwrap());
        let densify: Option<usize> = env("DENSIFY").map(|v| v.parse().unwrap());
        if genes.is_some() || densify.is_some() {
            let k = genes.unwrap_or(n_genes);
            let (mut ip, mut ix, mut dx) = (vec![0u64], Vec::new(), Vec::new());
            for r in 0..indptr.len() - 1 {
                let (a, b) = (indptr[r] as usize, indptr[r + 1] as usize);
                let mut t = a;
                for g in 0..k as i32 {
                    while t < b && indices[t] < g {
                        t += 1;
                    }
                    if t < b && indices[t] == g {
                        ix.push(g);
                        dx.push(data[t]);
                    } else if densify.is_some_and(|p| (7 * g as usize + r) % 100 < p) {
                        ix.push(g);
                        dx.push(1.0);
                    }
                }
                ip.push(ix.len() as u64);
            }
            (indptr, indices, data, n_genes) = (ip, ix, dx, k);
        }
        if let Some(r) = dense_rows.or(env("ROWS").map(|v| v.parse().unwrap())) {
            let r = r.min(indptr.len() - 1);
            indptr.truncate(r + 1);
            indices.truncate(indptr[r] as usize);
            data.truncate(indptr[r] as usize);
        }
        let rows = indptr.len() - 1;
        let values: Vec<f64> = if norm {
            let mut v = Vec::with_capacity(data.len());
            for r in 0..rows {
                let s = &data[indptr[r] as usize..indptr[r + 1] as usize];
                let total: f64 = s.iter().map(|&x| x as f64).sum::<f64>().max(1.0);
                v.extend(s.iter().map(|&x| (1e4 * x as f64 / total).ln_1p()));
            }
            v
        } else {
            data.iter().map(|&x| x as f64).collect()
        };
        let (vw, bytes): (usize, Vec<u8>) = match kind {
            "u8" => (1, values.iter().map(|&x| x.min(255.0) as u8).collect()),
            "u16" => (2, values.iter().flat_map(|&x| (x as u16).to_le_bytes()).collect()),
            "u32" => (4, values.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect()),
            "u64" => (8, values.iter().flat_map(|&x| (x as u64).to_le_bytes()).collect()),
            "f64" => (8, values.iter().flat_map(|&x| x.to_le_bytes()).collect()),
            _ => (4, values.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect()),
        };
        raw += indices.len() * (4 + vw);
        cells += rows;
        let dense_data: Vec<u8> = if dense_rows.is_some() {
            let g = n_genes as usize;
            let mut d = vec![0u8; rows * g * vw];
            for r in 0..rows {
                for t in indptr[r] as usize..indptr[r + 1] as usize {
                    let at = (r * g + indices[t] as usize) * vw;
                    d[at..at + vw].copy_from_slice(&bytes[t * vw..(t + 1) * vw]);
                }
            }
            d
        } else {
            Vec::new()
        };
        let src: MatrixRef = if dense_rows.is_some() {
            DenseRef { n_genes, value_width: vw, data: &dense_data }.into()
        } else {
            CsrRef { n_genes, value_width: vw, indptr: &indptr, indices: Indices::I32(&indices), data: &bytes }.into()
        };
        let mut r0 = 0;
        while r0 < rows {
            let end = (r0 + chunk_rows).min(rows);
            let sel = Rows::Range { start: r0, end };
            let t = Instant::now();
            let sizes = chunk_sizes(src, sel, &opt).unwrap();
            t_size += t.elapsed().as_secs_f64();
            // Buffers are reused across chunks, as a writer would.
            if out.len() < sizes.output_bytes {
                out = vec![0u8; sizes.output_bytes];
            }
            if ws.len() < sizes.workspace_words {
                ws = vec![0u64; sizes.workspace_words];
            }
            let t = Instant::now();
            let mut len = 0;
            for _ in 0..repeat {
                len = encode_chunk(src, sel, &opt, &mut out, &mut ws).unwrap();
            }
            t_enc += t.elapsed().as_secs_f64() / repeat as f64;
            let t = Instant::now();
            let need = packed_len(&out[..len], 4096).unwrap();
            if packed.len() < need {
                packed = vec![0u8; need];
            }
            let plen = pack_chunk(&out[..len], 4096, &mut packed).unwrap();
            t_pack += t.elapsed().as_secs_f64();
            packed_total += plen;
            for &b in &packed[..plen] {
                digest = (digest ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
            r0 = end;
        }
    }
    let mb = raw as f64 / 1e6;
    println!(
        "{kind} cells={cells} raw={mb:.1}MB ratio={:.3} | size {:.0} MB/s, encode {:.0} MB/s, pack {:.0} MB/s, total {:.0} MB/s | digest {digest:016x}",
        raw as f64 / packed_total as f64,
        mb / t_size,
        mb / t_enc,
        mb / t_pack,
        mb / (t_size + t_enc + t_pack),
    );
}
