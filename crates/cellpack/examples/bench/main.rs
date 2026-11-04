//! Real-data driver: access generation, offline planning and loading.
//!
//! The corpus is a directory with one subdirectory per dataset holding
//! `raw.cpk`, a flat directory of `.cpk` files (a copy of it, one file per
//! dataset) or a single `.cpk` file. Files are taken in path order, which is
//! the corpus manifest's dataset order, so cell ids are the global ids of the
//! loader benchmark and cache-policy. Every column's standard id is given
//! explicitly: `ids=FILE` holds little-endian u32 ids (`UNMAPPED`: dropped)
//! for every column of every corpus file, concatenated in path order (the
//! loader benchmark's `common/gene_map.u32` is such a file), and `vocab=N`
//! is the number of standard ids; `files=` selects files with their maps.
//! The driver infers and aligns nothing. `embed=DIR` adds
//! `.cpk` files laid out as the corpus with the same cell counts (an
//! embedding, say), their dims standard ids after the genes; the `emb` schema
//! reads them at the same positions (paired accesses).
//!
//! ```text
//! bench gen   <corpus> <out.u32> random|block|weighted|power <epochs> ids=FILE vocab=N [files=N] [seed=S]
//!             [block=256] [group=16] [repeat=R]
//! bench plan  <corpus> <seq> <out.plan> cache=BYTES [loader options] [plan options]
//!             [online=0|1] (online: the online compiler over the stored sequence)
//! bench online <corpus> <seq> cache=BYTES [loader options] [plan options]
//!             (compiles online without loading: throughput and modeled time)
//! bench run   <corpus> <seq> [loader options] [cache=off|plan:PATH|online:BYTES]
//!             [slots=S] [world=R] [warm=FRACTION] [plan options, for online]
//!
//! seq: a .u32 file of positions' cells, or shuffle:EPOCHS[:SEED[:BLOCK:GROUP]]
//! plan options: [objective=time|misses] [rows=0|1] [row_policy=incremental|global] [out=0|1]
//!             [block_pages=N] [threads=T] [horizon=SLOTS] [gap=SLOTS] [chunk=ACCESSES]
//!             [cost=auto|defaults|none|FILE.json]
//!             (auto, the default: pycellpack's lookup of measured records under
//!              $CELLPACK_HOME, default ~/.cellpack, falling back per part to its
//!              defaults with a warning, or the saved model $CELLPACK_COST_MODEL;
//!              defaults: those defaults alone; none: no cost model, every page
//!              hit equal; FILE.json: a model saved by pycellpack's CostModel.save)
//!             [pps=IOPS bandwidth= access= request= action= bucket= element= decode=
//!              row_element= process_<schema>= map_element= fill_element=]
//!             (a hand-set model, used when pps is given without cost=)
//! bench costs <corpus> [loader options] [cost=...]
//!             (the record keys and the cost model the plan options would use, as JSON)
//! bench curve <corpus> <accesses.u32> [loader options] [fractions=0,0.02,0.1,1]
//! bench ioraw <corpus> <accesses.u32> ids=FILE vocab=N [files=N] [workers=W] [depth=Q]
//!
//! loader options: ids=FILE vocab=N [files=N] [workers=W] [depth=Q] [claim=K] [slot=CELLS]
//!             [window=SLOTS] [backend=uring|pread] [pin=0|1] [profile=SITES|all]
//!             (profile: comma-separated sites or groups, see cellpack::prof)
//!             [schemas=sample,count,dense,dense:u32,emb]
//!             [embed=DIR embed_dim=16]
//! ```
//!
//! `slot=CELLS` sets the batch size (`LoaderConfig::slot_cells`). See
//! `../README.md` for build commands, input formats and a complete run sequence.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use cellpack::store::CpkFile;
use cellpack::loader::*;
use cellpack::opt::*;

#[path = "process/sampling.rs"]
mod sampling;
#[path = "costs/store.rs"]
mod costs;
use sampling::{sample, sample_scratch_bytes, count, count_dense, Candidates};

const SAMPLE: usize = 2048;

const USAGE: &str = "Usage: bench <command> <corpus> [arguments] ids=FILE vocab=N [options]
Commands:
  gen    <corpus> <out.u32> random|block|weighted|power <epochs>
  plan   <corpus> <seq> <out.plan> cache=BYTES
  online <corpus> <seq> cache=BYTES
  run    <corpus> <seq> [cache=off|plan:PATH|online:BYTES]
  costs  <corpus> [cost=auto|defaults|none|FILE.json]
  curve  <corpus> <accesses.u32> [fractions=0,0.02,0.1,1]
  ioraw  <corpus> <accesses.u32>
Loader options: workers=W depth=Q claim=K slot=CELLS slots=S backend=uring|pread
  schemas=sample,count,dense,dense:u32,emb window=SLOTS profile=SITES|all
Plan options: threads=T gap=SLOTS horizon=SLOTS chunk=ACCESSES rows=0|1
  row_policy=incremental|global objective=time|misses cost=auto|defaults|none|FILE.json
See crates/cellpack/examples/README.md for inputs and invocation examples.";

fn opts(args: &[String]) -> HashMap<String, String> {
    args.iter()
        .filter_map(|a| {
            a.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect()
}
fn get<T: std::str::FromStr>(o: &HashMap<String, String>, k: &str, default: T) -> T {
    o.get(k).map_or(default, |v| v.parse().ok().expect(k))
}

/// Every `.cpk` file of the corpus (or `*/raw.cpk`; the corpus itself if it
/// is a file) in path order.
fn all_files(dir: &Path) -> Vec<PathBuf> {
    if dir.is_file() {
        return vec![dir.to_path_buf()];
    }
    let entries: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    let mut all: Vec<PathBuf> = entries.iter().filter(|p| p.extension().is_some_and(|e| e == "cpk")).cloned().collect();
    if all.is_empty() {
        all = entries.iter().map(|p| p.join("raw.cpk")).filter(|p| p.is_file()).collect();
    }
    assert!(!all.is_empty(), "no .cpk or */raw.cpk files in {}", dir.display());
    all.sort();
    all
}

/// Indices of `files` corpus files spread evenly over `n` (all for 0).
fn picks(n: usize, files: usize) -> Vec<usize> {
    let files = if files == 0 { n } else { files.min(n) };
    (0..files).map(|k| k * n / files).collect()
}

/// The corpus files `files=` selects, in path order.
fn corpus_files(dir: &Path, files: usize) -> Vec<PathBuf> {
    let all = all_files(dir);
    picks(all.len(), files).into_iter().map(|k| all[k].clone()).collect()
}

/// The explicit standard ids: `vocab=N` and, from `ids=FILE`, every corpus
/// file's column -> standard id map (see the module docs).
struct Ids {
    vocab: usize,
    maps: Vec<Vec<u32>>,
}

fn ids(dir: &Path, o: &HashMap<String, String>) -> Ids {
    let path = o.get("ids").expect("ids=FILE required: every corpus column's standard id");
    let vocab: usize = o.get("vocab").expect("vocab=N required").parse().expect("vocab=N");
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("ids={path}: {e}"));
    assert!(bytes.len().is_multiple_of(4), "ids={path}: little-endian u32 expected");
    let all: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(all.iter().all(|&id| id == cellpack::scans::UNMAPPED || (id as usize) < vocab), "ids={path}: ids outside vocab={vocab}");
    let mut maps = Vec::new();
    let mut at = 0;
    for file in all_files(dir) {
        let n = CpkFile::open(&file).unwrap().n_cols() as usize;
        maps.push(all.get(at..at + n).unwrap_or_else(|| panic!("ids={path}: shorter than the corpus columns")).to_vec());
        at += n;
    }
    assert_eq!(at, all.len(), "ids={path}: longer than the corpus columns");
    Ids { vocab, maps }
}

/// The column maps of the files `files=` selects.
fn corpus_maps(files: usize, ids: &Ids) -> Vec<Vec<u32>> {
    picks(ids.maps.len(), files).into_iter().map(|k| ids.maps[k].clone()).collect()
}

/// The data source over the corpus files, columns mapped by the explicit ids,
/// then the embedding files of `embed` (dims as standard ids after `vocab`).
fn source(dir: &Path, files: usize, embed: Option<(&Path, usize)>, ids: &Ids) -> DataSource {
    let mut b = DataSourceBuilder::new(ids.vocab + embed.map_or(0, |e| e.1), 4096).unwrap().direct_io(true);
    for (path, map) in corpus_files(dir, files).iter().zip(corpus_maps(files, ids)) {
        b.add_cpk(path, &map).unwrap();
    }
    if let Some((dir, dim)) = embed {
        let dims: Vec<u32> = (ids.vocab..ids.vocab + dim).map(|s| s as u32).collect();
        for path in corpus_files(dir, files) {
            b.add_cpk(&path, &dims).unwrap();
        }
    }
    b.build().unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i as u64 + 1) as usize);
    }
}

fn generate(args: &[String]) {
    let o = opts(args);
    let c = source(Path::new(&args[0]), get(&o, "files", 0), None, &ids(Path::new(&args[0]), &o));
    let (pattern, epochs): (&str, usize) = (&args[2], args[3].parse().unwrap());
    let n = c.n_cells() as u32;
    let mut rng = Rng(get(&o, "seed", 1));
    let mut out: Vec<u32> = Vec::with_capacity(n as usize * epochs);
    if pattern == "power" {
        // (p_1^r ... p_n^r)^(epochs/r): identical cell frequencies and length
        // for every r; only the ordering of their repetitions changes.
        let repeat: usize = get(&o, "repeat", 1);
        assert!(repeat > 0 && epochs > 0 && epochs.is_multiple_of(repeat));
        let mut order: Vec<u32> = (0..n).collect();
        shuffle(&mut order, &mut rng);
        for _ in 0..epochs / repeat {
            for &cell in &order {
                out.extend(std::iter::repeat_n(cell, repeat));
            }
        }
    }
    for _ in 0..if pattern == "power" { 0 } else { epochs } {
        match pattern {
            "random" => {
                let mut e: Vec<u32> = (0..n).collect();
                shuffle(&mut e, &mut rng);
                out.extend(e);
            }
            // Explicit locality controls belong to the workload generator;
            // the loader receives only the resulting access sequence.
            "block" => {
                let block: u32 = get(&o, "block", 256);
                let group: usize = get(&o, "group", 16);
                assert!(block > 0 && group > 0, "block and group must be positive");
                let mut blocks: Vec<u32> = (0..n.div_ceil(block)).collect();
                shuffle(&mut blocks, &mut rng);
                for group in blocks.chunks(group) {
                    let mut cells: Vec<u32> = group
                        .iter()
                        .flat_map(|&b| {
                            b * block..(b as u64 * block as u64 + block as u64).min(n as u64) as u32
                        })
                        .collect();
                    shuffle(&mut cells, &mut rng);
                    out.extend(cells);
                }
            }
            // Cells drawn with replacement, each chunk weighted by the corpus
            // chunks it stands for (`weights.tsv`: dataset, chunk, weight), so
            // the sequence follows the corpus's cell distribution.
            "weighted" => {
                let weights: HashMap<(String, usize), f64> =
                    std::fs::read_to_string(Path::new(&args[0]).join("weights.tsv"))
                        .unwrap()
                        .lines()
                        .map(|l| {
                            let f: Vec<&str> = l.split('\t').collect();
                            ((f[0].to_string(), f[1].parse().unwrap()), f[2].parse().unwrap())
                        })
                        .collect();
                let mut cum = Vec::new();
                let mut total = 0.0;
                for (m, path) in corpus_files(Path::new(&args[0]), get(&o, "files", 0)).iter().enumerate() {
                    let (first, _) = c.matrix_cells(m as u32).unwrap();
                    let dataset = path.file_stem().unwrap().to_string_lossy().to_string();
                    for (k, e) in CpkFile::open(path).unwrap().chunks().iter().enumerate() {
                        total += weights[&(dataset.clone(), k)] * e.n_rows as f64;
                        cum.push((total, first + e.first_row as u32, e.n_rows as u64));
                    }
                }
                for _ in 0..n {
                    let x = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * total;
                    let k = cum.partition_point(|e| e.0 <= x).min(cum.len() - 1);
                    let (_, first, rows) = cum[k];
                    out.push(first + rng.below(rows) as u32);
                }
            }
            _ => panic!("pattern"),
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[1], bytes).unwrap();
    println!("cells={} accesses={} pages={}", n, out.len(), c.n_pages());
}

fn accesses(path: &str) -> Vec<CellId> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The loader's accesses for positions given by a stored file or
/// `shuffle:EPOCHS[:SEED[:BLOCK:GROUP]]` over the count cells: every schema
/// reads the position's cell, the `emb` schema its embedding (offset by the
/// count cells). Stored positions give stored accesses (offline plans bind
/// to them).
fn sequence(spec: &str, setup: &Setup) -> Box<dyn Accesses> {
    let offsets: Vec<CellId> = setup.names.iter().map(|n| if n == "emb" { setup.count_cells as CellId } else { 0 }).collect();
    let Some(rest) = spec.strip_prefix("shuffle:") else {
        let cells = accesses(spec);
        if offsets.len() == 1 {
            return Box::new(cells);
        }
        return Box::new(stored(&Offset::new(cells, offsets).unwrap()));
    };
    let p: Vec<u64> = rest.split(':').map(|x| x.parse().unwrap()).collect();
    let mut s = Shuffle::new(setup.count_cells, p[0], p.get(1).copied().unwrap_or(0)).unwrap();
    if p.len() >= 4 {
        s = s.blocks(p[2], p[3]).unwrap();
    }
    Box::new(Offset::new(s, offsets).unwrap())
}

/// A computed sequence, stored.
fn stored(seq: &dyn Accesses) -> Vec<CellId> {
    let mut out = vec![0; seq.len() as usize];
    seq.fill(0, &mut out).unwrap();
    out
}

/// What `run`, `plan` and `online` share: the loader, its schemas' names,
/// the count cells (positions' cell range) and the data source build time.
struct Setup {
    loader: Loader,
    names: Vec<String>,
    count_cells: u64,
    catalog_s: f64,
    /// The data source's files and each schema as the cost lookup keys them.
    paths: Vec<PathBuf>,
    keyed: Vec<costs::Schema>,
    backend: &'static str,
}

fn dtype_name(d: Dtype) -> &'static str {
    match d {
        Dtype::U8 => "uint8",
        Dtype::U16 => "uint16",
        Dtype::U32 => "uint32",
        Dtype::U64 => "uint64",
        Dtype::I8 => "int8",
        Dtype::I16 => "int16",
        Dtype::I32 => "int32",
        Dtype::I64 => "int64",
        Dtype::F32 => "float32",
        Dtype::F64 => "float64",
    }
}

/// The loader over the corpus (and an optional embedding), with the schemas
/// named by `schemas=`.
fn build_loader(o: &HashMap<String, String>, corpus: &Path) -> Setup {
    let t = Instant::now();
    let dim: usize = get(o, "embed_dim", 16);
    let files = get(o, "files", 0);
    let ids = ids(corpus, o);
    let source = Arc::new(source(corpus, files, o.get("embed").map(|d| (Path::new(d.as_str()), dim)), &ids));
    let catalog_s = t.elapsed().as_secs_f64();
    let genes = ids.vocab;
    let count_cells = (0..corpus_files(corpus, files).len()).map(|m| source.matrix_cells(m as u32).unwrap().1).sum::<u64>();
    let features: Vec<u32> = (0..genes as u32).collect();
    // Each corpus file's rows and column -> standard id map.
    let maps: Vec<(u64, Vec<u32>)> = corpus_files(corpus, files)
        .iter()
        .zip(corpus_maps(files, &ids))
        .map(|(path, map)| (CpkFile::open(path).unwrap().n_rows(), map))
        .collect();
    let names: Vec<String> = o.get("schemas").map_or("sample", |v| v.as_str()).split(',').map(str::to_string).collect();
    let mut schemas = Vec::new();
    let mut keyed = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let id = i as u64 + 1;
        let identity = name.bytes().fold(i as u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64));
        let schema = match name.as_str() {
            "sample" => {
                // CSR rows over the vocabulary; the candidates are each
                // file's columns in the vocabulary, in column order.
                let candidates = Arc::new(Candidates::new(
                    maps.iter().map(|(rows, map)| (*rows, map.as_slice())),
                    cellpack::scans::UNMAPPED,
                ));
                let table = candidates.table().cast::<c_void>();
                let scratch = sample_scratch_bytes(genes, candidates.max_genes());
                // SAFETY: the table lives as long as the schema (its owner); the
                // function obeys the output and scratch bounds below.
                let p = unsafe { ProcessFn::with_format(RowFormat::Csr, Some(sample), None, table, candidates) }.unwrap();
                keyed.push(Some(costs::Process { csr: sample as *const () as usize, dense: 0, identity, partial_rows: false, row_format: "csr" }));
                OutputSchema::process(id, features.clone(), Dtype::F32, SAMPLE * 8, p, identity, scratch).unwrap()
            }
            "count" => {
                // SAFETY: both functions write the cell's 8 bytes.
                let p = unsafe { ProcessFn::new(count, count_dense, std::ptr::null::<c_void>(), Arc::new(())) };
                keyed.push(Some(costs::Process { csr: count as *const () as usize, dense: count_dense as *const () as usize, identity, partial_rows: false, row_format: "auto" }));
                OutputSchema::process(id, features.clone(), Dtype::F32, 8, p, identity, 0).unwrap()
            }
            "dense" => OutputSchema::dense(id, features.clone(), Dtype::F32).unwrap(),
            "emb" => OutputSchema::dense(id, (genes as u32..(genes + dim) as u32).collect::<Vec<_>>(), Dtype::F32).unwrap(),
            other => match other.strip_prefix("dense:") {
                Some(t) => OutputSchema::dense(id, features.clone(), dtype(t)).unwrap(),
                None => panic!("schema"),
            },
        };
        if keyed.len() == i {
            keyed.push(None);
        }
        schemas.push(schema.named(name.clone()));
    }
    let keyed = schemas
        .iter()
        .zip(keyed)
        .zip(&names)
        .map(|((s, process), name)| costs::Schema { name: name.clone(), dtype: dtype_name(s.dtype()), cell_bytes: s.cell_bytes(), scratch_bytes: s.scratch_bytes(), process })
        .collect();
    let mut paths = corpus_files(corpus, files);
    if let Some(dir) = o.get("embed") {
        paths.extend(corpus_files(Path::new(dir.as_str()), files));
    }
    let config = LoaderConfig {
        workers: get(o, "workers", 32),
        backend: if get(o, "backend", "uring".to_string()) == "pread" { IoBackend::Pread } else { IoBackend::Uring },
        max_inflight_per_worker: get(o, "depth", 32),
        claim_batch: get(o, "claim", 8),
        slot_cells: get(o, "slot", 1024),
        window: o.get("window").map(|d| d.parse().unwrap()),
        profile: o.get("profile").filter(|v| v.as_str() != "0").map(|v| cellpack::prof::SiteMask::parse(&v.split(',').map(|s| if s == "1" { "all" } else { s }).collect::<Vec<_>>()).unwrap()),
        cpus: o.get("cpus").map(|v| v.split(',').map(|c| c.parse().expect("CPU index")).collect())
            .unwrap_or_else(|| if get(o, "pin", 0u8) != 0 { physical_cpus() } else { Vec::new() }),
    };
    let backend = if config.backend == IoBackend::Pread { "pread" } else { "uring" };
    Setup { loader: Loader::new(source, schemas, config).unwrap(), names, count_cells, catalog_s, paths, keyed, backend }
}

fn dtype(name: &str) -> Dtype {
    match name {
        "u8" => Dtype::U8,
        "u16" => Dtype::U16,
        "u32" => Dtype::U32,
        "u64" => Dtype::U64,
        "i32" => Dtype::I32,
        "i64" => Dtype::I64,
        "f32" => Dtype::F32,
        "f64" => Dtype::F64,
        _ => panic!("dtype"),
    }
}

fn process_curve(o: &HashMap<String, String>, name: &str) -> Vec<ProcessCost> {
    o.get(&format!("curve_{name}")).map_or_else(Vec::new, |text| text.split(',').map(|entry| {
        let values: Vec<_> = entry.split(':').collect();
        assert_eq!(values.len(), 3, "curve entries are max_elements:process:resolve");
        ProcessCost { max_elements: values[0].parse().unwrap(), process: values[1].parse().unwrap(), resolve: values[2].parse().unwrap() }
    }).collect())
}

/// Plan options from `cache= rows= objective= threads= block_pages= gap=`
/// and, with `pps=`, a hand-set cost model over the defaults: `pps` (device
/// IOPS), `bandwidth= cores= access= request= bucket= serial=`, `element=`
/// (split 80/20 between decoding and each schema's mapping), `gather=`,
/// `admit=`, per schema `process_<schema>=`.
/// The cost model `cost=` names (see the usage), with where each part came
/// from; None for `cost=none`, or `pps=` without `cost=` (hand-set below).
fn resolve_cost(o: &HashMap<String, String>, setup: &Setup) -> Option<costs::Resolved> {
    let cores = setup.loader.cores();
    let source = costs::Source { paths: &setup.paths, backend: setup.backend, direct_io: true, page_size: 4096 };
    match o.get("cost").map(String::as_str) {
        Some("none") => None,
        None if o.contains_key("pps") => None,
        None | Some("auto") => Some(costs::resolve(&source, &setup.keyed, cores)),
        Some("defaults") => Some(costs::defaults(&source, &setup.keyed, cores)),
        Some(path) => Some(costs::load(Path::new(path), &setup.keyed, cores).unwrap_or_else(|e| panic!("cost model: {e}"))),
    }
}

fn plan_options(o: &HashMap<String, String>, setup: &Setup, cache: usize, online: bool) -> PlanOptions {
    let loader = &setup.loader;
    let mut options = PlanOptions::new(cache, get(o, "gap", loader.window()));
    options.chunk = o.get("chunk").map(|v| v.parse().unwrap());
    options.row_policy = match get(o, "row_policy", "incremental".to_string()).as_str() {
        "global" => RowPolicy::Global,
        _ => RowPolicy::Incremental,
    };
    options.horizon = o.get("horizon").map(|v| v.parse().unwrap());
    options.threads = get(o, "threads", 32);
    options.cpus = o.get("compile_cpus").map_or_else(Vec::new, |v| v.split(',').map(|c| c.parse().expect("compiler CPU index")).collect());
    options.objective = match get(o, "objective", "time".to_string()).as_str() {
        "misses" => PlanObjective::Misses,
        "time" => PlanObjective::Time,
        _ => panic!("objective must be misses or time"),
    };
    options.rows = get(o, "rows", 0u8) != 0;
    options.block_pages = o.get("block_pages").map(|v| v.parse().unwrap());
    if let Some(pps) = o.get("pps") {
        let pps: f64 = pps.parse().unwrap();
        let element: f64 = get(o, "element", 3.1e-9);
        let builtin: Vec<bool> = setup.loader.schemas().iter().map(|s| s.is_builtin()).collect();
        let base = CostModel::defaults(get(o, "cores", setup.loader.cores()), Tier::Nvme, &builtin);
        options.cost = Some(CostModel {
            iops: pps,
            bandwidth: get(o, "bandwidth", pps * 4096.0 * 4.0),
            memory_bandwidth: get(o, "memory_bandwidth", base.memory_bandwidth),
            access: get(o, "access", base.access),
            request: get(o, "request", base.request),
            serial: get(o, "serial", 0.0),
            bucket: get(o, "bucket", base.bucket),
            input: InputCost { decode: get(o, "decode", 0.0), decode_element: get(o, "decode_element", element * 0.8), cold: get(o, "cold", 0.0) },
            rows: RowCost { gather: get(o, "gather", base.rows.gather), admit: get(o, "admit", base.rows.admit) },
            outputs: setup
                .names
                .iter()
                .map(|name| {
                    let default = match name.as_str() {
                        "sample" => 8e-6,
                        "count" => 1e-7,
                        _ => 0.0,
                    };
                    OutputCost { curve: process_curve(o, name), resolve_element: get(o, &format!("resolve_element_{name}"), 0.0), resolve: get(o, &format!("resolve_{name}"), 0.0), process: get(o, &format!("process_{name}"), default), map_element: get(o, &format!("map_{name}"), get(o, "map_element", element * 0.2)), fill_element: get(o, &format!("fill_{name}"), get(o, "fill_element", 2e-10)) }
                })
                .collect(),
            convert: ["same", "widen", "reinterpret", "int_to_f32", "int_to_f64", "f32_to_f64"].map(|name| get(o, &format!("convert_{name}"), 0.0)),
            ..base
        });
    }
    if let Some(r) = resolve_cost(o, setup) {
        let from: Vec<String> = r.provenance.iter().map(|(part, source)| format!("{part}={source}")).collect();
        eprintln!("cost {}{}", from.join(" "), if online { " (online: compiler costs added)" } else { "" });
        options.cost = Some(r.model);
    } else if o.get("cost").map(String::as_str) == Some("none") {
        options.cost = None;
    }
    options
}

/// The record keys and the cost model `plan` would use, as JSON.
fn show_costs(args: &[String]) {
    let o = opts(args);
    let setup = build_loader(&o, Path::new(&args[0]));
    let source = costs::Source { paths: &setup.paths, backend: setup.backend, direct_io: true, page_size: 4096 };
    let keys = costs::keys(&source, &setup.keyed);
    let model = resolve_cost(&o, &setup).map(|r| {
        let m = &r.model;
        serde_json::json!({
            "provenance": r.provenance.iter().map(|(a, b)| serde_json::json!([a, b])).collect::<Vec<_>>(),
            "cores": m.cores, "iops": m.iops, "bandwidth": m.bandwidth, "memory_bandwidth": m.memory_bandwidth,
            "serial": m.serial, "bucket": m.bucket, "access": m.access, "request": m.request, "compile": m.compile,
            "input": {"decode": m.input.decode, "decode_element": m.input.decode_element, "cold": m.input.cold},
            "rows": {"gather": m.rows.gather, "admit": m.rows.admit},
            "outputs": m.outputs.iter().zip(&setup.names).map(|(c, n)| serde_json::json!({"name": n, "process": c.process,
                "map_element": c.map_element, "fill_element": c.fill_element, "resolve": c.resolve, "resolve_element": c.resolve_element,
                "curve": c.curve.iter().map(|b| serde_json::json!([b.max_elements, b.process, b.resolve])).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "convert": m.convert,
        })
    });
    // libc's strlen, the same code in every process: a reference to check
    // the signature against `pycellpack.loader.costs.signature`.
    let reference = costs::function_signature(libc::strlen as *const () as usize);
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({"keys": costs::describe(&keys), "model": model, "strlen_signature": reference})).unwrap());
}

fn plan(args: &[String]) {
    let o = opts(args);
    let setup = build_loader(&o, Path::new(&args[0]));
    let loader = &setup.loader;
    let a = stored(sequence(&args[1], &setup).as_ref());
    let options = plan_options(&o, &setup, get(&o, "cache", 0), get(&o, "online", 0u8) != 0);
    let t = Instant::now();
    let p = if let Some(path) = o.get("reuse") {
        let plan = Plan::open(path).unwrap();
        loader.replay(&plan, &a).unwrap();
        plan
    } else if get(&o, "online", 0u8) != 0 {
        let (p, stats) = loader.compile_online(&a, &options).unwrap();
        println!("online {stats:?}");
        p
    } else {
        loader.compile(&a, &options).unwrap()
    };
    let built = t.elapsed();
    assert_eq!(&loader.replay(&p, &a).unwrap(), p.summary());
    let s = if get(&o, "reprice", 0u8) != 0 {
        loader.replay_with_cost(&p, &a, options.cost.as_ref().expect("reprice needs a cost model")).unwrap()
    } else {
        p.write(&args[2]).unwrap();
        p.summary().clone()
    };
    let pages = loader.source().n_pages();
    println!(
        "plan accesses={} page_uses={} page_hits={} hit_fraction={:.4} page_residencies={} row_residencies={} block_reads={} row_hits={} peak_MB={:.1} concurrent_peak_MB={:.1} cache_fraction={:.4} requests={} build_s={:.2}",
        s.accesses,
        s.page_uses,
        s.page_hits,
        s.page_hits as f64 / s.page_uses.max(1) as f64,
        s.page_residencies,
        s.row_residencies,
        s.block_reads,
        s.row_hits,
        s.peak_bytes as f64 / 1e6,
        s.concurrent_peak_bytes as f64 / 1e6,
        p.cache_bytes() as f64 / (pages * 4096) as f64,
        s.requests,
        built.as_secs_f64()
    );
    println!(
        "counts decoded_rows={} decoded_elements={} cold_rows={} row_hits={} process_calls={:?} requests={} read_bytes={} residencies={} row_residencies={}",
        s.decoded_rows,
        s.decoded_elements,
        s.cold_rows,
        s.row_hits,
        s.process_calls,
        s.requests,
        s.read_bytes,
        s.page_residencies + s.row_residencies + s.block_reads,
        s.row_residencies,
    );
    if p.cost().is_some() {
        println!("quarter_work requests={:?} service_s(requests,bytes,cores,memory)={:?}", s.quarter_requests, s.quarter_service_seconds);
        println!("work mapped={:?} filled={:?} converted={:?} partial_calls={:?} partial_elements={:?} row_elements={} row_bytes={} admitted_elements={} admitted_bytes={}", s.mapped_elements, s.filled_elements, s.converted_elements, s.partial_calls, s.partial_elements, s.row_elements, s.row_bytes, s.admitted_elements, s.admitted_bytes);
        let q = s.quarter_accesses();
        let t = s.quarter_seconds;
        println!(
            "model quarter_misses={:?} quarters_cells/s=[{}] overall={:.0} steady={:.0} worker_s={:.2}",
            (0..4).map(|i| s.quarter_uses[i] - s.quarter_hits[i]).collect::<Vec<_>>(),
            t.iter().zip(q).map(|(x, n)| format!("{:.0}", n as f64 / x)).collect::<Vec<_>>().join(" "),
            s.accesses as f64 / t.iter().sum::<f64>(),
            q[1..].iter().sum::<u64>() as f64 / t[1..].iter().sum::<f64>(),
            s.worker_seconds,
        );
    }
}

/// Page plans under the misses objective against the independent serial
/// optimum.
fn curve(args: &[String]) {
    let o = opts(args);
    let setup = build_loader(&o, Path::new(&args[0]));
    let loader = &setup.loader;
    assert_eq!(loader.schemas().len(), 1, "curves run one schema");
    let catalog = loader.source().clone();
    let accesses = accesses(&args[1]);
    let trace = PageTrace::from_source(&catalog, &accesses).unwrap();
    println!(
        "trace accesses={} page_uses={} distinct_pages={} catalog_pages={} window={} catalog={:016x} access={:016x}",
        accesses.len(),
        trace.page_uses(),
        trace.distinct_pages(),
        catalog.n_pages(),
        loader.window(),
        catalog.fingerprint(),
        access_fingerprint(&accesses)
    );
    println!("fraction,capacity_pages,opt_hits,plan_hits,hit_gap_pp,hit_capture,build_s");
    for fraction in get(&o, "fractions", "0,0.002,0.01,0.02,0.05,0.1,0.25,0.5,1".to_string()).split(',') {
        let fraction: f64 = fraction.parse().unwrap();
        let capacity = (fraction * catalog.n_pages() as f64) as usize;
        let optimal = trace.optimum(capacity);
        let mut options = PlanOptions::new(capacity * catalog.page_size(), loader.window());
        options.threads = get(&o, "threads", 8);
        options.objective = PlanObjective::Misses;
        let start = Instant::now();
        let plan = loader.compile(&accesses, &options).unwrap();
        let s = plan.summary();
        assert_eq!(s.page_uses, optimal.page_uses);
        assert!(s.page_hits <= optimal.hits, "plan exceeds the independent OPT bound");
        println!(
            "{fraction},{capacity},{},{},{:.6},{:.6},{:.3}",
            optimal.hits,
            s.page_hits,
            100.0 * (optimal.hits - s.page_hits) as f64 / optimal.page_uses.max(1) as f64,
            if optimal.hits == 0 { 1.0 } else { s.page_hits as f64 / optimal.hits as f64 },
            start.elapsed().as_secs_f64()
        );
    }
}

/// Compiles online without loading.
fn online(args: &[String]) {
    let o = opts(args);
    let setup = build_loader(&o, Path::new(&args[0]));
    let seq = sequence(&args[1], &setup);
    let options = plan_options(&o, &setup, get(&o, "cache", 0), true);
    let t = Instant::now();
    let s = setup.loader.plan_online(seq.as_ref(), &options).unwrap();
    let secs = t.elapsed().as_secs_f64();
    println!(
        "online accesses={} wall_s={:.2} accesses/s={:.0} compile_accesses/s={:.0} horizon={} chunk={} shards={} model_accesses/s={:.0} page_hits={} page_admissions={} row_hits={} row_admissions={} block_reads={} requests={} row_price={:e} prices={:?}",
        s.accesses,
        secs,
        s.accesses as f64 / secs,
        s.accesses as f64 / s.compile_seconds.max(1e-9),
        s.horizon,
        s.chunk,
        s.shards,
        s.accesses as f64 / s.modeled_seconds.max(1e-12),
        s.page_hits,
        s.page_admissions,
        s.row_hits,
        s.row_admissions,
        s.block_reads,
        s.requests,
        s.row_price,
        s.prices
    );
}

fn run(args: &[String]) {
    let o = opts(args);
    let setup = build_loader(&o, Path::new(&args[0]));
    let (loader, names, catalog_s) = (&setup.loader, &setup.names, setup.catalog_s);
    let seq = sequence(&args[1], &setup);
    let a: &dyn Accesses = seq.as_ref();
    let cache: String = get(&o, "cache", "off".to_string());
    let plan: Planning = if let Some(p) = cache.strip_prefix("plan:") {
        Planning::Fixed(Arc::new(Plan::open(PathBuf::from(p)).unwrap()))
    } else if let Some(b) = cache.strip_prefix("online:") {
        loader.online(plan_options(&o, &setup, b.parse().unwrap(), true)).unwrap()
    } else {
        Planning::None
    };
    let world: u32 = get(&o, "world", 1);
    let ring = Arc::new(Ring::allocate(loader.ring_layout(get(&o, "slots", 16), world)).unwrap());
    let schemas = loader.schemas().len() as u64;
    let positions = a.len() / schemas;
    // Steady state: from the first consumed batch at or past `warm` (default
    // one quarter, i.e. after the first of four epochs) to the end.
    let warm = (positions as f64 * get(&o, "warm", 0.25)) as u64;
    let qlen = positions.div_ceil(4).max(1);
    let base = Instant::now();
    let warm_at = AtomicU64::new(u64::MAX);
    // First-consumed time of each quarter of the sequence (epochs for 4).
    let quarter: [AtomicU64; 4] = std::array::from_fn(|_| AtomicU64::new(u64::MAX));
    // When the last batch was consumed: the end of loading, before teardown.
    let last = AtomicU64::new(0);
    let run = loader.prepare(a, ring, plan).unwrap();
    let (stats, digest) = std::thread::scope(|s| {
        let consumers: Vec<_> = (0..world)
            .map(|rank| {
                let (run, warm_at, quarter, last) = (&run, &warm_at, &quarter, &last);
                s.spawn(move || {
                    // A few 8-byte probes per output, keyed by position, give
                    // an order-independent consistency check.
                    let (mut b, mut digest) = (rank as u64, 0u64);
                    while let Some(lease) = run.wait(rank, b).unwrap() {
                        let now = base.elapsed().as_nanos() as u64;
                        if lease.first_position() >= warm {
                            warm_at.fetch_min(now, Relaxed);
                        }
                        quarter[((lease.first_position() / qlen) as usize).min(3)].fetch_min(now, Relaxed);
                        for k in 0..lease.schemas() {
                            let bytes = lease.cell_bytes(k);
                            let region = lease.region(k);
                            for r in 0..lease.rows() {
                                let row = &region[r * bytes..(r + 1) * bytes];
                                let mut h = (lease.first_position() + r as u64) ^ bytes as u64;
                                let head = &row[..row.len().min(64)];
                                let tail = if row.len() >= SAMPLE * 8 { &row[SAMPLE * 4..SAMPLE * 4 + 64] } else { &[][..] };
                                for w in head.chunks_exact(8).chain(tail.chunks_exact(8)) {
                                    h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(27);
                                }
                                digest = digest.wrapping_add(h);
                            }
                        }
                        drop(lease);
                        last.fetch_max(base.elapsed().as_nanos() as u64, Relaxed);
                        b += world as u64;
                    }
                    digest
                })
            })
            .collect();
        let stats = run.run().unwrap();
        (stats, consumers.into_iter().fold(0u64, |d, c| d.wrapping_add(c.join().unwrap())))
    });
    let end = last.load(Relaxed);
    let steady = (positions - warm) as f64 / ((end - warm_at.load(Relaxed).min(end)) as f64 / 1e9).max(1e-9);
    let secs = stats.elapsed.as_secs_f64();
    let n = positions as f64;
    let c = &stats.counters;
    println!(
        "cache={cache} schemas={} workers={} depth={} cells/s={:.0} steady_cells/s={:.0} read_GB/s={:.2} KB/cell={:.2} reqs/cell={:.3} page_uses={} page_reads={} page_hits={} hit_frac={:.3} admitted={} row_hits={} rows_admitted={} block_reads={} waited={} starved={} window_stalls={} plan_stalls={} output_stalls={} peak_span={} cache_MB={:.1} private_MB={:.1} elapsed_s={:.2} setup_s={:.2} catalog_s={:.2} digest={:016x} cellpack={}",
        names.join(","),
        get(&o, "workers", 32),
        get(&o, "depth", 32),
        n / secs,
        steady,
        stats.io.read_bytes as f64 / secs / 1e9,
        stats.io.read_bytes as f64 / n / 1024.0,
        stats.io.requests as f64 / n,
        c.page_uses,
        c.page_reads,
        c.page_hits,
        c.page_hits as f64 / c.page_uses.max(1) as f64,
        c.admitted,
        c.row_hits,
        c.rows_admitted,
        c.block_reads,
        c.waited,
        c.starved,
        c.window_stalls,
        c.plan_stalls,
        c.output_stalls,
        stats.peak_span,
        stats.cache_bytes as f64 / 1e6,
        stats.private_bytes as f64 / 1e6,
        secs,
        stats.setup_elapsed.as_secs_f64(),
        catalog_s,
        digest,
        cellpack::scans::kernel_set()
    );
    let q: Vec<u64> = quarter.iter().map(|t| t.load(Relaxed)).collect();
    let bounds = [q[0], q[1], q[2], q[3], end];
    if let Some(o) = &stats.online {
        println!(
            "online compile_s={:.2} waiting_s={:.2} accesses={} chunks={} page_hits={} row_hits={} block_reads={} row_price={:e} prices={:?}",
            o.compile_seconds, o.waiting_seconds, o.accesses, o.chunks, o.page_hits, o.row_hits, o.block_reads, o.row_price, o.prices
        );
    }
    println!(
        "quarters_cells/s {}",
        (0..4).map(|i| format!("{:.0}", qlen as f64 / ((bounds[i + 1].saturating_sub(bounds[i])) as f64 / 1e9).max(1e-9))).collect::<Vec<_>>().join(" ")
    );
    println!("io_sizes(1,2,3-4,5-8,9-16,17-32,33+ pages)={:?}", stats.io.sizes);
    if let Some(p) = &stats.profile {
        let us = |d: std::time::Duration| d.as_secs_f64() * 1e6 / n;
        let sites: Vec<String> = p.sites.iter().map(|s| format!("{}={:.3}/{:.3}", s.name, us(s.time), us(s.self_time))).collect();
        println!("profile_us_per_cell(total/self) {}", sites.join(" "));
        for s in p.sites.iter().filter(|s| s.hist.is_some()) {
            let h = s.hist.as_ref().unwrap();
            let calls: u64 = h.iter().map(|b| b.1).sum();
            let q = |f: f64| -> u64 {
                let mut acc = 0;
                h.iter().find(|b| { acc += b.1; acc as f64 >= f * calls as f64 }).map_or(0, |b| b.0)
            };
            let (p50, p99) = (q(0.5), q(0.99));
            println!("profile_hist {} calls={} mean_us={:.2} p50_ge_us={:.2} p99_ge_us={:.2}", s.name, s.calls, s.time.as_secs_f64() * 1e6 / s.calls.max(1) as f64, p50 as f64 / 1e3, p99 as f64 / 1e3);
        }
        if let Some(m) = p.site("map") {
            for d in &m.details {
                println!("schema_profile id={} name={} map_calls={} map_s={:.9}", d.key, names[d.key as usize], d.calls, d.time.as_secs_f64());
            }
        }
        if let Some(pr) = p.site("process") {
            for d in &pr.details {
                let (k, band, partial) = (d.key as usize / 2 / p.bands, d.key as usize / 2 % p.bands, d.key % 2 == 1);
                println!("process_band name={} upper={} partial={} calls={} s={:.9}", names[k], PROCESS_BOUNDS[band], partial, d.calls, d.time.as_secs_f64());
            }
        }
    }
}

fn ioraw(args: &[String]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    let o = opts(args);
    let c = source(Path::new(&args[0]), get(&o, "files", 0), None, &ids(Path::new(&args[0]), &o));
    let a = accesses(&args[1]);
    let workers: usize = get(&o, "workers", 32);
    let depth: usize = get(&o, "depth", 64);
    // align=A reads each record's exact bytes widened to A-byte boundaries
    // (O_DIRECT allows the device's logical block size); default whole pages.
    let align: u64 = get(&o, "align", 0);
    let extents: Vec<(i32, u64, usize)> = a
        .iter()
        .map(|&cell| match std::num::NonZeroU64::new(align) {
            None => {
                let (fd, off, pages) = c.cell_pages(cell).unwrap();
                (fd, off, pages * 4096)
            }
            Some(align) => {
                let (fd, off, len) = c.cell_bytes(cell).unwrap();
                let start = off / align * align.get();
                (
                    fd,
                    start,
                    ((off + len as u64).div_ceil(align.get()) * align.get() - start) as usize,
                )
            }
        })
        .collect();
    let max_pages = extents
        .iter()
        .map(|e| e.2.div_ceil(4096))
        .max()
        .unwrap_or(1)
        .max(1);
    let next = AtomicU64::new(0);
    let bytes = AtomicU64::new(0);
    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..workers {
            let (extents, next, bytes) = (&extents, &next, &bytes);
            s.spawn(move || {
                let mut ring = io_uring::IoUring::new(depth.next_power_of_two() as u32).unwrap();
                let layout =
                    std::alloc::Layout::from_size_align(depth * max_pages * 4096, 4096).unwrap();
                // SAFETY: nonzero, aligned layout; freed below after all CQEs.
                let buf = unsafe { std::alloc::alloc(layout) };
                let mut free: Vec<usize> = (0..depth).collect();
                let (mut inflight, mut mine, mut done) = (0usize, Vec::<u64>::new(), false);
                loop {
                    while !done && !free.is_empty() {
                        if mine.is_empty() {
                            let n = next.fetch_add(64, Ordering::Relaxed);
                            if n >= extents.len() as u64 {
                                done = true;
                                break;
                            }
                            mine.extend((n..(n + 64).min(extents.len() as u64)).rev());
                        }
                        let i = mine.pop().unwrap() as usize;
                        let (fd, off, bytes_len) = extents[i];
                        if bytes_len == 0 {
                            continue;
                        }
                        let slot = free.pop().unwrap();
                        let e = io_uring::opcode::Read::new(
                            io_uring::types::Fd(fd),
                            // SAFETY: the slot lies inside the buffer.
                            unsafe { buf.add(slot * max_pages * 4096) },
                            bytes_len as u32,
                        )
                        .offset(off)
                        .build()
                        .user_data(slot as u64);
                        // SAFETY: the slot's buffer is untouched until its CQE.
                        unsafe { ring.submission().push(&e).unwrap() };
                        inflight += 1;
                    }
                    if inflight == 0 {
                        break;
                    }
                    ring.submit_and_wait(1).unwrap();
                    for cqe in ring.completion() {
                        assert!(cqe.result() > 0);
                        bytes.fetch_add(cqe.result() as u64, Ordering::Relaxed);
                        free.push(cqe.user_data() as usize);
                        inflight -= 1;
                    }
                }
                // SAFETY: every submitted read has completed.
                unsafe { std::alloc::dealloc(buf, layout) };
            });
        }
    });
    let secs = t.elapsed().as_secs_f64();
    println!(
        "ioraw workers={workers} depth={depth} cells/s={:.0} read_GB/s={:.2} KB/cell={:.2}",
        a.len() as f64 / secs,
        bytes.load(Ordering::Relaxed) as f64 / secs / 1e9,
        bytes.load(Ordering::Relaxed) as f64 / a.len() as f64 / 1024.0
    );
}

/// One allowed CPU per physical core, alternating NUMA nodes.
/// One CPU per physical core, quietest cores first: per-CPU busy time is
/// sampled from /proc/stat, so cores shared with other tenants' busy threads
/// come last (a worker on a contended core holds back the whole window).
fn physical_cpus() -> Vec<usize> {
    let read = |p: String| {
        std::fs::read_to_string(p)
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let busy = || -> HashMap<usize, u64> {
        let stat = std::fs::read_to_string("/proc/stat").unwrap_or_default();
        stat.lines()
            .filter_map(|l| {
                let mut f = l.split_whitespace();
                let cpu: usize = f.next()?.strip_prefix("cpu")?.parse().ok()?;
                let t: Vec<u64> = f.filter_map(|x| x.parse().ok()).collect();
                // user nice system idle iowait irq softirq steal
                Some((
                    cpu,
                    t.iter().take(8).sum::<u64>()
                        - t.get(3).copied().unwrap_or(0)
                        - t.get(4).copied().unwrap_or(0),
                ))
            })
            .collect()
    };
    let before = busy();
    std::thread::sleep(std::time::Duration::from_millis(300));
    let after = busy();
    let load = |cpu: usize| {
        after
            .get(&cpu)
            .copied()
            .unwrap_or(0)
            .saturating_sub(before.get(&cpu).copied().unwrap_or(0))
    };
    let mut cores: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for cpu in 0..1024usize {
        // SAFETY: plain sched_getaffinity on a zeroed set.
        let allowed = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) == 0
                && libc::CPU_ISSET(cpu, &set)
        };
        if allowed {
            let core = read(format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id"));
            let pkg = read(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
            ));
            cores.entry((pkg, core)).or_default().push(cpu);
        }
    }
    let mut ranked: Vec<(u64, usize)> = cores
        .values()
        .map(|siblings| {
            let quietest = *siblings.iter().min_by_key(|&&c| load(c)).unwrap();
            (siblings.iter().map(|&c| load(c)).sum(), quietest)
        })
        .collect();
    ranked.sort();
    ranked.into_iter().map(|(_, cpu)| cpu).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    let required = match args.first().map(String::as_str) {
        Some("gen") => 5,
        Some("plan") => 4,
        Some("online" | "run" | "curve" | "ioraw") => 3,
        Some("costs") => 2,
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };
    if args.len() < required || args[1..required].iter().any(|a| a.contains('=')) {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    match args[0].as_str() {
        "gen" => generate(&args[1..]),
        "plan" => plan(&args[1..]),
        "online" => online(&args[1..]),
        "run" => run(&args[1..]),
        "ioraw" => ioraw(&args[1..]),
        "curve" => curve(&args[1..]),
        "costs" => show_costs(&args[1..]),
        _ => unreachable!(),
    }
}
