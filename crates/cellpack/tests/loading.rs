use std::ffi::c_void;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cellpack::store::{CpkOptions, CpkWriter};
use cellpack::format::SourceLayout;
use cellpack::loader::*;
use cellpack::opt::*;
use cellpack::scans::{CsrRef, DenseRef, EncodeOptions, Indices, LayoutChoice, Thresholds, ValueChoice};

const FEATURES: usize = 320;
const GENES: usize = 300;
/// Positions per slot, and the default window and reuse gap in slots.
const SLOT: u64 = 16;
const WINDOW: u64 = 8;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Fixture {
    root: PathBuf,
    source: Arc<DataSource>,
    expected: Vec<Vec<f32>>,
}

fn cpk_options(renumber: bool, chunk_rows: usize) -> CpkOptions {
    CpkOptions {
        encode: EncodeOptions {
            renumber,
            thresholds: Thresholds { indices: 0.6, values: 0.6 },
            max_lanes: 32,
            lane_min: 32,
            column_modes: Some(cellpack::scans::DEFAULT_MODE_SHARE),
            layout: LayoutChoice::Auto,
            values: ValueChoice::Auto,
        },
        chunk_rows,
        page_size: 4096,
    }
}

/// Writes sparse u16 rows to a `.cpk` file.
fn write_sparse(path: &PathBuf, rows: &[Vec<(u32, u16)>], renumber: bool, chunk_rows: usize) {
    let mut indptr = vec![0u64];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for r in rows {
        for &(g, v) in r {
            indices.push(g);
            data.extend_from_slice(&v.to_le_bytes());
        }
        indptr.push(indices.len() as u64);
    }
    let mut w = CpkWriter::create(path, GENES as u32, Dtype::U16, SourceLayout::Csr, cpk_options(renumber, chunk_rows)).unwrap();
    w.append(CsrRef { n_genes: GENES as u32, value_width: 2, indptr: &indptr, indices: Indices::U32(&indices), data: &data }).unwrap();
    w.finish().unwrap();
}

/// Writes row-major dense rows to a `.cpk` file.
fn write_dense(path: &PathBuf, data: &[u8], cols: usize, dtype: Dtype, chunk_rows: usize) {
    let mut w = CpkWriter::create(path, cols as u32, dtype, SourceLayout::Dense, cpk_options(true, chunk_rows)).unwrap();
    w.append(DenseRef { n_genes: cols as u32, value_width: dtype.width(), data }).unwrap();
    w.finish().unwrap();
}

impl Fixture {
    /// Two CSR matrices (one renumbered in several chunks, with multi-page
    /// and empty cells), then a dense matrix whose rows are stored dense.
    fn new(direct: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let base = if direct {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tmp/cellpack-tests")
        } else {
            std::env::temp_dir()
        };
        let root = base.join(format!("cellpack-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir_all(&root).unwrap();
        let mut rng = Rng(0x5eed_1234_abcd_0001);
        let map: Vec<u32> = (0..GENES as u32).map(|g| if g % 7 == 3 { cellpack::scans::UNMAPPED } else { (g * 13 + 5) % FEATURES as u32 }).collect();
        let mut expected = Vec::new();
        let mut builder = DataSourceBuilder::new(FEATURES, 4096).unwrap().direct_io(direct);
        for (renumber, cells, chunk_rows) in [(true, 150usize, 64usize), (false, 90, 4096)] {
            let mut rows = Vec::new();
            for c in 0..cells {
                let nnz = match c % 10 {
                    0 => 0,
                    1 => GENES,
                    _ => rng.below(120) as usize,
                };
                let mut genes: Vec<u32> = (0..GENES as u32).collect();
                for i in 0..nnz {
                    let j = i + rng.below((GENES - i) as u64) as usize;
                    genes.swap(i, j);
                }
                let mut row: Vec<(u32, u16)> = genes[..nnz].iter().map(|&g| (g, 1 + rng.below(if c % 10 == 1 { 60000 } else { 9 }) as u16)).collect();
                row.sort_unstable();
                let mut want = vec![0.0; FEATURES];
                for &(g, v) in &row {
                    if map[g as usize] != cellpack::scans::UNMAPPED {
                        want[map[g as usize] as usize] = v as f32;
                    }
                }
                expected.push(want);
                rows.push(row);
            }
            let path = root.join(format!("sparse-{renumber}.cpk"));
            write_sparse(&path, &rows, renumber, chunk_rows);
            builder.add_cpk(&path, &map).unwrap();
        }

        let (rows, cols) = (70usize, 1000usize);
        let mut raw = vec![0u8; rows * cols * 2];
        let mut seen = vec![false; FEATURES];
        let dense_map: Vec<u32> = (0..cols as u32)
            .map(|c| {
                let m = c % FEATURES as u32;
                if c % 5 == 0 || seen[m as usize] {
                    cellpack::scans::UNMAPPED
                } else {
                    seen[m as usize] = true;
                    m
                }
            })
            .collect();
        for r in 0..rows {
            let mut want = vec![0.0; FEATURES];
            for c in 0..cols {
                let v = (rng.below(2000) as i16) - 1000;
                raw[(r * cols + c) * 2..(r * cols + c) * 2 + 2].copy_from_slice(&v.to_le_bytes());
                if dense_map[c] != cellpack::scans::UNMAPPED {
                    want[dense_map[c] as usize] = v as f32;
                }
            }
            expected.push(want);
        }
        let path = root.join("dense.cpk");
        write_dense(&path, &raw, cols, Dtype::I16, 32);
        builder.add_cpk(&path, &dense_map).unwrap();
        Self {
            root,
            source: Arc::new(builder.build().unwrap()),
            expected,
        }
    }

    /// Shuffled epochs over every cell.
    fn accesses(&self, epochs: usize, seed: u64) -> Vec<CellId> {
        let n = self.source.n_cells() as usize;
        let mut rng = Rng(seed);
        let mut out = Vec::new();
        for _ in 0..epochs {
            let mut e: Vec<CellId> = (0..n as CellId).collect();
            for i in (1..n).rev() {
                e.swap(i, rng.below(i as u64 + 1) as usize);
            }
            out.extend(e);
        }
        out
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn config(backend: IoBackend) -> LoaderConfig {
    LoaderConfig { workers: 4, backend, max_inflight_per_worker: 8, claim_batch: 3, slot_cells: SLOT, window: Some(WINDOW), ..Default::default() }
}

fn panel() -> Vec<u32> {
    (0..FEATURES as u32).collect()
}

/// The built-in dense f32 output over every standard id.
fn dense_schema() -> OutputSchema {
    OutputSchema::dense(1, panel(), Dtype::F32).unwrap()
}

fn loader(f: &Fixture, config: LoaderConfig, schema: OutputSchema) -> Loader {
    Loader::new(f.source.clone(), vec![schema], config).unwrap()
}

fn ring(loader: &Loader, world: u32, slots: u32) -> Arc<Ring> {
    Arc::new(Ring::allocate(loader.ring_layout(slots, world)).unwrap())
}

/// Runs with the built-in kernel and checks every row against the fixture.
fn check(f: &Fixture, accesses: &[CellId], config: LoaderConfig, plan: Option<Arc<Plan>>, world: u32) -> RunStats {
    check_with(f, &accesses, config, plan.into(), world)
}

fn check_with(f: &Fixture, accesses: &dyn Accesses, config: LoaderConfig, planning: Planning, world: u32) -> RunStats {
    let mut cells = vec![0; accesses.len() as usize];
    accesses.fill(0, &mut cells).unwrap();
    let loader = loader(f, config, dense_schema());
    let run = loader.prepare(accesses, ring(&loader, world, WINDOW as u32 + 2), planning).unwrap();
    std::thread::scope(|s| {
        for rank in 0..world {
            let (run, expected, cells) = (&run, &f.expected, &cells);
            s.spawn(move || {
                let mut b = rank as u64;
                while let Some(batch) = run.wait(rank, b).unwrap() {
                    let bytes = batch.cell_bytes(0);
                    assert_eq!(bytes, FEATURES * 4);
                    for r in 0..batch.rows() {
                        let want = &expected[cells[(batch.first_position() + r as u64) as usize] as usize];
                        let got: Vec<f32> = batch.region(0)[r * bytes..(r + 1) * bytes].chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
                        assert_eq!(&got, want, "access {}", batch.first_position() + r as u64);
                    }
                    b += world as u64;
                }
            });
        }
        run.run().unwrap()
    })
}

fn cost(iops: f64, bucket: u64) -> CostModel {
    CostModel {
        cores: 4.0,
        iops,
        bandwidth: iops * 4096.0 * 4.0,
        memory_bandwidth: 1e12,
        serial: 0.0,
        bucket,
        access: 2e-6,
        request: 3e-6,
        compile: 0.0,
        input: InputCost { decode: 0.0, decode_element: 3e-9, cold: 0.0 },
        rows: RowCost { gather: 1e-10, admit: 0.0 },
        outputs: vec![OutputCost { curve: Vec::new(), resolve_element: 0.0, resolve: 0.0, process: 1e-6, map_element: 1e-9, fill_element: 1e-10 }],
        convert: [0.0; CONVERSIONS],
    }
}

fn plan_with(f: &Fixture, accesses: &[CellId], options: &PlanOptions) -> Arc<Plan> {
    let l = loader(f, config(IoBackend::Uring), dense_schema());
    Arc::new(l.compile(accesses, options).unwrap())
}

fn plan(f: &Fixture, accesses: &[CellId], cache_bytes: usize) -> Arc<Plan> {
    let mut options = PlanOptions::new(cache_bytes, WINDOW);
    options.threads = 3;
    plan_with(f, accesses, &options)
}

/// A run serves exactly the plan: every hit and read the replay counts.
fn assert_follows(stats: &RunStats, s: &PlanSummary) {
    let c = &stats.counters;
    assert_eq!(c.accesses, s.accesses);
    assert_eq!(c.page_uses, s.page_uses, "{c:?} {s:?}");
    assert_eq!(c.page_hits, s.page_hits, "{c:?} {s:?}");
    assert_eq!(c.page_reads * 4096, s.read_bytes, "{c:?} {s:?}");
    assert_eq!(c.row_hits, s.row_hits);
    assert_eq!(c.rows_admitted, s.row_residencies);
    assert_eq!(c.block_reads, s.block_reads);
    // Reads split only where a worker waits for private frames.
    if c.starved == 0 {
        assert_eq!(stats.io.requests, s.requests);
    }
}

#[test]
fn every_backend_writes_the_mapped_rows() {
    let f = Fixture::new(false);
    let accesses = f.accesses(3, 7);
    for backend in [IoBackend::Pread, IoBackend::Uring] {
        let stats = check(&f, &accesses, config(backend), None, 1);
        assert_eq!(stats.counters.page_hits, 0);
        assert_eq!(stats.counters.page_reads, stats.counters.page_uses);
        for pages in [2usize, 24, 4096] {
            let p = plan(&f, &accesses, pages << 12);
            assert!(p.summary().concurrent_peak_bytes <= (pages << 12) as u64);
            let stats = check(&f, &accesses, config(backend), Some(p.clone()), 2);
            assert_follows(&stats, p.summary());
            if pages == 4096 {
                assert!(p.summary().page_hits > 0);
                assert!(stats.counters.page_reads < p.summary().page_uses / 2);
            }
        }
    }
}

#[test]
fn compiled_plans_respect_the_budget_and_the_serial_optimum() {
    let f = Fixture::new(false);
    let accesses = f.accesses(3, 21);
    let optimum = PageTrace::from_source(&f.source, &accesses).unwrap();
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    for pages in [8usize, 64, 4096] {
        for objective in [PlanObjective::Misses, PlanObjective::Time] {
            let mut options = PlanOptions::new(pages << 12, WINDOW);
            options.objective = objective;
            options.cost = (objective == PlanObjective::Time).then(|| cost(1e5, 64));
            let p = Arc::new(l.compile(&accesses, &options).unwrap());
            let s = p.summary().clone();
            assert_eq!(l.replay(&p, &accesses).unwrap(), s);
            assert!(s.page_hits <= optimum.optimum(pages).hits);
            assert!(s.peak_bytes <= s.concurrent_peak_bytes && s.concurrent_peak_bytes <= (pages << 12) as u64);
            let stats = check(&f, &accesses, config(IoBackend::Uring), Some(p), 1);
            assert_follows(&stats, &s);
        }
    }
}

#[test]
fn replay_partitions_short_traces_into_the_exact_quarters() {
    let f = Fixture::new(false);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    for n in [0, 1, 2, 3, 4, 5, 7, 16, 17] {
        let accesses = vec![1; n];
        let mut expected = [0u64; 4];
        for at in 0..n { expected[at * 4 / n] += 1; }
        for bucket in [1, 3, 16, 4096] {
            let mut options = PlanOptions::new(0, WINDOW);
            options.threads = 1;
            options.cost = Some(cost(1e6, bucket));
            let plan = l.compile(&accesses, &options).unwrap();
            let summary = plan.summary();
            assert_eq!(summary.quarter_accesses(), expected);
            let pages = if n == 0 { 0 } else { summary.page_uses / n as u64 };
            assert_eq!(summary.quarter_uses, expected.map(|n| n * pages));
            assert_eq!(&l.replay(&plan, &accesses).unwrap(), summary);
        }
    }
}

#[test]
fn repeated_cells_stay_resident_from_their_first_read() {
    // Reuses within the reuse gap are kept like any other: a cell read
    // again and again is read once.
    let f = Fixture::new(false);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    let gap = (WINDOW * SLOT) as usize;
    for length in [1, gap, gap + 2, 3 * gap, (1 << 16) + gap + 2] {
        let accesses = vec![1; length];
        let optimum = PageTrace::from_source(&f.source, &accesses).unwrap().optimum(64);
        for threads in [1, 4] {
            let mut options = PlanOptions::new(64 << 12, WINDOW);
            options.threads = threads;
            options.horizon = Some((length as u64).div_ceil(SLOT).max(WINDOW));
            let offline = l.compile(&accesses, &options).unwrap();
            let (online, _) = l.compile_online(&accesses, &options).unwrap();
            for plan in [&offline, &online] {
                assert_eq!(plan.summary().page_hits, optimum.hits, "length={length} threads={threads}");
                assert_eq!(&l.replay(plan, &accesses).unwrap(), plan.summary());
            }
            let stats = check(&f, &accesses, config(IoBackend::Uring), Some(Arc::new(offline)), 1);
            assert_eq!(stats.counters.page_hits, optimum.hits);
        }
    }
}

#[test]
fn miss_objective_is_independent_of_throughput_calibration() {
    let f = Fixture::new(false);
    let accesses = f.accesses(4, 29);
    let mut options = PlanOptions::new(16 * 4096, 2);
    options.objective = PlanObjective::Misses;
    let base = plan_with(&f, &accesses, &options);
    for rate in [1.0, 1e6, 1e12] {
        options.cost = Some(cost(rate, 13));
        let p = plan_with(&f, &accesses, &options);
        assert_eq!(p.residencies(), base.residencies());
    }
}

#[test]
fn row_residencies_serve_accesses_within_the_budget() {
    let f = Fixture::new(false);
    let accesses = f.accesses(4, 31);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    for policy in [RowPolicy::Incremental, RowPolicy::Global] {
        for cache in [24 << 12, 512 << 12] {
            let mut options = PlanOptions::new(cache, WINDOW);
            options.threads = 3;
            // A fast device and slow decoding: workers bound the run, so
            // cached rows pay.
            let mut c = cost(1e9, 64);
            c.input.decode = 1e-5;
            options.cost = Some(c);
            options.rows = true;
            options.row_policy = policy;
            let p = Arc::new(l.compile(&accesses, &options).unwrap());
            let s = p.summary().clone();
            assert_eq!(l.replay(&p, &accesses).unwrap(), s);
            assert!(s.concurrent_peak_bytes <= cache as u64);
            assert!(cache < 512 << 12 || s.row_hits > 0, "{policy:?} {s:?}");
            let stats = check(&f, &accesses, config(IoBackend::Uring), Some(p.clone()), 1);
            assert_follows(&stats, &s);
        }
    }
}

unsafe extern "C" fn nnz_or_fail(call: *const CsrRow) -> i32 {
    // SAFETY: the loader passes valid pointers, f32 values and a 16-byte output.
    unsafe {
        let c = &*call;
        if c.context.position == *c.context.user_data.cast::<u64>() {
            return 7;
        }
        let sum: f32 = (0..c.nnz as usize).map(|i| *c.values.cast::<f32>().add(i)).sum();
        c.out.cast::<u64>().write_unaligned(c.nnz);
        c.out.add(8).cast::<f32>().write_unaligned(sum);
    }
    0
}

unsafe extern "C" fn dense_or_fail(call: *const DenseRow) -> i32 {
    // SAFETY: as above.
    unsafe {
        let c = &*call;
        if c.context.position == *c.context.user_data.cast::<u64>() {
            return 7;
        }
        let v = std::slice::from_raw_parts(c.values.cast::<f32>(), c.out_dim as usize);
        c.out.cast::<u64>().write_unaligned(u64::MAX);
        c.out.add(8).cast::<f32>().write_unaligned(v.iter().sum());
    }
    0
}

/// Process functions writing 12 of 16 bytes; both fail with status 7 at
/// position `fail_at`.
fn custom_schema(fail_at: &'static u64) -> OutputSchema {
    // SAFETY: both functions are thread-safe and write 12 bytes of 16.
    let process = unsafe { ProcessFn::new(nnz_or_fail, dense_or_fail, (fail_at as *const u64).cast::<c_void>(), Arc::new(())) };
    OutputSchema::process(1, panel(), Dtype::F32, 16, process, 2, 0).unwrap()
}

#[test]
fn plan_files_round_trip_and_bind_to_their_sequence() {
    let f = Fixture::new(false);
    let accesses = f.accesses(2, 11);
    let built = plan(&f, &accesses, 64 << 12);
    let path = f.root.join("seq.plan");
    built.write(&path).unwrap();
    let opened = Arc::new(Plan::open(&path).unwrap());
    assert_eq!(opened.residencies(), built.residencies());
    let a = check(&f, &accesses, config(IoBackend::Uring), Some(built.clone()), 1);
    let b = check(&f, &accesses, config(IoBackend::Uring), Some(opened.clone()), 1);
    assert_eq!(opened.summary(), built.summary());
    assert_eq!(a.counters.page_hits, b.counters.page_hits);

    let other = f.accesses(2, 12);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    assert!(matches!(l.prepare(&other, ring(&l, 1, 2), Planning::Fixed(opened.clone())), Err(Error::Plan(_))));
    // A window beyond the plan's reuse gap could overwrite frames in use.
    let mut wide = config(IoBackend::Uring);
    wide.window = Some(WINDOW + 1);
    let l = loader(&f, wide, dense_schema());
    assert!(matches!(l.prepare(&accesses, ring(&l, 1, 2), Planning::Fixed(opened)), Err(Error::Config(_))));
    // A damaged file never binds.
    let mut bytes = std::fs::read(&path).unwrap();
    let n = bytes.len();
    bytes[n / 2] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    assert!(Plan::open(&path).is_err());
}

#[test]
fn plan_costs_must_be_finite_and_rates_positive() {
    let f = Fixture::new(false);
    let accesses = f.accesses(1, 17);
    let mut options = PlanOptions::new(4096, 1);
    let base = cost(1e6, 1024);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    let mut short = base.clone();
    short.outputs.clear();
    for bad in [
        CostModel { bucket: 0, ..base.clone() },
        CostModel { iops: f64::INFINITY, ..base.clone() },
        CostModel { cores: 0.0, ..base.clone() },
        CostModel { request: -1.0, ..base.clone() },
        short,
    ] {
        options.cost = Some(bad);
        assert!(l.compile(&accesses, &options).is_err());
    }
    options.cost = Some(base);
    let p = l.compile(&accesses, &options).unwrap();
    assert!(p.summary().quarter_seconds.iter().all(|t| t.is_finite() && *t > 0.0));
    options.cost = None;
    options.rows = true;
    assert!(l.compile(&accesses, &options).is_err());
}

#[test]
fn multi_page_residencies_preserve_every_value_under_churn() {
    let mut f = Fixture::new(false);
    let (rows, columns) = (129usize, 2300usize);
    let mut bytes = vec![0u8; rows * columns * 4];
    let mapping: Vec<u32> = (0..columns).map(|c| if c % 7 == 0 && c / 7 < FEATURES { (c / 7) as u32 } else { cellpack::scans::UNMAPPED }).collect();
    f.expected.clear();
    for row in 0..rows {
        let mut expected = vec![0.0; FEATURES];
        for (column, &mapped) in mapping.iter().enumerate() {
            // Distinct values: every row is stored dense and raw, over 2 pages.
            let value = (row * 4096 + column) as f32 + 0.5;
            bytes[(row * columns + column) * 4..(row * columns + column) * 4 + 4].copy_from_slice(&value.to_le_bytes());
            if mapped != cellpack::scans::UNMAPPED {
                expected[mapped as usize] = value;
            }
        }
        f.expected.push(expected);
    }
    let path = f.root.join("churn.cpk");
    write_dense(&path, &bytes, columns, Dtype::F32, 64);
    let mut builder = DataSourceBuilder::new(FEATURES, 4096).unwrap();
    builder.add_cpk(&path, &mapping).unwrap();
    f.source = Arc::new(builder.build().unwrap());
    assert!(f.source.max_cell_pages() >= 3);
    for seed in 0..4 {
        let accesses = f.accesses(8, seed + 1);
        let options = PlanOptions::new(12 * 4096, 2);
        let p = plan_with(&f, &accesses, &options);
        let cfg = LoaderConfig { workers: 8, backend: IoBackend::Uring, max_inflight_per_worker: 8, claim_batch: 3, slot_cells: SLOT, ..Default::default() };
        let stats = check(&f, &accesses, cfg, Some(p.clone()), 2);
        assert!(stats.counters.admitted > 0);
        assert_follows(&stats, p.summary());
    }
}

#[test]
fn direct_io_on_shared_storage() {
    let f = Fixture::new(true);
    let accesses = f.accesses(2, 5);
    let p = plan(&f, &accesses, 32 << 12);
    check(&f, &accesses, config(IoBackend::Uring), Some(p), 1);
    check(&f, &accesses, config(IoBackend::Pread), None, 1);
}

#[test]
fn ranks_consume_interleaved_batches_under_backpressure() {
    let f = Fixture::new(false);
    let accesses = f.accesses(2, 3);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    let run = l.prepare(&accesses, ring(&l, 3, 2), Planning::None).unwrap();
    let seen = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for rank in 0..3u32 {
            let (run, seen) = (&run, &seen);
            s.spawn(move || {
                let mut b = rank as u64;
                while let Some(batch) = run.wait(rank, b).unwrap() {
                    assert_eq!(batch.index() % 3, rank as u64);
                    if rank == 1 {
                        std::thread::sleep(std::time::Duration::from_micros(300));
                    }
                    seen.fetch_add(batch.rows(), Ordering::Relaxed);
                    b += 3;
                }
            });
        }
        let stats = run.run().unwrap();
        assert!(stats.counters.window_stalls + stats.counters.output_stalls > 0);
    });
    assert_eq!(seen.load(Ordering::Relaxed), accesses.len());
}

#[test]
fn process_failure_aborts_consumers_and_reports_the_status() {
    let f = Fixture::new(false);
    let accesses = f.accesses(1, 9);
    static FAIL: u64 = 37;
    let l = loader(&f, config(IoBackend::Uring), custom_schema(&FAIL));
    let p = Arc::new(l.compile(&accesses, &PlanOptions::new(32 << 12, WINDOW)).unwrap());
    let run = l.prepare(&accesses, ring(&l, 1, 2), Planning::Fixed(p)).unwrap();
    std::thread::scope(|s| {
        let consumer = s.spawn(|| {
            let mut b = 0;
            loop {
                match run.wait(0, b) {
                    Ok(Some(batch)) => {
                        for r in 0..batch.rows() {
                            let row = &batch.region(0)[r * 16..r * 16 + 12];
                            let want = &f.expected[accesses[(batch.first_position() + r as u64) as usize] as usize];
                            let sum = f32::from_ne_bytes(row[8..12].try_into().unwrap());
                            assert!((sum - want.iter().sum::<f32>()).abs() <= 1e-3 * sum.abs().max(1.0));
                        }
                        b += 1;
                    }
                    Ok(None) => return false,
                    Err(Error::Aborted) => return true,
                    Err(e) => panic!("{e}"),
                }
            }
        });
        assert!(matches!(run.run(), Err(Error::Process(7))));
        assert!(consumer.join().unwrap());
    });
}

#[test]
fn a_consumer_abort_stops_the_loader() {
    let f = Fixture::new(false);
    let accesses = f.accesses(4, 1);
    static NEVER: u64 = u64::MAX;
    let l = loader(&f, config(IoBackend::Uring), custom_schema(&NEVER));
    let mut options = PlanOptions::new(32 << 12, WINDOW);
    options.threads = 2;
    let run = l.prepare(&accesses, ring(&l, 1, 2), l.online(options).unwrap()).unwrap();
    std::thread::scope(|s| {
        s.spawn(|| {
            run.wait(0, 0).unwrap().unwrap();
            run.cancel();
        });
        assert!(matches!(run.run(), Err(Error::Aborted)));
    });
}

#[test]
fn contiguous_blocks_replay_and_load_without_crossing_files() {
    let f = Fixture::new(true);
    let accesses = f.accesses(5, 323);
    let bytes = f.source.n_pages() as usize * 4096 * 4;
    let mut options = PlanOptions::new(bytes, WINDOW);
    options.threads = 2;
    options.cost = Some(CostModel { iops: 1000.0, bandwidth: 1e10, ..cost(1000.0, 128) });
    options.block_pages = Some(8);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    let p = l.compile(&accesses, &options).unwrap();
    assert!(p.summary().block_reads > 0);
    assert_eq!(&l.replay(&p, &accesses).unwrap(), p.summary());
    let path = f.root.join("blocks.plan");
    p.write(&path).unwrap();
    let p = Arc::new(Plan::open(&path).unwrap());
    for backend in [IoBackend::Pread, IoBackend::Uring] {
        let stats = check(&f, &accesses, config(backend), Some(p.clone()), 1);
        assert!(stats.counters.block_reads > 0);
        assert_follows(&stats, p.summary());
    }
    options.rows = true;
    options.cost.as_mut().unwrap().input.decode = 1e-3;
    let mixed = Arc::new(l.compile(&accesses, &options).unwrap());
    assert!(mixed.summary().row_residencies > 0);
    assert_eq!(&l.replay(&mixed, &accesses).unwrap(), mixed.summary());
    let stats = check(&f, &accesses, config(IoBackend::Uring), Some(mixed.clone()), 1);
    assert_follows(&stats, mixed.summary());
}

#[test]
fn rows_take_whole_frames() {
    let f = Fixture::new(false);
    // Small cells (few non-zeros), far enough apart to outlast the gap.
    let small: Vec<CellId> = (0..f.source.n_cells() as CellId).filter(|&c| matches!(c % 10, 2..=9) && f.expected[c as usize].iter().filter(|v| **v != 0.0).count() < 40).take(2).collect();
    let accesses: Vec<CellId> = (0..64).map(|i| small[i % 2]).collect();
    let mut c = cost(1e6, 8);
    c.input.decode = 1e-3;
    let mut cfg = config(IoBackend::Uring);
    cfg.slot_cells = 1;
    cfg.window = Some(1);
    let l = loader(&f, cfg, dense_schema());
    // Below a frame, nothing is cached.
    let mut options = PlanOptions::new(2048, 1);
    options.rows = true;
    options.cost = Some(c);
    assert_eq!(l.compile(&accesses, &options).unwrap().summary().row_residencies, 0);
    options.cache_bytes = 2 * 4096;
    let p = Arc::new(l.compile(&accesses, &options).unwrap());
    assert!(p.summary().row_residencies > 0);
    let run = l.prepare(&accesses, ring(&l, 1, 4), Planning::Fixed(p.clone())).unwrap();
    let stats = run.for_each_batch(|b| {
        let got: Vec<f32> = b.region(0).chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, f.expected[accesses[b.first_position() as usize] as usize]);
        Ok(())
    }).unwrap();
    assert!(stats.counters.row_hits > 0);
    assert_follows(&stats, p.summary());
}

#[test]
fn online_plans_with_a_full_horizon_equal_offline_plans() {
    let f = Fixture::new(false);
    let accesses = f.accesses(4, 41);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    for (rows, blocks) in [(false, false), (true, false), (true, true), (false, true)] {
        for cache in [16 << 12, 128 << 12, 2048 << 12] {
            for chunk in [1, 7, 64, 4096] {
                let mut options = PlanOptions::new(cache, WINDOW);
                options.threads = 3;
                options.cost = Some(cost(if rows { 1e9 } else { 2e5 }, 64));
                options.rows = rows;
                options.block_pages = blocks.then_some(8);
                options.chunk = Some(chunk);
                let offline = l.compile(&accesses, &options).unwrap();
                options.horizon = Some((accesses.len() as u64).div_ceil(SLOT));
                let (online, stats) = l.compile_online(&accesses, &options).unwrap();
                assert_eq!(online.summary(), offline.summary(), "rows={rows} blocks={blocks} cache={cache} chunk={chunk}");
                assert_eq!(online.residencies(), offline.residencies());
                assert_eq!(stats.accesses, accesses.len() as u64);
            }
        }
    }
}

#[test]
fn finite_horizons_keep_every_rule_for_any_chunk() {
    let f = Fixture::new(false);
    let accesses = f.accesses(6, 43);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    let d = WINDOW * SLOT;
    for chunk in [1, 5, d - 1, d, d + 1, 3 * d + 7] {
        for horizon in [WINDOW, 3 * WINDOW, (accesses.len() as u64 / 2).div_ceil(SLOT)] {
            for rows in [false, true] {
                let mut options = PlanOptions::new(96 << 12, WINDOW);
                options.threads = 4;
                options.chunk = Some(chunk);
                options.horizon = Some(horizon);
                options.rows = rows;
                options.cost = Some(cost(1e9, 64));
                let (plan, stats) = l.compile_online(&accesses, &options).unwrap();
                assert_eq!(&l.replay(&plan, &accesses).unwrap(), plan.summary(), "chunk={chunk} horizon={horizon}");
                assert!(plan.summary().concurrent_peak_bytes <= 96 << 12);
                assert!(stats.chunks >= accesses.len() as u64 / chunk);
            }
        }
    }
}

#[test]
fn online_row_plans_keep_capacity_across_chunks() {
    let f = Fixture::new(false);
    let accesses = f.accesses(64, 41);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    let mut options = PlanOptions::new(64 << 12, WINDOW);
    options.threads = 2;
    options.rows = true;
    let mut c = cost(1e9, 64);
    c.input.decode = 1e-4;
    options.cost = Some(c);
    options.horizon = Some(4 * f.source.n_cells() / SLOT);
    let (plan, stats) = l.compile_online(&accesses, &options).unwrap();
    assert!(stats.chunks > 1);
    assert!(stats.row_candidates > 0 && stats.row_kept > 0, "{stats:?}");
    assert!(plan.summary().row_hits > 0);
    assert_eq!(&l.replay(&plan, &accesses).unwrap(), plan.summary());
}

#[test]
fn online_runs_load_computed_sequences() {
    let f = Fixture::new(false);
    let n = f.source.n_cells();
    for (cache, rows, horizon, chunk) in [(0usize, false, None, None), (64 << 12, false, Some(n / 2 / SLOT), Some(5)), (256 << 12, true, None, Some(300)), (4096 << 12, true, Some(n / SLOT), None)] {
        let shuffle = Shuffle::new(n, 5, 7).unwrap().blocks(16, 2).unwrap();
        let mut options = PlanOptions::new(cache, WINDOW);
        options.threads = 3;
        let mut c = cost(1e9, 64);
        c.input.decode = 1e-4;
        options.cost = Some(c);
        options.rows = rows;
        options.horizon = horizon;
        options.chunk = chunk;
        let l = loader(&f, config(IoBackend::Uring), dense_schema());
        let planning = l.online(options.clone()).unwrap();
        let stats = check_with(&f, &shuffle, config(IoBackend::Uring), planning, 2);
        let online = stats.online.clone().unwrap();
        assert_eq!(online.accesses, shuffle.len());
        assert_eq!(stats.counters.page_hits, online.page_hits);
        assert_eq!(stats.counters.row_hits, online.row_hits);
        if cache >= 256 << 12 {
            assert!(stats.counters.page_hits + stats.counters.row_hits > 0, "{stats:?}");
        }
    }
}

#[test]
fn profiled_runs_report_their_sites_and_load_the_same_rows() {
    let f = Fixture::new(false);
    let accesses = f.accesses(2, 51);
    let p = plan(&f, &accesses, 64 << 12);
    let plain = check(&f, &accesses, config(IoBackend::Uring), Some(p.clone()), 1);
    assert!(plain.profile.is_none());
    for names in [&["all"][..], &["io", "decode"], &[]] {
        let mut cfg = config(IoBackend::Uring);
        cfg.profile = Some(cellpack::prof::SiteMask::parse(names).unwrap());
        let stats = check(&f, &accesses, cfg, Some(p.clone()), 1);
        let (a, b) = (&stats.counters, &plain.counters);
        assert_eq!((a.accesses, a.decoded_rows, a.page_uses, a.page_hits, a.page_reads, a.admitted), (b.accesses, b.decoded_rows, b.page_uses, b.page_hits, b.page_reads, b.admitted));
        let report = stats.profile.unwrap();
        assert_eq!(report.sites.len(), cellpack::prof::SiteMask::parse(names).unwrap().names().len());
        if names.contains(&"all") {
            assert!(report.site("decode").unwrap().calls > 0);
            assert!(report.site("worker").unwrap().time > std::time::Duration::ZERO);
            assert!(report.site("map").unwrap().details.iter().map(|d| d.calls).sum::<u64>() > 0);
        }
    }
}

/// Answers each access differently every time it is asked, as a sequence
/// that advances a shared generator would: the plan's verification pass sees
/// the true cells, its replay and the workers each see another.
struct Drifting {
    cells: Vec<CellId>,
    n: CellId,
    calls: Vec<AtomicUsize>,
}

impl Accesses for Drifting {
    fn len(&self) -> u64 {
        self.cells.len() as u64
    }
    fn fill(&self, start: u64, out: &mut [CellId]) -> Result<()> {
        for (i, o) in out.iter_mut().enumerate() {
            let at = start as usize + i;
            let k = self.calls[at].fetch_add(1, Ordering::Relaxed) as CellId;
            *o = (self.cells[at] + k) % self.n;
        }
        Ok(())
    }
}

#[test]
fn plans_stop_sequences_that_answer_differently() {
    let f = Fixture::new(false);
    let cells = f.accesses(2, 47);
    let l = loader(&f, config(IoBackend::Uring), dense_schema());
    let plan = plan(&f, &cells, 64 << 12);
    let calls = cells.iter().map(|_| AtomicUsize::new(0)).collect();
    let drifting = Drifting { cells, n: f.source.n_cells() as CellId, calls };
    let run = l.prepare(&drifting, ring(&l, 1, WINDOW as u32 + 2), Planning::Fixed(plan)).unwrap();
    match run.for_each_batch(|_| Ok(())) {
        Err(Error::Plan(message)) => assert!(message.contains("not deterministic"), "{message}"),
        other => panic!("expected a plan error, got {:?}", other.map(|_| ())),
    }
}
