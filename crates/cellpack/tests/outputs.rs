//! Output protocol: schemas over one data source, slots shared by every
//! schema, element types, defaults, process functions, paired accesses, and
//! rows cached in their stored type serving schemas of different types.

use cellpack::{format::SourceLayout, loader::*, opt::*, scans::*, store::*};
use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

/// Standard ids: genes `0..GENES`, then embedding dims.
const GENES: usize = 40;
const DIMS: usize = 5;
const VOCAB: usize = GENES + DIMS;

struct Fixture {
    root: PathBuf,
    source: Arc<DataSource>,
    /// Counts (u16, CSR records) of cells `0..rows`, by gene.
    counts: Vec<Vec<u16>>,
    /// Embeddings (i16, dense records) of cells `rows..2 rows`.
    embedding: Vec<Vec<i16>>,
    /// Dense u32 matrix of cells `2 rows..`: its column `c` is gene
    /// `GENES - 1 - 2c` (every other gene, reversed).
    dense: Vec<Vec<u32>>,
}

fn options(layout: LayoutChoice) -> CpkOptions {
    CpkOptions {
        encode: EncodeOptions { renumber: true, thresholds: Thresholds { indices: 0.6, values: 0.6 }, max_lanes: 32, lane_min: 32, column_modes: None, layout, values: ValueChoice::Auto },
        chunk_rows: 17,
        page_size: 4096,
    }
}

impl Fixture {
    fn new(rows: usize) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tmp/output-tests").join(format!("{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir_all(&root).unwrap();
        let mut builder = DataSourceBuilder::new(VOCAB, 4096).unwrap().direct_io(true);

        let (mut ip, mut ix, mut values, mut counts) = (vec![0u64], Vec::new(), Vec::new(), Vec::new());
        for r in 0..rows {
            let mut row = vec![0u16; GENES];
            for (c, v) in row.iter_mut().enumerate() {
                if (r + c) % 3 != 0 {
                    *v = (r * 7 + c + 1) as u16;
                    ix.push(c as u32);
                    values.extend(v.to_le_bytes());
                }
            }
            ip.push(ix.len() as u64);
            counts.push(row);
        }
        let path = root.join("counts.cpk");
        let mut w = CpkWriter::create(&path, GENES as u32, Dtype::U16, SourceLayout::Csr, options(LayoutChoice::Sparse)).unwrap();
        w.append(CsrRef { n_genes: GENES as u32, value_width: 2, indptr: &ip, indices: Indices::U32(&ix), data: &values }).unwrap();
        w.finish().unwrap();
        builder.add_cpk(&path, &(0..GENES as u32).collect::<Vec<_>>()).unwrap();

        let embedding: Vec<Vec<i16>> = (0..rows).map(|r| (0..DIMS).map(|c| r as i16 * 3 - c as i16).collect()).collect();
        let bytes: Vec<u8> = embedding.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let path = root.join("embedding.cpk");
        let mut w = CpkWriter::create(&path, DIMS as u32, Dtype::I16, SourceLayout::Dense, options(LayoutChoice::Dense)).unwrap();
        w.append(DenseRef { n_genes: DIMS as u32, value_width: 2, data: &bytes }).unwrap();
        w.finish().unwrap();
        let dims: Vec<u32> = (GENES..VOCAB).map(|s| s as u32).collect();
        builder.add_cpk(&path, &dims).unwrap();

        let cols = GENES / 2;
        let dense: Vec<Vec<u32>> = (0..rows).map(|r| (0..cols).map(|c| ((r * 13 + c * 5) % 11) as u32 * 70_000).collect()).collect();
        let bytes: Vec<u8> = dense.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let path = root.join("dense.cpk");
        let mut w = CpkWriter::create(&path, cols as u32, Dtype::U32, SourceLayout::Dense, options(LayoutChoice::Dense)).unwrap();
        w.append(DenseRef { n_genes: cols as u32, value_width: 4, data: &bytes }).unwrap();
        w.finish().unwrap();
        let genes: Vec<u32> = (0..cols).map(|c| (GENES - 1 - 2 * c) as u32).collect();
        builder.add_cpk(&path, &genes).unwrap();

        Self { root, source: Arc::new(builder.build().unwrap()), counts, embedding, dense }
    }
    fn rows(&self) -> usize {
        self.counts.len()
    }
    /// A cell's value by standard id: `None` where its matrix has no such
    /// column (missing); unstored counts are present zeros.
    fn value(&self, cell: CellId, id: usize) -> Option<f64> {
        let (n, c) = (self.rows(), cell as usize);
        if c < n {
            (id < GENES).then(|| self.counts[c][id] as f64)
        } else if c < 2 * n {
            (id >= GENES).then(|| self.embedding[c - n][id - GENES] as f64)
        } else {
            (id < GENES && (GENES - 1 - id).is_multiple_of(2)).then(|| self.dense[c - 2 * n][(GENES - 1 - id) / 2] as f64)
        }
    }
    /// Positions visiting every count cell `epochs` times in a fixed order.
    fn positions(&self, epochs: usize) -> Vec<CellId> {
        let n = self.rows();
        (0..epochs).flat_map(|e| (0..n).map(move |i| ((i * 37 + e * 11) % n) as CellId)).collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn genes() -> Vec<u32> {
    (0..GENES as u32).collect()
}
fn dims() -> Vec<u32> {
    (GENES as u32..VOCAB as u32).collect()
}
/// Every third gene, reversed: a partial panel in another order.
fn sparse_panel() -> Vec<u32> {
    (0..GENES as u32).rev().step_by(3).collect()
}
/// Every eighth gene, reversed: narrow enough for sparse column maps.
fn narrow_panel() -> Vec<u32> {
    (0..GENES as u32).rev().step_by(8).collect()
}

/// Writes (cell, position, sum of values, sum of (column + 1) * value,
/// kind): kind 0 for a CSR row, 1 for a dense row, 2 for a partial row.
unsafe fn summary(context: &ProcessContext, values: &[f64], columns: impl Iterator<Item = u32>, kind: u64, out: *mut u8) {
    let sum: f64 = values.iter().sum();
    let weighted: f64 = values.iter().zip(columns).map(|(v, c)| v * (c + 1) as f64).sum();
    let words = [context.cell, context.position, sum.to_bits(), weighted.to_bits(), kind];
    // SAFETY: the cell's output holds 40 bytes.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), out, 40) };
}

unsafe extern "C" fn sum_csr(call: *const CsrRow) -> i32 {
    // SAFETY: the loader passes a valid row of f64 values.
    unsafe {
        let c = &*call;
        let n = c.nnz as usize;
        if c.context.dtype != 9 {
            return 3;
        }
        if c.columns.is_null() {
            let (mut cols, mut vals) = (vec![0u32; n], vec![0f64; n]);
            if (c.resolve.unwrap())(c.row, std::ptr::null(), n as u64, cols.as_mut_ptr(), vals.as_mut_ptr().cast()) != 0 {
                return 4;
            }
            // Single positions resolve to the same elements.
            if n > 0 {
                let (mut col, mut val) = (0u32, 0f64);
                let at = (n - 1) as u32;
                if (c.resolve.unwrap())(c.row, &at, 1, &mut col, (&mut val as *mut f64).cast()) != 0 || col != cols[n - 1] || val != vals[n - 1] {
                    return 5;
                }
            }
            summary(&c.context, &vals, cols.into_iter(), 2, c.out);
        } else {
            let vals = std::slice::from_raw_parts(c.values.cast::<f64>(), n);
            summary(&c.context, vals, std::slice::from_raw_parts(c.columns, n).iter().copied(), 0, c.out);
        }
    }
    0
}

unsafe extern "C" fn sum_dense(call: *const DenseRow) -> i32 {
    // SAFETY: as above.
    unsafe {
        let c = &*call;
        let vals = std::slice::from_raw_parts(c.values.cast::<f64>(), c.out_dim as usize);
        summary(&c.context, vals, 0.., 1, c.out);
    }
    0
}

fn summing(id: u64, out_ids: Vec<u32>, partial: bool) -> OutputSchema {
    // SAFETY: both functions are thread-safe and write the 40 bytes of their cell.
    let mut p = unsafe { ProcessFn::new(sum_csr, sum_dense, std::ptr::null(), Arc::new(())) };
    if partial {
        // SAFETY: the CSR function resolves null rows.
        p = unsafe { p.with_partial_rows() };
    }
    OutputSchema::process(id, out_ids, Dtype::F64, 40, p, 1, 0).unwrap()
}

fn config(slot_cells: u64) -> LoaderConfig {
    LoaderConfig { workers: 3, backend: IoBackend::Uring, max_inflight_per_worker: 4, claim_batch: 3, slot_cells, window: Some(3), ..Default::default() }
}

/// Little-endian values of `dtype` as f64.
fn read(dtype: Dtype, bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(dtype.width())
        .map(|b| match dtype {
            Dtype::F32 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
            Dtype::F64 => f64::from_le_bytes(b.try_into().unwrap()),
            Dtype::U32 => u32::from_le_bytes(b.try_into().unwrap()) as f64,
            Dtype::U16 => u16::from_le_bytes(b.try_into().unwrap()) as f64,
            Dtype::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f64,
            Dtype::U64 => u64::from_le_bytes(b.try_into().unwrap()) as f64,
            d => panic!("{d:?}"),
        })
        .collect()
}

/// Checks schema `s`'s output of one cell at `position`.
fn check_cell(f: &Fixture, schema: &OutputSchema, cell: CellId, position: u64, bytes: &[u8]) {
    match schema.process_fn() {
        None => {
            let default = read(schema.dtype(), &schema.default_bits().to_le_bytes()[..schema.dtype().width()])[0];
            let want: Vec<f64> = schema.out_ids().iter().map(|&id| f.value(cell, id as usize).map_or(default, |v| read(schema.dtype(), &value_as(schema.dtype(), v))[0])).collect();
            assert_eq!(read(schema.dtype(), bytes), want, "schema {} cell {cell}", schema.id());
        }
        Some(_) => {
            let w: Vec<u64> = bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
            let row: Vec<(usize, f64)> = schema.out_ids().iter().enumerate().filter_map(|(j, &id)| f.value(cell, id as usize).map(|v| (j, v))).collect();
            assert_eq!((w[0], w[1]), (cell as u64, position), "schema {}", schema.id());
            assert_eq!(f64::from_bits(w[2]), row.iter().map(|x| x.1).sum::<f64>(), "schema {} cell {cell}", schema.id());
            assert_eq!(f64::from_bits(w[3]), row.iter().map(|&(j, v)| v * (j + 1) as f64).sum::<f64>(), "schema {} cell {cell}", schema.id());
        }
    }
}

/// `v` as little-endian bytes of `dtype` (`as` semantics).
fn value_as(dtype: Dtype, v: f64) -> Vec<u8> {
    match dtype {
        Dtype::F32 => (v as f32).to_le_bytes().to_vec(),
        Dtype::F64 => v.to_le_bytes().to_vec(),
        Dtype::U32 => (v as i64 as u32).to_le_bytes().to_vec(),
        Dtype::U16 => (v as i64 as u16).to_le_bytes().to_vec(),
        Dtype::I32 => (v as i32).to_le_bytes().to_vec(),
        Dtype::U64 => (v as i64 as u64).to_le_bytes().to_vec(),
        d => panic!("{d:?}"),
    }
}

/// Runs and checks every region of every slot; returns the stats.
fn run_and_check(f: &Fixture, loader: &Loader, accesses: &dyn Accesses, planning: Planning) -> RunStats {
    let s = loader.schemas().len();
    let mut cells = vec![0; accesses.len() as usize];
    accesses.fill(0, &mut cells).unwrap();
    let ring = Arc::new(Ring::allocate(loader.ring_layout(5, 1)).unwrap());
    let run = loader.prepare(accesses, ring, planning).unwrap();
    let mut seen = 0;
    let stats = run
        .for_each_batch(|b| {
            for (k, schema) in loader.schemas().iter().enumerate() {
                let bytes = b.cell_bytes(k);
                assert_eq!(b.region_offset(k) % REGION_ALIGN, 0);
                assert_eq!(b.region(k).len(), b.rows() * bytes);
                for r in 0..b.rows() {
                    let position = b.first_position() + r as u64;
                    check_cell(f, schema, cells[position as usize * s + k], position, &b.region(k)[r * bytes..(r + 1) * bytes]);
                }
            }
            seen += b.rows();
            Ok(())
        })
        .unwrap();
    assert_eq!(seen as u64 * s as u64, accesses.len());
    stats
}

#[test]
fn schemas_share_slots_and_convert_rows_to_their_types() {
    let f = Fixture::new(60);
    let schemas = vec![
        OutputSchema::dense(1, genes(), Dtype::F32).unwrap(),
        OutputSchema::dense(2, sparse_panel(), Dtype::U32).unwrap().with_default(7.0),
        summing(3, genes(), true),
        OutputSchema::dense(4, dims(), Dtype::F32).unwrap(),
        summing(5, sparse_panel(), false),
        OutputSchema::dense(6, narrow_panel(), Dtype::F32).unwrap().with_default(7.0),
    ];
    let loader = Loader::new(f.source.clone(), schemas, config(4)).unwrap();
    let layout = loader.ring_layout(2, 1);
    assert_eq!(layout.cell_bytes, vec![GENES as u32 * 4, sparse_panel().len() as u32 * 4, 40, DIMS as u32 * 4, 40, narrow_panel().len() as u32 * 4]);
    let n = f.rows() as CellId;
    let accesses = Offset::new(f.positions(3), vec![0, 0, 0, n, 0, 0]).unwrap();
    let stats = run_and_check(&f, &loader, &accesses, Planning::None);
    assert_eq!(stats.outputs.iter().map(|o| o.accesses).collect::<Vec<_>>(), vec![3 * n as u64; 6]);
    assert_eq!(stats.outputs[2].process_calls, 3 * n as u64);
    assert_eq!(stats.outputs[0].process_calls, 0);
}

#[test]
fn dense_records_and_irrelevant_cells_follow_the_contract() {
    let f = Fixture::new(40);
    let schemas = vec![OutputSchema::dense(1, genes(), Dtype::U64).unwrap().with_default(3.0), summing(2, genes(), true), OutputSchema::dense(3, dims(), Dtype::I32).unwrap().with_default(-1.0)];
    let loader = Loader::new(f.source.clone(), schemas, config(3)).unwrap();
    // Every cell of every matrix reaches every schema: dense u32 records
    // widen to u64, an embedding cell has no gene and a count cell no dim.
    let cells: Vec<CellId> = (0..f.source.n_cells() as CellId).chain((0..f.source.n_cells() as CellId).rev()).collect();
    let accesses = Replicate::new(cells, 3).unwrap();
    let ring = Arc::new(Ring::allocate(loader.ring_layout(4, 1)).unwrap());
    let run = loader.prepare(&accesses, ring, Planning::None).unwrap();
    let all: Vec<CellId> = (0..accesses.len()).map(|k| { let mut c = [0]; accesses.fill(k, &mut c).unwrap(); c[0] }).collect();
    run.for_each_batch(|b| {
        for r in 0..b.rows() {
            let position = b.first_position() + r as u64;
            let cell = all[position as usize * 3];
            for (k, schema) in loader.schemas().iter().enumerate() {
                let bytes = &b.region(k)[r * b.cell_bytes(k)..(r + 1) * b.cell_bytes(k)];
                check_cell(&f, schema, cell, position, bytes);
                if k == 1 {
                    // Dense records covering half the panel reach the dense
                    // function; the embedding covers none of it and arrives
                    // as an empty CSR row; CSR records with every gene in the
                    // panel arrive partial.
                    let kind = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
                    let n = f.rows() as CellId;
                    assert_eq!(kind, if (n..2 * n).contains(&cell) { 0 } else if cell >= 2 * n { 1 } else { 2 }, "cell {cell}");
                }
            }
        }
        Ok(())
    })
    .unwrap();
}

/// A CSR function that fails on a zero value; otherwise `summary` (kind 0).
unsafe extern "C" fn nonzero_csr(call: *const CsrRow) -> i32 {
    // SAFETY: the loader passes a valid eager row of f64 values.
    unsafe {
        let c = &*call;
        let n = c.nnz as usize;
        let vals = std::slice::from_raw_parts(c.values.cast::<f64>(), n);
        if vals.contains(&0.0) {
            return 6;
        }
        summary(&c.context, vals, std::slice::from_raw_parts(c.columns, n).iter().copied(), 0, c.out);
    }
    0
}

/// A dense function over a NaN default: (cell, position, sum of present
/// values, present dims, 1).
unsafe extern "C" fn presence_dense(call: *const DenseRow) -> i32 {
    // SAFETY: the loader passes a valid row of f64 values.
    unsafe {
        let c = &*call;
        let vals = std::slice::from_raw_parts(c.values.cast::<f64>(), c.out_dim as usize);
        let present: Vec<f64> = vals.iter().copied().filter(|v| !v.is_nan()).collect();
        let words = [c.context.cell, c.context.position, present.iter().sum::<f64>().to_bits(), present.len() as u64, 1];
        std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), c.out, 40);
    }
    0
}

#[test]
fn row_formats_follow_coverage_and_mark_missing_dims() {
    let f = Fixture::new(30);
    let wide: Vec<u32> = (0..VOCAB as u32).collect();
    // SAFETY: the functions are thread-safe and write the 40 bytes of their cell.
    let csr_only = unsafe { ProcessFn::with_format(RowFormat::Csr, Some(nonzero_csr), None, std::ptr::null(), Arc::new(())) }.unwrap();
    // SAFETY: as above.
    let dense_only = unsafe { ProcessFn::with_format(RowFormat::Dense, None, Some(presence_dense), std::ptr::null(), Arc::new(())) }.unwrap();
    // SAFETY: as above.
    assert!(unsafe { ProcessFn::with_format(RowFormat::Auto, Some(sum_csr), None, std::ptr::null(), Arc::new(())) }.is_err());
    let schemas = vec![
        summing(1, wide.clone(), true),
        OutputSchema::process(2, genes(), Dtype::F64, 40, csr_only, 2, 0).unwrap(),
        OutputSchema::process(3, genes(), Dtype::F64, 40, dense_only, 3, 0).unwrap().with_default(f64::NAN),
        OutputSchema::dense(4, wide, Dtype::F64).unwrap().with_default(f64::NAN),
    ];
    let loader = Loader::new(f.source.clone(), schemas, config(3)).unwrap();
    let n = f.rows() as CellId;
    let cells: Vec<CellId> = (0..f.source.n_cells() as CellId).collect();
    let accesses = Replicate::new(cells.clone(), 4).unwrap();
    let ring = Arc::new(Ring::allocate(loader.ring_layout(4, 1)).unwrap());
    let run = loader.prepare(&accesses, ring, Planning::None).unwrap();
    run.for_each_batch(|b| {
        for r in 0..b.rows() {
            let position = b.first_position() + r as u64;
            let cell = cells[position as usize];
            let words = |k: usize| -> Vec<u64> { b.region(k)[r * 40..(r + 1) * 40].chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect() };
            let present: Vec<(usize, f64)> = (0..GENES).filter_map(|id| f.value(cell, id).map(|v| (id, v))).collect();
            let sum: f64 = present.iter().map(|x| x.1).sum();
            // Auto over the whole vocabulary: the dense u32 matrix covers 20
            // of 45 dims and arrives as CSR, like the embedding (5 of 45).
            let auto = words(0);
            assert_eq!(auto[4], if cell < n { 2 } else { 0 }, "cell {cell}");
            let all: f64 = (0..VOCAB).filter_map(|id| f.value(cell, id)).sum();
            assert_eq!(f64::from_bits(auto[2]), all, "cell {cell}");
            // Forced CSR: dense records give their non-zeros only.
            let csr = words(1);
            assert_eq!((csr[4], f64::from_bits(csr[2])), (0, sum), "cell {cell}");
            // Forced dense with a NaN default: present zeros stay 0.
            let dense = words(2);
            assert_eq!((dense[4], dense[3], f64::from_bits(dense[2])), (1, present.len() as u64, sum), "cell {cell}");
            // The built-in output marks exactly the missing dims.
            let row = read(Dtype::F64, &b.region(3)[r * VOCAB * 8..(r + 1) * VOCAB * 8]);
            for (id, v) in row.iter().enumerate() {
                match f.value(cell, id) {
                    Some(want) => assert_eq!(*v, want, "cell {cell} dim {id}"),
                    None => assert!(v.is_nan(), "cell {cell} dim {id}"),
                }
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn compacted_dense_images_preserve_values_missing_dims_and_budget() {
    let f = Fixture::new(4);
    let (rows, cols, vocab) = (64usize, 128usize, 256usize);
    let values: Vec<f32> = (0..rows).flat_map(|r| (0..cols).map(move |c| {
        let density = [0, 12, 25, 50, 75, 100][r % 6];
        if (c * 37 + r * 17) % 100 < density { (1 + (r * 3 + c * 7) % 31) as f32 } else { 0.0 }
    })).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let map: Vec<u32> = (0..cols).map(|c| if c % 7 == 0 { UNMAPPED } else { (2 * c) as u32 }).collect();
    let mut builder = DataSourceBuilder::new(vocab, 4096).unwrap();
    for (i, coding) in [ValueChoice::Raw, ValueChoice::Auto, ValueChoice::Modes].into_iter().enumerate() {
        let path = f.root.join(format!("compact-{i}.cpk"));
        let mut opt = options(LayoutChoice::Dense);
        opt.encode.values = coding;
        opt.encode.column_modes = (coding == ValueChoice::Modes).then_some(0.0);
        let mut writer = CpkWriter::create(&path, cols as u32, Dtype::F32, SourceLayout::Dense, opt).unwrap();
        writer.append(DenseRef { n_genes: cols as u32, value_width: 4, data: &bytes }).unwrap();
        writer.finish().unwrap();
        builder.add_cpk(&path, &map).unwrap();
    }
    // SAFETY: the callback consumes f64 rows and writes exactly five words.
    let presence = unsafe { ProcessFn::with_format(RowFormat::Dense, None, Some(presence_dense), std::ptr::null(), Arc::new(())) }.unwrap();
    let ids: Vec<u32> = (0..vocab as u32).collect();
    let schemas = vec![
        summing(1, ids.clone(), false),
        OutputSchema::dense(2, ids.clone(), Dtype::F32).unwrap().with_default(f64::NAN),
        OutputSchema::process(3, ids, Dtype::F64, 40, presence, 3, 0).unwrap().with_default(f64::NAN),
        // Narrow: a sixteenth of the vocabulary, through sparse maps.
        OutputSchema::dense(4, (0..vocab as u32).step_by(16).collect::<Vec<_>>(), Dtype::F32).unwrap().with_default(f64::NAN),
    ];
    let loader = Loader::new(Arc::new(builder.build().unwrap()), schemas, config(4)).unwrap();
    let positions: Vec<CellId> = (0..32 * 3 * rows).map(|i| ((i * 37) % (3 * rows)) as CellId).collect();
    let accesses: Vec<CellId> = positions.iter().flat_map(|&cell| [cell; 4]).collect();
    let mut options = PlanOptions::new(128 << 10, 3);
    options.rows = true;
    options.cost = Some(exact_cost(4));
    options.threads = 2;
    for cached in [false, true] {
        let planning = if cached {
            let plan = loader.compile(&accesses, &options).unwrap();
            assert!(plan.summary().row_hits > 0);
            assert!(plan.summary().concurrent_peak_bytes <= options.cache_bytes as u64);
            assert_eq!(&loader.replay(&plan, &accesses).unwrap(), plan.summary());
            Planning::Fixed(Arc::new(plan))
        } else { Planning::None };
        let ring = Arc::new(Ring::allocate(loader.ring_layout(5, 1)).unwrap());
        let run = loader.prepare(&accesses, ring, planning).unwrap();
        let stats = run.for_each_batch(|batch| {
            for r in 0..batch.rows() {
                let position = batch.first_position() as usize + r;
                let cell = positions[position] as usize;
                let input = &values[cell % rows * cols..(cell % rows + 1) * cols];
                let sum: f64 = input.iter().zip(&map).filter(|(_, c)| **c != UNMAPPED).map(|(&v, _)| v as f64).sum();
                let weighted: f64 = input.iter().zip(&map).filter(|(_, c)| **c != UNMAPPED).map(|(&v, &c)| v as f64 * (c + 1) as f64).sum();
                let words = |k: usize| -> Vec<u64> { batch.region(k)[r * 40..(r + 1) * 40].chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect() };
                let got = words(0);
                assert_eq!((got[0], got[1], f64::from_bits(got[2]), f64::from_bits(got[3])), (cell as u64, position as u64, sum, weighted));
                let present = words(2);
                assert_eq!((present[0], present[1], f64::from_bits(present[2]), present[3]), (cell as u64, position as u64, sum, map.iter().filter(|&&c| c != UNMAPPED).count() as u64));
                let dense = &batch.region(1)[r * vocab * 4..(r + 1) * vocab * 4];
                for (c, b) in dense.chunks_exact(4).enumerate() {
                    let got = f32::from_le_bytes(b.try_into().unwrap());
                    match map.iter().position(|&m| m == c as u32) {
                        Some(i) => assert_eq!(got.to_bits(), input[i].to_bits()),
                        None => assert!(got.is_nan()),
                    }
                }
                let narrow = &batch.region(3)[r * (vocab / 16) * 4..(r + 1) * (vocab / 16) * 4];
                for (j, b) in narrow.chunks_exact(4).enumerate() {
                    let got = f32::from_le_bytes(b.try_into().unwrap());
                    match map.iter().position(|&m| m == 16 * j as u32) {
                        Some(i) => assert_eq!(got.to_bits(), input[i].to_bits()),
                        None => assert!(got.is_nan()),
                    }
                }
            }
            Ok(())
        }).unwrap();
        if cached {
            assert!(stats.counters.rows_admitted > 0);
            assert!(stats.outputs.iter().map(|o| o.row_hits).sum::<u64>() > 0);
        }
    }
}

#[test]
fn conversions_are_checked_against_every_relevant_matrix() {
    let f = Fixture::new(20);
    let build = |schema: OutputSchema| Loader::new(f.source.clone(), vec![schema], config(4));
    // u16 counts and u32 dense values: u16 is too narrow for the latter.
    assert!(build(OutputSchema::dense(1, genes(), Dtype::U16).unwrap()).is_err());
    // u32 to i32 is a same-width unsigned to signed conversion.
    assert!(build(OutputSchema::dense(1, genes(), Dtype::I32).unwrap()).is_err());
    assert!(build(OutputSchema::dense(1, genes(), Dtype::U32).unwrap()).is_ok());
    assert!(build(OutputSchema::dense(1, genes(), Dtype::F32).unwrap()).is_ok());
    // Embedding dims are i16 only: same-width and wider unsigned are allowed.
    assert!(build(OutputSchema::dense(1, dims(), Dtype::U16).unwrap()).is_ok());
    assert!(build(OutputSchema::dense(1, dims(), Dtype::U8).unwrap()).is_err());
    // A panel with no gene of the dense matrix does not check it.
    let even: Vec<u32> = (0..GENES as u32).filter(|g| (GENES as u32 - 1 - g) % 2 == 1).collect();
    assert!(build(OutputSchema::dense(1, even, Dtype::U16).unwrap()).is_ok());
    // Output dims are distinct standard ids inside the vocabulary.
    assert!(OutputSchema::dense(1, vec![1, 1], Dtype::F32).is_err());
    assert!(build(OutputSchema::dense(1, vec![VOCAB as u32], Dtype::F32).unwrap()).is_err());
}

fn exact_cost(schemas: usize) -> CostModel {
    CostModel {
        cores: 1.0,
        iops: 1e9,
        bandwidth: 1e14,
        memory_bandwidth: 1e14,
        serial: 0.0,
        bucket: 1 << 20,
        access: 1e-6,
        request: 1e-6,
        compile: 0.0,
        input: InputCost { decode: 1e-3, decode_element: 0.0, cold: 0.0 },
        rows: RowCost::default(),
        outputs: vec![OutputCost { curve: Vec::new(), resolve_element: 0.0, resolve: 0.0, process: 1e-7, map_element: 1e-9, fill_element: 0.0 }; schemas],
        convert: [0.0; CONVERSIONS],
    }
}

#[test]
fn rows_cached_in_their_stored_type_serve_every_schema() {
    let f = Fixture::new(50);
    let schemas = vec![OutputSchema::dense(1, genes(), Dtype::F32).unwrap(), OutputSchema::dense(2, sparse_panel(), Dtype::U32).unwrap().with_default(7.0), summing(3, genes(), true)];
    let loader = Loader::new(f.source.clone(), schemas, LoaderConfig { workers: 1, max_inflight_per_worker: 1, claim_batch: 1, slot_cells: 1, window: Some(1), ..Default::default() }).unwrap();
    // Count cells and dense cells, each position's cell read by every schema.
    let n = f.rows() as CellId;
    let positions: Vec<CellId> = (0..4).flat_map(|e| (0..30).map(move |i| if i % 3 == 0 { 2 * n + (i * 7 + e) % n } else { (i * 11 + e * 5) % n })).collect();
    let accesses: Vec<CellId> = positions.iter().flat_map(|&c| [c, c, c]).collect();
    let mut options = PlanOptions::new(1 << 20, 1);
    options.cost = Some(exact_cost(3));
    options.rows = true;
    options.threads = 2;
    let plan = Arc::new(loader.compile(&accesses, &options).unwrap());
    let s = plan.summary().clone();
    assert_eq!(loader.replay(&plan, &accesses).unwrap(), s);
    // Rows outlive the gap only across positions; every schema of a later
    // position hits the row its cell kept.
    assert!(s.row_hits > 0 && s.decoded_rows < accesses.len() as u64 / 2, "{s:?}");
    let stats = run_and_check(&f, &loader, &accesses, Planning::Fixed(plan));
    assert_eq!(stats.counters.row_hits, s.row_hits);
    assert_eq!(stats.counters.rows_admitted, s.row_residencies);
    assert!(stats.outputs.iter().all(|o| o.row_hits > 0), "{:?}", stats.outputs);
}

#[test]
fn plans_bind_to_slots_schemas_and_types() {
    let f = Fixture::new(30);
    let accesses = f.positions(4);
    let make = |slot: u64, dtype: Dtype| Loader::new(f.source.clone(), vec![OutputSchema::dense(1, genes(), dtype).unwrap()], config(slot)).unwrap();
    let a = make(4, Dtype::F32);
    let mut options = PlanOptions::new(64 << 12, 3);
    options.threads = 2;
    let plan = Arc::new(a.compile(&accesses, &options).unwrap());
    for other in [make(5, Dtype::F32), make(4, Dtype::F64)] {
        let ring = Arc::new(Ring::allocate(other.ring_layout(4, 1)).unwrap());
        assert!(matches!(other.prepare(&accesses, ring, Planning::Fixed(plan.clone())), Err(Error::Plan(_))));
    }
    // A ring of another layout is refused.
    let ring = Arc::new(Ring::allocate(make(5, Dtype::F32).ring_layout(4, 1)).unwrap());
    assert!(matches!(a.prepare(&accesses, ring, Planning::Fixed(plan.clone())), Err(Error::Config(_))));
    run_and_check(&f, &a, &accesses, Planning::Fixed(plan));
}

#[test]
fn online_runs_of_replicated_sequences() {
    let f = Fixture::new(64);
    let schemas = vec![OutputSchema::dense(1, genes(), Dtype::F32).unwrap(), summing(2, sparse_panel(), true)];
    let loader = Loader::new(f.source.clone(), schemas, config(4)).unwrap();
    let shuffle = Shuffle::new(f.rows() as u64, 6, 3).unwrap();
    let accesses = Replicate::new(shuffle, 2).unwrap();
    let mut options = PlanOptions::new(256 << 12, 3);
    options.cost = Some(exact_cost(2));
    options.rows = true;
    options.threads = 2;
    let planning = loader.online(options).unwrap();
    let stats = run_and_check(&f, &loader, &accesses, planning);
    assert_eq!(stats.online.unwrap().accesses, accesses.len());
}

#[test]
fn streamed_residencies_keep_the_budget_across_many_chunks() {
    let f = Fixture::new(128);
    let schemas = vec![OutputSchema::dense(1, genes(), Dtype::F32).unwrap(), summing(2, sparse_panel(), true)];
    let loader = Loader::new(f.source.clone(), schemas, config(4)).unwrap();
    let accesses: Vec<CellId> = f.positions(12_000).into_iter().flat_map(|cell| [cell, cell]).collect();
    let mut options = PlanOptions::new(32 << 10, 3);
    options.cost = Some(exact_cost(2));
    options.rows = true;
    options.threads = 4;
    options.chunk = Some(4096);
    let (plan, stats) = loader.compile_online(&accesses, &options).unwrap();
    assert!(stats.chunks > 8);
    assert!(plan.summary().row_hits > 0);
    assert!(plan.summary().concurrent_peak_bytes <= options.cache_bytes as u64);
    assert_eq!(&loader.replay(&plan, &accesses).unwrap(), plan.summary());
}

#[test]
fn leases_apply_backpressure_and_generations_prevent_stale_batches() {
    let f = Fixture::new(30);
    let loader = Loader::new(f.source.clone(), vec![OutputSchema::dense(1, genes(), Dtype::F32).unwrap(), summing(2, genes(), false)], config(4)).unwrap();
    let accesses = Replicate::new(f.positions(2), 2).unwrap();
    let ring = Arc::new(Ring::allocate(loader.ring_layout(2, 1)).unwrap());
    let run = loader.prepare(&accesses, ring.clone(), Planning::None).unwrap();
    std::thread::scope(|s| {
        let producer = s.spawn(|| run.run());
        let first = run.wait(0, 0).unwrap().unwrap();
        let second = run.wait(0, 1).unwrap().unwrap();
        // Two slots, both leased: the producer waits for a lease to return.
        std::thread::sleep(Duration::from_millis(50));
        assert!(!producer.is_finished());
        let held = first.region(0).to_vec();
        // A duplicate wait is refused and cancels the run; existing leases
        // stay readable.
        assert!(run.wait(0, 0).is_err());
        assert!(matches!(run.wait(0, 2), Err(Error::Aborted)));
        assert_eq!(first.region(0), &held[..]);
        drop(first);
        drop(second);
        assert!(matches!(producer.join().unwrap(), Err(Error::Aborted)));
    });
    drop(run);
    // An aborted ring cannot start another run.
    assert!(loader.prepare(&accesses, ring, Planning::None).is_err());
}

#[test]
fn short_row_residencies_survive_out_of_order_execution() {
    let f = Fixture::new(64);
    let schemas = vec![OutputSchema::dense(1, genes(), Dtype::F32).unwrap(), OutputSchema::dense(2, sparse_panel(), Dtype::U32).unwrap(), summing(3, genes(), true)];
    let cfg = LoaderConfig { workers: 6, max_inflight_per_worker: 6, claim_batch: 2, slot_cells: 2, window: Some(4), ..Default::default() };
    let loader = Loader::new(f.source.clone(), schemas, cfg).unwrap();
    let n = f.rows() as CellId;
    for seed in 0..6u32 {
        let positions: Vec<CellId> = (0..400u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed * 977) >> 7) % (n / 2) + if i % 4 == 0 { 2 * n } else { 0 }).collect();
        let accesses: Vec<CellId> = positions.iter().flat_map(|&c| [c, c, c]).collect();
        // A small budget: the densest intervals, a position's own schemas,
        // are the ones kept.
        let mut options = PlanOptions::new(8 << 10, 4);
        options.cost = Some(exact_cost(3));
        options.rows = true;
        let plan = Arc::new(loader.compile(&accesses, &options).unwrap());
        assert!(plan.residencies().iter().any(|r| matches!(r.object, Object::Row(_)) && r.end - r.start <= 4 * 2 * 3));
        let stats = run_and_check(&f, &loader, &accesses, Planning::Fixed(plan.clone()));
        assert!(stats.counters.row_hits > 0);
        assert_eq!(stats.counters.row_hits, plan.summary().row_hits, "{stats:?}");
    }
}

#[test]
fn replays_reprice_a_plan_without_recompiling_it() {
    let f = Fixture::new(30);
    let schemas = vec![OutputSchema::dense(1, genes(), Dtype::F32).unwrap(), summing(2, genes(), false)];
    let loader = Loader::new(f.source.clone(), schemas, config(4)).unwrap();
    let cells: Vec<_> = (0..f.rows() as u32).flat_map(|cell| [cell, cell]).collect();
    let mut options = PlanOptions::new(16 << 12, 3);
    options.cost = Some(exact_cost(2));
    let plan = loader.compile(&cells, &options).unwrap();
    assert_eq!(plan.summary().quarter_requests.iter().sum::<u64>(), plan.summary().requests);
    let mut price = plan.cost().unwrap().clone();
    price.access += 1.0;
    let priced = loader.replay_with_cost(&plan, &cells, &price).unwrap();
    assert_eq!(priced.requests, plan.summary().requests);
    assert!((priced.worker_seconds - plan.summary().worker_seconds - cells.len() as f64).abs() < 1e-6);
    price.cores = 0.0;
    assert!(loader.replay_with_cost(&plan, &cells, &price).is_err());
    // A position's schemas share one read and one decode.
    assert!(plan.summary().decoded_rows <= cells.len() as u64 / 2);
    let (online, _) = loader.compile_online(&cells, &options).unwrap();
    assert_eq!(online.summary().quarter_requests.iter().sum::<u64>(), online.summary().requests);
}
