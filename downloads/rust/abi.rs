mod common;

use std::error::Error;
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use cellpack::loader::{
    CellId, CsrRow, DataSourceBuilder, DenseRow, Dtype, ExternAccesses, IoBackend, Loader,
    LoaderConfig, OutputSchema, PROCESS_ABI_VERSION, Planning, ProcessContext, ProcessFn, Ring,
};

// #region record
/// The fixed-size record each cell produces: 12 bytes, written unaligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Summary {
    total: f32,
    peak_column: u32,
    nonzeros: u32,
}

/// Immutable settings both callbacks read through `user_data`.
struct Settings {
    scale: f32,
}

/// Accepts only the ABI and dtype this callback was written for.
fn settings(context: &ProcessContext) -> Option<&Settings> {
    if context.abi_version != PROCESS_ABI_VERSION || context.dtype != Dtype::F32 as u32 {
        return None;
    }
    // SAFETY: user_data points into the Arc<Settings> held by the ProcessFn owner.
    Some(unsafe { &*context.user_data.cast::<Settings>() })
}
// #endregion record

/// Folds one (column, value) pair; ties keep the smaller column, so both
/// callbacks agree whatever order CSR elements arrive in.
fn fold(acc: &mut (Summary, f32), column: u32, value: f32) {
    let (summary, peak) = acc;
    if value != 0.0 {
        summary.nonzeros += 1;
    }
    summary.total += value;
    if value > *peak || (value == *peak && column < summary.peak_column) {
        *peak = value;
        summary.peak_column = column;
    }
}

fn finish((mut summary, _): (Summary, f32), scale: f32, out: *mut u8) {
    summary.total *= scale;
    // SAFETY: the schema reserves size_of::<Summary>() bytes at `out`, with
    // no alignment guarantee.
    unsafe { out.cast::<Summary>().write_unaligned(summary) };
}

const EMPTY: (Summary, f32) = (Summary { total: 0.0, peak_column: u32::MAX, nonzeros: 0 }, f32::MIN);

// #region csr
unsafe extern "C" fn summarize_csr(call: *const CsrRow) -> i32 {
    // A panic must not unwind into the loader: report it as a status instead.
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the loader passes a valid CsrRow for the call's duration.
        let row = unsafe { &*call };
        let Some(settings) = settings(&row.context) else { return 1 };
        let n = row.nnz as usize;
        // SAFETY: full (non-partial) rows carry nnz columns and f32 values.
        let (columns, values) = unsafe {
            (
                std::slice::from_raw_parts(row.columns, n),
                std::slice::from_raw_parts(row.values.cast::<f32>(), n),
            )
        };
        let mut acc = EMPTY;
        for (&column, &value) in columns.iter().zip(values) {
            fold(&mut acc, column, value);
        }
        finish(acc, settings.scale, row.out);
        0
    }))
    .unwrap_or(2)
}
// #endregion csr

// #region dense
unsafe extern "C" fn summarize_dense(call: *const DenseRow) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the loader passes a valid DenseRow for the call's duration.
        let row = unsafe { &*call };
        let Some(settings) = settings(&row.context) else { return 1 };
        // SAFETY: dense rows carry out_dim values of the schema dtype.
        let values =
            unsafe { std::slice::from_raw_parts(row.values.cast::<f32>(), row.out_dim as usize) };
        let mut acc = EMPTY;
        for (column, &value) in values.iter().enumerate() {
            if value != 0.0 {
                fold(&mut acc, column as u32, value);
            }
        }
        finish(acc, settings.scale, row.out);
        0
    }))
    .unwrap_or(2)
}
// #endregion dense

// #region access
/// Native access order: position `p` loads cell `p * stride mod cells`.
#[repr(C)]
struct Stride {
    cells: u64,
    stride: u64,
}

unsafe extern "C" fn strided(context: *mut c_void, start: u64, count: u64, out: *mut CellId) -> i32 {
    // SAFETY: context is the Stride below, alive and read-only for the run.
    let stride = unsafe { &*context.cast::<Stride>() };
    // SAFETY: the loader passes exactly `count` writable cells.
    let out = unsafe { std::slice::from_raw_parts_mut(out, count as usize) };
    for (i, cell) in out.iter_mut().enumerate() {
        *cell = ((start + i as u64) * stride.stride % stride.cells) as CellId;
    }
    0
}
// #endregion access

fn main() -> Result<(), Box<dyn Error>> {
    let path = common::fixture()?;
    let mut builder = DataSourceBuilder::new(4, 4096)?.direct_io(false);
    builder.add_cpk(path, &[0, 1, 2, 3])?;

    // #region bind
    let settings = Arc::new(Settings { scale: 0.5 });
    let user_data = Arc::as_ptr(&settings).cast::<c_void>();
    // SAFETY: both callbacks are stateless apart from `settings`, which the
    // owner Arc keeps alive; they catch panics, write only 12 output bytes and
    // retain no row pointer.
    let process = unsafe { ProcessFn::new(summarize_csr, summarize_dense, user_data, settings) };
    let cell_bytes = size_of::<Summary>();
    // Bump the identity whenever callback behavior or `scale` changes.
    let schema = OutputSchema::process(1, vec![0, 1, 2, 3], Dtype::F32, cell_bytes, process, 0x5eed_0001, 0)?;
    // #endregion bind

    let config = LoaderConfig {
        workers: 2,
        backend: IoBackend::Pread,
        max_inflight_per_worker: 2,
        slot_cells: 2,
        ..LoaderConfig::default()
    };
    let loader = Loader::new(Arc::new(builder.build()?), vec![schema], config)?;

    // #region sequence
    let order = Stride { cells: 4, stride: 3 };
    // SAFETY: `strided` is pure, thread-safe and writes exactly `count` ids;
    // `order` outlives the run below.
    let accesses = unsafe { ExternAccesses::new(strided, (&raw const order).cast_mut().cast(), 4) };
    // #endregion sequence

    let ring = Arc::new(Ring::allocate(loader.ring_layout(2, 1))?);
    let run = loader.prepare(&accesses, ring, Planning::None)?;
    let mut summaries = Vec::new();
    run.for_each_batch(|batch| {
        for record in batch.region(0).chunks_exact(cell_bytes) {
            // SAFETY: each record is one Summary written by the callbacks.
            summaries.push(unsafe { record.as_ptr().cast::<Summary>().read_unaligned() });
        }
        Ok(())
    })?;

    let expect = |total, peak_column| Summary { total, peak_column, nonzeros: 2 };
    // Cells 0, 3, 2, 1 in strided order; totals scaled by 0.5.
    assert_eq!(summaries, [expect(1.5, 1), expect(7.5, 1), expect(5.5, 3), expect(3.5, 2)]);
    println!("summaries: {summaries:?}");
    Ok(())
}
