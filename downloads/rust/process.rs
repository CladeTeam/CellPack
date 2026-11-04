mod common;

use std::error::Error;
use std::sync::Arc;

use cellpack::loader::{
    DataSourceBuilder, DenseRow, Dtype, IoBackend, Loader, LoaderConfig, OutputSchema,
    PROCESS_ABI_VERSION, Planning, ProcessFn, Ring, RowFormat,
};

unsafe extern "C" fn sum_row(call: *const DenseRow) -> i32 {
    // SAFETY: the loader supplies a DenseRow for this callback's duration.
    let row = unsafe { &*call };
    if row.context.abi_version != PROCESS_ABI_VERSION || row.context.dtype != Dtype::F32 as u32 {
        return 1;
    }
    // SAFETY: this schema requests out_dim f32 values; the view does not escape.
    let values =
        unsafe { std::slice::from_raw_parts(row.values.cast::<f32>(), row.out_dim as usize) };
    let sum = values.iter().copied().sum::<f32>();
    // SAFETY: the schema reserves exactly four output bytes for this cell.
    unsafe { std::ptr::copy_nonoverlapping(sum.to_ne_bytes().as_ptr(), row.out, 4) };
    0
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = common::fixture()?;
    let mut builder = DataSourceBuilder::new(4, 4096)?.direct_io(false);
    builder.add_cpk(path, &[0, 1, 2, 3])?;
    // SAFETY: a static callback uses no context, is thread-safe and retains no pointers.
    let process = unsafe {
        ProcessFn::with_format(
            RowFormat::Dense,
            None,
            Some(sum_row),
            std::ptr::null(),
            Arc::new(()),
        )?
    };
    let schema = OutputSchema::process(1, vec![0, 1, 2, 3], Dtype::F32, 4, process, 1, 0)?;
    let config = LoaderConfig {
        workers: 2,
        backend: IoBackend::Pread,
        max_inflight_per_worker: 2,
        slot_cells: 2,
        ..LoaderConfig::default()
    };
    let loader = Loader::new(Arc::new(builder.build()?), vec![schema], config)?;
    let accesses = [0u32, 1, 2, 3];
    let ring = Arc::new(Ring::allocate(loader.ring_layout(2, 1))?);
    let run = loader.prepare(&accesses, ring, Planning::None)?;
    let mut sums = Vec::new();
    let stats = run.for_each_batch(|batch| {
        sums.extend(
            batch
                .region(0)
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap())),
        );
        Ok(())
    })?;
    assert_eq!(sums, [3.0, 7.0, 11.0, 15.0]);
    assert_eq!(stats.counters.process_calls, 4);
    println!("per-row sums: {sums:?}");
    Ok(())
}
