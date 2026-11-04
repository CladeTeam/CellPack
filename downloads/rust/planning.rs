mod common;

use std::error::Error;
use std::sync::Arc;

use cellpack::loader::{
    DataSourceBuilder, Dtype, IoBackend, Loader, LoaderConfig, OutputSchema, Planning, Ring,
    RunStats,
};
use cellpack::opt::{Plan, PlanObjective};

fn consume(
    loader: &Loader,
    accesses: &[u32],
    planning: Planning,
) -> Result<RunStats, Box<dyn Error>> {
    let ring = Arc::new(Ring::allocate(loader.ring_layout(2, 1))?);
    let run = loader.prepare(&accesses, ring, planning)?;
    let stats = run.for_each_batch(|batch| {
        for (row, bytes) in batch.region(0).chunks_exact(16).enumerate() {
            let position = batch.first_position() as usize + row;
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(values, common::MATRIX[accesses[position] as usize]);
        }
        Ok(())
    })?;
    Ok(stats)
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = common::fixture()?;
    let mut builder = DataSourceBuilder::new(4, 4096)?.direct_io(false);
    builder.add_cpk(path, &[0, 1, 2, 3])?;
    let source = Arc::new(builder.build()?);
    let schema = OutputSchema::dense(1, vec![0, 1, 2, 3], Dtype::F32)?;
    let config = LoaderConfig {
        workers: 2,
        backend: IoBackend::Pread,
        max_inflight_per_worker: 2,
        slot_cells: 2,
        ..LoaderConfig::default()
    };
    let loader = Loader::new(source, vec![schema], config)?;
    let accesses: Vec<u32> = (0..64).map(|i| (i % 4) as u32).collect();
    let uncached = consume(&loader, &accesses, Planning::None)?;

    let mut options = loader.plan_options(16 * 4096);
    options.threads = 2;
    options.objective = PlanObjective::Misses;
    let plan = loader.compile(&accesses, &options)?;
    let summary = loader.replay(&plan, &accesses)?;
    let path = common::directory()?.join("cache.plan");
    plan.write(&path)?;
    let fixed = Planning::Fixed(Arc::new(Plan::open(path)?));
    let memory = loader.memory_estimate_for(&fixed, 2, 1)?;
    let cached = consume(&loader, &accesses, fixed)?;
    assert_eq!(cached.counters.page_hits, summary.page_hits);
    assert_eq!(cached.io.requested_bytes, summary.read_bytes);
    assert_eq!(cached.io.requests, summary.requests);

    let mut row_options = loader.plan_options(16 * 4096);
    row_options.threads = 2;
    row_options.rows = true;
    let row_plan = loader.compile(&accesses, &row_options)?;
    let rows = consume(&loader, &accesses, Planning::Fixed(Arc::new(row_plan)))?;

    options.chunk = Some(8);
    options.horizon = Some(16);
    let online = consume(&loader, &accesses, loader.online(options)?)?;
    assert_eq!(online.counters.accesses, accesses.len() as u64);
    println!(
        "page reads: uncached={}, fixed={}, online={}; row hits={}; reservations={}",
        uncached.counters.page_reads,
        cached.counters.page_reads,
        online.counters.page_reads,
        rows.counters.row_hits,
        memory.known_reservation_bytes
    );
    Ok(())
}
