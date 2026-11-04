mod common;

use std::error::Error;
use std::sync::Arc;
use cellpack::loader::{
    DataSourceBuilder, Dtype, IoBackend, Loader, LoaderConfig,
    OutputSchema, Planning, Replicate, Ring,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = DataSourceBuilder::new(5, 4096)?.direct_io(false);
    builder.add_cpk(common::fixture()?, &[0, 1, 2, 3])?;
    let source = Arc::new(builder.build()?);
    let schemas = vec![
        OutputSchema::dense(1, vec![3, 1, 4], Dtype::F32)?.with_default(f64::NAN),
        OutputSchema::dense(2, vec![0, 2], Dtype::F64)?,
    ];
    let loader = Loader::new(source, schemas, LoaderConfig {
        workers: 2, backend: IoBackend::Pread, slot_cells: 2,
        ..LoaderConfig::default()
    })?;
    let cells = [3u32, 0, 2, 3, 1];
    let accesses = Replicate::new(cells, 2)?;
    let ring = Arc::new(Ring::allocate(loader.ring_layout(2, 1))?);
    let run = loader.prepare(&accesses, ring, Planning::None)?;
    let stats = run.for_each_batch(|batch| {
        let panel: Vec<f32> = batch.region(0).chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap())).collect();
        println!("positions {}..{}: panel {panel:?}",
            batch.first_position(), batch.first_position() + batch.rows() as u64);
        Ok(())
    })?;
    println!("delivered: {} positions, {} accesses", cells.len(), stats.counters.accesses);
    Ok(())
}
