mod common;

use std::error::Error;
use std::sync::Arc;

use cellpack::loader::{
    DataSourceBuilder, Dtype, IoBackend, Loader, LoaderConfig, OutputSchema, Planning, Replicate,
    Ring,
};
use cellpack::prof::SiteMask;

fn main() -> Result<(), Box<dyn Error>> {
    let path = common::fixture()?;
    let mut builder = DataSourceBuilder::new(5, 4096)?.direct_io(false);
    builder.add_cpk(&path, &[0, 1, 2, 3])?;
    let source = Arc::new(builder.build()?);
    let schemas = vec![
        OutputSchema::dense(1, vec![3, 1, 4], Dtype::F32)?
            .with_default(f64::NAN)
            .named("panel"),
        OutputSchema::dense(2, vec![0, 2], Dtype::F64)?.named("pair"),
    ];
    let config = LoaderConfig {
        workers: 2,
        backend: IoBackend::Pread,
        max_inflight_per_worker: 2,
        slot_cells: 2,
        profile: Some(SiteMask::parse(&["io", "cpu", "wait"])?),
        ..LoaderConfig::default()
    };
    let loader = Loader::new(source, schemas, config)?;
    let cells = [3u32, 0, 2, 3, 1];
    let accesses = Replicate::new(cells, 2)?;
    let ring = Arc::new(Ring::allocate(loader.ring_layout(2, 1))?);
    let run = loader.prepare(&accesses, ring, Planning::None)?;
    let mut positions = 0;
    let stats = run.for_each_batch(|batch| {
        let panel: Vec<f32> = batch
            .region(0)
            .chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
            .collect();
        let pair: Vec<f64> = batch
            .region(1)
            .chunks_exact(8)
            .map(|v| f64::from_ne_bytes(v.try_into().unwrap()))
            .collect();
        for row in 0..batch.rows() {
            let position = batch.first_position() as usize + row;
            let input = common::MATRIX[cells[position] as usize];
            assert_eq!(panel[row * 3], input[3]);
            assert_eq!(panel[row * 3 + 1], input[1]);
            assert!(panel[row * 3 + 2].is_nan());
            assert_eq!(
                &pair[row * 2..row * 2 + 2],
                &[input[0] as f64, input[2] as f64]
            );
        }
        positions += batch.rows();
        Ok(())
    })?;
    assert_eq!(positions, cells.as_slice().len());
    assert_eq!(stats.counters.accesses, 10);
    assert!(stats.profile.is_some());
    println!("loaded {positions} positions into two schemas, including a partial last batch");
    Ok(())
}
