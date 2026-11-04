mod common;

use std::error::Error;
use std::sync::Arc;

use cellpack::loader::{
    self, DataSourceBuilder, Dtype, IoBackend, Loader, LoaderConfig, OutputSchema, Planning, Ring,
};
use cellpack::scans::*;

fn main() -> Result<(), Box<dyn Error>> {
    // #region encoder
    // Encoders validate every in-memory matrix: here row 0 lists column 3
    // before column 1.
    let indptr = [0u64, 2];
    let indices = [3u32, 1];
    let values: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let unsorted = CsrRef {
        n_genes: 4,
        value_width: 4,
        indptr: &indptr,
        indices: Indices::U32(&indices),
        data: &values,
    };
    let options = common::options().encode;
    let encoded = chunk_sizes(&unsorted, Rows::All, &options).and_then(|sizes| {
        let mut out = vec![0u8; sizes.output_bytes];
        let mut workspace = vec![0u64; sizes.workspace_words];
        encode_chunk(&unsorted, Rows::All, &options, &mut out, &mut workspace)
    });
    assert!(matches!(encoded, Err(cellpack::Error::InvalidInput(_))));
    // #endregion encoder

    // #region map
    // Column maps are checked when a file is registered: two source columns
    // may not land on the same standard id.
    let path = common::fixture()?;
    let mut builder = DataSourceBuilder::new(4, 4096)?.direct_io(false);
    assert!(builder.add_cpk(&path, &[0, 0, 2, 3]).is_err());
    builder.add_cpk(&path, &[0, 1, 2, 3])?;
    // #endregion map

    // #region access
    // Cell ids are checked before they index the catalog.
    let schema = OutputSchema::dense(1, vec![0, 1, 2, 3], Dtype::F32)?;
    let config = LoaderConfig {
        workers: 1,
        backend: IoBackend::Pread,
        slot_cells: 2,
        ..LoaderConfig::default()
    };
    let loader = Loader::new(Arc::new(builder.build()?), vec![schema], config)?;
    let ring = Arc::new(Ring::allocate(loader.ring_layout(2, 1))?);
    let accesses = [0u32, 9];
    match loader.prepare(&accesses, ring, Planning::None) {
        Err(loader::Error::Access { position, cell }) => assert_eq!((position, cell), (1, 9)),
        Err(other) => return Err(other.into()),
        Ok(_) => panic!("cell 9 is outside a four-cell source"),
    }
    // #endregion access

    println!("unsorted CSR, duplicate map and unknown cell were all rejected");
    Ok(())
}
