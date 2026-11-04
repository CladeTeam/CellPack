use std::error::Error;
use std::path::PathBuf;

use cellpack::format::{Dtype, SourceLayout};
use cellpack::scans::*;
use cellpack::store::{CpkFile, CpkOptions, CpkWriter, ReadWorkspace};

fn main() -> Result<(), Box<dyn Error>> {
    let directory = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("pass an output directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    let path = directory.join("matrix.cpk");
    let values = [0.0f32, 2.0, 0.0, 1.0, 0.0, 3.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let options = CpkOptions::new(EncodeOptions {
        renumber: true,
        thresholds: Thresholds {
            indices: DEFAULT_MU_INDICES,
            values: DEFAULT_MU_VALUES,
        },
        max_lanes: DEFAULT_MAX_LANES,
        lane_min: DEFAULT_LANE_MIN,
        column_modes: Some(DEFAULT_MODE_SHARE),
        layout: LayoutChoice::Auto,
        values: ValueChoice::Auto,
    });
    let mut writer = CpkWriter::create(&path, 3, Dtype::F32, SourceLayout::Dense, options)?;
    writer.append(DenseRef {
        n_genes: 3,
        value_width: 4,
        data: &bytes,
    })?;
    writer.finish()?;

    let file = CpkFile::open(&path)?;
    assert_eq!((file.n_rows(), file.n_cols()), (2, 3));
    let selection = file.prepare_read(0..2)?;
    let mut workspace = ReadWorkspace::default();
    let mut dense = vec![0.0f32; selection.spec().dense_len()?];
    selection.read_dense_into(&mut dense, &mut workspace, 1)?;
    assert_eq!(dense, values);

    let csr = file.read_csr(0..2)?;
    assert_eq!(csr.indptr, [0, 1, 3]);
    assert_eq!(csr.indices, [1, 0, 2]);
    println!(
        "round-trip: {} rows, {} columns",
        file.n_rows(),
        file.n_cols()
    );
    Ok(())
}
