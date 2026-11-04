#![allow(dead_code)]

use std::error::Error;
use std::path::PathBuf;

use cellpack::format::{Dtype, SourceLayout};
use cellpack::scans::*;
use cellpack::store::{CpkOptions, CpkWriter};

pub const MATRIX: [[f32; 4]; 4] = [
    [0.0, 2.0, 0.0, 1.0],
    [3.0, 0.0, 4.0, 0.0],
    [0.0, 0.0, 5.0, 6.0],
    [7.0, 8.0, 0.0, 0.0],
];

pub fn options() -> CpkOptions {
    CpkOptions {
        encode: EncodeOptions {
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
        },
        chunk_rows: 2,
        page_size: 4096,
    }
}

pub fn directory() -> Result<PathBuf, Box<dyn Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("pass an output directory")?,
    );
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

pub fn fixture() -> Result<PathBuf, Box<dyn Error>> {
    let path = directory()?.join("matrix.cpk");
    let data: Vec<u8> = MATRIX
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut writer = CpkWriter::create(&path, 4, Dtype::F32, SourceLayout::Dense, options())?;
    writer.append(DenseRef {
        n_genes: 4,
        value_width: 4,
        data: &data,
    })?;
    writer.finish()?;
    Ok(path)
}
