mod common;

use std::error::Error;
use cellpack::format::{Dtype, SourceLayout};
use cellpack::scans::DenseRef;
use cellpack::store::{CpkFile, CpkWriter, ReadWorkspace};

fn main() -> Result<(), Box<dyn Error>> {
    let path = common::directory()?.join("example.cpk");
    let values = [0.0f32, 2.0, 0.0, 1.0, 0.0, 3.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut writer = CpkWriter::create(
        &path, 3, Dtype::F32, SourceLayout::Dense, common::options(),
    )?;
    writer.append(DenseRef { n_genes: 3, value_width: 4, data: &bytes })?;
    writer.finish()?;

    let file = CpkFile::open(&path)?;
    let selection = file.prepare_read(0..2)?;
    let mut out = vec![0.0f32; selection.spec().dense_len()?];
    selection.read_dense_into(&mut out, &mut ReadWorkspace::default(), 1)?;
    let csr = file.read_csr(0..2)?;
    println!("shape: {} × {}", file.n_rows(), file.n_cols());
    println!("dense: {out:?}");
    println!("CSR offsets: {:?}", csr.indptr);
    println!("CSR columns: {:?}", csr.indices);
    Ok(())
}
