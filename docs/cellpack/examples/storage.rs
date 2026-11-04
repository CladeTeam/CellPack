mod common;

use std::error::Error;

use cellpack::format::{Dtype, SourceLayout};
use cellpack::scans::{CsrRef, Indices, UNMAPPED};
use cellpack::store::{CpkFile, CpkWriter, ReadOptions, ReadWorkspace, RowSelection};

fn main() -> Result<(), Box<dyn Error>> {
    let path = common::fixture()?;
    let file = CpkFile::open(&path)?;
    let rows = [3, 0, 3];
    let map = [1, UNMAPPED, 0, UNMAPPED];
    let selection = file.prepare_selection(
        RowSelection::List(&rows),
        Some(&map),
        2,
        ReadOptions::default(),
    )?;
    let spec = selection.spec();
    let mut workspace = ReadWorkspace::default();
    let mut dense = vec![0.0f64; spec.dense_len()?];
    selection.read_dense_into(&mut dense, &mut workspace, 2)?;
    assert_eq!(dense, [0.0, 7.0, 0.0, 0.0, 0.0, 7.0]);

    let mut indptr = vec![0u64; spec.rows + 1];
    let mut indices = vec![0u32; spec.nnz];
    let mut values = vec![0.0f64; spec.nnz];
    let nnz = selection.read_csr_into(&mut indptr, &mut indices, &mut values, &mut workspace, 2)?;
    indices.truncate(nnz);
    values.truncate(nnz);
    assert_eq!(indptr, [0, 1, 1, 2]);
    assert_eq!(indices, [1, 1]);
    assert_eq!(values, [7.0, 7.0]);

    let csr_path = common::directory()?.join("counts.cpk");
    let indptr = [0u64, 2, 3];
    let indices = [0u32, 3, 2];
    let data: Vec<u8> = [2u16, 9, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut writer = CpkWriter::create(
        &csr_path,
        4,
        Dtype::U16,
        SourceLayout::Csr,
        common::options(),
    )?;
    writer.append(CsrRef {
        n_genes: 4,
        value_width: 2,
        indptr: &indptr,
        indices: Indices::U32(&indices),
        data: &data,
    })?;
    writer.finish()?;
    let counts = CpkFile::open(csr_path)?;
    let mut out = [0u16; 8];
    counts
        .prepare_read(0..2)?
        .read_dense_into(&mut out, &mut workspace, 1)?;
    assert_eq!(out, [2, 0, 0, 9, 0, 0, 4, 0]);
    println!("selection, conversion and CSR round-trip passed");
    Ok(())
}
