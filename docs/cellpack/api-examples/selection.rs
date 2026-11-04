mod common;

use std::error::Error;
use cellpack::scans::UNMAPPED;
use cellpack::store::{CpkFile, ReadOptions, ReadWorkspace, RowSelection};

fn main() -> Result<(), Box<dyn Error>> {
    let file = CpkFile::open(common::fixture()?)?;
    let rows = [3, 0, 3];
    // Each entry maps a source column to its output position.
    let map = [1, UNMAPPED, 0, UNMAPPED];
    let selection = file.prepare_selection(
        RowSelection::List(&rows), Some(&map), 2, ReadOptions::default(),
    )?;
    let spec = selection.spec();
    let mut workspace = ReadWorkspace::default();
    let mut out = vec![0.0f64; spec.dense_len()?];
    selection.read_dense_into(&mut out, &mut workspace, 1)?;
    let mut offsets = vec![0u64; spec.rows + 1];
    let mut columns = vec![0u32; spec.nnz];
    let mut values = vec![0.0f64; spec.nnz];
    let nnz = selection.read_csr_into(
        &mut offsets, &mut columns, &mut values, &mut workspace, 1,
    )?;
    println!("rows: {rows:?}");
    println!("output: {out:?}");
    println!("CSR offsets: {offsets:?}");
    println!("actual nnz: {nnz}");
    Ok(())
}
