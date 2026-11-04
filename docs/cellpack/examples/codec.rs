mod common;

use std::error::Error;

use cellpack::format::parse_meta;
use cellpack::scans::*;

fn main() -> Result<(), Box<dyn Error>> {
    let indptr = [0u64, 3, 5];
    let indices = [0u16, 2, 5, 1, 2];
    let values: Vec<u8> = [1u16, 1, 3, 2, 1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let source = CsrRef {
        n_genes: 6,
        value_width: 2,
        indptr: &indptr,
        indices: Indices::U16(&indices),
        data: &values,
    };
    let options = common::options().encode;
    let sizes = chunk_sizes(&source, Rows::All, &options)?;
    let mut bytes = vec![0u8; sizes.output_bytes];
    let mut encode_scratch = vec![0u64; sizes.workspace_words];
    let length = encode_chunk(
        &source,
        Rows::All,
        &options,
        &mut bytes,
        &mut encode_scratch,
    )?;
    bytes.truncate(length);

    let length = packed_len(&bytes, 4096)?;
    let mut packed = vec![0u8; length];
    let length = pack_chunk(&bytes, 4096, &mut packed)?;
    packed.truncate(length);
    let meta = parse_meta(&packed)?;
    let tables = decode_tables(&meta)?;
    let records = meta.records_of(&packed)?;
    let mut workspace = vec![0u64; decode_workspace_words(&meta)];
    let nnz = selection_nnz(&meta, Rows::All)? as usize;
    let mut out_indptr = [0u64; 3];
    let mut out_indices = vec![0u32; nnz];
    let mut out_values = vec![0u8; nnz * 2];
    decode_csr(
        &meta,
        &tables,
        records,
        Rows::All,
        IdSpace::Source,
        &mut out_indptr,
        &mut out_indices,
        &mut out_values,
        &mut workspace,
    )?;
    assert_eq!(out_indptr, indptr);
    assert_eq!(out_indices, [0, 2, 5, 1, 2]);
    assert_eq!(out_values, values);
    println!("packed chunk round-trip: {length} bytes");
    Ok(())
}
