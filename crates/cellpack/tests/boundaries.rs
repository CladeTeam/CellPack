use cellpack::Error;
use cellpack::format::*;
use cellpack::scans::*;

fn encode_rows(
    ids: &[u32],
    values: &[u8],
    indptr: &[u64],
    genes: u32,
    width: usize,
    renumber: bool,
    mu: f32,
) -> Vec<u8> {
    let src = CsrRef {
        n_genes: genes,
        value_width: width,
        indptr,
        indices: Indices::U32(ids),
        data: values,
    };
    let opt = EncodeOptions {
        renumber,
        thresholds: Thresholds {
            indices: mu,
            values: mu,
        },
        max_lanes: 32,
        lane_min: 32,
        column_modes: None,
        layout: LayoutChoice::Auto,
        values: ValueChoice::Auto,
    };
    let mut out = vec![0; chunk_bound(src, Rows::All, &opt).unwrap()];
    let mut ws = vec![0; chunk_workspace_words(src, Rows::All, &opt).unwrap()];
    let len = encode_chunk(src, Rows::All, &opt, &mut out, &mut ws).unwrap();
    out.truncate(len);
    out
}

#[test]
fn records_can_end_in_a_short_final_page() {
    let ids: Vec<u32> = (0..256).collect();
    let values: Vec<u8> = (0..256u32).flat_map(|i| (i * 7919).to_le_bytes()).collect();
    let chunk = encode_rows(&ids, &values, &[0, 256], 512, 4, true, 0.0);
    let meta = parse_meta(&chunk).unwrap();
    let tables = decode_tables(&meta).unwrap();
    let rec = meta.record(meta.records_of(&chunk).unwrap(), 0).unwrap();
    for start in [0, 17, 511] {
        let mut bytes = vec![0; start];
        bytes.extend_from_slice(rec);
        let pages: Vec<_> = bytes.chunks(512).collect();
        assert!(pages.last().unwrap().len() < 512);
        let paged = PagedRecord {
            pages: &pages,
            page_size: 512,
            start,
            len: rec.len(),
        };
        let mut ws = vec![0; decode_workspace_words(&meta)];
        let (mut ix, mut vx) = (vec![0u32; 256], vec![0; values.len()]);
        assert_eq!(
            decode_record_pages(
                &meta,
                &tables,
                paged,
                IdSpace::Source,
                &mut ix,
                &mut vx,
                &mut ws
            )
            .unwrap(),
            256
        );
        assert_eq!(ix, ids);
        assert_eq!(vx, values);
        let parts = decode_parts_pages(&meta, &tables, paged, IdSpace::Source, &mut ws).unwrap();
        parts.expand_values(&mut vx).unwrap();
        assert_eq!(vx, values);
        let mut row = vec![0; 512 * 4];
        decode_record_dense_pages(
            &meta,
            &tables,
            paged,
            IdSpace::Source,
            &mut row,
            Fill::Zero,
            &mut ws,
        )
        .unwrap();
        assert_eq!(&row[..values.len()], &values);
        assert!(row[values.len()..].iter().all(|&b| b == 0));
    }
}

#[test]
fn malformed_nnz_prefix_is_rejected_before_slicing_outputs() {
    let mut chunk = encode_rows(&[0, 1, 0, 1], &[1, 2, 3, 4], &[0, 2, 4], 4, 1, false, 0.0);
    let len = parse_meta(&chunk).unwrap().meta_len();
    chunk[32..40].copy_from_slice(&0u64.to_le_bytes());
    chunk[len - 8..len].copy_from_slice(&0u64.to_le_bytes());
    let meta = parse_meta(&chunk).unwrap();
    let tables = decode_tables(&meta).unwrap();
    let mut ws = vec![0; decode_workspace_words(&meta)];
    assert!(meta.cell_nnz(0).is_err());
    assert!(
        decode_csr(
            &meta,
            &tables,
            meta.records_of(&chunk).unwrap(),
            Rows::All,
            IdSpace::Source,
            &mut [0u64; 3],
            &mut [] as &mut [u32],
            &mut [],
            &mut ws
        )
        .is_err()
    );
}

#[test]
fn overflowing_chunk_extent_is_rejected() {
    let mut chunk = encode_rows(&[0], &[1], &[0, 1], 4, 1, false, 0.0);
    chunk[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(parse_meta(&chunk).is_err());
}

#[test]
fn tables_are_bound_to_the_model_and_gene_permutation() {
    let a: Vec<u32> = (0..128).collect();
    let b: Vec<u32> = (128..256).collect();
    let ca = encode_rows(&a, &[1; 128], &[0, 128], 512, 1, true, 1.0);
    let cb = encode_rows(&b, &[1; 128], &[0, 128], 512, 1, true, 1.0);
    let (ma, mb) = (parse_meta(&ca).unwrap(), parse_meta(&cb).unwrap());
    let (dt, et) = (decode_tables(&ma).unwrap(), encode_tables(&ma).unwrap());
    let rec = mb.record(mb.records_of(&cb).unwrap(), 0).unwrap();
    let mut ws = vec![0; decode_workspace_words(&mb)];
    assert_eq!(
        decode_record(
            &mb,
            &dt,
            rec,
            IdSpace::Source,
            &mut [0u32; 128],
            &mut [0; 128],
            &mut ws
        ),
        Err(Error::TablesMismatch)
    );
    let cell = CellRef {
        indices: Indices::U32(&b),
        values: &[1; 128],
    };
    let mut ws = vec![0; encode_workspace_words(512, 128)];
    let mut out = vec![0; record_bound(512, 1, 128, false)];
    assert_eq!(
        encode_record(&mb, &et, cell, &record_options(), &mut out, &mut ws),
        Err(Error::TablesMismatch)
    );

    let cc = encode_rows(&a, &[2; 128], &[0, 128], 512, 1, true, 1.0);
    let mc = parse_meta(&cc).unwrap();
    let mut ws = vec![0; decode_workspace_words(&mc)];
    assert_eq!(
        decode_record(
            &mc,
            &dt,
            mc.record(mc.records_of(&cc).unwrap(), 0).unwrap(),
            IdSpace::Source,
            &mut [0u32; 128],
            &mut [0; 128],
            &mut ws
        ),
        Err(Error::TablesMismatch)
    );
}

#[test]
fn record_encoder_respects_chunk_maxima() {
    let ids: Vec<u32> = (0..512).collect();
    let values: Vec<u8> = (0..512u64).flat_map(|i| (i % 200).to_le_bytes()).collect();
    let chunk = encode_rows(&ids, &values, &[0, 512], 1024, 8, false, 1.0);
    let meta = parse_meta(&chunk).unwrap();
    assert_eq!(meta.max_k(), 200);
    let et = encode_tables(&meta).unwrap();
    let dt = decode_tables(&meta).unwrap();
    let values: Vec<u8> = (0..512u64).flat_map(|i| (i % 201).to_le_bytes()).collect();
    let cell = CellRef {
        indices: Indices::U32(&ids),
        values: &values,
    };
    let mut out = vec![0; record_bound(1024, 8, 512, true)];
    let mut ws = vec![0; encode_workspace_words(1024, meta.max_elems())];
    let len = encode_record(&meta, &et, cell, &record_options(), &mut out, &mut ws).unwrap();
    let (mut ix, mut vx) = (vec![0u32; 512], vec![0; values.len()]);
    let mut dws = vec![0; decode_workspace_words(&meta)];
    decode_record(
        &meta,
        &dt,
        &out[..len],
        IdSpace::Source,
        &mut ix,
        &mut vx,
        &mut dws,
    )
    .unwrap();
    assert_eq!(ix, ids);
    assert_eq!(vx, values);

    let ids: Vec<u32> = (0..513).collect();
    let values = vec![0; 513 * 8];
    let cell = CellRef {
        indices: Indices::U32(&ids),
        values: &values,
    };
    let mut ws = vec![0; encode_workspace_words(1024, meta.max_elems())];
    let mut out = vec![0; record_bound(1024, 8, 513, true)];
    assert!(matches!(
        encode_record(&meta, &et, cell, &record_options(), &mut out, &mut ws),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn parts_expansion_respects_exact_output_extents() {
    let ids: Vec<u32> = (0..1025).collect();
    for width in [1, 2, 4, 8] {
        for k in [1, 8, 200] {
            let values: Vec<u8> = (0..1025u64)
                .flat_map(|i| (i % k).to_le_bytes()[..width].to_vec())
                .collect();
            let chunk = encode_rows(&ids, &values, &[0, 1025], 2048, width, false, 1.0);
            let meta = parse_meta(&chunk).unwrap();
            let tables = decode_tables(&meta).unwrap();
            let rec = meta.record(meta.records_of(&chunk).unwrap(), 0).unwrap();
            let mut ws = vec![0; decode_workspace_words(&meta)];
            let parts = decode_parts(&meta, &tables, rec, IdSpace::Source, &mut ws).unwrap();
            let mut out = vec![0xa5; values.len() + 32];
            assert!(parts.expand_values(&mut out[..values.len() - 1]).is_err());
            assert!(out.iter().all(|&v| v == 0xa5));
            parts.expand_values(&mut out[..values.len()]).unwrap();
            assert_eq!(&out[..values.len()], &values);
            assert!(out[values.len()..].iter().all(|&v| v == 0xa5));
        }
    }
}

/// Coding choices for single records: every coding allowed, no thresholds.
fn record_options() -> EncodeOptions {
    EncodeOptions {
        renumber: false,
        thresholds: Thresholds {
            indices: 1.0,
            values: 1.0,
        },
        max_lanes: 32,
        lane_min: 32,
        column_modes: None,
        layout: LayoutChoice::Auto,
        values: ValueChoice::Auto,
    }
}
