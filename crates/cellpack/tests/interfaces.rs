use cellpack::format::parse_meta;
use cellpack::scans::*;

#[test]
fn encoding_buffers_and_gene_order_compose() {
    let values = [0u16, 3, 0, 2, 0, 3, 1, 2, 0, 3, 0, 2];
    let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let input = DenseRef { n_genes: 4, value_width: 2, data: &data };
    let options = EncodeOptions {
        renumber: true,
        thresholds: Thresholds { indices: DEFAULT_MU_INDICES, values: DEFAULT_MU_VALUES },
        max_lanes: DEFAULT_MAX_LANES,
        lane_min: DEFAULT_LANE_MIN,
        column_modes: None,
        layout: LayoutChoice::Auto,
        values: ValueChoice::Auto,
    };
    let sizes = chunk_sizes(input, Rows::All, &options).unwrap();
    let mut encoded = vec![0; sizes.output_bytes];
    let mut scratch = vec![0; sizes.workspace_words];
    let n = encode_chunk(input, Rows::All, &options, &mut encoded, &mut scratch).unwrap();
    let meta = parse_meta(&encoded[..n]).unwrap();
    let encoding = encode_tables(&meta).unwrap();
    let decoding = decode_tables(&meta).unwrap();
    let source = decoding.source_of_coded().unwrap();
    let coded = encoding.coded_of_source().unwrap();
    for (i, &s) in source.iter().enumerate() {
        assert_eq!(coded[s as usize], i as u32);
    }
    let mut decoded = vec![0; dense_len(&meta, Rows::All).unwrap()];
    let mut workspace = vec![0; decode_workspace_words(&meta)];
    decode_dense(&meta, &decoding, meta.records_of(&encoded[..n]).unwrap(), Rows::All,
        IdSpace::Source, &mut decoded, Fill::Zero, &mut workspace).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn owned_metadata_and_prepared_encoder_are_composable() {
    let data = vec![1u8,0,2,0,1,0,2,0,3,0,2,0];
    let source = DenseRef { n_genes: 4, value_width: 1, data: &data };
    let options = EncodeOptions { renumber: true, thresholds: Thresholds { indices: DEFAULT_MU_INDICES, values: DEFAULT_MU_VALUES }, max_lanes: DEFAULT_MAX_LANES, lane_min: DEFAULT_LANE_MIN, column_modes: None, layout: LayoutChoice::Auto, values: ValueChoice::Auto };
    let encoder = PreparedEncoder::new(source, Rows::All, options).unwrap();
    let sizes = encoder.sizes(); let mut a=vec![0;sizes.output_bytes]; let mut b=a.clone(); let mut ws=vec![0;sizes.workspace_words];
    let na=encoder.encode_into(&mut a,&mut ws).unwrap(); let nb=encode_chunk(source,Rows::All,&options,&mut b,&mut ws).unwrap();
    assert_eq!(&a[..na],&b[..nb]);
    let owned=cellpack::format::OwnedMeta::new(&a[..na]).unwrap(); let meta=owned.view();
    let tables=decode_tables(&meta).unwrap(); let decoder=Decoder::new(&meta,&tables).unwrap();
    let mut scratch=vec![0;decoder.workspace_words()]; let mut row=vec![0;4];
    let record=meta.record(meta.records_of(&a[..na]).unwrap(),2).unwrap();
    decoder.dense(record,IdSpace::Source,&mut row,Fill::Zero,&mut scratch).unwrap();
    assert_eq!(row,&data[8..12]);
}

struct CountingAllocator;
thread_local! {
    static TRACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
fn count_allocation() {
    if TRACK.try_with(|v| v.get()).unwrap_or(false) { let _ = ALLOCATIONS.try_with(|n| n.set(n.get()+1)); }
}
// SAFETY: every allocation is forwarded unchanged to System; counters are thread-local.
unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        count_allocation();
        // SAFETY: caller supplies the allocation layout required by GlobalAlloc.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        count_allocation();
        // SAFETY: same contract, with System providing zero initialization.
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        count_allocation();
        // SAFETY: ptr/layout/size retain the caller's GlobalAlloc contract.
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: ptr/layout refer to a matching System allocation.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn prepared_encoder_reuses_all_caller_buffers() {
    let data: Vec<u8> = (0..512).map(|i| (i%9) as u8).collect();
    let source=DenseRef{n_genes:32,value_width:1,data:&data};
    let options=EncodeOptions{renumber:true,thresholds:Thresholds{indices:DEFAULT_MU_INDICES,values:DEFAULT_MU_VALUES},max_lanes:DEFAULT_MAX_LANES,lane_min:DEFAULT_LANE_MIN,column_modes:None,layout:LayoutChoice::Auto,values:ValueChoice::Auto};
    let encoder=PreparedEncoder::new(source,Rows::All,options).unwrap();
    let mut out=vec![0;encoder.sizes().output_bytes];let mut ws=vec![0;encoder.sizes().workspace_words];
    encoder.encode_into(&mut out,&mut ws).unwrap();
    ALLOCATIONS.with(|n|n.set(0));TRACK.with(|v|v.set(true));
    let result=encoder.encode_into(&mut out,&mut ws);
    TRACK.with(|v|v.set(false));
    result.unwrap();
    assert_eq!(ALLOCATIONS.with(|n|n.get()),0);
}
