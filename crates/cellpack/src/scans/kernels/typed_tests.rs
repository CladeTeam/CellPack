//! Equivalence tests of the eight-lane typed kernels against the scalar loops,
//! bit for bit, over every source and target pair they support and every value
//! encoding and shape.
//!
//! They also guard the gather preconditions: narrow gathers must not read past
//! the end of a row, which the guard-page test checks with a protected page
//! right after the values.

use super::{Dict, ModeDict, Parts, Ranks, Values};
use crate::runtime::Runtime;
use crate::scans::kernels::{Isa, typed};
use crate::scans::map::{CastInto, Element, Mapper, UNMAPPED};

fn bits<T: Element>(v: &[T]) -> &[u8] {
    // SAFETY: numeric values have no padding and the view retains the borrow.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn source<S: Element>(i: usize) -> S {
    if S::KIND == 2 {
        let x = [-0.0f64, 1.25, -19.5, 1e20, f64::INFINITY, f64::NEG_INFINITY][i % 6];
        return if S::WIDTH == 4 { S::from_le(&(x as f32).to_le_bytes()) } else { S::from_le(&x.to_le_bytes()) };
    }
    let x = [0, 1, u64::MAX, 1 << 63, (1 << 53) + 3, 0x807f_ffff_8000_ff80, 0x7fff_ffff_ffff_ffff, i as u64 * 7919][i % 8];
    S::from_le(&x.to_le_bytes())
}

macro_rules! pairs {
    ($f:ident; $($s:ty => $($t:ty),*;)*) => {$($($f::<$s, $t>();)*)*};
}

macro_rules! all_pairs {
    ($f:ident) => {
        pairs! { $f;
            u8 => u8,u16,u32,u64,i16,i32,i64,f32,f64;
            i8 => i8,u8,i16,u16,i32,u32,i64,u64,f32,f64;
            u16 => u16,u32,u64,i32,i64,f32,f64;
            i16 => i16,u16,i32,u32,i64,u64,f32,f64;
            u32 => u32,u64,i64,f32,f64;
            i32 => i32,u32,i64,u64,f32,f64;
            u64 => u64,f32,f64;
            i64 => i64,u64,f32,f64;
            f32 => f32,f64;
            f64 => f64;
        }
    };
}

fn check<S: CastInto<T>, T: Element>() {
    for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 65, 257] {
        let raw: Vec<S> = (0..n).map(source).collect();
        let entries: Vec<S> = (0..13).map(source).collect();
        let dict: Vec<T> = entries.iter().map(|&v| v.cast()).collect();
        let stored_modes: Vec<S> = (0..n).map(|i| source(i + 3)).collect();
        let modes: Vec<T> = stored_modes.iter().map(|&v| v.cast()).collect();
        let ids: Vec<u32> = (0..n as u32).collect();
        let ranks8: Vec<u8> = (0..n).map(|i| (i % 13) as u8).collect();
        let ranks32: Vec<u32> = ranks8.iter().map(|&x| x as u32).collect();
        let dict_at = |ranks| Values::Dict(Dict { padded: bits(&entries), k: 13, vw: S::WIDTH, ranks });
        let mode_at = |ranks| Values::Modes(ModeDict { padded: bits(&entries), k: 13, vw: S::WIDTH, ranks, modes: bits(&stored_modes) });
        for values in [Values::Raw(bits(&raw)), dict_at(Ranks::One), dict_at(Ranks::U8(&ranks8)), dict_at(Ranks::U32(&ranks32)), mode_at(Ranks::One), mode_at(Ranks::U8(&ranks8)), mode_at(Ranks::U32(&ranks32))] {
            for partial in [false, true] {
                let panel: Vec<u32> = (0..n).map(|i| if partial && i % 3 == 1 { UNMAPPED } else { (n - 1 - i) as u32 }).collect();
                let parts = Parts { ids: &ids, values, nnz: n, dense: false };
                let scalar = |i: usize| -> T { match values {
                    Values::Raw(_) => raw[i].cast(),
                    Values::Dict(d) => dict[match d.ranks { Ranks::One => 0, Ranks::U8(r) => r[i] as usize, Ranks::U32(r) => r[i] as usize }],
                    Values::Modes(d) => { let r = match d.ranks { Ranks::One => 0, Ranks::U8(r) => r[i] as usize, Ranks::U32(r) => r[i] as usize }; if r == 0 { modes[i] } else { dict[r-1] } }
                }};
                let zero: T = S::from_le(&[0; 8]).cast();
                let mut cols = vec![0; n];
                let mut got = vec![zero; n];
                let want: Vec<T> = (0..n).filter(|&i| panel[i] != UNMAPPED).map(scalar).collect();
                // SAFETY: all fixture indices and buffers are valid; the test
                // checks AVX-512 availability before invoking any pair.
                let kept = unsafe { typed::map::<S,T,0>(&parts, &[], &panel, &dict, &modes, &mut cols, &mut got) }.unwrap();
                assert_eq!(bits(&got[..kept]), bits(&want), "compact {} {} n={n}", std::any::type_name::<S>(), std::any::type_name::<T>());
                assert_eq!(&cols[..kept], panel.iter().copied().filter(|&c| c != UNMAPPED).collect::<Vec<_>>());
                let positions: Vec<u32> = (0..n).map(|i| ((i * 17 + 5) % n) as u32).collect();
                // SAFETY: same validated fixture, with arbitrary valid positions.
                unsafe { typed::map::<S,T,1>(&parts, &positions, &panel, &dict, &modes, &mut cols, &mut got) }.unwrap();
                let want: Vec<T> = positions.iter().map(|&p| scalar(p as usize)).collect();
                assert_eq!(bits(&got), bits(&want));
                assert_eq!(cols, positions.iter().map(|&p| panel[p as usize]).collect::<Vec<_>>());
                for dense in [false, true] {
                    let parts = Parts { dense, ..parts };
                    let fill = source::<S>(1).cast();
                    got.fill(fill);
                    let mut want = vec![fill; n];
                    for i in 0..n { if panel[i] != UNMAPPED { want[panel[i] as usize] = scalar(i); } }
                    // SAFETY: same fixture, injective output map bounded by n.
                    unsafe { typed::map::<S,T,2>(&parts, &[], &panel, &dict, &modes, &mut [], &mut got) }.unwrap();
                    assert_eq!(bits(&got), bits(&want), "scatter {} {} n={n}", std::any::type_name::<S>(), std::any::type_name::<T>());
                }
            }
        }
    }
}

#[test]
fn typed_dispatch_matrix_matches_scalar_bits() {
    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512dq")) { return; }
    all_pairs!(check);
}

#[test]
fn indexed_narrow_gathers_stop_at_the_guard_page() {
    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512dq")) { return; }
    // SAFETY: own two pages, protect the second, and keep every constructed
    // slice inside the first. The mapping remains live through all calls.
    unsafe {
        let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        let allocation = libc::mmap(std::ptr::null_mut(), 2 * page, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
        assert_ne!(allocation, libc::MAP_FAILED);
        let base = allocation.cast::<u8>();
        assert_eq!(libc::mprotect(base.add(page).cast(), page, libc::PROT_NONE), 0);
        let ids = [0, 1, 2];
        let positions = [2; 32];
        let mut cols = [0; 32];
        let mut values = [0u64; 32];
        for width in [1, 2] {
            let raw = std::slice::from_raw_parts_mut(base.add(page - 3 * width), 3 * width);
            raw.fill(0);
            raw[2 * width] = 127;
            let p = Parts { ids: &ids, values: Values::Raw(raw), nnz: 3, dense: false };
            if width == 1 { typed::map::<u8,u64,1>(&p, &positions, &ids, &[], &[], &mut cols, &mut values).unwrap(); }
            else { typed::map::<u16,u64,1>(&p, &positions, &ids, &[], &[], &mut cols, &mut values).unwrap(); }
            assert_eq!(cols, [2;32]);
            assert_eq!(values, [127;32]);
        }
        assert_eq!(libc::munmap(allocation, 2 * page), 0);
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn dispatched<S: CastInto<T>, T: Element, const SHAPE: u8>(m: &Mapper<S, T>, p: &Parts<'_>, positions: &[u32], panel: &[u32], dict: &[T], modes: &[T], cols: &mut [u32], out: &mut [T]) -> usize {
    use crate::scans::Fill;
    match SHAPE {
        0 => m.map_parts(p, panel, dict, modes, cols, out).unwrap(),
        1 => { m.map_parts_at(p, positions, panel, dict, modes, cols, out).unwrap(); positions.len() }
        _ => { m.map_dense(p, panel, dict, modes, out, Fill::AssumeZeroed).unwrap(); p.ids.len() }
    }
}

/// Checks the selected and every candidate route of each supported kernel
/// set against the scalar loops, bit for bit; times scalar against the
/// candidates (or the selection) when `timed`.
fn measure<S: CastInto<T>, T: Element, const SHAPE: u8>(p: &Parts<'_>, panel: &[u32], dict: &[T], modes: &[T], encoding: &str, timed: bool) {
    use std::{hint::black_box, time::Instant};
    let n = p.ids.len();
    let positions: Vec<u32> = (0..n as u32).rev().step_by(2).collect();
    let zero: T = S::from_le(&[0;8]).cast();
    let out0 = vec![zero; panel.len()];
    let scalar = Mapper::<S, T>::for_runtime(&Runtime::for_isa(Isa::Scalar));
    let mut scalar_out = out0.clone();
    let mut scalar_cols = vec![0; n];
    let kept = dispatched::<S,T,SHAPE>(&scalar, p, &positions, panel, dict, modes, &mut scalar_cols, &mut scalar_out);
    let end = if SHAPE == 2 { out0.len() } else { kept };
    for isa in Isa::ALL.into_iter().filter(|&isa| isa != Isa::Scalar && isa.supported()) {
        let rt = Runtime::for_isa(isa);
        for (route, mapper) in [("selected", Mapper::<S, T>::for_runtime(&rt)), ("candidate", Mapper::<S, T>::candidates(&rt))] {
            let (mut cols, mut out) = (vec![0; n], out0.clone());
            assert_eq!(dispatched::<S,T,SHAPE>(&mapper, p, &positions, panel, dict, modes, &mut cols, &mut out), kept);
            let context = format!("{isa} {route} {} -> {} {encoding} shape {SHAPE}", std::any::type_name::<S>(), std::any::type_name::<T>());
            assert_eq!(bits(&out[..end]), bits(&scalar_out[..end]), "{context}");
            if SHAPE != 2 { assert_eq!(cols[..kept], scalar_cols[..kept], "{context}"); }
        }
    }
    if !timed || n < 16 { return; }
    let selected = std::env::var_os("CELLPACK_BENCH_SELECTED").is_some();
    let isa = if std::env::var("CELLPACK_BENCH_ISA").as_deref() == Ok("avx2") { Isa::Avx2 } else { Isa::Avx512 };
    let rt = Runtime::for_isa(isa);
    let fast = if selected { Mapper::<S, T>::for_runtime(&rt) } else { Mapper::<S, T>::candidates(&rt) };
    let (mut cols, mut out) = (vec![0; n], out0.clone());
    let repeats = (1_048_576 / n.max(1)).clamp(8, 512) as u128;
    let mut times = [[0u128; 3]; 2];
    for round in [0, 1, 2] {
        for variant in if round % 2 == 0 { [0,1] } else { [1,0] } {
            let mapper = if variant == 0 { &scalar } else { &fast };
            let start = Instant::now();
            for _ in 0..repeats {
                black_box(dispatched::<S,T,SHAPE>(black_box(mapper), black_box(p), black_box(&positions), black_box(panel), black_box(dict), black_box(modes), black_box(&mut cols), black_box(&mut out)));
            }
            times[variant][round] = start.elapsed().as_nanos() / repeats;
        }
    }
    for t in &mut times { t.sort_unstable(); }
    println!("DISPATCH,{},{},{},{},{},{},{:.4}", std::any::type_name::<S>(), std::any::type_name::<T>(), encoding, SHAPE, times[0][1], times[1][1], times[0][1] as f64 / times[1][1] as f64);
}

fn exercise_pair<S: CastInto<T>, T: Element>(timed: bool) {
    let n = 2049;
    let raw: Vec<S> = (0..n).map(source).collect();
    let entries: Vec<S> = (0..127).map(source).collect();
    let dict: Vec<T> = entries.iter().map(|&v| v.cast()).collect();
    let modes: Vec<T> = (0..n).map(|i| source::<S>(i+3).cast()).collect();
    let ids: Vec<u32> = (0..n as u32).collect();
    let r8: Vec<u8> = (0..n).map(|i| ((i * 71) % 127) as u8).collect();
    let r32: Vec<u32> = r8.iter().map(|&r| r as u32).collect();
    let panel: Vec<u32> = (0..n).map(|i| if i % 5 == 2 { UNMAPPED } else { ((i * 53) % n) as u32 }).collect();
    let d = |ranks| Values::Dict(Dict { padded: bits(&entries), k:127, vw:S::WIDTH, ranks });
    let m = |ranks| Values::Modes(ModeDict { padded: bits(&entries), k:127, vw:S::WIDTH, ranks, modes:&[] });
    for (encoding, values) in [("raw", Values::Raw(bits(&raw))), ("const",d(Ranks::One)), ("dict8",d(Ranks::U8(&r8))), ("dict32",d(Ranks::U32(&r32))), ("mode",m(Ranks::One)), ("modes8",m(Ranks::U8(&r8))), ("modes32",m(Ranks::U32(&r32)))] {
        let p = Parts { ids:&ids, values, nnz:n, dense:false };
        for (mapping, panel) in [("partial", &panel[..]), ("identity", &ids[..])] {
            let label = format!("{encoding}-{mapping}");
            measure::<S,T,0>(&p, panel, &dict, &modes, &label, timed);
            measure::<S,T,1>(&p, panel, &dict, &modes, &label, timed);
            measure::<S,T,2>(&p, panel, &dict, &modes, &label, timed);
        }
    }
}

fn bench_pair<S: CastInto<T>, T: Element>() { exercise_pair::<S,T>(true); }
fn check_public_pair<S: CastInto<T>, T: Element>() { exercise_pair::<S,T>(false); }

#[test]
fn public_dispatch_matrix_matches_scalar_bits() {
    all_pairs!(check_public_pair);
}

#[test]
#[ignore = "dispatch benchmark; run on a pinned worker; CELLPACK_BENCH_ISA=avx2 measures the AVX2 set"]
fn typed_dispatch_benchmark() {
    assert!(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512dq"));
    all_pairs!(bench_pair);
}

fn real_pair<S: CastInto<T>, T: Element>() {
    use crate::format::Dtype;
    let sd = match (S::KIND, S::WIDTH) {
        (0,1)=>Dtype::U8,(0,2)=>Dtype::U16,(0,4)=>Dtype::U32,(0,8)=>Dtype::U64,
        (1,1)=>Dtype::I8,(1,2)=>Dtype::I16,(1,4)=>Dtype::I32,(1,8)=>Dtype::I64,
        (2,4)=>Dtype::F32,(2,8)=>Dtype::F64,_=>unreachable!()
    };
    for path in std::env::var("CELLPACK_DISPATCH_FILES").unwrap().split(':') {
        let file = crate::store::CpkFile::open(path).unwrap();
        if file.dtype() != sd { continue; }
        let chunk = file.read_chunk(0).unwrap();
        let meta = crate::format::parse_meta(&chunk).unwrap();
        let tables = crate::scans::decode_tables(&meta).unwrap();
        let mut ws = vec![0; crate::scans::decode_workspace_words(&meta)];
        let records = meta.records_of(&chunk).unwrap();
        let panel: Vec<u32> = (0..meta.n_dims()).map(|i| if i % 5 == 2 { UNMAPPED } else { meta.n_dims()-1-i }).collect();
        let identity: Vec<u32> = (0..meta.n_dims()).collect();
        for (mapping, panel) in [("partial", &panel[..]), ("identity", &identity[..])] {
        for row in 0..meta.n_cells().min(4) {
            let record = meta.record(records, row).unwrap();
            let p = crate::scans::decode_parts(&meta, &tables, record, crate::scans::IdSpace::Coded, &mut ws).unwrap();
            let (entry_bytes, mode_bytes, encoding) = match p.values {
                Values::Raw(_) => (&[][..],&[][..],"raw"),
                Values::Dict(d) => (d.entries(),&[][..], match d.ranks { Ranks::One=>"const",Ranks::U8(_)=>"dict8",Ranks::U32(_)=>"dict32" }),
                Values::Modes(d) => (d.entries(),d.modes(), match d.ranks { Ranks::One=>"mode",Ranks::U8(_)=>"modes8",Ranks::U32(_)=>"modes32" }),
            };
            let dict: Vec<T> = entry_bytes.chunks_exact(S::WIDTH).map(|b| S::from_le(b).cast()).collect();
            let modes: Vec<T> = mode_bytes.chunks_exact(S::WIDTH).map(|b| S::from_le(b).cast()).collect();
            let label = format!("real-record-{}-{row}-{encoding}-{mapping}",if p.dense {"dense"} else {"csr"});
            if !p.dense {
                measure::<S,T,0>(&p, panel, &dict, &modes, &label, true);
                measure::<S,T,1>(&p, panel, &dict, &modes, &label, true);
            }
            measure::<S,T,2>(&p, panel, &dict, &modes, &label, true);
            let ds:Vec<S> = entry_bytes.chunks_exact(S::WIDTH).map(S::from_le).collect();
            let ms:Vec<S> = mode_bytes.chunks_exact(S::WIDTH).map(S::from_le).collect();
            let raw:Vec<S> = (0..p.ids.len()).map(|i| match p.values {
                Values::Raw(b)=>S::from_le(&b[i*S::WIDTH..]),
                Values::Dict(d)=>ds[match d.ranks {Ranks::One=>0,Ranks::U8(r)=>r[i] as usize,Ranks::U32(r)=>r[i] as usize}],
                Values::Modes(d)=>{let r=match d.ranks {Ranks::One=>0,Ranks::U8(r)=>r[i] as usize,Ranks::U32(r)=>r[i] as usize};if r==0 {ms[p.ids[i] as usize]} else {ds[r-1]}}
            }).collect();
            let cached = Parts { values:Values::Raw(bits(&raw)), ..p };
            let label = format!("real-cache-{}-{row}-raw-{mapping}",if p.dense {"dense"} else {"csr"});
            if !p.dense {
                measure::<S,T,0>(&cached, panel, &[], &[], &label, true);
                measure::<S,T,1>(&cached, panel, &[], &[], &label, true);
            }
            measure::<S,T,2>(&cached, panel, &[], &[], &label, true);
        }
        }
    }
}

#[test]
#[ignore = "real chunks; set CELLPACK_DISPATCH_FILES and pin the worker CPU"]
fn typed_real_chunk_benchmark() {
    assert!(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512dq"));
    all_pairs!(real_pair);
}
