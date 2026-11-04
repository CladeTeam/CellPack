use std::ffi::c_void;
use std::sync::Arc;

use cellpack::loader::{Accesses, ExternCorpusAccesses, SliceAccesses};

unsafe extern "C" fn pairs(_: *mut c_void, start: u64, count: u64, datasets: *mut u32, rows: *mut u64) -> i32 {
    for i in 0..count as usize {
        // SAFETY: the caller provides count writable entries in both arrays.
        unsafe {
            *datasets.add(i) = ((start + i as u64) % 2) as u32;
            *rows.add(i) = (start + i as u64) % 3;
        }
    }
    0
}

#[test]
fn corpus_callback_maps_concurrent_repeated_intervals() {
    // SAFETY: pairs is stateless, concurrent, and writes exactly count entries.
    let accesses = Arc::new(unsafe { ExternCorpusAccesses::new(pairs, std::ptr::null_mut(), 1000, &[3, 3]).unwrap() });
    std::thread::scope(|scope| {
        for start in [0, 17, 4, 17, 600, 900] {
            let accesses = accesses.clone();
            scope.spawn(move || {
                let mut out = [0; 50];
                accesses.fill(start, &mut out).unwrap();
                for (i, &cell) in out.iter().enumerate() {
                    let at = start + i as u64;
                    assert_eq!(cell, (at % 3 + (at % 2) * 3) as u32);
                }
            });
        }
    });
    assert!(accesses.fill(999, &mut [0; 2]).is_err());
    // SAFETY: the callback contract holds even when produced rows fail validation.
    let invalid = unsafe { ExternCorpusAccesses::new(pairs, std::ptr::null_mut(), 10, &[1, 1]).unwrap() };
    assert!(invalid.fill(0, &mut [0; 3]).is_err());
}

#[test]
fn access_slices_bound_their_ranges() {
    let sequence = Arc::new((0..21).map(|i| i % 7).collect::<Vec<u32>>());
    let slice = SliceAccesses::new(sequence, 5, 12).unwrap();
    let mut out = [0; 12];
    slice.fill(0, &mut out).unwrap();
    assert_eq!(out, [5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2]);
    assert!(slice.as_slice().is_none());
    assert!(slice.fill(12, &mut [0]).is_err());
    SliceAccesses::new(Arc::new(Vec::<u32>::new()), 0, 0).unwrap().fill(0, &mut []).unwrap();
}
