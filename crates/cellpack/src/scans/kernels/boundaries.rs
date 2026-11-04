//! Carry checks for the running-sum kernels: a gap of `u32::MAX` makes every
//! value after it overflow its 32-bit lane, so the truncated lanes must still
//! agree with the exact totals the kernels return.

use super::{all_supported, ascend_tail};

#[test]
fn prefix_totals_include_carries_from_maximum_gaps() {
    for n in [1, 15, 16, 17, 31, 32, 33] {
        for at in 0..n {
            let mut gaps = vec![0u32; n];
            gaps[at] = u32::MAX;
            let mut expected = gaps.clone();
            let total = ascend_tail(&mut expected, u64::MAX);
            for set in all_supported() {
                let mut got = gaps.clone();
                // SAFETY: the kernel set is supported by this CPU, and
                // the slice holds exactly the values the kernel updates.
                let actual = unsafe { (set.ascend)(&mut got) };
                assert_eq!(actual, total, "{} n={n} at={at}", set.isa);
                assert_eq!(got, expected, "{} n={n} at={at}", set.isa);
            }
        }
    }
}
