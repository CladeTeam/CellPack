use std::error::Error;

use cellpack::loader::{Accesses, Offset, Replicate, Shuffle};

fn main() -> Result<(), Box<dyn Error>> {
    let replicated = Replicate::new(vec![0u32, 1, 2], 2)?;
    let mut out = [0u32; 6];
    replicated.fill(0, &mut out)?;
    assert_eq!(out, [0, 0, 1, 1, 2, 2]);

    let paired = Offset::new(vec![0u32, 1, 2], vec![0, 100])?;
    paired.fill(0, &mut out)?;
    assert_eq!(out, [0, 100, 1, 101, 2, 102]);

    let shuffled = Shuffle::new(4, 2, 42)?.blocks(2, 1)?;
    let mut epochs = [0u32; 8];
    shuffled.fill(0, &mut epochs)?;
    for epoch in epochs.chunks_exact(4) {
        let mut sorted = epoch.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3]);
    }
    let mut part = [0u32; 3];
    shuffled.fill(2, &mut part)?;
    assert_eq!(part, epochs[2..5]);
    println!("replicated={:?}; shuffled={:?}", replicated.len(), epochs);
    Ok(())
}
