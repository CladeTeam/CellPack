//! Source-level invariants of the per-cell paths: environment variables are
//! read and the CPU probed only where the runtime binds them, worker timing
//! goes through profiler sites, and the process binding is read only where
//! contexts are constructed. Each rule lists the files (and how
//! many times) a pattern may appear; anything else fails.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Occurrences of `pattern` per source file, relative to `src/`.
fn occurrences(pattern: &str) -> BTreeMap<String, usize> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&root, &mut files);
    files
        .into_iter()
        .filter_map(|path| {
            let text = std::fs::read_to_string(&path).unwrap();
            let n = text.matches(pattern).count();
            (n > 0).then(|| (path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/"), n))
        })
        .collect()
}

fn check(pattern: &str, allowed: &[(&str, usize)], why: &str) {
    let found = occurrences(pattern);
    let allowed: BTreeMap<String, usize> = allowed.iter().map(|&(f, n)| (f.to_string(), n)).collect();
    assert_eq!(found, allowed, "`{pattern}` outside its allowed places: {why}");
}

#[test]
fn environment_is_read_only_by_the_runtime() {
    check(
        "std::env::var",
        &[("runtime.rs", 1), ("scans/kernels/typed_tests.rs", 3)],
        "environment variables are parsed once in `runtime` (tests may read their own)",
    );
}

#[test]
fn cpu_is_probed_only_when_binding_kernels() {
    check(
        "is_x86_feature_detected!",
        &[("scans/kernels/x86.rs", 8), ("scans/kernels/typed_tests.rs", 12)],
        "the CPU is probed when the runtime selects a kernel set",
    );
}

#[test]
fn worker_timing_goes_through_profiler_sites() {
    check(
        "Instant::now",
        &[
            // The profiler's clock (its base, and TSC calibration).
            ("prof/mod.rs", 2),
            // Once per run: setup, the snapshot clock, the workers' start.
            ("loader/exec/mod.rs", 3),
            // Ring acquisition timeouts.
            ("loader/ring.rs", 2),
            // Compiler wall time, and its waits for plan ring space.
            ("opt/engine.rs", 2),
            ("scans/kernels/typed_tests.rs", 1),
        ],
        "per-cell and per-request timing goes through `prof` sites",
    );
}

#[test]
fn process_binding_is_read_only_by_constructors() {
    check(
        "runtime::get()",
        &[
            // Decode and encode tables bind the kernel set per chunk.
            ("scans/tables.rs", 2),
            // `Mapper::new` and the one-off free mapping functions.
            ("scans/map/mod.rs", 4),
            // One-off `Parts::expand_values`, chunk encoding and prepared
            // encoders (before their tables exist), `kernel_set`, and
            // loader `RowOps` at build time.
            ("scans/decode.rs", 1),
            ("scans/encode/mod.rs", 2),
            ("scans/kernels/mod.rs", 1),
            ("loader/convert.rs", 1),
        ],
        "hot paths use the kernels their context bound",
    );
}
