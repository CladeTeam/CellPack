# Installation

## Requirements

- Rust 1.96 for the documentation build and guide examples; edition 2024.
- Linux: storage uses Unix file interfaces, and the loader uses Linux memory,
  futex, and I/O interfaces.
- The default loader backend requires `io_uring`. Select `IoBackend::Pread`
  explicitly when the environment disables it.
- On x86-64, the runtime selects supported SIMD kernels. Scalar kernels are
  also available; see [runtime configuration](performance.md#runtime-kernels).

The Rust core has no Python dependency. `crates/pycellpack` is the separate PyO3
binding crate and is not needed for the workflows in this manual.

The core workspace declares `rust-version = "1.89"`. Treat that as its manifest
declaration rather than a verified minimum: the current intrinsic calls fail
the workspace's unsafe-operation lint under Rust 1.91. This guide uses Rust 1.96
for a reproducible build and executable examples.

## Depend on a checkout

The workspace currently sets `publish = false`. Add a path dependency to your
application's `Cargo.toml`, using the actual path to your checkout:

```toml
[dependencies]
cellpack = { path = "/absolute/path/to/CellPack/crates/cellpack" }
```

Point at the crate directory, rather than the workspace root. A relative path
works when your application and the checkout have a known relative layout.

## Features

| Feature | Enabled by default | Purpose |
|---|---|---|
| `serde` | No | Serde derives for cost-model types |

```toml
[dependencies]
cellpack = { path = "/absolute/path/to/CellPack/crates/cellpack", features = ["serde"] }
```

The binary plan format and `CostModel::to_bytes` / `from_bytes` do not require
`serde`. Profiling requires no build feature: timed and untimed loader instances
are included in the same build.

## Build and check

Run these commands from the repository root:

```sh
cargo check --locked -p cellpack
cargo test --locked -p cellpack --all-features
cargo doc --locked -p cellpack --all-features --no-deps
```

`-p cellpack` selects the core crate. The API entry point is
`target/doc/cellpack/index.html`.

On systems that restrict compilation on development machines, run compilation,
examples, and full documentation builds on an authorized compute node.
See [documentation development](contributing.md) for the documentation toolchain.

## Run a first example

The guide examples generate their own tiny matrices:

```sh
cargo run --locked --manifest-path docs/cellpack/examples/Cargo.toml \
  --bin quickstart -- tmp/cellpack-guide/quickstart
```

The example writes `matrix.cpk` in the supplied directory and checks its decoded
contents. Use a dedicated example directory: rerunning an example overwrites its
named outputs. Continue with the [quickstart](quickstart.md) for the full source.
