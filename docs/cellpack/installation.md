# Installation

## Requirements

- Rust 1.96 is used for the guide examples; edition 2024. The published crate
  declares Rust 1.89 as its minimum version.
- Linux: storage uses Unix file interfaces, and the loader uses Linux memory,
  futex, and I/O interfaces.
- The default loader backend requires `io_uring`. Select `IoBackend::Pread`
  explicitly when the environment disables it.
- On x86-64, the runtime selects supported SIMD kernels. Scalar kernels are
  also available; see [runtime configuration](performance.md#runtime-kernels).

The Rust core has no Python dependency. `crates/pycellpack` is the separate PyO3
binding crate and is not needed for the workflows in this manual.

## Add CellPack from crates.io

[CellPack is published on crates.io](https://crates.io/crates/cellpack).
Add it to an existing Rust application:

```sh
cargo add cellpack@0.1.0
```

Or add the dependency directly to your application's `Cargo.toml`:

```toml
[dependencies]
cellpack = "0.1.0"
```

Cargo downloads the crate and its dependencies from the registry. No CellPack
repository checkout is required.

## Features

| Feature | Enabled by default | Purpose |
|---|---|---|
| `serde` | No | Serde derives for cost-model types |

```toml
[dependencies]
cellpack = { version = "0.1.0", features = ["serde"] }
```

The binary plan format and `CostModel::to_bytes` / `from_bytes` do not require
`serde`. Profiling requires no build feature: timed and untimed loader instances
are included in the same build.

## Check your application

Run these commands from your application's directory:

```sh
cargo check
cargo doc -p cellpack --no-deps --open
```

The API entry point is `target/doc/cellpack/index.html`. The published Rust API
is also available on [docs.rs](https://docs.rs/cellpack/0.1.0/cellpack/).

On systems that restrict compilation on development machines, run compilation,
examples, and full documentation builds on an authorized compute node.
See [documentation development](contributing.md) for the documentation toolchain.

## Run a first example

Create an application and add the published crate:

```sh
cargo new cellpack-demo
cd cellpack-demo
cargo add cellpack@0.1.0
```

Save the [quickstart source](/downloads/rust/quickstart.rs) as `src/main.rs`, then run:

```sh
cargo run -- output/quickstart
```

The example generates its own tiny matrix, writes `matrix.cpk` in the supplied
directory, and checks its decoded contents. Use a dedicated output directory:
rerunning an example overwrites its named outputs. Continue with the
[quickstart](quickstart.md) for the full source. Other guide programs that declare
`mod common;` also need [common.rs](/downloads/rust/common.rs) saved as `src/common.rs`.
