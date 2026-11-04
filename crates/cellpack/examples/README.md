# Rust experimental tools

These programs exercise the current `cellpack::store`, `scans`, `loader` and
`opt` interfaces. Runnable API tutorials live in
[`docs/cellpack/examples`](../../../docs/cellpack/examples); the Rust guide starts
at [the quickstart](../../../docs/cellpack/quickstart.md).

| Cargo example | Source | Purpose |
|---|---|---|
| `bench` | [`bench/main.rs`](bench/main.rs) | Access generation, cache plans, loading, profiling and raw I/O |
| `codec` | [`bench/codec.rs`](bench/codec.rs) | Compression ratio and timings for individual decoding paths |
| `encode` | [`bench/encode.rs`](bench/encode.rs) | Encoding and page-packing throughput with reusable buffers |
| `belady` | [`belady.rs`](belady.rs) | Page-only plans against the independent serial optimum |

`bench/costs/store.rs` and `bench/process/sampling.rs` are support modules for
`bench`, not separate executables. Its sampling callback is a synthetic workload;
the Python model templates are under `pycellpack.loader.process_fn`.

Run builds and experiments on an existing worker. From the repository root:

```sh
cargo build --locked -p cellpack --release --examples
cargo run --locked -p cellpack --release --example bench -- --help
cargo run --locked -p cellpack --release --example codec -- --help
cargo run --locked -p cellpack --release --example encode -- --help
cargo run --locked -p cellpack --release --example belady -- --help
```

No optional crate features are required. `--features serde` additionally enables
Serde on public cost-model types. For useful timing results, use release builds,
record the input/configuration and CPU affinity, and compare equal workloads.
Small smoke runs only establish that the commands execute.

## Loading and planning: `bench`

The corpus can be one `.cpk` file, a flat directory of `.cpk` files, or a directory
of datasets each containing `raw.cpk`. Files are sorted by path. `ids=FILE` is a
little-endian `uint32` column-to-standard-id map concatenated in that same file
order; `vocab=N` is the number of standard ids. `0xffffffff` drops a column.
There is no implicit gene alignment. `files=N` selects a subset of those files,
but the map still describes the full corpus.

An access file is a flat little-endian `uint32` sequence of global cell ids.
`shuffle:EPOCHS[:SEED[:BLOCK:GROUP]]` supplies a computed sequence to `plan`,
`online` or `run`. The tool expands accesses to the selected schemas, including
paired embedding inputs when `embed=DIR` is supplied.

Example workflow, after supplying `corpus`, `ids.u32` and a vocabulary size:

```sh
target/release/examples/bench gen corpus accesses.u32 random 4 ids=ids.u32 vocab=30000
target/release/examples/bench costs corpus ids=ids.u32 vocab=30000 schemas=dense cost=defaults
target/release/examples/bench plan corpus accesses.u32 cache.plan ids=ids.u32 vocab=30000 \
  schemas=dense cache=1048576 cost=defaults threads=4
target/release/examples/bench run corpus accesses.u32 ids=ids.u32 vocab=30000 \
  schemas=dense cache=plan:cache.plan workers=4 slot=128 slots=4 backend=pread
target/release/examples/bench run corpus accesses.u32 ids=ids.u32 vocab=30000 \
  schemas=dense cache=online:1048576 cost=defaults threads=4 workers=4 slot=128 slots=4
```

`slot=CELLS` is the batch size (`LoaderConfig::slot_cells`); `slots=S` is the ring
capacity. Python exposes the batch size as `slot_rows`. `schemas=` defaults to
the sampling callback; use `dense` for a built-in dense output, or `count` for
the counting callback. Multiple comma-separated schemas share the output ring.
`backend=pread|uring` selects the actual loader I/O backend. The tool opens corpus
files with direct I/O, so the filesystem must support it.

`cost=auto` resolves measured records through the benchmark's cost lookup, with
defaults for missing parts. `cost=defaults` uses only defaults; `cost=none`
disables the cost model; `cost=FILE.json` reads a Python `CostModel.save()` file.
The program reports the source of its model. Defaults are not a calibration.

Other commands are `online` (compiler-only execution), `curve` (page-only hit
curves against Belady) and `ioraw` (raw asynchronous page reads). See `--help`
and the module documentation in `bench/main.rs` for their options.

## Encoding: `encode`

```sh
target/release/examples/encode csr_root 10 f32 4096
NORM=1 REPEAT=3 target/release/examples/encode csr_root 10 f32 4096
```

Each dataset directory under `csr_root`, sorted by path, contains raw
little-endian arrays: `indptr.i64`, `indices.i32` and `data.f32`. CSR rows must
have sorted, distinct columns. The column count is inferred from the indices;
`GENES=N` explicitly selects a panel width. The output value type is one of
`u8`, `u16`, `u32`, `u64`, `f32`, `f64`; integer types cast the input counts
(`u8` caps them at 255). `NORM=1` applies `log1p(1e4 * x / row_sum)` before casting.

The program reports sizing, encoding and packing time separately and a digest
of the packed bytes. Buffers are reused across chunks. `MODES`, `LAYOUT`,
`VALUES`, `RENUMBER`, `LANES` and `LANE_MIN` select encoding options; `ROWS`,
`DENSE`, `SKIP`, `REPEAT`, `GENES` and `DENSIFY` control the input or repetition.
Their definitions are in the module documentation in `bench/encode.rs`.

## Codec paths: `codec`

```sh
target/release/examples/codec 4096 < input.csr
```

The input stream consists of:

1. Five little-endian `uint64` words: rows, columns, index width in bytes,
   value code, and nnz.
2. `rows + 1` little-endian `uint64` row pointers.
3. `nnz` unsigned indices, each 2 or 4 bytes, with sorted distinct columns per row.
4. `nnz` values. Value codes 0/3 select 2-byte values, 1/4/6 select 4-byte values,
   and 2/5/7 select 8-byte values. The encoder preserves their bits; these stream
   tags are separate from the ProcessFn ABI's dtype codes.

Positional arguments are `chunk_rows`, `max_lanes`, `lane_min`, `renumber`
(`0|1`), `mu_indices`, `mu_values`, and `mode_share` (`0` disables column modes).
Trailing arguments can be omitted. The program compares source/coded record
decoding, chunk CSR, dense rows, decoded parts, mapping, and contiguous/paged
record input. These are isolated codec timings, not loader throughput.

## Serial reference: `belady`

```sh
target/release/examples/belady /absolute/path/to/scratch pages=4096 repetitions=16
```

The program creates one distinct dense float32 row per page, varies locality
while preserving the page histogram, compiles page-only miss-minimizing plans,
and checks replay against each plan summary and the independent Belady bound.
`repetitions` must be a power of two. It prints CSV and removes its temporary
`.cpk` after successful completion. This serial reference does not establish
global optimality for the concurrent loader's page/row/time decisions.
