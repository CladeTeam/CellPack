# Runnable API examples

Each example uses a tiny matrix generated inside its own temporary directory.
Read the highlighted source and expected output, then choose **Run example**.
Only fixed examples can run; the server accepts an example ID, never source code
or arguments. Both source files are read-only and available to download.

Enable **LSP** for rust-analyzer hover information and **F12** / **Ctrl/⌘ click**
definition lookup. Navigation stays in the selected file. Definitions in other
files or libraries do not open another page or file. The server enforces the same
restriction and rejects document changes.

## Write and reopen

Inspect the decoded dense matrix and its CSR structure.
The writer must be finished before the file is read.

<ApiExample example="write" />

## Gather and map

The fixture is a four-row, four-column matrix:

<DocDiagram name="example-matrix" />

The example gathers rows `[3, 0, 3]`. The map `[1, UNMAPPED, 0, UNMAPPED]` keeps
source column 2 at output position 0 and source column 0 at output position 1.
`spec.nnz` reserves capacity; the CSR read returns the actual count after mapping.

<ApiExample example="selection" />

## Ordered batch delivery

Five positions use the cell order `[3, 0, 2, 3, 1]`. Each position has two schemas,
so it accounts for two accesses. With two positions per slot, the final batch has
one row. Standard column 4 is absent from this source and receives NaN; existing
columns with sparse zero values still return zero.

<ApiExample example="loading" />

## Run locally

Download **main.rs** and **common.rs** from the same example and put both in the
`src/` directory of a Rust package. Add the published crate from crates.io:

```sh
cargo add cellpack@0.1.0
```

Use Rust 1.96.0 and edition 2024. No CellPack checkout is needed.
The helper takes an output directory as its first argument:

```sh
cargo run -- /path/to/example-output
```

Examples keep inputs small and use buffered `Pread` reads. They demonstrate API
behavior and are not throughput measurements. If the run service is offline,
the highlighted code, expected output, downloads, and normal API navigation remain available.

For server configuration, see the [playground operator guide](playground/README.md).
