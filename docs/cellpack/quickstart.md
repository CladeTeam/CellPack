# Quickstart

Write a two-row, three-column `f32` matrix, read it into a caller-owned dense
buffer, and inspect its CSR row offsets and column indices:

<DocDiagram name="quickstart-matrix" />

## Complete example

<<< @/examples/quickstart.rs

Follow [installation](installation.md#run-a-first-example) to create a Rust
application with the crates.io dependency. Save the
[complete source](/downloads/rust/quickstart.rs) as `src/main.rs`, then run from
your application's directory:

```sh
cargo run -- output/quickstart
```

Expected output:

```text
round-trip: 2 rows, 3 columns
```

## How it works

1. Convert values to little-endian bytes for `DenseRef`. The codec accepts a
   byte view; `value_width` must agree with the file's dtype.
2. `CpkWriter::create` declares the columns, dtype, source layout, and encoding
   options. `append` adds rows in order.
3. `finish` flushes the tail chunk, writes the directory, and finally publishes
   the valid superblock. Hand the file to readers only after it succeeds.
4. `prepare_read` prepares metadata and output dimensions. `read_dense_into`
   performs the record reads and writes the typed output.
5. `ReadWorkspace` retains reusable scratch. The application owns the output.

`CpkOptions::new` targets 4096-row chunks and 4096-byte pages. This matrix is
smaller than a chunk, so `finish` writes its two remaining rows.
`LayoutChoice::Auto` lets each record choose a suitable layout;
`SourceLayout::Dense` describes the input rather than forcing dense records.

Continue with [storage](storage.md) for CSR input, gathers, and column mapping,
or [loading](loading.md) for repeated batch delivery.
