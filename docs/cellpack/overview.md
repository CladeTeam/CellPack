# CellPack

CellPack is a Rust library for matrix storage and bounded data loading. It
compresses each row into an independently decodable record, packs records onto
pages, and delivers requested rows through a fixed-capacity output ring.
It is designed for single-cell training pipelines that need random row access,
explicit column mapping, multiple outputs, and planned reuse of input data.

This manual covers the Rust crate `cellpack`. The Python distribution and import
package are named `pycellpack` and have separate documentation. The examples here
use the Rust core directly and require neither Python bindings nor a GPU.

## Start here

<DocDiagram name="rust-start" />

See [installation](installation.md) for setup, [access sequences](accesses.md) and
[loading](loading.md) for delivery, [performance](performance.md) and
[profiling](profiling.md) for measurement, and [process functions](process-functions.md),
[file format](format.md) and [architecture](architecture.md) for integration.

## Read a guide or look up an API?

The [quickstart](quickstart.md) creates its own small matrix and checks the decoded
result. Follow the guides when assembling a workflow: each explains the objects
you need and the order in which to use them. Switch to the [API reference](api.md)
when you need an exact signature, buffer size, return value or ownership rule.
The reference links generated declarations to their source.

For ordered delivery, learn the [identity spaces](concepts.md) before creating
sources and accesses. For custom containers or native callbacks, read the
[data contracts](contracts.md) before composing lower-level interfaces.

## What the library provides

| Task | Entry point |
|---|---|
| Stream CSR or dense rows into a `.cpk` file | `store::CpkWriter` |
| Read ranges, gathers, and repeated rows into owned buffers | `store::CpkFile`, `ReadSelection` |
| Encode and decode chunks or records in caller buffers | `scans` |
| Combine matrices with an explicit shared column vocabulary | `loader::DataSourceBuilder` |
| Produce multiple fixed-size outputs per sequence position | `loader::Loader`, `OutputSchema` |
| Plan reuse of compressed pages and decoded input rows | `opt` |
| Inspect event counters and selected timing sites | `loader::RunStats`, `prof` |

A `.cpk` stream contains a matrix, without row labels, column labels, or
annotations. The application owns names, vocabulary construction, sampling,
training, and device transfers. Compression preserves value bit patterns;
requesting a different output dtype introduces that conversion's precision limits.
