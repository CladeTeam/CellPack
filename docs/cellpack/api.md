# Rust API

Find the interface for your task, try it on a small matrix, then follow its exact
declaration and ownership rules. This reference covers the Rust crate `cellpack`.

Type names inside declarations are clickable: open a definition beside the page,
inspect its source, or continue to the full API entry. **Run example** executes a
fixed, prebuilt program on the configured server. Code is read-only. Enable **LSP**
for type information and definition lookup within the current example file.
[Runnable examples](api-examples.md) explain the shared fixture and local setup.

## Start with a task

| I want to… | Start here | What to watch |
|---|---|---|
| Write matrix rows to a file | [CpkWriter](/generated-store-struct-CpkWriter) | Declare the input dtype/layout; call `finish` before opening the file |
| Read or reorder selected rows | [ReadSelection](/generated-store-struct-ReadSelection) | File-row IDs, output element counts, and exclusive workspace |
| Map columns into another order | [CpkFile::prepare_selection](/generated-store-struct-CpkFile#generated-store-struct-cpkfile-method-prepare-selection) | The map is source → output; lists preserve row repetitions |
| Produce batches for training | [Loader](/generated-loader-struct-Loader) | Source → schemas → accesses → ring → run |
| Consume batches safely | [Run](/generated-loader-struct-Run) | Overlap production and consumption; release leases when readers finish |
| Generate an access order | [Accesses](/generated-loader-trait-Accesses) | Global cell IDs; calls may be concurrent, repeated, or out of order |
| Reuse cached pages and rows | [Plans and costs](reference-opt.md) | Cache capacity is not total RSS; modeled time is not measured time |
| Encode records in my own container | [Codecs](reference-scans.md) | Caller buffers, matching metadata, and trusted-record requirements |

## The main workflows

<DocDiagram name="rust-storage" />

<DocDiagram name="rust-loading" />
Use [data contracts](contracts.md) for units, dtype conversions, ID spaces, and lifetime rules.

## Choose a module

| Module | Responsibility | Input | Output |
|---|---|---|---|
| [Matrix storage: store](reference-store.md) | Streaming files and ordered selections | Matrix row batches, files, selectors | `.cpk` streams and typed matrix buffers |
| [Codecs: scans](reference-scans.md) | Encoding, decoding, packing, casting and mapping | Borrowed bytes, metadata, tables, workspace | Encoded bytes, decoded arrays, borrowed row parts |
| [Metadata: format](reference-format.md) | Parse and inspect byte layouts | Stream/chunk metadata bytes | Borrowed views and owned directory records |
| [Data delivery: loader](reference-loader.md) | Sources, schemas, sequences, workers, rings and process ABI | Immutable input files and deterministic access requests | Ordered batches with per-schema byte regions |
| [Planning: opt](reference-opt.md) | Frame residency and cost-guided compilation | Graph, sequence, cache budget, cost model | Plans, replay summaries and online statistics |
| [Timing: prof](reference-prof.md) | Selected timing sites and report aggregation | Site masks and per-thread measurements | Site, detail and histogram reports |
| [Process settings: runtime](reference-runtime.md) | Kernel binding and core topology | Explicit options or environment | Process-lifetime runtime and physical-core count |
| [Errors and root exports](reference-errors.md) | Error domains and convenience imports | Operation results and failure causes | Typed errors and recovery scope |

## Browse interfaces

Search by name or responsibility, then filter by module and kind. **Quick view**
keeps your place while showing the declaration and a source excerpt.

<ApiExplorer />

The module indexes enumerate every documented public item, including
supporting enums, constants, traits and low-level helpers. Detail pages are
regenerated from the crate source at build time. Public signatures,
field types and method lists are therefore linked to the build, while semantic
contracts are maintained in the module references.

## Read contracts before composing layers

- **Representation:** [matrix inputs and dtype rules](contracts.md#matrix-inputs).
- **Sizes:** [shape, capacity, byte and workspace-word units](contracts.md#shapes-and-capacity).
- **Identity:** [column spaces, file rows, cells, positions and accesses](contracts.md#column-and-row-identities).
- **Ownership:** [borrowing and concurrent use](contracts.md#borrowing-and-concurrency).
- **Failures:** [validation boundary](contracts.md#failure-and-validation-boundary) and [error variants](reference-errors.md).


Declarations, member links, implemented traits, and source excerpts come from the
same build. Parameter explanations and worked examples add the usage context.
