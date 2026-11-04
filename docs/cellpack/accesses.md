# Access sequences

The application chooses samples. `Accesses` describes a pure function from an
access index to a global `CellId`. It does not read matrix values or schedule
workers. With one schema, one access produces one output row; with several
schemas, distinguish the sequence's access count from its position count.

## Choose the sequence representation

<DocChoices title="How is your order defined?" :items="[
  { label: 'Stored IDs', when: 'The application already has an exact order', href: '#stored-and-generated-sequences', detail: 'Use a Vec, slice or Arc. Repeats and order are retained; N IDs require 4N element bytes.' },
  { label: 'Shuffle / custom Accesses', when: 'Generate epochs without storing every access', href: '#stored-and-generated-sequences', detail: 'Address any requested interval deterministically. Shuffle can trade locality against mixing.' },
  { label: 'Replicate / Offset', when: 'Several schemas describe one logical position', href: '#compose-schemas', detail: 'Expand per-position IDs into the schema order expected by the loader.' },
  { label: 'ExternAccesses', when: 'Native application code computes sample IDs', href: '#native-sampling-callbacks', detail: 'Bind a concurrent range callback and retain its code and context explicitly.' },
]" />

## The Accesses contract

| Method | Contract |
|---|---|
| `len()` | Access count, divisible by the schema count |
| `fill(start, out)` | Fill `out.len()` ids starting at `start` |
| `as_slice()` | Return the entire materialized sequence when available |

Workers and online compiler threads request ranges concurrently, out of order,
and possibly more than once. `fill` must depend only on the requested positions.
Do not consume a mutable random-number stream inside it. Custom implementations
must honor the requested range and return `loader::Result<()>`.

## Stored and generated sequences

`Vec<CellId>`, `&[CellId]`, `Box<[CellId]>`, `Arc<[CellId]>`, and arrays implement
`Accesses`. A stored sequence of `N` ids uses `4N` bytes for its elements.
These are global IDs from data-source registration, not per-file row numbers.
Use `source.matrix_cells(matrix)` to obtain each matrix's global range.

For a short ordered pass, collect global IDs into `Vec<CellId>`. For a generated
ordered traversal, implement `Accesses` directly: map each requested index to its
cell from immutable counts and registration ranges.

`Shuffle::new(cells, epochs, seed)` computes a random-access permutation per epoch
without materializing all epochs. `Shuffle::subset` shuffles a supplied list,
which may contain repeated ids. `.blocks(block, group)` shuffles consecutive
blocks and then entries within groups of blocks to introduce locality.

## Compose schemas

`Replicate::new(inner, S)` repeats each position's cell for all `S` schemas.
`Offset::new(inner, offsets)` adds a schema-specific cell offset, useful when
registered matrices have known row pairing. Offsets must come from actual
registered ranges. The example's offset of 100 demonstrates sequence composition
and does not name a row in the guide's tiny data source.

<DocDiagram name="access-order" />

With two schemas, `[3, 3, 0, 0]` describes two positions. To request different cells
per schema, supply `[p0s0, p0s1, p1s0, p1s1, ...]` directly.
For `[3, 8, 0, 5]`, schema 0 receives cells `[3, 0]` and schema 1 receives
`[8, 5]`. Both output regions have two rows. The order of `OutputSchema`s in
`Loader::new` supplies the schema indices; their `id` values do not reorder accesses.

::: details Complete example: replication, paired offsets and repeatable shuffle ranges
<<< @/examples/accesses.rs
:::

In the application from [installation](installation.md), save
[accesses.rs](/downloads/rust/accesses.rs) as `src/main.rs`, then run:

```sh
cargo run
```

The example checks that a separately requested shuffle interval equals the same
interval of the full epoch sequence. It generates IDs only and requires no files.

## Plans and materialization

`Loader::compile` accepts a stored `&[CellId]`. Fixed plans bind to its exact
contents. When preparing a fixed plan with an `Accesses` that has no slice,
`prepare` materializes the sequence for verification. Prefer online planning for
very long generated sequences.

Keeping the same seed is insufficient if a callback's configuration or underlying
dataset order changes. Preserve the full access function and registration order
for the lifetime of a plan. See [planning](planning.md) for plan compatibility.

## Native sampling callbacks

The unsafe `ExternAccesses::new` wraps a native `AccessFn`. Its function and
context must remain valid, tolerate concurrent calls, and fill each requested
range. A nonzero status stops the run.

The C declarations are shipped in `crates/cellpack/include/access_fn_cabi.h`.
`CellPackAccessFn` fills global uint32 cell ids. `CellPackCorpusAccessFn`
instead fills uint32 dataset indices and uint64 dataset-local rows;
`ExternCorpusAccesses::new` binds the dataset row counts and translates pairs
into the same CellId space. Its per-thread row scratch is reused; callback
code and context must remain valid for the complete sequence lifetime.

`SliceAccesses::new(inner, start, len)` bounds any stored or computed sequence
without materializing its values. Python's owned AccessFn constructors are
documented in the [Python access API](/python/api/access-functions.html).
See [the ABI reference](abi.md#access-functions) for C signatures, checked ranges,
status handling and context lifetime.

API: [Accesses](/generated-loader-trait-Accesses),
[Shuffle](/generated-loader-struct-Shuffle),
[Replicate](/generated-loader-struct-Replicate),
[Offset](/generated-loader-struct-Offset).
