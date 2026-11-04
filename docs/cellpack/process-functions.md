# Native process functions

A process function turns a mapped input row into a fixed-size output record.
The loader maps columns and converts values to the schema dtype before the call.
The callback runs on worker threads and writes its own output and private scratch.

<DocDiagram name="process-path" />

## Choose the input your transform needs

<DocChoices title="What must the callback inspect?" :items="[
  { label: 'RowFormat::Dense', when: 'Inspect the full panel or distinguish missing columns', href: '#row-formats', detail: 'One callback sees output-order values. Use a sentinel default to recognize source-absent dimensions.' },
  { label: 'RowFormat::Csr', when: 'Operate on stored entries', href: '#row-formats', detail: 'One callback sees mapped columns and values. Columns need not be sorted; omitted dimensions carry no missingness flag.' },
  { label: 'RowFormat::Auto', when: 'Both implementations have equivalent semantics', href: '#row-formats', detail: 'Provide CSR and dense callbacks; the loader chooses the representation for each row.' },
]" />

First define the model's output record, then its input panel and dtype. For a
sum, the record can be one `f32` even when the input panel has thousands of
columns. Process results are delivered into output slots; input row caching does
not cache the callback's final output.

## A complete dense callback

This example requests four `f32` values and writes one `f32` sum per cell.
It forces dense delivery, so only a dense callback is needed.

::: details Complete program: dense callback, schema binding and output checks
<<< @/examples/process.rs
:::

In the application from [installation](installation.md), save
[process.rs](/downloads/rust/process.rs) as `src/main.rs` and
[common.rs](/downloads/rust/common.rs) as `src/common.rs`, then run:

```sh
cargo run -- output/process
```

Expected sums are `[3.0, 7.0, 11.0, 15.0]`. The callback writes bytes without
requiring alignment of its custom output record.

For this example, the schema input is four `f32` values, while `cell_bytes=4`
reserves one output `f32`. Consumers interpret those four bytes with
`f32::from_ne_bytes`. The byte count and interpretation belong to the
application's record contract.

## Row formats

| `RowFormat` | Required callbacks | Input |
|---|---|---|
| `Auto` | CSR and dense | Route depends on stored layout and internal thresholds |
| `Csr` | CSR | All rows as `CsrRow`, including compacted dense nonzeros |
| `Dense` | Dense | All rows as `DenseRow` |

`ProcessFn::new` selects `Auto`; `with_format` selects explicitly.
In `Auto`, both callbacks must produce equivalent results.
A file's declared source layout does not fix its callback route.

A `CsrRow` lists stored elements in coded traversal order, with columns already
mapped to output positions. Columns are not necessarily sorted. It cannot
distinguish omitted present zeros from missing dimensions. A `DenseRow` holds
zero for present omitted values and the schema default for missing dimensions.
Use dense delivery and a suitable default when that distinction matters.

For example, a source with columns A and B and row `[0, 2]`, mapped to output
`[B, C, A]` with `default=NaN`, yields dense `[2, NaN, 0]`. A sparse row can
report B's stored value but cannot identify C's absence just from omitted entries.

## ABI and ownership contract

`ProcessContext` contains `abi_version`, row `struct_size`, logical `position`,
global `cell`, schema index, dtype code, and `user_data`.
The current `PROCESS_ABI_VERSION` is 1. `schema` is the schema's vector index,
not its application `id`, and `struct_size` is the row struct's byte size.
Native integrations must use declarations matching the loader version.

Construction is unsafe because the application guarantees that callbacks:

- remain valid through their owner and tolerate concurrent calls;
- read input only during the call and retain no borrowed pointer;
- write only their cell's `cell_bytes` and declared scratch;
- initialize the output bytes that consumers will read;
- never unwind across the C ABI;
- return zero on success or a nonzero failure status.

An `Arc<dyn Send + Sync>` retains code and state. An unrelated owner does not
extend the lifetime of the raw `user_data` pointee.
A nonzero status stops the run with `loader::Error::Process(status)`.

The loader does not clear output or scratch on the callback's behalf. Initialize
every output byte the consumer reads, and treat scratch contents as unspecified
on entry. Neither pointer promises alignment for a typed record: use byte copies
or unaligned operations. See [ABI layouts and native examples](abi.md#process-functions)
when integrating C, C++ or a dynamically loaded library.

`OutputSchema::process` sets output size independently of input panel dimensions.
Its `identity` must change when callback behavior or captured settings change;
fixed plans verify that identity.

## Partial CSR delivery

Use this advanced path only when the transform needs a small subset of stored
elements. It is unnecessary for the full-panel sum example above.

Unsafe `with_partial_rows` permits null `columns` and `values`.
A callback then uses `resolve` for required element positions within `nnz`.
The resolver and derived pointers are valid only during the call. A position
at or beyond `nnz` makes `resolve` return a nonzero status.

Partial rows are offered only when every element maps to an output column.
A partial-aware callback must also handle ordinary full CSR rows.

API: [ProcessFn](/generated-loader-struct-ProcessFn),
[CsrRow](/generated-loader-struct-CsrRow),
[DenseRow](/generated-loader-struct-DenseRow),
[ProcessContext](/generated-loader-struct-ProcessContext).
