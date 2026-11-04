# ABI

CellPack calls application code at two points of a run, through plain C
function pointers and an opaque context pointer:

| Interface | Purpose | Rust binding | C declarations | Called by |
|---|---|---|---|---|
| **Process function** | Turns one mapped, converted row into a fixed-size output record | `ProcessFn` with `CsrFn` / `DenseFn` | `include/process_fn_cabi.h` | Loader workers, once per loaded cell of a process schema |
| **Access function** | Produces the cell id of each access in the sequence | `ExternAccesses` with `AccessFn` | `include/access_fn_cabi.h` | Workers and online planner threads, per range |
| **Corpus access function** | Produces (dataset, local row) pairs instead of global ids | `ExternCorpusAccesses` with `CorpusAccessFn` | `include/access_fn_cabi.h` | As above; the adapter translates pairs into cell ids |

Both interfaces are `unsafe extern "C"`. The loader validates the values that
come back where it can, but it cannot check what a callback reads, writes or
retains. Those obligations are listed per interface below; breaking one is
undefined behavior, not an error. See [data contracts](data-contracts.md) for
the matching rules on data the application passes in.

Headers live in `crates/cellpack/include/`. The process header keeps its
`PyCellPack` prefix because Python capsules share the same layout; the Rust
structs below are the same types.

<DocChoices title="Which side of the pipeline are you extending?" :items="[
  { label: 'Process callback', when: 'Transform one loaded row', href: '#process-functions', detail: 'Start with a fixed output record and a forced row format. Read layouts only when implementing the foreign binding.' },
  { label: 'Access callback', when: 'Generate global cell IDs', href: '#access-functions', detail: 'Fill arbitrary ranges deterministically; the callback does not read the matrix.' },
  { label: 'Corpus access callback', when: 'Your sampler uses dataset and local-row pairs', href: '#corpus-access-functions', detail: 'Bind row counts in data-source registration order and let the adapter translate IDs.' },
]" />

For the first Rust integration, follow the [process-function guide](process-functions.md)
or [access-sequence guide](accesses.md). This page supplies the layouts, native
signatures and foreign-code obligations those workflows rely on.

## Process functions

A process function sees each loaded cell after the loader has decoded its
record, mapped its columns to the schema's output dims and converted values to
the schema dtype. It writes exactly the schema's `cell_bytes` at `out`:

<DocDiagram name="process-path" />

### Signatures

```rust
pub type CsrFn = unsafe extern "C" fn(*const CsrRow) -> i32;
pub type DenseFn = unsafe extern "C" fn(*const DenseRow) -> i32;
pub type ResolveFn = unsafe extern "C" fn(
    row: *const c_void, positions: *const u32, count: u64,
    columns: *mut u32, values: *mut c_void,
) -> i32;
```

Return `0` on success. Any other value cancels the run with
`loader::Error::Process(status)`; the status is passed through unchanged, so
callbacks can encode their own failure reasons.

### Struct layouts

All three structs are `#[repr(C)]`. Offsets are for 64-bit targets.
Use the shipped header's declarations rather than recreating these layouts in
foreign code. `context.schema` is a vector index; it is not `OutputSchema::id`.

::: details Field types, 64-bit offsets and dtype codes
| `ProcessContext` field | Type | Offset | Meaning |
|---|---|---|---|
| `abi_version` | `u32` | 0 | `PROCESS_ABI_VERSION` (currently `1`) |
| `struct_size` | `u32` | 4 | Size of the row struct passed (112 for `CsrRow`, 80 for `DenseRow`) |
| `position` | `u64` | 8 | Position in the sequence; every schema of one position sees the same value |
| `cell` | `u64` | 16 | Global `CellId` being loaded |
| `schema` | `u32` | 24 | Schema index in the loader |
| `dtype` | `u32` | 28 | Dtype code of `values` (table below) |
| `user_data` | `*const c_void` | 32 | The pointer given to `ProcessFn::new` |

| `CsrRow` field | Offset | `DenseRow` field | Offset |
|---|---|---|---|
| `context` | 0 | `context` | 0 |
| `out_dim: u64` | 40 | `out_dim: u64` | 40 |
| `nnz: u64` | 48 | `values: *const c_void` | 48 |
| `columns: *const u32` | 56 | `out: *mut u8` | 56 |
| `values: *const c_void` | 64 | `scratch: *mut u8` | 64 |
| `row: *const c_void` | 72 | `scratch_bytes: u64` | 72 |
| `resolve: Option<ResolveFn>` | 80 | | |
| `out: *mut u8` | 88 | | |
| `scratch: *mut u8` | 96 | | |
| `scratch_bytes: u64` | 104 | | |

| Code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| dtype | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | f32 | f64 |
:::

Check `abi_version`, the expected row `struct_size`, and `dtype` on entry. A
callback compiled for one dtype that receives another reads the wrong element
width. Row sizes are 112 bytes for `CsrRow` and 80 for `DenseRow` on the 64-bit
layout above.

### What each row contains

| Row | `values` | Columns | Omitted output dims |
|---|---|---|---|
| `CsrRow` | `nnz` values of the schema dtype | `columns[i] < out_dim`, in the chunk's coded order, **not sorted** | Zero *or* missing: a CSR row cannot tell which |
| `DenseRow` | `out_dim` values of the schema dtype | Implicit: index `i` is output dim `i` | Present but unstored → `0`; missing from the matrix → the schema default |

`RowFormat` selects the callbacks the loader may call:

| `RowFormat` | Callbacks required | Delivered rows |
|---|---|---|
| `Auto` (`ProcessFn::new`) | CSR **and** dense | Chosen per record from layout, dims, conversion and cache state |
| `Csr` | CSR | Every record as a CSR row (dense records list their non-zeros) |
| `Dense` | Dense | Every record as a dense row |

Under `Auto` the two callbacks must produce identical output for the same
cell. Which one runs is a performance decision that can change between runs,
cache plans and library versions.

### The callback contract

`ProcessFn::new` and `ProcessFn::with_format` are `unsafe` because the loader
trusts all of the following:

| Obligation | If violated |
|---|---|
| Code and `user_data` stay valid for as long as the `ProcessFn` (keep them alive through `owner`) | Use-after-free on a worker thread |
| Callbacks tolerate concurrent calls from several workers | Data races on shared state |
| Write only `out[0..cell_bytes]` and `scratch[0..scratch_bytes]` | Corrupts neighbouring cells, slot headers or worker memory |
| Initialize every output byte consumers read | Consumers read stale bytes from an earlier batch |
| Read `values` / `columns` only within `nnz` or `out_dim`, and only during the call | Out-of-bounds or dangling reads |
| Never unwind (panic, C++ exception) across the ABI | Undefined behavior; typically an abort |
| Change the schema `identity` when behavior or captured settings change | Plans bound to the old identity are accepted as still valid |

`out` has **no alignment guarantee**. Write records with `write_unaligned`,
`copy_nonoverlapping` or `memcpy`. `scratch` is private to the call, at least
`scratch_bytes` long, and reused between calls on the same worker, so its
contents are unspecified on entry.

### Example: a CSR + dense pair in Rust

In the application from [installation](installation.md), save the
[complete program](/downloads/rust/abi.rs) as `src/main.rs` and the
[shared fixture](/downloads/rust/common.rs) as `src/common.rs`, then run:

```sh
cargo run -- output/abi
```

Each cell produces a 12-byte record: the scaled row total, the column of its
largest value, and its non-zero count. Settings reach both callbacks through
`user_data`:

::: details Output record and equivalent CSR/dense callbacks
<<< @/examples/abi.rs#record

The CSR callback cannot assume sorted columns, so tie-breaking uses the column
id rather than arrival order. A panic is turned into a status instead of
unwinding into the loader:

<<< @/examples/abi.rs#csr

The dense callback visits every output dim, skipping zeros so that its count
matches the CSR path:

<<< @/examples/abi.rs#dense
:::

Binding keeps `Settings` alive through the owner `Arc`, and the pointer handed
out as `user_data` points into that same allocation:

<<< @/examples/abi.rs#bind

::: warning Retain the storage addressed by `user_data`
`owner` only extends the lifetime of the allocation or dependencies it owns.
The example points `user_data` into the retained `Settings` allocation. An
unrelated `Arc::new(())` does not keep a stack value or separately dropped box
alive, even though the binding compiles.
:::

### Example: a dense callback in C

Any language that exports a C function works. With the published header:

::: details C implementation and Rust binding: one f32 mean per cell
```c
#include <string.h>
#include "process_fn_cabi.h"

/* Mean of the dense row; one f32 per cell. */
int32_t mean_dense(const PyCellPackDenseRow *row) {
  if (row->context.abi_version != PYCELLPACK_PROCESS_ABI_VERSION ||
      row->context.struct_size != sizeof(*row) ||
      row->context.dtype != 8 /* f32 */)
    return 1;
  const float *v = (const float *)row->values;
  double sum = 0.0;
  for (uint64_t i = 0; i < row->out_dim; i++) sum += v[i];
  float mean = row->out_dim ? (float)(sum / (double)row->out_dim) : 0.0f;
  memcpy(row->out, &mean, sizeof mean); /* out may be unaligned */
  return 0;
}
```

Link the object into the Rust binary (for example with the `cc` crate) and bind
it with a forced dense format:

```rust
unsafe extern "C" {
    fn mean_dense(row: *const DenseRow) -> i32;
}

// SAFETY: mean_dense is stateless, thread-safe, writes 4 bytes and retains nothing.
let process = unsafe {
    ProcessFn::with_format(RowFormat::Dense, None, Some(mean_dense), std::ptr::null(), Arc::new(()))?
};
let schema = OutputSchema::process(id, out_ids, Dtype::F32, 4, process, MEAN_V1, 0)?;
```

C++ code must mark the function `extern "C"` and must not let exceptions
escape it; wrap the body in `try { … } catch (...) { return 2; }`.
:::

### Scratch space

`OutputSchema::process(…, scratch_bytes)` gives each call a private buffer.
Use it instead of allocating per cell, for example to sort a CSR row's
columns before a rank-based transform. The following fragment only demonstrates
checking and borrowing scratch; supply the transform and output writes:

::: details Scratch-space fragment with an explicit alignment check
```rust
unsafe extern "C" fn ranked(call: *const CsrRow) -> i32 {
    let row = unsafe { &*call };
    let n = row.nnz as usize;
    // Declared scratch_bytes = out_dim * 4 covers n ≤ out_dim positions.
    if row.scratch_bytes < (n * 4) as u64 || row.scratch.align_offset(4) != 0 { return 3; }
    let order = unsafe { std::slice::from_raw_parts_mut(row.scratch.cast::<u32>(), n) };
    // … fill `order`, sort it by value, write ranks to `out` …
    0
}
```

The ABI does not promise an alignment for `scratch`. Check it, as above,
before forming a typed slice, or use unaligned reads and writes.
:::

### Partial CSR rows

`ProcessFn::with_partial_rows` (also `unsafe`) lets the loader skip
materializing columns and values. The callback then sees null `columns` and
`values` and asks `resolve` for the elements it needs. A partial-aware callback
must still handle full rows. The following fragment illustrates resolution,
not a complete output-producing callback:

::: details Partial-row resolution fragment
```rust
unsafe extern "C" fn first_k(call: *const CsrRow) -> i32 {
    let row = unsafe { &*call };
    let k = row.nnz.min(16) as usize;
    let mut columns = [0u32; 16];
    let mut values = [0f32; 16];
    if row.columns.is_null() {
        let positions: [u32; 16] = std::array::from_fn(|i| i as u32);
        // A position at or beyond nnz makes resolve return nonzero.
        let resolve = row.resolve.unwrap();
        let status = unsafe {
            resolve(row.row, positions.as_ptr(), k as u64, columns.as_mut_ptr(), values.as_mut_ptr().cast())
        };
        if status != 0 { return status; }
    } else {
        // Full rows still arrive: copy from row.columns / row.values.
    }
    // … write `out` …
    0
}
```
:::

`resolve` and `row` are valid only during the call. `columns` and `values`
must each have room for `count` entries, and `values` must be aligned for the
schema dtype. Passing a null
`positions` with `count == nnz` returns every element in row order. A
partial-aware callback must still handle full rows, because partial delivery
is offered only when every element of the row maps to an output column.

## Access functions

<DocDiagram name="access-callback" />

An access sequence is a pure function from access index to cell. With `S`
schemas, access `k` is schema `k % S` of position `k / S`. The native form
fills a range:

```rust
pub type AccessFn =
    unsafe extern "C" fn(context: *mut c_void, start: u64, count: u64, cells: *mut CellId) -> i32;
```

```c
typedef int32_t (*CellPackAccessFn)(void *context, uint64_t start,
                                    uint64_t count, uint32_t *cells);
```

### What the loader checks and what it trusts

| Property | Checked? |
|---|---|
| Requested range lies within `len` | Yes: the adapter rejects it before calling |
| Returned ids are below the source's cell count | Yes: `Error::Access { position, cell }` when the cell is located |
| Nonzero status | Yes: `Error::Accesses(status)` cancels the run |
| Writes stay within `cells[0..count]` | **No**: writing past `count` corrupts the caller's buffer |
| Output depends only on `(start + i)` and immutable configuration | Only under a plan: a worker whose cell differs from its plan entry's stops the run with `Error::Plan`. Without a plan a stateful generator silently gives different batches |
| `context` outlives the sequence and tolerates concurrent reads | **No**: use-after-free or data races |
| No unwinding across the boundary | **No** |

Do not advance a shared random-number generator inside the function. Derive randomness from the
position, for example by hashing `(seed, epoch, index)`, as `Shuffle` does.

### Example: a native order in Rust

<<< @/examples/abi.rs#access

<<< @/examples/abi.rs#sequence

The example then runs the loader with `&accesses` exactly as with a
`Vec<CellId>`. Because `ExternAccesses` has no backing slice, a fixed plan
must materialize it for verification; prefer online planning for long
generated sequences.

### Example: the same order in C

::: details C callback with immutable striding configuration
```c
#include "access_fn_cabi.h"

typedef struct { uint64_t cells, stride; } Stride;

CELLPACK_ACCESS_EXTERN CELLPACK_ACCESS_EXPORT
int32_t strided(void *context, uint64_t start, uint64_t count, uint32_t *cells) {
  const Stride *s = (const Stride *)context;   /* read-only: safe to share */
  if (s->cells == 0) return 1;
  for (uint64_t i = 0; i < count; i++)        /* exactly count entries */
    cells[i] = (uint32_t)(((start + i) * s->stride) % s->cells);
  return 0;
}
```
:::

Choose counts and stride values whose multiplication fits in `uint64_t` if you
need mathematical modular-stride behavior; C unsigned overflow otherwise wraps
before the remainder. The application must keep `Stride` alive through every
callback and keep its fields unchanged.

### Corpus access functions

When the application indexes cells as (dataset, row) pairs, implement
`CorpusAccessFn` instead and bind the row count of each registered matrix:

```c
typedef int32_t (*CellPackCorpusAccessFn)(void *context, uint64_t start,
                                          uint64_t count, uint32_t *dataset_ids,
                                          uint64_t *rows);
```

```rust
// counts[d] is the row count of the d-th registered matrix, in registration order.
let accesses = unsafe { ExternCorpusAccesses::new(corpus_fn, context, len, &counts)? };
```

The adapter translates each pair to `offset[dataset] + row` and rejects a
dataset or row outside the bound counts with an error. The same write-bounds,
determinism, lifetime and thread-safety obligations apply. `counts` must match
the order in which matrices were added to the `DataSourceBuilder`; a
different but in-range order is not detected and loads the wrong cells.

## Versioning

| Constant | Value | Covers |
|---|---|---|
| `loader::PROCESS_ABI_VERSION` / `PYCELLPACK_PROCESS_ABI_VERSION` | 1 | `ProcessContext`, `CsrRow`, `DenseRow`, `ResolveFn` |
| `CELLPACK_ACCESS_ABI_VERSION` | 1 | `AccessFn`, `CorpusAccessFn` |

`struct_size` reports the size of the row struct actually passed, which lets
a callback detect a newer, longer struct. A callback built against a different
`abi_version` must not be bound; reject the call with a nonzero status when the
version does not match.

API: [ProcessFn](/generated-loader-struct-ProcessFn),
[CsrRow](/generated-loader-struct-CsrRow),
[DenseRow](/generated-loader-struct-DenseRow),
[ProcessContext](/generated-loader-struct-ProcessContext),
[AccessFn](/generated-loader-type-AccessFn),
[ExternAccesses](/generated-loader-struct-ExternAccesses),
[ExternCorpusAccesses](/generated-loader-struct-ExternCorpusAccesses).
