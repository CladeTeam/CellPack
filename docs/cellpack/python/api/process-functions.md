# Process functions API

ProcessFn binds native row callbacks, and OutputLayout describes their fixed
output record. Read [the guide](/python/docs/process-functions.md) for a complete
CPU preset workflow and [the runnable program](/python/training/process-functions.md)
for packed CUDA transfer.

```python
from pycellpack.loader.process_fn import ProcessFn, OutputLayout, compile_library, presets
```

<DocChoices title="Find the contract you need" :items="[
  { label: 'Bind native code', when: 'Choose an exported library, source build or Numba callback.', href: '#construction-and-identity', detail: 'Code ownership and configuration identity are separate requirements.' },
  { label: 'Read named output', when: 'Define field offsets or retain processed results.', href: '#outputlayout-and-field-views', detail: 'A layout describes a fixed record; NumPy field views keep its slot leased.' },
  { label: 'Configure a preset', when: 'Use scGPT, Geneformer, UCE or STATE SE.', href: '#model-presets', detail: 'Tables follow schema output order, with float32 mapped input.' },
]" />

## Construction and identity

| Interface | Required input | Result / default |
| --- | --- | --- |
| `ProcessFn.from_library` | Library path or ctypes handle; CSR and/or dense functions | Bound function retaining library and native owners |
| `ProcessFn.from_source` | Source files and/or inline code; exported symbols | Compiled/cached library bound as above |
| `ProcessFn.from_cfunc` | CSR and/or dense Numba callbacks, or compilable functions | Native callback pointers; Numba loads on use |
| `ProcessFn(capsule, identity=...)` | Provider-owned `pycellpack.process.v1` capsule | Advanced binding with no reconstruction recipe |
| `process.schema(out_ids, ...)` | Declared OutputLayout and output-panel IDs | OutputSchema; custom input dtype defaults to float32 |
| `compile_library` | Source files and/or inline C/C++ code | Path to the compiled shared library |

Shared factory keywords:

| Parameter | Contract |
| --- | --- |
| `csr`, `dense` | Exported names, supported function objects or addresses; at least the selected format is required |
| `row_format=None` | Factories infer `"auto"` from both callbacks or the single supplied format; explicit `"auto"` requires both |
| `partial_rows=False` | Opt into partial CSR rows and resolver support; dense delivery has no partial-row protocol |
| `user_data` | Context passed to every call; ctypes/NumPy objects are retained, raw addresses remain the caller's responsibility |
| `key` | Configuration values hashed into the derived identity, including everything read through user_data |
| `identity` | Explicit integer in `[0, 2**64)`; use instead of `key`, not together |
| `owner` | Additional object retaining code or pointer-backed storage; retention alone does not contribute identity |
| `output` | OutputLayout or equivalent structured dtype; optional for opaque output |

Derived library identities include file contents, function names, delivery mode,
partial-row declaration and `key`. Numba identities also include function source
or bytecode and the Numba version; cover any globals or closures in your key.
Addresses need an explicit identity. Supplied context without a key or explicit
identity is rejected.

`process.schema` accepts `dtype`, `id`, `name`, `default` and `scratch_bytes` in
addition to `out_ids`. `OutputSchema(process=...)` infers `row_bytes` from a
declared layout and rejects a conflicting explicit size. Without a layout, use
its explicit `row_bytes` interface; `schema()` and `unpack()` then raise ValueError.

### Compilation and reconstruction

`compile_library` accepts `language`, `name`, `compiler`, `std`, `optimize`,
`flags`, `include_dirs`, `defines`, `depends`, `cache_dir` and `force`. Compiler
selection uses the explicit choice, then CXX/CC, then supported platform drivers.
The build cache includes sources, compiler identity, flags, ABI/preset headers and
declared dependencies. List your additional included files in `depends`; they
are not discovered as an arbitrary transitive dependency graph.

A compiler is resolved even for cached source builds. `from_library` can use an
already built library without compiling. Recipe-based bindings pickle when their
arguments do: source recipes rebuild from cache, library recipes reopen a path
and named symbols, and supported Numba recipes reconstruct callables. Raw
addresses, hand-made capsules or unsupported owner/context configurations do not
provide a portable recipe. Preset recipes retain table values and rebuild native
pointers in the receiver.

For generic factories, an explicit `owner` or integer `user_data` disables the
automatic recipe, even if the owner would itself be picklable. Library/source
recipes also require a path and named exports. Use a reconstruction factory for
such pointer-owning configurations instead of sending the live object.

## OutputLayout and field views

`OutputLayout(fields, align=True)` accepts NumPy structured dtype declarations,
including explicit offsets and itemsize. Fields must be named, nonempty,
native-endian numeric values within the record. Default C alignment includes
tail padding. `align=False` describes a packed layout. The declaration must match
the native callback; it does not guard that callback's output writes.

| Interface | Result |
| --- | --- |
| `layout.dtype` | Structured dtype with field offsets, shapes and padding |
| `layout.names` | Tuple of field names in declaration order |
| `layout.offsets` | Name → byte offset within the record |
| `layout.row_bytes` / `process.row_bytes` | Complete padded record byte size |
| `layout.unpack(raw, copy=False)` / `process.unpack(...)` | Dictionary of named NumPy fields |

`raw` must be a NumPy uint8 array shaped `[..., row_bytes]`; its byte axis is
contiguous, while leading axes may be strided. Scalar fields have the leading
shape; array fields append their declared field shape. The usual source is
`batch.region(name)`.

With `copy=False`, fields may be strided or unaligned and retain raw storage and
its slot lease, including after Batch release or Run closure. `copy=True` gives
independent aligned C-contiguous arrays. Retaining every batch's field views can
fill the output ring and block production. This helper accepts host NumPy arrays,
not CUDA tensors. See [ring ownership](rings.md).

For explicit nonblocking H2D, retain the packed host view and host Tensor until
the copy completes, then release them. Submitting the copy alone does not finish
reading the slot. Device field views retain device storage; coordinate their
CUDA streams independently. [The CPU/CUDA program](/python/training/process-functions.md)
shows the complete lifetime sequence.

## Callback input and ABI

`ProcessFn.include_dir()` and the module-level `include_dir()` locate
`process_fn_cabi.h`, whose capsule name is `pycellpack.process.v1`. The context
contains logical sequence position, global cell ID, schema index, input dtype and
`user_data`. Callback values have already been converted to the schema dtype;
column numbers address the mapped output panel. Every schema at one position
sees the same position value; `schema` is its vector index, not application `id`.

| Delivery | Requirement |
| --- | --- |
| `"csr"` | CSR callback; columns can be unsorted and list stored elements only |
| `"dense"` | Dense callback; present omitted values are zero, absent source dimensions receive the schema default; factories reject `partial_rows=True` |
| `"auto"` | Equivalent CSR and dense callbacks; internal layout/route choices may change which one runs |
| Partial CSR | Null column/value pointers require the supplied resolver; positions address stored elements, not output dimensions |

CSR omissions alone do not identify whether a dimension was missing or zero.
Use a distinguishing dense default or application-owned source metadata when that
information matters. Partial-row resolve accepts repeated/out-of-order element
positions; each must be below `nnz`. Null positions with `count == nnz` request
all elements in order. Its row and output pointers are valid only for the call;
propagate a nonzero resolver status.

Callbacks may run concurrently. Write within `row_bytes` and `scratch_bytes`,
retain no borrowed pointers, and prevent exceptions or unwinding from crossing
the C ABI. Native write bounds are the provider's responsibility. A nonzero
callback status fails the run as a RuntimeError; it is a provider-defined code.

## Model presets

All four factories return ProcessFn with a declared layout, `input_dtype=float32`
and `out_dim=len(tokens)`. Schemas reject incompatible mapped dtype or width.
Each token and auxiliary-table entry addresses **one schema output dimension in
out_ids order**, rather than a standard ID or original file column. Tables are
copied into immutable contiguous storage. Negative tokens exclude dimensions.

Let `L=length`, `Q=positive+negative` for UCE and
`Q=positive+negative+shared` for STATE SE.

| Factory | Required tables / special IDs | Fields per row |
| --- | --- | --- |
| `scgpt` | tokens; pad_token, cls_token | genes int64[L], values float32[L], targets float32[L] |
| `geneformer` | tokens, medians; pad_token, cls_token, eos_token | ids int64[L], length uint32 |
| `uce` | tokens, chroms, starts; pad_token, cls_token, chrom_offset, chrom_close_token | ids int64[L], loss_ids int64[Q], labels float32[Q], length uint32 |
| `state_se` | tokens; global_size | ids int64[L], tasks int64[Q], targets float32[Q], counts float32[L], total float32 |

Medians, chromosomes and starts have the token-table width. Tokens must be a
nonempty integer vector with at least one nonnegative entry. Factories validate
option domains and table dimensions; per-cell content can still fail in native
execution. Geneformer excludes nonpositive medians. UCE requires at least two
eligible expressed genes in a cell and sufficient sentence capacity; the header
also defines masking-related status codes. STATE SE requires nonnegative tokens
below `global_size`; `batch` groups consecutive access positions for shared query
draws and `cls_index` indexes the eligible panel.

Defaults and exact sampling arguments are listed in the declarations. They
implement the shipped templates rather than loading a checkpoint's preprocessing
configuration. Factories compile C++17 templates and pass build keywords to
`compile_library`; native/Python struct sizes and offsets are compile-time checked.
Headers are under `include/template/{scgpt,geneformer,uce,state_se}.hpp`.

::: details Instantiate a C++ template directly

```cpp
#include "template/scgpt.hpp"
struct Config : pycellpack::preset::ScGptDefaults {
    static constexpr uint32_t length = 1200;
};
PYCELLPACK_EXPORT_PROCESS_FN(scgpt, pycellpack::preset::ScGpt<Config>)
```

Bind `scgpt_csr` and `scgpt_dense` with `ProcessFn.from_source`. Supply the header's
exact runtime-table struct through `user_data`, a key covering its contents, and
a matching OutputLayout. `template/common.hpp` adapts these models to the C ABI.
:::

## Errors

Invalid layouts, identities, delivery modes, table dimensions and options raise
TypeError or ValueError. Missing compilers raise CompilerNotFoundError;
compilation failures raise ProcessBuildError; library loading or missing exports
raise ProcessLibraryError. Execution and callback failures surface during
iteration or joining. After failure, use the [run recovery rules](rings.md)
rather than consuming partially written output.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/process-functions-members.md-->

:::
