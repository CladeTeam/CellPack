# Access sequences and AccessFn API

```python
from pycellpack.loader import Accesses, AccessFn
from pycellpack.loader.access_fn import numba_bindings
```

This `Accesses` describes **global cell IDs**. `from pycellpack import Accesses`
instead imports the [training corpus dataclass](dataloader.md), containing dataset
IDs and local rows. Use the [sequence guide](../docs/access-functions.md) for
executable construction and inspection examples.

<DocChoices title="Select the sequence representation" :items="[
  { label: 'array / shuffle / subset', when: 'Use stored IDs or a built-in seeded order.', detail: 'Native Accesses can be sliced and composed without decoding matrix data.', href: '#stored-and-computed-sequences' },
  { label: 'replicate / offset', when: 'Adapt one cell per position to several schemas.', detail: 'The expanded sequence contains one cell per access, in position-first order.', href: '#multi-schema-composition' },
  { label: 'AccessFn', when: 'Supply your own native range generator.', detail: 'A binding retains code and context; the callback owns sampling policy.', href: '#accessfn-factories-and-binding' },
]" />

## Stored and computed sequences

| Factory | Arguments and defaults | Result / constraints |
| --- | --- | --- |
| `Accesses.array` | `cells` | Snapshot of a 1-D native-endian, aligned, C-contiguous uint32 ndarray; retains order and repeats |
| `Accesses.shuffle` | `cells, epochs, seed=0, block=1, group=1` | Computed epochs over `0..cells`; positive cells up to `2**32`, positive epochs, uint64-representable total length |
| `Accesses.subset` | uint32 `cells` buffer, `epochs, seed=0, block=1, group=1` | Snapshot the subset, then shuffle its entries; nonempty, repeats allowed |
| `sequence.slice` | `start=0, count=None` | Lazy bounded interval; None takes the remainder |
| `sequence.fill` | `start=0, count=None, out=None` | Materialize an interval as a NumPy uint32 vector |

`block` and `group` must be positive and their product fit uint64; `block` is
capped at the epoch's entry count. Each epoch permutes blocks of consecutive
entries and mixes entries within groups of shuffled blocks. `subset` permutes
entry positions, so repeated IDs preserve their multiplicity.

Direct native factories require the uint32 buffer above. The Python Loader's
ordinary array input separately accepts 1-D integer arrays, range-checks IDs in
`[0, 2**32)`, converts to contiguous uint32 and lets native code snapshot it.
Neither construction proves that every ID belongs to the particular source;
source bounds are checked when the sequence is used.

For `fill`, explicit `count` wins. With `count=None`, a supplied `out` determines
the count from its length; otherwise the remainder is generated. `out` must be a
writable C-contiguous 1-D uint32 ndarray of exactly that count, and is returned
unchanged as an object. The interval must lie within the sequence. Do not mutate
or resize borrowed arrays concurrently with a native call.

`slice` changes which interval is addressed without materializing callback
results. Its new index zero addresses the original `start`. `fill` generates
only the requested interval. Array, shuffle and composed sequences carry
reconstruction recipes; raw callback-bound native sequences cannot be pickled.

## Multi-schema composition

<DocDiagram name="sample-identities" />

With `S` schemas, access `k` is schema `k % S` of position `k // S`. Sequence
length must be divisible by `S`; it counts accesses, not output positions or
batches. Composition takes a native Accesses or a uint32 position vector:

| Factory | Meaning |
| --- | --- |
| `Accesses.replicate(positions, schemas)` | Repeat every position's cell for each schema; schemas must be positive |
| `Accesses.offset(positions, offsets)` | Position cell plus `offsets[s]` for schema `s`; nonempty uint32 offsets |

For positions `[5, 1]` and two schemas, replication yields `[5, 5, 1, 1]`.
Offsets `[0, 6]` yield `[5, 11, 1, 7]`. Offset overflow is an error, and resulting
cells still need to belong to the source. These offsets do not map columns.
A bare AccessFn is not a composition input: bind it to native Accesses first.

## AccessFn factories and binding

`from_library`, `from_source` and `from_cfunc` return an owned AccessFn.

| Parameter | Default / contract |
| --- | --- |
| `length` | Required integer in `[0, 2**64)`; bool rejected; counts callback results |
| `kind` | `"global"`; alternatively `"corpus"` |
| `function` | `"generate_accesses"` for library/source factories |
| `user_data` | None; nonnegative address, contiguous NumPy array, or supported ctypes storage |
| `key` / `identity` | Mutually exclusive; user_data requires one covering its values; identity is a uint64-range integer |
| `owner` | None; retains additional code/configuration storage |
| `sources` / `code` / `**build` | Source files, inline source and [compiler options](process-functions.md) for from_source |

Library/source factories retain the shared library, pointer-backed user_data and
owner. Derived identity hashes code, kind and key. Binding a library without a
file path or using a raw function address requires explicit `identity`.
Changing context values after deriving identity makes it inaccurate; replace
samplers and dependent plans when configuration changes.

| Operation | Returns / binding behavior |
| --- | --- |
| `sampling.bind(counts=None)` | Native Accesses retaining owners; corpus kind requires row counts in dataset registration order |
| `sampling.fill(start=0, count=None, counts=None, out=None)` | Bind then inspect global cell IDs, using the fill contract above |
| `AccessFn.include_dir()` | Directory containing `access_fn_cabi.h` |
| `numba_bindings()` | Lazily import Numba and expose `GLOBAL_SIGNATURE` / `CORPUS_SIGNATURE` |

ScDataLoader, native Loader start/compile/replay and PageTrace bind corpus counts
from their current dataset views. Global callbacks already name concatenated
cells; binding them does not remap IDs. Corpus row scratch is per thread and is
outside MemoryEstimate's known reservations. A fixed Plan may materialize a
computed sequence to verify exact values during preparation.

## Callback ABI and lifetime

```c
typedef int32_t (*CellPackAccessFn)(void *context, uint64_t start,
                                  uint64_t count, uint32_t *cells);
typedef int32_t (*CellPackCorpusAccessFn)(void *context, uint64_t start,
                                        uint64_t count, uint32_t *dataset_ids,
                                        uint64_t *rows);
```

Fill exactly `count` entries for `[start, start + count)`. Both corpus outputs
have `count` entries; dataset IDs index registered views and local rows must be
below that view's row count. Return zero on success. Calls may be concurrent,
repeated or out of order, and results must depend only on access index and
immutable configuration. Native code and context must outlive every call.
Never retain borrowed output pointers, write outside their bounds, or unwind
through the C ABI.

Invalid construction, interval, buffer or corpus-row declarations raise type,
value or numeric-range errors. A nonzero callback status becomes a RuntimeError;
failures may surface while inspecting, planning, iterating or joining. Discard
partially filled inspection output after failure. Build, compiler-discovery and
library-binding failures use the [process-function error classes](process-functions.md).

Picklable factory recipes reconstruct AccessFn in another process. Raw integer
user_data or additional process-local owners disable that recipe; rebuild from
values externally. `Accesses.function` and `corpus_function` expose raw-address
bindings for advanced integration and retain only the supplied owner. Neither
native pointer validation nor a retained owner establishes ABI correctness.

## Declarations

::: details Complete declarations and members

<!--@include: ../../_catalog/python/access-functions-members.md-->

:::
