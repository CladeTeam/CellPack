# Full training integration

This source is an **integration blueprint**, not a runnable small program. It
connects native row processing, a pinned output ring, packed CUDA transfer and a
scGPT model/loss loop. Supply application data, vocabulary, access order and model
configuration before adapting it. For a self-contained first run, use the
[DataLoader program](dataloader.md) or the [CPU process program](process-functions.md).

Install [PyCellPack from PyPI](/python/docs/installation.md) in your application
environment before adapting the [downloadable source](/downloads/examples/loader-example.py).

## Prepare the application inputs

| Input | What the blueprint assumes |
| --- | --- |
| `cpk_path` | An immutable file-backed matrix with suitable finite float32 counts |
| `gene_tokens` | A read-only int64 token per source column; negative entries exclude columns |
| `vocab` | The matching model vocabulary, including pad and CLS tokens |
| `accesses` | uint32 global cell IDs in the desired training order |
| Model and loss | The intended scGPT architecture, objectives and masking/padding conventions |
| Runtime | A CUDA-capable PyTorch environment, a GPU and the selected native binding tools |

This example uses an identity source/output panel, so table position `c` matches
file column `c`. If your application reorders the schema panel, reorder token and
auxiliary tables with it. PyCellPack does not align names or infer preprocessing
from the model checkpoint.

The file imports PyTorch, Numba and scGPT at the top. As written, all are needed
to import it; when extracting one binding path into your application, keep the
imports that path and training loop use. This page does not provide a tested
version lock for the external model package.

## Select one ProcessFn binding

<DocChoices title="Choose the provider you maintain" :items="[
  { label: 'process_from_so', when: 'You already maintain an exported provider library.', detail: 'Supply sampling.so with the matching context struct and sampling_csr / sampling_dense exports.' },
  { label: 'process_from_source', when: 'You want to compile and cache the C++ implementation.', detail: 'The blueprint selects this path. A compatible compiler and the actual source are required.' },
  { label: 'process_from_cfunc', when: 'You want native callbacks compiled through Numba.', detail: 'Use the independent CSR/dense implementation with matching signatures and numeric behavior.' },
]" />

These are alternative providers for one packed output contract. Choose one;
there is no reason to load all three into the producer. `user_data` points at
configuration, `owner` keeps token storage alive, and `key` describes the values
that affect output. A stable pointer address does not establish configuration
identity.

The blueprint uses an opaque fixed-size output with explicit `row_bytes` and
manual field offsets. For new named-output integrations, declare an
[OutputLayout](/python/api/process-functions.md#outputlayout-and-field-views)
and use `process.schema(...)` to infer record size. Whichever representation you
choose must match native field offsets and padding exactly.

## Preserve the transfer dependencies

<DocDiagram name="slot-lifecycle" />

Workers write directly into the ring's pinned host allocation. The application
then transfers the packed record once and constructs genes, values and targets
on the GPU. Keep the host array and Tensor alive until H2D completes; field views
and additional streams have their own storage lifetimes.

Use actual `batch.rows` for a tail batch and derive offsets from the declared
record contract. Drain output before joining the producer. A batch lease covers
all fields in its slot, so an outstanding reader of one field can delay reuse of
the entire slot. See [ring ownership](/python/api/rings.md) and the
[CPU/CUDA example](process-functions.md) for an executable lifetime sequence.

## Source blueprint

::: details Application blueprint: native bindings, pinned delivery and training

<<< @/../../example/loader-example.py

:::

[Download the source](/downloads/examples/loader-example.py).
Start with the [native loader guide](/python/docs/native-loader.md) to verify
values and ordering before adding model execution. The
[ProcessFn reference](/python/api/process-functions.md) defines the callback,
build, identity and output-layout contracts used here.
