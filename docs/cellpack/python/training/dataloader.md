# DataLoader and sample ordering

This complete program generates two small matrices in a temporary directory,
aligns their different column panels, and delivers scGPT preset fields for two
caller-defined epochs. It checks integration of sample order, native processing
and Tensor delivery; it does not construct a model or optimizer.

For a first dense `X` batch with checked numeric values and no process compiler,
start with [the training guide](/python/docs/dataloader.md).

## Choose a run

Every mode needs PyTorch and a C++17 compiler: the program always compiles its
scGPT process preset. The access-function mode additionally compiles a corpus
callback. Install PyTorch separately for the target CPU or CUDA environment.

From the repository root in a prepared environment:

```sh
uv run example/dataloader-example.py
uv run example/dataloader-example.py --access-fn
uv run example/dataloader-example.py --device cuda
```

| Command option | What changes | Required device |
| --- | --- | --- |
| No option | Materialized dataset/local-row arrays, copied CPU Tensor batches | CPU |
| `--access-fn` | Native callback returns the same kind of corpus pairs | CPU |
| `--device cuda` | Packed pinned-host transfer and contiguous genes/values fields | Compatible PyTorch CUDA installation and GPU |

Each epoch replaces `loading.accesses` or `loading.access_fn` before iteration.
The corpus has 12 samples and batch size 4, so each epoch prints three steps and
ends with `accesses=12`. Each batch contains `genes`, `values` and `targets`, all
with shape `(4, 8)`. CPU output lines therefore include:

```text
epoch=0 step=0 genes=(4, 8)@cpu, values=(4, 8)@cpu, targets=(4, 8)@cpu
```

The token table is in output-panel order. A returned field is a Tensor, while
the caller's Accesses rows still refer to dataset-local input rows. These are
separate identities.

## Complete source

<DocDiagram name="python-training" />

::: details Complete program: fixture, sampler, preset and two epochs

<<< @/../../example/dataloader-example.py

:::

[Download the source](/downloads/examples/dataloader-example.py).
See the [DataLoader API](/python/api/dataloader.md) for lifecycle and transfer
contracts, or [AccessFn](/python/docs/access-functions.md) for callback ownership
and deterministic range generation.
