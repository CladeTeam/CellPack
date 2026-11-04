# Python API reference

Start with the object that owns your result: a matrix view, a Tensor batch, or
a leased NumPy batch. The module pages explain parameters, returns and ownership;
their declaration sections come from current Python source and the native type
stub. Use the [guides](/python/docs/) for complete programs and expected output.

<DocChoices title="Find your entry point" :items="[
  { label: 'Store', when: 'Read or write a matrix', detail: 'Lazy selections; decoding returns NumPy or CSR arrays.', href: '/python/api/storage.html' },
  { label: 'ScDataLoader', when: 'Feed a training loop', detail: 'Corpus sampling and dictionaries of CPU or CUDA Tensors.', href: '/python/api/dataloader.html' },
  { label: 'Loader', when: 'Control native batch delivery', detail: 'Explicit sources, schemas, planning and leased NumPy output.', href: '/python/api/loader.html' },
  { label: 'scans', when: 'Build a custom encoded container', detail: 'Chunk and record codecs with explicit metadata and buffers.', href: '/python/api/codecs.html' },
]" />

## Browse by responsibility

| Area | Main interfaces | Reference |
| --- | --- | --- |
| Matrix storage and file tools | Store, File, Writer, CSR, read/write/compress, io CPK tools | [Storage](storage.md) |
| Training Tensor batches | ScDataset, corpus Accesses, ScDataLoader | [Training](dataloader.md) |
| Native NumPy loading | DataSource, OutputSchema, LoaderConfig, Loader | [Loader](loader.md) |
| Sample sequences | Native Accesses and AccessFn | [Access functions](access-functions.md) |
| Output memory and execution | RingLayout, Ring, Batch, Consumer, Run | [Rings and lifecycle](rings.md) |
| Cache planning | PlanOptions, Plan, PageTrace, CostModel | [Planning](planning.md) |
| Native row processing | ProcessFn, OutputLayout, model presets, compiler | [Process functions](process-functions.md) |
| One-host shared loading | SharedRing, DistributedLoader, DistributedRun | [Distributed](distributed.md) |
| Diagnostics and calibration | RunStats, MemoryEstimate, profile reports, costs | [Statistics](statistics.md) |
| Chunk and record codecs | scans functions, metadata, tables, reusable encoder/decoder | [Codecs](codecs.md) |
| Annotated matrices | read_h5, write_h5cp, write_h5ad, open_matrix, AnnData | [AnnData](anndata.md) |

## Imports and dependencies

The distribution and import name are both `pycellpack`. Common storage, training, planning and loading classes are re-exported at the root. Advanced ring/statistics interfaces live under `pycellpack.loader`; codecs live under `pycellpack.cellpack.scans`. The compiled `_cellpack` module is private.

Importing the root does not load SciPy, AnnData, h5py, pandas, PyTorch or Numba. Install the optional integration before using it. Torch is installed separately. See [installation](/python/docs/installation.md).

## Conventions

Unless noted, counts are integers and byte sizes are bytes. Store selections are
orthogonal and retain axes. Output IDs define column order; supply gene identity
explicitly rather than relying on names or matching column positions.

| Name | Import | Identity space |
| --- | --- | --- |
| Corpus sample order | `from pycellpack import Accesses` | Dataset ID and row within the dataset's Store view |
| Native access sequence | `from pycellpack.loader import Accesses` | Global cell ID assigned by DataSource |

These two `Accesses` types serve different layers. Use corpus accesses with
ScDataLoader and native accesses with Loader.

Native validation uses ValueError (and Python input type errors use TypeError), I/O uses OSError, and worker/cancellation/busy-state failures use RuntimeError. Async execution errors can arise while iterating or joining. Callback build errors have the process-function exception classes.

For NumPy batch retention, read [rings and lifecycle](rings.md) before starting a
long run. For Tensor transfer and application integration, continue with the
[training programs](/python/training/).
