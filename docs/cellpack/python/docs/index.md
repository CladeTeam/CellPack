# Using PyCellPack

PyCellPack stores numeric matrices in compressed CellPack streams and reads selected rows with native worker threads. Use it to prepare a matrix once, align datasets to a common gene vocabulary, and deliver ordered training batches.

## Choose your path

<DocDiagram name="python-start" />

Read [installation](installation.md), then run the [matrix quickstart](quickstart.md)
in a writable directory. It writes two rows and checks a reordered selection, so
you can establish storage correctness before adding sampling or a training runtime.

| You need… | Read next | Result |
| --- | --- | --- |
| One selected matrix | [Matrix access](storage.md) | An owned dense or CSR array |
| Several datasets with different column orders | [Gene alignment](alignment.md) | One explicit vocabulary and output panel |
| Training iteration | [DataLoader](dataloader.md) | A dictionary of Tensor fields per batch |
| Direct native integration | [Native loader](native-loader.md) | NumPy views with an explicit slot lease |
| A custom encoded container | [Codec components](codecs.md) | Encoded chunks and records in explicit buffers |

The [API reference](/python/api/) answers parameter, return and lifetime questions.
The [Training](/python/training/) section supplies larger integration programs with
their runtime requirements stated up front.

<DocDiagram name="python-ownership" />

## Objects you will use

`File` identifies an encoded stream. `Store` describes an immutable, lazy matrix view over it. A read decodes that view into an array; indexing alone does not decode values.

For training, wrap a file-backed Store in `ScDataset`, describe sample order with corpus `Accesses`, and iterate `ScDataLoader`. For direct NumPy delivery, register files in `DataSource`, describe outputs with `OutputSchema`, and start a `Loader` run.

PyCellPack does not infer gene identity from column positions or choose epoch sampling policy. Supply both explicitly. Streams store matrix values and positional metadata; names belong in your own metadata or AnnData.
