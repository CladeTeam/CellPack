# Align genes across datasets

Column position is not gene identity. Assign a stable nonnegative integer to each gene in your shared vocabulary, and keep that vocabulary with your metadata. PyCellPack accepts the integer maps; it does not match gene names.

<DocChoices title="Choose the mapping you need" :items="[
  { label: 'ScDataset.ids + out_ids', when: 'Combine training datasets with different panels.', detail: 'Map visible columns to standard identities, then choose model input order.', href: '#source-ids-and-output-ids' },
  { label: 'Store.map_genes', when: 'Reorder or widen one matrix view.', detail: 'Map source columns to output positions; this does not establish a shared vocabulary.', href: '#mapping-a-store-without-a-loader' },
]" />

## Source IDs and output IDs

Suppose dataset A stores genes `[ACTB, GAPDH, CD3D]`, while dataset B stores `[CD3D, ACTB]`. Choose standard IDs `ACTB=10`, `GAPDH=20`, `CD3D=30`.

<DocDiagram name="python-alignment" />

This complete CPU example writes two files and requires PyTorch for iteration.
Dataset A measures GAPDH as zero; dataset B has no GAPDH column at all.

```python
import numpy as np
import pycellpack as cp

first = cp.write("panel-a.cpk", np.array([[1, 0, 3]], np.float32), workers=1)
second = cp.write("panel-b.cpk", np.array([[7, 8]], np.float32), workers=1)
corpus = [
    cp.ScDataset(first, ids=np.array([10, 20, 30], np.uint32)),
    cp.ScDataset(second, ids=np.array([30, 10], np.uint32)),
]
order = cp.Accesses(np.array([0, 1], np.uint32), np.array([0, 0], np.uint64))
loading = cp.ScDataLoader(
    corpus, order, out_ids=np.array([30, 20, 10], np.uint32),
    default=-1, batch_size=2, workers=1, device="cpu", backend="pread", direct_io=False,
)
with loading:
    batches = list(loading)
assert len(batches) == 1
np.testing.assert_array_equal(batches[0]["X"].numpy(), [[3, 0, 1], [7, -1, 8]])
```

| Requested sample | CD3D, ID 30 | GAPDH, ID 20 | ACTB, ID 10 |
| --- | --- | --- | --- |
| Dataset A, row 0 | 3 | 0: present zero | 1 |
| Dataset B, row 0 | 7 | -1: absent column | 8 |

`ids[c]` identifies visible source column `c`. `out_ids[j]` identifies output column `j`, determining model input order. Non-dropped source IDs must be distinct within each ScDataset. `-1` or `uint32.max` drops a source column. Output IDs for ScDataLoader must be distinct, nonempty and below `uint32.max`.

A gene absent from a source takes `default`. A present gene whose sparse record has no entry has value **zero**, even when `default` is nonzero. The IDs establish presence at matrix level; numeric zero does not mean a column is absent. Choose a sentinel outside valid data values when the model must distinguish these cases.

## Dataset views

For `ScDataset(store[:, columns], ids=...)`, provide one ID per **visible** column, in its visible order. Access rows address the view, so row 0 of `store[[5, 2]]` is stored row 5. Ordinary subsets and reorderings stay lazy. One stored column cannot contribute more than one kept dataset dimension; dtype-converting Store views are also rejected. Select the output dtype on the DataLoader.

Keep the standard-ID vocabulary and each matrix's column IDs with your dataset
metadata. An output panel `[30, 20, 10]` names those identities; it does not mean
stored columns 30, 20 and 10. Within a dataset, use `-1` or `uint32.max` to drop
columns; neither marker is valid in ScDataLoader's output panel.

## Mapping a Store without a loader

`view.map_genes(gene_map, n_genes=...)` uses a different map: each source column maps to an **output position**, with `-1` dropping it. Missing output positions read as zero. Targets must be distinct. This positional map does not register a standard gene vocabulary.

For native loading, `DataSource([(source, ids)], vocab=...)` maps stored columns to standard IDs below `vocab`; `OutputSchema(out_ids)` chooses their output order. A DataSource Store may select rows but may not select columns. Use ScDataset to adapt ordinary column views.

Continue with [training batches](dataloader.md) for repeated samples, epochs and
tail batches. The [API reference](/python/api/dataloader.md) lists the exact input
validation and ownership rules.
