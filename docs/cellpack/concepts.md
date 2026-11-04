# Data and execution model

The same row has several identities as it moves from a file into a batch. This
page introduces them before you build [data sources](data-sources.md) and
[access sequences](accesses.md). Keep three questions separate: where a row is
stored, which sequence position requests it, and where its result is delivered.

## Matrices, chunks, records, and pages

A `.cpk` file holds one matrix. Rows are divided into chunks, each sharing a set
of coding models and an optional column permutation. Each row has an independently
decodable record; empty rows may have no record bytes. Records are packed onto
pages so a reader needs only the pages covering its record and the chunk's metadata.

Independent decoding removes dependencies between records. It still requires
the matching chunk metadata and decode tables. Neighboring records may share a page.

| Object | Meaning |
|---|---|
| `.cpk` | One matrix stream, finalized by `finish` |
| Chunk | A group of rows with common coding models |
| Record | One row's compressed representation |
| Page | The unit used for aligned reads and cache frames |
| Cached row | A decoded input row in its stored dtype |

## Four column spaces

<DocDiagram name="column-spaces" />

Storage reads normally restore source order. A loader takes a source-to-standard
map for each matrix and an output-to-standard list for each schema. It composes
these maps when constructed. Name lookup and vocabulary construction remain with
the application.

## Cells, positions, accesses, and batches

A `CellId` is a global `u32` row id assigned by a data source. A position is a
logical place in the requested sequence and can refer to a cell more than once.
With `S` output schemas, every position has `S` accesses:

<DocDiagram name="access-order" />

For two schemas, accesses `0` and `1` both belong to position `0`; access `2`
starts position `1`. Repeating a cell creates another requested output position.
It does not create another stored row. Cache reuse may share input work while
preserving the requested output order.

An output slot collects every schema's output for up to `slot_cells` consecutive
positions. A batch is a lease on a completed slot. The last batch may have fewer
rows. The schemas of a position may request the same cell or different cells.

## Bounded output and planned reuse

The output ring has a fixed number of slots. A producer cannot overwrite a slot
while a consumer holds its lease. Slow consumption applies backpressure.

A cache plan assigns input pages and decoded input rows to frames over intervals
of accesses. Workers execute those reads, waits, and decodes.
`Planning::None` disables planned caching but still uses worker-private I/O
buffers and the output ring.

The output ring, the planned cache, and the operating system's page cache serve
different purposes. See [memory and performance](performance.md) for their budgets.
