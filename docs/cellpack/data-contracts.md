<script setup>
import ContractBoundary from './.vitepress/theme/components/ContractBoundary.vue'
</script>

# Data contracts

CellPack checks arguments at API entry points and then trusts the data inside
its per-record loops. A broken argument returns an `Err`. Broken trusted data
is **undefined behavior**: out-of-bounds reads and writes, a segfault, or
silently wrong output, with no error returned.

This page lists what is trusted, who supplies it, and how to stay inside the
contract. For exact shapes, dtypes and identity spaces, see the
[shape and type reference](contracts.md). For callbacks, see [ABI](abi.md).

## Where the boundary lies

<ContractBoundary />

Release builds drop the trusted-side checks entirely. Debug builds keep them
as `debug_assert!`, so a violation panics there instead of corrupting memory.
No cargo feature adds checks; the only switch is `debug-assertions`.

## Summary

| Data | Supplied by | Checked | Trusted | If violated |
|---|---|---|---|---|
| In-memory matrices given to encoders and `CpkWriter::append` | Application | Shape, widths, offsets, index order and range | Nothing | `Err(InvalidInput)` |
| Column maps, schemas, row selections | Application | Length, distinctness, ranges | Nothing | `Err` |
| `.cpk` superblock and directory | File | Hashed and parsed | — | `Err` |
| Chunk metadata and per-cell tables | File | Parsed in full on open; every cell's extent and count on registration; chunk length and rows against the directory | — | `Err(CorruptMeta)` |
| **Chunk record bytes** | File, `append_packed` | Not checked before decoding | Well-formed, produced by the encoder for that chunk | **UB** |
| **Record ↔ metadata ↔ tables pairing** in `scans` | Application | Tables against metadata | The record belongs to that metadata | **UB** |
| Element positions for `map_parts_at` and partial-row `resolve` | Application | Each position below the row's element count | — | `Err(InvalidInput)`; `resolve` returns nonzero |
| Cell ids from an access sequence | Application | Against the source's cell count | — | `Err(Access)` |
| Determinism of `Accesses::fill` | Application | Under a plan, each access's cell against the cell its plan entry was built for | Same access → same cell, without a plan | `Err(Plan)` with a plan; wrong batches without |
| **Process and access callbacks** | Application | Status codes, required callbacks present | Write bounds, lifetime, thread safety, no unwinding | **UB**, see [ABI](abi.md) |
| **External ring memory** | Application | Null, alignment, size, overlap, header | Lifetime and exclusive access under the slot protocol | **UB** |

The rest of this page goes through the bold rows, and the checked rows that
still constrain how data should be produced.

## In-memory matrices are validated

Everything that enters the codec from memory is checked before encoding:
`n_genes` in `1..2^31`, value widths of 1, 2, 4 or 8 bytes, monotone row
offsets that stay inside the index and value arrays, and indices that are
strictly ascending and below `n_genes`. The checks run as part of the encode
pass, so there is no separate validation call to make:

<<< @/examples/contracts.rs#encoder

Column maps and schemas are checked when they are registered, and access ids
when they are used:

<<< @/examples/contracts.rs#map

<<< @/examples/contracts.rs#access

The complete program is `docs/cellpack/examples/contracts.rs`:

```sh
cargo run --locked --manifest-path docs/cellpack/examples/Cargo.toml \
  --bin contracts -- tmp/cellpack-guide/contracts
```

## Record bytes are trusted

A `.cpk` stream protects its superblock and directory with a hash, and every
chunk's metadata is parsed and validated when a `CpkFile` opens it or a
`DataSourceBuilder` registers it. **Record bytes are not.** The decoders assume
each record is exactly what the encoder wrote for that chunk: section lengths
inside the record, ids ascending and below the dimension count, dictionary
ranks below the dictionary size, element counts within the chunk maxima.

A damaged or forged record therefore reads and writes past the record, the
decode workspace or the output row. The decoded element count is compared with
the metadata afterwards and reported as `CorruptRecord`, but by then the damage
is done; treat that error as a symptom, not a guard.

In practice:

- Read only streams written by `CpkWriter` and stored where they cannot be
  modified by untrusted parties. CellPack is not a validator for files from
  untrusted sources.
- Keep files immutable while selections, data sources, loaders and plans use
  them. Rewriting a file in place under a running loader is the same as
  handing it corrupt records.
- `CpkWriter::append_packed` copies an externally packed chunk after parsing
  its metadata only. Its records are trusted from then on; produce such chunks
  only with `scans::encode_chunk` and `scans::pack_chunk`.
- `format::record_nnz` reads only the header and checks it, so it is safe on
  any bytes; matching the metadata count does not make the rest of the
  record valid.

To vet a file of uncertain origin, read it completely once in a build with
debug assertions enabled. Violations then panic at the first inconsistency
instead of corrupting memory:

```toml
# Cargo.toml of the application
[profile.validate]
inherits = "release"
debug-assertions = true
```

```sh
cargo run --profile validate --bin check-files -- data/*.cpk
```

This is a diagnostic, not a security boundary: it relies on the same parser
and costs a full decode.

## Low-level codec calls must keep records with their chunk

The `scans` functions take metadata, tables and record bytes as separate
arguments. They check that the tables match the metadata (`TablesMismatch`)
and that outputs and workspace are large enough. They cannot check that a
record came from the chunk the metadata describes. Derive all three from the
same packed chunk:

```rust
let meta = parse_meta(&packed)?;                 // validated metadata
let tables = decode_tables(&meta)?;              // checked against meta
let records = meta.records_of(&packed)?;         // records of this chunk
let mut workspace = vec![0u64; decode_workspace_words(&meta)];
decode_csr(&meta, &tables, records, Rows::All, IdSpace::Source,
           &mut indptr, &mut indices, &mut values, &mut workspace)?;
```

Mixing metadata from one chunk with records from another, or decoding a record
slice cut at the wrong offset, is undefined behavior even though every
argument type checks.

`Parts` views borrow the workspace, record and metadata they were decoded
from. Consume a `Parts` before reusing any of them; the borrow checker
enforces this for safe code, but raw pointers derived from a `Parts` are not
tracked.

## Positions passed back in

Two interfaces accept positions that address elements of an already decoded
row:

| Call | Requirement |
|---|---|
| `scans::map_parts_at(parts, positions, …)`, `Mapper::map_parts_at` | Every `positions[i] < parts.elements()` |
| Partial-row `resolve(row, positions, count, …)` in a process function | Every `positions[i] < nnz`; output arrays hold `count` entries |

Both check every position before mapping: `map_parts_at` returns
`InvalidInput` and `resolve` returns a nonzero status, which a process
function should pass on. Positions may repeat and need not be sorted. Compute
them from the row's own element count, never from a different row or a stale
value, so that the check never fires in a correct program:

```rust
let n = parts.elements() as u32;
let picks: Vec<u32> = sample.iter().map(|&p| p % n).collect(); // always in range
mapper.map_parts_at(&parts, &picks, &map, &dict, &modes, &mut cols, &mut vals)?;
```

## Access sequences must be pure

`Accesses` is a safe trait, but its contract is stronger than its signature:
`fill(start, out)` must return the same cells for the same accesses on every
call. Workers, the online planner and plan verification all call it, from
several threads, out of order and more than once.

The loader checks every returned id against the source (`Error::Access`).
Under a fixed or online plan, each plan entry also records the cell it was
built for, and a worker whose own `fill` disagrees stops the run with
`Error::Plan` before following the entry's cache frames. Without a plan there
is nothing to compare against: a nondeterministic sequence silently produces
batches that differ from the order the application intended.

Derive randomness from the position instead of from a mutable generator:

```rust
// Wrong: state advances with each call, so repeated ranges differ.
struct Sampled { rng: Mutex<StdRng>, cells: u32, len: u64 }

// Right: a pure function of (seed, index).
struct Sampled { seed: u64, cells: u32, len: u64 }

impl Accesses for Sampled {
    fn len(&self) -> u64 { self.len }
    fn fill(&self, start: u64, out: &mut [CellId]) -> loader::Result<()> {
        for (i, cell) in out.iter_mut().enumerate() {
            let h = splitmix64(self.seed ^ (start + i as u64));
            *cell = (h % self.cells as u64) as CellId;
        }
        Ok(())
    }
}
```

`Shuffle`, `Replicate`, `Offset`, `SliceAccesses` and stored slices satisfy
this contract. Native sequences (`ExternAccesses`, `ExternCorpusAccesses`)
carry the same requirement plus the write-bound and lifetime rules in
[ABI](abi.md#access-functions).

## External ring memory

`Ring::allocate` owns its memory and needs no care. The constructors over
caller memory are `unsafe`:

| Constructor | Checked | Trusted |
|---|---|---|
| `Ring::init(ptr, len, layout)` | Non-null, 4096-byte aligned, `len ≥ layout.bytes()` | Memory outlives the ring **and every batch lease**; there is no owner handle |
| `Ring::with_buffer(data, len, layout, owner)` | Alignment, `data_min_bytes ≤ len ≤ data_bytes` | `owner` keeps `data` allocated; nothing else writes slots being filled |
| `Ring::init_regions(control, …, data, …, owner)` | Control 64-byte aligned, sizes, regions disjoint | Exclusive during initialization; every process follows the slot protocol |
| `Ring::attach(ptr, len)` | 64-byte alignment, size, header magic and version, re-derived offsets | The mapping stays valid; other processes touch it only through the protocol |
| `Ring::attach_regions(control, …, data, …, owner)` | As `init_regions`, plus header | As above |

`REGION_ALIGN` is 64 and `SLOT_ALIGN` is 4096. Use `RingLayout::bytes`,
`control_bytes`, `data_bytes` and `data_min_bytes` to size allocations, and
pass data arrays aligned to 64 bytes so that every schema region inside a slot
is 64-byte aligned. For cross-process rings, map control memory `MAP_SHARED`:
waits use shared futexes.

A batch region stays valid while its lease is held. An asynchronous device
copy reads the region until it completes, so release the lease only after the
copy has finished.

## Checklist

Before handing data to CellPack:

- [ ] Files come from `CpkWriter` and are immutable while in use.
- [ ] Records, metadata and tables passed to `scans` come from the same chunk.
- [ ] `Accesses::fill` is a pure function of the access index.
- [ ] Native callbacks satisfy the [ABI](abi.md) obligations.
- [ ] External ring memory is sized from `RingLayout`, aligned, and outlives every lease.
- [ ] Data of uncertain origin has been read once under `debug-assertions = true`.

API: [Error](/generated-root-enum-Error),
[loader::Error](/generated-loader-enum-Error),
[Accesses](/generated-loader-trait-Accesses),
[Ring](/generated-loader-struct-Ring),
[RingLayout](/generated-loader-struct-RingLayout),
[Mapper](/generated-scans-struct-Mapper).
