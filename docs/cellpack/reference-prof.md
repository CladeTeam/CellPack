# Timing and observations: prof

`cellpack::prof` defines timing sites, masks, per-thread records and aggregated
reports. Ordinary loader users select a `SiteMask` and read `RunStats.profile`.
Applications instrumenting native work can use the lower-level profiler types.

For a measurement workflow, read [statistics and profiling](profiling.md).

<DocChoices title="Choose the interface level" :items="[
  { label: 'SiteMask + ProfileReport', when: 'Inspect a loader run', detail: 'Select timing sites in LoaderConfig, then inspect final RunStats.', href: '#application-facing-interfaces' },
  { label: 'Setup + Profiler', when: 'Instrument native work', detail: 'Own one mutable record per thread with a compatible shared clock and layout.', href: '#instrumentation-interfaces' },
]" />

## Application-facing interfaces

| Interface | Inputs | Returns / meaning |
|---|---|---|
| `SiteMask::parse` | Site/group names | Checked mask; unknown names return a loader configuration error |
| `SiteMask::NONE / ALL` | None | No sites / every declared site |
| `contains / names` | Site or mask | Membership / enabled site names |
| `SITES / GROUPS` | Static declarations | Descriptions and named groups |
| `ProfileReport::site` | Exact site name | Enabled site, including zero-call sites; None otherwise |
| `ProfileReport::time` | Name | Total duration, zero if not recorded |

## Mask selection

`SiteMask::parse` unions case-sensitive site/group names and accepts `"all"`.
An empty list produces `NONE`. Unknown names fail with `loader::Error::Config`.
`names()` follows `Site` declaration order, not input order.

A site name takes precedence over a same-named group. In particular,
`parse(&["worker"])` enables the worker lifetime site only. To expand the
worker group explicitly:

```rust
use cellpack::prof::{GROUPS, SiteMask};

let (_, members) = GROUPS.iter().find(|(name, _)| *name == "worker").unwrap();
let stages = SiteMask::parse(members)?;
```

`LoaderConfig.profile = None` selects `Off` and returns no timing report.
`Some(NONE)` selects `On` with an empty report. Both retain event counts.

## Report structure and units

`SiteInfo` records a name, `Kind`, `Detail`, optional parent, and description.
`Kind` selects spans or histograms. `Detail` selects a total, per-schema records,
or schema/size-band/partial-row records.

`ProfileReport` has schema count, band count and ordered site reports.
`SiteReport` includes calls, inclusive `time`, `self_time`, per-key details and
optional histogram entries. Histogram lower bounds are nanoseconds.
`DetailReport` identifies a schema or schema/band/partial key, calls and duration.

Missing timing data is not proof that a phase took no time. Check the selected
mask. Inclusive parent and child durations overlap; do not sum every site as an
end-to-end elapsed time. `self_time` subtracts only enabled direct children's
durations, saturating at zero; it is not exclusive CPU time. Disabled children
cannot be subtracted. Aggregated thread durations can exceed run wall time.

For `Detail::Schema`, `key` is the schema index in loader order, not its public ID.
For `Detail::SchemaBand`, decode it as:

```text
partial = key % 2
band    = (key / 2) % report.bands
schema  = (key / 2) / report.bands
```

`process` uses schema/band/partial detail; `map` uses schema detail.
`loader::PROCESS_BOUNDS` defines process size bands. Details and histogram buckets
are emitted only when their count is nonzero. A histogram pair
`(lower_bound_ns, count)` represents a log2 bucket, not an exact latency or
percentile. Time fields use `std::time::Duration`.

## Instrumentation interfaces

| Interface | Responsibility and contract |
|---|---|
| `Clock` | Clock readings and nanoseconds-per-tick conversion |
| `Layout::new(schemas, bands)` | Detail/histogram word layout; zero dimensions allocate at least one slot |
| `Layout::words` | Required record length in `u64` words |
| `Record::new` | Zeroed words for the layout |
| `Record::add` | Adds another record, growing the destination if necessary |
| `Profiler` | Start/stop, keyed detail, tick/mark accounting, per-thread fork and report setup |
| `Off` | Untimed implementation, no clock-read work |
| `On` | Timed implementation holding one thread's mutable record |
| `Setup` | Creates a compatible profiler and converts records to reports |
| `Snapshots` | Per-thread publication slots and an aggregated live sum |

Keep layout, site and detail keys compatible. A thread owns its own mutable
record; fork profilers for other threads rather than concurrently writing one.
Raw clock ticks are not nanoseconds until converted.

`Setup::new(mask, schemas, bands)` creates the shared clock and layout.
`setup.profiler()` creates an `On`; `fork()` shares clock, mask and layout but
starts a fresh zeroed record. Merge compatible records with `Record::add` before
`setup.report(&record)`. Addition follows word positions and does not check
semantic layout compatibility.

`stop_at` keys must be valid for the site's layout. A zero start/mark represents
a disabled site; pair start and stop with the same site and clock. The clock uses
invariant TSC where available and `Instant` otherwise; `nanos_per_tick` supplies
the conversion used at report time.

### Live publication

`Snapshots::new(threads, words)` allocates publication slots.
`publish(i, words)` is for the single owning writer of slot `i`.
`sum()` returns a consistent per-slot sum, retrying in-progress publications.
It is a live observation, not an atomic barrier across all threads.
Use one slot per owning writer and the same word count/layout for all slots.
`publish` copies only iterator words that fit: supply a full compatible record;
it does not clear words omitted by a short iterator.

## Complete interface index

<!--@include: ./_catalog/prof.md-->
