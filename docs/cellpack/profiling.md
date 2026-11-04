# Statistics, profiling, and cost models

Start with final event counts to establish what a run did, then enable selected
timing sites to investigate where it waited or worked. Keep inputs, access order,
schemas, consumer work and execution settings fixed when comparing runs.

<DocChoices title="Choose a measurement" :items="[
  { label: 'RunStats', when: 'Check delivered work and application I/O', detail: 'Always-on counts and run wall time; no timing mask required.', href: '#always-on-counters' },
  { label: 'Selected timing sites', when: 'Investigate a limiting stage', detail: 'Per-thread durations, details and histograms; compare with an untimed run.', href: '#enable-selected-timing-sites' },
  { label: 'CostModel', when: 'Price planned page and row reuse', detail: 'Resource estimates supplied for the actual machine and workload.', href: '#cost-model' },
]" />

## Always-on counters

Every successful run returns `RunStats`, even when profiling is disabled.

| Field | Interpretation |
|---|---|
| `elapsed` | Worker and plan-thread wall time after setup |
| `setup_elapsed` | Frame sizing and shared-state setup before execution |
| `counters` | Accesses, decoded rows, hits, admissions, bytes, and stalls |
| `io` | Request counts, requested and returned bytes, request-size buckets |
| `peak_span` | Largest access lead past the execution frontier |
| `cache_bytes`, `private_bytes` | Frame reservations |
| `outputs` | Per-schema access, decode, row-hit, and process counts |
| `profile` | Timing report when enabled |
| `online` | Online compiler statistics when used |

`counters.accesses` counts one delivery per schema per position. It is not the
number of distinct cells or batches. Multiple schemas of one position can share
one decode. `page_hits` count page uses served by cached frames; `row_hits`
refer to decoded input rows.

Live `Run::snapshot()` exposes published counters. Use final statistics after
successful completion for reporting. Workers publish every 1024 completed
positions, so a small or recently started run can still show zero counters.
Snapshots are not a simultaneous observation of all threads.

The [loading example](loading.md) requests five positions through two schemas.
It asserts `stats.counters.accesses == 10`, although there are only five logical
output rows. `stats.outputs` follows schema order: each schema receives five
accesses. Global `decoded_rows` counts shared decoding once; per-output counts
credit each schema using that decode, so their sum can differ from the global count.

`io.read_bytes` counts bytes returned to the application, including reads served
by the OS page cache. It is not physical-device traffic. `io.sizes` groups
requests by pages: `1`, `2`, `3–4`, `5–8`, `9–16`, `17–32`, and `33+`. See
[memory and performance](performance.md) for reservation and measurement boundaries.

## Enable selected timing sites

Configure the mask when constructing the loader; no Cargo feature is needed:

```rust
use cellpack::loader::LoaderConfig;
use cellpack::prof::SiteMask;

let config = LoaderConfig {
    profile: Some(SiteMask::parse(&["io", "cpu", "wait"])?),
    ..LoaderConfig::default()
};
```

This fragment fits into the [complete loading program](loading.md).
`SiteMask::ALL` selects every site; narrow the mask once you know which stage
you want to investigate.

| Group | Main coverage |
|---|---|
| `io` | Submit, reap, I/O wait, request latency |
| `cpu` | Decode, row admission/gather, map, process |
| `wait` | Frame, plan, window, and output waits; idle |
| `planner` | Access generation, row/page selection, freezing, and publication |
| `calibrate` | Sites used by cost calibration workflows |

Individual names such as `decode`, `process`, and `wait.gen` are also accepted.
`prof::SITES` describes each site's name, record kind, parent, and per-schema detail.
Names are case-sensitive. `worker` is both a site and a declared group; parsing
that name selects the **site** first, timing worker lifetime only. Use explicit
groups for stage coverage, or expand the worker group's member list from
`prof::GROUPS` as shown in the [prof reference](reference-prof.md#mask-selection).

Untimed and timed implementations are compiled together. `profile = None`
selects the untimed implementation. A timed implementation checks the selected
site mask; disabled sites avoid clock reads. Instrumentation still changes the
workload, so assess its overhead before interpreting performance.
`Some(SiteMask::NONE)` still selects the timed implementation with an empty mask;
use `None` for the untimed baseline.

## Diagnose a limiting stage

<DocDiagram name="profiling" />

Read the enabled site reports alongside counters:

| Observation | Next check |
|---|---|
| `wait.ring` grows | Consumer progress and batch leases; I/O concurrency cannot free held slots |
| `wait.plan` grows | Online planner progress and `plan.*` timing |
| `wait.window` grows | Execution window and the oldest unfinished position |
| `wait.gen` grows | The access admitting the required cached frame and its I/O completion |
| `decode`, `map` or `process` dominates | Stored layout, output dimensions, conversions and process details |
| `io.wait` or `io.latency` grows | Backend, request sizes, storage state and CPU scheduling |

These are leads, not standalone diagnoses. `Pread` completes reads inside `issue`;
it does not have the same submit/reap/wait breakdown as `Uring`. `io.latency`
measures queueing through completion harvest, including software delay, rather
than device-only latency.

After consuming a run, inspect its returned `stats`:

```rust
if let Some(report) = &stats.profile {
    if let Some(site) = report.site("decode") {
        println!("decode: {} calls, {:?} total", site.calls, site.time);
    }
    if let Some(site) = report.site("process") {
        for detail in &site.details {
            println!("process key {}: {} calls, {:?}", detail.key, detail.calls, detail.time);
        }
    }
}
```

Expect an entry for each enabled site, including zero-call sites; durations depend
on the workload. `report.site(name) == None` means the site was not selected or
its name is unknown, rather than proving it took no time.

Site durations sum intervals across threads and can exceed run wall time.
Parent/child intervals also overlap. `self_time` subtracts only **recorded direct
children**; it changes with the mask and is not CPU utilization. Histogram pairs
contain a bucket lower bound in nanoseconds and a count, not exact latencies.

## Cost model

`CostModel` expresses device request capacity, byte bandwidth, worker physical
cores, memory bandwidth, and operation costs in core-seconds. Its output entries
follow schema order. `Loader::plan_options` starts with NVMe default costs based
on `Loader::cores()`; replace them with measurements for your machine and workload.

For a priced window, let `R` be requests, `B` storage bytes, `C` core-seconds,
`M` cached-row memory bytes, and `s` the non-overlap share:

$$
d = \max(R / \mathrm{IOPS}, B / \mathrm{BW}),\qquad
c = C / \mathrm{cores}
$$

$$
T = \max\left(\max(d,c) + s\min(d,c),\ M / \mathrm{memory\_BW}\right).
$$

This is the current `CostModel::seconds` model, not measured runtime.
`service` exposes each resource's service time; `prices` derives marginal costs
used by planning decisions. Row gather/admission and process costs must match the
actual routes and schema behavior.

`to_bytes` / `from_bytes` serialize the binary cost model. Optional `serde`
enables Serde integration for applications that manage structured calibration
records. The Rust core does not automatically discover or calibrate your hardware.

Use [cache planning](planning.md) to apply a cost model. The
[opt reference](reference-opt.md) owns exact fields and serialization contracts;
[Python cost tools](/python/api/statistics.html) provide optional collection and
calibration workflows.

API: [RunStats](/generated-loader-struct-RunStats),
[SiteMask](/generated-prof-struct-SiteMask),
[ProfileReport](/generated-prof-struct-ProfileReport),
[CostModel](/generated-opt-struct-CostModel).
