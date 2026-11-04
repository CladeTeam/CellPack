# Process settings: runtime

`cellpack::runtime` binds CPU kernels once per process and reports physical-core
topology. Use it before constructing codec tables or a loader when your
application must control kernel selection. Worker count and CPU placement belong
to [`LoaderConfig`](reference-loader.md), independently of this binding.

<DocChoices title="Choose how to initialize" :items="[
  { label: 'Environment-based binding', when: 'The executable uses the default runtime.', detail: 'Set CELLPACK_KERNELS before launching it. The first get or codec context binds the process.', href: '#environment-and-errors' },
  { label: 'Explicit binding', when: 'Your application owns startup configuration.', detail: 'Call init before other CellPack work. Keep one options value for the process lifetime.', href: '#bind-explicitly-at-startup' },
]" />

## Bind explicitly at startup

This complete program caps kernels at AVX2 and prints the actual selected ISA.
Run it in a fresh process, before other CellPack initialization:

```rust
use cellpack::runtime::{self, KernelCap, RuntimeOptions};

fn main() -> cellpack::Result<()> {
    let options = RuntimeOptions { kernels: KernelCap::parse("avx2")? };
    let bound = runtime::init(options)?;
    println!("requested: {:?}; selected: {}", bound.options().kernels, bound.isa());
    assert!(std::ptr::eq(bound, runtime::init(options)?));
    Ok(())
}
```

The selected name can be `avx2` or `scalar`, depending on CPU support. A cap is
an upper bound, not a requirement to execute unsupported instructions.
`KernelCap::Auto` chooses the best supported set; `KernelCap::At(isa)` limits it.

The first successful `init` or `get` fixes the process binding. Repeating `init`
with **equal options** returns the same `&'static Runtime`. Different options
return `Error::InvalidOptions` and leave the binding intact, even if both options
would resolve to the same ISA. For example, `Auto` and `At(Isa::Avx512)` are
different requests. Use a separate process to compare caps.

## Environment and errors

`CELLPACK_KERNELS` accepts the case-sensitive names `scalar`, `avx2`, and `avx512`,
with surrounding whitespace ignored. An absent variable means `Auto`; the literal
string `auto` is not accepted.

| Operation | When the environment is read | Failure / effect |
|---|---|---|
| `RuntimeOptions::from_env()` | Every call | Returns options without binding; invalid or non-Unicode values return `InvalidOptions` |
| `get()` | Only if no runtime is bound | Parses the environment and binds; parsing failure leaves it unbound |
| `init(options)` | Never | Binds supplied options, or rejects a conflict with the existing binding |

After binding, changing the environment has no effect on `get` or existing
contexts. Decode/encode tables, mappers and loader conversion contexts retain
the selected kernel pointers. Initialize explicitly before constructing them if
you need to own the decision.

## Observe the selected kernels

| Interface | Input | Output / behavior |
|---|---|---|
| `Runtime::options()` | Bound runtime | Requested options, not resolved capability |
| `Runtime::isa()` | Bound runtime | Actual selected `scans::Isa` |
| `scans::Isa::supported()` | Candidate ISA | CPU support, independent of the process cap |
| `scans::kernel_set()` | None | Selected name; may trigger first binding; returns `"unbound"` if binding fails |

Use `runtime::get()?` to preserve initialization errors. Calling `kernel_set()`
just to inspect a name can itself fix the process binding.

## Core topology

`physical_cores(&cpus)` counts distinct package/core pairs from Linux sysfs,
deduplicating SMT siblings. Pass logical CPU IDs, or an empty slice to inspect
the **calling thread's affinity set**. The result is at least one.

This query reads topology on every call and does not initialize the kernel
binding. If a CPU's topology files are unavailable, its logical ID supplies a
fallback identity. If affinity cannot be read, the fallback uses the first
`available_parallelism()` logical IDs.

An explicit list is not validated for online status or affinity membership.
The count does not reserve cores, pin threads, or set `LoaderConfig.workers`.
Use `LoaderConfig.cpus` for worker placement; worker `i` is pinned to
`cpus[i % cpus.len()]` when that list is nonempty.

## Complete interface index

<!--@include: ./_catalog/runtime.md-->
