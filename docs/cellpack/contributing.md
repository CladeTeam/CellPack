# Documentation development

## Structure

This website has an English Rust manual at `/rust/` and a Python manual at
`/python/`, under one CellPack project homepage.
VitePress 1.6.4 renders the manual and landing page. Its theme extension supplies
the homepage, typography, navigation and logo player, while VitePress supplies
local search, page outlines, syntax highlighting, and code copying.

Authored guides and module contracts are ordinary Markdown files in this
directory. `reference.py` derives the public interface pages and catalogs from
rustdoc. `examples/` is an independent Cargo package whose Rust sources are
included directly in guide pages. This package depends on the published
`cellpack` crate from crates.io; the site's API generator reads the repository
source. Keep the guide dependency aligned with the release documented by the site.

| File or directory | Purpose |
|---|---|
| `.vitepress/config.mjs` | Site navigation, search, deployment base and build paths |
| `package.json / package-lock.json` | Pinned VitePress frontend dependencies |
| `pyproject.toml / uv.lock` | Python tools for API generation |
| `build.py` | Rustdoc, API generation, example validation and site build |
| `reference.py` | Declarations, member indexes, parameters, returns and field documentation |
| `_catalog/` | Generated module catalogs and sidebar metadata |
| `generated-*.md` | Generated public-interface pages |
| `_build/rustdoc/` | Private compiler output used to extract API metadata |
| `_build/html/` | Complete static website |

The Python bindings are not installed or imported to build this manual.

The canonical brand asset is `assets/logo.svg` at the repository root. The build
copies it to the public directory for the navbar, homepage illustration, and
browser icon. Update that source asset instead of editing a generated copy.

`assets/cellpack-animation.svg` supplies the homepage animation. A small default-theme
extension in `.vitepress/theme/` adds its player and navbar hover feedback.
The player runs once, supports pause/replay, pauses outside the viewport or in
background tabs, and falls back to the static logo for reduced-motion preferences.

## Build

From the repository root, on a machine where compilation is allowed:

```sh
rustup toolchain install 1.96.0 --profile minimal
uv sync --locked --project docs/cellpack
npm ci --prefix docs/cellpack
RUSTUP_TOOLCHAIN=1.96.0 uv run --locked --project docs/cellpack \
  python docs/cellpack/build.py --check-examples
```

The build generates version-matched rustdoc, refreshes the public-interface
catalog, runs every complete guide program when requested, and builds the
website with VitePress. Missing document links fail the frontend build.
Rustdoc warnings also fail the build.

Open `docs/cellpack/_build/html/index.html` through a local web server.
For example:

```sh
uv run --locked --project docs/cellpack \
  python -m http.server 8000 --directory docs/cellpack/_build/html
```

When previewing on a remote compute node, forward that port through your SSH
connection. Follow your environment's proxy and compute-node rules for package
installation, compilation, and browser checks.

The API example service also uses the registry crate. Before its offline
preparation step, populate the Cargo cache with the locked guide dependencies:

```sh
cargo fetch --locked --manifest-path docs/cellpack/examples/Cargo.toml
```

Preparation vendors those dependencies for offline execution and language
analysis. The operator workflow is described in the
[example service guide](playground/README.md).

## Live editing

Prepare generated content, then start the VitePress development server:

```sh
RUSTUP_TOOLCHAIN=1.96.0 uv run --locked --project docs/cellpack \
  python docs/cellpack/build.py --prepare-only
npm run dev --prefix docs/cellpack
```

Markdown edits refresh automatically. After changing Rust public interfaces or
their documentation, run the preparation step again.
`--site-only` renders already prepared content; `--output` selects another HTML
directory; `--base /en/latest/` builds for a URL prefix.

## Authoring

Use ordinary Markdown links for manual pages. The API generator produces
the searchable interface pages and links directly between them. Compiler output
remains private to the build. Edit Rust documentation and module contracts rather
than generated files.

VitePress includes Rust source directly with `<<< @/examples/quickstart.rs`.

Use `<DocDiagram name="…" />` for authored workflow diagrams, identity maps,
matrix fixtures and decision cards. Their shared content is in
`.vitepress/theme/components/doc-diagrams.mjs`; `DocDiagram.vue` renders it with
theme colors and responsive layouts. Keep parameter tables, executable source
and actual terminal output in Markdown. Edit authored pages rather than generated
API declarations or `_catalog` files.

Use `DocChoices` when readers need to choose between interfaces. Put the task
in `when`, the API or path name in `label`, and the deciding constraint in
`detail`; an optional `href` links to its canonical documentation URL. It uses
the existing theme, supports keyboard focus, and stacks cards on small screens.
Keep the number of choices small enough to compare in one reading.

```vue
<DocChoices title="Choose a reading path" :items="[
  { label: 'Store', when: 'Read a selected matrix', detail: 'Decode on demand.', href: '/python/docs/storage.html' },
  { label: 'ScDataLoader', when: 'Iterate training batches', detail: 'Choose sample order and output columns.', href: '/python/docs/dataloader.html' },
]" />
```

Guides should take readers from a stated input to an observable result. API
pages should lead with interface choices and the contracts needed to call them,
then expose exact declarations under a clearly named disclosure. Keep essential
ownership and error rules visible. Use ordinary headings for subjects readers
need to link to; avoid hiding the entire tutorial in cards or disclosures.
Python module API outlines show authored level-two sections. Generated symbols
use level-three headings, with members at level four, inside the declaration
disclosure. This keeps the page outline focused on usage contracts while symbols
and members retain their direct anchors.

Module references include generated Markdown catalogs with the standard
VitePress Markdown-include directive. Keep source examples executable, small,
and independent of external datasets.

Rust sidebar API entries follow `_catalog/navigation.json`. Route namespaces,
compatibility links, and subsite counterparts are defined in `.vitepress/routes.mjs`.
Python contracts are authored in `python/api/`. `python_docs.py` calls
`python_reference.py` to extract declarations from current Python source and
`_cellpack.pyi`, without importing the extension or optional packages;
training pages include the maintained programs in `example/` directly.
The subsite layout, badge switcher, and breadcrumbs live in `.vitepress/theme/site/`.
Navigation and sidebars are configured in `.vitepress/config.mjs`. Theme components and styles live in
`.vitepress/theme/`. `HeroLogo.vue` owns the SVG timeline and accepts the page's
pause state so the homepage motion control also stops its animation.

## Read the Docs

Use `docs/cellpack/.readthedocs.yaml` as the project's build configuration.
It selects Python 3.12, Rust 1.96, and Node.js 24. Its custom build commands
bootstrap the uv tool, install the locked Python and Node dependencies, and
build into `$READTHEDOCS_OUTPUT/html`.

The deployment base defaults to the English Read the Docs version path.
Set `CELLPACK_DOCS_BASE` explicitly for a different mount point. Local builds
default to `/`.

1. Push the source, configuration, and Python/Node/Cargo lockfiles to your Git host.
2. Import the repository in Read the Docs.
3. Select `docs/cellpack/.readthedocs.yaml` as the build configuration.
4. Build the intended branch and review the generated site.
5. Activate the release versions you want to publish.

Rust and Python share this build and deployment base. The Python reference is
prepared from source text without importing optional Python integrations or
building another extension wheel. Each subsite keeps its own navigation and search
scope. Clicking its product name returns to its home; Project home returns to `/`.

Official references: [VitePress](https://vuejs.github.io/vitepress/v1/),
[Read the Docs custom builds](https://docs.readthedocs.com/platform/stable/build-customization.html),
and [multiple projects in one repository](https://docs.readthedocs.com/platform/stable/guides/setup/monorepo.html).

## Website performance

Keep large features behind their page or interaction boundary. API search metadata
and highlighted runnable examples have separate data loaders; complete signatures
and source locations load when a definition is opened. Successful details are
cached, failed loads can be retried, and closing the dialog cancels an in-flight
request. Shared example files and their highlighted HTML are stored once.
Ordinary API pages should not fetch the example bundle or CodeMirror. The PDF reader loads its viewer
styles when mounted and removes them on navigation. VitePress local search loads
its index when opened. Site metadata is shared in a cacheable module, and only the
two primary Latin text fonts are preloaded.

`prepare-logo.mjs` derives the website's logo resources from the canonical SVGs
under `assets/`. Numeric shortening preserves the parsed number sequence, and
structural simplification is disabled for style- or text-sensitive input. The
animated SVG runs inside a same-origin iframe with scripts disabled; the player
controls its native timeline and retains a poster during loading or failure.
This isolates SVG nodes from the main document's style and layout tree; it does
not eliminate the SVG nodes or their memory use in the child document.

Page prefetching follows pointer hover, keyboard focus or touch intent instead
of fetching every visible sidebar link. It respects data-saving connections and
uses the pinned VitePress version's route-to-chunk resolver, including deployment
base prefixes. Recheck that integration when upgrading VitePress.

Homepage canvas scenes allocate their backing stores near the viewport and
release them offscreen or in background tabs. Cache geometry outside animation
frames; update it after resizing, font loading and layout preferences change.
Scenes that only depend on scroll position declare `continuous: false`, allowing
them to stop drawing once their progress settles. Keep static layouts and reduced
motion free of continuous scene drawing. Collapsed API sidebar groups retain their
accessible control targets while creating descendants only when expanded.

Validate frontend changes against a fresh static build on an authorized worker,
using an isolated output directory through `CELLPACK_DOCS_OUTPUT`. Compare cold
loads with the same viewport, CPU throttling and server compression settings.
Record transferred JavaScript/CSS, canvas backing-store pixels, layout reads and
long tasks alongside timing measurements. Exercise scene deep links, sidebar
expansion, scoped search, API quick views and PDF navigation at desktop and mobile
sizes. Local browser timings describe that test environment; they are not
measurements of public hosting latency.

## Dependency updates

Update Python dependencies through `uv add --project docs/cellpack ...`.
Update frontend dependencies in this directory and commit `package-lock.json`.
The guide programs have their own `examples/Cargo.lock`.
After changing their `cellpack` dependency, update that lockfile and run the
programs against the registry package. Website builds also copy their Rust
sources to `/downloads/rust/`, so readers can run them in ordinary Cargo
applications without a CellPack checkout.
Rebuild and validate after dependency or toolchain changes.
