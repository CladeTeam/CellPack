# API example service

The static API reference includes highlighted, read-only examples. A separate
service runs fixed, prebuilt programs and provides read-only rust-analyzer hover
information. Visitors cannot submit code, change a document, add arguments, or
navigate definitions outside the selected file.

## Prepare on a build worker

Use an authorized Linux worker with Rust 1.96.0 and its `rust-src` component,
rust-analyzer, GCC, bubblewrap with `--size` support, and `prlimit`. Follow your
environment's compute and network rules.

```sh
uv sync --locked --project docs/cellpack
uv run --locked --project docs/cellpack python docs/cellpack/playground/prepare.py
```

Preparation uses the Cargo cache offline, vendors the locked dependencies, and
builds the three approved example binaries. It snapshots their source and fixture.
Re-run preparation after changing the library or examples. Stop the service before
replacing prepared inputs and deploy the matching website and prepared service
together. `CELLPACK_BWRAP` can select a bubblewrap executable outside `PATH`.

## Start and connect

To serve the built website and fixed examples together on one origin:

```sh
uv run --locked --project docs/cellpack python docs/cellpack/playground/server.py \
  --site docs/cellpack/_build/html --port 8000 --origin http://127.0.0.1:8000
```

Use `--base /en/latest/` when the website was built with that prefix. This combined
server handles the example endpoints itself; a plain static file server cannot
handle **Run example** or LSP requests.

To keep an existing web server and reverse proxy the example service separately:

```sh
uv run --locked --project docs/cellpack python docs/cellpack/playground/server.py \
  --origin https://docs.example.org
```

The service listens on `127.0.0.1:8091`. Reverse proxy the website's `/playground/`
prefix to it, removing that prefix and enabling WebSocket upgrades. With a site
base such as `/en/latest/`, proxy `/en/latest/playground/` instead. The frontend
defaults to this same-origin path. `CELLPACK_PLAYGROUND_URL` can select a different
endpoint at website build time. Supply `--origin` once per allowed browser origin,
including scheme and port.

For worker-local development:

```sh
CELLPACK_PLAYGROUND_PROXY=http://127.0.0.1:8091 \
  npm run dev --prefix docs/cellpack -- --port 5191
```

Use `--origin http://127.0.0.1:5191` on the service for this preview. Production
hosting needs its own reverse proxy; static VitePress output does not start the
Python service.

## Request contract

`POST /run` accepts exactly one field:

```json
{"example": "write"}
```

Allowed IDs are `write`, `selection`, and `loading`. Additional fields, source
code, paths, arguments, and unknown IDs are rejected. The request runs an approved
binary built during preparation; it never compiles submitted source.

`GET /lsp?example=write&file=main.rs` opens read-only analysis of a fixed example.
Only `main.rs` and `common.rs` are accepted. The server supplies the approved source
regardless of text sent by the client. Document changes and arbitrary file reads
are rejected. Definition results outside the selected file are filtered out on
the server, and the browser applies the same restriction. Hover text never opens
external links. Switching file tabs closes the previous analysis session.

`GET /health` reports readiness, the prepared library build ID, and toolchain.
The server probes isolated compiler/analyzer startup before accepting requests.

## Resource boundary

The example programs and analyzer run in bubblewrap with separate user, PID, and
network namespaces. The host home, datasets, and credentials are not mounted.
The sandbox sees the toolchain, system libraries, prepared binaries, public crate
sources, vendored dependencies, and fixed example files. Inputs are read-only.
Temporary storage uses bounded tmpfs: 128 MiB at `/tmp`, 512 MiB at `/work`.
`prlimit` bounds address space, CPU time, process count, file size, and descriptors.
There is no host execution fallback.

There are two run slots and two analysis slots. Runs have a 50-second wall limit
and a 64 KiB limit per output stream. Analysis expires after five minutes or two
idle minutes. Client disconnects cancel active runs and close analyzers. Use a
dedicated unprivileged service account and service-level cgroup limits, plus rate
limiting at the public reverse proxy.

## Maintain and validate

`api_content.py` contains explanations and expected outputs. `api-examples/*.rs`
contains the fixed programs; `examples/common.rs` contains the shared fixture.
`reference.py` generates declarations, type links, source excerpts, and the example
catalog. Shiki highlights the examples at build time, including without JavaScript.
CodeMirror supplies the optional read-only LSP viewer.

On the authorized worker, after preparation:

```sh
uv run --locked --project docs/cellpack python docs/cellpack/playground/test_server.py
```

`test_browser.py` exercises the browser with `CELLPACK_REVIEW_URL` pointing to a
preview and `CELLPACK_CHROMIUM` optionally selecting its Chromium executable.
