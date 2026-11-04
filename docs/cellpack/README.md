# CellPack documentation

The English website has a project home and two documentation surfaces:
`/rust/` for CellPack and `/python/` for PyCellPack. Each has its own home,
navigation, documentation and API reference; Python also has training examples.

From the repository root, on an authorized build machine:

```sh
uv sync --locked --project docs/cellpack
npm ci --prefix docs/cellpack
RUSTUP_TOOLCHAIN=1.96.0 uv run --locked --project docs/cellpack \
  python docs/cellpack/build.py --check-examples
```

The static website is in `_build/html`. See [documentation development](contributing.md)
for live editing, URL prefixes, and Read the Docs deployment.

## Website structure

- `.vitepress/routes.mjs` maps existing Rust sources into `/rust/docs/` and
  `/rust/api/`; old URLs retain compatibility pages and client-side forwarding.
- `.vitepress/theme/site/` supplies the shared subsite layout, language badge,
  breadcrumbs and simple subsite homes. The project homepage and its animations
  remain in their existing components.
- Language badges include locally bundled logos. The leftmost Home icon returns
  to the project; the product name returns to the current subsite. Sidebar labels
  identify DOCS, API and TRAINING; API pages use their own reference header and accent.
- `python/` contains authored Python entry pages and the training-example index.
  `python_docs.py` calls `python_reference.py` to extract current public signatures, fields and docstrings
  into ignored build inputs. Guides and API parameter tables are authored separately
  under `python/docs/` and `python/api/`; do not replace tutorials with declaration dumps.
- `reference.py` continues to generate Rust declarations and API metadata.
- Searches inside a subsite are scoped to that surface. The project-home search
  covers both; result headings identify Python or Rust.

After changing the Python reference source, run `python_docs.py` or the normal
preparation workflow before rendering the site. Do not edit generated catalog files.

## Maintaining Python documentation

- Put step-by-step workflows, complete inputs, and expected results in `python/docs/`.
- Put parameter meanings, returns, errors and lifetime rules in `python/api/`.
- `python_reference.py` explicitly groups public declarations by topic and fails if a
  declared symbol disappears. Update its groups when adding or moving public APIs.
- Regenerate without compiling Rust or importing optional packages:

  ```sh
  uv run --no-sync python docs/cellpack/python_docs.py
  ```

- Render prepared content with `npm run build --prefix docs/cellpack`. VitePress
  checks internal page links. Run executable guide blocks in an isolated writable
  directory with the relevant optional packages installed. CUDA and distributed
  fragments require their stated runtime setup.
