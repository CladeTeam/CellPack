# Releases

CellPack is distributed as the [`cellpack` Rust crate](https://crates.io/crates/cellpack)
and the [`pycellpack` Python package](https://pypi.org/project/pycellpack/).
Release notes and downloadable archives are available on
[GitHub Releases](https://github.com/CladeTeam/CellPack/releases).

## Install a release

For a Python environment:

```sh
python -m pip install pycellpack==0.1.0
```

For AnnData and H5CP support:

```sh
python -m pip install 'pycellpack[anndata]==0.1.0'
```

For a Rust project:

```sh
cargo add cellpack@0.1.0
```

The Python distribution is named `pycellpack`; the Rust crate is named
`cellpack`. Both use the same release version.

## Platform and dependency support

- The storage engine and native loader target Linux.
- Python wheels are available for CPython 3.10 through 3.15 on x86_64 and
  aarch64. They use the `manylinux_2_28` baseline and require glibc 2.28 or newer.
- AnnData integration requires Python 3.12 or newer.
- The Rust crate declares Rust 1.89 as its minimum supported compiler version.
- PyTorch and other optional frameworks must be installed separately. Install
  builds that match your platform and, when applicable, your CUDA environment.

See the [installation guide](https://cladeteam.github.io/CellPack/installation)
and [Python documentation](https://cladeteam.github.io/CellPack/python/)
for setup and usage examples.

## Release contents

Each release provides:

- Twelve Python wheels: six CPython versions for each supported architecture.
- A Python source distribution for building from source.
- A Rust `.crate` source archive.
- A `SHA256SUMS` manifest covering those archives.

GitHub also provides source snapshots for the release tag. These snapshots
are separate from the Python and Rust package archives.

To verify all package archives downloaded from a release, place them in the
layout used by the checksum manifest:

```sh
mkdir -p python rust
mv pycellpack-*.whl pycellpack-*.tar.gz python/
mv cellpack-*.crate rust/
sha256sum --check SHA256SUMS
```

## Package and format versions

Release tags use `vX.Y.Z`; package versions use `X.Y.Z`. Prereleases use
Rust versions such as `X.Y.Z-alpha.N`, `X.Y.Z-beta.N`, or `X.Y.Z-rc.N`, with
corresponding Python versions `X.Y.ZaN`, `X.Y.ZbN`, or `X.Y.ZrcN`.

Package versions and file-format versions are independent. Release `0.1.0`
uses version 1 for the CPK and CPAK layouts, plan files, shared-ring layout,
cost records, and callback ABIs. The custom H5CP matrix encoding is `0.1.0`.
Standard AnnData encodings retain their own specification versions.

Consult the release notes for compatibility changes before upgrading.

## Validation scope

Release validation covers Rust tests on x86_64 and aarch64, the minimum
supported Rust compiler, Python tests on multiple interpreter versions,
PyTorch CPU integration, wheel installation, source-distribution rebuilds,
and archive metadata and checksums. GPU and CUDA execution are outside the
scope of the GitHub CPU runner checks.

For usage questions or reproducible problems, open an
[issue](https://github.com/CladeTeam/CellPack/issues) and include the package
version, Python or Rust version, operating system, and relevant error output.
