"""Check release archives and write a checksum manifest without uploading them."""

import argparse
import hashlib
import re
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def metadata(blob, version):
    value = BytesParser(policy=policy.default).parsebytes(blob)
    if value['Name'] != 'pycellpack' or value['Version'] != version:
        raise ValueError('Python archive metadata has the wrong name or version')
    expected = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['requires-python']
    if {x.strip() for x in str(value['Requires-Python']).split(',')} != {x.strip() for x in expected.split(',')}:
        raise ValueError('Python archive has the wrong supported Python versions')
    return value


def verify(root, complete=False, allow_native=False):
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    rust = tomllib.loads((ROOT / 'Cargo.toml').read_text())['workspace']['package']['version']
    wheels, sdists, crates = [], [], []
    forbidden = {'node_modules', '_build', '.venv', '__pycache__', '.env', 'target'}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    for path in paths:
        if path.suffix == '.whl':
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                info = [n for n in names if n.endswith('.dist-info/METADATA')]
                if len(info) != 1:
                    raise ValueError('Wheel must have one METADATA file')
                metadata(archive.read(info[0]), project['version'])
                if not any(re.fullmatch(r'pycellpack/_cellpack[^/]*\.so', n) for n in names):
                    raise ValueError('Wheel is missing the native extension')
                for required in ['pycellpack/py.typed', 'pycellpack/_cellpack.pyi']:
                    if required not in names:
                        raise ValueError(f'Wheel is missing {required}')
            if not allow_native and 'manylinux_2_28_' not in path.name:
                raise ValueError('Release wheels must use the manylinux_2_28 baseline')
            wheels.append(path)
        elif path.name.endswith('.tar.gz'):
            with tarfile.open(path) as archive:
                names = archive.getnames()
                for name in names:
                    if forbidden.intersection(Path(name).parts) or name.endswith(('.so', '.pyd')):
                        raise ValueError(f'Build/cache content in sdist: {name}')
                infos = [n for n in names if n.endswith('/PKG-INFO')]
                metadata(archive.extractfile(infos[0]).read(), project['version'])
                prefix = names[0].split('/')[0] + '/'
                for required in ['pyproject.toml', 'Cargo.toml', 'crates/cellpack/Cargo.toml',
                                 'crates/pycellpack/Cargo.toml', 'src/pycellpack/__init__.py', 'LICENSE']:
                    if prefix + required not in names:
                        raise ValueError(f'Sdist is missing {required}')
            sdists.append(path)
        elif path.suffix == '.crate':
            with tarfile.open(path) as archive:
                prefix = f'cellpack-{rust}/'
                names = archive.getnames()
                for required in ['Cargo.toml', 'README.md', 'LICENSE', 'src/lib.rs']:
                    if prefix + required not in names:
                        raise ValueError(f'Rust package is missing {required}')
                data = tomllib.loads(archive.extractfile(prefix + 'Cargo.toml').read().decode())
                if data['package']['name'] != 'cellpack' or data['package']['version'] != rust:
                    raise ValueError('Rust archive has the wrong name or version')
            crates.append(path)
        else:
            raise ValueError(f'Unexpected release file: {path.name}')
    if not paths:
        raise ValueError('No release files found')
    if complete:
        expected = {(f'cp3{n}', arch) for n in range(10, 16) for arch in ['x86_64', 'aarch64']}
        actual = set()
        for path in wheels:
            parts = path.name.split('-')
            abi, py, platform = parts[-2], parts[-3], parts[-1]
            if abi != py:
                raise ValueError('Only standard CPython wheels are in the release matrix')
            arch = 'aarch64' if 'aarch64' in platform else 'x86_64'
            actual.add((py, arch))
        if actual != expected or len(wheels) != len(expected) or len(sdists) != 1 or len(crates) != 1:
            raise ValueError(f'Incomplete release: {len(wheels)} wheels, {len(sdists)} sdist, {len(crates)} crate')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--allow-native', action='store_true')
    parser.add_argument('--write-checksums', action='store_true')
    args = parser.parse_args()
    try:
        paths = verify(args.directory, args.complete, args.allow_native)
    except (ValueError, KeyError, IndexError) as error:
        raise SystemExit(str(error)) from error
    if args.write_checksums:
        lines = [hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.relative_to(args.directory).as_posix()
                 for path in paths]
        (args.directory / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(f'Verified {len(paths)} release archives')


if __name__ == '__main__':
    main()
