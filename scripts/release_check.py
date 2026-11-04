"""Validate package versions and the source ref before preparing a release."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check(require_tag=False):
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    workspace = tomllib.loads((ROOT / 'Cargo.toml').read_text())['workspace']['package']
    core = tomllib.loads((ROOT / 'crates/cellpack/Cargo.toml').read_text())['package']
    bindings = tomllib.loads((ROOT / 'crates/pycellpack/Cargo.toml').read_text())['package']
    version = workspace['version']
    match = re.fullmatch(r'(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.(\d+))?', version)
    if not match:
        raise ValueError('Rust version must be X.Y.Z or X.Y.Z-{alpha,beta,rc}.N')
    python_version = match[1]
    if match[2]:
        python_version += {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}[match[2]] + match[3]
    if project['name'] != 'pycellpack' or core['name'] != 'cellpack':
        raise ValueError('Release package names must be pycellpack and cellpack')
    if project['version'] != python_version:
        raise ValueError(f"Python version must be {python_version}, got {project['version']}")
    if core.get('publish') != ['crates-io']:
        raise ValueError('Only the core crate may publish to crates-io')
    if workspace.get('publish') is not False or bindings.get('publish') != {'workspace': True}:
        raise ValueError('Python binding crate must keep workspace publish=false')
    lock = tomllib.loads((ROOT / 'Cargo.lock').read_text())
    for package in lock['package']:
        if package['name'] in {'cellpack', 'pycellpack'} and package['version'] != version:
            raise ValueError('Refresh Cargo.lock after changing package versions')
    tag = 'v' + version
    ref = os.environ.get('GITHUB_REF', '')
    if require_tag or ref.startswith('refs/tags/'):
        if ref != 'refs/tags/' + tag:
            raise ValueError(f'Release must run on tag {tag}, not {ref or "an untagged checkout"}')
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        tagged = subprocess.check_output(['git', 'rev-parse', tag + '^{commit}'], cwd=ROOT, text=True).strip()
        if tagged != source:
            raise ValueError('Checked-out source does not match the release tag')
        subprocess.run(['git', 'merge-base', '--is-ancestor', source, 'origin/main'], cwd=ROOT, check=True)
    return {'version': version, 'python_version': python_version, 'tag': tag,
            'prerelease': str(bool(match[2])).lower()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tag', action='store_true')
    args = parser.parse_args()
    try:
        result = check(args.require_tag)
    except (ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result))
    if output := os.environ.get('GITHUB_OUTPUT'):
        with Path(output).open('a') as stream:
            stream.writelines(f'{key}={value}\n' for key, value in result.items())


if __name__ == '__main__':
    main()
