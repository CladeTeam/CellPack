"""Prepare immutable compiler inputs on an authorized build worker."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

DOCS = Path(__file__).resolve().parents[1]
STATE = DOCS / "_build/playground"
sys.path.insert(0, str(DOCS))
from api_content import EXAMPLES


def main():
    env = {**os.environ, "RUSTUP_TOOLCHAIN": "1.96.0", "CARGO_NET_OFFLINE": "true"}
    target = DOCS / "_build/cargo"
    build = subprocess.run([
        "cargo", "build", "--locked", "--offline", "--manifest-path",
        str(DOCS / "examples/Cargo.toml"), "--target-dir", str(target), "--bin", "quickstart",
        "--message-format=json",
    ], check=True, env=env, stdout=subprocess.PIPE, text=True)
    artifacts = [json.loads(line) for line in build.stdout.splitlines()]
    STATE.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "cargo", "vendor", "--locked", "--offline", "--respect-source-config",
        "--manifest-path", str(DOCS / "examples/Cargo.toml"), str(STATE / "vendor"),
    ], check=True, env=env, stdout=subprocess.DEVNULL)
    (STATE / "cargo-config.toml").write_text(
        '[source.crates-io]\nreplace-with = "vendored-sources"\n'
        '[source.vendored-sources]\ndirectory = "/vendor"\n'
    )
    deps = STATE / "deps"
    if deps.exists():
        shutil.rmtree(deps)
    deps.mkdir(exist_ok=True)
    library = None
    for artifact in artifacts:
        if artifact.get("reason") != "compiler-artifact":
            continue
        for filename in artifact.get("filenames", []):
            source = Path(filename)
            if source.suffix in {".rlib", ".so"}:
                shutil.copy2(source, deps / source.name)
                if artifact["target"]["name"] == "cellpack" and source.suffix == ".rlib":
                    library = source
    if library is None:
        raise RuntimeError("Cargo did not report a cellpack library artifact")
    toolchain = subprocess.check_output(["rustup", "which", "--toolchain", "1.96.0", "rustc"], text=True).strip()
    if not (Path(toolchain).parent.parent / "lib/rustlib/src/rust/library/core/src/lib.rs").is_file():
        raise RuntimeError("Install the rust-src component for Rust 1.96.0 before preparing analysis")
    analyzer = subprocess.check_output(["rustup", "which", "rust-analyzer"], text=True).strip()
    shutil.copy2(DOCS / "examples/Cargo.lock", STATE / "Cargo.lock")
    shutil.copy2(DOCS / "examples/common.rs", STATE / "common.rs")
    examples = {}
    binaries = STATE / "bin"
    binaries.mkdir(exist_ok=True)
    sources = STATE / "examples"
    sources.mkdir(exist_ok=True)
    shutil.copy2(DOCS / "examples/common.rs", sources / "common.rs")
    for name, description in EXAMPLES.items():
        code = (DOCS / "api-examples" / (name + ".rs")).read_text()
        main = sources / (name + ".rs")
        main.write_text(code)
        subprocess.run([toolchain, "--edition", "2024", "-C", "strip=debuginfo",
            "--extern", "cellpack=" + str(deps / library.name), "-L", "dependency=" + str(deps),
            str(main), "-o", str(binaries / name)], check=True, env=env)
        examples[name] = {"code": code, "expected": description["expected"]}
    (STATE / "examples.json").write_text(json.dumps(examples) + "\n")
    shutil.copy2(analyzer, STATE / "rust-analyzer")
    manifest = {
        "toolchain": "Rust 1.96.0", "toolchain_path": str(Path(toolchain).parent.parent),
        "analyzer_toolchain_path": str(Path(analyzer).parent.parent),
        "library": library.name,
        "build": hashlib.sha256(library.read_bytes()).hexdigest()[:16],
    }
    (STATE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
