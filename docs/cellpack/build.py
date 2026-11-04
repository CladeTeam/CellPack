"""Build the CellPack manual and generated API reference."""

import argparse
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from reference import generate
from python_docs import prepare_python

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parents[1]
BUILD = DOCS / "_build"
TARGET = BUILD / "cargo"


def run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def build_api() -> None:
    env = os.environ.copy()
    env["RUSTDOCFLAGS"] = (env.get("RUSTDOCFLAGS", "") + " -D warnings").strip()
    run(
        "cargo", "doc", "--locked", "--no-deps", "--lib", "--all-features",
        "-p", "cellpack", "--target-dir", str(TARGET), env=env,
    )
    generated = BUILD / "rustdoc"
    if generated.exists():
        shutil.rmtree(generated)
    shutil.copytree(TARGET / "doc", generated)


def prepare() -> None:
    build_api()
    generate()
    prepare_python()


def prepare_static() -> None:
    public = DOCS / "public"
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "assets" / "logo.svg", public / "logo.svg")
    shutil.copy2(ROOT / "assets" / "cellpack-animation.svg", public / "cellpack-animation.svg")


def check_examples() -> None:
    manifest = DOCS / "examples" / "Cargo.toml"
    with manifest.open("rb") as source:
        examples = tomllib.load(source)["bin"]
    BUILD.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="examples-", dir=BUILD) as directory:
        for example in examples:
            output = Path(directory) / example["name"]
            output.mkdir()
            run(
                "cargo", "run", "--locked", "--manifest-path", str(manifest),
                "--target-dir", str(TARGET), "--bin", example["name"], "--", str(output),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true", help="generate API pages and public assets")
    stage.add_argument("--site-only", action="store_true", help="render previously prepared content")
    parser.add_argument("--check-examples", action="store_true", help="run every complete Rust guide example")
    parser.add_argument("--output", type=Path, default=BUILD / "html", help="HTML output directory")
    parser.add_argument("--base", help="deployment URL prefix, such as /en/latest/")
    args = parser.parse_args()
    if not args.site_only:
        prepare()
    prepare_static()
    if args.check_examples:
        check_examples()
    if not args.prepare_only:
        if not (DOCS / "node_modules" / "vitepress").is_dir():
            parser.error("install the documentation frontend with npm ci --prefix docs/cellpack")
        env = os.environ.copy()
        env["CELLPACK_DOCS_OUTPUT"] = str(args.output.resolve())
        if args.base:
            if not args.base.startswith("/") or not args.base.endswith("/"):
                parser.error("--base must start and end with /")
            env["CELLPACK_DOCS_BASE"] = args.base
        run("npm", "run", "build", "--prefix", str(DOCS), env=env)


if __name__ == "__main__":
    main()
