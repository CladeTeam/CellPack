"""Isolated Rust examples and rust-analyzer for the API manual."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

from aiohttp import WSMsgType, web

DOCS = Path(__file__).resolve().parents[1]
STATE = DOCS / "_build/playground"
MAX_CODE = 32_768
MAX_OUTPUT = 65_536
EXAMPLES = {"write", "selection", "loading"}


def kill(process):
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


class Playground:
    def __init__(self, state: Path, origins: set[str]):
        self.state = state.resolve()
        self.manifest = json.loads((state / "manifest.json").read_text())
        self.examples = json.loads((state / "examples.json").read_text())
        self.origins = origins
        self.run_slots = asyncio.Semaphore(2)
        self.lsp_slots = asyncio.Semaphore(2)
        self.processes = set()
        self.run_timeout = 50
        self.bwrap = os.environ.get("CELLPACK_BWRAP") or shutil.which("bwrap")
        if not self.bwrap or not shutil.which("prlimit"):
            raise RuntimeError("bubblewrap and prlimit are required; no host execution fallback exists")
        import subprocess
        help_text = subprocess.check_output([self.bwrap, "--help"], text=True)
        if "--size" not in help_text:
            raise RuntimeError("bubblewrap with --size support is required; set CELLPACK_BWRAP to a current executable")

    def command(self, directory: Path, *command: str) -> list[str]:
        args = [
            "prlimit", "--as=4294967296", "--cpu=45", "--nproc=4096", "--fsize=67108864", "--nofile=256", "--",
            self.bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--dev", "/dev", "--size", "134217728", "--tmpfs", "/tmp",
            "--size", "536870912", "--tmpfs", "/work", "--dir", "/work/src",
            "--ro-bind", str(directory / "src"), "/work/src",
            "--ro-bind", str(directory / "Cargo.toml"), "/work/Cargo.toml",
            "--ro-bind", str(self.state / "Cargo.lock"), "/work/Cargo.lock",
            "--ro-bind", str(self.state / "deps"), "/deps",
            "--ro-bind", str(self.state / "bin"), "/examples",
            "--ro-bind", self.manifest["toolchain_path"], "/toolchain",
            "--ro-bind", self.manifest["analyzer_toolchain_path"], "/analyzer-toolchain",
            "--ro-bind", str(self.state / "rust-analyzer"), "/rust-analyzer",
            "--ro-bind", str(self.state / "vendor"), "/vendor",
            "--dir", "/cargo", "--ro-bind", str(self.state / "cargo-config.toml"), "/cargo/config.toml",
            "--clearenv", "--setenv", "HOME", "/tmp",
            "--setenv", "PATH", "/toolchain/bin:/usr/bin", "--setenv", "TMPDIR", "/tmp",
            "--setenv", "LD_LIBRARY_PATH", "/toolchain/lib:/analyzer-toolchain/lib",
            "--setenv", "CARGO_HOME", "/cargo", "--setenv", "CARGO_TARGET_DIR", "/work/target",
            "--setenv", "CARGO_NET_OFFLINE", "true", "--setenv", "RAYON_NUM_THREADS", "2",
            "--setenv", "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER", "/usr/bin/gcc",
        ]
        args += ["--chdir", "/work", "--", *command]
        return args

    def workspace(self, directory: Path, code: str):
        (directory / "src").mkdir()
        (directory / "src/main.rs").write_text(code)
        shutil.copy2(self.state / "common.rs", directory / "src/common.rs")
        (directory / "Cargo.toml").write_text(
            '[package]\nname = "cellpack-guide-examples"\nversion = "0.1.0"\nedition = "2024"\n'
            '[dependencies]\ncellpack = { version = "0.1.0", features = ["serde"] }\n'
        )

    async def spawn(self, directory: Path, *command: str):
        process = await asyncio.create_subprocess_exec(*self.command(directory, *command),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True)
        self.processes.add(process)
        return process

    async def collect(self, stream, process):
        buffer = bytearray()
        exceeded = False
        while chunk := await stream.read(4096):
            if exceeded:
                continue
            buffer.extend(chunk)
            if len(buffer) > MAX_OUTPUT:
                exceeded = True
                del buffer[MAX_OUTPUT:]
                kill(process)
        return buffer.decode(errors="replace") + ("\nOutput limit exceeded." if exceeded else "")

    async def health(self, request):
        return web.json_response({"ready": True, "analysis": True, **{k: self.manifest[k] for k in ("toolchain", "build")}})

    async def run(self, request):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise web.HTTPBadRequest(text='{"error":"Expected a JSON request."}', content_type="application/json")
        if not isinstance(payload, dict) or set(payload) != {"example"} or not isinstance(payload["example"], str) or payload["example"] not in EXAMPLES:
            return web.json_response({"error": "Only a fixed example ID is accepted. Source code and execution arguments are not allowed."}, status=400)
        name = payload["example"]
        if self.run_slots.locked():
            return web.json_response({"error": "All execution slots are busy. Try again shortly."}, status=429)
        async with self.run_slots:
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="run-", dir=self.state) as temporary:
                directory = Path(temporary)
                self.workspace(directory, self.examples[name]["code"])
                process = await self.spawn(directory, "/examples/" + name, "/work/output")
                process.stdin.close()
                timed_out = False
                try:
                    stdout_task = asyncio.create_task(self.collect(process.stdout, process))
                    stderr_task = asyncio.create_task(self.collect(process.stderr, process))
                    try:
                        await asyncio.wait_for(process.wait(), timeout=self.run_timeout)
                    except asyncio.TimeoutError:
                        timed_out = True
                        kill(process)
                        await process.wait()
                    stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
                    phase = "run"
                    if timed_out:
                        stderr += "\nExecution time limit exceeded."
                    return web.json_response({"success": process.returncode == 0, "phase": phase,
                        "stdout": stdout, "stderr": stderr, "exit_code": process.returncode,
                        "elapsed_ms": round((time.monotonic() - started) * 1000), "toolchain": self.manifest["toolchain"]})
                finally:
                    kill(process)
                    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                    await process.wait()
                    self.processes.discard(process)

    async def lsp(self, request):
        name = request.query.get("example")
        filename = request.query.get("file", "main.rs")
        if name not in EXAMPLES or filename not in {"main.rs", "common.rs"}:
            return web.json_response({"error": "Choose a fixed example and one of its files."}, status=400)
        uri = "file:///work/src/" + filename
        document = self.examples[name]["code"] if filename == "main.rs" else (self.state / "common.rs").read_text()
        if self.lsp_slots.locked():
            return web.json_response({"error": "All analysis slots are busy."}, status=429)
        async with self.lsp_slots:
            ws = web.WebSocketResponse(max_msg_size=MAX_CODE * 3, receive_timeout=120, heartbeat=30)
            await ws.prepare(request)
            with tempfile.TemporaryDirectory(prefix="lsp-", dir=self.state) as temporary:
                directory = Path(temporary)
                self.workspace(directory, self.examples[name]["code"])
                process = await self.spawn(directory, "/rust-analyzer")
                definition_requests = set()

                async def output():
                    while True:
                        size = None
                        while line := await process.stdout.readline():
                            if line == b"\r\n":
                                break
                            if line.lower().startswith(b"content-length:"):
                                size = int(line.split(b":", 1)[1])
                        if not line or size is None or size > 2_000_000:
                            break
                        body = await process.stdout.readexactly(size)
                        response = json.loads(body)
                        if response.get("id") in definition_requests:
                            definition_requests.discard(response["id"])
                            locations = response.get("result") or []
                            if isinstance(locations, dict):
                                locations = [locations]
                            response["result"] = [location for location in locations if location.get("uri", location.get("targetUri")) == uri]
                        await ws.send_json(response)
                    await ws.close()

                async def input_messages():
                    async for message in ws:
                        if message.type != WSMsgType.TEXT:
                            continue
                        payload = json.loads(message.data)
                        method = payload.get("method", "")
                        if method not in {"", "initialize", "initialized", "shutdown", "exit",
                            "textDocument/didOpen", "textDocument/hover", "textDocument/definition"}:
                            if "id" in payload:
                                await ws.send_json({"jsonrpc": "2.0", "id": payload["id"], "error": {"code": -32601, "message": "Read-only analysis: method not allowed."}})
                            else:
                                await ws.close(code=1008, message=b"Read-only analysis")
                            continue
                        if method == "initialize":
                            payload["params"]["rootUri"] = "file:///work"
                            payload["params"]["rootPath"] = "/work"
                            payload["params"]["processId"] = None
                            payload["params"]["workspaceFolders"] = [{"uri": "file:///work", "name": name}]
                            payload["params"]["initializationOptions"] = {
                                "numThreads": 2, "cachePriming": {"enable": False, "numThreads": 2},
                                "checkOnSave": False,
                                "cargo": {"buildScripts": {"enable": False}}, "procMacro": {"enable": False},
                                "check": {"allTargets": False},
                            }
                        params = payload.get("params", {})
                        if method.startswith("textDocument/") and params.get("textDocument", {}).get("uri") != uri:
                            continue
                        if method == "textDocument/didOpen":
                            params["textDocument"]["text"] = document
                        if method == "textDocument/definition":
                            definition_requests.add(payload["id"])
                        body = json.dumps(payload).encode()
                        process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
                        await process.stdin.drain()

                tasks = [asyncio.create_task(output()), asyncio.create_task(input_messages()),
                         asyncio.create_task(self.collect(process.stderr, process))]
                try:
                    done, _ = await asyncio.wait(tasks[:2], timeout=300, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                except (ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, json.JSONDecodeError):
                    pass
                finally:
                    kill(process)
                    process.stdin.close()
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await process.communicate()
                    self.processes.discard(process)
                    await ws.close()
            return ws

    async def cleanup(self, app):
        for process in list(self.processes):
            kill(process)
        await asyncio.gather(*(process.wait() for process in self.processes))

    async def startup(self, app):
        with tempfile.TemporaryDirectory(prefix="probe-", dir=self.state) as temporary:
            directory = Path(temporary)
            self.workspace(directory, "fn main() {}\n")
            process = await self.spawn(directory, "/bin/sh", "-c", "/toolchain/bin/rustc --version && /rust-analyzer --version")
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
                if process.returncode != 0 or b"1.96.0" not in stdout:
                    raise RuntimeError("Sandbox probe failed: " + stderr.decode(errors="replace"))
            finally:
                kill(process)
                await process.wait()
                self.processes.discard(process)


SERVICE = web.AppKey("playground", Playground)


def create_app(state=STATE, origins=(), site=None, base="/"):
    service = Playground(Path(state), set(origins))

    @web.middleware
    async def origin_check(request, handler):
        origin = request.headers.get("Origin")
        if origin and origin not in service.origins:
            return web.json_response({"error": "Origin is not allowed."}, status=403)
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)
        if not site or request.path.startswith(base + "playground/"):
            response.headers["Cache-Control"] = "no-store"
        if origin:
            response.headers.update({"Access-Control-Allow-Origin": origin, "Vary": "Origin",
                "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "GET, POST, OPTIONS"})
        return response

    app = web.Application(client_max_size=MAX_CODE * 3, middlewares=[origin_check])
    app[SERVICE] = service
    if not base.startswith("/") or not base.endswith("/"):
        raise ValueError("Site base must start and end with /")
    prefix = base + "playground" if site else ""
    app.router.add_get(prefix + "/health", service.health)
    app.router.add_post(prefix + "/run", service.run)
    app.router.add_get(prefix + "/lsp", service.lsp)
    async def options(request):
        return web.Response()
    app.router.add_route("OPTIONS", "/{path:.*}", options)
    if site:
        root = Path(site).resolve()
        async def static_page(request):
            relative = request.match_info["path"] or "index.html"
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise web.HTTPNotFound()
            if path.is_dir():
                path = path / "index.html"
            elif not path.suffix:
                path = path.with_suffix(".html")
            if not path.is_file():
                raise web.HTTPNotFound()
            return web.FileResponse(path)
        app.router.add_get(base + "{path:.*}", static_page)
    app.on_cleanup.append(service.cleanup)
    app.on_startup.append(service.startup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--origin", action="append", default=[])
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--site", type=Path, help="serve a built website with the fixed-example service")
    parser.add_argument("--base", default="/", help="website URL prefix when --site is used")
    args = parser.parse_args()
    app = create_app(args.state, args.origin, args.site, args.base)
    web.run_app(app, host=args.host, port=args.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
