"""A small stand-in for the parts of the Hugging Face Hub that Flyweight uses.

Serves /api/models (search), /api/models/<repo> (info), the recursive file
tree with LFS sizes and SHA-256s, and /<repo>/resolve/main/<path> with HTTP
Range support. A file is either real bytes or a generated run of a given
size (for realistic sizes without real downloads). `delay` slows downloads
so progress can be watched.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubFile:
    def __init__(self, path: str, data: bytes | None = None, size: int = 0,
                 sha: str | None = None) -> None:
        self.path = path
        self.data = data
        self.size = len(data) if data is not None else size
        if sha is not None:  # for huge generated files nobody will finish
            self.sha = sha
        elif data is not None:
            self.sha = hashlib.sha256(data).hexdigest()
        else:  # generated content: a repeating block, hashed once
            hasher = hashlib.sha256()
            block = self.block()
            whole, rest = divmod(self.size, len(block))
            for _ in range(whole):
                hasher.update(block)
            hasher.update(block[:rest])
            self.sha = hashlib.sha256(b"").hexdigest() if self.size == 0 else hasher.hexdigest()

    @staticmethod
    def block() -> bytes:
        return (b"GGUF" + bytes(range(256)) * 4096)[: 1 << 20]

    def read(self, start: int, end: int) -> bytes:
        if self.data is not None:
            return self.data[start:end]
        block = self.block()
        out = bytearray()
        position = start
        while position < end:
            offset = position % len(block)
            take = min(len(block) - offset, end - position)
            out += block[offset:offset + take]
            position += take
        return bytes(out)


class HubStub:
    def __init__(self) -> None:
        self.repos: dict[str, dict] = {}
        self.delay = 0.0          # seconds per MiB served
        self.requests: list[str] = []
        self.server: ThreadingHTTPServer | None = None

    def add(self, repo: str, files: list[StubFile], architecture: str = "qwen3",
            downloads: int = 1000, gated: bool = False) -> None:
        self.repos[repo] = {"files": {f.path: f for f in files}, "architecture": architecture,
                            "downloads": downloads, "gated": gated}

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self) -> "HubStub":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def _json(self, data: object, status: int = 200) -> None:
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: C901 - a router
                stub.requests.append(self.path)
                parts = urllib.parse.urlsplit(self.path)
                path = urllib.parse.unquote(parts.path)
                query = urllib.parse.parse_qs(parts.query)
                if path == "/api/models":
                    needle = (query.get("search") or [""])[0].lower()
                    hits = [{"id": repo, "downloads": r["downloads"], "likes": 1,
                             "gated": r["gated"], "gguf": {"architecture": r["architecture"]}}
                            for repo, r in stub.repos.items() if needle in repo.lower()]
                    hits.sort(key=lambda h: -h["downloads"])
                    return self._json(hits)
                match = re.fullmatch(r"/api/models/([^/]+/[^/]+)(/tree/main)?", path)
                if match:
                    repo = stub.repos.get(match.group(1))
                    if repo is None:
                        return self._json({"error": "Repository not found"}, 404)
                    if match.group(2):
                        return self._json([
                            {"type": "file", "path": f.path, "size": f.size, "oid": "x",
                             "lfs": {"oid": f.sha, "size": f.size, "pointerSize": 134}}
                            for f in repo["files"].values()])
                    return self._json({"id": match.group(1), "gated": repo["gated"],
                                       "gguf": {"architecture": repo["architecture"]}})
                match = re.fullmatch(r"/([^/]+/[^/]+)/resolve/main/(.+)", path)
                if match and match.group(1) in stub.repos:
                    if stub.repos[match.group(1)]["gated"]:
                        return self._json({"error": "gated"}, 401)
                    f = stub.repos[match.group(1)]["files"].get(match.group(2))
                    if f is None:
                        return self._json({"error": "Entry not found"}, 404)
                    return self._send_file(f)
                self._json({"error": "not found"}, 404)

            def _send_file(self, f: StubFile) -> None:
                start, end = 0, f.size
                ranged = self.headers.get("Range", "")
                match = re.fullmatch(r"bytes=(\d+)-", ranged)
                if match:
                    start = int(match.group(1))
                    if start >= f.size:
                        self.send_response(416)
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end - 1}/{f.size}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(end - start))
                self.end_headers()
                position = start
                try:
                    while position < end:
                        chunk = f.read(position, min(end, position + (1 << 20)))
                        self.wfile.write(chunk)
                        position += len(chunk)
                        if stub.delay:
                            time.sleep(stub.delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
