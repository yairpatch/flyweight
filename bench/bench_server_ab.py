"""End-to-end A/B through the real server, split into prefill and decode.

The kernel work measured steady-state decode with the prompt already warm. A
server request is not that: it pays prefill first, and the batched prefill path
still runs the untouched `_rows` kernels. So a decode-only win can be real and
still invisible in a response's wall time.

This starts the server with a given flag set, streams a completion, and reports
the two halves separately -- time to first token (prefill) and the inter-token
rate after it (decode). Run it against two builds of flyweight_v2.so and the
difference lands in whichever half actually changed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL = "/home/yair/Downloads/gguf/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
PORT = 8137  # not 8080, so a server the user left running cannot be hit instead
BASE = f"http://127.0.0.1:{PORT}"

FILLER = (
    "Memory hierarchy design balances latency, capacity, and cost across "
    "registers, cache, main memory, and secondary storage. Locality of "
    "reference is what makes the hierarchy effective. "
)


def wait_ready(proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    raise SystemExit("server did not become ready")


def stream_once(prompt: str, max_tokens: int) -> tuple[float, float, int]:
    """Returns (time to first token, decode seconds, tokens after the first)."""
    body = json.dumps({
        "model": "local", "stream": True, "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first: float | None = None
    last = started
    tokens = 0
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError):
                continue
            if not delta.get("content"):
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            else:
                tokens += 1
            last = now
    if first is None:
        raise SystemExit("no tokens streamed")
    return first - started, last - first, tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--prompt-repeats", type=int, default=90,
                        help="~2k tokens at the default, so prefill is visible")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    command = [
        sys.executable, "-m", "flyweight.cli", "serve-v2", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT), "--backend", "cuda",
        "--max-new-tokens", "28264", "--context", "5000",
        "--gpu-cache-mib", "8500", "--moe-device=hybrid",
    ]
    env = dict(os.environ, PYTHONPATH="/home/yair/flyweight/src")
    proc = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        wait_ready(proc, timeout=1800)
        long_prompt = FILLER * args.prompt_repeats + "\n\nSummarize the above."
        short_prompt = "Count from one to twenty in words."
        # Warm: clocks ramp and the expert cache fills on the first request.
        stream_once(short_prompt, 32)
        rows = []
        for _ in range(args.repeats):
            rows.append(stream_once(long_prompt, args.tokens))
        short = [stream_once(short_prompt, args.tokens)
                 for _ in range(args.repeats)]
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    def report(name: str, data: list[tuple[float, float, int]]) -> None:
        ttft = statistics.median(r[0] for r in data)
        rate = statistics.median(r[2] / r[1] if r[1] > 0 else 0.0 for r in data)
        total = statistics.median(r[0] + r[1] for r in data)
        print(f"  {name:<14} ttft {ttft:6.2f}s   decode {rate:6.2f} tok/s   "
              f"total {total:6.2f}s")

    print(f"\n=== {args.label}")
    report("~2k prompt", rows)
    report("short prompt", short)


if __name__ == "__main__":
    main()
