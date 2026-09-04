"""Measure prefill and decode separately against an already-running server.

Splitting the two is the point: the expert-kernel work changed decode only, and
a server response also pays prefill, which still runs the untouched `_rows`
kernels. A decode win can be real and invisible in total wall time.

Counts `reasoning_content` deltas as well as `content` -- this checkpoint emits
its thinking through the former, and a reader that only looks at `content` sees
a stream with no tokens in it at all.

Prefers the server's own `flyweight.decode_elapsed_seconds` over inter-arrival
timing, since that excludes the HTTP/SSE overhead between us and the runtime.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

FILLER = (
    "Memory hierarchy design balances latency, capacity, and cost across "
    "registers, cache, main memory, and secondary storage. Locality of "
    "reference is what makes the hierarchy effective. "
)


def stream_once(base: str, prompt: str, max_tokens: int):
    body = json.dumps({
        "model": "local", "stream": True, "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    request = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    last = started
    tokens = 0
    server_tokens = 0
    server_decode = 0.0
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content")
            meta = chunk.get("flyweight") or {}
            if meta.get("generated_tokens"):
                server_tokens = int(meta["generated_tokens"])
                server_decode = float(meta.get("decode_elapsed_seconds", 0.0))
            if not piece:
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            tokens += 1
            last = now
    if first is None:
        raise SystemExit("no tokens streamed")
    return {
        "ttft": first - started,
        "wall_decode": last - first,
        "tokens": tokens,
        "server_tokens": server_tokens,
        "server_decode": server_decode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--prompt-repeats", type=int, default=90)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    long_prompt = FILLER * args.prompt_repeats + "\n\nSummarize the above."
    short_prompt = "Count from one to twenty in words."
    stream_once(base, short_prompt, 32)  # warm

    print(f"\n=== {args.label}")
    for name, prompt in (("~2k prompt", long_prompt),
                         ("short prompt", short_prompt)):
        rows = [stream_once(base, prompt, args.tokens)
                for _ in range(args.repeats)]
        ttft = statistics.median(r["ttft"] for r in rows)
        rate = statistics.median(
            (r["server_tokens"] / r["server_decode"]) if r["server_decode"]
            else (r["tokens"] / r["wall_decode"]) for r in rows)
        total = statistics.median(r["ttft"] + r["wall_decode"] for r in rows)
        print(f"  {name:<13} ttft {ttft:6.2f}s   decode {rate:6.2f} tok/s   "
              f"total {total:6.2f}s")


if __name__ == "__main__":
    main()
