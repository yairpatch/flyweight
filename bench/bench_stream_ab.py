"""Prefill and decode of any OpenAI-compatible server, measured from the client.

Streams one raw completion per repeat and reports time to first token
(prefill) and the inter-token rate after it (decode). Works against flyweight
and llama-server alike, which is the point: the same prompt, the same client,
two engines. The prompt is filler text repeated to roughly --tokens tokens with
a per-repeat note number at the START so no engine serves it from a prompt
cache, and it ends mid-sentence so a greedy continuation does not stop at
once. llama.cpp's server-side `timings` are recorded when present.

    python bench/bench_stream_ab.py --base http://127.0.0.1:8201 --label llamacpp \\
        --tokens 2048 --repeats 5 --out results.jsonl

bench/llamacpp-comparison-2026-09-18.md is a run of this against both engines.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

FILLER = (
    "Memory hierarchy design balances latency, capacity, and cost across "
    "registers, cache, main memory, and secondary storage. Locality of "
    "reference is what makes the hierarchy effective. Caches exploit temporal "
    "and spatial locality; virtual memory extends the illusion of a large flat "
    "address space over a small physical one. "
)

# Filler copies per requested token count, calibrated on the Qwen tokenizer.
TOKENS_PER_COPY = 62


def stream(base: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": "m",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = last = None
    chunks = 0
    usage = timings = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                text = choice.get("text") or (choice.get("delta") or {}).get("content")
                if not text:
                    continue
                now = time.perf_counter()
                first = first if first is not None else now
                last = now
                chunks += 1
    result: dict = {"chunks": chunks, "usage": usage, "timings": timings}
    if first is not None:
        result["client_ttft_s"] = round(first - started, 3)
        if usage:
            result["client_prefill_tps"] = round(
                usage.get("prompt_tokens", 0) / (first - started), 1
            )
    if chunks > 1 and last is not None and first is not None and last > first:
        result["client_decode_tps"] = round((chunks - 1) / (last - first), 2)
    if timings:  # llama.cpp reports its own split
        result["server_prefill_tps"] = round(
            timings["prompt_n"] / (timings["prompt_ms"] / 1000), 1
        )
        result["server_decode_tps"] = round(
            timings["predicted_n"] / (timings["predicted_ms"] / 1000), 2
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--note-offset", type=int, default=0,
                        help="distinct numbers for a warm-up run so it shares no prefix")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    copies = max(1, args.tokens // TOKENS_PER_COPY)
    with open(args.out, "a") as out:
        for repeat in range(args.repeats):
            note = repeat + 1 + args.note_offset
            prompt = (
                f"Study note {note}. " + FILLER * copies
                + "In short, the main point of the note is that"
            )
            result = stream(args.base, prompt, args.max_tokens)
            result["label"] = args.label
            result["repeat"] = repeat + 1
            out.write(json.dumps(result) + "\n")
            out.flush()
            shown = {
                key: result.get(key)
                for key in (
                    "label", "repeat", "client_ttft_s", "client_prefill_tps",
                    "client_decode_tps", "server_prefill_tps", "server_decode_tps",
                )
            }
            print(json.dumps(shown), flush=True)


if __name__ == "__main__":
    main()
