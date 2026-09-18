#!/usr/bin/env python3
"""Prefill/decode roofline probe for the DeepSeek-V4 runtime.

Reports tok/s for both phases plus the native phase breakdown, so a slow
number can be attributed to expert reads, attention or the head rather than
guessed at.

    PYTHONPATH=src python3 bench/bench_ds4_speed.py
"""

from __future__ import annotations

import os
import sys
import time

from flyweight.v2 import V2Model
from flyweight.deepseek4 import Deepseek4Runtime

MODEL = os.environ.get("FLYWEIGHT_MODEL", "")
PROMPT_TOKENS = int(os.environ.get("BENCH_PROMPT", "512"))
DECODE_TOKENS = int(os.environ.get("BENCH_DECODE", "32"))
CHUNK = int(os.environ.get("BENCH_CHUNK", "256"))
CONTEXT = int(os.environ.get("BENCH_CONTEXT", "8192"))
DEVICE = os.environ.get("BENCH_DEVICE", "0")


def ms(ns: int) -> float:
    return ns / 1e6


def delta(after: dict, before: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def main() -> int:
    if not MODEL:
        print("set FLYWEIGHT_MODEL to the first shard of a DeepSeek-V4 GGUF", file=sys.stderr)
        return 2
    model = V2Model(MODEL)
    print(f"model:   {os.path.basename(MODEL)}")
    print(f"backend: device={DEVICE}  context={CONTEXT}")

    opened = time.perf_counter()
    runtime = Deepseek4Runtime(model, CONTEXT)
    if DEVICE != "cpu":
        runtime.use_gpu(int(DEVICE))
    print(f"load:    {time.perf_counter() - opened:.1f} s to runtime ready")

    # A real prompt, repeated to length, so routing sees plausible text.
    seed = model.tokenize(
        "The history of computing is a history of moving data. "
        "Every generation rediscovers that arithmetic is cheap and memory is not. "
    )
    tokens = (list(seed) * (PROMPT_TOKENS // len(seed) + 1))[:PROMPT_TOKENS]

    # Warm: the first chunk faults the dense half in from disk, which is a
    # load cost, not a prefill cost.
    runtime.prefill(tokens[:CHUNK])
    runtime.reset()

    before = dict(runtime.info)
    started = time.perf_counter()
    for at in range(0, len(tokens) - 1, CHUNK):
        runtime.prefill(tokens[at:min(at + CHUNK, len(tokens) - 1)])
    prefill_wall = time.perf_counter() - started
    mid = dict(runtime.info)

    token = tokens[-1]
    started = time.perf_counter()
    for _ in range(DECODE_TOKENS):
        logits = runtime.forward(token)
        token = int(logits.argmax())
    decode_wall = time.perf_counter() - started
    after = dict(runtime.info)

    fed = len(tokens) - 1
    print()
    print(f"prefill: {fed} tokens in {prefill_wall:.2f} s "
          f"-> {fed / prefill_wall:.1f} tok/s")
    print(f"decode:  {DECODE_TOKENS} tokens in {decode_wall:.2f} s "
          f"-> {DECODE_TOKENS / decode_wall:.2f} tok/s "
          f"({decode_wall / DECODE_TOKENS * 1e3:.0f} ms/token)")

    for name, lo, hi, steps in (("PREFILL", before, mid, fed),
                                ("DECODE", mid, after, DECODE_TOKENS)):
        print(f"\n--- {name} phase breakdown (per step) ---")
        for key in ("routed_expert_nanoseconds", "shared_expert_nanoseconds",
                    "attention_nanoseconds", "attention_core_nanoseconds",
                    "hyper_nanoseconds", "matvec_nanoseconds",
                    "head_nanoseconds"):
            value = delta(hi, lo, key)
            if value:
                print(f"  {key[:-12]:24s} {ms(value) / steps:8.2f} ms")
        bytes_read = delta(hi, lo, "routed_expert_bytes")
        seconds = (prefill_wall if name == "PREFILL" else decode_wall)
        if bytes_read:
            print(f"  {'routed expert bytes':24s} "
                  f"{bytes_read / steps / 2**20:8.1f} MiB/step "
                  f"({bytes_read / seconds / 2**30:.2f} GiB/s)")
        hits = delta(hi, lo, "expert_cache_hits")
        misses = delta(hi, lo, "expert_cache_misses")
        if hits + misses:
            print(f"  {'expert cache':24s} "
                  f"{100 * hits / (hits + misses):8.1f}% hit "
                  f"({hits} hit / {misses} miss)")

    print(f"\ngpu weight bytes: {int(after.get('gpu_weight_bytes', 0)) / 2**30:.2f} GiB")
    print(f"expert cache:     {int(after.get('expert_cache_bytes', 0)) / 2**30:.2f} GiB "
          f"in {int(after.get('expert_cache_slots', 0))} slots")

    runtime.close()
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
