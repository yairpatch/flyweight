#!/usr/bin/env python3
"""Measure whether decode throughput decays with position inside one generation.

Prints tok/s per bucket of generated tokens so an O(context) attention ramp
(gentle, smooth) can be told apart from a leak or cache thrash (a cliff, or a
decay that keeps going after the KV growth alone can explain it).
"""
from __future__ import annotations

import sys
import time

from colibri_next.v2 import V2Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/yair/Downloads/Ling-3.0-tiny-Q6_K.gguf"
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
BUCKET = 64
CONTEXT = 8192
WARM = 128


def run(runtime, prompt, count):
    stamps: list[float] = []
    runtime.generate(prompt, count, lambda t: stamps.append(time.perf_counter()))
    return stamps


def report(label, stamps, start):
    print(f"\n== {label} ==")
    previous = start
    for base in range(0, len(stamps) - BUCKET + 1, BUCKET):
        end = stamps[base + BUCKET - 1]
        span = end - previous
        previous = end
        print(f"  tokens {base:5d}-{base + BUCKET:5d}  {BUCKET / span:7.2f} tok/s"
              f"  {span / BUCKET * 1000:7.2f} ms/tok")


def main() -> None:
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize("Write a long, detailed essay about memory bandwidth.\n\n")
        with model.native_runtime(context_limit=CONTEXT) as runtime:
            runtime.prepare()
            runtime.generate(prompt, WARM, lambda t: None)  # clock ramp + JIT
            runtime.reset()

            start = time.perf_counter()
            first = run(runtime, prompt, TOKENS)
            report("pass 1 (fresh cache)", first, start)

            # Same work again after a reset: if pass 2 is slower at matched
            # positions, the decay is state that reset() does not clear.
            runtime.reset()
            start = time.perf_counter()
            second = run(runtime, prompt, TOKENS)
            report("pass 2 (after reset)", second, start)

            runtime.reset()
            start = time.perf_counter()
            third = run(runtime, prompt, TOKENS)
            report("pass 3 (after reset)", third, start)
    finally:
        model.close()


if __name__ == "__main__":
    main()
