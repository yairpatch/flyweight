#!/usr/bin/env python3
"""Attribute decode-throughput variance to runtime phases, per bucket.

Samples the runtime's own counters (expert cache hits/misses, expert paging
time, route wait, expert compute) at bucket boundaries during one long
generation, so a slowdown can be charged to a phase instead of guessed at.
"""
from __future__ import annotations

import subprocess
import os
import sys
import time

from colibri_next.v2 import V2Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q6_K.gguf"
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
BUCKET = 128
CONTEXT = 8192

FIELDS = (
    "expert_cache_hits", "expert_cache_misses", "expert_page_nanoseconds",
    "route_wait_nanoseconds", "expert_compute_nanoseconds", "decode_nanoseconds",
    "decode_calls", "cpu_prefetch_nanoseconds",
)


def gpu() -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    sm, util = (int(p.strip()) for p in out.split(","))
    return sm, util


def main() -> None:
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize("Write a long, detailed essay about memory bandwidth.\n\n")
        cache = int(os.environ.get("CACHE_MIB", "0")) * 1024**2
        mode = os.environ.get("EXPERT_MODE", "auto")
        residency = os.environ.get("RESIDENCY") or None
        print(f"expert_mode={mode} expert_residency={residency}")
        with model.native_runtime(context_limit=CONTEXT, gpu_cache_bytes=cache,
                                  expert_mode=mode, expert_residency=residency) as runtime:
            runtime.prepare()
            runtime.generate(prompt, 128, lambda t: None)
            runtime.reset()

            samples: list[tuple] = []
            produced = 0
            started = time.perf_counter()
            mark = [started, {f: 0 for f in FIELDS}]

            def on_token(_token: int) -> None:
                nonlocal produced
                produced += 1
                if produced % BUCKET:
                    return
                now = time.perf_counter()
                info = runtime.info
                current = {f: int(info.get(f, 0)) for f in FIELDS}
                span = now - mark[0]
                delta = {f: current[f] - mark[1][f] for f in FIELDS}
                sm, util = gpu()
                samples.append((produced, span, delta, sm, util))
                mark[0], mark[1] = time.perf_counter(), current

            runtime.generate(prompt, TOKENS, on_token)

            print(f"\n{'tokens':>7} {'tok/s':>7} {'miss%':>6} {'page':>7} {'route':>7} "
                  f"{'xcomp':>7} {'prefch':>7} {'sm':>5} {'util':>5}   (ms/tok by phase)")
            for produced, span, delta, sm, util in samples:
                lookups = delta["expert_cache_hits"] + delta["expert_cache_misses"]
                miss = 100.0 * delta["expert_cache_misses"] / lookups if lookups else 0.0
                per = BUCKET * 1e6  # ns -> ms per token
                print(f"{produced:7d} {BUCKET / span:7.2f} {miss:6.1f} "
                      f"{delta['expert_page_nanoseconds'] / per:7.2f} "
                      f"{delta['route_wait_nanoseconds'] / per:7.2f} "
                      f"{delta['expert_compute_nanoseconds'] / per:7.2f} "
                      f"{delta['cpu_prefetch_nanoseconds'] / per:7.2f} "
                      f"{sm:5d} {util:5d}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
