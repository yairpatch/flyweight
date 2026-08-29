#!/usr/bin/env python3
"""Attribute the decode slowdown at long context to a runtime phase.

Decode falls ~23% between 2k and 47k of context. This splits a token at three
context lengths into expert paging, route wait (GPU), expert compute (CPU) and
the remainder, so the decline can be charged to attention growth or to the
expert cache outgrowing its capacity rather than guessed at.
"""
from __future__ import annotations

import sys
import time

from flyweight.v2 import V2Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q6_K.gguf"
POSITIONS = [2048, 16384, 49152]
KV = __import__("os").environ.get("KV", "f16")
DECODE = 64

FILLER = (
    "The runtime streams expert weights from host memory over PCIe, so the decode "
    "cost is dominated by how much of the routed expert set already sits resident. "
)
FIELDS = (
    "expert_cache_hits", "expert_cache_misses", "expert_page_nanoseconds",
    "route_wait_nanoseconds", "expert_compute_nanoseconds",
)


def main() -> None:
    model = V2Model(MODEL)
    try:
        unit = model.tokenize(FILLER)
        base = model.tokenize("Summarize.\n\n")
        print(f"cache_type_k/v = {KV}")
        with model.native_runtime(context_limit=65536, cache_type_k=KV, cache_type_v=KV) as runtime:
            runtime.prepare()
            runtime.generate(base, 64, lambda t: None)
            print(f"{'ctx':>7} {'tok/s':>7} {'ms/tok':>7} {'miss%':>6} "
                  f"{'page':>7} {'route':>7} {'xcomp':>7} {'other':>7}")
            for target in POSITIONS:
                prompt = list(base)
                while len(prompt) < target:
                    prompt.extend(unit)
                prompt = prompt[:target]
                runtime.reset()
                stamps: list[float] = []
                # Snapshot at the FIRST generated token, not before generate():
                # these counters also accumulate over prefill, and 49k tokens of
                # it dwarfs the 64-token decode window being measured.
                before: dict[str, int] = {}

                def on_token(_token: int) -> None:
                    stamps.append(time.perf_counter())
                    if len(stamps) == 1:
                        before.update(
                            {key: int(runtime.info[key]) for key in FIELDS}
                        )

                runtime.generate(prompt, DECODE + 1, on_token)
                after = {key: int(runtime.info[key]) for key in FIELDS}
                delta = {key: after[key] - before[key] for key in FIELDS}
                span = stamps[-1] - stamps[0]
                per = DECODE * 1e6
                lookups = (
                    delta["expert_cache_hits"] + delta["expert_cache_misses"]
                )
                total = span / DECODE * 1000
                page = delta["expert_page_nanoseconds"] / per
                route = delta["route_wait_nanoseconds"] / per
                compute = delta["expert_compute_nanoseconds"] / per
                print(
                    f"{target:7d} {DECODE / span:7.2f} {total:7.2f} "
                    f"{100 * delta['expert_cache_misses'] / lookups if lookups else 0:6.1f} "
                    f"{page:7.2f} {route:7.2f} {compute:7.2f} "
                    f"{total - page - route - compute:7.2f}",
                    flush=True,
                )
    finally:
        model.close()


if __name__ == "__main__":
    main()
