"""Measure decode tok/s + per-token latency jitter, for COLIBRI_V2_MLOCK on/off.

Run in its own process per setting (env read at model open). Reports mean
tok/s plus per-token latency percentiles — mlock/MAP_POPULATE should shrink
the tail (p99/max) by removing cold-page faults even if the mean barely
moves on a warm page cache.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

from colibri_next.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
CONTEXT = 8192
WARM = 128
N = 512
PROMPT_TEXT = (
    "Explain how mixture-of-experts routing and expert weight streaming from "
    "system RAM affect decode latency, and what caching strategies help.\n\n"
)


def main() -> None:
    setting = os.environ.get("COLIBRI_V2_MLOCK", "0")
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        with model.native_qwen_runtime(
            context_limit=CONTEXT,
            gpu_cache_bytes=8192 * 1024**2,
            moe_device="hybrid",
            mtp_drafts=0,
        ) as runtime:
            runtime.prepare()
            runtime.generate(prompt, WARM, lambda t: None)  # warm
            runtime.reset()

            stamps: list[float] = []

            def on_token(_t: int) -> None:
                stamps.append(time.perf_counter())

            start = time.perf_counter()
            runtime.generate(prompt, N, on_token)
            wall = time.perf_counter() - start

        deltas = [
            (stamps[i] - (stamps[i - 1] if i else start)) * 1000.0
            for i in range(len(stamps))
        ]
        deltas_sorted = sorted(deltas)

        def pct(p: float) -> float:
            return deltas_sorted[min(len(deltas_sorted) - 1, int(p * len(deltas_sorted)))]

        print(f"COLIBRI_V2_MLOCK={setting}  tokens={len(stamps)}  "
              f"tok/s={len(stamps) / wall:.2f}")
        print(f"  per-token ms: mean={statistics.mean(deltas):.3f}  "
              f"p50={pct(0.50):.3f}  p95={pct(0.95):.3f}  "
              f"p99={pct(0.99):.3f}  max={max(deltas):.3f}  "
              f"stdev={statistics.pstdev(deltas):.3f}")
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
