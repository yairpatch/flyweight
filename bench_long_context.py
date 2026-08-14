#!/usr/bin/env python3
"""Prefill and decode rate as a function of real context length.

The earlier position sweep stopped at 4096 tokens and found decode flat. A
coding agent runs at 40k+, where the KV is an order of magnitude larger, so
this walks out to the sizes those sessions actually reach and reports prefill
and decode separately.
"""
from __future__ import annotations

import subprocess
import sys
import time

from colibri_next.v2 import V2Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q6_K.gguf"
POSITIONS = [1024, 4096, 8192, 16384, 32768, 49152]
DECODE = 48
CONTEXT = 65536

FILLER = (
    "The runtime streams expert weights from host memory over PCIe, so the "
    "decode cost is dominated by how much of the routed expert set already "
    "sits resident in device memory when the token arrives. "
)


def clocks() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    sm, temp, watt = (part.strip() for part in out.split(","))
    return f"sm={sm}MHz {temp}C {watt}W"


def main() -> None:
    model = V2Model(MODEL)
    try:
        unit = model.tokenize(FILLER)
        base = model.tokenize("Summarize the following notes.\n\n")

        def prompt_of(length: int) -> list[int]:
            out = list(base)
            while len(out) < length:
                out.extend(unit)
            return out[:length]

        with model.native_runtime(context_limit=CONTEXT) as runtime:
            runtime.prepare()
            runtime.generate(base, 64, lambda t: None)  # settle clocks
            print(f"{'context':>8} {'prefill t/s':>12} {'decode t/s':>11} "
                  f"{'ms/tok':>8} {'miss%':>7}   gpu")
            for position in POSITIONS:
                prompt = prompt_of(position)
                runtime.reset()
                before = dict(runtime.info)
                stamps: list[float] = []
                started = time.perf_counter()
                runtime.generate(prompt, DECODE + 1,
                                 lambda t: stamps.append(time.perf_counter()))
                prefill_span = stamps[0] - started
                decode_span = stamps[-1] - stamps[0]
                after = runtime.info
                lookups = (
                    (after["expert_cache_hits"] - before["expert_cache_hits"])
                    + (after["expert_cache_misses"] - before["expert_cache_misses"])
                )
                misses = after["expert_cache_misses"] - before["expert_cache_misses"]
                print(
                    f"{position:8d} {position / prefill_span:12.1f} "
                    f"{DECODE / decode_span:11.2f} "
                    f"{decode_span / DECODE * 1000:8.2f} "
                    f"{100.0 * misses / lookups if lookups else 0.0:7.1f}   "
                    f"{clocks()}",
                    flush=True,
                )
    finally:
        model.close()


if __name__ == "__main__":
    main()
