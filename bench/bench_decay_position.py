#!/usr/bin/env python3
"""Separate position-dependent decode cost from time-dependent (thermal) drift.

The naive "generate a long sequence and watch tok/s" benchmark confounds the
two: later tokens sit at a longer context AND later in the thermal ramp. This
sweeps context positions ascending, then descending, at a GPU already held at
thermal equilibrium.

  - A positional cost shows the same ms/tok at the same position in both
    sweeps (the curves overlay).
  - Thermal/time drift shows the two sweeps mirroring each other (ascending
    slopes up, descending also slopes up in wall-clock order, so they cross).
"""
from __future__ import annotations

import subprocess
import sys
import time

from flyweight.v2 import V2Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/yair/Downloads/Qwen3.6-27B-UD-IQ2_XXS.gguf"
POSITIONS = [128, 512, 1024, 2048, 4096]
DECODE = 64
SETTLE_SECONDS = 90.0
CONTEXT = 8192

FILLER = (
    "Memory bandwidth is the dominant cost of autoregressive decode because "
    "every generated token must stream the full weight matrix from device "
    "memory into the streaming multiprocessors before any arithmetic runs. "
)


def clocks() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    sm, mem, temp, watt = (part.strip() for part in out.split(","))
    return f"sm={sm} mem={mem} {temp}C {watt}W"


def measure(runtime, prompt: list[int]) -> float:
    """ms/token for steady decode at the context length `prompt` implies.

    Timed from the first generated token onward so prompt prefill is excluded.
    """
    stamps: list[float] = []
    runtime.reset()
    runtime.generate(prompt, DECODE + 1, lambda t: stamps.append(time.perf_counter()))
    return (stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000.0


def main() -> None:
    model = V2Model(MODEL)
    try:
        unit = model.tokenize(FILLER)
        base = model.tokenize("Explain memory bandwidth.\n\n")

        def prompt_of(length: int) -> list[int]:
            out = list(base)
            while len(out) < length:
                out.extend(unit)
            return out[:length]

        with model.native_runtime(context_limit=CONTEXT) as runtime:
            runtime.prepare()
            # Hold the GPU at thermal equilibrium BEFORE any measurement, so
            # every sample below is taken in the same thermal state.
            print(f"settling {SETTLE_SECONDS:.0f}s ... start {clocks()}", flush=True)
            deadline = time.perf_counter() + SETTLE_SECONDS
            while time.perf_counter() < deadline:
                runtime.reset()
                runtime.generate(base, 64, lambda t: None)
            print(f"settled: {clocks()}\n", flush=True)

            order = POSITIONS + POSITIONS[::-1]
            results: list[tuple[str, int, float, str]] = []
            for index, position in enumerate(order):
                sweep = "up  " if index < len(POSITIONS) else "down"
                ms = measure(runtime, prompt_of(position))
                results.append((sweep, position, ms, clocks()))
                print(f"  {sweep} pos={position:5d}  {ms:7.3f} ms/tok  "
                      f"{1000 / ms:6.2f} tok/s   [{results[-1][3]}]", flush=True)

            print("\n== matched positions, ascending vs descending ==")
            for position in POSITIONS:
                up = next(r[2] for r in results if r[0] == "up  " and r[1] == position)
                down = next(r[2] for r in results if r[0] == "down" and r[1] == position)
                print(f"  pos={position:5d}  up={up:7.3f}  down={down:7.3f}  "
                      f"delta={down - up:+7.3f} ms ({(down / up - 1) * 100:+5.1f}%)")
    finally:
        model.close()


if __name__ == "__main__":
    main()
