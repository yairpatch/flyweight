#!/usr/bin/env python3
"""Separate thermal throttling from software decay.

Runs identical short bursts separated by idle cooldowns. Software decay (a
leak, cache thrash, growing bookkeeping) cannot be undone by sitting idle, so
if every burst after a cooldown returns to the same speed the decay is thermal.
"""
from __future__ import annotations

import subprocess
import sys
import time

from colibri_next.v2 import V2Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/yair/Downloads/Qwen3.6-27B-UD-IQ2_XXS.gguf"
BURST = 96
COOLDOWN = 75.0
ROUNDS = 4


def gpu() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    sm, mem, temp, watt = (part.strip() for part in out.split(","))
    return f"sm={sm}MHz mem={mem}MHz {temp}C {watt}W"


def main() -> None:
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize("Write a long, detailed essay about memory bandwidth.\n\n")
        with model.native_runtime(context_limit=8192) as runtime:
            runtime.prepare()
            runtime.generate(prompt, 32, lambda t: None)
            for round_index in range(ROUNDS):
                runtime.reset()
                before = gpu()
                started = time.perf_counter()
                count = 0

                def tally(_token: int) -> None:
                    nonlocal count
                    count += 1

                runtime.generate(prompt, BURST, tally)
                span = time.perf_counter() - started
                print(f"burst {round_index}: {count / span:6.2f} tok/s   "
                      f"pre[{before}]  post[{gpu()}]", flush=True)
                if round_index != ROUNDS - 1:
                    time.sleep(COOLDOWN)
    finally:
        model.close()


if __name__ == "__main__":
    main()
