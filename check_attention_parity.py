#!/usr/bin/env python3
"""Greedy-decode parity for every 256-dim attention kernel against the reference.

The split-K and grouped-query kernels compute attention in a different
arithmetic order from the serial ring kernels (online softmax merged across
tiles), so identical tokens out is the check that matters. A kernel that is
fast because it quietly changed the distribution is not a faster kernel.

Reference is the serial ring path, which is the simplest implementation and the
one the others are derived from.
"""
from __future__ import annotations

import os
import sys

from flyweight.v2 import V2Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q6_K.gguf"
POSITIONS = (2048, 16384, 49152)
TOKENS = 40
FILLER = (
    "Attention over a growing key-value cache is the part of decode that scales "
    "with context, and the kernel that reads it decides how fast that grows. "
)

# name -> environment selecting exactly that kernel
PATHS = {
    "reference (serial ring)": {
        "FLYWEIGHT_GQA_ATTENTION": "0", "FLYWEIGHT_CUBLAS_ATTENTION": "0",
        "FLYWEIGHT_FUSED_ATTENTION": "0",
    },
    "split-K per head": {
        "FLYWEIGHT_GQA_ATTENTION": "0", "FLYWEIGHT_CUBLAS_ATTENTION": "0",
        "FLYWEIGHT_FUSED_ATTENTION": "1",
    },
    "grouped query": {
        "FLYWEIGHT_GQA_ATTENTION": "1", "FLYWEIGHT_CUBLAS_ATTENTION": "0",
        "FLYWEIGHT_FUSED_ATTENTION": "1",
    },
}


def decode(model, prompt, kv, environment):
    for key, value in environment.items():
        os.environ[key] = value
    with model.native_runtime(context_limit=65536,
                              cache_type_k=kv, cache_type_v=kv) as runtime:
        runtime.prepare()
        out: list[int] = []
        runtime.generate(prompt, TOKENS, lambda t: out.append(t))
        return out


def main() -> None:
    model = V2Model(MODEL)
    failures = 0
    try:
        unit = model.tokenize(FILLER)
        base = list(model.tokenize("Continue this text.\n\n"))
        for target in POSITIONS:
            prompt = list(base)
            while len(prompt) < target:
                prompt.extend(unit)
            prompt = prompt[:target]
            for kv in ("f16", "q8_0"):
                expected = decode(model, prompt, kv,
                                  PATHS["reference (serial ring)"])
                for name, environment in PATHS.items():
                    if name.startswith("reference"):
                        continue
                    actual = decode(model, prompt, kv, environment)
                    ok = actual == expected
                    failures += 0 if ok else 1
                    print(f"ctx={target:6d} kv={kv:5s} {name:18s}: "
                          f"{'identical' if ok else 'DIVERGES'}", flush=True)
                    if not ok:
                        print(f"   expected: {model.decode_tokens(expected)[:80]!r}")
                        print(f"   actual  : {model.decode_tokens(actual)[:80]!r}")
    finally:
        model.close()
    print("PARITY OK" if not failures else f"PARITY FAILED ({failures})")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
