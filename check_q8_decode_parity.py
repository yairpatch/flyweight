#!/usr/bin/env python3
"""Greedy-token parity for the Q8-activation group-decode matvecs.

COLIBRI_IQ2_Q8_DECODE=0 forces every dense projection back onto the
per-element reference kernels, so running the same prompt both ways checks the
DP4A paths against the decode they replace.
"""

import argparse
import os
import subprocess
import sys


def sample(model: str, enabled: str, count: int) -> list[int]:
    source = (
        "import os,sys;"
        "from colibri_next.v2 import V2Model,V2QwenRuntime;"
        f"m=V2Model({model!r});"
        "r=V2QwenRuntime(m,context_limit=2048);r.prepare();"
        f"p=m.tokenize('Explain how a turbocharger works, step by step.');"
        "out=[];"
        f"r.generate(p,{count},lambda t: out.append(t) or True);"
        "print(','.join(map(str,out)))"
    )
    environment = dict(os.environ, COLIBRI_IQ2_Q8_DECODE=enabled, PYTHONPATH="src")
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True,
        env=environment, check=True,
    )
    return [int(v) for v in result.stdout.strip().splitlines()[-1].split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=48)
    arguments = parser.parse_args()

    reference = sample(arguments.model, "0", arguments.tokens)
    fast = sample(arguments.model, "1", arguments.tokens)
    agree = sum(a == b for a, b in zip(reference, fast))
    print(f"reference: {reference[:16]}")
    print(f"dp4a     : {fast[:16]}")
    if reference == fast:
        print(f"OK: {len(reference)}/{len(reference)} greedy tokens identical")
        return 0
    first = next(i for i, (a, b) in enumerate(zip(reference, fast)) if a != b)
    print(f"MISMATCH: {agree}/{len(reference)} agree, first divergence at {first}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
