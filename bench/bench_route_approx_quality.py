"""Quality of the opt-in routing approximations (FLYWEIGHT_ROUTE_PRIOR,
FLYWEIGHT_ROUTE_TAIL_SKIP) against exact routing.

Run once with no approximation to write the reference, then once per setting
(the env vars are read once per process):

    python bench/bench_route_approx_quality.py MODEL ref.json --write
    FLYWEIGHT_ROUTE_PRIOR=1.5 python bench/bench_route_approx_quality.py MODEL ref.json

For each prompt the reference is the exact greedy continuation. A run reports
teacher-forced top-1 agreement with it (each step is fed the reference token,
so one disagreement does not cascade) and where its own free greedy
continuation first departs. Hybrid decode is not bitwise reproducible (which
routes hit the GPU cache changes the arithmetic), so also run the exact
configuration a second time without --write: that is the noise floor.
"""
from __future__ import annotations

import json
import os
import sys

from flyweight.v2 import V2Model

PROMPTS = {
    "code": "Write a Python function that returns the nth Fibonacci number using memoization.",
    "math": "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Show your reasoning.",
    "spanish": "Explica brevemente en español por qué el cielo es azul.",
    "chinese": "用中文简要解释什么是机器学习。",
    "factual": "List the first five prime numbers and explain what makes a number prime.",
    "creative": "Write the opening paragraph of a mystery novel set in a lighthouse.",
    "json": "Return a JSON object describing a book with title, author, year and three tags.",
    "tool": "Write a bash one-liner that counts the lines of every .py file under src/ and prints the total.",
}
N = 96


def chat(model, text):
    return list(model.tokenize(
        "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))


def main() -> int:
    path, ref_path = sys.argv[1], sys.argv[2]
    write = "--write" in sys.argv
    model = V2Model(path)
    rt = model.native_runtime(context_limit=8192)
    rt.prepare()
    prompts = {name: chat(model, text) for name, text in PROMPTS.items()}

    def greedy(ids):
        out = []
        rt.reset()
        rt.generate(ids, N, out.append)
        return out

    if write:
        ref = {name: greedy(ids) for name, ids in prompts.items()}
        json.dump(ref, open(ref_path, "w"))
        print("reference written")
        return 0
    ref = json.load(open(ref_path))
    label = " ".join(f"{k}={v}" for k, v in os.environ.items()
                     if k.startswith("FLYWEIGHT_ROUTE")) or "exact (noise floor)"
    agrees, firsts = [], []
    for name, ids in prompts.items():
        reference = ref[name]
        rt.reset()
        for token in ids[:-1]:
            rt.decode(token)
        context, agree = ids[-1], 0
        for expected in reference:
            agree += int(rt.decode(context) == expected)
            context = expected
        gen = greedy(ids)
        first = next((i for i, (a, b) in enumerate(zip(gen, reference)) if a != b),
                     min(len(gen), len(reference)))
        agrees.append(agree / len(reference))
        firsts.append(first)
        print(f"  [{name:8s}] agree {agree / len(reference):6.1%}  first divergence {first:3d}/{N}",
              flush=True)
    print(f"{label}: mean agreement {sum(agrees) / len(agrees):.1%}, "
          f"min first divergence {min(firsts)}, median {sorted(firsts)[len(firsts) // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
