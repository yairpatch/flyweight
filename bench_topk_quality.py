"""Expert top-k quality robustness across diverse prompts.

For each prompt, builds the full-model greedy reference, then for each pruned
k measures teacher-forced top-1 agreement vs that reference (drift-free) and
the free-greedy first-divergence point, plus a text preview. Runtimes are
created once per k and reused across prompts (weights uploaded once).
"""
from __future__ import annotations

import sys

from colibri_next.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
CONTEXT = 8192
N = 64
PROMPTS = {
    "code": "Write a Python function that returns the nth Fibonacci number using memoization:\n",
    "math": "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Show your reasoning.\n",
    "spanish": "Explica brevemente en español por qué el cielo es azul.\n",
    "chinese": "用中文简要解释什么是机器学习。\n",
    "factual": "List the first five prime numbers and explain what makes a number prime.\n",
    "creative": "Write the opening paragraph of a mystery novel set in a lighthouse.\n",
}
LADDER = [7, 6, 5, 4]


def make_runtime(model, k):
    rt = model.native_qwen_runtime(
        context_limit=CONTEXT, gpu_cache_bytes=8192 * 1024**2,
        moe_device="hybrid", expert_top_k=k,
    )
    rt.prepare()
    return rt


def greedy(rt, prompt, n):
    out = []
    rt.reset()
    rt.generate(prompt, n, lambda t: out.append(t))
    return out


def agreement(rt, prompt, reference):
    rt.reset()
    for t in prompt[:-1]:
        rt.decode(t)
    context, agree = prompt[-1], 0
    for ref in reference:
        agree += int(rt.decode(context) == ref)
        context = ref
    return agree / max(1, len(reference))


def main() -> None:
    model = V2Model(MODEL)
    try:
        tokens = {name: model.tokenize(text) for name, text in PROMPTS.items()}
        full = int(model.config["expert_used_count"])
        print(f"top_k={full}, experts={model.config['expert_count']}, "
              f"gen={N}\n", flush=True)

        ref_rt = make_runtime(model, 0)
        refs = {name: greedy(ref_rt, tokens[name], N) for name in PROMPTS}
        ref_rt.close()
        for name in PROMPTS:
            print(f"[{name}] full: {model.decode_tokens(refs[name])[:110]!r}")
        print(flush=True)

        results: dict[int, dict[str, tuple[float, int]]] = {}
        for k in LADDER:
            rt = make_runtime(model, k)
            results[k] = {}
            print(f"===== expert_top_k={k} =====", flush=True)
            for name in PROMPTS:
                agr = agreement(rt, tokens[name], refs[name])
                gen = greedy(rt, tokens[name], N)
                first_div = next(
                    (i for i, (a, b) in enumerate(zip(gen, refs[name])) if a != b),
                    min(len(gen), len(refs[name])),
                )
                results[k][name] = (agr, first_div)
                print(f"  [{name}] agree={agr:5.1%}  first_div={first_div:2d}/{N}"
                      f"  {model.decode_tokens(gen)[:80]!r}", flush=True)
            rt.close()

        print("\n=== top-1 agreement matrix (rows=k, cols=prompt) ===")
        print("k   " + "  ".join(f"{n[:5]:>7}" for n in PROMPTS) + "   mean")
        for k in LADDER:
            vals = [results[k][n][0] for n in PROMPTS]
            row = "  ".join(f"{v:6.1%}" for v in vals)
            print(f"{k}   {row}   {sum(vals) / len(vals):5.1%}")
        print("\n=== free-greedy first-divergence (higher=better) ===")
        for k in LADDER:
            divs = [results[k][n][1] for n in PROMPTS]
            worst = min(divs)
            print(f"k={k}: min_first_div={worst}/{N}  "
                  f"{'OK' if worst >= 8 else 'DERAILS EARLY'}  "
                  + " ".join(f"{n[:4]}={results[k][n][1]}" for n in PROMPTS))
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
