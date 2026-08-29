"""Expert top-p (nucleus) routing: adaptive experts/token vs speed vs quality.

p=1.0/0.0 disables (== full, must be identical). For each p we report the
measured average experts/token (route_expert_sum telemetry), decode tok/s,
and teacher-forced top-1 agreement vs the full-model trajectory. A static
top-k=7 row is included so the dynamic vs static frontier is visible.
"""
from __future__ import annotations

import sys
import time

from flyweight.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
CONTEXT = 8192
N = 96
PROMPTS = {
    "code": "Write a Python function that returns the nth Fibonacci number using memoization:\n",
    "spanish": "Explica brevemente en español por qué el cielo es azul.\n",
    "factual": "List the first five prime numbers and explain what makes a number prime.\n",
    "creative": "Write the opening paragraph of a mystery novel set in a lighthouse.\n",
}


def make(model, *, top_k=0, top_p=0.0):
    rt = model.native_qwen_runtime(
        context_limit=CONTEXT, gpu_cache_bytes=8192 * 1024**2,
        moe_device="hybrid", expert_top_k=top_k, expert_top_p=top_p,
    )
    rt.prepare()
    return rt


def timed_greedy(rt, prompt, layers):
    out = []
    rt.reset()
    rt.generate(prompt, min(8, N), lambda t: out.append(t)); out.clear()
    rt.reset()
    before = rt.info
    t0 = time.perf_counter()
    rt.generate(prompt, N, lambda t: out.append(t))
    wall = time.perf_counter() - t0
    after = rt.info
    calls = after["decode_calls"] - before["decode_calls"]
    esum = after["route_expert_sum"] - before["route_expert_sum"]
    avg_experts = esum / max(1, calls) / layers
    return out, len(out) / wall, avg_experts


def agreement(rt, prompt, reference):
    rt.reset()
    for t in prompt[:-1]:
        rt.decode(t)
    ctx, agree = prompt[-1], 0
    for ref in reference:
        agree += int(rt.decode(ctx) == ref); ctx = ref
    return agree / max(1, len(reference))


def main() -> None:
    model = V2Model(MODEL)
    try:
        toks = {n: model.tokenize(t) for n, t in PROMPTS.items()}
        full_k = int(model.config["expert_used_count"])
        rt0 = make(model)
        layers = rt0.info["layers"]
        refs = {}
        for n in PROMPTS:
            out, _, _ = timed_greedy(rt0, toks[n], layers)
            refs[n] = out
        rt0.close()
        print(f"top_k={full_k} experts={model.config['expert_count']} "
              f"layers={layers} gen={N}\n", flush=True)

        configs = [("full", {}), ("static k=7", {"top_k": 7}),
                   ("p=0.95", {"top_p": 0.95}), ("p=0.90", {"top_p": 0.90}),
                   ("p=0.85", {"top_p": 0.85}), ("p=0.80", {"top_p": 0.80})]
        rows = []
        for label, kw in configs:
            rt = make(model, **kw)
            per = {}
            tok_s_all, exp_all = [], []
            for n in PROMPTS:
                out, tok_s, avg_exp = timed_greedy(rt, toks[n], layers)
                agr = 1.0 if not kw else agreement(rt, toks[n], refs[n])
                per[n] = (agr, avg_exp)
                tok_s_all.append(tok_s); exp_all.append(avg_exp)
            rt.close()
            mean_agr = sum(per[n][0] for n in PROMPTS) / len(PROMPTS)
            mean_exp = sum(exp_all) / len(exp_all)
            mean_toks = sum(tok_s_all) / len(tok_s_all)
            rows.append((label, mean_toks, mean_exp, mean_agr, per))
            print(f"{label:11s} tok/s~{mean_toks:5.1f}  experts/tok={mean_exp:4.2f}  "
                  f"agree={mean_agr:5.1%}  " +
                  " ".join(f"{n[:4]}:{per[n][0]:.0%}/{per[n][1]:.1f}" for n in PROMPTS),
                  flush=True)

        base = rows[0][1]
        print("\n=== frontier (speed vs quality; experts/tok is adaptive) ===")
        for label, toks_s, exp, agr, _ in rows:
            print(f"  {label:11s} {toks_s:5.1f} tok/s ({(toks_s/base-1)*100:+4.0f}%)"
                  f"  experts/tok={exp:4.2f}  top1_agree={agr:5.1%}")
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
