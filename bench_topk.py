"""Expert top-k pruning: speed vs quality curve.

For each expert_top_k, measures:
  - decode tok/s (fresh greedy generation)
  - quality = teacher-forced top-1 agreement vs the FULL-model greedy
    trajectory: feed the identical reference context at every position and
    check whether the pruned model's argmax next-token matches the full
    model's. This is trajectory-drift-free (unlike comparing two greedy
    generations), so it isolates routing fidelity.
Also prints the decoded greedy text at each k for eyeballing coherence.
Full top-k is re-measured at the end to bracket the laptop's thermal drift.
"""
from __future__ import annotations

import sys
import time

from colibri_next.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
CONTEXT = 8192
N = 96
PROMPT_TEXT = (
    "The three key advantages of mixture-of-experts language models over "
    "dense models are:\n"
)


def make_runtime(model, k):
    rt = model.native_qwen_runtime(
        context_limit=CONTEXT, gpu_cache_bytes=8192 * 1024**2,
        moe_device="hybrid", expert_top_k=k,
    )
    rt.prepare()
    return rt


def greedy(runtime, prompt, n):
    out = []
    runtime.reset()
    runtime.generate(prompt, n, lambda t: out.append(t))
    return out


def timed_greedy(runtime, prompt, n):
    out = []
    runtime.reset()
    runtime.generate(prompt, min(8, n), lambda t: out.append(t))  # warm
    out.clear()
    runtime.reset()
    start = time.perf_counter()
    runtime.generate(prompt, n, lambda t: out.append(t))
    return out, time.perf_counter() - start


def agreement(runtime, prompt, reference):
    """Top-1 agreement of pruned model vs reference trajectory (teacher-forced)."""
    runtime.reset()
    for t in prompt[:-1]:
        runtime.decode(t)
    context = prompt[-1]
    agree = 0
    for ref in reference:
        pred = runtime.decode(context)
        agree += int(pred == ref)
        context = ref  # feed the reference token, not our own prediction
    return agree / max(1, len(reference))


def main() -> None:
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        full = int(model.config["expert_used_count"])
        print(f"model expert_used_count(top_k)={full}  prompt={len(prompt)} "
              f"gen={N}\n", flush=True)

        # Reference trajectory from the full model.
        ref_rt = make_runtime(model, 0)
        reference, ref_wall = timed_greedy(ref_rt, prompt, N)
        ref_tok_s = len(reference) / ref_wall
        ref_text = model.decode_tokens(reference)
        print(f"[full k={full}] tok/s={ref_tok_s:.2f}")
        print(f"  text: {ref_text!r}\n", flush=True)
        ref_rt.close()

        ladder = sorted({full - 1, full - 2, full - 3, max(2, full // 2),
                         max(1, full - 4)}, reverse=True)
        rows = []
        for k in ladder:
            if k < 1 or k >= full:
                continue
            rt = make_runtime(model, k)
            agr = agreement(rt, prompt, reference)
            gen, wall = timed_greedy(rt, prompt, N)
            tok_s = len(gen) / wall
            text = model.decode_tokens(gen)
            first_div = next((i for i, (a, b) in enumerate(zip(gen, reference))
                              if a != b), min(len(gen), len(reference)))
            rt.close()
            rows.append((k, tok_s, agr, first_div, text))
            print(f"[k={k}] tok/s={tok_s:.2f}  top1_agreement={agr:.1%}  "
                  f"greedy_first_divergence={first_div}/{N}")
            print(f"  text: {text!r}\n", flush=True)

        # Bracket thermal drift.
        ref_rt2 = make_runtime(model, 0)
        _, ref_wall2 = timed_greedy(ref_rt2, prompt, N)
        ref_rt2.close()
        ref_tok_s2 = N / ref_wall2

        print("=== summary (full re-measured for drift bracket) ===")
        print(f"  k={full} (full): {ref_tok_s:.2f} -> {ref_tok_s2:.2f} tok/s "
              f"(drift bracket), agreement=100%")
        base = (ref_tok_s + ref_tok_s2) / 2
        for k, tok_s, agr, first_div, _ in rows:
            print(f"  k={k}: {tok_s:.2f} tok/s ({(tok_s / base - 1) * 100:+.1f}% "
                  f"vs full-avg)  top1_agree={agr:.1%}  first_div={first_div}")
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
