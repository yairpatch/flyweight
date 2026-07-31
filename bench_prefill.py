"""Prompt-processing (prefill) baseline for the v2 native Qwen runtime.

Measures how fast the runtime ingests a prompt of length P (chunked rows
prefill + per-token tail). For each P it reports prompt tok/s and the phase
breakdown from runtime.info deltas (decode/route/expert-page/tail nanoseconds
plus prompt-cache bypasses).

Run:
    PYTHONPATH=src python3 bench_prefill.py
Env knobs:
    COLIBRI_MODEL      GGUF path (default Qwen3.6-35B-A3B-UD-Q5_K_M.gguf)
    COLIBRI_MOE_DEVICE gpu|cpu|hybrid (default hybrid)
    COLIBRI_GPU_CACHE_MIB  total GPU budget in MiB (default 8192)
    COLIBRI_PREFILL_ROWS  chunk size (default 1024; 0 disables chunked prefill)
    COLIBRI_CONTEXT    context window (default 8192)
    COLIBRI_PROMPT_LENGTHS  comma list of P (default 512,1024,2048,4096,8192)
"""

from __future__ import annotations

import os
import sys
import time

from colibri_next.v2 import V2Model

MODEL = os.environ.get(
    "COLIBRI_MODEL",
    "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
)
MOE_DEVICE = os.environ.get("COLIBRI_MOE_DEVICE", "hybrid")
GPU_CACHE_MIB = int(os.environ.get("COLIBRI_GPU_CACHE_MIB", "8192"))
CONTEXT = int(os.environ.get("COLIBRI_CONTEXT", "8192"))
PROMPT_LENGTHS = [
    int(x)
    for x in os.environ.get("COLIBRI_PROMPT_LENGTHS", "512,1024,2048,4096,8192").split(
        ","
    )
    if x
]
SENTENCE = (
    "Mixture-of-experts models route each token to a small subset of experts, "
    "which saves compute compared with a dense network of equal capacity. "
)


def make_runtime(model: V2Model):
    rt = model.native_qwen_runtime(
        context_limit=CONTEXT,
        gpu_cache_bytes=GPU_CACHE_MIB * 1024 * 1024,
        moe_device=MOE_DEVICE,
    )
    rt.prepare()
    return rt


def main() -> int:
    model = V2Model(MODEL)
    try:
        print(f"model={MODEL}", file=sys.stderr, flush=True)
        print(
            f"moe_device={MOE_DEVICE} gpu_cache_mib={GPU_CACHE_MIB} context={CONTEXT}",
            file=sys.stderr,
            flush=True,
        )
        rt = make_runtime(model)

        # Build a fixed long token stream by repeating a sentence.
        base = model.tokenize(SENTENCE)
        full = (base * 256)[: max(PROMPT_LENGTHS)]

        phase_keys = [
            "decode_nanoseconds",
            "route_wait_nanoseconds",
            "expert_page_nanoseconds",
            "tail_wait_nanoseconds",
        ]
        print(
            f"\n{'P':>6} {'prompt_tok/s':>13} {'wall_s':>8} "
            f"{'route_ms':>9} {'page_ms':>9} {'miss':>7} {'hit':>6} "
            f"{'adm':>5} {'rej':>5} {'byp':>5}"
        )
        print("-" * 86)

        for P in PROMPT_LENGTHS:
            if P > len(full):
                print(
                    f"P={P} skipped (need {P} base tokens, have {len(full)})",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            prompt = full[:P]

            # warm
            rt.reset()
            rt.generate(prompt, 4, lambda t: None)
            rt.reset()

            before = dict(rt.info)
            t0 = time.perf_counter()
            rt.generate(prompt, P, lambda t: None)  # ingest P, emit 1
            wall = time.perf_counter() - t0
            after = dict(rt.info)

            def dms(k):
                return (after.get(k, 0) - before.get(k, 0)) / 1e6

            def d(k):
                return after.get(k, 0) - before.get(k, 0)

            prompt_tok_s = P / wall
            print(
                f"{P:>6} {prompt_tok_s:>13.1f} {wall:>8.2f} "
                f"{dms('route_wait_nanoseconds'):>9.1f} "
                f"{dms('expert_page_nanoseconds'):>9.1f} "
                f"{d('expert_cache_misses'):>7} {d('expert_cache_hits'):>6} "
                f"{d('expert_cache_admissions'):>5} {d('expert_cache_rejections'):>5} "
                f"{d('expert_cache_prompt_bypasses'):>5}"
            )

        rt.close()
        return 0
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
