"""Benchmark: native hybrid decode with MTP off vs on (rigorous).

- Real tokenized prompt (acceptance depends on realistic text).
- Long generation for steady-state (default 512 tokens).
- Per-run telemetry deltas: the native counters only accumulate and reset()
  does NOT clear them, so we snapshot info before/after the timed region.
- Repeats per setting to expose variance.
- Greedy decode is bit-identical across settings, so we verify the output
  token stream matches the sequential baseline exactly.
"""
from __future__ import annotations

import os
import sys
import time

from colibri_next.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MTP_MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-MTP-BF16.gguf"
GPU_CACHE_BYTES = 8192 * 1024**2
CONTEXT = 8192
MAX_TOKENS = 512
REPEATS = 2
PROMPT_TEXT = (
    "Write a detailed technical explanation of how speculative decoding "
    "works in large language models, covering draft models, verification, "
    "acceptance rates, and why it can speed up autoregressive generation. "
    "Then discuss the specific trade-offs for mixture-of-experts models.\n\n"
)

PHASE_FIELDS = (
    "decode_calls", "decode_nanoseconds",
    "mtp_draft_tokens", "mtp_accepted_tokens", "mtp_rejected_tokens",
    "mtp_draft_nanoseconds", "mtp_verify_nanoseconds",
    "mtp_rollback_nanoseconds",
)


def ms(ns: int) -> float:
    return ns / 1e6


def run_once(runtime, prompt: list[int]) -> tuple[list[int], float, dict]:
    runtime.reset()
    before = {f: runtime.info[f] for f in PHASE_FIELDS}
    tokens: list[int] = []
    started = time.perf_counter()
    runtime.generate(prompt, MAX_TOKENS, lambda t: tokens.append(t))
    elapsed = time.perf_counter() - started
    delta = {f: runtime.info[f] - before[f] for f in PHASE_FIELDS}
    return tokens, elapsed, delta


def bench(model: V2Model, prompt: list[int], drafts: int) -> dict:
    # This diagnostic compares the actual speculative path. The production
    # default is adaptive and may deliberately disable MTP after calibration.
    if drafts:
        os.environ["COLIBRI_MTP_ADAPTIVE"] = "0"
    else:
        os.environ.pop("COLIBRI_MTP_ADAPTIVE", None)
    with model.native_qwen_runtime(
        context_limit=CONTEXT,
        gpu_cache_bytes=GPU_CACHE_BYTES,
        moe_device="hybrid",
        mtp_drafts=drafts,
    ) as runtime:
        runtime.prepare()
        # Warm caches / NVRTC JIT (not timed, telemetry delta excludes it).
        run_once(runtime, prompt)
        runs = [run_once(runtime, prompt) for _ in range(REPEATS)]
        tok_s = [len(t) / e for t, e, _ in runs]
        best = max(range(len(runs)), key=lambda i: tok_s[i])
        return {
            "drafts": drafts,
            "tokens": len(runs[best][0]),
            "tok_s_runs": tok_s,
            "tok_s": tok_s[best],
            "stream": runs[best][0],
            "delta": runs[best][2],
        }


def main() -> None:
    model = V2Model(MODEL, mtp_model=MTP_MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        print(f"prompt tokens={len(prompt)}  gen={MAX_TOKENS}  "
              f"repeats={REPEATS}\n", flush=True)
        results = []
        for drafts in (0, 2, 3):
            r = bench(model, prompt, drafts)
            results.append(r)
            runs = "  ".join(f"{v:.2f}" for v in r["tok_s_runs"])
            print(f"--- mtp_drafts={r['drafts']} ---", flush=True)
            print(f"  tok/s runs=[{runs}]  best={r['tok_s']:.2f}", flush=True)
            d = r["delta"]
            print(f"  decode_calls={d['decode_calls']}  "
                  f"decode_ms={ms(d['decode_nanoseconds']):.0f}", flush=True)
            if drafts:
                drafted = d["mtp_draft_tokens"]
                acc = d["mtp_accepted_tokens"]
                print(f"  draft_ms={ms(d['mtp_draft_nanoseconds']):.0f}  "
                      f"verify_ms={ms(d['mtp_verify_nanoseconds']):.0f}  "
                      f"rollback_ms={ms(d['mtp_rollback_nanoseconds']):.0f}",
                      flush=True)
                print(f"  drafted={drafted}  accepted={acc}  "
                      f"rejected={d['mtp_rejected_tokens']}  "
                      f"accept={acc / max(1, drafted):.1%}  "
                      f"tokens/round={r['tokens'] / max(1, d['decode_calls'] // max(1, drafts)):.2f}",
                      flush=True)

        base = results[0]
        print("\n=== summary (baseline = sequential hybrid, best of "
              f"{REPEATS}) ===")
        for r in results:
            match = "identical" if r["stream"] == base["stream"] else "DIFFERENT!"
            delta = (r["tok_s"] / base["tok_s"] - 1) * 100
            print(f"  drafts={r['drafts']}: {r['tok_s']:.2f} tok/s "
                  f"({delta:+.1f}%)  output={match}")
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
