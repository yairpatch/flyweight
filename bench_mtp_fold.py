"""A/B: MTP rejection rollback via ReplaySSM fold vs full replay forward.

One runtime, greedy decode, interleaved arms (fold, replay, fold, replay) so
GPU clock drift and desktop contention hit both sides equally. The fold is
bitwise-identical by construction, so every run's token stream must match --
a stream mismatch fails the diagnostic, not just the timing.

Determinism guards per the Ornith findings: explicit gpu_cache_bytes,
COLIBRI_EXPERT_HISTORY=0, no prefill cache seed, single runtime for all arms.
"""
from __future__ import annotations

import os
import time

from colibri_next.v2 import V2Model

os.environ["COLIBRI_EXPERT_HISTORY"] = "0"
os.environ["COLIBRI_MTP_ADAPTIVE"] = "0"

MODEL = "/home/yair/Downloads/Ornith-1.5-35B-Q4_K_M.gguf"
GPU_CACHE_BYTES = 8192 * 1024**2
CONTEXT = 4096
MAX_TOKENS = 128
PAIRS = 2
DRAFTS = 3
PROMPT_TEXT = (
    "Write a short story about a lighthouse keeper who discovers an old "
    "journal in the attic. Describe what the journal reveals and how it "
    "changes her understanding of the island's history.\n\n"
)

FIELDS = (
    "decode_calls", "decode_nanoseconds",
    "mtp_draft_tokens", "mtp_accepted_tokens", "mtp_rejected_tokens",
    "mtp_draft_nanoseconds", "mtp_verify_nanoseconds",
    "mtp_rollback_nanoseconds",
)


def run_once(runtime, prompt, fold: bool):
    os.environ["COLIBRI_MTP_FOLD"] = "1" if fold else "0"
    runtime.reset()
    before = {f: runtime.info[f] for f in FIELDS}
    tokens: list[int] = []
    started = time.perf_counter()
    runtime.generate(prompt, MAX_TOKENS, lambda t: tokens.append(t))
    elapsed = time.perf_counter() - started
    delta = {f: runtime.info[f] - before[f] for f in FIELDS}
    return tokens, elapsed, delta


def main() -> None:
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        print(f"prompt={len(prompt)} tokens  gen={MAX_TOKENS}  "
              f"drafts={DRAFTS}  pairs={PAIRS}", flush=True)
        with model.native_qwen_runtime(
            context_limit=CONTEXT,
            gpu_cache_bytes=GPU_CACHE_BYTES,
            moe_device="hybrid",
            mtp_drafts=DRAFTS,
        ) as runtime:
            runtime.prepare()
            run_once(runtime, prompt, fold=True)  # warm caches / JIT, untimed
            runs = []
            for pair in range(PAIRS):
                for fold in (True, False):
                    tokens, elapsed, delta = run_once(runtime, prompt, fold)
                    runs.append((fold, tokens, elapsed, delta))
                    rej = delta["mtp_rejected_tokens"]
                    print(f"  [{pair}] fold={int(fold)}  "
                          f"{len(tokens) / elapsed:6.2f} tok/s  "
                          f"rollback={delta['mtp_rollback_nanoseconds'] / 1e6:8.1f} ms"
                          f"  rejected_rounds={rej}"
                          f"  per_reject={delta['mtp_rollback_nanoseconds'] / 1e3 / max(1, rej):7.0f} us",
                          flush=True)
            reference = runs[0][1]
            for fold, tokens, _, _ in runs:
                if tokens != reference:
                    raise SystemExit("TOKEN STREAM DIVERGED: fold="
                                     f"{int(fold)} differs from first run")
            print("\nall token streams identical", flush=True)
            for name, want in (("fold", True), ("replay", False)):
                arm = [(t, e, d) for f, t, e, d in runs if f is want]
                tok_s = [len(t) / e for t, e, _ in arm]
                roll = sum(d["mtp_rollback_nanoseconds"] for _, _, d in arm)
                rej = sum(d["mtp_rejected_tokens"] for _, _, d in arm)
                acc = sum(d["mtp_accepted_tokens"] for _, _, d in arm)
                drafted = sum(d["mtp_draft_tokens"] for _, _, d in arm)
                print(f"{name:>7}: tok/s={max(tok_s):.2f} (runs: "
                      + " ".join(f"{v:.2f}" for v in tok_s)
                      + f")  rollback_total={roll / 1e6:.1f} ms"
                      f"  accept={acc / max(1, drafted):.1%}"
                      f"  rejected_rounds={rej}", flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
