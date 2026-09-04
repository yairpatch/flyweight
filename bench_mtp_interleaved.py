"""MTP on vs off, interleaved, medians -- the only trustworthy form on this box.

Sequential runs of this benchmark disagree wildly: the same drafts=0 baseline
measured 53.8 tok/s in one process and 36.3 in the next, while the native
counters stayed stable to ~1%. That is the GPU clock/thermal state drifting, so
any A/B that runs all of A then all of B is measuring the drift.

Here every replicate visits every setting, and the reported number is the median
across replicates. Ratios between settings within a replicate are what carry
signal; absolute tok/s still does not.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

from flyweight.v2 import V2Model

# Direct expert paging is auto-enabled only when ~31 GiB of host RAM is free, so
# the second runtime built in a process silently falls back to staged copies and
# is not comparable to the first. Force it for every runtime.
os.environ["FLYWEIGHT_V2_DMA_PAGING"] = "1"

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MTP_MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-MTP-BF16.gguf"
GPU_CACHE_BYTES = 8192 * 1024**2
CONTEXT = 8192
MAX_TOKENS = 256
REPLICATES = 3
SETTINGS = (0, 2, 3)
PROMPT_TEXT = (
    "Write a detailed technical explanation of how speculative decoding "
    "works in large language models, covering draft models, verification, "
    "acceptance rates, and why it can speed up autoregressive generation. "
    "Then discuss the specific trade-offs for mixture-of-experts models.\n\n"
)
FIELDS = (
    "decode_nanoseconds", "route_wait_nanoseconds", "expert_page_nanoseconds",
    "expert_cache_hits", "expert_cache_misses",
    "mtp_draft_tokens", "mtp_accepted_tokens", "mtp_verify_nanoseconds",
    "mtp_draft_nanoseconds", "mtp_rollback_nanoseconds",
)


def once(model: V2Model, prompt: list[int], drafts: int) -> dict:
    if drafts:
        os.environ["FLYWEIGHT_MTP_ADAPTIVE"] = "0"
    else:
        os.environ.pop("FLYWEIGHT_MTP_ADAPTIVE", None)
    with model.native_qwen_runtime(
        context_limit=CONTEXT, gpu_cache_bytes=GPU_CACHE_BYTES,
        moe_device="hybrid", mtp_drafts=drafts,
    ) as runtime:
        runtime.prepare()
        runtime.reset()
        runtime.generate(prompt, 48, lambda t: None)  # JIT + clocks + cache
        runtime.reset()
        before = {f: runtime.info[f] for f in FIELDS}
        seen: list[int] = []
        started = time.perf_counter()
        runtime.generate(prompt, MAX_TOKENS, seen.append)
        elapsed = time.perf_counter() - started
        d = {f: runtime.info[f] - before[f] for f in FIELDS}
    d["tok_s"] = len(seen) / elapsed
    hits, misses = d["expert_cache_hits"], d["expert_cache_misses"]
    d["hit_rate"] = hits / max(1, hits + misses)
    if drafts:
        rounds = max(1.0, d["mtp_draft_tokens"] / drafts)
        d["verify_per_row_ms"] = (
            d["mtp_verify_nanoseconds"] / 1e6 / rounds / drafts)
        d["tokens_per_round"] = len(seen) / rounds
        d["accept"] = d["mtp_accepted_tokens"] / max(1, d["mtp_draft_tokens"])
    return d


def main() -> None:
    model = V2Model(MODEL, mtp_model=MTP_MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        print(f"prompt={len(prompt)} gen={MAX_TOKENS} "
              f"replicates={REPLICATES}\n", flush=True)
        results: dict[int, list[dict]] = {d: [] for d in SETTINGS}
        for replicate in range(REPLICATES):
            for drafts in SETTINGS:
                d = once(model, prompt, drafts)
                results[drafts].append(d)
                print(f"  rep{replicate} drafts={drafts}: "
                      f"{d['tok_s']:.2f} tok/s  hit {d['hit_rate']:.1%}",
                      flush=True)
        print("\n=== medians ===")
        base = statistics.median(r["tok_s"] for r in results[0])
        for drafts in SETTINGS:
            runs = results[drafts]
            med = statistics.median(r["tok_s"] for r in runs)
            spread = f"{min(r['tok_s'] for r in runs):.1f}-{max(r['tok_s'] for r in runs):.1f}"
            line = (f"drafts={drafts}: {med:6.2f} tok/s ({med / base - 1:+6.1%})"
                    f"  range {spread}"
                    f"  hit {statistics.median(r['hit_rate'] for r in runs):.1%}"
                    f"  route_wait "
                    f"{statistics.median(r['route_wait_nanoseconds'] for r in runs) / 1e6 / MAX_TOKENS:.2f} ms/tok")
            if drafts:
                line += (
                    f"\n            verify/row "
                    f"{statistics.median(r['verify_per_row_ms'] for r in runs):.1f} ms"
                    f"  tokens/round "
                    f"{statistics.median(r['tokens_per_round'] for r in runs):.2f}"
                    f"  accept "
                    f"{statistics.median(r['accept'] for r in runs):.1%}")
            print(line)
        print(f"\nbaseline ms/token (median): {1000 / base:.2f}")
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
