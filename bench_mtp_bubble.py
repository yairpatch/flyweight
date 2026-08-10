#!/usr/bin/env python3
"""Is MTP's route_wait real GPU work, or a bubble worth restructuring for?

route_wait is 15-24 ms/token under MTP against 12.7-13.8 baseline at identical
expert hit rate, which is the last unexplained gap. But route_wait only measures
"CPU blocked at the route sync" -- that is genuine GPU compute if the GPU is
saturated, and a bubble only if it is not.

COLIBRI_MTP_PROFILE=1 records GPU-side events inside the verification pass and
splits it into core (attention + dense projections), router, and the route
transfer. If core+router+transfer accounts for route_wait, the rows path is
GPU-bound and overlapping the host round trip buys nothing; the gap would then
have to be genuine extra GPU work. If it does not, the difference is the bubble.

The counters are shared with prefill, so prefill is measured on its own and
subtracted.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

from colibri_next.v2 import V2Model

os.environ["COLIBRI_V2_DMA_PAGING"] = "1"
os.environ["COLIBRI_MTP_PROFILE"] = "1"

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MTP_MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-MTP-BF16.gguf"
MAX_TOKENS = 192
REPLICATES = 2
PROMPT_TEXT = (
    "Write a detailed technical explanation of how speculative decoding "
    "works in large language models, covering draft models, verification, "
    "acceptance rates, and why it can speed up autoregressive generation. "
    "Then discuss the specific trade-offs for mixture-of-experts models.\n\n"
)
FIELDS = (
    "route_wait_nanoseconds", "expert_page_nanoseconds",
    "prefill_gpu_core_nanoseconds", "prefill_gpu_router_nanoseconds",
    "prefill_gpu_transfer_nanoseconds",
    "mtp_draft_tokens", "mtp_verify_nanoseconds",
    "expert_cache_hits", "expert_cache_misses",
)


def snapshot(runtime) -> dict:
    return {f: runtime.info[f] for f in FIELDS}


def run(model: V2Model, prompt: list[int], drafts: int) -> dict:
    if drafts:
        os.environ["COLIBRI_MTP_ADAPTIVE"] = "0"
    else:
        os.environ.pop("COLIBRI_MTP_ADAPTIVE", None)
    with model.native_qwen_runtime(
        context_limit=8192, gpu_cache_bytes=8192 * 1024**2,
        moe_device="hybrid", mtp_drafts=drafts,
    ) as runtime:
        runtime.prepare()
        runtime.reset()
        runtime.generate(prompt, 48, lambda t: None)

        runtime.reset()
        base = snapshot(runtime)
        runtime.generate(prompt, 1, lambda t: None)
        after_prefill = snapshot(runtime)
        prefill = {f: after_prefill[f] - base[f] for f in FIELDS}

        runtime.reset()
        base = snapshot(runtime)
        seen: list[int] = []
        started = time.perf_counter()
        runtime.generate(prompt, MAX_TOKENS, seen.append)
        elapsed = time.perf_counter() - started
        full = {f: runtime.info[f] - base[f] for f in FIELDS}

    d = {f: full[f] - prefill[f] for f in FIELDS}
    d["emitted"] = len(seen) - 1
    d["wall_s"] = elapsed - prefill["route_wait_nanoseconds"] / 1e9
    d["drafts"] = drafts
    return d


def report(runs: list[dict]) -> None:
    drafts = runs[0]["drafts"]
    def med(key: str) -> float:
        return statistics.median(r[key] for r in runs)
    emitted = med("emitted")
    def ms(key: str) -> float:
        return med(key) / 1e6 / emitted
    core, router, transfer = (ms("prefill_gpu_core_nanoseconds"),
                              ms("prefill_gpu_router_nanoseconds"),
                              ms("prefill_gpu_transfer_nanoseconds"))
    wait = ms("route_wait_nanoseconds")
    hits = med("expert_cache_hits")
    misses = med("expert_cache_misses")
    print(f"--- drafts={drafts}   "
          f"{emitted / med('wall_s'):.2f} tok/s   "
          f"hit {hits / max(1.0, hits + misses):.1%}")
    if drafts:
        print(f"  GPU core {core:6.2f}  router {router:5.2f}  "
              f"transfer {transfer:5.2f}  = {core + router + transfer:6.2f}"
              f"   vs route_wait {wait:6.2f}   (ms/token)")
        print(f"  unexplained by GPU work: "
              f"{wait - (core + router + transfer):6.2f} ms/token "
              f"({1 - (core + router + transfer) / max(wait, 1e-9):.0%} of the wait)")
    else:
        print(f"  route_wait {wait:6.2f}   (no verification pass to profile)")
    print(f"  expert_page {ms('expert_page_nanoseconds'):6.2f} ms/token")
    print(flush=True)


def main() -> int:
    model = V2Model(MODEL, mtp_model=MTP_MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        print(f"prompt={len(prompt)} gen={MAX_TOKENS} "
              f"replicates={REPLICATES}\n", flush=True)
        results: dict[int, list[dict]] = {0: [], 2: [], 3: []}
        for _ in range(REPLICATES):
            for drafts in (0, 2, 3):
                results[drafts].append(run(model, prompt, drafts))
        for drafts in (0, 2, 3):
            report(results[drafts])
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
