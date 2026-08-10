"""Attribution probe: where does an MTP round's time actually go?

bench_mtp.py shows MTP losing badly. This asks *why* by splitting the same runs
into expert-side counters (cache hit rate, CPU expert compute, route stall,
paging) versus everything else, for drafts=0/2/3.

Hypothesis under test: a W-row verification pass costs ~W decode steps, because
routed-expert work scales with rows and batching amortizes only the dense/attn
minority. If so, break-even needs more committed tokens per round than a round
can possibly yield, and no acceptance rate can save it.

Counter hygiene (see the KV-occupancy note): prefill route lookups would swamp
decode's, so prefill is measured once on its own and subtracted. reset() does
not clear the native counters, which is what makes that subtraction valid.
"""
from __future__ import annotations

import os
import sys
import time

from colibri_next.v2 import V2Model

# Direct expert paging is auto-enabled only when ~31 GiB of host RAM is free,
# so the SECOND runtime built in a process silently falls back to staged copies
# and is not comparable to the first. Force it for every runtime here.
os.environ["COLIBRI_V2_DMA_PAGING"] = "1"

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MTP_MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-MTP-BF16.gguf"
GPU_CACHE_BYTES = 8192 * 1024**2
CONTEXT = 8192
MAX_TOKENS = 384
PROMPT_TEXT = (
    "Write a detailed technical explanation of how speculative decoding "
    "works in large language models, covering draft models, verification, "
    "acceptance rates, and why it can speed up autoregressive generation. "
    "Then discuss the specific trade-offs for mixture-of-experts models.\n\n"
)

FIELDS = (
    "decode_calls", "decode_nanoseconds",
    "route_wait_nanoseconds", "expert_page_nanoseconds",
    "expert_compute_nanoseconds", "tail_wait_nanoseconds",
    "expert_cache_hits", "expert_cache_misses",
    "expert_cache_admissions", "route_expert_sum",
    "mtp_draft_tokens", "mtp_accepted_tokens", "mtp_rejected_tokens",
    "mtp_draft_nanoseconds", "mtp_verify_nanoseconds",
    "mtp_rollback_nanoseconds",
)


def snapshot(runtime) -> dict:
    return {f: runtime.info[f] for f in FIELDS}


def delta(before: dict, after: dict) -> dict:
    return {f: after[f] - before[f] for f in FIELDS}


def run(model: V2Model, prompt: list[int], drafts: int) -> dict:
    if drafts:
        os.environ["COLIBRI_MTP_ADAPTIVE"] = "0"
    else:
        os.environ.pop("COLIBRI_MTP_ADAPTIVE", None)
    with model.native_qwen_runtime(
        context_limit=CONTEXT, gpu_cache_bytes=GPU_CACHE_BYTES,
        moe_device="hybrid", mtp_drafts=drafts,
    ) as runtime:
        runtime.prepare()
        # Warm NVRTC JIT, the expert cache and the GPU clock ramp.
        runtime.reset()
        runtime.generate(prompt, 64, lambda t: None)

        # Prefill alone, to be subtracted from the full run below.
        runtime.reset()
        base = snapshot(runtime)
        runtime.generate(prompt, 1, lambda t: None)
        prefill = delta(base, snapshot(runtime))

        runtime.reset()
        base = snapshot(runtime)
        seen: list[int] = []
        started = time.perf_counter()
        runtime.generate(prompt, MAX_TOKENS, seen.append)
        elapsed = time.perf_counter() - started
        full = delta(base, snapshot(runtime))

    out = {f: full[f] - prefill[f] for f in FIELDS}
    # Wall clock still contains the prompt prefill; charge it out too.
    out["wall_s"] = elapsed - prefill["decode_nanoseconds"] / 1e9
    out["emitted"] = len(seen) - 1
    out["drafts"] = drafts
    return out


def report(d: dict) -> None:
    emitted = max(1, d["emitted"])
    def per(ns: int) -> float:
        return ns / 1e6 / emitted
    print(f"--- drafts={d['drafts']}   {emitted / d['wall_s']:.2f} tok/s "
          f"({d['wall_s'] * 1e3 / emitted:.2f} ms/token) ---")
    print(f"  route_wait  {per(d['route_wait_nanoseconds']):6.2f}   "
          f"expert_page {per(d['expert_page_nanoseconds']):6.2f}   "
          f"expert_cpu {per(d['expert_compute_nanoseconds']):6.2f}   "
          f"tail_wait {per(d['tail_wait_nanoseconds']):6.2f}   (ms/token)")
    hits, misses = d["expert_cache_hits"], d["expert_cache_misses"]
    total = max(1, hits + misses)
    print(f"  routed experts/token {d['route_expert_sum'] / emitted:6.1f}   "
          f"hit {hits / total:.1%}   misses/token {misses / emitted:5.1f}   "
          f"admissions {d['expert_cache_admissions']}")
    if d["drafts"]:
        drafted = max(1, d["mtp_draft_tokens"])
        rounds = drafted / d["drafts"]
        print(f"  draft {d['mtp_draft_nanoseconds'] / 1e6:5.0f} ms   "
              f"verify {d['mtp_verify_nanoseconds'] / 1e6:5.0f} ms   "
              f"rollback {d['mtp_rollback_nanoseconds'] / 1e6:5.0f} ms   "
              f"accept {d['mtp_accepted_tokens'] / drafted:.1%}")
        print(f"  rounds {rounds:.0f}   tokens/round {emitted / rounds:.2f}   "
              f"verify/round {d['mtp_verify_nanoseconds'] / 1e6 / rounds:.1f} ms"
              f"   verify/row "
              f"{d['mtp_verify_nanoseconds'] / 1e6 / rounds / d['drafts']:.1f} ms")


def main() -> None:
    model = V2Model(MODEL, mtp_model=MTP_MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        print(f"prompt={len(prompt)} gen={MAX_TOKENS}\n", flush=True)
        for drafts in (0, 2, 3):
            report(run(model, prompt, drafts))
            print(flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
