"""Is the GPU expert cache alive at all when MTP is enabled?

The attribution probe reported hit 0.0% / admissions 0 / misses == every route
under mtp_drafts, against 72.3% hit with MTP off. That is either a real
behaviour (nothing ever admits, so all 320+ routed experts per token run on the
CPU) or a counter that the rows path simply never increments. This distinguishes
them by reading the cache's own allocation and seeding counters.
"""
from __future__ import annotations

import os
import sys

from colibri_next.v2 import V2Model

# Direct expert paging is auto-enabled only when ~31 GiB of host RAM is free,
# so the SECOND runtime built in a process silently falls back to staged copies
# and is not comparable to the first. Force it for every runtime here.
os.environ["COLIBRI_V2_DMA_PAGING"] = "1"

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MTP_MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-MTP-BF16.gguf"
FIELDS = (
    "expert_cache_slots", "expert_cache_bytes", "expert_tensor_bytes",
    "expert_cache_hits", "expert_cache_misses", "expert_cache_admissions",
    "expert_cache_deferred_admissions", "expert_cache_rejections",
    "prefill_cache_seeded_experts", "expert_history_loaded_entries",
    "expert_residency_frozen", "route_expert_sum", "gpu_allocated_bytes",
    "mtp_tensor_bytes", "direct_paging", "expert_cache_prompt_bypasses",
)

PROMPT = (
    "Write a detailed technical explanation of how speculative decoding "
    "works in large language models, covering draft models, verification, "
    "acceptance rates, and why it can speed up autoregressive generation.\n\n"
)


def probe(model: V2Model, prompt: list[int], drafts: int) -> None:
    if drafts:
        os.environ["COLIBRI_MTP_ADAPTIVE"] = "0"
    else:
        os.environ.pop("COLIBRI_MTP_ADAPTIVE", None)
    with model.native_qwen_runtime(
        context_limit=8192, gpu_cache_bytes=8192 * 1024**2,
        moe_device="hybrid", mtp_drafts=drafts,
    ) as runtime:
        runtime.prepare()
        print(f"=== drafts={drafts} ===")
        print(f"  after prepare: slots={runtime.info['expert_cache_slots']} "
              f"cache={runtime.info['expert_cache_bytes'] / 1024**3:.2f} GiB "
              f"mtp_tensors={runtime.info['mtp_tensor_bytes'] / 1024**2:.0f} MiB "
              f"gpu={runtime.info['gpu_allocated_bytes'] / 1024**3:.2f} GiB")
        runtime.reset()
        runtime.generate(prompt, 96, lambda t: None)
        info = {f: runtime.info[f] for f in FIELDS}
        hits, misses = info["expert_cache_hits"], info["expert_cache_misses"]
        print(f"  after 96 tokens: hits={hits} misses={misses} "
              f"hit={hits / max(1, hits + misses):.1%}")
        print(f"    admissions={info['expert_cache_admissions']} "
              f"deferred={info['expert_cache_deferred_admissions']} "
              f"rejections={info['expert_cache_rejections']} "
              f"seeded={info['prefill_cache_seeded_experts']} "
              f"frozen={info['expert_residency_frozen']} "
              f"history={info['expert_history_loaded_entries']}")
        print(f"    direct_paging={info['direct_paging']} "
              f"prompt_bypasses={info['expert_cache_prompt_bypasses']}")
        print()


def main() -> None:
    model = V2Model(MODEL, mtp_model=MTP_MODEL)
    try:
        prompt = model.tokenize(PROMPT)
        for drafts in (0, 2):
            probe(model, prompt, drafts)
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
