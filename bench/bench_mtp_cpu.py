"""Does batched verification beat sequential decode *at equal expert cost*?

Under hybrid, enabling MTP collapses the GPU expert hit rate to 0% (nothing ever
admits in the verification phase), so the measured MTP loss confounds two very
different things: a cache defect, and the intrinsic economics of speculation on
a MoE.

Forcing expert_mode="cpu" runs every routed expert on the CPU in *both*
settings, taking the cache out of the comparison entirely. What is left is the
question that actually matters: does one W-row verification pass cost less than
W sequential decode steps?
"""
from __future__ import annotations

import os
import sys
import time

from flyweight.v2 import V2Model

# Direct expert paging is auto-enabled only when ~31 GiB of host RAM is free,
# so the SECOND runtime built in a process silently falls back to staged copies
# and is not comparable to the first. Force it for every runtime here.
os.environ["FLYWEIGHT_V2_DMA_PAGING"] = "1"

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MTP_MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-MTP-BF16.gguf"
MAX_TOKENS = 192
PROMPT = (
    "Write a detailed technical explanation of how speculative decoding "
    "works in large language models, covering draft models, verification, "
    "acceptance rates, and why it can speed up autoregressive generation.\n\n"
)
FIELDS = (
    "decode_calls", "decode_nanoseconds", "expert_page_nanoseconds",
    "mtp_draft_tokens", "mtp_accepted_tokens", "mtp_draft_nanoseconds",
    "mtp_verify_nanoseconds", "mtp_rollback_nanoseconds",
)


def run(model: V2Model, prompt: list[int], drafts: int) -> dict:
    if drafts:
        os.environ["FLYWEIGHT_MTP_ADAPTIVE"] = "0"
    else:
        os.environ.pop("FLYWEIGHT_MTP_ADAPTIVE", None)
    with model.native_qwen_runtime(
        context_limit=8192, gpu_cache_bytes=8192 * 1024**2,
        moe_device="cpu", mtp_drafts=drafts,
    ) as runtime:
        runtime.prepare()
        runtime.reset()
        runtime.generate(prompt, 32, lambda t: None)  # warm JIT + clocks
        runtime.reset()
        before = {f: runtime.info[f] for f in FIELDS}
        seen: list[int] = []
        started = time.perf_counter()
        runtime.generate(prompt, MAX_TOKENS, seen.append)
        elapsed = time.perf_counter() - started
        d = {f: runtime.info[f] - before[f] for f in FIELDS}
    d["wall_s"] = elapsed
    d["emitted"] = len(seen)
    d["drafts"] = drafts
    return d


def main() -> None:
    model = V2Model(MODEL, mtp_model=MTP_MODEL)
    try:
        prompt = model.tokenize(PROMPT)
        print(f"prompt={len(prompt)} gen={MAX_TOKENS} expert_mode=cpu\n")
        baseline = None
        for drafts in (0, 2, 3):
            d = run(model, prompt, drafts)
            rate = d["emitted"] / d["wall_s"]
            if baseline is None:
                baseline = rate
            print(f"--- drafts={drafts}: {rate:.2f} tok/s "
                  f"({rate / baseline - 1:+.1%})")
            if drafts:
                rounds = max(1.0, d["mtp_draft_tokens"] / drafts)
                verify_ms = d["mtp_verify_nanoseconds"] / 1e6
                print(f"    accept {d['mtp_accepted_tokens'] / max(1, d['mtp_draft_tokens']):.1%}"
                      f"   tokens/round {d['emitted'] / rounds:.2f}"
                      f"   verify/round {verify_ms / rounds:.1f} ms"
                      f"   verify/row {verify_ms / rounds / drafts:.1f} ms")
                print(f"    draft {d['mtp_draft_nanoseconds'] / 1e6:.0f} ms"
                      f"   rollback {d['mtp_rollback_nanoseconds'] / 1e6:.0f} ms")
            print(flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
