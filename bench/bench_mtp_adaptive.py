#!/usr/bin/env python3
"""Does the shipped adaptive gate actually turn MTP off?

Every other MTP benchmark here forces FLYWEIGHT_MTP_ADAPTIVE=0. The shipped
default instead calibrates: it times the first kQwenMtpBaselineTokens (16)
tokens with MTP off, then 4 MTP rounds, and keeps MTP only if it measures under
kQwenMtpKeepPercent (80%) of the baseline per-token cost. MTP measures 1.16-1.29x
baseline, so the gate should trip every time.

The worry is the *order*. Baseline is calibrated first, on a cold GPU whose
clocks are still ramping (this box runs kernels at a fraction of sustained clock
when idle), so the baseline per-token cost is over-estimated -- which biases the
comparison toward KEEPING MTP. That is the dangerous direction: a user passing
--mtp-drafts would silently eat the full regression.

So: fresh runtime per trial (the calibration state is never cleared by reset(),
so it is decided once per runtime lifetime), first generate() timed, and the
trial repeated on a cold and a warmed GPU.
"""
from __future__ import annotations

import os
import sys
import time

from flyweight.v2 import V2Model

os.environ["FLYWEIGHT_V2_DMA_PAGING"] = "1"
os.environ.pop("FLYWEIGHT_MTP_ADAPTIVE", None)  # shipped default

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MTP_MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-MTP-BF16.gguf"
MAX_TOKENS = 256
PROMPT_TEXT = (
    "Write a detailed technical explanation of how speculative decoding "
    "works in large language models, covering draft models, verification, "
    "acceptance rates, and why it can speed up autoregressive generation. "
    "Then discuss the specific trade-offs for mixture-of-experts models.\n\n"
)
FIELDS = ("mtp_draft_tokens", "mtp_accepted_tokens", "decode_calls")


def run_trial(model: V2Model, prompt: list[int], drafts: int, warm: bool) -> dict:
    """One fresh runtime, one timed generate -- what a user actually gets."""
    with model.native_qwen_runtime(
        context_limit=8192, gpu_cache_bytes=8192 * 1024**2,
        moe_device="hybrid", mtp_drafts=drafts,
    ) as runtime:
        runtime.prepare()
        if warm:
            # Ramp the GPU clock without touching calibration: a separate
            # runtime would reset it, so warm this one and accept that its
            # calibration then happens on a hot GPU. That is the comparison.
            runtime.generate(prompt, 64, lambda t: None)
            runtime.reset()
        before = {f: runtime.info[f] for f in FIELDS}
        seen: list[int] = []
        started = time.perf_counter()
        runtime.generate(prompt, MAX_TOKENS, seen.append)
        elapsed = time.perf_counter() - started
        after = {f: runtime.info[f] - before[f] for f in FIELDS}
    return {
        "tok_s": len(seen) / elapsed,
        "drafted": after["mtp_draft_tokens"],
        "drafts": drafts,
        "warm": warm,
    }


def main() -> int:
    model = V2Model(MODEL, mtp_model=MTP_MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        print(f"gen={MAX_TOKENS}  adaptive gate at its shipped default\n",
              flush=True)
        print("'KEPT MTP' means the gate failed to protect the user.\n",
              flush=True)
        # The calibration trial itself drafts kQwenMtpTrialRounds * drafts
        # tokens before deciding. At or below that, the gate tried MTP,
        # measured it and fell back -- which is the gate working.
        trial_drafts = 4 * 2
        for warm in (False, True):
            label = "warm GPU" if warm else "cold GPU"
            base = run_trial(model, prompt, 0, warm)
            mtp = run_trial(model, prompt, 2, warm)
            verdict = ("KEPT MTP" if mtp["drafted"] > trial_drafts
                       else "fell back after trial")
            print(f"--- {label} ---")
            print(f"  drafts=0: {base['tok_s']:6.2f} tok/s")
            print(f"  drafts=2: {mtp['tok_s']:6.2f} tok/s  "
                  f"({mtp['tok_s'] / base['tok_s'] - 1:+.1%})  "
                  f"gate {verdict}  drafted={mtp['drafted']}")
            print(flush=True)
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
