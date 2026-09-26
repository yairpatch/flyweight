"""Warm 2048-token prefill + FLYWEIGHT_MOE_PROFILE dump for UD-IQ4_XS.

One measured generate per process so the [moe] atexit dump is a single
prefill. Warm the page cache in a prior run with FLYWEIGHT_MOE_PROFILE=0.
"""

from __future__ import annotations

import os
import sys
import time

from flyweight.v2 import V2Model

MODEL = os.environ.get(
    "FLYWEIGHT_MODEL",
    "/home/yair/Downloads/gguf/UD-IQ4_XS/UD-IQ4_XS/"
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
)
MOE_DEVICE = os.environ.get("FLYWEIGHT_MOE_DEVICE", "hybrid")
GPU_CACHE_MIB = int(os.environ.get("FLYWEIGHT_GPU_CACHE_MIB", "8192"))
CONTEXT = int(os.environ.get("FLYWEIGHT_CONTEXT", "4096"))
PROMPT_LEN = int(os.environ.get("FLYWEIGHT_PROMPT_LENGTH", "2048"))
# -1 auto, 0 off (CPU MoE only — clean fold A/B), >0 explicit arena MiB
STREAM_MIB = int(os.environ.get("FLYWEIGHT_PREFILL_STREAM_MIB", "0"))
WARM_ONLY = os.environ.get("FLYWEIGHT_WARM_ONLY", "0") == "1"

SENTENCE = (
    "Mixture-of-experts models route each token to a small subset of experts, "
    "which saves compute compared with a dense network of equal capacity. "
)


def main() -> int:
    label = os.environ.get("FLYWEIGHT_ARM_LABEL", "?")
    print(
        f"arm={label} model={MODEL}\n"
        f"moe_device={MOE_DEVICE} gpu_cache_mib={GPU_CACHE_MIB} "
        f"context={CONTEXT} P={PROMPT_LEN} stream_mib={STREAM_MIB} "
        f"warm_only={WARM_ONLY}\n"
        f"ROWS_Q8_IQ3S={os.environ.get('FLYWEIGHT_ROWS_Q8_IQ3S', '<default>')}\n"
        f"MOE_PROFILE={os.environ.get('FLYWEIGHT_MOE_PROFILE', '<unset>')}",
        file=sys.stderr,
        flush=True,
    )
    model = V2Model(MODEL)
    try:
        rt = model.native_qwen_runtime(
            context_limit=CONTEXT,
            gpu_cache_bytes=GPU_CACHE_MIB * 1024 * 1024,
            moe_device=MOE_DEVICE,
            prefill_expert_stream_mib=STREAM_MIB,
        )
        rt.prepare()
        info = dict(rt.info)
        print(
            f"prepare: expert_mode={info.get('expert_mode')} "
            f"prefill_expert_stream_mib={info.get('prefill_expert_stream_mib')} "
            f"routed_moe={info.get('routed_moe')}",
            file=sys.stderr,
            flush=True,
        )
        base = model.tokenize(SENTENCE)
        prompt = (base * 256)[:PROMPT_LEN]

        if WARM_ONLY:
            t0 = time.perf_counter()
            rt.generate(prompt, 1, lambda _t: None)
            print(
                f"warm wall={time.perf_counter() - t0:.2f}s "
                f"({PROMPT_LEN / (time.perf_counter() - t0):.1f} tok/s)",
                file=sys.stderr,
                flush=True,
            )
            rt.close()
            return 0

        before = dict(rt.info)
        t0 = time.perf_counter()
        rt.generate(prompt, 1, lambda _t: None)
        wall = time.perf_counter() - t0
        after = dict(rt.info)

        def d(k: str) -> int:
            return int(after.get(k, 0) - before.get(k, 0))

        streamed = d("prefill_streamed_bytes")
        expert_ns = d("prefill_expert_nanoseconds")
        print(
            f"RESULT arm={label} P={PROMPT_LEN} wall_s={wall:.2f} "
            f"tok_s={PROMPT_LEN / wall:.1f} "
            f"prefill_expert_s={expert_ns / 1e9:.2f} "
            f"streamed_GiB={streamed / (1 << 30):.2f} "
            f"route_ms={d('route_wait_nanoseconds') / 1e6:.1f}",
            flush=True,
        )
        rt.close()
        return 0
    finally:
        model.close()


if __name__ == "__main__":
    raise SystemExit(main())
