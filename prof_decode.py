"""Profile a steady-state decode region for GPU-busy vs GPU-idle analysis.

Warms the runtime, then wraps a fixed sequential-decode region in a
cudaProfilerApi capture range so nsys records only steady-state decode
(no model load / prefill / JIT). Prints the wall time of the captured
region so it can be compared against nsys' GPU kernel + memcpy busy time:
  bubble fraction = 1 - (gpu_busy_time / wall_time).
If the GPU is idle a large fraction of the token, route_wait is a bubble
(overlap/prefetch wins); if ~fully busy, it is genuine GPU compute
(expert reduction / faster kernels win).
"""
from __future__ import annotations

import os
import time

from cupy.cuda import profiler

from colibri_next.v2 import V2Model

MODEL = os.environ.get(
    "COLIBRI_MODEL", "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf")
CONTEXT = int(os.environ.get("COLIBRI_CONTEXT", "8192"))
WARM_TOKENS = 256
PROFILE_TOKENS = 128
PROMPT_TEXT = (
    "Explain why memory bandwidth on mixture-of-experts weights dominates "
    "decode latency on consumer GPUs, and how expert caching helps.\n\n"
)


def main() -> None:
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        with model.native_qwen_runtime(
            context_limit=CONTEXT,
            gpu_cache_bytes=int(
                os.environ.get("COLIBRI_GPU_CACHE_MIB", "8192")) * 1024**2,
            moe_device="hybrid",
            mtp_drafts=0,
        ) as runtime:
            runtime.prepare()
            # Warm expert cache + NVRTC JIT to steady state (not profiled).
            runtime.generate(prompt, WARM_TOKENS, lambda t: None)
            runtime.reset()

            tokens: list[int] = []
            profiler.start()
            started = time.perf_counter()
            runtime.generate(prompt, PROFILE_TOKENS, lambda t: tokens.append(t))
            wall = time.perf_counter() - started
            profiler.stop()

            print(f"PROFILED_REGION tokens={len(tokens)} "
                  f"wall_s={wall:.4f} tok_s={len(tokens) / wall:.2f} "
                  f"wall_ms_per_token={wall / len(tokens) * 1000:.3f}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
