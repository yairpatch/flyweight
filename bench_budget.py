"""Where does a steady-state decode token go, at the current 51 tok/s?

Pure sequential decode (mtp_drafts=0). Reads the native telemetry deltas
around a warm 512-token generation: decode time and its wait components
(route wait, expert page-in, tail wait), plus the GPU/CPU expert split from
the expert-cache hit/miss counters. Repeats across GPU cache sizes to test
whether resident-expert coverage is a lever.
"""
from __future__ import annotations

import time

from colibri_next.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
CONTEXT = 8192
MAX_TOKENS = 512
PROMPT_TEXT = (
    "Write a detailed technical explanation of how a mixture-of-experts "
    "transformer routes tokens to experts, and why memory bandwidth on the "
    "expert weights dominates decode latency on consumer hardware.\n\n"
)

FIELDS = (
    "decode_calls", "decode_nanoseconds",
    "route_wait_nanoseconds", "expert_page_nanoseconds",
    "tail_wait_nanoseconds",
    "expert_cache_hits", "expert_cache_misses", "expert_cache_evictions",
    "expert_cache_admissions", "expert_cache_prompt_bypasses",
)


def ms(ns: int) -> float:
    return ns / 1e6


def measure(model: V2Model, prompt: list[int], gpu_cache_mib: int) -> None:
    with model.native_qwen_runtime(
        context_limit=CONTEXT,
        gpu_cache_bytes=gpu_cache_mib * 1024**2,
        moe_device="hybrid",
        mtp_drafts=0,
    ) as runtime:
        runtime.prepare()
        info0 = runtime.info
        slots = info0["expert_cache_slots"]
        cache_bytes = info0["expert_cache_bytes"]
        runtime.generate(prompt, MAX_TOKENS, lambda t: None)  # warm
        runtime.reset()
        before = {f: runtime.info[f] for f in FIELDS}
        tokens: list[int] = []
        started = time.perf_counter()
        runtime.generate(prompt, MAX_TOKENS, lambda t: tokens.append(t))
        wall = time.perf_counter() - started
        d = {f: runtime.info[f] - before[f] for f in FIELDS}

        n = len(tokens)
        decode_ms = ms(d["decode_nanoseconds"])
        route = ms(d["route_wait_nanoseconds"])
        page = ms(d["expert_page_nanoseconds"])
        tail = ms(d["tail_wait_nanoseconds"])
        compute = decode_ms - route - page - tail
        hits = d["expert_cache_hits"]
        misses = d["expert_cache_misses"]
        total = max(1, hits + misses)

        print(f"=== gpu_cache={gpu_cache_mib} MiB  "
              f"slots={slots}  cache_bytes={cache_bytes / 1024**3:.2f} GiB ===")
        print(f"  wall={wall:.2f}s  tok/s={n / wall:.1f}  "
              f"tokens={n}", flush=True)
        print(f"  decode_ms/token: total={decode_ms / n:.2f}  "
              f"compute={compute / n:.2f}  route_wait={route / n:.2f}  "
              f"expert_page={page / n:.2f}  tail_wait={tail / n:.2f}")
        print(f"  expert routes: hits={hits}  misses={misses}  "
              f"GPU-hit-rate={hits / total:.1%}  "
              f"evict={d['expert_cache_evictions']}  "
              f"admit={d['expert_cache_admissions']}", flush=True)
        print()


def main() -> None:
    model = V2Model(MODEL)
    try:
        prompt = model.tokenize(PROMPT_TEXT)
        print(f"prompt={len(prompt)} tok  gen={MAX_TOKENS}\n")
        for gpu_cache_mib in (6144, 8192, 10240):
            measure(model, prompt, gpu_cache_mib)
    finally:
        model.close()


if __name__ == "__main__":
    main()
