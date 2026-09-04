"""Verify GPU-cache auto-fit vs manual budget.

gpu_cache_bytes=0 -> auto (probe free VRAM). Positive -> manual bytes.
Reports the cache size each mode chose, plus a short decode to confirm it
works and the expert hit rate.
"""
from __future__ import annotations
import sys, time
from flyweight.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
N = 128
PROMPT = "Explain mixture-of-experts routing in one paragraph.\n"


def run(model, label, gpu_cache_bytes):
    rt = model.native_qwen_runtime(context_limit=8192,
                                   gpu_cache_bytes=gpu_cache_bytes,
                                   moe_device="hybrid")
    rt.prepare()
    info = rt.info
    prompt = model.tokenize(PROMPT)
    out = []
    rt.reset(); rt.generate(prompt, 16, lambda t: out.append(t)); out.clear()  # warm
    rt.reset()
    b = rt.info
    t0 = time.perf_counter(); rt.generate(prompt, N, lambda t: out.append(t))
    wall = time.perf_counter() - t0
    a = rt.info
    hits = a["expert_cache_hits"] - b["expert_cache_hits"]
    miss = a["expert_cache_misses"] - b["expert_cache_misses"]
    rt.close()
    print(f"{label:14s} cache={info['expert_cache_bytes']/1024**3:5.2f} GiB "
          f"slots={info['expert_cache_slots']:5d} "
          f"gpu_alloc={info['gpu_allocated_bytes']/1024**3:5.2f} GiB | "
          f"tok/s={N/wall:5.1f}  hit={hits/max(1,hits+miss):.0%}", flush=True)


def main():
    model = V2Model(MODEL)
    try:
        run(model, "auto (0)", 0)
        run(model, "manual 8 GiB", 8192 * 1024**2)
        run(model, "manual 4 GiB", 4096 * 1024**2)
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
