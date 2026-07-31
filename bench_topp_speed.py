"""Clean interleaved speed: full vs top-p=0.85, to defeat thermal drift."""
from __future__ import annotations
import sys, time
from colibri_next.v2 import V2Model

MODEL = "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
N = 256
PROMPT = "Explain how mixture-of-experts routing works and why it saves compute.\n"


def run(model, top_p):
    rt = model.native_qwen_runtime(context_limit=8192, gpu_cache_bytes=8192 * 1024**2,
                                   moe_device="hybrid", expert_top_p=top_p)
    rt.prepare()
    prompt = model.tokenize(PROMPT)
    layers = rt.info["layers"]
    out = []
    rt.reset(); rt.generate(prompt, 16, lambda t: out.append(t)); out.clear()  # warm
    rt.reset()
    b = rt.info
    t0 = time.perf_counter(); rt.generate(prompt, N, lambda t: out.append(t))
    wall = time.perf_counter() - t0
    a = rt.info
    exp = (a["route_expert_sum"] - b["route_expert_sum"]) / (a["decode_calls"] - b["decode_calls"]) / layers
    rt.close()
    return N / wall, exp


def main():
    model = V2Model(MODEL)
    try:
        res = {0.0: [], 0.85: []}
        for _ in range(3):
            for p in (0.0, 0.85):
                tok_s, exp = run(model, p)
                res[p].append((tok_s, exp))
                print(f"p={p or 'full':>4}  tok/s={tok_s:5.2f}  experts/tok={exp:.2f}", flush=True)
        full = sum(t for t, _ in res[0.0]) / 3
        pruned = sum(t for t, _ in res[0.85]) / 3
        exp = sum(e for _, e in res[0.85]) / 3
        print(f"\nfull avg={full:.2f} tok/s | p=0.85 avg={pruned:.2f} tok/s "
              f"({(pruned/full-1)*100:+.1f}%) at {exp:.2f} experts/tok")
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
