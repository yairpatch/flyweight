"""Is the expert cache's admission policy too aggressive?

The llama.cpp MoE-cache RFC (ggml-org/llama.cpp#24528) reported that admitting
an expert on first touch is too eager, and that requiring ~64 observations first
improved every workload they measured. Our cache admits on first touch too, and
in hybrid mode the admission is pure speculation: the miss is already being
served on the CPU, so the upload only pays off if a later token routes to the
same expert before it is evicted.

An absolute threshold was implemented and measured here on 2026-08-29. It never
won -- -3.0 / -4.5 / -14.0% decode at 4 / 16 / 64 against a 4.7% within-arm
spread, and on a small cache it changed nothing at all (threshold 4 admitted the
same 719 experts as no threshold). The reason it changed nothing is the knob
this bench compares instead: strict admission already refuses a candidate that
is not demonstrably hotter than the resident it would evict, which is the same
refusal expressed relatively. So the threshold is gone and the live question is
whether the relative comparator is set right.

What to read: `wasted` is the share of admitted experts evicted without one
route ever reading them -- the direct measure of speculative uploads that bought
nothing, which hit rate cannot show. It is NOT the objective. Decode tok/s is.
Those two disagreeing is the whole point (the threshold drove wasted 29% -> 0.1%
while making decode worse).

Arms are interleaved within each round because this box's clocks drift ~27%
across a long run. One runtime at a time: two do not coexist on a 12 GB card.
"""

from __future__ import annotations

import argparse
import os
import statistics

from flyweight.runtime_benchmark import measure_runtime_sample
from flyweight.v2 import V2Model

MIB = 1024 * 1024
ENV = "FLYWEIGHT_EXPERT_CACHE_STRICT_ADMISSION"

PROMPT = (
    "Explain how a mixture-of-experts transformer routes tokens to experts, "
    "and why the expert weights dominate decode latency once the model no "
    "longer fits in VRAM.\n\n"
)

COUNTERS = (
    "expert_cache_hits", "expert_cache_misses", "expert_cache_admissions",
    "expert_cache_evictions", "expert_cache_rejections",
    "expert_cache_unused_admissions",
)

ARMS = {"1": "strict (default)", "0": "legacy (ties admit)"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--gpu-cache-mib", type=int, default=8500)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--warmup-decode", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--moe-device", default="hybrid")
    args = parser.parse_args()

    arms = list(ARMS)
    # The .expert-history sidecar is an ambient input to placement -- restored
    # at prepare, rewritten at teardown -- so arm N+1 would start from what arm
    # N learned. Off for the whole comparison.
    os.environ["FLYWEIGHT_EXPERT_HISTORY"] = "0"
    model = V2Model(args.model)
    base = list(model.tokenize(PROMPT, capacity=args.context))
    repeats = (args.prompt_tokens + len(base) - 1) // len(base)
    prompt = (base * repeats)[:args.prompt_tokens]

    decode: dict[str, list[float]] = {a: [] for a in arms}
    prefill: dict[str, list[float]] = {a: [] for a in arms}
    counters: dict[str, dict[str, int]] = {}

    # Round 0 is discarded: an idle GPU on this box ramps its clocks over the
    # first seconds of work, and that ramp lands on whichever arm runs first.
    for round_index in range(args.repeats + 1):
        for arm in arms:
            os.environ[ENV] = arm
            with model.native_runtime(
                context_limit=args.context,
                gpu_cache_bytes=args.gpu_cache_mib * MIB,
                moe_device=args.moe_device,
            ) as runtime:
                runtime.prepare()
                sample = measure_runtime_sample(
                    runtime, prompt,
                    warmup_decode=args.warmup_decode,
                    decode_iterations=args.tokens,
                )
                info = runtime.info
            if not round_index:
                continue
            decode[arm].append(sample["decode_tokens_per_second"])
            prefill[arm].append(sample["native_prefill_tokens_per_second"])
            counters[arm] = {f: int(info[f]) for f in COUNTERS}
            print(f"  round {round_index} {ARMS[arm]:20s}: "
                  f"{decode[arm][-1]:6.2f} tok/s decode, "
                  f"{prefill[arm][-1]:7.1f} tok/s prefill", flush=True)
    os.environ.pop(ENV, None)

    baseline = statistics.median(decode[arms[0]])
    print()
    print(f"{'admission':>20}  {'decode':>16}  {'prefill':>9}  "
          f"{'hit rate':>8}  {'admits':>8}  {'wasted':>8}  {'evicts':>8}")
    for arm in arms:
        median = statistics.median(decode[arm])
        change = 100.0 * (median / baseline - 1.0) if baseline else 0.0
        counter = counters[arm]
        routed = counter["expert_cache_hits"] + counter["expert_cache_misses"]
        rate = 100.0 * counter["expert_cache_hits"] / routed if routed else 0.0
        admits = counter["expert_cache_admissions"]
        wasted = counter["expert_cache_unused_admissions"]
        print(f"{ARMS[arm]:>20}  {median:6.2f} tok/s {change:+5.1f}%  "
              f"{statistics.median(prefill[arm]):7.1f}  {rate:7.1f}%  "
              f"{admits:8d}  {(100.0 * wasted / admits if admits else 0.0):7.1f}%"
              f"  {counter['expert_cache_evictions']:8d}")
    spread = [max(v) / min(v) - 1.0 for v in decode.values() if min(v)]
    if spread:
        print(f"\nwithin-arm spread up to {100.0 * max(spread):.1f}% "
              f"over {args.repeats} rounds -- differences below that are noise")


if __name__ == "__main__":
    main()
