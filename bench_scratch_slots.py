"""Does shrinking the side slots actually make decode faster?

Slots reserve their whole context at prepare, and on a small card that
reservation comes out of the auto-fit expert cache. Since the MoE decode cost
is 95-97% expert staging memcpy, resident-expert coverage is the decode lever
-- so the chain to test is: smaller side slots -> more expert slots -> faster
decode. Anything that stops at "the reservation shrank" has measured
arithmetic, not a win.

Both arms get the SAME --gpu-cache-mib, which is the whole point: the budget is
fixed and the question is how it gets split. Run one arm per process (two
runtimes do not coexist on a 12 GB card), and note that an idle GPU on this box
ramps clocks slowly -- hence the warmup and the median over repeats.
"""

from __future__ import annotations

import argparse
import statistics
import time

from flyweight.v2 import V2Model

MIB = 1024 * 1024

PROMPT = (
    "Explain how a mixture-of-experts transformer routes tokens to experts, "
    "and why the expert weights dominate decode latency once the model no "
    "longer fits in VRAM.\n\n"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--scratch-context", type=int, default=0)
    parser.add_argument("--gpu-cache-mib", type=int, default=8500)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    model = V2Model(args.model)
    runtime = model.native_qwen_runtime(
        context_limit=args.context,
        parallel_sequences=args.parallel,
        scratch_context=args.scratch_context,
        gpu_cache_bytes=args.gpu_cache_mib * MIB,
        moe_device="hybrid",
    )
    runtime.prepare()
    info = runtime.info
    prompt = list(model.tokenize(PROMPT, capacity=args.context))

    # Where a decode token goes. The hybrid split runs the CPU experts while
    # the GPU kernel is in flight, so if raising the hit rate moves work onto
    # the slower pole this is where it shows: CPU compute down, total up, and
    # the difference waiting on the stream rather than paging.
    FIELDS = ("decode_calls", "decode_nanoseconds", "expert_compute_nanoseconds",
              "expert_page_nanoseconds", "route_wait_nanoseconds",
              "tail_wait_nanoseconds", "expert_cache_hits",
              "expert_cache_misses", "expert_cache_admissions")

    rates = []
    baseline: dict[str, int] = {}
    for index in range(args.repeats + 1):
        runtime.reset()
        produced: list[int] = []
        started = time.perf_counter()
        runtime.generate(prompt, args.tokens, produced.append)
        elapsed = time.perf_counter() - started
        if not index:
            # Warmup: clocks ramp and experts land in cache. Counters start here
            # so the fill traffic is not charged to the steady state.
            baseline = {f: int(runtime.info[f]) for f in FIELDS}
            continue
        rates.append(len(produced) / elapsed)
        print(f"  run {index}: {rates[-1]:6.2f} tok/s", flush=True)

    after = runtime.info
    delta = {f: int(after[f]) - baseline.get(f, 0) for f in FIELDS}
    print()
    print(f"scratch context  {args.scratch_context or 'off (symmetric slots)'}")
    print(f"reserved         {int(info['kv_reserved_bytes']) / MIB:8.1f} MiB")
    print(f"expert cache     {int(info['expert_cache_bytes']) / MIB:8.1f} MiB "
          f"({info['expert_cache_slots']} slots)")
    print(f"decode           {statistics.median(rates):6.2f} tok/s median "
          f"of {len(rates)} (min {min(rates):.2f}, max {max(rates):.2f})")
    hits, misses = delta["expert_cache_hits"], delta["expert_cache_misses"]
    if hits + misses:
        print(f"expert hit rate  {100.0 * hits / (hits + misses):5.1f}% "
              f"({hits} GPU / {misses} CPU routed experts)")
    calls = delta["decode_calls"]
    if not calls:
        return
    if not delta["decode_nanoseconds"]:
        print("(no timing: set FLYWEIGHT_TIMING=1)")
        return
    print(f"per token        {delta['decode_nanoseconds'] / calls / 1e6:7.3f} ms decode")
    for label, field in (("cpu experts", "expert_compute_nanoseconds"),
                         ("expert page", "expert_page_nanoseconds"),
                         ("route wait", "route_wait_nanoseconds"),
                         ("tail wait", "tail_wait_nanoseconds")):
        share = 100.0 * delta[field] / delta["decode_nanoseconds"]
        print(f"                 {delta[field] / calls / 1e6:7.3f} ms {label}"
              f"  ({share:4.1f}%)")
    print(f"admissions       {delta['expert_cache_admissions']} over {calls} tokens")


if __name__ == "__main__":
    main()
