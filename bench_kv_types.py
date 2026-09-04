"""What each KV cache precision costs and buys, on one model, interleaved.

The cache type is never chosen for you (f16 until asked), so the question this
answers is what you give up by asking. Three things move together and only one
of them is speed:

  * KV bytes -- the reservation per slot, which is what frees VRAM for experts;
  * decode rate -- attention reads the whole cache every token, so a narrower
    cache is less to read, and a quantized one costs a decode to use;
  * the tokens themselves -- a quantized cache is different arithmetic, and the
    greedy continuation is reported against f16 so the trade is visible rather
    than assumed.

Arms are interleaved within each round because this box's clocks drift ~27%
across a long run, and one runtime exists at a time because two do not fit.
The expert history is off: it is an ambient input to expert placement, and
placement decides both speed and output.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

from flyweight.runtime_benchmark import measure_runtime_sample
from flyweight.v2 import V2Model

MIB = 1024 * 1024
TYPE_NAMES = {0: "f32", 1: "f16", 2: "bf16", 3: "q8_0",
              4: "turbo3", 5: "turbo4", 6: "auto"}

PROMPT = (
    "Explain how a mixture-of-experts transformer routes tokens to experts, "
    "and why the expert weights dominate decode latency once the model no "
    "longer fits in VRAM.\n\n"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--types", default="f16,bf16,q8_0,turbo3,turbo4,auto,f32")
    parser.add_argument("--context", type=int, default=128000)
    parser.add_argument("--gpu-cache-mib", type=int, default=9000)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup-decode", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--routed-moe", action="store_true", default=True)
    args = parser.parse_args()

    arms = [t.strip() for t in args.types.split(",") if t.strip()]
    os.environ["FLYWEIGHT_EXPERT_HISTORY"] = "0"
    model = V2Model(args.model)
    base = list(model.tokenize(PROMPT, capacity=args.context))
    repeats = (args.prompt_tokens + len(base) - 1) // len(base)
    prompt = (base * repeats)[:args.prompt_tokens]

    decode: dict[str, list[float]] = {a: [] for a in arms}
    prefill: dict[str, list[float]] = {a: [] for a in arms}
    facts: dict[str, dict[str, object]] = {}
    tokens: dict[str, list[int]] = {}
    failed: dict[str, str] = {}

    # Round 0 is discarded: an idle GPU ramps its clocks over the first seconds
    # of work, and that ramp lands on whichever arm runs first.
    for round_index in range(args.repeats + 1):
        for arm in arms:
            if arm in failed:
                continue
            try:
                with model.native_runtime(
                    context_limit=args.context,
                    gpu_cache_bytes=args.gpu_cache_mib * MIB,
                    routed_moe=args.routed_moe,
                    cache_type_k=arm, cache_type_v=arm,
                ) as runtime:
                    started = time.perf_counter()
                    runtime.prepare()
                    prepare_seconds = time.perf_counter() - started
                    sample = measure_runtime_sample(
                        runtime, prompt,
                        warmup_decode=args.warmup_decode,
                        decode_iterations=args.tokens,
                    )
                    info = runtime.info
            except Exception as error:  # noqa: BLE001 - an arm that cannot fit is a result
                failed[arm] = str(error).split(".")[0]
                print(f"  {arm:7s}: FAILED  {failed[arm]}", flush=True)
                continue
            if not round_index:
                continue
            decode[arm].append(sample["decode_tokens_per_second"])
            prefill[arm].append(sample["native_prefill_tokens_per_second"])
            tokens[arm] = sample["generated_tokens"]
            facts[arm] = {
                "resolved": "/".join((
                    TYPE_NAMES.get(int(info["resolved_cache_type_k"]), "?"),
                    TYPE_NAMES.get(int(info["resolved_cache_type_v"]), "?"),
                )),
                "kv_mib": int(info["kv_reserved_bytes"]) // MIB,
                "cache_mib": int(info["expert_cache_bytes"]) // MIB,
                "slots": int(info["expert_cache_slots"]),
                "prepare_s": prepare_seconds,
            }
            print(f"  round {round_index} {arm:7s}: "
                  f"{decode[arm][-1]:6.2f} tok/s decode, "
                  f"{prefill[arm][-1]:7.1f} prefill, "
                  f"{facts[arm]['kv_mib']:5d} MiB KV", flush=True)

    live = [a for a in arms if decode[a]]
    if not live:
        print("\nno arm completed")
        return
    reference = "f16" if "f16" in live else live[0]
    baseline = statistics.median(decode[reference])
    print()
    print(f"{'kv type':>8}  {'resolved':>11}  {'KV':>8}  {'expert cache':>14}  "
          f"{'decode':>16}  {'prefill':>9}  {'vs ' + reference:>9}")
    for arm in arms:
        if arm in failed:
            print(f"{arm:>8}  refused: {failed[arm]}")
            continue
        if not decode[arm]:
            continue
        f = facts[arm]
        median = statistics.median(decode[arm])
        change = 100.0 * (median / baseline - 1.0) if baseline else 0.0
        same = "same" if tokens[arm] == tokens.get(reference) else "DIFFERENT"
        print(f"{arm:>8}  {f['resolved']:>11}  {f['kv_mib']:5d} MiB  "
              f"{f['slots']:5d} / {f['cache_mib']:5d} MiB  "
              f"{median:6.2f} tok/s {change:+5.1f}%  "
              f"{statistics.median(prefill[arm]):7.1f}  {same:>9}")
    spread = [max(v) / min(v) - 1.0 for v in decode.values() if len(v) > 1 and min(v)]
    if spread:
        print(f"\nwithin-arm spread up to {100.0 * max(spread):.1f}% over "
              f"{args.repeats} rounds -- differences below that are noise")


if __name__ == "__main__":
    main()
