"""Compare KV cache types at *genuine* context occupancy.

The earlier turbo4-vs-f16 comparison ran a ~28-token prompt inside a 128K
context, so the KV cache was allocated for 128K but only ever held ~550 tokens
-- 0.4% occupancy. Expert-slot counts are allocation-driven and survive that,
but attention cost is paid per *resident* token, so the decode numbers did not.

This fills the context for real before measuring, and interleaves the
configurations so the GPU clock ramp cannot order-bias the result (see the
clock-ramp note: an idle GPU on this box runs kernels at ~1/6 sustained clock).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from colibri_next.v2 import V2Model


def _prompt_of_length(model: V2Model, length: int) -> list[int]:
    """A prompt of `length` real tokens, from repeated natural-language text.

    Repetition is deliberate: it keeps the tokenizer output stable and the
    routing realistic without needing a huge corpus on disk. Attention cost
    depends on how many tokens are resident, not on their novelty.
    """
    seed = model.tokenize(
        "Memory hierarchy design balances latency, capacity, and cost across "
        "registers, cache, main memory, and secondary storage. Locality of "
        "reference is the principle that makes the hierarchy effective. ",
        capacity=4096,
    )
    if not seed:
        raise SystemExit("tokenizer returned no tokens")
    repeats = (length + len(seed) - 1) // len(seed)
    return (seed * repeats)[:length]


def _run(model: V2Model, prompt: list[int], context: int, cache_type: str,
         decode_tokens: int, warmup: int) -> dict[str, float]:
    options = {
        "expert_mode": "hybrid",
        "context_limit": context,
        "cache_type_k": cache_type,
        "cache_type_v": cache_type,
    }
    with model.native_runtime(**options) as runtime:
        started = time.perf_counter()
        runtime.prepare()
        prepare_seconds = time.perf_counter() - started

        # Prefill via generate, then step with decode() so the counters can be
        # snapshotted *between* phases. Sampling them around the whole generate
        # call instead lets an 8K-token prefill -- 8192*40*8 route lookups --
        # swamp the few hundred from decode, which silently turns the expert
        # hit rate and per-token expert compute into prefill statistics.
        arrivals: list[float] = []
        prefill_started = time.perf_counter()

        def receive(token: int) -> None:
            arrivals.append(time.perf_counter())
            last[0] = token

        last = [0]
        runtime.generate(prompt, 1, receive)
        first_token_seconds = arrivals[0] - prefill_started

        token = last[0]
        for _ in range(warmup):
            token = runtime.decode(token)
        runtime.synchronize()

        before = runtime.info
        measured_started = time.perf_counter()
        for _ in range(decode_tokens):
            token = runtime.decode(token)
        runtime.synchronize()
        span = time.perf_counter() - measured_started
        info = runtime.info

    measured = [0.0] * decode_tokens
    calls = info["decode_calls"] - before["decode_calls"]
    lookups = (
        (info["expert_cache_hits"] - before["expert_cache_hits"])
        + (info["expert_cache_misses"] - before["expert_cache_misses"])
    )
    return {
        "cache_type": info["cache_type_k"],
        "prepare_seconds": prepare_seconds,
        "prefill_seconds": first_token_seconds,
        "prefill_tokens_per_second": len(prompt) / first_token_seconds,
        "decode_tokens_per_second": len(measured) / span,
        "decode_ms_per_token": 1e3 * span / len(measured),
        "expert_compute_ms_per_token": (
            info["expert_compute_nanoseconds"]
            - before["expert_compute_nanoseconds"]
        ) / 1e6 / max(calls, 1),
        "expert_cache_slots": info["expert_cache_slots"],
        "expert_cache_hit_rate": (
            info["expert_cache_hits"] - before["expert_cache_hits"]
        ) / max(lookups, 1),
        "state_bytes": info["state_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument(
        "--occupancy", type=int, default=0,
        help="prompt tokens to resident-fill the cache (default: context - slack)",
    )
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument(
        "--cache-types", default="f16,turbo4",
        help="comma-separated, compared pairwise",
    )
    args = parser.parse_args()

    slack = args.warmup + args.decode_tokens + 16
    occupancy = args.occupancy or (args.context - slack)
    if occupancy + slack > args.context:
        raise SystemExit("occupancy + decode budget exceeds --context")
    types = [name.strip() for name in args.cache_types.split(",") if name.strip()]

    results: dict[str, list[dict[str, float]]] = {name: [] for name in types}
    with V2Model(args.model) as model:
        prompt = _prompt_of_length(model, occupancy)
        # Interleaved, not grouped: grouping lets the clock ramp favour whichever
        # configuration runs second.
        for replicate in range(args.replicates):
            for name in types:
                sample = _run(
                    model, prompt, args.context, name,
                    args.decode_tokens, args.warmup,
                )
                results[name].append(sample)
                print(
                    f"[rep {replicate}] {name:>7} "
                    f"prefill={sample['prefill_tokens_per_second']:8.1f} tok/s "
                    f"decode={sample['decode_tokens_per_second']:6.2f} tok/s "
                    f"expert={sample['expert_compute_ms_per_token']:5.2f} ms "
                    f"hit={sample['expert_cache_hit_rate']:.3f} "
                    f"slots={sample['expert_cache_slots']}",
                    flush=True,
                )

    summary = {
        "context": args.context,
        "occupancy_tokens": occupancy,
        "decode_tokens": args.decode_tokens,
        "replicates": args.replicates,
        "median": {
            name: {
                key: statistics.median(sample[key] for sample in samples)
                for key in (
                    "prefill_tokens_per_second", "decode_tokens_per_second",
                    "decode_ms_per_token", "expert_compute_ms_per_token",
                    "expert_cache_hit_rate",
                )
            }
            | {
                "expert_cache_slots": samples[0]["expert_cache_slots"],
                "state_bytes": samples[0]["state_bytes"],
                "resolved": samples[0]["cache_type"],
            }
            for name, samples in results.items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
