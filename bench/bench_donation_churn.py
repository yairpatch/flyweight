"""Does prefix donation thrash when conversations outnumber slots?

Each donation is individually a strict improvement -- the newcomer gets what
taking the slot would have given it, and the donor survives. But that argument
is per-request. With more live conversations than slots, something must be
evicted on every turn, and donation changes WHICH slot gets recycled: instead
of taking the slot it matched, a request relocates to the LRU one. Repeated
across turns that could shuttle conversations between slots and leave every
one of them cold.

Three conversations round-robin over two slots, all sharing a system prompt
long enough to clear the prefix bar (so donation is eligible every turn).

The primary metric is `prefix_cache_reprefilled_tokens` -- a counter, so it is
free of the wall-clock drift this box has ~10% of. Wall time is reported too,
but the counter is what decides it.
"""

from __future__ import annotations

import argparse
import os
import time

from flyweight.v2 import V2Model

MIB = 1024 * 1024

SYSTEM = (
    "You are a careful systems engineer. You explain memory hierarchies, "
    "quantization formats, and GPU scheduling precisely, and you never guess "
    "at a number you have not measured. When asked for a plan, you give "
    "phases with explicit gates.\n\n"
)
# Distinct continuations, so the three conversations share only the opening.
TOPICS = (
    "Explain how a mixture-of-experts transformer routes tokens to experts.\n\n",
    "Describe how a paged KV cache differs from a contiguous per-slot arena.\n\n",
    "Walk through why quantized attention keys change long-context recall.\n\n",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--gpu-cache-mib", type=int, default=8500)
    parser.add_argument("--donate", type=int, choices=(0, 1), default=1)
    parser.add_argument("--system-repeats", type=int, default=48)
    parser.add_argument("--turn-repeats", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()

    os.environ["FLYWEIGHT_PREFIX_DONATE"] = str(args.donate)
    shared = SYSTEM * args.system_repeats

    model = V2Model(args.model)
    runtime = model.native_qwen_runtime(
        context_limit=args.context,
        parallel_sequences=args.parallel,
        gpu_cache_bytes=args.gpu_cache_mib * MIB,
        moe_device="hybrid",
    )
    runtime.prepare()

    # Three conversations, each growing by its own topic every round.
    conversations = [shared + topic * args.turn_repeats for topic in TOPICS]
    started_all = time.perf_counter()
    for round_index in range(args.rounds):
        for which, _ in enumerate(conversations):
            prompt = list(model.tokenize(conversations[which], args.context))
            started = time.perf_counter()
            produced: list[int] = []
            runtime.generate(prompt, args.tokens, produced.append)
            elapsed = time.perf_counter() - started
            info = runtime.info
            print(f"  round {round_index + 1} conv {which}: "
                  f"prompt={len(prompt):>6} "
                  f"reused={info['prefix_cache_last_reused_tokens']:>6} "
                  f"donations={info['prefix_donations']:>2} "
                  f"{elapsed:6.2f}s", flush=True)
            conversations[which] += model.decode_token_bytes(produced).decode(
                "utf-8", errors="replace")
            conversations[which] += TOPICS[which] * args.turn_repeats
    wall = time.perf_counter() - started_all

    info = runtime.info
    print()
    print(f"donation          {'on' if args.donate else 'OFF'}")
    print(f"conversations     {len(conversations)} over {args.parallel} slots, "
          f"{args.rounds} rounds")
    print(f"reprefilled       {info['prefix_cache_reprefilled_tokens']} tokens "
          f"(the counter that decides this)")
    print(f"reused            {info['prefix_cache_reused_tokens']} tokens")
    print(f"donations         {info['prefix_donations']} "
          f"({info['prefix_donated_tokens']} tokens)")
    print(f"wall              {wall:.1f}s")


if __name__ == "__main__":
    main()
