"""Does donating a shared prefix save the main conversation's next turn?

The interleave experiment from plans/parallel-kv-slots.md, with the twist that
motivated phase 1 of plans/paged-kv-cache.md: the side-request shares the main
conversation's SYSTEM PROMPT. That clears the absolute prefix bar, so routing
sends it to the main conversation's slot -- where it reuses the shared opening
and then overwrites everything past it. The main agent's next turn pays for
that.

What decides the outcome is how much of the main conversation lies PAST the
shared opening, because that is what taking the slot overwrites. So the main
conversation is grown over several turns first: a short conversation behind a
long system prompt has almost nothing at risk, and measuring that shape was how
the first version of this rule was found to be wrong.

Then: side-request, then the main agent's next turn -- with donation and with
FLYWEIGHT_PREFIX_DONATE=0. What matters is that last turn's reused-token count
and its prompt wall time; the first is the mechanism, the second is what a user
feels.

Run one arm per process (`--donate 0|1`); the runtimes do not coexist on a
12 GB card, and the arms must not share an auto-fit VRAM probe anyway.
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
MAIN_TURN = (
    "Explain how a mixture-of-experts transformer routes tokens to experts, "
    "and why the expert weights dominate decode latency once the model no "
    "longer fits in VRAM.\n\n"
)
SIDE_TURN = "Give this conversation a four-word title.\n"


def _tokens(model: V2Model, text: str, cap: int) -> list[int]:
    out = model.tokenize(text, capacity=cap)
    if not out:
        raise SystemExit("tokenizer returned no tokens")
    return list(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--gpu-cache-mib", type=int, default=8500)
    parser.add_argument("--donate", type=int, choices=(0, 1), default=1)
    parser.add_argument("--system-repeats", type=int, default=48,
                        help="how long to make the shared opening")
    parser.add_argument("--main-turns", type=int, default=4,
                        help="turns of main conversation past the opening")
    parser.add_argument("--turn-repeats", type=int, default=24,
                        help="how much each of those turns adds")
    parser.add_argument("--tokens", type=int, default=32)
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

    def run(label: str, text: str) -> str:
        prompt = _tokens(model, text, args.context)
        started = time.perf_counter()
        produced: list[int] = []
        runtime.generate(prompt, args.tokens, produced.append)
        elapsed = time.perf_counter() - started
        info = runtime.info
        print(f"  {label:<12} prompt={len(prompt):>6} "
              f"reused={info['prefix_cache_last_reused_tokens']:>6} "
              f"lcp_live={info['prefix_cache_last_lcp_live']:>6} "
              f"{elapsed:6.2f}s", flush=True)
        return model.decode_token_bytes(produced).decode(
            "utf-8", errors="replace")

    # Grow the main conversation so there is real work past the shared opening.
    turn = MAIN_TURN * args.turn_repeats
    conversation = shared
    for index in range(args.main_turns):
        conversation += turn
        conversation += run(f"main turn {index + 1}", conversation)
    run("side", shared + SIDE_TURN)
    conversation += turn
    run("main next", conversation)

    info = runtime.info
    print()
    print(f"donation      {'on' if args.donate else 'OFF'}")
    print(f"donations     {info['prefix_donations']} "
          f"({info['prefix_donated_tokens']} tokens)")
    print(f"reprefilled   {info['prefix_cache_reprefilled_tokens']} tokens total")
    print(f"peak live KV  {info['kv_peak_live_bytes'] / MIB:.1f} MiB of "
          f"{info['kv_reserved_bytes'] / MIB:.1f} MiB reserved")


if __name__ == "__main__":
    main()
