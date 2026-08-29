"""What the KV slots reserve, against what a session ever asks them to hold.

Every `--parallel` slot reserves `state_bytes` for the FULL context at prepare,
whether or not a conversation fills it. On a 12 GB card that reservation comes
straight out of the auto-fit expert cache, and the expert cache is where decode
spends its time (the MoE decode cost is 95-97% expert staging memcpy). So the
question phase 0 of plans/paged-kv-cache.md has to answer is not "how much is
reserved" -- that is arithmetic -- but "how much is ever touched".

The workload is the one that motivated per-slot arenas in the first place
(plans/parallel-kv-slots.md): a main conversation that grows turn over turn,
with short side-requests -- title/topic detection, a subagent, a quota summary
-- interleaved between its turns. Those side-requests are what the extra slots
exist for, and they are also why the extra slots sit nearly empty.

Reads the native occupancy high-water marks, which the runtime samples at every
request boundary (inside a request `position` only grows, so the boundary is
that request's own maximum, and recycling a slot only lowers the sum).
"""

from __future__ import annotations

import argparse

from flyweight.v2 import V2Model

MIB = 1024 * 1024

MAIN_TURN = (
    "Walk me through how a mixture-of-experts transformer decides which experts "
    "to run for a token, and why the weights for those experts dominate decode "
    "latency once the model no longer fits in VRAM. Then explain what changes "
    "when the batch is a prompt rather than a single token.\n\n"
)
SIDE_REQUESTS = (
    "Give this conversation a four-word title.\n",
    "Summarize the last exchange in one sentence.\n",
    "Is the user asking about hardware or about software? One word.\n",
)


def _tokens(model: V2Model, text: str, cap: int = 8192) -> list[int]:
    out = model.tokenize(text, capacity=cap)
    if not out:
        raise SystemExit("tokenizer returned no tokens")
    return list(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=48,
                        help="tokens generated per request")
    parser.add_argument("--gpu-cache-mib", type=int, default=0,
                        help="0 = auto-fit; pin it to make runs comparable")
    args = parser.parse_args()

    model = V2Model(args.model)
    runtime = model.native_qwen_runtime(
        context_limit=args.context,
        parallel_sequences=args.parallel,
        gpu_cache_bytes=args.gpu_cache_mib * MIB,
        moe_device="hybrid",
    )
    runtime.prepare()

    # The main conversation grows by its own transcript; the side-requests do
    # not, which is the whole point -- they land on other slots and stay short.
    conversation = ""
    for turn in range(args.turns):
        conversation += MAIN_TURN
        produced: list[int] = []
        runtime.generate(_tokens(model, conversation, cap=args.context),
                         args.tokens, produced.append)
        conversation += model.decode_token_bytes(produced).decode(
            "utf-8", errors="replace")
        side = SIDE_REQUESTS[turn % len(SIDE_REQUESTS)]
        runtime.generate(_tokens(model, side), 16, lambda _t: None)
        info = runtime.info
        print(f"  turn {turn + 1}: peak live {info['kv_peak_live_bytes'] / MIB:7.1f} MiB "
              f"over {info['kv_peak_tokens']} tokens "
              f"(largest slot {info['kv_peak_tokens_max']})", flush=True)

    info = runtime.info
    reserved = int(info["kv_reserved_bytes"])
    peak = int(info["kv_peak_live_bytes"])
    slots = int(info["expert_cache_slots"])
    slot_bytes = int(info["expert_cache_bytes"]) // slots if slots else 0

    print()
    print(f"model            {args.model}")
    print(f"context          {args.context}   slots {info['kv_slots']}   "
          f"samples {info['kv_occupancy_samples']}")
    print(f"reserved         {reserved / MIB:8.1f} MiB "
          f"({int(info['state_bytes']) / MIB:.1f} MiB per slot)")
    print(f"peak live        {peak / MIB:8.1f} MiB "
          f"({100.0 * peak / reserved:.1f}% of the reservation)")
    print(f"reclaimable      {(reserved - peak) / MIB:8.1f} MiB", end="")
    if slot_bytes:
        print(f"  = {(reserved - peak) // slot_bytes} expert slots "
              f"on top of today's {slots}")
    else:
        print()
    print(f"peak tokens      {info['kv_peak_tokens']} summed, "
          f"{info['kv_peak_tokens_max']} in the largest slot, "
          f"{args.parallel * args.context} reserved for")


if __name__ == "__main__":
    main()
