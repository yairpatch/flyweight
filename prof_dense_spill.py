#!/usr/bin/env python3
"""Attribute decode time for a dense model whose FFN is partly on the host.

Reports decode wall time against the host-SwiGLU time the runtime already
tracks, so the spilled-block cost can be separated from everything else.
"""

import argparse
import time

from colibri_next.v2 import V2Model, V2QwenRuntime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--gpu-cache-mib", type=int, default=0)
    # Fewer prefill checkpoints shrink the snapshot pool, which is what makes a
    # manual budget usable at all on a tight card -- see the sizing note below.
    parser.add_argument("--checkpoint-slots", type=int, default=4)
    arguments = parser.parse_args()

    model = V2Model(arguments.model)
    runtime = V2QwenRuntime(
        model,
        context_limit=arguments.context,
        gpu_cache_bytes=arguments.gpu_cache_mib * 1024 * 1024,
        prefill_checkpoint_slots=arguments.checkpoint_slots,
    )
    runtime.prepare()

    prompt = model.tokenize("Explain how a turbocharger works, step by step.")
    counts: list[int] = []
    runtime.generate(prompt, arguments.warmup, lambda _token: True)
    runtime.synchronize()

    before = runtime.info
    started = time.perf_counter()
    runtime.generate(prompt[:1], arguments.tokens, lambda _t: counts.append(1) or True)
    runtime.synchronize()
    elapsed = time.perf_counter() - started
    after = runtime.info

    def delta(field: str) -> int:
        return int(after[field]) - int(before[field])

    decoded = max(1, delta("decode_calls"))
    host_ms = delta("dense_host_nanoseconds") / 1e6
    decode_ms = delta("decode_nanoseconds") / 1e6
    print(f"host FFN blocks    : {after['host_ffn_layers']} "
          f"({int(after['host_ffn_bytes']) / (1 << 20):.0f} MiB)")
    print(f"decoded tokens     : {decoded}")
    print(f"wall               : {elapsed * 1000 / decoded:8.2f} ms/token "
          f"({decoded / elapsed:.2f} tok/s)")
    print(f"native decode      : {decode_ms / decoded:8.2f} ms/token")
    print(f"host dense FFN     : {host_ms / decoded:8.2f} ms/token "
          f"({100 * host_ms / max(1e-9, decode_ms):.1f}% of decode)")
    print(f"everything else    : {(decode_ms - host_ms) / decoded:8.2f} ms/token")
    for field in ("tail_wait_nanoseconds", "route_wait_nanoseconds",
                  "sampling_nanoseconds", "expert_page_nanoseconds"):
        print(f"{field:19s}: {delta(field) / 1e6 / decoded:8.2f} ms/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
