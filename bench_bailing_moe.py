"""What limits the BailingMoE3 expert matmuls during prefill?

The routed experts are 35% of a prompt's time (COLIBRI_BAILING_TILED_PROFILE on
an 8192-token prefill: gate 2.84s + down 2.16s + shared 0.37s of 17.1s), and the
kernels are already grouped per expert -- one block column per expert, walking
the rows routed to it. So the question is not "should we group" but what the
grouping is bound by.

The hypothesis this tests: with a 128-row prefill tile and top-8 of 128 experts,
each expert receives only ~8 rows, so its weights are read to serve almost no
work. Arithmetic intensity is then set by rows-per-expert, and the fix would be
to run the expert phase on a WIDER row batch than the attention phase rather
than to write a new kernel. If that is right, throughput per token should climb
steeply with the tile width and then flatten.

Run: python bench_bailing_moe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cupy as cp

sys.path.insert(0, str(Path(__file__).resolve().parent / "native" / "tools"))
import kernel_harness as harness  # noqa: E402

# Ling-3.0-tiny.
HIDDEN = 1536
EXPERT_SIZE = 512
EXPERTS = 128
TOP_K = 8
GROUPS = 8
GROUPS_USED = 4
WEIGHT_SCALE = 2.5


def q6k_row_bytes(elements: int) -> int:
    """Q6_K: 210 bytes per 256-element super-block."""
    assert elements % 256 == 0
    return elements // 256 * 210


def main() -> None:
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"{name}, experts={EXPERTS} top_k={TOP_K} "
          f"hidden={HIDDEN} expert_size={EXPERT_SIZE}")
    print("per-token cost of the routed experts, at settled clocks\n")

    rng = cp.random.default_rng(11)
    narrow = EXPERT_SIZE * q6k_row_bytes(HIDDEN)     # gate/up, per expert
    wide = HIDDEN * q6k_row_bytes(EXPERT_SIZE)       # down, per expert
    gate = cp.asarray(rng.integers(0, 256, EXPERTS * narrow, dtype=cp.uint8))
    up = cp.asarray(rng.integers(0, 256, EXPERTS * narrow, dtype=cp.uint8))
    down = cp.asarray(rng.integers(0, 256, EXPERTS * wide, dtype=cp.uint8))
    bias = cp.zeros(EXPERTS, dtype=cp.float32)

    route = harness.kernel("bailing_route_rows")
    quantize = harness.kernel("bailing_quantize_q8_rows")
    swiglu = harness.kernel("bailing_q6_q8_expert_swiglu_mmq_rows")
    accumulate = harness.kernel("bailing_q6_f32_expert_accumulate_mmq_rows")

    harness.settle_clocks(3.0)
    header = (f"{'rows':>6} {'rows/expert':>12} {'gate+up ms':>11} "
              f"{'down ms':>9} {'total ms':>9} {'us/token':>9} {'vs 128':>7}")
    print(header)
    print("-" * len(header))

    baseline = None
    for rows in (128, 256, 512, 1024, 2048):
        hidden_rows = rng.standard_normal((rows, HIDDEN), dtype=cp.float32) * 0.05
        logits = rng.standard_normal((rows, EXPERTS), dtype=cp.float32)
        selected = cp.zeros((rows, TOP_K), dtype=cp.int32)
        weights = cp.zeros((rows, TOP_K), dtype=cp.float32)
        counts = cp.zeros(EXPERTS, dtype=cp.int32)
        max_routes = rows * TOP_K
        routes = cp.zeros(EXPERTS * max_routes, dtype=cp.int32)
        quantized = cp.zeros(rows * HIDDEN, dtype=cp.int8)
        scales = cp.zeros(rows * HIDDEN // 32, dtype=cp.float16)
        activated = cp.zeros(rows * TOP_K * EXPERT_SIZE, dtype=cp.float32)
        output = cp.zeros((rows, HIDDEN), dtype=cp.float32)

        route((rows,), (256,), (logits, bias, selected, weights, counts, routes,
                               max_routes, rows, EXPERTS, TOP_K, GROUPS,
                               GROUPS_USED, 1, cp.float32(WEIGHT_SCALE)))
        quantize((HIDDEN // 32, rows), (32,),
                 (hidden_rows, quantized, scales, rows, HIDDEN))
        cp.cuda.runtime.deviceSynchronize()
        per_expert = float(counts.mean())

        gate_ms = harness.time_kernel(
            swiglu, gate, up, counts, routes, quantized, scales, activated,
            cp.uint64(narrow), max_routes, TOP_K, HIDDEN, EXPERT_SIZE,
            grid=((EXPERT_SIZE + 31) // 32, EXPERTS), block=(128,))
        down_ms = harness.time_kernel(
            accumulate, down, counts, routes, activated, weights, output,
            cp.uint64(wide), max_routes, TOP_K, EXPERT_SIZE, HIDDEN,
            grid=((HIDDEN + 31) // 32, EXPERTS), block=(128,))

        total = gate_ms + down_ms
        per_token = total / rows * 1e3
        if baseline is None:
            baseline = per_token
        print(f"{rows:>6} {per_expert:>12.1f} {gate_ms:>11.2f} {down_ms:>9.2f} "
              f"{total:>9.2f} {per_token:>9.1f} {baseline / per_token:>6.2f}x")

        del hidden_rows, logits, selected, weights, counts, routes
        del quantized, scales, activated, output
        cp.get_default_memory_pool().free_all_blocks()

    print("\nOne layer's routed experts. A prefill runs this 23 times per tile.")


if __name__ == "__main__":
    main()
