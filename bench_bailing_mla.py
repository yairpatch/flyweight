"""Is the BailingMoE3 prefill attention worth rewriting, and as what?

MLA is 48% of a prompt's time (FLYWEIGHT_BAILING_TILED_PROFILE on an 8192-token
prefill: mla 8.25s of 17.1s), and the two shipped implementations are both
bound by the same thing: each reads the latent cache once per query row per
head. The three-pass path reads it in the score kernel and again in accumulate;
the fused path -- already an online softmax, and measurably SLOWER, which is why
it is off by default -- collapses that into one pass but gives every
(row, head) its own block, so nothing is shared either.

The theory this benchmark exists to test: stage a chunk of the latent cache in
shared memory once and let a TILE of query rows consume it, for both the score
and the accumulate. MLA makes that unusually attractive -- the key and the value
are the same latent row, so one staged chunk serves both halves of attention.

Run: python bench_bailing_mla.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "native" / "tools"))
import kernel_harness as harness  # noqa: E402

# Ling-3.0-tiny, which is what the measurements above were taken on.
HEADS = 16
KV_LORA = 512
QK_NOPE = 128
QK_ROPE = 64
ROWS = 128  # the production prefill tile (gpu.prefill_rows)

PROTOTYPE = r"""
// Query-tiled MLA attention with the latent chunk staged in shared memory.
//
// One block owns QT query rows of one head. Queries are loaded into registers
// once; the latent cache is then walked in chunks of PT positions, each staged
// in shared memory and used TWICE -- once for the scores and once for the
// weighted accumulation -- which is the whole point: in the absorbed form the
// key and the value are the same row.
extern "C" __global__ __launch_bounds__(256)
void proto_mla_tiled_QT_VALUE_PT_VALUE(
    const float* __restrict__ projected, const float* __restrict__ query_rope,
    const float* __restrict__ latents, const float* __restrict__ rope_keys,
    float* __restrict__ accumulated,
    const int rows, const int base_position, const int heads,
    const int qk_nope, const int qk_rope, const int kv_lora
) {
    const int QT = QT_VALUE;  // query rows per block
    const int PT = PT_VALUE;  // positions staged per chunk
    const int WARPS = 8;
    const int head = blockIdx.y;
    const int row0 = blockIdx.x * QT;
    if (row0 >= rows) return;
    const int rows_here = min(QT, rows - row0);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    extern __shared__ float smem[];
    float* s_lat = smem;                       // PT * kv_lora
    float* s_rope = s_lat + PT * kv_lora;      // PT * qk_rope
    float* s_score = s_rope + PT * qk_rope;    // QT * PT
    float* s_max = s_score + QT * PT;          // QT
    float* s_sum = s_max + QT;                 // QT
    float* s_factor = s_sum + QT;              // QT

    // Each warp owns rows warp, warp+WARPS, ... and keeps their queries in
    // registers for the whole cache walk: read once, not once per chunk.
    const int per_warp = (QT + WARPS - 1) / WARPS;      // 2
    const int lanes_span = kv_lora / 32;                // 16 floats per lane
    float q_reg[(QT_VALUE + 7) / 8][16];
    float q_rope_reg[(QT_VALUE + 7) / 8][2];
    for (int slot = 0; slot < per_warp; ++slot) {
        const int r = warp + slot * WARPS;
        if (r >= rows_here) break;
        const float* q = projected + ((long long)(row0 + r) * heads + head) * kv_lora;
        for (int i = 0; i < lanes_span; ++i) q_reg[slot][i] = q[i * 32 + lane];
        const float* qr = query_rope + ((long long)(row0 + r) * heads + head) * qk_rope;
        for (int i = 0; i < qk_rope / 32; ++i) q_rope_reg[slot][i] = qr[i * 32 + lane];
    }

    const int columns = kv_lora / 256;                  // 2 columns per thread
    float acc[QT_VALUE][2];
    for (int r = 0; r < QT; ++r)
        for (int c = 0; c < columns; ++c) acc[r][c] = 0.0f;
    if (tid < QT) {
        s_max[tid] = -3.0e38f;
        s_sum[tid] = 0.0f;
        s_factor[tid] = 0.0f;
        // A row past the end of the tile must never contribute a weight.
        for (int p = 0; p < PT; ++p) s_score[tid * PT + p] = 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf((float)(qk_nope + qk_rope));
    const int last = base_position + row0 + rows_here;  // exclusive
    for (int p0 = 0; p0 < last; p0 += PT) {
        const int count = min(PT, last - p0);
        for (int i = tid; i < count * kv_lora; i += 256)
            s_lat[i] = latents[(long long)(p0 + i / kv_lora) * kv_lora + i % kv_lora];
        for (int i = tid; i < count * qk_rope; i += 256)
            s_rope[i] = rope_keys[(long long)(p0 + i / qk_rope) * qk_rope + i % qk_rope];
        __syncthreads();

        for (int slot = 0; slot < per_warp; ++slot) {
            const int r = warp + slot * WARPS;
            if (r >= rows_here) break;
            const int visible = base_position + row0 + r;   // inclusive
            for (int p = 0; p < count; ++p) {
                const int position = p0 + p;
                float total = 0.0f;
                if (position <= visible) {
                    for (int i = 0; i < lanes_span; ++i)
                        total += q_reg[slot][i] * s_lat[p * kv_lora + i * 32 + lane];
                    for (int i = 0; i < qk_rope / 32; ++i)
                        total += q_rope_reg[slot][i] * s_rope[p * qk_rope + i * 32 + lane];
                    for (int o = 16; o; o >>= 1)
                        total += __shfl_down_sync(0xffffffff, total, o);
                    total *= scale;
                } else {
                    total = -3.0e38f;
                }
                if (lane == 0) s_score[r * PT + p] = total;
            }
        }
        __syncthreads();

        // Online softmax: one thread per row folds this chunk into the running
        // max and sum, and publishes the factor the accumulators rescale by.
        if (tid < QT) {
            float peak = s_max[tid];
            for (int p = 0; p < count; ++p) peak = fmaxf(peak, s_score[tid * PT + p]);
            const float factor = (s_max[tid] <= -3.0e38f)
                ? 0.0f : __expf(s_max[tid] - peak);
            float total = 0.0f;
            for (int p = 0; p < count; ++p) {
                const float w = (tid < rows_here)
                    ? __expf(s_score[tid * PT + p] - peak) : 0.0f;
                s_score[tid * PT + p] = w;
                total += w;
            }
            s_max[tid] = peak;
            s_sum[tid] = s_sum[tid] * factor + total;
            s_factor[tid] = factor;
        }
        __syncthreads();

        // Position outermost, rows innermost: one shared load of the latent
        // serves every query row in the tile. The other way round re-read the
        // same value once per row -- 16x the shared traffic for the same FMAs.
        // Both loops over the tile are bounded by the compile-time QT, not by
        // the runtime row count: a dynamically indexed accumulator does not fit
        // in registers and the compiler spills the whole thing to local memory.
        #pragma unroll
        for (int r = 0; r < QT; ++r)
            #pragma unroll
            for (int c = 0; c < 2; ++c) acc[r][c] *= s_factor[r];
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int column = tid + c * 256;
            for (int p = 0; p < count; ++p) {
                const float value = s_lat[p * kv_lora + column];
                #pragma unroll
                for (int r = 0; r < QT; ++r)
                    acc[r][c] += s_score[r * PT + p] * value;
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < rows_here; ++r) {
        const float inverse = s_sum[r] > 0.0f ? 1.0f / s_sum[r] : 0.0f;
        for (int c = 0; c < columns; ++c) {
            const int column = tid + c * 256;
            accumulated[((long long)(row0 + r) * heads + head) * kv_lora + column] =
                acc[r][c] * inverse;
        }
    }
}
"""


def reference(projected, query_rope, latents, rope_keys, base_position):
    """Absorbed MLA attention, as the shipped three kernels define it."""
    rows = projected.shape[0]
    visible = base_position + rows
    scale = 1.0 / np.sqrt(QK_NOPE + QK_ROPE)
    flat = projected.reshape(rows * HEADS, KV_LORA)
    scores = (flat @ latents[:visible].T).reshape(rows, HEADS, visible)
    scores += (query_rope.reshape(rows * HEADS, QK_ROPE)
               @ rope_keys[:visible].T).reshape(rows, HEADS, visible)
    scores *= scale
    positions = cp.arange(visible)[None, :]
    allowed = positions <= (base_position + cp.arange(rows))[:, None]
    scores = cp.where(allowed[:, None, :], scores, -cp.inf)
    scores -= scores.max(axis=2, keepdims=True)
    weights = cp.exp(scores)
    weights /= weights.sum(axis=2, keepdims=True)
    return (weights.reshape(rows * HEADS, visible) @ latents[:visible]).reshape(
        rows, HEADS, KV_LORA)


VECTOR_SCORES = r"""
// The shipped score kernel, with float4 loads. Same math, same layout, same
// grid: the only question it answers is whether that kernel is limited by how
// fast it can issue loads rather than by the memory behind them.
extern "C" __global__
void proto_mla_scores_vec4(
    const float* __restrict__ projected, const float* __restrict__ query_rope,
    const float* __restrict__ latents, const float* __restrict__ rope_keys,
    float* __restrict__ scores,
    const int rows, const int base_position, const int capacity,
    const int heads, const int qk_nope, const int qk_rope, const int kv_lora
) {
    const int query_index = blockIdx.x;
    const int row = query_index / heads, head = query_index % heads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int position = blockIdx.y * 8 + warp;
    const int visible = base_position + row + 1;
    if (row >= rows || position >= visible) return;
    const float4* p = (const float4*)(projected +
        ((long long)row * heads + head) * kv_lora);
    const float4* latent = (const float4*)(latents + (long long)position * kv_lora);
    float total = 0.0f;
    for (int i = lane; i < kv_lora / 4; i += 32) {
        const float4 a = p[i], b = latent[i];
        total += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    const float4* qr = (const float4*)(query_rope +
        ((long long)row * heads + head) * qk_rope);
    const float4* kr = (const float4*)(rope_keys + (long long)position * qk_rope);
    for (int i = lane; i < qk_rope / 4; i += 32) {
        const float4 a = qr[i], b = kr[i];
        total += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int offset = 16; offset; offset >>= 1)
        total += __shfl_down_sync(0xffffffff, total, offset);
    if (lane == 0)
        scores[(long long)query_index * capacity + position] =
            total * rsqrtf((float)(qk_nope + qk_rope));
}
"""


def variant(qt: int, pt: int):
    """Compile the prototype at one tile shape."""
    source = (PROTOTYPE.replace("QT_VALUE", str(qt)).replace("PT_VALUE", str(pt)))
    name = f"proto_mla_tiled_{qt}_{pt}"
    shared = (pt * KV_LORA + pt * QK_ROPE + qt * pt + 3 * qt) * 4
    return harness.kernel(name, source), shared


def main() -> None:
    rng = cp.random.default_rng(7)
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"{name}, heads={HEADS} kv_lora={KV_LORA} rows={ROWS}")
    print("times are median ms for one 128-row tile, at settled clocks\n")
    harness.settle_clocks(3.0)

    scores_kernel = harness.kernel("bailing_mla_scores_rows")
    softmax_kernel = harness.kernel("bailing_mla_softmax_rows")
    accumulate_kernel = harness.kernel("bailing_mla_accumulate_rows")
    fused_kernel = harness.kernel("bailing_mla_fused_rows")
    vector_scores = harness.kernel("proto_mla_scores_vec4", VECTOR_SCORES)
    shapes = [(16, 16)]
    variants = {f"QT{qt}/PT{pt}": variant(qt, pt) for qt, pt in shapes}

    columns = ["scores", "vec4 scores", "softmax", "accumulate",
               "3-pass", "fused"] + list(variants)
    print(f"{'base':>7} " + " ".join(f"{c:>10}" for c in columns) + f" {'best':>16}")
    print("-" * (8 + 11 * len(columns) + 17))

    for base_position in (4096, 16384, 32768):
        capacity = base_position + ROWS
        projected = rng.standard_normal((ROWS, HEADS, KV_LORA), dtype=cp.float32) * 0.05
        query_rope = rng.standard_normal((ROWS, HEADS, QK_ROPE), dtype=cp.float32) * 0.05
        latents = rng.standard_normal((capacity, KV_LORA), dtype=cp.float32) * 0.05
        rope_keys = rng.standard_normal((capacity, QK_ROPE), dtype=cp.float32) * 0.05
        scores = cp.zeros((ROWS * HEADS, capacity), dtype=cp.float32)
        out = cp.zeros((ROWS, HEADS, KV_LORA), dtype=cp.float32)
        want = reference(projected, query_rope, latents, rope_keys, base_position)
        scale = float(cp.abs(want).max())
        timings, errors = {}, {}

        def check(result) -> float:
            return float(cp.abs(result - want).max()) / scale

        scores_kernel((ROWS * HEADS, (base_position + ROWS + 7) // 8), (256,),
                      (projected, query_rope, latents, rope_keys, scores, ROWS,
                       base_position, capacity, HEADS, QK_NOPE, QK_ROPE, KV_LORA))
        softmax_kernel((ROWS * HEADS, 1), (128,),
                       (scores, ROWS, base_position, capacity, HEADS))
        accumulate_kernel((ROWS * HEADS, (KV_LORA + 127) // 128), (128,),
                          (scores, latents, out, ROWS, base_position, capacity,
                           HEADS, KV_LORA))
        errors["3-pass"] = check(out)
        timings["scores"] = harness.time_kernel(scores_kernel, projected, query_rope, latents,
            rope_keys, scores, ROWS, base_position, capacity,
            HEADS, QK_NOPE, QK_ROPE, KV_LORA,
            grid=(ROWS * HEADS, (base_position + ROWS + 7) // 8), block=(256,))
        timings["softmax"] = harness.time_kernel(
            softmax_kernel, scores, ROWS, base_position, capacity, HEADS,
            grid=(ROWS * HEADS, 1), block=(128,))
        timings["accumulate"] = harness.time_kernel(
            accumulate_kernel, scores, latents, out, ROWS, base_position,
            capacity, HEADS, KV_LORA,
            grid=(ROWS * HEADS, (KV_LORA + 127) // 128), block=(128,))
        timings["3-pass"] = (timings["scores"] + timings["softmax"]
                             + timings["accumulate"])

        out[...] = 0
        vector_scores((ROWS * HEADS, (base_position + ROWS + 7) // 8), (256,),
                      (projected, query_rope, latents, rope_keys, scores, ROWS,
                       base_position, capacity, HEADS, QK_NOPE, QK_ROPE, KV_LORA))
        softmax_kernel((ROWS * HEADS, 1), (128,),
                       (scores, ROWS, base_position, capacity, HEADS))
        accumulate_kernel((ROWS * HEADS, (KV_LORA + 127) // 128), (128,),
                          (scores, latents, out, ROWS, base_position, capacity,
                           HEADS, KV_LORA))
        errors["vec4 scores"] = check(out)
        timings["vec4 scores"] = (
            harness.time_kernel(vector_scores, projected, query_rope, latents,
                                rope_keys, scores, ROWS, base_position, capacity,
                                HEADS, QK_NOPE, QK_ROPE, KV_LORA,
                                grid=(ROWS * HEADS, (base_position + ROWS + 7) // 8),
                                block=(256,))
            )

        out[...] = 0
        fused_kernel((ROWS * HEADS, 1), (512,),
                     (projected, query_rope, latents, rope_keys, out, ROWS,
                      base_position, HEADS, QK_NOPE, QK_ROPE, KV_LORA))
        errors["fused"] = check(out)
        timings["fused"] = harness.time_kernel(
            fused_kernel, projected, query_rope, latents, rope_keys, out, ROWS,
            base_position, HEADS, QK_NOPE, QK_ROPE, KV_LORA,
            grid=(ROWS * HEADS, 1), block=(512,))

        for label, (fn, shared) in variants.items():
            qt = int(label.split("/")[0][2:])
            grid = ((ROWS + qt - 1) // qt, HEADS)
            out[...] = 0
            fn(grid, (256,), (projected, query_rope, latents, rope_keys, out,
                              ROWS, base_position, HEADS, QK_NOPE, QK_ROPE,
                              KV_LORA), shared_mem=shared)
            errors[label] = check(out)
            timings[label] = harness.time_kernel(
                fn, projected, query_rope, latents, rope_keys, out, ROWS,
                base_position, HEADS, QK_NOPE, QK_ROPE, KV_LORA,
                grid=grid, block=(256,), shared=shared)

        best = min(variants, key=lambda k: timings[k])
        row = " ".join(f"{timings[c]:>10.2f}" for c in columns)
        print(f"{base_position:>7} {row} "
              f"{best} {timings['3-pass'] / timings[best]:>5.2f}x")
        bad = {k: v for k, v in errors.items() if v > 1e-3}
        if bad:
            print(f"        WRONG: {bad}")
        del projected, query_rope, latents, rope_keys, scores, out, want
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
