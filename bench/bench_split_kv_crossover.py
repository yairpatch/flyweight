#!/usr/bin/env python3
"""Benchmark CUDA Split-KV Decode Attention vs Monolithic Attention on real GPU.

Characterizes:
1. Monolithic single-warp/block per query head: (heads, 1) CTAs
2. Split-KV partial reductions + merge pass: (heads, S) CTAs -> (heads, 1) merge
3. Context length sweep: 512, 1k, 2k, 4k, 8k, 16k, 32k, 65k tokens
4. Pinpoints the exact sequence length crossover threshold on hardware.
"""
import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    print("CuPy not installed; skipping GPU benchmark")
    sys.exit(0)

CUDA_CODE = r"""
#include <cuda_fp16.h>

extern "C" {

// 1. Monolithic Single-Pass Decode Attention
__global__ void attention_monolithic(
    const float* __restrict__ query,      // [heads, head_dim]
    const half* __restrict__ keys,        // [kv_heads, capacity, head_dim]
    const half* __restrict__ values,      // [kv_heads, capacity, head_dim]
    float* __restrict__ output,           // [heads, head_dim]
    const int heads,
    const int kv_heads,
    const int head_dim,
    const int tokens,
    const int capacity,
    const float scale
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int kv_head = head / (heads / kv_heads);
    const float* q = query + head * head_dim;

    // Allocate thread-local or shared scratch for scores
    // For large contexts, compute online softmax across tokens
    const half* k_base = keys + ((long long)kv_head * capacity) * head_dim;
    const half* v_base = values + ((long long)kv_head * capacity) * head_dim;

    float local_max = -1.0e30f;
    float local_denom = 0.0f;

    // Staged online softmax pass
    for (int t = 0; t < tokens; ++t) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * __half2float(k_base[t * head_dim + d]);
        }
        const float s = dot * scale;
        if (s > local_max) {
            const float rescale = expf(local_max - s);
            local_denom = local_denom * rescale + 1.0f;
            local_max = s;
        } else {
            local_denom += expf(s - local_max);
        }
    }

    const float inv_denom = local_denom > 0.0f ? 1.0f / local_denom : 0.0f;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < tokens; ++t) {
            float dot = 0.0f;
            for (int k_d = 0; k_d < head_dim; ++k_d) {
                dot += q[k_d] * __half2float(k_base[t * head_dim + k_d]);
            }
            const float s = dot * scale;
            const float w = expf(s - local_max) * inv_denom;
            acc += w * __half2float(v_base[t * head_dim + d]);
        }
        output[head * head_dim + d] = acc;
    }
}

// 2. Split-KV Partial Pass: (heads, S) CTAs
__global__ void attention_split_kv_partials(
    const float* __restrict__ query,      // [heads, head_dim]
    const half* __restrict__ keys,        // [kv_heads, capacity, head_dim]
    const half* __restrict__ values,      // [kv_heads, capacity, head_dim]
    float* __restrict__ partial_acc,      // [heads, splits, head_dim]
    float* __restrict__ partial_m,        // [heads, splits]
    float* __restrict__ partial_l,        // [heads, splits]
    const int heads,
    const int kv_heads,
    const int head_dim,
    const int tokens,
    const int capacity,
    const int splits,
    const float scale
) {
    const int head = blockIdx.x;
    const int split = blockIdx.y;
    if (head >= heads || split >= splits) return;

    const int kv_head = head / (heads / kv_heads);
    const float* q = query + head * head_dim;

    const int tokens_per_split = (tokens + splits - 1) / splits;
    const int t_start = split * tokens_per_split;
    const int t_end = min(tokens, t_start + tokens_per_split);
    if (t_start >= t_end) {
        if (threadIdx.x == 0) {
            partial_m[head * splits + split] = -1.0e30f;
            partial_l[head * splits + split] = 0.0f;
        }
        return;
    }

    const half* k_base = keys + ((long long)kv_head * capacity) * head_dim;
    const half* v_base = values + ((long long)kv_head * capacity) * head_dim;

    // Find local max and local denom
    float local_m = -1.0e30f;
    float local_l = 0.0f;

    for (int t = t_start; t < t_end; ++t) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * __half2float(k_base[t * head_dim + d]);
        }
        const float s = dot * scale;
        if (s > local_m) {
            local_l = local_l * expf(local_m - s) + 1.0f;
            local_m = s;
        } else {
            local_l += expf(s - local_m);
        }
    }

    if (threadIdx.x == 0) {
        partial_m[head * splits + split] = local_m;
        partial_l[head * splits + split] = local_l;
    }

    // Accumulate unnormalized values
    float* out_acc = partial_acc + (head * splits + split) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = t_start; t < t_end; ++t) {
            float dot = 0.0f;
            for (int k_d = 0; k_d < head_dim; ++k_d) {
                dot += q[k_d] * __half2float(k_base[t * head_dim + k_d]);
            }
            const float s = dot * scale;
            const float w = expf(s - local_m);
            acc += w * __half2float(v_base[t * head_dim + d]);
        }
        out_acc[d] = acc;
    }
}

// 3. Split-KV Merge Pass: (heads, 1) CTAs
__global__ void attention_split_kv_merge(
    const float* __restrict__ partial_acc,  // [heads, splits, head_dim]
    const float* __restrict__ partial_m,    // [heads, splits]
    const float* __restrict__ partial_l,    // [heads, splits]
    float* __restrict__ output,             // [heads, head_dim]
    const int heads,
    const int head_dim,
    const int splits
) {
    const int head = blockIdx.x;
    if (head >= heads) return;

    __shared__ float s_weights[64];
    __shared__ float s_global_denom;

    if (threadIdx.x == 0) {
        float g_max = -1.0e30f;
        for (int s = 0; s < splits; ++s) {
            const float m = partial_m[head * splits + s];
            const float l = partial_l[head * splits + s];
            if (l > 0.0f && m > g_max) g_max = m;
        }

        float g_denom = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float m = partial_m[head * splits + s];
            const float l = partial_l[head * splits + s];
            if (l > 0.0f && g_max > -1.0e29f) {
                const float w = expf(m - g_max);
                s_weights[s] = w;
                g_denom += l * w;
            } else {
                s_weights[s] = 0.0f;
            }
        }
        s_global_denom = g_denom;
    }
    __syncthreads();

    const float inv_denom = s_global_denom > 0.0f ? (1.0f / s_global_denom) : 0.0f;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float num = 0.0f;
        for (int s = 0; s < splits; ++s) {
            if (s_weights[s] != 0.0f) {
                num += partial_acc[(head * splits + s) * head_dim + d] * s_weights[s];
            }
        }
        output[head * head_dim + d] = num * inv_denom;
    }
}

}
"""

def bench(f, iters=50):
    for _ in range(5):
        f()
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        f()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / iters * 1000.0 # us

def main():
    print("==========================================================================================")
    print("  CUDA Decode Attention: Monolithic vs Split-KV Crossover Benchmark (RTX 5070 Ti)")
    print("==========================================================================================")
    device = cp.cuda.Device()
    print(f"Device: {device.id} ({cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()})")

    mod = cp.RawModule(code=CUDA_CODE, options=("--std=c++17",))
    k_mono = mod.get_function("attention_monolithic")
    k_split = mod.get_function("attention_split_kv_partials")
    k_merge = mod.get_function("attention_split_kv_merge")

    heads = 32
    kv_heads = 4
    head_dim = 128
    scale = np.float32(1.0 / np.sqrt(head_dim))
    block_dim = (128, 1, 1)

    print(f"Geometry: heads={heads}, kv_heads={kv_heads}, head_dim={head_dim} (FP16)")
    print(f"{'Tokens':>8} | {'Splits':>6} | {'Monolithic':>14} | {'Split-KV':>14} | {'Speedup':>10} | {'Winner':>12}")
    print("-" * 80)

    contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    for tokens in contexts:
        # Determine split count according to our heuristic
        if tokens < 1024:
            splits = 2
        elif tokens <= 4096:
            splits = 4
        elif tokens <= 16384:
            splits = 8
        else:
            splits = 16

        capacity = tokens
        query_gpu = cp.random.randn(heads, head_dim, dtype=cp.float32)
        keys_gpu = cp.random.randn(kv_heads, capacity, head_dim).astype(cp.float16)
        vals_gpu = cp.random.randn(kv_heads, capacity, head_dim).astype(cp.float16)
        out_mono = cp.zeros((heads, head_dim), dtype=cp.float32)
        out_split = cp.zeros((heads, head_dim), dtype=cp.float32)

        partial_acc = cp.zeros((heads, splits, head_dim), dtype=cp.float32)
        partial_m = cp.zeros((heads, splits), dtype=cp.float32)
        partial_l = cp.zeros((heads, splits), dtype=cp.float32)

        # Monolithic launch
        args_mono = (query_gpu, keys_gpu, vals_gpu, out_mono,
                     np.int32(heads), np.int32(kv_heads), np.int32(head_dim),
                     np.int32(tokens), np.int32(capacity), scale)

        def run_mono():
            k_mono((heads, 1, 1), block_dim, args_mono)

        # Split-KV launch (partials + merge)
        args_split = (query_gpu, keys_gpu, vals_gpu, partial_acc, partial_m, partial_l,
                      np.int32(heads), np.int32(kv_heads), np.int32(head_dim),
                      np.int32(tokens), np.int32(capacity), np.int32(splits), scale)
        args_merge = (partial_acc, partial_m, partial_l, out_split,
                      np.int32(heads), np.int32(head_dim), np.int32(splits))

        def run_split():
            k_split((heads, splits, 1), block_dim, args_split)
            k_merge((heads, 1, 1), block_dim, args_merge)

        # Measure
        us_mono = bench(run_mono, iters=20 if tokens > 8192 else 50)
        us_split = bench(run_split, iters=20 if tokens > 8192 else 50)

        speedup = us_mono / us_split
        winner = "Split-KV" if speedup > 1.05 else ("Monolithic" if speedup < 0.95 else "Tie")

        print(f"{tokens:8d} | {splits:6d} | {us_mono:11.2f} us | {us_split:11.2f} us | {speedup:9.2f}x | {winner:>12}")

    print("=" * 80)
    print("Crossover characterization completed.")

if __name__ == "__main__":
    main()
