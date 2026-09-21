#pragma once

// Flyweight Split-KV Decode Attention Primitives
//
// Accelerates long-context decode attention (T >= 1024..65536) by splitting
// the sequence dimension across parallel threadblocks/SMs and merging
// partial online softmax accumulators.
//
// Key Advantages:
// 1. Full SM Saturation: Monolithic decode launches only Q_heads (e.g. 32)
//    threadblocks, leaving 50+ SM GPUs under-utilized. Split-KV launches
//    Q_heads * Splits (e.g. 32 * 8 = 256) threadblocks, saturating all SMs.
// 2. Exact Numerical Invariance: Online softmax re-weighting
//    w_s = exp(m_s - M), L = sum_s l_s * w_s guarantees bit-for-bit
//    mathematical equivalence to single-pass softmax attention.
// 3. Adaptive Split Scaling: Automatically selects split count based on context
//    length to avoid reduction overhead on short sequences.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define FLYWEIGHT_SPLIT_HD __host__ __device__ __forceinline__
#else
#define FLYWEIGHT_SPLIT_HD inline
#endif

namespace flyweight::split_kv {

inline constexpr int kMaxSplits = 64;

// Adaptive split count heuristic based on context length T and GPU SM capacity
FLYWEIGHT_SPLIT_HD int optimal_split_count(int context_tokens, int num_q_heads = 32, int num_sms = 50) {
    if (context_tokens < 1024) {
        return 1; // Monolithic path is optimal for short sequences (<1k)
    }
    int target_tokens_per_split = 1024;
    if (context_tokens <= 4096) {
        target_tokens_per_split = 512;
    } else if (context_tokens <= 16384) {
        target_tokens_per_split = 1024;
    } else {
        target_tokens_per_split = 2048;
    }
    int splits = (context_tokens + target_tokens_per_split - 1) / target_tokens_per_split;
    if (splits < 2) splits = 2;
    if (splits > 16) splits = 16;
    return splits;
}

// Partial statistics for one split of a head
struct SplitPartialStat {
    float m = -1.0e30f; // local maximum logit: max_{t in split} s_t
    float l = 0.0f;     // local denominator: sum_{t in split} exp(s_t - m)
};

// Merge partial split statistics into global maximum and denominator
// Returns global denominator L
FLYWEIGHT_SPLIT_HD float merge_split_stats(
    const SplitPartialStat* __restrict__ partials,
    int num_splits,
    float* __restrict__ split_weights,
    float& out_global_max
) {
    float global_max = -1.0e30f;
    for (int s = 0; s < num_splits; ++s) {
        if (partials[s].l > 0.0f && partials[s].m > global_max) {
            global_max = partials[s].m;
        }
    }
    out_global_max = global_max;

    float global_denom = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        if (partials[s].l > 0.0f && global_max > -1.0e29f) {
            const float weight = std::exp(partials[s].m - global_max);
            split_weights[s] = weight;
            global_denom += partials[s].l * weight;
        } else {
            split_weights[s] = 0.0f;
        }
    }
    return global_denom;
}

// Merge partial value accumulators across splits
// out_output = (sum_s partial_acc[s, d] * split_weights[s]) / global_denom
FLYWEIGHT_SPLIT_HD void merge_split_values(
    const float* __restrict__ partial_acc, // [num_splits, head_dim]
    const float* __restrict__ split_weights, // [num_splits]
    float global_denom,
    int num_splits,
    int head_dim,
    float* __restrict__ out_output // [head_dim]
) {
    const float inv_denom = global_denom > 0.0f ? (1.0f / global_denom) : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float num = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            if (split_weights[s] != 0.0f) {
                num += partial_acc[s * head_dim + d] * split_weights[s];
            }
        }
        out_output[d] = num * inv_denom;
    }
}

} // namespace flyweight::split_kv
