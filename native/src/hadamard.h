#pragma once

// Sylvester-Hadamard orthonormal rotation for activation outlier suppression.
//
// In transformer attention, Key and Query matrices often have severe activation
// outliers in fixed channel dimensions. By applying an orthonormal transform R
// (R^T R = I), dot products are strictly invariant:
//
//     <R q, R k> == <q, R^T R k> == <q, k>
//
// And for value vectors V, an inverse rotation on the final accumulated output:
//
//     sum_i p_i v_i == R^T (sum_i p_i (R v_i))
//
// Unlike dense rotations, the Sylvester-Hadamard transform of dimension 2^n:
// 1. Is symmetric and self-inverse: R^T = R, R * R = I.
// 2. Can be computed in O(d log d) operations with butterfly stages.
// 3. On CUDA GPUs, can be evaluated entirely in-register across warp lanes
//    using __shfl_xor_sync with zero shared/global memory traffic.
// 4. Requires no random seeds or hash-based sign flips, keeping it fully
//    deterministic and constant-free.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVRTC__)
#include <cuda_runtime.h>
#endif

namespace flyweight {

// Exact canonical Sylvester-Hadamard transform on CPU:
// H_1 = [1], H_2 = [[1, 1], [1, -1]], H_{2k} = [[H_k, H_k], [H_k, -H_k]].
// Normalized by 1 / sqrt(dim) so that H^T H = I.
inline void hadamard_sylvester_cpu(const float* input, float* output, int dim) {
    std::vector<float> temp(input, input + dim);
    for (int span = 1; span < dim; span <<= 1) {
        for (int start = 0; start < dim; start += (span << 1)) {
            for (int offset = 0; offset < span; ++offset) {
                const float low = temp[start + offset];
                const float high = temp[start + offset + span];
                temp[start + offset] = low + high;
                temp[start + offset + span] = low - high;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int i = 0; i < dim; ++i) {
        output[i] = temp[i] * scale;
    }
}

// Warp-level butterfly simulation on CPU (for testing bit-exact parity with GPU warp registers).
// Lane l in 0..31 holds elements: values[r] for index l + 32 * r.
template <int Columns>
inline void hadamard_warp_butterfly_cpu(float values[32][Columns]) {
    // Stage 1: 5 lane butterfly strides across the 32 warp lanes
    for (int stride = 1; stride <= 16; stride <<= 1) {
        for (int r = 0; r < Columns; ++r) {
            float next_col[32];
            for (int l = 0; l < 32; ++l) {
                const int peer_l = l ^ stride;
                const float val = values[l][r];
                const float peer = values[peer_l][r];
                next_col[l] = ((l & stride) == 0) ? (val + peer) : (peer - val);
            }
            for (int l = 0; l < 32; ++l) {
                values[l][r] = next_col[l];
            }
        }
    }

    // Stage 2: column butterfly across the Columns in-register
    for (int span = 1; span < Columns; span <<= 1) {
        for (int base = 0; base < Columns; base += (span << 1)) {
            for (int offset = 0; offset < span; ++offset) {
                for (int l = 0; l < 32; ++l) {
                    const float low = values[l][base + offset];
                    const float high = values[l][base + offset + span];
                    values[l][base + offset] = low + high;
                    values[l][base + offset + span] = low - high;
                }
            }
        }
    }

    // Stage 3: normalization
    const int dim = 32 * Columns;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < Columns; ++r) {
            values[l][r] *= scale;
        }
    }
}

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVRTC__)

template <int Columns>
__device__ __forceinline__ void hadamard_d32_columns_inplace(float (&values)[Columns], int lane) {
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            const float val = values[col];
            const float peer = __shfl_xor_sync(FullMask, val, stride);
            values[col] = (lane & stride) == 0 ? (val + peer) : (peer - val);
        }
    }
}

// In-register D=256 Sylvester-Hadamard transform.
// Warp lane l holds 8 values: values[r] = dimension (l + 32 * r).
__device__ __forceinline__ void hadamard_d256_inplace(float (&values)[8], int lane) {
    hadamard_d32_columns_inplace<8>(values, lane);
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low = values[base + offset];
                const float high = values[base + offset + span];
                values[base + offset] = low + high;
                values[base + offset + span] = low - high;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        values[r] *= 0.0625f; // 1 / 16
    }
}

// In-register D=128 Sylvester-Hadamard transform.
// Warp lane l holds 4 values: values[r] = dimension (l + 32 * r).
__device__ __forceinline__ void hadamard_d128_inplace(float (&values)[4], int lane) {
    hadamard_d32_columns_inplace<4>(values, lane);
#pragma unroll
    for (int span = 1; span < 4; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 4; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low = values[base + offset];
                const float high = values[base + offset + span];
                values[base + offset] = low + high;
                values[base + offset + span] = low - high;
            }
        }
    }
    constexpr float scale128 = 0.08838834764831845f; // 1 / sqrt(128)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        values[r] *= scale128;
    }
}

#endif // CUDA

} // namespace flyweight
