#pragma once

namespace colibri::v2 {
inline constexpr char qwen_cuda_source[] = R"COLIBRI_CUDA(

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cub/block/block_radix_sort.cuh>

__device__ __forceinline__ float block_reduce_sum(float value) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

extern "C" __global__
void delta_conv_step(
    const float* mixed_qkv,
    const float* weights,
    float* state,
    float* output,
    const int channels,
    const int kernel_size
) {
    // Single-token variant: no cross-token recurrence, so one thread per
    // channel instead of one single-thread block per channel.
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    float* channel_state = state + channel * kernel_size;
    const float* channel_weights = weights + channel * kernel_size;
    for (int index = 0; index + 1 < kernel_size; ++index) {
        channel_state[index] = channel_state[index + 1];
    }
    channel_state[kernel_size - 1] = mixed_qkv[channel];
    float value = 0.0f;
    for (int index = 0; index < kernel_size; ++index) {
        value += channel_state[index] * channel_weights[index];
    }
    output[channel] = value / (1.0f + expf(-value));
}

extern "C" __global__
void delta_conv_sequence(
    const float* mixed_qkv,
    const float* weights,
    float* state,
    float* output,
    const int tokens,
    const int channels,
    const int kernel_size
) {
    const int channel = blockIdx.x;
    if (channel >= channels || threadIdx.x != 0) return;
    float* channel_state = state + channel * kernel_size;
    const float* channel_weights = weights + channel * kernel_size;
    for (int token = 0; token < tokens; ++token) {
        for (int index = 0; index + 1 < kernel_size; ++index) {
            channel_state[index] = channel_state[index + 1];
        }
        channel_state[kernel_size - 1] = mixed_qkv[token * channels + channel];
        float value = 0.0f;
        for (int index = 0; index < kernel_size; ++index) {
            value += channel_state[index] * channel_weights[index];
        }
        output[token * channels + channel] = value / (1.0f + expf(-value));
    }
}

extern "C" __global__
void delta_recurrent_sequence(
    const float* convolved,
    const float* gates,
    const float* beta_logits,
    const float* decay_logits,
    const float* a_log,
    const float* dt_bias,
    const float* norm_weights,
    float* state,
    float* output,
    const int tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float epsilon
) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= value_heads) return;
    const int key_head = head / (value_heads / key_heads);
    const int total_key_dim = key_heads * key_dim;
    const int conv_width = total_key_dim * 2 + value_heads * value_dim;
    __shared__ float query_inverse_norm;
    __shared__ float key_inverse_norm;
    __shared__ float decay_scale;
    __shared__ float beta;
    __shared__ float core_values[256];
    for (int token = 0; token < tokens; ++token) {
        const float* row = convolved + token * conv_width;
        const int key_offset = key_head * key_dim;
        if (lane == 0) {
            float query_square = 0.0f;
            float key_square = 0.0f;
            for (int index = 0; index < key_dim; ++index) {
                const float query = row[key_offset + index];
                const float key = row[total_key_dim + key_offset + index];
                query_square += query * query;
                key_square += key * key;
            }
            query_inverse_norm = rsqrtf(query_square + 1.0e-6f)
                * rsqrtf((float)key_dim);
            key_inverse_norm = rsqrtf(key_square + 1.0e-6f);
            beta = 1.0f / (1.0f + expf(-beta_logits[token * value_heads + head]));
            const float softplus_input =
                decay_logits[token * value_heads + head] + dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input : log1pf(expf(softplus_input));
            decay_scale = expf(-expf(a_log[head]) * softplus);
        }
        __syncthreads();
        float core = 0.0f;
        if (lane < value_dim) {
            float memory = 0.0f;
            for (int index = 0; index < key_dim; ++index) {
                const float key = row[total_key_dim + key_offset + index]
                    * key_inverse_norm;
                const int state_index =
                    (head * key_dim + index) * value_dim + lane;
                state[state_index] *= decay_scale;
                memory += state[state_index] * key;
            }
            const float value = row[
                total_key_dim * 2 + head * value_dim + lane
            ];
            const float delta = (value - memory) * beta;
            for (int index = 0; index < key_dim; ++index) {
                const float key = row[total_key_dim + key_offset + index]
                    * key_inverse_norm;
                const int state_index =
                    (head * key_dim + index) * value_dim + lane;
                state[state_index] += key * delta;
            }
        }
        __syncthreads();
        if (lane < value_dim) {
            for (int index = 0; index < key_dim; ++index) {
                const float query = row[key_offset + index] * query_inverse_norm;
                core += state[(head * key_dim + index) * value_dim + lane]
                    * query;
            }
        }
        core_values[lane] = lane < value_dim ? core : 0.0f;
        __syncthreads();
        __shared__ float core_inverse_rms;
        if (lane == 0) {
            float core_square = 0.0f;
            for (int index = 0; index < value_dim; ++index) {
                core_square += core_values[index] * core_values[index];
            }
            core_inverse_rms = rsqrtf(core_square / (float)value_dim + epsilon);
        }
        __syncthreads();
        if (lane < value_dim) {
            const int gate_index =
                token * value_heads * value_dim + head * value_dim + lane;
            const float gate = gates[gate_index];
            const float silu_gate = gate / (1.0f + expf(-gate));
            output[gate_index] = core * core_inverse_rms
                * norm_weights[lane] * silu_gate;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int pack_signed_chars(
    const int first,
    const int second,
    const int third,
    const int fourth
) {
    return (first & 255)
        | ((second & 255) << 8)
        | ((third & 255) << 16)
        | ((fourth & 255) << 24);
}

__device__ __forceinline__ int q4_q8_dot_block(
    const unsigned char* packed,
    const signed char* vector
) {
    int dot = 0;
    #pragma unroll
    for (int group = 0; group < 8; ++group) {
        const unsigned char first = packed[group * 2];
        const unsigned char second = packed[group * 2 + 1];
        const int weights = pack_signed_chars(
            (first & 15) - 8,
            (first >> 4) - 8,
            (second & 15) - 8,
            (second >> 4) - 8
        );
        const int activations = *((const int*)(vector + group * 4));
        dot = __dp4a(weights, activations, dot);
    }
    return dot;
}

extern "C" __global__
void quantize_q8_blocks(
    const float* input,
    signed char* output,
    __half* scales,
    const int elements
) {
    const int lane = threadIdx.x;
    const int index = blockIdx.x * 32 + lane;
    float value = index < elements ? input[index] : 0.0f;
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    if (lane == 0) scales[blockIdx.x] = __float2half(scale);
    if (index < elements) {
        const int quantized = max(-127, min(127, __float2int_rn(value / scale)));
        output[index] = (signed char)quantized;
    }
}

extern "C" __global__
void q4_q8_batched_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const signed char* vectors,
    const __half* vector_scales,
    float* output,
    const int rows,
    const int columns,
    const int expert_count,
    const int vector_count
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= rows || expert >= expert_count) return;
    const unsigned char* packed =
        (const unsigned char*)packed_addresses[expert];
    const __half* scales = (const __half*)scale_addresses[expert];
    const int blocks_per_row = columns >> 5;
    const int vector_index = vector_count == 1 ? 0 : expert;
    float partial = 0.0f;
    for (int block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
        const int weight_block = row * blocks_per_row + block;
        const int vector_block = vector_index * blocks_per_row + block;
        const int dot = q4_q8_dot_block(
            packed + weight_block * 16,
            vectors + vector_block * 32
        );
        partial += (float)dot
            * __half2float(scales[weight_block])
            * __half2float(vector_scales[vector_block]);
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[expert * rows + row] = partial;
}

extern "C" __global__
void q4_q8_batched_weighted_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const signed char* vectors,
    const __half* vector_scales,
    const float* routing_weights,
    float* output,
    const int rows,
    const int columns,
    const int expert_count
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int blocks_per_row = columns >> 5;
    float partial = 0.0f;
    for (int expert = 0; expert < expert_count; ++expert) {
        const unsigned char* packed =
            (const unsigned char*)packed_addresses[expert];
        const __half* scales = (const __half*)scale_addresses[expert];
        const int vector_offset = expert * blocks_per_row;
        float expert_partial = 0.0f;
        for (int block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
            const int weight_block = row * blocks_per_row + block;
            const int vector_block = vector_offset + block;
            const int dot = q4_q8_dot_block(
                packed + weight_block * 16,
                vectors + vector_block * 32
            );
            expert_partial += (float)dot
                * __half2float(scales[weight_block])
                * __half2float(vector_scales[vector_block]);
        }
        partial += routing_weights[expert] * expert_partial;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

__device__ __forceinline__ float block_reduce_max(float value) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    __shared__ float warp_maxima[8];
    if (lane == 0) warp_maxima[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warp_maxima[lane] : -3.402823466e+38F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
    }
    return value;
}

extern "C" __global__
void rms_norm(
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    const float* input,
    const float* weights,
    float* output,
    const int elements,
    const float epsilon,
    const int one_centered
) {
    float partial = 0.0f;
    for (int index = threadIdx.x; index < elements; index += blockDim.x) {
        partial += input[index] * input[index];
    }
    partial = block_reduce_sum(partial);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(partial / (float)elements + epsilon);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < elements; index += blockDim.x) {
        const float weight = one_centered ? 1.0f + weights[index] : weights[index];
        output[index] = input[index] * inverse_rms * weight;
    }
}

extern "C" __global__
void route_topk(
    const float* logits,
    int* selected,
    float* routing_weights,
    const int experts,
    const int top_k
) {
    extern __shared__ float probabilities[];
    float local_maximum = -3.402823466e+38F;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        local_maximum = fmaxf(local_maximum, logits[index]);
    }
    local_maximum = block_reduce_max(local_maximum);
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = local_maximum;
    __syncthreads();
    float local_sum = 0.0f;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        const float probability = expf(logits[index] - maximum);
        probabilities[index] = probability;
        local_sum += probability;
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float denominator;
    if (threadIdx.x == 0) denominator = local_sum;
    __syncthreads();
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        probabilities[index] /= denominator;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float selected_total = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_index = 0;
            float best_value = -1.0f;
            for (int expert = 0; expert < experts; ++expert) {
                if (probabilities[expert] > best_value) {
                    best_value = probabilities[expert];
                    best_index = expert;
                }
            }
            selected[rank] = best_index;
            routing_weights[rank] = best_value;
            selected_total += best_value;
            probabilities[best_index] = -1.0f;
        }
        for (int rank = 0; rank < top_k; ++rank) {
            routing_weights[rank] /= selected_total;
        }
    }
}

__device__ __forceinline__ unsigned long long sampling_sort_key(
    const float value, const int token
) {
    if (isnan(value) || token < 0) return 0;
    // Host sorting treats +0 and -0 as equal, so canonicalize zero before
    // encoding the float into a monotonically ordered unsigned key.
    const unsigned int bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    const unsigned int ordered = (bits & 0x80000000U)
        ? ~bits : (bits ^ 0x80000000U);
    // Descending composite order: higher logit first, then lower token ID.
    return (static_cast<unsigned long long>(ordered) << 32) |
        static_cast<unsigned int>(0x7fffffff - token);
}

__device__ __forceinline__ float sampling_key_value(
    const unsigned long long key
) {
    const unsigned int ordered = static_cast<unsigned int>(key >> 32);
    const unsigned int bits = (ordered & 0x80000000U)
        ? (ordered ^ 0x80000000U) : ~ordered;
    return __uint_as_float(bits);
}

// Sort 1,024 vocabulary logits per block and retain only that block's top-k.
extern "C" __global__
void sampling_block_topk_logits(
    const float* logits,
    int* output_indices,
    float* output_values,
    const int count,
    const int top_k
) {
    constexpr int items_per_thread = 4;
    using Sort = cub::BlockRadixSort<
        unsigned long long, 256, items_per_thread, int>;
    __shared__ typename Sort::TempStorage storage;
    unsigned long long keys[items_per_thread];
    int tokens[items_per_thread];
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int token = static_cast<int>(blockIdx.x) * 1024 +
            static_cast<int>(threadIdx.x) * items_per_thread + item;
        tokens[item] = token < count ? token : -1;
        keys[item] = token < count
            ? sampling_sort_key(logits[token], token) : 0;
    }
    Sort(storage).SortDescending(keys, tokens);
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * items_per_thread + item;
        if (rank < top_k) {
            const int output = static_cast<int>(blockIdx.x) * top_k + rank;
            output_indices[output] = tokens[item];
            output_values[output] = sampling_key_value(keys[item]);
        }
    }
}

// Merge top-k lists produced by the preceding stage. Repeated launches reduce
// the candidate set geometrically while retaining exact global top-k order.
extern "C" __global__
void sampling_block_topk_pairs(
    const int* input_indices,
    const float* input_values,
    int* output_indices,
    float* output_values,
    const int count,
    const int top_k
) {
    constexpr int items_per_thread = 4;
    using Sort = cub::BlockRadixSort<
        unsigned long long, 256, items_per_thread, int>;
    __shared__ typename Sort::TempStorage storage;
    unsigned long long keys[items_per_thread];
    int tokens[items_per_thread];
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int input = static_cast<int>(blockIdx.x) * 1024 +
            static_cast<int>(threadIdx.x) * items_per_thread + item;
        tokens[item] = input < count ? input_indices[input] : -1;
        keys[item] = input < count
            ? sampling_sort_key(input_values[input], tokens[item]) : 0;
    }
    Sort(storage).SortDescending(keys, tokens);
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * items_per_thread + item;
        if (rank < top_k) {
            const int output = static_cast<int>(blockIdx.x) * top_k + rank;
            output_indices[output] = tokens[item];
            output_values[output] = sampling_key_value(keys[item]);
        }
    }
}

extern "C" __global__
void route_topk_rows(
    const float* logits,
    int* selected,
    float* routing_weights,
    const int rows,
    const int experts,
    const int top_k
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float probabilities[];
    const int logits_offset = row * experts;
    const int output_offset = row * top_k;
    float local_maximum = -3.402823466e+38F;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        local_maximum = fmaxf(local_maximum, logits[logits_offset + index]);
    }
    local_maximum = block_reduce_max(local_maximum);
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = local_maximum;
    __syncthreads();
    float local_sum = 0.0f;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        const float probability = expf(logits[logits_offset + index] - maximum);
        probabilities[index] = probability;
        local_sum += probability;
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float denominator;
    if (threadIdx.x == 0) denominator = local_sum;
    __syncthreads();
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        probabilities[index] /= denominator;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float selected_total = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_index = 0;
            float best_value = -1.0f;
            for (int expert = 0; expert < experts; ++expert) {
                if (probabilities[expert] > best_value) {
                    best_value = probabilities[expert];
                    best_index = expert;
                }
            }
            selected[output_offset + rank] = best_index;
            routing_weights[output_offset + rank] = best_value;
            selected_total += best_value;
            probabilities[best_index] = -1.0f;
        }
        for (int rank = 0; rank < top_k; ++rank) {
            routing_weights[output_offset + rank] /= selected_total;
        }
    }
}

extern "C" __global__
void bf16_matvec(
    const unsigned short* weights,
    const float* vector,
    float* output,
    const int rows,
    const int columns
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    // 64-bit: this also serves the LM head, where rows*columns overruns int.
    const long long start = (long long)row * (long long)columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned int bits = ((unsigned int)weights[start + column]) << 16;
        partial += __uint_as_float(bits) * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

// Eight bf16 weights against eight f32 activations, decoded from one 128-bit
// load. A bf16 is the top half of the f32 with the same bits, so the low
// element of each 32-bit word is `word << 16` and the high element is
// `word & 0xffff0000`; no conversion instruction is needed.
__device__ __forceinline__ float bf16_dot8(const uint4 packed, const float* v) {
    const float4 lo = *(const float4*)v;
    const float4 hi = *(const float4*)(v + 4);
    float sum = __uint_as_float(packed.x << 16) * lo.x;
    sum += __uint_as_float(packed.x & 0xffff0000u) * lo.y;
    sum += __uint_as_float(packed.y << 16) * lo.z;
    sum += __uint_as_float(packed.y & 0xffff0000u) * lo.w;
    sum += __uint_as_float(packed.z << 16) * hi.x;
    sum += __uint_as_float(packed.z & 0xffff0000u) * hi.y;
    sum += __uint_as_float(packed.w << 16) * hi.z;
    sum += __uint_as_float(packed.w & 0xffff0000u) * hi.w;
    return sum;
}

// True when a row of `columns` bf16 weights can be walked with 128-bit loads:
// every row start stays 16-byte aligned and the activations line up with it.
__device__ __forceinline__ bool bf16_vectorizable(
    const unsigned short* weights, const float* vector, const int columns
) {
    return (columns & 7) == 0
        && (((unsigned long long)weights) & 15ull) == 0ull
        && (((unsigned long long)vector) & 15ull) == 0ull;
}

extern "C" __global__
void bf16_matvec_warp(
    const unsigned short* weights,
    const float* vector,
    float* output,
    const int rows,
    const int columns
) {
    // One warp per row, eight rows per 256-thread block. Against the
    // block-per-row form above this drops the shared-memory reduction and its
    // two barriers, and the 128-bit loads move 512 B per warp memory
    // instruction where the scalar `unsigned short` read moved 64 B.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    // 64-bit: this also serves the LM head, where rows*columns overruns int.
    const unsigned short* row_weights =
        weights + (long long)row * (long long)columns;
    float partial = 0.0f;
    if (bf16_vectorizable(row_weights, vector, columns)) {
        const uint4* packed = (const uint4*)row_weights;
        const int steps = columns >> 3;
        for (int step = lane; step < steps; step += 32)
            partial += bf16_dot8(packed[step], vector + step * 8);
    } else {
        for (int column = lane; column < columns; column += 32) {
            const unsigned int bits = ((unsigned int)row_weights[column]) << 16;
            partial += __uint_as_float(bits) * vector[column];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}

extern "C" __global__
void bf16_matmul(
    const unsigned short* weights,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= rows || token >= tokens) return;
    float partial = 0.0f;
    const int weight_start = row * columns;
    const int vector_start = token * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned int bits = ((unsigned int)weights[weight_start + column]) << 16;
        partial += __uint_as_float(bits) * vectors[vector_start + column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * rows + row] = partial;
}

extern "C" __global__
void bf16_matmul_small(
    const unsigned short* weights,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    // Small token batches (speculative verify): read each weight element
    // once and accumulate every token in registers, instead of re-streaming
    // the weight matrix per token like the (rows, tokens)-grid kernel.
    const int row = blockIdx.x;
    if (row >= rows || tokens > 8) return;
    float partial[8];
    for (int token = 0; token < 8; ++token) partial[token] = 0.0f;
    const int weight_start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned int bits =
            ((unsigned int)weights[weight_start + column]) << 16;
        const float weight = __uint_as_float(bits);
        for (int token = 0; token < tokens; ++token) {
            partial[token] += weight * vectors[token * columns + column];
        }
    }
    for (int token = 0; token < tokens; ++token) {
        const float total = block_reduce_sum(partial[token]);
        if (threadIdx.x == 0) output[token * rows + row] = total;
        __syncthreads();
    }
}

extern "C" __global__
void q4_matmul_small(
    const unsigned char* packed,
    const __half* scales,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    const int row = blockIdx.x;
    if (row >= rows || tokens > 8) return;
    float partial[8];
    for (int token = 0; token < 8; ++token) partial[token] = 0.0f;
    const int weight_start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = weight_start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        for (int token = 0; token < tokens; ++token) {
            partial[token] += weight * vectors[token * columns + column];
        }
    }
    for (int token = 0; token < tokens; ++token) {
        const float total = block_reduce_sum(partial[token]);
        if (threadIdx.x == 0) output[token * rows + row] = total;
        __syncthreads();
    }
}

extern "C" __global__
void q4_matmul(
    const unsigned char* packed,
    const __half* scales,
    const float* vectors,
    float* output,
    const int rows,
    const int columns,
    const int tokens
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= rows || token >= tokens) return;
    float partial = 0.0f;
    const int weight_start = row * columns;
    const int vector_start = token * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = weight_start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        partial += weight * vectors[vector_start + column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * rows + row] = partial;
}

extern "C" __global__
void rms_norm_rows(
    const float* input,
    const float* weights,
    float* output,
    const int rows,
    const int columns,
    const float epsilon,
    const int one_centered
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int start = row * columns;
    float partial = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[start + column];
        partial += value * value;
    }
    partial = block_reduce_sum(partial);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(partial / (float)columns + epsilon);
    }
    __syncthreads();
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = one_centered ? 1.0f + weights[column] : weights[column];
        output[start + column] = input[start + column] * inverse_rms * weight;
    }
}

extern "C" __global__
void silu_mul(
    const float* gate_up,
    float* output,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float gate = gate_up[index];
    const float exponential = gate >= 0.0f ? expf(-gate) : expf(gate);
    const float sigmoid = gate >= 0.0f
        ? 1.0f / (1.0f + exponential)
        : exponential / (1.0f + exponential);
    output[index] = gate * sigmoid * gate_up[elements + index];
}

extern "C" __global__
void scaled_add(
    float* target,
    const float* source,
    const float scale,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) target[index] += scale * source[index];
}

extern "C" __global__
void qwen_concat_pair(
    const float* left,
    const float* right,
    float* output,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = left[index];
    output[elements + index] = right[index];
}

extern "C" __global__
void qwen_copy_vector(
    const float* input,
    float* output,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index];
}

extern "C" __global__
void q4_batched_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const float* vector,
    float* output,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    const int rows,
    const int columns,
    const int expert_count
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= rows || expert >= expert_count) return;
    const unsigned char* packed =
        (const unsigned char*)packed_addresses[expert];
    const __half* scales = (const __half*)scale_addresses[expert];
    float partial = 0.0f;
    const int start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        partial += weight * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[expert * rows + row] = partial;
}

extern "C" __global__
void q4_selected_batched_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const int* selected_ids,
    const float* vector,
    float* output,
    const int rows,
    const int columns,
    const int selected_count
) {
    const int row = blockIdx.x;
    const int selected = blockIdx.y;
    if (row >= rows || selected >= selected_count) return;
    const int expert = selected_ids[selected];
    const unsigned char* packed =
        (const unsigned char*)packed_addresses[expert];
    const __half* scales = (const __half*)scale_addresses[expert];
    float partial = 0.0f;
    const int start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        partial += (float)(nibble - 8)
            * __half2float(scales[block]) * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[selected * rows + row] = partial;
}

extern "C" __global__
void silu_mul_batched(
    const float* gate_up,
    float* output,
    const int intermediate_size,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int expert = index / intermediate_size;
    const int within_expert = index - expert * intermediate_size;
    const int gate_offset = expert * intermediate_size * 2;
    const float gate = gate_up[gate_offset + within_expert];
    const float exponential = gate >= 0.0f ? expf(-gate) : expf(gate);
    const float sigmoid = gate >= 0.0f
        ? 1.0f / (1.0f + exponential)
        : exponential / (1.0f + exponential);
    output[index] = gate * sigmoid
        * gate_up[gate_offset + intermediate_size + within_expert];
}

extern "C" __global__
void q4_silu_batched(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const float* vector,
    float* output,
    const int intermediate_size,
    const int columns,
    const int expert_count
) {
    const int intermediate = blockIdx.x;
    const int expert = blockIdx.y;
    if (intermediate >= intermediate_size || expert >= expert_count) return;
    const unsigned char* packed =
        (const unsigned char*)packed_addresses[expert];
    const __half* scales = (const __half*)scale_addresses[expert];
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    const int gate_row = intermediate;
    const int up_row = intermediate_size + intermediate;
    const int gate_start = gate_row * columns;
    const int up_start = up_row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int gate_index = gate_start + column;
        const int gate_block = gate_index >> 5;
        const int gate_within = gate_index & 31;
        const unsigned char gate_byte =
            packed[gate_block * 16 + (gate_within >> 1)];
        const int gate_nibble = (gate_within & 1)
            ? (gate_byte >> 4) : (gate_byte & 15);
        gate_partial += (float)(gate_nibble - 8)
            * __half2float(scales[gate_block]) * vector[column];

        const int up_index = up_start + column;
        const int up_block = up_index >> 5;
        const int up_within = up_index & 31;
        const unsigned char up_byte =
            packed[up_block * 16 + (up_within >> 1)];
        const int up_nibble = (up_within & 1)
            ? (up_byte >> 4) : (up_byte & 15);
        up_partial += (float)(up_nibble - 8)
            * __half2float(scales[up_block]) * vector[column];
    }
    gate_partial = block_reduce_sum(gate_partial);
    up_partial = block_reduce_sum(up_partial);
    if (threadIdx.x == 0) {
        const float exponential = gate_partial >= 0.0f
            ? expf(-gate_partial) : expf(gate_partial);
        const float sigmoid = gate_partial >= 0.0f
            ? 1.0f / (1.0f + exponential)
            : exponential / (1.0f + exponential);
        output[expert * intermediate_size + intermediate] =
            gate_partial * sigmoid * up_partial;
    }
}

extern "C" __global__
void q4_batched_weighted_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const float* vectors,
    const float* routing_weights,
    float* output,
    const int rows,
    const int columns,
    const int expert_count
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    const int start = row * columns;
    for (int expert = 0; expert < expert_count; ++expert) {
        const unsigned char* packed =
            (const unsigned char*)packed_addresses[expert];
        const __half* scales = (const __half*)scale_addresses[expert];
        const float* vector = vectors + expert * columns;
        const float route = routing_weights[expert];
        for (int column = threadIdx.x; column < columns; column += blockDim.x) {
            const int index = start + column;
            const int block = index >> 5;
            const int within_block = index & 31;
            const unsigned char byte = packed[block * 16 + (within_block >> 1)];
            const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
            const float weight =
                (float)(nibble - 8) * __half2float(scales[block]);
            partial += route * weight * vector[column];
        }
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q4_selected_weighted_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const int* selected_ids,
    const float* vectors,
    const float* routing_weights,
    float* output,
    const int rows,
    const int columns,
    const int selected_count
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    const int start = row * columns;
    for (int selected = 0; selected < selected_count; ++selected) {
        const int expert = selected_ids[selected];
        const unsigned char* packed =
            (const unsigned char*)packed_addresses[expert];
        const __half* scales = (const __half*)scale_addresses[expert];
        const float* vector = vectors + selected * columns;
        const float route = routing_weights[selected];
        for (int column = threadIdx.x; column < columns; column += blockDim.x) {
            const int index = start + column;
            const int block = index >> 5;
            const int within_block = index & 31;
            const unsigned char byte = packed[block * 16 + (within_block >> 1)];
            const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
            const float weight = (float)(nibble - 8)
                * __half2float(scales[block]);
            partial += route * weight * vector[column];
        }
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q4_matvec(
    const unsigned char* packed,
    const __half* scales,
    const float* vector,
    float* output,
    const int rows,
    const int columns
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    const int start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = start + column;
        const int block = index >> 5;
        const int within_block = index & 31;
        const unsigned char byte = packed[block * 16 + (within_block >> 1)];
        const int nibble = (within_block & 1) ? (byte >> 4) : (byte & 15);
        const float weight = (float)(nibble - 8) * __half2float(scales[block]);
        partial += weight * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q8_matvec_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        // GGML dimension 0 is contiguous: each logical output row contains
        // input_size values even though GGUF reports [input, output].
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(
            *((const __half*)(packed + block * 34))
        );
        const signed char value = *((const signed char*)(packed + block * 34 + 2 + within));
        partial += ((float)value * scale) * vector[input];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

// Direct GGML Q4_0 readers used by Gemma 4 QAT checkpoints. Each 32-value
// block is [f16 scale | 16 packed nibbles], with low nibbles for values 0..15
// and high nibbles for 16..31.
__device__ __forceinline__ float ggml_q4_0_load(const unsigned char* packed, long long absolute) {
    const unsigned char* block=packed+(absolute>>5)*18;
    const int within=(int)(absolute&31);
    const unsigned char byte=block[2+(within&15)];
    const int quant=within<16?(byte&15):(byte>>4);
    return (float)(quant-8)*__half2float(*((const __half*)block));
}

extern "C" __global__ void gemma_q4_0_matvec(
    const unsigned char* packed,const float* vector,float* output,
    const int input_size,const int output_size
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(row>=output_size)return;
    float sum=0.0f;
    for(int input=lane;input<input_size;input+=32)
        sum+=ggml_q4_0_load(packed,(long long)row*input_size+input)*vector[input];
    for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0)output[row]=sum;
}

extern "C" __global__ void gemma_q4_0_embedding(
    const unsigned char* packed,float* output,const int token,
    const int hidden,const float scale
) {
    for(int d=blockIdx.x*blockDim.x+threadIdx.x;d<hidden;d+=blockDim.x*gridDim.x)
        output[d]=ggml_q4_0_load(packed,(long long)token*hidden+d)*scale;
}

__device__ __forceinline__ float gemma_gelu(float x) {
    return 0.5f*x*(1.0f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x)));
}

extern "C" __global__ void gemma_q4_0_geglu(
    const unsigned char* gate,const unsigned char* up,const float* input,float* output,
    const int input_size,const int intermediate
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(row>=intermediate)return;
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    float g=0.0f,u=0.0f;
    for(int i=lane;i<input_size;i+=32){
        const float x=input[i];
        g+=ggml_q4_0_load(gate,(long long)row*input_size+i)*x;
        u+=ggml_q4_0_load(up,(long long)row*input_size+i)*x;
    }
    for(int offset=16;offset;offset>>=1){g+=__shfl_down_sync(0xffffffff,g,offset);u+=__shfl_down_sync(0xffffffff,u,offset);}
    if(lane==0)output[row]=gemma_gelu(g)*u;
}

extern "C" __global__ void gemma_q4_0_grouped_geglu(
    const unsigned long long* gate_up_tables,const float* input,float* output,
    const int input_size,const int intermediate,const int expert_count
) {
    const int expert=blockIdx.y;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(expert>=expert_count||row>=intermediate)return;
    const unsigned char* gate_up=(const unsigned char*)gate_up_tables[expert];
    float g=0.0f,u=0.0f;
    for(int i=lane;i<input_size;i+=32){
        const float x=input[i];
        g+=ggml_q4_0_load(gate_up,(long long)row*input_size+i)*x;
        u+=ggml_q4_0_load(gate_up,(long long)(row+intermediate)*input_size+i)*x;
    }
    for(int offset=16;offset;offset>>=1){g+=__shfl_down_sync(0xffffffff,g,offset);u+=__shfl_down_sync(0xffffffff,u,offset);}
    if(lane==0)output[(long long)expert*intermediate+row]=gemma_gelu(g)*u;
}

extern "C" __global__ void gemma_q4_0_grouped_accumulate(
    const unsigned long long* down_tables,const float* activated,float* output,
    const float* routing_weights,const int intermediate,const int hidden,
    const int expert_count
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(row>=hidden)return;
    float total=0.0f;
    for(int expert=0;expert<expert_count;++expert){
        const unsigned char* down=(const unsigned char*)down_tables[expert];
        const float* vector=activated+(long long)expert*intermediate;
        float sum=0.0f;
        for(int i=lane;i<intermediate;i+=32)
            sum+=ggml_q4_0_load(down,(long long)row*intermediate+i)*vector[i];
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane==0)total+=routing_weights[expert]*sum;
    }
    if(lane==0)output[row]+=total;
}

extern "C" __global__ void gemma_head_norm_rope(
    const float* projected,const float* weights,float* output,
    const int heads,const int head_dim,const int rotary_dim,const int position,
    const float theta,const float epsilon,const float* freq_factors
) {
    const int head=blockIdx.x;if(head>=heads)return;
    const float* src=projected+head*head_dim;float* dst=output+head*head_dim;
    float sum=0.0f;for(int d=threadIdx.x;d<head_dim;d+=blockDim.x)sum+=src[d]*src[d];
    sum=block_reduce_sum(sum);__shared__ float inv;if(threadIdx.x==0)inv=rsqrtf(sum/head_dim+epsilon);__syncthreads();
    const int half=rotary_dim/2;
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float value=src[d]*inv*weights[d];
        if(d<rotary_dim){
            const int pair=d<half?d:d-half;const int other=d<half?d+half:d-half;
            const float other_value=src[other]*inv*weights[other];
            const float factor=freq_factors?freq_factors[pair]:1.0f;
            const float angle=(float)position/(powf(theta,2.0f*pair/rotary_dim)*factor);
            value=d<half?value*cosf(angle)-other_value*sinf(angle):other_value*sinf(angle)+value*cosf(angle);
        }
        dst[d]=value;
    }
}

extern "C" __global__ void gemma_head_rms(
    const float* input,float* output,const int heads,const int head_dim,const float epsilon
) {
    const int head=blockIdx.x;if(head>=heads)return;const float*src=input+head*head_dim;float*dst=output+head*head_dim;
    float sum=0.0f;for(int d=threadIdx.x;d<head_dim;d+=blockDim.x)sum+=src[d]*src[d];
    sum=block_reduce_sum(sum);__shared__ float inv;if(threadIdx.x==0)inv=rsqrtf(sum/head_dim+epsilon);__syncthreads();
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x)dst[d]=src[d]*inv;
}

extern "C" __global__ void gemma_router_input(
    const float* input,const float* scale,float* output,const int elements,const float epsilon
) {
    float sum=0.0f;for(int i=threadIdx.x;i<elements;i+=blockDim.x)sum+=input[i]*input[i];
    sum=block_reduce_sum(sum);__shared__ float factor;if(threadIdx.x==0)factor=rsqrtf(sum/elements+epsilon)*rsqrtf((float)elements);__syncthreads();
    for(int i=threadIdx.x;i<elements;i+=blockDim.x)output[i]=input[i]*scale[i]*factor;
}

extern "C" __global__ void gemma_scale_vector(float* values,const float* scale,const int elements) {
    const float factor=scale[0];for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<elements;i+=blockDim.x*gridDim.x)values[i]*=factor;
}

extern "C" __global__ void gemma_q4_0_lm_argmax(
    const unsigned char* packed,const float* input,unsigned long long* winner,
    const int hidden,const int vocabulary
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,token=blockIdx.x*8+warp;
    if(token>=vocabulary)return;float sum=0.0f;
    for(int i=lane;i<hidden;i+=32)sum+=ggml_q4_0_load(packed,(long long)token*hidden+i)*input[i];
    for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0){const unsigned int ordered=__float_as_uint(sum)^(sum>=0.0f?0x80000000u:0xffffffffu);const unsigned long long candidate=((unsigned long long)ordered<<32)|(0xffffffffu-(unsigned int)token);atomicMax(winner,candidate);}
}

extern "C" __global__
void q8_matvec_transposed_warp(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = lane; input < input_size; input += 32) {
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(
            *((const __half*)(packed + block * 34))
        );
        const signed char value = *((const signed char*)(
            packed + block * 34 + 2 + within
        ));
        partial += ((float)value * scale) * vector[input];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}

extern "C" __global__
void q8_swiglu_transposed_warp(
    const unsigned char* gate_packed,
    const unsigned char* up_packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = lane; input < input_size; input += 32) {
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float value = vector[input];
        const float gate_scale = __half2float(
            *((const __half*)(gate_packed + block * 34))
        );
        const float up_scale = __half2float(
            *((const __half*)(up_packed + block * 34))
        );
        const signed char gate_quant = *((const signed char*)(
            gate_packed + block * 34 + 2 + within
        ));
        const signed char up_quant = *((const signed char*)(
            up_packed + block * 34 + 2 + within
        ));
        gate += ((float)gate_quant * gate_scale) * value;
        up += ((float)up_quant * up_scale) * value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffff, gate, offset);
        up += __shfl_down_sync(0xffffffff, up, offset);
    }
    if (lane == 0) {
        const float exponential = gate >= 0.0f ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0f
            ? 1.0f / (1.0f + exponential)
            : exponential / (1.0f + exponential);
        output[row] = gate * sigmoid * up;
    }
}

extern "C" __global__
void bf16_lm_head_argmax_warp(
    const unsigned short* weights,
    const float* vector,
    unsigned long long* winners,
    const int input_size,
    const int output_size
) {
    // Same warp-per-row / block-fold structure as q8_lm_head_argmax_warp; only
    // the weight decode differs. bf16 lm_heads previously fell through to the
    // Q8_0 kernel, which reads the table as 34-byte blocks and returns noise.
    __shared__ unsigned long long warp_best[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (lane == 0) warp_best[warp] = 0ull;
    __syncthreads();
    if (row < output_size) {
        float partial = 0.0f;
        const unsigned short* row_weights =
            weights + (long long)row * (long long)input_size;
        if (bf16_vectorizable(row_weights, vector, input_size)) {
            const uint4* packed = (const uint4*)row_weights;
            const int steps = input_size >> 3;
            for (int step = lane; step < steps; step += 32)
                partial += bf16_dot8(packed[step], vector + step * 8);
        } else {
            for (int input = lane; input < input_size; input += 32) {
                const unsigned int bits =
                    ((unsigned int)row_weights[input]) << 16;
                partial += __uint_as_float(bits) * vector[input];
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (lane == 0) {
            const unsigned int bits = __float_as_uint(partial);
            const unsigned int ordered = bits ^ (
                ((int)bits < 0) ? 0xffffffffu : 0x80000000u
            );
            warp_best[warp] =
                ((unsigned long long)ordered << 32)
                | (unsigned int)(0xffffffffu - (unsigned int)row);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long best = warp_best[0];
        for (int i = 1; i < 8; ++i) best = max(best, warp_best[i]);
        atomicMax(winners, best);
    }
}

extern "C" __global__
void q8_lm_head_argmax_warp(
    const unsigned char* packed,
    const float* vector,
    unsigned long long* winners,
    const int input_size,
    const int output_size
) {
    // Both call sites launch 256 threads = 8 warps, and the grid maps 8 rows
    // per block (blockIdx.x*8 + warp), so exactly one warp handles each row.
    __shared__ unsigned long long warp_best[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    // Seed every slot (including warps whose row is out of range) before the
    // barrier so the block-level fold never reads an uninitialised winner.
    if (lane == 0) warp_best[warp] = 0ull;
    __syncthreads();
    if (row < output_size) {
        float partial = 0.0f;
        for (int input = lane; input < input_size; input += 32) {
            const int absolute = row * input_size + input;
            const int block = absolute >> 5;
            const int within = absolute & 31;
            const float scale = __half2float(
                *((const __half*)(packed + block * 34))
            );
            const signed char value = *((const signed char*)(
                packed + block * 34 + 2 + within
            ));
            partial += ((float)value * scale) * vector[input];
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (lane == 0) {
            const unsigned int bits = __float_as_uint(partial);
            const unsigned int ordered = bits ^ (
                ((int)bits < 0) ? 0xffffffffu : 0x80000000u
            );
            warp_best[warp] =
                ((unsigned long long)ordered << 32)
                | (unsigned int)(0xffffffffu - (unsigned int)row);
        }
    }
    __syncthreads();
    // Fold this block's 8 warp winners in shared memory and issue a single
    // global atomicMax into winners[0] (the value the host reads back). This
    // cuts global-atomic pressure ~8x versus one atomic per warp, so the
    // ~152k-row lm_head contends on winners[0] only ~19k times per token.
    if (threadIdx.x == 0) {
        unsigned long long best = warp_best[0];
        for (int i = 1; i < 8; ++i) best = max(best, warp_best[i]);
        atomicMax(winners, best);
    }
}

extern "C" __global__
void q8_lm_head_argmax_reduce(
    const unsigned long long* block_winners,
    unsigned int* output,
    const int block_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long best = block_winners[0];
    for (int i = 1; i < block_count; ++i)
        best = max(best, block_winners[i]);
    *output = (unsigned int)(0xffffffffu - (unsigned int)best);
}

__device__ __forceinline__ void q5k_scale_min(
    const unsigned char* scales, int index, int* scale, int* minimum
) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

__device__ __forceinline__ float q5k_value(
    const unsigned char* packed, int absolute
) {
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 176;
    const float d = __half2float(*((const __half*)(base)));
    const float dmin = __half2float(*((const __half*)(base + 2)));
    const unsigned char* scales = base + 4;
    const int group = within / 64;
    const int offset = within & 63;
    const int sub = offset / 32;
    const int qindex = group * 32 + (offset & 31);
    const unsigned char low = base[48 + qindex];
    const unsigned char high = base[16 + (offset & 31)];
    const int bit = (high >> (2 * group + sub)) & 1;
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    const int quant = ((offset < 32) ? (low & 15) : (low >> 4)) + 16 * bit;
    int scale, minimum;
    q5k_scale_min(scales, group * 2 + sub, &scale, &minimum);
    return d * (float)scale * (float)quant - dmin * (float)minimum;
}

extern "C" __global__
void q5k_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q5k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q5k_swiglu_transposed(
    const unsigned char* gate_packed,
    const unsigned char* up_packed,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q5k_value(gate_packed, absolute) * value;
        up += q5k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[row] = (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q5k_grouped_swiglu(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate_packed =
        (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up_packed =
        (const unsigned char*)up_ptrs[expert];
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q5k_value(gate_packed, absolute) * value;
        up += q5k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q5k_grouped_swiglu_rows(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const int* counts,
    const float* vectors,
    float* activated,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int route = blockIdx.y;
    const int token = route / top_k;
    const int rank = route - token * top_k;
    if (row >= output_size || token >= rows || rank >= counts[token]) return;
    const unsigned char* gate_packed =
        (const unsigned char*)gate_ptrs[route];
    const unsigned char* up_packed =
        (const unsigned char*)up_ptrs[route];
    const float* vector = vectors + token * input_size;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q5k_value(gate_packed, absolute) * value;
        up += q5k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[route * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q5k_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[expert];
            combined += weights[expert]
                * q5k_value(packed, row * input_size + input)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

extern "C" __global__
void q5k_grouped_accumulate_rows(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int* counts,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= output_size || token >= rows) return;
    const int base = token * top_k;
    const int count = counts[token];
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int rank = 0; rank < count; ++rank) {
            const int route = base + rank;
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[route];
            combined += weights[route]
                * q5k_value(packed, row * input_size + input)
                * activated[route * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * output_size + row] += partial;
}


)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Warp-per-row LM-head argmax for the weight types that lacked one. The
// dispatch used to fall through to the Q8_0 kernel for anything it did not
// recognize, which decodes the head as garbage and pins the argmax to a
// constant token.
#define COLIBRI_LM_HEAD_ARGMAX(name, decode)                                    \
extern "C" __global__                                                           \
void name(                                                                      \
    const unsigned char* packed,                                                \
    const float* vector,                                                        \
    unsigned long long* winners,                                                \
    const int input_size,                                                       \
    const int output_size                                                       \
) {                                                                             \
    __shared__ unsigned long long warp_best[8];                                 \
    const int lane = threadIdx.x & 31;                                          \
    const int warp = threadIdx.x >> 5;                                          \
    const int row = blockIdx.x * 8 + warp;                                      \
    if (lane == 0) warp_best[warp] = 0ull;                                      \
    __syncthreads();                                                            \
    if (row < output_size) {                                                    \
        float partial = 0.0f;                                                   \
        for (int input = lane; input < input_size; input += 32)                 \
            partial += decode(packed, row * input_size + input) * vector[input];\
        for (int offset = 16; offset > 0; offset >>= 1)                         \
            partial += __shfl_down_sync(0xffffffff, partial, offset);           \
        if (lane == 0) {                                                        \
            const unsigned int bits = __float_as_uint(partial);                 \
            const unsigned int ordered = bits ^ (                               \
                ((int)bits < 0) ? 0xffffffffu : 0x80000000u                     \
            );                                                                  \
            warp_best[warp] =                                                   \
                ((unsigned long long)ordered << 32)                             \
                | (unsigned int)(0xffffffffu - (unsigned int)row);              \
        }                                                                       \
    }                                                                           \
    __syncthreads();                                                            \
    if (threadIdx.x == 0) {                                                     \
        unsigned long long best = warp_best[0];                                 \
        for (int i = 1; i < 8; ++i) best = max(best, warp_best[i]);             \
        atomicMax(winners, best);                                               \
    }                                                                           \
}

__device__ __forceinline__ float f32_head_value(
    const unsigned char* packed, int absolute
) {
    return ((const float*)packed)[absolute];
}

// IQ2_XXS codebook. Each grid entry is eight bytes packed into a uint64 so a
// lookup is a single load; the sign table maps a 7-bit selector to eight
// sign bits. Both come from the llama.cpp reference tables.
__device__ __constant__ unsigned long long kIq2xxsGrid[256] = {
    578721382704613384ULL, 578721382704613419ULL, 578721382704617753ULL, 578721382704622344ULL,
    578721382704622379ULL, 578721382705727513ULL, 578721382705731848ULL, 578721382706907144ULL,
    578721382706907179ULL, 578721382706916104ULL, 578721382706916139ULL, 578721382989826073ULL,
    578721382989830408ULL, 578721382990940168ULL, 578721382990949128ULL, 578721382992119833ULL,
    578721382992124168ULL, 578721383291815944ULL, 578721383291815979ULL, 578721383291824939ULL,
    578721383294109739ULL, 578721455719057433ULL, 578721455719061768ULL, 578721455720171528ULL,
    578721455720175897ULL, 578721456004270088ULL, 578721456306264328ULL, 578721456307383048ULL,
    578721533028468744ULL, 578721533028468779ULL, 578721533030762539ULL, 578721533615671339ULL,
    578740074402285593ULL, 578740074402289928ULL, 578740074403399688ULL, 578740074404579353ULL,
    578740074404583688ULL, 578740074687498248ULL, 578740074687498283ULL, 578740074687507208ULL,
    578740074689792008ULL, 578740074989488153ULL, 578740074989492488ULL, 578740074990602248ULL,
    578740074991786248ULL, 578740147416729608ULL, 578740147416729643ULL, 578740147416738568ULL,
    578740147419023368ULL, 578740147701946667ULL, 578740147704245017ULL, 578740148003932168ULL,
    578740148005046297ULL, 578740224726149913ULL, 578740224727255048ULL, 578740225011353608ULL,
    578740225313347848ULL, 578740225315641608ULL, 578759865611585544ULL, 578759865611589913ULL,
    578759865611594504ULL, 578759865612704008ULL, 578759865613888264ULL, 578759865896798233ULL,
    578759865896802568ULL, 578759865897912328ULL, 578759865897912363ULL, 578759866198797064ULL,
    578759938626033928ULL, 578759938911242248ULL, 578760015935440939ULL, 578760015936559368ULL,
    583506457308694553ULL, 583506457308698888ULL, 583506457309808648ULL, 583506457310988313ULL,
    583506457593907208ULL, 583506457596200968ULL, 583506457895901448ULL, 583506457897011208ULL,
    583506457897015577ULL, 583506530323138568ULL, 583506530323147528ULL, 583506530325432328ULL,
    583506530609465352ULL, 583506530609474347ULL, 583506530910341128ULL, 583506607634848008ULL,
    583506607917766937ULL, 583525149006366728ULL, 583525149006375688ULL, 583525149008660488ULL,
    583525149008664857ULL, 583525149291588377ULL, 583525149593569288ULL, 583525222021933832ULL,
    583525222308317227ULL, 583525299330222088ULL, 583525299331340587ULL, 583544940215666713ULL,
    583544940215671048ULL, 583544940216780808ULL, 583544940500879368ULL, 583544940802869273ULL,
    583545013230110728ULL, 583545013230115097ULL, 583545013819607048ULL, 583545090825848857ULL,
    588573006889486344ULL, 588573006889486379ULL, 588573006889495339ULL, 588573007174703368ULL,
    588573007176992793ULL, 588573007476688904ULL, 588573007476688939ULL, 588573079906233113ULL,
    588573080189152008ULL, 588573157213341704ULL, 588573157213341739ULL, 588591698587158553ULL,
    588591698587162888ULL, 588591698588272648ULL, 588591698872371208ULL, 588591698873489707ULL,
    588591771601602568ULL, 588591771886815257ULL, 588591771889113352ULL, 588591849499330568ULL,
    588611489796467464ULL, 588611489798752264ULL, 588611490384779528ULL, 588611640405530888ULL,
    1803700481349388313ULL, 1803700481349392648ULL, 1803700481350502408ULL, 1803700481350511368ULL,
    1803700481351682073ULL, 1803700481351686408ULL, 1803700481634600968ULL, 1803700481634609928ULL,
    1803700481635719467ULL, 1803700481636894728ULL, 1803700481936590873ULL, 1803700481936595208ULL,
    1803700481937704968ULL, 1803700554363832328ULL, 1803700554366126088ULL, 1803700554651338777ULL,
    1803700554951034888ULL, 1803700554951039257ULL, 1803700631673243673ULL, 1803700631674357768ULL,
    1803700631958465288ULL, 1803700631959574827ULL, 1803700631960759048ULL, 1803719173047060488ULL,
    1803719173047069448ULL, 1803719173049354248ULL, 1803719173634263048ULL, 1803719173635386137ULL,
    1803719246062618667ULL, 1803719246063802632ULL, 1803719323370915848ULL, 1803738964256360473ULL,
    1803738964256364808ULL, 1803738964257474568ULL, 1803738964541573128ULL, 1803738964541577497ULL,
    1803739037270804488ULL, 1803739037557140232ULL, 1803739037558310937ULL, 1803739037858007083ULL,
    1803739114865432857ULL, 1803739115168532488ULL, 1808485555953469448ULL, 1808485555953478408ULL,
    1808485555954583577ULL, 1808485555954592537ULL, 1808485555955763208ULL, 1808485556540672008ULL,
    1808485556540680968ULL, 1808485628967917832ULL, 1808485629253126187ULL, 1808485629557414152ULL,
    1808485706865641497ULL, 1808504248239458312ULL, 1808504248239458347ULL, 1808504320665594667ULL,
    1808504397974997017ULL, 1808504398261328136ULL, 1808524038860441608ULL, 1808524038861555737ULL,
    1808524038861564697ULL, 1808524039147952392ULL, 1808524112160098312ULL, 1808524189184305928ULL,
    1813552105534265608ULL, 1813552105535375368ULL, 1813552105819473928ULL, 1813552105821776648ULL,
    1813552178548705288ULL, 1813552178835036441ULL, 1813552255859239688ULL, 1813552256145623048ULL,
    1813570797231933448ULL, 1813570797231937817ULL, 1813570870247491592ULL, 1813570870247491627ULL,
    1813570870833584392ULL, 1813590588726446123ULL, 3100737174032091144ULL, 3100737174032091179ULL,
    3100737174032100139ULL, 3100737174317303833ULL, 3100737174619293739ULL, 3100737247046539528ULL,
    3100737247047658248ULL, 3100737247331747848ULL, 3100737324357060633ULL, 3100755865729763353ULL,
    3100755865729767688ULL, 3100755865730877448ULL, 3100755865730881817ULL, 3100755866014976008ULL,
    3100755866017269768ULL, 3100755938744207368ULL, 3100755939029424427ULL, 3100755939332528392ULL,
    3100756016053627673ULL, 3100756016338831368ULL, 3100756016341125128ULL, 3100775656939063339ULL,
    3100775729953511688ULL, 3100775807264032793ULL, 3105522248636176648ULL, 3105522248637286408ULL,
    3105522248638470408ULL, 3105522248921384968ULL, 3105522249225668633ULL, 3105522321651734827ULL,
    3105522322237818888ULL, 3105522399245244697ULL, 3105540940333844488ULL, 3105540940336138283ULL,
    3105540940619061512ULL, 3105541013634615321ULL, 3105560732130347033ULL, 3105560804559882248ULL,
    3110588798216964139ULL, 3110588798503290888ULL, 3110588798804171033ULL, 3110588871231417113ULL,
    3110588948540819464ULL, 3110607489915759368ULL, 3110627281410263048ULL, 3110627354138384648ULL,
};
__device__ __constant__ unsigned char kIq2xxsSigns[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__device__ __forceinline__ unsigned int iq2xxs_unpack_signs(
    const unsigned char selector
) {
    // The eighth sign is the parity of the seven stored signs. Broadcasting
    // the byte lets the packed comparison intrinsics form four byte masks at
    // once, matching the representation consumed by DP4A.
    const unsigned int parity = __popc((unsigned int)selector) & 1;
    return (unsigned int)(selector ^ (parity << 7)) * 0x01010101u;
}

__device__ __forceinline__ float iq2xxs_value(
    const unsigned char* packed, int absolute
) {
    // 66 bytes per 256 values: d(2) then eight 32-element groups of two uint32.
    // The low word holds four grid indices, the high word four 7-bit sign
    // selectors plus a 4-bit scale in its top nibble.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 66;
    const float d = __half2float(*((const __half*)base));
    const int group = within / 32;
    const int rest = within & 31;
    const int quad = rest / 8;
    const int element = rest & 7;
    unsigned int low, high;
    memcpy(&low, base + 2 + group * 8, 4);
    memcpy(&high, base + 2 + group * 8 + 4, 4);
    const float scale = d * (0.5f + (float)(high >> 28)) * 0.25f;
    const unsigned char signs = kIq2xxsSigns[(high >> (7 * quad)) & 127];
    const unsigned long long pattern = kIq2xxsGrid[(low >> (8 * quad)) & 255];
    const float value = (float)((unsigned char)(pattern >> (8 * element)));
    return ((signs >> element) & 1) ? -scale * value : scale * value;
}

extern "C" __global__
void iq2xxs_matvec_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq2xxs_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// Decode IQ2_XXS against a vector quantized in independent 32-value Q8
// blocks.  IQ2's codebook magnitudes fit in signed bytes, so the 32 products
// in one weight group reduce to eight DP4A instructions instead of 32
// floating-point weight reconstructions and multiplies.
extern "C" __global__
void iq2xxs_q8_matvec_transposed_warp(
    const unsigned char* packed,
    const signed char* vector,
    const __half* vector_scales,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;

    const int blocks_per_row = input_size >> 8;
    const int groups_per_row = blocks_per_row << 3;
    const unsigned char* row_data =
        packed + (long long)row * blocks_per_row * 66;
    float partial = 0.0f;
    for (int linear_group = threadIdx.x;
         linear_group < groups_per_row;
         linear_group += blockDim.x) {
        const int block = linear_group >> 3;
        const int group = linear_group & 7;
        const unsigned char* base = row_data + block * 66;
        unsigned int low, high;
        memcpy(&low, base + 2 + group * 8, 4);
        memcpy(&high, base + 2 + group * 8 + 4, 4);

        int dot = 0;
        #pragma unroll
        for (int quad = 0; quad < 4; ++quad) {
            const unsigned int signs =
                iq2xxs_unpack_signs((high >> (7 * quad)) & 127);
            const unsigned long long pattern =
                kIq2xxsGrid[(low >> (8 * quad)) & 255];
            const int element = quad * 8;
            const int masks_first = __vcmpne4(
                signs & 0x08040201u, 0);
            const int masks_second = __vcmpne4(
                signs & 0x80402010u, 0);
            const int weights_first = __vsub4(
                (int)(unsigned int)pattern ^ masks_first, masks_first);
            const int weights_second = __vsub4(
                (int)(unsigned int)(pattern >> 32) ^ masks_second,
                masks_second);
            const signed char* activations =
                vector + linear_group * 32 + element;
            dot = __dp4a(
                weights_first, *((const int*)activations), dot);
            dot = __dp4a(
                weights_second, *((const int*)(activations + 4)), dot);
        }
        const int scale = (int)(high >> 27) | 1;
        partial += (float)(dot * scale) * 0.125f
            * __half2float(*((const __half*)base))
            * __half2float(vector_scales[linear_group]);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    __shared__ float warp_sums[4];
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        partial = lane < 4 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        if (lane == 0) output[row] = partial;
    }
}

__device__ __constant__ unsigned int kIq3xxsGrid[256] = {
    67372036u, 67372052u, 67372068u, 67374092u, 67374108u, 67374142u, 67376132u, 67376148u,
    67378188u, 67380244u, 67386908u, 67386924u, 67896332u, 67896348u, 67898372u, 67898388u,
    67900428u, 67900460u, 67902468u, 67902484u, 67904524u, 67906596u, 67911172u, 68420612u,
    68420628u, 68420644u, 68422668u, 68424708u, 68424724u, 68426764u, 68426780u, 68426814u,
    68430860u, 68430910u, 68435500u, 68944908u, 68944958u, 68946948u, 68946964u, 68949036u,
    68959748u, 69471260u, 69475390u, 69477412u, 69479486u, 69484060u, 69484076u, 69993484u,
    69993534u, 69999636u, 70003732u, 70523948u, 70530084u, 71175172u, 71175204u, 71175220u,
    71181340u, 71185420u, 201589772u, 201589788u, 201591812u, 201591828u, 201593868u, 201593884u,
    201595908u, 201595924u, 201595940u, 201598014u, 201600004u, 202114052u, 202114068u, 202116108u,
    202118148u, 202118164u, 202638348u, 202638364u, 202640388u, 202640404u, 202642444u, 202644484u,
    202653204u, 203162628u, 203162644u, 203166724u, 203168780u, 203170868u, 203174964u, 203686924u,
    203686956u, 203697156u, 204215300u, 204215332u, 204219444u, 204226060u, 204735532u, 205394964u,
    205399044u, 335807492u, 335807508u, 335809548u, 335809564u, 335811588u, 335811604u, 335811636u,
    335813644u, 335815700u, 336331788u, 336331804u, 336331820u, 336333828u, 336333844u, 336335884u,
    336337924u, 336344092u, 336344126u, 336346628u, 336856068u, 336856084u, 336858124u, 336858174u,
    336860164u, 336860180u, 336862270u, 336864260u, 336866348u, 337380364u, 337382404u, 337382436u,
    337395204u, 337395236u, 337910828u, 337914908u, 338428956u, 338433086u, 338437132u, 338443812u,
    339608588u, 339608604u, 339610676u, 339616812u, 470025228u, 470027268u, 470027284u, 470029324u,
    470029340u, 470035460u, 470037548u, 470040084u, 470549508u, 470549524u, 470553604u, 470555660u,
    470557732u, 470557748u, 471073804u, 471073820u, 471075844u, 471077932u, 471084052u, 471088660u,
    471600140u, 471604252u, 472128516u, 472130622u, 472137236u, 472646660u, 472646708u, 472650772u,
    472656940u, 473173028u, 473177140u, 473183260u, 473832476u, 473838596u, 604242980u, 604245054u,
    604249132u, 604249150u, 604253212u, 604253246u, 604782116u, 605295620u, 605297726u, 605299716u,
    605303812u, 605303860u, 605815870u, 605824044u, 606340132u, 606350348u, 606352420u, 606868524u,
    606872604u, 606879236u, 608044076u, 608046084u, 608046100u, 608050180u, 738462740u, 738468876u,
    738475524u, 738984964u, 738985012u, 738989108u, 738995244u, 739511332u, 739515412u, 739524116u,
    740033556u, 740043804u, 740559876u, 740561948u, 740561982u, 740572692u, 741082132u, 741088268u,
    741616644u, 742265892u, 742269972u, 872682532u, 872686628u, 872686644u, 872690724u, 873206796u,
    873214988u, 873729086u, 873739300u, 874257412u, 874257460u, 874783780u, 875299884u, 875310100u,
    875830300u, 876479516u, 876483596u, 1040450588u, 1040450604u, 1040450622u, 1040452612u, 1040456724u,
    1040460820u, 1040978996u, 1040983044u, 1041501204u, 1041507372u, 1041509396u, 1042023428u, 1042025516u,
    1042029596u, 1042035716u, 1042551820u, 1042555916u, 1043072004u, 1043072020u, 1043076132u, 1043602436u,
};

__device__ __forceinline__ float iq3xxs_value(
    const unsigned char* packed, int absolute
) {
    // 98 bytes per 256 values: d(2) qs[64] scales[32]. Scale and signs are laid
    // out as in IQ2_XXS, but grid entries are four values wide so each 8-output
    // quad reads two consecutive qs bytes.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 98;
    const float d = __half2float(*((const __half*)base));
    const int group = within / 32;
    const int rest = within & 31;
    const int quad = rest / 8;
    const int element = rest & 7;
    unsigned int aux;
    memcpy(&aux, base + 2 + 64 + group * 4, 4);
    const float scale = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const unsigned char signs = kIq2xxsSigns[(aux >> (7 * quad)) & 127];
    const unsigned int pattern =
        kIq3xxsGrid[base[2 + group * 8 + quad * 2 + (element >> 2)]];
    const float value = (float)((pattern >> (8 * (element & 3))) & 0xffu);
    return ((signs >> element) & 1) ? -scale * value : scale * value;
}

extern "C" __global__
void iq3xxs_matvec_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq3xxs_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(__device__ __constant__ unsigned long long kIq2sGrid[1024] = {
    578721382704613384ULL, 578721382704613419ULL, 578721382704617753ULL, 578721382704622344ULL,
    578721382704622379ULL, 578721382705727513ULL, 578721382705731848ULL, 578721382705731883ULL,
    578721382705736473ULL, 578721382706907144ULL, 578721382706907179ULL, 578721382706911513ULL,
    578721382706916104ULL, 578721382989826073ULL, 578721382989830408ULL, 578721382989830443ULL,
    578721382989835033ULL, 578721382990940168ULL, 578721382990940203ULL, 578721382990944537ULL,
    578721382990949128ULL, 578721382992119833ULL, 578721382992124168ULL, 578721382992124203ULL,
    578721382992128793ULL, 578721383291815944ULL, 578721383291815979ULL, 578721383291820313ULL,
    578721383291824904ULL, 578721383292930073ULL, 578721383292934408ULL, 578721383294109704ULL,
    578721383294114073ULL, 578721383294118699ULL, 578721455719057433ULL, 578721455719061768ULL,
    578721455719061803ULL, 578721455719066393ULL, 578721455720171528ULL, 578721455720171563ULL,
    578721455720175897ULL, 578721455720180488ULL, 578721455721351193ULL, 578721455721355528ULL,
    578721456004270088ULL, 578721456004270123ULL, 578721456004274457ULL, 578721456004279048ULL,
    578721456005384217ULL, 578721456005388552ULL, 578721456005388587ULL, 578721456005393177ULL,
    578721456006563848ULL, 578721456006568217ULL, 578721456006572808ULL, 578721456306259993ULL,
    578721456306264328ULL, 578721456307374088ULL, 578721456307374123ULL, 578721456307378457ULL,
    578721456308553753ULL, 578721456308558088ULL, 578721533028468744ULL, 578721533028468779ULL,
    578721533028473113ULL, 578721533028477704ULL, 578721533029582873ULL, 578721533029587208ULL,
    578721533030762504ULL, 578721533030771499ULL, 578721533313681433ULL, 578721533313685768ULL,
    578721533313685803ULL, 578721533313690393ULL, 578721533314795528ULL, 578721533314799897ULL,
    578721533615671304ULL, 578721533615675673ULL, 578721533615680299ULL, 578721533616789768ULL,
    578721533617965099ULL, 578740074402285593ULL, 578740074402289928ULL, 578740074402289963ULL,
    578740074402294553ULL, 578740074403399688ULL, 578740074403399723ULL, 578740074403404057ULL,
    578740074403408648ULL, 578740074404579353ULL, 578740074404583688ULL, 578740074404583723ULL,
    578740074404588313ULL, 578740074687498248ULL, 578740074687498283ULL, 578740074687502617ULL,
    578740074687507208ULL, 578740074687507243ULL, 578740074688612377ULL, 578740074688616712ULL,
    578740074688616747ULL, 578740074688621337ULL, 578740074689792008ULL, 578740074689792043ULL,
    578740074689796377ULL, 578740074989488153ULL, 578740074989492488ULL, 578740074989492523ULL,
    578740074989497113ULL, 578740074990602248ULL, 578740074990606617ULL, 578740074990611208ULL,
    578740074991781913ULL, 578740074991786248ULL, 578740147416729608ULL, 578740147416729643ULL,
    578740147416733977ULL, 578740147416738568ULL, 578740147416738603ULL, 578740147417843737ULL,
    578740147417848072ULL, 578740147417848107ULL, 578740147417852697ULL, 578740147419023368ULL,
    578740147419027737ULL, 578740147419032328ULL, 578740147701942297ULL, 578740147701946632ULL,
    578740147701946667ULL, 578740147701951257ULL, 578740147703056392ULL, 578740147703056427ULL,
    578740147703060761ULL, 578740147703065352ULL, 578740147704236057ULL, 578740147704240392ULL,
    578740148003932168ULL, 578740148003932203ULL, 578740148003936537ULL, 578740148003941128ULL,
    578740148005046297ULL, 578740148005050632ULL, 578740148006225928ULL, 578740224726140953ULL,
    578740224726145288ULL, 578740224726145323ULL, 578740224726149913ULL, 578740224727255048ULL,
    578740224727259417ULL, 578740225011353608ULL, 578740225011357977ULL, 578740225011362568ULL,
    578740225012467737ULL, 578740225012472072ULL, 578740225013647368ULL, 578740225313343513ULL,
    578740225313347848ULL, 578740225314457608ULL, 578759865611585544ULL, 578759865611585579ULL,
    578759865611589913ULL, 578759865611594504ULL, 578759865612699673ULL, 578759865612704008ULL,
    578759865612704043ULL, 578759865612708633ULL, 578759865613879304ULL, 578759865613883673ULL,
    578759865613888299ULL, 578759865896798233ULL, 578759865896802568ULL, 578759865896802603ULL,
    578759865896807193ULL, 578759865897912328ULL, 578759865897912363ULL, 578759865897916697ULL,
    578759865897921288ULL, 578759865899091993ULL, 578759865899096328ULL, 578759866198788104ULL,
    578759866198792473ULL, 578759866199906568ULL, 578759866201090859ULL, 578759938626029593ULL,
    578759938626033928ULL, 578759938627143688ULL, 578759938627143723ULL, 578759938627148057ULL,
    578759938627152648ULL, 578759938628323353ULL, 578759938911242248ULL, 578759938911246617ULL,
    578759938911251208ULL, 578759938912356377ULL, 578759938912360712ULL, 578759938913536008ULL,
    578759939213232153ULL, 578759939214346248ULL, 578760015935440904ULL, 578760015936555033ULL,
    578760015936559368ULL, 578760015937734699ULL, 578760015937743624ULL, 578760015937743659ULL,
    578760016221767688ULL, 578760016523766553ULL, 583506457308694553ULL, 583506457308698888ULL,
    583506457308698923ULL, 583506457308703513ULL, 583506457309808648ULL, 583506457309808683ULL,
    583506457309813017ULL, 583506457309817608ULL, 583506457310988313ULL, 583506457310992648ULL,
    583506457310992683ULL, 583506457593907208ULL, 583506457593907243ULL, 583506457593911577ULL,
    583506457593916168ULL, 583506457595021337ULL, 583506457595025672ULL, 583506457595025707ULL,
    583506457595030297ULL, 583506457596200968ULL, 583506457596201003ULL, 583506457596205337ULL,
    583506457596209928ULL, 583506457895897113ULL, 583506457895901448ULL, 583506457895901483ULL,
    583506457897011208ULL, 583506457897015577ULL, 583506457897020168ULL, 583506457898190873ULL,
    583506457898195208ULL, 583506530323138568ULL, 583506530323138603ULL, 583506530323142937ULL,
    583506530323147528ULL, 583506530323147563ULL, 583506530324252697ULL, 583506530324257032ULL,
    583506530324257067ULL, 583506530324261657ULL, 583506530325432328ULL, 583506530325432363ULL,
    583506530325436697ULL, 583506530325441288ULL, 583506530608351257ULL, 583506530608355592ULL,
    583506530608355627ULL, 583506530608360217ULL, 583506530609465352ULL, 583506530609465387ULL,
    583506530609469721ULL, 583506530609474312ULL, 583506530610645017ULL, 583506530610649352ULL,
    583506530910341128ULL, 583506530910341163ULL, 583506530910345497ULL, 583506530910350088ULL,
    583506530911455257ULL, 583506530911459592ULL, 583506607632549913ULL, 583506607632554248ULL,
    583506607632558873ULL, 583506607633664008ULL, 583506607633668377ULL, 583506607634843673ULL,
    583506607634848008ULL, 583506607917762568ULL, 583506607917766937ULL, 583506607918876697ULL,
    583506607918881032ULL, 583506608219752473ULL, 583506608219756808ULL, 583506608220866568ULL,
    583525149006366728ULL, 583525149006366763ULL, 583525149006371097ULL, 583525149006375688ULL,
    583525149007480857ULL, 583525149007485192ULL, 583525149007485227ULL, 583525149007489817ULL,
    583525149008660488ULL, 583525149008664857ULL, 583525149008669448ULL, 583525149291579417ULL,
    583525149291583752ULL, 583525149291583787ULL, 583525149291588377ULL, 583525149292693512ULL,
    583525149292693547ULL, 583525149292697881ULL, 583525149292702472ULL, 583525149293873177ULL,
    583525149293877512ULL, 583525149593569288ULL, 583525149593569323ULL, 583525149593573657ULL,
    583525149593578248ULL, 583525149594683417ULL, 583525149594687752ULL, 583525149595863048ULL,
    583525222020810777ULL, 583525222020815112ULL, 583525222020815147ULL, 583525222020819737ULL,
    583525222021924872ULL, 583525222021924907ULL, 583525222021929241ULL, 583525222021933832ULL,
    583525222023104537ULL, 583525222023108872ULL, 583525222306023432ULL, 583525222306023467ULL,
    583525222306027801ULL, 583525222306032392ULL, 583525222307137561ULL, 583525222307141896ULL,
    583525222308317192ULL, 583525222608013337ULL, 583525222608017672ULL, 583525222609127432ULL,
    583525299330222088ULL, 583525299330226457ULL, 583525299330231048ULL, 583525299331336217ULL,
    583525299331340552ULL, 583525299332515848ULL, 583525299615434777ULL, 583525299615439112ULL,
    583525299616548872ULL, 583525299917424648ULL, 583525299919727403ULL, 583544940215666713ULL,
    583544940215671048ULL, 583544940215671083ULL, 583544940215675673ULL, 583544940216780808ULL,
    583544940216785177ULL, 583544940216789768ULL, 583544940217960473ULL, 583544940500879368ULL,
    583544940500879403ULL, 583544940500883737ULL, 583544940500888328ULL, 583544940501993497ULL,
    583544940501997832ULL, 583544940503173128ULL, 583544940802869273ULL, 583544940802873608ULL,
    583545013230110728ULL, 583545013230110763ULL, 583545013230115097ULL, 583545013230119688ULL,
    583545013231224857ULL, 583545013231229192ULL, 583545013232404488ULL, 583545013515323417ULL,
    583545013515327752ULL, 583545013516437512ULL, 583545013517626137ULL, 583545013819607083ULL,
    583545090539526408ULL, 583545090540636168ULL, 583545090824734728ULL, 583545090825853227ULL,
    588573006889486344ULL, 588573006889486379ULL, 588573006889490713ULL, 588573006889495304ULL,
    588573006890600473ULL, 588573006890604808ULL, 588573006890604843ULL, 588573006890609433ULL,
    588573006891780104ULL, 588573006891784473ULL, 588573006891789099ULL, 588573007174699033ULL,
    588573007174703368ULL, 588573007175813128ULL, 588573007175813163ULL, 588573007175817497ULL,
    588573007176997128ULL, 588573007476688904ULL, 588573007476697899ULL, 588573007477807368ULL,
    588573007478991659ULL, 588573079903930393ULL, 588573079903934728ULL, 588573079905044488ULL,
    588573079905044523ULL, 588573079905048857ULL, 588573079906224153ULL, 588573080189143048ULL,
    588573080189143083ULL, 588573080189147417ULL, 588573080190257177ULL, 588573080190261512ULL,
    588573080191436808ULL, 588573080491132953ULL, 588573080491137288ULL, 588573080492247048ULL,
    588573157213341704ULL, 588573157213350699ULL, 588573157215635499ULL, 588573157215644424ULL,
    588573157215644459ULL, 588573157498558728ULL, 588573157499668488ULL, 588573157800553224ULL,
    588573157800553259ULL, 588573157802846984ULL, 588591698587158553ULL, 588591698587162888ULL,
    588591698587162923ULL, 588591698587167513ULL, 588591698588272648ULL, 588591698588277017ULL,
    588591698588281608ULL, 588591698589452313ULL, 588591698589456648ULL, 588591698872371208ULL,
    588591698872371243ULL, 588591698872375577ULL, 588591698872380168ULL, 588591698873485337ULL,
    588591698873489672ULL, 588591698874664968ULL, 588591699174361113ULL, 588591699174365448ULL,
    588591699175475208ULL, 588591771601602568ULL, 588591771601606937ULL, 588591771601611528ULL,
    588591771602716697ULL, 588591771602721032ULL, 588591771603896328ULL, 588591771886815257ULL,
    588591771886819592ULL, 588591771887929352ULL, 588591771889113387ULL, 588591772188805128ULL,
    588591848911013913ULL, 588591848911018248ULL, 588591848912128008ULL, 588591849196226568ULL,
    588591849197349657ULL, 588611489796458504ULL, 588611489796462873ULL, 588611489797572633ULL,
    588611489797576968ULL, 588611490081671193ULL, 588611490081675528ULL, 588611490082785288ULL,
    588611490383670059ULL, 588611490385963819ULL, 588611562810902553ULL, 588611562810906888ULL,
    588611562812016648ULL, 588611563399223577ULL, 588611640120322859ULL, 588611640122607659ULL,
    588611640407824648ULL, 588611640707525384ULL, 588611640707525419ULL, 1803700481349388313ULL,
    1803700481349392648ULL, 1803700481349392683ULL, 1803700481349397273ULL, 1803700481350502408ULL,
    1803700481350502443ULL, 1803700481350506777ULL, 1803700481350511368ULL, 1803700481350511403ULL,
    1803700481351682073ULL, 1803700481351686408ULL, 1803700481351686443ULL, 1803700481634600968ULL,
    1803700481634601003ULL, 1803700481634605337ULL, 1803700481634609928ULL, 1803700481634609963ULL,
    1803700481635715097ULL, 1803700481635719432ULL, 1803700481635719467ULL, 1803700481635724057ULL,
    1803700481636894728ULL, 1803700481636894763ULL, 1803700481636899097ULL, 1803700481936590873ULL,
    1803700481936595208ULL, 1803700481937704968ULL, 1803700481937709337ULL, 1803700481937713928ULL,
    1803700481938884633ULL, 1803700481938888968ULL, 1803700554363832328ULL, 1803700554363832363ULL,
    1803700554363836697ULL, 1803700554363841288ULL, 1803700554364946457ULL, 1803700554364950792ULL,
    1803700554364950827ULL, 1803700554364955417ULL, 1803700554366126088ULL, 1803700554366126123ULL,
    1803700554366130457ULL, 1803700554649045017ULL, 1803700554649049352ULL, 1803700554649049387ULL,
    1803700554649053977ULL, 1803700554650159112ULL, 1803700554650159147ULL, 1803700554650163481ULL,
    1803700554650168072ULL, 1803700554651338777ULL, 1803700554651343112ULL, 1803700554951034888ULL,
    1803700554951034923ULL, 1803700554951039257ULL, 1803700554951043848ULL, 1803700554952149017ULL,
    1803700554952153352ULL, 1803700554953328648ULL, 1803700631673243673ULL, 1803700631673248008ULL,
    1803700631674357768ULL, 1803700631674357803ULL, 1803700631674362137ULL, 1803700631674366728ULL,
    1803700631675541768ULL, 1803700631958456328ULL, 1803700631958460697ULL, 1803700631958465288ULL,
    1803700631959570457ULL, 1803700631959574792ULL, 1803700631960750088ULL, 1803700632260446233ULL,
    1803700632260450568ULL, 1803719173047060488ULL, 1803719173047060523ULL, 1803719173047064857ULL,
    1803719173047069448ULL, 1803719173047069483ULL, 1803719173048174617ULL, 1803719173048178952ULL,
    1803719173048178987ULL, 1803719173048183577ULL, 1803719173049354248ULL, 1803719173049354283ULL,
    1803719173049358617ULL, 1803719173049363208ULL, 1803719173332273177ULL, 1803719173332277512ULL,
    1803719173332277547ULL, 1803719173332282137ULL, 1803719173333387272ULL, 1803719173333387307ULL,
    1803719173333391641ULL, 1803719173333396232ULL, 1803719173334566937ULL, 1803719173334571272ULL,
    1803719173634263048ULL, 1803719173634263083ULL, 1803719173634267417ULL, 1803719173634272008ULL,
    1803719173635377177ULL, 1803719173635381512ULL, 1803719173636556808ULL, 1803719246061504537ULL,
    1803719246061508872ULL, 1803719246061508907ULL, 1803719246061513497ULL, 1803719246062618632ULL,
    1803719246062618667ULL, 1803719246062623001ULL, 1803719246062627592ULL, 1803719246063798297ULL,
    1803719246063802632ULL, 1803719246346717192ULL, 1803719246346717227ULL, 1803719246346721561ULL,
    1803719246346726152ULL, 1803719246347831321ULL, 1803719246347835656ULL, 1803719246349010952ULL,
    1803719246349019947ULL, 1803719246648707097ULL, 1803719246648711432ULL, 1803719246649821192ULL,
    1803719323370915848ULL, 1803719323370915883ULL, 1803719323370920217ULL, 1803719323370924808ULL,
    1803719323372029977ULL, 1803719323372034312ULL, 1803719323373209608ULL, 1803719323656128537ULL,
    1803719323656132872ULL, 1803719323657242632ULL, 1803719323958118408ULL, 1803719323960416537ULL,
    1803738964256360473ULL, 1803738964256364808ULL, 1803738964256369433ULL, 1803738964257474568ULL,
    1803738964257474603ULL, 1803738964257478937ULL, 1803738964257483528ULL, 1803738964258654233ULL,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    1803738964258658568ULL, 1803738964541573128ULL, 1803738964541573163ULL, 1803738964541577497ULL,
    1803738964541582088ULL, 1803738964542687257ULL, 1803738964542691592ULL, 1803738964543866888ULL,
    1803738964843567368ULL, 1803738964844677128ULL, 1803739037270804488ULL, 1803739037270804523ULL,
    1803739037270808857ULL, 1803739037270813448ULL, 1803739037271918617ULL, 1803739037271922952ULL,
    1803739037273098248ULL, 1803739037556017177ULL, 1803739037556021512ULL, 1803739037557131272ULL,
    1803739037858007048ULL, 1803739037859125547ULL, 1803739114580215833ULL, 1803739114580220168ULL,
    1803739114581329928ULL, 1803739114865428488ULL, 1808485555953469448ULL, 1808485555953469483ULL,
    1808485555953473817ULL, 1808485555953478408ULL, 1808485555954583577ULL, 1808485555954587912ULL,
    1808485555954587947ULL, 1808485555954592537ULL, 1808485555955763208ULL, 1808485555955763243ULL,
    1808485555955767577ULL, 1808485555955772168ULL, 1808485556238682137ULL, 1808485556238686472ULL,
    1808485556238686507ULL, 1808485556238691097ULL, 1808485556239796232ULL, 1808485556239796267ULL,
    1808485556239800601ULL, 1808485556239805192ULL, 1808485556240975897ULL, 1808485556240980232ULL,
    1808485556540672008ULL, 1808485556540672043ULL, 1808485556540676377ULL, 1808485556540680968ULL,
    1808485556541786137ULL, 1808485556541790472ULL, 1808485628967913497ULL, 1808485628967917832ULL,
    1808485628967917867ULL, 1808485628967922457ULL, 1808485628969027592ULL, 1808485628969027627ULL,
    1808485628969031961ULL, 1808485628969036552ULL, 1808485628970207257ULL, 1808485628970211592ULL,
    1808485629253126152ULL, 1808485629253126187ULL, 1808485629253130521ULL, 1808485629253135112ULL,
    1808485629254240281ULL, 1808485629254244616ULL, 1808485629255419912ULL, 1808485629555116057ULL,
    1808485629555120392ULL, 1808485629556230152ULL, 1808485706277324808ULL, 1808485706277329177ULL,
    1808485706277333768ULL, 1808485706278438937ULL, 1808485706278443272ULL, 1808485706279618568ULL,
    1808485706562537497ULL, 1808485706562541832ULL, 1808485706563651592ULL, 1808485706564840217ULL,
    1808485706864527368ULL, 1808504247651141657ULL, 1808504247651145992ULL, 1808504247651146027ULL,
    1808504247651150617ULL, 1808504247652255752ULL, 1808504247652255787ULL, 1808504247652260121ULL,
    1808504247652264712ULL, 1808504247653435417ULL, 1808504247653439752ULL, 1808504247936354312ULL,
    1808504247936354347ULL, 1808504247936358681ULL, 1808504247936363272ULL, 1808504247937468441ULL,
    1808504247937472776ULL, 1808504247938648072ULL, 1808504248238344217ULL, 1808504248238348552ULL,
    1808504248239458312ULL, 1808504320665585672ULL, 1808504320665585707ULL, 1808504320665590041ULL,
    1808504320665594632ULL, 1808504320666699801ULL, 1808504320666704136ULL, 1808504320667879432ULL,
    1808504320950798361ULL, 1808504320950802696ULL, 1808504320951912456ULL, 1808504321252788232ULL,
    1808504397974997017ULL, 1808504397975001352ULL, 1808504397976111112ULL, 1808504397977295147ULL,
    1808504398260209672ULL, 1808524038860441608ULL, 1808524038860441643ULL, 1808524038860445977ULL,
    1808524038860450568ULL, 1808524038861555737ULL, 1808524038861560072ULL, 1808524038862735368ULL,
    1808524039145654297ULL, 1808524039145658632ULL, 1808524039146768392ULL, 1808524039146777387ULL,
    1808524039447644168ULL, 1808524111874885657ULL, 1808524111874889992ULL, 1808524111875999752ULL,
    1808524112160098312ULL, 1808524189184296968ULL, 1808524189185420057ULL, 1808524189771503897ULL,
    1808524189773802248ULL, 1813552105534261273ULL, 1813552105534265608ULL, 1813552105534265643ULL,
    1813552105535375368ULL, 1813552105535375403ULL, 1813552105535379737ULL, 1813552105535384328ULL,
    1813552105536555033ULL, 1813552105536559368ULL, 1813552105819473928ULL, 1813552105819478297ULL,
    1813552105819482888ULL, 1813552105820588057ULL, 1813552105820592392ULL, 1813552105821767688ULL,
    1813552106121468168ULL, 1813552106122577928ULL, 1813552178548705288ULL, 1813552178548705323ULL,
    1813552178548709657ULL, 1813552178548714248ULL, 1813552178549819417ULL, 1813552178549823752ULL,
    1813552178550999048ULL, 1813552178833917977ULL, 1813552178833922312ULL, 1813552178835032072ULL,
    1813552179135907848ULL, 1813552179137030937ULL, 1813552255858120968ULL, 1813552255859230728ULL,
    1813552256143329288ULL, 1813552256144447787ULL, 1813552256447612953ULL, 1813570797231933448ULL,
    1813570797231937817ULL, 1813570797231942408ULL, 1813570797233047577ULL, 1813570797233051912ULL,
    1813570797234227208ULL, 1813570797517146137ULL, 1813570797517150472ULL, 1813570797518260232ULL,
    1813570797819136008ULL, 1813570870246377497ULL, 1813570870246381832ULL, 1813570870247491592ULL,
    1813570870531590152ULL, 1813570870531599147ULL, 1813570870533892872ULL, 1813570870834694187ULL,
    1813570947555788808ULL, 1813570948144109832ULL, 1813590588441233433ULL, 1813590588441237768ULL,
    1813590588442347528ULL, 1813590588728744217ULL, 1813590589029559048ULL, 1813590661455677448ULL,
    1813590661457980203ULL, 1813590739050301483ULL, 1813590739354585113ULL, 3100737174032091144ULL,
    3100737174032091179ULL, 3100737174032095513ULL, 3100737174032100104ULL, 3100737174033205273ULL,
    3100737174033209608ULL, 3100737174033214233ULL, 3100737174034384904ULL, 3100737174034389273ULL,
    3100737174317303833ULL, 3100737174317308168ULL, 3100737174318417928ULL, 3100737174318417963ULL,
    3100737174318422297ULL, 3100737174318426888ULL, 3100737174319597593ULL, 3100737174619293704ULL,
    3100737174619298073ULL, 3100737174620407833ULL, 3100737174620412168ULL, 3100737247046535193ULL,
    3100737247046539528ULL, 3100737247046544153ULL, 3100737247047649288ULL, 3100737247047649323ULL,
    3100737247047653657ULL, 3100737247047658248ULL, 3100737247048828953ULL, 3100737247048833288ULL,
    3100737247331747848ULL, 3100737247331747883ULL, 3100737247331752217ULL, 3100737247331756808ULL,
    3100737247332861977ULL, 3100737247332866312ULL, 3100737247633737753ULL, 3100737247633742088ULL,
    3100737247634851848ULL, 3100737247636040473ULL, 3100737324355946504ULL, 3100737324355950873ULL,
    3100737324355955499ULL, 3100737324357060633ULL, 3100737324357064968ULL, 3100737324641159193ULL,
    3100737324641163528ULL, 3100737324642273288ULL, 3100755865729763353ULL, 3100755865729767688ULL,
    3100755865729767723ULL, 3100755865729772313ULL, 3100755865730877448ULL, 3100755865730877483ULL,
    3100755865730881817ULL, 3100755865730886408ULL, 3100755865732057113ULL, 3100755866014976008ULL,
    3100755866014976043ULL, 3100755866014980377ULL, 3100755866014984968ULL, 3100755866016090137ULL,
    3100755866016094472ULL, 3100755866017269768ULL, 3100755866316965913ULL, 3100755866316970248ULL,
    3100755866318080008ULL, 3100755938744207368ULL, 3100755938744207403ULL, 3100755938744211737ULL,
    3100755938744216328ULL, 3100755938745321497ULL, 3100755938745325832ULL, 3100755938746501128ULL,
    3100755939029420057ULL, 3100755939029424392ULL, 3100755939030534152ULL, 3100755939331409928ULL,
    3100755939331418923ULL, 3100756016053618713ULL, 3100756016053623048ULL, 3100756016054732808ULL,
    3100756016055921433ULL, 3100756016338831368ULL, 3100775656939063304ULL, 3100775656939067673ULL,
    3100775656940177433ULL, 3100775656940181768ULL, 3100775657224275993ULL, 3100775657224280328ULL,
    3100775657225390088ULL, 3100775657528559659ULL, 3100775729953507353ULL, 3100775729953511688ULL,
    3100775730238720008ULL, 3100775730241018137ULL, 3100775807265212459ULL, 3100775807549254408ULL,
    3100775807549254443ULL, 3100775807850121259ULL, 3100775807852415019ULL, 3105522248636172313ULL,
    3105522248636176648ULL, 3105522248636181273ULL, 3105522248637286408ULL, 3105522248637286443ULL,
    3105522248637290777ULL, 3105522248637295368ULL, 3105522248638470408ULL, 3105522248921384968ULL,
    3105522248921385003ULL, 3105522248921389337ULL, 3105522248921393928ULL, 3105522248922499097ULL,
    3105522248922503432ULL, 3105522248923678728ULL, 3105522249223374873ULL, 3105522249223379208ULL,
    3105522249224488968ULL, 3105522321650616328ULL, 3105522321650620697ULL, 3105522321651730457ULL,
    3105522321651734792ULL, 3105522321935829017ULL, 3105522321935833352ULL, 3105522321936943112ULL,
    3105522321936952107ULL, 3105522398960027673ULL, 3105522398960032008ULL, 3105522398961141768ULL,
    3105522399245240328ULL, 3105522399549528363ULL, 3105540940333844488ULL, 3105540940333844523ULL,
    3105540940333848857ULL, 3105540940333853448ULL, 3105540940334958617ULL, 3105540940334962952ULL,
    3105540940336138248ULL, 3105540940619057177ULL, 3105540940619061512ULL, 3105540940620171272ULL,
    3105540940921047048ULL, 3105540940922165547ULL, 3105541013348288537ULL, 3105541013348292872ULL,
    3105541013349402632ULL, 3105541013633501192ULL, 3105541013936614152ULL, 3105541013937784857ULL,
    3105541090657699848ULL, 3105541090942916907ULL, 3105541090945210632ULL, 3105560731543144473ULL,
    3105560731543148808ULL, 3105560731544258568ULL, 3105560731545442603ULL, 3105560731828357128ULL,
    3105560732132649753ULL, 3105560804557588488ULL, 3105560804842810137ULL, 3105560804843915307ULL,
    3105560882455316488ULL, 3110588798216964104ULL, 3110588798216968473ULL, 3110588798216973099ULL,
    3110588798218082568ULL, 3110588798219257899ULL, 3110588798219266859ULL, 3110588798502176793ULL,
    3110588798502181128ULL, 3110588798503290888ULL, 3110588798806460459ULL, 3110588798806469419ULL,
    3110588871516620808ULL, 3110588871518918937ULL, 3110588948540819499ULL, 3110588948540828459ULL,
    3110588948543113259ULL, 3110588948543122184ULL, 3110588948543122219ULL, 3110588949128022059ULL,
    3110588949128030984ULL, 3110588949128031019ULL, 3110588949130324744ULL, 3110607489914636313ULL,
    3110607489914640648ULL, 3110607489915750408ULL, 3110607490199848968ULL, 3110607490501847833ULL,
    3110607490504136968ULL, 3110607562929080328ULL, 3110607562930203417ULL, 3110607640524818457ULL,
    3110627281123945259ULL, 3110627281126238984ULL, 3110627281713432619ULL, 3110627354424711432ULL,
    3110627354725587243ULL, 3110627431447800584ULL, 3110627431447800619ULL, 3110627431450085384ULL,
    3110627431450085419ULL, 3110627431450094344ULL, 3110627432035003144ULL, 3110627432037296939ULL,
};
__device__ __constant__ unsigned int kIq3sGrid[512] = {
    16843009u, 16843011u, 16843013u, 16843019u, 16843023u, 16843521u, 16843523u, 16843525u,
    16843529u, 16843533u, 16844033u, 16844035u, 16844043u, 16844551u, 16845057u, 16845061u,
    16845067u, 16845071u, 16845571u, 16845575u, 16846081u, 16846085u, 16846595u, 16846601u,
    16846607u, 16974081u, 16974083u, 16974085u, 16974089u, 16974593u, 16974595u, 16974603u,
    16975105u, 16975111u, 16975119u, 16975619u, 16975627u, 16976137u, 16977155u, 16977163u,
    16977669u, 17105153u, 17105155u, 17105163u, 17105167u, 17105665u, 17105671u, 17105677u,
    17106179u, 17106187u, 17106689u, 17106697u, 17107205u, 17107211u, 17107215u, 17107715u,
    17107719u, 17108737u, 17108743u, 17236231u, 17236739u, 17236747u, 17237249u, 17237253u,
    17237763u, 17237767u, 17237773u, 17238281u, 17238785u, 17238789u, 17239311u, 17239811u,
    17239819u, 17367297u, 17367815u, 17367823u, 17368323u, 17368329u, 17368837u, 17369345u,
    17369351u, 17369859u, 17370881u, 17498373u, 17498377u, 17499393u, 17499397u, 17499405u,
    17499911u, 17500419u, 17500427u, 17500431u, 17501453u, 17501959u, 17629453u, 17629955u,
    17629959u, 17630979u, 17632005u, 17633027u, 17760513u, 17760517u, 17760521u, 17761537u,
    17761541u, 17761549u, 17762055u, 17763073u, 17763081u, 50397441u, 50397443u, 50397445u,
    50397449u, 50397953u, 50397955u, 50397959u, 50397963u, 50397967u, 50398465u, 50398469u,
    50398979u, 50398985u, 50398989u, 50400009u, 50400013u, 50400515u, 50401029u, 50528513u,
    50528515u, 50528519u, 50528525u, 50529025u, 50529033u, 50529539u, 50530049u, 50530055u,
    50530563u, 50531073u, 50531077u, 50532097u, 50532109u, 50659585u, 50660101u, 50660107u,
    50660111u, 50660609u, 50660617u, 50661125u, 50661633u, 50661639u, 50662155u, 50662657u,
    50663173u, 50790659u, 50790665u, 50790671u, 50791169u, 50791175u, 50791683u, 50791695u,
    50792193u, 50792201u, 50792707u, 50793733u, 50794241u, 50921735u, 50921739u, 50922245u,
    50922249u, 50923267u, 50923271u, 50923781u, 50923789u, 50924289u, 50924297u, 51052803u,
    51053313u, 51053319u, 51053827u, 51054337u, 51054341u, 51055363u, 51184897u, 51184905u,
    51184911u, 51185929u, 51185933u, 51314947u, 51314951u, 51315457u, 51315461u, 51315971u,
    51316491u, 51316995u, 51318021u, 51318529u, 83951873u, 83951875u, 83951879u, 83951883u,
    83951887u, 83952385u, 83952389u, 83952393u, 83952397u, 83952899u, 83952903u, 83952911u,
    83953409u, 83953413u, 83953923u, 83953927u, 83953931u, 83954433u, 83954437u, 83954959u,
    83955457u, 83955463u, 83955467u, 84082945u, 84082949u, 84083457u, 84083463u, 84083471u,
    84083973u, 84083979u, 84084483u, 84084489u, 84084997u, 84085507u, 84214019u, 84214025u,
    84214031u, 84215043u, 84215047u, 84215553u, 84215567u, 84216067u, 84216583u, 84216591u,
    84217603u, 84217609u, 84345089u, 84345093u, 84345099u, 84345603u, 84346117u, 84346121u,
    84346627u, 84346631u, 84347141u, 84347649u, 84348173u, 84476163u, 84476175u, 84477185u,
    84477191u, 84477701u, 84477707u, 84478211u, 84479749u, 84479755u, 84607241u, 84607747u,
    84608261u, 84608783u, 84609281u, 84609799u, 84610817u, 84738305u, 84738309u, 84738319u,
    84739331u, 84740875u, 84741379u, 84869387u, 84869891u, 84870413u, 84870913u, 84871431u,
    84871937u, 117506309u, 117506819u, 117506823u, 117506827u, 117506831u, 117507333u, 117507843u,
    117507847u, 117507851u, 117508357u, 117508361u, 117508367u, 117508867u, 117509383u, 117509891u,
    117637379u, 117637383u, 117637387u, 117637897u, 117638403u, 117638407u, 117639425u, 117640449u,
    117640965u, 117640973u, 117768449u, 117768965u, 117769473u, 117769989u, 117769993u, 117771009u,
    117899523u, 117900033u, 117900041u, 117900547u, 117900551u, 117900559u, 117901057u, 117901571u,
    117901575u, 117901583u, 117902091u, 117903111u, 118030599u, 118031107u, 118031117u, 118031621u,
    118032131u, 118033157u, 118033665u, 118033673u, 118161667u, 118162177u, 118162181u, 118162699u,
    118163205u, 118163721u, 118164237u, 118165255u, 118293261u, 118294787u, 118423811u, 118423815u,
    118424833u, 118424837u, 118425355u, 151060737u, 151060745u, 151061253u, 151061761u, 151061769u,
    151061775u, 151062277u, 151062787u, 151063297u, 151064321u, 151191813u, 151191823u, 151192323u,
    151192327u, 151192837u, 151193345u, 151193355u, 151193863u, 151194371u, 151194379u, 151322883u,
    151322887u, 151323393u, 151323403u, 151323907u, 151324423u, 151324929u, 151325455u, 151325957u,
    151326465u, 151453961u, 151454467u, 151454471u, 151454977u, 151454981u, 151455491u, 151455499u,
    151585025u, 151585029u, 151586057u, 151586575u, 151587073u, 151588611u, 151716107u, 151716111u,
    151717123u, 151719173u, 151847687u, 151848713u, 151850241u, 151978753u, 151978763u, 151979777u,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    151980295u, 151980803u, 184615173u, 184615681u, 184615689u, 184616197u, 184617217u, 184617225u,
    184617231u, 184617733u, 184618253u, 184618761u, 184746243u, 184746247u, 184746251u, 184746757u,
    184747267u, 184747781u, 184749829u, 184877313u, 184877827u, 184878343u, 184878849u, 184878861u,
    184879879u, 185008389u, 185008399u, 185008897u, 185009423u, 185010441u, 185010947u, 185011467u,
    185011975u, 185139459u, 185139465u, 185140481u, 185140997u, 185141517u, 185271045u, 185271565u,
    185273091u, 185273095u, 185403653u, 185532677u, 185532681u, 185533701u, 218170115u, 218170119u,
    218170123u, 218171139u, 218171143u, 218172673u, 218300673u, 218301697u, 218301711u, 218303753u,
    218432261u, 218433289u, 218433797u, 218434315u, 218434821u, 218435329u, 218562817u, 218563337u,
    218563843u, 218564865u, 218694923u, 218695943u, 218696965u, 218824961u, 218824967u, 218826505u,
    218828033u, 218956043u, 218958081u, 219087619u, 219087623u, 251724033u, 251724041u, 251724047u,
    251725057u, 251725061u, 251725581u, 251726081u, 251726601u, 251727109u, 251855109u, 251855619u,
    251856137u, 251857159u, 251857163u, 251986179u, 251986185u, 251986689u, 251986701u, 251987203u,
    251987713u, 251988739u, 252117253u, 252118789u, 252118795u, 252119815u, 252248323u, 252248331u,
    252248839u, 252249345u, 252250881u, 252380421u, 252381445u, 252510469u, 252512003u, 252641537u,
};

__device__ __forceinline__ float iq2s_value(
    const unsigned char* packed, int absolute
) {
    // 82 bytes per 256 values: d(2) qs[32] signs[32] qh[8] scales[8]. Signs are
    // stored literally here, and qh supplies two extra grid index bits.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 82;
    const float d = __half2float(*((const __half*)base));
    const int group = within / 16;
    const int rest = within & 15;
    const int half = rest / 8;
    const int element = rest & 7;
    const int index = group * 2 + half;
    const unsigned char* quants = base + 2;
    const unsigned char* signs = base + 34;
    const unsigned char* high = base + 66;
    const unsigned char* scales = base + 74;
    const int entry =
        quants[index] | (((high[index >> 2] >> (2 * (index & 3))) & 3) << 8);
    const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
    const float db = d * (0.5f + (float)scale) * 0.25f;
    const float value = (float)((kIq2sGrid[entry] >> (8 * element)) & 0xffULL);
    return ((signs[index] >> element) & 1) ? -db * value : db * value;
}

__device__ __forceinline__ float iq3s_value(
    const unsigned char* packed, int absolute
) {
    // 110 bytes per 256 values: d(2) qs[64] qh[8] signs[32] scales[4]. One high
    // index bit per entry comes from qh; the scale is an odd multiplier.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 110;
    const float d = __half2float(*((const __half*)base));
    const unsigned char* quants = base + 2;
    const unsigned char* high = base + 66;
    const unsigned char* signs = base + 74;
    const unsigned char* scales = base + 106;
    const int index = within / 4;
    const int element = within & 3;
    const int entry = quants[index] | (((high[index >> 3] >> (index & 7)) & 1) << 8);
    const int group = within / 32;
    const int quad = (within % 32) / 8;
    const int bit = within & 7;
    const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
    const float db = d * (float)(1 + 2 * scale);
    const float value = (float)((kIq3sGrid[entry] >> (8 * element)) & 0xffu);
    return ((signs[group * 4 + quad] >> bit) & 1) ? -db * value : db * value;
}

extern "C" __global__
void iq2s_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq2s_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void iq3s_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq3s_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(__device__ __constant__ unsigned long long kIq2xsGrid[512] = {
    578721382704613384ULL, 578721382704613419ULL, 578721382704617753ULL, 578721382704622344ULL,
    578721382704622379ULL, 578721382705727513ULL, 578721382705731848ULL, 578721382705731883ULL,
    578721382705736473ULL, 578721382706907144ULL, 578721382706907179ULL, 578721382706911513ULL,
    578721382706916104ULL, 578721382989826073ULL, 578721382989830408ULL, 578721382989830443ULL,
    578721382989835033ULL, 578721382990940168ULL, 578721382990940203ULL, 578721382990944537ULL,
    578721382990949128ULL, 578721382992119833ULL, 578721382992124168ULL, 578721383291815944ULL,
    578721383291815979ULL, 578721383291820313ULL, 578721383291824904ULL, 578721383292930073ULL,
    578721383292934408ULL, 578721383292939033ULL, 578721383294109704ULL, 578721455719057433ULL,
    578721455719061768ULL, 578721455719061803ULL, 578721455719066393ULL, 578721455720171528ULL,
    578721455720171563ULL, 578721455720175897ULL, 578721455720180488ULL, 578721455720180523ULL,
    578721455721351193ULL, 578721455721355528ULL, 578721456004270088ULL, 578721456004270123ULL,
    578721456004274457ULL, 578721456004279048ULL, 578721456005384217ULL, 578721456005388552ULL,
    578721456006563848ULL, 578721456006572808ULL, 578721456306259993ULL, 578721456306264328ULL,
    578721456307374088ULL, 578721533028468744ULL, 578721533028468779ULL, 578721533028473113ULL,
    578721533028477704ULL, 578721533029582873ULL, 578721533029587208ULL, 578721533030762504ULL,
    578721533313681433ULL, 578721533313685768ULL, 578721533314795528ULL, 578721533314799897ULL,
    578721533615671304ULL, 578721533615680299ULL, 578740074402285593ULL, 578740074402289928ULL,
    578740074402289963ULL, 578740074402294553ULL, 578740074403399688ULL, 578740074403399723ULL,
    578740074403404057ULL, 578740074403408648ULL, 578740074404579353ULL, 578740074404583688ULL,
    578740074687498248ULL, 578740074687498283ULL, 578740074687502617ULL, 578740074687507208ULL,
    578740074688612377ULL, 578740074688616712ULL, 578740074688616747ULL, 578740074689792008ULL,
    578740074989488153ULL, 578740074989492488ULL, 578740074990602248ULL, 578740147416729608ULL,
    578740147416729643ULL, 578740147416733977ULL, 578740147416738568ULL, 578740147417843737ULL,
    578740147417848072ULL, 578740147419023368ULL, 578740147701942297ULL, 578740147701946632ULL,
    578740147703056392ULL, 578740147704236057ULL, 578740148003932168ULL, 578740224726140953ULL,
    578740224726145288ULL, 578740224727255048ULL, 578740224728439083ULL, 578740225011353608ULL,
    578740225011353643ULL, 578740225313347848ULL, 578759865611585544ULL, 578759865611585579ULL,
    578759865611589913ULL, 578759865611594504ULL, 578759865611594539ULL, 578759865612699673ULL,
    578759865612704008ULL, 578759865613879304ULL, 578759865613883673ULL, 578759865896798233ULL,
    578759865896802568ULL, 578759865897912328ULL, 578759865897921288ULL, 578759866198788104ULL,
    578759866201081864ULL, 578759866201090859ULL, 578759938626029593ULL, 578759938626033928ULL,
    578759938627143688ULL, 578759938911242248ULL, 578759939213232153ULL, 578759939213241113ULL,
    578760015935440904ULL, 578760015937734664ULL, 578760015937743624ULL, 578760016523761963ULL,
    578760016524937224ULL, 583506457308694553ULL, 583506457308698888ULL, 583506457308698923ULL,
    583506457308703513ULL, 583506457309808648ULL, 583506457309808683ULL, 583506457309813017ULL,
    583506457309817608ULL, 583506457310988313ULL, 583506457310992648ULL, 583506457593907208ULL,
    583506457593907243ULL, 583506457593911577ULL, 583506457593916168ULL, 583506457595021337ULL,
    583506457595025672ULL, 583506457596200968ULL, 583506457596209963ULL, 583506457895897113ULL,
    583506457895901448ULL, 583506457897011208ULL, 583506530323138568ULL, 583506530323138603ULL,
    583506530323142937ULL, 583506530323147528ULL, 583506530324252697ULL, 583506530324257032ULL,
    583506530325432328ULL, 583506530608351257ULL, 583506530608355592ULL, 583506530609465352ULL,
    583506530910341128ULL, 583506530911459592ULL, 583506530911459627ULL, 583506607632549913ULL,
    583506607632554248ULL, 583506607632554283ULL, 583506607633664008ULL, 583506607917762568ULL,
    583506607920056328ULL, 583525149006366728ULL, 583525149006366763ULL, 583525149006371097ULL,
    583525149006375688ULL, 583525149007480857ULL, 583525149007485192ULL, 583525149008660488ULL,
    583525149291579417ULL, 583525149291583752ULL, 583525149291588377ULL, 583525149292693512ULL,
    583525149293877512ULL, 583525149593569288ULL, 583525222020810777ULL, 583525222020815112ULL,
    583525222021924872ULL, 583525222306023432ULL, 583525299330222088ULL, 583525299331340552ULL,
    583525299615443737ULL, 583544940215666713ULL, 583544940215671048ULL, 583544940216780808ULL,
    583544940216780843ULL, 583544940500879368ULL, 583544940501997832ULL, 583544940802873643ULL,
    583545013230110728ULL, 583545013230115097ULL, 583545013517621547ULL, 583545090825848857ULL,
    583545091129027353ULL, 588573006889486344ULL, 588573006889486379ULL, 588573006889490713ULL,
    588573006889495304ULL, 588573006889495339ULL, 588573006890600473ULL, 588573006890604808ULL,
    588573006891780104ULL, 588573007174699033ULL, 588573007174703368ULL, 588573007175813128ULL,
    588573007476688904ULL, 588573007478982664ULL, 588573079903930393ULL, 588573079903934728ULL,
    588573079905044488ULL, 588573080189143048ULL, 588573080189152008ULL, 588573080191441177ULL,
    588573157213341704ULL, 588573157215635499ULL, 588573157800544264ULL, 588573157802846984ULL,
    588591698587158553ULL, 588591698587162888ULL, 588591698588272648ULL, 588591698589461273ULL,
    588591698872371208ULL, 588591771601602568ULL, 588591771886815257ULL, 588591771887929387ULL,
    588591772189928217ULL, 588591848911013913ULL, 588591848912137003ULL, 588591849500514603ULL,
    588611489796458504ULL, 588611489796467464ULL, 588611489796467499ULL, 588611489798752264ULL,
    588611490082789657ULL, 588611490383670024ULL, 588611490385954859ULL, 588611563098417928ULL,
    588611563399219208ULL, 588611640120322824ULL, 588611640122607624ULL, 588611640707516459ULL,
    588611640707525384ULL, 588611640707525419ULL, 1803700481349388313ULL, 1803700481349392648ULL,
    1803700481349392683ULL, 1803700481349397273ULL, 1803700481350502408ULL, 1803700481350502443ULL,
    1803700481350506777ULL, 1803700481350511368ULL, 1803700481351682073ULL, 1803700481351686408ULL,
    1803700481634600968ULL, 1803700481634601003ULL, 1803700481634605337ULL, 1803700481634609928ULL,
    1803700481634609963ULL, 1803700481635715097ULL, 1803700481635719432ULL, 1803700481636894728ULL,
    1803700481636899097ULL, 1803700481936590873ULL, 1803700481936595208ULL, 1803700481937704968ULL,
    1803700554363832328ULL, 1803700554363832363ULL, 1803700554363836697ULL, 1803700554363841288ULL,
    1803700554364946457ULL, 1803700554364950792ULL, 1803700554366126088ULL, 1803700554649045017ULL,
    1803700554649049352ULL, 1803700554650159112ULL, 1803700554951034888ULL, 1803700554951039257ULL,
    1803700554953328683ULL, 1803700631673243673ULL, 1803700631673248008ULL, 1803700631674357768ULL,
    1803700631674357803ULL, 1803700631675546393ULL, 1803700631958456328ULL, 1803719173047060488ULL,
    1803719173047060523ULL, 1803719173047064857ULL, 1803719173047069448ULL, 1803719173048174617ULL,
    1803719173048178952ULL, 1803719173048183577ULL, 1803719173049354248ULL, 1803719173332273177ULL,
    1803719173332277512ULL, 1803719173333387272ULL, 1803719173634263048ULL, 1803719173635381512ULL,
    1803719246061504537ULL, 1803719246061508872ULL, 1803719246062618632ULL, 1803719246063802632ULL,
    1803719246346717192ULL, 1803719246649830187ULL, 1803719323370915848ULL, 1803719323370924843ULL,
    1803719323656132872ULL, 1803719323657242632ULL, 1803738964256360473ULL, 1803738964256364808ULL,
    1803738964257474568ULL, 1803738964541573128ULL, 1803738964541577497ULL, 1803738964542691592ULL,
    1803738964543866923ULL, 1803739037270804488ULL, 1803739037271918617ULL, 1803739037556021512ULL,
    1803739037557131272ULL, 1803739037558319897ULL, 1803739114580220168ULL, 1808485555953469448ULL,
    1808485555953469483ULL, 1808485555953473817ULL, 1808485555953478408ULL, 1808485555954583577ULL,
    1808485555954587912ULL, 1808485555955763208ULL, 1808485555955772168ULL, 1808485556238682137ULL,
    1808485556238686472ULL, 1808485556239796232ULL, 1808485556540672008ULL, 1808485628967913497ULL,
    1808485628967917832ULL, 1808485628969027592ULL, 1808485628969031961ULL, 1808485629253126152ULL,
    1808485629253126187ULL, 1808485706277324808ULL, 1808485706562541832ULL, 1808485706866830123ULL,
    1808504247651141657ULL, 1808504247651145992ULL, 1808504247652255752ULL, 1808504247653435417ULL,
    1808504247936354312ULL, 1808504247938648072ULL, 1808504248238344217ULL, 1808504248240637977ULL,
    1808504320665585672ULL, 1808504320665594632ULL, 1808504321252788232ULL, 1808504321252797192ULL,
    1808504397977290777ULL, 1808504398262512392ULL, 1808504398564493337ULL, 1808524038860441608ULL,
    1808524038861560072ULL, 1808524039145654297ULL, 1808524039146768392ULL, 1808524039448767257ULL,
    1808524111876008747ULL, 1808524112160098312ULL, 1808524112160098347ULL, 1808524189771503897ULL,
    1813552105534261273ULL, 1813552105534265608ULL, 1813552105535375368ULL, 1813552105819473928ULL,
    1813552105820592392ULL, 1813552105821767723ULL, 1813552106121468203ULL, 1813552106123766553ULL,
    1813552178548705288ULL, 1813552255860414728ULL, 1813552256143338283ULL, 1813552256446433323ULL,
    1813570797231933448ULL, 1813570797233051947ULL, 1813570870247491592ULL, 1813570870531590152ULL,
    1813570870531594521ULL, 1813570870835878152ULL, 1813590588441233433ULL, 1813590588728748843ULL,
    1813590661457975577ULL, 1813590738765093163ULL, 1813590739051419912ULL, 1813590739052595243ULL,
    3100737174032091144ULL, 3100737174032091179ULL, 3100737174032095513ULL, 3100737174032100104ULL,
    3100737174033205273ULL, 3100737174033209608ULL, 3100737174034384904ULL, 3100737174034393899ULL,
    3100737174317303833ULL, 3100737174317308168ULL, 3100737174318417928ULL, 3100737174619293704ULL,
    3100737174619293739ULL, 3100737174621596424ULL, 3100737174621596459ULL, 3100737247046535193ULL,
    3100737247046539528ULL, 3100737247046539563ULL, 3100737247047649288ULL, 3100737247331747848ULL,
    3100737247332861977ULL, 3100737247332870937ULL, 3100737324355946504ULL, 3100737324358240264ULL,
    3100737324943149064ULL, 3100737324943149099ULL, 3100737324945442824ULL, 3100737324945451784ULL,
    3100755865729763353ULL, 3100755865729767688ULL, 3100755865730877448ULL, 3100755865730877483ULL,
    3100755865730881817ULL, 3100755866014976008ULL, 3100755866017269768ULL, 3100755866316974873ULL,
    3100755938744207368ULL, 3100755939029424392ULL, 3100755939333708057ULL, 3100756016054741768ULL,
    3100756016341134123ULL, 3100775656939063304ULL, 3100775656939072264ULL, 3100775656941361433ULL,
    3100775657225399083ULL, 3100775657526265864ULL, 3100775657526265899ULL, 3100775657528568584ULL,
    3100775729953511723ULL, 3100775807265212459ULL, 3100775807850121224ULL, 3100775807850130184ULL,
    3100775807851239723ULL, 3100775807852423944ULL, 3105522248636172313ULL, 3105522248636176648ULL,
    3105522248637286408ULL, 3105522248921384968ULL, 3105522248922503467ULL, 3105522249223379208ULL,
    3105522321650616328ULL, 3105522321652910123ULL, 3105522321938127112ULL, 3105522399246358827ULL,
    3105522399547239193ULL, 3105540940333844488ULL, 3105540940333848857ULL, 3105540940619061512ULL,
    3105540940620171272ULL, 3105540940620180232ULL, 3105541013350591257ULL, 3105541013936605192ULL,
    3105541013936605227ULL, 3105541090942912537ULL, 3105560731829471257ULL, 3105560732132645163ULL,
    3105560804842810137ULL, 3105560881868118297ULL, 3105560882154506248ULL, 3110588798216964104ULL,
    3110588798216964139ULL, 3110588798216973064ULL, 3110588798216973099ULL, 3110588798219257864ULL,
    3110588798219266859ULL, 3110588798806460424ULL, 3110588871517734937ULL, 3110588871517743897ULL,
    3110588871820908843ULL, 3110588948540819464ULL, 3110588948540819499ULL, 3110588948540828424ULL,
    3110588948543122219ULL, 3110588949128022024ULL, 3110588949130315784ULL, 3110607490199848968ULL,
    3110607490502957337ULL, 3110607640526002457ULL, 3110607640826817288ULL, 3110627281123945259ULL,
    3110627281126230024ULL, 3110627281126230059ULL, 3110627281126238984ULL, 3110627281713432584ULL,
    3110627281713441544ULL, 3110627354138384648ULL, 3110627354725587208ULL, 3110627354725587243ULL,
    3110627431450094344ULL, 3110627431450094379ULL, 3110627432036108313ULL, 3110627432037296939ULL,
};
__device__ __constant__ signed char kIq4nlValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

__device__ __forceinline__ float iq2xs_value(
    const unsigned char* packed, int absolute
) {
    // 74 bytes per 256 values: d(2) qs[32] as uint16, scales[8]. Each uint16
    // carries a 9-bit grid index and a 7-bit sign selector.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 74;
    const float d = __half2float(*((const __half*)base));
    const int index = within / 8;
    const int element = within & 7;
    unsigned short entry;
    memcpy(&entry, base + 2 + index * 2, 2);
    const int group = within / 16;
    const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
    const float db = d * (0.5f + (float)scale) * 0.25f;
    const unsigned char signs = kIq2xxsSigns[entry >> 9];
    const float value = (float)((kIq2xsGrid[entry & 511] >> (8 * element)) & 0xffULL);
    return ((signs >> element) & 1) ? -db * value : db * value;
}

__device__ __forceinline__ float iq4xs_value(
    const unsigned char* packed, int absolute
) {
    // 136 bytes per 256 values: d(2) scales_h(2) scales_l[4] qs[128]. Each
    // 4-bit code indexes the non-uniform IQ4_NL levels; each 32-value
    // sub-block has a 6-bit signed scale split across scales_l and scales_h.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 136;
    const float d = __half2float(*((const __half*)base));
    unsigned short scales_high;
    memcpy(&scales_high, base + 2, 2);
    const int sub = within / 32;
    const int element = within & 31;
    const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
    const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
    const unsigned char byte = base[8 + sub * 16 + (element & 15)];
    const int code = element < 16 ? (byte & 15) : (byte >> 4);
    return d * (float)scale * (float)kIq4nlValues[code];
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(extern "C" __global__
void iq2xs_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq2xs_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void iq4xs_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq4xs_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

// Grouped routed-expert kernels for the IQ codebook formats, which had none:
// the host dispatch ended in a k-quant fallback, so an IQ expert reaching the
// GPU was decoded as Q5_K. Low-bit MoE checkpoints quantize routed experts this
// way even when their dense projections are k-quants, so without these the
// whole expert phase is pinned to the CPU.
//
// These decode an octet at a time rather than a value at a time. Every IQ
// format packs 256 values as 32 groups of eight that share one grid entry, one
// sign byte and one scale, so the per-value decoders redo the block header,
// the codebook lookup and the scale extraction eight times over. Amortizing
// that across the octet is the difference between these kernels being worth
// running on the GPU at all and merely matching the vectorized CPU path.
__device__ __forceinline__ void iq2xs_octet(
    const unsigned char* packed, int block, int octet, float* out
) {
    const unsigned char* base = packed + block * 74;
    const float d = __half2float(*((const __half*)base));
    const int group = octet >> 1;
    const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
    const float db = d * (0.5f + (float)scale) * 0.25f;
    unsigned short entry;
    memcpy(&entry, base + 2 + octet * 2, 2);
    const unsigned long long grid = kIq2xsGrid[entry & 511];
    const unsigned char signs = kIq2xxsSigns[entry >> 9];
    for (int k = 0; k < 8; ++k) {
        const float value = (float)((grid >> (8 * k)) & 0xffULL);
        out[k] = ((signs >> k) & 1) ? -db * value : db * value;
    }
}

__device__ __forceinline__ void iq3xxs_octet(
    const unsigned char* packed, int block, int octet, float* out
) {
    const unsigned char* base = packed + block * 98;
    const float d = __half2float(*((const __half*)base));
    const int group = octet >> 2;
    const int quad = octet & 3;
    unsigned int aux;
    memcpy(&aux, base + 2 + 64 + group * 4, 4);
    const float scale = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const unsigned char* indices = base + 2 + group * 8 + quad * 2;
    const unsigned char signs = kIq2xxsSigns[(aux >> (7 * quad)) & 127];
    // The quad's eight magnitudes are two 4-byte grid entries end to end.
    const unsigned long long grid =
        (unsigned long long)kIq3xxsGrid[indices[0]] |
        ((unsigned long long)kIq3xxsGrid[indices[1]] << 32);
    for (int k = 0; k < 8; ++k) {
        const float value = (float)((grid >> (8 * k)) & 0xffULL);
        out[k] = ((signs >> k) & 1) ? -scale * value : scale * value;
    }
}

__device__ __forceinline__ void iq4xs_octet(
    const unsigned char* packed, int block, int octet, float* out
) {
    const unsigned char* base = packed + block * 136;
    unsigned short scales_high;
    memcpy(&scales_high, base + 2, 2);
    const float d = __half2float(*((const __half*)base));
    const int sub = octet >> 2;
    const int part = octet & 3;
    const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
    const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
    const float ds = d * (float)scale;
    // Low nibbles supply the sub-block's first sixteen values, high nibbles the
    // second sixteen, so the octet's nibble half follows from its index.
    const unsigned char* quants = base + 8 + sub * 16 + (part & 1) * 8;
    const int high = part >> 1;
    for (int k = 0; k < 8; ++k) {
        const unsigned char byte = quants[k];
        out[k] = ds * (float)kIq4nlValues[high ? (byte >> 4) : (byte & 15)];
    }
}

#define COLIBRI_IQ_GROUPED(prefix, octet_at)                                    \
extern "C" __global__                                                           \
void prefix##_grouped_swiglu(                                                   \
    const unsigned long long* gate_ptrs, const unsigned long long* up_ptrs,     \
    const float* vector, float* activated,                                      \
    const int input_size, const int output_size, const int experts             \
) {                                                                             \
    const int row = blockIdx.x;                                                 \
    const int expert = blockIdx.y;                                              \
    if (row >= output_size || expert >= experts) return;                        \
    const unsigned char* gate_packed = (const unsigned char*)gate_ptrs[expert]; \
    const unsigned char* up_packed = (const unsigned char*)up_ptrs[expert];     \
    const int octets = input_size >> 3;                                         \
    const long long row_base = (long long)row * input_size;                     \
    float gate = 0.0f, up = 0.0f, g[8], u[8];                                   \
    for (int octet = threadIdx.x; octet < octets; octet += blockDim.x) {        \
        const long long absolute = row_base + (long long)octet * 8;             \
        const int block = (int)(absolute >> 8);                                 \
        const int within = (int)((absolute & 255) >> 3);                        \
        octet_at(gate_packed, block, within, g);                                \
        octet_at(up_packed, block, within, u);                                  \
        const float* values = vector + octet * 8;                               \
        for (int k = 0; k < 8; ++k) {                                           \
            gate += g[k] * values[k];                                           \
            up += u[k] * values[k];                                             \
        }                                                                       \
    }                                                                           \
    gate = block_reduce_sum(gate);                                              \
    up = block_reduce_sum(up);                                                  \
    if (threadIdx.x == 0)                                                       \
        activated[expert * output_size + row] =                                 \
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;    \
}                                                                               \
extern "C" __global__                                                           \
void prefix##_grouped_swiglu_rows(                                              \
    const unsigned long long* gate_ptrs, const unsigned long long* up_ptrs,     \
    const int* counts, const float* vectors, float* activated,                  \
    const int input_size, const int output_size, const int top_k,               \
    const int rows                                                              \
) {                                                                             \
    const int row = blockIdx.x;                                                 \
    const int route = blockIdx.y;                                               \
    const int token = route / top_k;                                            \
    const int rank = route - token * top_k;                                     \
    if (row >= output_size || token >= rows || rank >= counts[token]) return;   \
    const unsigned char* gate_packed = (const unsigned char*)gate_ptrs[route];  \
    const unsigned char* up_packed = (const unsigned char*)up_ptrs[route];      \
    const float* vector = vectors + token * input_size;                         \
    const int octets = input_size >> 3;                                         \
    const long long row_base = (long long)row * input_size;                     \
    float gate = 0.0f, up = 0.0f, g[8], u[8];                                   \
    for (int octet = threadIdx.x; octet < octets; octet += blockDim.x) {        \
        const long long absolute = row_base + (long long)octet * 8;             \
        const int block = (int)(absolute >> 8);                                 \
        const int within = (int)((absolute & 255) >> 3);                        \
        octet_at(gate_packed, block, within, g);                                \
        octet_at(up_packed, block, within, u);                                  \
        const float* values = vector + octet * 8;                               \
        for (int k = 0; k < 8; ++k) {                                           \
            gate += g[k] * values[k];                                           \
            up += u[k] * values[k];                                             \
        }                                                                       \
    }                                                                           \
    gate = block_reduce_sum(gate);                                              \
    up = block_reduce_sum(up);                                                  \
    if (threadIdx.x == 0)                                                       \
        activated[route * output_size + row] =                                  \
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;    \
}                                                                               \
extern "C" __global__                                                           \
void prefix##_grouped_accumulate(                                               \
    const unsigned long long* down_ptrs, const float* activated,                \
    float* output, const float* weights,                                        \
    const int input_size, const int output_size, const int experts             \
) {                                                                             \
    const int row = blockIdx.x;                                                 \
    if (row >= output_size) return;                                             \
    const int octets = input_size >> 3;                                         \
    const long long row_base = (long long)row * input_size;                     \
    float partial = 0.0f, w[8];                                                 \
    for (int octet = threadIdx.x; octet < octets; octet += blockDim.x) {        \
        const long long absolute = row_base + (long long)octet * 8;             \
        const int block = (int)(absolute >> 8);                                 \
        const int within = (int)((absolute & 255) >> 3);                        \
        for (int expert = 0; expert < experts; ++expert) {                      \
            octet_at((const unsigned char*)down_ptrs[expert], block, within, w);\
            const float weight = weights[expert];                               \
            const float* values = activated + expert * input_size + octet * 8;  \
            for (int k = 0; k < 8; ++k) partial += weight * w[k] * values[k];   \
        }                                                                       \
    }                                                                           \
    partial = block_reduce_sum(partial);                                        \
    if (threadIdx.x == 0) output[row] += partial;                               \
}                                                                               \
extern "C" __global__                                                           \
void prefix##_grouped_accumulate_rows(                                          \
    const unsigned long long* down_ptrs, const float* activated,                \
    float* output, const float* weights, const int* counts,                     \
    const int input_size, const int output_size, const int top_k,               \
    const int rows                                                              \
) {                                                                             \
    const int row = blockIdx.x;                                                 \
    const int token = blockIdx.y;                                               \
    if (row >= output_size || token >= rows) return;                            \
    const int base = token * top_k;                                             \
    const int count = counts[token];                                            \
    const int octets = input_size >> 3;                                         \
    const long long row_base = (long long)row * input_size;                     \
    float partial = 0.0f, w[8];                                                 \
    for (int octet = threadIdx.x; octet < octets; octet += blockDim.x) {        \
        const long long absolute = row_base + (long long)octet * 8;             \
        const int block = (int)(absolute >> 8);                                 \
        const int within = (int)((absolute & 255) >> 3);                        \
        for (int rank = 0; rank < count; ++rank) {                              \
            const int route = base + rank;                                      \
            octet_at((const unsigned char*)down_ptrs[route], block, within, w); \
            const float weight = weights[route];                                \
            const float* values = activated + route * input_size + octet * 8;   \
            for (int k = 0; k < 8; ++k) partial += weight * w[k] * values[k];   \
        }                                                                       \
    }                                                                           \
    partial = block_reduce_sum(partial);                                        \
    if (threadIdx.x == 0) output[token * output_size + row] += partial;         \
}

COLIBRI_IQ_GROUPED(iq2xs, iq2xs_octet)
COLIBRI_IQ_GROUPED(iq3xxs, iq3xxs_octet)
COLIBRI_IQ_GROUPED(iq4xs, iq4xs_octet)

#undef COLIBRI_IQ_GROUPED

__device__ __forceinline__ float q2k_value(
    const unsigned char* packed, int absolute
) {
    // Q2_K: 84 bytes per 256 values -> scales[16] qs[64] d(2) dmin(2). Each
    // scales byte packs a 4-bit scale (low nibble) and a 4-bit min (high
    // nibble) for one 16-element group. The 256 values are two 128-element
    // halves; within a half the 2-bit quants for group j sit at bit offset
    // 2*j of that half's 32 qs bytes.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 84;
    const float d = __half2float(*((const __half*)(base + 80)));
    const float dmin = __half2float(*((const __half*)(base + 82)));
    const int half = within / 128;
    const int rest = within & 127;
    const int group = rest / 32;
    const int lane = rest & 31;
    const int sub = lane / 16;
    const int element = lane & 15;
    const int quant = (base[16 + half * 32 + sub * 16 + element] >> (2 * group)) & 3;
    const unsigned char scale_byte = base[half * 8 + group * 2 + sub];
    return d * (float)(scale_byte & 15) * (float)quant
        - dmin * (float)(scale_byte >> 4);
}

// The sixteen 6-bit Q3_K scales are packed into 12 bytes: the low and high
// nibbles of bytes 0..7 carry each scale's low 4 bits, and bytes 8..11 supply
// the top 2 bits.
__device__ __forceinline__ int q3k_scale(
    const unsigned char* scales, int index
) {
    const int group = index / 4;
    const int byte = index & 3;
    const int packed_low = scales[(group & 1) ? 4 + byte : byte];
    const int nibble = (group < 2) ? (packed_low & 15) : (packed_low >> 4);
    return nibble | (((scales[8 + byte] >> (2 * group)) & 3) << 4);
}

__device__ __forceinline__ float q3k_value(
    const unsigned char* packed, int absolute
) {
    // Q3_K: 110 bytes per 256 values -> hmask[32] qs[64] scales[12] d(2). The
    // quant is a 2-bit low part from qs plus an inverted high bit from hmask
    // (a set mask bit means "do not subtract 4"), giving a signed 3-bit value.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 110;
    const float d = __half2float(*((const __half*)(base + 108)));
    const int half = within / 128;
    const int rest = within & 127;
    const int group = rest / 32;
    const int lane = rest & 31;
    const int sub = lane / 16;
    const int element = lane & 15;
    const int low = (base[32 + half * 32 + sub * 16 + element] >> (2 * group)) & 3;
    const int high = (base[lane] & (1 << (half * 4 + group))) ? 0 : 4;
    const int scale = q3k_scale(base + 96, half * 8 + group * 2 + sub);
    return d * (float)(scale - 32) * (float)(low - high);
}


extern "C" __global__
void q2k_matvec_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q2k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q3k_matvec_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q3k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

__device__ __forceinline__ float q4k_value(
    const unsigned char* packed, int absolute
) {
    // Q4_K: like Q5_K but with no 5th-bit (qh) array, so qs starts at +16 and
    // the quant is a plain 4-bit nibble. Shares Q5_K's 6-bit scale/min packing.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 144;
    const float d = __half2float(*((const __half*)(base)));
    const float dmin = __half2float(*((const __half*)(base + 2)));
    const unsigned char* scales = base + 4;
    const int group = within / 64;
    const int offset = within & 63;
    const int sub = offset / 32;
    const int qindex = group * 32 + (offset & 31);
    const unsigned char low = base[16 + qindex];
    const int quant = (offset < 32) ? (low & 15) : (low >> 4);
    int scale, minimum;
    q5k_scale_min(scales, group * 2 + sub, &scale, &minimum);
    return d * (float)scale * (float)quant - dmin * (float)minimum;
}

extern "C" __global__
void q4k_grouped_swiglu(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate_packed =
        (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up_packed =
        (const unsigned char*)up_ptrs[expert];
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q4k_value(gate_packed, absolute) * value;
        up += q4k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q4k_grouped_swiglu_rows(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const int* counts,
    const float* vectors,
    float* activated,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int route = blockIdx.y;
    const int token = route / top_k;
    const int rank = route - token * top_k;
    if (row >= output_size || token >= rows || rank >= counts[token]) return;
    const unsigned char* gate_packed =
        (const unsigned char*)gate_ptrs[route];
    const unsigned char* up_packed =
        (const unsigned char*)up_ptrs[route];
    const float* vector = vectors + token * input_size;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q4k_value(gate_packed, absolute) * value;
        up += q4k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[route * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q4k_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[expert];
            combined += weights[expert]
                * q4k_value(packed, row * input_size + input)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

extern "C" __global__
void q4k_grouped_accumulate_rows(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int* counts,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= output_size || token >= rows) return;
    const int base = token * top_k;
    const int count = counts[token];
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int rank = 0; rank < count; ++rank) {
            const int route = base + rank;
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[route];
            combined += weights[route]
                * q4k_value(packed, row * input_size + input)
                * activated[route * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * output_size + row] += partial;
}

extern "C" __global__
void q4k_matvec_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q4k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q4k_lm_head_argmax_warp(
    const unsigned char* packed,
    const float* vector,
    unsigned long long* winners,
    const int input_size,
    const int output_size
) {
    __shared__ unsigned long long warp_best[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (lane == 0) warp_best[warp] = 0ull;
    __syncthreads();
    if (row < output_size) {
        float partial = 0.0f;
        for (int input = lane; input < input_size; input += 32)
            partial += q4k_value(packed, row * input_size + input) * vector[input];
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (lane == 0) {
            const unsigned int bits = __float_as_uint(partial);
            const unsigned int ordered = bits ^ (
                ((int)bits < 0) ? 0xffffffffu : 0x80000000u
            );
            warp_best[warp] =
                ((unsigned long long)ordered << 32)
                | (unsigned int)(0xffffffffu - (unsigned int)row);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long best = warp_best[0];
        for (int i = 1; i < 8; ++i) best = max(best, warp_best[i]);
        atomicMax(winners, best);
    }
}

)COLIBRI_CUDA" R"COLIBRI_CUDA(


__device__ __forceinline__ float q6k_value(
    const unsigned char* packed, int absolute
) {
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 210;
    const unsigned char* ql = base;
    const unsigned char* qh = base + 128;
    const signed char* scales = (const signed char*)(base + 192);
    const float d = __half2float(*((const __half*)(base + 208)));
    const int half = within / 128;
    const int offset = within & 127;
    const int lane = offset / 32;
    const int l = offset & 31;
    const int qindex = l + ((lane == 0 || lane == 2) ? 0 : 32);
    const unsigned char qbyte = ql[half * 64 + qindex];
    const unsigned char high = qh[half * 32 + l];
    const int nibble = (lane == 0 || lane == 1) ? (qbyte & 15) : (qbyte >> 4);
    const int quant = (nibble | (((high >> (lane * 2)) & 3) << 4)) - 32;
    const int scale_index = half * 8 + (l / 16) + lane * 2;
    return d * (float)scales[scale_index] * (float)quant;
}

// Decode one complete 256-value Q6_K super-block per warp.  The scalar helper
// above reloads ql/qh, d and the group scales for every reconstructed value.
// Keeping a lane fixed at l=[0,31] lets it reuse two low bytes and one high
// byte for each 128-value half, while the warp loads the 16 group scales and
// the common multiplier exactly once.
__device__ __forceinline__ float q6k_row_dot_warp(
    const unsigned char* packed,
    const float* vector,
    const int row,
    const int input_size,
    const int lane
) {
    // GGML K-quant matrix rows used by Qwen are block aligned. Retain the
    // scalar path for an unusual non-aligned width so this remains a general
    // matvec entry point rather than silently changing its addressing.
    if (input_size & 255) {
        float partial = 0.0f;
        for (int input = lane; input < input_size; input += 32)
            partial += q6k_value(
                packed, row * input_size + input) * vector[input];
        return partial;
    }

    const int blocks_per_row = input_size >> 8;
    const unsigned char* row_data =
        packed + (unsigned long long)row * blocks_per_row * 210ull;
    float partial = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
        const unsigned char* base = row_data + block * 210;
        const signed char* scales = (const signed char*)(base + 192);
        const int scale_value = lane < 16 ? (int)scales[lane] : 0;
        const float d = __shfl_sync(
            0xffffffffu,
            lane == 0 ? __half2float(*((const __half*)(base + 208))) : 0.0f,
            0);

        const unsigned char low_0 = base[lane];
        const unsigned char low_1 = base[32 + lane];
        const unsigned char high_0 = base[128 + lane];
        const unsigned char low_2 = base[64 + lane];
        const unsigned char low_3 = base[96 + lane];
        const unsigned char high_1 = base[160 + lane];
        const int input_base = block << 8;
        const int scale_group = lane >> 4;
        float block_partial = 0.0f;

        #pragma unroll
        for (int segment = 0; segment < 4; ++segment) {
            const unsigned char low =
                (segment & 1) ? low_1 : low_0;
            const int nibble =
                segment < 2 ? (low & 15) : (low >> 4);
            const int quant =
                (nibble | (((high_0 >> (segment * 2)) & 3) << 4)) - 32;
            const int scale = __shfl_sync(
                0xffffffffu, scale_value, scale_group + segment * 2);
            block_partial = fmaf(
                (float)(scale * quant),
                vector[input_base + segment * 32 + lane],
                block_partial);
        }
        #pragma unroll
        for (int segment = 0; segment < 4; ++segment) {
            const unsigned char low =
                (segment & 1) ? low_3 : low_2;
            const int nibble =
                segment < 2 ? (low & 15) : (low >> 4);
            const int quant =
                (nibble | (((high_1 >> (segment * 2)) & 3) << 4)) - 32;
            const int scale = __shfl_sync(
                0xffffffffu, scale_value, 8 + scale_group + segment * 2);
            block_partial = fmaf(
                (float)(scale * quant),
                vector[input_base + 128 + segment * 32 + lane],
                block_partial);
        }
        partial = fmaf(d, block_partial, partial);
    }
    return partial;
}

COLIBRI_LM_HEAD_ARGMAX(f32_lm_head_argmax_warp, f32_head_value)
COLIBRI_LM_HEAD_ARGMAX(q2k_lm_head_argmax_warp, q2k_value)
COLIBRI_LM_HEAD_ARGMAX(q3k_lm_head_argmax_warp, q3k_value)
COLIBRI_LM_HEAD_ARGMAX(q5k_lm_head_argmax_warp, q5k_value)
COLIBRI_LM_HEAD_ARGMAX(iq2xxs_lm_head_argmax_warp, iq2xxs_value)
COLIBRI_LM_HEAD_ARGMAX(iq3xxs_lm_head_argmax_warp, iq3xxs_value)
COLIBRI_LM_HEAD_ARGMAX(iq2s_lm_head_argmax_warp, iq2s_value)
COLIBRI_LM_HEAD_ARGMAX(iq3s_lm_head_argmax_warp, iq3s_value)
COLIBRI_LM_HEAD_ARGMAX(iq2xs_lm_head_argmax_warp, iq2xs_value)
COLIBRI_LM_HEAD_ARGMAX(iq4xs_lm_head_argmax_warp, iq4xs_value)

#undef COLIBRI_LM_HEAD_ARGMAX


extern "C" __global__
void q6k_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q6k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void q6k_matvec_transposed_warp(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    // Warp per row, eight rows per block. Q6_K's 210-byte blocks interleave
    // low nibbles, high bits and scales, so there is no 128-bit load to be
    // had here; the win is dropping the shared-memory reduction and barriers.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float partial =
        q6k_row_dot_warp(packed, vector, row, input_size, lane);
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}

extern "C" __global__
void q6k_lm_head_argmax_warp(
    const unsigned char* packed,
    const float* vector,
    unsigned long long* winners,
    const int input_size,
    const int output_size
) {
    __shared__ unsigned long long warp_best[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (lane == 0) warp_best[warp] = 0ull;
    __syncthreads();
    if (row < output_size) {
        float partial =
            q6k_row_dot_warp(packed, vector, row, input_size, lane);
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (lane == 0) {
            const unsigned int bits = __float_as_uint(partial);
            const unsigned int ordered = bits ^ (
                ((int)bits < 0) ? 0xffffffffu : 0x80000000u
            );
            warp_best[warp] =
                ((unsigned long long)ordered << 32)
                | (unsigned int)(0xffffffffu - (unsigned int)row);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long best = warp_best[0];
        for (int i = 1; i < 8; ++i) best = max(best, warp_best[i]);
        atomicMax(winners, best);
    }
}

extern "C" __global__
void q6k_grouped_swiglu(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate_packed =
        (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up_packed =
        (const unsigned char*)up_ptrs[expert];
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q6k_value(gate_packed, absolute) * value;
        up += q6k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q6k_grouped_swiglu_rows(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const int* counts,
    const float* vectors,
    float* activated,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int route = blockIdx.y;
    const int token = route / top_k;
    const int rank = route - token * top_k;
    if (row >= output_size || token >= rows || rank >= counts[token]) return;
    const unsigned char* gate_packed =
        (const unsigned char*)gate_ptrs[route];
    const unsigned char* up_packed =
        (const unsigned char*)up_ptrs[route];
    const float* vector = vectors + token * input_size;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += q6k_value(gate_packed, absolute) * value;
        up += q6k_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[route * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q6k_accumulate_transposed(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const float* weights,
    const int weight_index,
    const int input_size,
    const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += q6k_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += weights[weight_index] * partial;
}

extern "C" __global__
void q6k_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[expert];
            combined += weights[expert]
                * q6k_value(packed, row * input_size + input)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

extern "C" __global__
void q6k_grouped_accumulate_rows(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int* counts,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= output_size || token >= rows) return;
    const int base = token * top_k;
    const int count = counts[token];
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int rank = 0; rank < count; ++rank) {
            const int route = base + rank;
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[route];
            combined += weights[route]
                * q6k_value(packed, row * input_size + input)
                * activated[route * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * output_size + row] += partial;
}

extern "C" __global__
void q8_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[expert];
            const float scale = __half2float(
                *((const __half*)(packed + block * 34))
            );
            const signed char value = *((const signed char*)(
                packed + block * 34 + 2 + within
            ));
            combined += weights[expert] * ((float)value * scale)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

extern "C" __global__
void q8_grouped_accumulate_rows(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int* counts,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= output_size || token >= rows) return;
    const int base = token * top_k;
    const int count = counts[token];
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        float combined = 0.0f;
        for (int rank = 0; rank < count; ++rank) {
            const int route = base + rank;
            const unsigned char* packed =
                (const unsigned char*)down_ptrs[route];
            const float scale = __half2float(
                *((const __half*)(packed + block * 34))
            );
            const signed char value = *((const signed char*)(
                packed + block * 34 + 2 + within
            ));
            combined += weights[route] * ((float)value * scale)
                * activated[route * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * output_size + row] += partial;
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// ---- NVFP4 (GGML type 40): E2M1 4-bit float (1 sign + 2 exp + 1 mantissa, bias=1) -
// 36 bytes per 64 elements: d[4] E4M3 scales then qs[32] packed nibbles, with
// scale i governing bytes 4+8i..11+8i (16 elements). Nibbles are split-half like
// q4_0/q4_K: qs[lane] holds element `lane` low, `lane+8` high, within the
// sub-block. Must stay in lockstep with qwen_nvfp4_value() in v2_runtime.cpp.
__device__ __forceinline__ float ue4m3_to_float(unsigned char bits) {
    // OCP FN E4M3: e==0xF is finite (max 448 at 0x7E); only 0x7F/0xFF are NaN.
    // Real checkpoints do use e==0xF, so decoding it as infinity yields NaN rows.
    const int s = (bits >> 7) & 1;
    const int e = (bits >> 3) & 0xF;
    const int m = bits & 7;
    if (e == 0) {
        const float val = (float)m * (1.0f / 512.0f);
        return s ? -val : val;
    }
    // E4M3's three mantissa bits map directly to the high three f32 mantissa
    // bits. Avoid ldexpf in every weight decode.
    unsigned int widened = ((unsigned int)s << 31)
        | ((unsigned int)(e + 120) << 23) | ((unsigned int)m << 20);
    if (e == 0xF && m == 7)
        widened = ((unsigned int)s << 31) | 0x7fc00000u;
    return __uint_as_float(widened);
}
// S(1) E(2) M(1) -> float, bias 1:
//   0x0=0.0  0x1=0.5  0x2=1.0  0x3=1.5  0x4=2.0  0x5=3.0  0x6=4.0  0x7=6.0
// This was a `const float lut[16]` local to nvfp4_value, which nvcc placed in
// local memory and re-initialised on every call -- four st.local.v4 (64 B)
// per decoded weight, inside the expert GEMV inner loop. Assembling the f32
// bits instead keeps it in registers. Verify with `nvcc -ptx` that no NVFP4
// kernel contains `.local` before changing this.
__device__ __forceinline__ float fp4_e2m1_to_float(int val) {
    const unsigned int magnitude = (unsigned int)(val & 7);
    const unsigned int exponent = magnitude >> 1;
    // exponent==0 is the subnormal pair {0.0, 0.5}; otherwise the f32 exponent
    // is (exponent - 1) + 127 and the mantissa bit selects the x1.5 variant.
    const unsigned int bits = exponent
        ? (((exponent + 126u) << 23) | ((magnitude & 1u) << 22))
        : (magnitude ? 0x3F000000u : 0u);
    return __uint_as_float(bits | ((unsigned int)(val & 8) << 28));
}

__device__ __forceinline__ float nvfp4_value(
    const unsigned char* packed, int absolute
) {
    const int block = absolute >> 6;
    const int offset = absolute & 63;
    const int sub = offset >> 4;
    const int within = offset & 15;
    const unsigned char* base = packed + block * 36;
    // Every caller walks a row with one contiguous warp: each 16-lane
    // half-warp shares one E4M3 scale and each pair eight lanes apart shares
    // one packed byte. Broadcast those values instead of issuing 16 duplicate
    // scale loads/conversions and two duplicate nibble-byte loads.
    const int lane = threadIdx.x & 31;
    const int half_lane = lane & 15;
    int scale_bits = half_lane == 0 ? (int)base[sub] : 0;
    scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & 16);
    int byte = (half_lane & 8) == 0
        ? (int)base[4 + sub * 8 + (within & 7)] : 0;
    byte = __shfl_sync(0xffffffffu, byte, lane & ~8);
    const float scale = ue4m3_to_float((unsigned char)scale_bits);
    const int val = (within < 8) ? (byte & 0x0F) : (byte >> 4);
    return scale * fp4_e2m1_to_float(val);
}

// cuBLASLt's block-scaled FP4 Tensor Core path consumes E2M1 values and
// UE4M3 scales in separate arrays. GGUF type 40 interleaves four scales with
// 32 value bytes for every 64 weights, so split it on-device before GEMM.
// Scale factors use the 128x4 tiled layout required by
// CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3.
__device__ __forceinline__ unsigned long long nvfp4_scale_offset(
    int outer, int inner, int inner_scales
) {
    const int outer_tile = outer >> 7;
    const int inner_tile = inner >> 2;
    const int outer_in_tile = outer & 127;
    const unsigned long long tile =
        (unsigned long long)(outer_tile * ((inner_scales + 3) >> 2) + inner_tile);
    return tile * 512ull
        + (unsigned long long)(outer_in_tile & 31) * 16ull
        + (unsigned long long)(outer_in_tile >> 5) * 4ull
        + (unsigned long long)(inner & 3);
}

// GGUF stores each 16-value sub-block as eight bytes pairing lanes (i, i+8).
// CUDA FP4 GEMM operands instead store adjacent values in each byte: (2i,2i+1).
// Reorder the nibbles while separating GGUF's four leading scale bytes.
__device__ __forceinline__ void nvfp4_copy_gguf_values_cublaslt(
    const unsigned char* source,
    unsigned char* destination
) {
    for (int sub = 0; sub < 4; ++sub) {
        const unsigned char* packed = source + 4 + sub * 8;
        for (int pair = 0; pair < 8; ++pair) {
            const int first_index = pair * 2;
            const int second_index = first_index + 1;
            const unsigned char first_byte = packed[first_index & 7];
            const unsigned char second_byte = packed[second_index & 7];
            const unsigned char first = first_index < 8
                ? first_byte & 0x0f : first_byte >> 4;
            const unsigned char second = second_index < 8
                ? second_byte & 0x0f : second_byte >> 4;
            destination[sub * 8 + pair] =
                static_cast<unsigned char>(first | (second << 4));
        }
    }
}

extern "C" __global__
void nvfp4_repack_cublaslt(
    const unsigned char* gguf,
    unsigned char* values,
    unsigned char* scales,
    const int rows,
    const int columns
) {
    const int blocks_per_row = columns >> 6;
    const int block_count = rows * blocks_per_row;
    for (int block = blockIdx.x * blockDim.x + threadIdx.x;
         block < block_count; block += blockDim.x * gridDim.x) {
        const int row = block / blocks_per_row;
        const int column_block = block - row * blocks_per_row;
        const unsigned char* source = gguf + (unsigned long long)block * 36ull;
        unsigned char* destination =
            values + (unsigned long long)row * (columns >> 1)
            + (unsigned long long)column_block * 32ull;
        nvfp4_copy_gguf_values_cublaslt(source, destination);
        const int inner_scales = columns >> 4;
        for (int sub = 0; sub < 4; ++sub) {
            const int inner = column_block * 4 + sub;
            scales[nvfp4_scale_offset(row, inner, inner_scales)] = source[sub];
        }
    }
}

extern "C" __global__
void nvfp4_quantize_cublaslt(
    const float* input,
    unsigned char* values,
    unsigned char* scales,
    const int rows,
    const int columns
) {
    const int scaled_block = blockIdx.x;
    const int blocks_per_row = columns >> 4;
    const int row = scaled_block / blocks_per_row;
    const int inner = scaled_block - row * blocks_per_row;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    float value = lane < 16
        ? input[(unsigned long long)row * columns + inner * 16 + lane] : 0.0f;
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    maximum = __shfl_sync(0xffffffff, maximum, 0);

    unsigned int scale_code = 0;
    float quant_scale = 1.0f;
    if (lane == 0 && maximum > 0.0f) {
        __nv_fp8_e4m3 encoded(maximum * (1.0f / 6.0f));
        scale_code = encoded.__x;
        quant_scale = static_cast<float>(encoded);
        if (quant_scale == 0.0f) {
            scale_code = 1;
            __nv_fp8_e4m3 smallest;
            smallest.__x = 1;
            quant_scale = static_cast<float>(smallest);
        }
    }
    scale_code = __shfl_sync(0xffffffff, scale_code, 0);
    quant_scale = __shfl_sync(0xffffffff, quant_scale, 0);
    if (lane == 0)
        scales[nvfp4_scale_offset(row, inner, blocks_per_row)] =
            static_cast<unsigned char>(scale_code);

    if (lane < 8) {
        const float first = input[
            (unsigned long long)row * columns + inner * 16 + lane * 2];
        const float second = input[
            (unsigned long long)row * columns + inner * 16 + lane * 2 + 1];
        const float inverse = scale_code ? 1.0f / quant_scale : 0.0f;
        __nv_fp4x2_e2m1 encoded(make_float2(first * inverse, second * inverse));
        values[(unsigned long long)row * (columns >> 1) + inner * 8 + lane] =
            encoded.__x;
    }
}

// Stack routed gate matrices followed by routed up matrices into one tall
// matrix. A single M=(2*experts*intermediate), N=1 Tensor Core GEMM then
// replaces sixteen independent matvec launches for the usual top-8 route.
extern "C" __global__
void nvfp4_repack_stacked_moe_cublaslt(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    unsigned char* values,
    unsigned char* scales,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int rows = 2 * experts * output_size;
    const int blocks_per_row = input_size >> 6;
    const int block_count = rows * blocks_per_row;
    for (int block = blockIdx.x * blockDim.x + threadIdx.x;
         block < block_count; block += blockDim.x * gridDim.x) {
        const int stacked_row = block / blocks_per_row;
        const int column_block = block - stacked_row * blocks_per_row;
        const int matrix = stacked_row / output_size;
        const int row = stacked_row - matrix * output_size;
        const int expert = matrix < experts ? matrix : matrix - experts;
        const unsigned char* source_matrix = reinterpret_cast<const unsigned char*>(
            matrix < experts ? gate_ptrs[expert] : up_ptrs[expert]);
        const unsigned char* source = source_matrix
            + ((unsigned long long)row * blocks_per_row + column_block) * 36ull;
        unsigned char* destination = values
            + (unsigned long long)stacked_row * (input_size >> 1)
            + (unsigned long long)column_block * 32ull;
        nvfp4_copy_gguf_values_cublaslt(source, destination);
        const int inner_scales = input_size >> 4;
        for (int sub = 0; sub < 4; ++sub) {
            const int inner = column_block * 4 + sub;
            scales[nvfp4_scale_offset(
                stacked_row, inner, inner_scales)] = source[sub];
        }
    }
}

extern "C" __global__
void nvfp4_stacked_moe_swiglu(
    const float* projected,
    float* activated,
    const int output_size,
    const int experts,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = experts * output_size;
    if (index >= elements) return;
    const int expert = index / output_size;
    float gate = projected[index];
    float up = projected[elements + index];
    if (gate_scales) gate *= gate_scales[expert];
    if (up_scales) {
        const float down_scale = down_scales ? down_scales[expert] : 1.0f;
        up *= down_scale != 0.0f
            ? up_scales[expert] / down_scale : up_scales[expert];
    }
    activated[index] =
        (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

// Persistent-cache variant. The grouped gate/up GEMM has
// [experts*2*output_size,16] column-major output. Only column zero is consumed
// by single-token decode.
extern "C" __global__
void nvfp4_persistent_moe_swiglu(
    const float* projected,
    float* activated,
    const int output_size,
    const int experts,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    const int grouped_layout
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = experts * output_size;
    if (index >= elements) return;
    const int expert = index / output_size;
    const int row = index - expert * output_size;
    const float* expert_projected = projected +
        (unsigned long long)expert * 2ull * output_size *
            (grouped_layout ? 1ull : 16ull);
    float gate = expert_projected[row];
    float up = expert_projected[output_size + row];
    if (gate_scales) gate *= gate_scales[expert];
    if (up_scales) {
        const float down_scale = down_scales ? down_scales[expert] : 1.0f;
        up *= down_scale != 0.0f
            ? up_scales[expert] / down_scale : up_scales[expert];
    }
    activated[index] =
        (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void nvfp4_concat_native_gate_up_cublaslt(
    const unsigned long long* native_ptrs,
    unsigned char* values,
    unsigned char* scales,
    const unsigned long long value_bytes,
    const unsigned long long scale_bytes,
    const int experts
) {
    const unsigned long long value_total = value_bytes * experts;
    const unsigned long long scale_total = scale_bytes * experts;
    const unsigned long long count =
        value_total > scale_total ? value_total : scale_total;
    for (unsigned long long index =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += (unsigned long long)blockDim.x * gridDim.x) {
        if (index < value_total) {
            const int expert = (int)(index / value_bytes);
            const unsigned long long offset =
                index - (unsigned long long)expert * value_bytes;
            const unsigned char* source =
                reinterpret_cast<const unsigned char*>(native_ptrs[expert]);
            values[index] = source[offset];
        }
        if (index < scale_total) {
            const int expert = (int)(index / scale_bytes);
            const unsigned long long offset =
                index - (unsigned long long)expert * scale_bytes;
            const unsigned char* source =
                reinterpret_cast<const unsigned char*>(native_ptrs[expert])
                + value_bytes;
            scales[index] = source[offset];
        }
    }
}

extern "C" __global__
void nvfp4_concat_native_down_cublaslt(
    const unsigned long long* native_ptrs,
    unsigned char* values,
    unsigned char* scales,
    const int input_size,
    const int output_size,
    const int experts,
    const unsigned long long down_offset,
    const unsigned long long down_scale_offset
) {
    const int source_value_bytes = input_size >> 1;
    const int combined_value_bytes = experts * source_value_bytes;
    const unsigned long long value_count =
        (unsigned long long)output_size * combined_value_bytes;
    for (unsigned long long index =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < value_count;
         index += (unsigned long long)blockDim.x * gridDim.x) {
        const int row = (int)(index / combined_value_bytes);
        const int combined = (int)(index -
            (unsigned long long)row * combined_value_bytes);
        const int expert = combined / source_value_bytes;
        const int offset = combined - expert * source_value_bytes;
        const unsigned char* source =
            reinterpret_cast<const unsigned char*>(native_ptrs[expert])
            + down_offset;
        values[index] =
            source[(unsigned long long)row * source_value_bytes + offset];
    }
    const int source_inner_scales = input_size >> 4;
    const int combined_inner_scales = experts * source_inner_scales;
    const unsigned long long scale_count =
        (unsigned long long)output_size * combined_inner_scales;
    for (unsigned long long index =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < scale_count;
         index += (unsigned long long)blockDim.x * gridDim.x) {
        const int row = (int)(index / combined_inner_scales);
        const int combined = (int)(index -
            (unsigned long long)row * combined_inner_scales);
        const int expert = combined / source_inner_scales;
        const int inner = combined - expert * source_inner_scales;
        const unsigned char* source =
            reinterpret_cast<const unsigned char*>(native_ptrs[expert])
            + down_scale_offset;
        scales[nvfp4_scale_offset(row, combined, combined_inner_scales)] =
            source[nvfp4_scale_offset(row, inner, source_inner_scales)];
    }
}

extern "C" __global__
void nvfp4_validate_stacked_projection(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* input,
    const float* projected,
    float* stats,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int index = blockIdx.x;
    const int elements = experts * output_size;
    if (index >= 2 * elements) return;
    const int expert = index < elements
        ? index / output_size : (index - elements) / output_size;
    const int row = index % output_size;
    const unsigned char* packed = reinterpret_cast<const unsigned char*>(
        index < elements ? gate_ptrs[expert] : up_ptrs[expert]);
    float reference = 0.0f;
    for (int column = threadIdx.x; column < input_size; column += blockDim.x)
        reference += nvfp4_value(packed, row * input_size + column)
            * input[column];
    reference = block_reduce_sum(reference);
    if (threadIdx.x == 0) {
        const float actual = projected[index];
        atomicMax(reinterpret_cast<unsigned int*>(stats),
                  __float_as_uint(fabsf(reference)));
        atomicMax(reinterpret_cast<unsigned int*>(stats + 1),
                  __float_as_uint(fabsf(actual)));
        atomicMax(reinterpret_cast<unsigned int*>(stats + 2),
                  __float_as_uint(fabsf(reference - actual)));
        if (index == 0) {
            stats[3] = reference;
            stats[4] = actual;
        }
    }
}

// Tensor Core FP4 heuristics require an aligned N dimension. Materialize a
// 16-row activation matrix with the real token in row zero and zero padding in
// the other rows. Column zero of the GEMM result remains the desired matvec.
extern "C" __global__
void nvfp4_quantize_broadcast16_cublaslt(
    const float* input,
    unsigned char* values,
    unsigned char* scales,
    const int columns
) {
    const int blocks_per_row = columns >> 4;
    const int scaled_block = blockIdx.x;
    const int row = scaled_block / blocks_per_row;
    const int inner = scaled_block - row * blocks_per_row;
    if (row >= 16) return;
    const int lane = threadIdx.x & 31;
    const float value = row == 0 && lane < 16
        ? input[inner * 16 + lane] : 0.0f;
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    unsigned int scale_code = 0;
    float quant_scale = 1.0f;
    if (lane == 0 && maximum > 0.0f) {
        __nv_fp8_e4m3 encoded(maximum * (1.0f / 6.0f));
        scale_code = encoded.__x;
        quant_scale = static_cast<float>(encoded);
        if (quant_scale == 0.0f) {
            scale_code = 1;
            __nv_fp8_e4m3 smallest;
            smallest.__x = 1;
            quant_scale = static_cast<float>(smallest);
        }
    }
    scale_code = __shfl_sync(0xffffffff, scale_code, 0);
    quant_scale = __shfl_sync(0xffffffff, quant_scale, 0);
    if (lane == 0)
        scales[nvfp4_scale_offset(row, inner, blocks_per_row)] =
            static_cast<unsigned char>(scale_code);
    if (lane < 8) {
        const float inverse = scale_code ? 1.0f / quant_scale : 0.0f;
        const float first = row == 0 ? input[inner * 16 + lane * 2] : 0.0f;
        const float second =
            row == 0 ? input[inner * 16 + lane * 2 + 1] : 0.0f;
        __nv_fp4x2_e2m1 encoded(
            make_float2(first * inverse, second * inverse));
        values[(unsigned long long)row * (columns >> 1) + inner * 8 + lane] =
            encoded.__x;
    }
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// Concatenate each routed down matrix along K. Multiplying that matrix by the
// concatenated, route-weighted expert activations produces the final weighted
// expert sum directly, without an E-way post-GEMM reduction.
extern "C" __global__
void nvfp4_repack_concat_down_cublaslt(
    const unsigned long long* down_ptrs,
    unsigned char* values,
    unsigned char* scales,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int source_blocks_per_row = input_size >> 6;
    const int blocks_per_row = experts * source_blocks_per_row;
    const int block_count = output_size * blocks_per_row;
    const int combined_input = experts * input_size;
    for (int block = blockIdx.x * blockDim.x + threadIdx.x;
         block < block_count; block += blockDim.x * gridDim.x) {
        const int row = block / blocks_per_row;
        const int combined_block = block - row * blocks_per_row;
        const int expert = combined_block / source_blocks_per_row;
        const int column_block =
            combined_block - expert * source_blocks_per_row;
        const unsigned char* source_matrix =
            reinterpret_cast<const unsigned char*>(down_ptrs[expert]);
        const unsigned char* source = source_matrix
            + ((unsigned long long)row * source_blocks_per_row + column_block)
                * 36ull;
        unsigned char* destination = values
            + (unsigned long long)row * (combined_input >> 1)
            + (unsigned long long)combined_block * 32ull;
        nvfp4_copy_gguf_values_cublaslt(source, destination);
        const int inner_scales = combined_input >> 4;
        for (int sub = 0; sub < 4; ++sub) {
            const int inner = combined_block * 4 + sub;
            scales[nvfp4_scale_offset(row, inner, inner_scales)] = source[sub];
        }
    }
}

extern "C" __global__
void nvfp4_quantize_weighted_moe_cublaslt(
    const float* activated,
    const float* weights,
    const float* down_scales,
    unsigned char* values,
    unsigned char* scales,
    const int input_size,
    const int experts
) {
    // Per-expert down scales are around 3e-5, below UE4M3's minimum subnormal
    // once folded into the activation's per-16 scale. Shift them into the
    // representable range here; the GEMM alpha applies the exact reciprocal.
    constexpr float down_scale_compensation = 32768.0f;
    const int scaled_block = blockIdx.x;
    const int blocks_per_expert = input_size >> 4;
    const int blocks_per_row = experts * blocks_per_expert;
    const int row = scaled_block / blocks_per_row;
    const int row_block = scaled_block - row * blocks_per_row;
    const int expert = row_block / blocks_per_expert;
    const int expert_block = row_block - expert * blocks_per_expert;
    if (row >= 16 || expert >= experts) return;
    const int lane = threadIdx.x & 31;
    const int base = expert * input_size + expert_block * 16;
    const float route_weight =
        weights[expert] * (down_scales ? down_scales[expert] : 1.0f)
        * down_scale_compensation;
    float value = row == 0 && lane < 16
        ? activated[base + lane] * route_weight : 0.0f;
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    unsigned int scale_code = 0;
    float quant_scale = 1.0f;
    if (lane == 0 && maximum > 0.0f) {
        __nv_fp8_e4m3 encoded(maximum * (1.0f / 6.0f));
        scale_code = encoded.__x;
        quant_scale = static_cast<float>(encoded);
        if (quant_scale == 0.0f) {
            scale_code = 1;
            __nv_fp8_e4m3 smallest;
            smallest.__x = 1;
            quant_scale = static_cast<float>(smallest);
        }
    }
    scale_code = __shfl_sync(0xffffffff, scale_code, 0);
    quant_scale = __shfl_sync(0xffffffff, quant_scale, 0);
    const int inner = expert * blocks_per_expert + expert_block;
    if (lane == 0)
        scales[nvfp4_scale_offset(
            row, inner, experts * blocks_per_expert)] =
            static_cast<unsigned char>(scale_code);
    if (lane < 8) {
        const float inverse = scale_code ? 1.0f / quant_scale : 0.0f;
        const float first = row == 0
            ? activated[base + lane * 2] * route_weight : 0.0f;
        const float second = row == 0
            ? activated[base + lane * 2 + 1] * route_weight : 0.0f;
        __nv_fp4x2_e2m1 encoded(make_float2(
            first * inverse, second * inverse));
        values[(unsigned long long)row * (experts * input_size >> 1)
               + expert * (input_size >> 1) + expert_block * 8 + lane] =
                   encoded.__x;
    }
}

extern "C" __global__
void nvfp4_moe_add_first_column(
    const float* matrix,
    float* output,
    const int elements
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] += matrix[index];
}

extern "C" __global__
void nvfp4_validate_down_projection(
    const unsigned long long* down_ptrs,
    const float* activated,
    const float* weights,
    const float* down_scales,
    const float* projected,
    float* stats,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float reference = 0.0f;
    for (int column = threadIdx.x; column < input_size; column += blockDim.x) {
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed =
                reinterpret_cast<const unsigned char*>(down_ptrs[expert]);
            reference += weights[expert] * down_scales[expert]
                * nvfp4_value(packed, row * input_size + column)
                * activated[expert * input_size + column];
        }
    }
    reference = block_reduce_sum(reference);
    if (threadIdx.x == 0) {
        const float actual = projected[row];
        atomicMax(reinterpret_cast<unsigned int*>(stats),
                  __float_as_uint(fabsf(reference)));
        atomicMax(reinterpret_cast<unsigned int*>(stats + 1),
                  __float_as_uint(fabsf(actual)));
        atomicMax(reinterpret_cast<unsigned int*>(stats + 2),
                  __float_as_uint(fabsf(reference - actual)));
        if (row == 0) {
            stats[3] = reference;
            stats[4] = actual;
        }
    }
}

extern "C" __global__
void nvfp4_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size, const float scale
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += nvfp4_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    // `scale` is the tensor's weight_scale_2. It cannot be folded into the E4M3
    // block scales: it runs ~3e-5, far under E4M3's smallest subnormal (2^-9),
    // so folding would flush most blocks to zero. Apply it in f32 instead.
    if (threadIdx.x == 0) output[row] = partial * scale;
}

extern "C" __global__
void nvfp4_swiglu_transposed(
    const unsigned char* gate_packed,
    const unsigned char* up_packed,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const float gate_scale,
    const float up_scale
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += nvfp4_value(gate_packed, absolute) * value;
        up += nvfp4_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    // The scales must be applied before SiLU: the gate path is non-linear, so
    // they cannot be deferred to the down projection the way a plain factor can.
    gate *= gate_scale;
    up *= up_scale;
    if (threadIdx.x == 0)
        activated[row] = (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void nvfp4_matmul_rows(
    const unsigned char* packed,
    const float* input,
    float* output,
    const int input_size,
    const int output_size,
    const int rows,
    const float scale
) {
    const int output_row = blockIdx.x;
    const int token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    const float* vector = input + (long long)token * (long long)input_size;
    float partial = 0.0f;
    for (int column = threadIdx.x; column < input_size; column += blockDim.x)
        partial += nvfp4_value(packed, output_row * input_size + column)
            * vector[column];
    partial = block_reduce_sum(partial);
    // weight_scale_2 in f32, same reason as nvfp4_matvec_transposed.
    if (threadIdx.x == 0)
        output[(long long)token * (long long)output_size + output_row] =
            partial * scale;
}

extern "C" __global__
void nvfp4_grouped_swiglu_tiled(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const int experts,
    const float* gate_scales,
    const float* up_scales
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate_packed = (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up_packed = (const unsigned char*)up_ptrs[expert];
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = lane; input < input_size; input += 32) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += nvfp4_value(gate_packed, absolute) * value;
        up += nvfp4_value(up_packed, absolute) * value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffff, gate, offset);
        up += __shfl_down_sync(0xffffffff, up, offset);
    }
    // Per-expert weight_scale_2, indexed in lockstep with the pointer tables.
    // Applied here in f32 because E4M3 cannot represent it (see nvfp4_value).
    if (gate_scales) gate *= gate_scales[expert];
    if (up_scales) up *= up_scales[expert];
    if (lane == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void nvfp4_grouped_swiglu(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector,
    float* activated,
    const int input_size,
    const int output_size,
    const int experts,
    const float* gate_scales,
    const float* up_scales
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate_packed = (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up_packed = (const unsigned char*)up_ptrs[expert];
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += nvfp4_value(gate_packed, absolute) * value;
        up += nvfp4_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (gate_scales) gate *= gate_scales[expert];
    if (up_scales) up *= up_scales[expert];
    if (threadIdx.x == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void nvfp4_grouped_swiglu_rows(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const int* counts,
    const float* vectors,
    float* activated,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows,
    const float* gate_scales,
    const float* up_scales
) {
    const int row = blockIdx.x;
    const int route = blockIdx.y;
    const int token = route / top_k;
    const int rank = route - token * top_k;
    if (row >= output_size || token >= rows || rank >= counts[token]) return;
    const unsigned char* gate_packed = (const unsigned char*)gate_ptrs[route];
    const unsigned char* up_packed = (const unsigned char*)up_ptrs[route];
    const float* vector = vectors + token * input_size;
    float gate = 0.0f;
    float up = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        const int absolute = row * input_size + input;
        gate += nvfp4_value(gate_packed, absolute) * value;
        up += nvfp4_value(up_packed, absolute) * value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    // Per-route weight_scale_2 (see nvfp4_grouped_swiglu).
    if (gate_scales) gate *= gate_scales[route];
    if (up_scales) up *= up_scales[route];
    if (threadIdx.x == 0)
        activated[route * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void nvfp4_grouped_accumulate_tiled(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = lane; input < input_size; input += 32) {
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed = (const unsigned char*)down_ptrs[expert];
            combined += weights[expert]
                * nvfp4_value(packed, row * input_size + input)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] += partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(extern "C" __global__
void nvfp4_grouped_accumulate(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int input_size,
    const int output_size,
    const int experts
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            const unsigned char* packed = (const unsigned char*)down_ptrs[expert];
            combined += weights[expert]
                * nvfp4_value(packed, row * input_size + input)
                * activated[expert * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] += partial;
}

extern "C" __global__
void nvfp4_grouped_accumulate_rows(
    const unsigned long long* down_ptrs,
    const float* activated,
    float* output,
    const float* weights,
    const int* counts,
    const int input_size,
    const int output_size,
    const int top_k,
    const int rows
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= output_size || token >= rows) return;
    const int base = token * top_k;
    const int count = counts[token];
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        float combined = 0.0f;
        for (int rank = 0; rank < count; ++rank) {
            const int route = base + rank;
            const unsigned char* packed = (const unsigned char*)down_ptrs[route];
            combined += weights[route]
                * nvfp4_value(packed, row * input_size + input)
                * activated[route * input_size + input];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * output_size + row] += partial;
}

// ---- KV cache element codecs (Phase 1: f32, f16) --------------------------
// Overloaded by cache pointer type so the templated kernels below read/write
// the K/V cache in whatever precision it was allocated at.
__device__ __forceinline__ float kv_ld(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float kv_ld(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float kv_ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void kv_st(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void kv_st(__half* p, long long i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void kv_st(__nv_bfloat16* p, long long i, float v) { p[i] = __float2bfloat16(v); }

// Store one (K or V) cache row for the current token; launched once per cache so
// K and V precisions are independent and each type is a single kernel (no KxV
// combinatorics). blockIdx.x = kv_head.
template<typename T>
__device__ void kv_store_impl(
    const float* current, T* cache,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    const int kv_heads, const int head_dim, const int position, const int capacity
) {
    const int head = blockIdx.x;
    if (head >= kv_heads) return;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        kv_st(cache, ((long long)head * capacity + position) * head_dim + d,
              current[head * head_dim + d]);
}
#define KV_STORE(name, T) \
extern "C" __global__ void name(const float* current, T* cache, \
    const int kv_heads, const int head_dim, const int position, const int capacity \
) { kv_store_impl<T>(current, cache, kv_heads, head_dim, position, capacity); }
KV_STORE(kv_store_f32, float)
KV_STORE(kv_store_f16, __half)
KV_STORE(kv_store_bf16, __nv_bfloat16)
#undef KV_STORE

// ---- q8_0 blocked KV codec: 32 elems/block = [f16 scale | 32 int8], 34 bytes.
// A cache row is (head_dim/32) blocks; element e lives in block e/32 at e%32.
__device__ __forceinline__ float kv_ld_q8(const unsigned char* row, int elem) {
    const unsigned char* blk = row + (elem >> 5) * 34;
    const float scale = __half2float(*((const __half*)blk));
    return scale * (float)(((const signed char*)(blk + 2))[elem & 31]);
}
// Quantize and store one (K or V) cache row for the current token; one thread
// per 32-block computes the block absmax -> f16 scale -> symmetric int8.
extern "C" __global__ void kv_store_q8(
    const float* current, unsigned char* cache,
    const int kv_heads, const int head_dim, const int position, const int capacity
) {
    const int head = blockIdx.x;
    if (head >= kv_heads) return;
    const int blocks = head_dim / 32;
    unsigned char* row = cache + ((long long)head * capacity + position) * blocks * 34;
    const float* src = current + head * head_dim;
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) {
        const float* blk = src + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(blk[i]));
        const float scale = amax / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        unsigned char* dst = row + b * 34;
        *((__half*)dst) = __float2half(scale);
        signed char* q = (signed char*)(dst + 2);
        for (int i = 0; i < 32; ++i) {
            int v = __float2int_rn(blk[i] * inv);
            q[i] = (signed char)max(-127, min(127, v));
        }
    }
}

// ---- TurboQuant blocked KV codec (arXiv:2504.19874), 32 elems/block =
// [f16 scale | 32 packed indices], 14 bytes at 3-bit and 18 at 4-bit.
//
// The vector is rotated by a fixed orthogonal R = H*S (sign flip then
// Walsh-Hadamard) before quantizing, which spreads outlier channels evenly so
// one Gaussian codebook fits every head. Because R is shared by every vector
// in the cache it never has to be undone per entry:
//   scores: <q,k> == <Rq,Rk>, so a block rotates its query once into shared
//           memory and dots it against the stored keys directly.
//   values: sum p_i v_i == R^-1(sum p_i (R v_i)), so the weighted sum is
//           accumulated rotated and inverse-rotated once per head.
// This is what keeps the kernel signatures identical to the other cache types.
//
// Codebooks are the Lloyd-Max optimal scalar quantizers for the unit Gaussian,
// which is what the rotated coordinates converge to; they must stay identical
// to native/src/turboquant.h, whose contract test pins them against the paper.
__device__ const float kTurboCb3[8] = {
    -2.15194570f, -1.34390928f, -0.75600528f, -0.24509418f,
    0.24509418f, 0.75600528f, 1.34390928f, 2.15194570f
};
__device__ const float kTurboCb4[16] = {
    -2.73258956f, -2.06901721f, -1.61804637f, -1.25623118f,
    -0.94234045f, -0.65675911f, -0.38804829f, -0.12839503f,
    0.12839503f, 0.38804829f, 0.65675911f, 0.94234045f,
    1.25623118f, 1.61804637f, 2.06901721f, 2.73258956f
};
template<int BITS> __device__ __forceinline__ float turbo_cb(int i);
template<> __device__ __forceinline__ float turbo_cb<3>(int i) { return kTurboCb3[i]; }
template<> __device__ __forceinline__ float turbo_cb<4>(int i) { return kTurboCb4[i]; }
// 2 bytes of f16 scale plus 32 indices of BITS bits.
template<int BITS> __device__ __forceinline__ int turbo_block_bytes() { return 2 + BITS * 4; }

// Deterministic +-1 per coordinate so encode and decode agree without storing
// the rotation. Must match turbo_sign in native/src/turboquant.h.
__device__ __forceinline__ float turbo_sign_d(int index, unsigned stream) {
    unsigned h = (unsigned)index * 0x9e3779b9u + stream * 0x85ebca6bu + 0x165667b1u;
    h ^= h >> 15; h *= 0x2545f491u; h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}
// An index never spans more than two bytes because BITS <= 4.
__device__ __forceinline__ void turbo_pack_d(unsigned char* p, int slot, int bits, unsigned v) {
    const int bit = slot * bits, byte = bit >> 3, sh = bit & 7;
    p[byte] |= (unsigned char)(v << sh);
    if (sh + bits > 8) p[byte + 1] |= (unsigned char)(v >> (8 - sh));
}
__device__ __forceinline__ unsigned turbo_unpack_d(const unsigned char* p, int slot, int bits) {
    const int bit = slot * bits, byte = bit >> 3, sh = bit & 7;
    unsigned v = (unsigned)p[byte] >> sh;
    if (sh + bits > 8) v |= (unsigned)p[byte + 1] << (8 - sh);
    return v & ((1u << bits) - 1u);
}
template<int BITS>
__device__ __forceinline__ float kv_ld_turbo(const unsigned char* row, int elem) {
    const unsigned char* blk = row + (elem >> 5) * turbo_block_bytes<BITS>();
    return __half2float(*((const __half*)blk))
        * turbo_cb<BITS>(turbo_unpack_d(blk + 2, elem & 31, BITS));
}
// Orthonormal in-place Walsh-Hadamard over shared memory, cooperatively across
// the block. Orthonormal means it is its own inverse and preserves norms, which
// is what makes the two identities above exact.
__device__ __forceinline__ void turbo_fwht_shared(float* v, int dim) {
    for (int span = 1; span < dim; span <<= 1) {
        for (int i = threadIdx.x; i < (dim >> 1); i += blockDim.x) {
            const int block = i / span, offset = i % span;
            const int a = block * 2 * span + offset, b = a + span;
            const float lo = v[a], hi = v[b];
            v[a] = lo + hi; v[b] = lo - hi;
        }
        __syncthreads();
    }
    const float norm = rsqrtf((float)dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) v[i] *= norm;
    __syncthreads();
}
// Encode one already-rotated 32-block. The scale is picked from the block RMS
// and then refit by least squares against the chosen codewords, which removes
// the systematic shrinkage a fixed codebook otherwise induces and costs nothing
// at decode. Two passes rather than a 32-entry index array keeps this off local
// memory.
template<int BITS>
__device__ __forceinline__ void turbo_encode_block_d(const float* v, unsigned char* dst) {
    const int levels = 1 << BITS, bytes = turbo_block_bytes<BITS>();
    for (int i = 0; i < bytes; ++i) dst[i] = 0;
    float energy = 0.0f;
    for (int i = 0; i < 32; ++i) energy += v[i] * v[i];
    const float rms = sqrtf(energy / 32.0f);
    if (!(rms > 0.0f)) return;  // all-zero block: zero scale decodes to zeros
    const float inv = 1.0f / rms;
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float x = v[i] * inv;
        int best = 0; float bd = fabsf(x - turbo_cb<BITS>(0));
        for (int l = 1; l < levels; ++l) {
            const float d = fabsf(x - turbo_cb<BITS>(l));
            if (d < bd) { bd = d; best = l; }
        }
        const float c = turbo_cb<BITS>(best);
        num += v[i] * c; den += c * c;
    }
    *((__half*)dst) = __float2half(den > 0.0f ? num / den : rms);
    for (int i = 0; i < 32; ++i) {
        const float x = v[i] * inv;
        int best = 0; float bd = fabsf(x - turbo_cb<BITS>(0));
        for (int l = 1; l < levels; ++l) {
            const float d = fabsf(x - turbo_cb<BITS>(l));
            if (d < bd) { bd = d; best = l; }
        }
        turbo_pack_d(dst + 2, i, BITS, (unsigned)best);
    }
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// head_dim is guarded to a power of two <= 512 wherever a turbo cache type is
// selected, so the scratch below always holds a whole rotated row.
#define TURBO_MAX_DIM 512

// Rotate and store one (K or V) cache row for the current token.
template<int BITS>
__device__ void kv_store_turbo_impl(
    const float* current, unsigned char* cache,
    const int kv_heads, const int head_dim, const int position, const int capacity,
    const unsigned stream
) {
    const int head = blockIdx.x;
    if (head >= kv_heads) return;
    __shared__ float rotated[TURBO_MAX_DIM];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        rotated[d] = current[head * head_dim + d] * turbo_sign_d(d, stream);
    __syncthreads();
    turbo_fwht_shared(rotated, head_dim);
    const int blocks = head_dim / 32, bytes = turbo_block_bytes<BITS>();
    unsigned char* row = cache + ((long long)head * capacity + position) * blocks * bytes;
    for (int b = threadIdx.x; b < blocks; b += blockDim.x)
        turbo_encode_block_d<BITS>(rotated + b * 32, row + b * bytes);
}
// Keys rotate under stream 0 and values under stream 1, matching the harness
// and the CPU reference.
extern "C" __global__ void kv_store_turbo3_k(const float* current, unsigned char* cache,
    const int kv_heads, const int head_dim, const int position, const int capacity
) { kv_store_turbo_impl<3>(current, cache, kv_heads, head_dim, position, capacity, 0u); }
extern "C" __global__ void kv_store_turbo3_v(const float* current, unsigned char* cache,
    const int kv_heads, const int head_dim, const int position, const int capacity
) { kv_store_turbo_impl<3>(current, cache, kv_heads, head_dim, position, capacity, 1u); }
extern "C" __global__ void kv_store_turbo4_k(const float* current, unsigned char* cache,
    const int kv_heads, const int head_dim, const int position, const int capacity
) { kv_store_turbo_impl<4>(current, cache, kv_heads, head_dim, position, capacity, 0u); }
extern "C" __global__ void kv_store_turbo4_v(const float* current, unsigned char* cache,
    const int kv_heads, const int head_dim, const int position, const int capacity
) { kv_store_turbo_impl<4>(current, cache, kv_heads, head_dim, position, capacity, 1u); }

// Every thread in a scores block shares blockIdx.x = head, so the block rotates
// its query into shared memory once and amortizes it over the whole token tile.
// The head guard is uniform across the block; the token guard has to wait until
// after the last __syncthreads or the cooperative transform would deadlock.
template<int BITS, bool RING>
__device__ void kv_scores_turbo_impl(
    const float* query, const unsigned char* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const int first, const float scale
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    __shared__ float rq[TURBO_MAX_DIM];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        rq[d] = query[head * head_dim + d] * turbo_sign_d(d, 0u);
    __syncthreads();
    turbo_fwht_shared(rq, head_dim);
    const int token = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    const int kv_head = head / (heads / kv_heads);
    const int blocks = head_dim / 32, bytes = turbo_block_bytes<BITS>();
    const int slot = RING ? (first + token) % capacity : token;
    const unsigned char* k = keys + ((long long)kv_head * capacity + slot) * blocks * bytes;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; ++d) sum += rq[d] * kv_ld_turbo<BITS>(k, d);
    scores[head * tokens + token] = sum * scale;
}
#define KV_SCORES_TURBO(name, BITS) \
extern "C" __global__ void name(const float* query, const unsigned char* keys, float* scores, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const float scale) { \
    kv_scores_turbo_impl<BITS, false>(query, keys, scores, heads, kv_heads, head_dim, \
        tokens, capacity, 0, scale); \
}
KV_SCORES_TURBO(kv_attention_scores_turbo3, 3)
KV_SCORES_TURBO(kv_attention_scores_turbo4, 4)
#undef KV_SCORES_TURBO
#define KV_SCORES_TURBO_RING(name, BITS) \
extern "C" __global__ void name(const float* query, const unsigned char* keys, float* scores, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first, const float scale) { \
    kv_scores_turbo_impl<BITS, true>(query, keys, scores, heads, kv_heads, head_dim, \
        tokens, capacity, first, scale); \
}
KV_SCORES_TURBO_RING(kv_attention_scores_turbo3_ring, 3)
KV_SCORES_TURBO_RING(kv_attention_scores_turbo4_ring, 4)
#undef KV_SCORES_TURBO_RING

// One block per head, so the weighted sum is accumulated in the rotated domain
// in shared memory and inverse-rotated once at the end: R^-1 = R^T = S*H, i.e.
// Walsh-Hadamard first, then the sign flip.
template<int BITS, bool RING>
__device__ void kv_values_turbo_impl(
    float* scores, const unsigned char* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const int first
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int kv_head = head / (heads / kv_heads);
    const int blocks = head_dim / 32, bytes = turbo_block_bytes<BITS>();
    float* head_scores = scores + head * tokens;
    float local_maximum = -3.402823466e+38F;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x)
        local_maximum = fmaxf(local_maximum, head_scores[token]);
    const float reduced_maximum = block_reduce_max(local_maximum);
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = reduced_maximum;
    __syncthreads();
    float local_denominator = 0.0f;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        const float weight = expf(head_scores[token] - maximum);
        head_scores[token] = weight;
        local_denominator += weight;
    }
    const float reduced_denominator = block_reduce_sum(local_denominator);
    __shared__ float inverse_denominator;
    if (threadIdx.x == 0) inverse_denominator = 1.0f / reduced_denominator;
    __syncthreads();
    for (int token = threadIdx.x; token < tokens; token += blockDim.x)
        head_scores[token] *= inverse_denominator;
    __syncthreads();

    // The weighted sum is the hot loop: every thread walks the whole cache, so
    // anything left inside it costs O(tokens). Three things are hoisted out:
    //
    //  * the codebook moves to shared memory, off the global path;
    //  * a thread owns one fixed d for the whole loop, so its block offset,
    //    byte offset and bit shift are loop-invariant and computed once;
    //  * the ring wrap becomes two contiguous runs instead of a % per token,
    //    which also lets the row pointer advance by a fixed stride.
    //
    // The index read is branchless. `spill` is 1 only for the slots whose bits
    // straddle a byte, so when it is 0 the second load re-reads the same byte
    // and the mask ignores the duplicated high half. Both reads stay inside the
    // block: at BITS=3 the widest access is byte 12..13 of 14, at BITS=4 it is
    // byte 17 of 18. That keeps the whole warp on one path instead of splitting
    // it, which is what made this loop slow.
    __shared__ float codebook[1 << BITS];
    if (threadIdx.x < (1 << BITS)) codebook[threadIdx.x] = turbo_cb<BITS>(threadIdx.x);
    __syncthreads();
    __shared__ float accumulated[TURBO_MAX_DIM];
    const int row_bytes = blocks * bytes;
    const unsigned char* cache_base =
        values + (long long)kv_head * capacity * row_bytes;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const int block_offset = (d >> 5) * bytes;
        const int bit = (d & 31) * BITS;
        const int byte_offset = block_offset + 2 + (bit >> 3);
        const int shift = bit & 7;
        const int spill = (shift + BITS > 8) ? 1 : 0;
        constexpr unsigned mask = (1u << BITS) - 1u;
        float result = 0.0f;
        int token = 0, slot = RING ? first : 0;
        while (token < tokens) {
            const int run = RING ? min(tokens - token, capacity - slot) : tokens - token;
            const unsigned char* row = cache_base + (long long)slot * row_bytes;
            for (int i = 0; i < run; ++i, row += row_bytes) {
                const float scale = __half2float(*(const __half*)(row + block_offset));
                const unsigned low = row[byte_offset], high = row[byte_offset + spill];
                result += head_scores[token + i] * scale
                    * codebook[((low | (high << 8)) >> shift) & mask];
            }
            token += run;
            slot = 0;
        }
        accumulated[d] = result;
    }
    __syncthreads();
    turbo_fwht_shared(accumulated, head_dim);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        output[head * head_dim + d] = accumulated[d] * turbo_sign_d(d, 1u);
}
#define KV_VALUES_TURBO(name, BITS) \
extern "C" __global__ void name(float* scores, const unsigned char* values, float* output, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity) { \
    kv_values_turbo_impl<BITS, false>(scores, values, output, heads, kv_heads, head_dim, \
        tokens, capacity, 0); \
}
KV_VALUES_TURBO(kv_attention_values_turbo3, 3)
KV_VALUES_TURBO(kv_attention_values_turbo4, 4)
#undef KV_VALUES_TURBO
// Expand a turbo cache window into contiguous f16 so the cuBLAS attention path
// can run on it. The rotation is deliberately NOT undone: keys stay rotated and
// are matched against a rotated query, and values stay rotated with the single
// inverse applied to the attention output afterwards. Undoing it per entry here
// would cost a Walsh-Hadamard per token and defeat the point.
//
// The ring is unwrapped on the way out, so the caller passes capacity == tokens
// and first == 0 to cuBLAS and no wrap logic is needed downstream.
// Output layout is [kv_head][tokens][head_dim], one warp per token.
template<int BITS>
__device__ void kv_dequant_turbo_impl(
    const unsigned char* cache, __half* out,
    const int kv_heads, const int head_dim, const int tokens,
    const int capacity, const int first
) {
    constexpr int tokens_per_block = 8;
    const int kv_head = blockIdx.x;
    if (kv_head >= kv_heads) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int token = blockIdx.y * tokens_per_block + warp;
    if (token >= tokens) return;
    const int row_bytes = (head_dim / 32) * turbo_block_bytes<BITS>();
    int slot = first + token;
    if (slot >= capacity) slot -= capacity;
    const unsigned char* row = cache + ((long long)kv_head * capacity + slot) * row_bytes;
    __half* destination = out + ((long long)kv_head * tokens + token) * head_dim;
    for (int d = lane; d < head_dim; d += 32)
        destination[d] = __float2half(kv_ld_turbo<BITS>(row, d));
}
#define KV_DEQUANT_TURBO(name, BITS) \
extern "C" __global__ void name(const unsigned char* cache, __half* out, \
    const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first) { \
    kv_dequant_turbo_impl<BITS>(cache, out, kv_heads, head_dim, tokens, capacity, first); \
}
KV_DEQUANT_TURBO(kv_dequant_turbo3_f16, 3)
KV_DEQUANT_TURBO(kv_dequant_turbo4_f16, 4)
#undef KV_DEQUANT_TURBO

// Rotate (stream 0, for queries) or inverse-rotate (stream 1, for the attention
// output) `rows` vectors of `dim` floats in place, one block per row. Forward is
// R = H*S, inverse is R^-1 = R^T = S*H, so the two differ only in the order of
// the sign flip and the transform.
extern "C" __global__ void turbo_rotate_rows(
    float* data, const int rows, const int dim, const int stream_id
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float scratch[TURBO_MAX_DIM];
    float* base = data + (long long)row * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        scratch[i] = base[i] * turbo_sign_d(i, (unsigned)stream_id);
    __syncthreads();
    turbo_fwht_shared(scratch, dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) base[i] = scratch[i];
}
extern "C" __global__ void turbo_unrotate_rows(
    float* data, const int rows, const int dim, const int stream_id
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float scratch[TURBO_MAX_DIM];
    float* base = data + (long long)row * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) scratch[i] = base[i];
    __syncthreads();
    turbo_fwht_shared(scratch, dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        base[i] = scratch[i] * turbo_sign_d(i, (unsigned)stream_id);
}

#define KV_VALUES_TURBO_RING(name, BITS) \
extern "C" __global__ void name(float* scores, const unsigned char* values, float* output, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first) { \
    kv_values_turbo_impl<BITS, true>(scores, values, output, heads, kv_heads, head_dim, \
        tokens, capacity, first); \
}
KV_VALUES_TURBO_RING(kv_attention_values_turbo3_ring, 3)
KV_VALUES_TURBO_RING(kv_attention_values_turbo4_ring, 4)
#undef KV_VALUES_TURBO_RING

template<typename KT>
__device__ void kv_scores_impl(
    const float* query, const KT* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const float scale
) {
    const int head = blockIdx.x;
    const int token = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || token >= tokens) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const float* q = query + head * head_dim;
    const KT* k = keys + ((long long)kv_head * capacity + token) * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) score += q[d] * kv_ld(k, d);
    scores[head * tokens + token] = score * scale;
}
extern "C" __global__ void kv_attention_scores(
    const float* query, const float* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const float scale
) { kv_scores_impl<float>(query, keys, scores, heads, kv_heads, head_dim, tokens, capacity, scale); }
extern "C" __global__ void kv_attention_scores_f16(
    const float* query, const __half* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const float scale
) { kv_scores_impl<__half>(query, keys, scores, heads, kv_heads, head_dim, tokens, capacity, scale); }
extern "C" __global__ void kv_attention_scores_bf16(
    const float* query, const __nv_bfloat16* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const float scale
) { kv_scores_impl<__nv_bfloat16>(query, keys, scores, heads, kv_heads, head_dim, tokens, capacity, scale); }
extern "C" __global__ void kv_attention_scores_q8(
    const float* query, const unsigned char* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const float scale
) {
    const int head = blockIdx.x;
    const int token = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || token >= tokens) return;
    const int group = heads / kv_heads, kv_head = head / group, blocks = head_dim / 32;
    const float* q = query + head * head_dim;
    const unsigned char* k = keys + ((long long)kv_head * capacity + token) * blocks * 34;
    float s = 0.0f;
    for (int d = 0; d < head_dim; ++d) s += q[d] * kv_ld_q8(k, d);
    scores[head * tokens + token] = s * scale;
}

// Logical token 0 can live anywhere in a compact circular SWA cache.  Keeping
// this separate from the linear kernels preserves their ABI and fast path for
// global-attention layers.
template<typename KT>
__device__ void kv_scores_ring_impl(
    const float* query, const KT* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const int first, const float scale
) {
    const int head = blockIdx.x;
    const int token = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || token >= tokens) return;
    const int group = heads / kv_heads, kv_head = head / group;
    const int slot = (first + token) % capacity;
    const float* q = query + head * head_dim;
    const KT* k = keys + ((long long)kv_head * capacity + slot) * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) score += q[d] * kv_ld(k, d);
    scores[head * tokens + token] = score * scale;
}
#define KV_SCORES_RING(name, T) \
extern "C" __global__ void name(const float* query, const T* keys, float* scores, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first, const float scale) { \
    kv_scores_ring_impl<T>(query, keys, scores, heads, kv_heads, head_dim, tokens, capacity, first, scale); \
}
KV_SCORES_RING(kv_attention_scores_ring, float)
KV_SCORES_RING(kv_attention_scores_f16_ring, __half)
KV_SCORES_RING(kv_attention_scores_bf16_ring, __nv_bfloat16)
#undef KV_SCORES_RING
extern "C" __global__ void kv_attention_scores_q8_ring(
    const float* query, const unsigned char* keys, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const int first, const float scale
) {
    const int head=blockIdx.x, token=blockIdx.y*blockDim.x+threadIdx.x;
    if(head>=heads||token>=tokens)return;
    const int kv_head=head/(heads/kv_heads), blocks=head_dim/32;
    const int slot=(first+token)%capacity;
    const float* q=query+head*head_dim;
    const unsigned char* k=keys+((long long)kv_head*capacity+slot)*blocks*34;
    float score=0.0f;
    for(int d=0;d<head_dim;++d)score+=q[d]*kv_ld_q8(k,d);
    scores[head*tokens+token]=score*scale;
}

template<typename VT>
__device__ void kv_values_impl(
    float* scores, const VT* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    float* head_scores = scores + head * tokens;
    float local_maximum = -3.402823466e+38F;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, head_scores[token]);
    }
    const float reduced_maximum = block_reduce_max(local_maximum);
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = reduced_maximum;
    __syncthreads();
    float local_denominator = 0.0f;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        const float weight = expf(head_scores[token] - maximum);
        head_scores[token] = weight;
        local_denominator += weight;
    }
    const float reduced_denominator = block_reduce_sum(local_denominator);
    __shared__ float inverse_denominator;
    if (threadIdx.x == 0) inverse_denominator = 1.0f / reduced_denominator;
    __syncthreads();
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        head_scores[token] *= inverse_denominator;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float result = 0.0f;
        for (int token = 0; token < tokens; ++token) {
            result += head_scores[token] *
                kv_ld(values, (long long)(kv_head * capacity + token) * head_dim + d);
        }
        output[head * head_dim + d] = result;
    }
}
extern "C" __global__ void kv_attention_values(
    float* scores, const float* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity
) { kv_values_impl<float>(scores, values, output, heads, kv_heads, head_dim, tokens, capacity); }
extern "C" __global__ void kv_attention_values_f16(
    float* scores, const __half* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity
) { kv_values_impl<__half>(scores, values, output, heads, kv_heads, head_dim, tokens, capacity); }
extern "C" __global__ void kv_attention_values_bf16(
    float* scores, const __nv_bfloat16* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity
) { kv_values_impl<__nv_bfloat16>(scores, values, output, heads, kv_heads, head_dim, tokens, capacity); }
extern "C" __global__ void kv_attention_values_q8(
    float* scores, const unsigned char* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int group = heads / kv_heads, kv_head = head / group, blocks = head_dim / 32;
    float* head_scores = scores + head * tokens;
    float local_maximum = -3.402823466e+38F;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x)
        local_maximum = fmaxf(local_maximum, head_scores[token]);
    const float reduced_maximum = block_reduce_max(local_maximum);
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = reduced_maximum;
    __syncthreads();
    float local_denominator = 0.0f;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x)
        local_denominator += expf(head_scores[token] - maximum);
    const float reduced_denominator = block_reduce_sum(local_denominator);
    __shared__ float inverse_denominator;
    if (threadIdx.x == 0) inverse_denominator = 1.0f / reduced_denominator;
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float result = 0.0f;
        for (int token = 0; token < tokens; ++token) {
            const unsigned char* vrow = values + ((long long)kv_head * capacity + token) * blocks * 34;
            result += expf(head_scores[token] - maximum) * kv_ld_q8(vrow, d);
        }
        output[head * head_dim + d] = result * inverse_denominator;
    }
}

template<typename VT>
__device__ void kv_values_ring_impl(
    float* scores, const VT* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const int first
) {
    const int head=blockIdx.x;
    if(head>=heads)return;
    const int kv_head=head/(heads/kv_heads);
    float* head_scores=scores+head*tokens;
    float local_maximum=-3.402823466e+38F;
    for(int token=threadIdx.x;token<tokens;token+=blockDim.x)local_maximum=fmaxf(local_maximum,head_scores[token]);
    const float reduced_maximum=block_reduce_max(local_maximum);
    __shared__ float maximum;
    if(threadIdx.x==0)maximum=reduced_maximum;
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    __syncthreads();
    float local_denominator=0.0f;
    for(int token=threadIdx.x;token<tokens;token+=blockDim.x){
        const float weight=expf(head_scores[token]-maximum);
        head_scores[token]=weight;
        local_denominator+=weight;
    }
    const float reduced_denominator=block_reduce_sum(local_denominator);
    __shared__ float inverse_denominator;
    if(threadIdx.x==0)inverse_denominator=1.0f/reduced_denominator;
    __syncthreads();
    for(int token=threadIdx.x;token<tokens;token+=blockDim.x)
        head_scores[token]*=inverse_denominator;
    __syncthreads();
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float result=0.0f;
        for(int token=0;token<tokens;++token){
            const int slot=(first+token)%capacity;
            result+=head_scores[token]*kv_ld(values,(long long)(kv_head*capacity+slot)*head_dim+d);
        }
        output[head*head_dim+d]=result;
    }
}
#define KV_VALUES_RING(name, T) \
extern "C" __global__ void name(float* scores, const T* values, float* output, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first) { \
    kv_values_ring_impl<T>(scores, values, output, heads, kv_heads, head_dim, tokens, capacity, first); \
}
KV_VALUES_RING(kv_attention_values_ring, float)
KV_VALUES_RING(kv_attention_values_f16_ring, __half)
KV_VALUES_RING(kv_attention_values_bf16_ring, __nv_bfloat16)
#undef KV_VALUES_RING
extern "C" __global__ void kv_attention_values_q8_ring(
    float* scores, const unsigned char* values, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const int first
) {
    const int head=blockIdx.x;
    if(head>=heads)return;
    const int kv_head=head/(heads/kv_heads), blocks=head_dim/32;
    float* head_scores=scores+head*tokens;
    float local_maximum=-3.402823466e+38F;
    for(int token=threadIdx.x;token<tokens;token+=blockDim.x)local_maximum=fmaxf(local_maximum,head_scores[token]);
    const float reduced_maximum=block_reduce_max(local_maximum);
    __shared__ float maximum;
    if(threadIdx.x==0)maximum=reduced_maximum;
    __syncthreads();
    float local_denominator=0.0f;
    for(int token=threadIdx.x;token<tokens;token+=blockDim.x){
        const float weight=expf(head_scores[token]-maximum);
        head_scores[token]=weight;
        local_denominator+=weight;
    }
    const float reduced_denominator=block_reduce_sum(local_denominator);
    __shared__ float inverse_denominator;
    if(threadIdx.x==0)inverse_denominator=1.0f/reduced_denominator;
    __syncthreads();
    for(int token=threadIdx.x;token<tokens;token+=blockDim.x)
        head_scores[token]*=inverse_denominator;
    __syncthreads();
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float result=0.0f;
        for(int token=0;token<tokens;++token){
            const int slot=(first+token)%capacity;
            const unsigned char* row=values+((long long)kv_head*capacity+slot)*blocks*34;
            result+=head_scores[token]*kv_ld_q8(row,d);
        }
        output[head*head_dim+d]=result;
    }
}

// Tiled single-token attention. Every block computes an online-softmax state
// for one [head, 1024-token] tile. A second small kernel merges those states.
// Compared with the split score/value path this keeps long-context occupancy
// (many tiles are independent), avoids the [heads, tokens] score round trip,
// and reads each value only while its score is live.
//
// Qwen currently uses head_dim=128.  Other shapes retain the generic split
// kernels, keeping this fast path's local and shared storage fixed-size.
template<typename T>
__device__ __forceinline__ float kv_fused_load(
    const T* cache, long long row, int dimension, int head_dim
) {
    return kv_ld(cache, row * head_dim + dimension);
}

__device__ __forceinline__ float kv_fused_load(
    const unsigned char* cache, long long row, int dimension, int head_dim
) {
    return kv_ld_q8(
        cache + row * (head_dim / 32) * 34,
        dimension
    );
}

extern "C" __global__ void qwen_attention_query_f16(
    const float* input, __half* output, const int elements
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += blockDim.x * gridDim.x)
        output[index] = __float2half(input[index]);
}

// The cuBLAS attention path materializes one FP16 score/probability matrix per
// GQA group. Keeping it FP16 lets both QK^T and PV use tensor cores and halves
// score traffic. One block normalizes one query head.
extern "C" __global__ void kv_attention_softmax_f16(
    __half* scores, const int heads, const int tokens
) {
    const int head = blockIdx.x;
    if (head >= heads || tokens <= 0) return;
    __shared__ float reduction[256];
    __half* row = scores + (long long)head * tokens;
    float maximum = -3.402823466e+38F;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x)
        maximum = fmaxf(maximum, __half2float(row[token]));
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] =
                fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    float denominator = 0.0f;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        const float probability = __expf(__half2float(row[token]) - maximum);
        row[token] = __float2half(probability);
        denominator += probability;
    }
    reduction[threadIdx.x] = denominator;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = 1.0f / reduction[0];
    for (int token = threadIdx.x; token < tokens; token += blockDim.x)
        row[token] = __float2half(__half2float(row[token]) * inverse);
}

// Tensor-core prefill attention packs a small query-row tile so every KV head
// owns one contiguous [head_dim, tile_rows * GQA_group] matrix.  That layout
// lets one strided-batched GEMM cover all heads instead of launching a GEMM per
// query row.
extern "C" __global__ void qwen_attention_prefill_pack_f16(
    const float* queries, __half* packed, const int tile_start,
    const int tile_rows, const int heads, const int kv_heads,
    const int head_dim
) {
    const int elements = tile_rows * heads * head_dim;
    const int group = heads / kv_heads;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x) {
        const int dimension = index % head_dim;
        const int head = (index / head_dim) % heads;
        const int row = index / (heads * head_dim);
        const int kv_head = head / group;
        const int group_head = head - kv_head * group;
        const long long column = (long long)row * group + group_head;
        packed[((long long)kv_head * tile_rows * group + column) * head_dim
               + dimension] =
            __float2half(queries[((long long)tile_start + row) * heads * head_dim
                                + (long long)head * head_dim + dimension]);
    }
}

// Scores are the column-major cuBLAS result
// [kv_head][token][tile_row, group_head].  Normalize only the causally visible
// prefix for each query and zero the masked tail before the PV GEMM.
extern "C" __global__ void kv_attention_prefill_softmax_f16(
    const float* scores, __half* probabilities,
    const int tile_start, const int tile_rows,
    const int heads, const int kv_heads, const int tokens,
    const int base_position
) {
    const int head = blockIdx.x;
    const int row_index = blockIdx.y;
    if (head >= heads || row_index >= tile_rows || tokens <= 0) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const int group_head = head - kv_head * group;
    const int column = row_index * group + group_head;
    const float* row = scores
        + (long long)kv_head * tokens * tile_rows * group
        + (long long)column * tokens;
    __half* output = probabilities
        + (long long)kv_head * tokens * tile_rows * group
        + (long long)column * tokens;
    const int visible = min(tokens, base_position + tile_start + row_index + 1);
    __shared__ float reduction[256];
    float maximum = -3.402823466e+38F;
    for (int token = threadIdx.x; token < visible; token += blockDim.x)
        maximum = fmaxf(maximum, row[token]);
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] =
                fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    float denominator = 0.0f;
    for (int token = threadIdx.x; token < visible; token += blockDim.x) {
        const float probability = __expf(row[token] - maximum);
        output[token] = __float2half(probability);
        denominator += probability;
    }
    for (int token = visible + threadIdx.x; token < tokens;
         token += blockDim.x)
        output[token] = __float2half(0.0f);
    reduction[threadIdx.x] = denominator;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = 1.0f / reduction[0];
    for (int token = threadIdx.x; token < visible; token += blockDim.x)
        output[token] = __float2half(__half2float(output[token]) * inverse);
}

// Convert the column-major PV result back to row/head order, optionally
// applying Qwen's attention gate while the value is already in a register.
//
// GATE=false exists for the turbo cache path: its values are stored rotated, so
// the result has to be inverse-rotated before any elementwise nonlinearity, and
// the gate is a sigmoid. Fusing the gate here would apply it one step too early
// and silently corrupt the output.
template<bool GATE>
__device__ void qwen_attention_prefill_unpack_impl(
    const float* packed, const float* gates, float* output,
    const int tile_start, const int tile_rows, const int heads,
    const int kv_heads, const int head_dim
) {
    const int elements = tile_rows * heads * head_dim;
    const int group = heads / kv_heads;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x) {
        const int dimension = index % head_dim;
        const int head = (index / head_dim) % heads;
        const int row = index / (heads * head_dim);
        const int kv_head = head / group;
        const int group_head = head - kv_head * group;
        const long long column = (long long)row * group + group_head;
        const float value =
            packed[((long long)kv_head * tile_rows * group + column) * head_dim
                   + dimension];
        const long long destination =
            ((long long)tile_start + row) * heads * head_dim
            + (long long)head * head_dim + dimension;
        if (GATE) {
            const float gate = fminf(80.0f, fmaxf(-80.0f, gates[destination]));
            output[destination] = value / (1.0f + expf(-gate));
        } else {
            output[destination] = value;
        }
    }
}
extern "C" __global__ void qwen_attention_prefill_unpack_gate(
    const float* packed, const float* gates, float* output,
    const int tile_start, const int tile_rows, const int heads,
    const int kv_heads, const int head_dim
) {
    qwen_attention_prefill_unpack_impl<true>(
        packed, gates, output, tile_start, tile_rows, heads, kv_heads, head_dim);
}
extern "C" __global__ void qwen_attention_prefill_unpack(
    const float* packed, const float* gates, float* output,
    const int tile_start, const int tile_rows, const int heads,
    const int kv_heads, const int head_dim
) {
    qwen_attention_prefill_unpack_impl<false>(
        packed, gates, output, tile_start, tile_rows, heads, kv_heads, head_dim);
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(template<typename KT, typename VT>
__device__ void kv_attention_fused_tiles_impl(
    const float* query,
    const KT* keys,
    const VT* values,
    float* partial,
    const int heads,
    const int kv_heads,
    const int head_dim,
    const int tokens,
    const int capacity,
    const int first,
    const float scale
) {
    constexpr int warp_count = 8;
    constexpr int maximum_head_dim = 128;
    constexpr int tokens_per_tile = 1024;
    const int head = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (head >= heads || head_dim > maximum_head_dim || (head_dim & 31) != 0)
        return;

    const int tile_begin = tile * tokens_per_tile;
    const int tile_end = min(tokens, tile_begin + tokens_per_tile);
    const int kv_head = head / (heads / kv_heads);
    const float* q = query + head * head_dim;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    for (int token = tile_begin + warp;
         token < tile_end;
         token += warp_count) {
        int slot = first + token;
        if (slot >= capacity) slot -= capacity;
        const long long row = (long long)kv_head * capacity + slot;
        float dot = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int dimension = lane + part * 32;
            if (dimension < head_dim)
                dot += q[dimension]
                    * kv_fused_load(keys, row, dimension, head_dim);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        const float score =
            __shfl_sync(0xffffffff, dot, 0) * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float old_scale = maximum == -3.402823466e+38F
            ? 0.0f : __expf(maximum - next_maximum);
        const float token_scale = __expf(score - next_maximum);
        denominator = denominator * old_scale + token_scale;

        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int dimension = lane + part * 32;
            if (dimension < head_dim) {
                accumulator[part] = accumulator[part] * old_scale
                    + token_scale
                    * kv_fused_load(values, row, dimension, head_dim);
            }
        }
        maximum = next_maximum;
    }

    __shared__ float warp_maximum[warp_count];
    __shared__ float warp_denominator[warp_count];
    __shared__ float warp_scale[warp_count];
    __shared__ float merged_maximum;
    __shared__ float merged_denominator;
    __shared__ float partial_output[warp_count][maximum_head_dim];
    if (lane == 0) {
        warp_maximum[warp] = maximum;
        warp_denominator[warp] = denominator;
    }
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int dimension = lane + part * 32;
        if (dimension < head_dim)
            partial_output[warp][dimension] = accumulator[part];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float tile_maximum = warp_maximum[0];
        #pragma unroll
        for (int index = 1; index < warp_count; ++index)
            tile_maximum = fmaxf(tile_maximum, warp_maximum[index]);
        float tile_denominator = 0.0f;
        #pragma unroll
        for (int index = 0; index < warp_count; ++index) {
            const float factor = warp_denominator[index] == 0.0f
                ? 0.0f : __expf(warp_maximum[index] - tile_maximum);
            warp_scale[index] = factor;
            tile_denominator += warp_denominator[index] * factor;
        }
        merged_maximum = tile_maximum;
        merged_denominator = tile_denominator;
    }
    __syncthreads();

    const int tile_count = (tokens + tokens_per_tile - 1) / tokens_per_tile;
    float* record = partial
        + ((long long)head * tile_count + tile) * (maximum_head_dim + 2);
    if (threadIdx.x == 0) {
        record[0] = merged_maximum;
        record[1] = merged_denominator;
    }
    for (int dimension = threadIdx.x;
         dimension < head_dim;
         dimension += blockDim.x) {
        float result = 0.0f;
        #pragma unroll
        for (int index = 0; index < warp_count; ++index)
            result += partial_output[index][dimension] * warp_scale[index];
        record[dimension + 2] = result;
    }
}

// Decode one output row per warp (eight rows per block). The older low-bit
// kernels devoted all 256 threads to one row and paid two block barriers for
// every projection row. Keeping the same eight rows resident but reducing
// within a warp improves occupancy and removes shared-memory reductions.
#define COLIBRI_LOWBIT_MATVEC_WARP(name, value_at) \
extern "C" __global__ void name( \
    const unsigned char* packed, const float* vector, float* output, \
    const int input_size, const int output_size \
) { \
    const int lane = threadIdx.x & 31; \
    const int warp = threadIdx.x >> 5; \
    const int row = blockIdx.x * 8 + warp; \
    if (row >= output_size) return; \
    float partial = 0.0f; \
    for (int input = lane; input < input_size; input += 32) \
        partial += value_at(packed, row * input_size + input) * vector[input]; \
    for (int offset = 16; offset > 0; offset >>= 1) \
        partial += __shfl_down_sync(0xffffffffu, partial, offset); \
    if (lane == 0) output[row] = partial; \
}
COLIBRI_LOWBIT_MATVEC_WARP(q2k_matvec_transposed_warp, q2k_value)
COLIBRI_LOWBIT_MATVEC_WARP(q3k_matvec_transposed_warp, q3k_value)
COLIBRI_LOWBIT_MATVEC_WARP(q4k_matvec_transposed_warp, q4k_value)
COLIBRI_LOWBIT_MATVEC_WARP(q5k_matvec_transposed_warp, q5k_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq2xxs_matvec_transposed_warp, iq2xxs_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq3xxs_matvec_transposed_warp, iq3xxs_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq2s_matvec_transposed_warp, iq2s_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq3s_matvec_transposed_warp, iq3s_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq2xs_matvec_transposed_warp, iq2xs_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq4xs_matvec_transposed_warp, iq4xs_value)
#undef COLIBRI_LOWBIT_MATVEC_WARP

// Reuse each decoded weight across four prompt rows. This small register tile
// cuts quant decoding and global weight traffic by up to 4x without requiring
// a full dequantized copy of a dense matrix.
#define COLIBRI_LOWBIT_MATMUL_ROWS(name, value_at) \
extern "C" __global__ void name( \
    const unsigned char* packed, const float* vectors, float* output, \
    const int input_size, const int output_size, const int rows \
) { \
    constexpr int tile_rows = 4; \
    const int output_row = blockIdx.x; \
    const int token_base = blockIdx.y * tile_rows; \
    if (output_row >= output_size || token_base >= rows) return; \
    float partial[tile_rows] = {0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) { \
        const float weight = value_at( \
            packed, output_row * input_size + input); \
        for (int tile = 0; tile < tile_rows; ++tile) { \
            const int token = token_base + tile; \
            if (token < rows) \
                partial[tile] += weight * vectors[token * input_size + input]; \
        } \
    } \
    for (int tile = 0; tile < tile_rows; ++tile) { \
        partial[tile] = block_reduce_sum(partial[tile]); \
        if (threadIdx.x == 0 && token_base + tile < rows) \
            output[(token_base + tile) * output_size + output_row] = partial[tile]; \
    } \
}
COLIBRI_LOWBIT_MATMUL_ROWS(q2k_matmul_rows, q2k_value)
COLIBRI_LOWBIT_MATMUL_ROWS(q3k_matmul_rows, q3k_value)
COLIBRI_LOWBIT_MATMUL_ROWS(q4k_matmul_rows, q4k_value)
COLIBRI_LOWBIT_MATMUL_ROWS(q5k_matmul_rows, q5k_value)
COLIBRI_LOWBIT_MATMUL_ROWS(q6k_matmul_rows, q6k_value)
COLIBRI_LOWBIT_MATMUL_ROWS(iq2xxs_matmul_rows, iq2xxs_value)
COLIBRI_LOWBIT_MATMUL_ROWS(iq3xxs_matmul_rows, iq3xxs_value)
COLIBRI_LOWBIT_MATMUL_ROWS(iq2s_matmul_rows, iq2s_value)
COLIBRI_LOWBIT_MATMUL_ROWS(iq3s_matmul_rows, iq3s_value)
COLIBRI_LOWBIT_MATMUL_ROWS(iq2xs_matmul_rows, iq2xs_value)
COLIBRI_LOWBIT_MATMUL_ROWS(iq4xs_matmul_rows, iq4xs_value)
#undef COLIBRI_LOWBIT_MATMUL_ROWS

#define KV_ATTENTION_FUSED_TILES(name, KT, VT) \
extern "C" __global__ void name( \
    const float* query, const KT* keys, const VT* values, float* partial, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first, const float scale \
) { \
    kv_attention_fused_tiles_impl<KT, VT>( \
        query, keys, values, partial, heads, kv_heads, head_dim, tokens, \
        capacity, first, scale \
    ); \
}
KV_ATTENTION_FUSED_TILES(
    kv_attention_fused_f16_tiles, __half, __half
)
KV_ATTENTION_FUSED_TILES(
    kv_attention_fused_bf16_tiles, __nv_bfloat16, __nv_bfloat16
)
KV_ATTENTION_FUSED_TILES(
    kv_attention_fused_q8_tiles, unsigned char, unsigned char
)
#undef KV_ATTENTION_FUSED_TILES

extern "C" __global__ void kv_attention_fused_merge(
    const float* partial,
    float* output,
    const int heads,
    const int head_dim,
    const int tile_count
) {
    constexpr int maximum_head_dim = 128;
    constexpr int maximum_tiles = 512;
    const int head = blockIdx.x;
    if (head >= heads || head_dim > maximum_head_dim ||
        tile_count <= 0 || tile_count > maximum_tiles)
        return;
    const int stride = maximum_head_dim + 2;
    const float* head_partial =
        partial + (long long)head * tile_count * stride;
    __shared__ float tile_scale[maximum_tiles];
    __shared__ float inverse_denominator;
    if (threadIdx.x == 0) {
        float maximum = head_partial[0];
        for (int tile = 1; tile < tile_count; ++tile)
            maximum = fmaxf(maximum, head_partial[tile * stride]);
        float denominator = 0.0f;
        for (int tile = 0; tile < tile_count; ++tile) {
            const float factor =
                __expf(head_partial[tile * stride] - maximum);
            tile_scale[tile] = factor;
            denominator += head_partial[tile * stride + 1] * factor;
        }
        inverse_denominator = 1.0f / denominator;
    }
    __syncthreads();
    for (int dimension = threadIdx.x;
         dimension < head_dim;
         dimension += blockDim.x) {
        float result = 0.0f;
        for (int tile = 0; tile < tile_count; ++tile) {
            result += head_partial[tile * stride + dimension + 2]
                * tile_scale[tile];
        }
        output[head * head_dim + dimension] =
            result * inverse_denominator;
    }
}

// Portable compatibility entry point used by the small C ABI parity tests.
// The native Qwen decode path uses the parallel score/value kernels above.
extern "C" __global__
void kv_attention(
    const float* query,
    const float* keys,
    const float* values,
    float* output,
    const int heads,
    const int kv_heads,
    const int head_dim,
    const int tokens,
    const int capacity,
    const float scale
) {
    const int head = blockIdx.x;
    if (head >= heads || threadIdx.x != 0) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const float* q = query + head * head_dim;
    float maximum = -3.402823466e+38F;
    for (int token = 0; token < tokens; ++token) {
        float score = 0.0f;
        const float* k = keys + (kv_head * capacity + token) * head_dim;
        for (int d = 0; d < head_dim; ++d) score += q[d] * k[d];
        maximum = fmaxf(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (int d = 0; d < head_dim; ++d) output[head * head_dim + d] = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        float score = 0.0f;
        const float* k = keys + (kv_head * capacity + token) * head_dim;
        for (int d = 0; d < head_dim; ++d) score += q[d] * k[d];
        const float weight = expf(score * scale - maximum);
        denominator += weight;
        const float* v = values + (kv_head * capacity + token) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            output[head * head_dim + d] += weight * v[d];
        }
    }
    for (int d = 0; d < head_dim; ++d) {
        output[head * head_dim + d] /= denominator;
    }
}

template<typename KT, typename VT>
__device__ void kv_append_impl(
    const float* current_keys, const float* current_values,
    KT* cache_keys, VT* cache_values,
    const int kv_heads, const int head_dim, const int position, const int capacity
) {
    const int head = blockIdx.x;
    if (head >= kv_heads) return;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const long long slot = ((long long)head * capacity + position) * head_dim + d;
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        kv_st(cache_keys, slot, current_keys[head * head_dim + d]);
        kv_st(cache_values, slot, current_values[head * head_dim + d]);
    }
}
#define KV_APPEND(name, KT, VT) \
extern "C" __global__ void name( \
    const float* current_keys, const float* current_values, \
    KT* cache_keys, VT* cache_values, \
    const int kv_heads, const int head_dim, const int position, const int capacity \
) { kv_append_impl<KT, VT>(current_keys, current_values, cache_keys, cache_values, kv_heads, head_dim, position, capacity); }
KV_APPEND(kv_append, float, float) // combined f32 append, used only by the MTP path
#undef KV_APPEND
)COLIBRI_CUDA";
}
