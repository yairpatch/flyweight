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
    // Sized for the launch rather than a fixed eight warps, so single-block
    // reductions (rms_norm over the whole hidden vector) can use a full 1024
    // threads instead of leaving 3/4 of the SM idle.
    const int warps = (int)((blockDim.x + 31) >> 5);
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = (int)threadIdx.x < warps ? warp_sums[lane] : 0.0f;
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Same quantization as above, one row per blockIdx.y. Prefill quantizes every
// row of the batch with identical arguments apart from the base pointers, and
// at 64 rows the launches cost more than the work; scale_stride is in halves
// because the host lays the scale rows out on a float-sized stride.
extern "C" __global__
void quantize_q8_blocks_rows(
    const float* input,
    signed char* output,
    __half* scales,
    const int elements,
    const int scale_stride
) {
    const int lane = threadIdx.x;
    const long long row = blockIdx.y;
    const int index = blockIdx.x * 32 + lane;
    const float* row_input = input + row * elements;
    float value = index < elements ? row_input[index] : 0.0f;
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    if (lane == 0) scales[row * scale_stride + blockIdx.x] = __float2half(scale);
    if (index < elements) {
        const int quantized = max(-127, min(127, __float2int_rn(value / scale)));
        output[row * elements + index] = (signed char)quantized;
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
    const float* input,
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(void sampling_block_topk_logits(
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// Importance-matrix capture: the column-wise sum of squared activations, which
// is what says how much each input channel of a weight matrix actually
// contributes. One thread per column, looping the batch, so the accumulator is
// written once per column and needs no atomics.
extern "C" __global__
void qwen_imatrix_accumulate(
    const float* input,
    float* sums,
    const int width,
    const int rows
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    float total = 0.0f;
    for (int row = 0; row < rows; ++row) {
        const float value = input[(long long)row * width + column];
        total += value * value;
    }
    sums[column] += total;
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

// Expert grouping for the prefill GEMM path: pack the rows routed to one
// expert into a contiguous tile so the batched matmul kernels see an
// ordinary matrix, and fold the result back weighted per route. Rows within
// one launch are distinct tokens (a router picks each expert at most once
// per token), so the scatter needs no atomics.
extern "C" __global__
void qwen_gather_rows(
    const float* source,
    const int* indices,
    float* destination,
    const int width,
    const int count
) {
    const int row = blockIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count || column >= width) return;
    destination[(long long)row * width + column] =
        source[(long long)indices[row] * width + column];
}

extern "C" __global__
void qwen_scatter_add_rows(
    const float* source,
    const int* indices,
    const float* weights,
    float* destination,
    const int width,
    const int count
) {
    const int row = blockIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count || column >= width) return;
    destination[(long long)indices[row] * width + column] +=
        weights[row] * source[(long long)row * width + column];
}

extern "C" __global__
void q4_batched_matvec(
    const unsigned long long* packed_addresses,
    const unsigned long long* scale_addresses,
    const float* vector,
    float* output,
    const int rows,
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    if (threadIdx.x == 0) output[expert * rows + row] = partial;
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
    float g=0.0f,u=0.0f;
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// Whole-layer-pinned variants: every expert of the layer is resident in the
// cache at a fixed slot (layer_base + expert * slot_bytes), so the routed
// expert indices never leave the device. `selected` and `routing_weights` are
// route_topk's outputs read in rank order, exactly as the host pointer-table
// path consumed them; the per-expert f32 scale rides at the tail of each
// bundle, and the weight product keeps the host path's multiplication order
// so pinned layers stay bit-identical to the hybrid path they replace.
extern "C" __global__ void gemma_q4_0_pinned_geglu(
    const unsigned long long layer_base,const unsigned long long slot_bytes,
    const int* selected,const float* input,float* output,
    const int input_size,const int intermediate,const int expert_count
) {
    const int rank=blockIdx.y;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(rank>=expert_count||row>=intermediate)return;
    const unsigned char* gate_up=(const unsigned char*)(
        layer_base+(unsigned long long)selected[rank]*slot_bytes);
    float g=0.0f,u=0.0f;
    for(int i=lane;i<input_size;i+=32){
        const float x=input[i];
        g+=ggml_q4_0_load(gate_up,(long long)row*input_size+i)*x;
        u+=ggml_q4_0_load(gate_up,(long long)(row+intermediate)*input_size+i)*x;
    }
    for(int offset=16;offset;offset>>=1){g+=__shfl_down_sync(0xffffffff,g,offset);u+=__shfl_down_sync(0xffffffff,u,offset);}
    if(lane==0)output[(long long)rank*intermediate+row]=gemma_gelu(g)*u;
}

extern "C" __global__ void gemma_q4_0_pinned_accumulate(
    const unsigned long long layer_base,const unsigned long long slot_bytes,
    const unsigned long long down_offset,const unsigned long long scale_offset,
    const int* selected,const float* activated,float* output,
    const float* routing_weights,const int intermediate,const int hidden,
    const int expert_count
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(row>=hidden)return;
    float total=0.0f;
    for(int rank=0;rank<expert_count;++rank){
        const unsigned long long base=
            layer_base+(unsigned long long)selected[rank]*slot_bytes;
        const unsigned char* down=(const unsigned char*)(base+down_offset);
        const float scale=*(const float*)(base+scale_offset);
        const float* vector=activated+(long long)rank*intermediate;
        float sum=0.0f;
        for(int i=lane;i<intermediate;i+=32)
            sum+=ggml_q4_0_load(down,(long long)row*intermediate+i)*vector[i];
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane==0)total+=(routing_weights[rank]*scale)*sum;
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

// Rows-batched twins of the per-row Gemma kernels above: one launch covers a
// whole prefill chunk (grid.y or grid.x picks the row), with explicit element
// strides because the rows workspace interleaves hidden-wide and scratch-wide
// regions. Each mirrors its single-row original's arithmetic exactly, so a
// chunk stays bit-identical to the one-token path.
extern "C" __global__ void gemma_rms_rows(
    const float* inputs,const float* weights,float* outputs,
    const int columns,const int rows,
    const long long input_stride,const long long output_stride,
    const float epsilon
) {
    const int row=blockIdx.x;if(row>=rows)return;
    const float* input=inputs+row*input_stride;
    float* output=outputs+row*output_stride;
    float sum=0.0f;
    for(int i=threadIdx.x;i<columns;i+=blockDim.x)sum+=input[i]*input[i];
    sum=block_reduce_sum(sum);
    __shared__ float inverse;if(threadIdx.x==0)inverse=rsqrtf(sum/columns+epsilon);__syncthreads();
    for(int i=threadIdx.x;i<columns;i+=blockDim.x)output[i]=input[i]*inverse*weights[i];
}

extern "C" __global__ void gemma_scaled_add_rows(
    float* targets,const float* sources,const int count,const int rows,
    const long long target_stride,const long long source_stride
) {
    const int row=blockIdx.y;if(row>=rows)return;
    float* target=targets+row*target_stride;
    const float* source=sources+row*source_stride;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        target[i]+=source[i];
}

extern "C" __global__ void gemma_q4_0_matvec_rows(
    const unsigned char* packed,const float* vectors,float* outputs,
    const int input_size,const int output_size,const int rows,
    const long long input_stride,const long long output_stride
) {
    const int token=blockIdx.y;if(token>=rows)return;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(row>=output_size)return;
    const float* vector=vectors+token*input_stride;
    float sum=0.0f;
    for(int input=lane;input<input_size;input+=32)
        sum+=ggml_q4_0_load(packed,(long long)row*input_size+input)*vector[input];
    for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0)outputs[token*output_stride+row]=sum;
}

extern "C" __global__ void gemma_q4_0_geglu_rows(
    const unsigned char* gate,const unsigned char* up,
    const float* inputs,float* outputs,
    const int input_size,const int intermediate,const int rows,
    const long long input_stride,const long long output_stride
) {
    const int token=blockIdx.y;if(token>=rows)return;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(row>=intermediate)return;
    const float* input=inputs+token*input_stride;
    float g=0.0f,u=0.0f;
    for(int i=lane;i<input_size;i+=32){
        const float x=input[i];
        g+=ggml_q4_0_load(gate,(long long)row*input_size+i)*x;
        u+=ggml_q4_0_load(up,(long long)row*input_size+i)*x;
    }
    for(int offset=16;offset;offset>>=1){g+=__shfl_down_sync(0xffffffff,g,offset);u+=__shfl_down_sync(0xffffffff,u,offset);}
    if(lane==0)outputs[token*output_stride+row]=gemma_gelu(g)*u;
}

extern "C" __global__ void gemma_head_norm_rope_rows(
    const float* projected,const float* weights,float* outputs,
    const int heads,const int head_dim,const int rotary_dim,
    const int base_position,const float theta,const float epsilon,
    const float* freq_factors,const int rows,
    const long long input_stride,const long long output_stride
) {
    const int token=blockIdx.y;if(token>=rows)return;
    const int head=blockIdx.x;if(head>=heads)return;
    const int position=base_position+token;
    const float* src=projected+token*input_stride+head*head_dim;
    float* dst=outputs+token*output_stride+head*head_dim;
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

extern "C" __global__ void gemma_head_rms_rows(
    const float* inputs,float* outputs,const int heads,const int head_dim,
    const float epsilon,const int rows,
    const long long input_stride,const long long output_stride
) {
    const int token=blockIdx.y;if(token>=rows)return;
    const int head=blockIdx.x;if(head>=heads)return;
    const float* src=inputs+token*input_stride+head*head_dim;
    float* dst=outputs+token*output_stride+head*head_dim;
    float sum=0.0f;for(int d=threadIdx.x;d<head_dim;d+=blockDim.x)sum+=src[d]*src[d];
    sum=block_reduce_sum(sum);__shared__ float inv;if(threadIdx.x==0)inv=rsqrtf(sum/head_dim+epsilon);__syncthreads();
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x)dst[d]=src[d]*inv;
}

extern "C" __global__ void gemma_router_input_rows(
    const float* inputs,const float* scale,float* outputs,
    const int elements,const float epsilon,const int rows,
    const long long input_stride,const long long output_stride
) {
    const int row=blockIdx.x;if(row>=rows)return;
    const float* input=inputs+row*input_stride;
    float* output=outputs+row*output_stride;
    float sum=0.0f;for(int i=threadIdx.x;i<elements;i+=blockDim.x)sum+=input[i]*input[i];
    sum=block_reduce_sum(sum);__shared__ float factor;if(threadIdx.x==0)factor=rsqrtf(sum/elements+epsilon)*rsqrtf((float)elements);__syncthreads();
    for(int i=threadIdx.x;i<elements;i+=blockDim.x)output[i]=input[i]*scale[i]*factor;
}

extern "C" __global__ void gemma_f32_matvec_rows(
    const float* matrix,const float* vectors,float* outputs,
    const int input_size,const int output_size,const int rows,
    const long long input_stride,const long long output_stride
) {
    const int token=blockIdx.y;if(token>=rows)return;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*8+warp;
    if(row>=output_size)return;
    const float* vector=vectors+token*input_stride;
    const float* row_matrix=matrix+(long long)row*(long long)input_size;
    // Mirrors qwen_f32_matvec_warp's float4 fast path, including its exact
    // accumulation order, so a chunked router matvec matches the one-token
    // launch bit for bit.
    float partial=0.0f;
    if((input_size&3)==0
       &&(((unsigned long long)row_matrix)&15ull)==0ull
       &&(((unsigned long long)vector)&15ull)==0ull){
        const int steps=input_size>>2;
        for(int step=lane;step<steps;step+=32){
            const float4 w=((const float4*)row_matrix)[step];
            const float4 v=((const float4*)vector)[step];
            partial+=w.x*v.x+w.y*v.y+w.z*v.z+w.w*v.w;
        }
    }else{
        for(int column=lane;column<input_size;column+=32)
            partial+=row_matrix[column]*vector[column];
    }
    for(int offset=16;offset>0;offset>>=1)
        partial+=__shfl_down_sync(0xffffffff,partial,offset);
    if(lane==0)outputs[token*output_stride+row]=partial;
}

extern "C" __global__ void gemma_scale_vector_rows(
    float* values,const float* scale,const int elements,const int rows,
    const long long stride
) {
    const int row=blockIdx.y;if(row>=rows)return;
    float* target=values+row*stride;
    const float factor=scale[0];
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<elements;i+=blockDim.x*gridDim.x)
        target[i]*=factor;
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Batched-GEMM prefill for the Gemma dense projections. Activations quantize
// once per distinct input into a tight [row * elements] int8 layout (same
// max/127 half-scale convention as quantize_q8_blocks_rows, plus an input
// stride because the rows workspace interleaves widths), and the DP4A tile
// kernel reads each decoded weight block once for eight tokens instead of
// re-reading the whole matrix per row the way the matvec twins do.
extern "C" __global__ void gemma_quantize_q8_rows(
    const float* input,const long long input_stride,
    signed char* output,__half* scales,const int elements,const int rows
) {
    const int lane=threadIdx.x;
    const long long row=blockIdx.y;
    if(row>=rows)return;
    const int index=blockIdx.x*32+lane;
    const float* row_input=input+row*input_stride;
    float value=index<elements?row_input[index]:0.0f;
    float maximum=fabsf(value);
    for(int offset=16;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
    maximum=__shfl_sync(0xffffffff,maximum,0);
    const float scale=maximum>0.0f?maximum/127.0f:1.0f;
    if(lane==0)scales[row*(elements/32)+blockIdx.x]=__float2half(scale);
    if(index<elements){
        const int quantized=max(-127,min(127,__float2int_rn(value/scale)));
        output[row*elements+index]=(signed char)quantized;
    }
}

extern "C" __global__ void gemma_q4_0_q8_mmq_rows(
    const unsigned char* packed,const signed char* vectors,
    const __half* vector_scales,float* outputs,
    const int input_size,const int output_size,const int rows,
    const long long output_stride
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int output_base=blockIdx.x*32;
    const int token_base=blockIdx.y*8;
    if(output_base>=output_size||token_base>=rows)return;
    __shared__ int w_vals[32][8];
    __shared__ float w_scale[32];
    __shared__ int a_vals[8][8];
    __shared__ float a_scale[8];
    const int kblocks=input_size/32;
    float total=0.0f;
    for(int block=0;block<kblocks;++block){
        __syncthreads();
        {
            // 256 threads stage 32 weight rows x 8 ints: thread (r, i) decodes
            // int i of row r. Values 0..15 are the low nibbles of the 16
            // payload bytes, 16..31 the high, so ints 0..3 mask and 4..7
            // shift the same four packed words.
            const int r=threadIdx.x>>3,i=threadIdx.x&7;
            const int source_row=min(output_base+r,output_size-1);
            const unsigned char* base=packed+
                ((long long)source_row*kblocks+block)*18;
            int word;
            memcpy(&word,base+2+(i&3)*4,4);
            const int nibbles=(i<4?word:(word>>4))&0x0f0f0f0f;
            w_vals[r][i]=__vsub4(nibbles,0x08080808);
            if(i==0)w_scale[r]=__half2float(*((const __half*)base));
        }
        if(threadIdx.x<64){
            const int t=threadIdx.x>>3,i=threadIdx.x&7;
            const int source_token=min(token_base+t,rows-1);
            const signed char* values=vectors+
                (long long)source_token*input_size+block*32;
            int word;
            memcpy(&word,values+i*4,4);
            a_vals[t][i]=word;
            if(i==0)a_scale[t]=__half2float(
                vector_scales[(long long)source_token*kblocks+block]);
        }
        __syncthreads();
        int dot=0;
        #pragma unroll
        for(int i=0;i<8;++i)dot=__dp4a(w_vals[lane][i],a_vals[warp][i],dot);
        total+=w_scale[lane]*a_scale[warp]*(float)dot;
    }
    const int output_row=output_base+lane;
    const int token=token_base+warp;
    if(output_row<output_size&&token<rows)
        outputs[(long long)token*output_stride+output_row]=total;
}

// The fused chunk-attention template caps at head_dim 256 (q and acc live in
// registers), but Gemma 4's five global layers run head_dim 512 -- and they
// are exactly the layers whose per-row scores/values passes scale with the
// full context (measured 23.6 ms/token of a 24.6 ms total at 13k). Same
// online-softmax, two query rows per warp instead of four so the register
// budget stays near 100.
extern "C" __global__ void gemma_kv_prefill_wide_f16(
    const float* queries,const __half* keys,const __half* values,
    float* output,const int heads,const int kv_heads,
    const int head_dim,const int base_position,const int rows,
    const int capacity,const float scale
) {
    const int head=blockIdx.x;if(head>=heads)return;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5;
    const int tile=(blockIdx.y*warps+warp)*2;
    if(tile>=rows)return;
    const int kv_head=head/(heads/kv_heads);
    const int dims=head_dim/32;
    const int count=min(2,rows-tile);
    float q[2][16],acc[2][16],m[2],l[2];
    for(int i=0;i<2;++i){
        m[i]=-3.402823466e+38F;l[i]=0.0f;
        for(int d=0;d<16;++d){
            acc[i][d]=0.0f;
            q[i][d]=(i<count&&d<dims)
                ?queries[(long long)(tile+i)*heads*head_dim
                         +head*head_dim+lane+32*d]
                :0.0f;
        }
    }
    const int last=base_position+tile+count-1;
    for(int position=0;position<=last;++position){
        float k[16],v[16];
        const __half* key_row=keys+((long long)kv_head*capacity+position)*head_dim;
        const __half* value_row=values+((long long)kv_head*capacity+position)*head_dim;
        for(int d=0;d<16;++d){
            k[d]=d<dims?__half2float(key_row[lane+32*d]):0.0f;
            v[d]=d<dims?__half2float(value_row[lane+32*d]):0.0f;
        }
        for(int i=0;i<count;++i){
            if(position>base_position+tile+i)continue;
            float partial=0.0f;
            for(int d=0;d<16;++d)partial+=q[i][d]*k[d];
            for(int offset=16;offset>0;offset>>=1)
                partial+=__shfl_xor_sync(0xffffffff,partial,offset);
            const float score=partial*scale;
            const float peak=fmaxf(m[i],score);
            const float rescale=expf(m[i]-peak);
            const float weight=expf(score-peak);
            l[i]=l[i]*rescale+weight;
            for(int d=0;d<16;++d)acc[i][d]=acc[i][d]*rescale+weight*v[d];
            m[i]=peak;
        }
    }
    for(int i=0;i<count;++i){
        const float inverse=1.0f/l[i];
        for(int d=0;d<dims;++d)
            output[(long long)(tile+i)*heads*head_dim+head*head_dim+lane+32*d]
                =acc[i][d]*inverse;
    }
}

extern "C" __global__ void gemma_geglu_combine_rows(
    const float* gates,const float* ups,float* outputs,
    const int intermediate,const int rows,
    const long long gate_stride,const long long up_stride,
    const long long output_stride
) {
    const int row=blockIdx.y;if(row>=rows)return;
    const float* gate=gates+row*gate_stride;
    const float* up=ups+row*up_stride;
    float* output=outputs+row*output_stride;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<intermediate;i+=blockDim.x*gridDim.x)
        output[i]=gemma_gelu(gate[i])*up[i];
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__ void gemma_q4_0_lm_argmax(
    const unsigned char* packed,const float* input,unsigned long long* winner,
    const int hidden,const int vocabulary
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,token=blockIdx.x*8+warp;
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    if(token>=vocabulary)return;float sum=0.0f;
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

// Two decode-shaped rows share each Q8 weight load.  MTP verification almost
// always uses two candidates; launching the single-row kernel twice otherwise
// doubles the dominant dense-weight traffic without gaining GEMM occupancy.
extern "C" __global__
void q8_matvec_transposed_pair(
    const unsigned char* packed,
    const float* vectors,
    float* outputs,
    const int input_size,
    const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float first = 0.0f;
    float second = 0.0f;
    for (int input = lane; input < input_size; input += 32) {
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float weight = (float)(*((const signed char*)(
            packed + block * 34 + 2 + within))) * __half2float(
                *((const __half*)(packed + block * 34)));
        first += weight * vectors[input];
        second += weight * vectors[input_size + input];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        first += __shfl_down_sync(0xffffffff, first, offset);
        second += __shfl_down_sync(0xffffffff, second, offset);
    }
    if (lane == 0) {
        outputs[row] = first;
        outputs[output_size + row] = second;
    }
}

extern "C" __global__
void q8_matvec_transposed_triple(
    const unsigned char* packed,
    const float* vectors,
    float* outputs,
    const int input_size,
    const int output_size
) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int row=blockIdx.x*8+warp;
    if(row>=output_size)return;
    float first=0.0f,second=0.0f,third=0.0f;
    for(int input=lane;input<input_size;input+=32){
        const int absolute=row*input_size+input;
        const int block=absolute>>5,within=absolute&31;
        const float weight=(float)(*((const signed char*)(
            packed+block*34+2+within)))*__half2float(
                *((const __half*)(packed+block*34)));
        first+=weight*vectors[input];
        second+=weight*vectors[input_size+input];
        third+=weight*vectors[2*input_size+input];
    }
    for(int offset=16;offset;offset>>=1){
        first+=__shfl_down_sync(0xffffffff,first,offset);
        second+=__shfl_down_sync(0xffffffff,second,offset);
        third+=__shfl_down_sync(0xffffffff,third,offset);
    }
    if(lane==0){outputs[row]=first;outputs[output_size+row]=second;
        outputs[2*output_size+row]=third;}
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
    const int quant = ((offset < 32) ? (low & 15) : (low >> 4)) + 16 * bit;
    int scale, minimum;
    q5k_scale_min(scales, group * 2 + sub, &scale, &minimum);
    return d * (float)scale * (float)quant - dmin * (float)minimum;
}

// One block per output row: each thread walks a stride of 32-value groups and
// the block reduces. Right for the wide-but-short projections in a layer.
#define COLIBRI_Q8_MATVEC(name, group_fn, stride)                              \
extern "C" __global__ void name(                                               \
    const unsigned char* packed, const signed char* vector,                    \
    const __half* vector_scales, float* output,                                \
    const int input_size, const int output_size                                \
) {                                                                            \
    const int row = blockIdx.x;                                                \
    if (row >= output_size) return;                                            \
    const int blocks_per_row = input_size >> 8;                                \
    const int groups_per_row = blocks_per_row << 3;                            \
    const unsigned char* row_data =                                            \
        packed + (long long)row * blocks_per_row * stride;                     \
    float partial = 0.0f;                                                      \
    for (int g = threadIdx.x; g < groups_per_row; g += blockDim.x)             \
        partial += group_fn(row_data, vector, vector_scales, g);               \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    for (int offset = 16; offset > 0; offset >>= 1)                            \
        partial += __shfl_down_sync(0xffffffffu, partial, offset);             \
    __shared__ float warp_sums[4];                                             \
    if (lane == 0) warp_sums[warp] = partial;                                  \
    __syncthreads();                                                           \
    if (warp == 0) {                                                           \
        partial = lane < 4 ? warp_sums[lane] : 0.0f;                           \
        for (int offset = 16; offset > 0; offset >>= 1)                        \
            partial += __shfl_down_sync(0xffffffffu, partial, offset);         \
        if (lane == 0) output[row] = partial;                                  \
    }                                                                          \
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Prefill batches token rows against one weight matrix, and the kernel above is
// launched once per row -- which re-reads the whole matrix per row. On a dense
// model that is the entire prefill cost: N tokens do N full passes over the
// weights, which is why prefill ran no faster per token than decode.
//
// This decodes each weight group once into eight words of four int8 and dots it
// against every row in the batch, so the weight traffic is paid once per launch
// instead of once per row. Splitting decode from dot is the point: calling
// group_fn once per row would drop the traffic just the same, but for the
// codebook formats decode is a dependent table load plus sign expansion per
// eight weights -- more work than the two DP4A that consume it -- so repeating
// it per row just trades a memory bound for an ALU one.
//
// decode_fn fills words[8] and the two half-block scales; formats with a single
// scale per 32-value group pass it as both. COLIBRI_Q8_ROWS caps the batch: the
// accumulators are registers and the block reduction needs a shared slot per
// warp per row. The host chunks a wider batch into this many rows at a time;
// kQ8RowBatch in native/src/v2_mtp_verifier.inc must match.
#define COLIBRI_Q8_ROWS 8

#define COLIBRI_Q8_MATVEC_ROWS(name, decode_fn, stride)                        \
extern "C" __global__ void name(                                               \
    const unsigned char* packed, const signed char* vectors,                   \
    const __half* vector_scales, float* outputs,                               \
    const int input_size, const int output_size,                               \
    const int rows, const int scale_stride                                     \
) {                                                                            \
    const int row = blockIdx.x;                                                \
    if (row >= output_size) return;                                            \
    const int blocks_per_row = input_size >> 8;                                \
    const int groups_per_row = blocks_per_row << 3;                            \
    const unsigned char* row_data =                                            \
        packed + (long long)row * blocks_per_row * stride;                     \
    float partial[COLIBRI_Q8_ROWS];                                            \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) partial[r] = 0.0f;               \
    for (int g = threadIdx.x; g < groups_per_row; g += blockDim.x) {           \
        int words[8];                                                          \
        float scale_low = 0.0f, scale_high = 0.0f;                             \
        decode_fn(row_data, g, words, &scale_low, &scale_high);                \
        /* Unrolled over the compile-time cap and predicated on the runtime   */\
        /* row count: a runtime bound would index partial dynamically and     */\
        /* spill the accumulators to local memory.                            */\
        _Pragma("unroll")                                                      \
        for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) {                            \
            if (r >= rows) continue;                                           \
            const int4* activation_vectors = (const int4*)(                    \
                vectors + (long long)r * input_size + (long long)g * 32);      \
            const int4 activation_low = activation_vectors[0];                 \
            const int4 activation_high = activation_vectors[1];                \
            const int acts[8] = {                                              \
                activation_low.x, activation_low.y,                            \
                activation_low.z, activation_low.w,                            \
                activation_high.x, activation_high.y,                          \
                activation_high.z, activation_high.w};                         \
            int dot_low = 0, dot_high = 0;                                     \
            _Pragma("unroll")                                                  \
            for (int step = 0; step < 4; ++step) {                             \
                int dot = 0;                                                   \
                dot = __dp4a(words[step * 2], acts[step * 2], dot);            \
                dot = __dp4a(words[step * 2 + 1], acts[step * 2 + 1], dot);    \
                if (step < 2) dot_low += dot; else dot_high += dot;            \
            }                                                                  \
            partial[r] +=                                                      \
                ((float)dot_low * scale_low + (float)dot_high * scale_high)    \
                * __half2float(                                                \
                    vector_scales[(long long)r * scale_stride + g]);           \
        }                                                                      \
    }                                                                          \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    __shared__ float warp_sums[4][COLIBRI_Q8_ROWS];                            \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) {                                \
        float value = partial[r];                                              \
        for (int offset = 16; offset > 0; offset >>= 1)                        \
            value += __shfl_down_sync(0xffffffffu, value, offset);             \
        if (lane == 0) warp_sums[warp][r] = value;                             \
    }                                                                          \
    __syncthreads();                                                           \
    if (warp != 0) return;                                                     \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) {                                \
        float value = lane < 4 ? warp_sums[lane][r] : 0.0f;                    \
        for (int offset = 16; offset > 0; offset >>= 1)                        \
            value += __shfl_down_sync(0xffffffffu, value, offset);             \
        if (lane == 0 && r < rows)                                             \
            outputs[(long long)r * output_size + row] = value;                 \
    }                                                                          \
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// The same kernel for the asymmetric K-quants, which reconstruct as
// `d*scale*q - dmin*min` rather than `d*scale*q`.
//
// That per-sub-block minimum is a constant subtracted from every weight in the
// group, so it leaves the dot product and becomes `dmin*min * sum(activations)`
// -- one extra dp4a against 0x01010101 per quad, which is exactly what the
// single-row kernels already do. Two of them here, because Q2_K's sub-block is
// 16 values and its two halves carry different minima; Q4_K and Q5_K set both
// halves the same.
//
// Kept separate from COLIBRI_Q8_MATVEC_ROWS rather than folded into it with
// zero offsets: the symmetric formats would pay the activation sums for
// nothing, and this loop's whole purpose is that its inner work is small.
#define COLIBRI_Q8_MATVEC_ROWS_MIN(name, decode_fn, stride)                    \
extern "C" __global__ void name(                                               \
    const unsigned char* packed, const signed char* vectors,                   \
    const __half* vector_scales, float* outputs,                               \
    const int input_size, const int output_size,                               \
    const int rows, const int scale_stride                                     \
) {                                                                            \
    const int row = blockIdx.x;                                                \
    if (row >= output_size) return;                                            \
    const int blocks_per_row = input_size >> 8;                                \
    const int groups_per_row = blocks_per_row << 3;                            \
    const unsigned char* row_data =                                            \
        packed + (long long)row * blocks_per_row * stride;                     \
    float partial[COLIBRI_Q8_ROWS];                                            \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) partial[r] = 0.0f;               \
    for (int g = threadIdx.x; g < groups_per_row; g += blockDim.x) {           \
        int words[8];                                                          \
        float scale_low = 0.0f, scale_high = 0.0f;                             \
        float offset_low = 0.0f, offset_high = 0.0f;                           \
        decode_fn(row_data, g, words, &scale_low, &scale_high,                 \
                  &offset_low, &offset_high);                                  \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) {                            \
            if (r >= rows) continue;                                           \
            const int4* activation_vectors = (const int4*)(                    \
                vectors + (long long)r * input_size + (long long)g * 32);      \
            const int4 activation_low = activation_vectors[0];                 \
            const int4 activation_high = activation_vectors[1];                \
            const int acts[8] = {                                              \
                activation_low.x, activation_low.y,                            \
                activation_low.z, activation_low.w,                            \
                activation_high.x, activation_high.y,                          \
                activation_high.z, activation_high.w};                         \
            int dot_low = 0, dot_high = 0, sum_low = 0, sum_high = 0;          \
            _Pragma("unroll")                                                  \
            for (int step = 0; step < 4; ++step) {                             \
                int dot = 0, sum = 0;                                          \
                dot = __dp4a(words[step * 2], acts[step * 2], dot);            \
                dot = __dp4a(words[step * 2 + 1], acts[step * 2 + 1], dot);    \
                sum = __dp4a(0x01010101, acts[step * 2], sum);                 \
                sum = __dp4a(0x01010101, acts[step * 2 + 1], sum);             \
                if (step < 2) { dot_low += dot; sum_low += sum; }              \
                else { dot_high += dot; sum_high += sum; }                     \
            }                                                                  \
            partial[r] +=                                                      \
                ((float)dot_low * scale_low - (float)sum_low * offset_low      \
                 + (float)dot_high * scale_high                                \
                 - (float)sum_high * offset_high)                              \
                * __half2float(                                                \
                    vector_scales[(long long)r * scale_stride + g]);           \
        }                                                                      \
    }                                                                          \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    __shared__ float warp_sums[4][COLIBRI_Q8_ROWS];                            \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) {                                \
        float value = partial[r];                                              \
        for (int offset = 16; offset > 0; offset >>= 1)                        \
            value += __shfl_down_sync(0xffffffffu, value, offset);             \
        if (lane == 0) warp_sums[warp][r] = value;                             \
    }                                                                          \
    __syncthreads();                                                           \
    if (warp != 0) return;                                                     \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < COLIBRI_Q8_ROWS; ++r) {                                \
        float value = lane < 4 ? warp_sums[lane][r] : 0.0f;                    \
        for (int offset = 16; offset > 0; offset >>= 1)                        \
            value += __shfl_down_sync(0xffffffffu, value, offset);             \
        if (lane == 0 && r < rows)                                             \
            outputs[(long long)r * output_size + row] = value;                 \
    }                                                                          \
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// One warp-level int8 tensor-core step: D[16][8] += A[16][16] * B[16][8].
//
// This is the instruction the dp4a path is competing with. A warp issues one
// of these where the scalar loop issues 32 dp4a for the same 2048 MACs, which
// is the whole reason llama.cpp's prompt path outruns ours.
//
// K=16 rather than the wider m16n8k32 because the codebook formats carry two
// half-scales per 32-value group (IQ2_S's scale_low/scale_high split at
// exactly K=16), and a k32 step would mix them into one accumulator with no
// way to weight the halves differently. Two k16 steps per group keep the
// scales separable and cost nothing extra -- the format and the instruction
// happen to divide the same way.
//
// Fragment layout (PTX ISA, mma.m16n8k16 with .s8), lane l:
//   quad = l >> 2, slot = l & 3
//   A: rows {quad, quad+8}, K = slot*4 + {0,1,2,3}  -> a[0], a[1]
//   B: col (token) = quad,  K = slot*4 + {0,1,2,3}  -> b
//   D: rows {quad, quad+8} x cols {slot*2, slot*2+1} -> d[0..3]
// A's four consecutive K per register is exactly how *_q8_decode already packs
// words[k] (elements 4k..4k+3), so fragments come straight out of the decoder.
//
// The PTX form needs sm_75 or newer. NVRTC compiles this corpus for whatever
// the device actually reports (--gpu-architecture=compute_XY), so an
// unguarded mma here does not degrade on an older part -- it fails to compile,
// and every kernel in the corpus goes with it. Below sm_75 the emulation is
// taken instead, and the host-side dispatch prefers the dp4a tile kernel there
// so the emulation is never actually the hot path.
__device__ __forceinline__ void mma_m16n8k16_s8(int* d, const int* a, int b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    // Emulation: the CPU corpus (so the contract can check the layout maths
    // above without a GPU), and any device below sm_75. Correctness only -- it rebuilds each operand
    // through the warp shuffle the shim already emulates, which is far slower
    // than the scalar loop it stands in for and is never compiled for device.
    const int lane = (int)(threadIdx.x & 31u);
    const int quad = lane >> 2, slot = lane & 3;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int row = (item < 2) ? quad : quad + 8;
        const int col = slot * 2 + (item & 1);
        int total = 0;
        for (int word = 0; word < 4; ++word) {
            // A[row][*] lives in the lane owning that row, in a[0] for the
            // top half of the tile and a[1] for the bottom.
            const int a_word = __shfl_sync(
                0xffffffffu, (row < 8) ? a[0] : a[1], (row & 7) * 4 + word);
            const int b_word = __shfl_sync(0xffffffffu, b, col * 4 + word);
            for (int byte = 0; byte < 4; ++byte) {
                const int lhs = (int)(signed char)((a_word >> (8 * byte)) & 0xff);
                const int rhs = (int)(signed char)((b_word >> (8 * byte)) & 0xff);
                total += lhs * rhs;
            }
        }
        d[item] += total;
    }
#endif
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Two 8x8 b16 tiles out of shared memory in one instruction.
//
// This is the load the MMA above wants, spelled as one instruction instead of
// four. `mma.m16n8k16.s8` takes A as two registers -- rows {quad, quad+8},
// four consecutive K bytes each -- which is byte for byte a pair of m8n8.b16
// tiles. Building that by hand costs four LDS.32 with four computed indices;
// two of these cost two, and the LSU does the lane shuffle.
//
// The contract, which is what dictates the staging layout in the callers:
// lanes 0-15 supply the address of row (l % 8) of matrix (l / 8), and every
// lane receives in d[m] the four bytes at offset (l & 3) * 4 of row (l >> 2)
// of matrix m. So each row of each tile must be 16 contiguous, 16-byte
// aligned bytes, and the eight rows of a tile must land in eight different
// 16-byte bank groups or the load serializes -- `ldmatrix` picks up no
// conflict freedom for free. COLIBRI_MMQ_ROW_UNITS arranges both.
//
// x2 rather than the x4 that would fetch both k16 halves at once, which is
// what this was written as first: x4 needs an aligned register *quad*, and
// ptxas already clamps the MMQ kernel at 255 registers, so the extra
// allocation constraint made it spill (156 bytes, and 8% slower measured).
// The pair constraint of x2 costs nothing -- see the kernel header comment
// for what the whole exercise was worth.
__device__ __forceinline__ void ldmatrix_x2_b16(int* d, const void* source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    unsigned int address;
    asm("{ .reg .u64 wide;\n"
        "  cvta.to.shared.u64 wide, %1;\n"
        "  cvt.u32.u64 %0, wide; }"
        : "=r"(address) : "l"(source));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(address));
#else
    const int lane = (int)(threadIdx.x & 31u);
    const int* held_at = (const int*)source;
    const int held[4] = {held_at[0], held_at[1], held_at[2], held_at[3]};
    #pragma unroll
    for (int matrix = 0; matrix < 2; ++matrix) {
        const int owner = matrix * 8 + (lane >> 2);
        int value = 0;
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int candidate = __shfl_sync(0xffffffffu, held[word], owner);
            if (word == (lane & 3)) value = candidate;
        }
        d[matrix] = value;
    }
#endif
}

// Prefill twin of COLIBRI_Q8_MATVEC_ROWS, for wide token batches.
//
// The kernel above caps its batch at COLIBRI_Q8_ROWS because `partial[]` is a
// per-thread register array: every extra token costs a register in every
// thread, and past eight the accumulators spill and occupancy collapses.
// Measured on Qwen3.8-27B, raising that cap alone goes the wrong way --
// 8 -> 92 tok/s, 16 -> 84, 32 -> 48.
//
// The cost that matters at prefill widths is not weight *bandwidth* (a 1500
// token prefill moves ~98 GB/s on a part that can do ~600) but the decode ALU
// work, which the kernel above pays once per group per eight tokens. So this
// one decodes a K-tile of groups into shared memory once and lets every warp
// read it, which unbinds "tokens per decode" from "registers per thread": the
// tokens are split across warps, so each thread still holds only
// COLIBRI_Q8_TILE_TOKENS / COLIBRI_Q8_TILE_WARPS accumulators while the block
// covers eight times the batch.
//
// Layout: one block per output row. Lane l of every warp owns group l of the
// current tile; warp w owns tokens w, w+8, w+16, ... The final reduction is
// therefore over lanes within a warp, one shuffle chain per token.
#define COLIBRI_Q8_TILE_GROUPS 32
#define COLIBRI_Q8_TILE_WARPS 8
#define COLIBRI_Q8_TILE_TOKENS 32
#define COLIBRI_Q8_TILE_ROWS 16
#define COLIBRI_Q8_TILE_SLOTS (COLIBRI_Q8_TILE_TOKENS / COLIBRI_Q8_TILE_WARPS)

#define COLIBRI_Q8_MATMUL_TILED(name, decode_fn, stride)                       \
extern "C" __global__ void name(                                               \
    const unsigned char* packed, const signed char* vectors,                   \
    const __half* vector_scales, float* outputs,                               \
    const int input_size, const int output_size,                               \
    const int rows, const int scale_stride                                     \
) {                                                                            \
    const int row_base = blockIdx.x * COLIBRI_Q8_TILE_ROWS;                    \
    if (row_base >= output_size) return;                                       \
    const int blocks_per_row = input_size >> 8;                                \
    const int groups_per_row = blocks_per_row << 3;                            \
    /* [.][9] not [.][8]: lane l reads word k at l*9+k, and 9 is coprime with  */\
    /* the 32 banks, so the eight reads stay conflict-free. At [.][8] lanes    */\
    /* l and l+4 collide four ways.                                            */\
    __shared__ int tile_words[COLIBRI_Q8_TILE_ROWS][COLIBRI_Q8_TILE_GROUPS][9]; \
    __shared__ float tile_low[COLIBRI_Q8_TILE_ROWS][COLIBRI_Q8_TILE_GROUPS];    \
    __shared__ float tile_high[COLIBRI_Q8_TILE_ROWS][COLIBRI_Q8_TILE_GROUPS];   \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    float acc[COLIBRI_Q8_TILE_SLOTS][COLIBRI_Q8_TILE_ROWS];                    \
    _Pragma("unroll")                                                          \
    for (int s = 0; s < COLIBRI_Q8_TILE_SLOTS; ++s)                            \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < COLIBRI_Q8_TILE_ROWS; ++r) acc[s][r] = 0.0f;       \
    for (int base = 0; base < groups_per_row; base += COLIBRI_Q8_TILE_GROUPS) {\
        const int remaining = groups_per_row - base;                           \
        const int count = remaining < COLIBRI_Q8_TILE_GROUPS                   \
            ? remaining : COLIBRI_Q8_TILE_GROUPS;                              \
        __syncthreads();                                                       \
        /* One decode per (row, group) in the tile, spread over the block. */  \
        for (int i = threadIdx.x; i < COLIBRI_Q8_TILE_ROWS * count;            \
             i += blockDim.x) {                                                \
            const int r = i / count;                                           \
            const int g = i - r * count;                                       \
            if (row_base + r >= output_size) continue;                         \
            const unsigned char* row_data = packed                             \
                + (long long)(row_base + r) * blocks_per_row * stride;         \
            float low = 0.0f, high = 0.0f;                                     \
            decode_fn(row_data, base + g, tile_words[r][g], &low, &high);      \
            tile_low[r][g] = low;                                              \
            tile_high[r][g] = high;                                            \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane >= count) continue;                                           \
        const int group = base + lane;                                         \
        /* Both operands are staged before the row loop, so the row loop reads */\
        /* each weight word from shared once and reuses it across every token  */\
        /* slot. With those loads inside the slot loop the kernel issues about */\
        /* one shared load per dp4a; hoisting them cuts that to one per        */\
        /* COLIBRI_Q8_TILE_SLOTS. Shared throughput is what sets the pace here */\
        /* -- row tiling already took the bandwidth terms out of contention.   */\
        /* Inactive slots are zeroed rather than branched around, so the inner */\
        /* loop stays straight-line.                                           */\
        int acts[COLIBRI_Q8_TILE_SLOTS][8];                                    \
        float activation_scale[COLIBRI_Q8_TILE_SLOTS];                         \
        _Pragma("unroll")                                                      \
        for (int s = 0; s < COLIBRI_Q8_TILE_SLOTS; ++s) {                      \
            const int token = warp + s * COLIBRI_Q8_TILE_WARPS;                \
            if (token < rows) {                                                \
                const int4* activation_vectors = (const int4*)(                \
                    vectors + (long long)token * input_size                    \
                    + (long long)group * 32);                                  \
                const int4 activation_low = activation_vectors[0];             \
                const int4 activation_high = activation_vectors[1];            \
                acts[s][0] = activation_low.x;                                 \
                acts[s][1] = activation_low.y;                                 \
                acts[s][2] = activation_low.z;                                 \
                acts[s][3] = activation_low.w;                                 \
                acts[s][4] = activation_high.x;                                \
                acts[s][5] = activation_high.y;                                \
                acts[s][6] = activation_high.z;                                \
                acts[s][7] = activation_high.w;                                \
                activation_scale[s] = __half2float(                            \
                    vector_scales[(long long)token * scale_stride + group]);   \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int k = 0; k < 8; ++k) acts[s][k] = 0;                    \
                activation_scale[s] = 0.0f;                                    \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < COLIBRI_Q8_TILE_ROWS; ++r) {                       \
            int weights[8];                                                    \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < 8; ++k) weights[k] = tile_words[r][lane][k];   \
            const float weight_low = tile_low[r][lane];                        \
            const float weight_high = tile_high[r][lane];                      \
            _Pragma("unroll")                                                  \
            for (int s = 0; s < COLIBRI_Q8_TILE_SLOTS; ++s) {                  \
                int dot_low = 0, dot_high = 0;                                 \
                _Pragma("unroll")                                              \
                for (int step = 0; step < 4; ++step) {                         \
                    int dot = 0;                                               \
                    dot = __dp4a(weights[step * 2], acts[s][step * 2], dot);   \
                    dot = __dp4a(                                              \
                        weights[step * 2 + 1], acts[s][step * 2 + 1], dot);    \
                    if (step < 2) dot_low += dot; else dot_high += dot;        \
                }                                                              \
                acc[s][r] += ((float)dot_low * weight_low                      \
                              + (float)dot_high * weight_high)                 \
                    * activation_scale[s];                                     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int s = 0; s < COLIBRI_Q8_TILE_SLOTS; ++s) {                          \
        const int token = warp + s * COLIBRI_Q8_TILE_WARPS;                    \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < COLIBRI_Q8_TILE_ROWS; ++r) {                       \
            float value = acc[s][r];                                           \
            for (int offset = 16; offset > 0; offset >>= 1)                    \
                value += __shfl_down_sync(0xffffffffu, value, offset);         \
            if (lane == 0 && token < rows && row_base + r < output_size)       \
                outputs[(long long)token * output_size + row_base + r] = value;\
        }                                                                      \
    }                                                                          \
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Tensor-core prefill matmul. Same contract as COLIBRI_Q8_MATMUL_TILED, which
// it replaces wherever the batch is wide enough to fill it.
//
// The dp4a tile kernel holds its output tile in per-thread registers, which is
// what caps it: 128 registers a thread, two blocks an SM, 33% occupancy, and
// measurements ~8.5x off what its instruction count predicts -- it stalls
// rather than issues. A tensor-core fragment distributes the same tile across
// the warp instead, so the accumulators cost four registers a thread rather
// than sixty-four. The register cliff disappears and the occupancy that hides
// the latency comes back with it.
//
// Shape: a warp owns a COLIBRI_MMQ_ROW_FRAGS x COLIBRI_MMQ_TOKEN_FRAGS grid of
// m16n8k16 fragments -- 4 x 4 here, so 64 rows x 32 tokens per warp. Eight
// warps tile that 2 deep in rows and 4 wide in tokens, so a block covers
// 128 rows x 128 tokens with 256 threads. K advances COLIBRI_MMQ_GROUPS
// super-block groups at a time, staged through shared memory.
//
// Two things fall out of the format rather than being designed: the decoder
// already packs words[k] as elements 4k..4k+3, which is exactly A's four
// consecutive K per register, so fragments load straight from the staged
// words with no repacking; and a 32-value group is two k16 steps, which is
// where IQ2_S's scale_low/scale_high divide, so the halves stay separable.
//
// No cross-lane reduction at the end: the MMA already summed over K within the
// warp, so each thread's accumulators are finished values.
//
// What the shape is tuned against, measured by ablation on a 27B IQ2_S
// checkpoint (each figure is the kernel with that part removed):
//
//     scaling epilogue   +18%      <- largest single cost
//     codebook decode    +19%
//     activation reads    +4%
//     tensor-core MMA     +3%      <- arithmetic was never the problem
//
// So the kernel is bound by shared-memory traffic around the MMA, not by the
// MMA or by DRAM. Two things follow, and both are why a warp owns a *grid* of
// fragments rather than one:
//
//   - Reuse. A weight fragment and its two scales are loaded once and reused
//     across TOKEN_FRAGS MMAs; an activation fragment and its two scales once
//     across ROW_FRAGS. Per group per thread that is 4*TOKEN_FRAGS +
//     8*ROW_FRAGS shared loads for 2*ROW_FRAGS*TOKEN_FRAGS MMAs: 6 per MMA at
//     1x1, 3 at 2x2. Measured, at a fixed 64x64 tile: 1x1 403 tok/s (1024
//     threads), 2x1 441, 1x2 469, 2x2 515, 4x2 472, 4x4 291 -- it tracks the
//     load count down to 256 threads, then falls off as blocks get too small
//     to hide latency.
//   - Tile size. At one fragment per warp the tile *was* the thread count, and
//     128 rows would have needed 2048 threads. Decoupled, 128x128 fits in 256
//     threads and beat 64x64 by 6.1% (3/3 paired runs) because it quarters the
//     activation re-read, which is (output_size/ROWS) x TOKENS x K.
//
// GROUPS is the K depth staged per iteration and is also the block's decode
// parallelism -- the staging loops run over ROWS*GROUPS and TOKENS*GROUPS. It
// is what pays for the bigger tile here: at GROUPS 4 the 128x128 tile needs
// 42 KB, under the 48 KB a kernel gets without opting in, and 512 decode items
// still cover 256 threads twice over. (Cutting GROUPS to 4 at the old 16-row
// tile was a loss for the opposite reason: 64 items over 512 threads.)
//
// Two things that did *not* work, so they are not retried:
//
//   - Pipelining the K loop. Prefetching the next step into registers so its
//     global loads overlap this step's MMAs measured 10% *slower* (3/3 paired),
//     and prefetching only the weights was a wash (0.993). acc[4][4][4] is
//     already 64 registers and the 42 KB tile caps occupancy at 2 blocks/SM;
//     the GPU has enough resident warps to hide the decode without help, and
//     the prefetch buffer costs more than the latency it hides.
//   - Raising the >48 KB shared cap (cuFuncSetAttribute + dynamic shared). It
//     works, but the tiles that win fit statically and it measured a wash.
//   - `ldmatrix` for the *activation* fragments. One x4 would fetch all four
//     token fragments at once, and the staging layout below already supports
//     it, but it spilled ~100 bytes and cost 8%. See below for why.
//
// The weight fragments do come out of `ldmatrix` (see ldmatrix_x2_b16), which
// is what llama.cpp's mmq does and was the identified gap against it. Honest
// accounting: it is worth about +0.5%, three paired rounds, 1.006 at pp1024
// and 1.004 at pp2048 -- a wash, not the 1.7x the gap was supposed to hold.
// The reason it does not pay, and it is the more useful finding:
//
//   ptxas compiles this kernel at 255 registers with 256 threads, which is
//   65280 of the SM's 65536 -- *one* block per SM, eight warps, 12.5%
//   occupancy. Registers cap it, not the 42 KB of shared (which would allow
//   two). So the kernel is latency bound with almost nothing resident to hide
//   the latency with, and `ldmatrix` does not help that: four LDS.32 and two
//   LDSM.x2 move the same four 128-byte wavefronts, so it saves issue slots,
//   which are not the constraint. It also explains the pipelining result
//   above -- a prefetch buffer has to come out of a register file with no
//   slack -- and why anything that adds allocation constraints spills.
//
// Which is where the wide shape comes from: acc[4][4][4] is 64 registers of
// the budget and forcing __launch_bounds__(256, 2) to get a second block
// spills 792 bytes, so buying occupancy needs the fragment grid to shrink,
// not just a flag. The ptxas sweep (2026-08-25) found the spill-free point:
// keep the 128x128 tile -- so none of the weight-decode or activation
// re-reads the tile size was chosen to avoid come back -- and spread it over
// 16 warps at 2x4 fragments each instead of 8 warps at 4x4. Same tile, same
// shared layout, half the accumulator per thread: 128 registers, 8-32 bytes
// of spill, and 512 threads x 128 registers fills the register file exactly,
// so 16 warps are resident where the 4x4 shape leaves 8. (Shrinking the tile
// instead -- 128x64, 64x128, 64x64 at 256 threads -- also reaches 128
// registers, but every one of those re-pays decode or activation traffic
// that the 128x128 tile exists to amortize.)
//
// The defines are guarded so the GPU compile can inject the wide shape ahead
// of the corpus (COLIBRI_MMQ_WIDE; see qwen_mmq_wide() on the host). The CPU
// backend's copy of this corpus is generated at build time with the defaults
// and its launches stay at 256 threads -- occupancy is a GPU concept, and the
// shim ignores __launch_bounds__ entirely.
#ifndef COLIBRI_MMQ_ROW_WARPS
#define COLIBRI_MMQ_ROW_WARPS 2
#endif
#ifndef COLIBRI_MMQ_ROW_FRAGS
#define COLIBRI_MMQ_ROW_FRAGS 4
#endif
#ifndef COLIBRI_MMQ_TOKEN_WARPS
#define COLIBRI_MMQ_TOKEN_WARPS 4
#endif
#ifndef COLIBRI_MMQ_TOKEN_FRAGS
#define COLIBRI_MMQ_TOKEN_FRAGS 4
#endif
// The _MIN variant stages four more arrays (the two sub-block minimum offsets
// and the two activation sums), which at 64 rows would want 50 KB -- past the
// 48 KB a kernel gets without opting in. It keeps the old 32-row tile, at
// 37 KB. The host dispatches rows and threads per family; see kQ8MmqMinRows.
// The _MIN variant still runs one fragment per warp and so keeps its own,
// smaller tile: the plain kernel's 128x128 would need 1024 threads here and
// 50 KB of shared for the four extra staged arrays. 32x64 needs 18.5 KB and
// 512 threads. The host chunks tokens per family -- see kQ8MmqMinTokens.
#define COLIBRI_MMQ_MIN_TOKEN_WARPS 8
#define COLIBRI_MMQ_MIN_TOKENS (COLIBRI_MMQ_MIN_TOKEN_WARPS * 8)
#define COLIBRI_MMQ_MIN_ROW_WARPS 2
#define COLIBRI_MMQ_MIN_ROWS (COLIBRI_MMQ_MIN_ROW_WARPS * 16)
#define COLIBRI_MMQ_ROWS \
    (COLIBRI_MMQ_ROW_WARPS * COLIBRI_MMQ_ROW_FRAGS * 16)
#define COLIBRI_MMQ_TOKENS \
    (COLIBRI_MMQ_TOKEN_WARPS * COLIBRI_MMQ_TOKEN_FRAGS * 8)
#define COLIBRI_MMQ_GROUPS 4

// The staged tile is addressed in 16-byte units, because that is what
// `ldmatrix` reads: one row of an 8x8 b16 tile is 16 contiguous, 16-byte
// aligned bytes, which is exactly one k16 half of one (row, group). So the
// staging atom stops being "nine ints with an odd stride" and becomes an
// int4, two per (row, group).
//
// The old [9] padding spread the eight row-quads of a warp across the banks;
// it cannot survive here, since ldmatrix needs the unit stride to be a whole
// 16 bytes. Padding by a whole unit per *row* does the same job: the unit
// index is row * ROW_UNITS + group * 2 + half, and with an odd ROW_UNITS the
// bank group (unit mod 8) advances by one per row, so the eight rows of a
// tile land in eight different bank groups. Padding rather than an XOR
// swizzle because it leaves the two halves of a (row, group) adjacent, which
// is what lets the decoders keep writing eight contiguous ints -- staging
// through a register array to reassemble a swizzled pair costs ~45 registers
// in a kernel that ptxas already clamps at 255, and it spills.
//
// This is the same 36 ints per row the [9] layout used, so shared memory is
// unchanged; only where the slack sits moves.
#define COLIBRI_MMQ_ROW_UNITS (COLIBRI_MMQ_GROUPS * 2 + 1)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
#define COLIBRI_Q8_MMQ(name, decode_fn, stride)                                \
/* The bound caps registers at what the block's thread count leaves: 255 at  */\
/* the default 256 threads (no change), 128 at the wide shape's 512, which   */\
/* is what makes 16 warps launchable at all. The CPU shim defines this away. */\
extern "C" __global__                                                          \
__launch_bounds__(COLIBRI_MMQ_ROW_WARPS * COLIBRI_MMQ_TOKEN_WARPS * 32, 1)     \
void name(                                                                     \
    const unsigned char* packed, const signed char* vectors,                   \
    const __half* vector_scales, float* outputs,                               \
    const int input_size, const int output_size,                               \
    const int rows, const int scale_stride                                     \
) {                                                                            \
    const int row_base = blockIdx.x * COLIBRI_MMQ_ROWS;                        \
    if (row_base >= output_size) return;                                       \
    const int blocks_per_row = input_size >> 8;                                \
    const int groups_per_row = blocks_per_row << 3;                            \
    /* Staged in 16-byte units -- one k16 half of one (row, group), which is */\
    /* one ldmatrix row. See COLIBRI_MMQ_ROW_UNITS for the padding.          */\
    __shared__ int4 w_units[COLIBRI_MMQ_ROWS][COLIBRI_MMQ_ROW_UNITS];          \
    __shared__ float w_low[COLIBRI_MMQ_ROWS][COLIBRI_MMQ_GROUPS];              \
    __shared__ float w_high[COLIBRI_MMQ_ROWS][COLIBRI_MMQ_GROUPS];             \
    __shared__ int4 a_units[COLIBRI_MMQ_TOKENS][COLIBRI_MMQ_ROW_UNITS];        \
    __shared__ float a_scale[COLIBRI_MMQ_TOKENS][COLIBRI_MMQ_GROUPS];          \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    const int warp_row =                                                  \
        (warp / COLIBRI_MMQ_TOKEN_WARPS) * (COLIBRI_MMQ_ROW_FRAGS * 16);  \
    const int warp_token =                                                \
        (warp % COLIBRI_MMQ_TOKEN_WARPS) * (COLIBRI_MMQ_TOKEN_FRAGS * 8); \
    const int quad = lane >> 2;                                           \
    const int slot = lane & 3;                                            \
    float acc[COLIBRI_MMQ_ROW_FRAGS][COLIBRI_MMQ_TOKEN_FRAGS][4];         \
    _Pragma("unroll")                                                     \
    for (int rf = 0; rf < COLIBRI_MMQ_ROW_FRAGS; ++rf)                    \
        _Pragma("unroll")                                                 \
        for (int tf = 0; tf < COLIBRI_MMQ_TOKEN_FRAGS; ++tf)              \
            _Pragma("unroll")                                             \
            for (int i = 0; i < 4; ++i) acc[rf][tf][i] = 0.0f;            \
    for (int base = 0; base < groups_per_row; base += COLIBRI_MMQ_GROUPS) {    \
        __syncthreads();                                                       \
        for (int i = threadIdx.x;                                              \
             i < COLIBRI_MMQ_ROWS * COLIBRI_MMQ_GROUPS; i += blockDim.x) {     \
            const int r = i / COLIBRI_MMQ_GROUPS;                              \
            const int g = i - r * COLIBRI_MMQ_GROUPS;                          \
            float low = 0.0f, high = 0.0f;                                     \
            /* The two k16 halves are adjacent units, so this is still the    */\
            /* eight contiguous ints every decoder writes.                    */\
            int* const decoded = (int*)&w_units[r][g * 2];                     \
            if (row_base + r < output_size && base + g < groups_per_row) {     \
                decode_fn(packed + (long long)(row_base + r)                   \
                              * blocks_per_row * stride,                       \
                          base + g, decoded, &low, &high);                     \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int k = 0; k < 8; ++k) decoded[k] = 0;                    \
            }                                                                  \
            w_low[r][g] = low;                                                 \
            w_high[r][g] = high;                                               \
        }                                                                      \
        for (int i = threadIdx.x;                                              \
             i < COLIBRI_MMQ_TOKENS * COLIBRI_MMQ_GROUPS; i += blockDim.x) {   \
            const int t = i / COLIBRI_MMQ_GROUPS;                              \
            const int g = i - t * COLIBRI_MMQ_GROUPS;                          \
            if (t < rows && base + g < groups_per_row) {                       \
                const int4* source = (const int4*)(                            \
                    vectors + (long long)t * input_size                        \
                    + (long long)(base + g) * 32);                             \
                a_units[t][g * 2] = source[0];                                 \
                a_units[t][g * 2 + 1] = source[1];                             \
                a_scale[t][g] = __half2float(                                  \
                    vector_scales[(long long)t * scale_stride + base + g]);    \
            } else {                                                           \
                a_units[t][g * 2] = make_int4(0, 0, 0, 0);                     \
                a_units[t][g * 2 + 1] = make_int4(0, 0, 0, 0);                 \
                a_scale[t][g] = 0.0f;                                          \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                 \
        for (int g = 0; g < COLIBRI_MMQ_GROUPS; ++g) {                    \
            /* Activation fragment and its two output-column scales, out */\
            /* of the row loop: every row fragment reuses them. Scalar,  */\
            /* not ldmatrix -- see the header comment.                   */\
            int a_low[COLIBRI_MMQ_TOKEN_FRAGS];                           \
            int a_high[COLIBRI_MMQ_TOKEN_FRAGS];                          \
            float a_s0[COLIBRI_MMQ_TOKEN_FRAGS];                          \
            float a_s1[COLIBRI_MMQ_TOKEN_FRAGS];                          \
            _Pragma("unroll")                                             \
            for (int tf = 0; tf < COLIBRI_MMQ_TOKEN_FRAGS; ++tf) {        \
                const int tb = warp_token + tf * 8;                       \
                a_low[tf] = ((const int*)&a_units[tb + quad][g * 2])[slot];\
                a_high[tf] =                                              \
                    ((const int*)&a_units[tb + quad][g * 2 + 1])[slot];   \
                a_s0[tf] = a_scale[tb + slot * 2][g];                     \
                a_s1[tf] = a_scale[tb + slot * 2 + 1][g];                 \
            }                                                             \
            _Pragma("unroll")                                             \
            for (int rf = 0; rf < COLIBRI_MMQ_ROW_FRAGS; ++rf) {          \
                const int rb = warp_row + rf * 16;                        \
                /* Weight fragment and its two output-row scales, loaded */\
                /* once and reused across every token fragment. One      */\
                /* ldmatrix per k16 half: its two tiles are the two row  */\
                /* halves the MMA wants, so lane l hands it row l & 15.  */\
                int frag_low[2], frag_high[2];                            \
                ldmatrix_x2_b16(                                          \
                    frag_low, &w_units[rb + (lane & 15)][g * 2]);         \
                ldmatrix_x2_b16(                                          \
                    frag_high, &w_units[rb + (lane & 15)][g * 2 + 1]);    \
                const float wl0 = w_low[rb + quad][g];                    \
                const float wl1 = w_low[rb + quad + 8][g];                \
                const float wh0 = w_high[rb + quad][g];                   \
                const float wh1 = w_high[rb + quad + 8][g];               \
                _Pragma("unroll")                                         \
                for (int tf = 0; tf < COLIBRI_MMQ_TOKEN_FRAGS; ++tf) {    \
                    int low_dot[4] = {0, 0, 0, 0};                        \
                    int high_dot[4] = {0, 0, 0, 0};                       \
                    mma_m16n8k16_s8(low_dot, frag_low, a_low[tf]);        \
                    mma_m16n8k16_s8(high_dot, frag_high, a_high[tf]);     \
                    _Pragma("unroll")                                     \
                    for (int item = 0; item < 4; ++item) {                \
                        const float wl = (item < 2) ? wl0 : wl1;          \
                        const float wh = (item < 2) ? wh0 : wh1;          \
                        const float as = (item & 1) ? a_s1[tf] : a_s0[tf];\
                        acc[rf][tf][item] += ((float)low_dot[item] * wl   \
                            + (float)high_dot[item] * wh) * as;           \
                    }                                                     \
                }                                                         \
            }                                                             \
        }                                                                 \
    }                                                                          \
    _Pragma("unroll")                                                     \
    for (int rf = 0; rf < COLIBRI_MMQ_ROW_FRAGS; ++rf)                    \
        _Pragma("unroll")                                                 \
        for (int tf = 0; tf < COLIBRI_MMQ_TOKEN_FRAGS; ++tf)              \
            _Pragma("unroll")                                             \
            for (int item = 0; item < 4; ++item) {                        \
                const int r = row_base + warp_row + rf * 16               \
                    + ((item < 2) ? quad : quad + 8);                     \
                const int t = warp_token + tf * 8 + slot * 2 + (item & 1);\
                if (r < output_size && t < rows)                          \
                    outputs[(long long)t * output_size + r] =             \
                        acc[rf][tf][item];                                \
            }                                                             \
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// COLIBRI_Q8_MMQ for the asymmetric K-quants; see COLIBRI_Q8_MATVEC_ROWS_MIN
// for why the minimum needs the activation sums.
//
// The sums are staged once per (token, group) beside the activations rather
// than recomputed per output row, which is the advantage this shape has over
// the row kernel: 64 tokens share one pass, so the correction costs two shared
// arrays and two fused multiply-adds per accumulator, against a tensor-core
// MMA per fragment.
#define COLIBRI_Q8_MMQ_MIN(name, decode_fn, stride)                            \
extern "C" __global__ void name(                                               \
    const unsigned char* packed, const signed char* vectors,                   \
    const __half* vector_scales, float* outputs,                               \
    const int input_size, const int output_size,                               \
    const int rows, const int scale_stride                                     \
) {                                                                            \
    const int row_base = blockIdx.x * COLIBRI_MMQ_MIN_ROWS;                        \
    if (row_base >= output_size) return;                                       \
    const int blocks_per_row = input_size >> 8;                                \
    const int groups_per_row = blocks_per_row << 3;                            \
    __shared__ int w_words[COLIBRI_MMQ_MIN_ROWS][COLIBRI_MMQ_GROUPS][9];           \
    __shared__ float w_low[COLIBRI_MMQ_MIN_ROWS][COLIBRI_MMQ_GROUPS];              \
    __shared__ float w_high[COLIBRI_MMQ_MIN_ROWS][COLIBRI_MMQ_GROUPS];             \
    __shared__ float w_off_low[COLIBRI_MMQ_MIN_ROWS][COLIBRI_MMQ_GROUPS];          \
    __shared__ float w_off_high[COLIBRI_MMQ_MIN_ROWS][COLIBRI_MMQ_GROUPS];         \
    __shared__ int a_words[COLIBRI_MMQ_MIN_TOKENS][COLIBRI_MMQ_GROUPS][9];         \
    __shared__ float a_scale[COLIBRI_MMQ_MIN_TOKENS][COLIBRI_MMQ_GROUPS];          \
    __shared__ float a_sum_low[COLIBRI_MMQ_MIN_TOKENS][COLIBRI_MMQ_GROUPS];        \
    __shared__ float a_sum_high[COLIBRI_MMQ_MIN_TOKENS][COLIBRI_MMQ_GROUPS];       \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    const int warp_row = (warp / COLIBRI_MMQ_MIN_TOKEN_WARPS) * 16;                \
    const int warp_token = (warp % COLIBRI_MMQ_MIN_TOKEN_WARPS) * 8;               \
    const int quad = lane >> 2;                                                \
    const int slot = lane & 3;                                                 \
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};                                   \
    for (int base = 0; base < groups_per_row; base += COLIBRI_MMQ_GROUPS) {    \
        __syncthreads();                                                       \
        for (int i = threadIdx.x;                                              \
             i < COLIBRI_MMQ_MIN_ROWS * COLIBRI_MMQ_GROUPS; i += blockDim.x) {     \
            const int r = i / COLIBRI_MMQ_GROUPS;                              \
            const int g = i - r * COLIBRI_MMQ_GROUPS;                          \
            float low = 0.0f, high = 0.0f, off_low = 0.0f, off_high = 0.0f;    \
            if (row_base + r < output_size && base + g < groups_per_row) {     \
                decode_fn(packed + (long long)(row_base + r)                   \
                              * blocks_per_row * stride,                       \
                          base + g, w_words[r][g], &low, &high,                \
                          &off_low, &off_high);                                \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int k = 0; k < 8; ++k) w_words[r][g][k] = 0;              \
            }                                                                  \
            w_low[r][g] = low;                                                 \
            w_high[r][g] = high;                                               \
            w_off_low[r][g] = off_low;                                         \
            w_off_high[r][g] = off_high;                                       \
        }                                                                      \
        for (int i = threadIdx.x;                                              \
             i < COLIBRI_MMQ_MIN_TOKENS * COLIBRI_MMQ_GROUPS; i += blockDim.x) {   \
            const int t = i / COLIBRI_MMQ_GROUPS;                              \
            const int g = i - t * COLIBRI_MMQ_GROUPS;                          \
            if (t < rows && base + g < groups_per_row) {                       \
                const int4* source = (const int4*)(                            \
                    vectors + (long long)t * input_size                        \
                    + (long long)(base + g) * 32);                             \
                const int4 first = source[0], second = source[1];              \
                a_words[t][g][0] = first.x;  a_words[t][g][1] = first.y;       \
                a_words[t][g][2] = first.z;  a_words[t][g][3] = first.w;       \
                a_words[t][g][4] = second.x; a_words[t][g][5] = second.y;      \
                a_words[t][g][6] = second.z; a_words[t][g][7] = second.w;      \
                int sum_low = 0, sum_high = 0;                                 \
                _Pragma("unroll")                                              \
                for (int k = 0; k < 4; ++k) {                                  \
                    sum_low = __dp4a(0x01010101, a_words[t][g][k], sum_low);   \
                    sum_high =                                                 \
                        __dp4a(0x01010101, a_words[t][g][4 + k], sum_high);    \
                }                                                              \
                a_sum_low[t][g] = (float)sum_low;                              \
                a_sum_high[t][g] = (float)sum_high;                            \
                a_scale[t][g] = __half2float(                                  \
                    vector_scales[(long long)t * scale_stride + base + g]);    \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int k = 0; k < 8; ++k) a_words[t][g][k] = 0;              \
                a_scale[t][g] = 0.0f;                                          \
                a_sum_low[t][g] = 0.0f;                                        \
                a_sum_high[t][g] = 0.0f;                                       \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int g = 0; g < COLIBRI_MMQ_GROUPS; ++g) {                         \
            int fragment[2];                                                   \
            int low_dot[4] = {0, 0, 0, 0};                                     \
            int high_dot[4] = {0, 0, 0, 0};                                    \
            fragment[0] = w_words[warp_row + quad][g][slot];                   \
            fragment[1] = w_words[warp_row + quad + 8][g][slot];               \
            mma_m16n8k16_s8(                                                   \
                low_dot, fragment, a_words[warp_token + quad][g][slot]);       \
            fragment[0] = w_words[warp_row + quad][g][4 + slot];               \
            fragment[1] = w_words[warp_row + quad + 8][g][4 + slot];           \
            mma_m16n8k16_s8(                                                   \
                high_dot, fragment, a_words[warp_token + quad][g][4 + slot]);  \
            _Pragma("unroll")                                                  \
            for (int item = 0; item < 4; ++item) {                             \
                const int r = warp_row + ((item < 2) ? quad : quad + 8);       \
                const int t = warp_token + slot * 2 + (item & 1);              \
                acc[item] += ((float)low_dot[item] * w_low[r][g]               \
                              - a_sum_low[t][g] * w_off_low[r][g]              \
                              + (float)high_dot[item] * w_high[r][g]           \
                              - a_sum_high[t][g] * w_off_high[r][g])           \
                    * a_scale[t][g];                                           \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int item = 0; item < 4; ++item) {                                     \
        const int r = row_base + warp_row + ((item < 2) ? quad : quad + 8);    \
        const int t = warp_token + slot * 2 + (item & 1);                      \
        if (r < output_size && t < rows)                                       \
            outputs[(long long)t * output_size + r] = acc[item];               \
    }                                                                          \
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// One warp per output row, eight rows per block, argmax fused in. The LM head
// is ~250k rows of only ~160 groups each, so a block per row would spend most
// of its time in the cross-warp reduction; a warp per row keeps the reduction
// in registers. Same packed (ordered-float << 32 | ~row) result as the
// per-element LM-head kernels.
#define COLIBRI_Q8_LM_HEAD(name, group_fn, stride)                             \
extern "C" __global__ void name(                                               \
    const unsigned char* packed, const signed char* vector,                    \
    const __half* vector_scales, unsigned long long* winners,                  \
    const int input_size, const int output_size                                \
) {                                                                            \
    __shared__ unsigned long long warp_best[8];                                \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    const int row = blockIdx.x * 8 + warp;                                     \
    if (lane == 0) warp_best[warp] = 0ull;                                     \
    __syncthreads();                                                           \
    if (row < output_size) {                                                   \
        const int blocks_per_row = input_size >> 8;                            \
        const int groups_per_row = blocks_per_row << 3;                        \
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        const unsigned char* row_data =                                        \
            packed + (long long)row * blocks_per_row * stride;                 \
        float partial = 0.0f;                                                  \
        for (int g = lane; g < groups_per_row; g += 32)                        \
            partial += group_fn(row_data, vector, vector_scales, g);           \
        for (int offset = 16; offset > 0; offset >>= 1)                        \
            partial += __shfl_down_sync(0xffffffffu, partial, offset);         \
        if (lane == 0) {                                                       \
            const unsigned int bits = __float_as_uint(partial);                \
            const unsigned int ordered =                                       \
                bits ^ (((int)bits < 0) ? 0xffffffffu : 0x80000000u);          \
            warp_best[warp] = ((unsigned long long)ordered << 32)              \
                | (unsigned int)(0xffffffffu - (unsigned int)row);             \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    if (threadIdx.x == 0) {                                                    \
        unsigned long long best = warp_best[0];                                \
        for (int i = 1; i < 8; ++i) best = max(best, warp_best[i]);            \
        atomicMax(winners, best);                                              \
    }                                                                          \
}


// Q5_K against a Q8-blocked activation. A 32-value Q8 block lines up exactly
// with one Q5_K sub-block, so a single (scale, min) pair covers it and the
// affine reconstruction needs just the weight dot and the activation sum. The
// 5th bit comes from the shared qh array and ORs straight into the nibble.
__device__ __forceinline__ float q5k_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    float partial = 0.0f;

        const int block = linear_group >> 3;
        const int sub_block = linear_group & 7;
        const int group = sub_block >> 1;
        const int sub = sub_block & 1;
        const unsigned char* base = row_data + block * 176;
        const unsigned char* quants = base + 48 + group * 32;
        const unsigned char* highs = base + 16;
        const int shift = sub * 4;
        const int bit_shift = 2 * group + sub;
        const signed char* activations = vector + linear_group * 32;

        // 176-byte super-blocks with qs at +48 and qh at +16: everything here
        // is 16-byte aligned, so weights, high bits and activations all load as
        // int4. See the Q4_K kernel for why the load count is what matters.
        const int4* quant_vectors = (const int4*)quants;
        const int4* high_vectors = (const int4*)highs;
        const int4* activation_vectors = (const int4*)activations;
        int dot = 0, total = 0;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int4 packed_quads = quant_vectors[part];
            const int4 high_quads = high_vectors[part];
            const int4 activation_quads = activation_vectors[part];
            const int words[4] = {packed_quads.x, packed_quads.y,
                                  packed_quads.z, packed_quads.w};
            const int highs4[4] = {high_quads.x, high_quads.y,
                                   high_quads.z, high_quads.w};
            const int acts[4] = {activation_quads.x, activation_quads.y,
                                 activation_quads.z, activation_quads.w};
            #pragma unroll
            for (int quad = 0; quad < 4; ++quad) {
                const unsigned int weights =
                    (((unsigned int)words[quad] >> shift) & 0x0f0f0f0fu)
                    | ((((unsigned int)highs4[quad] >> bit_shift)
                        & 0x01010101u) << 4);
                dot = __dp4a((int)weights, acts[quad], dot);
                total = __dp4a(0x01010101, acts[quad], total);
            }
        }
        const float d = __half2float(*((const __half*)base));
        const float dmin = __half2float(*((const __half*)(base + 2)));
        int scale, minimum;
        q5k_scale_min(base + 4, sub_block, &scale, &minimum);
        partial += __half2float(vector_scales[linear_group])
            * (d * (float)scale * (float)dot
               - dmin * (float)minimum * (float)total);
    return partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(COLIBRI_Q8_MATVEC(q5k_q8_matvec_transposed_warp, q5k_q8_group, 176)
COLIBRI_Q8_LM_HEAD(q5k_q8_lm_head_argmax_warp, q5k_q8_group, 176)

// q5k_q8_group with the dot lifted out. Same sub-block geometry as Q4_K, plus
// the fifth bit from the qh plane.
__device__ __forceinline__ void q5k_q8_decode(
    const unsigned char* row_data, const int linear_group, int* words,
    float* scale_low, float* scale_high,
    float* offset_low, float* offset_high) {
    const int block = linear_group >> 3;
    const int sub_block = linear_group & 7;
    const int group = sub_block >> 1;
    const int sub = sub_block & 1;
    const unsigned char* base = row_data + block * 176;
    int scale, minimum;
    q5k_scale_min(base + 4, sub_block, &scale, &minimum);
    const unsigned char* quants = base + 48 + group * 32;
    const unsigned char* highs = base + 16;
    const int shift = sub * 4;
    const int bit_shift = 2 * group + sub;
    const int4* quant_vectors = (const int4*)quants;
    const int4* high_vectors = (const int4*)highs;
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int4 packed_quads = quant_vectors[part];
        const int4 high_quads = high_vectors[part];
        const int quads[4] = {packed_quads.x, packed_quads.y,
                              packed_quads.z, packed_quads.w};
        const int high4[4] = {high_quads.x, high_quads.y,
                              high_quads.z, high_quads.w};
        #pragma unroll
        for (int step = 0; step < 4; ++step)
            words[part * 4 + step] = (int)(
                (((unsigned int)quads[step] >> shift) & 0x0f0f0f0fu)
                | ((((unsigned int)high4[step] >> bit_shift) & 0x01010101u)
                   << 4));
    }
    const float d = __half2float(*((const __half*)base));
    const float dmin = __half2float(*((const __half*)(base + 2)));
    *scale_low = *scale_high = d * (float)scale;
    *offset_low = *offset_high = dmin * (float)minimum;
}

COLIBRI_Q8_MATVEC_ROWS_MIN(q5k_q8_matvec_transposed_rows, q5k_q8_decode, 176)
COLIBRI_Q8_MMQ_MIN(q5k_q8_mmq, q5k_q8_decode, 176)


// One Q5_K row per warp. The generic value-at-a-time kernel reloads the block
// scales and recomputes the block/within divisions for all eight values owned
// by a lane. Decode those values together, as the Q6_K path does below.
__device__ __forceinline__ float q5k_row_dot_warp(
    const unsigned char* packed, const float* vector,
    int row, int input_size, int lane
) {
    const int blocks = input_size >> 8;
    const unsigned char* row_data =
        packed + (long long)row * blocks * 176;
    float partial = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const unsigned char* base = row_data + block * 176;
        const float d = __half2float(*((const __half*)base));
        const float dmin = __half2float(*((const __half*)(base + 2)));
        const unsigned char* scales = base + 4;
        int lane_scale = 0, lane_minimum = 0;
        if (lane < 8)
            q5k_scale_min(scales, lane, &lane_scale, &lane_minimum);
        const unsigned char high = base[16 + lane];
        const int input_base = block << 8;
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            const unsigned char low = base[48 + group * 32 + lane];
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int index = group * 2 + sub;
                const int scale = __shfl_sync(
                    0xffffffffu, lane_scale, index);
                const int minimum = __shfl_sync(
                    0xffffffffu, lane_minimum, index);
                const int quant =
                    (sub == 0 ? (low & 15) : (low >> 4))
                    | (((high >> index) & 1) << 4);
                partial = fmaf(
                    d * (float)(scale * quant)
                        - dmin * (float)minimum,
                    vector[input_base + group * 64 + sub * 32 + lane],
                    partial);
            }
        }
    }
    return partial;
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// IQ2_XXS codebook. Each grid entry is eight bytes packed into a uint64 so a
// lookup is a single load; the sign table maps a 7-bit selector to eight
// sign bits. Both come from the llama.cpp reference tables.
// The IQ codebooks are indexed divergently -- every lane decodes a different
// grid entry -- which is the pathological case for __constant__ memory: the
// constant cache broadcasts one address per cycle and serialises up to 32 ways.
// In global memory these tables are a few KB, stay pinned in L1/L2 and service
// a fully divergent warp in one pass. Measured on the reference SM120 laptop
// this is worth ~5x on the IQ2_XXS matvec and ~11x on IQ3_XXS.
__device__ const unsigned long long kIq2xxsGrid[256] = {
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
__device__ const unsigned char kIq2xxsSigns[128] = {
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

// Decode IQ2_XXS against a vector quantized in independent 32-value Q8
// blocks.  IQ2's codebook magnitudes fit in signed bytes, so the 32 products
// in one weight group reduce to eight DP4A instructions instead of 32
// floating-point weight reconstructions and multiplies.
__device__ __forceinline__ float iq2xxs_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    float partial = 0.0f;

        const int block = linear_group >> 3;
        const int group = linear_group & 7;
        const unsigned char* base = row_data + block * 66;
        unsigned int low, high;
        memcpy(&low, base + 2 + group * 8, 4);
        memcpy(&high, base + 2 + group * 8 + 4, 4);

        // A 66-byte super-block leaves qs 2-byte aligned, so the codebook
        // indices stay on 32-bit loads, but the 32-byte aligned activation
        // block loads as two int4 instead of eight scalars.
        const int4* activation_vectors =
            (const int4*)(vector + linear_group * 32);
        const int4 activation_low = activation_vectors[0];
        const int4 activation_high = activation_vectors[1];
        const int acts[8] = {
            activation_low.x, activation_low.y,
            activation_low.z, activation_low.w,
            activation_high.x, activation_high.y,
            activation_high.z, activation_high.w};

        int dot = 0;
        #pragma unroll
        for (int quad = 0; quad < 4; ++quad) {
            const unsigned int signs =
                iq2xxs_unpack_signs((high >> (7 * quad)) & 127);
            const unsigned long long pattern =
                kIq2xxsGrid[(low >> (8 * quad)) & 255];
            const int masks_first = __vcmpne4(
                signs & 0x08040201u, 0);
            const int masks_second = __vcmpne4(
                signs & 0x80402010u, 0);
            const int weights_first = __vsub4(
                (int)(unsigned int)pattern ^ masks_first, masks_first);
            const int weights_second = __vsub4(
                (int)(unsigned int)(pattern >> 32) ^ masks_second,
                masks_second);
            dot = __dp4a(weights_first, acts[quad * 2], dot);
            dot = __dp4a(weights_second, acts[quad * 2 + 1], dot);
        }
        const int scale = (int)(high >> 27) | 1;
        partial += (float)(dot * scale) * 0.125f
            * __half2float(*((const __half*)base))
            * __half2float(vector_scales[linear_group]);
    return partial;
}

COLIBRI_Q8_MATVEC(iq2xxs_q8_matvec_transposed_warp, iq2xxs_q8_group, 66)
COLIBRI_Q8_LM_HEAD(iq2xxs_q8_lm_head_argmax_warp, iq2xxs_q8_group, 66)
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Batched twin of iq2xxs_q8_group. One scale covers the whole 32-value group
// here, so both halves get it.
__device__ __forceinline__ void iq2xxs_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 66;
    unsigned int low, high;
    memcpy(&low, base + 2 + group * 8, 4);
    memcpy(&high, base + 2 + group * 8 + 4, 4);
    #pragma unroll
    for (int quad = 0; quad < 4; ++quad) {
        const unsigned int signs =
            iq2xxs_unpack_signs((high >> (7 * quad)) & 127);
        const unsigned long long pattern =
            kIq2xxsGrid[(low >> (8 * quad)) & 255];
        const int masks_first = __vcmpne4(signs & 0x08040201u, 0);
        const int masks_second = __vcmpne4(signs & 0x80402010u, 0);
        words[quad * 2] = __vsub4(
            (int)(unsigned int)pattern ^ masks_first, masks_first);
        words[quad * 2 + 1] = __vsub4(
            (int)(unsigned int)(pattern >> 32) ^ masks_second, masks_second);
    }
    const float scale = (float)((int)(high >> 27) | 1) * 0.125f
        * __half2float(*((const __half*)base));
    *scale_low = scale;
    *scale_high = scale;
}

COLIBRI_Q8_MATVEC_ROWS(iq2xxs_q8_matvec_transposed_rows, iq2xxs_q8_decode, 66)
COLIBRI_Q8_MATMUL_TILED(iq2xxs_q8_matmul_tiled, iq2xxs_q8_decode, 66)
COLIBRI_Q8_MMQ(iq2xxs_q8_mmq, iq2xxs_q8_decode, 66)


// Decode Q4_K against a vector quantized in independent 32-value Q8 blocks.
// A Q4_K sub-block is exactly 32 values and its reconstruction is affine --
// d*scale*quant - dmin*minimum -- so the whole sub-block dot collapses to
// sum(quant*activation) and sum(activation), eight DP4A pairs, instead of 32
// nibble extractions each repeating the superblock header and scale unpack.
// The Q8 blocks are 32 values too, so block index and sub-block index coincide.
__device__ __forceinline__ float q4k_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    float partial = 0.0f;

        const int block = linear_group >> 3;
        const int sub_block = linear_group & 7;
        const unsigned char* base = row_data + block * 144;

        int scale, minimum;
        q5k_scale_min(base + 4, sub_block, &scale, &minimum);

        // Sub-block j lives in the 32 bytes at qs + (j/2)*32, low nibble for
        // even j and high nibble for odd j -- the same mapping q4k_value
        // recomputes per element.
        const unsigned char* quants = base + 16 + (sub_block >> 1) * 32;
        const int shift = (sub_block & 1) * 4;
        const signed char* activations = vector + linear_group * 32;

        // 16-byte loads: a Q4_K super-block is 144 bytes and qs starts at +16,
        // so both the weight quads and the (32-byte aligned) activation block
        // land on int4 boundaries. This is what separates a bandwidth-bound
        // matvec from a load-issue-bound one -- four loads per group, not
        // sixteen.
        const int4* quant_vectors = (const int4*)quants;
        const int4* activation_vectors = (const int4*)activations;
        int dot = 0, total = 0;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int4 packed_quads = quant_vectors[part];
            const int4 activation_quads = activation_vectors[part];
            const int words[4] = {packed_quads.x, packed_quads.y,
                                  packed_quads.z, packed_quads.w};
            const int acts[4] = {activation_quads.x, activation_quads.y,
                                 activation_quads.z, activation_quads.w};
            #pragma unroll
            for (int quad = 0; quad < 4; ++quad) {
                const int weights =
                    (int)(((unsigned int)words[quad] >> shift) & 0x0f0f0f0fu);
                dot = __dp4a(weights, acts[quad], dot);
                total = __dp4a(0x01010101, acts[quad], total);
            }
        }
        const float d = __half2float(*((const __half*)base));
        const float dmin = __half2float(*((const __half*)(base + 2)));
        partial += __half2float(vector_scales[linear_group])
            * (d * (float)scale * (float)dot
               - dmin * (float)minimum * (float)total);
    return partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(COLIBRI_Q8_MATVEC(q4k_q8_matvec_transposed_warp, q4k_q8_group, 144)
COLIBRI_Q8_LM_HEAD(q4k_q8_lm_head_argmax_warp, q4k_q8_group, 144)

// q4k_q8_group with the dot lifted out, for the batched prompt kernels. A
// 32-value group is exactly one Q4_K sub-block, so both halves share the
// scale and the minimum.
__device__ __forceinline__ void q4k_q8_decode(
    const unsigned char* row_data, const int linear_group, int* words,
    float* scale_low, float* scale_high,
    float* offset_low, float* offset_high) {
    const int block = linear_group >> 3;
    const int sub_block = linear_group & 7;
    const unsigned char* base = row_data + block * 144;
    int scale, minimum;
    q5k_scale_min(base + 4, sub_block, &scale, &minimum);
    const unsigned char* quants = base + 16 + (sub_block >> 1) * 32;
    const int shift = (sub_block & 1) * 4;
    const int4* quant_vectors = (const int4*)quants;
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int4 packed_quads = quant_vectors[part];
        const int quads[4] = {packed_quads.x, packed_quads.y,
                              packed_quads.z, packed_quads.w};
        #pragma unroll
        for (int step = 0; step < 4; ++step)
            words[part * 4 + step] =
                (int)(((unsigned int)quads[step] >> shift) & 0x0f0f0f0fu);
    }
    const float d = __half2float(*((const __half*)base));
    const float dmin = __half2float(*((const __half*)(base + 2)));
    *scale_low = *scale_high = d * (float)scale;
    *offset_low = *offset_high = dmin * (float)minimum;
}

COLIBRI_Q8_MATVEC_ROWS_MIN(q4k_q8_matvec_transposed_rows, q4k_q8_decode, 144)
COLIBRI_Q8_MMQ_MIN(q4k_q8_mmq, q4k_q8_decode, 144)


__device__ const unsigned int kIq3xxsGrid[256] = {
    67372036u, 67372052u, 67372068u, 67374092u, 67374108u, 67374142u, 67376132u, 67376148u,
    67378188u, 67380244u, 67386908u, 67386924u, 67896332u, 67896348u, 67898372u, 67898388u,
    67900428u, 67900460u, 67902468u, 67902484u, 67904524u, 67906596u, 67911172u, 68420612u,
    68420628u, 68420644u, 68422668u, 68424708u, 68424724u, 68426764u, 68426780u, 68426814u,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    68430860u, 68430910u, 68435500u, 68944908u, 68944958u, 68946948u, 68946964u, 68949036u,
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

// Decode IQ3_XXS against a Q8-blocked activation. Structurally the same as the
// IQ2_XXS kernel: one 32-value group is eight qs bytes plus a 4-byte aux word
// carrying the group scale and four 7-bit sign selectors. Each quad indexes two
// codebook entries -- four values wide here rather than eight -- so the eight
// products collapse to two DP4A instructions on sign-corrected bytes.
__device__ __forceinline__ float iq3xxs_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    float partial = 0.0f;

        const int block = linear_group >> 3;
        const int group = linear_group & 7;
        const unsigned char* base = row_data + block * 98;
        const unsigned char* quants = base + 2 + group * 8;
        unsigned int aux;
        memcpy(&aux, base + 66 + group * 4, 4);

        // A 98-byte super-block leaves qs 2-byte aligned, so the codebook
        // indices stay on byte loads; the activation block is 32-byte aligned.
        const int4* activation_vectors =
            (const int4*)(vector + linear_group * 32);
        const int4 activation_low = activation_vectors[0];
        const int4 activation_high = activation_vectors[1];
        const int acts[8] = {
            activation_low.x, activation_low.y,
            activation_low.z, activation_low.w,
            activation_high.x, activation_high.y,
            activation_high.z, activation_high.w};

        int dot = 0;
        #pragma unroll
        for (int quad = 0; quad < 4; ++quad) {
            const unsigned int signs =
                (unsigned int)kIq2xxsSigns[(aux >> (7 * quad)) & 127]
                * 0x01010101u;
            const unsigned int pattern_first =
                kIq3xxsGrid[quants[quad * 2]];
            const unsigned int pattern_second =
                kIq3xxsGrid[quants[quad * 2 + 1]];
            const int masks_first = __vcmpne4(signs & 0x08040201u, 0);
            const int masks_second = __vcmpne4(signs & 0x80402010u, 0);
            const int weights_first =
                __vsub4((int)pattern_first ^ masks_first, masks_first);
            const int weights_second =
                __vsub4((int)pattern_second ^ masks_second, masks_second);
            dot = __dp4a(weights_first, acts[quad * 2], dot);
            dot = __dp4a(weights_second, acts[quad * 2 + 1], dot);
        }
        const float scale = __half2float(*((const __half*)base))
            * (0.5f + (float)(aux >> 28)) * 0.5f;
        partial += (float)dot * scale
            * __half2float(vector_scales[linear_group]);
    return partial;
}

COLIBRI_Q8_MATVEC(iq3xxs_q8_matvec_transposed_warp, iq3xxs_q8_group, 98)
COLIBRI_Q8_LM_HEAD(iq3xxs_q8_lm_head_argmax_warp, iq3xxs_q8_group, 98)
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Batched twin of iq3xxs_q8_group. One scale covers the whole 32-value group
// here, so both halves get it.
__device__ __forceinline__ void iq3xxs_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 98;
    const unsigned char* quants = base + 2 + group * 8;
    unsigned int aux;
    memcpy(&aux, base + 66 + group * 4, 4);
    #pragma unroll
    for (int quad = 0; quad < 4; ++quad) {
        const unsigned int signs =
            (unsigned int)kIq2xxsSigns[(aux >> (7 * quad)) & 127] * 0x01010101u;
        const unsigned int pattern_first = kIq3xxsGrid[quants[quad * 2]];
        const unsigned int pattern_second = kIq3xxsGrid[quants[quad * 2 + 1]];
        const int masks_first = __vcmpne4(signs & 0x08040201u, 0);
        const int masks_second = __vcmpne4(signs & 0x80402010u, 0);
        words[quad * 2] =
            __vsub4((int)pattern_first ^ masks_first, masks_first);
        words[quad * 2 + 1] =
            __vsub4((int)pattern_second ^ masks_second, masks_second);
    }
    const float scale = __half2float(*((const __half*)base))
        * (0.5f + (float)(aux >> 28)) * 0.5f;
    *scale_low = scale;
    *scale_high = scale;
}

COLIBRI_Q8_MATVEC_ROWS(iq3xxs_q8_matvec_transposed_rows, iq3xxs_q8_decode, 98)
COLIBRI_Q8_MATMUL_TILED(iq3xxs_q8_matmul_tiled, iq3xxs_q8_decode, 98)
COLIBRI_Q8_MMQ(iq3xxs_q8_mmq, iq3xxs_q8_decode, 98)



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

__device__ const unsigned long long kIq2sGrid[1024] = {
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    583545013515327752ULL, 583545013516437512ULL, 583545013517626137ULL, 583545013819607083ULL,
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
    1803738964258658568ULL, 1803738964541573128ULL, 1803738964541573163ULL, 1803738964541577497ULL,
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    3110607490504136968ULL, 3110607562929080328ULL, 3110607562930203417ULL, 3110607640524818457ULL,
    3110627281123945259ULL, 3110627281126238984ULL, 3110627281713432619ULL, 3110627354424711432ULL,
    3110627354725587243ULL, 3110627431447800584ULL, 3110627431447800619ULL, 3110627431450085384ULL,
    3110627431450085419ULL, 3110627431450094344ULL, 3110627432035003144ULL, 3110627432037296939ULL,
};
__device__ const unsigned int kIq3sGrid[512] = {
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
    151980295u, 151980803u, 184615173u, 184615681u, 184615689u, 184616197u, 184617217u, 184617225u,
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// IQ2_S against a vector quantized in independent 32-value Q8 blocks: the
// DP4A twin of iq2xxs_q8_group above. The codebook magnitudes fit in signed
// bytes, so the 32 products of one Q8 block collapse into eight __dp4a
// instructions instead of 32 weight reconstructions.
//
// Two things differ from IQ2_XXS. The grid index takes two high bits from qh,
// and the signs are stored literally rather than through the 7-bit sign
// codebook -- broadcasting the sign byte across all four lanes reproduces
// exactly the mask layout the shared __vcmpne4 trick expects.
//
// The subtlety is the scale. A Q8 block spans 32 values, which is two of
// IQ2_S's 16-value scale groups, so the two halves have to accumulate
// separately and take their own nibble of the scale byte. Folding them into
// one dot product would apply one half's scale to all 32 weights.
__device__ __forceinline__ float iq2s_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 82;
    const unsigned char* quants = base + 2;
    const unsigned char* signs = base + 34;
    const unsigned char* high = base + 66;
    const unsigned char* scales = base + 74;

    // 82 bytes leaves the activation block 32-byte aligned, so it loads as two
    // int4 rather than eight scalars.
    const int4* activation_vectors = (const int4*)(vector + linear_group * 32);
    const int4 activation_low = activation_vectors[0];
    const int4 activation_high = activation_vectors[1];
    const int acts[8] = {
        activation_low.x, activation_low.y,
        activation_low.z, activation_low.w,
        activation_high.x, activation_high.y,
        activation_high.z, activation_high.w};

    // The four grid indices this block covers share one qh byte and one scale
    // byte, both selected by the group alone.
    const int first_index = group * 4;
    const unsigned int qh_byte = high[group];
    const unsigned int scale_byte = scales[group];

    int dot_low = 0, dot_high = 0;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int index = first_index + step;
        const int entry =
            quants[index] | (int)(((qh_byte >> (2 * step)) & 3u) << 8);
        const unsigned long long pattern = kIq2sGrid[entry];
        const unsigned int sign_word = (unsigned int)signs[index] * 0x01010101u;
        const int masks_first = __vcmpne4(sign_word & 0x08040201u, 0);
        const int masks_second = __vcmpne4(sign_word & 0x80402010u, 0);
        const int weights_first = __vsub4(
            (int)(unsigned int)pattern ^ masks_first, masks_first);
        const int weights_second = __vsub4(
            (int)(unsigned int)(pattern >> 32) ^ masks_second, masks_second);
        int dot = 0;
        dot = __dp4a(weights_first, acts[step * 2], dot);
        dot = __dp4a(weights_second, acts[step * 2 + 1], dot);
        if (step < 2) dot_low += dot; else dot_high += dot;
    }
    const float scale_low = 0.5f + (float)(scale_byte & 15u);
    const float scale_high = 0.5f + (float)((scale_byte >> 4) & 15u);
    return ((float)dot_low * scale_low + (float)dot_high * scale_high) * 0.25f
        * __half2float(*((const __half*)base))
        * __half2float(vector_scales[linear_group]);
}

COLIBRI_Q8_MATVEC(iq2s_q8_matvec_transposed_warp, iq2s_q8_group, 82)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Batched twin of iq2s_q8_group: decodes the group without applying the scales,
// which the caller needs held back because the two 16-value halves take
// different nibbles of the scale byte.
__device__ __forceinline__ void iq2s_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 82;
    const unsigned char* quants = base + 2;
    const unsigned char* signs = base + 34;
    const unsigned char* high = base + 66;
    const unsigned char* scales = base + 74;
    const int first_index = group * 4;
    const unsigned int qh_byte = high[group];
    const unsigned int scale_byte = scales[group];
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int index = first_index + step;
        const int entry =
            quants[index] | (int)(((qh_byte >> (2 * step)) & 3u) << 8);
        const unsigned long long pattern = kIq2sGrid[entry];
        const unsigned int sign_word = (unsigned int)signs[index] * 0x01010101u;
        const int masks_first = __vcmpne4(sign_word & 0x08040201u, 0);
        const int masks_second = __vcmpne4(sign_word & 0x80402010u, 0);
        words[step * 2] = __vsub4(
            (int)(unsigned int)pattern ^ masks_first, masks_first);
        words[step * 2 + 1] = __vsub4(
            (int)(unsigned int)(pattern >> 32) ^ masks_second, masks_second);
    }
    const float block_scale = __half2float(*((const __half*)base)) * 0.25f;
    *scale_low = (0.5f + (float)(scale_byte & 15u)) * block_scale;
    *scale_high = (0.5f + (float)((scale_byte >> 4) & 15u)) * block_scale;
}

COLIBRI_Q8_MATVEC_ROWS(iq2s_q8_matvec_transposed_rows, iq2s_q8_decode, 82)
COLIBRI_Q8_MATMUL_TILED(iq2s_q8_matmul_tiled, iq2s_q8_decode, 82)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
COLIBRI_Q8_MMQ(iq2s_q8_mmq, iq2s_q8_decode, 82)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
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

__device__ const unsigned long long kIq2xsGrid[512] = {
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    588611563399219208ULL, 588611640120322824ULL, 588611640122607624ULL, 588611640707516459ULL,
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
__device__ const signed char kIq4nlValues[16] = {
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
R"COLIBRI_CUDA(
// IQ4_XS against a Q8-quantized activation.
//
// This format lines up with Q8 better than any other codebook type here: its
// 6-bit scale covers exactly 32 values, which is exactly one Q8 block, so a
// group needs one scale on each side and no split into halves. What it cannot
// do is feed __dp4a raw -- the 4-bit codes are indices into the non-uniform
// IQ4_NL levels, not values -- so the codes go through kIq4nlValues first and
// the dot runs on the reconstructed int8.
//
// Byte j of a sub-block holds element j in its low nibble and element j+16 in
// its high one, which is why the two nibble halves land four words apart.
__device__ __forceinline__ float iq4xs_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    const int block = linear_group >> 3;
    const int sub = linear_group & 7;
    const unsigned char* base = row_data + block * 136;
    unsigned int codes[4];
    memcpy(codes, base + 8 + sub * 16, 16);
    const signed char* activations = vector + linear_group * 32;
    const int4* activation_vectors = (const int4*)activations;
    const int4 activation_low = activation_vectors[0];
    const int4 activation_high = activation_vectors[1];
    const int acts[8] = {
        activation_low.x, activation_low.y, activation_low.z, activation_low.w,
        activation_high.x, activation_high.y, activation_high.z,
        activation_high.w};

    int dot = 0;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const unsigned int word = codes[step];
        const int low_weights =
            ((int)(unsigned char)kIq4nlValues[(word >> 0) & 15])
            | ((int)(unsigned char)kIq4nlValues[(word >> 8) & 15] << 8)
            | ((int)(unsigned char)kIq4nlValues[(word >> 16) & 15] << 16)
            | ((int)(unsigned char)kIq4nlValues[(word >> 24) & 15] << 24);
        const int high_weights =
            ((int)(unsigned char)kIq4nlValues[(word >> 4) & 15])
            | ((int)(unsigned char)kIq4nlValues[(word >> 12) & 15] << 8)
            | ((int)(unsigned char)kIq4nlValues[(word >> 20) & 15] << 16)
            | ((int)(unsigned char)kIq4nlValues[(word >> 28) & 15] << 24);
        dot = __dp4a(low_weights, acts[step], dot);
        dot = __dp4a(high_weights, acts[step + 4], dot);
    }

    unsigned short scales_high;
    memcpy(&scales_high, base + 2, 2);
    const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
    const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
    const float d = __half2float(*((const __half*)base));
    return __half2float(vector_scales[linear_group]) * d * (float)scale *
           (float)dot;
}

COLIBRI_Q8_MATVEC(iq4xs_q8_matvec_transposed_warp, iq4xs_q8_group, 136)

// Batched twin. One scale for the whole group, so both halves get the same one.
__device__ __forceinline__ void iq4xs_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int sub = linear_group & 7;
    const unsigned char* base = row_data + block * 136;
    unsigned int codes[4];
    memcpy(codes, base + 8 + sub * 16, 16);
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const unsigned int word = codes[step];
        words[step] =
            ((int)(unsigned char)kIq4nlValues[(word >> 0) & 15])
            | ((int)(unsigned char)kIq4nlValues[(word >> 8) & 15] << 8)
            | ((int)(unsigned char)kIq4nlValues[(word >> 16) & 15] << 16)
            | ((int)(unsigned char)kIq4nlValues[(word >> 24) & 15] << 24);
        words[step + 4] =
            ((int)(unsigned char)kIq4nlValues[(word >> 4) & 15])
            | ((int)(unsigned char)kIq4nlValues[(word >> 12) & 15] << 8)
            | ((int)(unsigned char)kIq4nlValues[(word >> 20) & 15] << 16)
            | ((int)(unsigned char)kIq4nlValues[(word >> 28) & 15] << 24);
    }
    unsigned short scales_high;
    memcpy(&scales_high, base + 2, 2);
    const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
    const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
    const float combined = __half2float(*((const __half*)base)) * (float)scale;
    *scale_low = combined;
    *scale_high = combined;
}

COLIBRI_Q8_MATVEC_ROWS(iq4xs_q8_matvec_transposed_rows, iq4xs_q8_decode, 136)
COLIBRI_Q8_MATMUL_TILED(iq4xs_q8_matmul_tiled, iq4xs_q8_decode, 136)
COLIBRI_Q8_MMQ(iq4xs_q8_mmq, iq4xs_q8_decode, 136)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(

extern "C" __global__
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// IQ1_M shares IQ1_S's codebook: 2048 entries, each eight signed bytes packed
// into a 64-bit word. At 16 KiB this is too large for __constant__ memory and
// is indexed unpredictably (an 11-bit code per group of eight weights), so it
// lives in global memory and rides the L2 like the other IQ grids here.
__device__ const unsigned long long kIq1sGrid[2048] = {
    0xffffffffffffffffULL, 0xffffffffffffff01ULL, 0xffffffffffff0000ULL, 0xffffffffffff01ffULL,
    0xffffffffffff0101ULL, 0xffffffffff00ff00ULL, 0xffffffffff000000ULL, 0xffffffffff01ffffULL,
    0xffffffffff01ff01ULL, 0xffffffffff0101ffULL, 0xffffffffff010101ULL, 0xffffffff00ff0000ULL,
    0xffffffff0000ff00ULL, 0xffffffff000000ffULL, 0xffffffff00000001ULL, 0xffffffff00010000ULL,
    0xffffffff01ffffffULL, 0xffffffff01ffff01ULL, 0xffffffff01ff01ffULL, 0xffffffff01ff0101ULL,
    0xffffffff01000000ULL, 0xffffffff0101ffffULL, 0xffffffff0101ff01ULL, 0xffffffff010101ffULL,
    0xffffffff01010101ULL, 0xffffff00ffff00ffULL, 0xffffff00ffff0000ULL, 0xffffff00ff00ff00ULL,
    0xffffff00ff0000ffULL, 0xffffff00ff000001ULL, 0xffffff00ff000100ULL, 0xffffff00ff000101ULL,
    0xffffff00ff010000ULL, 0xffffff0000ffff00ULL, 0xffffff0000ff0001ULL, 0xffffff0000ff0100ULL,
    0xffffff000000ff01ULL, 0xffffff0000000000ULL, 0xffffff0000000101ULL, 0xffffff000001ff00ULL,
    0xffffff00000100ffULL, 0xffffff0000010001ULL, 0xffffff00000101ffULL, 0xffffff0001ff0000ULL,
    0xffffff000100ff00ULL, 0xffffff00010000ffULL, 0xffffff0001000001ULL, 0xffffff0001010000ULL,
    0xffffff01ffffffffULL, 0xffffff01ffffff01ULL, 0xffffff01ffff01ffULL, 0xffffff01ffff0101ULL,
    0xffffff01ff000000ULL, 0xffffff01ff01ffffULL, 0xffffff01ff01ff01ULL, 0xffffff01ff0101ffULL,
    0xffffff01ff010101ULL, 0xffffff0100ff0000ULL, 0xffffff010000ff00ULL, 0xffffff0100000100ULL,
    0xffffff01000100ffULL, 0xffffff0100010100ULL, 0xffffff0101ffffffULL, 0xffffff0101ffff01ULL,
    0xffffff0101ff01ffULL, 0xffffff0101ff0101ULL, 0xffffff010100ff00ULL, 0xffffff0101000000ULL,
    0xffffff0101000100ULL, 0xffffff010101ffffULL, 0xffffff010101ff01ULL, 0xffffff01010101ffULL,
    0xffffff0101010101ULL, 0xffff00ffff00ff00ULL, 0xffff00ffff0000ffULL, 0xffff00ffff000001ULL,
    0xffff00ffff010000ULL, 0xffff00ff00ffff00ULL, 0xffff00ff00ff0100ULL, 0xffff00ff00000000ULL,
    0xffff00ff00000101ULL, 0xffff00ff000100ffULL, 0xffff00ff00010000ULL, 0xffff00ff0100ff00ULL,
    0xffff00ff01000100ULL, 0xffff00ff01010000ULL, 0xffff0000ffffff00ULL, 0xffff0000ffff00ffULL,
    0xffff0000ffff0000ULL, 0xffff0000ffff0001ULL, 0xffff0000ff000000ULL, 0xffff0000ff0001ffULL,
    0xffff0000ff000101ULL, 0xffff0000ff010100ULL, 0xffff000000ffffffULL, 0xffff000000ff0000ULL,
    0xffff000000ff0101ULL, 0xffff00000000ffffULL, 0xffff00000000ff00ULL, 0xffff0000000000ffULL,
    0xffff000000000000ULL, 0xffff000000000001ULL, 0xffff000000000100ULL, 0xffff00000001ffffULL,
    0xffff00000001ff01ULL, 0xffff000000010000ULL, 0xffff0000000101ffULL, 0xffff000000010101ULL,
    0xffff000001ffff00ULL, 0xffff00000100ff00ULL, 0xffff000001000000ULL, 0xffff0000010001ffULL,
    0xffff000001000101ULL, 0xffff00000101ff00ULL, 0xffff0000010100ffULL, 0xffff000001010000ULL,
    0xffff000001010001ULL, 0xffff000001010100ULL, 0xffff0001ff0000ffULL, 0xffff0001ff000100ULL,
    0xffff000100ffff00ULL, 0xffff000100ff00ffULL, 0xffff00010000ffffULL, 0xffff00010000ff01ULL,
    0xffff000100000000ULL, 0xffff0001000001ffULL, 0xffff00010001ffffULL, 0xffff00010001ff00ULL,
    0xffff000100010001ULL, 0xffff000100010100ULL, 0xffff000101ff0000ULL, 0xffff00010100ff00ULL,
    0xffff0001010000ffULL, 0xffff000101000100ULL, 0xffff01ffffffffffULL, 0xffff01ffffffff01ULL,
    0xffff01ffffff01ffULL, 0xffff01ffffff0101ULL, 0xffff01ffff000000ULL, 0xffff01ffff01ffffULL,
    0xffff01ffff01ff01ULL, 0xffff01ffff0101ffULL, 0xffff01ffff010101ULL, 0xffff01ff00ff0000ULL,
    0xffff01ff0000ff00ULL, 0xffff01ff00000001ULL, 0xffff01ff00010000ULL, 0xffff01ff01ffffffULL,
    0xffff01ff01ffff01ULL, 0xffff01ff01ff01ffULL, 0xffff01ff01ff0101ULL, 0xffff01ff01000000ULL,
    0xffff01ff0101ffffULL, 0xffff01ff0101ff01ULL, 0xffff01ff010101ffULL, 0xffff01ff01010101ULL,
    0xffff0100ffff0000ULL, 0xffff0100ff00ff00ULL, 0xffff0100ff0000ffULL, 0xffff0100ff000100ULL,
    0xffff0100ff0100ffULL, 0xffff0100ff010000ULL, 0xffff010000ffff00ULL, 0xffff01000000ffffULL,
    0xffff01000000ff00ULL, 0xffff010000000000ULL, 0xffff01000001ff00ULL, 0xffff0100000100ffULL,
    0xffff010000010100ULL, 0xffff01000100ff00ULL, 0xffff0100010000ffULL, 0xffff010001000001ULL,
    0xffff010001000100ULL, 0xffff010001010000ULL, 0xffff0101ffffffffULL, 0xffff0101ffffff01ULL,
    0xffff0101ffff01ffULL, 0xffff0101ffff0101ULL, 0xffff0101ff000000ULL, 0xffff0101ff01ffffULL,
    0xffff0101ff01ff01ULL, 0xffff0101ff0101ffULL, 0xffff0101ff010101ULL, 0xffff010100ff0000ULL,
    0xffff01010000ff00ULL, 0xffff010100000100ULL, 0xffff01010001ff00ULL, 0xffff010100010000ULL,
    0xffff010101ffffffULL, 0xffff010101ffff01ULL, 0xffff010101ff0000ULL, 0xffff010101ff01ffULL,
    0xffff010101ff0101ULL, 0xffff010101000000ULL, 0xffff01010101ffffULL, 0xffff01010101ff01ULL,
    0xffff0101010101ffULL, 0xffff010101010101ULL, 0xff00ffffff00ffffULL, 0xff00ffffff00ff00ULL,
    0xff00ffffff0000ffULL, 0xff00ffffff000100ULL, 0xff00ffffff0100ffULL, 0xff00ffffff010000ULL,
    0xff00ffff00ffff00ULL, 0xff00ffff00ff00ffULL, 0xff00ffff0000ffffULL, 0xff00ffff00000000ULL,
    0xff00ffff000001ffULL, 0xff00ffff0001ff00ULL, 0xff00ffff000100ffULL, 0xff00ffff00010000ULL,
    0xff00ffff00010100ULL, 0xff00ffff0100ff00ULL, 0xff00ffff010000ffULL, 0xff00ffff01000001ULL,
    0xff00ffff0101ff00ULL, 0xff00ffff01010000ULL, 0xff00ff00ffffff00ULL, 0xff00ff00ffff00ffULL,
    0xff00ff00ffff0001ULL, 0xff00ff00ffff0100ULL, 0xff00ff00ff00ffffULL, 0xff00ff00ff00ff01ULL,
    0xff00ff00ff000000ULL, 0xff00ff00ff0001ffULL, 0xff00ff00ff01ff00ULL, 0xff00ff00ff0100ffULL,
    0xff00ff00ff010100ULL, 0xff00ff0000ff0000ULL, 0xff00ff0000ff0101ULL, 0xff00ff000000ffffULL,
    0xff00ff000000ff00ULL, 0xff00ff000000ff01ULL, 0xff00ff00000000ffULL, 0xff00ff0000000000ULL,
    0xff00ff0000000001ULL, 0xff00ff0000000100ULL, 0xff00ff000001ffffULL, 0xff00ff0000010000ULL,
    0xff00ff0001ff00ffULL, 0xff00ff000100ff01ULL, 0xff00ff0001000000ULL, 0xff00ff000101ff00ULL,
    0xff00ff00010100ffULL, 0xff00ff01ff00ff00ULL, 0xff00ff01ff0000ffULL, 0xff00ff01ff000001ULL,
    0xff00ff01ff010000ULL, 0xff00ff0100ffffffULL, 0xff00ff0100ff0001ULL, 0xff00ff0100ff0100ULL,
    0xff00ff010000ff01ULL, 0xff00ff0100000000ULL, 0xff00ff01000001ffULL, 0xff00ff0100000101ULL,
    0xff00ff01000100ffULL, 0xff00ff0100010001ULL, 0xff00ff0101ff0000ULL, 0xff00ff010100ff00ULL,
    0xff00ff01010000ffULL, 0xff00ff0101000001ULL, 0xff00ff0101010000ULL, 0xff0000ffffffff00ULL,
    0xff0000ffffff0001ULL, 0xff0000ffffff0100ULL, 0xff0000ffff0000ffULL, 0xff0000ffff000000ULL,
    0xff0000ffff0001ffULL, 0xff0000ffff000100ULL, 0xff0000ffff01ff00ULL, 0xff0000ffff010001ULL,
    0xff0000ff00ffff00ULL, 0xff0000ff00ff0000ULL, 0xff0000ff00ff0001ULL, 0xff0000ff00ff01ffULL,
    0xff0000ff00ff0101ULL, 0xff0000ff0000ff00ULL, 0xff0000ff000000ffULL, 0xff0000ff00000000ULL,
    0xff0000ff00000001ULL, 0xff0000ff00000100ULL, 0xff0000ff0001ff01ULL, 0xff0000ff00010000ULL,
    0xff0000ff000101ffULL, 0xff0000ff01ff00ffULL, 0xff0000ff01ff0100ULL, 0xff0000ff0100ffffULL,
    0xff0000ff010000ffULL, 0xff0000ff01000000ULL, 0xff0000ff010001ffULL, 0xff0000ff01000100ULL,
    0xff0000ff01000101ULL, 0xff0000ff0101ff00ULL, 0xff0000ff010100ffULL, 0xff0000ff01010000ULL,
    0xff0000ff01010100ULL, 0xff000000ffffff01ULL, 0xff000000ffff0000ULL, 0xff000000ffff0101ULL,
    0xff000000ff00ff00ULL, 0xff000000ff0000ffULL, 0xff000000ff000000ULL, 0xff000000ff000001ULL,
    0xff000000ff000100ULL, 0xff000000ff01ffffULL, 0xff000000ff01ff01ULL, 0xff000000ff010000ULL,
    0xff000000ff0101ffULL, 0xff000000ff010101ULL, 0xff00000000ffff00ULL, 0xff00000000ff00ffULL,
    0xff00000000ff0000ULL, 0xff00000000ff0001ULL, 0xff0000000000ff00ULL, 0xff0000000000ff01ULL,
    0xff000000000000ffULL, 0xff00000000000000ULL, 0xff00000000000001ULL, 0xff00000000000100ULL,
    0xff00000000000101ULL, 0xff0000000001ff00ULL, 0xff000000000100ffULL, 0xff00000000010000ULL,
    0xff00000000010001ULL, 0xff00000000010100ULL, 0xff00000001ffffffULL, 0xff00000001ffff01ULL,
    0xff00000001ff00ffULL, 0xff00000001ff0000ULL, 0xff00000001ff01ffULL, 0xff00000001ff0101ULL,
    0xff0000000100ffffULL, 0xff0000000100ff00ULL, 0xff000000010000ffULL, 0xff00000001000000ULL,
    0xff00000001000001ULL, 0xff00000001000100ULL, 0xff00000001000101ULL, 0xff0000000101ffffULL,
    0xff0000000101ff01ULL, 0xff00000001010000ULL, 0xff000001ffffff00ULL, 0xff000001ffff00ffULL,
    0xff000001ffff0000ULL, 0xff000001ffff0001ULL, 0xff000001ff000000ULL, 0xff000001ff000001ULL,
    0xff000001ff0001ffULL, 0xff000001ff000101ULL, 0xff000001ff01ff00ULL, 0xff000001ff010001ULL,
    0xff00000100ffffffULL, 0xff00000100ffff01ULL, 0xff00000100ff00ffULL, 0xff00000100ff0000ULL,
    0xff00000100ff01ffULL, 0xff00000100ff0101ULL, 0xff0000010000ff00ULL, 0xff00000100000000ULL,
    0xff00000100000001ULL, 0xff000001000001ffULL, 0xff00000100000100ULL, 0xff0000010001ff00ULL,
    0xff000001000100ffULL, 0xff00000100010000ULL, 0xff000001000101ffULL, 0xff00000100010100ULL,
    0xff00000100010101ULL, 0xff00000101ff0001ULL, 0xff00000101ff0101ULL, 0xff0000010100ff01ULL,
    0xff00000101000000ULL, 0xff000001010100ffULL, 0xff00000101010100ULL, 0xff0001ffff00ff00ULL,
    0xff0001ffff000001ULL, 0xff0001ffff010000ULL, 0xff0001ff00ffff00ULL, 0xff0001ff00ff00ffULL,
    0xff0001ff00ff0001ULL, 0xff0001ff00ff0100ULL, 0xff0001ff0000ffffULL, 0xff0001ff00000000ULL,
    0xff0001ff000001ffULL, 0xff0001ff00000101ULL, 0xff0001ff0001ffffULL, 0xff0001ff0001ff00ULL,
    0xff0001ff000100ffULL, 0xff0001ff00010001ULL, 0xff0001ff00010100ULL, 0xff0001ff01ff0000ULL,
    0xff0001ff0100ff00ULL, 0xff0001ff010000ffULL, 0xff0001ff01010000ULL, 0xff000100ff00ffffULL,
    0xff000100ff00ff01ULL, 0xff000100ff000000ULL, 0xff000100ff000101ULL, 0xff000100ff01ff00ULL,
    0xff000100ff010000ULL, 0xff00010000ffff01ULL, 0xff00010000ff00ffULL, 0xff00010000ff0000ULL,
    0xff00010000ff01ffULL, 0xff0001000000ff00ULL, 0xff000100000000ffULL, 0xff00010000000000ULL,
    0xff00010000000001ULL, 0xff00010000000100ULL, 0xff00010000000101ULL, 0xff0001000001ffffULL,
    0xff00010000010000ULL, 0xff00010000010101ULL, 0xff00010001ff0100ULL, 0xff0001000100ff00ULL,
    0xff0001000100ff01ULL, 0xff00010001000000ULL, 0xff000100010001ffULL, 0xff0001000101ff00ULL,
    0xff00010001010001ULL, 0xff00010001010100ULL, 0xff000101ffff0100ULL, 0xff000101ff000001ULL,
    0xff000101ff0100ffULL, 0xff000101ff010001ULL, 0xff00010100ff00ffULL, 0xff00010100ff0001ULL,
    0xff00010100ff0100ULL, 0xff0001010000ffffULL, 0xff0001010000ff01ULL, 0xff00010100000000ULL,
    0xff000101000001ffULL, 0xff0001010001ff00ULL, 0xff00010100010001ULL, 0xff00010100010100ULL,
    0xff00010101ff0000ULL, 0xff0001010100ff00ULL, 0xff00010101000001ULL, 0xff00010101000101ULL,
    0xff01ffffffffffffULL, 0xff01ffffffffff01ULL, 0xff01ffffffff01ffULL, 0xff01ffffffff0101ULL,
    0xff01ffffff000000ULL, 0xff01ffffff01ffffULL, 0xff01ffffff01ff01ULL, 0xff01ffffff010000ULL,
    0xff01ffffff0101ffULL, 0xff01ffffff010101ULL, 0xff01ffff00ff0000ULL, 0xff01ffff0000ff00ULL,
    0xff01ffff00000100ULL, 0xff01ffff0001ff00ULL, 0xff01ffff00010000ULL, 0xff01ffff01ffffffULL,
    0xff01ffff01ffff01ULL, 0xff01ffff01ff01ffULL, 0xff01ffff01ff0101ULL, 0xff01ffff01000000ULL,
    0xff01ffff0101ffffULL, 0xff01ffff0101ff01ULL, 0xff01ffff01010000ULL, 0xff01ffff010101ffULL,
    0xff01ffff01010101ULL, 0xff01ff00ffff0000ULL, 0xff01ff00ff00ff00ULL, 0xff01ff00ff0000ffULL,
    0xff01ff00ff000100ULL, 0xff01ff00ff010000ULL, 0xff01ff0000ffff01ULL, 0xff01ff0000ff00ffULL,
    0xff01ff0000ff0100ULL, 0xff01ff0000000000ULL, 0xff01ff00000001ffULL, 0xff01ff0000000101ULL,
    0xff01ff000001ff00ULL, 0xff01ff00000100ffULL, 0xff01ff0000010000ULL, 0xff01ff0000010001ULL,
    0xff01ff0001ff0000ULL, 0xff01ff000100ffffULL, 0xff01ff0001000001ULL, 0xff01ff0001000100ULL,
    0xff01ff0001010000ULL, 0xff01ff01ffffff00ULL, 0xff01ff01ffff01ffULL, 0xff01ff01ffff0101ULL,
    0xff01ff01ff00ff00ULL, 0xff01ff01ff000000ULL, 0xff01ff01ff01ffffULL, 0xff01ff01ff01ff01ULL,
    0xff01ff01ff0101ffULL, 0xff01ff01ff010101ULL, 0xff01ff0100ff0000ULL, 0xff01ff010000ff00ULL,
    0xff01ff0100000001ULL, 0xff01ff0100000100ULL, 0xff01ff0100010000ULL, 0xff01ff0101ffff00ULL,
    0xff01ff0101ff01ffULL, 0xff01ff0101ff0101ULL, 0xff01ff010100ff00ULL, 0xff01ff0101000000ULL,
    0xff01ff010101ffffULL, 0xff01ff010101ff01ULL, 0xff01ff01010101ffULL, 0xff01ff0101010101ULL,
    0xff0100ffffff0000ULL, 0xff0100ffff0000ffULL, 0xff0100ffff000001ULL, 0xff0100ffff000100ULL,
    0xff0100ffff010000ULL, 0xff0100ff00ff00ffULL, 0xff0100ff00ff0000ULL, 0xff0100ff00ff0001ULL,
    0xff0100ff00ff0100ULL, 0xff0100ff0000ff01ULL, 0xff0100ff00000000ULL, 0xff0100ff000001ffULL,
    0xff0100ff00000101ULL, 0xff0100ff00010001ULL, 0xff0100ff01ff0000ULL, 0xff0100ff0100ff00ULL,
    0xff0100ff010000ffULL, 0xff0100ff01000100ULL, 0xff0100ff0101ff00ULL, 0xff0100ff01010000ULL,
    0xff010000ffff0100ULL, 0xff010000ff000000ULL, 0xff010000ff01ff00ULL, 0xff010000ff010100ULL,
    0xff01000000ffffffULL, 0xff01000000ff0000ULL, 0xff01000000ff01ffULL, 0xff0100000000ff00ULL,
    0xff010000000000ffULL, 0xff01000000000000ULL, 0xff01000000000100ULL, 0xff0100000001ff01ULL,
    0xff01000000010000ULL, 0xff010000000101ffULL, 0xff01000001ff0100ULL, 0xff0100000100ffffULL,
    0xff010000010000ffULL, 0xff01000001000000ULL, 0xff010000010001ffULL, 0xff01000001000101ULL,
    0xff0100000101ff00ULL, 0xff010000010100ffULL, 0xff01000001010001ULL, 0xff01000001010100ULL,
    0xff010001ffff0000ULL, 0xff010001ff00ffffULL, 0xff010001ff00ff01ULL, 0xff010001ff000100ULL,
    0xff010001ff010000ULL, 0xff01000100ffff00ULL, 0xff01000100ff0100ULL, 0xff01000100000000ULL,
    0xff0100010001ffffULL, 0xff0100010001ff00ULL, 0xff01000100010100ULL, 0xff01000101ff00ffULL,
    0xff01000101ff0001ULL, 0xff0100010100ffffULL, 0xff01000101000101ULL, 0xff0101ffffffffffULL,
    0xff0101ffffffff01ULL, 0xff0101ffffff01ffULL, 0xff0101ffffff0101ULL, 0xff0101ffff000000ULL,
    0xff0101ffff01ffffULL, 0xff0101ffff01ff01ULL, 0xff0101ffff0101ffULL, 0xff0101ffff010101ULL,
    0xff0101ff00ff0000ULL, 0xff0101ff0000ff00ULL, 0xff0101ff000000ffULL, 0xff0101ff00010000ULL,
    0xff0101ff01ffffffULL, 0xff0101ff01ffff01ULL, 0xff0101ff01ff01ffULL, 0xff0101ff01ff0101ULL,
    0xff0101ff0101ffffULL, 0xff0101ff0101ff01ULL, 0xff0101ff010101ffULL, 0xff0101ff01010101ULL,
    0xff010100ffff0100ULL, 0xff010100ff00ff00ULL, 0xff010100ff0000ffULL, 0xff010100ff000100ULL,
    0xff010100ff010000ULL, 0xff01010000ff0001ULL, 0xff01010000ff0100ULL, 0xff0101000000ff01ULL,
    0xff01010000000000ULL, 0xff0101000001ff00ULL, 0xff010100000100ffULL, 0xff01010000010001ULL,
    0xff01010000010100ULL, 0xff01010001ff0000ULL, 0xff0101000100ffffULL, 0xff01010001000001ULL,
    0xff01010001000100ULL, 0xff010100010100ffULL, 0xff01010001010000ULL, 0xff010101ffffffffULL,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
    0xff010101ffffff01ULL, 0xff010101ffff01ffULL, 0xff010101ffff0101ULL, 0xff010101ff01ffffULL,
    0xff010101ff01ff01ULL, 0xff010101ff0101ffULL, 0xff010101ff010101ULL, 0xff01010100ff0000ULL,
    0xff0101010000ff00ULL, 0xff01010100000001ULL, 0xff01010100000100ULL, 0xff01010100010000ULL,
    0xff01010101ffffffULL, 0xff01010101ffff01ULL, 0xff01010101ff01ffULL, 0xff01010101ff0101ULL,
    0xff01010101000000ULL, 0xff0101010101ffffULL, 0xff0101010101ff01ULL, 0xff010101010101ffULL,
    0xff01010101010101ULL, 0x00ffffffffff0000ULL, 0x00ffffffff00ff00ULL, 0x00ffffffff000001ULL,
    0x00ffffffff010000ULL, 0x00ffffff00ff0100ULL, 0x00ffffff0000ff01ULL, 0x00ffffff00000000ULL,
    0x00ffffff000001ffULL, 0x00ffffff00000101ULL, 0x00ffffff0001ff00ULL, 0x00ffffff000100ffULL,
    0x00ffffff00010001ULL, 0x00ffffff010000ffULL, 0x00ffffff01000100ULL, 0x00ffffff0101ff00ULL,
    0x00ffffff01010001ULL, 0x00ffff00ffffffffULL, 0x00ffff00ffffff00ULL, 0x00ffff00ffff00ffULL,
    0x00ffff00ffff0001ULL, 0x00ffff00ffff0100ULL, 0x00ffff00ff00ff01ULL, 0x00ffff00ff000000ULL,
    0x00ffff00ff000001ULL, 0x00ffff00ff0001ffULL, 0x00ffff00ff000101ULL, 0x00ffff00ff01ff00ULL,
    0x00ffff00ff010001ULL, 0x00ffff00ff010100ULL, 0x00ffff0000ff0000ULL, 0x00ffff0000ff01ffULL,
    0x00ffff0000ff0101ULL, 0x00ffff000000ff00ULL, 0x00ffff00000000ffULL, 0x00ffff0000000000ULL,
    0x00ffff0000000001ULL, 0x00ffff0000000100ULL, 0x00ffff0000000101ULL, 0x00ffff0000010000ULL,
    0x00ffff00000101ffULL, 0x00ffff0000010101ULL, 0x00ffff0001ffff00ULL, 0x00ffff0001ff00ffULL,
    0x00ffff0001ff0001ULL, 0x00ffff000100ffffULL, 0x00ffff000100ff01ULL, 0x00ffff0001000000ULL,
    0x00ffff000101ffffULL, 0x00ffff000101ff00ULL, 0x00ffff000101ff01ULL, 0x00ffff01ffff0000ULL,
    0x00ffff01ff00ff00ULL, 0x00ffff01ff0000ffULL, 0x00ffff01ff000001ULL, 0x00ffff01ff010000ULL,
    0x00ffff0100ffff00ULL, 0x00ffff010000ff01ULL, 0x00ffff0100000000ULL, 0x00ffff0100000101ULL,
    0x00ffff01000100ffULL, 0x00ffff0100010100ULL, 0x00ffff0101ff0100ULL, 0x00ffff01010000ffULL,
    0x00ffff0101010000ULL, 0x00ff00ffffffff00ULL, 0x00ff00ffff000000ULL, 0x00ff00ffff000100ULL,
    0x00ff00ffff010100ULL, 0x00ff00ff00ff0000ULL, 0x00ff00ff00ff01ffULL, 0x00ff00ff00ff0101ULL,
    0x00ff00ff0000ff00ULL, 0x00ff00ff000000ffULL, 0x00ff00ff00000000ULL, 0x00ff00ff00000001ULL,
    0x00ff00ff0001ff00ULL, 0x00ff00ff0001ff01ULL, 0x00ff00ff00010000ULL, 0x00ff00ff000101ffULL,
    0x00ff00ff00010101ULL, 0x00ff00ff01ffff00ULL, 0x00ff00ff01ff0001ULL, 0x00ff00ff01ff0100ULL,
    0x00ff00ff0100ffffULL, 0x00ff00ff0100ff01ULL, 0x00ff00ff01000000ULL, 0x00ff00ff0101ffffULL,
    0x00ff00ff0101ff00ULL, 0x00ff00ff01010100ULL, 0x00ff0000ffffff00ULL, 0x00ff0000ffffff01ULL,
    0x00ff0000ffff0000ULL, 0x00ff0000ffff0101ULL, 0x00ff0000ff00ff00ULL, 0x00ff0000ff0000ffULL,
    0x00ff0000ff000000ULL, 0x00ff0000ff000001ULL, 0x00ff0000ff000100ULL, 0x00ff0000ff01ffffULL,
    0x00ff0000ff010000ULL, 0x00ff0000ff010101ULL, 0x00ff000000ffff00ULL, 0x00ff000000ff00ffULL,
    0x00ff000000ff0000ULL, 0x00ff000000ff0001ULL, 0x00ff000000ff0100ULL, 0x00ff00000000ffffULL,
    0x00ff00000000ff00ULL, 0x00ff0000000000ffULL, 0x00ff000000000000ULL, 0x00ff000000000001ULL,
    0x00ff0000000001ffULL, 0x00ff000000000100ULL, 0x00ff00000001ff00ULL, 0x00ff0000000100ffULL,
    0x00ff000000010000ULL, 0x00ff000000010001ULL, 0x00ff000000010100ULL, 0x00ff000001ffff01ULL,
    0x00ff000001ff00ffULL, 0x00ff000001ff0000ULL, 0x00ff000001ff01ffULL, 0x00ff00000100ff00ULL,
    0x00ff0000010000ffULL, 0x00ff000001000000ULL, 0x00ff000001000001ULL, 0x00ff000001000100ULL,
    0x00ff000001000101ULL, 0x00ff000001010000ULL, 0x00ff0000010101ffULL, 0x00ff000001010101ULL,
    0x00ff0001ffffff00ULL, 0x00ff0001ffff0000ULL, 0x00ff0001ffff0100ULL, 0x00ff0001ff0000ffULL,
    0x00ff0001ff000000ULL, 0x00ff0001ff0001ffULL, 0x00ff0001ff000101ULL, 0x00ff0001ff01ff00ULL,
    0x00ff0001ff0100ffULL, 0x00ff0001ff010100ULL, 0x00ff000100ffffffULL, 0x00ff000100ffff01ULL,
    0x00ff000100ff0000ULL, 0x00ff000100ff01ffULL, 0x00ff00010000ffffULL, 0x00ff00010000ff00ULL,
    0x00ff00010000ff01ULL, 0x00ff000100000000ULL, 0x00ff000100000001ULL, 0x00ff000100000100ULL,
    0x00ff00010001ff01ULL, 0x00ff000100010000ULL, 0x00ff0001000101ffULL, 0x00ff000101ffff00ULL,
    0x00ff000101ff0000ULL, 0x00ff000101ff0101ULL, 0x00ff0001010000ffULL, 0x00ff000101000000ULL,
    0x00ff00010101ff00ULL, 0x00ff0001010100ffULL, 0x00ff000101010001ULL, 0x00ff01ffffff0000ULL,
    0x00ff01ffff00ff00ULL, 0x00ff01ffff000000ULL, 0x00ff01ffff000101ULL, 0x00ff01ffff010000ULL,
    0x00ff01ff00ffff01ULL, 0x00ff01ff00ff0100ULL, 0x00ff01ff0000ffffULL, 0x00ff01ff00000000ULL,
    0x00ff01ff000001ffULL, 0x00ff01ff0001ff00ULL, 0x00ff01ff000100ffULL, 0x00ff01ff00010001ULL,
    0x00ff01ff00010100ULL, 0x00ff01ff01ff0000ULL, 0x00ff01ff0100ff00ULL, 0x00ff01ff010000ffULL,
    0x00ff01ff01000001ULL, 0x00ff01ff01000100ULL, 0x00ff01ff01010000ULL, 0x00ff0100ffffff00ULL,
    0x00ff0100ffff0000ULL, 0x00ff0100ffff0001ULL, 0x00ff0100ffff0101ULL, 0x00ff0100ff00ffffULL,
    0x00ff0100ff0000ffULL, 0x00ff0100ff000000ULL, 0x00ff0100ff0001ffULL, 0x00ff0100ff01ff00ULL,
    0x00ff0100ff0100ffULL, 0x00ff0100ff010001ULL, 0x00ff010000ffffffULL, 0x00ff010000ff0000ULL,
    0x00ff010000ff0101ULL, 0x00ff01000000ff00ULL, 0x00ff01000000ff01ULL, 0x00ff0100000000ffULL,
    0x00ff010000000000ULL, 0x00ff010000000001ULL, 0x00ff010000000100ULL, 0x00ff01000001ffffULL,
    0x00ff01000001ff01ULL, 0x00ff010000010000ULL, 0x00ff010000010001ULL, 0x00ff010000010101ULL,
    0x00ff010001ff0001ULL, 0x00ff010001ff0100ULL, 0x00ff01000100ff01ULL, 0x00ff010001000000ULL,
    0x00ff010001000001ULL, 0x00ff0100010001ffULL, 0x00ff01000101ff00ULL, 0x00ff0100010100ffULL,
    0x00ff010001010001ULL, 0x00ff010001010100ULL, 0x00ff0101ff000001ULL, 0x00ff010100ff00ffULL,
    0x00ff010100ff0001ULL, 0x00ff010100ff0100ULL, 0x00ff010100000000ULL, 0x00ff0101000001ffULL,
    0x00ff010100000101ULL, 0x00ff0101000100ffULL, 0x00ff010100010100ULL, 0x00ff0101010000ffULL,
    0x00ff010101010000ULL, 0x0000ffffffffff00ULL, 0x0000ffffffff00ffULL, 0x0000ffffffff0000ULL,
    0x0000ffffffff0001ULL, 0x0000ffffffff0100ULL, 0x0000ffffff00ff01ULL, 0x0000ffffff000000ULL,
    0x0000ffffff000101ULL, 0x0000ffffff01ff00ULL, 0x0000ffffff0100ffULL, 0x0000ffffff010100ULL,
    0x0000ffff00ffffffULL, 0x0000ffff00ff0000ULL, 0x0000ffff00ff01ffULL, 0x0000ffff0000ff00ULL,
    0x0000ffff000000ffULL, 0x0000ffff00000000ULL, 0x0000ffff00000001ULL, 0x0000ffff00000100ULL,
    0x0000ffff00010000ULL, 0x0000ffff000101ffULL, 0x0000ffff01ff0001ULL, 0x0000ffff01ff0100ULL,
    0x0000ffff01000000ULL, 0x0000ffff010001ffULL, 0x0000ffff0101ffffULL, 0x0000ffff0101ff00ULL,
    0x0000ffff01010001ULL, 0x0000ffff01010100ULL, 0x0000ff00ffff0000ULL, 0x0000ff00ffff01ffULL,
    0x0000ff00ffff0100ULL, 0x0000ff00ffff0101ULL, 0x0000ff00ff00ff00ULL, 0x0000ff00ff0000ffULL,
    0x0000ff00ff000000ULL, 0x0000ff00ff000001ULL, 0x0000ff00ff0001ffULL, 0x0000ff00ff000100ULL,
    0x0000ff00ff01ffffULL, 0x0000ff00ff010000ULL, 0x0000ff00ff010001ULL, 0x0000ff00ff0101ffULL,
    0x0000ff00ff010101ULL, 0x0000ff0000ffff00ULL, 0x0000ff0000ff00ffULL, 0x0000ff0000ff0000ULL,
    0x0000ff0000ff0001ULL, 0x0000ff0000ff0100ULL, 0x0000ff000000ffffULL, 0x0000ff000000ff00ULL,
    0x0000ff000000ff01ULL, 0x0000ff00000000ffULL, 0x0000ff0000000000ULL, 0x0000ff0000000001ULL,
    0x0000ff00000001ffULL, 0x0000ff0000000100ULL, 0x0000ff0000000101ULL, 0x0000ff000001ff00ULL,
    0x0000ff00000100ffULL, 0x0000ff0000010000ULL, 0x0000ff0000010001ULL, 0x0000ff0000010100ULL,
    0x0000ff0001ffff01ULL, 0x0000ff0001ff0000ULL, 0x0000ff000100ff00ULL, 0x0000ff00010000ffULL,
    0x0000ff0001000000ULL, 0x0000ff0001000001ULL, 0x0000ff0001000100ULL, 0x0000ff000101ffffULL,
    0x0000ff0001010000ULL, 0x0000ff0001010101ULL, 0x0000ff01ffffff00ULL, 0x0000ff01ffff0001ULL,
    0x0000ff01ff00ff01ULL, 0x0000ff01ff000000ULL, 0x0000ff01ff000101ULL, 0x0000ff01ff01ff00ULL,
    0x0000ff01ff0100ffULL, 0x0000ff0100ffff01ULL, 0x0000ff0100ff0000ULL, 0x0000ff0100ff0101ULL,
    0x0000ff010000ff00ULL, 0x0000ff01000000ffULL, 0x0000ff0100000000ULL, 0x0000ff0100000001ULL,
    0x0000ff0100000100ULL, 0x0000ff010001ff01ULL, 0x0000ff0100010000ULL, 0x0000ff0101ff0000ULL,
    0x0000ff010100ffffULL, 0x0000ff010100ff01ULL, 0x0000ff0101000000ULL, 0x0000ff0101000100ULL,
    0x0000ff0101000101ULL, 0x0000ff01010100ffULL, 0x000000ffffff00ffULL, 0x000000ffffff0000ULL,
    0x000000ffff00ff00ULL, 0x000000ffff0000ffULL, 0x000000ffff000000ULL, 0x000000ffff000001ULL,
    0x000000ffff0001ffULL, 0x000000ffff000100ULL, 0x000000ffff01ff00ULL, 0x000000ffff010000ULL,
    0x000000ffff0101ffULL, 0x000000ffff010101ULL, 0x000000ff00ffff00ULL, 0x000000ff00ff00ffULL,
    0x000000ff00ff0000ULL, 0x000000ff00ff0001ULL, 0x000000ff00ff0100ULL, 0x000000ff00ff0101ULL,
    0x000000ff0000ffffULL, 0x000000ff0000ff00ULL, 0x000000ff000000ffULL, 0x000000ff00000000ULL,
    0x000000ff00000001ULL, 0x000000ff000001ffULL, 0x000000ff00000100ULL, 0x000000ff00000101ULL,
    0x000000ff0001ff00ULL, 0x000000ff0001ff01ULL, 0x000000ff000100ffULL, 0x000000ff00010000ULL,
    0x000000ff00010001ULL, 0x000000ff00010100ULL, 0x000000ff01ffffffULL, 0x000000ff01ff01ffULL,
    0x000000ff01ff0101ULL, 0x000000ff0100ff00ULL, 0x000000ff010000ffULL, 0x000000ff01000000ULL,
    0x000000ff01000001ULL, 0x000000ff01000100ULL, 0x000000ff0101ff00ULL, 0x000000ff010100ffULL,
    0x000000ff01010000ULL, 0x000000ff01010101ULL, 0x00000000ffffff00ULL, 0x00000000ffffff01ULL,
    0x00000000ffff00ffULL, 0x00000000ffff0000ULL, 0x00000000ffff0001ULL, 0x00000000ffff0100ULL,
    0x00000000ff00ffffULL, 0x00000000ff00ff00ULL, 0x00000000ff00ff01ULL, 0x00000000ff0000ffULL,
    0x00000000ff000000ULL, 0x00000000ff000001ULL, 0x00000000ff000100ULL, 0x00000000ff000101ULL,
    0x00000000ff01ff00ULL, 0x00000000ff0100ffULL, 0x00000000ff010000ULL, 0x00000000ff010001ULL,
    0x00000000ff010100ULL, 0x0000000000ffffffULL, 0x0000000000ffff00ULL, 0x0000000000ffff01ULL,
    0x0000000000ff00ffULL, 0x0000000000ff0000ULL, 0x0000000000ff0001ULL, 0x0000000000ff01ffULL,
    0x0000000000ff0100ULL, 0x000000000000ffffULL, 0x000000000000ff00ULL, 0x000000000000ff01ULL,
    0x00000000000000ffULL, 0x0000000000000000ULL, 0x0000000000000001ULL, 0x00000000000001ffULL,
    0x0000000000000100ULL, 0x0000000000000101ULL, 0x000000000001ffffULL, 0x000000000001ff00ULL,
    0x00000000000100ffULL, 0x0000000000010000ULL, 0x0000000000010001ULL, 0x00000000000101ffULL,
    0x0000000000010100ULL, 0x0000000000010101ULL, 0x0000000001ffff00ULL, 0x0000000001ff00ffULL,
    0x0000000001ff0000ULL, 0x0000000001ff0100ULL, 0x0000000001ff0101ULL, 0x000000000100ffffULL,
    0x000000000100ff00ULL, 0x00000000010000ffULL, 0x0000000001000000ULL, 0x0000000001000001ULL,
    0x00000000010001ffULL, 0x0000000001000100ULL, 0x000000000101ff00ULL, 0x00000000010100ffULL,
    0x0000000001010000ULL, 0x0000000001010001ULL, 0x0000000001010100ULL, 0x00000001ffffffffULL,
    0x00000001ffffff00ULL, 0x00000001ffffff01ULL, 0x00000001ffff00ffULL, 0x00000001ffff0001ULL,
    0x00000001ffff01ffULL, 0x00000001ffff0100ULL, 0x00000001ff00ff00ULL, 0x00000001ff0000ffULL,
    0x00000001ff000000ULL, 0x00000001ff0001ffULL, 0x00000001ff000100ULL, 0x00000001ff01ffffULL,
    0x00000001ff01ff00ULL, 0x00000001ff01ff01ULL, 0x00000001ff0100ffULL, 0x00000001ff010000ULL,
    0x00000001ff010001ULL, 0x00000001ff0101ffULL, 0x00000001ff010100ULL, 0x0000000100ffff00ULL,
    0x0000000100ff0000ULL, 0x0000000100ff0001ULL, 0x0000000100ff01ffULL, 0x0000000100ff0100ULL,
    0x0000000100ff0101ULL, 0x000000010000ffffULL, 0x000000010000ff00ULL, 0x000000010000ff01ULL,
    0x00000001000000ffULL, 0x0000000100000000ULL, 0x0000000100000001ULL, 0x00000001000001ffULL,
    0x0000000100000100ULL, 0x0000000100000101ULL, 0x000000010001ff00ULL, 0x00000001000100ffULL,
    0x0000000100010000ULL, 0x0000000100010100ULL, 0x0000000101ffff01ULL, 0x0000000101ff0000ULL,
    0x0000000101ff0001ULL, 0x0000000101ff01ffULL, 0x0000000101ff0100ULL, 0x0000000101ff0101ULL,
    0x000000010100ff00ULL, 0x0000000101000000ULL, 0x0000000101000101ULL, 0x000000010101ff01ULL,
    0x0000000101010000ULL, 0x0000000101010001ULL, 0x00000001010101ffULL, 0x0000000101010100ULL,
    0x000001ffffff00ffULL, 0x000001ffffff0000ULL, 0x000001ffffff0001ULL, 0x000001ffffff0100ULL,
    0x000001ffff00ffffULL, 0x000001ffff000000ULL, 0x000001ffff0001ffULL, 0x000001ffff01ff00ULL,
    0x000001ffff010101ULL, 0x000001ff00ff0000ULL, 0x000001ff00ff01ffULL, 0x000001ff00ff0101ULL,
    0x000001ff0000ff00ULL, 0x000001ff000000ffULL, 0x000001ff00000000ULL, 0x000001ff00000001ULL,
    0x000001ff000001ffULL, 0x000001ff00000100ULL, 0x000001ff0001ffffULL, 0x000001ff0001ff01ULL,
    0x000001ff000100ffULL, 0x000001ff00010000ULL, 0x000001ff01ffff01ULL, 0x000001ff01ff0100ULL,
    0x000001ff0100ffffULL, 0x000001ff0100ff01ULL, 0x000001ff01000000ULL, 0x000001ff010001ffULL,
    0x000001ff0101ff00ULL, 0x000001ff01010100ULL, 0x00000100ffffff00ULL, 0x00000100ffffff01ULL,
    0x00000100ffff0000ULL, 0x00000100ffff0101ULL, 0x00000100ff00ff00ULL, 0x00000100ff0000ffULL,
    0x00000100ff000000ULL, 0x00000100ff000001ULL, 0x00000100ff000100ULL, 0x00000100ff010000ULL,
    0x0000010000ffff00ULL, 0x0000010000ff00ffULL, 0x0000010000ff0000ULL, 0x0000010000ff0001ULL,
    0x0000010000ff0100ULL, 0x000001000000ffffULL, 0x000001000000ff00ULL, 0x000001000000ff01ULL,
    0x00000100000000ffULL, 0x0000010000000000ULL, 0x0000010000000001ULL, 0x00000100000001ffULL,
    0x0000010000000100ULL, 0x0000010000000101ULL, 0x000001000001ff00ULL, 0x00000100000100ffULL,
    0x0000010000010000ULL, 0x0000010000010001ULL, 0x0000010000010100ULL, 0x0000010001ffff00ULL,
    0x0000010001ff0000ULL, 0x0000010001ff0100ULL, 0x000001000100ff00ULL, 0x00000100010000ffULL,
    0x0000010001000000ULL, 0x0000010001000001ULL, 0x00000100010001ffULL, 0x0000010001000100ULL,
    0x0000010001010000ULL, 0x00000101ffff00ffULL, 0x00000101ffff01ffULL, 0x00000101ff000000ULL,
    0x00000101ff000101ULL, 0x00000101ff01ffffULL, 0x00000101ff010000ULL, 0x00000101ff010001ULL,
    0x00000101ff010100ULL, 0x0000010100ff0000ULL, 0x0000010100ff01ffULL, 0x0000010100ff0100ULL,
    0x000001010000ff00ULL, 0x0000010100000000ULL, 0x0000010100000001ULL, 0x00000101000001ffULL,
    0x0000010100000100ULL, 0x000001010001ff01ULL, 0x0000010100010000ULL, 0x00000101000101ffULL,
    0x0000010100010101ULL, 0x0000010101ffff00ULL, 0x0000010101ff0101ULL, 0x000001010100ff01ULL,
    0x0000010101000000ULL, 0x0000010101000001ULL, 0x00000101010001ffULL, 0x0000010101000101ULL,
    0x000001010101ff00ULL, 0x0001ffffffff0000ULL, 0x0001ffffff0000ffULL, 0x0001ffffff000001ULL,
    0x0001ffffff000100ULL, 0x0001ffffff010000ULL, 0x0001ffff00ff00ffULL, 0x0001ffff0000ffffULL,
    0x0001ffff00000000ULL, 0x0001ffff00000001ULL, 0x0001ffff000001ffULL, 0x0001ffff00000101ULL,
    0x0001ffff0001ff00ULL, 0x0001ffff000100ffULL, 0x0001ffff00010001ULL, 0x0001ffff00010100ULL,
    0x0001ffff01ffff00ULL, 0x0001ffff01000001ULL, 0x0001ffff01010000ULL, 0x0001ff00ffffff00ULL,
    0x0001ff00ffff00ffULL, 0x0001ff00ffff0001ULL, 0x0001ff00ffff0100ULL, 0x0001ff00ff00ff01ULL,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
    0x0001ff00ff000000ULL, 0x0001ff00ff01ff00ULL, 0x0001ff00ff01ff01ULL, 0x0001ff00ff010001ULL,
    0x0001ff00ff010100ULL, 0x0001ff0000ff0000ULL, 0x0001ff0000ff0100ULL, 0x0001ff000000ff00ULL,
    0x0001ff0000000000ULL, 0x0001ff0000000001ULL, 0x0001ff0000000100ULL, 0x0001ff0000010000ULL,
    0x0001ff0000010001ULL, 0x0001ff0000010101ULL, 0x0001ff0001ff00ffULL, 0x0001ff0001ff0101ULL,
    0x0001ff000100ff01ULL, 0x0001ff0001000000ULL, 0x0001ff000101ff00ULL, 0x0001ff0001010001ULL,
    0x0001ff0001010100ULL, 0x0001ff01ff00ff00ULL, 0x0001ff01ff000001ULL, 0x0001ff01ff000100ULL,
    0x0001ff0100ffffffULL, 0x0001ff0100ffff00ULL, 0x0001ff0100ff0001ULL, 0x0001ff0100000000ULL,
    0x0001ff0100000001ULL, 0x0001ff01000001ffULL, 0x0001ff010001ffffULL, 0x0001ff0101ff0000ULL,
    0x0001ff010100ff00ULL, 0x0001ff0101000001ULL, 0x0001ff0101010000ULL, 0x000100ffff00ff00ULL,
    0x000100ffff00ff01ULL, 0x000100ffff000000ULL, 0x000100ffff000001ULL, 0x000100ffff000101ULL,
    0x000100ffff01ff00ULL, 0x000100ffff010001ULL, 0x000100ffff010100ULL, 0x000100ff00ffffffULL,
    0x000100ff00ffff01ULL, 0x000100ff00ff0000ULL, 0x000100ff00ff01ffULL, 0x000100ff00ff0101ULL,
    0x000100ff0000ff00ULL, 0x000100ff000000ffULL, 0x000100ff00000000ULL, 0x000100ff00000001ULL,
    0x000100ff00000100ULL, 0x000100ff00000101ULL, 0x000100ff0001ffffULL, 0x000100ff0001ff01ULL,
    0x000100ff00010000ULL, 0x000100ff01ff00ffULL, 0x000100ff01ff0000ULL, 0x000100ff01ff0100ULL,
    0x000100ff0100ffffULL, 0x000100ff0100ff01ULL, 0x000100ff010000ffULL, 0x000100ff01000000ULL,
    0x000100ff01000001ULL, 0x000100ff010001ffULL, 0x000100ff01000101ULL, 0x000100ff0101ff00ULL,
    0x000100ff010100ffULL, 0x000100ff01010100ULL, 0x00010000ffff0000ULL, 0x00010000ffff01ffULL,
    0x00010000ffff0101ULL, 0x00010000ff00ff00ULL, 0x00010000ff000000ULL, 0x00010000ff000001ULL,
    0x00010000ff000100ULL, 0x0001000000ff00ffULL, 0x0001000000ff0000ULL, 0x0001000000ff0001ULL,
    0x0001000000ff0100ULL, 0x000100000000ffffULL, 0x000100000000ff00ULL, 0x00010000000000ffULL,
    0x0001000000000000ULL, 0x0001000000000001ULL, 0x0001000000000100ULL, 0x000100000001ff00ULL,
    0x00010000000100ffULL, 0x0001000000010000ULL, 0x0001000000010001ULL, 0x0001000000010100ULL,
    0x0001000001ff0001ULL, 0x0001000001ff0100ULL, 0x0001000001ff0101ULL, 0x000100000100ff00ULL,
    0x0001000001000000ULL, 0x0001000001000001ULL, 0x0001000001000100ULL, 0x0001000001000101ULL,
    0x000100000101ff01ULL, 0x0001000001010000ULL, 0x0001000001010001ULL, 0x00010000010101ffULL,
    0x00010001ffffff01ULL, 0x00010001ffff0100ULL, 0x00010001ff000000ULL, 0x00010001ff01ffffULL,
    0x00010001ff010001ULL, 0x00010001ff0101ffULL, 0x00010001ff010100ULL, 0x0001000100ffffffULL,
    0x0001000100ff0000ULL, 0x0001000100ff01ffULL, 0x0001000100ff0101ULL, 0x000100010000ff00ULL,
    0x00010001000000ffULL, 0x0001000100000000ULL, 0x0001000100000001ULL, 0x00010001000001ffULL,
    0x0001000100000101ULL, 0x000100010001ffffULL, 0x0001000100010000ULL, 0x00010001000101ffULL,
    0x0001000101ffffffULL, 0x0001000101ffff01ULL, 0x0001000101ff0000ULL, 0x0001000101ff0101ULL,
    0x00010001010000ffULL, 0x0001000101000001ULL, 0x00010001010001ffULL, 0x0001000101000100ULL,
    0x000100010101ffffULL, 0x00010001010100ffULL, 0x0001000101010001ULL, 0x0001000101010101ULL,
    0x000101ffff000001ULL, 0x000101ffff000100ULL, 0x000101ffff010000ULL, 0x000101ff00ffff00ULL,
    0x000101ff0000ff01ULL, 0x000101ff00000000ULL, 0x000101ff00000101ULL, 0x000101ff0001ff00ULL,
    0x000101ff00010100ULL, 0x000101ff01ff0000ULL, 0x000101ff0100ff00ULL, 0x000101ff010001ffULL,
    0x000101ff01010001ULL, 0x00010100ffffff00ULL, 0x00010100ffff00ffULL, 0x00010100ff00ffffULL,
    0x00010100ff000000ULL, 0x00010100ff01ff00ULL, 0x00010100ff0100ffULL, 0x00010100ff010001ULL,
    0x00010100ff010100ULL, 0x0001010000ffffffULL, 0x0001010000ffff00ULL, 0x0001010000ff0000ULL,
    0x0001010000ff0001ULL, 0x0001010000ff01ffULL, 0x000101000000ff00ULL, 0x00010100000000ffULL,
    0x0001010000000000ULL, 0x0001010000000001ULL, 0x0001010000000100ULL, 0x000101000001ffffULL,
    0x0001010000010000ULL, 0x0001010000010101ULL, 0x0001010001ffff01ULL, 0x0001010001ff00ffULL,
    0x0001010001ff0101ULL, 0x0001010001000000ULL, 0x000101000101ff00ULL, 0x00010100010100ffULL,
    0x0001010001010000ULL, 0x0001010001010100ULL, 0x00010101ff00ff00ULL, 0x00010101ff000001ULL,
    0x00010101ff0001ffULL, 0x0001010100ffff00ULL, 0x0001010100ff00ffULL, 0x0001010100ff0100ULL,
    0x000101010000ffffULL, 0x0001010100000000ULL, 0x00010101000001ffULL, 0x0001010100000101ULL,
    0x00010101000100ffULL, 0x0001010100010000ULL, 0x0001010100010100ULL, 0x0001010101ff0001ULL,
    0x00010101010000ffULL, 0x00010101010001ffULL, 0x0001010101000101ULL, 0x0001010101010001ULL,
    0x01ffffffffffffffULL, 0x01ffffffffffff01ULL, 0x01ffffffffff01ffULL, 0x01ffffffffff0101ULL,
    0x01ffffffff01ffffULL, 0x01ffffffff01ff01ULL, 0x01ffffffff0101ffULL, 0x01ffffffff010101ULL,
    0x01ffffff00ff0000ULL, 0x01ffffff0000ffffULL, 0x01ffffff0000ff00ULL, 0x01ffffff000000ffULL,
    0x01ffffff00000001ULL, 0x01ffffff00000100ULL, 0x01ffffff00010000ULL, 0x01ffffff01ffffffULL,
    0x01ffffff01ffff01ULL, 0x01ffffff01ff01ffULL, 0x01ffffff01ff0101ULL, 0x01ffffff01000000ULL,
    0x01ffffff0101ffffULL, 0x01ffffff0101ff01ULL, 0x01ffffff010101ffULL, 0x01ffffff01010101ULL,
    0x01ffff00ffff0000ULL, 0x01ffff00ff00ff00ULL, 0x01ffff00ff0000ffULL, 0x01ffff00ff000001ULL,
    0x01ffff00ff000100ULL, 0x01ffff00ff010000ULL, 0x01ffff0000ffff00ULL, 0x01ffff0000ff00ffULL,
    0x01ffff0000ff0100ULL, 0x01ffff000000ffffULL, 0x01ffff000000ff01ULL, 0x01ffff0000000000ULL,
    0x01ffff0000000001ULL, 0x01ffff00000001ffULL, 0x01ffff0000000100ULL, 0x01ffff00000100ffULL,
    0x01ffff0000010001ULL, 0x01ffff0000010100ULL, 0x01ffff0001ff0000ULL, 0x01ffff0001ff0100ULL,
    0x01ffff00010000ffULL, 0x01ffff0001000001ULL, 0x01ffff0001000100ULL, 0x01ffff0001010000ULL,
    0x01ffff01ffffffffULL, 0x01ffff01ffffff01ULL, 0x01ffff01ffff01ffULL, 0x01ffff01ffff0101ULL,
    0x01ffff01ff000000ULL, 0x01ffff01ff01ffffULL, 0x01ffff01ff01ff01ULL, 0x01ffff01ff0101ffULL,
    0x01ffff01ff010101ULL, 0x01ffff010000ff00ULL, 0x01ffff01000000ffULL, 0x01ffff0100000100ULL,
    0x01ffff0100010000ULL, 0x01ffff0101ffffffULL, 0x01ffff0101ffff01ULL, 0x01ffff0101ff01ffULL,
    0x01ffff0101ff0101ULL, 0x01ffff0101000000ULL, 0x01ffff010101ffffULL, 0x01ffff010101ff01ULL,
    0x01ffff01010101ffULL, 0x01ffff0101010101ULL, 0x01ff00ffff0000ffULL, 0x01ff00ffff000100ULL,
    0x01ff00ff00ffff00ULL, 0x01ff00ff00ff00ffULL, 0x01ff00ff0000ff00ULL, 0x01ff00ff00000000ULL,
    0x01ff00ff00000101ULL, 0x01ff00ff0001ff00ULL, 0x01ff00ff000100ffULL, 0x01ff00ff00010100ULL,
    0x01ff00ff010000ffULL, 0x01ff00ff01000100ULL, 0x01ff0000ffffff00ULL, 0x01ff0000ffff0100ULL,
    0x01ff0000ff00ff01ULL, 0x01ff0000ff000000ULL, 0x01ff0000ff000101ULL, 0x01ff0000ff010001ULL,
    0x01ff0000ff010100ULL, 0x01ff000000ffffffULL, 0x01ff000000ffff00ULL, 0x01ff000000ff0000ULL,
    0x01ff000000ff01ffULL, 0x01ff00000000ff00ULL, 0x01ff0000000000ffULL, 0x01ff000000000000ULL,
    0x01ff000000000001ULL, 0x01ff000000000100ULL, 0x01ff000000000101ULL, 0x01ff000000010000ULL,
    0x01ff000000010001ULL, 0x01ff0000000101ffULL, 0x01ff000000010101ULL, 0x01ff000001ffff00ULL,
    0x01ff000001ff00ffULL, 0x01ff000001ff0001ULL, 0x01ff000001ff0100ULL, 0x01ff00000100ffffULL,
    0x01ff00000100ff01ULL, 0x01ff000001000000ULL, 0x01ff0000010001ffULL, 0x01ff000001010001ULL,
    0x01ff0001ff00ff00ULL, 0x01ff0001ff000001ULL, 0x01ff0001ff000100ULL, 0x01ff0001ff010000ULL,
    0x01ff000100ffff00ULL, 0x01ff000100ff00ffULL, 0x01ff000100ff0100ULL, 0x01ff000100ff0101ULL,
    0x01ff00010000ffffULL, 0x01ff000100000000ULL, 0x01ff000100000100ULL, 0x01ff000100000101ULL,
    0x01ff00010001ff00ULL, 0x01ff000100010001ULL, 0x01ff000100010101ULL, 0x01ff000101ff0000ULL,
    0x01ff00010100ff00ULL, 0x01ff000101000101ULL, 0x01ff0001010100ffULL, 0x01ff01ffffffffffULL,
    0x01ff01ffffffff01ULL, 0x01ff01ffffff01ffULL, 0x01ff01ffffff0101ULL, 0x01ff01ffff000000ULL,
    0x01ff01ffff01ffffULL, 0x01ff01ffff01ff01ULL, 0x01ff01ffff0101ffULL, 0x01ff01ffff010101ULL,
    0x01ff01ff00ffff00ULL, 0x01ff01ff00ff0000ULL, 0x01ff01ff0000ff00ULL, 0x01ff01ff000000ffULL,
    0x01ff01ff00000100ULL, 0x01ff01ff00010000ULL, 0x01ff01ff00010100ULL, 0x01ff01ff01ffffffULL,
    0x01ff01ff01ffff01ULL, 0x01ff01ff01ff01ffULL, 0x01ff01ff01ff0101ULL, 0x01ff01ff01000000ULL,
    0x01ff01ff0101ffffULL, 0x01ff01ff0101ff01ULL, 0x01ff01ff010101ffULL, 0x01ff01ff01010101ULL,
    0x01ff0100ffff0000ULL, 0x01ff0100ffff0001ULL, 0x01ff0100ff00ff00ULL, 0x01ff0100ff0000ffULL,
    0x01ff0100ff000001ULL, 0x01ff0100ff010000ULL, 0x01ff010000ffff00ULL, 0x01ff010000ff00ffULL,
    0x01ff010000ff0001ULL, 0x01ff010000ff0100ULL, 0x01ff01000000ffffULL, 0x01ff01000000ff01ULL,
    0x01ff010000000000ULL, 0x01ff010000000101ULL, 0x01ff01000001ff00ULL, 0x01ff0100000100ffULL,
    0x01ff010001ff0000ULL, 0x01ff010001000001ULL, 0x01ff010001000100ULL, 0x01ff010001010000ULL,
    0x01ff0101ffffffffULL, 0x01ff0101ffffff01ULL, 0x01ff0101ffff01ffULL, 0x01ff0101ffff0101ULL,
    0x01ff0101ff000000ULL, 0x01ff0101ff01ffffULL, 0x01ff0101ff01ff01ULL, 0x01ff0101ff0101ffULL,
    0x01ff0101ff010101ULL, 0x01ff010100ff0000ULL, 0x01ff01010000ff00ULL, 0x01ff0101000000ffULL,
    0x01ff010100000001ULL, 0x01ff010101ffffffULL, 0x01ff010101ffff01ULL, 0x01ff010101ff01ffULL,
    0x01ff010101ff0101ULL, 0x01ff010101000000ULL, 0x01ff01010101ffffULL, 0x01ff01010101ff01ULL,
    0x01ff0101010101ffULL, 0x01ff010101010101ULL, 0x0100ffffffff0000ULL, 0x0100ffffff00ff00ULL,
    0x0100ffffff000001ULL, 0x0100ffffff0001ffULL, 0x0100ffffff000100ULL, 0x0100ffffff010000ULL,
    0x0100ffff00ffff00ULL, 0x0100ffff00ff0001ULL, 0x0100ffff00ff0100ULL, 0x0100ffff00000000ULL,
    0x0100ffff000001ffULL, 0x0100ffff00000101ULL, 0x0100ffff00010100ULL, 0x0100ffff00010101ULL,
    0x0100ffff01ff0000ULL, 0x0100ffff0100ff00ULL, 0x0100ffff010000ffULL, 0x0100ffff01000001ULL,
    0x0100ffff01000100ULL, 0x0100ffff01010000ULL, 0x0100ff00ffffff00ULL, 0x0100ff00ffff00ffULL,
    0x0100ff00ffff0001ULL, 0x0100ff00ffff0100ULL, 0x0100ff00ff00ffffULL, 0x0100ff00ff000000ULL,
    0x0100ff00ff0001ffULL, 0x0100ff00ff000101ULL, 0x0100ff00ff01ff00ULL, 0x0100ff00ff0100ffULL,
    0x0100ff00ff010001ULL, 0x0100ff00ff010100ULL, 0x0100ff0000ffffffULL, 0x0100ff0000ff0000ULL,
    0x0100ff000000ffffULL, 0x0100ff000000ff00ULL, 0x0100ff00000000ffULL, 0x0100ff0000000000ULL,
    0x0100ff0000000001ULL, 0x0100ff0000000100ULL, 0x0100ff000001ff01ULL, 0x0100ff0000010000ULL,
    0x0100ff0001ff00ffULL, 0x0100ff0001ff0001ULL, 0x0100ff000100ff01ULL, 0x0100ff0001000000ULL,
    0x0100ff00010001ffULL, 0x0100ff000101ff00ULL, 0x0100ff00010100ffULL, 0x0100ff0001010001ULL,
    0x0100ff0001010100ULL, 0x0100ff01ffff0000ULL, 0x0100ff01ff00ff00ULL, 0x0100ff01ff0000ffULL,
    0x0100ff01ff000100ULL, 0x0100ff01ff010000ULL, 0x0100ff0100ff00ffULL, 0x0100ff0100ff0001ULL,
    0x0100ff0100ff0100ULL, 0x0100ff010000ffffULL, 0x0100ff010000ff01ULL, 0x0100ff0100000000ULL,
    0x0100ff01000001ffULL, 0x0100ff0100010001ULL, 0x0100ff0100010100ULL, 0x0100ff0101ff0000ULL,
    0x0100ff01010000ffULL, 0x0100ff0101000001ULL, 0x0100ff0101010100ULL, 0x010000ffffffff00ULL,
    0x010000ffffff00ffULL, 0x010000ffffff0001ULL, 0x010000ffff00ffffULL, 0x010000ffff000000ULL,
    0x010000ffff0001ffULL, 0x010000ffff010001ULL, 0x010000ff00ffffffULL, 0x010000ff00ff0101ULL,
    0x010000ff0000ff00ULL, 0x010000ff000000ffULL, 0x010000ff00000000ULL, 0x010000ff00000001ULL,
    0x010000ff000001ffULL, 0x010000ff00000100ULL, 0x010000ff0001ffffULL, 0x010000ff0001ff00ULL,
    0x010000ff0001ff01ULL, 0x010000ff00010000ULL, 0x010000ff01ff00ffULL, 0x010000ff01ff0001ULL,
    0x010000ff0100ff01ULL, 0x010000ff010000ffULL, 0x010000ff01000000ULL, 0x010000ff010001ffULL,
    0x010000ff0101ff00ULL, 0x010000ff01010100ULL, 0x01000000ffffffffULL, 0x01000000ffff0000ULL,
    0x01000000ffff01ffULL, 0x01000000ffff0101ULL, 0x01000000ff00ffffULL, 0x01000000ff00ff00ULL,
    0x01000000ff0000ffULL, 0x01000000ff000000ULL, 0x01000000ff000001ULL, 0x01000000ff000100ULL,
    0x01000000ff01ff00ULL, 0x01000000ff010000ULL, 0x01000000ff010100ULL, 0x01000000ff010101ULL,
    0x0100000000ffff00ULL, 0x0100000000ff00ffULL, 0x0100000000ff0000ULL, 0x0100000000ff0001ULL,
    0x0100000000ff0100ULL, 0x010000000000ffffULL, 0x010000000000ff00ULL, 0x010000000000ff01ULL,
    0x01000000000000ffULL, 0x0100000000000000ULL, 0x0100000000000001ULL, 0x01000000000001ffULL,
    0x0100000000000100ULL, 0x0100000000000101ULL, 0x010000000001ff00ULL, 0x01000000000100ffULL,
    0x0100000000010000ULL, 0x0100000000010001ULL, 0x0100000000010100ULL, 0x0100000001ffff00ULL,
    0x0100000001ff0000ULL, 0x0100000001ff01ffULL, 0x010000000100ff00ULL, 0x010000000100ff01ULL,
    0x01000000010000ffULL, 0x0100000001000000ULL, 0x0100000001000001ULL, 0x0100000001000100ULL,
    0x0100000001000101ULL, 0x010000000101ffffULL, 0x010000000101ff01ULL, 0x0100000001010000ULL,
    0x01000000010101ffULL, 0x0100000001010101ULL, 0x01000001ffffff00ULL, 0x01000001ffff00ffULL,
    0x01000001ff00ffffULL, 0x01000001ff000000ULL, 0x01000001ff000100ULL, 0x01000001ff01ffffULL,
    0x01000001ff010001ULL, 0x01000001ff010100ULL, 0x0100000100ff0000ULL, 0x0100000100ff01ffULL,
    0x0100000100ff0100ULL, 0x010000010000ff00ULL, 0x010000010000ff01ULL, 0x0100000100000000ULL,
    0x0100000100000001ULL, 0x0100000100000100ULL, 0x0100000100010000ULL, 0x01000001000101ffULL,
    0x0100000101ffff01ULL, 0x0100000101ff00ffULL, 0x0100000101ff0100ULL, 0x0100000101ff0101ULL,
    0x010000010100ff01ULL, 0x01000001010000ffULL, 0x0100000101000000ULL, 0x01000001010100ffULL,
    0x0100000101010001ULL, 0x0100000101010100ULL, 0x010001ffffff0000ULL, 0x010001ffff000001ULL,
    0x010001ffff000100ULL, 0x010001ffff010000ULL, 0x010001ff00ffff00ULL, 0x010001ff00ff0001ULL,
    0x010001ff0000ffffULL, 0x010001ff0000ff01ULL, 0x010001ff00000000ULL, 0x010001ff00000001ULL,
    0x010001ff00000101ULL, 0x010001ff000100ffULL, 0x010001ff00010000ULL, 0x010001ff01ff0000ULL,
    0x010001ff0100ff00ULL, 0x010001ff01000001ULL, 0x010001ff01000100ULL, 0x010001ff01010000ULL,
    0x01000100ffff00ffULL, 0x01000100ffff0001ULL, 0x01000100ffff0100ULL, 0x01000100ff00ffffULL,
    0x01000100ff00ff01ULL, 0x01000100ff000000ULL, 0x01000100ff0001ffULL, 0x01000100ff000101ULL,
    0x01000100ff01ffffULL, 0x01000100ff01ff00ULL, 0x01000100ff0100ffULL, 0x01000100ff010001ULL,
    0x0100010000ffffffULL, 0x0100010000ffff01ULL, 0x0100010000ff0000ULL, 0x0100010000ff01ffULL,
    0x0100010000ff0101ULL, 0x010001000000ff00ULL, 0x01000100000000ffULL, 0x0100010000000000ULL,
    0x0100010000000001ULL, 0x0100010000000100ULL, 0x010001000001ff01ULL, 0x0100010000010000ULL,
    0x0100010000010001ULL, 0x0100010000010101ULL, 0x0100010001ffff00ULL, 0x0100010001ff00ffULL,
    0x010001000100ffffULL, 0x010001000100ff01ULL, 0x0100010001000000ULL, 0x0100010001000101ULL,
    0x010001000101ff00ULL, 0x0100010001010001ULL, 0x01000101ffff0000ULL, 0x01000101ff000000ULL,
    0x01000101ff010000ULL, 0x0100010100ff00ffULL, 0x0100010100ff0001ULL, 0x0100010100ff0100ULL,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
    0x010001010000ffffULL, 0x0100010100000000ULL, 0x01000101000001ffULL, 0x010001010001ff00ULL,
    0x0100010101ff0000ULL, 0x010001010100ff00ULL, 0x01000101010000ffULL, 0x0100010101000000ULL,
    0x0100010101000001ULL, 0x0101ffffffffffffULL, 0x0101ffffffffff01ULL, 0x0101ffffffff01ffULL,
    0x0101ffffffff0101ULL, 0x0101ffffff000000ULL, 0x0101ffffff01ffffULL, 0x0101ffffff01ff01ULL,
    0x0101ffffff0101ffULL, 0x0101ffffff010101ULL, 0x0101ffff00ff0000ULL, 0x0101ffff0000ff00ULL,
    0x0101ffff000000ffULL, 0x0101ffff00000001ULL, 0x0101ffff00000100ULL, 0x0101ffff01ffffffULL,
    0x0101ffff01ffff01ULL, 0x0101ffff01ff01ffULL, 0x0101ffff01ff0101ULL, 0x0101ffff01000000ULL,
    0x0101ffff0101ffffULL, 0x0101ffff0101ff01ULL, 0x0101ffff010101ffULL, 0x0101ffff01010101ULL,
    0x0101ff00ffff0000ULL, 0x0101ff00ffff0100ULL, 0x0101ff00ff00ff00ULL, 0x0101ff00ff0000ffULL,
    0x0101ff00ff000001ULL, 0x0101ff00ff000100ULL, 0x0101ff00ff000101ULL, 0x0101ff0000ff0001ULL,
    0x0101ff0000ff0100ULL, 0x0101ff000000ff00ULL, 0x0101ff0000000000ULL, 0x0101ff00000001ffULL,
    0x0101ff0000000101ULL, 0x0101ff000001ff00ULL, 0x0101ff00000100ffULL, 0x0101ff0001ff0000ULL,
    0x0101ff000100ffffULL, 0x0101ff000100ff01ULL, 0x0101ff0001000001ULL, 0x0101ff0001000100ULL,
    0x0101ff01ffffff01ULL, 0x0101ff01ffff01ffULL, 0x0101ff01ffff0101ULL, 0x0101ff01ff00ffffULL,
    0x0101ff01ff000100ULL, 0x0101ff01ff01ff01ULL, 0x0101ff01ff0101ffULL, 0x0101ff01ff010101ULL,
    0x0101ff0100ff0000ULL, 0x0101ff010000ff00ULL, 0x0101ff0100000001ULL, 0x0101ff0100000100ULL,
    0x0101ff0100010000ULL, 0x0101ff0101ffffffULL, 0x0101ff0101ffff01ULL, 0x0101ff0101ff01ffULL,
    0x0101ff0101ff0101ULL, 0x0101ff0101000000ULL, 0x0101ff010101ffffULL, 0x0101ff010101ff01ULL,
    0x0101ff01010101ffULL, 0x0101ff0101010101ULL, 0x010100ffff000100ULL, 0x010100ffff010000ULL,
    0x010100ff00ffff00ULL, 0x010100ff00ff00ffULL, 0x010100ff0000ffffULL, 0x010100ff000000ffULL,
    0x010100ff00000000ULL, 0x010100ff000001ffULL, 0x010100ff00000101ULL, 0x010100ff0001ff00ULL,
    0x010100ff00010000ULL, 0x010100ff00010001ULL, 0x010100ff000101ffULL, 0x010100ff00010100ULL,
    0x010100ff01ff0000ULL, 0x01010000ffff0001ULL, 0x01010000ffff0100ULL, 0x01010000ff00ffffULL,
    0x01010000ff00ff01ULL, 0x01010000ff000000ULL, 0x01010000ff0001ffULL, 0x01010000ff010001ULL,
    0x01010000ff010100ULL, 0x0101000000ffff01ULL, 0x0101000000ff0000ULL, 0x010100000000ff00ULL,
    0x01010000000000ffULL, 0x0101000000000000ULL, 0x0101000000000001ULL, 0x0101000000000100ULL,
    0x0101000000010000ULL, 0x0101000000010101ULL, 0x0101000001ffff00ULL, 0x0101000001ff00ffULL,
    0x0101000001ff0000ULL, 0x0101000001ff0001ULL, 0x0101000001ff0100ULL, 0x010100000100ff01ULL,
    0x0101000001000000ULL, 0x01010000010001ffULL, 0x01010001ffff0000ULL, 0x01010001ff00ff00ULL,
    0x01010001ff000001ULL, 0x01010001ff000101ULL, 0x01010001ff01ff00ULL, 0x01010001ff010000ULL,
    0x0101000100ff00ffULL, 0x0101000100ff0001ULL, 0x0101000100ff0101ULL, 0x010100010000ff01ULL,
    0x0101000100000000ULL, 0x0101000100000001ULL, 0x01010001000001ffULL, 0x010100010001ffffULL,
    0x010100010001ff01ULL, 0x0101000101ff0001ULL, 0x010100010100ffffULL, 0x0101000101000000ULL,
    0x0101000101000001ULL, 0x0101000101000100ULL, 0x010100010101ff00ULL, 0x01010001010100ffULL,
    0x0101000101010001ULL, 0x010101ffffffffffULL, 0x010101ffffffff01ULL, 0x010101ffffff01ffULL,
    0x010101ffffff0101ULL, 0x010101ffff01ffffULL, 0x010101ffff01ff01ULL, 0x010101ffff0101ffULL,
    0x010101ffff010101ULL, 0x010101ff0000ff00ULL, 0x010101ff000000ffULL, 0x010101ff00000001ULL,
    0x010101ff00000100ULL, 0x010101ff01ffffffULL, 0x010101ff01ffff01ULL, 0x010101ff01ff01ffULL,
    0x010101ff01ff0101ULL, 0x010101ff01000000ULL, 0x010101ff0101ffffULL, 0x010101ff0101ff01ULL,
    0x010101ff010101ffULL, 0x010101ff01010101ULL, 0x01010100ffff0000ULL, 0x01010100ff0000ffULL,
    0x01010100ff000100ULL, 0x01010100ff01ff00ULL, 0x01010100ff010000ULL, 0x0101010000ffff00ULL,
    0x010101000000ffffULL, 0x0101010000000000ULL, 0x0101010000000101ULL, 0x010101000001ff00ULL,
    0x0101010000010001ULL, 0x0101010000010100ULL, 0x010101000100ffffULL, 0x0101010001000001ULL,
    0x01010101ffffffffULL, 0x01010101ffffff01ULL, 0x01010101ffff01ffULL, 0x01010101ffff0101ULL,
    0x01010101ff01ffffULL, 0x01010101ff01ff01ULL, 0x01010101ff0101ffULL, 0x01010101ff010101ULL,
    0x010101010000ff00ULL, 0x01010101000000ffULL, 0x0101010100000001ULL, 0x0101010101ffffffULL,
    0x0101010101ffff01ULL, 0x0101010101ff01ffULL, 0x0101010101ff0101ULL, 0x0101010101000000ULL,
    0x010101010101ffffULL, 0x010101010101ff01ULL, 0x01010101010101ffULL, 0x0101010101010101ULL,
};

// Scale of an IQ1_M super-block. Unlike every other K-quant here the f16 scale
// is not a field: its sixteen bits are scattered four at a time across the top
// nibbles of the four scale halfwords, which also carry the 3-bit sub-scales.
__device__ __forceinline__ float iq1m_scale(const unsigned char* base) {
    unsigned short sc[4];
    memcpy(sc, base + 48, 8);
    const unsigned short bits = (unsigned short)(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
        (sc[3] & 0xf000));
    return __half2float(*((const __half*)&bits));
}

__device__ __forceinline__ float iq1m_value(
    const unsigned char* packed, int absolute
) {
    // 56 bytes per 256 values: qs[32] qh[16] scales[8]. Every group of eight
    // weights takes an 11-bit grid index -- eight bits from qs, three from the
    // nibble of qh that covers it -- plus a per-group sign for a +-0.125 delta.
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 56;
    unsigned short sc[4];
    memcpy(sc, base + 48, 8);
    const int ib = within / 32;
    const int group = (within % 32) / 8;
    const int lane = within & 7;
    const unsigned char qh = base[32 + ib * 2 + group / 2];
    const unsigned int index = (unsigned int)base[ib * 4 + group] |
        (((unsigned int)qh << ((group & 1) ? 4 : 8)) & 0x700u);
    const float delta = (qh & ((group & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
    // The two halves of a sub-block carry separate 3-bit scales, three bits
    // apart in the halfword covering this pair of sub-blocks.
    const int shift = 6 * (ib & 1) + (group < 2 ? 0 : 3);
    const float scale =
        iq1m_scale(base) * (float)(2 * ((sc[ib / 2] >> shift) & 7) + 1);
    const float weight =
        (float)(signed char)((kIq1sGrid[index] >> (8 * lane)) & 0xffULL);
    return scale * (weight + delta);
}

extern "C" __global__
void iq1m_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq1m_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

// IQ1_S, the format the grid above is named for: 50 bytes per 256 values --
// d(2) qs[32] qh[8*2]. Everything IQ1_M scatters, this one keeps in one place.
// A single halfword of qh covers a whole 32-value group: the high three bits of
// each of its four 11-bit grid indices (the low eight are qs bytes), the
// group's 3-bit scale multiplier in bits 12-14, and in bit 15 the sign of the
// +-0.125 delta every weight in the group carries. A group is therefore exactly
// one Q8 activation block, which is what lets the DP4A path below be a plain
// *_MIN kernel rather than needing half-group scales.
__device__ __forceinline__ float iq1s_value(
    const unsigned char* packed, int absolute
) {
    const int block = absolute / 256;
    const int within = absolute & 255;
    const unsigned char* base = packed + block * 50;
    const int group = within >> 5, part = (within >> 3) & 3, lane = within & 7;
    unsigned short qh;
    memcpy(&qh, base + 34 + group * 2, 2);
    const unsigned int index = (unsigned int)base[2 + group * 4 + part] |
        ((((unsigned int)qh >> (3 * part)) & 7u) << 8);
    const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    const float scale = __half2float(*((const __half*)base)) *
        (float)(2 * ((qh >> 12) & 7) + 1);
    const float weight =
        (float)(signed char)((kIq1sGrid[index] >> (8 * lane)) & 0xffULL);
    return scale * (weight + delta);
}

extern "C" __global__
void iq1s_matvec_transposed(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x)
        partial += iq1s_value(packed, row * input_size + input) * vector[input];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// IQ1_S against a Q8-quantized activation.
//
// The grid entries are eight int8 in a 64-bit word, so a group of eight weights
// is two DP4A operands with no unpacking at all -- no sign expansion, no nibble
// split. The one thing in the way is the delta: a weight reconstructs as
// scale * (grid_value + delta) with delta = +-1/8 shared by the group, and a
// constant added to every weight is exactly what the _MIN kernels' offset
// handles, at the price of the activation sums and their smaller tile.
//
// It does not have to be paid, because the delta is a power of two. Grid values
// are -1, 0 and 1, so 8*grid_value +- 1 is an int8 in [-9, 9] and
//
//     scale * (grid_value + delta) == (scale / 8) * (8 * grid_value +- 1)
//
// exactly -- no rounding anywhere, the scale only loses an exponent. So the
// delta folds into the int8 weights themselves and IQ1_S becomes a symmetric
// format for every kernel downstream, which is what puts it on the wide MMQ
// tile instead of the 32-row _MIN one.
//
// Folding is three SIMD byte operations: shift the packed grid word left by
// three (masking off the bits that would cross into the next byte, since there
// is no byte-wise shift) and add the sign word, +1 or -1 per byte.
__device__ __forceinline__ void iq1s_q8_words(
    const unsigned long long entry, const unsigned int signs,
    int* low, int* high
) {
    *low = (int)__vadd4(((unsigned int)entry << 3) & 0xf8f8f8f8u, signs);
    *high = (int)__vadd4(
        ((unsigned int)(entry >> 32) << 3) & 0xf8f8f8f8u, signs);
}

__device__ __forceinline__ float iq1s_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 50;
    unsigned short qh;
    unsigned int codes;
    memcpy(&qh, base + 34 + group * 2, 2);
    memcpy(&codes, base + 2 + group * 4, 4);

    const int4* activation_vectors = (const int4*)(vector + linear_group * 32);
    const int4 activation_low = activation_vectors[0];
    const int4 activation_high = activation_vectors[1];
    const int acts[8] = {
        activation_low.x, activation_low.y,
        activation_low.z, activation_low.w,
        activation_high.x, activation_high.y,
        activation_high.z, activation_high.w};

    // -1 per byte when the group's delta is negative, +1 per byte when it is
    // positive. 0xffffffff is that -1: __vadd4 wraps per byte.
    const unsigned int signs = (qh & 0x8000) ? 0xffffffffu : 0x01010101u;
    int dot = 0;
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        const unsigned long long entry = kIq1sGrid[
            ((codes >> (8 * part)) & 255u) |
            ((((unsigned int)qh >> (3 * part)) & 7u) << 8)];
        int low, high;
        iq1s_q8_words(entry, signs, &low, &high);
        dot = __dp4a(low, acts[part * 2], dot);
        dot = __dp4a(high, acts[part * 2 + 1], dot);
    }
    const float scale = __half2float(*((const __half*)base)) *
        (float)(2 * ((qh >> 12) & 7) + 1) * 0.125f;
    return scale * (float)dot * __half2float(vector_scales[linear_group]);
}

COLIBRI_Q8_MATVEC(iq1s_q8_matvec_transposed_warp, iq1s_q8_group, 50)

// Batched twin of iq1s_q8_group. A group of 32 is one scale, so both halves of
// the Q8 block get it.
__device__ __forceinline__ void iq1s_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 50;
    unsigned short qh;
    unsigned int codes;
    memcpy(&qh, base + 34 + group * 2, 2);
    memcpy(&codes, base + 2 + group * 4, 4);
    const unsigned int signs = (qh & 0x8000) ? 0xffffffffu : 0x01010101u;
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        const unsigned long long entry = kIq1sGrid[
            ((codes >> (8 * part)) & 255u) |
            ((((unsigned int)qh >> (3 * part)) & 7u) << 8)];
        iq1s_q8_words(entry, signs, &words[part * 2], &words[part * 2 + 1]);
    }
    const float scale = __half2float(*((const __half*)base)) *
        (float)(2 * ((qh >> 12) & 7) + 1) * 0.125f;
    *scale_low = scale;
    *scale_high = scale;
}

COLIBRI_Q8_MATVEC_ROWS(iq1s_q8_matvec_transposed_rows, iq1s_q8_decode, 50)
COLIBRI_Q8_MATMUL_TILED(iq1s_q8_matmul_tiled, iq1s_q8_decode, 50)
COLIBRI_Q8_MMQ(iq1s_q8_mmq, iq1s_q8_decode, 50)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// IQ1_M against a Q8-quantized activation.
//
// The same fold as IQ1_S, since it is the same grid and the same +-1/8 delta.
// Two things differ, and both happen to fit what is already here. The sign of
// the delta is per group of eight rather than per group of 32, so the sign word
// is picked per grid entry inside the loop instead of once. And the 3-bit
// sub-scales cover 16 values each, which is exactly the half-block split
// scale_low/scale_high already carries for IQ2_S -- so the format needs no new
// kernel shape at all, only this decoder.
__device__ __forceinline__ void iq1m_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int ib = linear_group & 7;
    const unsigned char* base = row_data + block * 56;
    unsigned short sc[4];
    memcpy(sc, base + 48, 8);
    // 0.125 is the fold: the weights below carry 8*grid_value +- 1.
    const float d = iq1m_scale(base) * 0.125f;
    const int shift = 6 * (ib & 1);
    *scale_low = d * (float)(2 * ((sc[ib / 2] >> shift) & 7) + 1);
    *scale_high = d * (float)(2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1);
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        const unsigned int qh = base[32 + ib * 2 + group / 2];
        const unsigned int index = (unsigned int)base[ib * 4 + group] |
            ((qh << ((group & 1) ? 4 : 8)) & 0x700u);
        const unsigned int signs =
            (qh & ((group & 1) ? 0x80u : 0x08u)) ? 0xffffffffu : 0x01010101u;
        iq1s_q8_words(kIq1sGrid[index], signs,
                      &words[group * 2], &words[group * 2 + 1]);
    }
}

__device__ __forceinline__ float iq1m_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    int words[8];
    float scale_low, scale_high;
    iq1m_q8_decode(row_data, linear_group, words, &scale_low, &scale_high);

    const int4* activation_vectors = (const int4*)(vector + linear_group * 32);
    const int4 activation_low = activation_vectors[0];
    const int4 activation_high = activation_vectors[1];
    const int acts[8] = {
        activation_low.x, activation_low.y,
        activation_low.z, activation_low.w,
        activation_high.x, activation_high.y,
        activation_high.z, activation_high.w};

    int dot_low = 0, dot_high = 0;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int dot = 0;
        dot = __dp4a(words[step * 2], acts[step * 2], dot);
        dot = __dp4a(words[step * 2 + 1], acts[step * 2 + 1], dot);
        if (step < 2) dot_low += dot; else dot_high += dot;
    }
    return ((float)dot_low * scale_low + (float)dot_high * scale_high)
        * __half2float(vector_scales[linear_group]);
}

COLIBRI_Q8_MATVEC(iq1m_q8_matvec_transposed_warp, iq1m_q8_group, 56)
COLIBRI_Q8_MATVEC_ROWS(iq1m_q8_matvec_transposed_rows, iq1m_q8_decode, 56)
COLIBRI_Q8_MATMUL_TILED(iq1m_q8_matmul_tiled, iq1m_q8_decode, 56)
COLIBRI_Q8_MMQ(iq1m_q8_mmq, iq1m_q8_decode, 56)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// IQ2_XS against a Q8-quantized activation. Structurally this format is IQ2_S
// with IQ2_XXS's signs: the 4-bit scale nibble per 16 values is IQ2_S's, so the
// two halves of a 32-value Q8 block must accumulate separately, while the sign
// selector is the 7-bit codebook index IQ2_XXS uses rather than a literal byte.
// So the body below is iq2s_q8_group with the sign line swapped.
//
// The 9-bit grid index and the 7-bit sign selector share one uint16, which is
// why this needs no qh: entry & 511 picks the codebook row and entry >> 9 the
// sign pattern, both from the same load.
__device__ __forceinline__ float iq2xs_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 74;

    // 74 bytes per super-block leaves qs only 2-byte aligned, so the four
    // uint16 of this group come through memcpy and the compiler folds them
    // into whatever width the alignment actually permits.
    unsigned short entries[4];
    memcpy(entries, base + 2 + group * 8, 8);

    const int4* activation_vectors = (const int4*)(vector + linear_group * 32);
    const int4 activation_low = activation_vectors[0];
    const int4 activation_high = activation_vectors[1];
    const int acts[8] = {
        activation_low.x, activation_low.y,
        activation_low.z, activation_low.w,
        activation_high.x, activation_high.y,
        activation_high.z, activation_high.w};

    // Group g spans 16-value scale groups 2g and 2g+1, which are the two
    // nibbles of a single scale byte.
    const unsigned int scale_byte = base[66 + group];

    int dot_low = 0, dot_high = 0;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const unsigned int entry = entries[step];
        const unsigned long long pattern = kIq2xsGrid[entry & 511];
        const unsigned int sign_word =
            (unsigned int)kIq2xxsSigns[entry >> 9] * 0x01010101u;
        const int masks_first = __vcmpne4(sign_word & 0x08040201u, 0);
        const int masks_second = __vcmpne4(sign_word & 0x80402010u, 0);
        const int weights_first = __vsub4(
            (int)(unsigned int)pattern ^ masks_first, masks_first);
        const int weights_second = __vsub4(
            (int)(unsigned int)(pattern >> 32) ^ masks_second, masks_second);
        int dot = 0;
        dot = __dp4a(weights_first, acts[step * 2], dot);
        dot = __dp4a(weights_second, acts[step * 2 + 1], dot);
        if (step < 2) dot_low += dot; else dot_high += dot;
    }
    const float scale_low = 0.5f + (float)(scale_byte & 15u);
    const float scale_high = 0.5f + (float)((scale_byte >> 4) & 15u);
    return ((float)dot_low * scale_low + (float)dot_high * scale_high) * 0.25f
        * __half2float(*((const __half*)base))
        * __half2float(vector_scales[linear_group]);
}

COLIBRI_Q8_MATVEC(iq2xs_q8_matvec_transposed_warp, iq2xs_q8_group, 74)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Batched twin of iq2xs_q8_group, for prefill. Same split as IQ2_S: the scales
// are held back because the two 16-value halves take different nibbles.
__device__ __forceinline__ void iq2xs_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int group = linear_group & 7;
    const unsigned char* base = row_data + block * 74;
    unsigned short entries[4];
    memcpy(entries, base + 2 + group * 8, 8);
    const unsigned int scale_byte = base[66 + group];
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const unsigned int entry = entries[step];
        const unsigned long long pattern = kIq2xsGrid[entry & 511];
        const unsigned int sign_word =
            (unsigned int)kIq2xxsSigns[entry >> 9] * 0x01010101u;
        const int masks_first = __vcmpne4(sign_word & 0x08040201u, 0);
        const int masks_second = __vcmpne4(sign_word & 0x80402010u, 0);
        words[step * 2] = __vsub4(
            (int)(unsigned int)pattern ^ masks_first, masks_first);
        words[step * 2 + 1] = __vsub4(
            (int)(unsigned int)(pattern >> 32) ^ masks_second, masks_second);
    }
    const float block_scale = __half2float(*((const __half*)base)) * 0.25f;
    *scale_low = (0.5f + (float)(scale_byte & 15u)) * block_scale;
    *scale_high = (0.5f + (float)((scale_byte >> 4) & 15u)) * block_scale;
}

COLIBRI_Q8_MATVEC_ROWS(iq2xs_q8_matvec_transposed_rows, iq2xs_q8_decode, 74)
COLIBRI_Q8_MATMUL_TILED(iq2xs_q8_matmul_tiled, iq2xs_q8_decode, 74)
COLIBRI_Q8_MMQ(iq2xs_q8_mmq, iq2xs_q8_decode, 74)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
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

// IQ2_XXS: 66-byte super-blocks of 256; each octet is one grid pattern of
// unsigned magnitudes with a 7-bit sign selector, scaled by the group's
// 4-bit scale. The unsloth dynamic quants mix this into IQ1_S expert sets on
// their sensitive layers, so the grouped path has to speak it too.
__device__ __forceinline__ void iq2xxs_octet(
    const unsigned char* packed, int block, int octet, float* out
) {
    const unsigned char* base = packed + block * 66;
    const float d = __half2float(*((const __half*)base));
    const int group = octet >> 2;
    const int quad = octet & 3;
    unsigned int low, high;
    memcpy(&low, base + 2 + group * 8, 4);
    memcpy(&high, base + 2 + group * 8 + 4, 4);
    const float scale = d * (0.5f + (float)(high >> 28)) * 0.25f;
    const unsigned char signs = kIq2xxsSigns[(high >> (7 * quad)) & 127];
    const unsigned long long pattern = kIq2xxsGrid[(low >> (8 * quad)) & 255];
    for (int k = 0; k < 8; ++k) {
        const float value = scale *
            (float)((unsigned char)((pattern >> (8 * k)) & 0xffULL));
        out[k] = ((signs >> k) & 1) ? -value : value;
    }
}

// IQ4_NL: 18-byte blocks of 32 (d + 16 nibble bytes), no super-block. The
// grouped macro indexes octets in 256-element terms, so the 32-block and the
// nibble half are re-derived from the absolute octet here. Rows only need to
// be a multiple of 32 -- qwen4exp's 640-wide expert down rows included.
__device__ __forceinline__ void iq4nl_octet(
    const unsigned char* packed, int block, int octet, float* out
) {
    const int absolute_octet = block * 32 + octet;
    const unsigned char* base = packed + (absolute_octet >> 2) * 18;
    const float d = __half2float(*((const __half*)base));
    const int part = absolute_octet & 3;
    const unsigned char* quants = base + 2 + (part & 1) * 8;
    const int high = part >> 1;
    for (int k = 0; k < 8; ++k) {
        const unsigned char byte = quants[k];
        out[k] = d * (float)kIq4nlValues[high ? (byte >> 4) : (byte & 15)];
    }
}

// IQ1_S: 50-byte super-blocks of 256; each 32-value group carries a 3-bit odd
// scale and a signed +-0.125 delta, each octet is one 2048-entry grid entry
// indexed by 8 bits from qs and 3 from the group halfword.
__device__ __forceinline__ void iq1s_octet(
    const unsigned char* packed, int block, int octet, float* out
) {
    const unsigned char* base = packed + block * 50;
    const float d = __half2float(*((const __half*)base));
    const int group = octet >> 2;
    const int part = octet & 3;
    unsigned short qh;
    memcpy(&qh, base + 34 + group * 2, 2);
    const float scale = d * (float)(2 * ((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    const unsigned int index =
        (unsigned int)base[2 + group * 4 + part] |
        (((unsigned int)(qh >> (3 * part)) & 7u) << 8);
    const unsigned long long entry = kIq1sGrid[index];
    for (int k = 0; k < 8; ++k) {
        out[k] = scale *
            ((float)(signed char)((entry >> (8 * k)) & 0xffULL) + delta);
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(void prefix##_grouped_swiglu_rows(                                              \
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
COLIBRI_IQ_GROUPED(iq1s, iq1s_octet)
COLIBRI_IQ_GROUPED(iq4nl, iq4nl_octet)
COLIBRI_IQ_GROUPED(iq2xxs, iq2xxs_octet)

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

// Q2_K against a Q8-blocked activation. A 32-value Q8 block spans two of
// Q2_K's 16-element scale groups, so each block carries two (scale, min)
// pairs and the affine reconstruction splits into two halves of four DP4A
// pairs. Both halves read the same 2-bit field of the same 32 qs bytes.
__device__ __forceinline__ float q2k_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    float partial = 0.0f;

        const int block = linear_group >> 3;
        const int sub_block = linear_group & 7;
        const int half = sub_block >> 2;
        const int group = sub_block & 3;
        const unsigned char* base = row_data + block * 84;
        const unsigned char* quants = base + 16 + half * 32;
        const int shift = 2 * group;
        const signed char* activations = vector + linear_group * 32;

        // An 84-byte super-block leaves qs only 4-byte aligned, so the weights
        // stay on 32-bit loads; the activation block is 32-byte aligned and
        // moves to int4, cutting the per-group load count from 16 to 10.
        const int4* activation_vectors = (const int4*)activations;
        int dot[2] = {0, 0}, total[2] = {0, 0};
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int4 activation_quads = activation_vectors[part];
            const int acts[4] = {activation_quads.x, activation_quads.y,
                                 activation_quads.z, activation_quads.w};
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                unsigned int word;
                memcpy(&word, quants + (part * 4 + step) * 4, 4);
                const int weights = (int)((word >> shift) & 0x03030303u);
                dot[part] = __dp4a(weights, acts[step], dot[part]);
                total[part] = __dp4a(0x01010101, acts[step], total[part]);
            }
        }
        const float d = __half2float(*((const __half*)(base + 80)));
        const float dmin = __half2float(*((const __half*)(base + 82)));
        const unsigned char* scale_bytes = base + half * 8 + group * 2;
        float sum = 0.0f;
        #pragma unroll
        for (int index = 0; index < 2; ++index) {
            const unsigned char scale_byte = scale_bytes[index];
            sum += d * (float)(scale_byte & 15) * (float)dot[index]
                 - dmin * (float)(scale_byte >> 4) * (float)total[index];
        }
        partial += __half2float(vector_scales[linear_group]) * sum;
    return partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(COLIBRI_Q8_MATVEC(q2k_q8_matvec_transposed_warp, q2k_q8_group, 84)
COLIBRI_Q8_LM_HEAD(q2k_q8_lm_head_argmax_warp, q2k_q8_group, 84)

// q2k_q8_group with the dot lifted out. Unlike Q4_K and Q5_K the sub-block is
// 16 values, so a group spans two of them and the halves carry different
// scales and minima -- which is what the low/high split in the decode contract
// is for.
__device__ __forceinline__ void q2k_q8_decode(
    const unsigned char* row_data, const int linear_group, int* words,
    float* scale_low, float* scale_high,
    float* offset_low, float* offset_high) {
    const int block = linear_group >> 3;
    const int sub_block = linear_group & 7;
    const int half = sub_block >> 2;
    const int group = sub_block & 3;
    const unsigned char* base = row_data + block * 84;
    const unsigned char* quants = base + 16 + half * 32;
    const int shift = 2 * group;
    #pragma unroll
    for (int quad = 0; quad < 8; ++quad) {
        unsigned int word;
        memcpy(&word, quants + quad * 4, 4);
        words[quad] = (int)((word >> shift) & 0x03030303u);
    }
    const float d = __half2float(*((const __half*)(base + 80)));
    const float dmin = __half2float(*((const __half*)(base + 82)));
    const unsigned char* scale_bytes = base + half * 8 + group * 2;
    *scale_low = d * (float)(scale_bytes[0] & 15);
    *offset_low = dmin * (float)(scale_bytes[0] >> 4);
    *scale_high = d * (float)(scale_bytes[1] & 15);
    *offset_high = dmin * (float)(scale_bytes[1] >> 4);
}

COLIBRI_Q8_MATVEC_ROWS_MIN(q2k_q8_matvec_transposed_rows, q2k_q8_decode, 84)
COLIBRI_Q8_MMQ_MIN(q2k_q8_mmq, q2k_q8_decode, 84)


// Q3_K against a Q8-blocked activation. The signed 3-bit weight is a 2-bit low
// part minus 4 unless the hmask bit is set, so a whole quad reconstructs with
// byte-wise SIMD -- low + 4*mask_bit - 4 -- and feeds DP4A directly. Q3_K has
// no min term, so only the weight dot is needed. Like Q2_K a 32-value Q8 block
// straddles two 16-element scale groups.
__device__ __forceinline__ float q3k_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    float partial = 0.0f;

        const int block = linear_group >> 3;
        const int sub_block = linear_group & 7;
        const int half = sub_block >> 2;
        const int group = sub_block & 3;
        const unsigned char* base = row_data + block * 110;
        const unsigned char* quants = base + 32 + half * 32;
        const int shift = 2 * group;
        const int mask_shift = half * 4 + group;
        const signed char* activations = vector + linear_group * 32;

        // 110-byte super-blocks keep qs and hmask on 32-bit loads; only the
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        // 32-byte aligned activation block can widen to int4.
        const int4* activation_vectors = (const int4*)activations;
        int dot[2] = {0, 0};
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int4 activation_quads = activation_vectors[part];
            const int acts[4] = {activation_quads.x, activation_quads.y,
                                 activation_quads.z, activation_quads.w};
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int quad = part * 4 + step;
                unsigned int word, mask_word;
                memcpy(&word, quants + quad * 4, 4);
                memcpy(&mask_word, base + quad * 4, 4);
                const unsigned int low = (word >> shift) & 0x03030303u;
                const unsigned int high =
                    ((mask_word >> mask_shift) & 0x01010101u) << 2;
                const int weights =
                    __vsub4((int)__vadd4((int)low, (int)high), 0x04040404);
                dot[part] = __dp4a(weights, acts[step], dot[part]);
            }
        }
        const float d = __half2float(*((const __half*)(base + 108)));
        const int scale_base = half * 8 + group * 2;
        const float sum =
            (float)(q3k_scale(base + 96, scale_base) - 32) * (float)dot[0]
          + (float)(q3k_scale(base + 96, scale_base + 1) - 32) * (float)dot[1];
        partial += __half2float(vector_scales[linear_group]) * d * sum;
    return partial;
}

COLIBRI_Q8_MATVEC(q3k_q8_matvec_transposed_warp, q3k_q8_group, 110)
COLIBRI_Q8_LM_HEAD(q3k_q8_lm_head_argmax_warp, q3k_q8_group, 110)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// The same unpacking as q3k_q8_group, stopping one step earlier: the decoded
// weights come back in registers instead of being dotted against one
// activation vector on the spot.
//
// That split is the whole difference between decode-shaped and prompt-shaped
// work. The fused form re-reads the weight matrix for every token, which makes
// prefill a memory-bound repeat of decode -- measured at 384 GiB/s of weight
// traffic on a 27B, or 45 tok/s. The batched kernels this feeds reuse one
// decode across 8, 32 or 64 tokens.
//
// Only the symmetric K-quants can be served this way. Q2_K, Q4_K and Q5_K
// reconstruct as `d*scale*q - dmin*min`, and that per-sub-block offset needs a
// term the dot product here has no place for; Q3_K and Q6_K are `d*scale*q`
// and fit the contract exactly, two 16-element sub-block scales to a group.
__device__ __forceinline__ void q3k_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int sub_block = linear_group & 7;
    const int half = sub_block >> 2;
    const int group = sub_block & 3;
    const unsigned char* base = row_data + block * 110;
    const unsigned char* quants = base + 32 + half * 32;
    const int shift = 2 * group;
    const int mask_shift = half * 4 + group;
    #pragma unroll
    for (int quad = 0; quad < 8; ++quad) {
        unsigned int word, mask_word;
        memcpy(&word, quants + quad * 4, 4);
        memcpy(&mask_word, base + quad * 4, 4);
        const unsigned int low = (word >> shift) & 0x03030303u;
        // A set mask bit is the high bit of a 3-bit code, and the -4 below is
        // what makes the result signed.
        const unsigned int high = ((mask_word >> mask_shift) & 0x01010101u) << 2;
        words[quad] = __vsub4((int)__vadd4((int)low, (int)high), 0x04040404);
    }
    const float d = __half2float(*((const __half*)(base + 108)));
    const int scale_base = half * 8 + group * 2;
    *scale_low = d * (float)(q3k_scale(base + 96, scale_base) - 32);
    *scale_high = d * (float)(q3k_scale(base + 96, scale_base + 1) - 32);
}

COLIBRI_Q8_MATVEC_ROWS(q3k_q8_matvec_transposed_rows, q3k_q8_decode, 110)
COLIBRI_Q8_MATMUL_TILED(q3k_q8_matmul_tiled, q3k_q8_decode, 110)
COLIBRI_Q8_MMQ(q3k_q8_mmq, q3k_q8_decode, 110)


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

// Q6_K against a Q8-blocked activation. A 32-value Q8 block is one (half, lane)
// pair, so it reads 32 consecutive ql bytes and the 32 shared qh bytes with a
// fixed nibble and bit position. The quant is a plain 6-bit value biased by 32,
// so the whole quad reconstructs with an OR and one byte-wise subtract. Q6_K has
// no min term; the two 16-element scale groups split the block in half.
__device__ __forceinline__ float q6k_q8_group(
    const unsigned char* row_data,
    const signed char* vector,
    const __half* vector_scales,
    const int linear_group
) {
    float partial = 0.0f;

        const int block = linear_group >> 3;
        const int sub_block = linear_group & 7;
        const int half = sub_block >> 2;
        const int lane_group = sub_block & 3;
        const unsigned char* base = row_data + block * 210;
        const unsigned char* lows =
            base + half * 64 + ((lane_group & 1) ? 32 : 0);
        const unsigned char* highs = base + 128 + half * 32;
        const int shift = (lane_group >> 1) * 4;
        const int bit_shift = lane_group * 2;
        const signed char* activations = vector + linear_group * 32;

        // 210-byte super-blocks are only 2-byte aligned, so ql/qh stay on
        // 32-bit loads and just the activation block widens to int4.
        const int4* activation_vectors = (const int4*)activations;
        int dot[2] = {0, 0};
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int4 activation_quads = activation_vectors[part];
            const int acts[4] = {activation_quads.x, activation_quads.y,
                                 activation_quads.z, activation_quads.w};
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int quad = part * 4 + step;
                unsigned int word, high_word;
                memcpy(&word, lows + quad * 4, 4);
                memcpy(&high_word, highs + quad * 4, 4);
                const unsigned int weights = ((word >> shift) & 0x0f0f0f0fu)
                    | (((high_word >> bit_shift) & 0x03030303u) << 4);
                const int biased = __vsub4((int)weights, 0x20202020);
                dot[part] = __dp4a(biased, acts[step], dot[part]);
            }
        }
        const float d = __half2float(*((const __half*)(base + 208)));
        const signed char* scales = (const signed char*)(base + 192);
        const int scale_base = half * 8 + lane_group * 2;
        partial += __half2float(vector_scales[linear_group]) * d
            * ((float)scales[scale_base] * (float)dot[0]
               + (float)scales[scale_base + 1] * (float)dot[1]);
    return partial;
}

COLIBRI_Q8_MATVEC(q6k_q8_matvec_transposed_warp, q6k_q8_group, 210)
COLIBRI_Q8_LM_HEAD(q6k_q8_lm_head_argmax_warp, q6k_q8_group, 210)

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// q6k_q8_group with the dot product lifted out; see q3k_q8_decode for why the
// split matters and which K-quants can be served by it.
__device__ __forceinline__ void q6k_q8_decode(
    const unsigned char* row_data, const int linear_group,
    int* words, float* scale_low, float* scale_high) {
    const int block = linear_group >> 3;
    const int sub_block = linear_group & 7;
    const int half = sub_block >> 2;
    const int lane_group = sub_block & 3;
    const unsigned char* base = row_data + block * 210;
    const unsigned char* lows = base + half * 64 + ((lane_group & 1) ? 32 : 0);
    const unsigned char* highs = base + 128 + half * 32;
    const int shift = (lane_group >> 1) * 4;
    const int bit_shift = lane_group * 2;
    #pragma unroll
    for (int quad = 0; quad < 8; ++quad) {
        unsigned int word, high_word;
        memcpy(&word, lows + quad * 4, 4);
        memcpy(&high_word, highs + quad * 4, 4);
        const unsigned int weights = ((word >> shift) & 0x0f0f0f0fu)
            | (((high_word >> bit_shift) & 0x03030303u) << 4);
        words[quad] = __vsub4((int)weights, 0x20202020);
    }
    const float d = __half2float(*((const __half*)(base + 208)));
    const signed char* scales = (const signed char*)(base + 192);
    const int scale_base = half * 8 + lane_group * 2;
    *scale_low = d * (float)scales[scale_base];
    *scale_high = d * (float)scales[scale_base + 1];
}

COLIBRI_Q8_MATVEC_ROWS(q6k_q8_matvec_transposed_rows, q6k_q8_decode, 210)
COLIBRI_Q8_MATMUL_TILED(q6k_q8_matmul_tiled, q6k_q8_decode, 210)
COLIBRI_Q8_MMQ(q6k_q8_mmq, q6k_q8_decode, 210)


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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    const int input_size, const int output_size
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
void q8_grouped_swiglu(
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
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float gate_scale = __half2float(
            *((const __half*)(gate_packed + block * 34))
        );
        const float up_scale = __half2float(
            *((const __half*)(up_packed + block * 34))
        );
        const signed char gate_value = *((const signed char*)(
            gate_packed + block * 34 + 2 + within
        ));
        const signed char up_value = *((const signed char*)(
            up_packed + block * 34 + 2 + within
        ));
        const float input_value = vector[input];
        gate += ((float)gate_value * gate_scale) * input_value;
        up += ((float)up_value * up_scale) * input_value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[expert * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
}

extern "C" __global__
void q8_grouped_swiglu_rows(
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
        const int absolute = row * input_size + input;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float gate_scale = __half2float(
            *((const __half*)(gate_packed + block * 34))
        );
        const float up_scale = __half2float(
            *((const __half*)(up_packed + block * 34))
        );
        const signed char gate_value = *((const signed char*)(
            gate_packed + block * 34 + 2 + within
        ));
        const signed char up_value = *((const signed char*)(
            up_packed + block * 34 + 2 + within
        ));
        const float input_value = vector[input];
        gate += ((float)gate_value * gate_scale) * input_value;
        up += ((float)up_value * up_scale) * input_value;
    }
    gate = block_reduce_sum(gate);
    up = block_reduce_sum(up);
    if (threadIdx.x == 0)
        activated[route * output_size + row] =
            (gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))))) * up;
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    return __uint_as_float(bits | ((unsigned int)(val & 8) << 28));
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    const int row = scaled_block / blocks_per_row;
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

// Concatenate each routed down matrix along K. Multiplying that matrix by the
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        partial += combined;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] += partial;
}

extern "C" __global__
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
    const int kv_heads, const int head_dim, const int position, const int capacity
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// One block per head, so the weighted sum is accumulated in the rotated domain
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA() { kv_values_impl<__nv_bfloat16>(scores, values, output, heads, kv_heads, head_dim, tokens, capacity); }
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Flash-blocked variant of the prefill softmax: the visible prefix is walked
// in position blocks so the materialized score tile is bounded by the block,
// not by the context -- which is what let the caller's 16-row query tile
// collapse to 3 rows at 70k tokens and stream the whole cache per 3 queries.
// Per (head, query row) the kernel keeps a running maximum M and denominator
// S in `state` ([tile_row][head][2]) across blocks, writes this block's
// probabilities already on the new-M scale (so the PV GEMM accumulates
// directly), and leaves exp(M_old - M_new) in `rescale` for the accumulator
// fix-up that must run before that GEMM. The final normalize (divide by S)
// happens in the unpack. Same numerics as one big softmax, reassociated.
extern "C" __global__ void kv_attention_prefill_block_softmax_f16(
    const float* scores, __half* probabilities, float* state, float* rescale,
    const int tile_start, const int tile_rows,
    const int heads, const int kv_heads,
    const int block_start, const int block_tokens,
    const int base_position
) {
    const int head = blockIdx.x;
    const int row_index = blockIdx.y;
    if (head >= heads || row_index >= tile_rows || block_tokens <= 0) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const int group_head = head - kv_head * group;
    const int column = row_index * group + group_head;
    const float* row = scores
        + (long long)kv_head * block_tokens * tile_rows * group
        + (long long)column * block_tokens;
    __half* output = probabilities
        + (long long)kv_head * block_tokens * tile_rows * group
        + (long long)column * block_tokens;
    const int visible_global = base_position + tile_start + row_index + 1;
    const int visible = min(block_tokens, visible_global - block_start);
    const int slot = row_index * heads + head;
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
    const bool first = block_start == 0;
    const float previous_maximum = first ? -3.402823466e+38F : state[slot * 2];
    const float previous_sum = first ? 0.0f : state[slot * 2 + 1];
    const float new_maximum = fmaxf(previous_maximum, reduction[0]);
    float denominator = 0.0f;
    for (int token = threadIdx.x; token < visible; token += blockDim.x) {
        const float probability = __expf(row[token] - new_maximum);
        output[token] = __float2half(probability);
        denominator += probability;
    }
    for (int token = max(visible, 0) + threadIdx.x; token < block_tokens;
         token += blockDim.x)
        output[token] = __float2half(0.0f);
    reduction[threadIdx.x] = denominator;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // A fully masked block (visible <= 0) keeps the running state and
        // contributes zero probabilities; the correction is exp(M - M) = 1.
        const float correction =
            first ? 0.0f : __expf(previous_maximum - new_maximum);
        state[slot * 2] = new_maximum;
        state[slot * 2 + 1] = previous_sum * correction + reduction[0];
        rescale[slot] = correction;
    }
}

// Multiply the packed flash accumulator by this block's exp(M_old - M_new)
// before the PV GEMM adds the block's contribution on the new scale.
extern "C" __global__ void qwen_attention_prefill_rescale(
    float* packed, const float* rescale, const int tile_rows,
    const int heads, const int kv_heads, const int head_dim
) {
    const int elements = tile_rows * heads * head_dim;
    const int group = heads / kv_heads;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x) {
        const int dimension = index % head_dim;
        const int column = index / head_dim % (tile_rows * group);
        const int kv_head = index / (head_dim * tile_rows * group);
        const int row = column / group;
        const int head = kv_head * group + column - row * group;
        packed[((long long)kv_head * tile_rows * group + column) * head_dim
               + dimension] *= rescale[row * heads + head];
    }
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

// Flash-blocked unpack: the accumulator holds sum(p * V) with p on the final
// maximum's scale but unnormalized; divide by the running denominator from
// the block-softmax state while the value is in a register.
template<bool GATE>
__device__ void qwen_attention_prefill_unpack_norm_impl(
    const float* packed, const float* gates, const float* state, float* output,
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
        const float denominator = state[(row * heads + head) * 2 + 1];
        const float value =
            packed[((long long)kv_head * tile_rows * group + column) * head_dim
                   + dimension] / denominator;
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
extern "C" __global__ void qwen_attention_prefill_unpack_gate_norm(
    const float* packed, const float* gates, const float* state, float* output,
    const int tile_start, const int tile_rows, const int heads,
    const int kv_heads, const int head_dim
) {
    qwen_attention_prefill_unpack_norm_impl<true>(
        packed, gates, state, output, tile_start, tile_rows, heads, kv_heads,
        head_dim);
}
extern "C" __global__ void qwen_attention_prefill_unpack_norm(
    const float* packed, const float* gates, const float* state, float* output,
    const int tile_start, const int tile_rows, const int heads,
    const int kv_heads, const int head_dim
) {
    qwen_attention_prefill_unpack_norm_impl<false>(
        packed, gates, state, output, tile_start, tile_rows, heads, kv_heads,
        head_dim);
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
R"COLIBRI_CUDA(
template<typename KT, typename VT, int maximum_head_dim = 128>
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
    // One register slot per 32-lane sweep of the head dimension. Templating
    // this rather than fixing it at 128 is what lets a 256-dim head (Qwen3.6)
    // use the split-K path at all; it used to fall through to cuBLAS, which
    // reaches only ~35% of this card's read bandwidth on so skinny a GEMV.
    constexpr int parts = maximum_head_dim / 32;
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
    float accumulator[parts];
    #pragma unroll
    for (int part = 0; part < parts; ++part) accumulator[part] = 0.0f;
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    for (int token = tile_begin + warp;
         token < tile_end;
         token += warp_count) {
        int slot = first + token;
        if (slot >= capacity) slot -= capacity;
        const long long row = (long long)kv_head * capacity + slot;
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        float dot = 0.0f;
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
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
        for (int part = 0; part < parts; ++part) {
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
    for (int part = 0; part < parts; ++part) {
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
COLIBRI_LOWBIT_MATVEC_WARP(iq2xxs_matvec_transposed_warp, iq2xxs_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq3xxs_matvec_transposed_warp, iq3xxs_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq2s_matvec_transposed_warp, iq2s_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq3s_matvec_transposed_warp, iq3s_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq2xs_matvec_transposed_warp, iq2xs_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq4xs_matvec_transposed_warp, iq4xs_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq1m_matvec_transposed_warp, iq1m_value)
COLIBRI_LOWBIT_MATVEC_WARP(iq1s_matvec_transposed_warp, iq1s_value)
#undef COLIBRI_LOWBIT_MATVEC_WARP

extern "C" __global__ void q5k_matvec_transposed_warp(
    const unsigned char* packed, const float* vector, float* output,
    const int input_size, const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float partial =
        q5k_row_dot_warp(packed, vector, row, input_size, lane);
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    if (lane == 0) output[row] = partial;
}

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
COLIBRI_LOWBIT_MATMUL_ROWS(iq1m_matmul_rows, iq1m_value)
COLIBRI_LOWBIT_MATMUL_ROWS(iq1s_matmul_rows, iq1s_value)
#undef COLIBRI_LOWBIT_MATMUL_ROWS

#define KV_ATTENTION_FUSED_TILES_W(name, KT, VT, WIDTH) \
extern "C" __global__ void name( \
    const float* query, const KT* keys, const VT* values, float* partial, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first, const float scale \
) { \
    kv_attention_fused_tiles_impl<KT, VT, WIDTH>( \
        query, keys, values, partial, heads, kv_heads, head_dim, tokens, \
        capacity, first, scale \
    ); \
}
#define KV_ATTENTION_FUSED_TILES(name, KT, VT) \
    KV_ATTENTION_FUSED_TILES_W(name, KT, VT, 128) \
    KV_ATTENTION_FUSED_TILES_W(name##256, KT, VT, 256) \
    KV_ATTENTION_FUSED_TILES_W(name##512, KT, VT, 512)
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
#undef KV_ATTENTION_FUSED_TILES_W
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Turbo twin of the fused tiles kernel above. The cache rows live in the
// rotated (sign-flip + Walsh-Hadamard) domain, and the rotation is
// orthonormal, so the block rotates its query once into shared memory and
// every dot product happens in-place against the quantized rows; values
// accumulate still-rotated and the paired turbo merge applies the single
// inverse rotation per head. This is what replaces the cuBLAS staging path
// per decode step: that path first rewrites the whole cache window to f16
// (two O(context) launches, kv_dequant_turbo*_f16) and then reads the copy,
// where this grid reads each quantized row exactly once, directly.
template<int BITS, int maximum_head_dim>
__device__ void kv_attention_fused_turbo_tiles_impl(
    const float* query,
    const unsigned char* keys,
    const unsigned char* values,
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
    constexpr int parts = maximum_head_dim / 32;
    constexpr int tokens_per_tile = 1024;
    const int head = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // Guards stay uniform across the block: every thread must reach the
    // __syncthreads inside turbo_fwht_shared below. head_dim is a power of
    // two wherever a turbo cache type is admitted, and the width dispatch
    // only sends 128/256/512 here, but the kernel re-checks rather than
    // trusting the host.
    if (head >= heads || head_dim > maximum_head_dim || head_dim < 32 ||
        (head_dim & (head_dim - 1)) != 0)
        return;
    __shared__ float rotated_query[maximum_head_dim];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        rotated_query[d] = query[head * head_dim + d] * turbo_sign_d(d, 0u);
    __syncthreads();
    turbo_fwht_shared(rotated_query, head_dim);

    const int tile_begin = tile * tokens_per_tile;
    const int tile_end = min(tokens, tile_begin + tokens_per_tile);
    const int kv_head = head / (heads / kv_heads);
    const int row_bytes = (head_dim / 32) * turbo_block_bytes<BITS>();
    const unsigned char* key_base =
        keys + (long long)kv_head * capacity * row_bytes;
    const unsigned char* value_base =
        values + (long long)kv_head * capacity * row_bytes;
    float accumulator[parts];
    #pragma unroll
    for (int part = 0; part < parts; ++part) accumulator[part] = 0.0f;
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    for (int token = tile_begin + warp;
         token < tile_end;
         token += warp_count) {
        int slot = first + token;
        if (slot >= capacity) slot -= capacity;
        const unsigned char* key_row = key_base + (long long)slot * row_bytes;
        float dot = 0.0f;
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
            const int dimension = lane + part * 32;
            if (dimension < head_dim)
                dot += rotated_query[dimension]
                    * kv_ld_turbo<BITS>(key_row, dimension);
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
        const unsigned char* value_row =
            value_base + (long long)slot * row_bytes;
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
            const int dimension = lane + part * 32;
            if (dimension < head_dim)
                accumulator[part] = accumulator[part] * old_scale
                    + token_scale
                    * kv_ld_turbo<BITS>(value_row, dimension);
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
    for (int part = 0; part < parts; ++part) {
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

#define KV_ATTENTION_FUSED_TURBO_TILES_W(name, BITS, WIDTH) \
extern "C" __global__ void name( \
    const float* query, const unsigned char* keys, const unsigned char* values, \
    float* partial, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first, const float scale \
) { \
    kv_attention_fused_turbo_tiles_impl<BITS, WIDTH>( \
        query, keys, values, partial, heads, kv_heads, head_dim, tokens, \
        capacity, first, scale \
    ); \
}
#define KV_ATTENTION_FUSED_TURBO_TILES(name, BITS) \
    KV_ATTENTION_FUSED_TURBO_TILES_W(name, BITS, 128) \
    KV_ATTENTION_FUSED_TURBO_TILES_W(name##256, BITS, 256) \
    KV_ATTENTION_FUSED_TURBO_TILES_W(name##512, BITS, 512)
KV_ATTENTION_FUSED_TURBO_TILES(kv_attention_fused_turbo3_tiles, 3)
KV_ATTENTION_FUSED_TURBO_TILES(kv_attention_fused_turbo4_tiles, 4)
#undef KV_ATTENTION_FUSED_TURBO_TILES
#undef KV_ATTENTION_FUSED_TURBO_TILES_W

// Merge for the turbo tiles: identical tile combine, but the combined vector
// is still in the rotated domain, so it lands in shared memory and takes the
// single inverse rotation -- R^-1 = R^T, Walsh-Hadamard first, then the
// value-stream sign flip -- before the store. Must be paired with the turbo
// tiles kernel of the same width, exactly like the plain merges.
template<int maximum_head_dim>
__device__ void kv_attention_fused_turbo_merge_impl(
    const float* partial,
    float* output,
    const int heads,
    const int head_dim,
    const int tile_count
) {
    constexpr int maximum_tiles = 512;
    const int head = blockIdx.x;
    if (head >= heads || head_dim > maximum_head_dim || head_dim < 32 ||
        (head_dim & (head_dim - 1)) != 0 ||
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
    __shared__ float accumulated[maximum_head_dim];
    for (int dimension = threadIdx.x;
         dimension < head_dim;
         dimension += blockDim.x) {
        float result = 0.0f;
        for (int tile = 0; tile < tile_count; ++tile) {
            result += head_partial[tile * stride + dimension + 2]
                * tile_scale[tile];
        }
        accumulated[dimension] = result * inverse_denominator;
    }
    __syncthreads();
    turbo_fwht_shared(accumulated, head_dim);
    for (int dimension = threadIdx.x;
         dimension < head_dim;
         dimension += blockDim.x)
        output[head * head_dim + dimension] =
            accumulated[dimension] * turbo_sign_d(dimension, 1u);
}

extern "C" __global__ void kv_attention_fused_turbo_merge(
    const float* partial, float* output,
    const int heads, const int head_dim, const int tile_count
) {
    kv_attention_fused_turbo_merge_impl<128>(
        partial, output, heads, head_dim, tile_count);
}
extern "C" __global__ void kv_attention_fused_turbo_merge256(
    const float* partial, float* output,
    const int heads, const int head_dim, const int tile_count
) {
    kv_attention_fused_turbo_merge_impl<256>(
        partial, output, heads, head_dim, tile_count);
}
extern "C" __global__ void kv_attention_fused_turbo_merge512(
    const float* partial, float* output,
    const int heads, const int head_dim, const int tile_count
) {
    kv_attention_fused_turbo_merge_impl<512>(
        partial, output, heads, head_dim, tile_count);
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Grouped-query attention that reads each cached byte once per block instead of
// once per query head.
//
// The per-head kernel above keys its grid on query heads, so with GQA 8:1 the
// eight heads sharing a KV head each re-read and re-reduce the same bytes.
// Measured on this card at 49k tokens: 235 us at a sharing factor of 2 (428
// GB/s, memory-bound) against 675 us at 8 (149 GB/s) for *identical* KV -- the
// loss is entirely the redundancy, not DRAM.
//
// So key the grid on KV heads, stage a chunk of K and V in shared once, and give
// each warp one query head. Two consequences beyond the saved traffic: the eight
// reduction chains become independent, and because a warp owns its query head
// outright there is no cross-warp merge and no partial_output array -- which is
// what makes 256-dim heads fit, since eight warps' worth of 256 floats each
// would have been 64 KB of shared memory.
template<typename KT, typename VT, int maximum_head_dim, int share>
__device__ void kv_attention_gqa_tiles_impl(
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
    constexpr int parts = maximum_head_dim / 32;
    constexpr int tokens_per_tile = 512;
    // Tokens staged per round. 16 costs 32 KB of shared for K and V together,
    // which still leaves room for more than one block per SM.
    constexpr int chunk = 16;

    const int kv_head = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (kv_head >= kv_heads || head_dim > maximum_head_dim ||
        (head_dim & 31) != 0 || heads != kv_heads * share)
        return;

    const int tile_begin = tile * tokens_per_tile;
    const int tile_end = min(tokens, tile_begin + tokens_per_tile);
    if (tile_begin >= tile_end) return;

    const int head = kv_head * share + warp;
    const float* q = query + (long long)head * head_dim;
    float query_registers[parts];
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int dimension = lane + part * 32;
        query_registers[part] = dimension < head_dim ? q[dimension] : 0.0f;
    }

    __shared__ float key_stage[chunk][maximum_head_dim];
    __shared__ float value_stage[chunk][maximum_head_dim];

    float accumulator[parts];
    #pragma unroll
    for (int part = 0; part < parts; ++part) accumulator[part] = 0.0f;
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    for (int base = tile_begin; base < tile_end; base += chunk) {
        const int count = min(chunk, tile_end - base);
        for (int index = threadIdx.x;
             index < count * head_dim;
             index += blockDim.x) {
            const int token = index / head_dim;
            const int dimension = index - token * head_dim;
            int slot = first + base + token;
            if (slot >= capacity) slot -= capacity;
            const long long row = (long long)kv_head * capacity + slot;
            key_stage[token][dimension] =
                kv_fused_load(keys, row, dimension, head_dim);
            value_stage[token][dimension] =
                kv_fused_load(values, row, dimension, head_dim);
        }
        __syncthreads();
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
        for (int token = 0; token < count; ++token) {
            float dot = 0.0f;
            #pragma unroll
            for (int part = 0; part < parts; ++part) {
                const int dimension = lane + part * 32;
                if (dimension < head_dim)
                    dot += query_registers[part] * key_stage[token][dimension];
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            const float score = __shfl_sync(0xffffffff, dot, 0) * scale;
            const float next_maximum = fmaxf(maximum, score);
            const float old_scale = maximum == -3.402823466e+38F
                ? 0.0f : __expf(maximum - next_maximum);
            const float token_scale = __expf(score - next_maximum);
            denominator = denominator * old_scale + token_scale;
            #pragma unroll
            for (int part = 0; part < parts; ++part) {
                const int dimension = lane + part * 32;
                if (dimension < head_dim)
                    accumulator[part] = accumulator[part] * old_scale
                        + token_scale * value_stage[token][dimension];
            }
            maximum = next_maximum;
        }
        __syncthreads();
    }

    // Same record layout as the per-head kernel, so kv_attention_fused_merge*
    // combines these tiles unchanged.
    const int tile_count = (tokens + tokens_per_tile - 1) / tokens_per_tile;
    float* record = partial
        + ((long long)head * tile_count + tile) * (maximum_head_dim + 2);
    if (lane == 0) {
        record[0] = maximum;
        record[1] = denominator;
    }
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int dimension = lane + part * 32;
        if (dimension < head_dim)
            record[dimension + 2] = accumulator[part];
    }
}

#define KV_ATTENTION_GQA_TILES(name, KT, VT, WIDTH, SHARE) \
extern "C" __global__ void name( \
    const float* query, const KT* keys, const VT* values, float* partial, \
    const int heads, const int kv_heads, const int head_dim, const int tokens, \
    const int capacity, const int first, const float scale \
) { \
    kv_attention_gqa_tiles_impl<KT, VT, WIDTH, SHARE>( \
        query, keys, values, partial, heads, kv_heads, head_dim, tokens, \
        capacity, first, scale \
    ); \
}
KV_ATTENTION_GQA_TILES(kv_attention_gqa_f16_256_s8, __half, __half, 256, 8)
KV_ATTENTION_GQA_TILES(kv_attention_gqa_f16_256_s4, __half, __half, 256, 4)
KV_ATTENTION_GQA_TILES(
    kv_attention_gqa_bf16_256_s8, __nv_bfloat16, __nv_bfloat16, 256, 8)
KV_ATTENTION_GQA_TILES(
    kv_attention_gqa_bf16_256_s4, __nv_bfloat16, __nv_bfloat16, 256, 4)
KV_ATTENTION_GQA_TILES(
    kv_attention_gqa_q8_256_s8, unsigned char, unsigned char, 256, 8)
KV_ATTENTION_GQA_TILES(
    kv_attention_gqa_q8_256_s4, unsigned char, unsigned char, 256, 4)
#undef KV_ATTENTION_GQA_TILES

template<int maximum_head_dim>
__device__ void kv_attention_fused_merge_impl(
    const float* partial,
    float* output,
    const int heads,
    const int head_dim,
    const int tile_count
) {
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

extern "C" __global__ void kv_attention_fused_merge(
    const float* partial, float* output,
    const int heads, const int head_dim, const int tile_count
) {
    kv_attention_fused_merge_impl<128>(
        partial, output, heads, head_dim, tile_count);
}

// The 256-dim twin. The partial record stride follows maximum_head_dim, so a
// merge must be paired with the tiles kernel of the same width.
extern "C" __global__ void kv_attention_fused_merge256(
    const float* partial, float* output,
    const int heads, const int head_dim, const int tile_count
) {
    kv_attention_fused_merge_impl<256>(
        partial, output, heads, head_dim, tile_count);
}

// And the 512-dim twin, for Gemma 4's global-attention layers. The per-tile
// shared buffer grows to 16 KB, still inside the static budget.
extern "C" __global__ void kv_attention_fused_merge512(
    const float* partial, float* output,
    const int heads, const int head_dim, const int tile_count
) {
    kv_attention_fused_merge_impl<512>(
        partial, output, heads, head_dim, tile_count);
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
        kv_st(cache_keys, slot, current_keys[head * head_dim + d]);
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// ---- qwen4exp gated residual + PLE ----------------------------------------
// Semantics: plans/qwen4exp-semantics.md, pinned by
// native/tools/qwen4exp_reference_check.py. The heavy lifting (down/up/inject
// projections, ple key/value projections) runs through the ordinary dense
// matvec kernels; these cover only the elementwise/reduction glue.

// Stream init: the token embedding repeated hc times.
extern "C" __global__
void qwen4_hc_init(
    const float* hidden, float* streams, const int hc, const int hidden_size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hc * hidden_size) return;
    streams[index] = hidden[index % hidden_size];
}

// Grouped RMS norm: each group_size chunk normalized independently, one block
// per group, weight spanning the full width (baked (1+w) form).
extern "C" __global__
void qwen4_group_rms(
    const float* input, const float* weights, float* output,
    const int group_size, const float epsilon
) {
    const int group = blockIdx.x;
    const float* in = input + (long long)group * group_size;
    float* out = output + (long long)group * group_size;
    const float* w = weights + (long long)group * group_size;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        partial += in[i] * in[i];
    }
    partial = block_reduce_sum(partial);
    __shared__ float scale;
    if (threadIdx.x == 0)
        scale = rsqrtf(partial / (float)group_size + epsilon);
    __syncthreads();
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        out[i] = in[i] * scale * w[i];
    }
}

// silu(x / divisor) in place; the low-rank mixer's activation.
extern "C" __global__
void qwen4_silu_scale(float* values, const int count, const float divisor) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = values[index] / divisor;
    values[index] = x / (1.0f + expf(-x));
}

// block_input = mean over streams of sigmoid(up_out) * normed.
extern "C" __global__
void qwen4_hc_mix(
    const float* normed, const float* up_out, float* block_input,
    const int hc, const int hidden_size
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) return;
    float total = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const long long at = (long long)s * hidden_size + i;
        const float gate = 1.0f / (1.0f + expf(-up_out[at]));
        total += gate * normed[at];
    }
    block_input[i] = total / (float)hc;
}

// streams += 2*sigmoid(inject_raw/hc)[stream] * block_output. The residual
// base is the pre-norm streams, which is exactly what `streams` still holds.
extern "C" __global__
void qwen4_hc_inject(
    float* streams, const float* block_output, const float* inject_raw,
    const int hc, const int hidden_size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hc * hidden_size) return;
    const int s = index / hidden_size;
    const float w =
        2.0f / (1.0f + expf(-inject_raw[s] / (float)hc));
    streams[index] += w * block_output[index % hidden_size];
}

// Per-stream gate: signed sqrt of the key.query dot over sqrt(hidden).
// Stores the raw gate; the consumer applies the sigmoid.
extern "C" __global__
void qwen4_ple_gate(
    const float* key_normed, const float* query_normed, float* gates,
    const int hidden_size
) {
    const int s = blockIdx.x;
    const float* k = key_normed + (long long)s * hidden_size;
    const float* q = query_normed + (long long)s * hidden_size;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        partial += k[i] * q[i];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) {
        float g = partial / sqrtf((float)hidden_size);
        const float magnitude = sqrtf(fmaxf(fabsf(g), 1e-6f));
        gates[s] = g < 0.0f ? -magnitude : magnitude;
    }
}

// gated value: out[s][i] = sigmoid(gates[s]) * value[i].
extern "C" __global__
void qwen4_ple_gv(
    const float* value, const float* gates, float* out,
    const int hc, const int hidden_size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hc * hidden_size) return;
    const float gate = 1.0f / (1.0f + expf(-gates[index / hidden_size]));
    out[index] = gate * value[index % hidden_size];
}

// Dilated depthwise causal conv, single-token step. State is channel-major
// [(kernel-1)*dilation] per channel, oldest first; the newest column is the
// current input, which also shifts in. Output = silu(conv).
extern "C" __global__
void qwen4_ple_conv_step(
    const float* input, const float* weights, float* state, float* output,
    const int channels, const int kernel_size, const int dilation
) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const int len = (kernel_size - 1) * dilation;
    float* channel_state = state + (long long)channel * len;
    const float* channel_weights = weights + (long long)channel * kernel_size;
    const float current = input[channel];
    float value = channel_weights[kernel_size - 1] * current;
    for (int tap = 0; tap + 1 < kernel_size; ++tap) {
        // Tap `tap` reads (kernel-1-tap)*dilation steps back; newest state
        // column is time -1 at index len-1.
        value += channel_weights[tap] *
            channel_state[len - (kernel_size - 1 - tap) * dilation];
    }
    for (int index = 0; index + 1 < len; ++index)
        channel_state[index] = channel_state[index + 1];
    channel_state[len - 1] = current;
    output[channel] = value / (1.0f + expf(-value));
}

// streams += gv + conv_out (the PLE delta, added in stream space). Pure
// elementwise, so the rows path reuses it with count = rows * wide.
extern "C" __global__
void qwen4_ple_add(
    float* streams, const float* gv, const float* conv_out, const int wide
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= wide) return;
    streams[index] += gv[index] + conv_out[index];
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// ---- qwen4exp gated residual + PLE, rows (batched prefill) forms ----------
// Same math as the single-token kernels above with blockIdx.y as the row.
// qwen4_silu_scale and qwen4_ple_add serve both paths (elementwise).

extern "C" __global__
void qwen4_hc_init_rows(
    const float* hidden, float* streams, const int hc, const int hidden_size
) {
    const long long row = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hc * hidden_size) return;
    streams[row * hc * hidden_size + index] =
        hidden[row * hidden_size + index % hidden_size];
}

extern "C" __global__
void qwen4_group_rms_rows(
    const float* input, const float* weights, float* output,
    const int groups, const int group_size, const float epsilon
) {
    const long long row = blockIdx.y;
    const int group = blockIdx.x;
    const long long at = (row * groups + group) * group_size;
    const float* in = input + at;
    float* out = output + at;
    const float* w = weights + (long long)group * group_size;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x)
        partial += in[i] * in[i];
    partial = block_reduce_sum(partial);
    __shared__ float scale;
    if (threadIdx.x == 0)
        scale = rsqrtf(partial / (float)group_size + epsilon);
    __syncthreads();
    for (int i = threadIdx.x; i < group_size; i += blockDim.x)
        out[i] = in[i] * scale * w[i];
}

extern "C" __global__
void qwen4_hc_mix_rows(
    const float* normed, const float* up_out, float* block_input,
    const int hc, const int hidden_size
) {
    const long long row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) return;
    const long long wide = (long long)hc * hidden_size;
    float total = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const long long at = row * wide + (long long)s * hidden_size + i;
        const float gate = 1.0f / (1.0f + expf(-up_out[at]));
        total += gate * normed[at];
    }
    block_input[row * hidden_size + i] = total / (float)hc;
}

extern "C" __global__
void qwen4_hc_inject_rows(
    float* streams, const float* block_output, const float* inject_raw,
    const int hc, const int hidden_size
) {
    const long long row = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hc * hidden_size) return;
    const int s = index / hidden_size;
    const float w = 2.0f /
        (1.0f + expf(-inject_raw[row * hc + s] / (float)hc));
    streams[row * (long long)hc * hidden_size + index] +=
        w * block_output[row * hidden_size + index % hidden_size];
}

extern "C" __global__
void qwen4_ple_gate_rows(
    const float* key_normed, const float* query_normed, float* gates,
    const int hc, const int hidden_size
) {
    const long long row = blockIdx.y;
    const int s = blockIdx.x;
    const long long at = (row * hc + s) * (long long)hidden_size;
    const float* k = key_normed + at;
    const float* q = query_normed + at;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        partial += k[i] * q[i];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) {
        float g = partial / sqrtf((float)hidden_size);
        const float magnitude = sqrtf(fmaxf(fabsf(g), 1e-6f));
        gates[row * hc + s] = g < 0.0f ? -magnitude : magnitude;
    }
}

extern "C" __global__
void qwen4_ple_gv_rows(
    const float* value, const float* gates, float* out,
    const int hc, const int hidden_size
) {
    const long long row = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hc * hidden_size) return;
    const float gate = 1.0f /
        (1.0f + expf(-gates[row * hc + index / hidden_size]));
    out[row * (long long)hc * hidden_size + index] =
        gate * value[row * hidden_size + index % hidden_size];
}

// Dilated depthwise causal conv over a whole chunk, sequential in time per
// channel (like delta_conv_sequence), state carried across chunks.
extern "C" __global__
void qwen4_ple_conv_sequence(
    const float* input, const float* weights, float* state, float* output,
    const int tokens, const int channels, const int kernel_size,
    const int dilation
) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const int len = (kernel_size - 1) * dilation;
    float* channel_state = state + (long long)channel * len;
    const float* channel_weights = weights + (long long)channel * kernel_size;
    for (int token = 0; token < tokens; ++token) {
        const float current = input[(long long)token * channels + channel];
        float value = channel_weights[kernel_size - 1] * current;
        for (int tap = 0; tap + 1 < kernel_size; ++tap)
            value += channel_weights[tap] *
                channel_state[len - (kernel_size - 1 - tap) * dilation];
        for (int index = 0; index + 1 < len; ++index)
            channel_state[index] = channel_state[index + 1];
        channel_state[len - 1] = current;
        output[(long long)token * channels + channel] =
            value / (1.0f + expf(-value));
    }
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// QSA indexer (qwen4exp phase 3). Runs once per token on every selecting
// layer, in all three forward paths: stores the token's raw 128-wide index
// key into the per-layer ring and, when the token completes a `ratio`-block,
// derives the block key -- fp32 mean of the ring, rms with the baked k_norm,
// then the same partial half-split rope as qwen_attention_key, anchored at
// the block's FIRST position. A completed block key never changes again.
// One thread block; blockDim must be >= key_len.
extern "C" __global__
void qsa_key_store(
    const float* raw_key, float* ring, float* block_keys,
    const float* norm_weights, const int position, const int ratio,
    const int key_len, const int rotary_dim, const float theta,
    const float epsilon
) {
    const int lane = threadIdx.x;
    const int row = position % ratio;
    if (lane < key_len) ring[row * key_len + lane] = raw_key[lane];
    __syncthreads();
    if (row != ratio - 1) return;
    float pooled = 0.0f;
    if (lane < key_len) {
        for (int r = 0; r < ratio; ++r) pooled += ring[r * key_len + lane];
        pooled /= (float)ratio;
    }
    float square = block_reduce_sum(pooled * pooled);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) inverse_rms = rsqrtf(square / (float)key_len + epsilon);
    __syncthreads();
    // Stage the normed vector in the output row so the rope pair lanes can
    // read each other, then rotate in place.
    float* out = block_keys + (long long)(position / ratio) * key_len;
    const float normed = pooled * inverse_rms *
        (lane < key_len ? norm_weights[lane] : 0.0f);
    if (lane < key_len) out[lane] = normed;
    __syncthreads();
    float value = normed;
    if (lane < rotary_dim) {
        const int half = rotary_dim / 2;
        const int pair = lane < half ? lane : lane - half;
        const float other = out[lane < half ? lane + half : lane - half];
        const float angle = (float)(position - (ratio - 1))
            / powf(theta, 2.0f * (float)pair / (float)rotary_dim);
        value = lane < half
            ? value * cosf(angle) - other * sinf(angle)
            : value * cosf(angle) + other * sinf(angle);
    }
    __syncthreads();
    if (lane < key_len) out[lane] = value;
}

// Block scores for one query: sum over the indexer heads of relu(q_h . k_b),
// scaled by rsqrt(key_len). The query is already normed and roped
// (qwen_attention_key with the indexer geometry).
extern "C" __global__
void qsa_block_scores(
    const float* query, const float* block_keys, float* scores,
    const int blocks, const int heads, const int key_len
) {
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= blocks) return;
    const float* key = block_keys + (long long)block * key_len;
    float total = 0.0f;
    for (int head = 0; head < heads; ++head) {
        const float* q = query + head * key_len;
        float dot = 0.0f;
        for (int d = 0; d < key_len; ++d) dot += q[d] * key[d];
        total += fmaxf(dot, 0.0f);
    }
    scores[block] = total * rsqrtf((float)key_len);
}

// Sparse attention over an explicit slot list -- the ring kernels with the
// (first+token)%capacity indirection replaced by slots[token]. The list is
// the QSA selection: the top-k blocks' slots ascending plus the incomplete
// tail, so `tokens` is at most budget + ratio - 1.
template<typename KT>
__device__ void kv_scores_indexed_impl(
    const float* query, const KT* keys, const int* slots, float* scores,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const float scale
) {
    const int head = blockIdx.x;
    const int token = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || token >= tokens) return;
    const int kv_head = head / (heads / kv_heads);
    const float* q = query + head * head_dim;
    const KT* k = keys + ((long long)kv_head * capacity + slots[token]) * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) score += q[d] * kv_ld(k, d);
    scores[head * tokens + token] = score * scale;
}
#define KV_SCORES_INDEXED(name, T) \
extern "C" __global__ void name(const float* query, const T* keys, const int* slots, \
    float* scores, const int heads, const int kv_heads, const int head_dim, \
    const int tokens, const int capacity, const float scale) { \
    kv_scores_indexed_impl<T>(query, keys, slots, scores, heads, kv_heads, head_dim, tokens, capacity, scale); \
}
KV_SCORES_INDEXED(kv_attention_scores_indexed, float)
KV_SCORES_INDEXED(kv_attention_scores_f16_indexed, __half)
KV_SCORES_INDEXED(kv_attention_scores_bf16_indexed, __nv_bfloat16)
#undef KV_SCORES_INDEXED
extern "C" __global__ void kv_attention_scores_q8_indexed(
    const float* query, const unsigned char* keys, const int* slots,
    float* scores, const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity, const float scale
) {
    const int head=blockIdx.x, token=blockIdx.y*blockDim.x+threadIdx.x;
    if(head>=heads||token>=tokens)return;
    const int kv_head=head/(heads/kv_heads), blocks=head_dim/32;
    const float* q=query+head*head_dim;
    const unsigned char* k=keys+((long long)kv_head*capacity+slots[token])*blocks*34;
    float score=0.0f;
    for(int d=0;d<head_dim;++d)score+=q[d]*kv_ld_q8(k,d);
    scores[head*tokens+token]=score*scale;
}

template<typename VT>
__device__ void kv_values_indexed_impl(
    float* scores, const VT* values, const int* slots, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity
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
        for(int token=0;token<tokens;++token)
            result+=head_scores[token]*
                kv_ld(values,(long long)(kv_head*capacity+slots[token])*head_dim+d);
        output[head*head_dim+d]=result;
    }
}
#define KV_VALUES_INDEXED(name, T) \
extern "C" __global__ void name(float* scores, const T* values, const int* slots, \
    float* output, const int heads, const int kv_heads, const int head_dim, \
    const int tokens, const int capacity) { \
    kv_values_indexed_impl<T>(scores, values, slots, output, heads, kv_heads, head_dim, tokens, capacity); \
}
KV_VALUES_INDEXED(kv_attention_values_indexed, float)
KV_VALUES_INDEXED(kv_attention_values_f16_indexed, __half)
KV_VALUES_INDEXED(kv_attention_values_bf16_indexed, __nv_bfloat16)
#undef KV_VALUES_INDEXED
extern "C" __global__ void kv_attention_values_q8_indexed(
    float* scores, const unsigned char* values, const int* slots, float* output,
    const int heads, const int kv_heads, const int head_dim,
    const int tokens, const int capacity
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
            const unsigned char* vrow=values+((long long)kv_head*capacity+slots[token])*blocks*34;
            result+=head_scores[token]*kv_ld_q8(vrow,d);
        }
        output[head*head_dim+d]=result;
    }
}
)COLIBRI_CUDA";
}
