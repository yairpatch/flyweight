#pragma once

namespace colibri::v2 {
inline constexpr char qwen_cuda_source[] = R"COLIBRI_CUDA(

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
    const int start = row * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned int bits = ((unsigned int)weights[start + column]) << 16;
        partial += __uint_as_float(bits) * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
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

// Tiled single-token Q8 attention.  Every block computes an online-softmax
// state for one [head, 1024-token] tile.  A second small kernel merges those
// states.  Compared with the split score/value path this keeps long-context
// occupancy (many tiles are independent), avoids the [heads, tokens] score
// round trip, and reads each value only while its score is live.
//
// Qwen currently uses head_dim=128.  Other shapes retain the generic split
// kernels, keeping this fast path's local and shared storage fixed-size.
extern "C" __global__ void kv_attention_fused_q8_tiles(
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
    const int blocks = head_dim / 32;
    const float* q = query + head * head_dim;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    for (int token = tile_begin + warp;
         token < tile_end;
         token += warp_count) {
        int slot = first + token;
        if (slot >= capacity) slot -= capacity;
        const unsigned char* key_row =
            keys + ((long long)kv_head * capacity + slot) * blocks * 34;
        float dot = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int dimension = lane + part * 32;
            if (dimension < head_dim)
                dot += q[dimension] * kv_ld_q8(key_row, dimension);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        const float score =
            __shfl_sync(0xffffffff, dot, 0) * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float old_scale = maximum == -3.402823466e+38F
            ? 0.0f : expf(maximum - next_maximum);
        const float token_scale = expf(score - next_maximum);
        denominator = denominator * old_scale + token_scale;

        const unsigned char* value_row =
            values + ((long long)kv_head * capacity + slot) * blocks * 34;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int dimension = lane + part * 32;
            if (dimension < head_dim) {
                accumulator[part] = accumulator[part] * old_scale
                    + token_scale * kv_ld_q8(value_row, dimension);
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
                ? 0.0f : expf(warp_maximum[index] - tile_maximum);
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
                expf(head_partial[tile * stride] - maximum);
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
)COLIBRI_CUDA";
}
