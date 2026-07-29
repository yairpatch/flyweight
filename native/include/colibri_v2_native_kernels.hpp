#pragma once

namespace colibri::v2 {
inline constexpr char qwen_native_cuda_source[] = R"COLIBRI_CUDA(

extern "C" __global__
void qwen_q8_embedding(
    const unsigned char* packed, float* output,
    const int token, const int hidden
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += blockDim.x * gridDim.x) {
        const int absolute = token * hidden + index;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(*((const __half*)(packed + block * 34)));
        const signed char value = *((const signed char*)(packed + block * 34 + 2 + within));
        output[index] = scale * (float)value;
    }
}

extern "C" __global__
void qwen_q8_embedding_rows(
    const unsigned char* packed, const unsigned int* tokens, float* output,
    const int rows, const int hidden
) {
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int token = (int)tokens[row];
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += blockDim.x * gridDim.x) {
        const int absolute = token * hidden + index;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(*((const __half*)(packed + block * 34)));
        const signed char value = *((const signed char*)(packed + block * 34 + 2 + within));
        output[row * hidden + index] = scale * (float)value;
    }
}

// K-quant embedding tables. These share the super-block decoders defined in
// qwen_cuda_source, which is concatenated ahead of this translation unit, so a
// gather is just an indexed decode. Low-bit dense checkpoints commonly store
// token_embd as Q4_K or lower rather than Q8_0.
#define COLIBRI_KQUANT_EMBEDDING(name, decode)                                  \
extern "C" __global__                                                           \
void name(                                                                      \
    const unsigned char* packed, float* output,                                 \
    const int token, const int hidden                                           \
) {                                                                             \
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;                     \
         index < hidden; index += blockDim.x * gridDim.x) {                     \
        output[index] = decode(packed, token * hidden + index);                 \
    }                                                                           \
}                                                                               \
extern "C" __global__                                                           \
void name##_rows(                                                               \
    const unsigned char* packed, const unsigned int* tokens, float* output,     \
    const int rows, const int hidden                                            \
) {                                                                             \
    const int row = blockIdx.y;                                                 \
    if (row >= rows) return;                                                    \
    const int token = (int)tokens[row];                                         \
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;                     \
         index < hidden; index += blockDim.x * gridDim.x) {                     \
        output[row * hidden + index] = decode(packed, token * hidden + index);  \
    }                                                                           \
}

COLIBRI_KQUANT_EMBEDDING(qwen_q2k_embedding, q2k_value)
COLIBRI_KQUANT_EMBEDDING(qwen_q3k_embedding, q3k_value)
COLIBRI_KQUANT_EMBEDDING(qwen_q4k_embedding, q4k_value)
COLIBRI_KQUANT_EMBEDDING(qwen_q5k_embedding, q5k_value)
COLIBRI_KQUANT_EMBEDDING(qwen_q6k_embedding, q6k_value)
COLIBRI_KQUANT_EMBEDDING(qwen_iq2xxs_embedding, iq2xxs_value)
COLIBRI_KQUANT_EMBEDDING(qwen_iq3xxs_embedding, iq3xxs_value)
COLIBRI_KQUANT_EMBEDDING(qwen_iq2s_embedding, iq2s_value)
COLIBRI_KQUANT_EMBEDDING(qwen_iq3s_embedding, iq3s_value)
COLIBRI_KQUANT_EMBEDDING(qwen_iq2xs_embedding, iq2xs_value)
COLIBRI_KQUANT_EMBEDDING(qwen_iq4xs_embedding, iq4xs_value)

#undef COLIBRI_KQUANT_EMBEDDING

// bf16 embedding tables are stored as plain rows, not Q8_0 blocks. Reading them
// with qwen_q8_embedding reinterprets pairs of bf16 values as a block scale plus
// int8 codes, which yields ~100x-magnitude noise that swamps the whole residual
// stream, so the table type must pick the matching kernel.
extern "C" __global__
void qwen_bf16_embedding(
    const unsigned char* packed, float* output,
    const int token, const int hidden
) {
    const unsigned short* row =
        (const unsigned short*)packed + (long long)token * (long long)hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += blockDim.x * gridDim.x) {
        output[index] = __uint_as_float(((unsigned int)row[index]) << 16);
    }
}

extern "C" __global__
void qwen_bf16_embedding_rows(
    const unsigned char* packed, const unsigned int* tokens, float* output,
    const int rows, const int hidden
) {
    const int row = blockIdx.y;
    if (row >= rows) return;
    const unsigned short* source =
        (const unsigned short*)packed + (long long)tokens[row] * (long long)hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += blockDim.x * gridDim.x) {
        output[row * hidden + index] =
            __uint_as_float(((unsigned int)source[index]) << 16);
    }
}

extern "C" __global__
void qwen_f32_embedding(
    const unsigned char* packed, float* output,
    const int token, const int hidden
) {
    const float* row = (const float*)packed + (long long)token * (long long)hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += blockDim.x * gridDim.x) {
        output[index] = row[index];
    }
}

extern "C" __global__
void qwen_f32_embedding_rows(
    const unsigned char* packed, const unsigned int* tokens, float* output,
    const int rows, const int hidden
) {
    const int row = blockIdx.y;
    if (row >= rows) return;
    const float* source =
        (const float*)packed + (long long)tokens[row] * (long long)hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += blockDim.x * gridDim.x) {
        output[row * hidden + index] = source[index];
    }
}

extern "C" __global__
void qwen_f32_matvec(
    const float* matrix, const float* input, float* output,
    const int input_size, const int output_size
) {
    const int row = blockIdx.x;
    if (row >= output_size) return;
    float partial = 0.0f;
    for (int column = threadIdx.x; column < input_size; column += blockDim.x)
        partial += matrix[row * input_size + column] * input[column];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

extern "C" __global__
void qwen_f32_matvec_warp(
    const float* matrix, const float* input, float* output,
    const int input_size, const int output_size
) {
    // Warp per row, eight rows per block: no shared-memory reduction, and
    // float4 loads move 512 B per warp memory instruction instead of 128 B.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    const float* row_matrix = matrix + (long long)row * (long long)input_size;
    float partial = 0.0f;
    if ((input_size & 3) == 0
        && (((unsigned long long)row_matrix) & 15ull) == 0ull
        && (((unsigned long long)input) & 15ull) == 0ull) {
        const int steps = input_size >> 2;
        for (int step = lane; step < steps; step += 32) {
            const float4 w = ((const float4*)row_matrix)[step];
            const float4 v = ((const float4*)input)[step];
            partial += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
    } else {
        for (int column = lane; column < input_size; column += 32)
            partial += row_matrix[column] * input[column];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}

extern "C" __global__
void qwen_f32_matmul_rows(
    const float* matrix, const float* input, float* output,
    const int input_size, const int output_size, const int rows
) {
    const int output_row = blockIdx.x;
    const int token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    float partial = 0.0f;
    const float* vector = input + token * input_size;
    for (int column = threadIdx.x; column < input_size; column += blockDim.x)
        partial += matrix[output_row * input_size + column] * vector[column];
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * output_size + output_row] = partial;
}

extern "C" __global__
void bf16_matmul_rows(
    const unsigned short* matrix, const float* input, float* output,
    const int input_size, const int output_size, const int rows
) {
    const int output_row = blockIdx.x;
    const int token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    float partial = 0.0f;
    const float* vector = input + token * input_size;
    const int weight_start = output_row * input_size;
    for (int column = threadIdx.x; column < input_size; column += blockDim.x) {
        const unsigned int bits = ((unsigned int)matrix[weight_start + column]) << 16;
        partial += __uint_as_float(bits) * vector[column];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[token * output_size + output_row] = partial;
}

extern "C" __global__
void qwen_q8_matmul_rows(
    const unsigned char* packed, const float* input, float* output,
    const int input_size, const int output_size, const int rows
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * 8 + warp;
    const int token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    const float* vector = input + token * input_size;
    float partial = 0.0f;
    for (int column = lane; column < input_size; column += 32) {
        const int absolute = output_row * input_size + column;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(*((const __half*)(packed + block * 34)));
        const signed char value = *((const signed char*)(packed + block * 34 + 2 + within));
        partial += (float)value * scale * vector[column];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[token * output_size + output_row] = partial;
}

extern "C" __global__
void qwen_q8_swiglu_rows(
    const unsigned char* gate_packed, const unsigned char* up_packed,
    const float* input, float* output,
    const int input_size, const int output_size, const int rows
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * 8 + warp;
    const int token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    const float* vector = input + token * input_size;
    float gate = 0.0f, up = 0.0f;
    for (int column = lane; column < input_size; column += 32) {
        const int absolute = output_row * input_size + column;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float value = vector[column];
        const float gate_scale = __half2float(*((const __half*)(gate_packed + block * 34)));
        const float up_scale = __half2float(*((const __half*)(up_packed + block * 34)));
        const signed char gate_value = *((const signed char*)(gate_packed + block * 34 + 2 + within));
        const signed char up_value = *((const signed char*)(up_packed + block * 34 + 2 + within));
        gate += (float)gate_value * gate_scale * value;
        up += (float)up_value * up_scale * value;
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
        output[token * output_size + output_row] = gate * sigmoid * up;
    }
}

extern "C" __global__
void qwen_shared_scale_rows(
    const float* input, const float* gate, float* shared,
    const int rows, const int elements
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* vector = input + row * elements;
    float* output = shared + row * elements;
    float partial = 0.0f;
    for (int index = threadIdx.x; index < elements; index += blockDim.x)
        partial += vector[index] * gate[index];
    partial = block_reduce_sum(partial);
    __shared__ float scale;
    if (threadIdx.x == 0) scale = 1.0f / (1.0f + expf(-partial));
    __syncthreads();
    for (int index = threadIdx.x; index < elements; index += blockDim.x)
        output[index] *= scale;
}

extern "C" __global__
void qwen_shared_scale_rows_bf16(
    const float* input, const unsigned short* gate, float* shared,
    const int rows, const int elements
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* vector = input + row * elements;
    float* output = shared + row * elements;
    float partial = 0.0f;
    for (int index = threadIdx.x; index < elements; index += blockDim.x) {
        const unsigned int bits = ((unsigned int)gate[index]) << 16;
        partial += vector[index] * __uint_as_float(bits);
    }
    partial = block_reduce_sum(partial);
    __shared__ float scale;
    if (threadIdx.x == 0) scale = 1.0f / (1.0f + expf(-partial));
    __syncthreads();
    for (int index = threadIdx.x; index < elements; index += blockDim.x)
        output[index] *= scale;
}

extern "C" __global__
void qwen_q8_lm_head_argmax_rows(
    const unsigned char* packed, const float* vectors,
    unsigned long long* winners, const int input_size,
    const int output_size, const int rows
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * 8 + warp;
    const int token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    const float* vector = vectors + token * input_size;
    float partial = 0.0f;
    for (int column = lane; column < input_size; column += 32) {
        const int absolute = output_row * input_size + column;
        const int block = absolute >> 5;
        const int within = absolute & 31;
        const float scale = __half2float(*((const __half*)(packed + block * 34)));
        const signed char value = *((const signed char*)(packed + block * 34 + 2 + within));
        partial += (float)value * scale * vector[column];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) {
        const unsigned int bits = __float_as_uint(partial);
        const unsigned int ordered = bits ^ (((int)bits < 0) ? 0xffffffffu : 0x80000000u);
        const unsigned long long candidate =
            ((unsigned long long)ordered << 32) | (unsigned int)(0xffffffffu - output_row);
        atomicMax(winners + token, candidate);
    }
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// bf16 output tables, as shipped by the NVFP4 Qwen3.6 checkpoints. Same warp
// layout as the Q8_0 variant above, only the weight decode differs.
extern "C" __global__
void qwen_bf16_lm_head_argmax_rows(
    const unsigned short* weights, const float* vectors,
    unsigned long long* winners, const int input_size,
    const int output_size, const int rows
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * 8 + warp;
    const int token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    const float* vector = vectors + (long long)token * (long long)input_size;
    const long long start = (long long)output_row * (long long)input_size;
    float partial = 0.0f;
    for (int column = lane; column < input_size; column += 32) {
        const unsigned int bits = ((unsigned int)weights[start + column]) << 16;
        partial += __uint_as_float(bits) * vector[column];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) {
        const unsigned int bits = __float_as_uint(partial);
        const unsigned int ordered = bits ^ (((int)bits < 0) ? 0xffffffffu : 0x80000000u);
        const unsigned long long candidate =
            ((unsigned long long)ordered << 32) | (unsigned int)(0xffffffffu - output_row);
        atomicMax(winners + token, candidate);
    }
}

extern "C" __global__
void qwen_delta_recurrent(
    const float* convolved, const float* gates,
    const float* beta_logits, const float* decay_logits,
    const float* decay_coefficients, const float* dt_bias,
    const float* norm_weights, float* state, float* output,
    const int key_heads, const int value_heads,
    const int head_dim, const float epsilon
) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= value_heads) return;
    const int key_head = head % key_heads;
    const int total_key_dim = key_heads * head_dim;
    const int key_offset = key_head * head_dim;
    __shared__ float query_inverse_norm;
    __shared__ float key_inverse_norm;
    __shared__ float decay_scale;
    __shared__ float beta;
    __shared__ float core_values[256];
    if (lane == 0) {
        float query_square = 0.0f;
        float key_square = 0.0f;
        for (int index = 0; index < head_dim; ++index) {
            const float query = convolved[key_offset + index];
            const float key = convolved[total_key_dim + key_offset + index];
            query_square += query * query;
            key_square += key * key;
        }
        query_inverse_norm = rsqrtf(query_square + 1.0e-6f)
            * rsqrtf((float)head_dim);
        key_inverse_norm = rsqrtf(key_square + 1.0e-6f);
        beta = 1.0f / (1.0f + expf(-beta_logits[head]));
        const float softplus_input = decay_logits[head] + dt_bias[head];
        const float softplus = softplus_input > 20.0f
            ? softplus_input : log1pf(expf(softplus_input));
        decay_scale = expf(decay_coefficients[head] * softplus);
    }
    __syncthreads();
    float core = 0.0f;
    if (lane < head_dim) {
        float memory = 0.0f;
        for (int key_index = 0; key_index < head_dim; ++key_index) {
            const float key = convolved[total_key_dim + key_offset + key_index]
                * key_inverse_norm;
            const int state_index =
                (head * head_dim + key_index) * head_dim + lane;
            state[state_index] *= decay_scale;
            memory += state[state_index] * key;
        }
        const float value = convolved[total_key_dim * 2 + head * head_dim + lane];
        const float delta = (value - memory) * beta;
        for (int key_index = 0; key_index < head_dim; ++key_index) {
            const float key = convolved[total_key_dim + key_offset + key_index]
                * key_inverse_norm;
            const int state_index =
                (head * head_dim + key_index) * head_dim + lane;
            state[state_index] += key * delta;
            core += state[state_index]
                * convolved[key_offset + key_index] * query_inverse_norm;
        }
    }
    core_values[lane] = lane < head_dim ? core : 0.0f;
    __syncthreads();
    __shared__ float inverse_rms;
    if (lane == 0) {
        float square = 0.0f;
        for (int index = 0; index < head_dim; ++index)
            square += core_values[index] * core_values[index];
        inverse_rms = rsqrtf(square / (float)head_dim + epsilon);
    }
    __syncthreads();
    if (lane < head_dim) {
        const int output_index = head * head_dim + lane;
        const float gate = gates[output_index];
        output[output_index] = core * inverse_rms * norm_weights[lane]
            * gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))));
    }
}

// Sum partials[0..count) and hand the total to every thread. The whole block
// must reach this; entries between `count` and the enclosing power of two are
// zeroed here so the tree never reads uninitialised shared memory, and the
// closing pair of barriers lets the caller refill `partials` straight after.
__device__ __forceinline__ float delta_block_sum(float* partials, const int count) {
    int width = 1;
    while (width < count) width <<= 1;
    for (int index = (int)threadIdx.x + count; index < width; index += (int)blockDim.x)
        partials[index] = 0.0f;
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if ((int)threadIdx.x < stride)
            partials[threadIdx.x] += partials[threadIdx.x + stride];
    }
    __syncthreads();
    const float total = partials[0];
    __syncthreads();
    return total;
}

extern "C" __global__
void qwen_delta_recurrent_split(
    const float* convolved, const float* gates,
    const float* beta_logits, const float* decay_logits,
    const float* decay_coefficients, const float* dt_bias,
    const float* norm_weights, float* state, float* output,
    const int key_heads, const int value_heads,
    const int head_dim, const float epsilon
) {
    // Same recurrence as qwen_delta_recurrent, mapped differently. There, one
    // thread per output dim walked all head_dim keys on its own, half the block
    // sat out the loops entirely, and three reductions ran serially under
    // `if (lane == 0)`. Here the block is head_dim output dims by `slices` key
    // groups: the key loop is `slices` times shorter, every thread works, and
    // the reductions are block-wide. The state layout is unchanged, so
    // consecutive dims still read consecutive addresses.
    extern __shared__ float delta_shared[];
    const int head = blockIdx.x;
    if (head >= value_heads) return;
    const int slices = blockDim.x / head_dim;
    const int dim = threadIdx.x % head_dim;
    const int slice = threadIdx.x / head_dim;
    const int key_head = head % key_heads;
    const int total_key_dim = key_heads * head_dim;
    const int key_offset = key_head * head_dim;

    float* shared_query = delta_shared;
    float* shared_key = shared_query + head_dim;
    float* shared_delta = shared_key + head_dim;
    float* shared_core = shared_delta + head_dim;
    float* partials = shared_core + head_dim;

    __shared__ float query_inverse_norm;
    __shared__ float key_inverse_norm;
    __shared__ float decay_scale;
    __shared__ float beta;
    __shared__ float inverse_rms;

    // Query and key are read once into shared memory; the original reloaded
    // both from global on every one of the 2 * head_dim loop iterations.
    if (slice == 0) {
        shared_query[dim] = convolved[key_offset + dim];
        shared_key[dim] = convolved[total_key_dim + key_offset + dim];
    }
    __syncthreads();
    partials[threadIdx.x] = (int)threadIdx.x < head_dim
        ? shared_query[threadIdx.x] * shared_query[threadIdx.x] : 0.0f;
    const float query_square = delta_block_sum(partials, head_dim);
    partials[threadIdx.x] = (int)threadIdx.x < head_dim
        ? shared_key[threadIdx.x] * shared_key[threadIdx.x] : 0.0f;
    const float key_square = delta_block_sum(partials, head_dim);
    if (threadIdx.x == 0) {
        query_inverse_norm = rsqrtf(query_square + 1.0e-6f)
            * rsqrtf((float)head_dim);
        key_inverse_norm = rsqrtf(key_square + 1.0e-6f);
        beta = 1.0f / (1.0f + expf(-beta_logits[head]));
        const float softplus_input = decay_logits[head] + dt_bias[head];
        const float softplus = softplus_input > 20.0f
            ? softplus_input : log1pf(expf(softplus_input));
        decay_scale = expf(decay_coefficients[head] * softplus);
    }
    __syncthreads();
    if (slice == 0) shared_key[dim] *= key_inverse_norm;
    __syncthreads();

    // Decay the state and read the memory out of it, one key group per slice.
    float memory_partial = 0.0f;
    for (int key_index = slice; key_index < head_dim; key_index += slices) {
        const int state_index = (head * head_dim + key_index) * head_dim + dim;
        const float decayed = state[state_index] * decay_scale;
        state[state_index] = decayed;
        memory_partial += decayed * shared_key[key_index];
    }
    partials[threadIdx.x] = memory_partial;
    __syncthreads();
    if (slice == 0) {
        float memory = memory_partial;
        for (int other = 1; other < slices; ++other)
            memory += partials[other * head_dim + dim];
        const float value = convolved[total_key_dim * 2 + head * head_dim + dim];
        shared_delta[dim] = (value - memory) * beta;
    }
    __syncthreads();
    const float delta = shared_delta[dim];

    // Write the outer product back into the state and read the core out of it.
    float core_partial = 0.0f;
    for (int key_index = slice; key_index < head_dim; key_index += slices) {
        const int state_index = (head * head_dim + key_index) * head_dim + dim;
        const float updated = state[state_index] + shared_key[key_index] * delta;
        state[state_index] = updated;
        core_partial += updated * shared_query[key_index] * query_inverse_norm;
    }
    __syncthreads();
    partials[threadIdx.x] = core_partial;
    __syncthreads();
    if (slice == 0) {
        float core = core_partial;
        for (int other = 1; other < slices; ++other)
            core += partials[other * head_dim + dim];
        shared_core[dim] = core;
    }
    __syncthreads();
    partials[threadIdx.x] = (int)threadIdx.x < head_dim
        ? shared_core[threadIdx.x] * shared_core[threadIdx.x] : 0.0f;
    const float square = delta_block_sum(partials, head_dim);
    if (threadIdx.x == 0)
        inverse_rms = rsqrtf(square / (float)head_dim + epsilon);
    __syncthreads();
    if (slice == 0) {
        const int output_index = head * head_dim + dim;
        const float gate = gates[output_index];
        output[output_index] = shared_core[dim] * inverse_rms * norm_weights[dim]
            * gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))));
    }
}

extern "C" __global__
void qwen_delta_recurrent_rows(
    const float* convolved, const float* gates,
    const float* beta_logits, const float* decay_logits,
    const float* decay_coefficients, const float* dt_bias,
    const float* norm_weights, float* state, float* output,
    const int rows, const int key_heads, const int value_heads,
    const int head_dim, const float epsilon
) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= value_heads) return;
    const int key_head = head % key_heads;
    const int total_key_dim = key_heads * head_dim;
    const int key_offset = key_head * head_dim;
    __shared__ float query_inverse_norm;
    __shared__ float key_inverse_norm;
    __shared__ float decay_scale;
    __shared__ float beta;
    __shared__ float core_values[256];
    __shared__ float inverse_rms;
    for (int token = 0; token < rows; ++token) {
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        const float* row = convolved + token * (total_key_dim * 2 + value_heads * head_dim);
        if (lane == 0) {
            float query_square = 0.0f, key_square = 0.0f;
            for (int index = 0; index < head_dim; ++index) {
                const float query = row[key_offset + index];
                const float key = row[total_key_dim + key_offset + index];
                query_square += query * query;
                key_square += key * key;
            }
            query_inverse_norm = rsqrtf(query_square + 1.0e-6f)
                * rsqrtf((float)head_dim);
            key_inverse_norm = rsqrtf(key_square + 1.0e-6f);
            beta = 1.0f / (1.0f + expf(-beta_logits[token * value_heads + head]));
            const float softplus_input = decay_logits[token * value_heads + head] + dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input : log1pf(expf(softplus_input));
            decay_scale = expf(decay_coefficients[head] * softplus);
        }
        __syncthreads();
        float core = 0.0f;
        if (lane < head_dim) {
            float memory = 0.0f;
            for (int key_index = 0; key_index < head_dim; ++key_index) {
                const float key = row[total_key_dim + key_offset + key_index]
                    * key_inverse_norm;
                const int state_index =
                    (head * head_dim + key_index) * head_dim + lane;
                state[state_index] *= decay_scale;
                memory += state[state_index] * key;
            }
            const float value = row[total_key_dim * 2 + head * head_dim + lane];
            const float delta = (value - memory) * beta;
            for (int key_index = 0; key_index < head_dim; ++key_index) {
                const float key = row[total_key_dim + key_offset + key_index]
                    * key_inverse_norm;
                const int state_index =
                    (head * head_dim + key_index) * head_dim + lane;
                state[state_index] += key * delta;
                core += state[state_index]
                    * row[key_offset + key_index] * query_inverse_norm;
            }
        }
        core_values[lane] = lane < head_dim ? core : 0.0f;
        __syncthreads();
        if (lane == 0) {
            float square = 0.0f;
            for (int index = 0; index < head_dim; ++index)
                square += core_values[index] * core_values[index];
            inverse_rms = rsqrtf(square / (float)head_dim + epsilon);
        }
        __syncthreads();
        if (lane < head_dim) {
            const int output_index = token * value_heads * head_dim + head * head_dim + lane;
            const float gate = gates[output_index];
            output[output_index] = core * inverse_rms * norm_weights[lane]
                * gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))));
        }
        __syncthreads();
    }
}

extern "C" __global__
void qwen_attention_query(
    const float* projected, const float* norm_weights,
    float* queries, float* gates, const int heads,
    const int head_dim, const int rotary_dim, const int position,
    const float theta, const float epsilon
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const float* source = projected + head * 2 * head_dim;
    float square = 0.0f;
    for (int index = threadIdx.x; index < head_dim; index += blockDim.x)
        square += source[index] * source[index];
    square = block_reduce_sum(square);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) inverse_rms = rsqrtf(square / (float)head_dim + epsilon);
    __syncthreads();
    for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
        float value = source[index] * inverse_rms * norm_weights[index];
        if (index < rotary_dim) {
            const int half = rotary_dim / 2;
            const int pair = index < half ? index : index - half;
            const float other = source[index < half ? index + half : index - half]
                * inverse_rms * norm_weights[index < half ? index + half : index - half];
            const float angle = (float)position
                / powf(theta, 2.0f * (float)pair / (float)rotary_dim);
            value = index < half
                ? value * cosf(angle) - other * sinf(angle)
                : value * cosf(angle) + other * sinf(angle);
        }
        queries[head * head_dim + index] = value;
        gates[head * head_dim + index] = source[head_dim + index];
    }
}

extern "C" __global__
void qwen_attention_key(
    const float* projected, const float* norm_weights,
    float* keys, const int heads, const int head_dim,
    const int rotary_dim, const int position,
    const float theta, const float epsilon
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const float* source = projected + head * head_dim;
    float square = 0.0f;
    for (int index = threadIdx.x; index < head_dim; index += blockDim.x)
        square += source[index] * source[index];
    square = block_reduce_sum(square);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) inverse_rms = rsqrtf(square / (float)head_dim + epsilon);
    __syncthreads();
    for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
        float value = source[index] * inverse_rms * norm_weights[index];
        if (index < rotary_dim) {
            const int half = rotary_dim / 2;
            const int pair = index < half ? index : index - half;
            const float other = source[index < half ? index + half : index - half]
                * inverse_rms * norm_weights[index < half ? index + half : index - half];
            const float angle = (float)position
                / powf(theta, 2.0f * (float)pair / (float)rotary_dim);
            value = index < half
                ? value * cosf(angle) - other * sinf(angle)
                : value * cosf(angle) + other * sinf(angle);
        }
        keys[head * head_dim + index] = value;
    }
}

extern "C" __global__
void qwen_attention_gate(
    const float* attended, const float* gates, float* output,
    const int elements
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x)
        output[index] = attended[index]
            / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gates[index]))));
}

extern "C" __global__
void qwen_shared_scale(
    const float* input, const float* gate,
    float* shared, const int elements
) {
    float partial = 0.0f;
    for (int index = threadIdx.x; index < elements; index += blockDim.x)
        partial += input[index] * gate[index];
    partial = block_reduce_sum(partial);
    __shared__ float scale;
    if (threadIdx.x == 0) scale = 1.0f / (1.0f + expf(-partial));
    __syncthreads();
    for (int index = threadIdx.x; index < elements; index += blockDim.x)
        shared[index] *= scale;
}

extern "C" __global__
void qwen_shared_scale_bf16(
    const float* input, const unsigned short* gate,
    float* shared, const int elements
) {
    float partial = 0.0f;
    for (int index = threadIdx.x; index < elements; index += blockDim.x) {
        const unsigned int bits = ((unsigned int)gate[index]) << 16;
        partial += input[index] * __uint_as_float(bits);
    }
    partial = block_reduce_sum(partial);
    __shared__ float scale;
    if (threadIdx.x == 0) scale = 1.0f / (1.0f + expf(-partial));
    __syncthreads();
    for (int index = threadIdx.x; index < elements; index += blockDim.x)
        shared[index] *= scale;
}

extern "C" __global__
void delta_conv_chunk(
    const float* mixed_qkv, const float* weights, float* state,
    float* output, const int tokens, const int channels,
    const int kernel_size
) {
    // Thread-per-channel variant of delta_conv_sequence for prefill chunks:
    // the window state stays in registers across the token loop instead of
    // one single-thread block per channel.
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels || kernel_size > 8) return;
    float window[8], taps[8];
    float* channel_state = state + channel * kernel_size;
    const float* channel_weights = weights + channel * kernel_size;
    for (int index = 0; index < kernel_size; ++index) {
        window[index] = channel_state[index];
        taps[index] = channel_weights[index];
    }
    for (int token = 0; token < tokens; ++token) {
        for (int index = 0; index + 1 < kernel_size; ++index)
            window[index] = window[index + 1];
        window[kernel_size - 1] = mixed_qkv[(long long)token * channels + channel];
        float value = 0.0f;
        for (int index = 0; index < kernel_size; ++index)
            value += window[index] * taps[index];
        output[(long long)token * channels + channel] = value / (1.0f + expf(-value));
    }
    for (int index = 0; index < kernel_size; ++index)
        channel_state[index] = window[index];
}

extern "C" __global__
void qwen_delta_recurrent_chunk(
    const float* convolved, const float* gates,
    const float* beta_logits, const float* decay_logits,
    const float* decay_coefficients, const float* dt_bias,
    const float* norm_weights, float* state, float* output,
    const int rows, const int key_heads, const int value_heads,
    const int head_dim, const float epsilon
) {
    // DeltaNet recurrence over a whole prefill chunk with the per-head state
    // matrix held in registers: the sequential token loop stays on-chip
    // instead of two global read-modify-write passes over the state per
    // token. Requires head_dim == 128 (host falls back to the _rows kernel
    // otherwise). Launch: grid value_heads, block 128.
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= value_heads || head_dim != 128) return;
    const int key_head = head % key_heads;
    const int total_key_dim = key_heads * 128;
    const int key_offset = key_head * 128;
    float local_state[128];
    #pragma unroll
    for (int key = 0; key < 128; ++key)
        local_state[key] = state[(head * 128 + key) * 128 + lane];
    __shared__ float shared_key[128], shared_query[128];
    __shared__ float query_sums[4], key_sums[4], core_sums[4];
    __shared__ float beta, decay_scale, query_inverse_norm, key_inverse_norm, inverse_rms;
    for (int token = 0; token < rows; ++token) {
        const float* row = convolved
            + (long long)token * (total_key_dim * 2 + value_heads * 128);
        const float query_raw = row[key_offset + lane];
        const float key_raw = row[total_key_dim + key_offset + lane];
        float query_partial = query_raw * query_raw;
        float key_partial = key_raw * key_raw;
        for (int offset = 16; offset > 0; offset >>= 1) {
            query_partial += __shfl_down_sync(0xffffffff, query_partial, offset);
            key_partial += __shfl_down_sync(0xffffffff, key_partial, offset);
        }
        if ((lane & 31) == 0) {
            query_sums[lane >> 5] = query_partial;
            key_sums[lane >> 5] = key_partial;
        }
        __syncthreads();
        if (lane == 0) {
            const float query_square = query_sums[0] + query_sums[1] + query_sums[2] + query_sums[3];
            const float key_square = key_sums[0] + key_sums[1] + key_sums[2] + key_sums[3];
            query_inverse_norm = rsqrtf(query_square + 1.0e-6f) * rsqrtf(128.0f);
            key_inverse_norm = rsqrtf(key_square + 1.0e-6f);
            beta = 1.0f / (1.0f + expf(-beta_logits[token * value_heads + head]));
            const float softplus_input = decay_logits[token * value_heads + head] + dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input : log1pf(expf(softplus_input));
            decay_scale = expf(decay_coefficients[head] * softplus);
        }
        __syncthreads();
        shared_key[lane] = key_raw * key_inverse_norm;
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        shared_query[lane] = query_raw * query_inverse_norm;
        __syncthreads();
        const float value = row[total_key_dim * 2 + head * 128 + lane];
        float memory = 0.0f;
        #pragma unroll
        for (int key = 0; key < 128; ++key) {
            local_state[key] *= decay_scale;
            memory += local_state[key] * shared_key[key];
        }
        const float delta = (value - memory) * beta;
        float core = 0.0f;
        #pragma unroll
        for (int key = 0; key < 128; ++key) {
            local_state[key] += shared_key[key] * delta;
            core += local_state[key] * shared_query[key];
        }
        float core_partial = core * core;
        for (int offset = 16; offset > 0; offset >>= 1)
            core_partial += __shfl_down_sync(0xffffffff, core_partial, offset);
        if ((lane & 31) == 0) core_sums[lane >> 5] = core_partial;
        __syncthreads();
        if (lane == 0) {
            const float square = core_sums[0] + core_sums[1] + core_sums[2] + core_sums[3];
            inverse_rms = rsqrtf(square / 128.0f + epsilon);
        }
        __syncthreads();
        const int output_index = token * value_heads * 128 + head * 128 + lane;
        const float gate = gates[output_index];
        output[output_index] = core * inverse_rms * norm_weights[lane]
            * gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))));
        __syncthreads();
    }
    #pragma unroll
    for (int key = 0; key < 128; ++key)
        state[(head * 128 + key) * 128 + lane] = local_state[key];
}

extern "C" __global__
void q8_matmul_tiled(
    const unsigned char* packed, const float* input, float* output,
    const int input_size, const int output_size, const int rows
) {
    // Shared-tile Q8 GEMM for the chunked prefill path: a 32-row x 32-token
    // block dequantizes each weight tile once into shared memory and shares
    // the activation tile across all rows, so weight and activation bytes
    // are each reused 32x (vs once per token in the warp-per-row kernel).
    // Requires input_size % 32 == 0 (always true for Q8_0 matrices).
    __shared__ float weight_tile[32][33];
    __shared__ float input_tile[32][33];
    const int row_base = blockIdx.x * 32;
    const int token_base = blockIdx.y * 32;
    const int lane_token = threadIdx.x & 31;
    const int row_group = threadIdx.x >> 5;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < input_size; k0 += 32) {
        for (int index = threadIdx.x; index < 32 * 32; index += 256) {
            const int local = index >> 5, k = index & 31;
            const int token = token_base + local;
            input_tile[local][k] = token < rows
                ? input[(long long)token * input_size + k0 + k] : 0.0f;
            const int row = row_base + local;
            if (row < output_size) {
                const long long absolute = (long long)row * input_size + k0 + k;
                const long long block = absolute >> 5;
                const int within = (int)(absolute & 31);
                const float scale = __half2float(*((const __half*)(packed + block * 34)));
                weight_tile[local][k] =
                    (float)(*((const signed char*)(packed + block * 34 + 2 + within))) * scale;
            } else weight_tile[local][k] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float value = input_tile[lane_token][k];
            acc[0] += weight_tile[row_group * 4 + 0][k] * value;
            acc[1] += weight_tile[row_group * 4 + 1][k] * value;
            acc[2] += weight_tile[row_group * 4 + 2][k] * value;
            acc[3] += weight_tile[row_group * 4 + 3][k] * value;
        }
        __syncthreads();
    }
    const int token = token_base + lane_token;
    if (token >= rows) return;
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + row_group * 4 + r;
        if (row < output_size)
            output[(long long)token * output_size + row] = acc[r];
    }
}

template<typename KT, typename VT>
__device__ void kv_prefill_impl(
    const float* queries, const KT* keys, const VT* values,
    float* output, const int heads, const int kv_heads,
    const int head_dim, const int base_position, const int rows,
    const int capacity, const float scale
) {
    // Chunked-prefill attention: each warp owns 4 consecutive query rows for
    // one head and streams the KV cache ONCE for all of them with an online
    // softmax, instead of one full-cache pass per token. No shared memory or
    // block syncs; warps are fully independent. Requires head_dim % 32 == 0
    // and head_dim <= 256 (host falls back to the per-token path otherwise).
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int tile = (blockIdx.y * warps + warp) * 4;
    if (tile >= rows) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const int dims = head_dim / 32;
    const int count = min(4, rows - tile);
    float q[4][8], acc[4][8], m[4], l[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = -3.402823466e+38F;
        l[i] = 0.0f;
        for (int d = 0; d < 8; ++d) {
            acc[i][d] = 0.0f;
            q[i][d] = (i < count && d < dims)
                ? queries[(tile + i) * heads * head_dim
                          + head * head_dim + lane + 32 * d]
                : 0.0f;
        }
    }
    const int last = base_position + tile + count - 1;
    for (int position = 0; position <= last; ++position) {
        float k[8], v[8];
        const KT* key_row = keys + ((long long)kv_head * capacity + position) * head_dim;
        const VT* value_row = values + ((long long)kv_head * capacity + position) * head_dim;
        for (int d = 0; d < 8; ++d) {
            k[d] = d < dims ? kv_ld(key_row, lane + 32 * d) : 0.0f;
            v[d] = d < dims ? kv_ld(value_row, lane + 32 * d) : 0.0f;
        }
        for (int i = 0; i < count; ++i) {
            if (position > base_position + tile + i) continue;
            float partial = 0.0f;
            for (int d = 0; d < 8; ++d) partial += q[i][d] * k[d];
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, offset);
            const float score = partial * scale;
            const float peak = fmaxf(m[i], score);
            const float rescale = expf(m[i] - peak);
            const float weight = expf(score - peak);
            l[i] = l[i] * rescale + weight;
            for (int d = 0; d < 8; ++d)
                acc[i][d] = acc[i][d] * rescale + weight * v[d];
            m[i] = peak;
        }
    }
    for (int i = 0; i < count; ++i) {
        const float inverse = 1.0f / l[i];
        for (int d = 0; d < dims; ++d)
            output[(tile + i) * heads * head_dim + head * head_dim + lane + 32 * d]
                = acc[i][d] * inverse;
    }
}
#define KV_PREFILL(name, KT, VT) \
extern "C" __global__ void name( \
    const float* queries, const KT* keys, const VT* values, \
    float* output, const int heads, const int kv_heads, \
    const int head_dim, const int base_position, const int rows, \
    const int capacity, const float scale \
) { kv_prefill_impl<KT, VT>(queries, keys, values, output, heads, kv_heads, head_dim, base_position, rows, capacity, scale); }
// Fused prefill only for matched K/V precision; the host uses the per-token
// score/value path when cache_type_k != cache_type_v.
KV_PREFILL(kv_attention_prefill, float, float)
KV_PREFILL(kv_attention_prefill_f16, __half, __half)
KV_PREFILL(kv_attention_prefill_bf16, __nv_bfloat16, __nv_bfloat16)
#undef KV_PREFILL

// q8_0 fused prefill (diagonal K==V==q8_0). Same online-softmax as the templated
// path but K/V rows are (head_dim/32)-block byte rows read via kv_ld_q8.
extern "C" __global__ void kv_attention_prefill_q8(
    const float* queries, const unsigned char* keys, const unsigned char* values,
    float* output, const int heads, const int kv_heads,
    const int head_dim, const int base_position, const int rows,
    const int capacity, const float scale
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int tile = (blockIdx.y * warps + warp) * 4;
    if (tile >= rows) return;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const int dims = head_dim / 32;
    const int count = min(4, rows - tile);
    float q[4][8], acc[4][8], m[4], l[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = -3.402823466e+38F; l[i] = 0.0f;
        for (int d = 0; d < 8; ++d) {
            acc[i][d] = 0.0f;
            q[i][d] = (i < count && d < dims)
                ? queries[(tile + i) * heads * head_dim + head * head_dim + lane + 32 * d]
                : 0.0f;
        }
    }
    const int last = base_position + tile + count - 1;
    for (int position = 0; position <= last; ++position) {
        float k[8], v[8];
        const unsigned char* key_row = keys + ((long long)kv_head * capacity + position) * dims * 34;
        const unsigned char* value_row = values + ((long long)kv_head * capacity + position) * dims * 34;
        for (int d = 0; d < 8; ++d) {
            k[d] = d < dims ? kv_ld_q8(key_row, lane + 32 * d) : 0.0f;
            v[d] = d < dims ? kv_ld_q8(value_row, lane + 32 * d) : 0.0f;
        }
        for (int i = 0; i < count; ++i) {
            if (position > base_position + tile + i) continue;
            float partial = 0.0f;
            for (int d = 0; d < 8; ++d) partial += q[i][d] * k[d];
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, offset);
            const float score = partial * scale;
            const float peak = fmaxf(m[i], score);
            const float rescale = expf(m[i] - peak);
            const float weight = expf(score - peak);
            l[i] = l[i] * rescale + weight;
            for (int d = 0; d < 8; ++d) acc[i][d] = acc[i][d] * rescale + weight * v[d];
            m[i] = peak;
        }
    }
    for (int i = 0; i < count; ++i) {
        const float inverse = 1.0f / l[i];
        for (int d = 0; d < dims; ++d)
            output[(tile + i) * heads * head_dim + head * head_dim + lane + 32 * d]
                = acc[i][d] * inverse;
    }
}

extern "C" __global__
void qwen_argmax(const float* values, unsigned int* output, const int elements) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int best = 0;
    float maximum = values[0];
    for (int index = 1; index < elements; ++index) {
        if (values[index] > maximum) { maximum = values[index]; best = index; }
    }
    *output = (unsigned int)best;
}
)COLIBRI_CUDA";
}
