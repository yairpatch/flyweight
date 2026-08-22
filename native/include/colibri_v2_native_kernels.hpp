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


// bf16 output tables, as shipped by the NVFP4 Qwen3.6 checkpoints. Same warp
// layout as the Q8_0 variant above, only the weight decode differs.
extern "C" __global__
void qwen_bf16_lm_head_argmax_rows(
    const unsigned short* weights, const float* vectors,
    unsigned long long* winners, const int input_size,
)COLIBRI_CUDA"
R"COLIBRI_CUDA(    const int output_size, const int rows
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
        const float* row = convolved + token * (total_key_dim * 2 + value_heads * head_dim);
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(        queries[head * head_dim + index] = value;
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

// YaRN interpolation ramp. Pairs below `low` extrapolate (plain RoPE, so the
// high-frequency channels keep their trained resolution), pairs above `high`
// interpolate by the full context-scaling factor, and the band between the two
// blends. `low`/`high` are the correction dimensions the host derives from
// beta_fast/beta_slow.
__device__ __forceinline__ float laguna_yarn_ramp(
    const float low, const float high, const int pair
) {
    const float y = ((float)pair - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// One rotation angle, YaRN-corrected when ext_factor is non-zero. With
// ext_factor 0 and freq_scale 1 this reduces exactly to the plain RoPE angle
// the Qwen kernels use, so sliding-window layers stay bit-comparable.
__device__ __forceinline__ void laguna_rope_angle(
    const int pair, const int rotary_dim, const int position,
    const float theta, const float freq_scale, const float ext_factor,
    const float mscale, const float corr_low, const float corr_high,
    float* cos_out, float* sin_out
) {
    const float extrapolated = (float)position
        / powf(theta, 2.0f * (float)pair / (float)rotary_dim);
    float angle = freq_scale * extrapolated;
    if (ext_factor != 0.0f) {
        const float mix = laguna_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        angle = angle * (1.0f - mix) + extrapolated * mix;
    }
    *cos_out = cosf(angle) * mscale;
    *sin_out = sinf(angle) * mscale;
}

// Laguna Q/K projection tail: per-head RMS norm against learned weights, then
// RoPE over the leading `rotary_dim` channels. Channels past rotary_dim pass
// through unrotated, which is how the full-attention layers use 64 of their 128
// head channels. Q and K differ only in head count, so both use this kernel.
extern "C" __global__
void laguna_head_norm_rope(
    const float* projected, const float* norm_weights, float* output,
    const int heads, const int head_dim, const int rotary_dim,
    const int position, const float theta, const float epsilon,
    const float freq_scale, const float ext_factor, const float mscale,
    const float corr_low, const float corr_high
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
            const int partner = index < half ? index + half : index - half;
            const float other =
                source[partner] * inverse_rms * norm_weights[partner];
            float cos_angle, sin_angle;
            laguna_rope_angle(pair, rotary_dim, position, theta, freq_scale,
                              ext_factor, mscale, corr_low, corr_high,
                              &cos_angle, &sin_angle);
            value = index < half
                ? value * cos_angle - other * sin_angle
                : value * cos_angle + other * sin_angle;
        }
        output[head * head_dim + index] = value;
    }
}

// Laguna's attention output gate is one softplus-activated scalar per head,
// broadcast over that head's channels, applied before the output projection.
extern "C" __global__
void laguna_attention_gate(
    const float* attended, const float* gates, float* output,
    const int heads, const int head_dim
) {
    const int elements = heads * head_dim;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += blockDim.x * gridDim.x) {
        const float gate = gates[index / head_dim];
        // softplus; the linear branch keeps expf from overflowing for large
        // gates, where log1p(exp(x)) is x to well within float precision.
        const float scale = gate > 20.0f ? gate : log1pf(expf(gate));
        output[index] = attended[index] * scale;
    }
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(// Muse Glimmer Q/K projection tail: per-head RMS norm against learned weights,
// then RoPE over the leading `rotary_dim` channels.
//
// Two things differ from the Laguna kernel above and neither is cosmetic.
// First, the rotation pairs *consecutive* channels (2i, 2i+1) rather than
// splitting the head in half -- llama.cpp calls this NORM rope, and the
// conversion script unpermutes the HF weights specifically to feed it. Running
// the half-split form here produces fluent-looking but wrong text.
// Second, `rotary_dim` of 0 skips the rotation entirely, which is how the
// full-attention layers run NoPE while the sliding-window layers rotate.
//
// The q_norm weights carry the model's `qk_scale_factor` broadcast across the
// head (the k_norm weights are ones), so the scale rides in for free here and
// the attention scale stays the plain 1/sqrt(head_dim).
extern "C" __global__
void muse_head_norm_rope(
    const float* projected, const float* norm_weights, float* output,
    const int heads, const int head_dim, const int rotary_dim,
    const int position, const float theta, const float epsilon
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
            const int pair = index >> 1;
            const int partner = index ^ 1;
            const float other =
                source[partner] * inverse_rms * norm_weights[partner];
            const float angle = (float)position
                / powf(theta, 2.0f * (float)pair / (float)rotary_dim);
            const float cos_angle = cosf(angle), sin_angle = sinf(angle);
            value = (index & 1)
                ? value * cos_angle + other * sin_angle
                : value * cos_angle - other * sin_angle;
        }
        output[head * head_dim + index] = value;
    }
}

// Muse Glimmer's output multiplier and tanh logit softcap, fused.
//
// Both are monotonic, so neither can move an argmax and the greedy decode path
// skips this entirely. It runs for sampling, where the compression genuinely
// changes the distribution that temperature and top-p see.
extern "C" __global__
void muse_logit_softcap(
    float* logits, const int vocabulary, const float scale, const float softcap
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < vocabulary; index += blockDim.x * gridDim.x) {
        const float scaled = logits[index] * scale;
        logits[index] = softcap > 0.0f ? softcap * tanhf(scaled / softcap) : scaled;
    }
}

// Sigmoid-gated top-k routing with a score-correction bias (DeepSeek-V3 style,
// which Laguna shares). Selection ranks on score + bias, but the returned
// weights are the unbiased sigmoid scores: the bias steers load balancing only
// and must not reach the expert combination.
extern "C" __global__
void route_topk_sigmoid_bias(
    const float* logits,
    const float* bias,
    int* selected,
    float* routing_weights,
    const int experts,
    const int top_k,
    const int normalize,
    const float weight_scale
) {
    extern __shared__ float shared[];
    float* scores = shared;              // unbiased sigmoid probabilities
    float* ranking = shared + experts;   // selection scores, consumed in place
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        const float probability = 1.0f / (1.0f + expf(-logits[index]));
        scores[index] = probability;
        ranking[index] = probability + (bias ? bias[index] : 0.0f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_index = 0;
            float best_value = -3.402823466e+38F;
            for (int expert = 0; expert < experts; ++expert) {
                if (ranking[expert] > best_value) {
                    best_value = ranking[expert];
                    best_index = expert;
                }
            }
            selected[rank] = best_index;
            routing_weights[rank] = scores[best_index];
            total += scores[best_index];
            ranking[best_index] = -3.402823466e+38F;
        }
        const float inverse =
            normalize && total > 0.0f ? weight_scale / total : weight_scale;
        for (int rank = 0; rank < top_k; ++rank) routing_weights[rank] *= inverse;
    }
}

// Row-batched Laguna attention gate: one softplus scalar per (row, head),
// broadcast over that head's channels across the whole prefill chunk.
extern "C" __global__
void laguna_attention_gate_rows(
    const float* attended, const float* gates, float* output,
    const int rows, const int heads, const int head_dim
) {
    const long long elements = (long long)rows * heads * head_dim;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x) {
        const float gate = gates[index / head_dim];
        const float scale = gate > 20.0f ? gate : log1pf(expf(gate));
        output[index] = attended[index] * scale;
    }
}

// Row-batched counterpart of route_topk_sigmoid_bias: one block routes one
// token, so a whole prefill chunk routes in a single launch.
extern "C" __global__
void route_topk_sigmoid_bias_rows(
    const float* logits,
    const float* bias,
    int* selected,
    float* routing_weights,
    const int rows,
    const int experts,
    const int top_k,
    const int normalize,
    const float weight_scale
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float* scores = shared;
    float* ranking = shared + experts;
    const float* row_logits = logits + (long long)row * experts;
    for (int index = threadIdx.x; index < experts; index += blockDim.x) {
        const float probability = 1.0f / (1.0f + expf(-row_logits[index]));
        scores[index] = probability;
        ranking[index] = probability + (bias ? bias[index] : 0.0f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int* row_selected = selected + (long long)row * top_k;
        float* row_weights = routing_weights + (long long)row * top_k;
        float total = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_index = 0;
            float best_value = -3.402823466e+38F;
            for (int expert = 0; expert < experts; ++expert) {
                if (ranking[expert] > best_value) {
                    best_value = ranking[expert];
                    best_index = expert;
                }
            }
            row_selected[rank] = best_index;
            row_weights[rank] = scores[best_index];
            total += scores[best_index];
            ranking[best_index] = -3.402823466e+38F;
        }
        const float inverse =
            normalize && total > 0.0f ? weight_scale / total : weight_scale;
        for (int rank = 0; rank < top_k; ++rank) row_weights[rank] *= inverse;
    }
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
)COLIBRI_CUDA"
R"COLIBRI_CUDA(            const float softplus_input = decay_logits[token * value_heads + head] + dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input : log1pf(expf(softplus_input));
            decay_scale = expf(decay_coefficients[head] * softplus);
        }
        __syncthreads();
        shared_key[lane] = key_raw * key_inverse_norm;
        shared_query[lane] = query_raw * query_inverse_norm;
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

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__
void bailing_q4k_matvec(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    // Q4_K matvec that unpacks a sub-block's scales ONCE instead of per
    // element.
    //
    // q4k_matvec_transposed calls q4k_value per value, and that function
    // re-derives the block offsets and reloads both f16 scales every time --
    // roughly five loads and two half-to-float conversions to produce four
    // bits. Measured, the existing kernel runs these shapes at 58-131 GB/s on
    // a card that does ~670: instruction-bound, not bandwidth-bound.
    //
    // Here each warp owns one 32-element sub-block and lane i takes element i,
    // so the scale unpack is amortized 32x and the quant byte load is the only
    // per-element memory traffic. Same arrangement as qwen_q4k_dot_row on the
    // CPU side, for the same reason.
    //
    // Requires input_size % 32 == 0, which the quantizer already guarantees.
    const int row = blockIdx.x;
    if (row >= output_size) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const long long row_base = (long long)row * input_size;
    const int sub_blocks = input_size >> 5;
    float partial = 0.0f;
    for (int sb = warp; sb < sub_blocks; sb += warps) {
        const long long absolute = row_base + (long long)sb * 32;
        const unsigned char* base = packed + (absolute >> 8) * 144;
        const int within = (int)(absolute & 255);
        const __half* halves = (const __half*)base;
        const float d = __half2float(halves[0]);
        const float dmin = __half2float(halves[1]);
        const unsigned char* scales = base + 4;
        const int group = within >> 6;
        const int half_index = (within >> 5) & 1;
        const int index = group * 2 + half_index;
        int scale, minimum;
        if (index < 4) {
            scale = scales[index] & 63;
            minimum = scales[index + 4] & 63;
        } else {
            scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
            minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        }
        const unsigned char byte = base[16 + group * 32 + lane];
        const int quant = half_index == 0 ? (byte & 15) : (byte >> 4);
        partial += (d * scale * (float)quant - dmin * (float)minimum)
                 * vector[sb * 32 + lane];
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[row] = partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Small BailingMoE3 helpers. Each is the device half of a host function in
// colibri_v2_bailing.hpp and is checked against it the same way.

extern "C" __global__
void bailing_copy(float* destination, const float* source, const int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x)
        destination[i] = source[i];
}

extern "C" __global__
void bailing_partial_rope(
    float* rows, const int heads, const int head_dim, const int rope_dim,
    const int position, const float theta
) {
    // Adjacent-pair (NORM) rotation over the trailing rope_dim channels. See
    // partial_rope_norm for why this is adjacent pairs and not the half-split
    // form the reference appears to use.
    const int head = blockIdx.x;
    if (head >= heads) return;
    float* span = rows + (long long)head * head_dim + (head_dim - rope_dim);
    for (int pair = threadIdx.x; pair * 2 < rope_dim; pair += blockDim.x) {
        const float exponent = (float)(2 * pair) / (float)rope_dim;
        // Accurate variants, not the __-prefixed fast-math intrinsics: the CPU
        // shim that compiles this corpus as host C++ does not define those, and
        // rope angle error compounds over long contexts.
        const float frequency = powf(theta, -exponent);
        const float angle = (float)position * frequency;
        const float cosine = cosf(angle), sine = sinf(angle);
        const float even = span[pair * 2], odd = span[pair * 2 + 1];
        span[pair * 2] = even * cosine - odd * sine;
        span[pair * 2 + 1] = odd * cosine + even * sine;
    }
}

extern "C" __global__
void bailing_split_query(
    const float* query, float* nope, float* rope,
    const int heads, const int qk_nope, const int qk_rope
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int width = qk_nope + qk_rope;
    for (int i = threadIdx.x; i < qk_nope; i += blockDim.x)
        nope[(long long)head * qk_nope + i] = query[(long long)head * width + i];
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x)
        rope[(long long)head * qk_rope + i] = query[(long long)head * width + qk_nope + i];
}

extern "C" __global__
void bailing_head_gate(
    const float* logits, const int heads, const int value_dim, float* attention
) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const float gate = 1.0f / (1.0f + __expf(-logits[head]));
    for (int i = threadIdx.x; i < value_dim; i += blockDim.x)
        attention[(long long)head * value_dim + i] *= gate;
}

extern "C" __global__
void bailing_short_conv(
    float* values, const float* weights, float* window,
    const int channels, const int width
) {
    // Causal depthwise conv over one token plus SiLU, advancing the window.
    const int history = width - 1;
    for (int channel = blockIdx.x * blockDim.x + threadIdx.x; channel < channels;
         channel += blockDim.x * gridDim.x) {
        const float* taps = weights + (long long)channel * width;
        float* past = window + (long long)channel * history;
        const float input = values[channel];
        float total = 0.0f;
        for (int i = 0; i < history; ++i) total += past[i] * taps[i];
        total += input * taps[history];
        values[channel] = total / (1.0f + __expf(-total));
        for (int i = 0; i + 1 < history; ++i) past[i] = past[i + 1];
        if (history > 0) past[history - 1] = input;
    }
}

extern "C" __global__
void bailing_gated_head_norm(
    float* rows, const float* gain, const float* gate,
    const int heads, const int head_dim, const float epsilon
) {
    // RMS norm per head with a shared gain, times the sigmoid of the gate.
    // FusedRMSNormGated(activation='sigmoid'), not a norm followed by a gate.
    const int head = blockIdx.x;
    if (head >= heads) return;
    float* row = rows + (long long)head * head_dim;
    const float* head_gate = gate + (long long)head * head_dim;
    __shared__ float reduce[128];
    float partial = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        partial += row[i] * row[i];
    reduce[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduce[0] / (float)head_dim + epsilon);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        row[i] = row[i] * inverse * gain[i] / (1.0f + __expf(-head_gate[i]));
}

extern "C" __global__
void bailing_swiglu(
    const float* gate, const float* up, const int size, const float limit,
    float* output
) {
    const bool clamped = limit > 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        float g = gate[i], u = up[i];
        if (clamped) {
            g = fminf(fmaxf(g, -limit), limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        output[i] = (g / (1.0f + __expf(-g))) * u;
    }
}

// Complete row-batched Bailing prefill helpers. All layouts are row-major and
// remain device-resident across a prompt tile.
extern "C" __global__
void bailing_mla_prepare_rows(
    const float* compressed, const float* query, const float* latent_gain,
    float* latents, float* rope_keys, float* query_nope, float* query_rope,
    const int rows, const int base_position, const int heads,
    const int qk_nope, const int qk_rope, const int kv_lora,
    const float epsilon, const float theta
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int compressed_width = kv_lora + qk_rope;
    const float* source = compressed + (long long)row * compressed_width;
    float* latent = latents + (long long)(base_position + row) * kv_lora;
    __shared__ float reduce[256];
    float partial = 0.0f;
    for (int i = lane; i < kv_lora; i += 256) partial += source[i] * source[i];
    reduce[lane] = partial;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] += reduce[lane + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduce[0] / (float)kv_lora + epsilon);
    for (int i = lane; i < kv_lora; i += 256)
        latent[i] = source[i] * inverse * latent_gain[i];

    const int position = base_position + row;
    float* cached_rope = rope_keys + (long long)position * qk_rope;
    for (int pair = lane; pair * 2 < qk_rope; pair += 256) {
        const float exponent = (float)(2 * pair) / (float)qk_rope;
        const float angle = (float)position * powf(theta, -exponent);
        const float cosine = cosf(angle), sine = sinf(angle);
        const float even = source[kv_lora + pair * 2];
        const float odd = source[kv_lora + pair * 2 + 1];
        cached_rope[pair * 2] = even * cosine - odd * sine;
        cached_rope[pair * 2 + 1] = odd * cosine + even * sine;
    }
    const int qk = qk_nope + qk_rope;
    const float* query_row = query + (long long)row * heads * qk;
    float* nope_row = query_nope + (long long)row * heads * qk_nope;
    float* rope_row = query_rope + (long long)row * heads * qk_rope;
    for (int index = lane; index < heads * qk_nope; index += 256) {
        const int head = index / qk_nope, channel = index % qk_nope;
        nope_row[index] = query_row[head * qk + channel];
    }
    for (int index = lane; index < heads * (qk_rope / 2); index += 256) {
        const int head = index / (qk_rope / 2), pair = index % (qk_rope / 2);
        const float exponent = (float)(2 * pair) / (float)qk_rope;
        const float angle = (float)position * powf(theta, -exponent);
        const float cosine = cosf(angle), sine = sinf(angle);
        const float* span = query_row + head * qk + qk_nope;
        const float even = span[pair * 2], odd = span[pair * 2 + 1];
        float* target = rope_row + head * qk_rope;
        target[pair * 2] = even * cosine - odd * sine;
        target[pair * 2 + 1] = odd * cosine + even * sine;
    }
}

extern "C" __global__
void bailing_mla_project_rows(
    const float* query_nope, const float* kv_b, float* projected,
    const int rows, const int heads, const int qk_nope,
    const int v_head_dim, const int kv_lora
) {
    const int query_index = blockIdx.x;
    const int row = query_index / heads, head = query_index % heads;
    if (row >= rows) return;
    const float* weights = kv_b +
        (long long)head * (qk_nope + v_head_dim) * kv_lora;
    const float* query = query_nope +
        ((long long)row * heads + head) * qk_nope;
    float* target = projected + ((long long)row * heads + head) * kv_lora;
    for (int column = threadIdx.x; column < kv_lora; column += blockDim.x) {
        float total = 0.0f;
        for (int i = 0; i < qk_nope; ++i)
            total += query[i] * weights[(long long)i * kv_lora + column];
        target[column] = total;
    }
}

extern "C" __global__
void bailing_mla_scores_rows(
    const float* projected, const float* query_rope,
    const float* latents, const float* rope_keys, float* scores,
    const int rows, const int base_position, const int capacity,
    const int heads, const int qk_nope, const int qk_rope, const int kv_lora
) {
    const int query_index = blockIdx.x;
    const int row = query_index / heads, head = query_index % heads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int position = blockIdx.y * 8 + warp;
    const int visible = base_position + row + 1;
    if (row >= rows || position >= visible) return;
    const float* p = projected + ((long long)row * heads + head) * kv_lora;
    const float* latent = latents + (long long)position * kv_lora;
    const float* qr = query_rope + ((long long)row * heads + head) * qk_rope;
    const float* kr = rope_keys + (long long)position * qk_rope;
    float total = 0.0f;
    for (int i = lane; i < kv_lora; i += 32) total += p[i] * latent[i];
    for (int i = lane; i < qk_rope; i += 32) total += qr[i] * kr[i];
    for (int offset = 16; offset; offset >>= 1)
        total += __shfl_down_sync(0xffffffff, total, offset);
    if (lane == 0)
        scores[(long long)query_index * capacity + position] =
            total * rsqrtf((float)(qk_nope + qk_rope));
}

// bailing_mla_scores_rows with the head dimension folded into the block: a
// warp owns one (row, position) pair, loads each latent and rope-key value
// once, and computes every head's dot from it, cutting the quadratic phase's
// cache traffic by the head count. Per-head order matches
// bailing_mla_scores_rows exactly, so the scores are bit-identical.
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// The prefill score phase written as the GEMM it is:
// [rows*heads, kv_lora+qk_rope] x [kv_lora+qk_rope, positions], with the
// causal mask applied at the store. A 64x64 output tile per 256-thread
// block, 4x4 accumulators per thread, so each shared-memory value feeds four
// FMAs -- the pairwise kernels above spend one shared or global load per FMA
// and are instruction-bound. Every latent value is read once per query tile
// instead of once per (row, head). Summation runs k-ascending (kv_lora then
// rope), which rounds differently from the warp-reduced original; the logits
// gate allows 1e-3.
extern "C" __global__
void bailing_mla_scores_rows_gemm(
    const float* projected, const float* query_rope,
    const float* latents, const float* rope_keys, float* scores,
    const int rows, const int base_position, const int capacity,
    const int heads, const int qk_nope, const int qk_rope, const int kv_lora
) {
    const int q0 = blockIdx.x * 64;
    const int p0 = blockIdx.y * 64;
    const int queries = rows * heads;
    if (q0 >= queries) return;
    // The deepest row in this query tile decides whether any of these
    // positions are visible at all; whole tiles above the causal diagonal
    // exit before touching memory.
    const int row_max = min(rows - 1, (q0 + 63) / heads);
    if (p0 >= base_position + row_max + 1) return;
    const int depth = kv_lora + qk_rope;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    __shared__ float a_tile[16][64];
    __shared__ float b_tile[16][64];
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < depth; k0 += 16) {
        __syncthreads();
        for (int n = threadIdx.x; n < 1024; n += 256) {
            const int k = n & 15, m = n >> 4;
            const int c = k0 + k;
            const int q = q0 + m;
            float value = 0.0f;
            if (q < queries && c < depth)
                value = c < kv_lora
                    ? projected[(long long)q * kv_lora + c]
                    : query_rope[(long long)q * qk_rope + (c - kv_lora)];
            a_tile[k][m] = value;
            const int p = p0 + m;
            float key = 0.0f;
            if (p < base_position + rows && c < depth)
                key = c < kv_lora
                    ? latents[(long long)p * kv_lora + c]
                    : rope_keys[(long long)p * qk_rope + (c - kv_lora)];
            b_tile[k][m] = key;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a_frag[4], b_frag[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
    }
    const float scale = rsqrtf((float)(qk_nope + qk_rope));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = q0 + ty + 16 * i;
        if (q >= queries) continue;
        const int visible = base_position + q / heads + 1;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int p = p0 + tx + 16 * j;
            if (p < visible)
                scores[(long long)q * capacity + p] = acc[i][j] * scale;
        }
    }
}

// The value phase as the same GEMM shape: [rows*heads, positions] x
// [positions, kv_lora]. Masked score slots load as zero -- softmax never
// wrote them -- and the position dimension is the reduction, ascending, so a
// row's sum matches the serial kernel up to tile-boundary rounding.
extern "C" __global__
void bailing_mla_accumulate_rows_gemm(
    const float* scores, const float* latents, float* accumulated,
    const int rows, const int base_position, const int capacity,
    const int heads, const int kv_lora
) {
    const int q0 = blockIdx.x * 64;
    const int c0 = blockIdx.y * 64;
    const int queries = rows * heads;
    if (q0 >= queries || c0 >= kv_lora) return;
    const int row_max = min(rows - 1, (q0 + 63) / heads);
    const int visible_max = base_position + row_max + 1;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    __shared__ float a_tile[16][64];
    __shared__ float b_tile[16][64];
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < visible_max; k0 += 16) {
        __syncthreads();
        for (int n = threadIdx.x; n < 1024; n += 256) {
            const int k = n & 15, m = n >> 4;
            const int p = k0 + k;
            const int q = q0 + m;
            a_tile[k][m] =
                (q < queries && p < base_position + q / heads + 1)
                    ? scores[(long long)q * capacity + p] : 0.0f;
            const int c = c0 + m;
            b_tile[k][m] = (p < visible_max && c < kv_lora)
                ? latents[(long long)p * kv_lora + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a_frag[4], b_frag[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = q0 + ty + 16 * i;
        if (q >= queries) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + tx + 16 * j;
            if (c < kv_lora)
                accumulated[(long long)q * kv_lora + c] = acc[i][j];
        }
    }
}

extern "C" __global__
void bailing_mla_softmax_rows(
    float* scores, const int rows, const int base_position,
    const int capacity, const int heads
) {
    const int query_index = blockIdx.x;
    const int row = query_index / heads;
    if (row >= rows) return;
    const int visible = base_position + row + 1, lane = threadIdx.x;
    float* values = scores + (long long)query_index * capacity;
    __shared__ float reduce[128];
    float local = -3.0e38f;
    for (int i = lane; i < visible; i += 128) local = fmaxf(local, values[i]);
    reduce[lane] = local;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
        __syncthreads();
    }
    const float peak = reduce[0];
    local = 0.0f;
    for (int i = lane; i < visible; i += 128) {
        const float value = __expf(values[i] - peak);
        values[i] = value;
        local += value;
    }
    reduce[lane] = local;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] += reduce[lane + stride];
        __syncthreads();
    }
    const float inverse = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (int i = lane; i < visible; i += 128) values[i] *= inverse;
}

extern "C" __global__
void bailing_mla_accumulate_rows(
    const float* scores, const float* latents, float* accumulated,
    const int rows, const int base_position, const int capacity,
    const int heads, const int kv_lora
) {
    const int query_index = blockIdx.x;
    const int row = query_index / heads;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || column >= kv_lora) return;
    const int visible = base_position + row + 1;
    const float* weights = scores + (long long)query_index * capacity;
    float total = 0.0f;
    for (int position = 0; position < visible; ++position)
        total += weights[position] * latents[(long long)position * kv_lora + column];
    accumulated[(long long)query_index * kv_lora + column] = total;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__
void bailing_mla_fused_rows(
    const float* projected, const float* query_rope,
    const float* latents, const float* rope_keys, float* accumulated,
    const int rows, const int base_position, const int heads,
    const int qk_nope, const int qk_rope, const int kv_lora
) {
    const int query_index = blockIdx.x;
    const int row = query_index / heads, head = query_index % heads;
    const int lane = threadIdx.x;
    const int warp = lane >> 5, warp_lane = lane & 31;
    if (row >= rows || kv_lora > blockDim.x || blockDim.x != 512) return;
    const int visible = base_position + row + 1;
    const float* query = projected +
        ((long long)row * heads + head) * kv_lora;
    const float* rope = query_rope +
        ((long long)row * heads + head) * qk_rope;
    float value = 0.0f;
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float old_factor;
    __shared__ float tile_weights[16];
    if (lane == 0) {
        running_max = -3.402823466e+38F;
        running_sum = 0.0f;
    }
    __syncthreads();
    const float scale = rsqrtf((float)(qk_nope + qk_rope));
    for (int position_base = 0; position_base < visible; position_base += 16) {
        const int position = position_base + warp;
        float partial = 0.0f;
        if (position < visible) {
            const float* latent = latents + (long long)position * kv_lora;
            for (int i = warp_lane; i < kv_lora; i += 32)
                partial += query[i] * latent[i];
            for (int i = warp_lane; i < qk_rope; i += 32)
                partial += rope[i] *
                    rope_keys[(long long)position * qk_rope + i];
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        if (warp_lane == 0)
            tile_weights[warp] = position < visible
                ? partial * scale : -3.402823466e+38F;
        __syncthreads();
        if (lane == 0) {
            float peak = running_max;
            const int count = min(16, visible - position_base);
            for (int i = 0; i < count; ++i)
                peak = fmaxf(peak, tile_weights[i]);
            old_factor = running_sum > 0.0f
                ? __expf(running_max - peak) : 0.0f;
            float tile_sum = 0.0f;
            for (int i = 0; i < count; ++i) {
                tile_weights[i] = __expf(tile_weights[i] - peak);
                tile_sum += tile_weights[i];
            }
            running_sum = running_sum * old_factor + tile_sum;
            running_max = peak;
        }
        __syncthreads();
        if (lane < kv_lora) {
            float update = 0.0f;
            const int count = min(16, visible - position_base);
            for (int i = 0; i < count; ++i)
                update += tile_weights[i] * latents[
                    (long long)(position_base + i) * kv_lora + lane];
            value = value * old_factor + update;
        }
        __syncthreads();
    }
    if (lane < kv_lora)
        accumulated[(long long)query_index * kv_lora + lane] =
            running_sum > 0.0f ? value / running_sum : 0.0f;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(

extern "C" __global__
void bailing_mla_output_rows(
    const float* accumulated, const float* kv_b, const float* gates,
    float* output, const int rows, const int heads, const int qk_nope,
    const int v_head_dim, const int kv_lora
) {
    const int query_index = blockIdx.x;
    const int row = query_index / heads, head = query_index % heads;
    if (row >= rows) return;
    const float* values = accumulated + (long long)query_index * kv_lora;
    const float* weights = kv_b +
        ((long long)head * (qk_nope + v_head_dim) + qk_nope) * kv_lora;
    const float gate = 1.0f / (1.0f + __expf(-gates[(long long)row * heads + head]));
    for (int output_row = threadIdx.x; output_row < v_head_dim; output_row += blockDim.x) {
        const float* source = weights + (long long)output_row * kv_lora;
        float total = 0.0f;
        for (int i = 0; i < kv_lora; ++i) total += source[i] * values[i];
        output[((long long)row * heads + head) * v_head_dim + output_row] = total * gate;
    }
}

extern "C" __global__
void bailing_short_conv_rows(
    float* values, const float* weights, float* window,
    const int rows, const int channels, const int width
) {
    const int history = width - 1;
    for (int channel = blockIdx.x * blockDim.x + threadIdx.x; channel < channels;
         channel += blockDim.x * gridDim.x) {
        const float* taps = weights + (long long)channel * width;
        float* past = window + (long long)channel * history;
        for (int row = 0; row < rows; ++row) {
            const float input = values[(long long)row * channels + channel];
            float total = 0.0f;
            for (int i = 0; i < history; ++i) total += past[i] * taps[i];
            total += input * taps[history];
            values[(long long)row * channels + channel] =
                total / (1.0f + __expf(-total));
            for (int i = 0; i + 1 < history; ++i) past[i] = past[i + 1];
            if (history > 0) past[history - 1] = input;
        }
    }
}

extern "C" __global__
void bailing_gated_head_norm_rows(
    float* values, const float* gain, const float* gate,
    const int rows, const int heads, const int head_dim, const float epsilon
) {
    const int index = blockIdx.x;
    const int row_index = index / heads, head = index % heads;
    if (row_index >= rows) return;
    float* row = values + ((long long)row_index * heads + head) * head_dim;
    const float* row_gate = gate + ((long long)row_index * heads + head) * head_dim;
    __shared__ float reduce[128];
    float partial = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) partial += row[i] * row[i];
    reduce[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduce[0] / (float)head_dim + epsilon);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        row[i] = row[i] * inverse * gain[i] / (1.0f + __expf(-row_gate[i]));
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(extern "C" __global__
void bailing_route_rows(
    const float* logits, const float* bias, int* selected, float* weights,
    int* expert_counts, int* expert_routes, const int max_routes,
    const int rows, const int experts, const int top_k,
    const int groups, const int groups_used,
    const int normalize, const float weight_scale
) {
    const int row = blockIdx.x;
    if (row >= rows || experts > 256 || groups > 64 || top_k > 32) return;
    __shared__ float probability[256], ranking[256], group_score[64];
    __shared__ unsigned char allowed[256], group_kept[64];
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float p = 1.0f / (1.0f + expf(-logits[(long long)row * experts + expert]));
        probability[expert] = p;
        ranking[expert] = p + (bias ? bias[expert] : 0.0f);
        allowed[expert] = 1;
    }
    for (int group = threadIdx.x; group < groups; group += blockDim.x) group_kept[group] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    const bool limited = groups > 1 && groups_used > 0 && groups_used < groups &&
                         experts % groups == 0;
    if (limited) {
        const int span = experts / groups;
        for (int group = 0; group < groups; ++group) {
            float best = -3.402823466e+38F, second = -3.402823466e+38F;
            for (int offset = 0; offset < span; ++offset) {
                const float value = ranking[group * span + offset];
                if (value > best) { second = best; best = value; }
                else if (value > second) second = value;
            }
            group_score[group] = best + second;
        }
        for (int slot = 0; slot < groups_used; ++slot) {
            int best_group = -1; float best = -3.402823466e+38F;
            for (int group = 0; group < groups; ++group)
                if (!group_kept[group] && (best_group < 0 || group_score[group] > best)) {
                    best_group = group; best = group_score[group];
                }
            group_kept[best_group] = 1;
        }
        for (int expert = 0; expert < experts; ++expert)
            allowed[expert] = group_kept[expert / span];
    }
    int* row_selected = selected + (long long)row * top_k;
    float* row_weights = weights + (long long)row * top_k;
    float total = 0.0f;
    for (int rank = 0; rank < top_k; ++rank) {
        int best_expert = -1; float best = -3.402823466e+38F;
        for (int expert = 0; expert < experts; ++expert) {
            bool used = false;
            for (int prior = 0; prior < rank; ++prior) used |= row_selected[prior] == expert;
            if (!used && allowed[expert] && (best_expert < 0 || ranking[expert] > best)) {
                best_expert = expert; best = ranking[expert];
            }
        }
        row_selected[rank] = best_expert;
        row_weights[rank] = probability[best_expert];
        total += row_weights[rank];
    }
    const float factor = normalize && total > 0.0f ? weight_scale / total : weight_scale;
    for (int rank = 0; rank < top_k; ++rank) {
        row_weights[rank] *= factor;
        if (expert_counts && expert_routes) {
            const int expert = row_selected[rank];
            const int slot = atomicAdd(expert_counts + expert, 1);
            if (slot < max_routes)
                expert_routes[(long long)expert * max_routes + slot] =
                    row * top_k + rank;
        }
    }
}

extern "C" __global__
void bailing_q6_expert_swiglu_rows(
    const unsigned char* gate_base, const unsigned char* up_base,
    const int* selected, const float* input, float* activated,
    const unsigned long long expert_stride, const int rows, const int top_k,
    const int input_size, const int output_size
) {
    const int output_row = blockIdx.x, route = blockIdx.y;
    if (output_row >= output_size || route >= rows * top_k) return;
    const int token = route / top_k, expert = selected[route];
    const unsigned char* gate = gate_base + (unsigned long long)expert * expert_stride;
    const unsigned char* up = up_base + (unsigned long long)expert * expert_stride;
    const float* vector = input + (long long)token * input_size;
    float g = 0.0f, u = 0.0f;
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        const float value = vector[i];
        const int absolute = output_row * input_size + i;
        g += q6k_value(gate, absolute) * value;
        u += q6k_value(up, absolute) * value;
    }
    // block_reduce_sum reuses one shared warp-sum arena.  Without a barrier,
    // a fast warp can begin the second reduction and overwrite that arena
    // while warp 0 is still finishing the first one.  The resulting rare
    // corrupt gate value was deterministic for layer 19 / expert 0 here.
    g = block_reduce_sum(g);
    __syncthreads();
    u = block_reduce_sum(u);
    if (threadIdx.x == 0)
        activated[(long long)route * output_size + output_row] =
            (g / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, g))))) * u;
}

extern "C" __global__
void bailing_q6_expert_accumulate_rows(
    const unsigned char* down_base, const int* selected,
    const float* activated, const float* weights, float* output,
    const unsigned long long expert_stride, const int rows, const int top_k,
    const int input_size, const int output_size
) {
    const int output_row = blockIdx.x, token = blockIdx.y;
    if (output_row >= output_size || token >= rows) return;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        float combined = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            const int route = token * top_k + rank;
            const unsigned char* down = down_base +
                (unsigned long long)selected[route] * expert_stride;
            combined += weights[route] * q6k_value(down, output_row * input_size + i)
                * activated[(long long)route * input_size + i];
        }
        partial += combined;
    }
    partial = block_reduce_sum(partial);
    if (threadIdx.x == 0) output[(long long)token * output_size + output_row] = partial;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__
void bailing_q6_expert_swiglu_grouped_rows(
    const unsigned char* gate_base, const unsigned char* up_base,
    const int* expert_counts, const int* expert_routes,
    const float* input, float* activated,
    const unsigned long long expert_stride, const int max_routes,
    const int top_k, const int input_size, const int output_size
) {
    const int output_row = blockIdx.x, expert = blockIdx.y;
    if (output_row >= output_size) return;
    const int count = expert_counts[expert];
    if (count <= 0) return;
    const unsigned char* gate = gate_base +
        (unsigned long long)expert * expert_stride;
    const unsigned char* up = up_base +
        (unsigned long long)expert * expert_stride;
    __shared__ float reduced_gate[16][8];
    __shared__ float reduced_up[16][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int assignment_base = 0; assignment_base < count;
         assignment_base += 16) {
        float g[16] = {0.0f};
        float u[16] = {0.0f};
        for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
            const int absolute = output_row * input_size + i;
            const float gate_weight = q6k_value(gate, absolute);
            const float up_weight = q6k_value(up, absolute);
            #pragma unroll
            for (int tile = 0; tile < 16; ++tile) {
                const int assignment = assignment_base + tile;
                if (assignment < count) {
                    const int route = expert_routes[
                        (long long)expert * max_routes + assignment];
                    const int token = route / top_k;
                    const float value = input[(long long)token * input_size + i];
                    g[tile] += gate_weight * value;
                    u[tile] += up_weight * value;
                }
            }
        }
        #pragma unroll
        for (int tile = 0; tile < 16; ++tile) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                g[tile] += __shfl_down_sync(0xffffffffu, g[tile], offset);
                u[tile] += __shfl_down_sync(0xffffffffu, u[tile], offset);
            }
            if (lane == 0) {
                reduced_gate[tile][warp] = g[tile];
                reduced_up[tile][warp] = u[tile];
            }
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            const int tile = threadIdx.x;
            const int assignment = assignment_base + tile;
            if (assignment < count) {
                float gate_sum = 0.0f, up_sum = 0.0f;
                #pragma unroll
                for (int source_warp = 0; source_warp < 8; ++source_warp) {
                    gate_sum += reduced_gate[tile][source_warp];
                    up_sum += reduced_up[tile][source_warp];
                }
                const int route = expert_routes[
                    (long long)expert * max_routes + assignment];
                activated[(long long)route * output_size + output_row] =
                    (gate_sum / (1.0f + expf(-fminf(80.0f,
                        fmaxf(-80.0f, gate_sum))))) * up_sum;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__
void bailing_q6_expert_accumulate_grouped_rows(
    const unsigned char* down_base,
    const int* expert_counts, const int* expert_routes,
    const float* activated, const float* weights, float* output,
    const unsigned long long expert_stride, const int max_routes,
    const int top_k, const int input_size, const int output_size
) {
    const int output_row = blockIdx.x, expert = blockIdx.y;
    if (output_row >= output_size) return;
    const int count = expert_counts[expert];
    if (count <= 0) return;
    const unsigned char* down = down_base +
        (unsigned long long)expert * expert_stride;
    __shared__ float reduced[16][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int assignment_base = 0; assignment_base < count;
         assignment_base += 16) {
        float partial[16] = {0.0f};
        for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
            const float weight = q6k_value(
                down, output_row * input_size + i);
            #pragma unroll
            for (int tile = 0; tile < 16; ++tile) {
                const int assignment = assignment_base + tile;
                if (assignment < count) {
                    const int route = expert_routes[
                        (long long)expert * max_routes + assignment];
                    partial[tile] += weight *
                        activated[(long long)route * input_size + i];
                }
            }
        }
        #pragma unroll
        for (int tile = 0; tile < 16; ++tile) {
            for (int offset = 16; offset > 0; offset >>= 1)
                partial[tile] += __shfl_down_sync(
                    0xffffffffu, partial[tile], offset);
            if (lane == 0) reduced[tile][warp] = partial[tile];
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            const int tile = threadIdx.x;
            const int assignment = assignment_base + tile;
            if (assignment < count) {
                float total = 0.0f;
                #pragma unroll
                for (int source_warp = 0; source_warp < 8; ++source_warp)
                    total += reduced[tile][source_warp];
                const int route = expert_routes[
                    (long long)expert * max_routes + assignment];
                const int token = route / top_k;
                atomicAdd(output + (long long)token * output_size + output_row,
                          weights[route] * total);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__
void bailing_quantize_q8_rows(
    const float* input, signed char* output, __half* scales,
    const int rows, const int elements
) {
    const int row = blockIdx.y, group = blockIdx.x;
    const int lane = threadIdx.x;
    const int index = group * 32 + lane;
    if (row >= rows || index >= elements) return;
    const long long absolute = (long long)row * elements + index;
    const float value = input[absolute];
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    if (lane == 0) scales[(long long)row * (elements / 32) + group] =
        __float2half(scale);
    const int quantized = max(-127, min(127, __float2int_rn(value / scale)));
    output[absolute] = (signed char)quantized;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
__device__ __forceinline__ int bailing_mmq_load_i32(const void* pointer) {
    int value;
    memcpy(&value, pointer, 4);
    return value;
}

__device__ __forceinline__ float bailing_mmq_q6_q8_dot(
    const int* weights, const int* activations, const signed char* scales,
    const float weight_scale, const float* activation_scales
) {
    float total = 0.0f;
    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        int first = 0, second = 0;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int index = group * 4 + part;
            first = __dp4a(weights[2 * index], activations[2 * index], first);
            first = __dp4a(weights[2 * index + 1], activations[2 * index + 1], first);
            second = __dp4a(weights[2 * index + 4], activations[2 * index + 4], second);
            second = __dp4a(weights[2 * index + 5], activations[2 * index + 5], second);
        }
        total += activation_scales[group] *
            ((float)scales[2 * group] * first +
             (float)scales[2 * group + 1] * second);
    }
    return weight_scale * total;
}

extern "C" __global__
void bailing_q6_q8_mmq_rows(
    const unsigned char* packed, const signed char* vectors,
    const __half* vector_scales, float* output,
    const int input_size, const int output_size, const int rows
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_base = blockIdx.x * 32;
    const int token_base = blockIdx.y * 4;
    if (output_base >= output_size || token_base >= rows) return;

    __shared__ int weight_values[32 * 65];
    __shared__ float weight_d[32];
    __shared__ int weight_scales[32 * 4 + 4];
    __shared__ int activation_values[4 * 32];
    __shared__ float activation_d[4 * 4];

    const int blocks_per_row = input_size / 256;
    float total = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
        #pragma unroll
        for (int row_group = 0; row_group < 32; row_group += 4) {
            int row = row_group + warp;
            const int available = output_size - output_base - 1;
            row = min(row, available);
            const unsigned char* source = packed +
                ((long long)(output_base + row) * blocks_per_row + block) * 210;
            const int low = bailing_mmq_load_i32(source + lane * 4);
            const int low_first = low & 0x0f0f0f0f;
            const int low_second = (low >> 4) & 0x0f0f0f0f;
            const int high_index = 8 * (lane / 16) + lane % 8;
            const int high = bailing_mmq_load_i32(source + 128 + high_index * 4);
            const int shift = 2 * ((lane % 16) / 8);
            const int high_first = ((high >> shift) << 4) & 0x30303030;
            const int high_second = (high >> shift) & 0x30303030;
            const int first_index = 2 * lane - (2 * lane) % 32 + lane % 16;
            weight_values[row * 65 + first_index] =
                __vsub4(low_first | high_first, 0x20202020);
            weight_values[row * 65 + first_index + 16] =
                __vsub4(low_second | high_second, 0x20202020);
        }

        const int scale_row = (warp * 32 + lane) & 31;
        const int scale_source_row = min(
            scale_row, output_size - output_base - 1);
        const unsigned char* scale_source = packed +
            ((long long)(output_base + scale_source_row) * blocks_per_row + block) * 210;
        weight_d[scale_row] = __half2float(
            *((const __half*)(scale_source + 208)));

        const int packed_scale_row = (warp * 8 + lane / 4) & 31;
        const int packed_scale_source_row = min(
            packed_scale_row, output_size - output_base - 1);
        const unsigned char* packed_scale_source = packed +
            ((long long)(output_base + packed_scale_source_row) * blocks_per_row + block) * 210;
        weight_scales[packed_scale_row * 4 + packed_scale_row / 8 + lane % 4] =
            bailing_mmq_load_i32(packed_scale_source + 192 + (lane % 4) * 4);

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int activation_block = block * 8 + half * 4 + lane / 8;
            const int token = min(token_base + warp, rows - 1);
            const signed char* values = vectors +
                (long long)token * input_size + activation_block * 32;
            activation_values[warp * 32 + lane] =
                bailing_mmq_load_i32(values + (lane % 8) * 4);

            const int scale_token = (lane / 4) & 3;
            const int scale_block = block * 8 + half * 4 + lane % 4;
            const int source_token = min(token_base + scale_token, rows - 1);
            activation_d[scale_token * 4 + lane % 4] = __half2float(
                vector_scales[(long long)source_token * (input_size / 32) +
                              scale_block]);
            __syncthreads();

            const int output_row = lane;
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                const signed char* scales = (const signed char*)(weight_scales +
                    output_row * 4 + output_row / 8 + half * 2 + part);
                const int weight_index =
                    output_row * 65 + half * 32 + part * 16;
                const int activation_index = warp * 32 + part * 16;
                total += bailing_mmq_q6_q8_dot(
                    weight_values + weight_index,
                    activation_values + activation_index,
                    scales, weight_d[output_row],
                    activation_d + warp * 4 + part * 2);
            }
            __syncthreads();
        }
    }

    const int output_row = output_base + lane;
    const int token = token_base + warp;
    if (output_row < output_size && token < rows)
        output[(long long)token * output_size + output_row] = total;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
__device__ __forceinline__ float bailing_q6_q8_mmq_tile(
    const unsigned char* packed, const signed char* vectors,
    const __half* vector_scales, const int* vector_rows,
    const int input_size, const int output_size, const int output_base,
    int* weight_values, float* weight_d, int* weight_scales,
    int* activation_values, float* activation_d
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int blocks_per_row = input_size / 256;
    float total = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
        #pragma unroll
        for (int row_group = 0; row_group < 32; row_group += 4) {
            int row = min(row_group + warp, output_size - output_base - 1);
            const unsigned char* source = packed +
                ((long long)(output_base + row) * blocks_per_row + block) * 210;
            const int low = bailing_mmq_load_i32(source + lane * 4);
            const int low_first = low & 0x0f0f0f0f;
            const int low_second = (low >> 4) & 0x0f0f0f0f;
            const int high_index = 8 * (lane / 16) + lane % 8;
            const int high = bailing_mmq_load_i32(source + 128 + high_index * 4);
            const int shift = 2 * ((lane % 16) / 8);
            const int high_first = ((high >> shift) << 4) & 0x30303030;
            const int high_second = (high >> shift) & 0x30303030;
            const int first_index = 2 * lane - (2 * lane) % 32 + lane % 16;
            weight_values[row * 65 + first_index] =
                __vsub4(low_first | high_first, 0x20202020);
            weight_values[row * 65 + first_index + 16] =
                __vsub4(low_second | high_second, 0x20202020);
        }

        const int scale_row = lane;
        const int source_row = min(scale_row, output_size - output_base - 1);
        const unsigned char* source = packed +
            ((long long)(output_base + source_row) * blocks_per_row + block) * 210;
        weight_d[scale_row] = __half2float(*((const __half*)(source + 208)));

        const int packed_scale_row = (warp * 8 + lane / 4) & 31;
        const int packed_source_row = min(
            packed_scale_row, output_size - output_base - 1);
        const unsigned char* packed_source = packed +
            ((long long)(output_base + packed_source_row) * blocks_per_row + block) * 210;
        weight_scales[packed_scale_row * 4 + packed_scale_row / 8 + lane % 4] =
            bailing_mmq_load_i32(packed_source + 192 + (lane % 4) * 4);

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int activation_block = block * 8 + half * 4 + lane / 8;
            const signed char* values = vectors +
                (long long)vector_rows[warp] * input_size + activation_block * 32;
            activation_values[warp * 32 + lane] =
                bailing_mmq_load_i32(values + (lane % 8) * 4);
            const int scale_row_index = (lane / 4) & 3;
            const int scale_block = block * 8 + half * 4 + lane % 4;
            activation_d[scale_row_index * 4 + lane % 4] = __half2float(
                vector_scales[(long long)vector_rows[scale_row_index] *
                              (input_size / 32) + scale_block]);
            __syncthreads();
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                const signed char* scales = (const signed char*)(weight_scales +
                    lane * 4 + lane / 8 + half * 2 + part);
                total += bailing_mmq_q6_q8_dot(
                    weight_values + lane * 65 + half * 32 + part * 16,
                    activation_values + warp * 32 + part * 16,
                    scales, weight_d[lane],
                    activation_d + warp * 4 + part * 2);
            }
            __syncthreads();
        }
    }
    return total;
}

extern "C" __global__
void bailing_q6_q8_expert_swiglu_mmq_rows(
    const unsigned char* gate_base, const unsigned char* up_base,
    const int* expert_counts, const int* expert_routes,
    const signed char* input, const __half* input_scales, float* activated,
    const unsigned long long expert_stride, const int max_routes,
    const int top_k, const int input_size, const int output_size
) {
    const int expert = blockIdx.y;
    const int output_base = blockIdx.x * 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (output_base >= output_size || expert_counts[expert] <= 0) return;
    __shared__ int routes[4], vector_rows[4], valid[4];
    __shared__ int weight_values[32 * 65];
    __shared__ float weight_d[32];
    __shared__ int weight_scales[32 * 4 + 4];
    __shared__ int activation_values[4 * 32];
    __shared__ float activation_d[4 * 4];
    const int count = expert_counts[expert];
    const unsigned char* gate = gate_base +
        (unsigned long long)expert * expert_stride;
    const unsigned char* up = up_base +
        (unsigned long long)expert * expert_stride;
    for (int base = 0; base < count; base += 4) {
        if (threadIdx.x < 4) {
            const int assignment = base + threadIdx.x;
            valid[threadIdx.x] = assignment < count;
            routes[threadIdx.x] = expert_routes[
                (long long)expert * max_routes + min(assignment, count - 1)];
            vector_rows[threadIdx.x] = routes[threadIdx.x] / top_k;
        }
        __syncthreads();
        const float gate_total = bailing_q6_q8_mmq_tile(
            gate, input, input_scales, vector_rows, input_size, output_size,
            output_base, weight_values, weight_d, weight_scales,
            activation_values, activation_d);
        const float up_total = bailing_q6_q8_mmq_tile(
            up, input, input_scales, vector_rows, input_size, output_size,
            output_base, weight_values, weight_d, weight_scales,
            activation_values, activation_d);
        const int output_row = output_base + lane;
        if (valid[warp] && output_row < output_size)
            activated[(long long)routes[warp] * output_size + output_row] =
                (gate_total / (1.0f + expf(-fminf(80.0f,
                    fmaxf(-80.0f, gate_total))))) * up_total;
        __syncthreads();
    }
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__
void bailing_q6_f32_expert_accumulate_mmq_rows(
    const unsigned char* down_base,
    const int* expert_counts, const int* expert_routes,
    const float* activated, const float* weights, float* output,
    const unsigned long long expert_stride, const int max_routes,
    const int top_k, const int input_size, const int output_size
) {
    const int expert = blockIdx.y;
    const int output_base = blockIdx.x * 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (output_base >= output_size || expert_counts[expert] <= 0) return;
    __shared__ int routes[4], valid[4];
    __shared__ int weight_values[32 * 65];
    __shared__ float weight_d[32];
    __shared__ int weight_scales[32 * 4 + 4];
    const int count = expert_counts[expert];
    const int blocks_per_row = input_size / 256;
    const unsigned char* down = down_base +
        (unsigned long long)expert * expert_stride;
    for (int route_base = 0; route_base < count; route_base += 4) {
        if (threadIdx.x < 4) {
            const int assignment = route_base + threadIdx.x;
            valid[threadIdx.x] = assignment < count;
            routes[threadIdx.x] = expert_routes[
                (long long)expert * max_routes + min(assignment, count - 1)];
        }
        __syncthreads();
        float total = 0.0f;
        for (int block = 0; block < blocks_per_row; ++block) {
            #pragma unroll
            for (int row_group = 0; row_group < 32; row_group += 4) {
                int row = min(row_group + warp, output_size - output_base - 1);
                const unsigned char* source = down +
                    ((long long)(output_base + row) * blocks_per_row + block) * 210;
                const int low = bailing_mmq_load_i32(source + lane * 4);
                const int low_first = low & 0x0f0f0f0f;
                const int low_second = (low >> 4) & 0x0f0f0f0f;
                const int high_index = 8 * (lane / 16) + lane % 8;
                const int high = bailing_mmq_load_i32(
                    source + 128 + high_index * 4);
                const int shift = 2 * ((lane % 16) / 8);
                const int high_first = ((high >> shift) << 4) & 0x30303030;
                const int high_second = (high >> shift) & 0x30303030;
                const int first_index =
                    2 * lane - (2 * lane) % 32 + lane % 16;
                weight_values[row * 65 + first_index] =
                    __vsub4(low_first | high_first, 0x20202020);
                weight_values[row * 65 + first_index + 16] =
                    __vsub4(low_second | high_second, 0x20202020);
            }
            const int source_row = min(lane, output_size - output_base - 1);
            const unsigned char* source = down +
                ((long long)(output_base + source_row) * blocks_per_row + block) * 210;
            weight_d[lane] = __half2float(*((const __half*)(source + 208)));
            const int packed_scale_row = (warp * 8 + lane / 4) & 31;
            const int packed_source_row = min(
                packed_scale_row, output_size - output_base - 1);
            const unsigned char* packed_source = down +
                ((long long)(output_base + packed_source_row) * blocks_per_row + block) * 210;
            weight_scales[packed_scale_row * 4 +
                          packed_scale_row / 8 + lane % 4] =
                bailing_mmq_load_i32(
                    packed_source + 192 + (lane % 4) * 4);
            __syncthreads();

            const float* values = activated +
                (long long)routes[warp] * input_size + block * 256;
            const signed char* scales = (const signed char*)(weight_scales +
                lane * 4 + lane / 8);
            float block_total = 0.0f;
            #pragma unroll
            for (int packed_index = 0; packed_index < 64; ++packed_index) {
                const int packed_weight =
                    weight_values[lane * 65 + packed_index];
                float partial = 0.0f;
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const signed char weight = (signed char)
                        ((unsigned int)packed_weight >> (byte * 8));
                    partial += (float)weight *
                        values[packed_index * 4 + byte];
                }
                block_total += (float)scales[packed_index / 4] * partial;
            }
            total += weight_d[lane] * block_total;
            __syncthreads();
        }
        const int output_row = output_base + lane;
        if (valid[warp] && output_row < output_size) {
            const int route = routes[warp];
            const int token = route / top_k;
            atomicAdd(output + (long long)token * output_size + output_row,
                      weights[route] * total);
        }
        __syncthreads();
    }
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__
void bailing_q6_grouped_swiglu_warp(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector, float* activated,
    const int input_size, const int output_size, const int experts
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate =
        (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up =
        (const unsigned char*)up_ptrs[expert];
    float gate_total = q6k_row_dot_warp(
        gate, vector, row, input_size, lane);
    float up_total = q6k_row_dot_warp(
        up, vector, row, input_size, lane);
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_total += __shfl_down_sync(0xffffffffu, gate_total, offset);
        up_total += __shfl_down_sync(0xffffffffu, up_total, offset);
    }
    if (lane == 0) {
        const float clamped = fminf(80.0f, fmaxf(-80.0f, gate_total));
        activated[(long long)expert * output_size + row] =
            (gate_total / (1.0f + expf(-clamped))) * up_total;
    }
}

extern "C" __global__
void bailing_q6_grouped_accumulate_warp(
    const unsigned long long* down_ptrs,
    const float* activated, float* output, const float* weights,
    const int input_size, const int output_size, const int experts
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= output_size) return;
    float total = 0.0f;
    for (int expert = 0; expert < experts; ++expert) {
        const unsigned char* down =
            (const unsigned char*)down_ptrs[expert];
        const float* vector = activated + (long long)expert * input_size;
        float partial = q6k_row_dot_warp(
            down, vector, row, input_size, lane);
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        if (lane == 0) total += weights[expert] * partial;
    }
    if (lane == 0) output[row] += total;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
__device__ __forceinline__ void bailing_q6_dp4a_group(
    const unsigned char* row_data, const int group,
    int packed_weights[2][8], float* scale, int* scale_first, int* scale_second
) {
    const int block = group >> 3, sub_block = group & 7;
    const int half = sub_block >> 2, lane_group = sub_block & 3;
    const unsigned char* base = row_data + block * 210;
    const unsigned char* lows =
        base + half * 64 + ((lane_group & 1) ? 32 : 0);
    const unsigned char* highs = base + 128 + half * 32;
    const int shift = (lane_group >> 1) * 4;
    const int bit_shift = lane_group * 2;
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int quad = part * 4 + step;
            unsigned int word, high_word;
            memcpy(&word, lows + quad * 4, 4);
            memcpy(&high_word, highs + quad * 4, 4);
            const unsigned int weights = ((word >> shift) & 0x0f0f0f0fu)
                | (((high_word >> bit_shift) & 0x03030303u) << 4);
            packed_weights[part][step] =
                __vsub4((int)weights, 0x20202020);
        }
    }
    *scale = __half2float(*((const __half*)(base + 208)));
    const signed char* scales = (const signed char*)(base + 192);
    const int scale_base = half * 8 + lane_group * 2;
    *scale_first = scales[scale_base];
    *scale_second = scales[scale_base + 1];
}

extern "C" __global__
void bailing_q6_q8_expert_swiglu_grouped_rows(
    const unsigned char* gate_base, const unsigned char* up_base,
    const int* expert_counts, const int* expert_routes,
    const signed char* input, const __half* input_scales, float* activated,
    const unsigned long long expert_stride, const int max_routes,
    const int top_k, const int input_size, const int output_size
) {
    const int output_row = blockIdx.x, expert = blockIdx.y;
    if (output_row >= output_size) return;
    const int count = expert_counts[expert];
    if (count <= 0) return;
    const int groups = input_size / 32;
    const unsigned char* gate = gate_base +
        (unsigned long long)expert * expert_stride;
    const unsigned char* up = up_base +
        (unsigned long long)expert * expert_stride;
    const unsigned char* gate_row = gate +
        (long long)output_row * (input_size / 256) * 210;
    const unsigned char* up_row = up +
        (long long)output_row * (input_size / 256) * 210;
    __shared__ float reduced_gate[16][8], reduced_up[16][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int assignment_base = 0; assignment_base < count;
         assignment_base += 16) {
        float gate_partial[16] = {0.0f};
        float up_partial[16] = {0.0f};
        for (int group = threadIdx.x; group < groups; group += blockDim.x) {
            int gate_weights[2][8], up_weights[2][8];
            float gate_scale, up_scale;
            int gate_first, gate_second, up_first, up_second;
            bailing_q6_dp4a_group(gate_row, group, gate_weights,
                &gate_scale, &gate_first, &gate_second);
            bailing_q6_dp4a_group(up_row, group, up_weights,
                &up_scale, &up_first, &up_second);
            #pragma unroll
            for (int tile = 0; tile < 16; ++tile) {
                const int assignment = assignment_base + tile;
                if (assignment < count) {
                    const int route = expert_routes[
                        (long long)expert * max_routes + assignment];
                    const int token = route / top_k;
                    const signed char* values = input +
                        (long long)token * input_size + group * 32;
                    int gate_dot[2] = {0, 0}, up_dot[2] = {0, 0};
                    #pragma unroll
                    for (int part = 0; part < 2; ++part) {
                        #pragma unroll
                        for (int step = 0; step < 4; ++step) {
                            int activation;
                            memcpy(&activation, values + (part * 4 + step) * 4, 4);
                            gate_dot[part] = __dp4a(
                                gate_weights[part][step], activation, gate_dot[part]);
                            up_dot[part] = __dp4a(
                                up_weights[part][step], activation, up_dot[part]);
                        }
                    }
                    const float activation_scale = __half2float(
                        input_scales[(long long)token * groups + group]);
                    gate_partial[tile] += activation_scale * gate_scale *
                        ((float)gate_first * gate_dot[0] +
                         (float)gate_second * gate_dot[1]);
                    up_partial[tile] += activation_scale * up_scale *
                        ((float)up_first * up_dot[0] +
                         (float)up_second * up_dot[1]);
                }
            }
        }
        #pragma unroll
        for (int tile = 0; tile < 16; ++tile) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                gate_partial[tile] += __shfl_down_sync(
                    0xffffffffu, gate_partial[tile], offset);
                up_partial[tile] += __shfl_down_sync(
                    0xffffffffu, up_partial[tile], offset);
            }
            if (lane == 0) {
                reduced_gate[tile][warp] = gate_partial[tile];
                reduced_up[tile][warp] = up_partial[tile];
            }
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            const int tile = threadIdx.x;
            const int assignment = assignment_base + tile;
            if (assignment < count) {
                float gate_sum = 0.0f, up_sum = 0.0f;
                for (int source_warp = 0; source_warp < 8; ++source_warp) {
                    gate_sum += reduced_gate[tile][source_warp];
                    up_sum += reduced_up[tile][source_warp];
                }
                const int route = expert_routes[
                    (long long)expert * max_routes + assignment];
                activated[(long long)route * output_size + output_row] =
                    (gate_sum / (1.0f + expf(-fminf(80.0f,
                        fmaxf(-80.0f, gate_sum))))) * up_sum;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__
void bailing_q6_matmul_rows_16(
    const unsigned char* packed, const float* vectors, float* output,
    const int input_size, const int output_size, const int rows
) {
    const int output_row = blockIdx.x;
    const int token_base = blockIdx.y * 16;
    if (output_row >= output_size || token_base >= rows) return;
    float partial[16] = {0.0f};
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float weight = q6k_value(
            packed, output_row * input_size + input);
        #pragma unroll
        for (int tile = 0; tile < 16; ++tile) {
            const int token = token_base + tile;
            if (token < rows)
                partial[tile] += weight *
                    vectors[(long long)token * input_size + input];
        }
    }
    #pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
        partial[tile] = block_reduce_sum(partial[tile]);
        __syncthreads();
        if (threadIdx.x == 0 && token_base + tile < rows)
            output[(long long)(token_base + tile) * output_size + output_row] =
                partial[tile];
    }
}

extern "C" __global__
void bailing_q6_q8_matmul_rows(
    const unsigned char* packed, const signed char* vectors,
    const __half* vector_scales, float* output,
    const int input_size, const int output_size, const int rows
) {
    const int output_row = blockIdx.x;
    const int token_base = blockIdx.y * 4;
    if (output_row >= output_size || token_base >= rows) return;
    const int blocks_per_row = input_size >> 8;
    const int groups_per_row = blocks_per_row << 3;
    const unsigned char* row_data =
        packed + (long long)output_row * blocks_per_row * 210;
    float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int group = threadIdx.x; group < groups_per_row; group += blockDim.x) {
        const int block = group >> 3, sub_block = group & 7;
        const int half = sub_block >> 2, lane_group = sub_block & 3;
        const unsigned char* base = row_data + block * 210;
        const unsigned char* lows =
            base + half * 64 + ((lane_group & 1) ? 32 : 0);
        const unsigned char* highs = base + 128 + half * 32;
        const int shift = (lane_group >> 1) * 4;
        const int bit_shift = lane_group * 2;
        int dot[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (int part = 0; part < 2; ++part) {
            for (int step = 0; step < 4; ++step) {
                const int quad = part * 4 + step;
                unsigned int word, high_word;
                memcpy(&word, lows + quad * 4, 4);
                memcpy(&high_word, highs + quad * 4, 4);
                const unsigned int weights = ((word >> shift) & 0x0f0f0f0fu)
                    | (((high_word >> bit_shift) & 0x03030303u) << 4);
                const int biased = __vsub4((int)weights, 0x20202020);
                for (int tile = 0; tile < 4; ++tile) {
                    const int token = token_base + tile;
                    if (token < rows) {
                        const signed char* activation = vectors +
                            (long long)token * input_size + group * 32;
                        int values;
                        memcpy(&values, activation + quad * 4, 4);
                        dot[tile][part] = __dp4a(
                            biased, values, dot[tile][part]);
                    }
                }
            }
        }
        const float d = __half2float(*((const __half*)(base + 208)));
        const signed char* weight_scales = (const signed char*)(base + 192);
        const int scale_base = half * 8 + lane_group * 2;
        for (int tile = 0; tile < 4; ++tile) {
            const int token = token_base + tile;
            if (token < rows) {
                const float activation_scale = __half2float(
                    vector_scales[(long long)token * (input_size / 32) + group]);
                partial[tile] += activation_scale * d *
                    ((float)weight_scales[scale_base] * (float)dot[tile][0] +
                     (float)weight_scales[scale_base + 1] * (float)dot[tile][1]);
            }
        }
    }
    for (int tile = 0; tile < 4; ++tile) {
        partial[tile] = block_reduce_sum(partial[tile]);
        __syncthreads();
        if (threadIdx.x == 0 && token_base + tile < rows)
            output[(long long)(token_base + tile) * output_size + output_row] =
                partial[tile];
    }
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__
void bailing_mla_attention(
    const float* query_nope, const float* query_rope,
    const float* kv_b, const float* latents, const float* rope_keys,
    float* scores, float* output,
    const int positions, const int heads, const int qk_nope,
    const int qk_rope, const int v_head_dim, const int kv_lora
) {
    // Absorbed MLA attention for one token: one block per head.
    //
    // "Absorbed" means kv_b is folded into the query and the output rather than
    // used to decompress the cache, so attention runs against the raw 512-wide
    // latent. That is the whole reason the KV cache is 8.9x smaller here; see
    // mla_attention_absorbed in colibri_v2_bailing.hpp for the identity.
    //
    // `scores` is caller-supplied scratch of at least `heads * positions`,
    // because positions is unbounded and the softmax needs somewhere to live.
    // Requires kv_lora <= 512. Launch: grid heads, block 128.
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= heads || kv_lora > 512) return;

    __shared__ float projected[512];
    __shared__ float accumulated[512];
    __shared__ float reduce[128];
    __shared__ float shared_max, shared_sum;

    const float* head_weights = kv_b + (long long)head * (qk_nope + v_head_dim) * kv_lora;
    const float* qn = query_nope + (long long)head * qk_nope;
    const float* qr = query_rope + (long long)head * qk_rope;
    float* head_scores = scores + (long long)head * positions;
    const float scale = rsqrtf((float)(qk_nope + qk_rope));

    // 1. Pull the query through kv_b's key half once, instead of pushing every
    //    cached latent through it. O(1) per token rather than O(context).
    for (int i = lane; i < kv_lora; i += 128) {
        float total = 0.0f;
        for (int row = 0; row < qk_nope; ++row)
            total += qn[row] * head_weights[(long long)row * kv_lora + i];
        projected[i] = total;
    }
    __syncthreads();

    // 2. Scores against the latent, plus the shared rope half.
    for (int position = lane; position < positions; position += 128) {
        const float* latent = latents + (long long)position * kv_lora;
        float total = 0.0f;
        for (int i = 0; i < kv_lora; ++i) total += projected[i] * latent[i];
        const float* rope = rope_keys + (long long)position * qk_rope;
        for (int i = 0; i < qk_rope; ++i) total += qr[i] * rope[i];
        head_scores[position] = total * scale;
    }
    __syncthreads();

    // 3. Softmax, max-shifted. Two block reductions over a strided range.
    float local_max = -3.0e38f;
    for (int position = lane; position < positions; position += 128)
        local_max = fmaxf(local_max, head_scores[position]);
    reduce[lane] = local_max;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) shared_max = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int position = lane; position < positions; position += 128) {
        const float value = __expf(head_scores[position] - shared_max);
        head_scores[position] = value;
        local_sum += value;
    }
    reduce[lane] = local_sum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) reduce[lane] += reduce[lane + stride];
        __syncthreads();
    }
    if (lane == 0) shared_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    __syncthreads();
    for (int position = lane; position < positions; position += 128)
        head_scores[position] *= shared_sum;
    __syncthreads();

    // 4. Mix in latent space, then decompress once through kv_b's value half.
    for (int i = lane; i < kv_lora; i += 128) {
        float total = 0.0f;
        for (int position = 0; position < positions; ++position)
            total += head_scores[position] * latents[(long long)position * kv_lora + i];
        accumulated[i] = total;
    }
    __syncthreads();

    for (int row = lane; row < v_head_dim; row += 128) {
        const float* source = head_weights + (long long)(qk_nope + row) * kv_lora;
        float total = 0.0f;
        for (int i = 0; i < kv_lora; ++i) total += source[i] * accumulated[i];
        output[(long long)head * v_head_dim + row] = total;
    }
}

// Parallel absorbed-MLA decode.  The original kernel above is intentionally
// retained as the compact correctness fallback.  Splitting the operation lets
// the score phase scale across context tiles instead of assigning an entire
// history scan to one block per head.
extern "C" __global__
void bailing_mla_project(
    const float* query_nope, const float* kv_b, float* projected,
    const int heads, const int qk_nope, const int v_head_dim,
    const int kv_lora
) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= heads) return;
    const float* weights = kv_b +
        (long long)head * (qk_nope + v_head_dim) * kv_lora;
    const float* query = query_nope + (long long)head * qk_nope;
    for (int column = lane; column < kv_lora; column += blockDim.x) {
        float total = 0.0f;
        for (int row = 0; row < qk_nope; ++row)
            total += query[row] * weights[(long long)row * kv_lora + column];
        projected[(long long)head * kv_lora + column] = total;
    }
}

extern "C" __global__
void bailing_mla_scores(
    const float* projected, const float* query_rope,
    const float* latents, const float* rope_keys, float* scores,
    const int positions, const int heads, const int qk_nope,
    const int qk_rope, const int kv_lora
) {
    const int head = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int position = blockIdx.y * 8 + warp;
    if (head >= heads || position >= positions) return;
    const float* p = projected + (long long)head * kv_lora;
    const float* latent = latents + (long long)position * kv_lora;
    const float* query = query_rope + (long long)head * qk_rope;
    const float* rope = rope_keys + (long long)position * qk_rope;
    float total = 0.0f;
    for (int i = lane; i < kv_lora; i += 32) total += p[i] * latent[i];
    for (int i = lane; i < qk_rope; i += 32) total += query[i] * rope[i];
    for (int offset = 16; offset > 0; offset >>= 1)
        total += __shfl_down_sync(0xffffffff, total, offset);
    if (lane == 0)
        scores[(long long)head * positions + position] =
            total * rsqrtf((float)(qk_nope + qk_rope));
}

// Two heads attend to the same compressed MLA cache. A warp computes both
// head/position dots together so every latent and rope-key value is loaded
// once instead of twice. Unlike shared-memory head tiling, this retains one
// independent position per warp and has thousands of blocks at long context.
extern "C" __global__
void bailing_mla_scores_pair(
    const float* projected, const float* query_rope,
    const float* latents, const float* rope_keys, float* scores,
    const int positions, const int heads, const int qk_nope,
    const int qk_rope, const int kv_lora
) {
    const int first_head = blockIdx.x * 2;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int position = blockIdx.y * 8 + warp;
    if (first_head >= heads || position >= positions) return;
    const bool has_second = first_head + 1 < heads;
    const float* first_p = projected + (long long)first_head * kv_lora;
    const float* second_p = first_p + kv_lora;
    const float* first_query = query_rope + (long long)first_head * qk_rope;
    const float* second_query = first_query + qk_rope;
    const float* latent = latents + (long long)position * kv_lora;
    const float* rope = rope_keys + (long long)position * qk_rope;
    float first_total = 0.0f, second_total = 0.0f;
    for (int i = lane; i < kv_lora; i += 32) {
        const float cache = latent[i];
        first_total += first_p[i] * cache;
        if (has_second) second_total += second_p[i] * cache;
    }
    for (int i = lane; i < qk_rope; i += 32) {
        const float cache = rope[i];
        first_total += first_query[i] * cache;
        if (has_second) second_total += second_query[i] * cache;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        first_total += __shfl_down_sync(0xffffffff, first_total, offset);
        second_total += __shfl_down_sync(0xffffffff, second_total, offset);
    }
    if (lane == 0) {
        const float scale = rsqrtf((float)(qk_nope + qk_rope));
        scores[(long long)first_head * positions + position] =
            first_total * scale;
        if (has_second)
            scores[(long long)(first_head + 1) * positions + position] =
                second_total * scale;
    }
}

extern "C" __global__
void bailing_mla_softmax(float* scores, const int positions, const int heads) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= heads) return;
    float* values = scores + (long long)head * positions;
    __shared__ float reduce[128];
    float local = -3.0e38f;
    for (int i = lane; i < positions; i += 128) local = fmaxf(local, values[i]);
    reduce[lane] = local;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
        __syncthreads();
    }
    const float maximum = reduce[0];
    local = 0.0f;
    for (int i = lane; i < positions; i += 128) {
        const float value = __expf(values[i] - maximum);
        values[i] = value;
        local += value;
    }
    reduce[lane] = local;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] += reduce[lane + stride];
        __syncthreads();
    }
    const float inverse = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (int i = lane; i < positions; i += 128) values[i] *= inverse;
}

extern "C" __global__
void bailing_mla_accumulate(
    const float* scores, const float* latents, float* accumulated,
    const int positions, const int heads, const int kv_lora
) {
    const int head = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || column >= kv_lora) return;
    const float* weights = scores + (long long)head * positions;
    float total = 0.0f;
    for (int position = 0; position < positions; ++position)
        total += weights[position] * latents[(long long)position * kv_lora + column];
    accumulated[(long long)head * kv_lora + column] = total;
}

// Split the long value reduction across independent blocks. The baseline has
// only heads * ceil(kv_lora / 128) blocks, and every thread carries one serial
// dependency chain across the entire context. At long context that exposes
// latency instead of bandwidth. Partial sums provide enough blocks to occupy
// the GPU; the tiny follow-up reduction combines them.
extern "C" __global__
void bailing_mla_accumulate_split(
    const float* scores, const float* latents, float* partials,
    const int positions, const int heads, const int kv_lora, const int splits
) {
    const int head = blockIdx.x / splits;
    const int split = blockIdx.x - head * splits;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || column >= kv_lora) return;
    const int begin = (positions * split) / splits;
    const int end = (positions * (split + 1)) / splits;
    const float* weights = scores + (long long)head * positions;
    float total = 0.0f;
    for (int position = begin; position < end; ++position)
        total += weights[position] *
            latents[(long long)position * kv_lora + column];
    partials[((long long)split * heads + head) * kv_lora + column] = total;
}

extern "C" __global__
void bailing_mla_accumulate_reduce(
    const float* partials, float* accumulated,
    const int heads, const int kv_lora, const int splits
) {
    const int head = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || column >= kv_lora) return;
    float total = partials[(long long)head * kv_lora + column];
    for (int split = 1; split < splits; ++split)
        total += partials[
            ((long long)split * heads + head) * kv_lora + column];
    accumulated[(long long)head * kv_lora + column] = total;
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
// Position-parallel decode scores. The latent cache is shared across heads --
// that is what MLA compression means -- yet bailing_mla_scores re-reads it
// once per head (a heads-wide grid.x). Here a warp owns one position, loads
// each latent and rope-key value once, and feeds it to every head's dot
// product, so cache traffic no longer scales with head count. The per-head
// accumulation order is identical to bailing_mla_scores (lane-strided sum,
// then the same warp shuffle reduction), so the scores are bit-identical.
extern "C" __global__
void bailing_mla_scores_fused(
    const float* projected, const float* query_rope,
    const float* latents, const float* rope_keys, float* scores,
    const int positions, const int heads, const int qk_nope,
    const int qk_rope, const int kv_lora
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int position = blockIdx.x * (blockDim.x >> 5) + warp;
    if (position >= positions) return;
    const float* latent = latents + (long long)position * kv_lora;
    const float* rope = rope_keys + (long long)position * qk_rope;
    const float scale = rsqrtf((float)(qk_nope + qk_rope));
    for (int base = 0; base < heads; base += 16) {
        float totals[16];
        #pragma unroll
        for (int h = 0; h < 16; ++h) totals[h] = 0.0f;
        for (int i = lane; i < kv_lora; i += 32) {
            const float cache = latent[i];
            #pragma unroll
            for (int h = 0; h < 16; ++h)
                if (base + h < heads)
                    totals[h] +=
                        projected[(long long)(base + h) * kv_lora + i] * cache;
        }
        for (int i = lane; i < qk_rope; i += 32) {
            const float cache = rope[i];
            #pragma unroll
            for (int h = 0; h < 16; ++h)
                if (base + h < heads)
                    totals[h] +=
                        query_rope[(long long)(base + h) * qk_rope + i] * cache;
        }
        #pragma unroll
        for (int h = 0; h < 16; ++h) {
            if (base + h >= heads) continue;
            float total = totals[h];
            for (int offset = 16; offset > 0; offset >>= 1)
                total += __shfl_down_sync(0xffffffff, total, offset);
            if (lane == 0)
                scores[(long long)(base + h) * positions + position] =
                    total * scale;
        }
    }
}

// bailing_mla_project with the column loop unrolled onto a 2D grid. The
// original runs 16 blocks for an 8.4 MB f32 kv_b walk and is latency-bound;
// this one is the same serial row sum per column (bit-identical), just with
// four times the threads in flight.
extern "C" __global__
void bailing_mla_project_fused(
    const float* query_nope, const float* kv_b, float* projected,
    const int heads, const int qk_nope, const int v_head_dim,
    const int kv_lora
) {
    const int head = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || column >= kv_lora) return;
    const float* weights = kv_b +
        (long long)head * (qk_nope + v_head_dim) * kv_lora;
    const float* query = query_nope + (long long)head * qk_nope;
    float total = 0.0f;
    for (int row = 0; row < qk_nope; ++row)
        total += query[row] * weights[(long long)row * kv_lora + column];
    projected[(long long)head * kv_lora + column] = total;
}

// bailing_mla_output re-shaped from thread-per-row (each thread streaming its
// own 2 KB weight row, latency-bound at 16 blocks) to warp-per-row with
// coalesced lane-strided loads and a shuffle reduction. The reduction tree
// rounds differently from the serial sum; the logits gate allows 1e-3.
extern "C" __global__
void bailing_mla_output_fused(
    const float* accumulated, const float* kv_b, float* output,
    const int heads, const int qk_nope, const int v_head_dim,
    const int kv_lora
) {
    const int head = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.y * (blockDim.x >> 5) + warp;
    if (head >= heads || row >= v_head_dim) return;
    const float* values = accumulated + (long long)head * kv_lora;
    const float* source = kv_b +
        ((long long)head * (qk_nope + v_head_dim) + qk_nope + row) * kv_lora;
    float total = 0.0f;
    for (int i = lane; i < kv_lora; i += 32) total += source[i] * values[i];
    for (int offset = 16; offset > 0; offset >>= 1)
        total += __shfl_down_sync(0xffffffff, total, offset);
    if (lane == 0) output[(long long)head * v_head_dim + row] = total;
}

// The same shared-read restructuring for the value phase. Each block owns a
// contiguous position range and a column chunk; every latent value in that
// tile is loaded once and scattered into all heads' accumulators, where
// bailing_mla_accumulate re-walked the whole cache per head. Partials use the
// bailing_mla_accumulate_split layout so its reduce kernel finishes the job;
// within a split the position order matches the serial kernel, so splits == 1
// is bit-identical and splits > 1 differs only by the split-boundary rounding
// the _split kernel already introduced.
extern "C" __global__
void bailing_mla_accumulate_fused(
    const float* scores, const float* latents, float* partials,
    const int positions, const int heads, const int kv_lora, const int splits
) {
    const int split = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (split >= splits || column >= kv_lora) return;
    const int begin = (int)(((long long)positions * split) / splits);
    const int end = (int)(((long long)positions * (split + 1)) / splits);
    for (int base = 0; base < heads; base += 16) {
        float totals[16];
        #pragma unroll
        for (int h = 0; h < 16; ++h) totals[h] = 0.0f;
        for (int position = begin; position < end; ++position) {
            const float cache = latents[(long long)position * kv_lora + column];
            #pragma unroll
            for (int h = 0; h < 16; ++h)
                if (base + h < heads)
                    totals[h] +=
                        scores[(long long)(base + h) * positions + position] *
                        cache;
        }
        #pragma unroll
        for (int h = 0; h < 16; ++h)
            if (base + h < heads)
                partials[((long long)split * heads + base + h) * kv_lora +
                         column] = totals[h];
    }
}

extern "C" __global__
void bailing_mla_output(
    const float* accumulated, const float* kv_b, float* output,
    const int heads, const int qk_nope, const int v_head_dim,
    const int kv_lora
) {
    const int head = blockIdx.x;
    const int row = threadIdx.x;
    if (head >= heads) return;
    const float* values = accumulated + (long long)head * kv_lora;
    const float* weights = kv_b +
        ((long long)head * (qk_nope + v_head_dim) + qk_nope) * kv_lora;
    for (int output_row = row; output_row < v_head_dim; output_row += blockDim.x) {
        float total = 0.0f;
        const float* source = weights + (long long)output_row * kv_lora;
        for (int i = 0; i < kv_lora; ++i) total += source[i] * values[i];
        output[(long long)head * v_head_dim + output_row] = total;
    }
}

)COLIBRI_CUDA"
R"COLIBRI_CUDA(
extern "C" __global__
void bailing_kda_recurrent_chunk(
    const float* queries, const float* keys, const float* values,
    const float* gate_raw, const float* beta_logits,
    const float* a_log, const float* dt_bias,
    float* state, float* output,
    const int rows, const int heads, const int head_dim, const float epsilon
) {
    // Kimi Delta Attention over a chunk, one block per head, state held in
    // registers across the token loop. Structurally this is
    // qwen_delta_recurrent_chunk with one change, which is the whole difference
    // between DeltaNet and KDA:
    //
    //     DeltaNet   local_state[key] *= decay_scale    (one scalar per head)
    //     KDA        local_state[key] *= shared_decay[key]  (one per channel)
    //
    // The decay is therefore a vector, computed by thread `lane` for key=lane
    // and shared, because a thread owns one VALUE column and touches every KEY
    // row. The gate formula itself is identical to DeltaNet's
    // expf(coefficient * softplus(x + bias)) with a negative coefficient; only
    // its argument and bias are per channel here rather than per head.
    //
    // Requires head_dim == 128. Launch: grid heads, block 128.
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= heads || head_dim != 128) return;
    float local_state[128];
    #pragma unroll
    for (int key = 0; key < 128; ++key)
        local_state[key] = state[(head * 128 + key) * 128 + lane];
    __shared__ float shared_key[128], shared_query[128], shared_decay[128];
    __shared__ float query_sums[4], key_sums[4];
    __shared__ float beta, query_inverse_norm, key_inverse_norm;
    const float coefficient = -expf(a_log[head]);
    for (int token = 0; token < rows; ++token) {
        const long long base = ((long long)token * heads + head) * 128;
        const float query_raw = queries[base + lane];
        const float key_raw = keys[base + lane];
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
            const float query_square =
                query_sums[0] + query_sums[1] + query_sums[2] + query_sums[3];
            const float key_square =
                key_sums[0] + key_sums[1] + key_sums[2] + key_sums[3];
            // The query carries the 1/sqrt(head_dim) attention scale.
            query_inverse_norm = rsqrtf(query_square + epsilon) * rsqrtf(128.0f);
            key_inverse_norm = rsqrtf(key_square + epsilon);
            beta = 1.0f / (1.0f + expf(-beta_logits[token * heads + head]));
        }
        __syncthreads();
        shared_query[lane] = query_raw * query_inverse_norm;
        shared_key[lane] = key_raw * key_inverse_norm;
        {
            const float raw = gate_raw[base + lane] + dt_bias[head * 128 + lane];
            // Guarded softplus: above 20 the identity is exact in float and
            // expf would overflow.
            const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
            shared_decay[lane] = expf(coefficient * softplus);
        }
        __syncthreads();
        const float value = values[base + lane];
        float memory = 0.0f;
        #pragma unroll
        for (int key = 0; key < 128; ++key) {
            local_state[key] *= shared_decay[key];
            memory += local_state[key] * shared_key[key];
        }
        const float delta = (value - memory) * beta;
        float core = 0.0f;
        #pragma unroll
        for (int key = 0; key < 128; ++key) {
            local_state[key] += shared_key[key] * delta;
            core += local_state[key] * shared_query[key];
        }
        output[base + lane] = core;
        __syncthreads();
    }
    #pragma unroll
    for (int key = 0; key < 128; ++key)
        state[(head * 128 + key) * 128 + lane] = local_state[key];
}
)COLIBRI_CUDA"
R"COLIBRI_CUDA(
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


// Chunked WY-representation gated DeltaNet, used for prefill row batches.
// native/tools/deltanet_chunked.cu holds the same source for use with the
// CuPy harness in native/tools/kernel_harness.py; keep the two in step.
//
// The sequential kernels (qwen_delta_recurrent{,_split,_rows,_chunk}) walk one
// token at a time with the state matrix live in registers, so a prefill chunk
// of L tokens costs L serial steps across only `value_heads` blocks.  The
// chunked form rewrites the same recurrence
//
//     S_i = d_i (I - b_i k_i k_i^T) S_{i-1} + b_i k_i v_i^T
//
// so that everything inside a chunk of 64 tokens is matrix work and only the
// chunk-to-chunk state hand-off stays serial, cutting the critical path to
// L/64 steps and widening the grid to chunks x heads.
//
// Derivation.  Let g_i = log d_i and G_i the inclusive cumulative sum of g
// within the chunk (G anchored at 0 before the chunk's first token).  Writing
// S_i = exp(G_i) * Shat_i turns the recurrence into
//
//     Shat_i = Shat_{i-1} + k_i w_i^T,   w_i = exp(-G_i) beta_i v_i
//                                             - Shat_{i-1}^T (beta_i k_i)
//
// and substituting w_i = exp(-G_i) omega_i keeps every factor bounded by 1:
//
//     omega_i + sum_{j<i} A_ij omega_j = beta_i v_i - exp(G_i) (beta_i k_i)^T S_start
//     A_ij = exp(G_i - G_j) * beta_i * (k_i . k_j)      for j < i
//
// With T = (I + A)^-1 (unit lower triangular), Wm = T * diag(exp(G) beta) K and
// U = T * diag(beta) V, both computable without the state:
//
//     Omega   = U - Wm * S_start
//     S_end   = exp(G_last) * S_start + Ktilde^T * Omega,  Ktilde_j = exp(G_last - G_j) k_j
//     core_i  = S_start^T (exp(G_i) q_i) + sum_{j<=i} P_ij * omega_j
//     P_ij    = exp(G_i - G_j) * (q_i . k_j)            for j <= i
//
// q and k are the L2-normalized projections; q additionally carries the
// 1/sqrt(head_dim) scale, exactly as the sequential kernels apply it.
//
// Fixed to head_dim == 128 and chunk == 64; the host keeps the sequential
// kernels for decode and for any other geometry.  Shared memory per block is
// held at or below 32 KB because the driver never opts in past the 48 KB
// default.

#define DELTA_CHUNK 64
#define DELTA_DIM 128
#define DELTA_SLAB 32

// Pass 1: per-token scalars (log-decay cumsum, beta, normalization) and the two
// 64x64 score matrices.  Grid (chunks, value_heads), block 256.
extern "C" __global__
void qwen_delta_wy_scores(
    const float* convolved, const float* beta_logits, const float* decay_logits,
    const float* decay_coefficients, const float* dt_bias,
    float* attn, float* pmat, float* g_cumsum, float* beta_out,
    float* qinv_out, float* kinv_out,
    const int rows, const int key_heads, const int value_heads
) {
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int base = chunk * DELTA_CHUNK;
    const int valid = rows - base < DELTA_CHUNK ? rows - base : DELTA_CHUNK;
    if (valid <= 0) return;
    const int key_head = head % key_heads;
    const int total_key = key_heads * DELTA_DIM;
    const int key_off = key_head * DELTA_DIM;
    const long long stride = (long long)(total_key * 2 + value_heads * DELTA_DIM);
    const int tid = threadIdx.x;

    // Slabs are stored dimension-major so a thread's four tile rows are
    // contiguous in shared memory; the +1 pad keeps the strided reads off a
    // single bank.
    __shared__ float qs[DELTA_SLAB][DELTA_CHUNK + 1];
    __shared__ float ks[DELTA_SLAB][DELTA_CHUNK + 1];
    __shared__ float gcum[DELTA_CHUNK];
    __shared__ float betas[DELTA_CHUNK];
    __shared__ float qinv[DELTA_CHUNK];
    __shared__ float kinv[DELTA_CHUNK];

    // One warp per token: L2 norms, beta, and the per-token log decay.
    for (int t = tid >> 5; t < DELTA_CHUNK; t += 8) {
        const int lane = tid & 31;
        if (t >= valid) {
            if (lane == 0) { gcum[t] = 0.0f; betas[t] = 0.0f; qinv[t] = 0.0f; kinv[t] = 0.0f; }
            continue;
        }
        const float* row = convolved + (long long)(base + t) * stride;
        float key_square = 0.0f, query_square = 0.0f;
        for (int d = lane; d < DELTA_DIM; d += 32) {
            const float k = row[total_key + key_off + d];
            const float q = row[key_off + d];
            key_square += k * k;
            query_square += q * q;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            key_square += __shfl_down_sync(0xffffffff, key_square, offset);
            query_square += __shfl_down_sync(0xffffffff, query_square, offset);
        }
        if (lane == 0) {
            qinv[t] = rsqrtf(query_square + 1.0e-6f) * rsqrtf((float)DELTA_DIM);
)COLIBRI_CUDA"
R"COLIBRI_CUDA(            kinv[t] = rsqrtf(key_square + 1.0e-6f);
            const long long scalar = (long long)(base + t) * value_heads + head;
            betas[t] = 1.0f / (1.0f + expf(-beta_logits[scalar]));
            const float softplus_input = decay_logits[scalar] + dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input : log1pf(expf(softplus_input));
            gcum[t] = decay_coefficients[head] * softplus;
        }
    }
    __syncthreads();
    // Inclusive prefix sum of the log decays. 64 serial adds on one thread is
    // cheaper than the barriers a parallel scan of this length would cost.
    if (tid == 0) {
        float running = 0.0f;
        for (int t = 0; t < valid; ++t) { running += gcum[t]; gcum[t] = running; }
    }
    __syncthreads();
    if (tid < valid) {
        const long long scalar = (long long)(base + tid) * value_heads + head;
        g_cumsum[scalar] = gcum[tid];
        beta_out[scalar] = betas[tid];
        // Published so the solve and the state pass do not each repeat a full
        // pass over the projections just to recover these two scalars.
        qinv_out[scalar] = qinv[tid];
        kinv_out[scalar] = kinv[tid];
    }

    // P = Qn Kn^T and A = Kn Kn^T, both 64x64 over a 128-deep reduction, as one
    // blocked GEMM: 256 threads in a 16x16 arrangement, each holding a 4x4 tile
    // of both outputs. Twelve shared loads feed 32 FMAs per step, where the
    // dot-product-per-thread form managed one load per FMA.
    const int tx = tid & 15, ty = tid >> 4;
    const int i0 = ty * 4, j0 = tx * 4;
    float pacc[4][4] = {}, aacc[4][4] = {};
    for (int slab = 0; slab < DELTA_DIM; slab += DELTA_SLAB) {
        for (int index = tid; index < DELTA_SLAB * DELTA_CHUNK; index += blockDim.x) {
            const int t = index / DELTA_SLAB, d = index % DELTA_SLAB;
            float q = 0.0f, k = 0.0f;
            if (t < valid) {
                const float* row = convolved + (long long)(base + t) * stride;
                // Fold the normalizers in on the way to shared memory.
                q = row[key_off + slab + d] * qinv[t];
                k = row[total_key + key_off + slab + d] * kinv[t];
            }
            qs[d][t] = q;
            ks[d][t] = k;
        }
        __syncthreads();
        for (int d = 0; d < DELTA_SLAB; ++d) {
            float column[4], query_row[4], key_row[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) column[c] = ks[d][j0 + c];
            #pragma unroll
            for (int r = 0; r < 4; ++r) { query_row[r] = qs[d][i0 + r]; key_row[r] = ks[d][i0 + r]; }
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    pacc[r][c] += query_row[r] * column[c];
                    aacc[r][c] += key_row[r] * column[c];
                }
        }
        __syncthreads();
    }

    const long long mat = ((long long)chunk * value_heads + head)
        * (DELTA_CHUNK * DELTA_CHUNK);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = i0 + r;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int j = j0 + c;
            // Tail rows and columns of a partial chunk are zeroed here so the
            // solve and the state pass can run the full 64 without branching.
            const float decay = expf(gcum[i] - gcum[j]);
            const bool live = i < valid && j < valid;
            pmat[mat + i * DELTA_CHUNK + j] =
                (live && j <= i) ? pacc[r][c] * decay : 0.0f;
            attn[mat + i * DELTA_CHUNK + j] =
                (live && j < i) ? aacc[r][c] * betas[i] * decay : 0.0f;
        }
    }
}

// Pass 2: invert (I + A) and apply it to the two right-hand sides, producing
// Wm (64 x 128, against the state) and U (64 x 128, the state-free part of
// Omega).  Grid (chunks, value_heads), block 256.
extern "C" __global__
void qwen_delta_wy_solve(
    const float* convolved, const float* attn, const float* g_cumsum,
    const float* beta_in, const float* kinv_in, float* w_rows, float* u_rows,
    const int rows, const int key_heads, const int value_heads
) {
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int base = chunk * DELTA_CHUNK;
    const int valid = rows - base < DELTA_CHUNK ? rows - base : DELTA_CHUNK;
    if (valid <= 0) return;
    const int key_head = head % key_heads;
    const int total_key = key_heads * DELTA_DIM;
    const int key_off = key_head * DELTA_DIM;
    const long long stride = (long long)(total_key * 2 + value_heads * DELTA_DIM);
    const int tid = threadIdx.x;

    __shared__ float tri[DELTA_CHUNK][DELTA_CHUNK];   // A, overwritten by T
    __shared__ float slab[DELTA_CHUNK][DELTA_CHUNK];  // staged right-hand side
    __shared__ float kinv[DELTA_CHUNK];

    const long long mat = ((long long)chunk * value_heads + head)
        * (DELTA_CHUNK * DELTA_CHUNK);
    for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x)
        tri[index / DELTA_CHUNK][index % DELTA_CHUNK] = attn[mat + index];
    if (tid < valid)
        kinv[tid] = kinv_in[(long long)(base + tid) * value_heads + head];
    __syncthreads();

    // Forward substitution, one row at a time and in place. Row i reads A[i][m]
    // before it is overwritten, and rows m < i already hold T. Every thread
    // reaches both barriers so the row swap is ordered block-wide.
    for (int i = 0; i < valid; ++i) {
        float sum = 0.0f;
        if (tid < DELTA_CHUNK) {
            sum = (tid == i) ? 1.0f : 0.0f;
            for (int m = 0; m < i; ++m) sum -= tri[i][m] * tri[m][tid];
        }
        __syncthreads();
        if (tid < DELTA_CHUNK) tri[i][tid] = sum;
        __syncthreads();
    }

    // Apply T to both right-hand sides, 64 columns at a time.
    //   Wm[i][d] = sum_m T[i][m] * exp(G_m) * beta_m * kn_m[d]
    //   U[i][d]  = sum_m T[i][m] * beta_m * v_m[d]
    for (int side = 0; side < 2; ++side) {
        for (int slab_base = 0; slab_base < DELTA_DIM; slab_base += DELTA_CHUNK) {
            for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x) {
                const int m = index / DELTA_CHUNK, d = index % DELTA_CHUNK;
                float value = 0.0f;
                if (m < valid) {
                    const long long scalar = (long long)(base + m) * value_heads + head;
                    const float* row = convolved + (long long)(base + m) * stride;
                    const float beta = beta_in[scalar];
                    if (side == 0) {
                        value = row[total_key + key_off + slab_base + d] * kinv[m]
                            * beta * expf(g_cumsum[scalar]);
                    } else {
                        value = row[total_key * 2 + head * DELTA_DIM + slab_base + d] * beta;
                    }
                }
                slab[m][d] = value;
            }
            __syncthreads();
            for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x) {
                const int i = index / DELTA_CHUNK, d = index % DELTA_CHUNK;
                float sum = 0.0f;
                if (i < valid)
                    for (int m = 0; m <= i; ++m) sum += tri[i][m] * slab[m][d];
                if (i < valid) {
                    float* target = side == 0 ? w_rows : u_rows;
                    target[((long long)(base + i) * value_heads + head) * DELTA_DIM
                           + slab_base + d] = sum;
                }
            }
            __syncthreads();
        }
    }
}

// Pass 3: the only serial part. Walks chunks in order, carrying the state, and
// emits the un-normalized core outputs. The value dimension is independent
// across the whole recurrence, so the grid splits it into tiles and each block
// owns DELTA_TILE value columns of one head's state.
// Grid (value_heads, DELTA_DIM / DELTA_TILE), block 256.
#define DELTA_TILE 32

extern "C" __global__
void qwen_delta_state_pass(
    const float* convolved, const float* pmat, const float* g_cumsum,
    const float* qinv_in, const float* kinv_in,
    const float* w_rows, const float* u_rows, float* state, float* core,
    const int rows, const int key_heads, const int value_heads
) {
    const int head = blockIdx.x;
    const int tile = blockIdx.y * DELTA_TILE;
    const int key_head = head % key_heads;
    const int total_key = key_heads * DELTA_DIM;
    const int key_off = key_head * DELTA_DIM;
    const long long stride = (long long)(total_key * 2 + value_heads * DELTA_DIM);
    const int tid = threadIdx.x;
    const int chunks = (rows + DELTA_CHUNK - 1) / DELTA_CHUNK;

    __shared__ float carried[DELTA_DIM][DELTA_TILE];   // 16 KB
    __shared__ float omega[DELTA_CHUNK][DELTA_TILE];   // 8 KB
    // One staging buffer, re-carved per stage: the three matrix products need
    // different operands but never two of them at once, and a dedicated buffer
    // for each would not fit under the 48 KB the driver allows.
    __shared__ float scratch[DELTA_CHUNK * (DELTA_CHUNK + 1)];   // 16.6 KB
    // Per-token scales, hoisted out of the matrix loops: both the query and the
    // key factor depend only on the token, so computing them per output element
    // would run one expf per MAC.
    __shared__ float qscale[DELTA_CHUNK];   // qinv_i * exp(G_i)
    __shared__ float kscale[DELTA_CHUNK];   // kinv_j * exp(G_last - G_j)

    // Output tiles. Every stage gives a thread several outputs so the operand
    // loads amortize: the one-output-per-thread form spent two loads per FMA.
    const int col = (tid & 15) * 2;          // 2 value columns
    const int band = tid >> 4;               // 16 bands of rows
    const int row4 = band * 4;               // 4 token rows  (64-row stages)
    const int row8 = band * 8;               // 8 state rows  (128-row stage)

    for (int index = tid; index < DELTA_DIM * DELTA_TILE; index += blockDim.x) {
        const int m = index / DELTA_TILE, d = index % DELTA_TILE;
        carried[m][d] = state[((long long)head * DELTA_DIM + m) * DELTA_DIM + tile + d];
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int base = chunk * DELTA_CHUNK;
        const int valid = rows - base < DELTA_CHUNK ? rows - base : DELTA_CHUNK;
        const long long mat = ((long long)chunk * value_heads + head)
            * (DELTA_CHUNK * DELTA_CHUNK);
        const float g_last =
            g_cumsum[(long long)(base + valid - 1) * value_heads + head];
        // Fold the decay into the published normalizers; both scales depend
        // only on the token, so they must stay out of the matrix loops.
        if (tid < valid) {
            const long long scalar = (long long)(base + tid) * value_heads + head;
            const float g = g_cumsum[scalar];
            qscale[tid] = qinv_in[scalar] * expf(g);
            kscale[tid] = kinv_in[scalar] * expf(g_last - g);
        }
        __syncthreads();

        // Wm * S_start and Qtilde * S_start share the same S operand and the same
        // shape, so they run as one pass over the reduction dimension and read
        // the state tile once for both.
        float omega_acc[4][2] = {}, core_acc[4][2] = {};
        for (int slab = 0; slab < DELTA_DIM; slab += DELTA_SLAB) {
            // Consecutive threads walk the reduction dimension, which is the
            // contiguous axis of both sources.
            for (int index = tid; index < DELTA_SLAB * DELTA_CHUNK; index += blockDim.x) {
                const int i = index / DELTA_SLAB, m = index % DELTA_SLAB;
                float w = 0.0f, q = 0.0f;
                if (i < valid) {
                    w = w_rows[((long long)(base + i) * value_heads + head) * DELTA_DIM
                               + slab + m];
                    q = convolved[(long long)(base + i) * stride + key_off + slab + m]
                        * qscale[i];
                }
                scratch[m * (DELTA_CHUNK + 1) + i] = w;
                scratch[DELTA_SLAB * (DELTA_CHUNK + 1) + m * (DELTA_CHUNK + 1) + i] = q;
            }
            __syncthreads();
            for (int m = 0; m < DELTA_SLAB; ++m) {
                float state_column[2], w_row[4], q_row[4];
                #pragma unroll
                for (int c = 0; c < 2; ++c) state_column[c] = carried[slab + m][col + c];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    w_row[r] = scratch[m * (DELTA_CHUNK + 1) + row4 + r];
                    q_row[r] = scratch[DELTA_SLAB * (DELTA_CHUNK + 1)
                                       + m * (DELTA_CHUNK + 1) + row4 + r];
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r)
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        omega_acc[r][c] += w_row[r] * state_column[c];
                        core_acc[r][c] += q_row[r] * state_column[c];
                    }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = row4 + r;
            #pragma unroll
            for (int c = 0; c < 2; ++c)
                omega[i][col + c] = i < valid
                    ? u_rows[((long long)(base + i) * value_heads + head) * DELTA_DIM
                             + tile + col + c] - omega_acc[r][c]
                    : 0.0f;
        }
        __syncthreads();

        // core_i also takes sum_{j<=i} P_ij omega_j. Both terms are against the
        // incoming state, so this must complete before the state update.
        for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x)
            scratch[(index / DELTA_CHUNK) * (DELTA_CHUNK + 1) + index % DELTA_CHUNK] =
                pmat[mat + index];
        __syncthreads();
        for (int j = 0; j < DELTA_CHUNK; ++j) {
            float omega_column[2], p_row[4];
            #pragma unroll
            for (int c = 0; c < 2; ++c) omega_column[c] = omega[j][col + c];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                p_row[r] = scratch[(row4 + r) * (DELTA_CHUNK + 1) + j];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int c = 0; c < 2; ++c) core_acc[r][c] += p_row[r] * omega_column[c];
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = row4 + r;
            if (i >= valid) continue;
            #pragma unroll
            for (int c = 0; c < 2; ++c)
                core[((long long)(base + i) * value_heads + head) * DELTA_DIM
                     + tile + col + c] = core_acc[r][c];
        }
        __syncthreads();

        // S_end = exp(G_last) S_start + Ktilde^T Omega, over two halves of the
        // 128 state rows so the staged Ktilde slab reuses the same scratch.
        const float closing = expf(g_last);
        for (int half = 0; half < 2; ++half) {
            const int m_base = half * DELTA_CHUNK;
            for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x) {
                const int j = index / DELTA_CHUNK, m = index % DELTA_CHUNK;
                scratch[j * (DELTA_CHUNK + 1) + m] = j < valid
                    ? convolved[(long long)(base + j) * stride + total_key + key_off
                                + m_base + m] * kscale[j]
                    : 0.0f;
            }
            __syncthreads();
            // This half owns state rows [m_base, m_base + 64); each thread takes
            // four of them against its two value columns.
            float acc[4][2] = {};
            for (int j = 0; j < DELTA_CHUNK; ++j) {
                float omega_column[2], k_row[4];
                #pragma unroll
                for (int c = 0; c < 2; ++c) omega_column[c] = omega[j][col + c];
                #pragma unroll
)COLIBRI_CUDA"
R"COLIBRI_CUDA(                for (int r = 0; r < 4; ++r)
                    k_row[r] = scratch[j * (DELTA_CHUNK + 1) + row4 + r];
                #pragma unroll
                for (int r = 0; r < 4; ++r)
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) acc[r][c] += k_row[r] * omega_column[c];
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int m = m_base + row4 + r;
                #pragma unroll
                for (int c = 0; c < 2; ++c)
                    carried[m][col + c] = carried[m][col + c] * closing + acc[r][c];
            }
            __syncthreads();
        }
    }

    for (int index = tid; index < DELTA_DIM * DELTA_TILE; index += blockDim.x) {
        const int m = index / DELTA_TILE, d = index % DELTA_TILE;
        state[((long long)head * DELTA_DIM + m) * DELTA_DIM + tile + d] = carried[m][d];
    }
}

// Pass 4: the per-token epilogue the sequential kernels fold into their inner
// loop -- RMS norm across the head's value dimension, the learned weight, and
// the SiLU gate. Grid (rows, value_heads), block DELTA_DIM.
extern "C" __global__
void qwen_delta_norm_gate(
    const float* core, const float* gates, const float* norm_weights,
    float* output, const int value_heads, const float epsilon
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    const long long offset =
        ((long long)token * value_heads + head) * DELTA_DIM + lane;
    const float value = core[offset];

    __shared__ float sums[4];
    float square = value * value;
    for (int step = 16; step > 0; step >>= 1)
        square += __shfl_down_sync(0xffffffff, square, step);
    if ((lane & 31) == 0) sums[lane >> 5] = square;
    __syncthreads();
    const float total = sums[0] + sums[1] + sums[2] + sums[3];
    const float inverse_rms = rsqrtf(total / (float)DELTA_DIM + epsilon);

    const float gate = gates[offset];
    output[offset] = value * inverse_rms * norm_weights[lane]
        * gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))));
}

)COLIBRI_CUDA";
}
