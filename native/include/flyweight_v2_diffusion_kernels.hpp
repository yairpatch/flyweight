#pragma once

// Device kernels for the diffusion tower (native/src/v2_diffusion.inc): the
// Z-Image single-stream DiT, its Qwen3 text encoder, and the KL autoencoder
// decoder that renders the latent. Appended to the Qwen corpus after
// flyweight_v2_native_kernels.hpp, so block_reduce_sum and the vision GEMMs are
// in scope; everything here is f32 activations, [rows][width] row-major for the
// transformers and [channels][height][width] for the autoencoder.
//
// The GEMMs are not here: quantized weights go through the corpus's MMQ and
// rows kernels via the format table, f32/bf16 ones through vision_*_gemm_rows.

namespace flyweight::v2 {

inline constexpr char diffusion_cuda_source[] = R"FLYWEIGHT_CUDA(
// ---- Diffusion tower -------------------------------------------------------

// output[r][c] = input[r][c] + bias[c]. Adds a GEMM bias the MMQ path has no
// slot for; in place when output == input.
extern "C" __global__
void diff_add_bias_rows(const float* input, const float* bias, float* output,
                        const int width, const long long elements) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x)
        output[index] = input[index] + bias[index % width];
}

// x[r][c] *= scale[c] (adaLN's 1 + scale, one vector for every row).
extern "C" __global__
void diff_scale_columns(float* x, const float* scale, const int width,
                        const long long elements) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x)
        x[index] *= scale[index % width];
}

// x[r][c] += gate[c] * y[r][c]; gate may be null for a plain residual add.
extern "C" __global__
void diff_gated_add_rows(float* x, const float* y, const float* gate,
                         const int width, const long long elements) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x)
        x[index] += (gate ? gate[index % width] : 1.0f) * y[index];
}

// out = silu(gate) * up, separate buffers.
extern "C" __global__
void diff_silu_mul(const float* gate, const float* up, float* output,
                   const long long elements) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x) {
        const float g = gate[index];
        output[index] = g / (1.0f + expf(-g)) * up[index];
    }
}

// x = silu(x), in place.
extern "C" __global__
void diff_silu_rows(float* x, const long long elements) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x) {
        const float v = x[index];
        x[index] = v / (1.0f + expf(-v));
    }
}

// The four adaLN vectors of one block from its modulation projection:
// [scale_msa | gate_msa | scale_mlp | gate_mlp], each `width` wide. Scales
// become 1 + s, gates tanh(g), in place.
extern "C" __global__
void diff_modulation_prepare(float* modulation, const int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 4 * width) return;
    const int which = index / width;
    modulation[index] = (which & 1) ? tanhf(modulation[index]) : 1.0f + modulation[index];
}

// Overwrites rows [first, first + count) with one `width`-wide vector (the
// DiT's learned pad token on sequence padding).
extern "C" __global__
void diff_fill_rows(float* x, const float* vector, const int first,
                    const int count, const int width) {
    const long long elements = (long long)count * width;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x)
        x[(long long)first * width + index] = vector[index % width];
}

// Per-row LayerNorm without affine parameters (the DiT's final norm).
extern "C" __global__
void diff_layer_norm_rows(const float* input, float* output,
                          const int width, const int rows, const float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* source = input + (long long)row * width;
    float* target = output + (long long)row * width;
    float sum = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) sum += source[index];
    sum = block_reduce_sum(sum);
    __shared__ float mean;
    if (threadIdx.x == 0) mean = sum / (float)width;
    __syncthreads();
    float square = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float centered = source[index] - mean;
        square += centered * centered;
    }
    square = block_reduce_sum(square);
    __shared__ float inverse_deviation;
    if (threadIdx.x == 0) inverse_deviation = rsqrtf(square / (float)width + epsilon);
    __syncthreads();
    for (int index = threadIdx.x; index < width; index += blockDim.x)
        target[index] = (source[index] - mean) * inverse_deviation;
}

// RMSNorm over each head's head_dim slice of a strided row buffer: `x` is
// [rows][row_stride], the heads start at `offset` within the row. One warp
// per (row, head); head_dim <= 256. Covers the DiT's q/k norms inside the
// fused qkv rows and Qwen3's q_norm/k_norm on separate projections.
extern "C" __global__
void diff_head_rms_norm(float* x, const float* weight, const int heads,
                        const int head_dim, const int rows, const int row_stride,
                        const int offset, const float epsilon) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row = warp / heads, head = warp % heads;
    if (row >= rows) return;
    float* vector = x + (long long)row * row_stride + offset + head * head_dim;
    float sum = 0.0f;
    for (int d = lane; d < head_dim; d += 32) sum += vector[d] * vector[d];
    for (int shift = 16; shift > 0; shift >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, shift);
    const float inverse = rsqrtf(sum / (float)head_dim + epsilon);
    for (int d = lane; d < head_dim; d += 32) vector[d] = vector[d] * inverse * weight[d];
}

// Multi-axis rotary embedding on adjacent pairs (torch.view_as_complex over
// reshape(-1, 2)): head_dim is split into `axis_dims[a]` slices, each rotated
// by that axis of the row's position with its own ramp theta^(-2j/axis_dim).
// `frequencies` holds the ramp per pair (head_dim/2 doubles, host-computed:
// a device pow per pair per row was a third of the rope's cost). Applied to
// the q and k sections of a fused [rows][3*heads*head_dim] projection
// (blockIdx.y selects q or k). positions is [rows][3] int32.
extern "C" __global__
void diff_rope_axes_rows(float* qkv, const int* positions, const double* frequencies,
                         const int heads, const int head_dim, const int rows,
                         const int axis_dim0, const int axis_dim1) {
    const int row = blockIdx.x, section = blockIdx.y;
    if (row >= rows || section > 1) return;
    const int stride = 3 * heads * head_dim;
    float* base = qkv + (long long)row * stride + section * heads * head_dim;
    const int pairs = head_dim / 2;
    const int pairs0 = axis_dim0 / 2, pairs1 = axis_dim1 / 2;
    for (int index = threadIdx.x; index < heads * pairs; index += blockDim.x) {
        const int head = index / pairs, pair = index % pairs;
        const int axis = pair < pairs0 ? 0 : pair < pairs0 + pairs1 ? 1 : 2;
        const float angle = (float)((double)positions[row * 3 + axis] * frequencies[pair]);
        float s, c;
        sincosf(angle, &s, &c);
        float* vector = base + head * head_dim + 2 * pair;
        const float re = vector[0], im = vector[1];
        vector[0] = re * c - im * s;
        vector[1] = re * s + im * c;
    }
}

// Standard rotate-half rope (Qwen3): pairs (j, j + head_dim/2), one position
// per row, `frequencies[j]` = theta^(-2j/head_dim) (head_dim/2 doubles). `x`
// is [rows][row_stride] with the heads at `offset`.
extern "C" __global__
void diff_rope_half_rows(float* x, const int* positions, const double* frequencies,
                         const int heads, const int head_dim, const int rows,
                         const int row_stride, const int offset) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int half = head_dim / 2;
    float* base = x + (long long)row * row_stride + offset;
    const double position = (double)positions[row];
    for (int index = threadIdx.x; index < heads * half; index += blockDim.x) {
        const int head = index / half, j = index % half;
        float s, c;
        sincosf((float)(position * frequencies[j]), &s, &c);
        float* vector = base + head * head_dim;
        const float first = vector[j], second = vector[j + half];
        vector[j] = first * c - second * s;
        vector[j + half] = second * c + first * s;
    }
}

)FLYWEIGHT_CUDA"
R"FLYWEIGHT_CUDA(
// Attention over `rows` queries against the same rows of keys/values, with
// separate strided q/k/v pointers (a fused qkv row is q at stride 3*H*D, k
// and v at offsets inside it), grouped query heads (kv_head = head / (heads /
// kv_heads)), and an optional causal mask. Same shape as
// vision_attention_rows: one block per (head, 32 queries), keys streamed
// through shared memory 24 at a time, 8 threads per query with an online
// softmax merged by shuffles. head_dim <= 128; blockDim 256. Output is
// [rows][heads*head_dim].
#define FLYWEIGHT_DIFF_KEY_CHUNK 24
extern "C" __global__
void diff_attention_rows(
    const float* q, const float* k, const float* v, float* output,
    const int heads, const int kv_heads, const int head_dim, const int rows,
    const int q_stride, const int kv_stride, const float scale, const int causal
) {
    const int head = blockIdx.y;
    const int tile = blockIdx.x * 32;
    if (head >= heads || tile >= rows) return;
    const int kv_head = head / (heads / kv_heads);
    const int query_index = threadIdx.x >> 3;
    const int lane_in_query = threadIdx.x & 7;
    const int row = tile + query_index;
    const bool live = row < rows;
    __shared__ float k_tile[FLYWEIGHT_DIFF_KEY_CHUNK][129];
    __shared__ float v_tile[FLYWEIGHT_DIFF_KEY_CHUNK][129];
    __shared__ float q_tile[32][129];
    if (live)
        for (int d = lane_in_query; d < head_dim; d += 8)
            q_tile[query_index][d] = q[(long long)row * q_stride + head * head_dim + d] * scale;
    __syncthreads();
    float running_max = -3.0e38f, running_sum = 0.0f;
    float acc[128];
    for (int d = 0; d < 128; ++d) acc[d] = 0.0f;
    // With a causal mask no query in this tile sees past the tile's last row.
    const int key_limit = causal ? min(rows, tile + 32) : rows;
    for (int chunk = 0; chunk < key_limit; chunk += FLYWEIGHT_DIFF_KEY_CHUNK) {
        const int count = min(FLYWEIGHT_DIFF_KEY_CHUNK, key_limit - chunk);
        for (int load = threadIdx.x; load < FLYWEIGHT_DIFF_KEY_CHUNK * head_dim; load += blockDim.x) {
            const int key = load / head_dim, d = load % head_dim;
            if (key < count) {
                const long long base = (long long)(chunk + key) * kv_stride + kv_head * head_dim + d;
                k_tile[key][d] = k[base];
                v_tile[key][d] = v[base];
            }
        }
        __syncthreads();
        if (live) {
            for (int key = lane_in_query; key < count; key += 8) {
                if (causal && chunk + key > row) continue;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q_tile[query_index][d] * k_tile[key][d];
                if (dot > running_max) {
                    const float rescale = expf(running_max - dot);
                    running_sum *= rescale;
#pragma unroll
                    for (int d = 0; d < 128; ++d) acc[d] *= rescale;
                    running_max = dot;
                }
                const float weight = expf(dot - running_max);
                running_sum += weight;
#pragma unroll
                for (int d = 0; d < 128; ++d)
                    if (d < head_dim) acc[d] += weight * v_tile[key][d];
            }
        }
        __syncthreads();
    }
    float shared_max = running_max;
    for (int offset = 4; offset > 0; offset >>= 1)
        shared_max = fmaxf(shared_max, __shfl_xor_sync(0xffffffff, shared_max, offset));
    const float rescale = expf(running_max - shared_max);
    running_sum *= rescale;
    for (int offset = 4; offset > 0; offset >>= 1)
        running_sum += __shfl_xor_sync(0xffffffff, running_sum, offset);
#pragma unroll
    for (int d = 0; d < 128; ++d) {
        float value = acc[d] * rescale;
        for (int offset = 4; offset > 0; offset >>= 1)
            value += __shfl_xor_sync(0xffffffff, value, offset);
        acc[d] = value;
    }
    if (live) {
#pragma unroll
        for (int d = 0; d < 128; ++d)
            if (d < head_dim && (d & 7) == lane_in_query)
                output[(long long)row * heads * head_dim + head * head_dim + d] = acc[d] / running_sum;
    }
}

// ---- Tensor-core attention -------------------------------------------------
// The f32 kernel above is O(rows^2) at a few TFLOPS, which at 4096 image
// tokens is three seconds a step. This pair runs the same attention on bf16
// mma.sync tiles (sm_80+; the host falls back to diff_attention_rows
// elsewhere): 64 queries per 128-thread block, each warp owning 16 rows with
// the Q fragments in registers, keys and values streamed through shared
// memory 64 at a time with the flash online softmax. head_dim is 128.
//
// f32 rows -> head-major bf16 planes [heads][rows][128], the queries
// pre-scaled by `q_scale` (the softmax scale times log2 e, so the softmax
// runs on exp2). Round-to-nearest-even by hand: the corpus is also host C++.
__device__ __forceinline__ unsigned short diff_f32_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return (unsigned short)(bits >> 16);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}

extern "C" __global__
void diff_pack_attention_bf16(
    const float* q, const float* k, const float* v,
    unsigned short* q_out, unsigned short* k_out, unsigned short* v_out,
    const int heads, const int kv_heads, const int rows,
    const int q_stride, const int kv_stride, const float q_scale
) {
    const long long q_elements = (long long)heads * rows * 128;
    const long long kv_elements = (long long)kv_heads * rows * 128;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < q_elements + 2 * kv_elements; index += (long long)blockDim.x * gridDim.x) {
        if (index < q_elements) {
            const int d = (int)(index & 127);
            const long long rest = index >> 7;
            const int row = (int)(rest % rows), head = (int)(rest / rows);
            q_out[index] = diff_f32_to_bf16(q[(long long)row * q_stride + head * 128 + d] * q_scale);
        } else {
            const long long local = (index - q_elements) % kv_elements;
            const bool is_v = index - q_elements >= kv_elements;
            const int d = (int)(local & 127);
            const long long rest = local >> 7;
            const int row = (int)(rest % rows), head = (int)(rest / rows);
            const float value = (is_v ? v : k)[(long long)row * kv_stride + head * 128 + d];
            (is_v ? v_out : k_out)[local] = diff_f32_to_bf16(value);
        }
    }
}

#define FLYWEIGHT_DIFF_FLASH_QUERIES 64
#define FLYWEIGHT_DIFF_FLASH_KEYS 64
extern "C" __global__ __launch_bounds__(128)
void diff_flash_attention_bf16(
    const unsigned short* q, const unsigned short* k, const unsigned short* v,
    float* output, const int heads, const int kv_heads, const int rows, const int causal
) {
    // Declared as uint4 rows for 16-byte alignment (the host build of the
    // corpus has no __align__); read through the half-width views below.
    __shared__ uint4 k_tile4[FLYWEIGHT_DIFF_FLASH_KEYS][17];
    __shared__ uint4 vt_tile4[128][9];
    unsigned short (*k_tile)[136] = (unsigned short (*)[136])k_tile4;
    unsigned short (*vt_tile)[72] = (unsigned short (*)[72])vt_tile4;
    const int head = blockIdx.y;
    const int kv_head = head / (heads / kv_heads);
    const int query_base = blockIdx.x * FLYWEIGHT_DIFF_FLASH_QUERIES;
    if (query_base >= rows) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int quad = lane >> 2, pair = lane & 3;   // fragment row and column pair
    const int row0 = query_base + warp * 16 + quad, row1 = row0 + 8;
    const unsigned short* q_head = q + (long long)head * rows * 128;
    const unsigned short* k_head = k + (long long)kv_head * rows * 128;
    const unsigned short* v_head = v + (long long)kv_head * rows * 128;

    // Q fragments: A operand of m16n8k16 for eight k-steps of 16.
    unsigned int q_frag[8][4];
    for (int step = 0; step < 8; ++step) {
        const int column = step * 16 + pair * 2;
        q_frag[step][0] = row0 < rows ? *(const unsigned int*)(q_head + (long long)row0 * 128 + column) : 0u;
        q_frag[step][1] = row1 < rows ? *(const unsigned int*)(q_head + (long long)row1 * 128 + column) : 0u;
        q_frag[step][2] = row0 < rows ? *(const unsigned int*)(q_head + (long long)row0 * 128 + column + 8) : 0u;
        q_frag[step][3] = row1 < rows ? *(const unsigned int*)(q_head + (long long)row1 * 128 + column + 8) : 0u;
    }
    float o[16][4];
    for (int d = 0; d < 16; ++d) for (int i = 0; i < 4; ++i) o[d][i] = 0.0f;
    float m0 = -1.0e30f, m1 = -1.0e30f, l0 = 0.0f, l1 = 0.0f;

    const int key_limit = causal ? min(rows, query_base + FLYWEIGHT_DIFF_FLASH_QUERIES) : rows;
    for (int key_base = 0; key_base < key_limit; key_base += FLYWEIGHT_DIFF_FLASH_KEYS) {
        // Stage K [64][128] and V^T [128][64]; keys past the end read as zero.
        for (int load = threadIdx.x; load < FLYWEIGHT_DIFF_FLASH_KEYS * 16; load += 128) {
            const int key = load >> 4, chunk = (load & 15) * 8;
            const int key_index = key_base + key;
            uint4 k_bits = make_uint4(0u, 0u, 0u, 0u), v_bits = make_uint4(0u, 0u, 0u, 0u);
            if (key_index < rows) {
                k_bits = *(const uint4*)(k_head + (long long)key_index * 128 + chunk);
                v_bits = *(const uint4*)(v_head + (long long)key_index * 128 + chunk);
            }
            *(uint4*)(&k_tile[key][chunk]) = k_bits;
            const unsigned short* v_halves = (const unsigned short*)&v_bits;
            for (int i = 0; i < 8; ++i) vt_tile[chunk + i][key] = v_halves[i];
        }
        __syncthreads();

        // S = Q K^T over eight n-tiles of 8 keys.
        float s[8][4];
        for (int j = 0; j < 8; ++j) for (int i = 0; i < 4; ++i) s[j][i] = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const int key = j * 8 + quad;
            for (int step = 0; step < 8; ++step) {
                unsigned int b[2];
                b[0] = *(const unsigned int*)(&k_tile[key][step * 16 + pair * 2]);
                b[1] = *(const unsigned int*)(&k_tile[key][step * 16 + pair * 2 + 8]);
                kv_mma_m16n8k16(s[j], q_frag[step], b, (const __nv_bfloat16*)0);
            }
        }
        // Mask keys past the end and, for causal, past the query.
        float block_max0 = -1.0e30f, block_max1 = -1.0e30f;
        for (int j = 0; j < 8; ++j) {
            const int key = key_base + j * 8 + pair * 2;
            for (int i = 0; i < 2; ++i) {
                const int key_index = key + i;
                if (key_index >= rows || (causal && key_index > row0)) s[j][i] = -1.0e30f;
                if (key_index >= rows || (causal && key_index > row1)) s[j][2 + i] = -1.0e30f;
            }
            block_max0 = fmaxf(block_max0, fmaxf(s[j][0], s[j][1]));
            block_max1 = fmaxf(block_max1, fmaxf(s[j][2], s[j][3]));
        }
        for (int shift = 1; shift <= 2; shift <<= 1) {
            block_max0 = fmaxf(block_max0, __shfl_xor_sync(0xffffffff, block_max0, shift));
            block_max1 = fmaxf(block_max1, __shfl_xor_sync(0xffffffff, block_max1, shift));
        }
        const float new_m0 = fmaxf(m0, block_max0), new_m1 = fmaxf(m1, block_max1);
        const float alpha0 = exp2f(m0 - new_m0), alpha1 = exp2f(m1 - new_m1);
        float sum0 = 0.0f, sum1 = 0.0f;
        unsigned int p_frag[4][4];
        for (int j = 0; j < 8; ++j) {
            const float p0 = exp2f(s[j][0] - new_m0), p1 = exp2f(s[j][1] - new_m0);
            const float p2 = exp2f(s[j][2] - new_m1), p3 = exp2f(s[j][3] - new_m1);
            sum0 += p0 + p1;
            sum1 += p2 + p3;
            // C fragment of tiles (2t, 2t+1) is the A fragment of k-step t.
            const int t = j >> 1;
            if ((j & 1) == 0) {
                p_frag[t][0] = kv_mma_pack<__nv_bfloat16>(p0, p1);
                p_frag[t][1] = kv_mma_pack<__nv_bfloat16>(p2, p3);
            } else {
                p_frag[t][2] = kv_mma_pack<__nv_bfloat16>(p0, p1);
                p_frag[t][3] = kv_mma_pack<__nv_bfloat16>(p2, p3);
            }
        }
        for (int shift = 1; shift <= 2; shift <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffff, sum0, shift);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, shift);
        }
        l0 = l0 * alpha0 + sum0;
        l1 = l1 * alpha1 + sum1;
        m0 = new_m0;
        m1 = new_m1;
        for (int d = 0; d < 16; ++d) {
            o[d][0] *= alpha0; o[d][1] *= alpha0;
            o[d][2] *= alpha1; o[d][3] *= alpha1;
        }
        // O += P V over sixteen n-tiles of 8 dims, four k-steps of 16 keys.
        for (int d = 0; d < 16; ++d) {
            const int dim = d * 8 + quad;
            for (int t = 0; t < 4; ++t) {
                unsigned int b[2];
                b[0] = *(const unsigned int*)(&vt_tile[dim][t * 16 + pair * 2]);
                b[1] = *(const unsigned int*)(&vt_tile[dim][t * 16 + pair * 2 + 8]);
                kv_mma_m16n8k16(o[d], p_frag[t], b, (const __nv_bfloat16*)0);
            }
        }
        __syncthreads();
    }
    const float inverse0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inverse1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const long long out_stride = (long long)heads * 128;
    for (int d = 0; d < 16; ++d) {
        const int column = head * 128 + d * 8 + pair * 2;
        if (row0 < rows) {
            float2 value = make_float2(o[d][0] * inverse0, o[d][1] * inverse0);
            *(float2*)(output + row0 * out_stride + column) = value;
        }
        if (row1 < rows) {
            float2 value = make_float2(o[d][2] * inverse1, o[d][3] * inverse1);
            *(float2*)(output + row1 * out_stride + column) = value;
        }
    }
}

)FLYWEIGHT_CUDA"
R"FLYWEIGHT_CUDA(
// ---- bf16 tensor-core GEMM over Q8_0 weights --------------------------------
// output[rows][out] = input[rows][in] (bf16) x W^T with W stored Q8_0 (34-byte
// blocks of 32: f16 scale then 32 int8), dequantized to bf16 as each block is
// staged. The weights keep their stored precision and only the activations
// round to bf16 -- diffusers' own numerics -- where the MMQ path also rounds
// activations to int8. 128x64 tile per 256-thread block (4x2 warps of
// 32x32), K walked two Q8 blocks (64) at a time through shared memory, the
// next slab fetched into registers while the current one is multiplied.
// Grid: (ceil(out/64), ceil(rows/128)). input_size must be a multiple of 64.
extern "C" __global__ __launch_bounds__(256)
void diff_q8_bf16_gemm(const unsigned char* packed, const unsigned short* input, float* output,
                       const int input_size, const int output_size, const int rows) {
    __shared__ uint4 a4[128][9];   // [128 rows][72 halves]
    __shared__ uint4 b4[64][9];    // [64 cols][72 halves]
    unsigned short (*a_tile)[72] = (unsigned short (*)[72])a4;
    unsigned short (*b_tile)[72] = (unsigned short (*)[72])b4;
    const int row_base = blockIdx.y * 128, col_base = blockIdx.x * 64;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int quad = lane >> 2, pair = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const int blocks_per_row = input_size >> 5;
    float acc[2][4][4];
    for (int m = 0; m < 2; ++m) for (int n = 0; n < 4; ++n) for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.0f;

    // Each thread stages 4 uint4 of A (128 rows x 64 halves = 1024 uint4) and
    // one Q8 block of B (64 cols x 2 blocks = 128 blocks; 256 threads take
    // half a block each: 16 values).
    uint4 a_next[4];
    unsigned short b_next[16];
    auto fetch = [&](int slab) {
        for (int i = 0; i < 4; ++i) {
            const int index = threadIdx.x + i * 256;
            const int r = index >> 3, c = (index & 7) * 8;
            const int row = row_base + r;
            a_next[i] = row < rows
                ? *(const uint4*)(input + (long long)row * input_size + slab * 64 + c)
                : make_uint4(0u, 0u, 0u, 0u);
        }
        const int col = threadIdx.x >> 2, part = threadIdx.x & 3;   // part: block (0/1) x half (0/1)
        const int n = col_base + col;
        if (n < output_size) {
            const unsigned char* block = packed + ((long long)n * blocks_per_row + slab * 2 + (part >> 1)) * 34;
            unsigned short scale_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
            __half scale_half;
            memcpy(&scale_half, &scale_bits, sizeof(scale_half));
            const float scale = __half2float(scale_half);
            const signed char* q = (const signed char*)(block + 2 + (part & 1) * 16);
            for (int j = 0; j < 16; ++j) b_next[j] = diff_f32_to_bf16(scale * (float)q[j]);
        } else {
            for (int j = 0; j < 16; ++j) b_next[j] = 0;
        }
    };
    auto store = [&]() {
        for (int i = 0; i < 4; ++i) {
            const int index = threadIdx.x + i * 256;
            a4[index >> 3][index & 7] = a_next[i];
        }
        const int col = threadIdx.x >> 2, part = threadIdx.x & 3;
        uint4* target = &b4[col][part * 2];
        target[0] = *(const uint4*)&b_next[0];
        target[1] = *(const uint4*)&b_next[8];
    };

    const int slabs = input_size >> 6;
    fetch(0);
    for (int slab = 0; slab < slabs; ++slab) {
        store();
        __syncthreads();
        if (slab + 1 < slabs) fetch(slab + 1);
        for (int ks = 0; ks < 4; ++ks) {
            unsigned int a[2][4];
            for (int m = 0; m < 2; ++m) {
                const int r = wm * 32 + m * 16 + quad, k = ks * 16 + pair * 2;
                a[m][0] = *(const unsigned int*)&a_tile[r][k];
                a[m][1] = *(const unsigned int*)&a_tile[r + 8][k];
                a[m][2] = *(const unsigned int*)&a_tile[r][k + 8];
                a[m][3] = *(const unsigned int*)&a_tile[r + 8][k + 8];
            }
            for (int n = 0; n < 4; ++n) {
                const int c = wn * 32 + n * 8 + quad, k = ks * 16 + pair * 2;
                unsigned int b[2];
                b[0] = *(const unsigned int*)&b_tile[c][k];
                b[1] = *(const unsigned int*)&b_tile[c][k + 8];
                for (int m = 0; m < 2; ++m) kv_mma_m16n8k16(acc[m][n], a[m], b, (const __nv_bfloat16*)0);
            }
        }
        __syncthreads();
    }
    for (int m = 0; m < 2; ++m)
        for (int n = 0; n < 4; ++n) {
            const int col = col_base + wn * 32 + n * 8 + pair * 2;
            if (col >= output_size) continue;
            const int row0 = row_base + wm * 32 + m * 16 + quad, row1 = row0 + 8;
            if (row0 < rows) *(float2*)(output + (long long)row0 * output_size + col) = make_float2(acc[m][n][0], acc[m][n][1]);
            if (row1 < rows) *(float2*)(output + (long long)row1 * output_size + col) = make_float2(acc[m][n][2], acc[m][n][3]);
        }
}

// f32 rows -> bf16 rows, the GEMM above's activation input.
extern "C" __global__
void diff_pack_rows_bf16(const float* input, unsigned short* output, const long long elements) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x)
        output[index] = diff_f32_to_bf16(input[index]);
}

// Row softmax in place over [rows][width] (the autoencoder's mid-block
// attention scores, whose 512-wide single head the kernel above cannot hold).
extern "C" __global__
void diff_softmax_rows(float* x, const int width, const int rows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float* line = x + (long long)row * width;
    __shared__ float partial[32];
    float local_max = -3.0e38f;
    for (int index = threadIdx.x; index < width; index += blockDim.x)
        local_max = fmaxf(local_max, line[index]);
    for (int shift = 16; shift > 0; shift >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, shift));
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = local_max;
    __syncthreads();
    float row_max = -3.0e38f;
    for (int warp = 0; warp < (int)(blockDim.x >> 5); ++warp) row_max = fmaxf(row_max, partial[warp]);
    __syncthreads();
    float sum = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value = expf(line[index] - row_max);
        line[index] = value;
        sum += value;
    }
    sum = block_reduce_sum(sum);
    __shared__ float total;
    if (threadIdx.x == 0) total = sum;
    __syncthreads();
    const float inverse = 1.0f / total;
    for (int index = threadIdx.x; index < width; index += blockDim.x) line[index] *= inverse;
}

// out[c][r] = in[r][c] through a 32x32 shared tile.
extern "C" __global__
void diff_transpose(const float* input, float* output, const int rows, const int columns) {
    __shared__ float tile[32][33];
    const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;  // 256 threads: 32 x 8
    for (int i = ty; i < 32; i += 8) {
        const int r = r0 + i, c = c0 + tx;
        if (r < rows && c < columns) tile[i][tx] = input[(long long)r * columns + c];
    }
    __syncthreads();
    for (int i = ty; i < 32; i += 8) {
        const int c = c0 + i, r = r0 + tx;
        if (r < rows && c < columns) output[(long long)c * rows + r] = tile[tx][i];
    }
}

)FLYWEIGHT_CUDA"
R"FLYWEIGHT_CUDA(
// ---- KL autoencoder decoder ------------------------------------------------
// One image, [channels][height][width] f32.

// 2D convolution, kernel 3 with padding 1 or kernel 1, stride 1. weight is
// [out_channels][in_channels][k][k]. A block owns a 32x32 output tile of four
// output channels; each thread computes 2x2 pixels x 4 channels, staging the
// (32+2)^2 input window and the 4x9 weights per input channel in shared
// memory. The launcher's grid is 2-D, so grid.y runs over
// ceil(H/32) * ceil(out_channels/4) with the tile row varying fastest.
extern "C" __global__
void diff_conv2d(const float* input, const float* weight, const float* bias,
                 float* output, const int in_channels, const int out_channels,
                 const int height, const int width, const int kernel) {
    __shared__ float window[34][34];
    __shared__ float taps[4][9];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;   // 16 x 16 threads
    const int tile_rows = (height + 31) / 32;
    const int x0 = blockIdx.x * 32, y0 = (blockIdx.y % tile_rows) * 32;
    const int co0 = (blockIdx.y / tile_rows) * 4;
    const int px = x0 + tx * 2, py = y0 + ty * 2;
    float acc[4][2][2];
    for (int c = 0; c < 4; ++c)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) acc[c][i][j] = 0.0f;
    const int radius = kernel == 3 ? 1 : 0;
    for (int ci = 0; ci < in_channels; ++ci) {
        const float* plane = input + (long long)ci * height * width;
        for (int load = threadIdx.x; load < 34 * 34; load += 256) {
            const int wy = load / 34, wx = load % 34;
            const int sy = y0 + wy - 1, sx = x0 + wx - 1;
            window[wy][wx] = (sy >= 0 && sy < height && sx >= 0 && sx < width)
                ? plane[(long long)sy * width + sx] : 0.0f;
        }
        if (threadIdx.x < 4 * 9) {
            const int c = threadIdx.x / 9, t = threadIdx.x % 9;
            const int co = co0 + c;
            float value = 0.0f;
            if (co < out_channels) {
                if (kernel == 3) value = weight[((long long)co * in_channels + ci) * 9 + t];
                else if (t == 4) value = weight[(long long)co * in_channels + ci];
            }
            taps[c][t] = value;
        }
        __syncthreads();
        // The window index of output pixel (py + i, px + j) is (ty*2 + i + 1,
        // tx*2 + j + 1); tap (dy, dx) reads the neighbour at (+dy-1, +dx-1).
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                const int wy = ty * 2 + i + 1, wx = tx * 2 + j + 1;
                float pixels[9];
                if (radius) {
                    for (int t = 0; t < 9; ++t)
                        pixels[t] = window[wy + t / 3 - 1][wx + t % 3 - 1];
                } else {
                    for (int t = 0; t < 9; ++t) pixels[t] = 0.0f;
                    pixels[4] = window[wy][wx];
                }
                for (int c = 0; c < 4; ++c) {
                    float sum = 0.0f;
                    for (int t = 0; t < 9; ++t) sum += pixels[t] * taps[c][t];
                    acc[c][i][j] += sum;
                }
            }
        __syncthreads();
    }
    for (int c = 0; c < 4; ++c) {
        const int co = co0 + c;
        if (co >= out_channels) continue;
        float* plane = output + (long long)co * height * width;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                const int y = py + i, x = px + j;
                if (y < height && x < width)
                    plane[(long long)y * width + x] = acc[c][i][j] + (bias ? bias[co] : 0.0f);
            }
    }
}

// Tensor-core 3x3 / 1x1 convolution as an implicit GEMM: M = pixels of one
// output row (32 per warp, 128 per block), N = 64 output channels, K =
// in_channels per tap. Both operands arrive as single 64-bit loads because
// the channels inside every block of 16 are interleaved in fragment order
// (diff_channel_slot): the input is channels-last bf16 written that way by
// the group norm or the upsample feeding the conv, and the weights are
// packed [tap][out][in/16][16] the same way at load. Output is [C][H][W]
// f32 with the bias added. Needs in_channels % 16 == 0 and
// out_channels % 64 == 0 (the host keeps the f32 kernel otherwise).
// Grid: (ceil(W/128), H * out_channels/64), block 128.
//
// Slot of channel j (0..15) within its block: lane pair p = (j % 8) / 2
// holds channels {2p, 2p+1, 2p+8, 2p+9} of an m16n8k16 k-step, so those
// four go adjacent.
__device__ __forceinline__ int diff_channel_slot(int j) {
    return ((j & 7) >> 1) * 4 + (j >> 3) * 2 + (j & 1);
}

extern "C" __global__ __launch_bounds__(128)
void diff_conv2d_bf16(const unsigned short* input, const unsigned short* weight, const float* bias,
                      float* output, const int in_channels, const int out_channels,
                      const int height, const int width, const int kernel) {
    const int co_groups = out_channels / 64;
    const int y = blockIdx.y / co_groups, co0 = (blockIdx.y % co_groups) * 64;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int quad = lane >> 2, pair = lane & 3;
    const int x_base = blockIdx.x * 128 + warp * 32;
    const int radius = kernel == 3 ? 1 : 0, taps = kernel * kernel;
    const int blocks = in_channels >> 4;
    float acc[2][8][4];
    for (int m = 0; m < 2; ++m) for (int n = 0; n < 8; ++n) for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.0f;
    for (int tap = 0; tap < taps; ++tap) {
        const int dy = tap / kernel - radius, dx = tap % kernel - radius;
        const int sy = y + dy;
        const bool row_ok = sy >= 0 && sy < height;
        // Four pixel rows of the two m-tiles: quad, quad+8, quad+16, quad+24.
        const unsigned short* in_row[4];
        bool ok[4];
        for (int r = 0; r < 4; ++r) {
            const int sx = x_base + r * 8 + quad + dx;
            ok[r] = row_ok && sx >= 0 && sx < width;
            in_row[r] = input + ((long long)sy * width + sx) * in_channels + pair * 4;
        }
        const unsigned short* w_tap = weight + (long long)tap * out_channels * in_channels;
        for (int blk = 0; blk < blocks; ++blk) {
            unsigned int a[2][4];
            for (int r = 0; r < 4; ++r) {
                uint2 bits = ok[r] ? *(const uint2*)(in_row[r] + blk * 16) : make_uint2(0u, 0u);
                a[r >> 1][(r & 1)] = bits.x;        // a0 (row quad) / a1 (row quad+8)
                a[r >> 1][(r & 1) + 2] = bits.y;    // a2 / a3: channels +8
            }
            for (int n = 0; n < 8; ++n) {
                const uint2 bits = *(const uint2*)(w_tap + ((long long)(co0 + n * 8 + quad) * blocks + blk) * 16 + pair * 4);
                unsigned int b[2] = {bits.x, bits.y};
                kv_mma_m16n8k16(acc[0][n], a[0], b, (const __nv_bfloat16*)0);
                kv_mma_m16n8k16(acc[1][n], a[1], b, (const __nv_bfloat16*)0);
            }
        }
    }
    const long long plane = (long long)height * width;
    for (int n = 0; n < 8; ++n) {
        const int co = co0 + n * 8 + pair * 2;
        const float b0 = bias ? bias[co] : 0.0f, b1 = bias ? bias[co + 1] : 0.0f;
        for (int m = 0; m < 2; ++m)
            for (int half = 0; half < 2; ++half) {
                const int px = x_base + m * 16 + half * 8 + quad;
                if (px >= width) continue;
                output[(long long)co * plane + (long long)y * width + px] = acc[m][n][half * 2] + b0;
                output[(long long)(co + 1) * plane + (long long)y * width + px] = acc[m][n][half * 2 + 1] + b1;
            }
    }
}

// [C][H][W] f32 -> channels-last bf16 in slot order, for a tensor-core conv
// whose input did not come from a norm or an upsample (conv_in, shortcuts).
extern "C" __global__
void diff_to_hwc_bf16(const float* input, unsigned short* output, const int channels, const int plane) {
    const long long elements = (long long)channels * plane;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x) {
        const int c = (int)(index / plane);
        const long long pixel = index % plane;
        output[pixel * channels + (c & ~15) + diff_channel_slot(c & 15)] = diff_f32_to_bf16(input[index]);
    }
}

// Nearest 2x upsample straight into the channels-last bf16 layout the
// tensor-core conv reads: out[2h][2w][c] from in[c][h][w].
extern "C" __global__
void diff_upsample_nearest_2x_hwc_bf16(const float* input, unsigned short* output, const int channels,
                                       const int height, const int width) {
    const int out_width = width * 2, out_height = height * 2;
    const long long elements = (long long)channels * out_height * out_width;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x) {
        const int c = (int)(index % channels);
        const long long pixel = index / channels;
        const int x = (int)(pixel % out_width), y = (int)(pixel / out_width);
        output[pixel * channels + (c & ~15) + diff_channel_slot(c & 15)] =
            diff_f32_to_bf16(input[((long long)c * height + y / 2) * width + x / 2]);
    }
}

// Group norm, pass 1: per (group, slice) partial sum and sum of squares in
// double, into partials[group][slice][2]. Grid (groups, slices), block 256.
extern "C" __global__
void diff_group_norm_stats(const float* input, double* partials, const int channels,
                           const int plane, const int groups, const int slices) {
    const int group = blockIdx.x, slice = blockIdx.y;
    const int channels_per_group = channels / groups;
    const long long elements = (long long)channels_per_group * plane;
    const float* base = input + (long long)group * channels_per_group * plane;
    const long long per_slice = (elements + slices - 1) / slices;
    const long long begin = (long long)slice * per_slice;
    const long long end = min(elements, begin + per_slice);
    double sum = 0.0, square = 0.0;
    for (long long index = begin + threadIdx.x; index < end; index += blockDim.x) {
        const double value = base[index];
        sum += value;
        square += value * value;
    }
    for (int shift = 16; shift > 0; shift >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, shift);
        square += __shfl_xor_sync(0xffffffff, square, shift);
    }
    __shared__ double warp_sum[8], warp_square[8];
    if ((threadIdx.x & 31) == 0) {
        warp_sum[threadIdx.x >> 5] = sum;
        warp_square[threadIdx.x >> 5] = square;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0, total_square = 0.0;
        for (int warp = 0; warp < (int)(blockDim.x >> 5); ++warp) {
            total += warp_sum[warp];
            total_square += warp_square[warp];
        }
        partials[((long long)group * slices + slice) * 2] = total;
        partials[((long long)group * slices + slice) * 2 + 1] = total_square;
    }
}

// Group norm, pass 2: normalize with per-channel affine, optional SiLU.
// `output_hwc` non-null writes channels-last bf16 there (for the tensor-core
// conv) instead of [C][H][W] f32 to `output`.
extern "C" __global__
void diff_group_norm_apply(const float* input, const double* partials,
                           const float* weight, const float* bias, float* output,
                           unsigned short* output_hwc,
                           const int channels, const int plane, const int groups,
                           const int slices, const float epsilon, const int silu) {
    const int channels_per_group = channels / groups;
    const long long elements = (long long)channels * plane;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x) {
        const int channel = (int)(index / plane);
        const int group = channel / channels_per_group;
        double sum = 0.0, square = 0.0;
        for (int slice = 0; slice < slices; ++slice) {
            sum += partials[((long long)group * slices + slice) * 2];
            square += partials[((long long)group * slices + slice) * 2 + 1];
        }
        const double count = (double)channels_per_group * plane;
        const double mean = sum / count;
        const double variance = square / count - mean * mean;
        const float inverse = (float)(1.0 / sqrt(variance + (double)epsilon));
        float value = ((float)(input[index] - mean)) * inverse * weight[channel] + bias[channel];
        if (silu) value = value / (1.0f + expf(-value));
        if (output_hwc)
            output_hwc[(index % plane) * (long long)channels + (channel & ~15) + diff_channel_slot(channel & 15)] =
                diff_f32_to_bf16(value);
        else output[index] = value;
    }
}

// Nearest-neighbour 2x upsample: out[c][2h][2w].
extern "C" __global__
void diff_upsample_nearest_2x(const float* input, float* output, const int channels,
                              const int height, const int width) {
    const long long elements = (long long)channels * height * width * 4;
    const int out_width = width * 2;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (long long)blockDim.x * gridDim.x) {
        const int x = (int)(index % out_width);
        const long long rest = index / out_width;
        const int y = (int)(rest % (height * 2));
        const int c = (int)(rest / (height * 2));
        output[index] = input[((long long)c * height + y / 2) * width + x / 2];
    }
}

)FLYWEIGHT_CUDA";

}  // namespace flyweight::v2
