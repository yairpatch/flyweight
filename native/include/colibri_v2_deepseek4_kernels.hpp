#pragma once

// DeepSeek-V4 device kernels, compiled through NVRTC alongside the Qwen set.
//
// Kept separate from `colibri_v2_qwen_kernels.hpp` because these are reached
// only by the deepseek4 runtime; the Qwen kernels are shared and changing them
// would move another model's numbers.
inline constexpr char deepseek4_cuda_source[] = R"COLIBRI_CUDA(

// The 2048-entry IQ1_S grid is emitted ahead of this source by the runtime.
extern "C" __device__ __constant__ unsigned long long ds4_iq1s_grid[2048];

extern "C" __global__
void ds4_iq1s_matvec(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= output_size) return;
    const int blocks = input_size >> 8;
    const unsigned char* row_data =
        packed + (unsigned long long)row * blocks * 50;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const unsigned char* base = row_data + (unsigned long long)block * 50;
        const unsigned char* qs = base + 2;
        const unsigned short* qh = (const unsigned short*)(base + 34);
        const float d = __half2float(*((const __half*)base));
        #pragma unroll
        for (int group = 0; group < 8; ++group) {
            const unsigned short header = qh[group];
            const int part = lane >> 3;
            const int within = lane & 7;
            const unsigned int index = (unsigned int)qs[4 * group + part] |
                (((unsigned int)header >> (3 * part) & 7u) << 8);
            const unsigned long long entry = ds4_iq1s_grid[index];
            const signed char quant = (signed char)(entry >> (8 * within));
            const float delta = (header & 0x8000) ? -0.125f : 0.125f;
            const float scale = d * (float)(2 * ((header >> 12) & 7) + 1);
            result = fmaf(scale * ((float)quant + delta),
                          vector[block * 256 + group * 32 + lane], result);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        result += __shfl_down_sync(0xffffffff, result, offset);
    if (lane == 0) output[row] = result;
}

__device__ __forceinline__ float ds4_iq1s_value(
    const unsigned char* packed, const long long absolute
) {
    const int block = (int)(absolute >> 8);
    const int within = (int)(absolute & 255);
    const int group = within >> 5;
    const int part = (within >> 3) & 3;
    const int lane = within & 7;
    const unsigned char* base = packed + (long long)block * 50;
    const unsigned short header = *((const unsigned short*)(base + 34 + group * 2));
    const unsigned int index = (unsigned int)base[2 + 4 * group + part] |
        (((unsigned int)header >> (3 * part) & 7u) << 8);
    const signed char quant = (signed char)(ds4_iq1s_grid[index] >> (8 * lane));
    const float delta = (header & 0x8000) ? -0.125f : 0.125f;
    const float scale = __half2float(*((const __half*)base)) *
        (float)(2 * ((header >> 12) & 7) + 1);
    return scale * ((float)quant + delta);
}

extern "C" __global__
void ds4_iq1s_grouped_swiglu(
    const unsigned long long* gate_ptrs,
    const unsigned long long* up_ptrs,
    const float* vector, float* activated,
    const int input_size, const int output_size,
    const int experts, const float limit
) {
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    if (row >= output_size || expert >= experts) return;
    const unsigned char* gate = (const unsigned char*)gate_ptrs[expert];
    const unsigned char* up = (const unsigned char*)up_ptrs[expert];
    const long long base = (long long)row * input_size;
    float g = 0.0f, u = 0.0f;
    for (int input = threadIdx.x; input < input_size; input += blockDim.x) {
        const float value = vector[input];
        g += ds4_iq1s_value(gate, base + input) * value;
        u += ds4_iq1s_value(up, base + input) * value;
    }
    g = block_reduce_sum(g);
    u = block_reduce_sum(u);
    if (threadIdx.x == 0) {
        g = fminf(fmaxf(g, -limit), limit);
        u = fminf(fmaxf(u, -limit), limit);
        activated[expert * output_size + row] =
            (g / (1.0f + expf(-g))) * u;
    }
}

extern "C" __global__
void ds4_clamped_swiglu(
    const float* gate, const float* up, float* output,
    const int size, const float limit
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const float g = fminf(fmaxf(gate[index], -limit), limit);
    const float u = fminf(fmaxf(up[index], -limit), limit);
    output[index] = (g / (1.0f + expf(-g))) * u;
}

// Q8_0 matvec, four blocks of weights in flight per warp iteration.
//
// The shared kernel walks one 32-value block per warp iteration, which is 32
// bytes of weights moved per dependent load -- enough parallelism to fill the
// machine but not enough in flight to hide the latency of reaching it, and it
// measured 80 GiB/s where the card has far more. Issuing four independent
// blocks per iteration gives the memory system four requests to overlap.
//
// The layout is GGML's: a row is `input_size/32` blocks of 34 bytes, each a
// half scale followed by 32 signed bytes. Rows are contiguous, so the 34-byte
// stride is what stops the values being read four at a time -- every other
// block starts two bytes off a four-byte boundary. Lane-per-value keeps the
// reads coalesced across the warp instead, which costs nothing here.
extern "C" __global__
void ds4_q8_matvec(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= output_size) return;
    const int blocks = input_size >> 5;
    const unsigned char* base = packed + (unsigned long long)row * blocks * 34;
    float partial = 0.0f;
    int block = 0;
    for (; block + 4 <= blocks; block += 4) {
        const unsigned char* b0 = base + (unsigned long long)(block + 0) * 34;
        const unsigned char* b1 = base + (unsigned long long)(block + 1) * 34;
        const unsigned char* b2 = base + (unsigned long long)(block + 2) * 34;
        const unsigned char* b3 = base + (unsigned long long)(block + 3) * 34;
        const signed char v0 = *((const signed char*)(b0 + 2 + lane));
        const signed char v1 = *((const signed char*)(b1 + 2 + lane));
        const signed char v2 = *((const signed char*)(b2 + 2 + lane));
        const signed char v3 = *((const signed char*)(b3 + 2 + lane));
        const float s0 = __half2float(*((const __half*)b0));
        const float s1 = __half2float(*((const __half*)b1));
        const float s2 = __half2float(*((const __half*)b2));
        const float s3 = __half2float(*((const __half*)b3));
        partial += ((float)v0 * s0) * vector[(block + 0) * 32 + lane];
        partial += ((float)v1 * s1) * vector[(block + 1) * 32 + lane];
        partial += ((float)v2 * s2) * vector[(block + 2) * 32 + lane];
        partial += ((float)v3 * s3) * vector[(block + 3) * 32 + lane];
    }
    for (; block < blocks; ++block) {
        const unsigned char* b = base + (unsigned long long)block * 34;
        const signed char value = *((const signed char*)(b + 2 + lane));
        const float scale = __half2float(*((const __half*)b));
        partial += ((float)value * scale) * vector[block * 32 + lane];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}


// The grouped half of the output projection.
//
// The attention output is cut into `groups` chunks and chunk g goes through the
// g-th slice of the tensor's rows, so this is the matvec above with the vector
// offset by which slice a row belongs to. It is the largest single read in the
// attention half -- 34 MiB a layer -- and the shape is not one the shared
// kernels have, which is why it gets its own.
extern "C" __global__
void ds4_q8_grouped_matvec(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size,
    const int group_rows
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= output_size) return;
    const float* source = vector + (long long)(row / group_rows) * input_size;
    const int blocks = input_size >> 5;
    const unsigned char* base = packed + (unsigned long long)row * blocks * 34;
    float partial = 0.0f;
    int block = 0;
    for (; block + 4 <= blocks; block += 4) {
        const unsigned char* b0 = base + (unsigned long long)(block + 0) * 34;
        const unsigned char* b1 = base + (unsigned long long)(block + 1) * 34;
        const unsigned char* b2 = base + (unsigned long long)(block + 2) * 34;
        const unsigned char* b3 = base + (unsigned long long)(block + 3) * 34;
        const signed char v0 = *((const signed char*)(b0 + 2 + lane));
        const signed char v1 = *((const signed char*)(b1 + 2 + lane));
        const signed char v2 = *((const signed char*)(b2 + 2 + lane));
        const signed char v3 = *((const signed char*)(b3 + 2 + lane));
        partial += ((float)v0 * __half2float(*((const __half*)b0))) * source[(block + 0) * 32 + lane];
        partial += ((float)v1 * __half2float(*((const __half*)b1))) * source[(block + 1) * 32 + lane];
        partial += ((float)v2 * __half2float(*((const __half*)b2))) * source[(block + 2) * 32 + lane];
        partial += ((float)v3 * __half2float(*((const __half*)b3))) * source[(block + 3) * 32 + lane];
    }
    for (; block < blocks; ++block) {
        const unsigned char* b = base + (unsigned long long)block * 34;
        const signed char value = *((const signed char*)(b + 2 + lane));
        partial += ((float)value * __half2float(*((const __half*)b))) * source[block * 32 + lane];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}


// Q6_K matvec: scales read rather than shuffled, two blocks in flight.
//
// The shared kernel broadcasts each group scale across the warp with
// __shfl_sync -- eight of them per 256-value block, and a warp-wide shuffle is
// a synchronizing instruction. It measured 60 GiB/s against the CPU path's 46,
// which is why the shared expert did not want to be on the device at all.
//
// Every lane can simply load the scale it needs: sixteen bytes per block that
// every lane in a half-warp reads the same way, which is a broadcast out of L1
// rather than a warp instruction. Two blocks per iteration then gives the
// memory system something to overlap.
//
// The block is GGML's: 128 bytes of low nibbles, 64 of high bit-pairs, 16
// signed group scales, then the half multiplier -- 210 bytes for 256 values.
__device__ __forceinline__ float ds4_q6k_block(
    const unsigned char* base, const float* vector, const int lane
) {
    const signed char* scales = (const signed char*)(base + 192);
    const int scale_group = lane >> 4;
    const unsigned char low_0 = base[lane];
    const unsigned char low_1 = base[32 + lane];
    const unsigned char low_2 = base[64 + lane];
    const unsigned char low_3 = base[96 + lane];
    const unsigned char high_0 = base[128 + lane];
    const unsigned char high_1 = base[160 + lane];
    float partial = 0.0f;
    #pragma unroll
    for (int segment = 0; segment < 4; ++segment) {
        const unsigned char low = (segment & 1) ? low_1 : low_0;
        const int nibble = segment < 2 ? (low & 15) : (low >> 4);
        const int quant = (nibble | (((high_0 >> (segment * 2)) & 3) << 4)) - 32;
        partial = fmaf((float)((int)scales[scale_group + segment * 2] * quant),
                       vector[segment * 32 + lane], partial);
    }
    #pragma unroll
    for (int segment = 0; segment < 4; ++segment) {
        const unsigned char low = (segment & 1) ? low_3 : low_2;
        const int nibble = segment < 2 ? (low & 15) : (low >> 4);
        const int quant = (nibble | (((high_1 >> (segment * 2)) & 3) << 4)) - 32;
        partial = fmaf((float)((int)scales[8 + scale_group + segment * 2] * quant),
                       vector[128 + segment * 32 + lane], partial);
    }
    return partial * __half2float(*((const __half*)(base + 208)));
}

extern "C" __global__
void ds4_q6k_matvec(
    const unsigned char* packed,
    const float* vector,
    float* output,
    const int input_size,
    const int output_size
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= output_size) return;
    // A row that is not a whole number of blocks is not something this
    // checkpoint has; refusing it is better than addressing past the row.
    if (input_size & 255) {
        if (lane == 0) output[row] = 0.0f;
        return;
    }
    const int blocks = input_size >> 8;
    const unsigned char* base = packed + (unsigned long long)row * blocks * 210ull;
    float partial = 0.0f;
    int block = 0;
    for (; block + 2 <= blocks; block += 2) {
        partial += ds4_q6k_block(base + (unsigned long long)(block + 0) * 210,
                                 vector + (block + 0) * 256, lane);
        partial += ds4_q6k_block(base + (unsigned long long)(block + 1) * 210,
                                 vector + (block + 1) * 256, lane);
    }
    for (; block < blocks; ++block)
        partial += ds4_q6k_block(base + (unsigned long long)block * 210,
                                 vector + block * 256, lane);
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if (lane == 0) output[row] = partial;
}

)COLIBRI_CUDA";
