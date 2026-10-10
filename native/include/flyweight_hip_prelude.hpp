#pragma once

// The CUDA language surface the kernel corpus uses, spelled for HIP.
//
// hipRTC compiles the same corpus NVRTC does; flyweight_gpu_compile prepends
// this text when the ROCm backend is active. The corpus keeps its CUDA
// spelling -- one source for NVRTC, hipRTC and the CPU backend's generated
// copy -- and everything that differs is translated here, in the same spirit
// as flyweight_cpu_shim.hpp for the host.
//
// Only RDNA (gfx10.3 and later) is supported, and only in wave32 mode, which
// is what HIP compiles RDNA kernels for by default: the corpus hardcodes a
// 32-lane warp in its lane arithmetic and its launch geometry. The driver
// refuses a device that reports any other wavefront size.
//
// The inline-PTX tensor-core primitives stay behind their __CUDA_ARCH__
// guards, which HIP never defines: their emulation paths compile, and the
// host keeps them off the hot path because a ROCm device reports compute
// capability 0.0 (see gpu_probe in v2_runtime.cpp).

namespace flyweight::v2 {
inline constexpr char hip_prelude_source[] = R"FLYWEIGHT_HIP(
#define FLYWEIGHT_HIP 1

// NVRTC predeclares these; hipRTC's built-in headers do not.
typedef __UINTPTR_TYPE__ uintptr_t;
typedef __INTPTR_TYPE__ intptr_t;

// __half and its conversions are built into hipRTC. bf16 is not -- hip_bf16.h
// would need the ROCm development headers at run time -- and the corpus only
// stores it and converts it, so it is spelled out here like the CPU shim's.
struct __attribute__((aligned(2))) flyweight_bfloat16 {
    unsigned short __x;
    __device__ flyweight_bfloat16() = default;
    // Round to nearest even, NaN to the canonical quiet NaN: __float2bfloat16.
    __device__ flyweight_bfloat16(float value) {
        const unsigned int raw = __float_as_uint(value);
        __x = (value != value) ? (unsigned short)0x7fff
            : (unsigned short)((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
    }
    __device__ explicit operator float() const { return __uint_as_float((unsigned int)__x << 16); }
};
typedef flyweight_bfloat16 __nv_bfloat16;
typedef flyweight_bfloat16 nv_bfloat16;
__device__ __forceinline__ float __bfloat162float(__nv_bfloat16 value) { return (float)value; }
__device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float value) { return __nv_bfloat16(value); }

// Every shuffle in the corpus names the full 32-lane mask. HIP's *_sync
// spellings want a 64-bit mask and an opt-in macro, and wave32 has nothing
// to mask anyway, so the mask is dropped.
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)

// A wavefront executes in lockstep, so the warp barrier only has to order
// shared-memory traffic between its lanes.
__device__ __forceinline__ void flyweight_syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
#define __syncwarp(...) flyweight_syncwarp()

// Signed 4x8-bit dot product with accumulate. RDNA2 has v_dot4_i32_i8; RDNA3
// and RDNA4 replaced it with the mixed-sign v_dot4_i32_iu8, which with both
// sign flags set is the same operation.
__device__ __forceinline__ int flyweight_dp4a(int a, int b, int c) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    #pragma unroll
    for (int byte = 0; byte < 4; ++byte)
        c += (int)(signed char)(a >> (8 * byte)) * (int)(signed char)(b >> (8 * byte));
    return c;
#endif
}
#define __dp4a flyweight_dp4a

// Per-byte wrapping add, subtract and inequality mask, as SWAR.
__device__ __forceinline__ unsigned int flyweight_vadd4(unsigned int a, unsigned int b) {
    return ((a & 0x7f7f7f7fu) + (b & 0x7f7f7f7fu)) ^ ((a ^ b) & 0x80808080u);
}
__device__ __forceinline__ unsigned int flyweight_vsub4(unsigned int a, unsigned int b) {
    return ((a | 0x80808080u) - (b & 0x7f7f7f7fu)) ^ ((a ^ ~b) & 0x80808080u);
}
__device__ __forceinline__ unsigned int flyweight_vcmpne4(unsigned int a, unsigned int b) {
    const unsigned int x = a ^ b;
    const unsigned int nonzero = (((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u;
    return (nonzero >> 7) * 0xffu;
}
#define __vadd4 flyweight_vadd4
#define __vsub4 flyweight_vsub4
#define __vcmpne4 flyweight_vcmpne4

// OCP FP8 E4M3 with the conversion __nv_fp8_e4m3(float) performs: round to
// nearest even, saturate to +-448, NaN to 0x7f. Only the NVFP4 block-scale
// quantizers use it.
__device__ __forceinline__ unsigned char flyweight_fp8_e4m3_encode(float value) {
    const unsigned char sign = (__float_as_uint(value) >> 31) ? 0x80 : 0x00;
    const float a = fabsf(value);
    if (a != a) return 0x7f;
    if (a >= 448.0f) return (unsigned char)(sign | 0x7e);
    if (a < 0.015625f)  // below 2^-6: subnormal steps of 2^-9
        return (unsigned char)(sign | (unsigned int)rintf(a * 512.0f));
    int exponent;
    frexpf(a, &exponent);
    exponent -= 1;  // a = 1.m * 2^exponent
    int mantissa = (int)rintf((ldexpf(a, -exponent) - 1.0f) * 8.0f);
    if (mantissa == 8) { mantissa = 0; ++exponent; }
    unsigned int code = (unsigned int)((exponent + 7) << 3 | mantissa);
    if (code > 0x7e) code = 0x7e;
    return (unsigned char)(sign | code);
}
__device__ __forceinline__ float flyweight_fp8_e4m3_decode(unsigned char code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    float magnitude;
    if ((code & 0x7f) == 0x7f) magnitude = __uint_as_float(0x7fc00000u);
    else if (exponent == 0) magnitude = ldexpf((float)mantissa, -9);
    else magnitude = ldexpf(1.0f + (float)mantissa * 0.125f, exponent - 7);
    return (code & 0x80) ? -magnitude : magnitude;
}
struct __nv_fp8_e4m3 {
    unsigned char __x;
    __device__ __nv_fp8_e4m3() : __x(0) {}
    __device__ explicit __nv_fp8_e4m3(float value) : __x(flyweight_fp8_e4m3_encode(value)) {}
    __device__ explicit operator float() const { return flyweight_fp8_e4m3_decode(__x); }
};

// cub::BlockRadixSort, the corpus's one cub facility: a block-wide descending
// sort of ITEMS_PER_THREAD keys per thread in blocked arrangement. A bitonic
// network over shared memory, tie-broken on the original position so equal
// keys keep their order exactly as cub's stable radix sort does.
namespace cub {
template <class KeyT, int BLOCK_THREADS, int ITEMS_PER_THREAD, class ValueT>
class BlockRadixSort {
    static constexpr int kTotal = BLOCK_THREADS * ITEMS_PER_THREAD;
    static_assert((kTotal & (kTotal - 1)) == 0, "bitonic sort needs a power-of-two item count");

public:
    struct TempStorage {
        KeyT keys[kTotal];
        ValueT values[kTotal];
        int order[kTotal];
    };

    __device__ explicit BlockRadixSort(TempStorage& storage) : storage_(storage) {}

    __device__ void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                                   ValueT (&values)[ITEMS_PER_THREAD]) {
        const int thread = (int)threadIdx.x;
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const int slot = thread * ITEMS_PER_THREAD + item;
            storage_.keys[slot] = keys[item];
            storage_.values[slot] = values[item];
            storage_.order[slot] = slot;
        }
        __syncthreads();
        for (int size = 2; size <= kTotal; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int pair = thread; pair < kTotal / 2; pair += BLOCK_THREADS) {
                    const int low = 2 * pair - (pair & (stride - 1));
                    const int high = low + stride;
                    const KeyT low_key = storage_.keys[low], high_key = storage_.keys[high];
                    const int low_order = storage_.order[low], high_order = storage_.order[high];
                    // "first" is the element that belongs earlier in a
                    // descending, stable order.
                    const bool high_first = high_key > low_key
                        || (high_key == low_key && high_order < low_order);
                    const bool descending_run = (low & size) == 0;
                    if (high_first == descending_run) {
                        storage_.keys[low] = high_key;
                        storage_.keys[high] = low_key;
                        storage_.order[low] = high_order;
                        storage_.order[high] = low_order;
                        const ValueT value = storage_.values[low];
                        storage_.values[low] = storage_.values[high];
                        storage_.values[high] = value;
                    }
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const int slot = thread * ITEMS_PER_THREAD + item;
            keys[item] = storage_.keys[slot];
            values[item] = storage_.values[slot];
        }
        __syncthreads();
    }

private:
    TempStorage& storage_;
};
}  // namespace cub
)FLYWEIGHT_HIP";
}  // namespace flyweight::v2
