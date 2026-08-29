#pragma once

// CUDA language surface for compiling the kernel corpus as host C++.
// See flyweight_cpu_shim_geometry.hpp for the execution model.

#include <flyweight_cpu_shim_geometry.hpp>

#include <atomic>

// ---------------------------------------------------------------------------
// CUDA language surface
// ---------------------------------------------------------------------------

// Function qualifiers. __global__ kernels become ordinary host functions; the
// generated registry supplies the grid loop and argument unpacking.
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__
#define __constant__
#define __managed__

// See the header comment: one OS thread per block makes this exactly right.
#define __shared__ static thread_local

using half = flyweight::cpu::Half;
using half2 = flyweight::cpu::Half2;
using __half = flyweight::cpu::Half;
using __half2 = flyweight::cpu::Half2;
using __nv_bfloat16 = flyweight::cpu::BFloat16;
using nv_bfloat16 = flyweight::cpu::BFloat16;

#define blockIdx (::flyweight::cpu::t_block_index)
#define threadIdx (::flyweight::cpu::t_thread_index)
#define blockDim (::flyweight::cpu::t_block_dim)
#define gridDim (::flyweight::cpu::t_grid_dim)

inline void __syncthreads() { ::flyweight::cpu::t_scheduler->block_barrier(); }
inline void __threadfence_block() {}
inline void __threadfence() {}

// ---------------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------------

// The corpus uses the 128-bit vector loads to pull four packed quant words at a
// time. Layout and alignment have to match the device types, because the loads
// are reinterpret_casts over cache and weight memory.
#define FLYWEIGHT_VECTOR4(name, element)                       \
    struct alignas(16) name {                                \
        element x, y, z, w;                                  \
    };                                                       \
    inline name make_##name(element x, element y,             \
                            element z, element w) {           \
        return name{x, y, z, w};                             \
    }

FLYWEIGHT_VECTOR4(uint4, unsigned int)
FLYWEIGHT_VECTOR4(int4, int)
FLYWEIGHT_VECTOR4(float4, float)
FLYWEIGHT_VECTOR4(ulonglong2_pair, unsigned long long)
#undef FLYWEIGHT_VECTOR4

struct alignas(8) uint2 { unsigned int x, y; };
struct alignas(8) int2 { int x, y; };
struct alignas(8) float2 { float x, y; };
struct alignas(4) uchar4 { unsigned char x, y, z, w; };
struct alignas(4) char4 { signed char x, y, z, w; };
struct alignas(16) ulonglong2 { unsigned long long x, y; };

inline uint2 make_uint2(unsigned int x, unsigned int y) { return uint2{x, y}; }
inline int2 make_int2(int x, int y) { return int2{x, y}; }
inline float2 make_float2(float x, float y) { return float2{x, y}; }

// Narrow float types. Declared here rather than beside their conversions
// because __nv_fp4x2_e2m1 packs a float2, which only exists at this point.
using __nv_fp8_e4m3 = flyweight::cpu::__nv_fp8_e4m3;

struct __nv_fp4x2_e2m1 {
    // Low nibble holds .x, high nibble holds .y, matching the device type.
    unsigned char __x = 0;
    __nv_fp4x2_e2m1() = default;
    explicit __nv_fp4x2_e2m1(float2 pair)
        : __x(static_cast<unsigned char>(
              flyweight::cpu::float_to_e2m1_bits(pair.x) |
              (flyweight::cpu::float_to_e2m1_bits(pair.y) << 4))) {}
};

// ---------------------------------------------------------------------------
// Device math builtins
// ---------------------------------------------------------------------------

// CUDA puts integer min/max in the global namespace; host C++ only has the
// std:: templates, and the corpus calls the unqualified names. NOMINMAX covers
// our own windows.h include; this covers a translation unit that pulled one in
// ahead of the shim, where the macros would eat these definitions.
#if defined(_WIN32)
#  undef min
#  undef max
#endif

inline int min(int a, int b) { return a < b ? a : b; }
inline unsigned int min(unsigned int a, unsigned int b) { return a < b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline unsigned long long min(unsigned long long a, unsigned long long b) {
    return a < b ? a : b;
}
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned int max(unsigned int a, unsigned int b) { return a > b ? a : b; }
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) {
    return a > b ? a : b;
}

using std::isnan;
using std::isinf;
using std::isfinite;

inline float rsqrtf(float value) { return 1.0f / std::sqrt(value); }
inline float __frsqrt_rn(float value) { return 1.0f / std::sqrt(value); }
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }

// Read-only cache hints are pure performance annotations on device.
template <class T>
inline T __ldg(const T* address) { return *address; }

// Conversions.
inline float __half2float(flyweight::cpu::Half value) {
    return flyweight::cpu::half_bits_to_float(value.bits);
}
inline flyweight::cpu::Half __float2half(float value) { return flyweight::cpu::Half(value); }
inline float __bfloat162float(flyweight::cpu::BFloat16 value) {
    return static_cast<float>(value);
}
inline flyweight::cpu::BFloat16 __float2bfloat16(float value) {
    return flyweight::cpu::BFloat16(value);
}

inline float __uint_as_float(unsigned int bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
inline unsigned int __float_as_uint(float value) {
    unsigned int bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
inline int __float2int_rn(float value) {
    return static_cast<int>(std::nearbyintf(value));
}

inline float __expf(float value) { return std::exp(value); }
inline float __logf(float value) { return std::log(value); }
inline float __fdividef(float a, float b) { return a / b; }
inline int __popc(unsigned int value) {
    // Not std::popcount: the corpus is compiled as C++17-compatible text.
    int count = 0;
    while (value) { value &= value - 1u; ++count; }
    return count;
}

// SIMD-within-a-word intrinsics used by the integer quant kernels.
inline int __dp4a(unsigned int a, unsigned int b, int c) {
    int sum = c;
    for (int index = 0; index < 4; ++index) {
        const int lhs = static_cast<signed char>((a >> (index * 8)) & 0xffu);
        const int rhs = static_cast<signed char>((b >> (index * 8)) & 0xffu);
        sum += lhs * rhs;
    }
    return sum;
}

inline unsigned int __vadd4(unsigned int a, unsigned int b) {
    unsigned int out = 0;
    for (int index = 0; index < 4; ++index) {
        const unsigned int lhs = (a >> (index * 8)) & 0xffu;
        const unsigned int rhs = (b >> (index * 8)) & 0xffu;
        out |= ((lhs + rhs) & 0xffu) << (index * 8);
    }
    return out;
}

inline unsigned int __vsub4(unsigned int a, unsigned int b) {
    unsigned int out = 0;
    for (int index = 0; index < 4; ++index) {
        const unsigned int lhs = (a >> (index * 8)) & 0xffu;
        const unsigned int rhs = (b >> (index * 8)) & 0xffu;
        out |= ((lhs - rhs) & 0xffu) << (index * 8);
    }
    return out;
}

inline unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    unsigned int out = 0;
    for (int index = 0; index < 4; ++index) {
        const unsigned int lhs = (a >> (index * 8)) & 0xffu;
        const unsigned int rhs = (b >> (index * 8)) & 0xffu;
        out |= (lhs != rhs ? 0xffu : 0x00u) << (index * 8);
    }
    return out;
}

// Genuinely atomic, because the corpus uses these across blocks.
//
// These were once plain read-modify-write, on the reasoning that fibers in a
// block never run concurrently and separate blocks write separate slots. The
// second half is false: the lm_head argmax kernels fold every block's winner
// into a single `winners[0]`, and the backend runs blocks on all workers at
// once, so that was a race over the token being emitted.
template <class T>
inline T flyweight_atomic_max(T* address, T value) {
    auto* target = reinterpret_cast<std::atomic<T>*>(address);
    T previous = target->load(std::memory_order_relaxed);
    while (previous < value &&
           !target->compare_exchange_weak(previous, value,
                                          std::memory_order_relaxed)) {
    }
    return previous;
}

inline int atomicMax(int* address, int value) {
    return flyweight_atomic_max(address, value);
}
inline unsigned int atomicMax(unsigned int* address, unsigned int value) {
    return flyweight_atomic_max(address, value);
}
inline unsigned long long atomicMax(unsigned long long* address,
                                    unsigned long long value) {
    return flyweight_atomic_max(address, value);
}
inline long long atomicMax(long long* address, long long value) {
    return flyweight_atomic_max(address, value);
}

template <class T>
inline T flyweight_atomic_add(T* address, T value) {
    auto* target = reinterpret_cast<std::atomic<T>*>(address);
    T previous = target->load(std::memory_order_relaxed);
    while (!target->compare_exchange_weak(previous, previous + value,
                                          std::memory_order_relaxed)) {
    }
    return previous;
}

inline int atomicAdd(int* address, int value) {
    return flyweight_atomic_add(address, value);
}
inline float atomicAdd(float* address, float value) {
    return flyweight_atomic_add(address, value);
}

// ---------------------------------------------------------------------------
// Warp shuffles
// ---------------------------------------------------------------------------

namespace flyweight::cpu {

// Every shuffle in the corpus passes a full 0xffffffff mask, so the exchange is
// a warp-wide publish/read pair. Two barriers are needed: one so all lanes have
// published before anyone reads, one so nobody overwrites a slot another lane
// has yet to read.
template <class T>
inline T shuffle_exchange(T value, int source_lane) {
    static_assert(sizeof(T) <= sizeof(std::uint64_t), "unsupported shuffle width");
    auto* slots = t_scheduler->exchange();
    const unsigned int thread = t_thread_index.x;
    const unsigned int warp_base = (thread / kWarpSize) * kWarpSize;

    std::uint64_t packed = 0;
    std::memcpy(&packed, &value, sizeof(value));
    slots[thread] = packed;
    t_scheduler->warp_barrier();

    const unsigned int lane = static_cast<unsigned int>(source_lane) % kWarpSize;
    const unsigned int source = warp_base + lane;
    std::uint64_t received =
        source < t_scheduler->thread_count() ? slots[source] : packed;
    t_scheduler->warp_barrier();

    T result;
    std::memcpy(&result, &received, sizeof(result));
    return result;
}

}  // namespace flyweight::cpu

template <class T>
inline T __shfl_sync(unsigned int, T value, int source_lane, int = 32) {
    return flyweight::cpu::shuffle_exchange(value, source_lane);
}

template <class T>
inline T __shfl_down_sync(unsigned int, T value, unsigned int delta, int width = 32) {
    const int lane = static_cast<int>(flyweight::cpu::t_thread_index.x % flyweight::cpu::kWarpSize);
    const int target = lane + static_cast<int>(delta);
    // Out-of-range reads return the lane's own value, as on device.
    return flyweight::cpu::shuffle_exchange(value, target < width ? target : lane);
}

template <class T>
inline T __shfl_up_sync(unsigned int, T value, unsigned int delta, int = 32) {
    const int lane = static_cast<int>(flyweight::cpu::t_thread_index.x % flyweight::cpu::kWarpSize);
    const int target = lane - static_cast<int>(delta);
    return flyweight::cpu::shuffle_exchange(value, target >= 0 ? target : lane);
}

template <class T>
inline T __shfl_xor_sync(unsigned int, T value, int mask, int = 32) {
    const int lane = static_cast<int>(flyweight::cpu::t_thread_index.x % flyweight::cpu::kWarpSize);
    return flyweight::cpu::shuffle_exchange(value, lane ^ mask);
}

inline void __syncwarp(unsigned int = 0xffffffffu) {
    ::flyweight::cpu::t_scheduler->warp_barrier();
}
