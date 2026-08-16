#pragma once

// Launch geometry, block scheduling, and narrow float conversions for the
// host kernel backend.
//
// Split out of colibri_cpu_shim.hpp so translation units that only drive
// kernels -- cpu_backend.cpp above all -- can use the scheduler without
// pulling in the CUDA language macros, which redefine min/max and blank out
// __global__ for the whole file.
//
// Original notes on the execution model follow.
//
// Host execution environment for the CUDA kernel corpus.
//
// The corpus in colibri_v2_qwen_kernels.hpp / colibri_v2_native_kernels.hpp is
// the source of truth for what the runtime computes, and it is launched by
// string name through colibri_gpu_launch_named. That makes a second, host-side
// implementation of the same names a drop-in backend: the layer loop in
// v2_runtime.cpp never learns which one it is talking to.
//
// Rather than hand-port 126 kernels -- which would fork the numerics and let
// the two paths drift silently -- this header compiles the *same* CUDA text as
// C++ by supplying the language surface it uses. The corpus needs a small,
// closed set: no inline asm, no atomicAdd, no ballot, no cooperative groups.
//
// Execution model
// ---------------
// One block at a time per OS thread; the block's threads are ucontext fibers
// (Windows: native fibers) scheduled round-robin on that thread. Fibers are
// what make __syncthreads() expressible at all -- a barrier has to suspend a
// thread mid-kernel and resume it later, which a plain function call cannot do.
//
// Two consequences worth stating, because the rest of the file depends on them:
//
//   * __shared__ becomes `static thread_local`. All fibers of a block run on
//     one OS thread, and blocks run serially on that thread, so a thread-local
//     is shared by exactly the threads that should share it and by no others.
//     No source rewriting needed. CUDA leaves __shared__ uninitialized at each
//     launch and every kernel here writes before it reads, so the fact that a
//     thread_local persists across launches is not observable.
//
//   * Barriers count only *live* fibers. Kernels here routinely `return` early
//     on out-of-range lanes and then have surviving lanes hit __syncthreads().
//     On real hardware that works because the exited lanes stop counting; the
//     same rule is reproduced here.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
// windows.h defines min and max as function-like macros, which mangles the
// unqualified min/max overloads the corpus calls (see colibri_cpu_shim.hpp).
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <ucontext.h>
#endif

namespace colibri::cpu {

constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
// Launch geometry
// ---------------------------------------------------------------------------

struct Dim3 {
    unsigned int x = 1, y = 1, z = 1;
};

// The corpus only ever indexes .x and .y, but the full triple is kept so the
// declarations read like the CUDA ones.
extern thread_local Dim3 t_thread_index;
extern thread_local Dim3 t_block_index;
extern thread_local Dim3 t_block_dim;
extern thread_local Dim3 t_grid_dim;

// ---------------------------------------------------------------------------
// Fiber scheduling
// ---------------------------------------------------------------------------

// Per-block scheduler. Owns one fiber per thread and round-robins between them,
// switching only at barriers, so a kernel with no __syncthreads() and no shuffle
// runs each thread straight through with zero context switches.
class BlockScheduler {
public:
    void run(unsigned int threads, unsigned int shared_bytes,
             void (*body)(void*), void* payload);

    // Sets this scheduler up as the current one and sizes its dynamic shared
    // block, without creating fibers. For kernels the generator proved cannot
    // reach a barrier, the caller then runs the threads as a plain loop; they
    // still need `extern __shared__` to resolve to something.
    void prepare_direct(unsigned int threads, unsigned int shared_bytes);

    // Suspends the calling fiber until every live fiber in the block has
    // arrived. Called by __syncthreads().
    void block_barrier();

    // Suspends the calling fiber until every live fiber of *its warp* has
    // arrived. Shuffles need this rather than a block barrier: warp-local
    // reductions guarded by `if (warp == 0)` are common in the corpus, and a
    // block-wide barrier inside one would deadlock the other warps.
    void warp_barrier();

    // Called by the trampoline when a kernel returns. Drops the lane from
    // barrier accounting, which can itself complete a barrier the surviving
    // lanes are parked on.
    void retire(unsigned int thread);

    void return_to_scheduler();

    unsigned char* dynamic_shared() { return shared_.data(); }

    // Scratch for emulating register exchange between lanes. Indexed by
    // flattened thread id; sized once per block.
    std::uint64_t* exchange() { return exchange_.data(); }

    unsigned int thread_count() const { return thread_count_; }

private:
    enum class Waiting : unsigned char { none, block, warp };

    struct Fiber {
        bool alive = false;
        // Release is explicit rather than "resume anything parked". A drain
        // that resumed any waiter would let a warp barrier through that only
        // some lanes had reached, which is exactly the case the warp/block
        // split exists to keep straight.
        bool releasable = false;
        Waiting waiting = Waiting::none;
#if defined(_WIN32)
        void* handle = nullptr;
#else
        ucontext_t context{};
        std::vector<char> stack;
#endif
    };

    void park(Waiting reason);
    void release_block_waiters();
    void release_warp_waiters(unsigned int warp);
    void settle_barriers();

    std::vector<Fiber> fibers_;
    std::vector<unsigned char> shared_;
    std::vector<std::uint64_t> exchange_;
    std::vector<unsigned int> warp_arrived_;
    std::vector<unsigned int> warp_live_;
    unsigned int thread_count_ = 0;
    unsigned int block_arrived_ = 0;
    unsigned int block_live_ = 0;
    unsigned int current_ = 0;
#if defined(_WIN32)
    void* scheduler_handle_ = nullptr;
#else
    ucontext_t scheduler_context_{};
#endif
};

extern thread_local BlockScheduler* t_scheduler;

// Monotonic block counter, bumped once per block launch.
//
// __shared__ maps to `static thread_local`, which is correct for sharing within
// a block but wrong across them: blocks run sequentially on a worker and would
// otherwise inherit the previous block's shared memory. CUDA gives each block
// uninitialized shared memory, and parts of the corpus rely on that being
// benign -- block_reduce_sum declares warp_sums[8] but only writes the warps
// that exist, then reads all eight, so a 128-thread launch reads four slots it
// never wrote. On a GPU those are whatever the SM had; here they were the last
// block's real values, which is far more damaging.
extern thread_local std::uint64_t t_block_generation;

// Zeroes `storage` the first time it is reached in a given block. Emitted after
// every __shared__ declaration by generate_cpu_kernels.py.
void shared_zero_once(void* storage, std::size_t bytes);

// ---------------------------------------------------------------------------
// Half and bfloat16
// ---------------------------------------------------------------------------

// Storage-only types. The corpus converts to float for every arithmetic op
// except the handful of __half2 loads, so a bit-exact conversion pair is all
// that is required to match device results.
inline float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    std::uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            // Subnormal: renormalize into a float exponent.
            std::uint32_t shift = 0;
            std::uint32_t value = mantissa;
            while ((value & 0x400u) == 0) { value <<= 1; ++shift; }
            value &= 0x3ffu;
            out = sign | ((127u - 15u - shift + 1u) << 23) | (value << 13);
        }
    } else if (exponent == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

inline std::uint16_t float_to_half_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    }
    if (exponent >= 0x1f) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        // Round-to-nearest-even into the subnormal range.
        const std::uint32_t full = mantissa | 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
        const std::uint32_t rounded =
            (full + (1u << (shift - 1)) - 1u + ((full >> shift) & 1u)) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded =
        (mantissa + 0x00000fffu + ((mantissa >> 13) & 1u)) >> 13;
    // A mantissa carry bumps the exponent, which the addition below absorbs.
    return static_cast<std::uint16_t>(
        sign | ((static_cast<std::uint32_t>(exponent) << 10) + rounded));
}

struct alignas(2) Half {
    std::uint16_t bits = 0;
    Half() = default;
    Half(float value) : bits(float_to_half_bits(value)) {}
    explicit operator float() const { return half_bits_to_float(bits); }
};

struct alignas(4) Half2 {
    Half x, y;
};

// --- OCP FP8 E4M3 and FP4 E2M1 ------------------------------------------
//
// Used by the NVFP4 repack kernels, which write block scales as E4M3 and pairs
// of weights as packed E2M1. Both formats are tiny, so encoding is a nearest
// search over the decode table with ties resolved to an even mantissa. That is
// bit-exact against the hardware converters and obviously correct; the repack
// path is not hot enough to justify the bit-twiddling version.

// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. No infinities; 0x7f is NaN.
inline float e4m3_bits_to_float(unsigned char bits) {
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const unsigned int exponent = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u)
        return sign * std::numeric_limits<float>::quiet_NaN();
    if (exponent == 0)  // subnormal: 2^-6 * (m/8)
        return sign * std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                             static_cast<int>(exponent) - 7);
}

inline unsigned char float_to_e4m3_bits(float value) {
    if (std::isnan(value)) return 0x7fu;
    const unsigned char sign = std::signbit(value) ? 0x80u : 0x00u;
    const float magnitude = std::fabs(value);
    if (magnitude >= 448.0f) return static_cast<unsigned char>(sign | 0x7eu);

    unsigned char best = 0;
    float best_error = std::fabs(magnitude - e4m3_bits_to_float(0));
    for (unsigned int code = 1; code <= 0x7eu; ++code) {
        const float candidate = e4m3_bits_to_float(static_cast<unsigned char>(code));
        const float error = std::fabs(magnitude - candidate);
        if (error < best_error ||
            (error == best_error && (code & 1u) == 0u)) {  // tie -> even mantissa
            best_error = error;
            best = static_cast<unsigned char>(code);
        }
    }
    return static_cast<unsigned char>(sign | best);
}

struct __nv_fp8_e4m3 {
    unsigned char __x = 0;
    __nv_fp8_e4m3() = default;
    explicit __nv_fp8_e4m3(float value) : __x(float_to_e4m3_bits(value)) {}
    explicit operator float() const { return e4m3_bits_to_float(__x); }
};

// E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa -> {0, .5, 1, 1.5, 2, 3, 4, 6}.
inline float e2m1_bits_to_float(unsigned char code) {
    static constexpr float kMagnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float magnitude = kMagnitudes[code & 0x07u];
    return (code & 0x08u) ? -magnitude : magnitude;
}

inline unsigned char float_to_e2m1_bits(float value) {
    const unsigned char sign = std::signbit(value) ? 0x08u : 0x00u;
    const float magnitude = std::isnan(value) ? 0.0f : std::fabs(value);
    unsigned char best = 0;
    float best_error = magnitude;
    for (unsigned int code = 1; code < 8u; ++code) {
        const float candidate = e2m1_bits_to_float(static_cast<unsigned char>(code));
        const float error = std::fabs(magnitude - candidate);
        if (error < best_error || (error == best_error && (code & 1u) == 0u)) {
            best_error = error;
            best = static_cast<unsigned char>(code);
        }
    }
    return static_cast<unsigned char>(sign | best);
}

struct alignas(2) BFloat16 {
    std::uint16_t bits = 0;
    BFloat16() = default;
    BFloat16(float value) {
        std::uint32_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        // Round-to-nearest-even, matching __float2bfloat16.
        const std::uint32_t rounding = 0x7fffu + ((raw >> 16) & 1u);
        bits = static_cast<std::uint16_t>((raw + rounding) >> 16);
    }
    explicit operator float() const {
        const std::uint32_t raw = static_cast<std::uint32_t>(bits) << 16;
        float result;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }
};

}  // namespace colibri::cpu
