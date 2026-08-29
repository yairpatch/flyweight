// The one translation unit that compiles the K-quant encoders.
//
// Two jobs. It owns `-ffp-contract=off` (set on this file in CMakeLists.txt) so
// the flag does not have to be imposed on every consumer of the packers; if it
// is ever compiled without that flag, quantization output changes silently, and
// differently on machines with and without FMA. And it installs the AVX2 scale
// search when the CPU has AVX2, since the library itself is built for baseline
// x86-64.

#include "qwen_kquant_pack_api.hpp"

#include "qwen_iq_pack.h"
#include "qwen_kquant_pack.h"

#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace qwen_kpack {

#if defined(__x86_64__) || defined(_M_X64)
void fit_sub_block_scales_avx2_entry(const float* x, int nmax, float rmin,
                                     int nstep, float* scales, float* mins);
#endif

namespace {

// AVX2 plus OS support for the YMM state. The kernels here use no FMA and no
// f16c, so those are not required.
bool avx2_usable() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 1);
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#endif
#else
    return false;
#endif
}

// Installed before main, which is early enough: nothing packs during static
// initialization, and the hook is only read inside fit_qk_super_block.
const bool installed = [] {
#if defined(__x86_64__) || defined(_M_X64)
    // FLYWEIGHT_CPU_BACKEND=scalar forces the reference path, which is what makes
    // the two comparable on one machine.
    const char* backend = std::getenv("FLYWEIGHT_CPU_BACKEND");
    const bool forced_scalar = backend && std::strcmp(backend, "scalar") == 0;
    if (!forced_scalar && avx2_usable())
        fit_sub_block_scales_hook = &fit_sub_block_scales_avx2_entry;
#endif
    return true;
}();

}  // namespace

void pack_q8_0(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_q8_0(values, count, out);
}

void pack_q2_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_q2_k(values, count, out);
}

void pack_q3_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_q3_k(values, count, out);
}

void pack_iq3_xxs(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_iq3_xxs(values, count, out);
}

void pack_iq3_xxs(const float* values, std::uint64_t count, std::uint8_t* out,
                  const float* importance, std::uint64_t row,
                  std::uint64_t chunk, std::uint64_t element_begin) {
    if (!importance || !row) {
        qwen_pack_iq3_xxs(values, count, out);
        return;
    }
    const Iq3Importance weights{importance, row, chunk, element_begin};
    qwen_pack_iq3_xxs(values, count, out, &weights);
}

void pack_iq4_xs(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_iq4_xs(values, count, out);
}

void pack_iq2_xs(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_iq2_xs(values, count, out);
}

void pack_iq2_xs(const float* values, std::uint64_t count, std::uint8_t* out,
                 const float* importance, std::uint64_t row,
                 std::uint64_t chunk, std::uint64_t element_begin) {
    if (!importance || !row) {
        qwen_pack_iq2_xs(values, count, out);
        return;
    }
    const Iq3Importance weights{importance, row, chunk, element_begin};
    qwen_pack_iq2_xs(values, count, out, &weights);
}

void pack_iq4_xs(const float* values, std::uint64_t count, std::uint8_t* out,
                 const float* importance, std::uint64_t row,
                 std::uint64_t chunk, std::uint64_t element_begin) {
    if (!importance || !row) {
        qwen_pack_iq4_xs(values, count, out);
        return;
    }
    const Iq3Importance weights{importance, row, chunk, element_begin};
    qwen_pack_iq4_xs(values, count, out, &weights);
}

void pack_q4_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_q4_k(values, count, out);
}

void pack_q5_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_q5_k(values, count, out);
}

void pack_q6_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    qwen_pack_q6_k(values, count, out);
}

bool pack_uses_avx2() { return fit_sub_block_scales_hook != nullptr; }

}  // namespace qwen_kpack
