#include "colibri_native.h"
#include "q4_kernel.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace {

constexpr std::uint32_t kFeatureAvx2 = 1u << 0;
constexpr std::uint32_t kFeatureAvx512 = 1u << 1;

// Read XCR0 without the _xgetbv intrinsic, which GCC/Clang refuse to inline
// unless the translation unit enables the "xsave" target feature. Raw XGETBV
// has no such requirement and is only reached after the OSXSAVE bit is set.
std::uint64_t read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

std::uint32_t detect_features() {
#if defined(__x86_64__) || defined(_M_X64)
    int registers[4] = {};
#if defined(_MSC_VER)
    __cpuid(registers, 1);
#else
    __cpuid(1, registers[0], registers[1], registers[2], registers[3]);
#endif
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    const bool f16c = (registers[2] & (1 << 29)) != 0;
    if (!osxsave || !avx || !f16c) {
        return 0;
    }
    const std::uint64_t xcr0 = read_xcr0();
    if ((xcr0 & 0x6) != 0x6) {
        return 0;
    }
#if defined(_MSC_VER)
    __cpuidex(registers, 7, 0);
#else
    __cpuid_count(7, 0, registers[0], registers[1], registers[2], registers[3]);
#endif
    std::uint32_t features = 0;
    if ((registers[1] & (1 << 5)) != 0) {
        features |= kFeatureAvx2;
    }
    const bool avx512_hardware = (registers[1] & (1 << 16)) != 0;
    const bool avx512_bw = (registers[1] & (1 << 30)) != 0;
    const bool avx512_os = (xcr0 & 0xE0) == 0xE0;
    if (avx512_hardware && avx512_bw && avx512_os) {
        features |= kFeatureAvx512;
    }
    return features;
#else
    return 0;
#endif
}

Q4MatvecKernel select_kernel(std::uint32_t features) {
    if ((features & kFeatureAvx512) != 0) {
        return q4_matvec_avx512;
    }
    if ((features & kFeatureAvx2) != 0) {
        return q4_matvec_avx2;
    }
    return q4_matvec_scalar;
}

const std::uint32_t kCpuFeatures = detect_features();
const Q4MatvecKernel kQ4Kernel = select_kernel(kCpuFeatures);

}

extern "C" std::uint32_t colibri_native_version() {
    return 1;
}

extern "C" std::uint32_t colibri_cpu_features() {
    return kCpuFeatures;
}

extern "C" int colibri_q4_matvec(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
) {
    if (packed == nullptr || scales == nullptr || vector == nullptr
        || output == nullptr || rows <= 0 || columns <= 0) {
        return -1;
    }
    return kQ4Kernel(packed, scales, vector, output, rows, columns);
}
