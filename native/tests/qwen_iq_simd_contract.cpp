// Vectorized IQ dot products against the scalar reference.
//
// The IQ codebook formats decode a branch per weight in scalar form, which made
// low-bit MoE decode compute-bound. The AVX2 kernels replace that with a sign
// mask and a widening load, and they reassociate the group scale out of the
// inner accumulation, so they are not bit-identical to the scalar path -- but
// they must agree to float rounding, and any layout mistake shows up far above
// that. Random blocks cover every grid entry and sign pattern in aggregate.

#include "qwen_cpu_kernel.h"
#include "qwen_kquant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Detected locally rather than by linking the dispatch translation unit, which
// would drag in the whole q4 kernel chain for one predicate.
bool avx2_available() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool avx512_available() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

namespace {

int failures = 0;

void expect_close(const char* label, float actual, float expected, float scale) {
    // The vectorized kernels hoist the group scale out of the inner
    // accumulation, so they sum in a different order than the scalar path and
    // are not bit-identical. Tolerance rides on the accumulated magnitude
    // rather than the result: a dot of thousands of terms can cancel to near
    // zero while its terms are large, and a relative test would be meaningless
    // there. A layout error lands orders of magnitude outside this.
    const float tolerance = 1e-5f * (scale > 1.0f ? scale : 1.0f);
    if (std::fabs(actual - expected) <= tolerance) return;
    std::fprintf(
        stderr, "%s: vectorized %.9g vs scalar %.9g (tolerance %.3g)\n",
        label, actual, expected, tolerance);
    ++failures;
}

void check(const char* label, std::uint32_t type, std::uint32_t block_bytes,
           float (*scalar)(const std::uint8_t*, const float*, int, std::uint64_t)) {
    constexpr int kElements = 3072;   // one real expert row width
    constexpr int kRows = 6;
    std::mt19937 generator(1234u + type);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> activation(0.0f, 1.0f);

    const std::size_t blocks = kElements / 256;
    std::vector<std::uint8_t> packed(blocks * block_bytes * kRows);
    for (auto& value : packed) value = static_cast<std::uint8_t>(byte(generator));
    // Random bytes in the block scale would be an arbitrary fp16, including the
    // all-ones exponents that mean Inf and NaN. Everything else stays random so
    // grid indices and sign patterns are covered exhaustively in aggregate.
    const std::uint16_t block_scale = 0x2E66;  // ~0.1 in fp16
    for (int row = 0; row < kRows; ++row)
        for (std::size_t block = 0; block < blocks; ++block)
            std::memcpy(
                packed.data() + (row * blocks + block) * block_bytes,
                &block_scale, sizeof(block_scale));
    std::vector<float> input(kElements);
    for (auto& value : input) value = activation(generator);
    std::vector<float> second_input(input.rbegin(), input.rend());
    // Bound on the accumulated magnitude: every IQ magnitude is under 256 and
    // the group scale under 8, which is what the tolerance is measured against.
    float scale = 0.0f;
    for (const float value : input) scale += std::fabs(value);
    scale *= 0.1f * 256.0f * 8.0f;

    for (int row = 0; row < kRows; ++row) {
        const float expected = scalar(packed.data(), input.data(), kElements, row);
        const float actual = qwen_quant_dot_avx2(
            packed.data(), type, input.data(), kElements, row);
        char name[64];
        std::snprintf(name, sizeof(name), "%s row %d", label, row);
        expect_close(name, actual, expected, scale);
        if (type == 17 && avx512_available()) {
            const float wide = qwen_quant_dot_avx512(
                packed.data(), type, input.data(), kElements, row);
            std::snprintf(name, sizeof(name), "%s AVX-512 row %d", label, row);
            expect_close(name, wide, expected, scale);
            const int other_row = (row + 1) % kRows;
            const float other_expected = scalar(
                packed.data(), input.data(), kElements, other_row);
            const std::size_t row_bytes = blocks * block_bytes;
            float pair_first = 0.0f, pair_second = 0.0f;
            qwen_quant_dot_two_rows_avx512(
                packed.data() + row * row_bytes,
                packed.data() + other_row * row_bytes,
                type, input.data(), kElements, &pair_first, &pair_second);
            std::snprintf(name, sizeof(name), "%s AVX-512 fused first %d", label, row);
            expect_close(name, pair_first, expected, scale);
            std::snprintf(name, sizeof(name), "%s AVX-512 fused second %d", label, row);
            expect_close(name, pair_second, other_expected, scale);
            const float second_expected = scalar(
                packed.data(), second_input.data(), kElements, row);
            qwen_quant_dot_pair_avx512(
                packed.data(), type, input.data(), second_input.data(),
                kElements, row, &pair_first, &pair_second);
            std::snprintf(name, sizeof(name), "%s AVX-512 input pair first %d", label, row);
            expect_close(name, pair_first, expected, scale);
            std::snprintf(name, sizeof(name), "%s AVX-512 input pair second %d", label, row);
            expect_close(name, pair_second, second_expected, scale);
        }
    }
}

} // namespace

int main() {
    if (!avx2_available()) {
        std::fprintf(stderr, "AVX2 unavailable; skipping IQ SIMD contract\n");
        return 0;
    }
    check("iq2xs", 17, kIq2xsBlockBytes, qwen_iq2xs_dot_row);
    check("iq3xxs", 18, kIq3xxsBlockBytes, qwen_iq3xxs_dot_row);
    check("iq4xs", 23, kIq4xsBlockBytes, qwen_iq4xs_dot_row);
    if (failures) {
        std::fprintf(stderr, "%d IQ SIMD mismatches\n", failures);
        return 1;
    }
    std::printf("IQ SIMD contract OK\n");
    return 0;
}
