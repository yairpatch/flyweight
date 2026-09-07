// Contract for the AVX512-VNNI IQ1_S x Q8_K dot.
//
// Two references. The tight one feeds the scalar IQ1_S decoder the *dequantized*
// int8 activation, so the only difference left is float summation order and
// the kernel must agree to rounding. The loose one is the float-activation AVX2
// kernel the new path replaces, which bounds the int8 activation rounding the
// runtime accepts on the Q8_K expert path -- a regression that "roughly
// matches" the old kernel while decoding the wrong grid octet fails the first.
//
// Random bytes are legal IQ1_S blocks everywhere but the super-block scale, an
// f16 whose random exponent would put blocks at 1e4 and 1e-5 and drown the
// comparison in cancellation. It is pinned to 1.0; qs, qh (grid high bits,
// group scale and delta sign) stay random and cover the whole codebook.
#include "qwen_cpu_kernel.h"
#include "qwen_kquant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace {

bool has_avx512_vnni() {
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuid(registers, 0);
    if (registers[0] < 7) return false;
    __cpuidex(registers, 7, 0);
    const bool f = (registers[1] & (1 << 16)) != 0;
    const bool bw = (registers[1] & (1 << 30)) != 0;
    const bool vl = (registers[1] & (1 << 31)) != 0;
    const bool vnni = (registers[2] & (1 << 11)) != 0;
    return f && bw && vl && vnni && (_xgetbv(0) & 0xE6) == 0xE6;
#else
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
#endif
}

}  // namespace

int main() {
    if (!has_avx512_vnni()) {
        std::printf("qwen_iq1s_vnni512_contract: no AVX512-VNNI, skipped\n");
        return 0;
    }
    constexpr int kElements = 2560;   // qwen4exp's expert width, ten super-blocks
    constexpr int kRows = 64;
    constexpr int kBlocks = kElements / 256;
    constexpr std::uint64_t kRowBytes = static_cast<std::uint64_t>(kBlocks) * kIq1sBlockBytes;
    std::mt19937 generator(7);
    std::vector<std::uint8_t> packed(kRowBytes * kRows);
    for (auto& byte : packed) byte = static_cast<std::uint8_t>(generator());
    for (int row = 0; row < kRows; ++row)
        for (int block = 0; block < kBlocks; ++block) {
            auto* base = packed.data() + row * kRowBytes + block * kIq1sBlockBytes;
            base[0] = 0x00; base[1] = 0x3c;  // f16 1.0
        }
    std::uniform_real_distribution<float> spread(-1.0f, 1.0f);
    std::vector<float> input(kElements);
    for (auto& value : input) value = spread(generator) * spread(generator);
    std::vector<QwenQ8KBlock> quantized(kBlocks);
    qwen_quantize_q8_k_avx2(input.data(), kElements, quantized.data());
    std::vector<float> dequantized(kElements);
    for (int block = 0; block < kBlocks; ++block)
        for (int lane = 0; lane < 256; ++lane)
            dequantized[block * 256 + lane] =
                quantized[block].scale * static_cast<float>(quantized[block].values[lane]);

    int failures = 0;
    double worst_tight = 0.0, worst_loose = 0.0;
    for (int row = 0; row < kRows; ++row) {
        const auto* row_data = packed.data() + row * kRowBytes;
        double exact = 0.0, magnitude = 0.0;
        for (int index = 0; index < kElements; ++index) {
            const double term = static_cast<double>(qwen_iq1s_value(row_data, index)) * dequantized[index];
            exact += term;
            magnitude += std::fabs(term);
        }
        const float kernel = qwen_iq1s_dot_q8_k_vnni512(packed.data(), quantized.data(), kElements, row);
        const float floating = qwen_quant_dot_avx2(packed.data(), 19, input.data(), kElements, row);
        const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
        const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
        worst_tight = std::max(worst_tight, tight);
        worst_loose = std::max(worst_loose, loose);
        // Float summation over ten blocks of 256 terms: parts in 1e6.
        if (tight > 2e-5) {
            std::printf("row %d: kernel %.6f vs exact %.6f (rel %.2e)\n", row, kernel, exact, tight);
            ++failures;
        }
        // int8 activations: a quarter percent of the term magnitude, with margin.
        if (loose > 1e-2) {
            std::printf("row %d: kernel %.6f vs float path %.6f (rel %.2e)\n", row, kernel, floating, loose);
            ++failures;
        }
    }
    // The dequantizer must be bit-identical to the AVX2 decoder it displaces.
    std::vector<float> decoded_vnni(kElements), decoded_avx2(kElements);
    int dequant_mismatches = 0;
    for (int row = 0; row < kRows; ++row) {
        qwen_iq1s_dequant_row_vnni512(packed.data(), kElements, row, decoded_vnni.data());
        qwen_dequant_row_avx2(packed.data(), 19, kElements, row, decoded_avx2.data());
        for (int index = 0; index < kElements; ++index)
            if (std::memcmp(&decoded_vnni[index], &decoded_avx2[index], sizeof(float)) != 0)
                ++dequant_mismatches;
    }
    if (dequant_mismatches) {
        std::printf("dequantizer differs from the AVX2 decoder in %d of %d values\n",
                    dequant_mismatches, kRows * kElements);
        ++failures;
    }
    // IQ3_S through the same harness: random blocks, pinned d, exact reference
    // on the dequantized int8 input, AVX2 float dot as the loose bound.
    constexpr std::uint64_t kIq3sRowBytes = static_cast<std::uint64_t>(kBlocks) * kIq3sBlockBytes;
    std::vector<std::uint8_t> packed3(kIq3sRowBytes * kRows);
    for (auto& byte : packed3) byte = static_cast<std::uint8_t>(generator());
    for (int row = 0; row < kRows; ++row)
        for (int block = 0; block < kBlocks; ++block) {
            auto* base = packed3.data() + row * kIq3sRowBytes + block * kIq3sBlockBytes;
            base[0] = 0x00; base[1] = 0x3c;
        }
    double worst3_tight = 0.0, worst3_loose = 0.0;
    for (int row = 0; row < kRows; ++row) {
        const auto* row_data = packed3.data() + row * kIq3sRowBytes;
        double exact = 0.0, magnitude = 0.0;
        for (int index = 0; index < kElements; ++index) {
            const double term = static_cast<double>(qwen_iq3s_value(row_data, index)) * dequantized[index];
            exact += term;
            magnitude += std::fabs(term);
        }
        const float kernel = qwen_iq3s_dot_q8_k_vnni512(packed3.data(), quantized.data(), kElements, row);
        const float floating = qwen_quant_dot_avx2(packed3.data(), 21, input.data(), kElements, row);
        const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
        const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
        worst3_tight = std::max(worst3_tight, tight);
        worst3_loose = std::max(worst3_loose, loose);
        if (tight > 2e-5) {
            std::printf("iq3s row %d: kernel %.6f vs exact %.6f (rel %.2e)\n", row, kernel, exact, tight);
            ++failures;
        }
        if (loose > 1e-2) {
            std::printf("iq3s row %d: kernel %.6f vs float path %.6f (rel %.2e)\n", row, kernel, floating, loose);
            ++failures;
        }
    }
    int dequant3_mismatches = 0;
    for (int row = 0; row < kRows; ++row) {
        qwen_iq3s_dequant_row_vnni512(packed3.data(), kElements, row, decoded_vnni.data());
        qwen_dequant_row_avx2(packed3.data(), 21, kElements, row, decoded_avx2.data());
        for (int index = 0; index < kElements; ++index)
            if (std::memcmp(&decoded_vnni[index], &decoded_avx2[index], sizeof(float)) != 0)
                ++dequant3_mismatches;
    }
    if (dequant3_mismatches) {
        std::printf("iq3s dequantizer differs from the AVX2 decoder in %d values\n", dequant3_mismatches);
        ++failures;
    }
    std::printf("qwen_iq3s_vnni512: %d rows, worst rel error %.2e vs exact, %.2e vs float path\n",
                kRows, worst3_tight, worst3_loose);
    std::printf("qwen_iq1s_vnni512_contract: %d rows, worst rel error %.2e vs exact, %.2e vs float path, dequant bit-exact\n",
                kRows, worst_tight, worst_loose);
    return failures ? 1 : 0;
}
