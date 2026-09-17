// Contract for the AVX-VNNI (VEX, 256-bit) IQ1_S x Q8_K dot.
//
// The same two references as qwen_iq1s_vnni512_contract.cpp: the scalar IQ1_S
// decoder against the dequantized int8 activation (float summation order only),
// and the float-activation AVX2 kernel this path replaces on hybrid Intel parts
// (int8 activation rounding). Blocks are random except the super-block scale,
// pinned to 1.0 for the reason given there.
#include "qwen_cpu_kernel.h"
#include "qwen_kquant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace {

bool has_avx_vnni() {
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuid(registers, 0);
    if (registers[0] < 7) return false;
    __cpuidex(registers, 7, 1);
    return (registers[0] & (1 << 4)) != 0 && (_xgetbv(0) & 0x6) == 0x6;
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 1, eax, ebx, ecx, edx);
    return (eax & (1u << 4)) != 0 && __builtin_cpu_supports("avx2");
#endif
}

}  // namespace

int main() {
    if (!has_avx_vnni()) {
        std::printf("qwen_iq1s_avx_vnni_contract: no AVX-VNNI, skipped\n");
        return 0;
    }
    constexpr int kElements = 2560;   // qwen4exp's expert width, ten super-blocks
    constexpr int kRows = 64;
    constexpr int kBlocks = kElements / 256;
    constexpr std::uint64_t kRowBytes = static_cast<std::uint64_t>(kBlocks) * kIq1sBlockBytes;
    std::mt19937 generator(11);
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
        const float kernel = qwen_iq1s_dot_q8_k_avx_vnni(packed.data(), quantized.data(), kElements, row);
        const float floating = qwen_quant_dot_avx2(packed.data(), 19, input.data(), kElements, row);
        const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
        const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
        worst_tight = std::max(worst_tight, tight);
        worst_loose = std::max(worst_loose, loose);
        if (tight > 2e-5) {
            std::printf("row %d: kernel %.6f vs exact %.6f (rel %.2e)\n", row, kernel, exact, tight);
            ++failures;
        }
        if (loose > 1e-2) {
            std::printf("row %d: kernel %.6f vs float path %.6f (rel %.2e)\n", row, kernel, floating, loose);
            ++failures;
        }
    }
    std::printf("qwen_iq1s_avx_vnni_contract: worst tight %.2e, worst loose %.2e, %s\n",
                worst_tight, worst_loose, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
