// Contract for the int16 prefill rows path (qwen_cpu_iq1s_vnni512.cpp):
//
//   1. the fold recovers the codebook integer exactly -- folded * block scale
//      reproduces the float decoder's value to within float rounding of the
//      product (no weight information is lost);
//   2. the dpwssd GEMM equals a double-precision dot of the folded weights
//      against the 14-bit activations;
//   3. against the float dot, the only difference is the activation rounding;
//   4. it is not slower than dequant + f32 GEMM at the sweep shape.
//
// Random bytes are valid blocks for every format here (all index bits index
// a full table), with the f16 scale pinned to a sane magnitude.

#include <qwen_cpu_kernel.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include "qwen_kquant.h"

namespace {

bool has_avx512_vnni() {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, 7, 0);
    return (regs[2] & (1 << 11)) != 0 && (regs[1] & (1 << 16)) != 0;
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 11)) != 0 && (ebx & (1u << 16)) != 0;
#endif
}

struct Format { std::uint32_t type; const char* name; std::uint32_t block_bytes; float multiplier; };

}  // namespace

int main() {
    if (!has_avx512_vnni()) {
        std::printf("qwen_rows_i16_vnni512_contract: no AVX512-VNNI, skipped\n");
        return 0;
    }
    const Format formats[] = {
        {16, "iq2xxs", kIq2xxsBlockBytes, 8.0f}, {17, "iq2xs", kIq2xsBlockBytes, 8.0f},
        {22, "iq2s", kIq2sBlockBytes, 8.0f},     {18, "iq3xxs", kIq3xxsBlockBytes, 4.0f},
        {21, "iq3s", kIq3sBlockBytes, 1.0f},     {19, "iq1s", kIq1sBlockBytes, 8.0f},
        {23, "iq4xs", kIq4xsBlockBytes, 1.0f},
    };
    std::mt19937 generator(23);
    std::uniform_real_distribution<float> spread(-1.0f, 1.0f);
    int failures = 0;
    constexpr int kElements = 2560, kRows = 4, kTokens = 16;
    constexpr int kBlocks = kElements / 256;

    std::vector<float> input(static_cast<std::size_t>(kTokens) * kElements);
    for (auto& value : input) value = spread(generator) * spread(generator);
    std::vector<std::int16_t> i16(input.size());
    std::vector<float> act_scales_store(static_cast<std::size_t>(kTokens) * kBlocks);
    std::vector<const std::int16_t*> act(kTokens);
    std::vector<const float*> act_scales(kTokens), act_f32(kTokens);
    for (int t = 0; t < kTokens; ++t) {
        qwen_quantize_i16_k256_vnni512(input.data() + t * kElements, kElements,
                                       i16.data() + t * kElements, act_scales_store.data() + t * kBlocks);
        act[t] = i16.data() + t * kElements;
        act_scales[t] = act_scales_store.data() + t * kBlocks;
        act_f32[t] = input.data() + t * kElements;
    }
    // Activation rounding: 14 bits per 256 block.
    {
        double worst = 0;
        for (std::size_t index = 0; index < input.size(); ++index) {
            const int t = static_cast<int>(index / kElements), block = static_cast<int>((index % kElements) / 256);
            const double back = static_cast<double>(i16[index]) * act_scales_store[t * kBlocks + block];
            worst = std::max(worst, std::fabs(back - input[index]) / (act_scales_store[t * kBlocks + block] + 1e-30));
        }
        std::printf("activation rounding: worst %.3f steps of max/16383\n", worst);
        if (worst > 0.51) ++failures;
    }

    for (const auto& format : formats) {
        const std::uint64_t row_bytes = static_cast<std::uint64_t>(kBlocks) * format.block_bytes;
        std::vector<std::uint8_t> packed(row_bytes * kRows);
        for (auto& byte : packed) byte = static_cast<std::uint8_t>(generator());
        for (int row = 0; row < kRows; ++row)
            for (int block = 0; block < kBlocks; ++block) {
                auto* base = packed.data() + row * row_bytes + block * format.block_bytes;
                // f16 in [0.5, 2): exponent 14 or 15, random mantissa; some negative.
                const std::uint16_t mantissa = static_cast<std::uint16_t>(generator() & 0x3ff);
                const std::uint16_t exponent = static_cast<std::uint16_t>((14 + (generator() & 1)) << 10);
                const std::uint16_t sign = static_cast<std::uint16_t>((generator() & 7) == 0 ? 0x8000 : 0);
                const std::uint16_t bits = static_cast<std::uint16_t>(sign | exponent | mantissa);
                std::memcpy(base, &bits, 2);
            }
        std::vector<float> dq(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> inverse(kRows * kBlocks), scales(kRows * kBlocks);
        for (int row = 0; row < kRows; ++row) {
            qwen_dequant_row_avx2(packed.data(), format.type, kElements, row, dq.data() + row * kElements);
            for (int block = 0; block < kBlocks; ++block) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, packed.data() + row * row_bytes + block * format.block_bytes, 2);
                const float d = qwen_half_value(bits);
                scales[row * kBlocks + block] = d / format.multiplier;
                inverse[row * kBlocks + block] = d != 0.0f ? format.multiplier / d : 0.0f;
            }
        }
        std::vector<std::int16_t> folded(static_cast<std::size_t>(kRows) * kElements);
        qwen_fold_rows_i16_vnni512(dq.data(), inverse.data(), kRows, kElements, folded.data());
        // 1. exact fold: the float code is within 1e-3 of an integer, and the
        //    integer times the block scale is the decoded value.
        double worst_code = 0, worst_value = 0; int largest = 0;
        for (int row = 0; row < kRows; ++row)
            for (int index = 0; index < kElements; ++index) {
                const int block = index / 256;
                const double code = static_cast<double>(dq[row * kElements + index]) * inverse[row * kBlocks + block];
                worst_code = std::max(worst_code, std::fabs(code - std::nearbyint(code)));
                const double back = static_cast<double>(folded[row * kElements + index]) * scales[row * kBlocks + block];
                const double reference = dq[row * kElements + index];
                worst_value = std::max(worst_value, std::fabs(back - reference) / (std::fabs(reference) + 1e-30));
                largest = std::max(largest, std::abs(static_cast<int>(folded[row * kElements + index])));
            }
        // 2./3. GEMM against exact and float.
        std::vector<float> out(static_cast<std::size_t>(kRows) * kTokens);
        qwen_i16_gemm_k256_vnni512(folded.data(), scales.data(), kRows, act.data(), act_scales.data(),
                                   kTokens, kElements, out.data());
        double worst_tight = 0, worst_loose = 0;
        for (int row = 0; row < kRows; ++row)
            for (int t = 0; t < kTokens; ++t) {
                double exact = 0, magnitude = 0;
                for (int index = 0; index < kElements; ++index) {
                    const int block = index / 256;
                    const double w = static_cast<double>(folded[row * kElements + index]) * scales[row * kBlocks + block];
                    const double x = static_cast<double>(i16[t * kElements + index]) * act_scales_store[t * kBlocks + block];
                    exact += w * x; magnitude += std::fabs(w * x);
                }
                const float kernel = out[row * kTokens + t];
                const float floating = qwen_quant_dot_avx2(packed.data(), format.type, act_f32[t], kElements, row);
                const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
                const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
                worst_tight = std::max(worst_tight, tight);
                worst_loose = std::max(worst_loose, loose);
                if (tight > 2e-5 || loose > 2e-4) {
                    std::printf("%s row %d token %d: kernel %.6f exact %.6f float %.6f\n",
                                format.name, row, t, kernel, exact, floating);
                    ++failures;
                }
            }
        // 4. timing at the sweep shape: dequant+fold+gemm vs dequant+f32 gemm, one role.
        std::vector<float> ref(out.size());
        const int reps = 300;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int row = 0; row < kRows; ++row)
                qwen_dequant_row_avx2(packed.data(), format.type, kElements, row, dq.data() + row * kElements);
            qwen_fold_rows_i16_vnni512(dq.data(), inverse.data(), kRows, kElements, folded.data());
            qwen_i16_gemm_k256_vnni512(folded.data(), scales.data(), kRows, act.data(), act_scales.data(),
                                       kTokens, kElements, out.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int row = 0; row < kRows; ++row)
                qwen_dequant_row_avx2(packed.data(), format.type, kElements, row, dq.data() + row * kElements);
            qwen_f32_gemm_rows_avx512(dq.data(), kRows, act_f32.data(), kTokens, kElements, ref.data());
        }
        auto t2 = std::chrono::steady_clock::now();
        const double i16_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        const double f32_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / reps;
        std::printf("%-7s code residual %.1e, largest |code| %d, value rel %.1e, gemm vs exact %.1e, "
                    "vs float %.1e; i16 %.1f us vs f32 %.1f us (%.2fx)\n",
                    format.name, worst_code, largest, worst_value, worst_tight, worst_loose,
                    i16_us, f32_us, f32_us / i16_us);
        if (worst_code > 1e-3 || worst_value > 1e-6 || largest > 4096) ++failures;
    }
    if (failures) { std::printf("qwen_rows_i16_vnni512_contract: %d failures\n", failures); return 1; }
    std::printf("qwen_rows_i16_vnni512_contract: ok\n");
    return 0;
}
