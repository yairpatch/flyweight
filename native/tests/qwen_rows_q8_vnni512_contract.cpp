// Contract for the batched rows x tokens int8 expert path
// (qwen_cpu_iq1s_vnni512.cpp): folded IQ1_S / IQ4_NL rows against unsigned-8
// activations must match the exact dot over the same quantized activations to
// float-summation precision, and the float path to int8-activation precision.
// Also a small timing datum against dequant + f32 GEMM at the sweep's shape.

#include "qwen_cpu_kernel.h"
#include "qwen_kquant.h"

#include <chrono>
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
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, 7, 0);
    return (regs[2] & (1 << 11)) != 0 && (regs[1] & (1 << 16)) != 0;
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 11)) != 0 && (ebx & (1u << 16)) != 0;
#endif
}

}  // namespace

int main() {
    if (!has_avx512_vnni()) {
        std::printf("qwen_rows_q8_vnni512_contract: no AVX512-VNNI, skipped\n");
        return 0;
    }
    std::mt19937 generator(11);
    std::uniform_real_distribution<float> spread(-1.0f, 1.0f);
    int failures = 0;

    // ---- IQ1_S gate/up shape: 2560 wide, 4 rows, 16 tokens ----
    {
        constexpr int kElements = 2560, kRows = 4, kTokens = 16;
        constexpr int kBlocks = kElements / 256;
        constexpr std::uint64_t kRowBytes = static_cast<std::uint64_t>(kBlocks) * kIq1sBlockBytes;
        std::vector<std::uint8_t> packed(kRowBytes * kRows);
        for (auto& byte : packed) byte = static_cast<std::uint8_t>(generator());
        for (int row = 0; row < kRows; ++row)
            for (int block = 0; block < kBlocks; ++block) {
                auto* base = packed.data() + row * kRowBytes + block * kIq1sBlockBytes;
                base[0] = 0x00; base[1] = 0x3c;  // f16 1.0
            }
        std::vector<float> input(static_cast<std::size_t>(kTokens) * kElements);
        for (auto& value : input) value = spread(generator) * spread(generator);
        std::vector<std::uint8_t> u8(input.size());
        std::vector<float> scales(static_cast<std::size_t>(kTokens) * kBlocks);
        std::vector<float> sums(static_cast<std::size_t>(kTokens) * kBlocks * 8);
        std::vector<const std::uint8_t*> act(kTokens);
        std::vector<const float*> act_scales(kTokens), act_sums(kTokens);
        std::vector<const float*> act_f32(kTokens);
        for (int t = 0; t < kTokens; ++t) {
            qwen_quantize_u8_k256_vnni512(input.data() + t * kElements, kElements,
                                          u8.data() + t * kElements, scales.data() + t * kBlocks,
                                          sums.data() + t * kBlocks * 8);
            act[t] = u8.data() + t * kElements;
            act_scales[t] = scales.data() + t * kBlocks;
            act_sums[t] = sums.data() + t * kBlocks * 8;
            act_f32[t] = input.data() + t * kElements;
        }
        std::vector<std::int8_t> folded(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> fold_scales(kRows * kBlocks), fold_deltas(kRows * kBlocks * 8);
        std::vector<std::int32_t> fold_corr(kRows * kBlocks);
        qwen_iq1s_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                    fold_scales.data(), fold_corr.data(), fold_deltas.data());
        std::vector<float> out(static_cast<std::size_t>(kRows) * kTokens);
        qwen_u8_gemm_k256_vnni512(folded.data(), fold_scales.data(), fold_corr.data(),
                                  fold_deltas.data(), kRows, act.data(), act_scales.data(),
                                  act_sums.data(), kTokens, kElements, out.data());
        double worst_tight = 0, worst_loose = 0;
        for (int row = 0; row < kRows; ++row) {
            const auto* row_data = packed.data() + row * kRowBytes;
            for (int t = 0; t < kTokens; ++t) {
                double exact = 0, magnitude = 0;
                for (int index = 0; index < kElements; ++index) {
                    const double x = (static_cast<int>(u8[t * kElements + index]) - 128) *
                                     static_cast<double>(scales[t * kBlocks + index / 256]);
                    const double term = static_cast<double>(qwen_iq1s_value(row_data, index)) * x;
                    exact += term;
                    magnitude += std::fabs(term);
                }
                const float kernel = out[row * kTokens + t];
                const float floating = qwen_quant_dot_avx2(packed.data(), 19, act_f32[t], kElements, row);
                const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
                const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
                worst_tight = std::max(worst_tight, tight);
                worst_loose = std::max(worst_loose, loose);
                if (tight > 2e-5 || loose > 1e-2) {
                    std::printf("iq1s row %d token %d: kernel %.6f exact %.6f float %.6f\n",
                                row, t, kernel, exact, floating);
                    ++failures;
                }
            }
        }
        std::printf("iq1s gemm: worst vs exact %.2e, vs float %.2e\n", worst_tight, worst_loose);
        // Timing at the sweep shape: fold+gemm vs dequant+f32 gemm, gate+up.
        std::vector<float> dq(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> ref(out.size());
        const int reps = 400;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            qwen_iq1s_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                        fold_scales.data(), fold_corr.data(), fold_deltas.data());
            qwen_u8_gemm_k256_vnni512(folded.data(), fold_scales.data(), fold_corr.data(),
                                      fold_deltas.data(), kRows, act.data(), act_scales.data(),
                                      act_sums.data(), kTokens, kElements, out.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int row = 0; row < kRows; ++row)
                qwen_iq1s_dequant_row_vnni512(packed.data(), kElements, row, dq.data() + row * kElements);
            qwen_f32_gemm_rows_avx512(dq.data(), kRows, act_f32.data(), kTokens, kElements, ref.data());
        }
        auto t2 = std::chrono::steady_clock::now();
        auto t3 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep)
            qwen_iq1s_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                        fold_scales.data(), fold_corr.data(), fold_deltas.data());
        auto t4 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep)
            for (int row = 0; row < kRows; ++row)
                qwen_iq1s_dequant_row_vnni512(packed.data(), kElements, row, dq.data() + row * kElements);
        auto t5 = std::chrono::steady_clock::now();
        const double us = 1e-3 / reps;
        std::printf("iq1s 4x16x2560: int8 fold+gemm %.1f us (fold %.1f) | f32 dequant+gemm %.1f us (dequant %.1f)\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() * us,
                    std::chrono::duration<double, std::nano>(t4 - t3).count() * us,
                    std::chrono::duration<double, std::nano>(t2 - t1).count() * us,
                    std::chrono::duration<double, std::nano>(t5 - t4).count() * us);
    }

    // ---- IQ4_NL down shape: 640 wide, 4 rows, 16 tokens ----
    {
        constexpr int kElements = 640, kRows = 4, kTokens = 16;
        constexpr int kBlocks = kElements / 32, kPairFloats = (kElements / 64) * 16;
        constexpr std::uint64_t kRowBytes = static_cast<std::uint64_t>(kBlocks) * kIq4nlBlockBytes;
        std::vector<std::uint8_t> packed(kRowBytes * kRows);
        for (auto& byte : packed) byte = static_cast<std::uint8_t>(generator());
        for (int row = 0; row < kRows; ++row)
            for (int block = 0; block < kBlocks; ++block) {
                auto* base = packed.data() + row * kRowBytes + block * kIq4nlBlockBytes;
                base[0] = 0x00; base[1] = 0x3c;
            }
        std::vector<float> input(static_cast<std::size_t>(kTokens) * kElements);
        for (auto& value : input) value = spread(generator) * spread(generator);
        std::vector<std::uint8_t> u8(input.size());
        std::vector<float> scales(static_cast<std::size_t>(kTokens) * kPairFloats);
        std::vector<const std::uint8_t*> act(kTokens);
        std::vector<const float*> act_scales(kTokens);
        std::vector<const float*> act_f32(kTokens);
        for (int t = 0; t < kTokens; ++t) {
            qwen_quantize_u8_k32_vnni512(input.data() + t * kElements, kElements,
                                         u8.data() + t * kElements, scales.data() + t * kPairFloats);
            act[t] = u8.data() + t * kElements;
            act_scales[t] = scales.data() + t * kPairFloats;
            act_f32[t] = input.data() + t * kElements;
        }
        std::vector<std::int8_t> folded(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> fold_scales(kRows * kPairFloats);
        std::vector<std::int32_t> fold_init(kRows * kPairFloats);
        qwen_iq4nl_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                     fold_scales.data(), fold_init.data());
        std::vector<float> out(static_cast<std::size_t>(kRows) * kTokens);
        qwen_u8_gemm_k32_vnni512(folded.data(), fold_scales.data(), fold_init.data(), kRows,
                                 act.data(), act_scales.data(), kTokens, kElements, out.data());
        double worst_tight = 0, worst_loose = 0;
        for (int row = 0; row < kRows; ++row) {
            const auto* row_data = packed.data() + row * kRowBytes;
            for (int t = 0; t < kTokens; ++t) {
                double exact = 0, magnitude = 0;
                for (int index = 0; index < kElements; ++index) {
                    const int block = index / 32;
                    const float scale = scales[t * kPairFloats + (block / 2) * 16 + (block & 1) * 8];
                    const double x = (static_cast<int>(u8[t * kElements + index]) - 128) * static_cast<double>(scale);
                    const double term = static_cast<double>(qwen_iq4nl_value(row_data, index)) * x;
                    exact += term;
                    magnitude += std::fabs(term);
                }
                const float kernel = out[row * kTokens + t];
                const float floating = qwen_quant_dot_avx2(packed.data(), 20, act_f32[t], kElements, row);
                const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
                const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
                worst_tight = std::max(worst_tight, tight);
                worst_loose = std::max(worst_loose, loose);
                if (tight > 2e-5 || loose > 1e-2) {
                    std::printf("iq4nl row %d token %d: kernel %.6f exact %.6f float %.6f\n",
                                row, t, kernel, exact, floating);
                    ++failures;
                }
            }
        }
        std::printf("iq4nl gemm: worst vs exact %.2e, vs float %.2e\n", worst_tight, worst_loose);
        std::vector<float> dq(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> ref(out.size());
        const int reps = 2000;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            qwen_iq4nl_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                         fold_scales.data(), fold_init.data());
            qwen_u8_gemm_k32_vnni512(folded.data(), fold_scales.data(), fold_init.data(), kRows,
                                     act.data(), act_scales.data(), kTokens, kElements, out.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int row = 0; row < kRows; ++row)
                qwen_dequant_row_avx2(packed.data(), 20, kElements, row, dq.data() + row * kElements);
            qwen_f32_gemm_rows_avx512(dq.data(), kRows, act_f32.data(), kTokens, kElements, ref.data());
        }
        auto t2 = std::chrono::steady_clock::now();
        const double us = 1e-3 / reps;
        std::printf("iq4nl 4x16x640: int8 fold+gemm %.1f us | f32 dequant+gemm %.1f us\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() * us,
                    std::chrono::duration<double, std::nano>(t2 - t1).count() * us);
    }
    if (failures) { std::printf("FAILED: %d\n", failures); return 1; }
    std::printf("qwen_rows_q8_vnni512_contract: ok\n");
    return 0;
}
