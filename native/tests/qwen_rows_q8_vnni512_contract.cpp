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

    // ---- IQ3_S gate/up shape: 2560 wide, 4 rows, 16 tokens ----
    // Same discipline as the IQ1_S block, against the shared int16 path:
    // the direct-from-packed fold is exact integer arithmetic
    // ((1+2s) * grid * sign <= 225 in int16), so fold+gemm must match the
    // exact dot over the same 14-bit activations to float-summation
    // precision, and the float path to activation precision.
    {
        constexpr int kElements = 2560, kRows = 4, kTokens = 16;
        constexpr int kBlocks = kElements / 256;
        constexpr std::uint64_t kRowBytes = static_cast<std::uint64_t>(kBlocks) * kIq3sBlockBytes;
        std::vector<std::uint8_t> packed(kRowBytes * kRows);
        for (auto& byte : packed) byte = static_cast<std::uint8_t>(generator());
        for (int row = 0; row < kRows; ++row)
            for (int block = 0; block < kBlocks; ++block) {
                auto* base = packed.data() + row * kRowBytes + block * kIq3sBlockBytes;
                base[0] = 0x00; base[1] = 0x3c;  // f16 1.0
            }
        std::vector<float> input(static_cast<std::size_t>(kTokens) * kElements);
        for (auto& value : input) value = spread(generator) * spread(generator);
        std::vector<std::int16_t> x16(input.size());
        std::vector<float> ascales(static_cast<std::size_t>(kTokens) * kBlocks);
        std::vector<const std::int16_t*> act(kTokens);
        std::vector<const float*> act_scales(kTokens);
        std::vector<const float*> act_f32(kTokens);
        for (int t = 0; t < kTokens; ++t) {
            qwen_quantize_i16_k256_vnni512(input.data() + t * kElements, kElements,
                                           x16.data() + t * kElements,
                                           ascales.data() + t * kBlocks);
            act[t] = x16.data() + t * kElements;
            act_scales[t] = ascales.data() + t * kBlocks;
            act_f32[t] = input.data() + t * kElements;
        }
        std::vector<std::int16_t> folded(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> fold_scales(kRows * kBlocks);
        qwen_iq3s_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                    fold_scales.data());
        std::vector<float> out(static_cast<std::size_t>(kRows) * kTokens);
        qwen_i16_gemm_k256_vnni512(folded.data(), fold_scales.data(), kRows,
                                   act.data(), act_scales.data(),
                                   kTokens, kElements, out.data());
        double worst_tight = 0, worst_loose = 0;
        for (int row = 0; row < kRows; ++row) {
            for (int t = 0; t < kTokens; ++t) {
                double exact = 0, magnitude = 0;
                for (int index = 0; index < kElements; ++index) {
                    const double x = static_cast<double>(x16[t * kElements + index]) *
                                     static_cast<double>(ascales[t * kBlocks + index / 256]);
                    const std::uint64_t absolute =
                        static_cast<std::uint64_t>(row) * kElements + index;
                    const double term = static_cast<double>(qwen_iq3s_value(
                        packed.data(), absolute)) * x;
                    exact += term;
                    magnitude += std::fabs(term);
                }
                const float kernel = out[row * kTokens + t];
                const float floating = qwen_quant_dot_avx2(packed.data(), 21, act_f32[t], kElements, row);
                const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
                const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
                worst_tight = std::max(worst_tight, tight);
                worst_loose = std::max(worst_loose, loose);
                if (tight > 2e-5 || loose > 1e-2) {
                    std::printf("iq3s row %d token %d: kernel %.6f exact %.6f float %.6f\n",
                                row, t, kernel, exact, floating);
                    ++failures;
                }
            }
        }
        std::printf("iq3s gemm: worst vs exact %.2e, vs float %.2e\n", worst_tight, worst_loose);
        // Timing at the sweep shape: direct fold+gemm vs the incumbent
        // float dequant + float fold + int16 gemm.
        std::vector<float> dq(static_cast<std::size_t>(kRows) * kElements);
        std::vector<std::int16_t> fi16(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> finv(kRows * kBlocks), fsc(kRows * kBlocks);
        std::vector<float> ref(out.size());
        for (int row = 0; row < kRows; ++row)
            for (int block = 0; block < kBlocks; ++block) {
                // Multiplier 1 for IQ3_S: scales are d, inverses 1/d.
                const float d = fold_scales[row * kBlocks + block];
                fsc[row * kBlocks + block] = d;
                finv[row * kBlocks + block] = d != 0.0f ? 1.0f / d : 0.0f;
            }
        const int reps = 400;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            qwen_iq3s_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                        fold_scales.data());
            qwen_i16_gemm_k256_vnni512(folded.data(), fold_scales.data(), kRows,
                                       act.data(), act_scales.data(),
                                       kTokens, kElements, out.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int row = 0; row < kRows; ++row)
                qwen_iq3s_dequant_row_vnni512(packed.data(), kElements, row, dq.data() + row * kElements);
            qwen_fold_rows_i16_vnni512(dq.data(), finv.data(), kRows, kElements, fi16.data());
            qwen_i16_gemm_k256_vnni512(fi16.data(), fsc.data(), kRows,
                                       act.data(), act_scales.data(),
                                       kTokens, kElements, ref.data());
        }
        auto t2 = std::chrono::steady_clock::now();
        auto t3 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep)
            qwen_iq3s_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                        fold_scales.data());
        auto t4 = std::chrono::steady_clock::now();
        const double us = 1e-3 / reps;
        std::printf("iq3s 4x16x2560: direct fold+gemm %.1f us (fold %.1f) | dequant+fold+gemm %.1f us\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() * us,
                    std::chrono::duration<double, std::nano>(t4 - t3).count() * us,
                    std::chrono::duration<double, std::nano>(t2 - t1).count() * us);
    }

    // ---- IQ2_XS gate/up shape: same harness, multiplier 8 ----
    {
        constexpr int kElements = 2560, kRows = 4, kTokens = 16;
        constexpr int kBlocks = kElements / 256;
        constexpr std::uint64_t kRowBytes = static_cast<std::uint64_t>(kBlocks) * kIq2xsBlockBytes;
        std::vector<std::uint8_t> packed(kRowBytes * kRows);
        for (auto& byte : packed) byte = static_cast<std::uint8_t>(generator());
        for (int row = 0; row < kRows; ++row)
            for (int block = 0; block < kBlocks; ++block) {
                auto* base = packed.data() + row * kRowBytes + block * kIq2xsBlockBytes;
                base[0] = 0x00; base[1] = 0x3c;  // f16 1.0
            }
        std::vector<float> input(static_cast<std::size_t>(kTokens) * kElements);
        for (auto& value : input) value = spread(generator) * spread(generator);
        std::vector<std::int16_t> x16(input.size());
        std::vector<float> ascales(static_cast<std::size_t>(kTokens) * kBlocks);
        std::vector<const std::int16_t*> act(kTokens);
        std::vector<const float*> act_scales(kTokens);
        std::vector<const float*> act_f32(kTokens);
        for (int t = 0; t < kTokens; ++t) {
            qwen_quantize_i16_k256_vnni512(input.data() + t * kElements, kElements,
                                           x16.data() + t * kElements,
                                           ascales.data() + t * kBlocks);
            act[t] = x16.data() + t * kElements;
            act_scales[t] = ascales.data() + t * kBlocks;
            act_f32[t] = input.data() + t * kElements;
        }
        std::vector<std::int16_t> folded(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> fold_scales(kRows * kBlocks);
        qwen_iq2xs_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                     fold_scales.data());
        std::vector<float> out(static_cast<std::size_t>(kRows) * kTokens);
        qwen_i16_gemm_k256_vnni512(folded.data(), fold_scales.data(), kRows,
                                   act.data(), act_scales.data(),
                                   kTokens, kElements, out.data());
        double worst_tight = 0, worst_loose = 0;
        for (int row = 0; row < kRows; ++row) {
            for (int t = 0; t < kTokens; ++t) {
                double exact = 0, magnitude = 0;
                for (int index = 0; index < kElements; ++index) {
                    const double x = static_cast<double>(x16[t * kElements + index]) *
                                     static_cast<double>(ascales[t * kBlocks + index / 256]);
                    const std::uint64_t absolute =
                        static_cast<std::uint64_t>(row) * kElements + index;
                    const double term = static_cast<double>(qwen_iq2xs_value(
                        packed.data(), absolute)) * x;
                    exact += term;
                    magnitude += std::fabs(term);
                }
                const float kernel = out[row * kTokens + t];
                const float floating = qwen_quant_dot_avx2(packed.data(), 17, act_f32[t], kElements, row);
                const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
                const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
                worst_tight = std::max(worst_tight, tight);
                worst_loose = std::max(worst_loose, loose);
                if (tight > 2e-5 || loose > 1e-2) {
                    std::printf("iq2xs row %d token %d: kernel %.6f exact %.6f float %.6f\n",
                                row, t, kernel, exact, floating);
                    ++failures;
                }
            }
        }
        std::printf("iq2xs gemm: worst vs exact %.2e, vs float %.2e\n", worst_tight, worst_loose);
        std::vector<float> dq(static_cast<std::size_t>(kRows) * kElements);
        std::vector<std::int16_t> fi16(static_cast<std::size_t>(kRows) * kElements);
        std::vector<float> finv(kRows * kBlocks), fsc(kRows * kBlocks);
        std::vector<float> ref(out.size());
        for (int row = 0; row < kRows; ++row)
            for (int block = 0; block < kBlocks; ++block) {
                const float d = fold_scales[row * kBlocks + block] * 8.0f;
                fsc[row * kBlocks + block] = fold_scales[row * kBlocks + block];
                finv[row * kBlocks + block] = d != 0.0f ? 8.0f / d : 0.0f;
            }
        const int reps = 400;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            qwen_iq2xs_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                         fold_scales.data());
            qwen_i16_gemm_k256_vnni512(folded.data(), fold_scales.data(), kRows,
                                       act.data(), act_scales.data(),
                                       kTokens, kElements, out.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int row = 0; row < kRows; ++row)
                qwen_dequant_row_avx2(packed.data(), 17, kElements, row, dq.data() + row * kElements);
            qwen_fold_rows_i16_vnni512(dq.data(), finv.data(), kRows, kElements, fi16.data());
            qwen_i16_gemm_k256_vnni512(fi16.data(), fsc.data(), kRows,
                                       act.data(), act_scales.data(),
                                       kTokens, kElements, ref.data());
        }
        auto t2 = std::chrono::steady_clock::now();
        auto t3 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep)
            qwen_iq2xs_fold_rows_vnni512(packed.data(), kElements, 0, kRows, folded.data(),
                                         fold_scales.data());
        auto t4 = std::chrono::steady_clock::now();
        const double us = 1e-3 / reps;
        std::printf("iq2xs 4x16x2560: direct fold+gemm %.1f us (fold %.1f) | dequant+fold+gemm %.1f us\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() * us,
                    std::chrono::duration<double, std::nano>(t4 - t3).count() * us,
                    std::chrono::duration<double, std::nano>(t2 - t1).count() * us);
    }

    // ---- IQ2_S / IQ2_XXS / IQ3_XXS: same harness, multipliers 8/8/4 ----
    {
        struct FormatCase {
            const char* name;
            std::uint32_t type;
            std::uint32_t block_bytes;
            float multiplier;
            float (*value)(const std::uint8_t*, std::uint64_t);
            void (*fold)(const std::uint8_t*, int, std::uint64_t, int,
                         std::int16_t*, float*);
        };
        const FormatCase formats[] = {
            {"iq2s", 22, kIq2sBlockBytes, 8.0f, qwen_iq2s_value,
             qwen_iq2s_fold_rows_vnni512},
            {"iq2xxs", 16, kIq2xxsBlockBytes, 8.0f, qwen_iq2xxs_value,
             qwen_iq2xxs_fold_rows_vnni512},
            {"iq3xxs", 18, kIq3xxsBlockBytes, 4.0f, qwen_iq3xxs_value,
             qwen_iq3xxs_fold_rows_vnni512},
        };
        for (const auto& format : formats) {
            constexpr int kElements = 2560, kRows = 4, kTokens = 16;
            constexpr int kBlocks = kElements / 256;
            const std::uint64_t kRowBytes =
                static_cast<std::uint64_t>(kBlocks) * format.block_bytes;
            std::vector<std::uint8_t> packed(kRowBytes * kRows);
            for (auto& byte : packed) byte = static_cast<std::uint8_t>(generator());
            for (int row = 0; row < kRows; ++row)
                for (int block = 0; block < kBlocks; ++block) {
                    auto* base = packed.data() + row * kRowBytes +
                                 block * format.block_bytes;
                    base[0] = 0x00; base[1] = 0x3c;  // f16 1.0
                }
            std::vector<float> input(static_cast<std::size_t>(kTokens) * kElements);
            for (auto& value : input) value = spread(generator) * spread(generator);
            std::vector<std::int16_t> x16(input.size());
            std::vector<float> ascales(static_cast<std::size_t>(kTokens) * kBlocks);
            std::vector<const std::int16_t*> act(kTokens);
            std::vector<const float*> act_scales(kTokens);
            std::vector<const float*> act_f32(kTokens);
            for (int t = 0; t < kTokens; ++t) {
                qwen_quantize_i16_k256_vnni512(
                    input.data() + t * kElements, kElements,
                    x16.data() + t * kElements, ascales.data() + t * kBlocks);
                act[t] = x16.data() + t * kElements;
                act_scales[t] = ascales.data() + t * kBlocks;
                act_f32[t] = input.data() + t * kElements;
            }
            std::vector<std::int16_t> folded(static_cast<std::size_t>(kRows) * kElements);
            std::vector<float> fold_scales(kRows * kBlocks);
            format.fold(packed.data(), kElements, 0, kRows, folded.data(),
                        fold_scales.data());
            std::vector<float> out(static_cast<std::size_t>(kRows) * kTokens);
            qwen_i16_gemm_k256_vnni512(folded.data(), fold_scales.data(), kRows,
                                       act.data(), act_scales.data(),
                                       kTokens, kElements, out.data());
            double worst_tight = 0, worst_loose = 0;
            for (int row = 0; row < kRows; ++row) {
                for (int t = 0; t < kTokens; ++t) {
                    double exact = 0, magnitude = 0;
                    for (int index = 0; index < kElements; ++index) {
                        const double x =
                            static_cast<double>(x16[t * kElements + index]) *
                            static_cast<double>(ascales[t * kBlocks + index / 256]);
                        const std::uint64_t absolute =
                            static_cast<std::uint64_t>(row) * kElements + index;
                        const double term = static_cast<double>(
                            format.value(packed.data(), absolute)) * x;
                        exact += term;
                        magnitude += std::fabs(term);
                    }
                    const float kernel = out[row * kTokens + t];
                    const float floating = qwen_quant_dot_avx2(
                        packed.data(), format.type, act_f32[t], kElements, row);
                    const double tight = std::fabs(kernel - exact) / (magnitude + 1e-6);
                    const double loose = std::fabs(kernel - floating) / (magnitude + 1e-6);
                    worst_tight = std::max(worst_tight, tight);
                    worst_loose = std::max(worst_loose, loose);
                    if (tight > 2e-5 || loose > 1e-2) {
                        std::printf("%s row %d token %d: kernel %.6f exact %.6f float %.6f\n",
                                    format.name, row, t, kernel, exact, floating);
                        ++failures;
                    }
                }
            }
            std::printf("%s gemm: worst vs exact %.2e, vs float %.2e\n",
                        format.name, worst_tight, worst_loose);
            std::vector<float> dq(static_cast<std::size_t>(kRows) * kElements);
            std::vector<std::int16_t> fi16(static_cast<std::size_t>(kRows) * kElements);
            std::vector<float> finv(kRows * kBlocks), fsc(kRows * kBlocks);
            std::vector<float> ref(out.size());
            for (int row = 0; row < kRows; ++row)
                for (int block = 0; block < kBlocks; ++block) {
                    const float d = fold_scales[row * kBlocks + block] *
                                    format.multiplier;
                    fsc[row * kBlocks + block] = fold_scales[row * kBlocks + block];
                    finv[row * kBlocks + block] =
                        d != 0.0f ? format.multiplier / d : 0.0f;
                }
            const int reps = 400;
            auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < reps; ++rep) {
                format.fold(packed.data(), kElements, 0, kRows, folded.data(),
                            fold_scales.data());
                qwen_i16_gemm_k256_vnni512(folded.data(), fold_scales.data(), kRows,
                                           act.data(), act_scales.data(),
                                           kTokens, kElements, out.data());
            }
            auto t1 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < reps; ++rep) {
                for (int row = 0; row < kRows; ++row)
                    qwen_dequant_row_avx2(packed.data(), format.type, kElements, row,
                                          dq.data() + row * kElements);
                qwen_fold_rows_i16_vnni512(dq.data(), finv.data(), kRows, kElements,
                                           fi16.data());
                qwen_i16_gemm_k256_vnni512(fi16.data(), fsc.data(), kRows,
                                           act.data(), act_scales.data(),
                                           kTokens, kElements, ref.data());
            }
            auto t2 = std::chrono::steady_clock::now();
            auto t3 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < reps; ++rep)
                format.fold(packed.data(), kElements, 0, kRows, folded.data(),
                            fold_scales.data());
            auto t4 = std::chrono::steady_clock::now();
            const double us = 1e-3 / reps;
            std::printf("%s 4x16x2560: direct fold+gemm %.1f us (fold %.1f) | dequant+fold+gemm %.1f us\n",
                        format.name,
                        std::chrono::duration<double, std::nano>(t1 - t0).count() * us,
                        std::chrono::duration<double, std::nano>(t4 - t3).count() * us,
                        std::chrono::duration<double, std::nano>(t2 - t1).count() * us);
        }
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
