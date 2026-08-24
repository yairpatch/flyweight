// Contract for the Q4_0 expert GEMM: qwen_q4_0x8_q8_gemm_vnni512 against the
// multi-stream dot kernel it replaces, both judged against a double-precision
// reference built from exact integer dots. The GEMM keeps whole blocks
// integer-exact where the dot kernel rounds per-dword partials, so the
// contract requires the GEMM to sit at least as close to the reference as the
// dots do -- a regression that merely "roughly matches" the old kernel while
// drifting from the truth fails here.
#include "qwen_cpu_kernel.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <immintrin.h>

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
    return (registers[2] & (1 << 11)) != 0;
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ecx & (1u << 11)) != 0;
#endif
}

}  // namespace

static std::uint16_t to_half(float value) {
    return static_cast<std::uint16_t>(
        _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(value), 0), 0));
}

int main() {
    if (!has_avx512_vnni()) {
        std::printf("SKIP: no AVX512-VNNI\n");
        return 0;
    }
    const int rows = 64, elements = 256, tokens = 7;
    const int blocks = elements / 32;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> nibble(0, 15);
    std::uniform_real_distribution<float> scale(0.001f, 0.05f);
    std::uniform_real_distribution<float> act(-2.0f, 2.0f);

    // Build Q4_0 rows: per block 2B f16 scale + 16B nibbles.
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(rows) * blocks * 18);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < blocks; ++b) {
            std::uint8_t* base = packed.data() + (static_cast<std::size_t>(r) * blocks + b) * 18;
            const std::uint16_t half = to_half(scale(rng));
            std::memcpy(base, &half, 2);
            for (int k = 0; k < 16; ++k)
                base[2 + k] = static_cast<std::uint8_t>(nibble(rng) | (nibble(rng) << 4));
        }

    // Activations per token.
    std::vector<float> inputs(static_cast<std::size_t>(tokens) * elements);
    for (auto& v : inputs) v = act(rng);

    // Reference: existing block-quantize + per-row dot.
    std::vector<QwenQ80Block> q8(static_cast<std::size_t>(tokens) * blocks);
    for (int t = 0; t < tokens; ++t)
        qwen_quantize_q8_0(inputs.data() + static_cast<std::size_t>(t) * elements,
                           elements, q8.data() + static_cast<std::size_t>(t) * blocks);
    std::vector<float> reference(static_cast<std::size_t>(tokens) * rows);
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r)
            reference[static_cast<std::size_t>(t) * rows + r] = qwen_quant_dot_q4_0_q8_0_vnni(
                packed.data(), q8.data() + static_cast<std::size_t>(t) * blocks, elements, r);

    // GEMM path.
    std::vector<std::uint8_t> repacked(packed.size());
    qwen_q4_0_repack_x8(packed.data(), rows, elements, repacked.data());
    std::vector<std::int8_t> values(static_cast<std::size_t>(tokens) * elements);
    std::vector<float> scales(static_cast<std::size_t>(tokens) * blocks);
    std::vector<std::int32_t> bsums(static_cast<std::size_t>(tokens) * blocks);
    std::vector<const std::int8_t*> vp(tokens);
    std::vector<const float*> sp(tokens);
    std::vector<const std::int32_t*> bp(tokens);
    std::vector<float> out(static_cast<std::size_t>(tokens) * rows);
    std::vector<float*> op(tokens);
    for (int t = 0; t < tokens; ++t) {
        qwen_quantize_q8_gemm(inputs.data() + static_cast<std::size_t>(t) * elements, elements,
                              values.data() + static_cast<std::size_t>(t) * elements,
                              scales.data() + static_cast<std::size_t>(t) * blocks,
                              bsums.data() + static_cast<std::size_t>(t) * blocks);
        vp[t] = values.data() + static_cast<std::size_t>(t) * elements;
        sp[t] = scales.data() + static_cast<std::size_t>(t) * blocks;
        bp[t] = bsums.data() + static_cast<std::size_t>(t) * blocks;
        op[t] = out.data() + static_cast<std::size_t>(t) * rows;
    }
    // Quantization must be value-identical between the two layouts.
    for (int t = 0; t < tokens; ++t)
        for (int b = 0; b < blocks; ++b) {
            const auto& block = q8[static_cast<std::size_t>(t) * blocks + b];
            if (block.scale != sp[t][b] ||
                std::memcmp(block.values, vp[t] + b * 32, 32) != 0) {
                std::printf("FAIL quantize mismatch t=%d b=%d\n", t, b);
                return 1;
            }
        }
    qwen_q4_0x8_q8_gemm_vnni512(repacked.data(), rows, elements,
                                vp.data(), sp.data(), bp.data(), tokens, op.data());

    // Double-precision reference: exact int dots, double scale products. Both
    // float kernels must sit within summation-order distance of this; the one
    // that keeps whole blocks integer-exact should sit closer.
    auto exact = [&](int t, int r) {
        double sum = 0.0;
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* base =
                packed.data() + (static_cast<std::size_t>(r) * blocks + b) * 18;
            std::uint16_t half;
            std::memcpy(&half, base, 2);
            const float d4 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(half)));
            std::int64_t dot = 0;
            for (int k = 0; k < 32; ++k) {
                const int n = k < 16 ? (base[2 + k] & 15) : (base[2 + k - 16] >> 4);
                dot += static_cast<std::int64_t>(n - 8) * vp[t][b * 32 + k];
            }
            sum += static_cast<double>(d4) * static_cast<double>(sp[t][b]) *
                   static_cast<double>(dot);
        }
        return sum;
    };
    double worst_gemm = 0.0, worst_dots = 0.0;
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            const double truth = exact(t, r);
            const double magnitude = std::fabs(truth) + 1e-3;
            worst_gemm = std::max(worst_gemm,
                std::fabs(out[static_cast<std::size_t>(t) * rows + r] - truth) / magnitude);
            worst_dots = std::max(worst_dots,
                std::fabs(reference[static_cast<std::size_t>(t) * rows + r] - truth) / magnitude);
        }
    std::printf("gemm vs exact %.3e, dots vs exact %.3e -> %s\n",
                worst_gemm, worst_dots,
                worst_gemm < 5e-6 && worst_gemm < worst_dots ? "PASS" : "FAIL");
    return worst_gemm < 5e-6 && worst_gemm < worst_dots ? 0 : 1;
}
