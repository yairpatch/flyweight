// AVX512-VNNI expert GEMM for Q4_0 weights against Q8_0-style activations.
//
// The multi-stream dot kernels amortize the nibble decode over 8 tokens but
// still pay a convert+fmadd per 32-weight block per token, which caps them
// near 6 int8 MACs per instruction. This file processes 8 output rows per
// dpbusd instead: weights are repacked once per layer into an 8-row
// interleaved layout, the activation bytes broadcast across all 8 rows, and
// the zero-point correction happens in the integer domain, so the float tail
// is one convert and one fmadd per block for 8 rows at once.
//
// The repack is layout-only: the f16 block scales and the nibble bytes are
// the checkpoint's own, so any numeric difference against the dot kernels is
// purely float summation order (per-lane partial sums there, whole-block
// sums here), the same class of difference the rows forward already has
// against single-token decode.

#include <qwen_cpu_kernel.h>

// The same <cmath>/<algorithm> spelling as the reference quantizer this is
// bit-identical to. The GCC __builtin_ forms are not declared by MSVC.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

void qwen_q4_0_repack_x8(
    const std::uint8_t* packed, int rows, int elements, std::uint8_t* out
) {
    const int blocks = elements / 32;
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(blocks) * 18;
    for (int group = 0; group < rows / 8; ++group) {
        for (int block = 0; block < blocks; ++block) {
            std::uint8_t* destination =
                out + (static_cast<std::uint64_t>(group) * blocks + block) * 144;
            for (int r = 0; r < 8; ++r) {
                const std::uint8_t* source =
                    packed + (static_cast<std::uint64_t>(group) * 8 + r) * row_bytes
                    + static_cast<std::uint64_t>(block) * 18;
                // f16 scale bits, then the two 8-byte nibble units. Unit 0 is
                // bytes 0..7 (weights 0..7 low, 16..23 high), unit 1 is bytes
                // 8..15 (weights 8..15 low, 24..31 high).
                std::memcpy(destination + 2 * r, source, 2);
                std::memcpy(destination + 16 + r * 8, source + 2, 8);
                std::memcpy(destination + 80 + r * 8, source + 10, 8);
            }
        }
    }
}

void qwen_quantize_q8_gemm(
    const float* input, int elements,
    std::int8_t* values, float* scales, std::int32_t* bsums8
) {
    // Bit-identical to qwen_quantize_q8_0; only the storage layout differs,
    // plus the 8*sum sidecar the integer zero-point correction needs.
    for (int block = 0; block < elements / 32; ++block) {
        const float* source = input + block * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(source[i]));
        const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        scales[block] = amax / 127.0f;
        std::int32_t sum = 0;
        for (int i = 0; i < 32; ++i) {
            const std::int8_t q =
                static_cast<std::int8_t>(std::lrintf(source[i] * inverse));
            values[block * 32 + i] = q;
            sum += q;
        }
        bsums8[block] = 8 * sum;
    }
}

namespace {

// Lane indices that turn 8 row scales into pair form (each scale duplicated
// into the two int32 lanes its row's dpbusd results land in), and the even
// lanes that hold the finished row values.
alignas(64) constexpr std::int32_t kPairIndex[16] = {
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
alignas(64) constexpr std::int32_t kEvenIndex[16] = {
    0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0};

template <int Tokens>
void gemm_tile(
    const std::uint8_t* group_base, int blocks, int first_token,
    const std::int8_t* const* values, const float* const* scales,
    const std::int32_t* const* bsums8, float* const* outputs, int out_offset
) {
    const __m512i pair_index = _mm512_load_si512(kPairIndex);
    const __m512i even_index = _mm512_load_si512(kEvenIndex);
    const __m512i nibble_mask = _mm512_set1_epi8(15);
    __m512 accumulators[Tokens];
    for (int t = 0; t < Tokens; ++t) accumulators[t] = _mm512_setzero_ps();
    for (int block = 0; block < blocks; ++block) {
        const std::uint8_t* base = group_base + static_cast<std::uint64_t>(block) * 144;
        _mm_prefetch(reinterpret_cast<const char*>(base + 1152), _MM_HINT_T0);
        const __m256 row_scales = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(base)));
        const __m512 pair_scales = _mm512_permutexvar_ps(
            pair_index, _mm512_castps256_ps512(row_scales));
        const __m512i unit0 = _mm512_loadu_si512(base + 16);
        const __m512i unit1 = _mm512_loadu_si512(base + 80);
        const __m512i low0 = _mm512_and_si512(unit0, nibble_mask);
        const __m512i low1 = _mm512_and_si512(unit1, nibble_mask);
        const __m512i high0 =
            _mm512_and_si512(_mm512_srli_epi16(unit0, 4), nibble_mask);
        const __m512i high1 =
            _mm512_and_si512(_mm512_srli_epi16(unit1, 4), nibble_mask);
        for (int t = 0; t < Tokens; ++t) {
            const std::int8_t* activation = values[first_token + t] + block * 32;
            std::int64_t chunks[4];
            std::memcpy(chunks, activation, sizeof(chunks));
            __m512i dots = _mm512_dpbusd_epi32(
                _mm512_setzero_si512(), low0, _mm512_set1_epi64(chunks[0]));
            dots = _mm512_dpbusd_epi32(dots, low1, _mm512_set1_epi64(chunks[1]));
            dots = _mm512_dpbusd_epi32(dots, high0, _mm512_set1_epi64(chunks[2]));
            dots = _mm512_dpbusd_epi32(dots, high1, _mm512_set1_epi64(chunks[3]));
            // Pair lanes hold half-row partials; after the swap-add both hold
            // the full row's unsigned dot, and the zero point subtracts away
            // exactly (dot - 8*sum(a) == sum((n-8)*a)) while still integer.
            dots = _mm512_add_epi32(
                dots, _mm512_shuffle_epi32(dots, _MM_PERM_CDAB));
            dots = _mm512_sub_epi32(
                dots, _mm512_set1_epi32(bsums8[first_token + t][block]));
            accumulators[t] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(dots),
                _mm512_mul_ps(pair_scales,
                              _mm512_set1_ps(scales[first_token + t][block])),
                accumulators[t]);
        }
    }
    for (int t = 0; t < Tokens; ++t) {
        const __m512 rows = _mm512_permutexvar_ps(even_index, accumulators[t]);
        _mm256_storeu_ps(outputs[first_token + t] + out_offset,
                         _mm512_castps512_ps256(rows));
    }
}

}  // namespace

void qwen_q4_0x8_q8_gemm_vnni512(
    const std::uint8_t* repacked, int rows, int elements,
    const std::int8_t* const* values, const float* const* scales,
    const std::int32_t* const* bsums8, int tokens, float* const* outputs
) {
    const int blocks = elements / 32;
    for (int group = 0; group < rows / 8; ++group) {
        const std::uint8_t* group_base =
            repacked + static_cast<std::uint64_t>(group) * blocks * 144;
        const int out_offset = group * 8;
        int token = 0;
        for (; token + 4 <= tokens; token += 4)
            gemm_tile<4>(group_base, blocks, token, values, scales, bsums8,
                         outputs, out_offset);
        switch (tokens - token) {
        case 3: gemm_tile<3>(group_base, blocks, token, values, scales, bsums8,
                             outputs, out_offset); break;
        case 2: gemm_tile<2>(group_base, blocks, token, values, scales, bsums8,
                             outputs, out_offset); break;
        case 1: gemm_tile<1>(group_base, blocks, token, values, scales, bsums8,
                             outputs, out_offset); break;
        default: break;
        }
    }
}
