#include "q4_kernel.h"

#include <immintrin.h>

int q4_matvec_avx512(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
) {
    if ((columns & 31) != 0) {
        return q4_matvec_scalar(packed, scales, vector, output, rows, columns);
    }
    const std::int32_t blocks_per_row = columns / 32;
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    const __m256i zero_point = _mm256_set1_epi8(8);
    for (std::int32_t row = 0; row < rows; ++row) {
        // Two accumulators keep the two FMAs per block independent: a single
        // accumulator serializes every FMA on its latency chain, which caps
        // per-core throughput well below what the RAM stream can deliver.
        __m512 accumulator0 = _mm512_setzero_ps();
        __m512 accumulator1 = _mm512_setzero_ps();
        const std::int64_t block_start = static_cast<std::int64_t>(row)
            * blocks_per_row;
        for (std::int32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const std::int64_t block = block_start + block_index;
            _mm_prefetch(
                reinterpret_cast<const char*>(packed + block * 16 + 1024),
                _MM_HINT_T0
            );
            const __m128i bytes = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(packed + block * 16)
            );
            const __m128i low = _mm_and_si128(bytes, nibble_mask);
            const __m128i high = _mm_and_si128(
                _mm_srli_epi16(bytes, 4), nibble_mask
            );
            const __m128i pairs_low = _mm_unpacklo_epi8(low, high);
            const __m128i pairs_high = _mm_unpackhi_epi8(low, high);
            __m256i quantized = _mm256_set_m128i(pairs_high, pairs_low);
            quantized = _mm256_sub_epi8(quantized, zero_point);
            const __m512 q0 = _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized))
            );
            const __m512 q1 = _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized, 1))
            );
            const __m512 scale = _mm512_set1_ps(_mm_cvtss_f32(
                _mm_cvtph_ps(_mm_cvtsi32_si128(scales[block]))
            ));
            const float* input = vector + block_index * 32;
            accumulator0 = _mm512_fmadd_ps(
                _mm512_mul_ps(q0, scale), _mm512_loadu_ps(input), accumulator0
            );
            accumulator1 = _mm512_fmadd_ps(
                _mm512_mul_ps(q1, scale), _mm512_loadu_ps(input + 16), accumulator1
            );
        }
        output[row] = _mm512_reduce_add_ps(
            _mm512_add_ps(accumulator0, accumulator1)
        );
    }
    return 0;
}
