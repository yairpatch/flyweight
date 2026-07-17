#include "q4_kernel.h"

#include <immintrin.h>

namespace {

float horizontal_sum(__m256 value) {
    __m128 lower = _mm256_castps256_ps128(value);
    __m128 upper = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(lower, upper);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

}

int q4_matvec_avx2(
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
        __m256 accumulator = _mm256_setzero_ps();
        const std::int64_t block_start = static_cast<std::int64_t>(row)
            * blocks_per_row;
        for (std::int32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const std::int64_t block = block_start + block_index;
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
            const __m128i first = _mm256_castsi256_si128(quantized);
            const __m128i second = _mm256_extracti128_si256(quantized, 1);
            const float scale = _mm_cvtss_f32(
                _mm_cvtph_ps(_mm_cvtsi32_si128(scales[block]))
            );
            const __m256 scale_vector = _mm256_set1_ps(scale);
            const float* input = vector + block_index * 32;
            const __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first));
            const __m256 q1 = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(_mm_srli_si128(first, 8))
            );
            const __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second));
            const __m256 q3 = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(_mm_srli_si128(second, 8))
            );
            accumulator = _mm256_fmadd_ps(
                _mm256_mul_ps(q0, scale_vector), _mm256_loadu_ps(input), accumulator
            );
            accumulator = _mm256_fmadd_ps(
                _mm256_mul_ps(q1, scale_vector), _mm256_loadu_ps(input + 8), accumulator
            );
            accumulator = _mm256_fmadd_ps(
                _mm256_mul_ps(q2, scale_vector), _mm256_loadu_ps(input + 16), accumulator
            );
            accumulator = _mm256_fmadd_ps(
                _mm256_mul_ps(q3, scale_vector), _mm256_loadu_ps(input + 24), accumulator
            );
        }
        output[row] = horizontal_sum(accumulator);
    }
    return 0;
}
