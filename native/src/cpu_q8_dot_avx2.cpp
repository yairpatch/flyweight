// AVX2 Q8_0 row dot products. See cpu_q8_dot.h.
//
// Same structure as the AVX-512 version at half the width: 32 values per block
// become four 8-wide vectors instead of two 16-wide ones. Needs -mf16c for the
// fp16 block scale.

#include "cpu_q8_dot.h"

#include <cstring>
#include <immintrin.h>

namespace {

// 32 int8 -> four vectors of 8 floats.
inline void load_block_weights(const std::uint8_t* block, __m256 out[4]) {
    const __m128i first =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
    const __m128i second =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2 + 16));
    out[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first));
    out[1] = _mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(first, 8)));
    out[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second));
    out[3] = _mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(second, 8)));
}

inline float scale_of(const std::uint8_t* block) {
    // Not _cvtsh_ss: that convenience spelling is GCC/Clang only. This is the
    // same F16C instruction and the spelling the other kernels already use.
    std::uint16_t bits = 0;
    std::memcpy(&bits, block, sizeof(bits));  // no aliasing pun; same codegen
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

inline float horizontal_sum(__m256 value) {
    const __m128 high = _mm256_extractf128_ps(value, 1);
    const __m128 low = _mm256_castps256_ps128(value);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    return _mm_cvtss_f32(sum);
}

}  // namespace

float flyweight_q8_row_dot_avx2(const std::uint8_t* row_packed,
                              const float* vector, int blocks) {
    __m256 accumulator[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };

    const std::uint8_t* cursor = row_packed;
    for (int block = 0; block < blocks; ++block) {
        const __m256 scale = _mm256_set1_ps(scale_of(cursor));
        __m256 weights[4];
        load_block_weights(cursor, weights);
        for (int part = 0; part < 4; ++part) {
            const __m256 input = _mm256_loadu_ps(vector + block * 32 + part * 8);
            accumulator[part] = _mm256_fmadd_ps(
                _mm256_mul_ps(weights[part], scale), input, accumulator[part]);
        }
        cursor += 34;
    }
    return horizontal_sum(_mm256_add_ps(
        _mm256_add_ps(accumulator[0], accumulator[1]),
        _mm256_add_ps(accumulator[2], accumulator[3])));
}

void flyweight_q8_row_dot_pair_avx2(const std::uint8_t* gate_packed,
                                  const std::uint8_t* up_packed,
                                  const float* vector, int blocks,
                                  float* gate_out, float* up_out) {
    __m256 gate_accumulator[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    __m256 up_accumulator[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };

    const std::uint8_t* gate_cursor = gate_packed;
    const std::uint8_t* up_cursor = up_packed;
    for (int block = 0; block < blocks; ++block) {
        const __m256 gate_scale = _mm256_set1_ps(scale_of(gate_cursor));
        const __m256 up_scale = _mm256_set1_ps(scale_of(up_cursor));
        __m256 gate_weights[4], up_weights[4];
        load_block_weights(gate_cursor, gate_weights);
        load_block_weights(up_cursor, up_weights);
        for (int part = 0; part < 4; ++part) {
            const __m256 input = _mm256_loadu_ps(vector + block * 32 + part * 8);
            gate_accumulator[part] = _mm256_fmadd_ps(
                _mm256_mul_ps(gate_weights[part], gate_scale), input,
                gate_accumulator[part]);
            up_accumulator[part] = _mm256_fmadd_ps(
                _mm256_mul_ps(up_weights[part], up_scale), input,
                up_accumulator[part]);
        }
        gate_cursor += 34;
        up_cursor += 34;
    }
    *gate_out = horizontal_sum(_mm256_add_ps(
        _mm256_add_ps(gate_accumulator[0], gate_accumulator[1]),
        _mm256_add_ps(gate_accumulator[2], gate_accumulator[3])));
    *up_out = horizontal_sum(_mm256_add_ps(
        _mm256_add_ps(up_accumulator[0], up_accumulator[1]),
        _mm256_add_ps(up_accumulator[2], up_accumulator[3])));
}
