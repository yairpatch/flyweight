// AVX-512 Q8_0 row dot products. See cpu_q8_dot.h.
//
// Layout per block: fp16 scale at +0, then 32 int8 at +2, 34 bytes total.
// The 34-byte stride means blocks are not 16-byte aligned, so every load here
// is deliberately unaligned -- loadu, not load.

#include "cpu_q8_dot.h"

#include <immintrin.h>

namespace {

// 32 int8 -> two vectors of 16 floats.
inline void load_block_weights(const std::uint8_t* block, __m512& low,
                               __m512& high) {
    const __m128i first =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
    const __m128i second =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2 + 16));
    low = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(first));
    high = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(second));
}

inline float scale_of(const std::uint8_t* block) {
    return _cvtsh_ss(*reinterpret_cast<const std::uint16_t*>(block));
}

}  // namespace

float colibri_q8_row_dot_avx512(const std::uint8_t* row_packed,
                                const float* vector, int blocks) {
    // Two accumulators carried across all blocks: no per-block horizontal sum,
    // and the scale is folded into the weights rather than applied after, which
    // keeps the dependency chain to one FMA per half-block.
    __m512 accumulator_low = _mm512_setzero_ps();
    __m512 accumulator_high = _mm512_setzero_ps();

    const std::uint8_t* cursor = row_packed;
    for (int block = 0; block < blocks; ++block) {
        const __m512 scale = _mm512_set1_ps(scale_of(cursor));
        __m512 weight_low, weight_high;
        load_block_weights(cursor, weight_low, weight_high);

        const __m512 input_low = _mm512_loadu_ps(vector + block * 32);
        const __m512 input_high = _mm512_loadu_ps(vector + block * 32 + 16);

        accumulator_low = _mm512_fmadd_ps(_mm512_mul_ps(weight_low, scale),
                                          input_low, accumulator_low);
        accumulator_high = _mm512_fmadd_ps(_mm512_mul_ps(weight_high, scale),
                                           input_high, accumulator_high);
        cursor += 34;
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(accumulator_low, accumulator_high));
}

void colibri_q8_row_dot_pair_avx512(const std::uint8_t* gate_packed,
                                    const std::uint8_t* up_packed,
                                    const float* vector, int blocks,
                                    float* gate_out, float* up_out) {
    __m512 gate_low = _mm512_setzero_ps();
    __m512 gate_high = _mm512_setzero_ps();
    __m512 up_low = _mm512_setzero_ps();
    __m512 up_high = _mm512_setzero_ps();

    const std::uint8_t* gate_cursor = gate_packed;
    const std::uint8_t* up_cursor = up_packed;
    for (int block = 0; block < blocks; ++block) {
        // The activation pair is loaded once and used by both weight streams.
        const __m512 input_low = _mm512_loadu_ps(vector + block * 32);
        const __m512 input_high = _mm512_loadu_ps(vector + block * 32 + 16);

        const __m512 gate_scale = _mm512_set1_ps(scale_of(gate_cursor));
        __m512 gate_weight_low, gate_weight_high;
        load_block_weights(gate_cursor, gate_weight_low, gate_weight_high);
        gate_low = _mm512_fmadd_ps(_mm512_mul_ps(gate_weight_low, gate_scale),
                                   input_low, gate_low);
        gate_high = _mm512_fmadd_ps(_mm512_mul_ps(gate_weight_high, gate_scale),
                                    input_high, gate_high);

        const __m512 up_scale = _mm512_set1_ps(scale_of(up_cursor));
        __m512 up_weight_low, up_weight_high;
        load_block_weights(up_cursor, up_weight_low, up_weight_high);
        up_low = _mm512_fmadd_ps(_mm512_mul_ps(up_weight_low, up_scale),
                                 input_low, up_low);
        up_high = _mm512_fmadd_ps(_mm512_mul_ps(up_weight_high, up_scale),
                                  input_high, up_high);

        gate_cursor += 34;
        up_cursor += 34;
    }
    *gate_out = _mm512_reduce_add_ps(_mm512_add_ps(gate_low, gate_high));
    *up_out = _mm512_reduce_add_ps(_mm512_add_ps(up_low, up_high));
}
