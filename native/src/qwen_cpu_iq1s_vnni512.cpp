// IQ1_S row dot against Q8_K activations, AVX512-VNNI.
//
// The float IQ1_S kernel in qwen_cpu_avx2.cpp walks the 2048-entry grid one
// octet at a time: four scalar index computations, four table loads and three
// register inserts per group of 32 weights, then two horizontal reductions.
// Measured on the 9955HX that is ~24 cycles per group and 0.9 GB/s per core,
// so 16 cores stream IQ1_S experts at 12 GB/s against a 58 GB/s DRAM roof --
// the gate/up experts of the UD-IQ1_S checkpoints were two thirds of the
// expert phase for a third of its bytes.
//
// This kernel decodes a whole 256-value super-block at once. The 32 grid
// indices are assembled in one register (the qs byte in the low eight bits,
// the group's three qh bits above it), four gathers fetch the 32 octets, and
// four dpbusd instructions take the dot against the int8 activation. The grid
// holds {-1, 0, +1}; dpbusd wants an unsigned left operand, so the octets are
// lifted by one and the group's activation sum is subtracted back out through
// the Q8_K block sums -- the same sums carry the +-1/8 group delta. Measured
// 4.1 GB/s per core, 43 GB/s on 16 cores streaming.
//
// Numerically this is the Q8_K activation path the IQ2_XS/IQ3_XXS experts
// already take: the only approximation is the int8 activation rounding.

#include <qwen_cpu_kernel.h>

#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include "qwen_kquant.h"

namespace {

float half_value(const std::uint8_t* pointer) {
    std::uint16_t bits;
    std::memcpy(&bits, pointer, sizeof(bits));
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// Lane i of the 32 index lanes is group i/4, octet i%4: its qh halfword and
// the shift that brings that octet's three high bits down.
const __m512i kGroupOfLane = _mm512_set_epi16(
    7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4,
    3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
const __m512i kOctetShift = _mm512_set_epi16(
    9, 6, 3, 0, 9, 6, 3, 0, 9, 6, 3, 0, 9, 6, 3, 0,
    9, 6, 3, 0, 9, 6, 3, 0, 9, 6, 3, 0, 9, 6, 3, 0);
// dpbusd over 64 activations yields 16 dwords: the first eight belong to the
// even group of the pair, the rest to the odd one.
const __m512i kScaleLanes[4] = {
    _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2),
    _mm512_set_epi32(5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4),
    _mm512_set_epi32(7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
};

}  // namespace

float qwen_iq1s_dot_q8_k_vnni512(
    const std::uint8_t* packed,
    const QwenQ8KBlock* input,
    int elements,
    std::uint64_t row
) {
    const int blocks = elements / 256;
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(blocks) * kIq1sBlockBytes;
    const __m512i one = _mm512_set1_epi8(1);
    const __m512i seven = _mm512_set1_epi16(7);
    const __m256i plus_delta = _mm256_castps_si256(_mm256_set1_ps(kIq1sDelta - 1.0f));
    const __m256i minus_delta = _mm256_castps_si256(_mm256_set1_ps(-kIq1sDelta - 1.0f));
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const auto* base = row_data + block * kIq1sBlockBytes;
        const auto& q8 = input[block];
        // 32 grid indices: qs byte | (three qh bits of the group) << 8.
        const __m512i qs = _mm512_cvtepu8_epi16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 2)));
        const __m128i qh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base + 34));
        const __m512i qh_by_lane =
            _mm512_permutexvar_epi16(kGroupOfLane, _mm512_castsi128_si512(qh));
        const __m512i high = _mm512_slli_epi16(
            _mm512_and_si512(_mm512_srlv_epi16(qh_by_lane, kOctetShift), seven), 8);
        const __m512i index = _mm512_or_si512(qs, high);
        const __m512i index_low = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(index));
        const __m512i index_high =
            _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(index, 1));
        const __m256i index_octets[4] = {
            _mm512_castsi512_si256(index_low), _mm512_extracti64x4_epi64(index_low, 1),
            _mm512_castsi512_si256(index_high), _mm512_extracti64x4_epi64(index_high, 1),
        };
        // Per-group scale 2*s+1 from qh bits 12-14, as eight dwords.
        const __m256i qh32 = _mm256_cvtepu16_epi32(qh);
        const __m256i scales = _mm256_add_epi32(
            _mm256_slli_epi32(
                _mm256_and_si256(_mm256_srli_epi32(qh32, 12), _mm256_set1_epi32(7)), 1),
            _mm256_set1_epi32(1));
        const __m512i scales512 = _mm512_castsi256_si512(scales);
        __m512i accumulator = _mm512_setzero_si512();
        for (int pair = 0; pair < 4; ++pair) {
            const __m512i octets = _mm512_i32gather_epi64(
                index_octets[pair], static_cast<const void*>(kIq1sGrid), 8);
            const __m512i lifted = _mm512_add_epi8(octets, one);
            const __m512i activation = _mm512_loadu_si512(
                static_cast<const void*>(q8.values + pair * 64));
            const __m512i dots = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lifted, activation);
            accumulator = _mm512_add_epi32(
                accumulator,
                _mm512_mullo_epi32(dots, _mm512_permutexvar_epi32(kScaleLanes[pair], scales512)));
        }
        // The lift and the delta both ride the group's activation sum:
        // sum((w + 1) x) - sum(x) + delta * sum(x) = sum(w x) + delta * sum(x).
        const __m256i sums16 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8.sums));
        const __m256i group_sums = _mm256_madd_epi16(sums16, _mm256_set1_epi16(1));
        const __m256 scaled_sums =
            _mm256_cvtepi32_ps(_mm256_mullo_epi32(scales, group_sums));
        const __m256i negative = _mm256_srai_epi32(_mm256_slli_epi32(qh32, 16), 31);
        const __m256 delta_less_one = _mm256_castsi256_ps(
            _mm256_blendv_epi8(plus_delta, minus_delta, negative));
        const __m256 side = _mm256_mul_ps(scaled_sums, delta_less_one);
        __m128 side4 = _mm_add_ps(_mm256_castps256_ps128(side), _mm256_extractf128_ps(side, 1));
        side4 = _mm_hadd_ps(side4, side4);
        side4 = _mm_hadd_ps(side4, side4);
        result += half_value(base) * q8.scale *
            (static_cast<float>(_mm512_reduce_add_epi32(accumulator)) + _mm_cvtss_f32(side4));
    }
    return result;
}
