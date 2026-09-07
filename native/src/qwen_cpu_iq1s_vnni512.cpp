// IQ1_S and IQ3_S row dots against Q8_K activations, AVX512-VNNI.
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
//
// IQ3_S (the gate/up experts of the UD-IQ4_XS checkpoints) takes the same
// shape with different bookkeeping. Its 64 grid indices per super-block are a
// byte each plus one high bit, and those high bits are stored one per index
// in qh -- which is exactly a 16-lane mask per gather, so the high bit is a
// masked broadcast. Its signs are one bit per value, exactly a 64-lane byte
// mask, applied to the activation as a masked negate. The grid magnitudes are
// unsigned, so they feed dpbusd directly. The AVX2 float dot ran at 36.7 GB/s
// on 16 cores, 2.3 GB/s per core; this one runs at the DRAM roof.
//
// The row dequantizer below shares the index build and the gathers, and
// writes the same floats the AVX2 decoder does, bit for bit. The batched
// expert path (prefill, multi-sequence decode) decodes each routed row to
// float once per chunk before its GEMM, and on the UD-IQ1_S checkpoints that
// decode was the largest single cost of a chunk: 41 Gweights/s on 16 cores
// through the octet-at-a-time decoder, 73 through this one.

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

// The 32 grid indices of one super-block, as four vectors of eight dwords in
// octet order, ready for the gathers.
inline void iq1s_block_indices(const std::uint8_t* base, __m256i octets[4], __m128i& qh) {
    const __m512i qs = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 2)));
    qh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base + 34));
    const __m512i qh_by_lane =
        _mm512_permutexvar_epi16(kGroupOfLane, _mm512_castsi128_si512(qh));
    const __m512i high = _mm512_slli_epi16(
        _mm512_and_si512(_mm512_srlv_epi16(qh_by_lane, kOctetShift), _mm512_set1_epi16(7)), 8);
    const __m512i index = _mm512_or_si512(qs, high);
    const __m512i index_low = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(index));
    const __m512i index_high =
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(index, 1));
    octets[0] = _mm512_castsi512_si256(index_low);
    octets[1] = _mm512_extracti64x4_epi64(index_low, 1);
    octets[2] = _mm512_castsi512_si256(index_high);
    octets[3] = _mm512_extracti64x4_epi64(index_high, 1);
}

}  // namespace

void qwen_iq1s_dequant_row_vnni512(
    const std::uint8_t* packed,
    int elements,
    std::uint64_t row,
    float* output
) {
    const int blocks = elements / 256;
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(blocks) * kIq1sBlockBytes;
    const __m256 plus_delta = _mm256_set1_ps(kIq1sDelta);
    const __m256 minus_delta = _mm256_set1_ps(-kIq1sDelta);
    for (int block = 0; block < blocks; ++block) {
        const auto* base = row_data + block * kIq1sBlockBytes;
        __m256i index_octets[4];
        __m128i qh;
        iq1s_block_indices(base, index_octets, qh);
        // Per-group scale d * (2s + 1) and signed delta, spilled to eight
        // floats each: the same values the AVX2 decoder broadcasts, in the same
        // order of operations, so the output is bit-identical to it.
        const __m256i qh32 = _mm256_cvtepu16_epi32(qh);
        const __m256 scale = _mm256_mul_ps(
            _mm256_set1_ps(half_value(base)),
            _mm256_cvtepi32_ps(_mm256_add_epi32(
                _mm256_slli_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(qh32, 12), _mm256_set1_epi32(7)), 1),
                _mm256_set1_epi32(1))));
        const __m256i negative = _mm256_srai_epi32(_mm256_slli_epi32(qh32, 16), 31);
        const __m256 delta =
            _mm256_blendv_ps(plus_delta, minus_delta, _mm256_castsi256_ps(negative));
        alignas(32) float scales[8];
        alignas(32) float deltas[8];
        _mm256_store_ps(scales, scale);
        _mm256_store_ps(deltas, delta);
        float* out = output + block * 256;
        for (int pair = 0; pair < 4; ++pair) {
            const __m512i octets = _mm512_i32gather_epi64(
                index_octets[pair], static_cast<const void*>(kIq1sGrid), 8);
            const __m128i quarters[4] = {
                _mm512_castsi512_si128(octets), _mm512_extracti32x4_epi32(octets, 1),
                _mm512_extracti32x4_epi32(octets, 2), _mm512_extracti32x4_epi32(octets, 3),
            };
            for (int quarter = 0; quarter < 4; ++quarter) {
                const int group = pair * 2 + quarter / 2;
                const __m512 weights = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(quarters[quarter]));
                _mm512_storeu_ps(
                    out + pair * 64 + quarter * 16,
                    _mm512_mul_ps(_mm512_set1_ps(scales[group]),
                                  _mm512_add_ps(weights, _mm512_set1_ps(deltas[group]))));
            }
        }
    }
}

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
    const __m256i plus_delta = _mm256_castps_si256(_mm256_set1_ps(kIq1sDelta - 1.0f));
    const __m256i minus_delta = _mm256_castps_si256(_mm256_set1_ps(-kIq1sDelta - 1.0f));
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const auto* base = row_data + block * kIq1sBlockBytes;
        const auto& q8 = input[block];
        __m256i index_octets[4];
        __m128i qh;
        iq1s_block_indices(base, index_octets, qh);
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

// dpbusd over 64 values yields 16 dwords, one per grid entry: lanes 0-7 are
// the even group of the pair, 8-15 the odd one. Scale lanes for gather k are
// groups 2k and 2k+1, the same selection kScaleLanes already encodes.
float qwen_iq3s_dot_q8_k_vnni512(
    const std::uint8_t* packed,
    const QwenQ8KBlock* input,
    int elements,
    std::uint64_t row
) {
    const int blocks = elements / 256;
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(blocks) * kIq3sBlockBytes;
    const __m512i high_bit = _mm512_set1_epi32(256);
    const __m512i zero = _mm512_setzero_si512();
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const auto* base = row_data + block * kIq3sBlockBytes;
        const auto& q8 = input[block];
        const auto* quants = base + 2;
        const auto* high = base + 66;
        const auto* signs = base + 74;
        // Group scales 1 + 2 * nibble, eight dwords.
        std::uint32_t scale_bits = 0;
        std::memcpy(&scale_bits, base + 106, sizeof(scale_bits));
        const __m256i nibbles = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(scale_bits)),
                              _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28)),
            _mm256_set1_epi32(15));
        const __m512i scales512 = _mm512_castsi256_si512(_mm256_add_epi32(
            _mm256_slli_epi32(nibbles, 1), _mm256_set1_epi32(1)));
        __m512i accumulator = zero;
        for (int gather = 0; gather < 4; ++gather) {
            // Sixteen indices: the qs byte, plus bit 8 from the matching qh
            // bit -- two qh bytes are the 16-lane mask in index order.
            std::uint16_t high_mask = 0;
            std::memcpy(&high_mask, high + gather * 2, sizeof(high_mask));
            const __m512i index = _mm512_or_si512(
                _mm512_cvtepu8_epi32(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(quants + gather * 16))),
                _mm512_maskz_mov_epi32(static_cast<__mmask16>(high_mask), high_bit));
            const __m512i magnitudes = _mm512_i32gather_epi32(
                index, static_cast<const void*>(kIq3sGrid), 4);
            // Eight sign bytes are the 64-lane mask over this gather's values.
            std::uint64_t sign_mask = 0;
            std::memcpy(&sign_mask, signs + gather * 8, sizeof(sign_mask));
            const __m512i activation = _mm512_loadu_si512(
                static_cast<const void*>(q8.values + gather * 64));
            const __m512i signed_activation = _mm512_mask_sub_epi8(
                activation, static_cast<__mmask64>(sign_mask), zero, activation);
            const __m512i dots = _mm512_dpbusd_epi32(zero, magnitudes, signed_activation);
            accumulator = _mm512_add_epi32(
                accumulator,
                _mm512_mullo_epi32(dots, _mm512_permutexvar_epi32(kScaleLanes[gather], scales512)));
        }
        result += half_value(base) * q8.scale *
            static_cast<float>(_mm512_reduce_add_epi32(accumulator));
    }
    return result;
}

// IQ3_S row to float: gathered unsigned magnitudes, the sign bits as a mask
// over the float sign bit, one multiply by d * (1 + 2 * scale). Same values
// and operation order as the AVX2 decoder, so bit-identical to it.
void qwen_iq3s_dequant_row_vnni512(
    const std::uint8_t* packed,
    int elements,
    std::uint64_t row,
    float* output
) {
    const int blocks = elements / 256;
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(blocks) * kIq3sBlockBytes;
    const __m512i high_bit = _mm512_set1_epi32(256);
    const __m512i sign_bit = _mm512_set1_epi32(static_cast<int>(0x80000000u));
    for (int block = 0; block < blocks; ++block) {
        const auto* base = row_data + block * kIq3sBlockBytes;
        const auto* quants = base + 2;
        const auto* high = base + 66;
        const auto* signs = base + 74;
        const auto* scales = base + 106;
        const float d = half_value(base);
        float* out = output + block * 256;
        for (int gather = 0; gather < 4; ++gather) {
            std::uint16_t high_mask = 0;
            std::memcpy(&high_mask, high + gather * 2, sizeof(high_mask));
            const __m512i index = _mm512_or_si512(
                _mm512_cvtepu8_epi32(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(quants + gather * 16))),
                _mm512_maskz_mov_epi32(static_cast<__mmask16>(high_mask), high_bit));
            const __m512i magnitudes = _mm512_i32gather_epi32(
                index, static_cast<const void*>(kIq3sGrid), 4);
            std::uint64_t sign_mask = 0;
            std::memcpy(&sign_mask, signs + gather * 8, sizeof(sign_mask));
            const __m128i quarters[4] = {
                _mm512_castsi512_si128(magnitudes), _mm512_extracti32x4_epi32(magnitudes, 1),
                _mm512_extracti32x4_epi32(magnitudes, 2), _mm512_extracti32x4_epi32(magnitudes, 3),
            };
            for (int quarter = 0; quarter < 4; ++quarter) {
                const int group = gather * 2 + quarter / 2;
                const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
                const __m512 weight = _mm512_set1_ps(d * static_cast<float>(1 + 2 * scale));
                const __m512i bits = _mm512_castps_si512(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(quarters[quarter])));
                const auto lanes = static_cast<__mmask16>(sign_mask >> (quarter * 16));
                const __m512 signed_magnitudes = _mm512_castsi512_ps(
                    _mm512_mask_xor_epi32(bits, lanes, bits, sign_bit));
                _mm512_storeu_ps(out + gather * 64 + quarter * 16,
                                 _mm512_mul_ps(signed_magnitudes, weight));
            }
        }
    }
}
