#include "qwen_cpu_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <immintrin.h>

#include "qwen_kquant.h"

namespace {

float half_value(const std::uint8_t* pointer) {
    std::uint16_t bits;
    std::memcpy(&bits, pointer, sizeof(bits));
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// Sign application for the IQ codebook formats. Their grids store magnitudes
// and a separate 8-bit pattern says which of the eight are negated, which the
// scalar path does with a branch per weight. Negating a float is an XOR of the
// sign bit, so one 8-lane mask per pattern replaces all eight branches. The
// table is 8 KiB and indexed by that byte, so it stays hot in L1.
struct IqSignMasks {
    std::uint32_t lanes[256][8];
};

constexpr IqSignMasks build_iq_sign_masks() {
    IqSignMasks masks{};
    for (int pattern = 0; pattern < 256; ++pattern)
        for (int lane = 0; lane < 8; ++lane)
            masks.lanes[pattern][lane] =
                (pattern >> lane) & 1 ? 0x80000000u : 0u;
    return masks;
}

constexpr IqSignMasks kIqSignMasks = build_iq_sign_masks();

// Byte signs for the integer IQ x Q8_K kernels.  Applying the signs to the
// activation lets vpmaddubsw consume the codebook magnitudes as unsigned bytes
// without first expanding either operand to float.
struct IqSignBytes {
    std::int8_t lanes[256][8];
};

constexpr IqSignBytes build_iq_sign_bytes() {
    IqSignBytes signs{};
    for (int pattern = 0; pattern < 256; ++pattern)
        for (int lane = 0; lane < 8; ++lane)
            signs.lanes[pattern][lane] =
                (pattern >> lane) & 1 ? -1 : 1;
    return signs;
}

constexpr IqSignBytes kIqSignBytes = build_iq_sign_bytes();

// Eight consecutive codebook magnitudes, widened to float and signed.
inline __m256 iq_signed_octet(std::uint64_t grid, std::uint8_t signs) {
    const __m128i packed = _mm_cvtsi64_si128(static_cast<long long>(grid));
    const __m256 magnitudes =
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(packed));
    const __m256 mask = _mm256_loadu_ps(
        reinterpret_cast<const float*>(kIqSignMasks.lanes[signs]));
    return _mm256_xor_ps(magnitudes, mask);
}

float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

int horizontal_sum_i32(__m256i value) {
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

std::int64_t iq_sign_bytes(std::uint8_t pattern) {
    std::int64_t result;
    std::memcpy(&result, kIqSignBytes.lanes[pattern], sizeof(result));
    return result;
}

int dot_i8_8(__m128i left, __m128i right) {
    const __m128i products = _mm_mullo_epi16(
        _mm_cvtepi8_epi16(left), _mm_cvtepi8_epi16(right));
    const __m128i pairs = _mm_madd_epi16(products, _mm_set1_epi16(1));
    __m128i sum = _mm_hadd_epi32(pairs, pairs);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

int dot_i8_16(const std::int8_t* left, const std::int8_t* right) {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(left));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(right));
    const __m256i products = _mm256_mullo_epi16(
        _mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b));
    const __m256i pairs = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(pairs), _mm256_extracti128_si256(pairs, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

__m128i quantize_i8_16(const float* values, __m256 inverse) {
    const __m256i minimum = _mm256_set1_epi32(-127);
    const __m256i maximum = _mm256_set1_epi32(127);
    __m256i low = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_loadu_ps(values), inverse));
    __m256i high = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_loadu_ps(values + 8), inverse));
    low = _mm256_min_epi32(maximum, _mm256_max_epi32(minimum, low));
    high = _mm256_min_epi32(maximum, _mm256_max_epi32(minimum, high));

    // packs_epi32 works independently in each 128-bit lane. Reorder its four
    // 64-bit groups so the final pack stores values 0..15 contiguously.
    const __m256i packed16 = _mm256_permute4x64_epi64(
        _mm256_packs_epi32(low, high), 0xd8);
    return _mm_packs_epi16(
        _mm256_castsi256_si128(packed16),
        _mm256_extracti128_si256(packed16, 1));
}

std::int16_t sum_i8_16(__m128i values) {
    const __m256i widened = _mm256_cvtepi8_epi16(values);
    const __m256i pairs = _mm256_madd_epi16(
        widened, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(pairs), _mm256_extracti128_si256(pairs, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return static_cast<std::int16_t>(_mm_cvtsi128_si32(sum));
}

__m256 bytes_to_float(__m128i values) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(values));
}

float ue4m3_value(std::uint8_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x80u) << 24;
    const std::uint32_t exponent = (bits >> 3) & 0x0fu;
    const std::uint32_t mantissa = bits & 7u;
    if (exponent == 0) {
        const float value = static_cast<float>(mantissa) * (1.0f / 512.0f);
        return (bits & 0x80u) ? -value : value;
    }
    std::uint32_t widened = sign | ((exponent + 120u) << 23)
        | (mantissa << 20);
    if (exponent == 15 && mantissa == 7) widened = sign | 0x7fc00000u;
    float value;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

__m128i nvfp4_codes(const std::uint8_t* packed) {
    // Signed values scaled by two:
    // {0,.5,1,1.5,2,3,4,6} -> {0,1,2,3,4,6,8,12}.
    const __m128i lut = _mm_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12
    );
    const __m128i bytes = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(packed));
    const __m128i low = _mm_shuffle_epi8(
        lut, _mm_and_si128(bytes, _mm_set1_epi8(15)));
    const __m128i high = _mm_shuffle_epi8(
        lut, _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(15)));
    return _mm_unpacklo_epi64(low, high);
}

float nvfp4_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    for (int block = 0; block < elements / 64; ++block) {
        const auto* base = row_data + block * 36;
        for (int sub = 0; sub < 4; ++sub) {
            const __m128i codes = nvfp4_codes(base + 4 + sub * 8);
            const __m256 factor = _mm256_set1_ps(
                ue4m3_value(base[sub]) * 0.5f);
            const int offset = block * 64 + sub * 16;
            const __m256 low = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(codes));
            const __m256 high = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(_mm_srli_si128(codes, 8)));
            sum0 = _mm256_fmadd_ps(
                _mm256_mul_ps(low, factor),
                _mm256_loadu_ps(input + offset), sum0);
            sum1 = _mm256_fmadd_ps(
                _mm256_mul_ps(high, factor),
                _mm256_loadu_ps(input + offset + 8), sum1);
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

void nvfp4_dot_quad(
    const std::uint8_t* row_data, const float* const inputs[4],
    int elements, float outputs[4]
) {
    __m256 sums[4][2];
    for (auto& pair : sums) for (auto& sum : pair) sum = _mm256_setzero_ps();
    for (int block = 0; block < elements / 64; ++block) {
        const auto* base = row_data + block * 36;
        for (int sub = 0; sub < 4; ++sub) {
            const __m128i codes = nvfp4_codes(base + 4 + sub * 8);
            const __m256 factor = _mm256_set1_ps(
                ue4m3_value(base[sub]) * 0.5f);
            const __m256 low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)), factor);
            const __m256 high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(codes, 8))),
                factor);
            const int offset = block * 64 + sub * 16;
            for (int token = 0; token < 4; ++token) {
                sums[token][0] = _mm256_fmadd_ps(
                    low, _mm256_loadu_ps(inputs[token] + offset),
                    sums[token][0]);
                sums[token][1] = _mm256_fmadd_ps(
                    high, _mm256_loadu_ps(inputs[token] + offset + 8),
                    sums[token][1]);
            }
        }
    }
    for (int token = 0; token < 4; ++token)
        outputs[token] = horizontal_sum(
            _mm256_add_ps(sums[token][0], sums[token][1]));
}

void nvfp4_dequant(const std::uint8_t* row_data, float* output, int elements) {
    for (int block = 0; block < elements / 64; ++block) {
        const auto* base = row_data + block * 36;
        for (int sub = 0; sub < 4; ++sub) {
            const __m128i codes = nvfp4_codes(base + 4 + sub * 8);
            const __m256 factor = _mm256_set1_ps(
                ue4m3_value(base[sub]) * 0.5f);
            const int offset = block * 64 + sub * 16;
            _mm256_storeu_ps(
                output + offset,
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)), factor));
            _mm256_storeu_ps(
                output + offset + 8,
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_srli_si128(codes, 8))), factor));
        }
    }
}

// Q2_K and Q3_K decode 16-element groups; AVX2 covers each as two 8-wide
// halves. One quant load per (half, sub) feeds all four bit-offset groups.
float q2_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i two_bit_mask = _mm_set1_epi8(3);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ2KBlockBytes;
        const float d = half_value(base + 80), dmin = half_value(base + 82);
        const auto* quants = base + 16;
        const auto* vector = input + block * 256;
        for (int half = 0; half < 2; ++half) {
            for (int sub = 0; sub < 2; ++sub) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(quants + half * 32 + sub * 16));
                for (int group = 0; group < 4; ++group) {
                    const auto scale_byte = base[half * 8 + group * 2 + sub];
                    const __m256 ds = _mm256_set1_ps(d * (scale_byte & 15));
                    const __m256 dm = _mm256_set1_ps(dmin * (scale_byte >> 4));
                    const __m128i q = _mm_and_si128(
                        _mm_srli_epi16(packed, 2 * group), two_bit_mask);
                    const float* values = vector + half * 128 + group * 32 + sub * 16;
                    sum0 = _mm256_fmadd_ps(
                        _mm256_sub_ps(_mm256_mul_ps(bytes_to_float(q), ds), dm),
                        _mm256_loadu_ps(values), sum0);
                    sum1 = _mm256_fmadd_ps(
                        _mm256_sub_ps(
                            _mm256_mul_ps(bytes_to_float(_mm_srli_si128(q, 8)), ds), dm),
                        _mm256_loadu_ps(values + 8), sum1);
                }
            }
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

float q3_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i two_bit_mask = _mm_set1_epi8(3);
    const __m128i zero = _mm_setzero_si128();
    const __m128i four = _mm_set1_epi8(4);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ3KBlockBytes;
        const float d = half_value(base + 108);
        const auto* quants = base + 32;
        const auto* scales = base + 96;
        const auto* vector = input + block * 256;
        for (int half = 0; half < 2; ++half) {
            for (int sub = 0; sub < 2; ++sub) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(quants + half * 32 + sub * 16));
                const __m128i mask_bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(base + sub * 16));
                for (int group = 0; group < 4; ++group) {
                    const __m256 ds = _mm256_set1_ps(
                        d * (qwen_q3k_scale(scales, half * 8 + group * 2 + sub) - 32));
                    const __m128i low = _mm_and_si128(
                        _mm_srli_epi16(packed, 2 * group), two_bit_mask);
                    // A set mask bit means "do not subtract 4", so comparing
                    // against zero selects the lanes that still owe the offset.
                    const __m128i bit =
                        _mm_set1_epi8(static_cast<char>(1 << (half * 4 + group)));
                    const __m128i owes =
                        _mm_cmpeq_epi8(_mm_and_si128(mask_bytes, bit), zero);
                    const __m128i offset = _mm_and_si128(owes, four);
                    const float* values = vector + half * 128 + group * 32 + sub * 16;
                    sum0 = _mm256_fmadd_ps(
                        _mm256_mul_ps(
                            _mm256_sub_ps(bytes_to_float(low), bytes_to_float(offset)), ds),
                        _mm256_loadu_ps(values), sum0);
                    sum1 = _mm256_fmadd_ps(
                        _mm256_mul_ps(
                            _mm256_sub_ps(bytes_to_float(_mm_srli_si128(low, 8)),
                                          bytes_to_float(_mm_srli_si128(offset, 8))), ds),
                        _mm256_loadu_ps(values + 8), sum1);
                }
            }
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

void q2_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i two_bit_mask = _mm_set1_epi8(3);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ2KBlockBytes;
        const float d = half_value(base + 80), dmin = half_value(base + 82);
        const auto* quants = base + 16;
        float* out = output + block * 256;
        for (int half = 0; half < 2; ++half) {
            for (int sub = 0; sub < 2; ++sub) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(quants + half * 32 + sub * 16));
                for (int group = 0; group < 4; ++group) {
                    const auto scale_byte = base[half * 8 + group * 2 + sub];
                    const __m256 ds = _mm256_set1_ps(d * (scale_byte & 15));
                    const __m256 dm = _mm256_set1_ps(dmin * (scale_byte >> 4));
                    const __m128i q = _mm_and_si128(
                        _mm_srli_epi16(packed, 2 * group), two_bit_mask);
                    float* slot = out + half * 128 + group * 32 + sub * 16;
                    _mm256_storeu_ps(slot,
                        _mm256_sub_ps(_mm256_mul_ps(bytes_to_float(q), ds), dm));
                    _mm256_storeu_ps(slot + 8,
                        _mm256_sub_ps(
                            _mm256_mul_ps(bytes_to_float(_mm_srli_si128(q, 8)), ds), dm));
                }
            }
        }
    }
}

void q3_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i two_bit_mask = _mm_set1_epi8(3);
    const __m128i zero = _mm_setzero_si128();
    const __m128i four = _mm_set1_epi8(4);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ3KBlockBytes;
        const float d = half_value(base + 108);
        const auto* quants = base + 32;
        const auto* scales = base + 96;
        float* out = output + block * 256;
        for (int half = 0; half < 2; ++half) {
            for (int sub = 0; sub < 2; ++sub) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(quants + half * 32 + sub * 16));
                const __m128i mask_bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(base + sub * 16));
                for (int group = 0; group < 4; ++group) {
                    const __m256 ds = _mm256_set1_ps(
                        d * (qwen_q3k_scale(scales, half * 8 + group * 2 + sub) - 32));
                    const __m128i low = _mm_and_si128(
                        _mm_srli_epi16(packed, 2 * group), two_bit_mask);
                    const __m128i bit =
                        _mm_set1_epi8(static_cast<char>(1 << (half * 4 + group)));
                    const __m128i owes =
                        _mm_cmpeq_epi8(_mm_and_si128(mask_bytes, bit), zero);
                    const __m128i offset = _mm_and_si128(owes, four);
                    float* slot = out + half * 128 + group * 32 + sub * 16;
                    _mm256_storeu_ps(slot, _mm256_mul_ps(
                        _mm256_sub_ps(bytes_to_float(low), bytes_to_float(offset)), ds));
                    _mm256_storeu_ps(slot + 8, _mm256_mul_ps(
                        _mm256_sub_ps(bytes_to_float(_mm_srli_si128(low, 8)),
                                      bytes_to_float(_mm_srli_si128(offset, 8))), ds));
                }
            }
        }
    }
}

float q4_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 144;
        const float d = half_value(base), dmin = half_value(base + 2);
        const auto* scales = base + 4;
        const auto* quants = base + 16;
        const auto* vector = input + block * 256;
        for (int group = 0; group < 4; ++group) for (int sub = 0; sub < 2; ++sub) {
            const int index = group * 2 + sub;
            int scale, minimum;
            if (index < 4) {
                scale = scales[index] & 63;
                minimum = scales[index + 4] & 63;
            } else {
                scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
                minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
            }
            const __m256 ds = _mm256_set1_ps(d * scale);
            const __m256 dm = _mm256_set1_ps(dmin * minimum);
            const int offset = block * 256 + group * 64 + sub * 32;
            for (int lanes = 0; lanes < 32; lanes += 8) {
                __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                    quants + group * 32 + lanes));
                q = sub == 0 ? _mm_and_si128(q, nibble_mask)
                             : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                const __m256 weights = _mm256_sub_ps(
                    _mm256_mul_ps(bytes_to_float(q), ds), dm);
                const __m256 values = _mm256_loadu_ps(input + offset + lanes);
                if ((lanes & 8) == 0) sum0 = _mm256_fmadd_ps(weights, values, sum0);
                else sum1 = _mm256_fmadd_ps(weights, values, sum1);
            }
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

template<int Count>
void q4_dot_rows(
    const std::uint8_t* row_data, std::uint64_t row_bytes,
    const float* input, int elements, float* outputs
) {
    __m256 sums[Count][2];
    for (auto& pair : sums) for (auto& sum : pair) sum = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    for (int block = 0; block < elements / 256; ++block) {
        const std::uint8_t* bases[Count];
        const std::uint8_t* scales[Count];
        const std::uint8_t* quants[Count];
        float d[Count], dmin[Count];
        for (int row = 0; row < Count; ++row) {
            bases[row] = row_data + static_cast<std::uint64_t>(row) * row_bytes
                + static_cast<std::uint64_t>(block) * 144;
            d[row] = half_value(bases[row]);
            dmin[row] = half_value(bases[row] + 2);
            scales[row] = bases[row] + 4;
            quants[row] = bases[row] + 16;
        }
        for (int group = 0; group < 4; ++group) for (int sub = 0; sub < 2; ++sub) {
            const int index = group * 2 + sub;
            __m256 ds[Count], dm[Count];
            for (int row = 0; row < Count; ++row) {
                int scale, minimum;
                if (index < 4) {
                    scale = scales[row][index] & 63;
                    minimum = scales[row][index + 4] & 63;
                } else {
                    scale = (scales[row][index + 4] & 15)
                        | ((scales[row][index - 4] >> 6) << 4);
                    minimum = (scales[row][index + 4] >> 4)
                        | ((scales[row][index] >> 6) << 4);
                }
                ds[row] = _mm256_set1_ps(d[row] * scale);
                dm[row] = _mm256_set1_ps(dmin[row] * minimum);
            }
            const int offset = block * 256 + group * 64 + sub * 32;
            for (int lanes = 0; lanes < 32; lanes += 8) {
                const __m256 values = _mm256_loadu_ps(input + offset + lanes);
                const int accumulator = (lanes & 8) / 8;
                for (int row = 0; row < Count; ++row) {
                    __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                        quants[row] + group * 32 + lanes));
                    q = sub == 0 ? _mm_and_si128(q, nibble_mask)
                                 : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    const __m256 weights = _mm256_sub_ps(
                        _mm256_mul_ps(bytes_to_float(q), ds[row]), dm[row]);
                    sums[row][accumulator] = _mm256_fmadd_ps(
                        weights, values, sums[row][accumulator]);
                }
            }
        }
    }
    for (int row = 0; row < Count; ++row)
        outputs[row] = horizontal_sum(_mm256_add_ps(sums[row][0], sums[row][1]));
}

float q5_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i bit_mask = _mm_set1_epi8(1);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 176;
        const float d = half_value(base), dmin = half_value(base + 2);
        const auto* scales = base + 4;
        const auto* high = base + 16;
        const auto* low = base + 48;
        const auto* vector = input + block * 256;
        for (int group = 0; group < 4; ++group) {
            for (int sub = 0; sub < 2; ++sub) {
                const int index = group * 2 + sub;
                int scale, minimum;
                if (index < 4) {
                    scale = scales[index] & 63;
                    minimum = scales[index + 4] & 63;
                } else {
                    scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
                    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
                }
                const __m256 ds = _mm256_set1_ps(d * scale);
                const __m256 dm = _mm256_set1_ps(dmin * minimum);
                const int shift = 2 * group + sub;
                for (int lanes = 0; lanes < 32; lanes += 8) {
                    __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                        low + group * 32 + lanes));
                    q = sub == 0 ? _mm_and_si128(q, nibble_mask)
                                 : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i bits = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high + lanes));
                    bits = _mm_and_si128(_mm_srli_epi16(bits, shift), bit_mask);
                    q = _mm_add_epi8(q, _mm_slli_epi16(bits, 4));
                    const __m256 weights = _mm256_sub_ps(
                        _mm256_mul_ps(bytes_to_float(q), ds), dm);
                    const __m256 values = _mm256_loadu_ps(
                        vector + group * 64 + sub * 32 + lanes);
                    if ((lanes & 8) == 0) sum0 = _mm256_fmadd_ps(weights, values, sum0);
                    else sum1 = _mm256_fmadd_ps(weights, values, sum1);
                }
            }
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

float q6_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i high_mask = _mm_set1_epi8(3);
    const __m256 offset = _mm256_set1_ps(32.0f);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 210;
        const auto* ql = base;
        const auto* qh = base + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(base + 192);
        const float d = half_value(base + 208);
        const auto* vector = input + block * 256;
        for (int half = 0; half < 2; ++half) {
            for (int segment = 0; segment < 4; ++segment) {
                const int q_offset = (segment == 0 || segment == 2) ? 0 : 32;
                for (int lanes = 0; lanes < 32; lanes += 8) {
                    __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                        ql + half * 64 + q_offset + lanes));
                    q = segment < 2 ? _mm_and_si128(q, nibble_mask)
                                    : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                        qh + half * 32 + lanes));
                    high = _mm_and_si128(_mm_srli_epi16(high, segment * 2), high_mask);
                    q = _mm_add_epi8(q, _mm_slli_epi16(high, 4));
                    const int scale_index = half * 8 + lanes / 16 + segment * 2;
                    const __m256 factor = _mm256_set1_ps(d * scales[scale_index]);
                    const __m256 weights = _mm256_mul_ps(
                        _mm256_sub_ps(bytes_to_float(q), offset), factor);
                    const __m256 values = _mm256_loadu_ps(
                        vector + half * 128 + segment * 32 + lanes);
                    if ((lanes & 8) == 0) sum0 = _mm256_fmadd_ps(weights, values, sum0);
                    else sum1 = _mm256_fmadd_ps(weights, values, sum1);
                }
            }
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

float q8_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base = row_data + block * 34;
        const __m256 scale = _mm256_set1_ps(half_value(base));
        for (int lanes = 0; lanes < 32; lanes += 8) {
            const __m128i bytes = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(base + 2 + lanes));
            const __m256 q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            const __m256 values = _mm256_loadu_ps(input + block * 32 + lanes);
            if ((lanes & 8) == 0) sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), values, sum0);
            else sum1 = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), values, sum1);
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

// Q4_0: 18-byte blocks of one f16 scale and 16 packed nibbles, split-half like
// Q4_K -- byte j holds element j in the low nibble and element j+16 in the
// high. Gemma 4's experts ship in this type, and its 704-wide intermediate is
// not a multiple of the K-quant super-block, so this gates on 32 elements.
float q40_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i center = _mm_set1_epi8(8);
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base = row_data + block * 18;
        const __m256 scale = _mm256_set1_ps(half_value(base));
        const __m128i bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(base + 2));
        const __m128i low = _mm_sub_epi8(
            _mm_and_si128(bytes, nibble_mask), center);
        const __m128i high = _mm_sub_epi8(
            _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask), center);
        const __m128i quarters[4] = {
            low, _mm_srli_si128(low, 8), high, _mm_srli_si128(high, 8)};
        for (int part = 0; part < 4; ++part) {
            const __m256 q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(quarters[part]));
            const __m256 values = _mm256_loadu_ps(input + block * 32 + part * 8);
            if (part & 1) sum1 = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), values, sum1);
            else sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), values, sum0);
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

void q4_dot_quad(const std::uint8_t*row_data,const float*const inputs[4],int elements,float outputs[4]){
    __m256 sums[4][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm256_setzero_ps();const __m128i nibble_mask=_mm_set1_epi8(15);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*144;const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;const auto*quants=base+16;
        for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){const int index=group*2+sub;int scale,minimum;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}const __m256 ds=_mm256_set1_ps(d*scale),dm=_mm256_set1_ps(dmin*minimum);const int offset=block*256+group*64+sub*32;
            for(int lanes=0;lanes<32;lanes+=8){__m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants+group*32+lanes));q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);const __m256 weights=_mm256_sub_ps(_mm256_mul_ps(bytes_to_float(q),ds),dm);for(int token=0;token<4;++token)sums[token][(lanes&8)/8]=_mm256_fmadd_ps(weights,_mm256_loadu_ps(inputs[token]+offset+lanes),sums[token][(lanes&8)/8]);}
        }
    }
    for(int token=0;token<4;++token)outputs[token]=horizontal_sum(_mm256_add_ps(sums[token][0],sums[token][1]));
}

void q5_dot_quad(const std::uint8_t*row_data,const float*const inputs[4],int elements,float outputs[4]){
    __m256 sums[4][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm256_setzero_ps();const __m128i nibble_mask=_mm_set1_epi8(15),bit_mask=_mm_set1_epi8(1);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*176;const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;const auto*high=base+16;const auto*low=base+48;
        for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){const int index=group*2+sub;int scale,minimum;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}const __m256 ds=_mm256_set1_ps(d*scale),dm=_mm256_set1_ps(dmin*minimum);const int shift=2*group+sub,offset=block*256+group*64+sub*32;
            for(int lanes=0;lanes<32;lanes+=8){__m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(low+group*32+lanes));q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i bits=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(high+lanes));bits=_mm_and_si128(_mm_srli_epi16(bits,shift),bit_mask);q=_mm_add_epi8(q,_mm_slli_epi16(bits,4));const __m256 weights=_mm256_sub_ps(_mm256_mul_ps(bytes_to_float(q),ds),dm);for(int token=0;token<4;++token)sums[token][(lanes&8)/8]=_mm256_fmadd_ps(weights,_mm256_loadu_ps(inputs[token]+offset+lanes),sums[token][(lanes&8)/8]);}
        }
    }
    for(int token=0;token<4;++token)outputs[token]=horizontal_sum(_mm256_add_ps(sums[token][0],sums[token][1]));
}

void q6_dot_quad(const std::uint8_t*row_data,const float*const inputs[4],int elements,float outputs[4]){
    __m256 sums[4][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm256_setzero_ps();const __m128i nibble_mask=_mm_set1_epi8(15),high_mask=_mm_set1_epi8(3);const __m256 offset32=_mm256_set1_ps(32.0f);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*210;const auto*ql=base;const auto*qh=base+128;const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float d=half_value(base+208);
        for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){const int q_offset=(segment==0||segment==2)?0:32;
            for(int lanes=0;lanes<32;lanes+=8){__m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql+half*64+q_offset+lanes));q=segment<2?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i high=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh+half*32+lanes));high=_mm_and_si128(_mm_srli_epi16(high,segment*2),high_mask);q=_mm_add_epi8(q,_mm_slli_epi16(high,4));const int scale_index=half*8+lanes/16+segment*2,index=block*256+half*128+segment*32+lanes;const __m256 weights=_mm256_mul_ps(_mm256_sub_ps(bytes_to_float(q),offset32),_mm256_set1_ps(d*scales[scale_index]));for(int token=0;token<4;++token)sums[token][(lanes&8)/8]=_mm256_fmadd_ps(weights,_mm256_loadu_ps(inputs[token]+index),sums[token][(lanes&8)/8]);}
        }
    }
    for(int token=0;token<4;++token)outputs[token]=horizontal_sum(_mm256_add_ps(sums[token][0],sums[token][1]));
}

void q8_dot_quad(const std::uint8_t*row_data,const float*const inputs[4],int elements,float outputs[4]){
    __m256 sums[4][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm256_setzero_ps();
    for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const __m256 scale=_mm256_set1_ps(half_value(base));for(int lanes=0;lanes<32;lanes+=8){const __m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(base+2+lanes));const __m256 weights=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),scale);for(int token=0;token<4;++token)sums[token][(lanes&8)/8]=_mm256_fmadd_ps(weights,_mm256_loadu_ps(inputs[token]+block*32+lanes),sums[token][(lanes&8)/8]);}}
    for(int token=0;token<4;++token)outputs[token]=horizontal_sum(_mm256_add_ps(sums[token][0],sums[token][1]));
}

void q4_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i nibble_mask = _mm_set1_epi8(15);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 144;
        const float d = half_value(base), dmin = half_value(base + 2);
        const auto* scales = base + 4; const auto* quants = base + 16;
        for (int group = 0; group < 4; ++group) for (int sub = 0; sub < 2; ++sub) {
            const int index=group*2+sub; int scale,minimum;
            if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}
            else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}
            const __m256 ds=_mm256_set1_ps(d*scale),dm=_mm256_set1_ps(dmin*minimum);
            for(int lanes=0;lanes<32;lanes+=8){
                __m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants+group*32+lanes));
                q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);
                _mm256_storeu_ps(output+block*256+group*64+sub*32+lanes,_mm256_sub_ps(_mm256_mul_ps(bytes_to_float(q),ds),dm));
            }
        }
    }
}

void q5_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i nibble_mask = _mm_set1_epi8(15), bit_mask = _mm_set1_epi8(1);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 176;
        const float d = half_value(base), dmin = half_value(base + 2);
        const auto* scales = base + 4; const auto* high = base + 16; const auto* low = base + 48;
        for (int group = 0; group < 4; ++group) for (int sub = 0; sub < 2; ++sub) {
            const int index=group*2+sub; int scale,minimum;
            if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}
            else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}
            const __m256 ds=_mm256_set1_ps(d*scale),dm=_mm256_set1_ps(dmin*minimum);
            const int shift=2*group+sub;
            for(int lanes=0;lanes<32;lanes+=8){
                __m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(low+group*32+lanes));
                q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);
                __m128i bits=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(high+lanes));
                bits=_mm_and_si128(_mm_srli_epi16(bits,shift),bit_mask);q=_mm_add_epi8(q,_mm_slli_epi16(bits,4));
                _mm256_storeu_ps(output+block*256+group*64+sub*32+lanes,_mm256_sub_ps(_mm256_mul_ps(bytes_to_float(q),ds),dm));
            }
        }
    }
}

void q6_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i nibble_mask=_mm_set1_epi8(15),high_mask=_mm_set1_epi8(3);
    const __m256 offset=_mm256_set1_ps(32.0f);
    for(int block=0;block<elements/256;++block){
        const auto*base=row_data+block*210;const auto*ql=base;const auto*qh=base+128;
        const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float d=half_value(base+208);
        for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){
            const int q_offset=(segment==0||segment==2)?0:32;
            for(int lanes=0;lanes<32;lanes+=8){
                __m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql+half*64+q_offset+lanes));
                q=segment<2?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);
                __m128i high=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh+half*32+lanes));
                high=_mm_and_si128(_mm_srli_epi16(high,segment*2),high_mask);q=_mm_add_epi8(q,_mm_slli_epi16(high,4));
                const int scale_index=half*8+lanes/16+segment*2;const __m256 factor=_mm256_set1_ps(d*scales[scale_index]);
                _mm256_storeu_ps(output+block*256+half*128+segment*32+lanes,_mm256_mul_ps(_mm256_sub_ps(bytes_to_float(q),offset),factor));
            }
        }
    }
}

void q8_dequant(const std::uint8_t* row_data, float* output, int elements) {
    for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const __m256 scale=_mm256_set1_ps(half_value(base));for(int lanes=0;lanes<32;lanes+=8){const __m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(base+2+lanes));_mm256_storeu_ps(output+block*32+lanes,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),scale));}}
}

// IQ2_XS: 74 bytes per 256 values -> d(2) then sixteen groups of two 16-bit
// entries, then eight scale bytes holding a nibble per group. Each entry is a
// 9-bit grid index plus a 7-bit sign index covering eight outputs.
float iq2xs_dot(const std::uint8_t* row_data, const float* input, int elements) {
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2xsBlockBytes;
        const float* vector = input + block * 256;
        // The block scale factors out of the whole accumulation, so the group
        // scale is all that has to ride on the individual weights.
        __m256 accumulator = _mm256_setzero_ps();
        for (int group = 0; group < 16; ++group) {
            const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
            const __m256 weight = _mm256_set1_ps((0.5f + scale) * 0.25f);
            for (int half = 0; half < 2; ++half) {
                std::uint16_t entry = 0;
                std::memcpy(&entry, base + 2 + (group * 2 + half) * 2, 2);
                const __m256 magnitudes = iq_signed_octet(
                    kIq2xsGrid[entry & 511], kIq2xxsSigns[entry >> 9]);
                accumulator = _mm256_fmadd_ps(
                    _mm256_mul_ps(magnitudes, weight),
                    _mm256_loadu_ps(vector + group * 16 + half * 8),
                    accumulator);
            }
        }
        result += half_value(base) * horizontal_sum(accumulator);
    }
    return result;
}

// IQ2_S: 82 bytes per 256 values -> d(2) qs[32] signs[32] qh[8] scales[8].
// The same sixteen groups of two octets as IQ2_XS, and the same offset-and-
// scale nibble per group, with two differences: the grid index takes two high
// bits from qh, and the signs are stored literally rather than through the
// 7-bit sign codebook -- so the sign byte feeds iq_signed_octet unchanged.
//
// Without this the format fell to the scalar loop, which redoes the grid
// lookup and the sign test for every individual weight. That made a spilled
// IQ2_S block cost roughly an order of magnitude more on the host than a
// K-quant one, so a checkpoint whose feed-forward is IQ2_S was slower than a
// far larger K-quant that spilled twenty times as much.
float iq2s_dot(const std::uint8_t* row_data, const float* input, int elements) {
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2sBlockBytes;
        const auto* quants = base + 2;
        const auto* signs = base + 34;
        const auto* high = base + 66;
        const auto* scales = base + 74;
        const float* vector = input + block * 256;
        // The block scale factors out of the whole accumulation, so only the
        // group scale rides on the individual weights.
        __m256 accumulator = _mm256_setzero_ps();
        for (int group = 0; group < 16; ++group) {
            const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
            const __m256 weight = _mm256_set1_ps((0.5f + scale) * 0.25f);
            for (int half = 0; half < 2; ++half) {
                const int index = group * 2 + half;
                const int entry = quants[index] |
                    (((high[index >> 2] >> (2 * (index & 3))) & 3) << 8);
                const __m256 magnitudes =
                    iq_signed_octet(kIq2sGrid[entry], signs[index]);
                accumulator = _mm256_fmadd_ps(
                    _mm256_mul_ps(magnitudes, weight),
                    _mm256_loadu_ps(vector + group * 16 + half * 8),
                    accumulator);
            }
        }
        result += half_value(base) * horizontal_sum(accumulator);
    }
    return result;
}

// IQ3_XXS: d(2), 64 index bytes, then eight 32-bit auxiliaries. Each auxiliary
// carries a 4-bit group scale in its top nibble and four 7-bit sign indices.
// Two 32-bit grid entries supply the eight magnitudes one sign index covers.
// IQ2_XXS: d(2) qs[64]. Each 8-byte group packs four 8-bit codebook indices in
// its low word and four 7-bit sign selectors plus the group scale in its high
// word. A grid entry is already eight magnitudes, so one lookup and one sign
// byte cover a whole octet -- the scalar form redoes both per weight, which is
// what makes a spilled IQ2_XXS block ~30x slower on the host than a K-quant one.
float iq2xxs_dot(const std::uint8_t* row_data, const float* input, int elements) {
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2xxsBlockBytes;
        const float* vector = input + block * 256;
        __m256 accumulator = _mm256_setzero_ps();
        for (int group = 0; group < 8; ++group) {
            std::uint32_t low = 0, high = 0;
            std::memcpy(&low, base + 2 + group * 8, 4);
            std::memcpy(&high, base + 2 + group * 8 + 4, 4);
            const __m256 weight = _mm256_set1_ps((0.5f + (high >> 28)) * 0.25f);
            for (int quad = 0; quad < 4; ++quad) {
                std::uint64_t grid = 0;
                std::memcpy(&grid, kIq2xxsGrid[(low >> (8 * quad)) & 255], 8);
                const __m256 magnitudes = iq_signed_octet(
                    grid, kIq2xxsSigns[(high >> (7 * quad)) & 127]);
                accumulator = _mm256_fmadd_ps(
                    _mm256_mul_ps(magnitudes, weight),
                    _mm256_loadu_ps(vector + group * 32 + quad * 8),
                    accumulator);
            }
        }
        result += half_value(base) * horizontal_sum(accumulator);
    }
    return result;
}

float iq3xxs_dot(const std::uint8_t* row_data, const float* input, int elements) {
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq3xxsBlockBytes;
        const float* vector = input + block * 256;
        __m256 accumulator = _mm256_setzero_ps();
        for (int group = 0; group < 8; ++group) {
            std::uint32_t aux = 0;
            std::memcpy(&aux, base + 2 + 64 + group * 4, 4);
            const __m256 weight = _mm256_set1_ps((0.5f + (aux >> 28)) * 0.5f);
            for (int quad = 0; quad < 4; ++quad) {
                const auto* indices = base + 2 + group * 8 + quad * 2;
                // The quad's eight magnitudes are two 4-byte grid entries laid
                // end to end, which is exactly one signed octet.
                const std::uint64_t grid =
                    static_cast<std::uint64_t>(kIq3xxsGrid[indices[0]]) |
                    (static_cast<std::uint64_t>(kIq3xxsGrid[indices[1]]) << 32);
                const __m256 magnitudes = iq_signed_octet(
                    grid, kIq2xxsSigns[(aux >> (7 * quad)) & 127]);
                accumulator = _mm256_fmadd_ps(
                    _mm256_mul_ps(magnitudes, weight),
                    _mm256_loadu_ps(vector + group * 32 + quad * 8),
                    accumulator);
            }
        }
        result += half_value(base) * horizontal_sum(accumulator);
    }
    return result;
}

// IQ4_XS: d(2) scales_h(2) scales_l[4] qs[128]. Not a pattern codebook -- every
// 4-bit code indexes the sixteen IQ4_NL levels, so a byte shuffle is the whole
// decode, and each 32-value sub-block carries a 6-bit signed scale split across
// scales_l and scales_h.
float iq4xs_dot(const std::uint8_t* row_data, const float* input, int elements) {
    const __m128i levels =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(kIq4nlValues));
    const __m128i low_nibble = _mm_set1_epi8(15);
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq4xsBlockBytes;
        std::uint16_t scales_high = 0;
        std::memcpy(&scales_high, base + 2, 2);
        const float d = half_value(base);
        const float* vector = input + block * 256;
        for (int sub = 0; sub < 8; ++sub) {
            const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
            const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
            const __m128i quants = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(base + 8 + sub * 16));
            // Low nibbles feed the sub-block's first sixteen values, high
            // nibbles the second sixteen.
            const __m128i first =
                _mm_shuffle_epi8(levels, _mm_and_si128(quants, low_nibble));
            const __m128i second = _mm_shuffle_epi8(
                levels, _mm_and_si128(_mm_srli_epi16(quants, 4), low_nibble));
            const float* values = vector + sub * 32;
            __m256 accumulator = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first)),
                _mm256_loadu_ps(values));
            accumulator = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(first, 8))),
                _mm256_loadu_ps(values + 8), accumulator);
            accumulator = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second)),
                _mm256_loadu_ps(values + 16), accumulator);
            accumulator = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(second, 8))),
                _mm256_loadu_ps(values + 24), accumulator);
            result += d * scale * horizontal_sum(accumulator);
        }
    }
    return result;
}

// One weight row against several activation vectors. Decoding an IQ block is
// far more expensive than the multiply it feeds, so amortizing that decode
// across the tokens routed to the same expert is worth more here than in the
// k-quant formats. The block scale is folded into the group weight so a single
// accumulator per token carries the whole row.
template <int kTokens>
void iq2xs_dot_multi(const std::uint8_t* row_data, const float* const inputs[kTokens],
                     int elements, float outputs[kTokens]) {
    __m256 sums[kTokens];
    for (auto& sum : sums) sum = _mm256_setzero_ps();
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2xsBlockBytes;
        const float d = half_value(base);
        for (int group = 0; group < 16; ++group) {
            const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
            const __m256 weight = _mm256_set1_ps(d * (0.5f + scale) * 0.25f);
            for (int half = 0; half < 2; ++half) {
                std::uint16_t entry = 0;
                std::memcpy(&entry, base + 2 + (group * 2 + half) * 2, 2);
                const __m256 magnitudes = _mm256_mul_ps(
                    iq_signed_octet(kIq2xsGrid[entry & 511], kIq2xxsSigns[entry >> 9]),
                    weight);
                const int index = block * 256 + group * 16 + half * 8;
                for (int token = 0; token < kTokens; ++token)
                    sums[token] = _mm256_fmadd_ps(
                        magnitudes, _mm256_loadu_ps(inputs[token] + index), sums[token]);
            }
        }
    }
    for (int token = 0; token < kTokens; ++token)
        outputs[token] = horizontal_sum(sums[token]);
}

template <int kTokens>
void iq3xxs_dot_multi(const std::uint8_t* row_data, const float* const inputs[kTokens],
                      int elements, float outputs[kTokens]) {
    __m256 sums[kTokens];
    for (auto& sum : sums) sum = _mm256_setzero_ps();
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq3xxsBlockBytes;
        const float d = half_value(base);
        for (int group = 0; group < 8; ++group) {
            std::uint32_t aux = 0;
            std::memcpy(&aux, base + 2 + 64 + group * 4, 4);
            const __m256 weight = _mm256_set1_ps(d * (0.5f + (aux >> 28)) * 0.5f);
            for (int quad = 0; quad < 4; ++quad) {
                const auto* indices = base + 2 + group * 8 + quad * 2;
                const std::uint64_t grid =
                    static_cast<std::uint64_t>(kIq3xxsGrid[indices[0]]) |
                    (static_cast<std::uint64_t>(kIq3xxsGrid[indices[1]]) << 32);
                const __m256 magnitudes = _mm256_mul_ps(
                    iq_signed_octet(grid, kIq2xxsSigns[(aux >> (7 * quad)) & 127]),
                    weight);
                const int index = block * 256 + group * 32 + quad * 8;
                for (int token = 0; token < kTokens; ++token)
                    sums[token] = _mm256_fmadd_ps(
                        magnitudes, _mm256_loadu_ps(inputs[token] + index), sums[token]);
            }
        }
    }
    for (int token = 0; token < kTokens; ++token)
        outputs[token] = horizontal_sum(sums[token]);
}

template <int kTokens>
void iq4xs_dot_multi(const std::uint8_t* row_data, const float* const inputs[kTokens],
                     int elements, float outputs[kTokens]) {
    const __m128i levels =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(kIq4nlValues));
    const __m128i low_nibble = _mm_set1_epi8(15);
    __m256 sums[kTokens];
    for (auto& sum : sums) sum = _mm256_setzero_ps();
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq4xsBlockBytes;
        std::uint16_t scales_high = 0;
        std::memcpy(&scales_high, base + 2, 2);
        const float d = half_value(base);
        for (int sub = 0; sub < 8; ++sub) {
            const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
            const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
            const __m256 weight = _mm256_set1_ps(d * scale);
            const __m128i quants = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(base + 8 + sub * 16));
            const __m128i first =
                _mm_shuffle_epi8(levels, _mm_and_si128(quants, low_nibble));
            const __m128i second = _mm_shuffle_epi8(
                levels, _mm_and_si128(_mm_srli_epi16(quants, 4), low_nibble));
            const __m256 decoded[4] = {
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first)), weight),
                _mm256_mul_ps(_mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(first, 8))), weight),
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second)), weight),
                _mm256_mul_ps(_mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(second, 8))), weight),
            };
            const int index = block * 256 + sub * 32;
            for (int octet = 0; octet < 4; ++octet)
                for (int token = 0; token < kTokens; ++token)
                    sums[token] = _mm256_fmadd_ps(
                        decoded[octet],
                        _mm256_loadu_ps(inputs[token] + index + octet * 8),
                        sums[token]);
        }
    }
    for (int token = 0; token < kTokens; ++token)
        outputs[token] = horizontal_sum(sums[token]);
}

// Returns false when `type` has no IQ multi-token kernel.
template <int kTokens>
bool iq_dot_multi(const std::uint8_t* packed, std::uint32_t type,
                  const float* const inputs[kTokens], int elements,
                  std::uint64_t row, float outputs[kTokens]) {
    const auto blocks = static_cast<std::uint64_t>(elements / 256);
    switch (type) {
        case 17:
            iq2xs_dot_multi<kTokens>(
                packed + row * blocks * kIq2xsBlockBytes, inputs, elements, outputs);
            return true;
        case 18:
            iq3xxs_dot_multi<kTokens>(
                packed + row * blocks * kIq3xxsBlockBytes, inputs, elements, outputs);
            return true;
        case 23:
            iq4xs_dot_multi<kTokens>(
                packed + row * blocks * kIq4xsBlockBytes, inputs, elements, outputs);
            return true;
        default:
            return false;
    }
}

// f16 and bf16 rows. No blocks, no scales: convert eight weights at a time
// and fmadd. The tails are scalar, so any row length is admissible -- the
// dispatchers rely on that (simd_dot_granule in colibri_v2_bailing.hpp).
float f16_dot(const std::uint8_t* row, const float* input, int elements) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= elements; i += 8) {
        const __m128i bits =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i * 2));
        sum = _mm256_fmadd_ps(_mm256_cvtph_ps(bits),
                              _mm256_loadu_ps(input + i), sum);
    }
    float total = horizontal_sum(sum);
    for (; i < elements; ++i) total += half_value(row + i * 2) * input[i];
    return total;
}

inline __m256 bf16_octet(const std::uint8_t* pointer) {
    const __m128i bits =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer));
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

float bf16_scalar_value(const std::uint8_t* pointer) {
    std::uint16_t bits;
    std::memcpy(&bits, pointer, sizeof(bits));
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

float bf16_dot(const std::uint8_t* row, const float* input, int elements) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= elements; i += 8)
        sum = _mm256_fmadd_ps(bf16_octet(row + i * 2),
                              _mm256_loadu_ps(input + i), sum);
    float total = horizontal_sum(sum);
    for (; i < elements; ++i) total += bf16_scalar_value(row + i * 2) * input[i];
    return total;
}

// The multi-input variants decode each weight octet once and apply it to all
// four activations -- same reason the quantized quad kernels exist.
template <bool kBf16>
void half_dot_quad(const std::uint8_t* row, const float* const inputs[4],
                   int elements, float outputs[4]) {
    __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                      _mm256_setzero_ps(), _mm256_setzero_ps()};
    int i = 0;
    for (; i + 8 <= elements; i += 8) {
        const __m256 weights = kBf16
            ? bf16_octet(row + i * 2)
            : _mm256_cvtph_ps(_mm_loadu_si128(
                  reinterpret_cast<const __m128i*>(row + i * 2)));
        for (int token = 0; token < 4; ++token)
            sums[token] = _mm256_fmadd_ps(
                weights, _mm256_loadu_ps(inputs[token] + i), sums[token]);
    }
    for (int token = 0; token < 4; ++token)
        outputs[token] = horizontal_sum(sums[token]);
    for (; i < elements; ++i) {
        const float weight = kBf16 ? bf16_scalar_value(row + i * 2)
                                   : half_value(row + i * 2);
        for (int token = 0; token < 4; ++token)
            outputs[token] += weight * inputs[token][i];
    }
}

} // namespace

bool qwen_quant_dot_iq_multi_avx2(
    const std::uint8_t* packed, std::uint32_t type, const float* const inputs[],
    int token_count, int elements, std::uint64_t row, float* outputs
) {
    if (elements % 256) return false;
    if (token_count == 4)
        return iq_dot_multi<4>(packed, type, inputs, elements, row, outputs);
    if (token_count == 8)
        return iq_dot_multi<8>(packed, type, inputs, elements, row, outputs);
    return false;
}

float qwen_quant_dot_avx2(const std::uint8_t* packed,std::uint32_t type,const float* input,int elements,std::uint64_t row){
    if(type==1)return f16_dot(packed+row*static_cast<std::uint64_t>(elements)*2,input,elements);
    if(type==30)return bf16_dot(packed+row*static_cast<std::uint64_t>(elements)*2,input,elements);
    if(type==2)return q40_dot(packed+row*static_cast<std::uint64_t>(elements/32)*18,input,elements);
    if(type==16)return iq2xxs_dot(packed+row*static_cast<std::uint64_t>(elements/256)*kIq2xxsBlockBytes,input,elements);
    if(type==17)return iq2xs_dot(packed+row*static_cast<std::uint64_t>(elements/256)*kIq2xsBlockBytes,input,elements);
    if(type==18)return iq3xxs_dot(packed+row*static_cast<std::uint64_t>(elements/256)*kIq3xxsBlockBytes,input,elements);
    if(type==22)return iq2s_dot(packed+row*static_cast<std::uint64_t>(elements/256)*kIq2sBlockBytes,input,elements);
    if(type==23)return iq4xs_dot(packed+row*static_cast<std::uint64_t>(elements/256)*kIq4xsBlockBytes,input,elements);
    if(type==10)return q2_dot(packed+row*static_cast<std::uint64_t>(elements/256)*kQ2KBlockBytes,input,elements);
    if(type==11)return q3_dot(packed+row*static_cast<std::uint64_t>(elements/256)*kQ3KBlockBytes,input,elements);
    if(type==12)return q4_dot(packed+row*static_cast<std::uint64_t>(elements/256)*144,input,elements);
    if(type==13)return q5_dot(packed+row*static_cast<std::uint64_t>(elements/256)*176,input,elements);
    if(type==14)return q6_dot(packed+row*static_cast<std::uint64_t>(elements/256)*210,input,elements);
    if(type==40)return nvfp4_dot(packed+row*static_cast<std::uint64_t>(elements/64)*36,input,elements);
    return q8_dot(packed+row*static_cast<std::uint64_t>(elements/32)*34,input,elements);
}

void qwen_quant_dot_rows_avx2(
    const std::uint8_t* packed, std::uint32_t type, const float* input,
    int elements, std::uint64_t first_row, int row_count, float* outputs
) {
    if (type != 12 || row_count < 1 || row_count > 4) {
        for (int row = 0; row < row_count; ++row)
            outputs[row] = qwen_quant_dot_avx2(
                packed, type, input, elements, first_row + row);
        return;
    }
    const auto row_bytes = static_cast<std::uint64_t>(elements / 256) * 144;
    const auto* first = packed + first_row * row_bytes;
    switch (row_count) {
        case 1: q4_dot_rows<1>(first, row_bytes, input, elements, outputs); break;
        case 2: q4_dot_rows<2>(first, row_bytes, input, elements, outputs); break;
        case 3: q4_dot_rows<3>(first, row_bytes, input, elements, outputs); break;
        case 4: q4_dot_rows<4>(first, row_bytes, input, elements, outputs); break;
    }
}

void qwen_quant_dot_quad_avx2(
    const std::uint8_t*packed,std::uint32_t type,const float*const inputs[4],
    int elements,std::uint64_t row,float outputs[4]
){
    if(type==1)half_dot_quad<false>(packed+row*static_cast<std::uint64_t>(elements)*2,inputs,elements,outputs);
    else if(type==30)half_dot_quad<true>(packed+row*static_cast<std::uint64_t>(elements)*2,inputs,elements,outputs);
    else if(type==12)q4_dot_quad(packed+row*static_cast<std::uint64_t>(elements/256)*144,inputs,elements,outputs);
    else if(type==13)q5_dot_quad(packed+row*static_cast<std::uint64_t>(elements/256)*176,inputs,elements,outputs);
    else if(type==14)q6_dot_quad(packed+row*static_cast<std::uint64_t>(elements/256)*210,inputs,elements,outputs);
    else if(type==40)nvfp4_dot_quad(packed+row*static_cast<std::uint64_t>(elements/64)*36,inputs,elements,outputs);
    else q8_dot_quad(packed+row*static_cast<std::uint64_t>(elements/32)*34,inputs,elements,outputs);
}

void qwen_quantize_q8_k_avx2(
    const float* input, int elements, QwenQ8KBlock* output
) {
    for(int block=0;block<elements/256;++block){
        const float*values=input+block*256;
        __m256 maximum=_mm256_setzero_ps();
        const __m256 sign_mask=_mm256_set1_ps(-0.0f);
        for(int index=0;index<256;index+=8)
            maximum=_mm256_max_ps(maximum,_mm256_andnot_ps(sign_mask,_mm256_loadu_ps(values+index)));
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes,maximum);
        float max_value=0.0f;for(float lane:lanes)max_value=std::max(max_value,lane);
        auto&quantized=output[block];
        if(max_value==0.0f){quantized.scale=0.0f;std::memset(quantized.values,0,sizeof(quantized.values));std::memset(quantized.sums,0,sizeof(quantized.sums));continue;}
        quantized.scale=max_value/127.0f;
        const float inverse=1.0f/quantized.scale;
        const __m256 inverse_vector = _mm256_set1_ps(inverse);
        for(int index=0;index<256;index+=16){
            const __m128i packed = quantize_i8_16(values + index, inverse_vector);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(quantized.values + index), packed);
            quantized.sums[index / 16] = sum_i8_16(packed);
        }
    }
}

float qwen_quant_dot_q8_k_avx2(
    const std::uint8_t* packed, std::uint32_t type,
    const QwenQ8KBlock* input, int elements, std::uint64_t row
) {
    float result=0.0f;
    if(type==17){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*kIq2xsBlockBytes;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*kIq2xsBlockBytes;const auto&q8=input[block];
            __m256i sum=_mm256_setzero_si256();
            for(int group=0;group<16;group+=2){
                std::uint16_t codes[4];
                std::memcpy(codes,base+2+group*4,sizeof(codes));
                const __m256i magnitudes=_mm256_set_epi64x(
                    static_cast<long long>(kIq2xsGrid[codes[3]&511]),
                    static_cast<long long>(kIq2xsGrid[codes[2]&511]),
                    static_cast<long long>(kIq2xsGrid[codes[1]&511]),
                    static_cast<long long>(kIq2xsGrid[codes[0]&511]));
                const __m256i signs=_mm256_set_epi64x(
                    iq_sign_bytes(kIq2xxsSigns[codes[3]>>9]),
                    iq_sign_bytes(kIq2xxsSigns[codes[2]>>9]),
                    iq_sign_bytes(kIq2xxsSigns[codes[1]>>9]),
                    iq_sign_bytes(kIq2xxsSigns[codes[0]>>9]));
                const __m256i activation=_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q8.values+group*16));
                const __m256i pairs=_mm256_maddubs_epi16(
                    magnitudes,_mm256_sign_epi8(activation,signs));
                const int first=2*((base[66+(group>>1)]>>(4*(group&1)))&15)+1;
                const int second=2*((base[66+((group+1)>>1)]>>(4*((group+1)&1)))&15)+1;
                const __m256i scales=_mm256_set_m128i(
                    _mm_set1_epi16(static_cast<short>(second)),
                    _mm_set1_epi16(static_cast<short>(first)));
                sum=_mm256_add_epi32(sum,_mm256_madd_epi16(pairs,scales));
            }
            result+=half_value(base)*q8.scale*0.125f*horizontal_sum_i32(sum);
        }
    }else if(type==18){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*kIq3xxsBlockBytes;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*kIq3xxsBlockBytes;const auto&q8=input[block];
            const auto*indices=base+2;const auto*auxiliary=base+66;
            __m256i sum=_mm256_setzero_si256();
            for(int group=0;group<8;++group){
                std::uint32_t aux;
                std::memcpy(&aux,auxiliary+group*4,sizeof(aux));
                const auto*code=indices+group*8;
                const __m256i magnitudes=_mm256_set_epi32(
                    static_cast<int>(kIq3xxsGrid[code[7]]),
                    static_cast<int>(kIq3xxsGrid[code[6]]),
                    static_cast<int>(kIq3xxsGrid[code[5]]),
                    static_cast<int>(kIq3xxsGrid[code[4]]),
                    static_cast<int>(kIq3xxsGrid[code[3]]),
                    static_cast<int>(kIq3xxsGrid[code[2]]),
                    static_cast<int>(kIq3xxsGrid[code[1]]),
                    static_cast<int>(kIq3xxsGrid[code[0]]));
                const __m256i signs=_mm256_set_epi64x(
                    iq_sign_bytes(kIq2xxsSigns[(aux>>21)&127]),
                    iq_sign_bytes(kIq2xxsSigns[(aux>>14)&127]),
                    iq_sign_bytes(kIq2xxsSigns[(aux>>7)&127]),
                    iq_sign_bytes(kIq2xxsSigns[aux&127]));
                const __m256i activation=_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q8.values+group*32));
                const __m256i pairs=_mm256_maddubs_epi16(
                    magnitudes,_mm256_sign_epi8(activation,signs));
                const __m256i scales=_mm256_set1_epi16(
                    static_cast<short>(2*(aux>>28)+1));
                sum=_mm256_add_epi32(sum,_mm256_madd_epi16(pairs,scales));
            }
            result+=half_value(base)*q8.scale*0.25f*horizontal_sum_i32(sum);
        }
    }else if(type==13){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*176;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*176;const auto&q8=input[block];
            const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;
            const auto*high=base+16;const auto*low=base+48;
            for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){
                const int index=group*2+sub;int scale,minimum;
                if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}
                else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}
                int dot=0;const int offset=group*64+sub*32;
                for(int lanes=0;lanes<32;lanes+=8){
                    __m128i quant=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(low+group*32+lanes));
                    quant=sub==0?_mm_and_si128(quant,_mm_set1_epi8(15)):_mm_and_si128(_mm_srli_epi16(quant,4),_mm_set1_epi8(15));
                    __m128i bits=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(high+lanes));
                    bits=_mm_and_si128(_mm_srli_epi16(bits,2*group+sub),_mm_set1_epi8(1));
                    quant=_mm_add_epi8(quant,_mm_slli_epi16(bits,4));
                    const __m128i activation=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q8.values+offset+lanes));
                    dot+=dot_i8_8(quant,activation);
                }
                const int activation_sum=q8.sums[offset/16]+q8.sums[offset/16+1];
                result+=q8.scale*(d*scale*dot-dmin*minimum*activation_sum);
            }
        }
    }else if(type==14){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*210;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*210;const auto*low=base;const auto*high=base+128;
            const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float factor=half_value(base+208)*input[block].scale;
            for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){
                const int q_offset=(segment==0||segment==2)?0:32;const int output_offset=half*128+segment*32;
                alignas(16) std::int8_t quant[32];
                for(int lane=0;lane<32;++lane){const auto byte=low[half*64+q_offset+lane];const int nibble=segment<2?(byte&15):(byte>>4);quant[lane]=static_cast<std::int8_t>((nibble|(((high[half*32+lane]>>(segment*2))&3)<<4))-32);}
                for(int lanes=0;lanes<32;lanes+=16){const int scale_index=half*8+lanes/16+segment*2;result+=factor*scales[scale_index]*dot_i8_16(quant+lanes,input[block].values+output_offset+lanes);}
            }
        }
    }else{
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/32)*34;
        for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const int super=block/8,offset=(block%8)*32;int dot=0;for(int lanes=0;lanes<32;lanes+=16)dot+=dot_i8_16(reinterpret_cast<const std::int8_t*>(base+2+lanes),input[super].values+offset+lanes);result+=half_value(base)*input[super].scale*dot;}
    }
    return result;
}

void qwen_dequant_row_avx2(const std::uint8_t* packed,std::uint32_t type,int elements,std::uint64_t row,float* output){
    if(type==10)q2_dequant(packed+row*static_cast<std::uint64_t>(elements/256)*kQ2KBlockBytes,output,elements);
    else if(type==11)q3_dequant(packed+row*static_cast<std::uint64_t>(elements/256)*kQ3KBlockBytes,output,elements);
    else if(type==12)q4_dequant(packed+row*static_cast<std::uint64_t>(elements/256)*144,output,elements);
    else if(type==13)q5_dequant(packed+row*static_cast<std::uint64_t>(elements/256)*176,output,elements);
    else if(type==14)q6_dequant(packed+row*static_cast<std::uint64_t>(elements/256)*210,output,elements);
    else if(type==17){const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*kIq2xsBlockBytes;for(int i=0;i<elements;++i)output[i]=qwen_iq2xs_value(row_data,i);}
    else if(type==18){const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*kIq3xxsBlockBytes;for(int i=0;i<elements;++i)output[i]=qwen_iq3xxs_value(row_data,i);}
    else if(type==40)nvfp4_dequant(packed+row*static_cast<std::uint64_t>(elements/64)*36,output,elements);
    else q8_dequant(packed+row*static_cast<std::uint64_t>(elements/32)*34,output,elements);
}

void qwen_f32_dot_multi_avx2(const float* row,const float*const*inputs,int count,int elements,float*outputs){
    for(int token=0;token<count;++token){__m256 sum0=_mm256_setzero_ps(),sum1=_mm256_setzero_ps();for(int index=0;index<elements;index+=16){sum0=_mm256_fmadd_ps(_mm256_loadu_ps(row+index),_mm256_loadu_ps(inputs[token]+index),sum0);sum1=_mm256_fmadd_ps(_mm256_loadu_ps(row+index+8),_mm256_loadu_ps(inputs[token]+index+8),sum1);}outputs[token]=horizontal_sum(_mm256_add_ps(sum0,sum1));}
}

void qwen_f32_gemm_rows_avx2(const float*weights,int mr,const float*const*inputs,int count,int elements,float*out){
    for(int row=0;row<mr;++row)qwen_f32_dot_multi_avx2(weights+static_cast<std::size_t>(row)*elements,inputs,count,elements,out+static_cast<std::size_t>(row)*count);
}
