#pragma once

// Scalar element decoders for the low-bit K-quant super-block formats.
//
// These live in a header so the runtime dispatch, the SIMD kernels and the
// contract tests all decode from one definition: the bit layouts are fiddly
// enough that two copies would drift.

#include <cstdint>
#include <cstring>

#include "qwen_iq_tables.h"

inline float qwen_half_value(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1fu, fraction = bits & 0x3ffu, result = 0;
    if (exponent == 0) {
        if (!fraction) result = sign;
        else {
            exponent = 1;
            while ((fraction & 0x400u) == 0) { fraction <<= 1; --exponent; }
            result = sign | ((exponent + 112) << 23) | ((fraction & 0x3ffu) << 13);
        }
    } else if (exponent == 31) result = sign | 0x7f800000u | (fraction << 13);
    else result = sign | ((exponent + 112) << 23) | (fraction << 13);
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

// Q2_K (GGML type 10): 84 bytes per 256 values -> scales[16] qs[64] d(2) dmin(2).
// Each scales byte packs a 4-bit scale (low nibble) and a 4-bit min (high
// nibble) governing one 16-element group; value = d*scale*q - dmin*min. The
// 256 values are two 128-element halves, and within a half the 2-bit quants
// for group j sit at bit offset 2*j of that half's 32 qs bytes.
constexpr std::uint32_t kQ2KBlockBytes = 84;

// Q3_K (GGML type 11): 110 bytes per 256 values -> hmask[32] qs[64] scales[12] d(2).
// The quant is a 2-bit low part from qs plus an inverted high bit from hmask
// (a set mask bit means "do not subtract 4"), giving a signed 3-bit value;
// value = d*(scale-32)*q.
constexpr std::uint32_t kQ3KBlockBytes = 110;

// The 16 six-bit Q3_K scales are packed into 12 bytes: the low and high
// nibbles of bytes 0..7 carry the low 4 bits of each scale, and bytes 8..11
// supply the top 2 bits.
inline int qwen_q3k_scale(const std::uint8_t* scales, int index) {
    const int group = index / 4, byte = index & 3;
    const int packed_low = scales[(group & 1) ? 4 + byte : byte];
    const int nibble = (group < 2) ? (packed_low & 15) : (packed_low >> 4);
    return nibble | (((scales[8 + byte] >> (2 * group)) & 3) << 4);
}

inline float qwen_q2k_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kQ2KBlockBytes;
    std::uint16_t d_bits = 0, dmin_bits = 0;
    std::memcpy(&d_bits, base + 80, 2);
    std::memcpy(&dmin_bits, base + 82, 2);
    const int half = within / 128, rest = within & 127;
    const int group = rest / 32, lane = rest & 31, sub = lane / 16, element = lane & 15;
    const int quant = (base[16 + half * 32 + sub * 16 + element] >> (2 * group)) & 3;
    const auto scale_byte = base[half * 8 + group * 2 + sub];
    return qwen_half_value(d_bits) * (scale_byte & 15) * quant
        - qwen_half_value(dmin_bits) * (scale_byte >> 4);
}

// Dot one Q2_K row against `input`. Hoisting the per-group scale and min out
// of the inner loop is why this walks super-blocks rather than reusing
// qwen_q2k_value per element.
inline float qwen_q2k_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kQ2KBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ2KBlockBytes;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, base + 80, 2);
        std::memcpy(&dmin_bits, base + 82, 2);
        const float d = qwen_half_value(d_bits), dmin = qwen_half_value(dmin_bits);
        const float* vector = input + block * 256;
        for (int half = 0; half < 2; ++half)
            for (int group = 0; group < 4; ++group)
                for (int sub = 0; sub < 2; ++sub) {
                    const auto scale_byte = base[half * 8 + group * 2 + sub];
                    const float ds = d * (scale_byte & 15), dm = dmin * (scale_byte >> 4);
                    const auto* quants = base + 16 + half * 32 + sub * 16;
                    const auto* values = vector + half * 128 + group * 32 + sub * 16;
                    for (int lane = 0; lane < 16; ++lane)
                        result += (ds * ((quants[lane] >> (2 * group)) & 3) - dm) * values[lane];
                }
    }
    return result;
}

inline float qwen_q3k_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kQ3KBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base + 108, 2);
    const int half = within / 128, rest = within & 127;
    const int group = rest / 32, lane = rest & 31, sub = lane / 16, element = lane & 15;
    const int low = (base[32 + half * 32 + sub * 16 + element] >> (2 * group)) & 3;
    const int high = (base[lane] & (1 << (half * 4 + group))) ? 0 : 4;
    const int scale = qwen_q3k_scale(base + 96, half * 8 + group * 2 + sub);
    return qwen_half_value(d_bits) * (scale - 32) * (low - high);
}

// Dot one Q3_K row against `input`, accumulating each 16-element group before
// applying its shared scale.
inline float qwen_q3k_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kQ3KBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ3KBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base + 108, 2);
        const float d = qwen_half_value(d_bits);
        const auto* hmask = base;
        const auto* scales = base + 96;
        const float* vector = input + block * 256;
        for (int half = 0; half < 2; ++half)
            for (int group = 0; group < 4; ++group)
                for (int sub = 0; sub < 2; ++sub) {
                    const float ds =
                        d * (qwen_q3k_scale(scales, half * 8 + group * 2 + sub) - 32);
                    const int mask = 1 << (half * 4 + group);
                    const auto* quants = base + 32 + half * 32 + sub * 16;
                    const auto* values = vector + half * 128 + group * 32 + sub * 16;
                    float partial = 0.0f;
                    for (int lane = 0; lane < 16; ++lane) {
                        const int low = (quants[lane] >> (2 * group)) & 3;
                        const int high = (hmask[sub * 16 + lane] & mask) ? 0 : 4;
                        partial += static_cast<float>(low - high) * values[lane];
                    }
                    result += ds * partial;
                }
    }
    return result;
}

// IQ2_XXS (GGML type 16): 66 bytes per 256 values -> d(2) qs[32] as uint16.
// Each 32-element group is two uint32: the low word holds four 8-bit indices
// into the 256-entry grid of 8-value patterns, and the high word packs four
// 7-bit sign selectors (bits 0..27) plus a 4-bit scale in its top nibble.
constexpr std::uint32_t kIq2xxsBlockBytes = 66;

inline float qwen_iq2xxs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq2xxsBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const int group = within / 32, rest = within & 31;
    const int quad = rest / 8, element = rest & 7;
    std::uint32_t low = 0, high = 0;
    std::memcpy(&low, base + 2 + group * 8, 4);
    std::memcpy(&high, base + 2 + group * 8 + 4, 4);
    const float scale = qwen_half_value(d_bits) * (0.5f + (high >> 28)) * 0.25f;
    const std::uint8_t signs = kIq2xxsSigns[(high >> (7 * quad)) & 127];
    const std::uint8_t value = kIq2xxsGrid[(low >> (8 * quad)) & 255][element];
    return (signs >> element) & 1 ? -scale * value : scale * value;
}

// Dot one IQ2_XXS row. Decoding a whole 8-value grid entry at a time keeps the
// table lookup and the sign byte out of the innermost loop.
inline float qwen_iq2xxs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq2xxsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2xxsBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int group = 0; group < 8; ++group) {
            std::uint32_t low = 0, high = 0;
            std::memcpy(&low, base + 2 + group * 8, 4);
            std::memcpy(&high, base + 2 + group * 8 + 4, 4);
            const float scale = d * (0.5f + (high >> 28)) * 0.25f;
            float partial = 0.0f;
            for (int quad = 0; quad < 4; ++quad) {
                const std::uint8_t signs = kIq2xxsSigns[(high >> (7 * quad)) & 127];
                const std::uint8_t* pattern = kIq2xxsGrid[(low >> (8 * quad)) & 255];
                const float* values = vector + group * 32 + quad * 8;
                for (int element = 0; element < 8; ++element) {
                    const float weight = static_cast<float>(pattern[element]);
                    partial += ((signs >> element) & 1 ? -weight : weight) * values[element];
                }
            }
            result += scale * partial;
        }
    }
    return result;
}

// IQ3_XXS (GGML type 18): 98 bytes per 256 values -> d(2) qs[64] scales[32].
// Each 32-element group has one uint32 of scale-and-signs, laid out like
// IQ2_XXS, but the grid entries are only four values wide, so each 8-output
// quad consumes two consecutive qs bytes.
constexpr std::uint32_t kIq3xxsBlockBytes = 98;

inline float qwen_iq3xxs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq3xxsBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const int group = within / 32, rest = within & 31;
    const int quad = rest / 8, element = rest & 7;
    std::uint32_t aux = 0;
    std::memcpy(&aux, base + 2 + 64 + group * 4, 4);
    const float scale = qwen_half_value(d_bits) * (0.5f + (aux >> 28)) * 0.5f;
    const std::uint8_t signs = kIq2xxsSigns[(aux >> (7 * quad)) & 127];
    const std::uint32_t pattern =
        kIq3xxsGrid[base[2 + group * 8 + quad * 2 + (element >> 2)]];
    const float value =
        static_cast<float>((pattern >> (8 * (element & 3))) & 0xffu);
    return (signs >> element) & 1 ? -scale * value : scale * value;
}

inline float qwen_iq3xxs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq3xxsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq3xxsBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int group = 0; group < 8; ++group) {
            std::uint32_t aux = 0;
            std::memcpy(&aux, base + 2 + 64 + group * 4, 4);
            const float scale = d * (0.5f + (aux >> 28)) * 0.5f;
            float partial = 0.0f;
            for (int quad = 0; quad < 4; ++quad) {
                const std::uint8_t signs = kIq2xxsSigns[(aux >> (7 * quad)) & 127];
                const auto* indices = base + 2 + group * 8 + quad * 2;
                const float* values = vector + group * 32 + quad * 8;
                for (int half = 0; half < 2; ++half) {
                    const std::uint32_t pattern = kIq3xxsGrid[indices[half]];
                    for (int element = 0; element < 4; ++element) {
                        const float weight =
                            static_cast<float>((pattern >> (8 * element)) & 0xffu);
                        const int bit = half * 4 + element;
                        partial += ((signs >> bit) & 1 ? -weight : weight)
                            * values[bit];
                    }
                }
            }
            result += scale * partial;
        }
    }
    return result;
}

// IQ2_S (GGML type 22): 82 bytes per 256 values -> d(2) qs[32] signs[32]
// qh[8] scales[8]. Unlike the XXS formats the signs are stored literally, one
// byte per 8 outputs, and the grid index gains two high bits from qh.
constexpr std::uint32_t kIq2sBlockBytes = 82;

inline float qwen_iq2s_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq2sBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const int group = within / 16, rest = within & 15;
    const int half = rest / 8, element = rest & 7;
    const int index = group * 2 + half;
    const auto* quants = base + 2;
    const auto* signs = base + 34;
    const auto* high = base + 66;
    const auto* scales = base + 74;
    const int entry =
        quants[index] | (((high[index >> 2] >> (2 * (index & 3))) & 3) << 8);
    const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
    const float db = qwen_half_value(d_bits) * (0.5f + scale) * 0.25f;
    const float value =
        static_cast<float>((kIq2sGrid[entry] >> (8 * element)) & 0xffull);
    return (signs[index] >> element) & 1 ? -db * value : db * value;
}

inline float qwen_iq2s_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq2sBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2sBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const auto* quants = base + 2;
        const auto* signs = base + 34;
        const auto* high = base + 66;
        const auto* scales = base + 74;
        const float* vector = input + block * 256;
        for (int group = 0; group < 16; ++group) {
            const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
            const float db = d * (0.5f + scale) * 0.25f;
            float partial = 0.0f;
            for (int half = 0; half < 2; ++half) {
                const int index = group * 2 + half;
                const int entry =
                    quants[index] | (((high[index >> 2] >> (2 * (index & 3))) & 3) << 8);
                const std::uint64_t pattern = kIq2sGrid[entry];
                const std::uint8_t sign = signs[index];
                const float* values = vector + group * 16 + half * 8;
                for (int element = 0; element < 8; ++element) {
                    const float weight =
                        static_cast<float>((pattern >> (8 * element)) & 0xffull);
                    partial += ((sign >> element) & 1 ? -weight : weight) * values[element];
                }
            }
            result += db * partial;
        }
    }
    return result;
}

// IQ3_S (GGML type 21): 110 bytes per 256 values -> d(2) qs[64] qh[8]
// signs[32] scales[4]. One extra index bit per grid entry comes from qh, and
// the scale is an odd multiplier rather than an offset-and-scale pair.
constexpr std::uint32_t kIq3sBlockBytes = 110;

inline float qwen_iq3s_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq3sBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const auto* quants = base + 2;
    const auto* high = base + 66;
    const auto* signs = base + 74;
    const auto* scales = base + 106;
    const int index = within / 4, element = within & 3;
    const int entry = quants[index] | (((high[index >> 3] >> (index & 7)) & 1) << 8);
    const int group = within / 32, quad = (within % 32) / 8, bit = within & 7;
    const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
    const float db = qwen_half_value(d_bits) * (1 + 2 * scale);
    const float value =
        static_cast<float>((kIq3sGrid[entry] >> (8 * element)) & 0xffu);
    return (signs[group * 4 + quad] >> bit) & 1 ? -db * value : db * value;
}

inline float qwen_iq3s_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq3sBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq3sBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const auto* quants = base + 2;
        const auto* high = base + 66;
        const auto* signs = base + 74;
        const auto* scales = base + 106;
        const float* vector = input + block * 256;
        for (int group = 0; group < 8; ++group) {
            const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
            const float db = d * (1 + 2 * scale);
            float partial = 0.0f;
            for (int quad = 0; quad < 4; ++quad) {
                const std::uint8_t sign = signs[group * 4 + quad];
                const float* values = vector + group * 32 + quad * 8;
                for (int half = 0; half < 2; ++half) {
                    const int index = group * 8 + quad * 2 + half;
                    const int entry =
                        quants[index] | (((high[index >> 3] >> (index & 7)) & 1) << 8);
                    const std::uint32_t pattern = kIq3sGrid[entry];
                    for (int element = 0; element < 4; ++element) {
                        const float weight =
                            static_cast<float>((pattern >> (8 * element)) & 0xffu);
                        const int bit = half * 4 + element;
                        partial += ((sign >> bit) & 1 ? -weight : weight) * values[bit];
                    }
                }
            }
            result += db * partial;
        }
    }
    return result;
}

// IQ2_XS (GGML type 17): 74 bytes per 256 values -> d(2) qs[32] as uint16
// scales[8]. Each uint16 carries a 9-bit grid index and a 7-bit sign selector.
constexpr std::uint32_t kIq2xsBlockBytes = 74;

inline float qwen_iq2xs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq2xsBlockBytes;
    std::uint16_t d_bits = 0, entry = 0;
    std::memcpy(&d_bits, base, 2);
    const int index = within / 8, element = within & 7;
    std::memcpy(&entry, base + 2 + index * 2, 2);
    const int group = within / 16;
    const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
    const float db = qwen_half_value(d_bits) * (0.5f + scale) * 0.25f;
    const std::uint8_t signs = kIq2xxsSigns[entry >> 9];
    const float value =
        static_cast<float>((kIq2xsGrid[entry & 511] >> (8 * element)) & 0xffull);
    return (signs >> element) & 1 ? -db * value : db * value;
}

inline float qwen_iq2xs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq2xsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2xsBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int group = 0; group < 16; ++group) {
            const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
            const float db = d * (0.5f + scale) * 0.25f;
            float partial = 0.0f;
            for (int half = 0; half < 2; ++half) {
                std::uint16_t entry = 0;
                std::memcpy(&entry, base + 2 + (group * 2 + half) * 2, 2);
                const std::uint64_t pattern = kIq2xsGrid[entry & 511];
                const std::uint8_t signs = kIq2xxsSigns[entry >> 9];
                const float* values = vector + group * 16 + half * 8;
                for (int element = 0; element < 8; ++element) {
                    const float weight =
                        static_cast<float>((pattern >> (8 * element)) & 0xffull);
                    partial += ((signs >> element) & 1 ? -weight : weight) * values[element];
                }
            }
            result += db * partial;
        }
    }
    return result;
}

// IQ4_XS (GGML type 23): 136 bytes per 256 values -> d(2) scales_h(2)
// scales_l[4] qs[128]. Not a codebook of patterns like the other IQ formats:
// each 4-bit code indexes the sixteen non-uniform IQ4_NL levels, and each
// 32-value sub-block has a 6-bit signed scale split across scales_l/scales_h.
constexpr std::uint32_t kIq4xsBlockBytes = 136;

inline float qwen_iq4xs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq4xsBlockBytes;
    std::uint16_t d_bits = 0, scales_high = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&scales_high, base + 2, 2);
    const int sub = within / 32, element = within & 31;
    const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
    const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
    const std::uint8_t byte = base[8 + sub * 16 + (element & 15)];
    const int code = element < 16 ? (byte & 15) : (byte >> 4);
    return qwen_half_value(d_bits) * scale * kIq4nlValues[code];
}

inline float qwen_iq4xs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq4xsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq4xsBlockBytes;
        std::uint16_t d_bits = 0, scales_high = 0;
        std::memcpy(&d_bits, base, 2);
        std::memcpy(&scales_high, base + 2, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int sub = 0; sub < 8; ++sub) {
            const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
            const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
            const auto* quants = base + 8 + sub * 16;
            const float* values = vector + sub * 32;
            float partial = 0.0f;
            for (int element = 0; element < 16; ++element) {
                const std::uint8_t byte = quants[element];
                partial += kIq4nlValues[byte & 15] * values[element]
                    + kIq4nlValues[byte >> 4] * values[element + 16];
            }
            result += d * scale * partial;
        }
    }
    return result;
}
