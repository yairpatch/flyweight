#pragma once

// Asymmetric K8V4 KV-Cache Codec with Sylvester-Hadamard Outlier Suppression.
//
// Keys: 8-bit Q8_0 format (group-32 with FP16 scale, 34 bytes / 32 elements).
//       Rotated by canonical Sylvester-Hadamard matrix H.
//       Exact inner-product invariance: <H q, H k> == <q, k>.
//
// Values: 4-bit NVFP4-G16 (group-16 with FP8 E4M3 scale, 9 bytes / 16 elements)
//         or Turbo4 (group-32 with FP16 scale and Lloyd-Max Gaussian codebook, 18 bytes / 32 elements).
//         Rotated by canonical Sylvester-Hadamard matrix H.
//         Linear value accumulation: sum_i A_i V_i == H (sum_i A_i (H V_i)).
//         Outliers in V are dispersed across all D dimensions by ~1/sqrt(D),
//         preventing 4-bit underflow/collapse.
//
// Memory footprint:
// D=128: Key: 136 B, Value: 72 B  -> Total: 208 B/token/head (vs 512 B FP16, -59.4%)
// D=256: Key: 272 B, Value: 144 B -> Total: 416 B/token/head (vs 1024 B FP16, -59.4%)

#include "hadamard.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace flyweight::k8v4 {

// ============================================================================
// Numeric Conversions: FP16, FP8 (E4M3), FP4 (E2M1)
// ============================================================================

inline float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    std::uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            std::uint32_t shift = 0;
            std::uint32_t value = mantissa;
            while ((value & 0x400u) == 0) { value <<= 1; ++shift; }
            value &= 0x3ffu;
            out = sign | ((127u - 15u - shift + 1u) << 23) | (value << 13);
        }
    } else if (exponent == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

inline std::uint16_t float_to_half_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    }
    if (exponent >= 0x1f) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        const std::uint32_t full = mantissa | 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
        const std::uint32_t rounded =
            (full + (1u << (shift - 1)) - 1u + ((full >> shift) & 1u)) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded =
        (mantissa + 0x00000fffu + ((mantissa >> 13) & 1u)) >> 13;
    return static_cast<std::uint16_t>(
        sign | ((static_cast<std::uint32_t>(exponent) << 10) + rounded));
}

// OCP FP8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. Max finite = 448.0
inline float e4m3_bits_to_float(unsigned char bits) {
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const unsigned int exponent = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u)
        return sign * std::numeric_limits<float>::quiet_NaN();
    if (exponent == 0)
        return sign * std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                             static_cast<int>(exponent) - 7);
}

inline unsigned char float_to_e4m3_bits(float value) {
    if (std::isnan(value)) return 0x7fu;
    const unsigned char sign = std::signbit(value) ? 0x80u : 0x00u;
    const float magnitude = std::fabs(value);
    if (magnitude >= 448.0f) return static_cast<unsigned char>(sign | 0x7eu);

    unsigned char best = 0;
    float best_error = std::fabs(magnitude - e4m3_bits_to_float(0));
    for (unsigned int code = 1; code <= 0x7eu; ++code) {
        const float candidate = e4m3_bits_to_float(static_cast<unsigned char>(code));
        const float error = std::fabs(magnitude - candidate);
        if (error < best_error || (error == best_error && (code & 1u) == 0u)) {
            best_error = error;
            best = static_cast<unsigned char>(code);
        }
    }
    return static_cast<unsigned char>(sign | best);
}

// FP4 E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa -> {0, .5, 1, 1.5, 2, 3, 4, 6}.
inline float e2m1_bits_to_float(unsigned char code) {
    static constexpr float kMagnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float magnitude = kMagnitudes[code & 0x07u];
    return (code & 0x08u) ? -magnitude : magnitude;
}

inline unsigned char float_to_e2m1_bits(float value) {
    const unsigned char sign = std::signbit(value) ? 0x08u : 0x00u;
    const float magnitude = std::isnan(value) ? 0.0f : std::fabs(value);
    unsigned char best = 0;
    float best_error = magnitude;
    for (unsigned int code = 1; code < 8u; ++code) {
        const float candidate = e2m1_bits_to_float(static_cast<unsigned char>(code));
        const float error = std::fabs(magnitude - candidate);
        if (error < best_error || (error == best_error && (code & 1u) == 0u)) {
            best_error = error;
            best = static_cast<unsigned char>(code);
        }
    }
    return static_cast<unsigned char>(sign | best);
}

// Turbo4 Lloyd-Max Optimal Gaussian Codebook (16 levels)
inline constexpr float kTurboCb4[16] = {
    -2.73258956f, -2.06901721f, -1.61804637f, -1.25623118f,
    -0.94234045f, -0.65675911f, -0.38804829f, -0.12839503f,
     0.12839503f,  0.38804829f,  0.65675911f,  0.94234045f,
     1.25623118f,  1.61804637f,  2.06901721f,  2.73258956f
};

// ============================================================================
// Data Layouts
// ============================================================================

#pragma pack(push, 1)
struct Q8_0_Block {
    std::uint16_t scale_fp16; // 2 bytes
    std::int8_t   qs[32];      // 32 bytes
};
static_assert(sizeof(Q8_0_Block) == 34, "Q8_0_Block must be exactly 34 bytes");

struct NVFP4_G16_Block {
    std::uint8_t scale_fp8;   // 1 byte (FP8 E4M3)
    std::uint8_t codes[8];    // 8 bytes (16 4-bit nibbles: low=even, high=odd)
};
static_assert(sizeof(NVFP4_G16_Block) == 9, "NVFP4_G16_Block must be exactly 9 bytes");

struct Turbo4_Block {
    std::uint16_t scale_fp16; // 2 bytes
    std::uint8_t  codes[16];  // 16 bytes (32 4-bit nibbles)
};
static_assert(sizeof(Turbo4_Block) == 18, "Turbo4_Block must be exactly 18 bytes");
#pragma pack(pop)

// ============================================================================
// Key 8-bit (Q8_0) Codec
// ============================================================================

inline void q8_0_quantize_block(const float* src, Q8_0_Block* dst) {
    float max_abs = 0.0f;
    for (int i = 0; i < 32; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }
    if (max_abs == 0.0f) {
        dst->scale_fp16 = 0;
        std::memset(dst->qs, 0, sizeof(dst->qs));
        return;
    }
    const float scale = max_abs / 127.0f;
    dst->scale_fp16 = float_to_half_bits(scale);
    const float rep_scale = half_bits_to_float(dst->scale_fp16);
    const float inv_scale = (rep_scale > 0.0f) ? (1.0f / rep_scale) : 0.0f;
    for (int i = 0; i < 32; ++i) {
        int q = static_cast<int>(std::round(src[i] * inv_scale));
        q = std::max(-127, std::min(127, q));
        dst->qs[i] = static_cast<std::int8_t>(q);
    }
}

inline void q8_0_dequantize_block(const Q8_0_Block* src, float* dst) {
    const float scale = half_bits_to_float(src->scale_fp16);
    for (int i = 0; i < 32; ++i) {
        dst[i] = static_cast<float>(src->qs[i]) * scale;
    }
}

inline void k8_quantize_row(const float* src, std::uint8_t* dst, int dim) {
    const int num_blocks = dim / 32;
    auto* blocks = reinterpret_cast<Q8_0_Block*>(dst);
    for (int b = 0; b < num_blocks; ++b) {
        q8_0_quantize_block(src + b * 32, blocks + b);
    }
}

inline void k8_dequantize_row(const std::uint8_t* src, float* dst, int dim) {
    const int num_blocks = dim / 32;
    const auto* blocks = reinterpret_cast<const Q8_0_Block*>(src);
    for (int b = 0; b < num_blocks; ++b) {
        q8_0_dequantize_block(blocks + b, dst + b * 32);
    }
}

inline void k8_hadamard_quantize_row(const float* src, std::uint8_t* dst, int dim) {
    std::vector<float> rotated(dim);
    hadamard_sylvester_cpu(src, rotated.data(), dim);
    k8_quantize_row(rotated.data(), dst, dim);
}

// ============================================================================
// Value 4-bit (NVFP4-G16) Codec
// ============================================================================

inline void nvfp4_quantize_group16(const float* src, NVFP4_G16_Block* dst) {
    float max_abs = 0.0f;
    for (int i = 0; i < 16; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }
    if (max_abs == 0.0f) {
        dst->scale_fp8 = 0;
        std::memset(dst->codes, 0, sizeof(dst->codes));
        return;
    }
    const float raw_scale = max_abs / 6.0f;
    const float bounded = std::min(448.0f, std::max(0x1p-9f, raw_scale));
    dst->scale_fp8 = float_to_e4m3_bits(bounded);
    const float rep_scale = e4m3_bits_to_float(dst->scale_fp8);
    const float inv_scale = (rep_scale > 0.0f) ? (1.0f / rep_scale) : 0.0f;
    for (int pair = 0; pair < 8; ++pair) {
        const unsigned char c0 = float_to_e2m1_bits(src[2 * pair] * inv_scale);
        const unsigned char c1 = float_to_e2m1_bits(src[2 * pair + 1] * inv_scale);
        dst->codes[pair] = static_cast<std::uint8_t>((c1 << 4) | (c0 & 0x0fu));
    }
}

inline void nvfp4_dequantize_group16(const NVFP4_G16_Block* src, float* dst) {
    const float scale = e4m3_bits_to_float(src->scale_fp8);
    for (int pair = 0; pair < 8; ++pair) {
        const std::uint8_t byte = src->codes[pair];
        dst[2 * pair]     = e2m1_bits_to_float(byte & 0x0fu) * scale;
        dst[2 * pair + 1] = e2m1_bits_to_float((byte >> 4) & 0x0fu) * scale;
    }
}

inline void v4_nvfp4_quantize_row(const float* src, std::uint8_t* dst, int dim) {
    const int num_groups = dim / 16;
    auto* groups = reinterpret_cast<NVFP4_G16_Block*>(dst);
    for (int g = 0; g < num_groups; ++g) {
        nvfp4_quantize_group16(src + g * 16, groups + g);
    }
}

inline void v4_nvfp4_dequantize_row(const std::uint8_t* src, float* dst, int dim) {
    const int num_groups = dim / 16;
    const auto* groups = reinterpret_cast<const NVFP4_G16_Block*>(src);
    for (int g = 0; g < num_groups; ++g) {
        nvfp4_dequantize_group16(groups + g, dst + g * 16);
    }
}

inline void v4_nvfp4_hadamard_quantize_row(const float* src, std::uint8_t* dst, int dim) {
    std::vector<float> rotated(dim);
    hadamard_sylvester_cpu(src, rotated.data(), dim);
    v4_nvfp4_quantize_row(rotated.data(), dst, dim);
}

// ============================================================================
// Value 4-bit (Turbo4) Codec
// ============================================================================

inline void turbo4_quantize_block(const float* src, Turbo4_Block* dst) {
    float energy = 0.0f;
    for (int i = 0; i < 32; ++i) energy += src[i] * src[i];
    const float rms = std::sqrt(energy / 32.0f);
    if (rms <= 0.0f) {
        dst->scale_fp16 = 0;
        std::memset(dst->codes, 0, sizeof(dst->codes));
        return;
    }
    const float inv_rms = 1.0f / rms;
    // Least-squares fit of scale against chosen codebook levels
    float num = 0.0f, den = 0.0f;
    int best_indices[32];
    for (int i = 0; i < 32; ++i) {
        const float x = src[i] * inv_rms;
        int best = 0;
        float best_dist = std::fabs(x - kTurboCb4[0]);
        for (int l = 1; l < 16; ++l) {
            const float d = std::fabs(x - kTurboCb4[l]);
            if (d < best_dist) {
                best_dist = d;
                best = l;
            }
        }
        best_indices[i] = best;
        const float c = kTurboCb4[best];
        num += src[i] * c;
        den += c * c;
    }
    const float optimal_scale = (den > 0.0f) ? (num / den) : rms;
    dst->scale_fp16 = float_to_half_bits(optimal_scale);
    for (int pair = 0; pair < 16; ++pair) {
        const unsigned int c0 = static_cast<unsigned int>(best_indices[2 * pair]);
        const unsigned int c1 = static_cast<unsigned int>(best_indices[2 * pair + 1]);
        dst->codes[pair] = static_cast<std::uint8_t>((c1 << 4) | (c0 & 0x0fu));
    }
}

inline void turbo4_dequantize_block(const Turbo4_Block* src, float* dst) {
    const float scale = half_bits_to_float(src->scale_fp16);
    for (int pair = 0; pair < 16; ++pair) {
        const std::uint8_t byte = src->codes[pair];
        dst[2 * pair]     = kTurboCb4[byte & 0x0fu] * scale;
        dst[2 * pair + 1] = kTurboCb4[(byte >> 4) & 0x0fu] * scale;
    }
}

inline void v4_turbo4_quantize_row(const float* src, std::uint8_t* dst, int dim) {
    const int num_blocks = dim / 32;
    auto* blocks = reinterpret_cast<Turbo4_Block*>(dst);
    for (int b = 0; b < num_blocks; ++b) {
        turbo4_quantize_block(src + b * 32, blocks + b);
    }
}

inline void v4_turbo4_dequantize_row(const std::uint8_t* src, float* dst, int dim) {
    const int num_blocks = dim / 32;
    const auto* blocks = reinterpret_cast<const Turbo4_Block*>(src);
    for (int b = 0; b < num_blocks; ++b) {
        turbo4_dequantize_block(blocks + b, dst + b * 32);
    }
}

inline void v4_turbo4_hadamard_quantize_row(const float* src, std::uint8_t* dst, int dim) {
    std::vector<float> rotated(dim);
    hadamard_sylvester_cpu(src, rotated.data(), dim);
    v4_turbo4_quantize_row(rotated.data(), dst, dim);
}

// ============================================================================
// Full Attention Block Simulation & Quality Metrics
// ============================================================================

enum class V4Mode {
    NVFP4_G16,
    Turbo4
};

// Softmax helper
inline void softmax(float* scores, int length) {
    float max_s = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < length; ++i) max_s = std::max(max_s, scores[i]);
    float sum_exp = 0.0f;
    for (int i = 0; i < length; ++i) {
        scores[i] = std::exp(scores[i] - max_s);
        sum_exp += scores[i];
    }
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int i = 0; i < length; ++i) scores[i] *= inv_sum;
}

// Reference FP32 Attention Output:
// A = Softmax(Q K^T / sqrt(D))
// Out = sum_t A_t V_t
inline void attention_fp32_reference(
    const float* query,
    const float* const* keys,
    const float* const* values,
    float* output,
    int tokens,
    int dim
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> scores(tokens);
    for (int t = 0; t < tokens; ++t) {
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += query[d] * keys[t][d];
        }
        scores[t] = dot * scale;
    }
    softmax(scores.data(), tokens);

    std::fill_n(output, dim, 0.0f);
    for (int t = 0; t < tokens; ++t) {
        const float w = scores[t];
        for (int d = 0; d < dim; ++d) {
            output[d] += w * values[t][d];
        }
    }
}

// Unrotated K8V4 Attention Output (Q8 for K, NVFP4 or Turbo4 for V without rotation)
inline void attention_k8v4_unrotated(
    const float* query,
    const std::uint8_t* const* k8_cache,
    const std::uint8_t* const* v4_cache,
    float* output,
    int tokens,
    int dim,
    V4Mode v4_mode
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> scores(tokens);
    std::vector<float> temp_k(dim);
    for (int t = 0; t < tokens; ++t) {
        k8_dequantize_row(k8_cache[t], temp_k.data(), dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += query[d] * temp_k[d];
        }
        scores[t] = dot * scale;
    }
    softmax(scores.data(), tokens);

    std::fill_n(output, dim, 0.0f);
    std::vector<float> temp_v(dim);
    for (int t = 0; t < tokens; ++t) {
        if (v4_mode == V4Mode::NVFP4_G16) {
            v4_nvfp4_dequantize_row(v4_cache[t], temp_v.data(), dim);
        } else {
            v4_turbo4_dequantize_row(v4_cache[t], temp_v.data(), dim);
        }
        const float w = scores[t];
        for (int d = 0; d < dim; ++d) {
            output[d] += w * temp_v[d];
        }
    }
}

// Sylvester-Hadamard Rotated K8V4 Attention Output:
// 1. Q_rot = H Q
// 2. K_cache holds H K quantized to Q8
// 3. V_cache holds H V quantized to V4
// 4. Dot products: <Q_rot, K_rot_t> == <H Q, H K_t> == <Q, K_t>
// 5. Accumulate values directly in rotated basis: O_rot = sum_t A_t V_rot_t
// 6. Final output unrotated ONCE per head: O = H O_rot
inline void attention_k8v4_hadamard(
    const float* query,
    const std::uint8_t* const* k8_rot_cache,
    const std::uint8_t* const* v4_rot_cache,
    float* output,
    int tokens,
    int dim,
    V4Mode v4_mode
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> query_rot(dim);
    hadamard_sylvester_cpu(query, query_rot.data(), dim);

    std::vector<float> scores(tokens);
    std::vector<float> temp_k_rot(dim);
    for (int t = 0; t < tokens; ++t) {
        k8_dequantize_row(k8_rot_cache[t], temp_k_rot.data(), dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += query_rot[d] * temp_k_rot[d];
        }
        scores[t] = dot * scale;
    }
    softmax(scores.data(), tokens);

    std::vector<float> accumulated_rot(dim, 0.0f);
    std::vector<float> temp_v_rot(dim);
    for (int t = 0; t < tokens; ++t) {
        if (v4_mode == V4Mode::NVFP4_G16) {
            v4_nvfp4_dequantize_row(v4_rot_cache[t], temp_v_rot.data(), dim);
        } else {
            v4_turbo4_dequantize_row(v4_rot_cache[t], temp_v_rot.data(), dim);
        }
        const float w = scores[t];
        for (int d = 0; d < dim; ++d) {
            accumulated_rot[d] += w * temp_v_rot[d];
        }
    }

    // Unrotate ONCE at the end
    hadamard_sylvester_cpu(accumulated_rot.data(), output, dim);
}

// Metrics
inline float cosine_similarity(const float* a, const float* b, int dim) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denom > 0.0) ? static_cast<float>(dot / denom) : 0.0f;
}

inline float mean_squared_error(const float* a, const float* b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum / dim);
}

inline float relative_l2_error(const float* approx, const float* ref, int dim) {
    double sum_diff = 0.0, sum_ref = 0.0;
    for (int i = 0; i < dim; ++i) {
        const double diff = static_cast<double>(approx[i]) - static_cast<double>(ref[i]);
        sum_diff += diff * diff;
        sum_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return static_cast<float>(std::sqrt(sum_diff) / std::max(1e-12, std::sqrt(sum_ref)));
}

} // namespace flyweight::k8v4
