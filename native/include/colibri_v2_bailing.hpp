#pragma once

// BailingMoE3 (Ling 3.0) primitives.
//
// Only the pieces that genuinely differ from what the runtime already has. The
// router is here because `noaux_tc` group-limited routing has no equivalent in
// this tree: the DeepSeek-V4 router in colibri_v2_deepseek4.hpp shares the
// bias-steers-selection-but-not-weights structure, but scores with
// sqrt(softplus) and selects over a flat expert list, where this scores with
// sigmoid and selects over expert groups.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Both `include` and `src` are on this target's include path; the quantized
// decoders live beside their encoders in `src`.
#include "qwen_kquant.h"

// The AVX2/AVX-512 quantized row dots the CPU backend already carries. Their
// coverage is wider than what the HF quantizer emits -- K-quants, most IQ
// formats, NVFP4, and the f16/bf16 rows -- and the allowlists beside row_dot
// below record exactly which types each ISA serves, because the entry points
// themselves decode anything unrecognized as Q8_0 rather than rejecting it.
//
// COLIBRI_BAILING_NO_SIMD lets a translation unit that does not link the AVX
// objects (the standalone contract tests) fall back to the scalar path.
#ifndef COLIBRI_BAILING_NO_SIMD
#include "colibri_gpu_driver.h"
#include "qwen_cpu_kernel.h"
#endif

namespace colibri::v2::bailing {

// ---------------------------------------------------------------------------
// profiling
// ---------------------------------------------------------------------------
//
// Opt-in phase timers, enabled with COLIBRI_BAILING_PROFILE=1. Off, this costs
// one predictable branch per phase; the alternative was guessing, and guessing
// already cost a round of kernel work that bought 5%.
//
// Buckets are chosen to separate what batches from what does not, because that
// is the open question: projections and the feed-forward are weight-heavy and
// batched, while the KDA recurrence and MLA attention are sequential in
// position and cannot be.
struct Profile {
    double projections = 0.0;   // batched matmuls feeding attention
    double kda = 0.0;           // convolution + recurrence, sequential
    double mla = 0.0;           // rope + attention, sequential
    double moe = 0.0;           // router + routed experts + shared expert
    double moe_route = 0.0;     // router matmul + selection
    double moe_gather = 0.0;    // gather/scatter around the expert matmuls
    double moe_experts = 0.0;   // the routed expert matmuls themselves
    double moe_shared = 0.0;    // the always-on shared expert
    double dense_ffn = 0.0;
    double norms = 0.0;
    // Outside the layer loop, and the reason this report used to be
    // unfalsifiable: the lm head is one matvec over the whole vocabulary --
    // 0.26 GiB at Q5_K on Ling 3.0 Flash -- and it was in no bucket at all, so
    // the percentages were shares of about half the token's work.
    double head = 0.0;          // final norm + lm head matvec
    double embedding = 0.0;     // the input row gather
    std::uint64_t tokens = 0;
    std::uint64_t positions = 0;  // real tokens, not layer-token pairs

    void report(std::FILE* out) const {
        // The MoE sub-buckets are inside `moe`, so they are excluded here.
        const double total =
            projections + kda + mla + moe + dense_ffn + norms + head + embedding;
        if (total <= 0.0) return;
        std::fprintf(out, "[colibri-v2] bailing profile over %llu positions "
                          "(%llu layer-tokens, %.3f s accounted)\n",
                     static_cast<unsigned long long>(positions),
                     static_cast<unsigned long long>(tokens), total);
        const struct { const char* name; double value; } rows[] = {
            {"projections (batched)", projections},
            {"KDA conv+recurrence (sequential)", kda},
            {"MLA rope+attention (sequential)", mla},
            {"MoE total", moe},
            {"  .. routing", moe_route},
            {"  .. gather/scatter", moe_gather},
            {"  .. expert matmuls", moe_experts},
            {"  .. shared expert", moe_shared},
            {"dense FFN", dense_ffn},
            {"norms", norms},
            {"lm head (+ final norm)", head},
            {"embedding gather", embedding},
        };
        for (const auto& row : rows)
            std::fprintf(out, "    %-34s %7.3f s  %5.1f%%\n",
                         row.name, row.value, 100.0 * row.value / total);
        if (positions)
            std::fprintf(out, "    %-34s %7.3f ms\n",
                         "accounted per position", 1000.0 * total / positions);
    }
};

inline Profile& profile() { static Profile instance; return instance; }

inline bool profiling() {
    static const bool on = [] {
        const char* value = std::getenv("COLIBRI_BAILING_PROFILE");
        return value && value[0] == '1';
    }();
    return on;
}

// Adds its lifetime to `bucket` when profiling is on.
class ProfileScope {
public:
    explicit ProfileScope(double& bucket)
        : bucket_(bucket), start_(profiling() ? now() : 0.0) {}
    ~ProfileScope() { if (profiling()) bucket_ += now() - start_; }

private:
    static double now() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    double& bucket_;
    double start_;
};

// A weight matrix plus how it is stored. Implicitly constructible from a plain
// `const float*` so f32 callers -- the contract tests, and anything that has
// already dequantized -- need no changes.
struct Matrix {
    const void* data = nullptr;
    std::uint32_t type = 0;   // GGML type code; 0 = f32

    Matrix() = default;
    Matrix(const float* values) : data(values), type(0) {}
    Matrix(const void* bytes, std::uint32_t code) : data(bytes), type(code) {}
    explicit operator bool() const { return data != nullptr; }
};

// Element decoders for the storages qwen_kquant.h does not carry. All take a
// ROW base pointer, matching how this header addresses weights. MXFP4 and
// NVFP4 exist in the runtime too, but as .cpp-local functions this header
// cannot reach; the layouts are documented there (v2_runtime.cpp, around the
// kMxfp4/kNvfp4 constants) and verified against real checkpoints -- keep the
// two transcriptions in lockstep.
inline float f16_element(const std::uint8_t* row, std::size_t index) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, row + index * 2, 2);
    return qwen_half_value(bits);
}

// bf16 is the top half of an f32, so widening is a shift.
inline float bf16_element(const std::uint8_t* row, std::size_t index) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, row + index * 2, 2);
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

// The llama.cpp legacy quants: 32-element blocks, nibbles split low-half /
// high-half within the block -- byte j holds element j in its low nibble and
// element j+16 in its high one. Q5's fifth bit for element i is bit i of the
// 32-bit qh word.
inline float q4_0_element(const std::uint8_t* row, std::size_t index) {
    const auto* base = row + index / 32 * 18;
    const int within = static_cast<int>(index & 31);
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const auto byte = base[2 + (within & 15)];
    const int quant = (within < 16 ? (byte & 15) : (byte >> 4)) - 8;
    return qwen_half_value(d_bits) * quant;
}

inline float q4_1_element(const std::uint8_t* row, std::size_t index) {
    const auto* base = row + index / 32 * 20;
    const int within = static_cast<int>(index & 31);
    std::uint16_t d_bits = 0, m_bits = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&m_bits, base + 2, 2);
    const auto byte = base[4 + (within & 15)];
    const int quant = within < 16 ? (byte & 15) : (byte >> 4);
    return qwen_half_value(d_bits) * quant + qwen_half_value(m_bits);
}

inline float q5_0_element(const std::uint8_t* row, std::size_t index) {
    const auto* base = row + index / 32 * 22;
    const int within = static_cast<int>(index & 31);
    std::uint16_t d_bits = 0;
    std::uint32_t high = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&high, base + 2, 4);
    const auto byte = base[6 + (within & 15)];
    const int nibble = within < 16 ? (byte & 15) : (byte >> 4);
    const int quant = static_cast<int>(nibble | (((high >> within) & 1u) << 4)) - 16;
    return qwen_half_value(d_bits) * quant;
}

inline float q5_1_element(const std::uint8_t* row, std::size_t index) {
    const auto* base = row + index / 32 * 24;
    const int within = static_cast<int>(index & 31);
    std::uint16_t d_bits = 0, m_bits = 0;
    std::uint32_t high = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&m_bits, base + 2, 2);
    std::memcpy(&high, base + 4, 4);
    const auto byte = base[8 + (within & 15)];
    const int quant = static_cast<int>(
        (within < 16 ? (byte & 15) : (byte >> 4)) | (((high >> within) & 1u) << 4));
    return qwen_half_value(d_bits) * quant + qwen_half_value(m_bits);
}

inline float mxfp4_element(const std::uint8_t* row, std::size_t index) {
    // The FP4 codebook, doubled -- which is why the scale below is halved.
    static constexpr float lut[16] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
        0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
    const auto* base = row + index / 32 * 17;
    const int within = static_cast<int>(index & 31);
    // E8M0 exponent to float, halved: 2^(x-128). Values below 2 land in the
    // denormal range and are built from bit patterns rather than shifted.
    const std::uint8_t exponent = base[0];
    const std::uint32_t bits = exponent < 2
        ? (0x00200000u << exponent)
        : (static_cast<std::uint32_t>(exponent - 1) << 23);
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    const auto byte = base[1 + (within & 15)];
    return scale * lut[within < 16 ? (byte & 15) : (byte >> 4)];
}

inline float nvfp4_element(const std::uint8_t* row, std::size_t index) {
    // E2M1 LUT: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives).
    static constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    const auto* base = row + index / 64 * 36;
    const int offset = static_cast<int>(index & 63);
    const int sub = offset / 16, within = offset & 15;
    // OCP "FN" E4M3 block scale: e==0xF is FINITE except mantissa 7 (NaN);
    // there are no infinities, and real checkpoints do reach e==0xF.
    const std::uint8_t bits = base[sub];
    const int sign = (bits >> 7) & 1, e = (bits >> 3) & 0xF, m = bits & 7;
    float scale;
    if (e == 0xF && m == 7) scale = std::numeric_limits<float>::quiet_NaN();
    else if (e == 0) scale = (m / 8.0f) * std::exp2(-6.0f);
    else scale = std::exp2(static_cast<float>(e - 7)) * (1.0f + m / 8.0f);
    if (sign) scale = -scale;
    const auto byte = base[4 + sub * 8 + (within & 7)];
    return scale * lut[within < 8 ? (byte & 15) : (byte >> 4)];
}

// Bytes one row of `elements` occupies in each supported storage.
inline std::size_t row_bytes(std::uint32_t type, std::size_t elements) {
    switch (type) {
        case 0:  return elements * sizeof(float);
        case 1:  return elements * 2;                                  // f16
        case 30: return elements * 2;                                  // bf16
        case 2:  return elements / 32 * 18;                            // Q4_0
        case 3:  return elements / 32 * 20;                            // Q4_1
        case 6:  return elements / 32 * 22;                            // Q5_0
        case 7:  return elements / 32 * 24;                            // Q5_1
        case 8:  return elements / 32 * kQ8BlockSize;
        case 10: return elements / kBlockElements * kQ2KBlockBytes;
        case 11: return elements / kBlockElements * kQ3KBlockBytes;
        case 12: return elements / kBlockElements * kQ4KBlockSize;
        case 13: return elements / kBlockElements * kQ5KBlockSize;
        case 14: return elements / kBlockElements * kQ6KBlockSize;
        case 16: return elements / kBlockElements * kIq2xxsBlockBytes;
        case 17: return elements / kBlockElements * kIq2xsBlockBytes;
        case 18: return elements / kBlockElements * kIq3xxsBlockBytes;
        case 19: return elements / kBlockElements * kIq1sBlockBytes;
        // IQ4_NL blocks 32 values, not 256: a quantizer emits it precisely for
        // the rows a superblock format cannot tile.
        case 20: return elements / kIq4nlBlockElements * kIq4nlBlockBytes;
        case 21: return elements / kBlockElements * kIq3sBlockBytes;
        case 22: return elements / kBlockElements * kIq2sBlockBytes;
        case 23: return elements / kBlockElements * kIq4xsBlockBytes;
        case 29: return elements / kBlockElements * kIq1mBlockBytes;
        case 39: return elements / 32 * 17;                            // MXFP4
        case 40: return elements / 64 * 36;                            // NVFP4
        default: throw std::runtime_error(
            "bailing: unsupported weight type " + std::to_string(type));
    }
}

// Decode one stored row into f32.
inline void row_decode(const std::uint8_t* row, std::uint32_t type,
                       std::size_t elements, float* output) {
    switch (type) {
        case 0:  std::memcpy(output, row, elements * sizeof(float)); return;
        case 1:  for (std::size_t i = 0; i < elements; ++i) output[i] = f16_element(row, i); return;
        case 30: for (std::size_t i = 0; i < elements; ++i) output[i] = bf16_element(row, i); return;
        case 2:  for (std::size_t i = 0; i < elements; ++i) output[i] = q4_0_element(row, i); return;
        case 3:  for (std::size_t i = 0; i < elements; ++i) output[i] = q4_1_element(row, i); return;
        case 6:  for (std::size_t i = 0; i < elements; ++i) output[i] = q5_0_element(row, i); return;
        case 7:  for (std::size_t i = 0; i < elements; ++i) output[i] = q5_1_element(row, i); return;
        case 8:  for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_q8_value(row, i); return;
        case 10: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_q2k_value(row, i); return;
        case 11: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_q3k_value(row, i); return;
        case 12: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_q4k_value(row, i); return;
        case 13: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_q5_value(row, i); return;
        case 14: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_q6_value(row, i); return;
        case 16: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq2xxs_value(row, i); return;
        case 17: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq2xs_value(row, i); return;
        case 18: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq3xxs_value(row, i); return;
        case 19: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq1s_value(row, i); return;
        case 20: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq4nl_value(row, i); return;
        case 21: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq3s_value(row, i); return;
        case 22: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq2s_value(row, i); return;
        case 23: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq4xs_value(row, i); return;
        case 29: for (std::size_t i = 0; i < elements; ++i) output[i] = qwen_iq1m_value(row, i); return;
        case 39: for (std::size_t i = 0; i < elements; ++i) output[i] = mxfp4_element(row, i); return;
        case 40: for (std::size_t i = 0; i < elements; ++i) output[i] = nvfp4_element(row, i); return;
        default: throw std::runtime_error(
            "bailing: unsupported weight type " + std::to_string(type));
    }
}

// Decode a whole [outputs][inputs] matrix into f32.
inline void matrix_decode(Matrix weights, std::size_t inputs, std::size_t outputs,
                          float* output) {
    const auto stride = row_bytes(weights.type, inputs);
    const auto* base = reinterpret_cast<const std::uint8_t*>(weights.data);
    for (std::size_t row = 0; row < outputs; ++row)
        row_decode(base + row * stride, weights.type, inputs, output + row * inputs);
}

// Dot one stored row against an f32 vector. The K-quant helpers take the row
// pointer directly; the qwen_kquant.h IQ helpers compute their own row offset,
// so they are handed the row pointer as the matrix and row zero. The remaining
// storages decode element-wise -- the correctness path; the SIMD entry points
// carry the hot versions where they exist.
inline float row_dot(const std::uint8_t* row, std::uint32_t type,
                     const float* input, std::size_t elements) {
    switch (type) {
        case 0: {
            const auto* values = reinterpret_cast<const float*>(row);
            float total = 0.0f;
            for (std::size_t i = 0; i < elements; ++i) total += values[i] * input[i];
            return total;
        }
        case 1: {
            float total = 0.0f;
            for (std::size_t i = 0; i < elements; ++i) total += f16_element(row, i) * input[i];
            return total;
        }
        case 30: {
            float total = 0.0f;
            for (std::size_t i = 0; i < elements; ++i) total += bf16_element(row, i) * input[i];
            return total;
        }
        case 8:  return qwen_q8_dot_row(row, input, elements);
        case 12: return qwen_q4k_dot_row(row, input, elements);
        case 13: return qwen_q5k_dot_row(row, input, elements);
        case 14: return qwen_q6k_dot_row(row, input, elements);
        case 10: return qwen_q2k_dot_row(row, input, static_cast<int>(elements), 0);
        case 11: return qwen_q3k_dot_row(row, input, static_cast<int>(elements), 0);
        case 16: return qwen_iq2xxs_dot_row(row, input, static_cast<int>(elements), 0);
        case 17: return qwen_iq2xs_dot_row(row, input, static_cast<int>(elements), 0);
        case 18: return qwen_iq3xxs_dot_row(row, input, static_cast<int>(elements), 0);
        case 19: return qwen_iq1s_dot_row(row, input, static_cast<int>(elements), 0);
        case 20: return qwen_iq4nl_dot_row(row, input, static_cast<int>(elements), 0);
        case 21: return qwen_iq3s_dot_row(row, input, static_cast<int>(elements), 0);
        case 22: return qwen_iq2s_dot_row(row, input, static_cast<int>(elements), 0);
        case 23: return qwen_iq4xs_dot_row(row, input, static_cast<int>(elements), 0);
        case 29: return qwen_iq1m_dot_row(row, input, static_cast<int>(elements), 0);
        case 2: case 3: case 6: case 7: case 39: case 40: {
            float total = 0.0f;
            for (std::size_t i = 0; i < elements; ++i) {
                float value;
                switch (type) {
                    case 2:  value = q4_0_element(row, i); break;
                    case 3:  value = q4_1_element(row, i); break;
                    case 6:  value = q5_0_element(row, i); break;
                    case 7:  value = q5_1_element(row, i); break;
                    case 39: value = mxfp4_element(row, i); break;
                    default: value = nvfp4_element(row, i); break;
                }
                total += value * input[i];
            }
            return total;
        }
        default: throw std::runtime_error(
            "bailing: unsupported weight type " + std::to_string(type));
    }
}

// Which storages the AVX entry points decode with a dedicated kernel. They
// fall through to their Q8_0 path for anything they do not recognize, so
// admission MUST be an allowlist: a new type reaching them by default is
// silently mis-decoded, not rejected. The f16/bf16 rows (type 1/30) have
// explicit branches added alongside this change; the rest mirror the dispatch
// the entry points actually implement.
inline bool simd_dot_avx512_type(std::uint32_t type) {
    return type == 1 || type == 8 || type == 10 || type == 11 || type == 12 ||
           type == 13 || type == 14 || type == 17 || type == 30 || type == 40;
}
inline bool simd_dot_avx2_type(std::uint32_t type) {
    return type == 1 || type == 8 || type == 10 || type == 11 || type == 12 ||
           type == 13 || type == 14 || type == 16 || type == 17 || type == 18 ||
           type == 22 || type == 23 || type == 30 || type == 40;
}
// The row-length granule each SIMD kernel assumes. f16/bf16 kernels carry
// their own scalar tails, so any length is admissible.
inline std::size_t simd_dot_granule(std::uint32_t type) {
    if (type == 1 || type == 30) return 1;
    if (type == 40) return 64;
    return kBlockElements;
}

// Expert routing for one token: `noaux_tc`, transcribed from
// BailingMoeV3Gate.forward / group_limited_topk in the checkpoint's own
// modeling_bailing_moe_v3.py (lines 368-406).
//
// The shape of it:
//
//   1. score every expert with a sigmoid over the fp32 logits;
//   2. add the trained per-expert bias to get a *selection* score;
//   3. rank expert GROUPS by the sum of their best two selection scores, keep
//      the best `groups_used` groups, and mask the rest away;
//   4. take the top `used` experts among what survives;
//   5. weight them by their UNBIASED sigmoid scores, normalized, then scaled.
//
// Step 5 is the one that is quiet when wrong. The bias exists to steer load
// across experts during training; letting it reach the weights would distort
// every expert's contribution by its own load-balancing term. Selection uses
// the biased score, weighting uses the raw one.
//
// Step 3 is the "group-limited" part: with 128 experts in 8 groups of 16,
// keeping 4 groups means the top-8 selection only ever sees 64 candidates.
// Ranking groups by their top *two* members (not their best, not their sum) is
// what the reference does.
//
// Ties are broken by ascending index throughout, matching the DeepSeek-V4
// router here, so routing is deterministic run to run.
inline void moe_router(
    const float* logits,
    const float* bias,
    std::size_t experts,
    std::size_t used,
    std::size_t groups,
    std::size_t groups_used,
    float weight_scale,
    bool normalize,
    std::int32_t* chosen,
    float* weights
) {
    std::vector<float> probabilities(experts);
    for (std::size_t expert = 0; expert < experts; ++expert) {
        const float logit = logits[expert];
        probabilities[expert] = 1.0f / (1.0f + std::exp(-logit));
    }

    std::vector<float> selection(experts);
    for (std::size_t expert = 0; expert < experts; ++expert)
        selection[expert] = probabilities[expert] + (bias ? bias[expert] : 0.0f);

    // A zero or degenerate group count means no group limiting: every expert
    // stays a candidate. Keeps this usable for plain sigmoid top-k routing.
    const bool limited = groups > 1 && groups_used > 0 && groups_used < groups &&
                         experts % groups == 0;
    std::vector<std::uint8_t> allowed(experts, 1u);
    if (limited) {
        const std::size_t span = experts / groups;
        std::vector<float> group_score(groups, 0.0f);
        for (std::size_t group = 0; group < groups; ++group) {
            // Sum of the two largest selection scores in this group.
            float best = -std::numeric_limits<float>::infinity();
            float second = -std::numeric_limits<float>::infinity();
            for (std::size_t offset = 0; offset < span; ++offset) {
                const float value = selection[group * span + offset];
                if (value > best) { second = best; best = value; }
                else if (value > second) { second = value; }
            }
            group_score[group] = best + second;
        }
        std::vector<std::int32_t> order(groups);
        for (std::size_t group = 0; group < groups; ++group)
            order[group] = static_cast<std::int32_t>(group);
        std::partial_sort(
            order.begin(), order.begin() + static_cast<std::ptrdiff_t>(groups_used),
            order.end(),
            [&](std::int32_t left, std::int32_t right) {
                const float a = group_score[left], b = group_score[right];
                return a != b ? a > b : left < right;
            });
        std::fill(allowed.begin(), allowed.end(), 0u);
        for (std::size_t slot = 0; slot < groups_used; ++slot)
            std::fill_n(allowed.begin() + order[slot] * static_cast<std::int32_t>(span),
                        span, 1u);
    }

    std::vector<std::int32_t> order(experts);
    for (std::size_t expert = 0; expert < experts; ++expert)
        order[expert] = static_cast<std::int32_t>(expert);
    // Masked-out experts sort last rather than being removed, so the selection
    // still fills `used` slots even in the degenerate case where fewer
    // candidates survive than are asked for.
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(used), order.end(),
        [&](std::int32_t left, std::int32_t right) {
            if (allowed[left] != allowed[right]) return allowed[left] > allowed[right];
            const float a = selection[left], b = selection[right];
            return a != b ? a > b : left < right;
        });
    std::copy_n(order.begin(), used, chosen);

    float total = 0.0f;
    for (std::size_t slot = 0; slot < used; ++slot) {
        weights[slot] = probabilities[chosen[slot]];
        total += weights[slot];
    }
    // The reference divides by `sum + 1e-20` and only when more than one expert
    // is used; the epsilon is what keeps an all-zero row finite.
    if (normalize && used > 1)
        for (std::size_t slot = 0; slot < used; ++slot)
            weights[slot] /= (total + 1e-20f);
    for (std::size_t slot = 0; slot < used; ++slot) weights[slot] *= weight_scale;
}

// Partial rotary embedding for the MLA layers, over the trailing `rope_dim`
// channels of a row, pairing ADJACENT channels (ggml's NORM layout).
//
// Why adjacent pairs, when the reference appears to do something else.
// `apply_rotary_pos_emb_interleave` (modeling_bailing_moe_v3.py:541-577) first
// does `view(..., d/2, 2).transpose(4,3).reshape(..., d)` -- a de-interleave
// that moves even channels to the front half and odd channels to the back --
// and then applies the ordinary half-split (NEOX) rotation to THAT.
//
// Composing the two gives back exactly the adjacent-pair rotation, with the
// output coordinates left in the permuted order. And that leftover permutation
// is unobservable: it is applied identically to q and to k, it is orthogonal,
// and the only thing either feeds is their dot product. Verified numerically --
// the reference and this function disagree element for element and agree on
// every attention score.
//
// So implementing it "faithfully" by reproducing the reshape would be copying a
// no-op, and reading the half-split rotation out of the reference's second half
// -- the obvious misreading -- gives fluent, wrong text. Neither trap is
// visible without doing the composition.
inline void partial_rope_norm(
    float* row,
    std::size_t head_dim,
    std::size_t rope_dim,
    std::int32_t position,
    float theta
) {
    if (rope_dim == 0 || rope_dim > head_dim) return;
    float* span = row + (head_dim - rope_dim);
    for (std::size_t pair = 0; pair * 2 < rope_dim; ++pair) {
        const float exponent =
            static_cast<float>(2 * pair) / static_cast<float>(rope_dim);
        const float frequency = 1.0f / std::pow(theta, exponent);
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle), sine = std::sin(angle);
        const float even = span[pair * 2], odd = span[pair * 2 + 1];
        span[pair * 2] = even * cosine - odd * sine;
        span[pair * 2 + 1] = odd * cosine + even * sine;
    }
}

// The head-wise output gate the MLA layers carry, which DeepSeek's MLA does
// not: `g_proj` is hidden -> heads, and each head's attention output is scaled
// by the sigmoid of its own logit before the output projection
// (modeling:709-718). `element_wise` granularity would give one logit per
// output channel instead; this checkpoint is `head_wise`.
inline void apply_head_gate(
    const float* gate_logits,
    std::size_t heads,
    std::size_t value_dim,
    float* attention_output
) {
    for (std::size_t head = 0; head < heads; ++head) {
        const float gate = 1.0f / (1.0f + std::exp(-gate_logits[head]));
        float* row = attention_output + head * value_dim;
        for (std::size_t i = 0; i < value_dim; ++i) row[i] *= gate;
    }
}

// ---------------------------------------------------------------------------
// feed-forward
// ---------------------------------------------------------------------------

// SwiGLU: silu(gate) * up. `limit` bounds both halves before combining when the
// checkpoint carries a per-layer clamp; a non-positive limit disables it.
//
// Ling-3.0-tiny has `expert_swiglu_limit_list: null`, but the flash checkpoints
// do carry one -- llama.cpp shipped a metadata repair script for GGUFs that
// omitted it -- so the clamp is plumbed rather than assumed away.
inline void swiglu(
    const float* gate, const float* up, std::size_t size, float limit, float* output
) {
    const bool clamped = limit > 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        float g = gate[i], u = up[i];
        if (clamped) {
            g = std::min(std::max(g, -limit), limit);
            u = std::min(std::max(u, -limit), limit);
        }
        output[i] = (g / (1.0f + std::exp(-g))) * u;
    }
}

// Row-major [outputs][inputs] matrix times a vector, in whatever storage the
// matrix declares. Quantized rows are decoded a block at a time by row_dot
// rather than element by element, which is what keeps this competitive with the
// f32 path instead of an order of magnitude behind it.
// How many threads to spread a matvec over.
//
// OpenMP defaults to the LOGICAL cpu count, which is the wrong answer for this
// work. Measured on a Ryzen 9 9955HX (16 cores / 32 threads), Q4_K decode:
//
//     16 threads   0.095-0.101 s/token
//     32 threads   0.180 s/token
//
// -- nearly 2x slower at the default. Weight-streaming matvecs are limited by
// memory bandwidth and cache, not by issue width, so a second thread on the
// same physical core buys nothing and costs contention for its L1/L2.
//
// So the default is one thread per physical core, read from Linux's topology
// rather than assumed. An explicit OMP_NUM_THREADS always wins: if the caller
// said what they wanted, they meant it.
inline int matvec_threads() {
#ifdef _OPENMP
    static const int threads = [] {
        if (std::getenv("OMP_NUM_THREADS")) return omp_get_max_threads();
        int siblings = 1;
#if defined(__linux__)
        if (std::FILE* file = std::fopen(
                "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r")) {
            char line[256] = {};
            if (std::fgets(line, sizeof(line), file)) {
                // "0,16" or "0-1": one entry per hardware thread on this core.
                siblings = 1;
                for (const char* c = line; *c; ++c)
                    if (*c == ',') ++siblings;
                    else if (*c == '-') siblings = 2;
            }
            std::fclose(file);
        }
#endif
        const int procs = omp_get_num_procs();
        const int physical = siblings > 1 ? procs / siblings : procs;
        return physical > 0 ? physical : 1;
    }();
    return threads;
#else
    return 1;
#endif
}

// Rows are independent, so this parallelizes trivially. Worth doing: the
// scalar single-threaded version ran at ~4 GFLOP/s on a 32-core machine, which
// is one or two orders of magnitude off what the hardware can do.
//
// The kernel choice for one storage, resolved once and then applied row by
// row. Split out of `matvec` so a CALLER can own the parallel region: the
// routed experts want all of a layer's rows under one `omp parallel`, and
// calling matvec per expert per projection fires 24 regions where 2 will do.
//
// Admission is per ISA and per type, and MUST stay an allowlist: the SIMD
// entry points decode a type they do not recognize as Q8_0 rather than
// rejecting it, so a type that falls through by default is silently
// mis-decoded, not caught.
struct RowKernel {
    enum class Path { Scalar, Avx2, Avx512 };
    Path path = Path::Scalar;
    std::uint32_t type = 0;
    std::size_t stride = 0;

    float operator()(const std::uint8_t* base, const float* input,
                     std::size_t inputs, std::size_t row) const {
#ifndef COLIBRI_BAILING_NO_SIMD
        if (path == Path::Avx512)
            return qwen_quant_dot_avx512(base, type, input,
                                         static_cast<int>(inputs), row);
        if (path == Path::Avx2)
            return qwen_quant_dot_avx2(base, type, input,
                                       static_cast<int>(inputs), row);
#endif
        return row_dot(base + row * stride, type, input, inputs);
    }
};

inline RowKernel row_kernel(std::uint32_t type, std::size_t inputs) {
    RowKernel kernel;
    kernel.type = type;
    kernel.stride = row_bytes(type, inputs);
#ifndef COLIBRI_BAILING_NO_SIMD
    // The SIMD kernels index rows themselves and want whole granules.
    if (type != 0 && inputs % simd_dot_granule(type) == 0) {
        const auto features = colibri_cpu_features();
        if ((features & 2u) && simd_dot_avx512_type(type))
            kernel.path = RowKernel::Path::Avx512;
        else if ((features & 1u) && simd_dot_avx2_type(type))
            kernel.path = RowKernel::Path::Avx2;
    }
#endif
    return kernel;
}

// The threshold keeps the small projections -- the 16-wide head gate, the
// router -- out of the thread pool, where the fork cost dominates the work.
inline void matvec(
    Matrix weights, const float* input,
    std::size_t inputs, std::size_t outputs, float* output
) {
    const auto* base = reinterpret_cast<const std::uint8_t*>(weights.data);
    const auto kernel = row_kernel(weights.type, inputs);

#ifdef _OPENMP
    if (outputs * inputs >= 65536) {
#pragma omp parallel for schedule(static) num_threads(matvec_threads())
        for (std::int64_t row = 0; row < static_cast<std::int64_t>(outputs); ++row)
            output[row] = kernel(base, input, inputs,
                                 static_cast<std::size_t>(row));
        return;
    }
#endif
    for (std::size_t row = 0; row < outputs; ++row)
        output[row] = kernel(base, input, inputs, row);
}

// One token through a sparse MoE block: route, run the chosen experts, add the
// shared expert.
//
// Expert weights are the stacked blocks the loader builds, laid out
// [expert][outputs][inputs] -- expert-major, which is what concatenating
// per-expert row-major matrices already gives.
//
// The shared expert is added to the routed sum and is NOT itself scaled by
// `weight_scale`: the scaling applies to the router's weights, which the shared
// expert does not have (modeling:448-449).
inline void moe_block(
    const float* hidden,
    Matrix router_weights,
    const float* router_bias,
    Matrix gate_experts,
    Matrix up_experts,
    Matrix down_experts,
    Matrix shared_gate,
    Matrix shared_up,
    Matrix shared_down,
    std::size_t hidden_size,
    std::size_t expert_size,
    std::size_t shared_size,
    std::size_t experts,
    std::size_t used,
    std::size_t groups,
    std::size_t groups_used,
    float weight_scale,
    bool normalize,
    float swiglu_limit,
    float shared_swiglu_limit,
    float* output
) {
    std::vector<float> logits(experts);
    ProfileScope* route = new ProfileScope(profile().moe_route);
    matvec(router_weights, hidden, hidden_size, experts, logits.data());

    std::vector<std::int32_t> chosen(used);
    std::vector<float> weights(used);
    moe_router(logits.data(), router_bias, experts, used, groups, groups_used,
               weight_scale, normalize, chosen.data(), weights.data());
    delete route;

    // One expert's slice, in bytes: the stacked block is expert-major, so the
    // stride is however much one expert's matrix occupies in its own storage.
    const auto narrow_stride = expert_size * row_bytes(gate_experts.type, hidden_size);
    const auto wide_stride = hidden_size * row_bytes(down_experts.type, expert_size);
    const auto* gate_base = reinterpret_cast<const std::uint8_t*>(gate_experts.data);
    const auto* up_base = reinterpret_cast<const std::uint8_t*>(up_experts.data);
    const auto* down_base = reinterpret_cast<const std::uint8_t*>(down_experts.data);

    // All the chosen experts at once, in TWO parallel regions rather than the
    // three-per-expert that calling matvec here used to make.
    //
    // This matters because of what the expert weights are. At IQ2_XXS a single
    // core sustains ~3 GB/s -- the grid lookup is the cost, not the load -- so
    // unlike every other phase in this layer, the experts genuinely need all
    // the cores. But each individual projection is only ~0.5 MB, and 24 forks
    // a layer (960 a token, at 40 MoE layers) spend more on entering and
    // leaving the thread pool than they recover. Measured on Ling 3.0 Flash's
    // geometry: 29.8 GB/s per-expert against 40.5 GB/s fused, same kernel and
    // same bytes.
    ProfileScope* work = new ProfileScope(profile().moe_experts);
    std::vector<float> activated(used * expert_size);
    {
        // gate and up over every chosen expert. Row r decodes to (half, slot,
        // row): the two halves are laid end to end so one loop covers both.
        std::vector<float> gate_up(2 * used * expert_size);
        const auto narrow_kernel = row_kernel(gate_experts.type, hidden_size);
        const auto rows = static_cast<std::int64_t>(2 * used * expert_size);
        const auto span = static_cast<std::int64_t>(used * expert_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads())
#endif
        for (std::int64_t r = 0; r < rows; ++r) {
            const bool is_up = r >= span;
            const auto within = static_cast<std::size_t>(is_up ? r - span : r);
            const auto slot = within / expert_size, row = within % expert_size;
            const auto expert = static_cast<std::size_t>(chosen[slot]);
            const auto* base = (is_up ? up_base : gate_base) + expert * narrow_stride;
            gate_up[static_cast<std::size_t>(r)] =
                narrow_kernel(base, hidden, hidden_size, row);
        }
        swiglu(gate_up.data(), gate_up.data() + used * expert_size,
               used * expert_size, swiglu_limit, activated.data());
    }
    {
        // The down projection accumulates into `output`, so parallelize over
        // OUTPUT rows and keep the expert loop inside: every thread then owns
        // its rows outright and no reduction is needed. Same shape as the
        // device's bailing_q6_expert_accumulate_rows.
        const auto down_kernel = row_kernel(down_experts.type, expert_size);
        const auto rows = static_cast<std::int64_t>(hidden_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads())
#endif
        for (std::int64_t r = 0; r < rows; ++r) {
            const auto row = static_cast<std::size_t>(r);
            float total = 0.0f;
            for (std::size_t slot = 0; slot < used; ++slot) {
                const auto expert = static_cast<std::size_t>(chosen[slot]);
                total += weights[slot] * down_kernel(
                    down_base + expert * wide_stride,
                    activated.data() + slot * expert_size, expert_size, row);
            }
            output[row] = total;
        }
    }
    delete work;

    if (shared_gate && shared_up && shared_down && shared_size) {
        ProfileScope shared(profile().moe_shared);
        std::vector<float> sg(shared_size), su(shared_size), sa(shared_size),
            projected(hidden_size);
        matvec(shared_gate, hidden, hidden_size, shared_size, sg.data());
        matvec(shared_up, hidden, hidden_size, shared_size, su.data());
        swiglu(sg.data(), su.data(), shared_size, shared_swiglu_limit, sa.data());
        matvec(shared_down, sa.data(), shared_size, hidden_size, projected.data());
        for (std::size_t i = 0; i < hidden_size; ++i) output[i] += projected[i];
    }
}

// ---------------------------------------------------------------------------
// layer plan
// ---------------------------------------------------------------------------

// What each layer of a BailingMoE3 checkpoint must carry, by kind.
//
// This exists to be checked at load, not merely to document. The stage-1 loader
// gate proved the descriptors point at the right BYTES; it could not prove they
// carry the right MEANING, because nothing consumed them. A tensor mapped to
// the wrong name survives that gate and then shows up as a wrong kernel result
// much later, indistinguishable from a numerics bug. Resolving the full plan at
// open time is what closes that gap: a missing or misnamed tensor fails
// immediately, with its own name in the message.
//
// Returned as name suffixes rather than indices so the caller owns lookup.
struct LayerRequirements {
    std::vector<const char*> names;
    bool full_attention = false;
};

inline LayerRequirements layer_requirements(
    bool full_attention, bool routed_experts, bool shared_expert,
    bool query_lora
) {
    LayerRequirements out;
    out.full_attention = full_attention;
    out.names = {"attn_norm.weight", "ffn_norm.weight"};
    if (full_attention) {
        // MLA: compressed KV with a shared rope tail, and the head-wise output
        // gate that DeepSeek's MLA does not have. The query is low-rank only
        // when the checkpoint declares a rank -- Ling 3.0 Flash does not, and
        // carries an un-factored `attn_q` instead of the q_a/q_b pair.
        if (query_lora)
            for (const char* name : {
                     "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight"})
                out.names.push_back(name);
        else
            out.names.push_back("attn_q.weight");
        for (const char* name : {
                 "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                 "attn_kv_b.weight", "attn_output.weight", "attn_gate.weight"})
            out.names.push_back(name);
    } else {
        // KDA: three separately-convolved projections, the per-channel decay
        // pair (ssm_f / ssm_a / ssm_dt.bias), the per-head beta, and the gated
        // output norm.
        for (const char* name : {
                 "ssm_q.weight", "ssm_k.weight", "ssm_v.weight",
                 "ssm_q_conv1d.weight", "ssm_k_conv1d.weight",
                 "ssm_v_conv1d.weight", "ssm_f.weight", "ssm_b.weight",
                 "ssm_g.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight",
                 "ssm_out.weight"})
            out.names.push_back(name);
    }
    if (routed_experts) {
        for (const char* name : {"ffn_gate_inp.weight", "exp_probs_b.bias",
                                 "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                 "ffn_down_exps.weight"})
            out.names.push_back(name);
        if (shared_expert)
            for (const char* name : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                     "ffn_down_shexp.weight"})
                out.names.push_back(name);
    } else {
        for (const char* name : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
            out.names.push_back(name);
    }
    return out;
}

// A layer closes its group, or sits past the last whole group. Transcribed from
// modeling_bailing_moe_v3.py:1004-1009; the second clause is dead at 24 layers
// with group 4 but fires on counts that are not a multiple, so it is carried
// rather than assumed away.
inline bool layer_is_full_attention(
    std::uint32_t layer, std::uint32_t layers, std::uint32_t group
) {
    if (group == 0) return true;
    const std::uint32_t whole = layers / group * group;
    return ((layer + 1) % group == 0) || layer >= whole;
}

// ---------------------------------------------------------------------------
// MLA attention
// ---------------------------------------------------------------------------
//
// Two implementations of the same operator, kept side by side on purpose.
//
// `mla_attention_decompressed` caches per-head keys and values, the way every
// other attention layer in this tree does. It is the reference: simple, reuses
// the existing cache machinery, and obviously correct.
//
// `mla_attention_absorbed` caches the 512-wide latent instead and folds
// `kv_b_proj` into the query and output projections, so attention never
// decompresses. That is 8.9x less cache traffic on this checkpoint -- 0.91 GB
// against 8.05 GB at 128k context -- for 3.4x more arithmetic per cached
// position. On a bandwidth-bound decode that is the right trade, and it is what
// MLA exists to enable.
//
// The absorbed form rests on an algebraic identity:
//
//     q_nope . (W_k @ latent)  ==  (W_k^T @ q_nope) . latent
//     sum_p a_p (W_v @ latent_p)  ==  W_v @ (sum_p a_p latent_p)
//
// Identities that plainly hold are exactly where this codebase has hidden its
// silent bugs, so the decompressed path stays as the thing to diff against
// rather than being deleted once the absorbed one works.

// Softmax over `positions` scores, in place, max-shifted for stability.
inline void softmax_in_place(float* scores, std::size_t positions) {
    if (positions == 0) return;
    float peak = scores[0];
    for (std::size_t i = 1; i < positions; ++i) peak = std::max(peak, scores[i]);
    float total = 0.0f;
    for (std::size_t i = 0; i < positions; ++i) {
        scores[i] = std::exp(scores[i] - peak);
        total += scores[i];
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    for (std::size_t i = 0; i < positions; ++i) scores[i] *= inverse;
}

// Reference form. Keys are [positions][heads][qk_head_dim] with the rope span
// already written into their tail, values are [positions][heads][v_head_dim].
//
// The score scale is 1/sqrt(qk_head_dim) -- over the FULL 192, nope plus rope,
// not over head_dim or over the nope part alone (modeling:636).
inline void mla_attention_decompressed(
    const float* queries,
    const float* cache_keys,
    const float* cache_values,
    std::size_t positions,
    std::size_t heads,
    std::size_t qk_head_dim,
    std::size_t v_head_dim,
    float* output
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk_head_dim));
    std::vector<float> scores(positions);
    for (std::size_t head = 0; head < heads; ++head) {
        const float* query = queries + head * qk_head_dim;
        for (std::size_t position = 0; position < positions; ++position) {
            const float* key =
                cache_keys + (position * heads + head) * qk_head_dim;
            float total = 0.0f;
            for (std::size_t i = 0; i < qk_head_dim; ++i) total += query[i] * key[i];
            scores[position] = total * scale;
        }
        softmax_in_place(scores.data(), positions);
        float* target = output + head * v_head_dim;
        for (std::size_t i = 0; i < v_head_dim; ++i) target[i] = 0.0f;
        for (std::size_t position = 0; position < positions; ++position) {
            const float weight = scores[position];
            const float* value =
                cache_values + (position * heads + head) * v_head_dim;
            for (std::size_t i = 0; i < v_head_dim; ++i) target[i] += value[i] * weight;
        }
    }
}

// Absorbed form. The cache holds the normalized latent and the shared rope key.
//
// `kv_b` is the kv_b_proj weight as the checkpoint stores it, indexed
// [head][qk_nope + v_head_dim][kv_lora] -- output-major, which is what a
// row-major [heads*(qk_nope+v_head), kv_lora] matrix already is.
//
// `cache_rope` is one row per position, NOT per head: the rope half of the key
// is MQA, shared by every head (modeling:672-680).
inline void mla_attention_absorbed(
    const float* queries_nope,
    const float* queries_rope,
    const float* kv_b,
    const float* cache_latents,
    const float* cache_rope,
    std::size_t positions,
    std::size_t heads,
    std::size_t qk_nope,
    std::size_t qk_rope,
    std::size_t v_head_dim,
    std::size_t kv_lora,
    float* output
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk_nope + qk_rope));
    const std::size_t stride = qk_nope + v_head_dim;
    // Independent per head, and 34% of prefill sat here.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads()) \
        if (positions * heads >= 64)
#endif
    for (std::int64_t head_index = 0; head_index < static_cast<std::int64_t>(heads);
         ++head_index) {
        const auto head = static_cast<std::size_t>(head_index);
        std::vector<float> scores(positions);
        std::vector<float> projected(kv_lora), accumulated(kv_lora);
        const float* head_weights = kv_b + head * stride * kv_lora;
        const float* query_nope = queries_nope + head * qk_nope;
        const float* query_rope = queries_rope + head * qk_rope;

        // Pull the query through kv_b's key half once, instead of pushing every
        // cached latent through it. This is the whole saving: O(1) per token
        // rather than O(context).
        for (std::size_t i = 0; i < kv_lora; ++i) projected[i] = 0.0f;
        for (std::size_t row = 0; row < qk_nope; ++row) {
            const float weight = query_nope[row];
            if (weight == 0.0f) continue;
            const float* source = head_weights + row * kv_lora;
            for (std::size_t i = 0; i < kv_lora; ++i) projected[i] += weight * source[i];
        }

        for (std::size_t position = 0; position < positions; ++position) {
            const float* latent = cache_latents + position * kv_lora;
            float total = 0.0f;
            for (std::size_t i = 0; i < kv_lora; ++i) total += projected[i] * latent[i];
            const float* rope = cache_rope + position * qk_rope;
            for (std::size_t i = 0; i < qk_rope; ++i) total += query_rope[i] * rope[i];
            scores[position] = total * scale;
        }
        softmax_in_place(scores.data(), positions);

        // Mix in latent space, then decompress once.
        for (std::size_t i = 0; i < kv_lora; ++i) accumulated[i] = 0.0f;
        for (std::size_t position = 0; position < positions; ++position) {
            const float weight = scores[position];
            const float* latent = cache_latents + position * kv_lora;
            for (std::size_t i = 0; i < kv_lora; ++i) accumulated[i] += latent[i] * weight;
        }
        float* target = output + head * v_head_dim;
        for (std::size_t row = 0; row < v_head_dim; ++row) {
            const float* source = head_weights + (qk_nope + row) * kv_lora;
            float total = 0.0f;
            for (std::size_t i = 0; i < kv_lora; ++i) total += source[i] * accumulated[i];
            target[row] = total;
        }
    }
}

// Decompress one cached latent into the per-head keys and values the reference
// form expects. Used to build the decompressed cache, and to check the two
// forms against each other.
inline void mla_decompress(
    const float* latent,
    const float* rope_key,
    const float* kv_b,
    std::size_t heads,
    std::size_t qk_nope,
    std::size_t qk_rope,
    std::size_t v_head_dim,
    std::size_t kv_lora,
    float* keys,
    float* values
) {
    const std::size_t stride = qk_nope + v_head_dim;
    for (std::size_t head = 0; head < heads; ++head) {
        const float* head_weights = kv_b + head * stride * kv_lora;
        float* key = keys + head * (qk_nope + qk_rope);
        for (std::size_t row = 0; row < qk_nope; ++row) {
            const float* source = head_weights + row * kv_lora;
            float total = 0.0f;
            for (std::size_t i = 0; i < kv_lora; ++i) total += source[i] * latent[i];
            key[row] = total;
        }
        // The rope half is shared: every head gets the same tail.
        for (std::size_t i = 0; i < qk_rope; ++i) key[qk_nope + i] = rope_key[i];
        float* value = values + head * v_head_dim;
        for (std::size_t row = 0; row < v_head_dim; ++row) {
            const float* source = head_weights + (qk_nope + row) * kv_lora;
            float total = 0.0f;
            for (std::size_t i = 0; i < kv_lora; ++i) total += source[i] * latent[i];
            value[row] = total;
        }
    }
}

// Kimi Delta Attention recurrence over `rows` tokens, host side.
//
// Oracle: native/tools/kda_reference.py, itself pinned to
// flash-linear-attention's own `naive_recurrent_kda` / `naive_chunk_kda`.
//
// The relationship to the DeltaNet kernel already in this tree
// (qwen_delta_recurrent_chunk) is worth stating exactly, because it is almost
// the same computation:
//
//   DeltaNet   state[key] *= decay_scale          -- one scalar per head
//   KDA        state[key] *= decay[key]           -- one per key channel
//
// and the decay formula itself is *identical* -- `exp(coefficient * softplus(x
// + bias))` with a negative coefficient -- except that DeltaNet's softplus
// argument and bias are per head while KDA's are per channel. Everything after
// the decay (L2-normalized q/k, the beta-gated delta rule, the query readout,
// the 1/sqrt(head_dim) query scale) is the same.
//
// Layouts, all float32, matching kda_reference.py:
//
//   queries/keys/values  [rows][heads * head_dim]  post-convolution, post-SiLU
//   gate_raw             [rows][heads * head_dim]  f_proj output, pre-gate
//   beta_logits          [rows][heads]             b_proj output, pre-sigmoid
//   a_log                [heads]
//   dt_bias              [heads * head_dim]
//   state                [heads][head_dim][head_dim]   keys x values, updated
//   output               [rows][heads * head_dim]
//
// `state` is read and written in place, so a decode step is this with rows=1.
inline void kda_recurrence(
    const float* queries,
    const float* keys,
    const float* values,
    const float* gate_raw,
    const float* beta_logits,
    const float* a_log,
    const float* dt_bias,
    std::size_t rows,
    std::size_t heads,
    std::size_t head_dim,
    float epsilon,
    float* state,
    float* output
) {
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Heads are independent: head h's state evolves over rows without ever
    // touching another head's. So the nest is head-outer / row-inner, which
    // keeps the recurrence sequential where it must be (in position) while
    // parallelizing where it may be -- and forks once per call rather than once
    // per row. Profiling put 43% of prefill in this loop; it was scalar and
    // single-threaded.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads()) \
        if (rows * heads >= 16)
#endif
    for (std::int64_t head_index = 0; head_index < static_cast<std::int64_t>(heads);
         ++head_index) {
        const auto head = static_cast<std::size_t>(head_index);
        std::vector<float> decay(head_dim), key_row(head_dim), query_row(head_dim);
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t base = row * heads * head_dim + head * head_dim;
            float* matrix = state + head * head_dim * head_dim;

            // L2 normalization of q and k, per head, as
            // `use_qk_l2norm_in_kernel=True` asks for. The query also carries
            // the 1/sqrt(head_dim) attention scale.
            float query_square = 0.0f, key_square = 0.0f;
            for (std::size_t i = 0; i < head_dim; ++i) {
                query_square += queries[base + i] * queries[base + i];
                key_square += keys[base + i] * keys[base + i];
            }
            const float query_inverse = query_scale / std::sqrt(query_square + epsilon);
            const float key_inverse = 1.0f / std::sqrt(key_square + epsilon);
            for (std::size_t i = 0; i < head_dim; ++i) {
                query_row[i] = queries[base + i] * query_inverse;
                key_row[i] = keys[base + i] * key_inverse;
            }

            // Per-channel decay: g = -exp(A_log) * softplus(f + dt_bias), and
            // the state is multiplied by exp(g). Folded into one exp: the
            // coefficient is negative, so this is always a contraction.
            const float coefficient = -std::exp(a_log[head]);
            for (std::size_t i = 0; i < head_dim; ++i) {
                const float x = gate_raw[base + i] + dt_bias[head * head_dim + i];
                // Guarded softplus: above 20 the identity is exact in float32
                // and expf would overflow.
                const float softplus = x > 20.0f ? x : std::log1p(std::exp(x));
                decay[i] = std::exp(coefficient * softplus);
            }

            const float beta =
                1.0f / (1.0f + std::exp(-beta_logits[row * heads + head]));

            // Decay, then read what this key currently predicts.
            for (std::size_t key = 0; key < head_dim; ++key) {
                const float scale = decay[key];
                float* target = matrix + key * head_dim;
                for (std::size_t value = 0; value < head_dim; ++value)
                    target[value] *= scale;
            }
            for (std::size_t value = 0; value < head_dim; ++value) {
                float predicted = 0.0f;
                for (std::size_t key = 0; key < head_dim; ++key)
                    predicted += matrix[key * head_dim + value] * key_row[key];
                // Delta rule: move the stored value toward the observed one by
                // beta of the gap, spread across the key's own direction.
                const float correction = (values[base + value] - predicted) * beta;
                for (std::size_t key = 0; key < head_dim; ++key)
                    matrix[key * head_dim + value] += key_row[key] * correction;
            }
            for (std::size_t value = 0; value < head_dim; ++value) {
                float sum = 0.0f;
                for (std::size_t key = 0; key < head_dim; ++key)
                    sum += matrix[key * head_dim + value] * query_row[key];
                output[base + value] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// decoder layer
// ---------------------------------------------------------------------------

// RMS norm with an optional gain vector. Accumulates in double because the
// sums here run over 1536 channels and the reference normalizes in fp32 from a
// bf16 input, so the extra headroom costs nothing and removes a divergence.
inline void rms_norm(
    const float* input, const float* gain, std::size_t size, float epsilon,
    float* output
) {
    double square = 0.0;
    for (std::size_t i = 0; i < size; ++i)
        square += static_cast<double>(input[i]) * input[i];
    const float inverse = 1.0f / std::sqrt(
        static_cast<float>(square / static_cast<double>(size)) + epsilon);
    for (std::size_t i = 0; i < size; ++i)
        output[i] = input[i] * inverse * (gain ? gain[i] : 1.0f);
}

// Causal depthwise convolution over one token, plus SiLU, advancing the window.
//
// `window` holds the previous `width - 1` inputs per channel and is updated in
// place, so prefill and decode are the same call.
inline void short_conv_step(
    const float* input, const float* weights, std::size_t channels,
    std::size_t width, float* window, float* output
) {
    const std::size_t history = width - 1;
    for (std::size_t channel = 0; channel < channels; ++channel) {
        const float* taps = weights + channel * width;
        float* past = window + channel * history;
        float total = 0.0f;
        for (std::size_t i = 0; i < history; ++i) total += past[i] * taps[i];
        total += input[channel] * taps[history];
        // SiLU, matching ShortConvolution(activation='silu').
        output[channel] = total / (1.0f + std::exp(-total));
        for (std::size_t i = 0; i + 1 < history; ++i) past[i] = past[i + 1];
        if (history) past[history - 1] = input[channel];
    }
}

struct Geometry {
    std::size_t hidden = 0, heads = 0;
    std::size_t qk_nope = 0, qk_rope = 0, v_head_dim = 0;
    std::size_t q_lora = 0, kv_lora = 0;
    std::size_t head_dim = 0, conv_width = 0;
    std::size_t dense_size = 0, expert_size = 0, shared_size = 0;
    std::size_t experts = 0, experts_used = 0, groups = 0, groups_used = 0;
    float rope_theta = 0.0f, epsilon = 1e-6f;
    float weight_scale = 1.0f;
    bool normalize_weights = true;
};

struct MlaWeights {
    // The query arrives one of two ways, selected by `Geometry::q_lora`. The
    // large checkpoints factor it (q_a -> RMS norm -> q_b, the DeepSeek-V2
    // shape); Ling 3.0 Flash sets `q_lora_rank: null` and projects hidden
    // straight to heads*qk, so `q` is set and the three below are not. Both
    // land in the same `query` buffer, and everything after is identical.
    Matrix q;
    Matrix q_a;
    const float* q_a_norm = nullptr;
    Matrix q_b;
    Matrix kv_a_mqa;
    const float* kv_a_norm = nullptr;
    // Stays f32 even when the rest of the layer is quantized: the absorbed
    // attention accumulates along kv_b's rows rather than dotting them, so it
    // needs them materialized. At 4096x512 that is 8 MB a layer, 48 MB across
    // the six MLA layers -- negligible against the 4.6 GB the rest saves.
    const float* kv_b = nullptr;
    Matrix gate;
    Matrix output;
};

struct KdaWeights {
    Matrix query;
    Matrix key;
    Matrix value;
    const float* query_conv = nullptr;
    const float* key_conv = nullptr;
    const float* value_conv = nullptr;
    Matrix decay;   // f_proj
    Matrix beta;    // b_proj
    Matrix gate;    // g_proj
    const float* a_log = nullptr;
    const float* dt_bias = nullptr;
    const float* norm = nullptr;    // o_norm
    Matrix output;
};

struct FfnWeights {
    // Dense block (layers below first_k_dense_replace).
    Matrix gate;
    Matrix up;
    Matrix down;
    // Routed block.
    Matrix router;
    const float* router_bias = nullptr;
    Matrix gate_experts;
    Matrix up_experts;
    Matrix down_experts;
    Matrix shared_gate;
    Matrix shared_up;
    Matrix shared_down;
};

struct LayerWeights {
    bool full_attention = false;
    bool routed = false;
    const float* attention_norm = nullptr;
    const float* ffn_norm = nullptr;
    // The SwiGLU clamps are per LAYER and differ between the routed experts
    // and the shared one, so they live here rather than in Geometry. Ling 3.0
    // Flash turns them on only for the last eight blocks (4.0 routed; 5.0 then
    // 7.0 shared) and leaves the rest at zero, which reads as "off".
    float swiglu_limit_exp = 0.0f;
    float swiglu_limit_shexp = 0.0f;
    MlaWeights mla;
    KdaWeights kda;
    FfnWeights ffn;
};

// Per-layer state. MLA layers grow with context; KDA layers do not, which is
// the whole reason the architecture is 3:1 in KDA's favour.
struct LayerCache {
    // MLA: the compressed latent and the shared rope key, one row per position.
    std::vector<float> latents;
    std::vector<float> rope_keys;
    std::size_t positions = 0;
    // KDA: the recurrent state and the three convolution windows.
    std::vector<float> state;
    std::vector<float> query_window, key_window, value_window;
};

inline void reset_cache(LayerCache& cache, const LayerWeights& weights,
                        const Geometry& geometry, std::size_t capacity) {
    cache.positions = 0;
    if (weights.full_attention) {
        cache.latents.assign(capacity * geometry.kv_lora, 0.0f);
        cache.rope_keys.assign(capacity * geometry.qk_rope, 0.0f);
        return;
    }
    const std::size_t channels = geometry.heads * geometry.head_dim;
    const std::size_t history = geometry.conv_width - 1;
    cache.state.assign(geometry.heads * geometry.head_dim * geometry.head_dim, 0.0f);
    cache.query_window.assign(channels * history, 0.0f);
    cache.key_window.assign(channels * history, 0.0f);
    cache.value_window.assign(channels * history, 0.0f);
}

// One MLA token: low-rank query, compressed KV, rope, attention, head gate,
// output projection. `position` is the absolute position, which is what rope
// needs; the cache supplies everything before it.
inline void mla_step(
    const float* hidden, const MlaWeights& w, const Geometry& g,
    std::size_t position, LayerCache& cache, float* output
) {
    const std::size_t qk = g.qk_nope + g.qk_rope;
    std::vector<float> compressed(g.kv_lora + g.qk_rope);
    matvec(w.kv_a_mqa, hidden, g.hidden, g.kv_lora + g.qk_rope, compressed.data());
    rms_norm(compressed.data(), w.kv_a_norm, g.kv_lora, g.epsilon,
             cache.latents.data() + position * g.kv_lora);
    float* rope_key = cache.rope_keys.data() + position * g.qk_rope;
    for (std::size_t i = 0; i < g.qk_rope; ++i) rope_key[i] = compressed[g.kv_lora + i];
    // The rope span is the whole of this row, so head_dim == rope_dim here.
    partial_rope_norm(rope_key, g.qk_rope, g.qk_rope,
                      static_cast<std::int32_t>(position), g.rope_theta);
    cache.positions = position + 1;

    std::vector<float> query(g.heads * qk);
    if (g.q_lora) {
        std::vector<float> low_rank(g.q_lora), normalized(g.q_lora);
        matvec(w.q_a, hidden, g.hidden, g.q_lora, low_rank.data());
        rms_norm(low_rank.data(), w.q_a_norm, g.q_lora, g.epsilon, normalized.data());
        matvec(w.q_b, normalized.data(), g.q_lora, g.heads * qk, query.data());
    } else {
        matvec(w.q, hidden, g.hidden, g.heads * qk, query.data());
    }
    for (std::size_t head = 0; head < g.heads; ++head)
        partial_rope_norm(query.data() + head * qk, qk, g.qk_rope,
                          static_cast<std::int32_t>(position), g.rope_theta);

    std::vector<float> query_nope(g.heads * g.qk_nope), query_rope(g.heads * g.qk_rope);
    for (std::size_t head = 0; head < g.heads; ++head) {
        for (std::size_t i = 0; i < g.qk_nope; ++i)
            query_nope[head * g.qk_nope + i] = query[head * qk + i];
        for (std::size_t i = 0; i < g.qk_rope; ++i)
            query_rope[head * g.qk_rope + i] = query[head * qk + g.qk_nope + i];
    }

    std::vector<float> attended(g.heads * g.v_head_dim);
    mla_attention_absorbed(query_nope.data(), query_rope.data(), w.kv_b,
                           cache.latents.data(), cache.rope_keys.data(),
                           cache.positions, g.heads, g.qk_nope, g.qk_rope,
                           g.v_head_dim, g.kv_lora, attended.data());

    std::vector<float> gate(g.heads);
    matvec(w.gate, hidden, g.hidden, g.heads, gate.data());
    apply_head_gate(gate.data(), g.heads, g.v_head_dim, attended.data());
    matvec(w.output, attended.data(), g.heads * g.v_head_dim, g.hidden, output);
}

// One KDA token: three convolved projections, the recurrence, the gated output
// norm, then the output projection.
inline void kda_step(
    const float* hidden, const KdaWeights& w, const Geometry& g,
    LayerCache& cache, float* output
) {
    const std::size_t channels = g.heads * g.head_dim;
    std::vector<float> projected(channels);
    std::vector<float> queries(channels), keys(channels), values(channels);

    matvec(w.query, hidden, g.hidden, channels, projected.data());
    short_conv_step(projected.data(), w.query_conv, channels, g.conv_width,
                    cache.query_window.data(), queries.data());
    matvec(w.key, hidden, g.hidden, channels, projected.data());
    short_conv_step(projected.data(), w.key_conv, channels, g.conv_width,
                    cache.key_window.data(), keys.data());
    matvec(w.value, hidden, g.hidden, channels, projected.data());
    short_conv_step(projected.data(), w.value_conv, channels, g.conv_width,
                    cache.value_window.data(), values.data());

    std::vector<float> decay(channels), beta(g.heads), attended(channels);
    matvec(w.decay, hidden, g.hidden, channels, decay.data());
    matvec(w.beta, hidden, g.hidden, g.heads, beta.data());
    kda_recurrence(queries.data(), keys.data(), values.data(), decay.data(),
                   beta.data(), w.a_log, w.dt_bias, 1, g.heads, g.head_dim,
                   g.epsilon, cache.state.data(), attended.data());

    // Gated output norm: RMS norm per head with a shared gain, times the
    // sigmoid of the gate projection. This is FusedRMSNormGated(activation
    // 'sigmoid'), not a plain norm followed by a separate gate.
    std::vector<float> gate(channels);
    matvec(w.gate, hidden, g.hidden, channels, gate.data());
    for (std::size_t head = 0; head < g.heads; ++head) {
        float* row = attended.data() + head * g.head_dim;
        rms_norm(row, w.norm, g.head_dim, g.epsilon, row);
        const float* head_gate = gate.data() + head * g.head_dim;
        for (std::size_t i = 0; i < g.head_dim; ++i)
            row[i] *= 1.0f / (1.0f + std::exp(-head_gate[i]));
    }
    matvec(w.output, attended.data(), channels, g.hidden, output);
}

// Weight-stationary matmul: W [outputs][inputs] times `tokens` input vectors.
//
// The point is weight traffic. A matvec reads the whole matrix to produce one
// output vector; this reads each row once and applies it to every token, so a
// prefill of T tokens moves the weights once instead of T times. Decode is
// bandwidth-bound here, and prefill was paying that bound per token.
//
// `input` is [tokens][inputs], `output` is [tokens][outputs].
inline void matmul(
    Matrix weights, const float* input, std::size_t tokens,
    std::size_t inputs, std::size_t outputs, float* output
) {
    if (tokens == 1) {
        matvec(weights, input, inputs, outputs, output);
        return;
    }
    const auto stride = row_bytes(weights.type, inputs);
    const auto* base = reinterpret_cast<const std::uint8_t*>(weights.data);
    const auto type = weights.type;

    enum class Path { Scalar, Avx2, Avx512 };
    [[maybe_unused]] Path path = Path::Scalar;
    // 8, 12, 13, 14 are the quantized types the register-blocked multi-input
    // kernels cover (what the HF quantizer emits), plus the f16/bf16 rows
    // added alongside them. Types with only a single-input SIMD dot still take
    // `path`; they just skip the quad/oct loops below.
    [[maybe_unused]] const bool multi =
        type == 1 || type == 8 || type == 12 || type == 13 || type == 14 ||
        type == 30;
#ifndef COLIBRI_BAILING_NO_SIMD
    if (type != 0 && inputs % simd_dot_granule(type) == 0) {
        const auto features = colibri_cpu_features();
        if ((features & 2u) && simd_dot_avx512_type(type)) path = Path::Avx512;
        else if ((features & 1u) && simd_dot_avx2_type(type)) path = Path::Avx2;
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads()) \
        if (outputs * inputs >= 65536 && !omp_in_parallel())
#endif
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(outputs); ++row) {
        const auto index = static_cast<std::size_t>(row);
        const std::uint8_t* weight_row = base + index * stride;
        std::size_t token = 0;
#ifndef COLIBRI_BAILING_NO_SIMD
        // The register-blocked kernels decode a weight row ONCE and apply it to
        // several activation vectors. That is the actual point of batching --
        // the single-input kernel would re-read and re-decode the row per
        // token, which is why a naive batch only bought ~1.5x.
        if (path != Path::Scalar && multi) {
            const int width = static_cast<int>(inputs);
            for (; token + 8 <= tokens && path == Path::Avx512; token += 8) {
                const float* group[8];
                for (int i = 0; i < 8; ++i) group[i] = input + (token + i) * inputs;
                float values[8];
                qwen_quant_dot_oct_avx512(base, type, group, width, index, values);
                for (int i = 0; i < 8; ++i)
                    output[(token + i) * outputs + index] = values[i];
            }
            for (; token + 4 <= tokens; token += 4) {
                const float* group[4];
                for (int i = 0; i < 4; ++i) group[i] = input + (token + i) * inputs;
                float values[4];
                if (path == Path::Avx512)
                    qwen_quant_dot_quad_avx512(base, type, group, width, index, values);
                else
                    qwen_quant_dot_quad_avx2(base, type, group, width, index, values);
                for (int i = 0; i < 4; ++i)
                    output[(token + i) * outputs + index] = values[i];
            }
        }
#endif
        for (; token < tokens; ++token) {
            const float* x = input + token * inputs;
            float value;
#ifndef COLIBRI_BAILING_NO_SIMD
            if (path == Path::Avx512)
                value = qwen_quant_dot_avx512(base, type, x, static_cast<int>(inputs), index);
            else if (path == Path::Avx2)
                value = qwen_quant_dot_avx2(base, type, x, static_cast<int>(inputs), index);
            else
#endif
                value = row_dot(weight_row, type, x, inputs);
            output[token * outputs + index] = value;
        }
    }
}

// A batch of tokens through one MoE block.
//
// Routing is per token, so the expert matmuls cannot simply be widened. Instead
// the tokens are grouped by chosen expert and each expert runs once over its
// own group -- which is where the weight-traffic saving comes from, since a
// popular expert's 1536x512 matrices are then read once rather than once per
// token that selected it.
inline void moe_block_batch(
    const float* hidden, std::size_t tokens,
    Matrix router_weights, const float* router_bias,
    Matrix gate_experts, Matrix up_experts, Matrix down_experts,
    Matrix shared_gate, Matrix shared_up, Matrix shared_down,
    std::size_t hidden_size, std::size_t expert_size, std::size_t shared_size,
    std::size_t experts, std::size_t used, std::size_t groups,
    std::size_t groups_used, float weight_scale, bool normalize,
    float swiglu_limit, float shared_swiglu_limit, float* output
) {
    std::vector<float> logits(tokens * experts);
    ProfileScope* route = new ProfileScope(profile().moe_route);
    matmul(router_weights, hidden, tokens, hidden_size, experts, logits.data());

    std::vector<std::int32_t> chosen(tokens * used);
    std::vector<float> weights(tokens * used);
    for (std::size_t token = 0; token < tokens; ++token)
        moe_router(logits.data() + token * experts, router_bias, experts, used,
                   groups, groups_used, weight_scale, normalize,
                   chosen.data() + token * used, weights.data() + token * used);

    std::fill(output, output + tokens * hidden_size, 0.0f);
    delete route;

    // Invert the routing: which tokens picked each expert, and with what weight.
    std::vector<std::vector<std::pair<std::size_t, float>>> assignment(experts);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t slot = 0; slot < used; ++slot)
            assignment[static_cast<std::size_t>(chosen[token * used + slot])]
                .emplace_back(token, weights[token * used + slot]);

    const auto narrow_stride = expert_size * row_bytes(gate_experts.type, hidden_size);
    const auto wide_stride = hidden_size * row_bytes(down_experts.type, expert_size);
    const auto* gate_base = reinterpret_cast<const std::uint8_t*>(gate_experts.data);
    const auto* up_base = reinterpret_cast<const std::uint8_t*>(up_experts.data);
    const auto* down_base = reinterpret_cast<const std::uint8_t*>(down_experts.data);

    // The expert loop stays SERIAL, with the parallelism inside each expert's
    // matmul. An expert-parallel version was tried and measured slower --
    // 165 tok/s against 183 -- because experts accumulate into shared token
    // rows, so each thread needs a private tokens x hidden buffer and the
    // reduction over 16 of them costs more than the fork it saves.
    //
    // The hypothesis that motivated it (8800 OpenMP forks per prefill dominate)
    // was wrong: the expert matmuls run at ~25 GFLOP/s against ~520 for the
    // batched projections, but the gap is arithmetic intensity, not fork
    // overhead. Each expert sees ~16 tokens and 2.4M parameters, so its weights
    // are read almost per-token however the loop is arranged.
    std::vector<float> gathered, gate, up, activated, projected;
    for (std::size_t expert = 0; expert < experts; ++expert) {
        const auto& members = assignment[expert];
        if (members.empty()) continue;
        const std::size_t count = members.size();
        gathered.resize(count * hidden_size);
        { ProfileScope gather(profile().moe_gather);
          for (std::size_t i = 0; i < count; ++i)
            std::memcpy(gathered.data() + i * hidden_size,
                        hidden + members[i].first * hidden_size,
                        hidden_size * sizeof(float)); }
        gate.resize(count * expert_size);
        up.resize(count * expert_size);
        activated.resize(count * expert_size);
        projected.resize(count * hidden_size);
        ProfileScope* work = new ProfileScope(profile().moe_experts);
        matmul({gate_base + expert * narrow_stride, gate_experts.type},
               gathered.data(), count, hidden_size, expert_size, gate.data());
        matmul({up_base + expert * narrow_stride, up_experts.type},
               gathered.data(), count, hidden_size, expert_size, up.data());
        swiglu(gate.data(), up.data(), count * expert_size, swiglu_limit,
               activated.data());
        matmul({down_base + expert * wide_stride, down_experts.type},
               activated.data(), count, expert_size, hidden_size, projected.data());
        delete work;
        ProfileScope scatter(profile().moe_gather);
        for (std::size_t i = 0; i < count; ++i) {
            float* target = output + members[i].first * hidden_size;
            const float* source = projected.data() + i * hidden_size;
            const float weight = members[i].second;
            for (std::size_t j = 0; j < hidden_size; ++j) target[j] += source[j] * weight;
        }
    }

    if (shared_gate && shared_up && shared_down && shared_size) {
        ProfileScope shared(profile().moe_shared);
        std::vector<float> sg(tokens * shared_size), su(tokens * shared_size),
            sa(tokens * shared_size), sp(tokens * hidden_size);
        matmul(shared_gate, hidden, tokens, hidden_size, shared_size, sg.data());
        matmul(shared_up, hidden, tokens, hidden_size, shared_size, su.data());
        swiglu(sg.data(), su.data(), tokens * shared_size, shared_swiglu_limit, sa.data());
        matmul(shared_down, sa.data(), tokens, shared_size, hidden_size, sp.data());
        for (std::size_t i = 0; i < tokens * hidden_size; ++i) output[i] += sp[i];
    }
}

// One token through one decoder layer, in place.
//
// Both residual joins are plain adds around a pre-norm, which is the ordinary
// arrangement -- unlike Muse Glimmer, this architecture has no extra norm
// between the block and the residual (modeling:1058-1108).
inline void decoder_layer(
    const float* input, const LayerWeights& w, const Geometry& g,
    std::size_t position, LayerCache& cache, float* output
) {
    std::vector<float> normalized(g.hidden), branch(g.hidden), residual(g.hidden);

    // Instrumented with the same buckets as the batched path. It went without
    // for a long time, and the consequence was worse than a missing number:
    // `decoder_layer_batch` delegates here at tokens == 1, so every DECODE
    // token was invisible and the profile silently reported prefill only.
    if (profiling()) profile().tokens += 1;
    { ProfileScope scope(profile().norms);
      rms_norm(input, w.attention_norm, g.hidden, g.epsilon, normalized.data()); }
    if (w.full_attention) {
        ProfileScope scope(profile().mla);
        mla_step(normalized.data(), w.mla, g, position, cache, branch.data());
    } else {
        ProfileScope scope(profile().kda);
        kda_step(normalized.data(), w.kda, g, cache, branch.data());
    }
    for (std::size_t i = 0; i < g.hidden; ++i) residual[i] = input[i] + branch[i];

    { ProfileScope scope(profile().norms);
      rms_norm(residual.data(), w.ffn_norm, g.hidden, g.epsilon, normalized.data()); }
    if (w.routed) {
        ProfileScope scope(profile().moe);
        moe_block(normalized.data(), w.ffn.router, w.ffn.router_bias,
                  w.ffn.gate_experts, w.ffn.up_experts, w.ffn.down_experts,
                  w.ffn.shared_gate, w.ffn.shared_up, w.ffn.shared_down,
                  g.hidden, g.expert_size, g.shared_size, g.experts,
                  g.experts_used, g.groups, g.groups_used, g.weight_scale,
                  g.normalize_weights, w.swiglu_limit_exp, w.swiglu_limit_shexp,
                  branch.data());
    } else {
        ProfileScope scope(profile().dense_ffn);
        std::vector<float> gate(g.dense_size), up(g.dense_size), activated(g.dense_size);
        matvec(w.ffn.gate, normalized.data(), g.hidden, g.dense_size, gate.data());
        matvec(w.ffn.up, normalized.data(), g.hidden, g.dense_size, up.data());
        swiglu(gate.data(), up.data(), g.dense_size, w.swiglu_limit_exp, activated.data());
        matvec(w.ffn.down, activated.data(), g.dense_size, g.hidden, branch.data());
    }
    for (std::size_t i = 0; i < g.hidden; ++i) output[i] = residual[i] + branch[i];
}

// A batch of tokens through one decoder layer.
//
// What batches and what does not:
//   * norms, projections and the feed-forward batch cleanly -- they are the
//     weight-heavy parts and the whole reason to do this;
//   * the KDA recurrence and the MLA attention do NOT. Both are sequential in
//     position by construction, so they still run token by token, reusing the
//     projections computed above.
//
// `inputs` and `output` are [tokens][hidden]. `first_position` is the absolute
// position of the first token.
// Destination buffers for a KDA layer's per-row transition inputs, recorded
// during a speculative-verify pass. The recurrence's state update depends on
// exactly these five projections (the gate only shapes the discarded output),
// so retaining them lets a rejected round rebuild the recurrent state by
// replaying only the convolution and recurrence over the accepted prefix --
// the fold -- instead of re-running the whole forward. Row-major, sized
// rows x channels for q/k/v/decay and rows x heads for beta.
struct KdaCapture {
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* decay = nullptr;
    float* beta = nullptr;
};

inline void decoder_layer_batch(
    const float* inputs, std::size_t tokens, const LayerWeights& w,
    const Geometry& g, std::size_t first_position, LayerCache& cache,
    float* output, KdaCapture* capture = nullptr
) {
    // A capture must record what THIS pass computes, so it keeps the batch
    // path even for one token rather than delegating to the matvec-shaped
    // single-token layer.
    if (tokens == 1 && !capture) {
        decoder_layer(inputs, w, g, first_position, cache, output);
        return;
    }
    const std::size_t hidden = g.hidden;
    std::vector<float> normalized(tokens * hidden), residual(tokens * hidden),
        branch(tokens * hidden);

    { ProfileScope scope(profile().norms);
      for (std::size_t t = 0; t < tokens; ++t)
        rms_norm(inputs + t * hidden, w.attention_norm, hidden, g.epsilon,
                 normalized.data() + t * hidden); }
    if (profiling()) profile().tokens += tokens;

    if (w.full_attention) {
        // Batched: the low-rank query path, the compressed KV, and the gate.
        // Sequential: rope and the attention itself, which need the position.
        const std::size_t qk = g.qk_nope + g.qk_rope;
        std::vector<float> low_rank(tokens * g.q_lora), qnorm(tokens * g.q_lora),
            query(tokens * g.heads * qk), compressed(tokens * (g.kv_lora + g.qk_rope)),
            gate(tokens * g.heads), attended(tokens * g.heads * g.v_head_dim);
        ProfileScope* stage = new ProfileScope(profile().projections);
        if (g.q_lora) {
            matmul(w.mla.q_a, normalized.data(), tokens, hidden, g.q_lora, low_rank.data());
            for (std::size_t t = 0; t < tokens; ++t)
                rms_norm(low_rank.data() + t * g.q_lora, w.mla.q_a_norm, g.q_lora,
                         g.epsilon, qnorm.data() + t * g.q_lora);
            matmul(w.mla.q_b, qnorm.data(), tokens, g.q_lora, g.heads * qk, query.data());
        } else {
            matmul(w.mla.q, normalized.data(), tokens, hidden, g.heads * qk,
                   query.data());
        }
        matmul(w.mla.kv_a_mqa, normalized.data(), tokens, hidden,
               g.kv_lora + g.qk_rope, compressed.data());
        matmul(w.mla.gate, normalized.data(), tokens, hidden, g.heads, gate.data());
        delete stage;

        {
        ProfileScope attention(profile().mla);
        for (std::size_t t = 0; t < tokens; ++t) {
            const std::size_t position = first_position + t;
            const float* row = compressed.data() + t * (g.kv_lora + g.qk_rope);
            rms_norm(row, w.mla.kv_a_norm, g.kv_lora, g.epsilon,
                     cache.latents.data() + position * g.kv_lora);
            float* rope_key = cache.rope_keys.data() + position * g.qk_rope;
            std::memcpy(rope_key, row + g.kv_lora, g.qk_rope * sizeof(float));
            partial_rope_norm(rope_key, g.qk_rope, g.qk_rope,
                              static_cast<std::int32_t>(position), g.rope_theta);
            cache.positions = position + 1;

            float* q = query.data() + t * g.heads * qk;
            for (std::size_t head = 0; head < g.heads; ++head)
                partial_rope_norm(q + head * qk, qk, g.qk_rope,
                                  static_cast<std::int32_t>(position), g.rope_theta);
            std::vector<float> qn(g.heads * g.qk_nope), qr(g.heads * g.qk_rope);
            for (std::size_t head = 0; head < g.heads; ++head) {
                std::memcpy(qn.data() + head * g.qk_nope, q + head * qk,
                            g.qk_nope * sizeof(float));
                std::memcpy(qr.data() + head * g.qk_rope, q + head * qk + g.qk_nope,
                            g.qk_rope * sizeof(float));
            }
            float* out = attended.data() + t * g.heads * g.v_head_dim;
            mla_attention_absorbed(qn.data(), qr.data(), w.mla.kv_b,
                                   cache.latents.data(), cache.rope_keys.data(),
                                   cache.positions, g.heads, g.qk_nope, g.qk_rope,
                                   g.v_head_dim, g.kv_lora, out);
            apply_head_gate(gate.data() + t * g.heads, g.heads, g.v_head_dim, out);
        }
        }
        ProfileScope tail(profile().projections);
        matmul(w.mla.output, attended.data(), tokens, g.heads * g.v_head_dim,
               hidden, branch.data());
    } else {
        const std::size_t channels = g.heads * g.head_dim;
        std::vector<float> q(tokens * channels), k(tokens * channels),
            v(tokens * channels), decay(tokens * channels), beta(tokens * g.heads),
            gate(tokens * channels), attended(tokens * channels);
        ProfileScope* stage = new ProfileScope(profile().projections);
        matmul(w.kda.query, normalized.data(), tokens, hidden, channels, q.data());
        matmul(w.kda.key, normalized.data(), tokens, hidden, channels, k.data());
        matmul(w.kda.value, normalized.data(), tokens, hidden, channels, v.data());
        matmul(w.kda.decay, normalized.data(), tokens, hidden, channels, decay.data());
        matmul(w.kda.beta, normalized.data(), tokens, hidden, g.heads, beta.data());
        matmul(w.kda.gate, normalized.data(), tokens, hidden, channels, gate.data());
        delete stage;
        if (capture) {
            std::memcpy(capture->q, q.data(), tokens * channels * sizeof(float));
            std::memcpy(capture->k, k.data(), tokens * channels * sizeof(float));
            std::memcpy(capture->v, v.data(), tokens * channels * sizeof(float));
            std::memcpy(capture->decay, decay.data(), tokens * channels * sizeof(float));
            std::memcpy(capture->beta, beta.data(), tokens * g.heads * sizeof(float));
        }

        {
        ProfileScope recurrence(profile().kda);
        // The convolution and the recurrence are both sequential in position.
        std::vector<float> qc(tokens * channels), kc(tokens * channels),
            vc(tokens * channels);
        for (std::size_t t = 0; t < tokens; ++t) {
            short_conv_step(q.data() + t * channels, w.kda.query_conv, channels,
                            g.conv_width, cache.query_window.data(), qc.data() + t * channels);
            short_conv_step(k.data() + t * channels, w.kda.key_conv, channels,
                            g.conv_width, cache.key_window.data(), kc.data() + t * channels);
            short_conv_step(v.data() + t * channels, w.kda.value_conv, channels,
                            g.conv_width, cache.value_window.data(), vc.data() + t * channels);
        }
        kda_recurrence(qc.data(), kc.data(), vc.data(), decay.data(), beta.data(),
                       w.kda.a_log, w.kda.dt_bias, tokens, g.heads, g.head_dim,
                       g.epsilon, cache.state.data(), attended.data());
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t head = 0; head < g.heads; ++head) {
                float* row = attended.data() + t * channels + head * g.head_dim;
                rms_norm(row, w.kda.norm, g.head_dim, g.epsilon, row);
                const float* head_gate = gate.data() + t * channels + head * g.head_dim;
                for (std::size_t i = 0; i < g.head_dim; ++i)
                    row[i] *= 1.0f / (1.0f + std::exp(-head_gate[i]));
            }
        }
        ProfileScope tail(profile().projections);
        matmul(w.kda.output, attended.data(), tokens, channels, hidden, branch.data());
    }

    for (std::size_t i = 0; i < tokens * hidden; ++i)
        residual[i] = inputs[i] + branch[i];
    for (std::size_t t = 0; t < tokens; ++t)
        rms_norm(residual.data() + t * hidden, w.ffn_norm, hidden, g.epsilon,
                 normalized.data() + t * hidden);

    if (w.routed) {
        ProfileScope scope(profile().moe);
        moe_block_batch(normalized.data(), tokens, w.ffn.router, w.ffn.router_bias,
                        w.ffn.gate_experts, w.ffn.up_experts, w.ffn.down_experts,
                        w.ffn.shared_gate, w.ffn.shared_up, w.ffn.shared_down,
                        hidden, g.expert_size, g.shared_size, g.experts,
                        g.experts_used, g.groups, g.groups_used, g.weight_scale,
                        g.normalize_weights, w.swiglu_limit_exp,
                        w.swiglu_limit_shexp, branch.data());
    } else {
        ProfileScope scope(profile().dense_ffn);
        std::vector<float> gate(tokens * g.dense_size), up(tokens * g.dense_size),
            activated(tokens * g.dense_size);
        matmul(w.ffn.gate, normalized.data(), tokens, hidden, g.dense_size, gate.data());
        matmul(w.ffn.up, normalized.data(), tokens, hidden, g.dense_size, up.data());
        swiglu(gate.data(), up.data(), tokens * g.dense_size, w.swiglu_limit_exp,
               activated.data());
        matmul(w.ffn.down, activated.data(), tokens, g.dense_size, hidden, branch.data());
    }
    for (std::size_t i = 0; i < tokens * hidden; ++i)
        output[i] = residual[i] + branch[i];
}

}  // namespace colibri::v2::bailing
