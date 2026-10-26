#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "flyweight_v2_bailing.hpp"
#include "flyweight_cpu_shim_geometry.hpp"
#include "flyweight_gpu_driver.h"

namespace flyweight::v2::raon {

// ---------------------------------------------------------------------------
// profiling
// ---------------------------------------------------------------------------
//
// Same shape as bailing's Profile, but the buckets track what is sequential
// in Raon rather than what is batched: the mHC mixing and the RDA recurrence
// run per token, and the point of this report is to expose exactly the serial
// costs a batched model never pays. Enabled with FLYWEIGHT_RAON_PROFILE=1 and
// reported when the runtime is destroyed.
struct Profile {
    double mhc = 0.0;            // hc pre + post stream mixing, both blocks
    double rda_projection = 0.0; // rda_step matvecs and convolution
    double rda_recurrence = 0.0; // the head loop: decay, delta rule, readout
    double mla = 0.0;            // whole mla_step
    double moe = 0.0;            // whole ffn_step on a MoE layer
    double moe_route = 0.0;
    double moe_experts = 0.0;
    double moe_shared = 0.0;
    double dense_ffn = 0.0;
    std::uint64_t tokens = 0;

    void report(std::FILE* out) const {
        const double total = mhc + rda_projection + rda_recurrence + mla + moe + dense_ffn;
        if (total <= 0.0) return;
        std::fprintf(out, "[flyweight] raon profile over %llu tokens "
                          "(%.3f s accounted)\n",
                     static_cast<unsigned long long>(tokens), total);
        const struct { const char* name; double value; } rows[] = {
            {"mHC pre/post mixing", mhc},
            {"RDA projections+conv", rda_projection},
            {"RDA recurrence (per-head)", rda_recurrence},
            {"MLA", mla},
            {"MoE total", moe},
            {"  .. routing", moe_route},
            {"  .. expert matmuls", moe_experts},
            {"  .. shared expert", moe_shared},
            {"dense FFN", dense_ffn},
        };
        for (const auto& row : rows)
            std::fprintf(out, "    %-34s %7.3f s  %5.1f%%\n",
                         row.name, row.value, 100.0 * row.value / total);
        if (tokens)
            std::fprintf(out, "    %-34s %7.3f ms\n",
                         "accounted per token", 1000.0 * total / static_cast<double>(tokens));
    }
};

// Buckets are per-RUNTIME-instance in spirit but the structs are header state;
// one process-wide set keeps bailing's column identical, so anything reading
// a saved report gets the same shape either way.
inline Profile raon_profile;
inline Profile& profile() { return raon_profile; }

inline bool profiling() {
    static const bool on = [] {
        const char* value = std::getenv("FLYWEIGHT_RAON_PROFILE");
        return value && value[0] == '1';
    }();
    return on;
}

// Adds its lifetime to `bucket` when profiling is on; off, this costs one
// predictable branch per phase.
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

struct GpuLayerDispatch {
    bool is_moe = false;
    bool pinned = false;
    std::uint64_t router = 0;
    std::uint32_t router_type = 0;
    const float* router_bias = nullptr;
    std::uint64_t gate_exps = 0;
    std::uint32_t gate_type = 0;
    std::uint64_t up_exps = 0;
    std::uint32_t up_type = 0;
    std::uint64_t down_exps = 0;
    std::uint32_t down_type = 0;
    std::uint64_t shared_gate = 0;
    std::uint32_t shared_gate_type = 0;
    std::uint64_t shared_up = 0;
    std::uint32_t shared_up_type = 0;
    std::uint64_t shared_down = 0;
    std::uint32_t shared_down_type = 0;

    // Attention-side projection offload (independent of MoE pinning).
    // For RDA layers the slot order is q, k, v, gate (ssm_g), output
    // (attn_output); for MLA layers it is q_a, q_b, kv_a_mqa, gate
    // (attn_gate), output (attn_output). Slots 0..3 all consume the same
    // branch-norm input token and slot 4 consumes the attention output.
    //
    // When the shared-input projections agree on a quant type, the runtime
    // squeezes them into one fused matrix (rows concatenated in slot order
    // q, k, v, gate, or q_a, gate, kv_a_mqa) and sets attn_packed: one
    // matvec launch over the fused rows replaces four (or three) launches,
    // cutting per-layer launch/download latency. Slots 1..3 are then
    // unused; slot 4 always stays separate (it consumes the attention
    // output, which does not exist yet at pack time).
    bool attn_gpu = false;
    bool attn_packed = false;
    std::uint64_t attn_weight[5] = {};
    std::uint32_t attn_type[5] = {};
};

struct GpuScratchDispatch {
    std::uint64_t input = 0;
    std::uint64_t router_logits = 0;
    std::uint64_t expert_gate = 0;
    std::uint64_t expert_up = 0;
    std::uint64_t expert_act = 0;
    std::uint64_t expert_down = 0;
    std::uint64_t moe_accum = 0;
    // Attention-projection scratch. proj_pack holds the x_norm-fed launch
    // outputs contiguously (RDA: q/k/v/gate at channels f32 each; MLA:
    // [q_a 576 | gate heads*v_head | kv 576]) so one download covers the
    // whole phase; the output-projection pair feeds the CPU-side mHC post.
    std::uint64_t proj_pack = 0;    // 4 * channels f32
    std::uint64_t proj_query = 0;   // heads * (qk_nope + qk_rope) f32
    std::uint64_t attn_in = 0;      // channels f32
    std::uint64_t proj_hidden = 0;  // hidden f32
    int (*matvec_fn)(std::uint32_t type, std::uint64_t matrix, std::uint64_t in, std::uint64_t out, int in_size, int out_size, std::uint64_t stream, float scale) = nullptr;
    int (*launch_situ_glu)(std::uint64_t gate, std::uint64_t val, std::uint64_t out, int size, float beta, float linear_beta) = nullptr;
};

using Matrix = flyweight::v2::bailing::Matrix;
using flyweight::v2::bailing::row_bytes;
using flyweight::v2::bailing::row_decode;
using flyweight::v2::bailing::row_dot;
using flyweight::v2::bailing::row_kernel;
using flyweight::v2::bailing::matvec_threads;
using flyweight::v2::bailing::matvec;
using flyweight::v2::bailing::matmul;
using flyweight::v2::bailing::matrix_decode;
using flyweight::v2::bailing::rms_norm;
using flyweight::v2::bailing::short_conv_step;
using flyweight::v2::bailing::mla_attention_absorbed;

struct Geometry {
    std::size_t hidden = 2560;
    std::size_t heads = 40;
    std::size_t layers = 25;
    std::vector<std::uint32_t> head_count_kv;  // 0 = RDA, 1 = MLA

    // Raon-mHC
    std::size_t mhc_streams = 4;
    float mhc_eps = 1e-5f;
    float mhc_magnitude = 2.0f;

    // Activations & Norms
    float situ_beta = 4.0f;
    float situ_linear_beta = 25.0f;
    float epsilon = 1e-6f;
    float latent_epsilon = 1e-6f;

    // RDA
    std::size_t head_dim = 128;
    std::size_t conv_width = 4;
    std::size_t ssm_dt_rank = 128;
    float kda_gate_lower_bound = -5.0f;

    // MLA (NoPE)
    std::size_t q_lora = 576;
    std::size_t kv_lora = 512;
    std::size_t qk_nope = 128;
    std::size_t qk_rope = 64;
    std::size_t v_head_dim = 128;

    // FFN & MoE
    std::size_t dense_size = 12032;
    std::size_t expert_size = 1088;
    std::size_t leading_dense_layers = 1;
    std::size_t experts = 128;
    std::size_t experts_used = 6;
    std::size_t shared_experts = 1;
    std::size_t shared_size = 1088;
    float expert_weights_scale = 1.0f;
    bool expert_weights_norm = true;

    // Storage of the MLA cache (normalized latents and rope keys), in the
    // placement plan's codes: 0 f32, 1 f16, 2 bf16. The RDA state stays f32.
    int cache_type = 0;
    // False when the runtime keeps the RDA state and the MLA cache on the
    // device (device decode): reset_cache then allocates no host copies.
    bool host_cache = true;

    bool is_rda(std::size_t layer) const {
        if (layer < head_count_kv.size()) return head_count_kv[layer] == 0;
        return true;
    }
};

struct RaonMhcWeights {
    Matrix fn;                        // [17, 4 * hidden]
    const float* base = nullptr;       // [17]
    const float* scale = nullptr;      // [4]: alpha_pre, alpha_post, alpha_z, eta
};

struct RdaWeights {
    Matrix q, k, v;                   // [5120, hidden]
    const float* q_conv = nullptr;    // [5120 * conv_width]
    const float* k_conv = nullptr;    // [5120 * conv_width]
    const float* v_conv = nullptr;    // [5120 * conv_width]
    Matrix f_a;                       // [128, hidden]
    Matrix f_b;                       // [5120, 128]
    Matrix beta;                      // [40, hidden]
    Matrix rda;                       // [40, hidden] (ssm_rda)
    const float* a = nullptr;         // [40] (ssm_a holds -exp(A_log), not A_log)
    const float* dt_bias = nullptr;   // [5120]
    Matrix gate;                      // [5120, hidden] (ssm_g)
    const float* o_norm = nullptr;    // [128] (ssm_o_norm)
    Matrix output;                    // [hidden, 5120] (attn_output)
};

struct MlaWeights {
    Matrix q_a;                       // [576, hidden]
    const float* q_a_norm = nullptr;  // [576]
    Matrix q_b;                       // [heads * (128 + 64) = 7680, 576]
    Matrix kv_a_mqa;                  // [512 + 64 = 576, hidden]
    const float* kv_a_norm = nullptr; // [512]
    const float* kv_b = nullptr;      // decoded: [heads * (128 + 128), 512]
    Matrix gate;                      // [5120, hidden] (attn_gate)
    Matrix output;                    // [hidden, 5120] (attn_output)
};

struct FfnWeights {
    // Dense block (layer 0)
    Matrix gate;
    Matrix up;
    Matrix down;
    // MoE block (layers 1..24)
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
    bool is_rda = true;
    bool is_moe = false;
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    RaonMhcWeights mhc_attn;
    RaonMhcWeights mhc_ffn;
    RdaWeights rda;
    MlaWeights mla;
    FfnWeights ffn;
};

struct LayerCache {
    // RDA cache
    std::vector<float> rda_state;                      // [heads * head_dim * head_dim]
    std::vector<float> q_window, k_window, v_window;   // [channels * (conv_width - 1)]

    // MLA cache: f32 rows in latents / rope_keys, or with a half-precision
    // cache_type the 16-bit rows in latents16 / rope16 (the f32 vectors stay
    // empty). cache_store / cache_widen hide the difference.
    std::vector<float> latents;                        // [capacity * kv_lora]
    std::vector<float> rope_keys;                      // [capacity * qk_rope]
    std::vector<std::uint16_t> latents16, rope16;
    std::size_t positions = 0;
};

inline std::uint16_t cache_encode(float value, int type) {
    if (type == 1) return flyweight::cpu::float_to_half_bits(value);
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);  // nearest-even
}

inline float cache_decode(std::uint16_t bits, int type) {
    if (type == 1) return qwen_half_value(bits);
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

inline std::size_t cache_element_bytes(int type) { return type == 0 ? 4 : 2; }

// Writes one position: the already-normalized latent and its rope key.
inline void cache_store(LayerCache& cache, const Geometry& g, std::size_t position,
                        const float* latent, const float* rope) {
    if (g.cache_type == 0) {
        std::memcpy(cache.latents.data() + position * g.kv_lora, latent, g.kv_lora * sizeof(float));
        std::memcpy(cache.rope_keys.data() + position * g.qk_rope, rope, g.qk_rope * sizeof(float));
        return;
    }
    std::uint16_t* l = cache.latents16.data() + position * g.kv_lora;
    std::uint16_t* r = cache.rope16.data() + position * g.qk_rope;
    for (std::size_t i = 0; i < g.kv_lora; ++i) l[i] = cache_encode(latent[i], g.cache_type);
    for (std::size_t i = 0; i < g.qk_rope; ++i) r[i] = cache_encode(rope[i], g.cache_type);
}

// Positions [first, first + count) as f32 rows.
inline void cache_widen(const LayerCache& cache, const Geometry& g, std::size_t first,
                        std::size_t count, float* latents, float* rope) {
    if (g.cache_type == 0) {
        std::memcpy(latents, cache.latents.data() + first * g.kv_lora, count * g.kv_lora * sizeof(float));
        std::memcpy(rope, cache.rope_keys.data() + first * g.qk_rope, count * g.qk_rope * sizeof(float));
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads()) if (count >= 256)
#endif
    for (std::int64_t p = 0; p < static_cast<std::int64_t>(count); ++p) {
        const std::uint16_t* l = cache.latents16.data() + (first + p) * g.kv_lora;
        const std::uint16_t* r = cache.rope16.data() + (first + p) * g.qk_rope;
        for (std::size_t i = 0; i < g.kv_lora; ++i) latents[p * g.kv_lora + i] = cache_decode(l[i], g.cache_type);
        for (std::size_t i = 0; i < g.qk_rope; ++i) rope[p * g.qk_rope + i] = cache_decode(r[i], g.cache_type);
    }
}

inline void reset_cache(LayerCache& cache, const LayerWeights& weights,
                        const Geometry& g, std::size_t capacity) {
    if (!g.host_cache) {
        cache = LayerCache{};
        return;
    }
    if (weights.is_rda) {
        const std::size_t channels = g.heads * g.head_dim;
        const std::size_t history = g.conv_width > 1 ? g.conv_width - 1 : 0;
        cache.rda_state.assign(g.heads * g.head_dim * g.head_dim, 0.0f);
        cache.q_window.assign(channels * history, 0.0f);
        cache.k_window.assign(channels * history, 0.0f);
        cache.v_window.assign(channels * history, 0.0f);
    } else {
        cache.positions = 0;
        if (g.cache_type == 0) {
            cache.latents.assign(capacity * g.kv_lora, 0.0f);
            cache.rope_keys.assign(capacity * g.qk_rope, 0.0f);
        } else {
            cache.latents16.assign(capacity * g.kv_lora, 0);
            cache.rope16.assign(capacity * g.qk_rope, 0);
        }
    }
}

// SiTU-GLU activation:
// (gate_beta * tanh(gate / gate_beta) * sigmoid(gate)) * (value_beta * tanh(val / value_beta))
//
// Three transcendentals per element and fully independent across them, which
// is what makes this worth a shared region once the span is large: the dense
// projection spans 12032 elements and one token's routed experts span
// 6 x 1088, so the fork cost is tens of nanoseconds against hundreds of
// microseconds of scalar libm calls.
inline void situ_glu(const float* gate, const float* val, std::size_t size,
                     float beta, float value_beta, float* output) {
    const float inv_beta = 1.0f / beta;
    const float inv_val_beta = 1.0f / value_beta;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads()) \
        if (size >= 4096)
#endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(size); ++index) {
        const std::size_t i = static_cast<std::size_t>(index);
        const float g = gate[i];
        const float v = val[i];
        const float sig = 1.0f / (1.0f + std::exp(-g));
        const float g_act = beta * std::tanh(g * inv_beta) * sig;
        const float v_act = value_beta * std::tanh(v * inv_val_beta);
        output[i] = g_act * v_act;
    }
}

// The per-token half of the mHC pre-mixing, from the already rms-scaled
// mixes: pre weights [4], post weights [4] and the combination matrix [4, 4].
// Split out so the batched prefill can compute every token's mixes with one
// matmul and then run exactly this.
inline void raon_mhc_coefficients(
    const float* mixes, const RaonMhcWeights& w, const Geometry& g,
    float* pre, float* post_out, float* comb_out
) {
    const std::size_t hc = g.mhc_streams;
    // 3. Affine transformations:
    // w.scale: alpha_pre, alpha_post, alpha_z, eta
    // w.base: b_pre [4], b_post [4], b_z [9]
    const float alpha_pre = w.scale[0];
    const float alpha_post = w.scale[1];
    const float alpha_z = w.scale[2];
    const float eta = w.scale[3];

    for (std::size_t i = 0; i < hc; ++i) {
        const float t = mixes[i] * alpha_pre + w.base[i];
        pre[i] = 1.0f / (1.0f + std::exp(-t));
    }

    for (std::size_t i = 0; i < hc; ++i) {
        const float t = mixes[hc + i] * alpha_post + w.base[hc + i];
        post_out[i] = g.mhc_magnitude / (1.0f + std::exp(-t));
    }

    // Helmert coordinates Z [3, 3]
    float Z[3][3];
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            const std::size_t idx = 2 * hc + r * 3 + c;
            Z[r][c] = mixes[idx] * alpha_z + w.base[idx];
        }
    }

    // Helmert matrix Q [4, 3]
    static const float Q[4][3] = {
        {  1.0f / 1.4142135623730951f,  1.0f / 2.449489742783178f,   1.0f / 3.4641016151377544f },
        { -1.0f / 1.4142135623730951f,  1.0f / 2.449489742783178f,   1.0f / 3.4641016151377544f },
        {  0.0f,                        -2.0f / 2.449489742783178f,   1.0f / 3.4641016151377544f },
        {  0.0f,                         0.0f,                       -3.0f / 3.4641016151377544f }
    };

    // T = Q @ Z (4x3)
    float T[4][3];
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            T[r][c] = Q[r][0] * Z[0][c] + Q[r][1] * Z[1][c] + Q[r][2] * Z[2][c];
        }
    }

    // E = T @ Q^T (4x4)
    float E[4][4];
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            E[r][c] = T[r][0] * Q[c][0] + T[r][1] * Q[c][1] + T[r][2] * Q[c][2];
        }
    }

    // chart_base: 0.5 * (I + 0.25) -> diag 0.625, off-diag 0.125
    float max_neg_ratio = 0.0f;
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            const float base_val = (r == c) ? 0.625f : 0.125f;
            const float ratio = -E[r][c] / base_val;
            if (ratio > max_neg_ratio) max_neg_ratio = ratio;
        }
    }
    const float rho = 1.0f + max_neg_ratio;

    // phi = chart_base + E / rho
    // comb = I + sigmoid(eta) * (phi - I) = (1 - gamma) * I + gamma * phi
    const float gamma = 1.0f / (1.0f + std::exp(-eta));
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            const float base_val = (r == c) ? 0.625f : 0.125f;
            const float phi_val = base_val + E[r][c] / rho;
            const float identity = (r == c) ? 1.0f : 0.0f;
            comb_out[r * 4 + c] = identity + gamma * (phi_val - identity);
        }
    }
}

// Raon-mHC pre-mixing:
// Projects flattened 4 streams with RMSNorm scaling and Helmert combination
// to produce branch_in [hidden], post weights [4], and combination matrix [4, 4].
//
// This runs twice per layer per token, and everything it does is a handful of
// dots over the same 4 * hidden floats, so it lives or dies by whether those
// dots run on the AVX f32 kernels or as the scalar `row_dot` f32 fold. On the
// real geometry (17 rows x 10240 elements x 50 calls/token) the scalar loop
// measured ~3.2 ms/token -- more than the entire routed-expert matmul work of
// a small layer -- against a few hundred microseconds here.
inline void raon_mhc_pre(
    const float* streams,            // [4 * hidden]
    const RaonMhcWeights& w,
    const Geometry& g,
    float* branch_in,                // [hidden]
    float* post_out,                 // [4]
    float* comb_out                  // [4 * 4]
) {
    const std::size_t hc = g.mhc_streams;
    const std::size_t total_elements = hc * g.hidden;

#ifndef FLYWEIGHT_BAILING_NO_SIMD
    // The AVX f32 dot has no remainder handling: a width that is not a whole
    // number of granules reads past the input (the comment beside
    // qwen_f32_dot_multi_avx512 says as much). Everything else -- f16/bf16
    // fn storage, widths that violate the granule, machines without AVX --
    // keeps the scalar paths unchanged.
    const bool avx_usable = total_elements % 32 == 0 &&
                            (flyweight_cpu_features() & 3u) != 0;
#else
    const bool avx_usable = false;
#endif

    // 1. RMS norm factor of flattened streams
    double total_sq = 0.0;
    if (avx_usable) {
#ifndef FLYWEIGHT_BAILING_NO_SIMD
        float square = 0.0f;
        const float* self[1] = {streams};
        if ((flyweight_cpu_features() & 2u) != 0)
            qwen_f32_dot_multi_avx512(streams, self, 1,
                                      static_cast<int>(total_elements), &square);
        else
            qwen_f32_dot_multi_avx2(streams, self, 1,
                                    static_cast<int>(total_elements), &square);
        // Per-lane float partials instead of the scalar double fold: a ~1e-7
        // relative difference into a sqrt, invisible downstream.
        total_sq = square;
#endif
    } else {
        for (std::size_t i = 0; i < total_elements; ++i) {
            total_sq += static_cast<double>(streams[i]) * streams[i];
        }
    }
    const float inv_rms = 1.0f / std::sqrt(
        static_cast<float>(total_sq / static_cast<double>(total_elements)) + g.mhc_eps
    );

    // 2. Linear projection fn: [17, 4 * hidden]
    // mixes[r] = (fn_row[r] . streams) * inv_rms
    const std::size_t mix_dim = 2 * hc + (hc - 1) * (hc - 1); // 17 for hc=4
    // 32 covers hc up to 5; anything larger allocates, which no known
    // checkpoint does.
    float mixes_storage[32];
    std::vector<float> mixes_heap;
    float* mixes = mix_dim <= 32 ? mixes_storage : [&] {
        mixes_heap.resize(mix_dim);
        return mixes_heap.data();
    }();

    const auto stride = row_bytes(w.fn.type, total_elements);
    const auto* base_bytes = reinterpret_cast<const std::uint8_t*>(w.fn.data);
#ifndef FLYWEIGHT_BAILING_NO_SIMD
    if (avx_usable && w.fn.type == 0) {
        const bool avx512 = (flyweight_cpu_features() & 2u) != 0;
        for (std::size_t r = 0; r < mix_dim; ++r) {
            const auto* row_ptr =
                reinterpret_cast<const float*>(base_bytes + r * stride);
            if (avx512)
                qwen_f32_dot_multi_avx512(row_ptr, &streams, 1,
                                          static_cast<int>(total_elements), mixes + r);
            else
                qwen_f32_dot_multi_avx2(row_ptr, &streams, 1,
                                        static_cast<int>(total_elements), mixes + r);
            mixes[r] *= inv_rms;
        }
    } else
#endif
    {
        // Non-f32 fn storage rides the same RowKernel dispatch every matvec
        // uses: the check really produced F16 fn tensors in the wild, and the
        // scalar row_dot decode of 17 x 4*hidden elements x 50 calls/token
        // measured 20 ms of a decode. Admission is row_kernel's allowlist.
        const auto kernel = row_kernel(w.fn.type, total_elements);
        for (std::size_t r = 0; r < mix_dim; ++r) {
            mixes[r] = kernel(base_bytes, streams, total_elements, r) * inv_rms;
        }
    }

    float pre[4];
    raon_mhc_coefficients(mixes, w, g, pre, post_out, comb_out);

    // 4. Branch in aggregation: sum_i pre[i] * stream[i]
    for (std::size_t j = 0; j < g.hidden; ++j) {
        branch_in[j] = pre[0] * streams[0 * g.hidden + j] +
                       pre[1] * streams[1 * g.hidden + j] +
                       pre[2] * streams[2 * g.hidden + j] +
                       pre[3] * streams[3 * g.hidden + j];
    }
}

// Raon-mHC post-mixing:
// new_stream[dst] = (sum_{src} comb[dst, src] * stream[src]) + post[dst] * branch_out
//
// The naive spelling reloads all four input streams with a `hidden`-wide
// stride per output element. Accumulating stream-by-stream touches memory
// contiguously instead; the only numerics difference is the order adds are
// paired in, well inside 1e-6.
inline void raon_mhc_post(
    const float* streams_in,         // [4 * hidden]
    const float* branch_out,         // [hidden]
    const float* post,               // [4]
    const float* comb,               // [4 * 4]
    const Geometry& g,
    float* streams_out               // [4 * hidden]
) {
    for (std::size_t dst = 0; dst < g.mhc_streams; ++dst) {
        float* out = streams_out + dst * g.hidden;
        const float* c = comb + dst * 4;
        const float p = post[dst];
        for (std::size_t j = 0; j < g.hidden; ++j) {
            out[j] = p * branch_out[j];
        }
        for (std::size_t src = 0; src < g.mhc_streams; ++src) {
            const float scale = c[src];
            const float* in = streams_in + src * g.hidden;
            for (std::size_t j = 0; j < g.hidden; ++j) {
                out[j] += scale * in[j];
            }
        }
    }
}

// One head of one token through the RDA recurrence: q/k L2 norm, the decay
// gate, the write/erase ratio, the delta-rule update of this head's
// [head_dim x head_dim] state, the readout and the output norm + gate.
// Shared by the per-token step and the batched prefill so both run the same
// arithmetic; `f_b` is the head's slice of the f_b projection (dt_bias is
// added here), `out` receives head_dim values.
inline void rda_head_token(
    const float* q_conv, const float* k_conv, const float* v_conv,
    const float* f_b, float beta_raw, float rda_delta, const float* gate_row,
    const RdaWeights& w, const Geometry& g, std::size_t h,
    float* matrix, float* out_head
) {
    const std::size_t base = h * g.head_dim;

    // 1. L2 normalization of q and k
    float q_sq = 0.0f, k_sq = 0.0f;
    for (std::size_t i = 0; i < g.head_dim; ++i) {
        q_sq += q_conv[i] * q_conv[i];
        k_sq += k_conv[i] * k_conv[i];
    }
    const float q_scale = (1.0f / std::sqrt(static_cast<float>(g.head_dim))) /
                          std::sqrt(q_sq + 1e-6f);
    const float k_scale = 1.0f / std::sqrt(k_sq + 1e-6f);

    float q_norm[128], k_norm[128];
    for (std::size_t i = 0; i < g.head_dim; ++i) {
        q_norm[i] = q_conv[i] * q_scale;
        k_norm[i] = k_conv[i] * k_scale;
    }

    // 2. Channel-wise decay: scaled = exp(A_log) * (f_b + dt_bias)
    // ssm_a in the GGUF already holds -exp(A_log) (the converter's
    // convention, the same as Kimi's KDA), so exp(A_log) is its negation.
    const float a_exp = -w.a[h];
    float decay[128];
    for (std::size_t i = 0; i < g.head_dim; ++i) {
        const float decay_x = f_b[i] + (w.dt_bias ? w.dt_bias[base + i] : 0.0f);
        const float scaled = a_exp * decay_x;
        const float log_alpha = g.kda_gate_lower_bound * (1.0f / (1.0f + std::exp(-scaled)));
        decay[i] = std::exp(log_alpha);
    }

    // 3. Beta & write/erase ratio
    const float b = beta_raw;
    const float d = rda_delta;
    const float beta = 1.0f / (1.0f + std::exp(-b));
    const float sp1 = -b > 20.0f ? -b : std::log1p(std::exp(-b));
    const float sp2 = -(b + d) > 20.0f ? -(b + d) : std::log1p(std::exp(-(b + d)));
    const float ratio = std::exp(sp1 - sp2);

    // 4. Delta rule recurrence, state[v, k].
    //
    // Decay and read each row in one contiguous pass: the naive spelling
    // scaled whole k-columns at a stride of head_dim floats (512 bytes),
    // four passes of cache-hostile gathers on a 64 KB state. Elements are
    // scaled in place, then the prediction reads the row it just scaled.
    for (std::size_t v = 0; v < g.head_dim; ++v) {
        float* row = matrix + v * g.head_dim;
        float pred = 0.0f;
        for (std::size_t k = 0; k < g.head_dim; ++k) {
            row[k] *= decay[k];
            pred += row[k] * k_norm[k];
        }
        const float err = (v_conv[v] * ratio - pred) * beta;
        for (std::size_t k = 0; k < g.head_dim; ++k) {
            row[k] += err * k_norm[k];
        }
    }

    // out = state @ q, plus the output RMSNorm and elementwise sigmoid gate.
    for (std::size_t v = 0; v < g.head_dim; ++v) {
        const float* row = matrix + v * g.head_dim;
        float sum = 0.0f;
        for (std::size_t k = 0; k < g.head_dim; ++k) sum += row[k] * q_norm[k];
        out_head[v] = sum;
    }
    rms_norm(out_head, w.o_norm, g.head_dim, g.epsilon, out_head);
    for (std::size_t i = 0; i < g.head_dim; ++i)
        out_head[i] *= 1.0f / (1.0f + std::exp(-gate_row[i]));
}

// Step for an RDA layer (recurrent delta attention)
//
// Where the serial version's time went, worth recording because it is easy to
// reintroduce: the per-head loop was not parallelized, so 40 heads x ~65K
// state MACs ran on one core -- ~10 ms/token on the reference geometry, more
// than every routed expert matmul of a money-mixed layer combined. Heads are
// fully independent (each owns its own 128x128 state sliver), so the loop is
// a plain OpenMP split, the same one kda_recurrence already uses for the
// near-identical operator in bailing.
inline void rda_step(
    const float* x_norm,
    const RdaWeights& w,
    const Geometry& g,
    LayerCache& cache,
    float* branch_out,
    const GpuLayerDispatch* gpu_layer = nullptr,
    const GpuScratchDispatch* gpu_scratch = nullptr
) {
    const std::size_t channels = g.heads * g.head_dim; // 5120

    std::vector<float> attended(channels);
    std::optional<ProfileScope> phase;
    phase.emplace(profile().rda_projection);

    // The four channel-wide projections (q, k, v, gate) all read the same
    // branch norm and are independent: launching them back-to-back on the
    // default stream into one contiguous pack and bringing the pack back in
    // a single download costs one upload + one download + one sync per layer
    // instead of four of each. The small dt/gate projections (f_a, f_b,
    // beta, rda -- ~1.3 MB/layer) stay on the CPU, where a 5120-wide AVX
    // matvec beats a PCIe round trip.
    const bool gpu_project = gpu_layer && gpu_layer->attn_gpu && gpu_scratch &&
                             gpu_scratch->matvec_fn && gpu_scratch->proj_pack;

    std::vector<float> gate(channels), f_a(g.ssm_dt_rank), f_b(channels);
    std::vector<float> beta_raw(g.heads), rda_delta(g.heads);
    std::vector<float> q_proj(channels), k_proj(channels), v_proj(channels);

    if (gpu_project) {
        flyweight_gpu_upload_sync(gpu_scratch->input, x_norm, g.hidden * sizeof(float));
        if (gpu_layer->attn_packed) {
            // One fused matvec over the concatenated q/k/v/gate rows --
            // the pack layout below matches, so the unpack stays valid.
            gpu_scratch->matvec_fn(
                gpu_layer->attn_type[0], gpu_layer->attn_weight[0],
                gpu_scratch->input, gpu_scratch->proj_pack,
                static_cast<int>(g.hidden), static_cast<int>(4 * channels), 0, 1.0f);
        } else {
            const auto run = [&](std::size_t slot, std::size_t out_elements,
                                 std::size_t pack_elements_before) {
                gpu_scratch->matvec_fn(
                    gpu_layer->attn_type[slot], gpu_layer->attn_weight[slot],
                    gpu_scratch->input,
                    gpu_scratch->proj_pack + pack_elements_before * sizeof(float),
                    static_cast<int>(g.hidden), static_cast<int>(out_elements), 0, 1.0f);
            };
            run(0, channels, 0);
            run(1, channels, channels);
            run(2, channels, 2 * channels);
            run(3, channels, 3 * channels);
        }
        std::vector<float> pack(4 * channels);
        flyweight_gpu_download(pack.data(), gpu_scratch->proj_pack,
                               4 * channels * sizeof(float), 0);
        flyweight_gpu_stream_sync(0);
        std::memcpy(q_proj.data(), pack.data(), channels * sizeof(float));
        std::memcpy(k_proj.data(), pack.data() + channels, channels * sizeof(float));
        std::memcpy(v_proj.data(), pack.data() + 2 * channels, channels * sizeof(float));
        std::memcpy(gate.data(), pack.data() + 3 * channels, channels * sizeof(float));
    } else {
        matvec(w.q, x_norm, g.hidden, channels, q_proj.data());
        matvec(w.k, x_norm, g.hidden, channels, k_proj.data());
        matvec(w.v, x_norm, g.hidden, channels, v_proj.data());
        // Output-gate projection, hoisted out of the head loop: it feeds the
        // per-head norm+gate below. On the GPU path it is part of the pack
        // instead, so the hoisting here only covers the CPU fallback.
        matvec(w.gate, x_norm, g.hidden, channels, gate.data());
    }

    std::vector<float> q_conv(channels), k_conv(channels), v_conv(channels);
    short_conv_step(q_proj.data(), w.q_conv, channels, g.conv_width,
                    cache.q_window.data(), q_conv.data());
    short_conv_step(k_proj.data(), w.k_conv, channels, g.conv_width,
                    cache.k_window.data(), k_conv.data());
    short_conv_step(v_proj.data(), w.v_conv, channels, g.conv_width,
                    cache.v_window.data(), v_conv.data());

    matvec(w.f_a, x_norm, g.hidden, g.ssm_dt_rank, f_a.data());
    matvec(w.f_b, f_a.data(), g.ssm_dt_rank, channels, f_b.data());
    matvec(w.beta, x_norm, g.hidden, g.heads, beta_raw.data());
    matvec(w.rda, x_norm, g.hidden, g.heads, rda_delta.data());

    phase.reset();
    phase.emplace(profile().rda_recurrence);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads()) \
        if (g.heads >= 12)
#endif
    for (std::int64_t head_index = 0; head_index < static_cast<std::int64_t>(g.heads); ++head_index) {
        const auto h = static_cast<std::size_t>(head_index);
        const std::size_t base = h * g.head_dim;
        rda_head_token(q_conv.data() + base, k_conv.data() + base, v_conv.data() + base,
                       f_b.data() + base, beta_raw[h], rda_delta[h],
                       gate.data() + base, w, g, h,
                       cache.rda_state.data() + h * g.head_dim * g.head_dim,
                       attended.data() + base);
    }

    // 5. Output projection (GPU slot 4 consumes the attention output)
    if (gpu_project) {
        flyweight_gpu_upload_sync(gpu_scratch->attn_in, attended.data(),
                                  channels * sizeof(float));
        gpu_scratch->matvec_fn(
            gpu_layer->attn_type[4], gpu_layer->attn_weight[4],
            gpu_scratch->attn_in, gpu_scratch->proj_hidden,
            static_cast<int>(channels), static_cast<int>(g.hidden), 0, 1.0f);
        flyweight_gpu_download(branch_out, gpu_scratch->proj_hidden,
                               g.hidden * sizeof(float), 0);
        flyweight_gpu_stream_sync(0);
    } else {
        matvec(w.output, attended.data(), channels, g.hidden, branch_out);
    }
}

// bailing::mla_attention_absorbed over a 16-bit cache. The f32 version walks
// the cache once per head; here each cached row would then be widened forty
// times, so the loops turn inside out: per block of positions, widen each row
// once and score it for every head; softmax per head; then per block of
// positions again, widen once and fold the row into every head's
// accumulator (per-thread partials, summed after). Same math.
inline void mla_attention_absorbed_half(
    const float* queries_nope, const float* queries_rope, const float* kv_b,
    const LayerCache& cache, std::size_t positions, const Geometry& g, float* output
) {
    const std::size_t heads = g.heads, L = g.kv_lora, R = g.qk_rope;
    const std::size_t stride = g.qk_nope + g.v_head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(g.qk_nope + g.qk_rope));
    const int threads = matvec_threads();
    std::vector<float> projected(heads * L, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (heads >= 8)
#endif
    for (std::int64_t hi = 0; hi < static_cast<std::int64_t>(heads); ++hi) {
        const auto h = static_cast<std::size_t>(hi);
        float* target = projected.data() + h * L;
        const float* weights = kv_b + h * stride * L;
        for (std::size_t row = 0; row < g.qk_nope; ++row) {
            const float q = queries_nope[h * g.qk_nope + row];
            const float* source = weights + row * L;
            for (std::size_t i = 0; i < L; ++i) target[i] += q * source[i];
        }
    }

    // Position-split threads, at least 32 positions each: a thread's
    // [heads x L] partial costs its zeroing and its share of the fold.
    const int split = static_cast<int>(std::max<std::size_t>(
        1, std::min<std::size_t>(static_cast<std::size_t>(threads), positions / 32)));
    std::vector<float> scores(heads * positions);
#ifdef _OPENMP
#pragma omp parallel num_threads(split) if (split > 1)
#endif
    {
        std::vector<float> latent(L), rope(R);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t pi = 0; pi < static_cast<std::int64_t>(positions); ++pi) {
            const auto p = static_cast<std::size_t>(pi);
            const std::uint16_t* l = cache.latents16.data() + p * L;
            const std::uint16_t* r = cache.rope16.data() + p * R;
            for (std::size_t i = 0; i < L; ++i) latent[i] = cache_decode(l[i], g.cache_type);
            for (std::size_t i = 0; i < R; ++i) rope[i] = cache_decode(r[i], g.cache_type);
            for (std::size_t h = 0; h < heads; ++h) {
                const float* q = projected.data() + h * L;
                float total = 0.0f;
                for (std::size_t i = 0; i < L; ++i) total += q[i] * latent[i];
                const float* qr = queries_rope + h * R;
                for (std::size_t i = 0; i < R; ++i) total += qr[i] * rope[i];
                scores[h * positions + p] = total * scale;
            }
        }
    }
    for (std::size_t h = 0; h < heads; ++h)
        flyweight::v2::bailing::softmax_in_place(scores.data() + h * positions, positions);

    // Each thread folds its contiguous run of positions into its own partial;
    // the partials are then summed in parallel over the [heads x L] index.
    std::vector<float> partials(static_cast<std::size_t>(split) * heads * L, 0.0f);
#ifdef _OPENMP
#pragma omp parallel num_threads(split) if (split > 1)
#endif
    {
#ifdef _OPENMP
        const int part = omp_get_thread_num(), parts = omp_get_num_threads();
#else
        const int part = 0, parts = 1;
#endif
        std::vector<float> latent(L);
        float* partial = partials.data() + static_cast<std::size_t>(part) * heads * L;
        const std::size_t begin = positions * part / parts, end = positions * (part + 1) / parts;
        for (std::size_t p = begin; p < end; ++p) {
            const std::uint16_t* l = cache.latents16.data() + p * L;
            for (std::size_t i = 0; i < L; ++i) latent[i] = cache_decode(l[i], g.cache_type);
            for (std::size_t h = 0; h < heads; ++h) {
                const float w = scores[h * positions + p];
                float* acc = partial + h * L;
                for (std::size_t i = 0; i < L; ++i) acc[i] += w * latent[i];
            }
        }
    }
    std::vector<float> accumulated(heads * L, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (split > 1)
#endif
    for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(heads * L); ++ii) {
        float total = 0.0f;
        for (int part = 0; part < split; ++part)
            total += partials[static_cast<std::size_t>(part) * heads * L + static_cast<std::size_t>(ii)];
        accumulated[static_cast<std::size_t>(ii)] = total;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (heads >= 8)
#endif
    for (std::int64_t hi = 0; hi < static_cast<std::int64_t>(heads); ++hi) {
        const auto h = static_cast<std::size_t>(hi);
        const float* weights = kv_b + h * stride * L;
        const float* acc = accumulated.data() + h * L;
        for (std::size_t row = 0; row < g.v_head_dim; ++row) {
            const float* source = weights + (g.qk_nope + row) * L;
            float total = 0.0f;
            for (std::size_t i = 0; i < L; ++i) total += source[i] * acc[i];
            output[h * g.v_head_dim + row] = total;
        }
    }
}

// Step for an MLA layer (NoPE gated MLA)
inline void mla_step(
    const float* x_norm,
    const MlaWeights& w,
    const Geometry& g,
    std::size_t position,
    LayerCache& cache,
    float* branch_out,
    const GpuLayerDispatch* gpu_layer = nullptr,
    const GpuScratchDispatch* gpu_scratch = nullptr
) {
    ProfileScope scoped(profile().mla);
    // 1. KV low-rank projection
    std::vector<float> kv_full(g.kv_lora + g.qk_rope);

    // Attention-projection offload, same contract as rda_step: the q_a/gate/
    // kv_a launches share one uploaded x_norm and come back as one packed
    // download; the latent norm, cache writes, query unpacking and the
    // absorbed attention scan stay on the CPU. Pack layout:
    // [q_a (576) | gate (heads*v_head) | kv (kv_lora + rope)].
    const bool gpu_project = gpu_layer && gpu_layer->attn_gpu && gpu_scratch &&
                             gpu_scratch->matvec_fn && gpu_scratch->proj_pack;
    const auto proj_pack_base = gpu_scratch ? gpu_scratch->proj_pack : 0;

    std::vector<float> gate(g.heads * g.v_head_dim), q_low(g.q_lora);
    if (gpu_project) {
        flyweight_gpu_upload_sync(gpu_scratch->input, x_norm, g.hidden * sizeof(float));
        if (gpu_layer->attn_packed) {
            // Fused q_a + gate + kv_a rows, uploaded in this order; the
            // single matvec fills the same pack layout as the unpack.
            gpu_scratch->matvec_fn(
                gpu_layer->attn_type[0], gpu_layer->attn_weight[0],
                gpu_scratch->input, proj_pack_base,
                static_cast<int>(g.hidden),
                static_cast<int>(g.q_lora + g.heads * g.v_head_dim +
                                 g.kv_lora + g.qk_rope), 0, 1.0f);
        } else {
            const auto run = [&](std::size_t slot, std::size_t out_elements,
                                 std::size_t pack_elements_before) {
                gpu_scratch->matvec_fn(
                    gpu_layer->attn_type[slot], gpu_layer->attn_weight[slot],
                    gpu_scratch->input,
                    proj_pack_base + pack_elements_before * sizeof(float),
                    static_cast<int>(g.hidden), static_cast<int>(out_elements), 0, 1.0f);
            };
            run(0, g.q_lora, 0);                            // q_a
            run(3, g.heads * g.v_head_dim, g.q_lora);       // gate
            run(2, g.kv_lora + g.qk_rope,
                g.q_lora + g.heads * g.v_head_dim);         // kv_a
        }
        std::vector<float> pack(g.q_lora + g.heads * g.v_head_dim +
                                g.kv_lora + g.qk_rope);
        flyweight_gpu_download(pack.data(), proj_pack_base,
                               pack.size() * sizeof(float), 0);
        flyweight_gpu_stream_sync(0);
        std::memcpy(q_low.data(), pack.data(), g.q_lora * sizeof(float));
        std::memcpy(gate.data(), pack.data() + g.q_lora,
                    g.heads * g.v_head_dim * sizeof(float));
        std::memcpy(kv_full.data(),
                    pack.data() + g.q_lora + g.heads * g.v_head_dim,
                    (g.kv_lora + g.qk_rope) * sizeof(float));
    } else {
        matvec(w.kv_a_mqa, x_norm, g.hidden, g.kv_lora + g.qk_rope, kv_full.data());
        matvec(w.q_a, x_norm, g.hidden, g.q_lora, q_low.data());
        matvec(w.gate, x_norm, g.hidden, g.heads * g.v_head_dim, gate.data());
    }

    // Normalize the latent and store it with the unrotated rope key.
    rms_norm(kv_full.data(), w.kv_a_norm, g.kv_lora, g.latent_epsilon, kv_full.data());
    cache_store(cache, g, position, kv_full.data(), kv_full.data() + g.kv_lora);
    cache.positions = position + 1;

    // 2. Query low-rank projection
    std::vector<float> q_norm(g.q_lora);
    rms_norm(q_low.data(), w.q_a_norm, g.q_lora, g.latent_epsilon, q_norm.data());

    const std::size_t qk = g.qk_nope + g.qk_rope; // 192
    std::vector<float> query(g.heads * qk);
    if (gpu_project) {
        flyweight_gpu_upload_sync(gpu_scratch->input, q_norm.data(),
                                  g.q_lora * sizeof(float));
        gpu_scratch->matvec_fn(
            gpu_layer->attn_type[1], gpu_layer->attn_weight[1],
            gpu_scratch->input, gpu_scratch->proj_query,
            static_cast<int>(g.q_lora), static_cast<int>(g.heads * qk), 0, 1.0f);
        flyweight_gpu_download(query.data(), gpu_scratch->proj_query,
                               g.heads * qk * sizeof(float), 0);
        flyweight_gpu_stream_sync(0);
    } else {
        matvec(w.q_b, q_norm.data(), g.q_lora, g.heads * qk, query.data());
    }

    std::vector<float> query_nope(g.heads * g.qk_nope), query_rope(g.heads * g.qk_rope);
    for (std::size_t h = 0; h < g.heads; ++h) {
        for (std::size_t i = 0; i < g.qk_nope; ++i) {
            query_nope[h * g.qk_nope + i] = query[h * qk + i];
        }
        for (std::size_t i = 0; i < g.qk_rope; ++i) {
            query_rope[h * g.qk_rope + i] = query[h * qk + g.qk_nope + i];
        }
    }

    // 3. Absorbed attention over cached latents and unrotated rope keys
    std::vector<float> attended(g.heads * g.v_head_dim);
    if (g.cache_type == 0)
        mla_attention_absorbed(
            query_nope.data(), query_rope.data(), w.kv_b,
            cache.latents.data(), cache.rope_keys.data(), cache.positions,
            g.heads, g.qk_nope, g.qk_rope, g.v_head_dim, g.kv_lora,
            attended.data()
        );
    else
        mla_attention_absorbed_half(query_nope.data(), query_rope.data(), w.kv_b, cache,
                                    cache.positions, g, attended.data());

    // 4. Elementwise output gate: sigmoid(gate_proj @ hidden)
    for (std::size_t i = 0; i < g.heads * g.v_head_dim; ++i) {
        attended[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
    }

    // 5. Output projection (GPU slot 4 consumes the attention output)
    if (gpu_project) {
        flyweight_gpu_upload_sync(gpu_scratch->attn_in, attended.data(),
                                  g.heads * g.v_head_dim * sizeof(float));
        gpu_scratch->matvec_fn(
            gpu_layer->attn_type[4], gpu_layer->attn_weight[4],
            gpu_scratch->attn_in, gpu_scratch->proj_hidden,
            static_cast<int>(g.heads * g.v_head_dim),
            static_cast<int>(g.hidden), 0, 1.0f);
        flyweight_gpu_download(branch_out, gpu_scratch->proj_hidden,
                               g.hidden * sizeof(float), 0);
        flyweight_gpu_stream_sync(0);
    } else {
        matvec(w.output, attended.data(), g.heads * g.v_head_dim, g.hidden, branch_out);
    }
}

// FLYWEIGHT_RAON_ROUTE_TRACE=<path>: every decode token's chosen experts are
// appended to <path> as text lines "layer e0 e1 ...", for replaying cache
// policies offline. route_layer is set by the decode step (-1 elsewhere, so
// prefill routes are not written).
inline thread_local int route_layer = -1;
inline std::FILE* route_trace_file() {
    static std::FILE* file = [] {
        const char* path = std::getenv("FLYWEIGHT_RAON_ROUTE_TRACE");
        return path && path[0] ? std::fopen(path, "a") : nullptr;
    }();
    return file;
}

// Sigmoid routing for one token. The selection bias (exp_probs_b) only
// chooses the experts; their unbiased scores are the weights, normalized when
// the checkpoint says so and then scaled. Ties go to the lower expert index.
// `order` must hold g.experts entries; the first experts_used are the choice.
inline void route_token(const float* logits, const float* bias, const Geometry& g,
                        std::int32_t* order, float* weights) {
    std::vector<float> scores(g.experts), selection(g.experts);
    for (std::size_t i = 0; i < g.experts; ++i) {
        scores[i] = 1.0f / (1.0f + std::exp(-logits[i]));
        selection[i] = scores[i] + (bias ? bias[i] : 0.0f);
        order[i] = static_cast<std::int32_t>(i);
    }
    std::partial_sort(
        order, order + g.experts_used, order + g.experts,
        [&](std::int32_t a, std::int32_t b) {
            const float sa = selection[a], sb = selection[b];
            return sa != sb ? sa > sb : a < b;
        });
    float total_weight = 0.0f;
    for (std::size_t k = 0; k < g.experts_used; ++k) {
        weights[k] = scores[order[k]];
        total_weight += weights[k];
    }
    if (g.expert_weights_norm && g.experts_used > 1) {
        const float inv_total = 1.0f / (total_weight + 1e-20f);
        for (std::size_t k = 0; k < g.experts_used; ++k) weights[k] *= inv_total;
    }
    for (std::size_t k = 0; k < g.experts_used; ++k) weights[k] *= g.expert_weights_scale;
    if (route_layer >= 0) {
        if (std::FILE* file = route_trace_file()) {
            std::fprintf(file, "%d", route_layer);
            for (std::size_t k = 0; k < g.experts_used; ++k) std::fprintf(file, " %d", order[k]);
            std::fputc('\n', file);
        }
    }
}

// The router of one MoE layer for one token on the CPU: the chosen experts
// (the first experts_used of `order`, which holds g.experts) and their weights.
inline void moe_route_cpu(const float* x_norm, const LayerWeights& layer, const Geometry& g,
                          std::int32_t* order, float* weights) {
    ProfileScope route(profile().moe_route);
    std::vector<float> logits(g.experts);
    matvec(layer.ffn.router, x_norm, g.hidden, g.experts, logits.data());
    route_token(logits.data(), layer.ffn.router_bias, g, order, weights);
}

// `count` routed experts of one token on the CPU, weighted and summed into
// branch_out (overwritten; zero when count is 0). The device decode's expert
// cache passes only its misses.
inline void moe_experts_cpu(const float* x_norm, const LayerWeights& layer, const Geometry& g,
                            const std::int32_t* experts, const float* weights, std::size_t count,
                            float* branch_out) {
    ProfileScope moe(profile().moe);
    ProfileScope scoped(profile().moe_experts);
    if (count == 0) {
        std::fill(branch_out, branch_out + g.hidden, 0.0f);
        return;
    }
    const auto expert_size = g.expert_size;
    const auto narrow_stride = expert_size * row_bytes(layer.ffn.gate_experts.type, g.hidden);
    const auto wide_stride = g.hidden * row_bytes(layer.ffn.down_experts.type, expert_size);
    const auto* gate_base = reinterpret_cast<const std::uint8_t*>(layer.ffn.gate_experts.data);
    const auto* up_base = reinterpret_cast<const std::uint8_t*>(layer.ffn.up_experts.data);
    const auto* down_base = reinterpret_cast<const std::uint8_t*>(layer.ffn.down_experts.data);

    std::vector<float> activated(count * expert_size);
    {
        std::vector<float> gate_up(2 * count * expert_size);
        const auto narrow_kernel = row_kernel(layer.ffn.gate_experts.type, g.hidden);
        const auto rows = static_cast<std::int64_t>(2 * count * expert_size);
        const auto span = static_cast<std::int64_t>(count * expert_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads())
#endif
        for (std::int64_t r = 0; r < rows; ++r) {
            const bool is_up = r >= span;
            const auto within = static_cast<std::size_t>(is_up ? r - span : r);
            const auto slot = within / expert_size, row = within % expert_size;
            const auto expert = static_cast<std::size_t>(experts[slot]);
            const auto* base = (is_up ? up_base : gate_base) + expert * narrow_stride;
            gate_up[static_cast<std::size_t>(r)] = narrow_kernel(base, x_norm, g.hidden, row);
        }
        situ_glu(gate_up.data(), gate_up.data() + span, span, g.situ_beta, g.situ_linear_beta, activated.data());
    }
    const auto down_kernel = row_kernel(layer.ffn.down_experts.type, expert_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(matvec_threads())
#endif
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(g.hidden); ++r) {
        const auto row = static_cast<std::size_t>(r);
        float total = 0.0f;
        for (std::size_t slot = 0; slot < count; ++slot) {
            const auto expert = static_cast<std::size_t>(experts[slot]);
            total += weights[slot] * down_kernel(down_base + expert * wide_stride,
                                                 activated.data() + slot * expert_size, expert_size, row);
        }
        branch_out[row] = total;
    }
}

// The router and routed experts of one token on the CPU, into branch_out
// (the shared expert is not added).
inline void moe_routed_cpu(
    const float* x_norm, const LayerWeights& layer, const Geometry& g, float* branch_out
) {
    std::vector<std::int32_t> order(g.experts);
    std::vector<float> weights(g.experts_used);
    moe_route_cpu(x_norm, layer, g, order.data(), weights.data());
    moe_experts_cpu(x_norm, layer, g, order.data(), weights.data(), g.experts_used, branch_out);
}

// Step for FFN / MoE block
inline void ffn_step(
    const float* x_norm,
    const LayerWeights& layer,
    const Geometry& g,
    float* branch_out,
    const GpuLayerDispatch* gpu_layer = nullptr,
    const GpuScratchDispatch* gpu_scratch = nullptr
) {
    if (!layer.is_moe) {
        ProfileScope dense(profile().dense_ffn);
        // Dense SiTU-GLU (layer 0)
        std::vector<float> gate(g.dense_size), up(g.dense_size), act(g.dense_size);
        matvec(layer.ffn.gate, x_norm, g.hidden, g.dense_size, gate.data());
        matvec(layer.ffn.up, x_norm, g.hidden, g.dense_size, up.data());
        situ_glu(gate.data(), up.data(), g.dense_size, g.situ_beta, g.situ_linear_beta, act.data());
        matvec(layer.ffn.down, act.data(), g.dense_size, g.hidden, branch_out);
        return;
    }

    if (gpu_layer && gpu_layer->pinned && gpu_scratch && gpu_scratch->matvec_fn && gpu_scratch->launch_situ_glu) {
        // GPU MoE route
        ProfileScope moe(profile().moe);
        flyweight_gpu_upload_sync(gpu_scratch->input, x_norm, g.hidden * sizeof(float));

        std::vector<std::int32_t> order(g.experts);
        std::vector<float> weights(g.experts_used);
        {
            ProfileScope route(profile().moe_route);
            std::vector<float> logits(g.experts);
            gpu_scratch->matvec_fn(gpu_layer->router_type, gpu_layer->router, gpu_scratch->input, gpu_scratch->router_logits, static_cast<int>(g.hidden), static_cast<int>(g.experts), 0, 1.0f);
            flyweight_gpu_download(logits.data(), gpu_scratch->router_logits, g.experts * sizeof(float), 0);
            flyweight_gpu_stream_sync(0);
            route_token(logits.data(), gpu_layer->router_bias, g, order.data(), weights.data());
        }

        {
            ProfileScope experts(profile().moe_experts);
            flyweight_gpu_memset(gpu_scratch->moe_accum, 0, g.hidden * sizeof(float), 0);

            const auto narrow_stride = g.expert_size * row_bytes(gpu_layer->gate_type, g.hidden);
            const auto wide_stride = g.hidden * row_bytes(gpu_layer->down_type, g.expert_size);

            for (std::size_t slot = 0; slot < g.experts_used; ++slot) {
                const auto expert = static_cast<std::size_t>(order[slot]);
                const auto gate_addr = gpu_layer->gate_exps + expert * narrow_stride;
                const auto up_addr = gpu_layer->up_exps + expert * narrow_stride;
                const auto down_addr = gpu_layer->down_exps + expert * wide_stride;

                gpu_scratch->matvec_fn(gpu_layer->gate_type, gate_addr, gpu_scratch->input, gpu_scratch->expert_gate, static_cast<int>(g.hidden), static_cast<int>(g.expert_size), 0, 1.0f);
                gpu_scratch->matvec_fn(gpu_layer->up_type, up_addr, gpu_scratch->input, gpu_scratch->expert_up, static_cast<int>(g.hidden), static_cast<int>(g.expert_size), 0, 1.0f);
                if (gpu_scratch->launch_situ_glu(gpu_scratch->expert_gate, gpu_scratch->expert_up, gpu_scratch->expert_act, static_cast<int>(g.expert_size), g.situ_beta, g.situ_linear_beta) != 0) {
                    throw std::runtime_error("raon_situ_glu kernel launch failed for expert " + std::to_string(expert));
                }
                gpu_scratch->matvec_fn(gpu_layer->down_type, down_addr, gpu_scratch->expert_act, gpu_scratch->expert_down, static_cast<int>(g.expert_size), static_cast<int>(g.hidden), 0, 1.0f);

                flyweight_gpu_scaled_add(gpu_scratch->moe_accum, gpu_scratch->expert_down, weights[slot], static_cast<int>(g.hidden));
            }
        }

        if (g.shared_experts > 0 && gpu_layer->shared_gate != 0) {
            ProfileScope shared(profile().moe_shared);
            gpu_scratch->matvec_fn(gpu_layer->shared_gate_type, gpu_layer->shared_gate, gpu_scratch->input, gpu_scratch->expert_gate, static_cast<int>(g.hidden), static_cast<int>(g.shared_size), 0, 1.0f);
            gpu_scratch->matvec_fn(gpu_layer->shared_up_type, gpu_layer->shared_up, gpu_scratch->input, gpu_scratch->expert_up, static_cast<int>(g.hidden), static_cast<int>(g.shared_size), 0, 1.0f);
            if (gpu_scratch->launch_situ_glu(gpu_scratch->expert_gate, gpu_scratch->expert_up, gpu_scratch->expert_act, static_cast<int>(g.shared_size), g.situ_beta, g.situ_linear_beta) != 0) {
                throw std::runtime_error("raon_situ_glu kernel launch failed for shared expert");
            }
            gpu_scratch->matvec_fn(gpu_layer->shared_down_type, gpu_layer->shared_down, gpu_scratch->expert_act, gpu_scratch->expert_down, static_cast<int>(g.shared_size), static_cast<int>(g.hidden), 0, 1.0f);

            flyweight_gpu_scaled_add(gpu_scratch->moe_accum, gpu_scratch->expert_down, 1.0f, static_cast<int>(g.hidden));
        }

        flyweight_gpu_download(branch_out, gpu_scratch->moe_accum, g.hidden * sizeof(float), 0);
        flyweight_gpu_stream_sync(0);
        return;
    }

    // Sparse MoE (CPU fallback): routed experts, then the shared expert.
    moe_routed_cpu(x_norm, layer, g, branch_out);

    // 3. Shared expert
    if (g.shared_experts > 0 && layer.ffn.shared_gate) {
        ProfileScope shared(profile().moe_shared);
        std::vector<float> gate(g.shared_size), up(g.shared_size), act(g.shared_size), exp_out(g.hidden);
        matvec(layer.ffn.shared_gate, x_norm, g.hidden, g.shared_size, gate.data());
        matvec(layer.ffn.shared_up, x_norm, g.hidden, g.shared_size, up.data());
        situ_glu(gate.data(), up.data(), g.shared_size, g.situ_beta, g.situ_linear_beta, act.data());
        matvec(layer.ffn.shared_down, act.data(), g.shared_size, g.hidden, exp_out.data());

        for (std::size_t j = 0; j < g.hidden; ++j) {
            branch_out[j] += exp_out[j];
        }
    }
}

// Single token forward step across all 25 layers
inline void decoder_step(
    const float* token_embd,         // [hidden]
    const std::vector<LayerWeights>& layers,
    const Geometry& g,
    std::size_t position,
    std::vector<LayerCache>& caches,
    float* streams_inout,            // [4 * hidden]
    float* final_hidden_out,         // [hidden]
    const std::vector<GpuLayerDispatch>* gpu_layers = nullptr,
    const GpuScratchDispatch* gpu_scratch = nullptr
) {
    if (profiling()) profile().tokens++;

    // Each token starts with its embedding copied across all streams
    for (std::size_t dst = 0; dst < g.mhc_streams; ++dst) {
        std::memcpy(streams_inout + dst * g.hidden, token_embd, g.hidden * sizeof(float));
    }

    std::vector<float> branch_in(g.hidden), branch_norm(g.hidden), branch_out(g.hidden);
    std::vector<float> streams_next(g.mhc_streams * g.hidden);
    float post[4], comb[16];

    for (std::size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];
        auto& cache = caches[l];
        const GpuLayerDispatch* gpu_layer = (gpu_layers && l < gpu_layers->size()) ? &(*gpu_layers)[l] : nullptr;

        // --- Attention Block ---
        {
            ProfileScope mixing(profile().mhc);
            raon_mhc_pre(streams_inout, layer.mhc_attn, g, branch_in.data(), post, comb);
        }
        rms_norm(branch_in.data(), layer.attn_norm, g.hidden, g.epsilon, branch_norm.data());

        if (layer.is_rda) {
            rda_step(branch_norm.data(), layer.rda, g, cache, branch_out.data(),
                     gpu_layer, gpu_scratch);
        } else {
            mla_step(branch_norm.data(), layer.mla, g, position, cache,
                     branch_out.data(), gpu_layer, gpu_scratch);
        }

        {
            ProfileScope mixing(profile().mhc);
            raon_mhc_post(streams_inout, branch_out.data(), post, comb, g, streams_next.data());
        }
        std::memcpy(streams_inout, streams_next.data(), g.mhc_streams * g.hidden * sizeof(float));

        // --- FFN Block ---
        {
            ProfileScope mixing(profile().mhc);
            raon_mhc_pre(streams_inout, layer.mhc_ffn, g, branch_in.data(), post, comb);
        }
        rms_norm(branch_in.data(), layer.ffn_norm, g.hidden, g.epsilon, branch_norm.data());

        ffn_step(branch_norm.data(), layer, g, branch_out.data(), gpu_layer, gpu_scratch);

        {
            ProfileScope mixing(profile().mhc);
            raon_mhc_post(streams_inout, branch_out.data(), post, comb, g, streams_next.data());
        }
        std::memcpy(streams_inout, streams_next.data(), g.mhc_streams * g.hidden * sizeof(float));
    }

    // Final model output: 0.25 * sum_{i=0..3} stream[i]
    for (std::size_t j = 0; j < g.hidden; ++j) {
        final_hidden_out[j] = 0.25f * (
            streams_inout[0 * g.hidden + j] +
            streams_inout[1 * g.hidden + j] +
            streams_inout[2 * g.hidden + j] +
            streams_inout[3 * g.hidden + j]
        );
    }
}

} // namespace flyweight::v2::raon
