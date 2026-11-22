// Inkling 2 execution path. Included from v2_runtime.cpp after the model,
// tensor accessors, and GPU matvec dispatchers are defined.
//
// The design follows llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp
// and keeps every weight in its GGUF quantization. When a GPU backend is
// active, dense projections and the LM head stay resident in VRAM while MoE
// experts run through an LRU GPU expert cache that streams cold experts from
// the memory-mapped checkpoint via pinned host staging buffers.

namespace flyweight::v2::inkling2 {

using bailing::Matrix;
using bailing::matvec;
using bailing::row_bytes;
using bailing::row_decode;
using bailing::row_kernel;

static const char* const inkling2_cuda_source = R"FLYWEIGHT_CUDA(
extern "C" __global__
void inkling2_situ_glu(
    const float* gate_up, float* out, int ff, float beta, float linear_beta, float gu_scale
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ff) return;
    const float g = gate_up[i] * gu_scale;
    const float u = gate_up[ff + i] * gu_scale;
    const float sig = 1.0f / (1.0f + expf(-g));
    const float g_act = beta * tanhf(g / beta) * sig;
    const float u_act = linear_beta * tanhf(u / linear_beta);
    out[i] = g_act * u_act;
}

extern "C" __global__
void inkling2_situ_glu_rows(
    const float* gate_up, float* out, int rows, int ff, float beta, float linear_beta, float gu_scale
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= ff || r >= rows) return;
    const float* row = gate_up + (long long)r * (long long)(2 * ff);
    const float g = row[i] * gu_scale;
    const float u = row[ff + i] * gu_scale;
    const float sig = 1.0f / (1.0f + expf(-g));
    const float g_act = beta * tanhf(g / beta) * sig;
    const float u_act = linear_beta * tanhf(u / linear_beta);
    out[(long long)r * (long long)ff + i] = g_act * u_act;
}

extern "C" __global__
void inkling2_scaled_add(
    float* dst, const float* src, float scale, int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += scale * src[i];
}

extern "C" __global__
void inkling2_softcap(
    float* logits, int n, float cap
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    logits[i] = cap * tanhf(logits[i] / cap);
}
)FLYWEIGHT_CUDA";

static const char* const inkling2_cuda_decode_source = R"FLYWEIGHT_CUDA(
extern "C" __global__
void inkling2_rms_norm(
    const float* x, const float* w, float* out, int n, float eps
) {
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(smem[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * (w ? w[i] : 1.0f);
    }
}

extern "C" __global__
void inkling2_mix(
    const float* bank, int n_bank, const float* cur,
    const float* query, int n_cols, float* out, int hidden
) {
    const int c = blockIdx.x;
    if (c >= n_cols) return;
    const int n = n_bank + 1;
    const float* qc = query + (long long)c * hidden;
    __shared__ float smem_sq[256];
    __shared__ float smem_dot[256];
    __shared__ float inv_rms[64];
    __shared__ float weights[64];
    const float inv_sqrt_h = rsqrtf((float)hidden);

    for (int s = 0; s < n; ++s) {
        const float* v = (s < n_bank) ? (bank + (long long)s * hidden) : cur;
        float sq = 0.0f, dot = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float x = v[i];
            sq += x * x;
            dot += x * qc[i];
        }
        smem_sq[threadIdx.x] = sq;
        smem_dot[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                smem_sq[threadIdx.x] += smem_sq[threadIdx.x + stride];
                smem_dot[threadIdx.x] += smem_dot[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float ir = rsqrtf(smem_sq[0] / (float)hidden + 1e-6f);
            inv_rms[s] = ir;
            weights[s] = smem_dot[0] * ir * inv_sqrt_h;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float peak = weights[0];
        for (int s = 1; s < n; ++s) peak = fmaxf(peak, weights[s]);
        float total = 0.0f;
        for (int s = 0; s < n; ++s) {
            weights[s] = expf(weights[s] - peak);
            total += weights[s];
        }
        const float inv_t = total > 0.0f ? (1.0f / total) : 0.0f;
        for (int s = 0; s < n; ++s) weights[s] *= inv_t * inv_rms[s];
    }
    __syncthreads();
    float* oc = out + (long long)c * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n; ++s) {
            const float* v = (s < n_bank) ? (bank + (long long)s * hidden) : cur;
            acc += weights[s] * v[i];
        }
        oc[i] = acc;
    }
}

extern "C" __global__
void inkling2_control_pair(
    const float* cur, const float* msg, float* pair_out, int hidden, float eps
) {
    const float* src = blockIdx.x == 0 ? cur : msg;
    float* dst = pair_out + (long long)blockIdx.x * hidden;
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = src[i];
        sum += v * v;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(smem[0] / (float)hidden + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i] * scale;
}

extern "C" __global__
void inkling2_control_apply(
    const float* cur, const float* msg, const float* gain, const float* forget,
    float* out, int hidden
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    out[i] = (1.0f - tanhf(forget[i])) * cur[i] + tanhf(gain[i]) * msg[i];
}

extern "C" __global__
void inkling2_convolve(
    const float* input, const float* weight, const float* dyn_taps,
    float* history, float* out, int channels, int taps, int head_dim
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    const float x0 = input[i];
    const float* wi = weight + (long long)i * taps;
    float total = wi[taps - 1] * x0;
    for (int lag = 1; lag < taps; ++lag) {
        total += wi[taps - 1 - lag] * history[(long long)(lag - 1) * channels + i];
    }
    if (dyn_taps) {
        const int h = i / head_dim;
        const float* dt = dyn_taps + (long long)h * taps;
        total += dt[0] * x0;
        for (int lag = 1; lag < taps; ++lag) {
            total += dt[lag] * history[(long long)(lag - 1) * channels + i];
        }
    }
    for (int lag = taps - 1; lag > 1; --lag) {
        history[(long long)(lag - 1) * channels + i] = history[(long long)(lag - 2) * channels + i];
    }
    if (taps > 1) history[i] = x0;
    const float norm = sqrtf((dyn_taps ? 2.0f : 1.0f) * (float)taps + 1.0f);
    out[i] = (x0 + total) / norm;
}

extern "C" __global__
void inkling2_clip_rows(
    float* x, const float* scale, int rows, int dim, float threshold, float tau
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* row = x + (long long)r * dim;
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = row[i];
        if (scale) {
            v *= scale[(long long)r * dim + i];
            row[i] = v;
        }
        sum += v * v;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float norm = sqrtf(smem[0] + 1e-12f) + 1e-6f;
    const float factor = fminf(1.0f, threshold / norm) * tau;
    if (factor != 1.0f) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) row[i] *= factor;
    }
}

extern "C" __global__
void inkling2_v_norm_kv_append(
    const float* K, float* V, half* K_cache, half* V_cache, float* positions,
    int write_pos, int kv_channels, int kvh, int head_dim, float eps, float pos_val
) {
    const int h = blockIdx.x;
    if (h >= kvh) return;
    float* vh = V + (long long)h * head_dim;
    const float* kh = K + (long long)h * head_dim;
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = vh[i];
        sum += v * v;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(smem[0] / (float)head_dim + eps);
    half* kc = K_cache + (long long)write_pos * kv_channels + (long long)h * head_dim;
    half* vc = V_cache + (long long)write_pos * kv_channels + (long long)h * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float vn = vh[i] * scale;
        vh[i] = vn;
        kc[i] = __float2half(kh[i]);
        vc[i] = __float2half(vn);
    }
    if (h == 0 && threadIdx.x == 0) positions[write_pos] = pos_val;
}

extern "C" __global__
void inkling2_rel_attn(
    const float* Q, const half* K_cache, const half* V_cache, const float* positions,
    const float* R, const float* rel_bias, const float* sinks, const float* gate,
    float* out, int kv_len, int heads, int kvh, int head_dim, int rel_len,
    int extent, float pos_val
) {
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int kv_head = h / (heads / kvh);
    const int kv_channels = kvh * head_dim;
    const float* qh = Q + (long long)h * head_dim;
    const float* rh = R + (long long)h * rel_len;
    const float sink = sinks[h];
    const float inv_sqrt_hd = rsqrtf((float)head_dim);

    __shared__ float s_dot[128];
    __shared__ float s_rel[128];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_alpha;
    __shared__ float s_beta;

    float acc = 0.0f;
    if (threadIdx.x == 0) {
        s_max = 0.0f;  // zero-valued sink logit
        s_sum = 1.0f;  // exp(0 - 0) = 1
    }
    __syncthreads();

    for (int j = 0; j < kv_len; ++j) {
        const half* kj = K_cache + (long long)j * kv_channels + (long long)kv_head * head_dim;
        float partial = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
            partial += qh[i] * __half2float(kj[i]);
        s_dot[threadIdx.x] = partial;

        const float df = pos_val - positions[j];
        const int di = (df >= 0.0f && df < (float)extent) ? (int)df : -1;
        float rpart = 0.0f;
        if (di >= 0) {
            const float* tb = rel_bias + (long long)di * rel_len;
            for (int r = threadIdx.x; r < rel_len; r += blockDim.x)
                rpart += tb[r] * rh[r];
        }
        s_rel[threadIdx.x] = rpart;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
                s_rel[threadIdx.x] += s_rel[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = s_dot[0] * inv_sqrt_hd + (di >= 0 ? (s_rel[0] - sink) : -sink);
            const float new_max = fmaxf(s_max, score);
            const float alpha = expf(s_max - new_max);
            const float beta = expf(score - new_max);
            s_sum = s_sum * alpha + beta;
            s_max = new_max;
            s_alpha = alpha;
            s_beta = beta;
        }
        __syncthreads();

        const half* vj = V_cache + (long long)j * kv_channels + (long long)kv_head * head_dim;
        if (threadIdx.x < head_dim) {
            acc = acc * s_alpha + s_beta * __half2float(vj[threadIdx.x]);
        }
        __syncthreads();
    }

    const float inv_sum = 1.0f / s_sum;
    if (threadIdx.x < head_dim) {
        const float gval = gate[(long long)h * head_dim + threadIdx.x];
        const float sig = 1.0f / (1.0f + expf(-gval));
        out[(long long)h * head_dim + threadIdx.x] = (acc * inv_sum) * sig;
    }
}

extern "C" __global__
void inkling2_add(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}
)FLYWEIGHT_CUDA";

inline std::size_t align256(std::size_t n) {
    return (n + 255u) & ~std::size_t(255u);
}

inline bool gpu_type_supported(std::uint32_t type, std::size_t cols) {
    if (cols == 0) return false;
    switch (type) {
        case 0:  // F32
        case 1:  // F16
        case 30: // BF16
            return true;
        case 2:  // Q4_0
        case 8:  // Q8_0
        case 20: // IQ4_NL
        case 42: // Q2_0
            return (cols % 32) == 0;
        case 40: // NVFP4
            return (cols % 16) == 0;
        case 10: // Q2_K
        case 11: // Q3_K
        case 12: // Q4_K
        case 13: // Q5_K
        case 14: // Q6_K
        case 16: // IQ2_XXS
        case 17: // IQ2_XS
        case 18: // IQ3_XXS
        case 19: // IQ1_S
        case 21: // IQ3_S
        case 22: // IQ2_S
        case 23: // IQ4_XS
        case 29: // IQ1_M
            return (cols % 256) == 0;
        default:
            return false;
    }
}

struct Geometry {
    std::uint32_t hidden = 0, layers = 0, vocab = 0, vocab_unpadded = 0;
    std::uint32_t experts = 0, experts_used = 0, ff_exp = 0;
    std::uint32_t heads = 0, kv_heads = 0, head_dim = 0;
    std::uint32_t core_start = 0, core_end = 0, num_loops = 1, gate_rank = 0;
    std::uint32_t block_size = 1, conv_taps = 1, sliding_window = 0;
    std::uint32_t rel_len = 0, rel_extent = 0, log_scale_floor = 1;
    float qk_clip = 0.0f, rel_clip = 0.0f, log_scale_alpha = 0.0f;
    float softcap = 0.0f, expert_scale = 1.0f, eps = 1e-6f;
    float situ_beta = 4.0f, situ_linear_beta = 25.0f;
    std::uint32_t physical = 0, slots = 0;
    std::vector<std::uint32_t> heads_by_layer, kv_heads_by_layer;
    std::vector<std::uint8_t> swa;
};

inline Geometry geometry(const FlyweightV2Model& m) {
    Geometry g;
    const auto& c = m.config;
    g.hidden           = c.hidden_size;
    g.layers           = c.layer_count;
    g.vocab            = c.vocabulary_size;
    g.vocab_unpadded   = c.vocab_size_unpadded ? c.vocab_size_unpadded : c.vocabulary_size;
    g.experts          = c.expert_count;
    g.experts_used     = c.expert_used_count;
    g.ff_exp           = c.expert_intermediate_size;
    g.heads            = c.attention_heads;
    g.kv_heads         = c.attention_kv_heads;
    g.head_dim         = c.attention_head_dim ? c.attention_head_dim : c.key_length;
    g.core_start       = c.loop_core_start;
    g.core_end         = c.loop_core_end;
    g.num_loops        = c.num_loops ? c.num_loops : 1;
    g.gate_rank        = c.loop_gate_rank;
    g.block_size       = c.attn_res_block_size ? c.attn_res_block_size : 1;
    g.conv_taps        = c.conv_kernel ? c.conv_kernel : 1;
    g.sliding_window   = c.sliding_window;
    g.rel_len          = c.relative_length;
    g.rel_extent       = c.relative_extent;
    g.log_scale_floor  = c.log_scale_floor ? c.log_scale_floor : 1;
    g.qk_clip          = c.qk_clip;
    g.rel_clip         = c.relative_clip;
    g.log_scale_alpha  = c.log_scale_alpha;
    g.softcap          = c.final_logit_softcap;
    g.expert_scale     = c.expert_weights_scale;
    g.eps              = c.rms_norm_epsilon ? c.rms_norm_epsilon : 1e-6f;
    g.situ_beta        = c.situ_beta;
    g.situ_linear_beta = c.situ_linear_beta;
    g.heads_by_layer   = c.attention_heads_by_layer;
    g.kv_heads_by_layer= c.attention_kv_heads_by_layer;
    g.swa              = c.sliding_window_pattern;
    const std::uint32_t core = g.core_end - g.core_start;
    g.physical = g.core_end + (g.layers - (g.core_start + core * g.num_loops));
    g.slots    = g.physical + (g.num_loops - 1) * core;
    return g;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline void rms_norm_unit(const float* x, std::size_t n, float eps, float* out) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * scale;
}

inline void rms_norm_into(const float* x, const float* w, std::size_t n, float eps, float* out) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

// A row-wise L2 clip: scale the row down when its norm exceeds `threshold`.
inline void clip_rows(float* x, std::size_t rows, std::size_t dim,
                      const float* scale, float threshold) {
    for (std::size_t r = 0; r < rows; ++r) {
        float* row = x + r * dim;
        if (scale) for (std::size_t i = 0; i < dim; ++i) row[i] *= scale[r * dim + i];
        float sum = 1e-12f;
        for (std::size_t i = 0; i < dim; ++i) sum += row[i] * row[i];
        const float norm = std::sqrt(sum) + 1e-6f;
        const float factor = std::min(1.0f, threshold / norm);
        if (factor < 1.0f) for (std::size_t i = 0; i < dim; ++i) row[i] *= factor;
    }
}

inline void softmax_into(float* scores, std::size_t n) {
    if (!n) return;
    float peak = scores[0];
    for (std::size_t i = 1; i < n; ++i) peak = std::max(peak, scores[i]);
    float total = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { scores[i] = std::exp(scores[i] - peak); total += scores[i]; }
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    for (std::size_t i = 0; i < n; ++i) scores[i] *= inv;
}

inline void situ_glu_into(const float* gate, const float* up, std::size_t n,
                          float beta, float linear_beta, float* out, float gu_scale = 1.0f) {
    const float inv_beta = 1.0f / beta;
    const float inv_lin = 1.0f / linear_beta;
    for (std::size_t i = 0; i < n; ++i) {
        const float g = gate[i] * gu_scale;
        const float u = up[i] * gu_scale;
        const float g_act = beta * std::tanh(g * inv_beta) * sigmoid(g);
        const float u_act = linear_beta * std::tanh(u * inv_lin);
        out[i] = g_act * u_act;
    }
}

struct CacheSlot {
    std::vector<float> keys, values;   // [capacity][kv_heads * head_dim]
    std::vector<float> positions;
    std::size_t length = 0;
    // Four causal convolution histories: K, V, attention output, FFN output.
    std::array<std::vector<float>, 4> history;
};

struct LayerWeights {
    Matrix wq, wk, wv, wo, wgate;
    Matrix attn_norm, q_norm, k_norm, v_norm, ffn_norm;
    Matrix k_conv, v_conv, attn_post_conv, ffn_post_conv;
    Matrix k_conv_proj, v_conv_proj;
    Matrix q_clip, rel_clip, sink_proj, rel, rel_bias;
    Matrix attn_res_q, ffn_res_q;
    Matrix router, exp_bias, gate_up, down;
    Matrix gate_up_scale, down_scale;
    std::array<Matrix, 4> loop_down, loop_gain, loop_forget;
    bool has_attn_res = false;
    std::uint32_t attn_res_cols = 1;
    bool is_swa = false;

    const float* attn_norm_f32() const { return reinterpret_cast<const float*>(attn_norm.data); }
    const float* q_norm_f32() const { return reinterpret_cast<const float*>(q_norm.data); }
    const float* k_norm_f32() const { return reinterpret_cast<const float*>(k_norm.data); }
    const float* v_norm_f32() const { return reinterpret_cast<const float*>(v_norm.data); }
    const float* ffn_norm_f32() const { return reinterpret_cast<const float*>(ffn_norm.data); }
};

struct GpuMatrix {
    std::uint64_t ptr = 0;
    std::uint32_t type = 0;
    explicit operator bool() const { return ptr != 0; }
};

struct ExpertPool {
    std::uint64_t arena = 0;
    std::size_t slot_gu_bytes = 0;
    std::size_t slot_down_bytes = 0;
    std::size_t slot_bytes = 0;
    std::uint32_t cache_slots = 0;
    std::vector<std::uint64_t> key_of;
    std::vector<std::uint64_t> last_used;
    std::vector<std::int32_t> slot_of_key;
    std::uint32_t clock_hand = 0;
    std::uint64_t lru_tick = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;

    void clear() {
        arena = 0;
        slot_gu_bytes = slot_down_bytes = slot_bytes = 0;
        cache_slots = 0;
        key_of.clear();
        last_used.clear();
        slot_of_key.clear();
        clock_hand = 0;
        lru_tick = hits = misses = 0;
    }

    std::uint32_t evict_victim(const std::uint64_t* pinned, std::uint32_t n_pinned) {
        // Fast scan from clock_hand: pick first empty slot, else oldest among a 32-slot window
        // (or full scan if cache is tiny). Avoids O(N) scan on every miss.
        const std::uint32_t window = std::min<std::uint32_t>(cache_slots, 32u);
        std::uint32_t victim = clock_hand % cache_slots;
        std::uint64_t oldest = ~std::uint64_t(0);
        for (std::uint32_t step = 0; step < window; ++step) {
            const std::uint32_t s = (clock_hand + step) % cache_slots;
            const std::uint64_t skey = key_of[s];
            if (skey == ~std::uint64_t(0)) {
                clock_hand = (s + 1) % cache_slots;
                return s;
            }
            bool is_pinned = false;
            for (std::uint32_t m = 0; m < n_pinned; ++m) {
                if (pinned[m] == skey) { is_pinned = true; break; }
            }
            if (is_pinned) continue;
            if (last_used[s] < oldest) {
                oldest = last_used[s];
                victim = s;
            }
        }
        clock_hand = (victim + 1) % cache_slots;
        return victim;
    }
};

struct GpuLayerWeights {
    GpuMatrix wq, wk, wv, wo, wgate;
    GpuMatrix rel, sink_proj, router;
    GpuMatrix k_conv_proj, v_conv_proj;
    std::array<GpuMatrix, 4> loop_down, loop_gain, loop_forget;
    // Small F32 layer tensors for zero-sync GPU decode
    std::uint64_t d_attn_norm = 0, d_q_norm = 0, d_k_norm = 0, d_v_norm = 0, d_ffn_norm = 0;
    std::uint64_t d_k_conv = 0, d_v_conv = 0, d_attn_post_conv = 0, d_ffn_post_conv = 0;
    std::uint64_t d_q_clip = 0, d_rel_clip = 0, d_rel_bias = 0;
    std::uint64_t d_attn_res_q = 0, d_ffn_res_q = 0;
    bool moe_supported = false;
    int pool_idx = 0;
    std::size_t gu_stride = 0;
    std::size_t down_stride = 0;
};

struct GpuCacheSlot {
    std::uint64_t d_k = 0, d_v = 0, d_pos = 0;
    std::array<std::uint64_t, 4> d_hist = {0, 0, 0, 0};
    std::uint32_t length = 0;
};

struct GpuState {
    static constexpr std::uint32_t kPrefillChunk = 64;
    static constexpr std::uint32_t kMaxBlocks = 64;

    bool active = false;
    bool moe_enabled = false;
    bool int8_tensor_cores = false;
    bool full_decode_ready = false;
    bool kv_dirty = false;
    std::uint64_t stream = 0;
    std::uint64_t copy_stream = 0;
    std::uint64_t compute_done_event = 0;
    std::vector<std::uint64_t> copy_events;
    std::vector<std::uint64_t> owned;
    std::uint64_t dense_bytes = 0;

    // Scratch device buffers (decode)
    std::uint64_t d_in = 0;
    std::uint64_t d_in_a = 0, d_in_b = 0, d_in_c = 0;
    std::uint64_t d_out = 0;
    std::uint64_t d_ctrl_gate = 0, d_ctrl_gain = 0, d_ctrl_forget = 0;
    std::uint64_t d_q = 0, d_k = 0, d_v = 0, d_r = 0, d_sink = 0, d_gate = 0;
    std::uint64_t d_moe_in = 0, d_moe_gu = 0, d_moe_act = 0;
    std::uint64_t d_moe_down = 0, d_moe_accum = 0;

    // Full-GPU decode buffers & small global weights
    std::uint64_t d_embd_norm = 0, d_output_norm = 0, d_output_res_q = 0;
    std::uint64_t d_loop_res_q = 0, d_loop_kv_res_q = 0;
    std::uint64_t d_hidden = 0, d_inpSA = 0;
    std::array<std::uint64_t, 3> d_mixed = {0, 0, 0};
    std::uint64_t d_k_conv_out = 0, d_v_conv_out = 0, d_dyn_taps = 0;
    std::uint64_t d_attended = 0, d_proj = 0, d_attn_out = 0;
    std::uint64_t d_ffn_mixed = 0, d_ffn_out = 0;
    std::uint64_t d_blocks = 0, d_prelude = 0, d_prelude_blocks = 0;
    std::uint64_t d_bank = 0, d_bank_out = 0;
    std::array<std::uint64_t, 4> d_msg = {0, 0, 0, 0};
    std::vector<GpuCacheSlot> slots;

    // Scratch device buffers (batched prefill)
    std::uint64_t d_pf_in = 0, d_pf_q8 = 0, d_pf_scales = 0;
    std::uint64_t d_pf_out = 0, d_pf_act = 0;
    bool prefill_ready = false;

    // Resident weights
    GpuMatrix output;
    std::vector<GpuLayerWeights> layers;

    // Two-tier LRU GPU expert cache (small vs large experts) + pinned staging
    std::array<ExpertPool, 2> pools;
    std::size_t stage_slot_bytes = 0;
    std::size_t stage_gu_bytes = 0;
    void* h_stage = nullptr;
    std::uint32_t stage_slots = 0;

    std::uint64_t alloc(std::size_t bytes) {
        std::uint64_t ptr = 0;
        if (bytes == 0) return 0;
        if (flyweight_gpu_alloc(bytes, &ptr) != 0 || ptr == 0) return 0;
        owned.push_back(ptr);
        return ptr;
    }

    void release() {
        if (copy_stream) {
            flyweight_gpu_stream_sync(copy_stream);
            flyweight_gpu_stream_destroy(copy_stream);
            copy_stream = 0;
        }
        if (stream) {
            flyweight_gpu_stream_sync(stream);
            flyweight_gpu_stream_destroy(stream);
            stream = 0;
        }
        if (compute_done_event) {
            flyweight_gpu_event_destroy(compute_done_event);
            compute_done_event = 0;
        }
        for (auto ev : copy_events) if (ev) flyweight_gpu_event_destroy(ev);
        copy_events.clear();
        for (auto ptr : owned) flyweight_gpu_free(ptr);
        owned.clear();
        if (h_stage) {
            flyweight_gpu_host_free(h_stage);
            h_stage = nullptr;
        }
        active = false;
        moe_enabled = false;
        int8_tensor_cores = false;
        full_decode_ready = false;
        kv_dirty = false;
        prefill_ready = false;
        dense_bytes = 0;
        d_in = d_in_a = d_in_b = d_in_c = d_out = 0;
        d_ctrl_gate = d_ctrl_gain = d_ctrl_forget = 0;
        d_q = d_k = d_v = d_r = d_sink = d_gate = 0;
        d_moe_in = d_moe_gu = d_moe_act = d_moe_down = d_moe_accum = 0;
        d_pf_in = d_pf_q8 = d_pf_scales = d_pf_out = d_pf_act = 0;
        d_embd_norm = d_output_norm = d_output_res_q = d_loop_res_q = d_loop_kv_res_q = 0;
        d_hidden = d_inpSA = d_k_conv_out = d_v_conv_out = d_dyn_taps = 0;
        d_attended = d_proj = d_attn_out = d_ffn_mixed = d_ffn_out = 0;
        d_blocks = d_prelude = d_prelude_blocks = d_bank = d_bank_out = 0;
        d_mixed = {0, 0, 0};
        d_msg = {0, 0, 0, 0};
        slots.clear();
        output = {};
        layers.clear();
        pools[0].clear();
        pools[1].clear();
        stage_slot_bytes = stage_gu_bytes = 0;
        stage_slots = 0;
    }
};

struct Runtime {
    const FlyweightV2Model* model = nullptr;
    Geometry g;
    std::uint32_t capacity = 0;
    std::vector<LayerWeights> layers;
    std::vector<CacheSlot> slots;
    Matrix embeddings, output, output_norm;
    std::vector<float> loop_res_query, loop_kv_res_query;
    std::vector<float> position;                 // current token position
    std::vector<std::vector<float>> blocks;      // current pass's block outputs
    std::vector<std::vector<float>> prelude_blocks;
    std::vector<float> prelude;                  // prelude output
    mutable GpuState gpu;

    Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    ~Runtime() { gpu.release(); }

    const Tensor* find(const std::string& name) const {
        for (const auto& t : model->tensors) if (t.name == name) return &t;
        return nullptr;
    }
    Matrix matrix(const std::string& name) const {
        const auto* t = find(name);
        if (!t) throw std::runtime_error("inkling2: missing tensor " + name);
        return Matrix{tensor_data(*model, *t), t->type};
    }
    const float* floats(const std::string& name) const {
        const auto* t = find(name);
        if (!t) throw std::runtime_error("inkling2: missing tensor " + name);
        return reinterpret_cast<const float*>(tensor_data(*model, *t));
    }
    std::vector<float> float_vector(const std::string& name, std::size_t count) const {
        const float* p = floats(name);
        return std::vector<float>(p, p + count);
    }

    void reset() {
        for (auto& s : slots) {
            std::fill(s.keys.begin(), s.keys.end(), 0.0f);
            std::fill(s.values.begin(), s.values.end(), 0.0f);
            std::fill(s.positions.begin(), s.positions.end(), 0.0f);
            s.length = 0;
            for (auto& h : s.history) std::fill(h.begin(), h.end(), 0.0f);
        }
        if (gpu.active && gpu.full_decode_ready) {
            const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
            const std::size_t taps_m1 = g.conv_taps > 1 ? (g.conv_taps - 1) : 0;
            for (auto& gs : gpu.slots) {
                gs.length = 0;
                if (taps_m1 > 0) {
                    flyweight_gpu_memset(gs.d_hist[0], 0, taps_m1 * kv_channels * sizeof(float), gpu.stream);
                    flyweight_gpu_memset(gs.d_hist[1], 0, taps_m1 * kv_channels * sizeof(float), gpu.stream);
                    flyweight_gpu_memset(gs.d_hist[2], 0, taps_m1 * g.hidden * sizeof(float), gpu.stream);
                    flyweight_gpu_memset(gs.d_hist[3], 0, taps_m1 * g.hidden * sizeof(float), gpu.stream);
                }
            }
            flyweight_gpu_stream_sync(gpu.stream);
            gpu.kv_dirty = false;
        }
        position.clear();
        blocks.clear();
        prelude_blocks.clear();
        prelude.clear();
    }

    void init_gpu() {
        if (flyweight_backend_is_cpu()) return;
        if (!flyweight_gpu_available() || flyweight_gpu_init(0) != 0) return;

        std::vector<std::string> option_storage;
#if !defined(_WIN32)
        for (const char* path : {"/opt/cuda/include", "/usr/local/cuda/include", "/usr/include"}) {
            if (access((std::string(path) + "/cuda_fp16.h").c_str(), R_OK) == 0) {
                option_storage.push_back(std::string("-I") + path);
                if (access((std::string(path) + "/cccl/cub/config.cuh").c_str(), R_OK) == 0)
                    option_storage.push_back(std::string("-I") + path + "/cccl");
            }
        }
#endif
        std::vector<const char*> options;
        for (const auto& option : option_storage) options.push_back(option.c_str());
        std::array<char, 16384> log{};
        const std::string source =
            qwen_cuda_corpus() + inkling2_cuda_source + inkling2_cuda_decode_source;
        if (flyweight_gpu_compile(source.c_str(), options.data(),
                                  static_cast<std::int32_t>(options.size()), 0,
                                  log.data(), static_cast<std::int32_t>(log.size())) != 0) {
            std::fprintf(stderr, "[flyweight] inkling2: GPU kernel compile failed, using CPU: %s\n",
                         log.data());
            return;
        }

        if (flyweight_gpu_stream_create(&gpu.stream) != 0) {
            gpu.release();
            return;
        }
        flyweight_gpu_stream_create(&gpu.copy_stream);
        flyweight_gpu_event_create(&gpu.compute_done_event);

        std::uint32_t max_heads = g.heads;
        std::uint32_t max_kvh = g.kv_heads;
        for (auto h : g.heads_by_layer) max_heads = std::max(max_heads, h);
        for (auto h : g.kv_heads_by_layer) max_kvh = std::max(max_kvh, h);

        const std::size_t max_q = static_cast<std::size_t>(max_heads) * g.head_dim;
        const std::size_t max_kv = static_cast<std::size_t>(max_kvh) * g.head_dim;
        const std::size_t max_r = static_cast<std::size_t>(max_heads) * g.rel_len;
        const std::size_t max_in = std::max({
            static_cast<std::size_t>(2) * g.hidden,
            max_q,
            static_cast<std::size_t>(g.gate_rank),
        });
        const std::size_t max_out = std::max({
            static_cast<std::size_t>(g.vocab),
            static_cast<std::size_t>(g.hidden),
            max_q,
            max_kv,
            max_r,
            static_cast<std::size_t>(g.experts),
            static_cast<std::size_t>(g.gate_rank),
            static_cast<std::size_t>(g.conv_taps) * max_kvh,
        });

        gpu.d_in = gpu.alloc(max_in * sizeof(float));
        gpu.d_in_a = gpu.alloc(g.hidden * sizeof(float));
        gpu.d_in_b = gpu.alloc(g.hidden * sizeof(float));
        gpu.d_in_c = gpu.alloc(g.hidden * sizeof(float));
        gpu.d_out = gpu.alloc(max_out * sizeof(float));
        gpu.d_ctrl_gate = gpu.alloc(std::max<std::size_t>(1, g.gate_rank) * sizeof(float));
        gpu.d_ctrl_gain = gpu.alloc(g.hidden * sizeof(float));
        gpu.d_ctrl_forget = gpu.alloc(g.hidden * sizeof(float));
        gpu.d_q = gpu.alloc(max_q * sizeof(float));
        gpu.d_k = gpu.alloc(max_kv * sizeof(float));
        gpu.d_v = gpu.alloc(max_kv * sizeof(float));
        gpu.d_r = gpu.alloc(max_r * sizeof(float));
        gpu.d_sink = gpu.alloc(static_cast<std::size_t>(max_heads) * sizeof(float));
        gpu.d_gate = gpu.alloc(max_q * sizeof(float));
        gpu.d_moe_in = gpu.alloc(g.hidden * sizeof(float));
        gpu.d_moe_gu = gpu.alloc(static_cast<std::size_t>(2) * g.ff_exp * sizeof(float));
        gpu.d_moe_act = gpu.alloc(static_cast<std::size_t>(g.ff_exp) * sizeof(float));
        gpu.d_moe_down = gpu.alloc(g.hidden * sizeof(float));
        gpu.d_moe_accum = gpu.alloc(g.hidden * sizeof(float));

        if (!gpu.d_in || !gpu.d_in_a || !gpu.d_in_b || !gpu.d_in_c || !gpu.d_out ||
            !gpu.d_q || !gpu.d_k || !gpu.d_v || !gpu.d_r || !gpu.d_sink || !gpu.d_gate ||
            !gpu.d_moe_in || !gpu.d_moe_gu || !gpu.d_moe_act || !gpu.d_moe_down || !gpu.d_moe_accum) {
            gpu.release();
            return;
        }

        {
            const std::size_t pf_in_row = std::max({
                static_cast<std::size_t>(2) * g.hidden,
                max_q,
                static_cast<std::size_t>(g.ff_exp),
                static_cast<std::size_t>(g.gate_rank),
            });
            const std::size_t pf_out_row = std::max({
                static_cast<std::size_t>(g.hidden),
                max_q,
                max_kv,
                max_r,
                static_cast<std::size_t>(2) * g.ff_exp,
                static_cast<std::size_t>(g.experts),
                static_cast<std::size_t>(g.gate_rank),
                static_cast<std::size_t>(g.conv_taps) * max_kvh,
            });
            const std::size_t pf_in_elems = static_cast<std::size_t>(GpuState::kPrefillChunk) * pf_in_row;
            const std::size_t pf_out_elems = static_cast<std::size_t>(GpuState::kPrefillChunk) * pf_out_row;
            gpu.d_pf_in = gpu.alloc(pf_in_elems * sizeof(float));
            gpu.d_pf_q8 = gpu.alloc(pf_in_elems);
            gpu.d_pf_scales = gpu.alloc((pf_in_elems / 32 + GpuState::kPrefillChunk) * sizeof(float));
            gpu.d_pf_out = gpu.alloc(pf_out_elems * sizeof(float));
            gpu.d_pf_act = gpu.alloc(static_cast<std::size_t>(GpuState::kPrefillChunk) *
                                     std::max<std::size_t>(1, g.ff_exp) * sizeof(float));
            gpu.prefill_ready = (gpu.d_pf_in && gpu.d_pf_q8 && gpu.d_pf_scales &&
                                 gpu.d_pf_out && gpu.d_pf_act);
        }

        FlyweightV2GpuInfo info{};
        gpu_probe(info, 0);
        gpu.int8_tensor_cores = (info.compute_major * 10 + info.compute_minor >= 80);
        const std::uint64_t margin = info.total_memory >= (2ull << 30)
            ? (512ull << 20)
            : (info.total_memory / 4);
        std::uint64_t budget = info.free_memory > margin ? (info.free_memory - margin) : 0;

        auto upload_matrix = [&](const std::string& name, std::size_t cols) -> GpuMatrix {
            const auto* t = find(name);
            if (!t || !t->size || !gpu_type_supported(t->type, cols)) return {};
            if (t->size > budget) return {};
            const std::uint64_t ptr = gpu.alloc(t->size);
            if (!ptr) return {};
            if (flyweight_gpu_upload_sync(ptr, tensor_data(*model, *t), t->size) != 0) return {};
            budget -= t->size;
            gpu.dense_bytes += t->size;
            return GpuMatrix{ptr, t->type};
        };

        gpu.output = upload_matrix("output.weight", g.hidden);
        gpu.layers.resize(g.physical);
        std::size_t max_gu_stride = 0;
        std::size_t max_down_stride = 0;
        bool any_moe_supported = false;

        for (std::uint32_t p = 0; p < g.physical; ++p) {
            const std::string pre = "blk." + std::to_string(p) + ".";
            const LayerWeights& lw = layers[p];
            GpuLayerWeights& glw = gpu.layers[p];
            const std::uint32_t heads = g.heads_by_layer.size() > p ? g.heads_by_layer[p] : g.heads;

            glw.wq = upload_matrix(pre + "attn_q.weight", g.hidden);
            glw.wk = upload_matrix(pre + "attn_k.weight", g.hidden);
            glw.wv = upload_matrix(pre + "attn_v.weight", g.hidden);
            glw.wo = upload_matrix(pre + "attn_output.weight", static_cast<std::size_t>(heads) * g.head_dim);
            glw.wgate = upload_matrix(pre + "attn_gate.weight", g.hidden);
            glw.rel = upload_matrix(pre + "attn_rel.weight", g.hidden);
            glw.sink_proj = upload_matrix(pre + "attn_sink_proj.weight", g.hidden);
            glw.router = upload_matrix(pre + "ffn_gate_inp.weight", g.hidden);
            if (lw.k_conv_proj) glw.k_conv_proj = upload_matrix(pre + "attn_k_conv_proj.weight", g.hidden);
            if (lw.v_conv_proj) glw.v_conv_proj = upload_matrix(pre + "attn_v_conv_proj.weight", g.hidden);
            static const char* names[4] = {"attn", "k", "v", "mlp"};
            for (int c = 0; c < 4; ++c) {
                if (lw.loop_down[c]) {
                    const std::string stem = pre + "loop_" + names[c] + "_";
                    glw.loop_down[c]   = upload_matrix(stem + "down.weight", static_cast<std::size_t>(2) * g.hidden);
                    glw.loop_gain[c]   = upload_matrix(stem + "gain.weight", g.gate_rank);
                    glw.loop_forget[c] = upload_matrix(stem + "forget.weight", g.gate_rank);
                }
            }

            if (g.experts > 0 && g.ff_exp > 0 &&
                gpu_type_supported(lw.gate_up.type, g.hidden) &&
                gpu_type_supported(lw.down.type, g.ff_exp)) {
                glw.gu_stride = static_cast<std::size_t>(2) * g.ff_exp * row_bytes(lw.gate_up.type, g.hidden);
                glw.down_stride = static_cast<std::size_t>(g.hidden) * row_bytes(lw.down.type, g.ff_exp);
                const auto* gu_t = find(pre + "ffn_gate_up_exps.weight");
                const auto* dn_t = find(pre + "ffn_down_exps.weight");
                if (gu_t && dn_t &&
                    glw.gu_stride * g.experts <= gu_t->size &&
                    glw.down_stride * g.experts <= dn_t->size) {
                    glw.moe_supported = true;
                    any_moe_supported = true;
                    max_gu_stride = std::max(max_gu_stride, glw.gu_stride);
                    max_down_stride = std::max(max_down_stride, glw.down_stride);
                }
            }
        }

        // Allocate full-GPU decode state (small F32 tensors, conv histories, FP16 KV cache)
        // when all dense projections are resident and KV cache fits comfortably in budget.
        bool all_dense_gpu = bool(gpu.output) && g.head_dim <= 128 && g.physical <= 60;
        for (std::uint32_t p = 0; p < g.physical && all_dense_gpu; ++p) {
            const auto& lw = layers[p];
            const auto& glw = gpu.layers[p];
            if (!glw.wq || !glw.wk || !glw.wv || !glw.wo || !glw.wgate ||
                !glw.rel || !glw.sink_proj || !glw.router ||
                (lw.k_conv_proj && !glw.k_conv_proj) || (lw.v_conv_proj && !glw.v_conv_proj))
                all_dense_gpu = false;
            for (int c = 0; c < 4 && all_dense_gpu; ++c)
                if (lw.loop_down[c] && (!glw.loop_down[c] || !glw.loop_gain[c] || !glw.loop_forget[c]))
                    all_dense_gpu = false;
        }
        if (all_dense_gpu) {
            const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
            const std::uint64_t kv_bytes =
                static_cast<std::uint64_t>(g.slots) * capacity * (2 * kv_channels * 2 + 4);
            const std::uint64_t hist_bytes =
                static_cast<std::uint64_t>(g.slots) * std::max<std::size_t>(1, g.conv_taps - 1) *
                (2 * kv_channels + 2 * g.hidden) * sizeof(float);
            const std::uint64_t reserve_for_experts =
                any_moe_supported ? std::min<std::uint64_t>(budget / 2, 1536ull << 20) : 0ull;
            if (kv_bytes + hist_bytes + (32ull << 20) + reserve_for_experts <= budget) {
                auto upload_f32 = [&](const std::string& name) -> std::uint64_t {
                    const auto* t = find(name);
                    if (!t || !t->size || t->type != 0 || t->size > budget) return 0;
                    const std::uint64_t ptr = gpu.alloc(t->size);
                    if (!ptr || flyweight_gpu_upload_sync(ptr, tensor_data(*model, *t), t->size) != 0)
                        return 0;
                    budget -= t->size;
                    gpu.dense_bytes += t->size;
                    return ptr;
                };
                bool ok = true;
                gpu.d_embd_norm = upload_f32("token_embd_norm.weight");
                gpu.d_output_norm = upload_f32("output_norm.weight");
                gpu.d_output_res_q = upload_f32("output_res_query.weight");
                gpu.d_loop_res_q = upload_f32("loop_res_query.weight");
                gpu.d_loop_kv_res_q = upload_f32("loop_kv_res_query.weight");
                ok &= (gpu.d_embd_norm && gpu.d_output_norm && gpu.d_output_res_q &&
                       gpu.d_loop_res_q && gpu.d_loop_kv_res_q);
                for (std::uint32_t p = 0; p < g.physical && ok; ++p) {
                    const std::string pre = "blk." + std::to_string(p) + ".";
                    const auto& lw = layers[p];
                    auto& glw = gpu.layers[p];
                    if (lw.attn_norm) glw.d_attn_norm = upload_f32(pre + "attn_norm.weight");
                    if (lw.q_norm) glw.d_q_norm = upload_f32(pre + "attn_q_inp_norm.weight");
                    if (lw.k_norm) glw.d_k_norm = upload_f32(pre + "attn_k_inp_norm.weight");
                    if (lw.v_norm) glw.d_v_norm = upload_f32(pre + "attn_v_inp_norm.weight");
                    glw.d_ffn_norm = upload_f32(pre + "ffn_norm.weight");
                    glw.d_k_conv = upload_f32(pre + "attn_k_conv.weight");
                    glw.d_v_conv = upload_f32(pre + "attn_v_conv.weight");
                    glw.d_attn_post_conv = upload_f32(pre + "attn_post_conv.weight");
                    glw.d_ffn_post_conv = upload_f32(pre + "ffn_post_conv.weight");
                    glw.d_q_clip = upload_f32(pre + "attn_q_clip.weight");
                    glw.d_rel_clip = upload_f32(pre + "attn_rel_clip.weight");
                    glw.d_rel_bias = upload_f32(pre + "attn_rel_bias.weight");
                    if (lw.attn_res_q) glw.d_attn_res_q = upload_f32(pre + "attn_res_query.weight");
                    glw.d_ffn_res_q = upload_f32(pre + "ffn_res_query.weight");
                    ok &= (glw.d_ffn_norm && glw.d_k_conv && glw.d_v_conv &&
                           glw.d_attn_post_conv && glw.d_ffn_post_conv &&
                           glw.d_q_clip && glw.d_rel_clip && glw.d_rel_bias && glw.d_ffn_res_q);
                }
                if (ok) {
                    const std::size_t hbytes = g.hidden * sizeof(float);
                    gpu.d_hidden = gpu.alloc(hbytes);
                    gpu.d_inpSA = gpu.alloc(hbytes);
                    for (int c = 0; c < 3; ++c) gpu.d_mixed[c] = gpu.alloc(hbytes);
                    gpu.d_k_conv_out = gpu.alloc(max_kv * sizeof(float));
                    gpu.d_v_conv_out = gpu.alloc(max_kv * sizeof(float));
                    gpu.d_dyn_taps = gpu.alloc(static_cast<std::size_t>(g.conv_taps) * max_kvh * sizeof(float));
                    gpu.d_attended = gpu.alloc(max_q * sizeof(float));
                    gpu.d_proj = gpu.alloc(hbytes);
                    gpu.d_attn_out = gpu.alloc(hbytes);
                    gpu.d_ffn_mixed = gpu.alloc(hbytes);
                    gpu.d_ffn_out = gpu.alloc(hbytes);
                    gpu.d_blocks = gpu.alloc(static_cast<std::size_t>(GpuState::kMaxBlocks) * hbytes);
                    gpu.d_prelude = gpu.alloc(hbytes);
                    gpu.d_prelude_blocks = gpu.alloc(static_cast<std::size_t>(GpuState::kMaxBlocks) * hbytes);
                    gpu.d_bank = gpu.alloc(static_cast<std::size_t>(GpuState::kMaxBlocks) * hbytes);
                    gpu.d_bank_out = gpu.alloc(hbytes);
                    for (int c = 0; c < 4; ++c) gpu.d_msg[c] = gpu.alloc(hbytes);
                    ok &= (gpu.d_hidden && gpu.d_inpSA && gpu.d_mixed[0] && gpu.d_mixed[1] && gpu.d_mixed[2] &&
                           gpu.d_k_conv_out && gpu.d_v_conv_out && gpu.d_dyn_taps && gpu.d_attended &&
                           gpu.d_proj && gpu.d_attn_out && gpu.d_ffn_mixed && gpu.d_ffn_out &&
                           gpu.d_blocks && gpu.d_prelude && gpu.d_prelude_blocks &&
                           gpu.d_bank && gpu.d_bank_out &&
                           gpu.d_msg[0] && gpu.d_msg[1] && gpu.d_msg[2] && gpu.d_msg[3]);
                }
                if (ok) {
                    const std::size_t taps_m1 = std::max<std::size_t>(1, g.conv_taps - 1);
                    gpu.slots.resize(g.slots);
                    for (std::uint32_t s = 0; s < g.slots && ok; ++s) {
                        auto& gs = gpu.slots[s];
                        gs.d_k = gpu.alloc(static_cast<std::size_t>(capacity) * kv_channels * 2);
                        gs.d_v = gpu.alloc(static_cast<std::size_t>(capacity) * kv_channels * 2);
                        gs.d_pos = gpu.alloc(static_cast<std::size_t>(capacity) * sizeof(float));
                        gs.d_hist[0] = gpu.alloc(taps_m1 * kv_channels * sizeof(float));
                        gs.d_hist[1] = gpu.alloc(taps_m1 * kv_channels * sizeof(float));
                        gs.d_hist[2] = gpu.alloc(taps_m1 * g.hidden * sizeof(float));
                        gs.d_hist[3] = gpu.alloc(taps_m1 * g.hidden * sizeof(float));
                        gs.length = 0;
                        ok &= (gs.d_k && gs.d_v && gs.d_pos &&
                               gs.d_hist[0] && gs.d_hist[1] && gs.d_hist[2] && gs.d_hist[3]);
                        if (ok) {
                            flyweight_gpu_memset(gs.d_hist[0], 0, taps_m1 * kv_channels * sizeof(float), gpu.stream);
                            flyweight_gpu_memset(gs.d_hist[1], 0, taps_m1 * kv_channels * sizeof(float), gpu.stream);
                            flyweight_gpu_memset(gs.d_hist[2], 0, taps_m1 * g.hidden * sizeof(float), gpu.stream);
                            flyweight_gpu_memset(gs.d_hist[3], 0, taps_m1 * g.hidden * sizeof(float), gpu.stream);
                        }
                    }
                    if (ok) {
                        flyweight_gpu_stream_sync(gpu.stream);
                        budget -= (kv_bytes + hist_bytes);
                        gpu.full_decode_ready = true;
                    }
                }
            }
        }

        // Size the two-tier LRU GPU expert cache from remaining VRAM budget so
        // smaller IQ1_S/Q2_0 layers do not pay the slot padding of NVFP4 layers.
        if (any_moe_supported && max_gu_stride > 0 && max_down_stride > 0) {
            std::size_t min_total = ~std::size_t(0), max_total = 0;
            for (std::uint32_t p = 0; p < g.physical; ++p) {
                const auto& glw = gpu.layers[p];
                if (!glw.moe_supported) continue;
                const std::size_t sz = align256(glw.gu_stride) + align256(glw.down_stride);
                min_total = std::min(min_total, sz);
                max_total = std::max(max_total, sz);
            }
            const bool two_tier = (max_total > min_total + (min_total / 4));
            const std::size_t split_threshold = min_total + (max_total - min_total) / 2;
            std::uint64_t tier_experts[2] = {0, 0};
            std::size_t tier_gu[2] = {0, 0}, tier_dn[2] = {0, 0};
            for (std::uint32_t p = 0; p < g.physical; ++p) {
                auto& glw = gpu.layers[p];
                if (!glw.moe_supported) continue;
                const std::size_t sz = align256(glw.gu_stride) + align256(glw.down_stride);
                const int ti = (two_tier && sz > split_threshold) ? 1 : 0;
                glw.pool_idx = ti;
                tier_experts[ti] += g.experts;
                tier_gu[ti] = std::max(tier_gu[ti], align256(glw.gu_stride));
                tier_dn[ti] = std::max(tier_dn[ti], align256(glw.down_stride));
            }

            const std::uint32_t min_slots = std::max(1u, std::min(g.experts_used, g.experts));
            const int active_tiers = two_tier && tier_experts[1] > 0 ? 2 : 1;
            const std::uint64_t total_need =
                tier_experts[0] * (tier_gu[0] + tier_dn[0]) +
                (active_tiers > 1 ? tier_experts[1] * (tier_gu[1] + tier_dn[1]) : 0ull);
            bool all_ok = true;
            for (int ti = 0; ti < active_tiers; ++ti) {
                auto& pool = gpu.pools[ti];
                pool.slot_gu_bytes = tier_gu[ti];
                pool.slot_down_bytes = tier_dn[ti];
                pool.slot_bytes = tier_gu[ti] + tier_dn[ti];
                const std::uint64_t tier_need = tier_experts[ti] * pool.slot_bytes;
                const std::uint64_t tier_budget = (total_need > 0 && active_tiers > 1)
                    ? static_cast<std::uint64_t>((static_cast<double>(tier_need) / static_cast<double>(total_need)) *
                                                 static_cast<double>(budget))
                    : budget;
                std::uint64_t want = std::min(tier_experts[ti], tier_budget / pool.slot_bytes);
                if (want < min_slots && budget >= min_slots * pool.slot_bytes) want = min_slots;
                bool allocated = false;
                while (want >= min_slots) {
                    const std::uint64_t arena = gpu.alloc(static_cast<std::size_t>(want * pool.slot_bytes));
                    if (arena) {
                        pool.arena = arena;
                        pool.cache_slots = static_cast<std::uint32_t>(want);
                        pool.key_of.assign(pool.cache_slots, ~std::uint64_t(0));
                        pool.last_used.assign(pool.cache_slots, 0);
                        pool.slot_of_key.assign(static_cast<std::size_t>(g.physical) * g.experts, -1);
                        pool.clock_hand = 0;
                        allocated = true;
                        break;
                    }
                    want = (want * 3) / 4;
                }
                if (!allocated) all_ok = false;
            }
            if (all_ok) {
                gpu.stage_gu_bytes = align256(max_gu_stride);
                gpu.stage_slot_bytes = gpu.stage_gu_bytes + align256(max_down_stride);
                if (flyweight_gpu_host_alloc(static_cast<std::size_t>(min_slots) * gpu.stage_slot_bytes,
                                             &gpu.h_stage) == 0 && gpu.h_stage) {
                    gpu.stage_slots = min_slots;
                }
                gpu.copy_events.assign(min_slots, 0);
                for (std::uint32_t i = 0; i < min_slots; ++i)
                    flyweight_gpu_event_create(&gpu.copy_events[i]);
                gpu.moe_enabled = true;
            }
        }

        gpu.active = true;
        const std::uint64_t total_experts =
            static_cast<std::uint64_t>(g.physical) * g.experts;
        const std::uint32_t total_slots = gpu.pools[0].cache_slots + gpu.pools[1].cache_slots;
        const double dense_mib = static_cast<double>(gpu.dense_bytes) / (1024.0 * 1024.0);
        const double expert_mib =
            static_cast<double>(static_cast<std::uint64_t>(gpu.pools[0].cache_slots) * gpu.pools[0].slot_bytes +
                                static_cast<std::uint64_t>(gpu.pools[1].cache_slots) * gpu.pools[1].slot_bytes) /
            (1024.0 * 1024.0);
        std::fprintf(stderr,
            "[flyweight] inkling2: GPU decode enabled (dense %.1f MiB, expert cache %u/%llu slots, %.1f MiB)\n",
            dense_mib, total_slots,
            static_cast<unsigned long long>(total_experts), expert_mib);
    }

    void run_matvec(GpuMatrix gm, Matrix cpu_m, const float* x,
                    std::size_t cols, std::size_t rows, float* out) const {
        if (gpu.active && gm) {
            if (flyweight_gpu_upload(gpu.d_in, x, cols * sizeof(float), gpu.stream) == 0 &&
                qwen_gpu_matvec_by_type(gm.type, gm.ptr, gpu.d_in, gpu.d_out,
                                        static_cast<int>(cols), static_cast<int>(rows),
                                        gpu.stream) == 0 &&
                flyweight_gpu_download(out, gpu.d_out, rows * sizeof(float), gpu.stream) == 0 &&
                flyweight_gpu_stream_sync(gpu.stream) == 0) {
                return;
            }
        }
        matvec(cpu_m, x, cols, rows, out);
    }

    // Batched device GEMM: output[rows][out_cols] = input[rows][in_cols] * W^T.
    bool gpu_gemm(std::uint32_t type, std::uint64_t matrix,
                  std::uint64_t input, std::uint64_t output,
                  int rows, int in_cols, int out_cols,
                  float nvfp4_scale = 1.0f) const {
        if (rows <= 0) return true;
        if (rows == 1) {
            return qwen_gpu_matvec_by_type(type, matrix, input, output,
                                           in_cols, out_cols, gpu.stream, nvfp4_scale) == 0;
        }
        if (type == 40) {
            void* args[] = {&matrix, &input, &output, &in_cols, &out_cols, &rows, &nvfp4_scale};
            return flyweight_gpu_launch_named(
                "nvfp4_matmul_tiled",
                static_cast<std::uint32_t>((out_cols + 31) / 32),
                static_cast<std::uint32_t>((rows + 31) / 32),
                256, 0, gpu.stream, args) == 0;
        }
        const auto* format = flyweight::v2::qwen_format(type);
        if (format && format->matmul_q8_mmq && gpu.int8_tensor_cores &&
            (in_cols & 255) == 0 && rows >= 8) {
            int scale_stride = (in_cols / 32) * 2;
            std::uint64_t q8 = gpu.d_pf_q8;
            std::uint64_t scales = gpu.d_pf_scales;
            void* qargs[] = {&input, &q8, &scales, &in_cols, &scale_stride};
            if (flyweight_gpu_launch_named(
                    "quantize_q8_blocks_rows",
                    static_cast<std::uint32_t>((in_cols + 255) / 256),
                    static_cast<std::uint32_t>(rows), 256, 0, gpu.stream, qargs) != 0) {
                return false;
            }
            const auto shape = qwen_mmq_shape();
            const int mmq_rows = format->mmq_min ? kQ8MmqMinRows : shape.rows();
            const int mmq_threads = format->mmq_min ? kQ8MmqMinThreads : shape.threads();
            const int mmq_tokens = format->mmq_min ? kQ8MmqMinTokens : shape.tokens();
            const std::uint32_t shared =
                (format->mmq_dynamic_shared && !format->mmq_min)
                    ? qwen_mmq_dynamic_shared(shape) : 0u;
            for (int start = 0; start < rows; start += mmq_tokens) {
                int batch = std::min(mmq_tokens, rows - start);
                std::uint64_t batch_q8 = q8 + static_cast<std::uint64_t>(start) * in_cols;
                std::uint64_t batch_scales =
                    scales + static_cast<std::uint64_t>(start) * (in_cols / 32) * sizeof(float);
                std::uint64_t batch_output =
                    output + static_cast<std::uint64_t>(start) * out_cols * sizeof(float);
                void* margs[] = {&matrix, &batch_q8, &batch_scales, &batch_output,
                                 &in_cols, &out_cols, &batch, &scale_stride};
                if (flyweight_gpu_launch_named(
                        format->matmul_q8_mmq,
                        static_cast<std::uint32_t>((out_cols + mmq_rows - 1) / mmq_rows),
                        1, static_cast<std::uint32_t>(mmq_threads),
                        shared, gpu.stream, margs) != 0) {
                    return false;
                }
            }
            return true;
        }
        if (format && format->matmul_rows) {
            void* args[] = {&matrix, &input, &output, &in_cols, &out_cols, &rows};
            std::uint32_t gx = static_cast<std::uint32_t>(out_cols);
            std::uint32_t gy = static_cast<std::uint32_t>(rows);
            switch (format->matmul_rows_grid) {
                case flyweight::v2::RowsMatmulGrid::tiled32:
                    gx = static_cast<std::uint32_t>((out_cols + 31) / 32);
                    gy = static_cast<std::uint32_t>((rows + 31) / 32);
                    break;
                case flyweight::v2::RowsMatmulGrid::quad_pack:
                    gx = static_cast<std::uint32_t>(out_cols);
                    gy = static_cast<std::uint32_t>((rows + 3) / 4);
                    break;
                default:
                    break;
            }
            if (flyweight_gpu_launch_named(format->matmul_rows, gx, gy, 256, 0,
                                           gpu.stream, args) == 0) {
                return true;
            }
        }
        for (int r = 0; r < rows; ++r) {
            const std::uint64_t row_in =
                input + static_cast<std::uint64_t>(r) * in_cols * sizeof(float);
            const std::uint64_t row_out =
                output + static_cast<std::uint64_t>(r) * out_cols * sizeof(float);
            if (qwen_gpu_matvec_by_type(type, matrix, row_in, row_out,
                                        in_cols, out_cols, gpu.stream) != 0) {
                return false;
            }
        }
        return true;
    }

    void run_gemm(GpuMatrix gm, Matrix cpu_m, const float* x,
                  std::size_t rows, std::size_t in_cols, std::size_t out_cols,
                  float* out) const {
        if (rows == 0) return;
        if (gpu.active && gpu.prefill_ready && gm && rows <= GpuState::kPrefillChunk) {
            if (flyweight_gpu_upload(gpu.d_pf_in, x, rows * in_cols * sizeof(float), gpu.stream) == 0 &&
                gpu_gemm(gm.type, gm.ptr, gpu.d_pf_in, gpu.d_pf_out,
                         static_cast<int>(rows), static_cast<int>(in_cols),
                         static_cast<int>(out_cols)) &&
                flyweight_gpu_download(out, gpu.d_pf_out, rows * out_cols * sizeof(float), gpu.stream) == 0 &&
                flyweight_gpu_stream_sync(gpu.stream) == 0) {
                return;
            }
        }
        for (std::size_t r = 0; r < rows; ++r)
            matvec(cpu_m, x + r * in_cols, in_cols, out_cols, out + r * out_cols);
    }

    void init(const FlyweightV2Model* m, std::uint32_t context) {
        model = m;
        g = geometry(*m);
        capacity = std::max(1u, context);
        embeddings = matrix("token_embd.weight");
        output = matrix("output.weight");
        output_norm = matrix("output_norm.weight");
        loop_res_query = float_vector("loop_res_query.weight", g.hidden * 24);
        loop_kv_res_query = float_vector("loop_kv_res_query.weight", g.hidden * 24);

        for (std::uint32_t p = 0; p < g.physical; ++p) {
            const std::string pre = "blk." + std::to_string(p) + ".";
            LayerWeights lw;
            lw.wq = matrix(pre + "attn_q.weight");
            lw.wk = matrix(pre + "attn_k.weight");
            lw.wv = matrix(pre + "attn_v.weight");
            lw.wo = matrix(pre + "attn_output.weight");
            lw.wgate = matrix(pre + "attn_gate.weight");
            if (const auto* t = find(pre + "attn_norm.weight"))
                lw.attn_norm = Matrix{tensor_data(*m, *t), t->type};
            lw.ffn_norm = matrix(pre + "ffn_norm.weight");
            lw.k_conv = matrix(pre + "attn_k_conv.weight");
            lw.v_conv = matrix(pre + "attn_v_conv.weight");
            lw.attn_post_conv = matrix(pre + "attn_post_conv.weight");
            lw.ffn_post_conv = matrix(pre + "ffn_post_conv.weight");
            lw.q_clip = matrix(pre + "attn_q_clip.weight");
            lw.rel_clip = matrix(pre + "attn_rel_clip.weight");
            lw.sink_proj = matrix(pre + "attn_sink_proj.weight");
            lw.rel = matrix(pre + "attn_rel.weight");
            lw.rel_bias = matrix(pre + "attn_rel_bias.weight");
            lw.router = matrix(pre + "ffn_gate_inp.weight");
            lw.gate_up = matrix(pre + "ffn_gate_up_exps.weight");
            lw.down = matrix(pre + "ffn_down_exps.weight");
            if (const auto* t = find(pre + "ffn_gate_up_exps.scale"))
                lw.gate_up_scale = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "ffn_down_exps.scale"))
                lw.down_scale = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "exp_probs_b.bias"))
                lw.exp_bias = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "attn_q_inp_norm.weight"))
                lw.q_norm = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "attn_k_inp_norm.weight"))
                lw.k_norm = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "attn_v_inp_norm.weight"))
                lw.v_norm = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "attn_k_conv_proj.weight"))
                lw.k_conv_proj = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "attn_v_conv_proj.weight"))
                lw.v_conv_proj = Matrix{tensor_data(*m, *t), t->type};
            if (const auto* t = find(pre + "attn_res_query.weight")) {
                lw.attn_res_q = Matrix{tensor_data(*m, *t), t->type};
                lw.attn_res_cols = t->shape.size() >= 2 ? static_cast<std::uint32_t>(t->shape[1]) : 1u;
            }
            lw.has_attn_res = bool(lw.attn_res_q);
            lw.ffn_res_q = matrix(pre + "ffn_res_query.weight");
            lw.is_swa = p < g.swa.size() ? g.swa[p] != 0 : false;
            static const char* names[4] = {"attn", "k", "v", "mlp"};
            for (int c = 0; c < 4; ++c) {
                const std::string stem = pre + "loop_" + names[c] + "_";
                const auto* d = find(stem + "down.weight");
                const auto* gn = find(stem + "gain.weight");
                const auto* fg = find(stem + "forget.weight");
                if (d && gn && fg) {
                    lw.loop_down[c]   = Matrix{tensor_data(*m, *d), d->type};
                    lw.loop_gain[c]   = Matrix{tensor_data(*m, *gn), gn->type};
                    lw.loop_forget[c] = Matrix{tensor_data(*m, *fg), fg->type};
                }
            }
            layers.push_back(lw);
        }

        const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
        const std::size_t conv_channels = 2 * kv_channels + 2 * g.hidden;
        slots.resize(g.slots);
        for (auto& s : slots) {
            s.keys.resize(static_cast<std::size_t>(capacity) * kv_channels);
            s.values.resize(static_cast<std::size_t>(capacity) * kv_channels);
            s.positions.resize(capacity);
            for (auto& h : s.history)
                h.resize(static_cast<std::size_t>(g.conv_taps - 1) * conv_channels);
        }
        reset();
        init_gpu();
    }

    // Attention-residual mixing over the block bank (matches llama.cpp inkling2::mix).
    // States are RMS-normalized (eps=1e-6) for both keys and values; scores are
    // scaled by 1/sqrt(hidden).
    std::vector<float> mix(const std::vector<std::vector<float>>& bank,
                           const std::vector<float>& cur,
                           const float* query, std::size_t columns) const {
        const std::size_t n = bank.size() + 1;
        std::vector<float> states(n * g.hidden);
        for (std::size_t i = 0; i < bank.size(); ++i)
            rms_norm_unit(bank[i].data(), g.hidden, 1e-6f, states.data() + i * g.hidden);
        rms_norm_unit(cur.data(), g.hidden, 1e-6f, states.data() + bank.size() * g.hidden);

        const float scale = 1.0f / std::sqrt(static_cast<float>(g.hidden));
        std::vector<float> scores(n);
        std::vector<float> out(columns * g.hidden, 0.0f);
        for (std::size_t c = 0; c < columns; ++c) {
            const float* qc = query + c * g.hidden;
            for (std::size_t i = 0; i < n; ++i) {
                float dot = 0.0f;
                const float* s = states.data() + i * g.hidden;
                for (std::size_t j = 0; j < g.hidden; ++j) dot += s[j] * qc[j];
                scores[i] = dot * scale;
            }
            softmax_into(scores.data(), n);
            float* o = out.data() + c * g.hidden;
            for (std::size_t i = 0; i < n; ++i) {
                const float* s = states.data() + i * g.hidden;
                const float w = scores[i];
                for (std::size_t j = 0; j < g.hidden; ++j) o[j] += w * s[j];
            }
        }
        return out;
    }

    // Causal short convolution with a residual. `dynamic` adds per-head taps
    // for full-attention layers.
    std::vector<float> convolve(const std::vector<float>& input,
                                Matrix weight, const Matrix* dynamic,
                                GpuMatrix dynamic_gpu,
                                const float* dynamic_input,
                                std::vector<float>& history,
                                std::size_t channels) const {
        // Depthwise causal convolution: weight is [channels][conv_taps] f32,
        // ordered oldest-to-newest ([t-(L-1), ..., t-1, t]) like ggml_ssm_conv.
        if (weight.type != 0)
            throw std::runtime_error("inkling2: expected f32 convolution weights");
        const float* w = reinterpret_cast<const float*>(weight.data);
        std::vector<float> out(channels, 0.0f);
        const std::size_t last_tap = g.conv_taps - 1;
        for (std::size_t i = 0; i < channels; ++i) {
            const float* wi = w + i * g.conv_taps;
            float total = wi[last_tap] * input[i];
            for (std::size_t lag = 1; lag < g.conv_taps; ++lag)
                total += wi[last_tap - lag] * history[(lag - 1) * channels + i];
            out[i] = total;
        }

        // Dynamic taps read the pre-update history (lag 0 = current input, lag >= 1 = past).
        if (dynamic) {
            const std::size_t heads = channels / g.head_dim;
            std::vector<float> taps_values(g.conv_taps * heads);
            if (!dynamic_input)
                throw std::runtime_error("inkling2: dynamic convolution needs its input");
            run_matvec(dynamic_gpu, *dynamic, dynamic_input, g.hidden, taps_values.size(), taps_values.data());
            for (std::size_t h = 0; h < heads; ++h)
                for (std::size_t lag = 0; lag < g.conv_taps; ++lag) {
                    const float t = taps_values[h * g.conv_taps + lag];
                    const float* window = lag == 0 ? (input.data() + h * g.head_dim)
                        : (history.data() + (lag - 1) * channels + h * g.head_dim);
                    for (std::size_t k = 0; k < g.head_dim; ++k)
                        out[h * g.head_dim + k] += t * window[k];
                }
        }

        // The current input becomes the most recent history row.
        for (std::size_t lag = g.conv_taps - 1; lag > 1; --lag)
            std::copy(history.data() + (lag - 2) * channels,
                      history.data() + (lag - 1) * channels,
                      history.data() + (lag - 1) * channels);
        if (g.conv_taps > 1) std::copy(input.begin(), input.end(), history.data());

        const float norm = std::sqrt(static_cast<float>((dynamic ? 2 : 1) * g.conv_taps + 1));
        for (std::size_t i = 0; i < input.size(); ++i) out[i] = (input[i] + out[i]) / norm;
        return out;
    }

    std::vector<float> moe(const std::vector<float>& x, const LayerWeights& lw, std::uint32_t phys) const {
        const GpuLayerWeights* glw = gpu.active ? &gpu.layers[phys] : nullptr;
        std::vector<float> logits(g.experts);
        bool x_on_gpu = false;
        if (gpu.active && glw && glw->router) {
            if (flyweight_gpu_upload(gpu.d_moe_in, x.data(), g.hidden * sizeof(float), gpu.stream) == 0 &&
                qwen_gpu_matvec_by_type(glw->router.type, glw->router.ptr, gpu.d_moe_in, gpu.d_out,
                                        static_cast<int>(g.hidden), static_cast<int>(g.experts),
                                        gpu.stream) == 0 &&
                flyweight_gpu_download(logits.data(), gpu.d_out, g.experts * sizeof(float), gpu.stream) == 0 &&
                flyweight_gpu_stream_sync(gpu.stream) == 0) {
                x_on_gpu = true;
            } else {
                matvec(lw.router, x.data(), g.hidden, g.experts, logits.data());
            }
        } else {
            matvec(lw.router, x.data(), g.hidden, g.experts, logits.data());
        }

        std::vector<float> selection(g.experts);
        std::vector<std::uint32_t> order(g.experts);
        const float* bias = lw.exp_bias ? reinterpret_cast<const float*>(lw.exp_bias.data) : nullptr;
        for (std::uint32_t e = 0; e < g.experts; ++e) {
            selection[e] = sigmoid(logits[e]) + (bias ? bias[e] : 0.0f);
            order[e] = e;
        }
        const std::uint32_t used = std::min(g.experts_used, g.experts);
        std::partial_sort(order.begin(), order.begin() + used, order.end(),
            [&](std::uint32_t a, std::uint32_t b) {
                return selection[a] != selection[b] ? selection[a] > selection[b] : a < b;
            });
        std::vector<float> weights(used);
        float total = 0.0f;
        for (std::uint32_t k = 0; k < used; ++k) {
            weights[k] = sigmoid(logits[order[k]]);
            total += weights[k];
        }
        if (used > 1) {
            const float inv = 1.0f / (total + 1e-20f);
            for (auto& w : weights) w *= inv;
        }
        for (auto& w : weights) w *= g.expert_scale;

        const std::size_t ff = g.ff_exp;
        const auto* gu_base = reinterpret_cast<const std::uint8_t*>(lw.gate_up.data);
        const auto* down_base = reinterpret_cast<const std::uint8_t*>(lw.down.data);

        // GPU MoE path: hit-first execution overlapped with async PCIe uploads on copy_stream.
        if (gpu.active && gpu.moe_enabled && glw && glw->moe_supported) {
            auto& pool = gpu.pools[glw->pool_idx];
            std::vector<float> result(g.hidden, 0.0f);
            bool ok = ((x_on_gpu || flyweight_gpu_upload(gpu.d_moe_in, x.data(),
                                                         g.hidden * sizeof(float), gpu.stream) == 0) &&
                       flyweight_gpu_memset(gpu.d_moe_accum, 0,
                                            g.hidden * sizeof(float), gpu.stream) == 0);
            if (ok) {
                std::uint64_t active_keys[32];
                for (std::uint32_t k = 0; k < used; ++k)
                    active_keys[k] = static_cast<std::uint64_t>(phys) * g.experts + order[k];

                struct ExpertJob {
                    std::uint32_t k;
                    std::uint32_t slot;
                    std::uint32_t event_idx;
                };
                ExpertJob hit_jobs[32], miss_jobs[32];
                std::uint32_t n_hits = 0, n_misses = 0;

                const bool async_copy = (gpu.copy_stream != 0 && gpu.h_stage != nullptr &&
                                         gpu.stage_slots >= used && gpu.copy_events.size() >= used);
                if (async_copy && gpu.compute_done_event)
                    flyweight_gpu_stream_wait_event(gpu.copy_stream, gpu.compute_done_event);

                // 1. Classify hits vs misses; enqueue all miss uploads on copy_stream immediately.
                for (std::uint32_t k = 0; k < used && ok; ++k) {
                    const std::uint64_t key = active_keys[k];
                    const std::int32_t cached = pool.slot_of_key[key];
                    if (cached >= 0) {
                        const auto slot = static_cast<std::uint32_t>(cached);
                        pool.last_used[slot] = ++pool.lru_tick;
                        ++pool.hits;
                        hit_jobs[n_hits++] = ExpertJob{k, slot, 0};
                    } else {
                        ++pool.misses;
                        const std::uint32_t slot = pool.evict_victim(active_keys, used);
                        if (pool.key_of[slot] != ~std::uint64_t(0))
                            pool.slot_of_key[pool.key_of[slot]] = -1;
                        pool.key_of[slot] = key;
                        pool.slot_of_key[key] = static_cast<std::int32_t>(slot);
                        pool.last_used[slot] = ++pool.lru_tick;

                        const std::uint64_t slot_base =
                            pool.arena + static_cast<std::uint64_t>(slot) * pool.slot_bytes;
                        const std::uint64_t slot_gu = slot_base;
                        const std::uint64_t slot_down = slot_base + pool.slot_gu_bytes;
                        const auto* expert_gu = gu_base + static_cast<std::size_t>(order[k]) * glw->gu_stride;
                        const auto* expert_down = down_base + static_cast<std::size_t>(order[k]) * glw->down_stride;

                        const std::uint32_t m_idx = n_misses;
                        if (async_copy) {
                            auto* stage = reinterpret_cast<std::uint8_t*>(gpu.h_stage) +
                                          static_cast<std::size_t>(m_idx) * gpu.stage_slot_bytes;
                            std::memcpy(stage, expert_gu, glw->gu_stride);
                            std::memcpy(stage + gpu.stage_gu_bytes, expert_down, glw->down_stride);
                            ok = (flyweight_gpu_upload(slot_gu, stage, glw->gu_stride, gpu.copy_stream) == 0 &&
                                  flyweight_gpu_upload(slot_down, stage + gpu.stage_gu_bytes,
                                                       glw->down_stride, gpu.copy_stream) == 0 &&
                                  flyweight_gpu_event_record(gpu.copy_events[m_idx], gpu.copy_stream) == 0);
                        } else if (gpu.h_stage && m_idx < gpu.stage_slots) {
                            auto* stage = reinterpret_cast<std::uint8_t*>(gpu.h_stage) +
                                          static_cast<std::size_t>(m_idx) * gpu.stage_slot_bytes;
                            std::memcpy(stage, expert_gu, glw->gu_stride);
                            std::memcpy(stage + gpu.stage_gu_bytes, expert_down, glw->down_stride);
                            ok = (flyweight_gpu_upload(slot_gu, stage, glw->gu_stride, gpu.stream) == 0 &&
                                  flyweight_gpu_upload(slot_down, stage + gpu.stage_gu_bytes,
                                                       glw->down_stride, gpu.stream) == 0);
                        } else {
                            ok = (flyweight_gpu_upload_sync(slot_gu, expert_gu, glw->gu_stride) == 0 &&
                                  flyweight_gpu_upload_sync(slot_down, expert_down, glw->down_stride) == 0);
                        }
                        miss_jobs[n_misses++] = ExpertJob{k, slot, m_idx};
                    }
                }

                const std::uint32_t situ_blocks = static_cast<std::uint32_t>((ff + 255) / 256);
                const std::uint32_t add_blocks = static_cast<std::uint32_t>((g.hidden + 255) / 256);

                auto run_expert_job = [&](const ExpertJob& job) -> bool {
                    const std::uint32_t k = job.k;
                    const std::uint64_t slot_base =
                        pool.arena + static_cast<std::uint64_t>(job.slot) * pool.slot_bytes;
                    const std::uint64_t slot_gu = slot_base;
                    const std::uint64_t slot_down = slot_base + pool.slot_gu_bytes;

                    float gu_scale = lw.gate_up_scale
                        ? reinterpret_cast<const float*>(lw.gate_up_scale.data)[order[k]] : 1.0f;
                    const float dn_scale = lw.down_scale
                        ? reinterpret_cast<const float*>(lw.down_scale.data)[order[k]] : 1.0f;

                    int ff_i = static_cast<int>(ff);
                    float beta = g.situ_beta;
                    float lin_beta = g.situ_linear_beta;
                    void* situ_args[] = {&gpu.d_moe_gu, &gpu.d_moe_act, &ff_i, &beta, &lin_beta, &gu_scale};

                    float w_scale = weights[k] * dn_scale;
                    int hidden_i = static_cast<int>(g.hidden);
                    void* add_args[] = {&gpu.d_moe_accum, &gpu.d_moe_down, &w_scale, &hidden_i};

                    return (qwen_gpu_matvec_by_type(lw.gate_up.type, slot_gu, gpu.d_moe_in,
                                                    gpu.d_moe_gu, static_cast<int>(g.hidden),
                                                    static_cast<int>(2 * ff), gpu.stream, 1.0f) == 0 &&
                            flyweight_gpu_launch_named("inkling2_situ_glu", situ_blocks, 1, 256, 0,
                                                       gpu.stream, situ_args) == 0 &&
                            qwen_gpu_matvec_by_type(lw.down.type, slot_down, gpu.d_moe_act,
                                                    gpu.d_moe_down, static_cast<int>(ff),
                                                    static_cast<int>(g.hidden), gpu.stream, 1.0f) == 0 &&
                            flyweight_gpu_launch_named("inkling2_scaled_add", add_blocks, 1, 256, 0,
                                                       gpu.stream, add_args) == 0);
                };

                // 2. Run all cache hits FIRST while copy_stream uploads misses over PCIe.
                for (std::uint32_t i = 0; i < n_hits && ok; ++i)
                    ok = run_expert_job(hit_jobs[i]);

                // 3. Wait for each miss upload event and immediately compute it.
                for (std::uint32_t i = 0; i < n_misses && ok; ++i) {
                    if (async_copy)
                        ok = (flyweight_gpu_stream_wait_event(gpu.stream, gpu.copy_events[miss_jobs[i].event_idx]) == 0);
                    if (ok) ok = run_expert_job(miss_jobs[i]);
                }

                if (ok && gpu.compute_done_event)
                    flyweight_gpu_event_record(gpu.compute_done_event, gpu.stream);
            }
            if (ok &&
                flyweight_gpu_download(result.data(), gpu.d_moe_accum,
                                       g.hidden * sizeof(float), gpu.stream) == 0 &&
                flyweight_gpu_stream_sync(gpu.stream) == 0) {
                return result;
            }
        }

        // CPU fallback
        const auto gate_up_kernel = row_kernel(lw.gate_up.type, g.hidden);
        const auto down_kernel = row_kernel(lw.down.type, ff);
        const std::size_t gu_stride = 2 * ff * row_bytes(lw.gate_up.type, g.hidden);
        const std::size_t down_stride = g.hidden * row_bytes(lw.down.type, ff);
        std::vector<float> result(g.hidden, 0.0f);
        for (std::uint32_t k = 0; k < used; ++k) {
            const auto* expert_gu = gu_base + static_cast<std::size_t>(order[k]) * gu_stride;
            std::vector<float> gate(ff), up(ff), act(ff), down(g.hidden);
            for (std::size_t r = 0; r < ff; ++r) {
                gate[r] = gate_up_kernel(expert_gu, x.data(), g.hidden, r);
                up[r]   = gate_up_kernel(expert_gu, x.data(), g.hidden, ff + r);
            }
            const float gu_scale = lw.gate_up_scale
                ? reinterpret_cast<const float*>(lw.gate_up_scale.data)[order[k]] : 1.0f;
            const float dn_scale = lw.down_scale
                ? reinterpret_cast<const float*>(lw.down_scale.data)[order[k]] : 1.0f;
            situ_glu_into(gate.data(), up.data(), ff, g.situ_beta, g.situ_linear_beta, act.data(), gu_scale);
            const auto* expert_down = down_base + static_cast<std::size_t>(order[k]) * down_stride;
            for (std::size_t r = 0; r < g.hidden; ++r)
                down[r] = down_kernel(expert_down, act.data(), ff, r);
            const float w = weights[k] * dn_scale;
            for (std::size_t r = 0; r < g.hidden; ++r) result[r] += w * down[r];
        }
        return result;
    }

    std::vector<float> control(const std::vector<float>& cur,
                               const std::vector<float>& message,
                               const LayerWeights& lw, std::uint32_t phys, int c) const {
        const GpuLayerWeights* glw = gpu.active ? &gpu.layers[phys] : nullptr;
        std::vector<float> pair(2 * g.hidden), gate(g.gate_rank), gain(g.hidden), forget(g.hidden);
        std::vector<float> norm_cur(g.hidden), norm_msg(g.hidden);
        rms_norm_unit(cur.data(), g.hidden, g.eps, norm_cur.data());
        rms_norm_unit(message.data(), g.hidden, g.eps, norm_msg.data());
        std::copy(norm_cur.begin(), norm_cur.end(), pair.begin());
        std::copy(norm_msg.begin(), norm_msg.end(), pair.begin() + g.hidden);
        if (gpu.active && glw && glw->loop_down[c] && glw->loop_gain[c] && glw->loop_forget[c] &&
            gpu.d_ctrl_gate && gpu.d_ctrl_gain && gpu.d_ctrl_forget &&
            flyweight_gpu_upload(gpu.d_in, pair.data(), 2 * g.hidden * sizeof(float), gpu.stream) == 0 &&
            qwen_gpu_matvec_by_type(glw->loop_down[c].type, glw->loop_down[c].ptr, gpu.d_in, gpu.d_ctrl_gate,
                                    static_cast<int>(2 * g.hidden), static_cast<int>(g.gate_rank), gpu.stream) == 0 &&
            qwen_gpu_matvec_by_type(glw->loop_gain[c].type, glw->loop_gain[c].ptr, gpu.d_ctrl_gate, gpu.d_ctrl_gain,
                                    static_cast<int>(g.gate_rank), static_cast<int>(g.hidden), gpu.stream) == 0 &&
            qwen_gpu_matvec_by_type(glw->loop_forget[c].type, glw->loop_forget[c].ptr, gpu.d_ctrl_gate, gpu.d_ctrl_forget,
                                    static_cast<int>(g.gate_rank), static_cast<int>(g.hidden), gpu.stream) == 0 &&
            flyweight_gpu_download(gain.data(), gpu.d_ctrl_gain, g.hidden * sizeof(float), gpu.stream) == 0 &&
            flyweight_gpu_download(forget.data(), gpu.d_ctrl_forget, g.hidden * sizeof(float), gpu.stream) == 0 &&
            flyweight_gpu_stream_sync(gpu.stream) == 0) {
            // Fused all 3 loop controller projections into one stream sync.
        } else {
            run_matvec(glw ? glw->loop_down[c] : GpuMatrix{}, lw.loop_down[c],
                       pair.data(), 2 * g.hidden, g.gate_rank, gate.data());
            run_matvec(glw ? glw->loop_gain[c] : GpuMatrix{}, lw.loop_gain[c],
                       gate.data(), g.gate_rank, g.hidden, gain.data());
            run_matvec(glw ? glw->loop_forget[c] : GpuMatrix{}, lw.loop_forget[c],
                       gate.data(), g.gate_rank, g.hidden, forget.data());
        }
        for (auto& v : gain) v = std::tanh(v);
        for (auto& v : forget) v = std::tanh(v);
        std::vector<float> out(g.hidden);
        for (std::size_t i = 0; i < g.hidden; ++i)
            out[i] = (1.0f - forget[i]) * cur[i] + gain[i] * message[i];
        return out;
    }

    std::vector<float> layer_step(std::uint32_t slot, std::uint32_t phys,
                                  const std::vector<float>& input,
                                  const std::vector<std::vector<float>>* messages,
                                  std::size_t pass) {
        (void)pass;
        const LayerWeights& lw = layers[phys];
        const GpuLayerWeights* glw = gpu.active ? &gpu.layers[phys] : nullptr;
        CacheSlot& cache = slots[slot];
        const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
        const std::uint32_t heads = g.heads_by_layer.size() > phys ? g.heads_by_layer[phys] : g.heads;
        const std::uint32_t kvh = g.kv_heads_by_layer.size() > phys ? g.kv_heads_by_layer[phys] : g.kv_heads;

        std::vector<float> inpSA = input;
        std::vector<float> mixed[3];
        if (lw.has_attn_res) {
            const float* qsrc = reinterpret_cast<const float*>(lw.attn_res_q.data);
            if (lw.attn_res_cols >= 3) {
                auto m = mix(blocks, input, qsrc, 3);
                for (int c = 0; c < 3; ++c)
                    mixed[c].assign(m.begin() + c * g.hidden, m.begin() + (c + 1) * g.hidden);
            } else {
                auto m = mix(blocks, input, qsrc, 1);
                mixed[0] = m;
                mixed[1] = m;
                mixed[2] = m;
            }
        } else {
            mixed[0] = mixed[1] = mixed[2] = input;
        }
        if (messages) {
            // The four messages were already produced by `mix()` over the
            // previous pass's bank; each input gets its own controller.
            for (int c = 0; c < 3; ++c)
                mixed[c] = control(mixed[c], (*messages)[c], lw, phys, c);
        }
        const float* attn_norm = lw.attn_norm ? lw.attn_norm_f32() : nullptr;
        bool shared_mixed = false;
        if (lw.is_swa && messages) {
            if (!attn_norm) throw std::runtime_error("inkling2: missing attn_norm on a sliding layer");
            for (int c = 0; c < 3; ++c)
                rms_norm_into(mixed[c].data(), attn_norm, g.hidden, g.eps, mixed[c].data());
        } else if (lw.is_swa) {
            if (!attn_norm) throw std::runtime_error("inkling2: missing attn_norm on a sliding layer");
            rms_norm_into(mixed[0].data(), attn_norm, g.hidden, g.eps, mixed[0].data());
            mixed[1] = mixed[2] = mixed[0];
            shared_mixed = true;
        } else {
            const float* qn = lw.q_norm ? lw.q_norm_f32() : attn_norm;
            const float* kn = lw.k_norm ? lw.k_norm_f32() : attn_norm;
            const float* vn = lw.v_norm ? lw.v_norm_f32() : attn_norm;
            if (!qn || !kn || !vn)
                throw std::runtime_error("inkling2: missing input norms on a full-attention layer");
            rms_norm_into(mixed[0].data(), qn, g.hidden, g.eps, mixed[0].data());
            rms_norm_into(mixed[1].data(), kn, g.hidden, g.eps, mixed[1].data());
            rms_norm_into(mixed[2].data(), vn, g.hidden, g.eps, mixed[2].data());
        }

        std::vector<float> Q(heads * g.head_dim), K(kvh * g.head_dim),
                           V(kvh * g.head_dim), R(heads * g.rel_len),
                           sinks(heads, 0.0f), gate(heads * g.head_dim);

        bool batched_gpu_ok = false;
        if (gpu.active && glw && glw->wq && glw->wk && glw->wv &&
            glw->rel && glw->sink_proj && glw->wgate) {
            const std::uint64_t in0 = gpu.d_in_a;
            const std::uint64_t in1 = shared_mixed ? gpu.d_in_a : gpu.d_in_b;
            const std::uint64_t in2 = shared_mixed ? gpu.d_in_a : gpu.d_in_c;
            bool ok = (flyweight_gpu_upload(in0, mixed[0].data(), g.hidden * sizeof(float), gpu.stream) == 0);
            if (ok && !shared_mixed) {
                ok = (flyweight_gpu_upload(in1, mixed[1].data(), g.hidden * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_upload(in2, mixed[2].data(), g.hidden * sizeof(float), gpu.stream) == 0);
            }
            if (ok) {
                ok = (qwen_gpu_matvec_by_type(glw->wq.type, glw->wq.ptr, in0, gpu.d_q,
                                              static_cast<int>(g.hidden), static_cast<int>(Q.size()), gpu.stream) == 0 &&
                      qwen_gpu_matvec_by_type(glw->wk.type, glw->wk.ptr, in1, gpu.d_k,
                                              static_cast<int>(g.hidden), static_cast<int>(K.size()), gpu.stream) == 0 &&
                      qwen_gpu_matvec_by_type(glw->wv.type, glw->wv.ptr, in2, gpu.d_v,
                                              static_cast<int>(g.hidden), static_cast<int>(V.size()), gpu.stream) == 0 &&
                      qwen_gpu_matvec_by_type(glw->rel.type, glw->rel.ptr, in0, gpu.d_r,
                                              static_cast<int>(g.hidden), static_cast<int>(R.size()), gpu.stream) == 0 &&
                      qwen_gpu_matvec_by_type(glw->sink_proj.type, glw->sink_proj.ptr, in0, gpu.d_sink,
                                              static_cast<int>(g.hidden), static_cast<int>(sinks.size()), gpu.stream) == 0 &&
                      qwen_gpu_matvec_by_type(glw->wgate.type, glw->wgate.ptr, in0, gpu.d_gate,
                                              static_cast<int>(g.hidden), static_cast<int>(gate.size()), gpu.stream) == 0 &&
                      flyweight_gpu_download(Q.data(), gpu.d_q, Q.size() * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_download(K.data(), gpu.d_k, K.size() * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_download(V.data(), gpu.d_v, V.size() * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_download(R.data(), gpu.d_r, R.size() * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_download(sinks.data(), gpu.d_sink, sinks.size() * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_download(gate.data(), gpu.d_gate, gate.size() * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_stream_sync(gpu.stream) == 0);
            }
            batched_gpu_ok = ok;
        }
        if (!batched_gpu_ok) {
            run_matvec(glw ? glw->wq : GpuMatrix{}, lw.wq, mixed[0].data(), g.hidden, Q.size(), Q.data());
            run_matvec(glw ? glw->wk : GpuMatrix{}, lw.wk, mixed[1].data(), g.hidden, K.size(), K.data());
            run_matvec(glw ? glw->wv : GpuMatrix{}, lw.wv, mixed[2].data(), g.hidden, V.size(), V.data());
            run_matvec(glw ? glw->rel : GpuMatrix{}, lw.rel, mixed[0].data(), g.hidden, R.size(), R.data());
            run_matvec(glw ? glw->sink_proj : GpuMatrix{}, lw.sink_proj, mixed[0].data(), g.hidden, sinks.size(), sinks.data());
            run_matvec(glw ? glw->wgate : GpuMatrix{}, lw.wgate, mixed[0].data(), g.hidden, gate.size(), gate.data());
        }

        const Matrix* kdyn = lw.k_conv_proj ? &lw.k_conv_proj : nullptr;
        const Matrix* vdyn = lw.v_conv_proj ? &lw.v_conv_proj : nullptr;
        K = convolve(K, lw.k_conv, kdyn, glw ? glw->k_conv_proj : GpuMatrix{},
                     mixed[1].data(), cache.history[0], kvh * g.head_dim);
        V = convolve(V, lw.v_conv, vdyn, glw ? glw->v_conv_proj : GpuMatrix{},
                     mixed[2].data(), cache.history[1], kvh * g.head_dim);
        clip_rows(Q.data(), heads, g.head_dim,
                  reinterpret_cast<const float*>(lw.q_clip.data), g.qk_clip);
        clip_rows(K.data(), kvh, g.head_dim, nullptr, g.qk_clip);
        for (std::uint32_t h = 0; h < kvh; ++h)
            rms_norm_unit(V.data() + h * g.head_dim,
                          g.head_dim, g.eps, V.data() + h * g.head_dim);
        clip_rows(R.data(), heads, g.rel_len,
                  reinterpret_cast<const float*>(lw.rel_clip.data), g.rel_clip);

        const float pos = static_cast<float>(position.back());
        const float tau = 1.0f + g.log_scale_alpha *
            std::log(std::max(1.0f, (pos + 1.0f) / static_cast<float>(g.log_scale_floor)));
        if (!lw.is_swa) {
            for (auto& v : Q) v *= tau;
            for (auto& v : R) v *= tau;
        }

        // Store K/V in the slot before attention, as llama.cpp's KV cache does.
        const std::uint32_t write = cache.length;
        std::copy(K.begin(), K.end(), cache.keys.begin() + write * kv_channels);
        std::copy(V.begin(), V.end(), cache.values.begin() + write * kv_channels);
        cache.positions[write] = pos;
        ++cache.length;

        const std::uint32_t extent = lw.is_swa ? g.sliding_window : g.rel_extent;
        const std::uint32_t active_extent = std::min(extent, static_cast<std::uint32_t>(pos) + 1u);
        // The bias table is [extent][rel_len] and the GGUF checkpoint stores it
        // as f32. Project R through it only for reachable distances, then
        // subtract the learned sink logit.
        std::vector<float> bias((active_extent + 1) * heads, 0.0f);
        if (lw.rel_bias.type != 0)
            throw std::runtime_error("inkling2: expected an f32 relative-bias table");
        {
            const float* table = reinterpret_cast<const float*>(lw.rel_bias.data);
            for (std::uint32_t d = 0; d < active_extent; ++d)
                for (std::uint32_t h = 0; h < heads; ++h) {
                    float dot = 0.0f;
                    for (std::uint32_t r = 0; r < g.rel_len; ++r)
                        dot += table[static_cast<std::size_t>(d) * g.rel_len + r] *
                               R[static_cast<std::size_t>(h) * g.rel_len + r];
                    bias[d * heads + h] = dot - sinks[h];
                }
            for (std::uint32_t h = 0; h < heads; ++h)
                bias[active_extent * heads + h] = -sinks[h];
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));
        std::vector<float> attended(heads * g.head_dim, 0.0f);
        std::vector<float> scores(cache.length + 1, 0.0f);
        for (std::uint32_t h = 0; h < heads; ++h) {
            const std::uint32_t kv_head = h / std::max(1u, heads / kvh);
            const float* q = Q.data() + h * g.head_dim;
            for (std::uint32_t j = 0; j < cache.length; ++j) {
                float dot = 0.0f;
                const float* k = cache.keys.data() + j * kv_channels + kv_head * g.head_dim;
                for (std::uint32_t i = 0; i < g.head_dim; ++i) dot += q[i] * k[i];
                const float d = pos - cache.positions[j];
                const std::uint32_t idx = (d < 0 || d >= active_extent) ? active_extent : static_cast<std::uint32_t>(d);
                scores[j] = dot * scale + bias[idx * heads + h];
            }
            scores[cache.length] = 0.0f; // sink logit
            softmax_into(scores.data(), scores.size());
            for (std::uint32_t j = 0; j < cache.length; ++j) {
                const float* v = cache.values.data() + j * kv_channels + kv_head * g.head_dim;
                const float w = scores[j];
                for (std::uint32_t i = 0; i < g.head_dim; ++i)
                    attended[h * g.head_dim + i] += w * v[i];
            }
        }

        for (auto& v : gate) v = sigmoid(v);
        for (std::size_t i = 0; i < gate.size(); ++i) attended[i] *= gate[i];
        std::vector<float> proj(g.hidden);
        run_matvec(glw ? glw->wo : GpuMatrix{}, lw.wo, attended.data(),
                   heads * g.head_dim, g.hidden, proj.data());
        auto attn_out = convolve(proj, lw.attn_post_conv, nullptr, GpuMatrix{}, nullptr, cache.history[2], g.hidden);

        if (phys % g.block_size == 0) {
            blocks.push_back(inpSA);
            inpSA = attn_out;
        } else {
            for (std::size_t i = 0; i < g.hidden; ++i) inpSA[i] += attn_out[i];
        }

        std::vector<float> ffn_inp = inpSA;
        const float* ffn_q = reinterpret_cast<const float*>(lw.ffn_res_q.data);
        auto ffn_mixed = mix(blocks, inpSA, ffn_q, 1);
        if (messages) ffn_mixed = control(ffn_mixed, (*messages)[3], lw, phys, 3);
        std::vector<float> normed(g.hidden);
        rms_norm_into(ffn_mixed.data(), lw.ffn_norm_f32(), g.hidden, g.eps, normed.data());
        auto ffn_out = moe(normed, lw, phys);
        ffn_out = convolve(ffn_out, lw.ffn_post_conv, nullptr, GpuMatrix{}, nullptr, cache.history[3], g.hidden);
        for (std::size_t i = 0; i < g.hidden; ++i) ffn_inp[i] += ffn_out[i];
        return ffn_inp;
    }

    std::vector<float> compute_logits(const std::vector<std::vector<float>>& token_blocks,
                                      const std::vector<float>& hidden) const {
        std::vector<float> mixed = mix(token_blocks, hidden,
            floats("output_res_query.weight"), 1);
        std::vector<float> normed(g.hidden);
        rms_norm_into(mixed.data(), reinterpret_cast<const float*>(output_norm.data),
                      g.hidden, g.eps, normed.data());

        if (gpu.active && gpu.output) {
            std::vector<float> logits(g.vocab_unpadded);
            bool ok = (flyweight_gpu_upload(gpu.d_in, normed.data(),
                                            g.hidden * sizeof(float), gpu.stream) == 0 &&
                       qwen_gpu_matvec_by_type(gpu.output.type, gpu.output.ptr, gpu.d_in, gpu.d_out,
                                               static_cast<int>(g.hidden), static_cast<int>(g.vocab),
                                               gpu.stream) == 0);
            if (ok && g.softcap > 0.0f) {
                int n = static_cast<int>(g.vocab_unpadded);
                float cap = g.softcap;
                void* cap_args[] = {&gpu.d_out, &n, &cap};
                ok = (flyweight_gpu_launch_named(
                          "inkling2_softcap", static_cast<std::uint32_t>((n + 255) / 256),
                          1, 256, 0, gpu.stream, cap_args) == 0);
            }
            if (ok &&
                flyweight_gpu_download(logits.data(), gpu.d_out,
                                       g.vocab_unpadded * sizeof(float), gpu.stream) == 0 &&
                flyweight_gpu_stream_sync(gpu.stream) == 0) {
                return logits;
            }
        }

        std::vector<float> logits(g.vocab);
        matvec(output, normed.data(), g.hidden, g.vocab, logits.data());
        if (g.softcap > 0.0f)
            for (auto& v : logits) v = g.softcap * std::tanh(v / g.softcap);
        logits.resize(g.vocab_unpadded);
        return logits;
    }

    static std::uint16_t float_to_half_bits(float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::uint32_t sign = (bits >> 16) & 0x8000u;
        const std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
        std::uint32_t mant = bits & 0x7FFFFFu;
        if (exp <= 0) {
            if (exp < -10) return static_cast<std::uint16_t>(sign);
            mant |= 0x800000u;
            const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
            return static_cast<std::uint16_t>(sign | ((mant + (1u << (shift - 1u))) >> shift));
        }
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
    }

    static float half_bits_to_float(std::uint16_t h) {
        const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
        const std::uint32_t exp = (h >> 10) & 0x1Fu;
        const std::uint32_t mant = h & 0x3FFu;
        std::uint32_t bits = 0;
        if (exp == 0) {
            if (mant == 0) bits = sign;
            else {
                std::uint32_t m = mant;
                std::int32_t e = -14;
                while ((m & 0x400u) == 0) { m <<= 1; --e; }
                m &= 0x3FFu;
                bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    void sync_cpu_slot_to_gpu(std::uint32_t slot, std::size_t old_len) const {
        if (!gpu.active || !gpu.full_decode_ready || slot >= gpu.slots.size()) return;
        const CacheSlot& cs = slots[slot];
        GpuCacheSlot& gs = gpu.slots[slot];
        const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
        const std::size_t added = cs.length > old_len ? (cs.length - old_len) : 0;
        if (added > 0) {
            const std::size_t elems = added * kv_channels;
            std::vector<std::uint16_t> kh(elems), vh(elems);
            const float* ks = cs.keys.data() + old_len * kv_channels;
            const float* vs = cs.values.data() + old_len * kv_channels;
            for (std::size_t i = 0; i < elems; ++i) {
                kh[i] = float_to_half_bits(ks[i]);
                vh[i] = float_to_half_bits(vs[i]);
            }
            flyweight_gpu_upload(gs.d_k + old_len * kv_channels * 2, kh.data(), elems * 2, gpu.stream);
            flyweight_gpu_upload(gs.d_v + old_len * kv_channels * 2, vh.data(), elems * 2, gpu.stream);
            flyweight_gpu_upload(gs.d_pos + old_len * sizeof(float),
                                 cs.positions.data() + old_len, added * sizeof(float), gpu.stream);
        }
        const std::size_t taps_m1 = g.conv_taps > 1 ? (g.conv_taps - 1) : 0;
        if (taps_m1 > 0) {
            flyweight_gpu_upload(gs.d_hist[0], cs.history[0].data(), taps_m1 * kv_channels * sizeof(float), gpu.stream);
            flyweight_gpu_upload(gs.d_hist[1], cs.history[1].data(), taps_m1 * kv_channels * sizeof(float), gpu.stream);
            flyweight_gpu_upload(gs.d_hist[2], cs.history[2].data(), taps_m1 * g.hidden * sizeof(float), gpu.stream);
            flyweight_gpu_upload(gs.d_hist[3], cs.history[3].data(), taps_m1 * g.hidden * sizeof(float), gpu.stream);
        }
        flyweight_gpu_stream_sync(gpu.stream);
        gs.length = static_cast<std::uint32_t>(cs.length);
    }

    void sync_gpu_slots_to_cpu() {
        if (!gpu.active || !gpu.full_decode_ready || !gpu.kv_dirty) return;
        const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
        const std::size_t taps_m1 = g.conv_taps > 1 ? (g.conv_taps - 1) : 0;
        for (std::uint32_t s = 0; s < g.slots; ++s) {
            CacheSlot& cs = slots[s];
            const GpuCacheSlot& gs = gpu.slots[s];
            if (gs.length > cs.length) {
                const std::size_t old_len = cs.length;
                const std::size_t added = gs.length - old_len;
                const std::size_t elems = added * kv_channels;
                std::vector<std::uint16_t> kh(elems), vh(elems);
                flyweight_gpu_download(kh.data(), gs.d_k + old_len * kv_channels * 2, elems * 2, gpu.stream);
                flyweight_gpu_download(vh.data(), gs.d_v + old_len * kv_channels * 2, elems * 2, gpu.stream);
                flyweight_gpu_download(cs.positions.data() + old_len,
                                       gs.d_pos + old_len * sizeof(float), added * sizeof(float), gpu.stream);
                flyweight_gpu_stream_sync(gpu.stream);
                float* kd = cs.keys.data() + old_len * kv_channels;
                float* vd = cs.values.data() + old_len * kv_channels;
                for (std::size_t i = 0; i < elems; ++i) {
                    kd[i] = half_bits_to_float(kh[i]);
                    vd[i] = half_bits_to_float(vh[i]);
                }
                cs.length = gs.length;
            }
            if (taps_m1 > 0) {
                flyweight_gpu_download(cs.history[0].data(), gs.d_hist[0], taps_m1 * kv_channels * sizeof(float), gpu.stream);
                flyweight_gpu_download(cs.history[1].data(), gs.d_hist[1], taps_m1 * kv_channels * sizeof(float), gpu.stream);
                flyweight_gpu_download(cs.history[2].data(), gs.d_hist[2], taps_m1 * g.hidden * sizeof(float), gpu.stream);
                flyweight_gpu_download(cs.history[3].data(), gs.d_hist[3], taps_m1 * g.hidden * sizeof(float), gpu.stream);
                flyweight_gpu_stream_sync(gpu.stream);
            }
        }
        gpu.kv_dirty = false;
    }

    void gpu_control_device(std::uint64_t d_cur, std::uint64_t d_msg,
                            std::uint64_t d_out, const GpuLayerWeights& glw, int c) const {
        int hidden_i = static_cast<int>(g.hidden);
        float eps = g.eps;
        void* pair_args[] = {&d_cur, &d_msg, &gpu.d_in, &hidden_i, &eps};
        flyweight_gpu_launch_named("inkling2_control_pair", 2, 1, 256, 0, gpu.stream, pair_args);
        qwen_gpu_matvec_by_type(glw.loop_down[c].type, glw.loop_down[c].ptr, gpu.d_in, gpu.d_ctrl_gate,
                                static_cast<int>(2 * g.hidden), static_cast<int>(g.gate_rank), gpu.stream);
        qwen_gpu_matvec_by_type(glw.loop_gain[c].type, glw.loop_gain[c].ptr, gpu.d_ctrl_gate, gpu.d_ctrl_gain,
                                static_cast<int>(g.gate_rank), static_cast<int>(g.hidden), gpu.stream);
        qwen_gpu_matvec_by_type(glw.loop_forget[c].type, glw.loop_forget[c].ptr, gpu.d_ctrl_gate, gpu.d_ctrl_forget,
                                static_cast<int>(g.gate_rank), static_cast<int>(g.hidden), gpu.stream);
        void* app_args[] = {&d_cur, &d_msg, &gpu.d_ctrl_gain, &gpu.d_ctrl_forget, &d_out, &hidden_i};
        flyweight_gpu_launch_named("inkling2_control_apply",
                                   static_cast<std::uint32_t>((g.hidden + 255) / 256),
                                   1, 256, 0, gpu.stream, app_args);
    }

    // Full-GPU single-token layer step: only 1 WDDM stream sync per layer (router logits).
    void layer_step_gpu(std::uint32_t slot, std::uint32_t phys,
                        std::uint32_t& n_blocks, bool has_msg, float pos) {
        const LayerWeights& lw = layers[phys];
        GpuLayerWeights& glw = gpu.layers[phys];
        GpuCacheSlot& gs = gpu.slots[slot];
        const std::size_t hbytes = g.hidden * sizeof(float);
        const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
        const std::uint32_t heads = g.heads_by_layer.size() > phys ? g.heads_by_layer[phys] : g.heads;
        const std::uint32_t kvh = g.kv_heads_by_layer.size() > phys ? g.kv_heads_by_layer[phys] : g.kv_heads;
        const std::size_t q_dim = static_cast<std::size_t>(heads) * g.head_dim;
        const std::size_t kv_dim = static_cast<std::size_t>(kvh) * g.head_dim;
        const std::size_t r_dim = static_cast<std::size_t>(heads) * g.rel_len;
        int hidden_i = static_cast<int>(g.hidden);
        float eps = g.eps;

        // inpSA = hidden
        flyweight_gpu_copy_device(gpu.d_inpSA, gpu.d_hidden, hbytes, gpu.stream);

        // Attention residual mix
        bool shared_mixed = false;
        if (lw.has_attn_res) {
            int nb = static_cast<int>(n_blocks);
            if (lw.attn_res_cols >= 3) {
                int cols = 3;
                void* margs[] = {&gpu.d_blocks, &nb, &gpu.d_hidden, &glw.d_attn_res_q, &cols, &gpu.d_out, &hidden_i};
                flyweight_gpu_launch_named("inkling2_mix", 3, 1, 256, 0, gpu.stream, margs);
                for (int c = 0; c < 3; ++c)
                    flyweight_gpu_copy_device(gpu.d_mixed[c], gpu.d_out + c * hbytes, hbytes, gpu.stream);
            } else {
                int cols = 1;
                void* margs[] = {&gpu.d_blocks, &nb, &gpu.d_hidden, &glw.d_attn_res_q, &cols, &gpu.d_mixed[0], &hidden_i};
                flyweight_gpu_launch_named("inkling2_mix", 1, 1, 256, 0, gpu.stream, margs);
                if (!lw.is_swa || has_msg) {
                    flyweight_gpu_copy_device(gpu.d_mixed[1], gpu.d_mixed[0], hbytes, gpu.stream);
                    flyweight_gpu_copy_device(gpu.d_mixed[2], gpu.d_mixed[0], hbytes, gpu.stream);
                } else {
                    shared_mixed = true;
                }
            }
        } else {
            flyweight_gpu_copy_device(gpu.d_mixed[0], gpu.d_hidden, hbytes, gpu.stream);
            if (!lw.is_swa || has_msg) {
                flyweight_gpu_copy_device(gpu.d_mixed[1], gpu.d_hidden, hbytes, gpu.stream);
                flyweight_gpu_copy_device(gpu.d_mixed[2], gpu.d_hidden, hbytes, gpu.stream);
            } else {
                shared_mixed = true;
            }
        }

        if (has_msg) {
            for (int c = 0; c < 3; ++c)
                gpu_control_device(gpu.d_mixed[c], gpu.d_msg[c], gpu.d_mixed[c], glw, c);
        }

        // Input RMSNorm(s)
        auto launch_norm = [&](std::uint64_t x, std::uint64_t w, std::uint64_t out) {
            void* nargs[] = {&x, &w, &out, &hidden_i, &eps};
            flyweight_gpu_launch_named("inkling2_rms_norm", 1, 1, 256, 0, gpu.stream, nargs);
        };
        if (lw.is_swa && !has_msg) {
            launch_norm(gpu.d_mixed[0], glw.d_attn_norm, gpu.d_mixed[0]);
            shared_mixed = true;
        } else if (lw.is_swa) {
            for (int c = 0; c < 3; ++c) launch_norm(gpu.d_mixed[c], glw.d_attn_norm, gpu.d_mixed[c]);
        } else {
            launch_norm(gpu.d_mixed[0], glw.d_q_norm ? glw.d_q_norm : glw.d_attn_norm, gpu.d_mixed[0]);
            launch_norm(gpu.d_mixed[1], glw.d_k_norm ? glw.d_k_norm : glw.d_attn_norm, gpu.d_mixed[1]);
            launch_norm(gpu.d_mixed[2], glw.d_v_norm ? glw.d_v_norm : glw.d_attn_norm, gpu.d_mixed[2]);
        }

        const std::uint64_t in0 = gpu.d_mixed[0];
        const std::uint64_t in1 = shared_mixed ? gpu.d_mixed[0] : gpu.d_mixed[1];
        const std::uint64_t in2 = shared_mixed ? gpu.d_mixed[0] : gpu.d_mixed[2];

        // Attention input projections
        qwen_gpu_matvec_by_type(glw.wq.type, glw.wq.ptr, in0, gpu.d_q, hidden_i, static_cast<int>(q_dim), gpu.stream);
        qwen_gpu_matvec_by_type(glw.wk.type, glw.wk.ptr, in1, gpu.d_k, hidden_i, static_cast<int>(kv_dim), gpu.stream);
        qwen_gpu_matvec_by_type(glw.wv.type, glw.wv.ptr, in2, gpu.d_v, hidden_i, static_cast<int>(kv_dim), gpu.stream);
        qwen_gpu_matvec_by_type(glw.rel.type, glw.rel.ptr, in0, gpu.d_r, hidden_i, static_cast<int>(r_dim), gpu.stream);
        qwen_gpu_matvec_by_type(glw.sink_proj.type, glw.sink_proj.ptr, in0, gpu.d_sink, hidden_i, static_cast<int>(heads), gpu.stream);
        qwen_gpu_matvec_by_type(glw.wgate.type, glw.wgate.ptr, in0, gpu.d_gate, hidden_i, static_cast<int>(q_dim), gpu.stream);

        // K & V causal short convolutions (with optional dynamic taps)
        int kv_dim_i = static_cast<int>(kv_dim);
        int taps_i = static_cast<int>(g.conv_taps);
        int hd_i = static_cast<int>(g.head_dim);
        {
            std::uint64_t dyn_k = 0;
            if (lw.k_conv_proj) {
                qwen_gpu_matvec_by_type(glw.k_conv_proj.type, glw.k_conv_proj.ptr, in1, gpu.d_dyn_taps,
                                        hidden_i, static_cast<int>(g.conv_taps * kvh), gpu.stream);
                dyn_k = gpu.d_dyn_taps;
            }
            void* cargs[] = {&gpu.d_k, &glw.d_k_conv, &dyn_k, &gs.d_hist[0], &gpu.d_k_conv_out, &kv_dim_i, &taps_i, &hd_i};
            flyweight_gpu_launch_named("inkling2_convolve",
                                       static_cast<std::uint32_t>((kv_dim + 255) / 256), 1, 256, 0, gpu.stream, cargs);
        }
        {
            std::uint64_t dyn_v = 0;
            if (lw.v_conv_proj) {
                qwen_gpu_matvec_by_type(glw.v_conv_proj.type, glw.v_conv_proj.ptr, in2, gpu.d_dyn_taps,
                                        hidden_i, static_cast<int>(g.conv_taps * kvh), gpu.stream);
                dyn_v = gpu.d_dyn_taps;
            }
            void* cargs[] = {&gpu.d_v, &glw.d_v_conv, &dyn_v, &gs.d_hist[1], &gpu.d_v_conv_out, &kv_dim_i, &taps_i, &hd_i};
            flyweight_gpu_launch_named("inkling2_convolve",
                                       static_cast<std::uint32_t>((kv_dim + 255) / 256), 1, 256, 0, gpu.stream, cargs);
        }

        // Clip Q, K, R + tau scaling on full-attention layers
        const float tau = lw.is_swa ? 1.0f : (1.0f + g.log_scale_alpha *
            std::log(std::max(1.0f, (pos + 1.0f) / static_cast<float>(g.log_scale_floor))));
        {
            int heads_i = static_cast<int>(heads);
            float qk_clip = g.qk_clip;
            float tau_v = tau;
            void* qargs[] = {&gpu.d_q, &glw.d_q_clip, &heads_i, &hd_i, &qk_clip, &tau_v};
            flyweight_gpu_launch_named("inkling2_clip_rows", heads, 1, 256, 0, gpu.stream, qargs);

            std::uint64_t null_ptr = 0;
            int kvh_i = static_cast<int>(kvh);
            float one = 1.0f;
            void* kargs[] = {&gpu.d_k_conv_out, &null_ptr, &kvh_i, &hd_i, &qk_clip, &one};
            flyweight_gpu_launch_named("inkling2_clip_rows", kvh, 1, 256, 0, gpu.stream, kargs);

            int rl_i = static_cast<int>(g.rel_len);
            float rel_clip = g.rel_clip;
            void* rargs[] = {&gpu.d_r, &glw.d_rel_clip, &heads_i, &rl_i, &rel_clip, &tau_v};
            flyweight_gpu_launch_named("inkling2_clip_rows", heads, 1, 256, 0, gpu.stream, rargs);
        }

        // V head RMSNorm + append K/V to FP16 KV cache
        int write_pos = static_cast<int>(gs.length);
        int kv_ch_i = static_cast<int>(kv_channels);
        int kvh_i = static_cast<int>(kvh);
        float pos_f = pos;
        {
            void* vargs[] = {&gpu.d_k_conv_out, &gpu.d_v_conv_out, &gs.d_k, &gs.d_v, &gs.d_pos,
                             &write_pos, &kv_ch_i, &kvh_i, &hd_i, &eps, &pos_f};
            flyweight_gpu_launch_named("inkling2_v_norm_kv_append", kvh, 1, 256, 0, gpu.stream, vargs);
        }
        ++gs.length;
        gpu.kv_dirty = true;

        // Relative-bias attention + sink + sigmoid(gate)
        {
            int kv_len_i = static_cast<int>(gs.length);
            int heads_i = static_cast<int>(heads);
            int rl_i = static_cast<int>(g.rel_len);
            int ext_i = static_cast<int>(lw.is_swa ? g.sliding_window : g.rel_extent);
            void* aargs[] = {&gpu.d_q, &gs.d_k, &gs.d_v, &gs.d_pos, &gpu.d_r, &glw.d_rel_bias,
                             &gpu.d_sink, &gpu.d_gate, &gpu.d_attended, &kv_len_i, &heads_i,
                             &kvh_i, &hd_i, &rl_i, &ext_i, &pos_f};
            flyweight_gpu_launch_named("inkling2_rel_attn", heads, 1, 128, 0, gpu.stream, aargs);
        }

        // Output projection + post-attention conv
        qwen_gpu_matvec_by_type(glw.wo.type, glw.wo.ptr, gpu.d_attended, gpu.d_proj,
                                static_cast<int>(q_dim), hidden_i, gpu.stream);
        {
            std::uint64_t null_ptr = 0;
            void* cargs[] = {&gpu.d_proj, &glw.d_attn_post_conv, &null_ptr, &gs.d_hist[2],
                             &gpu.d_attn_out, &hidden_i, &taps_i, &hd_i};
            flyweight_gpu_launch_named("inkling2_convolve",
                                       static_cast<std::uint32_t>((g.hidden + 255) / 256), 1, 256, 0, gpu.stream, cargs);
        }

        // Block residual update
        if (phys % g.block_size == 0) {
            if (n_blocks < GpuState::kMaxBlocks) {
                flyweight_gpu_copy_device(gpu.d_blocks + n_blocks * hbytes, gpu.d_inpSA, hbytes, gpu.stream);
                ++n_blocks;
            }
            flyweight_gpu_copy_device(gpu.d_inpSA, gpu.d_attn_out, hbytes, gpu.stream);
        } else {
            void* add_args[] = {&gpu.d_inpSA, &gpu.d_attn_out, &gpu.d_inpSA, &hidden_i};
            flyweight_gpu_launch_named("inkling2_add",
                                       static_cast<std::uint32_t>((g.hidden + 255) / 256), 1, 256, 0, gpu.stream, add_args);
        }

        // FFN mix + optional loop controller + FFN norm + router
        {
            int nb = static_cast<int>(n_blocks);
            int cols = 1;
            void* margs[] = {&gpu.d_blocks, &nb, &gpu.d_inpSA, &glw.d_ffn_res_q, &cols, &gpu.d_ffn_mixed, &hidden_i};
            flyweight_gpu_launch_named("inkling2_mix", 1, 1, 256, 0, gpu.stream, margs);
        }
        if (has_msg) {
            gpu_control_device(gpu.d_ffn_mixed, gpu.d_msg[3], gpu.d_ffn_mixed, glw, 3);
        }
        launch_norm(gpu.d_ffn_mixed, glw.d_ffn_norm, gpu.d_moe_in);
        qwen_gpu_matvec_by_type(glw.router.type, glw.router.ptr, gpu.d_moe_in, gpu.d_out,
                                hidden_i, static_cast<int>(g.experts), gpu.stream);

        // THE ONLY SYNC PER LAYER: download 128 router logits (512 bytes)
        std::vector<float> logits(g.experts);
        flyweight_gpu_download(logits.data(), gpu.d_out, g.experts * sizeof(float), gpu.stream);
        flyweight_gpu_stream_sync(gpu.stream);

        // Top-k expert selection on CPU (microseconds)
        std::vector<float> selection(g.experts);
        std::vector<std::uint32_t> order(g.experts);
        const float* bias = lw.exp_bias ? reinterpret_cast<const float*>(lw.exp_bias.data) : nullptr;
        for (std::uint32_t e = 0; e < g.experts; ++e) {
            selection[e] = sigmoid(logits[e]) + (bias ? bias[e] : 0.0f);
            order[e] = e;
        }
        const std::uint32_t used = std::min(g.experts_used, g.experts);
        std::partial_sort(order.begin(), order.begin() + used, order.end(),
            [&](std::uint32_t a, std::uint32_t b) {
                return selection[a] != selection[b] ? selection[a] > selection[b] : a < b;
            });
        float weights[32];
        float total = 0.0f;
        for (std::uint32_t k = 0; k < used; ++k) {
            weights[k] = sigmoid(logits[order[k]]);
            total += weights[k];
        }
        const float inv_w = used > 1 ? (1.0f / (total + 1e-20f)) : 1.0f;
        for (std::uint32_t k = 0; k < used; ++k) weights[k] *= inv_w * g.expert_scale;

        // MoE on GPU (hit-first + dual-stream async upload, NO sync at end!)
        auto& pool = gpu.pools[glw.pool_idx];
        const std::size_t ff = g.ff_exp;
        const auto* gu_base = reinterpret_cast<const std::uint8_t*>(lw.gate_up.data);
        const auto* down_base = reinterpret_cast<const std::uint8_t*>(lw.down.data);

        flyweight_gpu_memset(gpu.d_moe_accum, 0, hbytes, gpu.stream);
        std::uint64_t active_keys[32];
        for (std::uint32_t k = 0; k < used; ++k)
            active_keys[k] = static_cast<std::uint64_t>(phys) * g.experts + order[k];

        struct ExpertJob { std::uint32_t k; std::uint32_t slot; std::uint32_t event_idx; };
        ExpertJob hit_jobs[32], miss_jobs[32];
        std::uint32_t n_hits = 0, n_misses = 0;
        const bool async_copy = (gpu.copy_stream != 0 && gpu.h_stage != nullptr &&
                                 gpu.stage_slots >= used && gpu.copy_events.size() >= used);
        if (async_copy && gpu.compute_done_event)
            flyweight_gpu_stream_wait_event(gpu.copy_stream, gpu.compute_done_event);

        for (std::uint32_t k = 0; k < used; ++k) {
            const std::uint64_t key = active_keys[k];
            const std::int32_t cached = pool.slot_of_key[key];
            if (cached >= 0) {
                const auto eslot = static_cast<std::uint32_t>(cached);
                pool.last_used[eslot] = ++pool.lru_tick;
                ++pool.hits;
                hit_jobs[n_hits++] = ExpertJob{k, eslot, 0};
            } else {
                ++pool.misses;
                const std::uint32_t eslot = pool.evict_victim(active_keys, used);
                if (pool.key_of[eslot] != ~std::uint64_t(0))
                    pool.slot_of_key[pool.key_of[eslot]] = -1;
                pool.key_of[eslot] = key;
                pool.slot_of_key[key] = static_cast<std::int32_t>(eslot);
                pool.last_used[eslot] = ++pool.lru_tick;

                const std::uint64_t slot_base =
                    pool.arena + static_cast<std::uint64_t>(eslot) * pool.slot_bytes;
                const auto* expert_gu = gu_base + static_cast<std::size_t>(order[k]) * glw.gu_stride;
                const auto* expert_down = down_base + static_cast<std::size_t>(order[k]) * glw.down_stride;
                const std::uint32_t m_idx = n_misses;
                if (async_copy) {
                    auto* stage = reinterpret_cast<std::uint8_t*>(gpu.h_stage) +
                                  static_cast<std::size_t>(m_idx) * gpu.stage_slot_bytes;
                    std::memcpy(stage, expert_gu, glw.gu_stride);
                    std::memcpy(stage + gpu.stage_gu_bytes, expert_down, glw.down_stride);
                    flyweight_gpu_upload(slot_base, stage, glw.gu_stride, gpu.copy_stream);
                    flyweight_gpu_upload(slot_base + pool.slot_gu_bytes,
                                         stage + gpu.stage_gu_bytes, glw.down_stride, gpu.copy_stream);
                    flyweight_gpu_event_record(gpu.copy_events[m_idx], gpu.copy_stream);
                } else {
                    flyweight_gpu_upload_sync(slot_base, expert_gu, glw.gu_stride);
                    flyweight_gpu_upload_sync(slot_base + pool.slot_gu_bytes, expert_down, glw.down_stride);
                }
                miss_jobs[n_misses++] = ExpertJob{k, eslot, m_idx};
            }
        }

        const std::uint32_t situ_blocks = static_cast<std::uint32_t>((ff + 255) / 256);
        const std::uint32_t add_blocks = static_cast<std::uint32_t>((g.hidden + 255) / 256);
        auto run_job = [&](const ExpertJob& job) {
            const std::uint32_t k = job.k;
            const std::uint64_t slot_base =
                pool.arena + static_cast<std::uint64_t>(job.slot) * pool.slot_bytes;
            float gu_scale = lw.gate_up_scale
                ? reinterpret_cast<const float*>(lw.gate_up_scale.data)[order[k]] : 1.0f;
            const float dn_scale = lw.down_scale
                ? reinterpret_cast<const float*>(lw.down_scale.data)[order[k]] : 1.0f;
            int ff_i = static_cast<int>(ff);
            float beta = g.situ_beta, lin_beta = g.situ_linear_beta;
            void* situ_args[] = {&gpu.d_moe_gu, &gpu.d_moe_act, &ff_i, &beta, &lin_beta, &gu_scale};
            float w_scale = weights[k] * dn_scale;
            void* add_args[] = {&gpu.d_moe_accum, &gpu.d_moe_down, &w_scale, &hidden_i};
            qwen_gpu_matvec_by_type(lw.gate_up.type, slot_base, gpu.d_moe_in, gpu.d_moe_gu,
                                    hidden_i, static_cast<int>(2 * ff), gpu.stream, 1.0f);
            flyweight_gpu_launch_named("inkling2_situ_glu", situ_blocks, 1, 256, 0, gpu.stream, situ_args);
            qwen_gpu_matvec_by_type(lw.down.type, slot_base + pool.slot_gu_bytes, gpu.d_moe_act,
                                    gpu.d_moe_down, ff_i, hidden_i, gpu.stream, 1.0f);
            flyweight_gpu_launch_named("inkling2_scaled_add", add_blocks, 1, 256, 0, gpu.stream, add_args);
        };
        for (std::uint32_t i = 0; i < n_hits; ++i) run_job(hit_jobs[i]);
        for (std::uint32_t i = 0; i < n_misses; ++i) {
            if (async_copy) flyweight_gpu_stream_wait_event(gpu.stream, gpu.copy_events[miss_jobs[i].event_idx]);
            run_job(miss_jobs[i]);
        }
        if (gpu.compute_done_event)
            flyweight_gpu_event_record(gpu.compute_done_event, gpu.stream);

        // Post-FFN conv + residual -> d_hidden (still NO sync!)
        {
            std::uint64_t null_ptr = 0;
            void* cargs[] = {&gpu.d_moe_accum, &glw.d_ffn_post_conv, &null_ptr, &gs.d_hist[3],
                             &gpu.d_ffn_out, &hidden_i, &taps_i, &hd_i};
            flyweight_gpu_launch_named("inkling2_convolve",
                                       static_cast<std::uint32_t>((g.hidden + 255) / 256), 1, 256, 0, gpu.stream, cargs);
            void* add_args[] = {&gpu.d_inpSA, &gpu.d_ffn_out, &gpu.d_hidden, &hidden_i};
            flyweight_gpu_launch_named("inkling2_add", add_blocks, 1, 256, 0, gpu.stream, add_args);
        }
    }

    std::vector<float> step_gpu(std::uint32_t token, bool need_logits) {
        const float pos = position.back();
        const std::size_t hbytes = g.hidden * sizeof(float);
        int hidden_i = static_cast<int>(g.hidden);
        float eps = g.eps;

        std::vector<float> emb(g.hidden);
        const auto stride = row_bytes(embeddings.type, g.hidden);
        row_decode(reinterpret_cast<const std::uint8_t*>(embeddings.data) +
                   static_cast<std::size_t>(token) * stride,
                   embeddings.type, g.hidden, emb.data());
        flyweight_gpu_upload(gpu.d_in, emb.data(), hbytes, gpu.stream);
        {
            void* nargs[] = {&gpu.d_in, &gpu.d_embd_norm, &gpu.d_hidden, &hidden_i, &eps};
            flyweight_gpu_launch_named("inkling2_rms_norm", 1, 1, 256, 0, gpu.stream, nargs);
        }

        std::uint32_t n_blocks = 0;
        const std::uint32_t core = g.core_end - g.core_start;
        for (std::uint32_t il = 0; il < g.core_start; ++il)
            layer_step_gpu(il, il, n_blocks, false, pos);

        flyweight_gpu_copy_device(gpu.d_prelude, gpu.d_hidden, hbytes, gpu.stream);
        if (n_blocks > 0)
            flyweight_gpu_copy_device(gpu.d_prelude_blocks, gpu.d_blocks, n_blocks * hbytes, gpu.stream);
        const std::uint32_t n_prelude_blocks = n_blocks;

        for (std::uint32_t pass = 0; pass < g.num_loops; ++pass) {
            std::uint32_t n_bank = 0;
            if (pass > 0) {
                if (n_blocks > n_prelude_blocks + 1) {
                    n_bank = n_blocks - (n_prelude_blocks + 1);
                    flyweight_gpu_copy_device(gpu.d_bank,
                                              gpu.d_blocks + (n_prelude_blocks + 1) * hbytes,
                                              n_bank * hbytes, gpu.stream);
                }
                flyweight_gpu_copy_device(gpu.d_bank_out, gpu.d_hidden, hbytes, gpu.stream);
                flyweight_gpu_copy_device(gpu.d_hidden, gpu.d_prelude, hbytes, gpu.stream);
                if (n_prelude_blocks > 0)
                    flyweight_gpu_copy_device(gpu.d_blocks, gpu.d_prelude_blocks,
                                              n_prelude_blocks * hbytes, gpu.stream);
                n_blocks = n_prelude_blocks;
            }
            for (std::uint32_t ip = g.core_start; ip < g.core_end; ++ip) {
                const std::uint32_t slot = pass == 0 ? ip
                    : g.physical + (pass - 1) * core + (ip - g.core_start);
                if (pass == 0) {
                    layer_step_gpu(slot, ip, n_blocks, false, pos);
                    continue;
                }
                const std::size_t idx = ip - g.core_start;
                const std::size_t max_pair = (loop_res_query.size() / g.hidden) / 2;
                const std::size_t pair_idx = max_pair ? (idx % max_pair) : 0;
                const std::uint64_t q_ptrs[4] = {
                    gpu.d_loop_res_q + (2 * pair_idx) * hbytes,
                    gpu.d_loop_kv_res_q + (2 * pair_idx) * hbytes,
                    gpu.d_loop_kv_res_q + (2 * pair_idx + 1) * hbytes,
                    gpu.d_loop_res_q + (2 * pair_idx + 1) * hbytes,
                };
                int nb = static_cast<int>(n_bank);
                int cols = 1;
                for (int c = 0; c < 4; ++c) {
                    std::uint64_t qp = q_ptrs[c];
                    void* margs[] = {&gpu.d_bank, &nb, &gpu.d_bank_out, &qp, &cols, &gpu.d_msg[c], &hidden_i};
                    flyweight_gpu_launch_named("inkling2_mix", 1, 1, 256, 0, gpu.stream, margs);
                }
                layer_step_gpu(slot, ip, n_blocks, true, pos);
            }
        }

        for (std::uint32_t il = g.core_end; il < g.physical; ++il)
            layer_step_gpu(il, il, n_blocks, false, pos);

        if (!need_logits) {
            flyweight_gpu_stream_sync(gpu.stream);
            return {};
        }

        {
            int nb = static_cast<int>(n_blocks);
            int cols = 1;
            void* margs[] = {&gpu.d_blocks, &nb, &gpu.d_hidden, &gpu.d_output_res_q, &cols, &gpu.d_ffn_mixed, &hidden_i};
            flyweight_gpu_launch_named("inkling2_mix", 1, 1, 256, 0, gpu.stream, margs);
            void* nargs[] = {&gpu.d_ffn_mixed, &gpu.d_output_norm, &gpu.d_in, &hidden_i, &eps};
            flyweight_gpu_launch_named("inkling2_rms_norm", 1, 1, 256, 0, gpu.stream, nargs);
        }
        qwen_gpu_matvec_by_type(gpu.output.type, gpu.output.ptr, gpu.d_in, gpu.d_out,
                                hidden_i, static_cast<int>(g.vocab), gpu.stream);
        if (g.softcap > 0.0f) {
            int n = static_cast<int>(g.vocab_unpadded);
            float cap = g.softcap;
            void* cap_args[] = {&gpu.d_out, &n, &cap};
            flyweight_gpu_launch_named("inkling2_softcap",
                                       static_cast<std::uint32_t>((n + 255) / 256), 1, 256, 0, gpu.stream, cap_args);
        }
        std::vector<float> logits(g.vocab_unpadded);
        flyweight_gpu_download(logits.data(), gpu.d_out, g.vocab_unpadded * sizeof(float), gpu.stream);
        flyweight_gpu_stream_sync(gpu.stream);
        return logits;
    }

    std::vector<float> step(std::uint32_t token, bool need_logits = true) {
        static const bool profile = [] {
            const char* v = std::getenv("FLYWEIGHT_INKLING2_PROFILE");
            return v && v[0] == '1';
        }();
        const auto t_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const std::uint64_t h0 = gpu.pools[0].hits + gpu.pools[1].hits;
        const std::uint64_t m0 = gpu.pools[0].misses + gpu.pools[1].misses;
        if (!position.empty()) ++position.back();
        else position.push_back(0);
        if (gpu.active && gpu.full_decode_ready && gpu.moe_enabled) {
            auto out = step_gpu(token, need_logits);
            if (profile && need_logits) {
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_start).count();
                const std::uint64_t dh = (gpu.pools[0].hits + gpu.pools[1].hits) - h0;
                const std::uint64_t dm = (gpu.pools[0].misses + gpu.pools[1].misses) - m0;
                std::fprintf(stderr,
                    "[inkling2 prof] tok=%.1f ms  hits=%llu misses=%llu (%.0f%% hit) [full-gpu]\n",
                    ms, static_cast<unsigned long long>(dh), static_cast<unsigned long long>(dm),
                    (dh + dm) ? (100.0 * static_cast<double>(dh) / static_cast<double>(dh + dm)) : 0.0);
            }
            return out;
        }
        blocks.clear();
        std::vector<float> hidden(g.hidden);
        const auto stride = row_bytes(embeddings.type, g.hidden);
        row_decode(reinterpret_cast<const std::uint8_t*>(embeddings.data) +
                   static_cast<std::size_t>(token) * stride,
                   embeddings.type, g.hidden, hidden.data());
        {
            const float* norm = floats("token_embd_norm.weight");
            std::vector<float> tmp(g.hidden);
            rms_norm_into(hidden.data(), norm, g.hidden, g.eps, tmp.data());
            hidden.swap(tmp);
        }

        const std::uint32_t core = g.core_end - g.core_start;
        for (std::uint32_t il = 0; il < g.core_start; ++il) {
            const std::uint32_t phys = il;
            hidden = layer_step(il, phys, hidden, nullptr, 0);
        }
        prelude = hidden;
        prelude_blocks = blocks;

        for (std::uint32_t pass = 0; pass < g.num_loops; ++pass) {
            std::vector<std::vector<float>> bank;
            std::vector<float> bank_out;
            if (pass > 0) {
                if (blocks.size() > prelude_blocks.size() + 1)
                    bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = hidden;
                hidden = prelude;
                blocks = prelude_blocks;
            }
            for (std::uint32_t ip = g.core_start; ip < g.core_end; ++ip) {
                const std::uint32_t slot = pass == 0 ? ip
                    : g.physical + (pass - 1) * core + (ip - g.core_start);
                if (pass == 0) {
                    hidden = layer_step(slot, ip, hidden, nullptr, pass);
                    continue;
                }
                const std::size_t idx = ip - g.core_start;
                const std::size_t max_pair = (loop_res_query.size() / g.hidden) / 2;
                const std::size_t pair_idx = max_pair ? (idx % max_pair) : 0;
                std::vector<std::vector<float>> messages(4);
                messages[0] = mix(bank, bank_out, loop_res_query.data() + (2 * pair_idx) * g.hidden, 1);
                messages[1] = mix(bank, bank_out, loop_kv_res_query.data() + (2 * pair_idx) * g.hidden, 1);
                messages[2] = mix(bank, bank_out, loop_kv_res_query.data() + (2 * pair_idx + 1) * g.hidden, 1);
                messages[3] = mix(bank, bank_out, loop_res_query.data() + (2 * pair_idx + 1) * g.hidden, 1);
                hidden = layer_step(slot, ip, hidden, &messages, pass);
            }
        }
        for (std::uint32_t il = g.core_end; il < g.physical; ++il)
            hidden = layer_step(il, il, hidden, nullptr, 0);

        if (!need_logits) return {};
        auto out = compute_logits(blocks, hidden);
        if (profile) {
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_start).count();
            const std::uint64_t dh = (gpu.pools[0].hits + gpu.pools[1].hits) - h0;
            const std::uint64_t dm = (gpu.pools[0].misses + gpu.pools[1].misses) - m0;
            std::fprintf(stderr,
                "[inkling2 prof] tok=%.1f ms  hits=%llu misses=%llu (%.0f%% hit)\n",
                ms, static_cast<unsigned long long>(dh), static_cast<unsigned long long>(dm),
                (dh + dm) ? (100.0 * static_cast<double>(dh) / static_cast<double>(dh + dm)) : 0.0);
        }
        return out;
    }

    // --- Batched GPU prefill -------------------------------------------------

    std::vector<float> control_batch(const std::vector<float>& cur,
                                     const std::vector<float>& message,
                                     std::size_t T, const LayerWeights& lw,
                                     std::uint32_t phys, int c) const {
        const GpuLayerWeights* glw = gpu.active ? &gpu.layers[phys] : nullptr;
        std::vector<float> pair(T * 2 * g.hidden);
        std::vector<float> gate(T * g.gate_rank);
        std::vector<float> gain(T * g.hidden);
        std::vector<float> forget(T * g.hidden);
        for (std::size_t t = 0; t < T; ++t) {
            const float* ct = cur.data() + t * g.hidden;
            const float* mt = message.data() + t * g.hidden;
            float* pt = pair.data() + t * 2 * g.hidden;
            rms_norm_unit(ct, g.hidden, g.eps, pt);
            rms_norm_unit(mt, g.hidden, g.eps, pt + g.hidden);
        }
        run_gemm(glw ? glw->loop_down[c] : GpuMatrix{}, lw.loop_down[c],
                 pair.data(), T, 2 * g.hidden, g.gate_rank, gate.data());
        run_gemm(glw ? glw->loop_gain[c] : GpuMatrix{}, lw.loop_gain[c],
                 gate.data(), T, g.gate_rank, g.hidden, gain.data());
        run_gemm(glw ? glw->loop_forget[c] : GpuMatrix{}, lw.loop_forget[c],
                 gate.data(), T, g.gate_rank, g.hidden, forget.data());
        std::vector<float> out(T * g.hidden);
        for (std::size_t i = 0; i < T * g.hidden; ++i)
            out[i] = (1.0f - std::tanh(forget[i])) * cur[i] + std::tanh(gain[i]) * message[i];
        return out;
    }

    // Grouped-by-expert MoE over a chunk of T tokens: each active expert is
    // loaded into the GPU cache once and evaluated across all tokens routed to it.
    std::vector<float> moe_batch(const std::vector<float>& x, std::size_t T,
                                 const LayerWeights& lw, std::uint32_t phys) const {
        const GpuLayerWeights* glw = gpu.active ? &gpu.layers[phys] : nullptr;
        std::vector<float> router_logits(T * g.experts);
        run_gemm(glw ? glw->router : GpuMatrix{}, lw.router,
                 x.data(), T, g.hidden, g.experts, router_logits.data());

        const std::uint32_t used = std::min(g.experts_used, g.experts);
        const float* bias = lw.exp_bias ? reinterpret_cast<const float*>(lw.exp_bias.data) : nullptr;
        struct Route { std::uint32_t tok; float weight; };
        std::vector<std::vector<Route>> buckets(g.experts);
        std::vector<float> selection(g.experts);
        std::vector<std::uint32_t> order(g.experts);
        for (std::size_t t = 0; t < T; ++t) {
            const float* lg = router_logits.data() + t * g.experts;
            for (std::uint32_t e = 0; e < g.experts; ++e) {
                selection[e] = sigmoid(lg[e]) + (bias ? bias[e] : 0.0f);
                order[e] = e;
            }
            std::partial_sort(order.begin(), order.begin() + used, order.end(),
                [&](std::uint32_t a, std::uint32_t b) {
                    return selection[a] != selection[b] ? selection[a] > selection[b] : a < b;
                });
            float total = 0.0f;
            for (std::uint32_t k = 0; k < used; ++k) total += sigmoid(lg[order[k]]);
            const float inv = used > 1 ? (1.0f / (total + 1e-20f)) : 1.0f;
            for (std::uint32_t k = 0; k < used; ++k) {
                const float w = sigmoid(lg[order[k]]) * inv * g.expert_scale;
                buckets[order[k]].push_back(Route{static_cast<std::uint32_t>(t), w});
            }
        }

        const std::size_t ff = g.ff_exp;
        const auto* gu_base = reinterpret_cast<const std::uint8_t*>(lw.gate_up.data);
        const auto* down_base = reinterpret_cast<const std::uint8_t*>(lw.down.data);
        std::vector<float> result(T * g.hidden, 0.0f);

        if (gpu.active && gpu.prefill_ready && gpu.moe_enabled && glw && glw->moe_supported) {
            auto& pool = gpu.pools[glw->pool_idx];
            std::vector<float> gather_in, down_host;
            for (std::uint32_t e = 0; e < g.experts; ++e) {
                const auto& routes = buckets[e];
                if (routes.empty()) continue;
                const std::size_t B = routes.size();
                const std::uint64_t key = static_cast<std::uint64_t>(phys) * g.experts + e;
                std::uint32_t slot = 0;
                const std::int32_t cached = pool.slot_of_key[key];
                bool ok = true;
                if (cached >= 0) {
                    slot = static_cast<std::uint32_t>(cached);
                    pool.last_used[slot] = ++pool.lru_tick;
                    ++pool.hits;
                } else {
                    ++pool.misses;
                    slot = pool.evict_victim(nullptr, 0);
                    if (pool.key_of[slot] != ~std::uint64_t(0))
                        pool.slot_of_key[pool.key_of[slot]] = -1;
                    pool.key_of[slot] = key;
                    pool.slot_of_key[key] = static_cast<std::int32_t>(slot);
                    pool.last_used[slot] = ++pool.lru_tick;

                    const std::uint64_t slot_base =
                        pool.arena + static_cast<std::uint64_t>(slot) * pool.slot_bytes;
                    const auto* expert_gu = gu_base + static_cast<std::size_t>(e) * glw->gu_stride;
                    const auto* expert_down = down_base + static_cast<std::size_t>(e) * glw->down_stride;
                    if (gpu.h_stage && gpu.stage_slots > 0) {
                        auto* stage = reinterpret_cast<std::uint8_t*>(gpu.h_stage);
                        std::memcpy(stage, expert_gu, glw->gu_stride);
                        std::memcpy(stage + gpu.stage_gu_bytes, expert_down, glw->down_stride);
                        ok = (flyweight_gpu_upload(slot_base, stage, glw->gu_stride, gpu.stream) == 0 &&
                              flyweight_gpu_upload(slot_base + pool.slot_gu_bytes,
                                                   stage + gpu.stage_gu_bytes,
                                                   glw->down_stride, gpu.stream) == 0);
                    } else {
                        ok = (flyweight_gpu_upload_sync(slot_base, expert_gu, glw->gu_stride) == 0 &&
                              flyweight_gpu_upload_sync(slot_base + pool.slot_gu_bytes,
                                                        expert_down, glw->down_stride) == 0);
                    }
                }
                if (!ok) {
                    // Fall back to CPU for this expert if upload failed.
                    const auto gu_k = row_kernel(lw.gate_up.type, g.hidden);
                    const auto dn_k = row_kernel(lw.down.type, ff);
                    const auto* expert_gu = gu_base + static_cast<std::size_t>(e) * glw->gu_stride;
                    const auto* expert_down = down_base + static_cast<std::size_t>(e) * glw->down_stride;
                    std::vector<float> gate(ff), up(ff), act(ff), down(g.hidden);
                    for (const auto& rt : routes) {
                        const float* xt = x.data() + rt.tok * g.hidden;
                        for (std::size_t r = 0; r < ff; ++r) {
                            gate[r] = gu_k(expert_gu, xt, g.hidden, r);
                            up[r] = gu_k(expert_gu, xt, g.hidden, ff + r);
                        }
                        situ_glu_into(gate.data(), up.data(), ff, g.situ_beta, g.situ_linear_beta, act.data());
                        for (std::size_t r = 0; r < g.hidden; ++r)
                            down[r] = dn_k(expert_down, act.data(), ff, r);
                        float* dst = result.data() + rt.tok * g.hidden;
                        for (std::size_t r = 0; r < g.hidden; ++r) dst[r] += rt.weight * down[r];
                    }
                    continue;
                }

                gather_in.resize(B * g.hidden);
                for (std::size_t b = 0; b < B; ++b)
                    std::memcpy(gather_in.data() + b * g.hidden,
                                x.data() + routes[b].tok * g.hidden,
                                g.hidden * sizeof(float));
                down_host.resize(B * g.hidden);

                const std::uint64_t slot_base =
                    pool.arena + static_cast<std::uint64_t>(slot) * pool.slot_bytes;
                const std::uint64_t slot_gu = slot_base;
                const std::uint64_t slot_down = slot_base + pool.slot_gu_bytes;

                float gu_scale = lw.gate_up_scale
                    ? reinterpret_cast<const float*>(lw.gate_up_scale.data)[e] : 1.0f;
                const float dn_scale = lw.down_scale
                    ? reinterpret_cast<const float*>(lw.down_scale.data)[e] : 1.0f;

                int rows_i = static_cast<int>(B);
                int ff_i = static_cast<int>(ff);
                float beta = g.situ_beta;
                float lin_beta = g.situ_linear_beta;
                void* situ_args[] = {&gpu.d_pf_out, &gpu.d_pf_act, &rows_i, &ff_i, &beta, &lin_beta, &gu_scale};

                ok = (flyweight_gpu_upload(gpu.d_pf_in, gather_in.data(),
                                           B * g.hidden * sizeof(float), gpu.stream) == 0 &&
                      gpu_gemm(lw.gate_up.type, slot_gu, gpu.d_pf_in, gpu.d_pf_out,
                               rows_i, static_cast<int>(g.hidden), static_cast<int>(2 * ff), 1.0f) &&
                      flyweight_gpu_launch_named("inkling2_situ_glu_rows",
                                                 static_cast<std::uint32_t>((ff + 255) / 256),
                                                 static_cast<std::uint32_t>(B), 256, 0,
                                                 gpu.stream, situ_args) == 0 &&
                      gpu_gemm(lw.down.type, slot_down, gpu.d_pf_act, gpu.d_pf_out,
                               rows_i, ff_i, static_cast<int>(g.hidden), 1.0f) &&
                      flyweight_gpu_download(down_host.data(), gpu.d_pf_out,
                                             B * g.hidden * sizeof(float), gpu.stream) == 0 &&
                      flyweight_gpu_stream_sync(gpu.stream) == 0);
                if (ok) {
                    for (std::size_t b = 0; b < B; ++b) {
                        float* dst = result.data() + routes[b].tok * g.hidden;
                        const float* src = down_host.data() + b * g.hidden;
                        const float w = routes[b].weight * dn_scale;
                        for (std::size_t r = 0; r < g.hidden; ++r) dst[r] += w * src[r];
                    }
                }
            }
            return result;
        }

        // CPU fallback
        const auto gu_k = row_kernel(lw.gate_up.type, g.hidden);
        const auto dn_k = row_kernel(lw.down.type, ff);
        const std::size_t gu_stride = 2 * ff * row_bytes(lw.gate_up.type, g.hidden);
        const std::size_t down_stride = g.hidden * row_bytes(lw.down.type, ff);
        std::vector<float> gate(ff), up(ff), act(ff), down(g.hidden);
        for (std::uint32_t e = 0; e < g.experts; ++e) {
            const auto& routes = buckets[e];
            if (routes.empty()) continue;
            const auto* expert_gu = gu_base + static_cast<std::size_t>(e) * gu_stride;
            const auto* expert_down = down_base + static_cast<std::size_t>(e) * down_stride;
            for (const auto& rt : routes) {
                const float* xt = x.data() + rt.tok * g.hidden;
                for (std::size_t r = 0; r < ff; ++r) {
                    gate[r] = gu_k(expert_gu, xt, g.hidden, r);
                    up[r] = gu_k(expert_gu, xt, g.hidden, ff + r);
                }
                const float gu_scale = lw.gate_up_scale
                    ? reinterpret_cast<const float*>(lw.gate_up_scale.data)[e] : 1.0f;
                const float dn_scale = lw.down_scale
                    ? reinterpret_cast<const float*>(lw.down_scale.data)[e] : 1.0f;
                situ_glu_into(gate.data(), up.data(), ff, g.situ_beta, g.situ_linear_beta, act.data(), gu_scale);
                for (std::size_t r = 0; r < g.hidden; ++r)
                    down[r] = dn_k(expert_down, act.data(), ff, r);
                float* dst = result.data() + rt.tok * g.hidden;
                const float w = rt.weight * dn_scale;
                for (std::size_t r = 0; r < g.hidden; ++r) dst[r] += w * down[r];
            }
        }
        return result;
    }

    void layer_batch(std::uint32_t slot, std::uint32_t phys, std::size_t T,
                     std::vector<float>& H,
                     std::vector<std::vector<std::vector<float>>>& blocks_batch,
                     const std::array<std::vector<float>, 4>* messages,
                     const std::vector<float>& tok_pos) {
        const LayerWeights& lw = layers[phys];
        const GpuLayerWeights* glw = gpu.active ? &gpu.layers[phys] : nullptr;
        CacheSlot& cache = slots[slot];
        const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
        const std::uint32_t heads = g.heads_by_layer.size() > phys ? g.heads_by_layer[phys] : g.heads;
        const std::uint32_t kvh = g.kv_heads_by_layer.size() > phys ? g.kv_heads_by_layer[phys] : g.kv_heads;
        const std::size_t q_dim = static_cast<std::size_t>(heads) * g.head_dim;
        const std::size_t kv_dim = static_cast<std::size_t>(kvh) * g.head_dim;
        const std::size_t r_dim = static_cast<std::size_t>(heads) * g.rel_len;

        std::vector<float> inpSA = H;
        std::vector<float> mixed[3];
        mixed[0].resize(T * g.hidden);
        if (lw.has_attn_res) {
            const float* qsrc = reinterpret_cast<const float*>(lw.attn_res_q.data);
            std::vector<float> cur_tok(g.hidden);
            if (lw.attn_res_cols >= 3) {
                mixed[1].resize(T * g.hidden);
                mixed[2].resize(T * g.hidden);
                for (std::size_t t = 0; t < T; ++t) {
                    std::memcpy(cur_tok.data(), H.data() + t * g.hidden, g.hidden * sizeof(float));
                    auto m = mix(blocks_batch[t], cur_tok, qsrc, 3);
                    for (int c = 0; c < 3; ++c)
                        std::memcpy(mixed[c].data() + t * g.hidden, m.data() + c * g.hidden, g.hidden * sizeof(float));
                }
            } else {
                for (std::size_t t = 0; t < T; ++t) {
                    std::memcpy(cur_tok.data(), H.data() + t * g.hidden, g.hidden * sizeof(float));
                    auto m = mix(blocks_batch[t], cur_tok, qsrc, 1);
                    std::memcpy(mixed[0].data() + t * g.hidden, m.data(), g.hidden * sizeof(float));
                }
                mixed[1] = mixed[0];
                mixed[2] = mixed[0];
            }
        } else {
            mixed[0] = H;
            mixed[1] = mixed[0];
            mixed[2] = mixed[0];
        }

        if (messages) {
            for (int c = 0; c < 3; ++c)
                mixed[c] = control_batch(mixed[c], (*messages)[c], T, lw, phys, c);
        }

        const float* attn_norm = lw.attn_norm ? lw.attn_norm_f32() : nullptr;
        bool shared_mixed = false;
        if (lw.is_swa && messages) {
            if (!attn_norm) throw std::runtime_error("inkling2: missing attn_norm on a sliding layer");
            for (int c = 0; c < 3; ++c)
                for (std::size_t t = 0; t < T; ++t)
                    rms_norm_into(mixed[c].data() + t * g.hidden, attn_norm, g.hidden, g.eps,
                                  mixed[c].data() + t * g.hidden);
        } else if (lw.is_swa) {
            if (!attn_norm) throw std::runtime_error("inkling2: missing attn_norm on a sliding layer");
            for (std::size_t t = 0; t < T; ++t)
                rms_norm_into(mixed[0].data() + t * g.hidden, attn_norm, g.hidden, g.eps,
                              mixed[0].data() + t * g.hidden);
            mixed[1] = mixed[0];
            mixed[2] = mixed[0];
            shared_mixed = true;
        } else {
            const float* qn = lw.q_norm ? lw.q_norm_f32() : attn_norm;
            const float* kn = lw.k_norm ? lw.k_norm_f32() : attn_norm;
            const float* vn = lw.v_norm ? lw.v_norm_f32() : attn_norm;
            if (!qn || !kn || !vn)
                throw std::runtime_error("inkling2: missing input norms on a full-attention layer");
            for (std::size_t t = 0; t < T; ++t) {
                rms_norm_into(mixed[0].data() + t * g.hidden, qn, g.hidden, g.eps,
                              mixed[0].data() + t * g.hidden);
                rms_norm_into(mixed[1].data() + t * g.hidden, kn, g.hidden, g.eps,
                              mixed[1].data() + t * g.hidden);
                rms_norm_into(mixed[2].data() + t * g.hidden, vn, g.hidden, g.eps,
                              mixed[2].data() + t * g.hidden);
            }
        }
        (void)shared_mixed;

        // Batched GPU projections for all T tokens.
        std::vector<float> Q(T * q_dim), K(T * kv_dim), V(T * kv_dim),
                           R(T * r_dim), Sinks(T * heads), Gate(T * q_dim);
        run_gemm(glw ? glw->wq : GpuMatrix{}, lw.wq, mixed[0].data(), T, g.hidden, q_dim, Q.data());
        run_gemm(glw ? glw->wk : GpuMatrix{}, lw.wk, mixed[1].data(), T, g.hidden, kv_dim, K.data());
        run_gemm(glw ? glw->wv : GpuMatrix{}, lw.wv, mixed[2].data(), T, g.hidden, kv_dim, V.data());
        run_gemm(glw ? glw->rel : GpuMatrix{}, lw.rel, mixed[0].data(), T, g.hidden, r_dim, R.data());
        run_gemm(glw ? glw->sink_proj : GpuMatrix{}, lw.sink_proj, mixed[0].data(), T, g.hidden, heads, Sinks.data());
        run_gemm(glw ? glw->wgate : GpuMatrix{}, lw.wgate, mixed[0].data(), T, g.hidden, q_dim, Gate.data());

        // Sequential per-token causal conv, clip, KV update, and relative-bias attention.
        const Matrix* kdyn = lw.k_conv_proj ? &lw.k_conv_proj : nullptr;
        const Matrix* vdyn = lw.v_conv_proj ? &lw.v_conv_proj : nullptr;
        const std::uint32_t extent = lw.is_swa ? g.sliding_window : g.rel_extent;
        if (lw.rel_bias.type != 0)
            throw std::runtime_error("inkling2: expected an f32 relative-bias table");
        const float* table = reinterpret_cast<const float*>(lw.rel_bias.data);
        const float scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));

        std::vector<float> Attended(T * q_dim, 0.0f);
        std::vector<float> k_tok(kv_dim), v_tok(kv_dim);
        std::vector<float> bias((extent + 1) * heads, 0.0f);

        for (std::size_t t = 0; t < T; ++t) {
            float* qt = Q.data() + t * q_dim;
            float* rt = R.data() + t * r_dim;
            const float* st = Sinks.data() + t * heads;
            std::memcpy(k_tok.data(), K.data() + t * kv_dim, kv_dim * sizeof(float));
            std::memcpy(v_tok.data(), V.data() + t * kv_dim, kv_dim * sizeof(float));

            k_tok = convolve(k_tok, lw.k_conv, kdyn, glw ? glw->k_conv_proj : GpuMatrix{},
                             mixed[1].data() + t * g.hidden, cache.history[0], kv_dim);
            v_tok = convolve(v_tok, lw.v_conv, vdyn, glw ? glw->v_conv_proj : GpuMatrix{},
                             mixed[2].data() + t * g.hidden, cache.history[1], kv_dim);

            clip_rows(qt, heads, g.head_dim, reinterpret_cast<const float*>(lw.q_clip.data), g.qk_clip);
            clip_rows(k_tok.data(), kvh, g.head_dim, nullptr, g.qk_clip);
            for (std::uint32_t h = 0; h < kvh; ++h)
                rms_norm_unit(v_tok.data() + h * g.head_dim,
                              g.head_dim, g.eps, v_tok.data() + h * g.head_dim);
            clip_rows(rt, heads, g.rel_len, reinterpret_cast<const float*>(lw.rel_clip.data), g.rel_clip);

            const float pos = tok_pos[t];
            const float tau = 1.0f + g.log_scale_alpha *
                std::log(std::max(1.0f, (pos + 1.0f) / static_cast<float>(g.log_scale_floor)));
            if (!lw.is_swa) {
                for (std::size_t i = 0; i < q_dim; ++i) qt[i] *= tau;
                for (std::size_t i = 0; i < r_dim; ++i) rt[i] *= tau;
            }

            const std::uint32_t write = static_cast<std::uint32_t>(cache.length);
            std::copy(k_tok.begin(), k_tok.end(), cache.keys.begin() + write * kv_channels);
            std::copy(v_tok.begin(), v_tok.end(), cache.values.begin() + write * kv_channels);
            cache.positions[write] = pos;
            ++cache.length;

            const std::uint32_t active_extent = std::min(extent, static_cast<std::uint32_t>(pos) + 1u);
            for (std::uint32_t d = 0; d < active_extent; ++d)
                for (std::uint32_t h = 0; h < heads; ++h) {
                    float dot = 0.0f;
                    for (std::uint32_t r = 0; r < g.rel_len; ++r)
                        dot += table[static_cast<std::size_t>(d) * g.rel_len + r] *
                               rt[static_cast<std::size_t>(h) * g.rel_len + r];
                    bias[d * heads + h] = dot - st[h];
                }
            for (std::uint32_t h = 0; h < heads; ++h)
                bias[active_extent * heads + h] = -st[h];

            float* att_t = Attended.data() + t * q_dim;
            std::vector<float> scores(cache.length + 1, 0.0f);
            for (std::uint32_t h = 0; h < heads; ++h) {
                const std::uint32_t kv_head = h / std::max(1u, heads / kvh);
                const float* q = qt + h * g.head_dim;
                for (std::uint32_t j = 0; j < cache.length; ++j) {
                    float dot = 0.0f;
                    const float* k = cache.keys.data() + j * kv_channels + kv_head * g.head_dim;
                    for (std::uint32_t i = 0; i < g.head_dim; ++i) dot += q[i] * k[i];
                    const float d = pos - cache.positions[j];
                    const std::uint32_t idx = (d < 0 || d >= active_extent) ? active_extent : static_cast<std::uint32_t>(d);
                    scores[j] = dot * scale + bias[idx * heads + h];
                }
                scores[cache.length] = 0.0f;
                softmax_into(scores.data(), scores.size());
                for (std::uint32_t j = 0; j < cache.length; ++j) {
                    const float* v = cache.values.data() + j * kv_channels + kv_head * g.head_dim;
                    const float w = scores[j];
                    for (std::uint32_t i = 0; i < g.head_dim; ++i)
                        att_t[h * g.head_dim + i] += w * v[i];
                }
            }
            const float* gt = Gate.data() + t * q_dim;
            for (std::size_t i = 0; i < q_dim; ++i) att_t[i] *= sigmoid(gt[i]);
        }

        // Batched output projection.
        std::vector<float> Proj(T * g.hidden);
        run_gemm(glw ? glw->wo : GpuMatrix{}, lw.wo, Attended.data(), T, q_dim, g.hidden, Proj.data());

        // Post-attention conv + block residual + FFN input prep.
        std::vector<float> ffn_mixed(T * g.hidden);
        const float* ffn_q = reinterpret_cast<const float*>(lw.ffn_res_q.data);
        std::vector<float> proj_tok(g.hidden), inp_tok(g.hidden);
        for (std::size_t t = 0; t < T; ++t) {
            std::memcpy(proj_tok.data(), Proj.data() + t * g.hidden, g.hidden * sizeof(float));
            auto attn_out = convolve(proj_tok, lw.attn_post_conv, nullptr, GpuMatrix{}, nullptr,
                                     cache.history[2], g.hidden);
            float* sa = inpSA.data() + t * g.hidden;
            if (phys % g.block_size == 0) {
                blocks_batch[t].emplace_back(sa, sa + g.hidden);
                std::memcpy(sa, attn_out.data(), g.hidden * sizeof(float));
            } else {
                for (std::size_t i = 0; i < g.hidden; ++i) sa[i] += attn_out[i];
            }
            std::memcpy(inp_tok.data(), sa, g.hidden * sizeof(float));
            auto fm = mix(blocks_batch[t], inp_tok, ffn_q, 1);
            std::memcpy(ffn_mixed.data() + t * g.hidden, fm.data(), g.hidden * sizeof(float));
        }

        if (messages) ffn_mixed = control_batch(ffn_mixed, (*messages)[3], T, lw, phys, 3);

        std::vector<float> normed(T * g.hidden);
        for (std::size_t t = 0; t < T; ++t)
            rms_norm_into(ffn_mixed.data() + t * g.hidden, lw.ffn_norm_f32(), g.hidden, g.eps,
                          normed.data() + t * g.hidden);

        // Expert-grouped batched GPU MoE.
        auto ffn_out = moe_batch(normed, T, lw, phys);

        // Post-FFN conv + residual.
        const std::size_t old_len = cache.length - T;
        std::vector<float> ffn_tok(g.hidden);
        for (std::size_t t = 0; t < T; ++t) {
            std::memcpy(ffn_tok.data(), ffn_out.data() + t * g.hidden, g.hidden * sizeof(float));
            ffn_tok = convolve(ffn_tok, lw.ffn_post_conv, nullptr, GpuMatrix{}, nullptr,
                               cache.history[3], g.hidden);
            const float* sa = inpSA.data() + t * g.hidden;
            float* dst = H.data() + t * g.hidden;
            for (std::size_t i = 0; i < g.hidden; ++i) dst[i] = sa[i] + ffn_tok[i];
        }
        sync_cpu_slot_to_gpu(slot, old_len);
    }

    std::vector<float> prefill_chunk(const std::uint32_t* tokens, std::size_t T,
                                     bool need_logits) {
        sync_gpu_slots_to_cpu();
        std::vector<float> tok_pos(T);
        const float start_pos = position.empty() ? 0.0f : (position.back() + 1.0f);
        for (std::size_t t = 0; t < T; ++t) tok_pos[t] = start_pos + static_cast<float>(t);
        if (position.empty()) position.push_back(tok_pos.back());
        else position.back() = tok_pos.back();

        std::vector<float> H(T * g.hidden);
        const auto stride = row_bytes(embeddings.type, g.hidden);
        const float* emb_norm = floats("token_embd_norm.weight");
        std::vector<float> tmp(g.hidden);
        for (std::size_t t = 0; t < T; ++t) {
            row_decode(reinterpret_cast<const std::uint8_t*>(embeddings.data) +
                       static_cast<std::size_t>(tokens[t]) * stride,
                       embeddings.type, g.hidden, tmp.data());
            rms_norm_into(tmp.data(), emb_norm, g.hidden, g.eps, H.data() + t * g.hidden);
        }

        std::vector<std::vector<std::vector<float>>> blocks_batch(T);
        const std::uint32_t core = g.core_end - g.core_start;
        for (std::uint32_t il = 0; il < g.core_start; ++il)
            layer_batch(il, il, T, H, blocks_batch, nullptr, tok_pos);

        std::vector<float> prelude_H = H;
        auto prelude_blocks_batch = blocks_batch;

        for (std::uint32_t pass = 0; pass < g.num_loops; ++pass) {
            std::vector<std::vector<std::vector<float>>> banks(T);
            std::vector<std::vector<float>> bank_outs(T);
            if (pass > 0) {
                for (std::size_t t = 0; t < T; ++t) {
                    if (blocks_batch[t].size() > prelude_blocks_batch[t].size() + 1)
                        banks[t].assign(blocks_batch[t].begin() + prelude_blocks_batch[t].size() + 1,
                                        blocks_batch[t].end());
                    bank_outs[t].assign(H.data() + t * g.hidden, H.data() + (t + 1) * g.hidden);
                }
                H = prelude_H;
                blocks_batch = prelude_blocks_batch;
            }
            for (std::uint32_t ip = g.core_start; ip < g.core_end; ++ip) {
                const std::uint32_t slot = pass == 0 ? ip
                    : g.physical + (pass - 1) * core + (ip - g.core_start);
                if (pass == 0) {
                    layer_batch(slot, ip, T, H, blocks_batch, nullptr, tok_pos);
                    continue;
                }
                const std::size_t idx = ip - g.core_start;
                const std::size_t max_pair = (loop_res_query.size() / g.hidden) / 2;
                const std::size_t pair_idx = max_pair ? (idx % max_pair) : 0;
                std::array<std::vector<float>, 4> messages;
                for (int c = 0; c < 4; ++c) messages[c].resize(T * g.hidden);
                const float* q_ptrs[4] = {
                    loop_res_query.data() + (2 * pair_idx) * g.hidden,
                    loop_kv_res_query.data() + (2 * pair_idx) * g.hidden,
                    loop_kv_res_query.data() + (2 * pair_idx + 1) * g.hidden,
                    loop_res_query.data() + (2 * pair_idx + 1) * g.hidden,
                };
                for (std::size_t t = 0; t < T; ++t) {
                    for (int c = 0; c < 4; ++c) {
                        auto m = mix(banks[t], bank_outs[t], q_ptrs[c], 1);
                        std::memcpy(messages[c].data() + t * g.hidden, m.data(), g.hidden * sizeof(float));
                    }
                }
                layer_batch(slot, ip, T, H, blocks_batch, &messages, tok_pos);
            }
        }

        for (std::uint32_t il = g.core_end; il < g.physical; ++il)
            layer_batch(il, il, T, H, blocks_batch, nullptr, tok_pos);

        if (!need_logits) return {};
        std::vector<float> last_hidden(H.begin() + (T - 1) * g.hidden, H.end());
        return compute_logits(blocks_batch[T - 1], last_hidden);
    }
};

} // namespace flyweight::v2::inkling2

extern "C" {

struct FlyweightV2Inkling2Runtime {
    flyweight::v2::inkling2::Runtime impl;
};

int flyweight_v2_inkling2_create(const FlyweightV2Model* model, std::uint32_t context_limit,
                                 FlyweightV2Inkling2Runtime** out) {
    return guarded([&] {
        if (!model || !out) throw std::runtime_error("invalid inkling2 runtime arguments");
        if (model->config.architecture != "inkling2")
            throw std::runtime_error("model is not an inkling2 checkpoint");
        auto runtime = std::make_unique<FlyweightV2Inkling2Runtime>();
        runtime->impl.init(model, context_limit);
        *out = runtime.release();
        return 0;
    });
}

void flyweight_v2_inkling2_destroy(FlyweightV2Inkling2Runtime* runtime) { delete runtime; }

int flyweight_v2_inkling2_reset(FlyweightV2Inkling2Runtime* runtime) {
    return guarded([&] {
        if (!runtime) throw std::runtime_error("invalid inkling2 runtime");
        runtime->impl.reset();
        return 0;
    });
}

int flyweight_v2_inkling2_position(FlyweightV2Inkling2Runtime* runtime, std::uint32_t* out) {
    return guarded([&] {
        if (!runtime || !out) throw std::runtime_error("invalid inkling2 runtime arguments");
        *out = runtime->impl.position.empty() ? 0 : runtime->impl.position.back();
        return 0;
    });
}

int flyweight_v2_inkling2_eval(FlyweightV2Inkling2Runtime* runtime,
                               const std::uint32_t* tokens, std::uint32_t count,
                               float* logits) {
    return guarded([&] {
        if (!runtime || !tokens || !count)
            throw std::runtime_error("invalid inkling2 eval arguments");
        const std::size_t need = runtime->impl.g.vocab_unpadded;
        std::vector<float> last;
        if (count > 1 && runtime->impl.gpu.active && runtime->impl.gpu.prefill_ready) {
            constexpr std::uint32_t chunk = flyweight::v2::inkling2::GpuState::kPrefillChunk;
            for (std::uint32_t offset = 0; offset < count; offset += chunk) {
                const std::uint32_t n = std::min(chunk, count - offset);
                const bool want = (logits != nullptr && offset + n == count);
                if (n == 1) last = runtime->impl.step(tokens[offset], want);
                else last = runtime->impl.prefill_chunk(tokens + offset, n, want);
            }
        } else {
            for (std::uint32_t i = 0; i < count; ++i) {
                const bool want = (logits != nullptr && i + 1 == count);
                last = runtime->impl.step(tokens[i], want);
            }
        }
        if (logits) {
            if (last.size() < need) throw std::runtime_error("inkling2 logits buffer is too small");
            std::copy(last.begin(), last.end(), logits);
        }
        return 0;
    });
}

int flyweight_v2_inkling2_uses_gpu(const FlyweightV2Inkling2Runtime* runtime) {
    return (runtime && runtime->impl.gpu.active) ? 1 : 0;
}

} // extern "C"
