// Inkling 2 execution path. Included from v2_runtime.cpp after the model and
// tensor accessors are defined.
//
// The design follows llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp
// but deliberately uses flyweight's existing quantized row kernels: the
// checkpoint is 8.5 GB of IQ2_XXS/IQ4_XS weights, so dequantizing it to f32
// would need tens of gigabytes. Prefill is a sequence of single-token steps;
// slower than a fused batched kernel, but it keeps one state machine for both
// prefill and decode.

namespace flyweight::v2::inkling2 {

using bailing::Matrix;
using bailing::matvec;
using bailing::row_bytes;
using bailing::row_decode;
using bailing::row_kernel;

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
                          float beta, float linear_beta, float* out) {
    for (std::size_t i = 0; i < n; ++i) {
        // situ(x) = x * sigmoid(beta*x) + linear_beta*x, clamped like the
        // reference implementation before it is multiplied by the up half.
        const float g = gate[i];
        const float activated = g * sigmoid(beta * g) + linear_beta * g;
        out[i] = activated * up[i];
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
    std::array<Matrix, 4> loop_down, loop_gain, loop_forget;
    bool has_attn_res = false;
    bool is_swa = false;

    const float* attn_norm_f32() const { return reinterpret_cast<const float*>(attn_norm.data); }
    const float* q_norm_f32() const { return reinterpret_cast<const float*>(q_norm.data); }
    const float* k_norm_f32() const { return reinterpret_cast<const float*>(k_norm.data); }
    const float* v_norm_f32() const { return reinterpret_cast<const float*>(v_norm.data); }
    const float* ffn_norm_f32() const { return reinterpret_cast<const float*>(ffn_norm.data); }
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
        position.clear();
        blocks.clear();
        prelude_blocks.clear();
        prelude.clear();
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

        const auto pad = [](const Tensor* t, std::size_t want) {
            return t ? t->shape.size() >= 2 ? static_cast<std::size_t>(t->shape[1]) : 1 : want;
        };
        (void)pad;

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
            if (const auto* t = find(pre + "attn_res_query.weight"))
                lw.attn_res_q = Matrix{tensor_data(*m, *t), t->type};
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
    }

    // Attention-residual mixing over the block bank. `columns` is the number
    // of learned queries (one for FFN, up to four for the loop messages).
    std::vector<float> mix(const std::vector<std::vector<float>>& bank,
                           const std::vector<float>& cur,
                           const float* query, std::size_t columns) const {
        const std::size_t n = bank.size() + 1;
        std::vector<float> states(n * g.hidden);
        std::vector<float> scores(n);
        for (std::size_t i = 0; i < bank.size(); ++i)
            rms_norm_into(bank[i].data(), bank[i].data(), g.hidden, 1e-6f, states.data() + i * g.hidden);
        rms_norm_into(cur.data(), cur.data(), g.hidden, 1e-6f, states.data() + bank.size() * g.hidden);
        const float scale = 1.0f / std::sqrt(static_cast<float>(g.hidden));
        for (std::size_t i = 0; i < n; ++i) {
            float dot = 0.0f;
            const float* s = states.data() + i * g.hidden;
            for (std::size_t j = 0; j < g.hidden; ++j) dot += s[j] * query[j];
            scores[i] = dot * scale;
        }
        softmax_into(scores.data(), n);
        std::vector<float> out(columns * g.hidden, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            const float* s = states.data() + i * g.hidden;
            for (std::size_t c = 0; c < columns; ++c) {
                float* o = out.data() + c * g.hidden;
                for (std::size_t j = 0; j < g.hidden; ++j) o[j] += scores[i] * s[j];
            }
        }
        return out;
    }

    // Causal short convolution with a residual. `dynamic` adds per-head taps
    // for full-attention layers.
    std::vector<float> convolve(const std::vector<float>& input,
                                Matrix weight, const Matrix* dynamic,
                                const float* dynamic_input,
                                std::vector<float>& history,
                                std::size_t channels) const {
        // Weight rows are taps; row j holds the channel weights for lag j.
        const auto kernel = row_kernel(weight.type, channels);
        const auto* base = reinterpret_cast<const std::uint8_t*>(weight.data);
        std::vector<float> out(input.size(), 0.0f);
        for (std::size_t i = 0; i < input.size(); ++i) {
            float total = kernel(base, input.data(), channels, 0);
            for (std::size_t lag = 1; lag < g.conv_taps; ++lag)
                total += kernel(base, history.data() + (lag - 1) * channels, channels, lag);
            out[i] = total;
        }
        // The current input becomes the most recent history row.
        for (std::size_t lag = g.conv_taps - 1; lag > 1; --lag)
            std::copy(history.data() + (lag - 2) * channels,
                      history.data() + (lag - 1) * channels,
                      history.data() + (lag - 1) * channels);
        if (g.conv_taps > 1) std::copy(input.begin(), input.end(), history.data());

        if (dynamic) {
            const std::size_t heads = channels / g.head_dim;
            std::vector<float> taps_values(g.conv_taps * heads);
            if (!dynamic_input)
                throw std::runtime_error("inkling2: dynamic convolution needs its input");
            matvec(*dynamic, dynamic_input, g.hidden, taps_values.size(), taps_values.data());
            for (std::size_t h = 0; h < heads; ++h)
                for (std::size_t lag = 0; lag < g.conv_taps; ++lag) {
                    const float t = taps_values[lag * heads + h];
                    const float* window = lag == 0 ? input.data()
                        : history.data() + (lag - 1) * channels + h * g.head_dim;
                    for (std::size_t k = 0; k < g.head_dim; ++k)
                        out[h * g.head_dim + k] += t * window[k];
                }
        }
        const float norm = std::sqrt(static_cast<float>((dynamic ? 2 : 1) * g.conv_taps + 1));
        for (std::size_t i = 0; i < input.size(); ++i) out[i] = (input[i] + out[i]) / norm;
        return out;
    }

    std::vector<float> moe(const std::vector<float>& x, const LayerWeights& lw) const {
        std::vector<float> logits(g.experts);
        matvec(lw.router, x.data(), g.hidden, g.experts, logits.data());
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
        const auto gate_up_kernel = row_kernel(lw.gate_up.type, g.hidden);
        const auto down_kernel = row_kernel(lw.down.type, ff);
        const std::size_t gu_stride = ff * row_bytes(lw.gate_up.type, g.hidden);
        const std::size_t down_stride = g.hidden * row_bytes(lw.down.type, ff);
        const auto* gu_base = reinterpret_cast<const std::uint8_t*>(lw.gate_up.data);
        const auto* down_base = reinterpret_cast<const std::uint8_t*>(lw.down.data);
        std::vector<float> result(g.hidden, 0.0f);
        for (std::uint32_t k = 0; k < used; ++k) {
            const auto* expert_gu = gu_base + static_cast<std::size_t>(order[k]) * gu_stride;
            std::vector<float> gate(ff), up(ff), act(ff), down(g.hidden);
            for (std::size_t r = 0; r < ff; ++r) {
                gate[r] = gate_up_kernel(expert_gu, x.data(), g.hidden, r);
                up[r]   = gate_up_kernel(expert_gu, x.data(), g.hidden, ff + r);
            }
            situ_glu_into(gate.data(), up.data(), ff, g.situ_beta, g.situ_linear_beta, act.data());
            const auto* expert_down = down_base + static_cast<std::size_t>(order[k]) * down_stride;
            for (std::size_t r = 0; r < g.hidden; ++r)
                down[r] = down_kernel(expert_down, act.data(), ff, r);
            for (std::size_t r = 0; r < g.hidden; ++r) result[r] += weights[k] * down[r];
        }
        return result;
    }

    std::vector<float> control(const std::vector<float>& cur,
                               const std::vector<float>& message,
                               const LayerWeights& lw, int c) const {
        std::vector<float> pair(2 * g.hidden), gate(g.gate_rank), gain(g.hidden), forget(g.hidden);
        std::vector<float> norm_cur(g.hidden), norm_msg(g.hidden);
        rms_norm_into(cur.data(), cur.data(), g.hidden, 1e-6f, norm_cur.data());
        rms_norm_into(message.data(), message.data(), g.hidden, 1e-6f, norm_msg.data());
        std::copy(norm_cur.begin(), norm_cur.end(), pair.begin());
        std::copy(norm_msg.begin(), norm_msg.end(), pair.begin() + g.hidden);
        matvec(lw.loop_down[c], pair.data(), 2 * g.hidden, g.gate_rank, gate.data());
        for (auto& v : gate) v = std::tanh(v);
        matvec(lw.loop_gain[c], gate.data(), g.gate_rank, g.hidden, gain.data());
        matvec(lw.loop_forget[c], gate.data(), g.gate_rank, g.hidden, forget.data());
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
        const LayerWeights& lw = layers[phys];
        CacheSlot& cache = slots[slot];
        const std::size_t kv_channels = static_cast<std::size_t>(g.kv_heads) * g.head_dim;
        const std::uint32_t heads = g.heads_by_layer.size() > phys ? g.heads_by_layer[phys] : g.heads;
        const std::uint32_t kvh = g.kv_heads_by_layer.size() > phys ? g.kv_heads_by_layer[phys] : g.kv_heads;

        std::vector<float> inpSA = input;
        std::vector<float> mixed[3];
        if (lw.has_attn_res) {
            std::vector<float> q(g.hidden);
            const float* qsrc = reinterpret_cast<const float*>(lw.attn_res_q.data);
            std::copy(qsrc, qsrc + g.hidden, q.begin());
            auto m = mix(blocks, input, q.data(), 1);
            mixed[0] = m;
            mixed[1] = m;
            mixed[2] = m;
        } else {
            mixed[0] = mixed[1] = mixed[2] = input;
        }
        if (messages) {
            // The four messages were already produced by `mix()` over the
            // previous pass's bank; each input gets its own controller.
            for (int c = 0; c < 3; ++c)
                mixed[c] = control(mixed[c], (*messages)[c], lw, c);
        }
        const float* attn_norm = lw.attn_norm ? lw.attn_norm_f32() : nullptr;
        if (lw.is_swa && messages) {
            if (!attn_norm) throw std::runtime_error("inkling2: missing attn_norm on a sliding layer");
            for (int c = 0; c < 3; ++c)
                rms_norm_into(mixed[c].data(), attn_norm, g.hidden, g.eps, mixed[c].data());
        } else if (lw.is_swa) {
            if (!attn_norm) throw std::runtime_error("inkling2: missing attn_norm on a sliding layer");
            rms_norm_into(mixed[0].data(), attn_norm, g.hidden, g.eps, mixed[0].data());
            mixed[1] = mixed[2] = mixed[0];
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
                           V(kvh * g.head_dim), R(heads * g.rel_len);
        matvec(lw.wq, mixed[0].data(), g.hidden, Q.size(), Q.data());
        matvec(lw.wk, mixed[1].data(), g.hidden, K.size(), K.data());
        matvec(lw.wv, mixed[2].data(), g.hidden, V.size(), V.data());
        matvec(lw.rel, mixed[0].data(), g.hidden, R.size(), R.data());

        const Matrix* kdyn = lw.k_conv_proj ? &lw.k_conv_proj : nullptr;
        const Matrix* vdyn = lw.v_conv_proj ? &lw.v_conv_proj : nullptr;
        K = convolve(K, lw.k_conv, kdyn, mixed[1].data(), cache.history[0], kvh * g.head_dim);
        V = convolve(V, lw.v_conv, vdyn, mixed[2].data(), cache.history[1], kvh * g.head_dim);
        clip_rows(Q.data(), heads, g.head_dim,
                  reinterpret_cast<const float*>(lw.q_clip.data), g.qk_clip);
        clip_rows(K.data(), kvh, g.head_dim, nullptr, g.qk_clip);
        for (std::uint32_t h = 0; h < kvh; ++h)
            rms_norm_into(V.data() + h * g.head_dim, V.data() + h * g.head_dim,
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
        std::vector<float> sinks(heads, 0.0f);
        {
            std::vector<float> proj(heads);
            matvec(lw.sink_proj, mixed[0].data(), g.hidden, heads, proj.data());
            for (std::uint32_t h = 0; h < heads; ++h) sinks[h] = proj[h];
        }
        // The bias table is [extent][rel_len] and the GGUF checkpoint stores it
        // as f32. Project R through it, then subtract the learned sink logit so
        // the extra zero-valued sink position can join the softmax.
        std::vector<float> bias((extent + 1) * heads, 0.0f);
        if (lw.rel_bias.type != 0)
            throw std::runtime_error("inkling2: expected an f32 relative-bias table");
        {
            const float* table = reinterpret_cast<const float*>(lw.rel_bias.data);
            for (std::uint32_t d = 0; d < extent; ++d)
                for (std::uint32_t h = 0; h < heads; ++h) {
                    float dot = 0.0f;
                    for (std::uint32_t r = 0; r < g.rel_len; ++r)
                        dot += table[static_cast<std::size_t>(d) * g.rel_len + r] *
                               R[static_cast<std::size_t>(h) * g.rel_len + r];
                    bias[d * heads + h] = dot - sinks[h];
                }
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));
        std::vector<float> attended(heads * g.head_dim, 0.0f);
        for (std::uint32_t h = 0; h < heads; ++h) {
            const std::uint32_t kv_head = h / std::max(1u, heads / kvh);
            const float* q = Q.data() + h * g.head_dim;
            std::vector<float> scores(cache.length + 1, 0.0f); // + sink
            for (std::uint32_t j = 0; j < cache.length; ++j) {
                float dot = 0.0f;
                const float* k = cache.keys.data() + j * kv_channels + kv_head * g.head_dim;
                for (std::uint32_t i = 0; i < g.head_dim; ++i) dot += q[i] * k[i];
                const float d = pos - cache.positions[j];
                const std::uint32_t idx = (d < 0 || d >= extent) ? extent : static_cast<std::uint32_t>(d);
                scores[j] = dot * scale + bias[idx * heads + h];
            }
            scores[cache.length] = 0.0f; // sink logit
            softmax_into(scores.data(), scores.size());
            for (std::uint32_t j = 0; j < cache.length; ++j) {
                const float* v = cache.values.data() + j * kv_channels + kv_head * g.head_dim;
                for (std::uint32_t i = 0; i < g.head_dim; ++i)
                    attended[h * g.head_dim + i] += scores[j] * v[i];
            }
        }

        std::vector<float> gate(heads * g.head_dim);
        matvec(lw.wgate, mixed[0].data(), g.hidden, gate.size(), gate.data());
        for (auto& v : gate) v = sigmoid(v);
        for (std::size_t i = 0; i < gate.size(); ++i) attended[i] *= gate[i];
        std::vector<float> proj(g.hidden);
        matvec(lw.wo, attended.data(), heads * g.head_dim, g.hidden, proj.data());
        auto attn_out = convolve(proj, lw.attn_post_conv, nullptr, nullptr, cache.history[2], g.hidden);

        if (phys % g.block_size == 0) blocks.push_back(inpSA);
        else for (std::size_t i = 0; i < g.hidden; ++i) inpSA[i] += attn_out[i];

        std::vector<float> ffn_inp = inpSA;
        const float* ffn_q = reinterpret_cast<const float*>(lw.ffn_res_q.data);
        auto ffn_mixed = mix(blocks, inpSA, ffn_q, 1);
        if (messages) ffn_mixed = control(ffn_mixed, (*messages)[3], lw, 3);
        std::vector<float> normed(g.hidden);
        rms_norm_into(ffn_mixed.data(), lw.ffn_norm_f32(), g.hidden, g.eps, normed.data());
        auto ffn_out = moe(normed, lw);
        ffn_out = convolve(ffn_out, lw.ffn_post_conv, nullptr, nullptr, cache.history[3], g.hidden);
        for (std::size_t i = 0; i < g.hidden; ++i) ffn_inp[i] += ffn_out[i];
        return ffn_inp;
    }

    std::vector<float> step(std::uint32_t token) {
        if (!position.empty()) ++position.back();
        else position.push_back(0);
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
            std::vector<std::vector<float>> messages(4);
            if (pass > 0) {
                const float* msg_q = loop_res_query.data();
                auto packed = mix(bank, bank_out, msg_q, 4);
                for (int c = 0; c < 4; ++c) {
                    messages[c].resize(g.hidden);
                    std::copy(packed.begin() + c * g.hidden,
                              packed.begin() + (c + 1) * g.hidden, messages[c].begin());
                }
            }
            for (std::uint32_t ip = g.core_start; ip < g.core_end; ++ip) {
                const std::uint32_t slot = pass == 0 ? ip
                    : g.physical + (pass - 1) * core + (ip - g.core_start);
                hidden = layer_step(slot, ip, hidden, pass > 0 ? &messages : nullptr, pass);
            }
        }
        for (std::uint32_t il = g.core_end; il < g.physical; ++il)
            hidden = layer_step(il, il, hidden, nullptr, 0);

        std::vector<float> mixed = mix(blocks, hidden,
            floats("output_res_query.weight"), 1);
        std::vector<float> normed(g.hidden);
        rms_norm_into(mixed.data(), reinterpret_cast<const float*>(output_norm.data),
                      g.hidden, g.eps, normed.data());
        std::vector<float> logits(g.vocab);
        matvec(output, normed.data(), g.hidden, g.vocab, logits.data());
        if (g.softcap > 0.0f)
            for (auto& v : logits) v = g.softcap * std::tanh(v / g.softcap);
        logits.resize(g.vocab_unpadded);
        return logits;
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
        if (!runtime || !tokens || !logits || !count)
            throw std::runtime_error("invalid inkling2 eval arguments");
        const std::size_t need = runtime->impl.g.vocab_unpadded;
        std::vector<float> last;
        for (std::uint32_t i = 0; i < count; ++i) last = runtime->impl.step(tokens[i]);
        if (last.size() < need) throw std::runtime_error("inkling2 logits buffer is too small");
        std::copy(last.begin(), last.end(), logits);
        return 0;
    });
}

} // extern "C"
