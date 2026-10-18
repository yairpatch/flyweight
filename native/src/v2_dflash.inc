// DFlash drafter with the LiLiCorr re-ranker: the draft model NVIDIA ships
// beside Nemotron 3.5 Bolt (GGUF arch `dflash`, attached through
// --mtp-model). Semantics follow llama.cpp cd0a5b6 (src/models/dflash.cpp and
// the draft-dflash path of common/speculative.cpp); plans/nemotron-dflash.md
// carries the derivation.
//
// Every target forward records the residual stream entering the drafter's
// target layers (capture_slot). After the forward, those rows are injected:
// fc + RMS norm turns each into a context row g_p, whose K/V per draft layer
// (k_norm, NEOX RoPE at p) land in a ring of window + slack positions. A draft
// runs one block [id_last, mask x 7] at pos0..pos0+7 through the trunk
// (dynamic two-tap conv around every sublayer, non-causal attention over the
// block and the windowed context, sinks), the target's LM head on rows 1..7,
// then LiLiCorr: top-8 candidates per slot, a two-layer transformer over the
// 56 (slot, candidate) nodes, factor heads, and a greedy walk over the
// lattice scores. The walk's tokens are the drafts; the verify pass is the
// existing speculative round with them as preset drafts.
//
// Draft numerics only move the acceptance rate -- the target re-scores every
// draft -- so the bf16 and f32 matrices go to the device as Q8_0.

namespace {

constexpr std::uint32_t kDflashAnchorSlots = 16;
// Ring slack past the window: a verify writes up to block-1 rows past the
// last accepted position, and none of them may land on a slot a later draft
// still reads.
constexpr std::uint32_t kDflashRingSlack = 16;
// Blocks per logit row in the first top-k stage; each leaves 8 (value, id)
// pairs and a (max, sum-exp) pair, 18 floats.
constexpr std::uint32_t kDflashTopkChunks = 64;

std::uint64_t dflash_align(std::uint64_t bytes) { return (bytes + 255) & ~std::uint64_t{255}; }

struct DflashWorkspace {
    // Offsets into QwenDflash::workspace.
    std::uint64_t g = 0, inject_k = 0, inject_v = 0, q8 = 0, q8_scales = 0;
    std::uint64_t tokens = 0, h = 0, a = 0, a2 = 0, q = 0, k = 0, v = 0, attended = 0,
        projected = 0, r = 0, coeff = 0, gate = 0, up = 0, y = 0, z = 0, logits = 0;
    std::uint64_t ids = 0, logprobs = 0, embedded = 0, tok = 0, pass = 0, features = 0,
        fs = 0, x = 0, normed = 0, qkv = 0, mixed = 0, delta = 0, mlp = 0, anchor = 0,
        anchor_normed = 0, concat = 0, phi = 0, out_factor = 0, in_factor = 0,
        anchor_factor = 0, scores = 0, topk_partial = 0;
    std::uint64_t bytes = 0;
};

DflashWorkspace dflash_layout(const QwenDflash& d, std::uint32_t vocabulary) {
    DflashWorkspace w;
    std::uint64_t cursor = 0;
    auto take = [&](std::uint64_t& field, std::uint64_t bytes) {
        field = cursor;
        cursor += dflash_align(bytes);
    };
    const std::uint64_t f = sizeof(float);
    const std::uint64_t cap = d.inject_capacity, kvw = d.kv_heads * d.head_dim;
    const std::uint64_t rows = d.block, qw = static_cast<std::uint64_t>(d.heads) * d.head_dim;
    const std::uint64_t max_in = std::max<std::uint64_t>(
        {d.features, d.ffn, d.hidden, qw, d.width, d.mlp});
    take(w.g, cap * d.hidden * f);
    take(w.inject_k, cap * kvw * f);
    take(w.inject_v, cap * kvw * f);
    // Q8 activations: up to 8 rows of the widest input, every injected row of
    // a K/V projection, or every LiLiCorr node of its widest Q8_0 input.
    const std::uint64_t q8_values = std::max<std::uint64_t>(
        {8 * max_in, cap * d.hidden, static_cast<std::uint64_t>(d.nodes) * 3 * d.width});
    take(w.q8, q8_values);
    take(w.q8_scales, (q8_values / 32) * f);
    take(w.tokens, rows * sizeof(std::uint32_t));
    for (auto* field : {&w.h, &w.a, &w.a2, &w.projected, &w.r, &w.y, &w.z})
        take(*field, rows * d.hidden * f);
    take(w.q, rows * qw * f);
    take(w.attended, rows * qw * f);
    take(w.k, rows * kvw * f);
    take(w.v, rows * kvw * f);
    take(w.coeff, rows * 4 * (d.hidden / d.conv_group) * f);
    take(w.gate, rows * d.ffn * f);
    take(w.up, rows * d.ffn * f);
    take(w.logits, (rows - 1) * vocabulary * f);
    const std::uint64_t nodes = d.nodes, width = d.width;
    take(w.ids, nodes * sizeof(std::int32_t));
    take(w.topk_partial, static_cast<std::uint64_t>(d.slots) * kDflashTopkChunks * 18 * f);
    take(w.logprobs, nodes * f);
    take(w.embedded, nodes * d.hidden * f);
    for (auto* field : {&w.tok, &w.features, &w.fs, &w.x, &w.normed, &w.mixed, &w.delta})
        take(*field, nodes * width * f);
    take(w.pass, d.slots * width * f);
    take(w.qkv, nodes * 3 * width * f);
    take(w.mlp, nodes * d.mlp * f);
    take(w.anchor, width * f);
    take(w.anchor_normed, width * f);
    take(w.concat, nodes * 3 * width * f);
    take(w.phi, nodes * d.factor * f);
    take(w.out_factor, nodes * d.factor * f);
    take(w.in_factor, nodes * d.factor * f);
    take(w.anchor_factor, d.factor * f);
    take(w.scores, d.slots * d.topk * d.topk * f);
    w.bytes = cursor;
    return w;
}

// FLYWEIGHT_DFLASH_PROFILE=1: every drafter launch bracketed by events and
// synchronized, its GPU time tallied by kernel name and reported every 32
// drafts. The wall time of a profiled run means nothing; the per-kernel
// durations do.
bool dflash_profile_enabled() {
    static const bool on = [] {
        const char* setting = std::getenv("FLYWEIGHT_DFLASH_PROFILE");
        return setting && setting[0] == '1';
    }();
    return on;
}

std::map<std::string, std::pair<std::uint64_t, double>>& dflash_profile() {
    static std::map<std::string, std::pair<std::uint64_t, double>> tally;
    return tally;
}

void dflash_profile_report() {
    auto& tally = dflash_profile();
    std::vector<std::pair<std::string, std::pair<std::uint64_t, double>>> ordered(tally.begin(),
                                                                                  tally.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    double total = 0.0;
    for (const auto& entry : ordered) total += entry.second.second;
    std::fprintf(stderr, "[dflash] %.2f ms of drafter kernels per profiled window\n", total);
    for (const auto& entry : ordered)
        std::fprintf(stderr, "[dflash]   %-36s %6llu calls %8.2f ms\n", entry.first.c_str(),
                     static_cast<unsigned long long>(entry.second.first), entry.second.second);
    tally.clear();
}

void dflash_launch(const char* name, std::uint32_t gx, std::uint32_t gy, std::uint32_t bx,
                   void** args, std::uint64_t stream, std::uint32_t shared = 0) {
    if (!dflash_profile_enabled()) {
        if (flyweight_gpu_launch_named(name, gx, gy, bx, shared, stream, args) != 0)
            throw std::runtime_error(std::string("DFlash drafter kernel failed: ") + name);
        return;
    }
    static std::uint64_t start = 0, stop = 0;
    if (!start) {
        flyweight_gpu_timed_event_create(&start);
        flyweight_gpu_timed_event_create(&stop);
    }
    flyweight_gpu_event_record(start, stream);
    if (flyweight_gpu_launch_named(name, gx, gy, bx, shared, stream, args) != 0)
        throw std::runtime_error(std::string("DFlash drafter kernel failed: ") + name);
    flyweight_gpu_event_record(stop, stream);
    flyweight_gpu_event_sync(stop);
    float ms = 0.0f;
    flyweight_gpu_event_elapsed(start, stop, &ms);
    auto& entry = dflash_profile()[name];
    ++entry.first;
    entry.second += ms;
}

std::uint64_t dflash_tensor_index(const FlyweightV2QwenRuntime& runtime, QwenDflash& d,
                                  const std::string& name) {
    const std::string full = "dflash." + name;
    if (!has_tensor(*runtime.model, full))
        throw std::runtime_error("DFlash drafter is missing tensor " + name);
    const auto index = tensor_index(*runtime.model, full);
    d.tensors.push_back(index);
    return index;
}

const Tensor& dflash_described(const FlyweightV2QwenRuntime& runtime, std::uint64_t index) {
    return runtime.model->tensors[index];
}

// y = W x for `rows` rows on whatever the tensor's device type is; NVFP4
// carries its per-tensor scale. The <=8-row NVFP4 batches take the Q8
// activation kernel like the target's own short batches.
void dflash_matmul(const FlyweightV2QwenRuntime& runtime, const QwenDflash& d,
                   const DflashWorkspace& w, std::uint64_t index, std::uint64_t input,
                   std::uint64_t output, int input_size, int output_size, int rows,
                   std::uint64_t stream) {
    std::uint64_t matrix = runtime.device_tensors[index];
    const auto type = qwen_device_type(runtime, index);
    if (type == 40) {
        float scale = qwen_nvfp4_weight_scale(runtime, static_cast<std::size_t>(index));
        if (qwen_nvfp4_q8_enabled() && (input_size & 63) == 0) {
            // Q8 activations at every batch size, as llama.cpp runs NVFP4, so
            // a prompt's injected rows match the ones decode and verify inject:
            // eight rows per pass of the Q8 rows kernel.
            std::uint64_t q8 = d.workspace + w.q8, q8_scales = d.workspace + w.q8_scales;
            int scale_stride = (input_size / 32) * 2;
            for (int first = 0; first < rows; first += 8) {
                int batch = std::min(8, rows - first);
                std::uint64_t batch_input = input +
                    static_cast<std::uint64_t>(first) * input_size * sizeof(float);
                std::uint64_t batch_output = output +
                    static_cast<std::uint64_t>(first) * output_size * sizeof(float);
                void* quant[] = {&batch_input, &q8, &q8_scales, &input_size, &scale_stride};
                dflash_launch("quantize_q8_blocks_rows", (input_size + 255) / 256,
                              static_cast<std::uint32_t>(batch), 256, quant, stream);
                void* args[] = {&matrix, &q8, &q8_scales, &batch_output, &input_size,
                                &output_size, &batch, &scale_stride, &scale};
                dflash_launch("nvfp4_q8_matvec_transposed_rows", (output_size + 7) / 8, 1, 256,
                              args, stream);
            }
            return;
        }
        void* args[] = {&matrix, &input, &output, &input_size, &output_size, &rows, &scale};
        if (rows <= 8)
            dflash_launch("nvfp4_matvec_transposed_warp_rows", (output_size + 7) / 8, 1, 256,
                          args, stream);
        else
            dflash_launch("nvfp4_matmul_tiled", (output_size + 31) / 32,
                          (static_cast<std::uint32_t>(rows) + 31) / 32, 256, args, stream);
        return;
    }
    if (type == 8 && (input_size & 255) == 0) {
        // Q8 activations against the Q8_0 rows: the dp4a rows kernel reads the
        // matrix once for up to 8 rows, the dp4a tile once per 32. The float
        // tile kernel ran the 570 MB head at ~30 GB/s.
        std::uint64_t q8 = d.workspace + w.q8, q8_scales = d.workspace + w.q8_scales;
        int scale_stride = (input_size / 32) * 2;
        void* quant[] = {&input, &q8, &q8_scales, &input_size, &scale_stride};
        dflash_launch("quantize_q8_blocks_rows", (input_size + 255) / 256,
                      static_cast<std::uint32_t>(rows), 256, quant, stream);
        if (rows <= 8) {
            static std::map<int, std::string> names;
            auto& name = names[rows];
            if (name.empty()) {
                const char* base = "q80_q8_matvec_transposed_rows";
                static const char* const suffixes[] = {"", "", "_r2", "_r3", "_r4", "_r5", "_r6", "_r7"};
                name = rows >= 2 && rows <= 7 ? std::string(base) + suffixes[rows] : base;
                if (!flyweight_gpu_kernel_available(name.c_str())) name = base;
            }
            void* args[] = {&matrix, &q8, &q8_scales, &output, &input_size, &output_size,
                            &rows, &scale_stride};
            dflash_launch(name.c_str(), static_cast<std::uint32_t>(output_size), 1,
                          qwen_q8_matvec_block(input_size), args, stream);
            return;
        }
        for (int first = 0; first < rows; first += 32) {
            int batch = std::min(32, rows - first);
            std::uint64_t batch_q8 = q8 + static_cast<std::uint64_t>(first) * input_size;
            std::uint64_t batch_scales = q8_scales +
                static_cast<std::uint64_t>(first) * (input_size / 32) * sizeof(float);
            std::uint64_t batch_output = output +
                static_cast<std::uint64_t>(first) * output_size * sizeof(float);
            void* args[] = {&matrix, &batch_q8, &batch_scales, &batch_output, &input_size,
                            &output_size, &batch, &scale_stride};
            dflash_launch("q80_q8_matmul_tiled", (output_size + 15) / 16, 1, 256, args, stream);
        }
        return;
    }
    void* args[] = {&matrix, &input, &output, &input_size, &output_size, &rows};
    if (type == 8)
        dflash_launch("q8_matmul_tiled", (output_size + 31) / 32,
                      (static_cast<std::uint32_t>(rows) + 31) / 32, 256, args, stream);
    else if (type == 0)
        dflash_launch("qwen_f32_matmul_rows", static_cast<std::uint32_t>(output_size),
                      static_cast<std::uint32_t>(rows), 256, args, stream);
    else
        throw std::runtime_error("DFlash drafter has no matmul for tensor type " +
                                 std::to_string(type));
}

void dflash_rms(const FlyweightV2QwenRuntime& runtime, const QwenDflash& d,
                std::uint64_t input, std::uint64_t weight_index, std::uint64_t output,
                int rows, int columns, std::uint64_t stream) {
    std::uint64_t weights = runtime.device_tensors[weight_index];
    float epsilon = d.epsilon;
    int one_centered = 0;
    void* args[] = {&input, &weights, &output, &rows, &columns, &epsilon, &one_centered};
    dflash_launch("rms_norm_rows", static_cast<std::uint32_t>(rows), 1, 256, args, stream);
}

void dflash_bias(const FlyweightV2QwenRuntime& runtime, std::uint64_t x,
                 std::uint64_t bias_index, int rows, int width, int act,
                 std::uint64_t stream) {
    std::uint64_t bias = runtime.device_tensors[bias_index];
    void* args[] = {&x, &bias, &width, &act};
    dflash_launch("dflash_bias_act_rows", (width + 255) / 256,
                  static_cast<std::uint32_t>(rows), 256, args, stream);
}

void dflash_head_norm(const FlyweightV2QwenRuntime& runtime, const QwenDflash& d,
                      std::uint64_t x, std::uint64_t weight_index, int rows, int heads,
                      int position_base, std::uint64_t stream) {
    std::uint64_t weight = runtime.device_tensors[weight_index];
    int head_dim = static_cast<int>(d.head_dim), stride = heads * head_dim, rotate = 1;
    float epsilon = d.epsilon, base = d.freq_base;
    void* args[] = {&x, &weight, &heads, &head_dim, &stride, &epsilon, &position_base,
                    &base, &rotate};
    dflash_launch("dflash_head_norm_rope_rows", static_cast<std::uint32_t>(heads),
                  static_cast<std::uint32_t>(rows), 128, args, stream);
}

void dflash_conv(const FlyweightV2QwenRuntime& runtime, const QwenDflash& d,
                 std::uint64_t x, std::uint64_t coeff, std::uint64_t base_index,
                 std::uint64_t residual, std::uint64_t out, int side, std::uint64_t stream) {
    std::uint64_t base = runtime.device_tensors[base_index];
    int channels = static_cast<int>(d.hidden), group = static_cast<int>(d.conv_group);
    void* args[] = {&x, &coeff, &base, &residual, &out, &channels, &group, &side};
    dflash_launch("dflash_conv_rows", (channels + 255) / 256, d.block, 256, args, stream);
}

// Embedding rows of the target table. `host` holds the ids when they are
// known on the host; otherwise they are read back from `device_ids`.
void dflash_embed(FlyweightV2QwenRuntime& runtime, const std::uint32_t* host,
                  std::uint64_t device_ids, int count, std::uint64_t output) {
    int hidden = static_cast<int>(runtime.model->config.hidden_size);
    std::uint64_t table = 0, index = device_ids;
    if (runtime.embeddings_host_resident) {
        std::vector<std::uint32_t> ids;
        if (!host) {
            ids.resize(count);
            if (flyweight_gpu_download(ids.data(), device_ids, count * sizeof(std::uint32_t),
                                       runtime.stream) != 0 ||
                flyweight_gpu_stream_sync(runtime.stream) != 0)
                throw std::runtime_error("DFlash candidate download failed");
            host = ids.data();
        }
        table = qwen_stage_embedding_rows(runtime, host, count);
        index = runtime.embedding_row_index;
    } else {
        table = runtime.device_tensors[runtime.token_embeddings];
    }
    void* args[] = {&table, &index, &output, &count, &hidden};
    dflash_launch(qwen_embedding_kernel(qwen_device_type(runtime, runtime.token_embeddings), true),
                  (hidden + 255) / 256, static_cast<std::uint32_t>(count), 256, args,
                  runtime.stream);
}

}  // namespace

// The drafter's plan, from the sidecar's config and tensor shapes. Runs at
// create, after the target plan; throws on any geometry it does not compute.
void qwen_dflash_plan(FlyweightV2QwenRuntime& runtime, std::uint32_t drafts) {
    const auto& sidecar = *runtime.model->dflash_sidecar;
    const auto& config = sidecar.config;
    auto d = std::make_unique<QwenDflash>();
    d->hidden = config.hidden_size;
    d->heads = config.attention_heads;
    d->kv_heads = config.attention_kv_heads;
    d->block = config.draft_block_size;
    d->conv_group = config.dflash_conv_group;
    d->window = config.sliding_window;
    d->mask_token = config.mask_token_id;
    d->features = static_cast<std::uint32_t>(config.target_layers.size()) * d->hidden;
    d->epsilon = config.rms_norm_epsilon > 0.0f ? config.rms_norm_epsilon : 1e-6f;
    d->freq_base = config.rope_freq_base > 0.0f ? config.rope_freq_base : 10000.0f;
    d->topk = config.lilicorr_topk;
    d->vector_eps = config.lilicorr_vector_eps > 0.0f ? config.lilicorr_vector_eps : 1e-4f;
    d->logit_scale = config.lilicorr_logit_scale > 0.0f ? config.lilicorr_logit_scale : 8.0f;
    if (!d->heads || !d->kv_heads || d->heads % d->kv_heads || d->block < 2 || d->block > 9 ||
        !d->window || !d->topk || d->topk > 8)
        throw std::runtime_error("DFlash drafter geometry is incomplete");
    if (config.dflash_conv_kernel != 2 || !d->conv_group || d->hidden % d->conv_group)
        throw std::runtime_error("DFlash drafter conv must be two taps over whole channel groups");
    for (const bool swa : config.sliding_window_pattern)
        if (!swa) throw std::runtime_error("DFlash drafter with full-attention layers is not implemented");
    if (d->mask_token >= runtime.model->config.vocabulary_size)
        throw std::runtime_error("DFlash drafter mask token is outside the target vocabulary");
    d->window = std::min<std::uint32_t>(d->window, runtime.options.context_limit);
    d->ring_slots = d->window + kDflashRingSlack;
    d->drafts = std::min<std::uint32_t>(drafts, d->block - 1);
    auto& D = *d;
    auto shape = [&](std::uint64_t index) -> const std::vector<std::uint64_t>& {
        return dflash_described(runtime, index).shape;
    };
    auto expect = [&](std::uint64_t index, std::uint64_t in, std::uint64_t out) {
        const auto& s = shape(index);
        if (s.size() < 2 || s[0] != in || s[1] != out)
            throw std::runtime_error("DFlash drafter tensor " +
                                     dflash_described(runtime, index).name +
                                     " has an unexpected shape");
    };
    D.fc = dflash_tensor_index(runtime, D, "fc.weight");
    expect(D.fc, D.features, D.hidden);
    D.enc_norm = dflash_tensor_index(runtime, D, "enc.output_norm.weight");
    D.output_norm = dflash_tensor_index(runtime, D, "output_norm.weight");
    const std::uint32_t layers = config.layer_count;
    if (!layers) throw std::runtime_error("DFlash drafter has no blocks");
    for (std::uint32_t l = 0; l < layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        QwenDflash::Block b;
        b.attn_norm = dflash_tensor_index(runtime, D, p + "attn_norm.weight");
        b.attn_conv_proj = dflash_tensor_index(runtime, D, p + "attn_conv_proj.weight");
        b.attn_conv_base = dflash_tensor_index(runtime, D, p + "attn_conv_base");
        b.q = dflash_tensor_index(runtime, D, p + "attn_q.weight");
        b.k = dflash_tensor_index(runtime, D, p + "attn_k.weight");
        b.v = dflash_tensor_index(runtime, D, p + "attn_v.weight");
        b.q_norm = dflash_tensor_index(runtime, D, p + "attn_q_norm.weight");
        b.k_norm = dflash_tensor_index(runtime, D, p + "attn_k_norm.weight");
        b.sinks = dflash_tensor_index(runtime, D, p + "attn_sinks.weight");
        b.o = dflash_tensor_index(runtime, D, p + "attn_output.weight");
        b.ffn_norm = dflash_tensor_index(runtime, D, p + "ffn_norm.weight");
        b.ffn_conv_proj = dflash_tensor_index(runtime, D, p + "ffn_conv_proj.weight");
        b.ffn_conv_base = dflash_tensor_index(runtime, D, p + "ffn_conv_base");
        b.gate = dflash_tensor_index(runtime, D, p + "ffn_gate.weight");
        b.up = dflash_tensor_index(runtime, D, p + "ffn_up.weight");
        b.down = dflash_tensor_index(runtime, D, p + "ffn_down.weight");
        if (l == 0) {
            D.head_dim = static_cast<std::uint32_t>(shape(b.q)[1] / D.heads);
            D.ffn = static_cast<std::uint32_t>(shape(b.gate)[1]);
        }
        const std::uint64_t qw = static_cast<std::uint64_t>(D.heads) * D.head_dim;
        const std::uint64_t kvw = static_cast<std::uint64_t>(D.kv_heads) * D.head_dim;
        const std::uint64_t coeff = 4ull * (D.hidden / D.conv_group);
        expect(b.q, D.hidden, qw);
        expect(b.k, D.hidden, kvw);
        expect(b.v, D.hidden, kvw);
        expect(b.o, qw, D.hidden);
        expect(b.attn_conv_proj, D.hidden, coeff);
        expect(b.ffn_conv_proj, D.hidden, coeff);
        expect(b.gate, D.hidden, D.ffn);
        expect(b.up, D.hidden, D.ffn);
        expect(b.down, D.ffn, D.hidden);
        for (const auto base : {b.attn_conv_base, b.ffn_conv_base})
            if (tensor_elements(dflash_described(runtime, base)) != 4ull * D.hidden ||
                dflash_described(runtime, base).type != 0)
                throw std::runtime_error("DFlash drafter conv base must be f32 {hidden, 2, 2}");
        if (tensor_elements(dflash_described(runtime, b.sinks)) != D.heads ||
            tensor_elements(dflash_described(runtime, b.q_norm)) != D.head_dim ||
            tensor_elements(dflash_described(runtime, b.k_norm)) != D.head_dim)
            throw std::runtime_error("DFlash drafter head norms or sinks disagree with the heads");
        D.blocks.push_back(b);
    }
    if (D.head_dim == 0 || D.head_dim > 256 || D.head_dim % 2)
        throw std::runtime_error("DFlash drafter head width is unsupported");
    // LiLiCorr. The converter numbers its layers after the trunk's (blk.L+i)
    // and puts the candidate-feature MLP at blk.L+layers.
    D.slots = D.block - 1;
    D.nodes = D.slots * D.topk;
    D.token_proj = dflash_tensor_index(runtime, D, "lilicorr.token_proj.weight");
    D.width = static_cast<std::uint32_t>(shape(D.token_proj)[1]);
    expect(D.token_proj, D.hidden, D.width);
    D.token_proj_bias = dflash_tensor_index(runtime, D, "lilicorr.token_proj.bias");
    D.pass_proj = dflash_tensor_index(runtime, D, "lilicorr.pass_hidden_proj.weight");
    expect(D.pass_proj, D.hidden, D.width);
    D.pass_proj_bias = dflash_tensor_index(runtime, D, "lilicorr.pass_hidden_proj.bias");
    D.context_proj = dflash_tensor_index(runtime, D, "lilicorr.context_proj.weight");
    expect(D.context_proj, D.hidden, D.width);
    D.context_proj_bias = dflash_tensor_index(runtime, D, "lilicorr.context_proj.bias");
    std::uint32_t corrector_layers = 0;
    while (has_tensor(*runtime.model, "dflash.lilicorr.layers." +
                      std::to_string(corrector_layers) + ".attn.in_proj.weight"))
        ++corrector_layers;
    if (!corrector_layers) throw std::runtime_error("DFlash LiLiCorr head has no layers");
    for (std::uint32_t i = 0; i < corrector_layers; ++i) {
        const std::string p = "blk." + std::to_string(layers + i) + ".";
        const std::string q = "lilicorr.layers." + std::to_string(i) + ".attn.in_proj.";
        QwenDflash::Corrector c;
        c.attn_norm = dflash_tensor_index(runtime, D, p + "attn_norm.weight");
        c.in_proj = dflash_tensor_index(runtime, D, q + "weight");
        c.in_proj_bias = dflash_tensor_index(runtime, D, q + "bias");
        c.out = dflash_tensor_index(runtime, D, p + "attn_output.weight");
        c.out_bias = dflash_tensor_index(runtime, D, p + "attn_output.bias");
        c.ffn_norm = dflash_tensor_index(runtime, D, p + "ffn_norm.weight");
        c.up = dflash_tensor_index(runtime, D, p + "ffn_up.weight");
        c.up_bias = dflash_tensor_index(runtime, D, p + "ffn_up.bias");
        c.down = dflash_tensor_index(runtime, D, p + "ffn_down.weight");
        c.down_bias = dflash_tensor_index(runtime, D, p + "ffn_down.bias");
        if (i == 0) D.mlp = static_cast<std::uint32_t>(shape(c.up)[1]);
        expect(c.in_proj, D.width, 3ull * D.width);
        expect(c.out, D.width, D.width);
        expect(c.up, D.width, D.mlp);
        expect(c.down, D.mlp, D.width);
        D.correctors.push_back(c);
    }
    const std::string feature = "blk." + std::to_string(layers + corrector_layers) + ".";
    D.feature_norm = dflash_tensor_index(runtime, D, feature + "ffn_norm.weight");
    D.feature_norm_bias = dflash_tensor_index(runtime, D, feature + "ffn_norm.bias");
    D.feature_up = dflash_tensor_index(runtime, D, feature + "ffn_up.weight");
    D.feature_up_bias = dflash_tensor_index(runtime, D, feature + "ffn_up.bias");
    D.feature_down = dflash_tensor_index(runtime, D, feature + "ffn_down.weight");
    D.feature_down_bias = dflash_tensor_index(runtime, D, feature + "ffn_down.bias");
    expect(D.feature_up, 5, D.width);
    expect(D.feature_down, D.width, D.width);
    if (dflash_described(runtime, D.feature_up).type != 0)
        throw std::runtime_error("DFlash LiLiCorr feature MLP input must be f32");
    D.slot_embedding = dflash_tensor_index(runtime, D, "lilicorr.slot_embedding.weight");
    D.rank_embedding = dflash_tensor_index(runtime, D, "lilicorr.rank_embedding.weight");
    expect(D.slot_embedding, D.width, D.slots);
    expect(D.rank_embedding, D.width, D.topk);
    D.attention_bias = dflash_tensor_index(runtime, D, "lilicorr.attention_bias.bias");
    {
        const auto& s = shape(D.attention_bias);
        if (s.size() != 3 || s[0] != D.nodes || s[1] != D.nodes)
            throw std::runtime_error("DFlash LiLiCorr attention bias must be {nodes, nodes, heads}");
        D.corrector_heads = static_cast<std::uint32_t>(s[2]);
        if (!D.corrector_heads || D.width % D.corrector_heads || D.nodes > 64)
            throw std::runtime_error("DFlash LiLiCorr head geometry is unsupported");
    }
    D.corrector_norm = dflash_tensor_index(runtime, D, "lilicorr.output_norm.weight");
    D.anchor_norm = dflash_tensor_index(runtime, D, "lilicorr.anchor_norm.weight");
    D.factor_input = dflash_tensor_index(runtime, D, "lilicorr.factor_input_proj.weight");
    D.factor = static_cast<std::uint32_t>(shape(D.factor_input)[1]);
    expect(D.factor_input, 3ull * D.width, D.factor);
    D.factor_input_bias = dflash_tensor_index(runtime, D, "lilicorr.factor_input_proj.bias");
    D.out_head = dflash_tensor_index(runtime, D, "lilicorr.out_head.weight");
    D.out_head_bias = dflash_tensor_index(runtime, D, "lilicorr.out_head.bias");
    D.in_head = dflash_tensor_index(runtime, D, "lilicorr.in_head.weight");
    D.in_head_bias = dflash_tensor_index(runtime, D, "lilicorr.in_head.bias");
    D.anchor_head = dflash_tensor_index(runtime, D, "lilicorr.anchor_out_head.weight");
    D.anchor_head_bias = dflash_tensor_index(runtime, D, "lilicorr.anchor_out_head.bias");
    expect(D.out_head, D.factor, D.factor);
    expect(D.in_head, D.factor, D.factor);
    expect(D.anchor_head, D.width, D.factor);
    // Matrices the kernels above read as Q8_0 when the file stores them wider,
    // and the LiLiCorr NVFP4 projections, which run 56 node rows where the
    // NVFP4 kernels only batch 8 (the weight_scale_2 folds into the Q8_0
    // blocks on upload).
    std::vector<std::uint64_t> corrector_nvfp4 = {D.token_proj, D.pass_proj, D.context_proj};
    for (const auto& c : D.correctors) {
        corrector_nvfp4.push_back(c.out);
        corrector_nvfp4.push_back(c.up);
        corrector_nvfp4.push_back(c.down);
    }
    for (const auto index : D.tensors) {
        const auto& t = dflash_described(runtime, index);
        const bool corrector = std::find(corrector_nvfp4.begin(), corrector_nvfp4.end(), index) !=
                               corrector_nvfp4.end();
        if (t.shape.size() == 2 && t.shape[1] > 1 && t.shape[0] % 32 == 0 &&
            ((t.type == 30 || t.type == 0) || (t.type == 40 && corrector)) &&
            index != D.feature_up && index != D.slot_embedding && index != D.rank_embedding)
            D.requantized.push_back(index);
        else if (t.type != 0 && t.type != 40 && t.type != 8)
            throw std::runtime_error("DFlash drafter tensor " + t.name + " has type " +
                                     std::to_string(t.type) + ", which it cannot read");
        runtime.static_tensor_bytes += t.size;
    }
    const auto lm_type = runtime.model->tensors[runtime.lm_head].type;
    if (lm_type != 8 && lm_type != 0 && lm_type != 40)
        throw std::runtime_error("DFlash drafter needs a Q8_0, NVFP4 or f32 target LM head");
    D.anchor_positions.fill(-1);
    runtime.dflash = std::move(d);
}

// Prepare: the drafter's weights join the static arena (Q8_0 for the wide
// matrices), with its ring and workspace counted against the GPU budget.
void qwen_dflash_mark_persistent(const FlyweightV2QwenRuntime& runtime,
                                 std::vector<bool>& persistent,
                                 std::vector<bool>& preserve_bf16) {
    if (!runtime.dflash) return;
    for (const auto index : runtime.dflash->tensors) {
        persistent[index] = true;
        preserve_bf16[index] = true;
    }
}

void qwen_dflash_device_types(FlyweightV2QwenRuntime& runtime) {
    if (!runtime.dflash) return;
    for (const auto index : runtime.dflash->requantized) runtime.device_tensor_types[index] = 8;
}

std::uint64_t qwen_dflash_device_bytes(FlyweightV2QwenRuntime& runtime) {
    if (!runtime.dflash) return 0;
    auto& d = *runtime.dflash;
    d.inject_capacity = std::min<std::uint32_t>(
        std::max<std::uint32_t>(runtime.forward_rows_capacity, 1), d.window);
    const auto layout = dflash_layout(d, runtime.model->config.vocabulary_size);
    d.workspace_bytes = layout.bytes;
    const std::uint64_t ring = static_cast<std::uint64_t>(d.blocks.size()) * d.ring_slots *
                               d.kv_heads * d.head_dim * sizeof(float);
    const std::uint64_t capture = static_cast<std::uint64_t>(runtime.forward_rows_capacity) *
                                  runtime.capture_layers * runtime.model->config.hidden_size *
                                  sizeof(float);
    return 2 * dflash_align(ring) + dflash_align(d.ring_slots * sizeof(std::int32_t)) +
           dflash_align(kDflashAnchorSlots * d.hidden * sizeof(float)) +
           dflash_align(d.workspace_bytes) + dflash_align(capture);
}

void qwen_dflash_invalidate(FlyweightV2QwenRuntime& runtime) {
    if (!runtime.dflash) return;
    auto& d = *runtime.dflash;
    d.anchor_positions.fill(-1);
    if (d.ring_tags &&
        flyweight_gpu_memset(d.ring_tags, 0xff, d.ring_slots * sizeof(std::int32_t),
                             runtime.stream) != 0)
        throw std::runtime_error("DFlash ring reset failed");
}

void qwen_dflash_allocate(FlyweightV2QwenRuntime& runtime) {
    if (!runtime.dflash) return;
    auto& d = *runtime.dflash;
    qwen_dflash_device_bytes(runtime);
    const std::uint64_t ring = static_cast<std::uint64_t>(d.blocks.size()) * d.ring_slots *
                               d.kv_heads * d.head_dim * sizeof(float);
    if (flyweight_gpu_alloc(ring, &d.ring_k) != 0 || flyweight_gpu_alloc(ring, &d.ring_v) != 0 ||
        flyweight_gpu_alloc(d.ring_slots * sizeof(std::int32_t), &d.ring_tags) != 0 ||
        flyweight_gpu_alloc(kDflashAnchorSlots * d.hidden * sizeof(float), &d.anchors) != 0 ||
        flyweight_gpu_alloc(d.workspace_bytes, &d.workspace) != 0)
        throw std::runtime_error("failed to allocate the DFlash drafter's device state");
    if (flyweight_gpu_memset(d.ring_k, 0, ring, runtime.stream) != 0 ||
        flyweight_gpu_memset(d.ring_v, 0, ring, runtime.stream) != 0)
        throw std::runtime_error("DFlash ring clear failed");
    qwen_dflash_invalidate(runtime);
    const auto mib = [](std::uint64_t bytes) {
        return static_cast<unsigned long long>((bytes + (1ull << 20) - 1) >> 20);
    };
    std::fprintf(stderr,
        "[flyweight] DFlash drafter: %zu blocks, %u drafts, window %u, ring %llu MiB, "
        "workspace %llu MiB\n",
        d.blocks.size(), d.drafts, d.window, mib(2 * ring), mib(d.workspace_bytes));
}

void qwen_dflash_release(FlyweightV2QwenRuntime& runtime) {
    if (!runtime.dflash) return;
    auto& d = *runtime.dflash;
    for (auto* pointer : {&d.ring_k, &d.ring_v, &d.ring_tags, &d.anchors, &d.workspace}) {
        if (*pointer) flyweight_gpu_free(*pointer);
        *pointer = 0;
    }
}

// After a target forward: inject the captured rows into the context ring.
// Enqueued on the runtime stream; the next draft is ordered after it.
void qwen_dflash_inject(FlyweightV2QwenRuntime& runtime) {
    if (!runtime.dflash || !runtime.dflash->workspace || !runtime.capture_rows) return;
    auto& d = *runtime.dflash;
    const auto stream = runtime.stream;
    const auto w = dflash_layout(d, runtime.model->config.vocabulary_size);
    const int total = static_cast<int>(runtime.capture_rows);
    const int first_position = static_cast<int>(runtime.capture_position);
    // Rows older than the window behind this forward's last row can never be
    // read by a later draft.
    const int stored = std::min<int>(total, static_cast<int>(d.inject_capacity));
    const int skipped = total - stored;
    const int first = first_position + skipped;
    const int hidden = static_cast<int>(d.hidden);
    const int kvw = static_cast<int>(d.kv_heads * d.head_dim);
    const std::uint64_t input = runtime.capture_features +
        static_cast<std::uint64_t>(skipped) * d.features * sizeof(float);
    const std::uint64_t g = d.workspace + w.g;
    dflash_matmul(runtime, d, w, d.fc, input, g, static_cast<int>(d.features), hidden, stored,
                  stream);
    dflash_rms(runtime, d, g, d.enc_norm, g, stored, hidden, stream);
    {
        // The anchor ring keeps the newest rows; everything this forward
        // rewrote past them is forgotten.
        const int kept = std::min<int>(stored, static_cast<int>(kDflashAnchorSlots));
        for (auto& position : d.anchor_positions)
            if (position >= first_position) position = -1;
        std::uint64_t source = g + static_cast<std::uint64_t>(stored - kept) * hidden * sizeof(float);
        std::uint64_t anchors = d.anchors;
        int base = first + stored - kept, slots = static_cast<int>(kDflashAnchorSlots);
        void* args[] = {&source, &source, &anchors, &anchors, const_cast<int*>(&hidden), &base,
                        &slots};
        dflash_launch("dflash_ring_store", (hidden + 255) / 256, static_cast<std::uint32_t>(kept),
                      256, args, stream);
        for (int i = 0; i < kept; ++i)
            d.anchor_positions[(base + i) % kDflashAnchorSlots] = base + i;
    }
    {
        std::uint64_t tags = d.ring_tags;
        int slots = static_cast<int>(d.ring_slots), from = first_position, claim = first,
            rows = stored;
        void* args[] = {&tags, &slots, &from, &claim, &rows};
        dflash_launch("dflash_ring_tags", (slots + 255) / 256, 1, 256, args, stream);
    }
    const std::uint64_t k = d.workspace + w.inject_k, v = d.workspace + w.inject_v;
    const std::uint64_t layer_bytes = static_cast<std::uint64_t>(d.ring_slots) * kvw * sizeof(float);
    for (std::size_t l = 0; l < d.blocks.size(); ++l) {
        const auto& b = d.blocks[l];
        dflash_matmul(runtime, d, w, b.k, g, k, hidden, kvw, stored, stream);
        dflash_matmul(runtime, d, w, b.v, g, v, hidden, kvw, stored, stream);
        dflash_head_norm(runtime, d, k, b.k_norm, stored, static_cast<int>(d.kv_heads), first,
                         stream);
        std::uint64_t ring_k = d.ring_k + l * layer_bytes, ring_v = d.ring_v + l * layer_bytes;
        std::uint64_t kk = k, vv = v;
        int width = kvw, base = first, slots = static_cast<int>(d.ring_slots);
        void* args[] = {&kk, &vv, &ring_k, &ring_v, &width, &base, &slots};
        dflash_launch("dflash_ring_store", (width + 255) / 256, static_cast<std::uint32_t>(stored),
                      256, args, stream);
    }
    d.injected_rows += static_cast<std::uint64_t>(stored);
}

// One draft block at the runtime's position. Returns up to `wanted` drafts in
// `out`, or 0 when the anchor row (the previous position's injection) is
// missing -- then the caller decodes one token plainly.
std::uint32_t qwen_dflash_draft(FlyweightV2QwenRuntime& runtime, std::uint32_t id_last,
                                std::uint32_t wanted, std::uint32_t* out) {
    if (!runtime.dflash || !runtime.dflash->workspace || !wanted) return 0;
    auto& d = *runtime.dflash;
    const auto started = std::chrono::steady_clock::now();
    const int pos0 = static_cast<int>(runtime.position);
    if (pos0 < 1 || d.anchor_positions[(pos0 - 1) % kDflashAnchorSlots] != pos0 - 1) return 0;
    wanted = std::min(wanted, d.drafts);
    const auto stream = runtime.stream;
    const auto w = dflash_layout(d, runtime.model->config.vocabulary_size);
    const auto at = [&](std::uint64_t offset) { return d.workspace + offset; };
    const int rows = static_cast<int>(d.block), hidden = static_cast<int>(d.hidden);
    const int heads = static_cast<int>(d.heads), kv_heads = static_cast<int>(d.kv_heads);
    const int head_dim = static_cast<int>(d.head_dim);
    const int qw = heads * head_dim, kvw = kv_heads * head_dim;
    const int coeff_width = 4 * hidden / static_cast<int>(d.conv_group);
    std::array<std::uint32_t, 9> tokens{};
    tokens[0] = id_last;
    for (int i = 1; i < rows; ++i) tokens[i] = d.mask_token;
    const std::uint64_t token_device = at(w.tokens);
    if (flyweight_gpu_upload(token_device, tokens.data(), rows * sizeof(std::uint32_t), stream) != 0)
        throw std::runtime_error("DFlash block token upload failed");
    const std::uint64_t h = at(w.h), a = at(w.a), a2 = at(w.a2), q = at(w.q), k = at(w.k),
        v = at(w.v), attended = at(w.attended), projected = at(w.projected), r = at(w.r),
        coeff = at(w.coeff), gate = at(w.gate), up = at(w.up), y = at(w.y), z = at(w.z);
    dflash_embed(runtime, tokens.data(), token_device, rows, h);
    const std::uint64_t layer_bytes = static_cast<std::uint64_t>(d.ring_slots) * kvw * sizeof(float);
    for (std::size_t l = 0; l < d.blocks.size(); ++l) {
        const auto& b = d.blocks[l];
        dflash_rms(runtime, d, h, b.attn_norm, a, rows, hidden, stream);
        dflash_matmul(runtime, d, w, b.attn_conv_proj, a, coeff, hidden, coeff_width, rows, stream);
        dflash_conv(runtime, d, a, coeff, b.attn_conv_base, 0, a2, 0, stream);
        dflash_matmul(runtime, d, w, b.q, a2, q, hidden, qw, rows, stream);
        dflash_matmul(runtime, d, w, b.k, a2, k, hidden, kvw, rows, stream);
        dflash_matmul(runtime, d, w, b.v, a2, v, hidden, kvw, rows, stream);
        dflash_head_norm(runtime, d, q, b.q_norm, rows, heads, pos0, stream);
        dflash_head_norm(runtime, d, k, b.k_norm, rows, kv_heads, pos0, stream);
        {
            std::uint64_t qq = q, kk = k, vv = v, ring_k = d.ring_k + l * layer_bytes,
                ring_v = d.ring_v + l * layer_bytes, tags = d.ring_tags,
                sinks = runtime.device_tensors[b.sinks], o = attended;
            int r_rows = rows, h_heads = heads, h_kv = kv_heads, h_dim = head_dim,
                slots = static_cast<int>(d.ring_slots), position = pos0,
                window = static_cast<int>(d.window);
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            void* args[] = {&qq, &kk, &vv, &ring_k, &ring_v, &tags, &sinks, &o, &r_rows,
                            &h_heads, &h_kv, &h_dim, &slots, &position, &window, &scale};
            dflash_launch("dflash_block_attention", static_cast<std::uint32_t>(heads),
                          static_cast<std::uint32_t>(rows), 128, args, stream,
                          static_cast<std::uint32_t>((d.ring_slots + d.block) * sizeof(float)));
        }
        dflash_matmul(runtime, d, w, b.o, attended, projected, qw, hidden, rows, stream);
        dflash_conv(runtime, d, projected, coeff, b.attn_conv_base, h, r, 1, stream);
        dflash_rms(runtime, d, r, b.ffn_norm, a, rows, hidden, stream);
        dflash_matmul(runtime, d, w, b.ffn_conv_proj, a, coeff, hidden, coeff_width, rows, stream);
        dflash_conv(runtime, d, a, coeff, b.ffn_conv_base, 0, a2, 0, stream);
        const int ffn = static_cast<int>(d.ffn);
        dflash_matmul(runtime, d, w, b.gate, a2, gate, hidden, ffn, rows, stream);
        dflash_matmul(runtime, d, w, b.up, a2, up, hidden, ffn, rows, stream);
        {
            std::uint64_t g = gate, u = up, o = gate;
            int count = rows * ffn;
            void* args[] = {&g, &u, &o, &count};
            dflash_launch("dflash_swiglu", (count + 255) / 256, 1, 256, args, stream);
        }
        dflash_matmul(runtime, d, w, b.down, gate, y, ffn, hidden, rows, stream);
        dflash_conv(runtime, d, y, coeff, b.ffn_conv_base, r, h, 1, stream);
    }
    dflash_rms(runtime, d, h, d.output_norm, z, rows, hidden, stream);
    const int vocabulary = static_cast<int>(runtime.model->config.vocabulary_size);
    const int slots = static_cast<int>(d.slots), topk = static_cast<int>(d.topk);
    const int nodes = static_cast<int>(d.nodes), width = static_cast<int>(d.width);
    const std::uint64_t z_rows = z + static_cast<std::uint64_t>(hidden) * sizeof(float);
    const std::uint64_t logits = at(w.logits);
    dflash_matmul(runtime, d, w, runtime.lm_head, z_rows, logits, hidden, vocabulary, slots,
                  stream);
    const std::uint64_t ids = at(w.ids), logprobs = at(w.logprobs);
    {
        // Two stages: kDflashTopkChunks blocks per row keep a top-k and a
        // partial log-sum-exp each, one block per row merges them.
        std::uint64_t x = logits, i = ids, lp = logprobs, partial = at(w.topk_partial);
        int vocab = vocabulary, kk = topk, chunks = static_cast<int>(kDflashTopkChunks);
        void* chunk_args[] = {&x, &vocab, &kk, &chunks, &partial};
        dflash_launch("dflash_topk_chunks", kDflashTopkChunks, static_cast<std::uint32_t>(slots),
                      256, chunk_args, stream);
        void* merge_args[] = {&partial, &kk, &chunks, &i, &lp};
        dflash_launch("dflash_topk_merge", static_cast<std::uint32_t>(slots), 1, 256, merge_args,
                      stream);
    }
    const std::uint64_t features = at(w.features), fs = at(w.fs);
    {
        std::uint64_t lp = logprobs, nw = runtime.device_tensors[d.feature_norm],
            nb = runtime.device_tensors[d.feature_norm_bias],
            uw = runtime.device_tensors[d.feature_up],
            ub = runtime.device_tensors[d.feature_up_bias], o = features;
        int kk = topk, ww = width;
        void* args[] = {&lp, &nw, &nb, &uw, &ub, &o, &kk, &ww};
        dflash_launch("dflash_lilicorr_features", static_cast<std::uint32_t>(nodes), 1, 256, args,
                      stream);
    }
    dflash_matmul(runtime, d, w, d.feature_down, features, fs, width, width, nodes, stream);
    const std::uint64_t embedded = at(w.embedded), tok = at(w.tok), pass = at(w.pass);
    dflash_embed(runtime, nullptr, ids, nodes, embedded);
    dflash_matmul(runtime, d, w, d.token_proj, embedded, tok, hidden, width, nodes, stream);
    dflash_matmul(runtime, d, w, d.pass_proj, z_rows, pass, hidden, width, slots, stream);
    const std::uint64_t x = at(w.x);
    {
        std::uint64_t xx = x, t = tok, tb = runtime.device_tensors[d.token_proj_bias], p = pass,
            pb = runtime.device_tensors[d.pass_proj_bias], f = fs,
            fb = runtime.device_tensors[d.feature_down_bias],
            se = runtime.device_tensors[d.slot_embedding],
            re = runtime.device_tensors[d.rank_embedding];
        int kk = topk, ww = width;
        void* args[] = {&xx, &t, &tb, &p, &pb, &f, &fb, &se, &re, &kk, &ww};
        dflash_launch("dflash_lilicorr_input", (width + 255) / 256,
                      static_cast<std::uint32_t>(nodes), 256, args, stream);
    }
    const std::uint64_t anchor = at(w.anchor), anchor_normed = at(w.anchor_normed);
    {
        const std::uint64_t anchor_row = d.anchors +
            static_cast<std::uint64_t>((pos0 - 1) % kDflashAnchorSlots) * hidden * sizeof(float);
        dflash_matmul(runtime, d, w, d.context_proj, anchor_row, anchor, hidden, width, 1, stream);
        dflash_bias(runtime, anchor, d.context_proj_bias, 1, width, 0, stream);
    }
    const std::uint64_t normed = at(w.normed), qkv = at(w.qkv), mixed = at(w.mixed),
        delta = at(w.delta), mlp = at(w.mlp);
    const int corrector_head_dim = width / static_cast<int>(d.corrector_heads);
    for (const auto& c : d.correctors) {
        dflash_rms(runtime, d, x, c.attn_norm, normed, nodes, width, stream);
        dflash_matmul(runtime, d, w, c.in_proj, normed, qkv, width, 3 * width, nodes, stream);
        dflash_bias(runtime, qkv, c.in_proj_bias, nodes, 3 * width, 0, stream);
        {
            std::uint64_t in = qkv, bias = runtime.device_tensors[d.attention_bias], o = mixed;
            int n = nodes, hh = static_cast<int>(d.corrector_heads), hd = corrector_head_dim;
            float scale = 1.0f / std::sqrt(static_cast<float>(corrector_head_dim));
            void* args[] = {&in, &bias, &o, &n, &hh, &hd, &scale};
            dflash_launch("dflash_node_attention", d.corrector_heads,
                          static_cast<std::uint32_t>(nodes), 64, args, stream);
        }
        dflash_matmul(runtime, d, w, c.out, mixed, delta, width, width, nodes, stream);
        {
            std::uint64_t t = x, dd = delta, b = runtime.device_tensors[c.out_bias];
            int ww = width;
            void* args[] = {&t, &dd, &b, &ww};
            dflash_launch("dflash_add_rows", (width + 255) / 256, static_cast<std::uint32_t>(nodes),
                          256, args, stream);
        }
        dflash_rms(runtime, d, x, c.ffn_norm, normed, nodes, width, stream);
        const int mlp_width = static_cast<int>(d.mlp);
        dflash_matmul(runtime, d, w, c.up, normed, mlp, width, mlp_width, nodes, stream);
        dflash_bias(runtime, mlp, c.up_bias, nodes, mlp_width, 1, stream);
        dflash_matmul(runtime, d, w, c.down, mlp, delta, mlp_width, width, nodes, stream);
        {
            std::uint64_t t = x, dd = delta, b = runtime.device_tensors[c.down_bias];
            int ww = width;
            void* args[] = {&t, &dd, &b, &ww};
            dflash_launch("dflash_add_rows", (width + 255) / 256, static_cast<std::uint32_t>(nodes),
                          256, args, stream);
        }
    }
    dflash_rms(runtime, d, x, d.corrector_norm, normed, nodes, width, stream);
    dflash_rms(runtime, d, anchor, d.anchor_norm, anchor_normed, 1, width, stream);
    const std::uint64_t concat = at(w.concat), phi = at(w.phi), out_factor = at(w.out_factor),
        in_factor = at(w.in_factor), anchor_factor = at(w.anchor_factor), scores = at(w.scores);
    {
        std::uint64_t xx = normed, an = anchor_normed, o = concat;
        int ww = width;
        void* args[] = {&xx, &an, &o, &ww};
        dflash_launch("dflash_factor_concat", (width + 255) / 256,
                      static_cast<std::uint32_t>(nodes), 256, args, stream);
    }
    const int factor = static_cast<int>(d.factor);
    dflash_matmul(runtime, d, w, d.factor_input, concat, phi, 3 * width, factor, nodes, stream);
    dflash_bias(runtime, phi, d.factor_input_bias, nodes, factor, 1, stream);
    dflash_matmul(runtime, d, w, d.out_head, phi, out_factor, factor, factor, nodes, stream);
    dflash_bias(runtime, out_factor, d.out_head_bias, nodes, factor, 0, stream);
    dflash_matmul(runtime, d, w, d.in_head, phi, in_factor, factor, factor, nodes, stream);
    dflash_bias(runtime, in_factor, d.in_head_bias, nodes, factor, 0, stream);
    dflash_matmul(runtime, d, w, d.anchor_head, anchor_normed, anchor_factor, width, factor, 1,
                  stream);
    dflash_bias(runtime, anchor_factor, d.anchor_head_bias, 1, factor, 0, stream);
    {
        float epsilon = d.vector_eps;
        int ff = factor;
        std::uint64_t o = out_factor, i = in_factor, a0 = anchor_factor;
        void* out_args[] = {&o, &ff, &epsilon};
        dflash_launch("dflash_l2_normalize_rows", static_cast<std::uint32_t>(nodes), 1, 256,
                      out_args, stream);
        void* in_args[] = {&i, &ff, &epsilon};
        dflash_launch("dflash_l2_normalize_rows", static_cast<std::uint32_t>(nodes), 1, 256,
                      in_args, stream);
        void* anchor_args[] = {&a0, &ff, &epsilon};
        dflash_launch("dflash_l2_normalize_rows", 1, 1, 256, anchor_args, stream);
    }
    // Slot 0 fills only its first predecessor row; the rest stays zero.
    if (flyweight_gpu_memset(scores, 0, static_cast<std::uint64_t>(slots) * topk * topk * sizeof(float),
                             stream) != 0)
        throw std::runtime_error("DFlash lattice reset failed");
    {
        std::uint64_t i = in_factor, o = out_factor, a0 = anchor_factor, s = scores;
        int kk = topk, ff = factor;
        float scale = d.logit_scale;
        void* args[] = {&i, &o, &a0, &s, &kk, &ff, &scale};
        dflash_launch("dflash_lattice_scores",
                      static_cast<std::uint32_t>(topk + (slots - 1) * topk * topk), 1, 128, args,
                      stream);
    }
    std::vector<std::int32_t> candidates(nodes);
    std::vector<float> lattice(static_cast<std::size_t>(slots) * topk * topk);
    if (flyweight_gpu_download(candidates.data(), ids, nodes * sizeof(std::int32_t), stream) != 0 ||
        flyweight_gpu_download(lattice.data(), scores, lattice.size() * sizeof(float), stream) != 0 ||
        flyweight_gpu_stream_sync(stream) != 0)
        throw std::runtime_error("DFlash lattice download failed");
    // Greedy walk (speculative.cpp): slot 0 scores against the anchor, each
    // later slot against the candidate picked before it; the first maximum wins.
    int previous = 0;
    for (std::uint32_t i = 0; i < wanted; ++i) {
        const float* row = lattice.data() + static_cast<std::size_t>(i) * topk * topk +
                           (i == 0 ? 0 : previous) * topk;
        int best = 0;
        for (int j = 1; j < topk; ++j)
            if (row[j] > row[best]) best = j;
        previous = best;
        out[i] = static_cast<std::uint32_t>(candidates[i * topk + best]);
    }
    ++d.rounds;
    d.drafted += wanted;
    if (dflash_profile_enabled() && d.rounds % 32 == 0) {
        static std::uint64_t window_start = 0;
        std::fprintf(stderr, "[dflash] %.2f ms wall per draft over the last 32\n",
                     (d.draft_nanoseconds - window_start) / 32.0 / 1e6);
        window_start = d.draft_nanoseconds;
        dflash_profile_report();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    d.draft_nanoseconds += static_cast<std::uint64_t>(elapsed);
    runtime.mtp_draft_nanoseconds += static_cast<std::uint64_t>(elapsed);
    return wanted;
}

// Diagnostics and tests: one draft at the runtime's position, with the draft
// block's logits (rows 1..block-1), the LiLiCorr candidates and the lattice
// scores read back. Returns the walk's draft count (0: no anchor row).
std::uint32_t qwen_dflash_probe(FlyweightV2QwenRuntime& runtime, std::uint32_t token,
                                float* logits, std::int32_t* candidates, float* scores,
                                std::uint32_t* drafts) {
    if (!runtime.dflash) throw std::runtime_error("no DFlash drafter is attached");
    auto& d = *runtime.dflash;
    const auto count = qwen_dflash_draft(runtime, token, d.drafts, drafts);
    if (!count) return 0;
    const auto w = dflash_layout(d, runtime.model->config.vocabulary_size);
    const std::uint64_t vocabulary = runtime.model->config.vocabulary_size;
    if (flyweight_gpu_download(logits, d.workspace + w.logits,
                               d.slots * vocabulary * sizeof(float), runtime.stream) != 0 ||
        flyweight_gpu_download(candidates, d.workspace + w.ids, d.nodes * sizeof(std::int32_t),
                               runtime.stream) != 0 ||
        flyweight_gpu_download(scores, d.workspace + w.scores,
                               static_cast<std::uint64_t>(d.slots) * d.topk * d.topk * sizeof(float),
                               runtime.stream) != 0 ||
        flyweight_gpu_stream_sync(runtime.stream) != 0)
        throw std::runtime_error("DFlash probe download failed");
    return count;
}
