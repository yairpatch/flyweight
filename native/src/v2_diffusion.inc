// Diffusion tower: Z-Image-Turbo end to end on the Qwen kernel corpus.
// Included by v2_runtime.cpp after the model and runtime structs, beside
// v2_vision.inc, and shaped like it: weights uploaded once at create, one
// workspace arena carved per phase, every kernel launched by name.
//
// Three components, each a FlyweightV2Model opened from its diffusers
// directory (see hf::config_from_zimage_transformer and friends):
//   * the text encoder, a plain Qwen3 decoder run without a KV cache and
//     stopped one layer early -- the DiT conditions on hidden_states[-2];
//   * the single-stream DiT (ZImageTransformer2DModel);
//   * the KL autoencoder, of which only the decoder runs.
// The reference is diffusers' pipeline_z_image.py / transformer_z_image.py;
// tools/zimage_reference.py dumps it, tools/check_zimage_parity.py compares.

namespace {

// A [out][in] weight as uploaded: quantized bytes straight from the arena, or
// f32/bf16 as stored. `bias` is an f32 device vector or zero.
struct DiffMatrix {
    std::uint64_t device = 0;
    std::uint32_t type = 0;
    int in = 0, out = 0;
    std::uint64_t bias = 0;
    // Host-resident mode: the bytes live in pinned host memory at `host` and
    // are staged into a device slot at `slot_offset` before the layer runs;
    // `device` then points into the slot for the duration of that layer.
    const void* host = nullptr;
    std::uint64_t bytes = 0, slot_offset = 0;
};

struct DiffBlock {
    std::uint64_t attention_norm1 = 0, attention_norm2 = 0, ffn_norm1 = 0, ffn_norm2 = 0;
    std::uint64_t q_norm = 0, k_norm = 0;
    DiffMatrix qkv, out, w1, w2, w3, adaln;
    bool modulated = false;
};

struct DiffEncoderLayer {
    std::uint64_t attn_norm = 0, ffn_norm = 0, q_norm = 0, k_norm = 0;
    DiffMatrix q, k, v, o, gate, up, down;
};

struct VaeConv {
    std::uint64_t weight = 0, bias = 0;
    int in = 0, out = 0, kernel = 3;
    // bf16 weights packed [tap][out][in/16][slot] for diff_conv2d_bf16; zero
    // when the shape does not fit the tensor-core kernel.
    std::uint64_t weight_bf16 = 0;
    bool tensor_core() const { return weight_bf16 != 0; }
};

// Slot of channel j within its block of 16 (see diff_channel_slot).
inline int diff_channel_slot_host(int j) { return ((j & 7) >> 1) * 4 + (j >> 3) * 2 + (j & 1); }

inline std::uint16_t diff_bf16_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) == 0x7f800000u) return static_cast<std::uint16_t>(bits >> 16);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

struct VaeNorm {
    std::uint64_t weight = 0, bias = 0;
};

struct VaeResnet {
    VaeNorm norm1, norm2;
    VaeConv conv1, conv2, shortcut;
    int in = 0, out = 0;
};

struct VaeAttention {
    VaeNorm norm;
    DiffMatrix q, k, v, o;
};

constexpr int kDiffSequenceMultiple = 32;   // SEQ_MULTI_OF
constexpr int kDiffAdalnWidth = 256;        // ADALN_EMBED_DIM
constexpr int kDiffTimeMlpWidth = 1024;     // TimestepEmbedder mid_size
constexpr int kVaeAttentionChunk = 1024;    // query rows per score tile
// Tiled decode, as diffusers' AutoencoderKL.enable_tiling: latents wider than
// 64 decode in 64x64 tiles at a stride of 48 and the 128-pixel overlaps are
// blended with linear ramps. Keeps the decoder's peak activation at the
// 512x512 figure whatever the output size.
constexpr int kVaeTileLatent = 64;
constexpr int kVaeTileStride = 48;
constexpr int kVaeBlendPixels = 128;

}  // namespace

struct FlyweightV2Diffusion {
    FlyweightV2Model* encoder = nullptr;      // borrowed
    FlyweightV2Model* transformer = nullptr;  // borrowed
    FlyweightV2Model* vae = nullptr;          // borrowed
    std::int32_t device = 0;
    std::uint64_t stream = 0;
    std::vector<std::uint64_t> allocations;
    std::uint64_t weight_bytes = 0;
    bool int8_tensor_cores = false;
    // bf16 mma attention (sm_80+, head_dim 128); the f32 kernel otherwise.
    bool flash_attention = false;
    // Exact mode: f32 activations through every GEMM (no Q8 activation
    // quantization), f32 attention and f32 convolutions. Several times
    // slower; the weights stay quantized as loaded.
    bool exact = false;
    // Balanced mode: Q8_0 weights dequantized to bf16 inside a tensor-core
    // GEMM with bf16 activations (diffusers' numerics), bf16 attention and
    // convolutions. Between fast and exact in both cost and error.
    bool balanced = false;

    // Text encoder.
    const Tensor* token_embedding = nullptr;
    std::vector<DiffEncoderLayer> encoder_layers;
    int encoder_hidden = 0, encoder_heads = 0, encoder_kv_heads = 0, encoder_head_dim = 0;
    int encoder_intermediate = 0;
    float encoder_epsilon = 1e-6f, encoder_theta = 1e6f;

    // DiT.
    DiffMatrix x_embed, cap_embed, t_mlp0, t_mlp1, final_adaln, final_linear;
    std::uint64_t cap_norm = 0, x_pad = 0, cap_pad = 0;
    std::vector<DiffBlock> noise_refiner, context_refiner, layers;
    int dim = 0, heads = 0, head_dim = 0, intermediate = 0, patch = 2, channels = 16;
    int caption_dim = 0;
    float epsilon = 1e-5f, rope_theta = 256.0f, time_scale = 1000.0f;
    int axis_dims[3] = {0, 0, 0};
    // Rope ramps per pair, doubles on the device (see diff_rope_axes_rows).
    std::uint64_t rope_frequencies = 0, encoder_rope_frequencies = 0;

    // Autoencoder decoder.
    VaeConv conv_in, conv_out;
    VaeNorm norm_out;
    VaeResnet mid_resnet[2];
    VaeAttention mid_attention;
    bool mid_has_attention = false;
    std::vector<std::vector<VaeResnet>> up_resnets;
    std::vector<VaeConv> upsamplers;   // one per non-final level
    int vae_channels[4] = {0, 0, 0, 0};
    int vae_groups = 32, latent_channels = 16;
    float latent_scale = 1.0f, latent_shift = 0.0f;

    // Limits the workspace was sized for.
    std::uint32_t max_width = 0, max_height = 0, max_prompt_tokens = 0;
    std::uint64_t workspace = 0, workspace_bytes = 0;
    std::uint64_t q8_capacity_rows = 0;

    // Host-resident weights: the encoder's and DiT's layer matrices stay in
    // pinned host memory and stream through two device slots, one layer
    // ahead of compute on `copy_stream`. Everything else (norms, embedders,
    // the VAE) stays on the device. Chosen when the tower would otherwise
    // take most of the card, so a chat model can still be planned beside it.
    bool host_weights = false;
    std::vector<void*> host_allocations;
    std::uint64_t host_bytes = 0;
    std::uint64_t copy_stream = 0;
    std::uint64_t staging[2] = {0, 0};
    std::uint64_t staged_event[2] = {0, 0}, consumed_event[2] = {0, 0};
    std::uint64_t staging_bytes = 0;   // the largest layer, which sizes a slot
    // Set while a layer's matrices are being uploaded at create: offsets
    // within the slot are dealt from this cursor.
    bool grouping = false;
    std::uint64_t group_cursor = 0;
};

namespace {

std::uint64_t diff_alloc(FlyweightV2Diffusion& d, std::uint64_t bytes) {
    std::uint64_t pointer = 0;
    if (flyweight_gpu_alloc(bytes, &pointer) != 0 || !pointer)
        throw std::runtime_error("diffusion tower device allocation failed (" +
                                 std::to_string(bytes / 1048576) + " MiB)");
    d.allocations.push_back(pointer);
    d.weight_bytes += bytes;
    return pointer;
}

const Tensor& diff_tensor(const FlyweightV2Model& model, const std::string& name) {
    for (const auto& tensor : model.tensors)
        if (tensor.name == name) return tensor;
    throw std::runtime_error(model.path + " is missing tensor " + name);
}

bool diff_has_tensor(const FlyweightV2Model& model, const std::string& name) {
    for (const auto& tensor : model.tensors)
        if (tensor.name == name) return true;
    return false;
}

std::uint64_t diff_align(std::uint64_t bytes) { return (bytes + 255) & ~std::uint64_t(255); }

std::uint64_t diff_elements(const Tensor& tensor) {
    std::uint64_t count = 1;
    for (const auto dim : tensor.shape) count *= dim;
    return count;
}

// Raw upload of a tensor's arena bytes.
std::uint64_t diff_upload_raw(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                              const Tensor& tensor) {
    const std::uint64_t pointer = diff_alloc(d, tensor.size);
    if (flyweight_gpu_upload_sync(pointer, tensor_data(model, tensor), tensor.size) != 0)
        throw std::runtime_error("diffusion tower upload failed for " + tensor.name);
    return pointer;
}

// An f32 device vector, widened from whatever the arena stores.
std::uint64_t diff_upload_f32(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                              const std::string& name, std::uint64_t expected) {
    const Tensor& tensor = diff_tensor(model, name);
    const std::uint64_t elements = diff_elements(tensor);
    if (expected && elements != expected)
        throw std::runtime_error(name + " has " + std::to_string(elements) +
                                 " elements, expected " + std::to_string(expected));
    std::vector<float> widened(elements);
    const auto* source = tensor_data(model, tensor);
    for (std::uint64_t i = 0; i < elements; ++i) widened[i] = tensor_value(source, tensor.type, i);
    const std::uint64_t pointer = diff_alloc(d, elements * sizeof(float));
    if (flyweight_gpu_upload_sync(pointer, widened.data(), elements * sizeof(float)) != 0)
        throw std::runtime_error("diffusion tower upload failed for " + name);
    return pointer;
}

// A weight matrix: [in][out] in GGUF order, a stacked one [in][out][pieces].
DiffMatrix diff_upload_matrix(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                              const std::string& name, int in, int out,
                              const std::string& bias_name = {}) {
    const Tensor& tensor = diff_tensor(model, name);
    if (tensor.shape.size() < 2 || tensor.shape[0] != static_cast<std::uint64_t>(in))
        throw std::runtime_error(name + " does not have input width " + std::to_string(in));
    std::uint64_t rows = tensor.shape[1];
    for (std::size_t axis = 2; axis < tensor.shape.size(); ++axis) rows *= tensor.shape[axis];
    if (rows != static_cast<std::uint64_t>(out))
        throw std::runtime_error(name + " does not have output width " + std::to_string(out));
    DiffMatrix matrix;
    matrix.type = tensor.type;
    matrix.in = in;
    matrix.out = out;
    if (matrix.type != 0 && matrix.type != 30 && !flyweight::v2::qwen_format(matrix.type))
        throw std::runtime_error(name + " is stored in a format without GEMM kernels");
    if (d.host_weights && d.grouping) {
        void* pinned = nullptr;
        if (flyweight_gpu_host_alloc(tensor.size, &pinned) != 0 || !pinned)
            throw std::runtime_error("diffusion tower pinned host allocation failed for " + name);
        d.host_allocations.push_back(pinned);
        d.host_bytes += tensor.size;
        std::memcpy(pinned, tensor_data(model, tensor), tensor.size);
        matrix.host = pinned;
        matrix.bytes = tensor.size;
        matrix.slot_offset = d.group_cursor;
        d.group_cursor += diff_align(tensor.size);
    } else {
        matrix.device = diff_upload_raw(d, model, tensor);
    }
    if (!bias_name.empty()) matrix.bias = diff_upload_f32(d, model, bias_name, out);
    return matrix;
}

// Brackets the upload of one layer's matrices so they share a staging slot.
struct DiffGroup {
    FlyweightV2Diffusion& d;
    explicit DiffGroup(FlyweightV2Diffusion& tower) : d(tower) {
        d.grouping = true;
        d.group_cursor = 0;
    }
    ~DiffGroup() {
        d.grouping = false;
        d.staging_bytes = std::max(d.staging_bytes, d.group_cursor);
    }
};

// The matrices a layer streams, in slot order.
struct DiffLayerRef {
    DiffMatrix* matrices[7] = {};
    int count = 0;
    void add(DiffMatrix& m) { matrices[count++] = &m; }
};

VaeConv diff_upload_conv(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                         const std::string& prefix, int in, int out, int kernel) {
    const Tensor& tensor = diff_tensor(model, prefix + ".weight");
    if (tensor.type != 0)
        throw std::runtime_error(prefix + ".weight must be f32 (got type " +
                                 std::to_string(tensor.type) + ")");
    const std::uint64_t expected = static_cast<std::uint64_t>(out) * in * kernel * kernel;
    if (diff_elements(tensor) != expected)
        throw std::runtime_error(prefix + ".weight has the wrong shape");
    VaeConv conv;
    conv.in = in;
    conv.out = out;
    conv.kernel = kernel;
    conv.weight = diff_upload_raw(d, model, tensor);
    conv.bias = diff_upload_f32(d, model, prefix + ".bias", out);
    if (d.flash_attention && !d.exact && in % 16 == 0 && out % 64 == 0) {
        // The same mma path as the attention; pack [out][in][k][k] f32 into
        // [tap][out][in/16][16 slots] bf16.
        const float* source = reinterpret_cast<const float*>(tensor_data(model, tensor));
        const int taps = kernel * kernel, blocks = in / 16;
        std::vector<std::uint16_t> packed(static_cast<std::size_t>(taps) * out * in);
        for (int co = 0; co < out; ++co)
            for (int ci = 0; ci < in; ++ci)
                for (int tap = 0; tap < taps; ++tap)
                    packed[(static_cast<std::size_t>(tap) * out + co) * in + (ci / 16) * 16 + diff_channel_slot_host(ci % 16)] =
                        diff_bf16_bits(source[(static_cast<std::size_t>(co) * in + ci) * taps + tap]);
        (void)blocks;
        conv.weight_bf16 = diff_alloc(d, packed.size() * 2);
        if (flyweight_gpu_upload_sync(conv.weight_bf16, packed.data(), packed.size() * 2) != 0)
            throw std::runtime_error("diffusion tower upload failed for " + prefix + " (bf16 pack)");
    }
    return conv;
}

VaeNorm diff_upload_norm(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                         const std::string& prefix, int channels) {
    VaeNorm norm;
    norm.weight = diff_upload_f32(d, model, prefix + ".weight", channels);
    norm.bias = diff_upload_f32(d, model, prefix + ".bias", channels);
    return norm;
}

VaeResnet diff_upload_resnet(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                             const std::string& prefix, int in, int out) {
    VaeResnet resnet;
    resnet.in = in;
    resnet.out = out;
    resnet.norm1 = diff_upload_norm(d, model, prefix + ".norm1", in);
    resnet.conv1 = diff_upload_conv(d, model, prefix + ".conv1", in, out, 3);
    resnet.norm2 = diff_upload_norm(d, model, prefix + ".norm2", out);
    resnet.conv2 = diff_upload_conv(d, model, prefix + ".conv2", out, out, 3);
    if (in != out) resnet.shortcut = diff_upload_conv(d, model, prefix + ".conv_shortcut", in, out, 1);
    return resnet;
}

DiffLayerRef diff_block_matrices(DiffBlock& block) {
    DiffLayerRef ref;
    ref.add(block.qkv);
    ref.add(block.out);
    ref.add(block.w1);
    ref.add(block.w3);
    ref.add(block.w2);
    if (block.modulated) ref.add(block.adaln);
    return ref;
}

DiffLayerRef diff_encoder_matrices(DiffEncoderLayer& layer) {
    DiffLayerRef ref;
    ref.add(layer.q);
    ref.add(layer.k);
    ref.add(layer.v);
    ref.add(layer.o);
    ref.add(layer.gate);
    ref.add(layer.up);
    ref.add(layer.down);
    return ref;
}

DiffBlock diff_upload_block(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                            const std::string& prefix, bool modulated) {
    DiffGroup group(d);
    DiffBlock block;
    block.modulated = modulated;
    const int dim = d.dim;
    block.attention_norm1 = diff_upload_f32(d, model, prefix + ".attention_norm1.weight", dim);
    block.attention_norm2 = diff_upload_f32(d, model, prefix + ".attention_norm2.weight", dim);
    block.ffn_norm1 = diff_upload_f32(d, model, prefix + ".ffn_norm1.weight", dim);
    block.ffn_norm2 = diff_upload_f32(d, model, prefix + ".ffn_norm2.weight", dim);
    block.q_norm = diff_upload_f32(d, model, prefix + ".attention.norm_q.weight", d.head_dim);
    block.k_norm = diff_upload_f32(d, model, prefix + ".attention.norm_k.weight", d.head_dim);
    block.qkv = diff_upload_matrix(d, model, prefix + ".attention.qkv.weight", dim, 3 * dim);
    block.out = diff_upload_matrix(d, model, prefix + ".attention.to_out.0.weight", dim, dim);
    block.w1 = diff_upload_matrix(d, model, prefix + ".feed_forward.w1.weight", dim, d.intermediate);
    block.w3 = diff_upload_matrix(d, model, prefix + ".feed_forward.w3.weight", dim, d.intermediate);
    block.w2 = diff_upload_matrix(d, model, prefix + ".feed_forward.w2.weight", d.intermediate, dim);
    if (modulated)
        block.adaln = diff_upload_matrix(d, model, prefix + ".adaLN_modulation.0.weight",
                                         kDiffAdalnWidth, 4 * dim,
                                         prefix + ".adaLN_modulation.0.bias");
    return block;
}

std::uint64_t diff_upload_doubles(FlyweightV2Diffusion& d, const std::vector<double>& values) {
    const std::uint64_t pointer = diff_alloc(d, values.size() * sizeof(double));
    if (flyweight_gpu_upload_sync(pointer, values.data(), values.size() * sizeof(double)) != 0)
        throw std::runtime_error("diffusion tower upload failed for a rope table");
    return pointer;
}

void diff_upload_encoder(FlyweightV2Diffusion& d) {
    const auto& model = *d.encoder;
    const auto& config = model.config;
    d.encoder_hidden = static_cast<int>(config.hidden_size);
    d.encoder_heads = static_cast<int>(config.attention_heads);
    d.encoder_kv_heads = static_cast<int>(config.attention_kv_heads);
    d.encoder_head_dim = static_cast<int>(config.attention_head_dim);
    d.encoder_intermediate = static_cast<int>(config.intermediate_size);
    d.encoder_epsilon = config.rms_norm_epsilon;
    d.encoder_theta = config.rope_freq_base;
    if (d.encoder_head_dim > 128)
        throw std::runtime_error("text encoder head width above 128 is not supported");
    if (d.encoder_hidden != d.caption_dim)
        throw std::runtime_error("text encoder width " + std::to_string(d.encoder_hidden) +
                                 " does not match the DiT's cap_feat_dim " +
                                 std::to_string(d.caption_dim));
    d.token_embedding = &diff_tensor(model, "token_embd.weight");
    // hidden_states[-2] is the residual stream entering the last layer, so
    // the last layer's weights are never read and never uploaded.
    const int used_layers = static_cast<int>(config.layer_count) - 1;
    if (used_layers < 1) throw std::runtime_error("text encoder has too few layers");
    {
        std::vector<double> frequencies(d.encoder_head_dim / 2);
        for (std::size_t j = 0; j < frequencies.size(); ++j)
            frequencies[j] = 1.0 / std::pow(static_cast<double>(d.encoder_theta),
                                            static_cast<double>(2 * j) / d.encoder_head_dim);
        d.encoder_rope_frequencies = diff_upload_doubles(d, frequencies);
    }
    const int q_width = d.encoder_heads * d.encoder_head_dim;
    const int kv_width = d.encoder_kv_heads * d.encoder_head_dim;
    d.encoder_layers.reserve(used_layers);
    for (int index = 0; index < used_layers; ++index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        DiffGroup group(d);
        DiffEncoderLayer layer;
        layer.attn_norm = diff_upload_f32(d, model, prefix + "attn_norm.weight", d.encoder_hidden);
        layer.ffn_norm = diff_upload_f32(d, model, prefix + "post_attention_norm.weight", d.encoder_hidden);
        layer.q_norm = diff_upload_f32(d, model, prefix + "attn_q_norm.weight", d.encoder_head_dim);
        layer.k_norm = diff_upload_f32(d, model, prefix + "attn_k_norm.weight", d.encoder_head_dim);
        layer.q = diff_upload_matrix(d, model, prefix + "attn_q.weight", d.encoder_hidden, q_width);
        layer.k = diff_upload_matrix(d, model, prefix + "attn_k.weight", d.encoder_hidden, kv_width);
        layer.v = diff_upload_matrix(d, model, prefix + "attn_v.weight", d.encoder_hidden, kv_width);
        layer.o = diff_upload_matrix(d, model, prefix + "attn_output.weight", q_width, d.encoder_hidden);
        layer.gate = diff_upload_matrix(d, model, prefix + "ffn_gate.weight", d.encoder_hidden, d.encoder_intermediate);
        layer.up = diff_upload_matrix(d, model, prefix + "ffn_up.weight", d.encoder_hidden, d.encoder_intermediate);
        layer.down = diff_upload_matrix(d, model, prefix + "ffn_down.weight", d.encoder_intermediate, d.encoder_hidden);
        d.encoder_layers.push_back(layer);
    }
}

void diff_upload_transformer(FlyweightV2Diffusion& d) {
    const auto& model = *d.transformer;
    const auto& config = model.config;
    d.dim = static_cast<int>(config.hidden_size);
    d.heads = static_cast<int>(config.attention_heads);
    d.head_dim = static_cast<int>(config.attention_head_dim);
    d.intermediate = static_cast<int>(config.intermediate_size);
    d.patch = static_cast<int>(config.patch_size);
    d.channels = static_cast<int>(config.in_channels);
    d.caption_dim = static_cast<int>(config.caption_dim);
    d.epsilon = config.rms_norm_epsilon;
    d.rope_theta = config.rope_freq_base;
    d.time_scale = config.time_scale;
    for (int axis = 0; axis < 3; ++axis) d.axis_dims[axis] = static_cast<int>(config.rope_sections[axis]);
    if (d.head_dim > 128) throw std::runtime_error("DiT head width above 128 is not supported");
    {
        // The reference builds each axis's ramp in float64 (RopeEmbedder.precompute_freqs_cis).
        std::vector<double> frequencies;
        for (int axis = 0; axis < 3; ++axis)
            for (int j = 0; j < d.axis_dims[axis] / 2; ++j)
                frequencies.push_back(1.0 / std::pow(static_cast<double>(d.rope_theta),
                                                     static_cast<double>(2 * j) / d.axis_dims[axis]));
        if (static_cast<int>(frequencies.size()) != d.head_dim / 2)
            throw std::runtime_error("DiT rope axes do not cover head_dim");
        d.rope_frequencies = diff_upload_doubles(d, frequencies);
    }
    if (config.attention_kv_heads != config.attention_heads)
        throw std::runtime_error("DiT with grouped key/value heads is not supported");
    const int patch_inputs = d.patch * d.patch * d.channels;
    const std::string key = std::to_string(d.patch) + "-1";
    d.x_embed = diff_upload_matrix(d, model, "all_x_embedder." + key + ".weight", patch_inputs, d.dim,
                                   "all_x_embedder." + key + ".bias");
    d.cap_norm = diff_upload_f32(d, model, "cap_embedder.0.weight", d.caption_dim);
    d.cap_embed = diff_upload_matrix(d, model, "cap_embedder.1.weight", d.caption_dim, d.dim,
                                     "cap_embedder.1.bias");
    d.t_mlp0 = diff_upload_matrix(d, model, "t_embedder.mlp.0.weight", kDiffAdalnWidth, kDiffTimeMlpWidth,
                                  "t_embedder.mlp.0.bias");
    d.t_mlp1 = diff_upload_matrix(d, model, "t_embedder.mlp.2.weight", kDiffTimeMlpWidth, kDiffAdalnWidth,
                                  "t_embedder.mlp.2.bias");
    d.x_pad = diff_upload_f32(d, model, "x_pad_token", d.dim);
    d.cap_pad = diff_upload_f32(d, model, "cap_pad_token", d.dim);
    d.final_adaln = diff_upload_matrix(d, model, "all_final_layer." + key + ".adaLN_modulation.1.weight",
                                       kDiffAdalnWidth, d.dim,
                                       "all_final_layer." + key + ".adaLN_modulation.1.bias");
    d.final_linear = diff_upload_matrix(d, model, "all_final_layer." + key + ".linear.weight",
                                        d.dim, patch_inputs,
                                        "all_final_layer." + key + ".linear.bias");
    for (std::uint32_t index = 0; index < config.refiner_layer_count; ++index) {
        d.noise_refiner.push_back(diff_upload_block(d, model, "noise_refiner." + std::to_string(index), true));
        d.context_refiner.push_back(diff_upload_block(d, model, "context_refiner." + std::to_string(index), false));
    }
    for (std::uint32_t index = 0; index < config.layer_count; ++index)
        d.layers.push_back(diff_upload_block(d, model, "layers." + std::to_string(index), true));
}

void diff_upload_vae(FlyweightV2Diffusion& d) {
    const auto& model = *d.vae;
    const auto& config = model.config;
    for (int level = 0; level < 4; ++level) d.vae_channels[level] = static_cast<int>(config.vae_block_channels[level]);
    d.vae_groups = static_cast<int>(config.vae_norm_groups);
    d.latent_channels = static_cast<int>(config.latent_channels);
    d.latent_scale = config.latent_scale;
    d.latent_shift = config.latent_shift;
    if (d.latent_channels != d.channels)
        throw std::runtime_error("autoencoder latent channels do not match the DiT's in_channels");
    // The decoder walks block_out_channels in reverse.
    const int top = d.vae_channels[3];
    d.conv_in = diff_upload_conv(d, model, "decoder.conv_in", d.latent_channels, top, 3);
    d.mid_resnet[0] = diff_upload_resnet(d, model, "decoder.mid_block.resnets.0", top, top);
    d.mid_resnet[1] = diff_upload_resnet(d, model, "decoder.mid_block.resnets.1", top, top);
    d.mid_has_attention = config.vae_mid_attention;
    if (d.mid_has_attention) {
        const std::string prefix = "decoder.mid_block.attentions.0";
        d.mid_attention.norm = diff_upload_norm(d, model, prefix + ".group_norm", top);
        d.mid_attention.q = diff_upload_matrix(d, model, prefix + ".to_q.weight", top, top, prefix + ".to_q.bias");
        d.mid_attention.k = diff_upload_matrix(d, model, prefix + ".to_k.weight", top, top, prefix + ".to_k.bias");
        d.mid_attention.v = diff_upload_matrix(d, model, prefix + ".to_v.weight", top, top, prefix + ".to_v.bias");
        d.mid_attention.o = diff_upload_matrix(d, model, prefix + ".to_out.0.weight", top, top, prefix + ".to_out.0.bias");
    }
    const int resnets_per_block = static_cast<int>(config.vae_layers_per_block) + 1;
    int previous = top;
    for (int level = 0; level < 4; ++level) {
        const int out = d.vae_channels[3 - level];
        std::vector<VaeResnet> resnets;
        for (int index = 0; index < resnets_per_block; ++index) {
            const std::string prefix = "decoder.up_blocks." + std::to_string(level) + ".resnets." + std::to_string(index);
            resnets.push_back(diff_upload_resnet(d, model, prefix, index == 0 ? previous : out, out));
        }
        d.up_resnets.push_back(std::move(resnets));
        if (level < 3)
            d.upsamplers.push_back(diff_upload_conv(
                d, model, "decoder.up_blocks." + std::to_string(level) + ".upsamplers.0.conv", out, out, 3));
        previous = out;
    }
    d.norm_out = diff_upload_norm(d, model, "decoder.conv_norm_out", previous);
    d.conv_out = diff_upload_conv(d, model, "decoder.conv_out", previous, static_cast<int>(config.in_channels), 3);
}

// ---- Workspace -------------------------------------------------------------

// A bump cursor over the arena, reset per phase.
struct DiffArena {
    std::uint64_t base = 0, cursor = 0, capacity = 0;
    std::uint64_t take(std::uint64_t bytes) {
        const std::uint64_t at = cursor;
        cursor += diff_align(bytes);
        if (cursor > capacity)
            throw std::runtime_error("diffusion workspace exhausted: the request is larger than "
                                     "the tower was created for");
        return base + at;
    }
};

// Bytes the Q8 activation staging needs for `rows` of `width`; the bf16
// staging of balanced mode reuses the same allocation at twice the bytes.
std::uint64_t diff_q8_bytes(std::uint64_t rows, int width) {
    return diff_align(rows * width * 2) + diff_align(rows * (width / 32) * 4);
}

std::uint64_t diff_transformer_workspace(const FlyweightV2Diffusion& d, std::uint64_t rows) {
    const std::uint64_t f = sizeof(float);
    std::uint64_t total = 0;
    total += diff_align(rows * d.dim * f) * 3;                 // hidden, normed, attn
    total += diff_align(rows * 3 * d.dim * f);                 // qkv
    total += diff_align(rows * d.intermediate * f) * 2;        // gate, up
    total += diff_align(rows * d.caption_dim * f) * 2;         // caption rows, normed
    total += diff_align(rows * 3 * sizeof(std::int32_t));      // positions
    total += diff_align(rows * d.patch * d.patch * d.channels * f) * 2;  // patches in, out
    total += diff_align(rows * d.dim * 2) * 3;                 // bf16 q, k, v planes
    total += diff_align(4 * d.dim * f);                        // modulation
    total += diff_align(kDiffTimeMlpWidth * f) + diff_align(kDiffAdalnWidth * f) * 2;
    total += diff_q8_bytes(rows, std::max(d.intermediate, d.dim));
    return total;
}

std::uint64_t diff_encoder_workspace(const FlyweightV2Diffusion& d, std::uint64_t rows) {
    const std::uint64_t f = sizeof(float);
    const int q_width = d.encoder_heads * d.encoder_head_dim;
    const int kv_width = d.encoder_kv_heads * d.encoder_head_dim;
    std::uint64_t total = 0;
    total += diff_align(rows * d.encoder_hidden * f) * 3;      // hidden, normed, projected
    total += diff_align(rows * q_width * f) * 2;               // q, attention output
    total += diff_align(rows * kv_width * f) * 2;              // k, v
    total += diff_align(rows * d.encoder_intermediate * f) * 2;
    total += diff_align(rows * sizeof(std::int32_t));
    total += diff_align(rows * q_width * 2) + diff_align(rows * kv_width * 2) * 2;  // bf16 planes
    total += diff_q8_bytes(rows, std::max(d.encoder_intermediate, d.encoder_hidden));
    return total;
}

// The largest activation the decoder holds, in floats: each level's channel
// count at the resolution its upsampler produces (the level-2 block's
// channels at the full output size is the peak, not the last block's).
std::uint64_t diff_vae_plane_max(const FlyweightV2Diffusion& d, int latent_h, int latent_w) {
    std::uint64_t plane_max = 0;
    for (int level = 0; level < 4; ++level) {
        const std::uint64_t side = static_cast<std::uint64_t>(1) << (level < 3 ? level + 1 : level);
        plane_max = std::max(plane_max,
            static_cast<std::uint64_t>(d.vae_channels[3 - level]) * (latent_h * side) * (latent_w * side));
    }
    return plane_max;
}

std::uint64_t diff_vae_workspace(const FlyweightV2Diffusion& d, int latent_h, int latent_w) {
    const std::uint64_t f = sizeof(float);
    const std::uint64_t plane_max = diff_vae_plane_max(d, latent_h, latent_w);
    const std::uint64_t top_plane = static_cast<std::uint64_t>(latent_h) * latent_w;
    const int top = d.vae_channels[3];
    std::uint64_t total = diff_align(plane_max * f) * 3;
    total += diff_align(plane_max * 2);                                     // channels-last bf16 conv input
    total += diff_align(top_plane * top * f) * 4;                           // q, k, v^T, attention out (HWC)
    total += diff_align(static_cast<std::uint64_t>(kVaeAttentionChunk) * top_plane * f);  // scores
    total += diff_align(static_cast<std::uint64_t>(d.vae_groups) * (64 * 2 * sizeof(double) + 2 * sizeof(float)));
    total += diff_align(static_cast<std::uint64_t>(top) * f);                 // query scale
    return total;
}

// ---- Launch helpers --------------------------------------------------------

struct DiffLaunch {
    FlyweightV2Diffusion& d;

    void run(const char* name, std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t block,
             void** arguments, std::uint32_t shared_bytes = 0) const {
        if (flyweight_gpu_launch_named(name, grid_x, grid_y, block, shared_bytes, d.stream, arguments) != 0)
            throw std::runtime_error(std::string("diffusion CUDA kernel failed: ") + name);
        // FLYWEIGHT_DIFF_SYNC=1 names the kernel a fault belongs to; launches
        // are asynchronous, so without it every fault surfaces at the next
        // download.
        static const bool sync_each = [] {
            const char* value = std::getenv("FLYWEIGHT_DIFF_SYNC");
            return value && value[0] == '1';
        }();
        static const bool trace = [] {
            const char* value = std::getenv("FLYWEIGHT_DIFF_TRACE");
            return value && value[0] == '1';
        }();
        if (sync_each) {
            const auto started = std::chrono::steady_clock::now();
            if (flyweight_gpu_stream_sync(d.stream) != 0)
                throw std::runtime_error(std::string("diffusion CUDA kernel faulted: ") + name + " grid " +
                                         std::to_string(grid_x) + "x" + std::to_string(grid_y));
            if (trace)
                std::fprintf(stderr, "[diff] %s grid %ux%u block %u %.3f ms\n", name, grid_x, grid_y, block,
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
        } else if (trace) {
            // Host clock, so gaps between launches (a stalled host, a full
            // launch queue) show up in the trace as well as kernel time does.
            static const auto origin = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[diff] %s grid %ux%u block %u at %.3f s\n", name, grid_x, grid_y, block,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - origin).count());
        }
    }

    static std::uint32_t grid_for(std::uint64_t elements) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>((elements + 255) / 256, 4096));
    }

    void add_bias(std::uint64_t x, std::uint64_t bias, int width, std::uint64_t rows) const {
        long long elements = static_cast<long long>(rows) * width;
        void* arguments[] = {&x, &bias, &x, &width, &elements};
        run("diff_add_bias_rows", grid_for(elements), 1, 256, arguments);
    }

    void scale_columns(std::uint64_t x, std::uint64_t scale, int width, std::uint64_t rows) const {
        long long elements = static_cast<long long>(rows) * width;
        void* arguments[] = {&x, &scale, &width, &elements};
        run("diff_scale_columns", grid_for(elements), 1, 256, arguments);
    }

    void gated_add(std::uint64_t x, std::uint64_t y, std::uint64_t gate, int width, std::uint64_t rows) const {
        long long elements = static_cast<long long>(rows) * width;
        void* arguments[] = {&x, &y, &gate, &width, &elements};
        run("diff_gated_add_rows", grid_for(elements), 1, 256, arguments);
    }

    void add(std::uint64_t x, std::uint64_t y, std::uint64_t elements) const {
        int count = static_cast<int>(elements);
        void* arguments[] = {&x, &y, &count};
        run("vision_add_rows", grid_for(elements), 1, 256, arguments);
    }

    void silu_mul(std::uint64_t gate, std::uint64_t up, std::uint64_t out, std::uint64_t elements) const {
        long long count = static_cast<long long>(elements);
        void* arguments[] = {&gate, &up, &out, &count};
        run("diff_silu_mul", grid_for(elements), 1, 256, arguments);
    }

    void silu(std::uint64_t x, std::uint64_t elements) const {
        long long count = static_cast<long long>(elements);
        void* arguments[] = {&x, &count};
        run("diff_silu_rows", grid_for(elements), 1, 256, arguments);
    }

    void rms_norm(std::uint64_t input, std::uint64_t weight, std::uint64_t output, int width,
                  std::uint64_t rows, float epsilon) const {
        int count = static_cast<int>(rows), zero = 0;
        void* arguments[] = {&input, &weight, &output, &count, &width, &epsilon, &zero};
        run("rms_norm_rows", count, 1, 256, arguments);
    }

    void layer_norm(std::uint64_t input, std::uint64_t output, int width, std::uint64_t rows,
                    float epsilon) const {
        int count = static_cast<int>(rows);
        void* arguments[] = {&input, &output, &width, &count, &epsilon};
        run("diff_layer_norm_rows", count, 1, 256, arguments);
    }

    void head_rms_norm(std::uint64_t x, std::uint64_t weight, int heads, int head_dim,
                       std::uint64_t rows, int row_stride, int offset, float epsilon) const {
        int count = static_cast<int>(rows);
        void* arguments[] = {&x, &weight, &heads, &head_dim, &count, &row_stride, &offset, &epsilon};
        const std::uint64_t warps = rows * heads;
        run("diff_head_rms_norm", static_cast<std::uint32_t>((warps + 7) / 8), 1, 256, arguments);
    }

    // q16/k16/v16 are bf16 scratch planes for the tensor-core path.
    void attention(std::uint64_t q, std::uint64_t k, std::uint64_t v, std::uint64_t out, int heads,
                   int kv_heads, int head_dim, std::uint64_t rows, int q_stride, int kv_stride,
                   bool causal, std::uint64_t q16, std::uint64_t k16, std::uint64_t v16) const {
        int count = static_cast<int>(rows), causal_flag = causal ? 1 : 0;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        if (d.flash_attention && !d.exact && head_dim == 128) {
            float q_scale = scale * 1.4426950408889634f;   // exp2 softmax
            void* pack_arguments[] = {&q, &k, &v, &q16, &k16, &v16, &heads, &kv_heads, &count,
                                      &q_stride, &kv_stride, &q_scale};
            run("diff_pack_attention_bf16",
                grid_for(rows * static_cast<std::uint64_t>(heads + 2 * kv_heads) * 128), 1, 256, pack_arguments);
            void* arguments[] = {&q16, &k16, &v16, &out, &heads, &kv_heads, &count, &causal_flag};
            run("diff_flash_attention_bf16", static_cast<std::uint32_t>((rows + 63) / 64),
                static_cast<std::uint32_t>(heads), 128, arguments);
            return;
        }
        void* arguments[] = {&q, &k, &v, &out, &heads, &kv_heads, &head_dim, &count,
                             &q_stride, &kv_stride, &scale, &causal_flag};
        run("diff_attention_rows", static_cast<std::uint32_t>((rows + 31) / 32),
            static_cast<std::uint32_t>(heads), 256, arguments);
    }

    // output[rows][out] = input[rows][in] * W^T (+ bias). Quantized weights
    // take the MMQ path when the width allows (Q8 activations, int8 tensor
    // cores), the per-format rows kernel otherwise; f32/bf16 take the vision
    // tiled GEMM, which carries the bias itself.
    void gemm(const DiffMatrix& m, std::uint64_t input, std::uint64_t output, std::uint64_t rows,
              std::uint64_t q8, std::uint64_t q8_scales) const {
        int in = m.in, out = m.out, count = static_cast<int>(rows);
        if (m.type == 0 || m.type == 30) {
            std::uint64_t bias = m.bias;
            void* arguments[] = {const_cast<std::uint64_t*>(&m.device), &bias, &input, &output, &in, &out, &count};
            run(m.type == 30 ? "vision_bf16_gemm_rows" : "vision_f32_gemm_rows",
                static_cast<std::uint32_t>((out + 63) / 64), static_cast<std::uint32_t>((rows + 63) / 64),
                256, arguments);
            return;
        }
        const auto* format = flyweight::v2::qwen_format(m.type);
        if (!format) throw std::runtime_error("diffusion GEMM: no kernels for weight type " + std::to_string(m.type));
        if (d.balanced && m.type == 8 && (in & 63) == 0 && q8 && d.flash_attention) {
            // q8 doubles as the bf16 activation staging: it is sized for
            // rows * width * 2 bytes when balanced (see the workspaces).
            long long elements = static_cast<long long>(rows) * in;
            void* pack_arguments[] = {&input, &q8, &in, &elements};
            run("diff_pack_rows_bf16", grid_for(static_cast<std::uint64_t>(elements)), 1, 256, pack_arguments);
            void* arguments[] = {const_cast<std::uint64_t*>(&m.device), &q8, &output, &in, &out, &count};
            run("diff_q8_bf16_gemm", static_cast<std::uint32_t>((out + 127) / 128),
                static_cast<std::uint32_t>((rows + 127) / 128), 256, arguments);
            if (m.bias) add_bias(output, m.bias, out, rows);
            return;
        }
        const bool mmq = format->matmul_q8_mmq && d.int8_tensor_cores && (in & 255) == 0 && q8 && !d.exact && !d.balanced;
        if (mmq) {
            int scale_stride = (in / 32) * 2;
            {
                void* arguments[] = {&input, &q8, &q8_scales, &in, &scale_stride};
                run("quantize_q8_blocks_rows", static_cast<std::uint32_t>((in + 31) / 32),
                    static_cast<std::uint32_t>(rows), 32, arguments);
            }
            const auto mmq_shape=qwen_mmq_shape();
            const int mmq_rows=format->mmq_min?kQ8MmqMinRows:mmq_shape.rows();
            const int mmq_threads=format->mmq_min?kQ8MmqMinThreads:mmq_shape.threads();
            const int mmq_tokens=format->mmq_min?kQ8MmqMinTokens:mmq_shape.tokens();
            // One launch for every token tile: grid.y walks the batches.
            void* arguments[] = {const_cast<std::uint64_t*>(&m.device), &q8, &q8_scales,
                                 &output, &in, &out, &count, &scale_stride};
            run(format->matmul_q8_mmq, static_cast<std::uint32_t>((out + mmq_rows - 1) / mmq_rows),
                static_cast<std::uint32_t>((count + mmq_tokens - 1) / mmq_tokens),
                static_cast<std::uint32_t>(mmq_threads), arguments,
                (format->mmq_dynamic_shared && !format->mmq_min) ? qwen_mmq_dynamic_shared(mmq_shape) : 0u);
        } else {
            if (!format->matmul_rows)
                throw std::runtime_error("diffusion GEMM: weight type " + std::to_string(m.type) +
                                         " has no rows kernel");
            void* arguments[] = {const_cast<std::uint64_t*>(&m.device), &input, &output, &in, &out, &count};
            if (format->matmul_rows_grid == flyweight::v2::RowsMatmulGrid::tiled32)
                run(format->matmul_rows, static_cast<std::uint32_t>((out + 31) / 32),
                    static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
            else
                run(format->matmul_rows, static_cast<std::uint32_t>(out),
                    static_cast<std::uint32_t>((rows + 3) / 4), 256, arguments);
        }
        if (m.bias) add_bias(output, m.bias, out, rows);
    }

    void upload(std::uint64_t destination, const void* source, std::uint64_t bytes) const {
        if (flyweight_gpu_upload(destination, source, bytes, d.stream) != 0)
            throw std::runtime_error("diffusion workspace upload failed");
    }

    void download(void* destination, std::uint64_t source, std::uint64_t bytes) const {
        if (flyweight_gpu_download(destination, source, bytes, d.stream) != 0 ||
            flyweight_gpu_stream_sync(d.stream) != 0)
            throw std::runtime_error("diffusion workspace download failed");
    }
};

// ---- Layer streaming -------------------------------------------------------

// Walks a sequence of layers, staging each one's matrices into the slot the
// layer before last released, one layer ahead of compute. With resident
// weights every call is a no-op and `bind` leaves the device pointers alone.
struct DiffStreamer {
    FlyweightV2Diffusion& d;
    std::vector<DiffLayerRef> refs;
    std::size_t next_to_stage = 0;

    void stage(std::size_t index) {
        if (!d.host_weights || index >= refs.size()) return;
        const int slot = static_cast<int>(index & 1);
        // The copy may not overwrite the slot until the layer that last ran
        // from it has finished; an event never recorded is no wait.
        if (flyweight_gpu_stream_wait_event(d.copy_stream, d.consumed_event[slot]) != 0)
            throw std::runtime_error("diffusion staging wait failed");
        const DiffLayerRef& ref = refs[index];
        for (int i = 0; i < ref.count; ++i) {
            const DiffMatrix& m = *ref.matrices[i];
            if (!m.host) continue;
            if (flyweight_gpu_upload(d.staging[slot] + m.slot_offset, m.host, m.bytes, d.copy_stream) != 0)
                throw std::runtime_error("diffusion weight staging upload failed");
        }
        if (flyweight_gpu_event_record(d.staged_event[slot], d.copy_stream) != 0)
            throw std::runtime_error("diffusion staging event failed");
        next_to_stage = index + 1;
    }

    // Before layer `index` runs: wait for its bytes, point its matrices at
    // the slot, and start the next layer's upload.
    void begin(std::size_t index) {
        if (!d.host_weights) return;
        if (next_to_stage <= index) stage(index);
        const int slot = static_cast<int>(index & 1);
        if (flyweight_gpu_stream_wait_event(d.stream, d.staged_event[slot]) != 0)
            throw std::runtime_error("diffusion staged wait failed");
        const DiffLayerRef& ref = refs[index];
        for (int i = 0; i < ref.count; ++i) {
            DiffMatrix& m = *ref.matrices[i];
            if (m.host) m.device = d.staging[slot] + m.slot_offset;
        }
        stage(index + 1);
    }

    void end(std::size_t index) {
        if (!d.host_weights) return;
        const int slot = static_cast<int>(index & 1);
        if (flyweight_gpu_event_record(d.consumed_event[slot], d.stream) != 0)
            throw std::runtime_error("diffusion consumed event failed");
    }
};

// ---- Text encoder ----------------------------------------------------------

// Runs the Qwen3 encoder over `tokens` and writes hidden_states[-2] as
// [count][hidden] f32 into `output`.
void diff_encode_text(FlyweightV2Diffusion& d, const std::uint32_t* tokens, std::uint64_t count,
                      float* output) {
    if (!count) throw std::runtime_error("empty prompt");
    if (count > d.max_prompt_tokens)
        throw std::runtime_error("prompt has " + std::to_string(count) + " tokens; the tower allows " +
                                 std::to_string(d.max_prompt_tokens));
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const std::uint64_t f = sizeof(float);
    const int hidden = d.encoder_hidden;
    const int q_width = d.encoder_heads * d.encoder_head_dim;
    const int kv_width = d.encoder_kv_heads * d.encoder_head_dim;
    const std::uint64_t d_hidden = arena.take(count * hidden * f);
    const std::uint64_t d_normed = arena.take(count * hidden * f);
    const std::uint64_t d_projected = arena.take(count * hidden * f);
    const std::uint64_t d_q = arena.take(count * q_width * f);
    const std::uint64_t d_attn = arena.take(count * q_width * f);
    const std::uint64_t d_k = arena.take(count * kv_width * f);
    const std::uint64_t d_v = arena.take(count * kv_width * f);
    const std::uint64_t d_gate = arena.take(count * d.encoder_intermediate * f);
    const std::uint64_t d_up = arena.take(count * d.encoder_intermediate * f);
    const std::uint64_t d_positions = arena.take(count * sizeof(std::int32_t));
    const std::uint64_t d_q16 = arena.take(count * q_width * 2);
    const std::uint64_t d_k16 = arena.take(count * kv_width * 2);
    const std::uint64_t d_v16 = arena.take(count * kv_width * 2);
    const int q8_width = std::max(d.encoder_intermediate, hidden);
    const std::uint64_t d_q8 = arena.take(count * q8_width * 2);
    const std::uint64_t d_q8_scales = arena.take(count * (q8_width / 32) * 4);

    // Token embeddings gathered on the host: the table is quantized and a
    // prompt is a few hundred rows. (Host buffers feeding the stream stay
    // alive until the download at the end synchronizes it.)
    std::vector<float> rows(count * hidden);
    std::vector<std::int32_t> positions(count);
    {
        const Tensor& table = *d.token_embedding;
        const auto* source = tensor_data(*d.encoder, table);
        const std::uint64_t vocabulary = table.shape.size() > 1 ? table.shape[1] : 0;
        for (std::uint64_t row = 0; row < count; ++row) {
            if (tokens[row] >= vocabulary)
                throw std::runtime_error("prompt token " + std::to_string(tokens[row]) + " is outside the vocabulary");
            const std::uint64_t base = static_cast<std::uint64_t>(tokens[row]) * hidden;
            for (int column = 0; column < hidden; ++column)
                rows[row * hidden + column] = tensor_value(source, table.type, base + column);
        }
        launch.upload(d_hidden, rows.data(), rows.size() * f);
        for (std::uint64_t row = 0; row < count; ++row) positions[row] = static_cast<std::int32_t>(row);
        launch.upload(d_positions, positions.data(), count * sizeof(std::int32_t));
    }

    DiffStreamer streamer{d, {}, 0};
    for (auto& layer : d.encoder_layers) streamer.refs.push_back(diff_encoder_matrices(layer));
    for (std::size_t index = 0; index < d.encoder_layers.size(); ++index) {
        auto& layer = d.encoder_layers[index];
        streamer.begin(index);
        launch.rms_norm(d_hidden, layer.attn_norm, d_normed, hidden, count, d.encoder_epsilon);
        launch.gemm(layer.q, d_normed, d_q, count, d_q8, d_q8_scales);
        launch.gemm(layer.k, d_normed, d_k, count, d_q8, d_q8_scales);
        launch.gemm(layer.v, d_normed, d_v, count, d_q8, d_q8_scales);
        launch.head_rms_norm(d_q, layer.q_norm, d.encoder_heads, d.encoder_head_dim, count, q_width, 0, d.encoder_epsilon);
        launch.head_rms_norm(d_k, layer.k_norm, d.encoder_kv_heads, d.encoder_head_dim, count, kv_width, 0, d.encoder_epsilon);
        {
            int heads = d.encoder_heads, kv_heads = d.encoder_kv_heads, head_dim = d.encoder_head_dim;
            int rows = static_cast<int>(count), zero = 0, q_stride = q_width, kv_stride = kv_width;
            std::uint64_t q = d_q, k = d_k, positions = d_positions, table = d.encoder_rope_frequencies;
            void* q_arguments[] = {&q, &positions, &table, &heads, &head_dim, &rows, &q_stride, &zero};
            launch.run("diff_rope_half_rows", rows, 1, 256, q_arguments);
            void* k_arguments[] = {&k, &positions, &table, &kv_heads, &head_dim, &rows, &kv_stride, &zero};
            launch.run("diff_rope_half_rows", rows, 1, 256, k_arguments);
        }
        launch.attention(d_q, d_k, d_v, d_attn, d.encoder_heads, d.encoder_kv_heads, d.encoder_head_dim,
                         count, q_width, kv_width, true, d_q16, d_k16, d_v16);
        launch.gemm(layer.o, d_attn, d_projected, count, d_q8, d_q8_scales);
        launch.add(d_hidden, d_projected, count * hidden);
        launch.rms_norm(d_hidden, layer.ffn_norm, d_normed, hidden, count, d.encoder_epsilon);
        launch.gemm(layer.gate, d_normed, d_gate, count, d_q8, d_q8_scales);
        launch.gemm(layer.up, d_normed, d_up, count, d_q8, d_q8_scales);
        launch.silu_mul(d_gate, d_up, d_gate, count * d.encoder_intermediate);
        launch.gemm(layer.down, d_gate, d_projected, count, d_q8, d_q8_scales);
        launch.add(d_hidden, d_projected, count * hidden);
        streamer.end(index);
    }
    launch.download(output, d_hidden, count * hidden * f);
}

// ---- DiT -------------------------------------------------------------------

struct DiffTransformerBuffers {
    std::uint64_t hidden = 0, normed = 0, attn = 0, qkv = 0, gate = 0, up = 0;
    std::uint64_t caption = 0, caption_normed = 0, positions = 0, patches = 0, patches_out = 0;
    std::uint64_t modulation = 0, t_freq = 0, t_mid = 0, t_emb = 0;
    std::uint64_t q8 = 0, q8_scales = 0;
    std::uint64_t q16 = 0, k16 = 0, v16 = 0;
};

DiffTransformerBuffers diff_transformer_buffers(FlyweightV2Diffusion& d, DiffArena& arena, std::uint64_t rows) {
    const std::uint64_t f = sizeof(float);
    DiffTransformerBuffers b;
    b.hidden = arena.take(rows * d.dim * f);
    b.normed = arena.take(rows * d.dim * f);
    b.attn = arena.take(rows * d.dim * f);
    b.qkv = arena.take(rows * 3 * d.dim * f);
    b.gate = arena.take(rows * d.intermediate * f);
    b.up = arena.take(rows * d.intermediate * f);
    b.caption = arena.take(rows * d.caption_dim * f);
    b.caption_normed = arena.take(rows * d.caption_dim * f);
    b.positions = arena.take(rows * 3 * sizeof(std::int32_t));
    b.patches = arena.take(rows * d.patch * d.patch * d.channels * f);
    b.patches_out = arena.take(rows * d.patch * d.patch * d.channels * f);
    b.q16 = arena.take(rows * d.dim * 2);
    b.k16 = arena.take(rows * d.dim * 2);
    b.v16 = arena.take(rows * d.dim * 2);
    b.modulation = arena.take(4 * d.dim * f);
    b.t_freq = arena.take(kDiffAdalnWidth * f);
    b.t_mid = arena.take(kDiffTimeMlpWidth * f);
    b.t_emb = arena.take(kDiffAdalnWidth * f);
    const int q8_width = std::max(d.intermediate, d.dim);
    b.q8 = arena.take(rows * q8_width * 2);
    b.q8_scales = arena.take(rows * (q8_width / 32) * 4);
    return b;
}

// One transformer block over rows [first, first + rows) of the hidden buffer.
void diff_run_block(FlyweightV2Diffusion& d, const DiffLaunch& launch, DiffBlock& block,
                    const DiffTransformerBuffers& b, std::uint64_t first, std::uint64_t rows) {
    const std::uint64_t f = sizeof(float);
    const int dim = d.dim;
    const std::uint64_t hidden = b.hidden + first * dim * f;
    const std::uint64_t normed = b.normed + first * dim * f;
    const std::uint64_t attn = b.attn + first * dim * f;
    const std::uint64_t qkv = b.qkv + first * 3 * dim * f;
    const std::uint64_t positions = b.positions + first * 3 * sizeof(std::int32_t);
    const std::uint64_t gate = b.gate + first * d.intermediate * f;
    const std::uint64_t up = b.up + first * d.intermediate * f;
    std::uint64_t scale_msa = 0, gate_msa = 0, scale_mlp = 0, gate_mlp = 0;
    if (block.modulated) {
        launch.gemm(block.adaln, b.t_emb, b.modulation, 1, 0, 0);
        int width = dim;
        std::uint64_t modulation = b.modulation;
        void* arguments[] = {&modulation, &width};
        launch.run("diff_modulation_prepare", static_cast<std::uint32_t>((4 * dim + 255) / 256), 1, 256, arguments);
        scale_msa = b.modulation;
        gate_msa = b.modulation + static_cast<std::uint64_t>(dim) * f;
        scale_mlp = b.modulation + static_cast<std::uint64_t>(2 * dim) * f;
        gate_mlp = b.modulation + static_cast<std::uint64_t>(3 * dim) * f;
    }

    // Attention: norm1 (scaled), fused qkv, per-head q/k norms, rope, attention,
    // output projection, norm2, gated residual.
    launch.rms_norm(hidden, block.attention_norm1, normed, dim, rows, d.epsilon);
    if (block.modulated) launch.scale_columns(normed, scale_msa, dim, rows);
    launch.gemm(block.qkv, normed, qkv, rows, b.q8, b.q8_scales);
    launch.head_rms_norm(qkv, block.q_norm, d.heads, d.head_dim, rows, 3 * dim, 0, 1e-5f);
    launch.head_rms_norm(qkv, block.k_norm, d.heads, d.head_dim, rows, 3 * dim, dim, 1e-5f);
    {
        int heads = d.heads, head_dim = d.head_dim, count = static_cast<int>(rows);
        int axis0 = d.axis_dims[0], axis1 = d.axis_dims[1];
        std::uint64_t buffer = qkv, position_buffer = positions, table = d.rope_frequencies;
        void* arguments[] = {&buffer, &position_buffer, &table, &heads, &head_dim, &count, &axis0, &axis1};
        launch.run("diff_rope_axes_rows", count, 2, 256, arguments);
    }
    launch.attention(qkv, qkv + static_cast<std::uint64_t>(dim) * f, qkv + static_cast<std::uint64_t>(2 * dim) * f,
                     attn, d.heads, d.heads, d.head_dim, rows, 3 * dim, 3 * dim, false, b.q16, b.k16, b.v16);
    launch.gemm(block.out, attn, normed, rows, b.q8, b.q8_scales);
    launch.rms_norm(normed, block.attention_norm2, attn, dim, rows, d.epsilon);
    launch.gated_add(hidden, attn, gate_msa, dim, rows);

    // Feed-forward: norm1 (scaled), SwiGLU, norm2, gated residual.
    launch.rms_norm(hidden, block.ffn_norm1, normed, dim, rows, d.epsilon);
    if (block.modulated) launch.scale_columns(normed, scale_mlp, dim, rows);
    launch.gemm(block.w1, normed, gate, rows, b.q8, b.q8_scales);
    launch.gemm(block.w3, normed, up, rows, b.q8, b.q8_scales);
    launch.silu_mul(gate, up, gate, rows * d.intermediate);
    launch.gemm(block.w2, gate, normed, rows, b.q8, b.q8_scales);
    launch.rms_norm(normed, block.ffn_norm2, attn, dim, rows, d.epsilon);
    launch.gated_add(hidden, attn, gate_mlp, dim, rows);
}

// Latent [C][H][W] -> token rows (h, w) of (ph, pw, c) features.
void diff_patchify(const FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                   std::vector<float>& rows) {
    const int p = d.patch, c_count = d.channels;
    const int tokens_h = latent_h / p, tokens_w = latent_w / p;
    const int features = p * p * c_count;
    rows.assign(static_cast<std::size_t>(tokens_h) * tokens_w * features, 0.0f);
    for (int th = 0; th < tokens_h; ++th)
        for (int tw = 0; tw < tokens_w; ++tw)
            for (int ph = 0; ph < p; ++ph)
                for (int pw = 0; pw < p; ++pw)
                    for (int c = 0; c < c_count; ++c)
                        rows[(static_cast<std::size_t>(th) * tokens_w + tw) * features + (ph * p + pw) * c_count + c] =
                            latents[(static_cast<std::size_t>(c) * latent_h + th * p + ph) * latent_w + tw * p + pw];
}

void diff_unpatchify(const FlyweightV2Diffusion& d, const std::vector<float>& rows, int latent_h, int latent_w,
                     float* latents) {
    const int p = d.patch, c_count = d.channels;
    const int tokens_h = latent_h / p, tokens_w = latent_w / p;
    const int features = p * p * c_count;
    for (int th = 0; th < tokens_h; ++th)
        for (int tw = 0; tw < tokens_w; ++tw)
            for (int ph = 0; ph < p; ++ph)
                for (int pw = 0; pw < p; ++pw)
                    for (int c = 0; c < c_count; ++c)
                        latents[(static_cast<std::size_t>(c) * latent_h + th * p + ph) * latent_w + tw * p + pw] =
                            rows[(static_cast<std::size_t>(th) * tokens_w + tw) * features + (ph * p + pw) * c_count + c];
}

// One transformer forward: velocity for `latents` at normalized time `t`
// (the pipeline's (1000 - timestep) / 1000) under `caption` [caption_tokens][caption_dim].
// Writes the raw model output (before the pipeline's negation).
void diff_transformer_forward(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                              const float* caption, std::uint64_t caption_tokens, float t, float* velocity) {
    if (latent_h % d.patch || latent_w % d.patch)
        throw std::runtime_error("latent sides must be multiples of the patch size");
    if (!caption_tokens) throw std::runtime_error("empty caption");
    const int tokens_h = latent_h / d.patch, tokens_w = latent_w / d.patch;
    const std::uint64_t image_rows = static_cast<std::uint64_t>(tokens_h) * tokens_w;
    const std::uint64_t image_padded = (image_rows + kDiffSequenceMultiple - 1) / kDiffSequenceMultiple * kDiffSequenceMultiple;
    const std::uint64_t caption_padded = (caption_tokens + kDiffSequenceMultiple - 1) / kDiffSequenceMultiple * kDiffSequenceMultiple;
    const std::uint64_t rows = image_padded + caption_padded;
    const std::uint64_t f = sizeof(float);
    const int dim = d.dim;
    const int features = d.patch * d.patch * d.channels;
    if (diff_transformer_workspace(d, rows) > d.workspace_bytes)
        throw std::runtime_error("image or prompt is larger than the tower was created for");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const DiffTransformerBuffers b = diff_transformer_buffers(d, arena, rows);
    // Host staging for the uploads below; alive until the final download syncs.
    std::vector<float> freq(kDiffAdalnWidth);
    std::vector<std::int32_t> positions(rows * 3, 0);
    std::vector<float> patches;
    std::vector<float> caption_rows(caption_padded * d.caption_dim);

    // Timestep embedding: [cos(t f), sin(t f)] over 128 log-spaced frequencies.
    {
        const int half = kDiffAdalnWidth / 2;
        const double scaled = static_cast<double>(t) * d.time_scale;
        for (int i = 0; i < half; ++i) {
            const double frequency = std::exp(-std::log(10000.0) * i / half);
            const double angle = scaled * frequency;
            freq[i] = static_cast<float>(std::cos(angle));
            freq[half + i] = static_cast<float>(std::sin(angle));
        }
        launch.upload(b.t_freq, freq.data(), kDiffAdalnWidth * f);
        launch.gemm(d.t_mlp0, b.t_freq, b.t_mid, 1, 0, 0);
        launch.silu(b.t_mid, kDiffTimeMlpWidth);
        launch.gemm(d.t_mlp1, b.t_mid, b.t_emb, 1, 0, 0);
    }

    // Positions: caption tokens at (1 + i, 0, 0); image tokens at
    // (caption_padded + 1, h, w); padding rows at the origin.
    {
        for (std::uint64_t row = 0; row < image_rows; ++row) {
            positions[row * 3] = static_cast<std::int32_t>(caption_padded + 1);
            positions[row * 3 + 1] = static_cast<std::int32_t>(row / tokens_w);
            positions[row * 3 + 2] = static_cast<std::int32_t>(row % tokens_w);
        }
        for (std::uint64_t i = 0; i < caption_padded; ++i)
            positions[(image_padded + i) * 3] = static_cast<std::int32_t>(1 + i);
        launch.upload(b.positions, positions.data(), rows * 3 * sizeof(std::int32_t));
    }

    // Image rows: patchify, embed, pad token on the padding rows, refine.
    {
        diff_patchify(d, latents, latent_h, latent_w, patches);
        patches.resize(image_padded * features, 0.0f);
        launch.upload(b.patches, patches.data(), image_padded * features * f);
        launch.gemm(d.x_embed, b.patches, b.hidden, image_padded, b.q8, b.q8_scales);
        if (image_padded > image_rows) {
            int first = static_cast<int>(image_rows), count = static_cast<int>(image_padded - image_rows), width = dim;
            std::uint64_t hidden = b.hidden, pad = d.x_pad;
            void* arguments[] = {&hidden, &pad, &first, &count, &width};
            launch.run("diff_fill_rows", DiffLaunch::grid_for(static_cast<std::uint64_t>(count) * width), 1, 256, arguments);
        }
    }

    // Caption rows: RMSNorm, embed, pad token, refine.
    {
        std::memcpy(caption_rows.data(), caption, caption_tokens * d.caption_dim * f);
        for (std::uint64_t row = caption_tokens; row < caption_padded; ++row)
            std::memcpy(caption_rows.data() + row * d.caption_dim,
                        caption + (caption_tokens - 1) * d.caption_dim, d.caption_dim * f);
        launch.upload(b.caption, caption_rows.data(), caption_padded * d.caption_dim * f);
        launch.rms_norm(b.caption, d.cap_norm, b.caption_normed, d.caption_dim, caption_padded, d.epsilon);
        const std::uint64_t caption_hidden = b.hidden + image_padded * dim * f;
        launch.gemm(d.cap_embed, b.caption_normed, caption_hidden, caption_padded, b.q8, b.q8_scales);
        if (caption_padded > caption_tokens) {
            int first = static_cast<int>(caption_tokens), count = static_cast<int>(caption_padded - caption_tokens), width = dim;
            std::uint64_t hidden = caption_hidden, pad = d.cap_pad;
            void* arguments[] = {&hidden, &pad, &first, &count, &width};
            launch.run("diff_fill_rows", DiffLaunch::grid_for(static_cast<std::uint64_t>(count) * width), 1, 256, arguments);
        }
    }

    // The blocks in run order: the noise refiner over the image rows, the
    // context refiner over the caption rows, then the stack over the unified
    // sequence (image rows then caption rows, already adjacent). One list so
    // host-resident weights stream straight through without a bubble.
    struct Run { DiffBlock* block; std::uint64_t first, rows; };
    std::vector<Run> runs;
    for (auto& block : d.noise_refiner) runs.push_back({&block, 0, image_padded});
    for (auto& block : d.context_refiner) runs.push_back({&block, image_padded, caption_padded});
    for (auto& block : d.layers) runs.push_back({&block, 0, rows});
    DiffStreamer streamer{d, {}, 0};
    for (const auto& run : runs) streamer.refs.push_back(diff_block_matrices(*run.block));
    for (std::size_t index = 0; index < runs.size(); ++index) {
        streamer.begin(index);
        diff_run_block(d, launch, *runs[index].block, b, runs[index].first, runs[index].rows);
        streamer.end(index);
    }

    // Final layer: LayerNorm (no affine) scaled by 1 + adaLN(silu(t)), linear.
    {
        // t_mid is free by now; reuse it for silu(t_emb) and the scale vector.
        if (flyweight_gpu_copy_device(b.t_mid, b.t_emb, kDiffAdalnWidth * f, d.stream) != 0)
            throw std::runtime_error("diffusion final-layer copy failed");
        launch.silu(b.t_mid, kDiffAdalnWidth);
        launch.gemm(d.final_adaln, b.t_mid, b.modulation, 1, 0, 0);
        {
            // 1 + scale: the prepare kernel's "scale" slots are even chunks.
            int width = dim;
            std::uint64_t modulation = b.modulation;
            void* arguments[] = {&modulation, &width};
            launch.run("diff_modulation_prepare", static_cast<std::uint32_t>((dim + 255) / 256), 1, 256, arguments);
        }
        launch.layer_norm(b.hidden, b.normed, dim, image_padded, 1e-6f);
        launch.scale_columns(b.normed, b.modulation, dim, image_padded);
        launch.gemm(d.final_linear, b.normed, b.patches_out, image_padded, b.q8, b.q8_scales);
    }
    std::vector<float> out_rows(image_padded * features);
    launch.download(out_rows.data(), b.patches_out, image_padded * features * f);
    diff_unpatchify(d, out_rows, latent_h, latent_w, velocity);
}

// ---- Autoencoder decoder ---------------------------------------------------

struct VaeState {
    FlyweightV2Diffusion& d;
    DiffLaunch launch;
    std::uint64_t partials = 0;
    std::uint64_t hwc = 0;   // channels-last bf16 staging for the tensor-core convs

    // `input` is [C][H][W] f32 for the f32 kernel and channels-last bf16 (the
    // hwc buffer) for the tensor-core one; the caller writes whichever the
    // conv wants, see input_for().
    void conv(const VaeConv& conv, std::uint64_t input, std::uint64_t output, int height, int width) const {
        int in = conv.in, out = conv.out, h = height, w = width, kernel = conv.kernel;
        std::uint64_t bias = conv.bias;
        if (conv.tensor_core()) {
            std::uint64_t weight = conv.weight_bf16;
            void* arguments[] = {&input, &weight, &bias, &output, &in, &out, &h, &w, &kernel};
            launch.run("diff_conv2d_bf16", static_cast<std::uint32_t>((width + 127) / 128),
                       static_cast<std::uint32_t>(height) * static_cast<std::uint32_t>(out / 64), 128, arguments);
            return;
        }
        std::uint64_t weight = conv.weight;
        void* arguments[] = {&input, &weight, &bias, &output, &in, &out, &h, &w, &kernel};
        launch.run("diff_conv2d", static_cast<std::uint32_t>((width + 31) / 32),
                   static_cast<std::uint32_t>((height + 31) / 32) * static_cast<std::uint32_t>((out + 3) / 4),
                   256, arguments);
    }

    // Group norm (+SiLU) into the buffer the next conv reads.
    void group_norm(const VaeNorm& norm, std::uint64_t input, std::uint64_t output, std::uint64_t output_hwc,
                    int channels, int plane, bool silu) const {
        int groups = d.vae_groups, slices = 64, silu_flag = silu ? 1 : 0, c = channels, p = plane;
        float epsilon = 1e-6f;
        long long count = static_cast<long long>(channels / groups) * plane;
        std::uint64_t partial = partials, weight = norm.weight, bias = norm.bias;
        // The 2 floats per group land past the partials, which are sized for it.
        std::uint64_t stat = partials + static_cast<std::uint64_t>(groups) * slices * 2 * sizeof(double);
        void* stats[] = {&input, &partial, &c, &p, &groups, &slices};
        launch.run("diff_group_norm_stats", static_cast<std::uint32_t>(groups), static_cast<std::uint32_t>(slices), 256, stats);
        void* finalize[] = {&partial, &stat, &groups, &slices, &count, &epsilon};
        launch.run("diff_group_norm_finalize", static_cast<std::uint32_t>((groups + 255) / 256), 1, 256, finalize);
        void* apply[] = {&input, &stat, &weight, &bias, &output, &output_hwc, &c, &p, &groups, &silu_flag};
        launch.run("diff_group_norm_apply", static_cast<std::uint32_t>((plane + 31) / 32),
                   static_cast<std::uint32_t>((channels + 31) / 32), 256, apply);
    }

    // [C][H][W] f32 -> the hwc staging buffer.
    void to_hwc(std::uint64_t input, int channels, int plane) const {
        int c = channels, p = plane;
        std::uint64_t out = hwc;
        void* arguments[] = {&input, &out, &c, &p};
        launch.run("diff_to_hwc_bf16", static_cast<std::uint32_t>((plane + 31) / 32),
                   static_cast<std::uint32_t>((channels + 31) / 32), 256, arguments);
    }

    // Runs `conv` on an f32 [C][H][W] input, staging it if the conv is tensor-core.
    void conv_from_f32(const VaeConv& conv, std::uint64_t input, std::uint64_t output, int height, int width) const {
        if (conv.tensor_core()) {
            to_hwc(input, conv.in, height * width);
            this->conv(conv, hwc, output, height, width);
        } else {
            this->conv(conv, input, output, height, width);
        }
    }

    // x (in channels) -> out channels, using two scratch planes; the result
    // lands in `output`, which may not alias `x`.
    void resnet(const VaeResnet& r, std::uint64_t x, std::uint64_t scratch, std::uint64_t output,
                int height, int width) const {
        const int plane = height * width;
        const bool tc1 = r.conv1.tensor_core(), tc2 = r.conv2.tensor_core();
        group_norm(r.norm1, x, scratch, tc1 ? hwc : 0, r.in, plane, true);
        conv(r.conv1, tc1 ? hwc : scratch, output, height, width);
        group_norm(r.norm2, output, scratch, tc2 ? hwc : 0, r.out, plane, true);
        conv(r.conv2, tc2 ? hwc : scratch, output, height, width);
        if (r.in != r.out) {
            conv_from_f32(r.shortcut, x, scratch, height, width);
            launch.add(output, scratch, static_cast<std::uint64_t>(r.out) * plane);
        } else {
            launch.add(output, x, static_cast<std::uint64_t>(r.out) * plane);
        }
    }
};

// Decodes one latent tile [C][latent_h][latent_w] to f32 pixels [3][8h][8w].
void diff_decode_tile(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                      float* pixels) {
    if (diff_vae_workspace(d, latent_h, latent_w) > d.workspace_bytes)
        throw std::runtime_error("image is larger than the tower was created for");
    const std::uint64_t f = sizeof(float);
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const std::uint64_t plane_max = diff_vae_plane_max(d, latent_h, latent_w);
    std::uint64_t buffers[3];
    for (auto& buffer : buffers) buffer = arena.take(plane_max * f);
    const int top = d.vae_channels[3];
    const std::uint64_t top_plane = static_cast<std::uint64_t>(latent_h) * latent_w;
    const std::uint64_t d_q = arena.take(top_plane * top * f);
    const std::uint64_t d_k = arena.take(top_plane * top * f);
    const std::uint64_t d_vt = arena.take(top_plane * top * f);
    const std::uint64_t d_attn = arena.take(top_plane * top * f);
    const std::uint64_t d_scores = arena.take(static_cast<std::uint64_t>(kVaeAttentionChunk) * top_plane * f);
    const std::uint64_t d_partials = arena.take(static_cast<std::uint64_t>(d.vae_groups) * (64 * 2 * sizeof(double) + 2 * sizeof(float)));
    const std::uint64_t d_scale = arena.take(static_cast<std::uint64_t>(top) * f);
    const std::vector<float> query_scale(top, 1.0f / std::sqrt(static_cast<float>(top)));
    const std::uint64_t d_hwc = arena.take(plane_max * 2);
    VaeState vae{d, launch, d_partials, d_hwc};

    // Scale the latent into the decoder's range on the host: z / scale + shift.
    std::vector<float> scaled(static_cast<std::size_t>(d.latent_channels) * top_plane);
    for (std::size_t i = 0; i < scaled.size(); ++i) scaled[i] = latents[i] / d.latent_scale + d.latent_shift;
    std::uint64_t x = buffers[0], scratch = buffers[1], y = buffers[2];
    launch.upload(x, scaled.data(), scaled.size() * f);
    int height = latent_h, width = latent_w;
    vae.conv_from_f32(d.conv_in, x, y, height, width);
    std::swap(x, y);

    // Mid block: resnet, attention, resnet.
    vae.resnet(d.mid_resnet[0], x, scratch, y, height, width);
    std::swap(x, y);
    if (d.mid_has_attention) {
        const int plane = height * width;
        vae.group_norm(d.mid_attention.norm, x, scratch, 0, top, plane, false);
        // [C][HW] -> [HW][C] for the projections.
        {
            int rows = top, columns = plane;
            std::uint64_t input = scratch, output = y;
            void* arguments[] = {&input, &output, &rows, &columns};
            launch.run("diff_transpose", static_cast<std::uint32_t>((columns + 31) / 32),
                       static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
        }
        launch.gemm(d.mid_attention.q, y, d_q, plane, 0, 0);
        launch.gemm(d.mid_attention.k, y, d_k, plane, 0, 0);
        launch.gemm(d.mid_attention.v, y, d_attn, plane, 0, 0);
        {
            // v [HW][C] -> v^T [C][HW] so the P.V product is a plain GEMM.
            int rows = plane, columns = top;
            std::uint64_t input = d_attn, output = d_vt;
            void* arguments[] = {&input, &output, &rows, &columns};
            launch.run("diff_transpose", static_cast<std::uint32_t>((columns + 31) / 32),
                       static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
        }
        // The softmax scale folds into the queries once.
        launch.upload(d_scale, query_scale.data(), static_cast<std::uint64_t>(top) * f);
        launch.scale_columns(d_q, d_scale, top, plane);
        for (int start = 0; start < plane; start += kVaeAttentionChunk) {
            const int count = std::min(kVaeAttentionChunk, plane - start);
            const std::uint64_t q_chunk = d_q + static_cast<std::uint64_t>(start) * top * f;
            DiffMatrix keys;  // k as a [HW][C] "weight": scores = q . k^T
            keys.device = d_k; keys.type = 0; keys.in = top; keys.out = plane;
            launch.gemm(keys, q_chunk, d_scores, count, 0, 0);
            {
                std::uint64_t scores = d_scores;
                int columns = plane, rows = count;
                void* softmax_arguments[] = {&scores, &columns, &rows};
                launch.run("diff_softmax_rows", static_cast<std::uint32_t>(rows), 1, 256, softmax_arguments);
            }
            DiffMatrix values;  // v^T as a [C][HW] "weight": out = p . v
            values.device = d_vt; values.type = 0; values.in = plane; values.out = top;
            launch.gemm(values, d_scores, d_attn + static_cast<std::uint64_t>(start) * top * f, count, 0, 0);
        }
        launch.gemm(d.mid_attention.o, d_attn, y, plane, 0, 0);
        {
            // [HW][C] -> [C][HW], then the residual.
            int rows = plane, columns = top;
            std::uint64_t input = y, output = scratch;
            void* arguments[] = {&input, &output, &rows, &columns};
            launch.run("diff_transpose", static_cast<std::uint32_t>((columns + 31) / 32),
                       static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
        }
        launch.add(x, scratch, static_cast<std::uint64_t>(top) * plane);
    }
    vae.resnet(d.mid_resnet[1], x, scratch, y, height, width);
    std::swap(x, y);

    // Up blocks.
    for (int level = 0; level < 4; ++level) {
        for (const auto& resnet : d.up_resnets[level]) {
            vae.resnet(resnet, x, scratch, y, height, width);
            std::swap(x, y);
        }
        if (level < 3) {
            const int channels = d.vae_channels[3 - level];
            const bool tc = d.upsamplers[level].tensor_core();
            {
                int c = channels, h = height, w = width;
                std::uint64_t input = x, output = tc ? d_hwc : y;
                void* arguments[] = {&input, &output, &c, &h, &w};
                launch.run(tc ? "diff_upsample_nearest_2x_hwc_bf16" : "diff_upsample_nearest_2x",
                           DiffLaunch::grid_for(static_cast<std::uint64_t>(channels) * height * width * 4), 1, 256, arguments);
            }
            height *= 2;
            width *= 2;
            vae.conv(d.upsamplers[level], tc ? d_hwc : y, x, height, width);
        }
    }
    const int final_channels = d.vae_channels[0];
    vae.group_norm(d.norm_out, x, scratch, 0, final_channels, height * width, true);
    vae.conv(d.conv_out, scratch, y, height, width);
    launch.download(pixels, y, static_cast<std::uint64_t>(3) * height * width * f);
}

// The whole latent to RGB bytes: one tile when it fits, otherwise the tiled
// walk with blended overlaps. Pixels are (x / 2 + 0.5) rounded, as the
// pipeline's postprocess does.
void diff_decode_latents(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                         std::uint8_t* rgb) {
    const int channels = d.channels;
    const int height = latent_h * 8, width = latent_w * 8;
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    std::vector<float> image(3 * plane);
    if (latent_h <= kVaeTileLatent && latent_w <= kVaeTileLatent) {
        diff_decode_tile(d, latents, latent_h, latent_w, image.data());
    } else {
        // Tiles in latent space, each decoded on its own, then blended into
        // the rows above and the columns to the left over kVaeBlendPixels.
        std::vector<float> tile_latents, tile_pixels;
        const int row_limit = kVaeTileLatent * 8 - kVaeBlendPixels;   // pixels kept per tile
        std::vector<std::vector<float>> row_tiles;   // the previous tile row, for blend_v
        std::vector<int> row_tile_widths;
        std::vector<std::vector<float>> previous_row;
        std::vector<int> previous_widths;
        int previous_height = 0;
        int out_y = 0;
        for (int ty = 0; ty < latent_h; ty += kVaeTileStride) {
            const int th = std::min(kVaeTileLatent, latent_h - ty);
            row_tiles.clear();
            row_tile_widths.clear();
            for (int tx = 0; tx < latent_w; tx += kVaeTileStride) {
                const int tw = std::min(kVaeTileLatent, latent_w - tx);
                tile_latents.resize(static_cast<std::size_t>(channels) * th * tw);
                for (int c = 0; c < channels; ++c)
                    for (int y = 0; y < th; ++y)
                        std::memcpy(tile_latents.data() + (static_cast<std::size_t>(c) * th + y) * tw,
                                    latents + (static_cast<std::size_t>(c) * latent_h + ty + y) * latent_w + tx,
                                    static_cast<std::size_t>(tw) * sizeof(float));
                tile_pixels.assign(static_cast<std::size_t>(3) * th * 8 * tw * 8, 0.0f);
                diff_decode_tile(d, tile_latents.data(), th, tw, tile_pixels.data());
                row_tiles.push_back(tile_pixels);
                row_tile_widths.push_back(tw * 8);
            }
            const int tile_height = th * 8;
            // blend_v against the tile above, blend_h against the tile to the left.
            for (std::size_t i = 0; i < row_tiles.size(); ++i) {
                auto& tile = row_tiles[i];
                const int tw8 = row_tile_widths[i];
                if (!previous_row.empty()) {
                    const auto& above = previous_row[i];
                    const int blend = std::min({kVaeBlendPixels, previous_height, tile_height});
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < blend; ++y) {
                            const float weight = static_cast<float>(y) / blend;
                            for (int x = 0; x < tw8; ++x) {
                                float& value = tile[(static_cast<std::size_t>(c) * tile_height + y) * tw8 + x];
                                const float over = above[(static_cast<std::size_t>(c) * previous_height +
                                                          previous_height - blend + y) * tw8 + x];
                                value = over * (1.0f - weight) + value * weight;
                            }
                        }
                }
                if (i > 0) {
                    const auto& left = row_tiles[i - 1];
                    const int left_width = row_tile_widths[i - 1];
                    const int blend = std::min({kVaeBlendPixels, left_width, tw8});
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < tile_height; ++y)
                            for (int x = 0; x < blend; ++x) {
                                const float weight = static_cast<float>(x) / blend;
                                float& value = tile[(static_cast<std::size_t>(c) * tile_height + y) * tw8 + x];
                                const float over = left[(static_cast<std::size_t>(c) * tile_height + y) * left_width +
                                                        left_width - blend + x];
                                value = over * (1.0f - weight) + value * weight;
                            }
                }
            }
            // Crop each tile to its stride and write the row.
            const int keep_rows = std::min(row_limit, height - out_y);
            int out_x = 0;
            for (std::size_t i = 0; i < row_tiles.size(); ++i) {
                const int tw8 = row_tile_widths[i];
                const int keep_columns = std::min(row_limit, width - out_x);
                for (int c = 0; c < 3; ++c)
                    for (int y = 0; y < keep_rows; ++y)
                        std::memcpy(image.data() + c * plane + static_cast<std::size_t>(out_y + y) * width + out_x,
                                    row_tiles[i].data() + (static_cast<std::size_t>(c) * tile_height + y) * tw8,
                                    static_cast<std::size_t>(keep_columns) * sizeof(float));
                out_x += keep_columns;
                if (out_x >= width) break;
            }
            out_y += keep_rows;
            previous_row = row_tiles;
            previous_widths = row_tile_widths;
            previous_height = tile_height;
            if (out_y >= height) break;
        }
    }
    for (std::size_t pixel = 0; pixel < plane; ++pixel)
        for (int c = 0; c < 3; ++c) {
            const float value = (image[c * plane + pixel] * 0.5f + 0.5f) * 255.0f;
            rgb[pixel * 3 + c] = static_cast<std::uint8_t>(std::min(255.0f, std::max(0.0f, std::round(value))));
        }
}

// ---- Scheduler -------------------------------------------------------------

// FlowMatchEulerDiscreteScheduler with a constant shift and sigma_min = 0:
// sigmas = linspace(1, 0, steps) shifted by s / (1 + (shift - 1) s), with a
// trailing zero. The last step has zero width and is skipped.
std::vector<float> diff_sigmas(int steps, float shift) {
    std::vector<float> sigmas;
    for (int i = 0; i < steps; ++i) {
        const float linear = steps > 1 ? 1.0f - static_cast<float>(i) / static_cast<float>(steps - 1) : 1.0f;
        sigmas.push_back(shift * linear / (1.0f + (shift - 1.0f) * linear));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

// Standard normal samples from a 64-bit seed (splitmix64 + Box-Muller). Not
// torch's generator, so seeds are reproducible here, not across engines.
void diff_gaussian_noise(std::uint64_t seed, std::vector<float>& out) {
    std::uint64_t state = seed ^ 0x9e3779b97f4a7c15ull;
    auto next = [&state]() {
        state += 0x9e3779b97f4a7c15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };
    auto uniform = [&next]() {
        return (static_cast<double>(next() >> 11) + 0.5) / 9007199254740992.0;
    };
    for (std::size_t i = 0; i < out.size(); i += 2) {
        const double u1 = uniform(), u2 = uniform();
        const double radius = std::sqrt(-2.0 * std::log(u1));
        out[i] = static_cast<float>(radius * std::cos(2.0 * 3.14159265358979323846 * u2));
        if (i + 1 < out.size()) out[i + 1] = static_cast<float>(radius * std::sin(2.0 * 3.14159265358979323846 * u2));
    }
}

void diff_release(FlyweightV2Diffusion& d) {
    if (d.stream) flyweight_gpu_stream_sync(d.stream);
    if (d.copy_stream) flyweight_gpu_stream_sync(d.copy_stream);
    for (const auto pointer : d.allocations) flyweight_gpu_free(pointer);
    d.allocations.clear();
    for (auto& slot : d.staging) {
        if (slot) flyweight_gpu_free(slot);
        slot = 0;
    }
    for (auto* pinned : d.host_allocations) flyweight_gpu_host_free(pinned);
    d.host_allocations.clear();
    for (int slot = 0; slot < 2; ++slot) {
        if (d.staged_event[slot]) flyweight_gpu_event_destroy(d.staged_event[slot]);
        if (d.consumed_event[slot]) flyweight_gpu_event_destroy(d.consumed_event[slot]);
        d.staged_event[slot] = d.consumed_event[slot] = 0;
    }
    if (d.workspace) flyweight_gpu_free(d.workspace);
    d.workspace = 0;
    if (d.copy_stream) flyweight_gpu_stream_destroy(d.copy_stream);
    d.copy_stream = 0;
    if (d.stream) flyweight_gpu_stream_destroy(d.stream);
    d.stream = 0;
}

// Bytes of weights the encoder and DiT would put on the device.
std::uint64_t diff_layer_weight_bytes(const FlyweightV2Model& encoder, const FlyweightV2Model& transformer) {
    std::uint64_t total = 0;
    for (const auto& tensor : encoder.tensors) total += tensor.size;
    for (const auto& tensor : transformer.tensors) total += tensor.size;
    return total;
}

}  // namespace

// ---- C API -----------------------------------------------------------------

int flyweight_v2_diffusion_create(FlyweightV2Model* encoder, FlyweightV2Model* transformer,
                                  FlyweightV2Model* vae, int32_t device, uint32_t max_width,
                                  uint32_t max_height, uint32_t max_prompt_tokens, uint32_t weights,
                                  FlyweightV2Diffusion** out) { return guarded([&]{
    const bool exact = (weights & FLYWEIGHT_V2_DIFFUSION_EXACT) != 0;
    const bool balanced = (weights & FLYWEIGHT_V2_DIFFUSION_BALANCED) != 0 && !exact;
    weights &= ~(FLYWEIGHT_V2_DIFFUSION_EXACT | FLYWEIGHT_V2_DIFFUSION_BALANCED);
    if (weights > FLYWEIGHT_V2_DIFFUSION_WEIGHTS_AUTO)
        throw std::runtime_error("invalid diffusion weights placement");
    if (!encoder || !transformer || !vae || !out) throw std::runtime_error("invalid diffusion create arguments");
    if (encoder->config.architecture != "qwen3")
        throw std::runtime_error("the text encoder must be a plain Qwen3 checkpoint (got " +
                                 encoder->config.architecture + ")");
    if (transformer->config.architecture != "zimage-dit")
        throw std::runtime_error("the transformer must be a ZImageTransformer2DModel checkpoint");
    if (vae->config.architecture != "autoencoder-kl")
        throw std::runtime_error("the autoencoder must be an AutoencoderKL checkpoint");
    if (!max_width || !max_height || max_width % 16 || max_height % 16)
        throw std::runtime_error("max_width and max_height must be positive multiples of 16");
    if (!max_prompt_tokens) throw std::runtime_error("max_prompt_tokens must be positive");
    auto d = std::make_unique<FlyweightV2Diffusion>();
    d->encoder = encoder;
    d->transformer = transformer;
    d->vae = vae;
    d->device = device;
    d->max_width = max_width;
    d->max_height = max_height;
    d->max_prompt_tokens = max_prompt_tokens;
    d->exact = exact;
    d->balanced = balanced;
    if (flyweight_gpu_init(device) != 0) throw std::runtime_error("failed to initialize native CUDA runtime");
    {
        std::vector<std::string> option_storage;
#if !defined(_WIN32)
        for (const char* path : {"/opt/cuda/include", "/usr/local/cuda/include", "/usr/include"})
            if (access((std::string(path) + "/cuda_fp16.h").c_str(), R_OK) == 0) {
                option_storage.push_back(std::string("-I") + path);
                option_storage.push_back(std::string("-I") + path + "/cccl");
            }
#endif
        if (const char* cuda_home = std::getenv("CUDA_HOME")) {
            option_storage.push_back(std::string("-I") + cuda_home + "/include");
            option_storage.push_back(std::string("-I") + cuda_home + "/include/cccl");
        }
        std::vector<const char*> compile_options;
        for (const auto& option : option_storage) compile_options.push_back(option.c_str());
        std::array<char, 16384> compile_log{};
        const std::string cuda_source = qwen_cuda_corpus();
        if (flyweight_gpu_compile(cuda_source.c_str(), compile_options.data(),
                                  static_cast<int32_t>(compile_options.size()), device,
                                  compile_log.data(), compile_log.size()) != 0)
            throw std::runtime_error(std::string("failed to compile native diffusion CUDA kernels: ") +
                                     compile_log.data());
    }
    try {
        if (flyweight_gpu_stream_create(&d->stream) != 0) throw std::runtime_error("failed to create diffusion stream");
        FlyweightV2GpuInfo probe{};
        const bool probed = gpu_probe(probe, device) == 0;
        if (probed) d->int8_tensor_cores = probe.compute_major * 10 + probe.compute_minor >= 75;
        // FLYWEIGHT_DIFF_FLASH=0 keeps the f32 attention, for A/B against it.
        d->flash_attention = probed && probe.compute_major >= 8 && !flyweight_backend_is_cpu() &&
            !(std::getenv("FLYWEIGHT_DIFF_FLASH") && std::getenv("FLYWEIGHT_DIFF_FLASH")[0] == '0');
        // Auto: stream the weights when they would take more than half the
        // card, which is the case a chat model has to share it with.
        d->host_weights = weights == FLYWEIGHT_V2_DIFFUSION_WEIGHTS_HOST ||
            (weights == FLYWEIGHT_V2_DIFFUSION_WEIGHTS_AUTO && probed && probe.total_memory &&
             diff_layer_weight_bytes(*encoder, *transformer) * 2 > probe.total_memory);
        if (d->host_weights) {
            if (flyweight_gpu_stream_create(&d->copy_stream) != 0)
                throw std::runtime_error("failed to create diffusion copy stream");
            for (int slot = 0; slot < 2; ++slot)
                if (flyweight_gpu_event_create(&d->staged_event[slot]) != 0 ||
                    flyweight_gpu_event_create(&d->consumed_event[slot]) != 0)
                    throw std::runtime_error("failed to create diffusion staging events");
        }
        diff_upload_transformer(*d);
        diff_upload_encoder(*d);
        diff_upload_vae(*d);
        if (d->host_weights) {
            for (auto& slot : d->staging)
                if (flyweight_gpu_alloc(d->staging_bytes, &slot) != 0 || !slot)
                    throw std::runtime_error("diffusion staging allocation failed (" +
                                             std::to_string(d->staging_bytes / 1048576) + " MiB)");
        }
        // One arena for every phase, sized for the largest.
        const int latent_h = static_cast<int>(max_height / 8), latent_w = static_cast<int>(max_width / 8);
        const std::uint64_t image_rows = static_cast<std::uint64_t>(latent_h / d->patch) * (latent_w / d->patch);
        const std::uint64_t rows = (image_rows + kDiffSequenceMultiple - 1) / kDiffSequenceMultiple * kDiffSequenceMultiple +
            (static_cast<std::uint64_t>(max_prompt_tokens) + kDiffSequenceMultiple - 1) / kDiffSequenceMultiple * kDiffSequenceMultiple;
        std::uint64_t bytes = diff_transformer_workspace(*d, rows);
        bytes = std::max(bytes, diff_encoder_workspace(*d, max_prompt_tokens));
        bytes = std::max(bytes, diff_vae_workspace(*d, std::min(latent_h, kVaeTileLatent),
                                                   std::min(latent_w, kVaeTileLatent)));
        if (flyweight_gpu_alloc(bytes, &d->workspace) != 0 || !d->workspace)
            throw std::runtime_error("diffusion workspace allocation failed (" + std::to_string(bytes / 1048576) +
                                     " MiB); lower the maximum image size");
        d->workspace_bytes = bytes;
        if (d->host_weights)
            std::fprintf(stderr,
                         "[flyweight] diffusion: %llu MiB of weights pinned in host memory, streamed through "
                         "2 x %llu MiB; %llu MiB resident, %llu MiB workspace for up to %ux%u\n",
                         static_cast<unsigned long long>(d->host_bytes / 1048576),
                         static_cast<unsigned long long>(d->staging_bytes / 1048576),
                         static_cast<unsigned long long>(d->weight_bytes / 1048576),
                         static_cast<unsigned long long>(bytes / 1048576), max_width, max_height);
        else
            std::fprintf(stderr, "[flyweight] diffusion: %llu MiB of weights, %llu MiB workspace for up to %ux%u\n",
                         static_cast<unsigned long long>(d->weight_bytes / 1048576),
                         static_cast<unsigned long long>(bytes / 1048576), max_width, max_height);
    } catch (...) {
        diff_release(*d);
        throw;
    }
    *out = d.release();
    return 0;
}); }

int flyweight_v2_diffusion_info(const FlyweightV2Diffusion* d, FlyweightV2DiffusionInfo* out) { return guarded([&]{
    if (!d || !out) throw std::runtime_error("invalid diffusion info arguments");
    out->host_weights = d->host_weights ? 1u : 0u;
    out->exact = d->exact ? 1u : 0u;
    out->balanced = d->balanced ? 1u : 0u;
    out->device_bytes = d->weight_bytes + d->workspace_bytes + 2 * (d->host_weights ? d->staging_bytes : 0);
    out->host_bytes = d->host_bytes;
    out->max_width = d->max_width;
    out->max_height = d->max_height;
    return 0;
}); }

void flyweight_v2_diffusion_destroy(FlyweightV2Diffusion* d) {
    if (!d) return;
    diff_release(*d);
    delete d;
}

int flyweight_v2_diffusion_encode_text(FlyweightV2Diffusion* d, const uint32_t* tokens, uint64_t count,
                                       float* output, uint64_t capacity) { return guarded([&]{
    if (!d || !tokens || !output) throw std::runtime_error("invalid diffusion encode arguments");
    if (capacity < count * static_cast<std::uint64_t>(d->encoder_hidden))
        throw std::runtime_error("diffusion encode output buffer is too small");
    diff_encode_text(*d, tokens, count, output);
    return 0;
}); }

int flyweight_v2_diffusion_transformer_step(FlyweightV2Diffusion* d, const float* latents, uint32_t latent_h,
                                            uint32_t latent_w, const float* caption, uint64_t caption_tokens,
                                            float time, float* velocity) { return guarded([&]{
    if (!d || !latents || !caption || !velocity) throw std::runtime_error("invalid diffusion step arguments");
    diff_transformer_forward(*d, latents, static_cast<int>(latent_h), static_cast<int>(latent_w), caption,
                             caption_tokens, time, velocity);
    return 0;
}); }

int flyweight_v2_diffusion_decode_latents(FlyweightV2Diffusion* d, const float* latents, uint32_t latent_h,
                                          uint32_t latent_w, uint8_t* rgb) { return guarded([&]{
    if (!d || !latents || !rgb) throw std::runtime_error("invalid diffusion decode arguments");
    diff_decode_latents(*d, latents, static_cast<int>(latent_h), static_cast<int>(latent_w), rgb);
    return 0;
}); }

int flyweight_v2_diffusion_generate(FlyweightV2Diffusion* d, const uint32_t* tokens, uint64_t count,
                                    uint32_t width, uint32_t height, uint32_t steps, float shift,
                                    uint64_t seed, const float* initial_latents,
                                    FlyweightV2DiffusionProgress progress, void* user_data,
                                    uint8_t* rgb) { return guarded([&]{
    if (!d || !tokens || !rgb) throw std::runtime_error("invalid diffusion generate arguments");
    if (!width || !height || width % 16 || height % 16)
        throw std::runtime_error("width and height must be positive multiples of 16");
    if (width > d->max_width || height > d->max_height)
        throw std::runtime_error("image exceeds the size the tower was created for");
    if (!steps) throw std::runtime_error("steps must be positive");
    const int latent_h = static_cast<int>(height / 8), latent_w = static_cast<int>(width / 8);
    const std::size_t latent_count = static_cast<std::size_t>(d->channels) * latent_h * latent_w;
    std::vector<float> caption(count * static_cast<std::uint64_t>(d->encoder_hidden));
    diff_encode_text(*d, tokens, count, caption.data());
    if (progress && progress(user_data, 0, steps) != 0) throw std::runtime_error("cancelled");
    std::vector<float> latents(latent_count), velocity(latent_count);
    if (initial_latents) std::memcpy(latents.data(), initial_latents, latent_count * sizeof(float));
    else diff_gaussian_noise(seed, latents);
    const auto sigmas = diff_sigmas(static_cast<int>(steps), shift);
    for (std::uint32_t step = 0; step < steps; ++step) {
        const float sigma = sigmas[step], next = sigmas[step + 1];
        if (sigma == next) continue;
        diff_transformer_forward(*d, latents.data(), latent_h, latent_w, caption.data(), count, 1.0f - sigma,
                                 velocity.data());
        // The model predicts the negative velocity: x_next = x + (next - sigma) * (-out).
        for (std::size_t i = 0; i < latent_count; ++i) latents[i] -= (next - sigma) * velocity[i];
        if (progress && progress(user_data, step + 1, steps) != 0) throw std::runtime_error("cancelled");
    }
    diff_decode_latents(*d, latents.data(), latent_h, latent_w, rgb);
    return 0;
}); }
