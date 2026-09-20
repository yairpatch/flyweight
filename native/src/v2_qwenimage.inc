// Qwen-Image-2.1 on the diffusion tower: text encoder, single-stream DiT and
// the RGBA autoencoder's decoder. Included by v2_runtime.cpp right after
// v2_diffusion.inc, whose DiffMatrix / DiffLaunch / DiffStreamer / DiffArena
// and upload helpers this file builds on -- the two models share the tower,
// the workspace arena, the staging slots and the text encoder path, and
// differ in the DiT block, the sequence layout and the decoder.
//
// The reference is diffusers' pipeline_qwenimage21.py,
// transformer_qwenimage21.py and autoencoder_kl_qwenimage21.py;
// plans/qwenimage21-diffusion.md lists what carries over and what does not,
// tools/qwenimage21_reference.py dumps and
// tools/check_qwenimage21_parity.py compares.

namespace {

// One single-stream block. Modulation is shared across blocks and lives on
// the tower, so a block holds only its projections and q/k norms; img_norm1
// and img_norm2 are LayerNorms without affine parameters.
struct QiBlock {
    std::uint64_t q_norm = 0, k_norm = 0;
    DiffMatrix qkv, out, gate, up, down;
};

// `F.normalize(x, dim=1) * sqrt(C) * gamma`: a per-pixel RMS norm over the
// channel axis with no bias.
struct QiNorm {
    std::uint64_t gamma = 0;
    int channels = 0;
};

struct QiResnet {
    QiNorm norm1, norm2;
    VaeConv conv1, conv2, shortcut;
    int in = 0, out = 0;
};

struct QiUpBlock {
    std::vector<QiResnet> resnets;
    VaeConv upsampler;
    bool upsamples = false;
    // DupUp3D: the source channel of output (oc, 2h + hs, 2w + ws) is
    // (oc * factor + t_keep * 4 + hs * 2 + ws) / repeats.
    int factor = 0, repeats = 0, t_keep = 0;
    int in = 0, out = 0;
};

// Sequence layout of one forward: the prompt's rows first, then the target
// image's. Text rows attend causally among themselves and modulate from
// t = 0; image rows attend to everything and modulate from the sampled t.
struct QiLayout {
    std::uint64_t text = 0, image = 0, rows = 0;
    int latent_h = 0, latent_w = 0;
};

constexpr int kQiTimeEmbedWidth = 256;   // QwenImage21TemporalTimesteps
// Tiled decode. The decoder upsamples 16x, so a 32-latent tile is 512 pixels
// out; at that size its widest activation is 288 channels over 512x512, which
// is 302 MiB a plane. Three planes plus the block-input copy is the decoder's
// peak whatever the image size, as Z-Image's 64-latent tiles were at 8x.
constexpr int kQiTileLatent = 32;
constexpr int kQiTileStride = 24;
constexpr int kQiBlendPixels = 128;
// FlowMatchEulerDiscreteScheduler's dynamic shift, from the scheduler config.
constexpr float kQiBaseSeqLen = 256.0f, kQiMaxSeqLen = 8192.0f;
constexpr float kQiBaseShift = 0.5f, kQiMaxShift = 0.9f;
constexpr float kQiShiftTerminal = 0.02f;

}  // namespace

struct QwenImage21Tower {
    // DiT.
    DiffMatrix img_in, txt_in, txt_out, time_mlp0, time_mlp1, modulation;
    DiffMatrix final_scale, proj_out;
    std::uint64_t txt_norm = 0;     // text_norm.weight, with the stored 1 folded in
    std::vector<QiBlock> blocks;
    int dim = 0, heads = 0, head_dim = 0, intermediate = 0;
    int channels = 0, caption_dim = 0;
    int axis_dims[3] = {0, 0, 0};
    float epsilon = 1e-6f, rope_theta = 10000.0f, time_scale = 1000.0f;
    bool causal_condition = true;
    std::uint64_t rope_frequencies = 0;

    // Autoencoder decoder.
    VaeConv post_quant, conv_in, conv_out;
    QiNorm norm_out, mid_norm;
    QiResnet mid_resnet[2];
    DiffMatrix mid_q, mid_k, mid_v, mid_proj;
    std::vector<QiUpBlock> up_blocks;
    int latent_channels = 0, out_channels = 0, top_channels = 0, spatial_scale = 16;
    std::vector<float> latents_mean, latents_std;
};

namespace {

void qi_destroy(QwenImage21Tower* tower) { delete tower; }

int qi_caption_dim(const FlyweightV2Diffusion& d) { return d.qwenimage->caption_dim; }
int qi_spatial_scale(const FlyweightV2Diffusion& d) { return d.qwenimage->spatial_scale; }
int qi_output_channels(const FlyweightV2Diffusion& d) { return d.qwenimage->out_channels; }
int qi_latent_channels(const FlyweightV2Diffusion& d) { return d.qwenimage->latent_channels; }

// ---- Upload ----------------------------------------------------------------

// An f32 device vector widened from whatever the arena stores, with `offset`
// added to every element. The DiT's caption norm stores `scale - 1`, so
// folding 1.0 in here lets the ordinary RMSNorm kernel run over it.
std::uint64_t qi_upload_vector(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                               const std::string& name, std::uint64_t expected, float offset) {
    const Tensor& tensor = diff_tensor(model, name);
    const std::uint64_t elements = diff_elements(tensor);
    if (expected && elements != expected)
        throw std::runtime_error(name + " has " + std::to_string(elements) +
                                 " elements, expected " + std::to_string(expected));
    std::vector<float> widened(elements);
    const auto* source = tensor_data(model, tensor);
    for (std::uint64_t i = 0; i < elements; ++i)
        widened[i] = tensor_value(source, tensor.type, i) + offset;
    const std::uint64_t pointer = diff_alloc(d, elements * sizeof(float));
    if (flyweight_gpu_upload_sync(pointer, widened.data(), elements * sizeof(float)) != 0)
        throw std::runtime_error("diffusion tower upload failed for " + name);
    return pointer;
}

// A convolution kernel, widened to f32 on the way in. Qwen-Image-2.1's
// autoencoder ships bf16 where Z-Image's shipped f32, and the f32 kernel reads
// [out][in][k][k] floats, so the widening happens here rather than at load.
VaeConv qi_upload_conv(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                       const std::string& prefix, int in, int out, int kernel) {
    const Tensor& tensor = diff_tensor(model, prefix + ".weight");
    const std::uint64_t expected = static_cast<std::uint64_t>(out) * in * kernel * kernel;
    if (diff_elements(tensor) != expected)
        throw std::runtime_error(prefix + ".weight has the wrong shape");
    std::vector<float> weights(expected);
    const auto* source = tensor_data(model, tensor);
    for (std::uint64_t i = 0; i < expected; ++i) weights[i] = tensor_value(source, tensor.type, i);
    VaeConv conv;
    conv.in = in;
    conv.out = out;
    conv.kernel = kernel;
    conv.weight = diff_alloc(d, expected * sizeof(float));
    if (flyweight_gpu_upload_sync(conv.weight, weights.data(), expected * sizeof(float)) != 0)
        throw std::runtime_error("diffusion tower upload failed for " + prefix);
    conv.bias = diff_upload_f32(d, model, prefix + ".bias", out);
    if (d.flash_attention && !d.exact && in % 16 == 0 && out % 64 == 0) {
        const int taps = kernel * kernel;
        std::vector<std::uint16_t> packed(static_cast<std::size_t>(taps) * out * in);
        for (int co = 0; co < out; ++co)
            for (int ci = 0; ci < in; ++ci)
                for (int tap = 0; tap < taps; ++tap)
                    packed[(static_cast<std::size_t>(tap) * out + co) * in +
                           (ci / 16) * 16 + diff_channel_slot_host(ci % 16)] =
                        diff_bf16_bits(weights[(static_cast<std::size_t>(co) * in + ci) * taps + tap]);
        conv.weight_bf16 = diff_alloc(d, packed.size() * 2);
        if (flyweight_gpu_upload_sync(conv.weight_bf16, packed.data(), packed.size() * 2) != 0)
            throw std::runtime_error("diffusion tower upload failed for " + prefix + " (bf16 pack)");
    }
    return conv;
}

// A 1x1 convolution run as a GEMM. Its weight arrives shaped [1][1][in][out]
// rather than the [in][out] diff_upload_matrix expects, but the bytes are the
// same [out][in] row-major the GEMM reads, so only the shape check differs.
DiffMatrix qi_upload_conv_matrix(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                                 const std::string& prefix, int in, int out) {
    const Tensor& tensor = diff_tensor(model, prefix + ".weight");
    if (diff_elements(tensor) != static_cast<std::uint64_t>(in) * out)
        throw std::runtime_error(prefix + ".weight is not a " + std::to_string(in) + "x" +
                                 std::to_string(out) + " 1x1 convolution");
    DiffMatrix matrix;
    matrix.type = tensor.type;
    matrix.in = in;
    matrix.out = out;
    if (matrix.type != 0 && matrix.type != 30 && !flyweight::v2::qwen_format(matrix.type))
        throw std::runtime_error(prefix + ".weight is stored in a format without GEMM kernels");
    matrix.device = diff_upload_raw(d, model, tensor);
    matrix.bias = diff_upload_f32(d, model, prefix + ".bias", out);
    return matrix;
}

QiNorm qi_upload_norm(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                      const std::string& prefix, int channels) {
    QiNorm norm;
    norm.channels = channels;
    norm.gamma = diff_upload_f32(d, model, prefix + ".gamma", channels);
    return norm;
}

QiResnet qi_upload_resnet(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                          const std::string& prefix, int in, int out) {
    QiResnet resnet;
    resnet.in = in;
    resnet.out = out;
    resnet.norm1 = qi_upload_norm(d, model, prefix + ".norm1", in);
    resnet.conv1 = qi_upload_conv(d, model, prefix + ".conv1", in, out, 3);
    resnet.norm2 = qi_upload_norm(d, model, prefix + ".norm2", out);
    resnet.conv2 = qi_upload_conv(d, model, prefix + ".conv2", out, out, 3);
    if (in != out) resnet.shortcut = qi_upload_conv(d, model, prefix + ".conv_shortcut", in, out, 1);
    return resnet;
}

DiffLayerRef qi_block_matrices(QiBlock& block) {
    DiffLayerRef ref;
    ref.add(block.qkv);
    ref.add(block.out);
    ref.add(block.gate);
    ref.add(block.up);
    ref.add(block.down);
    return ref;
}

QiBlock qi_upload_block(FlyweightV2Diffusion& d, const FlyweightV2Model& model,
                        const std::string& prefix) {
    const QwenImage21Tower& q = *d.qwenimage;
    DiffGroup group(d);
    QiBlock block;
    block.q_norm = diff_upload_f32(d, model, prefix + ".attn.norm_q.weight", q.head_dim);
    block.k_norm = diff_upload_f32(d, model, prefix + ".attn.norm_k.weight", q.head_dim);
    block.qkv = diff_upload_matrix(d, model, prefix + ".attn.qkv.weight", q.dim, 3 * q.dim);
    block.out = diff_upload_matrix(d, model, prefix + ".attn.to_out.0.weight", q.dim, q.dim);
    // SwiGLU: gate_layer is the gated half, proj the value half, out the
    // projection back (the reference's `out(silu(gate_layer(x)) * proj(x))`).
    block.gate = diff_upload_matrix(d, model, prefix + ".img_mlp.gate_layer.weight",
                                    q.dim, q.intermediate);
    block.up = diff_upload_matrix(d, model, prefix + ".img_mlp.proj.weight", q.dim, q.intermediate);
    block.down = diff_upload_matrix(d, model, prefix + ".img_mlp.out.weight", q.intermediate, q.dim);
    return block;
}

void qi_upload_transformer(FlyweightV2Diffusion& d) {
    if (!d.qwenimage) d.qwenimage = new QwenImage21Tower();
    QwenImage21Tower& q = *d.qwenimage;
    const auto& model = *d.transformer;
    const auto& config = model.config;
    q.dim = static_cast<int>(config.hidden_size);
    q.heads = static_cast<int>(config.attention_heads);
    q.head_dim = static_cast<int>(config.attention_head_dim);
    q.intermediate = static_cast<int>(config.intermediate_size);
    q.channels = static_cast<int>(config.in_channels);
    q.caption_dim = static_cast<int>(config.caption_dim);
    q.epsilon = config.rms_norm_epsilon;
    q.rope_theta = config.rope_freq_base;
    q.time_scale = config.time_scale;
    q.causal_condition = config.causal_condition;
    for (int axis = 0; axis < 3; ++axis) q.axis_dims[axis] = static_cast<int>(config.rope_sections[axis]);
    // The tensor-core attention wants exactly 128; anything narrower takes the
    // f32 rows kernel, which is what a small test checkpoint exercises.
    if (q.head_dim > 128) throw std::runtime_error("DiT head width above 128 is not supported");
    if (!q.causal_condition)
        throw std::runtime_error("QwenImage21Transformer2DModel without causal_condition is "
                                 "not supported");
    {
        // One ramp per rotary pair, float64 as the reference builds it.
        std::vector<double> frequencies;
        for (int axis = 0; axis < 3; ++axis)
            for (int j = 0; j < q.axis_dims[axis] / 2; ++j)
                frequencies.push_back(1.0 / std::pow(static_cast<double>(q.rope_theta),
                                                     static_cast<double>(2 * j) / q.axis_dims[axis]));
        if (static_cast<int>(frequencies.size()) != q.head_dim / 2)
            throw std::runtime_error("DiT rope axes do not cover head_dim");
        q.rope_frequencies = diff_upload_doubles(d, frequencies);
    }
    q.img_in = diff_upload_matrix(d, model, "img_in.weight", q.channels, q.dim);
    q.txt_norm = qi_upload_vector(d, model, "txt_in.text_norm.weight", q.caption_dim, 1.0f);
    q.txt_in = diff_upload_matrix(d, model, "txt_in.in_layer.weight", q.caption_dim, q.dim);
    q.txt_out = diff_upload_matrix(d, model, "txt_in.out_layer.weight", q.dim, q.dim);
    q.time_mlp0 = diff_upload_matrix(d, model, "time_text_embed.timestep_embedder.linear_1.weight",
                                     kQiTimeEmbedWidth, q.dim);
    q.time_mlp1 = diff_upload_matrix(d, model, "time_text_embed.timestep_embedder.linear_2.weight",
                                     q.dim, q.dim);
    q.modulation = diff_upload_matrix(d, model, "modulation.1.weight", q.dim, 4 * q.dim);
    q.final_scale = diff_upload_matrix(d, model, "norm_out.linear.weight", q.dim, q.dim);
    q.proj_out = diff_upload_matrix(d, model, "proj_out.weight", q.dim, q.channels);
    q.blocks.reserve(config.layer_count);
    for (std::uint32_t index = 0; index < config.layer_count; ++index)
        q.blocks.push_back(qi_upload_block(d, model, "transformer_blocks." + std::to_string(index)));
}

void qi_upload_vae(FlyweightV2Diffusion& d) {
    QwenImage21Tower& q = *d.qwenimage;
    const auto& model = *d.vae;
    const auto& config = model.config;
    q.latent_channels = static_cast<int>(config.latent_channels);
    q.out_channels = static_cast<int>(config.in_channels);
    q.spatial_scale = static_cast<int>(config.vae_spatial_scale);
    q.latents_mean = config.latents_mean;
    q.latents_std = config.latents_std;
    if (q.latent_channels != q.channels)
        throw std::runtime_error("autoencoder z_dim does not match the DiT's in_channels");
    const int levels = static_cast<int>(config.vae_level_count);
    const int top = static_cast<int>(config.vae_level_channels[levels - 1]);
    q.top_channels = top;
    q.post_quant = qi_upload_conv(d, model, "post_quant_conv", q.latent_channels, q.latent_channels, 1);
    q.conv_in = qi_upload_conv(d, model, "decoder.conv_in", q.latent_channels, top, 3);
    q.mid_resnet[0] = qi_upload_resnet(d, model, "decoder.mid_block.resnets.0", top, top);
    q.mid_resnet[1] = qi_upload_resnet(d, model, "decoder.mid_block.resnets.1", top, top);
    {
        const std::string prefix = "decoder.mid_block.attentions.0";
        q.mid_norm = qi_upload_norm(d, model, prefix + ".norm", top);
        // to_qkv is one 1x1 convolution; the GEMM reads its weight as
        // [out][in], so q, k and v are three contiguous slices of it.
        const DiffMatrix fused = qi_upload_conv_matrix(d, model, prefix + ".to_qkv", top, 3 * top);
        const std::uint64_t element = fused.type == 30 ? 2 : 4;
        DiffMatrix* slices[3] = {&q.mid_q, &q.mid_k, &q.mid_v};
        for (int piece = 0; piece < 3; ++piece) {
            *slices[piece] = fused;
            slices[piece]->in = top;
            slices[piece]->out = top;
            slices[piece]->device =
                fused.device + static_cast<std::uint64_t>(piece) * top * top * element;
            slices[piece]->bias =
                fused.bias + static_cast<std::uint64_t>(piece) * top * sizeof(float);
        }
        q.mid_proj = qi_upload_conv_matrix(d, model, prefix + ".proj", top, top);
    }
    // dims = [base * m for m in [mult[-1]] + reversed(mult)], so up block i
    // maps dims[i] -> dims[i + 1] and every level but the last upsamples.
    const int resnets_per_block = static_cast<int>(config.vae_layers_per_block) + 1;
    for (int level = 0; level < levels; ++level) {
        const std::string prefix = "decoder.up_blocks." + std::to_string(level);
        QiUpBlock block;
        block.in = level == 0 ? top : static_cast<int>(config.vae_level_channels[levels - level]);
        block.out = static_cast<int>(config.vae_level_channels[levels - 1 - level]);
        block.upsamples = level != levels - 1;
        int in = block.in;
        for (int index = 0; index < resnets_per_block; ++index) {
            block.resnets.push_back(qi_upload_resnet(
                d, model, prefix + ".resnets." + std::to_string(index), in, block.out));
            in = block.out;
        }
        if (block.upsamples) {
            block.upsampler = qi_upload_conv(d, model, prefix + ".upsampler.resample.1",
                                             block.out, block.out, 3);
            // factor_t is 2 on a temporally upsampling level; at one frame the
            // duplicating shortcut then keeps only the last temporal slot.
            const int factor_t = config.vae_temporal_upsample[level] ? 2 : 1;
            block.factor = factor_t * 4;
            block.t_keep = factor_t - 1;
            const std::uint64_t repeats =
                static_cast<std::uint64_t>(block.out) * block.factor / block.in;
            if (repeats * block.in != static_cast<std::uint64_t>(block.out) * block.factor)
                throw std::runtime_error("AutoencoderKLQwenImage21 up block " +
                                         std::to_string(level) +
                                         " has a shortcut that does not divide evenly");
            block.repeats = static_cast<int>(repeats);
        }
        q.up_blocks.push_back(std::move(block));
    }
    q.norm_out = qi_upload_norm(d, model, "decoder.norm_out",
                                static_cast<int>(config.vae_level_channels[0]));
    q.conv_out = qi_upload_conv(d, model, "decoder.conv_out",
                                static_cast<int>(config.vae_level_channels[0]), q.out_channels, 3);
}

// ---- Workspace -------------------------------------------------------------

std::uint64_t qi_transformer_workspace(const FlyweightV2Diffusion& d, std::uint64_t rows,
                                       std::uint64_t prompt_rows) {
    const QwenImage21Tower& q = *d.qwenimage;
    const std::uint64_t f = sizeof(float);
    std::uint64_t total = 0;
    total += diff_align(rows * q.dim * f) * 3;                  // hidden, normed, attn
    total += diff_align(rows * 3 * q.dim * f);                  // qkv
    total += diff_align(rows * q.intermediate * f) * 2;         // gate, up
    total += diff_align(prompt_rows * q.caption_dim * f) * 2;   // caption rows, normed
    total += diff_align(rows * 3 * sizeof(std::int32_t));       // positions
    total += diff_align(rows * q.channels * f) * 2;             // latent rows in, out
    total += diff_align(rows * q.dim * 2) * 3;                  // bf16 q, k, v planes
    total += diff_align(2 * 4 * q.dim * f);                     // modulation, t and t = 0
    total += diff_align(2 * kQiTimeEmbedWidth * f) + diff_align(2 * q.dim * f) * 2;
    total += diff_align(q.dim * f);                             // the final scale
    total += diff_q8_bytes(rows, std::max(q.intermediate, q.dim));
    return total;
}

// The widest activation the decoder holds, in floats. Every level's output at
// the resolution it leaves behind, plus the resnet working planes.
std::uint64_t qi_vae_plane_max(const FlyweightV2Diffusion& d, int latent_h, int latent_w) {
    const QwenImage21Tower& q = *d.qwenimage;
    std::uint64_t plane_max = static_cast<std::uint64_t>(q.top_channels) * latent_h * latent_w;
    std::uint64_t height = latent_h, width = latent_w;
    for (const auto& block : q.up_blocks) {
        plane_max = std::max(plane_max, static_cast<std::uint64_t>(block.out) * height * width);
        if (block.upsamples) {
            height *= 2;
            width *= 2;
            plane_max = std::max(plane_max, static_cast<std::uint64_t>(block.out) * height * width);
        }
    }
    return plane_max;
}

// The block input a residual up block has to keep alive across its resnets so
// the duplicating shortcut can read it.
std::uint64_t qi_vae_copy_max(const FlyweightV2Diffusion& d, int latent_h, int latent_w) {
    const QwenImage21Tower& q = *d.qwenimage;
    std::uint64_t copy_max = 0, height = latent_h, width = latent_w;
    for (const auto& block : q.up_blocks) {
        if (!block.upsamples) continue;
        copy_max = std::max(copy_max, static_cast<std::uint64_t>(block.in) * height * width);
        height *= 2;
        width *= 2;
    }
    return copy_max;
}

std::uint64_t qi_vae_workspace(const FlyweightV2Diffusion& d, int latent_h, int latent_w);

// The arena the DiT and the decoder need at the tower's maximum size. The
// decoder never sees more than one tile at a time, whatever the image.
std::uint64_t qi_workspace(const FlyweightV2Diffusion& d, std::uint32_t max_width,
                           std::uint32_t max_height, std::uint32_t max_prompt_tokens) {
    const int scale = d.qwenimage->spatial_scale;
    const int latent_h = static_cast<int>(max_height / scale);
    const int latent_w = static_cast<int>(max_width / scale);
    const std::uint64_t rows = static_cast<std::uint64_t>(latent_h) * latent_w + max_prompt_tokens;
    return std::max(qi_transformer_workspace(d, rows, max_prompt_tokens),
                    qi_vae_workspace(d, std::min(latent_h, kQiTileLatent),
                                     std::min(latent_w, kQiTileLatent)));
}

std::uint64_t qi_vae_workspace(const FlyweightV2Diffusion& d, int latent_h, int latent_w) {
    const QwenImage21Tower& q = *d.qwenimage;
    const std::uint64_t f = sizeof(float);
    const std::uint64_t plane_max = qi_vae_plane_max(d, latent_h, latent_w);
    const std::uint64_t top_plane = static_cast<std::uint64_t>(latent_h) * latent_w;
    const int top = q.top_channels;
    std::uint64_t total = diff_align(plane_max * f) * 3;
    total += diff_align(qi_vae_copy_max(d, latent_h, latent_w) * f);
    total += diff_align(plane_max * 2);                       // channels-last bf16 conv input
    total += diff_align(plane_max * f);                       // per-pixel norm reciprocals
    total += diff_align(top_plane * top * f) * 4;             // q, k, v^T, attention out
    total += diff_align(static_cast<std::uint64_t>(kVaeAttentionChunk) * top_plane * f);
    total += diff_align(static_cast<std::uint64_t>(top) * f); // query scale
    return total;
}

// ---- DiT -------------------------------------------------------------------

struct QiBuffers {
    std::uint64_t hidden = 0, normed = 0, attn = 0, qkv = 0, gate = 0, up = 0;
    std::uint64_t caption = 0, caption_normed = 0, positions = 0, latent_in = 0, latent_out = 0;
    std::uint64_t modulation = 0, time_frequencies = 0, time_mid = 0, time_embed = 0, final_scale = 0;
    std::uint64_t q8 = 0, q8_scales = 0, q16 = 0, k16 = 0, v16 = 0;
};

QiBuffers qi_transformer_buffers(FlyweightV2Diffusion& d, DiffArena& arena, std::uint64_t rows,
                                 std::uint64_t prompt_rows) {
    const QwenImage21Tower& q = *d.qwenimage;
    const std::uint64_t f = sizeof(float);
    QiBuffers b;
    b.hidden = arena.take(rows * q.dim * f);
    b.normed = arena.take(rows * q.dim * f);
    b.attn = arena.take(rows * q.dim * f);
    b.qkv = arena.take(rows * 3 * q.dim * f);
    b.gate = arena.take(rows * q.intermediate * f);
    b.up = arena.take(rows * q.intermediate * f);
    b.caption = arena.take(prompt_rows * q.caption_dim * f);
    b.caption_normed = arena.take(prompt_rows * q.caption_dim * f);
    b.positions = arena.take(rows * 3 * sizeof(std::int32_t));
    b.latent_in = arena.take(rows * q.channels * f);
    b.latent_out = arena.take(rows * q.channels * f);
    b.q16 = arena.take(rows * q.dim * 2);
    b.k16 = arena.take(rows * q.dim * 2);
    b.v16 = arena.take(rows * q.dim * 2);
    b.modulation = arena.take(2 * 4 * q.dim * f);
    b.time_frequencies = arena.take(2 * kQiTimeEmbedWidth * f);
    b.time_mid = arena.take(2 * q.dim * f);
    b.time_embed = arena.take(2 * q.dim * f);
    b.final_scale = arena.take(q.dim * f);
    const int q8_width = std::max(q.intermediate, q.dim);
    b.q8 = arena.take(rows * q8_width * 2);
    b.q8_scales = arena.take(rows * (q8_width / 32) * 4);
    return b;
}

// One block over the joint sequence. Every GEMM and norm runs over all rows;
// what splits is the modulation (text rows read the t = 0 row of `modulation`,
// image rows the sampled one) and the attention, whose block-causal mask is
// exactly "text rows causal over text, image rows full over everything".
void qi_run_block(FlyweightV2Diffusion& d, const DiffLaunch& launch, QiBlock& block,
                  const QiBuffers& b, const QiLayout& layout) {
    const QwenImage21Tower& q = *d.qwenimage;
    const std::uint64_t f = sizeof(float);
    const int dim = q.dim;
    const std::uint64_t rows = layout.rows, text = layout.text, image = layout.image;
    // modulation is [2][4 * dim]: row 0 is the sampled timestep, row 1 is
    // t = 0. Within a row: 1 + scale_attention, tanh gate, 1 + scale_mlp, gate.
    const std::uint64_t sampled = b.modulation;
    const std::uint64_t zero = b.modulation + static_cast<std::uint64_t>(4 * dim) * f;
    const auto slot = [&](std::uint64_t row, int index) {
        return row + static_cast<std::uint64_t>(index) * dim * f;
    };
    const auto scale_split = [&](std::uint64_t buffer, int index) {
        if (text) launch.scale_columns(buffer, slot(zero, index), dim, text);
        launch.scale_columns(buffer + text * dim * f, slot(sampled, index), dim, image);
    };
    const auto gated_add_split = [&](std::uint64_t target, std::uint64_t source, int index) {
        if (text) launch.gated_add(target, source, slot(zero, index), dim, text);
        launch.gated_add(target + text * dim * f, source + text * dim * f,
                         slot(sampled, index), dim, image);
    };

    launch.layer_norm(b.hidden, b.normed, dim, rows, q.epsilon);
    scale_split(b.normed, 0);
    launch.gemm(block.qkv, b.normed, b.qkv, rows, b.q8, b.q8_scales);
    launch.head_rms_norm(b.qkv, block.q_norm, q.heads, q.head_dim, rows, 3 * dim, 0, q.epsilon);
    launch.head_rms_norm(b.qkv, block.k_norm, q.heads, q.head_dim, rows, 3 * dim, dim, q.epsilon);
    {
        int heads = q.heads, head_dim = q.head_dim, count = static_cast<int>(rows);
        int axis0 = q.axis_dims[0], axis1 = q.axis_dims[1];
        std::uint64_t buffer = b.qkv, positions = b.positions, table = q.rope_frequencies;
        void* arguments[] = {&buffer, &positions, &table, &heads, &head_dim, &count, &axis0, &axis1};
        launch.run("diff_rope_axes_rows", count, 2, 256, arguments);
    }
    {
        const std::uint64_t k = b.qkv + static_cast<std::uint64_t>(dim) * f;
        const std::uint64_t v = b.qkv + static_cast<std::uint64_t>(2 * dim) * f;
        // The first call packs the whole sequence into the bf16 planes; the
        // second reuses them.
        if (text)
            launch.attention(b.qkv, k, v, b.attn, q.heads, q.heads, q.head_dim, rows,
                             3 * dim, 3 * dim, true, b.q16, b.k16, b.v16, 0, text, text, false);
        launch.attention(b.qkv, k, v, b.attn, q.heads, q.heads, q.head_dim, rows,
                         3 * dim, 3 * dim, false, b.q16, b.k16, b.v16, text, image, rows,
                         text != 0);
    }
    launch.gemm(block.out, b.attn, b.normed, rows, b.q8, b.q8_scales);
    gated_add_split(b.hidden, b.normed, 1);

    launch.layer_norm(b.hidden, b.normed, dim, rows, q.epsilon);
    scale_split(b.normed, 2);
    launch.gemm(block.gate, b.normed, b.gate, rows, b.q8, b.q8_scales);
    launch.gemm(block.up, b.normed, b.up, rows, b.q8, b.q8_scales);
    launch.silu_mul(b.gate, b.up, b.gate, rows * q.intermediate);
    launch.gemm(block.down, b.gate, b.normed, rows, b.q8, b.q8_scales);
    gated_add_split(b.hidden, b.normed, 3);
}

// Positions over the joint sequence. Text tokens advance one shared position
// on all three axes; the image block freezes the frame axis at the text length
// and lays its tokens out on height and width ramps centred on zero, so a
// block's spatial positions do not depend on where it sits in the sequence.
void qi_positions(const QiLayout& layout, std::vector<std::int32_t>& positions) {
    positions.assign(layout.rows * 3, 0);
    for (std::uint64_t row = 0; row < layout.text; ++row) {
        const auto value = static_cast<std::int32_t>(row);
        positions[row * 3] = positions[row * 3 + 1] = positions[row * 3 + 2] = value;
    }
    const int height = layout.latent_h, width = layout.latent_w;
    const int first_h = -(height - height / 2), first_w = -(width - width / 2);
    for (int h = 0; h < height; ++h)
        for (int w = 0; w < width; ++w) {
            const std::uint64_t row = layout.text + static_cast<std::uint64_t>(h) * width + w;
            positions[row * 3] = static_cast<std::int32_t>(layout.text);
            positions[row * 3 + 1] = static_cast<std::int32_t>(first_h + h);
            positions[row * 3 + 2] = static_cast<std::int32_t>(first_w + w);
        }
}

// One forward: the model's output for `latents` at time `t` (the sampler's
// sigma) under `caption` [caption_tokens][caption_dim], written to `velocity`
// as [channels][latent_h][latent_w].
void qi_transformer_forward(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                            const float* caption, std::uint64_t caption_tokens, float t,
                            float* velocity) {
    QwenImage21Tower& q = *d.qwenimage;
    if (!caption_tokens) throw std::runtime_error("empty caption");
    QiLayout layout;
    layout.latent_h = latent_h;
    layout.latent_w = latent_w;
    layout.text = caption_tokens;
    layout.image = static_cast<std::uint64_t>(latent_h) * latent_w;
    layout.rows = layout.text + layout.image;
    const std::uint64_t f = sizeof(float);
    const int dim = q.dim, channels = q.channels;
    diff_ensure_workspace(d, qi_transformer_workspace(d, layout.rows, caption_tokens), "image");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const QiBuffers b = qi_transformer_buffers(d, arena, layout.rows, caption_tokens);
    // Host staging for the uploads below; alive until the final download syncs.
    std::vector<float> frequencies(2 * kQiTimeEmbedWidth);
    std::vector<std::int32_t> positions;
    std::vector<float> rows(layout.image * channels);

    // Timestep embedding, two rows: the sampled time and t = 0, which is what
    // causal_condition modulates the text tokens from.
    {
        const int half = kQiTimeEmbedWidth / 2;
        for (int row = 0; row < 2; ++row) {
            const double scaled = (row == 0 ? static_cast<double>(t) : 0.0) * q.time_scale;
            for (int i = 0; i < half; ++i) {
                const double frequency = std::exp(-std::log(10000.0) * i / half);
                const double angle = scaled * frequency;
                frequencies[row * kQiTimeEmbedWidth + i] = static_cast<float>(std::cos(angle));
                frequencies[row * kQiTimeEmbedWidth + half + i] = static_cast<float>(std::sin(angle));
            }
        }
        launch.upload(b.time_frequencies, frequencies.data(), frequencies.size() * f);
        launch.gemm(q.time_mlp0, b.time_frequencies, b.time_mid, 2, 0, 0);
        launch.silu(b.time_mid, 2 * dim);
        launch.gemm(q.time_mlp1, b.time_mid, b.time_embed, 2, 0, 0);
        // modulation = Linear(SiLU(temb)); time_mid is free again by now.
        if (flyweight_gpu_copy_device(b.time_mid, b.time_embed, 2 * dim * f, d.stream) != 0)
            throw std::runtime_error("diffusion modulation copy failed");
        launch.silu(b.time_mid, 2 * dim);
        launch.gemm(q.modulation, b.time_mid, b.modulation, 2, 0, 0);
        for (int row = 0; row < 2; ++row) {
            int width = dim;
            std::uint64_t modulation = b.modulation + static_cast<std::uint64_t>(row) * 4 * dim * f;
            void* arguments[] = {&modulation, &width};
            launch.run("diff_modulation_prepare",
                       static_cast<std::uint32_t>((4 * dim + 255) / 256), 1, 256, arguments);
        }
    }

    qi_positions(layout, positions);
    launch.upload(b.positions, positions.data(), positions.size() * sizeof(std::int32_t));

    // Text rows: zero-centred RMSNorm, linear, GELU, linear.
    launch.upload(b.caption, caption, caption_tokens * q.caption_dim * f);
    launch.rms_norm(b.caption, q.txt_norm, b.caption_normed, q.caption_dim, caption_tokens, q.epsilon);
    launch.gemm(q.txt_in, b.caption_normed, b.hidden, caption_tokens, b.q8, b.q8_scales);
    {
        int elements = static_cast<int>(caption_tokens * dim);
        std::uint64_t hidden = b.hidden;
        void* arguments[] = {&hidden, &elements};
        launch.run("vision_gelu_rows", DiffLaunch::grid_for(caption_tokens * dim), 1, 256, arguments);
    }
    launch.gemm(q.txt_out, b.hidden, b.normed, caption_tokens, b.q8, b.q8_scales);
    if (flyweight_gpu_copy_device(b.hidden, b.normed, caption_tokens * dim * f, d.stream) != 0)
        throw std::runtime_error("diffusion caption copy failed");

    // Image rows: the latent unpatched, one row of `channels` per pixel.
    for (std::uint64_t pixel = 0; pixel < layout.image; ++pixel)
        for (int c = 0; c < channels; ++c)
            rows[pixel * channels + c] = latents[static_cast<std::uint64_t>(c) * layout.image + pixel];
    launch.upload(b.latent_in, rows.data(), rows.size() * f);
    launch.gemm(q.img_in, b.latent_in, b.hidden + layout.text * dim * f, layout.image,
                b.q8, b.q8_scales);

    DiffStreamer streamer{d, {}, 0};
    for (auto& block : q.blocks) streamer.refs.push_back(qi_block_matrices(block));
    for (std::size_t index = 0; index < q.blocks.size(); ++index) {
        streamer.begin(index);
        qi_run_block(d, launch, q.blocks[index], b, layout);
        streamer.end(index);
    }

    // The output is read off the image rows only, through a scale-only adaLN
    // from the sampled timestep.
    {
        if (flyweight_gpu_copy_device(b.time_mid, b.time_embed, dim * f, d.stream) != 0)
            throw std::runtime_error("diffusion final-layer copy failed");
        launch.silu(b.time_mid, dim);
        launch.gemm(q.final_scale, b.time_mid, b.final_scale, 1, 0, 0);
        {
            // 1 + scale: the prepare kernel's even chunks are the scales, and
            // one dim-wide vector is entirely chunk 0.
            int width = dim;
            std::uint64_t scale = b.final_scale;
            void* arguments[] = {&scale, &width};
            launch.run("diff_modulation_prepare", static_cast<std::uint32_t>((dim + 255) / 256),
                       1, 256, arguments);
        }
        const std::uint64_t image_hidden = b.hidden + layout.text * dim * f;
        launch.layer_norm(image_hidden, b.normed, dim, layout.image, q.epsilon);
        launch.scale_columns(b.normed, b.final_scale, dim, layout.image);
        launch.gemm(q.proj_out, b.normed, b.latent_out, layout.image, b.q8, b.q8_scales);
    }
    launch.download(rows.data(), b.latent_out, layout.image * channels * f);
    for (std::uint64_t pixel = 0; pixel < layout.image; ++pixel)
        for (int c = 0; c < channels; ++c)
            velocity[static_cast<std::uint64_t>(c) * layout.image + pixel] = rows[pixel * channels + c];
}

// ---- Autoencoder decoder ---------------------------------------------------

struct QiVaeState {
    FlyweightV2Diffusion& d;
    DiffLaunch launch;
    std::uint64_t scales = 0;   // one reciprocal per pixel
    std::uint64_t hwc = 0;      // channels-last bf16 staging for the tensor-core convs

    void conv(const VaeConv& conv, std::uint64_t input, std::uint64_t output,
              int height, int width) const {
        int in = conv.in, out = conv.out, h = height, w = width, kernel = conv.kernel;
        std::uint64_t bias = conv.bias;
        if (conv.tensor_core()) {
            std::uint64_t weight = conv.weight_bf16;
            void* arguments[] = {&input, &weight, &bias, &output, &in, &out, &h, &w, &kernel};
            launch.run("diff_conv2d_bf16", static_cast<std::uint32_t>((width + 127) / 128),
                       static_cast<std::uint32_t>(height) * static_cast<std::uint32_t>(out / 64),
                       128, arguments);
            return;
        }
        std::uint64_t weight = conv.weight;
        void* arguments[] = {&input, &weight, &bias, &output, &in, &out, &h, &w, &kernel};
        launch.run("diff_conv2d", static_cast<std::uint32_t>((width + 31) / 32),
                   static_cast<std::uint32_t>((height + 31) / 32) *
                       static_cast<std::uint32_t>((out + 3) / 4), 256, arguments);
    }

    void norm(const QiNorm& norm, std::uint64_t input, std::uint64_t output,
              std::uint64_t output_hwc, int plane, bool silu) const {
        int channels = norm.channels, p = plane, silu_flag = silu ? 1 : 0;
        std::uint64_t scale = scales, gamma = norm.gamma;
        void* stats[] = {&input, &scale, &channels, &p};
        launch.run("qi_channel_rms_stats", DiffLaunch::grid_for(plane), 1, 256, stats);
        void* apply[] = {&input, &scale, &gamma, &output, &output_hwc, &channels, &p, &silu_flag};
        launch.run("qi_channel_rms_apply", static_cast<std::uint32_t>((plane + 31) / 32),
                   static_cast<std::uint32_t>((channels + 31) / 32), 256, apply);
    }

    void to_hwc(std::uint64_t input, int channels, int plane) const {
        int c = channels, p = plane;
        std::uint64_t out = hwc;
        void* arguments[] = {&input, &out, &c, &p};
        launch.run("diff_to_hwc_bf16", static_cast<std::uint32_t>((plane + 31) / 32),
                   static_cast<std::uint32_t>((channels + 31) / 32), 256, arguments);
    }

    void conv_from_f32(const VaeConv& conv, std::uint64_t input, std::uint64_t output,
                       int height, int width) const {
        if (conv.tensor_core()) {
            to_hwc(input, conv.in, height * width);
            this->conv(conv, hwc, output, height, width);
        } else {
            this->conv(conv, input, output, height, width);
        }
    }

    // `output` may not alias `x`; `scratch` is clobbered.
    void resnet(const QiResnet& r, std::uint64_t x, std::uint64_t scratch, std::uint64_t output,
                int height, int width) const {
        const int plane = height * width;
        const bool tc1 = r.conv1.tensor_core(), tc2 = r.conv2.tensor_core();
        norm(r.norm1, x, scratch, tc1 ? hwc : 0, plane, true);
        conv(r.conv1, tc1 ? hwc : scratch, output, height, width);
        norm(r.norm2, output, scratch, tc2 ? hwc : 0, plane, true);
        conv(r.conv2, tc2 ? hwc : scratch, output, height, width);
        if (r.in != r.out) {
            conv_from_f32(r.shortcut, x, scratch, height, width);
            launch.add(output, scratch, static_cast<std::uint64_t>(r.out) * plane);
        } else {
            launch.add(output, x, static_cast<std::uint64_t>(r.out) * plane);
        }
    }
};

// Decodes one latent tile [C][latent_h][latent_w], already denormalized, into
// f32 pixels [out_channels][16 * latent_h][16 * latent_w].
void qi_decode_tile(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                    float* pixels) {
    const QwenImage21Tower& q = *d.qwenimage;
    diff_ensure_workspace(d, qi_vae_workspace(d, latent_h, latent_w), "image");
    const std::uint64_t f = sizeof(float);
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const std::uint64_t plane_max = qi_vae_plane_max(d, latent_h, latent_w);
    std::uint64_t buffers[3];
    for (auto& buffer : buffers) buffer = arena.take(plane_max * f);
    const std::uint64_t d_copy = arena.take(qi_vae_copy_max(d, latent_h, latent_w) * f);
    const int top = q.top_channels;
    const std::uint64_t top_plane = static_cast<std::uint64_t>(latent_h) * latent_w;
    const std::uint64_t d_q = arena.take(top_plane * top * f);
    const std::uint64_t d_k = arena.take(top_plane * top * f);
    const std::uint64_t d_vt = arena.take(top_plane * top * f);
    const std::uint64_t d_attn = arena.take(top_plane * top * f);
    const std::uint64_t d_scores = arena.take(static_cast<std::uint64_t>(kVaeAttentionChunk) * top_plane * f);
    const std::uint64_t d_scale = arena.take(static_cast<std::uint64_t>(top) * f);
    const std::uint64_t d_hwc = arena.take(plane_max * 2);
    const std::uint64_t d_scales = arena.take(plane_max * f);
    const std::vector<float> query_scale(top, 1.0f / std::sqrt(static_cast<float>(top)));
    QiVaeState vae{d, launch, d_scales, d_hwc};

    std::uint64_t x = buffers[0], scratch = buffers[1], y = buffers[2];
    launch.upload(x, latents, static_cast<std::uint64_t>(q.latent_channels) * top_plane * f);
    int height = latent_h, width = latent_w;
    vae.conv_from_f32(q.post_quant, x, y, height, width);
    std::swap(x, y);
    vae.conv_from_f32(q.conv_in, x, y, height, width);
    std::swap(x, y);

    // Mid block: resnet, one-head attention over the plane, resnet.
    vae.resnet(q.mid_resnet[0], x, scratch, y, height, width);
    std::swap(x, y);
    {
        const int plane = height * width;
        vae.norm(q.mid_norm, x, scratch, 0, plane, false);
        {
            // [C][HW] -> [HW][C] for the 1x1 projections.
            int rows = top, columns = plane;
            std::uint64_t input = scratch, output = y;
            void* arguments[] = {&input, &output, &rows, &columns};
            launch.run("diff_transpose", static_cast<std::uint32_t>((columns + 31) / 32),
                       static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
        }
        launch.gemm(q.mid_q, y, d_q, plane, 0, 0);
        launch.gemm(q.mid_k, y, d_k, plane, 0, 0);
        launch.gemm(q.mid_v, y, d_attn, plane, 0, 0);
        {
            int rows = plane, columns = top;
            std::uint64_t input = d_attn, output = d_vt;
            void* arguments[] = {&input, &output, &rows, &columns};
            launch.run("diff_transpose", static_cast<std::uint32_t>((columns + 31) / 32),
                       static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
        }
        launch.upload(d_scale, query_scale.data(), static_cast<std::uint64_t>(top) * f);
        launch.scale_columns(d_q, d_scale, top, plane);
        for (int start = 0; start < plane; start += kVaeAttentionChunk) {
            const int count = std::min(kVaeAttentionChunk, plane - start);
            const std::uint64_t q_chunk = d_q + static_cast<std::uint64_t>(start) * top * f;
            DiffMatrix keys;   // k as a [HW][C] "weight": scores = q . k^T
            keys.device = d_k; keys.type = 0; keys.in = top; keys.out = plane;
            launch.gemm(keys, q_chunk, d_scores, count, 0, 0);
            {
                std::uint64_t scores = d_scores;
                int columns = plane, rows = count;
                void* arguments[] = {&scores, &columns, &rows};
                launch.run("diff_softmax_rows", static_cast<std::uint32_t>(rows), 1, 256, arguments);
            }
            DiffMatrix values;   // v^T as a [C][HW] "weight": out = p . v
            values.device = d_vt; values.type = 0; values.in = plane; values.out = top;
            launch.gemm(values, d_scores, d_attn + static_cast<std::uint64_t>(start) * top * f,
                        count, 0, 0);
        }
        launch.gemm(q.mid_proj, d_attn, y, plane, 0, 0);
        {
            int rows = plane, columns = top;
            std::uint64_t input = y, output = scratch;
            void* arguments[] = {&input, &output, &rows, &columns};
            launch.run("diff_transpose", static_cast<std::uint32_t>((columns + 31) / 32),
                       static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
        }
        launch.add(x, scratch, static_cast<std::uint64_t>(top) * plane);
    }
    vae.resnet(q.mid_resnet[1], x, scratch, y, height, width);
    std::swap(x, y);

    // Residual up blocks: resnets, upsample, then the duplicating shortcut of
    // the block's own input added over the top.
    for (const auto& block : q.up_blocks) {
        if (block.upsamples)
            if (flyweight_gpu_copy_device(d_copy, x,
                                          static_cast<std::uint64_t>(block.in) * height * width * f,
                                          d.stream) != 0)
                throw std::runtime_error("diffusion up-block copy failed");
        const int copy_height = height, copy_width = width;
        for (const auto& resnet : block.resnets) {
            vae.resnet(resnet, x, scratch, y, height, width);
            std::swap(x, y);
        }
        if (!block.upsamples) continue;
        {
            const bool tc = block.upsampler.tensor_core();
            int c = block.out, h = height, w = width;
            std::uint64_t input = x, output = tc ? d_hwc : y;
            void* arguments[] = {&input, &output, &c, &h, &w};
            launch.run(tc ? "diff_upsample_nearest_2x_hwc_bf16" : "diff_upsample_nearest_2x",
                       DiffLaunch::grid_for(static_cast<std::uint64_t>(block.out) * height * width * 4),
                       1, 256, arguments);
            height *= 2;
            width *= 2;
            vae.conv(block.upsampler, tc ? d_hwc : y, x, height, width);
        }
        {
            int out = block.out, factor = block.factor, repeats = block.repeats;
            int t_keep = block.t_keep, h = copy_height, w = copy_width;
            std::uint64_t input = d_copy, output = x;
            void* arguments[] = {&input, &output, &out, &factor, &repeats, &t_keep, &h, &w};
            launch.run("qi_dup_up_add",
                       DiffLaunch::grid_for(static_cast<std::uint64_t>(block.out) * height * width),
                       1, 256, arguments);
        }
    }
    vae.norm(q.norm_out, x, scratch, 0, height * width, true);
    vae.conv(q.conv_out, scratch, y, height, width);
    launch.download(pixels, y, static_cast<std::uint64_t>(q.out_channels) * height * width * f);
}

// The whole latent to interleaved 8-bit pixels: one tile when it fits,
// otherwise the tiled walk with blended overlaps. Latents are denormalized per
// channel first (`z * std + mean`), and the decoder's output is clamped to
// [-1, 1] and mapped by `x / 2 + 0.5`, as the pipeline's postprocess does.
void qi_decode_latents(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                       std::uint8_t* pixels) {
    const QwenImage21Tower& q = *d.qwenimage;
    const int channels = q.latent_channels, out_channels = q.out_channels;
    const int scale = q.spatial_scale;
    const int height = latent_h * scale, width = latent_w * scale;
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    const std::size_t latent_plane = static_cast<std::size_t>(latent_h) * latent_w;
    std::vector<float> normalized(static_cast<std::size_t>(channels) * latent_plane);
    for (int c = 0; c < channels; ++c)
        for (std::size_t i = 0; i < latent_plane; ++i)
            normalized[c * latent_plane + i] =
                latents[c * latent_plane + i] * q.latents_std[c] + q.latents_mean[c];

    std::vector<float> image(static_cast<std::size_t>(out_channels) * plane);
    if (latent_h <= kQiTileLatent && latent_w <= kQiTileLatent) {
        qi_decode_tile(d, normalized.data(), latent_h, latent_w, image.data());
    } else {
        std::vector<float> tile_latents, tile_pixels;
        const int row_limit = kQiTileLatent * scale - kQiBlendPixels;   // pixels kept per tile
        std::vector<std::vector<float>> row_tiles, previous_row;
        std::vector<int> row_tile_widths;
        int previous_height = 0, out_y = 0;
        for (int ty = 0; ty < latent_h; ty += kQiTileStride) {
            const int th = std::min(kQiTileLatent, latent_h - ty);
            row_tiles.clear();
            row_tile_widths.clear();
            for (int tx = 0; tx < latent_w; tx += kQiTileStride) {
                const int tw = std::min(kQiTileLatent, latent_w - tx);
                tile_latents.resize(static_cast<std::size_t>(channels) * th * tw);
                for (int c = 0; c < channels; ++c)
                    for (int y = 0; y < th; ++y)
                        std::memcpy(tile_latents.data() + (static_cast<std::size_t>(c) * th + y) * tw,
                                    normalized.data() +
                                        (static_cast<std::size_t>(c) * latent_h + ty + y) * latent_w + tx,
                                    static_cast<std::size_t>(tw) * sizeof(float));
                tile_pixels.assign(static_cast<std::size_t>(out_channels) * th * scale * tw * scale, 0.0f);
                qi_decode_tile(d, tile_latents.data(), th, tw, tile_pixels.data());
                row_tiles.push_back(tile_pixels);
                row_tile_widths.push_back(tw * scale);
            }
            const int tile_height = th * scale;
            for (std::size_t i = 0; i < row_tiles.size(); ++i) {
                auto& tile = row_tiles[i];
                const int tile_width = row_tile_widths[i];
                if (!previous_row.empty()) {
                    const auto& above = previous_row[i];
                    const int blend = std::min({kQiBlendPixels, previous_height, tile_height});
                    for (int c = 0; c < out_channels; ++c)
                        for (int y = 0; y < blend; ++y) {
                            const float weight = static_cast<float>(y) / blend;
                            for (int x = 0; x < tile_width; ++x) {
                                float& value = tile[(static_cast<std::size_t>(c) * tile_height + y) * tile_width + x];
                                const float over = above[(static_cast<std::size_t>(c) * previous_height +
                                                          previous_height - blend + y) * tile_width + x];
                                value = over * (1.0f - weight) + value * weight;
                            }
                        }
                }
                if (i > 0) {
                    const auto& left = row_tiles[i - 1];
                    const int left_width = row_tile_widths[i - 1];
                    const int blend = std::min({kQiBlendPixels, left_width, tile_width});
                    for (int c = 0; c < out_channels; ++c)
                        for (int y = 0; y < tile_height; ++y)
                            for (int x = 0; x < blend; ++x) {
                                const float weight = static_cast<float>(x) / blend;
                                float& value = tile[(static_cast<std::size_t>(c) * tile_height + y) * tile_width + x];
                                const float over = left[(static_cast<std::size_t>(c) * tile_height + y) * left_width +
                                                        left_width - blend + x];
                                value = over * (1.0f - weight) + value * weight;
                            }
                }
            }
            const int keep_rows = std::min(row_limit, height - out_y);
            int out_x = 0;
            for (std::size_t i = 0; i < row_tiles.size(); ++i) {
                const int tile_width = row_tile_widths[i];
                const int keep_columns = std::min(row_limit, width - out_x);
                for (int c = 0; c < out_channels; ++c)
                    for (int y = 0; y < keep_rows; ++y)
                        std::memcpy(image.data() + c * plane + static_cast<std::size_t>(out_y + y) * width + out_x,
                                    row_tiles[i].data() + (static_cast<std::size_t>(c) * tile_height + y) * tile_width,
                                    static_cast<std::size_t>(keep_columns) * sizeof(float));
                out_x += keep_columns;
                if (out_x >= width) break;
            }
            out_y += keep_rows;
            previous_row = row_tiles;
            previous_height = tile_height;
            if (out_y >= height) break;
        }
    }
    for (std::size_t pixel = 0; pixel < plane; ++pixel)
        for (int c = 0; c < out_channels; ++c) {
            const float clamped = std::min(1.0f, std::max(-1.0f, image[c * plane + pixel]));
            const float value = (clamped * 0.5f + 0.5f) * 255.0f;
            pixels[pixel * out_channels + c] =
                static_cast<std::uint8_t>(std::min(255.0f, std::max(0.0f, std::round(value))));
        }
}

// ---- Scheduler -------------------------------------------------------------

// FlowMatchEulerDiscreteScheduler as the pipeline drives it: the schedule runs
// from 1 down to 1/steps (not to 0), is shifted by a factor the image's token
// count decides, and is then stretched so it terminates at shift_terminal. A
// positive `shift` overrides the resolution-dependent factor with a constant
// one, which is the same formula at exp(mu) = shift.
std::vector<float> qi_sigmas(int steps, float shift, std::uint64_t image_tokens) {
    const double factor = shift > 0.0f
        ? static_cast<double>(shift)
        : std::exp(static_cast<double>(
              kQiBaseShift + (kQiMaxShift - kQiBaseShift) *
                                 (static_cast<float>(image_tokens) - kQiBaseSeqLen) /
                                 (kQiMaxSeqLen - kQiBaseSeqLen)));
    std::vector<double> sigmas(steps);
    for (int i = 0; i < steps; ++i) {
        const double linear = steps > 1
            ? 1.0 - static_cast<double>(i) * (1.0 - 1.0 / steps) / (steps - 1)
            : 1.0;
        sigmas[i] = factor / (factor + (1.0 / linear - 1.0));
    }
    const double stretch = (1.0 - sigmas[steps - 1]) / (1.0 - kQiShiftTerminal);
    std::vector<float> out;
    out.reserve(steps + 1);
    for (int i = 0; i < steps; ++i)
        out.push_back(static_cast<float>(1.0 - (1.0 - sigmas[i]) / stretch));
    out.push_back(0.0f);
    return out;
}

void qi_generate(FlyweightV2Diffusion& d, const std::uint32_t* tokens, std::uint64_t count,
                 std::uint32_t caption_drop, std::uint32_t width, std::uint32_t height,
                 std::uint32_t steps, float shift, std::uint64_t seed, const float* initial_latents,
                 FlyweightV2DiffusionProgress progress, void* user_data, std::uint8_t* pixels) {
    const QwenImage21Tower& q = *d.qwenimage;
    const int scale = q.spatial_scale;
    const int latent_h = static_cast<int>(height / scale), latent_w = static_cast<int>(width / scale);
    const std::size_t latent_count = static_cast<std::size_t>(q.channels) * latent_h * latent_w;
    std::vector<float> caption(count * static_cast<std::uint64_t>(d.encoder_hidden));
    diff_encode_text(d, tokens, count, caption.data());
    // The system-role prefix is encoded -- the rows after it attend to it --
    // and then dropped before the transformer sees the conditioning.
    const float* conditioning = caption.data() +
        static_cast<std::uint64_t>(caption_drop) * d.encoder_hidden;
    const std::uint64_t caption_tokens = count - caption_drop;
    if (progress && progress(user_data, 0, steps) != 0) throw std::runtime_error("cancelled");
    std::vector<float> latents(latent_count), velocity(latent_count);
    if (initial_latents) std::memcpy(latents.data(), initial_latents, latent_count * sizeof(float));
    else diff_gaussian_noise(seed, latents);
    const auto sigmas = qi_sigmas(static_cast<int>(steps), shift,
                                  static_cast<std::uint64_t>(latent_h) * latent_w);
    for (std::uint32_t step = 0; step < steps; ++step) {
        const float sigma = sigmas[step], next = sigmas[step + 1];
        if (sigma == next) continue;
        qi_transformer_forward(d, latents.data(), latent_h, latent_w, conditioning, caption_tokens,
                               sigma, velocity.data());
        // Euler on the model's own output, which is the velocity as predicted.
        for (std::size_t i = 0; i < latent_count; ++i) latents[i] += (next - sigma) * velocity[i];
        if (progress && progress(user_data, step + 1, steps) != 0) throw std::runtime_error("cancelled");
    }
    qi_decode_latents(d, latents.data(), latent_h, latent_w, pixels);
}

}  // namespace
