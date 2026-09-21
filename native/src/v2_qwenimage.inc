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

// One run of the joint sequence: text rows, or one image block's latent rows.
// Text attends causally over everything before it and itself; an image block
// attends over everything before it plus itself, bidirectionally.
struct QiSegment {
    std::uint64_t start = 0, end = 0;
    bool image = false;
    int latent_h = 0, latent_w = 0;
};

// Sequence layout of one forward: the prefix -- the prompt's rows with each
// condition image's latent rows substituted where its tokens sat -- then the
// target image's rows. Prefix rows modulate from t = 0, target rows from the
// sampled t; the target attends to everything.
struct QiLayout {
    std::vector<QiSegment> prefix;
    std::uint64_t prefix_rows = 0, image = 0, rows = 0;
    int latent_h = 0, latent_w = 0;
};

// A condition image's latents, already normalized, as the transformer takes
// them: [channels][latent_h][latent_w].
struct QiCondition {
    std::vector<float> latents;
    int latent_h = 0, latent_w = 0;
};

// One image's contribution to the text encoder: the merged vision tokens
// projected into the decoder's width (plane 0), then one plane per deepstack
// layer, each [tokens][hidden]; and where its <|image_pad|> run starts.
struct QiEncodedImage {
    std::vector<std::vector<float>> planes;
    std::uint64_t token_offset = 0;
    std::uint32_t tokens = 0, grid_h = 0, grid_w = 0;   // merged grid
};

struct QiVisionBlock {
    std::uint64_t ln1_weight = 0, ln1_bias = 0, ln2_weight = 0, ln2_bias = 0;
    DiffMatrix qkv, out, up, down;
    // The deepstack merger reading this block's output, if any.
    bool deepstack = false;
    std::uint64_t deepstack_norm_weight = 0, deepstack_norm_bias = 0;
    DiffMatrix deepstack_fc1, deepstack_fc2;
};

struct QiDownBlock {
    std::vector<QiResnet> resnets;
    VaeConv downsampler;   // 3x3 at stride 2, when the level downsamples
    bool downsamples = false;
    // AvgDown3D: `factor` slots per input channel averaged in groups of `group`.
    int factor = 1, group = 1, factor_t = 1, factor_s = 1;
    int in = 0, out = 0;
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

    // Autoencoder encoder, for condition images. Its mid block has the same
    // shape as the decoder's at its own width.
    VaeConv enc_conv_in, enc_conv_out, quant_conv;
    QiNorm enc_norm_out, enc_mid_norm;
    QiResnet enc_mid_resnet[2];
    DiffMatrix enc_mid_q, enc_mid_k, enc_mid_v, enc_mid_proj;
    std::vector<QiDownBlock> down_blocks;
    int enc_top_channels = 0;
    bool has_encoder = false;

    // The text encoder's vision tower (Qwen3-VL), streamed like its blocks.
    bool has_vision = false;
    DiffMatrix patch;                 // [3 * 2 * p * p] -> E, both temporal frames
    std::uint64_t patch_bias = 0;
    std::vector<float> position_table;   // host [side*side][E], resampled per image
    std::vector<QiVisionBlock> vision_blocks;
    std::uint64_t post_ln_weight = 0, post_ln_bias = 0;
    DiffMatrix merger_fc1, merger_fc2;
    std::vector<int> deepstack_layers;   // ViT blocks whose output also feeds the decoder
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

    // The encoder half, on its own (narrower) base: dims = [base * m for m
    // in [1] + mult], down block i maps dims[i] -> dims[i + 1] and every
    // level but the last downsamples.
    if (!diff_has_tensor(model, "encoder.conv_in.weight")) return;
    q.has_encoder = true;
    const int base = static_cast<int>(config.vae_encoder_base_dim);
    const int decoder_base = static_cast<int>(config.vae_decoder_base_dim);
    q.enc_conv_in = qi_upload_conv(d, model, "encoder.conv_in", q.out_channels, base, 3);
    int previous = base;
    for (int level = 0; level < levels; ++level) {
        const std::string prefix = "encoder.down_blocks." + std::to_string(level);
        QiDownBlock block;
        block.in = previous;
        // dim_mult[level], recovered from the decoder's channel list.
        block.out = base * static_cast<int>(config.vae_level_channels[level]) / decoder_base;
        block.downsamples = level != levels - 1;
        int in = block.in;
        for (int index = 0; index < static_cast<int>(config.vae_layers_per_block); ++index) {
            block.resnets.push_back(qi_upload_resnet(
                d, model, prefix + ".resnets." + std::to_string(index), in, block.out));
            in = block.out;
        }
        if (block.downsamples) {
            block.downsampler = qi_upload_conv(d, model, prefix + ".downsampler.resample.1",
                                               block.out, block.out, 3);
            // temperal_downsample[i] decides the frame factor of a level that
            // downsamples; the last level has neither.
            block.factor_t = config.vae_temporal_upsample[levels - 2 - level] ? 2 : 1;
            block.factor_s = 2;
        }
        block.factor = block.factor_t * block.factor_s * block.factor_s;
        const std::uint64_t group = static_cast<std::uint64_t>(block.in) * block.factor / block.out;
        if (group * block.out != static_cast<std::uint64_t>(block.in) * block.factor)
            throw std::runtime_error("AutoencoderKLQwenImage21 down block " + std::to_string(level) +
                                     " has a shortcut that does not divide evenly");
        block.group = static_cast<int>(group);
        previous = block.out;
        q.down_blocks.push_back(std::move(block));
    }
    q.enc_top_channels = previous;
    q.enc_mid_resnet[0] = qi_upload_resnet(d, model, "encoder.mid_block.resnets.0", previous, previous);
    q.enc_mid_resnet[1] = qi_upload_resnet(d, model, "encoder.mid_block.resnets.1", previous, previous);
    {
        const std::string prefix = "encoder.mid_block.attentions.0";
        q.enc_mid_norm = qi_upload_norm(d, model, prefix + ".norm", previous);
        const DiffMatrix fused = qi_upload_conv_matrix(d, model, prefix + ".to_qkv", previous, 3 * previous);
        const std::uint64_t element = fused.type == 30 ? 2 : 4;
        DiffMatrix* slices[3] = {&q.enc_mid_q, &q.enc_mid_k, &q.enc_mid_v};
        for (int piece = 0; piece < 3; ++piece) {
            *slices[piece] = fused;
            slices[piece]->in = previous;
            slices[piece]->out = previous;
            slices[piece]->device =
                fused.device + static_cast<std::uint64_t>(piece) * previous * previous * element;
            slices[piece]->bias = fused.bias + static_cast<std::uint64_t>(piece) * previous * sizeof(float);
        }
        q.enc_mid_proj = qi_upload_conv_matrix(d, model, prefix + ".proj", previous, previous);
    }
    q.enc_norm_out = qi_upload_norm(d, model, "encoder.norm_out", previous);
    q.enc_conv_out = qi_upload_conv(d, model, "encoder.conv_out", previous, 2 * q.latent_channels, 3);
    q.quant_conv = qi_upload_conv(d, model, "quant_conv", 2 * q.latent_channels, 2 * q.latent_channels, 1);
}

// The text encoder's vision tower, from the same checkpoint the decoder came
// from (`model.visual.*`, under the mmproj names the loader gives them). The
// blocks stream through the staging slots like the decoder's do.
DiffLayerRef qi_vision_matrices(QiVisionBlock& block) {
    DiffLayerRef ref;
    ref.add(block.qkv);
    ref.add(block.out);
    ref.add(block.up);
    ref.add(block.down);
    return ref;
}

void qi_upload_vision(FlyweightV2Diffusion& d) {
    QwenImage21Tower& q = *d.qwenimage;
    const auto& model = *d.encoder;
    const auto& config = model.vision;
    if (!config.present || !diff_has_tensor(model, "v.patch_embd.weight")) return;
    flyweight::v2::validate_vision_config(config);
    if (config.projection_dim != static_cast<std::uint32_t>(d.encoder_hidden))
        throw std::runtime_error("the vision tower projects to " + std::to_string(config.projection_dim) +
                                 " but the text encoder is " + std::to_string(d.encoder_hidden) + " wide");
    const int width = static_cast<int>(config.embedding_length);
    const int ffn = static_cast<int>(config.feed_forward_length);
    const int patch_inputs = 3 * 2 * static_cast<int>(config.patch_size * config.patch_size);
    const int merged = width * static_cast<int>(config.spatial_merge_size * config.spatial_merge_size);
    const int projection = static_cast<int>(config.projection_dim);
    q.has_vision = true;
    q.patch = diff_upload_matrix(d, model, "v.patch_embd.weight", patch_inputs, width);
    q.patch_bias = diff_upload_f32(d, model, "v.patch_embd.bias", width);
    {
        const Tensor& table = diff_tensor(model, "v.position_embd.weight");
        const std::uint64_t side = config.position_grid_side();
        const std::uint64_t elements = side * side * width;
        if (diff_elements(table) != elements)
            throw std::runtime_error("vision position table does not match num_position_embeddings");
        q.position_table.resize(elements);
        const auto* data = tensor_data(model, table);
        for (std::uint64_t i = 0; i < elements; ++i) q.position_table[i] = tensor_value(data, table.type, i);
    }
    for (const auto layer : config.deepstack_layers) q.deepstack_layers.push_back(static_cast<int>(layer));
    q.vision_blocks.resize(config.block_count);
    for (std::uint32_t index = 0; index < config.block_count; ++index) {
        auto& block = q.vision_blocks[index];
        const std::string prefix = "v.blk." + std::to_string(index) + ".";
        block.ln1_weight = diff_upload_f32(d, model, prefix + "ln1.weight", width);
        block.ln1_bias = diff_upload_f32(d, model, prefix + "ln1.bias", width);
        block.ln2_weight = diff_upload_f32(d, model, prefix + "ln2.weight", width);
        block.ln2_bias = diff_upload_f32(d, model, prefix + "ln2.bias", width);
        {
            DiffGroup group(d);
            block.qkv = diff_upload_matrix(d, model, prefix + "attn_qkv.weight", width, 3 * width,
                                           prefix + "attn_qkv.bias");
            block.out = diff_upload_matrix(d, model, prefix + "attn_out.weight", width, width,
                                           prefix + "attn_out.bias");
            block.up = diff_upload_matrix(d, model, prefix + "ffn_up.weight", width, ffn, prefix + "ffn_up.bias");
            block.down = diff_upload_matrix(d, model, prefix + "ffn_down.weight", ffn, width, prefix + "ffn_down.bias");
        }
        const auto slot = std::find(q.deepstack_layers.begin(), q.deepstack_layers.end(), static_cast<int>(index));
        if (slot != q.deepstack_layers.end()) {
            const std::string stack = "v.deepstack_list." +
                std::to_string(slot - q.deepstack_layers.begin()) + ".";
            block.deepstack = true;
            block.deepstack_norm_weight = diff_upload_f32(d, model, stack + "norm.weight", merged);
            block.deepstack_norm_bias = diff_upload_f32(d, model, stack + "norm.bias", merged);
            block.deepstack_fc1 = diff_upload_matrix(d, model, stack + "fc1.weight", merged, merged, stack + "fc1.bias");
            block.deepstack_fc2 = diff_upload_matrix(d, model, stack + "fc2.weight", merged, projection, stack + "fc2.bias");
        }
    }
    q.post_ln_weight = diff_upload_f32(d, model, "v.post_ln.weight", width);
    q.post_ln_bias = diff_upload_f32(d, model, "v.post_ln.bias", width);
    q.merger_fc1 = diff_upload_matrix(d, model, "mm.0.weight", merged, merged, "mm.0.bias");
    q.merger_fc2 = diff_upload_matrix(d, model, "mm.2.weight", merged, projection, "mm.2.bias");
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

struct QiKvCache {
    std::uint64_t device = 0;
    std::uint64_t layer_bytes = 0;
    int layers = 0;
    int kv_heads = 0;
    std::uint64_t prefix = 0;

    QiKvCache() = default;

    QiKvCache(int num_layers, int num_heads, std::uint64_t prefix_rows)
        : layers(num_layers), kv_heads(num_heads), prefix(prefix_rows) {
        if (!layers || !kv_heads || !prefix) return;
        layer_bytes = 2ULL * kv_heads * prefix * 128 * sizeof(std::uint16_t);
        const std::uint64_t total_bytes = static_cast<std::uint64_t>(layers) * layer_bytes;
        std::uint64_t ptr = 0;
        if (flyweight_gpu_alloc(total_bytes, &ptr) == 0 && ptr) {
            device = ptr;
        }
    }

    ~QiKvCache() {
        if (device) flyweight_gpu_free(device);
    }

    QiKvCache(const QiKvCache&) = delete;
    QiKvCache& operator=(const QiKvCache&) = delete;

    QiKvCache(QiKvCache&& o) noexcept
        : device(o.device), layer_bytes(o.layer_bytes), layers(o.layers),
          kv_heads(o.kv_heads), prefix(o.prefix) {
        o.device = 0;
    }

    QiKvCache& operator=(QiKvCache&& o) noexcept {
        if (this != &o) {
            if (device) flyweight_gpu_free(device);
            device = o.device;
            layer_bytes = o.layer_bytes;
            layers = o.layers;
            kv_heads = o.kv_heads;
            prefix = o.prefix;
            o.device = 0;
        }
        return *this;
    }

    bool valid() const { return device != 0; }

    std::uint64_t k(int layer) const {
        return device + static_cast<std::uint64_t>(layer) * layer_bytes;
    }

    std::uint64_t v(int layer) const {
        return device + static_cast<std::uint64_t>(layer) * layer_bytes +
               static_cast<std::uint64_t>(kv_heads) * prefix * 128 * sizeof(std::uint16_t);
    }
};

// One block over the joint sequence. Every GEMM and norm runs over all rows;
// what splits is the modulation (text rows read the t = 0 row of `modulation`,
// image rows the sampled one) and the attention, whose block-causal mask is
// exactly "text rows causal over text, image rows full over everything".
void qi_run_block(FlyweightV2Diffusion& d, const DiffLaunch& launch, QiBlock& block,
                  const QiBuffers& b, const QiLayout& layout,
                  QiKvCache* cache = nullptr, int layer = 0) {
    const QwenImage21Tower& q = *d.qwenimage;
    const std::uint64_t f = sizeof(float);
    const int dim = q.dim;
    const std::uint64_t rows = layout.rows, text = layout.prefix_rows, image = layout.image;
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
        // Block-causal attention as one call per prefix segment -- a text
        // run causal over the keys up to its end, an image block full over
        // them -- and one for the target over everything. The first call
        // packs the whole sequence into the bf16 planes; the rest reuse them.
        bool packed = false;
        for (const auto& segment : layout.prefix) {
            launch.attention(b.qkv, k, v, b.attn, q.heads, q.heads, q.head_dim, rows,
                             3 * dim, 3 * dim, !segment.image, b.q16, b.k16, b.v16,
                             segment.start, segment.end - segment.start, segment.end, packed);
            packed = true;
        }
        launch.attention(b.qkv, k, v, b.attn, q.heads, q.heads, q.head_dim, rows,
                         3 * dim, 3 * dim, false, b.q16, b.k16, b.v16, text, image, rows, packed);
        if (cache && cache->valid() && text > 0 && d.flash_attention && !d.exact && q.head_dim == 128) {
            int kv_heads = q.heads, prefix_count = static_cast<int>(text), total_rows = static_cast<int>(rows);
            std::uint64_t k_in = b.k16, v_in = b.v16;
            std::uint64_t k_cache = cache->k(layer), v_cache = cache->v(layer);
            void* arguments[] = {&k_in, &v_in, &k_cache, &v_cache, &kv_heads, &prefix_count, &total_rows};
            launch.run("diff_kv_cache_extract",
                       DiffLaunch::grid_for(static_cast<std::uint64_t>(kv_heads) * text * 128),
                       1, 256, arguments);
        }
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

// Decode block specialized for steps > 0: prefix KV is restored from cache,
// and only target image rows are computed through layer norms, QKV GEMM,
// RoPE, Flash Attention, out GEMM, and MLP.
void qi_run_block_decode(FlyweightV2Diffusion& d, const DiffLaunch& launch, QiBlock& block,
                         const QiBuffers& b, const QiLayout& layout,
                         const QiKvCache& cache, int layer) {
    const QwenImage21Tower& q = *d.qwenimage;
    const std::uint64_t f = sizeof(float);
    const int dim = q.dim;
    const std::uint64_t rows = layout.rows, prefix = layout.prefix_rows, image = layout.image;
    const std::uint64_t sampled = b.modulation;
    const auto slot = [&](std::uint64_t row, int index) {
        return row + static_cast<std::uint64_t>(index) * dim * f;
    };
    const std::uint64_t image_hidden = b.hidden + prefix * dim * f;

    launch.layer_norm(image_hidden, b.normed, dim, image, q.epsilon);
    launch.scale_columns(b.normed, slot(sampled, 0), dim, image);
    launch.gemm(block.qkv, b.normed, b.qkv, image, b.q8, b.q8_scales);
    launch.head_rms_norm(b.qkv, block.q_norm, q.heads, q.head_dim, image, 3 * dim, 0, q.epsilon);
    launch.head_rms_norm(b.qkv, block.k_norm, q.heads, q.head_dim, image, 3 * dim, dim, q.epsilon);
    {
        int heads = q.heads, head_dim = q.head_dim, count = static_cast<int>(image);
        int axis0 = q.axis_dims[0], axis1 = q.axis_dims[1];
        std::uint64_t buffer = b.qkv;
        std::uint64_t positions = b.positions + prefix * 3 * sizeof(std::int32_t);
        std::uint64_t table = q.rope_frequencies;
        void* arguments[] = {&buffer, &positions, &table, &heads, &head_dim, &count, &axis0, &axis1};
        launch.run("diff_rope_axes_rows", count, 2, 256, arguments);
    }
    {
        int kv_heads = q.heads, prefix_count = static_cast<int>(prefix), total_rows = static_cast<int>(rows);
        std::uint64_t k_cache = cache.k(layer), v_cache = cache.v(layer);
        std::uint64_t k_out = b.k16, v_out = b.v16;
        void* arguments[] = {&k_cache, &v_cache, &k_out, &v_out, &kv_heads, &prefix_count, &total_rows};
        launch.run("diff_kv_cache_restore",
                   DiffLaunch::grid_for(static_cast<std::uint64_t>(kv_heads) * prefix * 128),
                   1, 256, arguments);
    }
    {
        int heads = q.heads, kv_heads = q.heads;
        int total_rows = static_cast<int>(rows), prefix_count = static_cast<int>(prefix), img_count = static_cast<int>(image);
        int q_stride = 3 * dim, kv_stride = 3 * dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(q.head_dim));
        float q_scale = scale * 1.4426950408889634f;
        std::uint64_t q_ptr = b.qkv;
        std::uint64_t k_ptr = b.qkv + static_cast<std::uint64_t>(dim) * f;
        std::uint64_t v_ptr = b.qkv + static_cast<std::uint64_t>(2 * dim) * f;
        std::uint64_t q16 = b.q16, k16 = b.k16, v16 = b.v16;
        void* arguments[] = {&q_ptr, &k_ptr, &v_ptr, &q16, &k16, &v16,
                             &heads, &kv_heads, &total_rows, &prefix_count, &img_count,
                             &q_stride, &kv_stride, &q_scale};
        launch.run("diff_pack_target_attention_bf16",
                   DiffLaunch::grid_for(static_cast<std::uint64_t>(image) * (heads + 2 * kv_heads) * 128),
                   1, 256, arguments);
    }
    {
        int heads = q.heads, kv_heads = q.heads;
        int plane = static_cast<int>(rows);
        int first = static_cast<int>(prefix);
        int queries = static_cast<int>(image);
        int keys = static_cast<int>(rows);
        int causal_flag = 0;
        std::uint64_t q16 = b.q16, k16 = b.k16, v16 = b.v16, out = b.attn;
        void* arguments[] = {&q16, &k16, &v16, &out, &heads, &kv_heads, &plane, &first,
                             &queries, &keys, &causal_flag};
        launch.run("diff_flash_attention_bf16", static_cast<std::uint32_t>((queries + 63) / 64),
                   static_cast<std::uint32_t>(heads), 128, arguments);
    }
    launch.gemm(block.out, b.attn + prefix * dim * f, b.normed, image, b.q8, b.q8_scales);
    launch.gated_add(image_hidden, b.normed, slot(sampled, 1), dim, image);

    launch.layer_norm(image_hidden, b.normed, dim, image, q.epsilon);
    launch.scale_columns(b.normed, slot(sampled, 2), dim, image);
    launch.gemm(block.gate, b.normed, b.gate, image, b.q8, b.q8_scales);
    launch.gemm(block.up, b.normed, b.up, image, b.q8, b.q8_scales);
    launch.silu_mul(b.gate, b.up, b.gate, image * q.intermediate);
    launch.gemm(block.down, b.gate, b.normed, image, b.q8, b.q8_scales);
    launch.gated_add(image_hidden, b.normed, slot(sampled, 3), dim, image);
}

// Positions over the joint sequence. Text tokens advance one shared position
// on all three axes; every image block freezes the frame axis at the position
// the preceding text reached, lays its tokens out on height and width ramps
// centred on zero (so a block's spatial positions do not depend on where it
// sits), and advances the position by its longer side.
void qi_positions(const QiLayout& layout, std::vector<std::int32_t>& positions) {
    positions.assign(layout.rows * 3, 0);
    std::int32_t position = 0;
    const auto image_block = [&](std::uint64_t start, int height, int width) {
        const int first_h = -(height - height / 2), first_w = -(width - width / 2);
        for (int h = 0; h < height; ++h)
            for (int w = 0; w < width; ++w) {
                const std::uint64_t row = start + static_cast<std::uint64_t>(h) * width + w;
                positions[row * 3] = position;
                positions[row * 3 + 1] = static_cast<std::int32_t>(first_h + h);
                positions[row * 3 + 2] = static_cast<std::int32_t>(first_w + w);
            }
        position += std::max(height, width);
    };
    for (const auto& segment : layout.prefix) {
        if (segment.image) {
            image_block(segment.start, segment.latent_h, segment.latent_w);
            continue;
        }
        for (std::uint64_t row = segment.start; row < segment.end; ++row, ++position)
            positions[row * 3] = positions[row * 3 + 1] = positions[row * 3 + 2] = position;
    }
    image_block(layout.prefix_rows, layout.latent_h, layout.latent_w);
}

// The joint sequence for a prompt of `caption_tokens` conditioning rows of
// which `pad_mask` marks the vision tokens, and `conditions` in the order
// their token runs appear: each run of vision tokens becomes its image's
// latent rows (four per token) and everything else stays a text row. The
// expanded conditioning is written to `caption_rows` (zero at image rows,
// which the transformer overwrites).
QiLayout qi_layout(const float* caption, std::uint64_t caption_tokens, int caption_dim,
                   const std::uint8_t* pad_mask, const std::vector<QiCondition>& conditions,
                   int latent_h, int latent_w, std::vector<float>& caption_rows) {
    QiLayout layout;
    layout.latent_h = latent_h;
    layout.latent_w = latent_w;
    layout.image = static_cast<std::uint64_t>(latent_h) * latent_w;
    caption_rows.clear();
    std::size_t condition = 0;
    std::uint64_t row = 0;
    for (std::uint64_t token = 0; token < caption_tokens;) {
        const bool pad = pad_mask && pad_mask[token];
        if (!pad) {
            std::uint64_t end = token;
            while (end < caption_tokens && !(pad_mask && pad_mask[end])) ++end;
            const std::uint64_t count = end - token;
            caption_rows.insert(caption_rows.end(), caption + token * caption_dim,
                                caption + end * caption_dim);
            layout.prefix.push_back({row, row + count, false, 0, 0});
            row += count;
            token = end;
            continue;
        }
        std::uint64_t end = token;
        while (end < caption_tokens && pad_mask[end]) ++end;
        if (condition >= conditions.size())
            throw std::runtime_error("the prompt carries more image token runs than condition images");
        const auto& image = conditions[condition++];
        const std::uint64_t latent_rows = static_cast<std::uint64_t>(image.latent_h) * image.latent_w;
        if (latent_rows != (end - token) * 4)
            throw std::runtime_error("a condition image's token run does not match its latent size (" +
                                     std::to_string(end - token) + " tokens for " +
                                     std::to_string(latent_rows) + " latent rows)");
        caption_rows.resize(caption_rows.size() + latent_rows * caption_dim, 0.0f);
        layout.prefix.push_back({row, row + latent_rows, true, image.latent_h, image.latent_w});
        row += latent_rows;
        token = end;
    }
    if (condition != conditions.size())
        throw std::runtime_error("the prompt carries fewer image token runs than condition images");
    layout.prefix_rows = row;
    layout.rows = row + layout.image;
    return layout;
}

// One forward: the model's output for `latents` at time `t` (the sampler's
// sigma) under `caption` [caption_tokens][caption_dim] with any condition
// images, written to `velocity` as [channels][latent_h][latent_w].
void qi_transformer_forward_joint(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                                  const float* caption, std::uint64_t caption_tokens,
                                  const std::uint8_t* pad_mask, const std::vector<QiCondition>& conditions,
                                  float t, float* velocity,
                                  QiKvCache* cache = nullptr, bool is_decode = false) {
    QwenImage21Tower& q = *d.qwenimage;
    if (!caption_tokens) throw std::runtime_error("empty caption");
    std::vector<float> caption_rows;
    const QiLayout layout = qi_layout(caption, caption_tokens, q.caption_dim, pad_mask, conditions,
                                      latent_h, latent_w, caption_rows);
    const std::uint64_t prefix = layout.prefix_rows;
    const std::uint64_t f = sizeof(float);
    const int dim = q.dim, channels = q.channels;
    diff_ensure_workspace(d, qi_transformer_workspace(d, layout.rows, prefix), "image");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const QiBuffers b = qi_transformer_buffers(d, arena, layout.rows, prefix);
    // Host staging for the uploads below; alive until the final download syncs.
    std::vector<float> frequencies(2 * kQiTimeEmbedWidth);
    std::vector<std::int32_t> positions;
    std::vector<float> rows(layout.rows * channels);

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

    if (!is_decode) {
        // Prefix rows: zero-centred RMSNorm, linear, GELU, linear over every row
        // (an image block's rows are zero here and are overwritten just below,
        // which costs less than gathering the text rows apart).
        launch.upload(b.caption, caption_rows.data(), prefix * q.caption_dim * f);
        launch.rms_norm(b.caption, q.txt_norm, b.caption_normed, q.caption_dim, prefix, q.epsilon);
        launch.gemm(q.txt_in, b.caption_normed, b.hidden, prefix, b.q8, b.q8_scales);
        {
            int elements = static_cast<int>(prefix * dim);
            std::uint64_t hidden = b.hidden;
            void* arguments[] = {&hidden, &elements};
            launch.run("vision_gelu_rows", DiffLaunch::grid_for(prefix * dim), 1, 256, arguments);
        }
        launch.gemm(q.txt_out, b.hidden, b.normed, prefix, b.q8, b.q8_scales);
        if (flyweight_gpu_copy_device(b.hidden, b.normed, prefix * dim * f, d.stream) != 0)
            throw std::runtime_error("diffusion caption copy failed");

        // Image rows, condition blocks then the target: each latent unpatched,
        // one row of `channels` per pixel, at its block's rows.
        {
            std::size_t condition = 0;
            const auto unpatch = [&](const float* source, std::uint64_t start, std::uint64_t count) {
                for (std::uint64_t pixel = 0; pixel < count; ++pixel)
                    for (int c = 0; c < channels; ++c)
                        rows[(start + pixel) * channels + c] = source[static_cast<std::uint64_t>(c) * count + pixel];
            };
            for (const auto& segment : layout.prefix)
                if (segment.image)
                    unpatch(conditions[condition++].latents.data(), segment.start, segment.end - segment.start);
            unpatch(latents, prefix, layout.image);
            launch.upload(b.latent_in, rows.data(), layout.rows * channels * f);
            for (const auto& segment : layout.prefix)
                if (segment.image)
                    launch.gemm(q.img_in, b.latent_in + segment.start * channels * f,
                                b.hidden + segment.start * dim * f, segment.end - segment.start,
                                b.q8, b.q8_scales);
            launch.gemm(q.img_in, b.latent_in + prefix * channels * f, b.hidden + prefix * dim * f,
                        layout.image, b.q8, b.q8_scales);
        }
    } else {
        for (std::uint64_t pixel = 0; pixel < layout.image; ++pixel)
            for (int c = 0; c < channels; ++c)
                rows[pixel * channels + c] = latents[static_cast<std::uint64_t>(c) * layout.image + pixel];
        launch.upload(b.latent_in, rows.data(), layout.image * channels * f);
        launch.gemm(q.img_in, b.latent_in, b.hidden + prefix * dim * f, layout.image, b.q8, b.q8_scales);
    }

    DiffStreamer streamer{d, {}, 0};
    for (auto& block : q.blocks) streamer.refs.push_back(qi_block_matrices(block));
    for (std::size_t index = 0; index < q.blocks.size(); ++index) {
        streamer.begin(index);
        if (!is_decode) {
            qi_run_block(d, launch, q.blocks[index], b, layout, cache, static_cast<int>(index));
        } else {
            qi_run_block_decode(d, launch, q.blocks[index], b, layout, *cache, static_cast<int>(index));
        }
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
        const std::uint64_t image_hidden = b.hidden + prefix * dim * f;
        launch.layer_norm(image_hidden, b.normed, dim, layout.image, q.epsilon);
        launch.scale_columns(b.normed, b.final_scale, dim, layout.image);
        launch.gemm(q.proj_out, b.normed, b.latent_out, layout.image, b.q8, b.q8_scales);
    }
    launch.download(rows.data(), b.latent_out, layout.image * channels * f);
    for (std::uint64_t pixel = 0; pixel < layout.image; ++pixel)
        for (int c = 0; c < channels; ++c)
            velocity[static_cast<std::uint64_t>(c) * layout.image + pixel] = rows[pixel * channels + c];
}

// Text-only conditioning: the shape the C API's transformer_step exposes.
// (The .inc files sit inside an extern "C" region, so no overloading.)
void qi_transformer_forward(FlyweightV2Diffusion& d, const float* latents, int latent_h, int latent_w,
                            const float* caption, std::uint64_t caption_tokens, float t,
                            float* velocity) {
    qi_transformer_forward_joint(d, latents, latent_h, latent_w, caption, caption_tokens, nullptr, {},
                                 t, velocity);
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

std::vector<float> qi_sigmas(int steps, float shift, std::uint64_t image_tokens);

// ---- Autoencoder encoder ---------------------------------------------------

// One-head attention over the plane, as the mid blocks of both halves have
// it: RMS norm, fused 1x1 q/k/v, chunked score GEMM + row softmax + GEMM,
// 1x1 projection, residual into `x`. `scratch` and `y` are clobbered.
void qi_mid_attention(FlyweightV2Diffusion& d, const DiffLaunch& launch, const QiVaeState& vae,
                      const QiNorm& norm, const DiffMatrix& mq, const DiffMatrix& mk,
                      const DiffMatrix& mv, const DiffMatrix& mproj,
                      std::uint64_t x, std::uint64_t scratch, std::uint64_t y, int height, int width,
                      std::uint64_t d_q, std::uint64_t d_k, std::uint64_t d_vt, std::uint64_t d_attn,
                      std::uint64_t d_scores, std::uint64_t d_scale) {
    const std::uint64_t f = sizeof(float);
    const int top = norm.channels, plane = height * width;
    vae.norm(norm, x, scratch, 0, plane, false);
    const auto transpose = [&](std::uint64_t input, std::uint64_t output, int rows, int columns) {
        void* arguments[] = {&input, &output, &rows, &columns};
        launch.run("diff_transpose", static_cast<std::uint32_t>((columns + 31) / 32),
                   static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
    };
    transpose(scratch, y, top, plane);   // [C][HW] -> [HW][C]
    launch.gemm(mq, y, d_q, plane, 0, 0);
    launch.gemm(mk, y, d_k, plane, 0, 0);
    launch.gemm(mv, y, d_attn, plane, 0, 0);
    transpose(d_attn, d_vt, plane, top);
    // d_scale holds 1/sqrt(C), uploaded by the caller (whose host copy
    // outlives the phase); folding it into the queries once is the softmax scale.
    launch.scale_columns(d_q, d_scale, top, plane);
    for (int start = 0; start < plane; start += kVaeAttentionChunk) {
        const int count = std::min(kVaeAttentionChunk, plane - start);
        DiffMatrix keys;   // k as a [HW][C] "weight": scores = q . k^T
        keys.device = d_k; keys.type = 0; keys.in = top; keys.out = plane;
        launch.gemm(keys, d_q + static_cast<std::uint64_t>(start) * top * f, d_scores, count, 0, 0);
        {
            std::uint64_t scores = d_scores;
            int columns = plane, rows = count;
            void* arguments[] = {&scores, &columns, &rows};
            launch.run("diff_softmax_rows", static_cast<std::uint32_t>(rows), 1, 256, arguments);
        }
        DiffMatrix values;   // v^T as a [C][HW] "weight": out = p . v
        values.device = d_vt; values.type = 0; values.in = plane; values.out = top;
        launch.gemm(values, d_scores, d_attn + static_cast<std::uint64_t>(start) * top * f, count, 0, 0);
    }
    launch.gemm(mproj, d_attn, y, plane, 0, 0);
    transpose(y, scratch, plane, top);
    launch.add(x, scratch, static_cast<std::uint64_t>(top) * plane);
}

std::uint64_t qi_encoder_workspace(const FlyweightV2Diffusion& d, int height, int width) {
    const QwenImage21Tower& q = *d.qwenimage;
    const std::uint64_t f = sizeof(float);
    const std::uint64_t base = static_cast<std::uint64_t>(q.down_blocks.front().out);
    const std::uint64_t plane_max = base * height * width;   // the widest activation is the first
    const std::uint64_t latent_plane = (static_cast<std::uint64_t>(height) / q.spatial_scale) *
                                       (static_cast<std::uint64_t>(width) / q.spatial_scale);
    const int top = q.enc_top_channels;
    std::uint64_t total = diff_align(plane_max * f) * 4;      // x, scratch, y, the block-input copy
    total += diff_align(plane_max * 2);                       // channels-last bf16 conv input
    total += diff_align(plane_max * f);                       // per-pixel norm reciprocals
    total += diff_align(latent_plane * top * f) * 4;          // q, k, v^T, attention out
    total += diff_align(static_cast<std::uint64_t>(kVaeAttentionChunk) * latent_plane * f);
    total += diff_align(static_cast<std::uint64_t>(top) * f);
    return total;
}

// Encodes an RGBA image (sides multiples of the spatial scale) to normalized
// latents: the posterior's mean (argmax sampling, as the pipeline encodes
// condition images), then `(z - mean) / std` per channel.
void qi_vae_encode(FlyweightV2Diffusion& d, const std::uint8_t* rgba, int width, int height,
                   QiCondition& out) {
    const QwenImage21Tower& q = *d.qwenimage;
    if (!q.has_encoder) throw std::runtime_error("the autoencoder checkpoint has no encoder half");
    const std::uint64_t f = sizeof(float);
    const std::uint64_t plane = static_cast<std::uint64_t>(height) * width;
    diff_ensure_workspace(d, qi_encoder_workspace(d, height, width), "condition image");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const std::uint64_t base = static_cast<std::uint64_t>(q.down_blocks.front().out);
    const std::uint64_t plane_max = base * plane;
    std::uint64_t buffers[3];
    for (auto& buffer : buffers) buffer = arena.take(plane_max * f);
    const std::uint64_t d_copy = arena.take(plane_max * f);
    const std::uint64_t d_hwc = arena.take(plane_max * 2);
    const std::uint64_t d_scales = arena.take(plane_max * f);
    const int top = q.enc_top_channels;
    const std::uint64_t latent_plane = (plane / q.spatial_scale) / q.spatial_scale;
    const std::uint64_t d_q = arena.take(latent_plane * top * f);
    const std::uint64_t d_k = arena.take(latent_plane * top * f);
    const std::uint64_t d_vt = arena.take(latent_plane * top * f);
    const std::uint64_t d_attn = arena.take(latent_plane * top * f);
    const std::uint64_t d_scores = arena.take(static_cast<std::uint64_t>(kVaeAttentionChunk) * latent_plane * f);
    const std::uint64_t d_scale = arena.take(static_cast<std::uint64_t>(top) * f);
    QiVaeState vae{d, launch, d_scales, d_hwc};
    const std::vector<float> query_scale(top, 1.0f / std::sqrt(static_cast<float>(top)));
    launch.upload(d_scale, query_scale.data(), static_cast<std::uint64_t>(top) * f);

    // [4][H][W] in [-1, 1], as VaeImageProcessor.preprocess leaves it.
    std::vector<float> pixels(static_cast<std::size_t>(q.out_channels) * plane);
    for (std::uint64_t p = 0; p < plane; ++p)
        for (int c = 0; c < q.out_channels; ++c)
            pixels[c * plane + p] = rgba[p * 4 + c] / 127.5f - 1.0f;
    std::uint64_t x = buffers[0], scratch = buffers[1], y = buffers[2];
    launch.upload(x, pixels.data(), pixels.size() * f);
    int h = height, w = width;
    vae.conv_from_f32(q.enc_conv_in, x, y, h, w);
    std::swap(x, y);

    for (const auto& block : q.down_blocks) {
        if (flyweight_gpu_copy_device(d_copy, x, static_cast<std::uint64_t>(block.in) * h * w * f, d.stream) != 0)
            throw std::runtime_error("diffusion down-block copy failed");
        const int copy_h = h, copy_w = w;
        for (const auto& resnet : block.resnets) {
            vae.resnet(resnet, x, scratch, y, h, w);
            std::swap(x, y);
        }
        if (block.downsamples) {
            int in = block.out, out = block.out, hh = h, ww = w;
            std::uint64_t input = x, weight = block.downsampler.weight, bias = block.downsampler.bias, output = y;
            void* arguments[] = {&input, &weight, &bias, &output, &in, &out, &hh, &ww};
            launch.run("qi_conv2d_stride2",
                       DiffLaunch::grid_for(static_cast<std::uint64_t>(out) * (h / 2) * (w / 2)), 1, 256, arguments);
            std::swap(x, y);
            h /= 2;
            w /= 2;
        }
        {
            int out = block.out, factor = block.factor, group = block.group;
            int factor_t = block.factor_t, factor_s = block.factor_s, hh = copy_h, ww = copy_w;
            std::uint64_t input = d_copy, output = x;
            void* arguments[] = {&input, &output, &out, &factor, &group, &factor_t, &factor_s, &hh, &ww};
            launch.run("qi_avg_down_add", DiffLaunch::grid_for(static_cast<std::uint64_t>(out) * h * w),
                       1, 256, arguments);
        }
    }
    vae.resnet(q.enc_mid_resnet[0], x, scratch, y, h, w);
    std::swap(x, y);
    qi_mid_attention(d, launch, vae, q.enc_mid_norm, q.enc_mid_q, q.enc_mid_k, q.enc_mid_v, q.enc_mid_proj,
                     x, scratch, y, h, w, d_q, d_k, d_vt, d_attn, d_scores, d_scale);
    vae.resnet(q.enc_mid_resnet[1], x, scratch, y, h, w);
    std::swap(x, y);
    vae.norm(q.enc_norm_out, x, scratch, 0, h * w, true);
    vae.conv_from_f32(q.enc_conv_out, scratch, y, h, w);
    vae.conv_from_f32(q.quant_conv, y, x, h, w);
    // The first z_dim channels are the mean; the log-variance half is not read.
    const std::uint64_t count = static_cast<std::uint64_t>(q.latent_channels) * h * w;
    out.latent_h = h;
    out.latent_w = w;
    out.latents.resize(count);
    launch.download(out.latents.data(), x, count * f);
    const std::uint64_t latent_pixels = static_cast<std::uint64_t>(h) * w;
    for (int c = 0; c < q.latent_channels; ++c)
        for (std::uint64_t p = 0; p < latent_pixels; ++p)
            out.latents[c * latent_pixels + p] =
                (out.latents[c * latent_pixels + p] - q.latents_mean[c]) / q.latents_std[c];
}

// ---- Vision tower ----------------------------------------------------------

std::uint64_t qi_vision_workspace(const FlyweightV2Diffusion& d, std::uint64_t patches) {
    const auto& config = d.encoder->vision;
    const std::uint64_t f = sizeof(float);
    const std::uint64_t width = config.embedding_length, ffn = config.feed_forward_length;
    const std::uint64_t merge = static_cast<std::uint64_t>(config.spatial_merge_size) * config.spatial_merge_size;
    const std::uint64_t tokens = patches / merge, merged = width * merge;
    const std::uint64_t patch_inputs = 3ull * 2 * config.patch_size * config.patch_size;
    std::uint64_t total = 0;
    total += diff_align(patches * patch_inputs * f);
    total += diff_align(patches * width * f) * 4;   // positions, hidden, normed, attn
    total += diff_align(patches * 3 * width * f);   // qkv
    total += diff_align(patches * ffn * f);
    total += diff_align(tokens * merged * f);
    total += diff_align(tokens * config.projection_dim * f);
    total += diff_align(patches * sizeof(std::int32_t)) * 2;
    total += diff_q8_bytes(patches, static_cast<int>(std::max(merged, ffn)));
    return total;
}

// Runs the vision tower over one RGBA image (alpha composited over white, as
// the checkpoint was trained) and fills `out.planes`: the merged tokens
// projected into the decoder's width, then one plane per deepstack layer.
void qi_vision_encode(FlyweightV2Diffusion& d, const std::uint8_t* rgba, std::uint32_t width,
                      std::uint32_t height, QiEncodedImage& out) {
    QwenImage21Tower& q = *d.qwenimage;
    if (!q.has_vision) throw std::runtime_error("the text encoder checkpoint has no vision tower");
    const auto& config = d.encoder->vision;
    const std::uint64_t f = sizeof(float);
    const std::uint64_t plane = static_cast<std::uint64_t>(width) * height;
    std::vector<float> rgb(plane * 3);
    for (std::uint64_t p = 0; p < plane; ++p) {
        const float alpha = rgba[p * 4 + 3] / 255.0f;
        for (int c = 0; c < 3; ++c) {
            const float value = (rgba[p * 4 + c] * alpha + 255.0f * (1.0f - alpha)) / 255.0f;
            rgb[p * 3 + c] = (value - config.image_mean[c]) / config.image_std[c];
        }
    }
    std::vector<float> patches768, positions;
    std::vector<std::int32_t> rows_y, rows_x;
    flyweight::v2::vision_patchify(config, rgb.data(), width, height, patches768, rows_y, rows_x);
    const std::uint32_t grid_w = width / config.patch_size, grid_h = height / config.patch_size;
    flyweight::v2::vision_position_embeddings(config, q.position_table.data(), grid_w, grid_h, positions);
    const std::uint64_t n = rows_y.size();
    const int merge = static_cast<int>(config.spatial_merge_size * config.spatial_merge_size);
    const std::uint64_t tokens = n / merge;
    const int vit_width = static_cast<int>(config.embedding_length), ffn = static_cast<int>(config.feed_forward_length);
    const int patch_pixels = static_cast<int>(config.patch_size * config.patch_size);
    const int patch_inputs = 3 * 2 * patch_pixels, merged = vit_width * merge;
    const int projection = static_cast<int>(config.projection_dim);
    const int heads = static_cast<int>(config.head_count), head_dim = static_cast<int>(config.head_dim());
    // The conv reads (channel, frame, y, x); a still image is the same pixels
    // in both frames, so each patch row is written twice, interleaved.
    std::vector<float> patches(n * patch_inputs);
    for (std::uint64_t p = 0; p < n; ++p)
        for (int c = 0; c < 3; ++c)
            for (int t = 0; t < 2; ++t)
                std::memcpy(patches.data() + p * patch_inputs + (c * 2 + t) * patch_pixels,
                            patches768.data() + p * 3 * patch_pixels + c * patch_pixels,
                            patch_pixels * sizeof(float));

    diff_ensure_workspace(d, qi_vision_workspace(d, n), "condition image");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const std::uint64_t d_patches = arena.take(n * patch_inputs * f);
    const std::uint64_t d_positions = arena.take(n * vit_width * f);
    const std::uint64_t d_hidden = arena.take(n * vit_width * f);
    const std::uint64_t d_normed = arena.take(n * vit_width * f);
    const std::uint64_t d_attn = arena.take(n * vit_width * f);
    const std::uint64_t d_qkv = arena.take(n * 3 * vit_width * f);
    const std::uint64_t d_mlp = arena.take(n * ffn * f);
    const std::uint64_t d_merge = arena.take(tokens * merged * f);
    const std::uint64_t d_out = arena.take(tokens * projection * f);
    const std::uint64_t d_rows_y = arena.take(n * sizeof(std::int32_t));
    const std::uint64_t d_rows_x = arena.take(n * sizeof(std::int32_t));
    const int q8_width = static_cast<int>(std::max<std::uint64_t>(merged, ffn));
    const std::uint64_t d_q8 = arena.take(n * q8_width * 2);
    const std::uint64_t d_q8_scales = arena.take(n * (q8_width / 32) * 4);
    launch.upload(d_patches, patches.data(), patches.size() * f);
    launch.upload(d_positions, positions.data(), positions.size() * f);
    launch.upload(d_rows_y, rows_y.data(), n * sizeof(std::int32_t));
    launch.upload(d_rows_x, rows_x.data(), n * sizeof(std::int32_t));

    const auto layer_norm = [&](std::uint64_t input, std::uint64_t weight, std::uint64_t bias,
                                std::uint64_t output, int norm_width, std::uint64_t rows) {
        int width_arg = norm_width, count = static_cast<int>(rows);
        float epsilon = config.layer_norm_epsilon;
        void* arguments[] = {&input, &weight, &bias, &output, &width_arg, &count, &epsilon};
        launch.run("vision_layer_norm_rows", static_cast<std::uint32_t>(rows), 1, 256, arguments);
    };
    const auto gelu = [&](std::uint64_t x, std::uint64_t elements) {
        int count = static_cast<int>(elements);
        void* arguments[] = {&x, &count};
        launch.run("vision_gelu_rows", DiffLaunch::grid_for(elements), 1, 256, arguments);
    };
    out.planes.assign(1 + q.deepstack_layers.size(), std::vector<float>(tokens * projection));
    out.tokens = static_cast<std::uint32_t>(tokens);
    out.grid_h = grid_h / config.spatial_merge_size;
    out.grid_w = grid_w / config.spatial_merge_size;

    // FLYWEIGHT_QI_TAPS=<dir> writes the residual stream after the patch
    // embedding and after blocks 0, 1 and 8 as raw f32, for parity work.
    const char* taps = std::getenv("FLYWEIGHT_QI_TAPS");
    const auto tap = [&](const char* name) {
        if (!taps) return;
        std::vector<float> host(n * vit_width);
        launch.download(host.data(), d_hidden, host.size() * f);
        std::ofstream file(std::string(taps) + "/vision_" + name + ".f32", std::ios::binary);
        file.write(reinterpret_cast<const char*>(host.data()), host.size() * f);
    };
    launch.gemm(q.patch, d_patches, d_hidden, n, d_q8, d_q8_scales);
    launch.add_bias(d_hidden, q.patch_bias, vit_width, n);
    tap("patch");
    launch.add(d_hidden, d_positions, n * vit_width);
    tap("pos");
    DiffStreamer streamer{d, {}, 0};
    for (auto& block : q.vision_blocks) streamer.refs.push_back(qi_vision_matrices(block));
    std::size_t deepstack_slot = 1;
    for (std::size_t index = 0; index < q.vision_blocks.size(); ++index) {
        auto& block = q.vision_blocks[index];
        streamer.begin(index);
        layer_norm(d_hidden, block.ln1_weight, block.ln1_bias, d_normed, vit_width, n);
        launch.gemm(block.qkv, d_normed, d_qkv, n, d_q8, d_q8_scales);
        {
            std::uint64_t qkv = d_qkv, y = d_rows_y, x = d_rows_x;
            int heads_arg = heads, head_dim_arg = head_dim, rows = static_cast<int>(n);
            float theta = 10000.0f;
            void* arguments[] = {&qkv, &y, &x, &heads_arg, &head_dim_arg, &rows, &theta};
            launch.run("vision_rope_rows", static_cast<std::uint32_t>(n), 2, 256, arguments);
        }
        {
            std::uint64_t qkv = d_qkv, output = d_attn;
            int heads_arg = heads, head_dim_arg = head_dim, rows = static_cast<int>(n);
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            void* arguments[] = {&qkv, &output, &heads_arg, &head_dim_arg, &rows, &scale};
            launch.run("vision_attention_rows", static_cast<std::uint32_t>((n + 31) / 32),
                       static_cast<std::uint32_t>(heads), 256, arguments);
        }
        launch.gemm(block.out, d_attn, d_normed, n, d_q8, d_q8_scales);
        launch.add(d_hidden, d_normed, n * vit_width);
        layer_norm(d_hidden, block.ln2_weight, block.ln2_bias, d_normed, vit_width, n);
        launch.gemm(block.up, d_normed, d_mlp, n, d_q8, d_q8_scales);
        gelu(d_mlp, n * ffn);
        launch.gemm(block.down, d_mlp, d_normed, n, d_q8, d_q8_scales);
        launch.add(d_hidden, d_normed, n * vit_width);
        if (index == 0) tap("block0");
        if (index == 1) tap("block1");
        if (index == 8) tap("block8");
        if (block.deepstack) {
            // The block output viewed as merged rows: the four patches of a
            // window are consecutive, so a token is one row of `merged`.
            layer_norm(d_hidden, block.deepstack_norm_weight, block.deepstack_norm_bias, d_normed, merged, tokens);
            launch.gemm(block.deepstack_fc1, d_normed, d_merge, tokens, d_q8, d_q8_scales);
            gelu(d_merge, tokens * merged);
            launch.gemm(block.deepstack_fc2, d_merge, d_out, tokens, d_q8, d_q8_scales);
            launch.download(out.planes[deepstack_slot++].data(), d_out, tokens * projection * f);
        }
        streamer.end(index);
    }
    layer_norm(d_hidden, q.post_ln_weight, q.post_ln_bias, d_normed, vit_width, n);
    launch.gemm(q.merger_fc1, d_normed, d_merge, tokens, d_q8, d_q8_scales);
    gelu(d_merge, tokens * merged);
    launch.gemm(q.merger_fc2, d_merge, d_out, tokens, d_q8, d_q8_scales);
    launch.download(out.planes[0].data(), d_out, tokens * projection * f);
}

// ---- Text encoder with images ----------------------------------------------

// The Qwen3-VL decoder over `tokens` with the images' vision rows in place of
// their <|image_pad|> tokens, interleaved M-RoPE positions, and each deepstack
// plane added to the image rows after the corresponding early layer. Writes
// the last layer's output (no final norm) as [count][hidden] f32.
void qi_encode_prompt(FlyweightV2Diffusion& d, const std::uint32_t* tokens, std::uint64_t count,
                      const std::vector<QiEncodedImage>& images, float* output) {
    if (!count) throw std::runtime_error("empty prompt");
    if (count > d.max_prompt_tokens)
        throw std::runtime_error("prompt has " + std::to_string(count) + " tokens; the tower allows " +
                                 std::to_string(d.max_prompt_tokens));
    const std::uint64_t f = sizeof(float);
    const int hidden = d.encoder_hidden;
    const int q_width = d.encoder_heads * d.encoder_head_dim;
    const int kv_width = d.encoder_kv_heads * d.encoder_head_dim;
    std::size_t planes = 1;
    for (const auto& image : images) planes = std::max(planes, image.planes.size());
    std::uint64_t image_rows = 0;
    for (const auto& image : images) image_rows += image.tokens;
    diff_ensure_workspace(d, diff_encoder_workspace(d, count) +
                              diff_align(image_rows * (planes - 1) * hidden * f) +
                              diff_align(count * 3 * sizeof(std::int32_t)), "prompt");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const std::uint64_t d_hidden = arena.take(count * hidden * f);
    const std::uint64_t d_normed = arena.take(count * hidden * f);
    const std::uint64_t d_projected = arena.take(count * hidden * f);
    const std::uint64_t d_q = arena.take(count * q_width * f);
    const std::uint64_t d_attn = arena.take(count * q_width * f);
    const std::uint64_t d_k = arena.take(count * kv_width * f);
    const std::uint64_t d_v = arena.take(count * kv_width * f);
    const std::uint64_t d_gate = arena.take(count * d.encoder_intermediate * f);
    const std::uint64_t d_up = arena.take(count * d.encoder_intermediate * f);
    const std::uint64_t d_positions = arena.take(count * 3 * sizeof(std::int32_t));
    const std::uint64_t d_q16 = arena.take(count * q_width * 2);
    const std::uint64_t d_k16 = arena.take(count * kv_width * 2);
    const std::uint64_t d_v16 = arena.take(count * kv_width * 2);
    const int q8_width = std::max(d.encoder_intermediate, hidden);
    const std::uint64_t d_q8 = arena.take(count * q8_width * 2);
    const std::uint64_t d_q8_scales = arena.take(count * (q8_width / 32) * 4);
    const std::uint64_t d_deepstack = arena.take(image_rows * (planes - 1) * hidden * f);

    // Token embeddings on the host, the images' projected rows over their
    // pad runs; positions per get_rope_index -- text advances all three axes,
    // an image freezes the frame axis and spreads its merged grid over
    // height and width from that position, then advances by its longer side.
    std::vector<float> rows(count * hidden);
    std::vector<std::int32_t> positions(count * 3);
    std::vector<float> deepstack(image_rows * (planes - 1) * hidden);
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
        std::vector<const QiEncodedImage*> ordered;
        for (const auto& image : images) ordered.push_back(&image);
        std::sort(ordered.begin(), ordered.end(),
                  [](const QiEncodedImage* a, const QiEncodedImage* b) { return a->token_offset < b->token_offset; });
        std::int32_t position = 0;
        std::uint64_t row = 0, stacked = 0;
        for (const auto* image : ordered) {
            if (image->token_offset < row || image->token_offset + image->tokens > count)
                throw std::runtime_error("an image's token run is outside the prompt or overlaps another");
            for (; row < image->token_offset; ++row, ++position)
                positions[row * 3] = positions[row * 3 + 1] = positions[row * 3 + 2] = position;
            std::memcpy(rows.data() + row * hidden, image->planes[0].data(),
                        static_cast<std::size_t>(image->tokens) * hidden * f);
            for (std::size_t plane = 1; plane < image->planes.size(); ++plane)
                std::memcpy(deepstack.data() + ((plane - 1) * image_rows + stacked) * hidden,
                            image->planes[plane].data(), static_cast<std::size_t>(image->tokens) * hidden * f);
            for (std::uint32_t h = 0; h < image->grid_h; ++h)
                for (std::uint32_t w = 0; w < image->grid_w; ++w, ++row) {
                    positions[row * 3] = position;
                    positions[row * 3 + 1] = position + static_cast<std::int32_t>(h);
                    positions[row * 3 + 2] = position + static_cast<std::int32_t>(w);
                }
            position += static_cast<std::int32_t>(std::max(image->grid_h, image->grid_w));
            stacked += image->tokens;
        }
        for (; row < count; ++row, ++position)
            positions[row * 3] = positions[row * 3 + 1] = positions[row * 3 + 2] = position;
        launch.upload(d_hidden, rows.data(), rows.size() * f);
        launch.upload(d_positions, positions.data(), positions.size() * sizeof(std::int32_t));
        if (!deepstack.empty()) launch.upload(d_deepstack, deepstack.data(), deepstack.size() * f);
        // Deepstack rows are contiguous per plane in `stacked` order; the
        // adds below need each image's offset within the prompt.
        (void)ordered;
    }
    const std::uint32_t* sections = d.encoder->config.rope_sections;
    int section_t = static_cast<int>(sections[0]), section_h = static_cast<int>(sections[1]);
    int section_w = static_cast<int>(sections[2]);
    if (!d.encoder->config.has_rope_sections()) { section_t = d.encoder_head_dim / 2; section_h = section_w = 0; }
    const auto rope = [&](std::uint64_t x, int heads, int stride) {
        int head_dim = d.encoder_head_dim, rows_arg = static_cast<int>(count), zero = 0;
        std::uint64_t positions_arg = d_positions, table = d.encoder_rope_frequencies;
        void* arguments[] = {&x, &positions_arg, &table, &heads, &head_dim, &rows_arg, &stride, &zero,
                             &section_t, &section_h, &section_w};
        launch.run("qi_rope_mrope_rows", static_cast<std::uint32_t>(count), 1, 256, arguments);
    };
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
        rope(d_q, d.encoder_heads, q_width);
        rope(d_k, d.encoder_kv_heads, kv_width);
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
        // Deepstack: the k-th vision plane joins the image rows after layer k.
        if (index + 1 < planes) {
            std::uint64_t stacked = 0;
            for (const auto& image : images) {
                launch.add(d_hidden + image.token_offset * hidden * f,
                           d_deepstack + (index * image_rows + stacked) * hidden * f,
                           static_cast<std::uint64_t>(image.tokens) * hidden);
                stacked += image.tokens;
            }
        }
        streamer.end(index);
    }
    launch.download(output, d_hidden, count * hidden * f);
}

// ---- Editing ---------------------------------------------------------------

void qi_check_images(const FlyweightV2Diffusion& d, const FlyweightV2DiffusionImage* images,
                     std::uint32_t image_count) {
    for (std::uint32_t i = 0; i < image_count; ++i) {
        if (!images[i].rgba || !images[i].width || !images[i].height ||
            images[i].width % 32 || images[i].height % 32)
            throw std::runtime_error("condition image sides must be positive multiples of 32");
        if (images[i].width > d.max_width || images[i].height > d.max_height)
            throw std::runtime_error("a condition image exceeds the tower's size limit");
    }
}

// Runs the vision tower over each image and marks its token run in
// `pad_mask` (sized `count`), checking the run against the image's size.
void qi_encode_images(FlyweightV2Diffusion& d, std::uint64_t count, std::uint32_t caption_drop,
                      const FlyweightV2DiffusionImage* images, std::uint32_t image_count,
                      std::vector<QiEncodedImage>& encoded, std::vector<std::uint8_t>& pad_mask) {
    const std::uint32_t token_side = d.encoder->vision.token_side();   // pixels per vision token
    encoded.assign(image_count, {});
    pad_mask.assign(count, 0);
    for (std::uint32_t i = 0; i < image_count; ++i) {
        const auto& image = images[i];
        const std::uint64_t expected = static_cast<std::uint64_t>(image.width / token_side) * (image.height / token_side);
        if (image.token_offset + expected > count || image.token_offset < caption_drop)
            throw std::runtime_error("a condition image's token run is outside the prompt");
        qi_vision_encode(d, image.rgba, image.width, image.height, encoded[i]);
        if (encoded[i].tokens != expected)
            throw std::runtime_error("vision token count does not match the image's size");
        // FLYWEIGHT_QI_VISION_PLANES=<prefix> swaps in reference planes from
        // <prefix><k>.f32, to tell the tower's error from the decoder's.
        if (const char* prefix = std::getenv("FLYWEIGHT_QI_VISION_PLANES"))
            for (std::size_t k = 0; k < encoded[i].planes.size(); ++k) {
                std::ifstream file(std::string(prefix) + std::to_string(k) + ".f32", std::ios::binary);
                if (file) file.read(reinterpret_cast<char*>(encoded[i].planes[k].data()),
                                    encoded[i].planes[k].size() * sizeof(float));
            }
        encoded[i].token_offset = image.token_offset;
        for (std::uint64_t t = 0; t < expected; ++t) {
            if (pad_mask[image.token_offset + t]) throw std::runtime_error("image token runs overlap");
            pad_mask[image.token_offset + t] = 1;
        }
    }
}

void qi_encode_prompt_images(FlyweightV2Diffusion& d, const std::uint32_t* tokens, std::uint64_t count,
                             const FlyweightV2DiffusionImage* images, std::uint32_t image_count,
                             float* output) {
    std::vector<QiEncodedImage> encoded;
    std::vector<std::uint8_t> pad_mask;
    qi_encode_images(d, count, 0, images, image_count, encoded, pad_mask);
    qi_encode_prompt(d, tokens, count, encoded, output);
}

std::uint32_t qi_encode_vision(FlyweightV2Diffusion& d, const std::uint8_t* rgba, std::uint32_t width,
                               std::uint32_t height, float* output, std::uint64_t capacity) {
    QiEncodedImage encoded;
    qi_vision_encode(d, rgba, width, height, encoded);
    std::uint64_t needed = 0;
    for (const auto& plane : encoded.planes) needed += plane.size();
    if (capacity < needed) throw std::runtime_error("vision output buffer is too small");
    for (const auto& plane : encoded.planes) {
        std::memcpy(output, plane.data(), plane.size() * sizeof(float));
        output += plane.size();
    }
    return static_cast<std::uint32_t>(encoded.planes.size());
}

void qi_encode_image(FlyweightV2Diffusion& d, const std::uint8_t* rgba, std::uint32_t width,
                     std::uint32_t height, float* output) {
    QiCondition condition;
    qi_vae_encode(d, rgba, static_cast<int>(width), static_cast<int>(height), condition);
    std::memcpy(output, condition.latents.data(), condition.latents.size() * sizeof(float));
}

void qi_edit(FlyweightV2Diffusion& d, const std::uint32_t* tokens, std::uint64_t count,
             std::uint32_t caption_drop, const FlyweightV2DiffusionImage* images,
             std::uint32_t image_count, std::uint32_t width, std::uint32_t height,
             std::uint32_t steps, float shift, std::uint64_t seed, const float* initial_latents,
             FlyweightV2DiffusionProgress progress, void* user_data, std::uint8_t* pixels) {
    const QwenImage21Tower& q = *d.qwenimage;
    const int scale = q.spatial_scale;
    std::vector<QiEncodedImage> encoded;
    std::vector<std::uint8_t> pad_mask;
    qi_encode_images(d, count, caption_drop, images, image_count, encoded, pad_mask);
    std::vector<QiCondition> conditions(image_count);
    for (std::uint32_t i = 0; i < image_count; ++i)
        qi_vae_encode(d, images[i].rgba, static_cast<int>(images[i].width),
                      static_cast<int>(images[i].height), conditions[i]);
    // Conditions in prompt order, which is the order their token runs appear.
    std::vector<std::size_t> order(image_count);
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return images[a].token_offset < images[b].token_offset; });
    std::vector<QiCondition> ordered;
    for (const auto index : order) ordered.push_back(std::move(conditions[index]));

    std::vector<float> caption(count * static_cast<std::uint64_t>(d.encoder_hidden));
    qi_encode_prompt(d, tokens, count, encoded, caption.data());
    if (progress && progress(user_data, 0, steps) != 0) throw std::runtime_error("cancelled");
    const float* conditioning = caption.data() + static_cast<std::uint64_t>(caption_drop) * d.encoder_hidden;
    const std::uint64_t caption_tokens = count - caption_drop;
    const std::uint8_t* mask = pad_mask.data() + caption_drop;

    const int latent_h = static_cast<int>(height / scale), latent_w = static_cast<int>(width / scale);
    const std::size_t latent_count = static_cast<std::size_t>(q.channels) * latent_h * latent_w;
    std::vector<float> latents(latent_count), velocity(latent_count);
    if (initial_latents) std::memcpy(latents.data(), initial_latents, latent_count * sizeof(float));
    else diff_gaussian_noise(seed, latents);
    const auto sigmas = qi_sigmas(static_cast<int>(steps), shift, static_cast<std::uint64_t>(latent_h) * latent_w);
    std::vector<float> dummy_caption_rows;
    const QiLayout layout = qi_layout(conditioning, caption_tokens, q.caption_dim, mask, ordered,
                                      latent_h, latent_w, dummy_caption_rows);
    const std::uint64_t prefix = layout.prefix_rows;
    bool use_kv_cache = d.flash_attention && !d.exact && q.head_dim == 128 && steps > 1 && prefix > 0;
    if (const char* env = std::getenv("FLYWEIGHT_DIFF_KV_CACHE")) {
        if (env[0] == '0') use_kv_cache = false;
    }
    std::unique_ptr<QiKvCache> cache;
    if (use_kv_cache) {
        cache = std::make_unique<QiKvCache>(static_cast<int>(q.blocks.size()), q.heads, prefix);
        if (!cache->valid()) {
            cache.reset();
            use_kv_cache = false;
        }
    }
    for (std::uint32_t step = 0; step < steps; ++step) {
        const float sigma = sigmas[step], next = sigmas[step + 1];
        if (sigma == next) continue;
        const bool is_decode = use_kv_cache && (step > 0);
        qi_transformer_forward_joint(d, latents.data(), latent_h, latent_w, conditioning, caption_tokens,
                                     mask, ordered, sigma, velocity.data(), cache.get(), is_decode);
        for (std::size_t i = 0; i < latent_count; ++i) latents[i] += (next - sigma) * velocity[i];
        if (progress && progress(user_data, step + 1, steps) != 0) throw std::runtime_error("cancelled");
    }
    qi_decode_latents(d, latents.data(), latent_h, latent_w, pixels);
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
    const std::uint64_t prefix = caption_tokens;
    bool use_kv_cache = d.flash_attention && !d.exact && q.head_dim == 128 && steps > 1 && prefix > 0;
    if (const char* env = std::getenv("FLYWEIGHT_DIFF_KV_CACHE")) {
        if (env[0] == '0') use_kv_cache = false;
    }
    std::unique_ptr<QiKvCache> cache;
    if (use_kv_cache) {
        cache = std::make_unique<QiKvCache>(static_cast<int>(q.blocks.size()), q.heads, prefix);
        if (!cache->valid()) {
            cache.reset();
            use_kv_cache = false;
        }
    }
    for (std::uint32_t step = 0; step < steps; ++step) {
        const float sigma = sigmas[step], next = sigmas[step + 1];
        if (sigma == next) continue;
        const bool is_decode = use_kv_cache && (step > 0);
        qi_transformer_forward_joint(d, latents.data(), latent_h, latent_w, conditioning, caption_tokens,
                                     nullptr, {}, sigma, velocity.data(), cache.get(), is_decode);
        // Euler on the model's own output, which is the velocity as predicted.
        for (std::size_t i = 0; i < latent_count; ++i) latents[i] += (next - sigma) * velocity[i];
        if (progress && progress(user_data, step + 1, steps) != 0) throw std::runtime_error("cancelled");
    }
    qi_decode_latents(d, latents.data(), latent_h, latent_w, pixels);
}

}  // namespace
