// MiniMax-H3: the packed video/audio DiT and its pipeline, on the diffusion
// tower's helpers (v2_diffusion.inc: streaming, GEMMs, attention, arena).
// Reference: diffusers' transformer_minimax_h3.py and the modular pipeline's
// before_denoise.py for the packed layout; stable-diffusion.cpp's
// minimax_h3.hpp for the pruned adaLN table the released GGUFs carry and
// for the fused fc1 = [gate; up] convention of those files.
//
// The sequence is [text | audio rows | video rows]: text rows sit on the
// time axis at their index, audio rows (channel-major, two channels) continue
// the clock one unit per 40 Hz latent, video rows follow the frame clock in
// 5/3 * (1, 4, 4, 4, 4) spans over an aspect-normalized 32-unit grid. Every
// row is modulated by the (timestep, modality) pair it belongs to.

namespace {

constexpr int kH3Modalities = 3;          // video 0, text 1, audio 2
constexpr int kH3AudioChannels = 2;
constexpr float kH3FrameRescale = 5.0f / 3.0f;
constexpr int kH3FramesPerLatent[5] = {1, 4, 4, 4, 4};
constexpr float kH3SpatialScale = 32.0f;

struct H3Block {
    std::uint64_t norm1 = 0, norm2 = 0, q_norm = 0, k_norm = 0;
    DiffMatrix qkv, out, fc1, fc2, adaln;   // adaln absent on refiner blocks
    bool modulated = false;
};

DiffLayerRef h3_block_matrices(H3Block& block) {
    DiffLayerRef ref;
    ref.add(block.qkv);
    ref.add(block.out);
    ref.add(block.fc1);
    ref.add(block.fc2);
    if (block.modulated) ref.add(block.adaln);
    return ref;
}

}  // namespace

struct H3VaeBlock {
    std::uint64_t norm1 = 0, norm2 = 0, scale1 = 0, scale2 = 0;
    DiffMatrix qkv, out, w1, w2;
};

struct FlyweightV2H3;
namespace {
void h3_upload_vae(FlyweightV2Diffusion& d, FlyweightV2H3& h3);
std::uint64_t h3_vae_workspace_bytes(const FlyweightV2H3& h3, std::uint64_t tokens, std::uint64_t voxels);
}  // namespace

// H3-specific state hangs off the shared tower.
struct FlyweightV2H3 {
    // The ViT video VAE decoder.
    std::vector<H3VaeBlock> vae_blocks;
    DiffMatrix vae_embed, vae_out, vae_post_quant;
    std::uint64_t vae_register = 0, vae_norm_out = 0, vae_norm_out_bias = 0, vae_ones = 0, vae_inv_freq = 0;
    std::vector<float> vae_mean, vae_std;
    int vae_dim = 0, vae_heads = 0, vae_head_dim = 0, vae_ffn = 0, vae_layers = 0, vae_registers = 0;
    int vae_latent_channels = 0, vae_spatial = 16, vae_temporal = 4, vae_clip_length = 17, vae_token_drop = 3;
    int vae_rope_freqs = 8;
    float vae_eps = 1e-5f;
    bool has_vae = false;
    std::vector<H3Block> refiner, blocks;
    std::uint64_t refiner_norm = 0;
    DiffMatrix video_in, audio_in, condition, video_out, audio_out, final_adaln;
    std::uint64_t final_norm = 0;
    std::uint64_t inv_freq = 0;              // rope.inv_freq on the device
    std::vector<float> adaln_table;          // [grid][width], host
    int adaln_grid = 0, adaln_width = 0;
    int hidden = 0, heads = 0, head_dim = 0, inner = 0, ffn = 0;
    int video_channels = 0, audio_channels = 0, patch = 2, text_dim = 0, rope_freqs = 0;
    float eps = 1e-5f;
};

namespace {

H3Block h3_upload_block(FlyweightV2Diffusion& d, FlyweightV2H3& h3, const FlyweightV2Model& model,
                        const std::string& prefix, bool modulated) {
    DiffGroup group(d);
    H3Block block;
    block.modulated = modulated;
    block.norm1 = diff_upload_f32(d, model, prefix + ".norm1.weight", h3.hidden);
    block.norm2 = diff_upload_f32(d, model, prefix + ".norm2.weight", h3.hidden);
    block.q_norm = diff_upload_f32(d, model, prefix + ".attn.q_norm.weight", h3.head_dim);
    block.k_norm = diff_upload_f32(d, model, prefix + ".attn.k_norm.weight", h3.head_dim);
    block.qkv = diff_upload_matrix(d, model, prefix + ".attn.qkv_proj.weight", h3.hidden, 3 * h3.inner);
    block.out = diff_upload_matrix(d, model, prefix + ".attn.out_proj.weight", h3.inner, h3.hidden);
    block.fc1 = diff_upload_matrix(d, model, prefix + ".mlp.fc1.weight", h3.hidden, 2 * h3.ffn);
    block.fc2 = diff_upload_matrix(d, model, prefix + ".mlp.fc2.weight", h3.ffn, h3.hidden);
    if (modulated)
        block.adaln = diff_upload_matrix(d, model, prefix + ".adaln_proj.linear.weight", h3.adaln_width,
                                         6 * kH3Modalities * h3.hidden, prefix + ".adaln_proj.linear.bias");
    return block;
}

void h3_upload_transformer(FlyweightV2Diffusion& d, FlyweightV2H3& h3) {
    const auto& model = *d.transformer;
    const auto& config = model.config;
    h3.hidden = static_cast<int>(config.hidden_size);
    h3.heads = static_cast<int>(config.attention_heads);
    h3.head_dim = static_cast<int>(config.attention_head_dim);
    h3.inner = h3.heads * h3.head_dim;
    h3.ffn = static_cast<int>(config.intermediate_size);
    h3.video_channels = static_cast<int>(config.in_channels);
    h3.audio_channels = static_cast<int>(config.audio_channels);
    h3.patch = static_cast<int>(config.patch_size);
    h3.text_dim = static_cast<int>(config.caption_dim);
    h3.rope_freqs = static_cast<int>(config.rope_freq_dim);
    h3.eps = config.rms_norm_epsilon;
    if (h3.head_dim > 128) throw std::runtime_error("H3 head width above 128 is not supported");
    if (2 * 3 * h3.rope_freqs > h3.head_dim) throw std::runtime_error("H3 rope covers more than the head");
    d.dim = h3.hidden;
    d.heads = h3.heads;
    d.head_dim = h3.head_dim;
    d.intermediate = h3.ffn;
    d.caption_dim = h3.text_dim;

    // The pruned adaLN: a [grid][width] timestep table read on the host.
    {
        const Tensor& table = diff_tensor(model, "adaln_t_table");
        if (table.shape.size() != 2) throw std::runtime_error("adaln_t_table is not a matrix");
        h3.adaln_width = static_cast<int>(table.shape[0]);
        h3.adaln_grid = static_cast<int>(table.shape[1]);
        h3.adaln_table.resize(static_cast<std::size_t>(h3.adaln_grid) * h3.adaln_width);
        const auto* source = tensor_data(model, table);
        for (std::size_t i = 0; i < h3.adaln_table.size(); ++i) h3.adaln_table[i] = tensor_value(source, table.type, i);
    }
    h3.inv_freq = diff_upload_f32(d, model, "rope.inv_freq", h3.rope_freqs);
    const int patch_inputs = h3.video_channels * h3.patch * h3.patch;
    h3.video_in = diff_upload_matrix(d, model, "video_patch_proj.weight", patch_inputs, h3.hidden, "video_patch_proj.bias");
    h3.audio_in = diff_upload_matrix(d, model, "audio_patch_proj.weight", h3.audio_channels, h3.hidden, "audio_patch_proj.bias");
    h3.condition = diff_upload_matrix(d, model, "condition_proj.weight", h3.text_dim, h3.hidden, "condition_proj.bias");
    h3.final_norm = diff_upload_f32(d, model, "final_layer.norm.weight", h3.hidden);
    h3.final_adaln = diff_upload_matrix(d, model, "final_layer.adaln_proj.linear.weight", h3.adaln_width, 2 * h3.hidden,
                                        "final_layer.adaln_proj.linear.bias");
    h3.video_out = diff_upload_matrix(d, model, "final_layer.video_out.weight", h3.hidden, patch_inputs, "final_layer.video_out.bias");
    h3.audio_out = diff_upload_matrix(d, model, "final_layer.audio_out.weight", h3.hidden, h3.audio_channels, "final_layer.audio_out.bias");
    h3.refiner_norm = diff_upload_f32(d, model, "token_refiner.final_norm.weight", h3.hidden);
    for (std::uint32_t index = 0; index < config.refiner_layer_count; ++index)
        h3.refiner.push_back(h3_upload_block(d, h3, model, "token_refiner.blocks." + std::to_string(index), false));
    for (std::uint32_t index = 0; index < config.layer_count; ++index)
        h3.blocks.push_back(h3_upload_block(d, h3, model, "blocks." + std::to_string(index), true));
}

// ---- The packed sequence ---------------------------------------------------

struct H3Layout {
    int text = 0, audio_latents = 0, latent_t = 0, latent_h = 0, latent_w = 0;
    int rows_per_frame = 0, audio_rows = 0, video_rows = 0, rows = 0;
    int audio_start = 0, video_start = 0;
    std::vector<float> positions;       // [rows][3]
    std::vector<std::int32_t> block_ids; // (timestep row * 3 + modality) per row
    std::vector<std::int32_t> final_ids; // timestep row per row
    std::vector<float> timesteps;        // distinct, in the order the ids index
};

// One aspect-normalized spatial axis, as `_spatial_position_grid`: dim/patch
// coordinates on [left, left + ratio) scaled by 32, endpoint excluded.
std::vector<float> h3_spatial_axis(int dim, int patch, double sqrt_area) {
    const int count = dim / patch;
    const double ratio = dim / sqrt_area, left = (1.0 - ratio) / 2.0;
    std::vector<float> axis(count);
    for (int i = 0; i < count; ++i)
        axis[i] = static_cast<float>((left + i * (ratio / count)) * kH3SpatialScale);
    return axis;
}

H3Layout h3_layout(const FlyweightV2H3& h3, int text, int latent_t, int latent_h, int latent_w,
                   int audio_latents, float t_video, float t_audio) {
    H3Layout layout;
    layout.text = text;
    layout.latent_t = latent_t;
    layout.latent_h = latent_h;
    layout.latent_w = latent_w;
    layout.audio_latents = audio_latents;
    const double sqrt_area = std::sqrt(static_cast<double>(latent_h) * latent_w);
    const auto h_axis = h3_spatial_axis(latent_h, h3.patch, sqrt_area);
    const auto w_axis = h3_spatial_axis(latent_w, h3.patch, sqrt_area);
    layout.rows_per_frame = static_cast<int>(h_axis.size() * w_axis.size());
    layout.audio_rows = audio_latents * kH3AudioChannels;
    layout.video_rows = latent_t * layout.rows_per_frame;
    layout.audio_start = text;
    layout.video_start = text + layout.audio_rows;
    layout.rows = layout.video_start + layout.video_rows;
    layout.positions.assign(static_cast<std::size_t>(layout.rows) * 3, 0.0f);
    layout.block_ids.assign(layout.rows, 0);
    layout.final_ids.assign(layout.rows, 0);
    auto timestep_row = [&](float value) {
        for (std::size_t i = 0; i < layout.timesteps.size(); ++i)
            if (layout.timesteps[i] == value) return static_cast<int>(i);
        layout.timesteps.push_back(value);
        return static_cast<int>(layout.timesteps.size() - 1);
    };
    const int video_row = timestep_row(t_video), audio_row = timestep_row(t_audio);
    auto place = [&](int row, float t, float h, float w, int timestep, int modality) {
        layout.positions[row * 3] = t;
        layout.positions[row * 3 + 1] = h;
        layout.positions[row * 3 + 2] = w;
        layout.block_ids[row] = timestep * kH3Modalities + modality;
        layout.final_ids[row] = timestep;
    };
    for (int i = 0; i < text; ++i) place(i, static_cast<float>(i), 0.0f, 0.0f, video_row, 1);
    // Audio: channel-major, one clock unit per latent, pinned to the width
    // grid's two ends; the clock starts where the text rows left it.
    for (int channel = 0; channel < kH3AudioChannels; ++channel)
        for (int t = 0; t < audio_latents; ++t)
            place(layout.audio_start + channel * audio_latents + t, static_cast<float>(text + t), 0.0f,
                  channel == 0 ? w_axis.front() : w_axis.back(), audio_row, 2);
    float cursor = static_cast<float>(text);
    for (int t = 0; t < latent_t; ++t) {
        for (std::size_t hi = 0; hi < h_axis.size(); ++hi)
            for (std::size_t wi = 0; wi < w_axis.size(); ++wi)
                place(layout.video_start + t * layout.rows_per_frame + static_cast<int>(hi * w_axis.size() + wi),
                      cursor, h_axis[hi], w_axis[wi], video_row, 0);
        cursor += kH3FrameRescale * kH3FramesPerLatent[t % 5];
    }
    return layout;
}

// Video latents [C][T][H][W] -> rows (t, h, w) of (c, ph, pw) features.
void h3_patchify(const FlyweightV2H3& h3, const float* latents, int latent_t, int latent_h, int latent_w,
                 std::vector<float>& rows) {
    const int p = h3.patch, c_count = h3.video_channels;
    const int th = latent_h / p, tw = latent_w / p, features = c_count * p * p;
    rows.assign(static_cast<std::size_t>(latent_t) * th * tw * features, 0.0f);
    for (int t = 0; t < latent_t; ++t)
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x)
                for (int c = 0; c < c_count; ++c)
                    for (int ph = 0; ph < p; ++ph)
                        for (int pw = 0; pw < p; ++pw)
                            rows[((static_cast<std::size_t>(t) * th + y) * tw + x) * features + (c * p + ph) * p + pw] =
                                latents[((static_cast<std::size_t>(c) * latent_t + t) * latent_h + y * p + ph) * latent_w + x * p + pw];
}

void h3_unpatchify(const FlyweightV2H3& h3, const std::vector<float>& rows, int latent_t, int latent_h, int latent_w,
                   float* latents) {
    const int p = h3.patch, c_count = h3.video_channels;
    const int th = latent_h / p, tw = latent_w / p, features = c_count * p * p;
    for (int t = 0; t < latent_t; ++t)
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x)
                for (int c = 0; c < c_count; ++c)
                    for (int ph = 0; ph < p; ++ph)
                        for (int pw = 0; pw < p; ++pw)
                            latents[((static_cast<std::size_t>(c) * latent_t + t) * latent_h + y * p + ph) * latent_w + x * p + pw] =
                                rows[((static_cast<std::size_t>(t) * th + y) * tw + x) * features + (c * p + ph) * p + pw];
}

// The 8-wide adaLN input for a timestep: linear interpolation in the table.
void h3_adaln_input(const FlyweightV2H3& h3, float t, float* out) {
    const float position = std::min(std::max(t, 0.0f), 1.0f) * (h3.adaln_grid - 1);
    const int lower = std::min(static_cast<int>(std::floor(position)), h3.adaln_grid - 2);
    const float fraction = position - lower;
    for (int i = 0; i < h3.adaln_width; ++i) {
        const float a = h3.adaln_table[static_cast<std::size_t>(lower) * h3.adaln_width + i];
        const float b = h3.adaln_table[static_cast<std::size_t>(lower + 1) * h3.adaln_width + i];
        out[i] = a + (b - a) * fraction;
    }
}

struct H3Buffers {
    std::uint64_t hidden = 0, normed = 0, attn = 0, qkv = 0, fused = 0, gated = 0;
    std::uint64_t q16 = 0, k16 = 0, v16 = 0, q8 = 0, q8_scales = 0;
    std::uint64_t positions = 0, block_ids = 0, final_ids = 0;
    std::uint64_t adaln_in = 0, mods = 0, final_mods = 0;
    std::uint64_t caption = 0, patches = 0, audio = 0, video_out = 0, audio_out = 0;
};

std::uint64_t h3_workspace_bytes(const FlyweightV2H3& h3, std::uint64_t rows, std::uint64_t text, std::uint64_t timesteps) {
    const std::uint64_t f = sizeof(float);
    std::uint64_t total = 0;
    total += diff_align(rows * h3.hidden * f) * 2;             // hidden, normed
    total += diff_align(rows * h3.inner * f);                  // attention output
    total += diff_align(rows * 3 * h3.inner * f);              // qkv
    total += diff_align(rows * 2 * h3.ffn * f);                // fused gate|up
    total += diff_align(rows * h3.ffn * f);                    // gated
    total += diff_align(rows * h3.inner * 2) * 3;              // bf16 attention planes
    total += diff_q8_bytes(rows, std::max(h3.ffn, h3.inner));  // GEMM input staging
    total += diff_align(rows * 3 * f) + diff_align(rows * 4) * 2;  // positions, ids
    total += diff_align(timesteps * h3.adaln_width * f);
    total += diff_align(timesteps * 6 * kH3Modalities * h3.hidden * f);
    total += diff_align(timesteps * 2 * h3.hidden * f);
    total += diff_align(text * h3.text_dim * f);
    total += diff_align(rows * h3.video_channels * h3.patch * h3.patch * f) * 2;  // patches in / out
    total += diff_align(rows * h3.audio_channels * f) * 2;
    return total;
}

H3Buffers h3_buffers(const FlyweightV2H3& h3, DiffArena& arena, std::uint64_t rows, std::uint64_t text, std::uint64_t timesteps) {
    const std::uint64_t f = sizeof(float);
    H3Buffers b;
    b.hidden = arena.take(rows * h3.hidden * f);
    b.normed = arena.take(rows * h3.hidden * f);
    b.attn = arena.take(rows * h3.inner * f);
    b.qkv = arena.take(rows * 3 * h3.inner * f);
    b.fused = arena.take(rows * 2 * h3.ffn * f);
    b.gated = arena.take(rows * h3.ffn * f);
    b.q16 = arena.take(rows * h3.inner * 2);
    b.k16 = arena.take(rows * h3.inner * 2);
    b.v16 = arena.take(rows * h3.inner * 2);
    const int q8_width = std::max(h3.ffn, h3.inner);
    b.q8 = arena.take(rows * q8_width * 2);
    b.q8_scales = arena.take(rows * (q8_width / 32) * 4);
    b.positions = arena.take(rows * 3 * f);
    b.block_ids = arena.take(rows * 4);
    b.final_ids = arena.take(rows * 4);
    b.adaln_in = arena.take(timesteps * h3.adaln_width * f);
    b.mods = arena.take(timesteps * 6 * kH3Modalities * h3.hidden * f);
    b.final_mods = arena.take(timesteps * 2 * h3.hidden * f);
    b.caption = arena.take(text * h3.text_dim * f);
    b.patches = arena.take(rows * h3.video_channels * h3.patch * h3.patch * f);
    b.video_out = arena.take(rows * h3.video_channels * h3.patch * h3.patch * f);
    b.audio = arena.take(rows * h3.audio_channels * f);
    b.audio_out = arena.take(rows * h3.audio_channels * f);
    return b;
}

// One block over rows [first, first + rows); `modulated` blocks read the
// per-row (timestep, modality) modulation, refiner blocks run plain and
// without rope.
void h3_run_block(FlyweightV2Diffusion& d, FlyweightV2H3& h3, const DiffLaunch& launch, H3Block& block,
                  const H3Buffers& b, std::uint64_t first, std::uint64_t rows, int timesteps, bool rope) {
    const std::uint64_t f = sizeof(float);
    const int hidden = h3.hidden, inner = h3.inner;
    const std::uint64_t x = b.hidden + first * hidden * f;
    const std::uint64_t normed = b.normed + first * hidden * f;
    const std::uint64_t attn = b.attn + first * inner * f;   // [rows][inner], the attention output
    const std::uint64_t qkv = b.qkv + first * 3 * inner * f;
    const std::uint64_t fused = b.fused + first * 2 * h3.ffn * f;
    const std::uint64_t gated = b.gated + first * h3.ffn * f;
    const std::uint64_t ids = b.block_ids + first * 4;
    int count = static_cast<int>(rows);
    const int mod_stride = 6 * hidden;
    if (block.modulated) {
        launch.gemm(block.adaln, b.adaln_in, b.mods, timesteps, 0, 0);
    }
    auto modulate = [&](std::uint64_t input, std::uint64_t output, int shift_index, int scale_index) {
        int width = hidden, stride = mod_stride, shift = shift_index, scale = scale_index;
        std::uint64_t mods = b.mods, row_ids = ids;
        void* arguments[] = {&input, &mods, &row_ids, &output, &width, &count, &stride, &shift, &scale};
        launch.run("h3_modulate_rows", DiffLaunch::grid_for(rows * hidden), 1, 256, arguments);
    };
    auto gated_add = [&](std::uint64_t target, std::uint64_t update, int gate_index) {
        if (!block.modulated) {
            launch.add(target, update, rows * hidden);
            return;
        }
        int width = hidden, stride = mod_stride, gate = gate_index;
        std::uint64_t mods = b.mods, row_ids = ids;
        void* arguments[] = {&target, &update, &mods, &row_ids, &width, &count, &stride, &gate};
        launch.run("h3_gated_add_rows", DiffLaunch::grid_for(rows * hidden), 1, 256, arguments);
    };

    // Attention.
    launch.rms_norm(x, block.norm1, normed, hidden, rows, h3.eps);
    if (block.modulated) modulate(normed, normed, 0, 1);
    launch.gemm(block.qkv, normed, qkv, rows, b.q8, b.q8_scales);
    launch.head_rms_norm(qkv, block.q_norm, h3.heads, h3.head_dim, rows, 3 * inner, 0, h3.eps);
    launch.head_rms_norm(qkv, block.k_norm, h3.heads, h3.head_dim, rows, 3 * inner, inner, h3.eps);
    if (rope) {
        int axes = 3, freqs = h3.rope_freqs, heads = h3.heads, head_dim = h3.head_dim, stride = 3 * inner;
        std::uint64_t buffer = qkv, positions = b.positions + first * 3 * f, table = h3.inv_freq;
        for (int section = 0; section < 2; ++section) {
            int offset = section * inner;
            void* arguments[] = {&buffer, &positions, &table, &axes, &freqs, &heads, &head_dim, &count, &stride, &offset};
            launch.run("h3_rope_rows", count, 1, 256, arguments);
        }
    }
    launch.attention(qkv, qkv + static_cast<std::uint64_t>(inner) * f, qkv + static_cast<std::uint64_t>(2 * inner) * f,
                     attn, h3.heads, h3.heads, h3.head_dim, rows, 3 * inner, 3 * inner, false,
                     b.q16 + first * inner * 2, b.k16 + first * inner * 2, b.v16 + first * inner * 2);
    launch.gemm(block.out, attn, normed, rows, b.q8, b.q8_scales);
    gated_add(x, normed, 2);

    // Feed-forward: fused [gate | up], SiLU-gated, down.
    launch.rms_norm(x, block.norm2, normed, hidden, rows, h3.eps);
    if (block.modulated) modulate(normed, normed, 3, 4);
    launch.gemm(block.fc1, normed, fused, rows, b.q8, b.q8_scales);
    {
        int ffn = h3.ffn;
        std::uint64_t input = fused, output = gated;
        void* arguments[] = {&input, &output, &ffn, &count};
        launch.run("h3_silu_mul_fused", DiffLaunch::grid_for(rows * h3.ffn), 1, 256, arguments);
    }
    launch.gemm(block.fc2, gated, normed, rows, b.q8, b.q8_scales);
    gated_add(x, normed, 5);
}

// One transformer forward. Video latents [C][T][H][W], audio latents
// [channels][audio_latents][audio_channels] (row per latent, channel-major),
// caption [text][text_dim]. Writes the raw video velocity in latent layout and
// the audio velocity in the audio layout.
void h3_forward(FlyweightV2Diffusion& d, FlyweightV2H3& h3, const float* video, int latent_t, int latent_h, int latent_w,
                const float* audio, int audio_latents, const float* caption, int text, float t_video, float t_audio,
                float* video_velocity, float* audio_velocity) {
    if (latent_h % h3.patch || latent_w % h3.patch) throw std::runtime_error("latent sides must be multiples of the patch");
    if (text <= 0) throw std::runtime_error("empty caption");
    const H3Layout layout = h3_layout(h3, text, latent_t, latent_h, latent_w, audio_latents, t_video, t_audio);
    const int timesteps = static_cast<int>(layout.timesteps.size());
    const std::uint64_t rows = layout.rows;
    const std::uint64_t f = sizeof(float);
    if (h3_workspace_bytes(h3, rows, text, timesteps) > d.workspace_bytes)
        throw std::runtime_error("clip is larger than the H3 tower was created for");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const H3Buffers b = h3_buffers(h3, arena, rows, text, timesteps);
    const int features = h3.video_channels * h3.patch * h3.patch;

    // Host staging, alive until the downloads at the end.
    std::vector<float> adaln_in(static_cast<std::size_t>(timesteps) * h3.adaln_width);
    for (int i = 0; i < timesteps; ++i) h3_adaln_input(h3, layout.timesteps[i], adaln_in.data() + i * h3.adaln_width);
    std::vector<float> patches;
    h3_patchify(h3, video, latent_t, latent_h, latent_w, patches);
    const std::size_t audio_values = static_cast<std::size_t>(layout.audio_rows) * h3.audio_channels;
    launch.upload(b.adaln_in, adaln_in.data(), adaln_in.size() * f);
    launch.upload(b.positions, layout.positions.data(), layout.positions.size() * f);
    launch.upload(b.block_ids, layout.block_ids.data(), layout.block_ids.size() * 4);
    launch.upload(b.final_ids, layout.final_ids.data(), layout.final_ids.size() * 4);
    launch.upload(b.caption, caption, static_cast<std::uint64_t>(text) * h3.text_dim * f);
    launch.upload(b.patches, patches.data(), patches.size() * f);
    if (audio_values) launch.upload(b.audio, audio, audio_values * f);

    // Embeddings into their rows: text through condition_proj and the token
    // refiner, audio and video through their patch projections.
    const std::uint64_t text_rows = b.hidden;
    const std::uint64_t audio_rows = b.hidden + static_cast<std::uint64_t>(layout.audio_start) * h3.hidden * f;
    const std::uint64_t video_rows = b.hidden + static_cast<std::uint64_t>(layout.video_start) * h3.hidden * f;
    launch.gemm(h3.condition, b.caption, text_rows, text, b.q8, b.q8_scales);
    if (layout.audio_rows) launch.gemm(h3.audio_in, b.audio, audio_rows, layout.audio_rows, b.q8, b.q8_scales);
    launch.gemm(h3.video_in, b.patches, video_rows, layout.video_rows, b.q8, b.q8_scales);

    // Refine the text rows (no rope, no modulation), then the final norm.
    {
        DiffStreamer streamer{d, {}, 0};
        for (auto& block : h3.refiner) streamer.refs.push_back(h3_block_matrices(block));
        for (std::size_t index = 0; index < h3.refiner.size(); ++index) {
            streamer.begin(index);
            h3_run_block(d, h3, launch, h3.refiner[index], b, 0, text, timesteps, false);
            streamer.end(index);
        }
        launch.rms_norm(text_rows, h3.refiner_norm, text_rows, h3.hidden, text, h3.eps);
    }
    // The stack over the packed sequence.
    {
        DiffStreamer streamer{d, {}, 0};
        for (auto& block : h3.blocks) streamer.refs.push_back(h3_block_matrices(block));
        for (std::size_t index = 0; index < h3.blocks.size(); ++index) {
            streamer.begin(index);
            h3_run_block(d, h3, launch, h3.blocks[index], b, 0, rows, timesteps, true);
            streamer.end(index);
        }
    }
    // Final layer: norm, per-timestep shift/scale, the two heads.
    launch.gemm(h3.final_adaln, b.adaln_in, b.final_mods, timesteps, 0, 0);
    launch.rms_norm(b.hidden, h3.final_norm, b.normed, h3.hidden, rows, h3.eps);
    {
        int width = h3.hidden, count = static_cast<int>(rows), stride = 2 * h3.hidden, shift = 0, scale = 1;
        std::uint64_t input = b.normed, mods = b.final_mods, row_ids = b.final_ids, output = b.normed;
        void* arguments[] = {&input, &mods, &row_ids, &output, &width, &count, &stride, &shift, &scale};
        launch.run("h3_modulate_rows", DiffLaunch::grid_for(rows * h3.hidden), 1, 256, arguments);
    }
    const std::uint64_t normed_video = b.normed + static_cast<std::uint64_t>(layout.video_start) * h3.hidden * f;
    const std::uint64_t normed_audio = b.normed + static_cast<std::uint64_t>(layout.audio_start) * h3.hidden * f;
    launch.gemm(h3.video_out, normed_video, b.video_out, layout.video_rows, 0, 0);
    if (layout.audio_rows) launch.gemm(h3.audio_out, normed_audio, b.audio_out, layout.audio_rows, 0, 0);
    std::vector<float> out_rows(static_cast<std::size_t>(layout.video_rows) * features);
    launch.download(out_rows.data(), b.video_out, out_rows.size() * f);
    h3_unpatchify(h3, out_rows, latent_t, latent_h, latent_w, video_velocity);
    if (audio_values) launch.download(audio_velocity, b.audio_out, audio_values * f);
}

}  // namespace

namespace {

// Frames -> latent frames for an aligned `17n + 5` count: `5n + 2`.
int h3_latent_frames(int frames) {
    if (frames < 5 || (frames - 5) % 17) throw std::runtime_error("frame count must be 17n + 5");
    return (frames - 5) / 17 * 5 + 2;
}

int h3_audio_latents(int frames) {
    return static_cast<int>(std::lround(frames / 24.0 * 40.0));
}

}  // namespace

void h3_attach(FlyweightV2Diffusion& d) {
    auto* h3 = new FlyweightV2H3();
    d.h3 = h3;
    d.release_extra = [](FlyweightV2Diffusion& tower) {
        delete tower.h3;
        tower.h3 = nullptr;
    };
    h3_upload_transformer(d, *h3);
}

void h3_attach_vae(FlyweightV2Diffusion& d) {
    if (d.host_weights) diff_register_model(d, *d.vae);
    h3_upload_vae(d, *d.h3);
}

constexpr int kH3VaeTilePixels = 256;
constexpr int kH3VaeTileOverlapPixels = 64;

std::uint64_t h3_workspace_for(const FlyweightV2Diffusion& d, std::uint32_t text, std::uint32_t width,
                               std::uint32_t height, std::uint32_t frames) {
    const auto& h3 = *d.h3;
    std::uint64_t vae_bytes = 0;
    if (h3.has_vae) {
        const int chunk = (h3.vae_clip_length + h3.vae_temporal - 1) / h3.vae_temporal;
        const int token_overlap = (-h3.vae_token_drop % chunk + chunk) % chunk;
        // The decoder runs in tiles of at most kH3VaeTilePixels a side.
        const std::uint64_t tile_h = std::min<std::uint64_t>(height, kH3VaeTilePixels) / h3.vae_spatial;
        const std::uint64_t tile_w = std::min<std::uint64_t>(width, kH3VaeTilePixels) / h3.vae_spatial;
        const std::uint64_t voxels = static_cast<std::uint64_t>(chunk + token_overlap) * tile_h * tile_w;
        vae_bytes = h3_vae_workspace_bytes(h3, voxels + h3.vae_registers + 1, voxels);
    }
    int aligned = static_cast<int>(frames);
    while (aligned < 5 || (aligned - 5) % 17) ++aligned;
    const int latent_t = h3_latent_frames(aligned);
    const int rows_per_frame = (static_cast<int>(height) / 16 / h3.patch) * (static_cast<int>(width) / 16 / h3.patch);
    const std::uint64_t rows = text + static_cast<std::uint64_t>(kH3AudioChannels) * h3_audio_latents(aligned) +
        static_cast<std::uint64_t>(latent_t) * rows_per_frame;
    return std::max(h3_workspace_bytes(h3, rows, text, 2), vae_bytes);
}

int flyweight_v2_h3_step(FlyweightV2Diffusion* d, const float* video, uint32_t latent_t, uint32_t latent_h, uint32_t latent_w,
                         const float* audio, uint32_t audio_latents, const float* caption, uint32_t text,
                         float t_video, float t_audio, float* video_velocity, float* audio_velocity) { return guarded([&]{
    if (!d || !d->h3 || !video || !caption || !video_velocity) throw std::runtime_error("invalid H3 step arguments");
    if (audio_latents && (!audio || !audio_velocity)) throw std::runtime_error("audio rows need audio buffers");
    h3_forward(*d, *d->h3, video, static_cast<int>(latent_t), static_cast<int>(latent_h), static_cast<int>(latent_w),
               audio, static_cast<int>(audio_latents), caption, static_cast<int>(text), t_video, t_audio,
               video_velocity, audio_velocity);
    return 0;
}); }

// ---- The ViT video VAE decoder ---------------------------------------------

namespace {

DiffLayerRef h3_vae_block_matrices(H3VaeBlock& block) {
    DiffLayerRef ref;
    ref.add(block.qkv);
    ref.add(block.out);
    ref.add(block.w1);
    ref.add(block.w2);
    return ref;
}

void h3_upload_vae(FlyweightV2Diffusion& d, FlyweightV2H3& h3) {
    const auto& model = *d.vae;
    const auto& config = model.config;
    h3.vae_dim = static_cast<int>(config.hidden_size);
    h3.vae_heads = static_cast<int>(config.attention_heads);
    h3.vae_head_dim = static_cast<int>(config.attention_head_dim);
    h3.vae_ffn = static_cast<int>(config.intermediate_size);
    h3.vae_layers = static_cast<int>(config.layer_count);
    h3.vae_registers = static_cast<int>(config.vae_register_tokens);
    h3.vae_latent_channels = static_cast<int>(config.latent_channels);
    h3.vae_spatial = static_cast<int>(config.patch_size);
    h3.vae_temporal = static_cast<int>(config.vae_temporal_patch);
    h3.vae_clip_length = static_cast<int>(config.vae_clip_length);
    h3.vae_token_drop = static_cast<int>(config.vae_token_drop);
    h3.vae_rope_freqs = static_cast<int>(config.rope_freq_dim);
    h3.vae_eps = config.rms_norm_epsilon;
    h3.vae_mean = config.vae_latents_mean;
    h3.vae_std = config.vae_latents_std;
    if (h3.vae_latent_channels != h3.video_channels)
        throw std::runtime_error("the VAE's latent channels do not match the DiT's");
    const int dim = h3.vae_dim;
    h3.vae_embed = diff_upload_matrix(d, model, "decoder.x_embedder.weight", h3.vae_latent_channels, dim, "decoder.x_embedder.bias");
    h3.vae_post_quant = diff_upload_matrix(d, model, "post_quant_conv.weight", h3.vae_latent_channels, h3.vae_latent_channels,
                                           "post_quant_conv.bias");
    h3.vae_register = diff_upload_f32(d, model, "decoder.register_tokens", static_cast<std::uint64_t>(h3.vae_registers) * dim);
    h3.vae_norm_out = diff_upload_f32(d, model, "decoder.norm_out.weight", dim);
    h3.vae_norm_out_bias = diff_upload_f32(d, model, "decoder.norm_out.bias", dim);
    const int pixels = 3 * h3.vae_temporal * h3.vae_spatial * h3.vae_spatial;
    h3.vae_out = diff_upload_matrix(d, model, "decoder.proj_out.weight", dim, pixels, "decoder.proj_out.bias");
    {
        std::vector<float> ones(h3.vae_head_dim, 1.0f);
        h3.vae_ones = diff_alloc(d, ones.size() * sizeof(float));
        flyweight_gpu_upload_sync(h3.vae_ones, ones.data(), ones.size() * sizeof(float));
        // inv_freq[i] = theta^(-i / freqs), one ramp shared by the three axes.
        std::vector<float> inv_freq(h3.vae_rope_freqs);
        for (int i = 0; i < h3.vae_rope_freqs; ++i)
            inv_freq[i] = static_cast<float>(std::pow(static_cast<double>(config.rope_freq_base), -static_cast<double>(i) / h3.vae_rope_freqs));
        h3.vae_inv_freq = diff_alloc(d, inv_freq.size() * sizeof(float));
        flyweight_gpu_upload_sync(h3.vae_inv_freq, inv_freq.data(), inv_freq.size() * sizeof(float));
    }
    for (int index = 0; index < h3.vae_layers; ++index) {
        const std::string prefix = "decoder.transformer_blocks." + std::to_string(index) + ".";
        DiffGroup group(d);
        H3VaeBlock block;
        block.norm1 = diff_upload_f32(d, model, prefix + "norm1.weight", dim);
        block.norm2 = diff_upload_f32(d, model, prefix + "norm2.weight", dim);
        block.scale1 = diff_upload_f32(d, model, prefix + "scale1", dim);
        block.scale2 = diff_upload_f32(d, model, prefix + "scale2", dim);
        block.qkv = diff_upload_matrix(d, model, prefix + "attn.to_qkv.weight", dim, 3 * dim, prefix + "attn.to_qkv.bias");
        block.out = diff_upload_matrix(d, model, prefix + "attn.to_out.weight", dim, dim, prefix + "attn.to_out.bias");
        block.w1 = diff_upload_matrix(d, model, prefix + "ff.w1.weight", dim, 2 * h3.vae_ffn, prefix + "ff.w1.bias");
        block.w2 = diff_upload_matrix(d, model, prefix + "ff.w2.weight", h3.vae_ffn, dim, prefix + "ff.w2.bias");
        h3.vae_blocks.push_back(block);
    }
    h3.has_vae = true;
}

std::uint64_t h3_vae_workspace_bytes(const FlyweightV2H3& h3, std::uint64_t tokens, std::uint64_t voxels) {
    const std::uint64_t f = sizeof(float);
    const int dim = h3.vae_dim;
    const int pixels = 3 * h3.vae_temporal * h3.vae_spatial * h3.vae_spatial;
    std::uint64_t total = 0;
    total += diff_align(voxels * h3.vae_latent_channels * f) * 2;   // latents in, post-quant
    total += diff_align(tokens * dim * f) * 4;                     // hidden, normed, attn, projected
    total += diff_align(tokens * 3 * dim * f);                     // qkv
    total += diff_align(tokens * 2 * h3.vae_ffn * f);              // fused gate|up
    total += diff_align(tokens * h3.vae_ffn * f);                  // gated
    total += diff_align(tokens * dim * 2) * 3;                     // bf16 planes
    total += diff_q8_bytes(tokens, std::max(h3.vae_ffn, dim));
    total += diff_align(tokens * 3 * f);                           // positions
    total += diff_align(voxels * pixels * f);                      // decoded rows
    return total;
}

// Decodes one clip of `frames` latent frames: z [C][frames][H][W]
// (already scaled) -> pixels [3][frames*temporal][H*spatial][W*spatial] f32.
void h3_vae_decode_clip(FlyweightV2Diffusion& d, FlyweightV2H3& h3, const float* z, int frames, int latent_h, int latent_w,
                        float* pixels) {
    const std::uint64_t f = sizeof(float);
    const int dim = h3.vae_dim, channels = h3.vae_latent_channels;
    const std::uint64_t voxels = static_cast<std::uint64_t>(frames) * latent_h * latent_w;
    const std::uint64_t tokens = voxels + h3.vae_registers + 1;
    const int pixels_per_token = 3 * h3.vae_temporal * h3.vae_spatial * h3.vae_spatial;
    if (h3_vae_workspace_bytes(h3, tokens, voxels) > d.workspace_bytes)
        throw std::runtime_error("clip is larger than the VAE workspace");
    DiffLaunch launch{d};
    DiffArena arena{d.workspace, 0, d.workspace_bytes};
    const std::uint64_t d_latents = arena.take(voxels * channels * f);
    const std::uint64_t d_post = arena.take(voxels * channels * f);
    const std::uint64_t d_hidden = arena.take(tokens * dim * f);
    const std::uint64_t d_normed = arena.take(tokens * dim * f);
    const std::uint64_t d_attn = arena.take(tokens * dim * f);
    const std::uint64_t d_proj = arena.take(tokens * dim * f);
    const std::uint64_t d_qkv = arena.take(tokens * 3 * dim * f);
    const std::uint64_t d_fused = arena.take(tokens * 2 * h3.vae_ffn * f);
    const std::uint64_t d_gated = arena.take(tokens * h3.vae_ffn * f);
    const std::uint64_t d_q16 = arena.take(tokens * dim * 2), d_k16 = arena.take(tokens * dim * 2), d_v16 = arena.take(tokens * dim * 2);
    const int q8_width = std::max(h3.vae_ffn, dim);
    const std::uint64_t d_q8 = arena.take(tokens * q8_width * 2), d_q8_scales = arena.take(tokens * (q8_width / 32) * 4);
    const std::uint64_t d_positions = arena.take(tokens * 3 * f);
    const std::uint64_t d_rows = arena.take(voxels * pixels_per_token * f);

    // Voxel rows (t, h, w) of channels, and their rope positions in 2*pi units
    // of the length-normalized [-1, 1) grid; registers and the zero token at 0.
    std::vector<float> latent_rows(voxels * channels), positions(tokens * 3, 0.0f);
    const float two_pi = 6.283185307179586f;
    for (int t = 0; t < frames; ++t)
        for (int y = 0; y < latent_h; ++y)
            for (int x = 0; x < latent_w; ++x) {
                const std::uint64_t row = (static_cast<std::uint64_t>(t) * latent_h + y) * latent_w + x;
                for (int c = 0; c < channels; ++c)
                    latent_rows[row * channels + c] = z[((static_cast<std::uint64_t>(c) * frames + t) * latent_h + y) * latent_w + x];
                positions[row * 3] = (2.0f * ((t + 0.5f) / frames) - 1.0f) * two_pi;
                positions[row * 3 + 1] = (2.0f * ((y + 0.5f) / latent_h) - 1.0f) * two_pi;
                positions[row * 3 + 2] = (2.0f * ((x + 0.5f) / latent_w) - 1.0f) * two_pi;
            }
    launch.upload(d_latents, latent_rows.data(), latent_rows.size() * f);
    launch.upload(d_positions, positions.data(), positions.size() * f);
    launch.gemm(h3.vae_post_quant, d_latents, d_post, voxels, 0, 0);
    launch.gemm(h3.vae_embed, d_post, d_hidden, voxels, d_q8, d_q8_scales);
    if (flyweight_gpu_copy_device(d_hidden + voxels * dim * f, h3.vae_register,
                                  static_cast<std::uint64_t>(h3.vae_registers) * dim * f, d.stream) != 0 ||
        flyweight_gpu_memset(d_hidden + (voxels + h3.vae_registers) * dim * f, 0, dim * f, d.stream) != 0)
        throw std::runtime_error("VAE token staging failed");

    DiffStreamer streamer{d, {}, 0};
    for (auto& block : h3.vae_blocks) streamer.refs.push_back(h3_vae_block_matrices(block));
    int count = static_cast<int>(tokens);
    for (std::size_t index = 0; index < h3.vae_blocks.size(); ++index) {
        auto& block = h3.vae_blocks[index];
        streamer.begin(index);
        launch.rms_norm(d_hidden, block.norm1, d_normed, dim, tokens, h3.vae_eps);
        launch.gemm(block.qkv, d_normed, d_qkv, tokens, d_q8, d_q8_scales);
        launch.head_rms_norm(d_qkv, h3.vae_ones, h3.vae_heads, h3.vae_head_dim, tokens, 3 * dim, 0, h3.vae_eps);
        launch.head_rms_norm(d_qkv, h3.vae_ones, h3.vae_heads, h3.vae_head_dim, tokens, 3 * dim, dim, h3.vae_eps);
        {
            int axes = 3, freqs = h3.vae_rope_freqs, heads = h3.vae_heads, head_dim = h3.vae_head_dim, stride = 3 * dim;
            std::uint64_t buffer = d_qkv, table = h3.vae_inv_freq, pos = d_positions;
            for (int section = 0; section < 2; ++section) {
                int offset = section * dim;
                void* arguments[] = {&buffer, &pos, &table, &axes, &freqs, &heads, &head_dim, &count, &stride, &offset};
                launch.run("h3_rope_rows", count, 1, 256, arguments);
            }
        }
        launch.attention(d_qkv, d_qkv + static_cast<std::uint64_t>(dim) * f, d_qkv + static_cast<std::uint64_t>(2 * dim) * f,
                         d_attn, h3.vae_heads, h3.vae_heads, h3.vae_head_dim, tokens, 3 * dim, 3 * dim, false,
                         d_q16, d_k16, d_v16);
        launch.gemm(block.out, d_attn, d_proj, tokens, d_q8, d_q8_scales);
        launch.gated_add(d_hidden, d_proj, block.scale1, dim, tokens);
        launch.rms_norm(d_hidden, block.norm2, d_normed, dim, tokens, h3.vae_eps);
        launch.gemm(block.w1, d_normed, d_fused, tokens, d_q8, d_q8_scales);
        {
            int ffn = h3.vae_ffn;
            std::uint64_t input = d_fused, output = d_gated;
            void* arguments[] = {&input, &output, &ffn, &count};
            launch.run("h3_silu_mul_fused", DiffLaunch::grid_for(tokens * h3.vae_ffn), 1, 256, arguments);
        }
        launch.gemm(block.w2, d_gated, d_proj, tokens, d_q8, d_q8_scales);
        launch.gated_add(d_hidden, d_proj, block.scale2, dim, tokens);
        streamer.end(index);
    }
    {
        int width = dim;
        std::uint64_t input = d_hidden, weight = h3.vae_norm_out, bias = h3.vae_norm_out_bias, output = d_normed;
        float epsilon = h3.vae_eps;
        void* arguments[] = {&input, &weight, &bias, &output, &width, &count, &epsilon};
        launch.run("vision_layer_norm_rows", count, 1, 256, arguments);
    }
    launch.gemm(h3.vae_out, d_normed, d_rows, voxels, d_q8, d_q8_scales);
    std::vector<float> rows(voxels * pixels_per_token);
    launch.download(rows.data(), d_rows, rows.size() * f);
    // Rows (t, y, x) of (c, pt, ph, pw) -> [3][frames*pt][H*ps][W*ps].
    const int pt = h3.vae_temporal, ps = h3.vae_spatial;
    const int out_h = latent_h * ps, out_w = latent_w * ps, out_t = frames * pt;
    for (int t = 0; t < frames; ++t)
        for (int y = 0; y < latent_h; ++y)
            for (int x = 0; x < latent_w; ++x) {
                const float* row = rows.data() + ((static_cast<std::uint64_t>(t) * latent_h + y) * latent_w + x) * pixels_per_token;
                for (int c = 0; c < 3; ++c)
                    for (int dt = 0; dt < pt; ++dt)
                        for (int dy = 0; dy < ps; ++dy)
                            std::memcpy(pixels + ((static_cast<std::uint64_t>(c) * out_t + t * pt + dt) * out_h + y * ps + dy) * out_w + x * ps,
                                        row + ((c * pt + dt) * ps + dy) * ps, ps * sizeof(float));
            }
}

// diffusers decodes in 256-pixel spatial tiles overlapping by at least 64
// (`use_tiling` is on by default and the pipeline keeps it): the ViT decoder
// was run that way, and attending over a whole wider frame drifts visibly.

// diffusers' `_split_tiles`: the fewest `tile`-wide tiles whose union covers
// `length` with every overlap at least `min_overlap`; the slack goes round
// robin onto the overlaps in whole `ratio` steps so boundaries stay
// latent-aligned. One tile when `tile >= length`.
void h3_vae_split_tiles(int length, int tile, int min_overlap, int ratio, std::vector<int>& starts,
                        std::vector<int>& overlaps) {
    starts.clear();
    overlaps.clear();
    starts.push_back(0);
    if (tile >= length) return;
    int num = (length + tile - 1) / tile;
    while (tile * num - min_overlap * (num - 1) - length < 0) ++num;
    overlaps.assign(num - 1, min_overlap);
    const int remaining = tile * num - min_overlap * (num - 1) - length;
    for (int i = 0; i < remaining / ratio; ++i) overlaps[i % (num - 1)] += ratio;
    for (int i = 0; i < num - 1; ++i) starts.push_back(starts.back() + tile - overlaps[i]);
}

// One clip through the spatial tiling and diffusers' `_stitch_tiles`: every
// tile is decoded raw, then blended linearly with the raw tile above and the
// raw tile to its left over the overlaps and cropped by its own overlaps.
void h3_vae_decode_clip_tiled(FlyweightV2Diffusion& d, FlyweightV2H3& h3, const float* z, int frames, int latent_h,
                              int latent_w, float* pixels) {
    const int ps = h3.vae_spatial, pt = h3.vae_temporal, channels = h3.vae_latent_channels;
    const int tile_latents = kH3VaeTilePixels / ps;
    if (latent_h <= tile_latents && latent_w <= tile_latents) {
        h3_vae_decode_clip(d, h3, z, frames, latent_h, latent_w, pixels);
        return;
    }
    std::vector<int> starts_y, overlaps_y, starts_x, overlaps_x;
    h3_vae_split_tiles(latent_h * ps, kH3VaeTilePixels, kH3VaeTileOverlapPixels, ps, starts_y, overlaps_y);
    h3_vae_split_tiles(latent_w * ps, kH3VaeTilePixels, kH3VaeTileOverlapPixels, ps, starts_x, overlaps_x);
    const int ty = std::min(latent_h, tile_latents), tx = std::min(latent_w, tile_latents);   // tile size in latents
    const int th = ty * ps, tw = tx * ps, out_t = frames * pt;
    const int out_h = latent_h * ps, out_w = latent_w * ps;
    const std::uint64_t tile_values = static_cast<std::uint64_t>(3) * out_t * th * tw;
    const std::size_t rows = starts_y.size(), cols = starts_x.size();
    std::vector<std::vector<float>> tiles(rows * cols);
    std::vector<float> tile_latents_buffer(static_cast<std::uint64_t>(channels) * frames * ty * tx);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            const int y0 = starts_y[i] / ps, x0 = starts_x[j] / ps;
            for (int c = 0; c < channels; ++c)
                for (int t = 0; t < frames; ++t)
                    for (int y = 0; y < ty; ++y)
                        std::memcpy(tile_latents_buffer.data() + ((static_cast<std::uint64_t>(c) * frames + t) * ty + y) * tx,
                                    z + ((static_cast<std::uint64_t>(c) * frames + t) * latent_h + y0 + y) * latent_w + x0,
                                    static_cast<std::size_t>(tx) * sizeof(float));
            tiles[i * cols + j].assign(tile_values, 0.0f);
            h3_vae_decode_clip(d, h3, tile_latents_buffer.data(), frames, ty, tx, tiles[i * cols + j].data());
        }
    auto at = [&](const std::vector<float>& tile, int c, int t, int y, int x) {
        return tile[((static_cast<std::uint64_t>(c) * out_t + t) * th + y) * tw + x];
    };
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            const std::vector<float>& tile = tiles[i * cols + j];
            const std::vector<float>* above = i > 0 ? &tiles[(i - 1) * cols + j] : nullptr;
            const std::vector<float>* left = j > 0 ? &tiles[i * cols + j - 1] : nullptr;
            const int oy = i > 0 ? overlaps_y[i - 1] : 0, ox = j > 0 ? overlaps_x[j - 1] : 0;
            const int keep_h = th - (i + 1 < rows ? overlaps_y[i] : 0);
            const int keep_w = tw - (j + 1 < cols ? overlaps_x[j] : 0);
            for (int c = 0; c < 3; ++c)
                for (int t = 0; t < out_t; ++t)
                    for (int y = 0; y < keep_h; ++y) {
                        float* out = pixels + ((static_cast<std::uint64_t>(c) * out_t + t) * out_h + starts_y[i] + y) * out_w + starts_x[j];
                        for (int x = 0; x < keep_w; ++x) {
                            float value = at(tile, c, t, y, x);
                            if (above && y < oy) {
                                const float w = static_cast<float>(y) / oy;
                                value = at(*above, c, t, th - oy + y, x) * (1.0f - w) + value * w;
                            }
                            if (left && x < ox) {
                                const float w = static_cast<float>(x) / ox;
                                value = at(*left, c, t, y, tw - ox + x) * (1.0f - w) + value * w;
                            }
                            out[x] = value;
                        }
                    }
        }
}

// The whole latent video, chunked as diffusers' `_decode`: every clip holds
// tokens_chunk_size + overlap latent frames, each decoded clip drops its first
// `pre_padding` frames, and consecutive clips cross-fade over `frame_overlap`.
// Latents arrive in the DiT's normalized space; the mean/std are undone here.
// Output: frames [F][H*16][W*16][3] uint8 with F the pixel frame count.
int h3_vae_decode(FlyweightV2Diffusion& d, FlyweightV2H3& h3, const float* latents, int latent_t, int latent_h, int latent_w,
                  std::uint8_t* rgb, int frames_capacity) {
    const int channels = h3.vae_latent_channels;
    const int temporal = h3.vae_temporal, ps = h3.vae_spatial;
    const int chunk = (h3.vae_clip_length + temporal - 1) / temporal;          // tokens per chunk (5)
    const int token_overlap = (-h3.vae_token_drop % chunk + chunk) % chunk;   // 2
    const int pre_padding = (-h3.vae_clip_length % temporal + temporal) % temporal;  // 3
    const int frame_overlap = std::max(token_overlap * temporal - pre_padding, 0);   // 5
    const int chunk_frames = chunk * temporal;                                 // 20
    const std::uint64_t voxels = static_cast<std::uint64_t>(latent_t) * latent_h * latent_w;
    // Undo the normalization: z = latents * std + mean per channel.
    std::vector<float> z(voxels * channels);
    for (int c = 0; c < channels; ++c) {
        const float mean = c < static_cast<int>(h3.vae_mean.size()) ? h3.vae_mean[c] : 0.0f;
        const float std = c < static_cast<int>(h3.vae_std.size()) ? h3.vae_std[c] : 1.0f;
        for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(latent_t) * latent_h * latent_w; ++i)
            z[c * latent_t * latent_h * latent_w + i] = latents[c * latent_t * latent_h * latent_w + i] * std + mean;
    }
    // Pad the token axis to whole chunks by repeating the last latent frame.
    const int num_tokens = latent_t + h3.vae_token_drop;
    int pad_tokens = (chunk - num_tokens % chunk) % chunk;
    int num_chunks = (num_tokens + pad_tokens) / chunk - (h3.vae_token_drop > 0 ? 1 : 0);
    if (num_chunks < 1) {
        pad_tokens += chunk;
        num_chunks += 1;
    }
    const int padded_t = latent_t + pad_tokens;
    std::vector<float> zp(static_cast<std::uint64_t>(padded_t) * latent_h * latent_w * channels);
    const std::uint64_t plane = static_cast<std::uint64_t>(latent_h) * latent_w;
    for (int c = 0; c < channels; ++c)
        for (int t = 0; t < padded_t; ++t)
            std::memcpy(zp.data() + (static_cast<std::uint64_t>(c) * padded_t + t) * plane,
                        z.data() + (static_cast<std::uint64_t>(c) * latent_t + std::min(t, latent_t - 1)) * plane, plane * sizeof(float));
    const int out_h = latent_h * ps, out_w = latent_w * ps;
    const std::uint64_t frame_values = static_cast<std::uint64_t>(3) * out_h * out_w;
    std::vector<float> decoded;   // [F][3][H][W] as it accumulates
    std::vector<float> overlap;   // the tail of the previous clip, [n][3][H][W]
    int decoded_frames = 0, overlap_frames = 0;
    std::vector<float> clip_latents, clip_pixels;
    auto frame_ptr = [&](std::vector<float>& buffer, int index) { return buffer.data() + static_cast<std::uint64_t>(index) * frame_values; };
    for (int i = 0; i < num_chunks; ++i) {
        const int start = i * chunk;
        const int count = std::min(chunk + token_overlap, padded_t - start);
        clip_latents.assign(static_cast<std::uint64_t>(count) * plane * channels, 0.0f);
        for (int c = 0; c < channels; ++c)
            for (int t = 0; t < count; ++t)
                std::memcpy(clip_latents.data() + (static_cast<std::uint64_t>(c) * count + t) * plane,
                            zp.data() + (static_cast<std::uint64_t>(c) * padded_t + start + t) * plane, plane * sizeof(float));
        const int clip_frames = count * temporal;
        clip_pixels.assign(static_cast<std::uint64_t>(clip_frames) * frame_values, 0.0f);
        // The decoder writes [3][F][H][W]; keep that and index frames through channels.
        h3_vae_decode_clip_tiled(d, h3, clip_latents.data(), count, latent_h, latent_w, clip_pixels.data());
        auto clip_frame = [&](int t, int c) { return clip_pixels.data() + (static_cast<std::uint64_t>(c) * clip_frames + t) * out_h * out_w; };
        // Segment j = 0: this chunk's frames; j = 1: the overlap tail for the next.
        for (int j = 0; j < (h3.vae_token_drop > 0 ? 2 : 1); ++j) {
            const int frame_start = j * chunk_frames;
            const int available = std::max(0, std::min(chunk_frames, clip_frames - frame_start) - pre_padding);
            if (available <= 0) continue;
            std::vector<float> segment(static_cast<std::uint64_t>(available) * frame_values);
            for (int t = 0; t < available; ++t)
                for (int c = 0; c < 3; ++c)
                    std::memcpy(frame_ptr(segment, t) + static_cast<std::uint64_t>(c) * out_h * out_w,
                                clip_frame(frame_start + pre_padding + t, c), static_cast<std::uint64_t>(out_h) * out_w * sizeof(float));
            if (j == 0) {
                if (overlap_frames) {
                    // Linear cross-fade of the first `frame_overlap` frames with the previous tail.
                    const int blend = std::min({frame_overlap, overlap_frames, available});
                    for (int t = 0; t < blend; ++t) {
                        const float weight = static_cast<float>(t) / blend;
                        float* target = frame_ptr(segment, t);
                        const float* previous = frame_ptr(overlap, overlap_frames - blend + t);
                        for (std::uint64_t v = 0; v < frame_values; ++v) target[v] = previous[v] * (1.0f - weight) + target[v] * weight;
                    }
                }
                decoded.insert(decoded.end(), segment.begin(), segment.end());
                decoded_frames += available;
            } else {
                overlap = std::move(segment);
                overlap_frames = available;
            }
        }
    }
    if (overlap_frames) {
        decoded.insert(decoded.end(), overlap.begin(), overlap.end());
        decoded_frames += overlap_frames;
    }
    // Cut the frames the padding added.
    if (pad_tokens > 0) {
        const int intra_tail = h3.vae_clip_length % temporal;
        const int before_pad = latent_t;
        int pad_frames = 0;
        for (int k = 0; k < pad_tokens; ++k)
            pad_frames += (intra_tail && (before_pad + k) % chunk == 0) ? intra_tail : temporal;
        decoded_frames = std::max(0, decoded_frames - pad_frames);
    }
    if (decoded_frames > frames_capacity) throw std::runtime_error("decoded more frames than the output holds");
    for (int t = 0; t < decoded_frames; ++t) {
        const float* frame = frame_ptr(decoded, t);
        for (std::uint64_t p = 0; p < static_cast<std::uint64_t>(out_h) * out_w; ++p)
            for (int c = 0; c < 3; ++c) {
                const float value = (frame[static_cast<std::uint64_t>(c) * out_h * out_w + p] * 0.5f + 0.5f) * 255.0f;
                rgb[(static_cast<std::uint64_t>(t) * out_h * out_w + p) * 3 + c] =
                    static_cast<std::uint8_t>(std::min(255.0f, std::max(0.0f, std::round(value))));
            }
    }
    return decoded_frames;
}

}  // namespace

int flyweight_v2_h3_decode(FlyweightV2Diffusion* d, const float* latents, uint32_t latent_t, uint32_t latent_h, uint32_t latent_w,
                           uint8_t* rgb, uint32_t frames_capacity, uint32_t* frames_out) { return guarded([&]{
    if (!d || !d->h3 || !d->h3->has_vae || !latents || !rgb || !frames_out) throw std::runtime_error("invalid H3 decode arguments");
    *frames_out = static_cast<uint32_t>(h3_vae_decode(*d, *d->h3, latents, static_cast<int>(latent_t), static_cast<int>(latent_h),
                                                      static_cast<int>(latent_w), rgb, static_cast<int>(frames_capacity)));
    return 0;
}); }
