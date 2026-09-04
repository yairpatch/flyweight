#pragma once
// Vision tower geometry and the host-side preparation an image needs before
// the CUDA tower sees it. Everything here is pure arithmetic over the mmproj
// GGUF's `clip.*` keys, kept header-only so the parity check can exercise it
// without a GPU.
//
// The tower is llama.cpp's `qwen3vl_merger` projector: a SigLIP-style ViT
// (pre-LN blocks with biases, GELU MLP, 2D rope on q/k, full attention over
// every patch) whose output is merged 2x2 into the language model's width.
// Patches are ordered window-major -- the four patches of each merge window
// are consecutive -- which is what lets the merger see a window as one row
// of `4 * embedding_length` without a gather.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace flyweight::v2 {

struct VisionConfig {
    bool present = false;
    std::string projector_type;
    std::uint32_t image_size = 0;          // side of the learned position grid, in pixels
    std::uint32_t patch_size = 0;
    std::uint32_t embedding_length = 0;    // ViT width
    std::uint32_t feed_forward_length = 0;
    std::uint32_t block_count = 0;
    std::uint32_t head_count = 0;
    std::uint32_t spatial_merge_size = 2;
    std::uint32_t projection_dim = 0;      // language model hidden size
    float layer_norm_epsilon = 1.0e-6f;
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3] = {0.5f, 0.5f, 0.5f};
    bool use_gelu = false;
    // ViT block indexes whose output also feeds the language model's first
    // blocks (Qwen3-VL "deepstack"). Empty on checkpoints without it.
    std::vector<std::uint32_t> deepstack_layers;

    std::uint32_t head_dim() const {
        return head_count ? embedding_length / head_count : 0;
    }
    // Pixels covered by one merged token along each axis.
    std::uint32_t token_side() const { return patch_size * spatial_merge_size; }
    std::uint32_t position_grid_side() const {
        return patch_size ? image_size / patch_size : 0;
    }
};

// Fail when a key the tower depends on is missing or inconsistent, before any
// weight is touched. The message names the key so a foreign mmproj is
// diagnosable from the error alone.
inline void validate_vision_config(const VisionConfig& config) {
    if (!config.present) throw std::runtime_error("mmproj carries no vision encoder");
    if (config.projector_type != "qwen3vl_merger")
        throw std::runtime_error("mmproj projector type '" + config.projector_type +
                                 "' is not qwen3vl_merger");
    if (!config.patch_size || !config.embedding_length || !config.block_count ||
        !config.head_count || !config.feed_forward_length || !config.projection_dim ||
        !config.image_size)
        throw std::runtime_error("mmproj clip.vision.* geometry is incomplete");
    if (config.embedding_length % config.head_count)
        throw std::runtime_error("mmproj embedding_length is not a multiple of head_count");
    if (config.head_dim() % 4)
        throw std::runtime_error("mmproj head width must be a multiple of 4 for 2D rope");
    if (config.head_dim() > 128)
        throw std::runtime_error("mmproj head width above 128 is not supported by the attention kernel");
    if (config.spatial_merge_size != 2)
        throw std::runtime_error("mmproj spatial_merge_size other than 2 is not supported");
    if (config.image_size % config.patch_size)
        throw std::runtime_error("mmproj image_size is not a multiple of patch_size");
}

// A weight matrix as the tower's GEMM reads it: bf16 straight off the mmproj
// mapping, or widened to f32 for any other storage type.
struct VisionMatrix {
    std::uint64_t device = 0;
    bool bf16 = false;
};

struct VisionBlockWeights {
    std::uint64_t ln1_weight = 0, ln1_bias = 0, ln2_weight = 0, ln2_bias = 0;
    VisionMatrix qkv, out, up, down;
    std::uint64_t qkv_bias = 0, out_bias = 0, up_bias = 0, down_bias = 0;
    // Deepstack merger on this block's output; absent (zero) on most blocks.
    bool deepstack = false;
    std::uint64_t deepstack_norm_weight = 0, deepstack_norm_bias = 0;
    VisionMatrix deepstack_fc1, deepstack_fc2;
    std::uint64_t deepstack_fc1_bias = 0, deepstack_fc2_bias = 0;
};

// Device-resident tower plus the host tables an image's preparation reads.
// Owned by the Qwen runtime; released with its other device state.
struct VisionTowerState {
    bool ready = false;
    std::vector<std::uint64_t> allocations;   // every device buffer, for release
    VisionMatrix patch;                        // [E][3*p*p], both temporal slices summed
    std::uint64_t patch_bias = 0;
    std::vector<float> position_table;         // host [side*side][E]
    std::vector<VisionBlockWeights> blocks;
    std::uint64_t post_weight = 0, post_bias = 0;
    VisionMatrix merger_fc1, merger_fc2;
    std::uint64_t merger_fc1_bias = 0, merger_fc2_bias = 0;
    std::uint64_t weight_bytes = 0;
    // Per-image workspace, grown on demand.
    std::uint64_t workspace = 0, workspace_bytes = 0;
};

// Pixel dimensions an image is resized to before patching: the Qwen
// `smart_resize` -- both sides rounded to the token side (32 px here), the
// aspect ratio kept, the area held within [min_pixels, max_pixels]. Token
// counts are the merged ones the language model sees.
struct VisionResize {
    std::uint32_t width = 0, height = 0;
    std::uint32_t grid_w = 0, grid_h = 0;   // patches per axis
    std::uint32_t tokens = 0;               // merged tokens
};

inline VisionResize vision_smart_resize(const VisionConfig& config,
                                        std::uint32_t width, std::uint32_t height,
                                        std::uint32_t min_tokens, std::uint32_t max_tokens) {
    if (!width || !height) throw std::runtime_error("image has no pixels");
    const double side = config.token_side();
    const double min_pixels = static_cast<double>(min_tokens) * side * side;
    const double max_pixels = static_cast<double>(max_tokens) * side * side;
    auto round_to = [&](double value) {
        return std::max(side, std::round(value / side) * side);
    };
    double h = round_to(height), w = round_to(width);
    if (h * w > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
        h = std::max(side, std::floor(height / beta / side) * side);
        w = std::max(side, std::floor(width / beta / side) * side);
    } else if (h * w < min_pixels) {
        const double beta = std::sqrt(min_pixels / (static_cast<double>(height) * width));
        h = std::ceil(height * beta / side) * side;
        w = std::ceil(width * beta / side) * side;
    }
    VisionResize out;
    out.width = static_cast<std::uint32_t>(w);
    out.height = static_cast<std::uint32_t>(h);
    out.grid_w = out.width / config.patch_size;
    out.grid_h = out.height / config.patch_size;
    out.tokens = (out.grid_w / config.spatial_merge_size) *
                 (out.grid_h / config.spatial_merge_size);
    return out;
}

// One patch per row, window-major, each row laid out (channel, y, x) to match
// the conv weight's `[out][3][patch][patch]` flattening. `pixels` is HWC f32,
// already normalized. Also emits each patch's (row, column) for the 2D rope.
inline void vision_patchify(const VisionConfig& config, const float* pixels,
                            std::uint32_t width, std::uint32_t height,
                            std::vector<float>& patches,
                            std::vector<std::int32_t>& rows,
                            std::vector<std::int32_t>& columns) {
    const std::uint32_t patch = config.patch_size, merge = config.spatial_merge_size;
    if (width % (patch * merge) || height % (patch * merge))
        throw std::runtime_error("image sides must be multiples of the merged patch");
    const std::uint32_t grid_w = width / patch, grid_h = height / patch;
    const std::size_t patch_elements = static_cast<std::size_t>(3) * patch * patch;
    const std::size_t count = static_cast<std::size_t>(grid_w) * grid_h;
    patches.assign(count * patch_elements, 0.0f);
    rows.resize(count);
    columns.resize(count);
    std::size_t index = 0;
    for (std::uint32_t wy = 0; wy < grid_h; wy += merge)
        for (std::uint32_t wx = 0; wx < grid_w; wx += merge)
            for (std::uint32_t dy = 0; dy < merge; ++dy)
                for (std::uint32_t dx = 0; dx < merge; ++dx, ++index) {
                    const std::uint32_t py = wy + dy, px = wx + dx;
                    rows[index] = static_cast<std::int32_t>(py);
                    columns[index] = static_cast<std::int32_t>(px);
                    float* out = patches.data() + index * patch_elements;
                    for (std::uint32_t c = 0; c < 3; ++c)
                        for (std::uint32_t y = 0; y < patch; ++y)
                            for (std::uint32_t x = 0; x < patch; ++x) {
                                const std::size_t source =
                                    (static_cast<std::size_t>(py * patch + y) * width +
                                     (px * patch + x)) * 3 + c;
                                out[(c * patch + y) * patch + x] = pixels[source];
                            }
                }
}

// The learned `side x side` position table, bilinearly resampled (corners
// aligned, as ggml's GGML_SCALE_FLAG_ALIGN_CORNERS and HF's linspace both do)
// to the patch grid and written in the same window-major order as the
// patches. `table` is [side*side][width] row-major.
inline void vision_position_embeddings(const VisionConfig& config, const float* table,
                                       std::uint32_t grid_w, std::uint32_t grid_h,
                                       std::vector<float>& out) {
    const std::uint32_t side = config.position_grid_side();
    const std::uint32_t width = config.embedding_length, merge = config.spatial_merge_size;
    const std::size_t count = static_cast<std::size_t>(grid_w) * grid_h;
    out.assign(count * width, 0.0f);
    auto axis = [&](std::uint32_t i, std::uint32_t n, std::uint32_t& lo, std::uint32_t& hi,
                    float& weight_hi) {
        const double position = n > 1
            ? static_cast<double>(i) * (side - 1) / static_cast<double>(n - 1)
            : 0.0;
        lo = static_cast<std::uint32_t>(std::floor(position));
        hi = std::min(lo + 1, side - 1);
        weight_hi = static_cast<float>(position - lo);
    };
    std::size_t index = 0;
    for (std::uint32_t wy = 0; wy < grid_h; wy += merge)
        for (std::uint32_t wx = 0; wx < grid_w; wx += merge)
            for (std::uint32_t dy = 0; dy < merge; ++dy)
                for (std::uint32_t dx = 0; dx < merge; ++dx, ++index) {
                    std::uint32_t y0, y1, x0, x1;
                    float wy1, wx1;
                    axis(wy + dy, grid_h, y0, y1, wy1);
                    axis(wx + dx, grid_w, x0, x1, wx1);
                    const float w00 = (1 - wy1) * (1 - wx1), w01 = (1 - wy1) * wx1;
                    const float w10 = wy1 * (1 - wx1), w11 = wy1 * wx1;
                    const float* r00 = table + (static_cast<std::size_t>(y0) * side + x0) * width;
                    const float* r01 = table + (static_cast<std::size_t>(y0) * side + x1) * width;
                    const float* r10 = table + (static_cast<std::size_t>(y1) * side + x0) * width;
                    const float* r11 = table + (static_cast<std::size_t>(y1) * side + x1) * width;
                    float* dst = out.data() + index * width;
                    for (std::uint32_t e = 0; e < width; ++e)
                        dst[e] = w00 * r00[e] + w01 * r01[e] + w10 * r10[e] + w11 * r11[e];
                }
}

// Interleaved M-RoPE for the language model (`rope.dimension_sections`,
// ggml's GGML_ROPE_TYPE_IMROPE). Rotary pair `p` takes the temporal, height
// or width position by `p % 3` while it lies inside three times that
// section; a pair matched by none of them falls back to the temporal one,
// which is also what a text token carries in all three. Returns the
// component index 0/1/2 for a pair. Mirrored by the CUDA rope kernels.
inline int mrope_component(int pair, const std::uint32_t sections[4]) {
    const int total = static_cast<int>(sections[0] + sections[1] + sections[2] + sections[3]);
    const int sector = total > 0 ? pair % total : pair;
    const int which = sector % 3;
    if (which == 1 && sector < 3 * static_cast<int>(sections[1])) return 1;
    if (which == 2 && sector < 3 * static_cast<int>(sections[2])) return 2;
    return 0;
}

}  // namespace flyweight::v2
