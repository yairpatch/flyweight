#pragma once

#include <cstdint>

namespace colibri::v2::workspace {

constexpr std::uint64_t kDeviceAlignment = 256;
// Largest top_k the GPU reduction serves. Above this the sampler falls back
// to downloading the full vocabulary and sorting on the host -- a ~600KB
// transfer plus a partial_sort of ~150K floats, every token. 32 left the
// common client default of top_k=40 on that fallback. The block-sort/merge
// is exact for any k (each stage keeps the top k of its 1024 inputs, and a
// global top-k element always ranks inside its own block's k), but the merge
// loop only shrinks while k < 1024, so the cap must stay well below that;
// 256 converges 4x per round and costs ~1MB of workspace.
constexpr std::uint64_t kSamplingTopKCapacity = 256;
// How far a constrained or penalized step widens its candidate set so the
// grammar and the penalties have alternatives to promote. Deliberately NOT
// the GPU capacity above: widening tracks what the step needs, and raising
// it with the capacity would change sampled output for every constrained
// request that predates the wider reduction.
constexpr std::uint64_t kSamplingConstrainedMinimum = 32;
constexpr std::uint64_t kSamplingSortItemsPerBlock = 1024;
constexpr std::uint64_t kSamplingSortBlockCapacity = 256;
constexpr std::uint64_t kSamplingSortCapacity =
    kSamplingTopKCapacity * kSamplingSortBlockCapacity;

constexpr std::uint64_t align(std::uint64_t bytes) {
    return (bytes + kDeviceAlignment - 1) / kDeviceAlignment * kDeviceAlignment;
}

struct Region {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;

    constexpr std::uint64_t address(std::uint64_t base) const {
        return base + offset;
    }
};

class Builder {
public:
    constexpr Region add(std::uint64_t bytes) {
        const Region region{cursor_, bytes};
        cursor_ += align(bytes);
        return region;
    }

    constexpr std::uint64_t bytes() const { return cursor_; }

private:
    std::uint64_t cursor_ = 0;
};

struct QwenDecodeWorkspaceLayout {
    Region hidden;
    Region residual;
    Region normalized;
    Region first;
    Region second;
    Region third;
    Region fourth;
    Region dense_q8;
    Region dense_q8_scales;
    Region activated;
    Region router_logits;
    Region selected_device;
    Region route_weights;
    Region logits;
    Region sampling_selected;
    Region sampling_logits;
    Region sampling_sort_indices_a;
    Region sampling_sort_values_a;
    Region sampling_sort_indices_b;
    Region sampling_sort_values_b;
    Region argmax_device;
    Region attention_scores;
    // qwen4exp gated residual. Zero-sized on every other architecture.
    // `streams` is the residual state itself (hc_count x hidden, alive across
    // the whole token); the rest is boundary scratch. `hc_gates` holds the raw
    // inject logits between hc_pre and the post-block inject. `ple_embed` is
    // the uploaded n-gram gather for the current token.
    Region streams;
    Region hc_normed;
    Region hc_wide;
    Region hc_low;
    Region hc_gates;
    Region ple_embed;
    std::uint64_t bytes = 0;
};

constexpr QwenDecodeWorkspaceLayout qwen_decode(
    std::uint64_t hidden, std::uint64_t scratch, std::uint64_t top_k,
    std::uint64_t intermediate, std::uint64_t experts, std::uint64_t vocabulary,
    std::uint64_t attention_heads, std::uint64_t context,
    std::uint64_t hc_count = 0, std::uint64_t hc_low_rank = 0,
    bool ple = false
) {
    Builder builder;
    QwenDecodeWorkspaceLayout layout;
    layout.hidden = builder.add(hidden * sizeof(float));
    layout.residual = builder.add(hidden * sizeof(float));
    layout.normalized = builder.add(hidden * sizeof(float));
    layout.first = builder.add(scratch * sizeof(float));
    layout.second = builder.add(scratch * sizeof(float));
    layout.third = builder.add(scratch * sizeof(float));
    layout.fourth = builder.add(scratch * sizeof(float));
    layout.dense_q8 = builder.add(scratch);
    layout.dense_q8_scales =
        builder.add(((scratch + 31) / 32) * sizeof(std::uint16_t));
    layout.activated = builder.add(top_k * intermediate * sizeof(float));
    layout.router_logits = builder.add(experts * sizeof(float));
    layout.selected_device = builder.add(top_k * sizeof(std::int32_t));
    layout.route_weights = builder.add(top_k * sizeof(float));
    layout.logits = builder.add(vocabulary * sizeof(float));
    layout.sampling_selected =
        builder.add(kSamplingTopKCapacity * sizeof(std::int32_t));
    layout.sampling_logits =
        builder.add(kSamplingTopKCapacity * sizeof(float));
    layout.sampling_sort_indices_a =
        builder.add(kSamplingSortCapacity * sizeof(std::int32_t));
    layout.sampling_sort_values_a =
        builder.add(kSamplingSortCapacity * sizeof(float));
    layout.sampling_sort_indices_b =
        builder.add(kSamplingSortCapacity * sizeof(std::int32_t));
    layout.sampling_sort_values_b =
        builder.add(kSamplingSortCapacity * sizeof(float));
    layout.argmax_device = builder.add(sizeof(std::uint64_t));
    layout.attention_scores =
        builder.add(attention_heads * context * sizeof(float));
    layout.streams = builder.add(hc_count * hidden * sizeof(float));
    layout.hc_normed = builder.add(hc_count * hidden * sizeof(float));
    layout.hc_wide = builder.add(hc_count * hidden * sizeof(float));
    layout.hc_low = builder.add(hc_low_rank * sizeof(float));
    layout.hc_gates = builder.add(hc_count * sizeof(float));
    layout.ple_embed = builder.add((ple ? hidden : 0) * sizeof(float));
    layout.bytes = builder.bytes();
    return layout;
}

struct QwenRowsWorkspaceLayout {
    Region hidden;
    Region residual;
    Region normalized;
    Region first;
    Region second;
    Region third;
    Region fourth;
    Region router_logits;
    Region selected_device;
    Region route_weights;
    Region gpu_activated;
    Region rows_q8;
    Region rows_q8_scales;
    Region gpu_gate_table;
    Region gpu_up_table;
    Region gpu_down_table;
    Region gpu_weight_table;
    Region gpu_gate_scale_table;
    Region gpu_up_scale_table;
    Region gpu_count_table;
    Region token_device;
    Region winners;
    Region attention_scores;
    // Chunked DeltaNet intermediates. Zero-sized unless the model has DeltaNet
    // layers with the head_dim the chunked kernels require.
    Region delta_attn;
    Region delta_pmat;
    Region delta_gcum;
    Region delta_beta;
    Region delta_qinv;
    Region delta_kinv;
    Region delta_w;
    Region delta_u;
    // Preserves the target hidden row preceding a chunk plus every output row
    // while the MTP cache builder reuses the rest of the workspace.
    Region mtp_prompt_hidden;
    // qwen4exp gated residual, rows form; zero-sized on every other arch.
    // `streams` carries the whole chunk's residual state; the rest is
    // boundary scratch, also rows-wide because every row's boundary runs in
    // one batched launch.
    Region streams;
    Region hc_normed;
    Region hc_wide;
    Region hc_low;
    Region hc_gates;
    Region ple_embed;
    std::uint64_t bytes = 0;
};

constexpr std::uint64_t kDeltaChunk = 64;
constexpr std::uint64_t kDeltaDim = 128;
constexpr std::uint64_t kDeltaChunkedMinimumRows = 1024;

constexpr bool use_chunked_delta(
    std::uint64_t rows, std::uint64_t head_dim,
    std::uint64_t value_heads, std::uint64_t prepared_value_heads
) {
    return rows >= kDeltaChunkedMinimumRows && head_dim == kDeltaDim &&
        value_heads != 0 && value_heads == prepared_value_heads;
}

constexpr QwenRowsWorkspaceLayout qwen_rows(
    std::uint64_t rows, std::uint64_t hidden, std::uint64_t scratch,
    std::uint64_t top_k, std::uint64_t intermediate, std::uint64_t experts,
    std::uint64_t attention_heads, std::uint64_t context,
    std::uint64_t delta_value_heads, bool mtp = false,
    std::uint64_t hc_count = 0, std::uint64_t hc_low_rank = 0,
    bool ple = false
) {
    Builder builder;
    QwenRowsWorkspaceLayout layout;
    layout.hidden = builder.add(rows * hidden * sizeof(float));
    layout.residual = builder.add(rows * hidden * sizeof(float));
    layout.normalized = builder.add(rows * hidden * sizeof(float));
    layout.first = builder.add(rows * scratch * sizeof(float));
    layout.second = builder.add(rows * scratch * sizeof(float));
    layout.third = builder.add(rows * scratch * sizeof(float));
    layout.fourth = builder.add(rows * scratch * sizeof(float));
    layout.router_logits = builder.add(rows * experts * sizeof(float));
    layout.selected_device = builder.add(rows * top_k * sizeof(std::int32_t));
    layout.route_weights = builder.add(rows * top_k * sizeof(float));
    layout.gpu_activated =
        builder.add(rows * top_k * intermediate * sizeof(float));
    // Q8-quantized activations for the DP4A projections. The MoE path borrows
    // `gpu_activated` for this, which is sized off top_k and therefore empty on
    // a dense model -- so the DP4A kernels had no scratch to work in and dense
    // rows fell back to reconstructing every weight in float. One byte per
    // element plus an fp16 scale per 32-element block; the scales keep a float
    // stride so the region stays aligned for either width.
    layout.rows_q8 = builder.add(rows * scratch);
    layout.rows_q8_scales = builder.add(rows * (scratch / 32 + 1) * sizeof(float));
    layout.gpu_gate_table =
        builder.add(rows * top_k * sizeof(std::uint64_t));
    layout.gpu_up_table =
        builder.add(rows * top_k * sizeof(std::uint64_t));
    layout.gpu_down_table =
        builder.add(rows * top_k * sizeof(std::uint64_t));
    layout.gpu_weight_table = builder.add(rows * top_k * sizeof(float));
    layout.gpu_gate_scale_table = builder.add(rows * top_k * sizeof(float));
    layout.gpu_up_scale_table = builder.add(rows * top_k * sizeof(float));
    layout.gpu_count_table = builder.add(rows * sizeof(std::int32_t));
    layout.token_device = builder.add(rows * sizeof(std::uint32_t));
    layout.winners = builder.add(rows * sizeof(std::uint64_t));
    layout.attention_scores =
        builder.add(attention_heads * context * sizeof(float));
    {
        // Sized so a partial trailing chunk still gets a full 64x64 pair of score
        // matrices. The `core` output is not here: it aliases `first`, which is
        // free once the causal conv has consumed it, and the epilogue rewrites it
        // in place.
        const std::uint64_t chunks =
            (rows + kDeltaChunk - 1) / kDeltaChunk * (delta_value_heads ? 1 : 0);
        const std::uint64_t pairs = chunks * delta_value_heads * kDeltaChunk * kDeltaChunk;
        const std::uint64_t scalars = rows * delta_value_heads * (delta_value_heads ? 1 : 0);
        const std::uint64_t vectors = scalars * kDeltaDim;
        layout.delta_attn = builder.add(pairs * sizeof(float));
        layout.delta_pmat = builder.add(pairs * sizeof(float));
        layout.delta_gcum = builder.add(scalars * sizeof(float));
        layout.delta_beta = builder.add(scalars * sizeof(float));
        layout.delta_qinv = builder.add(scalars * sizeof(float));
        layout.delta_kinv = builder.add(scalars * sizeof(float));
        layout.delta_w = builder.add(vectors * sizeof(float));
        layout.delta_u = builder.add(vectors * sizeof(float));
    }
    layout.mtp_prompt_hidden =
        builder.add((mtp ? rows + 1 : 0) * hidden * sizeof(float));
    layout.streams = builder.add(rows * hc_count * hidden * sizeof(float));
    layout.hc_normed = builder.add(rows * hc_count * hidden * sizeof(float));
    layout.hc_wide = builder.add(rows * hc_count * hidden * sizeof(float));
    layout.hc_low = builder.add(rows * hc_low_rank * sizeof(float));
    layout.hc_gates = builder.add(rows * hc_count * sizeof(float));
    layout.ple_embed = builder.add((ple ? rows * hidden : 0) * sizeof(float));
    layout.bytes = builder.bytes();
    return layout;
}

struct QwenDecodeHostLayout {
    Region selected;
    Region weights;
    Region input;
    Region activated;
    Region output;
    Region winner;
    std::uint64_t bytes = 0;
};

constexpr QwenDecodeHostLayout qwen_decode_host(
    std::uint64_t hidden, std::uint64_t top_k, std::uint64_t intermediate
) {
    Builder builder;
    QwenDecodeHostLayout layout;
    layout.selected = builder.add(top_k * sizeof(std::int32_t));
    layout.weights = builder.add(top_k * sizeof(float));
    layout.input = builder.add(hidden * sizeof(float));
    layout.activated = builder.add(top_k * intermediate * sizeof(float));
    layout.output = builder.add(hidden * sizeof(float));
    layout.winner = builder.add(sizeof(std::uint64_t));
    layout.bytes = builder.bytes();
    return layout;
}

struct QwenRowsHostLayout {
    Region selected;
    Region weights;
    Region input;
    Region activated;
    Region down_input;
    Region output;
    Region gate_table;
    Region up_table;
    Region down_table;
    Region weight_table;
    Region gate_scale_table;
    Region up_scale_table;
    Region counts;
    std::uint64_t bytes = 0;
};

constexpr QwenRowsHostLayout qwen_rows_host(
    std::uint64_t rows, std::uint64_t hidden, std::uint64_t top_k,
    std::uint64_t intermediate
) {
    Builder builder;
    QwenRowsHostLayout layout;
    layout.selected = builder.add(rows * top_k * sizeof(std::int32_t));
    layout.weights = builder.add(rows * top_k * sizeof(float));
    layout.input = builder.add(rows * hidden * sizeof(float));
    layout.activated =
        builder.add(rows * top_k * intermediate * sizeof(float));
    layout.down_input = builder.add(rows * top_k * hidden * sizeof(float));
    layout.output = builder.add(rows * hidden * sizeof(float));
    layout.gate_table =
        builder.add(rows * top_k * sizeof(std::uint64_t));
    layout.up_table =
        builder.add(rows * top_k * sizeof(std::uint64_t));
    layout.down_table =
        builder.add(rows * top_k * sizeof(std::uint64_t));
    layout.weight_table = builder.add(rows * top_k * sizeof(float));
    layout.gate_scale_table = builder.add(rows * top_k * sizeof(float));
    layout.up_scale_table = builder.add(rows * top_k * sizeof(float));
    layout.counts = builder.add(rows * sizeof(std::int32_t));
    layout.bytes = builder.bytes();
    return layout;
}

}  // namespace colibri::v2::workspace
