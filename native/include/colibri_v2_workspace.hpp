#pragma once

#include <cstdint>

namespace colibri::v2::workspace {

constexpr std::uint64_t kDeviceAlignment = 256;
constexpr std::uint64_t kSamplingTopKCapacity = 32;
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
    std::uint64_t bytes = 0;
};

constexpr QwenDecodeWorkspaceLayout qwen_decode(
    std::uint64_t hidden, std::uint64_t scratch, std::uint64_t top_k,
    std::uint64_t intermediate, std::uint64_t experts, std::uint64_t vocabulary,
    std::uint64_t attention_heads, std::uint64_t context
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
    std::uint64_t bytes = 0;
};

constexpr QwenRowsWorkspaceLayout qwen_rows(
    std::uint64_t rows, std::uint64_t hidden, std::uint64_t scratch,
    std::uint64_t top_k, std::uint64_t intermediate, std::uint64_t experts,
    std::uint64_t attention_heads, std::uint64_t context
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
