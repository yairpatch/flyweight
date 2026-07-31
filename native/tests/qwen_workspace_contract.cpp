#include "colibri_v2_workspace.hpp"

#include <array>
#include <cstdint>

namespace ws = colibri::v2::workspace;

template <std::size_t N>
bool valid_regions(
    const std::array<ws::Region, N>& regions, std::uint64_t total
) {
    std::uint64_t cursor = 0;
    for (const auto& region : regions) {
        if (region.offset != cursor ||
            region.offset % ws::kDeviceAlignment != 0 ||
            region.offset + region.size > total) return false;
        cursor += ws::align(region.size);
    }
    return cursor == total;
}

std::uint64_t legacy_decode_bytes(
    std::uint64_t hidden, std::uint64_t scratch, std::uint64_t top_k,
    std::uint64_t intermediate, std::uint64_t experts,
    std::uint64_t vocabulary, std::uint64_t heads, std::uint64_t context
) {
    return ws::align(hidden * sizeof(float)) * 3 +
        ws::align(scratch * sizeof(float)) * 4 +
        ws::align(scratch) +
        ws::align(((scratch + 31) / 32) * sizeof(std::uint16_t)) +
        ws::align(top_k * intermediate * sizeof(float)) +
        ws::align(experts * sizeof(float)) +
        ws::align(top_k * sizeof(std::int32_t)) +
        ws::align(top_k * sizeof(float)) +
        ws::align(vocabulary * sizeof(float)) +
        ws::align(sizeof(std::uint64_t)) +
        ws::align(heads * context * sizeof(float));
}

std::uint64_t legacy_rows_bytes(
    std::uint64_t rows, std::uint64_t hidden, std::uint64_t scratch,
    std::uint64_t top_k, std::uint64_t intermediate, std::uint64_t experts,
    std::uint64_t heads, std::uint64_t context
) {
    return ws::align(rows * hidden * sizeof(float)) * 3 +
        ws::align(rows * scratch * sizeof(float)) * 4 +
        ws::align(rows * experts * sizeof(float)) +
        ws::align(rows * top_k * sizeof(std::int32_t)) +
        ws::align(rows * top_k * sizeof(float)) +
        ws::align(rows * top_k * intermediate * sizeof(float)) +
        ws::align(rows * top_k * sizeof(std::uint64_t)) * 3 +
        ws::align(rows * top_k * sizeof(float)) * 3 +
        ws::align(rows * sizeof(std::int32_t)) +
        ws::align(rows * sizeof(std::uint32_t)) +
        ws::align(rows * sizeof(std::uint64_t)) +
        ws::align(heads * context * sizeof(float));
}

std::uint64_t legacy_decode_host_bytes(
    std::uint64_t hidden, std::uint64_t top_k, std::uint64_t intermediate
) {
    return ws::align(top_k * sizeof(std::int32_t)) +
        ws::align(top_k * sizeof(float)) +
        ws::align(hidden * sizeof(float)) +
        ws::align(top_k * intermediate * sizeof(float)) +
        ws::align(hidden * sizeof(float)) +
        ws::align(sizeof(std::uint64_t));
}

std::uint64_t legacy_rows_host_bytes(
    std::uint64_t rows, std::uint64_t hidden, std::uint64_t top_k,
    std::uint64_t intermediate
) {
    return ws::align(rows * top_k * sizeof(std::int32_t)) +
        ws::align(rows * top_k * sizeof(float)) +
        ws::align(rows * hidden * sizeof(float)) +
        ws::align(rows * top_k * intermediate * sizeof(float)) +
        ws::align(rows * top_k * hidden * sizeof(float)) +
        ws::align(rows * hidden * sizeof(float)) +
        ws::align(rows * top_k * sizeof(std::uint64_t)) * 3 +
        ws::align(rows * top_k * sizeof(float)) * 3 +
        ws::align(rows * sizeof(std::int32_t));
}

int main() {
    constexpr std::uint64_t hidden = 4096;
    constexpr std::uint64_t scratch = 12288;
    constexpr std::uint64_t top_k = 8;
    constexpr std::uint64_t intermediate = 1536;
    constexpr std::uint64_t experts = 256;
    constexpr std::uint64_t vocabulary = 151936;
    constexpr std::uint64_t heads = 32;
    constexpr std::uint64_t context = 32768;

    for (const auto active_top_k : {std::uint64_t{1}, std::uint64_t{4}, top_k}) {
        const auto layout = ws::qwen_decode(
            hidden, scratch, active_top_k, intermediate, experts, vocabulary,
            heads, context);
        const std::array regions{
            layout.hidden, layout.residual, layout.normalized, layout.first,
            layout.second, layout.third, layout.fourth, layout.dense_q8,
            layout.dense_q8_scales, layout.activated, layout.router_logits,
            layout.selected_device, layout.route_weights, layout.logits,
            layout.argmax_device, layout.attention_scores};
        if (!valid_regions(regions, layout.bytes) ||
            layout.bytes != legacy_decode_bytes(
                hidden, scratch, active_top_k, intermediate, experts,
                vocabulary, heads, context)) return 1;
        // Adjacent multi-decode slices must not overlap.
        if (layout.attention_scores.offset +
                ws::align(layout.attention_scores.size) != layout.bytes ||
            layout.attention_scores.address(layout.bytes) < layout.bytes)
            return 2;
    }

    for (const auto rows : {
             std::uint64_t{1}, std::uint64_t{9}, std::uint64_t{64},
             std::uint64_t{1024}, std::uint64_t{4096}}) {
        const auto layout = ws::qwen_rows(
            rows, hidden, scratch, top_k, intermediate, experts, heads,
            context);
        const std::array regions{
            layout.hidden, layout.residual, layout.normalized, layout.first,
            layout.second, layout.third, layout.fourth, layout.router_logits,
            layout.selected_device, layout.route_weights, layout.gpu_activated,
            layout.gpu_gate_table, layout.gpu_up_table, layout.gpu_down_table,
            layout.gpu_weight_table, layout.gpu_gate_scale_table,
            layout.gpu_up_scale_table, layout.gpu_count_table,
            layout.token_device, layout.winners, layout.attention_scores};
        if (!valid_regions(regions, layout.bytes) ||
            layout.bytes != legacy_rows_bytes(
                rows, hidden, scratch, top_k, intermediate, experts, heads,
                context)) return 3;

        const auto host =
            ws::qwen_rows_host(rows, hidden, top_k, intermediate);
        const std::array host_regions{
            host.selected, host.weights, host.input, host.activated,
            host.down_input, host.output, host.gate_table, host.up_table,
            host.down_table, host.weight_table, host.gate_scale_table,
            host.up_scale_table, host.counts};
        if (!valid_regions(host_regions, host.bytes) ||
            host.bytes !=
                legacy_rows_host_bytes(rows, hidden, top_k, intermediate))
            return 4;
    }

    const auto host = ws::qwen_decode_host(hidden, top_k, intermediate);
    const std::array host_regions{
        host.selected, host.weights, host.input, host.activated, host.output,
        host.winner};
    return valid_regions(host_regions, host.bytes) &&
            host.bytes == legacy_decode_host_bytes(hidden, top_k, intermediate)
        ? 0 : 5;
}
