#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace flyweight::v2::expert_seed {

constexpr std::size_t kAutoShortExpertsPerLayer = 4;
constexpr std::size_t kAutoLongExpertsPerLayer = 48;
constexpr std::uint64_t kAutoLongRequestTokenThreshold = 256;
// Upload budget for the automatic seed, in bytes rather than time. A
// wall-clock bound here made the pinned set depend on that run's upload speed
// (cold NVMe pages, PCIe contention, clock drift), which fed straight into
// expert placement and made greedy output vary across identical runs. 2 GiB
// preserves the old bound's intent -- it capped seeding at ~1 s, and 2 GiB is
// one second at a conservative cold-page-to-VRAM rate -- while making the
// stop point a pure function of the selection list. On slower disks the seed
// may now take longer than a second; that trade is deliberate, reproducibility
// over a hard latency cap.
constexpr std::uint64_t kAutoMaxUploadBytes = 2ull * 1024 * 1024 * 1024;
constexpr std::uint32_t kPromptWeight = 8;
constexpr std::uint32_t kMinimumPromptTokens = 32;
constexpr std::uint32_t kUsefulHistoryFrequency = 8;

constexpr std::size_t auto_experts_per_layer(
        std::size_t cache_slots, std::size_t cache_layers,
        std::uint64_t requested_generation_tokens) {
    if (!cache_layers) return 0;
    const auto capacity = cache_slots / cache_layers;
    const auto requested = requested_generation_tokens >=
            kAutoLongRequestTokenThreshold
        ? kAutoLongExpertsPerLayer
        : kAutoShortExpertsPerLayer;
    return std::min(requested, capacity);
}

constexpr std::uint64_t score(
        std::uint32_t prompt_frequency, std::uint32_t history_frequency) {
    const auto prompt = static_cast<std::uint64_t>(prompt_frequency) *
        kPromptWeight;
    return prompt + history_frequency;
}

constexpr bool has_useful_prompt(
        std::uint64_t prompt_observations, std::uint32_t layer_count,
        std::uint32_t experts_per_token) {
    const auto observations_per_token =
        static_cast<std::uint64_t>(layer_count) * experts_per_token;
    return observations_per_token &&
        prompt_observations >= observations_per_token * kMinimumPromptTokens;
}

constexpr bool should_seed(
        std::uint64_t prompt_observations, std::uint32_t layer_count,
        std::uint32_t experts_per_token, std::uint32_t max_history_frequency) {
    return has_useful_prompt(
        prompt_observations, layer_count, experts_per_token) ||
        max_history_frequency >= kUsefulHistoryFrequency;
}

}  // namespace flyweight::v2::expert_seed
