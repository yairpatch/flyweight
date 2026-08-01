#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace colibri::v2::expert_seed {

constexpr std::size_t kAutoShortExpertsPerLayer = 4;
constexpr std::size_t kAutoLongExpertsPerLayer = 48;
constexpr std::uint64_t kAutoLongRequestTokenThreshold = 256;
constexpr std::uint64_t kAutoMaxNanoseconds = 1000ull * 1000 * 1000;
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

}  // namespace colibri::v2::expert_seed
