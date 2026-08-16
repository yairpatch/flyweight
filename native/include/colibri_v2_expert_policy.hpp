#pragma once

#include <cstdint>

namespace colibri::v2 {

enum class ExpertExecutionMode : std::int32_t {
    streamed_gpu = 0,
    cpu = 1,
    hybrid = 2,
};

enum class ExpertExecutionPhase : std::uint8_t {
    prepare,
    prefill,
    decode,
    verification,
};

constexpr bool valid_expert_execution_mode(std::int32_t value) {
    return value >= static_cast<std::int32_t>(ExpertExecutionMode::streamed_gpu)
        && value <= static_cast<std::int32_t>(ExpertExecutionMode::hybrid);
}

constexpr ExpertExecutionMode expert_execution_mode(std::int32_t value) {
    return static_cast<ExpertExecutionMode>(value);
}

constexpr std::int32_t expert_execution_mode_value(ExpertExecutionMode mode) {
    return static_cast<std::int32_t>(mode);
}

struct ExpertExecutionPolicy {
    ExpertExecutionMode mode = ExpertExecutionMode::streamed_gpu;
    ExpertExecutionPhase phase = ExpertExecutionPhase::prepare;
    bool admission_enabled = true;
    bool hybrid_prefill_cpu = false;
    bool residency_frozen = false;

    constexpr bool is_streamed_gpu() const {
        return mode == ExpertExecutionMode::streamed_gpu;
    }

    constexpr bool is_cpu() const {
        return mode == ExpertExecutionMode::cpu;
    }

    constexpr bool is_hybrid() const {
        return mode == ExpertExecutionMode::hybrid;
    }

    constexpr bool routed_gpu_execution_allowed() const {
        return !is_cpu() &&
            !(is_hybrid() && phase == ExpertExecutionPhase::prefill &&
              hybrid_prefill_cpu);
    }

    constexpr bool routed_cpu_execution_allowed() const {
        // Rows forward keeps CPU fallback for non-resident experts even in
        // streamed-GPU mode. Single-token streamed decode pages every route.
        return !is_streamed_gpu() ||
            phase == ExpertExecutionPhase::prefill ||
            phase == ExpertExecutionPhase::verification;
    }

    constexpr bool misses_may_be_admitted() const {
        // Verification admits for the same reason decode does: under MTP it
        // *replaces* decode, so excluding it left the device cache permanently
        // empty and pushed every routed expert onto the CPU. Prefill still may
        // not churn the cache -- it only trains the history for a bulk seed.
        return routed_gpu_execution_allowed() &&
            (phase == ExpertExecutionPhase::decode ||
             phase == ExpertExecutionPhase::verification) &&
            admission_enabled && !residency_frozen;
    }

    constexpr bool residency_may_change() const {
        return misses_may_be_admitted();
    }

    constexpr bool records_prefill_frequency() const {
        return phase == ExpertExecutionPhase::prefill;
    }

    constexpr bool route_pruning_allowed() const {
        return !is_streamed_gpu();
    }
};

}  // namespace colibri::v2
