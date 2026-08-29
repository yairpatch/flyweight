#pragma once

#include <cstdint>

namespace flyweight::v2 {

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

// Slot admission for a routed expert that just missed the device cache. The
// miss itself is already served (CPU fallback in hybrid, transient staging in
// streamed), so admission is purely speculative: it spends one expert upload
// now to buy later tokens a hit.
//
// An absolute observation threshold before that bet is taken -- llama.cpp's
// MoE-cache RFC reported ADMIT_AFTER=64 improving every workload it measured
// (ggml-org/llama.cpp#24528) -- was implemented here and measured on
// Qwen3.6-35B-A3B, and it is not wired because it never won:
//
//   * gating free slots as well cost -12% decode (the 1886-slot cache never
//     filled; hit rate 8.0% -> 1.0%), so coverage must never be refused;
//   * gating displacement only still lost monotonically at -3.0 / -4.5 / -14.0%
//     for thresholds 4 / 16 / 64, against a 4.7% within-arm spread;
//   * and where it did not lose it did nothing: with a 221-slot cache,
//     threshold 4 admitted exactly the same 719 experts as no threshold.
//
// That last line is why: `strict_admission` below already refuses a candidate
// that is not demonstrably hotter than what it would evict, which is the same
// refusal expressed relatively instead of absolutely. The RFC's policy has no
// such comparator, so its threshold was standing in for one we already have.
constexpr bool expert_admission_allowed(
        std::uint32_t candidate_frequency, std::uint32_t victim_frequency,
        bool slot_is_free, bool allow_rejection, bool strict_admission) {
    // A deliberate placement (the post-prefill seed) chose its set by score,
    // not by recurrence, so the comparison does not apply to it.
    if (!allow_rejection) return true;
    // An empty slot displaces nothing.
    if (slot_is_free) return true;
    return strict_admission ? candidate_frequency > victim_frequency
                            : candidate_frequency >= victim_frequency;
}

}  // namespace flyweight::v2
