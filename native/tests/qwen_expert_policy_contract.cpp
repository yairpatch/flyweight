#include "colibri_v2_expert_policy.hpp"

#include <array>
#include <cstdint>

namespace v2 = colibri::v2;

struct Expected {
    v2::ExpertExecutionMode mode;
    v2::ExpertExecutionPhase phase;
    bool admission;
    bool cpu;
    bool gpu;
    bool admit;
    bool record;
    bool prune;
};

int main() {
    using Mode = v2::ExpertExecutionMode;
    using Phase = v2::ExpertExecutionPhase;
    constexpr std::array cases{
        Expected{Mode::streamed_gpu, Phase::prepare, true,
                 false, true, false, false, false},
        Expected{Mode::streamed_gpu, Phase::prefill, false,
                 true, true, false, true, false},
        Expected{Mode::streamed_gpu, Phase::decode, true,
                 false, true, true, false, false},
        Expected{Mode::streamed_gpu, Phase::decode, false,
                 false, true, false, false, false},
        Expected{Mode::streamed_gpu, Phase::verification, true,
                 true, true, true, false, false},
        Expected{Mode::cpu, Phase::prepare, true,
                 true, false, false, false, true},
        Expected{Mode::cpu, Phase::prefill, false,
                 true, false, false, true, true},
        Expected{Mode::cpu, Phase::decode, true,
                 true, false, false, false, true},
        Expected{Mode::cpu, Phase::decode, false,
                 true, false, false, false, true},
        Expected{Mode::cpu, Phase::verification, true,
                 true, false, false, false, true},
        Expected{Mode::hybrid, Phase::prepare, true,
                 true, true, false, false, true},
        Expected{Mode::hybrid, Phase::prefill, false,
                 true, true, false, true, true},
        Expected{Mode::hybrid, Phase::decode, true,
                 true, true, true, false, true},
        Expected{Mode::hybrid, Phase::decode, false,
                 true, true, false, false, true},
        Expected{Mode::hybrid, Phase::verification, true,
                 true, true, true, false, true},
    };

    for (const auto& expected : cases) {
        const v2::ExpertExecutionPolicy policy{
            expected.mode, expected.phase, expected.admission};
        if (policy.routed_cpu_execution_allowed() != expected.cpu ||
            policy.routed_gpu_execution_allowed() != expected.gpu ||
            policy.misses_may_be_admitted() != expected.admit ||
            policy.residency_may_change() != expected.admit ||
            policy.records_prefill_frequency() != expected.record ||
            policy.route_pruning_allowed() != expected.prune)
            return 1;
    }

    const v2::ExpertExecutionPolicy cpu_prefill{
        Mode::hybrid, Phase::prefill, false, true};
    if (!cpu_prefill.routed_cpu_execution_allowed() ||
        cpu_prefill.routed_gpu_execution_allowed() ||
        cpu_prefill.misses_may_be_admitted() ||
        !cpu_prefill.records_prefill_frequency() ||
        !cpu_prefill.route_pruning_allowed())
        return 4;

    const v2::ExpertExecutionPolicy unchanged_decode{
        Mode::hybrid, Phase::decode, true, true};
    if (!unchanged_decode.routed_gpu_execution_allowed() ||
        !unchanged_decode.misses_may_be_admitted())
        return 5;

    // MTP verification replaces decode, so it must admit: pinning this false
    // left the device expert cache permanently empty under MTP and ran every
    // routed expert on the CPU (measured -32% at drafts=2). CPU fallback stays
    // available for the routes admission declines.
    const v2::ExpertExecutionPolicy unchanged_verification{
        Mode::hybrid, Phase::verification, true, true};
    if (!unchanged_verification.routed_cpu_execution_allowed() ||
        !unchanged_verification.routed_gpu_execution_allowed() ||
        !unchanged_verification.misses_may_be_admitted())
        return 6;

    // ...but a frozen or admission-disabled runtime still overrides it.
    const v2::ExpertExecutionPolicy frozen_verification{
        Mode::hybrid, Phase::verification, true, false, true};
    if (frozen_verification.misses_may_be_admitted() ||
        frozen_verification.residency_may_change())
        return 8;
    const v2::ExpertExecutionPolicy closed_verification{
        Mode::hybrid, Phase::verification, false, false};
    if (closed_verification.misses_may_be_admitted()) return 9;

    const v2::ExpertExecutionPolicy frozen_decode{
        Mode::hybrid, Phase::decode, true, false, true};
    if (!frozen_decode.routed_cpu_execution_allowed() ||
        !frozen_decode.routed_gpu_execution_allowed() ||
        frozen_decode.misses_may_be_admitted() ||
        frozen_decode.residency_may_change())
        return 7;

    for (std::int32_t value = -2; value <= 4; ++value) {
        const bool expected = value >= 0 && value <= 2;
        if (v2::valid_expert_execution_mode(value) != expected) return 2;
        if (expected &&
            v2::expert_execution_mode_value(
                v2::expert_execution_mode(value)) != value)
            return 3;
    }
    return 0;
}
