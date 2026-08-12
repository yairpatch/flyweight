// Contract for the BailingMoE3 `noaux_tc` router.
//
// The numbers below were produced by running BailingMoeV3Gate's own selection
// path (modeling_bailing_moe_v3.py:368-406) under torch, not by running this
// implementation and recording what it did.
//
// The case is chosen so that every property the router has to get right is
// falsifiable in one shot:
//
//   experts        0      1      2      3      4      5      6      7
//   sigmoid     0.269  0.622  0.881  0.438  0.731  0.679  0.119  0.525
//   bias         0      0    -0.90    0      0      0    +0.80    0
//   groups      |---0---|   |---1---|   |---2---|   |---3---|
//
//   * Expert 2 has the HIGHEST raw score and is still not selected: its bias
//     pushes its group below the cut. A router that let the bias leak into the
//     weights, or that ignored the bias in selection, both fail here.
//   * Expert 6 has the LOWEST raw score and IS selected, on bias alone -- yet
//     its weight is computed from the raw 0.119, not from 0.919. That is the
//     bias-steers-selection-but-not-weights property, and it is the one that
//     produces fluent, subtly wrong output when broken.
//   * Groups are ranked by the sum of their best TWO members, so group 3
//     (0.119+0.80, 0.525 -> 1.444) beats group 0 (0.269, 0.622 -> 0.891)
//     despite holding the single worst expert.

#include "colibri_v2_bailing.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

bool close(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-6f;
}

bool selection_follows_the_bias_but_weights_do_not() {
    const float logits[8] = {-1.0f, 0.5f, 2.0f, -0.25f, 1.0f, 0.75f, -2.0f, 0.1f};
    const float bias[8] = {0.0f, 0.0f, -0.9f, 0.0f, 0.0f, 0.0f, 0.8f, 0.0f};
    std::int32_t chosen[2]{};
    float weights[2]{};
    colibri::v2::bailing::moe_router(logits, bias, 8, 2, 4, 2, 2.5f, true,
                                     chosen, weights);
    if (chosen[0] != 6 || chosen[1] != 4) return false;
    // Raw sigmoids 0.119202919 and 0.731058598, normalized then scaled by 2.5.
    if (!close(weights[0], 0.350488991f)) return false;
    if (!close(weights[1], 2.149510860f)) return false;
    return true;
}

// With the bias removed the same call must pick the two best raw scores inside
// the two best groups -- confirming the group limit is doing something and the
// previous result was not an accident of ordering.
bool group_limiting_restricts_the_candidates() {
    const float logits[8] = {-1.0f, 0.5f, 2.0f, -0.25f, 1.0f, 0.75f, -2.0f, 0.1f};
    const float zero[8] = {};
    std::int32_t chosen[2]{};
    float weights[2]{};
    colibri::v2::bailing::moe_router(logits, zero, 8, 2, 4, 2, 1.0f, true,
                                     chosen, weights);
    // Groups by top-2 sum: g0 0.891, g1 1.319, g2 1.410, g3 0.644.
    // Keeping g1 and g2 leaves experts {2,3,4,5}; the best two are 2 and 4.
    if (chosen[0] != 2 || chosen[1] != 4) return false;
    const float sum = 0.880797029f + 0.731058598f;
    return close(weights[0], 0.880797029f / sum) &&
           close(weights[1], 0.731058598f / sum);
}

// A degenerate group configuration must fall back to flat top-k rather than
// masking everything away.
bool ungrouped_routing_still_selects() {
    const float logits[8] = {-1.0f, 0.5f, 2.0f, -0.25f, 1.0f, 0.75f, -2.0f, 0.1f};
    const float zero[8] = {};
    std::int32_t chosen[3]{};
    float weights[3]{};
    colibri::v2::bailing::moe_router(logits, zero, 8, 3, 1, 1, 1.0f, false,
                                     chosen, weights);
    if (chosen[0] != 2 || chosen[1] != 4 || chosen[2] != 5) return false;
    // normalize=false leaves the raw sigmoids in place.
    return close(weights[0], 0.880797029f) && close(weights[1], 0.731058598f);
}

// Scaling applies after normalization, and a single used expert skips
// normalization entirely (the reference guards it on top_k > 1).
bool single_expert_skips_normalization() {
    const float logits[4] = {0.0f, 3.0f, -3.0f, 0.0f};
    const float zero[4] = {};
    std::int32_t chosen[1]{};
    float weights[1]{};
    colibri::v2::bailing::moe_router(logits, zero, 4, 1, 1, 1, 2.0f, true,
                                     chosen, weights);
    // sigmoid(3) = 0.952574127, doubled rather than normalized to 1.
    return chosen[0] == 1 && close(weights[0], 2.0f * 0.952574127f);
}


// The shared expert is added to the routed sum and is NOT scaled by
// `routed_scaling_factor` -- that factor applies to the router's weights, which
// the shared expert does not have (modeling:448-449). Folding it in would scale
// the shared path by 2.5x on this checkpoint and still produce fluent text.
//
// Driven with every routed expert zeroed, so the output IS the shared expert:
// changing the scale must then change nothing at all.
bool the_shared_expert_ignores_the_routing_scale() {
    constexpr std::size_t hidden = 4, expert_size = 3, shared_size = 3, experts = 4;
    const std::vector<float> router(experts * hidden, 0.25f);
    const std::vector<float> bias(experts, 0.0f);
    const std::vector<float> zeros(experts * expert_size * hidden, 0.0f);
    const std::vector<float> hidden_state{0.5f, -0.25f, 0.75f, 1.0f};

    std::vector<float> shared_gate(shared_size * hidden), shared_up(shared_size * hidden),
        shared_down(hidden * shared_size);
    for (std::size_t i = 0; i < shared_gate.size(); ++i)
        shared_gate[i] = 0.21f * std::sin(0.6f * static_cast<float>(i) + 0.4f);
    for (std::size_t i = 0; i < shared_up.size(); ++i)
        shared_up[i] = 0.33f * std::cos(0.8f * static_cast<float>(i) + 1.7f);
    for (std::size_t i = 0; i < shared_down.size(); ++i)
        shared_down[i] = 0.29f * std::sin(1.1f * static_cast<float>(i) + 0.2f);

    const auto run = [&](float scale) {
        std::vector<float> out(hidden);
        colibri::v2::bailing::moe_block(
            hidden_state.data(), router.data(), bias.data(), zeros.data(),
            zeros.data(), zeros.data(), shared_gate.data(), shared_up.data(),
            shared_down.data(), hidden, expert_size, shared_size, experts, 2, 2, 1,
            scale, true, 0.0f, out.data());
        return out;
    };
    const auto plain = run(1.0f), scaled = run(2.5f);
    for (std::size_t i = 0; i < hidden; ++i)
        if (std::fabs(plain[i] - scaled[i]) > 1e-6f) return false;
    // And it must not be silently zero, which would pass the comparison above.
    float magnitude = 0.0f;
    for (const float value : plain) magnitude += std::fabs(value);
    return magnitude > 1e-4f;
}

// The per-layer SwiGLU clamp bounds BOTH halves before combining, not just the
// gate. Ling-3.0-tiny ships no clamp but the flash checkpoints do.
bool the_swiglu_clamp_bounds_both_halves() {
    const float gate[2] = {50.0f, -50.0f};
    const float up[2] = {50.0f, 50.0f};
    float clamped[2]{}, unclamped[2]{};
    colibri::v2::bailing::swiglu(gate, up, 2, 1.0f, clamped);
    colibri::v2::bailing::swiglu(gate, up, 2, 0.0f, unclamped);
    // With a limit of 1: silu(1)*1 = 0.7310586, silu(-1)*1 = -0.2689414.
    if (std::fabs(clamped[0] - 0.731058598f) > 1e-6f) return false;
    if (std::fabs(clamped[1] + 0.268941432f) > 1e-6f) return false;
    // Unclamped must differ, or the limit is being ignored rather than applied.
    return std::fabs(unclamped[0] - clamped[0]) > 1.0f;
}

}  // namespace

int main() {
    if (!selection_follows_the_bias_but_weights_do_not()) return 1;
    if (!group_limiting_restricts_the_candidates()) return 2;
    if (!ungrouped_routing_still_selects()) return 3;
    if (!single_expert_skips_normalization()) return 4;
    if (!the_shared_expert_ignores_the_routing_scale()) return 5;
    if (!the_swiglu_clamp_bounds_both_halves()) return 6;
    return 0;
}
