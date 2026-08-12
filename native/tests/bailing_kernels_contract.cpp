// The BailingMoE3 corpus kernels against their host implementations.
//
// `bailing_kda_recurrent_chunk` and `bailing_mla_attention` are the CUDA text
// the GPU compiles. The build
// also compiles that same text as host C++ for the CPU backend, so it can be
// launched and checked here with no GPU present -- which is the only way this
// kernel gets tested at all on a machine whose PyTorch cannot even see the
// card.
//
// The reference is `bailing::kda_recurrence`, which is itself pinned to
// flash-linear-attention's own definitions by native/tools/kda_reference_check.py
// and verified against the real checkpoint's layer 0. So a pass here chains
// back to the reference implementation rather than to another copy of my own
// reasoning.

#include <colibri_cpu_backend.hpp>

#include "colibri_v2_bailing.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

bool run(int heads, int rows) {
    constexpr int kHeadDim = 128;   // the kernel requires it
    const std::size_t width = static_cast<std::size_t>(heads) * kHeadDim;
    std::mt19937 rng(1234 + heads * 31 + rows);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto fill = [&](std::size_t count) {
        std::vector<float> values(count);
        for (auto& value : values) value = normal(rng);
        return values;
    };

    auto queries = fill(width * rows);
    auto keys = fill(width * rows);
    auto values = fill(width * rows);
    auto gate_raw = fill(width * rows);
    auto beta_logits = fill(static_cast<std::size_t>(heads) * rows);
    auto a_log = fill(heads);
    auto dt_bias = fill(width);
    auto initial = fill(static_cast<std::size_t>(heads) * kHeadDim * kHeadDim);

    // Reference: the host implementation, on its own copy of the state.
    std::vector<float> expected_state = initial;
    std::vector<float> expected(width * rows, 0.0f);
    colibri::v2::bailing::kda_recurrence(
        queries.data(), keys.data(), values.data(), gate_raw.data(),
        beta_logits.data(), a_log.data(), dt_bias.data(),
        static_cast<std::size_t>(rows), static_cast<std::size_t>(heads),
        kHeadDim, 1e-6f, expected_state.data(), expected.data());

    std::vector<float> actual_state = initial;
    std::vector<float> actual(width * rows, 0.0f);
    const float* queries_p = queries.data();
    const float* keys_p = keys.data();
    const float* values_p = values.data();
    const float* gate_p = gate_raw.data();
    const float* beta_p = beta_logits.data();
    const float* a_p = a_log.data();
    const float* dt_p = dt_bias.data();
    float* state_p = actual_state.data();
    float* output_p = actual.data();
    int rows_arg = rows, heads_arg = heads, head_dim_arg = kHeadDim;
    float epsilon = 1e-6f;
    void* arguments[] = {&queries_p, &keys_p, &values_p, &gate_p, &beta_p,
                         &a_p, &dt_p, &state_p, &output_p,
                         &rows_arg, &heads_arg, &head_dim_arg, &epsilon};
    if (colibri_cpu_launch_named("bailing_kda_recurrent_chunk",
                                 static_cast<std::uint32_t>(heads), 1, 128, 0, 0,
                                 arguments) != 0) {
        std::printf("  launch failed for heads=%d rows=%d\n", heads, rows);
        return false;
    }

    // Scale-relative: the state and outputs grow with the recurrence, so an
    // absolute tolerance would be meaningless across shapes.
    double scale = 0.0;
    for (const float value : expected) scale += std::fabs(value);
    scale = scale / static_cast<double>(expected.size()) + 1e-6;
    double worst_output = 0.0, worst_state = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        worst_output = std::max(worst_output,
                                std::fabs(double(actual[i]) - expected[i]) / scale);
    double state_scale = 0.0;
    for (const float value : expected_state) state_scale += std::fabs(value);
    state_scale = state_scale / static_cast<double>(expected_state.size()) + 1e-6;
    for (std::size_t i = 0; i < expected_state.size(); ++i)
        worst_state = std::max(worst_state,
            std::fabs(double(actual_state[i]) - expected_state[i]) / state_scale);

    const bool ok = worst_output < 2e-3 && worst_state < 2e-3;
    std::printf("  heads=%2d rows=%3d  output %.3e  state %.3e  %s\n",
                heads, rows, worst_output, worst_state, ok ? "OK" : "FAIL");
    return ok;
}

// The MLA kernel against mla_attention_absorbed, which was itself checked
// against the real checkpoint's layer 3 at ~1e-6 on real weights.
bool run_mla(int heads, int positions) {
    constexpr int kNope = 128, kRope = 64, kValue = 128, kLora = 512;
    std::mt19937 rng(99 + heads * 7 + positions);
    std::normal_distribution<float> normal(0.0f, 0.3f);
    auto fill = [&](std::size_t count) {
        std::vector<float> values(count);
        for (auto& value : values) value = normal(rng);
        return values;
    };
    auto query_nope = fill(static_cast<std::size_t>(heads) * kNope);
    auto query_rope = fill(static_cast<std::size_t>(heads) * kRope);
    auto kv_b = fill(static_cast<std::size_t>(heads) * (kNope + kValue) * kLora);
    auto latents = fill(static_cast<std::size_t>(positions) * kLora);
    auto rope_keys = fill(static_cast<std::size_t>(positions) * kRope);

    std::vector<float> expected(static_cast<std::size_t>(heads) * kValue, 0.0f);
    colibri::v2::bailing::mla_attention_absorbed(
        query_nope.data(), query_rope.data(), kv_b.data(), latents.data(),
        rope_keys.data(), static_cast<std::size_t>(positions),
        static_cast<std::size_t>(heads), kNope, kRope, kValue, kLora,
        expected.data());

    std::vector<float> actual(static_cast<std::size_t>(heads) * kValue, 0.0f);
    std::vector<float> scratch(static_cast<std::size_t>(heads) * positions, 0.0f);
    const float* qn = query_nope.data();
    const float* qr = query_rope.data();
    const float* kvb = kv_b.data();
    const float* lat = latents.data();
    const float* rope = rope_keys.data();
    float* scores = scratch.data();
    float* out = actual.data();
    int positions_arg = positions, heads_arg = heads, nope = kNope, ropedim = kRope,
        value_dim = kValue, lora = kLora;
    void* arguments[] = {&qn, &qr, &kvb, &lat, &rope, &scores, &out,
                         &positions_arg, &heads_arg, &nope, &ropedim,
                         &value_dim, &lora};
    if (colibri_cpu_launch_named("bailing_mla_attention",
                                 static_cast<std::uint32_t>(heads), 1, 128, 0, 0,
                                 arguments) != 0) {
        std::printf("  MLA launch failed heads=%d positions=%d\n", heads, positions);
        return false;
    }
    double scale = 0.0;
    for (const float value : expected) scale += std::fabs(value);
    scale = scale / static_cast<double>(expected.size()) + 1e-9;
    double worst = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        worst = std::max(worst, std::fabs(double(actual[i]) - expected[i]) / scale);
    const bool ok = worst < 2e-3;
    std::printf("  heads=%2d positions=%4d  %.3e  %s\n", heads, positions, worst,
                ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    std::printf("bailing_kda_recurrent_chunk vs bailing::kda_recurrence\n");
    bool ok = true;
    // Multi-token shapes matter most: the state carries between tokens, so a
    // mistake in the loop only shows up after the first.
    ok &= run(1, 1);
    ok &= run(4, 1);
    ok &= run(4, 5);
    ok &= run(16, 17);
    ok &= run(2, 64);
    std::printf("bailing_mla_attention vs bailing::mla_attention_absorbed\n");
    // Positions crossing the 128-thread stride matter: the softmax reductions
    // walk a strided range, so an off-by-one shows up only past one block.
    ok &= run_mla(1, 1);
    ok &= run_mla(4, 7);
    ok &= run_mla(16, 128);
    ok &= run_mla(8, 257);
    return ok ? 0 : 1;
}
