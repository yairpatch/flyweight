// Contract for the BailingMoE3 KDA recurrence.
//
// Expected values come from native/tools/kda_reference.py, which is itself
// pinned to flash-linear-attention's `naive_recurrent_kda` / `naive_chunk_kda`
// by kda_reference_check.py. They were not produced by running this
// implementation.
//
// Beyond the numeric case, this pins the property the runtime depends on:
// feeding rows one at a time with the state carried forward must equal feeding
// them all at once. Prefill and decode share this kernel, so if that ever
// breaks, a conversation silently diverges from its own prefill.

#include "flyweight_v2_bailing.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

constexpr std::size_t kRows = 3, kHeads = 2, kHeadDim = 4;
constexpr std::size_t kWidth = kHeads * kHeadDim;

const float queries[] = {0.466000f, -0.106000f, -3.285000f, -0.875000f, -1.078000f, 0.279000f, 0.652000f, 1.627000f, 0.763000f, 0.414000f, -0.686000f, -0.094000f, -0.978000f, 0.183000f, 0.105000f, 0.965000f, 0.647000f, 1.016000f, 1.971000f, 0.416000f, -0.894000f, -2.046000f, -0.337000f, -0.241000f};
const float keys[] = {-0.625000f, -1.262000f, 0.646000f, 0.825000f, 0.160000f, -0.180000f, -0.336000f, -1.413000f, 0.634000f, 0.526000f, -0.077000f, 0.424000f, -0.451000f, -1.276000f, 0.728000f, -0.733000f, -0.754000f, -0.008000f, 0.785000f, 1.155000f, 2.025000f, -1.410000f, -1.388000f, 0.805000f};
const float values[] = {-1.458000f, 1.594000f, 1.465000f, 0.176000f, -0.672000f, 0.182000f, -2.463000f, 0.526000f, 0.502000f, -0.511000f, 0.580000f, 0.226000f, 1.422000f, -0.664000f, -0.092000f, 1.386000f, -0.407000f, -0.635000f, 0.181000f, 0.672000f, -0.914000f, -0.627000f, 0.472000f, 1.005000f};
const float gate_raw[] = {1.575000f, -1.167000f, 1.405000f, -1.117000f, -0.994000f, 1.287000f, 0.503000f, 0.660000f, 0.124000f, 0.794000f, -0.469000f, -0.728000f, -1.029000f, -0.171000f, -0.444000f, 0.544000f, 0.618000f, -0.050000f, -0.851000f, 0.135000f, -0.669000f, 0.102000f, -1.453000f, 1.545000f};
const float beta_logits[] = {0.662000f, 0.872000f, 1.839000f, -1.072000f, 1.211000f, -1.546000f};
const float a_log[] = {1.118000f, -0.724000f};
const float dt_bias[] = {-0.805000f, -0.483000f, -1.106000f, 1.530000f, 1.945000f, -1.292000f, -0.461000f, 0.087000f};
const float expected[] = {0.239543f, -0.261887f, -0.240693f, -0.028916f, 0.212407f, -0.057527f, 0.778511f, -0.166259f, 0.223263f, -0.232794f, 0.099446f, 0.058621f, 0.076383f, -0.014326f, 0.400930f, -0.116787f, -0.092343f, -0.069017f, 0.088411f, 0.118205f, 0.071923f, -0.056432f, -0.076789f, 0.141126f};

bool matches_the_reference() {
    std::vector<float> state(kHeads * kHeadDim * kHeadDim, 0.0f);
    std::vector<float> output(kRows * kWidth, 0.0f);
    flyweight::v2::bailing::kda_recurrence(
        queries, keys, values, gate_raw, beta_logits, a_log, dt_bias,
        kRows, kHeads, kHeadDim, 1e-6f, state.data(), output.data());
    for (std::size_t i = 0; i < kRows * kWidth; ++i)
        if (std::fabs(output[i] - expected[i]) > 1e-5f) return false;
    return true;
}

// Row-at-a-time with the state threaded through must equal all-at-once.
bool decode_matches_prefill() {
    std::vector<float> batched_state(kHeads * kHeadDim * kHeadDim, 0.0f);
    std::vector<float> batched(kRows * kWidth, 0.0f);
    flyweight::v2::bailing::kda_recurrence(
        queries, keys, values, gate_raw, beta_logits, a_log, dt_bias,
        kRows, kHeads, kHeadDim, 1e-6f, batched_state.data(), batched.data());

    std::vector<float> stepped_state(kHeads * kHeadDim * kHeadDim, 0.0f);
    std::vector<float> stepped(kRows * kWidth, 0.0f);
    for (std::size_t row = 0; row < kRows; ++row)
        flyweight::v2::bailing::kda_recurrence(
            queries + row * kWidth, keys + row * kWidth, values + row * kWidth,
            gate_raw + row * kWidth, beta_logits + row * kHeads,
            a_log, dt_bias, 1, kHeads, kHeadDim, 1e-6f,
            stepped_state.data(), stepped.data() + row * kWidth);

    for (std::size_t i = 0; i < kRows * kWidth; ++i)
        if (batched[i] != stepped[i]) return false;
    for (std::size_t i = 0; i < batched_state.size(); ++i)
        if (batched_state[i] != stepped_state[i]) return false;
    return true;
}

// The decay must always contract: the gate is -exp(A_log) * softplus(...),
// which is negative for every finite input, so exp(gate) < 1. A sign slip here
// would make the state grow without bound over a long context instead of
// failing outright. Drive it with a large positive gate, where a wrong sign is
// unmissable after a few hundred steps.
bool the_state_decays_rather_than_grows() {
    const std::size_t heads = 1, dim = 4, rows = 1;
    const float q[dim] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float k[dim] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float v[dim] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float gate[dim] = {8.0f, 8.0f, 8.0f, 8.0f};
    const float beta[1] = {4.0f};
    const float coefficient[1] = {0.5f};
    const float bias[dim] = {0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<float> state(dim * dim, 0.0f);
    std::vector<float> output(dim, 0.0f);
    float previous = 0.0f;
    for (int step = 0; step < 256; ++step) {
        flyweight::v2::bailing::kda_recurrence(
            q, k, v, gate, beta, coefficient, bias,
            rows, heads, dim, 1e-6f, state.data(), output.data());
        float magnitude = 0.0f;
        for (const float element : state) magnitude += std::fabs(element);
        if (!std::isfinite(magnitude) || magnitude > 1e3f) return false;
        previous = magnitude;
    }
    return previous > 0.0f;
}

}  // namespace

int main() {
    if (!matches_the_reference()) return 1;
    if (!decode_matches_prefill()) return 2;
    if (!the_state_decays_rather_than_grows()) return 3;
    return 0;
}
