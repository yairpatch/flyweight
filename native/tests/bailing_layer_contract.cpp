// Contract for the BailingMoE3 decoder layer.
//
// These are structural properties, not numeric parity. The numeric oracle --
// the real checkpoint under torch -- is currently unusable on this machine: it
// returns different activations on each process and roughly one run in three is
// NaN, with identical weights and inputs (see plans/hf-safetensors-bailingmoe3.md).
// Comparing against it would measure its corruption, not this code.
//
// So what is pinned here is everything that can be established without it: the
// residual structure, that state actually carries between tokens, and that the
// two layer kinds route to the two different attention implementations. The
// arithmetic inside each piece is already pinned against torch by the router,
// KDA and attention contracts.

#include "colibri_v2_bailing.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using colibri::v2::bailing::Geometry;
using colibri::v2::bailing::LayerCache;
using colibri::v2::bailing::LayerWeights;

Geometry small_geometry() {
    Geometry g;
    g.hidden = 8;
    g.heads = 2;
    g.head_dim = 4;
    g.qk_nope = 4;
    g.qk_rope = 4;
    g.v_head_dim = 4;
    g.q_lora = 4;
    g.kv_lora = 4;
    g.conv_width = 4;
    g.dense_size = 6;
    g.expert_size = 6;
    g.shared_size = 6;
    g.experts = 4;
    g.experts_used = 2;
    g.groups = 2;
    g.groups_used = 1;
    g.rope_theta = 10000.0f;
    g.epsilon = 1e-6f;
    g.weight_scale = 1.0f;
    g.normalize_weights = true;
    return g;
}

// Every projection zeroed means both branches contribute nothing, so a layer
// must be the identity. This is the residual structure on its own: it fails if
// either join is dropped, doubled, or applied to the normalized value instead
// of the raw input.
bool a_layer_with_no_weights_is_the_identity() {
    const Geometry g = small_geometry();
    for (const bool full : {true, false}) {
        std::vector<float> zeros(4096, 0.0f);
        std::vector<float> ones(g.hidden, 1.0f);
        LayerWeights w;
        w.full_attention = full;
        w.routed = false;
        w.attention_norm = ones.data();
        w.ffn_norm = ones.data();
        w.mla = {zeros.data(), zeros.data(), zeros.data(), zeros.data(),
                 zeros.data(), zeros.data(), zeros.data(), zeros.data()};
        w.kda = {zeros.data(), zeros.data(), zeros.data(), zeros.data(),
                 zeros.data(), zeros.data(), zeros.data(), zeros.data(),
                 zeros.data(), zeros.data(), zeros.data(), zeros.data(),
                 zeros.data()};
        w.ffn.gate = zeros.data();
        w.ffn.up = zeros.data();
        w.ffn.down = zeros.data();

        LayerCache cache;
        reset_cache(cache, w, g, 4);
        std::vector<float> input(g.hidden), output(g.hidden);
        for (std::size_t i = 0; i < g.hidden; ++i)
            input[i] = 0.5f * static_cast<float>(i) - 1.0f;
        decoder_layer(input.data(), w, g, 0, cache, output.data());
        for (std::size_t i = 0; i < g.hidden; ++i)
            if (std::fabs(output[i] - input[i]) > 1e-6f) return false;
    }
    return true;
}

// Build a layer whose weights are deterministic but not degenerate.
struct Fixture {
    std::vector<float> pool;
    std::vector<float> norm;
    LayerWeights weights;

    explicit Fixture(const Geometry& g, bool full) {
        pool.resize(64 * 1024);
        for (std::size_t i = 0; i < pool.size(); ++i)
            pool[i] = 0.13f * std::sin(0.37f * static_cast<float>(i) + 1.1f);
        norm.assign(std::max(g.hidden, g.head_dim), 1.0f);
        auto at = [&](std::size_t offset) { return pool.data() + offset; };
        weights.full_attention = full;
        weights.routed = false;
        weights.attention_norm = norm.data();
        weights.ffn_norm = norm.data();
        weights.mla = {at(0), norm.data(), at(500), at(1500), norm.data(),
                       at(2500), at(4000), at(5000)};
        weights.kda = {at(0),    at(300),  at(600),  at(900),  at(1000),
                       at(1100), at(1200), at(1500), at(1800), at(2100),
                       at(2200), norm.data(), at(2400)};
        weights.ffn.gate = at(6000);
        weights.ffn.up = at(6500);
        weights.ffn.down = at(7000);
    }
};

// Feeding tokens one at a time must leave the cache in a state that depends on
// what came before -- otherwise the layer is stateless and every token attends
// only to itself. Checked by running the same token at position 1 against two
// different histories and requiring the results to differ.
bool history_changes_the_result() {
    const Geometry g = small_geometry();
    for (const bool full : {true, false}) {
        Fixture fixture(g, full);
        std::vector<float> first(g.hidden), second(g.hidden), probe(g.hidden);
        for (std::size_t i = 0; i < g.hidden; ++i) {
            first[i] = 0.4f * static_cast<float>(i % 5) - 0.7f;
            second[i] = -0.9f * static_cast<float>(i % 3) + 0.5f;
            probe[i] = 0.25f * static_cast<float>(i % 7) - 0.3f;
        }
        std::vector<float> outputs[2];
        for (int trial = 0; trial < 2; ++trial) {
            LayerCache cache;
            reset_cache(cache, fixture.weights, g, 8);
            std::vector<float> scratch(g.hidden);
            const std::vector<float>& history = trial == 0 ? first : second;
            decoder_layer(history.data(), fixture.weights, g, 0, cache, scratch.data());
            outputs[trial].resize(g.hidden);
            decoder_layer(probe.data(), fixture.weights, g, 1, cache,
                          outputs[trial].data());
        }
        float difference = 0.0f;
        for (std::size_t i = 0; i < g.hidden; ++i)
            difference += std::fabs(outputs[0][i] - outputs[1][i]);
        if (difference < 1e-4f) return false;
    }
    return true;
}

// A KDA layer's cache must not grow with context -- that is the entire reason
// the architecture is 3:1 in KDA's favour -- while an MLA layer's must.
bool only_the_mla_cache_grows_with_context() {
    const Geometry g = small_geometry();
    Fixture linear(g, false), full(g, true);
    LayerCache linear_cache, full_cache;
    reset_cache(linear_cache, linear.weights, g, 8);
    reset_cache(full_cache, full.weights, g, 8);
    const std::size_t linear_bytes =
        linear_cache.state.size() + linear_cache.query_window.size() +
        linear_cache.key_window.size() + linear_cache.value_window.size();

    std::vector<float> token(g.hidden), scratch(g.hidden);
    for (std::size_t i = 0; i < g.hidden; ++i) token[i] = 0.2f * static_cast<float>(i) - 0.6f;
    for (std::size_t position = 0; position < 6; ++position) {
        decoder_layer(token.data(), linear.weights, g, position, linear_cache,
                      scratch.data());
        decoder_layer(token.data(), full.weights, g, position, full_cache,
                      scratch.data());
    }
    const std::size_t after =
        linear_cache.state.size() + linear_cache.query_window.size() +
        linear_cache.key_window.size() + linear_cache.value_window.size();
    if (after != linear_bytes) return false;          // KDA state is constant
    return full_cache.positions == 6;                  // MLA grew
}

// The two layer kinds must actually take different paths. Same weights pool,
// same input, same position: if the dispatch were ignored the outputs would
// coincide.
bool the_two_layer_kinds_differ() {
    const Geometry g = small_geometry();
    Fixture linear(g, false), full(g, true);
    std::vector<float> token(g.hidden), a(g.hidden), b(g.hidden);
    for (std::size_t i = 0; i < g.hidden; ++i) token[i] = 0.3f * static_cast<float>(i) - 0.5f;
    LayerCache linear_cache, full_cache;
    reset_cache(linear_cache, linear.weights, g, 4);
    reset_cache(full_cache, full.weights, g, 4);
    decoder_layer(token.data(), linear.weights, g, 0, linear_cache, a.data());
    decoder_layer(token.data(), full.weights, g, 0, full_cache, b.data());
    float difference = 0.0f;
    for (std::size_t i = 0; i < g.hidden; ++i) difference += std::fabs(a[i] - b[i]);
    return difference > 1e-4f;
}

// Every output must stay finite over a run long enough for a runaway state to
// show up. The KDA decay is a contraction, so the recurrent state cannot grow
// without bound unless its sign is wrong.
bool a_long_run_stays_finite() {
    const Geometry g = small_geometry();
    for (const bool full : {true, false}) {
        Fixture fixture(g, full);
        LayerCache cache;
        reset_cache(cache, fixture.weights, g, 128);
        std::vector<float> token(g.hidden), output(g.hidden);
        for (std::size_t position = 0; position < 128; ++position) {
            for (std::size_t i = 0; i < g.hidden; ++i)
                token[i] = std::sin(0.3f * static_cast<float>(position + i));
            decoder_layer(token.data(), fixture.weights, g, position, cache,
                          output.data());
            for (const float value : output)
                if (!std::isfinite(value) || std::fabs(value) > 1e6f) return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    if (!a_layer_with_no_weights_is_the_identity()) return 1;
    if (!history_changes_the_result()) return 2;
    if (!only_the_mla_cache_grows_with_context()) return 3;
    if (!the_two_layer_kinds_differ()) return 4;
    if (!a_long_run_stays_finite()) return 5;
    return 0;
}
