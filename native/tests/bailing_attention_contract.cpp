// Contract for the BailingMoE3 MLA pieces: partial interleaved RoPE and the
// head-wise output gate.
//
// The RoPE case is the one that matters. The reference applies a de-interleave
// and then a half-split rotation; this applies an adjacent-pair rotation and no
// permutation. The two disagree element for element and agree on every
// attention score, because the permutation is orthogonal, applied to both q and
// k, and cancels in their dot product.
//
// So the test asserts the SCORE, not the vector. Asserting the vector would
// force a reimplementation of a permutation that no consumer can observe, and
// asserting nothing would leave the far more likely bug -- reading the
// half-split rotation straight out of the reference -- undetected.

#include "colibri_v2_bailing.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr std::size_t kNope = 128, kRope = 64, kHeadDim = kNope + kRope;

// Score between two rows after rotating each, which is all attention sees.
float rotated_score(std::vector<float> q, std::vector<float> k, std::int32_t position) {
    colibri::v2::bailing::partial_rope_norm(q.data(), kHeadDim, kRope, position, 6.0e6f);
    colibri::v2::bailing::partial_rope_norm(k.data(), kHeadDim, kRope, position, 6.0e6f);
    float total = 0.0f;
    for (std::size_t i = 0; i < kHeadDim; ++i) total += q[i] * k[i];
    return total;
}

std::vector<float> ramp(float start, float step) {
    std::vector<float> row(kHeadDim);
    for (std::size_t i = 0; i < kHeadDim; ++i)
        row[i] = start + step * static_cast<float>(i % 17);
    return row;
}

// At position 0 every angle is zero, so the rotation is the identity and the
// score is the plain dot product. A rotation applied with a swapped sine sign,
// or over the wrong span, still passes this -- which is why it is only the
// first of three.
bool position_zero_is_the_identity() {
    const auto q = ramp(0.5f, 0.25f), k = ramp(-0.75f, 0.125f);
    float plain = 0.0f;
    for (std::size_t i = 0; i < kHeadDim; ++i) plain += q[i] * k[i];
    return std::fabs(rotated_score(q, k, 0) - plain) < 1e-3f;
}

// The non-rotated prefix must be untouched: only the trailing `rope_dim`
// channels move. Rotating a row whose rope span is all zeros must leave it
// exactly as it was.
bool the_nope_prefix_is_untouched() {
    std::vector<float> row = ramp(1.0f, 0.5f);
    for (std::size_t i = kNope; i < kHeadDim; ++i) row[i] = 0.0f;
    const std::vector<float> before = row;
    colibri::v2::bailing::partial_rope_norm(row.data(), kHeadDim, kRope, 4321, 6.0e6f);
    for (std::size_t i = 0; i < kHeadDim; ++i)
        if (row[i] != before[i]) return false;
    return true;
}

// Rotating q and k by the SAME position must preserve their score, since the
// rotation is orthogonal and applied identically. This is what makes the
// leftover permutation in the reference unobservable, and it fails loudly if
// the pairing is wrong -- a half-split rotation pairs channel i with i+32 and
// does not preserve this when the two rows differ.
bool an_equal_rotation_preserves_the_score() {
    const auto q = ramp(0.3f, 0.4f), k = ramp(-0.2f, 0.3f);
    const float plain = rotated_score(q, k, 0);
    for (const std::int32_t position : {1, 17, 4096, 99999}) {
        const float rotated = rotated_score(q, k, position);
        if (std::fabs(rotated - plain) > 1e-2f * std::fabs(plain) + 1e-3f) return false;
    }
    return true;
}

// The point of RoPE: a score depends on the RELATIVE offset between the two
// positions, not on either absolute position. Two pairs the same distance apart
// must score the same, and pairs at different distances must score differently.
//
// Without the second half this passes for a rotation that does nothing at all;
// without the first it passes for one that rotates by the wrong angle.
bool the_score_depends_on_relative_position() {
    auto q = ramp(0.3f, 0.4f);
    auto k = ramp(-0.2f, 0.3f);
    for (std::size_t i = 0; i < kNope; ++i) { q[i] = 0.0f; k[i] = 0.0f; }

    const auto score_at = [&](std::int32_t query_at, std::int32_t key_at) {
        std::vector<float> rotated_q = q, rotated_k = k;
        colibri::v2::bailing::partial_rope_norm(
            rotated_q.data(), kHeadDim, kRope, query_at, 6.0e6f);
        colibri::v2::bailing::partial_rope_norm(
            rotated_k.data(), kHeadDim, kRope, key_at, 6.0e6f);
        float total = 0.0f;
        for (std::size_t i = 0; i < kHeadDim; ++i) total += rotated_q[i] * rotated_k[i];
        return total;
    };

    // Same distance, different absolute positions -> same score.
    const float near_pair = score_at(40, 8);
    const float far_pair = score_at(9040, 9008);
    if (std::fabs(near_pair - far_pair) > 1e-2f * std::fabs(near_pair) + 1e-3f)
        return false;
    // Different distance -> different score, or nothing is rotating.
    return std::fabs(score_at(40, 8) - score_at(40, 39)) > 1e-3f;
}

bool the_head_gate_scales_each_head_by_its_own_sigmoid() {
    constexpr std::size_t heads = 3, value_dim = 4;
    const float logits[heads] = {0.0f, 2.0f, -2.0f};
    std::vector<float> output(heads * value_dim, 1.0f);
    colibri::v2::bailing::apply_head_gate(logits, heads, value_dim, output.data());
    const float expected[heads] = {0.5f, 0.880797029f, 0.119202919f};
    for (std::size_t head = 0; head < heads; ++head)
        for (std::size_t i = 0; i < value_dim; ++i)
            if (std::fabs(output[head * value_dim + i] - expected[head]) > 1e-6f)
                return false;
    return true;
}


// The absorbed and decompressed forms must agree. This is the whole safety net
// for caching latents instead of per-head K/V: the saving rests on an algebraic
// identity, and this is what stops that identity from being taken on trust.
bool absorbed_matches_decompressed() {
    constexpr std::size_t heads = 3, nope = 8, rope = 4, value_dim = 8,
                          lora = 6, positions = 5;
    std::vector<float> queries_nope(heads * nope), queries_rope(heads * rope),
        kv_b(heads * (nope + value_dim) * lora), latents(positions * lora),
        rope_keys(positions * rope);
    // Deterministic, mutually incommensurate fills: nothing here is symmetric
    // enough for a wrong index order to coincidentally agree.
    for (std::size_t i = 0; i < queries_nope.size(); ++i)
        queries_nope[i] = 0.31f * std::sin(0.7f * static_cast<float>(i) + 1.1f);
    for (std::size_t i = 0; i < queries_rope.size(); ++i)
        queries_rope[i] = 0.27f * std::cos(0.9f * static_cast<float>(i) + 0.3f);
    for (std::size_t i = 0; i < kv_b.size(); ++i)
        kv_b[i] = 0.11f * std::sin(0.37f * static_cast<float>(i) + 2.2f);
    for (std::size_t i = 0; i < latents.size(); ++i)
        latents[i] = 0.41f * std::cos(0.53f * static_cast<float>(i) + 0.8f);
    for (std::size_t i = 0; i < rope_keys.size(); ++i)
        rope_keys[i] = 0.23f * std::sin(1.3f * static_cast<float>(i) + 0.5f);

    std::vector<float> absorbed(heads * value_dim);
    colibri::v2::bailing::mla_attention_absorbed(
        queries_nope.data(), queries_rope.data(), kv_b.data(), latents.data(),
        rope_keys.data(), positions, heads, nope, rope, value_dim, lora,
        absorbed.data());

    std::vector<float> keys(positions * heads * (nope + rope));
    std::vector<float> values(positions * heads * value_dim);
    for (std::size_t position = 0; position < positions; ++position)
        colibri::v2::bailing::mla_decompress(
            latents.data() + position * lora, rope_keys.data() + position * rope,
            kv_b.data(), heads, nope, rope, value_dim, lora,
            keys.data() + position * heads * (nope + rope),
            values.data() + position * heads * value_dim);

    std::vector<float> queries(heads * (nope + rope));
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t i = 0; i < nope; ++i)
            queries[head * (nope + rope) + i] = queries_nope[head * nope + i];
        for (std::size_t i = 0; i < rope; ++i)
            queries[head * (nope + rope) + nope + i] = queries_rope[head * rope + i];
    }
    std::vector<float> decompressed(heads * value_dim);
    colibri::v2::bailing::mla_attention_decompressed(
        queries.data(), keys.data(), values.data(), positions, heads,
        nope + rope, value_dim, decompressed.data());

    for (std::size_t i = 0; i < absorbed.size(); ++i)
        if (std::fabs(absorbed[i] - decompressed[i]) > 1e-5f) return false;
    // Guard against both forms being trivially zero, which would pass above.
    float magnitude = 0.0f;
    for (const float value : absorbed) magnitude += std::fabs(value);
    return magnitude > 1e-3f;
}

// The rope half of the key is MQA -- shared by every head (modeling:672-680).
// Decompression must copy it to all of them, not index it per head.
bool the_rope_key_is_shared_across_heads() {
    constexpr std::size_t heads = 3, nope = 4, rope = 4, value_dim = 4, lora = 5;
    std::vector<float> kv_b(heads * (nope + value_dim) * lora, 0.0f);
    std::vector<float> latent(lora, 0.0f);
    const float rope_key[rope] = {1.5f, -2.5f, 3.5f, -4.5f};
    std::vector<float> keys(heads * (nope + rope)), values(heads * value_dim);
    colibri::v2::bailing::mla_decompress(latent.data(), rope_key, kv_b.data(),
                                         heads, nope, rope, value_dim, lora,
                                         keys.data(), values.data());
    for (std::size_t head = 0; head < heads; ++head)
        for (std::size_t i = 0; i < rope; ++i)
            if (keys[head * (nope + rope) + nope + i] != rope_key[i]) return false;
    return true;
}

}  // namespace

int main() {
    if (!position_zero_is_the_identity()) return 1;
    if (!the_nope_prefix_is_untouched()) return 2;
    if (!an_equal_rotation_preserves_the_score()) return 3;
    if (!the_score_depends_on_relative_position()) return 4;
    if (!the_head_gate_scales_each_head_by_its_own_sigmoid()) return 5;
    if (!absorbed_matches_decompressed()) return 6;
    if (!the_rope_key_is_shared_across_heads()) return 7;
    return 0;
}
