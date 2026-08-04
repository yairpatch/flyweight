#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// DeepSeek-V4 (`deepseek4`) building blocks, CPU reference implementations.
//
// The architecture replaces the plain residual with `hc` parallel streams. Each
// block reads a weighted collapse of the streams, and writes its output back
// into every stream through a learned mixing matrix that is Sinkhorn-normalized
// so the streams stay balanced.
namespace colibri::v2::deepseek4 {

// Plain RMS normalization with no gain, matching the reference's RMS_NORM over
// the flattened streams.
inline void rms_norm(const float* input, std::size_t size, float epsilon, float* output) {
    double total = 0.0;
    for (std::size_t i = 0; i < size; ++i) total += static_cast<double>(input[i]) * input[i];
    const float scale = 1.0f / std::sqrt(static_cast<float>(total / static_cast<double>(size)) + epsilon);
    for (std::size_t i = 0; i < size; ++i) output[i] = input[i] * scale;
}

inline float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

// Sinkhorn-normalize the [hc, hc] mixing matrix in place. `comb` is indexed
// [src*hc + dst], matching the reference's [dst, src] layout in memory.
//
// The schedule is softmax over dst, then epsilon, then one column
// normalization, then `iterations - 1` further row/column pairs. Every divisor
// carries the same epsilon, which matters: the sums are small and the epsilon
// is not negligible against them.
inline void sinkhorn(float* comb, std::size_t hc, std::uint32_t iterations, float epsilon) {
    for (std::size_t src = 0; src < hc; ++src) {
        float* row = comb + src * hc;
        const float peak = *std::max_element(row, row + hc);
        float total = 0.0f;
        for (std::size_t dst = 0; dst < hc; ++dst) {
            row[dst] = std::exp(row[dst] - peak);
            total += row[dst];
        }
        for (std::size_t dst = 0; dst < hc; ++dst) row[dst] = row[dst] / total + epsilon;
    }
    auto normalize_columns = [&] {
        for (std::size_t dst = 0; dst < hc; ++dst) {
            float total = epsilon;
            for (std::size_t src = 0; src < hc; ++src) total += comb[src * hc + dst];
            for (std::size_t src = 0; src < hc; ++src) comb[src * hc + dst] /= total;
        }
    };
    auto normalize_rows = [&] {
        for (std::size_t src = 0; src < hc; ++src) {
            float total = epsilon;
            for (std::size_t dst = 0; dst < hc; ++dst) total += comb[src * hc + dst];
            for (std::size_t dst = 0; dst < hc; ++dst) comb[src * hc + dst] /= total;
        }
    };
    normalize_columns();
    for (std::uint32_t iteration = 1; iteration < iterations; ++iteration) {
        normalize_rows();
        normalize_columns();
    }
}

// Derive the three mixing weights a block needs from its stream state.
//
// `streams` is [hc][n_embd]; `fn` is the [(2+hc)*hc] x [hc*n_embd] mixer with
// one output row per contiguous run; `scale` is 3 wide and `base` is
// 2*hc + hc*hc wide. `pre` collapses the streams for the block input, `post`
// weights the block output back into each stream, and `comb` carries each
// stream into each other.
inline void hyper_connection_weights(
    const float* streams,
    const float* fn,
    const float* scale,
    const float* base,
    std::size_t n_embd,
    std::size_t hc,
    std::uint32_t sinkhorn_iterations,
    float rms_epsilon,
    float hc_epsilon,
    float* pre,
    float* post,
    float* comb,
    float* mixes_out = nullptr
) {
    const std::size_t width = hc * n_embd;
    const std::size_t mix_dim = (2 + hc) * hc;

    std::vector<float> normalized(width);
    rms_norm(streams, width, rms_epsilon, normalized.data());

    std::vector<float> mixes(mix_dim);
    for (std::size_t row = 0; row < mix_dim; ++row) {
        const float* weights = fn + row * width;
        double total = 0.0;
        for (std::size_t i = 0; i < width; ++i) total += static_cast<double>(weights[i]) * normalized[i];
        mixes[row] = static_cast<float>(total);
    }
    if (mixes_out) std::copy(mixes.begin(), mixes.end(), mixes_out);

    for (std::size_t i = 0; i < hc; ++i)
        pre[i] = sigmoid(mixes[i] * scale[0] + base[i]) + hc_epsilon;
    for (std::size_t i = 0; i < hc; ++i)
        post[i] = sigmoid(mixes[hc + i] * scale[1] + base[hc + i]) * 2.0f;
    for (std::size_t i = 0; i < hc * hc; ++i)
        comb[i] = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
    sinkhorn(comb, hc, sinkhorn_iterations, hc_epsilon);
}

// Collapse the streams into the single vector a block consumes.
inline void hyper_connection_collapse(
    const float* streams, const float* pre, std::size_t n_embd, std::size_t hc, float* output
) {
    for (std::size_t i = 0; i < n_embd; ++i) output[i] = 0.0f;
    for (std::size_t stream = 0; stream < hc; ++stream) {
        const float weight = pre[stream];
        const float* source = streams + stream * n_embd;
        for (std::size_t i = 0; i < n_embd; ++i) output[i] += source[i] * weight;
    }
}

// Write a block's output back into every stream:
//   out[dst] = block * post[dst] + sum_src streams[src] * comb[dst, src]
inline void hyper_connection_combine(
    const float* block,
    const float* streams,
    const float* post,
    const float* comb,
    std::size_t n_embd,
    std::size_t hc,
    float* output
) {
    for (std::size_t dst = 0; dst < hc; ++dst) {
        float* target = output + dst * n_embd;
        const float weight = post[dst];
        for (std::size_t i = 0; i < n_embd; ++i) target[i] = block[i] * weight;
        for (std::size_t src = 0; src < hc; ++src) {
            const float mix = comb[src * hc + dst];
            const float* source = streams + src * n_embd;
            for (std::size_t i = 0; i < n_embd; ++i) target[i] += source[i] * mix;
        }
    }
}

// Rotary embedding over the trailing `rope_dim` of a row, pairing adjacent
// elements (ggml's NORM layout, which is what this architecture selects).
//
// Which parameters apply is per layer, not global: a block whose compress ratio
// is zero rotates at the model's own frequency base with no scaling, while the
// compressed blocks use a separate, much larger base together with YaRN. Passing
// `freq_scale` of 1 and no YaRN gives the first; the caller decides.
//
// `inverse` undoes the rotation, which the output path needs before projecting.
inline void rope(
    float* values,
    std::size_t rope_dim,
    std::int32_t position,
    float freq_base,
    float freq_scale,
    bool inverse
) {
    for (std::size_t i = 0; i + 1 < rope_dim; i += 2) {
        const float exponent = -static_cast<float>(i) / static_cast<float>(rope_dim);
        const float theta = static_cast<float>(position) * std::pow(freq_base, exponent) * freq_scale;
        const float cosine = std::cos(theta);
        const float sine = inverse ? -std::sin(theta) : std::sin(theta);
        const float first = values[i];
        const float second = values[i + 1];
        values[i] = first * cosine - second * sine;
        values[i + 1] = first * sine + second * cosine;
    }
}

// Attention over the shared KV latent, with one learned sink logit per head.
//
// This is MLA in absorbed form: keys and values are the same tensor, one
// `head_dim`-wide latent per position that every head reads. The sink joins the
// softmax as an extra logit carrying no value, so a head can attend to nothing
// by putting its mass there -- it damps the output rather than redistributing
// it. `mask` may be null; where present, a false entry hides that position.
inline void attention_with_sinks(
    const float* queries,
    const float* latents,
    const float* sinks,
    const std::uint8_t* mask,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t positions,
    float scale,
    float* output
) {
    std::vector<float> weights(positions);
    for (std::size_t head = 0; head < heads; ++head) {
        const float* query = queries + head * head_dim;
        float peak = sinks ? sinks[head] : -std::numeric_limits<float>::infinity();
        for (std::size_t position = 0; position < positions; ++position) {
            if (mask && !mask[position]) {
                weights[position] = -std::numeric_limits<float>::infinity();
                continue;
            }
            const float* latent = latents + position * head_dim;
            double total = 0.0;
            for (std::size_t i = 0; i < head_dim; ++i)
                total += static_cast<double>(query[i]) * latent[i];
            weights[position] = static_cast<float>(total) * scale;
            peak = std::max(peak, weights[position]);
        }
        float denominator = sinks ? std::exp(sinks[head] - peak) : 0.0f;
        for (std::size_t position = 0; position < positions; ++position) {
            weights[position] = weights[position] == -std::numeric_limits<float>::infinity()
                ? 0.0f
                : std::exp(weights[position] - peak);
            denominator += weights[position];
        }
        float* target = output + head * head_dim;
        for (std::size_t i = 0; i < head_dim; ++i) target[i] = 0.0f;
        for (std::size_t position = 0; position < positions; ++position) {
            const float weight = weights[position] / denominator;
            if (weight == 0.0f) continue;
            const float* latent = latents + position * head_dim;
            for (std::size_t i = 0; i < head_dim; ++i) target[i] += latent[i] * weight;
        }
    }
}

// The grouped half of the output projection. The head outputs are cut into
// `groups` contiguous chunks, and chunk g is multiplied by the g-th slice of the
// weight rather than by the whole matrix, so the projection costs a fraction of
// a dense one. `weights` is the file's [inputs, groups*rank] laid out
// output-major; `input` is groups*inputs wide and `output` groups*rank.
template <class Dot>
inline void grouped_projection(
    const float* input,
    std::size_t inputs,
    std::size_t rank,
    std::size_t groups,
    float* output,
    Dot row_dot
) {
    for (std::size_t group = 0; group < groups; ++group) {
        const float* source = input + group * inputs;
        for (std::size_t index = 0; index < rank; ++index)
            output[group * rank + index] = row_dot(source, group * rank + index);
    }
}

}  // namespace colibri::v2::deepseek4
