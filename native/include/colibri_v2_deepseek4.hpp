#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// DeepSeek-V4 (`deepseek4`) building blocks, CPU reference implementations.
//
// The architecture replaces the plain residual with `hc` parallel streams. Each
// block reads a weighted collapse of the streams, and writes its output back
// into every stream through a learned mixing matrix that is Sinkhorn-normalized
// so the streams stay balanced.
namespace colibri::v2::deepseek4 {

// How wide to fork a loop over rows or heads.
//
// One thread per core, not per hardware thread. Measured on this checkpoint:
// the default team on a 16-core/32-thread part runs the expert weights at
// 9.3 GiB/s where a 16-wide team reaches 24.6, because two threads sharing a
// core contend for one L1 and one set of decode units. An explicit
// OMP_NUM_THREADS still wins, which is how a part without SMT gets its cores
// back. The rule matches `qwen_cpu_thread_count`; this is the copy the kernels
// can see.
inline int thread_count() {
#if defined(_OPENMP)
    static const int team = [] {
        int chosen = omp_get_max_threads();
        if (std::getenv("OMP_NUM_THREADS") == nullptr) {
            const int physical = omp_get_num_procs() / 2;
            if (physical >= 1 && chosen > physical) chosen = physical;
        }
        return chosen;
    }();
    return team;
#else
    return 1;
#endif
}

// Plain RMS normalization with no gain, matching the reference's RMS_NORM over
// the flattened streams.
inline void rms_norm(const float* input, std::size_t size, float epsilon, float* output) {
    double total = 0.0;
    for (std::size_t i = 0; i < size; ++i) total += static_cast<double>(input[i]) * input[i];
    const float scale = 1.0f / std::sqrt(static_cast<float>(total / static_cast<double>(size)) + epsilon);
    for (std::size_t i = 0; i < size; ++i) output[i] = input[i] * scale;
}

inline float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

// Float to half, round to nearest even, matching what ggml stores.
//
// The caches hold half precision because the reference does, and the composed
// model already rounds its latents the same way to stay numerically alongside
// it. Storing anything wider here would not buy accuracy -- the value written
// has been through half precision either way -- it would only cost memory.
inline std::uint16_t half_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu)  // inf or NaN, kept as such
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    if (exponent >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);  // underflows to zero
        mantissa |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
        const std::uint32_t half = 1u << (shift - 1);
        const std::uint32_t sticky = (mantissa & ((half << 1) - 1)) != half ? 0u : 1u;
        const std::uint32_t rounded =
            (mantissa + half - (sticky ? ((mantissa >> shift) & 1u) ^ 1u : 0u)) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded = mantissa + 0x0FFFu + ((mantissa >> 13) & 1u);
    if (rounded & 0x800000u) {
        ++exponent;
        if (exponent >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10) | ((rounded >> 13) & 0x3FFu));
}

inline float half_value(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1Fu;
    std::uint32_t mantissa = bits & 0x3FFu;
    std::uint32_t out;
    if (exponent == 0) {
        if (!mantissa) out = sign;
        else {
            std::int32_t shift = 0;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; ++shift; }
            mantissa &= 0x3FFu;
            out = sign | (static_cast<std::uint32_t>(127 - 15 - shift + 1) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        out = sign | 0x7F800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

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
    // Twenty-four rows of sixteen thousand weights, twice a layer: a megabyte
    // and a half of reading each time, which is worth the fork even though the
    // row count is small.
#pragma omp parallel for schedule(static) num_threads(thread_count())
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(mix_dim); ++row) {
        const float* weights = fn + static_cast<std::size_t>(row) * width;
        double total = 0.0;
        for (std::size_t i = 0; i < width; ++i) total += static_cast<double>(weights[i]) * normalized[i];
        mixes[static_cast<std::size_t>(row)] = static_cast<float>(total);
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

// Collapse the streams for the output head.
//
// The head only ever reads the streams, so its mixer produces just the `hc`
// pre-weights rather than the pre/post/comb triple a block needs -- hence a
// [hc, hc*n_embd] mixer with a single scale, against the block's
// [(2+hc)*hc, hc*n_embd] with three.
inline void hyper_connection_head(
    const float* streams,
    const float* fn,
    const float* scale,
    const float* base,
    std::size_t n_embd,
    std::size_t hc,
    float rms_epsilon,
    float hc_epsilon,
    float* pre,
    float* output
) {
    const std::size_t width = hc * n_embd;
    std::vector<float> normalized(width);
    rms_norm(streams, width, rms_epsilon, normalized.data());
    for (std::size_t row = 0; row < hc; ++row) {
        const float* weights = fn + row * width;
        double total = 0.0;
        for (std::size_t i = 0; i < width; ++i)
            total += static_cast<double>(weights[i]) * normalized[i];
        pre[row] = sigmoid(static_cast<float>(total) * scale[0] + base[row]) + hc_epsilon;
    }
    for (std::size_t i = 0; i < n_embd; ++i) output[i] = 0.0f;
    for (std::size_t stream = 0; stream < hc; ++stream) {
        const float weight = pre[stream];
        const float* source = streams + stream * n_embd;
        for (std::size_t i = 0; i < n_embd; ++i) output[i] += source[i] * weight;
    }
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

// Gather the rows one block pools, ready for `compress_block`.
//
// A 4:1 layer overlaps: it pools the previous block's `ratio` rows taking the
// first half of each state row, then its own `ratio` rows taking the second
// half. Before the sequence starts there is no previous block, so those rows
// are left at zero value and -inf score, which the softmax drops. A 128:1 layer
// does not overlap and pools its own rows whole.
//
// `values` and `scores` are [positions][width] where width is 2*head_dim when
// overlapped and head_dim otherwise. Outputs are [rows][head_dim] with rows
// 2*ratio when overlapped, ratio otherwise.
inline std::size_t gather_block(
    const float* values,
    const float* scores,
    std::size_t width,
    std::size_t head_dim,
    std::size_t ratio,
    std::size_t block,
    bool overlapped,
    float* out_values,
    float* out_scores
) {
    const std::size_t rows = overlapped ? 2 * ratio : ratio;
    if (!overlapped) {
        for (std::size_t slot = 0; slot < ratio; ++slot) {
            const std::size_t source = (block * ratio + slot) * width;
            std::copy_n(values + source, head_dim, out_values + slot * head_dim);
            std::copy_n(scores + source, head_dim, out_scores + slot * head_dim);
        }
        return rows;
    }
    for (std::size_t i = 0; i < rows * head_dim; ++i) {
        out_values[i] = 0.0f;
        out_scores[i] = -std::numeric_limits<float>::infinity();
    }
    for (std::size_t slot = 0; slot < ratio; ++slot) {
        if (block > 0) {
            // The previous block contributes the first half of its rows.
            const std::size_t source = ((block - 1) * ratio + slot) * width;
            std::copy_n(values + source, head_dim, out_values + slot * head_dim);
            std::copy_n(scores + source, head_dim, out_scores + slot * head_dim);
        }
        // This block contributes the second half of its own.
        const std::size_t source = (block * ratio + slot) * width + head_dim;
        std::copy_n(values + source, head_dim, out_values + (ratio + slot) * head_dim);
        std::copy_n(scores + source, head_dim, out_scores + (ratio + slot) * head_dim);
    }
    return rows;
}

// Which keys a query at `position` may attend to.
//
// The raw sliding window and the compressed blocks are both visible and
// deliberately overlap: a token can be attended directly and again through its
// block's summary. A block becomes visible once every token it covers is at or
// before the query, so block b is available from b*ratio + ratio - 1 onward.
//
// Fills `mask` with `raw_positions + blocks` entries, raw first, and returns
// how many are visible. A zero ratio means the layer compresses nothing, in
// which case `blocks` should be zero.
inline std::size_t visible_keys(
    std::size_t position,
    std::size_t raw_positions,
    std::size_t blocks,
    std::size_t ratio,
    std::size_t window,
    std::uint8_t* mask
) {
    std::size_t visible = 0;
    const std::size_t first = window && position + 1 > window ? position + 1 - window : 0;
    for (std::size_t i = 0; i < raw_positions; ++i) {
        const bool seen = i >= first && i <= position;
        mask[i] = seen ? 1 : 0;
        visible += seen;
    }
    for (std::size_t b = 0; b < blocks; ++b) {
        const bool seen = ratio && b * ratio + ratio - 1 <= position;
        mask[raw_positions + b] = seen ? 1 : 0;
        visible += seen;
    }
    return visible;
}

// Compress a block of positions into a single latent.
//
// Both compressed attention kinds pool a run of tokens -- four for CSA, 128 for
// HCA -- into one latent that later positions attend to instead of the tokens
// themselves. The pooling is a softmax-weighted average taken *per channel*,
// not per position: each of the `width` channels softmaxes its own scores over
// the block and mixes the values accordingly, so different channels can draw
// from different tokens in the same block.
//
// `values` and `scores` are [positions][width], row-major. The scores already
// include the absolute position embedding for the slot within the block.
inline void compress_block(
    const float* values,
    const float* scores,
    std::size_t positions,
    std::size_t width,
    float* output
) {
    for (std::size_t channel = 0; channel < width; ++channel) {
        float peak = -std::numeric_limits<float>::infinity();
        for (std::size_t position = 0; position < positions; ++position)
            peak = std::max(peak, scores[position * width + channel]);
        float total = 0.0f;
        float mixed = 0.0f;
        for (std::size_t position = 0; position < positions; ++position) {
            const float weight = std::exp(scores[position * width + channel] - peak);
            total += weight;
            mixed += weight * values[position * width + channel];
        }
        output[channel] = total > 0.0f ? mixed / total : 0.0f;
    }
}

// Expert routing for one token.
//
// The probabilities are sqrt(softplus(logits)) -- not the sigmoid that
// `expert_gating_func` and the DeepSeek-V3 lineage would suggest. Selection
// uses a bias-corrected score, but the weights are gathered from the
// *unbiased* probabilities, so the bias steers which experts are chosen without
// distorting how much each contributes. The chosen weights are then normalized
// to sum to one and scaled.
//
// `select` false is the hash-layer case: those blocks read their expert ids
// from an int32 table indexed by token id, so `chosen` arrives filled in and
// only the weights are computed.
inline void moe_router(
    const float* logits,
    const float* bias,
    std::size_t experts,
    std::size_t used,
    float weight_scale,
    float sum_floor,
    bool select,
    std::int32_t* chosen,
    float* weights
) {
    std::vector<float> probabilities(experts);
    for (std::size_t expert = 0; expert < experts; ++expert) {
        const float logit = logits[expert];
        // softplus, guarded the way a stable implementation must be: for large
        // logits it is the identity, and expf would overflow.
        const float softplus = logit > 20.0f ? logit : std::log1p(std::exp(logit));
        probabilities[expert] = std::sqrt(softplus);
    }
    if (select) {
        std::vector<std::int32_t> order(experts);
        for (std::size_t expert = 0; expert < experts; ++expert)
            order[expert] = static_cast<std::int32_t>(expert);
        const auto score = [&](std::int32_t expert) {
            return probabilities[expert] + (bias ? bias[expert] : 0.0f);
        };
        std::partial_sort(
            order.begin(), order.begin() + static_cast<std::ptrdiff_t>(used), order.end(),
            [&](std::int32_t left, std::int32_t right) {
                const float a = score(left), b = score(right);
                return a != b ? a > b : left < right;
            });
        std::copy_n(order.begin(), used, chosen);
    }
    float total = 0.0f;
    for (std::size_t slot = 0; slot < used; ++slot) {
        weights[slot] = probabilities[chosen[slot]];
        total += weights[slot];
    }
    const float divisor = std::max(total, sum_floor);
    for (std::size_t slot = 0; slot < used; ++slot)
        weights[slot] = weights[slot] / divisor * weight_scale;
}

// SwiGLU with both halves clamped before combining, which is what the per-layer
// swiglu_clamp values bound.
inline void clamped_swiglu(
    const float* gate, const float* up, std::size_t size, float limit, float* output
) {
    for (std::size_t i = 0; i < size; ++i) {
        const float g = std::min(std::max(gate[i], -limit), limit);
        const float u = std::min(std::max(up[i], -limit), limit);
        output[i] = (g / (1.0f + std::exp(-g))) * u;
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
struct YarnParameters {
    // Zero ext_factor disables extension entirely, which is what the
    // uncompressed layers use.
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    std::uint32_t original_context = 0;
};

// The dimension at which a wavelength reaches `rotations` full turns within the
// original context, which is what bounds YaRN's interpolation ramp.
inline float yarn_correction_dim(
    float rotations, std::size_t rope_dim, std::uint32_t original_context, float freq_base
) {
    return static_cast<float>(rope_dim) *
        std::log(static_cast<float>(original_context) /
                 (rotations * 2.0f * 3.14159265358979323846f)) /
        (2.0f * std::log(freq_base));
}

inline void rope(
    float* values,
    std::size_t rope_dim,
    std::int32_t position,
    float freq_base,
    float freq_scale,
    bool inverse,
    const YarnParameters& yarn = {}
) {
    // YaRN blends between interpolating a frequency (scaling the position down)
    // and leaving it alone, per dimension: fast-rotating dimensions keep their
    // frequency while slow ones are interpolated, with a ramp between.
    float low = 0.0f, high = 0.0f;
    const bool extended = yarn.ext_factor != 0.0f && yarn.original_context > 0;
    if (extended) {
        low = std::floor(yarn_correction_dim(
            yarn.beta_fast, rope_dim, yarn.original_context, freq_base));
        high = std::ceil(yarn_correction_dim(
            yarn.beta_slow, rope_dim, yarn.original_context, freq_base));
        low = std::max(0.0f, low);
        high = std::min(static_cast<float>(rope_dim) - 1.0f, high);
    }
    const float magnitude = extended
        ? yarn.attn_factor * (1.0f + 0.1f * std::log(1.0f / freq_scale))
        : yarn.attn_factor;

    for (std::size_t i = 0; i + 1 < rope_dim; i += 2) {
        const float exponent = -static_cast<float>(i) / static_cast<float>(rope_dim);
        const float extrapolated = static_cast<float>(position) * std::pow(freq_base, exponent);
        float theta = extrapolated * freq_scale;
        if (extended) {
            const float ramp = std::min(1.0f, std::max(0.0f,
                (static_cast<float>(i) / 2.0f - low) / std::max(0.001f, high - low)));
            const float mix = (1.0f - ramp) * yarn.ext_factor;
            theta = theta * (1.0f - mix) + extrapolated * mix;
        }
        const float cosine = std::cos(theta) * magnitude;
        const float sine = (inverse ? -std::sin(theta) : std::sin(theta)) * magnitude;
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
    // Heads are independent and each writes only its own slice, so the split is
    // exact rather than merely close: the arithmetic within a head, including
    // the order of the softmax reduction, is untouched. The scratch moves
    // inside the loop because it is the one thing the heads shared.
#pragma omp parallel for schedule(static) num_threads(thread_count())
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(heads); ++index) {
        const std::size_t head = static_cast<std::size_t>(index);
        std::vector<float> weights(positions);
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

// The lightning indexer's score for every compressed block.
//
// Per head, the query is dotted with the block's key -- one key shared by all
// heads, as the cache stores a single 128-wide row per block -- rectified, and
// weighted by that head's share from the indexer projection. The rectifier is
// what makes this a selector rather than a second attention: a head contributes
// only where it agrees with the block, never against it.
//
// The reference rotates both query and keys through a fixed Hadamard matrix
// first. It is orthogonal, so every dot product here is unchanged by it; the
// rotation exists to spread quantization error across the cached channels, and
// this cache is not quantized.
inline void indexer_scores(
    const float* queries,
    const float* keys,
    const float* weights,
    std::size_t heads,
    std::size_t dim,
    std::size_t entries,
    float* out
) {
#pragma omp parallel for schedule(static) num_threads(thread_count())
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(entries); ++index) {
        const float* key = keys + static_cast<std::size_t>(index) * dim;
        double total = 0.0;
        for (std::size_t head = 0; head < heads; ++head) {
            const float* query = queries + head * dim;
            double dot = 0.0;
            for (std::size_t i = 0; i < dim; ++i)
                dot += static_cast<double>(query[i]) * key[i];
            if (dot > 0.0) total += dot * weights[head];
        }
        out[static_cast<std::size_t>(index)] = static_cast<float>(total);
    }
}

// Keep the `k` highest-scoring entries, marking the rest unusable.
//
// Selection is over the *visible* entries only, which is why the caller passes
// the count rather than the whole cache: an invisible block scoring highly must
// not take a slot from a visible one. Ties go to the lower index, which the
// reference's partial sort does not promise -- it cannot matter for two blocks
// whose scores are bit-identical, and it makes this deterministic.
inline void top_k_select(
    const float* scores, std::size_t entries, std::size_t k, std::uint8_t* keep
) {
    if (k >= entries) {
        std::fill(keep, keep + entries, static_cast<std::uint8_t>(1));
        return;
    }
    std::fill(keep, keep + entries, static_cast<std::uint8_t>(0));
    std::vector<std::uint32_t> order(entries);
    for (std::size_t i = 0; i < entries; ++i) order[i] = static_cast<std::uint32_t>(i);
    std::nth_element(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k),
                     order.end(), [&](std::uint32_t left, std::uint32_t right) {
        if (scores[left] != scores[right]) return scores[left] > scores[right];
        return left < right;
    });
    for (std::size_t i = 0; i < k; ++i) keep[order[i]] = 1;
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
