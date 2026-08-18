#pragma once

// IQ *encoder*: f32 -> IQ3_XXS.
//
// The K-quants in qwen_kquant_pack.h fit a scale and round to a lattice. The IQ
// formats have no lattice: 3.06 bits per weight buys 256 hand-chosen 4-value
// patterns, and quantizing means *searching* that codebook rather than rounding
// into it. The patterns are the same table the decoders read (kIq3xxsGrid), so
// encoder and decoder cannot drift apart.
//
// Three things make the search cheap enough to run over a 27B checkpoint:
//
// 1. Every grid element is one of eight magnitudes, so a 4-value pattern is a
//    point in an 8^4 = 4096 cube. The inverse map below precomputes, once, the
//    nearest grid entry for every point in that cube -- after which quantizing
//    four values is a rounding and one array read, not 256 distance
//    evaluations.
// 2. Signs are stored apart from magnitudes, so the search only ever sees |x|.
//    The catch is that only the 128 even-parity sign patterns are encodable; an
//    odd one costs the sign of whichever element has least to lose.
// 3. The per-group scale is fitted in closed form against the chosen patterns
//    rather than searched; only its starting point is swept.
//
// What is NOT here is an importance matrix. llama.cpp weights this search by
// per-channel activation statistics gathered over calibration data, and for the
// sub-3-bit formats (IQ2_XXS, IQ1_M) it refuses to quantize without them. The
// fallback weight used below -- sqrt(sigma2 + x^2) -- is what llama.cpp uses
// when it has no matrix, and it is why IQ3_XXS is the only IQ format offered.
//
// Compiled into the same translation unit as the K-quant packers, so it
// inherits `-ffp-contract=off` and the golden-hash guard with them.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "qwen_iq_tables.h"
#include "qwen_kquant.h"

namespace qwen_kpack {

constexpr int kIq3Levels = 8;
constexpr int kIq3GridEntries = 256;
constexpr std::uint32_t kIq3xxsBlockBytesPack = 98;

// The eight magnitudes a grid element can hold, ascending, plus the inverse map
// from a 4-level tuple to the nearest grid entry.
//
// Both are read off kIq3xxsGrid rather than restated, so this stays correct if
// the table is ever regenerated.
struct Iq3Codebook {
    std::uint8_t levels[kIq3Levels] = {};
    // Tuple l0 | l1<<3 | l2<<6 | l3<<9 -> the kCandidates grid entries closest
    // to it, nearest first. Only 256 of the 4096 tuples are themselves grid
    // points, so most of the cube snaps somewhere -- and which "somewhere" is
    // best depends on how much each of the four values matters, which the
    // distance here cannot see. Keeping a few and letting the caller score them
    // under its own weights is what recovers that.
    static constexpr int kCandidates = 8;
    std::uint8_t nearest[kIq3Levels * kIq3Levels * kIq3Levels * kIq3Levels]
                        [kCandidates] = {};

    Iq3Codebook() {
        bool present[256] = {};
        for (int entry = 0; entry < kIq3GridEntries; ++entry)
            for (int part = 0; part < 4; ++part)
                present[(kIq3xxsGrid[entry] >> (8 * part)) & 0xffu] = true;
        int count = 0;
        std::uint8_t level_of[256] = {};
        for (int value = 0; value < 256; ++value) {
            if (!present[value]) continue;
            level_of[value] = static_cast<std::uint8_t>(count);
            levels[count++] = static_cast<std::uint8_t>(value);
        }

        // Grid entries as level tuples, so distances are computed once.
        std::uint8_t entry_levels[kIq3GridEntries][4];
        for (int entry = 0; entry < kIq3GridEntries; ++entry)
            for (int part = 0; part < 4; ++part)
                entry_levels[entry][part] =
                    level_of[(kIq3xxsGrid[entry] >> (8 * part)) & 0xffu];

        constexpr int kTuples =
            kIq3Levels * kIq3Levels * kIq3Levels * kIq3Levels;
        for (int tuple = 0; tuple < kTuples; ++tuple) {
            int want[4];
            for (int part = 0; part < 4; ++part)
                want[part] = levels[(tuple >> (3 * part)) & 7];
            int best_distance[kCandidates];
            int best_entry[kCandidates];
            for (int slot = 0; slot < kCandidates; ++slot) {
                best_distance[slot] = 1 << 30;
                best_entry[slot] = 0;
            }
            for (int entry = 0; entry < kIq3GridEntries; ++entry) {
                int distance = 0;
                for (int part = 0; part < 4; ++part) {
                    const int difference = want[part] - levels[entry_levels[entry][part]];
                    distance += difference * difference;
                }
                if (distance >= best_distance[kCandidates - 1]) continue;
                int slot = kCandidates - 1;
                while (slot > 0 && best_distance[slot - 1] > distance) {
                    best_distance[slot] = best_distance[slot - 1];
                    best_entry[slot] = best_entry[slot - 1];
                    --slot;
                }
                best_distance[slot] = distance;
                best_entry[slot] = entry;
            }
            for (int slot = 0; slot < kCandidates; ++slot)
                nearest[tuple][slot] = static_cast<std::uint8_t>(best_entry[slot]);
        }
    }
};

inline const Iq3Codebook& iq3_codebook() {
    static const Iq3Codebook codebook;
    return codebook;
}

// The level whose magnitude is closest to `target`, which is |x| divided by the
// group's scale.
inline int iq3_nearest_level(const Iq3Codebook& codebook, float target) {
    int best = 0;
    float best_distance = std::fabs(target - static_cast<float>(codebook.levels[0]));
    for (int level = 1; level < kIq3Levels; ++level) {
        const float distance =
            std::fabs(target - static_cast<float>(codebook.levels[level]));
        if (distance < best_distance) { best_distance = distance; best = level; }
    }
    return best;
}

// One quad of 8 magnitudes -> two grid entries, written into `indices`.
//
// Rounding to the cube gives the neighbourhood; the weights decide which member
// of it. `weights` and `scale` are what turn a geometric nearest into the one
// that costs least here -- an entry that is two levels off on an element worth
// nothing beats one that is a level off on the element that carries the group.
inline void iq3_choose_pattern(const Iq3Codebook& codebook, const float* magnitudes,
                               const float* weights, float scale,
                               std::uint8_t* indices) {
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int half = 0; half < 2; ++half) {
        const float* part_magnitudes = magnitudes + half * 4;
        const float* part_weights = weights + half * 4;
        int tuple = 0;
        for (int part = 0; part < 4; ++part) {
            const int level =
                iq3_nearest_level(codebook, part_magnitudes[part] * inverse);
            tuple |= level << (3 * part);
        }
        int best = codebook.nearest[tuple][0];
        float best_error = -1.0f;
        for (int slot = 0; slot < Iq3Codebook::kCandidates; ++slot) {
            const int entry = codebook.nearest[tuple][slot];
            const std::uint32_t pattern = kIq3xxsGrid[entry];
            float error = 0.0f;
            for (int part = 0; part < 4; ++part) {
                const float value =
                    static_cast<float>((pattern >> (8 * part)) & 0xffu) * scale;
                const float difference = part_magnitudes[part] - value;
                error += part_weights[part] * difference * difference;
            }
            if (best_error < 0.0f || error < best_error) {
                best_error = error;
                best = entry;
            }
        }
        indices[half] = static_cast<std::uint8_t>(best);
    }
}

inline float iq3_pattern_value(const std::uint8_t* indices, int element) {
    const std::uint32_t pattern = kIq3xxsGrid[indices[element >> 2]];
    return static_cast<float>((pattern >> (8 * (element & 3))) & 0xffu);
}

// The 7 bits that reproduce these signs, with the parity the format requires.
//
// Only 128 of the 256 sign patterns are encodable: the table the decoder reads
// forces the eighth bit to whatever makes the byte even. An odd pattern
// therefore cannot be stored, and one element has to take the wrong sign. The
// cheapest one to give up is whichever contributes least to the reconstruction,
// so this picks the smallest |x| * pattern value rather than the smallest |x|:
// an element the grid rounded to a large magnitude is expensive to flip even
// when its input was small.
inline std::uint8_t iq3_encode_signs(const float* values, const std::uint8_t* indices) {
    int mask = 0, set = 0, cheapest = 0;
    float cheapest_cost = 0.0f;
    for (int element = 0; element < 8; ++element) {
        if (values[element] < 0.0f) { mask |= 1 << element; ++set; }
        const float cost =
            std::fabs(values[element]) * iq3_pattern_value(indices, element);
        if (element == 0 || cost < cheapest_cost) { cheapest_cost = cost; cheapest = element; }
    }
    if (set & 1) mask ^= 1 << cheapest;
    return static_cast<std::uint8_t>(mask & 127);
}

// IQ3_XXS: 98 bytes per 256 values. d, 64 grid indices, and one uint32 per
// 32-element group holding four 7-bit sign selectors and a 4-bit scale.
inline void qwen_pack_iq3_xxs(const float* values, std::uint64_t count,
                              std::uint8_t* out) {
    const auto& codebook = iq3_codebook();
    // Reconstruction is d * (0.5 + ls) * 0.5 * pattern, so a group's scale
    // spans [0.25 d, 7.75 d] in sixteen steps.
    constexpr float kMaxGroupFactor = 7.75f;

    for (std::uint64_t block = 0; block * 256 < count; ++block) {
        const float* x = values + block * 256;
        std::uint8_t* base = out + block * kIq3xxsBlockBytesPack;
        std::memset(base, 0, kIq3xxsBlockBytesPack);

        float sum_squares = 0.0f;
        for (int i = 0; i < 256; ++i) sum_squares += x[i] * x[i];
        // llama.cpp's weight when it has no importance matrix: an element is
        // worth attending to in proportion to its own size, floored by the
        // block's typical size so a near-zero element is not chased.
        const float sigma2 = 2.0f * sum_squares / 256.0f;

        float group_scales[8] = {};
        float max_scale = 0.0f;
        std::uint8_t indices[64] = {};

        for (int group = 0; group < 8; ++group) {
            const float* source = x + group * 32;
            float magnitudes[32], weights[32];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                magnitudes[i] = std::fabs(source[i]);
                weights[i] = std::sqrt(sigma2 + source[i] * source[i]);
                amax = std::max(amax, magnitudes[i]);
            }
            if (amax <= 0.0f) continue;

            // Start where the largest magnitude lands on the largest pattern
            // value, then sweep around it: the grid is coarse enough that the
            // best fit is often a step or two away from the extreme one.
            const float top = static_cast<float>(codebook.levels[kIq3Levels - 1]);
            float best_error = 0.0f, best_scale = 0.0f;
            std::uint8_t best_indices[8] = {};
            for (int step = -6; step <= 6; ++step) {
                const float scale = amax / top * (1.0f + 0.03f * static_cast<float>(step));
                if (scale <= 0.0f) continue;
                std::uint8_t trial[8];
                for (int quad = 0; quad < 4; ++quad)
                    iq3_choose_pattern(codebook, magnitudes + quad * 8,
                                       weights + quad * 8, scale, trial + quad * 2);
                // Closed-form scale for the patterns just chosen, which is
                // strictly better than the trial scale that produced them.
                float sum_ag = 0.0f, sum_gg = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const float g = iq3_pattern_value(trial + (i / 8) * 2, i & 7);
                    sum_ag += weights[i] * magnitudes[i] * g;
                    sum_gg += weights[i] * g * g;
                }
                if (sum_gg <= 0.0f) continue;
                const float fitted = sum_ag / sum_gg;
                float error = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const float g = iq3_pattern_value(trial + (i / 8) * 2, i & 7);
                    const float difference = magnitudes[i] - fitted * g;
                    error += weights[i] * difference * difference;
                }
                if (best_scale == 0.0f || error < best_error) {
                    best_error = error;
                    best_scale = fitted;
                    std::memcpy(best_indices, trial, sizeof(trial));
                }
            }
            group_scales[group] = best_scale;
            std::memcpy(indices + group * 8, best_indices, 8);
            max_scale = std::max(max_scale, best_scale);
        }

        if (max_scale <= 0.0f) continue;  // an all-zero block stays all zero

        const std::uint16_t d_bits = qwen_half_bits(max_scale / kMaxGroupFactor);
        std::memcpy(base, &d_bits, 2);
        const float d = qwen_half_value(d_bits);

        for (int group = 0; group < 8; ++group) {
            const float* source = x + group * 32;
            int ls = 0;
            if (d > 0.0f && group_scales[group] > 0.0f)
                ls = std::max(0, std::min(15, static_cast<int>(std::lround(
                    2.0f * group_scales[group] / d - 0.5f))));
            const float stored = d * (0.5f + static_cast<float>(ls)) * 0.5f;

            // Re-choose the patterns against the scale that was actually
            // stored, exactly as the K-quant packers re-derive their codes: the
            // 4-bit group scale is coarse, and patterns picked for the fitted
            // scale can be a level off once it is rounded.
            std::uint8_t* group_indices = indices + group * 8;
            if (stored > 0.0f) {
                float magnitudes[32], weights[32];
                for (int i = 0; i < 32; ++i) {
                    magnitudes[i] = std::fabs(source[i]);
                    weights[i] = std::sqrt(sigma2 + source[i] * source[i]);
                }
                for (int quad = 0; quad < 4; ++quad)
                    iq3_choose_pattern(codebook, magnitudes + quad * 8,
                                       weights + quad * 8, stored,
                                       group_indices + quad * 2);
            }

            std::uint32_t aux = static_cast<std::uint32_t>(ls) << 28;
            for (int quad = 0; quad < 4; ++quad)
                aux |= static_cast<std::uint32_t>(
                    iq3_encode_signs(source + quad * 8, group_indices + quad * 2))
                    << (7 * quad);
            std::memcpy(base + 2 + 64 + group * 4, &aux, 4);
            std::memcpy(base + 2 + group * 8, group_indices, 8);
        }
    }
}

}  // namespace qwen_kpack
