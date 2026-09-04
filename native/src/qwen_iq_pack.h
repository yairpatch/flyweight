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
// The search accepts an importance matrix: per-channel activation statistics
// gathered over calibration data (flyweight_v2_imatrix.hpp), applied exactly as
// llama.cpp applies them -- weight = qw[channel] * sqrt(sigma2 + x^2). Without
// one the weight is sqrt(sigma2 + x^2) alone, which is llama.cpp's own
// no-matrix fallback, and why IQ3_XXS is the only IQ format offered for
// packing: the sub-3-bit formats (IQ2_XXS, IQ1_M) need a matrix to be worth
// using at all.
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

// Per-channel importance for the block search. `values` has one weight per
// input column of the tensor's rows (`row` long), or per-expert rows of them
// (`chunk` elements apart) when the matrix was gathered per expert; null
// leaves the search unweighted. `element_begin` is where this call's values
// sit in the whole tensor, so tiled packing indexes the same channel a
// single-call pack would.
struct Iq3Importance {
    const float* values = nullptr;
    std::uint64_t row = 0;
    std::uint64_t chunk = 0;  // 0 means the same weights for every row
    std::uint64_t element_begin = 0;

    float at(std::uint64_t element) const {
        const std::uint64_t index = element_begin + element;
        const std::uint64_t column = index % row;
        return values[chunk ? index / chunk * row + column : column];
    }
};

// IQ3_XXS: 98 bytes per 256 values. d, 64 grid indices, and one uint32 per
// 32-element group holding four 7-bit sign selectors and a 4-bit scale.
inline void qwen_pack_iq3_xxs(const float* values, std::uint64_t count,
                              std::uint8_t* out,
                              const Iq3Importance* importance = nullptr) {
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

        // One group's search weights. With a matrix, each element's base
        // weight is scaled by its channel's activation energy; a group whose
        // channels never fired in calibration keeps the base weight rather
        // than zeroing real values out of the search.
        const auto fill_weights = [&](std::uint64_t first_element,
                                      const float* source, float* weights) {
            float max_importance = 0.0f;
            if (importance && importance->values) {
                for (int i = 0; i < 32; ++i) {
                    const float channel = importance->at(first_element + i);
                    weights[i] = channel;
                    max_importance = std::max(max_importance, channel);
                }
            }
            for (int i = 0; i < 32; ++i) {
                const float base = std::sqrt(sigma2 + source[i] * source[i]);
                weights[i] = max_importance > 0.0f ? weights[i] * base : base;
            }
        };

        float group_scales[8] = {};
        float max_scale = 0.0f;
        std::uint8_t indices[64] = {};

        for (int group = 0; group < 8; ++group) {
            const float* source = x + group * 32;
            float magnitudes[32], weights[32];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                magnitudes[i] = std::fabs(source[i]);
                amax = std::max(amax, magnitudes[i]);
            }
            if (amax <= 0.0f) continue;
            fill_weights(block * 256 + group * 32, source, weights);

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
                for (int i = 0; i < 32; ++i)
                    magnitudes[i] = std::fabs(source[i]);
                fill_weights(block * 256 + group * 32, source, weights);
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

// ---------------------------------------------------------------------------
// IQ2_XS
// ---------------------------------------------------------------------------
//
// The same codebook shape as IQ3_XXS one floor down: 512 grid entries of
// *eight* values each, drawn from three magnitudes {8, 25, 43}, with a 7-bit
// even-parity sign selector per entry and a 4-bit scale per 16 values. The
// inverse map is a base-3 cube over eight digits -- 6561 tuples -- against
// 512 entries, built once. This format is only offered for packing WITH an
// importance matrix (enforced by the loader, not here): at 2.31 bits the
// unweighted search has nothing to say about which of a block's channels can
// afford to be wrong, and llama.cpp refuses the format without a matrix for
// the same reason.

constexpr int kIq2Levels = 3;
constexpr int kIq2GridEntries = 512;
constexpr std::uint32_t kIq2xsBlockBytesPack = 74;

struct Iq2xsCodebook {
    std::uint8_t levels[kIq2Levels] = {};
    std::uint8_t level_of[256] = {};
    static constexpr int kCandidates = 8;
    static constexpr int kTuples = 6561;  // 3^8
    std::uint16_t nearest[kTuples][kCandidates] = {};

    Iq2xsCodebook() {
        bool present[256] = {};
        for (int entry = 0; entry < kIq2GridEntries; ++entry)
            for (int part = 0; part < 8; ++part)
                present[(kIq2xsGrid[entry] >> (8 * part)) & 0xffull] = true;
        int count = 0;
        for (int value = 0; value < 256; ++value) {
            if (!present[value]) continue;
            level_of[value] = static_cast<std::uint8_t>(count);
            if (count < kIq2Levels)
                levels[count] = static_cast<std::uint8_t>(value);
            ++count;
        }

        std::uint8_t entry_levels[kIq2GridEntries][8];
        for (int entry = 0; entry < kIq2GridEntries; ++entry)
            for (int part = 0; part < 8; ++part)
                entry_levels[entry][part] =
                    level_of[(kIq2xsGrid[entry] >> (8 * part)) & 0xffull];

        for (int tuple = 0; tuple < kTuples; ++tuple) {
            int want[8];
            int rest = tuple;
            for (int part = 0; part < 8; ++part) {
                want[part] = levels[rest % kIq2Levels];
                rest /= kIq2Levels;
            }
            int best_distance[kCandidates];
            int best_entry[kCandidates];
            for (int slot = 0; slot < kCandidates; ++slot) {
                best_distance[slot] = 1 << 30;
                best_entry[slot] = 0;
            }
            for (int entry = 0; entry < kIq2GridEntries; ++entry) {
                int distance = 0;
                for (int part = 0; part < 8; ++part) {
                    const int difference =
                        want[part] - levels[entry_levels[entry][part]];
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
                best_entry[slot] = static_cast<std::uint16_t>(entry);
            }
            for (int slot = 0; slot < kCandidates; ++slot)
                nearest[tuple][slot] =
                    static_cast<std::uint16_t>(best_entry[slot]);
        }
    }
};

inline const Iq2xsCodebook& iq2xs_codebook() {
    static const Iq2xsCodebook codebook;
    return codebook;
}

inline int iq2xs_nearest_level(const Iq2xsCodebook& codebook, float target) {
    int best = 0;
    float best_distance =
        std::fabs(target - static_cast<float>(codebook.levels[0]));
    for (int level = 1; level < kIq2Levels; ++level) {
        const float distance =
            std::fabs(target - static_cast<float>(codebook.levels[level]));
        if (distance < best_distance) { best_distance = distance; best = level; }
    }
    return best;
}

// One quad of 8 magnitudes -> the 9-bit grid entry that costs least under
// `weights` at reconstruction scale `scale`.
inline std::uint16_t iq2xs_choose_pattern(const Iq2xsCodebook& codebook,
                                          const float* magnitudes,
                                          const float* weights, float scale) {
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    int tuple = 0, radix = 1;
    for (int part = 0; part < 8; ++part) {
        tuple += radix
            * iq2xs_nearest_level(codebook, magnitudes[part] * inverse);
        radix *= kIq2Levels;
    }
    std::uint16_t best = codebook.nearest[tuple][0];
    float best_error = -1.0f;
    for (int slot = 0; slot < Iq2xsCodebook::kCandidates; ++slot) {
        const std::uint16_t entry = codebook.nearest[tuple][slot];
        const std::uint64_t pattern = kIq2xsGrid[entry];
        float error = 0.0f;
        for (int part = 0; part < 8; ++part) {
            const float value =
                static_cast<float>((pattern >> (8 * part)) & 0xffull) * scale;
            const float difference = magnitudes[part] - value;
            error += weights[part] * difference * difference;
        }
        if (best_error < 0.0f || error < best_error) {
            best_error = error;
            best = entry;
        }
    }
    return best;
}

// The 7-bit selector reproducing these signs. Only even-parity masks are
// representable -- kIq2xxsSigns completes the eighth bit from the low seven
// -- so an odd mask gives up the sign of whichever element contributes least
// to the reconstruction, exactly as the IQ3_XXS encoder does.
inline std::uint8_t iq2xs_encode_signs(const float* values,
                                       std::uint64_t pattern) {
    int mask = 0, set = 0, cheapest = 0;
    float cheapest_cost = 0.0f;
    for (int element = 0; element < 8; ++element) {
        if (values[element] < 0.0f) { mask |= 1 << element; ++set; }
        const float cost = std::fabs(values[element])
            * static_cast<float>((pattern >> (8 * element)) & 0xffull);
        if (element == 0 || cost < cheapest_cost) {
            cheapest_cost = cost;
            cheapest = element;
        }
    }
    if (set & 1) mask ^= 1 << cheapest;
    return static_cast<std::uint8_t>(mask & 127);
}

// IQ2_XS: 74 bytes per 256 values -> d(2), 32 uint16 entries (9-bit grid
// index, 7-bit sign selector), 8 bytes of 4-bit group scales.
inline void qwen_pack_iq2_xs(const float* values, std::uint64_t count,
                             std::uint8_t* out,
                             const Iq3Importance* importance = nullptr) {
    const auto& codebook = iq2xs_codebook();
    // Reconstruction is d * (0.5 + ls) * 0.25 * magnitude, so a group's scale
    // spans [0.125 d, 3.875 d] in sixteen steps.
    constexpr float kMaxGroupFactor = 3.875f;

    for (std::uint64_t block = 0; block * 256 < count; ++block) {
        const float* x = values + block * 256;
        std::uint8_t* base = out + block * kIq2xsBlockBytesPack;
        std::memset(base, 0, kIq2xsBlockBytesPack);

        float sum_squares = 0.0f;
        for (int i = 0; i < 256; ++i) sum_squares += x[i] * x[i];
        const float sigma2 = 2.0f * sum_squares / 256.0f;

        // Sixteen-wide groups here, against IQ3_XXS's thirty-two; otherwise
        // the same weighting, importance fallback included.
        const auto fill_weights = [&](std::uint64_t first_element,
                                      const float* source, float* weights) {
            float max_importance = 0.0f;
            if (importance && importance->values) {
                for (int i = 0; i < 16; ++i) {
                    const float channel = importance->at(first_element + i);
                    weights[i] = channel;
                    max_importance = std::max(max_importance, channel);
                }
            }
            for (int i = 0; i < 16; ++i) {
                const float base_weight =
                    std::sqrt(sigma2 + source[i] * source[i]);
                weights[i] = max_importance > 0.0f ? weights[i] * base_weight
                                                   : base_weight;
            }
        };

        float group_scales[16] = {};
        float max_scale = 0.0f;

        for (int group = 0; group < 16; ++group) {
            const float* source = x + group * 16;
            float magnitudes[16], weights[16];
            float amax = 0.0f;
            for (int i = 0; i < 16; ++i) {
                magnitudes[i] = std::fabs(source[i]);
                amax = std::max(amax, magnitudes[i]);
            }
            if (amax <= 0.0f) continue;
            fill_weights(block * 256 + group * 16, source, weights);

            const float top =
                static_cast<float>(codebook.levels[kIq2Levels - 1]);
            float best_error = 0.0f, best_scale = 0.0f;
            for (int step = -6; step <= 6; ++step) {
                const float scale =
                    amax / top * (1.0f + 0.03f * static_cast<float>(step));
                if (scale <= 0.0f) continue;
                std::uint16_t trial[2];
                for (int quad = 0; quad < 2; ++quad)
                    trial[quad] = iq2xs_choose_pattern(
                        codebook, magnitudes + quad * 8, weights + quad * 8,
                        scale);
                float sum_ag = 0.0f, sum_gg = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const float g = static_cast<float>(
                        (kIq2xsGrid[trial[i / 8]] >> (8 * (i & 7))) & 0xffull);
                    sum_ag += weights[i] * magnitudes[i] * g;
                    sum_gg += weights[i] * g * g;
                }
                if (sum_gg <= 0.0f) continue;
                const float fitted = sum_ag / sum_gg;
                float error = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const float g = static_cast<float>(
                        (kIq2xsGrid[trial[i / 8]] >> (8 * (i & 7))) & 0xffull);
                    const float difference = magnitudes[i] - fitted * g;
                    error += weights[i] * difference * difference;
                }
                if (best_scale == 0.0f || error < best_error) {
                    best_error = error;
                    best_scale = fitted;
                }
            }
            group_scales[group] = best_scale;
            max_scale = std::max(max_scale, best_scale);
        }

        if (max_scale <= 0.0f) continue;  // an all-zero block stays all zero

        const std::uint16_t d_bits = qwen_half_bits(max_scale / kMaxGroupFactor);
        std::memcpy(base, &d_bits, 2);
        const float d = qwen_half_value(d_bits);

        for (int group = 0; group < 16; ++group) {
            const float* source = x + group * 16;
            int ls = 0;
            if (d > 0.0f && group_scales[group] > 0.0f)
                ls = std::max(0, std::min(15, static_cast<int>(std::lround(
                    4.0f * group_scales[group] / d - 0.5f))));
            const float stored = d * (0.5f + static_cast<float>(ls)) * 0.25f;

            // Re-choose against the scale actually stored, then encode the
            // signs against the patterns actually chosen.
            float magnitudes[16], weights[16];
            for (int i = 0; i < 16; ++i)
                magnitudes[i] = std::fabs(source[i]);
            fill_weights(block * 256 + group * 16, source, weights);
            for (int quad = 0; quad < 2; ++quad) {
                const std::uint16_t entry = iq2xs_choose_pattern(
                    codebook, magnitudes + quad * 8, weights + quad * 8,
                    stored);
                const std::uint8_t selector =
                    iq2xs_encode_signs(source + quad * 8, kIq2xsGrid[entry]);
                const std::uint16_t packed = static_cast<std::uint16_t>(
                    entry | (static_cast<std::uint16_t>(selector) << 9));
                std::memcpy(base + 2 + (group * 2 + quad) * 2, &packed, 2);
            }
            base[66 + (group >> 1)] |=
                static_cast<std::uint8_t>(ls << (4 * (group & 1)));
        }
    }
}

// ---------------------------------------------------------------------------
// IQ4_XS
// ---------------------------------------------------------------------------
//
// Not a codebook format: each 4-bit code indexes the sixteen non-uniform
// IQ4_NL levels (kIq4nlValues, asymmetric: -127..113), and each 32-value
// sub-block has a 6-bit *signed* scale split across scales_l/scales_h. So
// encoding is a per-group scale search with independent nearest-level
// rounding -- K-quant-shaped work, not a pattern search -- and the weights
// (importance included) decide the scale, not the per-element rounding.

// The level whose value is closest to `target`. The table is 16 entries; a
// linear scan is cheaper than being clever.
inline int iq4_nearest_level(float target) {
    int best = 0;
    float best_distance = std::fabs(target - static_cast<float>(kIq4nlValues[0]));
    for (int level = 1; level < 16; ++level) {
        const float distance =
            std::fabs(target - static_cast<float>(kIq4nlValues[level]));
        if (distance < best_distance) { best_distance = distance; best = level; }
    }
    return best;
}

constexpr std::uint32_t kIq4xsBlockBytesPack = 136;

// IQ4_XS: 136 bytes per 256 values -> d(2) scales_h(2) scales_l[4] qs[128].
inline void qwen_pack_iq4_xs(const float* values, std::uint64_t count,
                             std::uint8_t* out,
                             const Iq3Importance* importance = nullptr) {
    for (std::uint64_t block = 0; block * 256 < count; ++block) {
        const float* x = values + block * 256;
        std::uint8_t* base = out + block * kIq4xsBlockBytesPack;
        std::memset(base, 0, kIq4xsBlockBytesPack);

        float sum_squares = 0.0f;
        for (int i = 0; i < 256; ++i) sum_squares += x[i] * x[i];
        const float sigma2 = 2.0f * sum_squares / 256.0f;

        // Same weighting scheme as the IQ3_XXS search above, importance
        // fallback included.
        const auto fill_weights = [&](std::uint64_t first_element,
                                      const float* source, float* weights) {
            float max_importance = 0.0f;
            if (importance && importance->values) {
                for (int i = 0; i < 32; ++i) {
                    const float channel = importance->at(first_element + i);
                    weights[i] = channel;
                    max_importance = std::max(max_importance, channel);
                }
            }
            for (int i = 0; i < 32; ++i) {
                const float base_weight =
                    std::sqrt(sigma2 + source[i] * source[i]);
                weights[i] = max_importance > 0.0f ? weights[i] * base_weight
                                                   : base_weight;
            }
        };

        float group_scales[8] = {};
        float max_scale_magnitude = 0.0f;

        for (int sub = 0; sub < 8; ++sub) {
            const float* source = x + sub * 32;
            float weights[32];
            float amax = 0.0f, pinned = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float magnitude = std::fabs(source[i]);
                if (magnitude > amax) { amax = magnitude; pinned = source[i]; }
            }
            if (amax <= 0.0f) continue;
            fill_weights(block * 256 + sub * 32, source, weights);

            // The table is asymmetric, so the sign of the scale matters: try
            // the largest element pinned near either end, swept around both.
            // The closed-form refit against the rounded levels is what
            // actually decides; the sweep only supplies neighbourhoods.
            float best_error = -1.0f, best_scale = 0.0f;
            for (const float anchor : {static_cast<float>(kIq4nlValues[0]),
                                       static_cast<float>(kIq4nlValues[15])}) {
                for (int step = -6; step <= 6; ++step) {
                    const float scale =
                        pinned / anchor * (1.0f + 0.03f * static_cast<float>(step));
                    if (scale == 0.0f) continue;
                    const float inverse = 1.0f / scale;
                    float sum_wxt = 0.0f, sum_wtt = 0.0f;
                    int trial[32];
                    for (int i = 0; i < 32; ++i) {
                        trial[i] = iq4_nearest_level(source[i] * inverse);
                        const float level =
                            static_cast<float>(kIq4nlValues[trial[i]]);
                        sum_wxt += weights[i] * source[i] * level;
                        sum_wtt += weights[i] * level * level;
                    }
                    if (sum_wtt <= 0.0f) continue;
                    const float fitted = sum_wxt / sum_wtt;
                    float error = 0.0f;
                    for (int i = 0; i < 32; ++i) {
                        const float difference = source[i]
                            - fitted * static_cast<float>(kIq4nlValues[trial[i]]);
                        error += weights[i] * difference * difference;
                    }
                    if (best_error < 0.0f || error < best_error) {
                        best_error = error;
                        best_scale = fitted;
                    }
                }
            }
            group_scales[sub] = best_scale;
            max_scale_magnitude =
                std::max(max_scale_magnitude, std::fabs(best_scale));
        }

        if (max_scale_magnitude <= 0.0f) continue;  // all-zero block

        // 6-bit signed group scales against a shared half d, exactly as the
        // decoder reads them back: scale = (low | high << 4) - 32.
        const std::uint16_t d_bits = qwen_half_bits(max_scale_magnitude / 32.0f);
        std::memcpy(base, &d_bits, 2);
        const float d = qwen_half_value(d_bits);
        std::uint16_t scales_high = 0;
        for (int sub = 0; sub < 8; ++sub) {
            int ls = 0;
            if (d > 0.0f && group_scales[sub] != 0.0f)
                ls = std::max(-32, std::min(31, static_cast<int>(
                    std::lround(group_scales[sub] / d))));
            const float stored_scale = d * static_cast<float>(ls);
            const float* source = x + sub * 32;
            std::uint8_t* quants = base + 8 + sub * 16;
            for (int i = 0; i < 32; ++i) {
                // Re-round against the scale that was actually stored; with a
                // zero scale the decode is zero whatever the code, and the
                // level nearest zero keeps the bytes deterministic.
                const int code = stored_scale != 0.0f
                    ? iq4_nearest_level(source[i] / stored_scale)
                    : 8;
                if (i < 16) quants[i] |= static_cast<std::uint8_t>(code);
                else quants[i - 16] |= static_cast<std::uint8_t>(code << 4);
            }
            const int biased = ls + 32;
            base[4 + (sub >> 1)] |=
                static_cast<std::uint8_t>((biased & 15) << (4 * (sub & 1)));
            scales_high |= static_cast<std::uint16_t>((biased >> 4) & 3) << (2 * sub);
        }
        std::memcpy(base + 2, &scales_high, 2);
    }
}

}  // namespace qwen_kpack
