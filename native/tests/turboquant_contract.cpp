// TurboQuant KV-cache codec contract.
//
// The codebooks are pinned against the paper rather than against themselves:
// the test re-derives the Lloyd-Max fixed point by Monte Carlo and checks the
// resulting distortion against the figures published in arXiv:2504.19874
// (0.034 for 3-bit, 0.009 for 4-bit). The rest of the file pins the two
// identities the attention path depends on -- that the rotation preserves
// inner products, and that a weighted sum of values may be accumulated in the
// rotated domain and inverse-rotated once at the end.

#include "turboquant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kHeadDim = 128;

// `distortion` is the Lloyd-Max quantizer's error on a unit Gaussian, from the
// paper. `score_bound` caps the relative RMS error of an attention score and is
// set just above the measured value, so a regression in the rotation or the
// scale refit shows up here rather than silently degrading generation.
struct CodebookCase {
    const char* label;
    TurboType type;
    float distortion;
    float score_bound;
};

const CodebookCase kCases[] = {
    {"turbo2", TurboType::Turbo2, 0.117482f, 0.18f},
    {"turbo3", TurboType::Turbo3, 0.034548f, 0.08f},
    {"turbo4", TurboType::Turbo4, 0.009501f, 0.035f},
};

int nearest(const float* codebook, int levels, float value) {
    int best = 0;
    float best_distance = std::fabs(value - codebook[0]);
    for (int level = 1; level < levels; ++level) {
        const float distance = std::fabs(value - codebook[level]);
        if (distance < best_distance) { best_distance = distance; best = level; }
    }
    return best;
}

bool half_contract() {
    for (const float value : {1.0f, 0.5f, 0.0f, 3.14159f, 1.0e-4f, 65504.0f, -2.5f}) {
        const float round_trip = turbo_half_value(turbo_half_bits(value));
        const float tolerance = std::max(1.0e-7f, std::fabs(value) * 1.0e-3f);
        if (std::fabs(round_trip - value) > tolerance) {
            std::printf("half round trip: %.9e -> %.9e\n", value, round_trip);
            return false;
        }
    }
    return true;
}

bool packing_contract() {
    std::mt19937 engine(1234);
    for (const auto& item : kCases) {
        const int bits = turbo_bits(item.type), levels = 1 << bits;
        std::uniform_int_distribution<std::uint32_t> pick(0, static_cast<std::uint32_t>(levels - 1));
        std::vector<std::uint32_t> indices(kTurboBlock);
        std::vector<std::uint8_t> packed(static_cast<std::size_t>(turbo_block_bytes(item.type)), 0);
        for (int i = 0; i < kTurboBlock; ++i) {
            indices[i] = pick(engine);
            turbo_pack_index(packed.data() + 2, i, bits, indices[i]);
        }
        for (int i = 0; i < kTurboBlock; ++i) {
            const std::uint32_t decoded = turbo_unpack_index(packed.data() + 2, i, bits);
            if (decoded != indices[i]) {
                std::printf("%s pack slot %d: expected %u, got %u\n",
                            item.label, i, indices[i], decoded);
                return false;
            }
        }
        // 2.5, 3.5 and 4.5 bits per value.
        const float per_value = 8.0f * static_cast<float>(turbo_block_bytes(item.type))
            / static_cast<float>(kTurboBlock);
        if (std::fabs(per_value - (static_cast<float>(bits) + 0.5f)) > 1.0e-6f) {
            std::printf("%s block size: %.3f bits per value\n", item.label, per_value);
            return false;
        }
    }
    return true;
}

// The transform must be its own inverse and preserve norms, or neither
// attention identity holds.
bool rotation_contract() {
    std::mt19937 engine(99);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> source(kHeadDim), rotated(kHeadDim), restored(kHeadDim);
    for (int i = 0; i < kHeadDim; ++i) source[i] = normal(engine);

    float source_energy = 0.0f;
    for (int i = 0; i < kHeadDim; ++i) source_energy += source[i] * source[i];

    turbo_rotate(source.data(), rotated.data(), kHeadDim, 7);
    float rotated_energy = 0.0f;
    for (int i = 0; i < kHeadDim; ++i) rotated_energy += rotated[i] * rotated[i];
    if (std::fabs(rotated_energy - source_energy) > 1.0e-3f * source_energy) {
        std::printf("rotation norm: %.9e -> %.9e\n", source_energy, rotated_energy);
        return false;
    }

    turbo_inverse_rotate(rotated.data(), restored.data(), kHeadDim, 7);
    for (int i = 0; i < kHeadDim; ++i) {
        if (std::fabs(restored[i] - source[i]) > 1.0e-4f) {
            std::printf("rotation inverse element %d: expected %.9e, got %.9e\n",
                        i, source[i], restored[i]);
            return false;
        }
    }

    // <Rq, Rk> == <q, k>: the identity the score path relies on.
    std::vector<float> key(kHeadDim), rotated_key(kHeadDim);
    for (int i = 0; i < kHeadDim; ++i) key[i] = normal(engine);
    turbo_rotate(key.data(), rotated_key.data(), kHeadDim, 7);
    float direct = 0.0f, through_rotation = 0.0f;
    for (int i = 0; i < kHeadDim; ++i) {
        direct += source[i] * key[i];
        through_rotation += rotated[i] * rotated_key[i];
    }
    if (std::fabs(direct - through_rotation) > 1.0e-3f * std::max(1.0f, std::fabs(direct))) {
        std::printf("rotation inner product: %.9e vs %.9e\n", direct, through_rotation);
        return false;
    }
    return true;
}

// Re-derive the codebook as a Lloyd-Max fixed point and check its distortion
// against the paper. This is what keeps the constants honest.
bool codebook_contract() {
    constexpr int kSamples = 4000000;
    for (const auto& item : kCases) {
        const int levels = 1 << turbo_bits(item.type);
        const float* codebook = turbo_codebook(item.type);
        std::vector<double> sums(static_cast<std::size_t>(levels), 0.0);
        std::vector<double> counts(static_cast<std::size_t>(levels), 0.0);
        double distortion = 0.0;

        std::mt19937 engine(20260731);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (int sample = 0; sample < kSamples; ++sample) {
            const float value = normal(engine);
            const int index = nearest(codebook, levels, value);
            sums[static_cast<std::size_t>(index)] += value;
            counts[static_cast<std::size_t>(index)] += 1.0;
            const double error = value - codebook[index];
            distortion += error * error;
        }
        distortion /= kSamples;

        for (int level = 0; level < levels; ++level) {
            if (counts[static_cast<std::size_t>(level)] < 1000.0) {
                std::printf("%s level %d: only %.0f samples\n",
                            item.label, level, counts[static_cast<std::size_t>(level)]);
                return false;
            }
            const double centroid =
                sums[static_cast<std::size_t>(level)] / counts[static_cast<std::size_t>(level)];
            if (std::fabs(centroid - codebook[level]) > 0.015) {
                std::printf("%s level %d not a Lloyd fixed point: stored %.8f, centroid %.8f\n",
                            item.label, level, codebook[level], centroid);
                return false;
            }
        }
        if (std::fabs(distortion - item.distortion) > 2.0e-3) {
            std::printf("%s distortion: paper %.6f, measured %.6f\n",
                        item.label, item.distortion, distortion);
            return false;
        }
        std::printf("%s: distortion %.6f (paper %.6f)\n",
                    item.label, distortion, item.distortion);
    }
    return true;
}

// End-to-end: encode real-ish vectors, then check reconstruction error, score
// error and the value-accumulation identity.
bool codec_contract() {
    constexpr int kVectors = 512;
    std::mt19937 engine(4242);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // A KV cache is not isotropic -- a few channels carry far more energy than
    // the rest. Give the test data that shape, since flattening it is exactly
    // what the rotation is supposed to do.
    std::vector<float> channel_scale(kHeadDim);
    for (int i = 0; i < kHeadDim; ++i)
        channel_scale[i] = (i % 17 == 0) ? 8.0f : 1.0f;

    for (const auto& item : kCases) {
        const int stride = turbo_block_bytes(item.type);
        const int blocks = kHeadDim / kTurboBlock;
        std::vector<std::uint8_t> packed(static_cast<std::size_t>(kVectors * blocks * stride), 0);
        std::vector<float> source(static_cast<std::size_t>(kVectors * kHeadDim));

        for (int vector = 0; vector < kVectors; ++vector) {
            float* values = source.data() + vector * kHeadDim;
            for (int i = 0; i < kHeadDim; ++i) values[i] = normal(engine) * channel_scale[i];
            turbo_encode_vector(
                values, kHeadDim, item.type, 3,
                packed.data() + static_cast<std::size_t>(vector) * blocks * stride);
        }

        // Reconstruction error, measured in the original domain.
        double error_energy = 0.0, source_energy = 0.0;
        std::vector<float> rotated(kHeadDim), restored(kHeadDim);
        for (int vector = 0; vector < kVectors; ++vector) {
            const float* values = source.data() + vector * kHeadDim;
            turbo_decode_vector(
                packed.data() + static_cast<std::size_t>(vector) * blocks * stride,
                kHeadDim, item.type, rotated.data());
            turbo_inverse_rotate(rotated.data(), restored.data(), kHeadDim, 3);
            for (int i = 0; i < kHeadDim; ++i) {
                const double error = restored[i] - values[i];
                error_energy += error * error;
                source_energy += static_cast<double>(values[i]) * values[i];
            }
        }
        // The least-squares scale refit is what buys the margin here: plain
        // quantization against a fixed codebook would land at the scalar
        // distortion, and the refit comes in about 15% under it.
        const double relative = error_energy / source_energy;
        if (relative > item.distortion) {
            std::printf("%s reconstruction: relative error %.6f, scalar bound %.6f\n",
                        item.label, relative, item.distortion);
            return false;
        }

        // Score path: dot an already-rotated query against stored keys and
        // compare with the exact inner product.
        std::vector<float> query(kHeadDim), rotated_query(kHeadDim);
        for (int i = 0; i < kHeadDim; ++i) query[i] = normal(engine) * channel_scale[i];
        turbo_rotate(query.data(), rotated_query.data(), kHeadDim, 3);

        double score_error = 0.0, score_energy = 0.0;
        for (int vector = 0; vector < kVectors; ++vector) {
            const float* values = source.data() + vector * kHeadDim;
            float exact = 0.0f;
            for (int i = 0; i < kHeadDim; ++i) exact += query[i] * values[i];
            const float approximate = turbo_dot_rotated(
                packed.data() + static_cast<std::size_t>(vector) * blocks * stride,
                rotated_query.data(), kHeadDim, item.type);
            score_error += static_cast<double>(approximate - exact) * (approximate - exact);
            score_energy += static_cast<double>(exact) * exact;
        }
        const double score_relative = std::sqrt(score_error / score_energy);
        if (score_relative > item.score_bound) {
            std::printf("%s score: relative RMS error %.6f exceeds %.6f\n",
                        item.label, score_relative, item.score_bound);
            return false;
        }

        // Value path: accumulating in the rotated domain and inverse-rotating
        // once must match decoding each value first. Both are linear, so this
        // is exact up to float rounding.
        std::vector<float> weights(kVectors);
        float weight_total = 0.0f;
        for (int vector = 0; vector < kVectors; ++vector) {
            weights[vector] = std::fabs(normal(engine));
            weight_total += weights[vector];
        }
        for (int vector = 0; vector < kVectors; ++vector) weights[vector] /= weight_total;

        std::vector<float> accumulated(kHeadDim, 0.0f), expected(kHeadDim, 0.0f);
        for (int vector = 0; vector < kVectors; ++vector) {
            turbo_decode_vector(
                packed.data() + static_cast<std::size_t>(vector) * blocks * stride,
                kHeadDim, item.type, rotated.data());
            for (int i = 0; i < kHeadDim; ++i) accumulated[i] += weights[vector] * rotated[i];
            turbo_inverse_rotate(rotated.data(), restored.data(), kHeadDim, 3);
            for (int i = 0; i < kHeadDim; ++i) expected[i] += weights[vector] * restored[i];
        }
        std::vector<float> folded(kHeadDim);
        turbo_inverse_rotate(accumulated.data(), folded.data(), kHeadDim, 3);
        for (int i = 0; i < kHeadDim; ++i) {
            if (std::fabs(folded[i] - expected[i]) > 1.0e-4f * std::max(1.0f, std::fabs(expected[i]))) {
                std::printf("%s value fold element %d: expected %.9e, got %.9e\n",
                            item.label, i, expected[i], folded[i]);
                return false;
            }
        }

        std::printf("%s: reconstruction %.6f, score RMS %.6f, %d bytes per %d values\n",
                    item.label, relative, score_relative, stride, kTurboBlock);
    }
    return true;
}

// The CUDA values kernel does not call turbo_unpack_index: it hoists the byte
// offset and shift per thread and reads two bytes unconditionally, using a
// `spill` flag so the whole warp stays on one path. That arithmetic is only
// exercised on the GPU, and only turbo3 ever spills, so it is replicated here
// against the reference unpack for every slot and both widths -- including the
// bound the kernel relies on, that neither byte ever leaves its own block.
bool branchless_unpack_contract() {
    for (const auto& item : kCases) {
        const int bits = turbo_bits(item.type);
        if (bits == 2) continue;  // the kernels only implement turbo3 and turbo4
        const int block_bytes = turbo_block_bytes(item.type);
        std::mt19937 engine(777);
        std::uniform_int_distribution<std::uint32_t> pick(0, (1u << bits) - 1u);
        std::vector<std::uint8_t> packed(static_cast<std::size_t>(block_bytes), 0);
        std::vector<std::uint32_t> indices(kTurboBlock);
        for (int i = 0; i < kTurboBlock; ++i) {
            indices[i] = pick(engine);
            turbo_pack_index(packed.data() + 2, i, bits, indices[i]);
        }
        for (int slot = 0; slot < kTurboBlock; ++slot) {
            const int bit = slot * bits;
            const int byte_offset = 2 + (bit >> 3);
            const int shift = bit & 7;
            const int spill = (shift + bits > 8) ? 1 : 0;
            const std::uint32_t mask = (1u << bits) - 1u;
            if (byte_offset + spill >= block_bytes) {
                std::printf("%s slot %d reads byte %d outside its %d-byte block\n",
                            item.label, slot, byte_offset + spill, block_bytes);
                return false;
            }
            const std::uint32_t low = packed[static_cast<std::size_t>(byte_offset)];
            const std::uint32_t high = packed[static_cast<std::size_t>(byte_offset + spill)];
            const std::uint32_t decoded = ((low | (high << 8)) >> shift) & mask;
            if (decoded != indices[slot]) {
                std::printf("%s slot %d branchless unpack: expected %u, got %u\n",
                            item.label, slot, indices[slot], decoded);
                return false;
            }
        }
    }
    return true;
}

// A zero block has to survive exactly: padded or unwritten cache slots are
// zero, and a NaN scale there would poison the whole softmax.
bool zero_block_contract() {
    for (const auto& item : kCases) {
        std::vector<float> zeros(kHeadDim, 0.0f), decoded(kHeadDim, 1.0f);
        const int stride = turbo_block_bytes(item.type);
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(kHeadDim / kTurboBlock * stride), 0xff);
        turbo_encode_vector(zeros.data(), kHeadDim, item.type, 0, packed.data());
        turbo_decode_vector(packed.data(), kHeadDim, item.type, decoded.data());
        for (int i = 0; i < kHeadDim; ++i) {
            if (decoded[i] != 0.0f) {
                std::printf("%s zero block element %d: got %.9e\n", item.label, i, decoded[i]);
                return false;
            }
        }
    }
    return true;
}

// head_dim must be a power of two for the butterfly. Qwen attention layers use
// 128, but the guard is what stops a 96 or 80 head silently producing garbage.
bool dimension_contract() {
    for (const int dimension : {32, 64, 128, 256}) {
        if (!turbo_dimension_supported(dimension)) {
            std::printf("dimension %d should be supported\n", dimension);
            return false;
        }
    }
    for (const int dimension : {0, 16, 48, 80, 96, 192}) {
        if (turbo_dimension_supported(dimension)) {
            std::printf("dimension %d should be rejected\n", dimension);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    if (!half_contract()) return 1;
    if (!packing_contract()) return 2;
    if (!rotation_contract()) return 3;
    if (!codebook_contract()) return 4;
    if (!codec_contract()) return 5;
    if (!zero_block_contract()) return 6;
    if (!dimension_contract()) return 7;
    if (!branchless_unpack_contract()) return 8;
    return 0;
}
