// TurboQuant KV-cache quality harness.
//
// Reports the metrics that actually decide whether a bit width is usable,
// measured at the attention output rather than on the raw tensors. Plain
// reconstruction MSE is the wrong criterion here: one llama.cpp port found
// perplexity and generation quality inversely correlated at head_dim=64, so
// this tool leads with softmax agreement and output error instead.
//
// It also reports the K/V norm ratio, which is what drives the asymmetric bit
// allocation between keys and values. Quantization error scales with the norm
// squared, and real checkpoints are lopsided -- Qwen2.5-7B shows a 106x
// difference -- so keys and values usually want different widths.
//
// Usage:
//   bench_turboquant [--dim N] [--keys N] [--ratio R] [--baseline-bits B]
//                    [--dump PATH]
//
//   --ratio R  synthetic K/V norm ratio, to sweep the allocation decision.
//   --dump     read real tensors instead of synthesising them. The file is
//              headerless little-endian f32: keys[count][dim] then
//              values[count][dim], preceded by two int32s (count, dim).
//              FlyweightV2QwenRuntime.dump_kv writes exactly this.
//   --baseline-bits  what to quote compression against; defaults to 16 for the
//              runtime's f16 cache default.
//
// Synthetic data is a starting point for picking candidates, not a substitute
// for a dump from the target checkpoint: the outlier structure it models is a
// guess, and outlier structure is exactly what the rotation has to defeat.

#include "turboquant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

struct Cache {
    int count = 0;
    int dimension = 0;
    std::vector<float> keys;
    std::vector<float> values;
};

const char* type_label(TurboType type) {
    switch (type) {
        case TurboType::Turbo2: return "turbo2";
        case TurboType::Turbo3: return "turbo3";
        default: return "turbo4";
    }
}

double vector_norm(const float* values, int dimension) {
    double energy = 0.0;
    for (int i = 0; i < dimension; ++i) energy += static_cast<double>(values[i]) * values[i];
    return std::sqrt(energy);
}

double mean_norm(const std::vector<float>& data, int count, int dimension) {
    double total = 0.0;
    for (int i = 0; i < count; ++i) total += vector_norm(data.data() + i * dimension, dimension);
    return total / std::max(1, count);
}

Cache synthesise(int count, int dimension, double ratio, std::uint32_t seed) {
    Cache cache;
    cache.count = count;
    cache.dimension = dimension;
    cache.keys.resize(static_cast<std::size_t>(count) * dimension);
    cache.values.resize(static_cast<std::size_t>(count) * dimension);

    std::mt19937 engine(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // A handful of high-energy channels plus a mild positional drift, which is
    // roughly what attention keys look like before rotation.
    std::vector<float> key_channel(dimension, 1.0f), value_channel(dimension, 1.0f);
    for (int i = 0; i < dimension; ++i) {
        if (i % 23 == 0) key_channel[i] = 12.0f;
        if (i % 31 == 0) value_channel[i] = 4.0f;
    }
    for (int entry = 0; entry < count; ++entry) {
        const float drift = 1.0f + 0.25f * static_cast<float>(entry) / static_cast<float>(count);
        for (int i = 0; i < dimension; ++i) {
            cache.keys[static_cast<std::size_t>(entry) * dimension + i] =
                normal(engine) * key_channel[i] * drift;
            cache.values[static_cast<std::size_t>(entry) * dimension + i] =
                normal(engine) * value_channel[i];
        }
    }

    // The outlier channels already skew the two norms apart, so rescale the
    // keys to land on the requested ratio exactly -- otherwise sweeping
    // --ratio would not be sweeping the quantity it names.
    const double key_norm = mean_norm(cache.keys, count, dimension);
    const double value_norm = mean_norm(cache.values, count, dimension);
    if (key_norm > 0.0 && value_norm > 0.0) {
        const float correction = static_cast<float>(ratio * value_norm / key_norm);
        for (float& key : cache.keys) key *= correction;
    }
    return cache;
}

bool load_dump(const char* path, Cache& cache) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) {
        std::printf("cannot open %s\n", path);
        return false;
    }
    std::int32_t header[2] = {0, 0};
    if (std::fread(header, sizeof(std::int32_t), 2, file) != 2
        || header[0] <= 0 || header[1] <= 0) {
        std::printf("%s: bad header\n", path);
        std::fclose(file);
        return false;
    }
    cache.count = header[0];
    cache.dimension = header[1];
    const std::size_t elements = static_cast<std::size_t>(cache.count) * cache.dimension;
    cache.keys.resize(elements);
    cache.values.resize(elements);
    const bool ok = std::fread(cache.keys.data(), sizeof(float), elements, file) == elements
        && std::fread(cache.values.data(), sizeof(float), elements, file) == elements;
    std::fclose(file);
    if (!ok) std::printf("%s: truncated\n", path);
    return ok;
}

void softmax(std::vector<float>& scores) {
    const float peak = *std::max_element(scores.begin(), scores.end());
    double total = 0.0;
    for (float& score : scores) {
        score = std::exp(score - peak);
        total += score;
    }
    for (float& score : scores) score = static_cast<float>(score / total);
}

struct Report {
    double score_rms = 0.0;
    double divergence = 0.0;
    double top1_agreement = 0.0;
    double output_error = 0.0;
    // exp(entropy of the exact softmax): roughly how many entries the step
    // actually attends to. Near 1 means the softmax is one-hot, which drives
    // the divergence to zero and makes it uninformative -- so it is reported
    // rather than left to be misread as a perfect score.
    double attended = 0.0;
};

// One attention step against the whole cache, exact vs quantized, repeated over
// several queries so the numbers are not a single draw.
Report evaluate(const Cache& cache, TurboType key_type, TurboType value_type, int queries) {
    const int dimension = cache.dimension;
    const int blocks = dimension / kTurboBlock;
    const int key_stride = turbo_block_bytes(key_type);
    const int value_stride = turbo_block_bytes(value_type);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));

    std::vector<std::uint8_t> packed_keys(
        static_cast<std::size_t>(cache.count) * blocks * key_stride, 0);
    std::vector<std::uint8_t> packed_values(
        static_cast<std::size_t>(cache.count) * blocks * value_stride, 0);
    for (int entry = 0; entry < cache.count; ++entry) {
        turbo_encode_vector(
            cache.keys.data() + static_cast<std::size_t>(entry) * dimension,
            dimension, key_type, 0,
            packed_keys.data() + static_cast<std::size_t>(entry) * blocks * key_stride);
        turbo_encode_vector(
            cache.values.data() + static_cast<std::size_t>(entry) * dimension,
            dimension, value_type, 1,
            packed_values.data() + static_cast<std::size_t>(entry) * blocks * value_stride);
    }

    std::mt19937 engine(31337);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Report report;
    double score_error = 0.0, score_energy = 0.0;
    double output_error = 0.0, output_energy = 0.0;
    long long agreements = 0;

    std::vector<float> query(dimension), rotated_query(dimension);
    std::vector<float> exact_scores(cache.count), approximate_scores(cache.count);
    std::vector<float> rotated(dimension), accumulated(dimension), folded(dimension);
    std::vector<float> exact_output(dimension);

    for (int step = 0; step < queries; ++step) {
        // Draw the query from the key distribution so the scores have realistic
        // spread; a white query would make the softmax uniform and hide error.
        const float* seed_key =
            cache.keys.data() + static_cast<std::size_t>(step % cache.count) * dimension;
        for (int i = 0; i < dimension; ++i) query[i] = seed_key[i] + 0.5f * normal(engine);
        turbo_rotate(query.data(), rotated_query.data(), dimension, 0);

        for (int entry = 0; entry < cache.count; ++entry) {
            const float* key = cache.keys.data() + static_cast<std::size_t>(entry) * dimension;
            float exact = 0.0f;
            for (int i = 0; i < dimension; ++i) exact += query[i] * key[i];
            exact_scores[entry] = exact * scale;
            approximate_scores[entry] = scale * turbo_dot_rotated(
                packed_keys.data() + static_cast<std::size_t>(entry) * blocks * key_stride,
                rotated_query.data(), dimension, key_type);
            const double delta = approximate_scores[entry] - exact_scores[entry];
            score_error += delta * delta;
            score_energy += static_cast<double>(exact_scores[entry]) * exact_scores[entry];
        }

        const int exact_top = static_cast<int>(
            std::max_element(exact_scores.begin(), exact_scores.end()) - exact_scores.begin());
        const int approximate_top = static_cast<int>(
            std::max_element(approximate_scores.begin(), approximate_scores.end())
            - approximate_scores.begin());
        if (exact_top == approximate_top) ++agreements;

        softmax(exact_scores);
        softmax(approximate_scores);
        double entropy = 0.0;
        for (int entry = 0; entry < cache.count; ++entry) {
            if (exact_scores[entry] > 1.0e-12f) entropy -=
                exact_scores[entry] * std::log(exact_scores[entry]);
            if (exact_scores[entry] > 1.0e-12f && approximate_scores[entry] > 1.0e-12f)
                report.divergence += exact_scores[entry]
                    * std::log(exact_scores[entry] / approximate_scores[entry]);
        }
        report.attended += std::exp(entropy);

        // Exact output, then the quantized one accumulated in the rotated
        // domain with a single inverse rotation -- the shape the kernel takes.
        std::fill(exact_output.begin(), exact_output.end(), 0.0f);
        std::fill(accumulated.begin(), accumulated.end(), 0.0f);
        for (int entry = 0; entry < cache.count; ++entry) {
            const float* value = cache.values.data() + static_cast<std::size_t>(entry) * dimension;
            for (int i = 0; i < dimension; ++i) exact_output[i] += exact_scores[entry] * value[i];
            turbo_decode_vector(
                packed_values.data() + static_cast<std::size_t>(entry) * blocks * value_stride,
                dimension, value_type, rotated.data());
            for (int i = 0; i < dimension; ++i)
                accumulated[i] += approximate_scores[entry] * rotated[i];
        }
        turbo_inverse_rotate(accumulated.data(), folded.data(), dimension, 1);
        for (int i = 0; i < dimension; ++i) {
            const double delta = folded[i] - exact_output[i];
            output_error += delta * delta;
            output_energy += static_cast<double>(exact_output[i]) * exact_output[i];
        }
    }

    report.score_rms = std::sqrt(score_error / std::max(1.0e-30, score_energy));
    report.divergence /= queries;
    report.attended /= queries;
    report.top1_agreement = static_cast<double>(agreements) / queries;
    report.output_error = std::sqrt(output_error / std::max(1.0e-30, output_energy));
    return report;
}

} // namespace

int main(int argc, char** argv) {
    int dimension = 128, count = 2048, queries = 32;
    double ratio = 1.0;
    float baseline_bits = 16.0f;
    const char* dump = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const bool has_value = i + 1 < argc;
        if (flag == "--dim" && has_value) dimension = std::atoi(argv[++i]);
        else if (flag == "--keys" && has_value) count = std::atoi(argv[++i]);
        else if (flag == "--queries" && has_value) queries = std::atoi(argv[++i]);
        else if (flag == "--ratio" && has_value) ratio = std::atof(argv[++i]);
        else if (flag == "--dump" && has_value) dump = argv[++i];
        else if (flag == "--baseline-bits" && has_value)
            baseline_bits = static_cast<float>(std::atof(argv[++i]));
        else {
            std::printf("usage: bench_turboquant [--dim N] [--keys N] [--queries N]"
                        " [--ratio R] [--baseline-bits B] [--dump PATH]\n");
            return 1;
        }
    }

    Cache cache;
    if (dump) {
        if (!load_dump(dump, cache)) return 1;
    } else {
        cache = synthesise(count, dimension, ratio, 20260731);
    }

    if (!turbo_dimension_supported(cache.dimension)) {
        std::printf("head_dim %d unsupported: the Walsh-Hadamard butterfly needs a"
                    " power of two >= %d\n", cache.dimension, kTurboBlock);
        return 1;
    }

    const double key_norm = mean_norm(cache.keys, cache.count, cache.dimension);
    const double value_norm = mean_norm(cache.values, cache.count, cache.dimension);
    const double measured_ratio = value_norm > 0.0 ? key_norm / value_norm : 0.0;

    std::printf("entries %d  head_dim %d  queries %d\n",
                cache.count, cache.dimension, queries);
    std::printf("mean |K| %.4f  mean |V| %.4f  K/V ratio %.2fx\n",
                key_norm, value_norm, measured_ratio);
    // The rotation concentrates the coordinates towards a Gaussian, and it does
    // so better as the dimension grows -- so a head_dim above the validated 128
    // is the benign direction. Below it the concentration is weaker, and that is
    // the regime where one port found perplexity and generation quality moving
    // in opposite directions.
    if (cache.dimension < 128) {
        std::printf("note: head_dim %d is below TurboQuant's validated 128, where"
                    " the rotation concentrates less; judge this on generation,"
                    " not perplexity\n", cache.dimension);
    } else if (cache.dimension > 128) {
        std::printf("note: head_dim %d is above TurboQuant's validated 128, which"
                    " favours the rotation\n", cache.dimension);
    }
    // Thresholds from the llama.cpp ports' sweep over real checkpoints.
    if (measured_ratio < 10.0) std::printf("allocation: uniform width should do\n");
    else if (measured_ratio < 60.0) std::printf("allocation: keys want ~1 bit more than values\n");
    else std::printf("allocation: keys want mixed precision, 8-bit outlier channels\n");

    const TurboType types[] = {TurboType::Turbo2, TurboType::Turbo3, TurboType::Turbo4};
    // Compression is quoted against whatever the cache runs at today, which
    // defaults to f16 (cache_type_k/v in FlyweightV2QwenRuntimeOptions). Pass
    // --baseline-bits 32 for f32 or 8.5 for q8_0, whose 34-byte block over 32
    // values costs half a bit of scale on top of the 8-bit payload.
    const float baseline = baseline_bits;
    std::printf("\n%-16s %8s %10s %10s %10s %10s %9s\n",
                "K/V", "bpv", "compress", "scoreRMS", "softmaxKL", "outRMS", "attended");
    for (const TurboType key_type : types) {
        for (const TurboType value_type : types) {
            const Report report = evaluate(cache, key_type, value_type, queries);
            const float bits_per_value = 0.5f * (
                8.0f * turbo_block_bytes(key_type) / kTurboBlock
                + 8.0f * turbo_block_bytes(value_type) / kTurboBlock);
            char label[32];
            std::snprintf(label, sizeof(label), "%s/%s",
                          type_label(key_type), type_label(value_type));
            std::printf("%-16s %8.2f %9.2fx %10.4f %10.5f %10.4f %9.1f",
                        label, bits_per_value, baseline / bits_per_value,
                        report.score_rms, report.divergence, report.output_error,
                        report.attended);
            if (report.top1_agreement < 1.0)
                std::printf("  top1 %.0f%%", 100.0 * report.top1_agreement);
            std::printf("\n");
        }
    }
    return 0;
}
