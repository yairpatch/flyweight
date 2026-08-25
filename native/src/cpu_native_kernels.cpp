// Hand-written host kernels.
//
// Each one replaces an emulated corpus kernel of the same name and must produce
// the same memory the emulated version would. That is not a claim to be taken
// on trust: every kernel here has a case in tests/cpu_parity_contract.cpp which
// runs both against randomized inputs and compares.
//
// The shape is deliberately different from the CUDA original. A GPU kernel maps
// one thread to one output element to get coalescing and latency hiding; a CPU
// wants long contiguous runs, few threads, and SIMD over rows. Nothing here
// should mirror the grid.

#include <colibri_cpu_native.hpp>
#include <colibri_cpu_shim_geometry.hpp>  // half_bits_to_float

#include "cpu_q8_dot.h"

extern "C" std::uint32_t colibri_cpu_features();

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace colibri::cpu {
namespace {

// --- rms_norm -------------------------------------------------------------
//
// Corpus signature:
//   void rms_norm(const float* input, const float* weights, float* output,
//                 int elements, float epsilon, int one_centered)
//
// The emulated version is a single 256-thread block doing a shuffle reduction,
// which is the worst case for the fiber scheduler. Serially it is two passes
// over a vector that is almost always L1- or L2-resident, so there is nothing
// to parallelize and no reason to involve the pool.
void rms_norm(const Launch&, void** arguments) {
    const float* input = *reinterpret_cast<const float**>(arguments[0]);
    const float* weights = *reinterpret_cast<const float**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int elements = *reinterpret_cast<const int*>(arguments[3]);
    const float epsilon = *reinterpret_cast<const float*>(arguments[4]);
    const int one_centered = *reinterpret_cast<const int*>(arguments[5]);

    // The corpus accumulates the sum of squares in fp32 across a shuffle tree.
    // A flat fp32 sum here would drift from that on long rows, so the
    // accumulation is done in double and rounded once -- closer to the tree's
    // error profile than a sequential fp32 sum, and stable regardless of width.
    double total = 0.0;
    for (int index = 0; index < elements; ++index)
        total += static_cast<double>(input[index]) * input[index];

    const float mean = static_cast<float>(total / elements);
    const float scale = 1.0f / std::sqrt(mean + epsilon);

    // one_centered folds the +1 that Gemma-style norms apply to the weight.
    if (one_centered) {
        for (int index = 0; index < elements; ++index)
            output[index] = input[index] * scale * (1.0f + weights[index]);
    } else {
        for (int index = 0; index < elements; ++index)
            output[index] = input[index] * scale * weights[index];
    }
}

// --- scaled_add -----------------------------------------------------------
//
// Corpus signature:
//   void scaled_add(float* target, const float* source, float scale,
//                   int elements)
//
// Memory bound and trivially vectorized; left to the compiler, which turns this
// into an unrolled AVX loop.
void scaled_add(const Launch&, void** arguments) {
    float* target = *reinterpret_cast<float**>(arguments[0]);
    const float* source = *reinterpret_cast<const float**>(arguments[1]);
    const float scale = *reinterpret_cast<const float*>(arguments[2]);
    const int elements = *reinterpret_cast<const int*>(arguments[3]);

    for (int index = 0; index < elements; ++index)
        target[index] += scale * source[index];
}

// --- shared helpers -------------------------------------------------------

// Splits `count` items into chunks and runs them across the pool. Going through
// parallel_for one item at a time would pay an atomic per row; chunking keeps
// the claim overhead proportional to the number of workers instead.
template <class Body>
void parallel_chunks(std::uint64_t count, const Body& body) {
    if (count == 0) return;
    struct Job {
        const Body* body;
        std::uint64_t count;
        std::uint64_t chunk;
    };
    // Enough chunks that a straggler cannot hold up the launch, few enough that
    // the handoff stays negligible.
    const std::uint64_t target_chunks = 64;
    const std::uint64_t chunk =
        count <= target_chunks ? 1 : (count + target_chunks - 1) / target_chunks;
    Job job{&body, count, chunk};
    parallel_for(
        (count + chunk - 1) / chunk,
        [](void* opaque, std::uint64_t index) {
            auto* task = static_cast<Job*>(opaque);
            const std::uint64_t first = index * task->chunk;
            const std::uint64_t last =
                std::min(first + task->chunk, task->count);
            for (std::uint64_t item = first; item < last; ++item)
                (*task->body)(item);
        },
        &job);
}

// Dot product with four independent accumulators.
//
// Not a plain running sum: the CUDA kernels reduce through a warp shuffle tree,
// so their rounding error grows with log(n) rather than n. Splitting the
// accumulation keeps the host result in the same error class -- a sequential
// fp32 sum over a few thousand terms drifts far enough to show up as a quality
// difference, not just a parity-test failure. It is also faster, since the four
// chains break the loop-carried dependency and vectorize cleanly.
inline float dot_f32(const float* left, const float* right, int elements) {
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    int index = 0;
    for (; index + 4 <= elements; index += 4) {
        a += left[index + 0] * right[index + 0];
        b += left[index + 1] * right[index + 1];
        c += left[index + 2] * right[index + 2];
        d += left[index + 3] * right[index + 3];
    }
    float tail = 0.0f;
    for (; index < elements; ++index) tail += left[index] * right[index];
    return ((a + b) + (c + d)) + tail;
}

// --- qwen_f32_matvec_warp -------------------------------------------------
//
// Corpus signature:
//   void qwen_f32_matvec_warp(const float* matrix, const float* input,
//                             float* output, int input_size, int output_size)
//
// The decode-path GEMV, and by measurement ~43% of a dense CPU decode. Row per
// task rather than warp per row: each row is a contiguous run, so the loop is
// bandwidth-bound and wants long sequential reads, not 32 lanes strided across
// one row.
void qwen_f32_matvec_warp(const Launch&, void** arguments) {
    const float* matrix = *reinterpret_cast<const float**>(arguments[0]);
    const float* input = *reinterpret_cast<const float**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int input_size = *reinterpret_cast<const int*>(arguments[3]);
    const int output_size = *reinterpret_cast<const int*>(arguments[4]);

    parallel_chunks(static_cast<std::uint64_t>(output_size),
                    [&](std::uint64_t row) {
        const float* row_matrix =
            matrix + static_cast<std::int64_t>(row) * input_size;
        output[row] = dot_f32(row_matrix, input, input_size);
    });
}

// --- qwen_f32_matmul_rows -------------------------------------------------
//
// Corpus signature:
//   void qwen_f32_matmul_rows(const float* matrix, const float* input,
//                             float* output, int input_size, int output_size,
//                             int rows)
//
// The prefill GEMM, ~48% of a dense CPU decode. Parallelized over output rows
// with the token loop inside, so each weight row is fetched once and reused
// across all tokens -- the arithmetic intensity the GPU version does not need
// and the CPU version lives on.
void qwen_f32_matmul_rows(const Launch&, void** arguments) {
    const float* matrix = *reinterpret_cast<const float**>(arguments[0]);
    const float* input = *reinterpret_cast<const float**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int input_size = *reinterpret_cast<const int*>(arguments[3]);
    const int output_size = *reinterpret_cast<const int*>(arguments[4]);
    const int rows = *reinterpret_cast<const int*>(arguments[5]);

    parallel_chunks(static_cast<std::uint64_t>(output_size),
                    [&](std::uint64_t output_row) {
        const float* row_matrix =
            matrix + static_cast<std::int64_t>(output_row) * input_size;
        for (int token = 0; token < rows; ++token) {
            output[static_cast<std::int64_t>(token) * output_size + output_row] =
                dot_f32(row_matrix, input + static_cast<std::int64_t>(token) * input_size,
                        input_size);
        }
    });
}

// --- Q8_0 ------------------------------------------------------------------

// Q8_0 block: fp16 scale followed by 32 int8 weights, 34 bytes. All the corpus
// Q8 kernels index blocks as (row * input_size + column) / 32, and input_size is
// always a multiple of 32, so a row never straddles a block and its blocks are
// contiguous -- which is what makes a sequential host walk possible at all.
constexpr int kQ8BlockBytes = 34;
constexpr int kQ8BlockValues = 32;

// ISA dispatch, resolved once. colibri_cpu_features() reports what the machine
// actually supports and honours COLIBRI_CPU_BACKEND for forcing a lower path,
// which is how the parity harness can be run against the scalar reference.
enum class Q8Path { scalar, avx2, avx512 };

Q8Path q8_path() {
    static const Q8Path path = [] {
        constexpr std::uint32_t kAvx2 = 1u << 0;
        constexpr std::uint32_t kAvx512 = 1u << 1;
        const std::uint32_t features = colibri_cpu_features();
        if (features & kAvx512) return Q8Path::avx512;
        if (features & kAvx2) return Q8Path::avx2;
        return Q8Path::scalar;
    }();
    return path;
}

// Portable reference. Kept rather than deleted: it is the fallback on machines
// without AVX2, and it is what the SIMD paths were checked against.
inline float q8_row_dot_scalar(const unsigned char* cursor, const float* vector,
                               int blocks) {
    float total = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        std::uint16_t scale_bits;
        std::memcpy(&scale_bits, cursor, sizeof(scale_bits));
        const float scale = half_bits_to_float(scale_bits);
        const auto* values = reinterpret_cast<const signed char*>(cursor + 2);
        const float* slice = vector + block * kQ8BlockValues;

        float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
        for (int index = 0; index < kQ8BlockValues; index += 4) {
            a += static_cast<float>(values[index + 0]) * slice[index + 0];
            b += static_cast<float>(values[index + 1]) * slice[index + 1];
            c += static_cast<float>(values[index + 2]) * slice[index + 2];
            d += static_cast<float>(values[index + 3]) * slice[index + 3];
        }
        total += ((a + b) + (c + d)) * scale;
        cursor += kQ8BlockBytes;
    }
    return total;
}

// One row of a Q8_0 matrix against an f32 vector.
//
// On the critical path of every Q8 kernel below, which together were ~53% of a
// real decode, so this is where the intrinsics go.
inline float q8_row_dot(const unsigned char* packed, std::int64_t row,
                        const float* vector, int input_size) {
    const int blocks = input_size / kQ8BlockValues;
    const auto* cursor = reinterpret_cast<const std::uint8_t*>(packed) +
                         row * blocks * kQ8BlockBytes;
    switch (q8_path()) {
        case Q8Path::avx512:
            return colibri_q8_row_dot_avx512(cursor, vector, blocks);
        case Q8Path::avx2:
            return colibri_q8_row_dot_avx2(cursor, vector, blocks);
        default:
            return q8_row_dot_scalar(cursor, vector, blocks);
    }
}

// Gate and up rows against one activation vector, for the fused SwiGLU.
inline void q8_row_dot_pair(const unsigned char* gate_packed,
                            const unsigned char* up_packed, std::int64_t row,
                            const float* vector, int input_size, float* gate,
                            float* up) {
    const int blocks = input_size / kQ8BlockValues;
    const auto* gate_cursor = reinterpret_cast<const std::uint8_t*>(gate_packed) +
                              row * blocks * kQ8BlockBytes;
    const auto* up_cursor = reinterpret_cast<const std::uint8_t*>(up_packed) +
                            row * blocks * kQ8BlockBytes;
    switch (q8_path()) {
        case Q8Path::avx512:
            colibri_q8_row_dot_pair_avx512(gate_cursor, up_cursor, vector,
                                           blocks, gate, up);
            return;
        case Q8Path::avx2:
            colibri_q8_row_dot_pair_avx2(gate_cursor, up_cursor, vector, blocks,
                                         gate, up);
            return;
        default:
            *gate = q8_row_dot_scalar(gate_cursor, vector, blocks);
            *up = q8_row_dot_scalar(up_cursor, vector, blocks);
            return;
    }
}

// --- q8_matvec_transposed_warp --------------------------------------------
//
// Corpus signature:
//   void q8_matvec_transposed_warp(const unsigned char* packed,
//                                  const float* vector, float* output,
//                                  int input_size, int output_size)
//
// Measured at 73% of a Q8_0 decode -- the single hottest kernel on a quantized
// model.
void q8_matvec_transposed_warp(const Launch&, void** arguments) {
    const unsigned char* packed =
        *reinterpret_cast<const unsigned char**>(arguments[0]);
    const float* vector = *reinterpret_cast<const float**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int input_size = *reinterpret_cast<const int*>(arguments[3]);
    const int output_size = *reinterpret_cast<const int*>(arguments[4]);

    parallel_chunks(static_cast<std::uint64_t>(output_size),
                    [&](std::uint64_t row) {
        output[row] = q8_row_dot(packed, static_cast<std::int64_t>(row), vector,
                                 input_size);
    });
}

// --- q8_matmul_tiled ------------------------------------------------------
//
// Corpus signature:
//   void q8_matmul_tiled(const unsigned char* packed, const float* input,
//                        float* output, int input_size, int output_size,
//                        int rows)
//
// The prefill counterpart, 20% of a Q8_0 decode. The CUDA version tiles through
// shared memory to reuse each weight byte across 32 tokens; the host version
// gets the same reuse for free by holding a row in cache while looping tokens,
// so no tiling is needed.
void q8_matmul_tiled(const Launch&, void** arguments) {
    const unsigned char* packed =
        *reinterpret_cast<const unsigned char**>(arguments[0]);
    const float* input = *reinterpret_cast<const float**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int input_size = *reinterpret_cast<const int*>(arguments[3]);
    const int output_size = *reinterpret_cast<const int*>(arguments[4]);
    const int rows = *reinterpret_cast<const int*>(arguments[5]);

    parallel_chunks(static_cast<std::uint64_t>(output_size),
                    [&](std::uint64_t output_row) {
        for (int token = 0; token < rows; ++token) {
            output[static_cast<std::int64_t>(token) * output_size + output_row] =
                q8_row_dot(packed, static_cast<std::int64_t>(output_row),
                           input + static_cast<std::int64_t>(token) * input_size,
                           input_size);
        }
    });
}

// --- rms_norm_rows --------------------------------------------------------
//
// Corpus signature:
//   void rms_norm_rows(const float* input, const float* weights,
//                      float* output, int rows, int columns, float epsilon,
//                      int one_centered)
//
// Per-row RMS norm; rows are independent so the pool splits them.
void rms_norm_rows(const Launch&, void** arguments) {
    const float* input = *reinterpret_cast<const float**>(arguments[0]);
    const float* weights = *reinterpret_cast<const float**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int rows = *reinterpret_cast<const int*>(arguments[3]);
    const int columns = *reinterpret_cast<const int*>(arguments[4]);
    const float epsilon = *reinterpret_cast<const float*>(arguments[5]);
    const int one_centered = *reinterpret_cast<const int*>(arguments[6]);

    parallel_chunks(static_cast<std::uint64_t>(rows), [&](std::uint64_t row) {
        const std::int64_t start = static_cast<std::int64_t>(row) * columns;
        // Double accumulation for the same reason as rms_norm: it lands closer
        // to the reference's reduction tree than a sequential fp32 sum would.
        double total = 0.0;
        for (int column = 0; column < columns; ++column) {
            const double value = input[start + column];
            total += value * value;
        }
        const float inverse_rms = 1.0f / std::sqrt(
            static_cast<float>(total / columns) + epsilon);
        for (int column = 0; column < columns; ++column) {
            const float weight =
                one_centered ? 1.0f + weights[column] : weights[column];
            output[start + column] = input[start + column] * inverse_rms * weight;
        }
    });
}

// --- qwen_shared_scale ----------------------------------------------------
//
// Corpus signature:
//   void qwen_shared_scale(const float* input, const float* gate,
//                          float* shared, int elements)
//
// One dot product, a sigmoid, and a scale over a single vector. Small enough
// that the pool handoff would cost more than the work, so it stays serial.
void qwen_shared_scale(const Launch&, void** arguments) {
    const float* input = *reinterpret_cast<const float**>(arguments[0]);
    const float* gate = *reinterpret_cast<const float**>(arguments[1]);
    float* shared = *reinterpret_cast<float**>(arguments[2]);
    const int elements = *reinterpret_cast<const int*>(arguments[3]);

    const float total = dot_f32(input, gate, elements);
    const float scale = 1.0f / (1.0f + std::exp(-total));
    for (int index = 0; index < elements; ++index) shared[index] *= scale;
}

// --- route_topk -----------------------------------------------------------
//
// Corpus signature:
//   void route_topk(const float* logits, int* selected,
//                   float* routing_weights, int experts, int top_k)
//
// Softmax over the expert logits, then top_k by repeated max-and-erase, then
// renormalize the winners. 256 experts and top_k 8 on the real checkpoint.
//
// The selection loop is reproduced exactly, including the strict `>` comparison
// that makes ties resolve to the lowest expert index, and the -1.0f sentinel
// written over a chosen expert. Routing is discrete: picking a different expert
// on a near-tie changes which weights the token is multiplied by, so this is
// one of the kernels where "close enough" is not a defensible standard.
void route_topk(const Launch&, void** arguments) {
    const float* logits = *reinterpret_cast<const float**>(arguments[0]);
    int* selected = *reinterpret_cast<int**>(arguments[1]);
    float* routing_weights = *reinterpret_cast<float**>(arguments[2]);
    const int experts = *reinterpret_cast<const int*>(arguments[3]);
    const int top_k = *reinterpret_cast<const int*>(arguments[4]);

    static thread_local std::vector<float> probabilities;
    if (static_cast<int>(probabilities.size()) < experts)
        probabilities.resize(experts);

    float maximum = -3.402823466e+38F;
    for (int index = 0; index < experts; ++index)
        maximum = std::fmax(maximum, logits[index]);

    float denominator = 0.0f;
    for (int index = 0; index < experts; ++index) {
        const float probability = std::exp(logits[index] - maximum);
        probabilities[index] = probability;
        denominator += probability;
    }
    for (int index = 0; index < experts; ++index)
        probabilities[index] /= denominator;

    float selected_total = 0.0f;
    for (int rank = 0; rank < top_k; ++rank) {
        int best_index = 0;
        float best_value = -1.0f;
        for (int expert = 0; expert < experts; ++expert) {
            if (probabilities[expert] > best_value) {
                best_value = probabilities[expert];
                best_index = expert;
            }
        }
        selected[rank] = best_index;
        routing_weights[rank] = best_value;
        selected_total += best_value;
        probabilities[best_index] = -1.0f;
    }
    for (int rank = 0; rank < top_k; ++rank)
        routing_weights[rank] /= selected_total;
}

// --- route_topk_rows ------------------------------------------------------
//
// Corpus signature:
//   void route_topk_rows(const float* logits, int* selected,
//                        float* routing_weights, int rows, int experts,
//                        int top_k)
//
// The row-batched route_topk used during prefill. Same selection semantics,
// including the strict `>` that resolves ties to the lowest expert index.
void route_topk_rows(const Launch&, void** arguments) {
    const float* logits = *reinterpret_cast<const float**>(arguments[0]);
    int* selected = *reinterpret_cast<int**>(arguments[1]);
    float* routing_weights = *reinterpret_cast<float**>(arguments[2]);
    const int rows = *reinterpret_cast<const int*>(arguments[3]);
    const int experts = *reinterpret_cast<const int*>(arguments[4]);
    const int top_k = *reinterpret_cast<const int*>(arguments[5]);

    parallel_chunks(static_cast<std::uint64_t>(rows), [&](std::uint64_t row) {
        static thread_local std::vector<float> probabilities;
        if (static_cast<int>(probabilities.size()) < experts)
            probabilities.resize(experts);

        const std::int64_t logits_offset =
            static_cast<std::int64_t>(row) * experts;
        const std::int64_t output_offset =
            static_cast<std::int64_t>(row) * top_k;

        float maximum = -3.402823466e+38F;
        for (int index = 0; index < experts; ++index)
            maximum = std::fmax(maximum, logits[logits_offset + index]);

        float denominator = 0.0f;
        for (int index = 0; index < experts; ++index) {
            const float probability =
                std::exp(logits[logits_offset + index] - maximum);
            probabilities[index] = probability;
            denominator += probability;
        }
        for (int index = 0; index < experts; ++index)
            probabilities[index] /= denominator;

        float selected_total = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_index = 0;
            float best_value = -1.0f;
            for (int expert = 0; expert < experts; ++expert) {
                if (probabilities[expert] > best_value) {
                    best_value = probabilities[expert];
                    best_index = expert;
                }
            }
            selected[output_offset + rank] = best_index;
            routing_weights[output_offset + rank] = best_value;
            selected_total += best_value;
            probabilities[best_index] = -1.0f;
        }
        for (int rank = 0; rank < top_k; ++rank)
            routing_weights[output_offset + rank] /= selected_total;
    });
}

// --- qwen_shared_scale_rows -----------------------------------------------
//
// Corpus signature:
//   void qwen_shared_scale_rows(const float* input, const float* gate,
//                               float* shared, int rows, int elements)
void qwen_shared_scale_rows(const Launch&, void** arguments) {
    const float* input = *reinterpret_cast<const float**>(arguments[0]);
    const float* gate = *reinterpret_cast<const float**>(arguments[1]);
    float* shared = *reinterpret_cast<float**>(arguments[2]);
    const int rows = *reinterpret_cast<const int*>(arguments[3]);
    const int elements = *reinterpret_cast<const int*>(arguments[4]);

    parallel_chunks(static_cast<std::uint64_t>(rows), [&](std::uint64_t row) {
        const float* vector = input + static_cast<std::int64_t>(row) * elements;
        float* output = shared + static_cast<std::int64_t>(row) * elements;
        const float total = dot_f32(vector, gate, elements);
        const float scale = 1.0f / (1.0f + std::exp(-total));
        for (int index = 0; index < elements; ++index) output[index] *= scale;
    });
}

// --- attention query/key head norm + RoPE ---------------------------------
//
// Corpus signatures:
//   void qwen_attention_query(const float* projected, const float* norm_weights,
//                             float* queries, float* gates, int heads,
//                             int head_dim, int rotary_dim, int position,
//                             float theta, float epsilon)
//   void qwen_attention_key(const float* projected, const float* norm_weights,
//                           float* keys, int heads, int head_dim,
//                           int rotary_dim, int position, float theta,
//                           float epsilon)
//
// Per-head RMS norm followed by a rotary embedding. The two differ only in the
// source stride (query rows carry an interleaved gate, so 2 * head_dim) and in
// whether the gate half is copied out, so one helper covers both.
//
// The rotation reads the *pre-rotation* normalized partner, which is why the
// normalized values are staged before any of them are written back -- rotating
// in place would feed already-rotated values into later pairs.
inline void attention_head_norm_rope(const float* source, const float* norm_weights,
                                     float* destination, int head_dim,
                                     int rotary_dim, int position, float theta,
                                     float epsilon) {
    // rotary_dim flows from GGUF metadata (rope.dimension_count). The partner
    // lookup below reads up to normalized[rotary_dim - 1] out of a buffer
    // sized head_dim, so a checkpoint claiming a wider rotary span than the
    // head would read past the heap. A span wider than the head is
    // meaningless anyway; clamp rather than trust.
    if (rotary_dim > head_dim) rotary_dim = head_dim;
    static thread_local std::vector<float> normalized;
    if (static_cast<int>(normalized.size()) < head_dim)
        normalized.resize(head_dim);

    double total = 0.0;
    for (int index = 0; index < head_dim; ++index) {
        const double value = source[index];
        total += value * value;
    }
    const float inverse_rms =
        1.0f / std::sqrt(static_cast<float>(total / head_dim) + epsilon);

    for (int index = 0; index < head_dim; ++index)
        normalized[index] = source[index] * inverse_rms * norm_weights[index];

    const int half = rotary_dim / 2;
    for (int index = 0; index < head_dim; ++index) {
        float value = normalized[index];
        if (index < rotary_dim) {
            const int pair = index < half ? index : index - half;
            const float other =
                normalized[index < half ? index + half : index - half];
            const float angle = static_cast<float>(position) /
                std::pow(theta, 2.0f * static_cast<float>(pair) /
                                    static_cast<float>(rotary_dim));
            value = index < half
                ? value * std::cos(angle) - other * std::sin(angle)
                : value * std::cos(angle) + other * std::sin(angle);
        }
        destination[index] = value;
    }
}

void qwen_attention_query(const Launch&, void** arguments) {
    const float* projected = *reinterpret_cast<const float**>(arguments[0]);
    const float* norm_weights = *reinterpret_cast<const float**>(arguments[1]);
    float* queries = *reinterpret_cast<float**>(arguments[2]);
    float* gates = *reinterpret_cast<float**>(arguments[3]);
    const int heads = *reinterpret_cast<const int*>(arguments[4]);
    const int head_dim = *reinterpret_cast<const int*>(arguments[5]);
    const int rotary_dim = *reinterpret_cast<const int*>(arguments[6]);
    const int position = *reinterpret_cast<const int*>(arguments[7]);
    const float theta = *reinterpret_cast<const float*>(arguments[8]);
    const float epsilon = *reinterpret_cast<const float*>(arguments[9]);

    parallel_chunks(static_cast<std::uint64_t>(heads), [&](std::uint64_t head) {
        const float* source =
            projected + static_cast<std::int64_t>(head) * 2 * head_dim;
        float* destination =
            queries + static_cast<std::int64_t>(head) * head_dim;
        attention_head_norm_rope(source, norm_weights, destination, head_dim,
                                 rotary_dim, position, theta, epsilon);
        // The gate half is copied straight through, unnormalized.
        float* gate_out = gates + static_cast<std::int64_t>(head) * head_dim;
        for (int index = 0; index < head_dim; ++index)
            gate_out[index] = source[head_dim + index];
    });
}

void qwen_attention_key(const Launch&, void** arguments) {
    const float* projected = *reinterpret_cast<const float**>(arguments[0]);
    const float* norm_weights = *reinterpret_cast<const float**>(arguments[1]);
    float* keys = *reinterpret_cast<float**>(arguments[2]);
    const int heads = *reinterpret_cast<const int*>(arguments[3]);
    const int head_dim = *reinterpret_cast<const int*>(arguments[4]);
    const int rotary_dim = *reinterpret_cast<const int*>(arguments[5]);
    const int position = *reinterpret_cast<const int*>(arguments[6]);
    const float theta = *reinterpret_cast<const float*>(arguments[7]);
    const float epsilon = *reinterpret_cast<const float*>(arguments[8]);

    parallel_chunks(static_cast<std::uint64_t>(heads), [&](std::uint64_t head) {
        const float* source =
            projected + static_cast<std::int64_t>(head) * head_dim;
        float* destination = keys + static_cast<std::int64_t>(head) * head_dim;
        attention_head_norm_rope(source, norm_weights, destination, head_dim,
                                 rotary_dim, position, theta, epsilon);
    });
}

// --- qwen_delta_recurrent_split -------------------------------------------
//
// Corpus signature:
//   void qwen_delta_recurrent_split(
//       const float* convolved, const float* gates, const float* beta_logits,
//       const float* decay_logits, const float* decay_coefficients,
//       const float* dt_bias, const float* norm_weights, float* state,
//       float* output, int key_heads, int value_heads, int head_dim,
//       float epsilon)
//
// The DeltaNet recurrence, and at 36% the largest single cost in a real decode.
//
// The CUDA version spreads one head across a 2D block of (head_dim output dims
// x slices key groups) with four block-wide reductions and four shared staging
// buffers. Almost all of that is scaffolding for the GPU's execution model. Per
// head the actual work is two sweeps over a head_dim x head_dim state, and both
// are contiguous in the output dimension, so the host version is a plain nested
// loop with the state row streamed once per sweep:
//
//   decay + read   state[k][d] *= decay_scale ; memory[d] += state[k][d]*key[k]
//   delta          delta[d] = (value[d] - memory[d]) * beta
//   update + read  state[k][d] += key[k]*delta[d] ; core[d] += state[k][d]*q[k]
//
// Heads are independent, so the pool parallelizes over them.
//
// The norm reductions deliberately reproduce delta_block_sum's power-of-two
// tree rather than summing sequentially. Those feed rsqrt and then the state
// update, so the result is carried into the next token; matching the reference
// ordering keeps the two backends from diverging over a long generation rather
// than merely within one call.
inline float delta_tree_sum(float* partials, int count) {
    int width = 1;
    while (width < count) width <<= 1;
    for (int index = count; index < width; ++index) partials[index] = 0.0f;
    for (int stride = width >> 1; stride > 0; stride >>= 1)
        for (int index = 0; index < stride; ++index)
            partials[index] += partials[index + stride];
    return partials[0];
}

void qwen_delta_recurrent_split(const Launch&, void** arguments) {
    const float* convolved = *reinterpret_cast<const float**>(arguments[0]);
    const float* gates = *reinterpret_cast<const float**>(arguments[1]);
    const float* beta_logits = *reinterpret_cast<const float**>(arguments[2]);
    const float* decay_logits = *reinterpret_cast<const float**>(arguments[3]);
    const float* decay_coefficients =
        *reinterpret_cast<const float**>(arguments[4]);
    const float* dt_bias = *reinterpret_cast<const float**>(arguments[5]);
    const float* norm_weights = *reinterpret_cast<const float**>(arguments[6]);
    float* state = *reinterpret_cast<float**>(arguments[7]);
    float* output = *reinterpret_cast<float**>(arguments[8]);
    const int key_heads = *reinterpret_cast<const int*>(arguments[9]);
    const int value_heads = *reinterpret_cast<const int*>(arguments[10]);
    const int head_dim = *reinterpret_cast<const int*>(arguments[11]);
    const float epsilon = *reinterpret_cast<const float*>(arguments[12]);

    const int total_key_dim = key_heads * head_dim;

    parallel_chunks(static_cast<std::uint64_t>(value_heads),
                    [&](std::uint64_t head_index) {
        const int head = static_cast<int>(head_index);
        const int key_head = head % key_heads;
        const int key_offset = key_head * head_dim;

        // Per-worker scratch, reused across heads and launches. Sized on first
        // use; head_dim is fixed for a given model.
        static thread_local std::vector<float> query, key, memory, delta, core,
            partials;
        const std::size_t padded = static_cast<std::size_t>(head_dim) * 2 + 2;
        if (query.size() < padded) {
            query.resize(padded); key.resize(padded); memory.resize(padded);
            delta.resize(padded); core.resize(padded); partials.resize(padded);
        }

        for (int dim = 0; dim < head_dim; ++dim) {
            query[dim] = convolved[key_offset + dim];
            key[dim] = convolved[total_key_dim + key_offset + dim];
        }

        for (int dim = 0; dim < head_dim; ++dim)
            partials[dim] = query[dim] * query[dim];
        const float query_square = delta_tree_sum(partials.data(), head_dim);
        for (int dim = 0; dim < head_dim; ++dim)
            partials[dim] = key[dim] * key[dim];
        const float key_square = delta_tree_sum(partials.data(), head_dim);

        const float query_inverse_norm =
            (1.0f / std::sqrt(query_square + 1.0e-6f)) *
            (1.0f / std::sqrt(static_cast<float>(head_dim)));
        const float key_inverse_norm = 1.0f / std::sqrt(key_square + 1.0e-6f);
        const float beta = 1.0f / (1.0f + std::exp(-beta_logits[head]));
        const float softplus_input = decay_logits[head] + dt_bias[head];
        const float softplus = softplus_input > 20.0f
            ? softplus_input
            : std::log1p(std::exp(softplus_input));
        const float decay_scale =
            std::exp(decay_coefficients[head] * softplus);

        for (int dim = 0; dim < head_dim; ++dim) key[dim] *= key_inverse_norm;

        // Sweep one: decay the state and read the memory out of it.
        for (int dim = 0; dim < head_dim; ++dim) memory[dim] = 0.0f;
        for (int k = 0; k < head_dim; ++k) {
            float* row = state + (static_cast<std::int64_t>(head) * head_dim + k) *
                                     head_dim;
            const float key_value = key[k];
            for (int dim = 0; dim < head_dim; ++dim) {
                const float decayed = row[dim] * decay_scale;
                row[dim] = decayed;
                memory[dim] += decayed * key_value;
            }
        }

        for (int dim = 0; dim < head_dim; ++dim) {
            const float value =
                convolved[total_key_dim * 2 + head * head_dim + dim];
            delta[dim] = (value - memory[dim]) * beta;
        }

        // Sweep two: rank-one update, and read the core out of the new state.
        for (int dim = 0; dim < head_dim; ++dim) core[dim] = 0.0f;
        for (int k = 0; k < head_dim; ++k) {
            float* row = state + (static_cast<std::int64_t>(head) * head_dim + k) *
                                     head_dim;
            const float key_value = key[k];
            const float query_value = query[k];
            for (int dim = 0; dim < head_dim; ++dim) {
                const float updated = row[dim] + key_value * delta[dim];
                row[dim] = updated;
                // query_inverse_norm is folded per term, as in the corpus.
                core[dim] += updated * query_value * query_inverse_norm;
            }
        }

        for (int dim = 0; dim < head_dim; ++dim)
            partials[dim] = core[dim] * core[dim];
        const float square = delta_tree_sum(partials.data(), head_dim);
        const float inverse_rms =
            1.0f / std::sqrt(square / static_cast<float>(head_dim) + epsilon);

        for (int dim = 0; dim < head_dim; ++dim) {
            const int output_index = head * head_dim + dim;
            const float gate = gates[output_index];
            const float clamped = std::fmin(80.0f, std::fmax(-80.0f, gate));
            output[output_index] = core[dim] * inverse_rms * norm_weights[dim] *
                                   gate / (1.0f + std::exp(-clamped));
        }
    });
}

// --- qwen_delta_recurrent_chunk -------------------------------------------
//
// Corpus signature:
//   void qwen_delta_recurrent_chunk(
//       const float* convolved, const float* gates, const float* beta_logits,
//       const float* decay_logits, const float* decay_coefficients,
//       const float* dt_bias, const float* norm_weights, float* state,
//       float* output, int rows, int key_heads, int value_heads,
//       int head_dim, float epsilon)
//
// The prefill form of the recurrence: the same per-token update as
// qwen_delta_recurrent_split, but looped over a whole chunk with the state kept
// live between tokens. Hard-wired to head_dim 128; the corpus returns without
// doing anything otherwise, and the runtime falls back to the _rows kernel.
//
// Two differences from _split that matter for matching, both about *when*
// normalization is applied rather than what it computes:
//   * query is scaled by query_inverse_norm once, up front, where _split folds
//     it into each term of the core accumulation;
//   * the norm reductions are a 32-lane shuffle tree per warp, then the four
//     warp results summed left to right -- not _split's power-of-two tree over
//     the whole block.
// Both are reproduced below.
//
// state[(head*128 + key)*128 + lane] is contiguous in lane, so each key is one
// sequential pass over 128 floats and the whole per-head state is 64 KiB, which
// stays resident across the token loop.
inline float warp_tree_sum_32(float* lanes) {
    for (int offset = 16; offset > 0; offset >>= 1)
        for (int lane = 0; lane < offset; ++lane)
            lanes[lane] += lanes[lane + offset];
    return lanes[0];
}

void qwen_delta_recurrent_chunk(const Launch&, void** arguments) {
    const float* convolved = *reinterpret_cast<const float**>(arguments[0]);
    const float* gates = *reinterpret_cast<const float**>(arguments[1]);
    const float* beta_logits = *reinterpret_cast<const float**>(arguments[2]);
    const float* decay_logits = *reinterpret_cast<const float**>(arguments[3]);
    const float* decay_coefficients =
        *reinterpret_cast<const float**>(arguments[4]);
    const float* dt_bias = *reinterpret_cast<const float**>(arguments[5]);
    const float* norm_weights = *reinterpret_cast<const float**>(arguments[6]);
    float* state = *reinterpret_cast<float**>(arguments[7]);
    float* output = *reinterpret_cast<float**>(arguments[8]);
    const int rows = *reinterpret_cast<const int*>(arguments[9]);
    const int key_heads = *reinterpret_cast<const int*>(arguments[10]);
    const int value_heads = *reinterpret_cast<const int*>(arguments[11]);
    const int head_dim = *reinterpret_cast<const int*>(arguments[12]);
    const float epsilon = *reinterpret_cast<const float*>(arguments[13]);

    // Matches the corpus guard exactly: anything else is a silent no-op there.
    if (head_dim != 128) return;
    constexpr int kDim = 128;
    const int total_key_dim = key_heads * kDim;
    const std::int64_t row_stride = total_key_dim * 2 + value_heads * kDim;

    parallel_chunks(static_cast<std::uint64_t>(value_heads),
                    [&](std::uint64_t head_index) {
        const int head = static_cast<int>(head_index);
        const int key_head = head % key_heads;
        const int key_offset = key_head * kDim;
        float* head_state =
            state + static_cast<std::int64_t>(head) * kDim * kDim;

        float shared_key[kDim], shared_query[kDim];
        float memory[kDim], delta[kDim], core[kDim];
        float lanes[32];

        for (int token = 0; token < rows; ++token) {
            const float* row =
                convolved + static_cast<std::int64_t>(token) * row_stride;

            // Per-warp shuffle tree, then the four warp sums left to right.
            float query_sums[4], key_sums[4];
            for (int warp = 0; warp < 4; ++warp) {
                for (int lane = 0; lane < 32; ++lane) {
                    const float value = row[key_offset + warp * 32 + lane];
                    lanes[lane] = value * value;
                }
                query_sums[warp] = warp_tree_sum_32(lanes);
                for (int lane = 0; lane < 32; ++lane) {
                    const float value =
                        row[total_key_dim + key_offset + warp * 32 + lane];
                    lanes[lane] = value * value;
                }
                key_sums[warp] = warp_tree_sum_32(lanes);
            }
            const float query_square =
                query_sums[0] + query_sums[1] + query_sums[2] + query_sums[3];
            const float key_square =
                key_sums[0] + key_sums[1] + key_sums[2] + key_sums[3];

            const float query_inverse_norm =
                (1.0f / std::sqrt(query_square + 1.0e-6f)) *
                (1.0f / std::sqrt(128.0f));
            const float key_inverse_norm =
                1.0f / std::sqrt(key_square + 1.0e-6f);
            const float beta = 1.0f / (1.0f + std::exp(
                -beta_logits[static_cast<std::int64_t>(token) * value_heads + head]));
            const float softplus_input =
                decay_logits[static_cast<std::int64_t>(token) * value_heads + head] +
                dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input
                : std::log1p(std::exp(softplus_input));
            const float decay_scale =
                std::exp(decay_coefficients[head] * softplus);

            for (int lane = 0; lane < kDim; ++lane) {
                shared_key[lane] =
                    row[total_key_dim + key_offset + lane] * key_inverse_norm;
                shared_query[lane] = row[key_offset + lane] * query_inverse_norm;
                memory[lane] = 0.0f;
                core[lane] = 0.0f;
            }

            for (int key = 0; key < kDim; ++key) {
                float* column = head_state + static_cast<std::int64_t>(key) * kDim;
                const float key_value = shared_key[key];
                for (int lane = 0; lane < kDim; ++lane) {
                    const float decayed = column[lane] * decay_scale;
                    column[lane] = decayed;
                    memory[lane] += decayed * key_value;
                }
            }

            for (int lane = 0; lane < kDim; ++lane) {
                const float value = row[total_key_dim * 2 + head * kDim + lane];
                delta[lane] = (value - memory[lane]) * beta;
            }

            for (int key = 0; key < kDim; ++key) {
                float* column = head_state + static_cast<std::int64_t>(key) * kDim;
                const float key_value = shared_key[key];
                const float query_value = shared_query[key];
                for (int lane = 0; lane < kDim; ++lane) {
                    const float updated = column[lane] + key_value * delta[lane];
                    column[lane] = updated;
                    core[lane] += updated * query_value;
                }
            }

            float core_sums[4];
            for (int warp = 0; warp < 4; ++warp) {
                for (int lane = 0; lane < 32; ++lane) {
                    const float value = core[warp * 32 + lane];
                    lanes[lane] = value * value;
                }
                core_sums[warp] = warp_tree_sum_32(lanes);
            }
            const float square =
                core_sums[0] + core_sums[1] + core_sums[2] + core_sums[3];
            const float inverse_rms =
                1.0f / std::sqrt(square / 128.0f + epsilon);

            const std::int64_t output_base =
                static_cast<std::int64_t>(token) * value_heads * kDim +
                head * kDim;
            for (int lane = 0; lane < kDim; ++lane) {
                const float gate = gates[output_base + lane];
                const float clamped = std::fmin(80.0f, std::fmax(-80.0f, gate));
                output[output_base + lane] = core[lane] * inverse_rms *
                                             norm_weights[lane] * gate /
                                             (1.0f + std::exp(-clamped));
            }
        }
    });
}

// --- q8_swiglu_transposed_warp --------------------------------------------
//
// Corpus signature:
//   void q8_swiglu_transposed_warp(const unsigned char* gate_packed,
//                                  const unsigned char* up_packed,
//                                  const float* vector, float* output,
//                                  int input_size, int output_size)
//
// 18% of decode on the real checkpoint. Gate and up are walked together in one
// pass so the activation slice is loaded once and stays in L1 across both
// weight streams -- the CUDA version reloads it per matrix because its lanes
// are strided, which costs nothing there and would cost bandwidth here.
void q8_swiglu_transposed_warp(const Launch&, void** arguments) {
    const unsigned char* gate_packed =
        *reinterpret_cast<const unsigned char**>(arguments[0]);
    const unsigned char* up_packed =
        *reinterpret_cast<const unsigned char**>(arguments[1]);
    const float* vector = *reinterpret_cast<const float**>(arguments[2]);
    float* output = *reinterpret_cast<float**>(arguments[3]);
    const int input_size = *reinterpret_cast<const int*>(arguments[4]);
    const int output_size = *reinterpret_cast<const int*>(arguments[5]);

    parallel_chunks(static_cast<std::uint64_t>(output_size),
                    [&](std::uint64_t row) {
        float gate = 0.0f;
        float up = 0.0f;
        q8_row_dot_pair(gate_packed, up_packed, static_cast<std::int64_t>(row),
                        vector, input_size, &gate, &up);

        // Reproduced branch for branch: the corpus picks the sigmoid form by
        // sign to keep expf out of its overflow range, and the two forms are
        // not bit-identical to each other.
        const float exponential =
            gate >= 0.0f ? std::exp(-gate) : std::exp(gate);
        const float sigmoid = gate >= 0.0f
            ? 1.0f / (1.0f + exponential)
            : exponential / (1.0f + exponential);
        output[row] = gate * sigmoid * up;
    });
}

// --- kv_attention_values_*_ring -------------------------------------------
//
// Corpus signature (one per cache element type):
//   void kv_attention_values_f16_ring(float* scores, const __half* values,
//                                     float* output, int heads, int kv_heads,
//                                     int head_dim, int tokens, int capacity,
//                                     int first)
//
// Softmax over one head's attention scores, then the score-weighted sum of the
// value vectors. `first` and `capacity` describe the circular KV window, so
// logical token t lives at slot (first + t) % capacity.
//
// Measured at 36% of decode while emulated, at 3.2 ms per call on a 37-token
// context where the arithmetic is a few microseconds' worth. All of that was
// the cooperative-kernel path: two block reductions and four barriers mean
// fiber switching dominates completely.
//
// The scores buffer is normalized in place, as in the corpus, because the
// caller reads it back.
template <class Element, class Load>
inline void kv_values_ring_host(float* scores, const Element* values,
                                float* output, int heads, int kv_heads,
                                int head_dim, int tokens, int capacity,
                                int first, Load load) {
    // Degenerate shapes are gated upstream, but a zero kv_heads or capacity
    // reaching this far would be a SIGFPE, not an error; refuse them here so
    // a bad launch is a no-op instead of a crash.
    if (heads <= 0 || kv_heads <= 0 || heads < kv_heads || capacity <= 0)
        return;
    const int group = heads / kv_heads;
    parallel_chunks(static_cast<std::uint64_t>(heads), [&](std::uint64_t head) {
        const int kv_head = static_cast<int>(head) / group;
        float* head_scores = scores + static_cast<std::int64_t>(head) * tokens;

        float maximum = -3.402823466e+38F;
        for (int token = 0; token < tokens; ++token)
            maximum = std::fmax(maximum, head_scores[token]);

        float denominator = 0.0f;
        for (int token = 0; token < tokens; ++token) {
            const float weight = std::exp(head_scores[token] - maximum);
            head_scores[token] = weight;
            denominator += weight;
        }
        const float inverse_denominator = 1.0f / denominator;
        for (int token = 0; token < tokens; ++token)
            head_scores[token] *= inverse_denominator;

        // Token-outer, dim-inner: each value row is contiguous in `d`, so this
        // walks the cache sequentially instead of striding per output element.
        float* out = output + static_cast<std::int64_t>(head) * head_dim;
        for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
        for (int token = 0; token < tokens; ++token) {
            const int slot = (first + token) % capacity;
            const Element* row = values +
                (static_cast<std::int64_t>(kv_head) * capacity + slot) * head_dim;
            const float weight = head_scores[token];
            for (int d = 0; d < head_dim; ++d) out[d] += weight * load(row[d]);
        }
    });
}

void kv_attention_values_f16_ring(const Launch&, void** arguments) {
    float* scores = *reinterpret_cast<float**>(arguments[0]);
    const auto* values = *reinterpret_cast<const std::uint16_t**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int heads = *reinterpret_cast<const int*>(arguments[3]);
    const int kv_heads = *reinterpret_cast<const int*>(arguments[4]);
    const int head_dim = *reinterpret_cast<const int*>(arguments[5]);
    const int tokens = *reinterpret_cast<const int*>(arguments[6]);
    const int capacity = *reinterpret_cast<const int*>(arguments[7]);
    const int first = *reinterpret_cast<const int*>(arguments[8]);
    kv_values_ring_host(scores, values, output, heads, kv_heads, head_dim,
                        tokens, capacity, first,
                        [](std::uint16_t bits) { return half_bits_to_float(bits); });
}

void kv_attention_values_ring(const Launch&, void** arguments) {
    float* scores = *reinterpret_cast<float**>(arguments[0]);
    const float* values = *reinterpret_cast<const float**>(arguments[1]);
    float* output = *reinterpret_cast<float**>(arguments[2]);
    const int heads = *reinterpret_cast<const int*>(arguments[3]);
    const int kv_heads = *reinterpret_cast<const int*>(arguments[4]);
    const int head_dim = *reinterpret_cast<const int*>(arguments[5]);
    const int tokens = *reinterpret_cast<const int*>(arguments[6]);
    const int capacity = *reinterpret_cast<const int*>(arguments[7]);
    const int first = *reinterpret_cast<const int*>(arguments[8]);
    kv_values_ring_host(scores, values, output, heads, kv_heads, head_dim,
                        tokens, capacity, first, [](float v) { return v; });
}

// --- q8_lm_head_argmax_warp -----------------------------------------------
//
// Corpus signature:
//   void q8_lm_head_argmax_warp(const unsigned char* packed,
//                               const float* vector,
//                               unsigned long long* winners,
//                               int input_size, int output_size)
//
// Measured at 45% of the launch time on a real 35B checkpoint -- 975 ms per
// call, because the vocabulary is 248k rows wide. Nothing about it is subtle;
// it was simply the largest emulated matvec in the model.
//
// The packing is load-bearing and reproduced exactly: the fp32 logit is mapped
// to an order-preserving uint32, placed in the high half, and the low half
// holds (0xffffffff - row) so that a plain unsigned max breaks ties toward the
// lowest row index. The kernel maxes into winners[0] rather than assigning,
// because the runtime seeds it and both call sites rely on that.
void q8_lm_head_argmax_warp(const Launch&, void** arguments) {
    const unsigned char* packed =
        *reinterpret_cast<const unsigned char**>(arguments[0]);
    const float* vector = *reinterpret_cast<const float**>(arguments[1]);
    auto* winners = *reinterpret_cast<unsigned long long**>(arguments[2]);
    const int input_size = *reinterpret_cast<const int*>(arguments[3]);
    const int output_size = *reinterpret_cast<const int*>(arguments[4]);

    std::atomic<unsigned long long> best{0};
    parallel_chunks(static_cast<std::uint64_t>(output_size),
                    [&](std::uint64_t row) {
        const float logit =
            q8_row_dot(packed, static_cast<std::int64_t>(row), vector, input_size);

        std::uint32_t bits;
        std::memcpy(&bits, &logit, sizeof(bits));
        const std::uint32_t ordered =
            bits ^ ((static_cast<std::int32_t>(bits) < 0) ? 0xffffffffu
                                                          : 0x80000000u);
        const unsigned long long candidate =
            (static_cast<unsigned long long>(ordered) << 32) |
            (0xffffffffu - static_cast<std::uint32_t>(row));

        // No fetch_max before C++26; the CAS retries only when a strictly
        // better candidate lands concurrently, which is rare across 64 chunks.
        unsigned long long current = best.load(std::memory_order_relaxed);
        while (candidate > current &&
               !best.compare_exchange_weak(current, candidate,
                                           std::memory_order_relaxed)) {
        }
    });

    const unsigned long long found = best.load(std::memory_order_relaxed);
    if (found > *winners) *winners = found;
}

}  // namespace

COLIBRI_CPU_NATIVE_KERNEL("rms_norm_rows", rms_norm_rows);
COLIBRI_CPU_NATIVE_KERNEL("qwen_shared_scale", qwen_shared_scale);
COLIBRI_CPU_NATIVE_KERNEL("route_topk", route_topk);
COLIBRI_CPU_NATIVE_KERNEL("route_topk_rows", route_topk_rows);
COLIBRI_CPU_NATIVE_KERNEL("qwen_shared_scale_rows", qwen_shared_scale_rows);
COLIBRI_CPU_NATIVE_KERNEL("qwen_attention_query", qwen_attention_query);
COLIBRI_CPU_NATIVE_KERNEL("qwen_attention_key", qwen_attention_key);
// --- qwen_imatrix_accumulate ----------------------------------------------
//
// Corpus signature:
//   void qwen_imatrix_accumulate(const float* input, float* sums, int width,
//                               int rows)
//
// Column-major accumulation over a row-major batch, so the host version walks
// rows outermost and lets the inner loop stream contiguously -- the transpose
// of the GPU's mapping, and the reason this is not simply the same loop.
void qwen_imatrix_accumulate(const Launch&, void** arguments) {
    const float* input = *reinterpret_cast<const float**>(arguments[0]);
    float* sums = *reinterpret_cast<float**>(arguments[1]);
    const int width = *reinterpret_cast<const int*>(arguments[2]);
    const int rows = *reinterpret_cast<const int*>(arguments[3]);
    for (int row = 0; row < rows; ++row) {
        const float* source = input + static_cast<std::int64_t>(row) * width;
        for (int column = 0; column < width; ++column)
            sums[column] += source[column] * source[column];
    }
}

COLIBRI_CPU_NATIVE_KERNEL("qwen_imatrix_accumulate", qwen_imatrix_accumulate);
COLIBRI_CPU_NATIVE_KERNEL("qwen_delta_recurrent_split", qwen_delta_recurrent_split);
COLIBRI_CPU_NATIVE_KERNEL("qwen_delta_recurrent_chunk", qwen_delta_recurrent_chunk);
COLIBRI_CPU_NATIVE_KERNEL("q8_swiglu_transposed_warp", q8_swiglu_transposed_warp);
COLIBRI_CPU_NATIVE_KERNEL("kv_attention_values_f16_ring", kv_attention_values_f16_ring);
COLIBRI_CPU_NATIVE_KERNEL("kv_attention_values_ring", kv_attention_values_ring);
COLIBRI_CPU_NATIVE_KERNEL("q8_lm_head_argmax_warp", q8_lm_head_argmax_warp);
COLIBRI_CPU_NATIVE_KERNEL("q8_matvec_transposed_warp", q8_matvec_transposed_warp);
COLIBRI_CPU_NATIVE_KERNEL("q8_matmul_tiled", q8_matmul_tiled);
COLIBRI_CPU_NATIVE_KERNEL("rms_norm", rms_norm);
COLIBRI_CPU_NATIVE_KERNEL("scaled_add", scaled_add);
COLIBRI_CPU_NATIVE_KERNEL("qwen_f32_matvec_warp", qwen_f32_matvec_warp);
COLIBRI_CPU_NATIVE_KERNEL("qwen_f32_matmul_rows", qwen_f32_matmul_rows);

}  // namespace colibri::cpu
