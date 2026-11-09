// Parity contract: every native host kernel must agree with the emulated
// corpus kernel of the same name.
//
// The emulated kernel is the same CUDA text the GPU compiles, so agreeing with
// it is the strongest correctness statement available without a GPU present --
// and it makes this test runnable in CI on any machine.
//
// Adding a native kernel means adding a case here. A case allocates inputs,
// runs the launch twice against separate output buffers (once with emulation
// forced, once normally) and compares. Kernels registered without a case are
// reported as untested and fail the run, so the two cannot drift apart.

#include <flyweight_backend.hpp>
#include <flyweight_cpu_kernels_api.hpp>
#include <flyweight_cpu_native.hpp>
#include <flyweight_cpu_shim_geometry.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

extern "C" {
int flyweight_cpu_launch_named(const char*, std::uint32_t, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint64_t,
                             void**);
}

namespace {

struct Case {
    std::string kernel;
    // Runs the launch; `outputs` is filled with whatever the case wants
    // compared. Called once per backend with a fresh set of buffers.
    std::function<void(std::vector<std::vector<float>>& outputs)> run;
    float tolerance = 1e-5f;
};

std::vector<Case>& cases() {
    static std::vector<Case> registry;
    return registry;
}

std::mt19937& rng() {
    static std::mt19937 generator(20260802);
    return generator;
}

std::vector<float> random_vector(std::size_t count, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (auto& value : values) value = distribution(rng());
    return values;
}

int run_case(const Case& item) {
    // Same seed for both runs so the inputs are identical.
    const auto seed = rng();

    rng() = seed;
    flyweight::cpu::set_force_emulation(true);
    std::vector<std::vector<float>> reference;
    item.run(reference);

    rng() = seed;
    flyweight::cpu::set_force_emulation(false);
    std::vector<std::vector<float>> native;
    item.run(native);

    flyweight::cpu::set_force_emulation(false);

    if (reference.size() != native.size()) {
        std::printf("  %-28s FAIL (output count %zu vs %zu)\n",
                    item.kernel.c_str(), reference.size(), native.size());
        return 1;
    }

    double worst = 0.0;
    for (std::size_t buffer = 0; buffer < reference.size(); ++buffer) {
        if (reference[buffer].size() != native[buffer].size()) {
            std::printf("  %-28s FAIL (buffer %zu size mismatch)\n",
                        item.kernel.c_str(), buffer);
            return 1;
        }
        for (std::size_t index = 0; index < reference[buffer].size(); ++index) {
            const double expected = reference[buffer][index];
            const double actual = native[buffer][index];
            if (std::isnan(expected) != std::isnan(actual)) {
                std::printf("  %-28s FAIL (NaN mismatch at %zu)\n",
                            item.kernel.c_str(), index);
                return 1;
            }
            if (std::isnan(expected)) continue;
            // Relative where the magnitudes are large, absolute near zero.
            const double scale = std::fmax(1.0, std::fabs(expected));
            worst = std::fmax(worst, std::fabs(expected - actual) / scale);
        }
    }

    const bool ok = worst <= item.tolerance;
    std::printf("  %-28s %s (worst %.3e, tol %.0e)\n", item.kernel.c_str(),
                ok ? "OK  " : "FAIL", worst, item.tolerance);
    return ok ? 0 : 1;
}

// --- cases ----------------------------------------------------------------

void add_cases() {
    cases().push_back(Case{
        "rms_norm",
        [](std::vector<std::vector<float>>& outputs) {
            // Widths on and off the 256-thread block boundary, so the emulated
            // reduction tree is exercised with and without a ragged tail.
            for (int elements : {256, 1024, 1536, 4096}) {
                for (int one_centered : {0, 1}) {
                    auto input = random_vector(elements, -4.0f, 4.0f);
                    auto weights = random_vector(elements, 0.25f, 1.75f);
                    std::vector<float> output(elements, 0.0f);

                    const float* input_pointer = input.data();
                    const float* weight_pointer = weights.data();
                    float* output_pointer = output.data();
                    float epsilon = 1e-6f;
                    void* arguments[] = {&input_pointer, &weight_pointer,
                                         &output_pointer, &elements, &epsilon,
                                         &one_centered};
                    flyweight_cpu_launch_named("rms_norm", 1, 1, 256, 0, 0,
                                             arguments);
                    outputs.push_back(std::move(output));
                }
            }
        },
        // The emulated path sums through a shuffle tree and the native path in
        // double; the orderings differ, so exact equality is not the contract.
        2e-6f,
    });

    cases().push_back(Case{
        "scaled_add",
        [](std::vector<std::vector<float>>& outputs) {
            for (int elements : {1, 255, 256, 4096, 5000}) {
                auto target = random_vector(elements, -2.0f, 2.0f);
                auto source = random_vector(elements, -2.0f, 2.0f);
                float scale = 0.6875f;

                float* target_pointer = target.data();
                const float* source_pointer = source.data();
                void* arguments[] = {&target_pointer, &source_pointer, &scale,
                                     &elements};
                flyweight_cpu_launch_named("scaled_add",
                                         (elements + 255) / 256, 1, 256, 0, 0,
                                         arguments);
                outputs.push_back(std::move(target));
            }
        },
        // Identical operation order on both paths.
        0.0f,
    });

    cases().push_back(Case{
        "qwen_f32_matvec_warp",
        [](std::vector<std::vector<float>>& outputs) {
            // Widths on and off the float4 boundary the corpus fast-path needs,
            // so both its vectorized and scalar branches are compared.
            const int shapes[][2] = {
                {256, 64}, {1024, 256}, {1023, 33}, {512, 8}, {130, 7},
            };
            for (const auto& shape : shapes) {
                int input_size = shape[0];
                int output_size = shape[1];
                auto matrix = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto input = random_vector(input_size, -1.0f, 1.0f);
                std::vector<float> output(output_size, 0.0f);

                const float* matrix_pointer = matrix.data();
                const float* input_pointer = input.data();
                float* output_pointer = output.data();
                void* arguments[] = {&matrix_pointer, &input_pointer,
                                     &output_pointer, &input_size, &output_size};
                flyweight_cpu_launch_named("qwen_f32_matvec_warp",
                                         (output_size + 7) / 8, 1, 256, 0, 0,
                                         arguments);
                outputs.push_back(std::move(output));
            }
        },
        // Both sides reduce in a tree, but not the same tree: the corpus uses a
        // 32-lane shuffle, the native kernel four accumulators. The residual is
        // fp32 reassociation, not a difference in what is computed.
        1e-5f,
    });

    cases().push_back(Case{
        "qwen_f32_matmul_rows",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][3] = {
                {256, 64, 1}, {512, 128, 7}, {255, 33, 4}, {128, 16, 32},
            };
            for (const auto& shape : shapes) {
                int input_size = shape[0];
                int output_size = shape[1];
                int rows = shape[2];
                auto matrix = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto input = random_vector(
                    static_cast<std::size_t>(input_size) * rows, -1.0f, 1.0f);
                std::vector<float> output(
                    static_cast<std::size_t>(output_size) * rows, 0.0f);

                const float* matrix_pointer = matrix.data();
                const float* input_pointer = input.data();
                float* output_pointer = output.data();
                void* arguments[] = {&matrix_pointer, &input_pointer,
                                     &output_pointer, &input_size,
                                     &output_size, &rows};
                flyweight_cpu_launch_named("qwen_f32_matmul_rows", output_size,
                                         rows, 256, 0, 0, arguments);
                outputs.push_back(std::move(output));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "f32_matmul_tiled64",
        [](std::vector<std::vector<float>>& outputs) {
            // On and off the 64x64 tile in both directions, and an input
            // width off the 16-wide K step.
            const int shapes[][3] = {
                {256, 64, 1}, {512, 130, 7}, {250, 33, 70}, {2048, 64, 64},
            };
            for (const auto& shape : shapes) {
                int input_size = shape[0];
                int output_size = shape[1];
                int rows = shape[2];
                auto matrix = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto input = random_vector(
                    static_cast<std::size_t>(input_size) * rows, -1.0f, 1.0f);
                std::vector<float> output(
                    static_cast<std::size_t>(output_size) * rows, -7.0f);
                const float* matrix_pointer = matrix.data();
                const float* input_pointer = input.data();
                float* output_pointer = output.data();
                void* arguments[] = {&matrix_pointer, &input_pointer,
                                     &output_pointer, &input_size,
                                     &output_size, &rows};
                flyweight_cpu_launch_named("f32_matmul_tiled64",
                                           (output_size + 63) / 64,
                                           (rows + 63) / 64, 256, 0, 0, arguments);
                outputs.push_back(std::move(output));
            }
        },
        // The corpus accumulates each dot sequentially along K with fmaf, so
        // its own error grows with input_size; the host's split accumulators
        // sit closer to exact. Same reasoning as q8_matmul_tiled below.
        1e-4f,
    });

    // Q8_0 blocks: fp16 scale + 32 int8, 34 bytes. Built here rather than
    // dequantized from floats so the packing the kernels read is exactly the
    // packing a GGUF carries.
    auto pack_q8 = [](const std::vector<float>& values, int input_size,
                      int output_size) {
        std::vector<unsigned char> packed(
            static_cast<std::size_t>(input_size) * output_size / 32 * 34);
        std::size_t cursor = 0;
        for (int row = 0; row < output_size; ++row) {
            for (int block = 0; block < input_size / 32; ++block) {
                const float* slice =
                    values.data() + static_cast<std::size_t>(row) * input_size +
                    block * 32;
                float absmax = 0.0f;
                for (int index = 0; index < 32; ++index)
                    absmax = std::fmax(absmax, std::fabs(slice[index]));
                const float scale = absmax / 127.0f;
                const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
                const std::uint16_t bits =
                    flyweight::cpu::float_to_half_bits(scale);
                std::memcpy(packed.data() + cursor, &bits, sizeof(bits));
                for (int index = 0; index < 32; ++index) {
                    const float scaled = slice[index] * inverse;
                    const int rounded = static_cast<int>(std::lrintf(scaled));
                    packed[cursor + 2 + index] = static_cast<unsigned char>(
                        static_cast<signed char>(
                            rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded)));
                }
                cursor += 34;
            }
        }
        return packed;
    };

    cases().push_back(Case{
        "q8_matvec_transposed_warp",
        [pack_q8](std::vector<std::vector<float>>& outputs) {
            const int shapes[][2] = {
                {256, 64}, {1024, 129}, {32, 8}, {2816, 33},
            };
            for (const auto& shape : shapes) {
                int input_size = shape[0];
                int output_size = shape[1];
                auto weights = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto packed = pack_q8(weights, input_size, output_size);
                auto vector = random_vector(input_size, -1.0f, 1.0f);
                std::vector<float> output(output_size, 0.0f);

                const unsigned char* packed_pointer = packed.data();
                const float* vector_pointer = vector.data();
                float* output_pointer = output.data();
                void* arguments[] = {&packed_pointer, &vector_pointer,
                                     &output_pointer, &input_size, &output_size};
                flyweight_cpu_launch_named("q8_matvec_transposed_warp",
                                         (output_size + 7) / 8, 1, 256, 0, 0,
                                         arguments);
                outputs.push_back(std::move(output));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "q8_matmul_tiled",
        [pack_q8](std::vector<std::vector<float>>& outputs) {
            // Shapes on and off the corpus kernel's 32x32 tile, so its ragged
            // edge handling is compared too.
            const int shapes[][3] = {
                {256, 64, 32}, {512, 33, 5}, {64, 32, 1}, {1024, 96, 40},
            };
            for (const auto& shape : shapes) {
                int input_size = shape[0];
                int output_size = shape[1];
                int rows = shape[2];
                auto weights = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto packed = pack_q8(weights, input_size, output_size);
                auto input = random_vector(
                    static_cast<std::size_t>(input_size) * rows, -1.0f, 1.0f);
                std::vector<float> output(
                    static_cast<std::size_t>(output_size) * rows, 0.0f);

                const unsigned char* packed_pointer = packed.data();
                const float* input_pointer = input.data();
                float* output_pointer = output.data();
                void* arguments[] = {&packed_pointer, &input_pointer,
                                     &output_pointer, &input_size,
                                     &output_size, &rows};
                flyweight_cpu_launch_named("q8_matmul_tiled",
                                         (output_size + 31) / 32,
                                         (rows + 31) / 32, 256, 0, 0, arguments);
                outputs.push_back(std::move(output));
            }
        },
        // Looser than the other kernels, and the reason is the reference, not
        // the native version. q8_matmul_tiled dequantizes into a tile and then
        // accumulates sequentially in fp32 across the whole row, so its own
        // error grows with input_size -- ~6e-5 relative at K=1024 in the worst
        // case. Measured against an exact double reference over 300 random
        // K=1024 rows, the native block-wise form is closer to exact in 242 of
        // them (mean |err| 1.0e-06 against the reference's 4.0e-06). The gap
        // this tolerance admits is the reference's noise; a real defect in a
        // matmul shows up as O(1) relative error, nowhere near this floor.
        1e-4f,
    });

    cases().push_back(Case{
        "q8_lm_head_argmax_warp",
        [pack_q8](std::vector<std::vector<float>>& outputs) {
            // Includes a vocabulary-shaped case, since the packing has to break
            // ties by row index consistently across hundreds of thousands of
            // rows, not just a handful.
            const int shapes[][2] = {
                {256, 64}, {2048, 1024}, {512, 4001}, {64, 8},
            };
            for (const auto& shape : shapes) {
                int input_size = shape[0];
                int output_size = shape[1];
                auto weights = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto packed = pack_q8(weights, input_size, output_size);
                auto vector = random_vector(input_size, -1.0f, 1.0f);

                // Seeded by the runtime; the kernel only ever maxes into it.
                unsigned long long winner = 0ull;
                const unsigned char* packed_pointer = packed.data();
                const float* vector_pointer = vector.data();
                unsigned long long* winner_pointer = &winner;
                void* arguments[] = {&packed_pointer, &vector_pointer,
                                     &winner_pointer, &input_size, &output_size};
                flyweight_cpu_launch_named("q8_lm_head_argmax_warp",
                                         (output_size + 7) / 8, 1, 256, 0, 0,
                                         arguments);
                // Compare the decoded row, not the raw word: an argmax that
                // picks a different token is the failure that matters, and a
                // one-ulp logit difference that keeps the same winner is not.
                const std::uint32_t row =
                    0xffffffffu - static_cast<std::uint32_t>(winner & 0xffffffffu);
                outputs.push_back({static_cast<float>(row)});
            }
        },
        // The winning row must match exactly.
        0.0f,
    });

    cases().push_back(Case{
        "q8_swiglu_transposed_warp",
        [pack_q8](std::vector<std::vector<float>>& outputs) {
            const int shapes[][2] = {
                {2048, 512}, {256, 33}, {512, 8}, {64, 1},
            };
            for (const auto& shape : shapes) {
                int input_size = shape[0];
                int output_size = shape[1];
                auto gate = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto up = random_vector(
                    static_cast<std::size_t>(input_size) * output_size, -1.0f, 1.0f);
                auto gate_packed = pack_q8(gate, input_size, output_size);
                auto up_packed = pack_q8(up, input_size, output_size);
                auto vector = random_vector(input_size, -1.0f, 1.0f);
                std::vector<float> output(output_size, 0.0f);

                const unsigned char* gate_pointer = gate_packed.data();
                const unsigned char* up_pointer = up_packed.data();
                const float* vector_pointer = vector.data();
                float* output_pointer = output.data();
                void* arguments[] = {&gate_pointer, &up_pointer, &vector_pointer,
                                     &output_pointer, &input_size, &output_size};
                flyweight_cpu_launch_named("q8_swiglu_transposed_warp",
                                         (output_size + 7) / 8, 1, 256, 0, 0,
                                         arguments);
                outputs.push_back(std::move(output));
            }
        },
        // Reduction reassociation again, amplified because the output is
        // silu(gate) * up: an error in the gate dot is scaled by |up|. Measured
        // against reduction length, which is what makes it rounding rather than
        // a defect -- worst relative error 2.98e-08 at K=64, 6.91e-07 at K=256,
        // 3.05e-05 at K=2048. A logic error here (gate/up swapped, wrong
        // sigmoid branch, bad block indexing) would be O(1) and K-independent.
        1e-4f,
    });

    cases().push_back(Case{
        "qwen_delta_recurrent_split",
        [](std::vector<std::vector<float>>& outputs) {
            // value_heads > key_heads exercises the head % key_heads sharing;
            // head_dim off a power of two exercises the reduction tree padding.
            const int shapes[][3] = {
                {4, 8, 64}, {2, 2, 32}, {4, 8, 48}, {1, 4, 128},
            };
            for (const auto& shape : shapes) {
                int key_heads = shape[0];
                int value_heads = shape[1];
                int head_dim = shape[2];
                const int total_key_dim = key_heads * head_dim;

                auto convolved = random_vector(
                    total_key_dim * 2 + value_heads * head_dim, -1.0f, 1.0f);
                auto gates = random_vector(value_heads * head_dim, -2.0f, 2.0f);
                auto beta_logits = random_vector(value_heads, -2.0f, 2.0f);
                auto decay_logits = random_vector(value_heads, -2.0f, 2.0f);
                // Negative, as the trained coefficients are, so decay_scale
                // stays below one and the recurrence is stable.
                auto decay_coefficients = random_vector(value_heads, -2.0f, -0.1f);
                auto dt_bias = random_vector(value_heads, -1.0f, 1.0f);
                auto norm_weights = random_vector(head_dim, 0.5f, 1.5f);
                auto state = random_vector(
                    static_cast<std::size_t>(value_heads) * head_dim * head_dim,
                    -0.5f, 0.5f);
                std::vector<float> output(value_heads * head_dim, 0.0f);

                const float* convolved_p = convolved.data();
                const float* gates_p = gates.data();
                const float* beta_p = beta_logits.data();
                const float* decay_p = decay_logits.data();
                const float* coeff_p = decay_coefficients.data();
                const float* dt_p = dt_bias.data();
                const float* norm_p = norm_weights.data();
                float* state_p = state.data();
                float* output_p = output.data();
                float epsilon = 1e-6f;
                // Both gate activations: silu (qwen3.5 lineage) and sigmoid
                // (qwen4exp's output_gate_type). Derived from the shape index
                // so the native and emulated passes see identical flags.
                int gate_sigmoid = static_cast<int>(&shape - shapes) & 1;
                void* arguments[] = {&convolved_p, &gates_p, &beta_p, &decay_p,
                                     &coeff_p, &dt_p, &norm_p, &state_p,
                                     &output_p, &key_heads, &value_heads,
                                     &head_dim, &epsilon, &gate_sigmoid};
                // Block geometry exactly as v2_runtime.cpp computes it. The
                // kernel has an implicit precondition that blockDim.x is a
                // multiple of head_dim -- otherwise the trailing threads land
                // in a slice the reduction never reads, and decay the state
                // twice. The runtime always satisfies it; a test that launches
                // a round 256 threads does not, for head_dim like 48.
                int slices = 1024 / head_dim;
                if (slices > 4) slices = 4;
                const int block = head_dim * slices;
                const std::uint32_t shared =
                    (head_dim * 4 + block) * sizeof(float);
                flyweight_cpu_launch_named("qwen_delta_recurrent_split",
                                         value_heads, 1, block, shared, 0,
                                         arguments);
                // The mutated state matters as much as the output: it is what
                // the next token reads.
                outputs.push_back(std::move(output));
                outputs.push_back(std::move(state));
            }
        },
        1e-4f,
    });

    cases().push_back(Case{
        "qwen_delta_recurrent_chunk",
        [](std::vector<std::vector<float>>& outputs) {
            // head_dim is fixed at 128 by the kernel itself. Multi-token runs
            // matter most: the state carries between tokens, so a mistake in
            // the loop shows up only after the first.
            const int shapes[][3] = {{4, 8, 1}, {4, 8, 5}, {2, 2, 17}};
            for (const auto& shape : shapes) {
                int key_heads = shape[0];
                int value_heads = shape[1];
                int rows = shape[2];
                int head_dim = 128;
                const int total_key_dim = key_heads * head_dim;
                const std::size_t row_stride =
                    total_key_dim * 2 + value_heads * head_dim;

                auto convolved = random_vector(row_stride * rows, -1.0f, 1.0f);
                auto gates = random_vector(
                    static_cast<std::size_t>(rows) * value_heads * head_dim,
                    -2.0f, 2.0f);
                auto beta_logits = random_vector(
                    static_cast<std::size_t>(rows) * value_heads, -2.0f, 2.0f);
                auto decay_logits = random_vector(
                    static_cast<std::size_t>(rows) * value_heads, -2.0f, 2.0f);
                auto decay_coefficients =
                    random_vector(value_heads, -2.0f, -0.1f);
                auto dt_bias = random_vector(value_heads, -1.0f, 1.0f);
                auto norm_weights = random_vector(head_dim, 0.5f, 1.5f);
                auto state = random_vector(
                    static_cast<std::size_t>(value_heads) * head_dim * head_dim,
                    -0.5f, 0.5f);
                std::vector<float> output(
                    static_cast<std::size_t>(rows) * value_heads * head_dim, 0.0f);

                const float* convolved_p = convolved.data();
                const float* gates_p = gates.data();
                const float* beta_p = beta_logits.data();
                const float* decay_p = decay_logits.data();
                const float* coeff_p = decay_coefficients.data();
                const float* dt_p = dt_bias.data();
                const float* norm_p = norm_weights.data();
                float* state_p = state.data();
                float* output_p = output.data();
                float epsilon = 1e-6f;
                int gate_sigmoid = static_cast<int>(&shape - shapes) & 1;
                void* arguments[] = {&convolved_p, &gates_p, &beta_p, &decay_p,
                                     &coeff_p, &dt_p, &norm_p, &state_p,
                                     &output_p, &rows, &key_heads, &value_heads,
                                     &head_dim, &epsilon, &gate_sigmoid};
                flyweight_cpu_launch_named("qwen_delta_recurrent_chunk",
                                         value_heads, 1, 128, 0, 0, arguments);
                outputs.push_back(std::move(output));
                outputs.push_back(std::move(state));
            }
        },
        1e-4f,
    });

    cases().push_back(Case{
        "rms_norm_rows",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][2] = {{2048, 1}, {256, 9}, {1024, 40}, {33, 3}};
            for (const auto& shape : shapes) {
                int columns = shape[0];
                int rows = shape[1];
                for (int one_centered : {0, 1}) {
                    auto input = random_vector(
                        static_cast<std::size_t>(rows) * columns, -4.0f, 4.0f);
                    auto weights = random_vector(columns, 0.25f, 1.75f);
                    std::vector<float> output(
                        static_cast<std::size_t>(rows) * columns, 0.0f);
                    const float* input_p = input.data();
                    const float* weight_p = weights.data();
                    float* output_p = output.data();
                    float epsilon = 1e-6f;
                    void* arguments[] = {&input_p, &weight_p, &output_p, &rows,
                                         &columns, &epsilon, &one_centered};
                    flyweight_cpu_launch_named("rms_norm_rows", rows, 1, 256, 0, 0,
                                             arguments);
                    outputs.push_back(std::move(output));
                }
            }
        },
        2e-6f,
    });

    cases().push_back(Case{
        "qwen_shared_scale",
        [](std::vector<std::vector<float>>& outputs) {
            for (int elements : {2048, 256, 33, 1}) {
                auto input = random_vector(elements, -1.0f, 1.0f);
                auto gate = random_vector(elements, -1.0f, 1.0f);
                auto shared = random_vector(elements, -2.0f, 2.0f);
                const float* input_p = input.data();
                const float* gate_p = gate.data();
                float* shared_p = shared.data();
                void* arguments[] = {&input_p, &gate_p, &shared_p, &elements};
                flyweight_cpu_launch_named("qwen_shared_scale", 1, 1, 256, 0, 0,
                                         arguments);
                outputs.push_back(std::move(shared));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "route_topk",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][2] = {{256, 8}, {128, 4}, {64, 1}, {32, 32}};
            for (const auto& shape : shapes) {
                int experts = shape[0];
                int top_k = shape[1];
                auto logits = random_vector(experts, -6.0f, 6.0f);
                std::vector<int> selected(top_k, -1);
                std::vector<float> weights(top_k, 0.0f);
                const float* logits_p = logits.data();
                int* selected_p = selected.data();
                float* weights_p = weights.data();
                void* arguments[] = {&logits_p, &selected_p, &weights_p,
                                     &experts, &top_k};
                flyweight_cpu_launch_named("route_topk", 1, 1, 256,
                                         experts * sizeof(float), 0, arguments);
                // Which experts were chosen is discrete and must match exactly;
                // it is compared alongside the weights, in the same buffer.
                std::vector<float> combined;
                for (int value : selected)
                    combined.push_back(static_cast<float>(value));
                for (float value : weights) combined.push_back(value);
                outputs.push_back(std::move(combined));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "route_topk_rows",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][3] = {{256, 8, 7}, {128, 4, 1}, {32, 32, 3}};
            for (const auto& shape : shapes) {
                int experts = shape[0];
                int top_k = shape[1];
                int rows = shape[2];
                auto logits = random_vector(
                    static_cast<std::size_t>(rows) * experts, -6.0f, 6.0f);
                std::vector<int> selected(
                    static_cast<std::size_t>(rows) * top_k, -1);
                std::vector<float> weights(
                    static_cast<std::size_t>(rows) * top_k, 0.0f);
                const float* logits_p = logits.data();
                int* selected_p = selected.data();
                float* weights_p = weights.data();
                void* arguments[] = {&logits_p, &selected_p, &weights_p, &rows,
                                     &experts, &top_k};
                flyweight_cpu_launch_named("route_topk_rows", rows, 1, 256,
                                         experts * sizeof(float), 0, arguments);
                std::vector<float> combined;
                for (int value : selected)
                    combined.push_back(static_cast<float>(value));
                for (float value : weights) combined.push_back(value);
                outputs.push_back(std::move(combined));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "qwen_shared_scale_rows",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][2] = {{2048, 5}, {256, 1}, {33, 9}};
            for (const auto& shape : shapes) {
                int elements = shape[0];
                int rows = shape[1];
                auto input = random_vector(
                    static_cast<std::size_t>(rows) * elements, -1.0f, 1.0f);
                auto gate = random_vector(elements, -1.0f, 1.0f);
                auto shared = random_vector(
                    static_cast<std::size_t>(rows) * elements, -2.0f, 2.0f);
                const float* input_p = input.data();
                const float* gate_p = gate.data();
                float* shared_p = shared.data();
                void* arguments[] = {&input_p, &gate_p, &shared_p, &rows,
                                     &elements};
                flyweight_cpu_launch_named("qwen_shared_scale_rows", rows, 1, 256,
                                         0, 0, arguments);
                outputs.push_back(std::move(shared));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "qwen_attention_query",
        [](std::vector<std::vector<float>>& outputs) {
            // rotary_dim < head_dim leaves a non-rotated tail; position 0 makes
            // every angle zero, which is the identity case worth pinning.
            const int shapes[][3] = {{16, 128, 128}, {2, 64, 32}, {4, 96, 64}};
            for (const auto& shape : shapes) {
                int heads = shape[0];
                int head_dim = shape[1];
                int rotary_dim = shape[2];
                for (int position : {0, 1, 517}) {
                    auto projected = random_vector(
                        static_cast<std::size_t>(heads) * 2 * head_dim, -2.0f, 2.0f);
                    auto norm_weights = random_vector(head_dim, 0.5f, 1.5f);
                    std::vector<float> queries(
                        static_cast<std::size_t>(heads) * head_dim, 0.0f);
                    std::vector<float> gates(
                        static_cast<std::size_t>(heads) * head_dim, 0.0f);
                    const float* projected_p = projected.data();
                    const float* norm_p = norm_weights.data();
                    float* queries_p = queries.data();
                    float* gates_p = gates.data();
                    float theta = 10000.0f;
                    float epsilon = 1e-6f;
                    void* arguments[] = {&projected_p, &norm_p, &queries_p,
                                         &gates_p, &heads, &head_dim,
                                         &rotary_dim, &position, &theta,
                                         &epsilon};
                    flyweight_cpu_launch_named("qwen_attention_query", heads, 1,
                                             256, 0, 0, arguments);
                    outputs.push_back(std::move(queries));
                    outputs.push_back(std::move(gates));
                }
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "kv_attention_values_f16_ring",
        [](std::vector<std::vector<float>>& outputs) {
            // first + tokens > capacity is the wrapped case; the corpus indexes
            // slots as (first + t) % capacity, so a run that straddles the end
            // of the ring has to be covered or the wrap is untested.
            const int shapes[][5] = {
                // heads, kv_heads, head_dim, tokens, first  (capacity = 256)
                {16, 2, 128, 37, 0},
                {16, 2, 128, 200, 0},
                {8, 8, 64, 90, 200},    // wraps
                {4, 2, 32, 256, 128},   // full ring, wraps
                {2, 1, 128, 1, 255},    // single token at the last slot
            };
            const int capacity = 256;
            for (const auto& shape : shapes) {
                int heads = shape[0], kv_heads = shape[1], head_dim = shape[2];
                int tokens = shape[3], first = shape[4];
                int cap = capacity;
                auto scores = random_vector(
                    static_cast<std::size_t>(heads) * tokens, -6.0f, 6.0f);
                auto raw = random_vector(
                    static_cast<std::size_t>(kv_heads) * capacity * head_dim,
                    -2.0f, 2.0f);
                // Round through fp16 so both paths read identical bits.
                std::vector<std::uint16_t> values(raw.size());
                for (std::size_t index = 0; index < raw.size(); ++index)
                    values[index] = flyweight::cpu::float_to_half_bits(raw[index]);
                std::vector<float> output(
                    static_cast<std::size_t>(heads) * head_dim, 0.0f);

                float* scores_p = scores.data();
                const std::uint16_t* values_p = values.data();
                float* output_p = output.data();
                void* arguments[] = {&scores_p, &values_p, &output_p, &heads,
                                     &kv_heads, &head_dim, &tokens, &cap, &first};
                flyweight_cpu_launch_named("kv_attention_values_f16_ring", heads,
                                         1, 256, 0, 0, arguments);
                outputs.push_back(std::move(output));
                // The scores buffer is softmaxed in place and read back by the
                // caller, so it is part of the contract.
                outputs.push_back(std::move(scores));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "kv_attention_values_ring",
        [](std::vector<std::vector<float>>& outputs) {
            // f32 cache variant of the same kernel; same wrap coverage.
            const int shapes[][5] = {
                {16, 2, 128, 37, 0}, {8, 8, 64, 90, 200}, {4, 2, 32, 256, 128},
            };
            const int capacity = 256;
            for (const auto& shape : shapes) {
                int heads = shape[0], kv_heads = shape[1], head_dim = shape[2];
                int tokens = shape[3], first = shape[4];
                int cap = capacity;
                auto scores = random_vector(
                    static_cast<std::size_t>(heads) * tokens, -6.0f, 6.0f);
                auto values = random_vector(
                    static_cast<std::size_t>(kv_heads) * capacity * head_dim,
                    -2.0f, 2.0f);
                std::vector<float> output(
                    static_cast<std::size_t>(heads) * head_dim, 0.0f);

                float* scores_p = scores.data();
                const float* values_p = values.data();
                float* output_p = output.data();
                void* arguments[] = {&scores_p, &values_p, &output_p, &heads,
                                     &kv_heads, &head_dim, &tokens, &cap, &first};
                flyweight_cpu_launch_named("kv_attention_values_ring", heads, 1,
                                         256, 0, 0, arguments);
                outputs.push_back(std::move(output));
                outputs.push_back(std::move(scores));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "qwen_attention_key",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][3] = {{2, 128, 128}, {8, 64, 32}, {1, 96, 64}};
            for (const auto& shape : shapes) {
                int heads = shape[0];
                int head_dim = shape[1];
                int rotary_dim = shape[2];
                for (int position : {0, 1, 517}) {
                    auto projected = random_vector(
                        static_cast<std::size_t>(heads) * head_dim, -2.0f, 2.0f);
                    auto norm_weights = random_vector(head_dim, 0.5f, 1.5f);
                    std::vector<float> keys(
                        static_cast<std::size_t>(heads) * head_dim, 0.0f);
                    const float* projected_p = projected.data();
                    const float* norm_p = norm_weights.data();
                    float* keys_p = keys.data();
                    float theta = 10000.0f;
                    float epsilon = 1e-6f;
                    void* arguments[] = {&projected_p, &norm_p, &keys_p, &heads,
                                         &head_dim, &rotary_dim, &position,
                                         &theta, &epsilon};
                    flyweight_cpu_launch_named("qwen_attention_key", heads, 1, 256,
                                             0, 0, arguments);
                    outputs.push_back(std::move(keys));
                }
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "qwen_imatrix_accumulate",
        [](std::vector<std::vector<float>>& outputs) {
            // Widths on and off the block boundary, and sums seeded non-zero
            // because the kernel accumulates rather than overwrites.
            for (int width : {256, 320, 1024}) {
                for (int rows : {1, 7}) {
                    auto input = random_vector(
                        static_cast<std::size_t>(rows) * width, -2.0f, 2.0f);
                    auto sums = random_vector(width, 0.0f, 1.0f);

                    const float* input_pointer = input.data();
                    float* sums_pointer = sums.data();
                    void* arguments[] = {&input_pointer, &sums_pointer, &width,
                                         &rows};
                    flyweight_cpu_launch_named("qwen_imatrix_accumulate",
                                             (width + 255) / 256, 1, 256, 0, 0,
                                             arguments);
                    outputs.push_back(std::move(sums));
                }
            }
        },
        1e-5f,
    });

    // K2-Horizon's norm is grouped: the hidden vector is cut into `groups`
    // contiguous spans and each is normalized against its own slice of the
    // weight row, rather than one reduction across the whole width. The group
    // count comes from the launch grid, so a case has to vary it to reach the
    // indexing at all -- a single-group launch degenerates to plain rms_norm
    // and would pass while the group stride was wrong.
    cases().push_back(Case{
        "qwen4_group_rms",
        [](std::vector<std::vector<float>>& outputs) {
            // group_size on and off the 256-thread block boundary, so the
            // emulated reduction tree is exercised with and without a tail.
            const int shapes[][2] = {{4, 256}, {4, 640}, {2, 1024}, {8, 33}};
            for (const auto& shape : shapes) {
                int groups = shape[0];
                int group_size = shape[1];
                const auto width =
                    static_cast<std::size_t>(groups) * group_size;
                auto input = random_vector(width, -4.0f, 4.0f);
                auto weights = random_vector(width, 0.25f, 1.75f);
                std::vector<float> output(width, 0.0f);

                const float* input_p = input.data();
                const float* weight_p = weights.data();
                float* output_p = output.data();
                float epsilon = 1e-6f;
                void* arguments[] = {&input_p, &weight_p, &output_p,
                                     &group_size, &epsilon};
                flyweight_cpu_launch_named("qwen4_group_rms", groups, 1, 256, 0,
                                         0, arguments);
                outputs.push_back(std::move(output));
            }
        },
        // Same split as rms_norm: the emulated path sums through a shuffle
        // tree, the native path in double.
        2e-6f,
    });

    // The row-batched twin. `groups` moves from the grid into an argument and
    // the row index becomes grid_y, so the address arithmetic is genuinely a
    // different expression and needs its own case -- rows > 1 is the point.
    cases().push_back(Case{
        "qwen4_group_rms_rows",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][3] = {
                {4, 256, 1}, {4, 640, 5}, {2, 1024, 3}, {8, 33, 9}};
            for (const auto& shape : shapes) {
                int groups = shape[0];
                int group_size = shape[1];
                int rows = shape[2];
                const auto width =
                    static_cast<std::size_t>(groups) * group_size;
                auto input =
                    random_vector(width * static_cast<std::size_t>(rows),
                                  -4.0f, 4.0f);
                // One weight row shared by every token, indexed by group only.
                auto weights = random_vector(width, 0.25f, 1.75f);
                std::vector<float> output(
                    width * static_cast<std::size_t>(rows), 0.0f);

                const float* input_p = input.data();
                const float* weight_p = weights.data();
                float* output_p = output.data();
                float epsilon = 1e-6f;
                void* arguments[] = {&input_p, &weight_p, &output_p, &groups,
                                     &group_size, &epsilon};
                flyweight_cpu_launch_named("qwen4_group_rms_rows", groups, rows,
                                         256, 0, 0, arguments);
                outputs.push_back(std::move(output));
            }
        },
        2e-6f,
    });

    // K2-Horizon's Q/K tail: half-split (NEOX) RoPE with no per-head norm and
    // no learned weights, so the pair partner sits rotary_dim/2 away rather
    // than adjacent. head_dim > rotary_dim covers the pass-through tail, which
    // is the half of the kernel a rotation-only case never touches.
    cases().push_back(Case{
        "k2_half_split_rope",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][3] = {
                {2, 128, 128}, {8, 64, 32}, {1, 96, 64}, {4, 128, 64}};
            for (const auto& shape : shapes) {
                int heads = shape[0];
                int head_dim = shape[1];
                int rotary_dim = shape[2];
                for (int position : {0, 1, 517}) {
                    auto projected = random_vector(
                        static_cast<std::size_t>(heads) * head_dim, -2.0f, 2.0f);
                    std::vector<float> rotated(
                        static_cast<std::size_t>(heads) * head_dim, 0.0f);

                    const float* projected_p = projected.data();
                    float* rotated_p = rotated.data();
                    float theta = 10000000.0f;
                    void* arguments[] = {&projected_p, &rotated_p, &heads,
                                         &head_dim, &rotary_dim, &position,
                                         &theta};
                    flyweight_cpu_launch_named("k2_half_split_rope", heads, 1,
                                             256, 0, 0, arguments);
                    outputs.push_back(std::move(rotated));
                }
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "qwen_attention_gate",
        [](std::vector<std::vector<float>>& outputs) {
            // Widths on and off the 256-thread block boundary, with gates
            // past the +-80 clamp the corpus applies.
            for (int elements : {256, 1024, 1537, 4096, 1}) {
                auto attended = random_vector(elements, -4.0f, 4.0f);
                auto gates = random_vector(elements, -100.0f, 100.0f);
                std::vector<float> output(elements, 0.0f);
                const float* attended_p = attended.data();
                const float* gates_p = gates.data();
                float* output_p = output.data();
                void* arguments[] = {&attended_p, &gates_p, &output_p,
                                     &elements};
                flyweight_cpu_launch_named(
                    "qwen_attention_gate",
                    static_cast<std::uint32_t>((elements + 255) / 256), 1,
                    256, 0, 0, arguments);
                outputs.push_back(std::move(output));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "route_topk_sigmoid_bias",
        [](std::vector<std::vector<float>>& outputs) {
            // qwen4exp's 512x10 shape first, then small and edge shapes;
            // with and without bias, normalized and raw, scaled and not.
            // Selection is discrete and compared exactly like route_topk.
            const int shapes[][2] = {{512, 10}, {256, 8}, {64, 1}, {32, 32}};
            for (const auto& shape : shapes) {
                for (int with_bias = 0; with_bias < 2; ++with_bias) {
                    for (int normalize = 0; normalize < 2; ++normalize) {
                        int experts = shape[0];
                        int top_k = shape[1];
                        auto logits = random_vector(experts, -6.0f, 6.0f);
                        auto bias = random_vector(experts, -1.0f, 1.0f);
                        std::vector<int> selected(top_k, -1);
                        std::vector<float> weights(top_k, 0.0f);
                        const float* logits_p = logits.data();
                        const float* bias_p =
                            with_bias ? bias.data() : nullptr;
                        int* selected_p = selected.data();
                        float* weights_p = weights.data();
                        float weight_scale = 2.5f;
                        void* arguments[] = {
                            &logits_p, &bias_p, &selected_p, &weights_p,
                            &experts, &top_k, &normalize, &weight_scale};
                        flyweight_cpu_launch_named(
                            "route_topk_sigmoid_bias", 1, 1, 256,
                            static_cast<std::uint32_t>(2 * experts *
                                                       sizeof(float)),
                            0, arguments);
                        std::vector<float> combined;
                        for (int value : selected)
                            combined.push_back(static_cast<float>(value));
                        for (float value : weights) combined.push_back(value);
                        outputs.push_back(std::move(combined));
                    }
                }
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "route_topk_sigmoid_bias_rows",
        [](std::vector<std::vector<float>>& outputs) {
            const int shapes[][3] = {{512, 10, 5}, {64, 4, 1}, {32, 32, 3}};
            for (const auto& shape : shapes) {
                int experts = shape[0];
                int top_k = shape[1];
                int rows = shape[2];
                auto logits = random_vector(
                    static_cast<std::size_t>(rows) * experts, -6.0f, 6.0f);
                auto bias = random_vector(experts, -1.0f, 1.0f);
                std::vector<int> selected(
                    static_cast<std::size_t>(rows) * top_k, -1);
                std::vector<float> weights(
                    static_cast<std::size_t>(rows) * top_k, 0.0f);
                const float* logits_p = logits.data();
                const float* bias_p = bias.data();
                int* selected_p = selected.data();
                float* weights_p = weights.data();
                int normalize = 1;
                float weight_scale = 1.0f;
                void* arguments[] = {
                    &logits_p, &bias_p, &selected_p, &weights_p, &rows,
                    &experts, &top_k, &normalize, &weight_scale};
                flyweight_cpu_launch_named(
                    "route_topk_sigmoid_bias_rows", rows, 1, 256,
                    static_cast<std::uint32_t>(2 * experts * sizeof(float)),
                    0, arguments);
                std::vector<float> combined;
                for (int value : selected)
                    combined.push_back(static_cast<float>(value));
                for (float value : weights) combined.push_back(value);
                outputs.push_back(std::move(combined));
            }
        },
        1e-5f,
    });

    cases().push_back(Case{
        "qwen_delta_recurrent_rows",
        [](std::vector<std::vector<float>>& outputs) {
            // Non-128 head dims: this kernel is the rows path the head_dim
            // guard selects instead of the chunk twin. Both gate modes.
            for (int head_dim : {32, 64}) {
                for (int rows : {1, 9}) {
                    for (int gate_sigmoid : {0, 1}) {
                        int key_heads = 2, value_heads = 4;
                        const int total_key = key_heads * head_dim;
                        const std::size_t width =
                            static_cast<std::size_t>(2 * total_key +
                                                     value_heads * head_dim);
                        auto convolved = random_vector(
                            static_cast<std::size_t>(rows) * width, -1.0f,
                            1.0f);
                        auto gates = random_vector(
                            static_cast<std::size_t>(rows) * value_heads *
                                head_dim,
                            -4.0f, 4.0f);
                        auto beta_logits = random_vector(
                            static_cast<std::size_t>(rows) * value_heads,
                            -2.0f, 2.0f);
                        auto decay_logits = random_vector(
                            static_cast<std::size_t>(rows) * value_heads,
                            -2.0f, 2.0f);
                        auto decay_coefficients = random_vector(value_heads,
                                                                -3.0f, -0.1f);
                        auto dt_bias =
                            random_vector(value_heads, -1.0f, 1.0f);
                        auto norm_weights =
                            random_vector(head_dim, 0.5f, 1.5f);
                        std::vector<float> state(
                            static_cast<std::size_t>(value_heads) * head_dim *
                                head_dim,
                            0.01f);
                        std::vector<float> out(
                            static_cast<std::size_t>(rows) * value_heads *
                            head_dim, 0.0f);
                        const float* convolved_p = convolved.data();
                        const float* gates_p = gates.data();
                        const float* beta_p = beta_logits.data();
                        const float* decay_p = decay_logits.data();
                        const float* coeff_p = decay_coefficients.data();
                        const float* dt_p = dt_bias.data();
                        const float* norm_p = norm_weights.data();
                        float* state_p = state.data();
                        float* out_p = out.data();
                        float epsilon = 1e-6f;
                        void* arguments[] = {
                            &convolved_p, &gates_p, &beta_p, &decay_p,
                            &coeff_p, &dt_p, &norm_p, &state_p, &out_p,
                            &rows, &key_heads, &value_heads, &head_dim,
                            &epsilon, &gate_sigmoid};
                        flyweight_cpu_launch_named(
                            "qwen_delta_recurrent_rows", value_heads, 1, 256,
                            0, 0, arguments);
                        outputs.push_back(std::move(out));
                        outputs.push_back(std::move(state));
                    }
                }
            }
        },
        1e-4f,
    });
}

// --- Q8-activation matvec families (cpu_q8_family.inc) ----------------------
//
// One case per registered kernel name, generated from the same format table
// the registrations come from. The packed weights are random bytes drawn from
// [0x00, 0x3b] and [0x80, 0xbb]: every byte pair then reads as a finite fp16
// below 1.0, so the per-format d / dmin fields stay finite whatever the layout
// puts where, while the quant nibbles, sign bits and codebook indices still
// cover a wide range.
struct Q8Format {
    const char* prefix;
    int stride;          // bytes per 256 values
    bool lm_head;
    bool minimum;        // FLYWEIGHT_Q8_MATVEC_ROWS_MIN rather than _ROWS
    bool tiled;
    bool flat;           // 32-wide blocks: input_size need not divide 256
};

const Q8Format kQ8Formats[] = {
    {"iq1m", 56, false, false, true, false},
    {"iq1s", 50, true, false, true, false},
    {"iq2s", 82, false, false, true, false},
    {"iq2xs", 74, false, false, true, false},
    {"iq2xxs", 66, true, false, true, false},
    {"iq3s", 110, true, false, true, false},
    {"iq3xxs", 98, true, false, true, false},
    {"iq4nl", 144, true, false, true, true},
    {"iq4xs", 136, false, false, true, false},
    {"q2k", 84, true, true, false, false},
    {"q3k", 110, true, false, true, false},
    {"q4k", 144, true, true, false, false},
    {"q5k", 176, true, true, false, false},
    {"q6k", 210, true, false, true, false},
    {"q80", 272, true, false, true, true},
};

std::vector<unsigned char> random_q8_weights(int input_size, int output_size,
                                             int stride) {
    const int groups = input_size / 32;
    const std::size_t row_bytes =
        static_cast<std::size_t>(groups / 8) * stride
        + static_cast<std::size_t>(groups % 8) * (stride / 8);
    std::uniform_int_distribution<int> byte(0, 2 * 0x3c - 1);
    std::vector<unsigned char> packed(row_bytes * output_size);
    for (auto& value : packed) {
        const int pick = byte(rng());
        value = static_cast<unsigned char>(pick < 0x3c ? pick : 0x80 + pick - 0x3c);
    }
    return packed;
}

struct Q8Activations {
    std::vector<signed char> values;
    std::vector<std::uint16_t> scales;
    int scale_stride;
};

Q8Activations random_q8_activations(int input_size, int rows) {
    Q8Activations out;
    const int groups = input_size / 32;
    out.scale_stride = groups + 2;  // the runtime pads scale rows the same way
    std::uniform_int_distribution<int> value(-127, 127);
    out.values.resize(static_cast<std::size_t>(input_size) * rows);
    for (auto& v : out.values) v = static_cast<signed char>(value(rng()));
    auto scales = random_vector(static_cast<std::size_t>(out.scale_stride) * rows,
                                0.002f, 0.02f);
    out.scales.resize(scales.size());
    for (std::size_t index = 0; index < scales.size(); ++index)
        out.scales[index] = flyweight::cpu::float_to_half_bits(scales[index]);
    return out;
}

// These kernels compare against an emulated reference whose own fp32 error is
// the larger term (the single-row ones also group the same sum differently:
// the corpus's fused *_q8_group against the decode form the host runs): random signed weights against random signed
// activations make every output a heavily cancelling sum, and the corpus
// reduces it in a different order (per-thread strides then a shuffle tree, or
// tile-sequential lanes for the tiled kernel). Replacing the native
// accumulation with an exact double sum leaves the gap where it was -- 1e-5 to
// 1e-4 relative across the formats -- so that gap is the reference's noise. A
// real defect in decode or addressing shows up as O(1) relative error.
constexpr float kQ8BatchTolerance = 2.5e-4f;

std::vector<int> q8_input_sizes(const Q8Format& format) {
    if (format.flat) return {256, 288, 768};
    return {256, 768};
}

void add_q8_family_cases() {
    for (const Q8Format& format : kQ8Formats) {
        const std::string prefix = format.prefix;

        cases().push_back(Case{
            prefix + "_q8_matvec_transposed_warp",
            [format](std::vector<std::vector<float>>& outputs) {
                for (int input_size : q8_input_sizes(format)) {
                    int output_size = 19;
                    auto packed = random_q8_weights(input_size, output_size, format.stride);
                    auto act = random_q8_activations(input_size, 1);
                    std::vector<float> output(output_size, 0.0f);
                    const unsigned char* packed_p = packed.data();
                    const signed char* vector_p = act.values.data();
                    const std::uint16_t* scales_p = act.scales.data();
                    float* output_p = output.data();
                    void* arguments[] = {&packed_p, &vector_p, &scales_p, &output_p,
                                         &input_size, &output_size};
                    flyweight_cpu_launch_named(
                        (std::string(format.prefix) + "_q8_matvec_transposed_warp").c_str(),
                        output_size, 1, 128, 0, 0, arguments);
                    outputs.push_back(std::move(output));
                }
            },
            kQ8BatchTolerance,
        });

        if (format.lm_head) {
            cases().push_back(Case{
                prefix + "_q8_lm_head_argmax_warp",
                [format](std::vector<std::vector<float>>& outputs) {
                    for (int input_size : q8_input_sizes(format)) {
                        int output_size = 61;
                        auto packed = random_q8_weights(input_size, output_size, format.stride);
                        auto act = random_q8_activations(input_size, 1);
                        unsigned long long winner = 0ull;
                        const unsigned char* packed_p = packed.data();
                        const signed char* vector_p = act.values.data();
                        const std::uint16_t* scales_p = act.scales.data();
                        unsigned long long* winner_p = &winner;
                        void* arguments[] = {&packed_p, &vector_p, &scales_p, &winner_p,
                                             &input_size, &output_size};
                        flyweight_cpu_launch_named(
                            (std::string(format.prefix) + "_q8_lm_head_argmax_warp").c_str(),
                            (output_size + 7) / 8, 1, 256, 0, 0, arguments);
                        // The row must match exactly; the logit to tolerance.
                        const unsigned int row = 0xffffffffu - static_cast<unsigned int>(winner);
                        unsigned int ordered = static_cast<unsigned int>(winner >> 32);
                        const unsigned int bits = (ordered & 0x80000000u)
                            ? (ordered ^ 0x80000000u) : ~ordered;
                        float logit;
                        std::memcpy(&logit, &bits, sizeof(logit));
                        outputs.push_back({static_cast<float>(row) * 1000.0f, logit});
                    }
                },
                kQ8BatchTolerance,
            });
        }

        // The batched rows family: the full-cap name plus its _r2.._r7 twins.
        const int caps[] = {8, 2, 3, 4, 5, 6, 7};
        for (int cap : caps) {
            const std::string name = prefix + "_q8_matvec_transposed_rows"
                + (cap == 8 ? std::string() : "_r" + std::to_string(cap));
            cases().push_back(Case{
                name,
                [format, cap, name](std::vector<std::vector<float>>& outputs) {
                    const std::vector<int> sizes = format.minimum
                        ? std::vector<int>{256, 768} : q8_input_sizes(format);
                    for (int input_size : sizes) {
                        // A full batch and a short one: rows < cap leaves
                        // accumulators the kernel must not write.
                        for (int rows : {cap, cap > 2 ? cap - 2 : 1}) {
                            int output_size = 13;
                            auto packed = random_q8_weights(input_size, output_size, format.stride);
                            auto act = random_q8_activations(input_size, rows);
                            std::vector<float> output(
                                static_cast<std::size_t>(output_size) * cap, -7.0f);
                            const unsigned char* packed_p = packed.data();
                            const signed char* vectors_p = act.values.data();
                            const std::uint16_t* scales_p = act.scales.data();
                            float* output_p = output.data();
                            int scale_stride = act.scale_stride;
                            void* arguments[] = {&packed_p, &vectors_p, &scales_p, &output_p,
                                                 &input_size, &output_size, &rows,
                                                 &scale_stride};
                            flyweight_cpu_launch_named(name.c_str(), output_size, 1, 128,
                                                       0, 0, arguments);
                            outputs.push_back(std::move(output));
                        }
                    }
                },
                kQ8BatchTolerance,
            });
        }

        if (format.tiled) {
            cases().push_back(Case{
                prefix + "_q8_matmul_tiled",
                [format](std::vector<std::vector<float>>& outputs) {
                    for (int input_size : q8_input_sizes(format)) {
                        // Rows off the 16-row tile and tokens off the 8-warp
                        // split, plus a full 32-token batch.
                        for (int rows : {32, 5}) {
                            int output_size = 21;
                            auto packed = random_q8_weights(input_size, output_size, format.stride);
                            auto act = random_q8_activations(input_size, rows);
                            std::vector<float> output(
                                static_cast<std::size_t>(output_size) * rows, -7.0f);
                            const unsigned char* packed_p = packed.data();
                            const signed char* vectors_p = act.values.data();
                            const std::uint16_t* scales_p = act.scales.data();
                            float* output_p = output.data();
                            int scale_stride = act.scale_stride;
                            void* arguments[] = {&packed_p, &vectors_p, &scales_p, &output_p,
                                                 &input_size, &output_size, &rows,
                                                 &scale_stride};
                            flyweight_cpu_launch_named(
                                (std::string(format.prefix) + "_q8_matmul_tiled").c_str(),
                                (output_size + 15) / 16, 1, 256, 0, 0, arguments);
                            outputs.push_back(std::move(output));
                        }
                    }
                },
                kQ8BatchTolerance,
            });
        }
    }

    // Quantizers: bytes and scales must match exactly, so tolerance is zero.
    cases().push_back(Case{
        "quantize_q8_blocks",
        [](std::vector<std::vector<float>>& outputs) {
            for (int elements : {32, 1000, 2560}) {
                auto input = random_vector(elements, -3.0f, 3.0f);
                // An all-zero block takes the scale = 1 branch.
                for (int index = 0; index < std::min(elements, 32); ++index)
                    input[index] = 0.0f;
                const int blocks = (elements + 31) / 32;
                std::vector<signed char> output(elements, 0);
                std::vector<std::uint16_t> scales(blocks, 0);
                const float* input_p = input.data();
                signed char* output_p = output.data();
                std::uint16_t* scales_p = scales.data();
                void* arguments[] = {&input_p, &output_p, &scales_p, &elements};
                flyweight_cpu_launch_named("quantize_q8_blocks", blocks, 1, 32, 0, 0,
                                           arguments);
                outputs.emplace_back(output.begin(), output.end());
                std::vector<float> scale_values(blocks);
                for (int b = 0; b < blocks; ++b)
                    scale_values[b] = flyweight::cpu::half_bits_to_float(scales[b]);
                outputs.push_back(std::move(scale_values));
            }
        },
        0.0f,
    });

    cases().push_back(Case{
        "quantize_q8_blocks_rows",
        [](std::vector<std::vector<float>>& outputs) {
            for (int elements : {64, 1000, 2560}) {
                int rows = 3;
                int scale_stride = (elements + 31) / 32 + 2;
                auto input = random_vector(static_cast<std::size_t>(elements) * rows,
                                           -3.0f, 3.0f);
                std::vector<signed char> output(static_cast<std::size_t>(elements) * rows, 0);
                std::vector<std::uint16_t> scales(
                    static_cast<std::size_t>(scale_stride) * rows, 0);
                const float* input_p = input.data();
                signed char* output_p = output.data();
                std::uint16_t* scales_p = scales.data();
                void* arguments[] = {&input_p, &output_p, &scales_p, &elements,
                                     &scale_stride};
                const int warps = 256 / 32;
                const int blocks32 = (elements + 31) / 32;
                flyweight_cpu_launch_named("quantize_q8_blocks_rows",
                                           (blocks32 + warps - 1) / warps, rows, 256,
                                           0, 0, arguments);
                outputs.emplace_back(output.begin(), output.end());
                std::vector<float> scale_values(scales.size());
                for (std::size_t b = 0; b < scales.size(); ++b)
                    scale_values[b] = flyweight::cpu::half_bits_to_float(scales[b]);
                outputs.push_back(std::move(scale_values));
            }
        },
        0.0f,
    });
}

// --- M-RoPE query / key ---------------------------------------------------
void add_mrope_cases() {
    struct Shape { int heads, head_dim, rotary_dim, t, h, w, e; };
    // Qwen 3.5's real split is [11, 11, 10] over 64 pairs; the small shape
    // has an extra section and a non-rotated tail.
    const Shape shapes[] = {{4, 256, 64, 11, 11, 10, 0}, {3, 64, 32, 6, 5, 4, 1}};

    for (bool key : {false, true}) {
        for (bool batched : {false, true}) {
            const std::string name = std::string(key ? "qwen_attention_key_mrope"
                                                     : "qwen_attention_query_mrope")
                + (batched ? "_rows" : "");
            cases().push_back(Case{
                name,
                [=](std::vector<std::vector<float>>& outputs) {
                    for (const Shape& shape : shapes) {
                        int heads = shape.heads, head_dim = shape.head_dim;
                        int rotary_dim = shape.rotary_dim;
                        int position = 37, position_h = 5, position_w = 9;
                        int st = shape.t, sh = shape.h, sw = shape.w, se = shape.e;
                        float theta = 10000000.0f, epsilon = 1e-6f;
                        const int rows = batched ? 3 : 1;
                        const int per_head = key ? head_dim : 2 * head_dim;
                        int projected_stride = heads * per_head + 8;
                        int out_stride = heads * head_dim + 4;
                        auto projected = random_vector(
                            static_cast<std::size_t>(projected_stride) * rows, -2.0f, 2.0f);
                        auto norm = random_vector(head_dim, 0.5f, 1.5f);
                        std::vector<float> first(static_cast<std::size_t>(out_stride) * rows, 0.0f);
                        std::vector<float> second(static_cast<std::size_t>(out_stride) * rows, 0.0f);
                        const float* projected_p = projected.data();
                        const float* norm_p = norm.data();
                        float* first_p = first.data();
                        float* second_p = second.data();
                        std::vector<void*> arguments = {&projected_p, &norm_p, &first_p};
                        if (!key) arguments.push_back(&second_p);
                        for (void* a : std::initializer_list<void*>{
                                 &heads, &head_dim, &rotary_dim, &position, &theta,
                                 &epsilon, &position_h, &position_w, &st, &sh, &sw, &se})
                            arguments.push_back(a);
                        if (batched) {
                            arguments.push_back(&projected_stride);
                            arguments.push_back(&out_stride);
                            if (!key) arguments.push_back(&out_stride);
                        }
                        flyweight_cpu_launch_named(name.c_str(), heads, rows, 256, 0, 0,
                                                   arguments.data());
                        outputs.push_back(std::move(first));
                        if (!key) outputs.push_back(std::move(second));
                    }
                },
                1e-5f,
            });
        }
    }
}

}  // namespace

int main() {
    flyweight_backend_select(kFlyweightBackendCpu);
    add_cases();
    add_q8_family_cases();
    add_mrope_cases();

    std::printf("CPU kernel parity (native vs emulated corpus)\n");
    int failures = 0;
    for (const auto& item : cases()) {
        if (flyweight::cpu::find_native_kernel(item.kernel.c_str()) == nullptr) {
            std::printf("  %-28s FAIL (no native kernel registered)\n",
                        item.kernel.c_str());
            ++failures;
            continue;
        }
        failures += run_case(item);
    }

    // A native kernel with no case would be serving production traffic with
    // nothing checking it against the reference.
    const std::size_t total = flyweight::cpu::kernel_count();
    for (std::size_t index = 0; index < total; ++index) {
        const char* name = flyweight::cpu::kernel_name(index);
        if (name == nullptr) continue;
        if (flyweight::cpu::find_native_kernel(name) == nullptr) continue;
        bool covered = false;
        for (const auto& item : cases())
            if (item.kernel == name) { covered = true; break; }
        if (!covered) {
            std::printf("  %-28s FAIL (native kernel has no parity case)\n", name);
            ++failures;
        }
    }

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
