// IQ device-kernel contract: each CUDA decoder must agree with the CPU one.
//
// iq1m_value is a second implementation of a format that already had one --
// qwen_iq1m_value in qwen_kquant.h, which qwen_kquant_contract pins to real
// super-blocks taken out of a checkpoint. Two decoders for a format this
// fiddly (an f16 scale scattered four bits at a time across four halfwords, an
// 11-bit grid index split between two arrays, a sign bit whose position
// depends on the group's parity) is exactly the kind of pair that drifts.
//
// The corpus kernel compiled here is the same CUDA text the GPU compiles, so
// agreeing with it is the strongest statement available without a GPU present.
// The reference side is the CPU dot product rather than fresh expectations,
// which keeps this test one hop from the real-checkpoint fixture.

#include <flyweight_backend.hpp>
#include <flyweight_cpu_kernels_api.hpp>
#include <flyweight_cpu_shim_geometry.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include "flyweight_v2_moe_align.hpp"

#include <cstdio>
#include <random>
#include <vector>

#include "qwen_kquant.h"

extern "C" {
int flyweight_cpu_launch_named(const char*, std::uint32_t, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint64_t,
                             void**);
}

namespace {

// Random blocks, except for the super-block scale.
//
// Every 56-byte pattern is a *legal* IQ1_M block -- the grid index is 11 bits
// into a 2048-entry table and the sub-scales are unconstrained -- so random
// bytes cover the codebook far better than a fixture would. But the super-block
// scale is an f16 assembled from sixteen scattered bits, and random bits there
// mean a random exponent: blocks land at 1e4 or 1e-5, and a row that sums such
// blocks loses most of its significant digits to cancellation. That is a
// property of the test input, not of either decoder, and it shows up as the two
// summation orders disagreeing in the fourth digit.
//
// So the scale's four nibbles are pinned to 1.0 (f16 0x3c00) and everything
// else stays random. The 3-bit sub-scales still cover 1..15, the grid is still
// fully exercised, and a row is now a well-conditioned sum.
void fill_iq1m(std::mt19937& rng, std::vector<std::uint8_t>& packed) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : packed) value = static_cast<std::uint8_t>(byte(rng));

    constexpr std::uint16_t kOne = 0x3c00;
    for (std::size_t base = 0; base + kIq1mBlockBytes <= packed.size();
         base += kIq1mBlockBytes) {
        for (int half = 0; half < 4; ++half) {
            // qwen_iq1m_scale reads nibble `half` of the f16 out of the top
            // nibble of scale halfword `half`; the low twelve bits are the
            // sub-scales and are left alone.
            std::uint16_t word;
            std::memcpy(&word, packed.data() + base + 48 + half * 2, 2);
            const std::uint16_t nibble =
                static_cast<std::uint16_t>((kOne >> (4 * half)) & 0xf);
            word = static_cast<std::uint16_t>((word & 0x0fff) | (nibble << 12));
            std::memcpy(packed.data() + base + 48 + half * 2, &word, 2);
        }
    }
}

// Every other format here keeps its f16 super-block scale in the first two
// bytes, so pinning it is the whole of what they need; the rest of a block is
// legal whatever it holds. IQ1_S is one of them -- an 11-bit grid index into a
// 2048-entry table, a 3-bit multiplier and a sign bit are all unconstrained.
template <std::size_t kBlockBytes>
void fill_leading_scale(std::mt19937& rng, std::vector<std::uint8_t>& packed) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : packed) value = static_cast<std::uint8_t>(byte(rng));
    constexpr std::uint16_t kOne = 0x3c00;
    for (std::size_t base = 0; base + kBlockBytes <= packed.size();
         base += kBlockBytes)
        std::memcpy(packed.data() + base, &kOne, 2);
}

// The K-quants keep their f16 scales at a per-format offset rather than the
// front of the block, and two of them carry a second one for the group minimum.
// Random bits there are infinities and NaNs, which decode at a different speed
// and compare to nothing, so both are pinned to 1.0 and everything else stays
// random.
template <std::size_t kBlockBytes, std::size_t kFirst, std::size_t kSecond>
void fill_scales_at(std::mt19937& rng, std::vector<std::uint8_t>& packed) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : packed) value = static_cast<std::uint8_t>(byte(rng));
    constexpr std::uint16_t kOne = 0x3c00;
    for (std::size_t base = 0; base + kBlockBytes <= packed.size();
         base += kBlockBytes) {
        std::memcpy(packed.data() + base + kFirst, &kOne, 2);
        if (kSecond != kFirst)
            std::memcpy(packed.data() + base + kSecond, &kOne, 2);
    }
}

// A format under test: how wide its super-block is, how to fill one, and the
// CPU decoder the corpus kernel has to agree with.
struct Format {
    std::size_t block_bytes;
    void (*fill)(std::mt19937&, std::vector<std::uint8_t>&);
    float (*value_at)(const std::uint8_t*, std::uint64_t);
    // Elements per block. Every K-quant and IQ super-block here is 256 wide;
    // IQ4_NL is a flat 32 with no super-block, so the packed size cannot be
    // derived from a shared constant.
    std::uint32_t block_elements = 256;
};

const Format kIq1m{kIq1mBlockBytes, fill_iq1m, qwen_iq1m_value};
const Format kIq4nl{
    kIq4nlBlockBytes, fill_leading_scale<kIq4nlBlockBytes>, qwen_iq4nl_value,
    kIq4nlBlockElements};
const Format kIq1s{
    kIq1sBlockBytes, fill_leading_scale<kIq1sBlockBytes>, qwen_iq1s_value};
const Format kIq2s{
    kIq2sBlockBytes, fill_leading_scale<kIq2sBlockBytes>, qwen_iq2s_value};
const Format kIq2xxs{
    kIq2xxsBlockBytes, fill_leading_scale<kIq2xxsBlockBytes>, qwen_iq2xxs_value};
const Format kIq2xs{
    kIq2xsBlockBytes, fill_leading_scale<kIq2xsBlockBytes>, qwen_iq2xs_value};
const Format kIq3xxs{
    kIq3xxsBlockBytes, fill_leading_scale<kIq3xxsBlockBytes>, qwen_iq3xxs_value};
const Format kIq3s{
    kIq3sBlockBytes, fill_leading_scale<kIq3sBlockBytes>, qwen_iq3s_value};
const Format kIq4xs{
    kIq4xsBlockBytes, fill_leading_scale<kIq4xsBlockBytes>, qwen_iq4xs_value};

// The K-quants, for the routed MMQ. Q4_K and Q5_K are what most MoE
// checkpoints in the wild actually ship their experts as, which is the whole
// reason the _MIN macro needed a routed form too.
const Format kQ4k{144, fill_scales_at<144, 0, 2>, qwen_q4k_value};
const Format kQ5k{176, fill_scales_at<176, 0, 2>, qwen_q5_value};
const Format kQ6k{210, fill_scales_at<210, 208, 208>, qwen_q6_value};
const Format kQ2k{84, fill_scales_at<84, 80, 82>, qwen_q2k_value};
const Format kQ3k{110, fill_scales_at<110, 108, 108>, qwen_q3k_value};

// The kernel accumulates a row in f32 in a tree; the reference here does it in
// double, elementwise, through the other decoder. The error that separates them
// is bounded by the f32 epsilon times the magnitude *accumulated*, not the
// magnitude *returned* -- a row whose products cancel down to near zero is
// perfectly correct and still has a huge relative error against its own result.
// So the denominator is sum|w_i v_i|, which is what conditions the sum.
float worst_error(const Format& format, const std::vector<std::uint8_t>& packed,
                  const float* vector, int input_size, int row, float got) {
    const std::size_t blocks =
        static_cast<std::size_t>(input_size) / format.block_elements;
    const std::uint8_t* base =
        packed.data() + static_cast<std::size_t>(row) * blocks * format.block_bytes;
    double dot = 0.0;
    double magnitude = 0.0;
    for (int index = 0; index < input_size; ++index) {
        const double term =
            static_cast<double>(format.value_at(base, index)) * vector[index];
        dot += term;
        magnitude += std::fabs(term);
    }
    return static_cast<float>(std::fabs(dot - got) /
                              std::fmax(1.0, magnitude));
}

int report(const char* kernel, float worst) {
    const float tolerance = 1e-5f;
    if (!(worst <= tolerance)) {
        std::printf("  %-28s FAIL (worst %.3e, tol %.0e)\n", kernel, worst,
                    tolerance);
        return 1;
    }
    std::printf("  %-28s OK   (worst %.3e, tol %.0e)\n", kernel, worst,
                tolerance);
    return 0;
}

// Rows are whole numbers of 256-value blocks, which is what the runtime
// guarantees before it hands a tensor to this kernel. The shapes cover a row
// count on and off the 8-rows-per-block tiling of the warp kernel.
const int kShapes[][2] = {
    {256, 8}, {512, 33}, {1024, 1}, {2048, 17},
};

int check(const char* kernel, const Format& format, bool warp_tiled,
          std::uint32_t block) {
    std::mt19937 rng(20260816);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    float worst = 0.0f;

    for (const auto& shape : kShapes) {
        int input_size = shape[0];
        int output_size = shape[1];
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(input_size) / 256 * output_size *
            format.block_bytes);
        format.fill(rng, packed);
        std::vector<float> vector(input_size);
        for (auto& value : vector) value = real(rng);

        std::vector<float> output(output_size, 0.0f);
        const unsigned char* packed_pointer = packed.data();
        const float* vector_pointer = vector.data();
        float* output_pointer = output.data();
        void* arguments[] = {&packed_pointer, &vector_pointer, &output_pointer,
                             &input_size, &output_size};
        flyweight_cpu_launch_named(
            kernel,
            warp_tiled ? static_cast<std::uint32_t>((output_size + 7) / 8)
                       : static_cast<std::uint32_t>(output_size),
            1, block, 0, 0, arguments);

        for (int row = 0; row < output_size; ++row)
            worst = std::fmax(worst, worst_error(format, packed, vector.data(),
                                                 input_size, row, output[row]));
    }
    return report(kernel, worst);
}

// The batched twin. Same decoder, so what this adds over the matvec cases is
// the four-token tiling and the name wiring -- a kernel registered under a name
// nothing dispatches would otherwise look fine right up until a prefill.
int check_rows(const char* kernel, const Format& format) {
    std::mt19937 rng(20260816);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    float worst = 0.0f;

    // Token counts on and off the kernel's 4-row tile.
    for (int tokens : {1, 4, 7}) {
        int input_size = 512;
        int output_size = 9;
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(input_size) / format.block_elements *
            output_size * format.block_bytes);
        format.fill(rng, packed);
        std::vector<float> vectors(
            static_cast<std::size_t>(input_size) * tokens);
        for (auto& value : vectors) value = real(rng);

        std::vector<float> output(
            static_cast<std::size_t>(output_size) * tokens, 0.0f);
        const unsigned char* packed_pointer = packed.data();
        const float* vectors_pointer = vectors.data();
        float* output_pointer = output.data();
        void* arguments[] = {&packed_pointer, &vectors_pointer, &output_pointer,
                             &input_size, &output_size, &tokens};
        flyweight_cpu_launch_named(kernel,
                                 static_cast<std::uint32_t>(output_size),
                                 static_cast<std::uint32_t>((tokens + 3) / 4),
                                 256, 0, 0, arguments);

        for (int token = 0; token < tokens; ++token)
            for (int row = 0; row < output_size; ++row)
                worst = std::fmax(
                    worst, worst_error(format, packed,
                                       vectors.data() + token * input_size,
                                       input_size, row,
                                       output[token * output_size + row]));
    }
    return report(kernel, worst);
}

// ---------------------------------------------------------------------------
// IQ4_XS against Q8-quantized activations.
//
// The Q8 path is a second decoder for a format that already had two (the
// per-element iq4xs_value and the CPU qwen_iq4xs_value), and it is the one that
// is easy to get wrong: it reconstructs four weights at a time into a packed
// int32 for __dp4a, so a mistake in the nibble-to-element mapping shows up as a
// permutation inside a group rather than as garbage.
//
// The activations here are quantized by this test rather than by the runtime's
// quantize_q8_blocks, because what is under test is "does the kernel compute
// sum(w_i * q_i * scale)" -- the reference is built from the same int8 codes
// the kernel is handed, so the quantizer itself is not in the loop.
struct Q8Activations {
    std::vector<std::int8_t> codes;
    std::vector<std::uint16_t> scales;  // one f16 per 32 values, per row
};

Q8Activations quantize_rows(std::mt19937& rng, int input_size, int rows) {
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    Q8Activations out;
    out.codes.resize(static_cast<std::size_t>(input_size) * rows);
    out.scales.resize(static_cast<std::size_t>(input_size) / 32 * rows);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < input_size / 32; ++block) {
            float values[32];
            float absmax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                values[i] = real(rng);
                absmax = std::fmax(absmax, std::fabs(values[i]));
            }
            const float scale = absmax / 127.0f;
            const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
            const std::size_t base =
                static_cast<std::size_t>(row) * input_size + block * 32;
            for (int i = 0; i < 32; ++i) {
                const int rounded =
                    static_cast<int>(std::lrintf(values[i] * inverse));
                out.codes[base + i] = static_cast<std::int8_t>(
                    rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded));
            }
            out.scales[static_cast<std::size_t>(row) * (input_size / 32) + block] =
                flyweight::cpu::float_to_half_bits(scale);
        }
    }
    return out;
}

double reference_row(const Format& format,
                     const std::vector<std::uint8_t>& packed, int row,
                     const Q8Activations& activations, int input_size,
                     int activation_row, double* magnitude) {
    const std::size_t blocks =
        static_cast<std::size_t>(input_size) / format.block_elements;
    const std::uint8_t* base =
        packed.data() + static_cast<std::size_t>(row) * blocks * format.block_bytes;
    double dot = 0.0;
    *magnitude = 0.0;
    for (int index = 0; index < input_size; ++index) {
        const std::size_t code_at =
            static_cast<std::size_t>(activation_row) * input_size + index;
        const std::size_t scale_at =
            static_cast<std::size_t>(activation_row) * (input_size / 32) +
            index / 32;
        const double activation =
            static_cast<double>(activations.codes[code_at]) *
            qwen_half_value(activations.scales[scale_at]);
        const double term =
            static_cast<double>(format.value_at(base, index)) * activation;
        dot += term;
        *magnitude += std::fabs(term);
    }
    return dot;
}

int check_q8(const char* kernel, const Format& format) {
    std::mt19937 rng(20260816);
    float worst = 0.0f;
    for (const auto& shape : kShapes) {
        int input_size = shape[0];
        int output_size = shape[1];
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(input_size) / 256 * output_size *
            format.block_bytes);
        format.fill(rng, packed);
        const auto activations = quantize_rows(rng, input_size, 1);

        std::vector<float> output(output_size, 0.0f);
        const unsigned char* packed_pointer = packed.data();
        const std::int8_t* codes = activations.codes.data();
        const std::uint16_t* scales = activations.scales.data();
        float* output_pointer = output.data();
        void* arguments[] = {&packed_pointer, &codes, &scales, &output_pointer,
                             &input_size, &output_size};
        flyweight_cpu_launch_named(kernel,
                                 static_cast<std::uint32_t>(output_size), 1, 128,
                                 0, 0, arguments);

        for (int row = 0; row < output_size; ++row) {
            double magnitude = 0.0;
            const double reference = reference_row(format, packed, row,
                                                   activations, input_size, 0,
                                                   &magnitude);
            worst = std::fmax(
                worst, static_cast<float>(std::fabs(reference - output[row]) /
                                          std::fmax(1.0, magnitude)));
        }
    }
    return report(kernel, worst);
}

int check_q8_rows(const char* kernel, const Format& format) {
    std::mt19937 rng(20260816);
    float worst = 0.0f;
    // Row counts on and off the kernel's compile-time cap of 8.
    for (int rows : {1, 5, 8}) {
        int input_size = 512;
        int output_size = 9;
        int scale_stride = input_size / 32;
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(input_size) / 256 * output_size *
            format.block_bytes);
        format.fill(rng, packed);
        const auto activations = quantize_rows(rng, input_size, rows);

        std::vector<float> output(
            static_cast<std::size_t>(output_size) * rows, 0.0f);
        const unsigned char* packed_pointer = packed.data();
        const std::int8_t* codes = activations.codes.data();
        const std::uint16_t* scales = activations.scales.data();
        float* output_pointer = output.data();
        void* arguments[] = {&packed_pointer, &codes,        &scales,
                             &output_pointer, &input_size,   &output_size,
                             &rows,           &scale_stride};
        flyweight_cpu_launch_named(kernel,
                                 static_cast<std::uint32_t>(output_size), 1, 128,
                                 0, 0, arguments);

        for (int row = 0; row < rows; ++row) {
            for (int out = 0; out < output_size; ++out) {
                double magnitude = 0.0;
                const double reference = reference_row(
                    format, packed, out, activations, input_size, row,
                    &magnitude);
                worst = std::fmax(
                    worst,
                    static_cast<float>(
                        std::fabs(reference -
                                  output[static_cast<std::size_t>(row) *
                                             output_size + out]) /
                        std::fmax(1.0, magnitude)));
            }
        }
    }
    return report(kernel, worst);
}

// ---------------------------------------------------------------------------
// The tiled prefill GEMM, for every type wired to it.
//
// One macro serves all of them, so the shared logic -- the barrier, the
// lane->group mapping, the per-warp reduction, the row tile -- is covered by
// any one of them. What is per-type, and what this therefore has to check once
// each, is the `stride` the macro is instantiated with: a wrong super-block
// size walks the wrong bytes and every other type still passes.
//
// `threads` is the kernel's block size, and it has to be passed in rather than
// assumed: the tiled and MMQ kernels have different block shapes, and until the
// MMQ tile went from 16x64 to 32x64 they happened to agree at 256. They no
// longer do, and an undersized launch does not fail loudly -- the warps that
// own the upper rows simply never run, and the check reports a plausible-
// looking ~0.2 error. Must track FLYWEIGHT_Q8_TILE_* / FLYWEIGHT_MMQ_* in
// native/include/flyweight_v2_qwen_kernels.hpp, like kQ8Tile*/kQ8Mmq* on the host.
int check_tiled(const char* kernel, const Format& format,
                std::uint32_t threads) {
    std::mt19937 rng(20260816);
    float worst = 0.0f;
    // {input_size, output_size, tokens}. rows must stay <=
    // FLYWEIGHT_Q8_TILE_TOKENS: the kernel computes that many per launch and the
    // host chunks to match (kQ8TileTokens), exactly as the rows kernel pairs
    // with FLYWEIGHT_Q8_ROWS. A wider batch is dropped, not wrapped, so these sit
    // at and just under the cap. output_size values straddle the row tile and
    // input sizes straddle the 32-group K tile.
    const int kCases[][3] = {
        {512, 9, 1},   {512, 9, 8},   {512, 9, 17},  {512, 9, 32},
        {2304, 5, 31}, {256, 3, 32},  {1024, 17, 7}, {512, 64, 32},
    };
    for (const auto& c : kCases) {
        int input_size = c[0], output_size = c[1], rows = c[2];
        int scale_stride = input_size / 32;
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(input_size) / 256 * output_size *
            format.block_bytes);
        format.fill(rng, packed);
        const auto activations = quantize_rows(rng, input_size, rows);

        std::vector<float> output(
            static_cast<std::size_t>(output_size) * rows, 0.0f);
        const unsigned char* packed_pointer = packed.data();
        const std::int8_t* codes = activations.codes.data();
        const std::uint16_t* scales = activations.scales.data();
        float* output_pointer = output.data();
        void* arguments[] = {&packed_pointer, &codes,      &scales,
                             &output_pointer, &input_size, &output_size,
                             &rows,           &scale_stride};
        // Over-provision blocks: the kernel derives its row base from
        // blockIdx.x * FLYWEIGHT_Q8_TILE_ROWS and returns early past the last
        // row, so output_size blocks covers any tile height without this test
        // having to track the macro.
        flyweight_cpu_launch_named(kernel,
                                 static_cast<std::uint32_t>(output_size), 1,
                                 threads, 0, 0, arguments);

        const std::size_t blocks = static_cast<std::size_t>(input_size) / 256;
        for (int token = 0; token < rows; ++token) {
            for (int out = 0; out < output_size; ++out) {
                const std::uint8_t* base =
                    packed.data() +
                    static_cast<std::size_t>(out) * blocks * format.block_bytes;
                double dot = 0.0, magnitude = 0.0;
                for (int index = 0; index < input_size; ++index) {
                    const std::size_t code_at =
                        static_cast<std::size_t>(token) * input_size + index;
                    const std::size_t scale_at =
                        static_cast<std::size_t>(token) * scale_stride +
                        index / 32;
                    const double activation =
                        static_cast<double>(activations.codes[code_at]) *
                        qwen_half_value(activations.scales[scale_at]);
                    const double term =
                        static_cast<double>(format.value_at(base, index)) *
                        activation;
                    dot += term;
                    magnitude += std::fabs(term);
                }
                const double got =
                    output[static_cast<std::size_t>(token) * output_size + out];
                worst = std::fmax(worst,
                                  static_cast<float>(std::fabs(dot - got) /
                                                     std::fmax(1.0, magnitude)));
            }
        }
    }
    return report(kernel, worst);
}

// The routed MMQ against a double-precision reference over the block table.
//
// This checks the two things the routed form adds to the MMQ core at once: that
// a CUDA block reads the expert its block table names, and that slot s of that
// block computes the token `sorted_routes[s] / top_k` and no other. A reference
// built from the plain kernel would only catch the second, and would pass an
// off-by-one in the expert pointer whenever the two experts happened to be
// adjacent in memory.
//
// Padded slots must be exactly 0.0: the SwiGLU and the scatter behind this run
// over the padded buffer unconditionally, so a stale value there is a wrong
// answer for some real token, not merely wasted work.
int check_routed_mmq(const char* kernel, const Format& format,
                     std::uint32_t threads) {
    std::mt19937 rng(20260828);
    const int input_size = 512, output_size = 9;
    const int rows = 7, top_k = 3, experts = 4;
    const int routes = rows * top_k;
    const int scale_stride = input_size / 32;
    const std::size_t matrix_bytes =
        static_cast<std::size_t>(input_size) / format.block_elements *
        output_size * format.block_bytes;

    std::vector<std::uint8_t> weights(matrix_bytes * experts);
    format.fill(rng, weights);
    const auto activations = quantize_rows(rng, input_size, rows);
    std::vector<std::int32_t> selected(static_cast<std::size_t>(routes));
    std::uniform_int_distribution<int> pick(0, experts - 1);
    for (auto& value : selected) value = pick(rng);

    flyweight::v2::moe::AlignedRoutes aligned;
    flyweight::v2::moe::align_blocks(selected.data(), nullptr, routes, experts,
                                   flyweight::v2::moe::kMmqBlockSize, aligned);
    const int blocks = static_cast<int>(aligned.block_experts.size());
    std::vector<unsigned long long> expert_ptrs(
        static_cast<std::size_t>(experts));
    for (int expert = 0; expert < experts; ++expert)
        expert_ptrs[expert] = reinterpret_cast<unsigned long long>(
            weights.data() + static_cast<std::size_t>(expert) * matrix_bytes);

    std::vector<float> output(
        static_cast<std::size_t>(aligned.padded_total) * output_size, -1.0f);
    {
        const unsigned long long* ptrs = expert_ptrs.data();
        const int* be = aligned.block_experts.data();
        const int* sr = aligned.sorted_routes.data();
        const std::int8_t* codes = activations.codes.data();
        const std::uint16_t* scales = activations.scales.data();
        float* out = output.data();
        int in = input_size, on = output_size, tk = top_k, ss = scale_stride;
        int nb = blocks;
        void* arguments[] = {&ptrs, &be, &sr, &codes, &scales, &out,
                             &in,   &on, &tk, &ss,    &nb};
        flyweight_cpu_launch_named(kernel, static_cast<std::uint32_t>(output_size),
                                 static_cast<std::uint32_t>(blocks), threads, 0,
                                 0, arguments);
    }

    float worst = 0.0f;
    int compared = 0;
    for (int block = 0; block < blocks; ++block) {
        const int expert = aligned.block_experts[block];
        for (int s = 0; s < flyweight::v2::moe::kMmqBlockSize; ++s) {
            const std::size_t slot =
                static_cast<std::size_t>(block) * flyweight::v2::moe::kMmqBlockSize + s;
            const auto route = aligned.sorted_routes[slot];
            for (int out = 0; out < output_size; ++out) {
                const double got = output[slot * output_size + out];
                if (route == flyweight::v2::moe::kEmpty) {
                    if (got != 0.0) {
                        std::printf("  %-28s FAIL (padded slot %zu row %d = %g)\n",
                                    kernel, slot, out, got);
                        return 1;
                    }
                    continue;
                }
                const std::uint8_t* base =
                    weights.data() + static_cast<std::size_t>(expert) * matrix_bytes +
                    static_cast<std::size_t>(out) *
                        (input_size / format.block_elements) * format.block_bytes;
                // Not reference_row: that one indexes rows from the start of
                // the buffer, and here the row lives inside a chosen expert's
                // slice, which is precisely the addressing under test.
                double expected = 0.0, magnitude = 0.0;
                for (int index = 0; index < input_size; ++index) {
                    const std::size_t code_at =
                        static_cast<std::size_t>(route / top_k) * input_size + index;
                    const std::size_t scale_at =
                        static_cast<std::size_t>(route / top_k) * scale_stride +
                        index / 32;
                    const double activation =
                        static_cast<double>(activations.codes[code_at]) *
                        qwen_half_value(activations.scales[scale_at]);
                    const double term =
                        static_cast<double>(format.value_at(base, index)) * activation;
                    expected += term;
                    magnitude += std::fabs(term);
                }
                worst = std::fmax(worst, static_cast<float>(
                    std::fabs(expected - got) / std::fmax(1.0, magnitude)));
                ++compared;
            }
        }
    }
    if (!compared) {
        std::printf("  %-28s FAIL (nothing compared)\n", kernel);
        return 1;
    }
    return report(kernel, worst);
}

}  // namespace


// The fused block-major MoE kernel against the route-major grouped kernel it
// replaces: identical arithmetic, different traversal, so any disagreement is a
// traversal bug -- a block reading the wrong expert's weights, or a padded slot
// overwriting a real one. Both run on the emulated corpus.
int check_block_swiglu(const char* block_kernel, const char* rows_kernel,
                       const Format& format) {
    std::mt19937 rng(20260828);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    const int input_size = 512, output_size = 9;
    const int rows = 7, top_k = 3, experts = 4;
    const int routes = rows * top_k;
    const std::size_t matrix_bytes =
        static_cast<std::size_t>(input_size) / format.block_elements *
        output_size * format.block_bytes;

    std::vector<std::uint8_t> gate(matrix_bytes * experts);
    std::vector<std::uint8_t> up(matrix_bytes * experts);
    format.fill(rng, gate);
    format.fill(rng, up);
    std::vector<float> vectors(static_cast<std::size_t>(rows) * input_size);
    for (auto& value : vectors) value = real(rng);
    std::vector<std::int32_t> selected(static_cast<std::size_t>(routes));
    std::uniform_int_distribution<int> pick(0, experts - 1);
    for (auto& value : selected) value = pick(rng);
    std::vector<int> counts(static_cast<std::size_t>(rows), top_k);

    std::vector<unsigned long long> gate_by_route(routes), up_by_route(routes);
    for (int route = 0; route < routes; ++route) {
        gate_by_route[route] = reinterpret_cast<unsigned long long>(
            gate.data() + static_cast<std::size_t>(selected[route]) * matrix_bytes);
        up_by_route[route] = reinterpret_cast<unsigned long long>(
            up.data() + static_cast<std::size_t>(selected[route]) * matrix_bytes);
    }
    std::vector<float> reference(
        static_cast<std::size_t>(routes) * output_size, 0.0f);
    {
        const unsigned long long* g = gate_by_route.data();
        const unsigned long long* u = up_by_route.data();
        const int* c = counts.data();
        const float* v = vectors.data();
        float* out = reference.data();
        int in = input_size, on = output_size, tk = top_k, rw = rows;
        void* args[] = {&g, &u, &c, &v, &out, &in, &on, &tk, &rw};
        flyweight_cpu_launch_named(rows_kernel, static_cast<std::uint32_t>(output_size),
                                 static_cast<std::uint32_t>(routes), 256, 0, 0, args);
    }

    flyweight::v2::moe::AlignedRoutes aligned;
    flyweight::v2::moe::align_blocks(selected.data(), nullptr, routes, experts,
                                   flyweight::v2::moe::kBlockSize, aligned);
    std::vector<unsigned long long> gate_by_slot, up_by_slot;
    std::vector<std::int32_t> block_slot;
    for (const auto expert : aligned.block_experts) {
        block_slot.push_back(static_cast<std::int32_t>(gate_by_slot.size()));
        gate_by_slot.push_back(reinterpret_cast<unsigned long long>(
            gate.data() + static_cast<std::size_t>(expert) * matrix_bytes));
        up_by_slot.push_back(reinterpret_cast<unsigned long long>(
            up.data() + static_cast<std::size_t>(expert) * matrix_bytes));
    }
    std::vector<float> got(
        static_cast<std::size_t>(aligned.padded_total) * output_size, 0.0f);
    {
        const unsigned long long* g = gate_by_slot.data();
        const unsigned long long* u = up_by_slot.data();
        const int* be = block_slot.data();
        const int* sr = aligned.sorted_routes.data();
        const float* v = vectors.data();
        float* out = got.data();
        int in = input_size, on = output_size, tk = top_k;
        int blocks = static_cast<int>(aligned.block_experts.size());
        void* args[] = {&g, &u, &be, &sr, &v, &out, &in, &on, &tk, &blocks};
        flyweight_cpu_launch_named(block_kernel, static_cast<std::uint32_t>(output_size),
                                 static_cast<std::uint32_t>(blocks), 256, 0, 0, args);
    }

    float worst = 0.0f;
    int compared = 0;
    for (std::size_t slot = 0; slot < aligned.sorted_routes.size(); ++slot) {
        const auto route = aligned.sorted_routes[slot];
        if (route == flyweight::v2::moe::kEmpty) continue;
        for (int row = 0; row < output_size; ++row) {
            const double expected =
                reference[static_cast<std::size_t>(route) * output_size + row];
            const double actual = got[slot * output_size + row];
            worst = std::fmax(worst, static_cast<float>(
                std::fabs(expected - actual) /
                std::fmax(1.0, std::fabs(expected))));
            ++compared;
        }
    }
    if (!compared) {
        std::printf("  %-28s FAIL (nothing compared)\n", block_kernel);
        return 1;
    }
    return report(block_kernel, worst);
}


// The block-major down projection against the token-major kernel it replaces.
// The block form writes one weighted row per slot (it cannot accumulate into the
// token directly -- see the corpus note about atomicAdd), so the comparison folds
// a token's slots first. Padded slots must contribute exactly zero.
int check_block_accumulate(const char* block_kernel, const char* rows_kernel,
                           const Format& format) {
    std::mt19937 rng(20260828);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    const int input_size = 512, output_size = 9;
    const int rows = 7, top_k = 3, experts = 4;
    const int routes = rows * top_k;
    const std::size_t matrix_bytes =
        static_cast<std::size_t>(input_size) / format.block_elements *
        output_size * format.block_bytes;

    std::vector<std::uint8_t> down(matrix_bytes * experts);
    format.fill(rng, down);
    std::vector<std::int32_t> selected(static_cast<std::size_t>(routes));
    std::uniform_int_distribution<int> pick(0, experts - 1);
    for (auto& value : selected) value = pick(rng);
    std::vector<float> weights(static_cast<std::size_t>(routes));
    for (auto& value : weights) value = real(rng);
    std::vector<int> counts(static_cast<std::size_t>(rows), top_k);
    std::vector<float> activated(static_cast<std::size_t>(routes) * input_size);
    for (auto& value : activated) value = real(rng);

    flyweight::v2::moe::AlignedRoutes aligned;
    flyweight::v2::moe::align_blocks(selected.data(), nullptr, routes, experts,
                                   flyweight::v2::moe::kBlockSize, aligned);
    std::vector<float> activated_slots(
        static_cast<std::size_t>(aligned.padded_total) * input_size, 0.0f);
    for (std::size_t slot = 0; slot < aligned.sorted_routes.size(); ++slot) {
        const auto route = aligned.sorted_routes[slot];
        if (route == flyweight::v2::moe::kEmpty) continue;
        for (int i = 0; i < input_size; ++i)
            activated_slots[slot * input_size + i] =
                activated[static_cast<std::size_t>(route) * input_size + i];
    }

    std::vector<unsigned long long> down_by_route(routes);
    for (int route = 0; route < routes; ++route)
        down_by_route[route] = reinterpret_cast<unsigned long long>(
            down.data() + static_cast<std::size_t>(selected[route]) * matrix_bytes);
    std::vector<float> reference(
        static_cast<std::size_t>(rows) * output_size, 0.0f);
    {
        const unsigned long long* d = down_by_route.data();
        const float* a = activated.data();
        float* out = reference.data();
        const float* w = weights.data();
        const int* c = counts.data();
        int in = input_size, on = output_size, tk = top_k, rw = rows;
        void* args[] = {&d, &a, &out, &w, &c, &in, &on, &tk, &rw};
        flyweight_cpu_launch_named(rows_kernel, static_cast<std::uint32_t>(output_size),
                                 static_cast<std::uint32_t>(rows), 256, 0, 0, args);
    }

    std::vector<unsigned long long> down_by_slot;
    std::vector<std::int32_t> block_slot;
    for (const auto expert : aligned.block_experts) {
        block_slot.push_back(static_cast<std::int32_t>(down_by_slot.size()));
        down_by_slot.push_back(reinterpret_cast<unsigned long long>(
            down.data() + static_cast<std::size_t>(expert) * matrix_bytes));
    }
    std::vector<float> per_slot(
        static_cast<std::size_t>(aligned.padded_total) * output_size, 1e30f);
    {
        const unsigned long long* d = down_by_slot.data();
        const int* be = block_slot.data();
        const int* sr = aligned.sorted_routes.data();
        const float* a = activated_slots.data();
        float* out = per_slot.data();
        const float* w = weights.data();
        int in = input_size, on = output_size;
        int blocks = static_cast<int>(aligned.block_experts.size());
        void* args[] = {&d, &be, &sr, &a, &out, &w, &in, &on, &blocks};
        flyweight_cpu_launch_named(block_kernel, static_cast<std::uint32_t>(output_size),
                                 static_cast<std::uint32_t>(blocks), 256, 0, 0, args);
    }

    std::vector<float> folded(static_cast<std::size_t>(rows) * output_size, 0.0f);
    for (std::size_t slot = 0; slot < aligned.sorted_routes.size(); ++slot) {
        const auto route = aligned.sorted_routes[slot];
        if (route == flyweight::v2::moe::kEmpty) {
            for (int row = 0; row < output_size; ++row)
                if (per_slot[slot * output_size + row] != 0.0f) {
                    std::printf("  %-28s FAIL (padded slot not zero)\n", block_kernel);
                    return 1;
                }
            continue;
        }
        const int token = route / top_k;
        for (int row = 0; row < output_size; ++row)
            folded[static_cast<std::size_t>(token) * output_size + row] +=
                per_slot[slot * output_size + row];
    }

    float worst = 0.0f;
    for (int token = 0; token < rows; ++token)
        for (int row = 0; row < output_size; ++row) {
            const double expected =
                reference[static_cast<std::size_t>(token) * output_size + row];
            const double actual =
                folded[static_cast<std::size_t>(token) * output_size + row];
            worst = std::fmax(worst, static_cast<float>(
                std::fabs(expected - actual) / std::fmax(1.0, std::fabs(expected))));
        }
    return report(block_kernel, worst);
}

// The grouped routed-expert kernels against the CPU decoder in double. The
// block-major checks above pin iq1s/iq2xxs octet decoders transitively (block
// vs rows form, both on the corpus); IQ3_S has no block-major twin, so its
// octet decoder gets the direct comparison, decode form and rows form both.
//
// The SwiGLU error bound folds the two dot products' condition numbers: an
// f32-accumulated gate perturbs the output through sigma(g) + g*sigma'(g),
// which is bounded by ~1.1, so |up|*sum|g_i v_i| + |silu(g)|*sum|u_i v_i|
// conditions the product the way sum|w_i v_i| conditions a plain dot.
int check_grouped_swiglu(const char* kernel, const Format& format,
                         bool rows_form) {
    std::mt19937 rng(20260901);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    const int input_size = 512, output_size = 9;
    const int rows = 7, top_k = 3, experts = 4;
    const int routes = rows * top_k;
    const std::size_t matrix_bytes =
        static_cast<std::size_t>(input_size) / format.block_elements *
        output_size * format.block_bytes;

    std::vector<std::uint8_t> gate(matrix_bytes * experts);
    std::vector<std::uint8_t> up(matrix_bytes * experts);
    format.fill(rng, gate);
    format.fill(rng, up);
    const int vector_count = rows_form ? rows : 1;
    std::vector<float> vectors(
        static_cast<std::size_t>(vector_count) * input_size);
    for (auto& value : vectors) value = real(rng);

    const int slots = rows_form ? routes : experts;
    std::vector<std::int32_t> selected(static_cast<std::size_t>(slots));
    std::uniform_int_distribution<int> pick(0, experts - 1);
    for (auto& value : selected) value = pick(rng);
    std::vector<unsigned long long> gate_ptrs(slots), up_ptrs(slots);
    for (int slot = 0; slot < slots; ++slot) {
        gate_ptrs[slot] = reinterpret_cast<unsigned long long>(
            gate.data() + static_cast<std::size_t>(selected[slot]) * matrix_bytes);
        up_ptrs[slot] = reinterpret_cast<unsigned long long>(
            up.data() + static_cast<std::size_t>(selected[slot]) * matrix_bytes);
    }
    std::vector<int> counts(static_cast<std::size_t>(rows), top_k);

    std::vector<float> output(
        static_cast<std::size_t>(slots) * output_size, 0.0f);
    {
        const unsigned long long* g = gate_ptrs.data();
        const unsigned long long* u = up_ptrs.data();
        const int* c = counts.data();
        const float* v = vectors.data();
        float* out = output.data();
        int in = input_size, on = output_size, tk = top_k, rw = rows;
        int ex = experts;
        if (rows_form) {
            void* args[] = {&g, &u, &c, &v, &out, &in, &on, &tk, &rw};
            flyweight_cpu_launch_named(
                kernel, static_cast<std::uint32_t>(output_size),
                static_cast<std::uint32_t>(routes), 256, 0, 0, args);
        } else {
            void* args[] = {&g, &u, &v, &out, &in, &on, &ex};
            flyweight_cpu_launch_named(
                kernel, static_cast<std::uint32_t>(output_size),
                static_cast<std::uint32_t>(experts), 256, 0, 0, args);
        }
    }

    float worst = 0.0f;
    for (int slot = 0; slot < slots; ++slot) {
        const float* vector =
            vectors.data() +
            static_cast<std::size_t>(rows_form ? slot / top_k : 0) * input_size;
        const std::size_t expert_at =
            static_cast<std::size_t>(selected[slot]) * matrix_bytes;
        for (int row = 0; row < output_size; ++row) {
            const std::size_t row_at =
                static_cast<std::size_t>(row) *
                (input_size / format.block_elements) * format.block_bytes;
            double gate_dot = 0.0, up_dot = 0.0;
            double gate_mag = 0.0, up_mag = 0.0;
            for (int index = 0; index < input_size; ++index) {
                const double gv = static_cast<double>(format.value_at(
                    gate.data() + expert_at + row_at, index)) * vector[index];
                const double uv = static_cast<double>(format.value_at(
                    up.data() + expert_at + row_at, index)) * vector[index];
                gate_dot += gv;
                gate_mag += std::fabs(gv);
                up_dot += uv;
                up_mag += std::fabs(uv);
            }
            const double silu = gate_dot / (1.0 + std::exp(-gate_dot));
            const double expected = silu * up_dot;
            const double got =
                output[static_cast<std::size_t>(slot) * output_size + row];
            worst = std::fmax(
                worst,
                static_cast<float>(
                    std::fabs(expected - got) /
                    std::fmax(1.0, gate_mag * std::fabs(up_dot) +
                                       std::fabs(silu) * up_mag)));
        }
    }
    return report(kernel, worst);
}

// The grouped down projection, decode form (experts looped inside one block
// per row, += into the token's vector) and rows form (one token per grid row,
// += per token). Reference is the CPU decoder in double, normalized by the
// accumulated magnitude exactly as worst_error does for a plain dot.
int check_grouped_accumulate(const char* kernel, const Format& format,
                             bool rows_form) {
    std::mt19937 rng(20260901);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    const int input_size = 512, output_size = 9;
    const int rows = 7, top_k = 3, experts = 4;
    const int routes = rows * top_k;
    const std::size_t matrix_bytes =
        static_cast<std::size_t>(input_size) / format.block_elements *
        output_size * format.block_bytes;

    std::vector<std::uint8_t> down(matrix_bytes * experts);
    format.fill(rng, down);
    const int slots = rows_form ? routes : experts;
    std::vector<std::int32_t> selected(static_cast<std::size_t>(slots));
    std::uniform_int_distribution<int> pick(0, experts - 1);
    for (auto& value : selected) value = pick(rng);
    std::vector<unsigned long long> down_ptrs(slots);
    for (int slot = 0; slot < slots; ++slot)
        down_ptrs[slot] = reinterpret_cast<unsigned long long>(
            down.data() + static_cast<std::size_t>(selected[slot]) * matrix_bytes);
    std::vector<float> weights(static_cast<std::size_t>(slots));
    for (auto& value : weights) value = real(rng);
    std::vector<float> activated(
        static_cast<std::size_t>(slots) * input_size);
    for (auto& value : activated) value = real(rng);
    std::vector<int> counts(static_cast<std::size_t>(rows), top_k);

    const int tokens = rows_form ? rows : 1;
    std::vector<float> output(
        static_cast<std::size_t>(tokens) * output_size, 0.0f);
    {
        const unsigned long long* d = down_ptrs.data();
        const float* a = activated.data();
        float* out = output.data();
        const float* w = weights.data();
        const int* c = counts.data();
        int in = input_size, on = output_size, tk = top_k, rw = rows;
        int ex = experts;
        if (rows_form) {
            void* args[] = {&d, &a, &out, &w, &c, &in, &on, &tk, &rw};
            flyweight_cpu_launch_named(
                kernel, static_cast<std::uint32_t>(output_size),
                static_cast<std::uint32_t>(rows), 256, 0, 0, args);
        } else {
            void* args[] = {&d, &a, &out, &w, &in, &on, &ex};
            flyweight_cpu_launch_named(
                kernel, static_cast<std::uint32_t>(output_size), 1, 256, 0, 0,
                args);
        }
    }

    float worst = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        const int base = rows_form ? token * top_k : 0;
        const int count = rows_form ? top_k : experts;
        for (int row = 0; row < output_size; ++row) {
            const std::size_t row_at =
                static_cast<std::size_t>(row) *
                (input_size / format.block_elements) * format.block_bytes;
            double expected = 0.0, magnitude = 0.0;
            for (int rank = 0; rank < count; ++rank) {
                const int slot = base + rank;
                const std::uint8_t* row_data =
                    down.data() +
                    static_cast<std::size_t>(selected[slot]) * matrix_bytes +
                    row_at;
                for (int index = 0; index < input_size; ++index) {
                    const double term =
                        static_cast<double>(weights[slot]) *
                        static_cast<double>(format.value_at(row_data, index)) *
                        activated[static_cast<std::size_t>(slot) * input_size +
                                  index];
                    expected += term;
                    magnitude += std::fabs(term);
                }
            }
            const double got =
                output[static_cast<std::size_t>(token) * output_size + row];
            worst = std::fmax(worst,
                              static_cast<float>(std::fabs(expected - got) /
                                                 std::fmax(1.0, magnitude)));
        }
    }
    return report(kernel, worst);
}

int main() {
    std::printf("IQ kernel contract (corpus CUDA vs CPU reference)\n");
    int failures = 0;
    // Block sizes of the two tile shapes; see check_tiled.
    const std::uint32_t kTiledThreads = 256;   // FLYWEIGHT_Q8_TILE_WARPS * 32
    const std::uint32_t kMmqThreads = 256;     // ROW_WARPS * TOKEN_WARPS * 32
    // One warp per row, eight rows per block.
    failures += check("iq1m_matvec_transposed_warp", kIq1m, true, 256);
    // One block per row, reduced across the block.
    failures += check("iq1m_matvec_transposed", kIq1m, false, 256);
    failures += check_rows("iq1m_matmul_rows", kIq1m);
    failures += check("iq1s_matvec_transposed_warp", kIq1s, true, 256);
    failures += check("iq1s_matvec_transposed", kIq1s, false, 256);
    failures += check_rows("iq1s_matmul_rows", kIq1s);
    // IQ4_NL's rows matmul is what puts a checkpoint whose expert down
    // projection is IQ4_NL onto the prefill expert-GEMM path; its accessor is
    // hand-written rather than shared with IQ4_XS, whose 256 super-block it
    // does not use.
    failures += check_rows("iq4nl_matmul_rows", kIq4nl);
    // Block-major MoE against the route-major kernel it replaces.
    failures += check_block_swiglu("iq1s_block_swiglu",
                                   "iq1s_grouped_swiglu_rows", kIq1s);
    failures += check_block_swiglu("iq2xxs_block_swiglu",
                                   "iq2xxs_grouped_swiglu_rows", kIq2xxs);
    failures += check_block_accumulate("iq1s_block_accumulate",
                                       "iq1s_grouped_accumulate_rows", kIq1s);
    failures += check_block_accumulate("iq2xxs_block_accumulate",
                                       "iq2xxs_grouped_accumulate_rows", kIq2xxs);
    failures += check_q8("iq4xs_q8_matvec_transposed_warp", kIq4xs);
    failures += check_q8_rows("iq4xs_q8_matvec_transposed_rows", kIq4xs);
    // The IQ1 pair through the Q8 path, where the delta is folded into the int8
    // weights: a sign error there is invisible to the f32 kernels above, which
    // never separate the delta from the weight. IQ1_M is the stricter of the
    // two -- its sign is picked once per eight weights rather than per 32, and
    // its two sub-scales have to land on the right halves of the block.
    failures += check_q8("iq1s_q8_matvec_transposed_warp", kIq1s);
    failures += check_q8_rows("iq1s_q8_matvec_transposed_rows", kIq1s);
    failures += check_tiled("iq1s_q8_matmul_tiled", kIq1s, kTiledThreads);
    failures += check_tiled("iq1s_q8_mmq", kIq1s, kMmqThreads);
    failures += check_q8("iq1m_q8_matvec_transposed_warp", kIq1m);
    failures += check_q8_rows("iq1m_q8_matvec_transposed_rows", kIq1m);
    failures += check_tiled("iq1m_q8_matmul_tiled", kIq1m, kTiledThreads);
    failures += check_tiled("iq1m_q8_mmq", kIq1m, kMmqThreads);
    failures += check_tiled("iq2s_q8_matmul_tiled", kIq2s, kTiledThreads);
    failures += check_tiled("iq2s_q8_mmq", kIq2s, kMmqThreads);
    failures += check_tiled("iq2xxs_q8_mmq", kIq2xxs, kMmqThreads);
    failures += check_tiled("iq3xxs_q8_mmq", kIq3xxs, kMmqThreads);
    failures += check_tiled("iq2xs_q8_mmq", kIq2xs, kMmqThreads);
    failures += check_tiled("iq4xs_q8_mmq", kIq4xs, kMmqThreads);
    // The same core driven by a block table. Its thread count is fixed by
    // FLYWEIGHT_MOE_MMQ_ROW_WARPS * FLYWEIGHT_MOE_MMQ_TOKEN_WARPS * 32, which is
    // the same 256 -- a different split of the warps, not a different budget.
    // IQ3_S, new with the qwen4exp UD checkpoints whose gate/up expert stacks
    // ship in it. No block-major twin yet, so the grouped kernels compare to
    // the CPU decoder directly, and the per-element kernels pin iq3s_value
    // itself (nothing else in the suite did).
    failures += check("iq3s_matvec_transposed", kIq3s, false, 256);
    failures += check_rows("iq3s_matmul_rows", kIq3s);
    failures += check_grouped_swiglu("iq3s_grouped_swiglu", kIq3s, false);
    failures += check_grouped_swiglu("iq3s_grouped_swiglu_rows", kIq3s, true);
    failures += check_grouped_accumulate("iq3s_grouped_accumulate", kIq3s, false);
    failures += check_grouped_accumulate("iq3s_grouped_accumulate_rows", kIq3s,
                                         true);
    failures += check_routed_mmq("iq3s_q8_mmq_routed", kIq3s, kMmqThreads);
    failures += check_routed_mmq("iq1s_q8_mmq_routed", kIq1s, kMmqThreads);
    failures += check_routed_mmq("iq2xxs_q8_mmq_routed", kIq2xxs, kMmqThreads);
    failures += check_routed_mmq("iq2xs_q8_mmq_routed", kIq2xs, kMmqThreads);
    failures += check_routed_mmq("iq3xxs_q8_mmq_routed", kIq3xxs, kMmqThreads);
    failures += check_routed_mmq("iq4xs_q8_mmq_routed", kIq4xs, kMmqThreads);
    // IQ4_NL exercises the flat-block shifts (5, 0) rather than the
    // super-block (8, 3) every other format above takes, which is the whole
    // reason the expert down projection can reach this kernel at all.
    failures += check_routed_mmq("iq4nl_q8_mmq_routed", kIq4nl, kMmqThreads);
    failures += check_routed_mmq("iq2s_q8_mmq_routed", kIq2s, kMmqThreads);
    failures += check_routed_mmq("iq1m_q8_mmq_routed", kIq1m, kMmqThreads);
    failures += check_routed_mmq("q3k_q8_mmq_routed", kQ3k, kMmqThreads);
    failures += check_routed_mmq("q6k_q8_mmq_routed", kQ6k, kMmqThreads);
    // The _MIN routed macro: 64x32 over 16 warps, so 512 threads rather than
    // the plain form's 256. Its extra staged arrays carry the group minimum
    // and the activation sums that cancel it.
    const std::uint32_t kMmqMinRoutedThreads = 512;
    failures += check_routed_mmq("q4k_q8_mmq_routed", kQ4k, kMmqMinRoutedThreads);
    failures += check_routed_mmq("q5k_q8_mmq_routed", kQ5k, kMmqMinRoutedThreads);
    failures += check_routed_mmq("q2k_q8_mmq_routed", kQ2k, kMmqMinRoutedThreads);
    failures += check_tiled("iq2xxs_q8_matmul_tiled", kIq2xxs, kTiledThreads);
    failures += check_tiled("iq3xxs_q8_matmul_tiled", kIq3xxs, kTiledThreads);
    failures += check_tiled("iq2xs_q8_matmul_tiled", kIq2xs, kTiledThreads);
    failures += check_tiled("iq4xs_q8_matmul_tiled", kIq4xs, kTiledThreads);
    std::printf(failures ? "FAILED (%d failures)\n" : "PASSED (%d failures)\n",
                failures);
    return failures ? 1 : 0;
}
