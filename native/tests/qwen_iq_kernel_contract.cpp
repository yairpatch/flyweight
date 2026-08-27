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

#include <colibri_backend.hpp>
#include <colibri_cpu_kernels_api.hpp>
#include <colibri_cpu_shim_geometry.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

#include "qwen_kquant.h"

extern "C" {
int colibri_cpu_launch_named(const char*, std::uint32_t, std::uint32_t,
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
const Format kIq4xs{
    kIq4xsBlockBytes, fill_leading_scale<kIq4xsBlockBytes>, qwen_iq4xs_value};

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
        colibri_cpu_launch_named(
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
        colibri_cpu_launch_named(kernel,
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
                colibri::cpu::float_to_half_bits(scale);
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
        colibri_cpu_launch_named(kernel,
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
        colibri_cpu_launch_named(kernel,
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
// looking ~0.2 error. Must track COLIBRI_Q8_TILE_* / COLIBRI_MMQ_* in
// native/include/colibri_v2_qwen_kernels.hpp, like kQ8Tile*/kQ8Mmq* on the host.
int check_tiled(const char* kernel, const Format& format,
                std::uint32_t threads) {
    std::mt19937 rng(20260816);
    float worst = 0.0f;
    // {input_size, output_size, tokens}. rows must stay <=
    // COLIBRI_Q8_TILE_TOKENS: the kernel computes that many per launch and the
    // host chunks to match (kQ8TileTokens), exactly as the rows kernel pairs
    // with COLIBRI_Q8_ROWS. A wider batch is dropped, not wrapped, so these sit
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
        // blockIdx.x * COLIBRI_Q8_TILE_ROWS and returns early past the last
        // row, so output_size blocks covers any tile height without this test
        // having to track the macro.
        colibri_cpu_launch_named(kernel,
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

}  // namespace

int main() {
    std::printf("IQ kernel contract (corpus CUDA vs CPU reference)\n");
    int failures = 0;
    // Block sizes of the two tile shapes; see check_tiled.
    const std::uint32_t kTiledThreads = 256;   // COLIBRI_Q8_TILE_WARPS * 32
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
    failures += check_tiled("iq2xxs_q8_matmul_tiled", kIq2xxs, kTiledThreads);
    failures += check_tiled("iq3xxs_q8_matmul_tiled", kIq3xxs, kTiledThreads);
    failures += check_tiled("iq2xs_q8_matmul_tiled", kIq2xs, kTiledThreads);
    failures += check_tiled("iq4xs_q8_matmul_tiled", kIq4xs, kTiledThreads);
    std::printf(failures ? "FAILED (%d failures)\n" : "PASSED (%d failures)\n",
                failures);
    return failures ? 1 : 0;
}
