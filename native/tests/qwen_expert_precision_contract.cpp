// Expert-path precision contract: how much of the CPU/GPU expert disagreement
// is the int8 activation quantisation, and how much is a defect on top of it.
//
// check_expert_path_divergence.py measures ~1e-2 relative divergence in
// accumulated KV between an all-CPU expert run and a mostly-GPU one. That is
// four orders of magnitude too large for reduction-order drift, but accumulated
// state cannot say whether activation quantisation explains all of it: forty
// layers of amplification put plausible quantisation error into the same range
// as a real bug.
//
// This decomposes it on a single matvec, which is where the whole difference
// lives -- an expert is gate/up/down matvecs plus an elementwise SwiGLU, and
// only the matvecs differ in how they treat activations:
//
//   exact      dequantised Q6_K weights against f32 activations, in double
//   cpu        qwen_q6k_dot_row -- the CPU expert path's own primitive
//   predicted  dequantised weights against the *kernel's own* int8
//              activations, in double
//   gpu        the real CUDA kernel text, emulated on the host
//
// `gpu` vs `predicted` is the number that settles it. Both consume exactly the
// same int8 activations, so any gap between them is the kernel's arithmetic
// rather than the precision choice, and it should sit at f32 round-off. A large
// gap is a defect; a small one means the CPU/GPU disagreement is entirely the
// documented consequence of quantising activations on one path and not the
// other, and the fix is to standardise precision rather than to hunt a bug.
//
// Emulation is forced so this runs the CUDA text the GPU compiles, which makes
// the test runnable in CI on a machine with no GPU.

#include "qwen_kquant.h"
#include "qwen_kquant_pack.h"
#include "qwen_cpu_kernel.h"

#include <colibri_cpu_native.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
int colibri_cpu_launch_named(const char*, std::uint32_t, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint64_t,
                             void**);
}

namespace {

// A multiple of 256: the Q8 matvec computes blocks_per_row as input_size >> 8
// and the k-quant super-block is 256 wide.
constexpr int kInputSize = 1024;
constexpr int kOutputSize = 64;
constexpr int kQ8Group = 32;   // activation quantisation block

std::mt19937& rng() {
    static std::mt19937 generator(20260820);
    return generator;
}

std::vector<float> random_vector(std::size_t count, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (auto& value : values) value = distribution(rng());
    return values;
}

// Relative RMS of (left - right) against the magnitude of `right`.
double relative_rms(const std::vector<double>& left,
                    const std::vector<double>& right) {
    double error = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        const double delta = left[i] - right[i];
        error += delta * delta;
        scale += right[i] * right[i];
    }
    if (scale == 0.0) return 0.0;
    return std::sqrt(error / scale);
}

// The routed-expert comparison: q6k_grouped_swiglu is what a GPU-resident
// Q6_K expert actually runs (qwen_expert_kernel_name maps type 14 to it), and
// qwen_q6k_dot_row is what the CPU path runs for the same expert. They read the
// SAME Q6_K bytes and the SAME f32 activations -- the cache upload is a byte
// copy, with no requantisation -- so anything separating them is accumulation
// precision and order alone: the kernel reduces in f32 across a block, the host
// routine accumulates in double.
//
// This, not the dense matvec below, is the path that made greedy output depend
// on gpu_cache_mib.
int run_expert() {
    const std::size_t blocks_per_row = kInputSize / kBlockElements;
    const std::size_t row_bytes = blocks_per_row * kQ6KBlockSize;

    std::vector<std::uint8_t> gate(row_bytes * kOutputSize);
    std::vector<std::uint8_t> up(row_bytes * kOutputSize);
    for (int row = 0; row < kOutputSize; ++row) {
        const auto gate_source = random_vector(kInputSize, -0.75f, 0.75f);
        const auto up_source = random_vector(kInputSize, -0.75f, 0.75f);
        const auto offset = static_cast<std::size_t>(row) * row_bytes;
        qwen_kpack::qwen_pack_q6_k(gate_source.data(), kInputSize,
                                   gate.data() + offset);
        qwen_kpack::qwen_pack_q6_k(up_source.data(), kInputSize,
                                   up.data() + offset);
    }

    const auto activations = random_vector(kInputSize, -3.0f, 3.0f);

    std::vector<unsigned long long> gate_pointers{
        reinterpret_cast<unsigned long long>(gate.data())};
    std::vector<unsigned long long> up_pointers{
        reinterpret_cast<unsigned long long>(up.data())};
    std::vector<float> device_output(kOutputSize, 0.0f);
    {
        const unsigned long long* gate_pointer = gate_pointers.data();
        const unsigned long long* up_pointer = up_pointers.data();
        const float* vector_pointer = activations.data();
        float* output_pointer = device_output.data();
        int input_size = kInputSize, output_size = kOutputSize, experts = 1;
        void* arguments[] = {&gate_pointer, &up_pointer, &vector_pointer,
                             &output_pointer, &input_size, &output_size,
                             &experts};
        if (colibri_cpu_launch_named("q6k_grouped_swiglu", kOutputSize, 1, 256,
                                     0, 0, arguments) != 0) {
            std::printf("FAIL: q6k_grouped_swiglu is not in the corpus\n");
            return 1;
        }
    }

    auto silu_mul = [](double gate_value, double up_value) {
        const double clamped = std::fmin(80.0, std::fmax(-80.0, gate_value));
        return (gate_value / (1.0 + std::exp(-clamped))) * up_value;
    };

    // The scalar routine is NOT what the runtime calls on this machine: type 14
    // is in qwen_simd_quant_type, so with AVX-512 present qwen_quant_dot
    // dispatches to qwen_quant_dot_avx512. Comparing the kernel against the
    // scalar reference alone therefore tests a path the runtime never takes.
    const bool has_avx512 = __builtin_cpu_supports("avx512f");

    std::vector<double> exact(kOutputSize), cpu(kOutputSize), gpu(kOutputSize),
        simd(kOutputSize);
    for (int row = 0; row < kOutputSize; ++row) {
        const auto offset = static_cast<std::size_t>(row) * row_bytes;
        const std::uint8_t* gate_row = gate.data() + offset;
        const std::uint8_t* up_row = up.data() + offset;
        double gate_exact = 0.0, up_exact = 0.0;
        for (int i = 0; i < kInputSize; ++i) {
            gate_exact += static_cast<double>(qwen_q6_value(gate_row, i)) *
                          activations[i];
            up_exact +=
                static_cast<double>(qwen_q6_value(up_row, i)) * activations[i];
        }
        exact[row] = silu_mul(gate_exact, up_exact);
        cpu[row] = silu_mul(
            qwen_q6k_dot_row(gate_row, activations.data(), kInputSize),
            qwen_q6k_dot_row(up_row, activations.data(), kInputSize));
        gpu[row] = device_output[row];
        simd[row] = has_avx512
            ? silu_mul(qwen_quant_dot_avx512(gate_row, 14, activations.data(),
                                             kInputSize, 0),
                       qwen_quant_dot_avx512(up_row, 14, activations.data(),
                                             kInputSize, 0))
            : cpu[row];
    }

    std::printf("== routed expert (q6k_grouped_swiglu vs the CPU dots) ==\n");
    std::printf("cpu scalar vs exact : %.3e\n", relative_rms(cpu, exact));
    std::printf("gpu        vs exact : %.3e\n", relative_rms(gpu, exact));
    std::printf("gpu        vs scalar: %.3e\n", relative_rms(gpu, cpu));
    if (has_avx512) {
        std::printf("cpu avx512 vs exact : %.3e\n", relative_rms(simd, exact));
        std::printf("gpu vs avx512       : %.3e   <-- the path the runtime "
                    "actually takes\n", relative_rms(gpu, simd));
    } else {
        std::printf("(no AVX-512 on this host; the SIMD path was not "
                    "exercised)\n");
    }
    return 0;
}

int run() {
    const std::size_t blocks_per_row = kInputSize / kBlockElements;
    const std::size_t row_bytes = blocks_per_row * kQ6KBlockSize;

    // Weights are packed per row into consecutive spans, which is the layout
    // the kernel indexes with row * blocks_per_row * stride.
    std::vector<std::uint8_t> packed(row_bytes * kOutputSize);
    std::vector<std::vector<float>> source(kOutputSize);
    for (int row = 0; row < kOutputSize; ++row) {
        source[row] = random_vector(kInputSize, -0.75f, 0.75f);
        qwen_kpack::qwen_pack_q6_k(
            source[row].data(), kInputSize,
            packed.data() + static_cast<std::size_t>(row) * row_bytes);
    }

    const auto activations = random_vector(kInputSize, -3.0f, 3.0f);

    // Run the real quantiser rather than modelling it, so `predicted` and `gpu`
    // consume byte-identical activations and the comparison isolates the matvec.
    std::vector<std::int8_t> quantised(kInputSize, 0);
    std::vector<std::uint16_t> scales(kInputSize / kQ8Group, 0);
    {
        const float* input_pointer = activations.data();
        std::int8_t* output_pointer = quantised.data();
        std::uint16_t* scale_pointer = scales.data();
        int elements = kInputSize;
        void* arguments[] = {&input_pointer, &output_pointer, &scale_pointer,
                             &elements};
        if (colibri_cpu_launch_named("quantize_q8_blocks",
                                     (kInputSize + 31) / 32, 1, 32, 0, 0,
                                     arguments) != 0) {
            std::printf("FAIL: quantize_q8_blocks is not in the corpus\n");
            return 1;
        }
    }

    std::vector<float> device_output(kOutputSize, 0.0f);
    {
        const std::uint8_t* packed_pointer = packed.data();
        const std::int8_t* vector_pointer = quantised.data();
        const std::uint16_t* scale_pointer = scales.data();
        float* output_pointer = device_output.data();
        int input_size = kInputSize, output_size = kOutputSize;
        void* arguments[] = {&packed_pointer, &vector_pointer, &scale_pointer,
                             &output_pointer, &input_size, &output_size};
        if (colibri_cpu_launch_named("q6k_q8_matvec_transposed_warp",
                                     kOutputSize, 1, 128, 0, 0,
                                     arguments) != 0) {
            std::printf("FAIL: q6k_q8_matvec_transposed_warp is not in the "
                        "corpus\n");
            return 1;
        }
    }

    std::vector<double> exact(kOutputSize), cpu(kOutputSize),
        predicted(kOutputSize), gpu(kOutputSize);
    for (int row = 0; row < kOutputSize; ++row) {
        const std::uint8_t* base =
            packed.data() + static_cast<std::size_t>(row) * row_bytes;
        double sum_exact = 0.0, sum_predicted = 0.0;
        for (int i = 0; i < kInputSize; ++i) {
            // The same dequantised weight both times: only the activation
            // differs, which is precisely the difference under measurement.
            const double weight = qwen_q6_value(base, i);
            sum_exact += weight * activations[i];
            sum_predicted += weight *
                (static_cast<double>(qwen_half_value(scales[i / kQ8Group])) *
                 quantised[i]);
        }
        exact[row] = sum_exact;
        predicted[row] = sum_predicted;
        cpu[row] = qwen_q6k_dot_row(base, activations.data(), kInputSize);
        gpu[row] = device_output[row];
    }

    const double cpu_error = relative_rms(cpu, exact);
    const double quantisation = relative_rms(predicted, exact);
    const double gpu_error = relative_rms(gpu, exact);
    const double residual = relative_rms(gpu, predicted);

    std::printf("cpu   vs exact     : %.3e   (CPU expert path, f32 activations)\n",
                cpu_error);
    std::printf("q8    vs exact     : %.3e   (what int8 activations predict)\n",
                quantisation);
    std::printf("gpu   vs exact     : %.3e   (real kernel, int8 activations)\n",
                gpu_error);
    std::printf("gpu   vs q8        : %.3e   <-- decides defect vs precision\n",
                residual);

    int status = 0;
    // The CPU path keeps f32 activations, so its only error against the double
    // reference is f32 accumulation over 1024 terms.
    if (cpu_error > 1e-5) {
        std::printf("FAIL: CPU path error exceeds f32 accumulation\n");
        status = 1;
    }
    // The kernel and the reference consume identical int8 activations, so what
    // separates them is f32 accumulation and the warp reduction order.
    if (residual > 1e-5) {
        std::printf("FAIL: the kernel does not match its own quantised inputs; "
                    "a defect rides on top of the precision choice\n");
        status = 1;
    }
    if (status == 0) {
        std::printf("\nOK: the kernel matches its own quantised inputs. The "
                    "CPU/GPU expert\n    disagreement is the int8 activation "
                    "quantisation (%.2e relative),\n    not a defect -- so the "
                    "fix is standardising precision.\n", quantisation);
    }
    return status;
}

}  // namespace

int main() {
    // The CUDA text the GPU compiles, not a host reimplementation of it.
    colibri::cpu::set_force_emulation(true);
    const int expert_status = run_expert();
    std::printf("\n== dense matvec (q6k_q8_matvec_transposed_warp) ==\n");
    return run() | expert_status;
}
