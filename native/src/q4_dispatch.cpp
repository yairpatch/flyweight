#include "colibri_native.h"
#include "q4_kernel.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace {

constexpr std::uint32_t kFeatureAvx2 = 1u << 0;
constexpr std::uint32_t kFeatureAvx512 = 1u << 1;

// Read XCR0 without the _xgetbv intrinsic, which GCC/Clang refuse to inline
// unless the translation unit enables the "xsave" target feature. Raw XGETBV
// has no such requirement and is only reached after the OSXSAVE bit is set.
std::uint64_t read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

std::uint32_t detect_features() {
#if defined(__x86_64__) || defined(_M_X64)
    int registers[4] = {};
#if defined(_MSC_VER)
    __cpuid(registers, 1);
#else
    __cpuid(1, registers[0], registers[1], registers[2], registers[3]);
#endif
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    const bool f16c = (registers[2] & (1 << 29)) != 0;
    if (!osxsave || !avx || !f16c) {
        return 0;
    }
    const std::uint64_t xcr0 = read_xcr0();
    if ((xcr0 & 0x6) != 0x6) {
        return 0;
    }
#if defined(_MSC_VER)
    __cpuidex(registers, 7, 0);
#else
    __cpuid_count(7, 0, registers[0], registers[1], registers[2], registers[3]);
#endif
    std::uint32_t features = 0;
    if ((registers[1] & (1 << 5)) != 0) {
        features |= kFeatureAvx2;
    }
    const bool avx512_hardware = (registers[1] & (1 << 16)) != 0;
    const bool avx512_bw = (registers[1] & (1 << 30)) != 0;
    const bool avx512_os = (xcr0 & 0xE0) == 0xE0;
    if (avx512_hardware && avx512_bw && avx512_os) {
        features |= kFeatureAvx512;
    }
    return features;
#else
    return 0;
#endif
}

Q4MatvecKernel select_kernel(std::uint32_t features) {
    if ((features & kFeatureAvx512) != 0) {
        return q4_matvec_avx512;
    }
    if ((features & kFeatureAvx2) != 0) {
        return q4_matvec_avx2;
    }
    return q4_matvec_scalar;
}

const std::uint32_t kCpuFeatures = detect_features();
const Q4MatvecKernel kQ4Kernel = select_kernel(kCpuFeatures);

}

extern "C" std::uint32_t colibri_native_version() {
    return 2;
}

extern "C" std::uint32_t colibri_cpu_features() {
    return kCpuFeatures;
}

extern "C" int colibri_q4_matvec(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
) {
    if (packed == nullptr || scales == nullptr || vector == nullptr
        || output == nullptr || rows <= 0 || columns <= 0) {
        return -1;
    }
    return kQ4Kernel(packed, scales, vector, output, rows, columns);
}

extern "C" int colibri_q4_moe(
    const std::uint8_t* const* gate_up_packed,
    const std::uint16_t* const* gate_up_scales,
    const std::uint8_t* const* down_packed,
    const std::uint16_t* const* down_scales,
    const float* weights,
    const float* input,
    float* output,
    std::int32_t num_experts,
    std::int32_t hidden_size,
    std::int32_t intermediate_size
) {
    if (gate_up_packed == nullptr || gate_up_scales == nullptr
        || down_packed == nullptr || down_scales == nullptr
        || weights == nullptr || input == nullptr || output == nullptr
        || num_experts <= 0 || hidden_size <= 0 || intermediate_size <= 0) {
        return -1;
    }
    const std::int32_t gate_rows = 2 * intermediate_size;

    // Streaming the expert weights is the bottleneck, and a single core cannot
    // saturate RAM bandwidth. Parallelize over (expert, row-chunk) work items
    // so the whole team streams concurrently; the row-offset arithmetic
    // requires whole Q4 blocks per row.
    if (hidden_size % 32 == 0 && intermediate_size % 32 == 0) {
#if defined(_OPENMP)
        // SMT siblings fight over the shared load ports and the spin barriers
        // between phases, which collapses throughput (measured ~50x slower at
        // 32 threads than 16 on a 16C/32T part). Default the team to the
        // physical core count unless the user pinned OMP_NUM_THREADS.
        int team = omp_get_max_threads();
        if (std::getenv("OMP_NUM_THREADS") == nullptr) {
            const int physical = omp_get_num_procs() / 2;
            if (physical >= 1 && team > physical) {
                team = physical;
            }
        }
#endif
        constexpr std::int32_t kChunkRows = 64;
        const std::int32_t gate_blocks_per_row = hidden_size / 32;
        const std::int32_t down_blocks_per_row = intermediate_size / 32;
        const std::int32_t gate_chunks =
            (gate_rows + kChunkRows - 1) / kChunkRows;
        const std::int32_t down_chunks =
            (hidden_size + kChunkRows - 1) / kChunkRows;
        const std::int32_t gate_tasks = num_experts * gate_chunks;
        const std::int32_t down_tasks = num_experts * down_chunks;
        std::vector<float> gate(
            static_cast<std::size_t>(num_experts) * gate_rows
        );
        std::vector<float> activated(
            static_cast<std::size_t>(num_experts) * intermediate_size
        );
        std::vector<float> expert_output(
            static_cast<std::size_t>(num_experts) * hidden_size
        );
#if defined(_OPENMP)
#pragma omp parallel num_threads(team)
#endif
        {
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
            for (std::int32_t task = 0; task < gate_tasks; ++task) {
                const std::int32_t expert = task / gate_chunks;
                const std::int32_t row0 = (task % gate_chunks) * kChunkRows;
                const std::int32_t rows =
                    row0 + kChunkRows > gate_rows ? gate_rows - row0 : kChunkRows;
                const std::int64_t block0 =
                    static_cast<std::int64_t>(row0) * gate_blocks_per_row;
                kQ4Kernel(
                    gate_up_packed[expert] + block0 * 16,
                    gate_up_scales[expert] + block0,
                    input,
                    gate.data()
                        + static_cast<std::size_t>(expert) * gate_rows + row0,
                    rows,
                    hidden_size
                );
            }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (std::int32_t index = 0;
                 index < num_experts * intermediate_size; ++index) {
                const std::int32_t expert = index / intermediate_size;
                const std::int32_t column = index % intermediate_size;
                const float* expert_gate = gate.data()
                    + static_cast<std::size_t>(expert) * gate_rows;
                const float value = expert_gate[column];
                const float silu = value / (1.0f + std::exp(-value));
                activated[index] = silu * expert_gate[intermediate_size + column];
            }
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
            for (std::int32_t task = 0; task < down_tasks; ++task) {
                const std::int32_t expert = task / down_chunks;
                const std::int32_t row0 = (task % down_chunks) * kChunkRows;
                const std::int32_t rows =
                    row0 + kChunkRows > hidden_size
                        ? hidden_size - row0
                        : kChunkRows;
                const std::int64_t block0 =
                    static_cast<std::int64_t>(row0) * down_blocks_per_row;
                kQ4Kernel(
                    down_packed[expert] + block0 * 16,
                    down_scales[expert] + block0,
                    activated.data()
                        + static_cast<std::size_t>(expert) * intermediate_size,
                    expert_output.data()
                        + static_cast<std::size_t>(expert) * hidden_size + row0,
                    rows,
                    intermediate_size
                );
            }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (std::int32_t index = 0; index < hidden_size; ++index) {
                float total = 0.0f;
                for (std::int32_t expert = 0; expert < num_experts; ++expert) {
                    total += weights[expert] * expert_output[
                        static_cast<std::size_t>(expert) * hidden_size + index
                    ];
                }
                output[index] = total;
            }
        }
        return 0;
    }

#if defined(_OPENMP)
    // Fallback: parallelize across experts only, capped at the expert count so
    // idle threads do not spin at the barrier.
    int thread_count = omp_get_max_threads();
    if (thread_count > num_experts) {
        thread_count = num_experts;
    }
    if (thread_count < 1) {
        thread_count = 1;
    }
#else
    const int thread_count = 1;
#endif
    std::vector<float> partials(
        static_cast<std::size_t>(thread_count) * hidden_size, 0.0f
    );

#if defined(_OPENMP)
#pragma omp parallel num_threads(thread_count)
#endif
    {
#if defined(_OPENMP)
        const int thread_id = omp_get_thread_num();
#else
        const int thread_id = 0;
#endif
        float* accumulator = partials.data()
            + static_cast<std::size_t>(thread_id) * hidden_size;
        std::vector<float> gate_up(gate_rows);
        std::vector<float> activated(intermediate_size);
        std::vector<float> down_output(hidden_size);
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
        for (std::int32_t expert = 0; expert < num_experts; ++expert) {
            kQ4Kernel(
                gate_up_packed[expert],
                gate_up_scales[expert],
                input,
                gate_up.data(),
                gate_rows,
                hidden_size
            );
            for (std::int32_t index = 0; index < intermediate_size; ++index) {
                const float gate = gate_up[index];
                const float silu = gate / (1.0f + std::exp(-gate));
                activated[index] = silu * gate_up[intermediate_size + index];
            }
            kQ4Kernel(
                down_packed[expert],
                down_scales[expert],
                activated.data(),
                down_output.data(),
                hidden_size,
                intermediate_size
            );
            const float weight = weights[expert];
            for (std::int32_t index = 0; index < hidden_size; ++index) {
                accumulator[index] += weight * down_output[index];
            }
        }
    }

    for (std::int32_t index = 0; index < hidden_size; ++index) {
        float total = 0.0f;
        for (int thread = 0; thread < thread_count; ++thread) {
            total += partials[
                static_cast<std::size_t>(thread) * hidden_size + index
            ];
        }
        output[index] = total;
    }
    return 0;
}
