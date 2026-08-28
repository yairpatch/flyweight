// One MoE layer's CPU expert phase, at production shape and threading.
//
// WHY THIS EXISTS. Chunked prefill spends nearly all of its wall clock in
// qwen_cpu_moe_rows, and for two sessions that cost was attributed to the wrong
// things -- kernel tiling, dispatch counts, upload bandwidth, activation
// quantization -- because the only available numbers were end-to-end. What was
// actually happening is that one weight type had no vectorized row decoder and
// fell to the per-element form, which costs about 10x.
//
// This harness makes that visible in one line. It reproduces the production
// loop exactly: the same CSR grouping by expert, the same 4-row task blocks,
// the same dynamic schedule, the same per-region timers COLIBRI_MOE_PROFILE
// reports -- but with the weight type on the command line. A format whose
// dequant/gemm ratio is far above its neighbours' is a format on the scalar
// path, and the runtime prints a one-time warning naming it (v2_runtime.cpp,
// g_scalar_dequant_types).
//
// Threads are plain std::thread over an atomic task counter, which is what
// `schedule(dynamic, chunk)` is, so the tool needs no OpenMP of its own.
//
// The optional third argument repeats the measured pass for that many seconds
// and prints every pass. A prefill runs this loop on every core for tens of
// seconds; a benchmark that measures a 50 ms burst reads the clocks at boost
// and flatters itself. Comparing the first pass against the last is how you
// tell a software limit from a thermal one.
//
//   colibri_qwen_moe_layer_bench [rows] [threads] [sustain-seconds]

#include <qwen_cpu_kernel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "qwen_kquant.h"

namespace {

constexpr int kHidden = 2560;       // gate/up input width
constexpr int kIntermediate = 640;  // gate/up output rows, down input width
constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kRowBlock = 4;
constexpr int kScheduleChunk = 4;

double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::size_t block_bytes_of(std::uint32_t type) {
    if (type == 16) return kIq2xxsBlockBytes;
    if (type == 17) return kIq2xsBlockBytes;
    if (type == 18) return kIq3xxsBlockBytes;
    if (type == 19) return kIq1sBlockBytes;
    return kIq4nlBlockBytes;
}

int block_elements_of(std::uint32_t type) { return type == 20 ? 32 : 256; }

const char* name_of(std::uint32_t type) {
    switch (type) {
        case 16: return "IQ2_XXS gate+up";
        case 17: return "IQ2_XS  gate+up";
        case 18: return "IQ3_XXS gate+up";
        case 19: return "IQ1_S   gate+up";
        default: return "IQ4_NL  gate+up";
    }
}

// Random codes with sane f16 block scales. Random scale bit patterns would be
// infinities and NaNs, which decode at a different speed and compare to
// nothing.
std::vector<std::uint8_t> random_packed(
    std::size_t bytes, std::uint32_t type, std::mt19937& engine
) {
    std::vector<std::uint8_t> data(bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : data) value = static_cast<std::uint8_t>(byte(engine));
    const auto stride = block_bytes_of(type);
    for (std::size_t offset = 0; offset + 2 <= data.size(); offset += stride) {
        const std::uint16_t scale =
            0x2C00 | static_cast<std::uint16_t>(byte(engine) & 0x3FF);
        std::memcpy(data.data() + offset, &scale, sizeof(scale));
    }
    return data;
}

// `schedule(dynamic, chunk)` over [0, tasks), run on `threads` workers.
template <typename Body>
void dynamic_for(int tasks, int threads, Body&& body) {
    std::atomic<int> cursor{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (int worker = 0; worker < threads; ++worker)
        workers.emplace_back([&] {
            for (;;) {
                const int first =
                    cursor.fetch_add(kScheduleChunk, std::memory_order_relaxed);
                if (first >= tasks) return;
                const int last = std::min(first + kScheduleChunk, tasks);
                for (int task = first; task < last; ++task) body(task);
            }
        });
    for (auto& worker : workers) worker.join();
}

struct Routing {
    std::vector<int> counts, offsets, occurrences, group_experts;
};

Routing route(int rows, std::mt19937& engine) {
    Routing routing;
    std::vector<int> selected(static_cast<std::size_t>(rows) * kTopK);
    std::uniform_int_distribution<int> pick(0, kExperts - 1);
    for (auto& value : selected) value = pick(engine);
    routing.counts.assign(kExperts, 0);
    for (const int expert : selected) ++routing.counts[expert];
    routing.offsets.resize(kExperts + 1);
    routing.offsets[0] = 0;
    for (int expert = 0; expert < kExperts; ++expert)
        routing.offsets[expert + 1] =
            routing.offsets[expert] + routing.counts[expert];
    routing.occurrences.resize(selected.size());
    std::vector<int> cursor(routing.offsets.begin(), routing.offsets.end() - 1);
    for (std::size_t route_index = 0; route_index < selected.size(); ++route_index)
        routing.occurrences[cursor[selected[route_index]]++] =
            static_cast<int>(route_index);
    for (int expert = 0; expert < kExperts; ++expert)
        if (routing.counts[expert]) routing.group_experts.push_back(expert);
    return routing;
}

struct Split {
    double dequant = 0.0, gemm = 0.0, store = 0.0, wall = 0.0;
};

// The gate/up half of qwen_cpu_moe_rows: decode a 4-row block of both stacks,
// GEMM it against every token routed to that expert, SwiGLU into `activated`.
Split gate_phase(
    std::uint32_t type, const std::vector<std::uint8_t>& gate,
    const std::vector<std::uint8_t>& up, const Routing& routing,
    const std::vector<const float*>& vectors, std::vector<float>& activated,
    int threads
) {
    const std::size_t expert_bytes =
        block_bytes_of(type) * (kHidden / block_elements_of(type)) * kIntermediate;
    const int groups = static_cast<int>(routing.group_experts.size());
    const int blocks = (kIntermediate + kRowBlock - 1) / kRowBlock;
    std::atomic<double> dequant{0.0}, gemm{0.0}, store{0.0};
    const double start = now_seconds();
    dynamic_for(groups * blocks, threads, [&](int task) {
        const int expert = routing.group_experts[task / blocks];
        const int row0 = (task % blocks) * kRowBlock;
        const int mr = std::min(kRowBlock, kIntermediate - row0);
        const int begin = routing.offsets[expert];
        const int count = routing.counts[expert];
        if (count <= 2) return;
        const auto* gate_data = gate.data() + expert * expert_bytes;
        const auto* up_data = up.data() + expert * expert_bytes;
        thread_local std::vector<float> gate_block, up_block, gate_out, up_out;
        gate_block.resize(static_cast<std::size_t>(kRowBlock) * kHidden);
        up_block.resize(static_cast<std::size_t>(kRowBlock) * kHidden);
        gate_out.resize(static_cast<std::size_t>(kRowBlock) * count);
        up_out.resize(static_cast<std::size_t>(kRowBlock) * count);
        const double t0 = now_seconds();
        for (int i = 0; i < mr; ++i) {
            qwen_dequant_row_avx2(
                gate_data, type, kHidden, row0 + i,
                gate_block.data() + static_cast<std::size_t>(i) * kHidden);
            qwen_dequant_row_avx2(
                up_data, type, kHidden, row0 + i,
                up_block.data() + static_cast<std::size_t>(i) * kHidden);
        }
        const double t1 = now_seconds();
        qwen_f32_gemm_rows_avx512(gate_block.data(), mr, &vectors[begin], count,
                                  kHidden, gate_out.data());
        qwen_f32_gemm_rows_avx512(up_block.data(), mr, &vectors[begin], count,
                                  kHidden, up_out.data());
        const double t2 = now_seconds();
        for (int occurrence = 0; occurrence < count; ++occurrence) {
            const int route = routing.occurrences[begin + occurrence];
            float* destination = activated.data() +
                static_cast<std::size_t>(route) * kIntermediate + row0;
            for (int i = 0; i < mr; ++i) {
                const float value =
                    gate_out[static_cast<std::size_t>(i) * count + occurrence];
                const float clipped = std::max(-80.0f, std::min(80.0f, value));
                destination[i] = value / (1.0f + std::exp(-clipped)) *
                    up_out[static_cast<std::size_t>(i) * count + occurrence];
            }
        }
        const double t3 = now_seconds();
        // Relaxed accumulation of thread-seconds, exactly as the runtime's own
        // region counters do it.
        for (auto* slot : {&dequant, &gemm, &store}) {
            const double delta = slot == &dequant ? t1 - t0
                : slot == &gemm ? t2 - t1 : t3 - t2;
            double expected = slot->load(std::memory_order_relaxed);
            while (!slot->compare_exchange_weak(expected, expected + delta,
                                                std::memory_order_relaxed)) {
            }
        }
    });
    return {dequant.load(), gemm.load(), store.load(), now_seconds() - start};
}

void report(const char* label, std::uint32_t type, const Split& split,
            double macs, int threads) {
    std::printf(
        "  %-22s type %2u  wall %7.3f s  %7.1f GMAC/s  %6.1f per thread\n"
        "      dequant %7.3f  gemm %7.3f  store %7.3f  (thread-seconds)"
        "   dequant/gemm %6.2f\n",
        label, type, split.wall, macs / split.wall / 1e9,
        macs / split.wall / 1e9 / threads, split.dequant, split.gemm,
        split.store, split.gemm > 0.0 ? split.dequant / split.gemm : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int threads = argc > 2
        ? std::atoi(argv[2])
        : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    const double sustain = argc > 3 ? std::atof(argv[3]) : 0.0;
    if (rows <= 0 || threads <= 0 || sustain < 0.0) {
        std::fprintf(stderr, "usage: %s [rows] [threads] [sustain-seconds]\n",
                     argv[0]);
        return 2;
    }
    std::mt19937 engine(99);
    const Routing routing = route(rows, engine);

    std::vector<float> input(static_cast<std::size_t>(rows) * kHidden);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (auto& value : input) value = normal(engine);
    std::vector<const float*> vectors(routing.occurrences.size());
    for (std::size_t slot = 0; slot < routing.occurrences.size(); ++slot)
        vectors[slot] = input.data() +
            static_cast<std::size_t>(routing.occurrences[slot] / kTopK) * kHidden;
    std::vector<float> activated(
        static_cast<std::size_t>(rows) * kTopK * kIntermediate, 0.0f);

    // Gate and up only: they are two thirds of the expert FLOPs and the half
    // where qwen4exp's UD mix switches format between layers.
    const double macs = static_cast<double>(rows) * kTopK * 2.0 * kIntermediate *
        kHidden;
    std::printf("%d rows, %d experts, top-%d, %d threads -- %.1f GMAC of gate+up\n",
                rows, kExperts, kTopK, threads, macs / 1e9);
    for (const std::uint32_t type : {std::uint32_t{19}, std::uint32_t{16},
                                     std::uint32_t{17}, std::uint32_t{18}}) {
        const std::size_t expert_bytes = block_bytes_of(type) *
            (kHidden / block_elements_of(type)) * kIntermediate;
        const auto gate = random_packed(expert_bytes * kExperts, type, engine);
        const auto up = random_packed(expert_bytes * kExperts, type, engine);
        const char* label = name_of(type);
        gate_phase(type, gate, up, routing, vectors, activated, threads);  // warm
        report(label, type,
               gate_phase(type, gate, up, routing, vectors, activated, threads),
               macs, threads);
        if (sustain <= 0.0) continue;
        const double until = now_seconds() + sustain;
        int pass = 0;
        while (now_seconds() < until) {
            const Split repeated =
                gate_phase(type, gate, up, routing, vectors, activated, threads);
            std::printf("      sustained pass %2d  %7.1f GMAC/s\n", ++pass,
                        macs / repeated.wall / 1e9);
        }
    }
    return 0;
}
