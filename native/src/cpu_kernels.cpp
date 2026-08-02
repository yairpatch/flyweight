// Host translation unit for the CUDA kernel corpus.
//
// The kernel bodies here are the generated copy of the same text NVRTC compiles
// for the GPU backend (native/tools/generate_cpu_kernels.py). Nothing in this
// file should hand-implement a kernel: if the numerics need to change, they
// change in the CUDA headers and both backends follow.

#include <colibri_cpu_shim.hpp>
#include <colibri_cpu_kernels_api.hpp>

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// cub::BlockRadixSort
// ---------------------------------------------------------------------------

// The corpus uses exactly one cub facility, in the two sampling top-k kernels:
// a block-wide descending sort of ITEMS_PER_THREAD keys per thread in blocked
// arrangement. Emulating the interface is far less code than special-casing
// those kernels out of the generated corpus.
namespace cub {

template <class KeyT, int BLOCK_THREADS, int ITEMS_PER_THREAD, class ValueT>
class BlockRadixSort {
public:
    struct TempStorage {
        KeyT keys[BLOCK_THREADS * ITEMS_PER_THREAD];
        ValueT values[BLOCK_THREADS * ITEMS_PER_THREAD];
    };

    explicit BlockRadixSort(TempStorage& storage) : storage_(storage) {}

    void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                        ValueT (&values)[ITEMS_PER_THREAD]) {
        const unsigned int thread = threadIdx.x;
        const unsigned int base = thread * ITEMS_PER_THREAD;
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            storage_.keys[base + item] = keys[item];
            storage_.values[base + item] = values[item];
        }
        __syncthreads();

        // cub's sort is stable across the block; one lane sorting the whole
        // buffer reproduces that ordering without any cross-lane machinery.
        if (thread == 0) {
            constexpr int total = BLOCK_THREADS * ITEMS_PER_THREAD;
            // Deliberately not stack arrays: at 1024 items these are ~16 KiB,
            // which would dominate the fiber stack budget that the whole
            // backend's throughput is tuned around. thread_local is safe here
            // for the same reason __shared__ is -- one block per OS thread.
            static thread_local int order[total];
            static thread_local KeyT sorted_keys[total];
            static thread_local ValueT sorted_values[total];
            for (int index = 0; index < total; ++index) order[index] = index;
            std::stable_sort(order, order + total,
                             [this](int left, int right) {
                                 return storage_.keys[left] > storage_.keys[right];
                             });
            for (int index = 0; index < total; ++index) {
                sorted_keys[index] = storage_.keys[order[index]];
                sorted_values[index] = storage_.values[order[index]];
            }
            std::memcpy(storage_.keys, sorted_keys, sizeof(sorted_keys));
            std::memcpy(storage_.values, sorted_values, sizeof(sorted_values));
        }
        __syncthreads();

        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            keys[item] = storage_.keys[base + item];
            values[item] = storage_.values[base + item];
        }
        __syncthreads();
    }

private:
    TempStorage& storage_;
};

}  // namespace cub

// ---------------------------------------------------------------------------
// Generated corpus and dispatch table
// ---------------------------------------------------------------------------

// The anonymous namespace is load-bearing. The corpus defines the IQ codebook
// tables (kIq2xxsGrid, kIq4nlValues, ...) at file scope under the same names
// the host quant path already defines in qwen_iq_tables.h, and both land in
// libcolibri_v2. Internal linkage keeps the device-side copies private to this
// translation unit instead of colliding at link time, and does the same for any
// helper the two sides happen to name alike in future.
namespace {

#include "colibri_cpu_kernels.inc"

struct ColibriCpuKernelEntry {
    const char* name;
    void (*launch)(void** arguments);
    // True when the kernel can reach a barrier, shuffle, or block-wide sort.
    // Only these need fiber-per-thread execution; see cpu_backend.cpp.
    bool cooperative;
};

#include "colibri_cpu_kernel_table.inc"

}  // namespace

namespace colibri::cpu {

ColibriCpuKernel find_kernel(const char* name) {
    // The table is generated in corpus order; a linear scan happens once per
    // launch site during warmup and the runtime caches the result.
    for (const auto& entry : kColibriCpuKernels) {
        if (std::strcmp(entry.name, name) == 0) return entry.launch;
    }
    return nullptr;
}

bool find_kernel_entry(const char* name, ColibriCpuKernel* kernel,
                       bool* cooperative) {
    for (const auto& entry : kColibriCpuKernels) {
        if (std::strcmp(entry.name, name) != 0) continue;
        if (kernel != nullptr) *kernel = entry.launch;
        if (cooperative != nullptr) *cooperative = entry.cooperative;
        return true;
    }
    return false;
}

std::size_t kernel_count() {
    return sizeof(kColibriCpuKernels) / sizeof(kColibriCpuKernels[0]);
}

const char* kernel_name(std::size_t index) {
    return index < kernel_count() ? kColibriCpuKernels[index].name : nullptr;
}

}  // namespace colibri::cpu
