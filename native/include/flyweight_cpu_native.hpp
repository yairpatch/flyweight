#pragma once

// Hand-written host kernels and the parity contract they must satisfy.
//
// The emulated corpus (cpu_kernels.cpp) is the reference implementation: it is
// the same CUDA text the GPU runs, so it defines what every kernel means. It is
// far too slow to serve from -- a CUDA block maps onto fibers, and even after
// the barrier-free fast path a cooperative kernel costs the better part of a
// millisecond. Production CPU execution therefore comes from native kernels
// registered here, and the emulated version becomes the oracle they are tested
// against.
//
// A native kernel receives the whole launch rather than one thread of one
// block. That is the point: the CUDA decomposition (one thread per output
// element, per-lane quant block decode) is tuned for coalescing and latency
// hiding and is the wrong shape for a CPU, which wants blocked row-major SIMD
// with the codebook decode amortized across rows. Owning the launch lets a
// native kernel ignore the grid entirely and use whatever loop structure suits.

#include <cstdint>

namespace flyweight::cpu {

// Geometry the runtime asked for. A native kernel may honour or ignore it, so
// long as the memory it writes matches what the emulated kernel would write.
struct Launch {
    std::uint32_t grid_x = 1;
    std::uint32_t grid_y = 1;
    std::uint32_t block_x = 1;
    std::uint32_t shared_bytes = 0;
    std::uint64_t stream = 0;
};

using NativeKernel = void (*)(const Launch& launch, void** arguments);

// Registers a native implementation for a corpus kernel name. Returns false if
// the name is not in the corpus -- a typo would otherwise silently leave the
// slow path in place.
bool register_native_kernel(const char* name, NativeKernel kernel);

// Null when no native implementation is registered.
NativeKernel find_native_kernel(const char* name);

// Distributes `count` items across the launch pool, with the calling thread
// participating. This is the only parallelism primitive native kernels should
// use, so the whole backend shares one thread budget.
void parallel_for(std::uint64_t count, void (*body)(void*, std::uint64_t),
                  void* payload);

// Forces the emulated corpus even where a native kernel exists. The parity
// harness uses this to obtain reference output; nothing else should.
void set_force_emulation(bool force);
bool force_emulation();

// Registration helper: declaring one of these at namespace scope registers the
// kernel before main runs.
struct NativeKernelRegistration {
    NativeKernelRegistration(const char* name, NativeKernel kernel);
};

}  // namespace flyweight::cpu

// Registers `fn` as the native implementation of corpus kernel `name`.
#define FLYWEIGHT_CPU_NATIVE_KERNEL(name, fn)                          \
    static const ::flyweight::cpu::NativeKernelRegistration            \
        flyweight_cpu_registration_##fn(name, &fn)
