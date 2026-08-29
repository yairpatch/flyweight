#pragma once

// Lookup surface over the generated host kernel table. Kept separate from the
// shim so the CPU backend can resolve kernels without pulling the CUDA language
// macros (__global__, threadIdx, ...) into its own translation unit.

#include <cstddef>

namespace flyweight::cpu {

// Unpacks a CUDA-style void** argument array and runs one thread of one block.
// The caller sets the launch geometry thread-locals first; see cpu_backend.cpp.
using FlyweightCpuKernel = void (*)(void** arguments);

// Null when the corpus has no kernel of that name.
FlyweightCpuKernel find_kernel(const char* name);

// Same lookup, also reporting whether the kernel can reach a barrier, shuffle,
// or block-wide sort. False means the block can run as a plain loop over thread
// indices with no fibers, which is the difference between ~20us and ~2ms per
// block. Returns false if the name is unknown.
bool find_kernel_entry(const char* name, FlyweightCpuKernel* kernel,
                       bool* cooperative);

std::size_t kernel_count();
const char* kernel_name(std::size_t index);

}  // namespace flyweight::cpu
