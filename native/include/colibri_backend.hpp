#pragma once

// Backend selection for the device interface in colibri_gpu_driver.h.
//
// The runtime is written against one device API and launches compute by kernel
// name. Choosing a backend therefore does not change the layer loop at all --
// it changes where colibri_gpu_* sends the work.
//
// The CPU backend hooks in at two points inside gpu_driver.cpp rather than
// duplicating its ~58 entry points:
//
//   * colibri_gpu_compile fills g_functions with CPU sentinel handles instead
//     of NVRTC-compiled CUfunctions, so every existing wrapper -- including the
//     ones with hand-tuned grid geometry like colibri_gpu_q4k_matvec_transposed
//     -- resolves and dispatches without modification.
//
//   * launch() recognizes those sentinels and forwards to the host kernel
//     registry.
//
// Only the memory, stream, and event primitives need explicit branches, because
// those bottom out in the CUDA driver rather than in launch().

#include <cstdint>

// Matches colibri_gpu_driver.h; the library is built with -fvisibility=hidden,
// so selection has to be exported explicitly for the CLI and the tests.
// COLIBRI_V2_STATIC is for the contract executables, which link these sources
// directly: on Windows they neither import nor export, and dllimport on a
// definition is a hard error rather than a warning.
#if defined(COLIBRI_V2_STATIC)
#  define COLIBRI_BACKEND_API
#elif defined(_WIN32)
#  if defined(COLIBRI_V2_BUILD)
#    define COLIBRI_BACKEND_API __declspec(dllexport)
#  else
#    define COLIBRI_BACKEND_API __declspec(dllimport)
#  endif
#else
#  define COLIBRI_BACKEND_API __attribute__((visibility("default")))
#endif

extern "C" {

enum ColibriBackend {
    kColibriBackendCuda = 0,
    kColibriBackendCpu = 1,
};

// Selects the backend for the process. Must be called before runtime prepare;
// switching afterwards would leave allocations owned by the wrong backend.
// Returns non-zero if the requested backend is unavailable.
COLIBRI_BACKEND_API int colibri_backend_select(int backend);

COLIBRI_BACKEND_API int colibri_backend_active();

// Hot-path predicate; checked at every branch site in gpu_driver.cpp.
COLIBRI_BACKEND_API int colibri_backend_is_cpu();

// Writes the per-kernel launch profile to stderr. No-op unless the process was
// started with COLIBRI_CPU_PROFILE=1. Used to decide which emulated kernels are
// worth hand-writing next.
COLIBRI_BACKEND_API void colibri_cpu_profile_dump();

}  // extern "C"

namespace colibri {

// Sentinel CUfunction values for host kernels. The tag occupies bits no real
// pointer can carry on any supported platform, and the low bits index the
// generated kernel table so launch() can recover the name.
inline constexpr std::uint64_t kCpuFunctionTag = 0xC01B'0000'0000'0000ull;
inline constexpr std::uint64_t kCpuFunctionMask = 0xFFFF'0000'0000'0000ull;

inline bool is_cpu_function(const void* function) {
    const auto value = reinterpret_cast<std::uint64_t>(function);
    return (value & kCpuFunctionMask) == kCpuFunctionTag;
}

inline std::uint64_t cpu_function_index(const void* function) {
    return reinterpret_cast<std::uint64_t>(function) & ~kCpuFunctionMask;
}

inline void* make_cpu_function(std::uint64_t index) {
    return reinterpret_cast<void*>(kCpuFunctionTag | index);
}

}  // namespace colibri
