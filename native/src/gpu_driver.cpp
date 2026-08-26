// Optional CUDA driving from the native library. libcuda and libnvrtc are
// dlopen'd at runtime, so the library builds and runs without any CUDA
// dependency; the GPU entry points simply report unavailability. The driver
// retains the device's primary context (the same one CuPy's runtime API
// uses) and launches on the legacy default stream, so device pointers taken
// from CuPy arrays are valid here and ordering with CuPy-issued work is
// automatic.
#include "colibri_gpu_driver.h"

#include <colibri_backend.hpp>
#include <colibri_cpu_backend.hpp>

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUevent = void*;
using CUdeviceptr = unsigned long long;

using nvrtcResult = int;
using nvrtcProgram = void*;
using cublasStatus_t = int;
using cublasHandle_t = void*;
using cudaDataType_t = int;
using cublasComputeType_t = int;
using cublasOperation_t = int;
using cublasGemmAlgo_t = int;
using cublasLtHandle_t = void*;
using cublasLtMatmulDesc_t = void*;
using cublasLtMatrixLayout_t = void*;
using cublasLtMatmulPreference_t = void*;
struct CublasLtMatmulAlgo { std::uint64_t data[8]; };
struct CublasLtHeuristicResult {
    CublasLtMatmulAlgo algo;
    size_t workspace_size;
    cublasStatus_t state;
    float waves_count;
    int reserved[4];
};

struct CudaApi {
    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxSetFlags)(CUdevice, unsigned int) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxSetFlags)(unsigned int) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuModuleLoadDataEx)(
        CUmodule*, const void*, unsigned int, int*, void**
    ) = nullptr;
    CUresult (*cuModuleGetFunction)(
        CUfunction*, CUmodule, const char*
    ) = nullptr;
    CUresult (*cuLaunchKernel)(
        CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, CUstream, void**, void**
    ) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemHostAlloc)(void**, size_t, unsigned int) = nullptr;
    CUresult (*cuMemFreeHost)(void*) = nullptr;
    CUresult (*cuMemHostRegister)(void*, size_t, unsigned int) = nullptr;
    CUresult (*cuMemHostUnregister)(void*) = nullptr;
    CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuStreamCreate)(CUstream*, unsigned int) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream) = nullptr;
    CUresult (*cuEventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventDestroy)(CUevent) = nullptr;
    CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent) = nullptr;
    CUresult (*cuStreamBeginCapture)(CUstream, int) = nullptr;
    CUresult (*cuStreamEndCapture)(CUstream, void**) = nullptr;
    CUresult (*cuGraphInstantiateWithFlags)(
        void**, void*, unsigned long long
    ) = nullptr;
    CUresult (*cuGraphLaunch)(void*, CUstream) = nullptr;
    CUresult (*cuGraphDestroy)(void*) = nullptr;
    CUresult (*cuGraphExecDestroy)(void*) = nullptr;

    nvrtcResult (*nvrtcCreateProgram)(
        nvrtcProgram*, const char*, const char*, int, const char* const*,
        const char* const*
    ) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(
        nvrtcProgram, int, const char* const*
    ) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;

    bool loaded = false;
};

CudaApi g_api;
struct CublasApi {
    cublasStatus_t (*create)(cublasHandle_t*) = nullptr;
    cublasStatus_t (*destroy)(cublasHandle_t) = nullptr;
    cublasStatus_t (*set_stream)(cublasHandle_t, CUstream) = nullptr;
    cublasStatus_t (*gemm_strided_batched_ex)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t,
        int, int, int, const void*, const void*, cudaDataType_t, int,
        long long, const void*, cudaDataType_t, int, long long,
        const void*, void*, cudaDataType_t, int, long long, int,
        cublasComputeType_t, cublasGemmAlgo_t
    ) = nullptr;
    void* library = nullptr;
    bool attempted = false;
};
CublasApi g_cublas;
cublasHandle_t g_cublas_handle = nullptr;
struct CublasLtApi {
    cublasStatus_t (*create)(cublasLtHandle_t*) = nullptr;
    cublasStatus_t (*destroy)(cublasLtHandle_t) = nullptr;
    cublasStatus_t (*matmul_desc_create)(
        cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t) = nullptr;
    cublasStatus_t (*matmul_desc_destroy)(cublasLtMatmulDesc_t) = nullptr;
    cublasStatus_t (*matmul_desc_set)(
        cublasLtMatmulDesc_t, int, const void*, size_t) = nullptr;
    cublasStatus_t (*layout_create)(
        cublasLtMatrixLayout_t*, cudaDataType_t, std::uint64_t,
        std::uint64_t, std::int64_t) = nullptr;
    cublasStatus_t (*layout_destroy)(cublasLtMatrixLayout_t) = nullptr;
    cublasStatus_t (*preference_create)(cublasLtMatmulPreference_t*) = nullptr;
    cublasStatus_t (*preference_destroy)(cublasLtMatmulPreference_t) = nullptr;
    cublasStatus_t (*preference_set)(
        cublasLtMatmulPreference_t, int, const void*, size_t) = nullptr;
    cublasStatus_t (*heuristic)(
        cublasLtHandle_t, cublasLtMatmulDesc_t,
        cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
        cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
        cublasLtMatmulPreference_t, int, CublasLtHeuristicResult*, int*) = nullptr;
    cublasStatus_t (*matmul)(
        cublasLtHandle_t, cublasLtMatmulDesc_t,
        const void*, const void*, cublasLtMatrixLayout_t,
        const void*, cublasLtMatrixLayout_t, const void*,
        const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
        const CublasLtMatmulAlgo*, void*, size_t, CUstream) = nullptr;
    void* library = nullptr;
    bool attempted = false;
};
CublasLtApi g_cublas_lt;
cublasLtHandle_t g_cublas_lt_handle = nullptr;
struct Nvfp4Scratch {
    CUdeviceptr weight_values = 0, weight_scales = 0;
    CUdeviceptr input_values = 0, input_scales = 0;
    CUdeviceptr projected = 0, expert_pointers = 0;
    size_t weight_values_bytes = 0, weight_scales_bytes = 0;
    size_t input_values_bytes = 0, input_scales_bytes = 0;
    size_t projected_bytes = 0, expert_pointers_bytes = 0;
    CUstream stream = nullptr;
    // Orders cross-stream reuse of this single-buffered scratch without a
    // host synchronization; see nvfp4_scratch_switch_stream.
    CUevent handoff = nullptr;
};
Nvfp4Scratch g_nvfp4_scratch;
struct Nvfp4LtPlan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr, d_layout = nullptr;
    CublasLtMatmulAlgo algo{};
};
std::unordered_map<std::uint64_t, Nvfp4LtPlan> g_nvfp4_plans;
bool g_nvfp4_validation_done = false;
struct Nvfp4MoeProfile {
    CUevent events[9]{};
    bool initialized = false;
    std::uint64_t calls = 0;
    double milliseconds[8]{};
};
Nvfp4MoeProfile g_nvfp4_moe_profile;
std::mutex g_cublas_mutex;
CUcontext g_context = nullptr;
CUmodule g_module = nullptr;
CUstream g_stream = nullptr;

constexpr int kAttributeComputeMajor = 75;
constexpr int kAttributeComputeMinor = 76;
constexpr unsigned int kThreadsPerBlock = 256;

struct Kernels {
    CUfunction rms_norm = nullptr;
    CUfunction q4_matvec = nullptr;
    CUfunction bf16_matvec = nullptr;
    CUfunction delta_conv = nullptr;
    CUfunction delta_conv_step = nullptr;
    CUfunction delta_recurrent = nullptr;
    CUfunction scaled_add = nullptr;
    CUfunction q4_batched = nullptr;
    CUfunction q4_silu = nullptr;
    CUfunction q4_weighted = nullptr;
    CUfunction attention = nullptr;
    CUfunction kv_append = nullptr;
    CUfunction q8_matvec_transposed = nullptr;
    CUfunction route_topk = nullptr;
    CUfunction sampling_block_topk_logits = nullptr;
    CUfunction sampling_block_topk_pairs = nullptr;
    CUfunction q5_grouped_swiglu = nullptr;
    CUfunction q4k_grouped_swiglu = nullptr;
    CUfunction q4k_grouped_accumulate = nullptr;
    CUfunction q5k_grouped_accumulate = nullptr;
    CUfunction q6_grouped_accumulate = nullptr;
    CUfunction q8_grouped_accumulate = nullptr;
    CUfunction nvfp4_grouped_swiglu = nullptr;
    CUfunction nvfp4_grouped_accumulate = nullptr;
    CUfunction nvfp4_grouped_swiglu_tiled = nullptr;
    CUfunction nvfp4_grouped_accumulate_tiled = nullptr;
};

Kernels g_kernels;
std::unordered_map<std::string, CUfunction> g_functions;

template <typename T>
bool load_symbol(void* library, const char* name, T& target) {
#if defined(_WIN32)
    target = reinterpret_cast<T>(
        GetProcAddress(static_cast<HMODULE>(library), name));
    return target != nullptr;
#else
    target = reinterpret_cast<T>(dlsym(library, name));
    return target != nullptr;
#endif
}

#if defined(_WIN32)
// NVRTC ships with the CUDA Toolkit (not the driver), so its bin directory is
// usually absent from the default DLL search path. Try bare names first, then
// resolve through CUDA_PATH.
void* win_load_nvrtc() {
    const wchar_t* names[] = {
        L"nvrtc64_130_0.dll", L"nvrtc64_120_0.dll", L"nvrtc64_112_0.dll",
        L"nvrtc64_111_0.dll", L"nvrtc64_110_0.dll",
    };
    // Prefer a full path from CUDA_PATH with LOAD_WITH_ALTERED_SEARCH_PATH so the
    // loader resolves nvrtc's own companion (nvrtc-builtins64_*.dll) from the
    // toolkit bin directory rather than the default search path.
    wchar_t base[4096];
    DWORD length = GetEnvironmentVariableW(L"CUDA_PATH", base, 4096);
    if (length > 0 && length < 4096) {
        // nvrtc lazily LoadLibrary's its companion nvrtc-builtins64_*.dll by bare
        // name at compile time. Python calls SetDefaultDllDirectories, dropping
        // PATH from the search, so register the toolkit bin directories as user
        // search dirs (searched under LOAD_LIBRARY_SEARCH_USER_DIRS) to make the
        // builtins resolvable.
        const wchar_t* subdirs[] = {L"\\bin\\x64", L"\\bin"};
        for (const wchar_t* subdir : subdirs)
            AddDllDirectory((std::wstring(base) + subdir).c_str());
        const wchar_t* files[] = {L"\\bin\\x64\\", L"\\bin\\"};
        for (const wchar_t* subdir : files)
            for (const wchar_t* name : names) {
                std::wstring path = std::wstring(base) + subdir + name;
                HMODULE handle = LoadLibraryExW(
                    path.c_str(), nullptr,
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
                if (handle) return handle;
            }
    }
    for (const wchar_t* name : names) {
        HMODULE handle = LoadLibraryW(name);
        if (handle) return handle;
    }
    return nullptr;
}
#endif

bool load_apis() {
    if (g_api.loaded) {
        return true;
    }
    void* cuda = nullptr;
    void* nvrtc = nullptr;
#if defined(_WIN32)
    cuda = reinterpret_cast<void*>(LoadLibraryW(L"nvcuda.dll"));
    nvrtc = win_load_nvrtc();
#else
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (cuda == nullptr) {
        cuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    }
    nvrtc = dlopen("libnvrtc.so", RTLD_NOW | RTLD_GLOBAL);
#endif
    if (cuda == nullptr || nvrtc == nullptr) {
        return false;
    }
    bool ok = true;
    ok &= load_symbol(cuda, "cuInit", g_api.cuInit);
    ok &= load_symbol(
        cuda, "cuDevicePrimaryCtxRetain", g_api.cuDevicePrimaryCtxRetain
    );
    load_symbol(
        cuda, "cuDevicePrimaryCtxSetFlags", g_api.cuDevicePrimaryCtxSetFlags
    );
    ok &= load_symbol(cuda, "cuCtxSetCurrent", g_api.cuCtxSetCurrent);
    load_symbol(cuda, "cuCtxSetFlags", g_api.cuCtxSetFlags);
    ok &= load_symbol(
        cuda, "cuDeviceGetAttribute", g_api.cuDeviceGetAttribute
    );
    ok &= load_symbol(cuda, "cuModuleLoadDataEx", g_api.cuModuleLoadDataEx);
    ok &= load_symbol(cuda, "cuModuleGetFunction", g_api.cuModuleGetFunction);
    ok &= load_symbol(cuda, "cuLaunchKernel", g_api.cuLaunchKernel);
    ok &= load_symbol(cuda, "cuMemcpyDtoH_v2", g_api.cuMemcpyDtoH);
    ok &= load_symbol(cuda, "cuMemcpyHtoD_v2", g_api.cuMemcpyHtoD);
    ok &= load_symbol(cuda, "cuMemcpyDtoHAsync_v2", g_api.cuMemcpyDtoHAsync);
    ok &= load_symbol(cuda, "cuMemcpyHtoDAsync_v2", g_api.cuMemcpyHtoDAsync);
    ok &= load_symbol(cuda, "cuMemAlloc_v2", g_api.cuMemAlloc);
    ok &= load_symbol(cuda, "cuMemFree_v2", g_api.cuMemFree);
    ok &= load_symbol(cuda, "cuMemHostAlloc", g_api.cuMemHostAlloc);
    ok &= load_symbol(cuda, "cuMemFreeHost", g_api.cuMemFreeHost);
    // Optional: DMA page-ins straight from a registered mmap (not fatal if absent).
    load_symbol(cuda, "cuMemHostRegister_v2", g_api.cuMemHostRegister);
    load_symbol(cuda, "cuMemHostUnregister", g_api.cuMemHostUnregister);
    ok &= load_symbol(cuda, "cuMemsetD8Async", g_api.cuMemsetD8Async);
    ok &= load_symbol(cuda, "cuStreamSynchronize", g_api.cuStreamSynchronize);
    ok &= load_symbol(cuda, "cuStreamCreate", g_api.cuStreamCreate);
    ok &= load_symbol(cuda, "cuStreamDestroy_v2", g_api.cuStreamDestroy);
    ok &= load_symbol(cuda, "cuEventCreate", g_api.cuEventCreate);
    ok &= load_symbol(cuda, "cuEventRecord", g_api.cuEventRecord);
    ok &= load_symbol(cuda, "cuStreamWaitEvent", g_api.cuStreamWaitEvent);
    ok &= load_symbol(cuda, "cuEventSynchronize", g_api.cuEventSynchronize);
    ok &= load_symbol(cuda, "cuEventDestroy_v2", g_api.cuEventDestroy);
    ok &= load_symbol(cuda, "cuEventElapsedTime", g_api.cuEventElapsedTime);
    ok &= load_symbol(
        cuda, "cuStreamBeginCapture_v2", g_api.cuStreamBeginCapture
    );
    ok &= load_symbol(cuda, "cuStreamEndCapture", g_api.cuStreamEndCapture);
    ok &= load_symbol(
        cuda, "cuGraphInstantiateWithFlags", g_api.cuGraphInstantiateWithFlags
    );
    ok &= load_symbol(cuda, "cuGraphLaunch", g_api.cuGraphLaunch);
    ok &= load_symbol(cuda, "cuGraphDestroy", g_api.cuGraphDestroy);
    ok &= load_symbol(cuda, "cuGraphExecDestroy", g_api.cuGraphExecDestroy);
    ok &= load_symbol(nvrtc, "nvrtcCreateProgram", g_api.nvrtcCreateProgram);
    ok &= load_symbol(nvrtc, "nvrtcCompileProgram", g_api.nvrtcCompileProgram);
    ok &= load_symbol(nvrtc, "nvrtcGetPTXSize", g_api.nvrtcGetPTXSize);
    ok &= load_symbol(nvrtc, "nvrtcGetPTX", g_api.nvrtcGetPTX);
    ok &= load_symbol(
        nvrtc, "nvrtcGetProgramLogSize", g_api.nvrtcGetProgramLogSize
    );
    ok &= load_symbol(nvrtc, "nvrtcGetProgramLog", g_api.nvrtcGetProgramLog);
    ok &= load_symbol(nvrtc, "nvrtcDestroyProgram", g_api.nvrtcDestroyProgram);
    g_api.loaded = ok;
    return ok;
}

bool load_cublas() {
    if (g_cublas_handle != nullptr) return true;
    if (g_cublas.attempted) return false;
    g_cublas.attempted = true;
#if defined(_WIN32)
    const wchar_t* names[] = {
        L"cublas64_13.dll", L"cublas64_12.dll", L"cublas64_11.dll",
    };
    for (const wchar_t* name : names) {
        g_cublas.library = reinterpret_cast<void*>(LoadLibraryW(name));
        if (g_cublas.library != nullptr) break;
    }
#else
    const char* names[] = {
        "libcublas.so.13", "libcublas.so.12", "libcublas.so.11",
        "libcublas.so",
    };
    for (const char* name : names) {
        g_cublas.library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (g_cublas.library != nullptr) break;
    }
#endif
    if (g_cublas.library == nullptr) return false;
    bool ok = true;
    ok &= load_symbol(g_cublas.library, "cublasCreate_v2", g_cublas.create);
    ok &= load_symbol(g_cublas.library, "cublasDestroy_v2", g_cublas.destroy);
    ok &= load_symbol(
        g_cublas.library, "cublasSetStream_v2", g_cublas.set_stream
    );
    ok &= load_symbol(
        g_cublas.library, "cublasGemmStridedBatchedEx",
        g_cublas.gemm_strided_batched_ex
    );
    if (!ok || g_cublas.create(&g_cublas_handle) != 0) {
        g_cublas_handle = nullptr;
        return false;
    }
    return true;
}

bool load_cublas_lt() {
    if (g_cublas_lt_handle != nullptr) return true;
    if (g_cublas_lt.attempted) return false;
    g_cublas_lt.attempted = true;
#if defined(_WIN32)
    const wchar_t* names[] = {
        L"cublasLt64_13.dll", L"cublasLt64_12.dll", L"cublasLt64_11.dll",
    };
    for (const wchar_t* name : names) {
        g_cublas_lt.library = reinterpret_cast<void*>(LoadLibraryW(name));
        if (g_cublas_lt.library != nullptr) break;
    }
    if (g_cublas_lt.library == nullptr) {
        wchar_t base[4096];
        const DWORD length = GetEnvironmentVariableW(L"CUDA_PATH", base, 4096);
        if (length > 0 && length < 4096) {
            for (const wchar_t* subdir : {L"\\bin\\x64\\", L"\\bin\\"}) {
                for (const wchar_t* name : names) {
                    const std::wstring path =
                        std::wstring(base) + subdir + name;
                    HMODULE handle = LoadLibraryExW(
                        path.c_str(), nullptr,
                        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
                    if (handle) {
                        g_cublas_lt.library =
                            reinterpret_cast<void*>(handle);
                        break;
                    }
                }
                if (g_cublas_lt.library != nullptr) break;
            }
        }
    }
#else
    const char* names[] = {
        "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11",
        "libcublasLt.so",
    };
    for (const char* name : names) {
        g_cublas_lt.library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (g_cublas_lt.library != nullptr) break;
    }
#endif
    if (g_cublas_lt.library == nullptr) return false;
    bool ok = true;
    ok &= load_symbol(g_cublas_lt.library, "cublasLtCreate", g_cublas_lt.create);
    ok &= load_symbol(g_cublas_lt.library, "cublasLtDestroy", g_cublas_lt.destroy);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmulDescCreate",
        g_cublas_lt.matmul_desc_create);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmulDescDestroy",
        g_cublas_lt.matmul_desc_destroy);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmulDescSetAttribute",
        g_cublas_lt.matmul_desc_set);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatrixLayoutCreate",
        g_cublas_lt.layout_create);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatrixLayoutDestroy",
        g_cublas_lt.layout_destroy);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmulPreferenceCreate",
        g_cublas_lt.preference_create);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmulPreferenceDestroy",
        g_cublas_lt.preference_destroy);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmulPreferenceSetAttribute",
        g_cublas_lt.preference_set);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmulAlgoGetHeuristic",
        g_cublas_lt.heuristic);
    ok &= load_symbol(
        g_cublas_lt.library, "cublasLtMatmul", g_cublas_lt.matmul);
    if (!ok || g_cublas_lt.create(&g_cublas_lt_handle) != 0) {
        g_cublas_lt_handle = nullptr;
        return false;
    }
    return true;
}

// Phase profiling under COLIBRI_SEG_DEBUG: 0=start 1=mixer 2=route 3=copies
// 4=experts 5=writeback.
double g_phase_totals[6] = {};
std::chrono::steady_clock::time_point g_phase_last{};

void phase_mark(int phase) {
    static const bool enabled =
        std::getenv("COLIBRI_SEG_DEBUG") != nullptr;
    if (!enabled) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (phase > 0) {
        g_phase_totals[phase] +=
            std::chrono::duration<double, std::milli>(now - g_phase_last).count();
    }
    g_phase_last = now;
}

int launch(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int block_x,
    void** args,
    unsigned int shared_bytes = 0,
    CUstream stream = nullptr
) {
    // Every wrapper in this file bottoms out here, so recognizing a CPU
    // sentinel is all it takes for the host backend to inherit the whole set --
    // including the hand-tuned grid geometry each wrapper computes.
    if (colibri::is_cpu_function(function)) {
        const char* name = colibri_cpu_kernel_name(
            colibri::cpu_function_index(function));
        if (name == nullptr) return -1;
        return colibri_cpu_launch_named(
            name, grid_x, grid_y, block_x, shared_bytes,
            reinterpret_cast<std::uint64_t>(stream), args);
    }
    return g_api.cuLaunchKernel(
        function,
        grid_x, grid_y, 1,
        block_x, 1, 1,
        shared_bytes, stream, args, nullptr
    );
}

// Enqueue one DeltaNet layer's GPU chain (mixer through router logits) on
// the given stream. Every pointer is static per layer, which is what makes
// the sequence graph-capturable.
int enqueue_layer(
    const ColibriDeltaParams* params,
    const ColibriDeltaLayer& layer,
    CUstream stream
) {
    std::int32_t hidden_size = params->hidden_size;
    std::int32_t conv_dim = params->conv_dim;
    std::int32_t value_dim = params->value_dim;
    float eps = params->rms_norm_eps;
    int one = 1;
    int tokens = 1;
    float unit = 1.0f;
    std::uint64_t hidden = params->hidden;
    std::uint64_t normalized = params->normalized;
    std::uint64_t projected = params->projected;
    std::uint64_t gates = params->gates;
    std::uint64_t convolved = params->convolved;
    std::uint64_t cores = params->cores;
    std::uint64_t mixed = params->mixed;
    std::uint64_t moe_normalized = params->moe_normalized;
    std::uint64_t router_logits = params->router_logits;
    std::uint64_t input_norm = layer.input_norm;
    std::uint64_t qz_packed = layer.qz_packed;
    std::uint64_t qz_scales = layer.qz_scales;
    std::uint64_t ba_weights = layer.ba_weights;
    std::uint64_t out_packed = layer.out_proj_packed;
    std::uint64_t out_scales = layer.out_proj_scales;
    std::uint64_t conv_weights = layer.conv_weights;
    std::uint64_t conv_state = layer.conv_state;
    std::uint64_t recurrent_state = layer.recurrent_state;
    std::uint64_t a_log = layer.a_log;
    std::uint64_t dt_bias = layer.dt_bias;
    std::uint64_t delta_norm = layer.delta_norm;
    std::uint64_t post_norm = layer.post_attention_norm;
    std::uint64_t router_gate = layer.router_gate;
    std::int32_t qz_rows = params->qz_rows;
    std::int32_t ba_rows = params->ba_rows;
    std::int32_t conv_kernel = params->conv_kernel;
    std::int32_t key_heads = params->num_key_heads;
    std::int32_t value_heads = params->num_value_heads;
    std::int32_t key_head_dim = params->key_head_dim;
    std::int32_t value_head_dim = params->value_head_dim;
    std::int32_t router_rows = params->num_experts + 1;
    const std::int32_t add_blocks =
        (hidden_size + static_cast<std::int32_t>(kThreadsPerBlock) - 1)
        / static_cast<std::int32_t>(kThreadsPerBlock);
    {
        void* args[] = {
            &hidden, &input_norm, &normalized, &hidden_size, &eps, &one,
        };
        if (launch(
                g_kernels.rms_norm, 1, 1, kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        void* args[] = {
            &qz_packed, &qz_scales, &normalized, &projected,
            &qz_rows, &hidden_size,
        };
        if (launch(
                g_kernels.q4_matvec,
                static_cast<unsigned int>(qz_rows), 1,
                kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        void* args[] = {
            &ba_weights, &normalized, &gates, &ba_rows, &hidden_size,
        };
        if (launch(
                g_kernels.bf16_matvec,
                static_cast<unsigned int>(ba_rows), 1,
                kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        void* args[] = {
            &projected, &conv_weights, &conv_state, &convolved,
            &conv_dim, &conv_kernel,
        };
        const unsigned int conv_blocks =
            (static_cast<unsigned int>(conv_dim) + kThreadsPerBlock - 1)
            / kThreadsPerBlock;
        if (launch(
                g_kernels.delta_conv_step,
                conv_blocks, 1, kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        std::uint64_t z = projected
            + static_cast<std::uint64_t>(conv_dim) * sizeof(float);
        std::uint64_t beta = gates;
        std::uint64_t decay = gates
            + static_cast<std::uint64_t>(value_heads) * sizeof(float);
        void* args[] = {
            &convolved, &z, &beta, &decay, &a_log, &dt_bias, &delta_norm,
            &recurrent_state, &cores, &tokens, &key_heads, &value_heads,
            &key_head_dim, &value_head_dim, &eps,
        };
        if (launch(
                g_kernels.delta_recurrent,
                static_cast<unsigned int>(value_heads), 1,
                kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        void* args[] = {
            &out_packed, &out_scales, &cores, &mixed,
            &hidden_size, &value_dim,
        };
        if (launch(
                g_kernels.q4_matvec,
                static_cast<unsigned int>(hidden_size), 1,
                kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        void* args[] = {&mixed, &hidden, &unit, &hidden_size};
        if (launch(
                g_kernels.scaled_add,
                static_cast<unsigned int>(add_blocks), 1,
                kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        void* args[] = {
            &mixed, &post_norm, &moe_normalized, &hidden_size, &eps, &one,
        };
        if (launch(
                g_kernels.rms_norm, 1, 1, kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    {
        void* args[] = {
            &router_gate, &moe_normalized, &router_logits,
            &router_rows, &hidden_size,
        };
        if (launch(
                g_kernels.bf16_matvec,
                static_cast<unsigned int>(router_rows), 1,
                kThreadsPerBlock, args, 0, stream
            ) != 0) {
            return -3;
        }
    }
    return 0;
}

}  // namespace

extern "C" int colibri_gpu_available() {
    if (colibri_backend_is_cpu()) return colibri_cpu_backend_available();
    return load_apis() ? 1 : 0;
}

extern "C" int colibri_gpu_init(std::int32_t device) {
    // No driver to load and no context to retain; the host backend is ready as
    // soon as it is selected.
    if (colibri_backend_is_cpu()) return 0;
    if (!load_apis()) {
        return -1;
    }
    if (g_api.cuInit(0) != 0) {
        return -2;
    }
    // CUDA's default AUTO scheduling actively spins a host core while a stream
    // is synchronized. Decode has a host/device boundary at every routed MoE
    // layer, so that spin can heat a shared-power laptop enough to force the
    // GPU into a much lower firmware power state. Blocking synchronization
    // preserves the same ordering while putting the waiting thread to sleep.
    // Keep an opt-out for latency-sensitive systems with independent cooling.
    constexpr unsigned int kCtxSchedBlockingSync = 0x04;
    const char* spin_wait = std::getenv("COLIBRI_CUDA_SPIN_WAIT");
    const bool blocking_sync = !spin_wait || spin_wait[0] != '1';
    if (blocking_sync && g_api.cuDevicePrimaryCtxSetFlags != nullptr) {
        // This can report PRIMARY_CONTEXT_ACTIVE when another CUDA consumer
        // initialized first. cuCtxSetFlags below handles that case on drivers
        // which expose the current-context API.
        (void)g_api.cuDevicePrimaryCtxSetFlags(
            device, kCtxSchedBlockingSync
        );
    }
    if (g_context == nullptr) {
        // Retain exactly once per process. Every runtime prepare comes through
        // here, and an unbalanced retain per open kept the primary context
        // refcount climbing with nothing ever releasing it.
        if (g_api.cuDevicePrimaryCtxRetain(&g_context, device) != 0) {
            return -3;
        }
    }
    if (g_api.cuCtxSetCurrent(g_context) != 0) {
        return -4;
    }
    if (blocking_sync && g_api.cuCtxSetFlags != nullptr)
        (void)g_api.cuCtxSetFlags(kCtxSchedBlockingSync);
    return 0;
}

struct Entry {
    const char* name;
    CUfunction* slot;
};
// Kernels the driver holds direct handles to, shared by both backends so the
// CUDA and CPU paths can never disagree about which names are required.
const Entry kNamedKernels[] = {
    {"rms_norm", &g_kernels.rms_norm},
    {"q4_matvec", &g_kernels.q4_matvec},
    {"bf16_matvec", &g_kernels.bf16_matvec},
    {"delta_conv_sequence", &g_kernels.delta_conv},
    {"delta_conv_step", &g_kernels.delta_conv_step},
    {"delta_recurrent_sequence", &g_kernels.delta_recurrent},
    {"scaled_add", &g_kernels.scaled_add},
    {"q4_batched_matvec", &g_kernels.q4_batched},
    {"q4_silu_batched", &g_kernels.q4_silu},
    {"q4_batched_weighted_matvec", &g_kernels.q4_weighted},
    {"kv_attention", &g_kernels.attention},
    {"kv_append", &g_kernels.kv_append},
    {"q8_matvec_transposed_warp", &g_kernels.q8_matvec_transposed},
    {"route_topk", &g_kernels.route_topk},
    {"sampling_block_topk_logits", &g_kernels.sampling_block_topk_logits},
    {"sampling_block_topk_pairs", &g_kernels.sampling_block_topk_pairs},
    {"q5k_grouped_swiglu", &g_kernels.q5_grouped_swiglu},
    {"q4k_grouped_swiglu", &g_kernels.q4k_grouped_swiglu},
    {"q4k_grouped_accumulate", &g_kernels.q4k_grouped_accumulate},
    {"q5k_grouped_accumulate", &g_kernels.q5k_grouped_accumulate},
    {"q6k_grouped_accumulate", &g_kernels.q6_grouped_accumulate},
    {"q8_grouped_accumulate", &g_kernels.q8_grouped_accumulate},
    {"nvfp4_grouped_swiglu", &g_kernels.nvfp4_grouped_swiglu},
    {"nvfp4_grouped_accumulate", &g_kernels.nvfp4_grouped_accumulate},
    {"nvfp4_grouped_swiglu_tiled", &g_kernels.nvfp4_grouped_swiglu_tiled},
    {"nvfp4_grouped_accumulate_tiled", &g_kernels.nvfp4_grouped_accumulate_tiled},
};

extern "C" int colibri_gpu_compile(
    const char* source,
    const char* const* options,
    std::int32_t option_count,
    std::int32_t device,
    char* log_buffer,
    std::int32_t log_capacity
) {
    // CPU mode has no source to compile: the host kernels were compiled into
    // this library. Publishing sentinel handles under the same names is what
    // makes every wrapper below work unchanged.
    if (colibri_backend_is_cpu()) {
        g_functions.clear();
        const int total = colibri_cpu_backend_kernel_count();
        for (int index = 0; index < total; ++index) {
            const char* name = colibri_cpu_kernel_name(
                static_cast<std::uint64_t>(index));
            if (name == nullptr) continue;
            g_functions[name] = reinterpret_cast<CUfunction>(
                colibri::make_cpu_function(static_cast<std::uint64_t>(index)));
        }
        for (const Entry& entry : kNamedKernels) {
            const auto found = g_functions.find(entry.name);
            if (found == g_functions.end()) {
                // A name the driver dereferences directly but the corpus does
                // not define would fault at launch; fail here instead.
                if (log_buffer != nullptr && log_capacity > 0) {
                    std::snprintf(log_buffer, log_capacity,
                                  "CPU backend is missing kernel: %s", entry.name);
                }
                return -5;
            }
            *entry.slot = found->second;
        }
        return 0;
    }
    if (!g_api.loaded || g_context == nullptr) {
        return -1;
    }
    static std::mutex compile_mutex;
    std::lock_guard<std::mutex> compile_lock(compile_mutex);
    int major = 0;
    int minor = 0;
    g_api.cuDeviceGetAttribute(&major, kAttributeComputeMajor, device);
    g_api.cuDeviceGetAttribute(&minor, kAttributeComputeMinor, device);
    char arch[64];
    std::snprintf(
        arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor
    );
    std::vector<const char*> all_options;
    all_options.push_back(arch);
    // nvrtc has no default header search path for the CUDA toolkit headers
    // (cuda_fp16.h etc.). Point it at CUDA_PATH\include so the kernels compile.
    std::vector<std::string> include_flags;
#if defined(_WIN32)
    {
        wchar_t base[4096];
        DWORD length = GetEnvironmentVariableW(L"CUDA_PATH", base, 4096);
        if (length > 0 && length < 4096) {
            std::wstring winclude = std::wstring(base) + L"\\include";
            int need = WideCharToMultiByte(
                CP_UTF8, 0, winclude.c_str(), -1, nullptr, 0, nullptr, nullptr);
            if (need > 1) {
                std::string include(static_cast<size_t>(need - 1), '\0');
                WideCharToMultiByte(
                    CP_UTF8, 0, winclude.c_str(), -1, include.data(), need,
                    nullptr, nullptr);
                include_flags.push_back("-I" + include);
                include_flags.push_back("-I" + include + "/cccl");
            }
        }
    }
#else
    if (const char* cuda_path = std::getenv("CUDA_PATH")) {
        const std::string include = std::string(cuda_path) + "/include";
        include_flags.push_back("-I" + include);
        include_flags.push_back("-I" + include + "/cccl");
    }
#endif
    for (const auto& include_flag : include_flags) {
        all_options.push_back(include_flag.c_str());
    }
    for (std::int32_t index = 0; index < option_count; ++index) {
        all_options.push_back(options[index]);
    }
    // One loaded module per distinct (options, source) pair, kept for the
    // life of the process. Reopening a model used to recompile and overwrite
    // g_module while the previous module stayed loaded in the context -- a
    // VRAM leak per reopen, and NVRTC time besides. Unloading old modules is
    // not an option: g_functions keeps entries resolved from earlier corpora
    // (another architecture's kernels), and those must stay launchable. The
    // cache makes the loaded set bounded by the number of distinct corpora.
    static std::unordered_map<std::string, CUmodule> module_cache;
    std::string cache_key;
    cache_key.reserve(std::strlen(source) + 256);
    for (const char* flag : all_options) {
        cache_key += flag;
        cache_key += '\x1f';
    }
    cache_key += source;
    const auto cached = module_cache.find(cache_key);
    if (cached != module_cache.end()) {
        g_module = cached->second;
    } else {
        nvrtcProgram program = nullptr;
        if (g_api.nvrtcCreateProgram(
                &program, source, "colibri_kernels.cu", 0, nullptr, nullptr
            ) != 0) {
            return -2;
        }
        const nvrtcResult compiled = g_api.nvrtcCompileProgram(
            program, static_cast<int>(all_options.size()), all_options.data()
        );
        if (log_buffer != nullptr && log_capacity > 0) {
            size_t log_size = 0;
            g_api.nvrtcGetProgramLogSize(program, &log_size);
            std::vector<char> log(log_size + 1, '\0');
            g_api.nvrtcGetProgramLog(program, log.data());
            std::strncpy(log_buffer, log.data(), log_capacity - 1);
            log_buffer[log_capacity - 1] = '\0';
        }
        if (compiled != 0) {
            g_api.nvrtcDestroyProgram(&program);
            return -3;
        }
        size_t ptx_size = 0;
        if (g_api.nvrtcGetPTXSize(program, &ptx_size) != 0 || ptx_size == 0) {
            g_api.nvrtcDestroyProgram(&program);
            return -3;
        }
        std::vector<char> ptx(ptx_size);
        if (g_api.nvrtcGetPTX(program, ptx.data()) != 0) {
            g_api.nvrtcDestroyProgram(&program);
            return -3;
        }
        g_api.nvrtcDestroyProgram(&program);
        if (g_api.cuModuleLoadDataEx(&g_module, ptx.data(), 0, nullptr, nullptr)
            != 0) {
            return -4;
        }
        module_cache.emplace(std::move(cache_key), g_module);
    }
    for (const Entry& entry : kNamedKernels) {
        if (g_api.cuModuleGetFunction(entry.slot, g_module, entry.name) != 0) {
            return -5;
        }
        g_functions[entry.name] = *entry.slot;
    }
    for (const char* name : {
             "qwen_q8_embedding", "qwen_f32_matvec",
             "bf16_matvec_warp", "qwen_f32_matvec_warp",
             "qwen_f16_matvec_warp", "qwen_f16_embedding",
             "qwen_f16_embedding_rows", "qwen_f16_matmul_rows",
             "q2k_matvec_transposed_warp", "q3k_matvec_transposed_warp",
             "q4k_matvec_transposed_warp", "q5k_matvec_transposed_warp",
             "q6k_matvec_transposed_warp",
             "iq2xxs_matvec_transposed_warp",
             "iq2xxs_q8_matvec_transposed_warp",
             "q4k_q8_matvec_transposed_warp",
             "q2k_q8_matvec_transposed_warp",
             "q3k_q8_matvec_transposed_warp",
             "q5k_q8_matvec_transposed_warp",
             "q6k_q8_matvec_transposed_warp",
             "iq3xxs_q8_matvec_transposed_warp",
             "q2k_q8_lm_head_argmax_warp",
             "q3k_q8_lm_head_argmax_warp",
             "q4k_q8_lm_head_argmax_warp",
             "q5k_q8_lm_head_argmax_warp",
             "q6k_q8_lm_head_argmax_warp",
             "iq2xxs_q8_lm_head_argmax_warp",
             "iq3xxs_q8_lm_head_argmax_warp",
             "quantize_q8_blocks", "iq3xxs_matvec_transposed_warp",
             "iq2s_matvec_transposed_warp", "iq3s_matvec_transposed_warp",
             "iq2xs_matvec_transposed_warp", "iq4xs_matvec_transposed_warp",
             "iq1m_matvec_transposed_warp", "iq1s_matvec_transposed_warp",
             "q2k_matmul_rows", "q3k_matmul_rows", "q4k_matmul_rows",
             "q5k_matmul_rows", "q6k_matmul_rows",
             "iq2xxs_matmul_rows", "iq3xxs_matmul_rows",
             "iq2s_matmul_rows", "iq3s_matmul_rows",
             "iq2xs_matmul_rows", "iq4xs_matmul_rows",
             "iq1m_matmul_rows", "iq1s_matmul_rows",
             "qwen_delta_recurrent", "qwen_delta_recurrent_split",
             "qwen_attention_query",
             "qwen_attention_key", "qwen_attention_gate",
             "laguna_head_norm_rope", "laguna_attention_gate",
             "laguna_attention_gate_rows",
             "muse_head_norm_rope", "muse_logit_softcap",
             "iq2s_q8_matvec_transposed_warp",
             "iq2s_q8_matvec_transposed_rows",
             "iq2s_q8_matmul_tiled",
             "iq2s_q8_mmq",
             "iq2xxs_q8_mmq", "iq3xxs_q8_mmq",
             "iq2xs_q8_mmq", "iq4xs_q8_mmq",
             "iq2xxs_q8_matmul_tiled", "iq3xxs_q8_matmul_tiled",
             "iq2xs_q8_matmul_tiled", "iq4xs_q8_matmul_tiled",
             "iq2xs_q8_matvec_transposed_warp",
             "iq2xs_q8_matvec_transposed_rows",
             "iq2xxs_q8_matvec_transposed_rows",
             "iq3xxs_q8_matvec_transposed_rows",
             "iq4xs_q8_matvec_transposed_rows",
             // The symmetric K-quants take the same batched prompt path.
             "q3k_q8_matvec_transposed_rows", "q3k_q8_matmul_tiled", "q3k_q8_mmq",
             "q6k_q8_matvec_transposed_rows", "q6k_q8_matmul_tiled", "q6k_q8_mmq",
             // And the asymmetric ones, which carry a per-sub-block minimum.
             "q2k_q8_matvec_transposed_rows", "q2k_q8_mmq",
             "q4k_q8_matvec_transposed_rows", "q4k_q8_mmq",
             "q5k_q8_matvec_transposed_rows", "q5k_q8_mmq",
             // The IQ1 pair, symmetric once the delta is folded into the
             // weights; IQ1_M's sub-scales use the half-block split.
             "iq1s_q8_matvec_transposed_warp",
             "iq1s_q8_matvec_transposed_rows", "iq1s_q8_matmul_tiled",
             "iq1s_q8_mmq",
             "iq1m_q8_matvec_transposed_warp",
             "iq1m_q8_matvec_transposed_rows", "iq1m_q8_matmul_tiled",
             "iq1m_q8_mmq",
             "iq4xs_q8_matvec_transposed_warp",
             "quantize_q8_blocks_rows",
             "route_topk_sigmoid_bias", "route_topk_sigmoid_bias_rows",
             // Grouped routed-expert kernels for the IQ codebook formats.
             "iq2xs_grouped_swiglu", "iq2xs_grouped_swiglu_rows",
             "iq2xs_grouped_accumulate", "iq2xs_grouped_accumulate_rows",
             "iq3xxs_grouped_swiglu", "iq3xxs_grouped_swiglu_rows",
             "iq3xxs_grouped_accumulate", "iq3xxs_grouped_accumulate_rows",
             "iq4xs_grouped_swiglu", "iq4xs_grouped_swiglu_rows",
             "iq4xs_grouped_accumulate", "iq4xs_grouped_accumulate_rows",
             "iq1s_grouped_swiglu", "iq1s_grouped_swiglu_rows",
             "iq1s_grouped_accumulate", "iq1s_grouped_accumulate_rows",
             "iq4nl_grouped_swiglu", "iq4nl_grouped_swiglu_rows",
             "iq4nl_grouped_accumulate", "iq4nl_grouped_accumulate_rows",
             "iq2xxs_grouped_swiglu", "iq2xxs_grouped_swiglu_rows",
             "iq2xxs_grouped_accumulate", "iq2xxs_grouped_accumulate_rows",
             "qwen_attention_query_f16", "kv_attention_softmax_f16",
             "qwen_attention_prefill_pack_f16",
             "kv_attention_prefill_softmax_f16",
             "kv_attention_prefill_block_softmax_f16",
             "qwen_attention_prefill_rescale",
             "qwen_attention_prefill_unpack_gate", "qwen_attention_prefill_unpack",
             "qwen_attention_prefill_unpack_gate_norm",
             "qwen_attention_prefill_unpack_norm",
             "kv_attention_scores", "kv_attention_values",
             "qwen_shared_scale", "qwen_argmax", "qwen_concat_pair",
             "qwen_shared_scale_bf16", "qwen_copy_vector", "silu_mul",
             "qwen_gather_rows", "qwen_scatter_add_rows",
             "qwen_imatrix_accumulate",
             "q8_swiglu_transposed_warp", "q8_lm_head_argmax_warp",
             "qwen_q8_embedding_rows", "qwen_f32_matmul_rows",
             "qwen_q8_matmul_rows", "qwen_q8_swiglu_rows",
              "qwen_shared_scale_rows", "qwen_q8_lm_head_argmax_rows",
              "qwen_shared_scale_rows_bf16", "bf16_matmul_rows",
             "qwen_delta_recurrent_rows", "route_topk_rows", "rms_norm_rows",
              "q5k_grouped_swiglu_rows", "q5k_grouped_accumulate_rows", "q6k_grouped_swiglu",
              "q4k_grouped_swiglu_rows", "q4k_grouped_accumulate_rows",
              "q4k_matvec_transposed", "q4k_lm_head_argmax_warp",
              "q6k_lm_head_argmax_warp", "q6k_matvec_transposed",
             "q6k_grouped_swiglu_rows", "q6k_grouped_accumulate_rows",
             "q8_grouped_swiglu", "q8_grouped_swiglu_rows",
             "q8_grouped_accumulate_rows", "nvfp4_grouped_swiglu_rows",
             "nvfp4_grouped_accumulate_rows", "nvfp4_matvec_transposed",
             "nvfp4_swiglu_transposed", "kv_attention_prefill",
             "nvfp4_repack_cublaslt", "nvfp4_quantize_cublaslt",
             "nvfp4_repack_stacked_moe_cublaslt",
             "nvfp4_stacked_moe_swiglu",
             "nvfp4_persistent_moe_swiglu",
             "nvfp4_concat_native_gate_up_cublaslt",
             "nvfp4_concat_native_down_cublaslt",
             "nvfp4_quantize_broadcast16_cublaslt",
             "nvfp4_repack_concat_down_cublaslt",
             "nvfp4_quantize_weighted_moe_cublaslt",
             "nvfp4_moe_add_first_column",
             "nvfp4_validate_stacked_projection",
             "nvfp4_validate_down_projection",
             "qwen_bf16_embedding", "qwen_bf16_embedding_rows",
             "qwen_f32_embedding", "qwen_f32_embedding_rows",
             "q2k_matvec_transposed", "q3k_matvec_transposed", "q5k_matvec_transposed",
             "f32_lm_head_argmax_warp", "q2k_lm_head_argmax_warp",
             "iq2xxs_matvec_transposed", "iq2xxs_lm_head_argmax_warp",
             "iq3xxs_matvec_transposed", "iq3xxs_lm_head_argmax_warp",
             "iq2s_matvec_transposed", "iq2s_lm_head_argmax_warp",
             "iq2xs_matvec_transposed", "iq2xs_lm_head_argmax_warp",
             "iq4xs_matvec_transposed", "iq4xs_lm_head_argmax_warp",
             "iq1m_matvec_transposed", "iq1s_matvec_transposed",
             "qwen_iq2xs_embedding", "qwen_iq2xs_embedding_rows",
             "qwen_iq4xs_embedding", "qwen_iq4xs_embedding_rows",
             "iq3s_matvec_transposed", "iq3s_lm_head_argmax_warp",
             "qwen_iq2s_embedding", "qwen_iq2s_embedding_rows",
             "qwen_iq3s_embedding", "qwen_iq3s_embedding_rows",
             "qwen_iq3xxs_embedding", "qwen_iq3xxs_embedding_rows",
             "qwen_iq2xxs_embedding", "qwen_iq2xxs_embedding_rows",
             "q3k_lm_head_argmax_warp", "q5k_lm_head_argmax_warp",
             "qwen_q2k_embedding", "qwen_q2k_embedding_rows",
             "qwen_q3k_embedding", "qwen_q3k_embedding_rows",
             "qwen_q4k_embedding", "qwen_q4k_embedding_rows",
             "qwen_q5k_embedding", "qwen_q5k_embedding_rows",
             "qwen_q6k_embedding", "qwen_q6k_embedding_rows",
             "bf16_lm_head_argmax_warp", "qwen_bf16_lm_head_argmax_rows",
             "nvfp4_matmul_rows",
             "q8_matmul_tiled", "q8_matvec_transposed_pair",
             "q8_matvec_transposed_triple", "delta_conv_chunk",
             "qwen_delta_recurrent_chunk",
             "qwen_delta_wy_scores", "qwen_delta_wy_solve",
             "qwen_delta_state_pass", "qwen_delta_norm_gate",
             "kv_store_f32", "kv_store_f16", "kv_store_bf16", "kv_store_q8",
             "kv_attention_scores_f16", "kv_attention_scores_bf16", "kv_attention_scores_q8",
             "kv_attention_values_f16", "kv_attention_values_bf16", "kv_attention_values_q8",
             "kv_attention_scores_ring", "kv_attention_scores_f16_ring", "kv_attention_scores_bf16_ring", "kv_attention_scores_q8_ring",
             "kv_attention_values_ring", "kv_attention_values_f16_ring", "kv_attention_values_bf16_ring", "kv_attention_values_q8_ring",
             "kv_store_turbo3_k", "kv_store_turbo3_v", "kv_store_turbo4_k", "kv_store_turbo4_v",
             "kv_attention_scores_turbo3", "kv_attention_scores_turbo4",
             "kv_attention_values_turbo3", "kv_attention_values_turbo4",
             "kv_attention_scores_turbo3_ring", "kv_attention_scores_turbo4_ring",
             "kv_attention_values_turbo3_ring", "kv_attention_values_turbo4_ring",
             "kv_dequant_turbo3_f16", "kv_dequant_turbo4_f16",
             "turbo_rotate_rows", "turbo_unrotate_rows",
             "kv_attention_fused_f16_tiles", "kv_attention_fused_bf16_tiles",
             "kv_attention_fused_q8_tiles", "kv_attention_fused_merge",
             // 256-dim twins, for heads the 128-wide instantiation cannot cover
             // (Qwen3.6 runs head_dim 256).
             "kv_attention_fused_f16_tiles256", "kv_attention_fused_bf16_tiles256",
             "kv_attention_fused_q8_tiles256", "kv_attention_fused_merge256",
             "kv_attention_fused_f16_tiles512", "kv_attention_fused_bf16_tiles512",
             "kv_attention_fused_q8_tiles512", "kv_attention_fused_merge512",
             // Turbo twins: dot in the rotated domain against the quantized
             // rows directly, paired with the merge that undoes the rotation.
             "kv_attention_fused_turbo3_tiles", "kv_attention_fused_turbo4_tiles",
             "kv_attention_fused_turbo3_tiles256", "kv_attention_fused_turbo4_tiles256",
             "kv_attention_fused_turbo3_tiles512", "kv_attention_fused_turbo4_tiles512",
             "kv_attention_fused_turbo_merge", "kv_attention_fused_turbo_merge256",
             "kv_attention_fused_turbo_merge512",
             // Grouped-query variants: one block per KV head, one warp per
             // query head, KV staged through shared memory.
             "kv_attention_gqa_f16_256_s8", "kv_attention_gqa_f16_256_s4",
             "kv_attention_gqa_bf16_256_s8", "kv_attention_gqa_bf16_256_s4",
             "kv_attention_gqa_q8_256_s8", "kv_attention_gqa_q8_256_s4",
             "kv_attention_prefill_f16", "kv_attention_prefill_bf16", "kv_attention_prefill_q8",
             "gemma_q4_0_matvec", "gemma_q4_0_embedding", "gemma_q4_0_geglu",
             "gemma_q4_0_grouped_geglu", "gemma_q4_0_grouped_accumulate",
             "gemma_q4_0_pinned_geglu", "gemma_q4_0_pinned_accumulate",
             "gemma_rms_rows", "gemma_scaled_add_rows", "gemma_q4_0_matvec_rows",
             "gemma_q4_0_geglu_rows", "gemma_head_norm_rope_rows",
             "gemma_head_rms_rows", "gemma_router_input_rows",
             "gemma_f32_matvec_rows", "gemma_scale_vector_rows",
             "gemma_quantize_q8_rows", "gemma_q4_0_q8_mmq_rows",
             "gemma_geglu_combine_rows", "gemma_kv_prefill_wide_f16",
             "gemma_head_norm_rope", "gemma_head_rms", "gemma_router_input", "gemma_scale_vector",
             "gemma_q4_0_lm_argmax",
             // DeepSeek-V4. Resolved the same way and simply absent when the
             // source those kernels live in was not compiled in.
             "ds4_q8_matvec", "ds4_q8_grouped_matvec", "ds4_q6k_matvec",
             "ds4_iq1s_matvec", "ds4_iq1s_grouped_swiglu",
             "ds4_clamped_swiglu",
             // BailingMoE3. Same treatment: resolved if present, absent
             // otherwise, so a build without them still loads.
             "bailing_kda_recurrent_chunk", "bailing_mla_attention",
             "bailing_mla_project", "bailing_mla_scores",
             "bailing_mla_scores_pair", "bailing_mla_scores_fused",
             "bailing_mla_project_fused", "bailing_mla_output_fused",
             "bailing_mla_softmax", "bailing_mla_accumulate",
             "bailing_mla_accumulate_split", "bailing_mla_accumulate_fused",
             "bailing_mla_accumulate_reduce",
             "bailing_mla_output",
             "bailing_mla_prepare_rows", "bailing_mla_project_rows",
             "bailing_mla_scores_rows", "bailing_mla_softmax_rows",
             "bailing_mla_accumulate_rows", "bailing_mla_output_rows",
             "bailing_mla_scores_rows_gemm",
             "bailing_mla_accumulate_rows_gemm",
             "bailing_short_conv_rows", "bailing_gated_head_norm_rows",
             "bailing_route_rows", "bailing_q6_expert_swiglu_rows",
             "bailing_q6_expert_accumulate_rows", "bailing_quantize_q8_rows",
             "bailing_q6_expert_swiglu_grouped_rows",
             "bailing_q6_expert_accumulate_grouped_rows",
             "bailing_q6_q8_expert_swiglu_grouped_rows",
             "bailing_q6_q8_expert_swiglu_mmq_rows",
             "bailing_q6_f32_expert_accumulate_mmq_rows",
             "bailing_q6_grouped_swiglu_warp",
             "bailing_q6_grouped_accumulate_warp",
             "bailing_q6_matmul_rows_16", "bailing_q6_q8_matmul_rows",
             "bailing_q6_q8_mmq_rows",
             "bailing_mla_fused_rows",
             "bailing_copy", "bailing_partial_rope", "bailing_split_query",
             "bailing_head_gate", "bailing_short_conv",
             "bailing_gated_head_norm", "bailing_swiglu",
             "bailing_q4k_matvec",
             // qwen4exp gated residual + PLE glue
             "qwen4_hc_init", "qwen4_group_rms", "qwen4_silu_scale",
             "qwen4_hc_mix", "qwen4_hc_inject", "qwen4_ple_gate",
             "qwen4_ple_gv", "qwen4_ple_conv_step", "qwen4_ple_add",
             "qwen4_hc_init_rows", "qwen4_group_rms_rows",
             "qwen4_hc_mix_rows", "qwen4_hc_inject_rows",
             "qwen4_ple_gate_rows", "qwen4_ple_gv_rows",
             "qwen4_ple_conv_sequence"
         }) {
        CUfunction function = nullptr;
        if (g_api.cuModuleGetFunction(&function, g_module, name) == 0)
            g_functions[name] = function;
    }
    return 0;
}

extern "C" int colibri_gpu_rms_norm(
    std::uint64_t input,
    std::uint64_t weights,
    std::uint64_t output,
    std::int32_t size,
    float epsilon,
    std::int32_t one_centered
) {
    if (g_kernels.rms_norm == nullptr) {
        return -1;
    }
    void* args[] = {
        &input, &weights, &output, &size, &epsilon, &one_centered,
    };
    if (launch(g_kernels.rms_norm, 1, 1, kThreadsPerBlock, args) != 0) {
        return -2;
    }
    return 0;
}

extern "C" int colibri_gpu_q4_matvec(
    std::uint64_t packed,
    std::uint64_t scales,
    std::uint64_t input,
    std::uint64_t output,
    std::uint64_t stream,
    std::int32_t rows,
    std::int32_t columns
) {
    if (g_kernels.q4_matvec == nullptr || g_context == nullptr
        || packed == 0 || scales == 0 || input == 0 || output == 0
        || rows <= 0 || columns <= 0
        || g_api.cuCtxSetCurrent(g_context) != 0) {
        return -1;
    }
    void* args[] = {
        &packed, &scales, &input, &output, &rows, &columns,
    };
    return launch(
        g_kernels.q4_matvec,
        static_cast<unsigned int>(rows), 1, kThreadsPerBlock, args,
        0, reinterpret_cast<CUstream>(stream)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_scaled_add(
    std::uint64_t target,
    std::uint64_t source,
    float scale,
    std::int32_t elements
) {
    if (g_kernels.scaled_add == nullptr || target == 0 || source == 0
        || elements <= 0) {
        return -1;
    }
    void* args[] = {&target, &source, &scale, &elements};
    const auto blocks = static_cast<unsigned int>(
        (elements + static_cast<std::int32_t>(kThreadsPerBlock) - 1)
        / static_cast<std::int32_t>(kThreadsPerBlock)
    );
    return launch(g_kernels.scaled_add, blocks, 1, kThreadsPerBlock, args)
        == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_attention(
    std::uint64_t query,
    std::uint64_t keys,
    std::uint64_t values,
    std::uint64_t output,
    std::int32_t heads,
    std::int32_t kv_heads,
    std::int32_t head_dim,
    std::int32_t tokens,
    float scale
) {
    if (g_kernels.attention == nullptr || query == 0 || keys == 0
        || values == 0 || output == 0 || heads <= 0 || kv_heads <= 0
        || head_dim <= 0 || tokens <= 0 || heads % kv_heads != 0) {
        return -1;
    }
    std::int32_t capacity = tokens;
    void* args[] = {
        &query, &keys, &values, &output, &heads, &kv_heads,
        &head_dim, &tokens, &capacity, &scale,
    };
    return launch(
        g_kernels.attention,
        static_cast<unsigned int>(heads), 1, kThreadsPerBlock, args
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_attention_cache(
    std::uint64_t query, std::uint64_t keys, std::uint64_t values,
    std::uint64_t output, std::int32_t heads, std::int32_t kv_heads,
    std::int32_t head_dim, std::int32_t tokens, std::int32_t capacity,
    float scale
) {
    if (g_kernels.attention == nullptr || query == 0 || keys == 0
        || values == 0 || output == 0 || heads <= 0 || kv_heads <= 0
        || head_dim <= 0 || tokens <= 0 || capacity < tokens
        || heads % kv_heads != 0) return -1;
    void* args[] = {&query, &keys, &values, &output, &heads, &kv_heads,
                    &head_dim, &tokens, &capacity, &scale};
    return launch(g_kernels.attention, static_cast<unsigned int>(heads), 1,
                  kThreadsPerBlock, args) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_kv_append(
    std::uint64_t current_keys, std::uint64_t current_values,
    std::uint64_t cache_keys, std::uint64_t cache_values,
    std::int32_t kv_heads, std::int32_t head_dim,
    std::int32_t position, std::int32_t capacity
) {
    if (g_kernels.kv_append == nullptr || current_keys == 0
        || current_values == 0 || cache_keys == 0 || cache_values == 0
        || kv_heads <= 0 || head_dim <= 0 || position < 0
        || position >= capacity) return -1;
    void* args[] = {&current_keys, &current_values, &cache_keys,
                    &cache_values, &kv_heads, &head_dim, &position,
                    &capacity};
    return launch(g_kernels.kv_append,
                  static_cast<unsigned int>(kv_heads), 1,
                  kThreadsPerBlock, args) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_q4_moe(
    std::uint64_t gate_up_packed,
    std::uint64_t gate_up_scales,
    std::uint64_t down_packed,
    std::uint64_t down_scales,
    std::uint64_t weights,
    std::uint64_t input,
    std::uint64_t gate_output,
    std::uint64_t activated,
    std::uint64_t output,
    std::uint64_t stream,
    std::int32_t expert_count,
    std::int32_t hidden_size,
    std::int32_t intermediate_size
) {
    if (g_kernels.q4_silu == nullptr
        || g_kernels.q4_weighted == nullptr
        || gate_up_packed == 0 || gate_up_scales == 0 || down_packed == 0
        || down_scales == 0 || weights == 0 || input == 0
        || gate_output == 0 || activated == 0 || output == 0
        || expert_count <= 0 || hidden_size <= 0 || intermediate_size <= 0
        || g_context == nullptr || g_api.cuCtxSetCurrent(g_context) != 0) {
        return -1;
    }
    void* gate_args[] = {
        &gate_up_packed, &gate_up_scales, &input, &activated,
        &intermediate_size, &hidden_size, &expert_count,
    };
    if (launch(
            g_kernels.q4_silu,
            static_cast<unsigned int>(intermediate_size),
            static_cast<unsigned int>(expert_count), kThreadsPerBlock,
            gate_args, 0, reinterpret_cast<CUstream>(stream)
        ) != 0) {
        return -2;
    }
    void* down_args[] = {
        &down_packed, &down_scales, &activated, &weights, &output,
        &hidden_size, &intermediate_size, &expert_count,
    };
    if (launch(
            g_kernels.q4_weighted,
            static_cast<unsigned int>(hidden_size), 1, kThreadsPerBlock,
            down_args, 0, reinterpret_cast<CUstream>(stream)
        ) != 0) {
        return -3;
    }
    return 0;
}

extern "C" int colibri_gpu_sync() {
    if (colibri_backend_is_cpu()) return colibri_cpu_sync();

    if (!g_api.loaded) {
        return -1;
    }
    return g_api.cuStreamSynchronize(nullptr) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_alloc(std::uint64_t bytes, std::uint64_t* pointer) {
    if (colibri_backend_is_cpu()) return colibri_cpu_alloc(bytes, pointer);

    if (g_context == nullptr || pointer == nullptr || bytes == 0
        || g_api.cuCtxSetCurrent(g_context) != 0) return -1;
    CUdeviceptr allocation = 0;
    if (g_api.cuMemAlloc(&allocation, static_cast<size_t>(bytes)) != 0) return -2;
    *pointer = static_cast<std::uint64_t>(allocation);
    return 0;
}

extern "C" int colibri_gpu_free(std::uint64_t pointer) {
    if (colibri_backend_is_cpu()) return colibri_cpu_free(pointer);

    if (pointer == 0) return 0;
    if (g_context == nullptr || g_api.cuCtxSetCurrent(g_context) != 0) return -1;
    return g_api.cuMemFree(static_cast<CUdeviceptr>(pointer)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_host_alloc(std::uint64_t bytes, void** pointer) {
    if (colibri_backend_is_cpu()) return colibri_cpu_host_alloc(bytes, pointer);

    if (pointer == nullptr || bytes == 0) return -1;
    return g_api.cuMemHostAlloc(pointer, static_cast<size_t>(bytes), 0) == 0
        ? 0 : -2;
}

extern "C" int colibri_gpu_host_free(void* pointer) {
    if (colibri_backend_is_cpu()) return colibri_cpu_host_free(pointer);

    if (pointer == nullptr) return 0;
    return g_api.cuMemFreeHost(pointer) == 0 ? 0 : -1;
}

// Page-lock an existing host range (e.g. the model mmap) so cuMemcpyHtoDAsync
// DMAs straight from it with no CPU staging copy. Read-only file mappings need
// the READ_ONLY flag; fall back to portable/plain for older drivers.
extern "C" int colibri_gpu_host_register(const void* pointer, std::uint64_t bytes) {
    if (colibri_backend_is_cpu()) return colibri_cpu_host_register(pointer, bytes);

    if (pointer == nullptr || bytes == 0) return -1;
    if (g_api.cuMemHostRegister == nullptr) return -3;
    void* host = const_cast<void*>(pointer);
    // READ_ONLY (0x08) is unsupported on many drivers (CUDA_ERROR_NOT_SUPPORTED);
    // PORTABLE (0x01) works once the mapping is writable copy-on-write.
    for (unsigned int flags : {0x08u, 0x01u, 0x00u}) {
        if (g_api.cuMemHostRegister(host, static_cast<size_t>(bytes), flags) == 0) return 0;
    }
    return -2;
}

extern "C" int colibri_gpu_host_unregister(const void* pointer) {
    if (colibri_backend_is_cpu()) return colibri_cpu_host_unregister(pointer);

    if (pointer == nullptr) return 0;
    if (g_api.cuMemHostUnregister == nullptr) return -3;
    return g_api.cuMemHostUnregister(const_cast<void*>(pointer)) == 0 ? 0 : -1;
}

extern "C" int colibri_gpu_upload(
    std::uint64_t destination, const void* source, std::uint64_t bytes,
    std::uint64_t stream
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_upload(destination, source, bytes, stream);

    if (destination == 0 || source == nullptr || bytes == 0) return -1;
    return g_api.cuMemcpyHtoDAsync(
        static_cast<CUdeviceptr>(destination), source, static_cast<size_t>(bytes),
        reinterpret_cast<CUstream>(stream)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_upload_sync(
    std::uint64_t destination, const void* source, std::uint64_t bytes
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_upload_sync(destination, source, bytes);

    if (destination == 0 || source == nullptr || bytes == 0) return -1;
    return g_api.cuMemcpyHtoD(
        static_cast<CUdeviceptr>(destination), source, static_cast<size_t>(bytes)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_download(
    void* destination, std::uint64_t source, std::uint64_t bytes,
    std::uint64_t stream
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_download(destination, source, bytes, stream);

    if (destination == nullptr || source == 0 || bytes == 0) return -1;
    return g_api.cuMemcpyDtoHAsync(
        destination, static_cast<CUdeviceptr>(source), static_cast<size_t>(bytes),
        reinterpret_cast<CUstream>(stream)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_memset(
    std::uint64_t destination, std::uint8_t value, std::uint64_t bytes,
    std::uint64_t stream
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_memset(destination, value, bytes, stream);

    if (destination == 0 || bytes == 0) return -1;
    return g_api.cuMemsetD8Async(
        static_cast<CUdeviceptr>(destination), value, static_cast<size_t>(bytes),
        reinterpret_cast<CUstream>(stream)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_stream_create(std::uint64_t* stream) {
    if (colibri_backend_is_cpu()) return colibri_cpu_stream_create(stream);

    if (stream == nullptr) return -1;
    CUstream created = nullptr;
    if (g_api.cuStreamCreate(&created, 1) != 0) return -2;
    *stream = reinterpret_cast<std::uint64_t>(created);
    return 0;
}

extern "C" int colibri_gpu_stream_destroy(std::uint64_t stream) {
    if (colibri_backend_is_cpu()) return colibri_cpu_stream_destroy(stream);

    if (stream == 0) return 0;
    const auto cuda_stream = reinterpret_cast<CUstream>(stream);
    {
        // NVFP4 scratch is process-global and remembers the stream that last
        // used it. Release it before that stream becomes invalid; otherwise
        // model reloads retain VRAM and later try to synchronize a stale handle.
        std::lock_guard<std::mutex> lock(g_cublas_mutex);
        if (g_nvfp4_scratch.stream == cuda_stream) {
            // Free only after a successful drain: a failed synchronize means
            // work may still be reading these buffers, and freeing them then
            // trades a bounded leak for a use-after-free. The handles are
            // forgotten either way -- they must not outlive the stream.
            const bool drained =
                g_api.cuStreamSynchronize(cuda_stream) == 0;
            if (drained) {
                const auto release = [](CUdeviceptr pointer) {
                    if (pointer) g_api.cuMemFree(pointer);
                };
                release(g_nvfp4_scratch.weight_values);
                release(g_nvfp4_scratch.weight_scales);
                release(g_nvfp4_scratch.input_values);
                release(g_nvfp4_scratch.input_scales);
                release(g_nvfp4_scratch.projected);
                release(g_nvfp4_scratch.expert_pointers);
            }
            const CUevent handoff = g_nvfp4_scratch.handoff;
            g_nvfp4_scratch = Nvfp4Scratch{};
            // The handoff event belongs to the context, not the stream; it
            // stays usable for the next scratch owner.
            g_nvfp4_scratch.handoff = handoff;
        }
    }
    return g_api.cuStreamDestroy(cuda_stream) == 0
        ? 0 : -1;
}

extern "C" int colibri_gpu_stream_sync(std::uint64_t stream) {
    if (colibri_backend_is_cpu()) return colibri_cpu_stream_sync(stream);

    return g_api.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)) == 0
        ? 0 : -1;
}

extern "C" int colibri_gpu_graph_begin(std::uint64_t stream) {
    if (colibri_backend_is_cpu()) return colibri_cpu_graph_begin(stream);

    if (stream == 0 || g_api.cuStreamBeginCapture == nullptr) return -1;
    return g_api.cuStreamBeginCapture(
        reinterpret_cast<CUstream>(stream), 2 /* relaxed */
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_graph_end(
    std::uint64_t stream, std::uint64_t* handle
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_graph_end(stream, handle);

    if (stream == 0 || handle == nullptr || g_api.cuStreamEndCapture == nullptr)
        return -1;
    *handle = 0;
    void* graph = nullptr;
    if (g_api.cuStreamEndCapture(reinterpret_cast<CUstream>(stream), &graph) != 0
        || graph == nullptr)
        return -2;
    void* executable = nullptr;
    const int status = g_api.cuGraphInstantiateWithFlags(&executable, graph, 0);
    g_api.cuGraphDestroy(graph);
    if (status != 0 || executable == nullptr) return -3;
    *handle = reinterpret_cast<std::uint64_t>(executable);
    return 0;
}

extern "C" int colibri_gpu_graph_launch(
    std::uint64_t graph, std::uint64_t stream
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_graph_launch(graph, stream);

    if (graph == 0 || stream == 0 || g_api.cuGraphLaunch == nullptr) return -1;
    return g_api.cuGraphLaunch(
        reinterpret_cast<void*>(graph), reinterpret_cast<CUstream>(stream)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_graph_destroy(std::uint64_t graph) {
    if (colibri_backend_is_cpu()) return colibri_cpu_graph_destroy(graph);

    if (graph == 0) return 0;
    if (g_api.cuGraphExecDestroy == nullptr) return -1;
    return g_api.cuGraphExecDestroy(reinterpret_cast<void*>(graph)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_event_create(std::uint64_t* event) {
    if (colibri_backend_is_cpu()) return colibri_cpu_event_create(event);

    if (event == nullptr) return -1;
    CUevent created = nullptr;
    if (g_api.cuEventCreate(&created, 2 /* disable timing */) != 0) return -2;
    *event = reinterpret_cast<std::uint64_t>(created);
    return 0;
}

extern "C" int colibri_gpu_timed_event_create(std::uint64_t* event) {
    if (colibri_backend_is_cpu()) return colibri_cpu_timed_event_create(event);

    if (event == nullptr) return -1;
    CUevent created = nullptr;
    if (g_api.cuEventCreate(&created, 0) != 0) return -2;
    *event = reinterpret_cast<std::uint64_t>(created);
    return 0;
}

extern "C" int colibri_gpu_event_record(
    std::uint64_t event, std::uint64_t stream
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_event_record(event, stream);

    if (event == 0) return -1;
    return g_api.cuEventRecord(
        reinterpret_cast<CUevent>(event), reinterpret_cast<CUstream>(stream)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_event_sync(std::uint64_t event) {
    if (colibri_backend_is_cpu()) return colibri_cpu_event_sync(event);

    if (event == 0) return -1;
    return g_api.cuEventSynchronize(reinterpret_cast<CUevent>(event)) == 0
        ? 0 : -2;
}

extern "C" int colibri_gpu_stream_wait_event(
    std::uint64_t stream, std::uint64_t event
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_stream_wait_event(stream, event);

    if (event == 0) return -1;
    return g_api.cuStreamWaitEvent(
        reinterpret_cast<CUstream>(stream), reinterpret_cast<CUevent>(event), 0
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_event_destroy(std::uint64_t event) {
    if (colibri_backend_is_cpu()) return colibri_cpu_event_destroy(event);

    if (event == 0) return 0;
    return g_api.cuEventDestroy(reinterpret_cast<CUevent>(event)) == 0
        ? 0 : -1;
}

extern "C" int colibri_gpu_event_elapsed(
    std::uint64_t start, std::uint64_t end, float* milliseconds
) {
    if (colibri_backend_is_cpu()) return colibri_cpu_event_elapsed(start, end, milliseconds);

    if (start == 0 || end == 0 || milliseconds == nullptr) return -1;
    return g_api.cuEventElapsedTime(
        milliseconds, reinterpret_cast<CUevent>(start),
        reinterpret_cast<CUevent>(end)
    ) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_q8_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    if (!g_kernels.q8_matvec_transposed || !packed || !input || !output
        || input_size <= 0 || output_size <= 0) return -1;
    void* args[] = {&packed, &input, &output, &input_size, &output_size};
    const auto blocks = static_cast<unsigned int>((output_size + 7) / 8);
    return launch(g_kernels.q8_matvec_transposed, blocks, 1, 256, args, 0,
                  reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

// Q4_K, with the sub-block scale unpack amortized across a warp.
//
// The generic `q4k_matvec_transposed_warp` below calls `q4k_value` per element,
// which re-derives the block offsets and reloads both f16 scales to produce
// four bits -- instruction-bound, not bandwidth-bound, and measured at 58-131
// GB/s on a card that does ~670. `bailing_q4k_matvec` was written to fix that
// and has been compiled and resolved into the function table all along with
// nothing to launch it; this is that launcher.
//
// Block per row, one warp per 32-element sub-block, block-reduced at the end,
// so the launch geometry differs from the warp-per-row kernels. Returns -1 when
// the kernel is absent, which is what lets the caller fall back.
extern "C" int colibri_gpu_q4k_matvec_subblock(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    if (!packed || !input || !output || input_size <= 0 || output_size <= 0)
        return -1;
    // The kernel walks whole 32-element sub-blocks; a ragged row would read
    // past the last one.
    if (input_size % 32 != 0) return -1;
    auto it = g_functions.find("bailing_q4k_matvec");
    if (it == g_functions.end()) return -1;
    void* args[] = {&packed, &input, &output, &input_size, &output_size};
    return launch(it->second, static_cast<unsigned int>(output_size), 1, 256,
                  args, 0, reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_q4k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    if (!packed || !input || !output || input_size <= 0 || output_size <= 0)
        return -1;
    CUfunction function = nullptr;
    bool warp_mapped = false;
    auto it = g_functions.find("q4k_matvec_transposed_warp");
    if (it == g_functions.end()) it = g_functions.find("q4k_matvec_transposed");
    else warp_mapped = true;
    if (it != g_functions.end()) function = it->second;
    if (!function) return -1;
    void* args[] = {&packed, &input, &output, &input_size, &output_size};
    const auto blocks = warp_mapped
        ? static_cast<unsigned int>((output_size + 7) / 8)
        : static_cast<unsigned int>(output_size);
    return launch(function, blocks, 1, 256,
                  args, 0, reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

namespace {

// Prefer the warp-per-row implementation while retaining the original
// block-per-row entry point as a compatibility fallback.
int launch_kquant_matvec(
    const char* warp_name, const char* fallback_name,
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    if (!packed || !input || !output || input_size <= 0 || output_size <= 0)
        return -1;
    bool warp_mapped = std::strcmp(warp_name, fallback_name) != 0;
    auto it = g_functions.find(warp_mapped ? warp_name : fallback_name);
    if (it == g_functions.end() && warp_mapped) {
        warp_mapped = false;
        it = g_functions.find(fallback_name);
    }
    if (it == g_functions.end() || !it->second) return -1;
    void* args[] = {&packed, &input, &output, &input_size, &output_size};
    const auto blocks = warp_mapped
        ? static_cast<unsigned int>((output_size + 7) / 8)
        : static_cast<unsigned int>(output_size);
    return launch(it->second, blocks, 1, 256,
                  args, 0, reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

} // namespace

extern "C" int colibri_gpu_q2k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "q2k_matvec_transposed_warp", "q2k_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_q3k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "q3k_matvec_transposed_warp", "q3k_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_q5k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "q5k_matvec_transposed_warp", "q5k_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq2xxs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq2xxs_matvec_transposed_warp", "iq2xxs_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq1m_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq1m_matvec_transposed_warp", "iq1m_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq1s_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq1s_matvec_transposed_warp", "iq1s_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq3xxs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq3xxs_matvec_transposed_warp", "iq3xxs_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq2s_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq2s_matvec_transposed_warp", "iq2s_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq3s_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq3s_matvec_transposed_warp", "iq3s_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq2xs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq2xs_matvec_transposed_warp", "iq2xs_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_iq4xs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    return launch_kquant_matvec(
        "iq4xs_matvec_transposed_warp", "iq4xs_matvec_transposed",
        packed, input, output, input_size, output_size, stream);
}

extern "C" int colibri_gpu_q6k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    if (!packed || !input || !output || input_size <= 0 || output_size <= 0)
        return -1;
    CUfunction function = nullptr;
    bool warp_mapped = false;
    auto it = g_functions.find("q6k_matvec_transposed_warp");
    if (it == g_functions.end()) it = g_functions.find("q6k_matvec_transposed");
    else warp_mapped = true;
    if (it != g_functions.end()) function = it->second;
    if (!function) return -1;
    void* args[] = {&packed, &input, &output, &input_size, &output_size};
    const auto blocks = warp_mapped
        ? static_cast<unsigned int>((output_size + 7) / 8)
        : static_cast<unsigned int>(output_size);
    return launch(function, blocks, 1, 256,
                  args, 0, reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_bf16_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
) {
    if (!g_kernels.bf16_matvec || !packed || !input || !output
        || input_size <= 0 || output_size <= 0) return -1;
    // bf16_matvec takes (rows, columns), i.e. the output/input sizes in the
    // opposite order from the quantized transposed matvecs.
    void* args[] = {&packed, &input, &output, &output_size, &input_size};
    return launch(g_kernels.bf16_matvec, static_cast<unsigned int>(output_size),
                  1, 256, args, 0,
                  reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_route_topk(
    std::uint64_t logits, std::uint64_t selected, std::uint64_t weights,
    std::int32_t experts, std::int32_t top_k, std::uint64_t stream
) {
    if (!g_kernels.route_topk || !logits || !selected || !weights
        || experts <= 0 || top_k <= 0 || top_k > experts) return -1;
    void* args[] = {&logits, &selected, &weights, &experts, &top_k};
    return launch(g_kernels.route_topk, 1, 1, 256, args,
                  static_cast<unsigned int>(experts * sizeof(float)),
                  reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_sampling_topk(
    std::uint64_t logits, std::uint64_t selected,
    std::uint64_t selected_logits, std::uint64_t sort_indices_a,
    std::uint64_t sort_values_a, std::uint64_t sort_indices_b,
    std::uint64_t sort_values_b, std::int32_t vocabulary,
    std::int32_t top_k, std::uint64_t stream
) {
    if (!g_kernels.sampling_block_topk_logits
        || !g_kernels.sampling_block_topk_pairs || !logits || !selected
        || !selected_logits || !sort_indices_a || !sort_values_a
        || !sort_indices_b || !sort_values_b
        || vocabulary <= 0 || top_k <= 0 || top_k > 256
        || top_k > vocabulary) return -1;
    // 256 must track workspace::kSamplingTopKCapacity: the sort buffers hold
    // blocks * top_k pairs, and the merge loop below only makes progress
    // while top_k is comfortably under items_per_block.
    constexpr std::int32_t items_per_block = 1024;
    std::int32_t blocks = (vocabulary + items_per_block - 1) / items_per_block;
    if (blocks > 256) return -1;
    auto cuda_stream = reinterpret_cast<CUstream>(stream);
    void* first_args[] = {
        &logits, &sort_indices_a, &sort_values_a, &vocabulary, &top_k
    };
    if (launch(g_kernels.sampling_block_topk_logits,
               static_cast<unsigned int>(blocks), 1, 256,
               first_args, 0, cuda_stream) != 0) return -2;
    std::int32_t count = blocks * top_k;
    std::uint64_t input_indices = sort_indices_a;
    std::uint64_t input_values = sort_values_a;
    std::uint64_t output_indices = sort_indices_b;
    std::uint64_t output_values = sort_values_b;
    while (count > items_per_block) {
        blocks = (count + items_per_block - 1) / items_per_block;
        void* merge_args[] = {
            &input_indices, &input_values, &output_indices, &output_values,
            &count, &top_k
        };
        if (launch(g_kernels.sampling_block_topk_pairs,
                   static_cast<unsigned int>(blocks), 1, 256,
                   merge_args, 0, cuda_stream) != 0) return -2;
        count = blocks * top_k;
        std::swap(input_indices, output_indices);
        std::swap(input_values, output_values);
    }
    void* final_args[] = {
        &input_indices, &input_values, &selected, &selected_logits,
        &count, &top_k
    };
    if (launch(g_kernels.sampling_block_topk_pairs, 1, 1, 256,
               final_args, 0, cuda_stream) != 0) return -2;
    return 0;
}

extern "C" int colibri_gpu_q5_grouped_swiglu(
    std::uint64_t gate_pointers, std::uint64_t up_pointers,
    std::uint64_t input, std::uint64_t activated,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) {
    if (!g_kernels.q5_grouped_swiglu || !gate_pointers || !up_pointers
        || !input || !activated || input_size <= 0 || output_size <= 0
        || experts <= 0) return -1;
    void* args[] = {&gate_pointers, &up_pointers, &input, &activated,
                    &input_size, &output_size, &experts};
    return launch(g_kernels.q5_grouped_swiglu,
                  static_cast<unsigned int>(output_size),
                  static_cast<unsigned int>(experts), 256, args, 0,
                  reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

int grouped_accumulate(
    CUfunction kernel, std::uint64_t down_pointers,
    std::uint64_t activated, std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) {
    if (!kernel || !down_pointers || !activated || !output || !weights
        || input_size <= 0 || output_size <= 0 || experts <= 0) return -1;
    void* args[] = {&down_pointers, &activated, &output, &weights,
                    &input_size, &output_size, &experts};
    return launch(kernel, static_cast<unsigned int>(output_size), 1, 256,
                  args, 0, reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_q6_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) { return grouped_accumulate(g_kernels.q6_grouped_accumulate, down_pointers,
    activated, output, weights, input_size, output_size, experts, stream); }

extern "C" int colibri_gpu_q4k_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) { return grouped_accumulate(g_kernels.q4k_grouped_accumulate, down_pointers,
    activated, output, weights, input_size, output_size, experts, stream); }

extern "C" int colibri_gpu_q5k_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) { return grouped_accumulate(g_kernels.q5k_grouped_accumulate, down_pointers,
    activated, output, weights, input_size, output_size, experts, stream); }

extern "C" int colibri_gpu_q8_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) { return grouped_accumulate(g_kernels.q8_grouped_accumulate, down_pointers,
    activated, output, weights, input_size, output_size, experts, stream); }

extern "C" int colibri_gpu_nvfp4_grouped_swiglu(
    std::uint64_t gate_pointers, std::uint64_t up_pointers,
    std::uint64_t input, std::uint64_t activated,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) {
    const char* tiled_env = std::getenv("COLIBRI_NVFP4_TILED");
    const bool tiled = tiled_env && tiled_env[0] == '1';
    const auto kernel = tiled ? g_kernels.nvfp4_grouped_swiglu_tiled
                              : g_kernels.nvfp4_grouped_swiglu;
    if (!kernel || !gate_pointers || !up_pointers
        || !input || !activated || input_size <= 0 || output_size <= 0
        || experts <= 0) return -1;
    void* args[] = {&gate_pointers, &up_pointers, &input, &activated,
                    &input_size, &output_size, &experts};
    return launch(kernel,
                  static_cast<unsigned int>(
                      tiled ? (output_size + 7) / 8 : output_size),
                  static_cast<unsigned int>(experts), 256, args, 0,
                  reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

extern "C" int colibri_gpu_nvfp4_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
) {
    // Not reachable through launch(), so the CPU backend cannot substitute a
    // host kernel for it: this bottoms out in cuBLAS directly. Report failure
    // so the runtime falls back to the kernel path it already has. Returning
    // success here silently skipped attention entirely for sequences of 128+
    // tokens, which is where cuBLAS attention becomes eligible.
    if (colibri_backend_is_cpu()) return -1;

    const char* tiled_env = std::getenv("COLIBRI_NVFP4_TILED");
    const bool tiled = tiled_env && tiled_env[0] == '1';
    const auto kernel = tiled ? g_kernels.nvfp4_grouped_accumulate_tiled
                              : g_kernels.nvfp4_grouped_accumulate;
    if (!kernel || !down_pointers || !activated
        || !output || !weights || input_size <= 0 || output_size <= 0
        || experts <= 0) return -1;
    void* args[] = {&down_pointers, &activated, &output, &weights,
                    &input_size, &output_size, &experts};
    return launch(kernel,
                  static_cast<unsigned int>(
                      tiled ? (output_size + 7) / 8 : output_size), 1, 256,
                  args, 0, reinterpret_cast<CUstream>(stream)) == 0 ? 0 : -2;
}

static size_t nvfp4_cublas_scale_bytes(int outer, int inner) {
    const size_t outer_tiles = (static_cast<size_t>(outer) + 127) / 128;
    const size_t inner_tiles =
        ((static_cast<size_t>(inner) + 15) / 16 + 3) / 4;
    return outer_tiles * inner_tiles * 512;
}

static void nvfp4_clear_cublas_plans() {
    for (auto& entry : g_nvfp4_plans) {
        auto& plan = entry.second;
        if (plan.d_layout) g_cublas_lt.layout_destroy(plan.d_layout);
        if (plan.c_layout) g_cublas_lt.layout_destroy(plan.c_layout);
        if (plan.b_layout) g_cublas_lt.layout_destroy(plan.b_layout);
        if (plan.a_layout) g_cublas_lt.layout_destroy(plan.a_layout);
        if (plan.operation) g_cublas_lt.matmul_desc_destroy(plan.operation);
    }
    g_nvfp4_plans.clear();
}

// The scratch is single-buffered across streams. This used to be a full
// cuStreamSynchronize whenever prefill and decode alternated streams -- a
// host stall per alternation. Recording an event on the old stream and
// making the new one wait keeps the ordering entirely on-device. Callers
// that free the scratch (the grow path) still host-sync first.
static int nvfp4_scratch_switch_stream(CUstream cuda_stream) {
    if (g_nvfp4_scratch.stream == nullptr
        || g_nvfp4_scratch.stream == cuda_stream)
        return 0;
    if (g_nvfp4_scratch.handoff == nullptr
        && g_api.cuEventCreate(
               &g_nvfp4_scratch.handoff, 2 /* disable timing */) != 0)
        return -1;
    if (g_api.cuEventRecord(
            g_nvfp4_scratch.handoff, g_nvfp4_scratch.stream) != 0)
        return -1;
    return g_api.cuStreamWaitEvent(cuda_stream, g_nvfp4_scratch.handoff, 0)
               == 0 ? 0 : -1;
}

static int nvfp4_run_quantized_gemm(
    int input_size, int output_size, int rows, float alpha, float beta,
    std::uint64_t output, CUstream cuda_stream,
    std::uint64_t weight_values = 0, std::uint64_t weight_scales = 0,
    std::uint64_t input_values = 0, std::uint64_t input_scales = 0
) {
    constexpr int kCudaR32F = 0;
    constexpr int kCudaR4E2M1 = 33;
    constexpr int kCompute32F = 68;
    constexpr int kOpN = 0, kOpT = 1;
    constexpr int kDescTransA = 3, kDescTransB = 4;
    constexpr int kDescAScalePointer = 17, kDescBScalePointer = 18;
    constexpr int kDescAScaleMode = 31, kDescBScaleMode = 32;
    constexpr int kScaleVec16Ue4m3 = 1;
    constexpr int kPreferenceMaxWorkspace = 1;
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(input_size))
         << 40) |
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_size))
         << 16) |
        static_cast<std::uint16_t>(rows);
    auto found = g_nvfp4_plans.find(key);
    if (found == g_nvfp4_plans.end()) {
        Nvfp4LtPlan plan;
        cublasLtMatmulPreference_t preference = nullptr;
        auto cleanup = [&]() {
            if (preference) g_cublas_lt.preference_destroy(preference);
            if (plan.d_layout) g_cublas_lt.layout_destroy(plan.d_layout);
            if (plan.c_layout) g_cublas_lt.layout_destroy(plan.c_layout);
            if (plan.b_layout) g_cublas_lt.layout_destroy(plan.b_layout);
            if (plan.a_layout) g_cublas_lt.layout_destroy(plan.a_layout);
            if (plan.operation)
                g_cublas_lt.matmul_desc_destroy(plan.operation);
        };
        if (g_cublas_lt.matmul_desc_create(
                &plan.operation, kCompute32F, kCudaR32F) != 0 ||
            g_cublas_lt.layout_create(
                &plan.a_layout, kCudaR4E2M1, input_size, output_size,
                input_size) != 0 ||
            g_cublas_lt.layout_create(
                &plan.b_layout, kCudaR4E2M1, input_size, rows,
                input_size) != 0 ||
            g_cublas_lt.layout_create(
                &plan.c_layout, kCudaR32F, output_size, rows,
                output_size) != 0 ||
            g_cublas_lt.layout_create(
                &plan.d_layout, kCudaR32F, output_size, rows,
                output_size) != 0 ||
            g_cublas_lt.preference_create(&preference) != 0) {
            cleanup();
            return -1;
        }
        const void* a_scale_pointer = reinterpret_cast<const void*>(
            static_cast<std::uintptr_t>(weight_scales
                ? weight_scales : g_nvfp4_scratch.weight_scales));
        const void* b_scale_pointer = reinterpret_cast<const void*>(
            static_cast<std::uintptr_t>(input_scales
                ? input_scales : g_nvfp4_scratch.input_scales));
        size_t no_workspace = 0;
        if (g_cublas_lt.matmul_desc_set(
                plan.operation, kDescTransA, &kOpT, sizeof(kOpT)) != 0 ||
            g_cublas_lt.matmul_desc_set(
                plan.operation, kDescTransB, &kOpN, sizeof(kOpN)) != 0 ||
            g_cublas_lt.matmul_desc_set(
                plan.operation, kDescAScaleMode, &kScaleVec16Ue4m3,
                sizeof(kScaleVec16Ue4m3)) != 0 ||
            g_cublas_lt.matmul_desc_set(
                plan.operation, kDescBScaleMode, &kScaleVec16Ue4m3,
                sizeof(kScaleVec16Ue4m3)) != 0 ||
            g_cublas_lt.matmul_desc_set(
                plan.operation, kDescAScalePointer, &a_scale_pointer,
                sizeof(a_scale_pointer)) != 0 ||
            g_cublas_lt.matmul_desc_set(
                plan.operation, kDescBScalePointer, &b_scale_pointer,
                sizeof(b_scale_pointer)) != 0 ||
            g_cublas_lt.preference_set(
                preference, kPreferenceMaxWorkspace, &no_workspace,
                sizeof(no_workspace)) != 0) {
            cleanup();
            return -2;
        }
        CublasLtHeuristicResult result{};
        int result_count = 0;
        if (g_cublas_lt.heuristic(
                g_cublas_lt_handle, plan.operation, plan.a_layout,
                plan.b_layout, plan.c_layout, plan.d_layout, preference, 1,
                &result, &result_count) != 0 ||
            result_count != 1 || result.state != 0) {
            cleanup();
            return -3;
        }
        plan.algo = result.algo;
        g_cublas_lt.preference_destroy(preference);
        preference = nullptr;
        found = g_nvfp4_plans.emplace(key, plan).first;
    }
    const auto& plan = found->second;
    if (!weight_values) weight_values = g_nvfp4_scratch.weight_values;
    if (!weight_scales) weight_scales = g_nvfp4_scratch.weight_scales;
    if (!input_values) input_values = g_nvfp4_scratch.input_values;
    if (!input_scales) input_scales = g_nvfp4_scratch.input_scales;
    const void* a_scale_pointer = reinterpret_cast<const void*>(
        static_cast<std::uintptr_t>(weight_scales));
    const void* b_scale_pointer = reinterpret_cast<const void*>(
        static_cast<std::uintptr_t>(input_scales));
    if (g_cublas_lt.matmul_desc_set(
            plan.operation, kDescAScalePointer, &a_scale_pointer,
            sizeof(a_scale_pointer)) != 0 ||
        g_cublas_lt.matmul_desc_set(
            plan.operation, kDescBScalePointer, &b_scale_pointer,
            sizeof(b_scale_pointer)) != 0)
        return -5;
    const void* a = reinterpret_cast<const void*>(
        static_cast<std::uintptr_t>(weight_values));
    const void* b = reinterpret_cast<const void*>(
        static_cast<std::uintptr_t>(input_values));
    void* d = reinterpret_cast<void*>(static_cast<std::uintptr_t>(output));
    const int status = g_cublas_lt.matmul(
        g_cublas_lt_handle, plan.operation, &alpha, a, plan.a_layout, b,
        plan.b_layout, &beta, d, plan.c_layout, d, plan.d_layout, &plan.algo,
        nullptr, 0, cuda_stream);
    return status == 0 ? 0 : -4;
}

extern "C" int colibri_gpu_nvfp4_matmul_cublas(
    std::uint64_t weights, std::uint64_t input, std::uint64_t output,
    std::uint64_t stream, std::int32_t input_size,
    std::int32_t output_size, std::int32_t rows, float scale
) {
    // Not reachable through launch(), so the CPU backend cannot substitute a
    // host kernel for it: this bottoms out in cuBLAS directly. Report failure
    // so the runtime falls back to the kernel path it already has. Returning
    // success here silently skipped attention entirely for sequences of 128+
    // tokens, which is where cuBLAS attention becomes eligible.
    if (colibri_backend_is_cpu()) return -1;

    std::lock_guard<std::mutex> lock(g_cublas_mutex);
    if (!weights || !input || !output || input_size <= 0 || output_size <= 0
        || rows <= 0 || (input_size & 63) || (output_size & 15)
        || !load_cublas_lt()) return -1;
    const auto repack = g_functions.find("nvfp4_repack_cublaslt");
    const auto quantize = g_functions.find("nvfp4_quantize_cublaslt");
    if (repack == g_functions.end() || quantize == g_functions.end()) return -2;
    const auto cuda_stream = reinterpret_cast<CUstream>(stream);
    if (nvfp4_scratch_switch_stream(cuda_stream) != 0) return -3;

    const auto scale_bytes = [](std::int32_t outer, std::int32_t inner) {
        const size_t outer_tiles = (static_cast<size_t>(outer) + 127) / 128;
        const size_t inner_tiles =
            ((static_cast<size_t>(inner) + 15) / 16 + 3) / 4;
        return outer_tiles * inner_tiles * 512;
    };
    const size_t weight_value_bytes =
        static_cast<size_t>(output_size) * input_size / 2;
    const size_t weight_scale_bytes = scale_bytes(output_size, input_size);
    const size_t input_value_bytes =
        static_cast<size_t>(rows) * input_size / 2;
    const size_t input_scale_bytes = scale_bytes(rows, input_size);
    bool grow =
        weight_value_bytes > g_nvfp4_scratch.weight_values_bytes ||
        weight_scale_bytes > g_nvfp4_scratch.weight_scales_bytes ||
        input_value_bytes > g_nvfp4_scratch.input_values_bytes ||
        input_scale_bytes > g_nvfp4_scratch.input_scales_bytes;
    if (grow && g_nvfp4_scratch.stream != nullptr
        && g_api.cuStreamSynchronize(g_nvfp4_scratch.stream) != 0) return -4;
    if (grow) nvfp4_clear_cublas_plans();
    auto reserve = [&](CUdeviceptr& pointer, size_t& capacity, size_t bytes) {
        if (bytes <= capacity) return true;
        if (pointer && g_api.cuMemFree(pointer) != 0) return false;
        pointer = 0;
        capacity = 0;
        if (g_api.cuMemAlloc(&pointer, bytes) != 0) return false;
        capacity = bytes;
        return true;
    };
    if (!reserve(g_nvfp4_scratch.weight_values,
                 g_nvfp4_scratch.weight_values_bytes, weight_value_bytes) ||
        !reserve(g_nvfp4_scratch.weight_scales,
                 g_nvfp4_scratch.weight_scales_bytes, weight_scale_bytes) ||
        !reserve(g_nvfp4_scratch.input_values,
                 g_nvfp4_scratch.input_values_bytes, input_value_bytes) ||
        !reserve(g_nvfp4_scratch.input_scales,
                 g_nvfp4_scratch.input_scales_bytes, input_scale_bytes))
        return -5;
    g_nvfp4_scratch.stream = cuda_stream;

    std::uint64_t weight_values = g_nvfp4_scratch.weight_values;
    std::uint64_t weight_scales = g_nvfp4_scratch.weight_scales;
    void* repack_args[] = {
        &weights, &weight_values, &weight_scales, &output_size, &input_size};
    const unsigned int weight_blocks =
        static_cast<unsigned int>(
            (static_cast<std::uint64_t>(output_size) * input_size / 64 + 255)
            / 256);
    if (launch(repack->second, weight_blocks, 1, 256, repack_args, 0,
               cuda_stream) != 0) return -6;

    std::uint64_t input_values = g_nvfp4_scratch.input_values;
    std::uint64_t input_scales = g_nvfp4_scratch.input_scales;
    void* quantize_args[] = {
        &input, &input_values, &input_scales, &rows, &input_size};
    const unsigned int input_blocks =
        static_cast<unsigned int>(
            static_cast<std::uint64_t>(rows) * input_size / 16);
    if (launch(quantize->second, input_blocks, 1, 32, quantize_args, 0,
               cuda_stream) != 0) return -7;

    // Column-major view:
    //   A = W as KxM, op(A)=T; B = X as KxN; D = MxN.
    // D's column-major bytes are the runtime's row-major [N,M] output.
    // The descriptors, layouts, preference and heuristic query used to be
    // rebuilt and torn down on every call; nvfp4_run_quantized_gemm keeps a
    // plan per (K, M, N) -- the same cache the MoE path already uses, cleared
    // together with it when the scratch grows.
    const int status = nvfp4_run_quantized_gemm(
        input_size, output_size, rows, scale, 0.0f, output, cuda_stream,
        weight_values, weight_scales, input_values, input_scales);
    return status == 0 ? 0 : -8;
}

extern "C" int colibri_gpu_nvfp4_moe_cublas(
    std::uint64_t gate_pointers, std::uint64_t up_pointers,
    std::uint64_t down_pointers, std::uint64_t input,
    std::uint64_t activated, std::uint64_t output,
    std::uint64_t route_weights, std::uint64_t gate_scales,
    std::uint64_t up_scales, std::uint64_t down_scales,
    std::uint64_t stream,
    std::int32_t hidden_size, std::int32_t intermediate_size,
    std::int32_t experts
) {
    // Not reachable through launch(), so the CPU backend cannot substitute a
    // host kernel for it: this bottoms out in cuBLAS directly. Report failure
    // so the runtime falls back to the kernel path it already has. Returning
    // success here silently skipped attention entirely for sequences of 128+
    // tokens, which is where cuBLAS attention becomes eligible.
    if (colibri_backend_is_cpu()) return -1;

    std::lock_guard<std::mutex> lock(g_cublas_mutex);
    if (!gate_pointers || !up_pointers || !down_pointers || !input
        || !activated || !output || !route_weights || hidden_size <= 0
        || intermediate_size <= 0 || experts <= 0 || (hidden_size & 63)
        || (intermediate_size & 63) || !load_cublas_lt()) return -1;
    const bool profile =
        std::getenv("COLIBRI_NVFP4_TENSOR_CORE_PROFILE") != nullptr;
    if (profile && !g_nvfp4_moe_profile.initialized) {
        bool initialized = true;
        for (auto& event : g_nvfp4_moe_profile.events) {
            if (g_api.cuEventCreate(&event, 0) != 0) {
                initialized = false;
                break;
            }
        }
        g_nvfp4_moe_profile.initialized = initialized;
    }
    auto profile_record = [&](int index) {
        if (profile && g_nvfp4_moe_profile.initialized)
            g_api.cuEventRecord(g_nvfp4_moe_profile.events[index],
                                reinterpret_cast<CUstream>(stream));
    };
    const auto stacked_repack =
        g_functions.find("nvfp4_repack_stacked_moe_cublaslt");
    const auto swiglu = g_functions.find("nvfp4_stacked_moe_swiglu");
    const auto down_repack =
        g_functions.find("nvfp4_repack_concat_down_cublaslt");
    const auto input_quantize =
        g_functions.find("nvfp4_quantize_broadcast16_cublaslt");
    const auto down_quantize =
        g_functions.find("nvfp4_quantize_weighted_moe_cublaslt");
    const auto add_first = g_functions.find("nvfp4_moe_add_first_column");
    const auto validate_gate =
        g_functions.find("nvfp4_validate_stacked_projection");
    const auto validate_down =
        g_functions.find("nvfp4_validate_down_projection");
    if (stacked_repack == g_functions.end() || swiglu == g_functions.end()
        || down_repack == g_functions.end()
        || input_quantize == g_functions.end()
        || down_quantize == g_functions.end()
        || add_first == g_functions.end()) return -2;
    const bool validate =
        std::getenv("COLIBRI_NVFP4_TENSOR_CORE_VALIDATE")
        && !g_nvfp4_validation_done
        && validate_gate != g_functions.end()
        && validate_down != g_functions.end();
    const auto cuda_stream = reinterpret_cast<CUstream>(stream);
    if (nvfp4_scratch_switch_stream(cuda_stream) != 0) return -3;

    const int gate_rows = 2 * experts * intermediate_size;
    const int down_input = experts * intermediate_size;
    const size_t weight_value_bytes = std::max(
        static_cast<size_t>(gate_rows) * hidden_size / 2,
        static_cast<size_t>(hidden_size) * down_input / 2);
    const size_t weight_scale_bytes = std::max(
        nvfp4_cublas_scale_bytes(gate_rows, hidden_size),
        nvfp4_cublas_scale_bytes(hidden_size, down_input));
    const size_t input_value_bytes = std::max(
        static_cast<size_t>(16) * hidden_size / 2,
        static_cast<size_t>(16) * down_input / 2);
    const size_t input_scale_bytes = std::max(
        nvfp4_cublas_scale_bytes(16, hidden_size),
        nvfp4_cublas_scale_bytes(16, down_input));
    const size_t projected_bytes =
        static_cast<size_t>(gate_rows) * 16 * sizeof(float);
    const bool grow =
        weight_value_bytes > g_nvfp4_scratch.weight_values_bytes ||
        weight_scale_bytes > g_nvfp4_scratch.weight_scales_bytes ||
        input_value_bytes > g_nvfp4_scratch.input_values_bytes ||
        input_scale_bytes > g_nvfp4_scratch.input_scales_bytes ||
        projected_bytes > g_nvfp4_scratch.projected_bytes;
    if (grow && g_nvfp4_scratch.stream != nullptr
        && g_api.cuStreamSynchronize(g_nvfp4_scratch.stream) != 0) return -4;
    if (grow) nvfp4_clear_cublas_plans();
    auto reserve = [&](CUdeviceptr& pointer, size_t& capacity, size_t bytes) {
        if (bytes <= capacity) return true;
        if (pointer && g_api.cuMemFree(pointer) != 0) return false;
        pointer = 0;
        capacity = 0;
        if (g_api.cuMemAlloc(&pointer, bytes) != 0) return false;
        capacity = bytes;
        return true;
    };
    if (!reserve(g_nvfp4_scratch.weight_values,
                 g_nvfp4_scratch.weight_values_bytes, weight_value_bytes) ||
        !reserve(g_nvfp4_scratch.weight_scales,
                 g_nvfp4_scratch.weight_scales_bytes, weight_scale_bytes) ||
        !reserve(g_nvfp4_scratch.input_values,
                 g_nvfp4_scratch.input_values_bytes, input_value_bytes) ||
        !reserve(g_nvfp4_scratch.input_scales,
                 g_nvfp4_scratch.input_scales_bytes, input_scale_bytes) ||
        !reserve(g_nvfp4_scratch.projected,
                 g_nvfp4_scratch.projected_bytes, projected_bytes))
        return -5;
    g_nvfp4_scratch.stream = cuda_stream;
    profile_record(0);

    std::uint64_t weight_values = g_nvfp4_scratch.weight_values;
    std::uint64_t weight_scales = g_nvfp4_scratch.weight_scales;
    void* stacked_args[] = {
        &gate_pointers, &up_pointers, &weight_values, &weight_scales,
        &hidden_size, &intermediate_size, &experts};
    const unsigned int gate_blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(gate_rows) * hidden_size / 64 + 255) / 256);
    if (launch(stacked_repack->second, gate_blocks, 1, 256, stacked_args, 0,
               cuda_stream) != 0) return -6;
    profile_record(1);

    std::uint64_t input_values = g_nvfp4_scratch.input_values;
    std::uint64_t input_scales = g_nvfp4_scratch.input_scales;
    void* input_args[] = {
        &input, &input_values, &input_scales, &hidden_size};
    if (launch(input_quantize->second, hidden_size, 1, 32, input_args, 0,
               cuda_stream) != 0) return -7;
    profile_record(2);
    const std::uint64_t projected = g_nvfp4_scratch.projected;
    const int gate_gemm_status = nvfp4_run_quantized_gemm(
        hidden_size, gate_rows, 16, 1.0f, 0.0f, projected, cuda_stream);
    if (gate_gemm_status != 0) return -80 + gate_gemm_status;
    profile_record(3);
    if (validate) {
        std::uint64_t stats =
            projected + static_cast<std::uint64_t>(gate_rows) * sizeof(float);
        if (g_api.cuMemsetD8Async(stats, 0, 5 * sizeof(float), cuda_stream) == 0) {
            void* validate_args[] = {
                &gate_pointers, &up_pointers, &input,
                const_cast<std::uint64_t*>(&projected), &stats,
                &hidden_size, &intermediate_size, &experts};
            float host[5]{};
            if (launch(validate_gate->second, gate_rows, 1, 256,
                       validate_args, 0, cuda_stream) == 0
                && g_api.cuMemcpyDtoHAsync(
                       host, stats, sizeof(host), cuda_stream) == 0
                && g_api.cuStreamSynchronize(cuda_stream) == 0) {
                std::fprintf(
                    stderr,
                    "[nvfp4-validate] gate/up ref_max=%g tc_max=%g "
                    "max_abs_error=%g first_ref=%g first_tc=%g\n",
                    host[0], host[1], host[2], host[3], host[4]);
            }
        }
    }
    void* swiglu_args[] = {
        const_cast<std::uint64_t*>(&projected), &activated,
        &intermediate_size, &experts, &gate_scales, &up_scales,
        &down_scales};
    if (launch(swiglu->second,
               static_cast<unsigned int>(
                   (static_cast<std::uint64_t>(experts) * intermediate_size
                    + 255) / 256),
               1, 256, swiglu_args, 0, cuda_stream) != 0) return -9;
    profile_record(4);

    void* down_repack_args[] = {
        &down_pointers, &weight_values, &weight_scales, &intermediate_size,
        &hidden_size, &experts};
    const unsigned int down_blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(hidden_size) * down_input / 64 + 255)
        / 256);
    if (launch(down_repack->second, down_blocks, 1, 256, down_repack_args, 0,
               cuda_stream) != 0) return -10;
    profile_record(5);
    void* down_quantize_args[] = {
        &activated, &route_weights, &down_scales, &input_values, &input_scales,
        &intermediate_size, &experts};
    if (launch(down_quantize->second, down_input, 1, 32,
               down_quantize_args, 0, cuda_stream) != 0) return -11;
    profile_record(6);
    const int down_gemm_status = nvfp4_run_quantized_gemm(
        down_input, hidden_size, 16, 1.0f / 32768.0f, 0.0f, projected,
        cuda_stream);
    if (down_gemm_status != 0) return -120 + down_gemm_status;
    profile_record(7);
    if (validate) {
        std::uint64_t stats =
            projected + static_cast<std::uint64_t>(hidden_size) * sizeof(float);
        if (g_api.cuMemsetD8Async(stats, 0, 5 * sizeof(float), cuda_stream) == 0) {
            void* validate_args[] = {
                &down_pointers, &activated, &route_weights, &down_scales,
                const_cast<std::uint64_t*>(&projected), &stats,
                &intermediate_size, &hidden_size, &experts};
            float host[5]{};
            if (launch(validate_down->second, hidden_size, 1, 256,
                       validate_args, 0, cuda_stream) == 0
                && g_api.cuMemcpyDtoHAsync(
                       host, stats, sizeof(host), cuda_stream) == 0
                && g_api.cuStreamSynchronize(cuda_stream) == 0) {
                std::fprintf(
                    stderr,
                    "[nvfp4-validate] down ref_max=%g tc_max=%g "
                    "max_abs_error=%g first_ref=%g first_tc=%g\n",
                    host[0], host[1], host[2], host[3], host[4]);
            }
        }
        g_nvfp4_validation_done = true;
    }
    void* add_args[] = {
        const_cast<std::uint64_t*>(&projected), &output, &hidden_size};
    const int add_status = launch(
        add_first->second, static_cast<unsigned int>((hidden_size + 255) / 256),
        1, 256, add_args, 0, cuda_stream);
    if (add_status != 0) return -13;
    profile_record(8);
    if (profile && g_nvfp4_moe_profile.initialized
        && g_api.cuEventSynchronize(g_nvfp4_moe_profile.events[8]) == 0) {
        for (int phase = 0; phase < 8; ++phase) {
            float elapsed = 0.0f;
            if (g_api.cuEventElapsedTime(
                    &elapsed, g_nvfp4_moe_profile.events[phase],
                    g_nvfp4_moe_profile.events[phase + 1]) == 0)
                g_nvfp4_moe_profile.milliseconds[phase] += elapsed;
        }
        ++g_nvfp4_moe_profile.calls;
        if ((g_nvfp4_moe_profile.calls % 40) == 0) {
            static const char* names[8] = {
                "gate_repack", "input_quant", "gate_gemm", "swiglu",
                "down_repack", "down_quant", "down_gemm", "add"};
            std::fprintf(stderr, "[nvfp4-tc-profile] calls=%llu",
                         static_cast<unsigned long long>(
                             g_nvfp4_moe_profile.calls));
            for (int phase = 0; phase < 8; ++phase) {
                std::fprintf(
                    stderr, " %s_ms=%.4f", names[phase],
                    g_nvfp4_moe_profile.milliseconds[phase] / 40.0);
                g_nvfp4_moe_profile.milliseconds[phase] = 0.0;
            }
            std::fprintf(stderr, "\n");
        }
    }
    return 0;
}

static size_t nvfp4_native_expert_bytes(
    int hidden_size, int intermediate_size
) {
    const size_t gate_up_values =
        static_cast<size_t>(intermediate_size) * hidden_size;
    const size_t gate_up_scales =
        nvfp4_cublas_scale_bytes(2 * intermediate_size, hidden_size);
    const size_t down_values =
        static_cast<size_t>(hidden_size) * intermediate_size / 2;
    const size_t down_scales =
        nvfp4_cublas_scale_bytes(hidden_size, intermediate_size);
    return gate_up_values + gate_up_scales + down_values + down_scales;
}

extern "C" int colibri_gpu_nvfp4_prepare_expert(
    std::uint64_t gate, std::uint64_t up, std::uint64_t down,
    std::uint64_t native, std::uint64_t stream,
    std::int32_t hidden_size, std::int32_t intermediate_size
) {
    // Not reachable through launch(), so the CPU backend cannot substitute a
    // host kernel for it: this bottoms out in cuBLAS directly. Report failure
    // so the runtime falls back to the kernel path it already has. Returning
    // success here silently skipped attention entirely for sequences of 128+
    // tokens, which is where cuBLAS attention becomes eligible.
    if (colibri_backend_is_cpu()) return -1;

    if (!gate || !up || !down || !native || hidden_size <= 0
        || intermediate_size <= 0 || (hidden_size & 63)
        || (intermediate_size & 127)) return -1;
    const auto repack = g_functions.find("nvfp4_repack_cublaslt");
    if (repack == g_functions.end()) return -2;
    const auto cuda_stream = reinterpret_cast<CUstream>(stream);
    const size_t one_values =
        static_cast<size_t>(intermediate_size) * hidden_size / 2;
    const size_t one_scales =
        nvfp4_cublas_scale_bytes(intermediate_size, hidden_size);
    const size_t gate_up_values = 2 * one_values;
    const size_t gate_up_scales = 2 * one_scales;
    const size_t down_values =
        static_cast<size_t>(hidden_size) * intermediate_size / 2;
    const std::uint64_t gate_values = native;
    const std::uint64_t up_values = native + one_values;
    const std::uint64_t gate_scales = native + gate_up_values;
    const std::uint64_t up_scales = gate_scales + one_scales;
    const std::uint64_t down_values_ptr =
        native + gate_up_values + gate_up_scales;
    const std::uint64_t down_scales =
        down_values_ptr + down_values;
    auto run = [&](std::uint64_t source, std::uint64_t values,
                   std::uint64_t scales, int rows, int columns) {
        void* args[] = {
            &source, &values, &scales, &rows, &columns};
        const unsigned int blocks = static_cast<unsigned int>(
            (static_cast<std::uint64_t>(rows) * columns / 64 + 255) / 256);
        return launch(
            repack->second, blocks, 1, 256, args, 0, cuda_stream);
    };
    if (run(gate, gate_values, gate_scales,
            intermediate_size, hidden_size) != 0) return -3;
    if (run(up, up_values, up_scales,
            intermediate_size, hidden_size) != 0) return -4;
    if (run(down, down_values_ptr, down_scales,
            hidden_size, intermediate_size) != 0) return -5;
    return nvfp4_native_expert_bytes(hidden_size, intermediate_size) > 0
        ? 0 : -6;
}

extern "C" int colibri_gpu_nvfp4_moe_persistent(
    const std::uint64_t* native_experts, std::uint64_t route_weights,
    std::uint64_t gate_scales, std::uint64_t up_scales,
    std::uint64_t down_scales, std::uint64_t input,
    std::uint64_t activated, std::uint64_t output, std::uint64_t stream,
    std::int32_t hidden_size, std::int32_t intermediate_size,
    std::int32_t experts
) {
    // Not reachable through launch(), so the CPU backend cannot substitute a
    // host kernel for it: this bottoms out in cuBLAS directly. Report failure
    // so the runtime falls back to the kernel path it already has. Returning
    // success here silently skipped attention entirely for sequences of 128+
    // tokens, which is where cuBLAS attention becomes eligible.
    if (colibri_backend_is_cpu()) return -1;

    std::lock_guard<std::mutex> lock(g_cublas_mutex);
    if (!native_experts || !route_weights || !input || !activated || !output
        || hidden_size <= 0 || intermediate_size <= 0 || experts <= 0
        || (hidden_size & 63) || (intermediate_size & 127)
        || !load_cublas_lt()) return -1;
    const auto quantize =
        g_functions.find("nvfp4_quantize_broadcast16_cublaslt");
    const auto weighted_quantize =
        g_functions.find("nvfp4_quantize_weighted_moe_cublaslt");
    const auto swiglu =
        g_functions.find("nvfp4_persistent_moe_swiglu");
    const auto concat_gate =
        g_functions.find("nvfp4_concat_native_gate_up_cublaslt");
    const auto concat_down =
        g_functions.find("nvfp4_concat_native_down_cublaslt");
    const auto add_first =
        g_functions.find("nvfp4_moe_add_first_column");
    if (quantize == g_functions.end() ||
        weighted_quantize == g_functions.end() ||
        swiglu == g_functions.end() || add_first == g_functions.end())
        return -2;
    const char* grouped_env =
        std::getenv("COLIBRI_NVFP4_PERSISTENT_GROUPED");
    const bool grouped = grouped_env && grouped_env[0] == '1';
    if (grouped && (concat_gate == g_functions.end() ||
                    concat_down == g_functions.end())) return -2;
    const auto cuda_stream = reinterpret_cast<CUstream>(stream);
    const int max_input =
        std::max(hidden_size, experts * intermediate_size);
    const size_t input_value_bytes =
        static_cast<size_t>(16) * max_input / 2;
    const size_t input_scale_bytes =
        nvfp4_cublas_scale_bytes(16, max_input);
    const size_t gate_projected_bytes =
        static_cast<size_t>(experts) * 2 * intermediate_size * 16
        * sizeof(float);
    const size_t down_projected_bytes =
        static_cast<size_t>(hidden_size) * 16 * sizeof(float);
    const size_t projected_bytes =
        std::max(gate_projected_bytes, down_projected_bytes);
    const size_t one_values =
        static_cast<size_t>(intermediate_size) * hidden_size / 2;
    const size_t gate_up_values = 2 * one_values;
    const size_t gate_up_scales =
        nvfp4_cublas_scale_bytes(2 * intermediate_size, hidden_size);
    const size_t down_values =
        static_cast<size_t>(hidden_size) * intermediate_size / 2;
    const size_t gate_weight_values =
        static_cast<size_t>(experts) * gate_up_values;
    const size_t down_weight_values =
        static_cast<size_t>(hidden_size) * experts * intermediate_size / 2;
    const size_t weight_value_bytes =
        std::max(gate_weight_values, down_weight_values);
    const size_t gate_weight_scales =
        nvfp4_cublas_scale_bytes(
            experts * 2 * intermediate_size, hidden_size);
    const size_t down_weight_scales =
        nvfp4_cublas_scale_bytes(
            hidden_size, experts * intermediate_size);
    const size_t weight_scale_bytes =
        std::max(gate_weight_scales, down_weight_scales);
    const size_t expert_pointer_bytes =
        static_cast<size_t>(experts) * sizeof(std::uint64_t);
    const bool grow =
        weight_value_bytes > g_nvfp4_scratch.weight_values_bytes ||
        weight_scale_bytes > g_nvfp4_scratch.weight_scales_bytes ||
        input_value_bytes > g_nvfp4_scratch.input_values_bytes ||
        input_scale_bytes > g_nvfp4_scratch.input_scales_bytes ||
        projected_bytes > g_nvfp4_scratch.projected_bytes ||
        expert_pointer_bytes > g_nvfp4_scratch.expert_pointers_bytes;
    if (grow && g_nvfp4_scratch.stream != nullptr
        && g_api.cuStreamSynchronize(g_nvfp4_scratch.stream) != 0) return -3;
    if (grow) nvfp4_clear_cublas_plans();
    auto reserve = [&](CUdeviceptr& pointer, size_t& capacity, size_t bytes) {
        if (bytes <= capacity) return true;
        if (pointer && g_api.cuMemFree(pointer) != 0) return false;
        pointer = 0;
        capacity = 0;
        if (g_api.cuMemAlloc(&pointer, bytes) != 0) return false;
        capacity = bytes;
        return true;
    };
    if (!reserve(g_nvfp4_scratch.weight_values,
                 g_nvfp4_scratch.weight_values_bytes, weight_value_bytes) ||
        !reserve(g_nvfp4_scratch.weight_scales,
                 g_nvfp4_scratch.weight_scales_bytes, weight_scale_bytes) ||
        !reserve(g_nvfp4_scratch.input_values,
                 g_nvfp4_scratch.input_values_bytes, input_value_bytes) ||
        !reserve(g_nvfp4_scratch.input_scales,
                 g_nvfp4_scratch.input_scales_bytes, input_scale_bytes) ||
        !reserve(g_nvfp4_scratch.projected,
                 g_nvfp4_scratch.projected_bytes, projected_bytes) ||
        !reserve(g_nvfp4_scratch.expert_pointers,
                 g_nvfp4_scratch.expert_pointers_bytes,
                 expert_pointer_bytes))
        return -4;
    g_nvfp4_scratch.stream = cuda_stream;

    std::uint64_t weight_values = g_nvfp4_scratch.weight_values;
    std::uint64_t weight_scales = g_nvfp4_scratch.weight_scales;
    std::uint64_t input_values = g_nvfp4_scratch.input_values;
    std::uint64_t input_scales = g_nvfp4_scratch.input_scales;
    std::uint64_t projected = g_nvfp4_scratch.projected;
    std::uint64_t expert_pointers = g_nvfp4_scratch.expert_pointers;
    if (grouped) {
        if (g_api.cuMemcpyHtoDAsync(
                g_nvfp4_scratch.expert_pointers, native_experts,
                expert_pointer_bytes, cuda_stream) != 0) return -5;
        std::uint64_t gate_value_bytes_arg = gate_up_values;
        std::uint64_t gate_scale_bytes_arg = gate_up_scales;
        void* concat_gate_args[] = {
            &expert_pointers, &weight_values, &weight_scales,
            &gate_value_bytes_arg, &gate_scale_bytes_arg, &experts};
        const size_t gate_copy_bytes =
            std::max(gate_weight_values,
                     static_cast<size_t>(experts) * gate_up_scales);
        if (launch(
                concat_gate->second,
                static_cast<unsigned int>((gate_copy_bytes + 255) / 256),
                1, 256, concat_gate_args, 0, cuda_stream) != 0) return -6;
    }
    void* input_args[] = {
        &input, &input_values, &input_scales, &hidden_size};
    if (launch(quantize->second, hidden_size, 1, 32, input_args, 0,
               cuda_stream) != 0) return -7;
    if (grouped) {
        const int gate_status = nvfp4_run_quantized_gemm(
            hidden_size, experts * 2 * intermediate_size, 16, 1.0f, 0.0f,
            projected, cuda_stream, weight_values, weight_scales,
            input_values, input_scales);
        if (gate_status != 0) return -20 + gate_status;
    } else {
        for (int expert = 0; expert < experts; ++expert) {
            const std::uint64_t expert_values = native_experts[expert];
            const std::uint64_t expert_scales =
                expert_values + gate_up_values;
            const std::uint64_t expert_output = projected
                + static_cast<std::uint64_t>(expert) *
                  2 * intermediate_size * 16 * sizeof(float);
            const int status = nvfp4_run_quantized_gemm(
                hidden_size, 2 * intermediate_size, 16, 1.0f, 0.0f,
                expert_output, cuda_stream, expert_values, expert_scales,
                input_values, input_scales);
            if (status != 0) return -20 + status;
        }
    }
    int grouped_layout = grouped ? 1 : 0;
    void* swiglu_args[] = {
        &projected, &activated, &intermediate_size, &experts,
        &gate_scales, &up_scales, &down_scales, &grouped_layout};
    if (launch(
            swiglu->second,
            static_cast<unsigned int>(
                (static_cast<std::uint64_t>(experts) * intermediate_size
                 + 255) / 256),
            1, 256, swiglu_args, 0, cuda_stream) != 0) return -8;

    const float alpha = 1.0f / 32768.0f;
    if (grouped) {
        void* down_input_args[] = {
            &activated, &route_weights, &down_scales, &input_values,
            &input_scales, &intermediate_size, &experts};
        const unsigned int down_input_blocks =
            static_cast<unsigned int>(
                static_cast<std::uint64_t>(experts) * intermediate_size);
        if (launch(weighted_quantize->second, down_input_blocks, 1, 32,
                   down_input_args, 0, cuda_stream) != 0) return -9;
        std::uint64_t down_offset = gate_up_values + gate_up_scales;
        std::uint64_t down_scale_offset = down_offset + down_values;
        void* concat_down_args[] = {
            &expert_pointers, &weight_values, &weight_scales,
            &intermediate_size, &hidden_size, &experts,
            &down_offset, &down_scale_offset};
        const size_t down_scale_items =
            static_cast<size_t>(hidden_size) * experts *
            (intermediate_size / 16);
        const size_t down_copy_items =
            std::max(down_weight_values, down_scale_items);
        if (launch(
                concat_down->second,
                static_cast<unsigned int>((down_copy_items + 255) / 256),
                1, 256, concat_down_args, 0, cuda_stream) != 0) return -10;
        const int down_status = nvfp4_run_quantized_gemm(
            experts * intermediate_size, hidden_size, 16, alpha, 0.0f,
            projected, cuda_stream, weight_values, weight_scales,
            input_values, input_scales);
        if (down_status != 0) return -40 + down_status;
    } else {
        for (int expert = 0; expert < experts; ++expert) {
            std::uint64_t expert_input = activated
                + static_cast<std::uint64_t>(expert) * intermediate_size *
                  sizeof(float);
            std::uint64_t expert_weight =
                route_weights +
                static_cast<std::uint64_t>(expert) * sizeof(float);
            std::uint64_t expert_down_scale = down_scales
                ? down_scales +
                    static_cast<std::uint64_t>(expert) * sizeof(float) : 0;
            int one_expert = 1;
            void* down_input_args[] = {
                &expert_input, &expert_weight, &expert_down_scale,
                &input_values, &input_scales, &intermediate_size,
                &one_expert};
            if (launch(weighted_quantize->second, intermediate_size, 1, 32,
                       down_input_args, 0, cuda_stream) != 0) return -9;
            const std::uint64_t expert_values = native_experts[expert]
                + gate_up_values + gate_up_scales;
            const std::uint64_t expert_scales =
                expert_values + down_values;
            const float beta = expert == 0 ? 0.0f : 1.0f;
            const int status = nvfp4_run_quantized_gemm(
                intermediate_size, hidden_size, 16, alpha, beta,
                projected, cuda_stream, expert_values, expert_scales,
                input_values, input_scales);
            if (status != 0) return -40 + status;
        }
    }
    void* add_args[] = {&projected, &output, &hidden_size};
    return launch(
        add_first->second, static_cast<unsigned int>((hidden_size + 255) / 256),
        1, 256, add_args, 0, cuda_stream) == 0 ? 0 : -11;
}

extern "C" int colibri_gpu_launch_named(
    const char* name, std::uint32_t grid_x, std::uint32_t grid_y,
    std::uint32_t block_x, std::uint32_t shared_bytes,
    std::uint64_t stream, void** arguments
) {
    if (name == nullptr || arguments == nullptr || grid_x == 0 || grid_y == 0
        || block_x == 0) return -1;
    const auto found = g_functions.find(name);
    if (found == g_functions.end()) return -2;
    return launch(found->second, grid_x, grid_y, block_x, arguments,
                  shared_bytes, reinterpret_cast<CUstream>(stream)) == 0
        ? 0 : -3;
}

extern "C" int colibri_gpu_attention_f16_cublas(
    std::uint64_t query, std::uint64_t query_f16,
    std::uint64_t keys, std::uint64_t values,
    std::uint64_t scores_f16, std::uint64_t output,
    std::uint64_t stream, std::int32_t heads, std::int32_t kv_heads,
    std::int32_t head_dim, std::int32_t tokens, std::int32_t capacity,
    std::int32_t first, float scale
) {
    // Not reachable through launch(), so the CPU backend cannot substitute a
    // host kernel for it: this bottoms out in cuBLAS directly. Report failure
    // so the runtime falls back to the kernel path it already has. Returning
    // success here silently skipped attention entirely for sequences of 128+
    // tokens, which is where cuBLAS attention becomes eligible.
    if (colibri_backend_is_cpu()) return -1;

    std::lock_guard<std::mutex> lock(g_cublas_mutex);
    if (!query || !query_f16 || !keys || !values || !scores_f16 || !output
        || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0
        || head_dim <= 0 || tokens <= 0 || capacity < tokens || first < 0
        || first + tokens > capacity || !load_cublas())
        return -1;
    const auto cuda_stream = reinterpret_cast<CUstream>(stream);
    if (g_cublas.set_stream(g_cublas_handle, cuda_stream) != 0) return -2;
    const int group = heads / kv_heads;
    const int query_elements = heads * head_dim;
    auto conversion = g_functions.find("qwen_attention_query_f16");
    auto softmax = g_functions.find("kv_attention_softmax_f16");
    if (conversion == g_functions.end() || softmax == g_functions.end())
        return -3;
    void* conversion_args[] = {&query, &query_f16,
                               const_cast<int*>(&query_elements)};
    if (launch(conversion->second, (query_elements + 255) / 256, 1, 256,
               conversion_args, 0, cuda_stream) != 0)
        return -4;

    constexpr int kCudaR16F = 2;
    constexpr int kCudaR32F = 0;
    constexpr int kCublasOpN = 0;
    constexpr int kCublasOpT = 1;
    constexpr int kCompute32F = 68;
    constexpr int kTensorOp = 99;
    const float zero = 0.0f;
    const auto key_base = reinterpret_cast<const void*>(
        keys + static_cast<std::uint64_t>(first) * head_dim * sizeof(std::uint16_t)
    );
    const long long cache_stride =
        static_cast<long long>(capacity) * head_dim;
    const long long query_stride = static_cast<long long>(group) * head_dim;
    const long long score_stride = static_cast<long long>(tokens) * group;
    if (g_cublas.gemm_strided_batched_ex(
            g_cublas_handle, kCublasOpT, kCublasOpN,
            tokens, group, head_dim, &scale,
            key_base, kCudaR16F, head_dim, cache_stride,
            reinterpret_cast<const void*>(query_f16), kCudaR16F, head_dim,
            query_stride, &zero, reinterpret_cast<void*>(scores_f16),
            kCudaR16F, tokens, score_stride, kv_heads,
            kCompute32F, kTensorOp) != 0)
        return -5;
    void* softmax_args[] = {&scores_f16, &heads, &tokens};
    if (launch(softmax->second, heads, 1, 256, softmax_args, 0,
               cuda_stream) != 0)
        return -6;

    const float one = 1.0f;
    const auto value_base = reinterpret_cast<const void*>(
        values + static_cast<std::uint64_t>(first) * head_dim * sizeof(std::uint16_t)
    );
    if (g_cublas.gemm_strided_batched_ex(
            g_cublas_handle, kCublasOpN, kCublasOpN,
            head_dim, group, tokens, &one,
            value_base, kCudaR16F, head_dim, cache_stride,
            reinterpret_cast<const void*>(scores_f16), kCudaR16F, tokens,
            score_stride, &zero, reinterpret_cast<void*>(output),
            kCudaR32F, head_dim, query_stride, kv_heads,
            kCompute32F, kTensorOp) != 0)
        return -7;
    return 0;
}

extern "C" int colibri_gpu_attention_prefill_f16_cublas(
    std::uint64_t queries, std::uint64_t gates,
    std::uint64_t keys, std::uint64_t values,
    std::uint64_t packed_queries, std::uint64_t scores_f32,
    std::uint64_t probabilities_f16,
    std::uint64_t packed_output, std::uint64_t flash_state,
    std::uint64_t output,
    std::uint64_t stream, std::int32_t heads, std::int32_t kv_heads,
    std::int32_t head_dim, std::int32_t rows, std::int32_t capacity,
    std::int32_t base_position, std::int32_t tile_rows_limit,
    std::int32_t block_tokens, float scale,
    std::int32_t apply_gate
) {
    // Not reachable through launch(), so the CPU backend cannot substitute a
    // host kernel for it: this bottoms out in cuBLAS directly. Report failure
    // so the runtime falls back to the kernel path it already has. Returning
    // success here silently skipped attention entirely for sequences of 128+
    // tokens, which is where cuBLAS attention becomes eligible.
    if (colibri_backend_is_cpu()) return -1;

    std::lock_guard<std::mutex> lock(g_cublas_mutex);
    if (!queries || !gates || !keys || !values || !packed_queries
        || !scores_f32 || !probabilities_f16 || !packed_output || !output
        || !flash_state || heads <= 0
        || kv_heads <= 0 || heads % kv_heads != 0 || head_dim <= 0
        || rows <= 0 || capacity <= 0 || base_position < 0
        || base_position + rows > capacity || tile_rows_limit <= 0
        || block_tokens <= 0 || !load_cublas())
        return -1;
    const auto cuda_stream = reinterpret_cast<CUstream>(stream);
    if (g_cublas.set_stream(g_cublas_handle, cuda_stream) != 0) return -2;
    auto pack = g_functions.find("qwen_attention_prefill_pack_f16");
    auto softmax = g_functions.find("kv_attention_prefill_block_softmax_f16");
    auto rescale_fn = g_functions.find("qwen_attention_prefill_rescale");
    // Turbo values arrive rotated, so that caller takes the ungated variant and
    // applies the gate itself after the inverse rotation.
    auto unpack = g_functions.find(
        apply_gate ? "qwen_attention_prefill_unpack_gate_norm"
                   : "qwen_attention_prefill_unpack_norm");
    if (pack == g_functions.end() || softmax == g_functions.end()
        || rescale_fn == g_functions.end() || unpack == g_functions.end())
        return -3;

    constexpr int kCudaR16F = 2;
    constexpr int kCudaR32F = 0;
    constexpr int kCublasOpN = 0;
    constexpr int kCublasOpT = 1;
    constexpr int kCompute32F = 68;
    constexpr int kTensorOp = 99;
    const int group = heads / kv_heads;
    const long long cache_stride =
        static_cast<long long>(capacity) * head_dim;
    const float zero = 0.0f;
    const float one = 1.0f;
    const int tile_limit = std::min(16, tile_rows_limit);
    // The rescale factors live behind the (M, S) pairs in the state buffer.
    const std::uint64_t rescale_buffer =
        flash_state + static_cast<std::uint64_t>(tile_limit) * heads * 2
            * sizeof(float);
    for (int tile_start = 0; tile_start < rows; tile_start += tile_limit) {
        int tile_rows = std::min(tile_limit, rows - tile_start);
        const int columns = tile_rows * group;
        const int query_elements = tile_rows * heads * head_dim;
        void* pack_args[] = {
            &queries, &packed_queries, &tile_start, &tile_rows,
            &heads, &kv_heads, &head_dim
        };
        if (launch(pack->second, (query_elements + 255) / 256, 1, 256,
                   pack_args, 0, cuda_stream) != 0)
            return -4;
        // The visible prefix is walked in position blocks so the materialized
        // score tile is bounded by `block_tokens` rather than the context.
        // The un-blocked form shrank the query tile to fit `tokens` in the
        // score workspace, which at 70k context meant streaming the whole KV
        // cache once per 3 query rows; here the tile stays at 16 and the
        // running max/denominator carries the softmax across blocks.
        const int tokens = base_position + tile_start + tile_rows;
        const long long query_stride =
            static_cast<long long>(columns) * head_dim;
        const long long output_stride =
            static_cast<long long>(head_dim) * columns;
        for (int block_start = 0; block_start < tokens;
             block_start += block_tokens) {
            const int block = std::min(block_tokens, tokens - block_start);
            const long long score_stride =
                static_cast<long long>(block) * columns;
            const std::uint64_t block_keys = keys
                + static_cast<std::uint64_t>(block_start) * head_dim
                    * sizeof(std::uint16_t);
            const std::uint64_t block_values = values
                + static_cast<std::uint64_t>(block_start) * head_dim
                    * sizeof(std::uint16_t);
            if (g_cublas.gemm_strided_batched_ex(
                    g_cublas_handle, kCublasOpT, kCublasOpN,
                    block, columns, head_dim, &scale,
                    reinterpret_cast<const void*>(block_keys), kCudaR16F,
                    head_dim, cache_stride,
                    reinterpret_cast<const void*>(packed_queries), kCudaR16F,
                    head_dim, query_stride, &zero,
                    reinterpret_cast<void*>(scores_f32), kCudaR32F, block,
                    score_stride, kv_heads, kCompute32F, kTensorOp) != 0)
                return -5;
            void* softmax_args[] = {
                &scores_f32, &probabilities_f16,
                const_cast<std::uint64_t*>(&flash_state),
                const_cast<std::uint64_t*>(&rescale_buffer),
                &tile_start, &tile_rows, &heads, &kv_heads,
                const_cast<int*>(&block_start), const_cast<int*>(&block),
                &base_position
            };
            if (launch(softmax->second, heads, tile_rows, 256, softmax_args, 0,
                       cuda_stream) != 0)
                return -6;
            const bool first = block_start == 0;
            if (!first) {
                void* rescale_args[] = {
                    const_cast<std::uint64_t*>(&packed_output),
                    const_cast<std::uint64_t*>(&rescale_buffer),
                    &tile_rows, &heads, &kv_heads, &head_dim
                };
                if (launch(rescale_fn->second, (query_elements + 255) / 256, 1,
                           256, rescale_args, 0, cuda_stream) != 0)
                    return -7;
            }
            if (g_cublas.gemm_strided_batched_ex(
                    g_cublas_handle, kCublasOpN, kCublasOpN,
                    head_dim, columns, block, &one,
                    reinterpret_cast<const void*>(block_values), kCudaR16F,
                    head_dim, cache_stride,
                    reinterpret_cast<const void*>(probabilities_f16), kCudaR16F,
                    block,
                    score_stride, first ? &zero : &one,
                    reinterpret_cast<void*>(packed_output), kCudaR32F, head_dim,
                    output_stride, kv_heads, kCompute32F, kTensorOp) != 0)
                return -8;
        }
        void* unpack_args[] = {
            &packed_output, &gates, const_cast<std::uint64_t*>(&flash_state),
            &output, &tile_start, &tile_rows,
            &heads, &kv_heads, &head_dim
        };
        if (launch(unpack->second, (query_elements + 255) / 256, 1, 256,
                   unpack_args, 0, cuda_stream) != 0)
            return -9;
    }
    return 0;
}

extern "C" int colibri_delta_moe_segment(
    const ColibriDeltaParams* params,
    const ColibriDeltaLayer* layers,
    std::int32_t count
) {
    if (params == nullptr || layers == nullptr || count <= 0) {
        return -1;
    }
    if (g_kernels.rms_norm == nullptr || g_kernels.q4_matvec == nullptr
        || g_kernels.bf16_matvec == nullptr || g_kernels.delta_conv == nullptr
        || g_kernels.delta_recurrent == nullptr
        || g_kernels.scaled_add == nullptr) {
        return -2;
    }
    const std::int32_t hidden_size = params->hidden_size;
    const std::size_t hidden_bytes =
        static_cast<std::size_t>(hidden_size) * sizeof(float);
    const std::int32_t router_rows = params->num_experts + 1;
    std::vector<float> logits(params->num_experts);
    std::vector<std::int32_t> selected(params->top_k);
    std::vector<float> weights(params->top_k + 1);
    std::vector<const std::uint8_t*> gate_packed(params->top_k + 1);
    std::vector<const std::uint16_t*> gate_scales(params->top_k + 1);
    std::vector<const std::uint8_t*> down_packed(params->top_k + 1);
    std::vector<const std::uint16_t*> down_scales(params->top_k + 1);

    // The incoming hidden state is written by the caller on the legacy null
    // stream; drain it once so graph replays on the capture stream see it.
    if (g_api.cuStreamSynchronize(nullptr) != 0) {
        return -3;
    }
    for (std::int32_t index = 0; index < count; ++index) {
        phase_mark(0);
        const ColibriDeltaLayer& layer = layers[index];
        if (layer.graph != 0) {
            if (g_api.cuGraphLaunch(
                    reinterpret_cast<void*>(layer.graph), nullptr
                ) != 0) {
                return -3;
            }
        } else {
            const int enqueued = enqueue_layer(params, layer, nullptr);
            if (enqueued != 0) {
                return enqueued;
            }
        }
        phase_mark(1);
        if (params->bundle_floats > 0) {
            if (g_api.cuMemcpyDtoH(
                    params->hidden_host, params->mixed,
                    static_cast<std::size_t>(params->bundle_floats)
                        * sizeof(float)
                ) != 0) {
                return -4;
            }
        } else if (
            g_api.cuMemcpyDtoH(params->hidden_host, params->mixed, hidden_bytes)
                != 0
            || g_api.cuMemcpyDtoH(
                   params->normalized_host, params->moe_normalized,
                   hidden_bytes
               ) != 0
            || g_api.cuMemcpyDtoH(
                   params->logits_host, params->router_logits,
                   static_cast<std::size_t>(router_rows) * sizeof(float)
               ) != 0) {
            return -4;
        }
        phase_mark(2);

        // Host: top-k over the router logits, fused Q4 experts, residual.
        for (std::int32_t expert = 0; expert < params->num_experts; ++expert) {
            logits[expert] = params->logits_host[expert];
        }
        float max_logit = -1e30f;
        for (std::int32_t expert = 0; expert < params->num_experts; ++expert) {
            if (logits[expert] > max_logit) {
                max_logit = logits[expert];
            }
        }
        for (std::int32_t expert = 0; expert < params->num_experts; ++expert) {
            logits[expert] = std::exp(logits[expert] - max_logit);
        }
        float selected_total = 0.0f;
        for (std::int32_t rank = 0; rank < params->top_k; ++rank) {
            std::int32_t best = -1;
            float best_value = -1.0f;
            for (std::int32_t expert = 0; expert < params->num_experts;
                 ++expert) {
                bool taken = false;
                for (std::int32_t previous = 0; previous < rank; ++previous) {
                    if (selected[previous] == expert) {
                        taken = true;
                        break;
                    }
                }
                if (!taken && logits[expert] > best_value) {
                    best_value = logits[expert];
                    best = expert;
                }
            }
            if (best < 0) {
                return -7;
            }
            selected[rank] = best;
            weights[rank] = best_value;
            selected_total += best_value;
        }
        for (std::int32_t rank = 0; rank < params->top_k; ++rank) {
            weights[rank] /= selected_total;
            gate_packed[rank] = layer.expert_gate_packed[selected[rank]];
            gate_scales[rank] = layer.expert_gate_scales[selected[rank]];
            down_packed[rank] = layer.expert_down_packed[selected[rank]];
            down_scales[rank] = layer.expert_down_scales[selected[rank]];
        }
        const float shared_logit = params->logits_host[params->num_experts];
        weights[params->top_k] = 1.0f / (1.0f + std::exp(-shared_logit));
        gate_packed[params->top_k] = layer.shared_gate_up_packed;
        gate_scales[params->top_k] = layer.shared_gate_up_scales;
        down_packed[params->top_k] = layer.shared_down_packed;
        down_scales[params->top_k] = layer.shared_down_scales;
        phase_mark(3);
        const int moe_status = colibri_q4_moe(
            gate_packed.data(),
            gate_scales.data(),
            down_packed.data(),
            down_scales.data(),
            weights.data(),
            params->normalized_host,
            params->moe_host,
            params->top_k + 1,
            hidden_size,
            params->moe_intermediate
        );
        if (moe_status != 0) {
            return -5;
        }
        phase_mark(4);
        for (std::int32_t column = 0; column < hidden_size; ++column) {
            params->moe_host[column] += params->hidden_host[column];
        }
        if (g_api.cuMemcpyHtoD(params->hidden, params->moe_host, hidden_bytes)
            != 0) {
            return -6;
        }
        phase_mark(5);
    }
    if (std::getenv("COLIBRI_SEG_DEBUG") != nullptr) {
        std::fprintf(
            stderr,
            "[phases ms] gpu=%.3f copy=%.3f topk=%.3f experts=%.3f wb=%.3f\n",
            g_phase_totals[1], g_phase_totals[2], g_phase_totals[3],
            g_phase_totals[4], g_phase_totals[5]
        );
        for (int i = 0; i < 6; ++i) g_phase_totals[i] = 0.0;
    }
    return 0;
}

extern "C" int colibri_delta_graph_build(
    const ColibriDeltaParams* params,
    const ColibriDeltaLayer* layer,
    std::uint64_t* handle
) {
    if (params == nullptr || layer == nullptr || handle == nullptr) {
        return -1;
    }
    if (g_api.cuStreamBeginCapture == nullptr) {
        return -2;
    }
    if (g_stream == nullptr
        && g_api.cuStreamCreate(&g_stream, 1 /* non-blocking */) != 0) {
        return -2;
    }
    if (g_api.cuStreamBeginCapture(g_stream, 0 /* global */) != 0) {
        return -3;
    }
    const int enqueued = enqueue_layer(params, *layer, g_stream);
    void* graph = nullptr;
    if (g_api.cuStreamEndCapture(g_stream, &graph) != 0 || enqueued != 0) {
        if (graph != nullptr) {
            g_api.cuGraphDestroy(graph);
        }
        return enqueued != 0 ? enqueued : -4;
    }
    void* executable = nullptr;
    const int instantiated =
        g_api.cuGraphInstantiateWithFlags(&executable, graph, 0);
    g_api.cuGraphDestroy(graph);
    if (instantiated != 0) {
        return -5;
    }
    *handle = reinterpret_cast<std::uint64_t>(executable);
    return 0;
}

extern "C" int colibri_delta_graph_destroy(std::uint64_t handle) {
    if (handle == 0 || g_api.cuGraphExecDestroy == nullptr) {
        return -1;
    }
    return g_api.cuGraphExecDestroy(reinterpret_cast<void*>(handle)) == 0
        ? 0
        : -2;
}
