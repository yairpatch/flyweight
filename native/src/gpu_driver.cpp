// Optional CUDA driving from the native library. libcuda and libnvrtc are
// dlopen'd at runtime, so the library builds and runs without any CUDA
// dependency; the GPU entry points simply report unavailability. The driver
// retains the device's primary context (the same one CuPy's runtime API
// uses) and launches on the legacy default stream, so device pointers taken
// from CuPy arrays are valid here and ordering with CuPy-issued work is
// automatic.
#include "colibri_native.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = unsigned long long;

using nvrtcResult = int;
using nvrtcProgram = void*;

struct CudaApi {
    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
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
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuStreamCreate)(CUstream*, unsigned int) = nullptr;
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
};

Kernels g_kernels;

template <typename T>
bool load_symbol(void* library, const char* name, T& target) {
#if defined(_WIN32)
    (void)library;
    (void)name;
    (void)target;
    return false;
#else
    target = reinterpret_cast<T>(dlsym(library, name));
    return target != nullptr;
#endif
}

bool load_apis() {
#if defined(_WIN32)
    return false;
#else
    if (g_api.loaded) {
        return true;
    }
    void* cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (cuda == nullptr) {
        cuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    }
    void* nvrtc = dlopen("libnvrtc.so", RTLD_NOW | RTLD_GLOBAL);
    if (cuda == nullptr || nvrtc == nullptr) {
        return false;
    }
    bool ok = true;
    ok &= load_symbol(cuda, "cuInit", g_api.cuInit);
    ok &= load_symbol(
        cuda, "cuDevicePrimaryCtxRetain", g_api.cuDevicePrimaryCtxRetain
    );
    ok &= load_symbol(cuda, "cuCtxSetCurrent", g_api.cuCtxSetCurrent);
    ok &= load_symbol(
        cuda, "cuDeviceGetAttribute", g_api.cuDeviceGetAttribute
    );
    ok &= load_symbol(cuda, "cuModuleLoadDataEx", g_api.cuModuleLoadDataEx);
    ok &= load_symbol(cuda, "cuModuleGetFunction", g_api.cuModuleGetFunction);
    ok &= load_symbol(cuda, "cuLaunchKernel", g_api.cuLaunchKernel);
    ok &= load_symbol(cuda, "cuMemcpyDtoH_v2", g_api.cuMemcpyDtoH);
    ok &= load_symbol(cuda, "cuMemcpyHtoD_v2", g_api.cuMemcpyHtoD);
    ok &= load_symbol(cuda, "cuStreamSynchronize", g_api.cuStreamSynchronize);
    ok &= load_symbol(cuda, "cuStreamCreate", g_api.cuStreamCreate);
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
#endif
}

// Phase profiling under COLIBRI_SEG_DEBUG: 0=start 1=mixer 2=route 3=copies
// 4=experts 5=writeback.
double g_phase_totals[6] = {};
timespec g_phase_last = {};

void phase_mark(int phase) {
    static const bool enabled =
        std::getenv("COLIBRI_SEG_DEBUG") != nullptr;
    if (!enabled) {
        return;
    }
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (phase > 0) {
        g_phase_totals[phase] +=
            (now.tv_sec - g_phase_last.tv_sec) * 1e3
            + (now.tv_nsec - g_phase_last.tv_nsec) / 1e6;
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
    return load_apis() ? 1 : 0;
}

extern "C" int colibri_gpu_init(std::int32_t device) {
    if (!load_apis()) {
        return -1;
    }
    if (g_api.cuInit(0) != 0) {
        return -2;
    }
    if (g_api.cuDevicePrimaryCtxRetain(&g_context, device) != 0) {
        return -3;
    }
    if (g_api.cuCtxSetCurrent(g_context) != 0) {
        return -4;
    }
    return 0;
}

extern "C" int colibri_gpu_compile(
    const char* source,
    const char* const* options,
    std::int32_t option_count,
    std::int32_t device,
    char* log_buffer,
    std::int32_t log_capacity
) {
    if (!g_api.loaded || g_context == nullptr) {
        return -1;
    }
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
    for (std::int32_t index = 0; index < option_count; ++index) {
        all_options.push_back(options[index]);
    }
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
    g_api.nvrtcGetPTXSize(program, &ptx_size);
    std::vector<char> ptx(ptx_size);
    g_api.nvrtcGetPTX(program, ptx.data());
    g_api.nvrtcDestroyProgram(&program);
    if (g_api.cuModuleLoadDataEx(&g_module, ptx.data(), 0, nullptr, nullptr)
        != 0) {
        return -4;
    }
    struct Entry {
        const char* name;
        CUfunction* slot;
    };
    const Entry entries[] = {
        {"rms_norm", &g_kernels.rms_norm},
        {"q4_matvec", &g_kernels.q4_matvec},
        {"bf16_matvec", &g_kernels.bf16_matvec},
        {"delta_conv_sequence", &g_kernels.delta_conv},
        {"delta_conv_step", &g_kernels.delta_conv_step},
        {"delta_recurrent_sequence", &g_kernels.delta_recurrent},
        {"scaled_add", &g_kernels.scaled_add},
    };
    for (const Entry& entry : entries) {
        if (g_api.cuModuleGetFunction(entry.slot, g_module, entry.name) != 0) {
            return -5;
        }
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

extern "C" int colibri_gpu_sync() {
    if (!g_api.loaded) {
        return -1;
    }
    return g_api.cuStreamSynchronize(nullptr) == 0 ? 0 : -2;
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
