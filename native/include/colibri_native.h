#pragma once

#include <cstdint>

#if defined(_WIN32)
#if defined(COLIBRI_NATIVE_BUILD)
#define COLIBRI_API __declspec(dllexport)
#else
#define COLIBRI_API __declspec(dllimport)
#endif
#else
#define COLIBRI_API __attribute__((visibility("default")))
#endif

extern "C" {

COLIBRI_API std::uint32_t colibri_native_version();
COLIBRI_API std::uint32_t colibri_cpu_features();

COLIBRI_API int colibri_q4_matvec(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
);

// Fused Q4 SwiGLU mixture-of-experts for a single token. Every routed expert
// (the caller appends the shared expert as the final entry with its own
// weight) is computed and weight-accumulated inside one call, threaded across
// experts, so the Python side ships pointers once instead of orchestrating a
// matvec per expert.
COLIBRI_API int colibri_q4_moe(
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
);

// Expert-major Q4 SwiGLU mixture-of-experts over a batch of tokens. The
// caller lists each (expert, token, weight) assignment sorted by expert, so
// every unique expert's weights are streamed from RAM once per call instead
// of once per routed token. Outputs receive the weighted expert sums per
// token (the residual is the caller's job).
COLIBRI_API int colibri_q4_moe_grouped(
    const std::uint8_t* const* gate_up_packed,
    const std::uint16_t* const* gate_up_scales,
    const std::uint8_t* const* down_packed,
    const std::uint16_t* const* down_scales,
    const std::int32_t* assignment_expert,
    const std::int32_t* assignment_token,
    const float* assignment_weight,
    const float* inputs,
    float* outputs,
    std::int32_t assignments,
    std::int32_t tokens,
    std::int32_t hidden_size,
    std::int32_t intermediate_size
);

// Optional CUDA driving (libcuda/libnvrtc are dlopen'd at runtime; these
// return nonzero when unavailable). The driver shares the device's primary
// context and the legacy default stream with CuPy.
COLIBRI_API int colibri_gpu_available();
COLIBRI_API int colibri_gpu_init(std::int32_t device);
COLIBRI_API int colibri_gpu_compile(
    const char* source,
    const char* const* options,
    std::int32_t option_count,
    std::int32_t device,
    char* log_buffer,
    std::int32_t log_capacity
);
COLIBRI_API int colibri_gpu_rms_norm(
    std::uint64_t input,
    std::uint64_t weights,
    std::uint64_t output,
    std::int32_t size,
    float epsilon,
    std::int32_t one_centered
);
COLIBRI_API int colibri_gpu_sync();

// One CPU-offloaded DeltaNet decoder layer, fully pointer-resolved so the
// per-token loop needs no interpreter. Device pointers are raw CUdeviceptr
// values from CuPy arrays whose lifetime the caller guarantees.
typedef struct ColibriDeltaLayer {
    std::uint64_t qz_packed;
    std::uint64_t qz_scales;
    std::uint64_t ba_weights;
    std::uint64_t out_proj_packed;
    std::uint64_t out_proj_scales;
    std::uint64_t input_norm;
    std::uint64_t conv_weights;
    std::uint64_t a_log;
    std::uint64_t dt_bias;
    std::uint64_t delta_norm;
    std::uint64_t conv_state;
    std::uint64_t recurrent_state;
    std::uint64_t router_gate;
    std::uint64_t post_attention_norm;
    const std::uint8_t* const* expert_gate_packed;
    const std::uint16_t* const* expert_gate_scales;
    const std::uint8_t* const* expert_down_packed;
    const std::uint16_t* const* expert_down_scales;
    const std::uint8_t* shared_gate_up_packed;
    const std::uint16_t* shared_gate_up_scales;
    const std::uint8_t* shared_down_packed;
    const std::uint16_t* shared_down_scales;
} ColibriDeltaLayer;

typedef struct ColibriDeltaParams {
    std::int32_t hidden_size;
    std::int32_t conv_dim;
    std::int32_t conv_kernel;
    std::int32_t value_dim;
    std::int32_t num_key_heads;
    std::int32_t num_value_heads;
    std::int32_t key_head_dim;
    std::int32_t value_head_dim;
    std::int32_t qz_rows;
    std::int32_t ba_rows;
    std::int32_t num_experts;
    std::int32_t top_k;
    std::int32_t moe_intermediate;
    float rms_norm_eps;
    std::uint64_t hidden;
    std::uint64_t normalized;
    std::uint64_t projected;
    std::uint64_t gates;
    std::uint64_t convolved;
    std::uint64_t cores;
    std::uint64_t mixed;
    std::uint64_t moe_normalized;
    std::uint64_t router_logits;
    float* hidden_host;
    float* normalized_host;
    float* moe_host;
    float* logits_host;
} ColibriDeltaParams;

COLIBRI_API int colibri_delta_moe_segment(
    const ColibriDeltaParams* params,
    const ColibriDeltaLayer* layers,
    std::int32_t count
);

}
