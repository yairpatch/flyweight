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

}
