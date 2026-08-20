#pragma once

#include <cstdint>

#if defined(_WIN32)
#if defined(COLIBRI_V2_BUILD)
#define COLIBRI_API __declspec(dllexport)
#else
#define COLIBRI_API __declspec(dllimport)
#endif
#else
#define COLIBRI_API __attribute__((visibility("default")))
#endif

extern "C" {

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
COLIBRI_API int colibri_gpu_q4_matvec(
    std::uint64_t packed,
    std::uint64_t scales,
    std::uint64_t input,
    std::uint64_t output,
    std::uint64_t stream,
    std::int32_t rows,
    std::int32_t columns
);
COLIBRI_API int colibri_gpu_scaled_add(
    std::uint64_t target,
    std::uint64_t source,
    float scale,
    std::int32_t elements
);
COLIBRI_API int colibri_gpu_attention(
    std::uint64_t query,
    std::uint64_t keys,
    std::uint64_t values,
    std::uint64_t output,
    std::int32_t heads,
    std::int32_t kv_heads,
    std::int32_t head_dim,
    std::int32_t tokens,
    float scale
);
COLIBRI_API int colibri_gpu_attention_cache(
    std::uint64_t query,
    std::uint64_t keys,
    std::uint64_t values,
    std::uint64_t output,
    std::int32_t heads,
    std::int32_t kv_heads,
    std::int32_t head_dim,
    std::int32_t tokens,
    std::int32_t capacity,
    float scale
);
COLIBRI_API int colibri_gpu_attention_f16_cublas(
    std::uint64_t query,
    std::uint64_t query_f16,
    std::uint64_t keys,
    std::uint64_t values,
    std::uint64_t scores_f16,
    std::uint64_t output,
    std::uint64_t stream,
    std::int32_t heads,
    std::int32_t kv_heads,
    std::int32_t head_dim,
    std::int32_t tokens,
    std::int32_t capacity,
    std::int32_t first,
    float scale
);
COLIBRI_API int colibri_gpu_attention_prefill_f16_cublas(
    std::uint64_t queries,
    std::uint64_t gates,
    std::uint64_t keys,
    std::uint64_t values,
    std::uint64_t packed_queries,
    std::uint64_t scores_f32,
    std::uint64_t probabilities_f16,
    std::uint64_t packed_output,
    std::uint64_t output,
    std::uint64_t stream,
    std::int32_t heads,
    std::int32_t kv_heads,
    std::int32_t head_dim,
    std::int32_t rows,
    std::int32_t capacity,
    std::int32_t base_position,
    std::int32_t tile_rows,
    float scale,
    /* 0 skips the fused output gate, for callers that must post-process the
       attention result before applying an elementwise nonlinearity. */
    std::int32_t apply_gate
);
COLIBRI_API int colibri_gpu_kv_append(
    std::uint64_t current_keys,
    std::uint64_t current_values,
    std::uint64_t cache_keys,
    std::uint64_t cache_values,
    std::int32_t kv_heads,
    std::int32_t head_dim,
    std::int32_t position,
    std::int32_t capacity
);
COLIBRI_API int colibri_gpu_q4_moe(
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
);
COLIBRI_API int colibri_gpu_sync();
COLIBRI_API int colibri_gpu_alloc(std::uint64_t bytes, std::uint64_t* pointer);
COLIBRI_API int colibri_gpu_free(std::uint64_t pointer);
COLIBRI_API int colibri_gpu_host_alloc(std::uint64_t bytes, void** pointer);
COLIBRI_API int colibri_gpu_host_free(void* pointer);
COLIBRI_API int colibri_gpu_host_register(const void* pointer, std::uint64_t bytes);
COLIBRI_API int colibri_gpu_host_unregister(const void* pointer);
COLIBRI_API int colibri_gpu_upload(
    std::uint64_t destination, const void* source, std::uint64_t bytes,
    std::uint64_t stream
);
COLIBRI_API int colibri_gpu_upload_sync(
    std::uint64_t destination, const void* source, std::uint64_t bytes
);
COLIBRI_API int colibri_gpu_download(
    void* destination, std::uint64_t source, std::uint64_t bytes,
    std::uint64_t stream
);
COLIBRI_API int colibri_gpu_memset(
    std::uint64_t destination, std::uint8_t value, std::uint64_t bytes,
    std::uint64_t stream
);
COLIBRI_API int colibri_gpu_stream_create(std::uint64_t* stream);
COLIBRI_API int colibri_gpu_stream_destroy(std::uint64_t stream);
COLIBRI_API int colibri_gpu_stream_sync(std::uint64_t stream);
COLIBRI_API int colibri_gpu_graph_begin(std::uint64_t stream);
COLIBRI_API int colibri_gpu_graph_end(
    std::uint64_t stream, std::uint64_t* graph
);
COLIBRI_API int colibri_gpu_graph_launch(
    std::uint64_t graph, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_graph_destroy(std::uint64_t graph);
COLIBRI_API int colibri_gpu_event_create(std::uint64_t* event);
COLIBRI_API int colibri_gpu_timed_event_create(std::uint64_t* event);
COLIBRI_API int colibri_gpu_event_record(
    std::uint64_t event, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_event_sync(std::uint64_t event);
COLIBRI_API int colibri_gpu_stream_wait_event(
    std::uint64_t stream, std::uint64_t event
);
COLIBRI_API int colibri_gpu_event_destroy(std::uint64_t event);
COLIBRI_API int colibri_gpu_event_elapsed(
    std::uint64_t start, std::uint64_t end, float* milliseconds
);
COLIBRI_API int colibri_gpu_q8_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q4k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
// The sub-block-scale Q4_K matvec. Returns -1 when it is unavailable (kernel
// absent, or a row that is not a whole number of 32-element sub-blocks), so
// callers fall back to the transposed one above.
COLIBRI_API int colibri_gpu_q4k_matvec_subblock(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q6k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q2k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q3k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q5k_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq2xxs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq1m_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq1s_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq3xxs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq2xs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq4xs_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq2s_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_iq3s_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_bf16_matvec_transposed(
    std::uint64_t packed, std::uint64_t input, std::uint64_t output,
    std::int32_t input_size, std::int32_t output_size, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_route_topk(
    std::uint64_t logits, std::uint64_t selected, std::uint64_t weights,
    std::int32_t experts, std::int32_t top_k, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_sampling_topk(
    std::uint64_t logits, std::uint64_t selected,
    std::uint64_t selected_logits, std::uint64_t sort_indices_a,
    std::uint64_t sort_values_a, std::uint64_t sort_indices_b,
    std::uint64_t sort_values_b, std::int32_t vocabulary,
    std::int32_t top_k, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q5_grouped_swiglu(
    std::uint64_t gate_pointers, std::uint64_t up_pointers,
    std::uint64_t input, std::uint64_t activated,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q6_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q4k_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q5k_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_q8_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_nvfp4_grouped_swiglu(
    std::uint64_t gate_pointers, std::uint64_t up_pointers,
    std::uint64_t input, std::uint64_t activated,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
);
COLIBRI_API int colibri_gpu_nvfp4_grouped_accumulate(
    std::uint64_t down_pointers, std::uint64_t activated,
    std::uint64_t output, std::uint64_t weights,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t experts, std::uint64_t stream
);
// Native Blackwell block-scaled FP4 GEMM. The weight matrix remains in GGUF
// type-40 layout; the driver repacks it and quantizes the f32 activation rows
// into cuBLASLt's NVFP4 layout. Nonzero means the caller should use its
// decode-oriented CUDA fallback.
COLIBRI_API int colibri_gpu_nvfp4_matmul_cublas(
    std::uint64_t weights, std::uint64_t input, std::uint64_t output,
    std::uint64_t stream,
    std::int32_t input_size, std::int32_t output_size,
    std::int32_t rows, float scale
);
// Native FP4 routed-MoE decode: one stacked gate/up GEMM followed by one
// concatenated down GEMM. Pointer and scale arrays live on the device.
COLIBRI_API int colibri_gpu_nvfp4_moe_cublas(
    std::uint64_t gate_pointers, std::uint64_t up_pointers,
    std::uint64_t down_pointers, std::uint64_t input,
    std::uint64_t activated, std::uint64_t output,
    std::uint64_t route_weights, std::uint64_t gate_scales,
    std::uint64_t up_scales, std::uint64_t down_scales,
    std::uint64_t stream,
    std::int32_t hidden_size, std::int32_t intermediate_size,
    std::int32_t experts
);
// Convert one cached GGUF expert bundle to persistent Tensor-Core layout.
COLIBRI_API int colibri_gpu_nvfp4_prepare_expert(
    std::uint64_t gate, std::uint64_t up, std::uint64_t down,
    std::uint64_t native, std::uint64_t stream,
    std::int32_t hidden_size, std::int32_t intermediate_size
);
// Single-token routed MoE over persistent native expert slots. Pointer and
// scale arrays are host-resident and remain valid for the duration of the call.
COLIBRI_API int colibri_gpu_nvfp4_moe_persistent(
    const std::uint64_t* native_experts, std::uint64_t route_weights,
    std::uint64_t gate_scales, std::uint64_t up_scales,
    std::uint64_t down_scales, std::uint64_t input,
    std::uint64_t activated, std::uint64_t output, std::uint64_t stream,
    std::int32_t hidden_size, std::int32_t intermediate_size,
    std::int32_t experts
);
COLIBRI_API int colibri_gpu_launch_named(
    const char* name, std::uint32_t grid_x, std::uint32_t grid_y,
    std::uint32_t block_x, std::uint32_t shared_bytes,
    std::uint64_t stream, void** arguments
);

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
    // Instantiated CUDA graph replaying this layer's kernel chain (0 = launch
    // kernels individually). Built via colibri_delta_graph_build.
    std::uint64_t graph;
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
    // When positive, mixed|moe_normalized|router_logits are contiguous on
    // the device (and their host mirrors likewise), and one copy of this
    // many floats replaces the three separate transfers.
    std::int32_t bundle_floats;
} ColibriDeltaParams;

COLIBRI_API int colibri_delta_moe_segment(
    const ColibriDeltaParams* params,
    const ColibriDeltaLayer* layers,
    std::int32_t count
);
COLIBRI_API int colibri_delta_graph_build(
    const ColibriDeltaParams* params,
    const ColibriDeltaLayer* layer,
    std::uint64_t* handle
);
COLIBRI_API int colibri_delta_graph_destroy(std::uint64_t handle);

}
