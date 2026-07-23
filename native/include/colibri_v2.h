#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(COLIBRI_NATIVE_BUILD)
#    define COLIBRI_V2_API __declspec(dllexport)
#  else
#    define COLIBRI_V2_API __declspec(dllimport)
#  endif
#else
#  define COLIBRI_V2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColibriV2Model ColibriV2Model;
typedef struct ColibriV2Session ColibriV2Session;
typedef struct ColibriV2KvCache ColibriV2KvCache;
typedef struct ColibriV2QwenRuntime ColibriV2QwenRuntime;

typedef struct ColibriV2ModelInfo {
    uint32_t gguf_version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t file_size;
    uint32_t alignment;
    char architecture[64];
    char name[128];
    char format[32];
} ColibriV2ModelInfo;

typedef struct ColibriV2TensorInfo {
    uint32_t dimensions;
    uint64_t shape[4];
    uint32_t ggml_type;
    uint64_t offset;
    uint64_t size;
    char name[192];
} ColibriV2TensorInfo;
typedef struct ColibriV2ModelConfig {
    char architecture[64];
    uint32_t hidden_size, layer_count, attention_heads, attention_kv_heads;
    uint32_t context_length, intermediate_size, expert_count;
    uint32_t expert_used_count, vocabulary_size;
    uint32_t rotary_dimension, full_attention_interval, sliding_window;
    uint32_t sliding_window_pattern_length;
    float rms_norm_epsilon, rope_freq_base;
} ColibriV2ModelConfig;

typedef struct ColibriV2Stats {
    uint64_t prompt_tokens;
    uint64_t decoded_tokens;
    uint64_t decode_calls;
    uint64_t bytes_mapped;
} ColibriV2Stats;

typedef struct ColibriV2GpuInfo {
    int32_t available;
    int32_t device;
    int32_t compute_major;
    int32_t compute_minor;
    uint64_t total_memory;
    uint64_t free_memory;
} ColibriV2GpuInfo;

typedef struct ColibriV2MemoryPlan {
    uint64_t budget;
    uint64_t static_weights;
    uint64_t kv_state;
    uint64_t workspace;
    uint64_t active_experts;
    uint64_t staging;
    uint64_t unused;
} ColibriV2MemoryPlan;

typedef struct ColibriV2QwenRuntimeOptions {
    int32_t device;
    int32_t moe_device; /* 0 = streamed GPU, 1 = CPU, 2 = hybrid */
    uint32_t mtp_drafts; /* 0 disables MTP; native verifier supports up to 8 */
    uint32_t expert_top_k; /* 0 = model default; else route at most this many experts/token */
    uint64_t context_limit;
    uint64_t gpu_cache_bytes;
    float expert_top_p; /* 0 or >=1 disables; else keep experts to cumulative router prob p */
    int32_t cache_type_k; /* KV cache K precision: 0=f32, 1=f16 */
    int32_t cache_type_v; /* KV cache V precision: 0=f32, 1=f16 */
    uint32_t prefill_checkpoint_interval; /* position of the first mid-prefill checkpoint; 0 disables (end snapshots only) */
    uint32_t prefill_checkpoint_slots; /* total prefix-reuse snapshot slots; 0 = default (4) */
    uint32_t parallel_sequences; /* independent KV/decode slots (llama.cpp --parallel); 0/1 = single-sequence */
    uint32_t prompt_cache_mib; /* host RAM budget for spilled slot state (llama.cpp prompt cache); 0 disables */
    uint32_t swa_full; /* keep full-size SWA KV caches for unrestricted prefix reuse */
    uint32_t prefill_cache_seed; /* hottest prompt-routed experts to seed per layer; 0 disables */
    uint32_t expert_paging; /* 0=auto, 1=staged copy, 2=direct registered-host DMA */
    uint32_t cpu_prefetch_mib; /* prompt-trained host expert page warmup budget; 0 disables */
    uint32_t cpu_prefetch_auto; /* size from host memory and skip unless enough pages are cold */
} ColibriV2QwenRuntimeOptions;

typedef struct ColibriV2QwenRuntimeInfo {
    uint32_t layers;
    uint32_t deltanet_layers;
    uint32_t attention_layers;
    uint32_t swa_layers;
    uint32_t sliding_window;
    uint32_t swa_full;
    uint32_t hidden_size;
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint32_t mtp_available;
    uint32_t mtp_enabled;
    uint32_t mtp_drafts;
    uint32_t mtp_layer;
    uint64_t context_limit;
    uint64_t static_tensor_bytes;
    uint64_t expert_tensor_bytes;
    uint64_t gpu_allocated_bytes;
    uint64_t workspace_bytes;
    uint64_t state_bytes;
    uint64_t expert_staging_bytes;
    uint64_t expert_cache_bytes;
    uint64_t expert_cache_slots;
    uint64_t expert_cache_hits;
    uint64_t expert_cache_misses;
    uint64_t expert_cache_evictions;
    uint64_t expert_cache_admissions;
    uint64_t expert_cache_rejections;
    uint64_t expert_cache_prompt_bypasses;
    uint64_t prefix_cache_hits;
    uint64_t prefix_cache_misses;
    uint64_t prefix_cache_reused_tokens;
    uint64_t mtp_tensor_bytes;
    uint64_t mtp_draft_tokens;
    uint64_t mtp_accepted_tokens;
    uint64_t mtp_rejected_tokens;
    uint64_t mtp_draft_nanoseconds;
    uint64_t mtp_verify_nanoseconds;
    uint64_t mtp_rollback_nanoseconds;
    uint64_t decode_calls;
    uint64_t decode_nanoseconds;
    uint64_t route_wait_nanoseconds;
    uint64_t expert_page_nanoseconds;
    uint64_t tail_wait_nanoseconds;
    uint64_t position;
    int32_t device;
    int32_t moe_device;
    int32_t cuda_ready;
    int32_t decode_ready;
    uint64_t route_expert_sum; /* sum of experts routed across all decode layers */
    uint64_t expert_compute_nanoseconds; /* CPU-side expert matmul (qwen_cpu_moe) time */
    /* Prefix-reuse diagnostics: quantify how much prompt is reprefilled per turn
       and where the new prompt diverges from what is already cached. On a hybrid
       DeltaNet model the recurrent state cannot be rewound, so reuse only happens
       at exact-prefix boundaries (live state or a snapshot); these fields reveal
       whether misses are early (prefix churn) or only at the tail. */
    uint64_t prefix_cache_reprefilled_tokens; /* cumulative prompt tokens actually prefilled */
    uint64_t prefix_cache_last_prompt_tokens;  /* prompt length of the most recent generate */
    uint64_t prefix_cache_last_reused_tokens;  /* tokens reused (prompt_start) last generate */
    uint64_t prefix_cache_last_lcp_live;       /* longest common prefix vs live processed_tokens */
    uint64_t prefix_cache_last_lcp_snapshot;   /* longest common prefix vs the best snapshot */
    uint64_t prompt_cache_entries;      /* host prompt cache: conversations held */
    uint64_t prompt_cache_used_bytes;   /* host prompt cache: RAM in use */
    uint64_t prefill_cache_seeded_experts; /* experts bulk-loaded from prompt route history */
    uint64_t prefill_cache_seed_nanoseconds; /* wall time spent bulk-seeding experts */
    uint64_t direct_paging; /* direct registered-host expert paging is active */
    uint64_t paging_registration_nanoseconds; /* one-time host registration cost */
    uint64_t host_available_bytes; /* available host memory observed during prepare */
    uint64_t cpu_prefetch_experts; /* prompt-hot expert bundles warmed in host page cache */
    uint64_t cpu_prefetch_bytes; /* expert bytes covered by host page warmup */
    uint64_t cpu_prefetch_nanoseconds; /* wall time spent warming host expert pages */
    uint64_t cpu_prefetch_pages; /* mapped pages inspected for host expert warmup */
    uint64_t cpu_prefetch_cold_pages; /* inspected pages nonresident before warmup */
    uint64_t cpu_prefetch_loaded_pages; /* cold pages actually faulted into host memory */
    uint64_t cpu_prefetch_auto_skips; /* auto-mode prompts skipped for low cold-page pressure */
    uint64_t cpu_prefetch_last_budget_bytes; /* most recent effective candidate-set budget */
    uint64_t prefill_calls; /* batched prompt chunks executed */
    uint64_t prefill_tokens; /* prompt tokens executed by the batched rows path */
    uint64_t prefill_nanoseconds; /* total batched rows wall time */
    uint64_t prefill_route_wait_nanoseconds; /* batched route readback synchronization */
    uint64_t prefill_expert_nanoseconds; /* batched CPU/hybrid expert phase wall time */
    uint64_t prefill_direct_quant; /* direct quantized expert path is enabled */
    uint64_t prefill_direct_quant_width; /* routed tokens sharing each packed-weight decode */
    uint64_t prefill_profile; /* detailed CUDA prefill profiling is enabled */
    uint64_t prefill_gpu_core_nanoseconds; /* DeltaNet/attention CUDA work before MoE */
    uint64_t prefill_gpu_router_nanoseconds; /* MoE norm, router projection, and top-k */
    uint64_t prefill_gpu_transfer_nanoseconds; /* selected routes, weights, and activations DtoH */
} ColibriV2QwenRuntimeInfo;

/* Cooperative multi-request engine: tasks are submitted from any thread; ONE
   thread drives all CUDA work by calling colibri_v2_qwen_engine_step in a loop.
   Each step runs one bounded unit (a prefill chunk or one decode token) per
   runnable task in round-robin order, so a short request interleaves with a
   long prefill instead of queueing behind it. */
typedef struct ColibriV2QwenTaskEvent {
    uint64_t task_id;
    uint32_t token;
    uint32_t kind; /* 0 = token emitted, 1 = task finished, 2 = task error */
} ColibriV2QwenTaskEvent;

typedef int (*ColibriV2TokenCallback)(uint32_t token, void* user_data);

COLIBRI_V2_API int colibri_v2_model_open(const char* path, ColibriV2Model** out);
COLIBRI_V2_API void colibri_v2_model_close(ColibriV2Model* model);
COLIBRI_V2_API int colibri_v2_model_info(const ColibriV2Model* model, ColibriV2ModelInfo* out);
COLIBRI_V2_API int colibri_v2_model_config(const ColibriV2Model* model, ColibriV2ModelConfig* out);
/* Returns the layer's trained attention window. 0 means global attention. */
COLIBRI_V2_API int colibri_v2_model_attention_window(const ColibriV2Model* model, uint32_t layer, uint32_t* out);
COLIBRI_V2_API int colibri_v2_tensor_info(const ColibriV2Model* model, uint64_t index, ColibriV2TensorInfo* out);
COLIBRI_V2_API int colibri_v2_tensor_find(const ColibriV2Model* model, const char* name, ColibriV2TensorInfo* out);
COLIBRI_V2_API int colibri_v2_qwen_validate(const ColibriV2Model* model);
COLIBRI_V2_API int colibri_v2_qwen_tensor_role(const ColibriV2Model* model, const char* role, ColibriV2TensorInfo* out);
COLIBRI_V2_API int colibri_v2_qwen_layer_tensor(const ColibriV2Model* model, uint32_t layer, const char* role, ColibriV2TensorInfo* out);
COLIBRI_V2_API int colibri_v2_qwen_embedding(const ColibriV2Model* model, uint32_t token, float* output, uint64_t elements);
COLIBRI_V2_API int colibri_v2_qwen_lm_head(const ColibriV2Model* model, const float* hidden, float* logits, uint64_t vocabulary, uint64_t elements);
COLIBRI_V2_API int colibri_v2_qwen_token_text(const ColibriV2Model* model, uint32_t token, char* output, uint64_t capacity);
COLIBRI_V2_API int colibri_v2_token_id(const ColibriV2Model* model, const char* text, uint32_t* token);
COLIBRI_V2_API int colibri_v2_tokenize(const ColibriV2Model* model, const char* text, uint32_t* tokens, uint64_t capacity, uint64_t* count);
COLIBRI_V2_API int colibri_v2_tensor_read(const ColibriV2Model* model, uint64_t index, void* dst, uint64_t bytes);
COLIBRI_V2_API int colibri_v2_tensor_read_slice(const ColibriV2Model* model, uint64_t index, uint64_t offset, void* dst, uint64_t bytes);
COLIBRI_V2_API int colibri_v2_tensor_view(const ColibriV2Model* model, uint64_t index, uint64_t offset, uint64_t bytes, const void** out);
COLIBRI_V2_API int colibri_v2_qwen_runtime_create(ColibriV2Model* model, const ColibriV2QwenRuntimeOptions* options, ColibriV2QwenRuntime** out);
COLIBRI_V2_API void colibri_v2_qwen_runtime_destroy(ColibriV2QwenRuntime* runtime);
COLIBRI_V2_API int colibri_v2_qwen_runtime_info(const ColibriV2QwenRuntime* runtime, ColibriV2QwenRuntimeInfo* out);
COLIBRI_V2_API int colibri_v2_qwen_runtime_reset(ColibriV2QwenRuntime* runtime);
COLIBRI_V2_API int colibri_v2_qwen_runtime_cancel(ColibriV2QwenRuntime* runtime);
COLIBRI_V2_API int colibri_v2_qwen_runtime_prepare(ColibriV2QwenRuntime* runtime);
COLIBRI_V2_API int colibri_v2_qwen_runtime_synchronize(ColibriV2QwenRuntime* runtime);
COLIBRI_V2_API int colibri_v2_qwen_runtime_decode(ColibriV2QwenRuntime* runtime, uint32_t input_token, uint32_t* output_token);
COLIBRI_V2_API int colibri_v2_qwen_runtime_generate(ColibriV2QwenRuntime* runtime, const uint32_t* prompt_tokens, uint64_t prompt_count, uint64_t max_tokens, ColibriV2TokenCallback callback, void* user_data);
COLIBRI_V2_API int colibri_v2_qwen_task_submit(ColibriV2QwenRuntime* runtime, const uint32_t* prompt_tokens, uint64_t prompt_count, uint64_t max_tokens, const uint32_t* stop_tokens, uint64_t stop_count, uint64_t* task_id);
COLIBRI_V2_API int colibri_v2_qwen_engine_step(ColibriV2QwenRuntime* runtime, ColibriV2QwenTaskEvent* events, uint64_t capacity, uint64_t* count);
COLIBRI_V2_API int colibri_v2_qwen_task_cancel(ColibriV2QwenRuntime* runtime, uint64_t task_id);

COLIBRI_V2_API int colibri_v2_session_create(ColibriV2Model* model, uint64_t context_limit, ColibriV2Session** out);
COLIBRI_V2_API void colibri_v2_session_destroy(ColibriV2Session* session);
COLIBRI_V2_API int colibri_v2_session_prompt(ColibriV2Session* session, const uint32_t* tokens, uint64_t count);
COLIBRI_V2_API int colibri_v2_session_decode(ColibriV2Session* session, uint32_t* token, float* logits, uint64_t logits_count);
COLIBRI_V2_API int colibri_v2_session_generate(ColibriV2Session* session, uint64_t max_tokens, ColibriV2TokenCallback callback, void* user_data);
COLIBRI_V2_API int colibri_v2_session_cancel(ColibriV2Session* session);
COLIBRI_V2_API int colibri_v2_session_sync(ColibriV2Session* session);
COLIBRI_V2_API int colibri_v2_session_stats(const ColibriV2Session* session, ColibriV2Stats* out);
COLIBRI_V2_API int colibri_v2_session_attach_kv_cache(ColibriV2Session* session, ColibriV2KvCache* cache);
COLIBRI_V2_API int colibri_v2_session_detach_kv_cache(ColibriV2Session* session);

COLIBRI_V2_API const char* colibri_v2_last_error(void);
COLIBRI_V2_API uint32_t colibri_v2_version(void);
COLIBRI_V2_API int colibri_v2_gpu_probe(int32_t device, ColibriV2GpuInfo* out);
COLIBRI_V2_API int colibri_v2_memory_plan(uint64_t budget, uint64_t static_weights, uint64_t kv_state, uint64_t workspace, uint64_t active_experts, uint64_t staging, ColibriV2MemoryPlan* out);
COLIBRI_V2_API int colibri_v2_gpu_available(void);
COLIBRI_V2_API int colibri_v2_gpu_init(int32_t device);
COLIBRI_V2_API int colibri_v2_gpu_compile(const char* source, const char* const* options, int32_t option_count, int32_t device, char* log_buffer, int32_t log_capacity);
COLIBRI_V2_API int colibri_v2_gpu_rms_norm(uint64_t input, uint64_t weights, uint64_t output, int32_t size, float epsilon, int32_t one_centered);
COLIBRI_V2_API int colibri_v2_gpu_q4_matvec(uint64_t packed, uint64_t scales, uint64_t input, uint64_t output, uint64_t stream, int32_t rows, int32_t columns);
COLIBRI_V2_API int colibri_v2_gpu_dense_projection(uint64_t input, uint64_t norm_weights, uint64_t normalized, uint64_t packed, uint64_t scales, uint64_t projection, int32_t rows, int32_t columns, float epsilon, int32_t one_centered);
COLIBRI_V2_API int colibri_v2_gpu_dense_residual(uint64_t input, uint64_t norm_weights, uint64_t normalized, uint64_t packed, uint64_t scales, uint64_t output, int32_t rows, int32_t columns, float epsilon, int32_t one_centered);
COLIBRI_V2_API int colibri_v2_gpu_attention(uint64_t query, uint64_t keys, uint64_t values, uint64_t output, int32_t heads, int32_t kv_heads, int32_t head_dim, int32_t tokens, float scale);
COLIBRI_V2_API int colibri_v2_gpu_decoder_attention_step(uint64_t input, uint64_t norm_weights, uint64_t normalized, uint64_t qkv_packed, uint64_t qkv_scales, uint64_t qkv, uint64_t cache_keys, uint64_t cache_values, uint64_t attention_output, uint64_t out_packed, uint64_t out_scales, uint64_t output, int32_t hidden_size, int32_t heads, int32_t kv_heads, int32_t head_dim, int32_t position, int32_t capacity, float epsilon, int32_t one_centered);
COLIBRI_V2_API int colibri_v2_kv_cache_create(uint64_t cache_keys, uint64_t cache_values, int32_t capacity, int32_t kv_heads, int32_t head_dim, ColibriV2KvCache** out);
COLIBRI_V2_API void colibri_v2_kv_cache_destroy(ColibriV2KvCache* cache);
COLIBRI_V2_API int colibri_v2_kv_cache_reset(ColibriV2KvCache* cache);
COLIBRI_V2_API int colibri_v2_kv_cache_position(const ColibriV2KvCache* cache, int32_t* out);
COLIBRI_V2_API int colibri_v2_gpu_decoder_attention_cached(ColibriV2KvCache* cache, uint64_t input, uint64_t norm_weights, uint64_t normalized, uint64_t qkv_packed, uint64_t qkv_scales, uint64_t qkv, uint64_t attention_output, uint64_t out_packed, uint64_t out_scales, uint64_t output, int32_t hidden_size, int32_t heads, float epsilon, int32_t one_centered);

#ifdef __cplusplus
}
#endif
