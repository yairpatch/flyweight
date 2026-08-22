#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(COLIBRI_V2_BUILD)
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
typedef struct ColibriV2KvCache ColibriV2KvCache;
typedef struct ColibriV2QwenRuntime ColibriV2QwenRuntime;
typedef struct ColibriV2Deepseek4Runtime ColibriV2Deepseek4Runtime;
typedef struct ColibriV2DsparkRuntime ColibriV2DsparkRuntime;
typedef struct ColibriV2Deepseek4Snapshot ColibriV2Deepseek4Snapshot;

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
    /* Terminator ids straight from the GGUF tokenizer metadata. UINT32_MAX when
       the key is absent. `eot` ends one chat turn where `eos` ends generation;
       models that close a turn with a dedicated end-of-turn token never emit
       `eos` in conversation, so a stop set built from `eos` alone runs on. */
    uint32_t eos_token_id, eot_token_id, bos_token_id;
    /* DeepSeek-V4 (`deepseek4`) geometry. Zero on every other architecture.
       `compress_ratios_length` is the length of the per-layer attention-kind
       array, which does not fit a flat struct; read it with
       colibri_v2_model_compress_ratios. */
    uint32_t q_lora_rank, kv_lora_rank, output_lora_rank, output_group_count;
    uint32_t indexer_head_count, indexer_key_length, indexer_top_k;
    uint32_t hyper_connection_count, sinkhorn_iterations;
    uint32_t expert_shared_count, hash_layer_count, compress_ratios_length;
    float sinkhorn_epsilon, compress_rope_freq_base;
    /* YaRN rope extension, needed by the compressed layers. A zero factor means
       the checkpoint asked for no scaling. */
    float rope_scaling_factor, yarn_beta_fast, yarn_beta_slow;
    uint32_t rope_original_context_length;
    uint32_t draft_block_size, target_layers_length, mask_token_id;
    /* Head-output transforms. `logit_scale` multiplies the logits and
       `final_logit_softcap` tanh-compresses them; zero means the checkpoint
       asked for neither. Both are monotonic, so they change sampling but never
       a greedy argmax. */
    float logit_scale, final_logit_softcap;
    /* `noaux_tc` group-limited routing: experts are split into
       `expert_group_count` groups and only the best `expert_group_used` supply
       candidates. Zero means routing is not group limited. Appended at the end
       so the struct stays layout-compatible with existing callers. */
    uint32_t expert_group_count, expert_group_used;
} ColibriV2ModelConfig;

typedef struct ColibriV2BailingRuntime ColibriV2BailingRuntime;

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
    int32_t cache_type_k; /* 0=f32, 1=f16, 2=bf16, 3=q8_0, 4=turbo3, 5=turbo4, 6=auto */
    int32_t cache_type_v; /* 0=f32, 1=f16, 2=bf16, 3=q8_0, 4=turbo3, 5=turbo4, 6=auto */
    uint32_t prefill_checkpoint_interval; /* position of the first mid-prefill checkpoint; 0 disables (end snapshots only) */
    uint32_t prefill_checkpoint_slots; /* total prefix-reuse snapshot slots; 0 = default (4) */
    uint32_t parallel_sequences; /* independent KV/decode slots (llama.cpp --parallel); 0/1 = single-sequence */
    uint32_t prompt_cache_mib; /* host RAM budget for spilled slot state (llama.cpp prompt cache); 0 disables */
    uint32_t swa_full; /* keep full-size SWA KV caches for unrestricted prefix reuse */
    uint32_t prefill_cache_seed; /* hottest prompt-routed experts to seed per layer; 0 disables */
    uint32_t expert_paging; /* 0=auto, 1=staged copy, 2=direct registered-host DMA */
    uint32_t cpu_prefetch_mib; /* prompt-trained host expert page warmup budget; 0 disables */
    uint32_t cpu_prefetch_auto; /* size from host memory and skip unless enough pages are cold */
    uint32_t next_layer_prefetch; /* transition-predicted experts to page-hint; 0 disables */
    uint32_t cpu_threads; /* OpenMP workers for CPU expert execution; 0 = automatic */
    uint32_t hybrid_prefill_cpu; /* 0 = split resident GPU/CPU; 1 = routed experts on CPU */
    uint32_t immutable_residency; /* hybrid decode: 0 = mutable; 1 = freeze per request/engine epoch */
    uint32_t prefill_cache_seed_auto; /* immutable hybrid: bounded automatic post-prefill placement */
    uint32_t strict_resident; /* streamed GPU: require and prepare the complete routed-expert set */
    uint32_t dense_requant; /* 0=auto from GPU pressure, 1=force BF16->Q8_0, 2=off */
    int32_t prefill_expert_stream_mib; /* GPU expert-GEMM budget for host-routed prefill: -1 auto, 0 off */
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
    uint64_t expert_history_loaded_entries; /* nonzero persisted expert counters restored */
    uint64_t expert_history_saves; /* successful atomic sidecar replacements */
    uint64_t next_layer_prefetch_predictions; /* predicted expert IDs issued */
    uint64_t next_layer_prefetch_hits; /* predictions present in the next real route */
    uint64_t next_layer_prefetch_bytes; /* expert tensor bytes covered by page hints */
    uint64_t next_layer_prefetch_trained_pairs; /* cross-layer route pairs observed */
    uint64_t nvfp4_tensor_core_moe_calls; /* routed decode layers completed by native FP4 GEMM */
    uint64_t nvfp4_tensor_core_moe_fallbacks; /* attempted routed FP4 layers using CUDA fallback */
    int64_t nvfp4_tensor_core_moe_last_status; /* zero on success, otherwise the most recent fallback status */
    uint64_t host_ffn_layers; /* dense blocks whose feed-forward runs on the CPU from the mapping */
    uint64_t host_ffn_bytes; /* weight bytes kept off the GPU by that spill */
    uint64_t dense_host_nanoseconds; /* wall time spent in the host-side dense SwiGLU */
    uint64_t expert_cache_deferred_admissions; /* misses recorded while residency is frozen */
    uint64_t expert_residency_epochs; /* immutable request/engine epochs started */
    uint64_t expert_residency_frozen; /* current immutable-residency state */
    uint64_t prefill_cache_seed_bytes; /* expert bytes uploaded by post-prefill placement */
    uint64_t prefill_cache_seed_selected_experts; /* resident or uploaded experts selected */
    uint64_t prefill_cache_seed_hits; /* later decode routes served by the pinned seed */
    uint64_t prefill_cache_seed_avoided_misses; /* frozen CPU fallbacks avoided by seed hits */
    uint64_t prefill_cache_seed_auto_skips; /* auto decisions retaining the prior map */
    uint64_t prefill_cache_seed_budget_stops; /* auto phases stopped at byte/time bound */
    uint64_t sampling_gpu_topk_calls; /* sampled tokens reduced to top-k on device */
    uint64_t sampling_gpu_topk_bytes; /* candidate ID/logit bytes downloaded */
    uint64_t sampling_full_download_bytes; /* fallback full-vocabulary bytes downloaded */
    uint64_t sampling_nanoseconds; /* second projection, top-k, transfer, and draw */
    uint64_t route_recurrence_observations; /* decode routes with a prior token on that layer */
    uint64_t route_recurrence_prev_hits; /* ...also routed by the immediately previous token */
    uint64_t route_recurrence_window_hits; /* ...routed by any of the last 4 tokens */
    uint64_t route_recurrence_layer_samples; /* (token, layer) pairs contributing above */
    uint64_t route_recurrence_window_experts; /* distinct experts in the window, summed over samples */
    uint64_t route_recurrence_resident; /* routed experts already in the GPU cache */
    uint64_t route_recurrence_miss_in_window; /* misses the recency window already knew about */
    uint64_t route_recurrence_miss_cold; /* misses outside the recency window */
    int32_t resolved_cache_type_k; /* cache_type_k after `auto` resolution */
    int32_t resolved_cache_type_v; /* cache_type_v after `auto` resolution */
    uint64_t grammar_constrained_steps; /* decode steps inside an open tool call */
    uint64_t grammar_rejected_candidates; /* candidates the grammar removed */
    uint64_t grammar_empty_candidate_sets; /* steps where nothing was allowed */
    uint64_t multi_decode_batches; /* batched multi-sequence decode calls */
    uint64_t multi_decode_tokens; /* tokens decoded through those batches */
    uint64_t prefill_streamed_bytes; /* expert bytes staged to the GPU during prefill */
} ColibriV2QwenRuntimeInfo;

/* Cooperative multi-request engine: tasks are submitted from any thread; ONE
   thread drives all CUDA work by calling colibri_v2_qwen_engine_step in a loop.
   Each step runs one bounded unit (a prefill chunk or one decode token) per
   runnable task in round-robin order, so a short request interleaves with a
   long prefill instead of queueing behind it. */
typedef struct ColibriV2QwenTaskEvent {
    uint64_t task_id;
    uint32_t token;
    uint32_t kind; /* 0 = token, 1 = finished, 2 = error, 3 = prefill progress */
} ColibriV2QwenTaskEvent;

typedef int (*ColibriV2TokenCallback)(uint32_t token, void* user_data);

/* One quantization a safetensors checkpoint could be loaded as. `arena_bytes`
   is what the load would produce, computed from the descriptors rather than
   estimated, and `cache_bytes` is non-zero when that arena is already packed on
   disk -- the difference between opening in a second and repacking the whole
   checkpoint. */
typedef struct ColibriV2HfQuantOption {
    char name[8];
    uint64_t arena_bytes;
    uint64_t cache_bytes;
    char cache_path[512];
    /* Empty when the option can be loaded. Otherwise why not, and the sizes
       above are zero: the routed-expert kernels decode fewer types than the
       dense path does, so the smallest two are unavailable on an MoE
       checkpoint. */
    char unavailable[128];
} ColibriV2HfQuantOption;

/* Describe what `directory` could be loaded as, without loading it: shard
   headers are parsed, no weight byte is read. Fails on anything that is not a
   readable HF checkpoint this runtime understands, which a caller offering a
   choice should treat as "do not offer one". */
COLIBRI_V2_API int colibri_v2_hf_quant_options(const char* directory,
    ColibriV2HfQuantOption* out, uint32_t capacity, uint32_t* count);

COLIBRI_V2_API int colibri_v2_model_open(const char* path, ColibriV2Model** out);
/* Replace only the embedded Qwen MTP block with tensors from a compatible
   MTP-only GGUF. The sidecar mapping is owned by `model` after attachment. */
COLIBRI_V2_API int colibri_v2_model_attach_mtp(ColibriV2Model* model, const char* path);
COLIBRI_V2_API void colibri_v2_model_close(ColibriV2Model* model);
COLIBRI_V2_API int colibri_v2_model_info(const ColibriV2Model* model, ColibriV2ModelInfo* out);
COLIBRI_V2_API int colibri_v2_model_config(const ColibriV2Model* model, ColibriV2ModelConfig* out);
/* Multiply a model tensor by a vector, decoding its stored weight type.
   GGUF reports a matrix as [inputs, outputs]. */
COLIBRI_V2_API int colibri_v2_matvec(const ColibriV2Model* model, const char* name,
    const float* input, int32_t input_size, float* output, int32_t output_size);
/* Grouped matvec: input cut into `groups` chunks, each against its own slice
   of the tensor's output rows. */
COLIBRI_V2_API int colibri_v2_grouped_matvec(const ColibriV2Model* model, const char* name,
    const float* input, int32_t inputs, float* output, int32_t rank, int32_t groups);
/* Collapse the streams for the output head: a [hc, hc*n_embd] mixer and a
   single scale, producing pre-weights and the collapsed vector. */
COLIBRI_V2_API int colibri_v2_deepseek4_head(const float* streams, const float* fn,
    const float* scale, const float* base, int32_t n_embd, int32_t hc,
    float rms_epsilon, float hc_epsilon, float* pre, float* output);
typedef struct ColibriV2Deepseek4Info {
    uint32_t layers;
    uint32_t window_layers, csa_layers, hca_layers;
    uint32_t context_limit;
    uint32_t positions;
    uint64_t state_bytes;
    uint32_t resolved_tensors;
    /* Where a token's time goes, so paging work can be aimed by measurement.
       `routed_expert_bytes` counts the weight bytes the selected experts span,
       which is what a page-in would have to fetch. */
    uint64_t forward_calls, forward_nanoseconds;
    uint64_t routed_expert_nanoseconds, shared_expert_nanoseconds;
    uint64_t attention_nanoseconds, head_nanoseconds, attention_core_nanoseconds;
    uint64_t routed_expert_bytes;
    /* Compressed blocks the indexer kept and considered; zero while a sequence
       is short enough that it selects everything. */
    uint64_t indexer_selections, indexer_candidates;
    uint64_t expert_prefetch_bytes;
    uint64_t gpu_weight_bytes, gpu_matvec_calls, gpu_batches;
    uint64_t hyper_nanoseconds, matvec_nanoseconds;
    uint64_t prefill_calls, prefill_tokens, prefill_nanoseconds;
    uint64_t expert_cache_bytes, expert_cache_slots;
    uint64_t expert_cache_hits, expert_cache_misses, expert_cache_evictions;
} ColibriV2Deepseek4Info;

/* One sequence's DeepSeek-V4 state. Raw latents are bounded by the sliding
   window and the compressor keeps only the blocks a future block can read, so
   the footprint is far below the context length would suggest. */
COLIBRI_V2_API int colibri_v2_deepseek4_runtime_create(ColibriV2Model* model,
    uint32_t context_limit, ColibriV2Deepseek4Runtime** out);
COLIBRI_V2_API void colibri_v2_deepseek4_runtime_free(ColibriV2Deepseek4Runtime* runtime);
COLIBRI_V2_API int colibri_v2_deepseek4_runtime_reset(ColibriV2Deepseek4Runtime* runtime);
/* Run one token through the stack, advancing the sequence. `logits` may be
   NULL for prompt tokens whose distribution is not wanted. */
COLIBRI_V2_API int colibri_v2_deepseek4_forward(ColibriV2Deepseek4Runtime* runtime,
    uint32_t token, float* logits);
/* Select target-layer inputs to retain during forward. Captures are the exact
   mean of the hyper-connection streams used by DFlash/DSpark encoders. */
COLIBRI_V2_API int colibri_v2_deepseek4_capture_layers(ColibriV2Deepseek4Runtime* runtime,
    const uint32_t* layers, uint32_t count);
COLIBRI_V2_API int colibri_v2_deepseek4_captured(const ColibriV2Deepseek4Runtime* runtime,
    float* output, uint64_t elements);
COLIBRI_V2_API int colibri_v2_deepseek4_lm_head(ColibriV2Deepseek4Runtime* runtime,
    const float* hidden, uint32_t rows, float* logits, uint64_t elements);
/* Feed a contiguous prompt chunk without producing intermediate logits. This
   keeps the cross-language scheduler out of the per-token loop and is the ABI
   used by progressively wider native prefill implementations. */
COLIBRI_V2_API int colibri_v2_deepseek4_prefill(ColibriV2Deepseek4Runtime* runtime,
    const uint32_t* tokens, uint32_t count);
COLIBRI_V2_API int colibri_v2_deepseek4_forward_batch(ColibriV2Deepseek4Runtime* runtime,
    const uint32_t* tokens, uint32_t count, float* logits, uint64_t elements);
COLIBRI_V2_API int colibri_v2_deepseek4_forward_batch_capture(ColibriV2Deepseek4Runtime* runtime,
    const uint32_t* tokens, uint32_t count, float* logits, uint64_t logits_elements,
    float* captures, uint64_t capture_elements);
COLIBRI_V2_API int colibri_v2_deepseek4_snapshot(const ColibriV2Deepseek4Runtime* runtime,
    ColibriV2Deepseek4Snapshot** out);
COLIBRI_V2_API int colibri_v2_deepseek4_restore(ColibriV2Deepseek4Runtime* runtime,
    const ColibriV2Deepseek4Snapshot* snapshot);
COLIBRI_V2_API void colibri_v2_deepseek4_snapshot_free(ColibriV2Deepseek4Snapshot* snapshot);
COLIBRI_V2_API int colibri_v2_deepseek4_runtime_info(const ColibriV2Deepseek4Runtime* runtime,
    ColibriV2Deepseek4Info* out);
/* Round-trip a float through half precision, as the caches store it. */
COLIBRI_V2_API float colibri_v2_deepseek4_half_round_trip(float value);
/* Gather the state rows one compressed block pools, ready for compression.
   Outputs are [rows][head_dim] with rows 2*ratio when overlapped, else ratio. */
COLIBRI_V2_API int colibri_v2_deepseek4_gather_block(const float* values, const float* scores,
    int32_t width, int32_t head_dim, int32_t ratio, int32_t block, int32_t overlapped,
    float* out_values, float* out_scores, int32_t* rows);
/* Attention mask for one query position: raw window entries then compressed
   block entries. Writes `raw_positions + blocks` bytes. */
COLIBRI_V2_API int colibri_v2_deepseek4_visible_keys(int32_t position, int32_t raw_positions,
    int32_t blocks, int32_t ratio, int32_t window, uint8_t* mask, int32_t* visible);
/* Pool `positions` rows of `width` into one compressed latent, softmaxing the
   scores per channel. */
/* The lightning indexer's score for each compressed block, and the selection
   built from it. Exposed so the ranking can be checked without a prompt long
   enough to make the runtime run it. */
/* Upload the dense half of the model to the device and run its matvecs there.
   COLIBRI_DS4_EXPERT_CACHE_MIB optionally enables a per-layer routed cache. */
COLIBRI_V2_API int colibri_v2_deepseek4_runtime_gpu(ColibriV2Deepseek4Runtime* runtime,
    int32_t device);
/* Borrow immutable device weights and the serialized scheduler workspace from
   another runtime. The owner must outlive the borrower. */
COLIBRI_V2_API int colibri_v2_deepseek4_runtime_gpu_share(
    ColibriV2Deepseek4Runtime* runtime, const ColibriV2Deepseek4Runtime* owner);
/* Make the calling thread's CUDA context current. Needed once per thread that
   drives a runtime whose weights were uploaded from another. */
COLIBRI_V2_API int colibri_v2_deepseek4_runtime_gpu_attach(
    ColibriV2Deepseek4Runtime* runtime);
/* Put one checkpoint tensor through the GPU and through the CPU with the same
   input, so the device kernels can be checked against the runtime's own before
   anything is placed on the device. */
COLIBRI_V2_API int colibri_v2_deepseek4_gpu_matvec_check(ColibriV2Model* model,
    const char* name, const float* input, int32_t inputs, int32_t outputs,
    float* out_gpu, float* out_cpu, int32_t device, int32_t iterations,
    double* seconds);
/* One stored lightning-indexer key, dequantized. The cache is built from the
   first token but nothing reads it until a sequence is long enough for the
   indexer to run, so this is how the compressor behind it gets checked. */
COLIBRI_V2_API int colibri_v2_deepseek4_indexer_key(const ColibriV2Deepseek4Runtime* runtime,
    uint32_t layer, uint32_t block, float* out, int32_t outputs);
COLIBRI_V2_API int colibri_v2_deepseek4_indexer_scores(const float* queries,
    const float* keys, const float* weights, int32_t heads, int32_t dim,
    int32_t entries, float* out);
COLIBRI_V2_API int colibri_v2_deepseek4_top_k(const float* scores, int32_t entries,
    int32_t keep, uint8_t* out);
COLIBRI_V2_API int colibri_v2_deepseek4_compress(const float* values, const float* scores,
    int32_t positions, int32_t width, float* output);
/* Matvec against one expert of a stacked [inputs, outputs, experts] tensor. */
COLIBRI_V2_API int colibri_v2_expert_matvec(const ColibriV2Model* model, const char* name,
    int32_t expert, const float* input, int32_t inputs, float* output, int32_t outputs);
/* Expert routing for one token. `bias` may be NULL. `select` false means the
   caller already filled `chosen` from a hash table and only wants weights. */
COLIBRI_V2_API int colibri_v2_deepseek4_router(const float* logits, const float* bias,
    int32_t experts, int32_t used, float weight_scale, float sum_floor,
    int32_t select, int32_t* chosen, float* weights);
/* SwiGLU with both halves clamped to +/- limit. */
COLIBRI_V2_API int colibri_v2_deepseek4_swiglu(const float* gate, const float* up,
    int32_t size, float limit, float* output);
/* Rotary embedding over the trailing `rope_dim` of each of `count` rows of
   `stride`. Adjacent-pair layout. `inverse` undoes the rotation. */
COLIBRI_V2_API int colibri_v2_deepseek4_rope(float* values, int32_t stride, int32_t rope_dim,
    int32_t count, int32_t position, float freq_base, float freq_scale, int32_t inverse,
    float ext_factor, float attn_factor, float beta_fast, float beta_slow,
    int32_t original_context);
/* Attention over the shared KV latent, with one sink logit per head. `sinks`
   and `mask` may be NULL. */
COLIBRI_V2_API int colibri_v2_deepseek4_attention(const float* queries, const float* latents,
    const float* sinks, const uint8_t* mask, int32_t heads, int32_t head_dim,
    int32_t positions, float scale, float* output);
/* RMS normalization over `count` rows of `size`, with an optional gain. */
COLIBRI_V2_API int colibri_v2_deepseek4_rms_norm(const float* input, const float* weight,
    int32_t size, int32_t count, float epsilon, float* output);
/* One DeepSeek-V4 hyper-connection step. `mixes`, `collapsed` and `combined`
   are optional; `combined` needs `block`. Sizes: streams/combined hc*n_embd,
   fn (2+hc)*hc rows of hc*n_embd, scale 3, base 2*hc+hc*hc, pre/post hc,
   comb hc*hc, collapsed/block n_embd. */
COLIBRI_V2_API int colibri_v2_deepseek4_hyper_connection(
    const float* streams, const float* fn, const float* scale, const float* base,
    int32_t n_embd, int32_t hc, int32_t sinkhorn_iterations,
    float rms_epsilon, float hc_epsilon, const float* block,
    float* mixes, float* pre, float* post, float* comb,
    float* collapsed, float* combined);
/* Whether the runtime can decode a GGML weight type on any backend. */
COLIBRI_V2_API int colibri_v2_quant_supported(uint32_t type);
/* Per-layer attention kinds for `deepseek4`. Returns the entry count when `out`
   is NULL or `capacity` is zero, else the number of entries written. */
COLIBRI_V2_API int colibri_v2_model_compress_ratios(const ColibriV2Model* model, uint32_t* out, int32_t capacity);
COLIBRI_V2_API int colibri_v2_model_target_layers(const ColibriV2Model* model, uint32_t* out, int32_t capacity);
/* DFlash feature fusion: fc(concat(target layer inputs)) followed by the
   sidecar encoder RMS norm. */
COLIBRI_V2_API int colibri_v2_dspark_encode(const ColibriV2Model* model,
    const float* features, uint64_t elements, float* output, uint64_t output_elements);
COLIBRI_V2_API int colibri_v2_dspark_runtime_create(ColibriV2Model* model,
    uint32_t context_limit, ColibriV2DsparkRuntime** out);
COLIBRI_V2_API void colibri_v2_dspark_runtime_free(ColibriV2DsparkRuntime* runtime);
COLIBRI_V2_API int colibri_v2_dspark_runtime_reset(ColibriV2DsparkRuntime* runtime);
COLIBRI_V2_API int colibri_v2_dspark_inject(ColibriV2DsparkRuntime* runtime,
    const float* fused, uint64_t elements);
COLIBRI_V2_API int colibri_v2_dspark_cached(const ColibriV2DsparkRuntime* runtime,
    uint32_t layer, uint32_t position, float* output, uint64_t elements);
/* Apply the chained DSpark Markov bias and confidence projection to one draft
   block. `base_logits` and `hidden` are row-major. */
COLIBRI_V2_API int colibri_v2_dspark_heads(const ColibriV2Model* model,
    const float* base_logits, const float* hidden, uint32_t rows,
    uint32_t anchor_token, float* logits, float* confidence, uint32_t* tokens);
COLIBRI_V2_API int colibri_v2_dspark_attention(const ColibriV2DsparkRuntime* runtime,
    uint32_t layer, const float* queries, const float* noise_kv, uint32_t rows,
    float* output, uint64_t elements);
COLIBRI_V2_API int colibri_v2_dspark_attention_stage(ColibriV2DsparkRuntime* runtime,
    uint32_t layer, const float* streams, uint32_t rows, float* output, uint64_t elements);
COLIBRI_V2_API int colibri_v2_dspark_ffn_stage(ColibriV2DsparkRuntime* runtime,
    uint32_t layer, const float* streams, uint32_t rows, float* output, uint64_t elements);
COLIBRI_V2_API int colibri_v2_dspark_decode_hidden(ColibriV2DsparkRuntime* runtime,
    const float* embeddings, uint32_t rows, float* hidden, float* normalized,
    uint64_t elements);
/* Copies tokenizer.chat_template from GGUF metadata. `length` receives the
   UTF-8 byte count without the trailing NUL. A null/zero-capacity output can
   query the required size; an absent key reports length zero. */
COLIBRI_V2_API int colibri_v2_model_chat_template(const ColibriV2Model* model, char* output, uint64_t capacity, uint64_t* length);
/* Returns the layer's trained attention window. 0 means global attention. */
COLIBRI_V2_API int colibri_v2_model_attention_window(const ColibriV2Model* model, uint32_t layer, uint32_t* out);
COLIBRI_V2_API int colibri_v2_tensor_info(const ColibriV2Model* model, uint64_t index, ColibriV2TensorInfo* out);
COLIBRI_V2_API int colibri_v2_tensor_find(const ColibriV2Model* model, const char* name, ColibriV2TensorInfo* out);
/* BailingMoE3 (Ling 3.0) host execution.

   A straightforward f32 forward pass on the CPU: correct, unoptimized, and the
   thing the eventual kernels are checked against. Requires a model whose
   tensors are f32 (open an HF checkpoint with COLIBRI_HF_QUANT=F32); the
   quantized path runs through the main runtime once its kernels land.

   `colibri_v2_bailing_eval` consumes `count` tokens starting at the runtime's
   current position and writes `vocabulary_size` logits for the LAST one. Call
   it repeatedly to decode; call reset to start a new sequence. */
COLIBRI_V2_API int colibri_v2_bailing_create(const ColibriV2Model* model, uint32_t capacity, ColibriV2BailingRuntime** out);
COLIBRI_V2_API void colibri_v2_bailing_destroy(ColibriV2BailingRuntime* runtime);
COLIBRI_V2_API int colibri_v2_bailing_reset(ColibriV2BailingRuntime* runtime);
COLIBRI_V2_API int colibri_v2_bailing_eval(ColibriV2BailingRuntime* runtime, const uint32_t* tokens, uint32_t count, float* logits);

COLIBRI_V2_API int colibri_v2_qwen_validate(const ColibriV2Model* model);
COLIBRI_V2_API int colibri_v2_qwen_tensor_role(const ColibriV2Model* model, const char* role, ColibriV2TensorInfo* out);
COLIBRI_V2_API int colibri_v2_qwen_layer_tensor(const ColibriV2Model* model, uint32_t layer, const char* role, ColibriV2TensorInfo* out);
COLIBRI_V2_API int colibri_v2_qwen_embedding(const ColibriV2Model* model, uint32_t token, float* output, uint64_t elements);
COLIBRI_V2_API int colibri_v2_qwen_lm_head(const ColibriV2Model* model, const float* hidden, float* logits, uint64_t vocabulary, uint64_t elements);
COLIBRI_V2_API int colibri_v2_qwen_token_text(const ColibriV2Model* model, uint32_t token, char* output, uint64_t capacity);
COLIBRI_V2_API int colibri_v2_token_id(const ColibriV2Model* model, const char* text, uint32_t* token);
COLIBRI_V2_API int colibri_v2_tokenize(const ColibriV2Model* model, const char* text, uint32_t* tokens, uint64_t capacity, uint64_t* count);
/* Pre-tokenizer piece boundaries as byte offsets, with a trailing end offset,
   so `count` is one more than the number of pieces. A NULL `offsets` asks for
   the count alone. */
COLIBRI_V2_API int colibri_v2_pretokenize(const ColibriV2Model* model, const char* text, uint64_t* offsets, uint64_t capacity, uint64_t* count);
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
/* Writes one attention layer's live KV window as f32 for the TurboQuant
   quality harness: int32 count, int32 head_dim, then the keys and the values. */
COLIBRI_V2_API int colibri_v2_qwen_runtime_dump_kv(ColibriV2QwenRuntime* runtime, uint32_t layer_index, const char* path);
COLIBRI_V2_API int colibri_v2_qwen_task_submit(ColibriV2QwenRuntime* runtime, const uint32_t* prompt_tokens, uint64_t prompt_count, uint64_t max_tokens, const uint32_t* stop_tokens, uint64_t stop_count, uint64_t* task_id);
COLIBRI_V2_API int colibri_v2_qwen_task_submit_sampling(ColibriV2QwenRuntime* runtime, const uint32_t* prompt_tokens, uint64_t prompt_count, uint64_t max_tokens, const uint32_t* stop_tokens, uint64_t stop_count, float temperature, uint32_t top_k, float top_p, uint64_t seed, uint32_t has_seed, uint64_t* task_id);
/* As above, plus the repetition penalties. `repetition_penalty` scales a
   recently generated token's logit toward zero (1 = off, [1, 2]);
   `presence_penalty` and `frequency_penalty` subtract from it once per token
   seen and once per occurrence ([0, 2]). `penalty_window` is how many of the
   most recent generated tokens are considered -- the prompt is never
   penalized. Without these a low-bit checkpoint can lock onto a line and
   repeat it until the token budget runs out. */
COLIBRI_V2_API int colibri_v2_qwen_task_submit_penalties(ColibriV2QwenRuntime* runtime, const uint32_t* prompt_tokens, uint64_t prompt_count, uint64_t max_tokens, const uint32_t* stop_tokens, uint64_t stop_count, float temperature, uint32_t top_k, float top_p, float repetition_penalty, float presence_penalty, float frequency_penalty, uint32_t penalty_window, uint64_t seed, uint32_t has_seed, uint64_t* task_id);
/* As above, plus a tool specification -- `[{"name": ..., "parameters":
   [{"name": ..., "required": true}]}]` -- that constrains the sampler while a
   tool call is open, so a required parameter cannot be skipped. Null or empty
   leaves the sampler unconstrained. */
COLIBRI_V2_API int colibri_v2_qwen_task_submit_grammar(ColibriV2QwenRuntime* runtime, const uint32_t* prompt_tokens, uint64_t prompt_count, uint64_t max_tokens, const uint32_t* stop_tokens, uint64_t stop_count, float temperature, uint32_t top_k, float top_p, float repetition_penalty, float presence_penalty, float frequency_penalty, uint32_t penalty_window, uint64_t seed, uint32_t has_seed, const char* tool_specification, uint64_t* task_id);
COLIBRI_V2_API int colibri_v2_qwen_engine_step(ColibriV2QwenRuntime* runtime, ColibriV2QwenTaskEvent* events, uint64_t capacity, uint64_t* count);
COLIBRI_V2_API int colibri_v2_qwen_task_cancel(ColibriV2QwenRuntime* runtime, uint64_t task_id);

COLIBRI_V2_API const char* colibri_v2_last_error(void);
COLIBRI_V2_API uint32_t colibri_v2_version(void);
COLIBRI_V2_API uint64_t colibri_v2_runtime_options_size(void);
COLIBRI_V2_API uint64_t colibri_v2_runtime_info_size(void);
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
