#include "colibri_v2.h"
#include "colibri_gpu_driver.h"
#include <colibri_backend.hpp>

#include <atomic>
#include "colibri_v2_provider.hpp"
#include "colibri_v2_config.hpp"
#include "colibri_v2_attention_policy.hpp"
#include "colibri_v2_expert_policy.hpp"
#include "colibri_v2_expert_seed.hpp"
#include "colibri_v2_qwen_kernels.hpp"
#include "colibri_v2_native_kernels.hpp"
#include "colibri_v2_deepseek4_kernels.hpp"
#include "colibri_v2_workspace.hpp"
#include "qwen_cpu_kernel.h"
#include "qwen_kquant.h"
#include "turboquant.h"
#include "unicode_categories.h"
#include "colibri_v2_deepseek4.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <map>
#include <vector>

#if defined(_OPENMP)
#  include <omp.h>
#endif

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sched.h>
#  include <unistd.h>
#  include <dlfcn.h>
#endif

thread_local std::string error;
void fail(const char* message) { error = message; }
template <class F> int guarded(F&& f) { try { return f(); } catch (const std::exception& e) { error = e.what(); return -1; } catch (...) { error = "unknown native exception"; return -1; } }

struct Tensor : colibri::v2::TensorDescriptor {
    // Null means the primary model mapping. MTP overlays point into the
    // sidecar mapping retained by ColibriV2Model::mtp_sidecar.
    const uint8_t* source = nullptr;
};

struct Reader {
    const uint8_t* p; const uint8_t* end;
    template <class T> T get() { if (end-p < static_cast<ptrdiff_t>(sizeof(T))) throw std::runtime_error("truncated GGUF"); T v; std::memcpy(&v,p,sizeof(v)); p += sizeof(v); return v; }
    std::string str() { auto n=get<uint64_t>(); if (n > static_cast<uint64_t>(end-p)) throw std::runtime_error("invalid GGUF string"); std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(n)); p+=n; return s; }
    void skip(size_t n) { if (n > static_cast<size_t>(end-p)) throw std::runtime_error("truncated GGUF value"); p+=n; }
    void value(uint32_t type) {
        switch(type) {
        case 0: case 1: case 7: skip(1); break;
        case 2: case 3: skip(2); break;
        case 4: case 5: case 6: skip(4); break;
        case 8: str(); break;
        case 9: { auto element_type=get<uint32_t>(); auto n=get<uint64_t>(); for(uint64_t i=0;i<n;i++) value(element_type); break; }
        case 10: case 11: case 12: skip(8); break;
        default: throw std::runtime_error("unsupported GGUF metadata type");
        }
    }
};

// Execution consumes this descriptor rather than GGUF internals. GGUF is the
// first WeightProvider; future providers can populate the same
// tensor contract without changing CUDA/runtime code.
struct ColibriV2Model : colibri::v2::WeightProvider { const uint8_t* data=nullptr; size_t size=0; uint32_t version=0, alignment=32; uint64_t metadata=0; std::string path, architecture, name, format_name="gguf", chat_template; colibri::v2::ModelConfig config; uint32_t mtp_layer=std::numeric_limits<uint32_t>::max(); std::vector<std::string> vocabulary, merges; std::unordered_map<std::string,int> merge_ranks; std::unordered_map<std::string,uint32_t> vocabulary_ids; std::vector<Tensor> tensors; std::unique_ptr<ColibriV2Model> mtp_sidecar;
    // GGUF tokenizer.ggml.pre selects the pre-tokenizer. Control tokens (GGUF
    // token type 3) never come out of BPE and have to be split off ahead of it.
    std::string tokenizer_pre; std::vector<std::uint32_t> token_types;
    std::vector<std::pair<std::string,std::uint32_t>> control_tokens;
    // Split GGUF: `split.no` is this file's 0-based index of `split.count`, and
    // `split.tensors.count` is the descriptor total across every shard. A zero
    // count means an ordinary single-file checkpoint.
    std::uint32_t split_no=0, split_count=0; std::uint64_t split_tensors=0;
    // Shards other than the one that was opened. Their descriptors are merged
    // into `tensors` with Tensor::source pointing at the shard mapping, so a
    // split checkpoint is indistinguishable downstream from a single file.
    std::vector<std::unique_ptr<ColibriV2Model>> shards;
#if !defined(_WIN32)
    int fd=-1;
#else
    HANDLE file=nullptr, mapping=nullptr;
#endif
    ~ColibriV2Model() {
#if !defined(_WIN32)
        if(data&&data!=MAP_FAILED)munmap(const_cast<uint8_t*>(data),size);
        if(fd>=0)::close(fd);
#else
        if(data)UnmapViewOfFile(reinterpret_cast<LPCVOID>(data));
        if(mapping)CloseHandle(mapping);
        if(file&&file!=INVALID_HANDLE_VALUE)CloseHandle(file);
#endif
    }
    const char* format() const override { return format_name.c_str(); }
    uint64_t tensor_count() const override { return tensors.size(); }
    const colibri::v2::TensorDescriptor* tensor(uint64_t index) const override { return index < tensors.size() ? &tensors[index] : nullptr; }
    int read_tensor(uint64_t index, void* destination, uint64_t bytes) const override { if(index >= tensors.size() || !destination || bytes < tensors[index].size) return -1; const auto& tensor=tensors[index];const auto*base=tensor.source?tensor.source:data;std::memcpy(destination, base + tensor.offset, tensor.size); return 0; }
    // Every mapping that can hold tensor bytes, primary file first. Nothing may
    // assume `data`/`size` spans the model: a split checkpoint keeps all of its
    // metadata in one shard and all of its tensors in the others.
    template <class F> void for_each_mapping(F&& visit) const {
        if(data) visit(data,static_cast<std::uint64_t>(size));
        for(const auto& shard:shards) if(shard&&shard->data) visit(shard->data,static_cast<std::uint64_t>(shard->size));
    }
    std::uint64_t mapped_bytes() const { std::uint64_t total=size; for(const auto& shard:shards) if(shard) total+=shard->size; return total; }
};

const uint8_t* tensor_data(const ColibriV2Model& model, const Tensor& tensor) {
    return (tensor.source ? tensor.source : model.data) + tensor.offset;
}
struct ColibriV2KvCache { std::uint64_t keys, values; std::int32_t capacity, kv_heads, head_dim, position=0; };

struct QwenLayerPlan {
    bool attention = false;
    // Dense checkpoints (Qwen3.6-27B and friends) carry one ffn_gate/ffn_up/
    // ffn_down triple per block instead of a router, a shared expert and the
    // stacked routed experts, so there is nothing to route or page.
    bool dense_ffn = false;
    // Set when this block's SwiGLU runs on the CPU straight from the mapping
    // because its weights did not fit in the GPU budget. Only the feed-forward
    // moves: attention and the DeltaNet recurrence stay device-resident.
    bool ffn_on_host = false;
    std::uint32_t attention_window = 0; // 0 = global attention
    std::uint64_t cache_capacity = 0;
    std::uint32_t attention_heads = 0, kv_heads = 0, head_dim = 0;
    std::uint32_t rotary_dim = 0, expert_tensor_count = 3;
    float rope_theta = 0.0f;
    // YaRN extension for this layer. A zero ext_factor means plain RoPE, which
    // is what every architecture except Laguna's full-attention layers uses.
    float rope_freq_scale = 1.0f, rope_ext_factor = 0.0f, rope_attn_factor = 1.0f;
    float rope_beta_fast = 32.0f, rope_beta_slow = 1.0f;
    std::uint32_t rope_orig_context = 0;
    std::vector<std::uint64_t> static_tensors;
    std::array<std::uint64_t, 3> expert_tensors{};
    std::uint64_t shared_graph = 0;
    bool shared_graph_attempted = false;
    // Laguna's router score-correction bias. Kept out of static_tensors so the
    // feed-forward slots line up with the Qwen layout the FFN code addresses.
    std::uint64_t router_bias = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t expert_gate_scale = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t expert_up_scale = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t expert_down_scale = std::numeric_limits<std::uint64_t>::max();
    // weight_scale_2 for the shared expert (NVFP4 checkpoints ship one f32 per
    // tensor). Kept separate from the routed-expert scales, which are per expert.
    float shared_gate_scale = 1.0f;
    float shared_up_scale = 1.0f;
    float shared_down_scale = 1.0f;
    std::uint64_t state_first = 0;
    std::uint64_t state_second = 0;
    std::uint64_t snapshot_first = 0;
    std::uint64_t snapshot_second = 0;
};

struct QwenExpertSlot {
    std::uint64_t key = 0;
    std::uint64_t last_used = 0;
    bool valid = false;
    bool native_valid = false;
    bool pinned = false;
    // One reservation per sequence that predicted this expert for its next
    // layer. A count (rather than a bool) matters under --parallel: two
    // sequences may predict the same resident expert, and the first one to
    // consume its route must not release the second one's protection.
    std::uint32_t prefetch_pins = 0;
};

struct QwenExpertHistory {
    std::uint32_t frequency = 0;
    std::uint64_t last_used = 0;
};

struct QwenCudaLayerProfile {
    std::uint64_t pre_start=0,pre_end=0;
    // Carved out of the pre phase so the gated-delta recurrence can be told
    // apart from the bf16 projections it sits between; the two have completely
    // different cures and the phase total does not distinguish them.
    std::uint64_t recurrent_start=0,recurrent_end=0;
    std::uint64_t shared_start=0,shared_end=0;
    std::uint64_t expert_start=0,expert_end=0;
};

// DeltaNet conv+recurrent states captured at the end of a prompt prefill,
// keyed by the exact prompt token sequence. Lets a follow-up request reuse
// the whole conversation prefix even though the recurrent state cannot be
// rewound: agentic clients re-encode the assistant reply differently than
// it was generated, so the plain processed_tokens extension check misses
// every multi-turn request. Attention KV needs no copy (position bounds it).
struct QwenPrefillSnapshot {
    std::uint64_t device = 0;
    std::vector<std::uint32_t> tokens;
    std::uint32_t last_output = 0;
    std::uint64_t clock = 0;
    bool valid = false;
};

// How many previously decoded tokens the route-recurrence probe keeps per
// layer. Depth 1 answers "does token t reuse token t-1's experts"; the deeper
// slots answer "how much does a small union buy", which is the difference
// between prefetching a point prediction and prefetching a working set.
constexpr std::size_t kRouteRecurrenceDepth = 16;

struct QwenExpertPrefetchState {
    std::vector<std::int32_t> previous_layer_route;
    std::vector<std::int32_t> pending_predictions;
    std::uint32_t previous_route_layer =
        std::numeric_limits<std::uint32_t>::max();
    // Route-recurrence probe (measurement only; nothing reads these to make a
    // decision). Per layer, a ring of the last kRouteRecurrenceDepth decoded
    // tokens' route sets: history is layers*depth*stride entries, counts and
    // the ring cursor/fill are layers*depth and layers respectively.
    std::vector<std::int32_t> recurrence_history;
    std::vector<std::uint8_t> recurrence_counts;
    std::vector<std::uint8_t> recurrence_cursor;
    std::vector<std::uint8_t> recurrence_filled;
    std::size_t recurrence_stride = 0;
};

// One decode sequence: its own attention-KV + DeltaNet state arena plus the
// bookkeeping that pins prefix reuse to it. Agentic clients multiplex several
// logical conversations (main agent, subagents, title/quota side-calls) onto
// the runtime; giving each its own arena means a short side-request routed to
// another slot no longer overwrites the main conversation's KV and forces a
// full reprefill next turn. The active slot's fields are mirrored onto the
// runtime (so the compute kernels, which read runtime.state at fixed offsets,
// are untouched); qwen_switch_sequence saves/loads them on a slot change.
struct QwenSequence {
    std::uint64_t state = 0;            // device KV+DeltaNet arena for this slot
    std::uint64_t position = 0;
    std::uint32_t last_output_token = 0;
    std::vector<std::uint32_t> processed_tokens;
    std::vector<QwenPrefillSnapshot> prefill_snapshots; // per-slot reuse checkpoints
    std::uint64_t prefill_snapshot_clock = 0;
    std::uint64_t clock = 0;            // LRU stamp across slots
    QwenExpertPrefetchState expert_prefetch;
    // Routes observed while this sequence prefills its current request.
    // Placement is shared, but evidence is not: cooperative requests must not
    // overwrite one another's prompt-local ranking signal.
    std::vector<std::uint32_t> prompt_expert_frequency;
    std::vector<std::uint32_t> prompt_expert_history;
    std::uint64_t prompt_expert_observations = 0;
};

// A reuse checkpoint spilled to host RAM alongside a QwenHostPrompt.
struct QwenHostSnapshot {
    std::vector<std::uint32_t> tokens;
    void* state = nullptr;             // host copy of the DeltaNet checkpoint buffer
    std::uint32_t last_output = 0;
};

// A conversation's full slot state spilled to host RAM (llama.cpp's prompt
// cache). When an LRU slot is recycled for a new conversation, its arena and
// reuse checkpoints are DtoH-copied here instead of discarded, so a later
// request that continues it restores from RAM rather than reprefilling ~30k
// tokens cold. The checkpoints matter: the end-of-prompt one (tokens = prompt)
// is what lets the next turn reuse past the prompt boundary regardless of how
// the client re-renders the prior assistant reply.
struct QwenHostPrompt {
    std::vector<std::uint32_t> tokens;
    void* state = nullptr;             // host copy of the slot's KV+DeltaNet arena
    std::vector<QwenHostSnapshot> snapshots;
    std::uint64_t position = 0;
    std::uint32_t last_output_token = 0;
    std::uint64_t bytes = 0;           // total host bytes held (arena + snapshots)
    std::uint64_t clock = 0;           // LRU stamp within the host cache
};

// Resumable prompt-processing state shared by the blocking generate path and
// the cooperative engine: where reuse let the prefill start, the token that is
// already known (on a full-reuse hit), and the mid-prefill checkpoint targets.
struct QwenPromptPlan {
    std::uint64_t prompt_start = 0;
    std::uint32_t next_token = 0;
    std::vector<std::uint64_t> targets;
    std::size_t next_target = 0;
};

struct QwenSamplingState {
    float temperature = 0.0f;
    std::uint32_t top_k = 20;
    float top_p = 0.95f;
    std::uint64_t rng = 0;

    bool enabled() const { return temperature > 0.0f; }
};

// One in-flight engine request. phase: 0 = pending (waiting for a slot),
// 1 = prefilling, 2 = decoding.
struct QwenEngineTask {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> prompt;
    std::vector<std::uint32_t> stop_tokens;
    std::uint64_t max_tokens = 0;
    QwenPromptPlan plan;
    std::uint64_t index = 0;
    std::uint32_t next_token = 0;
    std::uint64_t emitted = 0;
    // Tokens an MTP round committed beyond next_token, emitted one per engine
    // visit so a drafting task still interleaves with the others.
    std::deque<std::uint32_t> drafted;
    QwenSamplingState sampling;
    std::size_t slot = 0;
    int phase = 0;
    bool cancelled = false;
};

struct ColibriV2QwenRuntime {
    ColibriV2Model* model = nullptr;
    bool gemma4 = false;
    bool laguna = false;
    ColibriV2QwenRuntimeOptions options{};
    colibri::v2::ExpertExecutionMode expert_mode =
        colibri::v2::ExpertExecutionMode::streamed_gpu;
    std::vector<QwenLayerPlan> layers;
    QwenLayerPlan mtp_layer_plan;
    std::array<std::uint64_t, 4> mtp_special_tensors{};
    bool mtp_available = false;
    std::uint64_t token_embeddings = 0;
    // The embedding table is a gather of one row per token, so holding all of
    // it in VRAM costs a full vocab x hidden matrix to serve hidden elements
    // per token. When the checkpoint keeps a separate lm_head (untied), the
    // table is staged row-by-row from the mapping instead and the arena bytes
    // go to the expert cache. Tied embeddings must stay resident: lm_head reads
    // the whole matrix every token.
    bool embeddings_host_resident = false;
    std::uint64_t embedding_row_bytes = 0;
    std::uint64_t embedding_stage = 0;        // device scratch, capacity rows
    std::uint64_t embedding_stage_bytes = 0;
    // Staging for the turbo attention path: one layer's live KV window expanded
    // to f16 so cuBLAS can run on it, reused across every attention layer. Only
    // allocated when a turbo cache type is configured.
    std::uint64_t turbo_kv_stage = 0;
    std::uint64_t turbo_kv_stage_bytes = 0;
    std::uint64_t turbo_kv_stage_stride = 0;  // bytes from the K half to the V half
    void* embedding_host = nullptr;           // pinned mirror of the above
    std::uint64_t embedding_row_index = 0;    // device [0,1,..capacity-1]
    std::uint64_t embedding_event = 0;        // guards reuse of embedding_host
    std::uint64_t final_norm = 0;
    std::uint64_t lm_head = 0;
    std::uint32_t lm_head_type = 2;
    std::uint64_t rope_factors = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t static_tensor_bytes = 0;
    std::uint64_t expert_tensor_bytes = 0;
    std::uint64_t mtp_tensor_bytes = 0;
    std::uint32_t scratch_elements = 0;
    std::uint32_t moe_intermediate = 0;
    std::vector<std::uint64_t> device_tensors;
    // Effective on-device type per tensor, which is the checkpoint's own type
    // except where prepare requantized a bf16 weight to Q8_0. Every device
    // dispatch must read this rather than model->tensors[i].type; the host
    // paths still decode the mapping with the original type.
    std::vector<std::uint32_t> device_tensor_types;
    std::uint64_t requantized_tensors = 0;
    std::uint64_t requantized_saved_bytes = 0;
    std::uint64_t static_arena = 0;
    std::uint64_t static_arena_bytes = 0;
    // Static tensors served directly from the GGUF mapping instead of a device
    // copy (CPU backend only). Reported on stderr at prepare, alongside the
    // requantization summary, rather than through the info struct -- the ABI is
    // shared with the GPU build where these are always zero.
    std::uint64_t aliased_tensors = 0;
    std::uint64_t aliased_tensor_bytes = 0;
    std::uint64_t workspace = 0;
    std::uint64_t workspace_bytes = 0;
    colibri::v2::workspace::QwenDecodeWorkspaceLayout decode_workspace_layout;
    colibri::v2::workspace::QwenRowsWorkspaceLayout rows_workspace_layout;
    // Non-zero when the model's DeltaNet layers match the chunked prefill
    // kernels' fixed head_dim, which is what sized the delta_* workspace regions.
    std::uint32_t delta_value_heads = 0;
    colibri::v2::workspace::QwenDecodeHostLayout decode_host_layout;
    colibri::v2::workspace::QwenRowsHostLayout rows_host_layout;
    // Pinned scratch for the host-side dense SwiGLU: normalized input, the
    // gate/up activation and the projected output for one token.
    void* dense_host = nullptr;
    std::uint64_t dense_host_bytes = 0;
    std::uint32_t host_ffn_layers = 0;
    std::uint64_t host_ffn_bytes = 0;
    std::uint64_t dense_host_nanoseconds = 0;
    // Cacheable mirror of the pinned dense scratch; see qwen_cpu_dense_ffn.
    std::vector<float> dense_scratch;
    std::uint64_t last_sampling_normalized = 0;
    std::uint64_t last_sampling_logits = 0;
    std::uint64_t state = 0;
    std::uint64_t state_bytes = 0;
    std::uint64_t expert_staging = 0;
    std::uint64_t expert_staging_bytes = 0;
    std::uint64_t expert_cache = 0;
    std::uint64_t expert_cache_bytes = 0;
    std::uint64_t expert_native_cache = 0;
    std::uint64_t expert_native_cache_bytes = 0;
    std::uint64_t expert_slot_bytes = 0;
    std::vector<QwenExpertSlot> expert_slots;
    // Laguna caches complete routed-expert layers instead of spreading a
    // partial working set across every layer. Entries are the first cache slot
    // for that layer, or -1 for layers whose routed experts remain on the CPU.
    std::vector<std::int32_t> whole_expert_layer_slots;
    std::vector<QwenExpertHistory> expert_history;
    std::string expert_history_path;
    std::uint64_t expert_history_fingerprint = 0;
    std::uint64_t expert_history_loaded_entries = 0;
    std::uint64_t expert_history_saves = 0;
    std::vector<std::uint16_t> expert_transitions;
    std::uint64_t next_layer_prefetch_predictions = 0;
    std::uint64_t next_layer_prefetch_hits = 0;
    std::uint64_t next_layer_prefetch_bytes = 0;
    std::uint64_t next_layer_prefetch_trained_pairs = 0;
    // Route-recurrence probe. `observations` is the denominator (routed experts
    // seen at decode with at least one prior token on the same layer);
    // `prev_hits` counts those the immediately preceding token also routed to,
    // `window_hits` those any of the last kRouteRecurrenceDepth tokens did.
    std::uint64_t route_recurrence_observations = 0;
    std::uint64_t route_recurrence_prev_hits = 0;
    std::uint64_t route_recurrence_window_hits = 0;
    std::uint64_t route_recurrence_layer_samples = 0;
    // Distinct experts held by the window, summed over layer samples. The
    // window's hit rate is only meaningful against this: it is the residency
    // the window would cost if it were used as a prefetch set.
    std::uint64_t route_recurrence_window_experts = 0;
    // Miss decomposition against the same window: how much of the residual
    // miss rate a whole-token prefetch plan could have covered in advance.
    std::uint64_t route_recurrence_resident = 0;
    std::uint64_t route_recurrence_miss_in_window = 0;
    std::uint64_t route_recurrence_miss_cold = 0;
    std::uint64_t nvfp4_tensor_core_moe_calls = 0;
    std::uint64_t nvfp4_tensor_core_moe_fallbacks = 0;
    std::int64_t nvfp4_tensor_core_moe_last_status = 0;
    std::unordered_map<std::uint64_t, std::size_t> expert_residency;
    std::uint64_t expert_clock = 0;
    std::uint64_t expert_cache_hits = 0;
    std::uint64_t expert_cache_misses = 0;
    std::uint64_t expert_cache_evictions = 0;
    std::uint64_t expert_cache_admissions = 0;
    std::uint64_t expert_cache_rejections = 0;
    std::uint64_t expert_cache_prompt_bypasses = 0;
    std::uint64_t expert_cache_deferred_admissions = 0;
    std::uint64_t expert_residency_epochs = 0;
    std::uint64_t prefix_cache_hits = 0;
    std::uint64_t prefix_cache_misses = 0;
    std::uint64_t prefix_cache_reused_tokens = 0;
    std::uint64_t prefix_cache_reprefilled_tokens = 0;
    std::uint64_t prefix_cache_last_prompt_tokens = 0;
    std::uint64_t prefix_cache_last_reused_tokens = 0;
    std::uint64_t prefix_cache_last_lcp_live = 0;
    std::uint64_t prefix_cache_last_lcp_snapshot = 0;
    std::uint64_t mtp_draft_tokens = 0;
    std::uint64_t mtp_accepted_tokens = 0;
    std::uint64_t mtp_rejected_tokens = 0;
    std::uint64_t mtp_draft_nanoseconds = 0;
    std::uint64_t mtp_verify_nanoseconds = 0;
    std::uint64_t mtp_rollback_nanoseconds = 0;
    std::uint64_t mtp_calibration_decode_nanoseconds = 0;
    std::uint64_t mtp_calibration_round_nanoseconds = 0;
    std::uint64_t mtp_calibration_round_tokens = 0;
    std::uint32_t mtp_calibration_decode_tokens = 0;
    std::uint32_t mtp_calibration_rounds = 0;
    bool mtp_adaptive_disabled = false;
    bool mtp_adaptive_reported = false;
    std::uint64_t mtp_target_hidden_offset = 0;
    std::uint64_t mtp_draft_hidden_offset = 0;
    std::uint64_t mtp_verified_hidden_offset = 0;
    std::uint64_t mtp_snapshot_offset = 0;
    std::uint64_t mtp_snapshot_bytes = 0;
    std::uint64_t mtp_cache_tokens = 0;
    std::uint64_t decode_calls = 0;
    std::uint64_t decode_nanoseconds = 0;
    std::uint64_t route_wait_nanoseconds = 0;
    std::uint64_t expert_page_nanoseconds = 0;
    std::uint64_t tail_wait_nanoseconds = 0;
    std::uint64_t route_expert_sum = 0;
    std::uint64_t expert_compute_nanoseconds = 0;
    std::uint64_t prefill_cache_seeded_experts = 0;
    std::uint64_t prefill_cache_seed_nanoseconds = 0;
    std::uint64_t prefill_cache_seed_bytes = 0;
    std::uint64_t prefill_cache_seed_selected_experts = 0;
    std::uint64_t prefill_cache_seed_hits = 0;
    std::uint64_t prefill_cache_seed_avoided_misses = 0;
    std::uint64_t prefill_cache_seed_auto_skips = 0;
    std::uint64_t prefill_cache_seed_budget_stops = 0;
    std::uint64_t sampling_gpu_topk_calls = 0;
    std::uint64_t sampling_gpu_topk_bytes = 0;
    std::uint64_t sampling_full_download_bytes = 0;
    std::uint64_t sampling_nanoseconds = 0;
    std::uint64_t paging_registration_nanoseconds = 0;
    std::uint64_t host_available_bytes = 0;
    std::uint64_t cpu_prefetch_experts = 0;
    std::uint64_t cpu_prefetch_bytes = 0;
    std::uint64_t cpu_prefetch_nanoseconds = 0;
    std::uint64_t cpu_prefetch_pages = 0;
    std::uint64_t cpu_prefetch_cold_pages = 0;
    std::uint64_t cpu_prefetch_loaded_pages = 0;
    std::uint64_t cpu_prefetch_auto_skips = 0;
    std::uint64_t cpu_prefetch_last_budget_bytes = 0;
    std::uint64_t prefill_calls = 0;
    std::uint64_t prefill_tokens = 0;
    std::uint64_t prefill_nanoseconds = 0;
    std::uint64_t prefill_route_wait_nanoseconds = 0;
    std::uint64_t prefill_expert_nanoseconds = 0;
    std::uint64_t prefill_gpu_core_nanoseconds = 0;
    std::uint64_t prefill_gpu_router_nanoseconds = 0;
    std::uint64_t prefill_gpu_transfer_nanoseconds = 0;
    std::uint8_t cpu_prefetch_checksum = 0;
    void* host_staging = nullptr;
    std::uint64_t host_staging_bytes = 0;
    std::uint32_t forward_rows_capacity = 0;
    std::uint32_t prefill_rows = 0;
    std::vector<QwenPrefillSnapshot> prefill_snapshots{std::vector<QwenPrefillSnapshot>(2)};
    std::uint64_t prefill_snapshot_bytes = 0;
    std::uint32_t prefill_checkpoint_interval = 0; // tokens before the first mid-prefill checkpoint; 0 disables
    std::uint64_t prefill_snapshot_clock = 0;
    std::uint64_t stream = 0;
    std::uint64_t graph_stream = 0;
    std::uint64_t route_event = 0;
    std::uint64_t prefill_layer_start_event = 0;
    std::uint64_t prefill_core_end_event = 0;
    std::uint64_t prefill_router_end_event = 0;
    bool prefill_profile = false;
    bool cuda_profile = false;
    bool cuda_graphs = false;
    std::uint64_t cuda_graph_builds = 0;
    std::uint64_t cuda_graph_replays = 0;
    std::uint64_t cuda_graph_fallbacks = 0;
    bool cuda_graph_trace_reported = false;
    bool fused_attention = true;
    // Interleaving two distant expert matrices hurts mmap/TLB locality on
    // memory-bound CPU MoE. Keep the experimental kernel opt-in until it can
    // demonstrate a win across representative hardware and expert routing.
    bool fused_moe_gate_up = false;
    std::vector<QwenCudaLayerProfile> cuda_layer_profiles;
    std::uint64_t cuda_tail_start = 0, cuda_lm_start = 0;
    std::uint64_t cuda_lm_end = 0, cuda_tail_end = 0;
    std::uint64_t position = 0;
    std::uint32_t last_output_token = 0;
    std::vector<std::uint32_t> processed_tokens;
    // Parallel decode slots (see QwenSequence). sequences[active_sequence] owns
    // the arena that runtime.state currently points at; its bookkeeping is the
    // live runtime.{position,last_output_token,processed_tokens}. Default 1 slot
    // reproduces the legacy single-sequence runtime exactly.
    std::vector<QwenSequence> sequences;
    std::size_t active_sequence = 0;
    std::uint64_t sequence_clock = 0;
    std::uint32_t parallel_sequences = 1;
    // Host-backed prompt cache (see QwenHostPrompt). limit=0 disables it.
    std::vector<QwenHostPrompt> host_prompts;
    std::uint64_t host_cache_limit_bytes = 0;
    std::uint64_t host_cache_used_bytes = 0;
    std::uint64_t host_cache_clock = 0;
    // Cooperative engine (see colibri_v2_qwen_engine_step). engine_pending /
    // engine_cancel_requests / engine_next_task_id are guarded by engine_mutex
    // (submit/cancel arrive from request threads); engine_tasks / slot_owner
    // are touched only by the single engine-stepping thread.
    std::vector<struct QwenEngineTask> engine_pending;
    std::vector<std::uint64_t> engine_cancel_requests;
    std::uint64_t engine_next_task_id = 1;
    std::vector<struct QwenEngineTask> engine_tasks;
    std::vector<long long> slot_owner;
    std::size_t engine_cursor = 0;
    std::mutex engine_mutex;
    // Multi-sequence decode (engine Phase B): per-slot route events let each
    // sequence's router readback complete independently; the staging event
    // fences reuse of the shared expert-paging staging area against its async
    // (stream-queued) consumers. Capacity is how many per-sequence workspace
    // slices fit; host block is one sequence's private host staging.
    std::vector<std::uint64_t> slot_events;
    std::uint64_t staging_event = 0;
    std::uint64_t prefetch_stream = 0;
    std::uint64_t prefetch_event = 0;
    bool prefetch_pending = false;
    std::uint32_t prefetch_target_layer =
        std::numeric_limits<std::uint32_t>::max();
    std::uint64_t decode_slice_bytes = 0;
    std::uint64_t decode_host_block_bytes = 0;
    std::uint32_t multi_decode_capacity = 1;
    bool cancelled = false;
    bool cache_admission_enabled = true;
    bool expert_residency_frozen = false;
    bool strict_cache_admission = true;
    bool mtp_has_target_hidden = false;
    bool cuda_ready = false;
    bool decode_ready = false;
    bool dma_paging = false;       // expert page-ins go straight from the registered mmap
    bool model_registered = false; // whether we cuMemHostRegister'd model->data
};

colibri::v2::ExpertExecutionPolicy qwen_expert_policy(
    const ColibriV2QwenRuntime& runtime,
    colibri::v2::ExpertExecutionPhase phase
) {
    return {runtime.expert_mode,phase,runtime.cache_admission_enabled,
            runtime.options.hybrid_prefill_cpu!=0,
            runtime.expert_residency_frozen};
}

bool qwen_immutable_residency(const ColibriV2QwenRuntime& runtime) {
    return runtime.options.immutable_residency!=0&&
        runtime.expert_mode==colibri::v2::ExpertExecutionMode::hybrid;
}

bool qwen_freeze_expert_residency(ColibriV2QwenRuntime& runtime) {
    if(!qwen_immutable_residency(runtime)||runtime.expert_residency_frozen)
        return false;
    runtime.expert_residency_frozen=true;
    ++runtime.expert_residency_epochs;
    return true;
}

void qwen_unfreeze_expert_residency(ColibriV2QwenRuntime& runtime) {
    runtime.expert_residency_frozen=false;
}

struct QwenResidencyEpochGuard {
    ColibriV2QwenRuntime& runtime;
    bool owned = false;
    explicit QwenResidencyEpochGuard(ColibriV2QwenRuntime& value)
        :runtime(value),owned(qwen_freeze_expert_residency(value)){}
    ~QwenResidencyEpochGuard(){
        if(owned)qwen_unfreeze_expert_residency(runtime);
    }
};

std::size_t qwen_cache_layer_count(const ColibriV2QwenRuntime& runtime) {
    // The MTP block's routed experts execute directly on the CPU and do
    // not consult expert_residency. Only target layers consume GPU cache slots
    // or persistent routing history.
    return runtime.layers.size();
}

constexpr std::size_t kNoExpertSlot = std::numeric_limits<std::size_t>::max();

// GGUF quantization block sizes (bytes per block).
constexpr std::uint32_t kQ8BlockSize = 34;   // Q8_0: 2-byte scale + 32 bytes per 32 elements
constexpr std::uint32_t kQ4BlockSize = 18;   // Q4_0: 2-byte scale + 16 nibbles + 2 padding per 32 elements
constexpr std::uint32_t kTurbo3BlockSize = 14; // TurboQuant 3-bit: f16 scale + 32 packed indices
constexpr std::uint32_t kTurbo4BlockSize = 18; // TurboQuant 4-bit: f16 scale + 32 packed indices

// Bytes needed to hold `elements` KV slots at cache precision `t`
// (0=f32, 1=f16, 2=bf16, 3=q8_0, 4=turbo3, 5=turbo4).
inline std::uint64_t kv_type_bytes(std::uint64_t elements, int t) {
    if (t == 4) return (elements / 32) * kTurbo3BlockSize;
    if (t == 5) return (elements / 32) * kTurbo4BlockSize;
    return t == 3 ? (elements / 32) * kQ8BlockSize : elements * (t == 0 ? 4 : 2);
}
inline bool kv_type_is_turbo(int t) { return t == 4 || t == 5; }

constexpr std::uint32_t kQ2KBlockSize = kQ2KBlockBytes;   // Q2_K: 84 bytes per 256 elements
constexpr std::uint32_t kQ3KBlockSize = kQ3KBlockBytes;   // Q3_K: 110 bytes per 256 elements
constexpr std::uint32_t kIq2xxsBlockSize = kIq2xxsBlockBytes; // IQ2_XXS: 66 bytes per 256 elements
constexpr std::uint32_t kIq3xxsBlockSize = kIq3xxsBlockBytes; // IQ3_XXS: 98 bytes per 256 elements
constexpr std::uint32_t kQ5KBlockSize = 176;  // Q5_K: 176 bytes per 256 elements
constexpr std::uint32_t kQ6KBlockSize = 210;  // Q6_K: 210 bytes per 256 elements
constexpr std::uint32_t kQ4KBlockSize = 144;  // Q4_K: 144 bytes per 256 elements
// IQ1_S: d(2) + qs[32] + qh[8*2] = 50 bytes per 256 values, 1.5625 bits each.
constexpr std::uint32_t kIq1sBlockSize = 50;
constexpr float kIq1sDelta = 0.125f;
constexpr std::uint32_t kMxfp4BlockSize = 17;      // MXFP4: e[1] E8M0 scale + qs[16] nibbles
constexpr std::uint32_t kMxfp4BlockElements = 32;
// The FP4 codebook, doubled -- which is why the scale is halved to match.
constexpr float kMxfp4Lut[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};
// E8M0 exponent to float, halved: 2^(x-128). Values below 2 land in the
// denormal range and are built from bit patterns rather than shifted.
inline float mxfp4_scale(std::uint8_t exponent) {
    const std::uint32_t bits = exponent < 2
        ? (0x00200000u << exponent)
        : (static_cast<std::uint32_t>(exponent - 1) << 23);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
constexpr std::uint32_t kNvfp4BlockSize = 36;      // NVFP4: d[4] E4M3 scales + qs[32] nibbles
constexpr std::uint32_t kNvfp4BlockElements = 64;  // 4 sub-blocks of 16 elements
constexpr std::uint32_t kNvfp4SubBlock = 16;       // elements governed by one scale
constexpr std::uint32_t kBlockElements = 256;  // Super-block element count for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K

// The AVX2/AVX-512 entry points decode anything they do not recognize with
// their Q8_0 path, so this has to be an allowlist of the formats that actually
// have a hand-written SIMD kernel. A blocklist silently mis-decodes every type
// nobody remembered to add to it. NVFP4 is dispatched separately because it has
// an AVX2 kernel but no AVX-512 one.
constexpr bool qwen_simd_quant_type(std::uint32_t type) {
    return type == 8 || type == 10 || type == 11 || type == 12 || type == 13 || type == 14;
}

// The multi-input entry points (pair/quad/oct) only have register-blocked
// kernels for a subset, so they take a narrower allowlist. Anything excluded
// still reaches SIMD one row at a time through the single-row fallback.
constexpr bool qwen_simd_multi_type(std::uint32_t type) {
    return type == 8 || type == 12 || type == 13 || type == 14;
}

constexpr char kExpertHistoryMagic[8] = {'C','O','L','H','I','S','T','1'};
constexpr std::uint32_t kExpertHistoryVersion = 1;

static std::uint64_t expert_history_hash_bytes(
    std::uint64_t hash, const void* data, std::size_t size
) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::uint64_t qwen_model_fingerprint(const ColibriV2Model& model) {
    std::uint64_t hash = 1469598103934665603ULL;
    // Total mapped bytes, not the opened file's: a split checkpoint's first
    // shard is a few megabytes of metadata and would not distinguish models.
    // This equals `size` for a single file, so existing fingerprints hold.
    const std::uint64_t bytes = model.mapped_bytes();
    hash = expert_history_hash_bytes(hash, &bytes, sizeof(bytes));
    hash = expert_history_hash_bytes(
        hash, model.architecture.data(), model.architecture.size()
    );
    hash = expert_history_hash_bytes(hash, model.name.data(), model.name.size());
    for (const auto& tensor : model.tensors) {
        hash = expert_history_hash_bytes(hash, tensor.name.data(), tensor.name.size());
        hash = expert_history_hash_bytes(hash, &tensor.offset, sizeof(tensor.offset));
        hash = expert_history_hash_bytes(hash, &tensor.size, sizeof(tensor.size));
    }
    return hash;
}

static std::string qwen_expert_history_path(const ColibriV2Model& model) {
    const char* setting = std::getenv("COLIBRI_EXPERT_HISTORY");
    if (setting && (!std::strcmp(setting, "0") || !std::strcmp(setting, "off"))) {
        return {};
    }
    if (setting && *setting && std::strcmp(setting, "1") &&
        std::strcmp(setting, "auto")) {
        return setting;
    }
    return model.path.empty() ? std::string{} : model.path + ".expert-history";
}

template <typename T>
static bool expert_history_read(std::ifstream& input, T& value) {
    return static_cast<bool>(
        input.read(reinterpret_cast<char*>(&value), sizeof(value))
    );
}

template <typename T>
static bool expert_history_write(std::ofstream& output, const T& value) {
    return static_cast<bool>(
        output.write(reinterpret_cast<const char*>(&value), sizeof(value))
    );
}

static void qwen_load_expert_history(ColibriV2QwenRuntime& runtime) {
    runtime.expert_history_path = qwen_expert_history_path(*runtime.model);
    runtime.expert_history_fingerprint = qwen_model_fingerprint(*runtime.model);
    if (runtime.expert_history_path.empty()) return;
    std::ifstream input(runtime.expert_history_path, std::ios::binary);
    if (!input) return;
    char magic[sizeof(kExpertHistoryMagic)]{};
    std::uint32_t version = 0, layers = 0, experts = 0;
    std::uint64_t fingerprint = 0, clock = 0, entries = 0;
    if (!input.read(magic, sizeof(magic)) ||
        std::memcmp(magic, kExpertHistoryMagic, sizeof(magic)) ||
        !expert_history_read(input, version) ||
        !expert_history_read(input, fingerprint) ||
        !expert_history_read(input, layers) ||
        !expert_history_read(input, experts) ||
        !expert_history_read(input, clock) ||
        !expert_history_read(input, entries) ||
        version != kExpertHistoryVersion ||
        fingerprint != runtime.expert_history_fingerprint ||
        layers != runtime.layers.size() ||
        experts != runtime.model->config.expert_count ||
        entries != runtime.expert_history.size()) {
        std::fprintf(
            stderr,
            "[colibri-v2] ignoring incompatible expert history: %s\n",
            runtime.expert_history_path.c_str()
        );
        return;
    }
    std::vector<QwenExpertHistory> loaded(runtime.expert_history.size());
    std::uint64_t loaded_entries = 0, maximum_last_used = 0;
    for (auto& item : loaded) {
        if (!expert_history_read(input, item.frequency) ||
            !expert_history_read(input, item.last_used)) {
            std::fprintf(
                stderr,
                "[colibri-v2] ignoring truncated expert history: %s\n",
                runtime.expert_history_path.c_str()
            );
            return;
        }
        // Age the prior process's workload once at startup while retaining at
        // least one observation for experts that were ever useful.
        if (item.frequency) item.frequency = std::max(1U, item.frequency / 2U);
        if (item.frequency) ++loaded_entries;
        maximum_last_used = std::max(maximum_last_used, item.last_used);
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
        std::fprintf(
            stderr,
            "[colibri-v2] ignoring oversized expert history: %s\n",
            runtime.expert_history_path.c_str()
        );
        return;
    }
    runtime.expert_history = std::move(loaded);
    runtime.expert_clock = std::max(clock, maximum_last_used);
    runtime.expert_history_loaded_entries = loaded_entries;
    std::fprintf(
        stderr,
        "[colibri-v2] restored %llu learned expert entries from %s\n",
        static_cast<unsigned long long>(loaded_entries),
        runtime.expert_history_path.c_str()
    );
}

static void qwen_save_expert_history(ColibriV2QwenRuntime& runtime) {
    if (runtime.expert_history_path.empty() || runtime.expert_history.empty()) return;
    const std::string temporary = runtime.expert_history_path + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) return;
    const std::uint32_t layers = static_cast<std::uint32_t>(runtime.layers.size());
    const std::uint32_t experts = runtime.model->config.expert_count;
    const std::uint64_t entries = runtime.expert_history.size();
    output.write(kExpertHistoryMagic, sizeof(kExpertHistoryMagic));
    bool valid =
        expert_history_write(output, kExpertHistoryVersion) &&
        expert_history_write(output, runtime.expert_history_fingerprint) &&
        expert_history_write(output, layers) &&
        expert_history_write(output, experts) &&
        expert_history_write(output, runtime.expert_clock) &&
        expert_history_write(output, entries);
    for (const auto& item : runtime.expert_history) {
        valid = valid && expert_history_write(output, item.frequency);
        valid = valid && expert_history_write(output, item.last_used);
    }
    output.flush();
    valid = valid && static_cast<bool>(output);
    output.close();
    if (!valid || std::rename(temporary.c_str(), runtime.expert_history_path.c_str())) {
        std::remove(temporary.c_str());
        return;
    }
    ++runtime.expert_history_saves;
}

struct QwenExpertHistorySaveGuard {
    ColibriV2QwenRuntime& runtime;
    ~QwenExpertHistorySaveGuard() { qwen_save_expert_history(runtime); }
};

static std::size_t select_expert_cache_slot(
    ColibriV2QwenRuntime& runtime, std::uint32_t layer,
    std::uint32_t expert, bool allow_rejection, bool record_access = true
);
static void qwen_wait_for_prefetch_layer(
    ColibriV2QwenRuntime& runtime, std::uint32_t layer
) {
    if (!runtime.prefetch_pending ||
        runtime.prefetch_target_layer != layer) return;
    if (runtime.prefetch_event &&
        colibri_gpu_stream_wait_event(
            runtime.stream, runtime.prefetch_event
        ) != 0)
        throw std::runtime_error("native Qwen prefetch wait failed");
    runtime.prefetch_pending = false;
    runtime.prefetch_target_layer =
        std::numeric_limits<std::uint32_t>::max();
}

// Measures how much of a decode token's routing a *previous* token already
// told us, per layer. This is the cheap predictor the current design never
// consults: it is known before the token starts, for every layer at once,
// where the transition table only ever sees one layer of runway. Pure
// measurement -- no caching, prefetch, or execution decision reads these.
static void qwen_observe_route_recurrence(
    ColibriV2QwenRuntime& runtime, QwenExpertPrefetchState& state,
    std::uint32_t layer, const std::int32_t* selected, int selected_count
) {
    // Off unless asked for: the distinct-count pass below is O(depth^2 * top_k)
    // per layer, which is measurement overhead nobody serving should pay.
    static const bool enabled = [] {
        const char* setting = std::getenv("COLIBRI_ROUTE_RECURRENCE");
        return setting && setting[0] != '0';
    }();
    if (!enabled) return;
    const auto layers = runtime.layers.size();
    if (!layers || layer >= layers || selected_count <= 0) return;
    const auto stride = static_cast<std::size_t>(selected_count);
    // Route width can shrink mid-run (top-k / top-p pruning). Sizing the ring
    // to the model's trained top-k keeps one allocation valid for the session.
    const auto capacity = std::max<std::size_t>(
        stride, runtime.model->config.expert_used_count);
    if (state.recurrence_stride != capacity) {
        state.recurrence_stride = capacity;
        state.recurrence_history.assign(
            layers * kRouteRecurrenceDepth * capacity, -1);
        state.recurrence_counts.assign(layers * kRouteRecurrenceDepth, 0);
        state.recurrence_cursor.assign(layers, 0);
        state.recurrence_filled.assign(layers, 0);
    }
    const auto base = static_cast<std::size_t>(layer) * kRouteRecurrenceDepth;
    const auto filled = state.recurrence_filled[layer];
    if (filled) {
        ++runtime.route_recurrence_layer_samples;
        // Slot `cursor - 1` (mod depth) is the most recently written token.
        const auto newest =
            (state.recurrence_cursor[layer] + kRouteRecurrenceDepth - 1) %
            kRouteRecurrenceDepth;
        for (int rank = 0; rank < selected_count; ++rank) {
            const auto expert = selected[rank];
            if (expert < 0) continue;
            ++runtime.route_recurrence_observations;
            bool in_window = false;
            for (std::size_t depth = 0; depth < filled; ++depth) {
                const auto slot = base + (newest + kRouteRecurrenceDepth - depth) %
                    kRouteRecurrenceDepth;
                const auto count = state.recurrence_counts[slot];
                const auto* entries =
                    state.recurrence_history.data() + slot * capacity;
                const bool present =
                    std::find(entries, entries + count, expert) !=
                    entries + count;
                if (!present) continue;
                in_window = true;
                if (depth == 0) ++runtime.route_recurrence_prev_hits;
                break;
            }
            if (in_window) ++runtime.route_recurrence_window_hits;
            // Sizes the latency-hiding win. Residency is read before this
            // layer's own lookup and before the next-layer prefetch (which
            // only touches layer+1's slot range), so it is exactly what the
            // layer is about to see. A miss the window already knew about is
            // one a whole-token prefetch plan could have covered in advance;
            // a miss outside the window is genuinely cold.
            const auto key = (static_cast<std::uint64_t>(layer) << 32) |
                static_cast<std::uint32_t>(expert);
            if (runtime.expert_residency.find(key) !=
                runtime.expert_residency.end()) {
                ++runtime.route_recurrence_resident;
            } else if (in_window) {
                ++runtime.route_recurrence_miss_in_window;
            } else {
                ++runtime.route_recurrence_miss_cold;
            }
        }
        // Distinct size of the same window, counted once per layer sample.
        std::uint32_t distinct = 0;
        for (std::size_t depth = 0; depth < filled; ++depth) {
            const auto slot = base + (newest + kRouteRecurrenceDepth - depth) %
                kRouteRecurrenceDepth;
            const auto* entries =
                state.recurrence_history.data() + slot * capacity;
            for (std::size_t index = 0; index < state.recurrence_counts[slot];
                 ++index) {
                const auto expert = entries[index];
                bool seen = false;
                for (std::size_t earlier = 0; earlier < depth && !seen;
                     ++earlier) {
                    const auto other =
                        base + (newest + kRouteRecurrenceDepth - earlier) %
                            kRouteRecurrenceDepth;
                    const auto* prior =
                        state.recurrence_history.data() + other * capacity;
                    seen = std::find(
                        prior, prior + state.recurrence_counts[other], expert
                    ) != prior + state.recurrence_counts[other];
                }
                if (!seen) {
                    // Also dedupe against earlier ranks of this same slot.
                    seen = std::find(entries, entries + index, expert) !=
                        entries + index;
                }
                if (!seen) ++distinct;
            }
        }
        runtime.route_recurrence_window_experts += distinct;
    }
    const auto write = base + state.recurrence_cursor[layer];
    std::copy(selected, selected + selected_count,
              state.recurrence_history.begin() +
                  static_cast<std::ptrdiff_t>(write * capacity));
    state.recurrence_counts[write] = static_cast<std::uint8_t>(selected_count);
    state.recurrence_cursor[layer] =
        static_cast<std::uint8_t>((state.recurrence_cursor[layer] + 1) %
                                  kRouteRecurrenceDepth);
    if (filled < kRouteRecurrenceDepth)
        state.recurrence_filled[layer] = static_cast<std::uint8_t>(filled + 1);
}

static void qwen_observe_and_prefetch_next_layer(
    ColibriV2QwenRuntime& runtime, QwenExpertPrefetchState& state,
    std::uint32_t layer,
    const std::int32_t* selected, int selected_count
) {
    const auto budget = runtime.options.next_layer_prefetch;
    const std::uint32_t experts = runtime.model->config.expert_count;
    const auto policy=qwen_expert_policy(
        runtime,colibri::v2::ExpertExecutionPhase::decode);
    if (!budget || !experts || selected_count <= 0 ||
        runtime.expert_transitions.empty()) {
        return;
    }
    if (layer == 0) {
        // A completed token has no pending final-layer prediction. Clear any
        // stale bookkeeping defensively without carrying transition context
        // across token boundaries.
        for (const auto prediction : state.pending_predictions) {
            if (state.previous_route_layer ==
                std::numeric_limits<std::uint32_t>::max()) break;
            const auto target_layer = state.previous_route_layer + 1;
            const auto key =
                (static_cast<std::uint64_t>(target_layer) << 32) |
                static_cast<std::uint32_t>(prediction);
            const auto resident = runtime.expert_residency.find(key);
            if (resident != runtime.expert_residency.end()) {
                auto& slot = runtime.expert_slots[resident->second];
                if (slot.prefetch_pins) --slot.prefetch_pins;
            }
        }
        state.previous_layer_route.clear();
        state.pending_predictions.clear();
        state.previous_route_layer =
            std::numeric_limits<std::uint32_t>::max();
    }
    if (state.previous_route_layer !=
            std::numeric_limits<std::uint32_t>::max() &&
        state.previous_route_layer + 1 == layer) {
        for (const auto prediction : state.pending_predictions) {
            for (int rank = 0; rank < selected_count; ++rank) {
                if (selected[rank] == prediction) {
                    ++runtime.next_layer_prefetch_hits;
                    break;
                }
            }
            const auto key =
                (static_cast<std::uint64_t>(layer) << 32) |
                static_cast<std::uint32_t>(prediction);
            const auto resident = runtime.expert_residency.find(key);
            if (resident != runtime.expert_residency.end()) {
                auto& slot = runtime.expert_slots[resident->second];
                if (slot.prefetch_pins) --slot.prefetch_pins;
            }
        }
        const std::size_t boundary =
            static_cast<std::size_t>(layer - 1) * experts * experts;
        for (const auto source : state.previous_layer_route) {
            if (source < 0 || static_cast<std::uint32_t>(source) >= experts) continue;
            for (int rank = 0; rank < selected_count; ++rank) {
                const auto target = selected[rank];
                if (target < 0 || static_cast<std::uint32_t>(target) >= experts) continue;
                auto& count = runtime.expert_transitions[
                    boundary + static_cast<std::size_t>(source) * experts +
                    static_cast<std::uint32_t>(target)
                ];
                if (count != std::numeric_limits<std::uint16_t>::max()) ++count;
                ++runtime.next_layer_prefetch_trained_pairs;
            }
        }
    }
    state.previous_layer_route.assign(selected, selected + selected_count);
    state.previous_route_layer = layer;
    state.pending_predictions.clear();
    if (layer + 1 >= runtime.layers.size()) return;

    const std::size_t boundary =
        static_cast<std::size_t>(layer) * experts * experts;
    std::vector<std::pair<std::uint64_t, std::int32_t>> scores;
    scores.reserve(experts);
    for (std::uint32_t candidate = 0; candidate < experts; ++candidate) {
        std::uint64_t score = 0;
        for (const auto source : state.previous_layer_route) {
            if (source >= 0 && static_cast<std::uint32_t>(source) < experts) {
                score += runtime.expert_transitions[
                    boundary + static_cast<std::size_t>(source) * experts +
                    candidate
                ];
            }
        }
        if (score) scores.emplace_back(score, static_cast<std::int32_t>(candidate));
    }
    const auto count = std::min<std::size_t>(budget, scores.size());
    std::partial_sort(
        scores.begin(), scores.begin() + count, scores.end(),
        [](const auto& left, const auto& right) {
            return left.first != right.first
                ? left.first > right.first : left.second < right.second;
        }
    );
    const auto& next = runtime.layers[layer + 1];
    bool queued_gpu_upload = false;
    for (std::size_t index = 0; index < count; ++index) {
        const auto expert = scores[index].second;
        state.pending_predictions.push_back(expert);
        ++runtime.next_layer_prefetch_predictions;
        const auto key =
            (static_cast<std::uint64_t>(layer + 1) << 32) |
            static_cast<std::uint32_t>(expert);
        const auto already_resident = runtime.expert_residency.find(key);
        if (already_resident != runtime.expert_residency.end()) {
            if (policy.is_streamed_gpu())
                ++runtime.expert_slots[already_resident->second].prefetch_pins;
            continue;
        }
        // Faulting the mapping in is only worth it when the page-in still goes
        // through the host. Under direct paging the mmap is CUDA-registered, so
        // its pages are already pinned and resident, and this walk is 432 stray
        // reads per role per expert on the critical path for nothing.
        if (!runtime.dma_paging) {
#if !defined(_WIN32)
        if (runtime.model->fd >= 0) {
            for (int role = 0; role < 3; ++role) {
                const auto& tensor =
                    runtime.model->tensors[next.expert_tensors[role]];
                const auto bytes = tensor.size / experts;
                const auto offset =
                    tensor.offset + static_cast<std::uint64_t>(expert) * bytes;
                (void)posix_fadvise(
                    runtime.model->fd, static_cast<off_t>(offset),
                    static_cast<off_t>(bytes), POSIX_FADV_WILLNEED
                );
                runtime.next_layer_prefetch_bytes += bytes;
            }
        }
#else
        for (int role = 0; role < 3; ++role) {
            const auto& tensor =
                runtime.model->tensors[next.expert_tensors[role]];
            const auto bytes = tensor.size / experts;
            const auto offset =
                tensor.offset + static_cast<std::uint64_t>(expert) * bytes;
            const auto* base = tensor_data(*runtime.model,tensor) +
                static_cast<std::uint64_t>(expert) * bytes;
            const std::size_t page_size = 4096;
            for (std::size_t pos = 0; pos < bytes; pos += page_size)
                (void)base[pos];
            runtime.next_layer_prefetch_bytes += bytes;
        }
#endif
        }
        if (policy.is_streamed_gpu() && runtime.dma_paging &&
            !runtime.expert_slots.empty()) {
            const auto slot_index = select_expert_cache_slot(
                runtime, layer + 1, expert, true, false
            );
            if (slot_index == kNoExpertSlot) continue;
            auto& slot = runtime.expert_slots[slot_index];
            const auto slot_base =
                runtime.expert_cache + slot_index * runtime.expert_slot_bytes;
            std::uint64_t role_offset = 0;
            for (int role = 0; role < 3; ++role) {
                const auto& tensor =
                    runtime.model->tensors[next.expert_tensors[role]];
                const auto bytes = tensor.size / experts;
                const auto offset = static_cast<std::uint64_t>(expert) * bytes;
                if (colibri_gpu_upload(
                        slot_base + role_offset,
                        tensor_data(*runtime.model,tensor) + offset,
                        bytes, runtime.prefetch_stream
                    ) != 0)
                    throw std::runtime_error(
                        "native Qwen GPU prefetch DMA upload failed"
                    );
                role_offset += bytes;
                runtime.next_layer_prefetch_bytes += bytes;
            }
            slot.key = key;
            slot.valid = true;
            slot.last_used = ++runtime.expert_clock;
            ++slot.prefetch_pins;
            runtime.expert_residency[key] = slot_index;
            queued_gpu_upload = true;
        }
    }
    if (queued_gpu_upload) {
        if (colibri_gpu_event_record(
                runtime.prefetch_event, runtime.prefetch_stream
            ) != 0)
            throw std::runtime_error("native Qwen prefetch event record failed");
        runtime.prefetch_pending = true;
        runtime.prefetch_target_layer = layer + 1;
    }
}

QwenExpertHistory& record_expert_access(
    ColibriV2QwenRuntime& runtime, std::uint32_t layer, std::uint32_t expert
) {
    const auto experts = runtime.model->config.expert_count;
    if (!runtime.cache_admission_enabled &&
        runtime.active_sequence < runtime.sequences.size()) {
        auto& sequence = runtime.sequences[runtime.active_sequence];
        const auto index = static_cast<std::size_t>(layer) * experts + expert;
        if (sequence.prompt_expert_frequency.size() ==
            static_cast<std::size_t>(runtime.layers.size()) * experts) {
            auto& frequency = sequence.prompt_expert_frequency[index];
            if (frequency != std::numeric_limits<std::uint32_t>::max())
                ++frequency;
            ++sequence.prompt_expert_observations;
        }
    }
    auto& history = runtime.expert_history[
        static_cast<std::size_t>(layer) * experts + expert
    ];
    ++runtime.expert_clock;
    if (history.frequency != std::numeric_limits<std::uint32_t>::max()) {
        ++history.frequency;
    }
    history.last_used = runtime.expert_clock;
    if ((runtime.expert_clock & 32767U) == 0) {
        for (auto& item : runtime.expert_history) {
            if (item.frequency > 1) item.frequency = (item.frequency + 1) / 2;
        }
    }
    return history;
}

static void record_expert_cache_hit(
        ColibriV2QwenRuntime& runtime, const QwenExpertSlot& slot) {
    ++runtime.expert_cache_hits;
    if (runtime.cache_admission_enabled && slot.pinned) {
        ++runtime.prefill_cache_seed_hits;
        ++runtime.prefill_cache_seed_avoided_misses;
    }
}

std::size_t select_expert_cache_slot(
    ColibriV2QwenRuntime& runtime, std::uint32_t layer,
    std::uint32_t expert, bool allow_rejection, bool record_access
) {
    if(runtime.expert_residency_frozen){
        if(record_access)record_expert_access(runtime,layer,expert);
        ++runtime.expert_cache_deferred_admissions;
        return kNoExpertSlot;
    }
    const auto policy=qwen_expert_policy(
        runtime,colibri::v2::ExpertExecutionPhase::decode);
    if (!policy.misses_may_be_admitted()) {
        // Prompt prefill must not churn the device cache, but its routes are
        // valuable training data for a later bulk seed of the hottest experts.
        record_expert_access(runtime, layer, expert);
        ++runtime.expert_cache_prompt_bypasses;
        return kNoExpertSlot;
    }
    auto& candidate = runtime.expert_history[
        static_cast<std::size_t>(layer) *
            runtime.model->config.expert_count + expert
    ];
    if (record_access) record_expert_access(runtime, layer, expert);
    if(!runtime.whole_expert_layer_slots.empty()){
        if(layer>=runtime.whole_expert_layer_slots.size()||
           runtime.whole_expert_layer_slots[layer]<0){
            ++runtime.expert_cache_rejections;
            return kNoExpertSlot;
        }
        const auto slot=static_cast<std::size_t>(
            runtime.whole_expert_layer_slots[layer])+expert;
        if(slot>=runtime.expert_slots.size())
            throw std::runtime_error(
                "native Laguna whole-layer expert slot is out of range");
        return slot;
    }
    const auto cache_layers = qwen_cache_layer_count(runtime);
    const auto slot_begin = runtime.expert_slots.size() * layer / cache_layers;
    const auto slot_end = runtime.expert_slots.size() * (layer + 1) / cache_layers;
    auto begin = runtime.expert_slots.begin() + static_cast<std::ptrdiff_t>(slot_begin);
    auto end = runtime.expert_slots.begin() + static_cast<std::ptrdiff_t>(slot_end);
    if (begin == end) throw std::runtime_error("native Qwen layer has no expert cache slots");
    auto free_slot = std::find_if(begin, end, [](const QwenExpertSlot& slot) {
        return !slot.valid;
    });
    if (free_slot != end) {
        ++runtime.expert_cache_admissions;
        return static_cast<std::size_t>(free_slot - runtime.expert_slots.begin());
    }
    auto victim = std::min_element(begin, end, [&](const QwenExpertSlot& left, const QwenExpertSlot& right) {
        const bool left_protected = left.pinned || left.prefetch_pins;
        const bool right_protected = right.pinned || right.prefetch_pins;
        if (left_protected != right_protected) return !left_protected;
        const auto left_expert = static_cast<std::uint32_t>(left.key);
        const auto right_expert = static_cast<std::uint32_t>(right.key);
        const auto& left_history = runtime.expert_history[static_cast<std::size_t>(layer) * runtime.model->config.expert_count + left_expert];
        const auto& right_history = runtime.expert_history[static_cast<std::size_t>(layer) * runtime.model->config.expert_count + right_expert];
        return left_history.frequency != right_history.frequency
            ? left_history.frequency < right_history.frequency
            : left.last_used < right.last_used;
    });
    if (victim->pinned || victim->prefetch_pins) {
        ++runtime.expert_cache_rejections;
        return kNoExpertSlot;
    }
    const auto victim_expert = static_cast<std::uint32_t>(victim->key);
    const auto& victim_history = runtime.expert_history[
        static_cast<std::size_t>(layer) * runtime.model->config.expert_count + victim_expert
    ];
    // The legacy policy adapts quickly by admitting an equally frequent but
    // newer candidate. Strict admission avoids replacing a resident until the
    // candidate is demonstrably hotter.
    if (allow_rejection &&
        (runtime.strict_cache_admission
             ? static_cast<std::uint64_t>(candidate.frequency) <=
                   static_cast<std::uint64_t>(victim_history.frequency)
             : static_cast<std::uint64_t>(candidate.frequency) <
                   static_cast<std::uint64_t>(victim_history.frequency))) {
        ++runtime.expert_cache_rejections;
        return kNoExpertSlot;
    }
    const auto slot = static_cast<std::size_t>(victim - runtime.expert_slots.begin());
    runtime.expert_residency.erase(victim->key);
    victim->native_valid = false;
    ++runtime.expert_cache_evictions;
    ++runtime.expert_cache_admissions;
    return slot;
}

std::uint64_t device_align(std::uint64_t bytes) {
    return colibri::v2::workspace::align(bytes);
}

void release_qwen_device(ColibriV2QwenRuntime& runtime) {
    qwen_save_expert_history(runtime);
    if (runtime.stream) colibri_gpu_stream_sync(runtime.stream);
    if (runtime.model_registered && runtime.model) {
        runtime.model->for_each_mapping([](const std::uint8_t* base, std::uint64_t) {
            colibri_gpu_host_unregister(base);
        });
        runtime.model_registered = false;
        runtime.dma_paging = false;
    }
    colibri_gpu_host_free(runtime.host_staging);
    colibri_gpu_host_free(runtime.dense_host);
    runtime.dense_host = nullptr;
    runtime.dense_host_bytes = 0;
    if (runtime.embedding_event) colibri_gpu_event_destroy(runtime.embedding_event);
    colibri_gpu_host_free(runtime.embedding_host);
    colibri_gpu_free(runtime.turbo_kv_stage);
    runtime.turbo_kv_stage = 0;
    runtime.turbo_kv_stage_bytes = 0;
    colibri_gpu_free(runtime.embedding_stage);
    colibri_gpu_free(runtime.embedding_row_index);
    runtime.embedding_event = 0;
    runtime.embedding_host = nullptr;
    runtime.embedding_stage = 0;
    runtime.embedding_stage_bytes = 0;
    runtime.embedding_row_index = 0;
    runtime.embeddings_host_resident = false;
    // The active slot's checkpoint pool is mirrored in runtime.prefill_snapshots;
    // inactive slots keep theirs in sequences[i]. Each buffer lives in exactly
    // one of the two, so freeing both frees every buffer once.
    for (auto& snapshot : runtime.prefill_snapshots) {
        colibri_gpu_free(snapshot.device);
        snapshot = QwenPrefillSnapshot{};
    }
    runtime.prefill_snapshot_bytes = 0;
    colibri_gpu_free(runtime.expert_cache);
    colibri_gpu_free(runtime.expert_native_cache);
    colibri_gpu_free(runtime.expert_staging);
    // runtime.state aliases sequences[active].state; free each slot's arena +
    // its checkpoint pool once.
    for (auto& seq : runtime.sequences) {
        for (auto& snapshot : seq.prefill_snapshots) colibri_gpu_free(snapshot.device);
        colibri_gpu_free(seq.state);
        seq.state = 0;
    }
    runtime.sequences.clear();
    runtime.active_sequence = 0;
    for (auto& e : runtime.host_prompts) {
        std::free(e.state);
        for (auto& snapshot : e.snapshots) std::free(snapshot.state);
    }
    runtime.host_prompts.clear();
    runtime.host_cache_used_bytes = 0;
    for (auto& layer : runtime.layers) {
        colibri_gpu_graph_destroy(layer.shared_graph);
        layer.shared_graph = 0;
        layer.shared_graph_attempted = false;
    }
    colibri_gpu_free(runtime.workspace);
    colibri_gpu_free(runtime.static_arena);
    for (auto& event : runtime.slot_events) colibri_gpu_event_destroy(event);
    runtime.slot_events.clear();
    colibri_gpu_event_destroy(runtime.staging_event);
    runtime.staging_event = 0;
    colibri_gpu_event_destroy(runtime.prefetch_event);
    runtime.prefetch_event = 0;
    colibri_gpu_event_destroy(runtime.route_event);
    colibri_gpu_event_destroy(runtime.prefill_layer_start_event);
    colibri_gpu_event_destroy(runtime.prefill_core_end_event);
    colibri_gpu_event_destroy(runtime.prefill_router_end_event);
    for(auto&profile:runtime.cuda_layer_profiles){
        colibri_gpu_event_destroy(profile.pre_start);
        colibri_gpu_event_destroy(profile.pre_end);
        colibri_gpu_event_destroy(profile.recurrent_start);
        colibri_gpu_event_destroy(profile.recurrent_end);
        colibri_gpu_event_destroy(profile.shared_start);
        colibri_gpu_event_destroy(profile.shared_end);
        colibri_gpu_event_destroy(profile.expert_start);
        colibri_gpu_event_destroy(profile.expert_end);
    }
    runtime.cuda_layer_profiles.clear();
    colibri_gpu_event_destroy(runtime.cuda_tail_start);
    colibri_gpu_event_destroy(runtime.cuda_lm_start);
    colibri_gpu_event_destroy(runtime.cuda_lm_end);
    colibri_gpu_event_destroy(runtime.cuda_tail_end);
    runtime.cuda_tail_start=runtime.cuda_lm_start=0;
    runtime.cuda_lm_end=runtime.cuda_tail_end=0;
    runtime.cuda_profile=false;
    runtime.cuda_graphs=false;
    runtime.cuda_graph_builds=runtime.cuda_graph_replays=runtime.cuda_graph_fallbacks=0;
    runtime.cuda_graph_trace_reported=false;
    colibri_gpu_stream_destroy(runtime.prefetch_stream);
    runtime.prefetch_stream = 0;
    colibri_gpu_stream_destroy(runtime.graph_stream);
    runtime.graph_stream = 0;
    colibri_gpu_stream_destroy(runtime.stream);
    runtime.device_tensors.clear();
    runtime.host_staging = nullptr;
    runtime.expert_staging = runtime.state = runtime.workspace = 0;
    runtime.expert_cache = 0;
    runtime.expert_native_cache = 0;
    runtime.static_arena = runtime.stream = 0;
    runtime.route_event = 0;
    runtime.prefill_layer_start_event=runtime.prefill_core_end_event=
        runtime.prefill_router_end_event=0;
    runtime.prefill_profile=false;
    runtime.static_arena_bytes = runtime.workspace_bytes = 0;
    runtime.decode_workspace_layout = {};
    runtime.rows_workspace_layout = {};
    runtime.decode_host_layout = {};
    runtime.rows_host_layout = {};
    runtime.state_bytes = runtime.expert_staging_bytes = 0;
    runtime.host_staging_bytes = 0;
    runtime.forward_rows_capacity = 0;
    runtime.expert_cache_bytes = runtime.expert_native_cache_bytes =
        runtime.expert_slot_bytes = 0;
    runtime.expert_slots.clear();
    runtime.whole_expert_layer_slots.clear();
    runtime.expert_history.clear();
    runtime.expert_residency.clear();
    runtime.decode_ready = false;
}

uint64_t align_to(uint64_t n, uint32_t a) { return (n + a - 1) / a * a; }

std::uint64_t available_host_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength=sizeof(status);
    return GlobalMemoryStatusEx(&status)?status.ullAvailPhys:0;
#elif defined(__linux__)
    // MemAvailable includes reclaimable page cache. That matters for mmap'd
    // GGUFs: after one pass most model bytes are cached, while AVPHYS counts
    // only free pages and would incorrectly disable direct paging.
    std::ifstream memory("/proc/meminfo");
    std::string key,unit;
    std::uint64_t kib=0;
    while(memory>>key>>kib>>unit)
        if(key=="MemAvailable:")return kib*1024;
    return 0;
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const auto pages=sysconf(_SC_AVPHYS_PAGES),page_size=sysconf(_SC_PAGESIZE);
    return pages>0&&page_size>0?static_cast<std::uint64_t>(pages)*page_size:0;
#else
    return 0;
#endif
}
void copy_text(char* dst, size_t cap, const std::string& value) { if (!cap) return; std::strncpy(dst, value.c_str(), cap-1); dst[cap-1]=0; }

// Qwen3-Next GGUF files include the optional MTP draft block in ``block_count``.
// A block carrying ``nextn`` tensors is not part of the causal decoder stack and
// must not be executed before the final norm. Split checkpoints only see their
// full tensor list once the shards are merged, so this runs again after the
// merge rather than only at the end of the shard's own parse.
void detect_mtp_layer(ColibriV2Model& m) {
    for (const auto& tensor : m.tensors) {
        const auto marker = tensor.name.find(".nextn.");
        if (marker == std::string::npos || tensor.name.rfind("blk.", 0) != 0) continue;
        try {
            const auto draft_layer = static_cast<uint32_t>(std::stoul(tensor.name.substr(4, marker - 4)));
            if (m.mtp_layer != std::numeric_limits<uint32_t>::max() &&
                m.mtp_layer != draft_layer) {
                throw std::runtime_error("multiple Qwen MTP draft layers are unsupported");
            }
            m.mtp_layer = draft_layer;
            if (draft_layer < m.config.layer_count) m.config.layer_count = draft_layer;
        } catch (const std::exception&) {
            throw std::runtime_error("invalid Qwen MTP tensor layer index");
        }
    }
}

int parse(ColibriV2Model& m) {
    if (m.size < 24 || std::memcmp(m.data,"GGUF",4)!=0) throw std::runtime_error("not a GGUF file");
    Reader r{m.data+4,m.data+m.size}; m.version=r.get<uint32_t>(); if (m.version<2 || m.version>3) throw std::runtime_error("unsupported GGUF version");
    uint64_t count=r.get<uint64_t>(); m.metadata=r.get<uint64_t>();
    auto read_uint = [&](uint32_t type)->uint64_t { if(type==4)return r.get<uint32_t>(); if(type==10)return r.get<uint64_t>(); throw std::runtime_error("GGUF architecture value is not an integer"); };
    auto is_integer = [](uint32_t type){ return type==0||type==1||type==2||type==3||type==4||type==5||type==10||type==11; };
    auto read_any_uint = [&](uint32_t type)->uint64_t {
        std::int64_t value=0;
        switch(type){
        case 0: value=r.get<std::uint8_t>(); break;
        case 1: value=r.get<std::int8_t>(); break;
        case 2: value=r.get<std::uint16_t>(); break;
        case 3: value=r.get<std::int16_t>(); break;
        case 4: value=r.get<std::uint32_t>(); break;
        case 5: value=r.get<std::int32_t>(); break;
        case 10: return r.get<std::uint64_t>();
        case 11: value=r.get<std::int64_t>(); break;
        default: throw std::runtime_error("GGUF value is not an integer");
        }
        if(value<0)throw std::runtime_error("GGUF integer value is negative");
        return static_cast<std::uint64_t>(value);
    };
    auto read_float = [&](uint32_t type)->float { if(type==6)return r.get<float>(); if(type==12)return static_cast<float>(r.get<double>()); throw std::runtime_error("GGUF architecture value is not a float"); };
    auto read_uint_array = [&](uint32_t type)->std::vector<std::uint32_t> {
        if(type!=9)throw std::runtime_error("GGUF architecture value is not an array");
        const auto element_type=r.get<uint32_t>();const auto count=r.get<uint64_t>();
        if(element_type!=4&&element_type!=5&&element_type!=10&&element_type!=11)
            throw std::runtime_error("GGUF architecture array is not integer");
        std::vector<std::uint32_t> values;values.reserve(static_cast<std::size_t>(count));
        for(std::uint64_t i=0;i<count;++i){
            std::int64_t value=0;
            if(element_type==4)value=r.get<std::uint32_t>();
            else if(element_type==5)value=r.get<std::int32_t>();
            else if(element_type==10)value=static_cast<std::int64_t>(r.get<std::uint64_t>());
            else value=r.get<std::int64_t>();
            if(value<0||static_cast<std::uint64_t>(value)>std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("GGUF architecture array value is out of range");
            values.push_back(static_cast<std::uint32_t>(value));
        }
        return values;
    };
    auto read_float_array = [&](uint32_t type)->std::vector<float> {
        if(type!=9)throw std::runtime_error("GGUF architecture value is not an array");
        const auto element_type=r.get<uint32_t>();const auto count=r.get<uint64_t>();
        if(element_type!=6&&element_type!=12)
            throw std::runtime_error("GGUF architecture array is not float");
        std::vector<float> values;values.reserve(static_cast<std::size_t>(count));
        for(std::uint64_t i=0;i<count;++i)
            values.push_back(element_type==6?r.get<float>():static_cast<float>(r.get<double>()));
        return values;
    };
    auto set_config = [&](const std::string& key, uint32_t type)->bool {
        if(type!=4 && type!=10)return false;
        auto suffix=[&](const char* text){size_t n=std::strlen(text);return key.size()>=n && key.compare(key.size()-n,n,text)==0;};
        uint32_t* target=nullptr;
        // DeepSeek-V4 scalars first: several of them end in strings that would
        // otherwise be caught by the shorter generic suffixes below.
        if(suffix(".attention.q_lora_rank"))target=&m.config.q_lora_rank;
        else if(suffix(".attention.kv_lora_rank"))target=&m.config.kv_lora_rank;
        else if(suffix(".attention.output_lora_rank"))target=&m.config.output_lora_rank;
        else if(suffix(".attention.output_group_count"))target=&m.config.output_group_count;
        else if(suffix(".attention.indexer.head_count"))target=&m.config.indexer_head_count;
        else if(suffix(".attention.indexer.key_length"))target=&m.config.indexer_key_length;
        else if(suffix(".attention.indexer.top_k"))target=&m.config.indexer_top_k;
        else if(suffix(".hyper_connection.sinkhorn_iterations"))target=&m.config.sinkhorn_iterations;
        else if(suffix(".hyper_connection.count"))target=&m.config.hyper_connection_count;
        else if(suffix(".expert_shared_count"))target=&m.config.expert_shared_count;
        else if(suffix(".hash_layer_count"))target=&m.config.hash_layer_count;
        else if(suffix(".block_size"))target=&m.config.draft_block_size;
        else if(suffix(".embedding_length"))target=&m.config.hidden_size;
        else if(suffix(".embedding_length_per_layer_input"))target=&m.config.per_layer_embedding_size;
        else if(suffix(".block_count"))target=&m.config.layer_count;
        else if(suffix(".attention.shared_kv_layers"))target=&m.config.shared_kv_layers;
        else if(suffix(".attention.head_count_kv"))target=&m.config.attention_kv_heads;
        else if(suffix(".attention.head_count"))target=&m.config.attention_heads;
        else if(suffix(".context_length"))target=&m.config.context_length;
        else if(suffix(".expert_shared_feed_forward_length"))target=&m.config.expert_shared_intermediate_size;
        else if(suffix(".expert_feed_forward_length"))target=&m.config.expert_intermediate_size;
        else if(suffix(".feed_forward_length"))target=&m.config.dense_intermediate_size;
        else if(suffix(".expert_count"))target=&m.config.expert_count;
        else if(suffix(".expert_used_count"))target=&m.config.expert_used_count;
        else if(suffix(".leading_dense_block_count"))target=&m.config.leading_dense_block_count;
        else if(suffix(".expert_gating_func"))target=&m.config.expert_gating_func;
        else if(suffix(".rope.scaling.original_context_length"))target=&m.config.rope_original_context_length;
        else if(key=="tokenizer.ggml.vocab_size")target=&m.config.vocabulary_size;
        if(!target)return false;
        *target=static_cast<uint32_t>(read_uint(type));
        return true;
    };
    for(uint64_t i=0;i<m.metadata;i++) { std::string key=r.str(); uint32_t type=r.get<uint32_t>();
        if (key=="general.alignment" && type==4) m.alignment=r.get<uint32_t>();
        else if (key=="general.architecture" && type==8) {m.architecture=r.str();m.config.architecture=m.architecture;}
        else if (key=="general.name" && type==8) m.name=r.str();
        // gguf-split writes the shard index and count as uint16 and the tensor
        // total as int32, so these do not go through read_uint.
        else if (key=="split.no" && is_integer(type)) m.split_no=static_cast<uint32_t>(read_any_uint(type));
        else if (key=="split.count" && is_integer(type)) m.split_count=static_cast<uint32_t>(read_any_uint(type));
        else if (key=="split.tensors.count" && is_integer(type)) m.split_tensors=read_any_uint(type);
        else if (key=="tokenizer.chat_template" && type==8) m.chat_template=r.str();
        else if (key=="tokenizer.ggml.tokens" && type==9) {uint32_t element_type=r.get<uint32_t>();uint64_t count_tokens=r.get<uint64_t>();m.config.vocabulary_size=static_cast<uint32_t>(count_tokens);m.vocabulary.reserve(static_cast<size_t>(count_tokens));for(uint64_t token_index=0;token_index<count_tokens;token_index++){if(element_type==8)m.vocabulary.push_back(r.str());else r.value(element_type);}}
        else if (key=="tokenizer.ggml.pre" && type==8) m.tokenizer_pre=r.str();
        else if (key=="tokenizer.ggml.eos_token_id" && (type==4||type==10)) m.config.eos_token_id=static_cast<uint32_t>(read_uint(type));
        else if (key=="tokenizer.ggml.eot_token_id" && (type==4||type==10)) m.config.eot_token_id=static_cast<uint32_t>(read_uint(type));
        else if (key=="tokenizer.ggml.bos_token_id" && (type==4||type==10)) m.config.bos_token_id=static_cast<uint32_t>(read_uint(type));
        else if (key=="tokenizer.ggml.mask_token_id" && is_integer(type)) m.config.mask_token_id=static_cast<uint32_t>(read_any_uint(type));
        else if (key=="tokenizer.ggml.token_type" && type==9) m.token_types=read_uint_array(type);
        else if (key=="tokenizer.ggml.merges" && type==9) {uint32_t element_type=r.get<uint32_t>();uint64_t count_merges=r.get<uint64_t>();m.merges.reserve(static_cast<size_t>(count_merges));for(uint64_t merge_index=0;merge_index<count_merges;merge_index++){if(element_type==8)m.merges.push_back(r.str());else r.value(element_type);}}
        else if (key.size()>=21 && key.compare(key.size()-21,21,".rope.dimension_count")==0 && (type==4 || type==10)) m.config.rotary_dimension=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=24 && key.compare(key.size()-24,24,".full_attention_interval")==0 && (type==4 || type==10)) m.config.full_attention_interval=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=24 && key.compare(key.size()-24,24,".attention.head_count_kv")==0 && type==9) m.config.attention_kv_heads_by_layer=read_uint_array(type);
        else if (key.size()>=21 && key.compare(key.size()-21,21,".attention.head_count")==0 && type==9) m.config.attention_heads_by_layer=read_uint_array(type);
        else if (key.size()>=25 && key.compare(key.size()-25,25,".attention.sliding_window")==0 && (type==4 || type==10)) m.config.sliding_window=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=33 && key.compare(key.size()-33,33,".attention.sliding_window_pattern")==0 && type==9) {
            const auto element_type=r.get<uint32_t>();
            const auto elements=r.get<uint64_t>();
            if(element_type!=7)throw std::runtime_error("GGUF sliding-window pattern must be boolean");
            m.config.sliding_window_pattern.reserve(static_cast<std::size_t>(elements));
            for(std::uint64_t element=0;element<elements;++element)m.config.sliding_window_pattern.push_back(r.get<std::uint8_t>()?1:0);
        }
        else if (key.size()>=33 && key.compare(key.size()-33,33,".attention.layer_norm_rms_epsilon")==0 && (type==6 || type==12)) m.config.rms_norm_epsilon=read_float(type);
        else if (key.size()>=21 && key.compare(key.size()-21,21,".attention.key_length")==0 && (type==4 || type==10)) m.config.key_length=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=23 && key.compare(key.size()-23,23,".attention.value_length")==0 && (type==4 || type==10)) m.config.value_length=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=25 && key.compare(key.size()-25,25,".attention.key_length_swa")==0 && (type==4 || type==10)) m.config.key_length_swa=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=27 && key.compare(key.size()-27,27,".attention.value_length_swa")==0 && (type==4 || type==10)) m.config.value_length_swa=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=25 && key.compare(key.size()-25,25,".rope.dimension_count_swa")==0 && (type==4 || type==10)) m.config.rotary_dimension_swa=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=19 && key.compare(key.size()-19,19,".rope.freq_base_swa")==0 && (type==6 || type==12)) m.config.rope_freq_base_swa=read_float(type);
        else if (key.size()>=24 && key.compare(key.size()-24,24,".final_logit_softcapping")==0 && (type==6 || type==12)) m.config.final_logit_softcap=read_float(type);
        else if (key.size()>=15 && key.compare(key.size()-15,15,".rope.freq_base")==0 && (type==6 || type==12)) m.config.rope_freq_base=read_float(type);
        else if (key.size()>=20 && key.compare(key.size()-20,20,".rope.scaling.factor")==0 && (type==6 || type==12)) m.config.rope_scaling_factor=read_float(type);
        else if (key.size()>=30 && key.compare(key.size()-30,30,".rope.scaling.yarn_attn_factor")==0 && (type==6 || type==12)) m.config.yarn_attn_factor=read_float(type);
        else if (key.size()>=28 && key.compare(key.size()-28,28,".rope.scaling.yarn_beta_fast")==0 && (type==6 || type==12)) m.config.yarn_beta_fast=read_float(type);
        else if (key.size()>=28 && key.compare(key.size()-28,28,".rope.scaling.yarn_beta_slow")==0 && (type==6 || type==12)) m.config.yarn_beta_slow=read_float(type);
        else if (key.size()>=18 && key.compare(key.size()-18,18,".rope.scaling.type")==0 && type==8) m.config.rope_scaling_yarn=r.str()=="yarn";
        else if (key.size()>=21 && key.compare(key.size()-21,21,".expert_weights_scale")==0 && (type==6 || type==12)) m.config.expert_weights_scale=read_float(type);
        else if (key.size()>=20 && key.compare(key.size()-20,20,".expert_weights_norm")==0 && type==7) m.config.expert_weights_norm=r.get<std::uint8_t>()!=0;
        else if (key.size()>=26 && key.compare(key.size()-26,26,".attention.compress_ratios")==0 && type==9) m.config.compress_ratios=read_uint_array(type);
        else if (key.size()>=14 && key.compare(key.size()-14,14,".target_layers")==0 && type==9) m.config.target_layers=read_uint_array(type);
        else if (key.size()>=34 && key.compare(key.size()-34,34,".attention.compress_rope_freq_base")==0 && (type==6 || type==12)) m.config.compress_rope_freq_base=read_float(type);
        else if (key.size()>=25 && key.compare(key.size()-25,25,".hyper_connection.epsilon")==0 && (type==6 || type==12)) m.config.sinkhorn_epsilon=read_float(type);
        else if (key.size()>=19 && key.compare(key.size()-19,19,".swiglu_clamp_shexp")==0 && type==9) m.config.swiglu_clamp_shexp=read_float_array(type);
        else if (key.size()>=17 && key.compare(key.size()-17,17,".swiglu_clamp_exp")==0 && type==9) m.config.swiglu_clamp_exp=read_float_array(type);
        else if (set_config(key,type)) {}
        else r.value(type);
    }
    m.tensors.reserve(static_cast<size_t>(count));
    for(uint64_t i=0;i<count;i++) { Tensor t; t.name=r.str(); auto dims=r.get<uint32_t>(); if(dims>4) throw std::runtime_error("GGUF rank exceeds v2 ABI"); for(uint32_t d=0;d<dims;d++) t.shape.push_back(r.get<uint64_t>()); t.type=r.get<uint32_t>(); t.offset=r.get<uint64_t>(); m.tensors.push_back(std::move(t)); }
    detect_mtp_layer(m);
    // An array-valued head count leaves the scalar unset. Workspaces and score
    // buffers are sized off the scalar, so it has to cover the widest layer.
    if(!m.config.attention_heads&&!m.config.attention_heads_by_layer.empty())
        m.config.attention_heads=*std::max_element(
            m.config.attention_heads_by_layer.begin(),
            m.config.attention_heads_by_layer.end());
    // Laguna does not ship a sliding-window pattern: the layout is implied by
    // the architecture as a period-4 cycle that starts with a full-attention
    // layer (12 full, 36 sliding for the 48-block S checkpoint).  The per-layer
    // head-count array is the independent witness of that layout, so cross-check
    // against it rather than trusting the period blindly.
    if(m.architecture=="laguna"&&m.config.sliding_window&&
       m.config.sliding_window_pattern.empty()&&m.config.layer_count){
        m.config.sliding_window_pattern.assign(m.config.layer_count,0);
        for(std::uint32_t layer=0;layer<m.config.layer_count;++layer)
            m.config.sliding_window_pattern[layer]=layer%4?1:0;
        const auto& heads=m.config.attention_heads_by_layer;
        if(heads.size()>=m.config.layer_count)
            for(std::uint32_t layer=0;layer<m.config.layer_count;++layer)
                if((heads[layer]==heads[0])!=(m.config.sliding_window_pattern[layer]==0))
                    throw std::runtime_error(
                        "Laguna sliding-window layout disagrees with the per-layer head count");
    }
    // DeepSeek-V4 ships no explicit kv_lora_rank: `attention.key_length` is the
    // width of the compressed KV latent, and the decoupled RoPE half is carried
    // separately in `rope.dimension_count`. The shared expert likewise arrives
    // as a count of routed-expert widths rather than a width.
    if(m.architecture=="deepseek4"||m.architecture=="dflash"){
        if(!m.config.kv_lora_rank)m.config.kv_lora_rank=m.config.key_length;
        if(!m.config.expert_shared_intermediate_size&&m.config.expert_shared_count)
            m.config.expert_shared_intermediate_size=
                m.config.expert_intermediate_size*m.config.expert_shared_count;
        if(!m.config.compress_ratios.empty()&&
           m.config.compress_ratios.size()<m.config.layer_count)
            throw std::runtime_error(
                "deepseek4 compress-ratio array is shorter than the model layer count");
    }
    if(!m.config.sliding_window_pattern.empty()&&m.config.sliding_window_pattern.size()<m.config.layer_count)
        throw std::runtime_error("GGUF sliding-window pattern is shorter than the model layer count");
    if(!m.config.attention_kv_heads_by_layer.empty()&&m.config.attention_kv_heads_by_layer.size()<m.config.layer_count)
        throw std::runtime_error("GGUF per-layer KV-head array is shorter than the model layer count");
    if(!m.config.intermediate_size)m.config.intermediate_size=m.config.expert_intermediate_size?m.config.expert_intermediate_size:m.config.dense_intermediate_size;
    uint64_t data_offset=align_to(static_cast<uint64_t>(r.p-m.data),m.alignment ? m.alignment : 32); if(data_offset>m.size) throw std::runtime_error("GGUF tensor data is outside the file"); for(auto& t:m.tensors) { if(t.offset>m.size-data_offset) throw std::runtime_error("GGUF tensor offset out of bounds"); t.offset += data_offset; }
    for(size_t i=0;i<m.tensors.size();i++) { auto& t=m.tensors[i]; uint64_t next=m.size; for(auto const& other:m.tensors) if(other.offset>t.offset) next=std::min(next,other.offset); t.size=next-t.offset; }
    // Tokenizer lookup tables, built once per model: rebuilding these
    // ~150k-entry maps per tokenize call dominated short calls.
    for(int rank=0;rank<static_cast<int>(m.merges.size());rank++)m.merge_ranks[m.merges[rank]]=rank;
    for(uint32_t id=0;id<static_cast<uint32_t>(m.vocabulary.size());id++)m.vocabulary_ids.emplace(m.vocabulary[id],id);
    // Longest first, so a control token that prefixes another still matches the
    // longer one when scanning input left to right.
    for(uint32_t id=0;id<static_cast<uint32_t>(m.vocabulary.size());id++){
        if(id>=m.token_types.size()||m.token_types[id]!=3)continue;
        if(m.vocabulary[id].empty())continue;
        m.control_tokens.emplace_back(m.vocabulary[id],id);
    }
    std::sort(m.control_tokens.begin(),m.control_tokens.end(),
        [](const auto& left,const auto& right){
            return left.first.size()>right.first.size();
        });
    return 0;
}

std::uint32_t attention_window(const ColibriV2Model& model, std::uint32_t layer) {
    if(layer>=model.config.layer_count)throw std::runtime_error("attention layer index is out of range");
    if(!model.config.sliding_window)return 0;
    const auto& pattern=model.config.sliding_window_pattern;
    return pattern.empty()||pattern[layer]?model.config.sliding_window:0;
}

int fill(const Tensor& t, ColibriV2TensorInfo& out) { std::memset(&out,0,sizeof(out)); out.dimensions=static_cast<uint32_t>(t.shape.size()); std::copy(t.shape.begin(),t.shape.end(),out.shape); out.ggml_type=t.type; out.offset=t.offset; out.size=t.size; copy_text(out.name,sizeof(out.name),t.name); return 0; }

std::uint64_t tensor_index(const ColibriV2Model& model, const std::string& name) {
    for (std::uint64_t index = 0; index < model.tensors.size(); ++index) {
        if (model.tensors[index].name == name) return index;
    }
    throw std::runtime_error("required Qwen tensor is missing: " + name);
}

bool has_tensor(const ColibriV2Model& model, const std::string& name) {
    for (const auto& tensor : model.tensors) if (tensor.name == name) return true;
    return false;
}

std::uint64_t first_tensor_index(
    const ColibriV2Model& model, std::initializer_list<const char*> names
) {
    for (const char* name : names) if (has_tensor(model, name)) return tensor_index(model, name);
    throw std::runtime_error("required Qwen model tensor is missing");
}

void add_static_tensor(
    ColibriV2QwenRuntime& runtime, QwenLayerPlan& layer,
    const std::string& name
) {
    const auto index = tensor_index(*runtime.model, name);
    layer.static_tensors.push_back(index);
    runtime.static_tensor_bytes += runtime.model->tensors[index].size;
}

void build_qwen_plan(ColibriV2QwenRuntime& runtime) {
    auto& model = *runtime.model;
    runtime.token_embeddings = first_tensor_index(
        model, {"token_embd.weight", "model.embed_tokens.weight", "embed_tokens.weight"}
    );
    runtime.final_norm = first_tensor_index(
        model, {"output_norm.weight", "model.norm.weight", "norm.weight"}
    );
    runtime.lm_head = first_tensor_index(model, {"output.weight", "lm_head.weight"});
    runtime.lm_head_type = model.tensors[runtime.lm_head].type;
    runtime.static_tensor_bytes += model.tensors[runtime.token_embeddings].size;
    runtime.static_tensor_bytes += model.tensors[runtime.final_norm].size;
    runtime.static_tensor_bytes += model.tensors[runtime.lm_head].size;
    runtime.layers.reserve(model.config.layer_count);
    for (std::uint32_t layer_index = 0; layer_index < model.config.layer_count; ++layer_index) {
        const std::string prefix = "blk." + std::to_string(layer_index) + ".";
        QwenLayerPlan layer;
        layer.attention = has_tensor(model, prefix + "attn_q.weight");
        if(layer.attention)layer.attention_window=attention_window(model,layer_index);
        add_static_tensor(runtime, layer, prefix + "attn_norm.weight");
        if (layer.attention) {
            for (const char* suffix : {
                     "attn_q.weight", "attn_k.weight", "attn_v.weight",
                     "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"
                 }) add_static_tensor(runtime, layer, prefix + suffix);
            layer.attention_heads=model.config.attention_heads;
            layer.kv_heads=model.config.attention_kv_heads;
            layer.head_dim=static_cast<std::uint32_t>(model.tensors[layer.static_tensors[2]].shape[1]/layer.kv_heads);
            layer.rotary_dim=model.config.rotary_dimension?model.config.rotary_dimension:layer.head_dim;
            layer.rope_theta=model.config.rope_freq_base?model.config.rope_freq_base:1000000.0f;
        } else {
            for (const char* suffix : {
                     "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
                     "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
                     "ssm_dt.bias", "ssm_a", "ssm_norm.weight"
                 }) add_static_tensor(runtime, layer, prefix + suffix);
        }
        layer.dense_ffn = has_tensor(model, prefix + "ffn_gate.weight");
        if (layer.dense_ffn) {
            // Keeps post_attention_norm at the same slot the MoE layout uses,
            // so moe_base still addresses the feed-forward block.
            for (const char* suffix : {
                     "post_attention_norm.weight", "ffn_gate.weight",
                     "ffn_up.weight", "ffn_down.weight"
                 }) add_static_tensor(runtime, layer, prefix + suffix);
            runtime.layers.push_back(std::move(layer));
            continue;
        }
        for (const char* suffix : {
                 "post_attention_norm.weight", "ffn_gate_inp.weight",
                 "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                 "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"
             }) add_static_tensor(runtime, layer, prefix + suffix);
        const std::array<std::string, 3> experts = {
            prefix + "ffn_gate_exps.weight",
            prefix + "ffn_up_exps.weight",
            prefix + "ffn_down_exps.weight",
        };
        for (std::size_t role = 0; role < experts.size(); ++role) {
            layer.expert_tensors[role] = tensor_index(model, experts[role]);
            runtime.expert_tensor_bytes += model.tensors[layer.expert_tensors[role]].size;
        }
        {
            if (has_tensor(model, prefix + "ffn_gate_exps.scale")) layer.expert_gate_scale = tensor_index(model, prefix + "ffn_gate_exps.scale");
            if (has_tensor(model, prefix + "ffn_up_exps.scale")) layer.expert_up_scale = tensor_index(model, prefix + "ffn_up_exps.scale");
            if (has_tensor(model, prefix + "ffn_down_exps.scale")) layer.expert_down_scale = tensor_index(model, prefix + "ffn_down_exps.scale");
        }
        {
            // Shared-expert weight_scale_2. NVFP4 stores it out-of-line as a
            // 1-element f32 tensor; without it the shared branch comes out
            // ~30000x too large and swamps the residual stream.
            auto scalar_scale = [&](const std::string& name) {
                if (!has_tensor(model, name)) return 1.0f;
                const auto& st = model.tensors[tensor_index(model, name)];
                float value = 1.0f;
                std::memcpy(&value, tensor_data(model,st), sizeof(float));
                return value;
            };
            layer.shared_gate_scale = scalar_scale(prefix + "ffn_gate_shexp.scale");
            layer.shared_up_scale = scalar_scale(prefix + "ffn_up_shexp.scale");
            layer.shared_down_scale = scalar_scale(prefix + "ffn_down_shexp.scale");
        }
        runtime.layers.push_back(std::move(layer));
    }
    if (model.mtp_layer != std::numeric_limits<std::uint32_t>::max()) {
        const auto draft_index = model.mtp_layer;
        const std::string prefix = "blk." + std::to_string(draft_index) + ".";
        QwenLayerPlan draft;
        draft.attention = true;
        auto add_mtp_static = [&](const std::string& name) {
            const auto index = tensor_index(model, name);
            draft.static_tensors.push_back(index);
            runtime.mtp_tensor_bytes += model.tensors[index].size;
        };
        add_mtp_static(prefix + "attn_norm.weight");
        for (const char* suffix : {
                 "attn_q.weight", "attn_k.weight", "attn_v.weight",
                 "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"
             }) add_mtp_static(prefix + suffix);
        draft.dense_ffn = has_tensor(model, prefix + "ffn_gate.weight");
        if (draft.dense_ffn) {
            for (const char* suffix : {
                     "post_attention_norm.weight", "ffn_gate.weight",
                     "ffn_up.weight", "ffn_down.weight"
                 }) add_mtp_static(prefix + suffix);
        } else {
        for (const char* suffix : {
                 "post_attention_norm.weight", "ffn_gate_inp.weight",
                 "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                 "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"
             }) add_mtp_static(prefix + suffix);
        const std::array<std::string, 3> draft_experts = {
            prefix + "ffn_gate_exps.weight", prefix + "ffn_up_exps.weight",
            prefix + "ffn_down_exps.weight",
        };
        for (std::size_t role = 0; role < draft_experts.size(); ++role) {
            draft.expert_tensors[role] = tensor_index(model, draft_experts[role]);
            runtime.mtp_tensor_bytes += model.tensors[draft.expert_tensors[role]].size;
        }
        // NVFP4 weight_scale_2, exactly as for the ordinary blocks above; the
        // draft layer is quantized the same way and is just as unusable without it.
        if (has_tensor(model, prefix + "ffn_gate_exps.scale")) draft.expert_gate_scale = tensor_index(model, prefix + "ffn_gate_exps.scale");
        if (has_tensor(model, prefix + "ffn_up_exps.scale")) draft.expert_up_scale = tensor_index(model, prefix + "ffn_up_exps.scale");
        if (has_tensor(model, prefix + "ffn_down_exps.scale")) draft.expert_down_scale = tensor_index(model, prefix + "ffn_down_exps.scale");
        {
            auto draft_scalar_scale = [&](const std::string& name) {
                if (!has_tensor(model, name)) return 1.0f;
                const auto& st = model.tensors[tensor_index(model, name)];
                float value = 1.0f;
                std::memcpy(&value, tensor_data(model,st), sizeof(float));
                return value;
            };
            draft.shared_gate_scale = draft_scalar_scale(prefix + "ffn_gate_shexp.scale");
            draft.shared_up_scale = draft_scalar_scale(prefix + "ffn_up_shexp.scale");
            draft.shared_down_scale = draft_scalar_scale(prefix + "ffn_down_shexp.scale");
        }
        }
        const std::array<std::string, 4> special = {
            prefix + "nextn.eh_proj.weight", prefix + "nextn.enorm.weight",
            prefix + "nextn.hnorm.weight", prefix + "nextn.shared_head_norm.weight",
        };
        for (std::size_t role = 0; role < special.size(); ++role) {
            runtime.mtp_special_tensors[role] = tensor_index(model, special[role]);
            runtime.mtp_tensor_bytes += model.tensors[runtime.mtp_special_tensors[role]].size;
        }
        runtime.mtp_layer_plan = std::move(draft);
        runtime.mtp_available = true;
    }
    for(const auto&layer:runtime.layers)for(auto index:layer.static_tensors){const auto&t=model.tensors[index];if(t.shape.size()==2)runtime.scratch_elements=std::max(runtime.scratch_elements,static_cast<std::uint32_t>(t.shape[1]));}
    if(runtime.mtp_available)for(auto index:runtime.mtp_layer_plan.static_tensors){const auto&t=model.tensors[index];if(t.shape.size()==2)runtime.scratch_elements=std::max(runtime.scratch_elements,static_cast<std::uint32_t>(t.shape[1]));}
    // The feed-forward width comes from the gate projection: the shared expert's
    // for MoE checkpoints, the block's own for dense ones. Both sit one slot
    // past post_attention_norm plus the router that only MoE layers carry.
    const auto& front = runtime.layers.front();
    const std::size_t ffn_base = front.attention ? 7 : 10;
    const auto& gate = model.tensors[front.static_tensors[ffn_base + (front.dense_ffn ? 1 : 2)]];
    runtime.moe_intermediate=static_cast<std::uint32_t>(gate.shape[1]);
    // The dense SwiGLU stages gate and up contiguously so silu_mul can read one
    // buffer, so one scratch slot has to hold both halves.
    if(front.dense_ffn)
        runtime.scratch_elements=std::max(runtime.scratch_elements,2u*runtime.moe_intermediate);
}

// YaRN correction band. `beta_fast` and `beta_slow` are expressed as numbers of
// full rotations across the trained context; converting each to the rotary pair
// index that completes that many rotations gives the band over which the kernel
// blends extrapolation into interpolation.
void qwen_yarn_correction_dims(
    int rotary_dim, std::uint32_t original_context, float theta,
    float beta_fast, float beta_slow, float& low, float& high
) {
    if(rotary_dim<=0||!original_context||theta<=1.0f){low=0.0f;high=0.0f;return;}
    const auto dimension=[&](float rotations){
        if(rotations<=0.0f)return 0.0f;
        return static_cast<float>(rotary_dim)*
            std::log(static_cast<float>(original_context)/
                     (rotations*2.0f*3.14159265358979323846f))/
            (2.0f*std::log(theta));
    };
    low=std::max(0.0f,std::floor(dimension(beta_fast)));
    high=std::min(static_cast<float>(rotary_dim-1),std::ceil(dimension(beta_slow)));
    if(high<low)high=low;
}

// Slot of the block's feed-forward norm within static_tensors, which every
// later feed-forward tensor is addressed relative to. Laguna's attention blocks
// carry an extra gate projection ahead of it; Qwen's DeltaNet blocks carry the
// wider recurrent set.
std::size_t qwen_ffn_base(const ColibriV2QwenRuntime& runtime, const QwenLayerPlan& layer) {
    if(runtime.laguna)return 8;
    return layer.attention?7:10;
}

// Laguna (poolside): every block is full attention or sliding-window attention
// with a softplus per-head output gate, a single pre-FFN norm (no post-attention
// or post-FFN norm), a leading run of dense blocks, and sigmoid-routed MoE with
// a score-correction bias and one always-on ungated shared expert.
void build_laguna_plan(ColibriV2QwenRuntime& runtime) {
    auto& model=*runtime.model;
    runtime.laguna=true;
    runtime.token_embeddings=tensor_index(model,"token_embd.weight");
    runtime.final_norm=tensor_index(model,"output_norm.weight");
    // Laguna ships an untied head, but fall back to the table if a future
    // checkpoint ties them.
    runtime.lm_head=has_tensor(model,"output.weight")
        ?tensor_index(model,"output.weight"):runtime.token_embeddings;
    runtime.lm_head_type=model.tensors[runtime.lm_head].type;
    runtime.static_tensor_bytes+=model.tensors[runtime.token_embeddings].size;
    runtime.static_tensor_bytes+=model.tensors[runtime.final_norm].size;
    if(runtime.lm_head!=runtime.token_embeddings)
        runtime.static_tensor_bytes+=model.tensors[runtime.lm_head].size;
    const auto head_dim=model.config.key_length?model.config.key_length:128u;
    runtime.layers.reserve(model.config.layer_count);
    for(std::uint32_t layer_index=0;layer_index<model.config.layer_count;++layer_index){
        const std::string prefix="blk."+std::to_string(layer_index)+".";
        QwenLayerPlan layer;layer.attention=true;
        layer.attention_window=attention_window(model,layer_index);
        layer.attention_heads=model.config.attention_heads_by_layer.empty()
            ?model.config.attention_heads
            :model.config.attention_heads_by_layer[layer_index];
        layer.kv_heads=model.config.attention_kv_heads;
        layer.head_dim=head_dim;
        // Full-attention layers run YaRN over a partial rotary span with the
        // long-context theta; sliding-window layers run plain RoPE over the
        // whole head with the short theta.
        layer.rotary_dim=layer.attention_window
            ?(model.config.rotary_dimension_swa?model.config.rotary_dimension_swa:head_dim)
            :(model.config.rotary_dimension?model.config.rotary_dimension:head_dim);
        layer.rope_theta=layer.attention_window
            ?(model.config.rope_freq_base_swa?model.config.rope_freq_base_swa:10000.0f)
            :(model.config.rope_freq_base?model.config.rope_freq_base:500000.0f);
        if(!layer.attention_window&&model.config.rope_scaling_yarn&&
           model.config.rope_scaling_factor>1.0f){
            layer.rope_freq_scale=1.0f/model.config.rope_scaling_factor;
            layer.rope_ext_factor=1.0f;
            layer.rope_attn_factor=model.config.yarn_attn_factor;
            layer.rope_beta_fast=model.config.yarn_beta_fast;
            layer.rope_beta_slow=model.config.yarn_beta_slow;
            layer.rope_orig_context=model.config.rope_original_context_length
                ?model.config.rope_original_context_length:model.config.context_length;
        }
        for(const char* suffix:{
            "attn_norm.weight","attn_q.weight","attn_k.weight","attn_v.weight",
            "attn_output.weight","attn_q_norm.weight","attn_k_norm.weight",
            "attn_gate.weight","ffn_norm.weight"
        })add_static_tensor(runtime,layer,prefix+suffix);
        // The gate is per-head (one scalar broadcast over head_dim). The
        // per-element variant the larger Laguna checkpoints use would need a
        // different apply kernel, so reject it rather than silently mis-scale.
        const auto gate_width=model.tensors[layer.static_tensors[7]].shape[1];
        if(gate_width!=layer.attention_heads)
            throw std::runtime_error(
                "native Laguna supports only the per-head attention gate");
        layer.dense_ffn=has_tensor(model,prefix+"ffn_gate.weight");
        if(layer.dense_ffn){
            for(const char* suffix:{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"})
                add_static_tensor(runtime,layer,prefix+suffix);
        }else{
            // Same slot order the Qwen MoE block uses (router, then the shared
            // expert's gate/up/down) so the shared feed-forward code addresses
            // both architectures off moe_base. Laguna's shared expert has no
            // gate tensor, and its router bias hangs off the plan instead.
            for(const char* suffix:{
                "ffn_gate_inp.weight","ffn_gate_shexp.weight",
                "ffn_up_shexp.weight","ffn_down_shexp.weight"
            })add_static_tensor(runtime,layer,prefix+suffix);
            // Appended last so it lands past the shared expert's slots and the
            // Qwen feed-forward addressing is undisturbed; it still has to be a
            // static tensor to reach the device.
            add_static_tensor(runtime,layer,prefix+"exp_probs_b.bias");
            layer.router_bias=layer.static_tensors.back();
            const std::array<std::string,3> experts={
                prefix+"ffn_gate_exps.weight",
                prefix+"ffn_up_exps.weight",
                prefix+"ffn_down_exps.weight",
            };
            for(std::size_t role=0;role<experts.size();++role){
                layer.expert_tensors[role]=tensor_index(model,experts[role]);
                runtime.expert_tensor_bytes+=model.tensors[layer.expert_tensors[role]].size;
            }
        }
        for(auto index:layer.static_tensors){const auto&t=model.tensors[index];if(t.shape.size()==2)runtime.scratch_elements=std::max(runtime.scratch_elements,static_cast<std::uint32_t>(t.shape[1]));}
        runtime.layers.push_back(std::move(layer));
    }
    runtime.moe_intermediate=model.config.expert_intermediate_size;
    // The shared expert and the routed experts may be sized differently; both
    // SwiGLU stages write gate and up contiguously, so reserve the wider pair.
    const auto shared_intermediate=model.config.expert_shared_intermediate_size
        ?model.config.expert_shared_intermediate_size:runtime.moe_intermediate;
    runtime.scratch_elements=std::max(
        runtime.scratch_elements,
        2u*std::max({runtime.moe_intermediate,shared_intermediate,
                     model.config.dense_intermediate_size}));
}

void build_gemma4_plan(ColibriV2QwenRuntime& runtime) {
    auto& model=*runtime.model;
    runtime.gemma4=true;
    runtime.token_embeddings=tensor_index(model,"token_embd.weight");
    runtime.final_norm=tensor_index(model,"output_norm.weight");
    runtime.lm_head=runtime.token_embeddings;
    runtime.static_tensor_bytes+=model.tensors[runtime.token_embeddings].size;
    runtime.static_tensor_bytes+=model.tensors[runtime.final_norm].size;
    if(has_tensor(model,"rope_freqs.weight")){
        runtime.rope_factors=tensor_index(model,"rope_freqs.weight");
        runtime.static_tensor_bytes+=model.tensors[runtime.rope_factors].size;
    }
    runtime.layers.reserve(model.config.layer_count);
    for(std::uint32_t layer_index=0;layer_index<model.config.layer_count;++layer_index){
        const std::string prefix="blk."+std::to_string(layer_index)+".";
        QwenLayerPlan layer;layer.attention=true;
        layer.attention_window=attention_window(model,layer_index);
        layer.attention_heads=model.config.attention_heads;
        layer.kv_heads=model.config.attention_kv_heads_by_layer.empty()
            ?model.config.attention_kv_heads:model.config.attention_kv_heads_by_layer[layer_index];
        layer.head_dim=layer.attention_window
            ?(model.config.key_length_swa?model.config.key_length_swa:model.config.rotary_dimension_swa)
            :(model.config.key_length?model.config.key_length:model.config.rotary_dimension);
        layer.rotary_dim=layer.attention_window
            ?(model.config.rotary_dimension_swa?model.config.rotary_dimension_swa:layer.head_dim)
            :(model.config.rotary_dimension?model.config.rotary_dimension:layer.head_dim);
        layer.rope_theta=layer.attention_window
            ?(model.config.rope_freq_base_swa?model.config.rope_freq_base_swa:10000.0f)
            :(model.config.rope_freq_base?model.config.rope_freq_base:1000000.0f);
        for(const char* suffix:{"attn_norm.weight","attn_q.weight","attn_k.weight"})
            add_static_tensor(runtime,layer,prefix+suffix);
        // Gemma 4 global attention can be configured with K == V. Those
        // checkpoints intentionally omit attn_v.weight; re-projecting with K
        // gives the unnormalized value states before K takes its learned norm.
        if(has_tensor(model,prefix+"attn_v.weight"))
            add_static_tensor(runtime,layer,prefix+"attn_v.weight");
        else
            layer.static_tensors.push_back(layer.static_tensors[2]);
        for(const char* suffix:{
            "attn_output.weight","attn_q_norm.weight","attn_k_norm.weight",
            "post_attention_norm.weight","ffn_norm.weight","ffn_gate.weight",
            "ffn_up.weight","ffn_down.weight","post_ffw_norm_1.weight",
            "ffn_gate_inp.scale","ffn_gate_inp.weight","pre_ffw_norm_2.weight",
            "post_ffw_norm_2.weight","post_ffw_norm.weight","layer_output_scale.weight"
        })add_static_tensor(runtime,layer,prefix+suffix);
        layer.expert_tensors[0]=tensor_index(model,prefix+"ffn_gate_up_exps.weight");
        layer.expert_tensors[1]=tensor_index(model,prefix+"ffn_down_exps.weight");
        layer.expert_tensors[2]=tensor_index(model,prefix+"ffn_down_exps.scale");
        layer.expert_tensor_count=3;
        for(auto index:layer.expert_tensors)runtime.expert_tensor_bytes+=model.tensors[index].size;
        for(auto index:layer.static_tensors){const auto&t=model.tensors[index];if(t.shape.size()==2)runtime.scratch_elements=std::max(runtime.scratch_elements,static_cast<std::uint32_t>(t.shape[1]));}
        runtime.layers.push_back(std::move(layer));
    }
    runtime.moe_intermediate=model.config.expert_intermediate_size;
}

// qwen_half_value now lives in src/qwen_kquant.h so the contract tests can
// decode super-blocks without linking the whole runtime.

// bf16 is just the top half of an f32, so widening is a shift.
float qwen_bf16_value(std::uint16_t bits) {
    const std::uint32_t widened=static_cast<std::uint32_t>(bits)<<16;
    float value;std::memcpy(&value,&widened,sizeof(value));return value;
}

// Narrowing counterpart for Q8_0 block scales. Round-half-to-even keeps the
// scale error under the codes' own quantization error; a mantissa carry lands
// in the exponent field on its own because the fields are packed adjacently.
std::uint16_t qwen_half_bits(float value) {
    std::uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
    const std::uint32_t sign=(bits>>16)&0x8000u;
    const std::int32_t exponent=
        static_cast<std::int32_t>((bits>>23)&0xffu)-127+15;
    const std::uint32_t fraction=bits&0x7fffffu;
    if(exponent>=31)return static_cast<std::uint16_t>(sign|0x7c00u);
    // Block scales are absmax/127 of real weights, never subnormal in f16;
    // flushing is still the right answer for an all-zero block.
    if(exponent<=0)return static_cast<std::uint16_t>(sign);
    std::uint32_t packed=(static_cast<std::uint32_t>(exponent)<<10)|(fraction>>13);
    const std::uint32_t remainder=fraction&0x1fffu;
    if(remainder>0x1000u||(remainder==0x1000u&&(packed&1u)))++packed;
    return static_cast<std::uint16_t>(sign|packed);
}

// Pack `count` f32 values (a multiple of 32) into Q8_0 blocks: one f16 scale
// followed by 32 int8 codes. Matches qwen_q8_value and the q8 CUDA kernels.
void qwen_pack_q8_0(const float* values, std::uint64_t count, std::uint8_t* out) {
    for(std::uint64_t block=0;block*32<count;++block){
        const float* source=values+block*32;
        float absmax=0.0f;
        for(int i=0;i<32;++i)absmax=std::max(absmax,std::fabs(source[i]));
        const float scale=absmax/127.0f;
        // Quantize against the scale that will actually be stored, so decode
        // reproduces these codes exactly rather than the pre-rounding value.
        const std::uint16_t scale_bits=qwen_half_bits(scale);
        const float stored=qwen_half_value(scale_bits);
        const float inverse=stored>0.0f?1.0f/stored:0.0f;
        auto* destination=out+block*kQ8BlockSize;
        std::memcpy(destination,&scale_bits,2);
        for(int i=0;i<32;++i){
            const int code=static_cast<int>(std::lround(source[i]*inverse));
            destination[2+i]=static_cast<std::uint8_t>(
                static_cast<std::int8_t>(std::min(127,std::max(-127,code))));
        }
    }
}

float qwen_q5_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/kBlockElements;const int within=static_cast<int>(absolute&(kBlockElements-1));const auto*base=packed+block*kQ5KBlockSize;std::uint16_t d_bits=0,dmin_bits=0;std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);const auto*scales=base+4;const int group=within/64,offset=within&63,sub=offset/32,qindex=group*32+(offset&31);const int bit=(base[16+(offset&31)]>>(2*group+sub))&1;const int quant=((offset<32)?(base[48+qindex]&15):(base[48+qindex]>>4))+16*bit;const int index=group*2+sub;int scale=0,minimum=0;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}return qwen_half_value(d_bits)*scale*quant-qwen_half_value(dmin_bits)*minimum;}
float qwen_q6_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/kBlockElements;const int within=static_cast<int>(absolute&(kBlockElements-1));const auto*base=packed+block*kQ6KBlockSize;const auto*ql=base;const auto*qh=base+128;const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);std::uint16_t d_bits=0;std::memcpy(&d_bits,base+208,2);const int half=within/128,offset=within&127,lane=offset/32,l=offset&31,qindex=l+((lane==0||lane==2)?0:32);const auto qbyte=ql[half*64+qindex],high=qh[half*32+l];const int nibble=(lane==0||lane==1)?(qbyte&15):(qbyte>>4);const int quant=(nibble|(((high>>(lane*2))&3)<<4))-32;const int scale_index=half*8+(l/16)+lane*2;return qwen_half_value(d_bits)*scales[scale_index]*quant;}
float qwen_q8_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/32,within=absolute&31;std::uint16_t scale_bits=0;std::memcpy(&scale_bits,packed+block*kQ8BlockSize,2);std::int8_t value=0;std::memcpy(&value,packed+block*kQ8BlockSize+2+within,1);return qwen_half_value(scale_bits)*value;}
// Q2_K and Q3_K element decoders are shared with the SIMD kernels and the
// contract tests; see src/qwen_kquant.h for the block layouts.
// Q4_K super-block (GGML type 12): 144 bytes per 256 values -> d(2) dmin(2)
// scales[12] (same 6-bit packing as Q5_K) qs[128] (4-bit). Like Q5_K but with
// no 5th-bit array; value = d*scale*q - dmin*min.
float qwen_q4k_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/kBlockElements;const int within=static_cast<int>(absolute&(kBlockElements-1));const auto*base=packed+block*kQ4KBlockSize;std::uint16_t d_bits=0,dmin_bits=0;std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);const auto*scales=base+4;const int group=within/64,offset=within&63,sub=offset/32,qindex=group*32+(offset&31);const int quant=(offset<32)?(base[16+qindex]&15):(base[16+qindex]>>4);const int index=group*2+sub;int scale=0,minimum=0;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}return qwen_half_value(d_bits)*scale*quant-qwen_half_value(dmin_bits)*minimum;}
// NVFP4 (GGML type 40): E2M1 4-bit float (1 sign + 2 exp + 1 mantissa, bias=1).
// Verified against a real NVFP4 checkpoint (Qwen3.6-35B-Fast-NVFP4.gguf):
// 36 bytes per 64 elements -- d[4] E4M3 scales then qs[32] packed nibbles, with
// scale i governing bytes 4+8i..11+8i (16 elements). Nibbles are split-half like
// q4_0/q4_K: qs[lane] holds element `lane` in the low nibble and `lane+8` in the
// high nibble, relative to the sub-block. (Byte-position entropy pins the scales
// to bytes 0-3; every sub-block's largest magnitude decodes to exactly 6.0,
// confirming the grouping.)
// This layout must stay in lockstep with nvfp4_value() in
// colibri_v2_qwen_kernels.hpp -- the CPU and GPU expert paths decode the very
// same bytes, and the expert-scale folding in the MoE staging path rewrites the
// four scale bytes of every 36-byte block in place.
// E2M1 LUT: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives).
constexpr float kNvfp4Lut[16]={
    0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
    0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
// OCP "FN" E4M3, which is what NVFP4 block scales use: e==0xF is FINITE (up to
// 448 at 0x7E) and only 0x7F/0xFF are NaN -- there are no infinities. Real
// checkpoints do reach e==0xF (0.34% of scales in Qwen3.6-35B-Fast-NVFP4), so
// decoding that exponent as infinity poisons whole blocks with NaN.
float ue4m3_to_float(std::uint8_t bits){
    const int s=(bits>>7)&1;const int e=(bits>>3)&0xF;const int m=bits&7;
    float val;
    if(e==0xF&&m==7){val=std::numeric_limits<float>::quiet_NaN();}
    else if(e==0){val=(m/8.0f)*std::exp2(-6.0f);}
    else{val=std::exp2(static_cast<float>(e-7))*(1.0f+m/8.0f);}
    return s?-val:val;
}
// NOTE: there is deliberately no float_to_ue4m3 here. NVFP4's per-tensor
// weight_scale_2 runs ~3e-5, far below E4M3's smallest subnormal (2^-9), so
// folding it back into the block scales flushes ~56% of them to zero. The scale
// is carried in f32 to the kernels instead.
float qwen_nvfp4_value(const std::uint8_t*packed,std::uint64_t absolute){
    const std::uint64_t block=absolute/kNvfp4BlockElements;
    const int offset=static_cast<int>(absolute%kNvfp4BlockElements);
    const int sub=offset/kNvfp4SubBlock,within=offset%kNvfp4SubBlock;
    const auto*base=packed+block*kNvfp4BlockSize;
    const float scale=ue4m3_to_float(base[sub]);
    const auto byte=base[4+sub*8+(within&7)];
    const int val=(within<8)?(byte&0x0F):(byte>>4);
    return scale*kNvfp4Lut[val];
}
float qwen_quant_dot(const std::uint8_t*packed,std::uint32_t type,const float*input,int elements,std::uint64_t row){
    if(qwen_simd_quant_type(type)&&(colibri_cpu_features()&2u)!=0&&elements%kBlockElements==0)return qwen_quant_dot_avx512(packed,type,input,elements,row);
    if(type==17&&(colibri_cpu_features()&2u)!=0&&elements%256==0){
        const char*setting=std::getenv("COLIBRI_IQ_AVX512");
        if(!setting||setting[0]!='0')
            return qwen_quant_dot_avx512(packed,type,input,elements,row);
    }
    if(type==40&&(colibri_cpu_features()&1u)!=0&&elements%kNvfp4BlockElements==0)return qwen_quant_dot_avx2(packed,type,input,elements,row);
    // The IQ codebook formats decode a branch per weight in scalar form, which
    // is what made low-bit MoE decode compute-bound rather than bandwidth-bound.
    if((type==16||type==17||type==18||type==23)&&(colibri_cpu_features()&1u)!=0&&elements%256==0)
        return qwen_quant_dot_avx2(packed,type,input,elements,row);
    if(qwen_simd_quant_type(type)&&(colibri_cpu_features()&1u)!=0&&elements%kBlockElements==0)return qwen_quant_dot_avx2(packed,type,input,elements,row);
    float result=0.0f;
    if(type==2){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/32)*18;
        for(int block=0;block<elements/32;++block){const auto*base=row_data+block*18;std::uint16_t scale_bits=0;std::memcpy(&scale_bits,base,2);const float scale=qwen_half_value(scale_bits);const auto*vector=input+block*32;for(int lane=0;lane<16;++lane){const auto byte=base[2+lane];result+=scale*((byte&15)-8)*vector[lane];result+=scale*((byte>>4)-8)*vector[lane+16];}}
    }else if(type==13){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/kBlockElements)*kQ5KBlockSize;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*kQ5KBlockSize;std::uint16_t d_bits=0,dmin_bits=0;
            std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);
            const float d=qwen_half_value(d_bits),dmin=qwen_half_value(dmin_bits);
            const auto*scales=base+4;const auto*high=base+16;const auto*low=base+48;
            const auto*vector=input+block*256;
            for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){
                const int scale_index=group*2+sub;int scale=0,minimum=0;
                if(scale_index<4){scale=scales[scale_index]&63;minimum=scales[scale_index+4]&63;}
                else{scale=(scales[scale_index+4]&15)|((scales[scale_index-4]>>6)<<4);minimum=(scales[scale_index+4]>>4)|((scales[scale_index]>>6)<<4);}
                const float ds=d*scale,dm=dmin*minimum;const int shift=2*group+sub;
                const auto*values=vector+group*64+sub*32;const auto*quants=low+group*32;
                for(int lane=0;lane<32;++lane){const int quant=((sub==0)?(quants[lane]&15):(quants[lane]>>4))+16*((high[lane]>>shift)&1);result+=(ds*quant-dm)*values[lane];}
            }
        }
    }else if(type==14){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/kBlockElements)*kQ6KBlockSize;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*kQ6KBlockSize;const auto*ql=base;const auto*qh=base+128;
            const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);std::uint16_t d_bits=0;
            std::memcpy(&d_bits,base+208,2);const float d=qwen_half_value(d_bits);const auto*vector=input+block*256;
            for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){
                const int q_offset=(segment==0||segment==2)?0:32;const int shift=segment*2;
                const auto*values=vector+half*128+segment*32;
                for(int lane=0;lane<32;++lane){const auto qbyte=ql[half*64+q_offset+lane];const int nibble=(segment<2)?(qbyte&15):(qbyte>>4);const int quant=(nibble|(((qh[half*32+lane]>>shift)&3)<<4))-32;const int scale_index=half*8+(lane/16)+segment*2;result+=d*scales[scale_index]*quant*values[lane];}
            }
        }
    }else if(type==8){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/32)*kQ8BlockSize;
        for(int block=0;block<elements/32;++block){const auto*base=row_data+block*kQ8BlockSize;std::uint16_t scale_bits=0;std::memcpy(&scale_bits,base,2);const float scale=qwen_half_value(scale_bits);const auto*values=reinterpret_cast<const std::int8_t*>(base+2);const auto*vector=input+block*32;for(int lane=0;lane<32;++lane)result+=scale*values[lane]*vector[lane];}
    }else if(type==12){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/kBlockElements)*kQ4KBlockSize;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*kQ4KBlockSize;std::uint16_t d_bits=0,dmin_bits=0;
            std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);
            const float d=qwen_half_value(d_bits),dmin=qwen_half_value(dmin_bits);
            const auto*scales=base+4;const auto*low=base+16;
            const auto*vector=input+block*256;
            for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){
                const int scale_index=group*2+sub;int scale=0,minimum=0;
                if(scale_index<4){scale=scales[scale_index]&63;minimum=scales[scale_index+4]&63;}
                else{scale=(scales[scale_index+4]&15)|((scales[scale_index-4]>>6)<<4);minimum=(scales[scale_index+4]>>4)|((scales[scale_index]>>6)<<4);}
                const float ds=d*scale,dm=dmin*minimum;
                const auto*values=vector+group*64+sub*32;const auto*quants=low+group*32;
                for(int lane=0;lane<32;++lane){const int quant=(sub==0)?(quants[lane]&15):(quants[lane]>>4);result+=(ds*quant-dm)*values[lane];}
            }
        }
    }else if(type==10){
        result+=qwen_q2k_dot_row(packed,input,elements,row);
    }else if(type==11){
        result+=qwen_q3k_dot_row(packed,input,elements,row);
    }else if(type==16){
        result+=qwen_iq2xxs_dot_row(packed,input,elements,row);
    }else if(type==18){
        result+=qwen_iq3xxs_dot_row(packed,input,elements,row);
    }else if(type==22){
        result+=qwen_iq2s_dot_row(packed,input,elements,row);
    }else if(type==21){
        result+=qwen_iq3s_dot_row(packed,input,elements,row);
    }else if(type==17){
        result+=qwen_iq2xs_dot_row(packed,input,elements,row);
    }else if(type==23){
        result+=qwen_iq4xs_dot_row(packed,input,elements,row);
    }else if(type==19){
        // IQ1_S. Each group of eight weights is one 11-bit index into a shared
        // grid -- eight bits from qs, three more from qh -- which is what makes
        // 1.5 bits a weight possible. qh also carries the group's scale in bits
        // 12-14 and, in its top bit, the sign of a delta applied to every weight
        // in the group.
        const int blocks=elements/256;
        const std::uint64_t row_offset=static_cast<std::uint64_t>(row)*blocks*kIq1sBlockSize;
        for(int block=0;block<blocks;++block){
            const auto*base=packed+row_offset+block*kIq1sBlockSize;
            std::uint16_t scale_bits=0;std::memcpy(&scale_bits,base,2);
            const float d=qwen_half_value(scale_bits);
            const auto*qs=base+2;
            const auto*qh_bytes=base+34;
            for(int group=0;group<8;++group){
                std::uint16_t qh=0;std::memcpy(&qh,qh_bytes+group*2,2);
                const float scale=d*static_cast<float>(2*((qh>>12)&7)+1);
                const float delta=(qh&0x8000)?-kIq1sDelta:kIq1sDelta;
                const auto*vector=input+block*256+group*32;
                float partial=0.0f;
                for(int part=0;part<4;++part){
                    const std::uint32_t index=
                        static_cast<std::uint32_t>(qs[4*group+part])|
                        ((static_cast<std::uint32_t>(qh>>(3*part))&7u)<<8);
                    const std::uint64_t entry=kIq1sGrid[index];
                    for(int lane=0;lane<8;++lane){
                        const auto weight=static_cast<std::int8_t>((entry>>(8*lane))&0xFF);
                        partial+=(static_cast<float>(weight)+delta)*vector[part*8+lane];
                    }
                }
                result+=scale*partial;
            }
        }
    }else if(type==39){
        // MXFP4: one E8M0 exponent per 32 values, then 16 packed nibbles where
        // byte j holds element j in the low half and element j+16 in the high.
        const int blocks=elements/kMxfp4BlockElements;
        const std::uint64_t row_offset=static_cast<std::uint64_t>(row)*blocks*kMxfp4BlockSize;
        for(int block=0;block<blocks;++block){
            const auto*base=packed+row_offset+block*kMxfp4BlockSize;
            const float scale=mxfp4_scale(base[0]);
            const auto*vector=input+block*kMxfp4BlockElements;
            float partial=0.0f;
            for(int lane=0;lane<16;++lane){
                const auto byte=base[1+lane];
                partial+=kMxfp4Lut[byte&15]*vector[lane]+kMxfp4Lut[byte>>4]*vector[lane+16];
            }
            result+=scale*partial;
        }
    }else if(type==40){
        const int blocks=elements/kNvfp4BlockElements;
        const std::uint64_t row_offset=static_cast<std::uint64_t>(row)*blocks*kNvfp4BlockSize;
        for(int block=0;block<blocks;++block){
            const auto*base=packed+row_offset+block*kNvfp4BlockSize;
            for(int sub=0;sub<4;++sub){
                const float scale=ue4m3_to_float(base[sub]);
                const auto*vector=input+block*kNvfp4BlockElements+sub*kNvfp4SubBlock;
                float partial=0.0f;
                // qs[lane] holds element `lane` in the low nibble, `lane+8` in the high.
                for(int lane=0;lane<8;++lane){const auto byte=base[4+sub*8+lane];partial+=kNvfp4Lut[byte&15]*vector[lane]+kNvfp4Lut[byte>>4]*vector[lane+8];}
                result+=scale*partial;
            }
        }
    }else if(type==30){
        // bf16 experts, as the Qwen3.6 NVFP4 checkpoints ship the nextn (MTP) block.
        const auto*row_data=reinterpret_cast<const std::uint16_t*>(packed)
            +row*static_cast<std::uint64_t>(elements);
        for(int index=0;index<elements;++index)
            result+=qwen_bf16_value(row_data[index])*input[index];
    }else if(type==0){
        // Unquantized weights, as the dense host feed-forward can be handed.
        // qwen_f32_dot_multi already has the vectorized single-row case.
        const auto*row_data=reinterpret_cast<const float*>(packed)
            +row*static_cast<std::uint64_t>(elements);
        if((colibri_cpu_features()&2u)!=0&&elements%32==0){
            float value=0.0f;qwen_f32_dot_multi_avx512(row_data,&input,1,elements,&value);return value;
        }
        if((colibri_cpu_features()&1u)!=0&&elements%16==0){
            float value=0.0f;qwen_f32_dot_multi_avx2(row_data,&input,1,elements,&value);return value;
        }
        for(int index=0;index<elements;++index)result+=row_data[index]*input[index];
    }else throw std::runtime_error(
        "unsupported native CPU expert quantization: "+std::to_string(type));
    return result;
}

void qwen_quant_dot_rows(
    const std::uint8_t* packed, std::uint32_t type, const float* input,
    int elements, std::uint64_t first_row, int row_count, float* outputs
) {
    if (type == 12 && elements % kBlockElements == 0
        && row_count >= 1 && row_count <= 4) {
        const auto features = colibri_cpu_features();
        if ((features & 2u) != 0) {
            qwen_quant_dot_rows_avx512(
                packed, type, input, elements, first_row, row_count, outputs);
            return;
        }
        if ((features & 1u) != 0) {
            qwen_quant_dot_rows_avx2(
                packed, type, input, elements, first_row, row_count, outputs);
            return;
        }
    }
    for (int row = 0; row < row_count; ++row)
        outputs[row] = qwen_quant_dot(
            packed, type, input, elements, first_row + row);
}

void qwen_quant_dot_pair(const std::uint8_t*packed,std::uint32_t type,const float*first,const float*second,int elements,std::uint64_t row,float&first_output,float&second_output){
    if(type==17&&(colibri_cpu_features()&2u)!=0&&elements%256==0){
        const char*setting=std::getenv("COLIBRI_IQ_AVX512");
        if(!setting||setting[0]!='0'){
            qwen_quant_dot_pair_avx512(
                packed,type,first,second,elements,row,&first_output,&second_output);
            return;
        }
    }
    if(qwen_simd_multi_type(type)&&(colibri_cpu_features()&2u)!=0&&elements%kBlockElements==0){qwen_quant_dot_pair_avx512(packed,type,first,second,elements,row,&first_output,&second_output);return;}
    first_output=qwen_quant_dot(packed,type,first,elements,row);second_output=qwen_quant_dot(packed,type,second,elements,row);
}

// Weight types the grouped GPU expert kernels can execute. The IQ codebook
// formats have no grouped kernel: they have to stay on the CPU expert path,
// which decodes every type qwen_quant_dot supports.
// Kernel-name prefix for the IQ codebook formats, which share one generated
// family of grouped expert kernels. Null for everything else.
const char* qwen_iq_kernel_prefix(std::uint32_t type) {
    // Only the formats with a device octet decoder. IQ2_XXS, IQ2_S and IQ3_S
    // pack their signs and grid indices differently and have no grouped kernel,
    // so models using them still route experts to the CPU.
    switch(type){
        case 17: return "iq2xs";
        case 18: return "iq3xxs";
        case 23: return "iq4xs";
        default: return nullptr;
    }
}

// Grouped kernel name for an IQ type, empty when the type is not one.
std::string qwen_iq_grouped_kernel(std::uint32_t type, const char* suffix) {
    const char* prefix=qwen_iq_kernel_prefix(type);
    return prefix?std::string(prefix)+suffix:std::string();
}

bool qwen_gpu_expert_type_supported(std::uint32_t type) {
    return type==8||type==12||type==13||type==14||type==40||
           qwen_iq_kernel_prefix(type)!=nullptr;
}

// Grouped SwiGLU kernel for the routed experts' gate/up type. The trailing
// k-quant case is the historical default and stays reachable only for types
// this build actually decodes, because prepare rejects anything else.
std::string qwen_grouped_swiglu_name(std::uint32_t type, bool nvfp4_tiled, bool rows) {
    const char* suffix=rows?"_grouped_swiglu_rows":"_grouped_swiglu";
    auto iq=qwen_iq_grouped_kernel(type,suffix);
    if(!iq.empty())return iq;
    if(type==40)return rows?"nvfp4_grouped_swiglu_rows"
        :(nvfp4_tiled?"nvfp4_grouped_swiglu_tiled":"nvfp4_grouped_swiglu");
    if(type==8)return rows?"q8_grouped_swiglu_rows":"q8_grouped_swiglu";
    if(type==14)return rows?"q6k_grouped_swiglu_rows":"q6k_grouped_swiglu";
    if(type==12)return rows?"q4k_grouped_swiglu_rows":"q4k_grouped_swiglu";
    return rows?"q5k_grouped_swiglu_rows":"q5k_grouped_swiglu";
}

// Row-batched grouped accumulate, which the prefill path launches by name for
// every weight type rather than through a driver entry point.
std::string qwen_grouped_accumulate_rows_name(std::uint32_t type) {
    auto iq=qwen_iq_grouped_kernel(type,"_grouped_accumulate_rows");
    if(!iq.empty())return iq;
    if(type==8)return "q8_grouped_accumulate_rows";
    if(type==40)return "nvfp4_grouped_accumulate_rows";
    if(type==12)return "q4k_grouped_accumulate_rows";
    if(type==13)return "q5k_grouped_accumulate_rows";
    return "q6k_grouped_accumulate_rows";
}

// The k-quant and NVFP4 accumulates go through their own driver entry points;
// the IQ family has no such wrapper and launches by name with the identical
// grid, one block per output row.
int qwen_launch_grouped_accumulate(
    std::uint64_t stream, std::uint32_t down_type, std::uint64_t down_table,
    std::uint64_t activated, std::uint64_t output, std::uint64_t weights,
    int intermediate, int hidden_size, int count
) {
    const auto iq=qwen_iq_grouped_kernel(down_type,"_grouped_accumulate");
    if(!iq.empty()){
        void* args[]={&down_table,&activated,&output,&weights,
                      &intermediate,&hidden_size,&count};
        return colibri_gpu_launch_named(
            iq.c_str(),static_cast<std::uint32_t>(hidden_size),1,256,0,stream,args);
    }
    switch(down_type){
        case 8:return colibri_gpu_q8_grouped_accumulate(down_table,activated,output,weights,intermediate,hidden_size,count,stream);
        case 40:return colibri_gpu_nvfp4_grouped_accumulate(down_table,activated,output,weights,intermediate,hidden_size,count,stream);
        case 12:return colibri_gpu_q4k_grouped_accumulate(down_table,activated,output,weights,intermediate,hidden_size,count,stream);
        case 13:return colibri_gpu_q5k_grouped_accumulate(down_table,activated,output,weights,intermediate,hidden_size,count,stream);
        default:return colibri_gpu_q6_grouped_accumulate(down_table,activated,output,weights,intermediate,hidden_size,count,stream);
    }
}

// True when any routed expert tensor uses an IQ codebook format.
bool qwen_model_has_iq_experts(const ColibriV2QwenRuntime& runtime) {
    for(const auto& layer:runtime.layers){
        if(layer.dense_ffn||!layer.expert_tensors[0])continue;
        for(const auto index:layer.expert_tensors)
            if(qwen_iq_kernel_prefix(runtime.model->tensors[index].type))return true;
    }
    return false;
}

// True when every routed expert tensor in the model can run on the GPU.
bool qwen_gpu_experts_executable(const ColibriV2QwenRuntime& runtime) {
    for(const auto& layer:runtime.layers){
        if(layer.dense_ffn||!layer.expert_tensors[0])continue;
        for(const auto index:layer.expert_tensors)
            if(!qwen_gpu_expert_type_supported(runtime.model->tensors[index].type))
                return false;
    }
    return true;
}

// Weight types the CPU expert path can execute, which is exactly what
// qwen_quant_dot decodes. The IQ codebook formats matter for the published
// low-bit MoE checkpoints: their routed experts are IQ2/IQ3/IQ4 even when the
// dense projections are k-quants.
bool qwen_cpu_expert_type_supported(std::uint32_t type) {
    switch(type){
        case 0: case 2: case 8: case 10: case 11: case 12: case 13: case 14:
        case 16: case 17: case 18: case 19: case 21: case 22: case 23: case 30:
        case 39: case 40:
            return true;
        default:
            return false;
    }
}

void qwen_quant_dot_two_rows(
    const std::uint8_t* first_matrix,
    const std::uint8_t* second_matrix,
    std::uint32_t type,
    const float* input,
    int elements,
    std::uint64_t row,
    float& first_output,
    float& second_output
) {
    std::uint64_t row_bytes = 0;
    if (type == 12) {
        row_bytes = static_cast<std::uint64_t>(elements / kBlockElements) * kQ4KBlockSize;
    } else if (type == 13) {
        row_bytes = static_cast<std::uint64_t>(elements / kBlockElements) * kQ5KBlockSize;
    } else if (type == 14) {
        row_bytes = static_cast<std::uint64_t>(elements / kBlockElements) * kQ6KBlockSize;
    } else if (type == 17) {
        row_bytes = static_cast<std::uint64_t>(elements / 256) * kIq2xsBlockBytes;
    } else if (type == 8) {
        row_bytes = static_cast<std::uint64_t>(elements / 32) * kQ8BlockSize;
    } else if (type == 40) {
        row_bytes = static_cast<std::uint64_t>(elements / kNvfp4BlockElements) * kNvfp4BlockSize;
    } else if (type == 30) {
        row_bytes = static_cast<std::uint64_t>(elements) * 2;
    }
    const auto* first_row = first_matrix + row * row_bytes;
    const auto* second_row = second_matrix + row * row_bytes;
    if ((colibri_cpu_features() & 2u) != 0) {
        const char* iq_setting = std::getenv("COLIBRI_IQ_AVX512");
        const bool iq_avx512 = !iq_setting || iq_setting[0] != '0';
        const bool supported =
            (type == 8 && elements % 32 == 0)
            || (type == 17 && iq_avx512 && elements % 256 == 0)
            || ((type == 12 || type == 13 || type == 14) && elements % 256 == 0);
        if (supported) {
            qwen_quant_dot_two_rows_avx512(
                first_row, second_row, type, input, elements,
                &first_output, &second_output
            );
            return;
        }
    }
    first_output = qwen_quant_dot(first_matrix, type, input, elements, row);
    second_output = qwen_quant_dot(second_matrix, type, input, elements, row);
}

#if defined(_OPENMP)
int qwen_cpu_thread_count(const ColibriV2QwenRuntime& runtime) {
    if (runtime.options.cpu_threads)
        return std::min<int>(runtime.options.cpu_threads, omp_get_num_procs());
    int team = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        const int physical = omp_get_num_procs() / 2;
        if (physical >= 1 && team > physical) team = physical;
    }
    return team;
}
#endif

void gemma_cpu_moe(const ColibriV2QwenRuntime&runtime,const QwenLayerPlan&layer,
        const std::int32_t*selected,const float*weights,int routed_count,const float*input,
        float*activated,float*output){
    const int experts=runtime.model->config.expert_count;
    const int hidden=runtime.model->config.hidden_size,intermediate=runtime.moe_intermediate;
    const auto&gate_up_tensor=runtime.model->tensors[layer.expert_tensors[0]];
    const auto&down_tensor=runtime.model->tensors[layer.expert_tensors[1]];
    const auto&scale_tensor=runtime.model->tensors[layer.expert_tensors[2]];
    if(gate_up_tensor.type!=2||down_tensor.type!=2||scale_tensor.type!=0)
        throw std::runtime_error("native Gemma 4 expects Q4_0 experts and f32 expert scales");
    const auto gate_up_bytes=gate_up_tensor.size/experts,down_bytes=down_tensor.size/experts;
    const auto*expert_scales=reinterpret_cast<const float*>(tensor_data(*runtime.model,scale_tensor));
    #pragma omp parallel for schedule(dynamic,4) num_threads(qwen_cpu_thread_count(runtime))
    for(int task=0;task<routed_count*intermediate;++task){
        const int rank=task/intermediate,row=task%intermediate,expert=selected[rank];
        if(expert<0||expert>=experts)continue;
        const auto*gate_up=tensor_data(*runtime.model,gate_up_tensor)+static_cast<std::uint64_t>(expert)*gate_up_bytes;
        const float gate=qwen_quant_dot(gate_up,2,input,hidden,row);
        const float up=qwen_quant_dot(gate_up,2,input,hidden,row+intermediate);
        const float cubic=gate*gate*gate;
        const float gelu=0.5f*gate*(1.0f+std::tanh(0.7978845608028654f*(gate+0.044715f*cubic)));
        activated[task]=gelu*up;
    }
    #pragma omp parallel for schedule(dynamic,4) num_threads(qwen_cpu_thread_count(runtime))
    for(int row=0;row<hidden;++row){
        float sum=0.0f;
        for(int rank=0;rank<routed_count;++rank){const int expert=selected[rank];if(expert<0||expert>=experts)continue;const auto*down=tensor_data(*runtime.model,down_tensor)+static_cast<std::uint64_t>(expert)*down_bytes;sum+=weights[rank]*expert_scales[expert]*qwen_quant_dot(down,2,activated+rank*intermediate,intermediate,row);}
        output[row]=sum;
    }
}

// Host-side dense SwiGLU for a block whose feed-forward weights stayed in the
// mapping instead of the GPU arena. Reads the quantized bytes directly, so it
// costs no VRAM at all -- this is what lets a dense model exceed the card.
void qwen_cpu_dense_ffn(
    ColibriV2QwenRuntime& runtime, const QwenLayerPlan& layer,
    const float* input, float* output
) {
    const int hidden=runtime.model->config.hidden_size;
    const int intermediate=static_cast<int>(runtime.moe_intermediate);
    const std::size_t ffn_base=layer.attention?7:10;
    const auto&gate_tensor=runtime.model->tensors[layer.static_tensors[ffn_base+1]];
    const auto&up_tensor=runtime.model->tensors[layer.static_tensors[ffn_base+2]];
    const auto&down_tensor=runtime.model->tensors[layer.static_tensors[ffn_base+3]];
    const auto*gate_data=tensor_data(*runtime.model,gate_tensor);
    const auto*up_data=tensor_data(*runtime.model,up_tensor);
    const auto*down_data=tensor_data(*runtime.model,down_tensor);
    // The pinned buffers the caller hands over are DMA staging, and every
    // output row re-reads the whole activation vector -- roughly 700 MiB of
    // re-reads per block. Doing that against page-locked memory costs ~5x, so
    // the vectors are mirrored into ordinary cacheable scratch first.
    runtime.dense_scratch.resize(static_cast<std::size_t>(hidden)+intermediate);
    float*local_input=runtime.dense_scratch.data();
    float*activated=local_input+hidden;
    std::memcpy(local_input,input,static_cast<std::size_t>(hidden)*sizeof(float));
    #pragma omp parallel for schedule(static) num_threads(qwen_cpu_thread_count(runtime))
    for(int row=0;row<intermediate;++row){
        const float gate=qwen_quant_dot(gate_data,gate_tensor.type,local_input,hidden,row);
        const float up=qwen_quant_dot(up_data,up_tensor.type,local_input,hidden,row);
        activated[row]=gate/(1.0f+std::exp(-std::min(80.0f,std::max(-80.0f,gate))))*up;
    }
    #pragma omp parallel for schedule(static) num_threads(qwen_cpu_thread_count(runtime))
    for(int row=0;row<hidden;++row)
        output[row]=qwen_quant_dot(down_data,down_tensor.type,activated,intermediate,row);
}

// Dequantize one weight row to f32 so it can be reused across every token
// routed to the same expert within a batch: the quantized bytes are decoded
// once per batch instead of once per token.
void qwen_dequant_row(const std::uint8_t*packed,std::uint32_t type,int elements,std::uint64_t row,float*output){
    if(qwen_simd_quant_type(type)&&(colibri_cpu_features()&2u)!=0&&elements%kBlockElements==0){qwen_dequant_row_avx512(packed,type,elements,row,output);return;}
    if(type==40&&(colibri_cpu_features()&1u)!=0&&elements%kNvfp4BlockElements==0){qwen_dequant_row_avx2(packed,type,elements,row,output);return;}
    if(qwen_simd_quant_type(type)&&(colibri_cpu_features()&1u)!=0&&elements%kBlockElements==0){qwen_dequant_row_avx2(packed,type,elements,row,output);return;}
    const auto base=row*static_cast<std::uint64_t>(elements);
    if(type==13)for(int index=0;index<elements;++index)output[index]=qwen_q5_value(packed,base+index);
    else if(type==14)for(int index=0;index<elements;++index)output[index]=qwen_q6_value(packed,base+index);
    else if(type==8)for(int index=0;index<elements;++index)output[index]=qwen_q8_value(packed,base+index);
    else if(type==12)for(int index=0;index<elements;++index)output[index]=qwen_q4k_value(packed,base+index);
    else if(type==10)for(int index=0;index<elements;++index)output[index]=qwen_q2k_value(packed,base+index);
    else if(type==11)for(int index=0;index<elements;++index)output[index]=qwen_q3k_value(packed,base+index);
    else if(type==16)for(int index=0;index<elements;++index)output[index]=qwen_iq2xxs_value(packed,base+index);
    else if(type==18)for(int index=0;index<elements;++index)output[index]=qwen_iq3xxs_value(packed,base+index);
    else if(type==22)for(int index=0;index<elements;++index)output[index]=qwen_iq2s_value(packed,base+index);
    else if(type==21)for(int index=0;index<elements;++index)output[index]=qwen_iq3s_value(packed,base+index);
    else if(type==17)for(int index=0;index<elements;++index)output[index]=qwen_iq2xs_value(packed,base+index);
    else if(type==23)for(int index=0;index<elements;++index)output[index]=qwen_iq4xs_value(packed,base+index);
    else if(type==40)for(int index=0;index<elements;++index)output[index]=qwen_nvfp4_value(packed,base+index);
    else if(type==30){const auto*row_data=reinterpret_cast<const std::uint16_t*>(packed)+base;for(int index=0;index<elements;++index)output[index]=qwen_bf16_value(row_data[index]);}
    else if(type==0){const auto*row_data=reinterpret_cast<const float*>(packed)+base;for(int index=0;index<elements;++index)output[index]=row_data[index];}
    else throw std::runtime_error("unsupported native CPU expert quantization");
}

void qwen_f32_dot_multi(const float*row,const float*const*inputs,int count,int elements,float*outputs){
    if((colibri_cpu_features()&2u)!=0&&elements%32==0){qwen_f32_dot_multi_avx512(row,inputs,count,elements,outputs);return;}
    if((colibri_cpu_features()&1u)!=0&&elements%16==0){qwen_f32_dot_multi_avx2(row,inputs,count,elements,outputs);return;}
    for(int token=0;token<count;++token){
        const float*vector=inputs[token];
        float sum=0.0f;
        for(int index=0;index<elements;++index)sum+=row[index]*vector[index];
        outputs[token]=sum;
    }
}

// Register-blocked expert GEMM over a block of <=4 weight rows: out[i*count+j].
void qwen_f32_gemm_rows(const float*weights,int mr,const float*const*inputs,int count,int elements,float*out){
    if((colibri_cpu_features()&2u)!=0&&elements%32==0){qwen_f32_gemm_rows_avx512(weights,mr,inputs,count,elements,out);return;}
    if((colibri_cpu_features()&1u)!=0&&elements%16==0){qwen_f32_gemm_rows_avx2(weights,mr,inputs,count,elements,out);return;}
    for(int i=0;i<mr;++i){
        const float*row=weights+static_cast<std::size_t>(i)*elements;float*o=out+static_cast<std::size_t>(i)*count;
        for(int j=0;j<count;++j){const float*v=inputs[j];float sum=0.0f;for(int k=0;k<elements;++k)sum+=row[k]*v[k];o[j]=sum;}
    }
}

void qwen_quant_dot_quad(
    const std::uint8_t*packed,std::uint32_t type,const float*const inputs[4],
    int elements,std::uint64_t row,float outputs[4]
){
    if(qwen_simd_multi_type(type)&&(colibri_cpu_features()&2u)!=0&&elements%256==0){
        qwen_quant_dot_quad_avx512(packed,type,inputs,elements,row,outputs);return;
    }
    if(type==40&&(colibri_cpu_features()&1u)!=0&&elements%64==0){
        qwen_quant_dot_quad_avx2(packed,type,inputs,elements,row,outputs);return;
    }
    if(qwen_simd_multi_type(type)&&(colibri_cpu_features()&1u)!=0&&elements%256==0){
        qwen_quant_dot_quad_avx2(packed,type,inputs,elements,row,outputs);return;
    }
    if((colibri_cpu_features()&1u)!=0&&
       qwen_quant_dot_iq_multi_avx2(packed,type,inputs,4,elements,row,outputs))return;
    qwen_quant_dot_pair(packed,type,inputs[0],inputs[1],elements,row,outputs[0],outputs[1]);
    qwen_quant_dot_pair(packed,type,inputs[2],inputs[3],elements,row,outputs[2],outputs[3]);
}

void qwen_quant_dot_oct(
    const std::uint8_t*packed,std::uint32_t type,const float*const inputs[8],
    int elements,std::uint64_t row,float outputs[8]
){
    if(!qwen_simd_multi_type(type)){
        // The IQ formats have an eight-token AVX2 kernel; taking it here rather
        // than as two quads halves the codebook decodes.
        if((colibri_cpu_features()&1u)!=0&&
           qwen_quant_dot_iq_multi_avx2(packed,type,inputs,8,elements,row,outputs))return;
        // NVFP4 uses two register-blocked AVX2 quads; the formats without an
        // AVX-512 oct kernel fall back through quad to the ordinary pair path,
        // because the oct entry point would otherwise decode them as Q8_0.
        qwen_quant_dot_quad(packed,type,inputs,elements,row,outputs);
        qwen_quant_dot_quad(packed,type,inputs+4,elements,row,outputs+4);
        return;
    }
    qwen_quant_dot_oct_avx512(packed,type,inputs,elements,row,outputs);
}

// The embedding table is indexed, not multiplied, so nothing downstream can
// detect a wrong decode -- picking the kernel by tensor type is the only guard.
// Throwing beats defaulting to Q8_0: an unsupported table silently produces
// ~100x-magnitude noise that drowns the residual stream instead of failing.
// Picking the LM-head kernel by tensor type, and refusing anything without a
// kernel, rather than defaulting to Q8_0: a mismatched head decodes to noise
// and pins the argmax to one token, which looks like a model bug, not a
// dispatch bug.
// One dispatch point from tensor type to the device matvec kernels, so the
// batched prefill path can fall back on them for any quantization that has no
// batched kernel of its own. Returns nonzero when the type has no kernel.
int qwen_gpu_matvec_by_type(
    std::uint32_t type, std::uint64_t matrix, std::uint64_t input,
    std::uint64_t output, int input_size, int output_size, std::uint64_t stream
) {
    switch (type) {
        case 8: return colibri_gpu_q8_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 10: return colibri_gpu_q2k_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 11: return colibri_gpu_q3k_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 12: return colibri_gpu_q4k_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 13: return colibri_gpu_q5k_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 14: return colibri_gpu_q6k_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 16: return colibri_gpu_iq2xxs_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 18: return colibri_gpu_iq3xxs_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 21: return colibri_gpu_iq3s_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 17: return colibri_gpu_iq2xs_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 22: return colibri_gpu_iq2s_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        case 23: return colibri_gpu_iq4xs_matvec_transposed(matrix, input, output, input_size, output_size, stream);
        default: break;
    }
    return -1;
}

// Q8-activation group-decode LM head, or null where the type has no such
// kernel and the per-element fused head has to stand in.
const char* qwen_q8_lm_head_kernel(std::uint32_t type) {
    switch (type) {
        case 10: return "q2k_q8_lm_head_argmax_warp";
        case 11: return "q3k_q8_lm_head_argmax_warp";
        case 12: return "q4k_q8_lm_head_argmax_warp";
        case 13: return "q5k_q8_lm_head_argmax_warp";
        case 14: return "q6k_q8_lm_head_argmax_warp";
        case 16: return "iq2xxs_q8_lm_head_argmax_warp";
        case 18: return "iq3xxs_q8_lm_head_argmax_warp";
        default: return nullptr;
    }
}

const char* qwen_lm_head_argmax_kernel(std::uint32_t type) {
    switch (type) {
        case 0: return "f32_lm_head_argmax_warp";
        case 8: return "q8_lm_head_argmax_warp";
        case 10: return "q2k_lm_head_argmax_warp";
        case 11: return "q3k_lm_head_argmax_warp";
        case 12: return "q4k_lm_head_argmax_warp";
        case 13: return "q5k_lm_head_argmax_warp";
        case 14: return "q6k_lm_head_argmax_warp";
        case 16: return "iq2xxs_lm_head_argmax_warp";
        case 18: return "iq3xxs_lm_head_argmax_warp";
        case 22: return "iq2s_lm_head_argmax_warp";
        case 21: return "iq3s_lm_head_argmax_warp";
        case 17: return "iq2xs_lm_head_argmax_warp";
        case 23: return "iq4xs_lm_head_argmax_warp";
        case 30: return "bf16_lm_head_argmax_warp";
        default: break;
    }
    throw std::runtime_error(
        "native Qwen LM-head type is unsupported: " + std::to_string(type));
}

// Effective device type for a tensor: what actually got uploaded, which is the
// checkpoint type unless prepare requantized it.
std::uint32_t qwen_device_type(
    const ColibriV2QwenRuntime& runtime, std::uint64_t index
) {
    if (index < runtime.device_tensor_types.size())
        return runtime.device_tensor_types[index];
    return runtime.model->tensors[index].type;
}

// Bytes the tensor occupies in the static arena under its effective type.
std::uint64_t qwen_device_tensor_size(
    const ColibriV2QwenRuntime& runtime, std::uint64_t index
) {
    const auto& tensor = runtime.model->tensors[index];
    if (qwen_device_type(runtime, index) == 8 && tensor.type != 8) {
        std::uint64_t elements = 1;
        for (auto dimension : tensor.shape) elements *= dimension;
        return (elements / 32) * kQ8BlockSize;
    }
    return tensor.size;
}

// Host pointer a static tensor can be used from directly, or 0 if it has to be
// copied into the device arena.
//
// On the CPU backend "device memory" is ordinary host memory, and the weights
// are already resident in the GGUF mapping -- so uploading them allocates and
// fills a second full copy of the model. For a large checkpoint that is the
// difference between running and exhausting RAM, and it buys nothing: the
// mapping is MAP_PRIVATE and no kernel writes to weights.
//
// Aliasing is refused whenever the device representation is not byte-identical
// to the file (the bf16 -> q8_0 requantization path genuinely transforms the
// data) or the mapped address is not aligned enough for the 128-bit vector
// loads the kernels use.
const std::uint8_t* qwen_alias_static_tensor(
    const ColibriV2QwenRuntime& runtime, std::uint64_t index
) {
    if (!colibri_backend_is_cpu()) return nullptr;
    // Escape hatch: forces the copying path so a suspected aliasing problem can
    // be confirmed or ruled out without a rebuild.
    static const bool disabled = [] {
        const char* setting = std::getenv("COLIBRI_CPU_NO_ALIAS");
        return setting != nullptr && setting[0] == '1';
    }();
    if (disabled) return nullptr;
    const auto& tensor = runtime.model->tensors[index];
    if (qwen_device_type(runtime, index) != tensor.type) return nullptr;
    if (qwen_device_tensor_size(runtime, index) != tensor.size) return nullptr;
    const std::uint8_t* data = tensor_data(*runtime.model, tensor);
    if (data == nullptr) return nullptr;
    constexpr std::uintptr_t kVectorAlignment = 16;
    if (reinterpret_cast<std::uintptr_t>(data) % kVectorAlignment != 0)
        return nullptr;
    return data;
}

const char* qwen_embedding_kernel(std::uint32_t type, bool rows) {
    switch (type) {
        case 0: return rows ? "qwen_f32_embedding_rows" : "qwen_f32_embedding";
        case 8: return rows ? "qwen_q8_embedding_rows" : "qwen_q8_embedding";
        case 30: return rows ? "qwen_bf16_embedding_rows" : "qwen_bf16_embedding";
        case 10: return rows ? "qwen_q2k_embedding_rows" : "qwen_q2k_embedding";
        case 16: return rows ? "qwen_iq2xxs_embedding_rows" : "qwen_iq2xxs_embedding";
        case 18: return rows ? "qwen_iq3xxs_embedding_rows" : "qwen_iq3xxs_embedding";
        case 22: return rows ? "qwen_iq2s_embedding_rows" : "qwen_iq2s_embedding";
        case 21: return rows ? "qwen_iq3s_embedding_rows" : "qwen_iq3s_embedding";
        case 17: return rows ? "qwen_iq2xs_embedding_rows" : "qwen_iq2xs_embedding";
        case 23: return rows ? "qwen_iq4xs_embedding_rows" : "qwen_iq4xs_embedding";
        case 11: return rows ? "qwen_q3k_embedding_rows" : "qwen_q3k_embedding";
        case 12: return rows ? "qwen_q4k_embedding_rows" : "qwen_q4k_embedding";
        case 13: return rows ? "qwen_q5k_embedding_rows" : "qwen_q5k_embedding";
        case 14: return rows ? "qwen_q6k_embedding_rows" : "qwen_q6k_embedding";
        default: break;
    }
    throw std::runtime_error(
        "native Qwen embedding table type is unsupported: " + std::to_string(type));
}

// Stage `count` embedding rows from the host mapping into device scratch and
// return the base pointer for the embedding kernels. Those kernels address a
// row as token*hidden, so the staged copy is indexed 0..count-1 rather than by
// the real token id -- callers pass 0 (single token) or embedding_row_index
// (rows variant) instead of the token itself.
//
// Returns the resident table unchanged when the embeddings were kept in VRAM,
// so every call site is identical in both modes.
std::uint64_t qwen_stage_embedding_rows(
    ColibriV2QwenRuntime& runtime, const std::uint32_t* tokens, int count
) {
    if (!runtime.embeddings_host_resident)
        return runtime.device_tensors[runtime.token_embeddings];
    const auto& table = runtime.model->tensors[runtime.token_embeddings];
    const auto row_bytes = runtime.embedding_row_bytes;
    const auto vocabulary = table.size / row_bytes;
    const auto* source = tensor_data(*runtime.model,table);
    // Single-token decode is the hot case. When the mapping is registered the
    // row can be DMA'd straight out of it, which drops both the host memcpy and
    // the pinned-buffer reuse hazard from the per-token path. A rows chunk
    // gathers scattered rows, so it stays on the pack-then-one-upload path
    // rather than issuing one small DMA per row.
    if (count == 1 && runtime.dma_paging) {
        if (tokens[0] >= vocabulary)
            throw std::runtime_error(
                "native Qwen embedding token is out of range: "
                + std::to_string(tokens[0]));
        if (colibri_gpu_upload(
                runtime.embedding_stage,
                source + static_cast<std::uint64_t>(tokens[0]) * row_bytes,
                row_bytes, runtime.stream) != 0)
            throw std::runtime_error("native Qwen embedding row staging failed");
        return runtime.embedding_stage;
    }
    // The pinned mirror is refilled every token, so the previous upload has to
    // have drained first. The event is recorded immediately after each upload,
    // so by the time the next token reaches here the wait is already satisfied.
    if (runtime.embedding_event) colibri_gpu_event_sync(runtime.embedding_event);
    auto* host = static_cast<std::uint8_t*>(runtime.embedding_host);
    for (int row = 0; row < count; ++row) {
        if (tokens[row] >= vocabulary)
            throw std::runtime_error(
                "native Qwen embedding token is out of range: "
                + std::to_string(tokens[row]));
        std::memcpy(
            host + static_cast<std::uint64_t>(row) * row_bytes,
            source + static_cast<std::uint64_t>(tokens[row]) * row_bytes,
            row_bytes);
    }
    if (colibri_gpu_upload(
            runtime.embedding_stage, host,
            static_cast<std::uint64_t>(count) * row_bytes, runtime.stream) != 0)
        throw std::runtime_error("native Qwen embedding row staging failed");
    if (runtime.embedding_event)
        colibri_gpu_event_record(runtime.embedding_event, runtime.stream);
    return runtime.embedding_stage;
}

// NVFP4 checkpoints carry an optional per-expert f32 scale tensor alongside each
// expert role (ffn_{gate,up,down}_exps.scale). Returns 1.0 when the model has none.
float qwen_expert_role_scale(
    const ColibriV2QwenRuntime& runtime, std::uint64_t scale_tensor, int expert
) {
    if (scale_tensor == std::numeric_limits<std::uint64_t>::max()) return 1.0f;
    const auto& st = runtime.model->tensors[scale_tensor];
    const auto scale_bytes = st.size / runtime.model->config.expert_count;
    float value = 1.0f;
    std::memcpy(
        &value,
        tensor_data(*runtime.model,st)
            + static_cast<std::uint64_t>(expert) * scale_bytes,
        sizeof(float)
    );
    return value;
}

// Per-phase MoE timing. The expert path does not go through launch_named, so
// COLIBRI_CPU_PROFILE cannot see inside it; this is the only view of where the
// ~43% of decode that lands here actually goes. Off unless COLIBRI_MOE_PROFILE=1.
std::atomic<std::uint64_t> g_moe_gate_up_ns{0}, g_moe_quant_ns{0}, g_moe_down_ns{0};
std::atomic<std::uint64_t> g_moe_calls{0};

bool moe_profiling() {
    static const bool on = [] {
        const char* s = std::getenv("COLIBRI_MOE_PROFILE");
        return s && s[0] == '1';
    }();
    return on;
}

extern "C" COLIBRI_BACKEND_API void colibri_moe_profile_dump() {
    if (!moe_profiling()) return;
    const double gate_up = g_moe_gate_up_ns.load() / 1e6;
    const double quant = g_moe_quant_ns.load() / 1e6;
    const double down = g_moe_down_ns.load() / 1e6;
    const double total = gate_up + quant + down;
    if (total <= 0.0) return;
    std::fprintf(stderr,
        "\n[colibri-moe] %llu calls, %.1f ms total\n"
        "  gate+up+swiglu %8.1f ms  %5.1f%%\n"
        "  activation q8  %8.1f ms  %5.1f%%\n"
        "  down           %8.1f ms  %5.1f%%\n",
        static_cast<unsigned long long>(g_moe_calls.load()), total,
        gate_up, 100*gate_up/total, quant, 100*quant/total,
        down, 100*down/total);
}

void qwen_cpu_moe(
    const ColibriV2QwenRuntime& runtime,
    const QwenLayerPlan& layer,
    const std::int32_t* selected,
    const float* weights,
    int routed_count,
    const float* input,
    float* activated,
    float* output
) {
    const int experts = runtime.model->config.expert_count;
    const int hidden = runtime.model->config.hidden_size;
    const int intermediate = runtime.moe_intermediate;
    if (routed_count < 0 || routed_count > 256) {
        throw std::runtime_error("native CPU MoE routed count is unsupported");
    }
    for (int role = 0; role < 3; ++role) {
        const auto type = runtime.model->tensors[layer.expert_tensors[role]].type;
        if (!qwen_cpu_expert_type_supported(type)) {
            throw std::runtime_error(
                "unsupported native CPU expert quantization: " + std::to_string(type));
        }
    }
    std::array<const std::uint8_t*, 256> gate{}, up{}, down{};
    // The GPU path folds the per-expert gate/up scales into the staged block
    // scales; here the weights are read straight from the mmap, so apply them to
    // the dot products instead. That is exact (the scale is a plain per-expert
    // factor) and has to happen before the SwiGLU, which is non-linear. The down
    // scale is already folded into `weights` by the caller.
    std::array<float, 256> gate_scale{}, up_scale{};
    for (int rank = 0; rank < routed_count; ++rank) {
        if (selected[rank] < 0 || selected[rank] >= experts) {
            throw std::runtime_error("native CPU MoE selected an invalid expert");
        }
        for (int role = 0; role < 3; ++role) {
            const auto& t = runtime.model->tensors[layer.expert_tensors[role]];
            const auto bytes = t.size / experts;
            const auto* pointer = tensor_data(*runtime.model,t)
                + static_cast<std::uint64_t>(selected[rank]) * bytes;
            if (role == 0) gate[rank] = pointer;
            else if (role == 1) up[rank] = pointer;
            else down[rank] = pointer;
        }
        gate_scale[rank] =
            qwen_expert_role_scale(runtime, layer.expert_gate_scale, selected[rank]);
        up_scale[rank] =
            qwen_expert_role_scale(runtime, layer.expert_up_scale, selected[rank]);
    }
    const auto gate_type = runtime.model->tensors[layer.expert_tensors[0]].type;
    const auto up_type = runtime.model->tensors[layer.expert_tensors[1]].type;
    const auto down_type = runtime.model->tensors[layer.expert_tensors[2]].type;
    const char* q8_setting = std::getenv("COLIBRI_Q8_ACTIVATIONS");
    const auto cpu_features = colibri_cpu_features();
    // Q8-K activations only pay off, and are only decoded, for these quants.
    // Q4_K, NVFP4 and bf16 are faster on the vectorized f32 path (their small
    // codes do not amortize the activation quantization). IQ2_XS and IQ3_XXS
    // use the same integer codebook matvec strategy as llama.cpp.
    const auto q8_activations_ok = [](std::uint32_t type) {
        return type == 8 || type == 13 || type == 14 || type == 17 || type == 18;
    };
    const bool laguna_iq_q8 = (cpu_features & 4u) != 0
        && gate_type == 17 && up_type == 17 && down_type == 18
        && (!q8_setting || q8_setting[0] != '0');
    // On the CPU backend, default the Q8-K activation path on for any expert
    // quantization q8_activations_ok() accepts. It was previously reachable for
    // Q8_0 only through COLIBRI_Q8_ACTIVATIONS=1, even though the type is on
    // that list -- measured worth ~14% on a 35B Q8_0 MoE (medians 7.53/7.16
    // without against 8.33/8.42 with, two replications, non-overlapping in the
    // second). Scoped to the CPU backend because that is the configuration the
    // measurement covers; a GPU machine running hybrid or cpu expert placement
    // keeps the old behaviour until someone measures it there.
    const bool q8_default_on = colibri_backend_is_cpu()
        && !(q8_setting && q8_setting[0] == '0');
    const bool use_q8 = (cpu_features & 1u) != 0 && hidden % 256 == 0
        && intermediate % 256 == 0
        && ((q8_setting && q8_setting[0] == '1') || laguna_iq_q8 || q8_default_on)
        && q8_activations_ok(gate_type) && q8_activations_ok(up_type)
        && q8_activations_ok(down_type);
    static constexpr char q4_tile_name[] = {
        'C','O','L','I','B','R','I','_','Q','4','_','R','O','W','_','T','I','L','E','S','\0'
    };
    const char* q4_tile_setting = std::getenv(q4_tile_name);
    const bool q4_tiled = !use_q8 && (cpu_features & 3u) != 0
        && hidden % 256 == 0 && intermediate % 256 == 0
        && gate_type == 12 && up_type == 12 && down_type == 12
        && (!q4_tile_setting || q4_tile_setting[0] != '0');
    thread_local std::vector<QwenQ8KBlock> input_q8, activated_q8;
    if (use_q8) {
        input_q8.resize(hidden / 256);
        activated_q8.resize(static_cast<std::size_t>(routed_count) * (intermediate / 256));
        qwen_quantize_q8_k_avx2(input, hidden, input_q8.data());
    }
#if defined(_OPENMP)
    // Decode is bandwidth-bound on expert weights. SMT siblings contend for the
    // same load ports and memory bandwidth, so use physical cores by default.
    // Keep an explicit OMP_NUM_THREADS override for machine-specific tuning.
    const int team = qwen_cpu_thread_count(runtime);
#endif
    const auto* input_q8_data = input_q8.data();
    auto* activated_q8_data = activated_q8.data();
    const auto q8_dot = [cpu_features](
        const std::uint8_t* packed, std::uint32_t type,
        const QwenQ8KBlock* input, int elements, std::uint64_t row
    ) {
        return (cpu_features & 4u) != 0
            ? qwen_quant_dot_q8_k_avx_vnni(packed,type,input,elements,row)
            : qwen_quant_dot_q8_k_avx2(packed,type,input,elements,row);
    };
    const char* fused_gate_up_setting = std::getenv("COLIBRI_FUSED_MOE_GATE_UP");
    const char* iq_avx512_setting = std::getenv("COLIBRI_IQ_AVX512");
    const bool auto_fused_iq2xs = gate_type == 17
        && (colibri_cpu_features() & 2u) != 0
        && (!iq_avx512_setting || iq_avx512_setting[0] != '0')
        && (!fused_gate_up_setting || fused_gate_up_setting[0] != '0');
    const auto gate_up_task = [&](int task) {
        const int rank = task / intermediate;
        const int row = task % intermediate;
        float gate_value = 0.0f;
        float up_value = 0.0f;
        if (use_q8) {
            gate_value = q8_dot(
                gate[rank], gate_type, input_q8_data, hidden, row
            );
            up_value = q8_dot(
                up[rank], up_type, input_q8_data, hidden, row
            );
        } else if ((runtime.fused_moe_gate_up || auto_fused_iq2xs)
                   && gate_type == up_type) {
            qwen_quant_dot_two_rows(
                gate[rank], up[rank], gate_type, input, hidden,
                row, gate_value, up_value
            );
        } else {
            gate_value = qwen_quant_dot(gate[rank], gate_type, input, hidden, row);
            up_value = qwen_quant_dot(up[rank], up_type, input, hidden, row);
        }
        gate_value *= gate_scale[rank];
        up_value *= up_scale[rank];
        const float clipped = std::max(-80.0f, std::min(80.0f, gate_value));
        activated[task] = gate_value / (1.0f + std::exp(-clipped)) * up_value;
    };
    const auto down_row = [&](int row) {
        float value = 0.0f;
        for (int rank = 0; rank < routed_count; ++rank) {
            const float expert_value = use_q8
                ? q8_dot(
                    down[rank], down_type,
                    activated_q8_data + static_cast<std::size_t>(rank) * (intermediate / 256),
                    intermediate, row
                )
                : qwen_quant_dot(down[rank], down_type, activated + rank * intermediate, intermediate, row);
            value += weights[rank] * expert_value;
        }
        output[row] = value;
    };
    const auto gate_up_tile = [&](int task) {
        constexpr int tile_rows = 4;
        const int tiles_per_expert = (intermediate + tile_rows - 1) / tile_rows;
        const int rank = task / tiles_per_expert;
        const int first_row = (task % tiles_per_expert) * tile_rows;
        const int count = std::min(tile_rows, intermediate - first_row);
        float gate_values[tile_rows]{}, up_values[tile_rows]{};
        qwen_quant_dot_rows(
            gate[rank], gate_type, input, hidden,
            first_row, count, gate_values);
        qwen_quant_dot_rows(
            up[rank], up_type, input, hidden,
            first_row, count, up_values);
        for (int lane = 0; lane < count; ++lane) {
            const float gate_value = gate_values[lane] * gate_scale[rank];
            const float up_value = up_values[lane] * up_scale[rank];
            const float clipped = std::max(-80.0f, std::min(80.0f, gate_value));
            activated[rank * intermediate + first_row + lane] =
                gate_value / (1.0f + std::exp(-clipped)) * up_value;
        }
    };
    const auto down_tile = [&](int first_row) {
        constexpr int tile_rows = 4;
        const int count = std::min(tile_rows, hidden - first_row);
        float sums[tile_rows]{};
        for (int rank = 0; rank < routed_count; ++rank) {
            float values[tile_rows]{};
            qwen_quant_dot_rows(
                down[rank], down_type, activated + rank * intermediate,
                intermediate, first_row, count, values);
            for (int lane = 0; lane < count; ++lane)
                sums[lane] += weights[rank] * values[lane];
        }
        for (int lane = 0; lane < count; ++lane)
            output[first_row + lane] = sums[lane];
    };
    if (use_q8) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(team)
#endif
        {
            // `omp for` carries an implicit barrier, so a timestamp taken by one
            // thread between them is a true phase boundary rather than a sample
            // of whichever thread got there first.
            const bool profile = moe_profiling();
            auto mark = std::chrono::steady_clock::time_point{};
            if (profile) mark = std::chrono::steady_clock::now();
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int task = 0; task < routed_count * intermediate; ++task)
                gate_up_task(task);
#if defined(_OPENMP)
#pragma omp master
#endif
            if (profile) {
                const auto now = std::chrono::steady_clock::now();
                g_moe_gate_up_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now-mark).count());
                mark = now;
            }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int rank = 0; rank < routed_count; ++rank) {
                qwen_quantize_q8_k_avx2(
                    activated + static_cast<std::size_t>(rank) * intermediate,
                    intermediate,
                    activated_q8_data + static_cast<std::size_t>(rank) * (intermediate / 256)
                );
            }
#if defined(_OPENMP)
#pragma omp master
#endif
            if (profile) {
                const auto now = std::chrono::steady_clock::now();
                g_moe_quant_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now-mark).count());
                mark = now;
            }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int row = 0; row < hidden; ++row) down_row(row);
#if defined(_OPENMP)
#pragma omp master
#endif
            if (profile) {
                g_moe_down_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-mark).count());
                ++g_moe_calls;
            }
        }
    } else {
        // One team covers both dependent phases. The implicit barrier after the
        // first loop replaces a second parallel-region launch on every layer.
#if defined(_OPENMP)
#pragma omp parallel num_threads(team)
#endif
        {
            if (q4_tiled) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int task = 0;
                     task < routed_count * ((intermediate + 3) / 4); ++task)
                    gate_up_tile(task);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int row = 0; row < hidden; row += 4) down_tile(row);
            } else {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int task = 0; task < routed_count * intermediate; ++task)
                    gate_up_task(task);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int row = 0; row < hidden; ++row) down_row(row);
            }
        }
    }
    if (std::getenv("COLIBRI_MOE_DEBUG")) {
        static int calls = 0;
        int n = calls++;
        if (n < 45) {
            double s = 0; for (int i = 0; i < hidden; ++i) s += std::abs(output[i]);
            std::fprintf(stderr, "[moe-dbg] call=%02d sel0=%d w0=%.4f sum|out|=%.4f\n",
                n, selected[0], weights[0], s);
        }
    }
}

bool qwen_prefill_direct_quant_enabled(
    const ColibriV2QwenRuntime& runtime
) {
    if ((colibri_cpu_features() & 3u) == 0) return false;
    if (const char* setting = std::getenv("COLIBRI_PREFILL_DIRECT_QUANT"))
        return setting[0] == '1';
    if (!runtime.laguna) return false;
    bool found = false;
    for (const auto& layer : runtime.layers) {
        if (layer.dense_ffn || !layer.expert_tensors[0]) continue;
        for (const auto tensor : layer.expert_tensors) {
            const auto type = runtime.model->tensors[tensor].type;
            if (type != 17 && type != 18 && type != 23) return false;
            found = true;
        }
    }
    return found;
}

// Region timers for the batched CPU MoE. `qwen_cpu_moe` is the largest block of
// prefill (58% of wall on the 35B-A3B) and ~3.5x off its roofline, but it does
// not go through `launch_named`, so COLIBRI_CPU_PROFILE never sees it. Enable
// with COLIBRI_MOE_PROFILE=1; the dump prints once at exit. Off by default --
// the clock reads sit inside the OpenMP task loop.
struct QwenCpuMoeProfile {
    std::atomic<std::uint64_t> setup{0};
    std::atomic<std::uint64_t> gate_dequant{0};
    std::atomic<std::uint64_t> gate_gemm{0};
    std::atomic<std::uint64_t> gate_activate{0};
    std::atomic<std::uint64_t> down_dequant{0};
    std::atomic<std::uint64_t> down_gemm{0};
    std::atomic<std::uint64_t> down_store{0};
    std::atomic<std::uint64_t> combine{0};
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> tasks_gate{0};
    std::atomic<std::uint64_t> tasks_down{0};

    ~QwenCpuMoeProfile() {
        if (!calls.load()) return;
        const double ms = 1.0e6;
        std::fprintf(stderr,
            "[moe] calls=%llu gate{dequant=%.1fms gemm=%.1fms act=%.1fms} "
            "down{dequant=%.1fms gemm=%.1fms store=%.1fms} setup=%.1fms "
            "combine=%.1fms tasks{gate=%llu down=%llu}\n",
            (unsigned long long)calls.load(),
            gate_dequant.load()/ms, gate_gemm.load()/ms, gate_activate.load()/ms,
            down_dequant.load()/ms, down_gemm.load()/ms, down_store.load()/ms,
            setup.load()/ms, combine.load()/ms,
            (unsigned long long)tasks_gate.load(),
            (unsigned long long)tasks_down.load());
    }
};
QwenCpuMoeProfile g_cpu_moe_profile;
bool qwen_cpu_moe_profile_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("COLIBRI_MOE_PROFILE");
        return v && v[0] != '0';
    }();
    return on;
}
inline std::uint64_t qwen_moe_now() {
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

void qwen_cpu_moe_rows(
    const ColibriV2QwenRuntime& runtime, const QwenLayerPlan& layer,
    const std::int32_t* selected, const float* weights, int rows,
    int routed_count, const float* input, float* activated,
    float* down_values, float* output
) {
    const int experts=runtime.model->config.expert_count;
    const int hidden=runtime.model->config.hidden_size;
    const int intermediate=runtime.moe_intermediate;
    const bool direct_quant=qwen_prefill_direct_quant_enabled(runtime);
    const bool direct_oct=direct_quant&&(colibri_cpu_features()&2u)!=0;
    if(rows<=0||rows>4096||routed_count<=0||routed_count>256)
        throw std::runtime_error("native CPU batched MoE shape is unsupported");
    const auto gate_type=runtime.model->tensors[layer.expert_tensors[0]].type;
    const auto up_type=runtime.model->tensors[layer.expert_tensors[1]].type;
    const auto down_type=runtime.model->tensors[layer.expert_tensors[2]].type;
    const char* fused_gate_up_setting=std::getenv("COLIBRI_FUSED_MOE_GATE_UP");
    const char* iq_avx512_setting=std::getenv("COLIBRI_IQ_AVX512");
    const bool auto_fused_iq2xs=gate_type==17
        &&(colibri_cpu_features()&2u)!=0
        &&(!iq_avx512_setting||iq_avx512_setting[0]!='0')
        &&(!fused_gate_up_setting||fused_gate_up_setting[0]!='0');
    for(const auto type:{gate_type,up_type,down_type})
        if(!qwen_cpu_expert_type_supported(type))
            throw std::runtime_error(
                "unsupported native CPU expert quantization: "+std::to_string(type));
    auto expert_data=[&](int role,int expert){
        const auto&t=runtime.model->tensors[layer.expert_tensors[role]];
        return tensor_data(*runtime.model,t)+static_cast<std::uint64_t>(expert)*(t.size/experts);
    };
    // Group routes by expert (CSR over token_rank slots) so each expert's
    // weight rows are decoded once per batch and dotted with every token
    // routed to it. Routes with weight 0 were already claimed by the GPU
    // expert cache (or pruned) and are skipped everywhere below.
    const bool moe_profile=qwen_cpu_moe_profile_enabled();
    const std::uint64_t t_setup0=moe_profile?qwen_moe_now():0;
    const int total=rows*routed_count;
    std::vector<int> counts(experts,0);
    for(int route=0;route<total;++route){
        if(weights[route]==0.0f)continue;
        const int expert=selected[route];
        if(expert<0||expert>=experts)
            throw std::runtime_error("native CPU batched MoE selected an invalid expert");
        ++counts[expert];
    }
    std::vector<int> offsets(experts+1,0);
    for(int expert=0;expert<experts;++expert)offsets[expert+1]=offsets[expert]+counts[expert];
    std::vector<int> occurrences(offsets[experts]);
    std::vector<const float*> vectors(offsets[experts]);
    {
        std::vector<int> cursor(offsets.begin(),offsets.end()-1);
        for(int route=0;route<total;++route){
            if(weights[route]==0.0f)continue;
            const int slot=cursor[selected[route]]++;
            occurrences[slot]=route;
            vectors[slot]=input+static_cast<std::size_t>(route/routed_count)*hidden;
        }
    }
    std::vector<int> group_experts;
    group_experts.reserve(256);
    for(int expert=0;expert<experts;++expert)if(counts[expert])group_experts.push_back(expert);
    const int group_count=static_cast<int>(group_experts.size());
    if(moe_profile){g_cpu_moe_profile.setup+=qwen_moe_now()-t_setup0;
        ++g_cpu_moe_profile.calls;}
    constexpr int kRowBlock=4;
    // Direct IQ rows are already grouped by expert and have thousands of tasks
    // available. Larger hand-out chunks avoid making OpenMP's dynamic scheduler
    // a measurable part of short prompts; keep the finer balance for the f32
    // fallback, whose per-task cost varies much more with route count.
    const int schedule_chunk=direct_quant?32:4;
    const int gate_blocks=(intermediate+kRowBlock-1)/kRowBlock;
#pragma omp parallel for schedule(dynamic,schedule_chunk) num_threads(qwen_cpu_thread_count(runtime))
    for(int task=0;task<group_count*gate_blocks;++task){
        const int group=task/gate_blocks;const int row0=(task%gate_blocks)*kRowBlock;
        const int mr=std::min(kRowBlock,intermediate-row0);
        const int expert=group_experts[group];
        const int begin=offsets[expert],count=counts[expert];
        const auto*gate_data=expert_data(0,expert);
        const auto*up_data=expert_data(1,expert);
        // Per-expert gate/up scales, applied before the (non-linear) SwiGLU. See
        // the matching comment in qwen_cpu_moe.
        const float gate_scale=qwen_expert_role_scale(runtime,layer.expert_gate_scale,expert);
        const float up_scale=qwen_expert_role_scale(runtime,layer.expert_up_scale,expert);
        if(count<=2){
            for(int i=0;i<mr;++i){
                float gate_values[2]{},up_values[2]{};
                if((runtime.fused_moe_gate_up||auto_fused_iq2xs)&&gate_type==up_type){
                    if(count==1){
                        qwen_quant_dot_two_rows(
                            gate_data, up_data, gate_type, vectors[begin],
                            hidden, row0+i, gate_values[0], up_values[0]
                        );
                    }else{
                        qwen_quant_dot_two_rows(
                            gate_data, up_data, gate_type, vectors[begin],
                            hidden, row0+i, gate_values[0], up_values[0]
                        );
                        qwen_quant_dot_two_rows(
                            gate_data, up_data, gate_type, vectors[begin+1],
                            hidden, row0+i, gate_values[1], up_values[1]
                        );
                    }
                }else{
                    if(count==1){
                        gate_values[0]=qwen_quant_dot(gate_data,gate_type,vectors[begin],hidden,row0+i);
                        up_values[0]=qwen_quant_dot(up_data,up_type,vectors[begin],hidden,row0+i);
                    }else{
                        qwen_quant_dot_pair(gate_data,gate_type,vectors[begin],vectors[begin+1],hidden,row0+i,gate_values[0],gate_values[1]);
                        qwen_quant_dot_pair(up_data,up_type,vectors[begin],vectors[begin+1],hidden,row0+i,up_values[0],up_values[1]);
                    }
                }
                for(int occurrence=0;occurrence<count;++occurrence){const int token_rank=occurrences[begin+occurrence];const float gate_value=gate_values[occurrence]*gate_scale,clipped=std::max(-80.0f,std::min(80.0f,gate_value));activated[static_cast<std::size_t>(token_rank)*intermediate+(row0+i)]=gate_value/(1.0f+std::exp(-clipped))*up_values[occurrence]*up_scale;}
            }
            continue;
        }
        if(direct_quant){
            for(int i=0;i<mr;++i)for(int occurrence=0;occurrence<count;){
                const int remaining=count-occurrence;
                const int tile=direct_oct&&remaining>=8?8:std::min(4,remaining);
                float gate_values[8]{},up_values[8]{};
                if(tile==8){
                    const float*tile_vectors[8]={vectors[begin+occurrence],vectors[begin+occurrence+1],vectors[begin+occurrence+2],vectors[begin+occurrence+3],vectors[begin+occurrence+4],vectors[begin+occurrence+5],vectors[begin+occurrence+6],vectors[begin+occurrence+7]};
                    qwen_quant_dot_oct(gate_data,gate_type,tile_vectors,hidden,row0+i,gate_values);
                    qwen_quant_dot_oct(up_data,up_type,tile_vectors,hidden,row0+i,up_values);
                }else if(tile==4){
                    const float*tile_vectors[4]={vectors[begin+occurrence],vectors[begin+occurrence+1],vectors[begin+occurrence+2],vectors[begin+occurrence+3]};
                    qwen_quant_dot_quad(gate_data,gate_type,tile_vectors,hidden,row0+i,gate_values);
                    qwen_quant_dot_quad(up_data,up_type,tile_vectors,hidden,row0+i,up_values);
                }else for(int token=0;token<tile;++token){
                    gate_values[token]=qwen_quant_dot(gate_data,gate_type,vectors[begin+occurrence+token],hidden,row0+i);
                    up_values[token]=qwen_quant_dot(up_data,up_type,vectors[begin+occurrence+token],hidden,row0+i);
                }
                for(int token=0;token<tile;++token){const int token_rank=occurrences[begin+occurrence+token];const float gate_value=gate_values[token]*gate_scale,clipped=std::max(-80.0f,std::min(80.0f,gate_value));activated[static_cast<std::size_t>(token_rank)*intermediate+(row0+i)]=gate_value/(1.0f+std::exp(-clipped))*up_values[token]*up_scale;}
                occurrence+=tile;
            }
            continue;
        }
        thread_local std::vector<float> gate_block,up_block,gate_values,up_values;
        gate_block.resize(static_cast<std::size_t>(kRowBlock)*hidden);up_block.resize(static_cast<std::size_t>(kRowBlock)*hidden);
        gate_values.resize(static_cast<std::size_t>(kRowBlock)*count);up_values.resize(static_cast<std::size_t>(kRowBlock)*count);
        const std::uint64_t t_dq0=moe_profile?qwen_moe_now():0;
        for(int i=0;i<mr;++i){
            qwen_dequant_row(gate_data,gate_type,hidden,row0+i,gate_block.data()+static_cast<std::size_t>(i)*hidden);
            qwen_dequant_row(up_data,up_type,hidden,row0+i,up_block.data()+static_cast<std::size_t>(i)*hidden);
        }
        const std::uint64_t t_gemm0=moe_profile?qwen_moe_now():0;
        qwen_f32_gemm_rows(gate_block.data(),mr,&vectors[begin],count,hidden,gate_values.data());
        qwen_f32_gemm_rows(up_block.data(),mr,&vectors[begin],count,hidden,up_values.data());
        const std::uint64_t t_act0=moe_profile?qwen_moe_now():0;
        if(moe_profile){g_cpu_moe_profile.gate_dequant+=t_gemm0-t_dq0;
            g_cpu_moe_profile.gate_gemm+=t_act0-t_gemm0;
            ++g_cpu_moe_profile.tasks_gate;}
        for(int occurrence=0;occurrence<count;++occurrence){
            const int token_rank=occurrences[begin+occurrence];
            float*dst=activated+static_cast<std::size_t>(token_rank)*intermediate+row0;
            for(int i=0;i<mr;++i){
                const float gate_value=gate_values[static_cast<std::size_t>(i)*count+occurrence]*gate_scale;
                const float clipped=std::max(-80.0f,std::min(80.0f,gate_value));
                dst[i]=gate_value/(1.0f+std::exp(-clipped))*up_values[static_cast<std::size_t>(i)*count+occurrence]*up_scale;
            }
        }
        if(moe_profile)g_cpu_moe_profile.gate_activate+=qwen_moe_now()-t_act0;
    }
    std::vector<const float*> activated_vectors(offsets[experts]);
    for(int slot=0;slot<offsets[experts];++slot)
        activated_vectors[slot]=activated+static_cast<std::size_t>(occurrences[slot])*intermediate;
    const int down_blocks=(hidden+kRowBlock-1)/kRowBlock;
#pragma omp parallel for schedule(dynamic,schedule_chunk) num_threads(qwen_cpu_thread_count(runtime))
    for(int task=0;task<group_count*down_blocks;++task){
        const int group=task/down_blocks;const int row0=(task%down_blocks)*kRowBlock;
        const int mr=std::min(kRowBlock,hidden-row0);
        const int expert=group_experts[group];
        const int begin=offsets[expert],count=counts[expert];
        const auto*down_data=expert_data(2,expert);
        // The down projection is linear, so its per-expert scale rides on the
        // routing weight. Both callers pass the router weights unscaled, unlike
        // the single-token path which folds it in before calling qwen_cpu_moe.
        const float down_scale=qwen_expert_role_scale(runtime,layer.expert_down_scale,expert);
        if(count<=2){
            for(int i=0;i<mr;++i){
                float values[2]{};
                if(count==1)values[0]=qwen_quant_dot(down_data,down_type,activated_vectors[begin],intermediate,row0+i);
                else qwen_quant_dot_pair(down_data,down_type,activated_vectors[begin],activated_vectors[begin+1],intermediate,row0+i,values[0],values[1]);
                for(int occurrence=0;occurrence<count;++occurrence){const int token_rank=occurrences[begin+occurrence];down_values[static_cast<std::size_t>(token_rank)*hidden+(row0+i)]=weights[token_rank]*down_scale*values[occurrence];}
            }
            continue;
        }
        if(direct_quant){
            for(int i=0;i<mr;++i)for(int occurrence=0;occurrence<count;){
                const int remaining=count-occurrence;
                const int tile=direct_oct&&remaining>=8?8:std::min(4,remaining);
                float values[8]{};
                if(tile==8){
                    const float*tile_vectors[8]={activated_vectors[begin+occurrence],activated_vectors[begin+occurrence+1],activated_vectors[begin+occurrence+2],activated_vectors[begin+occurrence+3],activated_vectors[begin+occurrence+4],activated_vectors[begin+occurrence+5],activated_vectors[begin+occurrence+6],activated_vectors[begin+occurrence+7]};
                    qwen_quant_dot_oct(down_data,down_type,tile_vectors,intermediate,row0+i,values);
                }else if(tile==4){
                    const float*tile_vectors[4]={activated_vectors[begin+occurrence],activated_vectors[begin+occurrence+1],activated_vectors[begin+occurrence+2],activated_vectors[begin+occurrence+3]};
                    qwen_quant_dot_quad(down_data,down_type,tile_vectors,intermediate,row0+i,values);
                }else for(int token=0;token<tile;++token)
                    values[token]=qwen_quant_dot(down_data,down_type,activated_vectors[begin+occurrence+token],intermediate,row0+i);
                for(int token=0;token<tile;++token){const int token_rank=occurrences[begin+occurrence+token];down_values[static_cast<std::size_t>(token_rank)*hidden+(row0+i)]=weights[token_rank]*down_scale*values[token];}
                occurrence+=tile;
            }
            continue;
        }
        thread_local std::vector<float> down_block,values;
        down_block.resize(static_cast<std::size_t>(kRowBlock)*intermediate);values.resize(static_cast<std::size_t>(kRowBlock)*count);
        const std::uint64_t t_ddq0=moe_profile?qwen_moe_now():0;
        for(int i=0;i<mr;++i)qwen_dequant_row(down_data,down_type,intermediate,row0+i,down_block.data()+static_cast<std::size_t>(i)*intermediate);
        const std::uint64_t t_dgemm0=moe_profile?qwen_moe_now():0;
        qwen_f32_gemm_rows(down_block.data(),mr,&activated_vectors[begin],count,intermediate,values.data());
        const std::uint64_t t_dst0=moe_profile?qwen_moe_now():0;
        if(moe_profile){g_cpu_moe_profile.down_dequant+=t_dgemm0-t_ddq0;
            g_cpu_moe_profile.down_gemm+=t_dst0-t_dgemm0;
            ++g_cpu_moe_profile.tasks_down;}
        // Token-outermost so each token's mr results land contiguously. The
        // transposed order wrote one 4-byte value per 8 KB-strided cache line
        // of a 67 MB buffer -- a full line fetched, and read-for-owned, per
        // store. `values` is only kRowBlock*count floats, so the strided read
        // it leaves behind stays in L1. Arithmetic is unchanged: the original
        // already grouped as (weights*down_scale)*values, left to right.
        for(int occurrence=0;occurrence<count;++occurrence){
            const int token_rank=occurrences[begin+occurrence];
            const float scale=weights[token_rank]*down_scale;
            float*dst=down_values+static_cast<std::size_t>(token_rank)*hidden+row0;
            for(int i=0;i<mr;++i)
                dst[i]=scale*values[static_cast<std::size_t>(i)*count+occurrence];
        }
        if(moe_profile)g_cpu_moe_profile.down_store+=qwen_moe_now()-t_dst0;
    }
    const std::uint64_t t_comb0=moe_profile?qwen_moe_now():0;
#pragma omp parallel for schedule(static) num_threads(qwen_cpu_thread_count(runtime))
    for(int task=0;task<rows*hidden;++task){
        const int token=task/hidden,row=task%hidden;
        float value=0.0f;
        for(int rank=0;rank<routed_count;++rank){
            const int token_rank=token*routed_count+rank;
            if(weights[token_rank]!=0.0f)
                value+=down_values[static_cast<std::size_t>(token_rank)*hidden+row];
        }
        output[task]=value;
    }
    if(moe_profile)g_cpu_moe_profile.combine+=qwen_moe_now()-t_comb0;
    if (std::getenv("COLIBRI_MOE_DEBUG")) {
        static int rcalls = 0;
        int n = rcalls++;
        if (n < 6) {
            double s = 0; for (int i = 0; i < rows * hidden; ++i) s += std::abs(output[i]);
            std::fprintf(stderr, "[rows-dbg] call=%d rows=%d g/u/d=%u/%u/%u sum|out|=%.4f\n",
                n, rows, gate_type, up_type, down_type, s);
        }
    }
}

int gpu_probe(ColibriV2GpuInfo& out, int device) {
    std::memset(&out, 0, sizeof(out)); out.device = device;
#if defined(_WIN32)
    HMODULE lib = LoadLibraryW(L"nvcuda.dll");
    if (!lib) return 0;
    using Init = int (*)(unsigned int); using Retain = int (*)(void**, int); using Set = int (*)(void*); using SetFlags = int (*)(int, unsigned int); using Attr = int (*)(int*, int, int); using Mem = int (*)(size_t*, size_t*);
    auto init=reinterpret_cast<Init>(GetProcAddress(lib,"cuInit")); auto retain=reinterpret_cast<Retain>(GetProcAddress(lib,"cuDevicePrimaryCtxRetain")); auto set=reinterpret_cast<Set>(GetProcAddress(lib,"cuCtxSetCurrent")); auto set_flags=reinterpret_cast<SetFlags>(GetProcAddress(lib,"cuDevicePrimaryCtxSetFlags")); auto attr=reinterpret_cast<Attr>(GetProcAddress(lib,"cuDeviceGetAttribute")); auto mem=reinterpret_cast<Mem>(GetProcAddress(lib,"cuMemGetInfo_v2"));
    if (!init || !retain || !set || !attr || init(0)!=0) { FreeLibrary(lib); return 0; }
    if(set_flags&&(!std::getenv("COLIBRI_CUDA_SPIN_WAIT")||std::getenv("COLIBRI_CUDA_SPIN_WAIT")[0]!='1'))(void)set_flags(device,0x04);
    void* context=nullptr; if(retain(&context,device)!=0 || set(context)!=0) { FreeLibrary(lib); return 0; }
    int major=0,minor=0; if(attr(&major,75,device)!=0 || attr(&minor,76,device)!=0) { FreeLibrary(lib); return 0; } out.available=1; out.compute_major=major; out.compute_minor=minor; if(mem) { size_t free_bytes=0,total_bytes=0; if(mem(&free_bytes,&total_bytes)==0) { out.free_memory=free_bytes; out.total_memory=total_bytes; } } FreeLibrary(lib); return 0;
#else
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL); if (!lib) lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL); if (!lib) return 0;
    using Init = int (*)(unsigned int); using Retain = int (*)(void**, int); using Set = int (*)(void*); using SetFlags = int (*)(int, unsigned int); using Attr = int (*)(int*, int, int); using Mem = int (*)(size_t*, size_t*);
    auto init=reinterpret_cast<Init>(dlsym(lib,"cuInit")); auto retain=reinterpret_cast<Retain>(dlsym(lib,"cuDevicePrimaryCtxRetain")); auto set=reinterpret_cast<Set>(dlsym(lib,"cuCtxSetCurrent")); auto set_flags=reinterpret_cast<SetFlags>(dlsym(lib,"cuDevicePrimaryCtxSetFlags")); auto attr=reinterpret_cast<Attr>(dlsym(lib,"cuDeviceGetAttribute")); auto mem=reinterpret_cast<Mem>(dlsym(lib,"cuMemGetInfo_v2"));
    if (!init || !retain || !set || !attr || init(0)!=0) { dlclose(lib); return 0; }
    if(set_flags&&(!std::getenv("COLIBRI_CUDA_SPIN_WAIT")||std::getenv("COLIBRI_CUDA_SPIN_WAIT")[0]!='1'))(void)set_flags(device,0x04);
    void* context=nullptr; if(retain(&context,device)!=0 || set(context)!=0) { dlclose(lib); return 0; }
    int major=0,minor=0; if(attr(&major,75,device)!=0 || attr(&minor,76,device)!=0) { dlclose(lib); return 0; } out.available=1; out.compute_major=major; out.compute_minor=minor; if(mem) { size_t free_bytes=0,total_bytes=0; if(mem(&free_bytes,&total_bytes)==0) { out.free_memory=free_bytes; out.total_memory=total_bytes; } } dlclose(lib); return 0;
#endif
}

int plan_memory(ColibriV2MemoryPlan& out, uint64_t budget, uint64_t static_weights, uint64_t kv_state, uint64_t workspace, uint64_t active, uint64_t staging) {
    if (static_weights > budget || kv_state > budget-static_weights || workspace > budget-static_weights-kv_state) { fail("persistent v2 GPU allocations exceed budget"); return -1; }
    out={}; out.budget=budget; out.static_weights=static_weights; out.kv_state=kv_state; out.workspace=workspace; uint64_t left=budget-static_weights-kv_state-workspace; out.active_experts=std::min(active,left); left-=out.active_experts; out.staging=std::min(staging,left); out.unused=left-out.staging; return 0;
}

extern "C" {
uint32_t colibri_v2_version() { return 5; }
uint64_t colibri_v2_runtime_options_size() { return sizeof(ColibriV2QwenRuntimeOptions); }
uint64_t colibri_v2_runtime_info_size() { return sizeof(ColibriV2QwenRuntimeInfo); }
const char* colibri_v2_last_error() { return error.c_str(); }
int colibri_v2_gpu_probe(int32_t device, ColibriV2GpuInfo* out) { return guarded([&]{ if(!out||device<0) throw std::runtime_error("invalid GPU probe arguments"); return gpu_probe(*out,device); }); }
int colibri_v2_memory_plan(uint64_t budget,uint64_t static_weights,uint64_t kv_state,uint64_t workspace,uint64_t active,uint64_t staging,ColibriV2MemoryPlan*out){return guarded([&]{if(!out)throw std::runtime_error("memory plan output is required");return plan_memory(*out,budget,static_weights,kv_state,workspace,active,staging);});}
int colibri_v2_gpu_available(){return colibri_gpu_available();}
int colibri_v2_gpu_init(int32_t device){return colibri_gpu_init(device);}
int colibri_v2_gpu_compile(const char* source,const char*const*options,int32_t count,int32_t device,char*log,int32_t capacity){return colibri_gpu_compile(source,options,count,device,log,capacity);}
int colibri_v2_gpu_rms_norm(uint64_t input,uint64_t weights,uint64_t output,int32_t size,float epsilon,int32_t one_centered){return colibri_gpu_rms_norm(input,weights,output,size,epsilon,one_centered);}
int colibri_v2_gpu_q4_matvec(uint64_t packed,uint64_t scales,uint64_t input,uint64_t output,uint64_t stream,int32_t rows,int32_t columns){return colibri_gpu_q4_matvec(packed,scales,input,output,stream,rows,columns);}
int colibri_v2_gpu_dense_projection(uint64_t input,uint64_t norm_weights,uint64_t normalized,uint64_t packed,uint64_t scales,uint64_t projection,int32_t rows,int32_t columns,float epsilon,int32_t one_centered){int status=colibri_gpu_rms_norm(input,norm_weights,normalized,columns,epsilon,one_centered);if(status)return status;return colibri_gpu_q4_matvec(packed,scales,normalized,projection,0,rows,columns);}
int colibri_v2_gpu_dense_residual(uint64_t input,uint64_t norm_weights,uint64_t normalized,uint64_t packed,uint64_t scales,uint64_t output,int32_t rows,int32_t columns,float epsilon,int32_t one_centered){if(rows!=columns)return -3;int status=colibri_gpu_rms_norm(input,norm_weights,normalized,columns,epsilon,one_centered);if(status)return status;status=colibri_gpu_q4_matvec(packed,scales,normalized,output,0,rows,columns);if(status)return status;return colibri_gpu_scaled_add(output,input,1.0f,rows);}
int colibri_v2_gpu_attention(uint64_t query,uint64_t keys,uint64_t values,uint64_t output,int32_t heads,int32_t kv_heads,int32_t head_dim,int32_t tokens,float scale){return colibri_gpu_attention(query,keys,values,output,heads,kv_heads,head_dim,tokens,scale);}
int colibri_v2_gpu_decoder_attention_step(uint64_t input,uint64_t norm_weights,uint64_t normalized,uint64_t qkv_packed,uint64_t qkv_scales,uint64_t qkv,uint64_t cache_keys,uint64_t cache_values,uint64_t attention_output,uint64_t out_packed,uint64_t out_scales,uint64_t output,int32_t hidden_size,int32_t heads,int32_t kv_heads,int32_t head_dim,int32_t position,int32_t capacity,float epsilon,int32_t one_centered){if(heads<=0||kv_heads<=0||heads%kv_heads!=0||head_dim<=0||position<0||position>=capacity)return -3;int status=colibri_gpu_rms_norm(input,norm_weights,normalized,hidden_size,epsilon,one_centered);if(status)return status;const int q_rows=heads*head_dim;const int kv_rows=kv_heads*head_dim;status=colibri_gpu_q4_matvec(qkv_packed,qkv_scales,normalized,qkv,0,q_rows+2*kv_rows,hidden_size);if(status)return status;status=colibri_gpu_kv_append(qkv+q_rows*sizeof(float),qkv+(q_rows+kv_rows)*sizeof(float),cache_keys,cache_values,kv_heads,head_dim,position,capacity);if(status)return status;status=colibri_gpu_attention_cache(qkv,cache_keys,cache_values,attention_output,heads,kv_heads,head_dim,position+1,capacity,1.0f/sqrtf(static_cast<float>(head_dim)));if(status)return status;status=colibri_gpu_q4_matvec(out_packed,out_scales,attention_output,output,0,hidden_size,heads*head_dim);if(status)return status;return colibri_gpu_scaled_add(output,input,1.0f,hidden_size);}
int colibri_v2_kv_cache_create(uint64_t keys,uint64_t values,int32_t capacity,int32_t kv_heads,int32_t head_dim,ColibriV2KvCache**out){return guarded([&]{if(!out||!keys||!values||capacity<=0||kv_heads<=0||head_dim<=0)throw std::runtime_error("invalid KV cache arguments");*out=new ColibriV2KvCache{keys,values,capacity,kv_heads,head_dim,0};return 0;});}
void colibri_v2_kv_cache_destroy(ColibriV2KvCache*cache){try{delete cache;}catch(...){}}
int colibri_v2_kv_cache_reset(ColibriV2KvCache*cache){return guarded([&]{if(!cache){fail("invalid KV cache");return -1;}cache->position=0;return 0;});}
int colibri_v2_kv_cache_position(const ColibriV2KvCache*cache,int32_t*out){return guarded([&]{if(!cache||!out){fail("invalid KV cache position");return -1;}*out=cache->position;return 0;});}
int colibri_v2_gpu_decoder_attention_cached(ColibriV2KvCache*cache,uint64_t input,uint64_t norm_weights,uint64_t normalized,uint64_t qkv_packed,uint64_t qkv_scales,uint64_t qkv,uint64_t attention_output,uint64_t out_packed,uint64_t out_scales,uint64_t output,int32_t hidden_size,int32_t heads,float epsilon,int32_t one_centered){return guarded([&]{if(!cache){fail("invalid KV cache");return -1;}int status=colibri_v2_gpu_decoder_attention_step(input,norm_weights,normalized,qkv_packed,qkv_scales,qkv,cache->keys,cache->values,attention_output,out_packed,out_scales,output,hidden_size,heads,cache->kv_heads,cache->head_dim,cache->position,cache->capacity,epsilon,one_centered);if(status)return status;++cache->position;return 0;});}
// Map one GGUF file and parse its header. Shards of a split checkpoint come
// through here too, which is why it does not attach shards itself.
void map_and_parse(const char* path, ColibriV2Model& model) { ColibriV2Model* m=&model; m->path=path;
#if !defined(_WIN32)
    m->fd=open(path,O_RDONLY); if(m->fd<0) throw std::runtime_error("cannot open GGUF"); struct stat st{}; if(fstat(m->fd,&st)!=0) throw std::runtime_error("cannot stat GGUF"); m->size=static_cast<size_t>(st.st_size);
    const char* lock_env=std::getenv("COLIBRI_V2_MLOCK"); const bool lock_model=lock_env&&lock_env[0]=='1';
    int map_flags=MAP_PRIVATE;
#ifdef MAP_POPULATE
    if(lock_model) map_flags|=MAP_POPULATE; // prefault every page so decode never page-faults on a cold expert
#endif
    // MAP_PRIVATE keeps the GGUF immutable on disk, while PROT_WRITE makes the
    // mapping eligible for CUDA host registration selected later by runtime
    // options. Model open necessarily happens before `--expert-paging` is
    // applied, so mapping permissions cannot depend on that option.
    m->data=static_cast<const uint8_t*>(mmap(nullptr,m->size,PROT_READ|PROT_WRITE,map_flags,m->fd,0)); if(m->data==MAP_FAILED) throw std::runtime_error("cannot map GGUF");
#if defined(MADV_HUGEPAGE)
    // Ask for 2 MiB pages. On the CPU backend the weights are read straight out
    // of this mapping (see qwen_alias_static_tensor), so a decode walks the
    // whole non-expert weight set every token -- 2.5 GiB for a 35B MoE, which
    // is ~650k pages at 4 KiB.
    //
    // Measured no effect on the machine this was written on: smaps_rollup
    // reported FilePmdMapped = 0 afterwards, i.e. the kernel declined to back a
    // file mapping with large folios despite THP being set to `always`. It is
    // kept because it is advisory, free, and correct where file THP is
    // supported -- but do not assume it is doing anything without checking
    // FilePmdMapped, and do not attribute a speedup to it on that assumption.
    (void)madvise(const_cast<uint8_t*>(m->data),m->size,MADV_HUGEPAGE);
#endif
    if(lock_model&&mlock(m->data,m->size)!=0) std::fprintf(stderr,"colibri_v2: mlock(%zu bytes) failed: %s; continuing without pinning (raise RLIMIT_MEMLOCK to pin)\n",m->size,std::strerror(errno));
#else
    m->file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(m->file==INVALID_HANDLE_VALUE){m->file=nullptr;throw std::runtime_error("cannot open GGUF");}
    LARGE_INTEGER file_size{}; if(!GetFileSizeEx(m->file,&file_size)) throw std::runtime_error("cannot stat GGUF");
    m->size=static_cast<size_t>(file_size.QuadPart);
    // Read-only mapping: the GGUF is immutable (m->data is const and nothing
    // writes through it). Unlike PAGE_WRITECOPY, PAGE_READONLY is file-backed and
    // charges no pagefile commit, so mapping a 20+ GiB model does not fail under
    // commit pressure. CUDA host registration uses the READ_ONLY flag (0x08).
    m->mapping=CreateFileMappingW(m->file,nullptr,PAGE_READONLY,0,0,nullptr);
    if(!m->mapping) throw std::runtime_error("cannot create GGUF mapping");
    m->data=static_cast<const uint8_t*>(MapViewOfFile(m->mapping,FILE_MAP_READ,0,0,0));
    if(!m->data) throw std::runtime_error("cannot map GGUF");
    const char* lock_env=std::getenv("COLIBRI_V2_MLOCK"); const bool lock_model=lock_env&&lock_env[0]=='1';
    if(lock_model){
        // Touch every page to prefault (MAP_POPULATE equivalent on Windows)
        volatile uint8_t sink=0; const size_t page_size=4096;
        for(size_t offset=0;offset<m->size;offset+=page_size) sink^=m->data[offset];
        // Attempt to enable SeLockMemoryPrivilege in the process token so
        // VirtualLock can pin the full working set.  This avoids requiring
        // the user to configure Local Security Policy manually.
        {
            HANDLE token=nullptr;
            if(OpenProcessToken(GetCurrentProcess(),TOKEN_ADJUST_PRIVILEGES|TOKEN_QUERY,&token)){
                TOKEN_PRIVILEGES tp{}; tp.PrivilegeCount=1;
                tp.Privileges[0].Attributes=SE_PRIVILEGE_ENABLED;
                if(LookupPrivilegeValueW(nullptr,L"SeLockMemoryPrivilege",&tp.Privileges[0].Luid))
                    AdjustTokenPrivileges(token,FALSE,&tp,0,nullptr,nullptr);
                CloseHandle(token);
            }
        }
        const size_t chunk=1024ULL*1024ULL*1024ULL; size_t locked=0;
        while(locked<m->size){
            size_t len=m->size-locked; if(len>chunk) len=chunk;
            if(VirtualLock(const_cast<uint8_t*>(m->data+locked),len)!=0) locked+=len;
            else break;
        }
        if(locked==0)
            std::fprintf(stderr,"colibri_v2: VirtualLock failed (error %lu) — pages prefaulted "
                "but not pinned; enable SeLockMemoryPrivilege for pinning\n",GetLastError());
        else if(locked<m->size)
            std::fprintf(stderr,"colibri_v2: pinned %zu/%zu bytes\n",locked,m->size);
        else
            std::fprintf(stderr,"colibri_v2: pinned all %zu bytes\n",locked);
    }
#endif
    parse(*m); }

// gguf-split names shards "<prefix>-00001-of-00004.gguf" with a 1-based file
// index, while GGUF's own split.no is 0-based.
std::string split_shard_path(const std::string& path, std::uint32_t index, std::uint32_t count) {
    constexpr std::size_t kSuffix = 20; // "-00001-of-00004.gguf"
    const auto shaped=[&]{
        if(path.size()<=kSuffix)return false;
        const char* tail=path.c_str()+path.size()-kSuffix;
        if(tail[0]!='-'||std::memcmp(tail+6,"-of-",4)!=0||std::memcmp(tail+15,".gguf",5)!=0)return false;
        for(int digit=0;digit<5;++digit)
            if(!std::isdigit(static_cast<unsigned char>(tail[1+digit]))||
               !std::isdigit(static_cast<unsigned char>(tail[10+digit])))return false;
        return true;
    }();
    if(!shaped)throw std::runtime_error(
        "split GGUF path does not carry a \"-00001-of-0000N.gguf\" suffix: "+path);
    char suffix[32];
    std::snprintf(suffix,sizeof(suffix),"-%05u-of-%05u.gguf",index+1,count);
    return path.substr(0,path.size()-kSuffix)+suffix;
}

// Merge the remaining shards of a split checkpoint into `m`. Their descriptors
// keep pointing at their own mapping through Tensor::source, so downstream code
// never learns the model arrived in pieces.
void attach_split_shards(ColibriV2Model& m) {
    if(m.split_count<=1)return;
    if(m.split_no>=m.split_count)
        throw std::runtime_error("split GGUF shard index is outside its own split count");
    m.shards.reserve(m.split_count-1);
    for(std::uint32_t index=0;index<m.split_count;++index){
        if(index==m.split_no)continue;
        const auto path=split_shard_path(m.path,index,m.split_count);
        auto shard=std::make_unique<ColibriV2Model>();
        map_and_parse(path.c_str(),*shard);
        if(shard->split_count!=m.split_count||shard->split_no!=index)
            throw std::runtime_error("split GGUF shard disagrees about its place in the split: "+path);
        for(const auto& tensor:shard->tensors){
            auto merged=tensor;
            merged.source=shard->data;
            m.tensors.push_back(std::move(merged));
        }
        m.shards.push_back(std::move(shard));
    }
    if(m.split_tensors&&m.tensors.size()!=m.split_tensors)
        throw std::runtime_error("split GGUF holds "+std::to_string(m.tensors.size())+
            " tensors but declares "+std::to_string(m.split_tensors));
    // The draft block can live in any shard, so this only becomes decidable
    // once every descriptor is present.
    detect_mtp_layer(m);
}

int colibri_v2_model_open(const char* path, ColibriV2Model** out) { return guarded([&]{
    if(!path||!out) throw std::runtime_error("path and output are required");
    auto m=std::make_unique<ColibriV2Model>();
    map_and_parse(path,*m);
    attach_split_shards(*m);
    *out=m.release(); return 0; }); }
int colibri_v2_model_attach_mtp(ColibriV2Model* m,const char*path){return guarded([&]{
    if(!m||!path||!path[0])throw std::runtime_error("model and MTP sidecar path are required");
    if(m->mtp_sidecar)throw std::runtime_error("an MTP sidecar is already attached");
    ColibriV2Model*opened=nullptr;
    if(colibri_v2_model_open(path,&opened)!=0)
        throw std::runtime_error(error.empty()?"failed to open MTP sidecar":error);
    std::unique_ptr<ColibriV2Model>sidecar(opened);
    if(sidecar->mtp_layer==std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("MTP sidecar has no draft block");
    if(m->mtp_layer!=std::numeric_limits<std::uint32_t>::max()&&
       sidecar->mtp_layer!=m->mtp_layer)
        throw std::runtime_error("MTP sidecar draft layer does not match the base GGUF");
    if(sidecar->config.hidden_size!=m->config.hidden_size||
       sidecar->config.attention_heads!=m->config.attention_heads||
       sidecar->config.attention_kv_heads!=m->config.attention_kv_heads||
       sidecar->config.expert_count!=m->config.expert_count||
       sidecar->config.expert_used_count!=m->config.expert_used_count)
        throw std::runtime_error("MTP sidecar model geometry does not match the base GGUF");
    if(m->mtp_layer==std::numeric_limits<std::uint32_t>::max())
        m->mtp_layer=sidecar->mtp_layer;
    const std::string prefix="blk."+std::to_string(m->mtp_layer)+".";
    std::size_t replaced=0;
    for(auto&target:m->tensors){
        if(target.name.rfind(prefix,0)!=0)continue;
        const auto found=std::find_if(sidecar->tensors.begin(),sidecar->tensors.end(),
            [&](const Tensor&candidate){return candidate.name==target.name;});
        if(found==sidecar->tensors.end())
            throw std::runtime_error("MTP sidecar is missing tensor: "+target.name);
        auto elements=[](const Tensor&tensor){std::uint64_t result=1;for(const auto dimension:tensor.shape)result*=dimension;return result;};
        if(found->shape!=target.shape&&elements(*found)!=elements(target))
            throw std::runtime_error("MTP sidecar tensor shape mismatch: "+target.name);
        target.type=found->type;
        target.offset=found->offset;
        target.size=found->size;
        target.source=sidecar->data;
        ++replaced;
    }
    // Trunk-only GGUFs omit the optional draft block entirely. Append its
    // descriptors while keeping their bytes in the sidecar mapping.
    for(const auto&candidate:sidecar->tensors){
        if(candidate.name.rfind(prefix,0)!=0)continue;
        const auto present=std::any_of(m->tensors.begin(),m->tensors.end(),
            [&](const Tensor&target){return target.name==candidate.name;});
        if(present)continue;
        auto added=candidate;
        added.source=sidecar->data;
        m->tensors.push_back(std::move(added));
        ++replaced;
    }
    if(!replaced)throw std::runtime_error("MTP sidecar did not replace any tensors");
    m->mtp_sidecar=std::move(sidecar);
    return 0;
});}
void colibri_v2_model_close(ColibriV2Model* m) { try{delete m;}catch(...){} }
int colibri_v2_model_info(const ColibriV2Model* m, ColibriV2ModelInfo* out) { return guarded([&]{if(!m||!out)throw std::runtime_error("invalid model info handle"); std::memset(out,0,sizeof(*out));out->gguf_version=m->version;out->tensor_count=m->tensor_count();out->metadata_count=m->metadata;out->file_size=m->size;out->alignment=m->alignment;copy_text(out->architecture,sizeof(out->architecture),m->architecture);copy_text(out->name,sizeof(out->name),m->name);copy_text(out->format,sizeof(out->format),m->format());return 0;}); }
int colibri_v2_model_chat_template(const ColibriV2Model* m,char*out,uint64_t capacity,uint64_t*length){return guarded([&]{if(!m||!length)throw std::runtime_error("invalid model chat-template arguments");*length=static_cast<uint64_t>(m->chat_template.size());if(!out||capacity==0)return 0;if(capacity<=m->chat_template.size())throw std::runtime_error("chat-template output buffer is too small");std::memcpy(out,m->chat_template.data(),m->chat_template.size());out[m->chat_template.size()]=0;return 0;});}
int colibri_v2_model_config(const ColibriV2Model* m, ColibriV2ModelConfig* out){return guarded([&]{if(!m||!out)throw std::runtime_error("invalid model config handle");std::memset(out,0,sizeof(*out));copy_text(out->architecture,sizeof(out->architecture),m->config.architecture);out->hidden_size=m->config.hidden_size;out->layer_count=m->config.layer_count;out->attention_heads=m->config.attention_heads;out->attention_kv_heads=m->config.attention_kv_heads;out->context_length=m->config.context_length;out->intermediate_size=m->config.intermediate_size;out->expert_count=m->config.expert_count;out->expert_used_count=m->config.expert_used_count;out->vocabulary_size=m->config.vocabulary_size;out->rotary_dimension=m->config.rotary_dimension;out->full_attention_interval=m->config.full_attention_interval;out->sliding_window=m->config.sliding_window;out->sliding_window_pattern_length=static_cast<uint32_t>(m->config.sliding_window_pattern.size());out->rms_norm_epsilon=m->config.rms_norm_epsilon;out->rope_freq_base=m->config.rope_freq_base;out->eos_token_id=m->config.eos_token_id;out->eot_token_id=m->config.eot_token_id;out->bos_token_id=m->config.bos_token_id;
out->q_lora_rank=m->config.q_lora_rank;out->kv_lora_rank=m->config.kv_lora_rank;out->output_lora_rank=m->config.output_lora_rank;out->output_group_count=m->config.output_group_count;
out->indexer_head_count=m->config.indexer_head_count;out->indexer_key_length=m->config.indexer_key_length;out->indexer_top_k=m->config.indexer_top_k;
out->hyper_connection_count=m->config.hyper_connection_count;out->sinkhorn_iterations=m->config.sinkhorn_iterations;
out->expert_shared_count=m->config.expert_shared_count;out->hash_layer_count=m->config.hash_layer_count;
out->compress_ratios_length=static_cast<std::uint32_t>(m->config.compress_ratios.size());
out->sinkhorn_epsilon=m->config.sinkhorn_epsilon;out->compress_rope_freq_base=m->config.compress_rope_freq_base;
out->rope_scaling_factor=m->config.rope_scaling_factor;out->yarn_beta_fast=m->config.yarn_beta_fast;out->yarn_beta_slow=m->config.yarn_beta_slow;
out->rope_original_context_length=m->config.rope_original_context_length;
out->draft_block_size=m->config.draft_block_size;
out->target_layers_length=static_cast<std::uint32_t>(m->config.target_layers.size());
out->mask_token_id=m->config.mask_token_id;return 0;});}

// Multiply a model tensor by a vector, decoding whatever weight type the
// checkpoint stores it as. GGUF reports a matrix as [inputs, outputs], so
// `output_size` rows are each a dot product over `input_size` elements.
int colibri_v2_matvec(
    const ColibriV2Model* m, const char* name, const float* input,
    int32_t input_size, float* output, int32_t output_size
){return guarded([&]{
    if(!m||!name||!input||!output)throw std::runtime_error("matvec arguments are required");
    const auto found=std::find_if(m->tensors.begin(),m->tensors.end(),
        [&](const Tensor& tensor){return tensor.name==name;});
    if(found==m->tensors.end())throw std::runtime_error(std::string("tensor not found: ")+name);
    if(found->shape.size()!=2)throw std::runtime_error("matvec needs a 2D tensor");
    if(static_cast<std::int64_t>(found->shape[0])!=input_size||
       static_cast<std::int64_t>(found->shape[1])!=output_size)
        throw std::runtime_error("matvec shape does not match the tensor");
    if(!qwen_cpu_expert_type_supported(found->type))
        throw std::runtime_error("matvec cannot decode this weight type");
    const auto* packed=tensor_data(*m,*found);
    for(std::int32_t row=0;row<output_size;++row)
        output[row]=qwen_quant_dot(packed,found->type,input,input_size,
            static_cast<std::uint64_t>(row));
    return 0;});}

// RMS normalization over `count` consecutive rows of `size`, optionally scaled
// by a per-element gain. DeepSeek-V4 uses both forms: the query and KV latents
// are normalized with a learned gain, while the per-head query norm has none.
int colibri_v2_deepseek4_rms_norm(
    const float* input, const float* weight, int32_t size, int32_t count,
    float epsilon, float* output
){return guarded([&]{
    if(!input||!output||size<=0||count<=0)
        throw std::runtime_error("rms-norm arguments are invalid");
    for(std::int32_t row=0;row<count;++row){
        const auto offset=static_cast<std::size_t>(row)*static_cast<std::size_t>(size);
        colibri::v2::deepseek4::rms_norm(
            input+offset,static_cast<std::size_t>(size),epsilon,output+offset);
        if(weight)
            for(std::int32_t i=0;i<size;++i)output[offset+i]*=weight[i];
    }
    return 0;});}

// Grouped matvec: the input is cut into `groups` chunks and chunk g is
// multiplied by the g-th slice of the tensor's output rows. `colibri_v2_matvec`
// is the groups==1 case; DeepSeek-V4's output projection uses eight.
int colibri_v2_grouped_matvec(
    const ColibriV2Model* m, const char* name, const float* input,
    int32_t inputs, float* output, int32_t rank, int32_t groups
){return guarded([&]{
    if(!m||!name||!input||!output||inputs<=0||rank<=0||groups<=0)
        throw std::runtime_error("grouped matvec arguments are invalid");
    const auto found=std::find_if(m->tensors.begin(),m->tensors.end(),
        [&](const Tensor& tensor){return tensor.name==name;});
    if(found==m->tensors.end())throw std::runtime_error(std::string("tensor not found: ")+name);
    if(found->shape.size()!=2||static_cast<std::int64_t>(found->shape[0])!=inputs||
       static_cast<std::int64_t>(found->shape[1])!=static_cast<std::int64_t>(rank)*groups)
        throw std::runtime_error("grouped matvec shape does not match the tensor");
    if(!qwen_cpu_expert_type_supported(found->type))
        throw std::runtime_error("grouped matvec cannot decode this weight type");
    const auto* packed=tensor_data(*m,*found);
    colibri::v2::deepseek4::grouped_projection(
        input,static_cast<std::size_t>(inputs),static_cast<std::size_t>(rank),
        static_cast<std::size_t>(groups),output,
        [&](const float* source,std::size_t row){
            return qwen_quant_dot(packed,found->type,source,inputs,
                static_cast<std::uint64_t>(row));
        });
    return 0;});}

// Matvec against one expert of a stacked expert tensor. The experts are laid
// out back to back, so expert e's rows start at e*outputs.
int colibri_v2_expert_matvec(
    const ColibriV2Model* m, const char* name, int32_t expert,
    const float* input, int32_t inputs, float* output, int32_t outputs
){return guarded([&]{
    if(!m||!name||!input||!output||expert<0||inputs<=0||outputs<=0)
        throw std::runtime_error("expert matvec arguments are invalid");
    const auto found=std::find_if(m->tensors.begin(),m->tensors.end(),
        [&](const Tensor& tensor){return tensor.name==name;});
    if(found==m->tensors.end())throw std::runtime_error(std::string("tensor not found: ")+name);
    if(found->shape.size()!=3||static_cast<std::int64_t>(found->shape[0])!=inputs||
       static_cast<std::int64_t>(found->shape[1])!=outputs)
        throw std::runtime_error("expert matvec shape does not match the tensor");
    if(expert>=static_cast<std::int64_t>(found->shape[2]))
        throw std::runtime_error("expert index is out of range");
    if(!qwen_cpu_expert_type_supported(found->type))
        throw std::runtime_error("expert matvec cannot decode this weight type");
    const auto* packed=tensor_data(*m,*found);
    const auto base=static_cast<std::uint64_t>(expert)*outputs;
    for(std::int32_t row=0;row<outputs;++row)
        output[row]=qwen_quant_dot(packed,found->type,input,inputs,base+row);
    return 0;});}

// Collapse the streams for the output head, which needs only pre-weights.
int colibri_v2_deepseek4_head(
    const float* streams, const float* fn, const float* scale, const float* base,
    int32_t n_embd, int32_t hc, float rms_epsilon, float hc_epsilon,
    float* pre, float* output
){return guarded([&]{
    if(!streams||!fn||!scale||!base||!pre||!output||n_embd<=0||hc<=0)
        throw std::runtime_error("head arguments are invalid");
    colibri::v2::deepseek4::hyper_connection_head(
        streams,fn,scale,base,static_cast<std::size_t>(n_embd),
        static_cast<std::size_t>(hc),rms_epsilon,hc_epsilon,pre,output);
    return 0;});}

// One sequence's DeepSeek-V4 state, sized from the architecture rather than
// from the context length wherever the architecture allows it.
//
// Raw latents are bounded by the sliding window on every layer, so they live in
// a ring of `window` entries rather than growing with the sequence. The
// compressor keeps only what a future block can still read: a 4:1 layer pools
// the previous block's rows alongside its own, so two blocks' worth, and a
// 128:1 layer pools only its own. Just the compressed caches scale with
// context, at context/ratio entries.
// Every weight one block reads, resolved to a tensor index once.
//
// The lookups this replaces are linear scans by name over all 1328 descriptors.
// A block reads about twenty weights, so a 43-layer token would spend roughly
// 900 scans of a 1328-entry vector on nothing but finding them again.
struct Deepseek4LayerPlan {
    static constexpr std::uint64_t kAbsent = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t attn_norm = kAbsent, q_a = kAbsent, q_a_norm = kAbsent, q_b = kAbsent;
    std::uint64_t kv = kAbsent, kv_a_norm = kAbsent, sinks = kAbsent;
    std::uint64_t output_a = kAbsent, output_b = kAbsent;
    std::uint64_t hc_attn_fn = kAbsent, hc_attn_scale = kAbsent, hc_attn_base = kAbsent;
    std::uint64_t hc_ffn_fn = kAbsent, hc_ffn_scale = kAbsent, hc_ffn_base = kAbsent;
    std::uint64_t ffn_norm = kAbsent, gate_inp = kAbsent;
    std::uint64_t gate_exps = kAbsent, up_exps = kAbsent, down_exps = kAbsent;
    std::uint64_t gate_shexp = kAbsent, up_shexp = kAbsent, down_shexp = kAbsent;
    // Exactly one of these two: hash layers route by table, the rest by bias.
    std::uint64_t tid2eid = kAbsent, exp_probs_b = kAbsent;
    // Compressed layers only.
    std::uint64_t comp_kv = kAbsent, comp_gate = kAbsent, comp_ape = kAbsent, comp_norm = kAbsent;
    // 4:1 layers only.
    std::uint64_t indexer_proj = kAbsent, indexer_q_b = kAbsent;
    std::uint64_t indexer_comp_kv = kAbsent, indexer_comp_gate = kAbsent;
    std::uint64_t indexer_comp_ape = kAbsent, indexer_comp_norm = kAbsent;

    std::uint32_t resolved() const {
        const std::uint64_t* first = &attn_norm;
        const std::uint64_t* last = &indexer_comp_norm;
        std::uint32_t count = 0;
        for (const std::uint64_t* it = first; it <= last; ++it) count += (*it != kAbsent);
        return count;
    }
};

struct Deepseek4LayerState {
    Deepseek4LayerPlan plan;
    std::uint32_t ratio = 0, window = 0, head_dim = 0;
    std::uint32_t state_width = 0, state_rows = 0, block_capacity = 0;
    // Half precision, as the reference stores them. The composed model rounds
    // its latents the same way, so nothing is lost by matching.
    std::vector<std::uint16_t> latents;    // window x head_dim, ring
    std::vector<std::uint16_t> compressed; // block_capacity x head_dim
    std::vector<float> state_values;  // state_rows x state_width, ring
    std::vector<float> state_scores;
    std::uint32_t positions = 0, blocks = 0;
    // The lightning indexer's own compressor and cache, on 4:1 layers only. The
    // same shape as the pair above at a narrower width -- 128 against 512 --
    // because it only has to rank blocks, not answer with them.
    std::uint32_t indexer_dim = 0, indexer_width = 0;
    std::vector<std::uint16_t> indexer_compressed;
    std::vector<float> indexer_state_values, indexer_state_scores;

    std::uint64_t bytes() const {
        return (latents.capacity() + compressed.capacity() + indexer_compressed.capacity())
                   * sizeof(std::uint16_t) +
               (state_values.capacity() + state_scores.capacity() +
                indexer_state_values.capacity() + indexer_state_scores.capacity()) * sizeof(float);
    }
    // The ring holds the newest `state_rows` rows, so a row's slot is its
    // position modulo the ring size.
    std::uint32_t slot(std::uint32_t position) const { return position % state_rows; }
};

// Scratch a single position's forward needs, allocated once per runtime.
struct Deepseek4Scratch {
    std::vector<float> streams, next_streams, collapsed, hidden;
    std::vector<float> pre, post, comb;
    std::vector<float> low_rank, query, latent, block_out, grouped;
    std::vector<float> keys, attn, derope;
    std::vector<float> state_row, gate, up, activated, expert_out, moe, logits;
    std::vector<float> router, after, embedding, head_pre;
    std::vector<float> routed_gate, routed_up, routed_activated, routed_out;
    std::vector<float> indexer_query, indexer_weights, indexer_scores, indexer_keys;
    std::vector<std::uint8_t> mask, indexer_keep;
    std::vector<std::int32_t> experts;
    std::vector<float> weights;
    std::vector<std::uint8_t> gpu_experts;
};

struct Deepseek4ExpertSlot {
    std::uint64_t key = 0, last_used = 0;
    bool valid = false;
};

struct ColibriV2Deepseek4Runtime {
    ColibriV2Model* model = nullptr;
    std::uint32_t context_limit = 0;
    std::vector<Deepseek4LayerState> layers;
    Deepseek4Scratch scratch;
    // Geometry the loop reads on every layer.
    std::uint32_t n_embd = 0, hc = 0, heads = 0, head_dim = 0, rope_dim = 0;
    std::uint32_t q_lora = 0, groups = 0, lora_rank = 0, experts = 0, experts_used = 0;
    std::uint32_t expert_ffn = 0, vocabulary = 0, sinkhorn_iterations = 0, hash_layers = 0;
    std::uint32_t indexer_heads = 0, indexer_dim = 0, indexer_top_k = 0;
    float epsilon = 0.0f, freq_base = 0.0f, weight_scale = 1.5f, clamp = 10.0f;
    float compress_base = 0.0f, freq_scale = 1.0f, attn_factor = 1.0f;
    float beta_fast = 32.0f, beta_slow = 1.0f;
    std::uint32_t original_context = 0;
    std::uint64_t head_fn = 0, head_scale = 0, head_base = 0, output_norm = 0, output = 0;
    std::uint64_t token_embd = 0;

    // Where a token's time goes. Attribution rather than a single number: the
    // routed experts are believed to be nearly all of it, and paging work is
    // only worth doing if that is measured rather than assumed.
    std::uint64_t forward_calls = 0;
    std::uint64_t forward_nanoseconds = 0;
    std::uint64_t routed_expert_nanoseconds = 0, shared_expert_nanoseconds = 0;
    std::uint64_t attention_nanoseconds = 0, head_nanoseconds = 0;
    std::uint64_t attention_core_nanoseconds = 0;
    std::uint64_t routed_expert_bytes = 0;
    // How much the indexer is actually discarding, which is zero until a
    // sequence is long enough for it to run at all.
    std::uint64_t indexer_selections = 0, indexer_candidates = 0;
    // Weight prefetch, on unless COLIBRI_DS4_PREFETCH=off, which exists so the
    // two regimes can be compared rather than assumed.
    bool expert_prefetch = true;
    std::uint64_t expert_prefetch_bytes = 0;
    // The dense half, resident on the device: tensor index -> device pointer.
    // The routed experts are never in here -- 90 GiB against 12 of VRAM -- which
    // is what makes the split this model wants the opposite of the usual one.
    bool gpu = false;
    bool gpu_owner = false;
    std::int32_t gpu_device = 0;
    std::unordered_map<std::uint64_t, std::uint64_t> resident;
    std::uint64_t gpu_input = 0, gpu_output = 0;
    std::uint64_t gpu_input_capacity = 0, gpu_output_capacity = 0;
    std::uint64_t gpu_weight_bytes = 0, gpu_matvec_calls = 0, gpu_batches = 0;
    std::uint64_t gpu_batch = 0, gpu_batch_capacity = 0;
    std::uint64_t expert_cache = 0, expert_cache_bytes = 0, expert_slot_bytes = 0;
    std::uint32_t expert_slots_per_layer = 0;
    std::uint64_t expert_cache_clock = 0, expert_cache_hits = 0;
    std::uint64_t expert_cache_misses = 0, expert_cache_evictions = 0;
    std::vector<Deepseek4ExpertSlot> expert_slots;
    std::unordered_map<std::uint64_t, std::size_t> expert_residency;
    std::unordered_map<std::uint64_t, std::uint32_t> expert_frequency;
    ColibriV2Deepseek4Runtime* gpu_cache_owner = nullptr;
    std::uint64_t hyper_nanoseconds = 0, matvec_nanoseconds = 0;
    std::uint64_t prefill_calls = 0, prefill_tokens = 0, prefill_nanoseconds = 0;
    std::vector<std::uint32_t> capture_layers;
    std::vector<float> captured;

    std::uint64_t state_bytes() const {
        std::uint64_t total = 0;
        for (const auto& layer : layers) total += layer.bytes();
        return total;
    }
    void reset() {
        for (auto& layer : layers) { layer.positions = 0; layer.blocks = 0; }
    }
};
struct ColibriV2Deepseek4Snapshot {
    std::vector<Deepseek4LayerState> layers;
    std::vector<float> captured;
};

int colibri_v2_deepseek4_runtime_create(
    ColibriV2Model* model, uint32_t context_limit, ColibriV2Deepseek4Runtime** out
){return guarded([&]{
    if(!model||!out||!context_limit)
        throw std::runtime_error("model, context limit and output are required");
    const bool draft_model=model->config.architecture=="dflash";
    if(model->config.architecture!="deepseek4"&&!draft_model)
        throw std::runtime_error("not a deepseek4/dflash model");
    const auto& config=model->config;
    if(config.context_length&&context_limit>config.context_length)
        throw std::runtime_error("deepseek4 context limit exceeds the checkpoint context length");
    if(config.compress_ratios.size()<config.layer_count)
        throw std::runtime_error("compress-ratio array is shorter than the layer count");
    // One pass over the descriptors builds the name index every layer plan
    // draws from, so resolution is linear in the model rather than quadratic.
    std::unordered_map<std::string,std::uint64_t> by_name;
    by_name.reserve(model->tensors.size()*2);
    for(std::uint64_t index=0;index<model->tensors.size();++index)
        by_name.emplace(model->tensors[index].name,index);
    auto runtime=std::make_unique<ColibriV2Deepseek4Runtime>();
    runtime->model=model;
    runtime->context_limit=context_limit;
    runtime->layers.resize(config.layer_count);
    const auto head_dim=config.kv_lora_rank?config.kv_lora_rank:config.key_length;
    // A zero sliding window would mean unbounded raw attention; the checkpoint
    // sets 128, and falling back to the context keeps the arithmetic honest
    // rather than silently allocating nothing.
    const auto window=config.sliding_window?config.sliding_window:context_limit;
    for(std::uint32_t index=0;index<config.layer_count;++index){
        auto& layer=runtime->layers[index];
        layer.ratio=config.compress_ratios[index];
        layer.window=std::min(window,context_limit);
        layer.head_dim=head_dim;
        layer.latents.assign(static_cast<std::size_t>(layer.window)*head_dim,0u);

        const std::string prefix="blk."+std::to_string(index)+".";
        auto find=[&](const char* suffix,bool required)->std::uint64_t{
            const auto found=by_name.find(prefix+suffix);
            if(found!=by_name.end())return found->second;
            if(required)
                throw std::runtime_error("deepseek4 layer is missing "+prefix+suffix);
            return Deepseek4LayerPlan::kAbsent;
        };
        auto& plan=layer.plan;
        plan.attn_norm=find("attn_norm.weight",true);
        plan.q_a=find("attn_q_a.weight",true);
        plan.q_a_norm=find("attn_q_a_norm.weight",true);
        plan.q_b=find("attn_q_b.weight",true);
        plan.kv=find("attn_kv.weight",true);
        plan.kv_a_norm=find("attn_kv_a_norm.weight",true);
        plan.sinks=find("attn_sinks.weight",true);
        plan.output_a=find("attn_output_a.weight",true);
        plan.output_b=find("attn_output_b.weight",true);
        plan.hc_attn_fn=find("hc_attn_fn.weight",true);
        plan.hc_attn_scale=find("hc_attn_scale.weight",true);
        plan.hc_attn_base=find("hc_attn_base.weight",true);
        plan.hc_ffn_fn=find("hc_ffn_fn.weight",true);
        plan.hc_ffn_scale=find("hc_ffn_scale.weight",true);
        plan.hc_ffn_base=find("hc_ffn_base.weight",true);
        plan.ffn_norm=find("ffn_norm.weight",true);
        plan.gate_inp=find("ffn_gate_inp.weight",true);
        plan.gate_exps=find("ffn_gate_exps.weight",true);
        plan.up_exps=find("ffn_up_exps.weight",true);
        plan.down_exps=find("ffn_down_exps.weight",true);
        plan.gate_shexp=find("ffn_gate_shexp.weight",true);
        plan.up_shexp=find("ffn_up_shexp.weight",true);
        plan.down_shexp=find("ffn_down_shexp.weight",true);
        // A hash layer routes from a table and carries no bias; every other
        // layer is the reverse. Requiring exactly one catches a checkpoint whose
        // hash_layer_count disagrees with its tensors.
        const bool hashed=index<config.hash_layer_count;
        plan.tid2eid=find("ffn_gate_tid2eid.weight",hashed);
        plan.exp_probs_b=find("exp_probs_b.bias",!hashed);
        if((plan.tid2eid!=Deepseek4LayerPlan::kAbsent)==
           (plan.exp_probs_b!=Deepseek4LayerPlan::kAbsent))
            throw std::runtime_error(
                "deepseek4 layer "+std::to_string(index)+
                " must carry either a routing table or a router bias, not both or neither");

        if(!layer.ratio)continue;
        plan.comp_kv=find("attn_compressor_kv.weight",true);
        plan.comp_gate=find("attn_compressor_gate.weight",true);
        plan.comp_ape=find("attn_compressor_ape.weight",true);
        plan.comp_norm=find("attn_compressor_norm.weight",true);
        if(layer.ratio==4){
            plan.indexer_proj=find("indexer.proj.weight",true);
            plan.indexer_q_b=find("indexer.attn_q_b.weight",true);
            plan.indexer_comp_kv=find("indexer_compressor_kv.weight",true);
            plan.indexer_comp_gate=find("indexer_compressor_gate.weight",true);
            plan.indexer_comp_ape=find("indexer_compressor_ape.weight",true);
            plan.indexer_comp_norm=find("indexer_compressor_norm.weight",true);
        }
        const bool overlapped=layer.ratio==4;
        layer.state_width=(overlapped?2u:1u)*head_dim;
        layer.state_rows=(overlapped?2u:1u)*layer.ratio;
        layer.block_capacity=context_limit/layer.ratio+1;
        // Compressed caches grow with the sequence. Eagerly materializing the
        // advertised one-million-token context consumed several GiB before the
        // first token even arrived.
        layer.compressed.clear();
        const std::size_t rows=static_cast<std::size_t>(layer.state_rows)*layer.state_width;
        layer.state_values.assign(rows,0.0f);
        layer.state_scores.assign(rows,-std::numeric_limits<float>::infinity());
        if(layer.ratio==4&&config.indexer_key_length){
            // The indexer compresses the same blocks at the same ratio, so it
            // shares the ring geometry and differs only in width.
            layer.indexer_dim=config.indexer_key_length;
            layer.indexer_width=2u*layer.indexer_dim;
            layer.indexer_compressed.clear();
            const std::size_t indexer_rows=
                static_cast<std::size_t>(layer.state_rows)*layer.indexer_width;
            layer.indexer_state_values.assign(indexer_rows,0.0f);
            layer.indexer_state_scores.assign(
                indexer_rows,-std::numeric_limits<float>::infinity());
        }
    }
    auto& g=*runtime;
    g.n_embd=config.hidden_size; g.hc=config.hyper_connection_count;
    g.heads=config.attention_heads; g.head_dim=head_dim;
    g.rope_dim=config.rotary_dimension; g.q_lora=config.q_lora_rank;
    g.groups=config.output_group_count; g.lora_rank=config.output_lora_rank;
    g.experts=config.expert_count; g.experts_used=config.expert_used_count;
    g.expert_ffn=config.expert_intermediate_size?config.expert_intermediate_size:config.intermediate_size;
    g.vocabulary=config.vocabulary_size; g.sinkhorn_iterations=config.sinkhorn_iterations;
    g.hash_layers=config.hash_layer_count; g.epsilon=config.rms_norm_epsilon;
    {
        const char* setting=std::getenv("COLIBRI_DS4_PREFETCH");
        g.expert_prefetch=!(setting&&std::string(setting)=="off");
    }
    g.indexer_heads=config.indexer_head_count; g.indexer_dim=config.indexer_key_length;
    g.indexer_top_k=config.indexer_top_k;
    g.freq_base=config.rope_freq_base; g.compress_base=config.compress_rope_freq_base;
    g.weight_scale=config.expert_weights_scale?config.expert_weights_scale:1.0f;
    g.freq_scale=config.rope_scaling_factor>0.0f?1.0f/config.rope_scaling_factor:1.0f;
    g.attn_factor=1.0f/(1.0f+0.1f*std::log(1.0f/g.freq_scale));
    g.beta_fast=config.yarn_beta_fast; g.beta_slow=config.yarn_beta_slow;
    g.original_context=config.rope_original_context_length;
    auto require=[&](const char* name)->std::uint64_t{
        const auto found=by_name.find(name);
        if(found==by_name.end())throw std::runtime_error(std::string("model is missing ")+name);
        return found->second;
    };
    g.head_fn=require("output_hc_fn.weight"); g.head_scale=require("output_hc_scale.weight");
    g.head_base=require("output_hc_base.weight"); g.output_norm=require("output_norm.weight");
    if(draft_model){
        g.output=Deepseek4LayerPlan::kAbsent;
        g.token_embd=Deepseek4LayerPlan::kAbsent;
    }else{
        g.output=require("output.weight"); g.token_embd=require("token_embd.weight");
    }

    auto& sc=g.scratch;
    const std::size_t wide=static_cast<std::size_t>(g.heads)*g.head_dim;
    sc.streams.assign(static_cast<std::size_t>(g.hc)*g.n_embd,0.0f);
    sc.next_streams.assign(sc.streams.size(),0.0f);
    sc.collapsed.assign(g.n_embd,0.0f); sc.hidden.assign(g.n_embd,0.0f);
    sc.pre.assign(g.hc,0.0f); sc.post.assign(g.hc,0.0f); sc.comb.assign(static_cast<std::size_t>(g.hc)*g.hc,0.0f);
    sc.low_rank.assign(g.q_lora,0.0f); sc.query.assign(wide,0.0f);
    sc.latent.assign(g.head_dim,0.0f); sc.block_out.assign(g.n_embd,0.0f);
    sc.grouped.assign(static_cast<std::size_t>(g.groups)*g.lora_rank,0.0f);
    sc.attn.assign(wide,0.0f); sc.derope.assign(wide,0.0f);
    sc.state_row.assign(2ull*g.head_dim,0.0f);
    sc.gate.assign(g.expert_ffn,0.0f); sc.up.assign(g.expert_ffn,0.0f);
    sc.activated.assign(g.expert_ffn,0.0f); sc.expert_out.assign(g.n_embd,0.0f);
    sc.moe.assign(g.n_embd,0.0f); sc.logits.assign(g.vocabulary,0.0f);
    sc.router.assign(g.experts,0.0f);
    sc.after.assign(static_cast<std::size_t>(g.hc)*g.n_embd,0.0f);
    sc.embedding.assign(g.n_embd,0.0f); sc.head_pre.assign(g.hc,0.0f);
    sc.routed_gate.assign(static_cast<std::size_t>(g.experts_used)*g.expert_ffn,0.0f);
    sc.routed_up.assign(sc.routed_gate.size(),0.0f);
    sc.routed_activated.assign(sc.routed_gate.size(),0.0f);
    sc.routed_out.assign(static_cast<std::size_t>(g.experts_used)*g.n_embd,0.0f);
    sc.experts.assign(g.experts_used,0); sc.weights.assign(g.experts_used,0.0f);
    sc.gpu_experts.assign(g.experts_used,0);
    // Keys for one attention: the window plus every compressed block.
    std::uint32_t widest=0;
    for(const auto& layer:g.layers){
        const auto compressed = layer.indexer_dim && g.indexer_top_k
            ? std::min(layer.block_capacity,g.indexer_top_k)
            : layer.block_capacity;
        widest=std::max(widest,layer.window+compressed);
    }
    sc.keys.assign(static_cast<std::size_t>(widest)*g.head_dim,0.0f);
    sc.mask.assign(widest,0);
    // The indexer's scratch: one query per head, one weight per head, and a
    // score and a verdict per compressed block.
    if(g.indexer_dim){
        std::uint32_t blocks=0;
        for(const auto& layer:g.layers)
            if(layer.indexer_dim)blocks=std::max(blocks,layer.block_capacity);
        sc.indexer_query.assign(static_cast<std::size_t>(g.indexer_heads)*g.indexer_dim,0.0f);
        sc.indexer_weights.assign(g.indexer_heads,0.0f);
        sc.indexer_scores.assign(blocks,0.0f);
        sc.indexer_keep.assign(blocks,0);
    }

    *out=runtime.release();
    return 0;});}

void colibri_v2_deepseek4_runtime_free(ColibriV2Deepseek4Runtime* runtime){
    try{
        if(runtime&&runtime->gpu_owner){
            for(const auto& entry:runtime->resident)colibri_gpu_free(entry.second);
            if(runtime->gpu_input)colibri_gpu_free(runtime->gpu_input);
            if(runtime->gpu_output)colibri_gpu_free(runtime->gpu_output);
            if(runtime->gpu_batch)colibri_gpu_free(runtime->gpu_batch);
            if(runtime->expert_cache)colibri_gpu_free(runtime->expert_cache);
        }
        delete runtime;
    }catch(...){}
}

int colibri_v2_deepseek4_runtime_reset(ColibriV2Deepseek4Runtime* runtime){return guarded([&]{
    if(!runtime)throw std::runtime_error("invalid runtime");
    runtime->reset();
    return 0;});}

namespace {

// How wide to fork a matvec: one thread per core, the policy the kernels share.
//
// Measured on this checkpoint: the default team is one thread per *hardware*
// thread, and on a 16-core/32-thread part that is 2.6x slower than one thread
// per core -- 9.3 GiB/s of expert weights against 24.6, and 1.9 tok/s against
// 3.2. Two threads sharing a core contend for the same L1 and the same decode
// units, and the loop forks about eight hundred times per token, so the wider
// team costs barrier time as well. deepseek4's pragmas simply never took a
// `num_threads`.
int ds4_thread_count() { return colibri::v2::deepseek4::thread_count(); }

void ds4_pin_current_thread() {
#if defined(_OPENMP) && !defined(_WIN32)
    static const bool enabled=[] {
        const char* setting=std::getenv("COLIBRI_DS4_AFFINITY");
        return setting&&std::string(setting)=="on";
    }();
    static const std::vector<int> cpus=[] {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        std::vector<int> result;
        if(sched_getaffinity(0,sizeof(allowed),&allowed)==0)
            for(int cpu=0;cpu<CPU_SETSIZE;++cpu)
                if(CPU_ISSET(cpu,&allowed))result.push_back(cpu);
        return result;
    }();
    thread_local bool pinned=false;
    if(!enabled||pinned||cpus.empty())return;
    const int target=omp_get_thread_num();
    // The OpenMP master is also the long-lived inference/scheduler thread.
    // Pinning it leaked affinity into unrelated work after the parallel region;
    // workers are persistent and safe to bind, while the master stays mobile.
    if(target==0)return;
    const int cpu=cpus[static_cast<std::size_t>(target)%cpus.size()];
    cpu_set_t one;
    CPU_ZERO(&one); CPU_SET(cpu,&one);
    if(sched_setaffinity(0,sizeof(one),&one)==0)pinned=true;
#endif
}

// Compile the CUDA kernels once per process.
//
// The set is the Qwen one: the dense half of deepseek4 is Q8_0, Q6_K, BF16 and
// F32, and a matvec for each of those already exists because the Qwen work
// needed them. What deepseek4 adds later -- the Sinkhorn mixer, the compressor,
// the indexer -- goes into the same source, so this stays the one place that
// builds them.
void ds4_gpu_compile(std::int32_t device) {
    static bool built = false;
    static std::mutex guard;
    std::lock_guard<std::mutex> held(guard);
    if (built) return;
    std::vector<std::string> option_storage;
#if !defined(_WIN32)
    for (const char* path : {"/opt/cuda/include", "/usr/local/cuda/include", "/usr/include"})
        if (access((std::string(path) + "/cuda_fp16.h").c_str(), R_OK) == 0) {
            option_storage.push_back(std::string("-I") + path);
            if (access((std::string(path) + "/cccl/cub/config.cuh").c_str(), R_OK) == 0)
                option_storage.push_back(std::string("-I") + path + "/cccl");
        }
#endif
    if (const char* cuda_home = std::getenv("CUDA_HOME")) {
        option_storage.push_back(std::string("-I") + cuda_home + "/include");
        option_storage.push_back(std::string("-I") + cuda_home + "/include/cccl");
    }
    std::vector<const char*> options;
    for (const auto& option : option_storage) options.push_back(option.c_str());
    std::array<char, 16384> log{};
    std::string iq1_grid =
        "extern \"C\" __device__ __constant__ unsigned long long "
        "ds4_iq1s_grid[2048]={";
    for (const auto entry : kIq1sGrid) {
        iq1_grid += std::to_string(entry);
        iq1_grid += "ULL,";
    }
    iq1_grid += "};\n";
    const std::string source =
        std::string(colibri::v2::qwen_cuda_source) + colibri::v2::qwen_native_cuda_source +
        iq1_grid + deepseek4_cuda_source;
    if (colibri_gpu_compile(source.c_str(), options.data(),
                            static_cast<std::int32_t>(options.size()), device,
                            log.data(), static_cast<std::int32_t>(log.size())) != 0)
        throw std::runtime_error(std::string("failed to compile CUDA kernels: ") + log.data());
    built = true;
}

// Dispatch a device matvec on the weight's own type. The dense half of this
// checkpoint uses four of these; anything else is a placement mistake caught
// here rather than a wrong answer later.
int ds4_gpu_matvec(std::uint32_t type, std::uint64_t weights, std::uint64_t input,
                   std::uint64_t output, std::int32_t inputs, std::int32_t outputs,
                   std::uint64_t stream) {
    switch (type) {
        case 19: {  // IQ1_S
            void* arguments[] = {&weights, &input, &output, &inputs, &outputs};
            const auto blocks = static_cast<std::uint32_t>((outputs + 7) / 8);
            return colibri_gpu_launch_named("ds4_iq1s_matvec", blocks, 1, 256, 0,
                                            stream, arguments);
        }
        case 8: {  // Q8_0
            // Four blocks in flight per warp against the shared kernel's one.
            void* arguments[] = {&weights, &input, &output, &inputs, &outputs};
            const auto blocks = static_cast<std::uint32_t>((outputs + 7) / 8);
            return colibri_gpu_launch_named("ds4_q8_matvec", blocks, 1, 256, 0,
                                            stream, arguments);
        }
        case 14: {  // Q6_K
            void* arguments[] = {&weights, &input, &output, &inputs, &outputs};
            const auto blocks = static_cast<std::uint32_t>((outputs + 7) / 8);
            return colibri_gpu_launch_named("ds4_q6k_matvec", blocks, 1, 256, 0,
                                            stream, arguments);
        }
        case 30:  // BF16
            return colibri_gpu_bf16_matvec_transposed(weights, input, output, inputs, outputs, stream);
        default:
            return qwen_gpu_matvec_by_type(
                type, weights, input, output, inputs, outputs, stream);
    }
}

// Hint that a weight range is about to be read.
//
// On the mapping rather than the file, because a split checkpoint backs
// different tensors with different descriptors and the address is the one thing
// that identifies a range whichever shard it came from.
void ds4_prefetch(const std::uint8_t* address, std::uint64_t bytes) {
#if !defined(_WIN32)
    if (!address || !bytes) return;
    static const std::uintptr_t page =
        static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
    const auto start = reinterpret_cast<std::uintptr_t>(address) & ~(page - 1);
    const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
    (void)madvise(reinterpret_cast<void*>(start), static_cast<std::size_t>(end - start),
                  MADV_WILLNEED);
#else
    (void)address; (void)bytes;
#endif
}

// A monotonic nanosecond stamp, for attributing a token's time.
std::uint64_t ds4_now() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Multiply a resolved tensor by a vector. The plan holds indices, so this does
// no lookup -- the difference from colibri_v2_matvec, which searches by name.
// Several projections of the same vector, in one round trip.
//
// A layer reads its normalized hidden state through six or seven different
// weights -- the query's low-rank half, the key/value latent, both compressors
// and the indexer's -- and each of those was crossing separately: upload the
// same four kilobytes, launch, download, wait. The weights they multiply do not
// move at all, so the crossing was most of what a small projection cost.
//
// Uploading once, launching all of them, and waiting once at the end turns
// six stalls into one. The launches are independent and ordered on a single
// stream, so nothing here changes an answer.
struct Ds4Projection {
    std::uint64_t index;
    float* out;
    std::int32_t outputs;
    bool download = true;
};

void ds4_matvec(ColibriV2Deepseek4Runtime& rt, std::uint64_t index,
                const float* input, std::int32_t inputs, float* out,
                std::int32_t outputs);

bool ds4_matvec_batch(ColibriV2Deepseek4Runtime& rt, const Ds4Projection* batch,
                      std::size_t count, const float* input, std::int32_t inputs) {
    // Off by an environment switch so the two shapes can be compared in one
    // process: the machine's page cache and this laptop's clocks drift enough
    // between runs to swamp the difference otherwise.
    static const bool batching = [] {
        const char* setting = std::getenv("COLIBRI_DS4_BATCH");
        return !(setting && std::string(setting) == "off");
    }();
    bool device = rt.gpu && batching;
    std::uint64_t total = 0;
    for (std::size_t item = 0; device && item < count; ++item) {
        if (rt.resident.find(batch[item].index) == rt.resident.end()) device = false;
        total += static_cast<std::uint64_t>(batch[item].outputs) * sizeof(float);
    }
    const auto input_bytes = static_cast<std::uint64_t>(inputs) * sizeof(float);
    if (!device || input_bytes > rt.gpu_input_capacity || total > rt.gpu_batch_capacity) {
        for (std::size_t item = 0; item < count; ++item)
            ds4_matvec(rt, batch[item].index, input, inputs, batch[item].out,
                       batch[item].outputs);
        return false;
    }
    const auto batch_started = ds4_now();
    int status = colibri_gpu_upload(rt.gpu_input, input, input_bytes, 0);
    std::uint64_t offset = 0;
    for (std::size_t item = 0; item < count && status == 0; ++item) {
        const auto& tensor = rt.model->tensors[batch[item].index];
        status = ds4_gpu_matvec(tensor.type, rt.resident[batch[item].index], rt.gpu_input,
                                rt.gpu_batch + offset, inputs, batch[item].outputs, 0);
        offset += static_cast<std::uint64_t>(batch[item].outputs) * sizeof(float);
    }
    offset = 0;
    for (std::size_t item = 0; item < count && status == 0; ++item) {
        const auto bytes = static_cast<std::uint64_t>(batch[item].outputs) * sizeof(float);
        if (batch[item].download)
            status = colibri_gpu_download(batch[item].out, rt.gpu_batch + offset, bytes, 0);
        offset += bytes;
    }
    if (status == 0) status = colibri_gpu_sync();
    if (status != 0) throw std::runtime_error("a batched device matvec failed");
    rt.matvec_nanoseconds += ds4_now() - batch_started;
    rt.gpu_matvec_calls += count;
    ++rt.gpu_batches;
    return true;
}

void ds4_cached_experts(
    ColibriV2Deepseek4Runtime& rt, std::uint32_t layer_index,
    const Deepseek4LayerPlan& plan, Deepseek4Scratch& sc
) {
    std::fill(sc.gpu_experts.begin(),sc.gpu_experts.end(),0);
    auto* cache=rt.gpu_cache_owner?rt.gpu_cache_owner:&rt;
    if(!rt.gpu||!cache->expert_cache||!cache->expert_slots_per_layer)return;
    const auto& model=*rt.model;
    const auto& gate=model.tensors[plan.gate_exps];
    const auto& up=model.tensors[plan.up_exps];
    const auto& down=model.tensors[plan.down_exps];
    // The profitable path groups every hit into two launches. IQ2_XXS has only
    // a scalar matvec kernel, and the launch-heavy version benchmarked slower
    // than the fused host loop, so those layers deliberately remain on CPU.
    if(gate.type!=19||up.type!=19||down.type!=18)return;
    const std::uint64_t gate_bytes=gate.size/rt.experts;
    const std::uint64_t up_bytes=up.size/rt.experts;
    const std::uint64_t down_bytes=down.size/rt.experts;
    const auto begin=static_cast<std::size_t>(layer_index)*cache->expert_slots_per_layer;
    const auto end=begin+cache->expert_slots_per_layer;
    std::array<std::uint64_t,8> gate_ptrs{},up_ptrs{},down_ptrs{};
    std::array<float,8> gpu_weights{};
    std::uint32_t gpu_count=0;
    for(std::uint32_t selected=0;selected<rt.experts_used;++selected){
        const auto expert=static_cast<std::uint32_t>(sc.experts[selected]);
        const auto key=(static_cast<std::uint64_t>(layer_index)<<32)|expert;
        auto found=cache->expert_residency.find(key);
        std::size_t slot_index=0;
        if(found!=cache->expert_residency.end()){
            slot_index=found->second;
            ++cache->expert_cache_hits;
            auto& frequency=cache->expert_frequency[key];
            if(frequency!=std::numeric_limits<std::uint32_t>::max())++frequency;
        }else{
            ++cache->expert_cache_misses;
            auto& frequency=cache->expert_frequency[key];
            if(frequency!=std::numeric_limits<std::uint32_t>::max())++frequency;
            // A one-off route is cheaper on the fused CPU path than a PCIe
            // upload. Admit only after recurrence proves there is reuse.
            if(frequency<2)continue;
            auto free=std::find_if(cache->expert_slots.begin()+begin,
                cache->expert_slots.begin()+end,
                [](const Deepseek4ExpertSlot& slot){return !slot.valid;});
            auto victim=free;
            if(victim==cache->expert_slots.begin()+end){
                victim=std::min_element(cache->expert_slots.begin()+begin,
                    cache->expert_slots.begin()+end,
                    [&](const Deepseek4ExpertSlot& a,const Deepseek4ExpertSlot& b){
                        const auto af=cache->expert_frequency[a.key];
                        const auto bf=cache->expert_frequency[b.key];
                        return af!=bf?af<bf:a.last_used<b.last_used;
                    });
                if(frequency<=cache->expert_frequency[victim->key])continue;
                cache->expert_residency.erase(victim->key);
                ++cache->expert_cache_evictions;
            }
            slot_index=static_cast<std::size_t>(victim-cache->expert_slots.begin());
            const auto base=cache->expert_cache+slot_index*cache->expert_slot_bytes;
            std::uint64_t offset=0;
            for(const auto role:{plan.gate_exps,plan.up_exps,plan.down_exps}){
                const auto& tensor=model.tensors[role];
                const auto bytes=tensor.size/rt.experts;
                if(colibri_gpu_upload(base+offset,tensor_data(model,tensor)+
                        static_cast<std::uint64_t>(expert)*bytes,bytes,0)!=0)
                    throw std::runtime_error("DeepSeek expert-cache upload failed");
                offset+=bytes;
            }
            victim->key=key; victim->valid=true;
            cache->expert_residency[key]=slot_index;
            // The admission upload is allowed to finish alongside this token's
            // CPU expert calculation; it becomes a GPU hit next time.
            victim->last_used=++cache->expert_cache_clock;
            continue;
        }
        auto& slot=cache->expert_slots[slot_index];
        slot.last_used=++cache->expert_cache_clock;
        const auto base=cache->expert_cache+slot_index*cache->expert_slot_bytes;
        gate_ptrs[gpu_count]=base;
        up_ptrs[gpu_count]=base+gate_bytes;
        down_ptrs[gpu_count]=base+gate_bytes+up_bytes;
        gpu_weights[gpu_count]=sc.weights[selected];
        sc.gpu_experts[selected]=1;
        ++gpu_count;
    }
    if(!gpu_count)return;
    struct PointerBundle {
        std::uint64_t gate[8],up[8],down[8];
        float weights[8];
    } bundle{};
    std::copy_n(gate_ptrs.begin(),gpu_count,bundle.gate);
    std::copy_n(up_ptrs.begin(),gpu_count,bundle.up);
    std::copy_n(down_ptrs.begin(),gpu_count,bundle.down);
    std::copy_n(gpu_weights.begin(),gpu_count,bundle.weights);
    std::uint64_t pointer_buffer=rt.gpu_batch;
    std::uint64_t gate_device=pointer_buffer;
    std::uint64_t up_device=gate_device+sizeof(bundle.gate);
    std::uint64_t down_device=up_device+sizeof(bundle.up);
    std::uint64_t weights_device=down_device+sizeof(bundle.down);
    std::uint64_t aggregate=weights_device+sizeof(bundle.weights);
    int input_size=static_cast<int>(rt.n_embd);
    int intermediate=static_cast<int>(rt.expert_ffn);
    int output_size=static_cast<int>(rt.n_embd);
    int count=static_cast<int>(gpu_count);
    int status=colibri_gpu_upload(rt.gpu_input,sc.hidden.data(),
        static_cast<std::uint64_t>(rt.n_embd)*sizeof(float),0);
    if(status==0)status=colibri_gpu_upload(pointer_buffer,&bundle,sizeof(bundle),0);
    if(status==0)status=colibri_gpu_memset(
        aggregate,0,static_cast<std::uint64_t>(rt.n_embd)*sizeof(float),0);
    void* gate_args[]={&gate_device,&up_device,&rt.gpu_input,&rt.gpu_output,
        &input_size,&intermediate,&count,&rt.clamp};
    if(status==0)status=colibri_gpu_launch_named("ds4_iq1s_grouped_swiglu",
        rt.expert_ffn,gpu_count,256,0,0,gate_args);
    void* down_args[]={&down_device,&rt.gpu_output,&aggregate,&weights_device,
        &intermediate,&output_size,&count};
    if(status==0)status=colibri_gpu_launch_named("iq3xxs_grouped_accumulate",
        rt.n_embd,1,256,0,0,down_args);
    if(status==0)status=colibri_gpu_download(sc.expert_out.data(),aggregate,
        static_cast<std::uint64_t>(rt.n_embd)*sizeof(float),0);
    if(status==0)status=colibri_gpu_sync();
    if(status!=0)throw std::runtime_error("DeepSeek grouped cached experts failed");
    for(std::uint32_t i=0;i<rt.n_embd;++i)sc.moe[i]+=sc.expert_out[i];
    rt.gpu_matvec_calls+=3*gpu_count;
}

void ds4_quant_dot_many(
    const std::uint8_t* packed, std::uint32_t type,
    const std::vector<const float*>& vectors, int begin, int count,
    int elements, std::uint64_t row, float* output
) {
    if(count==8){
        const float* inputs[8]={vectors[begin],vectors[begin+1],vectors[begin+2],
            vectors[begin+3],vectors[begin+4],vectors[begin+5],vectors[begin+6],
            vectors[begin+7]};
        qwen_quant_dot_oct(packed,type,inputs,elements,row,output);
        return;
    }
    int offset=0;
    for(;offset+4<=count;offset+=4){
        const float* inputs[4]={vectors[begin+offset],vectors[begin+offset+1],
            vectors[begin+offset+2],vectors[begin+offset+3]};
        qwen_quant_dot_quad(packed,type,inputs,elements,row,output+offset);
    }
    if(offset+2<=count){
        qwen_quant_dot_pair(packed,type,vectors[begin+offset],vectors[begin+offset+1],
            elements,row,output[offset],output[offset+1]);
        offset+=2;
    }
    if(offset<count)
        output[offset]=qwen_quant_dot(packed,type,vectors[begin+offset],elements,row);
}

void ds4_cpu_moe_rows(
    ColibriV2Deepseek4Runtime& rt, std::uint32_t layer_index,
    std::vector<Deepseek4Scratch>& rows
) {
    const auto& model=*rt.model;
    const auto& plan=rt.layers[layer_index].plan;
    const int row_count=static_cast<int>(rows.size());
    const int top_k=static_cast<int>(rt.experts_used);
    const int total=row_count*top_k;
    std::vector<int> counts(rt.experts,0),offsets(rt.experts+1,0);
    for(int row=0;row<row_count;++row)
        for(int slot=0;slot<top_k;++slot)++counts[rows[row].experts[slot]];
    for(std::uint32_t expert=0;expert<rt.experts;++expert)
        offsets[expert+1]=offsets[expert]+counts[expert];
    std::vector<int> occurrences(total),cursor(offsets.begin(),offsets.end()-1);
    std::vector<const float*> hidden(total),activated(total);
    for(int row=0;row<row_count;++row)for(int slot=0;slot<top_k;++slot){
        const int route=row*top_k+slot;
        const int expert=rows[row].experts[slot];
        const int position=cursor[expert]++;
        occurrences[position]=route;
        hidden[position]=rows[row].hidden.data();
        activated[position]=rows[row].routed_activated.data()+
            static_cast<std::size_t>(slot)*rt.expert_ffn;
    }
    std::vector<int> active;
    for(std::uint32_t expert=0;expert<rt.experts;++expert)
        if(counts[expert])active.push_back(static_cast<int>(expert));
    const auto& gate=model.tensors[plan.gate_exps];
    const auto& up=model.tensors[plan.up_exps];
    const auto& down=model.tensors[plan.down_exps];
    const auto gate_span=gate.size/rt.experts,up_span=up.size/rt.experts;
    const auto down_span=down.size/rt.experts;
#pragma omp parallel for schedule(dynamic,16) num_threads(ds4_thread_count())
    for(std::int64_t task=0;task<static_cast<std::int64_t>(active.size())*rt.expert_ffn;++task){
        const int group=static_cast<int>(task/rt.expert_ffn);
        const int output=static_cast<int>(task%rt.expert_ffn);
        const int expert=active[group],begin=offsets[expert],count=counts[expert];
        float gate_values[8]{},up_values[8]{};
        ds4_quant_dot_many(tensor_data(model,gate)+expert*gate_span,gate.type,
            hidden,begin,count,rt.n_embd,output,gate_values);
        ds4_quant_dot_many(tensor_data(model,up)+expert*up_span,up.type,
            hidden,begin,count,rt.n_embd,output,up_values);
        for(int item=0;item<count;++item){
            const int route=occurrences[begin+item];
            const int row=route/top_k,slot=route%top_k;
            const float g=std::clamp(gate_values[item],-rt.clamp,rt.clamp);
            const float u=std::clamp(up_values[item],-rt.clamp,rt.clamp);
            rows[row].routed_activated[static_cast<std::size_t>(slot)*rt.expert_ffn+output]=
                g/(1.0f+std::exp(-g))*u;
        }
    }
#pragma omp parallel for schedule(dynamic,16) num_threads(ds4_thread_count())
    for(std::int64_t task=0;task<static_cast<std::int64_t>(active.size())*rt.n_embd;++task){
        const int group=static_cast<int>(task/rt.n_embd);
        const int output=static_cast<int>(task%rt.n_embd);
        const int expert=active[group],begin=offsets[expert],count=counts[expert];
        float values[8]{};
        ds4_quant_dot_many(tensor_data(model,down)+expert*down_span,down.type,
            activated,begin,count,rt.expert_ffn,output,values);
        for(int item=0;item<count;++item){
            const int route=occurrences[begin+item];
            const int row=route/top_k,slot=route%top_k;
            rows[row].routed_out[static_cast<std::size_t>(slot)*rt.n_embd+output]=values[item];
        }
    }
    for(int row=0;row<row_count;++row){
        auto& sc=rows[row];
        std::fill(sc.moe.begin(),sc.moe.end(),0.0f);
        for(int slot=0;slot<top_k;++slot){
            const float weight=sc.weights[slot];
            const float* output=sc.routed_out.data()+static_cast<std::size_t>(slot)*rt.n_embd;
            for(std::uint32_t item=0;item<rt.n_embd;++item)sc.moe[item]+=output[item]*weight;
        }
    }
}

void ds4_matvec(ColibriV2Deepseek4Runtime& rt, std::uint64_t index,
                const float* input, std::int32_t inputs, float* out, std::int32_t outputs) {
    const auto& model = *rt.model;
    const auto& tensor = model.tensors[index];
    // On the device when the weights are already there. The activations cross
    // for every call, which is a few kilobytes against the megabytes of weights
    // that do not -- that asymmetry is the whole reason residency pays.
    if (rt.gpu) {
        const auto found = rt.resident.find(index);
        if (found != rt.resident.end()) {
            const auto input_bytes = static_cast<std::uint64_t>(inputs) * sizeof(float);
            const auto output_bytes = static_cast<std::uint64_t>(outputs) * sizeof(float);
            if (input_bytes <= rt.gpu_input_capacity && output_bytes <= rt.gpu_output_capacity) {
                // Asynchronous: the upload, the launch and the download are
                // ordered on one stream, so a single wait at the end covers all
                // three. Waiting on the upload as well doubled the stalls, and
                // at four hundred and seventy calls a token that is the cost
                // that decides whether crossing per call is affordable at all.
                const auto call_started = ds4_now();
                int status = colibri_gpu_upload(rt.gpu_input, input, input_bytes, 0);
                if (status == 0)
                    status = ds4_gpu_matvec(tensor.type, found->second, rt.gpu_input,
                                            rt.gpu_output, inputs, outputs, 0);
                if (status == 0) status = colibri_gpu_download(out, rt.gpu_output, output_bytes, 0);
                if (status == 0) status = colibri_gpu_sync();
                if (status != 0) throw std::runtime_error("a device matvec failed");
                ++rt.gpu_matvec_calls;
                rt.matvec_nanoseconds += ds4_now() - call_started;
                return;
            }
        }
    }
    const auto* packed = tensor_data(model, tensor);
    // Rows are independent, and a token walks about a thousand of these across
    // 43 layers, so the row loop is where the cores go. The test is on the work
    // rather than the row count: the hyper-connection mixer is 24 rows of 16384
    // weights, which a row-count threshold leaves serial despite being a
    // megabyte and a half of reading.
    if (static_cast<std::int64_t>(outputs) * inputs >= (1 << 16)) {
#pragma omp parallel for schedule(static) num_threads(ds4_thread_count())
        for (std::int32_t row = 0; row < outputs; ++row)
            out[row] = qwen_quant_dot(packed, tensor.type, input, inputs,
                                      static_cast<std::uint64_t>(row));
        return;
    }
    for (std::int32_t row = 0; row < outputs; ++row)
        out[row] = qwen_quant_dot(packed, tensor.type, input, inputs,
                                  static_cast<std::uint64_t>(row));
}

void ds4_expert_matvec(const ColibriV2Model& model, std::uint64_t index, std::int32_t expert,
                       const float* input, std::int32_t inputs, float* out, std::int32_t outputs) {
    const auto& tensor = model.tensors[index];
    const auto* packed = tensor_data(model, tensor);
    const auto base = static_cast<std::uint64_t>(expert) * outputs;
    // The routed experts are most of the arithmetic in a token: seven of them,
    // three matrices each, 43 layers.
#pragma omp parallel for schedule(static) num_threads(ds4_thread_count())
    for (std::int32_t row = 0; row < outputs; ++row)
        out[row] = qwen_quant_dot(packed, tensor.type, input, inputs, base + row);
}

// A resolved f32 tensor's data, for the norms and mixers the loop reads whole.
const float* ds4_f32(const ColibriV2Model& model, std::uint64_t index) {
    return reinterpret_cast<const float*>(tensor_data(model, model.tensors[index]));
}

// Close one compressed block: pool the rows it covers, normalize, rotate at the
// block's own position, and store it half precision.
//
// The main path and the indexer differ only in width and in which norm they
// read, so they share this. A 4:1 block pools the previous block's rows
// alongside its own, which is why the ring holds twice the ratio and why the
// halves of a row are read separately: the low half belongs to the block a row
// was written for, the high half to the one after it.
void ds4_close_block(const ColibriV2Deepseek4Runtime& rt, Deepseek4LayerState& layer,
                     std::uint32_t block, std::uint32_t dim, std::uint32_t width,
                     const std::vector<float>& source_values,
                     const std::vector<float>& source_scores,
                     std::uint64_t norm, std::vector<std::uint16_t>& target) {
    namespace ds4 = colibri::v2::deepseek4;
    const auto& model = *rt.model;
    const bool overlapped = layer.ratio == 4;
    const auto rows = layer.state_rows;
    std::vector<float> pooled_values(static_cast<std::size_t>(rows) * dim, 0.0f);
    std::vector<float> pooled_scores(pooled_values.size(),
                                     -std::numeric_limits<float>::infinity());
    for (std::uint32_t slot_index = 0; slot_index < layer.ratio; ++slot_index) {
        const auto current = block * layer.ratio + slot_index;
        const auto* cv = source_values.data() + static_cast<std::size_t>(current % rows) * width;
        const auto* cs = source_scores.data() + static_cast<std::size_t>(current % rows) * width;
        if (!overlapped) {
            std::copy_n(cv, dim, pooled_values.data() + static_cast<std::size_t>(slot_index) * dim);
            std::copy_n(cs, dim, pooled_scores.data() + static_cast<std::size_t>(slot_index) * dim);
            continue;
        }
        if (block > 0) {
            const auto previous = (block - 1) * layer.ratio + slot_index;
            const auto* pv = source_values.data() + static_cast<std::size_t>(previous % rows) * width;
            const auto* ps = source_scores.data() + static_cast<std::size_t>(previous % rows) * width;
            std::copy_n(pv, dim, pooled_values.data() + static_cast<std::size_t>(slot_index) * dim);
            std::copy_n(ps, dim, pooled_scores.data() + static_cast<std::size_t>(slot_index) * dim);
        }
        std::copy_n(cv + dim, dim,
                    pooled_values.data() + static_cast<std::size_t>(layer.ratio + slot_index) * dim);
        std::copy_n(cs + dim, dim,
                    pooled_scores.data() + static_cast<std::size_t>(layer.ratio + slot_index) * dim);
    }
    std::vector<float> pooled(dim, 0.0f);
    ds4::compress_block(pooled_values.data(), pooled_scores.data(), rows, dim, pooled.data());
    ds4::rms_norm(pooled.data(), dim, rt.epsilon, pooled.data());
    {
        const float* gain = ds4_f32(model, norm);
        for (std::uint32_t i = 0; i < dim; ++i) pooled[i] *= gain[i];
    }
    ds4::rope(pooled.data() + (dim - rt.rope_dim), rt.rope_dim,
              static_cast<std::int32_t>(block), rt.compress_base, rt.freq_scale, false,
              {1.0f, rt.attn_factor, rt.beta_fast, rt.beta_slow, rt.original_context});
    std::uint16_t* slot = target.data() + static_cast<std::size_t>(block) * dim;
    for (std::uint32_t i = 0; i < dim; ++i) slot[i] = ds4::half_bits(pooled[i]);
}

// Run one block over one position, reading and updating its cache.
void ds4_layer_attention_prepare(ColibriV2Deepseek4Runtime& rt, std::uint32_t index,
                                 std::uint32_t position, std::uint32_t token,
                                 const float* streams, float* out_streams,
                                 Deepseek4Scratch& sc) {
    namespace ds4 = colibri::v2::deepseek4;
    auto& layer = rt.layers[index];
    const auto& plan = layer.plan;
    const auto& model = *rt.model;
    const auto n_embd = rt.n_embd, hc = rt.hc, head_dim = rt.head_dim;
    const auto wide = static_cast<std::int32_t>(rt.heads) * head_dim;
    const auto attention_started = ds4_now();

    const bool compressed_rope = layer.ratio != 0;
    ds4::YarnParameters yarn{};
    float base = rt.freq_base, scale = 1.0f;
    if (compressed_rope) {
        base = rt.compress_base; scale = rt.freq_scale;
        yarn = {1.0f, rt.attn_factor, rt.beta_fast, rt.beta_slow, rt.original_context};
    }

    // Attention side of the hyper-connection.
    const auto hyper_started = ds4_now();
    ds4::hyper_connection_weights(
        streams, ds4_f32(model, plan.hc_attn_fn), ds4_f32(model, plan.hc_attn_scale),
        ds4_f32(model, plan.hc_attn_base), n_embd, hc, rt.sinkhorn_iterations,
        rt.epsilon, rt.epsilon, sc.pre.data(), sc.post.data(), sc.comb.data());
    ds4::hyper_connection_collapse(streams, sc.pre.data(), n_embd, hc, sc.collapsed.data());
    ds4::rms_norm(sc.collapsed.data(), n_embd, rt.epsilon, sc.hidden.data());
    {
        const float* gain = ds4_f32(model, plan.attn_norm);
        for (std::uint32_t i = 0; i < n_embd; ++i) sc.hidden[i] *= gain[i];
    }
    rt.hyper_nanoseconds += ds4_now() - hyper_started;

    // Everything a layer projects from the normalized hidden state, issued
    // together: the crossing is per round trip, not per weight, so six small
    // projections that shared an input were paying six stalls for one upload's
    // worth of data.
    bool query_chained=false;
    {
        Ds4Projection batch[6];
        std::size_t count = 0;
        static const bool query_chain_enabled=[] {
            const char* setting=std::getenv("COLIBRI_DS4_QUERY_CHAIN");
            return !(setting&&std::string(setting)=="off");
        }();
        const bool chain_query = query_chain_enabled && rt.gpu &&
            rt.resident.find(plan.q_a_norm) != rt.resident.end() &&
            rt.resident.find(plan.q_b) != rt.resident.end();
        batch[count++] = {plan.q_a, sc.low_rank.data(),
                          static_cast<std::int32_t>(rt.q_lora), !chain_query};
        batch[count++] = {plan.kv, sc.latent.data(), static_cast<std::int32_t>(head_dim)};
        if (layer.ratio) {
            const auto slot = position % layer.state_rows;
            batch[count++] = {plan.comp_kv,
                layer.state_values.data() + static_cast<std::size_t>(slot) * layer.state_width,
                static_cast<std::int32_t>(layer.state_width)};
            batch[count++] = {plan.comp_gate,
                layer.state_scores.data() + static_cast<std::size_t>(slot) * layer.state_width,
                static_cast<std::int32_t>(layer.state_width)};
        }
        if (layer.indexer_dim) {
            const auto slot = position % layer.state_rows;
            batch[count++] = {plan.indexer_comp_kv,
                layer.indexer_state_values.data() + static_cast<std::size_t>(slot) * layer.indexer_width,
                static_cast<std::int32_t>(layer.indexer_width)};
            batch[count++] = {plan.indexer_comp_gate,
                layer.indexer_state_scores.data() + static_cast<std::size_t>(slot) * layer.indexer_width,
                static_cast<std::int32_t>(layer.indexer_width)};
        }
        const bool device_batch=ds4_matvec_batch(rt, batch, count, sc.hidden.data(), n_embd);
        if(chain_query&&device_batch){
            // q_a is the first batched output. Keep it resident, normalize it
            // into the input workspace, and feed q_b without a download/upload
            // pair. The final query alone returns to the CPU attention path.
            int status=colibri_gpu_rms_norm(rt.gpu_batch,rt.resident[plan.q_a_norm],
                rt.gpu_input,static_cast<std::int32_t>(rt.q_lora),rt.epsilon,0);
            const auto& q_b=model.tensors[plan.q_b];
            if(status==0)status=ds4_gpu_matvec(q_b.type,rt.resident[plan.q_b],
                rt.gpu_input,rt.gpu_output,static_cast<std::int32_t>(rt.q_lora),wide,0);
            if(status==0)status=colibri_gpu_download(sc.query.data(),rt.gpu_output,
                static_cast<std::uint64_t>(wide)*sizeof(float),0);
            if(status==0)status=colibri_gpu_sync();
            if(status!=0)throw std::runtime_error("the resident query chain failed");
            ++rt.gpu_matvec_calls;
            query_chained=true;
        }
    }

    // Query: low-rank path, per-head norm without gain, then rotate.
    if(!query_chained){
        ds4::rms_norm(sc.low_rank.data(), rt.q_lora, rt.epsilon, sc.low_rank.data());
        const float* gain = ds4_f32(model, plan.q_a_norm);
        for (std::uint32_t i = 0; i < rt.q_lora; ++i) sc.low_rank[i] *= gain[i];
        ds4_matvec(rt, plan.q_b, sc.low_rank.data(), rt.q_lora, sc.query.data(), wide);
    }
    for (std::uint32_t head = 0; head < rt.heads; ++head) {
        float* row = sc.query.data() + static_cast<std::size_t>(head) * head_dim;
        ds4::rms_norm(row, head_dim, rt.epsilon, row);
        ds4::rope(row + (head_dim - rt.rope_dim), rt.rope_dim, static_cast<std::int32_t>(position),
                  base, scale, false, yarn);
    }

    // Key and value are the same latent, stored half precision. Its projection
    // went out with the batch above.
    ds4::rms_norm(sc.latent.data(), head_dim, rt.epsilon, sc.latent.data());
    {
        const float* gain = ds4_f32(model, plan.kv_a_norm);
        for (std::uint32_t i = 0; i < head_dim; ++i) sc.latent[i] *= gain[i];
    }
    ds4::rope(sc.latent.data() + (head_dim - rt.rope_dim), rt.rope_dim,
              static_cast<std::int32_t>(position), base, scale, false, yarn);
    {
        std::uint16_t* slot = layer.latents.data() +
            static_cast<std::size_t>(position % layer.window) * head_dim;
        for (std::uint32_t i = 0; i < head_dim; ++i) slot[i] = ds4::half_bits(sc.latent[i]);
    }

    // The indexer's compressor, kept on every 4:1 layer whatever the length.
    //
    // The cache has to be built from the first token even though nothing reads
    // it until there are more blocks than the indexer may select: a block's
    // rows are projected from a hidden state that is gone by the next step, so
    // there is no way to fill this in later.
    if (layer.indexer_dim) {
        const auto width = layer.indexer_width;
        const auto slot = position % layer.state_rows;
        float* scores = layer.indexer_state_scores.data() + static_cast<std::size_t>(slot) * width;
        // The two projections were issued with the rest of the layer's above.
        const float* ape = ds4_f32(model, plan.indexer_comp_ape) +
            static_cast<std::size_t>(position % layer.ratio) * width;
        for (std::uint32_t i = 0; i < width; ++i) scores[i] += ape[i];
        if ((position + 1) % layer.ratio == 0 && layer.blocks < layer.block_capacity) {
            layer.indexer_compressed.resize(
                (static_cast<std::size_t>(layer.blocks)+1)*layer.indexer_dim);
            ds4_close_block(rt, layer, layer.blocks, layer.indexer_dim, width,
                            layer.indexer_state_values, layer.indexer_state_scores,
                            plan.indexer_comp_norm, layer.indexer_compressed);
        }
    }

    // Compressor state, closing a block when it fills.
    if (layer.ratio) {
        const auto width = layer.state_width;
        const auto slot = position % layer.state_rows;
        float* scores = layer.state_scores.data() + static_cast<std::size_t>(slot) * width;
        const float* ape = ds4_f32(model, plan.comp_ape) +
            static_cast<std::size_t>(position % layer.ratio) * width;
        for (std::uint32_t i = 0; i < width; ++i) scores[i] += ape[i];
        if ((position + 1) % layer.ratio == 0 && layer.blocks < layer.block_capacity) {
            layer.compressed.resize(
                (static_cast<std::size_t>(layer.blocks)+1)*head_dim);
            ds4_close_block(rt, layer, layer.blocks, head_dim, width,
                            layer.state_values, layer.state_scores,
                            plan.comp_norm, layer.compressed);
            ++layer.blocks;
        }
    }

    // The lightning indexer: which of the compressed blocks this position is
    // allowed to see.
    //
    // Only run once there are more visible blocks than it may select. Below
    // that the top-k is everything and the mask it would build is the mask
    // already in force, so skipping is exact rather than an approximation --
    // and it saves reading the query projection, which is the expensive half.
    std::uint32_t visible_blocks = 0;
    for (std::uint32_t block = 0; block < layer.blocks; ++block) {
        if (block * layer.ratio + layer.ratio - 1 > position) break;
        ++visible_blocks;
    }
    const bool selecting = layer.indexer_dim && rt.indexer_top_k &&
                           visible_blocks > rt.indexer_top_k;
    if (selecting) {
        const auto indexer_dim = layer.indexer_dim;
        ds4_matvec(rt, plan.indexer_q_b, sc.low_rank.data(),
                   static_cast<std::int32_t>(rt.q_lora), sc.indexer_query.data(),
                   static_cast<std::int32_t>(rt.indexer_heads * indexer_dim));
        for (std::uint32_t head = 0; head < rt.indexer_heads; ++head) {
            float* row = sc.indexer_query.data() + static_cast<std::size_t>(head) * indexer_dim;
            ds4::rope(row + (indexer_dim - rt.rope_dim), rt.rope_dim,
                      static_cast<std::int32_t>(position), base, scale, false, yarn);
        }
        ds4_matvec(rt, plan.indexer_proj, sc.hidden.data(), n_embd,
                   sc.indexer_weights.data(), static_cast<std::int32_t>(rt.indexer_heads));
        // The reference scales the weights rather than the scores; with the
        // rectifier in between the two are not the same thing.
        const float scale_weights =
            1.0f / std::sqrt(static_cast<float>(indexer_dim * rt.indexer_heads));
        for (std::uint32_t head = 0; head < rt.indexer_heads; ++head)
            sc.indexer_weights[head] *= scale_weights;
        // Score directly from the half cache. Expanding every candidate into a
        // context-sized float scratch buffer added another ~128 MiB at 1M.
#pragma omp parallel for schedule(static) num_threads(ds4_thread_count())
        for (std::int64_t block = 0; block < static_cast<std::int64_t>(visible_blocks); ++block) {
            const std::uint16_t* key = layer.indexer_compressed.data() +
                static_cast<std::size_t>(block) * indexer_dim;
            double total = 0.0;
            for (std::uint32_t head = 0; head < rt.indexer_heads; ++head) {
                const float* query = sc.indexer_query.data() +
                    static_cast<std::size_t>(head) * indexer_dim;
                double dot = 0.0;
                for (std::uint32_t i = 0; i < indexer_dim; ++i)
                    dot += static_cast<double>(query[i]) * ds4::half_value(key[i]);
                if (dot > 0.0) total += dot * sc.indexer_weights[head];
            }
            sc.indexer_scores[static_cast<std::size_t>(block)] = static_cast<float>(total);
        }
        ds4::top_k_select(sc.indexer_scores.data(), visible_blocks, rt.indexer_top_k,
                          sc.indexer_keep.data());
        rt.indexer_selections += rt.indexer_top_k;
        rt.indexer_candidates += visible_blocks;
    }

    // Gather the keys this position may see, raw window then compressed blocks.
    const auto core_started = ds4_now();
    std::uint32_t keys = 0;
    const std::uint32_t first = position + 1 > layer.window ? position + 1 - layer.window : 0;
    for (std::uint32_t p = first; p <= position; ++p) {
        const std::uint16_t* slot = layer.latents.data() +
            static_cast<std::size_t>(p % layer.window) * head_dim;
        float* target = sc.keys.data() + static_cast<std::size_t>(keys) * head_dim;
        for (std::uint32_t i = 0; i < head_dim; ++i) target[i] = ds4::half_value(slot[i]);
        ++keys;
    }
    for (std::uint32_t block = 0; block < visible_blocks; ++block) {
        if (selecting && !sc.indexer_keep[block]) continue;
        const std::uint16_t* slot = layer.compressed.data() +
            static_cast<std::size_t>(block) * head_dim;
        float* target = sc.keys.data() + static_cast<std::size_t>(keys) * head_dim;
        for (std::uint32_t i = 0; i < head_dim; ++i) target[i] = ds4::half_value(slot[i]);
        ++keys;
    }

    ds4::attention_with_sinks(sc.query.data(), sc.keys.data(), ds4_f32(model, plan.sinks),
                              nullptr, rt.heads, head_dim, keys,
                              1.0f / std::sqrt(static_cast<float>(head_dim)), sc.attn.data());
    rt.attention_core_nanoseconds += ds4_now() - core_started;

    // Undo the rotation, then the grouped output projection.
    std::copy(sc.attn.begin(), sc.attn.end(), sc.derope.begin());
    for (std::uint32_t head = 0; head < rt.heads; ++head)
        ds4::rope(sc.derope.data() + static_cast<std::size_t>(head) * head_dim + (head_dim - rt.rope_dim),
                  rt.rope_dim, static_cast<std::int32_t>(position), base, scale, true, yarn);
    {
        const auto inputs = static_cast<std::int32_t>(wide / rt.groups);
        const auto& tensor = model.tensors[plan.output_a];
        const auto resident = rt.gpu ? rt.resident.find(plan.output_a) : rt.resident.end();
        const auto grouped_input_bytes = static_cast<std::uint64_t>(wide) * sizeof(float);
        const auto grouped_output_bytes =
            static_cast<std::uint64_t>(rt.groups * rt.lora_rank) * sizeof(float);
        if (rt.gpu && resident != rt.resident.end() &&
            grouped_input_bytes <= rt.gpu_input_capacity &&
            grouped_output_bytes <= rt.gpu_output_capacity) {
            const auto rows = static_cast<std::int32_t>(rt.groups * rt.lora_rank);
            const auto group_rows = static_cast<std::int32_t>(rt.lora_rank);
            const auto input_bytes = grouped_input_bytes;
            const auto output_bytes = grouped_output_bytes;
            std::uint64_t weights = resident->second;
            std::uint64_t in = rt.gpu_input, out_buffer = rt.gpu_output;
            std::int32_t inputs_arg = inputs, rows_arg = rows, group_arg = group_rows;
            void* arguments[] = {&weights, &in, &out_buffer, &inputs_arg, &rows_arg, &group_arg};
            int status = colibri_gpu_upload(rt.gpu_input, sc.derope.data(), input_bytes, 0);
            if (status == 0)
                status = colibri_gpu_launch_named("ds4_q8_grouped_matvec",
                    static_cast<std::uint32_t>((rows + 7) / 8), 1, 256, 0, 0, arguments);
            if (status == 0)
                status = colibri_gpu_download(sc.grouped.data(), rt.gpu_output, output_bytes, 0);
            if (status == 0) status = colibri_gpu_sync();
            if (status != 0) throw std::runtime_error("the grouped device matvec failed");
            ++rt.gpu_matvec_calls;
            goto grouped_done;
        }
        {
        const auto* packed = tensor_data(model, tensor);
        // Every group's rows are independent, so this is one flat loop over
        // them rather than two nested ones -- eight groups would otherwise
        // fork eight times, and at 34 MiB a layer this is the largest single
        // read in the attention half.
        const auto rows = static_cast<std::int32_t>(rt.groups * rt.lora_rank);
#pragma omp parallel for schedule(static) num_threads(ds4_thread_count())
        for (std::int32_t row = 0; row < rows; ++row) {
            const auto group = static_cast<std::size_t>(row) / rt.lora_rank;
            sc.grouped[static_cast<std::size_t>(row)] = qwen_quant_dot(
                packed, tensor.type, sc.derope.data() + group * inputs, inputs,
                static_cast<std::uint64_t>(row));
        }
        }
        grouped_done:;
    }
    ds4_matvec(rt, plan.output_b, sc.grouped.data(),
               static_cast<std::int32_t>(rt.groups * rt.lora_rank), sc.block_out.data(), n_embd);

    ds4::hyper_connection_combine(sc.block_out.data(), streams, sc.post.data(), sc.comb.data(),
                                  n_embd, hc, out_streams);
    rt.attention_nanoseconds += ds4_now() - attention_started;

    // Feed-forward side.
    const auto hyper_ffn_started = ds4_now();
    ds4::hyper_connection_weights(
        out_streams, ds4_f32(model, plan.hc_ffn_fn), ds4_f32(model, plan.hc_ffn_scale),
        ds4_f32(model, plan.hc_ffn_base), n_embd, hc, rt.sinkhorn_iterations,
        rt.epsilon, rt.epsilon, sc.pre.data(), sc.post.data(), sc.comb.data());
    ds4::hyper_connection_collapse(out_streams, sc.pre.data(), n_embd, hc, sc.collapsed.data());
    ds4::rms_norm(sc.collapsed.data(), n_embd, rt.epsilon, sc.hidden.data());
    {
        const float* gain = ds4_f32(model, plan.ffn_norm);
        for (std::uint32_t i = 0; i < n_embd; ++i) sc.hidden[i] *= gain[i];
    }

    rt.hyper_nanoseconds += ds4_now() - hyper_ffn_started;
    std::fill(sc.router.begin(),sc.router.end(),0.0f);
    ds4_matvec(rt, plan.gate_inp, sc.hidden.data(), n_embd, sc.router.data(), rt.experts);
    const bool hashed = index < rt.hash_layers;
    if (hashed) {
        const auto* table = reinterpret_cast<const std::int32_t*>(
            tensor_data(model, model.tensors[plan.tid2eid]));
        std::copy_n(table + static_cast<std::size_t>(token) * rt.experts_used,
                    rt.experts_used, sc.experts.begin());
    }
    ds4::moe_router(sc.router.data(), hashed ? nullptr : ds4_f32(model, plan.exp_probs_b),
                    rt.experts, rt.experts_used, rt.weight_scale, 1e-20f, !hashed,
                    sc.experts.data(), sc.weights.data());

}

void ds4_layer_ffn_complete(ColibriV2Deepseek4Runtime& rt, std::uint32_t index,
                            float* out_streams, Deepseek4Scratch& sc,
                            bool routed_ready=false) {
    namespace ds4 = colibri::v2::deepseek4;
    const auto& layer = rt.layers[index];
    const auto& plan = layer.plan;
    const auto& model = *rt.model;
    const auto n_embd = rt.n_embd, hc = rt.hc;

    if(!routed_ready){
    std::fill(sc.moe.begin(), sc.moe.end(), 0.0f);
    // Ask for every routed expert's weights before touching any of them.
    //
    // The six are known the moment the router runs, but the loop below reads
    // them one matrix at a time, so a miss stalls on each in turn. Hinting all
    // eighteen ranges up front lets the kernel fetch the later ones while the
    // first is being multiplied. It costs nothing when they are already
    // resident, which measurement says they usually are -- this is for the
    // regime where they are not.
    if (rt.expert_prefetch) {
        for (std::uint32_t slot = 0; slot < rt.experts_used; ++slot) {
            for (const auto matrix : {plan.gate_exps, plan.up_exps, plan.down_exps}) {
                const auto& tensor = model.tensors[matrix];
                const auto span = tensor.size / rt.experts;
                ds4_prefetch(tensor_data(model, tensor) +
                                 static_cast<std::uint64_t>(sc.experts[slot]) * span,
                             span);
                rt.expert_prefetch_bytes += span;
            }
        }
    }
    const auto experts_started = ds4_now();
    ds4_cached_experts(rt,index,plan,sc);
    static const bool expert_batch_enabled=[] {
        const char* setting=std::getenv("COLIBRI_DS4_EXPERT_BATCH");
        return !(setting&&std::string(setting)=="off");
    }();
    // One team spans every selected expert instead of entering OpenMP eighteen
    // times per layer. The row arithmetic and later weighted accumulation keep
    // their original order, so this is bit-identical to the serial expert loop.
    if(expert_batch_enabled){
        const auto& gate_tensor=model.tensors[plan.gate_exps];
        const auto& up_tensor=model.tensors[plan.up_exps];
        const auto& down_tensor=model.tensors[plan.down_exps];
        const auto* gate_packed=tensor_data(model,gate_tensor);
        const auto* up_packed=tensor_data(model,up_tensor);
        const auto* down_packed=tensor_data(model,down_tensor);
#pragma omp parallel for schedule(static) num_threads(ds4_thread_count())
        for(std::int64_t item=0;item<static_cast<std::int64_t>(rt.experts_used)*rt.expert_ffn;++item){
            ds4_pin_current_thread();
            const auto slot=static_cast<std::uint32_t>(item/rt.expert_ffn);
            if(sc.gpu_experts[slot])continue;
            const auto row=static_cast<std::uint32_t>(item%rt.expert_ffn);
            const auto absolute=static_cast<std::uint64_t>(sc.experts[slot])*rt.expert_ffn+row;
            sc.routed_gate[static_cast<std::size_t>(item)]=qwen_quant_dot(
                gate_packed,gate_tensor.type,sc.hidden.data(),n_embd,absolute);
            sc.routed_up[static_cast<std::size_t>(item)]=qwen_quant_dot(
                up_packed,up_tensor.type,sc.hidden.data(),n_embd,absolute);
        }
#pragma omp parallel for schedule(static) num_threads(ds4_thread_count())
        for(std::int64_t item=0;item<static_cast<std::int64_t>(rt.experts_used)*rt.expert_ffn;++item){
            ds4_pin_current_thread();
            const auto slot=static_cast<std::uint32_t>(item/rt.expert_ffn);
            if(sc.gpu_experts[slot])continue;
            const float gate=std::clamp(sc.routed_gate[static_cast<std::size_t>(item)],-rt.clamp,rt.clamp);
            const float up=std::clamp(sc.routed_up[static_cast<std::size_t>(item)],-rt.clamp,rt.clamp);
            sc.routed_activated[static_cast<std::size_t>(item)]=gate/(1.0f+std::exp(-gate))*up;
        }
#pragma omp parallel for schedule(static) num_threads(ds4_thread_count())
        for(std::int64_t item=0;item<static_cast<std::int64_t>(rt.experts_used)*n_embd;++item){
            ds4_pin_current_thread();
            const auto slot=static_cast<std::uint32_t>(item/n_embd);
            if(sc.gpu_experts[slot])continue;
            const auto row=static_cast<std::uint32_t>(item%n_embd);
            const auto absolute=static_cast<std::uint64_t>(sc.experts[slot])*n_embd+row;
            sc.routed_out[static_cast<std::size_t>(item)]=qwen_quant_dot(
                down_packed,down_tensor.type,
                sc.routed_activated.data()+static_cast<std::size_t>(slot)*rt.expert_ffn,
                rt.expert_ffn,absolute);
        }
        for (std::uint32_t slot = 0; slot < rt.experts_used; ++slot) {
            if(sc.gpu_experts[slot])continue;
            const float weight = sc.weights[slot];
            const float* expert_out=sc.routed_out.data()+static_cast<std::size_t>(slot)*n_embd;
            for (std::uint32_t i = 0; i < n_embd; ++i) sc.moe[i] += expert_out[i] * weight;
        }
    }else{
        for (std::uint32_t slot = 0; slot < rt.experts_used; ++slot) {
            if(sc.gpu_experts[slot]){
                continue;
            }
            const auto expert = sc.experts[slot];
            ds4_expert_matvec(model, plan.gate_exps, expert, sc.hidden.data(), n_embd,
                              sc.gate.data(), rt.expert_ffn);
            ds4_expert_matvec(model, plan.up_exps, expert, sc.hidden.data(), n_embd,
                              sc.up.data(), rt.expert_ffn);
            ds4::clamped_swiglu(sc.gate.data(), sc.up.data(), rt.expert_ffn, rt.clamp,
                                sc.activated.data());
            ds4_expert_matvec(model, plan.down_exps, expert, sc.activated.data(), rt.expert_ffn,
                              sc.expert_out.data(), n_embd);
            const float weight = sc.weights[slot];
            for (std::uint32_t i = 0; i < n_embd; ++i) sc.moe[i] += sc.expert_out[i] * weight;
        }
    }
    for (std::uint32_t slot = 0; slot < rt.experts_used; ++slot) {
        // One expert's share of the three matrices, which is what a page-in
        // would have to fetch.
        for (const auto matrix : {plan.gate_exps, plan.up_exps, plan.down_exps})
            rt.routed_expert_bytes += model.tensors[matrix].size / rt.experts;
    }
    rt.routed_expert_nanoseconds += ds4_now() - experts_started;
    }
    const auto shared_started = ds4_now();
    ds4_matvec(rt, plan.gate_shexp, sc.hidden.data(), n_embd, sc.gate.data(), rt.expert_ffn);
    ds4_matvec(rt, plan.up_shexp, sc.hidden.data(), n_embd, sc.up.data(), rt.expert_ffn);
    ds4::clamped_swiglu(sc.gate.data(), sc.up.data(), rt.expert_ffn, rt.clamp, sc.activated.data());
    ds4_matvec(rt, plan.down_shexp, sc.activated.data(), rt.expert_ffn,
               sc.expert_out.data(), n_embd);
    for (std::uint32_t i = 0; i < n_embd; ++i) sc.moe[i] += sc.expert_out[i];
    rt.shared_expert_nanoseconds += ds4_now() - shared_started;

    ds4::hyper_connection_combine(sc.moe.data(), out_streams, sc.post.data(), sc.comb.data(),
                                  n_embd, hc, sc.after.data());
    std::copy(sc.after.begin(), sc.after.end(), out_streams);
}

void ds4_layer_forward(ColibriV2Deepseek4Runtime& rt, std::uint32_t index,
                       std::uint32_t position, std::uint32_t token,
                       const float* streams, float* out_streams) {
    auto& sc=rt.scratch;
    ds4_layer_attention_prepare(rt,index,position,token,streams,out_streams,sc);
    ds4_layer_ffn_complete(rt,index,out_streams,sc);
}

}  // namespace

int colibri_v2_deepseek4_runtime_info(
    const ColibriV2Deepseek4Runtime* runtime, ColibriV2Deepseek4Info* out
){return guarded([&]{
    if(!runtime||!out)throw std::runtime_error("invalid runtime info request");
    std::memset(out,0,sizeof(*out));
    out->layers=static_cast<std::uint32_t>(runtime->layers.size());
    out->context_limit=runtime->context_limit;
    out->state_bytes=runtime->state_bytes();
    out->positions=runtime->layers.empty()?0:runtime->layers.front().positions;
    for(const auto& layer:runtime->layers)out->resolved_tensors+=layer.plan.resolved();
    for(const auto& layer:runtime->layers){
        if(layer.ratio==4)++out->csa_layers;
        else if(layer.ratio)++out->hca_layers;
        else ++out->window_layers;
    }
    out->forward_calls=runtime->forward_calls;
    out->forward_nanoseconds=runtime->forward_nanoseconds;
    out->routed_expert_nanoseconds=runtime->routed_expert_nanoseconds;
    out->shared_expert_nanoseconds=runtime->shared_expert_nanoseconds;
    out->attention_nanoseconds=runtime->attention_nanoseconds;
    out->attention_core_nanoseconds=runtime->attention_core_nanoseconds;
    out->head_nanoseconds=runtime->head_nanoseconds;
    out->routed_expert_bytes=runtime->routed_expert_bytes;
    out->expert_prefetch_bytes=runtime->expert_prefetch_bytes;
    out->gpu_weight_bytes=runtime->gpu_weight_bytes;
    out->gpu_matvec_calls=runtime->gpu_matvec_calls;
    out->gpu_batches=runtime->gpu_batches;
    out->hyper_nanoseconds=runtime->hyper_nanoseconds;
    out->matvec_nanoseconds=runtime->matvec_nanoseconds;
    out->prefill_calls=runtime->prefill_calls;
    out->prefill_tokens=runtime->prefill_tokens;
    out->prefill_nanoseconds=runtime->prefill_nanoseconds;
    const auto* cache=runtime->gpu_cache_owner?runtime->gpu_cache_owner:runtime;
    out->expert_cache_bytes=cache->expert_cache_bytes;
    out->expert_cache_slots=cache->expert_slots.size();
    out->expert_cache_hits=cache->expert_cache_hits;
    out->expert_cache_misses=cache->expert_cache_misses;
    out->expert_cache_evictions=cache->expert_cache_evictions;
    out->indexer_selections=runtime->indexer_selections;
    out->indexer_candidates=runtime->indexer_candidates;
    return 0;});}

// Run one token through every block and the head.
//
// Positions advance with the call, so feeding a prompt is repeated calls and
// decoding is the same call again -- prefill and decode share one path, which
// is what keeps them from disagreeing. `logits` may be null for prompt tokens
// whose distribution nobody wants.
int colibri_v2_deepseek4_forward(
    ColibriV2Deepseek4Runtime* runtime, uint32_t token, float* logits
){return guarded([&]{
    if(!runtime)throw std::runtime_error("invalid runtime");
    namespace ds4=colibri::v2::deepseek4;
    auto& rt=*runtime;
    const auto& model=*rt.model;
    auto& sc=rt.scratch;
    if(token>=rt.vocabulary)throw std::runtime_error("token id is outside the vocabulary");
    const auto forward_started=ds4_now();
    const auto position=rt.layers.empty()?0u:rt.layers.front().positions;
    if(position>=rt.context_limit)throw std::runtime_error("sequence is past the context limit");

    // Every stream starts as a copy of the embedding.
    {
        const auto& tensor=model.tensors[rt.token_embd];
        if(colibri_v2_qwen_embedding(&model,token,sc.embedding.data(),
               static_cast<std::int32_t>(rt.n_embd))!=0)
            throw std::runtime_error("cannot read the token embedding");
        for(std::uint32_t stream=0;stream<rt.hc;++stream)
            std::copy(sc.embedding.begin(),sc.embedding.end(),
                      sc.streams.begin()+static_cast<std::size_t>(stream)*rt.n_embd);
    }

    for(std::uint32_t index=0;index<rt.layers.size();++index){
        const auto capture=std::find(rt.capture_layers.begin(),rt.capture_layers.end(),index);
        if(capture!=rt.capture_layers.end()){
            const auto slot=static_cast<std::size_t>(capture-rt.capture_layers.begin());
            auto* destination=rt.captured.data()+slot*rt.n_embd;
            const float inverse=1.0f/static_cast<float>(rt.hc);
            for(std::uint32_t column=0;column<rt.n_embd;++column){
                float sum=0.0f;
                for(std::uint32_t stream=0;stream<rt.hc;++stream)
                    sum+=sc.streams[static_cast<std::size_t>(stream)*rt.n_embd+column];
                destination[column]=sum*inverse;
            }
        }
        ds4_layer_forward(rt,index,position,token,sc.streams.data(),sc.next_streams.data());
        std::swap(sc.streams,sc.next_streams);
        ++rt.layers[index].positions;
    }
    const auto final_capture=std::find(
        rt.capture_layers.begin(),rt.capture_layers.end(),
        static_cast<std::uint32_t>(rt.layers.size()));
    if(final_capture!=rt.capture_layers.end()){
        const auto slot=static_cast<std::size_t>(final_capture-rt.capture_layers.begin());
        auto* destination=rt.captured.data()+slot*rt.n_embd;
        const float inverse=1.0f/static_cast<float>(rt.hc);
        for(std::uint32_t column=0;column<rt.n_embd;++column){
            float sum=0.0f;
            for(std::uint32_t stream=0;stream<rt.hc;++stream)
                sum+=sc.streams[static_cast<std::size_t>(stream)*rt.n_embd+column];
            destination[column]=sum*inverse;
        }
    }

    if(logits){
        const auto head_started=ds4_now();
        ds4::hyper_connection_head(sc.streams.data(),ds4_f32(model,rt.head_fn),
            ds4_f32(model,rt.head_scale),ds4_f32(model,rt.head_base),
            rt.n_embd,rt.hc,rt.epsilon,rt.epsilon,sc.head_pre.data(),sc.collapsed.data());
        ds4::rms_norm(sc.collapsed.data(),rt.n_embd,rt.epsilon,sc.hidden.data());
        const float* gain=ds4_f32(model,rt.output_norm);
        for(std::uint32_t i=0;i<rt.n_embd;++i)sc.hidden[i]*=gain[i];
        ds4_matvec(rt,rt.output,sc.hidden.data(),rt.n_embd,logits,
                   static_cast<std::int32_t>(rt.vocabulary));
        rt.head_nanoseconds+=ds4_now()-head_started;
    }
    ++rt.forward_calls;
    rt.forward_nanoseconds+=ds4_now()-forward_started;
    return 0;});}

int colibri_v2_deepseek4_capture_layers(
    ColibriV2Deepseek4Runtime* runtime,const uint32_t* layers,uint32_t count
){return guarded([&]{
    if(!runtime||(!layers&&count))throw std::runtime_error("invalid capture-layer arguments");
    std::vector<std::uint32_t> selected;
    if(count)selected.assign(layers,layers+count);
    std::unordered_set<std::uint32_t> unique;
    for(const auto layer:selected){
        if(!unique.insert(layer).second)throw std::runtime_error("capture layers must be unique");
        if(layer>runtime->layers.size())throw std::runtime_error("capture layer is outside the target model");
    }
    runtime->capture_layers=std::move(selected);
    runtime->captured.assign(static_cast<std::size_t>(count)*runtime->n_embd,0.0f);
    return 0;});}

int colibri_v2_deepseek4_captured(
    const ColibriV2Deepseek4Runtime* runtime,float* output,uint64_t elements
){return guarded([&]{
    if(!runtime||(!output&&elements))throw std::runtime_error("invalid captured-feature arguments");
    if(elements<runtime->captured.size())throw std::runtime_error("captured-feature output is too small");
    std::copy(runtime->captured.begin(),runtime->captured.end(),output);
    return static_cast<int>(runtime->capture_layers.size());});}

int colibri_v2_deepseek4_lm_head(ColibriV2Deepseek4Runtime*r,const float*hidden,
    uint32_t rows,float*logits,uint64_t elements){return guarded([&]{
    if(!r||!hidden||!logits||!rows||r->output==Deepseek4LayerPlan::kAbsent)
        throw std::runtime_error("invalid DeepSeek LM-head arguments");
    const auto needed=static_cast<std::uint64_t>(rows)*r->vocabulary;
    if(elements<needed)throw std::runtime_error("DeepSeek LM-head output is too small");
    for(std::uint32_t row=0;row<rows;++row)
        ds4_matvec(*r,r->output,hidden+static_cast<std::size_t>(row)*r->n_embd,r->n_embd,
            logits+static_cast<std::size_t>(row)*r->vocabulary,r->vocabulary);
    return 0;});}

static int ds4_prefill_impl(
    ColibriV2Deepseek4Runtime* runtime,const uint32_t* tokens,uint32_t count,float* all_logits
){return guarded([&]{
    if(!runtime||(!tokens&&count))throw std::runtime_error("invalid prefill arguments");
    if(!count)return 0;
    auto& rt=*runtime;
    const auto& model=*rt.model;
    std::uint32_t row_limit=4;
    if(const char* setting=std::getenv("COLIBRI_DS4_PREFILL_ROWS"))
        row_limit=std::max(1u,static_cast<std::uint32_t>(std::strtoul(setting,nullptr,10)));
    row_limit=std::min({row_limit,count,8u});
    const auto first_position=rt.layers.empty()?0u:rt.layers.front().positions;
    if(static_cast<std::uint64_t>(first_position)+count>rt.context_limit)
        throw std::runtime_error("prefill exceeds the DeepSeek context limit");
    for(std::uint32_t index=0;index<count;++index)
        if(tokens[index]>=rt.vocabulary)
            throw std::runtime_error("prefill token id is outside the vocabulary");
    const auto started=ds4_now();
    for(std::uint32_t chunk=0;chunk<count;chunk+=row_limit){
        const auto rows=std::min(row_limit,count-chunk);
        std::vector<Deepseek4Scratch> scratch(rows,rt.scratch);
        for(std::uint32_t row=0;row<rows;++row){
            auto& sc=scratch[row];
            if(colibri_v2_qwen_embedding(&model,tokens[chunk+row],sc.embedding.data(),
                   static_cast<std::int32_t>(rt.n_embd))!=0)
                throw std::runtime_error("cannot read a prefill token embedding");
            for(std::uint32_t stream=0;stream<rt.hc;++stream)
                std::copy(sc.embedding.begin(),sc.embedding.end(),
                    sc.streams.begin()+static_cast<std::size_t>(stream)*rt.n_embd);
        }
        for(std::uint32_t layer=0;layer<rt.layers.size();++layer){
            const auto position=rt.layers[layer].positions;
            for(std::uint32_t row=0;row<rows;++row){
                auto& sc=scratch[row];
                ds4_layer_attention_prepare(rt,layer,position+row,tokens[chunk+row],
                    sc.streams.data(),sc.next_streams.data(),sc);
            }
            const auto experts_started=ds4_now();
            ds4_cpu_moe_rows(rt,layer,scratch);
            rt.routed_expert_nanoseconds+=ds4_now()-experts_started;
            for(std::uint32_t row=0;row<rows;++row)
                for(std::uint32_t slot=0;slot<rt.experts_used;++slot)
                    for(const auto matrix:{rt.layers[layer].plan.gate_exps,
                                           rt.layers[layer].plan.up_exps,
                                           rt.layers[layer].plan.down_exps})
                        rt.routed_expert_bytes+=model.tensors[matrix].size/rt.experts;
            for(std::uint32_t row=0;row<rows;++row){
                auto& sc=scratch[row];
                ds4_layer_ffn_complete(rt,layer,sc.next_streams.data(),sc,true);
                std::swap(sc.streams,sc.next_streams);
            }
            rt.layers[layer].positions+=rows;
        }
        if(all_logits){
            namespace ds4=colibri::v2::deepseek4;
            for(std::uint32_t row=0;row<rows;++row){
                auto&sc=scratch[row];
                ds4::hyper_connection_head(sc.streams.data(),ds4_f32(model,rt.head_fn),
                    ds4_f32(model,rt.head_scale),ds4_f32(model,rt.head_base),rt.n_embd,rt.hc,
                    rt.epsilon,rt.epsilon,sc.head_pre.data(),sc.collapsed.data());
                ds4::rms_norm(sc.collapsed.data(),rt.n_embd,rt.epsilon,sc.hidden.data());
                const auto*gain=ds4_f32(model,rt.output_norm);
                for(std::uint32_t i=0;i<rt.n_embd;++i)sc.hidden[i]*=gain[i];
                ds4_matvec(rt,rt.output,sc.hidden.data(),rt.n_embd,
                    all_logits+static_cast<std::size_t>(chunk+row)*rt.vocabulary,rt.vocabulary);
            }
        }
        rt.forward_calls+=rows;
    }
    const auto elapsed=ds4_now()-started;
    ++runtime->prefill_calls;
    runtime->prefill_tokens+=count;
    runtime->prefill_nanoseconds+=elapsed;
    runtime->forward_nanoseconds+=elapsed;
    return 0;});}

int colibri_v2_deepseek4_prefill(ColibriV2Deepseek4Runtime*r,const uint32_t*t,uint32_t n){
    return ds4_prefill_impl(r,t,n,nullptr);
}
int colibri_v2_deepseek4_forward_batch(ColibriV2Deepseek4Runtime*r,const uint32_t*t,
    uint32_t n,float*logits,uint64_t elements){
    if(!r||!logits||elements<static_cast<std::uint64_t>(n)*r->vocabulary){
        error="invalid DeepSeek forward-batch output";return -1;
    }
    return ds4_prefill_impl(r,t,n,logits);
}
int colibri_v2_deepseek4_snapshot(const ColibriV2Deepseek4Runtime*r,ColibriV2Deepseek4Snapshot**out){return guarded([&]{
    if(!r||!out)throw std::runtime_error("invalid DeepSeek snapshot arguments");
    auto snapshot=std::make_unique<ColibriV2Deepseek4Snapshot>();snapshot->layers=r->layers;
    snapshot->captured=r->captured;*out=snapshot.release();return 0;});}
int colibri_v2_deepseek4_restore(ColibriV2Deepseek4Runtime*r,const ColibriV2Deepseek4Snapshot*s){return guarded([&]{
    if(!r||!s||s->layers.size()!=r->layers.size())throw std::runtime_error("invalid DeepSeek snapshot");
    r->layers=s->layers;r->captured=s->captured;return 0;});}
void colibri_v2_deepseek4_snapshot_free(ColibriV2Deepseek4Snapshot*s){try{delete s;}catch(...){}}

// Put one checkpoint tensor through the GPU and report how far the answer
// drifts from the CPU's.
//
// The dense half of this model is 6.9 GiB against 90 GiB of routed experts, and
// it is read in full every token at CPU memory bandwidth -- which is why it is
// the half worth moving. Before any of that is built, this establishes the one
// thing everything else assumes: that the existing quantized matvec kernels,
// written for Qwen, decode a deepseek4 tensor to the same numbers this
// runtime's own kernels do. Both paths see the same input vector, so a
// disagreement here is a decode difference and nothing else.
// Put the dense half of the model on the device and keep it there.
//
// Which half is the whole point: the routed experts are 90 GiB and can never be
// resident, while everything else is under 7 and is read in full on every
// token. The activations that cross for each call are a few kilobytes against
// megabytes of weights that do not move at all, and that asymmetry is what
// makes this pay even before the fiddlier pieces have device kernels.
//
// The grouped output projection is left off for now: it does not go through
// ds4_matvec, so uploading it would spend 1.4 GiB of VRAM on nothing.
int colibri_v2_deepseek4_runtime_gpu(
    ColibriV2Deepseek4Runtime* runtime, int32_t device
){return guarded([&]{
    if(!runtime)throw std::runtime_error("invalid runtime");
    auto& rt=*runtime;
    if(rt.gpu)return 0;
    if(colibri_gpu_init(device)!=0)throw std::runtime_error("failed to initialize CUDA");
    ds4_gpu_compile(device);
    rt.gpu_device=device;

    std::vector<std::uint64_t> wanted;
    auto want=[&](std::uint64_t index){
        if(index!=Deepseek4LayerPlan::kAbsent)wanted.push_back(index);
    };
    for(const auto& layer:rt.layers){
        const auto& plan=layer.plan;
        // Whether the shared expert belongs on the device is a measurement
        // rather than a deduction, and the first attempt at it was taken on a
        // 100 W supply shared with a 16-core CPU. COLIBRI_DS4_SHEXP=cpu keeps
        // it off so the two can be compared with everything else held still.
        static const bool shexp_on_device = [] {
            const char* setting = std::getenv("COLIBRI_DS4_SHEXP");
            return !(setting && std::string(setting) == "cpu");
        }();
        for(const auto index:{plan.q_a,plan.q_a_norm,plan.q_b,plan.kv,plan.output_b,plan.comp_kv,
                              plan.comp_gate,plan.gate_inp,plan.indexer_proj,plan.indexer_q_b,
                              plan.indexer_comp_kv,plan.indexer_comp_gate,plan.output_a})
            want(index);
        if(shexp_on_device)
            for(const auto index:{plan.gate_shexp,plan.up_shexp,plan.down_shexp})
                want(index);
    }
    want(rt.output);
    std::sort(wanted.begin(),wanted.end());
    wanted.erase(std::unique(wanted.begin(),wanted.end()),wanted.end());

    std::uint32_t widest_input=0,widest_output=0;
    std::unordered_map<std::uint64_t,std::uint64_t> resident;
    std::uint64_t weight_bytes=0,input_buffer=0,output_buffer=0,batch_buffer=0;
    std::uint64_t expert_cache=0,expert_cache_bytes=0,expert_slot_bytes=0;
    std::uint32_t expert_slots_per_layer=0;
    // Device allocation has several failure points. Keep ownership local until
    // the complete set is usable so a failed enable neither leaks VRAM nor
    // leaves a half-populated runtime that cannot safely be retried.
    auto release_pending=[&]{
        for(const auto& entry:resident)colibri_gpu_free(entry.second);
        if(input_buffer)colibri_gpu_free(input_buffer);
        if(output_buffer)colibri_gpu_free(output_buffer);
        if(batch_buffer)colibri_gpu_free(batch_buffer);
        if(expert_cache)colibri_gpu_free(expert_cache);
        resident.clear(); input_buffer=output_buffer=batch_buffer=0;
        expert_cache=0;
    };
    for(const auto index:wanted){
        const auto& tensor=rt.model->tensors[index];
        // Only what there is a device matvec for; anything else stays on the
        // CPU rather than being uploaded and never used.
        if(tensor.type!=0&&tensor.type!=8&&tensor.type!=14&&tensor.type!=30)continue;
        std::uint64_t pointer=0;
        if(colibri_gpu_alloc(tensor.size,&pointer)!=0){
            release_pending();
            throw std::runtime_error("out of device memory for the dense weights");
        }
        if(colibri_gpu_upload_sync(pointer,tensor_data(*rt.model,tensor),tensor.size)!=0){
            colibri_gpu_free(pointer);
            release_pending();
            throw std::runtime_error("failed to upload a dense weight");
        }
        resident.emplace(index,pointer);
        weight_bytes+=tensor.size;
        const auto rows=tensor.shape.size()>0?tensor.shape[0]:0;
        std::uint64_t columns=1;
        for(std::size_t axis=1;axis<tensor.shape.size();++axis)columns*=tensor.shape[axis];
        widest_input=std::max<std::uint32_t>(widest_input,static_cast<std::uint32_t>(rows));
        widest_output=std::max<std::uint32_t>(widest_output,static_cast<std::uint32_t>(columns));
    }
    // The grouped output projection hands over every head's output at once,
    // which is wider than any single tensor's row count, so the activation
    // buffer is sized for the largest thing that actually crosses rather than
    // for the largest weight.
    widest_input=std::max(widest_input,rt.heads*rt.head_dim);
    const auto input_capacity=static_cast<std::uint64_t>(widest_input)*sizeof(float);
    const auto output_capacity=static_cast<std::uint64_t>(widest_output)*sizeof(float);
    // One batch holds every projection of the hidden state a layer makes.
    const auto batch_capacity=8ull*output_capacity;
    if(colibri_gpu_alloc(input_capacity,&input_buffer)!=0||
       colibri_gpu_alloc(output_capacity,&output_buffer)!=0||
       colibri_gpu_alloc(batch_capacity,&batch_buffer)!=0){
        release_pending();
        throw std::runtime_error("out of device memory for the activation buffers");
    }
    // Keep a layer-partitioned cache. A global LRU is pathological here: the
    // next token visits all 43 layers in the same order and evicts every early
    // layer before it can be reused. Equal per-layer partitions preserve route
    // recurrence and guarantee that one layer cannot churn another's experts.
    for(const auto& layer:rt.layers){
        std::uint64_t bytes=0;
        for(const auto index:{layer.plan.gate_exps,layer.plan.up_exps,layer.plan.down_exps})
            bytes+=rt.model->tensors[index].size/rt.experts;
        expert_slot_bytes=std::max(expert_slot_bytes,bytes);
    }
    expert_slot_bytes=device_align(expert_slot_bytes);
    // Opt-in on this hardware: even grouped execution is slightly slower than
    // the 16-core IQ path at its measured route recurrence. Larger-memory GPUs
    // can enable it explicitly and inspect the exported hit-rate counters.
    std::uint64_t requested_mib=0;
    if(const char* setting=std::getenv("COLIBRI_DS4_EXPERT_CACHE_MIB")){
        if(std::string(setting)=="off")requested_mib=0;
        else requested_mib=std::strtoull(setting,nullptr,10);
    }
    ColibriV2GpuInfo gpu_info{};
    gpu_probe(gpu_info,device);
    constexpr std::uint64_t reserve=768ull*1024*1024;
    const auto available=gpu_info.free_memory>reserve?gpu_info.free_memory-reserve:0;
    const auto budget=std::min<std::uint64_t>(requested_mib*1024ull*1024ull,available);
    if(expert_slot_bytes&&!rt.layers.empty()){
        expert_slots_per_layer=static_cast<std::uint32_t>(
            budget/expert_slot_bytes/rt.layers.size());
        expert_cache_bytes=static_cast<std::uint64_t>(expert_slots_per_layer)*
            rt.layers.size()*expert_slot_bytes;
        if(expert_slots_per_layer<rt.experts_used)expert_cache_bytes=0;
    }
    if(expert_cache_bytes&&colibri_gpu_alloc(expert_cache_bytes,&expert_cache)!=0){
        // Caching is an optimization. Dense placement remains useful when a
        // fragmented or concurrently used device cannot satisfy this request.
        expert_cache=expert_cache_bytes=0;
        expert_slots_per_layer=0;
    }
    rt.resident=std::move(resident);
    rt.gpu_weight_bytes=weight_bytes;
    rt.gpu_input=input_buffer; rt.gpu_output=output_buffer; rt.gpu_batch=batch_buffer;
    rt.gpu_input_capacity=input_capacity; rt.gpu_output_capacity=output_capacity;
    rt.gpu_batch_capacity=batch_capacity;
    rt.expert_cache=expert_cache; rt.expert_cache_bytes=expert_cache_bytes;
    rt.expert_slot_bytes=expert_slot_bytes;
    rt.expert_slots_per_layer=expert_slots_per_layer;
    rt.expert_slots.resize(static_cast<std::size_t>(expert_slots_per_layer)*rt.layers.size());
    rt.gpu_cache_owner=&rt;
    input_buffer=output_buffer=batch_buffer=expert_cache=0;
    rt.gpu=true;
    rt.gpu_owner=true;
    return 0;});}

int colibri_v2_deepseek4_runtime_gpu_share(
    ColibriV2Deepseek4Runtime* runtime,const ColibriV2Deepseek4Runtime* owner
){return guarded([&]{
    if(!runtime||!owner||!owner->gpu)
        throw std::runtime_error("a GPU-enabled owner runtime is required");
    if(runtime->gpu)throw std::runtime_error("runtime already has a GPU workspace");
    if(runtime->model!=owner->model||runtime->n_embd!=owner->n_embd||
       runtime->heads!=owner->heads||runtime->vocabulary!=owner->vocabulary)
        throw std::runtime_error("GPU workspace owner is incompatible");
    runtime->gpu=true; runtime->gpu_owner=false; runtime->gpu_device=owner->gpu_device;
    runtime->resident=owner->resident;
    runtime->gpu_input=owner->gpu_input; runtime->gpu_output=owner->gpu_output;
    runtime->gpu_batch=owner->gpu_batch;
    runtime->gpu_input_capacity=owner->gpu_input_capacity;
    runtime->gpu_output_capacity=owner->gpu_output_capacity;
    runtime->gpu_batch_capacity=owner->gpu_batch_capacity;
    runtime->gpu_weight_bytes=owner->gpu_weight_bytes;
    runtime->expert_cache=owner->expert_cache;
    runtime->expert_cache_bytes=owner->expert_cache_bytes;
    runtime->expert_slot_bytes=owner->expert_slot_bytes;
    runtime->expert_slots_per_layer=owner->expert_slots_per_layer;
    runtime->gpu_cache_owner=const_cast<ColibriV2Deepseek4Runtime*>(owner);
    return 0;});}

// Make this thread's CUDA context current.
//
// The driver retains the device's primary context and makes it current on the
// calling thread, and "current" is per thread. The weights are uploaded from
// whichever thread built the runtime, but the scheduler forwards from its own,
// where nothing was current and every launch failed. Cheap and idempotent, so
// a worker can simply call it once when it starts.
int colibri_v2_deepseek4_runtime_gpu_attach(ColibriV2Deepseek4Runtime* runtime
){return guarded([&]{
    if(!runtime)throw std::runtime_error("invalid runtime");
    if(!runtime->gpu)return 0;
    if(colibri_gpu_init(runtime->gpu_device)!=0)
        throw std::runtime_error("cannot attach this thread to the CUDA context");
    return 0;});}

int colibri_v2_deepseek4_gpu_matvec_check(
    ColibriV2Model* model, const char* name, const float* input, int32_t inputs,
    int32_t outputs, float* out_gpu, float* out_cpu, int32_t device,
    int32_t iterations, double* seconds
){return guarded([&]{
    if(!model||!name||!input||!out_gpu||!out_cpu||inputs<=0||outputs<=0)
        throw std::runtime_error("gpu matvec check arguments are invalid");
    const auto found=std::find_if(model->tensors.begin(),model->tensors.end(),
        [&](const Tensor& tensor){return tensor.name==name;});
    if(found==model->tensors.end())
        throw std::runtime_error(std::string("tensor not found: ")+name);
    if(colibri_gpu_init(device)!=0)
        throw std::runtime_error("failed to initialize CUDA");
    ds4_gpu_compile(device);

    // The CPU answer first, from the same path the runtime uses.
    for(std::int32_t row=0;row<outputs;++row)
        out_cpu[row]=qwen_quant_dot(tensor_data(*model,*found),found->type,input,inputs,
                                    static_cast<std::uint64_t>(row));

    // A three-dimensional routed tensor can be checked one expert at a time;
    // allocating its complete 18 GiB backing just to exercise an 8 MiB kernel
    // slice would make the checker unusable on the device this path targets.
    const std::uint64_t weight_bytes = found->shape.size() == 3
        ? found->size / found->shape[2] : found->size;
    std::uint64_t weights=0,vector=0,result=0;
    if(colibri_gpu_alloc(weight_bytes,&weights)!=0)
        throw std::runtime_error("cannot allocate device weights");
    if(colibri_gpu_alloc(static_cast<std::uint64_t>(inputs)*sizeof(float),&vector)!=0)
        throw std::runtime_error("cannot allocate the device input");
    if(colibri_gpu_alloc(static_cast<std::uint64_t>(outputs)*sizeof(float),&result)!=0)
        throw std::runtime_error("cannot allocate the device output");
    int status=colibri_gpu_upload_sync(weights,tensor_data(*model,*found),weight_bytes);
    if(status==0)status=colibri_gpu_upload_sync(vector,input,
        static_cast<std::uint64_t>(inputs)*sizeof(float));
    if(status==0)status=ds4_gpu_matvec(found->type,weights,vector,result,inputs,outputs,0);
    if(status==0)status=colibri_gpu_sync();
    // Repeat the resident matvec for timing. The weights are already on the
    // device, so this measures the kernel against VRAM rather than the upload,
    // which is the number that decides whether moving the dense half is worth
    // it. This laptop's clocks ramp under load, so the caller must ask for
    // enough iterations to get past that.
    if(status==0&&iterations>0){
        const auto started=ds4_now();
        for(std::int32_t i=0;i<iterations&&status==0;++i)
            status=ds4_gpu_matvec(found->type,weights,vector,result,inputs,outputs,0);
        if(status==0)status=colibri_gpu_sync();
        if(seconds)*seconds=static_cast<double>(ds4_now()-started)/1e9;
    }
    if(status==0)status=colibri_gpu_download(out_gpu,result,
        static_cast<std::uint64_t>(outputs)*sizeof(float),0);
    if(status==0)status=colibri_gpu_sync();
    colibri_gpu_free(weights);colibri_gpu_free(vector);colibri_gpu_free(result);
    if(status!=0)throw std::runtime_error(
        "the device matvec failed with status "+std::to_string(status));
    return 0;});}

int colibri_v2_deepseek4_indexer_key(
    const ColibriV2Deepseek4Runtime* runtime, uint32_t layer, uint32_t block,
    float* out, int32_t outputs
){return guarded([&]{
    if(!runtime||!out)throw std::runtime_error("invalid indexer key request");
    if(layer>=runtime->layers.size())throw std::runtime_error("layer is out of range");
    const auto& state=runtime->layers[layer];
    if(!state.indexer_dim)throw std::runtime_error("this layer carries no indexer");
    if(block>=state.blocks)throw std::runtime_error("that block has not been compressed yet");
    if(outputs!=static_cast<std::int32_t>(state.indexer_dim))
        throw std::runtime_error("output size does not match the indexer key length");
    const std::uint16_t* stored=state.indexer_compressed.data()+
        static_cast<std::size_t>(block)*state.indexer_dim;
    for(std::uint32_t i=0;i<state.indexer_dim;++i)
        out[i]=colibri::v2::deepseek4::half_value(stored[i]);
    return 0;});}

int colibri_v2_deepseek4_indexer_scores(
    const float* queries, const float* keys, const float* weights,
    int32_t heads, int32_t dim, int32_t entries, float* out
){return guarded([&]{
    if(!queries||!keys||!weights||!out||heads<=0||dim<=0||entries<0)
        throw std::runtime_error("indexer score arguments are invalid");
    colibri::v2::deepseek4::indexer_scores(queries,keys,weights,
        static_cast<std::size_t>(heads),static_cast<std::size_t>(dim),
        static_cast<std::size_t>(entries),out);
    return 0;});}

int colibri_v2_deepseek4_top_k(
    const float* scores, int32_t entries, int32_t keep, uint8_t* out
){return guarded([&]{
    if(!scores||!out||entries<0||keep<0)
        throw std::runtime_error("top-k arguments are invalid");
    colibri::v2::deepseek4::top_k_select(scores,static_cast<std::size_t>(entries),
        static_cast<std::size_t>(keep),out);
    return 0;});}

// Round-trip a float through half precision, so the storage format the caches
// use can be checked from outside.
float colibri_v2_deepseek4_half_round_trip(float value){
    return colibri::v2::deepseek4::half_value(colibri::v2::deepseek4::half_bits(value));
}

// Gather the state rows one compressed block pools.
int colibri_v2_deepseek4_gather_block(
    const float* values, const float* scores, int32_t width, int32_t head_dim,
    int32_t ratio, int32_t block, int32_t overlapped,
    float* out_values, float* out_scores, int32_t* rows
){return guarded([&]{
    if(!values||!scores||!out_values||!out_scores||width<=0||head_dim<=0||ratio<=0||block<0)
        throw std::runtime_error("gather-block arguments are invalid");
    const std::int32_t expected=overlapped?2*head_dim:head_dim;
    if(width!=expected)
        throw std::runtime_error("state width does not match the block kind");
    const auto count=colibri::v2::deepseek4::gather_block(
        values,scores,static_cast<std::size_t>(width),static_cast<std::size_t>(head_dim),
        static_cast<std::size_t>(ratio),static_cast<std::size_t>(block),overlapped!=0,
        out_values,out_scores);
    if(rows)*rows=static_cast<std::int32_t>(count);
    return 0;});}

// Build the attention mask for one query position of a DeepSeek-V4 layer.
int colibri_v2_deepseek4_visible_keys(
    int32_t position, int32_t raw_positions, int32_t blocks, int32_t ratio,
    int32_t window, uint8_t* mask, int32_t* visible
){return guarded([&]{
    if(!mask||position<0||raw_positions<0||blocks<0||ratio<0||window<0)
        throw std::runtime_error("visible-key arguments are invalid");
    if(blocks&&!ratio)throw std::runtime_error("compressed blocks need a non-zero ratio");
    const auto count=colibri::v2::deepseek4::visible_keys(
        static_cast<std::size_t>(position),static_cast<std::size_t>(raw_positions),
        static_cast<std::size_t>(blocks),static_cast<std::size_t>(ratio),
        static_cast<std::size_t>(window),mask);
    if(visible)*visible=static_cast<std::int32_t>(count);
    return 0;});}

// Pool one block of positions into a single compressed latent.
int colibri_v2_deepseek4_compress(
    const float* values, const float* scores, int32_t positions, int32_t width,
    float* output
){return guarded([&]{
    if(!values||!scores||!output||positions<=0||width<=0)
        throw std::runtime_error("compress arguments are invalid");
    colibri::v2::deepseek4::compress_block(
        values,scores,static_cast<std::size_t>(positions),
        static_cast<std::size_t>(width),output);
    return 0;});}

// Expert routing for one token: pick the experts and weight them.
int colibri_v2_deepseek4_router(
    const float* logits, const float* bias, int32_t experts, int32_t used,
    float weight_scale, float sum_floor, int32_t select, int32_t* chosen, float* weights
){return guarded([&]{
    if(!logits||!chosen||!weights||experts<=0||used<=0||used>experts)
        throw std::runtime_error("router arguments are invalid");
    if(!select)
        for(std::int32_t slot=0;slot<used;++slot)
            if(chosen[slot]<0||chosen[slot]>=experts)
                throw std::runtime_error("routed expert id is out of range");
    colibri::v2::deepseek4::moe_router(
        logits,bias,static_cast<std::size_t>(experts),static_cast<std::size_t>(used),
        weight_scale,sum_floor,select!=0,chosen,weights);
    return 0;});}

// SwiGLU with both halves clamped, as the per-layer swiglu limits require.
int colibri_v2_deepseek4_swiglu(
    const float* gate, const float* up, int32_t size, float limit, float* output
){return guarded([&]{
    if(!gate||!up||!output||size<=0)throw std::runtime_error("swiglu arguments are invalid");
    colibri::v2::deepseek4::clamped_swiglu(
        gate,up,static_cast<std::size_t>(size),limit,output);
    return 0;});}

// Rotary embedding over `count` rows, each `stride` wide, rotating the trailing
// `rope_dim` elements at `position`.
int colibri_v2_deepseek4_rope(
    float* values, int32_t stride, int32_t rope_dim, int32_t count,
    int32_t position, float freq_base, float freq_scale, int32_t inverse,
    float ext_factor, float attn_factor, float beta_fast, float beta_slow,
    int32_t original_context
){return guarded([&]{
    if(!values||stride<=0||rope_dim<=0||rope_dim>stride||count<=0)
        throw std::runtime_error("rope arguments are invalid");
    colibri::v2::deepseek4::YarnParameters yarn{
        ext_factor,attn_factor,beta_fast,beta_slow,
        static_cast<std::uint32_t>(original_context<0?0:original_context)};
    for(std::int32_t row=0;row<count;++row)
        colibri::v2::deepseek4::rope(
            values+static_cast<std::size_t>(row)*stride+(stride-rope_dim),
            static_cast<std::size_t>(rope_dim),position,freq_base,freq_scale,inverse!=0,yarn);
    return 0;});}

// Attention over the shared KV latent with per-head sink logits.
int colibri_v2_deepseek4_attention(
    const float* queries, const float* latents, const float* sinks,
    const uint8_t* mask, int32_t heads, int32_t head_dim, int32_t positions,
    float scale, float* output
){return guarded([&]{
    if(!queries||!latents||!output||heads<=0||head_dim<=0||positions<=0)
        throw std::runtime_error("attention arguments are invalid");
    colibri::v2::deepseek4::attention_with_sinks(
        queries,latents,sinks,mask,static_cast<std::size_t>(heads),
        static_cast<std::size_t>(head_dim),static_cast<std::size_t>(positions),
        scale,output);
    return 0;});}

// One DeepSeek-V4 hyper-connection step: derive the mixing weights from the
// stream state, collapse the streams into the vector a block reads, and -- when
// `block` is supplied -- write that block's output back into every stream.
// Exposed so the pieces can be diffed against the reference implementation one
// at a time, before there is a whole forward pass to compare.
int colibri_v2_deepseek4_hyper_connection(
    const float* streams, const float* fn, const float* scale, const float* base,
    int32_t n_embd, int32_t hc, int32_t sinkhorn_iterations,
    float rms_epsilon, float hc_epsilon,
    const float* block,
    float* mixes, float* pre, float* post, float* comb,
    float* collapsed, float* combined
){return guarded([&]{
    if(!streams||!fn||!scale||!base||!pre||!post||!comb)
        throw std::runtime_error("hyper-connection inputs are required");
    if(n_embd<=0||hc<=0||sinkhorn_iterations<=0)
        throw std::runtime_error("hyper-connection geometry is invalid");
    namespace ds4=colibri::v2::deepseek4;
    ds4::hyper_connection_weights(
        streams,fn,scale,base,static_cast<std::size_t>(n_embd),static_cast<std::size_t>(hc),
        static_cast<std::uint32_t>(sinkhorn_iterations),rms_epsilon,hc_epsilon,
        pre,post,comb,mixes);
    if(collapsed)
        ds4::hyper_connection_collapse(streams,pre,static_cast<std::size_t>(n_embd),
            static_cast<std::size_t>(hc),collapsed);
    if(block&&combined)
        ds4::hyper_connection_combine(block,streams,post,comb,
            static_cast<std::size_t>(n_embd),static_cast<std::size_t>(hc),combined);
    return 0;});}

// Whether the runtime can decode a GGML weight type at all. This is the CPU
// path's set, which is the widest one: a type missing here cannot be executed
// by any backend, so a checkpoint carrying it is unusable rather than slow.
int colibri_v2_quant_supported(uint32_t type){return qwen_cpu_expert_type_supported(type)?1:0;}

// Copy the per-layer attention-kind array. Returns the number of entries written.
int colibri_v2_model_compress_ratios(const ColibriV2Model* m,uint32_t* out,int32_t capacity){return guarded([&]{
    if(!m)throw std::runtime_error("invalid model handle");
    const auto& ratios=m->config.compress_ratios;
    if(!out||capacity<=0)return static_cast<int>(ratios.size());
    const auto written=std::min(static_cast<std::size_t>(capacity),ratios.size());
    std::copy_n(ratios.begin(),written,out);
    return static_cast<int>(written);});}
int colibri_v2_model_target_layers(const ColibriV2Model* m,uint32_t* out,int32_t capacity){return guarded([&]{
    if(!m)throw std::runtime_error("invalid model handle");
    const auto& layers=m->config.target_layers;
    if(!out||capacity<=0)return static_cast<int>(layers.size());
    const auto written=std::min(static_cast<std::size_t>(capacity),layers.size());
    std::copy_n(layers.begin(),written,out);
    return static_cast<int>(written);});}
int colibri_v2_model_attention_window(const ColibriV2Model* m,uint32_t layer,uint32_t*out){return guarded([&]{if(!m||!out)throw std::runtime_error("invalid model attention-window handle");*out=attention_window(*m,layer);return 0;});}
int colibri_v2_tensor_info(const ColibriV2Model* m,uint64_t i,ColibriV2TensorInfo* out){return guarded([&]{if(!m||!out||i>=m->tensors.size())throw std::runtime_error("tensor index out of range");return fill(m->tensors[i],*out);});}
int colibri_v2_tensor_find(const ColibriV2Model* m,const char* name,ColibriV2TensorInfo* out){return guarded([&]{if(!m||!name||!out)throw std::runtime_error("invalid tensor lookup");for(auto const&t:m->tensors)if(t.name==name)return fill(t,*out);throw std::runtime_error("tensor not found");});}
int colibri_v2_qwen_validate(const ColibriV2Model*m){return guarded([&]{if(!m)throw std::runtime_error("invalid model handle");if(m->config.architecture.find("qwen")!=0)throw std::runtime_error("model architecture is not Qwen");if(!m->config.hidden_size||!m->config.layer_count||!m->config.attention_heads)throw std::runtime_error("Qwen config is incomplete");return 0;});}
int colibri_v2_qwen_tensor_role(const ColibriV2Model*m,const char*role,ColibriV2TensorInfo*out){return guarded([&]{if(!m||!role||!out)throw std::runtime_error("invalid Qwen tensor role lookup");std::vector<std::string> candidates;if(std::strcmp(role,"token_embeddings")==0)candidates={"token_embd.weight","model.embed_tokens.weight","embed_tokens.weight"};else if(std::strcmp(role,"final_norm")==0)candidates={"output_norm.weight","model.norm.weight","norm.weight"};else if(std::strcmp(role,"lm_head")==0)candidates={"output.weight","lm_head.weight"};else throw std::runtime_error("unknown Qwen tensor role");for(auto const&candidate:candidates)for(auto const&t:m->tensors)if(t.name==candidate)return fill(t,*out);throw std::runtime_error("Qwen tensor role is missing");});}
int colibri_v2_qwen_layer_tensor(const ColibriV2Model*m,uint32_t layer,const char*role,ColibriV2TensorInfo*out){return guarded([&]{if(!m||!role||!out)throw std::runtime_error("invalid Qwen layer tensor lookup");std::string prefix="blk."+std::to_string(layer)+".";std::vector<std::string> suffixes;if(std::strcmp(role,"input_norm")==0)suffixes={"attn_norm.weight"};else if(std::strcmp(role,"qkv")==0)suffixes={"attn_qkv.weight"};else if(std::strcmp(role,"attention_q")==0)suffixes={"attn_q.weight"};else if(std::strcmp(role,"attention_k")==0)suffixes={"attn_k.weight"};else if(std::strcmp(role,"attention_v")==0)suffixes={"attn_v.weight"};else if(std::strcmp(role,"attention_output")==0)suffixes={"attn_output.weight","attn_out.weight"};else if(std::strcmp(role,"attention_gate")==0)suffixes={"attn_gate.weight"};else if(std::strcmp(role,"ssm_output")==0)suffixes={"ssm_out.weight"};else if(std::strcmp(role,"ssm_alpha")==0)suffixes={"ssm_alpha.weight"};else if(std::strcmp(role,"ssm_beta")==0)suffixes={"ssm_beta.weight"};else if(std::strcmp(role,"ssm_conv")==0)suffixes={"ssm_conv1d.weight"};else if(std::strcmp(role,"ssm_dt_bias")==0)suffixes={"ssm_dt.bias"};else if(std::strcmp(role,"ssm_a")==0)suffixes={"ssm_a"};else if(std::strcmp(role,"ssm_norm")==0)suffixes={"ssm_norm.weight"};else if(std::strcmp(role,"post_attention_norm")==0)suffixes={"post_attention_norm.weight"};else if(std::strcmp(role,"router")==0)suffixes={"ffn_gate_inp.weight"};else if(std::strcmp(role,"shared_gate")==0)suffixes={"ffn_gate_shexp.weight"};else throw std::runtime_error("unknown Qwen layer tensor role");for(auto const&suffix:suffixes)for(auto const&t:m->tensors)if(t.name==prefix+suffix)return fill(t,*out);throw std::runtime_error("Qwen layer tensor role is missing");});}
float half_to_float(uint16_t bits){uint32_t sign=(bits&0x8000u)<<16, exponent=(bits>>10)&0x1fu, fraction=bits&0x3ffu;uint32_t result;if(exponent==0){if(!fraction)result=sign;else{exponent=1;while((fraction&0x400u)==0){fraction<<=1;--exponent;}result=sign|((exponent+112)<<23)|((fraction&0x3ffu)<<13);}}else if(exponent==31)result=sign|0x7f800000u|(fraction<<13);else result=sign|((exponent+112)<<23)|(fraction<<13);float value;std::memcpy(&value,&result,sizeof(value));return value;}
float tensor_value(const uint8_t*data,uint32_t type,uint64_t index){if(type==0){float value;std::memcpy(&value,data+index*4,4);return value;}if(type==1){uint16_t value;std::memcpy(&value,data+index*2,2);return half_to_float(value);}if(type==30){uint16_t value;std::memcpy(&value,data+index*2,2);uint32_t bits=static_cast<uint32_t>(value)<<16;float result;std::memcpy(&result,&bits,4);return result;}if(type==8){uint64_t block=index/32,within=index%32;uint16_t scale;std::memcpy(&scale,data+block*kQ8BlockSize,2);int8_t quant;std::memcpy(&quant,data+block*kQ8BlockSize+2+within,1);return half_to_float(scale)*static_cast<float>(quant);}if(type==40)return qwen_nvfp4_value(data,index);if(type==10)return qwen_q2k_value(data,index);if(type==11)return qwen_q3k_value(data,index);if(type==16)return qwen_iq2xxs_value(data,index);if(type==18)return qwen_iq3xxs_value(data,index);if(type==22)return qwen_iq2s_value(data,index);if(type==21)return qwen_iq3s_value(data,index);if(type==17)return qwen_iq2xs_value(data,index);if(type==23)return qwen_iq4xs_value(data,index);if(type==12)return qwen_q4k_value(data,index);if(type==13)return qwen_q5_value(data,index);if(type==14)return qwen_q6_value(data,index);throw std::runtime_error("unsupported Qwen CPU tensor type");}
int colibri_v2_dspark_encode(const ColibriV2Model*m,const float*features,uint64_t elements,float*output,uint64_t output_elements){return guarded([&]{
    if(!m||!features||!output)throw std::runtime_error("DSpark encoder arguments are required");
    if(m->config.architecture!="dflash")throw std::runtime_error("not a DFlash/DSpark sidecar");
    const auto width=m->config.hidden_size;
    const auto expected=static_cast<std::uint64_t>(m->config.target_layers.size())*width;
    if(!width||elements!=expected||output_elements<width)throw std::runtime_error("DSpark encoder feature shape is invalid");
    if(colibri_v2_matvec(m,"fc.weight",features,static_cast<int32_t>(elements),output,static_cast<int32_t>(width))!=0)
        throw std::runtime_error(error.empty()?"DSpark fusion projection failed":error);
    const auto norm=std::find_if(m->tensors.begin(),m->tensors.end(),[](const Tensor&t){return t.name=="enc.output_norm.weight";});
    if(norm==m->tensors.end())throw std::runtime_error("DSpark encoder norm is missing");
    double square=0.0;for(std::uint32_t i=0;i<width;++i)square+=static_cast<double>(output[i])*output[i];
    const float scale=1.0f/std::sqrt(static_cast<float>(square/width)+m->config.rms_norm_epsilon);
    const auto*data=tensor_data(*m,*norm);
    for(std::uint32_t i=0;i<width;++i)output[i]*=scale*tensor_value(data,norm->type,i);
    return 0;});}

struct ColibriV2DsparkRuntime {
    ColibriV2Model* model=nullptr;
    std::uint32_t context_limit=0,position=0,width=0,rope_dim=0,window=0;
    float epsilon=0.0f,freq_base=10000.0f;
    std::vector<std::uint16_t> cache;
    std::vector<float> projected;
    ColibriV2Deepseek4Runtime* decoder=nullptr;
};

int colibri_v2_dspark_runtime_create(ColibriV2Model*m,uint32_t context_limit,ColibriV2DsparkRuntime**out){return guarded([&]{
    if(!m||!out||!context_limit)throw std::runtime_error("DSpark model, context and output are required");
    if(m->config.architecture!="dflash")throw std::runtime_error("not a DFlash/DSpark sidecar");
    if(context_limit>m->config.context_length)throw std::runtime_error("DSpark context exceeds checkpoint limit");
    auto runtime=std::make_unique<ColibriV2DsparkRuntime>();
    runtime->model=m;runtime->context_limit=context_limit;runtime->width=m->config.kv_lora_rank;
    runtime->rope_dim=m->config.rotary_dimension;runtime->window=std::min(context_limit,m->config.sliding_window);
    runtime->epsilon=m->config.rms_norm_epsilon;runtime->freq_base=m->config.rope_freq_base;
    if(!runtime->width||runtime->rope_dim>runtime->width||!runtime->window)
        throw std::runtime_error("DSpark KV geometry is incomplete");
    runtime->cache.assign(static_cast<std::size_t>(m->config.layer_count)*runtime->window*runtime->width,0);
    runtime->projected.resize(runtime->width);
    if(colibri_v2_deepseek4_runtime_create(m,context_limit,&runtime->decoder)!=0)
        throw std::runtime_error(error.empty()?"cannot create DSpark decoder plan":error);
    *out=runtime.release();return 0;});}
void colibri_v2_dspark_runtime_free(ColibriV2DsparkRuntime*r){try{if(r)colibri_v2_deepseek4_runtime_free(r->decoder);delete r;}catch(...){}}

int colibri_v2_dspark_inject(ColibriV2DsparkRuntime*r,const float*fused,uint64_t elements){return guarded([&]{
    if(!r||!fused||elements!=r->model->config.hidden_size)throw std::runtime_error("invalid DSpark injection");
    if(r->position>=r->context_limit)throw std::runtime_error("DSpark cache is past its context limit");
    auto& m=*r->model;
    for(std::uint32_t layer=0;layer<m.config.layer_count;++layer){
        const auto prefix="blk."+std::to_string(layer)+".";
        if(colibri_v2_matvec(&m,(prefix+"attn_kv.weight").c_str(),fused,static_cast<int32_t>(elements),r->projected.data(),r->width)!=0)
            throw std::runtime_error(error.empty()?"DSpark KV projection failed":error);
        const auto norm_name=prefix+"attn_kv_a_norm.weight";
        const auto norm=std::find_if(m.tensors.begin(),m.tensors.end(),[&](const Tensor&t){return t.name==norm_name;});
        if(norm==m.tensors.end())throw std::runtime_error("DSpark KV norm is missing");
        double square=0.0;for(const auto value:r->projected)square+=static_cast<double>(value)*value;
        const float scale=1.0f/std::sqrt(static_cast<float>(square/r->width)+r->epsilon);
        const auto*gain=tensor_data(m,*norm);
        for(std::uint32_t i=0;i<r->width;++i)r->projected[i]*=scale*tensor_value(gain,norm->type,i);
        colibri::v2::deepseek4::rope(r->projected.data()+r->width-r->rope_dim,r->rope_dim,
            static_cast<std::int32_t>(r->position),r->freq_base,1.0f,false);
        auto*slot=r->cache.data()+(static_cast<std::size_t>(layer)*r->window+r->position%r->window)*r->width;
        auto*decoder_slot=r->decoder->layers[layer].latents.data()+
            static_cast<std::size_t>(r->position%r->window)*r->width;
        for(std::uint32_t i=0;i<r->width;++i)slot[i]=decoder_slot[i]=colibri::v2::deepseek4::half_bits(r->projected[i]);
        r->decoder->layers[layer].positions=r->position+1;
    }
    ++r->position;return 0;});}

int colibri_v2_dspark_cached(const ColibriV2DsparkRuntime*r,uint32_t layer,uint32_t position,float*output,uint64_t elements){return guarded([&]{
    if(!r||!output||layer>=r->model->config.layer_count||position>=r->position||
       r->position-position>r->window||elements<r->width)throw std::runtime_error("invalid DSpark cache read");
    const auto*slot=r->cache.data()+(static_cast<std::size_t>(layer)*r->window+position%r->window)*r->width;
    for(std::uint32_t i=0;i<r->width;++i)output[i]=colibri::v2::deepseek4::half_value(slot[i]);
    return 0;});}

int colibri_v2_dspark_heads(const ColibriV2Model*m,const float*base_logits,const float*hidden,
    uint32_t rows,uint32_t anchor_token,float*logits,float*confidence,uint32_t*tokens){return guarded([&]{
    if(!m||!base_logits||!hidden||!logits||!confidence||!tokens)
        throw std::runtime_error("DSpark head arguments are required");
    if(m->config.architecture!="dflash"||!rows||rows>m->config.draft_block_size)
        throw std::runtime_error("invalid DSpark head block");
    const auto vocab=m->config.vocabulary_size,width=m->config.hidden_size;
    if(anchor_token>=vocab)throw std::runtime_error("DSpark anchor is outside the vocabulary");
    auto find=[&](const char*name)->const Tensor&{
        const auto found=std::find_if(m->tensors.begin(),m->tensors.end(),[&](const Tensor&t){return t.name==name;});
        if(found==m->tensors.end())throw std::runtime_error(std::string("DSpark tensor is missing: ")+name);
        return *found;
    };
    const auto&w1=find("markov_w1.weight");const auto&w2=find("markov_w2.weight");
    const auto&conf=find("conf_proj.weight");
    if(w1.shape.size()!=2||w1.shape[0]!=256||w1.shape[1]!=vocab||
       w2.shape.size()!=2||w2.shape[0]!=256||w2.shape[1]!=vocab||
       conf.shape.size()!=2||conf.shape[0]!=width+256||conf.shape[1]!=1)
        throw std::runtime_error("DSpark head tensor geometry is invalid");
    const auto*w1_data=tensor_data(*m,w1);const auto*conf_data=tensor_data(*m,conf);
    std::vector<float> feature(256),bias(vocab);
    auto previous=anchor_token;
    for(std::uint32_t row=0;row<rows;++row){
        for(std::uint32_t i=0;i<256;++i)
            feature[i]=tensor_value(w1_data,w1.type,static_cast<std::uint64_t>(previous)*256+i);
        if(colibri_v2_matvec(m,"markov_w2.weight",feature.data(),256,bias.data(),vocab)!=0)
            throw std::runtime_error(error.empty()?"DSpark Markov projection failed":error);
        auto*out=logits+static_cast<std::size_t>(row)*vocab;
        const auto*base=base_logits+static_cast<std::size_t>(row)*vocab;
        std::uint32_t best=0;
        for(std::uint32_t token=0;token<vocab;++token){
            out[token]=base[token]+bias[token];if(out[token]>out[best])best=token;
        }
        double score=0.0;const auto*state=hidden+static_cast<std::size_t>(row)*width;
        for(std::uint32_t i=0;i<width;++i)score+=static_cast<double>(state[i])*tensor_value(conf_data,conf.type,i);
        for(std::uint32_t i=0;i<256;++i)score+=static_cast<double>(feature[i])*tensor_value(conf_data,conf.type,width+i);
        confidence[row]=colibri::v2::deepseek4::sigmoid(static_cast<float>(score));
        tokens[row]=best;previous=best;
    }
    return 0;});}

int colibri_v2_dspark_attention(const ColibriV2DsparkRuntime*r,uint32_t layer,
    const float*queries,const float*noise_kv,uint32_t rows,float*output,uint64_t elements){return guarded([&]{
    if(!r||!queries||!noise_kv||!output||layer>=r->model->config.layer_count||
       !rows||rows>r->model->config.draft_block_size)
        throw std::runtime_error("invalid DSpark attention block");
    const auto heads=r->model->config.attention_heads,width=r->width;
    const auto needed=static_cast<std::uint64_t>(rows)*heads*width;
    if(elements<needed)throw std::runtime_error("DSpark attention output is too small");
    const auto sink_name="blk."+std::to_string(layer)+".attn_sinks.weight";
    const auto sink=std::find_if(r->model->tensors.begin(),r->model->tensors.end(),[&](const Tensor&t){return t.name==sink_name;});
    if(sink==r->model->tensors.end())throw std::runtime_error("DSpark attention sinks are missing");
    const auto*sink_data=tensor_data(*r->model,*sink);
    std::vector<float>sinks(heads),keys(static_cast<std::size_t>(r->window+rows)*width);
    for(std::uint32_t h=0;h<heads;++h)sinks[h]=tensor_value(sink_data,sink->type,h);
    const float scale=1.0f/std::sqrt(static_cast<float>(width));
    for(std::uint32_t row=0;row<rows;++row){
        const auto query_position=r->position+row;
        const auto first=query_position+1>r->window?query_position+1-r->window:0u;
        std::uint32_t count=0;
        for(std::uint32_t position=first;position<r->position;++position){
            const auto*slot=r->cache.data()+(static_cast<std::size_t>(layer)*r->window+position%r->window)*width;
            auto*target=keys.data()+static_cast<std::size_t>(count++)*width;
            for(std::uint32_t i=0;i<width;++i)target[i]=colibri::v2::deepseek4::half_value(slot[i]);
        }
        for(std::uint32_t noise=0;noise<rows;++noise){
            const auto key_position=r->position+noise;
            if(key_position<=query_position&&query_position-key_position>=r->window)continue;
            std::copy_n(noise_kv+static_cast<std::size_t>(noise)*width,width,
                keys.data()+static_cast<std::size_t>(count++)*width);
        }
        colibri::v2::deepseek4::attention_with_sinks(
            queries+static_cast<std::size_t>(row)*heads*width,keys.data(),sinks.data(),nullptr,
            heads,width,count,scale,output+static_cast<std::size_t>(row)*heads*width);
    }
    return 0;});}

int colibri_v2_dspark_attention_stage(ColibriV2DsparkRuntime*r,uint32_t layer,
    const float*streams,uint32_t rows,float*output,uint64_t elements){return guarded([&]{
    if(!r||!streams||!output||layer>=r->decoder->layers.size()||!rows||rows>r->model->config.draft_block_size)
        throw std::runtime_error("invalid DSpark attention stage");
    auto&rt=*r->decoder;const auto&plan=rt.layers[layer].plan;const auto&model=*rt.model;
    namespace ds4=colibri::v2::deepseek4;
    const auto stream_width=static_cast<std::size_t>(rt.hc)*rt.n_embd;
    if(elements<static_cast<std::uint64_t>(rows)*stream_width)throw std::runtime_error("DSpark stage output is too small");
    const auto wide=static_cast<std::size_t>(rt.heads)*rt.head_dim;
    std::vector<Deepseek4Scratch>scratch(rows,rt.scratch);
    std::vector<float>queries(static_cast<std::size_t>(rows)*wide);
    std::vector<float>latents(static_cast<std::size_t>(rows)*rt.head_dim);
    std::vector<float>attended(static_cast<std::size_t>(rows)*wide);
    for(std::uint32_t row=0;row<rows;++row){
        auto&sc=scratch[row];const auto*input=streams+static_cast<std::size_t>(row)*stream_width;
        ds4::hyper_connection_weights(input,ds4_f32(model,plan.hc_attn_fn),ds4_f32(model,plan.hc_attn_scale),
            ds4_f32(model,plan.hc_attn_base),rt.n_embd,rt.hc,rt.sinkhorn_iterations,rt.epsilon,rt.epsilon,
            sc.pre.data(),sc.post.data(),sc.comb.data());
        ds4::hyper_connection_collapse(input,sc.pre.data(),rt.n_embd,rt.hc,sc.collapsed.data());
        ds4::rms_norm(sc.collapsed.data(),rt.n_embd,rt.epsilon,sc.hidden.data());
        const auto*attn_gain=ds4_f32(model,plan.attn_norm);
        for(std::uint32_t i=0;i<rt.n_embd;++i)sc.hidden[i]*=attn_gain[i];
        ds4_matvec(rt,plan.q_a,sc.hidden.data(),rt.n_embd,sc.low_rank.data(),rt.q_lora);
        ds4::rms_norm(sc.low_rank.data(),rt.q_lora,rt.epsilon,sc.low_rank.data());
        const auto*q_gain=ds4_f32(model,plan.q_a_norm);
        for(std::uint32_t i=0;i<rt.q_lora;++i)sc.low_rank[i]*=q_gain[i];
        ds4_matvec(rt,plan.q_b,sc.low_rank.data(),rt.q_lora,sc.query.data(),static_cast<int32_t>(wide));
        ds4_matvec(rt,plan.kv,sc.hidden.data(),rt.n_embd,sc.latent.data(),rt.head_dim);
        const auto position=r->position+row;
        for(std::uint32_t head=0;head<rt.heads;++head){
            auto*q=sc.query.data()+static_cast<std::size_t>(head)*rt.head_dim;
            ds4::rms_norm(q,rt.head_dim,rt.epsilon,q);
            ds4::rope(q+rt.head_dim-rt.rope_dim,rt.rope_dim,position,rt.freq_base,1.0f,false);
        }
        ds4::rms_norm(sc.latent.data(),rt.head_dim,rt.epsilon,sc.latent.data());
        const auto*kv_gain=ds4_f32(model,plan.kv_a_norm);
        for(std::uint32_t i=0;i<rt.head_dim;++i)sc.latent[i]*=kv_gain[i];
        ds4::rope(sc.latent.data()+rt.head_dim-rt.rope_dim,rt.rope_dim,position,rt.freq_base,1.0f,false);
        std::copy_n(sc.query.data(),wide,queries.data()+static_cast<std::size_t>(row)*wide);
        std::copy_n(sc.latent.data(),rt.head_dim,latents.data()+static_cast<std::size_t>(row)*rt.head_dim);
    }
    if(colibri_v2_dspark_attention(r,layer,queries.data(),latents.data(),rows,attended.data(),attended.size())!=0)
        throw std::runtime_error(error.empty()?"DSpark block attention failed":error);
    for(std::uint32_t row=0;row<rows;++row){
        auto&sc=scratch[row];const auto position=r->position+row;
        std::copy_n(attended.data()+static_cast<std::size_t>(row)*wide,wide,sc.derope.data());
        for(std::uint32_t head=0;head<rt.heads;++head)
            ds4::rope(sc.derope.data()+static_cast<std::size_t>(head)*rt.head_dim+rt.head_dim-rt.rope_dim,
                rt.rope_dim,position,rt.freq_base,1.0f,true);
        const auto&wa=model.tensors[plan.output_a];const auto*packed=tensor_data(model,wa);
        const auto inputs=static_cast<std::int32_t>(wide/rt.groups);
        for(std::uint32_t out_row=0;out_row<rt.groups*rt.lora_rank;++out_row){
            const auto group=out_row/rt.lora_rank;
            sc.grouped[out_row]=qwen_quant_dot(packed,wa.type,sc.derope.data()+static_cast<std::size_t>(group)*inputs,inputs,out_row);
        }
        ds4_matvec(rt,plan.output_b,sc.grouped.data(),rt.groups*rt.lora_rank,sc.block_out.data(),rt.n_embd);
        const auto*input=streams+static_cast<std::size_t>(row)*stream_width;
        auto*out=output+static_cast<std::size_t>(row)*stream_width;
        ds4::hyper_connection_combine(sc.block_out.data(),input,sc.post.data(),sc.comb.data(),rt.n_embd,rt.hc,out);
    }
    return 0;});}

int colibri_v2_dspark_ffn_stage(ColibriV2DsparkRuntime*r,uint32_t layer,
    const float*streams,uint32_t rows,float*output,uint64_t elements){return guarded([&]{
    if(!r||!streams||!output||layer>=r->decoder->layers.size()||!rows||rows>r->model->config.draft_block_size)
        throw std::runtime_error("invalid DSpark FFN stage");
    auto&rt=*r->decoder;const auto&plan=rt.layers[layer].plan;const auto&model=*rt.model;
    namespace ds4=colibri::v2::deepseek4;
    const auto stream_width=static_cast<std::size_t>(rt.hc)*rt.n_embd;
    if(elements<static_cast<std::uint64_t>(rows)*stream_width)throw std::runtime_error("DSpark FFN output is too small");
    for(std::uint32_t row=0;row<rows;++row){
        Deepseek4Scratch sc=rt.scratch;
        const auto*input=streams+static_cast<std::size_t>(row)*stream_width;
        auto*out=output+static_cast<std::size_t>(row)*stream_width;
        std::copy_n(input,stream_width,out);
        ds4::hyper_connection_weights(out,ds4_f32(model,plan.hc_ffn_fn),ds4_f32(model,plan.hc_ffn_scale),
            ds4_f32(model,plan.hc_ffn_base),rt.n_embd,rt.hc,rt.sinkhorn_iterations,rt.epsilon,rt.epsilon,
            sc.pre.data(),sc.post.data(),sc.comb.data());
        ds4::hyper_connection_collapse(out,sc.pre.data(),rt.n_embd,rt.hc,sc.collapsed.data());
        ds4::rms_norm(sc.collapsed.data(),rt.n_embd,rt.epsilon,sc.hidden.data());
        const auto*gain=ds4_f32(model,plan.ffn_norm);
        for(std::uint32_t i=0;i<rt.n_embd;++i)sc.hidden[i]*=gain[i];
        std::fill(sc.router.begin(),sc.router.end(),0.0f);
        ds4_matvec(rt,plan.gate_inp,sc.hidden.data(),rt.n_embd,sc.router.data(),rt.experts);
        ds4::moe_router(sc.router.data(),ds4_f32(model,plan.exp_probs_b),rt.experts,rt.experts_used,
            rt.weight_scale,1e-20f,true,sc.experts.data(),sc.weights.data());
        ds4_layer_ffn_complete(rt,layer,out,sc);
    }
    return 0;});}

int colibri_v2_dspark_decode_hidden(ColibriV2DsparkRuntime*r,const float*embeddings,
    uint32_t rows,float*hidden,float*normalized,uint64_t elements){return guarded([&]{
    if(!r||!embeddings||!hidden||!normalized||!rows||rows>r->model->config.draft_block_size)
        throw std::runtime_error("invalid DSpark decode block");
    auto&rt=*r->decoder;const auto width=rt.n_embd;
    if(elements<static_cast<std::uint64_t>(rows)*width)throw std::runtime_error("DSpark hidden output is too small");
    const auto stream_width=static_cast<std::size_t>(rt.hc)*width;
    std::vector<float>streams(static_cast<std::size_t>(rows)*stream_width),next(streams.size());
    for(std::uint32_t row=0;row<rows;++row)for(std::uint32_t stream=0;stream<rt.hc;++stream)
        std::copy_n(embeddings+static_cast<std::size_t>(row)*width,width,
            streams.data()+static_cast<std::size_t>(row)*stream_width+static_cast<std::size_t>(stream)*width);
    for(std::uint32_t layer=0;layer<rt.layers.size();++layer){
        if(colibri_v2_dspark_attention_stage(r,layer,streams.data(),rows,next.data(),next.size())!=0)
            throw std::runtime_error(error.empty()?"DSpark attention stage failed":error);
        if(colibri_v2_dspark_ffn_stage(r,layer,next.data(),rows,streams.data(),streams.size())!=0)
            throw std::runtime_error(error.empty()?"DSpark FFN stage failed":error);
    }
    namespace ds4=colibri::v2::deepseek4;const auto&model=*rt.model;
    for(std::uint32_t row=0;row<rows;++row){
        std::vector<float>pre(rt.hc);
        auto*out=hidden+static_cast<std::size_t>(row)*width;
        ds4::hyper_connection_head(streams.data()+static_cast<std::size_t>(row)*stream_width,
            ds4_f32(model,rt.head_fn),ds4_f32(model,rt.head_scale),ds4_f32(model,rt.head_base),
            width,rt.hc,rt.epsilon,rt.epsilon,pre.data(),out);
        auto*norm=normalized+static_cast<std::size_t>(row)*width;
        ds4::rms_norm(out,width,rt.epsilon,norm);const auto*gain=ds4_f32(model,rt.output_norm);
        for(std::uint32_t i=0;i<width;++i)norm[i]*=gain[i];
    }
    return 0;});}
const Tensor& qwen_role_tensor(const ColibriV2Model&m,const char*role){std::vector<std::string> candidates;if(std::strcmp(role,"token_embeddings")==0)candidates={"token_embd.weight","model.embed_tokens.weight","embed_tokens.weight"};else if(std::strcmp(role,"lm_head")==0)candidates={"output.weight","lm_head.weight"};else throw std::runtime_error("unknown Qwen tensor role");for(auto const&candidate:candidates)for(auto const&t:m.tensors)if(t.name==candidate)return t;throw std::runtime_error("Qwen tensor role is missing");}
int colibri_v2_qwen_embedding(const ColibriV2Model*m,uint32_t token,float*out,uint64_t elements){return guarded([&]{if(!m||!out)throw std::runtime_error("invalid embedding arguments");const Tensor&t=qwen_role_tensor(*m,"token_embeddings");if(t.shape.size()!=2)throw std::runtime_error("embedding shape is invalid");uint64_t width=t.shape[0]==m->config.hidden_size?t.shape[0]:t.shape[1],vocab=t.shape[0]==m->config.hidden_size?t.shape[1]:t.shape[0];if(token>=vocab||elements<width)throw std::runtime_error("embedding token or buffer is invalid");const auto*data=tensor_data(*m,t);for(uint64_t i=0;i<width;i++)out[i]=tensor_value(data,t.type,static_cast<uint64_t>(token)*width+i);return 0;});}
int colibri_v2_qwen_lm_head(const ColibriV2Model*m,const float*hidden,float*logits,uint64_t vocabulary,uint64_t elements){return guarded([&]{if(!m||!hidden||!logits)throw std::runtime_error("invalid LM-head arguments");const Tensor&t=qwen_role_tensor(*m,"lm_head");if(t.shape.size()!=2)throw std::runtime_error("LM-head shape is invalid");uint64_t width=t.shape[0]==m->config.hidden_size?t.shape[0]:t.shape[1],vocab=t.shape[0]==m->config.hidden_size?t.shape[1]:t.shape[0];if(vocabulary<vocab||elements<width)throw std::runtime_error("LM-head shape or buffer is invalid");const auto*data=tensor_data(*m,t);for(uint64_t row=0;row<vocab;row++){float sum=0;for(uint64_t column=0;column<width;column++)sum+=tensor_value(data,t.type,row*width+column)*hidden[column];logits[row]=sum;}return 0;});}
int colibri_v2_qwen_token_text(const ColibriV2Model*m,uint32_t token,char*out,uint64_t capacity){return guarded([&]{if(!m||!out||capacity==0)throw std::runtime_error("invalid token text arguments");if(token>=m->vocabulary.size())throw std::runtime_error("token is outside the GGUF vocabulary");const std::string&value=m->vocabulary[token];if(capacity<=value.size())throw std::runtime_error("token text buffer is too small");std::memcpy(out,value.data(),value.size());out[value.size()]=0;return 0;});}
int colibri_v2_token_id(const ColibriV2Model*m,const char*text,uint32_t*token){return guarded([&]{if(!m||!text||!token)throw std::runtime_error("invalid token lookup arguments");const auto it=m->vocabulary_ids.find(text);if(it==m->vocabulary_ids.end())throw std::runtime_error("token text is not in the GGUF vocabulary");*token=it->second;return 0;});}
std::string gguf_byte_encode(const char*text){static const int direct[] = {33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,161,162,163,164,165,166,167,168,169,170,171,172,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255};std::array<int,256>map{};for(int i=0;i<256;i++)map[i]=-1;for(int i=0;i<static_cast<int>(sizeof(direct)/sizeof(direct[0]));i++)map[direct[i]]=direct[i];int extra=0;for(int i=0;i<256;i++)if(map[i]<0)map[i]=256+extra++;std::string out;for(const unsigned char*p=reinterpret_cast<const unsigned char*>(text);*p;p++){int cp=map[*p];if(cp<128)out.push_back(static_cast<char>(cp));else if(cp<2048){out.push_back(static_cast<char>(0xC0|(cp>>6)));out.push_back(static_cast<char>(0x80|(cp&63)));}else{out.push_back(static_cast<char>(0xE0|(cp>>12)));out.push_back(static_cast<char>(0x80|((cp>>6)&63)));out.push_back(static_cast<char>(0x80|(cp&63)));}}return out;}
std::vector<std::string> gguf_utf8_symbols(const std::string&text){std::vector<std::string> symbols;for(size_t i=0;i<text.size();){unsigned char c=text[i];size_t width=(c<0x80)?1:(c<0xE0?2:(c<0xF0?3:4));if(i+width>text.size())width=1;symbols.emplace_back(text.data()+i,width);i+=width;}return symbols;}
// One UTF-8 codepoint at `offset`, with its encoded width.
std::uint32_t gguf_utf8_codepoint(const std::string& text, size_t offset, size_t& width) {
    const auto lead=static_cast<unsigned char>(text[offset]);
    width=lead<0x80?1:(lead<0xE0?2:(lead<0xF0?3:4));
    if(offset+width>text.size()){width=1;return lead;}
    if(width==1)return lead;
    std::uint32_t value=lead&(0xFFu>>(width+1));
    for(size_t i=1;i<width;++i)
        value=(value<<6)|(static_cast<unsigned char>(text[offset+i])&0x3Fu);
    return value;
}

// Letter/number classification for the pre-tokenizer's \p{L} and \p{N}.
// Exact over ASCII, which is what a code model overwhelmingly sees; above ASCII
// it treats general punctuation, symbol and private-use blocks as non-letters
// and everything else as a letter, rather than carrying a full Unicode category
// table. Non-Latin prose can therefore split differently from the reference.
bool gguf_codepoint_is_number(std::uint32_t codepoint) {
    return codepoint>='0'&&codepoint<='9';
}
bool gguf_codepoint_is_letter(std::uint32_t codepoint) {
    if(codepoint<0x80)
        return (codepoint>='A'&&codepoint<='Z')||(codepoint>='a'&&codepoint<='z');
    if(codepoint>=0x2000&&codepoint<=0x206F)return false;  // general punctuation
    if(codepoint>=0x2070&&codepoint<=0x2BFF)return false;  // sub/superscripts, symbols, arrows
    if(codepoint>=0x3000&&codepoint<=0x303F)return false;  // CJK punctuation
    if(codepoint>=0xE000&&codepoint<=0xF8FF)return false;  // private use
    if(codepoint>=0xFE00&&codepoint<=0xFE0F)return false;  // variation selectors
    if(codepoint>=0xFF00&&codepoint<=0xFF0F)return false;  // fullwidth punctuation
    if(codepoint>=0x1F000)return false;                    // emoji and pictographs
    return true;
}
bool gguf_codepoint_is_space(std::uint32_t codepoint) {
    return codepoint==' '||codepoint=='\t'||codepoint=='\n'||codepoint=='\r'||
           codepoint=='\f'||codepoint=='\v'||codepoint==0x85||codepoint==0xA0||
           (codepoint>=0x2000&&codepoint<=0x200A)||codepoint==0x2028||
           codepoint==0x2029||codepoint==0x202F||codepoint==0x205F||codepoint==0x3000;
}

// Laguna's BPE pre-tokenizer, as a direct transcription of the two regexes the
// reference implementation applies in sequence. Splitting here keeps merges from
// running across word, digit and whitespace boundaries the model never saw:
// notably each digit stands alone, so numbers do not collapse into one token.
std::vector<std::string> laguna_pretokenize(const std::string& text) {
    std::vector<std::string> pieces;
    // First regex: "[^\n]+|[\n]+" -- newline runs separate from everything else.
    for(size_t start=0;start<text.size();){
        const bool newline=text[start]=='\n';
        size_t end=start;
        while(end<text.size()&&(text[end]=='\n')==newline)++end;
        const std::string segment=text.substr(start,end-start);
        start=end;
        if(newline){pieces.push_back(segment);continue;}
        // Second regex, greedy alternation at each position.
        for(size_t at=0;at<segment.size();){
            size_t width=0;
            const auto first=gguf_utf8_codepoint(segment,at,width);
            const size_t begin=at;
            // "'s" and friends, either case.
            if(first=='\''&&at+1<segment.size()){
                const char follow=static_cast<char>(std::tolower(
                    static_cast<unsigned char>(segment[at+1])));
                const char third=at+2<segment.size()
                    ?static_cast<char>(std::tolower(static_cast<unsigned char>(segment[at+2]))):0;
                size_t length=0;
                if(follow=='s'||follow=='t'||follow=='m'||follow=='d')length=2;
                else if((follow=='r'&&third=='e')||(follow=='v'&&third=='e')||
                        (follow=='l'&&third=='l'))length=3;
                if(length){pieces.push_back(segment.substr(at,length));at+=length;continue;}
            }
            // "[^\r\n\p{L}\p{N}]?\p{L}+": an optional single leading non-letter.
            {
                size_t probe=at,probe_width=width;
                auto codepoint=first;
                if(!gguf_codepoint_is_letter(codepoint)&&!gguf_codepoint_is_number(codepoint)&&
                   codepoint!='\r'&&codepoint!='\n'){
                    const size_t after=probe+probe_width;
                    if(after<segment.size()){
                        size_t next_width=0;
                        const auto next=gguf_utf8_codepoint(segment,after,next_width);
                        if(gguf_codepoint_is_letter(next)){probe=after;probe_width=next_width;codepoint=next;}
                    }
                }
                if(gguf_codepoint_is_letter(codepoint)){
                    size_t end_at=probe;
                    while(end_at<segment.size()){
                        size_t letter_width=0;
                        if(!gguf_codepoint_is_letter(
                                gguf_utf8_codepoint(segment,end_at,letter_width)))break;
                        end_at+=letter_width;
                    }
                    pieces.push_back(segment.substr(begin,end_at-begin));at=end_at;continue;
                }
            }
            // "\p{N}": exactly one digit.
            if(gguf_codepoint_is_number(first)){
                pieces.push_back(segment.substr(at,width));at+=width;continue;
            }
            // " ?[^\s\p{L}\p{N}]+[\r\n]*"
            {
                size_t probe=at;
                if(first==' '){
                    const size_t after=at+width;
                    if(after<segment.size()){
                        size_t next_width=0;
                        const auto next=gguf_utf8_codepoint(segment,after,next_width);
                        if(!gguf_codepoint_is_space(next)&&!gguf_codepoint_is_letter(next)&&
                           !gguf_codepoint_is_number(next))probe=after;
                    }
                }
                if(probe<segment.size()){
                    size_t end_at=probe,run_width=0;
                    while(end_at<segment.size()){
                        const auto codepoint=gguf_utf8_codepoint(segment,end_at,run_width);
                        if(gguf_codepoint_is_space(codepoint)||gguf_codepoint_is_letter(codepoint)||
                           gguf_codepoint_is_number(codepoint))break;
                        end_at+=run_width;
                    }
                    if(end_at>probe){
                        while(end_at<segment.size()&&
                              (segment[end_at]=='\r'||segment[end_at]=='\n'))++end_at;
                        pieces.push_back(segment.substr(begin,end_at-begin));at=end_at;continue;
                    }
                }
            }
            // "\s+(?!\S)" and "\s+": a whitespace run, with the last space split
            // off when more non-space follows, which is what the lookahead does.
            {
                size_t end_at=at,space_width=0;
                while(end_at<segment.size()&&
                      gguf_codepoint_is_space(gguf_utf8_codepoint(segment,end_at,space_width)))
                    end_at+=space_width;
                if(end_at>at){
                    size_t stop=end_at;
                    if(end_at<segment.size()&&end_at-at>1)--stop;
                    pieces.push_back(segment.substr(at,stop-at));at=stop;continue;
                }
            }
            pieces.push_back(segment.substr(at,width));at+=width;
        }
    }
    return pieces;
}

// UTF-8 decoded into codepoints alongside the byte offset each one starts at,
// with a trailing entry for the end so a piece can be sliced by codepoint index.
struct Utf8Text {
    std::vector<std::uint32_t> code;
    std::vector<std::size_t> offset;
};

Utf8Text gguf_utf8_decode(const std::string& text) {
    Utf8Text out;
    for(std::size_t at=0;at<text.size();){
        std::size_t width=0;
        const auto codepoint=gguf_utf8_codepoint(text,at,width);
        if(!width)width=1;
        out.code.push_back(codepoint);
        out.offset.push_back(at);
        at+=width;
    }
    out.offset.push_back(text.size());
    return out;
}

// The reference pre-tokenizer applies its regexes in sequence, each one
// splitting the pieces the previous one produced. Within a piece every match
// becomes a piece of its own, and so does every gap between matches.
// Takes a function pointer rather than a template parameter: this lives inside
// the extern "C" block, which templates may not.
using GgufMatcher = std::size_t (*)(const std::vector<std::uint32_t>&, std::size_t);

std::vector<std::string> gguf_regex_split(
    const std::vector<std::string>& pieces, GgufMatcher match_at
) {
    std::vector<std::string> out;
    for(const auto& piece:pieces){
        const auto text=gguf_utf8_decode(piece);
        std::size_t gap=0;
        for(std::size_t at=0;at<text.code.size();){
            const auto length=match_at(text.code,at);
            if(!length){++at;continue;}
            if(at>gap)
                out.push_back(piece.substr(text.offset[gap],text.offset[at]-text.offset[gap]));
            out.push_back(piece.substr(text.offset[at],text.offset[at+length]-text.offset[at]));
            at+=length;
            gap=at;
        }
        if(gap<text.code.size())out.push_back(piece.substr(text.offset[gap]));
    }
    return out;
}

// The reference does not evaluate \p{...} against real Unicode categories for
// ASCII. It "collapses" the text first: codepoints below 128 are left as
// themselves and every category in the pattern is rewritten as an explicit
// ASCII character class, while each non-ASCII codepoint becomes a single marker
// byte standing for its category. Those ASCII classes are not the true
// categories. '~' is the one ASCII character listed in neither the punctuation
// class nor the symbol class, so it matches no category at all and splits
// differently from every other ASCII symbol -- " +++" stays one piece while
// " ~~~" becomes " " and "~~~". Reproduced here deliberately: agreeing with the
// reference tokenizer matters more than agreeing with Unicode.
bool joyai_is_letter(std::uint32_t c) {
    if(c<128)return (c>='A'&&c<='Z')||(c>='a'&&c<='z');
    return colibri::unicode::is_letter(c);
}
bool joyai_is_number(std::uint32_t c) {
    if(c<128)return c>='0'&&c<='9';
    return colibri::unicode::is_number(c);
}
bool joyai_is_accent_mark(std::uint32_t c) {
    // The collapse lists no sub-128 codepoints for \p{M}.
    return c<128?false:colibri::unicode::is_accent_mark(c);
}
bool joyai_is_punctuation(std::uint32_t c) {
    if(c<128)
        return (c>=0x21&&c<=0x23)||(c>=0x25&&c<=0x2A)||(c>=0x2C&&c<=0x2F)||
               (c>=0x3A&&c<=0x3B)||(c>=0x3F&&c<=0x40)||(c>=0x5B&&c<=0x5D)||
               c==0x5F||c==0x7B||c==0x7D;
    return colibri::unicode::is_punctuation(c);
}
bool joyai_is_symbol(std::uint32_t c) {
    if(c<128)
        return c==0x24||c==0x2B||(c>=0x3C&&c<=0x3E)||c==0x5E||c==0x60||c==0x7C;
    return colibri::unicode::is_symbol(c);
}

// "\p{N}{1,3}": digits break into groups of at most three.
std::size_t joyai_match_number(const std::vector<std::uint32_t>& code,std::size_t at) {
    std::size_t length=0;
    while(length<3&&at+length<code.size()&&joyai_is_number(code[at+length]))++length;
    return length;
}

// "[一-龥぀-ゟ゠-ヿ]+": CJK ideographs, hiragana and katakana stand apart from
// the Latin-oriented word rule below.
bool joyai_is_cjk(std::uint32_t codepoint) {
    return (codepoint>=0x4E00&&codepoint<=0x9FA5)||
           (codepoint>=0x3040&&codepoint<=0x309F)||
           (codepoint>=0x30A0&&codepoint<=0x30FF);
}

std::size_t joyai_match_cjk(const std::vector<std::uint32_t>& code,std::size_t at) {
    std::size_t length=0;
    while(at+length<code.size()&&joyai_is_cjk(code[at+length]))++length;
    return length;
}

// The third regex, an ordered alternation. Each branch is tried in the order it
// appears in the pattern and the first that matches wins:
//
//   [ASCII punctuation][A-Za-z]+
// | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
// | ?[\p{P}\p{S}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
std::size_t joyai_match_word(const std::vector<std::uint32_t>& code,std::size_t at) {
    namespace unicode=colibri::unicode;
    const auto size=code.size();
    const auto first=code[at];
    auto letter_or_mark=[](std::uint32_t c){return joyai_is_letter(c)||joyai_is_accent_mark(c);};
    auto punct_or_symbol=[](std::uint32_t c){return joyai_is_punctuation(c)||joyai_is_symbol(c);};
    // Every ASCII punctuation and symbol character, spelled out in the pattern
    // as a literal class rather than as \p{P}\p{S}.
    auto ascii_mark=[](std::uint32_t c){
        return (c>=0x21&&c<=0x2F)||(c>=0x3A&&c<=0x40)||(c>=0x5B&&c<=0x60)||(c>=0x7B&&c<=0x7E);
    };
    auto ascii_letter=[](std::uint32_t c){return (c>='A'&&c<='Z')||(c>='a'&&c<='z');};

    if(ascii_mark(first)){
        std::size_t length=1;
        while(at+length<size&&ascii_letter(code[at+length]))++length;
        if(length>1)return length;
    }
    // The optional leading character is greedy, so the branch that consumes it
    // is tried before the branch that does not.
    for(int leading=1;leading>=0;--leading){
        if(leading&&(first=='\r'||first=='\n'||unicode::is_letter(first)||punct_or_symbol(first)))
            continue;
        const std::size_t start=at+static_cast<std::size_t>(leading);
        std::size_t length=0;
        while(start+length<size&&letter_or_mark(code[start+length]))++length;
        if(length)return static_cast<std::size_t>(leading)+length;
    }
    for(int leading=1;leading>=0;--leading){
        if(leading&&first!=' ')continue;
        const std::size_t start=at+static_cast<std::size_t>(leading);
        std::size_t length=0;
        while(start+length<size&&punct_or_symbol(code[start+length]))++length;
        if(!length)continue;
        std::size_t tail=0;
        while(start+length+tail<size&&
              (code[start+length+tail]=='\r'||code[start+length+tail]=='\n'))++tail;
        return static_cast<std::size_t>(leading)+length+tail;
    }
    std::size_t run=0;
    while(at+run<size&&unicode::is_whitespace(code[at+run]))++run;
    if(run){
        // "\s*[\r\n]+" backtracks so the match ends on the last line break in
        // the whitespace run.
        std::size_t through_break=0;
        for(std::size_t index=0;index<run;++index)
            if(code[at+index]=='\r'||code[at+index]=='\n')through_break=index+1;
        if(through_break)return through_break;
        // "\s+(?!\S)" forbids a non-space immediately after the match, so a run
        // that is followed by text hands its last character back to that text.
        const std::size_t trimmed=(at+run<size)?run-1:run;
        if(trimmed)return trimmed;
        return run;  // "\s+"
    }
    return 0;
}

// DeepSeek-V4's `joyai-llm` pre-tokenizer, transcribed from the three regexes
// the reference applies in sequence. Unicode categories come from a generated
// table (unicode_categories.h) rather than the codepoint-range approximation
// the other pre-tokenizers use: this model is CJK-heavy and the pattern draws
// explicit \p{L}/\p{M}/\p{P}/\p{S} distinctions that the approximation blurs.
std::vector<std::string> deepseek4_pretokenize(const std::string& text) {
    std::vector<std::string> pieces{text};
    pieces=gguf_regex_split(pieces,joyai_match_number);
    pieces=gguf_regex_split(pieces,joyai_match_cjk);
    pieces=gguf_regex_split(pieces,joyai_match_word);
    return pieces;
}

// BPE over one pre-tokenized piece, appending its ids to `result`.
void gguf_bpe_piece(const ColibriV2Model& m, const std::string& piece,
                    std::vector<uint32_t>& result) {
    const std::string encoded=gguf_byte_encode(piece.c_str());
    auto symbols=gguf_utf8_symbols(encoded);
    if(symbols.empty())return;
    const auto&ranks=m.merge_ranks;
    struct Candidate{int rank;int left;int right;};
    struct Later{bool operator()(const Candidate&a,const Candidate&b)const{return a.rank!=b.rank?a.rank>b.rank:a.left>b.left;}};
    std::vector<int> next(symbols.size()),previous(symbols.size());
    std::vector<bool> alive(symbols.size(),true);
    for(size_t i=0;i<symbols.size();i++){next[i]=static_cast<int>(i)+1;previous[i]=static_cast<int>(i)-1;}
    next[symbols.size()-1]=-1;
    std::priority_queue<Candidate,std::vector<Candidate>,Later> queue;
    auto propose=[&](int left){
        if(left<0)return;
        const int right=next[left];
        if(right<0)return;
        const auto it=ranks.find(symbols[left]+" "+symbols[right]);
        if(it!=ranks.end())queue.push({it->second,left,right});
    };
    for(size_t i=0;i+1<symbols.size();i++)propose(static_cast<int>(i));
    while(!queue.empty()){
        const auto candidate=queue.top();queue.pop();
        const int left=candidate.left,right=candidate.right;
        if(!alive[left]||!alive[right]||next[left]!=right)continue;
        const auto it=ranks.find(symbols[left]+" "+symbols[right]);
        if(it==ranks.end()||it->second!=candidate.rank)continue;
        symbols[left]+=symbols[right];
        alive[right]=false;
        next[left]=next[right];
        if(next[left]>=0)previous[next[left]]=left;
        propose(previous[left]);
        propose(left);
    }
    for(int index=0;index>=0;index=next[index]){
        const auto it=m.vocabulary_ids.find(symbols[index]);
        result.push_back(it==m.vocabulary_ids.end()?0:it->second);
    }
}

// Pre-tokenizer boundaries, so the split can be checked against the reference
// patterns without a vocabulary in the way. Writes the byte offset each piece
// starts at plus a trailing end offset, so `count` is one more than the number
// of pieces. A null `offsets` asks for the count alone.
int colibri_v2_pretokenize(const ColibriV2Model*m,const char*text,uint64_t*offsets,uint64_t capacity,uint64_t*count){return guarded([&]{
    if(!m||!text||!count)throw std::runtime_error("invalid pretokenize arguments");
    const std::string input(text);
    const auto pieces=m->tokenizer_pre=="joyai-llm"
        ?deepseek4_pretokenize(input)
        :laguna_pretokenize(input);
    *count=pieces.size()+1;
    if(!offsets)return 0;
    if(capacity<*count)throw std::runtime_error("pretokenize output buffer is too small");
    std::uint64_t at=0,index=0;
    for(const auto& piece:pieces){offsets[index++]=at;at+=piece.size();}
    offsets[index]=at;
    return 0;});}

int colibri_v2_tokenize(const ColibriV2Model*m,const char*text,uint32_t*tokens,uint64_t capacity,uint64_t*count){return guarded([&]{
    if(!m||!text||!count)throw std::runtime_error("invalid tokenize arguments");
    if(m->tokenizer_pre=="laguna"||m->tokenizer_pre=="joyai-llm"){
        // Control tokens are split out by exact match first: they are ordinary
        // text to BPE, and Laguna spells them with characters whose merges would
        // never reassemble the single reserved id.
        std::vector<uint32_t> result;
        const std::string input(text);
        size_t at=0,plain=0;
        auto flush=[&](size_t end){
            if(end<=plain)return;
            const auto segment=input.substr(plain,end-plain);
            const auto pieces=m->tokenizer_pre=="joyai-llm"
                ?deepseek4_pretokenize(segment)
                :laguna_pretokenize(segment);
            for(const auto& piece:pieces)gguf_bpe_piece(*m,piece,result);
        };
        while(at<input.size()){
            const auto* match=static_cast<const std::pair<std::string,std::uint32_t>*>(nullptr);
            for(const auto& control:m->control_tokens)
                if(input.compare(at,control.first.size(),control.first)==0){match=&control;break;}
            if(!match){++at;continue;}
            flush(at);
            result.push_back(match->second);
            at+=match->first.size();
            plain=at;
        }
        flush(input.size());
        *count=result.size();
        if(capacity<result.size()||!tokens)throw std::runtime_error("token output buffer is too small");
        std::copy(result.begin(),result.end(),tokens);
        return 0;
    }
    const std::string encoded=gguf_byte_encode(text);
    auto symbols=gguf_utf8_symbols(encoded);
    // Greedy BPE with the same semantics as a full rescan per merge (lowest
    // merge rank first, leftmost pair on ties) but heap-driven over a linked
    // list: O(n log n). The naive rescan was O(n^2) and took minutes on the
    // ~100KB system prompts agentic clients send.
    const auto&ranks=m->merge_ranks;
    struct Candidate{int rank;int left;int right;};
    struct Later{bool operator()(const Candidate&a,const Candidate&b)const{return a.rank!=b.rank?a.rank>b.rank:a.left>b.left;}};
    std::vector<int> next(symbols.size()),previous(symbols.size());
    std::vector<bool> alive(symbols.size(),true);
    for(size_t i=0;i<symbols.size();i++){next[i]=static_cast<int>(i)+1;previous[i]=static_cast<int>(i)-1;}
    if(!symbols.empty())next[symbols.size()-1]=-1;
    std::priority_queue<Candidate,std::vector<Candidate>,Later> queue;
    auto propose=[&](int left){
        if(left<0)return;
        const int right=next[left];
        if(right<0)return;
        const auto it=ranks.find(symbols[left]+" "+symbols[right]);
        if(it!=ranks.end())queue.push({it->second,left,right});
    };
    for(size_t i=0;i+1<symbols.size();i++)propose(static_cast<int>(i));
    while(!queue.empty()){
        const auto candidate=queue.top();queue.pop();
        const int left=candidate.left,right=candidate.right;
        if(!alive[left]||!alive[right]||next[left]!=right)continue;
        // Stale heap entries survive after a neighbor merged; the pair text
        // may have changed, so re-validate its rank before applying.
        const auto it=ranks.find(symbols[left]+" "+symbols[right]);
        if(it==ranks.end()||it->second!=candidate.rank)continue;
        symbols[left]+=symbols[right];
        alive[right]=false;
        next[left]=next[right];
        if(next[left]>=0)previous[next[left]]=left;
        propose(previous[left]);
        propose(left);
    }
    std::vector<uint32_t> result;
    for(int index=symbols.empty()?-1:0;index>=0;index=next[index]){
        const auto it=m->vocabulary_ids.find(symbols[index]);
        result.push_back(it==m->vocabulary_ids.end()?0:it->second);
    }
    *count=result.size();
    if(capacity<result.size()||!tokens)throw std::runtime_error("token output buffer is too small");
    std::copy(result.begin(),result.end(),tokens);
    return 0;
});}
int colibri_v2_tensor_read(const ColibriV2Model* m,uint64_t i,void* dst,uint64_t bytes){return guarded([&]{if(!m||m->read_tensor(i,dst,bytes)!=0)throw std::runtime_error("invalid tensor read");return 0;});}
int colibri_v2_tensor_read_slice(const ColibriV2Model*m,uint64_t i,uint64_t offset,void*dst,uint64_t bytes){return guarded([&]{if(!m||!dst||i>=m->tensors.size())throw std::runtime_error("invalid tensor slice read");const auto&t=m->tensors[i];if(offset>t.size||bytes>t.size-offset)throw std::runtime_error("tensor slice is out of bounds");std::memcpy(dst,tensor_data(*m,t)+offset,static_cast<size_t>(bytes));return 0;});}
int colibri_v2_tensor_view(const ColibriV2Model*m,uint64_t i,uint64_t offset,uint64_t bytes,const void**out){return guarded([&]{if(!m||!out||i>=m->tensors.size())throw std::runtime_error("invalid tensor view");const auto&t=m->tensors[i];if(offset>t.size||bytes>t.size-offset)throw std::runtime_error("tensor view is out of bounds");*out=tensor_data(*m,t)+offset;return 0;});}
int colibri_v2_qwen_runtime_create(ColibriV2Model*m,const ColibriV2QwenRuntimeOptions*options,ColibriV2QwenRuntime**out){return guarded([&]{
    if(!m||!out)throw std::runtime_error("model and runtime output are required");
    const bool gemma4=m->config.architecture=="gemma4";
    const bool laguna=m->config.architecture=="laguna";
    // DeepSeek-V4 loads and describes itself but has no execution path yet, so
    // it gets its own message rather than looking like an unknown format.
    if(m->config.architecture=="deepseek4")throw std::runtime_error(
        "deepseek4 checkpoints load and report their configuration, but the native runtime "
        "cannot execute them yet (no hyper-connection, compressed-attention or indexer path)");
    if(m->config.architecture.find("qwen")!=0&&!gemma4&&!laguna)throw std::runtime_error("native runtime supports Qwen, Gemma 4 and Laguna models");
    // Dense checkpoints report no experts at all, so only require a routing
    // width from the ones that actually route.
    const bool dense_ffn=has_tensor(*m,"blk.0.ffn_gate.weight");
    if(!m->config.hidden_size||!m->config.layer_count)throw std::runtime_error("Qwen runtime config is incomplete");
    if(!dense_ffn&&(!m->config.expert_count||!m->config.expert_used_count))throw std::runtime_error("Qwen runtime config is incomplete");
    auto runtime=std::make_unique<ColibriV2QwenRuntime>();
    runtime->model=m;
    runtime->options=options?*options:ColibriV2QwenRuntimeOptions{};
    if(runtime->options.device<0)throw std::runtime_error("Qwen runtime device must be non-negative");
    if(!colibri::v2::valid_expert_execution_mode(runtime->options.moe_device))
        throw std::runtime_error("Qwen runtime MoE device is invalid");
    runtime->expert_mode=
        colibri::v2::expert_execution_mode(runtime->options.moe_device);
    // Every expert placement other than `cpu` exists to divide work between a
    // host and a device. With no device there is nothing to divide, and the
    // resident/hybrid/auto paths degrade into copying expert weights from the
    // GGUF mapping into another host buffer before computing on them -- 13x
    // slower than running them in place. `auto` in particular is the serving
    // default, so leaving this to the caller means the default configuration is
    // the pathological one.
    if(colibri_backend_is_cpu()&&
       runtime->expert_mode!=colibri::v2::ExpertExecutionMode::cpu){
        std::fprintf(stderr,
            "[colibri-v2] expert placement forced to CPU: the CPU backend has "
            "no device to page experts to\n");
        runtime->expert_mode=colibri::v2::ExpertExecutionMode::cpu;
        runtime->options.moe_device=
            colibri::v2::expert_execution_mode_value(runtime->expert_mode);
    }
    if(runtime->options.mtp_drafts>8)throw std::runtime_error("native Qwen MTP supports at most 8 drafts");
    if(gemma4&&runtime->options.mtp_drafts)throw std::runtime_error("native Gemma 4 MTP is not implemented");
    if(gemma4&&qwen_expert_policy(
            *runtime,colibri::v2::ExpertExecutionPhase::prepare
        ).is_streamed_gpu())
        throw std::runtime_error("native Gemma 4 supports --moe-device cpu or hybrid");
    if(laguna&&runtime->options.mtp_drafts)throw std::runtime_error("native Laguna MTP is not implemented");
    if(gemma4&&m->config.per_layer_embedding_size)throw std::runtime_error("native Gemma 4 per-layer embeddings are not implemented");
    if(gemma4&&m->config.shared_kv_layers)throw std::runtime_error("native Gemma 4 shared-KV tail layers are not implemented");
    if(runtime->options.expert_top_k>m->config.expert_used_count)throw std::runtime_error("native Qwen expert_top_k cannot exceed the model's trained expert_used_count");
    if(runtime->options.expert_top_p<0.0f||runtime->options.expert_top_p>1.0f)throw std::runtime_error("native Qwen expert_top_p must be within [0, 1]");
    if(runtime->options.prefill_cache_seed>256)throw std::runtime_error("native Qwen prefill_cache_seed supports at most 256 experts per layer");
    if(runtime->options.cpu_prefetch_auto>1)throw std::runtime_error("native Qwen cpu_prefetch_auto must be boolean");
    if(runtime->options.cpu_prefetch_mib&&runtime->options.cpu_prefetch_auto)throw std::runtime_error("native Qwen CPU prefetch modes are mutually exclusive");
    if(runtime->options.next_layer_prefetch>64)throw std::runtime_error("native Qwen next-layer prefetch supports at most 64 experts");
    if(runtime->options.hybrid_prefill_cpu>1)
        throw std::runtime_error("native Qwen hybrid prefill policy is invalid");
    if(runtime->options.immutable_residency>1)
        throw std::runtime_error("native Qwen expert residency policy is invalid");
    if(runtime->options.prefill_cache_seed_auto>1)
        throw std::runtime_error("native Qwen prefill cache seed auto policy is invalid");
    if(runtime->options.strict_resident>1)
        throw std::runtime_error("native Qwen strict resident policy is invalid");
    if(runtime->options.dense_requant>2)
        throw std::runtime_error("native Qwen dense requant policy is invalid");
    if(runtime->options.strict_resident&&
       runtime->expert_mode!=colibri::v2::ExpertExecutionMode::streamed_gpu)
        throw std::runtime_error(
            "native Qwen strict resident policy requires streamed GPU execution");
     if(gemma4&&runtime->options.next_layer_prefetch)throw std::runtime_error("native Gemma 4 next-layer prefetch is not implemented");
     if(runtime->options.next_layer_prefetch&&runtime->options.mtp_drafts)throw std::runtime_error("native Qwen next-layer prefetch does not support MTP yet");
    if(runtime->options.cache_type_k<0||runtime->options.cache_type_k>6)throw std::runtime_error("native Qwen cache_type_k must be 0 (f32), 1 (f16), 2 (bf16), 3 (q8_0), 4 (turbo3), 5 (turbo4), or 6 (auto)");
    if(runtime->options.cache_type_v<0||runtime->options.cache_type_v>6)throw std::runtime_error("native Qwen cache_type_v must be 0 (f32), 1 (f16), 2 (bf16), 3 (q8_0), 4 (turbo3), 5 (turbo4), or 6 (auto)");
    if(!runtime->options.context_limit)runtime->options.context_limit=m->config.context_length?m->config.context_length:4096;
    if(gemma4)build_gemma4_plan(*runtime);
    else if(laguna)build_laguna_plan(*runtime);
    else build_qwen_plan(*runtime);
    // Resolve cache type `auto`. Must run after the layer plan, which is what
    // supplies head_dim; the rule and its measurements live in
    // colibri::v2::attention so they can be pinned by a contract test.
    if(runtime->options.cache_type_k==colibri::v2::attention::kCacheTypeAuto||
       runtime->options.cache_type_v==colibri::v2::attention::kCacheTypeAuto){
        bool head_dim_ok=true;
        for(const auto&layer:runtime->layers){
            if(!layer.attention)continue;
            if(!colibri::v2::attention::turbo_head_dim_ok(layer.head_dim)){
                head_dim_ok=false;break;
            }
        }
        const auto resolved=colibri::v2::attention::auto_cache_type(
            runtime->options.context_limit,
            runtime->model->config.expert_count>0,
            head_dim_ok);
        if(runtime->options.cache_type_k==colibri::v2::attention::kCacheTypeAuto)
            runtime->options.cache_type_k=resolved;
        if(runtime->options.cache_type_v==colibri::v2::attention::kCacheTypeAuto)
            runtime->options.cache_type_v=resolved;
    }
    if(runtime->options.next_layer_prefetch&&runtime->layers.size()>1){
        const auto experts=static_cast<std::size_t>(m->config.expert_count);
        runtime->expert_transitions.resize(
            (runtime->layers.size()-1)*experts*experts
        );
    }
    if(runtime->options.mtp_drafts&&!runtime->mtp_available)throw std::runtime_error("native Qwen MTP was requested but the model has no draft block");
    // Prompt tokens are processed through the batched rows forward in chunks
    // of this size. Bigger chunks amortize expert weight reads further (the
    // CPU MoE reads each routed expert once per chunk); the cost is ~200MB
    // of workspace + pinned staging at 1024. 0 or 1 falls back to
    // one-token-at-a-time decode.
    // Gemma 4 prefill uses a single-row path for now. Raise to align
    // with the Qwen default when batching is supported.
    // Dense low-bit checkpoints need a much wider gate/up scratch than MoE,
    // and their quantized row kernels amortize weights in small token tiles.
    // A 1024-row allocation consumes ~633 MiB for Qwen3.6-27B and can evict
    // several whole FFN blocks to the CPU. 64 rows preserves useful batching
    // without making decode pay that residency penalty.
    // Gemma 4 has no rows forward. Laguna does, but its leading dense block is
    // 12x the expert width, so the row scratch is far larger per token than a
    // pure-MoE Qwen checkpoint's; 256 keeps the batch worth batching without
    // sizing every scratch region for a 1024-row dense SwiGLU.
    runtime->prefill_rows=gemma4?1:(laguna?256:(dense_ffn?64:1024));
    if(const char*env=std::getenv("COLIBRI_PREFILL_ROWS")){
        const long value=std::strtol(env,nullptr,10);
        runtime->prefill_rows=static_cast<std::uint32_t>(std::clamp<long>(value,0,4096));
    }
    // Mid-prefill recurrent-state checkpoints let a follow-up request whose
    // prompt diverges mid-stream (agentic clients mutate the prefix: injected
    // reminders, re-rendered tool calls) resume from the nearest checkpoint <=
    // the divergence point instead of reprefilling the whole prompt. `interval`
    // is the position of the first checkpoint; the rest are geometric
    // (interval<<k) so early coverage (the stable system+tools prefix) is dense.
    // `slots` is the total snapshot pool (one reserved for the exact
    // end-of-prompt snapshot). interval=0 disables mid checkpoints (end
    // snapshots only, legacy behavior); slots=0 falls back to the default.
    runtime->prefill_checkpoint_interval=runtime->options.prefill_checkpoint_interval;
    const std::size_t checkpoint_slots=runtime->options.prefill_checkpoint_slots
        ? std::clamp<std::size_t>(runtime->options.prefill_checkpoint_slots,1,256):4;
    runtime->prefill_snapshots.assign(checkpoint_slots,QwenPrefillSnapshot{});
    // Independent decode slots (llama.cpp --parallel): 0/1 = single-sequence.
    runtime->parallel_sequences=std::clamp<std::uint32_t>(
        runtime->options.parallel_sequences?runtime->options.parallel_sequences:1,1,16);
    // Host RAM budget for the spilled-slot prompt cache. UINT32_MAX selects a
    // conservative automatic budget: one eighth of currently available RAM,
    // capped at llama-server's practical 8 GiB default. This leaves ample
    // headroom for the model mapping, filesystem cache, and request buffers.
    if(runtime->options.prompt_cache_mib==std::numeric_limits<std::uint32_t>::max()){
        const auto available=available_host_memory();
        runtime->host_cache_limit_bytes=std::min<std::uint64_t>(
            8ull*1024*1024*1024,available/8
        );
        runtime->host_cache_limit_bytes=
            (runtime->host_cache_limit_bytes/(1024*1024))*(1024*1024);
    }else{
        runtime->host_cache_limit_bytes=
            static_cast<std::uint64_t>(runtime->options.prompt_cache_mib)*1024ull*1024;
    }
    if(runtime->host_cache_limit_bytes)
        std::fprintf(stderr,"[colibri-v2] prompt cache budget %.1f GiB%s\n",
            runtime->host_cache_limit_bytes/static_cast<double>(1024ull*1024*1024),
            runtime->options.prompt_cache_mib==std::numeric_limits<std::uint32_t>::max()
                ? " (auto)":"");
    runtime->cuda_ready=false;
    runtime->decode_ready=false;
    *out=runtime.release();
    return 0;
});}
void colibri_v2_qwen_runtime_destroy(ColibriV2QwenRuntime*runtime){try{if(runtime)release_qwen_device(*runtime);delete runtime;}catch(...){}}
int colibri_v2_qwen_runtime_info(const ColibriV2QwenRuntime*runtime,ColibriV2QwenRuntimeInfo*out){return guarded([&]{
    if(!runtime||!out)throw std::runtime_error("invalid Qwen runtime info handle");
    std::memset(out,0,sizeof(*out));
    out->layers=static_cast<uint32_t>(runtime->layers.size());
    out->attention_layers=static_cast<uint32_t>(std::count_if(runtime->layers.begin(),runtime->layers.end(),[](const QwenLayerPlan&layer){return layer.attention;}));
    out->swa_layers=static_cast<uint32_t>(std::count_if(runtime->layers.begin(),runtime->layers.end(),[](const QwenLayerPlan&layer){return layer.attention_window!=0;}));
    out->sliding_window=runtime->model->config.sliding_window;
    out->swa_full=runtime->options.swa_full?1:0;
    out->deltanet_layers=out->layers-out->attention_layers;
    out->hidden_size=runtime->model->config.hidden_size;
    out->expert_count=runtime->model->config.expert_count;
    out->expert_used_count=runtime->model->config.expert_used_count;
    out->mtp_available=runtime->mtp_available?1:0;
    out->mtp_enabled=runtime->options.mtp_drafts?1:0;
    out->mtp_drafts=runtime->options.mtp_drafts;
    out->mtp_layer=runtime->mtp_available?runtime->model->mtp_layer:std::numeric_limits<std::uint32_t>::max();
    out->context_limit=runtime->options.context_limit;
    out->resolved_cache_type_k=runtime->options.cache_type_k;
    out->resolved_cache_type_v=runtime->options.cache_type_v;
    out->static_tensor_bytes=runtime->static_tensor_bytes;
    out->expert_tensor_bytes=runtime->expert_tensor_bytes;
    const std::uint64_t sequence_slots=std::max<std::size_t>(1,runtime->sequences.size());
    out->gpu_allocated_bytes=runtime->static_arena_bytes+runtime->workspace_bytes+
        sequence_slots*runtime->state_bytes+runtime->expert_staging_bytes+
        runtime->expert_cache_bytes+runtime->expert_native_cache_bytes+
        sequence_slots*runtime->prefill_snapshots.size()*
            runtime->prefill_snapshot_bytes;
    out->workspace_bytes=runtime->workspace_bytes;
    out->state_bytes=runtime->state_bytes;
    out->expert_staging_bytes=runtime->expert_staging_bytes;
    out->expert_cache_bytes=runtime->expert_cache_bytes;
    out->expert_cache_slots=runtime->expert_slots.size();
    out->expert_cache_hits=runtime->expert_cache_hits;
    out->expert_cache_misses=runtime->expert_cache_misses;
    out->expert_cache_evictions=runtime->expert_cache_evictions;
    out->expert_cache_admissions=runtime->expert_cache_admissions;
    out->expert_cache_rejections=runtime->expert_cache_rejections;
    out->expert_cache_prompt_bypasses=runtime->expert_cache_prompt_bypasses;
    out->prefix_cache_hits=runtime->prefix_cache_hits;
    out->prefix_cache_misses=runtime->prefix_cache_misses;
    out->prefix_cache_reused_tokens=runtime->prefix_cache_reused_tokens;
    out->prefix_cache_reprefilled_tokens=runtime->prefix_cache_reprefilled_tokens;
    out->prefix_cache_last_prompt_tokens=runtime->prefix_cache_last_prompt_tokens;
    out->prefix_cache_last_reused_tokens=runtime->prefix_cache_last_reused_tokens;
    out->prefix_cache_last_lcp_live=runtime->prefix_cache_last_lcp_live;
    out->prefix_cache_last_lcp_snapshot=runtime->prefix_cache_last_lcp_snapshot;
    out->prompt_cache_entries=runtime->host_prompts.size();
    out->prompt_cache_used_bytes=runtime->host_cache_used_bytes;
    out->mtp_tensor_bytes=runtime->mtp_tensor_bytes;
    out->mtp_draft_tokens=runtime->mtp_draft_tokens;
    out->mtp_accepted_tokens=runtime->mtp_accepted_tokens;
    out->mtp_rejected_tokens=runtime->mtp_rejected_tokens;
    out->mtp_draft_nanoseconds=runtime->mtp_draft_nanoseconds;
    out->mtp_verify_nanoseconds=runtime->mtp_verify_nanoseconds;
    out->mtp_rollback_nanoseconds=runtime->mtp_rollback_nanoseconds;
    out->decode_calls=runtime->decode_calls;
    out->decode_nanoseconds=runtime->decode_nanoseconds;
    out->route_wait_nanoseconds=runtime->route_wait_nanoseconds;
    out->expert_page_nanoseconds=runtime->expert_page_nanoseconds;
    out->tail_wait_nanoseconds=runtime->tail_wait_nanoseconds;
    out->position=runtime->position;
    out->device=runtime->options.device;
    out->moe_device=
        colibri::v2::expert_execution_mode_value(runtime->expert_mode);
    out->cuda_ready=runtime->cuda_ready?1:0;
    out->decode_ready=runtime->decode_ready?1:0;
    out->route_expert_sum=runtime->route_expert_sum;
    out->expert_compute_nanoseconds=runtime->expert_compute_nanoseconds;
    out->prefill_cache_seeded_experts=runtime->prefill_cache_seeded_experts;
    out->prefill_cache_seed_nanoseconds=runtime->prefill_cache_seed_nanoseconds;
    out->direct_paging=runtime->dma_paging?1:0;
    out->paging_registration_nanoseconds=runtime->paging_registration_nanoseconds;
    out->host_available_bytes=runtime->host_available_bytes;
    out->cpu_prefetch_experts=runtime->cpu_prefetch_experts;
    out->cpu_prefetch_bytes=runtime->cpu_prefetch_bytes;
    out->cpu_prefetch_nanoseconds=runtime->cpu_prefetch_nanoseconds;
    out->cpu_prefetch_pages=runtime->cpu_prefetch_pages;
    out->cpu_prefetch_cold_pages=runtime->cpu_prefetch_cold_pages;
    out->cpu_prefetch_loaded_pages=runtime->cpu_prefetch_loaded_pages;
    out->cpu_prefetch_auto_skips=runtime->cpu_prefetch_auto_skips;
    out->cpu_prefetch_last_budget_bytes=runtime->cpu_prefetch_last_budget_bytes;
    out->prefill_calls=runtime->prefill_calls;
    out->prefill_tokens=runtime->prefill_tokens;
    out->prefill_nanoseconds=runtime->prefill_nanoseconds;
    out->prefill_route_wait_nanoseconds=runtime->prefill_route_wait_nanoseconds;
    out->prefill_expert_nanoseconds=runtime->prefill_expert_nanoseconds;
    out->prefill_direct_quant=qwen_prefill_direct_quant_enabled(*runtime);
    out->prefill_direct_quant_width=out->prefill_direct_quant?
        ((colibri_cpu_features()&2u)!=0?8:4):0;
    out->prefill_profile=runtime->prefill_profile?1:0;
    out->prefill_gpu_core_nanoseconds=runtime->prefill_gpu_core_nanoseconds;
    out->prefill_gpu_router_nanoseconds=runtime->prefill_gpu_router_nanoseconds;
    out->prefill_gpu_transfer_nanoseconds=runtime->prefill_gpu_transfer_nanoseconds;
    out->expert_history_loaded_entries=runtime->expert_history_loaded_entries;
    out->expert_history_saves=runtime->expert_history_saves;
    out->next_layer_prefetch_predictions=runtime->next_layer_prefetch_predictions;
    out->next_layer_prefetch_hits=runtime->next_layer_prefetch_hits;
    out->next_layer_prefetch_bytes=runtime->next_layer_prefetch_bytes;
    out->next_layer_prefetch_trained_pairs=runtime->next_layer_prefetch_trained_pairs;
    out->route_recurrence_observations=runtime->route_recurrence_observations;
    out->route_recurrence_prev_hits=runtime->route_recurrence_prev_hits;
    out->route_recurrence_window_hits=runtime->route_recurrence_window_hits;
    out->route_recurrence_layer_samples=runtime->route_recurrence_layer_samples;
    out->route_recurrence_window_experts=runtime->route_recurrence_window_experts;
    out->route_recurrence_resident=runtime->route_recurrence_resident;
    out->route_recurrence_miss_in_window=runtime->route_recurrence_miss_in_window;
    out->route_recurrence_miss_cold=runtime->route_recurrence_miss_cold;
    out->nvfp4_tensor_core_moe_calls=runtime->nvfp4_tensor_core_moe_calls;
    out->nvfp4_tensor_core_moe_fallbacks=runtime->nvfp4_tensor_core_moe_fallbacks;
    out->nvfp4_tensor_core_moe_last_status=runtime->nvfp4_tensor_core_moe_last_status;
    out->host_ffn_layers=runtime->host_ffn_layers;
    out->host_ffn_bytes=runtime->host_ffn_bytes;
    out->dense_host_nanoseconds=runtime->dense_host_nanoseconds;
    out->expert_cache_deferred_admissions=
        runtime->expert_cache_deferred_admissions;
    out->expert_residency_epochs=runtime->expert_residency_epochs;
    out->expert_residency_frozen=runtime->expert_residency_frozen?1:0;
    out->prefill_cache_seed_bytes=runtime->prefill_cache_seed_bytes;
    out->prefill_cache_seed_selected_experts=
        runtime->prefill_cache_seed_selected_experts;
    out->prefill_cache_seed_hits=runtime->prefill_cache_seed_hits;
    out->prefill_cache_seed_avoided_misses=
        runtime->prefill_cache_seed_avoided_misses;
    out->prefill_cache_seed_auto_skips=runtime->prefill_cache_seed_auto_skips;
    out->prefill_cache_seed_budget_stops=
        runtime->prefill_cache_seed_budget_stops;
    out->sampling_gpu_topk_calls=runtime->sampling_gpu_topk_calls;
    out->sampling_gpu_topk_bytes=runtime->sampling_gpu_topk_bytes;
    out->sampling_full_download_bytes=runtime->sampling_full_download_bytes;
    out->sampling_nanoseconds=runtime->sampling_nanoseconds;
    return 0;
});}
int colibri_v2_qwen_runtime_reset(ColibriV2QwenRuntime*runtime){return guarded([&]{if(!runtime)throw std::runtime_error("invalid Qwen runtime handle");if(runtime->state&&colibri_gpu_memset(runtime->state,0,runtime->state_bytes,runtime->stream)!=0)throw std::runtime_error("failed to reset native Qwen state");runtime->position=0;runtime->last_output_token=0;runtime->processed_tokens.clear();runtime->mtp_cache_tokens=0;runtime->mtp_has_target_hidden=false;runtime->cancelled=false;runtime->cache_admission_enabled=true;qwen_unfreeze_expert_residency(*runtime);return 0;});}
int colibri_v2_qwen_runtime_cancel(ColibriV2QwenRuntime*runtime){return guarded([&]{if(!runtime)throw std::runtime_error("invalid Qwen runtime handle");runtime->cancelled=true;return 0;});}
int colibri_v2_qwen_runtime_prepare(ColibriV2QwenRuntime*runtime){return guarded([&]{
    if(!runtime)throw std::runtime_error("invalid Qwen runtime handle");
    if(runtime->static_arena)return 0;
    if(colibri_gpu_init(runtime->options.device)!=0)throw std::runtime_error("failed to initialize native CUDA runtime");
    std::vector<std::string> option_storage;
#if !defined(_WIN32)
    for(const char*path:{"/opt/cuda/include","/usr/local/cuda/include","/usr/include"})if(access((std::string(path)+"/cuda_fp16.h").c_str(),R_OK)==0){option_storage.push_back(std::string("-I")+path);if(access((std::string(path)+"/cccl/cub/config.cuh").c_str(),R_OK)==0)option_storage.push_back(std::string("-I")+path+"/cccl");}
#endif
    if(const char*cuda_home=std::getenv("CUDA_HOME")){option_storage.push_back(std::string("-I")+cuda_home+"/include");option_storage.push_back(std::string("-I")+cuda_home+"/include/cccl");}
    std::vector<const char*>compile_options;for(const auto&option:option_storage)compile_options.push_back(option.c_str());
    std::array<char,16384>compile_log{};
    const std::string cuda_source=std::string(colibri::v2::qwen_cuda_source)+colibri::v2::qwen_native_cuda_source;
    if(colibri_gpu_compile(cuda_source.c_str(),compile_options.data(),static_cast<int32_t>(compile_options.size()),runtime->options.device,compile_log.data(),compile_log.size())!=0)throw std::runtime_error(std::string("failed to compile native Qwen CUDA kernels: ")+compile_log.data());
    runtime->cuda_ready=true;
    if(colibri_gpu_stream_create(&runtime->stream)!=0)throw std::runtime_error("failed to create native CUDA stream");
    if(colibri_gpu_stream_create(&runtime->prefetch_stream)!=0){colibri_gpu_stream_destroy(runtime->stream);throw std::runtime_error("failed to create native Qwen prefetch stream");}
    const char*graph_setting=std::getenv("COLIBRI_CUDA_GRAPHS");
    runtime->cuda_graphs=graph_setting&&graph_setting[0]=='1';
    if(runtime->cuda_graphs&&colibri_gpu_stream_create(&runtime->graph_stream)!=0)
        runtime->cuda_graphs=false;
    runtime->prefill_profile=std::getenv("COLIBRI_PREFILL_PROFILE")&&
        std::getenv("COLIBRI_PREFILL_PROFILE")[0]=='1';
    if(const char*fused=std::getenv("COLIBRI_FUSED_ATTENTION"))
        runtime->fused_attention=fused[0]!='0';
    else if(const char*fused=std::getenv("COLIBRI_FUSED_Q8_ATTENTION"))
        runtime->fused_attention=fused[0]!='0'; // legacy name
    if(const char*fused=std::getenv("COLIBRI_FUSED_MOE_GATE_UP"))
        runtime->fused_moe_gate_up=fused[0]!='0';
    if(const char*strict=std::getenv("COLIBRI_EXPERT_CACHE_STRICT_ADMISSION"))
        runtime->strict_cache_admission=strict[0]!='0';
    const int route_event_status=runtime->prefill_profile
        ?colibri_gpu_timed_event_create(&runtime->route_event)
        :colibri_gpu_event_create(&runtime->route_event);
    if(route_event_status!=0){release_qwen_device(*runtime);throw std::runtime_error("failed to create native Qwen route event");}
    if(runtime->prefill_profile&&(
       colibri_gpu_timed_event_create(&runtime->prefill_layer_start_event)!=0||
       colibri_gpu_timed_event_create(&runtime->prefill_core_end_event)!=0||
       colibri_gpu_timed_event_create(&runtime->prefill_router_end_event)!=0)){
        release_qwen_device(*runtime);
        throw std::runtime_error("failed to create native Qwen prefill profiling events");
    }
    if(const char*profile=std::getenv("COLIBRI_CUDA_PROFILE");profile&&profile[0]&&profile[0]!='0'){
        runtime->cuda_profile=true;
        runtime->cuda_layer_profiles.resize(runtime->layers.size());
        auto create_profile_event=[&](std::uint64_t&event){
            if(colibri_gpu_timed_event_create(&event)!=0){
                release_qwen_device(*runtime);
                throw std::runtime_error("failed to create native Qwen CUDA profiling event");
            }
        };
        for(auto&events:runtime->cuda_layer_profiles){
            create_profile_event(events.pre_start);create_profile_event(events.pre_end);
            create_profile_event(events.recurrent_start);create_profile_event(events.recurrent_end);
            create_profile_event(events.shared_start);create_profile_event(events.shared_end);
            create_profile_event(events.expert_start);create_profile_event(events.expert_end);
        }
        create_profile_event(runtime->cuda_tail_start);
        create_profile_event(runtime->cuda_lm_start);
        create_profile_event(runtime->cuda_lm_end);
        create_profile_event(runtime->cuda_tail_end);
    }
    try {

        // Byte cursor into the state arena. Attention KV regions size per the
        // configured cache precision (f32=4B, f16=2B/elem); DeltaNet conv/recurrent
        // state stays f32. Each region is 16-byte aligned so mixed f16/f32 regions
        // never leave a following f32 access misaligned.
        // KV region bytes per element count and cache type (q8_0 = 34B/32-elem block).
        auto kv_bytes=[](std::uint64_t elems,int type){return kv_type_bytes(elems,type);};
        const int ck_type=runtime->options.cache_type_k, cv_type=runtime->options.cache_type_v;
        std::uint64_t state_cursor=0;
        auto reserve=[&](std::uint64_t bytes)->std::uint64_t{const auto at=state_cursor;state_cursor=(state_cursor+bytes+15)/16*16;return at;};
        std::uint64_t max_vector=runtime->model->config.vocabulary_size;
        for(std::uint32_t layer_number=0;layer_number<runtime->layers.size();++layer_number){
            auto&layer=runtime->layers[layer_number];
            for(auto index:layer.static_tensors){const auto&t=runtime->model->tensors[index];if(t.shape.size()==2)max_vector=std::max(max_vector,t.shape[1]);}
            if(layer.attention){
                const auto&key=runtime->model->tensors[tensor_index(*runtime->model,"blk."+std::to_string(layer_number)+".attn_k.weight")];
                const auto head_dim=layer.head_dim;
                const auto batch_room=std::max<std::uint64_t>(runtime->prefill_rows,9);
                layer.cache_capacity=layer.attention_window&&!runtime->options.swa_full
                    ?std::min<std::uint64_t>(runtime->options.context_limit,static_cast<std::uint64_t>(layer.attention_window)+batch_room)
                    :runtime->options.context_limit;
                // The TurboQuant rotation is a Walsh-Hadamard butterfly over the
                // whole head, so head_dim has to be a power of two, and the
                // rotated row is staged in a fixed shared-memory scratch that
                // caps it at 512. Rejecting here beats silently corrupting the
                // cache on a 96- or 80-wide head.
                if((kv_type_is_turbo(ck_type)||kv_type_is_turbo(cv_type))
                   &&(head_dim<32||head_dim>512||(head_dim&(head_dim-1))!=0))
                    throw std::runtime_error(
                        "native Qwen turbo3/turbo4 KV cache needs a head_dim that is a power of two "
                        "between 32 and 512, but layer "+std::to_string(layer_number)+" has "
                        +std::to_string(head_dim));
                const auto cache_floats=layer.kv_heads*layer.cache_capacity*head_dim;
                layer.state_first=reserve(kv_bytes(cache_floats,ck_type));
                layer.state_second=reserve(kv_bytes(cache_floats,cv_type));
            }else{
                const auto&conv=runtime->model->tensors[tensor_index(*runtime->model,"blk."+std::to_string(layer_number)+".ssm_conv1d.weight")];
                const auto&a=runtime->model->tensors[tensor_index(*runtime->model,"blk."+std::to_string(layer_number)+".ssm_a")];
                const auto&norm=runtime->model->tensors[tensor_index(*runtime->model,"blk."+std::to_string(layer_number)+".ssm_norm.weight")];
                layer.state_first=reserve(conv.shape[0]*conv.shape[1]*sizeof(float));
                layer.state_second=reserve(a.shape[0]*norm.shape[0]*norm.shape[0]*sizeof(float));
            }
        }
        if(runtime->options.mtp_drafts){
            auto&layer=runtime->mtp_layer_plan;
            layer.cache_capacity=runtime->options.context_limit;
            const auto&key=runtime->model->tensors[layer.static_tensors[2]];
            const auto head_dim=key.shape[1]/runtime->model->config.attention_kv_heads;
            const auto cache_floats=runtime->model->config.attention_kv_heads*runtime->options.context_limit*head_dim;
            layer.state_first=reserve(kv_bytes(cache_floats,ck_type));
            layer.state_second=reserve(kv_bytes(cache_floats,cv_type));
            runtime->mtp_target_hidden_offset=reserve(runtime->model->config.hidden_size*sizeof(float));
            runtime->mtp_draft_hidden_offset=reserve(runtime->model->config.hidden_size*sizeof(float));
            runtime->mtp_verified_hidden_offset=reserve(
                8ULL*runtime->model->config.hidden_size*sizeof(float));
            runtime->mtp_snapshot_offset=state_cursor;
            const auto snapshot_start=state_cursor;
            for(std::uint32_t layer_number=0;layer_number<runtime->layers.size();++layer_number){
                auto&target=runtime->layers[layer_number];
                if(target.attention)continue;
                const auto&conv=runtime->model->tensors[tensor_index(*runtime->model,"blk."+std::to_string(layer_number)+".ssm_conv1d.weight")];
                const auto&a=runtime->model->tensors[tensor_index(*runtime->model,"blk."+std::to_string(layer_number)+".ssm_a")];
                const auto&norm=runtime->model->tensors[tensor_index(*runtime->model,"blk."+std::to_string(layer_number)+".ssm_norm.weight")];
                target.snapshot_first=reserve(conv.shape[0]*conv.shape[1]*sizeof(float));
                target.snapshot_second=reserve(a.shape[0]*norm.shape[0]*norm.shape[0]*sizeof(float));
            }
            runtime->mtp_snapshot_bytes=state_cursor-snapshot_start;
        }
        runtime->state_bytes=device_align(state_cursor);
        runtime->decode_workspace_layout=colibri::v2::workspace::qwen_decode(
            runtime->model->config.hidden_size,runtime->scratch_elements,
            runtime->model->config.expert_used_count,runtime->moe_intermediate,
            runtime->model->config.expert_count,
            runtime->model->config.vocabulary_size,
            runtime->model->config.attention_heads,
            runtime->options.context_limit);
        // The batched rows forward (MTP verification and chunked prefill)
        // needs workspace and host staging proportional to its row capacity.
        runtime->forward_rows_capacity=std::max<std::uint32_t>(runtime->prefill_rows,9);
        const std::uint64_t rows=runtime->forward_rows_capacity;
        const std::uint64_t hidden=runtime->model->config.hidden_size;
        const std::uint64_t top_k=runtime->model->config.expert_used_count;
        // The chunked DeltaNet prefill kernels need per-chunk intermediates, but
        // only for models that have DeltaNet layers at the head_dim they support.
        // Anything else leaves these regions empty and keeps the sequential path.
        std::uint64_t delta_value_heads=0;
        for(const auto&layer:runtime->layers){
            if(layer.attention)continue;
            const auto heads=runtime->model->tensors[layer.static_tensors[8]].shape[0];
            const auto dim=runtime->model->tensors[layer.static_tensors[9]].shape[0];
            if(dim==colibri::v2::workspace::kDeltaDim)delta_value_heads=heads;
            break;
        }
        runtime->delta_value_heads=static_cast<std::uint32_t>(delta_value_heads);
        runtime->rows_workspace_layout=colibri::v2::workspace::qwen_rows(
            rows,hidden,runtime->scratch_elements,top_k,
            runtime->moe_intermediate,runtime->model->config.expert_count,
            runtime->model->config.attention_heads,
            runtime->options.context_limit,delta_value_heads);
        runtime->rows_host_layout=
            colibri::v2::workspace::qwen_rows_host(
                rows,hidden,top_k,runtime->moe_intermediate);
        runtime->decode_host_layout=
            colibri::v2::workspace::qwen_decode_host(
                hidden,top_k,runtime->moe_intermediate);
        // Multi-sequence decode slices one decode workspace per sequence out of
        // the (rows-forward-sized) workspace; capacity is bounded by what fits.
        runtime->decode_slice_bytes=runtime->decode_workspace_layout.bytes;
        runtime->decode_host_block_bytes=runtime->decode_host_layout.bytes;
        runtime->workspace_bytes=device_align(std::max<std::uint64_t>(
            16ULL*1024*1024,
            std::max({max_vector*sizeof(float)*8,
                      runtime->decode_workspace_layout.bytes,
                      runtime->rows_workspace_layout.bytes})
        ));
        // The embedding table is a per-token gather, so it need not be resident:
        // staging just the row each token reads returns the whole
        // vocab x hidden matrix to the expert cache. Requires an untied lm_head
        // (a tied head multiplies by the full matrix every token) and rows that
        // divide evenly, which every 2-D vocab x hidden layout does. Decided
        // here because the dense spill accounting below and the arena sizing
        // both depend on it.
        //
        // Off by default: it trades VRAM for a per-token read of a random row
        // of the mapping, and host registration does not populate those pages,
        // so the read is a cold major fault on the critical path. Measured on
        // the reference SM120 laptop with Qwen3.6-35B-Fast-NVFP4 it buys 575
        // more expert slots and 30% fewer misses but adds ~1 ms/token to
        // route_wait, for ~2% net loss. Worth revisiting for cards where the
        // freed GiB changes the miss rate by much more than it does at an
        // 8 GiB budget. COLIBRI_EMBED_HOST=1 enables it.
        {
            const auto& table=runtime->model->tensors[runtime->token_embeddings];
            const auto vocabulary=runtime->model->config.vocabulary_size;
            const char* embed_env=std::getenv("COLIBRI_EMBED_HOST");
            runtime->embedding_row_bytes=
                (vocabulary&&table.size%vocabulary==0)?table.size/vocabulary:0;
            runtime->embeddings_host_resident=
                runtime->embedding_row_bytes!=0&&
                runtime->lm_head!=runtime->token_embeddings&&
                !runtime->gemma4&&
                embed_env&&embed_env[0]=='1';
            if(runtime->embeddings_host_resident)
                runtime->static_tensor_bytes-=table.size;
        }
        // A dense model has no experts to page, so every weight would otherwise
        // have to be resident and the card becomes a hard ceiling. Spill whole
        // blocks' feed-forward to the host until the static set fits, working
        // from the last block back; attention and the DeltaNet recurrence stay
        // on the GPU because they are far smaller and latency-critical.
        if(!runtime->layers.empty()&&runtime->layers.front().dense_ffn){
            std::uint64_t resident=0;
            for(const auto&layer:runtime->layers)for(auto tensor:layer.static_tensors)resident+=device_align(runtime->model->tensors[tensor].size);
            for(auto tensor:{runtime->token_embeddings,runtime->final_norm,runtime->lm_head}){
                if(tensor==runtime->token_embeddings&&runtime->embeddings_host_resident)continue;
                resident+=device_align(runtime->model->tensors[tensor].size);
            }
            if(runtime->mtp_available){
                for(auto tensor:runtime->mtp_layer_plan.static_tensors)resident+=device_align(runtime->model->tensors[tensor].size);
                for(auto tensor:runtime->mtp_special_tensors)resident+=device_align(runtime->model->tensors[tensor].size);
            }
            std::uint64_t budget=runtime->options.gpu_cache_bytes;
            if(!budget){
                ColibriV2GpuInfo gi{};
                if(gpu_probe(gi,runtime->options.device)==0&&gi.free_memory>0){
                    // Deliberately much tighter than the expert-cache auto-fit
                    // margin: that one reserves room for a cache this model does
                    // not have, whereas here the workspace, KV state, staging and
                    // an alignment headroom are all subtracted explicitly below.
                    // Every 96 MiB left unused here is another block whose
                    // feed-forward has to be re-read from RAM on every token.
                    const std::uint64_t margin=std::max<std::uint64_t>(384ull*1024*1024,gi.total_memory/32);
                    budget=gi.free_memory>margin?gi.free_memory-margin:0;
                }
            }
            // The workspace, KV state and staging still have to fit alongside
            // the weights, so reserve them before deciding what stays.
            // Headroom on top of the named arenas: per-tensor alignment padding
            // and the snapshot pool land on the same budget, and stopping the
            // spill exactly at the limit leaves the allocation to fail by a
            // rounding error.
            const std::uint64_t reserved=runtime->workspace_bytes
                +std::max<std::uint32_t>(1u,runtime->parallel_sequences)*runtime->state_bytes
                +runtime->expert_staging_bytes
                +std::max<std::uint64_t>(1024ull*1024,budget/64);
            const std::uint64_t weight_budget=budget>reserved?budget-reserved:0;
            // Mixed-quant checkpoints deliberately assign slower IQ formats
            // to important layers. Their CPU fallback is scalar today, while
            // Q2_K/Q3_K/Q5_K/Q6_K have AVX2/AVX-512 row kernels. Spill the
            // SIMD-capable blocks first instead of blindly taking the tail and
            // turning several IQ3_S/IQ4_XS layers into the decode bottleneck.
            auto spill_pass=[&](bool simd_only){
                for(auto layer=runtime->layers.rbegin();
                    layer!=runtime->layers.rend()&&resident>weight_budget;++layer){
                    if(!layer->dense_ffn||layer->ffn_on_host)continue;
                    const std::size_t ffn_base=layer->attention?7:10;
                    bool all_simd=true;
                    for(std::size_t role=1;role<=3;++role)
                        all_simd=all_simd&&qwen_simd_quant_type(
                            runtime->model->tensors[layer->static_tensors[ffn_base+role]].type);
                    if(simd_only&&!all_simd)continue;
                    std::uint64_t freed=0;
                    for(std::size_t role=1;role<=3;++role)
                        freed+=device_align(runtime->model->tensors[
                            layer->static_tensors[ffn_base+role]].size);
                    layer->ffn_on_host=true;
                    resident-=freed;
                    runtime->host_ffn_bytes+=freed;
                    ++runtime->host_ffn_layers;
                }
            };
            spill_pass(true);
            spill_pass(false);
            if(runtime->host_ffn_layers)
                std::fprintf(stderr,
                    "[colibri-v2] dense feed-forward: %u of %zu blocks on CPU (%llu MiB spilled)\n",
                    runtime->host_ffn_layers,runtime->layers.size(),
                    static_cast<unsigned long long>(runtime->host_ffn_bytes/(1024ull*1024)));
        }
        std::vector<bool> persistent(runtime->model->tensors.size(),false);
        std::vector<bool> preserve_bf16(runtime->model->tensors.size(),false);
        if(!runtime->embeddings_host_resident)
            persistent[runtime->token_embeddings]=true;
        persistent[runtime->final_norm]=true;
        persistent[runtime->lm_head]=true;
        if(runtime->rope_factors!=std::numeric_limits<std::uint64_t>::max())
            persistent[runtime->rope_factors]=true;
        for(const auto&layer:runtime->layers){
            const std::size_t ffn_base=qwen_ffn_base(*runtime,layer);
            for(std::size_t slot=0;slot<layer.static_tensors.size();++slot){
                // A spilled block reads its gate/up/down from the mapping, so
                // those three must not consume arena space.
                if(layer.ffn_on_host&&slot>ffn_base&&slot<=ffn_base+3)continue;
                persistent[layer.static_tensors[slot]]=true;
            }
        }
        if(runtime->options.mtp_drafts){
            for(auto tensor:runtime->mtp_layer_plan.static_tensors){persistent[tensor]=true;preserve_bf16[tensor]=true;}
            for(auto tensor:runtime->mtp_special_tensors){persistent[tensor]=true;preserve_bf16[tensor]=true;}
        }
        // NVFP4 checkpoints quantize only the routed experts and ship every
        // dense weight as bf16, which costs ~1.9x Q8_0's bytes for the same
        // values. Those weights are re-read in full on every token, so the
        // difference lands directly in route_wait -- on the reference SM120
        // laptop the NVFP4 build reads 2.72 GB before the router against the
        // Q6_K build's 1.53 GB, and that gap is the entire decode gap between
        // them. Requantize them once here instead. Q8_0 is what the K-quant
        // checkpoints already use for the same tensors, and every consumer
        // dispatches on the effective type, so nothing downstream changes.
        //
        // The routed experts are deliberately excluded: they are paged through
        // the expert cache from the mapping, not the static arena, and they are
        // the tensors NVFP4 exists to compress. The public dense_requant policy
        // controls this; COLIBRI_REQUANT_BF16 remains a legacy override only
        // while the policy is auto.
        runtime->device_tensor_types.resize(runtime->model->tensors.size());
        for(std::uint64_t index=0;index<runtime->model->tensors.size();++index)
            runtime->device_tensor_types[index]=runtime->model->tensors[index].type;
        {
            std::uint32_t requant_mode=runtime->options.dense_requant;
            if(requant_mode==0){
                const char* legacy=std::getenv("COLIBRI_REQUANT_BF16");
                if(legacy&&legacy[0]=='0')requant_mode=2;
                else if(legacy&&legacy[0]=='1')requant_mode=1;
            }
            std::uint64_t candidate_count=0;
            std::uint64_t persistent_bytes=0;
            for(std::uint64_t index=0;index<persistent.size();++index){
                if(!persistent[index])continue;
                const auto&tensor=runtime->model->tensors[index];
                persistent_bytes+=device_align(tensor.size);
                if(preserve_bf16[index]||tensor.type!=30)continue;
                std::uint64_t elements=1;
                for(auto dimension:tensor.shape)elements*=dimension;
                if(elements&&elements%32==0)++candidate_count;
            }
            std::uint64_t auto_budget=runtime->options.gpu_cache_bytes;
            if(requant_mode==0&&auto_budget==0){
                ColibriV2GpuInfo gi{};
                if(gpu_probe(gi,runtime->options.device)==0&&gi.free_memory>0){
                    const std::uint64_t margin=std::max<std::uint64_t>(
                        2048ull*1024*1024,gi.total_memory/8);
                    if(gi.free_memory>margin)auto_budget=gi.free_memory-margin;
                }
            }
            const std::uint64_t requant_slots=runtime->options.mtp_drafts?1:
                std::max<std::uint32_t>(1u,runtime->parallel_sequences);
            const std::uint64_t estimated_base=persistent_bytes+
                runtime->workspace_bytes+requant_slots*runtime->state_bytes;
            const auto requant_policy=qwen_expert_policy(
                *runtime,colibri::v2::ExpertExecutionPhase::prepare);
            const std::uint64_t useful_expert_cache=
                requant_policy.routed_gpu_execution_allowed()
                    ?runtime->expert_tensor_bytes:0;
            const std::uint64_t estimated_need=estimated_base+useful_expert_cache;
            const bool auto_pressure=candidate_count&&
                (auto_budget==0||estimated_need>auto_budget);
            const bool requantize=requant_mode==1||
                (requant_mode==0&&auto_pressure);
            if(requantize){
                for(std::uint64_t index=0;index<persistent.size();++index){
                    if(!persistent[index])continue;
                    if(preserve_bf16[index])continue;
                    const auto& tensor=runtime->model->tensors[index];
                    if(tensor.type!=30)continue;
                    std::uint64_t elements=1;
                    for(auto dimension:tensor.shape)elements*=dimension;
                    // A partial trailing block would need its own padding path;
                    // every Qwen dense weight divides evenly.
                    if(elements==0||elements%32)continue;
                    runtime->device_tensor_types[index]=8;
                    ++runtime->requantized_tensors;
                    runtime->requantized_saved_bytes+=
                        tensor.size-(elements/32)*kQ8BlockSize;
                }
            }else if(requant_mode==0&&candidate_count){
                std::fprintf(stderr,
                    "[colibri-v2] dense requant auto: kept %llu bf16 tensors "
                    "(estimated %llu MiB need within %llu MiB GPU budget)\n",
                    static_cast<unsigned long long>(candidate_count),
                    static_cast<unsigned long long>(estimated_need/(1024ull*1024)),
                    static_cast<unsigned long long>(auto_budget/(1024ull*1024)));
            }
        }
        // lm_head_type is captured from the checkpoint at plan time, so it has
        // to follow the head through requantization.
        runtime->lm_head_type=qwen_device_type(*runtime,runtime->lm_head);
        if(runtime->requantized_tensors){
            runtime->static_tensor_bytes-=runtime->requantized_saved_bytes;
            std::fprintf(stderr,
                "[colibri-v2] dense requant %s: %llu bf16 tensors to Q8_0 "
                "(%llu MiB freed)\n",
                runtime->options.dense_requant==1?"q8":"auto",
                static_cast<unsigned long long>(runtime->requantized_tensors),
                static_cast<unsigned long long>(
                    runtime->requantized_saved_bytes/(1024ull*1024)));
        }
        // Aliased tensors are read straight out of the GGUF mapping and need no
        // arena space; see qwen_alias_static_tensor.
        for(std::uint64_t index=0;index<persistent.size();++index)if(persistent[index]&&!qwen_alias_static_tensor(*runtime,index))runtime->static_arena_bytes+=device_align(qwen_device_tensor_size(*runtime,index));
        // Dense checkpoints have no routed experts, so there is nothing to size
        // a paging slot from -- and expert_count is zero, which would trap.
        std::uint64_t one_expert=0;
        if(runtime->model->config.expert_count)
            for(const auto&layer:runtime->layers){if(layer.dense_ffn)continue;std::uint64_t bytes=0;for(auto tensor:layer.expert_tensors)bytes+=runtime->model->tensors[tensor].size/runtime->model->config.expert_count;one_expert=std::max(one_expert,bytes);}
        runtime->expert_staging_bytes=device_align(one_expert*runtime->model->config.expert_used_count*2);
        runtime->host_staging_bytes=std::max(
            runtime->expert_staging_bytes,
            device_align(runtime->rows_host_layout.bytes));
        auto prepare_policy=qwen_expert_policy(
            *runtime,colibri::v2::ExpertExecutionPhase::prepare);
        runtime->multi_decode_capacity=1;
        const std::uint64_t decode_slots=runtime->options.mtp_drafts?1:std::max<std::uint32_t>(1u,runtime->parallel_sequences);
        // Gemma 4 supports independent sequence slots, but its hybrid expert
        // path currently schedules those slots sequentially. The Qwen
        // layer-overlapped multi-decode driver assumes separate gate/up/down
        // expert tensors and must not consume Gemma's fused Q4_0 bundles.
        if(decode_slots>1&&runtime->model->config.expert_count&&
           prepare_policy.routed_cpu_execution_allowed()&&
           !runtime->gemma4&&!runtime->laguna){
            const std::uint64_t by_workspace=runtime->workspace_bytes/runtime->decode_slice_bytes;
            runtime->multi_decode_capacity=static_cast<std::uint32_t>(
                std::min(std::min(decode_slots,by_workspace),std::uint64_t{4}));
            // Sequence-private host blocks sit BEFORE the shared paging area, so
            // the paging area keeps its full expert_staging_bytes capacity.
            if(runtime->multi_decode_capacity>1)
                runtime->host_staging_bytes=std::max(runtime->host_staging_bytes,
                    runtime->multi_decode_capacity*runtime->decode_host_block_bytes+runtime->expert_staging_bytes);
        }
        runtime->expert_slot_bytes=device_align(one_expert);
        // Prefill snapshot slots hold every DeltaNet layer's conv+recurrent
        // state; MTP manages its own snapshots inside the state arena, so
        // the prefix-reuse slots are only allocated without MTP.
        std::uint64_t snapshot_floats=0;
        if(!runtime->options.mtp_drafts)for(const auto&layer:runtime->layers){
            if(layer.attention)continue;
            const auto&conv=runtime->model->tensors[layer.static_tensors[6]];
            const auto&a=runtime->model->tensors[layer.static_tensors[8]];
            const auto&norm=runtime->model->tensors[layer.static_tensors[9]];
            snapshot_floats+=conv.shape[0]*conv.shape[1]+a.shape[0]*norm.shape[0]*norm.shape[0];
        }
        runtime->prefill_snapshot_bytes=snapshot_floats?device_align(snapshot_floats*sizeof(float)):0;
        // One KV+DeltaNet state arena per parallel decode slot. MTP manages its
        // own state inside the arena, so it stays single-slot.
        const std::size_t slot_count=runtime->options.mtp_drafts?1:std::max<std::uint32_t>(1u,runtime->parallel_sequences);
        runtime->sequences.assign(slot_count,QwenSequence{});
        runtime->slot_owner.assign(slot_count,-1);
        runtime->slot_events.assign(slot_count,0);
        for(auto&event:runtime->slot_events)
            if(colibri_gpu_event_create(&event)!=0)throw std::runtime_error("failed to create native Qwen slot events");
        if(colibri_gpu_event_create(&runtime->staging_event)!=0)throw std::runtime_error("failed to create native Qwen staging event");
        if(colibri_gpu_event_create(&runtime->prefetch_event)!=0){colibri_gpu_event_destroy(runtime->staging_event);throw std::runtime_error("failed to create native Qwen prefetch event");}
        const auto base_total=runtime->static_arena_bytes+runtime->workspace_bytes+slot_count*runtime->state_bytes+runtime->expert_staging_bytes+slot_count*runtime->prefill_snapshots.size()*runtime->prefill_snapshot_bytes;
        const char* nvfp4_persistent_env =
            std::getenv("COLIBRI_NVFP4_PERSISTENT");
        const bool persistent_nvfp4_eligible=
            runtime->model->config.expert_count&&
            std::all_of(runtime->layers.begin(),runtime->layers.end(),
                [&](const QwenLayerPlan&layer){
                    return layer.expert_tensors[0]<
                               runtime->model->tensors.size()&&
                           layer.expert_tensors[1]<
                               runtime->model->tensors.size()&&
                           layer.expert_tensors[2]<
                               runtime->model->tensors.size()&&
                           runtime->model->tensors[
                               layer.expert_tensors[0]].type==40&&
                           runtime->model->tensors[
                               layer.expert_tensors[1]].type==40&&
                           runtime->model->tensors[
                               layer.expert_tensors[2]].type==40;
                });
        const bool persistent_nvfp4_requested=
            persistent_nvfp4_eligible&&nvfp4_persistent_env&&
            nvfp4_persistent_env[0]=='1';
        // gpu_cache_bytes is the TOTAL GPU budget (base allocations + expert
        // cache). 0 = auto-fit: probe free VRAM and use most of it, leaving a
        // headroom margin. Any positive value is an exact manual budget.
        std::uint64_t gpu_budget=runtime->options.gpu_cache_bytes;
        const bool auto_fit=(gpu_budget==0);
        if(auto_fit&&prepare_policy.routed_gpu_execution_allowed()){
            ColibriV2GpuInfo gi{};
            if(gpu_probe(gi,runtime->options.device)==0&&gi.free_memory>0){
                // Leave headroom for the CUDA context, activations and other
                // apps. A flat margin starves tight cards (leaving <2 GiB free
                // measurably slows decode via memory pressure), so scale it
                // with total VRAM: max(2 GiB, 1/8 of the card).
                const std::uint64_t margin=std::max<std::uint64_t>(2048ull*1024*1024, gi.total_memory/8);
                if(gi.free_memory>margin)gpu_budget=gi.free_memory-margin;
            }
        }
        if(!runtime->options.strict_resident&&
           !auto_fit&&gpu_budget&&base_total>gpu_budget){
            auto mib=[](std::uint64_t b){return std::to_string(b/(1024ull*1024));};
            throw std::runtime_error(
                "native Qwen base CUDA allocations ("+mib(base_total)+" MiB = static weights "
                +mib(runtime->static_arena_bytes)+" + workspace "+mib(runtime->workspace_bytes)+" + "
                +std::to_string(slot_count)+"x KV slot "+mib(runtime->state_bytes)+" ("
                +mib(slot_count*runtime->state_bytes)+") + staging "+mib(runtime->expert_staging_bytes)
                +") exceed the --gpu-cache-mib budget ("+mib(gpu_budget)
                +" MiB), before any expert cache. Lower --parallel or --context-window, or raise --gpu-cache-mib.");
        }
        // expert_slot_bytes is zero for a dense model, which has no expert set
        // to cache and would trap on the slot arithmetic below.
        if(prepare_policy.routed_gpu_execution_allowed()&&
           gpu_budget>base_total&&runtime->expert_slot_bytes){
            auto available=gpu_budget-base_total;
            const std::uint64_t cache_copies=
                persistent_nvfp4_requested?2:1;
            const auto slot_budget=runtime->expert_slot_bytes*cache_copies;
            auto cache=(available/slot_budget)*runtime->expert_slot_bytes;
            // Never allocate more cache than the whole expert set (every
            // (layer,expert) resident => zero misses); saves VRAM on big GPUs.
            const auto max_cache=static_cast<std::uint64_t>(
                qwen_cache_layer_count(*runtime))*
                runtime->model->config.expert_count*runtime->expert_slot_bytes;
            if(cache>max_cache)cache=max_cache;
            const char*whole_layer_setting=
                std::getenv("COLIBRI_LAGUNA_WHOLE_LAYERS");
            const bool whole_layer_enabled=runtime->laguna&&
                qwen_model_has_iq_experts(*runtime)&&
                (!whole_layer_setting||whole_layer_setting[0]!='0');
            if(whole_layer_enabled&&!runtime->options.strict_resident){
                const auto experts=runtime->model->config.expert_count;
                std::vector<std::uint32_t> candidates;
                for(std::uint32_t layer=0;layer<runtime->layers.size();++layer)
                    if(!runtime->layers[layer].dense_ffn)
                        candidates.push_back(layer);
                std::size_t layer_count=experts
                    ?std::min<std::size_t>(
                        candidates.size(),cache/runtime->expert_slot_bytes/experts)
                    :0;
                char*end=nullptr;
                const auto requested=whole_layer_setting
                    ?std::strtoul(whole_layer_setting,&end,10):0;
                if(whole_layer_setting&&end!=whole_layer_setting&&requested)
                    layer_count=std::min<std::size_t>(layer_count,requested);
                runtime->whole_expert_layer_slots.assign(
                    runtime->layers.size(),-1);
                std::size_t slot=0;
                for(std::size_t index=0;index<layer_count;++index){
                    const auto layer=candidates[candidates.size()-layer_count+index];
                    runtime->whole_expert_layer_slots[layer]=
                        static_cast<std::int32_t>(slot);
                    slot+=experts;
                }
                cache=slot*runtime->expert_slot_bytes;
                if(!layer_count)
                    runtime->whole_expert_layer_slots.clear();
                std::fprintf(stderr,
                    "[colibri-v2] Laguna whole-layer placement selected %zu "
                    "layers (%zu expert bundles)\n",layer_count,slot);
            }
            runtime->expert_cache_bytes=(cache/runtime->expert_slot_bytes<runtime->model->config.expert_used_count)?0:cache;
        }
        const auto resident_expert_bytes=static_cast<std::uint64_t>(
            qwen_cache_layer_count(*runtime))*
            runtime->model->config.expert_count*runtime->expert_slot_bytes;
        if(runtime->options.strict_resident&&resident_expert_bytes&&
           runtime->expert_cache_bytes<resident_expert_bytes){
            auto mib=[](std::uint64_t bytes){
                return std::to_string((bytes+1024ull*1024-1)/(1024ull*1024));
            };
            const auto kv_bytes=slot_count*runtime->state_bytes;
            const auto snapshot_bytes=slot_count*
                runtime->prefill_snapshots.size()*runtime->prefill_snapshot_bytes;
            const auto required=runtime->static_arena_bytes+kv_bytes+
                runtime->workspace_bytes+runtime->expert_staging_bytes+
                snapshot_bytes+resident_expert_bytes*
                    (persistent_nvfp4_requested?2:1);
            throw std::runtime_error(
                "resident expert mode requires "+mib(required)+
                " MiB CUDA budget = static "+mib(runtime->static_arena_bytes)+
                " MiB + KV/state "+mib(kv_bytes)+
                " MiB + workspace "+mib(runtime->workspace_bytes)+
                " MiB + staging "+mib(runtime->expert_staging_bytes)+
                " MiB + snapshots "+mib(snapshot_bytes)+
                " MiB + experts "+mib(resident_expert_bytes)+
                (persistent_nvfp4_requested
                    ? " MiB + persistent NVFP4 mirror "+
                        mib(resident_expert_bytes)
                    : "")+
                " MiB; available budget is "+mib(gpu_budget)+
                " MiB. Resident mode never falls back to paging; use "
                "expert_mode='auto' or raise --gpu-cache-mib.");
        }
        // A dense model routes nothing, so the expert staging arena is empty --
        // and a zero-byte CUDA allocation is an error, not a no-op.
        if(colibri_gpu_alloc(runtime->static_arena_bytes,&runtime->static_arena)!=0||
           colibri_gpu_alloc(runtime->workspace_bytes,&runtime->workspace)!=0||
           (runtime->expert_staging_bytes&&colibri_gpu_alloc(runtime->expert_staging_bytes,&runtime->expert_staging)!=0)||
           (runtime->host_staging_bytes&&colibri_gpu_host_alloc(runtime->host_staging_bytes,&runtime->host_staging)!=0))throw std::runtime_error("failed to allocate native Qwen CUDA arenas");
        // A turbo cache is expanded to f16 one layer at a time so the cuBLAS
        // attention path can run on it, so this only has to hold the widest
        // attention layer's live window (K and V), not the whole cache.
        if(kv_type_is_turbo(runtime->options.cache_type_k)
           ||kv_type_is_turbo(runtime->options.cache_type_v)){
            std::uint64_t widest=0;
            for(const auto& layer:runtime->layers){
                if(!layer.attention)continue;
                widest=std::max<std::uint64_t>(widest,
                    static_cast<std::uint64_t>(layer.kv_heads)*layer.cache_capacity*layer.head_dim);
            }
            if(widest){
                runtime->turbo_kv_stage_stride=device_align(widest*sizeof(std::uint16_t));
                runtime->turbo_kv_stage_bytes=runtime->turbo_kv_stage_stride*2;
                if(colibri_gpu_alloc(runtime->turbo_kv_stage_bytes,&runtime->turbo_kv_stage)!=0)
                    throw std::runtime_error("failed to allocate native Qwen turbo KV staging");
            }
        }
        if(runtime->embeddings_host_resident){
            // Sized for the widest gather: the rows forward embeds a whole
            // prefill chunk in one launch, single-token decode uses row 0 only.
            const std::uint64_t capacity=
                std::max<std::uint64_t>(1,runtime->forward_rows_capacity);
            runtime->embedding_stage_bytes=
                device_align(capacity*runtime->embedding_row_bytes);
            if(colibri_gpu_alloc(runtime->embedding_stage_bytes,&runtime->embedding_stage)!=0||
               colibri_gpu_host_alloc(runtime->embedding_stage_bytes,&runtime->embedding_host)!=0||
               colibri_gpu_alloc(device_align(capacity*sizeof(std::uint32_t)),&runtime->embedding_row_index)!=0||
               colibri_gpu_event_create(&runtime->embedding_event)!=0)
                throw std::runtime_error(
                    "failed to allocate native Qwen embedding staging");
            // The rows kernel gathers by token id; the staged copy is already
            // in row order, so it is indexed by a fixed identity permutation.
            std::vector<std::uint32_t> identity(capacity);
            for(std::uint64_t row=0;row<capacity;++row)
                identity[row]=static_cast<std::uint32_t>(row);
            if(colibri_gpu_upload_sync(runtime->embedding_row_index,identity.data(),
                                       capacity*sizeof(std::uint32_t))!=0)
                throw std::runtime_error(
                    "failed to seed native Qwen embedding row index");
        }
        if(runtime->host_ffn_layers){
            // Pinned so the per-token round trip to the host SwiGLU is a plain DMA.
            // Only the DMA endpoints need pinning now: one hidden-sized vector
            // in, one out. The activation scratch lives on the heap.
            runtime->dense_host_bytes=device_align(
                2ull*runtime->model->config.hidden_size*sizeof(float));
            if(colibri_gpu_host_alloc(runtime->dense_host_bytes,&runtime->dense_host)!=0)
                throw std::runtime_error("failed to allocate native Qwen dense host scratch");
        }
        for(auto&seq:runtime->sequences)if(colibri_gpu_alloc(runtime->state_bytes,&seq.state)!=0)throw std::runtime_error("failed to allocate native Qwen sequence state");
        runtime->state=runtime->sequences[0].state;
        runtime->active_sequence=0;
        if(runtime->expert_cache_bytes&&colibri_gpu_alloc(runtime->expert_cache_bytes,&runtime->expert_cache)!=0)throw std::runtime_error("failed to allocate native Qwen expert cache");
        if(runtime->expert_cache_bytes&&persistent_nvfp4_requested){
            runtime->expert_native_cache_bytes=runtime->expert_cache_bytes;
            if(colibri_gpu_alloc(runtime->expert_native_cache_bytes,
                                 &runtime->expert_native_cache)!=0)
                throw std::runtime_error(
                    "failed to allocate persistent NVFP4 expert cache");
        }
        if(runtime->prefill_snapshot_bytes){
            // Slot 0's checkpoint pool lives in runtime->prefill_snapshots (the
            // active mirror); slots 1..N-1 own theirs in sequences[i].
            for(auto&snapshot:runtime->prefill_snapshots)if(colibri_gpu_alloc(runtime->prefill_snapshot_bytes,&snapshot.device)!=0)throw std::runtime_error("failed to allocate native Qwen prefill snapshots");
            for(std::size_t i=1;i<runtime->sequences.size();++i){
                runtime->sequences[i].prefill_snapshots.assign(runtime->prefill_snapshots.size(),QwenPrefillSnapshot{});
                for(auto&snapshot:runtime->sequences[i].prefill_snapshots)if(colibri_gpu_alloc(runtime->prefill_snapshot_bytes,&snapshot.device)!=0)throw std::runtime_error("failed to allocate native Qwen prefill snapshots");
            }
        }
        if(runtime->expert_slot_bytes)runtime->expert_slots.resize(runtime->expert_cache_bytes/runtime->expert_slot_bytes);
        // Hybrid is an optimization over the CPU expert path, not a hard
        // requirement. Auto-fit can legitimately leave no room for even one
        // routed-expert working set after static weights, KV state, workspace,
        // and staging are allocated (notably for large Q6 models on 12 GiB
        // cards). In that case, keep the CUDA-resident shared model and execute
        // routed experts on CPU instead of failing on the first decode.
        // Routed experts in an IQ codebook format have no grouped GPU kernel.
        // Without this the dispatch below falls through to the k-quant kernel
        // and decodes codebook bytes as super-block scales, which produces
        // fluent-looking output for as long as the expert cache stays cold and
        // then degenerates once experts become resident.
        // IQ experts do have grouped GPU kernels now, but splitting a layer
        // between the router on the GPU and the experts on the host costs a
        // round trip per layer. Laguna therefore concentrates its cache into
        // complete pinned layers; other IQ models still require an explicitly
        // seeded set before the GPU path is allowed.
        const bool seeded_placement=runtime->options.prefill_cache_seed!=0||
                                    runtime->options.prefill_cache_seed_auto!=0;
        // Gated on the decode policy, not the prepare one: `auto` prepares as
        // CPU and only promotes a hot set at decode, so a prepare-phase test
        // would let IQ experts reach the GPU anyway.
        const bool gpu_at_decode=!qwen_expert_policy(
            *runtime,colibri::v2::ExpertExecutionPhase::decode).is_cpu();
        if(gpu_at_decode&&runtime->model->config.expert_count&&
           (!qwen_gpu_experts_executable(*runtime)||
            (qwen_model_has_iq_experts(*runtime)&&!seeded_placement&&
             runtime->whole_expert_layer_slots.empty()))){
            runtime->expert_mode=colibri::v2::ExpertExecutionMode::cpu;
            runtime->options.moe_device=colibri::v2::expert_execution_mode_value(
                runtime->expert_mode);
            prepare_policy=qwen_expert_policy(
                *runtime,colibri::v2::ExpertExecutionPhase::prepare);
            std::fprintf(stderr,
                "[colibri-v2] routed experts stay on the CPU MoE; set "
                "--prefill-cache-seed or COLIBRI_LAGUNA_WHOLE_LAYERS to "
                "place IQ experts on the GPU\n");
        }
        if(prepare_policy.is_hybrid()&&runtime->model->config.expert_count&&
           runtime->expert_slots.empty()){
            runtime->expert_mode=colibri::v2::ExpertExecutionMode::cpu;
            runtime->options.moe_device=colibri::v2::expert_execution_mode_value(
                runtime->expert_mode);
            prepare_policy=qwen_expert_policy(
                *runtime,colibri::v2::ExpertExecutionPhase::prepare);
            std::fprintf(stderr,
                "[colibri-v2] hybrid expert cache does not fit; falling back to CPU MoE\n");
        }
        runtime->expert_history.resize(
            qwen_cache_layer_count(*runtime) *
            runtime->model->config.expert_count
        );
        qwen_load_expert_history(*runtime);
        runtime->device_tensors.assign(runtime->model->tensors.size(),0);
        std::uint64_t cursor=0;
        std::vector<float> widened;
        std::vector<std::uint8_t> packed;
        for(std::uint64_t index=0;index<persistent.size();++index){
            if(!persistent[index])continue;
            const auto&t=runtime->model->tensors[index];
            const auto device_bytes=qwen_device_tensor_size(*runtime,index);
            if(const auto* aliased=qwen_alias_static_tensor(*runtime,index)){
                // Point at the mapping instead of copying into the arena, and
                // leave the cursor alone -- no arena space was reserved.
                runtime->device_tensors[index]=
                    reinterpret_cast<std::uint64_t>(aliased);
                runtime->aliased_tensors++;
                runtime->aliased_tensor_bytes+=device_bytes;
                continue;
            }
            runtime->device_tensors[index]=runtime->static_arena+cursor;
            if(qwen_device_type(*runtime,index)==8&&t.type==30){
                const std::uint64_t elements=device_bytes/kQ8BlockSize*32;
                widened.resize(elements);
                packed.resize(device_bytes);
                const auto* source=reinterpret_cast<const std::uint16_t*>(
                    tensor_data(*runtime->model,t));
                for(std::uint64_t i=0;i<elements;++i)
                    widened[i]=qwen_bf16_value(source[i]);
                qwen_pack_q8_0(widened.data(),elements,packed.data());
                if(colibri_gpu_upload_sync(runtime->device_tensors[index],packed.data(),device_bytes)!=0)throw std::runtime_error("failed to upload requantized native Qwen static tensor");
            }else if(colibri_gpu_upload_sync(runtime->device_tensors[index],tensor_data(*runtime->model,t),t.size)!=0)throw std::runtime_error("failed to upload native Qwen static tensor");
            cursor+=device_align(device_bytes);
        }
        if(runtime->aliased_tensors)
            std::fprintf(stderr,
                "[colibri-v2] %llu static tensors (%llu MiB) served from the GGUF "
                "mapping; device arena is %llu MiB\n",
                static_cast<unsigned long long>(runtime->aliased_tensors),
                static_cast<unsigned long long>(runtime->aliased_tensor_bytes/(1024ull*1024)),
                static_cast<unsigned long long>(runtime->static_arena_bytes/(1024ull*1024)));
        if(runtime->options.strict_resident&&resident_expert_bytes){
            const auto experts=runtime->model->config.expert_count;
            const auto cache_layers=qwen_cache_layer_count(*runtime);
            for(std::uint32_t layer=0;layer<cache_layers;++layer){
                const auto&plan=layer<runtime->layers.size()
                    ? runtime->layers[layer]:runtime->mtp_layer_plan;
                for(std::uint32_t expert=0;expert<experts;++expert){
                    const auto slot_index=
                        static_cast<std::size_t>(layer)*experts+expert;
                    auto&slot=runtime->expert_slots[slot_index];
                    const auto slot_base=runtime->expert_cache+
                        slot_index*runtime->expert_slot_bytes;
                    std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){
                        const auto&t=runtime->model->tensors[
                            plan.expert_tensors[role]];
                        const auto bytes=t.size/experts;
                        const auto source_offset=
                            static_cast<std::uint64_t>(expert)*bytes;
                        if(colibri_gpu_upload(
                                slot_base+role_offset,
                                tensor_data(*runtime->model,t)+source_offset,
                                bytes,runtime->stream)!=0)
                            throw std::runtime_error(
                                "failed to prepare resident expert tensor");
                        role_offset+=bytes;
                    }
                    if(role_offset>runtime->expert_slot_bytes)
                        throw std::runtime_error(
                            "resident expert bundle exceeds its cache slot");
                    slot.key=(static_cast<std::uint64_t>(layer)<<32)|expert;
                    slot.valid=true;
                    slot.last_used=++runtime->expert_clock;
                    runtime->expert_residency[slot.key]=slot_index;
                }
            }
            std::fprintf(stderr,
                "[colibri-v2] resident expert mode prepared %zu experts (%llu MiB)\n",
                runtime->expert_slots.size(),
                static_cast<unsigned long long>(
                    resident_expert_bytes/(1024ull*1024)));
        }else if(!runtime->whole_expert_layer_slots.empty()){
            const auto experts=runtime->model->config.expert_count;
            std::size_t prepared=0;
            for(std::uint32_t layer=0;layer<runtime->layers.size();++layer){
                const auto first_slot=runtime->whole_expert_layer_slots[layer];
                if(first_slot<0)continue;
                const auto&plan=runtime->layers[layer];
                for(std::uint32_t expert=0;expert<experts;++expert){
                    const auto slot_index=static_cast<std::size_t>(first_slot)+expert;
                    auto&slot=runtime->expert_slots.at(slot_index);
                    const auto slot_base=runtime->expert_cache+
                        slot_index*runtime->expert_slot_bytes;
                    std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){
                        const auto&t=runtime->model->tensors[
                            plan.expert_tensors[role]];
                        const auto bytes=t.size/experts;
                        const auto source_offset=
                            static_cast<std::uint64_t>(expert)*bytes;
                        if(colibri_gpu_upload(
                                slot_base+role_offset,
                                tensor_data(*runtime->model,t)+source_offset,
                                bytes,runtime->stream)!=0)
                            throw std::runtime_error(
                                "failed to prepare Laguna whole-layer expert");
                        role_offset+=bytes;
                    }
                    if(role_offset>runtime->expert_slot_bytes)
                        throw std::runtime_error(
                            "Laguna whole-layer expert exceeds its cache slot");
                    slot.key=(static_cast<std::uint64_t>(layer)<<32)|expert;
                    slot.valid=true;
                    slot.pinned=true;
                    slot.last_used=++runtime->expert_clock;
                    runtime->expert_residency[slot.key]=slot_index;
                    ++prepared;
                }
            }
            std::fprintf(stderr,
                "[colibri-v2] prepared %zu pinned Laguna whole-layer experts "
                "(%llu MiB)\n",prepared,
                static_cast<unsigned long long>(
                    runtime->expert_cache_bytes/(1024ull*1024)));
        }
        if(colibri_gpu_memset(runtime->workspace,0,runtime->workspace_bytes,runtime->stream)!=0||
           colibri_gpu_memset(runtime->state,0,runtime->state_bytes,runtime->stream)!=0||
           colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("failed to initialize native Qwen CUDA arenas");
    }catch(...){release_qwen_device(*runtime);throw;}
        const auto paging_policy=qwen_expert_policy(
            *runtime,colibri::v2::ExpertExecutionPhase::prepare);
        if(runtime->options.expert_paging>2)
            throw std::runtime_error("native Qwen expert paging mode is invalid");
        runtime->host_available_bytes=available_host_memory();
        const bool forced_direct=runtime->options.expert_paging==2||
            std::getenv("COLIBRI_V2_DMA_PAGING");
        const auto model_bytes=runtime->model->mapped_bytes();
        const auto registration_headroom=std::max<std::uint64_t>(
            4ull*1024*1024*1024,model_bytes/4);
        const bool auto_direct=runtime->options.expert_paging==0&&
            runtime->host_available_bytes>=model_bytes+registration_headroom;
        // Both GPU-side MoE paths page experts in from the mmap, so both benefit
        // from registering it; only the pure-CPU path never touches the device
        // cache. This used to read `moe_device==2`, which left the streamed-GPU
        // path staging every miss through a host memcpy and made the GPU branch
        // of the next-layer prefetcher (which tests moe_device==0) dead code.
        // A dense model pages no experts, so pinning the whole mapping buys
        // nothing and costs seconds of registration plus locked host pages.
        const bool pages_experts=!runtime->options.strict_resident&&
            !runtime->layers.empty()&&!runtime->layers.front().dense_ffn;
        if(pages_experts&&paging_policy.routed_gpu_execution_allowed()&&
           (forced_direct||auto_direct)){
            const auto registration_started=std::chrono::steady_clock::now();
            // Every shard of a split checkpoint has to be registered; a partial
            // registration is rolled back so the staged path stays coherent.
            int registration=0;
            std::vector<const std::uint8_t*> registered;
            runtime->model->for_each_mapping([&](const std::uint8_t* base,std::uint64_t bytes){
                if(registration)return;
                registration=colibri_gpu_host_register(base,bytes);
                if(!registration)registered.push_back(base);
            });
            if(registration)
                for(const auto* base:registered)colibri_gpu_host_unregister(base);
            runtime->paging_registration_nanoseconds=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-registration_started).count();
            if(registration==0){
                runtime->model_registered=true;runtime->dma_paging=true;
                std::fprintf(stderr,
                    "[colibri-v2] direct expert paging on (registered %.1f GiB mmap in %.2fs)\n",
                    model_bytes/1073741824.0,
                    runtime->paging_registration_nanoseconds/1e9);
            }else if(forced_direct){
                throw std::runtime_error(
                    "direct expert paging requested, but CUDA host registration failed");
            }else{
                std::fprintf(stderr,
                    "[colibri-v2] direct expert paging unavailable; using staged copies\n");
            }
        }
        runtime->decode_ready=true;
    return 0;
});}

// KV cache kernel selection by configured precision (0=f32, 1=f16, 2=bf16).
// K and V are independent: append is one kv_store_<t> launch per cache; scores
// read K, values read V; the fused prefill only runs when K==V (else per-token).
// KV byte size for `elems` elements at a given type (q8_0 = 34 bytes per 32-elem block).
inline std::uint64_t kv_region_bytes(std::uint64_t elems,int type){return type==3?(elems/32)*kQ8BlockSize:elems*(type==0?4:2);}
// Cache precision codes: 0=f32, 1=f16, 2=bf16, 3=q8_0, 4=turbo3, 5=turbo4.
// The turbo store kernels come in K and V flavours because the two rotate under
// different sign streams, which is what lets values be accumulated rotated.
inline const char* kv_store_kernel(int t,bool key){return t==5?(key?"kv_store_turbo4_k":"kv_store_turbo4_v"):t==4?(key?"kv_store_turbo3_k":"kv_store_turbo3_v"):t==3?"kv_store_q8":t==2?"kv_store_bf16":t==1?"kv_store_f16":"kv_store_f32";}
inline const char* kv_scores_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_k;return t==5?"kv_attention_scores_turbo4":t==4?"kv_attention_scores_turbo3":t==3?"kv_attention_scores_q8":t==2?"kv_attention_scores_bf16":t==1?"kv_attention_scores_f16":"kv_attention_scores";}
inline const char* kv_values_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_v;return t==5?"kv_attention_values_turbo4":t==4?"kv_attention_values_turbo3":t==3?"kv_attention_values_q8":t==2?"kv_attention_values_bf16":t==1?"kv_attention_values_f16":"kv_attention_values";}
inline const char* kv_scores_ring_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_k;return t==5?"kv_attention_scores_turbo4_ring":t==4?"kv_attention_scores_turbo3_ring":t==3?"kv_attention_scores_q8_ring":t==2?"kv_attention_scores_bf16_ring":t==1?"kv_attention_scores_f16_ring":"kv_attention_scores_ring";}
inline const char* kv_values_ring_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_v;return t==5?"kv_attention_values_turbo4_ring":t==4?"kv_attention_values_turbo3_ring":t==3?"kv_attention_values_q8_ring":t==2?"kv_attention_values_bf16_ring":t==1?"kv_attention_values_f16_ring":"kv_attention_values_ring";}
inline const char* kv_fused_tiles_kernel(const ColibriV2QwenRuntime& r){
    if(r.options.cache_type_k!=r.options.cache_type_v)return nullptr;
    const int t=r.options.cache_type_k;
    return t==3?"kv_attention_fused_q8_tiles":
           t==2?"kv_attention_fused_bf16_tiles":
           t==1?"kv_attention_fused_f16_tiles":nullptr;
}
// Prefill counterpart of qwen_turbo_cublas_attention: stage the whole causal
// window as f16 once for the chunk, then run the fused f16 prefill kernel over
// it instead of walking the cache once per row.
//
// The cuBLAS prefill entry point is deliberately not used here even though it is
// faster for f16: it fuses the output gate, and the gate is a sigmoid, so
// applying it before the inverse rotation would be wrong. The fused kernel keeps
// the gate as a separate launch, which composes with the rotation correctly.
//
// Stages the causal window as f16 and rotates the query rows in place, leaving
// the caller to run either cuBLAS or the fused kernel over the staged copy and
// then inverse-rotate and gate. Returns false, having touched nothing, when the
// configuration is not eligible.
inline bool qwen_turbo_prefill_stage(
    ColibriV2QwenRuntime& runtime, const QwenLayerPlan& layer,
    std::uint64_t queries, std::uint64_t cache_keys, std::uint64_t cache_values,
    int heads, int kv_heads, int head_dim, int rows, int base,
    std::uint64_t& stage_keys, std::uint64_t& stage_values
){
    const int ck=runtime.options.cache_type_k, cv=runtime.options.cache_type_v;
    if(ck!=cv||!kv_type_is_turbo(ck))return false;
    if(!runtime.turbo_kv_stage||rows<=0||heads<=0||kv_heads<=0)return false;
    if(heads%kv_heads!=0||head_dim<32||(head_dim&31)!=0||head_dim>256)return false;
    if(layer.attention_window)return false;  // ring layers keep the per-token path
    const char*env=std::getenv("COLIBRI_TURBO_CUBLAS");
    if(env&&env[0]=='0')return false;
    const int window=base+rows;
    if(window<=0||static_cast<std::uint64_t>(window)>layer.cache_capacity)return false;
    const std::uint64_t needed=
        static_cast<std::uint64_t>(kv_heads)*window*head_dim*sizeof(std::uint16_t);
    if(needed>runtime.turbo_kv_stage_stride)return false;

    auto launch=[&](const char*name,std::uint32_t gx,std::uint32_t gy,void**args){
        if(colibri_gpu_launch_named(name,gx,gy,256,0,runtime.stream,args)!=0)
            throw std::runtime_error(std::string("native Qwen turbo prefill kernel failed: ")+name);
    };
    stage_keys=runtime.turbo_kv_stage;
    stage_values=runtime.turbo_kv_stage+runtime.turbo_kv_stage_stride;
    const char*dequant=ck==5?"kv_dequant_turbo4_f16":"kv_dequant_turbo3_f16";
    const std::uint32_t token_blocks=(static_cast<std::uint32_t>(window)+7)/8;
    int tokens=window,slot_capacity=static_cast<int>(layer.cache_capacity),origin=0;
    void*key_args[]={&cache_keys,&stage_keys,&kv_heads,&head_dim,&tokens,&slot_capacity,&origin};
    launch(dequant,static_cast<std::uint32_t>(kv_heads),token_blocks,key_args);
    void*value_args[]={&cache_values,&stage_values,&kv_heads,&head_dim,&tokens,&slot_capacity,&origin};
    launch(dequant,static_cast<std::uint32_t>(kv_heads),token_blocks,value_args);

    // Queries and attended rows are both [row][head][head_dim] contiguous, so
    // the rotation runs over rows*heads vectors in one launch.
    int vectors=rows*heads,key_stream=0;
    void*rotate_args[]={&queries,&vectors,&head_dim,&key_stream};
    launch("turbo_rotate_rows",static_cast<std::uint32_t>(vectors),1,rotate_args);
    return true;
}

inline int kv_fused_tile_tokens(const ColibriV2QwenRuntime&){return 1024;}
inline int kv_fused_grid_heads(const ColibriV2QwenRuntime&,int heads,int){return heads;}
inline int qwen_cublas_attention_min_tokens(){
    static const int threshold=[]{
        const char*env=std::getenv("COLIBRI_CUBLAS_ATTENTION_MIN_TOKENS");
        if(!env||!env[0])
            return colibri::v2::attention::kDefaultCublasMinTokens;
        char*end=nullptr;
        const long value=std::strtol(env,&end,10);
        if(end==env||*end!='\0'||value<1||
           value>std::numeric_limits<std::int32_t>::max())
            return colibri::v2::attention::kDefaultCublasMinTokens;
        return static_cast<int>(value);
    }();
    return threshold;
}
inline bool qwen_cublas_attention_eligible(
    const ColibriV2QwenRuntime&runtime,int tokens,int first_slot,int capacity
){
    const char*env=std::getenv("COLIBRI_CUBLAS_ATTENTION");
    const bool enabled=!env||env[0]!='0';
    return colibri::v2::attention::cublas_eligible(
        runtime.options.cache_type_k,runtime.options.cache_type_v,
        tokens,first_slot,capacity,enabled,
        qwen_cublas_attention_min_tokens());
}

// Attention for a turbo KV cache: expand the live window to f16 in the shared
// staging buffer and hand it to cuBLAS, which is roughly 2x faster than the
// warp-per-token fused kernel and much faster than the scores+values pair.
//
// Keys and values stay rotated in the staging copy, so the query is rotated on
// the way in and the output inverse-rotated on the way out. Both are only
// heads*head_dim elements, negligible beside the window itself, and it avoids a
// Walsh-Hadamard per cached token.
//
// Eligibility is decided before anything is mutated, so a false return leaves
// the caller's normal dispatch free to run on an untouched query. Failures
// after that point throw rather than fall back, because the query has already
// been rotated in place by then.
inline bool qwen_turbo_cublas_attention(
    ColibriV2QwenRuntime& runtime, std::uint64_t queries, std::uint64_t query_f16,
    std::uint64_t cache_keys, std::uint64_t cache_values, std::uint64_t scores_f16,
    std::uint64_t attended, int heads, int kv_heads, int head_dim,
    int tokens, int capacity, int first_slot, float scale
){
    const int ck=runtime.options.cache_type_k, cv=runtime.options.cache_type_v;
    const bool diagnose=std::getenv("COLIBRI_TURBO_DIAG")!=nullptr;
    auto decline=[&](const char*why)->bool{
        if(diagnose){
            static const char*last=nullptr;
            if(last!=why){last=why;std::fprintf(stderr,"[turbo] declined: %s\n",why);}
        }
        return false;
    };
    if(ck!=cv||!kv_type_is_turbo(ck))return decline("cache type not a matched turbo pair");
    if(!runtime.turbo_kv_stage)return decline("no staging buffer");
    if(tokens<=0||heads<=0||kv_heads<=0)return decline("degenerate shape");
    if(heads%kv_heads!=0||head_dim<32||(head_dim&31)!=0)return decline("head geometry");
    if(tokens<qwen_cublas_attention_min_tokens())return decline("below cuBLAS token threshold");
    if(first_slot<0||first_slot>=capacity)return decline("ring slot out of range");
    const char*env=std::getenv("COLIBRI_TURBO_CUBLAS");
    if(env&&env[0]=='0')return decline("disabled by COLIBRI_TURBO_CUBLAS=0");
    const std::uint64_t needed=
        static_cast<std::uint64_t>(kv_heads)*tokens*head_dim*sizeof(std::uint16_t);
    if(needed>runtime.turbo_kv_stage_stride)return decline("staging buffer too small");
    if(diagnose){
        static bool announced=false;
        if(!announced){announced=true;std::fprintf(stderr,"[turbo] cuBLAS attention path engaged\n");}
    }

    auto launch=[&](const char*name,std::uint32_t gx,std::uint32_t gy,void**args){
        if(colibri_gpu_launch_named(name,gx,gy,256,0,runtime.stream,args)!=0)
            throw std::runtime_error(std::string("native Qwen turbo attention kernel failed: ")+name);
    };
    std::uint64_t stage_keys=runtime.turbo_kv_stage;
    std::uint64_t stage_values=runtime.turbo_kv_stage+runtime.turbo_kv_stage_stride;
    const char*dequant=ck==5?"kv_dequant_turbo4_f16":"kv_dequant_turbo3_f16";
    const std::uint32_t token_blocks=(static_cast<std::uint32_t>(tokens)+7)/8;
    void*key_args[]={&cache_keys,&stage_keys,&kv_heads,&head_dim,&tokens,&capacity,&first_slot};
    launch(dequant,static_cast<std::uint32_t>(kv_heads),token_blocks,key_args);
    void*value_args[]={&cache_values,&stage_values,&kv_heads,&head_dim,&tokens,&capacity,&first_slot};
    launch(dequant,static_cast<std::uint32_t>(kv_heads),token_blocks,value_args);

    int key_stream=0,value_stream=1,unwrapped=0;
    void*rotate_args[]={&queries,&heads,&head_dim,&key_stream};
    launch("turbo_rotate_rows",static_cast<std::uint32_t>(heads),1,rotate_args);
    // The staging copy is already unwrapped, so cuBLAS sees a linear window.
    if(colibri_gpu_attention_f16_cublas(queries,query_f16,stage_keys,stage_values,
            scores_f16,attended,runtime.stream,heads,kv_heads,head_dim,tokens,
            tokens,unwrapped,scale)!=0)
        throw std::runtime_error("native Qwen turbo cuBLAS attention failed");
    void*unrotate_args[]={&attended,&heads,&head_dim,&value_stream};
    launch("turbo_unrotate_rows",static_cast<std::uint32_t>(heads),1,unrotate_args);
    return true;
}

void qwen_mtp_dense_projection(
    ColibriV2QwenRuntime& runtime, std::size_t tensor_index_value,
    std::uint64_t input, std::uint64_t output,
    int input_size, int output_size
) {
    std::uint64_t matrix = runtime.device_tensors[tensor_index_value];
    const auto type = qwen_device_type(runtime, tensor_index_value);
    auto launch = [&](const char* name, std::uint32_t grid_x, void** args) {
        if (colibri_gpu_launch_named(
                name, grid_x, 1, 256, 0, runtime.stream, args) != 0) {
            throw std::runtime_error(
                std::string("native MTP dense projection failed: ") + name);
        }
    };
    switch (type) {
        case 0: {
            void* args[] = {
                &matrix, &input, &output, &input_size, &output_size};
            launch("qwen_f32_matvec_warp", (output_size + 7) / 8, args);
            return;
        }
        case 30: {
            // bf16_matvec takes (rows, columns), the reverse of the
            // quantized matvec helpers.
            void* args[] = {
                &matrix, &input, &output, &output_size, &input_size};
            launch("bf16_matvec_warp", (output_size + 7) / 8, args);
            return;
        }
        case 8:
            if (colibri_gpu_q8_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 16:
            if (colibri_gpu_iq2xxs_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 18:
            if (colibri_gpu_iq3xxs_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 22:
            if (colibri_gpu_iq2s_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 21:
            if (colibri_gpu_iq3s_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 17:
            if (colibri_gpu_iq2xs_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 23:
            if (colibri_gpu_iq4xs_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 10:
            if (colibri_gpu_q2k_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 11:
            if (colibri_gpu_q3k_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 13:
            if (colibri_gpu_q5k_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 12:
            if (colibri_gpu_q4k_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        case 14:
            if (colibri_gpu_q6k_matvec_transposed(
                    matrix,input,output,input_size,output_size,runtime.stream)==0) return;
            break;
        default:
            throw std::runtime_error(
                "native MTP dense projection type is unsupported: " +
                std::to_string(type));
    }
    throw std::runtime_error("native MTP dense projection failed");
}

void qwen_mtp_append_pair(
    ColibriV2QwenRuntime& runtime, std::uint32_t token,
    std::uint64_t input_hidden
) {
    const int hidden = static_cast<int>(runtime.model->config.hidden_size);
    const int kv_heads = static_cast<int>(runtime.model->config.attention_kv_heads);
    const auto& layer = runtime.mtp_layer_plan;
    const int head_dim = static_cast<int>(
        runtime.model->tensors[layer.static_tensors[2]].shape[1] / kv_heads
    );
    const int kv_size = kv_heads * head_dim;
    const float epsilon = runtime.model->config.rms_norm_epsilon
        ? runtime.model->config.rms_norm_epsilon : 1.0e-6f;
    std::uint64_t cursor = runtime.workspace;
    const auto workspace_end = runtime.workspace + runtime.workspace_bytes;
    auto take = [&](std::uint64_t bytes) {
        const auto result = cursor;
        cursor += device_align(bytes);
        if (cursor > workspace_end) throw std::runtime_error("native MTP workspace is too small");
        return result;
    };
    const auto embedding = take(hidden * sizeof(float));
    const auto normalized_embedding = take(hidden * sizeof(float));
    const auto normalized_hidden = take(hidden * sizeof(float));
    const auto concatenated = take(2ULL * hidden * sizeof(float));
    const auto fused = take(hidden * sizeof(float));
    const auto normalized = take(hidden * sizeof(float));
    const auto keys = take(kv_size * sizeof(float));
    const auto values = take(kv_size * sizeof(float));
    auto launch = [&](const char* name, std::uint32_t grid_x,
                      std::uint32_t grid_y, void** arguments) {
        if (colibri_gpu_launch_named(
                name, grid_x, grid_y, 256, 0, runtime.stream, arguments
            ) != 0) {
            throw std::runtime_error(std::string("native MTP CUDA kernel failed: ") + name);
        }
    };
    const std::uint32_t embedding_token = token;
    const auto embedding_matrix =
        qwen_stage_embedding_rows(runtime, &embedding_token, 1);
    int token_value = runtime.embeddings_host_resident
        ? 0 : static_cast<int>(token);
    void* embedding_args[] = {
        const_cast<std::uint64_t*>(&embedding_matrix),
        const_cast<std::uint64_t*>(&embedding), &token_value,
        const_cast<int*>(&hidden),
    };
    launch(qwen_embedding_kernel(qwen_device_type(runtime, runtime.token_embeddings), false), (hidden + 255) / 256, 1, embedding_args);
    auto rms = [&](std::uint64_t input, std::uint64_t weight,
                   std::uint64_t output) {
        int one_centered = 0;
        void* args[] = {&input, &weight, &output, const_cast<int*>(&hidden),
                        const_cast<float*>(&epsilon), &one_centered};
        launch("rms_norm", 1, 1, args);
    };
    rms(embedding, runtime.device_tensors[runtime.mtp_special_tensors[1]], normalized_embedding);
    rms(input_hidden, runtime.device_tensors[runtime.mtp_special_tensors[2]], normalized_hidden);
    void* concat_args[] = {
        const_cast<std::uint64_t*>(&normalized_embedding),
        const_cast<std::uint64_t*>(&normalized_hidden),
        const_cast<std::uint64_t*>(&concatenated), const_cast<int*>(&hidden),
    };
    launch("qwen_concat_pair", (hidden + 255) / 256, 1, concat_args);
    qwen_mtp_dense_projection(
        runtime, runtime.mtp_special_tensors[0], concatenated,
        fused, hidden * 2, hidden);
    rms(fused, runtime.device_tensors[layer.static_tensors[0]], normalized);
    qwen_mtp_dense_projection(
        runtime, layer.static_tensors[2], normalized,
        keys, hidden, kv_size);
    qwen_mtp_dense_projection(
        runtime, layer.static_tensors[3], normalized,
        values, hidden, kv_size);
    const int rotary = static_cast<int>(runtime.model->config.rotary_dimension
        ? runtime.model->config.rotary_dimension : head_dim);
    const int position = static_cast<int>(runtime.mtp_cache_tokens);
    const float theta = runtime.model->config.rope_freq_base
        ? runtime.model->config.rope_freq_base : 1000000.0f;
    const auto key_norm = runtime.device_tensors[layer.static_tensors[6]];
    const auto rotated_keys = fused;
    void* key_args[] = {
        const_cast<std::uint64_t*>(&keys), const_cast<std::uint64_t*>(&key_norm),
        const_cast<std::uint64_t*>(&rotated_keys), const_cast<int*>(&kv_heads),
        const_cast<int*>(&head_dim), const_cast<int*>(&rotary),
        const_cast<int*>(&position), const_cast<float*>(&theta),
        const_cast<float*>(&epsilon),
    };
    launch("qwen_attention_key", kv_heads, 1, key_args);
    auto cache_keys = runtime.state + layer.state_first;
    auto cache_values = runtime.state + layer.state_second;
    int capacity = static_cast<int>(runtime.options.context_limit);
    // One store per cache so K and V follow their configured precisions
    // independently, exactly as the target decode path does.
    void* key_store_args[] = {
        const_cast<std::uint64_t*>(&rotated_keys), &cache_keys,
        const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim),
        const_cast<int*>(&position), &capacity,
    };
    launch(kv_store_kernel(runtime.options.cache_type_k,true), kv_heads, 1, key_store_args);
    void* value_store_args[] = {
        const_cast<std::uint64_t*>(&values), &cache_values,
        const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim),
        const_cast<int*>(&position), &capacity,
    };
    launch(kv_store_kernel(runtime.options.cache_type_v,false), kv_heads, 1, value_store_args);
    ++runtime.mtp_cache_tokens;
}

void qwen_mtp_append_prompt_pair(
    ColibriV2QwenRuntime& runtime, std::uint32_t token
) {
    if (!runtime.options.mtp_drafts || !runtime.mtp_has_target_hidden) return;
    qwen_mtp_append_pair(
        runtime, token,
        runtime.state + runtime.mtp_target_hidden_offset);
}

std::uint32_t qwen_mtp_draft(
    ColibriV2QwenRuntime& runtime, std::uint32_t token,
    std::uint64_t input_hidden
) {
    const int hidden_size=static_cast<int>(runtime.model->config.hidden_size);
    const int experts=static_cast<int>(runtime.model->config.expert_count);
    const int top_k=static_cast<int>(runtime.model->config.expert_used_count);
    const int intermediate=static_cast<int>(runtime.moe_intermediate);
    const int heads=static_cast<int>(runtime.model->config.attention_heads);
    const int kv_heads=static_cast<int>(runtime.model->config.attention_kv_heads);
    const auto&layer=runtime.mtp_layer_plan;
    const int head_dim=static_cast<int>(runtime.model->tensors[layer.static_tensors[2]].shape[1]/kv_heads);
    const int q_size=heads*2*head_dim,kv_size=kv_heads*head_dim;
    const float epsilon=runtime.model->config.rms_norm_epsilon?runtime.model->config.rms_norm_epsilon:1.0e-6f;
    std::uint64_t cursor=runtime.workspace;
    const auto workspace_end=runtime.workspace+runtime.workspace_bytes;
    auto take=[&](std::uint64_t bytes){const auto result=cursor;cursor+=device_align(bytes);if(cursor>workspace_end)throw std::runtime_error("native MTP workspace is too small");return result;};
    const auto embedding=take(hidden_size*sizeof(float));
    const auto norm_embedding=take(hidden_size*sizeof(float));
    const auto norm_hidden=take(hidden_size*sizeof(float));
    const auto concatenated=take(2ULL*hidden_size*sizeof(float));
    std::uint64_t hidden=take(hidden_size*sizeof(float));
    std::uint64_t residual=take(hidden_size*sizeof(float));
    const auto normalized=take(hidden_size*sizeof(float));
    const auto first=take(runtime.scratch_elements*sizeof(float));
    const auto second=take(runtime.scratch_elements*sizeof(float));
    const auto third=take(runtime.scratch_elements*sizeof(float));
    const auto fourth=take(runtime.scratch_elements*sizeof(float));
    const auto activated=take(top_k*runtime.moe_intermediate*sizeof(float));
    const auto router_logits=take(experts*sizeof(float));
    const auto selected_device=take(top_k*sizeof(std::int32_t));
    const auto route_weights=take(top_k*sizeof(float));
    const auto argmax_device=take(sizeof(std::uint64_t));
    const auto attention_scores=take(static_cast<std::uint64_t>(heads)*runtime.options.context_limit*sizeof(float));
    auto launch=[&](const char*name,std::uint32_t gx,std::uint32_t gy,void**args){if(colibri_gpu_launch_named(name,gx,gy,256,0,runtime.stream,args)!=0)throw std::runtime_error(std::string("native MTP CUDA kernel failed: ")+name);};
    auto dense_index=[&](std::size_t index,std::uint64_t input,std::uint64_t output,int input_size,int output_size){
        qwen_mtp_dense_projection(
            runtime,index,input,output,input_size,output_size);
    };
    auto rms=[&](std::uint64_t input,std::uint64_t weight,std::uint64_t output){int one_centered=0;void*args[]={&input,&weight,&output,const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon),&one_centered};launch("rms_norm",1,1,args);};
    auto add=[&](std::uint64_t target,std::uint64_t source){float scale=1.0f;int count=hidden_size;void*args[]={&target,&source,&scale,&count};launch("scaled_add",(hidden_size+255)/256,1,args);};
    auto tensor=[&](std::size_t role){return runtime.device_tensors[layer.static_tensors.at(role)];};
    auto dense=[&](std::size_t role,std::uint64_t input,std::uint64_t output,int input_size,int output_size){dense_index(layer.static_tensors.at(role),input,output,input_size,output_size);};
    const std::uint32_t embedding_token=token;
    const auto embedding_matrix=qwen_stage_embedding_rows(runtime,&embedding_token,1);
    int token_value=runtime.embeddings_host_resident?0:static_cast<int>(token);
    void*embedding_args[]={const_cast<std::uint64_t*>(&embedding_matrix),const_cast<std::uint64_t*>(&embedding),&token_value,const_cast<int*>(&hidden_size)};
    launch(qwen_embedding_kernel(qwen_device_type(runtime,runtime.token_embeddings),false),(hidden_size+255)/256,1,embedding_args);

    rms(embedding,runtime.device_tensors[runtime.mtp_special_tensors[1]],norm_embedding);
    rms(input_hidden,runtime.device_tensors[runtime.mtp_special_tensors[2]],norm_hidden);
    void*concat_args[]={const_cast<std::uint64_t*>(&norm_embedding),const_cast<std::uint64_t*>(&norm_hidden),const_cast<std::uint64_t*>(&concatenated),const_cast<int*>(&hidden_size)};
    launch("qwen_concat_pair",(hidden_size+255)/256,1,concat_args);
    dense_index(runtime.mtp_special_tensors[0],concatenated,hidden,hidden_size*2,hidden_size);

    rms(hidden,tensor(0),normalized);
    dense(1,normalized,first,hidden_size,q_size);
    dense(2,normalized,second,hidden_size,kv_size);
    dense(3,normalized,third,hidden_size,kv_size);
    const int rotary=static_cast<int>(runtime.model->config.rotary_dimension?runtime.model->config.rotary_dimension:head_dim);
    const int position=static_cast<int>(runtime.mtp_cache_tokens);
    const float theta=runtime.model->config.rope_freq_base?runtime.model->config.rope_freq_base:1000000.0f;
    auto qnorm=tensor(5),knorm=tensor(6);std::uint64_t queries=fourth,gates=fourth+q_size/2*sizeof(float);
    void*q_args[]={const_cast<std::uint64_t*>(&first),&qnorm,&queries,&gates,const_cast<int*>(&heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),const_cast<int*>(&position),const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
    launch("qwen_attention_query",heads,1,q_args);
    std::uint64_t keys=first;
    void*k_args[]={const_cast<std::uint64_t*>(&second),&knorm,&keys,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),const_cast<int*>(&position),const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
    launch("qwen_attention_key",kv_heads,1,k_args);
    auto cache_keys=runtime.state+layer.state_first,cache_values=runtime.state+layer.state_second;int capacity=static_cast<int>(runtime.options.context_limit);
    void*key_store_args[]={&keys,&cache_keys,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),const_cast<int*>(&position),&capacity};
    launch(kv_store_kernel(runtime.options.cache_type_k,true),kv_heads,1,key_store_args);
    void*value_store_args[]={const_cast<std::uint64_t*>(&third),&cache_values,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),const_cast<int*>(&position),&capacity};
    launch(kv_store_kernel(runtime.options.cache_type_v,false),kv_heads,1,value_store_args);
    std::uint64_t attended=second;int tokens=position+1;float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    void*score_args[]={&queries,&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&scale};
    launch(kv_scores_kernel(runtime),heads,(tokens+255)/256,score_args);
    void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity};
    launch(kv_values_kernel(runtime),heads,1,value_args);
    std::uint64_t gated=third;int elements=heads*head_dim;void*gate_args[]={&attended,&gates,&gated,&elements};launch("qwen_attention_gate",(elements+255)/256,1,gate_args);
    dense(4,gated,residual,elements,hidden_size);add(residual,hidden);

    rms(residual,tensor(7),normalized);
    auto*staging=static_cast<std::uint8_t*>(runtime.host_staging);
    if(layer.dense_ffn){
        // Dense draft block: the nextn layer of a dense checkpoint carries its
        // own gate/up/down instead of a router and experts.
        const int dense_intermediate=static_cast<int>(runtime.moe_intermediate);
        const auto up_half=first+static_cast<std::uint64_t>(dense_intermediate)*sizeof(float);
        dense(8,normalized,first,hidden_size,dense_intermediate);
        dense(9,normalized,up_half,hidden_size,dense_intermediate);
        int dense_count=dense_intermediate;
        void*dense_silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&dense_count};
        launch("silu_mul",(static_cast<std::uint32_t>(dense_count)+255)/256,1,dense_silu_args);
        dense(10,second,third,dense_intermediate,hidden_size);
        add(residual,third);std::swap(hidden,residual);
    }else{
    dense(8,normalized,router_logits,hidden_size,experts);

    if(colibri_gpu_route_topk(router_logits,selected_device,route_weights,experts,top_k,runtime.stream)!=0)throw std::runtime_error("native MTP routing failed");
    auto*selected_host=reinterpret_cast<std::int32_t*>(staging);
    const auto weights_offset=device_align(top_k*sizeof(std::int32_t));const auto input_offset=weights_offset+device_align(top_k*sizeof(float));const auto cpu_activated_offset=input_offset+device_align(hidden_size*sizeof(float));const auto output_offset=cpu_activated_offset+device_align(top_k*runtime.moe_intermediate*sizeof(float));
    auto*cpu_weights=reinterpret_cast<float*>(staging+weights_offset);auto*cpu_input=reinterpret_cast<float*>(staging+input_offset);auto*cpu_activated=reinterpret_cast<float*>(staging+cpu_activated_offset);auto*cpu_output=reinterpret_cast<float*>(staging+output_offset);
    if(colibri_gpu_download(selected_host,selected_device,top_k*sizeof(std::int32_t),runtime.stream)!=0||colibri_gpu_download(cpu_weights,route_weights,top_k*sizeof(float),runtime.stream)!=0||colibri_gpu_download(cpu_input,normalized,hidden_size*sizeof(float),runtime.stream)!=0)throw std::runtime_error("native MTP route transfer failed");
    auto shared_gate_matrix=tensor(9),shared_up_matrix=tensor(10);
    const auto shexp_type=runtime.model->tensors[layer.static_tensors.at(9)].type;
    if(shexp_type==40){
        // NVFP4 shared expert: gate/up scales before the non-linear SiLU, the
        // linear down scale on the down projection itself.
        auto shared_gate_scale=layer.shared_gate_scale,shared_up_scale=layer.shared_up_scale;
        void*silu_args[]={&shared_gate_matrix,&shared_up_matrix,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&second),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate),&shared_gate_scale,&shared_up_scale};
        launch("nvfp4_swiglu_transposed",intermediate,1,silu_args);
        auto shared_down_matrix=tensor(11);auto shared_down_scale=layer.shared_down_scale;
        void*down_args[]={&shared_down_matrix,const_cast<std::uint64_t*>(&second),const_cast<std::uint64_t*>(&third),const_cast<int*>(&intermediate),const_cast<int*>(&hidden_size),&shared_down_scale};
        launch("nvfp4_matvec_transposed",hidden_size,1,down_args);
    }else if(shexp_type==8){
        void*silu_args[]={&shared_gate_matrix,&shared_up_matrix,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&second),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate)};
        launch("q8_swiglu_transposed_warp",(intermediate+7)/8,1,silu_args);
        dense(11,second,third,intermediate,hidden_size);
    }else{
        // No fused SwiGLU kernel for this weight type: project gate and up into one
        // contiguous scratch pair and let silu_mul combine them.
        auto up_half=first+static_cast<std::uint64_t>(intermediate)*sizeof(float);
        dense(9,normalized,first,hidden_size,intermediate);
        dense(10,normalized,up_half,hidden_size,intermediate);
        int count=intermediate;
        void*silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&count};
        launch("silu_mul",(static_cast<std::uint32_t>(count)+255)/256,1,silu_args);
        dense(11,second,third,intermediate,hidden_size);
    }
    auto shared_gate=tensor(12);
    {
        const auto sg_type=runtime.model->tensors[layer.static_tensors.at(12)].type;
        void*shared_args[]={const_cast<std::uint64_t*>(&normalized),&shared_gate,const_cast<std::uint64_t*>(&third),const_cast<int*>(&hidden_size)};
        launch(sg_type==30?"qwen_shared_scale_bf16":"qwen_shared_scale",1,1,shared_args);
    }
    if(colibri_gpu_stream_sync(runtime.stream)!=0)throw std::runtime_error("native MTP route synchronization failed");
    // qwen_cpu_moe expects the linear down scale already folded into the routing
    // weights; it applies gate/up itself.
    for(int rank=0;rank<top_k;++rank){
        if(selected_host[rank]<0||selected_host[rank]>=experts)
            throw std::runtime_error("native MTP routing selected an invalid expert");
        cpu_weights[rank]*=qwen_expert_role_scale(runtime,layer.expert_down_scale,selected_host[rank]);
    }
    qwen_cpu_moe(runtime,layer,selected_host,cpu_weights,top_k,cpu_input,cpu_activated,cpu_output);
    if(colibri_gpu_upload(fourth,cpu_output,hidden_size*sizeof(float),runtime.stream)!=0)throw std::runtime_error("native MTP expert upload failed");
    add(third,fourth);add(residual,third);std::swap(hidden,residual);
    }
    auto draft_hidden=runtime.state+runtime.mtp_draft_hidden_offset;void*copy_args[]={&hidden,&draft_hidden,const_cast<int*>(&hidden_size)};launch("qwen_copy_vector",(hidden_size+255)/256,1,copy_args);
    rms(hidden,runtime.device_tensors[runtime.mtp_special_tensors[3]],normalized);
    if(colibri_gpu_memset(argmax_device,0,sizeof(std::uint64_t),runtime.stream)!=0)throw std::runtime_error("native MTP argmax reset failed");
    int vocabulary=static_cast<int>(runtime.model->config.vocabulary_size);auto lm_head=runtime.device_tensors[runtime.lm_head];void*argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&argmax_device),const_cast<int*>(&hidden_size),&vocabulary};launch(qwen_lm_head_argmax_kernel(runtime.lm_head_type),(vocabulary+7)/8,1,argmax_args);
    auto*packed_winner=reinterpret_cast<std::uint64_t*>(staging);if(colibri_gpu_download(packed_winner,argmax_device,sizeof(*packed_winner),runtime.stream)!=0||colibri_gpu_stream_sync(runtime.stream)!=0)throw std::runtime_error("native MTP output synchronization failed");
    ++runtime.mtp_cache_tokens;++runtime.mtp_draft_tokens;
    return 0xffffffffu-static_cast<std::uint32_t>(*packed_winner);
}

void qwen_snapshot_delta_state(ColibriV2QwenRuntime& runtime, bool restore) {
    auto launch_copy=[&](std::uint64_t source,std::uint64_t destination,std::uint64_t bytes){
        const int elements=static_cast<int>(bytes/sizeof(float));
        void*args[]={&source,&destination,const_cast<int*>(&elements)};
        if(colibri_gpu_launch_named("qwen_copy_vector",(elements+255)/256,1,256,0,runtime.stream,args)!=0)
            throw std::runtime_error("native MTP state snapshot failed");
    };
    for(std::uint32_t layer_number=0;layer_number<runtime.layers.size();++layer_number){
        auto&layer=runtime.layers[layer_number];
        if(layer.attention)continue;
        const auto&conv=runtime.model->tensors[tensor_index(*runtime.model,"blk."+std::to_string(layer_number)+".ssm_conv1d.weight")];
        const auto&a=runtime.model->tensors[tensor_index(*runtime.model,"blk."+std::to_string(layer_number)+".ssm_a")];
        const auto&norm=runtime.model->tensors[tensor_index(*runtime.model,"blk."+std::to_string(layer_number)+".ssm_norm.weight")];
        const auto conv_bytes=conv.shape[0]*conv.shape[1]*sizeof(float);
        const auto recurrent_bytes=a.shape[0]*norm.shape[0]*norm.shape[0]*sizeof(float);
        const auto live_conv=runtime.state+layer.state_first;
        const auto live_recurrent=runtime.state+layer.state_second;
        const auto saved_conv=runtime.state+layer.snapshot_first;
        const auto saved_recurrent=runtime.state+layer.snapshot_second;
        launch_copy(restore?saved_conv:live_conv,restore?live_conv:saved_conv,conv_bytes);
        launch_copy(restore?saved_recurrent:live_recurrent,restore?live_recurrent:saved_recurrent,recurrent_bytes);
    }
}

// Model-agnostic expert-router pruning. Given the `count` experts selected
// by route_topk with their (softmax) weights in the parallel `weights` array,
// keep the smallest nucleus whose cumulative weight fraction reaches top_p,
// after a hard cap of cap_k (0 = no extra cap), always keeping at least one.
// Reorders selected/weights into descending-weight order over the kept set and
// renormalizes the kept weights to sum to 1 (so the MoE combination preserves
// scale). Returns the number of experts kept. No model-specific constants:
// this is a function of the router distribution alone, so it generalizes to
// any MoE (Qwen, Gemma, DeepSeek, ...) and any expert quantization.
static int apply_expert_router_policy(std::int32_t* selected, float* weights,
        int count, int cap_k, float top_p) {
    if(count<=0) return count;
    if(cap_k>0 && cap_k<count) count=cap_k;
    std::array<int,256> order{};
    for(int i=0;i<count;++i) order[i]=i;
    std::sort(order.begin(), order.begin()+count,
        [&](int a,int b){return weights[a]>weights[b];});
    std::array<float,256> w{}; std::array<std::int32_t,256> e{};
    float total=0.0f;
    for(int i=0;i<count;++i){ w[i]=weights[order[i]]; e[i]=selected[order[i]]; total+=w[i]; }
    if(total<=0.0f) total=1.0f;
    int keep=count;
    if(top_p>0.0f && top_p<1.0f){
        float cum=0.0f; keep=0;
        for(int i=0;i<count;++i){ cum+=w[i]; ++keep; if(cum/total>=top_p) break; }
        if(keep<1) keep=1;
    }
    float sum=0.0f;
    for(int i=0;i<keep;++i) sum+=w[i];
    if(sum<=0.0f) sum=1.0f;
    for(int i=0;i<keep;++i){ selected[i]=e[i]; weights[i]=w[i]/sum; }
    return keep;
}

// Copy every DeltaNet layer's conv+recurrent state between the live state
// arena and a prefill snapshot buffer (packed sequentially in layer order).
void qwen_prefill_snapshot_copy(
    ColibriV2QwenRuntime& runtime, std::uint64_t snapshot, bool restore
) {
    auto launch_copy=[&](std::uint64_t source,std::uint64_t destination,std::uint64_t bytes){
        const int elements=static_cast<int>(bytes/sizeof(float));
        void*args[]={&source,&destination,const_cast<int*>(&elements)};
        if(colibri_gpu_launch_named("qwen_copy_vector",(static_cast<std::uint32_t>(elements)+255)/256,1,256,0,runtime.stream,args)!=0)
            throw std::runtime_error("native Qwen prefill snapshot copy failed");
    };
    std::uint64_t cursor=snapshot;
    for(const auto&layer:runtime.layers){
        if(layer.attention)continue;
        const auto&conv=runtime.model->tensors[layer.static_tensors[6]];
        const auto&a=runtime.model->tensors[layer.static_tensors[8]];
        const auto&norm=runtime.model->tensors[layer.static_tensors[9]];
        const auto conv_bytes=conv.shape[0]*conv.shape[1]*sizeof(float);
        const auto recurrent_bytes=a.shape[0]*norm.shape[0]*norm.shape[0]*sizeof(float);
        const auto live_conv=runtime.state+layer.state_first;
        const auto live_recurrent=runtime.state+layer.state_second;
        launch_copy(restore?cursor:live_conv,restore?live_conv:cursor,conv_bytes);
        cursor+=conv_bytes;
        launch_copy(restore?cursor:live_recurrent,restore?live_recurrent:cursor,recurrent_bytes);
        cursor+=recurrent_bytes;
    }
}

struct AttentionCacheView { int slot, first, tokens, capacity; };
inline AttentionCacheView attention_cache_view(const QwenLayerPlan& layer,std::uint64_t position){
    const auto capacity=static_cast<int>(layer.cache_capacity);
    const auto visible=layer.attention_window
        ?std::min<std::uint64_t>(position+1,layer.attention_window):position+1;
    const auto first_absolute=position+1-visible;
    return {static_cast<int>(position%layer.cache_capacity),static_cast<int>(first_absolute%layer.cache_capacity),static_cast<int>(visible),capacity};
}
inline bool swa_snapshot_is_resident(const ColibriV2QwenRuntime& runtime,std::uint64_t snapshot_position,std::uint64_t live_position){
    if(snapshot_position>live_position)return false;
    for(const auto& layer:runtime.layers)if(layer.attention_window&&layer.cache_capacity<runtime.options.context_limit){
        const auto rollback_room=layer.cache_capacity-layer.attention_window;
        if(live_position-snapshot_position>rollback_room)return false;
    }
    return true;
}
// No fused turbo prefill kernel yet, so a turbo cache falls back to the
// separate scores+values path, which handles every precision.
inline bool kv_fused_prefill_ok(const ColibriV2QwenRuntime& r){return r.options.cache_type_k==r.options.cache_type_v&&!kv_type_is_turbo(r.options.cache_type_k);}
inline const char* kv_prefill_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_k;return t==3?"kv_attention_prefill_q8":t==2?"kv_attention_prefill_bf16":t==1?"kv_attention_prefill_f16":"kv_attention_prefill";}

// Dump one attention layer's live KV window for the TurboQuant quality harness
// (native/tools/bench_turboquant.cpp). The file is little-endian and
// headerless apart from two int32s: count, head_dim, then count*head_dim f32
// keys followed by the same again for values, one row per (kv_head, position).
//
// Whatever precision the cache runs at is decoded to f32 on the way out, so a
// dump taken under cache_type f16 or q8_0 still describes the distribution the
// codec would actually see rather than an idealized f32 one. Only the live
// window is written: for a sliding-window layer the ring has stale slots
// outside it, and feeding those to the harness would skew the norm statistics
// that drive the K/V bit allocation.
int colibri_v2_qwen_runtime_dump_kv(
    ColibriV2QwenRuntime* runtime, uint32_t layer_index, const char* path
){return guarded([&]{
    if(!runtime||!path)throw std::runtime_error("invalid Qwen KV dump arguments");
    if(!runtime->state)throw std::runtime_error("native Qwen runtime is not prepared");
    if(runtime->position==0)throw std::runtime_error("native Qwen KV dump needs at least one decoded token");
    if(layer_index>=runtime->layers.size())throw std::runtime_error("native Qwen KV dump layer index out of range");
    const auto& layer=runtime->layers[layer_index];
    if(!layer.attention)throw std::runtime_error("native Qwen KV dump requires an attention layer");

    const auto view=attention_cache_view(layer,runtime->position-1);
    const int head_dim=static_cast<int>(layer.head_dim);
    const int kv_heads=static_cast<int>(layer.kv_heads);
    const int capacity=view.capacity,tokens=view.tokens;
    const int ck=runtime->options.cache_type_k,cv=runtime->options.cache_type_v;
    if((ck>=3||cv>=3)&&head_dim%32!=0)
        throw std::runtime_error("native Qwen KV dump cannot decode a blocked cache type with a head_dim that is not a multiple of 32");

    const std::uint64_t elements=static_cast<std::uint64_t>(kv_heads)*capacity*head_dim;
    auto region_bytes=[&](int type){return kv_type_bytes(elements,type);};

    std::vector<std::uint8_t> raw_keys(region_bytes(ck)),raw_values(region_bytes(cv));
    if(colibri_gpu_download(raw_keys.data(),runtime->state+layer.state_first,raw_keys.size(),runtime->stream)!=0
       ||colibri_gpu_download(raw_values.data(),runtime->state+layer.state_second,raw_values.size(),runtime->stream)!=0)
        throw std::runtime_error("failed to download native Qwen KV cache");
    if(colibri_gpu_stream_sync(runtime->stream)!=0)
        throw std::runtime_error("failed to synchronize native Qwen KV dump");

    auto element=[&](const std::uint8_t* raw,int type,std::uint64_t row,int d)->float{
        if(type==3){
            const auto* base=raw+row*static_cast<std::uint64_t>(head_dim/32)*kQ8BlockSize;
            return qwen_q8_value(base,static_cast<std::uint64_t>(d));
        }
        const std::uint64_t index=row*head_dim+d;
        if(type==0){float value=0.0f;std::memcpy(&value,raw+index*4,4);return value;}
        std::uint16_t bits=0;std::memcpy(&bits,raw+index*2,2);
        if(type==2){const std::uint32_t wide=static_cast<std::uint32_t>(bits)<<16;float value=0.0f;std::memcpy(&value,&wide,4);return value;}
        return qwen_half_value(bits);
    };

    const std::int32_t count=kv_heads*tokens;
    std::vector<float> keys(static_cast<std::size_t>(count)*head_dim);
    std::vector<float> values(static_cast<std::size_t>(count)*head_dim);
    // A turbo cache stores rotated rows, so decoding one means undoing the
    // rotation with the same sign stream the store kernel used (0 for keys,
    // 1 for values). Routing that through the CPU reference in turboquant.h
    // also means this dump reads back what the kernels wrote using the
    // definition the contract test pins.
    std::vector<float> rotated(static_cast<std::size_t>(head_dim));
    auto fill=[&](const std::uint8_t* raw,int type,std::uint64_t row,float* out,std::uint32_t stream){
        if(kv_type_is_turbo(type)){
            const auto turbo=type==4?TurboType::Turbo3:TurboType::Turbo4;
            const auto stride=static_cast<std::uint64_t>(head_dim/kTurboBlock)*turbo_block_bytes(turbo);
            turbo_decode_vector(raw+row*stride,head_dim,turbo,rotated.data());
            turbo_inverse_rotate(rotated.data(),out,head_dim,stream);
            return;
        }
        for(int d=0;d<head_dim;++d)out[d]=element(raw,type,row,d);
    };
    for(int head=0;head<kv_heads;++head)for(int t=0;t<tokens;++t){
        const std::uint64_t row=static_cast<std::uint64_t>(head)*capacity+(view.first+t)%capacity;
        const auto offset=(static_cast<std::size_t>(head)*tokens+t)*head_dim;
        fill(raw_keys.data(),ck,row,keys.data()+offset,0u);
        fill(raw_values.data(),cv,row,values.data()+offset,1u);
    }

    std::ofstream out(path,std::ios::binary);
    if(!out)throw std::runtime_error("failed to open native Qwen KV dump for writing");
    const std::int32_t header[2]={count,static_cast<std::int32_t>(head_dim)};
    out.write(reinterpret_cast<const char*>(header),sizeof(header));
    out.write(reinterpret_cast<const char*>(keys.data()),
              static_cast<std::streamsize>(keys.size()*sizeof(float)));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size()*sizeof(float)));
    if(!out)throw std::runtime_error("failed to write native Qwen KV dump");
    return 0;
});}

#include "v2_mtp_verifier.inc"

static int gemma4_decode(ColibriV2QwenRuntime& runtime, std::uint32_t input_token,
                         std::uint32_t& output_token) {
    const auto decode_started=std::chrono::steady_clock::now();
    const auto expert_policy=qwen_expert_policy(
        runtime,colibri::v2::ExpertExecutionPhase::decode);
    const int hidden_size=static_cast<int>(runtime.model->config.hidden_size);
    const int experts=static_cast<int>(runtime.model->config.expert_count);
    const int top_k=static_cast<int>(runtime.model->config.expert_used_count);
    const int dense_intermediate=static_cast<int>(runtime.model->config.dense_intermediate_size);
    const float epsilon=runtime.model->config.rms_norm_epsilon
        ?runtime.model->config.rms_norm_epsilon:1.0e-6f;
    std::uint64_t cursor=runtime.workspace;
    const auto workspace_end=runtime.workspace+runtime.workspace_bytes;
    auto take=[&](std::uint64_t bytes){const auto result=cursor;cursor+=device_align(bytes);if(cursor>workspace_end)throw std::runtime_error("native Gemma 4 workspace is too small");return result;};
    std::uint64_t hidden=take(hidden_size*sizeof(float));
    std::uint64_t residual=take(hidden_size*sizeof(float));
    const std::uint64_t normalized=take(hidden_size*sizeof(float));
    const std::uint64_t first=take(runtime.scratch_elements*sizeof(float));
    const std::uint64_t second=take(runtime.scratch_elements*sizeof(float));
    const std::uint64_t third=take(runtime.scratch_elements*sizeof(float));
    const std::uint64_t fourth=take(runtime.scratch_elements*sizeof(float));
    const std::uint64_t activated=take(static_cast<std::uint64_t>(top_k)*runtime.moe_intermediate*sizeof(float));
    const std::uint64_t router_logits=take(experts*sizeof(float));
    const std::uint64_t selected_device=take(top_k*sizeof(std::int32_t));
    const std::uint64_t route_weights=take(top_k*sizeof(float));
    const std::uint64_t argmax_device=take(sizeof(std::uint64_t));
    const std::uint64_t attention_scores=take(
        static_cast<std::uint64_t>(runtime.model->config.attention_heads)*
        runtime.options.context_limit*sizeof(float));
    auto* staging=static_cast<std::uint8_t*>(runtime.host_staging);
    auto* selected_host=reinterpret_cast<std::int32_t*>(staging);
    const auto weights_offset=device_align(top_k*sizeof(std::int32_t));
    const auto input_offset=weights_offset+device_align(top_k*sizeof(float));
    const auto activated_offset=input_offset+device_align(hidden_size*sizeof(float));
    const auto output_offset=activated_offset+
        device_align(static_cast<std::uint64_t>(top_k)*runtime.moe_intermediate*sizeof(float));
    auto* cpu_weights=reinterpret_cast<float*>(staging+weights_offset);
    auto* cpu_input=reinterpret_cast<float*>(staging+input_offset);
    auto* cpu_activated=reinterpret_cast<float*>(staging+activated_offset);
    auto* cpu_output=reinterpret_cast<float*>(staging+output_offset);
    if(output_offset+hidden_size*sizeof(float)>runtime.host_staging_bytes)
        throw std::runtime_error("native Gemma 4 CPU MoE workspace overflow");
    auto launch=[&](const char* name,std::uint32_t grid_x,std::uint32_t grid_y,
                    std::uint32_t block_x,void** arguments,std::uint32_t shared=0){
        if(colibri_gpu_launch_named(name,grid_x,grid_y,block_x,shared,runtime.stream,arguments)!=0)
            throw std::runtime_error(std::string("native Gemma 4 CUDA kernel failed: ")+name);
    };
    auto rms=[&](std::uint64_t input,std::uint64_t weights,std::uint64_t output){
        int one_centered=0;
        void* args[]={&input,&weights,&output,const_cast<int*>(&hidden_size),
                      const_cast<float*>(&epsilon),&one_centered};
        launch("rms_norm",1,1,256,args);
    };
    auto q4=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,
                int input_size,int output_size){
        void* args[]={&matrix,&input,&output,&input_size,&output_size};
        launch("gemma_q4_0_matvec",(output_size+7)/8,1,256,args);
    };
    auto f32=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,
                 int input_size,int output_size){
        void* args[]={&matrix,&input,&output,&input_size,&output_size};
        launch("qwen_f32_matvec_warp",(output_size+7)/8,1,256,args);
    };
    auto add=[&](std::uint64_t target,std::uint64_t source){
        float scale=1.0f;int count=hidden_size;
        void* args[]={&target,&source,&scale,&count};
        launch("scaled_add",(hidden_size+255)/256,1,256,args);
    };
    {
        const auto embedding=runtime.device_tensors[runtime.token_embeddings];
        const int token=static_cast<int>(input_token);
        const float scale=std::sqrt(static_cast<float>(hidden_size));
        void* args[]={const_cast<std::uint64_t*>(&embedding),&hidden,
                      const_cast<int*>(&token),const_cast<int*>(&hidden_size),
                      const_cast<float*>(&scale)};
        launch("gemma_q4_0_embedding",(hidden_size+255)/256,1,256,args);
    }
    for(std::uint32_t layer_number=0;layer_number<runtime.layers.size();++layer_number){
        auto& layer=runtime.layers[layer_number];
        auto tensor=[&](std::size_t role){return runtime.device_tensors[layer.static_tensors.at(role)];};

        // Attention: learned pre/post RMS norms, per-head Q/K normalization,
        // unscaled dot products, and a layer-specific local/global KV shape.
        rms(hidden,tensor(0),normalized);
        const int heads=static_cast<int>(layer.attention_heads);
        const int kv_heads=static_cast<int>(layer.kv_heads);
        const int head_dim=static_cast<int>(layer.head_dim);
        const int q_elements=heads*head_dim,kv_elements=kv_heads*head_dim;
        q4(tensor(1),normalized,first,hidden_size,q_elements);
        q4(tensor(2),normalized,second,hidden_size,kv_elements);
        q4(tensor(3),normalized,third,hidden_size,kv_elements);
        const int rotary=static_cast<int>(layer.rotary_dim);
        const int position=static_cast<int>(runtime.position);
        const float theta=layer.rope_theta;
        const std::uint64_t rope_factors=layer.attention_window||
            runtime.rope_factors==std::numeric_limits<std::uint64_t>::max()
            ?0:runtime.device_tensors[runtime.rope_factors];
        auto qnorm=tensor(5),knorm=tensor(6);
        void* q_args[]={const_cast<std::uint64_t*>(&first),&qnorm,
                        const_cast<std::uint64_t*>(&fourth),const_cast<int*>(&heads),
                        const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                        const_cast<int*>(&position),const_cast<float*>(&theta),
                        const_cast<float*>(&epsilon),const_cast<std::uint64_t*>(&rope_factors)};
        launch("gemma_head_norm_rope",heads,1,256,q_args);
        void* k_args[]={const_cast<std::uint64_t*>(&second),&knorm,
                        const_cast<std::uint64_t*>(&first),const_cast<int*>(&kv_heads),
                        const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                        const_cast<int*>(&position),const_cast<float*>(&theta),
                        const_cast<float*>(&epsilon),const_cast<std::uint64_t*>(&rope_factors)};
        launch("gemma_head_norm_rope",kv_heads,1,256,k_args);
        void* v_norm_args[]={const_cast<std::uint64_t*>(&third),
                             const_cast<std::uint64_t*>(&second),const_cast<int*>(&kv_heads),
                             const_cast<int*>(&head_dim),const_cast<float*>(&epsilon)};
        launch("gemma_head_rms",kv_heads,1,256,v_norm_args);
        std::uint64_t cache_keys=runtime.state+layer.state_first;
        std::uint64_t cache_values=runtime.state+layer.state_second;
        const auto view=attention_cache_view(layer,runtime.position);
        int slot=view.slot,capacity=view.capacity;
        void* k_store_args[]={const_cast<std::uint64_t*>(&first),&cache_keys,
                              const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),
                              &slot,&capacity};
        launch(kv_store_kernel(runtime.options.cache_type_k,true),kv_heads,1,256,k_store_args);
        void* v_store_args[]={const_cast<std::uint64_t*>(&second),&cache_values,
                              const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),
                              &slot,&capacity};
        launch(kv_store_kernel(runtime.options.cache_type_v,false),kv_heads,1,256,v_store_args);
        int tokens=view.tokens,first_slot=view.first;float attention_scale=1.0f;
        void* score_args[]={const_cast<std::uint64_t*>(&fourth),&cache_keys,
                            const_cast<std::uint64_t*>(&attention_scores),
                            const_cast<int*>(&heads),const_cast<int*>(&kv_heads),
                            const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot,
                            &attention_scale};
        launch(kv_scores_ring_kernel(runtime),heads,(tokens+255)/256,256,score_args);
        void* value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,
                            const_cast<std::uint64_t*>(&third),const_cast<int*>(&heads),
                            const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),
                            &tokens,&capacity,&first_slot};
        launch(kv_values_ring_kernel(runtime),heads,1,256,value_args);
        q4(tensor(4),third,residual,q_elements,hidden_size);
        rms(residual,tensor(7),normalized);
        add(hidden,normalized);

        // Dense GEGLU path.
        rms(hidden,tensor(8),normalized);
        auto dense_gate=tensor(9),dense_up=tensor(10);
        void* dense_args[]={&dense_gate,&dense_up,const_cast<std::uint64_t*>(&normalized),
                            const_cast<std::uint64_t*>(&first),const_cast<int*>(&hidden_size),
                            const_cast<int*>(&dense_intermediate)};
        launch("gemma_q4_0_geglu",(dense_intermediate+7)/8,1,256,dense_args);
        q4(tensor(11),first,second,dense_intermediate,hidden_size);
        rms(second,tensor(12),third);

        // Router consumes the pre-FFN residual. Experts consume their own
        // learned normalization of that same residual and execute on CPU.
        auto router_scale=tensor(13);
        void* router_input_args[]={&hidden,&router_scale,
                                   const_cast<std::uint64_t*>(&normalized),
                                   const_cast<int*>(&hidden_size),
                                   const_cast<float*>(&epsilon)};
        launch("gemma_router_input",1,1,256,router_input_args);
        f32(tensor(14),normalized,router_logits,hidden_size,experts);
        if(colibri_gpu_route_topk(router_logits,selected_device,route_weights,
                                 experts,top_k,runtime.stream)!=0)
            throw std::runtime_error("native Gemma 4 routing failed");
        rms(hidden,tensor(15),normalized);
        if(colibri_gpu_download(selected_host,selected_device,top_k*sizeof(std::int32_t),runtime.stream)!=0||
           colibri_gpu_download(cpu_weights,route_weights,top_k*sizeof(float),runtime.stream)!=0||
           colibri_gpu_download(cpu_input,normalized,hidden_size*sizeof(float),runtime.stream)!=0)
            throw std::runtime_error("native Gemma 4 CPU MoE input transfer failed");
        if(colibri_gpu_event_record(runtime.route_event,runtime.stream)!=0||
           colibri_gpu_event_sync(runtime.route_event)!=0)
            throw std::runtime_error("native Gemma 4 route synchronization failed");
        runtime.route_expert_sum+=top_k;
        if(expert_policy.is_cpu()){
            const auto compute_started=std::chrono::steady_clock::now();
            gemma_cpu_moe(runtime,layer,selected_host,cpu_weights,top_k,cpu_input,
                          cpu_activated,cpu_output);
            runtime.expert_compute_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-compute_started).count();
            if(colibri_gpu_upload(fourth,cpu_output,hidden_size*sizeof(float),runtime.stream)!=0)
                throw std::runtime_error("native Gemma 4 CPU MoE output upload failed");
        }else{
            const auto pager_started=std::chrono::steady_clock::now();
            std::array<std::int32_t,256> cpu_selected{};
            std::array<float,256> cpu_compact_weights{},gpu_compact_weights{};
            std::array<std::uint64_t,256> gate_up_pointers{},down_pointers{};
            int cpu_count=0,gpu_count=0;
            const auto& gate_up_tensor=runtime.model->tensors[layer.expert_tensors[0]];
            const auto& down_tensor=runtime.model->tensors[layer.expert_tensors[1]];
            const auto& scale_tensor=runtime.model->tensors[layer.expert_tensors[2]];
            const auto gate_up_bytes=gate_up_tensor.size/experts;
            const auto down_bytes=down_tensor.size/experts;
            const auto scale_bytes=scale_tensor.size/experts;
            const auto* expert_scales=reinterpret_cast<const float*>(tensor_data(*runtime.model,scale_tensor));
            std::uint64_t paging_cursor=device_align(output_offset+hidden_size*sizeof(float));
            for(int rank=0;rank<top_k;++rank){
                const int expert=selected_host[rank];
                if(expert<0||expert>=experts)throw std::runtime_error("native Gemma 4 selected an invalid expert");
                const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
                auto resident=runtime.expert_residency.find(cache_key);
                if(resident!=runtime.expert_residency.end()){
                    const auto slot_index=resident->second;
                    auto& cache_slot=runtime.expert_slots[slot_index];
                    const auto&history=record_expert_access(
                        runtime,layer_number,expert);
                    if(expert_policy.residency_may_change())
                        cache_slot.last_used=history.last_used;
                    const auto base=runtime.expert_cache+slot_index*runtime.expert_slot_bytes;
                    gate_up_pointers[gpu_count]=base;
                    down_pointers[gpu_count]=base+gate_up_bytes;
                    gpu_compact_weights[gpu_count]=cpu_weights[rank]*expert_scales[expert];
                    ++gpu_count;record_expert_cache_hit(runtime,cache_slot);
                    continue;
                }
                ++runtime.expert_cache_misses;
                cpu_selected[cpu_count]=expert;
                cpu_compact_weights[cpu_count]=cpu_weights[rank];
                ++cpu_count;
                if(runtime.expert_slots.empty())continue;
                const auto slot_index=select_expert_cache_slot(runtime,layer_number,expert,true);
                if(slot_index==kNoExpertSlot)continue;
                auto& cache_slot=runtime.expert_slots[slot_index];
                cache_slot.key=cache_key;cache_slot.valid=true;cache_slot.last_used=runtime.expert_clock;
                runtime.expert_residency[cache_key]=slot_index;
                const auto bundle_bytes=gate_up_bytes+down_bytes+scale_bytes;
                if(paging_cursor+bundle_bytes>runtime.host_staging_bytes)
                    throw std::runtime_error("native Gemma 4 hybrid paging workspace overflow");
                std::memcpy(staging+paging_cursor,tensor_data(*runtime.model,gate_up_tensor)+static_cast<std::uint64_t>(expert)*gate_up_bytes,gate_up_bytes);
                std::memcpy(staging+paging_cursor+gate_up_bytes,tensor_data(*runtime.model,down_tensor)+static_cast<std::uint64_t>(expert)*down_bytes,down_bytes);
                std::memcpy(staging+paging_cursor+gate_up_bytes+down_bytes,tensor_data(*runtime.model,scale_tensor)+static_cast<std::uint64_t>(expert)*scale_bytes,scale_bytes);
                const auto base=runtime.expert_cache+slot_index*runtime.expert_slot_bytes;
                if(colibri_gpu_upload(base,staging+paging_cursor,bundle_bytes,runtime.stream)!=0)
                    throw std::runtime_error("native Gemma 4 hybrid expert upload failed");
                paging_cursor+=device_align(bundle_bytes);
            }
            if(colibri_gpu_memset(fourth,0,hidden_size*sizeof(float),runtime.stream)!=0)
                throw std::runtime_error("native Gemma 4 hybrid output reset failed");
            if(gpu_count){
                const auto table_bytes=device_align(static_cast<std::uint64_t>(gpu_count)*(
                    2*sizeof(std::uint64_t)+sizeof(float)));
                const auto table_host=device_align(paging_cursor);
                if(table_host+table_bytes>runtime.host_staging_bytes||table_bytes>runtime.expert_staging_bytes)
                    throw std::runtime_error("native Gemma 4 hybrid pointer workspace overflow");
                std::memcpy(staging+table_host,gate_up_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+gpu_count*sizeof(std::uint64_t),down_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+2*gpu_count*sizeof(std::uint64_t),gpu_compact_weights.data(),gpu_count*sizeof(float));
                const auto table_device=runtime.expert_staging+runtime.expert_staging_bytes-table_bytes;
                if(colibri_gpu_upload(table_device,staging+table_host,table_bytes,runtime.stream)!=0)
                    throw std::runtime_error("native Gemma 4 hybrid pointer upload failed");
                const auto gate_table=table_device;
                const auto down_table=gate_table+gpu_count*sizeof(std::uint64_t);
                const auto weight_table=down_table+gpu_count*sizeof(std::uint64_t);
                const int intermediate=static_cast<int>(runtime.moe_intermediate);
                void* gate_args[]={const_cast<std::uint64_t*>(&gate_table),
                                   const_cast<std::uint64_t*>(&normalized),
                                   const_cast<std::uint64_t*>(&activated),
                                   const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate),&gpu_count};
                launch("gemma_q4_0_grouped_geglu",(intermediate+7)/8,gpu_count,256,gate_args);
                void* down_args[]={const_cast<std::uint64_t*>(&down_table),
                                   const_cast<std::uint64_t*>(&activated),
                                   const_cast<std::uint64_t*>(&fourth),
                                   const_cast<std::uint64_t*>(&weight_table),
                                   const_cast<int*>(&intermediate),const_cast<int*>(&hidden_size),&gpu_count};
                launch("gemma_q4_0_grouped_accumulate",(hidden_size+7)/8,1,256,down_args);
            }
            if(cpu_count){
                const auto compute_started=std::chrono::steady_clock::now();
                gemma_cpu_moe(runtime,layer,cpu_selected.data(),cpu_compact_weights.data(),
                              cpu_count,cpu_input,cpu_activated,cpu_output);
                runtime.expert_compute_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-compute_started).count();
                if(colibri_gpu_upload(first,cpu_output,hidden_size*sizeof(float),runtime.stream)!=0)
                    throw std::runtime_error("native Gemma 4 hybrid CPU output upload failed");
                add(fourth,first);
            }
            runtime.expert_page_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-pager_started).count();
        }
        rms(fourth,tensor(16),first);
        add(third,first);
        rms(third,tensor(17),residual);
        add(hidden,residual);
        auto layer_scale=tensor(18);
        void* scale_args[]={&hidden,&layer_scale,const_cast<int*>(&hidden_size)};
        launch("gemma_scale_vector",(hidden_size+255)/256,1,256,scale_args);
    }
    rms(hidden,runtime.device_tensors[runtime.final_norm],normalized);
    const int vocabulary=static_cast<int>(runtime.model->config.vocabulary_size);
    if(colibri_gpu_memset(argmax_device,0,sizeof(std::uint64_t),runtime.stream)!=0)
        throw std::runtime_error("native Gemma 4 argmax reset failed");
    auto lm_head=runtime.device_tensors[runtime.lm_head];
    void* argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),
                         const_cast<std::uint64_t*>(&argmax_device),
                         const_cast<int*>(&hidden_size),const_cast<int*>(&vocabulary)};
    launch("gemma_q4_0_lm_argmax",(vocabulary+7)/8,1,256,argmax_args);
    auto* packed_winner=reinterpret_cast<std::uint64_t*>(staging);
    if(colibri_gpu_download(packed_winner,argmax_device,sizeof(*packed_winner),runtime.stream)!=0||
       colibri_gpu_stream_sync(runtime.stream)!=0)
        throw std::runtime_error("native Gemma 4 output synchronization failed");
    output_token=0xffffffffu-static_cast<std::uint32_t>(*packed_winner);
    runtime.last_output_token=output_token;
    runtime.processed_tokens.push_back(input_token);
    ++runtime.position;
    ++runtime.decode_calls;
    runtime.decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-decode_started).count();
    return 0;
}

int colibri_v2_qwen_runtime_synchronize(ColibriV2QwenRuntime*runtime){return guarded([&]{if(!runtime||!runtime->stream)throw std::runtime_error("native Qwen runtime is not prepared");if(colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("native Qwen CUDA synchronization failed");return 0;});}
int colibri_v2_qwen_runtime_decode(ColibriV2QwenRuntime*runtime,uint32_t input_token,uint32_t*output_token){return guarded([&]{
    if(!runtime||!output_token)throw std::runtime_error("invalid native Qwen decode arguments");
    if(!runtime->decode_ready)throw std::runtime_error("native Qwen runtime is not prepared for decode");
    if(runtime->cancelled)throw std::runtime_error("native Qwen runtime is cancelled");
    if(runtime->position>=runtime->options.context_limit)throw std::runtime_error("native Qwen context limit exceeded");
    if(input_token>=runtime->model->config.vocabulary_size)throw std::runtime_error("native Qwen input token is out of range");
    if(runtime->cache_admission_enabled)
        qwen_freeze_expert_residency(*runtime);
    if(runtime->gemma4)return gemma4_decode(*runtime,input_token,*output_token);
    const auto expert_phase=runtime->cache_admission_enabled
        ?colibri::v2::ExpertExecutionPhase::decode
        :colibri::v2::ExpertExecutionPhase::prefill;
    const auto expert_policy=qwen_expert_policy(*runtime,expert_phase);
    qwen_mtp_append_prompt_pair(*runtime,input_token);
    const auto decode_started=std::chrono::steady_clock::now();
    const int hidden_size=static_cast<int>(runtime->model->config.hidden_size);
    const int experts=static_cast<int>(runtime->model->config.expert_count);
    // Optionally route fewer experts than the model trained with (top-k pruning):
    // route_topk renormalizes the softmax over the kept experts, and every
    // downstream buffer/loop is sized from this count, so the whole MoE phase
    // (routing, DtoH, GPU grouped kernels, CPU experts) shrinks proportionally.
    const int configured_top_k=static_cast<int>(runtime->model->config.expert_used_count);
    const int top_k=(runtime->options.expert_top_k>0&&static_cast<int>(runtime->options.expert_top_k)<configured_top_k)?static_cast<int>(runtime->options.expert_top_k):configured_top_k;
    const float epsilon=runtime->model->config.rms_norm_epsilon?runtime->model->config.rms_norm_epsilon:1.0e-6f;
    const auto&workspace_layout=runtime->decode_workspace_layout;
    if(workspace_layout.bytes>runtime->workspace_bytes)
        throw std::runtime_error("native Qwen workspace is too small");
    std::uint64_t hidden=workspace_layout.hidden.address(runtime->workspace);
    std::uint64_t residual=workspace_layout.residual.address(runtime->workspace);
    const std::uint64_t normalized=workspace_layout.normalized.address(runtime->workspace);
    const std::uint64_t first=workspace_layout.first.address(runtime->workspace);
    const std::uint64_t second=workspace_layout.second.address(runtime->workspace);
    const std::uint64_t third=workspace_layout.third.address(runtime->workspace);
    const std::uint64_t fourth=workspace_layout.fourth.address(runtime->workspace);
    const std::uint64_t dense_q8=workspace_layout.dense_q8.address(runtime->workspace);
    const std::uint64_t dense_q8_scales=workspace_layout.dense_q8_scales.address(runtime->workspace);
    const std::uint64_t activated=workspace_layout.activated.address(runtime->workspace);
    const std::uint64_t router_logits=workspace_layout.router_logits.address(runtime->workspace);
    const std::uint64_t selected_device=workspace_layout.selected_device.address(runtime->workspace);
    const std::uint64_t route_weights=workspace_layout.route_weights.address(runtime->workspace);
    const std::uint64_t logits=workspace_layout.logits.address(runtime->workspace);
    const std::uint64_t argmax_device=workspace_layout.argmax_device.address(runtime->workspace);
    const std::uint64_t attention_scores=workspace_layout.attention_scores.address(runtime->workspace);
    auto*staging=static_cast<std::uint8_t*>(runtime->host_staging);
    auto*selected_host=reinterpret_cast<std::int32_t*>(staging);
    std::uint64_t launch_stream=runtime->stream;
    // COLIBRI_KERNEL_PROFILE=1 brackets every decode launch with CUDA events and
    // accumulates GPU time per kernel name. The events serialize the stream, so
    // the absolute total runs high; the point is the relative split, which is
    // what says whether a token is spent reading weights or somewhere else.
    static const bool lm_diagnostics=std::getenv("COLIBRI_LM_DIAG")!=nullptr;
    static const bool kernel_profile=
        std::getenv("COLIBRI_KERNEL_PROFILE")&&
        std::getenv("COLIBRI_KERNEL_PROFILE")[0]=='1';
    struct KernelProfileEntry{double milliseconds=0.0;std::uint64_t calls=0;};
    static std::map<std::string,KernelProfileEntry> kernel_profile_totals;
    // Events are recorded on the stream and only read back once the token is
    // done. Syncing per launch would stall the CPU, and an idle GPU drops from
    // ~2.2 GHz to a few hundred MHz -- which silently rescales every number the
    // profile reports. Nothing here blocks until the decode has been issued.
    static constexpr std::size_t kKernelProfileCapacity=4096;
    static std::vector<std::uint64_t> kernel_profile_events;
    static std::vector<std::string> kernel_profile_labels;
    static std::size_t kernel_profile_used=0;
    static std::uint64_t kernel_profile_overflow=0;
    static std::uint64_t kernel_profile_decodes=0;
    if(kernel_profile&&kernel_profile_events.empty()){
        kernel_profile_events.resize(2*kKernelProfileCapacity);
        kernel_profile_labels.resize(kKernelProfileCapacity);
        for(auto&event:kernel_profile_events)
            colibri_gpu_timed_event_create(&event);
    }
    auto launch_named=[&](const char*name,std::uint32_t grid_x,std::uint32_t grid_y,std::uint32_t block_x,void**arguments,std::uint32_t shared=0){
        const bool traced=kernel_profile&&kernel_profile_used<kKernelProfileCapacity;
        const std::size_t slot=kernel_profile_used;
        if(traced)colibri_gpu_event_record(kernel_profile_events[2*slot],launch_stream);
        if(colibri_gpu_launch_named(name,grid_x,grid_y,block_x,shared,launch_stream,arguments)!=0)throw std::runtime_error(std::string("native Qwen CUDA kernel failed: ")+name);
        if(traced){
            colibri_gpu_event_record(kernel_profile_events[2*slot+1],launch_stream);
            char key[192];
            std::snprintf(key,sizeof(key),"%s grid=%u",name,grid_x);
            kernel_profile_labels[slot]=key;
            ++kernel_profile_used;
        }else if(kernel_profile)++kernel_profile_overflow;
    };
    auto kernel_profile_collect=[&]{
        if(!kernel_profile)return;
        for(std::size_t slot=0;slot<kernel_profile_used;++slot){
            float milliseconds=0.0f;
            if(colibri_gpu_event_elapsed(kernel_profile_events[2*slot],
                                         kernel_profile_events[2*slot+1],
                                         &milliseconds)!=0)continue;
            auto&entry=kernel_profile_totals[kernel_profile_labels[slot]];
            entry.milliseconds+=milliseconds;++entry.calls;
        }
        kernel_profile_used=0;
    };
    auto kernel_profile_report=[&]{
        if(!kernel_profile)return;
        double total=0.0;

        for(const auto&entry:kernel_profile_totals)total+=entry.second.milliseconds;
        std::vector<std::pair<std::string,KernelProfileEntry>> sorted(
            kernel_profile_totals.begin(),kernel_profile_totals.end());
        std::sort(sorted.begin(),sorted.end(),[](const auto&a,const auto&b){
            return a.second.milliseconds>b.second.milliseconds;});
        std::uint64_t calls=0;
        for(const auto&entry:kernel_profile_totals)calls+=entry.second.calls;
        const double decodes=static_cast<double>(kernel_profile_decodes?kernel_profile_decodes:1);
        std::fprintf(stderr,
            "[kernel-profile] %.2f ms/token of kernel time over %.0f decodes, "
            "%.0f launches/token across %zu distinct kernels (untraced=%llu)\n",
            total/decodes,decodes,calls/decodes,kernel_profile_totals.size(),
            static_cast<unsigned long long>(kernel_profile_overflow));
        for(const auto&entry:sorted)
            std::fprintf(stderr,"[kernel-profile]   %-52s %7.3f ms/tok %6.1f%%  %5.1f calls/tok %7.1f us\n",
                entry.first.c_str(),entry.second.milliseconds/decodes,
                100.0*entry.second.milliseconds/total,
                entry.second.calls/decodes,
                1000.0*entry.second.milliseconds/entry.second.calls);
        kernel_profile_totals.clear();
        kernel_profile_overflow=0;kernel_profile_decodes=0;
    };
    // The Q8 copy of `normalized` is reused across the projections that share
    // it, so rms -- its only writer -- has to drop that memo.
    std::uint64_t q8_cached_input=0;
    auto rms=[&](std::uint64_t input,std::uint64_t weights,std::uint64_t output){int one_centered=0;q8_cached_input=0;void*args[]={&input,&weights,&output,const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon),&one_centered};launch_named("rms_norm",1,1,1024,args);};
    auto q8=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,int input_size,int output_size){if(colibri_gpu_q8_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)!=0)throw std::runtime_error("native Qwen Q8 projection failed");};
    auto f32=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,int input_size,int output_size){void*args[]={&matrix,&input,&output,&input_size,&output_size};launch_named("qwen_f32_matvec_warp",(output_size+7)/8,1,256,args);};
    const char*iq2_q8_setting=std::getenv("COLIBRI_IQ2_Q8_DECODE");
    const bool iq2_q8_enabled=
        !iq2_q8_setting||iq2_q8_setting[0]!='0';
    // Quantize the activation to 32-value Q8 blocks once, then run a group-wise
    // DP4A matvec instead of reconstructing every weight in f32. Requires a
    // superblock-aligned row so the Q8 blocks line up with the weight groups.
    auto q8_decode=[&](const char*kernel,std::uint64_t matrix,std::uint64_t input,
                       std::uint64_t output,int input_size,int output_size){
        if(!iq2_q8_enabled||(input_size&255))return false;
        // q/k/v and gate/up all project the same normalized vector, so the
        // quantization is hoisted out of the repeats. Only `normalized` is
        // memoized: every other source buffer is rewritten in place by kernels
        // this lambda cannot see.
        if(input!=normalized||input!=q8_cached_input){
            void*quant_args[]={&input,const_cast<std::uint64_t*>(&dense_q8),
                const_cast<std::uint64_t*>(&dense_q8_scales),&input_size};
            launch_named("quantize_q8_blocks",(input_size+31)/32,1,32,quant_args);
            q8_cached_input=(input==normalized)?input:0;
        }
        void*matvec_args[]={&matrix,const_cast<std::uint64_t*>(&dense_q8),
            const_cast<std::uint64_t*>(&dense_q8_scales),&output,
            &input_size,&output_size};
        launch_named(kernel,output_size,1,128,matvec_args);
        return true;
    };
    // Dense projections keep whatever type the checkpoint stored. The NVFP4 Qwen3.6
    // builds ship attention/SSM/router weights as bf16, older checkpoints as Q8_0;
    // reading one as the other reinterprets the bytes and drives the residual stream
    // to ~1e7, so the tensor type has to pick the kernel.
    auto dense_matvec=[&](std::size_t index,std::uint64_t input,std::uint64_t output,int input_size,int output_size){
        std::uint64_t matrix=runtime->device_tensors[index];
        const auto type=qwen_device_type(*runtime,index);
        switch(type){
            case 0:{void*args[]={&matrix,&input,&output,&input_size,&output_size};launch_named("qwen_f32_matvec_warp",(output_size+7)/8,1,256,args);return;}
            // bf16_matvec takes (rows, columns), the reverse of the quantized matvecs.
            case 30:{void*args[]={&matrix,&input,&output,&output_size,&input_size};launch_named("bf16_matvec_warp",(output_size+7)/8,1,256,args);return;}
            case 8:if(colibri_gpu_q8_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;break;
            case 16:
                if(q8_decode("iq2xxs_q8_matvec_transposed_warp",matrix,input,output,input_size,output_size))return;
                if(colibri_gpu_iq2xxs_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;
                break;
            case 18:
                if(q8_decode("iq3xxs_q8_matvec_transposed_warp",matrix,input,output,input_size,output_size))return;
                if(colibri_gpu_iq3xxs_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;
                break;
            case 22:if(colibri_gpu_iq2s_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;break;
            case 21:if(colibri_gpu_iq3s_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;break;
            case 17:if(colibri_gpu_iq2xs_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;break;
            case 23:if(colibri_gpu_iq4xs_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;break;
            case 10:
                if(q8_decode("q2k_q8_matvec_transposed_warp",matrix,input,output,input_size,output_size))return;
                if(colibri_gpu_q2k_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;
                break;
            case 11:
                if(q8_decode("q3k_q8_matvec_transposed_warp",matrix,input,output,input_size,output_size))return;
                if(colibri_gpu_q3k_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;
                break;
            case 13:
                if(q8_decode("q5k_q8_matvec_transposed_warp",matrix,input,output,input_size,output_size))return;
                if(colibri_gpu_q5k_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;
                break;
            case 12:
                if(q8_decode("q4k_q8_matvec_transposed_warp",matrix,input,output,input_size,output_size))return;
                if(colibri_gpu_q4k_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;
                break;
            case 14:
                if(q8_decode("q6k_q8_matvec_transposed_warp",matrix,input,output,input_size,output_size))return;
                if(colibri_gpu_q6k_matvec_transposed(matrix,input,output,input_size,output_size,launch_stream)==0)return;
                break;
            default:throw std::runtime_error("native Qwen dense projection type is unsupported: "+std::to_string(type));
        }
        throw std::runtime_error("native Qwen dense projection failed");
    };
    auto add=[&](std::uint64_t target,std::uint64_t source){float scale=1.0f;int count=hidden_size;void*args[]={&target,&source,&scale,&count};launch_named("scaled_add",(hidden_size+255)/256,1,256,args);};
    auto profile_record=[&](std::uint64_t event){if(runtime->cuda_profile&&colibri_gpu_event_record(event,runtime->stream)!=0)throw std::runtime_error("native Qwen CUDA profiling event failed");};
    {
        const std::uint32_t embedding_token=input_token;
        const auto embedding=qwen_stage_embedding_rows(*runtime,&embedding_token,1);
        const int token=runtime->embeddings_host_resident?0:static_cast<int>(input_token);int width=hidden_size;
        void*args[]={const_cast<std::uint64_t*>(&embedding),const_cast<std::uint64_t*>(&hidden),const_cast<int*>(&token),&width};
        launch_named(qwen_embedding_kernel(qwen_device_type(*runtime,runtime->token_embeddings),false),(hidden_size+255)/256,1,256,args);
    }
    if(std::getenv("COLIBRI_LM_DIAG")){
        static int ec=0;
        if(ec<2){++ec;float v[8]={};if(colibri_gpu_download(v,hidden,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
            std::fprintf(stderr,"[diag] after_embed hidden[0..7]=% .6e % .6e % .6e % .6e % .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);}
    }
    for(std::uint32_t layer_number=0;layer_number<runtime->layers.size();++layer_number){
        auto&layer=runtime->layers[layer_number];
        auto*profile=runtime->cuda_profile?&runtime->cuda_layer_profiles[layer_number]:nullptr;
        if(profile)profile_record(profile->pre_start);
        auto tensor=[&](std::size_t role){return runtime->device_tensors[layer.static_tensors.at(role)];};
        auto dense=[&](std::size_t role,std::uint64_t input,std::uint64_t output,int input_size,int output_size){dense_matvec(layer.static_tensors.at(role),input,output,input_size,output_size);};
        rms(hidden,tensor(0),normalized);
        std::size_t moe_base=0;
        if(!layer.attention){
            int channels=static_cast<int>(runtime->model->tensors[layer.static_tensors[1]].shape[1]);
            int gate_elements=static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1]);
            int value_heads=static_cast<int>(runtime->model->tensors[layer.static_tensors[8]].shape[0]);
            int head_dim=static_cast<int>(runtime->model->tensors[layer.static_tensors[9]].shape[0]);
            int value_dim=value_heads*head_dim;
            int key_heads=(channels-value_dim)/(2*head_dim);
            dense(1,normalized,first,hidden_size,channels);
            dense(2,normalized,second,hidden_size,gate_elements);
            dense(4,normalized,third,hidden_size,value_heads);
            dense(5,normalized,third+value_heads*sizeof(float),hidden_size,value_heads);
            if(std::getenv("COLIBRI_LM_DIAG")&&layer_number==0){
                float v[4]={};colibri_gpu_stream_sync(runtime->stream);
                if(colibri_gpu_download(v,first,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                    std::fprintf(stderr,"[diag] L0 after_qkv first[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
                if(colibri_gpu_download(v,second,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                    std::fprintf(stderr,"[diag] L0 after_gate second[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
            }
            int kernel_size=static_cast<int>(runtime->model->tensors[layer.static_tensors[6]].shape[0]);
            std::uint64_t conv_state=runtime->state+layer.state_first;
            auto conv_weights=tensor(6);
            void*conv_args[]={const_cast<std::uint64_t*>(&first),&conv_weights,&conv_state,const_cast<std::uint64_t*>(&fourth),&channels,&kernel_size};
            launch_named("delta_conv_step",(channels+255)/256,1,256,conv_args);
            std::uint64_t recurrent_state=runtime->state+layer.state_second;
            auto decay=tensor(8),dt=tensor(7),norm=tensor(9);
            std::uint64_t beta=third+value_heads*sizeof(float);
            void*recurrent_args[]={const_cast<std::uint64_t*>(&fourth),const_cast<std::uint64_t*>(&second),&beta,const_cast<std::uint64_t*>(&third),&decay,&dt,&norm,&recurrent_state,const_cast<std::uint64_t*>(&first),&key_heads,&value_heads,&head_dim,const_cast<float*>(&epsilon)};
            // Split the key loop across `slices` groups of head_dim threads. Four
            // groups is the most a 1024-thread block allows at head_dim 128, which
            // is what every Qwen3.6 checkpoint uses; wider heads get fewer, and
            // anything past a 1024-thread block falls back to the serial kernel.
            // COLIBRI_DELTA_SERIAL forces that fallback for A/B measurement --
            // the split form is 2.9x here (1.92 -> 0.67 ms/token over 30 layers).
            if(profile)profile_record(profile->recurrent_start);
            int recurrent_slices=head_dim>0?1024/head_dim:0;
            if(recurrent_slices>4)recurrent_slices=4;
            if(recurrent_slices>=1&&!std::getenv("COLIBRI_DELTA_SERIAL")){
                const int recurrent_block=head_dim*recurrent_slices;
                const std::uint32_t recurrent_shared=
                    static_cast<std::uint32_t>((4*head_dim+recurrent_block)*sizeof(float));
                launch_named("qwen_delta_recurrent_split",value_heads,1,recurrent_block,recurrent_args,recurrent_shared);
            }else{
                launch_named("qwen_delta_recurrent",value_heads,1,256,recurrent_args);
            }
            if(profile)profile_record(profile->recurrent_end);
            if(std::getenv("COLIBRI_LM_DIAG")&&layer_number==0){
                float v[4]={};colibri_gpu_stream_sync(runtime->stream);
                if(colibri_gpu_download(v,fourth,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                    std::fprintf(stderr,"[diag] L0 after_recur fourth[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
                if(colibri_gpu_download(v,first,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                    std::fprintf(stderr,"[diag] L0 after_recur first[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
            }
            dense(3,first,residual,value_dim,hidden_size);
            if(std::getenv("COLIBRI_LM_DIAG")&&layer_number==0){
                float v[4]={};colibri_gpu_stream_sync(runtime->stream);
                if(colibri_gpu_download(v,residual,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                    std::fprintf(stderr,"[diag] L0 after_ssm_out residual[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
                fprintf(stderr,"[diag] L0 value_dim=%d hidden_size=%d\n",value_dim,hidden_size);
                // Also download first bytes of ssm_out tensor to verify data
                auto ssm_out_gpu=tensor(3);
                unsigned char buf[34]={};
                if(colibri_gpu_download(buf,ssm_out_gpu,sizeof(buf),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0){
                    fprintf(stderr,"[diag] L0 ssm_out bytes[0..33]: %02x %02x %02x %02x %02x %02x\n",buf[0],buf[1],buf[2],buf[3],buf[4],buf[5]);
                }
            }
            add(residual,hidden);
            moe_base=10;
        }else if(runtime->laguna){
            // Laguna attention: plain Q/K/V projections (no fused gate), per-head
            // QK RMS norm, per-layer-type RoPE, then a softplus per-head output
            // gate computed from the same pre-attention hidden state.
            const int heads=static_cast<int>(layer.attention_heads);
            const int kv_heads=static_cast<int>(layer.kv_heads);
            const int head_dim=static_cast<int>(layer.head_dim);
            const int q_size=heads*head_dim,kv_size=kv_heads*head_dim;
            if(static_cast<std::uint64_t>(q_size)+heads>runtime->scratch_elements)
                throw std::runtime_error("native Laguna attention scratch is too small");
            dense(1,normalized,first,hidden_size,q_size);
            dense(2,normalized,second,hidden_size,kv_size);
            dense(3,normalized,third,hidden_size,kv_size);
            const std::uint64_t queries=fourth;
            const std::uint64_t gates=fourth+static_cast<std::uint64_t>(q_size)*sizeof(float);
            dense(7,normalized,gates,hidden_size,heads);
            const int rotary=static_cast<int>(layer.rotary_dim);
            const int position=static_cast<int>(runtime->position);
            const float theta=layer.rope_theta;
            const float freq_scale=layer.rope_freq_scale;
            const float ext_factor=layer.rope_ext_factor;
            const float mscale=layer.rope_attn_factor;
            float corr_low=0.0f,corr_high=0.0f;
            if(ext_factor!=0.0f)
                qwen_yarn_correction_dims(
                    rotary,layer.rope_orig_context,theta,
                    layer.rope_beta_fast,layer.rope_beta_slow,corr_low,corr_high);
            auto qnorm=tensor(5),knorm=tensor(6);
            void*q_args[]={const_cast<std::uint64_t*>(&first),&qnorm,
                           const_cast<std::uint64_t*>(&queries),const_cast<int*>(&heads),
                           const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                           const_cast<int*>(&position),const_cast<float*>(&theta),
                           const_cast<float*>(&epsilon),const_cast<float*>(&freq_scale),
                           const_cast<float*>(&ext_factor),const_cast<float*>(&mscale),
                           &corr_low,&corr_high};
            launch_named("laguna_head_norm_rope",heads,1,256,q_args);
            const std::uint64_t keys=first;
            void*k_args[]={const_cast<std::uint64_t*>(&second),&knorm,
                           const_cast<std::uint64_t*>(&keys),const_cast<int*>(&kv_heads),
                           const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                           const_cast<int*>(&position),const_cast<float*>(&theta),
                           const_cast<float*>(&epsilon),const_cast<float*>(&freq_scale),
                           const_cast<float*>(&ext_factor),const_cast<float*>(&mscale),
                           &corr_low,&corr_high};
            launch_named("laguna_head_norm_rope",kv_heads,1,256,k_args);
            std::uint64_t cache_keys=runtime->state+layer.state_first,cache_values=runtime->state+layer.state_second;
            const auto view=attention_cache_view(layer,runtime->position);
            int slot=view.slot,capacity=view.capacity;
            void*k_store_args[]={const_cast<std::uint64_t*>(&keys),&cache_keys,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&slot,&capacity};
            launch_named(kv_store_kernel(runtime->options.cache_type_k,true),kv_heads,1,256,k_store_args);
            void*v_store_args[]={const_cast<std::uint64_t*>(&third),&cache_values,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&slot,&capacity};
            launch_named(kv_store_kernel(runtime->options.cache_type_v,false),kv_heads,1,256,v_store_args);
            std::uint64_t attended=second;int tokens=view.tokens,first_slot=view.first;
            float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
            void*score_args[]={const_cast<std::uint64_t*>(&queries),&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot,&scale};
            launch_named(kv_scores_ring_kernel(*runtime),heads,(tokens+255)/256,256,score_args);
            void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot};
            launch_named(kv_values_ring_kernel(*runtime),heads,1,256,value_args);
            std::uint64_t gated=third;
            void*gate_args[]={&attended,const_cast<std::uint64_t*>(&gates),&gated,
                              const_cast<int*>(&heads),const_cast<int*>(&head_dim)};
            launch_named("laguna_attention_gate",(q_size+255)/256,1,256,gate_args);
            dense(4,gated,residual,q_size,hidden_size);
            add(residual,hidden);
            moe_base=8;
        }else{
            const int heads=static_cast<int>(runtime->model->config.attention_heads);
            const int kv_heads=static_cast<int>(runtime->model->config.attention_kv_heads);
            const int head_dim=static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1]/kv_heads);
            const int q_size=heads*2*head_dim,kv_size=kv_heads*head_dim;
            dense(1,normalized,first,hidden_size,q_size);
            dense(2,normalized,second,hidden_size,kv_size);
            dense(3,normalized,third,hidden_size,kv_size);
            const int rotary=static_cast<int>(runtime->model->config.rotary_dimension?runtime->model->config.rotary_dimension:head_dim);
            const int position=static_cast<int>(runtime->position);
            const float theta=runtime->model->config.rope_freq_base?runtime->model->config.rope_freq_base:1000000.0f;
            auto qnorm=tensor(5),knorm=tensor(6);std::uint64_t queries=fourth,gates=fourth+q_size/2*sizeof(float);
            void*q_args[]={const_cast<std::uint64_t*>(&first),&qnorm,&queries,&gates,const_cast<int*>(&heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),const_cast<int*>(&position),const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
            launch_named("qwen_attention_query",heads,1,256,q_args);
            std::uint64_t keys=first;
            void*k_args[]={const_cast<std::uint64_t*>(&second),&knorm,&keys,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),const_cast<int*>(&position),const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
            launch_named("qwen_attention_key",kv_heads,1,256,k_args);
            std::uint64_t cache_keys=runtime->state+layer.state_first,cache_values=runtime->state+layer.state_second;
            const auto view=attention_cache_view(layer,runtime->position);
            int slot=view.slot,capacity=view.capacity;
            void*k_store_args[]={&keys,&cache_keys,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&slot,&capacity};
            launch_named(kv_store_kernel(runtime->options.cache_type_k,true),kv_heads,1,256,k_store_args);
            void*v_store_args[]={const_cast<std::uint64_t*>(&third),&cache_values,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&slot,&capacity};
            launch_named(kv_store_kernel(runtime->options.cache_type_v,false),kv_heads,1,256,v_store_args);
            std::uint64_t attended=second;int tokens=view.tokens,first_slot=view.first;float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
            if(profile)profile_record(profile->recurrent_start);
            const char* fused_tiles=kv_fused_tiles_kernel(*runtime);
            const int fused_tile_tokens=kv_fused_tile_tokens(*runtime);
            const bool cublas_done=
                qwen_turbo_cublas_attention(
                    *runtime,queries,first,cache_keys,cache_values,
                    attention_scores,attended,heads,kv_heads,head_dim,tokens,
                    capacity,first_slot,scale)||
                (qwen_cublas_attention_eligible(
                    *runtime,tokens,first_slot,capacity)&&
                colibri_gpu_attention_f16_cublas(
                    queries,first,cache_keys,cache_values,attention_scores,
                    attended,runtime->stream,heads,kv_heads,head_dim,tokens,
                    capacity,first_slot,scale)==0);
            if(cublas_done){
                // Tensor-core GQA attention already wrote `attended`.
            }else if(runtime->fused_attention&&fused_tiles&&
               head_dim==128&&
               heads/kv_heads<=8&&
               (tokens+fused_tile_tokens-1)/fused_tile_tokens<=512&&
               static_cast<std::uint64_t>((tokens+fused_tile_tokens-1)/fused_tile_tokens)*130<=
                   runtime->options.context_limit){
                const int tile_count=(tokens+fused_tile_tokens-1)/fused_tile_tokens;
                void*fused_args[]={&queries,&cache_keys,&cache_values,
                    const_cast<std::uint64_t*>(&attention_scores),
                    const_cast<int*>(&heads),const_cast<int*>(&kv_heads),
                    const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot,&scale};
                launch_named(fused_tiles,kv_fused_grid_heads(*runtime,heads,kv_heads),tile_count,256,fused_args);
                void*merge_args[]={const_cast<std::uint64_t*>(&attention_scores),
                    &attended,const_cast<int*>(&heads),
                    const_cast<int*>(&head_dim),const_cast<int*>(&tile_count)};
                launch_named("kv_attention_fused_merge",heads,1,256,merge_args);
            }else{
                void*score_args[]={&queries,&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot,&scale};
                launch_named(kv_scores_ring_kernel(*runtime),heads,(tokens+255)/256,256,score_args);
                void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot};
                launch_named(kv_values_ring_kernel(*runtime),heads,1,256,value_args);
            }
            if(profile)profile_record(profile->recurrent_end);
            std::uint64_t gated=third;int elements=heads*head_dim;
            void*gate_args[]={&attended,&gates,&gated,&elements};
            launch_named("qwen_attention_gate",(elements+255)/256,1,256,gate_args);
            dense(4,gated,residual,elements,hidden_size);
            add(residual,hidden);
            moe_base=7;
        }
        rms(residual,tensor(moe_base),normalized);
        if(std::getenv("COLIBRI_LM_DIAG")&&layer_number==0){
            float v[4]={};colibri_gpu_stream_sync(runtime->stream);
            if(colibri_gpu_download(v,normalized,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                std::fprintf(stderr,"[diag] L0 after_rms normalized[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
            if(colibri_gpu_download(v,residual,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                std::fprintf(stderr,"[diag] L0 after_rms residual[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
        }
        if(layer.dense_ffn){
            // Dense block: one SwiGLU over the layer's own gate/up/down, with no
            // router, shared expert or expert paging to run. Laguna mixes dense
            // leading blocks with MoE blocks in one model, and the two widths
            // differ, so take this block's width from its own gate projection.
            const int dense_intermediate=runtime->laguna
                ?static_cast<int>(runtime->model->tensors[layer.static_tensors.at(moe_base+1)].shape[1])
                :static_cast<int>(runtime->moe_intermediate);
            if(layer.ffn_on_host){
                auto*scratch=static_cast<float*>(runtime->dense_host);
                float*host_input=scratch;
                float*host_output=host_input+hidden_size;
                if(colibri_gpu_download(host_input,normalized,hidden_size*sizeof(float),runtime->stream)!=0||
                   colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("native dense host FFN input transfer failed");
                const auto host_started=std::chrono::steady_clock::now();
                qwen_cpu_dense_ffn(*runtime,layer,host_input,host_output);
                runtime->dense_host_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-host_started).count();
                if(colibri_gpu_upload(third,host_output,hidden_size*sizeof(float),runtime->stream)!=0)throw std::runtime_error("native dense host FFN output transfer failed");
            }else{
            const auto up_half=first+static_cast<std::uint64_t>(dense_intermediate)*sizeof(float);
            dense(moe_base+1,normalized,first,hidden_size,dense_intermediate);
            dense(moe_base+2,normalized,up_half,hidden_size,dense_intermediate);
            int dense_count=dense_intermediate;
            void*dense_silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&dense_count};
            launch_named("silu_mul",(static_cast<std::uint32_t>(dense_count)+255)/256,1,256,dense_silu_args);
            dense(moe_base+3,second,third,dense_intermediate,hidden_size);
            }
            if(profile){profile_record(profile->pre_end);profile_record(profile->shared_start);profile_record(profile->shared_end);profile_record(profile->expert_start);}
        }else{
        dense(moe_base+1,normalized,router_logits,hidden_size,experts);
        if(runtime->laguna){
            // Sigmoid routing with the score-correction bias, sum-normalized
            // over the selection and scaled by the trained routing factor.
            auto bias=runtime->device_tensors[layer.router_bias];
            int normalize=runtime->model->config.expert_weights_norm?1:0;
            float weight_scale=runtime->model->config.expert_weights_scale;
            void*route_args[]={const_cast<std::uint64_t*>(&router_logits),&bias,
                               const_cast<std::uint64_t*>(&selected_device),
                               const_cast<std::uint64_t*>(&route_weights),
                               const_cast<int*>(&experts),const_cast<int*>(&top_k),
                               &normalize,&weight_scale};
            launch_named("route_topk_sigmoid_bias",1,1,256,route_args,
                         static_cast<std::uint32_t>(2*experts*sizeof(float)));
        }else if(colibri_gpu_route_topk(router_logits,selected_device,route_weights,experts,top_k,runtime->stream)!=0)throw std::runtime_error("native Qwen routing failed");
        const auto cpu_weights_offset=device_align(top_k*sizeof(std::int32_t));
        const auto cpu_input_offset=cpu_weights_offset+device_align(top_k*sizeof(float));
        const auto cpu_activated_offset=cpu_input_offset+device_align(hidden_size*sizeof(float));
        const auto cpu_output_offset=cpu_activated_offset+device_align(top_k*runtime->moe_intermediate*sizeof(float));
        auto*cpu_weights=reinterpret_cast<float*>(staging+cpu_weights_offset);auto*cpu_input=reinterpret_cast<float*>(staging+cpu_input_offset);auto*cpu_activated=reinterpret_cast<float*>(staging+cpu_activated_offset);auto*cpu_output=reinterpret_cast<float*>(staging+cpu_output_offset);
        if(colibri_gpu_download(selected_host,selected_device,top_k*sizeof(std::int32_t),runtime->stream)!=0)throw std::runtime_error("native Qwen route transfer failed");
        if(expert_policy.routed_cpu_execution_allowed()&&
           (colibri_gpu_download(cpu_weights,route_weights,top_k*sizeof(float),runtime->stream)!=0||
            colibri_gpu_download(cpu_input,normalized,hidden_size*sizeof(float),runtime->stream)!=0))
            throw std::runtime_error("native Qwen CPU MoE input transfer failed");
        if(colibri_gpu_event_record(runtime->route_event,runtime->stream)!=0)throw std::runtime_error("native Qwen route event failed");
        if(profile){profile_record(profile->pre_end);profile_record(profile->shared_start);}
        const int intermediate=static_cast<int>(runtime->moe_intermediate);
        auto enqueue_shared=[&]{
            auto shared_gate_matrix=tensor(moe_base+2),shared_up_matrix=tensor(moe_base+3);
            // The device type, not the stored one: bf16 shared experts are
            // requantized to Q8_0 on upload and must still take the fused path.
            const auto shexp_type=qwen_device_type(*runtime,layer.static_tensors.at(moe_base+2));
            if(shexp_type==40){
                // NVFP4 shared expert. One block per output row (the kernel block-reduces),
                // and weight_scale_2 passed in f32 -- it is far too small for E4M3.
                auto shared_gate_scale=layer.shared_gate_scale,shared_up_scale=layer.shared_up_scale;
                void*silu_args[]={&shared_gate_matrix,&shared_up_matrix,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&second),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate),&shared_gate_scale,&shared_up_scale};
                launch_named("nvfp4_swiglu_transposed",intermediate,1,256,silu_args);
                auto shared_down_matrix=tensor(moe_base+4);auto shared_down_scale=layer.shared_down_scale;
                void*down_args[]={&shared_down_matrix,const_cast<std::uint64_t*>(&second),const_cast<std::uint64_t*>(&third),const_cast<int*>(&intermediate),const_cast<int*>(&hidden_size),&shared_down_scale};
                launch_named("nvfp4_matvec_transposed",hidden_size,1,256,down_args);
            }else if(shexp_type==8){
                void*silu_args[]={&shared_gate_matrix,&shared_up_matrix,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&second),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate)};
                launch_named("q8_swiglu_transposed_warp",(intermediate+7)/8,1,256,silu_args);
                dense(moe_base+4,second,third,intermediate,hidden_size);
            }else{
                // No fused SwiGLU kernel for this weight type (Laguna ships a
                // k-quant shared expert): project gate and up into one
                // contiguous pair and let silu_mul combine them.
                const auto up_half=first+static_cast<std::uint64_t>(intermediate)*sizeof(float);
                dense(moe_base+2,normalized,first,hidden_size,intermediate);
                dense(moe_base+3,normalized,up_half,hidden_size,intermediate);
                int count=intermediate;
                void*silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&count};
                launch_named("silu_mul",(static_cast<std::uint32_t>(count)+255)/256,1,256,silu_args);
                dense(moe_base+4,second,third,intermediate,hidden_size);
            }
            // Qwen gates the shared expert on a learned projection; Laguna's is
            // always on, so its block ends at the down projection.
            if(!runtime->laguna){
                auto shared_gate=tensor(moe_base+5);
                const auto sg_type=runtime->model->tensors[layer.static_tensors.at(moe_base+5)].type;
                void*shared_args[]={const_cast<std::uint64_t*>(&normalized),&shared_gate,const_cast<std::uint64_t*>(&third),const_cast<int*>(&hidden_size)};
                launch_named(sg_type==30?"qwen_shared_scale_bf16":"qwen_shared_scale",1,1,256,shared_args);
            }
        };
        bool shared_launched=false;
        if(runtime->cuda_graphs&&layer.shared_graph){
            if(colibri_gpu_graph_launch(layer.shared_graph,runtime->stream)==0){
                ++runtime->cuda_graph_replays;
                shared_launched=true;
            }else{
                colibri_gpu_graph_destroy(layer.shared_graph);
                layer.shared_graph=0;
                ++runtime->cuda_graph_fallbacks;
            }
        }
        if(runtime->cuda_graphs&&!shared_launched&&!layer.shared_graph_attempted){
            layer.shared_graph_attempted=true;
            if(colibri_gpu_graph_begin(runtime->graph_stream)==0){
                bool capture_enqueued=true;
                launch_stream=runtime->graph_stream;
                try{enqueue_shared();}catch(...){capture_enqueued=false;}
                launch_stream=runtime->stream;
                std::uint64_t captured_graph=0;
                const int capture_status=colibri_gpu_graph_end(runtime->graph_stream,&captured_graph);
                if(capture_enqueued&&capture_status==0){
                    layer.shared_graph=captured_graph;
                    ++runtime->cuda_graph_builds;
                    if(colibri_gpu_graph_launch(layer.shared_graph,runtime->stream)==0){
                        ++runtime->cuda_graph_replays;
                        shared_launched=true;
                    }else{
                        colibri_gpu_graph_destroy(layer.shared_graph);
                        layer.shared_graph=0;
                        ++runtime->cuda_graph_fallbacks;
                    }
                }else{
                    colibri_gpu_graph_destroy(captured_graph);
                    ++runtime->cuda_graph_fallbacks;
                }
            }else{
                ++runtime->cuda_graph_fallbacks;
            }
        }
        if(!shared_launched)enqueue_shared();
        if(profile)profile_record(profile->shared_end);
        const auto route_wait_started=std::chrono::steady_clock::now();
        if(colibri_gpu_event_sync(runtime->route_event)!=0)throw std::runtime_error("native Qwen route event failed");
        runtime->route_wait_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-route_wait_started).count();
        if(profile)profile_record(profile->expert_start);
        // Optional adaptive expert pruning (top-p / hard top-k). Runs only for
        // the CPU/hybrid paths, whose expert weights are the host `cpu_weights`
        // this reorders/renormalizes; the streamed-GPU path keeps its device
        // weights and is left at full top_k for now.
        int route_count=top_k;
        if(expert_policy.route_pruning_allowed()&&
           (runtime->options.expert_top_k>0||
            (runtime->options.expert_top_p>0.0f&&runtime->options.expert_top_p<1.0f)))
            route_count=apply_expert_router_policy(selected_host,cpu_weights,top_k,static_cast<int>(runtime->options.expert_top_k),runtime->options.expert_top_p);
        runtime->route_expert_sum+=static_cast<std::uint64_t>(route_count);
        // Consume the previous layer's prefetch before issuing this one. The
        // uploads pending here were predicted for *this* layer, so this is where
        // they have to land; waiting after the call below would instead have made
        // the main stream block on the layer-ahead prefetch it had just queued,
        // serializing the copy engine against the very kernels it runs ahead of.
        const auto pager_started=std::chrono::steady_clock::now();
        qwen_observe_route_recurrence(
            *runtime,
            runtime->sequences[runtime->active_sequence].expert_prefetch,
            layer_number,selected_host,route_count
        );
        qwen_wait_for_prefetch_layer(*runtime,layer_number);
        qwen_observe_and_prefetch_next_layer(
            *runtime,
            runtime->sequences[runtime->active_sequence].expert_prefetch,
            layer_number,selected_host,route_count
        );
        if(expert_policy.is_cpu()&&expert_policy.records_prefill_frequency())
            for(int rank=0;rank<route_count;++rank)
                record_expert_access(*runtime,layer_number,
                    static_cast<std::uint32_t>(selected_host[rank]));
        if(expert_policy.is_cpu()){
            if(cpu_output_offset+hidden_size*sizeof(float)>runtime->expert_staging_bytes)throw std::runtime_error("native CPU MoE workspace overflow");
            const auto compute_started=std::chrono::steady_clock::now();
            qwen_cpu_moe(*runtime,layer,selected_host,cpu_weights,route_count,cpu_input,cpu_activated,cpu_output);
            runtime->expert_compute_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-compute_started).count();
            if(colibri_gpu_upload(fourth,cpu_output,hidden_size*sizeof(float),runtime->stream)!=0)throw std::runtime_error("native CPU MoE output upload failed");
            add(third,fourth);
        }else if(expert_policy.is_hybrid()){
            if(runtime->expert_slots.empty())throw std::runtime_error("native hybrid MoE requires an expert cache budget");
            std::array<std::int32_t,256>cpu_selected{};
            std::array<float,256>cpu_compact_weights{},gpu_compact_weights{};
            std::array<float,256>gpu_gate_scales{},gpu_up_scales{},gpu_down_scales{};
            std::array<std::uint64_t,256>gate_pointers{},up_pointers{},
                down_pointers{},native_pointers{};
            int cpu_count=0,gpu_count=0;
            std::uint64_t staging_cursor=device_align(cpu_output_offset+hidden_size*sizeof(float));
            struct PendingUpload{
                std::uint64_t device,host_offset,bytes;
                std::size_t slot_index;
            };
            std::array<PendingUpload,256>pending{};int pending_count=0;
            for(int rank=0;rank<route_count;++rank){
                const int expert=selected_host[rank];if(expert<0||expert>=experts)throw std::runtime_error("native hybrid MoE selected an invalid expert");
                const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
                auto resident=runtime->expert_residency.find(cache_key);
                if(resident!=runtime->expert_residency.end()){
                    const auto slot_index=resident->second;auto&slot=runtime->expert_slots[slot_index];
                    const auto&history=record_expert_access(
                        *runtime,layer_number,expert);
                    if(expert_policy.residency_may_change())
                        slot.last_used=history.last_used;
                    record_expert_cache_hit(*runtime,slot);
                    const auto device_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){const auto bytes=runtime->model->tensors[layer.expert_tensors[role]].size/experts;const auto pointer=device_base+role_offset;if(role==0)gate_pointers[gpu_count]=pointer;else if(role==1)up_pointers[gpu_count]=pointer;else down_pointers[gpu_count]=pointer;role_offset+=bytes;}
                    if(slot.native_valid&&runtime->expert_native_cache)
                        native_pointers[gpu_count]=
                            runtime->expert_native_cache+
                            slot_index*runtime->expert_slot_bytes;
                    gpu_compact_weights[gpu_count]=cpu_weights[rank];
                    // Scales ride alongside the pointer table. SiLU makes the gate path
                    // non-linear, so gate/up must be applied before the activation; the
                    // down projection is linear, so its scale rides on `up` and comes
                    // out of the accumulate unchanged.
                    gpu_gate_scales[gpu_count]=qwen_expert_role_scale(*runtime,layer.expert_gate_scale,expert);
                    gpu_down_scales[gpu_count]=qwen_expert_role_scale(*runtime,layer.expert_down_scale,expert);
                    gpu_up_scales[gpu_count]=qwen_expert_role_scale(*runtime,layer.expert_up_scale,expert)
                        *gpu_down_scales[gpu_count];
                    ++gpu_count;continue;
                }
                ++runtime->expert_cache_misses;cpu_selected[cpu_count]=expert;cpu_compact_weights[cpu_count]=cpu_weights[rank];
                if(layer.expert_down_scale!=std::numeric_limits<std::uint64_t>::max()){
                    const auto&st=runtime->model->tensors[layer.expert_down_scale];
                    const auto experts_count=runtime->model->config.expert_count;
                    const auto scale_bytes=st.size/experts_count;
                    float ds=1.0f;std::memcpy(&ds,tensor_data(*runtime->model,st)+static_cast<std::uint64_t>(expert)*scale_bytes,sizeof(float));
                    cpu_compact_weights[cpu_count]*=ds;
                }
                ++cpu_count;
                const auto slot_index=select_expert_cache_slot(*runtime,layer_number,expert,true);
                if(slot_index==kNoExpertSlot)continue;
                auto&slot=runtime->expert_slots[slot_index];slot.key=cache_key;slot.valid=true;slot.native_valid=false;slot.last_used=++runtime->expert_clock;runtime->expert_residency[cache_key]=slot_index;
                const auto slot_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;
                if(runtime->dma_paging){
                    // DMA each role straight from the registered mmap into the cache slot;
                    // no CPU staging memcpy (the 4.4 ms/token page-in cost we are attacking).
                    std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;if(colibri_gpu_upload(slot_base+role_offset,tensor_data(*runtime->model,t)+offset,bytes,runtime->stream)!=0)throw std::runtime_error("native hybrid MoE DMA cache upload failed");role_offset+=bytes;}
                    if(runtime->expert_native_cache){
                        const auto gate_bytes=runtime->model->tensors[
                            layer.expert_tensors[0]].size/experts;
                        const auto up_bytes=runtime->model->tensors[
                            layer.expert_tensors[1]].size/experts;
                        const int status=colibri_gpu_nvfp4_prepare_expert(
                            slot_base,slot_base+gate_bytes,
                            slot_base+gate_bytes+up_bytes,
                            runtime->expert_native_cache+
                                slot_index*runtime->expert_slot_bytes,
                            runtime->stream,hidden_size,intermediate);
                        if(status!=0)throw std::runtime_error(
                            "persistent NVFP4 expert preparation failed");
                        slot.native_valid=true;
                    }
                }else{
                    const auto bundle_start=staging_cursor;
                    for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;if(staging_cursor+bytes>runtime->expert_staging_bytes)throw std::runtime_error("native hybrid MoE staging overflow");std::memcpy(staging+staging_cursor,tensor_data(*runtime->model,t)+offset,bytes);staging_cursor+=bytes;}
                    pending[pending_count++]={slot_base,bundle_start,
                        staging_cursor-bundle_start,slot_index};
                }
            }
            if(gpu_count){
                const auto table_bytes=static_cast<std::uint64_t>(gpu_count)*(3*sizeof(std::uint64_t)+4*sizeof(float));
                const auto table_host=device_align(staging_cursor);const auto table_device=runtime->expert_staging+runtime->expert_staging_bytes-device_align(table_bytes);
                if(table_host+table_bytes>runtime->expert_staging_bytes)throw std::runtime_error("native hybrid MoE pointer staging overflow");
                std::memcpy(staging+table_host,gate_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+gpu_count*sizeof(std::uint64_t),up_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+2*gpu_count*sizeof(std::uint64_t),down_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+3*gpu_count*sizeof(std::uint64_t),gpu_compact_weights.data(),gpu_count*sizeof(float));
                std::memcpy(staging+table_host+3*gpu_count*sizeof(std::uint64_t)+gpu_count*sizeof(float),gpu_gate_scales.data(),gpu_count*sizeof(float));
                std::memcpy(staging+table_host+3*gpu_count*sizeof(std::uint64_t)+2*gpu_count*sizeof(float),gpu_up_scales.data(),gpu_count*sizeof(float));
                std::memcpy(staging+table_host+3*gpu_count*sizeof(std::uint64_t)+3*gpu_count*sizeof(float),gpu_down_scales.data(),gpu_count*sizeof(float));
                if(colibri_gpu_upload(table_device,staging+table_host,table_bytes,runtime->stream)!=0)throw std::runtime_error("native hybrid MoE table upload failed");
                const auto gate_table=table_device,up_table=gate_table+gpu_count*sizeof(std::uint64_t),down_table=up_table+gpu_count*sizeof(std::uint64_t),weight_table=down_table+gpu_count*sizeof(std::uint64_t);
                const auto gate_scale_table=weight_table+gpu_count*sizeof(float),up_scale_table=gate_scale_table+gpu_count*sizeof(float),down_scale_table=up_scale_table+gpu_count*sizeof(float);
                const auto gate_type=runtime->model->tensors[layer.expert_tensors[0]].type;
                const auto down_type=runtime->model->tensors[layer.expert_tensors[2]].type;
        // The grouped dispatch below ends in a k-quant fallback, so an
        // unhandled type would be decoded as Q5_K rather than rejected. Prepare
        // already routes such models to the CPU; this makes the silent path
        // unreachable if that ever stops holding.
        if(!qwen_gpu_expert_type_supported(gate_type)||
           !qwen_gpu_expert_type_supported(down_type))
            throw std::runtime_error(
                "native GPU expert quantization is unsupported: "+
                std::to_string(gate_type)+"/"+std::to_string(down_type));
                const char*persistent_env=
                    std::getenv("COLIBRI_NVFP4_PERSISTENT");
                const bool persistent_enabled=persistent_env&&
                    persistent_env[0]=='1'&&runtime->expert_native_cache&&
                    std::all_of(native_pointers.begin(),
                        native_pointers.begin()+gpu_count,
                        [](std::uint64_t pointer){return pointer!=0;});
                const char*tc_env=std::getenv("COLIBRI_NVFP4_DECODE_TENSOR_CORES");
                const bool tc_enabled=tc_env&&tc_env[0]=='1';
                bool tc_done=false;
                if(persistent_enabled&&gate_type==40&&down_type==40){
                    const int tc_status=colibri_gpu_nvfp4_moe_persistent(
                        native_pointers.data(),weight_table,
                        gate_scale_table,up_scale_table,down_scale_table,
                        normalized,activated,third,runtime->stream,
                        hidden_size,intermediate,gpu_count);
                    if(tc_status==0){++runtime->nvfp4_tensor_core_moe_calls;runtime->nvfp4_tensor_core_moe_last_status=0;tc_done=true;}
                    else{
                        ++runtime->nvfp4_tensor_core_moe_fallbacks;
                        runtime->nvfp4_tensor_core_moe_last_status=tc_status;
                        if(std::getenv("COLIBRI_NVFP4_TENSOR_CORE_TRACE"))
                            std::fprintf(stderr,"[nvfp4-persistent] hybrid fallback status=%d experts=%d\n",tc_status,gpu_count);
                    }
                }
                if(!tc_done&&tc_enabled&&gate_type==40&&down_type==40){
                    const int tc_status=colibri_gpu_nvfp4_moe_cublas(
                        gate_table,up_table,down_table,normalized,activated,third,
                        weight_table,gate_scale_table,up_scale_table,down_scale_table,runtime->stream,
                        hidden_size,intermediate,gpu_count);
                    if(tc_status==0){++runtime->nvfp4_tensor_core_moe_calls;runtime->nvfp4_tensor_core_moe_last_status=0;tc_done=true;}
                    else{
                        ++runtime->nvfp4_tensor_core_moe_fallbacks;
                        runtime->nvfp4_tensor_core_moe_last_status=tc_status;
                        if(std::getenv("COLIBRI_NVFP4_TENSOR_CORE_TRACE"))
                            std::fprintf(stderr,"[nvfp4-tc] hybrid fallback status=%d experts=%d\n",tc_status,gpu_count);
                    }
                }
                if(!tc_done){
                    const char* tiled_env=std::getenv("COLIBRI_NVFP4_TILED");
                    const bool nvfp4_tiled=tiled_env&&tiled_env[0]=='1';
                    void*gate_up_args[]={const_cast<std::uint64_t*>(&gate_table),const_cast<std::uint64_t*>(&up_table),const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&activated),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate),&gpu_count,const_cast<std::uint64_t*>(&gate_scale_table),const_cast<std::uint64_t*>(&up_scale_table)};
                    launch_named(qwen_grouped_swiglu_name(gate_type,nvfp4_tiled,false).c_str(),gate_type==40&&nvfp4_tiled?(intermediate+7)/8:intermediate,gpu_count,256,gate_up_args);
                    const int status=qwen_launch_grouped_accumulate(runtime->stream,down_type,down_table,activated,third,weight_table,intermediate,hidden_size,gpu_count);
                    if(status!=0)throw std::runtime_error("native hybrid MoE down projection failed");
                }
            }
            for(int index=0;index<pending_count;++index){
                const auto&upload=pending[index];
                if(colibri_gpu_upload(upload.device,
                        staging+upload.host_offset,upload.bytes,
                        runtime->stream)!=0)
                    throw std::runtime_error(
                        "native hybrid MoE cache upload failed");
                if(runtime->expert_native_cache){
                    const auto gate_bytes=runtime->model->tensors[
                        layer.expert_tensors[0]].size/experts;
                    const auto up_bytes=runtime->model->tensors[
                        layer.expert_tensors[1]].size/experts;
                    const int status=colibri_gpu_nvfp4_prepare_expert(
                        upload.device,upload.device+gate_bytes,
                        upload.device+gate_bytes+up_bytes,
                        runtime->expert_native_cache+
                            upload.slot_index*runtime->expert_slot_bytes,
                        runtime->stream,hidden_size,intermediate);
                    if(status!=0)throw std::runtime_error(
                        "persistent NVFP4 expert preparation failed");
                    runtime->expert_slots[
                        upload.slot_index].native_valid=true;
                }
            }
            if(cpu_count){
                const auto compute_started=std::chrono::steady_clock::now();
                qwen_cpu_moe(*runtime,layer,cpu_selected.data(),cpu_compact_weights.data(),cpu_count,cpu_input,cpu_activated,cpu_output);
                runtime->expert_compute_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-compute_started).count();
                if(colibri_gpu_upload(fourth,cpu_output,hidden_size*sizeof(float),runtime->stream)!=0)throw std::runtime_error("native hybrid MoE output upload failed");
                add(third,fourth);
            }
        }else{
        std::uint64_t staging_cursor=device_align(top_k*sizeof(std::int32_t));
        bool has_uncached_expert=false;
        std::array<std::uint64_t,256> gate_pointers{},up_pointers{},down_pointers{};
        std::array<float,256> route_gate_scales{},route_up_scales{},route_down_scales{};
        for(int rank=0;rank<top_k;++rank){
            const int expert=selected_host[rank];if(expert<0||expert>=experts)throw std::runtime_error("native Qwen router selected an invalid expert");
            // Same convention as the cached path: gate/up before SiLU, down folded
            // into `up` because the accumulate that follows is linear.
            route_gate_scales[rank]=qwen_expert_role_scale(*runtime,layer.expert_gate_scale,expert);
            route_down_scales[rank]=qwen_expert_role_scale(*runtime,layer.expert_down_scale,expert);
            route_up_scales[rank]=qwen_expert_role_scale(*runtime,layer.expert_up_scale,expert)
                *route_down_scales[rank];
            std::uint64_t device_base=0;
            const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
            if(!runtime->expert_slots.empty()){
                auto resident=runtime->expert_residency.find(cache_key);
                if(resident!=runtime->expert_residency.end()){
                    const auto slot_index=resident->second;
                    auto&slot=runtime->expert_slots[slot_index];
                    record_expert_cache_hit(*runtime,slot);
                    const auto&history=record_expert_access(
                        *runtime,layer_number,expert);
                    if(expert_policy.residency_may_change())
                        slot.last_used=history.last_used;
                    device_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;
                    std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){const auto bytes=runtime->model->tensors[layer.expert_tensors[role]].size/experts;const auto pointer=device_base+role_offset;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;role_offset+=bytes;}
                }else{
                    ++runtime->expert_cache_misses;
                    const auto slot_index=select_expert_cache_slot(*runtime,layer_number,expert,true);
                    if(slot_index!=kNoExpertSlot){
                        auto&slot=runtime->expert_slots[slot_index];slot.key=cache_key;slot.valid=true;slot.last_used=++runtime->expert_clock;runtime->expert_residency[cache_key]=slot_index;
                        device_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;
                        if(runtime->dma_paging){
                            // The mmap is CUDA-registered, so each role goes straight from
                            // the file mapping into the cache slot. That removes the host
                            // memcpy from the critical path -- the CPU has already synced on
                            // route_event by here, so every byte it copies is GPU idle time.
                            std::uint64_t role_offset=0;
                            for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;if(colibri_gpu_upload(device_base+role_offset,tensor_data(*runtime->model,t)+offset,bytes,runtime->stream)!=0)throw std::runtime_error("native Qwen cached expert DMA upload failed");role_offset+=bytes;}
                        }else{
                            std::uint64_t bundle_bytes=0;
                            for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;std::memcpy(staging+staging_cursor+bundle_bytes,tensor_data(*runtime->model,t)+offset,bytes);bundle_bytes+=bytes;}
                            if(staging_cursor+bundle_bytes>runtime->expert_staging_bytes||colibri_gpu_upload(device_base,staging+staging_cursor,bundle_bytes,runtime->stream)!=0)throw std::runtime_error("native Qwen cached expert upload failed");
                            staging_cursor+=bundle_bytes;
                        }
                        std::uint64_t role_offset=0;
                        for(int role=0;role<3;++role){const auto bytes=runtime->model->tensors[layer.expert_tensors[role]].size/experts;const auto pointer=device_base+role_offset;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;role_offset+=bytes;}
                    }else{
                        has_uncached_expert=true;
                        for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;std::memcpy(staging+staging_cursor,tensor_data(*runtime->model,t)+offset,bytes);const auto pointer=runtime->expert_staging+staging_cursor;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;staging_cursor+=bytes;}
                    }
                }
            }else{
                ++runtime->expert_cache_misses;
                has_uncached_expert=true;
                for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;std::memcpy(staging+staging_cursor,tensor_data(*runtime->model,t)+offset,bytes);const auto pointer=runtime->expert_staging+staging_cursor;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;staging_cursor+=bytes;}
            }
        }
        const auto table_bytes=static_cast<std::uint64_t>(top_k)*(sizeof(std::uint64_t)*3+sizeof(float)*3);const auto table_host=device_align(staging_cursor);const auto table_device=runtime->expert_staging+runtime->expert_staging_bytes-device_align(table_bytes);const auto gate_table=table_device;const auto up_table=gate_table+top_k*sizeof(std::uint64_t);const auto down_table=up_table+top_k*sizeof(std::uint64_t);std::memcpy(staging+table_host,gate_pointers.data(),top_k*sizeof(std::uint64_t));std::memcpy(staging+table_host+top_k*sizeof(std::uint64_t),up_pointers.data(),top_k*sizeof(std::uint64_t));std::memcpy(staging+table_host+2*top_k*sizeof(std::uint64_t),down_pointers.data(),top_k*sizeof(std::uint64_t));
        const auto gate_scale_table=down_table+top_k*sizeof(std::uint64_t),up_scale_table=gate_scale_table+top_k*sizeof(float),down_scale_table=up_scale_table+top_k*sizeof(float);
        std::memcpy(staging+table_host+3*top_k*sizeof(std::uint64_t),route_gate_scales.data(),top_k*sizeof(float));
        std::memcpy(staging+table_host+3*top_k*sizeof(std::uint64_t)+top_k*sizeof(float),route_up_scales.data(),top_k*sizeof(float));
        std::memcpy(staging+table_host+3*top_k*sizeof(std::uint64_t)+2*top_k*sizeof(float),route_down_scales.data(),top_k*sizeof(float));
        if(table_host+table_bytes>runtime->expert_staging_bytes)throw std::runtime_error("native Qwen expert staging overflow");
        if(has_uncached_expert&&staging_cursor&&colibri_gpu_upload(runtime->expert_staging,staging,staging_cursor,runtime->stream)!=0)throw std::runtime_error("native Qwen expert upload failed");
        if(colibri_gpu_upload(table_device,staging+table_host,table_bytes,runtime->stream)!=0)throw std::runtime_error("native Qwen expert pointer upload failed");
        const auto gate_type=runtime->model->tensors[layer.expert_tensors[0]].type;
        const auto down_type=runtime->model->tensors[layer.expert_tensors[2]].type;
        // The grouped dispatch below ends in a k-quant fallback, so an
        // unhandled type would be decoded as Q5_K rather than rejected. Prepare
        // already routes such models to the CPU; this makes the silent path
        // unreachable if that ever stops holding.
        if(!qwen_gpu_expert_type_supported(gate_type)||
           !qwen_gpu_expert_type_supported(down_type))
            throw std::runtime_error(
                "native GPU expert quantization is unsupported: "+
                std::to_string(gate_type)+"/"+std::to_string(down_type));
        const char*tc_env=std::getenv("COLIBRI_NVFP4_DECODE_TENSOR_CORES");
        const bool tc_enabled=tc_env&&tc_env[0]=='1';
        bool tc_done=false;
        if(tc_enabled&&gate_type==40&&down_type==40){
            const int tc_status=colibri_gpu_nvfp4_moe_cublas(
                gate_table,up_table,down_table,normalized,activated,third,
                route_weights,gate_scale_table,up_scale_table,down_scale_table,runtime->stream,
                hidden_size,intermediate,top_k);
            if(tc_status==0){++runtime->nvfp4_tensor_core_moe_calls;runtime->nvfp4_tensor_core_moe_last_status=0;tc_done=true;}
            else{
                ++runtime->nvfp4_tensor_core_moe_fallbacks;
                runtime->nvfp4_tensor_core_moe_last_status=tc_status;
                if(std::getenv("COLIBRI_NVFP4_TENSOR_CORE_TRACE"))
                    std::fprintf(stderr,"[nvfp4-tc] decode fallback status=%d experts=%d\n",tc_status,top_k);
            }
        }
        if(!tc_done){
            const char* tiled_env=std::getenv("COLIBRI_NVFP4_TILED");
            const bool nvfp4_tiled=tiled_env&&tiled_env[0]=='1';
            void*gate_up_args[]={const_cast<std::uint64_t*>(&gate_table),const_cast<std::uint64_t*>(&up_table),const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&activated),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate),const_cast<int*>(&top_k),const_cast<std::uint64_t*>(&gate_scale_table),const_cast<std::uint64_t*>(&up_scale_table)};
            launch_named(qwen_grouped_swiglu_name(gate_type,nvfp4_tiled,false).c_str(),gate_type==40&&nvfp4_tiled?(intermediate+7)/8:intermediate,top_k,256,gate_up_args);
            const int down_status=qwen_launch_grouped_accumulate(runtime->stream,down_type,down_table,activated,third,route_weights,intermediate,hidden_size,top_k);
            if(down_status!=0)throw std::runtime_error("native Qwen expert down projection failed");
        }
        }
        if(std::getenv("COLIBRI_LM_DIAG")&&layer_number==0){
            float v[4]={};colibri_gpu_stream_sync(runtime->stream);
            if(colibri_gpu_download(v,third,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                std::fprintf(stderr,"[diag] L0 before_final_add third[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
            if(colibri_gpu_download(v,residual,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                std::fprintf(stderr,"[diag] L0 before_final_add residual[0..3]=% .6e % .6e % .6e % .6e\n",v[0],v[1],v[2],v[3]);
        }
        runtime->expert_page_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-pager_started).count();
        }
        add(residual,third);
        if(profile)profile_record(profile->expert_end);
        std::swap(hidden,residual);
        if(std::getenv("COLIBRI_LM_DIAG")&&(layer_number==0||layer_number==3||layer_number==33||layer_number==39)){
            static int lc=0;
            if(lc<8){++lc;float v[4]={};if(colibri_gpu_download(v,hidden,sizeof(v),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0)
                std::fprintf(stderr,"[diag] after_layer_%u hidden[0..3]=% .6e % .6e % .6e % .6e\n",layer_number,v[0],v[1],v[2],v[3]);}
        }
    }
    if(runtime->cuda_profile)profile_record(runtime->cuda_tail_start);
    if(runtime->options.mtp_drafts){
        auto target_hidden=runtime->state+runtime->mtp_target_hidden_offset;
        void*copy_args[]={const_cast<std::uint64_t*>(&hidden),&target_hidden,const_cast<int*>(&hidden_size)};
        launch_named("qwen_copy_vector",(hidden_size+255)/256,1,256,copy_args);
        runtime->mtp_has_target_hidden=true;
    }
    rms(hidden,runtime->device_tensors[runtime->final_norm],normalized);
    if(runtime->cuda_profile)profile_record(runtime->cuda_lm_start);
    runtime->last_sampling_normalized=normalized;
    runtime->last_sampling_logits=logits;
    if(std::getenv("COLIBRI_LM_DIAG")){
        static int diag_count=0;
        if(diag_count<2){
            ++diag_count;
            float nvec[8]={};
            if(colibri_gpu_download(nvec,normalized,sizeof(nvec),runtime->stream)==0&&colibri_gpu_stream_sync(runtime->stream)==0){
                std::fprintf(stderr,"[lm-diag] normalized[0..7]=% .6e % .6e % .6e % .6e % .6e % .6e % .6e % .6e\n",nvec[0],nvec[1],nvec[2],nvec[3],nvec[4],nvec[5],nvec[6],nvec[7]);
            }
            std::fprintf(stderr,"[lm-diag] lm_head_type=%u lm_head_tensor=%zu hidden_size=%d\n",runtime->lm_head_type,runtime->lm_head,hidden_size);
        }
    }
    int vocabulary=static_cast<int>(runtime->model->config.vocabulary_size);
    if(colibri_gpu_memset(argmax_device,0,sizeof(std::uint64_t),runtime->stream)!=0)throw std::runtime_error("native Qwen argmax reset failed");
    auto lm_head=runtime->device_tensors[runtime->lm_head];
    // The LM head is the largest single tensor read per token. Where the type
    // has a group-decode kernel, quantize the activation once and use it; the
    // fused per-element head remains the fallback.
    const char*lm_q8_kernel=qwen_q8_lm_head_kernel(runtime->lm_head_type);
    if(lm_q8_kernel&&iq2_q8_enabled&&(hidden_size&255)==0){
        if(normalized!=q8_cached_input){
            void*quant_args[]={const_cast<std::uint64_t*>(&normalized),
                const_cast<std::uint64_t*>(&dense_q8),
                const_cast<std::uint64_t*>(&dense_q8_scales),
                const_cast<int*>(&hidden_size)};
            launch_named("quantize_q8_blocks",(hidden_size+31)/32,1,32,quant_args);
            q8_cached_input=normalized;
        }
        void*lm_args[]={&lm_head,const_cast<std::uint64_t*>(&dense_q8),
            const_cast<std::uint64_t*>(&dense_q8_scales),
            const_cast<std::uint64_t*>(&argmax_device),
            const_cast<int*>(&hidden_size),&vocabulary};
        launch_named(lm_q8_kernel,(vocabulary+7)/8,1,256,lm_args);
    }else{
        void*argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&argmax_device),const_cast<int*>(&hidden_size),&vocabulary};
        launch_named(qwen_lm_head_argmax_kernel(runtime->lm_head_type),(vocabulary+7)/8,1,256,argmax_args);
    }
    if(runtime->cuda_profile)profile_record(runtime->cuda_lm_end);
    auto*packed_winner=reinterpret_cast<std::uint64_t*>(staging);
    if(colibri_gpu_download(packed_winner,argmax_device,sizeof(*packed_winner),runtime->stream)!=0)throw std::runtime_error("native Qwen output transfer failed");
    if(runtime->cuda_profile)profile_record(runtime->cuda_tail_end);
    const auto tail_wait_started=std::chrono::steady_clock::now();
    if(colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("native Qwen output synchronization failed");
    runtime->tail_wait_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tail_wait_started).count();
    kernel_profile_collect();
    if(kernel_profile)++kernel_profile_decodes;
    if(runtime->cuda_profile){
        float delta_ms=0.0f,attention_ms=0.0f,shared_ms=0.0f,expert_ms=0.0f,tail_ms=0.0f,lm_ms=0.0f,recurrent_ms=0.0f,attention_core_ms=0.0f;
        std::uint32_t delta_layers=0,attention_layers=0;
        auto elapsed=[](std::uint64_t start,std::uint64_t end){float value=0.0f;if(colibri_gpu_event_elapsed(start,end,&value)!=0)throw std::runtime_error("native Qwen CUDA profiling measurement failed");return value;};
        for(std::size_t index=0;index<runtime->layers.size();++index){
            const auto&events=runtime->cuda_layer_profiles[index];
            const float pre=elapsed(events.pre_start,events.pre_end);
            if(runtime->layers[index].attention){attention_ms+=pre;attention_core_ms+=elapsed(events.recurrent_start,events.recurrent_end);++attention_layers;}else{delta_ms+=pre;++delta_layers;recurrent_ms+=elapsed(events.recurrent_start,events.recurrent_end);}
            shared_ms+=elapsed(events.shared_start,events.shared_end);
            expert_ms+=elapsed(events.expert_start,events.expert_end);
        }
        tail_ms=elapsed(runtime->cuda_tail_start,runtime->cuda_tail_end);
        lm_ms=elapsed(runtime->cuda_lm_start,runtime->cuda_lm_end);
        std::fprintf(stderr,"[cuda-profile] position=%llu delta=%.3fms/%u (recurrent=%.3fms) attention=%.3fms/%u (core=%.3fms) shared=%.3fms expert=%.3fms tail=%.3fms (lm=%.3fms) total=%.3fms\n",
            static_cast<unsigned long long>(runtime->position),delta_ms,delta_layers,recurrent_ms,attention_ms,attention_layers,attention_core_ms,
            shared_ms,expert_ms,tail_ms,lm_ms,delta_ms+attention_ms+shared_ms+expert_ms+tail_ms);
    }
    *output_token=0xffffffffu-static_cast<std::uint32_t>(*packed_winner);
    if(std::getenv("COLIBRI_TOK_DEBUG")){
        static int tc=0; if(tc++<60)
            std::fprintf(stderr,"[tok-dbg] step=%d in=%u -> out=%u\n",tc,input_token,*output_token);
    }
    runtime->last_output_token=*output_token;
    runtime->processed_tokens.push_back(input_token);
    ++runtime->position;
    if(runtime->cuda_graphs&&!runtime->cuda_graph_trace_reported&&
       std::getenv("COLIBRI_CUDA_GRAPH_TRACE"))
    {
        std::fprintf(stderr,
            "[cuda-graphs] builds=%llu replays=%llu fallbacks=%llu\n",
            static_cast<unsigned long long>(runtime->cuda_graph_builds),
            static_cast<unsigned long long>(runtime->cuda_graph_replays),
            static_cast<unsigned long long>(runtime->cuda_graph_fallbacks));
        runtime->cuda_graph_trace_reported=true;
    }
    ++runtime->decode_calls;
    runtime->decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-decode_started).count();
    if(runtime->decode_calls%50==0)kernel_profile_report();
    return 0;
});}

static double qwen_sampling_uniform(QwenSamplingState& sampling) {
    std::uint64_t value=sampling.rng;
    if(!value)value=0x9e3779b97f4a7c15ULL;
    value^=value>>12;value^=value<<25;value^=value>>27;
    sampling.rng=value;
    const std::uint64_t random=value*2685821657736338717ULL;
    return static_cast<double>(random>>11)*(1.0/9007199254740992.0);
}

static std::uint32_t qwen_sample_last_logits(
        ColibriV2QwenRuntime& runtime,QwenSamplingState& sampling,
        std::uint32_t greedy_token) {
    if(!sampling.enabled())return greedy_token;
    struct SamplingTimer {
        std::uint64_t& nanoseconds;
        std::chrono::steady_clock::time_point started;
        ~SamplingTimer(){nanoseconds+=std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();}
    } sampling_timer{runtime.sampling_nanoseconds,
                     std::chrono::steady_clock::now()};
    if(runtime.gemma4)throw std::runtime_error(
        "native Gemma 4 sampling is not implemented yet");
    if(!runtime.last_sampling_normalized||!runtime.last_sampling_logits)
        throw std::runtime_error("native Qwen sampling state is unavailable");
    const auto vocabulary=runtime.model->config.vocabulary_size;
    const auto bytes=static_cast<std::uint64_t>(vocabulary)*sizeof(float);
    const std::size_t count=sampling.top_k?
        std::min<std::size_t>(sampling.top_k,vocabulary):vocabulary;
    // Sampling from a single candidate is identical to greedy decode. Avoid
    // both the second LM-head projection and all candidate transfers.
    if(count==1){runtime.last_output_token=greedy_token;return greedy_token;}
    const auto lm_head=runtime.device_tensors[runtime.lm_head];
    if(runtime.lm_head_type==12?
            colibri_gpu_q4k_matvec_transposed(
                lm_head,runtime.last_sampling_normalized,runtime.last_sampling_logits,
                static_cast<std::int32_t>(runtime.model->config.hidden_size),
                static_cast<std::int32_t>(vocabulary),runtime.stream):
        runtime.lm_head_type==14?
            colibri_gpu_q6k_matvec_transposed(
                lm_head,runtime.last_sampling_normalized,runtime.last_sampling_logits,
                static_cast<std::int32_t>(runtime.model->config.hidden_size),
                static_cast<std::int32_t>(vocabulary),runtime.stream):
        runtime.lm_head_type==30?
            colibri_gpu_bf16_matvec_transposed(
                lm_head,runtime.last_sampling_normalized,runtime.last_sampling_logits,
                static_cast<std::int32_t>(runtime.model->config.hidden_size),
                static_cast<std::int32_t>(vocabulary),runtime.stream):
            colibri_gpu_q8_matvec_transposed(
                lm_head,runtime.last_sampling_normalized,runtime.last_sampling_logits,
                static_cast<std::int32_t>(runtime.model->config.hidden_size),
                static_cast<std::int32_t>(vocabulary),runtime.stream))
        throw std::runtime_error("native Qwen sampling LM-head projection failed");
    std::vector<std::uint32_t> candidates;
    std::vector<float> candidate_logits;
    const char*gpu_topk_setting=std::getenv("COLIBRI_SAMPLING_GPU_TOPK");
    const bool gpu_topk_enabled=
        !gpu_topk_setting||gpu_topk_setting[0]!='0';
    if(gpu_topk_enabled&&sampling.top_k&&
       count<=colibri::v2::workspace::kSamplingTopKCapacity&&
       vocabulary<=colibri::v2::workspace::kSamplingSortItemsPerBlock*
                   colibri::v2::workspace::kSamplingSortBlockCapacity){
        const auto selected_device=runtime.decode_workspace_layout.
            sampling_selected.address(runtime.workspace);
        const auto values_device=runtime.decode_workspace_layout.
            sampling_logits.address(runtime.workspace);
        const auto sort_indices_a=runtime.decode_workspace_layout.
            sampling_sort_indices_a.address(runtime.workspace);
        const auto sort_values_a=runtime.decode_workspace_layout.
            sampling_sort_values_a.address(runtime.workspace);
        const auto sort_indices_b=runtime.decode_workspace_layout.
            sampling_sort_indices_b.address(runtime.workspace);
        const auto sort_values_b=runtime.decode_workspace_layout.
            sampling_sort_values_b.address(runtime.workspace);
        if(colibri_gpu_sampling_topk(
                runtime.last_sampling_logits,selected_device,values_device,
                sort_indices_a,sort_values_a,sort_indices_b,sort_values_b,
                static_cast<std::int32_t>(vocabulary),
                static_cast<std::int32_t>(count),runtime.stream)!=0)
            throw std::runtime_error("native Qwen sampling top-k reduction failed");
        const auto candidate_bytes=count*sizeof(std::uint32_t);
        const auto value_offset=device_align(candidate_bytes);
        const auto value_bytes=count*sizeof(float);
        if(value_offset+value_bytes>runtime.host_staging_bytes)
            throw std::runtime_error("native Qwen sampling candidate workspace is too small");
        auto*selected_host=static_cast<std::uint32_t*>(runtime.host_staging);
        auto*values_host=reinterpret_cast<float*>(
            static_cast<std::uint8_t*>(runtime.host_staging)+value_offset);
        if(colibri_gpu_download(
                selected_host,selected_device,candidate_bytes,runtime.stream)!=0||
           colibri_gpu_download(
                values_host,values_device,value_bytes,runtime.stream)!=0||
           colibri_gpu_stream_sync(runtime.stream)!=0)
            throw std::runtime_error("native Qwen sampling candidate transfer failed");
        candidates.assign(selected_host,selected_host+count);
        candidate_logits.assign(values_host,values_host+count);
        if(candidates.empty()||candidates.front()>=vocabulary)
            return greedy_token;
        ++runtime.sampling_gpu_topk_calls;
        runtime.sampling_gpu_topk_bytes+=candidate_bytes+value_bytes;
    }else{
        if(bytes>runtime.host_staging_bytes)
            throw std::runtime_error("native Qwen sampling workspace is too small");
        auto*logits=static_cast<float*>(runtime.host_staging);
        if(colibri_gpu_download(
                logits,runtime.last_sampling_logits,bytes,runtime.stream)!=0||
           colibri_gpu_stream_sync(runtime.stream)!=0)
            throw std::runtime_error("native Qwen sampling logits transfer failed");
        runtime.sampling_full_download_bytes+=bytes;
        candidates.resize(vocabulary);
        for(std::uint32_t token=0;token<vocabulary;++token)candidates[token]=token;
        auto greater=[&](std::uint32_t left,std::uint32_t right){
            const float a=logits[left],b=logits[right];
            if(std::isnan(a))return false;
            if(std::isnan(b))return true;
            return a!=b?a>b:left<right;
        };
        if(count<candidates.size()){
            std::partial_sort(candidates.begin(),candidates.begin()+count,
                              candidates.end(),greater);
            candidates.resize(count);
        }else std::sort(candidates.begin(),candidates.end(),greater);
        candidate_logits.reserve(candidates.size());
        for(const auto token:candidates)candidate_logits.push_back(logits[token]);
    }
    const double maximum=static_cast<double>(candidate_logits.front())/
        sampling.temperature;
    std::vector<double> probabilities(candidates.size());
    double total=0.0;
    for(std::size_t index=0;index<candidates.size();++index){
        const double scaled=static_cast<double>(candidate_logits[index])/
            sampling.temperature;
        const double probability=std::isfinite(scaled)?
            std::exp(scaled-maximum):0.0;
        probabilities[index]=probability;total+=probability;
    }
    if(!(total>0.0)||!std::isfinite(total))return greedy_token;
    std::size_t keep=probabilities.size();
    if(sampling.top_p<1.0f){
        double cumulative=0.0;
        for(std::size_t index=0;index<probabilities.size();++index){
            cumulative+=probabilities[index]/total;
            if(cumulative>=sampling.top_p){keep=index+1;break;}
        }
    }
    double kept_total=0.0;
    for(std::size_t index=0;index<keep;++index)kept_total+=probabilities[index];
    const double threshold=qwen_sampling_uniform(sampling)*kept_total;
    double cumulative=0.0;
    for(std::size_t index=0;index<keep;++index){
        cumulative+=probabilities[index];
        if(threshold<=cumulative){
            runtime.last_output_token=candidates[index];
            return candidates[index];
        }
    }
    runtime.last_output_token=candidates[keep-1];
    return candidates[keep-1];
}

// Leading tokens shared by a slot's committed tokens and the new prompt.
static std::uint64_t qwen_sequence_match(
        const std::vector<std::uint32_t>& tokens,
        const std::uint32_t* prompt, std::uint64_t prompt_count) {
    const std::uint64_t limit = std::min<std::uint64_t>(tokens.size(), prompt_count);
    std::uint64_t i = 0;
    while (i < limit && tokens[i] == prompt[i]) ++i;
    return i;
}

// A large absolute shared prefix is worth restoring even when it is less than
// half of a very long previous conversation. The fractional rule alone caused
// 15-30k reusable agent prefixes to be discarded after a long side branch.
static bool qwen_cache_match_useful(
        std::uint64_t match, std::size_t cached_tokens) {
    return match != 0 &&
        (match >= 2048 || match >= (cached_tokens + 1) / 2);
}

// Save the live (active) working set into its slot and load `target`'s. The KV
// arena is per-slot, so only the pointer + host bookkeeping move -- no device
// copy. Snapshots stay global; the reuse KV-safety guard keeps them correct
// across slots by tying reuse to the (per-slot) processed_tokens.
static void qwen_switch_sequence(ColibriV2QwenRuntime& runtime, std::size_t target) {
    if (target == runtime.active_sequence) return;
    QwenSequence& cur = runtime.sequences[runtime.active_sequence];
    cur.position = runtime.position;
    cur.last_output_token = runtime.last_output_token;
    cur.processed_tokens.swap(runtime.processed_tokens);
    cur.prefill_snapshots.swap(runtime.prefill_snapshots);
    cur.prefill_snapshot_clock = runtime.prefill_snapshot_clock;
    QwenSequence& next = runtime.sequences[target];
    runtime.state = next.state;
    runtime.position = next.position;
    runtime.last_output_token = next.last_output_token;
    runtime.processed_tokens.swap(next.processed_tokens);
    runtime.prefill_snapshots.swap(next.prefill_snapshots);
    runtime.prefill_snapshot_clock = next.prefill_snapshot_clock;
    runtime.active_sequence = target;
}

// KV region bytes for `elems` elements at a cache precision. Shares the one
// definition with the prepare-time sizing so a new precision cannot size the
// arena one way and the live-state ranges another.
static std::uint64_t qwen_kv_region_bytes(std::uint64_t elems, int type) {
    return kv_type_bytes(elems, type);
}

// The device ranges of a slot arena that hold live conversation state at
// `position`: per attention layer, each kv-head's used [0, position) prefix of
// the K and V slabs (layout is head-major: head*capacity + pos, so the used
// prefix is contiguous per head); per DeltaNet layer, the full conv+recurrent
// state. Bytes beyond `position` in the KV slabs are never read (attention is
// position-bounded), so spill/restore can skip them: a spill copies
// position/capacity of the KV instead of the whole arena.
extern "C++" {
static std::vector<std::pair<std::uint64_t, std::uint64_t>> qwen_used_state_ranges(
        const ColibriV2QwenRuntime& runtime, std::uint64_t position) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    const int ck = runtime.options.cache_type_k, cv = runtime.options.cache_type_v;
    for (std::uint32_t layer_number = 0; layer_number < runtime.layers.size(); ++layer_number) {
        const auto& layer = runtime.layers[layer_number];
        if (layer.attention) {
            const auto& key = runtime.model->tensors[tensor_index(
                *runtime.model, "blk." + std::to_string(layer_number) + ".attn_k.weight")];
            const auto head_dim = layer.head_dim;
            const auto kv_heads = layer.kv_heads;
            for (int cache = 0; cache < 2; ++cache) {
                const auto base = cache ? layer.state_second : layer.state_first;
                const int type = cache ? cv : ck;
                const auto slab = qwen_kv_region_bytes(layer.cache_capacity * head_dim, type);
                const auto used = qwen_kv_region_bytes(std::min<std::uint64_t>(position,layer.cache_capacity) * head_dim, type);
                if (!used) continue;
                for (std::uint64_t head = 0; head < kv_heads; ++head)
                    ranges.emplace_back(base + head * slab, used);
            }
        } else {
            const auto& conv = runtime.model->tensors[layer.static_tensors[6]];
            const auto& a = runtime.model->tensors[layer.static_tensors[8]];
            const auto& norm = runtime.model->tensors[layer.static_tensors[9]];
            ranges.emplace_back(layer.state_first, conv.shape[0] * conv.shape[1] * sizeof(float));
            ranges.emplace_back(layer.state_second, a.shape[0] * norm.shape[0] * norm.shape[0] * sizeof(float));
        }
    }
    return ranges;
}
}  // extern "C++"

static void qwen_free_host_prompt(ColibriV2QwenRuntime& runtime, std::size_t idx) {
    QwenHostPrompt& e = runtime.host_prompts[idx];
    std::free(e.state);
    for (auto& s : e.snapshots) std::free(s.state);
    runtime.host_cache_used_bytes -= e.bytes;
    runtime.host_prompts.erase(runtime.host_prompts.begin() + idx);
}

static void qwen_evict_host_lru(ColibriV2QwenRuntime& runtime) {
    if (runtime.host_prompts.empty()) return;
    std::size_t lru = 0;
    for (std::size_t i = 1; i < runtime.host_prompts.size(); ++i)
        if (runtime.host_prompts[i].clock < runtime.host_prompts[lru].clock) lru = i;
    qwen_free_host_prompt(runtime, lru);
}

// Spill an inactive slot's full arena (KV + DeltaNet state) and its reuse
// checkpoints to host RAM, so a later request that continues this conversation
// restores from RAM instead of reprefilling ~30k tokens cold. Only substantial
// conversations are worth caching; short side-requests are skipped.
static void qwen_spill_slot_to_host(ColibriV2QwenRuntime& runtime, std::size_t slot) {
    if (!runtime.host_cache_limit_bytes || !runtime.state_bytes) return;
    const QwenSequence& seq = runtime.sequences[slot];
    const bool active = slot == runtime.active_sequence;
    const auto& tokens = active ? runtime.processed_tokens : seq.processed_tokens;
    const auto position = active ? runtime.position : seq.position;
    const auto last_output = active ? runtime.last_output_token : seq.last_output_token;
    const auto& snapshots = active ? runtime.prefill_snapshots : seq.prefill_snapshots;
    // Ignore tiny utility prompts, but retain ordinary conversations before a
    // title/quota/subagent request displaces the sole GPU slot.
    if (tokens.size() < 256) return;
    for (auto& e : runtime.host_prompts)
        if (e.tokens == tokens) { e.clock = ++runtime.host_cache_clock; return; }
    // Pack only the live ranges (used KV prefixes + DeltaNet state) instead of
    // the whole arena: at small positions the spill is a fraction of state_bytes.
    const auto ranges = qwen_used_state_ranges(runtime, position);
    std::uint64_t packed_bytes = 0;
    for (const auto& r : ranges) packed_bytes += r.second;
    if (packed_bytes > runtime.host_cache_limit_bytes) return;
    std::uint64_t valid_snaps = 0;
    for (const auto& s : snapshots) if (s.valid) ++valid_snaps;
    const std::uint64_t snapshot_capacity = runtime.prefill_snapshot_bytes
        ? (runtime.host_cache_limit_bytes-packed_bytes)/runtime.prefill_snapshot_bytes
        : 0;
    const std::uint64_t snapshots_to_copy=std::min(valid_snaps,snapshot_capacity);
    const std::uint64_t need =
        packed_bytes + snapshots_to_copy * runtime.prefill_snapshot_bytes;
    while (runtime.host_cache_used_bytes > runtime.host_cache_limit_bytes - need
           && !runtime.host_prompts.empty())
        qwen_evict_host_lru(runtime);
    if (runtime.host_cache_used_bytes > runtime.host_cache_limit_bytes - need) return;
    void* buf = std::malloc(packed_bytes);
    if (!buf) return;
    std::uint64_t cursor = 0;
    bool copy_failed = false;
    for (const auto& r : ranges) {
        if (colibri_gpu_download(static_cast<char*>(buf) + cursor,
                seq.state + r.first, r.second, runtime.stream) != 0) { copy_failed = true; break; }
        cursor += r.second;
    }
    if (copy_failed || colibri_gpu_stream_sync(runtime.stream) != 0) { std::free(buf); return; }
    QwenHostPrompt e;
    e.tokens = tokens;
    e.state = buf;
    e.position = position;
    e.last_output_token = last_output;
    e.bytes = packed_bytes;
    std::uint64_t copied_snapshots=0;
    for (const auto& s : snapshots) {
        if (!s.valid || s.tokens.empty() || copied_snapshots>=snapshots_to_copy) continue;
        void* sbuf = std::malloc(runtime.prefill_snapshot_bytes);
        if (!sbuf) break;  // partial checkpoint set is fine; arena reuse still works
        if (colibri_gpu_download(sbuf, s.device, runtime.prefill_snapshot_bytes, runtime.stream) != 0
            || colibri_gpu_stream_sync(runtime.stream) != 0) { std::free(sbuf); break; }
        e.snapshots.push_back({s.tokens, sbuf, s.last_output});
        e.bytes += runtime.prefill_snapshot_bytes;
        ++copied_snapshots;
    }
    e.clock = ++runtime.host_cache_clock;
    runtime.host_cache_used_bytes += e.bytes;
    runtime.host_prompts.push_back(std::move(e));
}

// Restore a host-cached conversation into an (inactive) victim slot: HtoD the
// saved arena, its checkpoints, and bookkeeping. Any leftover victim checkpoints
// beyond the restored set are invalidated.
static bool qwen_restore_host_to_slot(ColibriV2QwenRuntime& runtime,
        std::size_t entry_idx, std::size_t victim) {
    QwenHostPrompt& e = runtime.host_prompts[entry_idx];
    QwenSequence& seq = runtime.sequences[victim];
    const bool active = victim == runtime.active_sequence;
    auto& tokens = active ? runtime.processed_tokens : seq.processed_tokens;
    auto& snapshots = active ? runtime.prefill_snapshots : seq.prefill_snapshots;
    auto& snapshot_clock = active
        ? runtime.prefill_snapshot_clock : seq.prefill_snapshot_clock;
    // The packed layout is a pure function of (runtime config, position), so the
    // ranges recomputed here match the spill order exactly.
    const auto ranges = qwen_used_state_ranges(runtime, e.position);
    std::uint64_t cursor = 0;
    for (const auto& r : ranges) {
        if (colibri_gpu_upload(seq.state + r.first,
                static_cast<const char*>(e.state) + cursor, r.second, runtime.stream) != 0)
            return false;
        cursor += r.second;
    }
    if (colibri_gpu_stream_sync(runtime.stream) != 0) return false;
    tokens = std::move(e.tokens);
    if (active) {
        runtime.position = e.position;
        runtime.last_output_token = e.last_output_token;
    } else {
        seq.position = e.position;
        seq.last_output_token = e.last_output_token;
    }
    std::size_t slot_i = 0;
    for (auto& hs : e.snapshots) {
        if (slot_i >= snapshots.size()) break;
        auto& dst = snapshots[slot_i++];
        if (colibri_gpu_upload(dst.device, hs.state, runtime.prefill_snapshot_bytes, runtime.stream) != 0
            || colibri_gpu_stream_sync(runtime.stream) != 0)
            break;
        dst.tokens = std::move(hs.tokens);
        dst.last_output = hs.last_output;
        dst.valid = true;
        dst.clock = ++snapshot_clock;
    }
    for (std::size_t i = slot_i; i < snapshots.size(); ++i)
        snapshots[i].valid = false;
    qwen_free_host_prompt(runtime, entry_idx);
    return true;
}

// Route a prompt to the slot/host entry it continues; recycle the LRU slot
// otherwise. A slot/entry owns the prompt when it covers most of its committed
// tokens or provides a substantial absolute prefix. Before a divergent request
// overwrites that slot, the host cache preserves it even in one-slot mode.
static void qwen_route_sequence(ColibriV2QwenRuntime& runtime,
        const std::uint32_t* prompt, std::uint64_t prompt_count) {
    const bool host_cache = runtime.host_cache_limit_bytes != 0;
    if (runtime.sequences.size() <= 1 && !host_cache) return;
    constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    std::size_t best = kNone;
    std::uint64_t best_match = 0;
    auto consider = [&](std::size_t i, const std::vector<std::uint32_t>& tokens) {
        const std::uint64_t m = qwen_sequence_match(tokens, prompt, prompt_count);
        if (m > best_match && qwen_cache_match_useful(m,tokens.size())) {
            best_match = m; best = i;
        }
    };
    consider(runtime.active_sequence, runtime.processed_tokens);
    for (std::size_t i = 0; i < runtime.sequences.size(); ++i)
        if (i != runtime.active_sequence) consider(i, runtime.sequences[i].processed_tokens);
    std::size_t best_host = kNone;
    std::uint64_t best_host_match = 0;
    if (host_cache)
        for (std::size_t i = 0; i < runtime.host_prompts.size(); ++i) {
            const auto& t = runtime.host_prompts[i].tokens;
            const std::uint64_t m = qwen_sequence_match(t, prompt, prompt_count);
            if (m > best_host_match && qwen_cache_match_useful(m,t.size())) {
                best_host_match = m; best_host = i;
            }
        }
    auto lru_slot = [&]() {
        std::size_t v = 0;
        for (std::size_t i = 1; i < runtime.sequences.size(); ++i)
            if (runtime.sequences[i].clock < runtime.sequences[v].clock) v = i;
        return v;  // != active: the active slot is the most-recently-stamped
    };
    if (std::getenv("COLIBRI_ROUTE_TRACE"))
        std::fprintf(stderr, "[route] active=%zu best=%lld match=%llu host_best=%lld host_match=%llu entries=%zu\n",
            runtime.active_sequence, best==kNone?-1LL:(long long)best, (unsigned long long)best_match,
            best_host==kNone?-1LL:(long long)best_host, (unsigned long long)best_host_match,
            runtime.host_prompts.size());
    if (best != kNone && best_match >= best_host_match) {
        const auto& best_tokens = best == runtime.active_sequence
            ? runtime.processed_tokens : runtime.sequences[best].processed_tokens;
        if (host_cache && best_match < best_tokens.size())
            qwen_spill_slot_to_host(runtime, best);
        if (best != runtime.active_sequence) qwen_switch_sequence(runtime, best);
    } else if (best_host != kNone) {
        const std::size_t victim = lru_slot();
        qwen_spill_slot_to_host(runtime, victim);
        // Spilling may evict or reorder host entries to stay within budget;
        // resolve the index again rather than restoring a stale vector index.
        best_host=kNone;
        best_host_match=0;
        for(std::size_t i=0;i<runtime.host_prompts.size();++i){
            const auto&t=runtime.host_prompts[i].tokens;
            const auto m=qwen_sequence_match(t,prompt,prompt_count);
            if(m>best_host_match&&qwen_cache_match_useful(m,t.size())){
                best_host_match=m;best_host=i;
            }
        }
        if(best_host!=kNone)qwen_restore_host_to_slot(runtime,best_host,victim);
        qwen_switch_sequence(runtime, victim);
    } else {
        const std::size_t victim = lru_slot();
        if (host_cache) qwen_spill_slot_to_host(runtime, victim);
        if (victim != runtime.active_sequence) qwen_switch_sequence(runtime, victim);
        if (std::getenv("COLIBRI_ROUTE_TRACE"))
            std::fprintf(stderr, "[route] victim=%zu spilled_entries=%zu\n", victim, runtime.host_prompts.size());
    }
    runtime.sequences[runtime.active_sequence].clock = ++runtime.sequence_clock;
}

// Prompt admission: diagnostics, exact-prefix reuse (live or snapshot, with the
// KV-safety guard), reset on miss, reuse stats, and the spread checkpoint
// targets. Shared verbatim by the blocking generate path and the engine so the
// two cannot drift. Disables expert-cache admission (prompt tokens must not
// pollute it); callers re-enable it when prefill completes.
static int qwen_prompt_begin(ColibriV2QwenRuntime* runtime,
        const uint32_t* prompt, uint64_t prompt_count, QwenPromptPlan& plan,
        bool require_last_logits=false) {
    // Prefix-reuse diagnostics (computed before any state mutation below). The
    // longest common prefix against the live processed_tokens and against the
    // best snapshot shows where this prompt diverges from what is cached: a
    // small LCP means early prefix churn (system/tool schema), a large LCP with
    // a miss means divergence only near the tail. Reuse itself still requires an
    // *exact* prefix (DeltaNet recurrent state cannot be rewound mid-sequence).
    auto lcp_with=[&](const std::uint32_t*tokens,std::uint64_t count)->std::uint64_t{
        const std::uint64_t limit=std::min<std::uint64_t>(count,prompt_count);
        std::uint64_t i=0;
        while(i<limit&&tokens[i]==prompt[i])++i;
        return i;
    };
    runtime->prefix_cache_last_prompt_tokens=prompt_count;
    runtime->prefix_cache_last_lcp_live=runtime->processed_tokens.empty()?0:
        lcp_with(runtime->processed_tokens.data(),runtime->processed_tokens.size());
    std::uint64_t best_snapshot_lcp=0;
    for(const auto&candidate:runtime->prefill_snapshots){
        if(!candidate.valid||candidate.tokens.empty())continue;
        best_snapshot_lcp=std::max(best_snapshot_lcp,lcp_with(candidate.tokens.data(),candidate.tokens.size()));
    }
    runtime->prefix_cache_last_lcp_snapshot=best_snapshot_lcp;
    std::uint64_t prompt_start=0;
    bool reusable=!runtime->processed_tokens.empty()&&
        runtime->processed_tokens.size()<=prompt_count;
    if(reusable){
        reusable=std::equal(
            runtime->processed_tokens.begin(),runtime->processed_tokens.end(),
            prompt
        );
        // An exact full-prompt hit remembers the previously selected token but
        // not the full LM-head logits. Sampling must replay at least the final
        // uncached prompt section so it can draw a fresh token from real logits.
        if(require_last_logits&&runtime->processed_tokens.size()==prompt_count)
            reusable=false;
    }
    uint32_t next_token=0;
    if(reusable){
        prompt_start=runtime->processed_tokens.size();
        next_token=runtime->last_output_token;
        ++runtime->prefix_cache_hits;
        runtime->prefix_cache_reused_tokens+=prompt_start;
        runtime->cancelled=false;
    }else{
        // The live state diverged (typically: the client re-encoded the
        // previous assistant reply differently than it was generated), but a
        // prefill snapshot may still match the conversation prefix exactly.
        // Only attention KV that already holds this exact prefix is valid: the
        // KV cache reflects the live processed_tokens (recurrent DeltaNet state
        // is restored from the snapshot, but attention KV is not copied). So a
        // snapshot is reusable only when its tokens are a prefix of BOTH the new
        // prompt AND the sequence currently in the KV cache. Skipping the latter
        // check would splice stale KV from a diverged conversation.
        QwenPrefillSnapshot*snapshot=nullptr;
        for(auto&candidate:runtime->prefill_snapshots){
            if(!candidate.valid||candidate.tokens.empty()||candidate.tokens.size()>prompt_count)continue;
            if(require_last_logits&&candidate.tokens.size()==prompt_count)continue;
            if(!swa_snapshot_is_resident(*runtime,candidate.tokens.size(),runtime->position))continue;
            if(!std::equal(candidate.tokens.begin(),candidate.tokens.end(),prompt))continue;
            if(candidate.tokens.size()>runtime->processed_tokens.size())continue;
            if(!std::equal(candidate.tokens.begin(),candidate.tokens.end(),runtime->processed_tokens.begin()))continue;
            if(!snapshot||candidate.tokens.size()>snapshot->tokens.size())snapshot=&candidate;
        }
        if(snapshot){
            qwen_prefill_snapshot_copy(*runtime,snapshot->device,true);
            runtime->processed_tokens.assign(snapshot->tokens.begin(),snapshot->tokens.end());
            runtime->position=snapshot->tokens.size();
            runtime->last_output_token=snapshot->last_output;
            prompt_start=snapshot->tokens.size();
            next_token=snapshot->last_output;
            snapshot->clock=++runtime->prefill_snapshot_clock;
            ++runtime->prefix_cache_hits;
            runtime->prefix_cache_reused_tokens+=prompt_start;
            runtime->cancelled=false;
        }else{
            ++runtime->prefix_cache_misses;
            const int status=colibri_v2_qwen_runtime_reset(runtime);if(status)return status;
        }
    }
    runtime->prefix_cache_last_reused_tokens=prompt_start;
    runtime->prefix_cache_reprefilled_tokens+=prompt_count-prompt_start;
    if(std::getenv("COLIBRI_PREFIX_TRACE"))
        std::fprintf(stderr,
            "[prefix] prompt=%llu reused=%llu reprefill=%llu lcp_live=%llu lcp_snapshot=%llu\n",
            static_cast<unsigned long long>(prompt_count),
            static_cast<unsigned long long>(prompt_start),
            static_cast<unsigned long long>(prompt_count-prompt_start),
            static_cast<unsigned long long>(runtime->prefix_cache_last_lcp_live),
            static_cast<unsigned long long>(runtime->prefix_cache_last_lcp_snapshot));
    auto& prompt_sequence = runtime->sequences[runtime->active_sequence];
    prompt_sequence.prompt_expert_frequency.assign(
        static_cast<std::size_t>(runtime->layers.size()) *
            runtime->model->config.expert_count,
        0
    );
    prompt_sequence.prompt_expert_history.resize(runtime->expert_history.size());
    std::transform(
        runtime->expert_history.begin(), runtime->expert_history.end(),
        prompt_sequence.prompt_expert_history.begin(),
        [](const QwenExpertHistory& history) { return history.frequency; }
    );
    prompt_sequence.prompt_expert_observations = 0;
    runtime->cache_admission_enabled=false;
    // Mid-prefill checkpoint targets, spread EVENLY across this prompt (like
    // llama.cpp's spread context checkpoints) so a mid-conversation divergence
    // always finds a checkpoint within one spacing of the divergence point.
    // Spacing adapts to the prompt: max(interval, prompt/(mids+1)), so short
    // prompts keep dense early coverage while a 30k prompt spreads its slots
    // over the whole range instead of clustering at 256/512/1024 (measured live:
    // geometric placement reused only 1024 of a 13313-token shared prefix). The
    // last slot is reserved for the exact end-of-prompt snapshot saved below.
    // Targets already covered by the reused prefix are skipped.
    plan.targets.clear();
    if(runtime->prefill_snapshot_bytes&&!runtime->options.mtp_drafts&&
       runtime->prefill_checkpoint_interval&&runtime->prefill_snapshots.size()>1){
        const std::size_t mid_slots=runtime->prefill_snapshots.size()-1;
        const std::uint64_t spacing=std::max<std::uint64_t>(
            runtime->prefill_checkpoint_interval,prompt_count/(mid_slots+1));
        for(std::uint64_t pos=spacing;
            pos<prompt_count&&plan.targets.size()<mid_slots;pos+=spacing)
            plan.targets.push_back(pos);
    }
    plan.next_target=0;
    while(plan.next_target<plan.targets.size()&&plan.targets[plan.next_target]<=prompt_start)++plan.next_target;
    plan.prompt_start=prompt_start;
    plan.next_token=next_token;
    return 0;
}

// Save every checkpoint target the prefill has reached.
static void qwen_prompt_checkpoints(ColibriV2QwenRuntime* runtime,
        const uint32_t* prompt, QwenPromptPlan& plan) {
    while(plan.next_target<plan.targets.size()&&runtime->position>=plan.targets[plan.next_target]){
        auto&slot=runtime->prefill_snapshots[plan.next_target];
        qwen_prefill_snapshot_copy(*runtime,slot.device,false);
        slot.tokens.assign(prompt,prompt+runtime->position);
        slot.last_output=0; // mid-prefill resume keeps prefilling; last_output unused
        slot.valid=true;
        slot.clock=++runtime->prefill_snapshot_clock;
        ++plan.next_target;
    }
}

// One bounded prefill unit: a single rows-chunk (clamped to the next checkpoint
// target) while enough prompt remains, then the single-token tail (bounded per
// unit so a unit's latency stays small even when chunking is disabled). Sets
// `done` once the whole prompt has been processed and next_token is known.
// Chunked prefill batches prompt tokens through the rows forward so weight
// reads amortize across the chunk; the final prompt token still runs through
// single-token decode to produce next_token. MTP needs per-token prompt pairs
// from decode, so it keeps the one-token path.
static int qwen_prefill_unit(ColibriV2QwenRuntime* runtime, const uint32_t* prompt,
        uint64_t prompt_count, QwenPromptPlan& plan, uint64_t& index,
        uint32_t& next_token, bool& done,QwenSamplingState* sampling=nullptr) {
    done=false;
    // index+3<=prompt_count is prompt_count-1-index>=2 without the unsigned
    // underflow that fired when a cache/snapshot reuse left index==prompt_count
    // (it read 1024 tokens past the prompt -> "input token out of range").
    if(runtime->prefill_rows>1&&!runtime->options.mtp_drafts&&prompt_count>1&&
       index+3<=prompt_count&&!runtime->cancelled){
        uint64_t rows=std::min<uint64_t>(runtime->prefill_rows,prompt_count-1-index);
        // Stop the chunk exactly at the next checkpoint target so checkpoints
        // land at precise positions instead of only at chunk boundaries (a
        // 1024-row chunk would otherwise overshoot an early divergence point).
        if(plan.next_target<plan.targets.size()){
            const uint64_t target=plan.targets[plan.next_target];
            if(target>index&&target-index<rows)rows=target-index;
        }
        qwen_prefill_rows(*runtime,prompt+index,static_cast<int>(rows));
        index+=rows;
        qwen_prompt_checkpoints(runtime,prompt,plan);
        if(index+3<=prompt_count&&!runtime->cancelled)return 0; // more chunks pending
    }
    uint64_t budget=64; // bounds the one-token path when chunking is off
    for(;index<prompt_count&&budget;--budget,index++){
        const int status=colibri_v2_qwen_runtime_decode(runtime,prompt[index],&next_token);
        if(status)return status;
        if(sampling&&sampling->enabled()&&index+1==prompt_count)
            next_token=qwen_sample_last_logits(*runtime,*sampling,next_token);
        qwen_prompt_checkpoints(runtime,prompt,plan);
    }
    done=(index>=prompt_count);
    return 0;
}

// Post-prefill expert placement. Manual mode selects N experts per layer.
// Automatic immutable-hybrid mode chooses a request-horizon-sized prepared
// working set from prompt-local frequency plus decayed persistent history,
// bounded by cache capacity and wall time. A low-value prompt leaves the
// previous pinned set untouched, which is important for short side requests.
static void qwen_seed_prefill_experts(
        ColibriV2QwenRuntime& runtime,
        std::uint64_t requested_generation_tokens) {
    const char* setting=std::getenv("COLIBRI_PREFILL_CACHE_SEED");
    long requested=static_cast<long>(runtime.options.prefill_cache_seed);
    bool automatic=runtime.options.prefill_cache_seed_auto!=0;
    if(setting){
        if(std::strcmp(setting,"auto")==0){
            automatic=true;requested=0;
        }else if(std::strcmp(setting,"off")==0){
            automatic=false;requested=0;
        }else{
            automatic=false;requested=std::strtol(setting,nullptr,10);
        }
    }
    const auto expert_policy=qwen_expert_policy(
        runtime,colibri::v2::ExpertExecutionPhase::prepare);
    if(runtime.gemma4||
       !expert_policy.is_hybrid()||runtime.expert_slots.empty())return;
    // Auto placement is the prepared-map policy for immutable residency.
    // Mutable mode remains unchanged until the public mode consolidation.
    if(automatic&&!runtime.options.immutable_residency)return;
    if(!automatic&&requested<=0)return;
    if(runtime.expert_residency_frozen){
        if(automatic)++runtime.prefill_cache_seed_auto_skips;
        return;
    }
    const auto started=std::chrono::steady_clock::now();
    const auto experts=runtime.model->config.expert_count;
    const auto cache_layers=qwen_cache_layer_count(runtime);
    const auto& sequence=runtime.sequences[runtime.active_sequence];
    std::uint32_t max_history_frequency=0;
    for(const auto frequency:sequence.prompt_expert_history)
        max_history_frequency=std::max(max_history_frequency,frequency);
    const bool useful_prompt=colibri::v2::expert_seed::has_useful_prompt(
        sequence.prompt_expert_observations,
        static_cast<std::uint32_t>(runtime.layers.size()),
        runtime.model->config.expert_used_count);
    if(automatic&&!colibri::v2::expert_seed::should_seed(
            sequence.prompt_expert_observations,
            static_cast<std::uint32_t>(runtime.layers.size()),
            runtime.model->config.expert_used_count,
            max_history_frequency)){
        ++runtime.prefill_cache_seed_auto_skips;
        return;
    }
    if(automatic&&!useful_prompt&&std::any_of(
            runtime.expert_slots.begin(),runtime.expert_slots.end(),
            [](const QwenExpertSlot& slot){return slot.pinned;})){
        ++runtime.prefill_cache_seed_auto_skips;
        return;
    }
    const std::size_t per_layer=automatic
        ? colibri::v2::expert_seed::auto_experts_per_layer(
            runtime.expert_slots.size(),cache_layers,
            requested_generation_tokens)
        : static_cast<std::size_t>(std::clamp<long>(requested,1,256));
    if(!per_layer){
        if(automatic)++runtime.prefill_cache_seed_auto_skips;
        return;
    }

    // Rank every layer first, then consume candidates round-robin by rank.
    // This prevents a time/byte bound from letting early layers monopolize
    // the placement phase.
    std::vector<std::vector<std::uint32_t>> ranked(runtime.layers.size());
    for(std::uint32_t layer=0;layer<runtime.layers.size();++layer){
        const auto slot_begin=runtime.expert_slots.size()*layer/cache_layers;
        const auto slot_end=runtime.expert_slots.size()*(layer+1)/cache_layers;
        const auto layer_budget=std::min(per_layer,slot_end-slot_begin);
        auto& candidates=ranked[layer];
        candidates.reserve(experts);
        for(std::uint32_t expert=0;expert<experts;++expert){
            const auto index=static_cast<std::size_t>(layer)*experts+expert;
            const auto prompt_frequency=useful_prompt&&
                index<sequence.prompt_expert_frequency.size()
                    ? sequence.prompt_expert_frequency[index]:0;
            const auto history_frequency=
                index<sequence.prompt_expert_history.size()
                    ? sequence.prompt_expert_history[index]:0;
            if(prompt_frequency||history_frequency)
                candidates.push_back(expert);
        }
        std::sort(candidates.begin(),candidates.end(),[&](std::uint32_t left,std::uint32_t right){
            const auto left_index=static_cast<std::size_t>(layer)*experts+left;
            const auto right_index=static_cast<std::size_t>(layer)*experts+right;
            const auto left_prompt=useful_prompt&&
                left_index<sequence.prompt_expert_frequency.size()
                ? sequence.prompt_expert_frequency[left_index]:0;
            const auto right_prompt=useful_prompt&&
                right_index<sequence.prompt_expert_frequency.size()
                ? sequence.prompt_expert_frequency[right_index]:0;
            const auto left_history=left_index<sequence.prompt_expert_history.size()
                ? sequence.prompt_expert_history[left_index]:0;
            const auto right_history=right_index<sequence.prompt_expert_history.size()
                ? sequence.prompt_expert_history[right_index]:0;
            const auto& a=runtime.expert_history[left_index];
            const auto& b=runtime.expert_history[right_index];
            const auto a_score=colibri::v2::expert_seed::score(left_prompt,left_history);
            const auto b_score=colibri::v2::expert_seed::score(right_prompt,right_history);
            return a_score!=b_score?a_score>b_score:
                (a.last_used!=b.last_used?a.last_used>b.last_used:left<right);
        });
        if(candidates.size()>layer_budget)candidates.resize(layer_budget);
    }
    struct Selection{std::uint32_t layer,expert;};
    std::vector<Selection> selections;
    for(std::size_t rank=0;rank<per_layer;++rank)
        for(std::uint32_t layer=0;layer<ranked.size();++layer)
            if(rank<ranked[layer].size())
                selections.push_back({layer,ranked[layer][rank]});
    if(selections.empty()){
        if(automatic)++runtime.prefill_cache_seed_auto_skips;
        return;
    }

    // The replacement is now known to be useful. Release the previous prompt's
    // pins while retaining its pages as ordinary LRU candidates.
    for(auto& slot:runtime.expert_slots)slot.pinned=false;
    auto* staging=static_cast<std::uint8_t*>(runtime.host_staging);
    std::uint64_t seeded=0,selected=0,uploaded_bytes=0;
    struct PendingUpload{std::uint64_t device,host_offset,bytes;};
    std::vector<PendingUpload> pending;
    std::uint64_t cursor=0;
    auto flush=[&](){
        for(const auto& upload:pending)
            if(colibri_gpu_upload(upload.device,staging+upload.host_offset,
                                  upload.bytes,runtime.stream)!=0)
                throw std::runtime_error("native Qwen prefill cache seed upload failed");
        if(!pending.empty()&&colibri_gpu_stream_sync(runtime.stream)!=0)
            throw std::runtime_error("native Qwen prefill cache seed synchronization failed");
        pending.clear();cursor=0;
    };
    for(const auto selection:selections){
        const auto elapsed=static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-started).count());
        if(automatic&&elapsed>=colibri::v2::expert_seed::kAutoMaxNanoseconds){
            ++runtime.prefill_cache_seed_budget_stops;break;
        }
        const auto layer=selection.layer,expert=selection.expert;
        const auto key=(static_cast<std::uint64_t>(layer)<<32)|expert;
        const auto resident=runtime.expert_residency.find(key);
        if(resident!=runtime.expert_residency.end()){
            runtime.expert_slots[resident->second].pinned=true;
            ++selected;
            continue;
        }
        const auto& plan=runtime.layers[layer];
        std::uint64_t bundle_bytes=0;
        for(int role=0;role<3;++role)
            bundle_bytes+=runtime.model->tensors[plan.expert_tensors[role]].size/experts;
        const auto slot_index=select_expert_cache_slot(
            runtime,layer,expert,false,false);
        if(slot_index==kNoExpertSlot)continue;
            if(bundle_bytes>runtime.host_staging_bytes)
                throw std::runtime_error("native Qwen prefill cache seed expert exceeds staging capacity");
            if(cursor+bundle_bytes>runtime.host_staging_bytes)flush();
            const auto bundle_start=cursor;
            for(int role=0;role<3;++role){
                const auto& tensor=runtime.model->tensors[plan.expert_tensors[role]];
                const auto bytes=tensor.size/experts;
                const auto offset=static_cast<std::uint64_t>(expert)*bytes;
                std::memcpy(staging+cursor,tensor_data(*runtime.model,tensor)+offset,bytes);
                cursor+=bytes;
            }
            auto& slot=runtime.expert_slots[slot_index];
            if(slot.valid)runtime.expert_residency.erase(slot.key);
            slot.key=key;slot.valid=true;slot.pinned=true;
            slot.prefetch_pins=0;slot.last_used=++runtime.expert_clock;
            runtime.expert_residency[key]=slot_index;
            pending.push_back({runtime.expert_cache+slot_index*runtime.expert_slot_bytes,
                               bundle_start,bundle_bytes});
            ++seeded;
            ++selected;
            uploaded_bytes+=bundle_bytes;
    }
    flush();
    runtime.prefill_cache_seeded_experts+=seeded;
    runtime.prefill_cache_seed_selected_experts+=selected;
    runtime.prefill_cache_seed_bytes+=uploaded_bytes;
    runtime.prefill_cache_seed_nanoseconds+=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
}

// Warm the file-backed pages for prompt-hot CPU experts before the first
// generated token. Selection is round-robin by rank across layers: every layer
// runs once per token, so allowing one layer to consume the entire byte budget
// would make a poor first-token working set. The pages remain ordinary,
// reclaimable OS cache; no second copy of the model is allocated.
static void qwen_prefetch_cpu_experts(ColibriV2QwenRuntime& runtime) {
    const bool automatic=runtime.options.cpu_prefetch_auto!=0;
    const auto expert_policy=qwen_expert_policy(
        runtime,colibri::v2::ExpertExecutionPhase::decode);
    if(runtime.gemma4||
       !expert_policy.routed_cpu_execution_allowed()||
       (!runtime.options.cpu_prefetch_mib&&!automatic))return;
    constexpr std::uint64_t mib=1024ULL*1024ULL;
    constexpr std::uint64_t reserve=2ULL*1024ULL*1024ULL*1024ULL;
    const auto available=available_host_memory();
    if(available<=reserve)return;
    std::uint64_t budget=0;
    if(automatic){
        budget=std::clamp<std::uint64_t>(available/64,64*mib,256*mib);
    }else budget=static_cast<std::uint64_t>(runtime.options.cpu_prefetch_mib)*mib;
    budget=std::min(budget,available-reserve);
    if(!budget)return;
    runtime.cpu_prefetch_last_budget_bytes=budget;
    const auto experts=runtime.model->config.expert_count;
    struct Candidate { std::uint32_t expert; std::uint32_t frequency; std::uint64_t last_used; };
    std::vector<std::vector<Candidate>> ranked(runtime.layers.size());
    for(std::uint32_t layer=0;layer<runtime.layers.size();++layer){
        auto& list=ranked[layer];
        for(std::uint32_t expert=0;expert<experts;++expert){
            const auto& history=runtime.expert_history[static_cast<std::size_t>(layer)*experts+expert];
            if(history.frequency)list.push_back({expert,history.frequency,history.last_used});
        }
        std::sort(list.begin(),list.end(),[](const Candidate&a,const Candidate&b){
            return a.frequency!=b.frequency?a.frequency>b.frequency:a.last_used>b.last_used;
        });
    }
    // A split checkpoint spreads experts over several mappings, so a range is
    // only meaningful against the mapping its tensor came from.
    struct Range { const std::uint8_t* base; std::uint64_t offset,bytes; };
    std::vector<Range> ranges;
    std::uint64_t selected_bytes=0,selected_experts=0;
    for(std::size_t rank=0;;++rank){
        bool any=false;
        for(std::uint32_t layer=0;layer<runtime.layers.size();++layer){
            if(rank>=ranked[layer].size())continue;
            any=true;
            const auto expert=ranked[layer][rank].expert;
            const auto& plan=runtime.layers[layer];
            std::uint64_t bundle=0;
            for(int role=0;role<3;++role)
                bundle+=runtime.model->tensors[plan.expert_tensors[role]].size/experts;
            if(bundle>budget-selected_bytes)continue;
            for(int role=0;role<3;++role){
                const auto& tensor=runtime.model->tensors[plan.expert_tensors[role]];
                const auto bytes=tensor.size/experts;
                ranges.push_back({tensor.source?tensor.source:runtime.model->data,
                    tensor.offset+static_cast<std::uint64_t>(expert)*bytes,bytes});
            }
            selected_bytes+=bundle;++selected_experts;
        }
        if(!any||selected_bytes==budget)break;
    }
    if(ranges.empty())return;
    std::sort(ranges.begin(),ranges.end(),[](const Range&a,const Range&b){
        return a.base!=b.base?a.base<b.base:a.offset<b.offset;});
    const auto started=std::chrono::steady_clock::now();
#if defined(_WIN32)
    SYSTEM_INFO si{}; GetSystemInfo(&si);
    const std::uint64_t page=si.dwPageSize>0?static_cast<std::uint64_t>(si.dwPageSize):4096ULL;
#else
    const long page_value=sysconf(_SC_PAGESIZE);
    const std::uint64_t page=page_value>0?static_cast<std::uint64_t>(page_value):4096ULL;
#endif
    struct ColdPage { const volatile std::uint8_t* address; };
    std::vector<std::uintptr_t> inspected;
    std::vector<ColdPage> cold;
    bool residency_known=true;
    inspected.reserve(static_cast<std::size_t>((selected_bytes+page-1)/page));
    cold.reserve(static_cast<std::size_t>((selected_bytes+page-1)/page));
    for(const auto& range:ranges){
        const auto begin=reinterpret_cast<std::uintptr_t>(range.base+range.offset);
        const auto end=begin+range.bytes;
        const auto aligned_begin=begin-(begin%page);
        const auto aligned_end=((end+page-1)/page)*page;
        const auto count=(aligned_end-aligned_begin)/page;
#if defined(__linux__)
        std::vector<unsigned char> resident(static_cast<std::size_t>(count));
        const bool known=mincore(reinterpret_cast<void*>(aligned_begin),
                                 aligned_end-aligned_begin,resident.data())==0;
        if(!known)residency_known=false;
#elif defined(_WIN32)
        // Use QueryWorkingSetEx to check page residency on Windows
        std::vector<PSAPI_WORKING_SET_EX_INFORMATION> wsinfo(static_cast<std::size_t>(count));
        for(std::uint64_t i=0;i<count;++i)
            wsinfo[i].VirtualAddress=reinterpret_cast<PVOID>(aligned_begin+i*page);
        const bool known=QueryWorkingSetEx(GetCurrentProcess(),wsinfo.data(),
            static_cast<DWORD>(count*sizeof(PSAPI_WORKING_SET_EX_INFORMATION)))!=0;
        if(!known)residency_known=false;
#else
        residency_known=false;
#endif
        for(std::uint64_t index=0;index<count;++index){
            const auto address=aligned_begin+index*page;
            inspected.push_back(address);
#if defined(__linux__)
            if(known&&(resident[static_cast<std::size_t>(index)]&1U))continue;
#elif defined(_WIN32)
            if(known&&wsinfo[index].VirtualAttributes.Valid)continue;
#endif
            cold.push_back({reinterpret_cast<const volatile std::uint8_t*>(address)});
        }
    }
    std::sort(inspected.begin(),inspected.end());
    inspected.erase(std::unique(inspected.begin(),inspected.end()),inspected.end());
    std::sort(cold.begin(),cold.end(),[](const ColdPage&a,const ColdPage&b){
        return a.address<b.address;
    });
    cold.erase(std::unique(cold.begin(),cold.end(),[](const ColdPage&a,const ColdPage&b){
        return a.address==b.address;
    }),cold.end());
    const auto observed_cold_pages=cold.size();
    if(automatic){
        const bool enough_pages=observed_cold_pages*page>=8*mib;
        const bool enough_pressure=observed_cold_pages*10>=inspected.size();
        if(!residency_known||!enough_pages||!enough_pressure){
            cold.clear();
            ++runtime.cpu_prefetch_auto_skips;
        }
    }
#if !defined(_WIN32)
    for(const auto& range:ranges)
        if(!cold.empty()&&runtime.model->fd>=0)
            (void)posix_fadvise(runtime.model->fd,static_cast<off_t>(range.offset),
                               static_cast<off_t>(range.bytes),POSIX_FADV_WILLNEED);
#endif
    std::uint8_t checksum=runtime.cpu_prefetch_checksum;
    for(const auto& item:cold)checksum^=*item.address;
    runtime.cpu_prefetch_checksum=checksum;
    runtime.cpu_prefetch_experts+=selected_experts;
    runtime.cpu_prefetch_bytes+=selected_bytes;
    runtime.cpu_prefetch_pages+=inspected.size();
    runtime.cpu_prefetch_cold_pages+=observed_cold_pages;
    runtime.cpu_prefetch_loaded_pages+=cold.size();
    runtime.cpu_prefetch_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-started).count();
}

// Save this prompt's end-of-prefill state so the next turn only prefills its
// suffix, and re-enable expert-cache admission for decode.
static void qwen_prompt_finish(ColibriV2QwenRuntime* runtime,
        const uint32_t* prompt, uint64_t prompt_count, uint32_t next_token,
        uint64_t requested_generation_tokens) {
    runtime->cache_admission_enabled=true;
    qwen_prefetch_cpu_experts(*runtime);
    qwen_seed_prefill_experts(*runtime,requested_generation_tokens);
    if(!runtime->prefill_snapshot_bytes||runtime->options.mtp_drafts)return;
    // Prefer the reserved slot when mid checkpoints exist, else the slot already
    // tracking this conversation, else a free slot, else the LRU.
    QwenPrefillSnapshot*slot=nullptr;
    if(runtime->prefill_checkpoint_interval&&runtime->prefill_snapshots.size()>1){
        // Mid-prefill checkpoints own slots [0,size-1); the exact end-of-prompt
        // snapshot has the reserved last slot so it never evicts a mid.
        slot=&runtime->prefill_snapshots.back();
    }
    if(!slot)for(auto&candidate:runtime->prefill_snapshots)
        if(candidate.valid&&candidate.tokens.size()<=prompt_count&&std::equal(candidate.tokens.begin(),candidate.tokens.end(),prompt)){slot=&candidate;break;}
    if(!slot)for(auto&candidate:runtime->prefill_snapshots)if(!candidate.valid){slot=&candidate;break;}
    if(!slot){slot=&runtime->prefill_snapshots[0];for(auto&candidate:runtime->prefill_snapshots)if(candidate.clock<slot->clock)slot=&candidate;}
    if(!(slot->valid&&slot->tokens.size()==prompt_count)){
        qwen_prefill_snapshot_copy(*runtime,slot->device,false);
        slot->tokens.assign(prompt,prompt+prompt_count);
        slot->last_output=next_token;
        slot->valid=true;
    }
    slot->clock=++runtime->prefill_snapshot_clock;
}

// Drafting recursively feeds the MTP block's own hidden state into the next
// speculative step. Those entries are useful only inside the current round:
// once the target has verified a prefix, its KV entries must be rebuilt from
// the target model's true hidden states. Keeping the recursive draft entries
// makes the cache drift farther from training semantics after every round and
// rapidly destroys acceptance.
static void qwen_mtp_commit_true_cache(
    ColibriV2QwenRuntime& runtime, std::uint64_t base_cache_tokens,
    const std::uint32_t* inputs, std::uint32_t count,
    std::uint64_t verified_hidden
) {
    if (!count) return;
    const int hidden = static_cast<int>(runtime.model->config.hidden_size);
    const int hidden_elements = static_cast<int>(count) * hidden;
    const auto stable_hidden =
        runtime.state + runtime.mtp_verified_hidden_offset;
    void* batch_copy_args[] = {
        &verified_hidden, const_cast<std::uint64_t*>(&stable_hidden),
        const_cast<int*>(&hidden_elements),
    };
    if (colibri_gpu_launch_named(
            "qwen_copy_vector", (hidden_elements + 255) / 256, 1, 256, 0,
            runtime.stream, batch_copy_args) != 0) {
        throw std::runtime_error("native MTP verified-hidden snapshot failed");
    }

    runtime.mtp_cache_tokens = base_cache_tokens;
    const auto initial_hidden =
        runtime.state + runtime.mtp_target_hidden_offset;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto preceding_hidden = index == 0
            ? initial_hidden
            : stable_hidden + static_cast<std::uint64_t>(index - 1)
                * hidden * sizeof(float);
        qwen_mtp_append_pair(
            runtime, inputs[index], preceding_hidden);
    }

    auto final_source = stable_hidden + static_cast<std::uint64_t>(count - 1)
        * hidden * sizeof(float);
    auto final_target = initial_hidden;
    void* final_copy_args[] = {
        &final_source, &final_target, const_cast<int*>(&hidden),
    };
    if (colibri_gpu_launch_named(
            "qwen_copy_vector", (hidden + 255) / 256, 1, 256, 0,
            runtime.stream, final_copy_args) != 0 ||
        colibri_gpu_stream_sync(runtime.stream) != 0) {
        throw std::runtime_error("native MTP true-cache commit failed");
    }
}

// One MTP draft-and-verify round for the active sequence: draft `wanted` tokens
// from the draft block, verify them in a single batched target pass, and commit
// the prefix the target agrees with. Writes the committed tokens to `out` (which
// must hold at least `wanted` entries) and returns how many were committed --
// always at least one, because the verifier's own token is committed even when
// every draft is rejected. The returned tokens are target outputs, so drafting
// quality only moves the acceptance rate, never the text.
static uint32_t qwen_mtp_round(ColibriV2QwenRuntime&runtime,uint32_t next_token,uint32_t wanted,uint32_t*out){
    if(wanted>8)wanted=8;
    if(!wanted)wanted=1;
    // The draft block's KV cache is sized for the context limit and, unlike the
    // target caches, is never rewound by a new request. Recycling it when it
    // would overflow keeps a long-running server in bounds; the cost is a few
    // poor drafts, not a wrong answer.
    if(runtime.mtp_cache_tokens+wanted+1>runtime.options.context_limit)runtime.mtp_cache_tokens=0;
    const auto base_cache_tokens=runtime.mtp_cache_tokens;
    std::array<uint32_t,8>drafts{};
    uint32_t draft_input=next_token;
    std::uint64_t draft_hidden=runtime.state+runtime.mtp_target_hidden_offset;
    const auto draft_started=std::chrono::steady_clock::now();
    for(uint32_t index=0;index<wanted;++index){
        drafts[index]=qwen_mtp_draft(runtime,draft_input,draft_hidden);
        draft_input=drafts[index];
        draft_hidden=runtime.state+runtime.mtp_draft_hidden_offset;
    }
    runtime.mtp_draft_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-draft_started).count();
    std::array<uint32_t,8>inputs{},verified{};
    inputs[0]=next_token;
    for(uint32_t index=1;index<wanted;++index)inputs[index]=drafts[index-1];
    qwen_snapshot_delta_state(runtime,false);
    const auto batch_started=std::chrono::steady_clock::now();
    std::uint64_t batch_hidden=0;
    qwen_verify_target_rows(runtime,inputs.data(),static_cast<int>(wanted),verified.data(),&batch_hidden);
    runtime.mtp_verify_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-batch_started).count();
    uint32_t valid=0;
    bool rejected=false;
    for(;valid<wanted;++valid)if(verified[valid]!=drafts[valid]){++valid;rejected=true;break;}
    if(rejected&&valid<wanted){
        const auto rollback_started=std::chrono::steady_clock::now();
        const auto batch_rejected_token=verified[valid-1];
        std::vector<float>batch_trace;
        const bool trace=std::getenv("COLIBRI_MTP_TRACE")!=nullptr;
        const int trace_hidden=static_cast<int>(runtime.model->config.hidden_size);
        if(trace){
            batch_trace.resize(trace_hidden);
            const auto trace_source=batch_hidden+static_cast<std::uint64_t>(valid-1)*trace_hidden*sizeof(float);
            if(colibri_gpu_download(batch_trace.data(),trace_source,trace_hidden*sizeof(float),runtime.stream)!=0||colibri_gpu_stream_sync(runtime.stream)!=0)throw std::runtime_error("native MTP trace download failed");
        }
        qwen_snapshot_delta_state(runtime,true);
        std::uint64_t replay_hidden=0;
        const auto replay_started=std::chrono::steady_clock::now();
        qwen_verify_target_rows(runtime,inputs.data(),static_cast<int>(valid),verified.data(),&replay_hidden);
        qwen_mtp_commit_true_cache(
            runtime,base_cache_tokens,inputs.data(),valid,replay_hidden);
        runtime.processed_tokens.insert(runtime.processed_tokens.end(),inputs.begin(),inputs.begin()+valid);
        runtime.position+=valid;
        runtime.last_output_token=verified[valid-1];
        runtime.decode_calls+=valid;
        runtime.decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-replay_started).count();
        if(trace){
            std::vector<float>replay_trace(trace_hidden);
            const auto trace_source=runtime.state+runtime.mtp_target_hidden_offset;
            if(colibri_gpu_download(replay_trace.data(),trace_source,trace_hidden*sizeof(float),runtime.stream)!=0||colibri_gpu_stream_sync(runtime.stream)!=0)throw std::runtime_error("native MTP replay trace download failed");
            float maximum=0.0f;
            for(int index=0;index<trace_hidden;++index)maximum=std::max(maximum,std::fabs(batch_trace[index]-replay_trace[index]));
            std::fprintf(stderr,"mtp reject position=%llu row=%u draft=%u batch=%u replay=%u hidden_max_diff=%g\n",static_cast<unsigned long long>(runtime.position-valid),valid-1,drafts[valid-1],batch_rejected_token,verified[valid-1],maximum);
        }
        runtime.mtp_accepted_tokens+=valid-1;
        ++runtime.mtp_rejected_tokens;
        runtime.mtp_rollback_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-rollback_started).count();
    }else{
        qwen_mtp_commit_true_cache(
            runtime,base_cache_tokens,inputs.data(),wanted,batch_hidden);
        runtime.processed_tokens.insert(runtime.processed_tokens.end(),inputs.begin(),inputs.begin()+wanted);
        runtime.position+=wanted;
        runtime.last_output_token=verified[wanted-1];
        runtime.decode_calls+=wanted;
        runtime.decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-batch_started).count();
        runtime.mtp_accepted_tokens+=rejected?wanted-1:wanted;
        if(rejected)++runtime.mtp_rejected_tokens;
        valid=wanted;
    }
    for(uint32_t index=0;index<valid;++index)out[index]=verified[index];
    return valid;
}

static bool qwen_mtp_adaptive_enabled(){
    const char*setting=std::getenv("COLIBRI_MTP_ADAPTIVE");
    return !setting||setting[0]!='0';
}

static bool qwen_mtp_should_draft(const ColibriV2QwenRuntime&runtime){
    if(!qwen_mtp_adaptive_enabled())return runtime.options.mtp_drafts!=0;
    return runtime.options.mtp_drafts>=2&&
        !runtime.mtp_adaptive_disabled&&
        runtime.mtp_calibration_decode_tokens>=4;
}

static void qwen_mtp_record_decode(
    ColibriV2QwenRuntime&runtime,std::uint64_t nanoseconds
){
    if(!qwen_mtp_adaptive_enabled()||
       runtime.mtp_calibration_decode_tokens>=4)return;
    runtime.mtp_calibration_decode_nanoseconds+=nanoseconds;
    ++runtime.mtp_calibration_decode_tokens;
}

static void qwen_mtp_record_round(
    ColibriV2QwenRuntime&runtime,std::uint64_t nanoseconds,
    std::uint32_t committed
){
    if(!qwen_mtp_adaptive_enabled()||runtime.mtp_adaptive_disabled)return;
    runtime.mtp_calibration_round_nanoseconds+=nanoseconds;
    runtime.mtp_calibration_round_tokens+=committed;
    ++runtime.mtp_calibration_rounds;
    if(runtime.mtp_calibration_rounds<4||
       !runtime.mtp_calibration_round_tokens||
       !runtime.mtp_calibration_decode_tokens)return;
    const auto baseline_per_token=
        runtime.mtp_calibration_decode_nanoseconds/
        runtime.mtp_calibration_decode_tokens;
    const auto mtp_per_token=
        runtime.mtp_calibration_round_nanoseconds/
        runtime.mtp_calibration_round_tokens;
    // Keep speculation only when it clears a useful margin. Tiny apparent
    // wins are run-to-run noise and do not repay rejection variance.
    if(mtp_per_token*100>=baseline_per_token*95){
        runtime.mtp_adaptive_disabled=true;
        if(!runtime.mtp_adaptive_reported){
            std::fprintf(stderr,
                "[colibri-v2] MTP adaptive fallback: %llu us/token vs "
                "%llu us/token ordinary decode\n",
                static_cast<unsigned long long>(mtp_per_token/1000),
                static_cast<unsigned long long>(baseline_per_token/1000));
            runtime.mtp_adaptive_reported=true;
        }
    }
}

int colibri_v2_qwen_runtime_generate(ColibriV2QwenRuntime*runtime,const uint32_t*prompt,uint64_t prompt_count,uint64_t max_tokens,ColibriV2TokenCallback callback,void*user){return guarded([&]{
    if(!runtime||!prompt||!prompt_count||!max_tokens||!callback)throw std::runtime_error("invalid native Qwen generation arguments");
    if(prompt_count>runtime->options.context_limit||
       max_tokens>runtime->options.context_limit-prompt_count)
        throw std::runtime_error("native Qwen generation exceeds the context limit");
    if(!runtime->engine_tasks.empty()||!runtime->engine_pending.empty())throw std::runtime_error("the cooperative engine is active; blocking generate is unavailable");
    qwen_unfreeze_expert_residency(*runtime);
    QwenExpertHistorySaveGuard history_save{*runtime};
    // Pick the decode slot for this prompt before any reuse/diagnostics run.
    qwen_route_sequence(*runtime, prompt, prompt_count);
    QwenPromptPlan plan;
    int status=qwen_prompt_begin(runtime,prompt,prompt_count,plan);if(status)return status;
    uint64_t index=plan.prompt_start;
    uint32_t next_token=plan.next_token;
    bool prefill_done=false;
    while(!prefill_done){
        status=qwen_prefill_unit(runtime,prompt,prompt_count,plan,index,next_token,prefill_done);
        if(status)return status;
    }
    qwen_prompt_finish(runtime,prompt,prompt_count,next_token,max_tokens);
    QwenResidencyEpochGuard residency_epoch{*runtime};
    if(runtime->options.mtp_drafts){
        uint64_t emitted=0;
        if(callback(next_token,user)!=0)return 0;
        ++emitted;
        while(emitted<max_tokens&&!runtime->cancelled){
            if(!qwen_mtp_should_draft(*runtime)){
                const auto decode_started=std::chrono::steady_clock::now();
                status=colibri_v2_qwen_runtime_decode(
                    runtime,next_token,&next_token);
                if(status)return status;
                qwen_mtp_record_decode(
                    *runtime,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-decode_started).count());
                if(callback(next_token,user)!=0)return 0;
                ++emitted;
                continue;
            }
            const auto wanted=static_cast<uint32_t>(std::min<uint64_t>(
                runtime->options.mtp_drafts,max_tokens-emitted
            ));
            std::array<uint32_t,8>produced{};
            const auto round_started=std::chrono::steady_clock::now();
            const auto valid=qwen_mtp_round(*runtime,next_token,wanted,produced.data());
            qwen_mtp_record_round(
                *runtime,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-round_started).count(),
                valid);
            for(uint32_t index=0;index<valid&&emitted<max_tokens;++index){
                next_token=produced[index];
                if(callback(next_token,user)!=0)return 0;
                ++emitted;
            }
        }
        return 0;
    }
    for(uint64_t index=0;index<max_tokens&&!runtime->cancelled;index++){
        if(callback(next_token,user)!=0)break;
        if(index+1<max_tokens){status=colibri_v2_qwen_runtime_decode(runtime,next_token,&next_token);if(status)return status;}
    }
    return 0;
});}

static void qwen_task_submit_impl(ColibriV2QwenRuntime*runtime,
        const uint32_t*prompt,uint64_t prompt_count,uint64_t max_tokens,
        const uint32_t*stop_tokens,uint64_t stop_count,float temperature,
        uint32_t top_k,float top_p,uint64_t seed,bool has_seed,
        uint64_t*task_id) {
    if(!runtime||!prompt||!prompt_count||!max_tokens||!task_id)throw std::runtime_error("invalid native Qwen task arguments");
    if(prompt_count>runtime->options.context_limit||
       max_tokens>runtime->options.context_limit-prompt_count)
        throw std::runtime_error("native Qwen generation exceeds the context limit");
    if(!std::isfinite(temperature)||temperature<0.0f)
        throw std::runtime_error("native Qwen temperature must be finite and non-negative");
    if(!std::isfinite(top_p)||top_p<=0.0f||top_p>1.0f)
        throw std::runtime_error("native Qwen top_p must be in (0, 1]");
    if(runtime->gemma4&&temperature>0.0f)
        throw std::runtime_error("native Gemma 4 sampling is not implemented yet");
    QwenEngineTask task;
    task.prompt.assign(prompt,prompt+prompt_count);
    if(stop_tokens&&stop_count)task.stop_tokens.assign(stop_tokens,stop_tokens+stop_count);
    task.max_tokens=max_tokens;
    task.sampling.temperature=temperature;
    task.sampling.top_k=top_k;
    task.sampling.top_p=top_p;
    std::lock_guard<std::mutex> lock(runtime->engine_mutex);
    constexpr std::size_t kMaxQueuedTasks = 256;
    if(runtime->engine_pending.size()>=kMaxQueuedTasks||
       runtime->engine_tasks.size()>=kMaxQueuedTasks-runtime->engine_pending.size())
        throw std::runtime_error("native Qwen engine queue is full");
    task.id=runtime->engine_next_task_id++;
    task.sampling.rng=has_seed?seed:
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())^
        (task.id*0x9e3779b97f4a7c15ULL);
    if(!task.sampling.rng)task.sampling.rng=0x9e3779b97f4a7c15ULL;
    *task_id=task.id;
    runtime->engine_pending.push_back(std::move(task));
}

int colibri_v2_qwen_task_submit(ColibriV2QwenRuntime*runtime,const uint32_t*prompt,uint64_t prompt_count,uint64_t max_tokens,const uint32_t*stop_tokens,uint64_t stop_count,uint64_t*task_id){return guarded([&]{
    qwen_task_submit_impl(runtime,prompt,prompt_count,max_tokens,stop_tokens,
        stop_count,0.0f,20,0.95f,0,false,task_id);
    return 0;
});}

int colibri_v2_qwen_task_submit_sampling(ColibriV2QwenRuntime*runtime,const uint32_t*prompt,uint64_t prompt_count,uint64_t max_tokens,const uint32_t*stop_tokens,uint64_t stop_count,float temperature,uint32_t top_k,float top_p,uint64_t seed,uint32_t has_seed,uint64_t*task_id){return guarded([&]{
    qwen_task_submit_impl(runtime,prompt,prompt_count,max_tokens,stop_tokens,
        stop_count,temperature,top_k,top_p,seed,has_seed!=0,task_id);
    return 0;
});}

int colibri_v2_qwen_task_cancel(ColibriV2QwenRuntime*runtime,uint64_t task_id){return guarded([&]{
    if(!runtime||!task_id)throw std::runtime_error("invalid native Qwen task handle");
    std::lock_guard<std::mutex> lock(runtime->engine_mutex);
    runtime->engine_cancel_requests.push_back(task_id);
    return 0;
});}

// Move the active slot's mirrored bookkeeping (position/last_output/processed
// tokens) back into sequences[active] (park) or out again (unpark), so code can
// address every slot uniformly through sequences[] without switching.
static void qwen_park_active(ColibriV2QwenRuntime& rt, bool park) {
    QwenSequence& a = rt.sequences[rt.active_sequence];
    if (park) {
        a.position = rt.position;
        a.last_output_token = rt.last_output_token;
        a.processed_tokens.swap(rt.processed_tokens);
    } else {
        rt.position = a.position;
        rt.last_output_token = a.last_output_token;
        rt.processed_tokens.swap(a.processed_tokens);
    }
}

// Decode ONE token for each of `n` sequences with layer-level CPU/GPU overlap.
// Everything runs on the single stream with the existing single-row kernels;
// the win is scheduling: per layer, every sequence's GPU pre-MoE work (norms,
// attention/DeltaNet, router + async readback, shared experts) is queued FIRST,
// so while the CPU runs sequence A's expert phase (route policy, paging,
// qwen_cpu_moe) the GPU is already executing sequence B's queued work. Decode
// is dominated by that CPU phase, so overlap approaches aggregate 2x at n=2.
// Per-sequence workspace/host slices keep buffers independent; the shared
// expert-paging staging area is fenced with staging_event against its async
// stream-queued consumers. Requires the caller to have parked the active
// mirror; requires a decode policy with CPU expert execution and no MTP.
static void qwen_decode_multi(ColibriV2QwenRuntime* runtime, std::size_t n,
        const std::size_t* slots, const std::uint32_t* inputs, std::uint32_t* outputs) {
    const auto decode_started = std::chrono::steady_clock::now();
    const auto expert_policy=qwen_expert_policy(
        *runtime,colibri::v2::ExpertExecutionPhase::decode);
    const int hidden_size = static_cast<int>(runtime->model->config.hidden_size);
    const int experts = static_cast<int>(runtime->model->config.expert_count);
    const int configured_top_k = static_cast<int>(runtime->model->config.expert_used_count);
    const int top_k = (runtime->options.expert_top_k > 0 && static_cast<int>(runtime->options.expert_top_k) < configured_top_k)
        ? static_cast<int>(runtime->options.expert_top_k) : configured_top_k;
    const float epsilon = runtime->model->config.rms_norm_epsilon ? runtime->model->config.rms_norm_epsilon : 1.0e-6f;
    const int intermediate = static_cast<int>(runtime->moe_intermediate);
    auto* staging = static_cast<std::uint8_t*>(runtime->host_staging);
    const std::uint64_t shared_base = n * runtime->decode_host_block_bytes;
    struct Seq {
        std::size_t slot; std::uint32_t input; std::uint64_t state, position;
        std::uint64_t hidden, residual, normalized, first, second, third, fourth,
            dense_q8, dense_q8_scales, activated, router_logits,
            selected_device, route_weights, logits, argmax_device,
            attention_scores;
        std::int32_t* selected_host; float *cpu_weights, *cpu_input, *cpu_activated, *cpu_output;
        std::uint64_t* winner_host;
    };
    std::vector<Seq> seqs(n);
    const auto&workspace_layout=runtime->decode_workspace_layout;
    const auto&host_layout=runtime->decode_host_layout;
    if(n>runtime->workspace_bytes/workspace_layout.bytes)
        throw std::runtime_error("native Qwen multi-decode workspace overflow");
    for (std::size_t i = 0; i < n; ++i) {
        Seq& s = seqs[i];
        s.slot = slots[i]; s.input = inputs[i];
        s.state = runtime->sequences[s.slot].state;
        s.position = runtime->sequences[s.slot].position;
        if (s.position >= runtime->options.context_limit) throw std::runtime_error("native Qwen context limit exceeded");
        if (s.input >= runtime->model->config.vocabulary_size) throw std::runtime_error("native Qwen input token is out of range");
        const auto base=runtime->workspace+i*workspace_layout.bytes;
        s.hidden = workspace_layout.hidden.address(base);
        s.residual = workspace_layout.residual.address(base);
        s.normalized = workspace_layout.normalized.address(base);
        s.first = workspace_layout.first.address(base);
        s.second = workspace_layout.second.address(base);
        s.third = workspace_layout.third.address(base);
        s.fourth = workspace_layout.fourth.address(base);
        s.dense_q8 = workspace_layout.dense_q8.address(base);
        s.dense_q8_scales = workspace_layout.dense_q8_scales.address(base);
        s.activated = workspace_layout.activated.address(base);
        s.router_logits = workspace_layout.router_logits.address(base);
        s.selected_device = workspace_layout.selected_device.address(base);
        s.route_weights = workspace_layout.route_weights.address(base);
        s.logits = workspace_layout.logits.address(base);
        s.argmax_device = workspace_layout.argmax_device.address(base);
        s.attention_scores = workspace_layout.attention_scores.address(base);
        auto* block = staging + i * runtime->decode_host_block_bytes;
        s.selected_host = reinterpret_cast<std::int32_t*>(
            block+host_layout.selected.offset);
        s.cpu_weights = reinterpret_cast<float*>(block+host_layout.weights.offset);
        s.cpu_input = reinterpret_cast<float*>(block+host_layout.input.offset);
        s.cpu_activated = reinterpret_cast<float*>(
            block+host_layout.activated.offset);
        s.cpu_output = reinterpret_cast<float*>(block+host_layout.output.offset);
        s.winner_host = reinterpret_cast<std::uint64_t*>(
            block+host_layout.winner.offset);
    }
    auto launch_named = [&](const char* name, std::uint32_t gx, std::uint32_t gy, std::uint32_t bx, void** args) { if (colibri_gpu_launch_named(name, gx, gy, bx, 0, runtime->stream, args) != 0) throw std::runtime_error(std::string("native Qwen CUDA kernel failed: ") + name); };
    std::uint64_t q8_cached_input = 0, q8_cached_normalized = 0;
    auto rms = [&](std::uint64_t input, std::uint64_t weights, std::uint64_t output) { int one_centered = 0; q8_cached_input = 0; void* args[] = {&input, &weights, &output, const_cast<int*>(&hidden_size), const_cast<float*>(&epsilon), &one_centered}; launch_named("rms_norm", 1, 1, 1024, args); };
    auto q8 = [&](std::uint64_t matrix, std::uint64_t input, std::uint64_t output, int in_size, int out_size) { if (colibri_gpu_q8_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) != 0) throw std::runtime_error("native Qwen Q8 projection failed"); };
    auto f32 = [&](std::uint64_t matrix, std::uint64_t input, std::uint64_t output, int in_size, int out_size) { void* args[] = {&matrix, &input, &output, &in_size, &out_size}; launch_named("qwen_f32_matvec_warp", (out_size + 7) / 8, 1, 256, args); };
    const char* iq2_q8_setting = std::getenv("COLIBRI_IQ2_Q8_DECODE");
    const bool iq2_q8_enabled =
        !iq2_q8_setting || iq2_q8_setting[0] != '0';
    std::uint64_t active_dense_q8 = 0, active_dense_q8_scales = 0;
    auto q8_decode = [&](const char* kernel, std::uint64_t matrix,
                         std::uint64_t input, std::uint64_t output,
                         int in_size, int out_size) {
        if (!iq2_q8_enabled || (in_size & 255) ||
            !active_dense_q8 || !active_dense_q8_scales) return false;
        if (input != q8_cached_normalized || input != q8_cached_input) {
            void* quant_args[] = {
                &input, &active_dense_q8, &active_dense_q8_scales, &in_size};
            launch_named(
                "quantize_q8_blocks", (in_size + 31) / 32, 1, 32, quant_args);
            q8_cached_input = (input == q8_cached_normalized) ? input : 0;
        }
        void* matvec_args[] = {
            &matrix, &active_dense_q8, &active_dense_q8_scales, &output,
            &in_size, &out_size};
        launch_named(kernel, out_size, 1, 128, matvec_args);
        return true;
    };
    // Same type dispatch as the single-token decode: dense weights carry the
    // checkpoint's own type (bf16 for the NVFP4 builds), not always Q8_0.
    auto dense_matvec = [&](std::size_t index, std::uint64_t input, std::uint64_t output, int in_size, int out_size) {
        std::uint64_t matrix = runtime->device_tensors[index];
        const auto type = qwen_device_type(*runtime, index);
        switch (type) {
            case 0: { void* args[] = {&matrix, &input, &output, &in_size, &out_size}; launch_named("qwen_f32_matvec_warp", (out_size + 7) / 8, 1, 256, args); return; }
            case 30: { void* args[] = {&matrix, &input, &output, &out_size, &in_size}; launch_named("bf16_matvec_warp", (out_size + 7) / 8, 1, 256, args); return; }
            case 8: if (colibri_gpu_q8_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return; break;
            case 16:
                if (q8_decode("iq2xxs_q8_matvec_transposed_warp", matrix, input, output, in_size, out_size)) return;
                if (colibri_gpu_iq2xxs_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return;
                break;
            case 18:
                if (q8_decode("iq3xxs_q8_matvec_transposed_warp", matrix, input, output, in_size, out_size)) return;
                if (colibri_gpu_iq3xxs_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return;
                break;
            case 22: if (colibri_gpu_iq2s_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return; break;
            case 21: if (colibri_gpu_iq3s_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return; break;
            case 17: if (colibri_gpu_iq2xs_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return; break;
            case 23: if (colibri_gpu_iq4xs_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return; break;
            case 10:
                if (q8_decode("q2k_q8_matvec_transposed_warp", matrix, input, output, in_size, out_size)) return;
                if (colibri_gpu_q2k_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return;
                break;
            case 11:
                if (q8_decode("q3k_q8_matvec_transposed_warp", matrix, input, output, in_size, out_size)) return;
                if (colibri_gpu_q3k_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return;
                break;
            case 13:
                if (q8_decode("q5k_q8_matvec_transposed_warp", matrix, input, output, in_size, out_size)) return;
                if (colibri_gpu_q5k_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return;
                break;
            case 12:
                if (q8_decode("q4k_q8_matvec_transposed_warp", matrix, input, output, in_size, out_size)) return;
                if (colibri_gpu_q4k_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return;
                break;
            case 14:
                if (q8_decode("q6k_q8_matvec_transposed_warp", matrix, input, output, in_size, out_size)) return;
                if (colibri_gpu_q6k_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) == 0) return;
                break;
            default: throw std::runtime_error("native Qwen dense projection type is unsupported: " + std::to_string(type));
        }
        throw std::runtime_error("native Qwen dense projection failed");
    };
    auto add = [&](std::uint64_t target, std::uint64_t source) { float scale = 1.0f; int count = hidden_size; void* args[] = {&target, &source, &scale, &count}; launch_named("scaled_add", (hidden_size + 255) / 256, 1, 256, args); };
    for (auto& s : seqs) {
        const std::uint32_t embedding_token = s.input;
        const auto embedding = qwen_stage_embedding_rows(*runtime, &embedding_token, 1);
        const int token = runtime->embeddings_host_resident ? 0 : static_cast<int>(s.input); int width = hidden_size;
        void* args[] = {const_cast<std::uint64_t*>(&embedding), const_cast<std::uint64_t*>(&s.hidden), const_cast<int*>(&token), &width};
        launch_named(qwen_embedding_kernel(qwen_device_type(*runtime, runtime->token_embeddings), false), (hidden_size + 255) / 256, 1, 256, args);
    }
    for (std::uint32_t layer_number = 0; layer_number < runtime->layers.size(); ++layer_number) {
        auto& layer = runtime->layers[layer_number];
        auto tensor = [&](std::size_t role) { return runtime->device_tensors[layer.static_tensors.at(role)]; };
        auto dense = [&](std::size_t role, std::uint64_t input, std::uint64_t output, int in_size, int out_size) { dense_matvec(layer.static_tensors.at(role), input, output, in_size, out_size); };
        const std::size_t moe_base = layer.attention ? 7 : 10;
        // Pass A: queue every sequence's GPU work up to (and including) the
        // router readback + shared experts, recording that sequence's event.
        for (auto& s : seqs) {
            active_dense_q8 = s.dense_q8;
            active_dense_q8_scales = s.dense_q8_scales;
            // Each sequence owns its own scratch, so switching sequences drops
            // the memo along with the buffer it referred to.
            q8_cached_normalized = s.normalized;
            q8_cached_input = 0;
            rms(s.hidden, tensor(0), s.normalized);
            if (!layer.attention) {
                int channels = static_cast<int>(runtime->model->tensors[layer.static_tensors[1]].shape[1]);
                int gate_elements = static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1]);
                int value_heads = static_cast<int>(runtime->model->tensors[layer.static_tensors[8]].shape[0]);
                int head_dim = static_cast<int>(runtime->model->tensors[layer.static_tensors[9]].shape[0]);
                int value_dim = value_heads * head_dim;
                int key_heads = (channels - value_dim) / (2 * head_dim);
                dense(1, s.normalized, s.first, hidden_size, channels);
                dense(2, s.normalized, s.second, hidden_size, gate_elements);
                dense(4, s.normalized, s.third, hidden_size, value_heads);
                dense(5, s.normalized, s.third + value_heads * sizeof(float), hidden_size, value_heads);
                int kernel_size = static_cast<int>(runtime->model->tensors[layer.static_tensors[6]].shape[0]);
                std::uint64_t conv_state = s.state + layer.state_first;
                auto conv_weights = tensor(6);
                void* conv_args[] = {const_cast<std::uint64_t*>(&s.first), &conv_weights, &conv_state, const_cast<std::uint64_t*>(&s.fourth), &channels, &kernel_size};
                launch_named("delta_conv_step", (channels + 255) / 256, 1, 256, conv_args);
                std::uint64_t recurrent_state = s.state + layer.state_second;
                auto decay = tensor(8), dt = tensor(7), norm = tensor(9);
                std::uint64_t beta = s.third + value_heads * sizeof(float);
                void* recurrent_args[] = {const_cast<std::uint64_t*>(&s.fourth), const_cast<std::uint64_t*>(&s.second), &beta, const_cast<std::uint64_t*>(&s.third), &decay, &dt, &norm, &recurrent_state, const_cast<std::uint64_t*>(&s.first), &key_heads, &value_heads, &head_dim, const_cast<float*>(&epsilon)};
                launch_named("qwen_delta_recurrent", value_heads, 1, 256, recurrent_args);
                dense(3, s.first, s.residual, value_dim, hidden_size);
                add(s.residual, s.hidden);
            } else {
                const int heads = static_cast<int>(runtime->model->config.attention_heads);
                const int kv_heads = static_cast<int>(runtime->model->config.attention_kv_heads);
                const int head_dim = static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1] / kv_heads);
                const int q_size = heads * 2 * head_dim, kv_size = kv_heads * head_dim;
                dense(1, s.normalized, s.first, hidden_size, q_size);
                dense(2, s.normalized, s.second, hidden_size, kv_size);
                dense(3, s.normalized, s.third, hidden_size, kv_size);
                const int rotary = static_cast<int>(runtime->model->config.rotary_dimension ? runtime->model->config.rotary_dimension : head_dim);
                const int position = static_cast<int>(s.position);
                const float theta = runtime->model->config.rope_freq_base ? runtime->model->config.rope_freq_base : 1000000.0f;
                auto qnorm = tensor(5), knorm = tensor(6);
                std::uint64_t queries = s.fourth, gates = s.fourth + q_size / 2 * sizeof(float);
                void* q_args[] = {const_cast<std::uint64_t*>(&s.first), &qnorm, &queries, &gates, const_cast<int*>(&heads), const_cast<int*>(&head_dim), const_cast<int*>(&rotary), const_cast<int*>(&position), const_cast<float*>(&theta), const_cast<float*>(&epsilon)};
                launch_named("qwen_attention_query", heads, 1, 256, q_args);
                std::uint64_t keys = s.first;
                void* k_args[] = {const_cast<std::uint64_t*>(&s.second), &knorm, &keys, const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), const_cast<int*>(&rotary), const_cast<int*>(&position), const_cast<float*>(&theta), const_cast<float*>(&epsilon)};
                launch_named("qwen_attention_key", kv_heads, 1, 256, k_args);
                std::uint64_t cache_keys = s.state + layer.state_first, cache_values = s.state + layer.state_second;
                const auto view=attention_cache_view(layer,s.position);
                int slot=view.slot,capacity=view.capacity;
                void* k_store_args[] = {&keys, &cache_keys, const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), &slot, &capacity};
                launch_named(kv_store_kernel(runtime->options.cache_type_k,true), kv_heads, 1, 256, k_store_args);
                void* v_store_args[] = {const_cast<std::uint64_t*>(&s.third), &cache_values, const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), &slot, &capacity};
                launch_named(kv_store_kernel(runtime->options.cache_type_v,false), kv_heads, 1, 256, v_store_args);
                std::uint64_t attended = s.second; int tokens=view.tokens,first_slot=view.first;
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                const char* fused_tiles=kv_fused_tiles_kernel(*runtime);
                const int fused_tile_tokens=kv_fused_tile_tokens(*runtime);
                const bool cublas_done=
                    qwen_turbo_cublas_attention(
                        *runtime,queries,s.first,cache_keys,cache_values,
                        s.attention_scores,attended,heads,kv_heads,head_dim,
                        tokens,capacity,first_slot,scale)||
                    (qwen_cublas_attention_eligible(
                        *runtime,tokens,first_slot,capacity)&&
                    colibri_gpu_attention_f16_cublas(
                        queries,s.first,cache_keys,cache_values,
                        s.attention_scores,attended,runtime->stream,
                        heads,kv_heads,head_dim,tokens,capacity,first_slot,
                        scale)==0);
                if(cublas_done){
                    // Tensor-core GQA attention already wrote `attended`.
                }else if(runtime->fused_attention&&fused_tiles&&
                   head_dim==128&&
                   heads/kv_heads<=8&&
                   (tokens+fused_tile_tokens-1)/fused_tile_tokens<=512&&
                   static_cast<std::uint64_t>((tokens+fused_tile_tokens-1)/fused_tile_tokens)*130<=
                       runtime->options.context_limit){
                    const int tile_count=(tokens+fused_tile_tokens-1)/fused_tile_tokens;
                    void* fused_args[] = {&queries, &cache_keys, &cache_values,
                        const_cast<std::uint64_t*>(&s.attention_scores),
                        const_cast<int*>(&heads),
                        const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim),
                        &tokens, &capacity, &first_slot, &scale};
                    launch_named(fused_tiles, kv_fused_grid_heads(*runtime,heads,kv_heads), tile_count, 256, fused_args);
                    void* merge_args[] = {
                        const_cast<std::uint64_t*>(&s.attention_scores),
                        &attended, const_cast<int*>(&heads),
                        const_cast<int*>(&head_dim),
                        const_cast<int*>(&tile_count)};
                    launch_named("kv_attention_fused_merge", heads, 1, 256,
                                 merge_args);
                }else{
                    void* score_args[] = {&queries, &cache_keys, const_cast<std::uint64_t*>(&s.attention_scores), const_cast<int*>(&heads), const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), &tokens, &capacity, &first_slot, &scale};
                    launch_named(kv_scores_ring_kernel(*runtime), heads, (tokens + 255) / 256, 256, score_args);
                    void* value_args[] = {const_cast<std::uint64_t*>(&s.attention_scores), &cache_values, &attended, const_cast<int*>(&heads), const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), &tokens, &capacity, &first_slot};
                    launch_named(kv_values_ring_kernel(*runtime), heads, 1, 256, value_args);
                }
                std::uint64_t gated = s.third; int elements = heads * head_dim;
                void* gate_args[] = {&attended, &gates, &gated, &elements};
                launch_named("qwen_attention_gate", (elements + 255) / 256, 1, 256, gate_args);
                dense(4, gated, s.residual, elements, hidden_size);
                add(s.residual, s.hidden);
            }
            rms(s.residual, tensor(moe_base), s.normalized);
            if (layer.dense_ffn) {
                // Dense block: one SwiGLU over the layer's own gate/up/down,
                // with no router, shared expert or expert paging to run.
                const int dense_intermediate = static_cast<int>(runtime->moe_intermediate);
                if (layer.ffn_on_host) {
                    auto* scratch = static_cast<float*>(runtime->dense_host);
                    float* host_input = scratch;
                    // dense_host owns exactly two hidden-sized DMA endpoints;
                    // qwen_cpu_dense_ffn keeps its activation in dense_scratch.
                    float* host_output = host_input + hidden_size;
                    if (colibri_gpu_download(host_input, s.normalized, hidden_size * sizeof(float), runtime->stream) != 0 ||
                        colibri_gpu_stream_sync(runtime->stream) != 0) throw std::runtime_error("native dense host FFN input transfer failed");
                    const auto host_started = std::chrono::steady_clock::now();
                    qwen_cpu_dense_ffn(*runtime, layer, host_input, host_output);
                    runtime->dense_host_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - host_started).count();
                    if (colibri_gpu_upload(s.third, host_output, hidden_size * sizeof(float), runtime->stream) != 0) throw std::runtime_error("native dense host FFN output transfer failed");
                } else {
                const auto up_half = s.first + static_cast<std::uint64_t>(dense_intermediate) * sizeof(float);
                dense(moe_base + 1, s.normalized, s.first, hidden_size, dense_intermediate);
                dense(moe_base + 2, s.normalized, up_half, hidden_size, dense_intermediate);
                int dense_count = dense_intermediate;
                void* dense_silu_args[] = {const_cast<std::uint64_t*>(&s.first), const_cast<std::uint64_t*>(&s.second), &dense_count};
                launch_named("silu_mul", (static_cast<std::uint32_t>(dense_count) + 255) / 256, 1, 256, dense_silu_args);
                dense(moe_base + 3, s.second, s.third, dense_intermediate, hidden_size);
                }
            } else {
            dense(moe_base + 1, s.normalized, s.router_logits, hidden_size, experts);
            if (colibri_gpu_route_topk(s.router_logits, s.selected_device, s.route_weights, experts, top_k, runtime->stream) != 0) throw std::runtime_error("native Qwen routing failed");
            if (colibri_gpu_download(s.selected_host, s.selected_device, top_k * sizeof(std::int32_t), runtime->stream) != 0 ||
                colibri_gpu_download(s.cpu_weights, s.route_weights, top_k * sizeof(float), runtime->stream) != 0 ||
                colibri_gpu_download(s.cpu_input, s.normalized, hidden_size * sizeof(float), runtime->stream) != 0) throw std::runtime_error("native Qwen route transfer failed");
            if (colibri_gpu_event_record(runtime->slot_events[s.slot], runtime->stream) != 0) throw std::runtime_error("native Qwen route event failed");
            auto shared_gate_matrix = tensor(moe_base + 2), shared_up_matrix = tensor(moe_base + 3);
            const auto shexp_type = runtime->model->tensors[layer.static_tensors.at(moe_base + 2)].type;
            if (shexp_type == 40) {
                auto shared_gate_scale = layer.shared_gate_scale, shared_up_scale = layer.shared_up_scale;
                void* silu_args[] = {&shared_gate_matrix, &shared_up_matrix, const_cast<std::uint64_t*>(&s.normalized), const_cast<std::uint64_t*>(&s.second), const_cast<int*>(&hidden_size), const_cast<int*>(&intermediate), &shared_gate_scale, &shared_up_scale};
                launch_named("nvfp4_swiglu_transposed", intermediate, 1, 256, silu_args);
                auto shared_down_matrix = tensor(moe_base + 4); auto shared_down_scale = layer.shared_down_scale;
                void* down_args[] = {&shared_down_matrix, const_cast<std::uint64_t*>(&s.second), const_cast<std::uint64_t*>(&s.third), const_cast<int*>(&intermediate), const_cast<int*>(&hidden_size), &shared_down_scale};
                launch_named("nvfp4_matvec_transposed", hidden_size, 1, 256, down_args);
            } else {
                void* silu_args[] = {&shared_gate_matrix, &shared_up_matrix, const_cast<std::uint64_t*>(&s.normalized), const_cast<std::uint64_t*>(&s.second), const_cast<int*>(&hidden_size), const_cast<int*>(&intermediate)};
                launch_named("q8_swiglu_transposed_warp", (intermediate + 7) / 8, 1, 256, silu_args);
                dense(moe_base + 4, s.second, s.third, intermediate, hidden_size);
            }
            auto shared_gate = tensor(moe_base + 5);
            {
                const auto sg_type=runtime->model->tensors[layer.static_tensors.at(moe_base+5)].type;
                void* shared_args[] = {const_cast<std::uint64_t*>(&s.normalized), &shared_gate, const_cast<std::uint64_t*>(&s.third), const_cast<int*>(&hidden_size)};
                launch_named(sg_type==30?"qwen_shared_scale_bf16":"qwen_shared_scale", 1, 1, 256, shared_args);
            }
        }
        // All predictions queued by the preceding layer target this layer.
        // One stream wait covers every sequence's upload batch; predictions
        // made below target layer+1 and continue overlapping this layer.
        qwen_wait_for_prefetch_layer(*runtime,layer_number);
        const char*batch_cpu_moe=std::getenv("COLIBRI_BATCHED_CPU_MOE");
        const bool batch_cpu_supported=(colibri_cpu_features()&2u)!=0||(batch_cpu_moe&&batch_cpu_moe[0]=='1');
        if(expert_policy.is_cpu()&&n==2&&batch_cpu_supported&&
           !(batch_cpu_moe&&batch_cpu_moe[0]=='0')){
            const auto expert_started=std::chrono::steady_clock::now();
            thread_local std::vector<std::int32_t> batch_selected;
            thread_local std::vector<float> batch_weights,batch_inputs,batch_activated,batch_down,batch_outputs;
            batch_selected.assign(n*top_k,0);batch_weights.assign(n*top_k,0.0f);
            batch_inputs.resize(n*hidden_size);batch_activated.resize(n*top_k*intermediate);
            batch_down.resize(n*top_k*hidden_size);batch_outputs.resize(n*hidden_size);
            for(std::size_t index=0;index<n;++index){
                auto&s=seqs[index];
                const auto wait_started=std::chrono::steady_clock::now();
                if(colibri_gpu_event_sync(runtime->slot_events[s.slot])!=0)
                    throw std::runtime_error("native Qwen route event failed");
                runtime->route_wait_nanoseconds+=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-wait_started
                    ).count();
                int route_count=top_k;
                if(runtime->options.expert_top_k>0||
                   (runtime->options.expert_top_p>0.0f&&
                    runtime->options.expert_top_p<1.0f))
                    route_count=apply_expert_router_policy(
                        s.selected_host,s.cpu_weights,top_k,
                        static_cast<int>(runtime->options.expert_top_k),
                        runtime->options.expert_top_p
                    );
                runtime->route_expert_sum+=
                    static_cast<std::uint64_t>(route_count);
                qwen_observe_and_prefetch_next_layer(
                    *runtime,
                    runtime->sequences[s.slot].expert_prefetch,
                    layer_number,s.selected_host,route_count
                );
                std::copy_n(
                    s.selected_host,route_count,
                    batch_selected.data()+index*top_k
                );
                std::copy_n(
                    s.cpu_weights,route_count,
                    batch_weights.data()+index*top_k
                );
                std::copy_n(
                    s.cpu_input,hidden_size,
                    batch_inputs.data()+index*hidden_size
                );
            }
            const auto compute_started=std::chrono::steady_clock::now();
            qwen_cpu_moe_rows(*runtime,layer,batch_selected.data(),batch_weights.data(),static_cast<int>(n),top_k,batch_inputs.data(),batch_activated.data(),batch_down.data(),batch_outputs.data());
            runtime->expert_compute_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-compute_started).count();
            for(std::size_t index=0;index<n;++index){auto&s=seqs[index];if(colibri_gpu_upload(s.fourth,batch_outputs.data()+index*hidden_size,hidden_size*sizeof(float),runtime->stream)!=0)throw std::runtime_error("native CPU MoE output upload failed");add(s.third,s.fourth);add(s.residual,s.third);std::swap(s.hidden,s.residual);}
            runtime->expert_page_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-expert_started).count();
            continue;
        }
        // Pass B: serial CPU expert phases; while sequence i runs on the CPU,
        // the GPU is still executing the queued pass-A work of the later ones.
        for (auto& s : seqs) {
            const auto route_wait_started = std::chrono::steady_clock::now();
            if (colibri_gpu_event_sync(runtime->slot_events[s.slot]) != 0) throw std::runtime_error("native Qwen route event failed");
            runtime->route_wait_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - route_wait_started).count();
            int route_count = top_k;
            if (runtime->options.expert_top_k > 0 || (runtime->options.expert_top_p > 0.0f && runtime->options.expert_top_p < 1.0f))
                route_count = apply_expert_router_policy(s.selected_host, s.cpu_weights, top_k, static_cast<int>(runtime->options.expert_top_k), runtime->options.expert_top_p);
            runtime->route_expert_sum += static_cast<std::uint64_t>(route_count);
            const auto pager_started = std::chrono::steady_clock::now();
            qwen_observe_and_prefetch_next_layer(
                *runtime,
                runtime->sequences[s.slot].expert_prefetch,
                layer_number,s.selected_host,route_count
            );
            if (expert_policy.is_cpu()) {
                const auto compute_started = std::chrono::steady_clock::now();
                qwen_cpu_moe(*runtime, layer, s.selected_host, s.cpu_weights, route_count, s.cpu_input, s.cpu_activated, s.cpu_output);
                runtime->expert_compute_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - compute_started).count();
                if (colibri_gpu_upload(s.fourth, s.cpu_output, hidden_size * sizeof(float), runtime->stream) != 0) throw std::runtime_error("native CPU MoE output upload failed");
                add(s.third, s.fourth);
            } else {
                if (runtime->expert_slots.empty()) throw std::runtime_error("native hybrid MoE requires an expert cache budget");
                // The shared paging area's previous contents may still be
                // consumed by stream-queued uploads; fence before rewriting.
                if (colibri_gpu_event_sync(runtime->staging_event) != 0) throw std::runtime_error("native Qwen staging event failed");
                auto* pag = staging + shared_base;
                std::array<std::int32_t, 256> cpu_selected{};
                std::array<float, 256> cpu_compact_weights{}, gpu_compact_weights{};
                std::array<float, 256> gpu_gate_scales{}, gpu_up_scales{}, gpu_down_scales{};
                std::array<std::uint64_t, 256> gate_pointers{}, up_pointers{},
                    down_pointers{}, native_pointers{};
                int cpu_count = 0, gpu_count = 0;
                std::uint64_t staging_cursor = 0;
                struct PendingUpload {
                    std::uint64_t device, host_offset, bytes;
                    std::size_t slot_index;
                };
                std::array<PendingUpload, 256> pending{}; int pending_count = 0;
                for (int rank = 0; rank < route_count; ++rank) {
                    const int expert = s.selected_host[rank];
                    if (expert < 0 || expert >= experts) throw std::runtime_error("native hybrid MoE selected an invalid expert");
                    const auto cache_key = (static_cast<std::uint64_t>(layer_number) << 32) | static_cast<std::uint32_t>(expert);
                    auto resident = runtime->expert_residency.find(cache_key);
                    if (resident != runtime->expert_residency.end()) {
                        const auto slot_index = resident->second; auto& slot = runtime->expert_slots[slot_index];
                        const auto& history = record_expert_access(
                            *runtime, layer_number, expert);
                        if (expert_policy.residency_may_change())
                            slot.last_used = history.last_used;
                        record_expert_cache_hit(*runtime,slot);
                        const auto device_base = runtime->expert_cache + slot_index * runtime->expert_slot_bytes; std::uint64_t role_offset = 0;
                        for (int role = 0; role < 3; ++role) { const auto bytes = runtime->model->tensors[layer.expert_tensors[role]].size / experts; const auto pointer = device_base + role_offset; if (role == 0) gate_pointers[gpu_count] = pointer; else if (role == 1) up_pointers[gpu_count] = pointer; else down_pointers[gpu_count] = pointer; role_offset += bytes; }
                        if (slot.native_valid && runtime->expert_native_cache)
                            native_pointers[gpu_count] =
                                runtime->expert_native_cache +
                                slot_index * runtime->expert_slot_bytes;
                        gpu_compact_weights[gpu_count] = s.cpu_weights[rank];
                        // See the decode path: gate/up before SiLU, down folded into `up`.
                        gpu_gate_scales[gpu_count] = qwen_expert_role_scale(*runtime, layer.expert_gate_scale, expert);
                        gpu_down_scales[gpu_count] = qwen_expert_role_scale(*runtime, layer.expert_down_scale, expert);
                        gpu_up_scales[gpu_count] = qwen_expert_role_scale(*runtime, layer.expert_up_scale, expert)
                            * gpu_down_scales[gpu_count];
                        ++gpu_count; continue;
                    }
                    ++runtime->expert_cache_misses; cpu_selected[cpu_count] = expert; cpu_compact_weights[cpu_count]=s.cpu_weights[rank];
                    if(layer.expert_down_scale!=std::numeric_limits<std::uint64_t>::max()){
                        const auto&st=runtime->model->tensors[layer.expert_down_scale];
                        const auto experts_count=runtime->model->config.expert_count;
                        const auto scale_bytes=st.size/experts_count;
                        float ds=1.0f;std::memcpy(&ds,tensor_data(*runtime->model,st)+static_cast<std::uint64_t>(expert)*scale_bytes,sizeof(float));
                        cpu_compact_weights[cpu_count]*=ds;
                    }
                    ++cpu_count;
                    const auto slot_index = select_expert_cache_slot(*runtime, layer_number, expert, true);
                    if (slot_index == kNoExpertSlot) continue;
                    auto& slot = runtime->expert_slots[slot_index]; slot.key = cache_key; slot.valid = true; slot.native_valid = false; slot.last_used = ++runtime->expert_clock; runtime->expert_residency[cache_key] = slot_index;
                    const auto slot_base = runtime->expert_cache + slot_index * runtime->expert_slot_bytes;
                    if (runtime->dma_paging) {
                        std::uint64_t role_offset = 0;
                        for (int role = 0; role < 3; ++role) { const auto& t = runtime->model->tensors[layer.expert_tensors[role]]; const auto bytes = t.size / experts; const auto offset = static_cast<std::uint64_t>(expert) * bytes; if (colibri_gpu_upload(slot_base + role_offset, tensor_data(*runtime->model,t) + offset, bytes, runtime->stream) != 0) throw std::runtime_error("native hybrid MoE DMA cache upload failed"); role_offset += bytes; }
                        if(runtime->expert_native_cache){
                            const auto gate_bytes=runtime->model->tensors[
                                layer.expert_tensors[0]].size/experts;
                            const auto up_bytes=runtime->model->tensors[
                                layer.expert_tensors[1]].size/experts;
                            const int status=colibri_gpu_nvfp4_prepare_expert(
                                slot_base,slot_base+gate_bytes,
                                slot_base+gate_bytes+up_bytes,
                                runtime->expert_native_cache+
                                    slot_index*runtime->expert_slot_bytes,
                                runtime->stream,hidden_size,intermediate);
                            if(status!=0)throw std::runtime_error(
                                "persistent NVFP4 expert preparation failed");
                            slot.native_valid=true;
                        }
                    } else {
                        const auto bundle_start = staging_cursor;
                        for (int role = 0; role < 3; ++role) { const auto& t = runtime->model->tensors[layer.expert_tensors[role]]; const auto bytes = t.size / experts; const auto offset = static_cast<std::uint64_t>(expert) * bytes; if (staging_cursor + bytes > runtime->expert_staging_bytes) throw std::runtime_error("native hybrid MoE staging overflow"); std::memcpy(pag + staging_cursor, tensor_data(*runtime->model,t) + offset, bytes); staging_cursor += bytes; }
                        pending[pending_count++] = {
                            slot_base, bundle_start,
                            staging_cursor - bundle_start, slot_index};
                    }
                }
                if (gpu_count) {
                    const auto table_bytes = static_cast<std::uint64_t>(gpu_count) * (3 * sizeof(std::uint64_t) + 4 * sizeof(float));
                    const auto table_host = device_align(staging_cursor); const auto table_device = runtime->expert_staging + runtime->expert_staging_bytes - device_align(table_bytes);
                    if (table_host + table_bytes > runtime->expert_staging_bytes) throw std::runtime_error("native hybrid MoE pointer staging overflow");
                    std::memcpy(pag + table_host, gate_pointers.data(), gpu_count * sizeof(std::uint64_t));
                    std::memcpy(pag + table_host + gpu_count * sizeof(std::uint64_t), up_pointers.data(), gpu_count * sizeof(std::uint64_t));
                    std::memcpy(pag + table_host + 2 * gpu_count * sizeof(std::uint64_t), down_pointers.data(), gpu_count * sizeof(std::uint64_t));
                    std::memcpy(pag + table_host + 3 * gpu_count * sizeof(std::uint64_t), gpu_compact_weights.data(), gpu_count * sizeof(float));
                    std::memcpy(pag + table_host + 3 * gpu_count * sizeof(std::uint64_t) + gpu_count * sizeof(float), gpu_gate_scales.data(), gpu_count * sizeof(float));
                    std::memcpy(pag + table_host + 3 * gpu_count * sizeof(std::uint64_t) + 2 * gpu_count * sizeof(float), gpu_up_scales.data(), gpu_count * sizeof(float));
                    std::memcpy(pag + table_host + 3 * gpu_count * sizeof(std::uint64_t) + 3 * gpu_count * sizeof(float), gpu_down_scales.data(), gpu_count * sizeof(float));
                    if (colibri_gpu_upload(table_device, pag + table_host, table_bytes, runtime->stream) != 0) throw std::runtime_error("native hybrid MoE table upload failed");
                    const auto gate_table = table_device, up_table = gate_table + gpu_count * sizeof(std::uint64_t), down_table = up_table + gpu_count * sizeof(std::uint64_t), weight_table = down_table + gpu_count * sizeof(std::uint64_t);
                    const auto gate_scale_table = weight_table + gpu_count * sizeof(float), up_scale_table = gate_scale_table + gpu_count * sizeof(float), down_scale_table = up_scale_table + gpu_count * sizeof(float);
                    const auto gate_type = runtime->model->tensors[layer.expert_tensors[0]].type;
                    const auto down_type = runtime->model->tensors[layer.expert_tensors[2]].type;
        // The grouped dispatch below ends in a k-quant fallback, so an
        // unhandled type would be decoded as Q5_K rather than rejected. Prepare
        // already routes such models to the CPU; this makes the silent path
        // unreachable if that ever stops holding.
        if(!qwen_gpu_expert_type_supported(gate_type)||
           !qwen_gpu_expert_type_supported(down_type))
            throw std::runtime_error(
                "native GPU expert quantization is unsupported: "+
                std::to_string(gate_type)+"/"+std::to_string(down_type));
                    const char* persistent_env =
                        std::getenv("COLIBRI_NVFP4_PERSISTENT");
                    const bool persistent_enabled =
                        persistent_env && persistent_env[0]=='1' &&
                        runtime->expert_native_cache &&
                        std::all_of(
                            native_pointers.begin(),
                            native_pointers.begin()+gpu_count,
                            [](std::uint64_t pointer){return pointer!=0;});
                    const char* tc_env = std::getenv("COLIBRI_NVFP4_DECODE_TENSOR_CORES");
                    const bool tc_enabled = tc_env && tc_env[0] == '1';
                    bool tc_done = false;
                    if (persistent_enabled && gate_type == 40 &&
                        down_type == 40) {
                        const int tc_status =
                            colibri_gpu_nvfp4_moe_persistent(
                                native_pointers.data(),weight_table,gate_scale_table,
                                up_scale_table,down_scale_table,s.normalized,
                                s.activated,s.third,runtime->stream,
                                hidden_size,intermediate,gpu_count);
                        if(tc_status==0){
                            ++runtime->nvfp4_tensor_core_moe_calls;
                            runtime->nvfp4_tensor_core_moe_last_status=0;
                            tc_done=true;
                        }else{
                            ++runtime->nvfp4_tensor_core_moe_fallbacks;
                            runtime->nvfp4_tensor_core_moe_last_status=
                                tc_status;
                            if(std::getenv(
                                   "COLIBRI_NVFP4_TENSOR_CORE_TRACE"))
                                std::fprintf(
                                    stderr,
                                    "[nvfp4-persistent] fallback status=%d "
                                    "experts=%d\n",tc_status,gpu_count);
                        }
                    }
                    if (!tc_done && tc_enabled && gate_type == 40 && down_type == 40) {
                        const int tc_status = colibri_gpu_nvfp4_moe_cublas(
                            gate_table, up_table, down_table, s.normalized,
                            s.activated, s.third, weight_table, gate_scale_table,
                            up_scale_table, down_scale_table, runtime->stream, hidden_size,
                            intermediate, gpu_count);
                        if (tc_status == 0) {
                            ++runtime->nvfp4_tensor_core_moe_calls;
                            runtime->nvfp4_tensor_core_moe_last_status = 0;
                            tc_done = true;
                        } else {
                            ++runtime->nvfp4_tensor_core_moe_fallbacks;
                            runtime->nvfp4_tensor_core_moe_last_status = tc_status;
                            if (std::getenv("COLIBRI_NVFP4_TENSOR_CORE_TRACE"))
                                std::fprintf(stderr, "[nvfp4-tc] multi-decode fallback status=%d experts=%d\n", tc_status, gpu_count);
                        }
                    }
                    if (!tc_done) {
                        const char* tiled_env = std::getenv("COLIBRI_NVFP4_TILED");
                        const bool nvfp4_tiled = tiled_env && tiled_env[0] == '1';
                        void* gate_up_args[] = {const_cast<std::uint64_t*>(&gate_table), const_cast<std::uint64_t*>(&up_table), &s.normalized, &s.activated, const_cast<int*>(&hidden_size), const_cast<int*>(&intermediate), &gpu_count, const_cast<std::uint64_t*>(&gate_scale_table), const_cast<std::uint64_t*>(&up_scale_table)};
                        launch_named(qwen_grouped_swiglu_name(gate_type,nvfp4_tiled,false).c_str(), gate_type == 40 && nvfp4_tiled ? (intermediate + 7) / 8 : intermediate, gpu_count, 256, gate_up_args);
                        const int status = qwen_launch_grouped_accumulate(runtime->stream,down_type,down_table,s.activated,s.third,weight_table,intermediate,hidden_size,gpu_count);
                        if (status != 0) throw std::runtime_error("native hybrid MoE down projection failed");
                    }
                }
                for (int index = 0; index < pending_count; ++index) {
                    const auto& upload=pending[index];
                    if (colibri_gpu_upload(
                            upload.device,pag+upload.host_offset,upload.bytes,
                            runtime->stream)!=0)
                        throw std::runtime_error(
                            "native hybrid MoE cache upload failed");
                    if(runtime->expert_native_cache){
                        const auto gate_bytes=runtime->model->tensors[
                            layer.expert_tensors[0]].size/experts;
                        const auto up_bytes=runtime->model->tensors[
                            layer.expert_tensors[1]].size/experts;
                        const int status=colibri_gpu_nvfp4_prepare_expert(
                            upload.device,upload.device+gate_bytes,
                            upload.device+gate_bytes+up_bytes,
                            runtime->expert_native_cache+
                                upload.slot_index*runtime->expert_slot_bytes,
                            runtime->stream,hidden_size,intermediate);
                        if(status!=0)throw std::runtime_error(
                            "persistent NVFP4 expert preparation failed");
                        runtime->expert_slots[
                            upload.slot_index].native_valid=true;
                    }
                }
                if (colibri_gpu_event_record(runtime->staging_event, runtime->stream) != 0) throw std::runtime_error("native Qwen staging event failed");
                if (cpu_count) {
                    const auto compute_started = std::chrono::steady_clock::now();
                    qwen_cpu_moe(*runtime, layer, cpu_selected.data(), cpu_compact_weights.data(), cpu_count, s.cpu_input, s.cpu_activated, s.cpu_output);
                    runtime->expert_compute_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - compute_started).count();
                    if (colibri_gpu_upload(s.fourth, s.cpu_output, hidden_size * sizeof(float), runtime->stream) != 0) throw std::runtime_error("native hybrid MoE output upload failed");
                    add(s.third, s.fourth);
                }
            }
            runtime->expert_page_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pager_started).count();
            }
            add(s.residual, s.third);
            std::swap(s.hidden, s.residual);
        }
    }
    for (auto& s : seqs) {
        rms(s.hidden, runtime->device_tensors[runtime->final_norm], s.normalized);
        int vocabulary = static_cast<int>(runtime->model->config.vocabulary_size);
        if (colibri_gpu_memset(s.argmax_device, 0, sizeof(std::uint64_t), runtime->stream) != 0) throw std::runtime_error("native Qwen argmax reset failed");
        auto lm_head = runtime->device_tensors[runtime->lm_head];
        void* argmax_args[] = {&lm_head, const_cast<std::uint64_t*>(&s.normalized), const_cast<std::uint64_t*>(&s.argmax_device), const_cast<int*>(&hidden_size), &vocabulary};
        launch_named(qwen_lm_head_argmax_kernel(runtime->lm_head_type), (vocabulary + 7) / 8, 1, 256, argmax_args);
        if (colibri_gpu_download(s.winner_host, s.argmax_device, sizeof(std::uint64_t), runtime->stream) != 0) throw std::runtime_error("native Qwen output transfer failed");
    }
    const auto tail_wait_started = std::chrono::steady_clock::now();
    if (colibri_gpu_stream_sync(runtime->stream) != 0) throw std::runtime_error("native Qwen output synchronization failed");
    runtime->tail_wait_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tail_wait_started).count();
    for (std::size_t i = 0; i < n; ++i) {
        outputs[i] = 0xffffffffu - static_cast<std::uint32_t>(*seqs[i].winner_host);
        auto& sequence = runtime->sequences[seqs[i].slot];
        sequence.last_output_token = outputs[i];
        sequence.processed_tokens.push_back(inputs[i]);
        ++sequence.position;
    }
    runtime->decode_calls += n;
    runtime->decode_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - decode_started).count();
}

// Route a pending task to a slot the way qwen_route_sequence does, but aware of
// slot ownership: a slot owned by another running task is untouchable, and if
// the busiest match for this prompt IS an owned slot (same conversation already
// generating), the task waits instead of duplicating the conversation elsewhere.
static bool qwen_engine_try_start(ColibriV2QwenRuntime& runtime, QwenEngineTask& task) {
    const auto* prompt = task.prompt.data();
    const std::uint64_t prompt_count = task.prompt.size();
    constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    auto tokens_of = [&](std::size_t i) -> const std::vector<std::uint32_t>& {
        return i == runtime.active_sequence ? runtime.processed_tokens
                                            : runtime.sequences[i].processed_tokens;
    };
    auto owned = [&](std::size_t i) { return runtime.slot_owner[i] >= 0; };
    std::size_t best = kNone;
    std::uint64_t best_match = 0, busy_match = 0;
    for (std::size_t i = 0; i < runtime.sequences.size(); ++i) {
        const auto& tokens = tokens_of(i);
        const std::uint64_t m = qwen_sequence_match(tokens, prompt, prompt_count);
        if (!qwen_cache_match_useful(m,tokens.size())) continue;
        if (owned(i)) busy_match = std::max(busy_match, m);
        else if (m > best_match) { best_match = m; best = i; }
    }
    std::size_t free_lru = kNone;
    for (std::size_t i = 0; i < runtime.sequences.size(); ++i)
        if (!owned(i) && (free_lru == kNone ||
            runtime.sequences[i].clock < runtime.sequences[free_lru].clock)) free_lru = i;
    if (free_lru == kNone) return false;  // every slot busy: wait
    const bool host_cache = runtime.host_cache_limit_bytes != 0;
    std::size_t best_host = kNone;
    std::uint64_t best_host_match = 0;
    if (host_cache)
        for (std::size_t i = 0; i < runtime.host_prompts.size(); ++i) {
            const auto& t = runtime.host_prompts[i].tokens;
            const std::uint64_t m = qwen_sequence_match(t, prompt, prompt_count);
            if (m > best_host_match && qwen_cache_match_useful(m,t.size())) {
                best_host_match = m; best_host = i;
            }
        }
    // The conversation is already live on a busy slot and nothing free/host
    // beats it: wait for that task to finish rather than forking the history.
    if (busy_match > best_match && busy_match > best_host_match) return false;
    std::size_t chosen;
    if (best != kNone && best_match >= best_host_match) {
        chosen = best;
        const auto& tokens = tokens_of(chosen);
        if (host_cache && best_match < tokens.size())
            qwen_spill_slot_to_host(runtime, chosen);
    } else if (best_host != kNone) {
        qwen_spill_slot_to_host(runtime, free_lru);
        best_host=kNone;
        best_host_match=0;
        for(std::size_t i=0;i<runtime.host_prompts.size();++i){
            const auto&t=runtime.host_prompts[i].tokens;
            const auto m=qwen_sequence_match(t,prompt,prompt_count);
            if(m>best_host_match&&qwen_cache_match_useful(m,t.size())){
                best_host_match=m;best_host=i;
            }
        }
        if(best_host!=kNone)qwen_restore_host_to_slot(runtime,best_host,free_lru);
        chosen = free_lru;
    } else {
        if (host_cache) qwen_spill_slot_to_host(runtime, free_lru);
        chosen = free_lru;
    }
    qwen_switch_sequence(runtime, chosen);
    runtime.sequences[chosen].clock = ++runtime.sequence_clock;
    runtime.slot_owner[chosen] = static_cast<long long>(task.id);
    task.slot = chosen;
    if (qwen_prompt_begin(&runtime, prompt, prompt_count, task.plan,
                          task.sampling.enabled()) != 0)
        throw std::runtime_error("native Qwen prompt admission failed");
    task.index = task.plan.prompt_start;
    task.next_token = task.plan.next_token;
    if (task.index >= prompt_count) {  // full reuse: nothing to prefill
        qwen_prompt_finish(
            &runtime,prompt,prompt_count,task.next_token,task.max_tokens);
        task.phase = 2;
    } else {
        task.phase = 1;
    }
    return true;
}

int colibri_v2_qwen_engine_step(ColibriV2QwenRuntime*runtime,ColibriV2QwenTaskEvent*events,uint64_t capacity,uint64_t*count){return guarded([&]{
    if(!runtime||!events||!capacity||!count)throw std::runtime_error("invalid native Qwen engine arguments");
    if(!runtime->decode_ready)throw std::runtime_error("native Qwen runtime is not prepared for decode");
    *count=0;
    {   // Admit new submissions and cancellation requests.
        std::lock_guard<std::mutex> lock(runtime->engine_mutex);
        for(auto&task:runtime->engine_pending)runtime->engine_tasks.push_back(std::move(task));
        runtime->engine_pending.clear();
        for(const auto id:runtime->engine_cancel_requests)
            for(auto&task:runtime->engine_tasks)if(task.id==id)task.cancelled=true;
        runtime->engine_cancel_requests.clear();
    }
    if(runtime->engine_tasks.empty()){
        qwen_unfreeze_expert_residency(*runtime);
        return 0;
    }
    auto emit=[&](std::uint64_t id,std::uint32_t token,std::uint32_t kind){
        events[*count]=ColibriV2QwenTaskEvent{id,token,kind};++*count;
    };
    std::vector<std::uint64_t> finished;
    std::vector<QwenEngineTask*> pending_decode;
    const std::size_t total=runtime->engine_tasks.size();
    for(std::size_t visit=0;visit<total&&*count+2<=capacity;++visit){
        auto&task=runtime->engine_tasks[(runtime->engine_cursor+visit)%total];
        try{
            if(task.cancelled&&task.phase!=2){finished.push_back(task.id);emit(task.id,0,1);continue;}
            if(task.phase==0){
                if(!qwen_engine_try_start(*runtime,task))continue;
                // Report the exact prefix/snapshot reuse point before doing
                // more work. Yield this visit so the server can print a useful
                // "starting at" line before the first potentially long chunk.
                emit(task.id,static_cast<std::uint32_t>(task.index),3);
                continue;
            }
            if(task.phase==1){
                qwen_switch_sequence(*runtime,task.slot);
                runtime->cache_admission_enabled=false;
                bool done=false;
                const int status=qwen_prefill_unit(runtime,task.prompt.data(),task.prompt.size(),task.plan,task.index,task.next_token,done,&task.sampling);
                if(status)throw std::runtime_error(
                    error.empty()?"native Qwen prefill failed":
                    "native Qwen prefill failed: "+error);
                if(done){
                    qwen_prompt_finish(
                        runtime,task.prompt.data(),task.prompt.size(),
                        task.next_token,task.max_tokens);
                    task.phase=2;
                }
                emit(task.id,static_cast<std::uint32_t>(task.index),3);
                continue;
            }
            // phase 2: emit the buffered token, then either finish or defer
            // computing the next one -- the emit-then-compute order matches the
            // blocking decode loop, so a single-task engine run stays
            // bit-identical to blocking generate.
            qwen_freeze_expert_residency(*runtime);
            emit(task.id,task.next_token,0);
            ++task.emitted;
            const bool stopped=std::find(task.stop_tokens.begin(),task.stop_tokens.end(),task.next_token)!=task.stop_tokens.end();
            if(stopped||task.cancelled||task.emitted>=task.max_tokens){
                finished.push_back(task.id);emit(task.id,0,1);continue;
            }
            if(!task.drafted.empty()){
                // An earlier MTP round already committed this token; no decode.
                task.next_token=task.drafted.front();
                task.drafted.pop_front();
                continue;
            }
            pending_decode.push_back(&task);
        }catch(const std::exception&error){
            // Isolate the failure: this task dies, the others keep running.
            std::fprintf(stderr,"[colibri-v2] engine task %llu failed: %s\n",
                static_cast<unsigned long long>(task.id),error.what());
            finished.push_back(task.id);emit(task.id,0,2);
        }
    }
    // Compute the deferred next tokens. Two or more sequences go through the
    // multi-sequence driver (layer-level CPU/GPU overlap); a single one keeps
    // the plain decode path.
    if(!pending_decode.empty()){
        runtime->cache_admission_enabled=true;
        std::size_t batched=0;
        const bool all_greedy=std::all_of(pending_decode.begin(),pending_decode.end(),
            [](const QwenEngineTask*task){return !task->sampling.enabled();});
        const auto expert_policy=qwen_expert_policy(
            *runtime,colibri::v2::ExpertExecutionPhase::decode);
        if(all_greedy&&pending_decode.size()>=2&&
           runtime->multi_decode_capacity>=2&&
           !runtime->expert_residency_frozen&&
           expert_policy.routed_cpu_execution_allowed()){
            batched=std::min<std::size_t>(pending_decode.size(),runtime->multi_decode_capacity);
            std::vector<std::size_t> slots(batched);
            std::vector<std::uint32_t> batch_inputs(batched),batch_outputs(batched);
            for(std::size_t i=0;i<batched;++i){slots[i]=pending_decode[i]->slot;batch_inputs[i]=pending_decode[i]->next_token;}
            qwen_park_active(*runtime,true);
            try{
                qwen_decode_multi(runtime,batched,slots.data(),batch_inputs.data(),batch_outputs.data());
                qwen_park_active(*runtime,false);
                for(std::size_t i=0;i<batched;++i)pending_decode[i]->next_token=batch_outputs[i];
            }catch(const std::exception&error){
                qwen_park_active(*runtime,false);
                // The whole batch shares one stream of work; fail all of it.
                std::fprintf(stderr,"[colibri-v2] engine decode batch failed: %s\n",error.what());
                for(std::size_t i=0;i<batched;++i){finished.push_back(pending_decode[i]->id);emit(pending_decode[i]->id,0,2);}
            }
        }
        for(std::size_t i=batched;i<pending_decode.size();++i){
            auto*task=pending_decode[i];
            try{
                qwen_switch_sequence(*runtime,task->slot);
                // Speculative decoding commits the verifier's greedy argmax, so
                // it can only serve tasks that are themselves greedy; a sampled
                // task would silently lose its temperature.
                if(runtime->options.mtp_drafts&&!task->sampling.enabled()&&
                   qwen_mtp_should_draft(*runtime)){
                    const auto wanted=static_cast<uint32_t>(std::min<std::uint64_t>(
                        runtime->options.mtp_drafts,task->max_tokens-task->emitted
                    ));
                    std::array<uint32_t,8>produced{};
                    const auto round_started=std::chrono::steady_clock::now();
                    const auto valid=qwen_mtp_round(*runtime,task->next_token,wanted,produced.data());
                    qwen_mtp_record_round(
                        *runtime,
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now()-round_started).count(),
                        valid);
                    task->next_token=produced[0];
                    for(uint32_t index=1;index<valid;++index)task->drafted.push_back(produced[index]);
                }else{
                    const auto decode_started=std::chrono::steady_clock::now();
                    const int status=colibri_v2_qwen_runtime_decode(runtime,task->next_token,&task->next_token);
                    if(status)throw std::runtime_error("native Qwen decode failed");
                    if(runtime->options.mtp_drafts&&!task->sampling.enabled())
                        qwen_mtp_record_decode(
                            *runtime,
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now()-decode_started).count());
                    if(task->sampling.enabled())
                        task->next_token=qwen_sample_last_logits(*runtime,task->sampling,task->next_token);
                }
            }catch(const std::exception&error){
                std::fprintf(stderr,"[colibri-v2] engine task %llu decode failed: %s\n",
                    static_cast<unsigned long long>(task->id),error.what());
                finished.push_back(task->id);emit(task->id,0,2);
            }
        }
    }
    if(total)runtime->engine_cursor=(runtime->engine_cursor+1)%total;
    for(const auto id:finished){
        for(std::size_t i=0;i<runtime->slot_owner.size();++i)
            if(runtime->slot_owner[i]==static_cast<long long>(id))runtime->slot_owner[i]=-1;
        runtime->engine_tasks.erase(
            std::remove_if(runtime->engine_tasks.begin(),runtime->engine_tasks.end(),
                [&](const QwenEngineTask&t){return t.id==id;}),
            runtime->engine_tasks.end());
    }
    if(!finished.empty())qwen_save_expert_history(*runtime);
    if(runtime->engine_tasks.empty())
        qwen_unfreeze_expert_residency(*runtime);
    return 0;
});}

}
