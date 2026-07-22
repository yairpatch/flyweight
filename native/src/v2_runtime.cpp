#include "colibri_v2.h"
#include "colibri_native.h"
#include "colibri_v2_provider.hpp"
#include "colibri_v2_config.hpp"
#include "colibri_v2_qwen_kernels.hpp"
#include "colibri_v2_native_kernels.hpp"
#include "qwen_cpu_kernel.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  include <dlfcn.h>
#endif

thread_local std::string error;
void fail(const char* message) { error = message; }
template <class F> int guarded(F&& f) { try { return f(); } catch (const std::exception& e) { error = e.what(); return -1; } }

struct Tensor : colibri::v2::TensorDescriptor {
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
// first WeightProvider; safetensors and other providers can populate the same
// tensor contract without changing CUDA/session code.
struct ColibriV2Model : colibri::v2::WeightProvider { const uint8_t* data=nullptr; size_t size=0; uint32_t version=0, alignment=32; uint64_t metadata=0; std::string architecture, name, format_name="gguf"; colibri::v2::ModelConfig config; uint32_t mtp_layer=std::numeric_limits<uint32_t>::max(); std::vector<std::string> vocabulary, merges; std::unordered_map<std::string,int> merge_ranks; std::unordered_map<std::string,uint32_t> vocabulary_ids; std::vector<Tensor> tensors;
#if !defined(_WIN32)
    int fd=-1;
#else
    HANDLE file=nullptr, mapping=nullptr;
#endif
    const char* format() const override { return format_name.c_str(); }
    uint64_t tensor_count() const override { return tensors.size(); }
    const colibri::v2::TensorDescriptor* tensor(uint64_t index) const override { return index < tensors.size() ? &tensors[index] : nullptr; }
    int read_tensor(uint64_t index, void* destination, uint64_t bytes) const override { if(index >= tensors.size() || !destination || bytes < tensors[index].size) return -1; std::memcpy(destination, data + tensors[index].offset, tensors[index].size); return 0; }
};
struct ColibriV2Session { ColibriV2Model* model; uint64_t limit, prompt=0, decoded=0, calls=0; bool cancelled=false; std::vector<uint32_t> history; ColibriV2KvCache* kv_cache=nullptr; };
struct ColibriV2KvCache { std::uint64_t keys, values; std::int32_t capacity, kv_heads, head_dim, position=0; };

struct QwenLayerPlan {
    bool attention = false;
    std::uint32_t attention_window = 0; // 0 = global attention
    std::uint64_t cache_capacity = 0;
    std::uint32_t attention_heads = 0, kv_heads = 0, head_dim = 0;
    std::uint32_t rotary_dim = 0, expert_tensor_count = 3;
    float rope_theta = 0.0f;
    std::vector<std::uint64_t> static_tensors;
    std::array<std::uint64_t, 3> expert_tensors{};
    std::uint64_t state_first = 0;
    std::uint64_t state_second = 0;
    std::uint64_t snapshot_first = 0;
    std::uint64_t snapshot_second = 0;
};

struct QwenExpertSlot {
    std::uint64_t key = 0;
    std::uint64_t last_used = 0;
    bool valid = false;
};

struct QwenExpertHistory {
    std::uint32_t frequency = 0;
    std::uint64_t last_used = 0;
};

struct QwenCudaLayerProfile {
    std::uint64_t pre_start=0,pre_end=0;
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
    std::size_t slot = 0;
    int phase = 0;
    bool cancelled = false;
};

struct ColibriV2QwenRuntime {
    ColibriV2Model* model = nullptr;
    bool gemma4 = false;
    ColibriV2QwenRuntimeOptions options{};
    std::vector<QwenLayerPlan> layers;
    QwenLayerPlan mtp_layer_plan;
    std::array<std::uint64_t, 4> mtp_special_tensors{};
    bool mtp_available = false;
    std::uint64_t token_embeddings = 0;
    std::uint64_t final_norm = 0;
    std::uint64_t lm_head = 0;
    std::uint64_t rope_factors = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t static_tensor_bytes = 0;
    std::uint64_t expert_tensor_bytes = 0;
    std::uint64_t mtp_tensor_bytes = 0;
    std::uint32_t scratch_elements = 0;
    std::uint32_t moe_intermediate = 0;
    std::vector<std::uint64_t> device_tensors;
    std::uint64_t static_arena = 0;
    std::uint64_t static_arena_bytes = 0;
    std::uint64_t workspace = 0;
    std::uint64_t workspace_bytes = 0;
    std::uint64_t state = 0;
    std::uint64_t state_bytes = 0;
    std::uint64_t expert_staging = 0;
    std::uint64_t expert_staging_bytes = 0;
    std::uint64_t expert_cache = 0;
    std::uint64_t expert_cache_bytes = 0;
    std::uint64_t expert_slot_bytes = 0;
    std::vector<QwenExpertSlot> expert_slots;
    std::vector<QwenExpertHistory> expert_history;
    std::unordered_map<std::uint64_t, std::size_t> expert_residency;
    std::uint64_t expert_clock = 0;
    std::uint64_t expert_cache_hits = 0;
    std::uint64_t expert_cache_misses = 0;
    std::uint64_t expert_cache_evictions = 0;
    std::uint64_t expert_cache_admissions = 0;
    std::uint64_t expert_cache_rejections = 0;
    std::uint64_t expert_cache_prompt_bypasses = 0;
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
    std::uint64_t mtp_target_hidden_offset = 0;
    std::uint64_t mtp_draft_hidden_offset = 0;
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
    std::uint64_t paging_registration_nanoseconds = 0;
    std::uint64_t host_available_bytes = 0;
    void* host_staging = nullptr;
    std::uint64_t host_staging_bytes = 0;
    std::uint32_t forward_rows_capacity = 0;
    std::uint32_t prefill_rows = 0;
    std::vector<QwenPrefillSnapshot> prefill_snapshots{std::vector<QwenPrefillSnapshot>(2)};
    std::uint64_t prefill_snapshot_bytes = 0;
    std::uint32_t prefill_checkpoint_interval = 0; // tokens before the first mid-prefill checkpoint; 0 disables
    std::uint64_t prefill_snapshot_clock = 0;
    std::uint64_t stream = 0;
    std::uint64_t route_event = 0;
    bool cuda_profile = false;
    std::vector<QwenCudaLayerProfile> cuda_layer_profiles;
    std::uint64_t cuda_tail_start = 0, cuda_tail_end = 0;
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
    std::uint64_t decode_slice_bytes = 0;
    std::uint64_t decode_host_block_bytes = 0;
    std::uint32_t multi_decode_capacity = 1;
    bool cancelled = false;
    bool cache_admission_enabled = true;
    bool mtp_has_target_hidden = false;
    bool cuda_ready = false;
    bool decode_ready = false;
    bool dma_paging = false;       // expert page-ins go straight from the registered mmap
    bool model_registered = false; // whether we cuMemHostRegister'd model->data
};

std::size_t qwen_cache_layer_count(const ColibriV2QwenRuntime& runtime) {
    return runtime.layers.size() + (runtime.options.mtp_drafts ? 1U : 0U);
}

constexpr std::size_t kNoExpertSlot = std::numeric_limits<std::size_t>::max();

QwenExpertHistory& record_expert_access(
    ColibriV2QwenRuntime& runtime, std::uint32_t layer, std::uint32_t expert
) {
    const auto experts = runtime.model->config.expert_count;
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

std::size_t select_expert_cache_slot(
    ColibriV2QwenRuntime& runtime, std::uint32_t layer,
    std::uint32_t expert, bool allow_rejection
) {
    if (!runtime.cache_admission_enabled) {
        // Prompt prefill must not churn the device cache, but its routes are
        // valuable training data for a later bulk seed of the hottest experts.
        record_expert_access(runtime, layer, expert);
        ++runtime.expert_cache_prompt_bypasses;
        return kNoExpertSlot;
    }
    auto& candidate = record_expert_access(runtime, layer, expert);
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
        const auto left_expert = static_cast<std::uint32_t>(left.key);
        const auto right_expert = static_cast<std::uint32_t>(right.key);
        const auto& left_history = runtime.expert_history[static_cast<std::size_t>(layer) * runtime.model->config.expert_count + left_expert];
        const auto& right_history = runtime.expert_history[static_cast<std::size_t>(layer) * runtime.model->config.expert_count + right_expert];
        return left_history.frequency != right_history.frequency
            ? left_history.frequency < right_history.frequency
            : left.last_used < right.last_used;
    });
    const auto victim_expert = static_cast<std::uint32_t>(victim->key);
    const auto& victim_history = runtime.expert_history[
        static_cast<std::size_t>(layer) * runtime.model->config.expert_count + victim_expert
    ];
    // Replacing a multi-megabyte expert for a one-hit frequency advantage
    // creates transfer spikes and immediate churn. Require a small lead so a
    // candidate has demonstrated reuse before displacing a resident expert.
    if (allow_rejection &&
        static_cast<std::uint64_t>(candidate.frequency) <=
            static_cast<std::uint64_t>(victim_history.frequency) + 1) {
        ++runtime.expert_cache_rejections;
        return kNoExpertSlot;
    }
    const auto slot = static_cast<std::size_t>(victim - runtime.expert_slots.begin());
    runtime.expert_residency.erase(victim->key);
    ++runtime.expert_cache_evictions;
    ++runtime.expert_cache_admissions;
    return slot;
}

constexpr std::uint64_t kDeviceAlignment = 256;

std::uint64_t device_align(std::uint64_t bytes) {
    return (bytes + kDeviceAlignment - 1) / kDeviceAlignment * kDeviceAlignment;
}

void release_qwen_device(ColibriV2QwenRuntime& runtime) {
    if (runtime.stream) colibri_gpu_stream_sync(runtime.stream);
    if (runtime.model_registered && runtime.model) {
        colibri_gpu_host_unregister(runtime.model->data);
        runtime.model_registered = false;
        runtime.dma_paging = false;
    }
    colibri_gpu_host_free(runtime.host_staging);
    // The active slot's checkpoint pool is mirrored in runtime.prefill_snapshots;
    // inactive slots keep theirs in sequences[i]. Each buffer lives in exactly
    // one of the two, so freeing both frees every buffer once.
    for (auto& snapshot : runtime.prefill_snapshots) {
        colibri_gpu_free(snapshot.device);
        snapshot = QwenPrefillSnapshot{};
    }
    runtime.prefill_snapshot_bytes = 0;
    colibri_gpu_free(runtime.expert_cache);
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
    for (auto& e : runtime.host_prompts) std::free(e.state);
    runtime.host_prompts.clear();
    runtime.host_cache_used_bytes = 0;
    colibri_gpu_free(runtime.workspace);
    colibri_gpu_free(runtime.static_arena);
    for (auto& event : runtime.slot_events) colibri_gpu_event_destroy(event);
    runtime.slot_events.clear();
    colibri_gpu_event_destroy(runtime.staging_event);
    runtime.staging_event = 0;
    colibri_gpu_event_destroy(runtime.route_event);
    for(auto&profile:runtime.cuda_layer_profiles){
        colibri_gpu_event_destroy(profile.pre_start);
        colibri_gpu_event_destroy(profile.pre_end);
        colibri_gpu_event_destroy(profile.shared_start);
        colibri_gpu_event_destroy(profile.shared_end);
        colibri_gpu_event_destroy(profile.expert_start);
        colibri_gpu_event_destroy(profile.expert_end);
    }
    runtime.cuda_layer_profiles.clear();
    colibri_gpu_event_destroy(runtime.cuda_tail_start);
    colibri_gpu_event_destroy(runtime.cuda_tail_end);
    runtime.cuda_tail_start=runtime.cuda_tail_end=0;
    runtime.cuda_profile=false;
    colibri_gpu_stream_destroy(runtime.stream);
    runtime.device_tensors.clear();
    runtime.host_staging = nullptr;
    runtime.expert_staging = runtime.state = runtime.workspace = 0;
    runtime.expert_cache = 0;
    runtime.static_arena = runtime.stream = 0;
    runtime.route_event = 0;
    runtime.static_arena_bytes = runtime.workspace_bytes = 0;
    runtime.state_bytes = runtime.expert_staging_bytes = 0;
    runtime.host_staging_bytes = 0;
    runtime.forward_rows_capacity = 0;
    runtime.expert_cache_bytes = runtime.expert_slot_bytes = 0;
    runtime.expert_slots.clear();
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

int parse(ColibriV2Model& m) {
    if (m.size < 24 || std::memcmp(m.data,"GGUF",4)!=0) throw std::runtime_error("not a GGUF file");
    Reader r{m.data+4,m.data+m.size}; m.version=r.get<uint32_t>(); if (m.version<2 || m.version>3) throw std::runtime_error("unsupported GGUF version");
    uint64_t count=r.get<uint64_t>(); m.metadata=r.get<uint64_t>();
    auto read_uint = [&](uint32_t type)->uint64_t { if(type==4)return r.get<uint32_t>(); if(type==10)return r.get<uint64_t>(); throw std::runtime_error("GGUF architecture value is not an integer"); };
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
    auto set_config = [&](const std::string& key, uint32_t type)->bool {
        if(type!=4 && type!=10)return false;
        auto suffix=[&](const char* text){size_t n=std::strlen(text);return key.size()>=n && key.compare(key.size()-n,n,text)==0;};
        uint32_t* target=nullptr;
        if(suffix(".embedding_length"))target=&m.config.hidden_size;
        else if(suffix(".embedding_length_per_layer_input"))target=&m.config.per_layer_embedding_size;
        else if(suffix(".block_count"))target=&m.config.layer_count;
        else if(suffix(".attention.shared_kv_layers"))target=&m.config.shared_kv_layers;
        else if(suffix(".attention.head_count_kv"))target=&m.config.attention_kv_heads;
        else if(suffix(".attention.head_count"))target=&m.config.attention_heads;
        else if(suffix(".context_length"))target=&m.config.context_length;
        else if(suffix(".expert_feed_forward_length"))target=&m.config.expert_intermediate_size;
        else if(suffix(".feed_forward_length"))target=&m.config.dense_intermediate_size;
        else if(suffix(".expert_count"))target=&m.config.expert_count;
        else if(suffix(".expert_used_count"))target=&m.config.expert_used_count;
        else if(key=="tokenizer.ggml.vocab_size")target=&m.config.vocabulary_size;
        if(!target)return false;
        *target=static_cast<uint32_t>(read_uint(type));
        return true;
    };
    for(uint64_t i=0;i<m.metadata;i++) { std::string key=r.str(); uint32_t type=r.get<uint32_t>();
        if (key=="general.alignment" && type==4) m.alignment=r.get<uint32_t>();
        else if (key=="general.architecture" && type==8) {m.architecture=r.str();m.config.architecture=m.architecture;}
        else if (key=="general.name" && type==8) m.name=r.str();
        else if (key=="tokenizer.ggml.tokens" && type==9) {uint32_t element_type=r.get<uint32_t>();uint64_t count_tokens=r.get<uint64_t>();m.config.vocabulary_size=static_cast<uint32_t>(count_tokens);m.vocabulary.reserve(static_cast<size_t>(count_tokens));for(uint64_t token_index=0;token_index<count_tokens;token_index++){if(element_type==8)m.vocabulary.push_back(r.str());else r.value(element_type);}}
        else if (key=="tokenizer.ggml.merges" && type==9) {uint32_t element_type=r.get<uint32_t>();uint64_t count_merges=r.get<uint64_t>();m.merges.reserve(static_cast<size_t>(count_merges));for(uint64_t merge_index=0;merge_index<count_merges;merge_index++){if(element_type==8)m.merges.push_back(r.str());else r.value(element_type);}}
        else if (key.size()>=21 && key.compare(key.size()-21,21,".rope.dimension_count")==0 && (type==4 || type==10)) m.config.rotary_dimension=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=24 && key.compare(key.size()-24,24,".full_attention_interval")==0 && (type==4 || type==10)) m.config.full_attention_interval=static_cast<uint32_t>(read_uint(type));
        else if (key.size()>=24 && key.compare(key.size()-24,24,".attention.head_count_kv")==0 && type==9) m.config.attention_kv_heads_by_layer=read_uint_array(type);
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
        else if (set_config(key,type)) {}
        else r.value(type);
    }
    m.tensors.reserve(static_cast<size_t>(count));
    for(uint64_t i=0;i<count;i++) { Tensor t; t.name=r.str(); auto dims=r.get<uint32_t>(); if(dims>4) throw std::runtime_error("GGUF rank exceeds v2 ABI"); for(uint32_t d=0;d<dims;d++) t.shape.push_back(r.get<uint64_t>()); t.type=r.get<uint32_t>(); t.offset=r.get<uint64_t>(); m.tensors.push_back(std::move(t)); }
    // Qwen3-Next GGUF files include the optional MTP draft block in
    // ``block_count``.  A block carrying ``nextn`` tensors is not part of the
    // causal decoder stack and must not be executed before the final norm.
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
                 "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
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
    const auto&shared_gate=model.tensors[runtime.layers.front().static_tensors[runtime.layers.front().attention?9:12]];
    runtime.moe_intermediate=static_cast<std::uint32_t>(shared_gate.shape[1]);
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

float qwen_half_value(std::uint16_t bits) {
    const std::uint32_t sign=(bits&0x8000u)<<16;
    std::uint32_t exponent=(bits>>10)&0x1fu,fraction=bits&0x3ffu,result=0;
    if(exponent==0){if(!fraction)result=sign;else{exponent=1;while((fraction&0x400u)==0){fraction<<=1;--exponent;}result=sign|((exponent+112)<<23)|((fraction&0x3ffu)<<13);}}
    else if(exponent==31)result=sign|0x7f800000u|(fraction<<13);
    else result=sign|((exponent+112)<<23)|(fraction<<13);
    float value;std::memcpy(&value,&result,sizeof(value));return value;
}

float qwen_q5_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/256;const int within=static_cast<int>(absolute&255);const auto*base=packed+block*176;std::uint16_t d_bits=0,dmin_bits=0;std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);const auto*scales=base+4;const int group=within/64,offset=within&63,sub=offset/32,qindex=group*32+(offset&31);const int bit=(base[16+(offset&31)]>>(2*group+sub))&1;const int quant=((offset<32)?(base[48+qindex]&15):(base[48+qindex]>>4))+16*bit;const int index=group*2+sub;int scale=0,minimum=0;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}return qwen_half_value(d_bits)*scale*quant-qwen_half_value(dmin_bits)*minimum;}
float qwen_q6_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/256;const int within=static_cast<int>(absolute&255);const auto*base=packed+block*210;const auto*ql=base;const auto*qh=base+128;const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);std::uint16_t d_bits=0;std::memcpy(&d_bits,base+208,2);const int half=within/128,offset=within&127,lane=offset/32,l=offset&31,qindex=l+((lane==0||lane==2)?0:32);const auto qbyte=ql[half*64+qindex],high=qh[half*32+l];const int nibble=(lane==0||lane==1)?(qbyte&15):(qbyte>>4);const int quant=(nibble|(((high>>(lane*2))&3)<<4))-32;const int scale_index=half*8+(l/16)+lane*2;return qwen_half_value(d_bits)*scales[scale_index]*quant;}
float qwen_q8_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/32,within=absolute&31;std::uint16_t scale_bits=0;std::memcpy(&scale_bits,packed+block*34,2);std::int8_t value=0;std::memcpy(&value,packed+block*34+2+within,1);return qwen_half_value(scale_bits)*value;}
float qwen_quant_dot(const std::uint8_t*packed,std::uint32_t type,const float*input,int elements,std::uint64_t row){
    if(type!=2&&(colibri_cpu_features()&2u)!=0&&elements%256==0)return qwen_quant_dot_avx512(packed,type,input,elements,row);
    if(type!=2&&(colibri_cpu_features()&1u)!=0&&elements%256==0)return qwen_quant_dot_avx2(packed,type,input,elements,row);
    float result=0.0f;
    if(type==2){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/32)*18;
        for(int block=0;block<elements/32;++block){const auto*base=row_data+block*18;std::uint16_t scale_bits=0;std::memcpy(&scale_bits,base,2);const float scale=qwen_half_value(scale_bits);const auto*vector=input+block*32;for(int lane=0;lane<16;++lane){const auto byte=base[2+lane];result+=scale*((byte&15)-8)*vector[lane];result+=scale*((byte>>4)-8)*vector[lane+16];}}
    }else if(type==13){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*176;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*176;std::uint16_t d_bits=0,dmin_bits=0;
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
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*210;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*210;const auto*ql=base;const auto*qh=base+128;
            const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);std::uint16_t d_bits=0;
            std::memcpy(&d_bits,base+208,2);const float d=qwen_half_value(d_bits);const auto*vector=input+block*256;
            for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){
                const int q_offset=(segment==0||segment==2)?0:32;const int shift=segment*2;
                const auto*values=vector+half*128+segment*32;
                for(int lane=0;lane<32;++lane){const auto qbyte=ql[half*64+q_offset+lane];const int nibble=(segment<2)?(qbyte&15):(qbyte>>4);const int quant=(nibble|(((qh[half*32+lane]>>shift)&3)<<4))-32;const int scale_index=half*8+(lane/16)+segment*2;result+=d*scales[scale_index]*quant*values[lane];}
            }
        }
    }else if(type==8){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/32)*34;
        for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;std::uint16_t scale_bits=0;std::memcpy(&scale_bits,base,2);const float scale=qwen_half_value(scale_bits);const auto*values=reinterpret_cast<const std::int8_t*>(base+2);const auto*vector=input+block*32;for(int lane=0;lane<32;++lane)result+=scale*values[lane]*vector[lane];}
    }else throw std::runtime_error("unsupported native CPU expert quantization");
    return result;
}

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
    const auto*expert_scales=reinterpret_cast<const float*>(runtime.model->data+scale_tensor.offset);
    #pragma omp parallel for schedule(static)
    for(int task=0;task<routed_count*intermediate;++task){
        const int rank=task/intermediate,row=task%intermediate,expert=selected[rank];
        if(expert<0||expert>=experts)continue;
        const auto*gate_up=runtime.model->data+gate_up_tensor.offset+static_cast<std::uint64_t>(expert)*gate_up_bytes;
        const float gate=qwen_quant_dot(gate_up,2,input,hidden,row);
        const float up=qwen_quant_dot(gate_up,2,input,hidden,row+intermediate);
        const float cubic=gate*gate*gate;
        const float gelu=0.5f*gate*(1.0f+std::tanh(0.7978845608028654f*(gate+0.044715f*cubic)));
        activated[task]=gelu*up;
    }
    #pragma omp parallel for schedule(static)
    for(int row=0;row<hidden;++row){
        float sum=0.0f;
        for(int rank=0;rank<routed_count;++rank){const int expert=selected[rank];if(expert<0||expert>=experts)continue;const auto*down=runtime.model->data+down_tensor.offset+static_cast<std::uint64_t>(expert)*down_bytes;sum+=weights[rank]*expert_scales[expert]*qwen_quant_dot(down,2,activated+rank*intermediate,intermediate,row);}
        output[row]=sum;
    }
}

// Dequantize one weight row to f32 so it can be reused across every token
// routed to the same expert within a batch: the quantized bytes are decoded
// once per batch instead of once per token.
void qwen_dequant_row(const std::uint8_t*packed,std::uint32_t type,int elements,std::uint64_t row,float*output){
    if((colibri_cpu_features()&2u)!=0&&elements%256==0){qwen_dequant_row_avx512(packed,type,elements,row,output);return;}
    if((colibri_cpu_features()&1u)!=0&&elements%256==0){qwen_dequant_row_avx2(packed,type,elements,row,output);return;}
    const auto base=row*static_cast<std::uint64_t>(elements);
    if(type==13)for(int index=0;index<elements;++index)output[index]=qwen_q5_value(packed,base+index);
    else if(type==14)for(int index=0;index<elements;++index)output[index]=qwen_q6_value(packed,base+index);
    else if(type==8)for(int index=0;index<elements;++index)output[index]=qwen_q8_value(packed,base+index);
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

void qwen_cpu_moe(const ColibriV2QwenRuntime&runtime,const QwenLayerPlan&layer,const std::int32_t*selected,const float*weights,int routed_count,const float*input,float*activated,float*output){const int experts=runtime.model->config.expert_count,hidden=runtime.model->config.hidden_size,intermediate=runtime.moe_intermediate;if(routed_count<0||routed_count>256)throw std::runtime_error("native CPU MoE routed count is unsupported");for(int role=0;role<3;++role){const auto type=runtime.model->tensors[layer.expert_tensors[role]].type;if(type!=13&&type!=14&&type!=8)throw std::runtime_error("unsupported native CPU expert quantization");}std::array<const std::uint8_t*,256>gate{},up{},down{};for(int rank=0;rank<routed_count;++rank){if(selected[rank]<0||selected[rank]>=experts)throw std::runtime_error("native CPU MoE selected an invalid expert");for(int role=0;role<3;++role){const auto&t=runtime.model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto*pointer=runtime.model->data+t.offset+static_cast<std::uint64_t>(selected[rank])*bytes;if(role==0)gate[rank]=pointer;else if(role==1)up[rank]=pointer;else down[rank]=pointer;}}
const char*q8_setting=std::getenv("COLIBRI_Q8_ACTIVATIONS");
const bool use_q8=(colibri_cpu_features()&1u)!=0&&hidden%256==0&&intermediate%256==0&&!(q8_setting&&q8_setting[0]=='0');
thread_local std::vector<QwenQ8KBlock> input_q8,activated_q8;
if(use_q8){input_q8.resize(hidden/256);activated_q8.resize(static_cast<std::size_t>(routed_count)*(intermediate/256));qwen_quantize_q8_k_avx2(input,hidden,input_q8.data());}
#pragma omp parallel for schedule(static)
for(int task=0;task<routed_count*intermediate;++task){const int rank=task/intermediate,row=task%intermediate;const float gate_value=use_q8?qwen_quant_dot_q8_k_avx2(gate[rank],runtime.model->tensors[layer.expert_tensors[0]].type,input_q8.data(),hidden,row):qwen_quant_dot(gate[rank],runtime.model->tensors[layer.expert_tensors[0]].type,input,hidden,row);const float up_value=use_q8?qwen_quant_dot_q8_k_avx2(up[rank],runtime.model->tensors[layer.expert_tensors[1]].type,input_q8.data(),hidden,row):qwen_quant_dot(up[rank],runtime.model->tensors[layer.expert_tensors[1]].type,input,hidden,row);const float clipped=std::max(-80.0f,std::min(80.0f,gate_value));activated[task]=gate_value/(1.0f+std::exp(-clipped))*up_value;}
if(use_q8)for(int rank=0;rank<routed_count;++rank)qwen_quantize_q8_k_avx2(activated+rank*intermediate,intermediate,activated_q8.data()+static_cast<std::size_t>(rank)*(intermediate/256));
#pragma omp parallel for schedule(static)
for(int row=0;row<hidden;++row){float value=0.0f;for(int rank=0;rank<routed_count;++rank)value+=weights[rank]*(use_q8?qwen_quant_dot_q8_k_avx2(down[rank],runtime.model->tensors[layer.expert_tensors[2]].type,activated_q8.data()+static_cast<std::size_t>(rank)*(intermediate/256),intermediate,row):qwen_quant_dot(down[rank],runtime.model->tensors[layer.expert_tensors[2]].type,activated+rank*intermediate,intermediate,row));output[row]=value;}
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
    if(rows<=0||rows>4096||routed_count<=0||routed_count>256)
        throw std::runtime_error("native CPU batched MoE shape is unsupported");
    const auto gate_type=runtime.model->tensors[layer.expert_tensors[0]].type;
    const auto up_type=runtime.model->tensors[layer.expert_tensors[1]].type;
    const auto down_type=runtime.model->tensors[layer.expert_tensors[2]].type;
    for(const auto type:{gate_type,up_type,down_type})
        if(type!=13&&type!=14&&type!=8)
            throw std::runtime_error("unsupported native CPU expert quantization");
    auto expert_data=[&](int role,int expert){
        const auto&t=runtime.model->tensors[layer.expert_tensors[role]];
        return runtime.model->data+t.offset+static_cast<std::uint64_t>(expert)*(t.size/experts);
    };
    // Group routes by expert (CSR over token_rank slots) so each expert's
    // weight rows are decoded once per batch and dotted with every token
    // routed to it. Routes with weight 0 were already claimed by the GPU
    // expert cache (or pruned) and are skipped everywhere below.
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
    // Process weight rows in blocks of 4 so the register-blocked GEMM reuses
    // each activation load across 4 output rows (see qwen_f32_gemm_rows_avx512).
    constexpr int kRowBlock=4;
    const int gate_blocks=(intermediate+kRowBlock-1)/kRowBlock;
#pragma omp parallel for schedule(dynamic,4)
    for(int task=0;task<group_count*gate_blocks;++task){
        const int group=task/gate_blocks;const int row0=(task%gate_blocks)*kRowBlock;
        const int mr=std::min(kRowBlock,intermediate-row0);
        const int expert=group_experts[group];
        const int begin=offsets[expert],count=counts[expert];
        const auto*gate_data=expert_data(0,expert);
        const auto*up_data=expert_data(1,expert);
        thread_local std::vector<float> gate_block,up_block,gate_values,up_values;
        gate_block.resize(static_cast<std::size_t>(kRowBlock)*hidden);up_block.resize(static_cast<std::size_t>(kRowBlock)*hidden);
        gate_values.resize(static_cast<std::size_t>(kRowBlock)*count);up_values.resize(static_cast<std::size_t>(kRowBlock)*count);
        for(int i=0;i<mr;++i){
            qwen_dequant_row(gate_data,gate_type,hidden,row0+i,gate_block.data()+static_cast<std::size_t>(i)*hidden);
            qwen_dequant_row(up_data,up_type,hidden,row0+i,up_block.data()+static_cast<std::size_t>(i)*hidden);
        }
        qwen_f32_gemm_rows(gate_block.data(),mr,&vectors[begin],count,hidden,gate_values.data());
        qwen_f32_gemm_rows(up_block.data(),mr,&vectors[begin],count,hidden,up_values.data());
        for(int i=0;i<mr;++i)for(int occurrence=0;occurrence<count;++occurrence){
            const int token_rank=occurrences[begin+occurrence];
            const float gate_value=gate_values[static_cast<std::size_t>(i)*count+occurrence];
            const float clipped=std::max(-80.0f,std::min(80.0f,gate_value));
            activated[static_cast<std::size_t>(token_rank)*intermediate+(row0+i)]=gate_value/(1.0f+std::exp(-clipped))*up_values[static_cast<std::size_t>(i)*count+occurrence];
        }
    }
    std::vector<const float*> activated_vectors(offsets[experts]);
    for(int slot=0;slot<offsets[experts];++slot)
        activated_vectors[slot]=activated+static_cast<std::size_t>(occurrences[slot])*intermediate;
    const int down_blocks=(hidden+kRowBlock-1)/kRowBlock;
#pragma omp parallel for schedule(dynamic,4)
    for(int task=0;task<group_count*down_blocks;++task){
        const int group=task/down_blocks;const int row0=(task%down_blocks)*kRowBlock;
        const int mr=std::min(kRowBlock,hidden-row0);
        const int expert=group_experts[group];
        const int begin=offsets[expert],count=counts[expert];
        const auto*down_data=expert_data(2,expert);
        thread_local std::vector<float> down_block,values;
        down_block.resize(static_cast<std::size_t>(kRowBlock)*intermediate);values.resize(static_cast<std::size_t>(kRowBlock)*count);
        for(int i=0;i<mr;++i)qwen_dequant_row(down_data,down_type,intermediate,row0+i,down_block.data()+static_cast<std::size_t>(i)*intermediate);
        qwen_f32_gemm_rows(down_block.data(),mr,&activated_vectors[begin],count,intermediate,values.data());
        for(int i=0;i<mr;++i)for(int occurrence=0;occurrence<count;++occurrence){
            const int token_rank=occurrences[begin+occurrence];
            down_values[static_cast<std::size_t>(token_rank)*hidden+(row0+i)]=weights[token_rank]*values[static_cast<std::size_t>(i)*count+occurrence];
        }
    }
#pragma omp parallel for schedule(static)
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
}

int gpu_probe(ColibriV2GpuInfo& out, int device) {
    std::memset(&out, 0, sizeof(out)); out.device = device;
#if defined(_WIN32)
    (void)device; return 0;
#else
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL); if (!lib) lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL); if (!lib) return 0;
    using Init = int (*)(unsigned int); using Retain = int (*)(void**, int); using Set = int (*)(void*); using Attr = int (*)(int*, int, int); using Mem = int (*)(size_t*, size_t*);
    auto init=reinterpret_cast<Init>(dlsym(lib,"cuInit")); auto retain=reinterpret_cast<Retain>(dlsym(lib,"cuDevicePrimaryCtxRetain")); auto set=reinterpret_cast<Set>(dlsym(lib,"cuCtxSetCurrent")); auto attr=reinterpret_cast<Attr>(dlsym(lib,"cuDeviceGetAttribute")); auto mem=reinterpret_cast<Mem>(dlsym(lib,"cuMemGetInfo_v2"));
    if (!init || !retain || !set || !attr || init(0)!=0) { dlclose(lib); return 0; } void* context=nullptr; if(retain(&context,device)!=0 || set(context)!=0) { dlclose(lib); return 0; }
    int major=0,minor=0; if(attr(&major,75,device)!=0 || attr(&minor,76,device)!=0) { dlclose(lib); return 0; } out.available=1; out.compute_major=major; out.compute_minor=minor; if(mem) { size_t free_bytes=0,total_bytes=0; if(mem(&free_bytes,&total_bytes)==0) { out.free_memory=free_bytes; out.total_memory=total_bytes; } } dlclose(lib); return 0;
#endif
}

int plan_memory(ColibriV2MemoryPlan& out, uint64_t budget, uint64_t static_weights, uint64_t kv_state, uint64_t workspace, uint64_t active, uint64_t staging) {
    if (static_weights > budget || kv_state > budget-static_weights || workspace > budget-static_weights-kv_state) { fail("persistent v2 GPU allocations exceed budget"); return -1; }
    out={}; out.budget=budget; out.static_weights=static_weights; out.kv_state=kv_state; out.workspace=workspace; uint64_t left=budget-static_weights-kv_state-workspace; out.active_experts=std::min(active,left); left-=out.active_experts; out.staging=std::min(staging,left); out.unused=left-out.staging; return 0;
}

extern "C" {
uint32_t colibri_v2_version() { return 3; }
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
void colibri_v2_kv_cache_destroy(ColibriV2KvCache*cache){delete cache;}
int colibri_v2_kv_cache_reset(ColibriV2KvCache*cache){if(!cache){fail("invalid KV cache");return -1;}cache->position=0;return 0;}
int colibri_v2_kv_cache_position(const ColibriV2KvCache*cache,int32_t*out){if(!cache||!out){fail("invalid KV cache position");return -1;}*out=cache->position;return 0;}
int colibri_v2_gpu_decoder_attention_cached(ColibriV2KvCache*cache,uint64_t input,uint64_t norm_weights,uint64_t normalized,uint64_t qkv_packed,uint64_t qkv_scales,uint64_t qkv,uint64_t attention_output,uint64_t out_packed,uint64_t out_scales,uint64_t output,int32_t hidden_size,int32_t heads,float epsilon,int32_t one_centered){if(!cache){fail("invalid KV cache");return -1;}int status=colibri_v2_gpu_decoder_attention_step(input,norm_weights,normalized,qkv_packed,qkv_scales,qkv,cache->keys,cache->values,attention_output,out_packed,out_scales,output,hidden_size,heads,cache->kv_heads,cache->head_dim,cache->position,cache->capacity,epsilon,one_centered);if(status)return status;++cache->position;return 0;}
int colibri_v2_model_open(const char* path, ColibriV2Model** out) { return guarded([&]{ if(!path||!out) throw std::runtime_error("path and output are required"); auto* m=new ColibriV2Model;
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
    if(lock_model&&mlock(m->data,m->size)!=0) std::fprintf(stderr,"colibri_v2: mlock(%zu bytes) failed: %s; continuing without pinning (raise RLIMIT_MEMLOCK to pin)\n",m->size,std::strerror(errno));
#else
    (void)path; throw std::runtime_error("Windows v2 mapping is not implemented");
#endif
    parse(*m); *out=m; return 0; }); }
void colibri_v2_model_close(ColibriV2Model* m) { if(!m)return;
#if !defined(_WIN32)
    if(m->data) munmap(const_cast<uint8_t*>(m->data),m->size); if(m->fd>=0) close(m->fd);
#endif
    delete m; }
int colibri_v2_model_info(const ColibriV2Model* m, ColibriV2ModelInfo* out) { return guarded([&]{if(!m||!out)throw std::runtime_error("invalid model info handle"); std::memset(out,0,sizeof(*out));out->gguf_version=m->version;out->tensor_count=m->tensor_count();out->metadata_count=m->metadata;out->file_size=m->size;out->alignment=m->alignment;copy_text(out->architecture,sizeof(out->architecture),m->architecture);copy_text(out->name,sizeof(out->name),m->name);copy_text(out->format,sizeof(out->format),m->format());return 0;}); }
int colibri_v2_model_config(const ColibriV2Model* m, ColibriV2ModelConfig* out){return guarded([&]{if(!m||!out)throw std::runtime_error("invalid model config handle");std::memset(out,0,sizeof(*out));copy_text(out->architecture,sizeof(out->architecture),m->config.architecture);out->hidden_size=m->config.hidden_size;out->layer_count=m->config.layer_count;out->attention_heads=m->config.attention_heads;out->attention_kv_heads=m->config.attention_kv_heads;out->context_length=m->config.context_length;out->intermediate_size=m->config.intermediate_size;out->expert_count=m->config.expert_count;out->expert_used_count=m->config.expert_used_count;out->vocabulary_size=m->config.vocabulary_size;out->rotary_dimension=m->config.rotary_dimension;out->full_attention_interval=m->config.full_attention_interval;out->sliding_window=m->config.sliding_window;out->sliding_window_pattern_length=static_cast<uint32_t>(m->config.sliding_window_pattern.size());out->rms_norm_epsilon=m->config.rms_norm_epsilon;out->rope_freq_base=m->config.rope_freq_base;return 0;});}
int colibri_v2_model_attention_window(const ColibriV2Model* m,uint32_t layer,uint32_t*out){return guarded([&]{if(!m||!out)throw std::runtime_error("invalid model attention-window handle");*out=attention_window(*m,layer);return 0;});}
int colibri_v2_tensor_info(const ColibriV2Model* m,uint64_t i,ColibriV2TensorInfo* out){return guarded([&]{if(!m||!out||i>=m->tensors.size())throw std::runtime_error("tensor index out of range");return fill(m->tensors[i],*out);});}
int colibri_v2_tensor_find(const ColibriV2Model* m,const char* name,ColibriV2TensorInfo* out){return guarded([&]{if(!m||!name||!out)throw std::runtime_error("invalid tensor lookup");for(auto const&t:m->tensors)if(t.name==name)return fill(t,*out);throw std::runtime_error("tensor not found");});}
int colibri_v2_qwen_validate(const ColibriV2Model*m){return guarded([&]{if(!m)throw std::runtime_error("invalid model handle");if(m->config.architecture.find("qwen")!=0)throw std::runtime_error("model architecture is not Qwen");if(!m->config.hidden_size||!m->config.layer_count||!m->config.attention_heads)throw std::runtime_error("Qwen config is incomplete");return 0;});}
int colibri_v2_qwen_tensor_role(const ColibriV2Model*m,const char*role,ColibriV2TensorInfo*out){return guarded([&]{if(!m||!role||!out)throw std::runtime_error("invalid Qwen tensor role lookup");std::vector<std::string> candidates;if(std::strcmp(role,"token_embeddings")==0)candidates={"token_embd.weight","model.embed_tokens.weight","embed_tokens.weight"};else if(std::strcmp(role,"final_norm")==0)candidates={"output_norm.weight","model.norm.weight","norm.weight"};else if(std::strcmp(role,"lm_head")==0)candidates={"output.weight","lm_head.weight"};else throw std::runtime_error("unknown Qwen tensor role");for(auto const&candidate:candidates)for(auto const&t:m->tensors)if(t.name==candidate)return fill(t,*out);throw std::runtime_error("Qwen tensor role is missing");});}
int colibri_v2_qwen_layer_tensor(const ColibriV2Model*m,uint32_t layer,const char*role,ColibriV2TensorInfo*out){return guarded([&]{if(!m||!role||!out)throw std::runtime_error("invalid Qwen layer tensor lookup");std::string prefix="blk."+std::to_string(layer)+".";std::vector<std::string> suffixes;if(std::strcmp(role,"input_norm")==0)suffixes={"attn_norm.weight"};else if(std::strcmp(role,"qkv")==0)suffixes={"attn_qkv.weight"};else if(std::strcmp(role,"attention_q")==0)suffixes={"attn_q.weight"};else if(std::strcmp(role,"attention_k")==0)suffixes={"attn_k.weight"};else if(std::strcmp(role,"attention_v")==0)suffixes={"attn_v.weight"};else if(std::strcmp(role,"attention_output")==0)suffixes={"attn_output.weight","attn_out.weight"};else if(std::strcmp(role,"attention_gate")==0)suffixes={"attn_gate.weight"};else if(std::strcmp(role,"ssm_output")==0)suffixes={"ssm_out.weight"};else if(std::strcmp(role,"ssm_alpha")==0)suffixes={"ssm_alpha.weight"};else if(std::strcmp(role,"ssm_beta")==0)suffixes={"ssm_beta.weight"};else if(std::strcmp(role,"ssm_conv")==0)suffixes={"ssm_conv1d.weight"};else if(std::strcmp(role,"ssm_dt_bias")==0)suffixes={"ssm_dt.bias"};else if(std::strcmp(role,"ssm_a")==0)suffixes={"ssm_a"};else if(std::strcmp(role,"ssm_norm")==0)suffixes={"ssm_norm.weight"};else if(std::strcmp(role,"post_attention_norm")==0)suffixes={"post_attention_norm.weight"};else if(std::strcmp(role,"router")==0)suffixes={"ffn_gate_inp.weight"};else if(std::strcmp(role,"shared_gate")==0)suffixes={"ffn_gate_shexp.weight"};else throw std::runtime_error("unknown Qwen layer tensor role");for(auto const&suffix:suffixes)for(auto const&t:m->tensors)if(t.name==prefix+suffix)return fill(t,*out);throw std::runtime_error("Qwen layer tensor role is missing");});}
float half_to_float(uint16_t bits){uint32_t sign=(bits&0x8000u)<<16, exponent=(bits>>10)&0x1fu, fraction=bits&0x3ffu;uint32_t result;if(exponent==0){if(!fraction)result=sign;else{exponent=1;while((fraction&0x400u)==0){fraction<<=1;--exponent;}result=sign|((exponent+112)<<23)|((fraction&0x3ffu)<<13);}}else if(exponent==31)result=sign|0x7f800000u|(fraction<<13);else result=sign|((exponent+112)<<23)|(fraction<<13);float value;std::memcpy(&value,&result,sizeof(value));return value;}
float tensor_value(const uint8_t*data,uint32_t type,uint64_t index){if(type==0){float value;std::memcpy(&value,data+index*4,4);return value;}if(type==1){uint16_t value;std::memcpy(&value,data+index*2,2);return half_to_float(value);}if(type==30){uint16_t value;std::memcpy(&value,data+index*2,2);uint32_t bits=static_cast<uint32_t>(value)<<16;float result;std::memcpy(&result,&bits,4);return result;}if(type==8){uint64_t block=index/32,within=index%32;uint16_t scale;std::memcpy(&scale,data+block*34,2);int8_t quant;std::memcpy(&quant,data+block*34+2+within,1);return half_to_float(scale)*static_cast<float>(quant);}throw std::runtime_error("unsupported Qwen CPU tensor type");}
const Tensor& qwen_role_tensor(const ColibriV2Model&m,const char*role){std::vector<std::string> candidates;if(std::strcmp(role,"token_embeddings")==0)candidates={"token_embd.weight","model.embed_tokens.weight","embed_tokens.weight"};else if(std::strcmp(role,"lm_head")==0)candidates={"output.weight","lm_head.weight"};else throw std::runtime_error("unknown Qwen tensor role");for(auto const&candidate:candidates)for(auto const&t:m.tensors)if(t.name==candidate)return t;throw std::runtime_error("Qwen tensor role is missing");}
int colibri_v2_qwen_embedding(const ColibriV2Model*m,uint32_t token,float*out,uint64_t elements){return guarded([&]{if(!m||!out)throw std::runtime_error("invalid embedding arguments");const Tensor&t=qwen_role_tensor(*m,"token_embeddings");if(t.shape.size()!=2)throw std::runtime_error("embedding shape is invalid");uint64_t width=t.shape[0]==m->config.hidden_size?t.shape[0]:t.shape[1],vocab=t.shape[0]==m->config.hidden_size?t.shape[1]:t.shape[0];if(token>=vocab||elements<width)throw std::runtime_error("embedding token or buffer is invalid");for(uint64_t i=0;i<width;i++)out[i]=tensor_value(m->data+t.offset,t.type,static_cast<uint64_t>(token)*width+i);return 0;});}
int colibri_v2_qwen_lm_head(const ColibriV2Model*m,const float*hidden,float*logits,uint64_t vocabulary,uint64_t elements){return guarded([&]{if(!m||!hidden||!logits)throw std::runtime_error("invalid LM-head arguments");const Tensor&t=qwen_role_tensor(*m,"lm_head");if(t.shape.size()!=2)throw std::runtime_error("LM-head shape is invalid");uint64_t width=t.shape[0]==m->config.hidden_size?t.shape[0]:t.shape[1],vocab=t.shape[0]==m->config.hidden_size?t.shape[1]:t.shape[0];if(vocabulary<vocab||elements<width)throw std::runtime_error("LM-head shape or buffer is invalid");for(uint64_t row=0;row<vocab;row++){float sum=0;for(uint64_t column=0;column<width;column++)sum+=tensor_value(m->data+t.offset,t.type,row*width+column)*hidden[column];logits[row]=sum;}return 0;});}
int colibri_v2_qwen_token_text(const ColibriV2Model*m,uint32_t token,char*out,uint64_t capacity){return guarded([&]{if(!m||!out||capacity==0)throw std::runtime_error("invalid token text arguments");if(token>=m->vocabulary.size())throw std::runtime_error("token is outside the GGUF vocabulary");const std::string&value=m->vocabulary[token];if(capacity<=value.size())throw std::runtime_error("token text buffer is too small");std::memcpy(out,value.data(),value.size());out[value.size()]=0;return 0;});}
int colibri_v2_token_id(const ColibriV2Model*m,const char*text,uint32_t*token){return guarded([&]{if(!m||!text||!token)throw std::runtime_error("invalid token lookup arguments");const auto it=m->vocabulary_ids.find(text);if(it==m->vocabulary_ids.end())throw std::runtime_error("token text is not in the GGUF vocabulary");*token=it->second;return 0;});}
std::string gguf_byte_encode(const char*text){static const int direct[] = {33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,161,162,163,164,165,166,167,168,169,170,171,172,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255};std::array<int,256>map{};for(int i=0;i<256;i++)map[i]=-1;for(int i=0;i<static_cast<int>(sizeof(direct)/sizeof(direct[0]));i++)map[direct[i]]=direct[i];int extra=0;for(int i=0;i<256;i++)if(map[i]<0)map[i]=256+extra++;std::string out;for(const unsigned char*p=reinterpret_cast<const unsigned char*>(text);*p;p++){int cp=map[*p];if(cp<128)out.push_back(static_cast<char>(cp));else if(cp<2048){out.push_back(static_cast<char>(0xC0|(cp>>6)));out.push_back(static_cast<char>(0x80|(cp&63)));}else{out.push_back(static_cast<char>(0xE0|(cp>>12)));out.push_back(static_cast<char>(0x80|((cp>>6)&63)));out.push_back(static_cast<char>(0x80|(cp&63)));}}return out;}
std::vector<std::string> gguf_utf8_symbols(const std::string&text){std::vector<std::string> symbols;for(size_t i=0;i<text.size();){unsigned char c=text[i];size_t width=(c<0x80)?1:(c<0xE0?2:(c<0xF0?3:4));if(i+width>text.size())width=1;symbols.emplace_back(text.data()+i,width);i+=width;}return symbols;}
int colibri_v2_tokenize(const ColibriV2Model*m,const char*text,uint32_t*tokens,uint64_t capacity,uint64_t*count){return guarded([&]{
    if(!m||!text||!count)throw std::runtime_error("invalid tokenize arguments");
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
int colibri_v2_tensor_read_slice(const ColibriV2Model*m,uint64_t i,uint64_t offset,void*dst,uint64_t bytes){return guarded([&]{if(!m||!dst||i>=m->tensors.size())throw std::runtime_error("invalid tensor slice read");const auto&t=m->tensors[i];if(offset>t.size||bytes>t.size-offset)throw std::runtime_error("tensor slice is out of bounds");std::memcpy(dst,m->data+t.offset+offset,static_cast<size_t>(bytes));return 0;});}
int colibri_v2_tensor_view(const ColibriV2Model*m,uint64_t i,uint64_t offset,uint64_t bytes,const void**out){return guarded([&]{if(!m||!out||i>=m->tensors.size())throw std::runtime_error("invalid tensor view");const auto&t=m->tensors[i];if(offset>t.size||bytes>t.size-offset)throw std::runtime_error("tensor view is out of bounds");*out=m->data+t.offset+offset;return 0;});}
int colibri_v2_qwen_runtime_create(ColibriV2Model*m,const ColibriV2QwenRuntimeOptions*options,ColibriV2QwenRuntime**out){return guarded([&]{
    if(!m||!out)throw std::runtime_error("model and runtime output are required");
    const bool gemma4=m->config.architecture=="gemma4";
    if(m->config.architecture.find("qwen")!=0&&!gemma4)throw std::runtime_error("native runtime supports Qwen and Gemma 4 models");
    if(!m->config.hidden_size||!m->config.layer_count||!m->config.expert_count||!m->config.expert_used_count)throw std::runtime_error("Qwen runtime config is incomplete");
    auto runtime=std::make_unique<ColibriV2QwenRuntime>();
    runtime->model=m;
    runtime->options=options?*options:ColibriV2QwenRuntimeOptions{};
    if(runtime->options.device<0)throw std::runtime_error("Qwen runtime device must be non-negative");
    if(runtime->options.moe_device<0||runtime->options.moe_device>2)throw std::runtime_error("Qwen runtime MoE device is invalid");
    if(runtime->options.mtp_drafts>8)throw std::runtime_error("native Qwen MTP supports at most 8 drafts");
    if(gemma4&&runtime->options.mtp_drafts)throw std::runtime_error("native Gemma 4 MTP is not implemented");
    if(gemma4&&runtime->options.moe_device==0)throw std::runtime_error("native Gemma 4 supports --moe-device cpu or hybrid");
    if(gemma4&&m->config.per_layer_embedding_size)throw std::runtime_error("native Gemma 4 per-layer embeddings are not implemented");
    if(gemma4&&m->config.shared_kv_layers)throw std::runtime_error("native Gemma 4 shared-KV tail layers are not implemented");
    if(runtime->options.expert_top_k>m->config.expert_used_count)throw std::runtime_error("native Qwen expert_top_k cannot exceed the model's trained expert_used_count");
    if(runtime->options.expert_top_p<0.0f||runtime->options.expert_top_p>1.0f)throw std::runtime_error("native Qwen expert_top_p must be within [0, 1]");
    if(runtime->options.prefill_cache_seed>256)throw std::runtime_error("native Qwen prefill_cache_seed supports at most 256 experts per layer");
    if(runtime->options.cache_type_k<0||runtime->options.cache_type_k>3)throw std::runtime_error("native Qwen cache_type_k must be 0 (f32), 1 (f16), 2 (bf16), or 3 (q8_0)");
    if(runtime->options.cache_type_v<0||runtime->options.cache_type_v>3)throw std::runtime_error("native Qwen cache_type_v must be 0 (f32), 1 (f16), 2 (bf16), or 3 (q8_0)");
    if(runtime->options.mtp_drafts&&(runtime->options.cache_type_k||runtime->options.cache_type_v))throw std::runtime_error("native Qwen MTP currently requires f32 KV cache");
    if(!runtime->options.context_limit)runtime->options.context_limit=m->config.context_length?m->config.context_length:4096;
    if(gemma4)build_gemma4_plan(*runtime);else build_qwen_plan(*runtime);
    if(runtime->options.mtp_drafts&&!runtime->mtp_available)throw std::runtime_error("native Qwen MTP was requested but the model has no draft block");
    // Prompt tokens are processed through the batched rows forward in chunks
    // of this size. Bigger chunks amortize expert weight reads further (the
    // CPU MoE reads each routed expert once per chunk); the cost is ~200MB
    // of workspace + pinned staging at 1024. 0 or 1 falls back to
    // one-token-at-a-time decode.
    runtime->prefill_rows=gemma4?1:1024;
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
    // Host RAM budget for the spilled-slot prompt cache (0 disables).
    runtime->host_cache_limit_bytes=
        static_cast<std::uint64_t>(runtime->options.prompt_cache_mib)*1024ull*1024;
    runtime->cuda_ready=false;
    runtime->decode_ready=false;
    *out=runtime.release();
    return 0;
});}
void colibri_v2_qwen_runtime_destroy(ColibriV2QwenRuntime*runtime){if(runtime)release_qwen_device(*runtime);delete runtime;}
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
    out->static_tensor_bytes=runtime->static_tensor_bytes;
    out->expert_tensor_bytes=runtime->expert_tensor_bytes;
    const std::uint64_t sequence_slots=std::max<std::size_t>(1,runtime->sequences.size());
    out->gpu_allocated_bytes=runtime->static_arena_bytes+runtime->workspace_bytes+
        sequence_slots*runtime->state_bytes+runtime->expert_staging_bytes+
        runtime->expert_cache_bytes+sequence_slots*runtime->prefill_snapshots.size()*runtime->prefill_snapshot_bytes;
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
    out->moe_device=runtime->options.moe_device;
    out->cuda_ready=runtime->cuda_ready?1:0;
    out->decode_ready=runtime->decode_ready?1:0;
    out->route_expert_sum=runtime->route_expert_sum;
    out->expert_compute_nanoseconds=runtime->expert_compute_nanoseconds;
    out->prefill_cache_seeded_experts=runtime->prefill_cache_seeded_experts;
    out->prefill_cache_seed_nanoseconds=runtime->prefill_cache_seed_nanoseconds;
    out->direct_paging=runtime->dma_paging?1:0;
    out->paging_registration_nanoseconds=runtime->paging_registration_nanoseconds;
    out->host_available_bytes=runtime->host_available_bytes;
    return 0;
});}
int colibri_v2_qwen_runtime_reset(ColibriV2QwenRuntime*runtime){return guarded([&]{if(!runtime)throw std::runtime_error("invalid Qwen runtime handle");if(runtime->state&&colibri_gpu_memset(runtime->state,0,runtime->state_bytes,runtime->stream)!=0)throw std::runtime_error("failed to reset native Qwen state");runtime->position=0;runtime->last_output_token=0;runtime->processed_tokens.clear();runtime->mtp_cache_tokens=0;runtime->mtp_has_target_hidden=false;runtime->cancelled=false;runtime->cache_admission_enabled=true;return 0;});}
int colibri_v2_qwen_runtime_cancel(ColibriV2QwenRuntime*runtime){return guarded([&]{if(!runtime)throw std::runtime_error("invalid Qwen runtime handle");runtime->cancelled=true;return 0;});}
int colibri_v2_qwen_runtime_prepare(ColibriV2QwenRuntime*runtime){return guarded([&]{
    if(!runtime)throw std::runtime_error("invalid Qwen runtime handle");
    if(runtime->static_arena)return 0;
    if(colibri_gpu_init(runtime->options.device)!=0)throw std::runtime_error("failed to initialize native CUDA runtime");
    std::vector<std::string> option_storage;
#if !defined(_WIN32)
    for(const char*path:{"/opt/cuda/include","/usr/local/cuda/include","/usr/include"})if(access((std::string(path)+"/cuda_fp16.h").c_str(),R_OK)==0)option_storage.push_back(std::string("-I")+path);
#endif
    if(const char*cuda_home=std::getenv("CUDA_HOME"))option_storage.push_back(std::string("-I")+cuda_home+"/include");
    std::vector<const char*>compile_options;for(const auto&option:option_storage)compile_options.push_back(option.c_str());
    std::array<char,16384>compile_log{};
    const std::string cuda_source=std::string(colibri::v2::qwen_cuda_source)+colibri::v2::qwen_native_cuda_source;
    if(colibri_gpu_compile(cuda_source.c_str(),compile_options.data(),static_cast<int32_t>(compile_options.size()),runtime->options.device,compile_log.data(),compile_log.size())!=0)throw std::runtime_error(std::string("failed to compile native Qwen CUDA kernels: ")+compile_log.data());
    runtime->cuda_ready=true;
    if(colibri_gpu_stream_create(&runtime->stream)!=0)throw std::runtime_error("failed to create native CUDA stream");
    if(colibri_gpu_event_create(&runtime->route_event)!=0){release_qwen_device(*runtime);throw std::runtime_error("failed to create native Qwen route event");}
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
            create_profile_event(events.shared_start);create_profile_event(events.shared_end);
            create_profile_event(events.expert_start);create_profile_event(events.expert_end);
        }
        create_profile_event(runtime->cuda_tail_start);
        create_profile_event(runtime->cuda_tail_end);
    }
    try {
        std::vector<bool> persistent(runtime->model->tensors.size(),false);
        persistent[runtime->token_embeddings]=true;
        persistent[runtime->final_norm]=true;
        persistent[runtime->lm_head]=true;
        if(runtime->rope_factors!=std::numeric_limits<std::uint64_t>::max())
            persistent[runtime->rope_factors]=true;
        for(const auto&layer:runtime->layers)for(auto tensor:layer.static_tensors)persistent[tensor]=true;
        if(runtime->options.mtp_drafts){
            for(auto tensor:runtime->mtp_layer_plan.static_tensors)persistent[tensor]=true;
            for(auto tensor:runtime->mtp_special_tensors)persistent[tensor]=true;
        }
        for(std::uint64_t index=0;index<persistent.size();++index)if(persistent[index])runtime->static_arena_bytes+=device_align(runtime->model->tensors[index].size);

        // Byte cursor into the state arena. Attention KV regions size per the
        // configured cache precision (f32=4B, f16=2B/elem); DeltaNet conv/recurrent
        // state stays f32. Each region is 16-byte aligned so mixed f16/f32 regions
        // never leave a following f32 access misaligned.
        // KV region bytes per element count and cache type (q8_0 = 34B/32-elem block).
        auto kv_bytes=[](std::uint64_t elems,int type){return type==3?(elems/32)*34:elems*(type==0?4:2);};
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
        const auto attention_score_bytes =
            static_cast<std::uint64_t>(runtime->model->config.attention_heads) *
            runtime->options.context_limit * sizeof(float);
        const auto decode_workspace_bytes =
            device_align(runtime->model->config.hidden_size * sizeof(float)) * 3 +
            device_align(runtime->scratch_elements * sizeof(float)) * 4 +
            device_align(static_cast<std::uint64_t>(runtime->model->config.expert_used_count) * runtime->moe_intermediate * sizeof(float)) +
            device_align(runtime->model->config.expert_count * sizeof(float)) +
            device_align(runtime->model->config.expert_used_count * sizeof(std::int32_t)) +
            device_align(runtime->model->config.expert_used_count * sizeof(float)) +
            device_align(runtime->model->config.vocabulary_size * sizeof(float)) +
            device_align(sizeof(std::uint64_t)) +
            device_align(attention_score_bytes);
        // The batched rows forward (MTP verification and chunked prefill)
        // needs workspace and host staging proportional to its row capacity;
        // mirror the take()/offset sequence in v2_mtp_verifier.inc.
        runtime->forward_rows_capacity=std::max<std::uint32_t>(runtime->prefill_rows,9);
        const std::uint64_t rows=runtime->forward_rows_capacity;
        const std::uint64_t hidden=runtime->model->config.hidden_size;
        const std::uint64_t top_k=runtime->model->config.expert_used_count;
        const std::uint64_t forward_workspace_bytes=
            device_align(rows*hidden*sizeof(float))*3+
            device_align(rows*runtime->scratch_elements*sizeof(float))*4+
            device_align(rows*runtime->model->config.expert_count*sizeof(float))+
            device_align(rows*top_k*sizeof(std::int32_t))+
            device_align(rows*top_k*sizeof(float))+
            device_align(rows*top_k*runtime->moe_intermediate*sizeof(float))+
            device_align(rows*top_k*sizeof(std::uint64_t))*3+
            device_align(rows*top_k*sizeof(float))+
            device_align(rows*sizeof(std::int32_t))+
            device_align(rows*sizeof(std::uint32_t))+
            device_align(rows*sizeof(std::uint64_t))+
            device_align(attention_score_bytes);
        const std::uint64_t forward_host_bytes=
            device_align(rows*top_k*sizeof(std::int32_t))+
            device_align(rows*top_k*sizeof(float))+
            device_align(rows*hidden*sizeof(float))+
            device_align(rows*top_k*runtime->moe_intermediate*sizeof(float))+
            device_align(rows*top_k*hidden*sizeof(float))+
            device_align(rows*hidden*sizeof(float))+
            device_align(rows*top_k*sizeof(std::uint64_t))*3+
            device_align(rows*top_k*sizeof(float))+
            device_align(rows*sizeof(std::int32_t));
        // Multi-sequence decode slices one decode workspace per sequence out of
        // the (rows-forward-sized) workspace; capacity is bounded by what fits.
        runtime->decode_slice_bytes=decode_workspace_bytes;
        runtime->decode_host_block_bytes=
            device_align(top_k*sizeof(std::int32_t))+
            device_align(top_k*sizeof(float))+
            device_align(hidden*sizeof(float))+
            device_align(top_k*runtime->moe_intermediate*sizeof(float))+
            device_align(hidden*sizeof(float))+
            device_align(sizeof(std::uint64_t));
        runtime->workspace_bytes=device_align(std::max<std::uint64_t>(
            16ULL*1024*1024,
            std::max({max_vector*sizeof(float)*8, decode_workspace_bytes, forward_workspace_bytes})
        ));
        std::uint64_t one_expert=0;
        for(const auto&layer:runtime->layers){std::uint64_t bytes=0;for(auto tensor:layer.expert_tensors)bytes+=runtime->model->tensors[tensor].size/runtime->model->config.expert_count;one_expert=std::max(one_expert,bytes);}
        runtime->expert_staging_bytes=device_align(one_expert*runtime->model->config.expert_used_count*2);
        runtime->host_staging_bytes=std::max(runtime->expert_staging_bytes,device_align(forward_host_bytes));
        runtime->multi_decode_capacity=1;
        const std::uint64_t decode_slots=runtime->options.mtp_drafts?1:std::max<std::uint32_t>(1u,runtime->parallel_sequences);
        // Gemma 4 supports independent sequence slots, but its hybrid expert
        // path currently schedules those slots sequentially. The Qwen
        // layer-overlapped multi-decode driver assumes separate gate/up/down
        // expert tensors and must not consume Gemma's fused Q4_0 bundles.
        if(decode_slots>1&&runtime->options.moe_device!=0&&!runtime->gemma4){
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
        const auto base_total=runtime->static_arena_bytes+runtime->workspace_bytes+slot_count*runtime->state_bytes+runtime->expert_staging_bytes+slot_count*runtime->prefill_snapshots.size()*runtime->prefill_snapshot_bytes;
        // gpu_cache_bytes is the TOTAL GPU budget (base allocations + expert
        // cache). 0 = auto-fit: probe free VRAM and use most of it, leaving a
        // headroom margin. Any positive value is an exact manual budget.
        std::uint64_t gpu_budget=runtime->options.gpu_cache_bytes;
        const bool auto_fit=(gpu_budget==0);
        if(auto_fit&&runtime->options.moe_device!=1){
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
        if(!auto_fit&&gpu_budget&&base_total>gpu_budget){
            auto mib=[](std::uint64_t b){return std::to_string(b/(1024ull*1024));};
            throw std::runtime_error(
                "native Qwen base CUDA allocations ("+mib(base_total)+" MiB = static weights "
                +mib(runtime->static_arena_bytes)+" + workspace "+mib(runtime->workspace_bytes)+" + "
                +std::to_string(slot_count)+"x KV slot "+mib(runtime->state_bytes)+" ("
                +mib(slot_count*runtime->state_bytes)+") + staging "+mib(runtime->expert_staging_bytes)
                +") exceed the --gpu-cache-mib budget ("+mib(gpu_budget)
                +" MiB), before any expert cache. Lower --parallel or --context-window, or raise --gpu-cache-mib.");
        }
        if(runtime->options.moe_device!=1&&gpu_budget>base_total){
            auto available=gpu_budget-base_total;
            auto cache=(available/runtime->expert_slot_bytes)*runtime->expert_slot_bytes;
            // Never allocate more cache than the whole expert set (every
            // (layer,expert) resident => zero misses); saves VRAM on big GPUs.
            const auto max_cache=static_cast<std::uint64_t>(runtime->layers.size())*runtime->model->config.expert_count*runtime->expert_slot_bytes;
            if(cache>max_cache)cache=max_cache;
            runtime->expert_cache_bytes=(cache/runtime->expert_slot_bytes<runtime->model->config.expert_used_count)?0:cache;
        }
        if(colibri_gpu_alloc(runtime->static_arena_bytes,&runtime->static_arena)!=0||
           colibri_gpu_alloc(runtime->workspace_bytes,&runtime->workspace)!=0||
           colibri_gpu_alloc(runtime->expert_staging_bytes,&runtime->expert_staging)!=0||
           colibri_gpu_host_alloc(runtime->host_staging_bytes,&runtime->host_staging)!=0)throw std::runtime_error("failed to allocate native Qwen CUDA arenas");
        for(auto&seq:runtime->sequences)if(colibri_gpu_alloc(runtime->state_bytes,&seq.state)!=0)throw std::runtime_error("failed to allocate native Qwen sequence state");
        runtime->state=runtime->sequences[0].state;
        runtime->active_sequence=0;
        if(runtime->expert_cache_bytes&&colibri_gpu_alloc(runtime->expert_cache_bytes,&runtime->expert_cache)!=0)throw std::runtime_error("failed to allocate native Qwen expert cache");
        if(runtime->prefill_snapshot_bytes){
            // Slot 0's checkpoint pool lives in runtime->prefill_snapshots (the
            // active mirror); slots 1..N-1 own theirs in sequences[i].
            for(auto&snapshot:runtime->prefill_snapshots)if(colibri_gpu_alloc(runtime->prefill_snapshot_bytes,&snapshot.device)!=0)throw std::runtime_error("failed to allocate native Qwen prefill snapshots");
            for(std::size_t i=1;i<runtime->sequences.size();++i){
                runtime->sequences[i].prefill_snapshots.assign(runtime->prefill_snapshots.size(),QwenPrefillSnapshot{});
                for(auto&snapshot:runtime->sequences[i].prefill_snapshots)if(colibri_gpu_alloc(runtime->prefill_snapshot_bytes,&snapshot.device)!=0)throw std::runtime_error("failed to allocate native Qwen prefill snapshots");
            }
        }
        runtime->expert_slots.resize(runtime->expert_cache_bytes/runtime->expert_slot_bytes);
        runtime->expert_history.resize(
            qwen_cache_layer_count(*runtime) *
            runtime->model->config.expert_count
        );
        runtime->device_tensors.assign(runtime->model->tensors.size(),0);
        std::uint64_t cursor=0;
        for(std::uint64_t index=0;index<persistent.size();++index){
            if(!persistent[index])continue;
            const auto&t=runtime->model->tensors[index];
            runtime->device_tensors[index]=runtime->static_arena+cursor;
            if(colibri_gpu_upload_sync(runtime->device_tensors[index],runtime->model->data+t.offset,t.size)!=0)throw std::runtime_error("failed to upload native Qwen static tensor");
            cursor+=device_align(t.size);
        }
        if(colibri_gpu_memset(runtime->workspace,0,runtime->workspace_bytes,runtime->stream)!=0||
           colibri_gpu_memset(runtime->state,0,runtime->state_bytes,runtime->stream)!=0||
           colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("failed to initialize native Qwen CUDA arenas");
    }catch(...){release_qwen_device(*runtime);throw;}
        if(runtime->options.expert_paging>2)
            throw std::runtime_error("native Qwen expert paging mode is invalid");
        runtime->host_available_bytes=available_host_memory();
        const bool forced_direct=runtime->options.expert_paging==2||
            std::getenv("COLIBRI_V2_DMA_PAGING");
        const auto registration_headroom=std::max<std::uint64_t>(
            4ull*1024*1024*1024,runtime->model->size/4);
        const bool auto_direct=runtime->options.expert_paging==0&&
            runtime->host_available_bytes>=runtime->model->size+registration_headroom;
        if(runtime->options.moe_device==2&&(forced_direct||auto_direct)){
            const auto registration_started=std::chrono::steady_clock::now();
            const int registration=colibri_gpu_host_register(
                runtime->model->data,runtime->model->size);
            runtime->paging_registration_nanoseconds=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-registration_started).count();
            if(registration==0){
                runtime->model_registered=true;runtime->dma_paging=true;
                std::fprintf(stderr,
                    "[colibri-v2] direct expert paging on (registered %.1f GiB mmap in %.2fs)\n",
                    runtime->model->size/1073741824.0,
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

void qwen_mtp_append_prompt_pair(
    ColibriV2QwenRuntime& runtime, std::uint32_t token
) {
    if (!runtime.options.mtp_drafts || !runtime.mtp_has_target_hidden) return;
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
    auto q8 = [&](std::uint64_t matrix, std::uint64_t input,
                  std::uint64_t output, int input_size, int output_size) {
        if (colibri_gpu_q8_matvec_transposed(
                matrix, input, output, input_size, output_size, runtime.stream
            ) != 0) throw std::runtime_error("native MTP Q8 projection failed");
    };
    const auto embedding_matrix = runtime.device_tensors[runtime.token_embeddings];
    int token_value = static_cast<int>(token);
    void* embedding_args[] = {
        const_cast<std::uint64_t*>(&embedding_matrix),
        const_cast<std::uint64_t*>(&embedding), &token_value,
        const_cast<int*>(&hidden),
    };
    launch("qwen_q8_embedding", (hidden + 255) / 256, 1, embedding_args);
    auto rms = [&](std::uint64_t input, std::uint64_t weight,
                   std::uint64_t output) {
        int one_centered = 0;
        void* args[] = {&input, &weight, &output, const_cast<int*>(&hidden),
                        const_cast<float*>(&epsilon), &one_centered};
        launch("rms_norm", 1, 1, args);
    };
    rms(embedding, runtime.device_tensors[runtime.mtp_special_tensors[1]], normalized_embedding);
    const auto target_hidden = runtime.state + runtime.mtp_target_hidden_offset;
    rms(target_hidden, runtime.device_tensors[runtime.mtp_special_tensors[2]], normalized_hidden);
    void* concat_args[] = {
        const_cast<std::uint64_t*>(&normalized_embedding),
        const_cast<std::uint64_t*>(&normalized_hidden),
        const_cast<std::uint64_t*>(&concatenated), const_cast<int*>(&hidden),
    };
    launch("qwen_concat_pair", (hidden + 255) / 256, 1, concat_args);
    q8(runtime.device_tensors[runtime.mtp_special_tensors[0]], concatenated,
       fused, hidden * 2, hidden);
    rms(fused, runtime.device_tensors[layer.static_tensors[0]], normalized);
    q8(runtime.device_tensors[layer.static_tensors[2]], normalized, keys, hidden, kv_size);
    q8(runtime.device_tensors[layer.static_tensors[3]], normalized, values, hidden, kv_size);
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
    void* append_args[] = {
        const_cast<std::uint64_t*>(&rotated_keys),
        const_cast<std::uint64_t*>(&values), &cache_keys, &cache_values,
        const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim),
        const_cast<int*>(&position), &capacity,
    };
    launch("kv_append", kv_heads, 1, append_args);
    ++runtime.mtp_cache_tokens;
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
    auto q8=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,int input_size,int output_size){if(colibri_gpu_q8_matvec_transposed(matrix,input,output,input_size,output_size,runtime.stream)!=0)throw std::runtime_error("native MTP Q8 projection failed");};
    auto rms=[&](std::uint64_t input,std::uint64_t weight,std::uint64_t output){int one_centered=0;void*args[]={&input,&weight,&output,const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon),&one_centered};launch("rms_norm",1,1,args);};
    auto add=[&](std::uint64_t target,std::uint64_t source){float scale=1.0f;int count=hidden_size;void*args[]={&target,&source,&scale,&count};launch("scaled_add",(hidden_size+255)/256,1,args);};
    auto tensor=[&](std::size_t role){return runtime.device_tensors[layer.static_tensors.at(role)];};
    const auto embedding_matrix=runtime.device_tensors[runtime.token_embeddings];
    int token_value=static_cast<int>(token);
    void*embedding_args[]={const_cast<std::uint64_t*>(&embedding_matrix),const_cast<std::uint64_t*>(&embedding),&token_value,const_cast<int*>(&hidden_size)};
    launch("qwen_q8_embedding",(hidden_size+255)/256,1,embedding_args);
    rms(embedding,runtime.device_tensors[runtime.mtp_special_tensors[1]],norm_embedding);
    rms(input_hidden,runtime.device_tensors[runtime.mtp_special_tensors[2]],norm_hidden);
    void*concat_args[]={const_cast<std::uint64_t*>(&norm_embedding),const_cast<std::uint64_t*>(&norm_hidden),const_cast<std::uint64_t*>(&concatenated),const_cast<int*>(&hidden_size)};
    launch("qwen_concat_pair",(hidden_size+255)/256,1,concat_args);
    q8(runtime.device_tensors[runtime.mtp_special_tensors[0]],concatenated,hidden,hidden_size*2,hidden_size);
    rms(hidden,tensor(0),normalized);
    q8(tensor(1),normalized,first,hidden_size,q_size);
    q8(tensor(2),normalized,second,hidden_size,kv_size);
    q8(tensor(3),normalized,third,hidden_size,kv_size);
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
    void*append_args[]={&keys,const_cast<std::uint64_t*>(&third),&cache_keys,&cache_values,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),const_cast<int*>(&position),&capacity};
    launch("kv_append",kv_heads,1,append_args);
    std::uint64_t attended=second;int tokens=position+1;float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    void*score_args[]={&queries,&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&scale};
    launch("kv_attention_scores",heads,(tokens+255)/256,score_args);
    void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity};
    launch("kv_attention_values",heads,1,value_args);
    std::uint64_t gated=third;int elements=heads*head_dim;void*gate_args[]={&attended,&gates,&gated,&elements};launch("qwen_attention_gate",(elements+255)/256,1,gate_args);
    q8(tensor(4),gated,residual,elements,hidden_size);add(residual,hidden);
    rms(residual,tensor(7),normalized);
    std::uint64_t router=tensor(8);int router_rows=experts;void*router_args[]={&router,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&router_logits),&router_rows,const_cast<int*>(&hidden_size)};launch("bf16_matvec",router_rows,1,router_args);
    if(colibri_gpu_route_topk(router_logits,selected_device,route_weights,experts,top_k,runtime.stream)!=0)throw std::runtime_error("native MTP routing failed");
    auto*staging=static_cast<std::uint8_t*>(runtime.host_staging);auto*selected_host=reinterpret_cast<std::int32_t*>(staging);
    const auto weights_offset=device_align(top_k*sizeof(std::int32_t));const auto input_offset=weights_offset+device_align(top_k*sizeof(float));const auto cpu_activated_offset=input_offset+device_align(hidden_size*sizeof(float));const auto output_offset=cpu_activated_offset+device_align(top_k*runtime.moe_intermediate*sizeof(float));
    auto*cpu_weights=reinterpret_cast<float*>(staging+weights_offset);auto*cpu_input=reinterpret_cast<float*>(staging+input_offset);auto*cpu_activated=reinterpret_cast<float*>(staging+cpu_activated_offset);auto*cpu_output=reinterpret_cast<float*>(staging+output_offset);
    if(colibri_gpu_download(selected_host,selected_device,top_k*sizeof(std::int32_t),runtime.stream)!=0||colibri_gpu_download(cpu_weights,route_weights,top_k*sizeof(float),runtime.stream)!=0||colibri_gpu_download(cpu_input,normalized,hidden_size*sizeof(float),runtime.stream)!=0)throw std::runtime_error("native MTP route transfer failed");
    auto shared_gate_matrix=tensor(9),shared_up_matrix=tensor(10);void*silu_args[]={&shared_gate_matrix,&shared_up_matrix,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&second),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate)};launch("q8_swiglu_transposed_warp",(intermediate+7)/8,1,silu_args);
    q8(tensor(11),second,third,intermediate,hidden_size);auto shared_gate=tensor(12);void*shared_args[]={const_cast<std::uint64_t*>(&normalized),&shared_gate,const_cast<std::uint64_t*>(&third),const_cast<int*>(&hidden_size)};launch("qwen_shared_scale_bf16",1,1,shared_args);
    if(colibri_gpu_stream_sync(runtime.stream)!=0)throw std::runtime_error("native MTP route synchronization failed");
    qwen_cpu_moe(runtime,layer,selected_host,cpu_weights,top_k,cpu_input,cpu_activated,cpu_output);
    if(colibri_gpu_upload(fourth,cpu_output,hidden_size*sizeof(float),runtime.stream)!=0)throw std::runtime_error("native MTP expert upload failed");
    add(third,fourth);add(residual,third);std::swap(hidden,residual);
    auto draft_hidden=runtime.state+runtime.mtp_draft_hidden_offset;void*copy_args[]={&hidden,&draft_hidden,const_cast<int*>(&hidden_size)};launch("qwen_copy_vector",(hidden_size+255)/256,1,copy_args);
    rms(hidden,runtime.device_tensors[runtime.mtp_special_tensors[3]],normalized);
    if(colibri_gpu_memset(argmax_device,0,sizeof(std::uint64_t),runtime.stream)!=0)throw std::runtime_error("native MTP argmax reset failed");
    int vocabulary=static_cast<int>(runtime.model->config.vocabulary_size);auto lm_head=runtime.device_tensors[runtime.lm_head];void*argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&argmax_device),const_cast<int*>(&hidden_size),&vocabulary};launch("q8_lm_head_argmax_warp",(vocabulary+7)/8,1,argmax_args);
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

// KV cache kernel selection by configured precision (0=f32, 1=f16, 2=bf16).
// K and V are independent: append is one kv_store_<t> launch per cache; scores
// read K, values read V; the fused prefill only runs when K==V (else per-token).
// KV byte size for `elems` elements at a given type (q8_0 = 34 bytes per 32-elem block).
inline std::uint64_t kv_region_bytes(std::uint64_t elems,int type){return type==3?(elems/32)*34:elems*(type==0?4:2);}
inline const char* kv_store_kernel(int t){return t==3?"kv_store_q8":t==2?"kv_store_bf16":t==1?"kv_store_f16":"kv_store_f32";}
inline const char* kv_scores_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_k;return t==3?"kv_attention_scores_q8":t==2?"kv_attention_scores_bf16":t==1?"kv_attention_scores_f16":"kv_attention_scores";}
inline const char* kv_values_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_v;return t==3?"kv_attention_values_q8":t==2?"kv_attention_values_bf16":t==1?"kv_attention_values_f16":"kv_attention_values";}
inline const char* kv_scores_ring_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_k;return t==3?"kv_attention_scores_q8_ring":t==2?"kv_attention_scores_bf16_ring":t==1?"kv_attention_scores_f16_ring":"kv_attention_scores_ring";}
inline const char* kv_values_ring_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_v;return t==3?"kv_attention_values_q8_ring":t==2?"kv_attention_values_bf16_ring":t==1?"kv_attention_values_f16_ring":"kv_attention_values_ring";}

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
inline bool kv_fused_prefill_ok(const ColibriV2QwenRuntime& r){return r.options.cache_type_k==r.options.cache_type_v;}
inline const char* kv_prefill_kernel(const ColibriV2QwenRuntime& r){int t=r.options.cache_type_k;return t==3?"kv_attention_prefill_q8":t==2?"kv_attention_prefill_bf16":t==1?"kv_attention_prefill_f16":"kv_attention_prefill";}

#include "v2_mtp_verifier.inc"

static int gemma4_decode(ColibriV2QwenRuntime& runtime, std::uint32_t input_token,
                         std::uint32_t& output_token) {
    const auto decode_started=std::chrono::steady_clock::now();
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
        launch("qwen_f32_matvec",output_size,1,256,args);
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
        launch(kv_store_kernel(runtime.options.cache_type_k),kv_heads,1,256,k_store_args);
        void* v_store_args[]={const_cast<std::uint64_t*>(&second),&cache_values,
                              const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),
                              &slot,&capacity};
        launch(kv_store_kernel(runtime.options.cache_type_v),kv_heads,1,256,v_store_args);
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
        if(runtime.options.moe_device==1){
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
            const auto* expert_scales=reinterpret_cast<const float*>(runtime.model->data+scale_tensor.offset);
            std::uint64_t paging_cursor=device_align(output_offset+hidden_size*sizeof(float));
            for(int rank=0;rank<top_k;++rank){
                const int expert=selected_host[rank];
                if(expert<0||expert>=experts)throw std::runtime_error("native Gemma 4 selected an invalid expert");
                const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
                auto resident=runtime.expert_residency.find(cache_key);
                if(resident!=runtime.expert_residency.end()){
                    const auto slot_index=resident->second;
                    auto& cache_slot=runtime.expert_slots[slot_index];
                    cache_slot.last_used=record_expert_access(runtime,layer_number,expert).last_used;
                    const auto base=runtime.expert_cache+slot_index*runtime.expert_slot_bytes;
                    gate_up_pointers[gpu_count]=base;
                    down_pointers[gpu_count]=base+gate_up_bytes;
                    gpu_compact_weights[gpu_count]=cpu_weights[rank]*expert_scales[expert];
                    ++gpu_count;++runtime.expert_cache_hits;
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
                std::memcpy(staging+paging_cursor,runtime.model->data+gate_up_tensor.offset+static_cast<std::uint64_t>(expert)*gate_up_bytes,gate_up_bytes);
                std::memcpy(staging+paging_cursor+gate_up_bytes,runtime.model->data+down_tensor.offset+static_cast<std::uint64_t>(expert)*down_bytes,down_bytes);
                std::memcpy(staging+paging_cursor+gate_up_bytes+down_bytes,runtime.model->data+scale_tensor.offset+static_cast<std::uint64_t>(expert)*scale_bytes,scale_bytes);
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
    if(runtime->gemma4)return gemma4_decode(*runtime,input_token,*output_token);
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
    std::uint64_t cursor=runtime->workspace;
    const auto workspace_end=runtime->workspace+runtime->workspace_bytes;
    auto take=[&](std::uint64_t bytes){const auto result=cursor;cursor+=device_align(bytes);if(cursor>workspace_end)throw std::runtime_error("native Qwen workspace is too small");return result;};
    std::uint64_t hidden=take(hidden_size*sizeof(float));
    std::uint64_t residual=take(hidden_size*sizeof(float));
    const std::uint64_t normalized=take(hidden_size*sizeof(float));
    const std::uint64_t first=take(runtime->scratch_elements*sizeof(float));
    const std::uint64_t second=take(runtime->scratch_elements*sizeof(float));
    const std::uint64_t third=take(runtime->scratch_elements*sizeof(float));
    const std::uint64_t fourth=take(runtime->scratch_elements*sizeof(float));
    const std::uint64_t activated=take(top_k*runtime->moe_intermediate*sizeof(float));
    const std::uint64_t router_logits=take(experts*sizeof(float));
    const std::uint64_t selected_device=take(top_k*sizeof(std::int32_t));
    const std::uint64_t route_weights=take(top_k*sizeof(float));
    const std::uint64_t logits=take(runtime->model->config.vocabulary_size*sizeof(float));
    const std::uint64_t argmax_device=take(sizeof(std::uint64_t));
    const std::uint64_t attention_scores=take(
        static_cast<std::uint64_t>(runtime->model->config.attention_heads) *
        runtime->options.context_limit * sizeof(float)
    );
    auto*staging=static_cast<std::uint8_t*>(runtime->host_staging);
    auto*selected_host=reinterpret_cast<std::int32_t*>(staging);
    auto launch_named=[&](const char*name,std::uint32_t grid_x,std::uint32_t grid_y,std::uint32_t block_x,void**arguments,std::uint32_t shared=0){if(colibri_gpu_launch_named(name,grid_x,grid_y,block_x,shared,runtime->stream,arguments)!=0)throw std::runtime_error(std::string("native Qwen CUDA kernel failed: ")+name);};
    auto rms=[&](std::uint64_t input,std::uint64_t weights,std::uint64_t output){int one_centered=0;void*args[]={&input,&weights,&output,const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon),&one_centered};launch_named("rms_norm",1,1,256,args);};
    auto q8=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,int input_size,int output_size){if(colibri_gpu_q8_matvec_transposed(matrix,input,output,input_size,output_size,runtime->stream)!=0)throw std::runtime_error("native Qwen Q8 projection failed");};
    auto f32=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,int input_size,int output_size){void*args[]={&matrix,&input,&output,&input_size,&output_size};launch_named("qwen_f32_matvec",output_size,1,256,args);};
    auto add=[&](std::uint64_t target,std::uint64_t source){float scale=1.0f;int count=hidden_size;void*args[]={&target,&source,&scale,&count};launch_named("scaled_add",(hidden_size+255)/256,1,256,args);};
    auto profile_record=[&](std::uint64_t event){if(runtime->cuda_profile&&colibri_gpu_event_record(event,runtime->stream)!=0)throw std::runtime_error("native Qwen CUDA profiling event failed");};
    {
        const auto embedding=runtime->device_tensors[runtime->token_embeddings];
        const int token=static_cast<int>(input_token);int width=hidden_size;
        void*args[]={const_cast<std::uint64_t*>(&embedding),const_cast<std::uint64_t*>(&hidden),const_cast<int*>(&token),&width};
        launch_named("qwen_q8_embedding",(hidden_size+255)/256,1,256,args);
    }
    for(std::uint32_t layer_number=0;layer_number<runtime->layers.size();++layer_number){
        auto&layer=runtime->layers[layer_number];
        auto*profile=runtime->cuda_profile?&runtime->cuda_layer_profiles[layer_number]:nullptr;
        if(profile)profile_record(profile->pre_start);
        auto tensor=[&](std::size_t role){return runtime->device_tensors[layer.static_tensors.at(role)];};
        rms(hidden,tensor(0),normalized);
        std::size_t moe_base=0;
        if(!layer.attention){
            int channels=static_cast<int>(runtime->model->tensors[layer.static_tensors[1]].shape[1]);
            int gate_elements=static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1]);
            int value_heads=static_cast<int>(runtime->model->tensors[layer.static_tensors[8]].shape[0]);
            int head_dim=static_cast<int>(runtime->model->tensors[layer.static_tensors[9]].shape[0]);
            int value_dim=value_heads*head_dim;
            int key_heads=(channels-value_dim)/(2*head_dim);
            q8(tensor(1),normalized,first,hidden_size,channels);
            q8(tensor(2),normalized,second,hidden_size,gate_elements);
            f32(tensor(4),normalized,third,hidden_size,value_heads);
            f32(tensor(5),normalized,third+value_heads*sizeof(float),hidden_size,value_heads);
            int kernel_size=static_cast<int>(runtime->model->tensors[layer.static_tensors[6]].shape[0]);
            std::uint64_t conv_state=runtime->state+layer.state_first;
            auto conv_weights=tensor(6);
            void*conv_args[]={const_cast<std::uint64_t*>(&first),&conv_weights,&conv_state,const_cast<std::uint64_t*>(&fourth),&channels,&kernel_size};
            launch_named("delta_conv_step",(channels+255)/256,1,256,conv_args);
            std::uint64_t recurrent_state=runtime->state+layer.state_second;
            auto decay=tensor(8),dt=tensor(7),norm=tensor(9);
            std::uint64_t beta=third+value_heads*sizeof(float);
            void*recurrent_args[]={const_cast<std::uint64_t*>(&fourth),const_cast<std::uint64_t*>(&second),&beta,const_cast<std::uint64_t*>(&third),&decay,&dt,&norm,&recurrent_state,const_cast<std::uint64_t*>(&first),&key_heads,&value_heads,&head_dim,const_cast<float*>(&epsilon)};
            launch_named("qwen_delta_recurrent",value_heads,1,256,recurrent_args);
            q8(tensor(3),first,residual,value_dim,hidden_size);
            add(residual,hidden);
            moe_base=10;
        }else{
            const int heads=static_cast<int>(runtime->model->config.attention_heads);
            const int kv_heads=static_cast<int>(runtime->model->config.attention_kv_heads);
            const int head_dim=static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1]/kv_heads);
            const int q_size=heads*2*head_dim,kv_size=kv_heads*head_dim;
            q8(tensor(1),normalized,first,hidden_size,q_size);
            q8(tensor(2),normalized,second,hidden_size,kv_size);
            q8(tensor(3),normalized,third,hidden_size,kv_size);
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
            launch_named(kv_store_kernel(runtime->options.cache_type_k),kv_heads,1,256,k_store_args);
            void*v_store_args[]={const_cast<std::uint64_t*>(&third),&cache_values,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&slot,&capacity};
            launch_named(kv_store_kernel(runtime->options.cache_type_v),kv_heads,1,256,v_store_args);
            std::uint64_t attended=second;int tokens=view.tokens,first_slot=view.first;float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
            void*score_args[]={&queries,&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot,&scale};
            launch_named(kv_scores_ring_kernel(*runtime),heads,(tokens+255)/256,256,score_args);
            void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot};
            launch_named(kv_values_ring_kernel(*runtime),heads,1,256,value_args);
            std::uint64_t gated=third;int elements=heads*head_dim;
            void*gate_args[]={&attended,&gates,&gated,&elements};
            launch_named("qwen_attention_gate",(elements+255)/256,1,256,gate_args);
            q8(tensor(4),gated,residual,elements,hidden_size);
            add(residual,hidden);
            moe_base=7;
        }
        rms(residual,tensor(moe_base),normalized);
        f32(tensor(moe_base+1),normalized,router_logits,hidden_size,experts);
        if(colibri_gpu_route_topk(router_logits,selected_device,route_weights,experts,top_k,runtime->stream)!=0)throw std::runtime_error("native Qwen routing failed");
        const auto cpu_weights_offset=device_align(top_k*sizeof(std::int32_t));
        const auto cpu_input_offset=cpu_weights_offset+device_align(top_k*sizeof(float));
        const auto cpu_activated_offset=cpu_input_offset+device_align(hidden_size*sizeof(float));
        const auto cpu_output_offset=cpu_activated_offset+device_align(top_k*runtime->moe_intermediate*sizeof(float));
        auto*cpu_weights=reinterpret_cast<float*>(staging+cpu_weights_offset);auto*cpu_input=reinterpret_cast<float*>(staging+cpu_input_offset);auto*cpu_activated=reinterpret_cast<float*>(staging+cpu_activated_offset);auto*cpu_output=reinterpret_cast<float*>(staging+cpu_output_offset);
        if(colibri_gpu_download(selected_host,selected_device,top_k*sizeof(std::int32_t),runtime->stream)!=0)throw std::runtime_error("native Qwen route transfer failed");
        if(runtime->options.moe_device!=0&&(colibri_gpu_download(cpu_weights,route_weights,top_k*sizeof(float),runtime->stream)!=0||colibri_gpu_download(cpu_input,normalized,hidden_size*sizeof(float),runtime->stream)!=0))throw std::runtime_error("native Qwen CPU MoE input transfer failed");
        if(colibri_gpu_event_record(runtime->route_event,runtime->stream)!=0)throw std::runtime_error("native Qwen route event failed");
        if(profile){profile_record(profile->pre_end);profile_record(profile->shared_start);}
        const int intermediate=static_cast<int>(runtime->moe_intermediate);
        auto shared_gate_matrix=tensor(moe_base+2),shared_up_matrix=tensor(moe_base+3);
        void*silu_args[]={&shared_gate_matrix,&shared_up_matrix,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&second),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate)};
        launch_named("q8_swiglu_transposed_warp",(intermediate+7)/8,1,256,silu_args);
        q8(tensor(moe_base+4),second,third,intermediate,hidden_size);
        auto shared_gate=tensor(moe_base+5);void*shared_args[]={const_cast<std::uint64_t*>(&normalized),&shared_gate,const_cast<std::uint64_t*>(&third),const_cast<int*>(&hidden_size)};
        launch_named("qwen_shared_scale",1,1,256,shared_args);
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
        if(runtime->options.moe_device!=0&&(runtime->options.expert_top_k>0||(runtime->options.expert_top_p>0.0f&&runtime->options.expert_top_p<1.0f)))
            route_count=apply_expert_router_policy(selected_host,cpu_weights,top_k,static_cast<int>(runtime->options.expert_top_k),runtime->options.expert_top_p);
        runtime->route_expert_sum+=static_cast<std::uint64_t>(route_count);
        const auto pager_started=std::chrono::steady_clock::now();
        if(runtime->options.moe_device==1){
            if(cpu_output_offset+hidden_size*sizeof(float)>runtime->expert_staging_bytes)throw std::runtime_error("native CPU MoE workspace overflow");
            const auto compute_started=std::chrono::steady_clock::now();
            qwen_cpu_moe(*runtime,layer,selected_host,cpu_weights,route_count,cpu_input,cpu_activated,cpu_output);
            runtime->expert_compute_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-compute_started).count();
            if(colibri_gpu_upload(fourth,cpu_output,hidden_size*sizeof(float),runtime->stream)!=0)throw std::runtime_error("native CPU MoE output upload failed");
            add(third,fourth);
        }else if(runtime->options.moe_device==2){
            if(runtime->expert_slots.empty())throw std::runtime_error("native hybrid MoE requires an expert cache budget");
            std::array<std::int32_t,256>cpu_selected{};
            std::array<float,256>cpu_compact_weights{},gpu_compact_weights{};
            std::array<std::uint64_t,256>gate_pointers{},up_pointers{},down_pointers{};
            int cpu_count=0,gpu_count=0;
            std::uint64_t staging_cursor=device_align(cpu_output_offset+hidden_size*sizeof(float));
            struct PendingUpload{std::uint64_t device,host_offset,bytes;};
            std::array<PendingUpload,256>pending{};int pending_count=0;
            for(int rank=0;rank<route_count;++rank){
                const int expert=selected_host[rank];if(expert<0||expert>=experts)throw std::runtime_error("native hybrid MoE selected an invalid expert");
                const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
                auto resident=runtime->expert_residency.find(cache_key);
                if(resident!=runtime->expert_residency.end()){
                    const auto slot_index=resident->second;auto&slot=runtime->expert_slots[slot_index];
                    if(runtime->cache_admission_enabled)slot.last_used=record_expert_access(*runtime,layer_number,expert).last_used;else slot.last_used=++runtime->expert_clock;
                    ++runtime->expert_cache_hits;
                    const auto device_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){const auto bytes=runtime->model->tensors[layer.expert_tensors[role]].size/experts;const auto pointer=device_base+role_offset;if(role==0)gate_pointers[gpu_count]=pointer;else if(role==1)up_pointers[gpu_count]=pointer;else down_pointers[gpu_count]=pointer;role_offset+=bytes;}
                    gpu_compact_weights[gpu_count++]=cpu_weights[rank];continue;
                }
                ++runtime->expert_cache_misses;cpu_selected[cpu_count]=expert;cpu_compact_weights[cpu_count++]=cpu_weights[rank];
                const auto slot_index=select_expert_cache_slot(*runtime,layer_number,expert,true);
                if(slot_index==kNoExpertSlot)continue;
                auto&slot=runtime->expert_slots[slot_index];slot.key=cache_key;slot.valid=true;slot.last_used=++runtime->expert_clock;runtime->expert_residency[cache_key]=slot_index;
                const auto slot_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;
                if(runtime->dma_paging){
                    // DMA each role straight from the registered mmap into the cache slot;
                    // no CPU staging memcpy (the 4.4 ms/token page-in cost we are attacking).
                    std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;if(colibri_gpu_upload(slot_base+role_offset,runtime->model->data+t.offset+offset,bytes,runtime->stream)!=0)throw std::runtime_error("native hybrid MoE DMA cache upload failed");role_offset+=bytes;}
                }else{
                    const auto bundle_start=staging_cursor;
                    for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;if(staging_cursor+bytes>runtime->expert_staging_bytes)throw std::runtime_error("native hybrid MoE staging overflow");std::memcpy(staging+staging_cursor,runtime->model->data+t.offset+offset,bytes);staging_cursor+=bytes;}
                    pending[pending_count++]={slot_base,bundle_start,staging_cursor-bundle_start};
                }
            }
            if(gpu_count){
                const auto table_bytes=static_cast<std::uint64_t>(gpu_count)*(3*sizeof(std::uint64_t)+sizeof(float));
                const auto table_host=device_align(staging_cursor);const auto table_device=runtime->expert_staging+runtime->expert_staging_bytes-device_align(table_bytes);
                if(table_host+table_bytes>runtime->expert_staging_bytes)throw std::runtime_error("native hybrid MoE pointer staging overflow");
                std::memcpy(staging+table_host,gate_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+gpu_count*sizeof(std::uint64_t),up_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+2*gpu_count*sizeof(std::uint64_t),down_pointers.data(),gpu_count*sizeof(std::uint64_t));
                std::memcpy(staging+table_host+3*gpu_count*sizeof(std::uint64_t),gpu_compact_weights.data(),gpu_count*sizeof(float));
                if(colibri_gpu_upload(table_device,staging+table_host,table_bytes,runtime->stream)!=0)throw std::runtime_error("native hybrid MoE table upload failed");
                const auto gate_table=table_device,up_table=gate_table+gpu_count*sizeof(std::uint64_t),down_table=up_table+gpu_count*sizeof(std::uint64_t),weight_table=down_table+gpu_count*sizeof(std::uint64_t);
                if(colibri_gpu_q5_grouped_swiglu(gate_table,up_table,normalized,activated,hidden_size,intermediate,gpu_count,runtime->stream)!=0)throw std::runtime_error("native hybrid MoE gate/up failed");
                const auto down_type=runtime->model->tensors[layer.expert_tensors[2]].type;
                const int status=down_type==8?colibri_gpu_q8_grouped_accumulate(down_table,activated,third,weight_table,intermediate,hidden_size,gpu_count,runtime->stream):colibri_gpu_q6_grouped_accumulate(down_table,activated,third,weight_table,intermediate,hidden_size,gpu_count,runtime->stream);
                if(status!=0)throw std::runtime_error("native hybrid MoE down projection failed");
            }
            for(int index=0;index<pending_count;++index)if(colibri_gpu_upload(pending[index].device,staging+pending[index].host_offset,pending[index].bytes,runtime->stream)!=0)throw std::runtime_error("native hybrid MoE cache upload failed");
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
        for(int rank=0;rank<top_k;++rank){
            const int expert=selected_host[rank];if(expert<0||expert>=experts)throw std::runtime_error("native Qwen router selected an invalid expert");
            std::uint64_t device_base=0;
            const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
            if(!runtime->expert_slots.empty()){
                auto resident=runtime->expert_residency.find(cache_key);
                if(resident!=runtime->expert_residency.end()){
                    const auto slot_index=resident->second;++runtime->expert_cache_hits;
                    auto&slot=runtime->expert_slots[slot_index];
                    if(runtime->cache_admission_enabled)slot.last_used=record_expert_access(*runtime,layer_number,expert).last_used;else slot.last_used=++runtime->expert_clock;
                    device_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;
                    std::uint64_t role_offset=0;
                    for(int role=0;role<3;++role){const auto bytes=runtime->model->tensors[layer.expert_tensors[role]].size/experts;const auto pointer=device_base+role_offset;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;role_offset+=bytes;}
                }else{
                    ++runtime->expert_cache_misses;
                    const auto slot_index=select_expert_cache_slot(*runtime,layer_number,expert,true);
                    if(slot_index!=kNoExpertSlot){
                        auto&slot=runtime->expert_slots[slot_index];slot.key=cache_key;slot.valid=true;slot.last_used=++runtime->expert_clock;runtime->expert_residency[cache_key]=slot_index;
                        std::uint64_t bundle_bytes=0;
                        for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;std::memcpy(staging+staging_cursor+bundle_bytes,runtime->model->data+t.offset+offset,bytes);bundle_bytes+=bytes;}
                        device_base=runtime->expert_cache+slot_index*runtime->expert_slot_bytes;
                        if(staging_cursor+bundle_bytes>runtime->expert_staging_bytes||colibri_gpu_upload(device_base,staging+staging_cursor,bundle_bytes,runtime->stream)!=0)throw std::runtime_error("native Qwen cached expert upload failed");
                        staging_cursor+=bundle_bytes;
                        std::uint64_t role_offset=0;
                        for(int role=0;role<3;++role){const auto bytes=runtime->model->tensors[layer.expert_tensors[role]].size/experts;const auto pointer=device_base+role_offset;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;role_offset+=bytes;}
                    }else{
                        has_uncached_expert=true;
                        for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;std::memcpy(staging+staging_cursor,runtime->model->data+t.offset+offset,bytes);const auto pointer=runtime->expert_staging+staging_cursor;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;staging_cursor+=bytes;}
                    }
                }
            }else{
                ++runtime->expert_cache_misses;
                has_uncached_expert=true;
                for(int role=0;role<3;++role){const auto&t=runtime->model->tensors[layer.expert_tensors[role]];const auto bytes=t.size/experts;const auto offset=static_cast<std::uint64_t>(expert)*bytes;std::memcpy(staging+staging_cursor,runtime->model->data+t.offset+offset,bytes);const auto pointer=runtime->expert_staging+staging_cursor;if(role==0)gate_pointers[rank]=pointer;else if(role==1)up_pointers[rank]=pointer;else down_pointers[rank]=pointer;staging_cursor+=bytes;}
            }
        }
        const auto table_bytes=static_cast<std::uint64_t>(top_k)*sizeof(std::uint64_t)*3;const auto table_host=device_align(staging_cursor);const auto table_device=runtime->expert_staging+runtime->expert_staging_bytes-table_bytes;const auto gate_table=table_device;const auto up_table=gate_table+top_k*sizeof(std::uint64_t);const auto down_table=up_table+top_k*sizeof(std::uint64_t);std::memcpy(staging+table_host,gate_pointers.data(),top_k*sizeof(std::uint64_t));std::memcpy(staging+table_host+top_k*sizeof(std::uint64_t),up_pointers.data(),top_k*sizeof(std::uint64_t));std::memcpy(staging+table_host+2*top_k*sizeof(std::uint64_t),down_pointers.data(),top_k*sizeof(std::uint64_t));
        if(table_host+table_bytes>runtime->expert_staging_bytes)throw std::runtime_error("native Qwen expert staging overflow");
        if(has_uncached_expert&&staging_cursor&&colibri_gpu_upload(runtime->expert_staging,staging,staging_cursor,runtime->stream)!=0)throw std::runtime_error("native Qwen expert upload failed");
        if(colibri_gpu_upload(table_device,staging+table_host,table_bytes,runtime->stream)!=0)throw std::runtime_error("native Qwen expert pointer upload failed");
        if(colibri_gpu_q5_grouped_swiglu(gate_table,up_table,normalized,activated,hidden_size,intermediate,top_k,runtime->stream)!=0)throw std::runtime_error("native Qwen expert gate/up failed");
        const auto down_type=runtime->model->tensors[layer.expert_tensors[2]].type;
        const int down_status=down_type==8?colibri_gpu_q8_grouped_accumulate(down_table,activated,third,route_weights,intermediate,hidden_size,top_k,runtime->stream):colibri_gpu_q6_grouped_accumulate(down_table,activated,third,route_weights,intermediate,hidden_size,top_k,runtime->stream);
        if(down_status!=0)throw std::runtime_error("native Qwen expert down projection failed");
        }
        runtime->expert_page_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-pager_started).count();
        add(residual,third);
        if(profile)profile_record(profile->expert_end);
        std::swap(hidden,residual);
    }
    if(runtime->cuda_profile)profile_record(runtime->cuda_tail_start);
    if(runtime->options.mtp_drafts){
        auto target_hidden=runtime->state+runtime->mtp_target_hidden_offset;
        void*copy_args[]={const_cast<std::uint64_t*>(&hidden),&target_hidden,const_cast<int*>(&hidden_size)};
        launch_named("qwen_copy_vector",(hidden_size+255)/256,1,256,copy_args);
        runtime->mtp_has_target_hidden=true;
    }
    rms(hidden,runtime->device_tensors[runtime->final_norm],normalized);
    int vocabulary=static_cast<int>(runtime->model->config.vocabulary_size);
    if(colibri_gpu_memset(argmax_device,0,sizeof(std::uint64_t),runtime->stream)!=0)throw std::runtime_error("native Qwen argmax reset failed");
    auto lm_head=runtime->device_tensors[runtime->lm_head];
    void*argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&argmax_device),const_cast<int*>(&hidden_size),&vocabulary};
    launch_named("q8_lm_head_argmax_warp",(vocabulary+7)/8,1,256,argmax_args);
    auto*packed_winner=reinterpret_cast<std::uint64_t*>(staging);
    if(colibri_gpu_download(packed_winner,argmax_device,sizeof(*packed_winner),runtime->stream)!=0)throw std::runtime_error("native Qwen output transfer failed");
    if(runtime->cuda_profile)profile_record(runtime->cuda_tail_end);
    const auto tail_wait_started=std::chrono::steady_clock::now();
    if(colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("native Qwen output synchronization failed");
    runtime->tail_wait_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tail_wait_started).count();
    if(runtime->cuda_profile){
        float delta_ms=0.0f,attention_ms=0.0f,shared_ms=0.0f,expert_ms=0.0f,tail_ms=0.0f;
        std::uint32_t delta_layers=0,attention_layers=0;
        auto elapsed=[](std::uint64_t start,std::uint64_t end){float value=0.0f;if(colibri_gpu_event_elapsed(start,end,&value)!=0)throw std::runtime_error("native Qwen CUDA profiling measurement failed");return value;};
        for(std::size_t index=0;index<runtime->layers.size();++index){
            const auto&events=runtime->cuda_layer_profiles[index];
            const float pre=elapsed(events.pre_start,events.pre_end);
            if(runtime->layers[index].attention){attention_ms+=pre;++attention_layers;}else{delta_ms+=pre;++delta_layers;}
            shared_ms+=elapsed(events.shared_start,events.shared_end);
            expert_ms+=elapsed(events.expert_start,events.expert_end);
        }
        tail_ms=elapsed(runtime->cuda_tail_start,runtime->cuda_tail_end);
        std::fprintf(stderr,"[cuda-profile] position=%llu delta=%.3fms/%u attention=%.3fms/%u shared=%.3fms expert=%.3fms tail=%.3fms total=%.3fms\n",
            static_cast<unsigned long long>(runtime->position),delta_ms,delta_layers,attention_ms,attention_layers,
            shared_ms,expert_ms,tail_ms,delta_ms+attention_ms+shared_ms+expert_ms+tail_ms);
    }
    *output_token=0xffffffffu-static_cast<std::uint32_t>(*packed_winner);
    runtime->last_output_token=*output_token;
    runtime->processed_tokens.push_back(input_token);
    ++runtime->position;
    ++runtime->decode_calls;
    runtime->decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-decode_started).count();
    return 0;
});}

// Leading tokens shared by a slot's committed tokens and the new prompt.
static std::uint64_t qwen_sequence_match(
        const std::vector<std::uint32_t>& tokens,
        const std::uint32_t* prompt, std::uint64_t prompt_count) {
    const std::uint64_t limit = std::min<std::uint64_t>(tokens.size(), prompt_count);
    std::uint64_t i = 0;
    while (i < limit && tokens[i] == prompt[i]) ++i;
    return i;
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

// KV region bytes for `elems` elements at a cache precision (mirrors the
// prepare-time sizing lambda; q8_0 = 34B per 32-elem block).
static std::uint64_t qwen_kv_region_bytes(std::uint64_t elems, int type) {
    return type == 3 ? (elems / 32) * 34 : elems * (type == 0 ? 4 : 2);
}

// The device ranges of a slot arena that hold live conversation state at
// `position`: per attention layer, each kv-head's used [0, position) prefix of
// the K and V slabs (layout is head-major: head*capacity + pos, so the used
// prefix is contiguous per head); per DeltaNet layer, the full conv+recurrent
// state. Bytes beyond `position` in the KV slabs are never read (attention is
// position-bounded), so spill/restore can skip them: a spill copies
// position/capacity of the KV instead of the whole arena.
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
    if (seq.processed_tokens.size() < 2048) return;
    for (auto& e : runtime.host_prompts)
        if (e.tokens == seq.processed_tokens) { e.clock = ++runtime.host_cache_clock; return; }
    // Pack only the live ranges (used KV prefixes + DeltaNet state) instead of
    // the whole arena: at small positions the spill is a fraction of state_bytes.
    const auto ranges = qwen_used_state_ranges(runtime, seq.position);
    std::uint64_t packed_bytes = 0;
    for (const auto& r : ranges) packed_bytes += r.second;
    std::uint64_t valid_snaps = 0;
    for (const auto& s : seq.prefill_snapshots) if (s.valid) ++valid_snaps;
    const std::uint64_t need = packed_bytes + valid_snaps * runtime.prefill_snapshot_bytes;
    if (need > runtime.host_cache_limit_bytes) return;
    while (runtime.host_cache_used_bytes + need > runtime.host_cache_limit_bytes
           && !runtime.host_prompts.empty())
        qwen_evict_host_lru(runtime);
    if (runtime.host_cache_used_bytes + need > runtime.host_cache_limit_bytes) return;
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
    e.tokens = seq.processed_tokens;
    e.state = buf;
    e.position = seq.position;
    e.last_output_token = seq.last_output_token;
    e.bytes = packed_bytes;
    for (const auto& s : seq.prefill_snapshots) {
        if (!s.valid || s.tokens.empty()) continue;
        void* sbuf = std::malloc(runtime.prefill_snapshot_bytes);
        if (!sbuf) break;  // partial checkpoint set is fine; arena reuse still works
        if (colibri_gpu_download(sbuf, s.device, runtime.prefill_snapshot_bytes, runtime.stream) != 0
            || colibri_gpu_stream_sync(runtime.stream) != 0) { std::free(sbuf); break; }
        e.snapshots.push_back({s.tokens, sbuf, s.last_output});
        e.bytes += runtime.prefill_snapshot_bytes;
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
    seq.processed_tokens = std::move(e.tokens);
    seq.position = e.position;
    seq.last_output_token = e.last_output_token;
    std::size_t slot_i = 0;
    for (auto& hs : e.snapshots) {
        if (slot_i >= seq.prefill_snapshots.size()) break;
        auto& dst = seq.prefill_snapshots[slot_i++];
        if (colibri_gpu_upload(dst.device, hs.state, runtime.prefill_snapshot_bytes, runtime.stream) != 0
            || colibri_gpu_stream_sync(runtime.stream) != 0)
            break;
        dst.tokens = std::move(hs.tokens);
        dst.last_output = hs.last_output;
        dst.valid = true;
        dst.clock = ++seq.prefill_snapshot_clock;
    }
    for (std::size_t i = slot_i; i < seq.prefill_snapshots.size(); ++i)
        seq.prefill_snapshots[i].valid = false;
    qwen_free_host_prompt(runtime, entry_idx);
    return true;
}

// Route a prompt to the slot/host entry it continues; recycle the LRU slot
// otherwise. A slot/entry only "owns" the prompt if the match covers most of
// its committed tokens, so a short side-request sharing the system prefix can't
// evict a big conversation. The host cache (>=2 slots) lets an LRU-recycled
// conversation survive in RAM and be restored later instead of reprefiling cold.
static void qwen_route_sequence(ColibriV2QwenRuntime& runtime,
        const std::uint32_t* prompt, std::uint64_t prompt_count) {
    const bool host_cache = runtime.host_cache_limit_bytes && runtime.sequences.size() >= 2;
    if (runtime.sequences.size() <= 1 && !host_cache) return;
    constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    std::size_t best = kNone;
    std::uint64_t best_match = 0;
    auto consider = [&](std::size_t i, const std::vector<std::uint32_t>& tokens) {
        const std::uint64_t m = qwen_sequence_match(tokens, prompt, prompt_count);
        if (m > best_match && m * 2 >= tokens.size()) { best_match = m; best = i; }
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
            if (m > best_host_match && m * 2 >= t.size()) { best_host_match = m; best_host = i; }
        }
    auto lru_slot = [&]() {
        std::size_t v = 0;
        for (std::size_t i = 1; i < runtime.sequences.size(); ++i)
            if (runtime.sequences[i].clock < runtime.sequences[v].clock) v = i;
        return v;  // != active: the active slot is the most-recently-stamped
    };
    if (std::getenv("COLIBRI_ROUTE_TRACE"))
        std::fprintf(stderr, "[route] active=%zu best=%zd match=%llu host_best=%zd host_match=%llu entries=%zu\n",
            runtime.active_sequence, best==kNone?-1:(ssize_t)best, (unsigned long long)best_match,
            best_host==kNone?-1:(ssize_t)best_host, (unsigned long long)best_host_match,
            runtime.host_prompts.size());
    if (best != kNone && best_match >= best_host_match) {
        if (best != runtime.active_sequence) qwen_switch_sequence(runtime, best);
    } else if (best_host != kNone) {
        const std::size_t victim = lru_slot();
        qwen_spill_slot_to_host(runtime, victim);
        qwen_restore_host_to_slot(runtime, best_host, victim);
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
        const uint32_t* prompt, uint64_t prompt_count, QwenPromptPlan& plan) {
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
        uint32_t& next_token, bool& done) {
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
        qwen_prompt_checkpoints(runtime,prompt,plan);
    }
    done=(index>=prompt_count);
    return 0;
}

// Experimental prompt-trained expert-cache seed. Batched prefill records route
// frequency without admitting pages; after the prompt is complete, load only
// the hottest N experts per layer in bulk. This avoids prefill-time cache churn
// while giving the first generated token a warm, prompt-specific working set.
// The runtime option enables it; COLIBRI_PREFILL_CACHE_SEED can override that
// value for profiling experiments.
static void qwen_seed_prefill_experts(ColibriV2QwenRuntime& runtime) {
    const char* setting=std::getenv("COLIBRI_PREFILL_CACHE_SEED");
    long requested=static_cast<long>(runtime.options.prefill_cache_seed);
    if(setting)requested=std::strtol(setting,nullptr,10);
    if(runtime.gemma4||runtime.options.mtp_drafts||
       runtime.options.moe_device!=2||runtime.expert_slots.empty())return;
    if(requested<=0)return;
    const std::size_t per_layer=static_cast<std::size_t>(std::clamp<long>(requested,1,256));
    const auto started=std::chrono::steady_clock::now();
    auto* staging=static_cast<std::uint8_t*>(runtime.host_staging);
    const auto experts=runtime.model->config.expert_count;
    const auto cache_layers=qwen_cache_layer_count(runtime);
    std::uint64_t seeded=0;
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
    for(std::uint32_t layer=0;layer<runtime.layers.size();++layer){
        const auto slot_begin=runtime.expert_slots.size()*layer/cache_layers;
        const auto slot_end=runtime.expert_slots.size()*(layer+1)/cache_layers;
        const auto layer_budget=std::min(per_layer,slot_end-slot_begin);
        std::vector<std::uint32_t> candidates;
        candidates.reserve(experts);
        for(std::uint32_t expert=0;expert<experts;++expert){
            const auto key=(static_cast<std::uint64_t>(layer)<<32)|expert;
            const auto& history=runtime.expert_history[static_cast<std::size_t>(layer)*experts+expert];
            if(history.frequency&&runtime.expert_residency.find(key)==runtime.expert_residency.end())
                candidates.push_back(expert);
        }
        std::sort(candidates.begin(),candidates.end(),[&](std::uint32_t left,std::uint32_t right){
            const auto& a=runtime.expert_history[static_cast<std::size_t>(layer)*experts+left];
            const auto& b=runtime.expert_history[static_cast<std::size_t>(layer)*experts+right];
            return a.frequency!=b.frequency?a.frequency>b.frequency:a.last_used>b.last_used;
        });
        if(candidates.size()>layer_budget)candidates.resize(layer_budget);
        const auto& plan=runtime.layers[layer];
        for(const auto expert:candidates){
            const auto slot_index=select_expert_cache_slot(runtime,layer,expert,false);
            if(slot_index==kNoExpertSlot)continue;
            std::uint64_t bundle_bytes=0;
            for(int role=0;role<3;++role)
                bundle_bytes+=runtime.model->tensors[plan.expert_tensors[role]].size/experts;
            if(bundle_bytes>runtime.host_staging_bytes)
                throw std::runtime_error("native Qwen prefill cache seed expert exceeds staging capacity");
            if(cursor+bundle_bytes>runtime.host_staging_bytes)flush();
            const auto bundle_start=cursor;
            for(int role=0;role<3;++role){
                const auto& tensor=runtime.model->tensors[plan.expert_tensors[role]];
                const auto bytes=tensor.size/experts;
                const auto offset=static_cast<std::uint64_t>(expert)*bytes;
                std::memcpy(staging+cursor,runtime.model->data+tensor.offset+offset,bytes);
                cursor+=bytes;
            }
            const auto key=(static_cast<std::uint64_t>(layer)<<32)|expert;
            auto& slot=runtime.expert_slots[slot_index];
            if(slot.valid)runtime.expert_residency.erase(slot.key);
            slot.key=key;slot.valid=true;slot.last_used=++runtime.expert_clock;
            runtime.expert_residency[key]=slot_index;
            pending.push_back({runtime.expert_cache+slot_index*runtime.expert_slot_bytes,
                               bundle_start,bundle_bytes});
            ++seeded;
        }
        flush();
    }
    runtime.prefill_cache_seeded_experts+=seeded;
    runtime.prefill_cache_seed_nanoseconds+=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
}

// Save this prompt's end-of-prefill state so the next turn only prefills its
// suffix, and re-enable expert-cache admission for decode.
static void qwen_prompt_finish(ColibriV2QwenRuntime* runtime,
        const uint32_t* prompt, uint64_t prompt_count, uint32_t next_token) {
    runtime->cache_admission_enabled=true;
    qwen_seed_prefill_experts(*runtime);
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

int colibri_v2_qwen_runtime_generate(ColibriV2QwenRuntime*runtime,const uint32_t*prompt,uint64_t prompt_count,uint64_t max_tokens,ColibriV2TokenCallback callback,void*user){return guarded([&]{
    if(!runtime||!prompt||!prompt_count||!max_tokens||!callback)throw std::runtime_error("invalid native Qwen generation arguments");
    if(prompt_count+max_tokens>runtime->options.context_limit)throw std::runtime_error("native Qwen generation exceeds the context limit");
    if(!runtime->engine_tasks.empty()||!runtime->engine_pending.empty())throw std::runtime_error("the cooperative engine is active; blocking generate is unavailable");
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
    qwen_prompt_finish(runtime,prompt,prompt_count,next_token);
    if(runtime->options.mtp_drafts){
        uint64_t emitted=0;
        if(callback(next_token,user)!=0)return 0;
        ++emitted;
        while(emitted<max_tokens&&!runtime->cancelled){
            const auto base_cache_tokens=runtime->mtp_cache_tokens;
            const auto wanted=static_cast<uint32_t>(std::min<uint64_t>(
                runtime->options.mtp_drafts,max_tokens-emitted
            ));
            std::array<uint32_t,8>drafts{};
            uint32_t draft_input=next_token;
            std::uint64_t draft_hidden=runtime->state+runtime->mtp_target_hidden_offset;
            const auto draft_started=std::chrono::steady_clock::now();
            for(uint32_t index=0;index<wanted;++index){
                drafts[index]=qwen_mtp_draft(*runtime,draft_input,draft_hidden);
                draft_input=drafts[index];
                draft_hidden=runtime->state+runtime->mtp_draft_hidden_offset;
            }
            runtime->mtp_draft_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-draft_started).count();
            std::array<uint32_t,8>inputs{},verified{};
            inputs[0]=next_token;
            for(uint32_t index=1;index<wanted;++index)inputs[index]=drafts[index-1];
            qwen_snapshot_delta_state(*runtime,false);
            const auto batch_started=std::chrono::steady_clock::now();
            std::uint64_t batch_hidden=0;
            qwen_verify_target_rows(*runtime,inputs.data(),static_cast<int>(wanted),verified.data(),&batch_hidden);
            runtime->mtp_verify_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-batch_started).count();
            uint32_t valid=0;
            bool rejected=false;
            for(;valid<wanted;++valid)if(verified[valid]!=drafts[valid]){++valid;rejected=true;break;}
            if(rejected&&valid<wanted){
                const auto rollback_started=std::chrono::steady_clock::now();
                const auto batch_rejected_token=verified[valid-1];
                std::vector<float>batch_trace;
                const bool trace=std::getenv("COLIBRI_MTP_TRACE")!=nullptr;
                const int trace_hidden=static_cast<int>(runtime->model->config.hidden_size);
                if(trace){
                    batch_trace.resize(trace_hidden);
                    const auto trace_source=batch_hidden+static_cast<std::uint64_t>(valid-1)*trace_hidden*sizeof(float);
                    if(colibri_gpu_download(batch_trace.data(),trace_source,trace_hidden*sizeof(float),runtime->stream)!=0||colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("native MTP trace download failed");
                }
                qwen_snapshot_delta_state(*runtime,true);
                runtime->mtp_cache_tokens=base_cache_tokens+valid;
                std::uint64_t replay_hidden=0;
                const auto replay_started=std::chrono::steady_clock::now();
                qwen_verify_target_rows(*runtime,inputs.data(),static_cast<int>(valid),verified.data(),&replay_hidden);
                const int hidden_size=static_cast<int>(runtime->model->config.hidden_size);
                auto replay_source=replay_hidden+static_cast<std::uint64_t>(valid-1)*hidden_size*sizeof(float);
                auto replay_target=runtime->state+runtime->mtp_target_hidden_offset;
                void*replay_copy_args[]={&replay_source,&replay_target,const_cast<int*>(&hidden_size)};
                if(colibri_gpu_launch_named("qwen_copy_vector",(hidden_size+255)/256,1,256,0,runtime->stream,replay_copy_args)!=0||colibri_gpu_stream_sync(runtime->stream)!=0)
                    throw std::runtime_error("native MTP rollback hidden commit failed");
                runtime->processed_tokens.insert(runtime->processed_tokens.end(),inputs.begin(),inputs.begin()+valid);
                runtime->position+=valid;
                runtime->last_output_token=verified[valid-1];
                runtime->decode_calls+=valid;
                runtime->decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-replay_started).count();
                if(trace){
                    std::vector<float>replay_trace(trace_hidden);
                    const auto trace_source=replay_target;
                    if(colibri_gpu_download(replay_trace.data(),trace_source,trace_hidden*sizeof(float),runtime->stream)!=0||colibri_gpu_stream_sync(runtime->stream)!=0)throw std::runtime_error("native MTP replay trace download failed");
                    float maximum=0.0f;
                    for(int index=0;index<trace_hidden;++index)maximum=std::max(maximum,std::fabs(batch_trace[index]-replay_trace[index]));
                    std::fprintf(stderr,"mtp reject position=%llu row=%u draft=%u batch=%u replay=%u hidden_max_diff=%g\n",static_cast<unsigned long long>(runtime->position-valid),valid-1,drafts[valid-1],batch_rejected_token,verified[valid-1],maximum);
                }
                runtime->mtp_accepted_tokens+=valid-1;
                ++runtime->mtp_rejected_tokens;
                runtime->mtp_rollback_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-rollback_started).count();
            }else{
                const int hidden_size=static_cast<int>(runtime->model->config.hidden_size);
                auto source=batch_hidden+static_cast<std::uint64_t>(wanted-1)*hidden_size*sizeof(float);
                auto target=runtime->state+runtime->mtp_target_hidden_offset;
                void*copy_args[]={&source,&target,const_cast<int*>(&hidden_size)};
                if(colibri_gpu_launch_named("qwen_copy_vector",(hidden_size+255)/256,1,256,0,runtime->stream,copy_args)!=0||colibri_gpu_stream_sync(runtime->stream)!=0)
                    throw std::runtime_error("native MTP verifier hidden commit failed");
                runtime->processed_tokens.insert(runtime->processed_tokens.end(),inputs.begin(),inputs.begin()+wanted);
                runtime->position+=wanted;
                runtime->last_output_token=verified[wanted-1];
                runtime->decode_calls+=wanted;
                runtime->decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-batch_started).count();
                runtime->mtp_accepted_tokens+=rejected?wanted-1:wanted;
                if(rejected)++runtime->mtp_rejected_tokens;
            }
            for(uint32_t index=0;index<valid&&emitted<max_tokens;++index){
                next_token=verified[index];
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

int colibri_v2_qwen_task_submit(ColibriV2QwenRuntime*runtime,const uint32_t*prompt,uint64_t prompt_count,uint64_t max_tokens,const uint32_t*stop_tokens,uint64_t stop_count,uint64_t*task_id){return guarded([&]{
    if(!runtime||!prompt||!prompt_count||!max_tokens||!task_id)throw std::runtime_error("invalid native Qwen task arguments");
    if(prompt_count+max_tokens>runtime->options.context_limit)throw std::runtime_error("native Qwen generation exceeds the context limit");
    if(runtime->options.mtp_drafts)throw std::runtime_error("the cooperative engine does not support MTP");
    QwenEngineTask task;
    task.prompt.assign(prompt,prompt+prompt_count);
    if(stop_tokens&&stop_count)task.stop_tokens.assign(stop_tokens,stop_tokens+stop_count);
    task.max_tokens=max_tokens;
    std::lock_guard<std::mutex> lock(runtime->engine_mutex);
    task.id=runtime->engine_next_task_id++;
    *task_id=task.id;
    runtime->engine_pending.push_back(std::move(task));
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
// mirror; requires moe_device != 0 and no MTP.
static void qwen_decode_multi(ColibriV2QwenRuntime* runtime, std::size_t n,
        const std::size_t* slots, const std::uint32_t* inputs, std::uint32_t* outputs) {
    const auto decode_started = std::chrono::steady_clock::now();
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
            activated, router_logits, selected_device, route_weights, logits,
            argmax_device, attention_scores;
        std::int32_t* selected_host; float *cpu_weights, *cpu_input, *cpu_activated, *cpu_output;
        std::uint64_t* winner_host;
    };
    std::vector<Seq> seqs(n);
    std::uint64_t cursor = runtime->workspace;
    const auto workspace_end = runtime->workspace + runtime->workspace_bytes;
    auto take = [&](std::uint64_t bytes) { const auto r = cursor; cursor += device_align(bytes); if (cursor > workspace_end) throw std::runtime_error("native Qwen multi-decode workspace overflow"); return r; };
    for (std::size_t i = 0; i < n; ++i) {
        Seq& s = seqs[i];
        s.slot = slots[i]; s.input = inputs[i];
        s.state = runtime->sequences[s.slot].state;
        s.position = runtime->sequences[s.slot].position;
        if (s.position >= runtime->options.context_limit) throw std::runtime_error("native Qwen context limit exceeded");
        if (s.input >= runtime->model->config.vocabulary_size) throw std::runtime_error("native Qwen input token is out of range");
        s.hidden = take(hidden_size * sizeof(float));
        s.residual = take(hidden_size * sizeof(float));
        s.normalized = take(hidden_size * sizeof(float));
        s.first = take(runtime->scratch_elements * sizeof(float));
        s.second = take(runtime->scratch_elements * sizeof(float));
        s.third = take(runtime->scratch_elements * sizeof(float));
        s.fourth = take(runtime->scratch_elements * sizeof(float));
        s.activated = take(top_k * runtime->moe_intermediate * sizeof(float));
        s.router_logits = take(experts * sizeof(float));
        s.selected_device = take(top_k * sizeof(std::int32_t));
        s.route_weights = take(top_k * sizeof(float));
        s.logits = take(runtime->model->config.vocabulary_size * sizeof(float));
        s.argmax_device = take(sizeof(std::uint64_t));
        s.attention_scores = take(static_cast<std::uint64_t>(runtime->model->config.attention_heads) * runtime->options.context_limit * sizeof(float));
        auto* block = staging + i * runtime->decode_host_block_bytes;
        std::uint64_t off = 0;
        auto htake = [&](std::uint64_t bytes) { auto* p = block + off; off += device_align(bytes); return p; };
        s.selected_host = reinterpret_cast<std::int32_t*>(htake(top_k * sizeof(std::int32_t)));
        s.cpu_weights = reinterpret_cast<float*>(htake(top_k * sizeof(float)));
        s.cpu_input = reinterpret_cast<float*>(htake(hidden_size * sizeof(float)));
        s.cpu_activated = reinterpret_cast<float*>(htake(top_k * runtime->moe_intermediate * sizeof(float)));
        s.cpu_output = reinterpret_cast<float*>(htake(hidden_size * sizeof(float)));
        s.winner_host = reinterpret_cast<std::uint64_t*>(htake(sizeof(std::uint64_t)));
    }
    auto launch_named = [&](const char* name, std::uint32_t gx, std::uint32_t gy, std::uint32_t bx, void** args) { if (colibri_gpu_launch_named(name, gx, gy, bx, 0, runtime->stream, args) != 0) throw std::runtime_error(std::string("native Qwen CUDA kernel failed: ") + name); };
    auto rms = [&](std::uint64_t input, std::uint64_t weights, std::uint64_t output) { int one_centered = 0; void* args[] = {&input, &weights, &output, const_cast<int*>(&hidden_size), const_cast<float*>(&epsilon), &one_centered}; launch_named("rms_norm", 1, 1, 256, args); };
    auto q8 = [&](std::uint64_t matrix, std::uint64_t input, std::uint64_t output, int in_size, int out_size) { if (colibri_gpu_q8_matvec_transposed(matrix, input, output, in_size, out_size, runtime->stream) != 0) throw std::runtime_error("native Qwen Q8 projection failed"); };
    auto f32 = [&](std::uint64_t matrix, std::uint64_t input, std::uint64_t output, int in_size, int out_size) { void* args[] = {&matrix, &input, &output, &in_size, &out_size}; launch_named("qwen_f32_matvec", out_size, 1, 256, args); };
    auto add = [&](std::uint64_t target, std::uint64_t source) { float scale = 1.0f; int count = hidden_size; void* args[] = {&target, &source, &scale, &count}; launch_named("scaled_add", (hidden_size + 255) / 256, 1, 256, args); };
    for (auto& s : seqs) {
        const auto embedding = runtime->device_tensors[runtime->token_embeddings];
        const int token = static_cast<int>(s.input); int width = hidden_size;
        void* args[] = {const_cast<std::uint64_t*>(&embedding), const_cast<std::uint64_t*>(&s.hidden), const_cast<int*>(&token), &width};
        launch_named("qwen_q8_embedding", (hidden_size + 255) / 256, 1, 256, args);
    }
    for (std::uint32_t layer_number = 0; layer_number < runtime->layers.size(); ++layer_number) {
        auto& layer = runtime->layers[layer_number];
        auto tensor = [&](std::size_t role) { return runtime->device_tensors[layer.static_tensors.at(role)]; };
        const std::size_t moe_base = layer.attention ? 7 : 10;
        // Pass A: queue every sequence's GPU work up to (and including) the
        // router readback + shared experts, recording that sequence's event.
        for (auto& s : seqs) {
            rms(s.hidden, tensor(0), s.normalized);
            if (!layer.attention) {
                int channels = static_cast<int>(runtime->model->tensors[layer.static_tensors[1]].shape[1]);
                int gate_elements = static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1]);
                int value_heads = static_cast<int>(runtime->model->tensors[layer.static_tensors[8]].shape[0]);
                int head_dim = static_cast<int>(runtime->model->tensors[layer.static_tensors[9]].shape[0]);
                int value_dim = value_heads * head_dim;
                int key_heads = (channels - value_dim) / (2 * head_dim);
                q8(tensor(1), s.normalized, s.first, hidden_size, channels);
                q8(tensor(2), s.normalized, s.second, hidden_size, gate_elements);
                f32(tensor(4), s.normalized, s.third, hidden_size, value_heads);
                f32(tensor(5), s.normalized, s.third + value_heads * sizeof(float), hidden_size, value_heads);
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
                q8(tensor(3), s.first, s.residual, value_dim, hidden_size);
                add(s.residual, s.hidden);
            } else {
                const int heads = static_cast<int>(runtime->model->config.attention_heads);
                const int kv_heads = static_cast<int>(runtime->model->config.attention_kv_heads);
                const int head_dim = static_cast<int>(runtime->model->tensors[layer.static_tensors[2]].shape[1] / kv_heads);
                const int q_size = heads * 2 * head_dim, kv_size = kv_heads * head_dim;
                q8(tensor(1), s.normalized, s.first, hidden_size, q_size);
                q8(tensor(2), s.normalized, s.second, hidden_size, kv_size);
                q8(tensor(3), s.normalized, s.third, hidden_size, kv_size);
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
                launch_named(kv_store_kernel(runtime->options.cache_type_k), kv_heads, 1, 256, k_store_args);
                void* v_store_args[] = {const_cast<std::uint64_t*>(&s.third), &cache_values, const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), &slot, &capacity};
                launch_named(kv_store_kernel(runtime->options.cache_type_v), kv_heads, 1, 256, v_store_args);
                std::uint64_t attended = s.second; int tokens=view.tokens,first_slot=view.first;
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                void* score_args[] = {&queries, &cache_keys, const_cast<std::uint64_t*>(&s.attention_scores), const_cast<int*>(&heads), const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), &tokens, &capacity, &first_slot, &scale};
                launch_named(kv_scores_ring_kernel(*runtime), heads, (tokens + 255) / 256, 256, score_args);
                void* value_args[] = {const_cast<std::uint64_t*>(&s.attention_scores), &cache_values, &attended, const_cast<int*>(&heads), const_cast<int*>(&kv_heads), const_cast<int*>(&head_dim), &tokens, &capacity, &first_slot};
                launch_named(kv_values_ring_kernel(*runtime), heads, 1, 256, value_args);
                std::uint64_t gated = s.third; int elements = heads * head_dim;
                void* gate_args[] = {&attended, &gates, &gated, &elements};
                launch_named("qwen_attention_gate", (elements + 255) / 256, 1, 256, gate_args);
                q8(tensor(4), gated, s.residual, elements, hidden_size);
                add(s.residual, s.hidden);
            }
            rms(s.residual, tensor(moe_base), s.normalized);
            f32(tensor(moe_base + 1), s.normalized, s.router_logits, hidden_size, experts);
            if (colibri_gpu_route_topk(s.router_logits, s.selected_device, s.route_weights, experts, top_k, runtime->stream) != 0) throw std::runtime_error("native Qwen routing failed");
            if (colibri_gpu_download(s.selected_host, s.selected_device, top_k * sizeof(std::int32_t), runtime->stream) != 0 ||
                colibri_gpu_download(s.cpu_weights, s.route_weights, top_k * sizeof(float), runtime->stream) != 0 ||
                colibri_gpu_download(s.cpu_input, s.normalized, hidden_size * sizeof(float), runtime->stream) != 0) throw std::runtime_error("native Qwen route transfer failed");
            if (colibri_gpu_event_record(runtime->slot_events[s.slot], runtime->stream) != 0) throw std::runtime_error("native Qwen route event failed");
            auto shared_gate_matrix = tensor(moe_base + 2), shared_up_matrix = tensor(moe_base + 3);
            void* silu_args[] = {&shared_gate_matrix, &shared_up_matrix, const_cast<std::uint64_t*>(&s.normalized), const_cast<std::uint64_t*>(&s.second), const_cast<int*>(&hidden_size), const_cast<int*>(&intermediate)};
            launch_named("q8_swiglu_transposed_warp", (intermediate + 7) / 8, 1, 256, silu_args);
            q8(tensor(moe_base + 4), s.second, s.third, intermediate, hidden_size);
            auto shared_gate = tensor(moe_base + 5);
            void* shared_args[] = {const_cast<std::uint64_t*>(&s.normalized), &shared_gate, const_cast<std::uint64_t*>(&s.third), const_cast<int*>(&hidden_size)};
            launch_named("qwen_shared_scale", 1, 1, 256, shared_args);
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
            if (runtime->options.moe_device == 1) {
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
                std::array<std::uint64_t, 256> gate_pointers{}, up_pointers{}, down_pointers{};
                int cpu_count = 0, gpu_count = 0;
                std::uint64_t staging_cursor = 0;
                struct PendingUpload { std::uint64_t device, host_offset, bytes; };
                std::array<PendingUpload, 256> pending{}; int pending_count = 0;
                for (int rank = 0; rank < route_count; ++rank) {
                    const int expert = s.selected_host[rank];
                    if (expert < 0 || expert >= experts) throw std::runtime_error("native hybrid MoE selected an invalid expert");
                    const auto cache_key = (static_cast<std::uint64_t>(layer_number) << 32) | static_cast<std::uint32_t>(expert);
                    auto resident = runtime->expert_residency.find(cache_key);
                    if (resident != runtime->expert_residency.end()) {
                        const auto slot_index = resident->second; auto& slot = runtime->expert_slots[slot_index];
                        if (runtime->cache_admission_enabled) slot.last_used = record_expert_access(*runtime, layer_number, expert).last_used; else slot.last_used = ++runtime->expert_clock;
                        ++runtime->expert_cache_hits;
                        const auto device_base = runtime->expert_cache + slot_index * runtime->expert_slot_bytes; std::uint64_t role_offset = 0;
                        for (int role = 0; role < 3; ++role) { const auto bytes = runtime->model->tensors[layer.expert_tensors[role]].size / experts; const auto pointer = device_base + role_offset; if (role == 0) gate_pointers[gpu_count] = pointer; else if (role == 1) up_pointers[gpu_count] = pointer; else down_pointers[gpu_count] = pointer; role_offset += bytes; }
                        gpu_compact_weights[gpu_count++] = s.cpu_weights[rank]; continue;
                    }
                    ++runtime->expert_cache_misses; cpu_selected[cpu_count] = expert; cpu_compact_weights[cpu_count++] = s.cpu_weights[rank];
                    const auto slot_index = select_expert_cache_slot(*runtime, layer_number, expert, true);
                    if (slot_index == kNoExpertSlot) continue;
                    auto& slot = runtime->expert_slots[slot_index]; slot.key = cache_key; slot.valid = true; slot.last_used = ++runtime->expert_clock; runtime->expert_residency[cache_key] = slot_index;
                    const auto slot_base = runtime->expert_cache + slot_index * runtime->expert_slot_bytes;
                    if (runtime->dma_paging) {
                        std::uint64_t role_offset = 0;
                        for (int role = 0; role < 3; ++role) { const auto& t = runtime->model->tensors[layer.expert_tensors[role]]; const auto bytes = t.size / experts; const auto offset = static_cast<std::uint64_t>(expert) * bytes; if (colibri_gpu_upload(slot_base + role_offset, runtime->model->data + t.offset + offset, bytes, runtime->stream) != 0) throw std::runtime_error("native hybrid MoE DMA cache upload failed"); role_offset += bytes; }
                    } else {
                        const auto bundle_start = staging_cursor;
                        for (int role = 0; role < 3; ++role) { const auto& t = runtime->model->tensors[layer.expert_tensors[role]]; const auto bytes = t.size / experts; const auto offset = static_cast<std::uint64_t>(expert) * bytes; if (staging_cursor + bytes > runtime->expert_staging_bytes) throw std::runtime_error("native hybrid MoE staging overflow"); std::memcpy(pag + staging_cursor, runtime->model->data + t.offset + offset, bytes); staging_cursor += bytes; }
                        pending[pending_count++] = {slot_base, bundle_start, staging_cursor - bundle_start};
                    }
                }
                if (gpu_count) {
                    const auto table_bytes = static_cast<std::uint64_t>(gpu_count) * (3 * sizeof(std::uint64_t) + sizeof(float));
                    const auto table_host = device_align(staging_cursor); const auto table_device = runtime->expert_staging + runtime->expert_staging_bytes - device_align(table_bytes);
                    if (table_host + table_bytes > runtime->expert_staging_bytes) throw std::runtime_error("native hybrid MoE pointer staging overflow");
                    std::memcpy(pag + table_host, gate_pointers.data(), gpu_count * sizeof(std::uint64_t));
                    std::memcpy(pag + table_host + gpu_count * sizeof(std::uint64_t), up_pointers.data(), gpu_count * sizeof(std::uint64_t));
                    std::memcpy(pag + table_host + 2 * gpu_count * sizeof(std::uint64_t), down_pointers.data(), gpu_count * sizeof(std::uint64_t));
                    std::memcpy(pag + table_host + 3 * gpu_count * sizeof(std::uint64_t), gpu_compact_weights.data(), gpu_count * sizeof(float));
                    if (colibri_gpu_upload(table_device, pag + table_host, table_bytes, runtime->stream) != 0) throw std::runtime_error("native hybrid MoE table upload failed");
                    const auto gate_table = table_device, up_table = gate_table + gpu_count * sizeof(std::uint64_t), down_table = up_table + gpu_count * sizeof(std::uint64_t), weight_table = down_table + gpu_count * sizeof(std::uint64_t);
                    if (colibri_gpu_q5_grouped_swiglu(gate_table, up_table, s.normalized, s.activated, hidden_size, intermediate, gpu_count, runtime->stream) != 0) throw std::runtime_error("native hybrid MoE gate/up failed");
                    const auto down_type = runtime->model->tensors[layer.expert_tensors[2]].type;
                    const int status = down_type == 8 ? colibri_gpu_q8_grouped_accumulate(down_table, s.activated, s.third, weight_table, intermediate, hidden_size, gpu_count, runtime->stream) : colibri_gpu_q6_grouped_accumulate(down_table, s.activated, s.third, weight_table, intermediate, hidden_size, gpu_count, runtime->stream);
                    if (status != 0) throw std::runtime_error("native hybrid MoE down projection failed");
                }
                for (int index = 0; index < pending_count; ++index) if (colibri_gpu_upload(pending[index].device, pag + pending[index].host_offset, pending[index].bytes, runtime->stream) != 0) throw std::runtime_error("native hybrid MoE cache upload failed");
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
        launch_named("q8_lm_head_argmax_warp", (vocabulary + 7) / 8, 1, 256, argmax_args);
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
        if (!(m && m * 2 >= tokens.size())) continue;
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
            if (m > best_host_match && m * 2 >= t.size()) { best_host_match = m; best_host = i; }
        }
    // The conversation is already live on a busy slot and nothing free/host
    // beats it: wait for that task to finish rather than forking the history.
    if (busy_match > best_match && busy_match > best_host_match) return false;
    std::size_t chosen;
    if (best != kNone && best_match >= best_host_match) {
        chosen = best;
    } else if (best_host != kNone) {
        qwen_spill_slot_to_host(runtime, free_lru);
        qwen_restore_host_to_slot(runtime, best_host, free_lru);
        chosen = free_lru;
    } else {
        if (host_cache) qwen_spill_slot_to_host(runtime, free_lru);
        chosen = free_lru;
    }
    qwen_switch_sequence(runtime, chosen);
    runtime.sequences[chosen].clock = ++runtime.sequence_clock;
    runtime.slot_owner[chosen] = static_cast<long long>(task.id);
    task.slot = chosen;
    if (qwen_prompt_begin(&runtime, prompt, prompt_count, task.plan) != 0)
        throw std::runtime_error("native Qwen prompt admission failed");
    task.index = task.plan.prompt_start;
    task.next_token = task.plan.next_token;
    if (task.index >= prompt_count) {  // full reuse: nothing to prefill
        qwen_prompt_finish(&runtime, prompt, prompt_count, task.next_token);
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
    if(runtime->engine_tasks.empty())return 0;
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
            if(task.phase==0&&!qwen_engine_try_start(*runtime,task))continue;
            if(task.phase==1){
                qwen_switch_sequence(*runtime,task.slot);
                runtime->cache_admission_enabled=false;
                bool done=false;
                const int status=qwen_prefill_unit(runtime,task.prompt.data(),task.prompt.size(),task.plan,task.index,task.next_token,done);
                if(status)throw std::runtime_error("native Qwen prefill failed");
                if(done){
                    qwen_prompt_finish(runtime,task.prompt.data(),task.prompt.size(),task.next_token);
                    task.phase=2;
                }
                continue;
            }
            // phase 2: emit the buffered token, then either finish or defer
            // computing the next one -- the emit-then-compute order matches the
            // blocking decode loop, so a single-task engine run stays
            // bit-identical to blocking generate.
            emit(task.id,task.next_token,0);
            ++task.emitted;
            const bool stopped=std::find(task.stop_tokens.begin(),task.stop_tokens.end(),task.next_token)!=task.stop_tokens.end();
            if(stopped||task.cancelled||task.emitted>=task.max_tokens){
                finished.push_back(task.id);emit(task.id,0,1);continue;
            }
            pending_decode.push_back(&task);
        }catch(const std::exception&){
            // Isolate the failure: this task dies, the others keep running.
            finished.push_back(task.id);emit(task.id,0,2);
        }
    }
    // Compute the deferred next tokens. Two or more sequences go through the
    // multi-sequence driver (layer-level CPU/GPU overlap); a single one keeps
    // the plain decode path.
    if(!pending_decode.empty()){
        runtime->cache_admission_enabled=true;
        std::size_t batched=0;
        if(pending_decode.size()>=2&&runtime->multi_decode_capacity>=2&&runtime->options.moe_device!=0){
            batched=std::min<std::size_t>(pending_decode.size(),runtime->multi_decode_capacity);
            std::vector<std::size_t> slots(batched);
            std::vector<std::uint32_t> batch_inputs(batched),batch_outputs(batched);
            for(std::size_t i=0;i<batched;++i){slots[i]=pending_decode[i]->slot;batch_inputs[i]=pending_decode[i]->next_token;}
            qwen_park_active(*runtime,true);
            try{
                qwen_decode_multi(runtime,batched,slots.data(),batch_inputs.data(),batch_outputs.data());
                qwen_park_active(*runtime,false);
                for(std::size_t i=0;i<batched;++i)pending_decode[i]->next_token=batch_outputs[i];
            }catch(const std::exception&){
                qwen_park_active(*runtime,false);
                // The whole batch shares one stream of work; fail all of it.
                for(std::size_t i=0;i<batched;++i){finished.push_back(pending_decode[i]->id);emit(pending_decode[i]->id,0,2);}
            }
        }
        for(std::size_t i=batched;i<pending_decode.size();++i){
            auto*task=pending_decode[i];
            try{
                qwen_switch_sequence(*runtime,task->slot);
                const int status=colibri_v2_qwen_runtime_decode(runtime,task->next_token,&task->next_token);
                if(status)throw std::runtime_error("native Qwen decode failed");
            }catch(const std::exception&){
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
    return 0;
});}

int colibri_v2_session_create(ColibriV2Model*m,uint64_t limit,ColibriV2Session**out){return guarded([&]{if(!m||!out||!limit)throw std::runtime_error("invalid session arguments");*out=new ColibriV2Session{m,limit};return 0;});}
void colibri_v2_session_destroy(ColibriV2Session*s){delete s;}
int colibri_v2_session_prompt(ColibriV2Session*s,const uint32_t*t,uint64_t n){return guarded([&]{if(!s||(!t&&n)||s->history.size()+n>s->limit)throw std::runtime_error("context limit exceeded");s->history.insert(s->history.end(),t,t+n);s->prompt+=n;return 0;});}
int colibri_v2_session_decode(ColibriV2Session*s,uint32_t*out,float*logits,uint64_t n){return guarded([&]{if(!s||!out||s->cancelled||s->history.size()>=s->limit)throw std::runtime_error(s&&s->cancelled?"session cancelled":"context limit exceeded"); uint64_t h=1469598103934665603ULL;for(auto x:s->history)h=(h^x)*1099511628211ULL;*out=static_cast<uint32_t>((h^(h>>32))&0x7fffffff);if(logits)for(uint64_t i=0;i<n;i++)logits[i]=-INFINITY; s->history.push_back(*out);s->decoded++;s->calls++;return 0;});}
int colibri_v2_session_generate(ColibriV2Session*s,uint64_t max,ColibriV2TokenCallback callback,void*user){return guarded([&]{if(!s||!callback)throw std::runtime_error("session and callback are required");for(uint64_t i=0;i<max&&!s->cancelled;i++){uint32_t token=0;int status=colibri_v2_session_decode(s,&token,nullptr,0);if(status) return status;if(callback(token,user)!=0){s->cancelled=true;break;}}return 0;});}
int colibri_v2_session_cancel(ColibriV2Session*s){if(!s){fail("invalid session");return -1;}s->cancelled=true;return 0;}
int colibri_v2_session_sync(ColibriV2Session*s){if(!s){fail("invalid session");return -1;}return 0;}
int colibri_v2_session_stats(const ColibriV2Session*s,ColibriV2Stats*out){return guarded([&]{if(!s||!out)throw std::runtime_error("invalid stats handle");*out={s->prompt,s->decoded,s->calls,s->model->size};return 0;});}
int colibri_v2_session_attach_kv_cache(ColibriV2Session*s,ColibriV2KvCache*cache){return guarded([&]{if(!s||!cache)throw std::runtime_error("invalid session or KV cache");if(cache->position!=static_cast<int32_t>(s->history.size()))throw std::runtime_error("KV cache position does not match session context");s->kv_cache=cache;return 0;});}
int colibri_v2_session_detach_kv_cache(ColibriV2Session*s){if(!s){fail("invalid session");return -1;}s->kv_cache=nullptr;return 0;}
}
