// CPU implementation of the device interface in colibri_gpu_driver.h.
//
// The runtime's layer loop talks to the device exclusively through this
// interface and launches compute by kernel *name*, so supplying a second
// implementation is enough to run the whole model on the host -- v2_runtime.cpp
// needs no parallel code path. Device pointers become host pointers, streams
// and events become bookkeeping, and colibri_cpu_launch_named runs the grid
// over the kernels generated from the CUDA corpus (see cpu_kernels.cpp).
//
// Ordering: every launch here is synchronous. That trivially satisfies the
// stream and event semantics the runtime relies on -- a stream is an ordered
// queue, and everything is already complete by the time the call returns -- so
// the sync and wait entry points have nothing to do. Overlap is a stage-2
// concern; correctness does not depend on it.

#include <colibri_backend.hpp>
#include <colibri_cpu_backend.hpp>
#include <colibri_cpu_kernels_api.hpp>
#include <colibri_cpu_native.hpp>
#include <colibri_cpu_shim_geometry.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>

#if defined(_OPENMP)
#  include <omp.h>
#endif

namespace {

// Device allocations are plain host allocations. 256-byte alignment matches
// CUDA's guarantee, which the corpus relies on for the 128-bit vector loads.
constexpr std::size_t kDeviceAlignment = 256;

void* aligned_allocate(std::size_t bytes) {
    if (bytes == 0) bytes = 1;
    const std::size_t rounded =
        (bytes + kDeviceAlignment - 1) / kDeviceAlignment * kDeviceAlignment;
#if defined(_WIN32)
    return _aligned_malloc(rounded, kDeviceAlignment);
#else
    return std::aligned_alloc(kDeviceAlignment, rounded);
#endif
}

void aligned_release(void* pointer) {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

struct Event {
    bool timed = false;
    std::chrono::steady_clock::time_point stamp{};
    bool recorded = false;
};

struct Stream {
    // Present so stream handles are distinct, non-null, and freeable; all work
    // is synchronous, so there is nothing to order.
    unsigned int identifier = 0;
};

// CUDA until something selects otherwise, so an unmodified GPU deployment
// behaves exactly as before.
std::atomic<int> g_backend{kColibriBackendCuda};

std::mutex g_object_mutex;
std::unordered_map<std::uint64_t, std::unique_ptr<Stream>> g_streams;
std::unordered_map<std::uint64_t, std::unique_ptr<Event>> g_events;
std::atomic<std::uint64_t> g_next_handle{1};

struct Resolved {
    colibri::cpu::ColibriCpuKernel kernel = nullptr;
    colibri::cpu::NativeKernel native = nullptr;
    bool cooperative = true;
};

// Native kernels are registered before main by static initializers, so this map
// is only written during startup and read-only thereafter.
std::unordered_map<std::string, colibri::cpu::NativeKernel>& native_registry() {
    static std::unordered_map<std::string, colibri::cpu::NativeKernel> registry;
    return registry;
}

// COLIBRI_CPU_FORCE_EMULATION=1 routes every kernel back to the emulated
// corpus. Slow, but it is the reference implementation, so it is the first
// thing to try when the CPU backend produces wrong output: if a defect
// survives it, the defect is not in a hand-written kernel.
std::atomic<bool> g_force_emulation{
    [] {
        const char* setting = std::getenv("COLIBRI_CPU_FORCE_EMULATION");
        return setting != nullptr && setting[0] == '1';
    }()
};

// --- launch profiling -----------------------------------------------------
//
// Which kernels to hand-write next is an empirical question, and the answer is
// not the same as on the GPU: emulated cooperative kernels cost far more per
// call than their CUDA versions, so the CPU ordering has to be measured rather
// than inherited. Off unless COLIBRI_CPU_PROFILE=1, and the check is a single
// relaxed load on the launch path.
struct ProfileEntry {
    std::uint64_t calls = 0;
    std::uint64_t nanoseconds = 0;
    bool native = false;
};

bool profiling_enabled() {
    static const bool enabled = [] {
        const char* setting = std::getenv("COLIBRI_CPU_PROFILE");
        return setting != nullptr && setting[0] == '1';
    }();
    return enabled;
}

std::mutex g_profile_mutex;
std::unordered_map<std::string, ProfileEntry> g_profile;

void record_launch(const char* name, std::uint64_t nanoseconds, bool native) {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    auto& entry = g_profile[name];
    entry.calls++;
    entry.nanoseconds += nanoseconds;
    entry.native = native;
}

// Resolved kernels, keyed by the name the runtime launches.
//
// The cache is filled under a lock but read without one. Entries are only ever
// added, never mutated or erased, and a racing pair of lookups for the same
// name resolves to the same function pointer, so a reader either misses and
// takes the slow path or sees a complete entry. Holding a mutex per launch
// would serialize every kernel in the model behind one lock.
std::mutex g_kernel_mutex;
std::unordered_map<std::string, Resolved> g_kernel_cache;
std::atomic<bool> g_kernel_cache_dirty{false};

// COLIBRI_CPU_EMULATE=name1,name2 forces just those kernels back to the
// emulated corpus. Bisecting a wrong-output bug with COLIBRI_CPU_FORCE_EMULATION
// works but costs ~100x, which is impractical for a defect that only appears
// after a hundred tokens; this narrows the search without that penalty.
bool emulation_forced_for(const char* name) {
    static const std::string list = [] {
        const char* setting = std::getenv("COLIBRI_CPU_EMULATE");
        return std::string(setting ? setting : "");
    }();
    if (list.empty()) return false;
    const std::string needle(name);
    std::size_t at = 0;
    while ((at = list.find(needle, at)) != std::string::npos) {
        const bool start = at == 0 || list[at - 1] == ',';
        const std::size_t end = at + needle.size();
        const bool finish = end == list.size() || list[end] == ',';
        if (start && finish) return true;
        at = end;
    }
    return false;
}

Resolved resolve(const char* name) {
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        const auto found = g_kernel_cache.find(name);
        if (found != g_kernel_cache.end()) return found->second;
    }
    Resolved resolved;
    colibri::cpu::find_kernel_entry(name, &resolved.kernel, &resolved.cooperative);
    const auto& registry = native_registry();
    const auto native = registry.find(name);
    if (native != registry.end() && !emulation_forced_for(name))
        resolved.native = native->second;
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    g_kernel_cache.emplace(name, resolved);
    return resolved;
}

struct LaunchPayload {
    colibri::cpu::ColibriCpuKernel kernel;
    void** arguments;
};

void run_thread(void* payload) {
    auto* launch = static_cast<LaunchPayload*>(payload);
    launch->kernel(launch->arguments);
}

int worker_count() {
    if (const char* setting = std::getenv("COLIBRI_CPU_THREADS")) {
        const int requested = std::atoi(setting);
        if (requested > 0) return requested;
    }
    // Physical cores, not logical. Decode is bandwidth-bound on weights, and
    // SMT siblings share load ports and cache, so the second thread on a core
    // adds contention rather than throughput. Measured on a 16-core/32-thread
    // Ryzen 9 9955HX with a 35B Q8_0 MoE: 8.30 tok/s at 16 threads against
    // 5.73 at 32, so the logical-core default was costing 31%.
    //
    // qwen_cpu_thread_count() applies the same rule to the CPU MoE path; an
    // explicit OMP_NUM_THREADS still wins in both places.
#if defined(_OPENMP)
    int team = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        const int physical = omp_get_num_procs() / 2;
        if (physical >= 1 && team > physical) team = physical;
    }
    return team;
#else
    const unsigned int detected = std::thread::hardware_concurrency();
    if (detected == 0) return 1;
    const int physical = static_cast<int>(detected) / 2;
    return physical >= 1 ? physical : 1;
#endif
}

// Persistent worker pool for kernel launches.
//
// A fork-join per launch costs several microseconds, and the runtime issues
// roughly a thousand launches per token -- enough to dominate a decode before
// any arithmetic happens. Workers are therefore created once and parked between
// launches on a generation counter.
//
// Parking is spin-then-sleep because the two cases have opposite needs: inside
// a token, launches arrive microseconds apart and a condition-variable wakeup
// would cost more than the kernel; between requests the pool may idle for
// seconds and must not burn cores. Workers spin briefly, then yield, then sleep.
class LaunchPool {
public:
    ~LaunchPool() { shutdown(); }

    void run(std::uint64_t blocks, void (*body)(void*, std::uint64_t),
             void* payload, std::string* failure) {
        ensure_started();

        const int workers = static_cast<int>(threads_.size());
        // Below this, the handoff costs more than the work it distributes.
        if (workers == 0 || blocks <= 1) {
            run_range(blocks, body, payload, failure);
            return;
        }

        body_ = body;
        payload_ = payload;
        blocks_ = blocks;
        failure_sink_ = failure;
        next_block_.store(0, std::memory_order_relaxed);
        remaining_.store(workers, std::memory_order_relaxed);

        // Publishing the generation releases the job fields above to workers.
        generation_.fetch_add(1, std::memory_order_release);
        if (sleepers_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lock(sleep_mutex_);
            sleep_signal_.notify_all();
        }

        // The calling thread is a worker too, so an N-way pool uses N+1 threads
        // only transiently and never leaves the caller idle.
        consume();

        while (remaining_.load(std::memory_order_acquire) != 0)
            std::this_thread::yield();
    }

    void shutdown() {
        if (threads_.empty()) return;
        stopping_.store(true, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(sleep_mutex_);
            sleep_signal_.notify_all();
        }
        for (auto& thread : threads_)
            if (thread.joinable()) thread.join();
        threads_.clear();
    }

private:
    void ensure_started() {
        std::call_once(started_, [this] {
            const int workers = worker_count() - 1;  // the caller is the rest
            for (int index = 0; index < workers; ++index)
                threads_.emplace_back([this] { worker_loop(); });
        });
    }

    void record_failure(const char* message) {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        if (failure_sink_ != nullptr && failure_sink_->empty())
            *failure_sink_ = message;
    }

    void run_range(std::uint64_t blocks, void (*body)(void*, std::uint64_t),
                   void* payload, std::string* failure) {
        for (std::uint64_t block = 0; block < blocks; ++block) {
            try {
                body(payload, block);
            } catch (const std::exception& error) {
                if (failure != nullptr && failure->empty()) *failure = error.what();
                return;
            }
        }
    }

    // Claim blocks until the grid is exhausted. Dynamic rather than a static
    // split: attention block cost scales with sequence position and expert
    // block cost with routing, so an even division strands workers.
    void consume() {
        for (;;) {
            const std::uint64_t block =
                next_block_.fetch_add(1, std::memory_order_relaxed);
            if (block >= blocks_) return;
            try {
                body_(payload_, block);
            } catch (const std::exception& error) {
                record_failure(error.what());
                // Drain the grid so the launch still terminates.
                next_block_.store(blocks_, std::memory_order_relaxed);
                return;
            }
        }
    }

    void worker_loop() {
        // Deliberately a constant rather than a load of generation_. A worker
        // starts running some time after std::thread returns, so latching the
        // live counter here can happen *after* the first launch has already
        // bumped it -- the worker would then wait for the next generation while
        // run() waits for a decrement that never comes. Starting from 0 makes
        // generation 1 every worker's first job, whenever it actually wakes up.
        std::uint64_t seen = 0;
        for (;;) {
            constexpr int kSpins = 2000;
            int spins = 0;
            std::uint64_t current = generation_.load(std::memory_order_acquire);
            while (current == seen) {
                if (stopping_.load(std::memory_order_acquire)) return;
                if (++spins < kSpins) {
                    std::this_thread::yield();
                } else {
                    std::unique_lock<std::mutex> lock(sleep_mutex_);
                    sleepers_.fetch_add(1, std::memory_order_release);
                    sleep_signal_.wait_for(lock, std::chrono::milliseconds(2));
                    sleepers_.fetch_sub(1, std::memory_order_release);
                }
                current = generation_.load(std::memory_order_acquire);
            }
            seen = current;
            if (stopping_.load(std::memory_order_acquire)) return;
            consume();
            remaining_.fetch_sub(1, std::memory_order_release);
        }
    }

    std::vector<std::thread> threads_;
    std::once_flag started_;

    void (*body_)(void*, std::uint64_t) = nullptr;
    void* payload_ = nullptr;
    std::uint64_t blocks_ = 0;
    std::string* failure_sink_ = nullptr;
    std::mutex failure_mutex_;

    std::atomic<std::uint64_t> next_block_{0};
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<int> remaining_{0};
    std::atomic<int> sleepers_{0};
    std::atomic<bool> stopping_{false};

    std::mutex sleep_mutex_;
    std::condition_variable sleep_signal_;
};

LaunchPool g_pool;

}  // namespace

extern "C" {

// --- capability -----------------------------------------------------------

int colibri_cpu_backend_available() { return 1; }

int colibri_cpu_backend_kernel_count() {
    return static_cast<int>(colibri::cpu::kernel_count());
}

const char* colibri_cpu_kernel_name(std::uint64_t index) {
    return colibri::cpu::kernel_name(static_cast<std::size_t>(index));
}

long long colibri_cpu_kernel_index(const char* name) {
    if (name == nullptr) return -1;
    const std::size_t total = colibri::cpu::kernel_count();
    for (std::size_t index = 0; index < total; ++index) {
        const char* candidate = colibri::cpu::kernel_name(index);
        if (candidate != nullptr && std::strcmp(candidate, name) == 0)
            return static_cast<long long>(index);
    }
    return -1;
}

// --- backend selection ----------------------------------------------------

int colibri_backend_select(int backend) {
    if (backend == kColibriBackendCpu) {
        g_backend.store(kColibriBackendCpu, std::memory_order_relaxed);
        return 0;
    }
    if (backend == kColibriBackendCuda) {
        g_backend.store(kColibriBackendCuda, std::memory_order_relaxed);
        return 0;
    }
    return -1;
}

int colibri_backend_active() {
    return g_backend.load(std::memory_order_relaxed);
}

int colibri_backend_is_cpu() {
    return g_backend.load(std::memory_order_relaxed) == kColibriBackendCpu;
}

// --- memory ---------------------------------------------------------------

int colibri_cpu_alloc(std::uint64_t bytes, std::uint64_t* pointer) {
    if (pointer == nullptr) return -1;
    void* memory = aligned_allocate(static_cast<std::size_t>(bytes));
    if (memory == nullptr) return -2;
    *pointer = reinterpret_cast<std::uint64_t>(memory);
    return 0;
}

int colibri_cpu_free(std::uint64_t pointer) {
    if (pointer == 0) return 0;
    aligned_release(reinterpret_cast<void*>(pointer));
    return 0;
}

int colibri_cpu_host_alloc(std::uint64_t bytes, void** pointer) {
    if (pointer == nullptr) return -1;
    void* memory = aligned_allocate(static_cast<std::size_t>(bytes));
    if (memory == nullptr) return -2;
    *pointer = memory;
    return 0;
}

int colibri_cpu_host_free(void* pointer) {
    aligned_release(pointer);
    return 0;
}

// Page pinning exists so a GPU can DMA out of host memory. There is no device
// here, so registration must FAIL rather than trivially succeed.
//
// This is not pedantry. The runtime treats a successful registration as "direct
// expert paging is available" and switches routed experts to the hybrid device
// path, which on this backend copies expert weights out of the GGUF mapping
// into another host buffer and computes nothing faster for it. Measured on a
// 35B MoE: 0.43 tok/s with registration reported as succeeding, 5.67 tok/s with
// the CPU MoE path -- a 13x loss from one over-helpful return value.
int colibri_cpu_host_register(const void*, std::uint64_t) { return -1; }
int colibri_cpu_host_unregister(const void*) { return 0; }

int colibri_cpu_upload(std::uint64_t destination, const void* source,
                       std::uint64_t bytes, std::uint64_t) {
    if (destination == 0 || source == nullptr) return -1;
    std::memcpy(reinterpret_cast<void*>(destination), source,
                static_cast<std::size_t>(bytes));
    return 0;
}

int colibri_cpu_upload_sync(std::uint64_t destination, const void* source,
                            std::uint64_t bytes) {
    return colibri_cpu_upload(destination, source, bytes, 0);
}

int colibri_cpu_download(void* destination, std::uint64_t source,
                         std::uint64_t bytes, std::uint64_t) {
    if (destination == nullptr || source == 0) return -1;
    std::memcpy(destination, reinterpret_cast<const void*>(source),
                static_cast<std::size_t>(bytes));
    return 0;
}

int colibri_cpu_memset(std::uint64_t pointer, int value, std::uint64_t bytes,
                       std::uint64_t) {
    if (pointer == 0) return -1;
    std::memset(reinterpret_cast<void*>(pointer), value,
                static_cast<std::size_t>(bytes));
    return 0;
}

int colibri_cpu_sync() { return 0; }

// --- streams and events ---------------------------------------------------

int colibri_cpu_stream_create(std::uint64_t* stream) {
    if (stream == nullptr) return -1;
    const std::uint64_t handle = g_next_handle.fetch_add(1);
    std::lock_guard<std::mutex> lock(g_object_mutex);
    g_streams.emplace(handle, std::make_unique<Stream>());
    *stream = handle;
    return 0;
}

int colibri_cpu_stream_destroy(std::uint64_t stream) {
    std::lock_guard<std::mutex> lock(g_object_mutex);
    g_streams.erase(stream);
    return 0;
}

int colibri_cpu_stream_sync(std::uint64_t) { return 0; }

int colibri_cpu_event_create(std::uint64_t* event) {
    if (event == nullptr) return -1;
    const std::uint64_t handle = g_next_handle.fetch_add(1);
    std::lock_guard<std::mutex> lock(g_object_mutex);
    g_events.emplace(handle, std::make_unique<Event>());
    *event = handle;
    return 0;
}

int colibri_cpu_timed_event_create(std::uint64_t* event) {
    const int status = colibri_cpu_event_create(event);
    if (status != 0) return status;
    std::lock_guard<std::mutex> lock(g_object_mutex);
    g_events[*event]->timed = true;
    return 0;
}

int colibri_cpu_event_record(std::uint64_t event, std::uint64_t) {
    std::lock_guard<std::mutex> lock(g_object_mutex);
    const auto found = g_events.find(event);
    if (found == g_events.end()) return -1;
    found->second->stamp = std::chrono::steady_clock::now();
    found->second->recorded = true;
    return 0;
}

int colibri_cpu_event_sync(std::uint64_t) { return 0; }
int colibri_cpu_stream_wait_event(std::uint64_t, std::uint64_t) { return 0; }

int colibri_cpu_event_destroy(std::uint64_t event) {
    std::lock_guard<std::mutex> lock(g_object_mutex);
    g_events.erase(event);
    return 0;
}

int colibri_cpu_event_elapsed(std::uint64_t start, std::uint64_t end,
                              float* milliseconds) {
    if (milliseconds == nullptr) return -1;
    std::lock_guard<std::mutex> lock(g_object_mutex);
    const auto first = g_events.find(start);
    const auto second = g_events.find(end);
    if (first == g_events.end() || second == g_events.end()) return -1;
    if (!first->second->recorded || !second->second->recorded) return -1;
    const auto delta = second->second->stamp - first->second->stamp;
    *milliseconds =
        std::chrono::duration<float, std::milli>(delta).count();
    return 0;
}

// --- graphs ---------------------------------------------------------------

// CUDA graphs exist to amortize launch overhead, which host launches do not
// have. Capture is refused so the runtime keeps using the eager path; it treats
// a failed graph_begin as "graphs unavailable" and falls back.
int colibri_cpu_graph_begin(std::uint64_t) { return -1; }
int colibri_cpu_graph_end(std::uint64_t, std::uint64_t*) { return -1; }
int colibri_cpu_graph_launch(std::uint64_t, std::uint64_t) { return -1; }
int colibri_cpu_graph_destroy(std::uint64_t) { return 0; }

// --- launch ---------------------------------------------------------------

int colibri_cpu_launch_named(const char* name, std::uint32_t grid_x,
                             std::uint32_t grid_y, std::uint32_t block_x,
                             std::uint32_t shared_bytes, std::uint64_t stream,
                             void** arguments) {
    if (name == nullptr || arguments == nullptr || grid_x == 0 || grid_y == 0
        || block_x == 0) return -1;
    const Resolved resolved = resolve(name);
    if (resolved.kernel == nullptr) return -2;

    const bool profile = profiling_enabled();
    const auto started = profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    struct ProfileScope {
        bool active; const char* name; bool native;
        std::chrono::steady_clock::time_point started;
        ~ProfileScope() {
            if (!active) return;
            const auto elapsed = std::chrono::steady_clock::now() - started;
            record_launch(name, static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
                    .count()), native);
        }
    } scope{profile, name,
            resolved.native != nullptr &&
                !g_force_emulation.load(std::memory_order_relaxed),
            started};

    // A native kernel owns the whole launch; the grid is advisory. Emulation is
    // the reference path and is only forced by the parity harness.
    if (resolved.native != nullptr &&
        !g_force_emulation.load(std::memory_order_relaxed)) {
        const colibri::cpu::Launch launch{grid_x, grid_y, block_x, shared_bytes,
                                          stream};
        try {
            resolved.native(launch, arguments);
        } catch (const std::exception&) {
            return -3;
        }
        return 0;
    }

    LaunchPayload payload{resolved.kernel, arguments};
    const std::uint64_t blocks =
        static_cast<std::uint64_t>(grid_x) * static_cast<std::uint64_t>(grid_y);

    // Barrier-free kernels -- roughly a third of the corpus, including the KV
    // stores, the elementwise ops, and the single-token DeltaNet steps -- never
    // suspend, so a block is just a loop over thread indices. Skipping the
    // fibers avoids the per-block stack working set that otherwise dominates
    // everything: measured ~2ms/block with fibers against ~20us without.
    const auto set_geometry = [&](std::uint64_t block) {
        colibri::cpu::t_grid_dim = {grid_x, grid_y, 1};
        colibri::cpu::t_block_dim = {block_x, 1, 1};
        colibri::cpu::t_block_index = {
            static_cast<unsigned int>(block % grid_x),
            static_cast<unsigned int>(block / grid_x), 0};
    };

    const auto run_block = [&](colibri::cpu::BlockScheduler& scheduler,
                               std::uint64_t block) {
        set_geometry(block);
        // New block, so any __shared__ it touches must look freshly allocated.
        ++colibri::cpu::t_block_generation;
        if (resolved.cooperative) {
            scheduler.run(block_x, shared_bytes, &run_thread, &payload);
            return;
        }
        scheduler.prepare_direct(block_x, shared_bytes);
        for (unsigned int thread = 0; thread < block_x; ++thread) {
            colibri::cpu::t_thread_index = {thread, 0, 0};
            resolved.kernel(arguments);
        }
    };

    // run_block is passed as an opaque pointer to its closure, so distributing
    // a launch across the pool costs no allocation. The closure captures locals
    // of this function by reference, which outlive the call below.
    std::string failure;
    g_pool.run(
        blocks,
        [](void* opaque, std::uint64_t block) {
            // One scheduler per worker, reused across blocks and launches: the
            // fiber stacks are the expensive part of a cooperative block and
            // there is no reason to rebuild them.
            static thread_local colibri::cpu::BlockScheduler scheduler;
            (*static_cast<const decltype(run_block)*>(opaque))(scheduler, block);
        },
        const_cast<void*>(static_cast<const void*>(&run_block)), &failure);

    if (!failure.empty()) return -3;
    return 0;
}

}  // extern "C"

namespace colibri::cpu {

bool register_native_kernel(const char* name, NativeKernel kernel) {
    if (name == nullptr || kernel == nullptr) return false;
    // Refuse names the corpus does not define: without this a typo would leave
    // the emulated kernel silently serving production traffic.
    if (find_kernel(name) == nullptr) return false;
    native_registry()[name] = kernel;
    return true;
}

NativeKernel find_native_kernel(const char* name) {
    const auto& registry = native_registry();
    const auto found = registry.find(name);
    return found == registry.end() ? nullptr : found->second;
}

void parallel_for(std::uint64_t count, void (*body)(void*, std::uint64_t),
                  void* payload) {
    std::string failure;
    g_pool.run(count, body, payload, &failure);
}

void set_force_emulation(bool force) {
    g_force_emulation.store(force, std::memory_order_relaxed);
    // Resolution is cached per name, so the flag has to invalidate it.
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    g_kernel_cache.clear();
}

bool force_emulation() {
    return g_force_emulation.load(std::memory_order_relaxed);
}

NativeKernelRegistration::NativeKernelRegistration(const char* name,
                                                   NativeKernel kernel) {
    register_native_kernel(name, kernel);
}

}  // namespace colibri::cpu

extern "C" COLIBRI_BACKEND_API void colibri_cpu_profile_dump() {
    if (!profiling_enabled()) return;
    std::vector<std::pair<std::string, ProfileEntry>> rows;
    {
        std::lock_guard<std::mutex> lock(g_profile_mutex);
        rows.assign(g_profile.begin(), g_profile.end());
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second.nanoseconds > b.second.nanoseconds;
    });
    std::uint64_t total = 0;
    for (const auto& row : rows) total += row.second.nanoseconds;
    if (total == 0) return;

    std::fprintf(stderr,
        "\n[colibri-cpu] launch profile (%.1f ms total)\n", total / 1e6);
    std::fprintf(stderr, "  %-38s %8s %10s %7s %8s  %s\n",
                 "kernel", "calls", "total ms", "share", "us/call", "impl");
    double cumulative = 0.0;
    for (const auto& row : rows) {
        const double share = 100.0 * row.second.nanoseconds / total;
        cumulative += share;
        std::fprintf(stderr, "  %-38s %8llu %10.3f %6.1f%% %8.1f  %s\n",
            row.first.c_str(),
            static_cast<unsigned long long>(row.second.calls),
            row.second.nanoseconds / 1e6, share,
            row.second.nanoseconds / 1e3 / static_cast<double>(row.second.calls),
            row.second.native ? "native" : "emulated");
        if (cumulative > 99.0) break;
    }
}
