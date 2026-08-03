// Fiber scheduler backing the CUDA-to-host shim. See colibri_cpu_shim.hpp for
// the execution model and why fibers rather than threads.

#include <colibri_cpu_shim.hpp>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace colibri::cpu {

thread_local Dim3 t_thread_index{};
thread_local Dim3 t_block_index{};
thread_local Dim3 t_block_dim{};
thread_local Dim3 t_grid_dim{};
thread_local BlockScheduler* t_scheduler = nullptr;
thread_local std::uint64_t t_block_generation = 0;

bool shared_zeroing_enabled() {
    static const bool on = [] {
        const char* setting = std::getenv("COLIBRI_CPU_SHARED_ZERO");
        return setting == nullptr || setting[0] != '0';  // default on
    }();
    return on;
}

void shared_zero_once(void* storage, std::size_t bytes) {
    if (!shared_zeroing_enabled()) return;
    // Keyed by address: each __shared__ declaration has a distinct
    // thread_local instance, so the address identifies the declaration site.
    static thread_local std::unordered_map<const void*, std::uint64_t> seen;
    auto& generation = seen[storage];
    if (generation == t_block_generation) return;
    generation = t_block_generation;
    std::memset(storage, 0, bytes);
}

namespace {

// The single most performance-sensitive constant in the backend. A block's
// fiber stacks are touched in round-robin order, so the block's working set is
// this times the block size -- 256 threads at 256 KiB was 64 MB per block and
// cost ~2ms even for a kernel that never synchronizes. At 32 KiB it is 8 MB.
//
// The floor is set by the deepest kernel frame: the corpus itself stays shallow
// (the IQ dequant helpers are the worst), and the block-wide sort was moved off
// the stack in cpu_kernels.cpp specifically so it would not set this budget.
// Kernels that cannot reach a barrier bypass fibers entirely and are unaffected.
constexpr std::size_t kFiberStackBytes = 32u * 1024u;

struct FiberEntry {
    void (*body)(void*) = nullptr;
    void* payload = nullptr;
    BlockScheduler* scheduler = nullptr;
    unsigned int thread = 0;
};

thread_local FiberEntry t_entry{};

void fiber_trampoline();

}  // namespace

void BlockScheduler::run(unsigned int threads, unsigned int shared_bytes,
                         void (*body)(void*), void* payload) {
    if (threads == 0) return;

    thread_count_ = threads;
    if (shared_.size() < shared_bytes) shared_.resize(shared_bytes);
    if (exchange_.size() < threads) exchange_.resize(threads);

    const unsigned int warps = (threads + kWarpSize - 1) / kWarpSize;
    warp_arrived_.assign(warps, 0);
    warp_live_.assign(warps, 0);
    for (unsigned int warp = 0; warp < warps; ++warp)
        warp_live_[warp] = std::min<unsigned int>(kWarpSize, threads - warp * kWarpSize);
    block_arrived_ = 0;
    block_live_ = threads;

    if (fibers_.size() < threads) fibers_.resize(threads);

    BlockScheduler* const previous_scheduler = t_scheduler;
    t_scheduler = this;

#if defined(_WIN32)
    // Returns null when the OS thread is already a fiber, which happens for the
    // worker threads the backend keeps alive across launches.
    void* const converted = ConvertThreadToFiber(nullptr);
    scheduler_handle_ = converted != nullptr ? converted : GetCurrentFiber();
#endif

    for (unsigned int thread = 0; thread < threads; ++thread) {
        Fiber& fiber = fibers_[thread];
        fiber.alive = true;
        fiber.releasable = false;
        fiber.waiting = Waiting::none;

        // The trampoline copies t_entry on entry, so each fiber has to be
        // started before the next assignment overwrites it.
        t_entry = FiberEntry{body, payload, this, thread};
#if defined(_WIN32)
        if (fiber.handle != nullptr) DeleteFiber(fiber.handle);
        fiber.handle = CreateFiber(kFiberStackBytes,
                                   [](void*) { fiber_trampoline(); }, nullptr);
        if (fiber.handle == nullptr) throw std::runtime_error("cannot create block fiber");
#else
        if (fiber.stack.size() != kFiberStackBytes) fiber.stack.resize(kFiberStackBytes);
        if (getcontext(&fiber.context) != 0)
            throw std::runtime_error("cannot capture fiber context");
        fiber.context.uc_stack.ss_sp = fiber.stack.data();
        fiber.context.uc_stack.ss_size = fiber.stack.size();
        fiber.context.uc_link = &scheduler_context_;
        makecontext(&fiber.context,
                    reinterpret_cast<void (*)()>(&fiber_trampoline), 0);
#endif
        current_ = thread;
#if defined(_WIN32)
        SwitchToFiber(fiber.handle);
#else
        swapcontext(&scheduler_context_, &fiber.context);
#endif
    }

    // Every fiber has now either finished or parked on a barrier. Resume the
    // ones a barrier has released, repeatedly, until the block is done.
    while (block_live_ > 0) {
        bool resumed = false;
        for (unsigned int thread = 0; thread < threads; ++thread) {
            Fiber& fiber = fibers_[thread];
            if (!fiber.alive || !fiber.releasable) continue;
            fiber.releasable = false;
            fiber.waiting = Waiting::none;
            current_ = thread;
            resumed = true;
#if defined(_WIN32)
            SwitchToFiber(fiber.handle);
#else
            swapcontext(&scheduler_context_, &fiber.context);
#endif
        }
        if (!resumed) {
            // Live fibers with nothing releasable means the kernel's barriers
            // cannot be satisfied -- divergent __syncthreads, which is invalid
            // on device too. Fail loudly rather than hang.
            throw std::runtime_error(
                "CPU kernel deadlocked on a divergent block barrier");
        }
    }

#if defined(_WIN32)
    for (unsigned int thread = 0; thread < threads; ++thread) {
        if (fibers_[thread].handle != nullptr) {
            DeleteFiber(fibers_[thread].handle);
            fibers_[thread].handle = nullptr;
        }
    }
#endif
    t_scheduler = previous_scheduler;
}

void BlockScheduler::prepare_direct(unsigned int threads,
                                    unsigned int shared_bytes) {
    thread_count_ = threads;
    if (shared_.size() < shared_bytes) shared_.resize(shared_bytes);
    if (exchange_.size() < threads) exchange_.resize(threads);
    // No fibers, no barrier accounting: a direct-mode kernel cannot reach one.
    t_scheduler = this;
}

void BlockScheduler::park(Waiting reason) {
    Fiber& fiber = fibers_[current_];
    fiber.waiting = reason;
    fiber.releasable = false;

    const Dim3 saved_thread = t_thread_index;
#if defined(_WIN32)
    SwitchToFiber(scheduler_handle_);
#else
    swapcontext(&fiber.context, &scheduler_context_);
#endif
    // Resumed on the same OS thread, but other fibers have run since and left
    // their own indices in the thread-locals.
    t_thread_index = saved_thread;
    t_scheduler = this;
}

void BlockScheduler::release_block_waiters() {
    block_arrived_ = 0;
    for (unsigned int thread = 0; thread < thread_count_; ++thread) {
        Fiber& fiber = fibers_[thread];
        if (fiber.alive && fiber.waiting == Waiting::block) fiber.releasable = true;
    }
}

void BlockScheduler::release_warp_waiters(unsigned int warp) {
    warp_arrived_[warp] = 0;
    const unsigned int base = warp * kWarpSize;
    const unsigned int end = std::min(base + kWarpSize, thread_count_);
    for (unsigned int thread = base; thread < end; ++thread) {
        Fiber& fiber = fibers_[thread];
        if (fiber.alive && fiber.waiting == Waiting::warp) fiber.releasable = true;
    }
}

void BlockScheduler::block_barrier() {
    if (block_live_ <= 1) return;
    if (++block_arrived_ >= block_live_) {
        // Last arrival releases the others and falls through itself.
        release_block_waiters();
        return;
    }
    park(Waiting::block);
}

void BlockScheduler::warp_barrier() {
    const unsigned int warp = t_thread_index.x / kWarpSize;
    if (warp >= warp_live_.size() || warp_live_[warp] <= 1) return;
    if (++warp_arrived_[warp] >= warp_live_[warp]) {
        release_warp_waiters(warp);
        return;
    }
    park(Waiting::warp);
}

void BlockScheduler::settle_barriers() {
    // A lane leaving can be the arrival the survivors were waiting on.
    if (block_live_ > 0 && block_arrived_ >= block_live_) release_block_waiters();
    for (unsigned int warp = 0; warp < warp_live_.size(); ++warp) {
        if (warp_live_[warp] > 0 && warp_arrived_[warp] >= warp_live_[warp])
            release_warp_waiters(warp);
    }
}

void BlockScheduler::retire(unsigned int thread) {
    fibers_[thread].alive = false;
    fibers_[thread].releasable = false;
    fibers_[thread].waiting = Waiting::none;
    if (block_live_ > 0) --block_live_;
    const unsigned int warp = thread / kWarpSize;
    if (warp < warp_live_.size() && warp_live_[warp] > 0) --warp_live_[warp];
    settle_barriers();
}

void BlockScheduler::return_to_scheduler() {
#if defined(_WIN32)
    SwitchToFiber(scheduler_handle_);
#endif
}

namespace {

void fiber_trampoline() {
    const FiberEntry entry = t_entry;
    BlockScheduler* const scheduler = entry.scheduler;

    t_scheduler = scheduler;
    t_thread_index = Dim3{entry.thread, 0, 0};

    entry.body(entry.payload);

    // Lane exit: stop counting toward barriers so survivors are not waiting on
    // a thread that has already returned.
    scheduler->retire(entry.thread);

#if defined(_WIN32)
    scheduler->return_to_scheduler();
#endif
    // POSIX returns to uc_link, which is the scheduler context.
}

}  // namespace

}  // namespace colibri::cpu
