#pragma once

// Host CPU topology for sizing the OpenMP teams.
//
// Two counts, because the two phases want different teams on a hybrid part. A
// decode step is bandwidth-bound and every OpenMP phase waits for its slowest
// thread, so a team that spills onto efficiency cores is paced by them. Prefill
// is compute-bound and takes every core it can get. Measured on an i9-13980HX
// (8 P-cores + 16 E-cores, 32 threads) with a 35B-A3B Q4_K_S MoE, experts on
// CPU:
//
//   team                              decode tok/s   prefill tok/s
//   16, unpinned (logical / 2)            23.8            120
//   8, the P-cores                        30.1             87
//   24, every physical core               27.8            124
//
// The old rule -- logical processors / 2 -- is exact on uniform SMT parts but
// counts 16 cores on that 24-core part. Both counts are taken inside the
// process affinity mask, so taskset still scopes a run.

#include <algorithm>
#include <cstdio>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
#endif

namespace flyweight::cpu_topology {

struct Topology {
    int physical = 0;     // physical cores the process may run on
    int performance = 0;  // of those, the cores of the fastest class
    // Linux only: the logical processors of each fast core, in core order.
    // Empty where the platform does not say, which leaves pinning off.
    std::vector<std::vector<int>> performance_cores;
};

namespace detail {

#if defined(__linux__)
// A sysfs cpu list ("0-15,32,34-35") as a membership vector; empty when absent.
inline std::vector<bool> read_cpu_list(const char* path) {
    std::vector<bool> cpus;
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) return cpus;
    char line[4096] = {};
    const bool read = std::fgets(line, sizeof(line), file) != nullptr;
    std::fclose(file);
    if (!read) return cpus;
    const auto number = [](const char*& c) {
        int value = 0;
        while (*c >= '0' && *c <= '9') value = value * 10 + (*c++ - '0');
        return value;
    };
    for (const char* c = line; *c >= '0' && *c <= '9';) {
        const int first = number(c);
        int last = first;
        if (*c == '-') {
            ++c;
            last = number(c);
        }
        if (last >= static_cast<int>(cpus.size()))
            cpus.resize(static_cast<std::size_t>(last) + 1, false);
        for (int cpu = first; cpu <= last; ++cpu) cpus[static_cast<std::size_t>(cpu)] = true;
        if (*c == ',') ++c;
    }
    return cpus;
}

inline long read_number(const char* path) {
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) return -1;
    long value = -1;
    if (std::fscanf(file, "%ld", &value) != 1) value = -1;
    std::fclose(file);
    return value;
}

inline Topology detect() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return {};
    // The fast class. Intel's hybrid PMU lists its P-cores directly; elsewhere
    // ACPI CPPC's highest_perf ranks the cores (69 for a Raptor Lake P-core, 40
    // for an E-core), and anything within 20% of the best is fast -- which also
    // absorbs the few-percent "preferred core" spread on uniform parts.
    // cpu_capacity is no help: x86 kernels report 1024 for every core.
    const std::vector<bool> intel_performance = read_cpu_list("/sys/devices/cpu_core/cpus");
    const bool intel_hybrid = !intel_performance.empty()
        && !read_cpu_list("/sys/devices/cpu_atom/cpus").empty();
    struct Core {
        int id;                 // first logical processor of the core
        long rank;              // higher is faster; -1 when unknown
        std::vector<int> cpus;  // its logical processors inside the mask
    };
    std::vector<Core> cores;
    char path[128];
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        const std::vector<bool> siblings = read_cpu_list(path);
        const auto first = std::find(siblings.begin(), siblings.end(), true);
        const int id = first == siblings.end() ? cpu : static_cast<int>(first - siblings.begin());
        const auto known =
            std::find_if(cores.begin(), cores.end(), [id](const Core& core) { return core.id == id; });
        if (known != cores.end()) {
            known->cpus.push_back(cpu);
            continue;
        }
        long rank = -1;
        if (intel_hybrid) {
            rank = cpu < static_cast<int>(intel_performance.size())
                && intel_performance[static_cast<std::size_t>(cpu)] ? 1 : 0;
        } else {
            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/acpi_cppc/highest_perf", cpu);
            rank = read_number(path);
        }
        cores.push_back({id, rank, {cpu}});
    }
    if (cores.empty()) return {};
    long best = 0;
    for (const auto& core : cores) best = std::max(best, core.rank);
    Topology topology;
    topology.physical = static_cast<int>(cores.size());
    for (const auto& core : cores)
        if (best <= 0 || core.rank * 5 >= best * 4)
            topology.performance_cores.push_back(core.cpus);
    topology.performance = static_cast<int>(topology.performance_cores.size());
    return topology;
}
#elif defined(_WIN32)
inline Topology detect() {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length == 0) return {};
    std::vector<char> buffer(length);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length))
        return {};
    // GetProcessAffinityMask describes one processor group, so the mask only
    // filters on single-group machines; with several groups every core counts.
    DWORD_PTR process_mask = 0, system_mask = 0;
    const bool filter = GetActiveProcessorGroupCount() == 1
        && GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);
    // EfficiencyClass is 0 on uniform parts and grows with core performance.
    std::vector<int> classes;
    for (DWORD offset = 0; offset < length;) {
        const auto* info =
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        const auto& core = info->Processor;
        if (!filter || (core.GroupMask[0].Mask & process_mask) != 0)
            classes.push_back(core.EfficiencyClass);
        offset += info->Size;
    }
    if (classes.empty()) return {};
    const int best = *std::max_element(classes.begin(), classes.end());
    Topology topology;
    topology.physical = static_cast<int>(classes.size());
    topology.performance = static_cast<int>(std::count(classes.begin(), classes.end(), best));
    return topology;
}
#else
inline Topology detect() { return {}; }
#endif

}  // namespace detail

inline const Topology& host() {
    static const Topology topology = [] {
        Topology detected = detail::detect();
        if (detected.physical <= 0) {
            // Undetected: assume SMT, the rule this header replaces.
            const unsigned logical = std::thread::hardware_concurrency();
            detected.physical = logical >= 2 ? static_cast<int>(logical / 2) : 1;
        }
        if (detected.performance <= 0) detected.performance = detected.physical;
        return detected;
    }();
    return topology;
}

// Team for compute-bound batch work: prefill chunks, multi-row expert sweeps.
inline int batch_threads() { return host().physical; }

// Team for one bandwidth-bound decode step: on a hybrid part, the fast cores
// alone once there are enough of them to load the memory bus by themselves. A
// compute-bound step (low-bit codebook experts) wants batch_threads() instead;
// the caller knows which it is running. The four-core floor
// (below it, as on a 2P+8E ultrabook, the efficiency cores are kept) is a
// judgement, not a measurement -- only the 8P+16E part above was measured.
inline int decode_threads() {
    const auto& topology = host();
    return topology.performance >= 4 ? topology.performance : topology.physical;
}

}  // namespace flyweight::cpu_topology
