#pragma once

// Host implementations of the device primitives, declared for gpu_driver.cpp.
// Signatures mirror their flyweight_gpu_* counterparts in flyweight_gpu_driver.h;
// see cpu_backend.cpp for the semantics (notably: every launch is synchronous,
// which is what makes the stream and event entry points trivial).

#include <cstdint>

extern "C" {

int flyweight_cpu_backend_available();
int flyweight_cpu_backend_kernel_count();

int flyweight_cpu_sync();
int flyweight_cpu_alloc(std::uint64_t bytes, std::uint64_t* pointer);
int flyweight_cpu_free(std::uint64_t pointer);
int flyweight_cpu_host_alloc(std::uint64_t bytes, void** pointer);
int flyweight_cpu_host_free(void* pointer);
int flyweight_cpu_host_register(const void* pointer, std::uint64_t bytes);
int flyweight_cpu_host_unregister(const void* pointer);
int flyweight_cpu_upload(std::uint64_t destination, const void* source,
                       std::uint64_t bytes, std::uint64_t stream);
int flyweight_cpu_upload_sync(std::uint64_t destination, const void* source,
                            std::uint64_t bytes);
int flyweight_cpu_download(void* destination, std::uint64_t source,
                         std::uint64_t bytes, std::uint64_t stream);
int flyweight_cpu_copy_device(std::uint64_t destination, std::uint64_t source,
                            std::uint64_t bytes, std::uint64_t stream);
int flyweight_cpu_memset(std::uint64_t pointer, int value, std::uint64_t bytes,
                       std::uint64_t stream);

int flyweight_cpu_stream_create(std::uint64_t* stream);
int flyweight_cpu_stream_destroy(std::uint64_t stream);
int flyweight_cpu_stream_sync(std::uint64_t stream);
int flyweight_cpu_stream_wait_event(std::uint64_t stream, std::uint64_t event);

int flyweight_cpu_graph_begin(std::uint64_t stream);
int flyweight_cpu_graph_end(std::uint64_t stream, std::uint64_t* graph);
int flyweight_cpu_graph_launch(std::uint64_t graph, std::uint64_t stream);
int flyweight_cpu_graph_destroy(std::uint64_t graph);

int flyweight_cpu_event_create(std::uint64_t* event);
int flyweight_cpu_timed_event_create(std::uint64_t* event);
int flyweight_cpu_event_record(std::uint64_t event, std::uint64_t stream);
int flyweight_cpu_event_sync(std::uint64_t event);
int flyweight_cpu_event_destroy(std::uint64_t event);
int flyweight_cpu_event_elapsed(std::uint64_t start, std::uint64_t end,
                              float* milliseconds);

int flyweight_cpu_launch_named(const char* name, std::uint32_t grid_x,
                             std::uint32_t grid_y, std::uint32_t block_x,
                             std::uint32_t shared_bytes, std::uint64_t stream,
                             void** arguments);

// Name for a registry index, or null when out of range. Lets gpu_driver.cpp
// recover a kernel name from a CPU sentinel handle.
const char* flyweight_cpu_kernel_name(std::uint64_t index);

// Registry index for a name, or a negative value when the corpus has no such
// kernel. Used to build the sentinel handles at compile time.
long long flyweight_cpu_kernel_index(const char* name);

}  // extern "C"
