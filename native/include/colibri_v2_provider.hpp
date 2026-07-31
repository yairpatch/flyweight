#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace colibri::v2 {

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

// Format-neutral weight contract. Execution code should depend on this
// interface rather than file headers, metadata encodings, or mmap details.
class WeightProvider {
public:
    virtual ~WeightProvider() = default;
    virtual const char* format() const = 0;
    virtual std::uint64_t tensor_count() const = 0;
    virtual const TensorDescriptor* tensor(std::uint64_t index) const = 0;
    virtual int read_tensor(std::uint64_t index, void* destination,
                            std::uint64_t bytes) const = 0;
};

}  // namespace colibri::v2
