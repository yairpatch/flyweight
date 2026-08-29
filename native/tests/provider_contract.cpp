#include "flyweight_v2_provider.hpp"

#include <cstring>
#include <vector>

namespace {
class SyntheticProvider final : public flyweight::v2::WeightProvider {
public:
    SyntheticProvider() : bytes{1, 2, 3, 4} {
        descriptor.name = "synthetic.weight";
        descriptor.shape = {2, 2};
        descriptor.type = 0;
        descriptor.size = bytes.size();
    }

    const char* format() const override { return "synthetic"; }
    std::uint64_t tensor_count() const override { return 1; }
    const flyweight::v2::TensorDescriptor* tensor(std::uint64_t index) const override {
        return index == 0 ? &descriptor : nullptr;
    }
    int read_tensor(std::uint64_t index, void* destination,
                    std::uint64_t size) const override {
        if (index != 0 || destination == nullptr || size < bytes.size()) return -1;
        std::memcpy(destination, bytes.data(), bytes.size());
        return 0;
    }

private:
    flyweight::v2::TensorDescriptor descriptor;
    std::vector<unsigned char> bytes;
};
}  // namespace

int main() {
    SyntheticProvider provider;
    unsigned char output[4]{};
    if (std::strcmp(provider.format(), "synthetic") != 0
        || provider.tensor_count() != 1
        || provider.tensor(0) == nullptr
        || provider.read_tensor(0, output, sizeof(output)) != 0
        || output[0] != 1 || output[3] != 4) return 1;
    return 0;
}
