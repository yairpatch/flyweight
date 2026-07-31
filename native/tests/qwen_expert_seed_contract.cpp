#include "colibri_v2_expert_seed.hpp"

#include <cassert>
#include <cstdint>
#include <limits>

int main() {
    using namespace colibri::v2::expert_seed;

    assert(auto_experts_per_layer(383, 80) == 4);
    assert(auto_experts_per_layer(3, 80) == 0);
    assert(auto_experts_per_layer(8, 2) == 4);
    assert(auto_experts_per_layer(8, 0) == 0);

    assert(score(3, 2) == 26);
    assert(score(0, 9) == 9);
    assert(score(std::numeric_limits<std::uint32_t>::max(),
                 std::numeric_limits<std::uint32_t>::max()) >
           std::numeric_limits<std::uint32_t>::max());

    assert(!should_seed(31ull * 80 * 8, 80, 8, 7));
    assert(should_seed(32ull * 80 * 8, 80, 8, 0));
    assert(should_seed(0, 80, 8, 8));
    assert(!should_seed(0, 80, 8, 7));
    assert(!has_useful_prompt(31ull * 80 * 8, 80, 8));
    assert(has_useful_prompt(32ull * 80 * 8, 80, 8));
    return 0;
}
