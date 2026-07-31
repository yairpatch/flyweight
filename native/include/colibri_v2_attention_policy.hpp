#pragma once

#include <cstdint>

namespace colibri::v2::attention {

// On the RTX 5070 Ti reference system, cuBLAS is already 4.6% faster at 128
// tokens and its advantage grows with context. Keep the very small fused path
// for launch-sensitive devices while avoiding its linear slowdown thereafter.
inline constexpr std::int32_t kDefaultCublasMinTokens = 128;

constexpr bool cublas_eligible(
    std::int32_t cache_type_k,
    std::int32_t cache_type_v,
    std::int32_t tokens,
    std::int32_t first_slot,
    std::int32_t capacity,
    bool enabled,
    std::int32_t minimum_tokens
) {
    return enabled &&
        cache_type_k == 1 &&
        cache_type_v == 1 &&
        tokens >= minimum_tokens &&
        first_slot >= 0 &&
        first_slot + tokens <= capacity;
}

}  // namespace colibri::v2::attention
