#pragma once

#include <cstdint>

namespace flyweight::v2::attention {

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

// KV cache precision codes. `auto` is not a codec -- it is resolved to a
// concrete code at prepare time by auto_cache_type below.
inline constexpr std::int32_t kCacheTypeF16 = 1;
inline constexpr std::int32_t kCacheTypeTurbo4 = 5;
inline constexpr std::int32_t kCacheTypeAuto = 6;

// Above this context the VRAM that KV compression frees is worth more as
// routed-expert slots than as KV precision. Measured on Qwen3.6-35B-A3B-Q6_K
// (12 GB): turbo4 holds expert compute flat (~4.4-4.8 ms/token) at every
// context while f16 gives up slots as context grows -- expert hit rate is
// 68.2%/70.1% at 32K, 55.7%/67.4% at 128K, and 27.2%/63.6% at 256K, where f16
// is down to 277 slots. At or below 32K the gain is under the measurement
// noise floor, so f16 stays and short-context output is unchanged.
inline constexpr std::uint64_t kTurboAutoContextThreshold = 32768;

// `head_dim_ok` must be false if ANY attention layer has a head_dim outside
// [32, 512] or not a power of two: the TurboQuant rotation is a Walsh-Hadamard
// butterfly over the whole head and the runtime throws on such geometry, which
// is acceptable for an explicit flag but never for a default.
//
// `has_routed_experts` gates on the win existing at all. The entire benefit is
// KV bytes converting into expert slots, so a dense checkpoint gains nothing
// while still paying for turbo's staging buffer -- and on
// Qwen3.6-27B-UD-IQ2_XXS at 64K that extra allocation turns a config that
// works under f16 into a hard prepare failure.
constexpr std::int32_t auto_cache_type(
    std::uint64_t context_limit,
    bool has_routed_experts,
    bool head_dim_ok
) {
    return context_limit > kTurboAutoContextThreshold &&
        has_routed_experts && head_dim_ok
        ? kCacheTypeTurbo4
        : kCacheTypeF16;
}

constexpr bool turbo_head_dim_ok(std::int64_t head_dim) {
    return head_dim >= 32 && head_dim <= 512 &&
        (head_dim & (head_dim - 1)) == 0;
}

}  // namespace flyweight::v2::attention
