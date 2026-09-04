#include "flyweight_v2_attention_policy.hpp"

namespace attention = flyweight::v2::attention;

int main() {
    constexpr auto threshold = attention::kDefaultCublasMinTokens;
    if (attention::cublas_eligible(1, 1, threshold - 1, 0, 8192, true, threshold))
        return 1;
    if (!attention::cublas_eligible(1, 1, threshold, 0, 8192, true, threshold))
        return 2;
    if (attention::cublas_eligible(0, 1, threshold, 0, 8192, true, threshold))
        return 3;
    if (attention::cublas_eligible(1, 1, threshold, 0, 8192, false, threshold))
        return 4;
    if (attention::cublas_eligible(
            1, 1, threshold, 8192 - threshold + 1, 8192, true, threshold))
        return 5;
    if (!attention::cublas_eligible(1, 1, 1024, 0, 8192, true, 1024))
        return 6;

    // bf16 reads in place through the tensor cores exactly as f16 does -- the
    // GEMM takes the element type as an argument -- so it is eligible too.
    if (!attention::cublas_eligible(2, 2, threshold, 0, 8192, true, threshold))
        return 13;
    // A quantized cache is not: cuBLAS cannot address it, and staging it to f16
    // first measured slower than reading the quantized rows directly (q8 at
    // 49k: 547 us staged against 511 us) as well as costing scratch VRAM.
    if (attention::cublas_eligible(3, 3, threshold, 0, 8192, true, threshold))
        return 14;
    // A mismatched pair has no single element type to hand the GEMM.
    if (attention::cublas_eligible(2, 1, threshold, 0, 8192, true, threshold) ||
        attention::cublas_eligible(1, 2, threshold, 0, 8192, true, threshold))
        return 15;

    // Cache-type `auto` resolution. Each guard below corresponds to a measured
    // failure mode, not a style preference -- see the header for the numbers.
    constexpr auto ctx = attention::kTurboAutoContextThreshold;
    // At or below the threshold the win is under the noise floor, so f16 stays
    // and short-context output is unchanged.
    if (attention::auto_cache_type(ctx, true, true) != attention::kCacheTypeF16)
        return 7;
    if (attention::auto_cache_type(ctx + 1, true, true) !=
        attention::kCacheTypeTurbo4)
        return 8;
    // A dense checkpoint has no expert slots to gain and turbo still costs its
    // staging buffer; on the 27B at 64K that turns a working config into a
    // prepare failure.
    if (attention::auto_cache_type(ctx * 4, false, true) !=
        attention::kCacheTypeF16)
        return 9;
    // Ineligible head_dim must never be auto-selected: the runtime throws.
    if (attention::auto_cache_type(ctx * 4, true, false) !=
        attention::kCacheTypeF16)
        return 10;
    // Walsh-Hadamard needs a power of two within [32, 512].
    if (!attention::turbo_head_dim_ok(128) ||
        !attention::turbo_head_dim_ok(256) ||
        !attention::turbo_head_dim_ok(512) ||
        !attention::turbo_head_dim_ok(32))
        return 11;
    if (attention::turbo_head_dim_ok(96) || attention::turbo_head_dim_ok(80) ||
        attention::turbo_head_dim_ok(1024) || attention::turbo_head_dim_ok(16) ||
        attention::turbo_head_dim_ok(0))
        return 12;
    return 0;
}
