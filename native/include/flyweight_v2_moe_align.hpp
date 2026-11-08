#pragma once

#include <cstdint>
#include <vector>

// Sorting routes by expert into block-aligned runs -- the layout a fused MoE
// kernel consumes.
//
// WHY THIS EXISTS. The rows forward has two GPU expert paths and each pays a
// cost the other avoids (plans/decode-device-dispatch.md, "The prefill
// bottleneck"):
//
//   - the grouped table path launches once for every route and re-reads an
//     expert's weights per routed token;
//   - the streaming path reads an expert's weights once but issues ~8 launches
//     per expert -- 153,440 launches for a 784-token prompt, at 5 blocks each,
//     which cannot fill the device.
//
// Neither is "one launch, weights read once per expert-block". Getting there
// needs the routes grouped by expert AND padded to a block multiple, so a single
// kernel can walk blocks and read its expert id from a table instead of the host
// issuing a launch per expert. This is the same shape as vLLM's
// `moe_align_block_size` and llama.cpp's grouped `mul_mat_id`.
//
// This header is the host reference and the production builder both: it is small,
// exact, and pinned by native/tests/qwen_moe_align_contract.cpp. A device version
// can replace it later without changing the layout it defines.

namespace flyweight::v2::moe {

// `sorted_routes` holds route indices (token * top_k + rank) grouped by expert,
// each expert's run padded up to `block_size` with kEmpty. `block_experts` gives
// the owning expert of each block, so block b covers
// sorted_routes[b*block_size .. (b+1)*block_size). `padded_total` is the used
// length of `sorted_routes`; anything past it is not written.
inline constexpr std::int32_t kEmpty = -1;

// Tokens per block. MUST match FLYWEIGHT_MOE_BLOCK in flyweight_v2_qwen_kernels.hpp,
// which sizes the kernel's per-token accumulators. 8 by measurement: at this
// model's ~20 routes per expert the padding waste is 14.4% here against 37.6% at
// 32 and 68.8% at 64 (plans/decode-device-dispatch.md).
inline constexpr int kBlockSize = 8;

// Tokens per block for the routed MMQ path, which consumes the same layout with
// a different tile. MUST match FLYWEIGHT_MOE_MMQ_TOKENS in the kernel corpus.
//
// 32 rather than kBlockSize's 8 because the two kernels are bound by different
// things. The octet kernels decode a weight row in scalar float, so padding is
// what hurts them (14.4% at 8, 37.6% at 32). The MMQ tile decodes into shared
// memory and reuses it through tensor-core MMAs, so the padded lanes are nearly
// free and what matters is decoding an expert's rows once for 32 tokens instead
// of once for 8 -- 2.9x fewer decodes at this model's ~20 routes per expert.
inline constexpr int kMmqBlockSize = 32;

// The wide routed MMQ tile ("_w" kernels). MUST match the tokens of
// FLYWEIGHT_MOE_MMQ_WIDE_TOKEN_FRAGS. A 128-token block costs ~2.3x a 32-token
// one (fwbench routed_mmq_wide_bench.py, IQ2_S / IQ2_XXS), so a wide block
// pays once it holds more than 64 routes and loses below.
inline constexpr int kMmqWideBlockSize = 128;
inline constexpr int kMmqWideMinRoutes = 65;

struct AlignedRoutes {
    std::vector<std::int32_t> sorted_routes;
    std::vector<std::int32_t> block_experts;
    std::int32_t padded_total = 0;
    // align_blocks_mixed only: the first `wide_blocks` entries of
    // block_experts are wide blocks covering sorted_routes[0, wide_slots).
    std::int32_t wide_blocks = 0;
    std::int32_t wide_slots = 0;
};

// Routes whose weight is exactly zero are skipped: the rows path uses a zero
// weight to mark a route already claimed by the GPU expert cache, or pruned by
// top-k/top-p. Passing `weights == nullptr` keeps every route.
//
// Deterministic by construction: experts are visited in ascending id and routes
// within an expert in ascending route index, so the same batch always produces
// the same buffer regardless of how the counts fall.
inline void align_blocks(
    const std::int32_t* selected, const float* weights, int routes,
    int experts, int block_size, AlignedRoutes& out
) {
    out.sorted_routes.clear();
    out.block_experts.clear();
    out.padded_total = 0;
    if (!selected || routes <= 0 || experts <= 0 || block_size <= 0) return;

    std::vector<std::int32_t> counts(static_cast<std::size_t>(experts), 0);
    for (int route = 0; route < routes; ++route) {
        if (weights && weights[route] == 0.0f) continue;
        const auto expert = selected[route];
        if (expert < 0 || expert >= experts) continue;
        ++counts[static_cast<std::size_t>(expert)];
    }

    // Block-aligned offsets. An expert with no routes takes no blocks at all --
    // padding to a full block would make the kernel walk blocks that decode
    // weights for nothing, which is the cost this layout exists to remove.
    std::vector<std::int32_t> offsets(static_cast<std::size_t>(experts), 0);
    std::int32_t cursor = 0;
    for (int expert = 0; expert < experts; ++expert) {
        offsets[static_cast<std::size_t>(expert)] = cursor;
        const auto count = counts[static_cast<std::size_t>(expert)];
        if (!count) continue;
        const auto blocks = (count + block_size - 1) / block_size;
        for (int block = 0; block < blocks; ++block)
            out.block_experts.push_back(expert);
        cursor += blocks * block_size;
    }
    out.padded_total = cursor;
    out.sorted_routes.assign(static_cast<std::size_t>(cursor), kEmpty);

    std::vector<std::int32_t> filled(static_cast<std::size_t>(experts), 0);
    for (int route = 0; route < routes; ++route) {
        if (weights && weights[route] == 0.0f) continue;
        const auto expert = selected[route];
        if (expert < 0 || expert >= experts) continue;
        const auto slot = offsets[static_cast<std::size_t>(expert)] +
            filled[static_cast<std::size_t>(expert)]++;
        out.sorted_routes[static_cast<std::size_t>(slot)] = route;
    }
}

// align_blocks with two tiles. An expert's routes fill whole `wide_size`
// blocks, plus one more wide block when what is left reaches `wide_min`; the
// rest go to `block_size` blocks. Wide blocks are laid out first. Same
// determinism as align_blocks; `wide_min` of 0 means no wide blocks.
inline void align_blocks_mixed(
    const std::int32_t* selected, const float* weights, int routes,
    int experts, int block_size, int wide_size, int wide_min,
    AlignedRoutes& out
) {
    out.sorted_routes.clear();
    out.block_experts.clear();
    out.padded_total = 0;
    out.wide_blocks = 0;
    out.wide_slots = 0;
    if (!selected || routes <= 0 || experts <= 0 || block_size <= 0) return;

    std::vector<std::int32_t> counts(static_cast<std::size_t>(experts), 0);
    for (int route = 0; route < routes; ++route) {
        if (weights && weights[route] == 0.0f) continue;
        const auto expert = selected[route];
        if (expert < 0 || expert >= experts) continue;
        ++counts[static_cast<std::size_t>(expert)];
    }
    const bool any_wide = wide_min > 0 && wide_size > 0;
    std::vector<std::int32_t> wide_count(static_cast<std::size_t>(experts), 0);
    if (any_wide)
        for (int expert = 0; expert < experts; ++expert) {
            const auto count = counts[static_cast<std::size_t>(expert)];
            const auto rest = count % wide_size;
            wide_count[static_cast<std::size_t>(expert)] =
                count - rest + (rest >= wide_min ? rest : 0);
        }
    std::vector<std::int32_t> wide_offsets(static_cast<std::size_t>(experts), 0);
    std::vector<std::int32_t> offsets(static_cast<std::size_t>(experts), 0);
    std::int32_t cursor = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (int expert = 0; expert < experts; ++expert) {
            const auto wide = wide_count[static_cast<std::size_t>(expert)];
            const auto count = pass == 0
                ? wide : counts[static_cast<std::size_t>(expert)] - wide;
            if (!count) continue;
            const int size = pass == 0 ? wide_size : block_size;
            (pass == 0 ? wide_offsets : offsets)[static_cast<std::size_t>(expert)] =
                cursor;
            const auto blocks = (count + size - 1) / size;
            for (int block = 0; block < blocks; ++block)
                out.block_experts.push_back(expert);
            cursor += blocks * size;
        }
        if (pass == 0) {
            out.wide_blocks = static_cast<std::int32_t>(out.block_experts.size());
            out.wide_slots = cursor;
        }
    }
    out.padded_total = cursor;
    out.sorted_routes.assign(static_cast<std::size_t>(cursor), kEmpty);

    std::vector<std::int32_t> filled(static_cast<std::size_t>(experts), 0);
    for (int route = 0; route < routes; ++route) {
        if (weights && weights[route] == 0.0f) continue;
        const auto expert = selected[route];
        if (expert < 0 || expert >= experts) continue;
        const auto e = static_cast<std::size_t>(expert);
        const auto index = filled[e]++;
        const auto slot = index < wide_count[e]
            ? wide_offsets[e] + index : offsets[e] + index - wide_count[e];
        out.sorted_routes[static_cast<std::size_t>(slot)] = route;
    }
}

}  // namespace flyweight::v2::moe
