// Pins the block-aligned route layout that a fused MoE kernel will consume.
//
// The properties below are what a kernel walking this buffer relies on. They are
// checked against randomised batches rather than a handful of hand-written cases,
// because the failure that matters -- a route silently lost or duplicated -- is
// invisible in a small example and produces fluent-wrong output downstream.

#include "flyweight_v2_moe_align.hpp"

#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("  FAIL %s\n", what);
    ++failures;
}

// Every property a consumer of the layout depends on, verified together so a
// case that satisfies one and breaks another cannot pass.
void verify(const std::vector<std::int32_t>& selected,
            const std::vector<float>& weights, int experts, int block_size,
            const char* label) {
    flyweight::v2::moe::AlignedRoutes out;
    const float* w = weights.empty() ? nullptr : weights.data();
    flyweight::v2::moe::align_blocks(selected.data(), w,
                                   static_cast<int>(selected.size()), experts,
                                   block_size, out);

    // The buffer is exactly the blocks it claims.
    check(out.padded_total ==
              static_cast<std::int32_t>(out.block_experts.size()) * block_size,
          "padded_total matches block count");
    check(out.sorted_routes.size() == static_cast<std::size_t>(out.padded_total),
          "sorted_routes sized to padded_total");

    // Every kept route appears exactly once, and nothing else appears.
    std::multiset<std::int32_t> expected, seen;
    for (std::size_t route = 0; route < selected.size(); ++route) {
        if (w && w[route] == 0.0f) continue;
        const auto expert = selected[route];
        if (expert < 0 || expert >= experts) continue;
        expected.insert(static_cast<std::int32_t>(route));
    }
    for (const auto route : out.sorted_routes)
        if (route != flyweight::v2::moe::kEmpty) seen.insert(route);
    check(expected == seen, "every kept route present exactly once");

    // A block's routes all belong to the block's expert. This is the property
    // that lets one kernel launch read weights once per block.
    for (std::size_t block = 0; block < out.block_experts.size(); ++block) {
        const auto expert = out.block_experts[block];
        for (int slot = 0; slot < block_size; ++slot) {
            const auto route =
                out.sorted_routes[block * static_cast<std::size_t>(block_size) +
                                  static_cast<std::size_t>(slot)];
            if (route == flyweight::v2::moe::kEmpty) continue;
            if (selected[static_cast<std::size_t>(route)] != expert) {
                check(false, "block routes match block expert");
                return;
            }
        }
    }

    // Padding is only ever at the tail of an expert's run, never in the middle:
    // a kernel may stop at the first kEmpty within its block.
    std::map<std::int32_t, bool> ended;
    for (std::size_t block = 0; block < out.block_experts.size(); ++block) {
        bool tail = false;
        for (int slot = 0; slot < block_size; ++slot) {
            const auto route =
                out.sorted_routes[block * static_cast<std::size_t>(block_size) +
                                  static_cast<std::size_t>(slot)];
            if (route == flyweight::v2::moe::kEmpty) { tail = true; continue; }
            if (tail) { check(false, "no gap inside a block"); return; }
        }
        (void)ended;
    }

    // No expert occupies two disjoint runs -- blocks for one expert are
    // contiguous, so a kernel can address them by a single offset.
    std::set<std::int32_t> closed;
    std::int32_t previous = -1;
    for (const auto expert : out.block_experts) {
        if (expert != previous) {
            check(closed.insert(expert).second, "expert blocks are contiguous");
            previous = expert;
        }
    }

    // Deterministic: same input, same buffer.
    flyweight::v2::moe::AlignedRoutes again;
    flyweight::v2::moe::align_blocks(selected.data(), w,
                                   static_cast<int>(selected.size()), experts,
                                   block_size, again);
    check(again.sorted_routes == out.sorted_routes &&
              again.block_experts == out.block_experts,
          "deterministic");
    (void)label;
}

// align_blocks_mixed: two tiles in one buffer. Wide blocks cover
// [0, wide_slots), narrow ones the rest, and each slot's route still belongs
// to its block's expert. The padding bound is what the runtime's table sizing
// assumes: at most wide_size - 1 per expert.
void verify_mixed(const std::vector<std::int32_t>& selected,
                  const std::vector<float>& weights, int experts,
                  int block_size, int wide_size, int wide_min) {
    flyweight::v2::moe::AlignedRoutes out;
    const float* w = weights.empty() ? nullptr : weights.data();
    flyweight::v2::moe::align_blocks_mixed(
        selected.data(), w, static_cast<int>(selected.size()), experts,
        block_size, wide_size, wide_min, out);
    const auto wide_blocks = static_cast<std::size_t>(out.wide_blocks);
    check(out.wide_slots ==
              static_cast<std::int32_t>(wide_blocks) * wide_size,
          "mixed: wide slots match wide blocks");
    check(out.padded_total == out.wide_slots +
              static_cast<std::int32_t>(out.block_experts.size() - wide_blocks) *
                  block_size,
          "mixed: padded_total matches block counts");
    check(out.sorted_routes.size() == static_cast<std::size_t>(out.padded_total),
          "mixed: sorted_routes sized to padded_total");
    if (!wide_min) check(out.wide_blocks == 0, "mixed: wide_min 0 means no wide");

    std::multiset<std::int32_t> expected, seen;
    std::map<std::int32_t, int> count;
    for (std::size_t route = 0; route < selected.size(); ++route) {
        if (w && w[route] == 0.0f) continue;
        const auto expert = selected[route];
        if (expert < 0 || expert >= experts) continue;
        expected.insert(static_cast<std::int32_t>(route));
        ++count[expert];
    }
    for (const auto route : out.sorted_routes)
        if (route != flyweight::v2::moe::kEmpty) seen.insert(route);
    check(expected == seen, "mixed: every kept route present exactly once");

    std::size_t slot = 0;
    std::map<std::int32_t, int> padding;
    for (std::size_t block = 0; block < out.block_experts.size(); ++block) {
        const int size = block < wide_blocks ? wide_size : block_size;
        const auto expert = out.block_experts[block];
        for (int i = 0; i < size; ++i, ++slot) {
            const auto route = out.sorted_routes[slot];
            if (route == flyweight::v2::moe::kEmpty) { ++padding[expert]; continue; }
            if (selected[static_cast<std::size_t>(route)] != expert) {
                check(false, "mixed: block routes match block expert");
                return;
            }
        }
    }
    for (const auto& [expert, pad] : padding)
        if (pad > wide_size - 1) { check(false, "mixed: padding within bound"); return; }
    // Every wide block but an expert's last is full, and the last holds at
    // least wide_min routes: the rule that makes a wide block worth its cost.
    for (std::size_t block = 0; block < wide_blocks; ++block) {
        int used = 0;
        for (int i = 0; i < wide_size; ++i)
            used += out.sorted_routes[block * static_cast<std::size_t>(wide_size) +
                                      static_cast<std::size_t>(i)] !=
                    flyweight::v2::moe::kEmpty;
        if (used < wide_min) { check(false, "mixed: wide blocks hold wide_min"); return; }
    }

    flyweight::v2::moe::AlignedRoutes again;
    flyweight::v2::moe::align_blocks_mixed(
        selected.data(), w, static_cast<int>(selected.size()), experts,
        block_size, wide_size, wide_min, again);
    check(again.sorted_routes == out.sorted_routes &&
              again.block_experts == out.block_experts,
          "mixed: deterministic");
}

}  // namespace

int main() {
    std::printf("MoE block-align contract\n");

    // Degenerate shapes a caller can legitimately hit.
    {
        flyweight::v2::moe::AlignedRoutes out;
        flyweight::v2::moe::align_blocks(nullptr, nullptr, 0, 8, 16, out);
        check(out.padded_total == 0 && out.sorted_routes.empty() &&
                  out.block_experts.empty(), "empty input yields empty output");
        std::vector<std::int32_t> one{3};
        flyweight::v2::moe::align_blocks(one.data(), nullptr, 1, 8, 16, out);
        check(out.padded_total == 16 && out.block_experts.size() == 1 &&
                  out.block_experts[0] == 3 && out.sorted_routes[0] == 0,
              "single route takes one block");
    }

    // An exact multiple must not allocate a spare empty block.
    {
        std::vector<std::int32_t> selected(32, 5);
        flyweight::v2::moe::AlignedRoutes out;
        flyweight::v2::moe::align_blocks(selected.data(), nullptr, 32, 8, 16, out);
        check(out.block_experts.size() == 2, "exact multiple takes no spare block");
    }

    // The real shape: 512 experts, top-10, a prefill-sized batch, plus the
    // zero-weight routes the rows path uses to mark experts claimed elsewhere.
    std::mt19937 rng(20260828);
    for (const int experts : {8, 64, 512}) {
        for (const int block_size : {8, 16, 32, 64}) {
            for (const int rows : {1, 7, 64, 1024}) {
                const int top_k = 10;
                const int routes = rows * top_k;
                std::vector<std::int32_t> selected(
                    static_cast<std::size_t>(routes));
                std::vector<float> weights(static_cast<std::size_t>(routes), 1.0f);
                std::uniform_int_distribution<int> pick(0, experts - 1);
                std::uniform_real_distribution<float> drop(0.0f, 1.0f);
                for (int route = 0; route < routes; ++route) {
                    selected[static_cast<std::size_t>(route)] = pick(rng);
                    if (drop(rng) < 0.3f)
                        weights[static_cast<std::size_t>(route)] = 0.0f;
                }
                verify(selected, weights, experts, block_size, "random");
                verify(selected, {}, experts, block_size, "random, no weights");
            }
        }
    }

    // Skewed routing: one hot expert plus a long tail, which is what a real
    // prompt looks like and what makes the padding waste visible.
    {
        const int experts = 512, routes = 10240;
        std::vector<std::int32_t> selected(static_cast<std::size_t>(routes));
        std::uniform_real_distribution<float> hot(0.0f, 1.0f);
        std::uniform_int_distribution<int> tail(0, experts - 1);
        for (int route = 0; route < routes; ++route)
            selected[static_cast<std::size_t>(route)] =
                hot(rng) < 0.4f ? 7 : tail(rng);
        verify(selected, {}, experts, 32, "skewed");
        for (const int wide_min : {0, 1, 65, 128})
            verify_mixed(selected, {}, experts, 32, 128, wide_min);
    }

    // Mixed tiles over prefill-sized batches, with and without dropped routes.
    for (const int rows : {1, 64, 1024, 4096}) {
        const int experts = 512, routes = rows * 10;
        std::vector<std::int32_t> selected(static_cast<std::size_t>(routes));
        std::vector<float> weights(static_cast<std::size_t>(routes), 1.0f);
        std::uniform_int_distribution<int> pick(0, experts - 1);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        for (int route = 0; route < routes; ++route) {
            selected[static_cast<std::size_t>(route)] =
                unit(rng) < 0.3f ? pick(rng) % 16 : pick(rng);
            if (unit(rng) < 0.2f) weights[static_cast<std::size_t>(route)] = 0.0f;
        }
        verify_mixed(selected, weights, experts, 32, 128, 65);
        verify_mixed(selected, {}, experts, 32, 128, 65);
    }

    if (failures) {
        std::printf("FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
