// The invariant load-time quantization is chunked on.
//
// `hf::quantize` packs a tensor in 64 Ki-element tiles across threads rather
// than in one call per tensor, which is what keeps the widen buffer in L2 and
// the thread pool busy to the end. That is only sound because a K-quant block
// is self-contained: no scale, min, or code depends on a neighbouring block.
//
// Nothing in the packers announces that property, so it is pinned here. If a
// future packer ever grows cross-block state -- a running scale, a shared
// codebook -- this fails, and it should, because tiling would then silently
// change the weights.
//
// The tensor path also has to survive a tile that lands in the middle of one:
// the element counts below are deliberately not multiples of the tile.

#include "qwen_kquant.h"
#include "qwen_kquant_pack_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (condition) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

struct Packer {
    const char* name;
    void (*pack)(const float*, std::uint64_t, std::uint8_t*);
    std::uint64_t block;
    std::uint64_t bytes;
};

const Packer kPackers[] = {
    {"q8_0", qwen_kpack::pack_q8_0, 32, 34},
    {"q4_K", qwen_kpack::pack_q4_k, 256, 144},
    {"q5_K", qwen_kpack::pack_q5_k, 256, 176},
    {"q6_K", qwen_kpack::pack_q6_k, 256, 210},
};

// Values with structure rather than noise: the K-quant scale search behaves
// differently on a block that is nearly uniform than on one with an outlier,
// and both should appear across a run this long.
std::vector<float> sample(std::uint64_t count) {
    std::vector<float> out(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        const float base = std::sin(static_cast<float>(i) * 0.0007f);
        const float ramp = 1.0f + static_cast<float>(i % 97) * 0.01f;
        // One outlier per 4096, which is what stretches a block's scale.
        out[i] = base * ramp * (i % 4096 == 17 ? 40.0f : 1.0f);
    }
    return out;
}

void check(const Packer& packer, std::uint64_t elements, std::uint64_t tile) {
    const auto values = sample(elements);
    const std::uint64_t bytes = elements / packer.block * packer.bytes;
    std::vector<std::uint8_t> whole(static_cast<std::size_t>(bytes), 0);
    std::vector<std::uint8_t> tiled(static_cast<std::size_t>(bytes), 0xAB);

    packer.pack(values.data(), elements, whole.data());

    // Exactly the arithmetic in hf::quantize: tiles are whole blocks, and a
    // tile's destination is derived from its first block, not from a cursor.
    const std::uint64_t stride = tile / packer.block * packer.block;
    for (std::uint64_t begin = 0; begin < elements; begin += stride) {
        const std::uint64_t take = std::min(stride, elements - begin);
        packer.pack(values.data() + begin, take,
                    tiled.data() + begin / packer.block * packer.bytes);
    }

    char what[128];
    std::snprintf(what, sizeof(what), "%s: %llu elements in %llu-element tiles",
                  packer.name, static_cast<unsigned long long>(elements),
                  static_cast<unsigned long long>(stride));
    expect(std::memcmp(whole.data(), tiled.data(), whole.size()) == 0, what);
}

// The bytes themselves, pinned.
//
// The tiling check above only proves the packers agree with *themselves*. This
// proves they agree with what they emitted when the HF cache format's
// `kPackerVersion` was last set -- which matters because the arena is written to
// disk and keyed by that version, so a change here that nobody notices hands
// back a cache packed by different arithmetic.
//
// It is also the guard on `-ffp-contract=off`. Drop that flag and GCC fuses the
// multiply-adds in the scale search, these hashes change, and this fails. That
// is the whole point: without it the failure is a fraction of a bit of
// perplexity, which no test would have caught.
//
// If a deliberate change to the packers lands, re-measure these AND bump
// kPackerVersion in colibri_v2_hf_cache.hpp. They travel together.
struct Golden {
    const char* name;
    void (*pack)(const float*, std::uint64_t, std::uint8_t*);
    std::uint64_t block;
    std::uint64_t bytes;
    std::uint64_t hash;
};

const Golden kGolden[] = {
    {"q8_0", qwen_kpack::pack_q8_0, 32, 34, 0x2c01736abd78f494ull},
    {"q4_K", qwen_kpack::pack_q4_k, 256, 144, 0xd1d80e944961a86aull},
    {"q5_K", qwen_kpack::pack_q5_k, 256, 176, 0x8f750028afb2771eull},
    {"q6_K", qwen_kpack::pack_q6_k, 256, 210, 0x62da10453f32c722ull},
};

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t state = 1469598103934665603ull;
    for (const auto byte : bytes) {
        state ^= byte;
        state *= 1099511628211ull;
    }
    return state;
}

void check_golden() {
    const std::uint64_t elements = 256 * 512;
    const auto values = sample(elements);
    for (const auto& entry : kGolden) {
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(elements / entry.block * entry.bytes));
        entry.pack(values.data(), elements, packed.data());
        const auto hash = fnv1a(packed);
        if (hash == entry.hash) continue;
        std::printf(
            "FAIL: %s packed to 0x%016llx, expected 0x%016llx -- the encoder or "
            "its build flags changed\n",
            entry.name, static_cast<unsigned long long>(hash),
            static_cast<unsigned long long>(entry.hash));
        ++failures;
    }
}

}  // namespace

int main() {
    check_golden();
    for (const auto& packer : kPackers) {
        // The tile hf::quantize actually uses, over a run several tiles long
        // whose length is not a multiple of it.
        check(packer, 64 * 1024 * 5 + packer.block * 3, 64 * 1024);
        // A tile of one block, which is the harshest arrangement the same
        // arithmetic can produce.
        check(packer, packer.block * 41, packer.block);
        // A tile larger than the tensor: the whole thing arrives in one call,
        // which is the pre-chunking behaviour and must agree too.
        check(packer, packer.block * 7, 1 << 20);
    }

    if (failures) {
        std::printf("hf_quantize_tiling_contract: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("hf_quantize_tiling_contract: ok\n");
    return 0;
}
