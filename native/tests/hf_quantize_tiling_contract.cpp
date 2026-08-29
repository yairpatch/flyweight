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
//
// The other half of this file is the type policy: which target a descriptor
// gets, which is a per-row question and was not always treated as one.

#include "flyweight_v2_hf_quantize.hpp"
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
    {"iq2_xs", qwen_kpack::pack_iq2_xs, 256, 74},
    {"q2_K", qwen_kpack::pack_q2_k, 256, 84},
    {"q3_K", qwen_kpack::pack_q3_k, 256, 110},
    // A codebook search rather than a lattice fit, and the one packer whose
    // blocks could in principle have shared state -- they must not.
    {"iq3_xxs", qwen_kpack::pack_iq3_xxs, 256, 98},
    {"iq4_xs", qwen_kpack::pack_iq4_xs, 256, 136},
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
// kPackerVersion in flyweight_v2_hf_cache.hpp. They travel together.
struct Golden {
    const char* name;
    void (*pack)(const float*, std::uint64_t, std::uint8_t*);
    std::uint64_t block;
    std::uint64_t bytes;
    std::uint64_t hash;
};

const Golden kGolden[] = {
    {"q8_0", qwen_kpack::pack_q8_0, 32, 34, 0x2c01736abd78f494ull},
    {"iq2_xs", qwen_kpack::pack_iq2_xs, 256, 74, 0x2d5a8e5462add072ull},
    {"q2_K", qwen_kpack::pack_q2_k, 256, 84, 0x73f68a8976f06085ull},
    {"q3_K", qwen_kpack::pack_q3_k, 256, 110, 0x0874546bd7fc0665ull},
    {"iq3_xxs", qwen_kpack::pack_iq3_xxs, 256, 98, 0x3fc952552f2e3d49ull},
    {"iq4_xs", qwen_kpack::pack_iq4_xs, 256, 136, 0xe114716bcc53707bull},
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

// Encode, then decode with the kernels that will actually read these bytes.
//
// The tiling and golden checks above prove the packers agree with themselves
// and with their past. Neither would notice a packer writing a well-formed
// block whose fields are in the wrong places -- the decoders are the other half
// of the format, and only running them closes that. The error bands are wide:
// what they catch is a layout or scale-search mistake, which costs tens of
// percent, not the last fraction of a bit.
struct RoundTrip {
    const char* name;
    void (*pack)(const float*, std::uint64_t, std::uint8_t*);
    std::uint64_t bytes;
    float (*value)(const std::uint8_t*, std::uint64_t);
    float worst_relative_rms;
};

const RoundTrip kRoundTrips[] = {
    // Measured on the sample below, then rounded up. Each format also has to
    // beat the one before it, which is the property that actually matters: a
    // bit bought must be a bit paid for.
    // Measured 0.367 *unweighted* -- worse than Q2_K, which is exactly why
    // the loader refuses this format without an importance matrix. This
    // entry pins the unweighted floor; the matrix is what buys the rest.
    {"iq2_xs", qwen_kpack::pack_iq2_xs, 74, qwen_iq2xs_value, 0.38f},
    {"q2_K", qwen_kpack::pack_q2_k, 84, qwen_q2k_value, 0.32f},
    // 3.06 bits, between the two K-quants either side of it. Without an
    // importance matrix it does not beat the K-quant curve -- it sits on it --
    // and the band says so.
    {"iq3_xxs", qwen_kpack::pack_iq3_xxs, 98, qwen_iq3xxs_value, 0.24f},
    {"q3_K", qwen_kpack::pack_q3_k, 110, qwen_q3k_value, 0.18f},
    // 4.25 bits against Q4_K's 4.5: the nonlinear 16-level table spends its
    // codes where weights actually live, so it must land between the K-quants
    // around it despite the eighth of a bit it gives up.
    {"iq4_xs", qwen_kpack::pack_iq4_xs, 136, qwen_iq4xs_value, 0.11f},
    {"q4_K", qwen_kpack::pack_q4_k, 144, qwen_q4k_value, 0.09f},
    {"q5_K", qwen_kpack::pack_q5_k, 176, qwen_q5_value, 0.05f},
    {"q6_K", qwen_kpack::pack_q6_k, 210, qwen_q6_value, 0.03f},
};

// Weight-shaped values: zero-mean, roughly normal, with the occasional large
// one. `sample` above is a smooth sine, which is fine for pinning bytes but
// wrong for comparing formats -- a signal that sits far from zero flatters the
// asymmetric K-quants, whose per-group minimum absorbs the offset for free,
// and penalizes the symmetric IQ grid for no reason a real tensor would.
//
// Box-Muller over a deterministic LCG rather than <random>, whose generators
// are portable but whose distributions are not.
std::vector<float> weight_sample(std::uint64_t count) {
    std::vector<float> out(count);
    std::uint64_t state = 0x9e3779b97f4a7c15ull;
    const auto uniform = [&state] {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>((state >> 40) + 1) / static_cast<float>(1 << 24);
    };
    for (std::uint64_t i = 0; i < count; i += 2) {
        const float radius = std::sqrt(-2.0f * std::log(uniform()));
        const float angle = 6.2831853f * uniform();
        out[i] = 0.02f * radius * std::cos(angle);
        if (i + 1 < count) out[i + 1] = 0.02f * radius * std::sin(angle);
    }
    // The outliers every quantizer's scale search exists for.
    for (std::uint64_t i = 31; i < count; i += 512) out[i] *= 12.0f;
    return out;
}

void check_round_trip() {
    const std::uint64_t elements = 256 * 64;
    const auto values = weight_sample(elements);
    double norm = 0.0;
    for (std::uint64_t i = 0; i < elements; ++i)
        norm += double(values[i]) * values[i];

    float previous = 0.0f;
    for (const auto& entry : kRoundTrips) {
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(elements / 256 * entry.bytes));
        entry.pack(values.data(), elements, packed.data());
        double error = 0.0;
        for (std::uint64_t i = 0; i < elements; ++i) {
            const double decoded = entry.value(packed.data(), i);
            error += (decoded - values[i]) * (decoded - values[i]);
        }
        const float relative = static_cast<float>(std::sqrt(error / norm));
        char what[160];
        std::snprintf(what, sizeof(what),
                      "%s round-trips within %.3f of the weights (got %.4f)",
                      entry.name, entry.worst_relative_rms, relative);
        expect(relative < entry.worst_relative_rms, what);
        if (previous > 0.0f) {
            std::snprintf(what, sizeof(what),
                          "%s is more accurate than the format below it", entry.name);
            expect(relative < previous, what);
        }
        previous = relative;
    }
}

// A block has to fit inside a row, because every kernel that reads these
// weights indexes by row. The embedding table is the case that had to be
// spelled out: it takes its own, higher target, and taking it unconditionally
// packed a 128-wide row across a 256-element super-block, so every row but the
// first decoded from a predecessor's scales.
flyweight::v2::hf::HfTensor described(const char* name,
                                    std::vector<std::uint64_t> shape) {
    flyweight::v2::hf::HfTensor tensor;
    tensor.name = name;
    tensor.shape = std::move(shape);
    tensor.type = 30;  // bf16
    return tensor;
}

void check_row_policy() {
    using flyweight::v2::hf::Target;
    flyweight::v2::hf::Policy policy;  // Q6_K weights, Q6_K embedding, f32 small

    const auto target = [&](const char* name, std::vector<std::uint64_t> shape) {
        return target_for(described(name, std::move(shape)), policy);
    };

    expect(target("token_embd.weight", {2048, 512}) == Target::Q6_K,
           "a whole number of blocks per row keeps the embedding target");
    expect(target("token_embd.weight", {128, 512}) == Target::Q8_0,
           "a 128-wide embedding row falls back to a block that fits");
    expect(target("output.weight", {320, 512}) == Target::Q8_0,
           "a row that is not a multiple of 256 cannot hold a K-quant block");
    expect(target("blk.0.ffn_gate.weight", {2048, 512}) == Target::Q6_K,
           "bulk weights take the policy target");
    expect(target("blk.0.attn_q_b.weight", {128, 512}) == Target::Q8_0,
           "a narrow 2-D weight stays quantized so the GPU can matvec it");
    expect(target("blk.0.ssm_conv1d.weight", {4, 512}) == Target::F32,
           "a 4-tap convolution row fits no block and stays f32");
    expect(target("blk.0.attn_norm.weight", {2048}) == Target::F32,
           "1-D tensors stay f32");
}

}  // namespace

int main() {
    check_golden();
    check_round_trip();
    check_row_policy();
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
