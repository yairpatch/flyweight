// Structural invariants of the format dispatch table. The table replaces the
// per-site type switches, so consistency errors here are exactly the drift
// class those switches used to hide: a family's row kernel wired to another
// family's name, a batch kernel listed without the warp kernel that gates it,
// or an entry whose grid enum disagrees with its kernel set.

#include "colibri_v2_format_dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace v2 = colibri::v2;

namespace {

int failures = 0;

void expect(bool condition, std::uint32_t type, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "format dispatch contract: type %u: %s\n", type, what);
    ++failures;
}

bool contains(const char* name, const char* stem) {
    return name && stem && std::strstr(name, stem) != nullptr;
}

}  // namespace

int main() {
    // Lookup basics: every entry resolves to itself, unknown types to null.
    for (const auto& format : v2::kQwenFormats)
        expect(v2::qwen_format(format.type) == &format, format.type,
               "lookup does not resolve to the entry");
    expect(v2::qwen_format(7) == nullptr, 7, "unknown type must resolve null");
    expect(v2::qwen_format(1000) == nullptr, 1000,
           "unknown type must resolve null");

    // Unique types.
    for (const auto& a : v2::kQwenFormats) {
        int seen = 0;
        for (const auto& b : v2::kQwenFormats)
            if (a.type == b.type) ++seen;
        expect(seen == 1, a.type, "duplicate table entry");
    }

    for (const auto& f : v2::kQwenFormats) {
        expect(f.family != nullptr, f.type, "entry has no family stem");

        // Every kernel name must carry the family stem: catches the
        // copy-paste cross-wiring (q5k row launching the q4k kernel) that a
        // 12-arm switch cannot make visible.
        const char* named[] = {
            f.matvec_q8_warp,   f.matvec_q8_rows, f.matmul_q8_tiled,
            f.matmul_q8_mmq,    f.matmul_rows,    f.lm_head_argmax,
            f.lm_head_argmax_q8, f.embedding,     f.embedding_rows,
        };
        for (const char* name : named)
            expect(!name || contains(name, f.family), f.type,
                   "kernel name does not carry the family stem");

        // The rows-forward batch kernels are unreachable without the warp
        // kernel that opens the Q8-activation block.
        const bool batch = f.matvec_q8_rows || f.matmul_q8_tiled ||
                           f.matmul_q8_mmq;
        expect(!batch || f.matvec_q8_warp, f.type,
               "batch Q8 kernels listed without the warp kernel");
        expect(!f.rows_q8_gate || f.matvec_q8_warp, f.type,
               "rows Q8 gate open without a warp kernel");

        // The _MIN tile geometry exists only for the asymmetric K-quants.
        const bool asymmetric = f.type == 10 || f.type == 12 || f.type == 13;
        expect(!f.mmq_min || (f.matmul_q8_mmq && asymmetric), f.type,
               "mmq_min outside the asymmetric K-quants");
        expect(!asymmetric || !f.matmul_q8_tiled, f.type,
               "asymmetric K-quant carries a tiled kernel it cannot use");

        // The Q8 argmax head is an optimization over the plain one.
        expect(!f.lm_head_argmax_q8 || f.lm_head_argmax, f.type,
               "Q8 argmax head without the plain head");

        // Embedding kernels come in pairs, and the batch grid enum exists
        // exactly when the batch kernel does.
        expect((f.embedding == nullptr) == (f.embedding_rows == nullptr),
               f.type, "embedding kernels are not paired");
        expect((f.matmul_rows == nullptr) ==
                   (f.matmul_rows_grid == v2::RowsMatmulGrid::none),
               f.type, "matmul_rows and its grid enum disagree");

        // The grouped IQ expert stem is the family stem where present.
        expect(!f.iq_expert_prefix ||
                   std::strcmp(f.iq_expert_prefix, f.family) == 0,
               f.type, "IQ expert prefix disagrees with the family");
    }

    // The two recorded drifts stay recorded until a measured commit changes
    // them; if either flips, this contract must flip with it deliberately.
    expect(v2::qwen_format(23)->rows_q8_gate == false, 23,
           "IQ4_XS rows gate changed; retire the drift note");
    expect(v2::qwen_format(29)->cpu_expert == false, 29,
           "IQ1_M CPU expert support changed; retire the drift note");

    if (failures) {
        std::fprintf(stderr, "format dispatch contract: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::puts("format dispatch contract: ok");
    return 0;
}
