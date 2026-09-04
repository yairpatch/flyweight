#pragma once

// The K-quant encoders, as a translation unit rather than a header.
//
// qwen_kquant_pack.h is header-only and must be compiled with
// `-ffp-contract=off`; see the note at the top of it. Rather than impose that
// flag on every consumer -- v2_runtime.cpp is 6000 lines of unrelated numerics
// that has no reason to give up FMA -- the packers are compiled once, in
// qwen_kquant_pack.cpp, which carries the flag alone.
//
// So: include this, not the header, unless you are the .cpp.

#include <cstdint>

namespace qwen_kpack {

void pack_q8_0(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_q2_k(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_q3_k(const float* values, std::uint64_t count, std::uint8_t* out);
// Codebook rather than lattice; see qwen_iq_pack.h.
void pack_iq3_xxs(const float* values, std::uint64_t count, std::uint8_t* out);
// The same search weighted by an importance matrix: `importance` holds one
// weight per input column (`row` long), or per-expert rows of them spaced
// `chunk` elements apart (0 for the same weights on every row), and
// `element_begin` is where `values` sits in the whole tensor so tiled calls
// index the channel a single-call pack would. Null importance is exactly
// pack_iq3_xxs.
void pack_iq3_xxs(const float* values, std::uint64_t count, std::uint8_t* out,
                  const float* importance, std::uint64_t row,
                  std::uint64_t chunk, std::uint64_t element_begin);
// Nonlinear 16-level lattice, not a codebook; see qwen_iq_pack.h. The same
// importance convention as pack_iq3_xxs.
void pack_iq4_xs(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_iq4_xs(const float* values, std::uint64_t count, std::uint8_t* out,
                 const float* importance, std::uint64_t row,
                 std::uint64_t chunk, std::uint64_t element_begin);
// Codebook like iq3_xxs, one floor down; the loader only offers it with an
// importance matrix, but the packer itself runs without one for the tests.
void pack_iq2_xs(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_iq2_xs(const float* values, std::uint64_t count, std::uint8_t* out,
                 const float* importance, std::uint64_t row,
                 std::uint64_t chunk, std::uint64_t element_begin);
void pack_q4_k(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_q5_k(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_q6_k(const float* values, std::uint64_t count, std::uint8_t* out);

// Whether the AVX2 scale search was installed. For tests and diagnostics: the
// two paths are bit-exact, so this changes speed and nothing else.
bool pack_uses_avx2();

}  // namespace qwen_kpack
