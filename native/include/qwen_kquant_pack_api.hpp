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
void pack_q4_k(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_q5_k(const float* values, std::uint64_t count, std::uint8_t* out);
void pack_q6_k(const float* values, std::uint64_t count, std::uint8_t* out);

// Whether the AVX2 scale search was installed. For tests and diagnostics: the
// two paths are bit-exact, so this changes speed and nothing else.
bool pack_uses_avx2();

}  // namespace qwen_kpack
