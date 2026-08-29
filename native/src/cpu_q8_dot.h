#pragma once

// SIMD Q8_0 row dot products for the native CPU kernels.
//
// q8_row_dot sits on the critical path of q8_matvec_transposed_warp,
// q8_matmul_tiled and q8_lm_head_argmax_warp, which together were ~53% of a
// real decode once those three were ported. It is the single highest-leverage
// function in the backend, so it gets explicit intrinsics rather than relying
// on the compiler to vectorize an int8 -> float conversion loop.
//
// Split into per-ISA translation units for the same reason as the qwen_cpu_*
// family: each needs its own -mavx512f / -mavx2 compile options, which cannot
// be applied to a file that must also run on machines without them.

#include <cstdint>

// One Q8_0 row (fp16 scale + 32 int8 per 34-byte block) against an f32 vector.
// `blocks` is elements / 32. `row_packed` points at the row's first block.
float flyweight_q8_row_dot_avx512(const std::uint8_t* row_packed,
                                const float* vector, int blocks);

float flyweight_q8_row_dot_avx2(const std::uint8_t* row_packed,
                              const float* vector, int blocks);

// Gate and up rows against the same vector, for the fused SwiGLU. Sharing the
// activation load across both weight streams is the whole point.
void flyweight_q8_row_dot_pair_avx512(const std::uint8_t* gate_packed,
                                    const std::uint8_t* up_packed,
                                    const float* vector, int blocks,
                                    float* gate_out, float* up_out);

void flyweight_q8_row_dot_pair_avx2(const std::uint8_t* gate_packed,
                                  const std::uint8_t* up_packed,
                                  const float* vector, int blocks,
                                  float* gate_out, float* up_out);
