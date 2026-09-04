#pragma once

#include <cstdint>

struct QwenQ8KBlock {
    float scale;
    std::int8_t values[256];
    std::int16_t sums[16];
};

static_assert(sizeof(QwenQ8KBlock) == 292);

// 32-wide activation blocks matching Q4_0's own granularity. Gemma 4's widths
// (2880 hidden, 704 expert intermediate) divide 32 but not 256, which is what
// keeps its experts off the Q8-K path above.
struct QwenQ80Block {
    float scale;
    std::int8_t values[32];
};

static_assert(sizeof(QwenQ80Block) == 36);

void qwen_quantize_q8_0(
    const float* input,
    int elements,
    QwenQ80Block* output
);

float qwen_quant_dot_q4_0_q8_0_vnni(
    const std::uint8_t* packed,
    const QwenQ80Block* input,
    int elements,
    std::uint64_t row
);

// One Q4_0 weight row against up to 8 activation streams: the nibble decode
// happens once per block and the independent accumulators hide the FMA
// latency a single stream serializes on.
void qwen_quant_dot_q4_0_q8_0_multi_vnni(
    const std::uint8_t* packed,
    const QwenQ80Block* const* inputs,
    int count,
    int elements,
    std::uint64_t row,
    float* outputs
);

// 8-row interleaved repack of Q4_0 rows for the expert GEMM below. Per 8-row
// group and 32-element block: 8 f16 scales (2 bytes each), then the rows' two
// 8-byte nibble units interleaved by row -- 144 bytes, the same total as the
// 8 source blocks, so a full repack costs no extra memory rate over reading.
// rows must divide 8 and elements 32.
void qwen_q4_0_repack_x8(
    const std::uint8_t* packed,
    int rows,
    int elements,
    std::uint8_t* out
);

// Q8_0 quantization into flat GEMM layout: contiguous int8 values, per-block
// scales, and 8*sum(values) per block for the integer zero-point correction.
// Value-identical to qwen_quantize_q8_0.
void qwen_quantize_q8_gemm(
    const float* input,
    int elements,
    std::int8_t* values,
    float* scales,
    std::int32_t* bsums8
);

// repacked-Q4_0 x q8 GEMM: 8 output rows per dpbusd via broadcast activations.
// outputs[t][row] receives the plain dot; routing weights and expert scales
// fold in by the caller. Requires AVX512-VNNI (feature bit 3).
void qwen_q4_0x8_q8_gemm_vnni512(
    const std::uint8_t* repacked,
    int rows,
    int elements,
    const std::int8_t* const* values,
    const float* const* scales,
    const std::int32_t* const* bsums8,
    int tokens,
    float* const* outputs
);

float qwen_quant_dot_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* input,
    int elements,
    std::uint64_t row
);

float qwen_quant_dot_avx2(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* input,
    int elements,
    std::uint64_t row
);

// One IQ weight row against 4 or 8 activation vectors, amortizing the codebook
// decode across them. False when the type or token count has no such kernel.
bool qwen_quant_dot_iq_multi_avx2(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* const inputs[],
    int token_count,
    int elements,
    std::uint64_t row,
    float* outputs
);

void qwen_quant_dot_rows_avx2(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* input,
    int elements,
    std::uint64_t first_row,
    int row_count,
    float* outputs
);

void qwen_quant_dot_rows_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* input,
    int elements,
    std::uint64_t first_row,
    int row_count,
    float* outputs
);

void qwen_quant_dot_pair_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* first,
    const float* second,
    int elements,
    std::uint64_t row,
    float* first_output,
    float* second_output
);

void qwen_quant_dot_two_rows_avx512(
    const std::uint8_t* first_row,
    const std::uint8_t* second_row,
    std::uint32_t type,
    const float* input,
    int elements,
    float* first_output,
    float* second_output
);

void qwen_quant_dot_quad_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* const inputs[4],
    int elements,
    std::uint64_t row,
    float outputs[4]
);

void qwen_quant_dot_oct_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* const inputs[8],
    int elements,
    std::uint64_t row,
    float outputs[8]
);

void qwen_quant_dot_quad_avx2(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* const inputs[4],
    int elements,
    std::uint64_t row,
    float outputs[4]
);

void qwen_quantize_q8_k_avx2(
    const float* input,
    int elements,
    QwenQ8KBlock* output
);

float qwen_quant_dot_q8_k_avx2(
    const std::uint8_t* packed,
    std::uint32_t type,
    const QwenQ8KBlock* input,
    int elements,
    std::uint64_t row
);

float qwen_quant_dot_q8_k_avx_vnni(
    const std::uint8_t* packed,
    std::uint32_t type,
    const QwenQ8KBlock* input,
    int elements,
    std::uint64_t row
);

float qwen_quant_dot_q8_k_avx_vnni(
    const std::uint8_t* packed,
    std::uint32_t type,
    const QwenQ8KBlock* input,
    int elements,
    std::uint64_t row
);

void qwen_dequant_row_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    int elements,
    std::uint64_t row,
    float* output
);

void qwen_dequant_row_avx2(
    const std::uint8_t* packed,
    std::uint32_t type,
    int elements,
    std::uint64_t row,
    float* output
);

void qwen_f32_dot_multi_avx512(
    const float* row,
    const float* const* inputs,
    int count,
    int elements,
    float* outputs
);

void qwen_f32_dot_multi_avx2(
    const float* row,
    const float* const* inputs,
    int count,
    int elements,
    float* outputs
);

void qwen_f32_gemm_rows_avx512(
    const float* weights,
    int mr,
    const float* const* inputs,
    int count,
    int elements,
    float* out
);

void qwen_f32_gemm_rows_avx2(
    const float* weights,
    int mr,
    const float* const* inputs,
    int count,
    int elements,
    float* out
);
