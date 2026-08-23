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
