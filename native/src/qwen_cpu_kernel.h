#pragma once

#include <cstdint>

float qwen_quant_dot_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* input,
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

void qwen_f32_dot_multi_avx512(
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
