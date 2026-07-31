#pragma once

#include <cstdint>

using Q4MatvecKernel = int (*)(
    const std::uint8_t*,
    const std::uint16_t*,
    const float*,
    float*,
    std::int32_t,
    std::int32_t
);

int q4_matvec_scalar(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
);

int q4_matvec_avx2(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
);

int q4_matvec_avx512(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
);
