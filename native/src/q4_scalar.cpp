#include "q4_kernel.h"

#include <bit>
#include <cmath>
#include <cstdint>

namespace {

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1F;
    std::uint32_t mantissa = value & 0x03FF;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 113;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03FF;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(bits);
}

}

int q4_matvec_scalar(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
) {
    for (std::int32_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const std::int64_t row_start = static_cast<std::int64_t>(row) * columns;
        for (std::int32_t column = 0; column < columns; ++column) {
            const std::int64_t index = row_start + column;
            const std::int64_t block = index >> 5;
            const std::int32_t within_block = static_cast<std::int32_t>(index & 31);
            const std::uint8_t byte = packed[block * 16 + (within_block >> 1)];
            const std::int32_t nibble = (within_block & 1) != 0
                ? byte >> 4
                : byte & 0x0F;
            sum += static_cast<float>(nibble - 8)
                * half_to_float(scales[block]) * vector[column];
        }
        output[row] = sum;
    }
    return 0;
}
