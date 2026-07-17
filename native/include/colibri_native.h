#pragma once

#include <cstdint>

#if defined(_WIN32)
#if defined(COLIBRI_NATIVE_BUILD)
#define COLIBRI_API __declspec(dllexport)
#else
#define COLIBRI_API __declspec(dllimport)
#endif
#else
#define COLIBRI_API __attribute__((visibility("default")))
#endif

extern "C" {

COLIBRI_API std::uint32_t colibri_native_version();
COLIBRI_API std::uint32_t colibri_cpu_features();

COLIBRI_API int colibri_q4_matvec(
    const std::uint8_t* packed,
    const std::uint16_t* scales,
    const float* vector,
    float* output,
    std::int32_t rows,
    std::int32_t columns
);

}
