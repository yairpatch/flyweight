cmake_minimum_required(VERSION 3.24)
project(colibri_native LANGUAGES CXX)

add_library(
    colibri_native
    SHARED
    src/q4_dispatch.cpp
    src/q4_scalar.cpp
    src/q4_avx2.cpp
    src/q4_avx512.cpp
    src/gpu_driver.cpp
)

if(NOT WIN32)
    target_link_libraries(colibri_native PRIVATE ${CMAKE_DL_LIBS})
endif()

target_compile_features(colibri_native PRIVATE cxx_std_20)
target_include_directories(colibri_native PRIVATE include)
target_compile_definitions(colibri_native PRIVATE COLIBRI_NATIVE_BUILD)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(colibri_native PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(colibri_native PRIVATE /O2 /EHsc /permissive-)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    target_compile_options(colibri_native PRIVATE -O3 -fvisibility=hidden)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
endif()

set_target_properties(
    colibri_native
    PROPERTIES
    PREFIX ""
    OUTPUT_NAME "colibri_native"
)
