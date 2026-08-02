cmake_minimum_required(VERSION 3.24)
project(colibri_v2 LANGUAGES CXX)

include(CTest)

option(
    COLIBRI_BUILD_DEVELOPMENT_TARGETS
    "Build native contract tests and benchmark executables"
    ON
)

if(MSVC)
    # The whole public ABI is extern "C", and /EHsc (CMake's default) tells MSVC
    # those functions never throw -- so it drops the unwind state around calls
    # into them and a std::runtime_error unwinds straight past the guarded()
    # try/catch and out of the DLL as SEH 0xE06D7363, losing the message. /EHs
    # keeps the same C++ semantics but assumes extern "C" can throw.
    string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# The CPU backend compiles the CUDA corpus as host C++ rather than carrying a
# hand-written second copy of the numerics; these two artifacts are that corpus
# plus its name -> launcher table. Regenerated whenever a kernel header changes.
set(COLIBRI_CPU_KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(COLIBRI_CPU_KERNEL_BODY "${COLIBRI_CPU_KERNEL_DIR}/colibri_cpu_kernels.inc")
set(COLIBRI_CPU_KERNEL_TABLE "${COLIBRI_CPU_KERNEL_DIR}/colibri_cpu_kernel_table.inc")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT ${COLIBRI_CPU_KERNEL_BODY} ${COLIBRI_CPU_KERNEL_TABLE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${COLIBRI_CPU_KERNEL_DIR}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_kernels.py
        ${COLIBRI_CPU_KERNEL_BODY}
        ${COLIBRI_CPU_KERNEL_TABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/colibri_v2_qwen_kernels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/colibri_v2_native_kernels.hpp
        --preprocessor "${CMAKE_CXX_COMPILER} -E -P -x c++"
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_kernels.py
        ${CMAKE_CURRENT_SOURCE_DIR}/include/colibri_v2_qwen_kernels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/colibri_v2_native_kernels.hpp
    COMMENT "Generating host kernel corpus for the CPU backend"
    VERBATIM
)

add_custom_target(
    colibri_cpu_kernel_sources
    DEPENDS ${COLIBRI_CPU_KERNEL_BODY} ${COLIBRI_CPU_KERNEL_TABLE}
)

if(COLIBRI_BUILD_DEVELOPMENT_TARGETS)
add_executable(colibri_v2_provider_contract tests/provider_contract.cpp)
target_compile_features(colibri_v2_provider_contract PRIVATE cxx_std_20)
target_include_directories(colibri_v2_provider_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(NAME colibri_v2_provider_contract COMMAND colibri_v2_provider_contract)
endif()

add_executable(colibri_qwen_workspace_contract tests/qwen_workspace_contract.cpp)
target_compile_features(colibri_qwen_workspace_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_workspace_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(NAME colibri_qwen_workspace_contract COMMAND colibri_qwen_workspace_contract)
endif()

add_executable(
    colibri_qwen_attention_policy_contract
    tests/qwen_attention_policy_contract.cpp
)
target_compile_features(
    colibri_qwen_attention_policy_contract PRIVATE cxx_std_20
)
target_include_directories(
    colibri_qwen_attention_policy_contract PRIVATE include
)
if(BUILD_TESTING)
    add_test(
        NAME colibri_qwen_attention_policy_contract
        COMMAND colibri_qwen_attention_policy_contract
    )
endif()

add_executable(
    colibri_qwen_expert_policy_contract
    tests/qwen_expert_policy_contract.cpp
)
target_compile_features(colibri_qwen_expert_policy_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_expert_policy_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(
        NAME colibri_qwen_expert_policy_contract
        COMMAND colibri_qwen_expert_policy_contract
    )
endif()

add_executable(
    colibri_qwen_expert_seed_contract
    tests/qwen_expert_seed_contract.cpp
)
target_compile_features(colibri_qwen_expert_seed_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_expert_seed_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(
        NAME colibri_qwen_expert_seed_contract
        COMMAND colibri_qwen_expert_seed_contract
    )
endif()

add_executable(
    colibri_qwen_kquant_contract
    tests/qwen_kquant_contract.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(colibri_qwen_kquant_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_kquant_contract PRIVATE src)
if(BUILD_TESTING)
    add_test(NAME colibri_qwen_kquant_contract COMMAND colibri_qwen_kquant_contract)
endif()

add_executable(
    colibri_qwen_iq_simd_contract
    tests/qwen_iq_simd_contract.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(colibri_qwen_iq_simd_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_iq_simd_contract PRIVATE src include)
if(BUILD_TESTING)
    add_test(
        NAME colibri_qwen_iq_simd_contract
        COMMAND colibri_qwen_iq_simd_contract
    )
endif()

add_executable(
    colibri_turboquant_contract
    tests/turboquant_contract.cpp
)
target_compile_features(colibri_turboquant_contract PRIVATE cxx_std_20)
target_include_directories(colibri_turboquant_contract PRIVATE src)
if(BUILD_TESTING)
    add_test(
        NAME colibri_turboquant_contract
        COMMAND colibri_turboquant_contract
    )
endif()

add_executable(
    colibri_qwen_avx2_contract
    tests/qwen_avx2_contract.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(colibri_qwen_avx2_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_avx2_contract PRIVATE src)
if(NOT MSVC)
    set_source_files_properties(src/qwen_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
endif()
if(BUILD_TESTING)
    add_test(NAME colibri_qwen_avx2_contract COMMAND colibri_qwen_avx2_contract)
endif()

add_executable(
    colibri_qwen_cpu_bench
    tools/bench_q8_cpu.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(colibri_qwen_cpu_bench PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_cpu_bench PRIVATE src)

add_executable(
    colibri_cpu_parity_contract
    tests/cpu_parity_contract.cpp
    src/cpu_backend.cpp
    src/cpu_kernels.cpp
    src/cpu_native_kernels.cpp
    src/cpu_q8_dot_avx2.cpp
    src/cpu_q8_dot_avx512.cpp
    src/cpu_shim.cpp
    src/q4_dispatch.cpp
    src/q4_scalar.cpp
    src/q4_avx2.cpp
    src/q4_avx512.cpp
)
target_compile_features(colibri_cpu_parity_contract PRIVATE cxx_std_20)
target_include_directories(
    colibri_cpu_parity_contract PRIVATE include src ${COLIBRI_CPU_KERNEL_DIR}
)
add_dependencies(colibri_cpu_parity_contract colibri_cpu_kernel_sources)
if(NOT WIN32)
    target_link_libraries(colibri_cpu_parity_contract PRIVATE ${CMAKE_DL_LIBS})
endif()
find_package(Threads REQUIRED)
target_link_libraries(colibri_cpu_parity_contract PRIVATE Threads::Threads)
if(BUILD_TESTING)
    add_test(
        NAME colibri_cpu_parity_contract
        COMMAND colibri_cpu_parity_contract
    )
endif()

add_executable(colibri_turboquant_bench tools/bench_turboquant.cpp)
target_compile_features(colibri_turboquant_bench PRIVATE cxx_std_20)
target_include_directories(colibri_turboquant_bench PRIVATE src)
endif()

add_library(
    colibri_v2
    SHARED
    src/v2_runtime.cpp
    src/gpu_driver.cpp
    src/cpu_backend.cpp
    src/cpu_kernels.cpp
    src/cpu_native_kernels.cpp
    src/cpu_q8_dot_avx2.cpp
    src/cpu_q8_dot_avx512.cpp
    src/cpu_shim.cpp
    src/q4_dispatch.cpp
    src/q4_scalar.cpp
    src/q4_avx2.cpp
    src/q4_avx512.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_avx512.cpp
    ${COLIBRI_CPU_KERNEL_BODY}
    ${COLIBRI_CPU_KERNEL_TABLE}
)
add_dependencies(colibri_v2 colibri_cpu_kernel_sources)
set_source_files_properties(
    ${COLIBRI_CPU_KERNEL_BODY} ${COLIBRI_CPU_KERNEL_TABLE}
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)
target_include_directories(colibri_v2 PRIVATE ${COLIBRI_CPU_KERNEL_DIR})
target_compile_features(colibri_v2 PRIVATE cxx_std_20)
target_include_directories(colibri_v2 PRIVATE include src)
target_compile_definitions(colibri_v2 PRIVATE COLIBRI_V2_BUILD)
if(NOT WIN32)
    target_compile_options(colibri_v2 PRIVATE -O3 -fvisibility=hidden)
endif()
set_target_properties(colibri_v2 PROPERTIES PREFIX "" OUTPUT_NAME "colibri_v2")

install(
    TARGETS colibri_v2
    LIBRARY DESTINATION colibri_next/_native
    RUNTIME DESTINATION colibri_next/_native
)

if(NOT WIN32)
    target_link_libraries(colibri_v2 PRIVATE ${CMAKE_DL_LIBS})
    target_compile_options(colibri_v2 PRIVATE -O3 -fvisibility=hidden)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
    set_source_files_properties(src/cpu_q8_dot_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/cpu_q8_dot_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(colibri_v2 PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/qwen_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/qwen_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/cpu_q8_dot_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/cpu_q8_dot_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
endif()

if(MSVC)
    set_source_files_properties(src/qwen_cpu_avx_vnni.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
else()
    set_source_files_properties(
        src/qwen_cpu_avx_vnni.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni;-mf16c"
    )
endif()
