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
    # Prevent <windows.h> from defining min/max macros that clash with
    # std::min/std::max.
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
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
        --preprocessor "${CMAKE_CXX_COMPILER}"
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

add_executable(colibri_bailing_router_contract tests/bailing_router_contract.cpp)
target_compile_features(colibri_bailing_router_contract PRIVATE cxx_std_20)
target_include_directories(colibri_bailing_router_contract PRIVATE include src)
target_compile_definitions(colibri_bailing_router_contract PRIVATE COLIBRI_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME colibri_bailing_router_contract
        COMMAND colibri_bailing_router_contract
    )
endif()

add_executable(colibri_bailing_kda_contract tests/bailing_kda_contract.cpp)
target_compile_features(colibri_bailing_kda_contract PRIVATE cxx_std_20)
target_include_directories(colibri_bailing_kda_contract PRIVATE include src)
target_compile_definitions(colibri_bailing_kda_contract PRIVATE COLIBRI_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME colibri_bailing_kda_contract
        COMMAND colibri_bailing_kda_contract
    )
endif()

add_executable(colibri_bailing_attention_contract tests/bailing_attention_contract.cpp)
target_compile_features(colibri_bailing_attention_contract PRIVATE cxx_std_20)
target_include_directories(colibri_bailing_attention_contract PRIVATE include src)
target_compile_definitions(colibri_bailing_attention_contract PRIVATE COLIBRI_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME colibri_bailing_attention_contract
        COMMAND colibri_bailing_attention_contract
    )
endif()

add_executable(colibri_bailing_layer_contract tests/bailing_layer_contract.cpp)
target_compile_features(colibri_bailing_layer_contract PRIVATE cxx_std_20)
target_include_directories(colibri_bailing_layer_contract PRIVATE include src)
target_compile_definitions(colibri_bailing_layer_contract PRIVATE COLIBRI_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME colibri_bailing_layer_contract
        COMMAND colibri_bailing_layer_contract
    )
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
    colibri_qwen_format_dispatch_contract
    tests/qwen_format_dispatch_contract.cpp
)
target_compile_features(colibri_qwen_format_dispatch_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_format_dispatch_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(
        NAME colibri_qwen_format_dispatch_contract
        COMMAND colibri_qwen_format_dispatch_contract
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
    colibri_qwen_q4_gemm_contract
    tests/qwen_q4_gemm_contract.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_q4_gemm_vnni512.cpp
)
target_compile_features(colibri_qwen_q4_gemm_contract PRIVATE cxx_std_20)
target_include_directories(colibri_qwen_q4_gemm_contract PRIVATE src)
if(NOT MSVC)
    set_source_files_properties(
        tests/qwen_q4_gemm_contract.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c"
    )
endif()
if(BUILD_TESTING)
    add_test(NAME colibri_qwen_q4_gemm_contract COMMAND colibri_qwen_q4_gemm_contract)
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
    colibri_bailing_kernels_contract
    tests/bailing_kernels_contract.cpp
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
target_compile_features(colibri_bailing_kernels_contract PRIVATE cxx_std_20)
target_compile_definitions(colibri_bailing_kernels_contract PRIVATE COLIBRI_V2_STATIC)
target_include_directories(
    colibri_bailing_kernels_contract PRIVATE include src ${COLIBRI_CPU_KERNEL_DIR}
)
add_dependencies(colibri_bailing_kernels_contract colibri_cpu_kernel_sources)
if(OpenMP_CXX_FOUND)
    target_link_libraries(colibri_bailing_kernels_contract PRIVATE OpenMP::OpenMP_CXX)
endif()
if(BUILD_TESTING)
    add_test(
        NAME colibri_bailing_kernels_contract
        COMMAND colibri_bailing_kernels_contract
    )
endif()

add_executable(
    colibri_hf_quantize_tiling_contract
    tests/hf_quantize_tiling_contract.cpp
    src/qwen_kquant_pack.cpp
    src/qwen_kquant_pack_avx2.cpp
)
target_compile_features(
    colibri_hf_quantize_tiling_contract PRIVATE cxx_std_20
)
target_include_directories(
    colibri_hf_quantize_tiling_contract PRIVATE include src
)
# The golden hashes cover this file's own sample generator too, which has a
# multiply-add in it.
if(NOT MSVC)
    target_compile_options(
        colibri_hf_quantize_tiling_contract PRIVATE -ffp-contract=off
    )
endif()
if(BUILD_TESTING)
    add_test(
        NAME colibri_hf_quantize_tiling_contract
        COMMAND colibri_hf_quantize_tiling_contract
    )
endif()

add_executable(
    colibri_tool_grammar_contract
    tests/tool_grammar_contract.cpp
)
target_compile_features(colibri_tool_grammar_contract PRIVATE cxx_std_20)
target_include_directories(colibri_tool_grammar_contract PRIVATE include src)
if(BUILD_TESTING)
    add_test(
        NAME colibri_tool_grammar_contract
        COMMAND colibri_tool_grammar_contract
    )
endif()

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
target_compile_definitions(colibri_cpu_parity_contract PRIVATE COLIBRI_V2_STATIC)
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

# Shares the CPU backend sources with the parity contract: it runs the emulated
# corpus kernel, so it needs the shim that resolves a kernel by name.
add_executable(
    colibri_qwen_expert_precision_contract
    tests/qwen_expert_precision_contract.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx512.cpp
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
target_compile_features(colibri_qwen_expert_precision_contract PRIVATE cxx_std_20)
target_compile_definitions(
    colibri_qwen_expert_precision_contract PRIVATE COLIBRI_V2_STATIC
)
target_include_directories(
    colibri_qwen_expert_precision_contract
    PRIVATE include src ${COLIBRI_CPU_KERNEL_DIR}
)
add_dependencies(
    colibri_qwen_expert_precision_contract colibri_cpu_kernel_sources
)
if(NOT WIN32)
    target_link_libraries(
        colibri_qwen_expert_precision_contract PRIVATE ${CMAKE_DL_LIBS}
    )
endif()
target_link_libraries(
    colibri_qwen_expert_precision_contract PRIVATE Threads::Threads
)
if(BUILD_TESTING)
    add_test(
        NAME colibri_qwen_expert_precision_contract
        COMMAND colibri_qwen_expert_precision_contract
    )
endif()

add_executable(
    colibri_iq_kernel_contract
    tests/qwen_iq_kernel_contract.cpp
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
target_compile_features(colibri_iq_kernel_contract PRIVATE cxx_std_20)
target_compile_definitions(colibri_iq_kernel_contract PRIVATE COLIBRI_V2_STATIC)
target_include_directories(
    colibri_iq_kernel_contract PRIVATE include src ${COLIBRI_CPU_KERNEL_DIR}
)
add_dependencies(colibri_iq_kernel_contract colibri_cpu_kernel_sources)
if(NOT WIN32)
    target_link_libraries(colibri_iq_kernel_contract PRIVATE ${CMAKE_DL_LIBS})
endif()
target_link_libraries(colibri_iq_kernel_contract PRIVATE Threads::Threads)
if(BUILD_TESTING)
    add_test(
        NAME colibri_iq_kernel_contract
        COMMAND colibri_iq_kernel_contract
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
    src/qwen_cpu_q4_gemm_vnni512.cpp
    src/qwen_kquant_pack.cpp
    src/qwen_kquant_pack_avx2.cpp
    ${COLIBRI_CPU_KERNEL_BODY}
    ${COLIBRI_CPU_KERNEL_TABLE}
)
add_dependencies(colibri_v2 colibri_cpu_kernel_sources)
# The K-quant encoders must not have their multiply-adds fused. See the header
# note in src/qwen_kquant_pack.h: contraction changes what a checkpoint
# quantizes to, and changes it differently on machines with and without FMA.
# Applied per source rather than per target, so the rest of the runtime keeps
# its FMAs.
if(NOT MSVC)
    set_source_files_properties(
        src/qwen_kquant_pack.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
    )
    # -mavx2 without -mfma: the search must not have an FMA to fuse into.
    set_source_files_properties(
        src/qwen_kquant_pack_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off"
    )
endif()
set_source_files_properties(
    ${COLIBRI_CPU_KERNEL_BODY} ${COLIBRI_CPU_KERNEL_TABLE}
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)
target_include_directories(colibri_v2 PRIVATE ${COLIBRI_CPU_KERNEL_DIR})
target_compile_features(colibri_v2 PRIVATE cxx_std_20)
target_include_directories(colibri_v2 PRIVATE include src)
target_compile_definitions(colibri_v2 PRIVATE COLIBRI_V2_BUILD)
if(MSVC)
    # Match the -O3 optimization level used on GCC/Clang.  Without this the
    # default build type is Debug (/Od) and every kernel runs unoptimised.
    target_compile_options(colibri_v2 PRIVATE /O2)
else()
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
    set_source_files_properties(src/qwen_kquant_pack_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    # Native kernels contain hot loops (dot_f32, rms_norm, route_topk) that
    # benefit from AVX2 vectorisation.  Without this flag MSVC targets SSE2.
    set_source_files_properties(src/cpu_native_kernels.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()

if(MSVC)
    set_source_files_properties(src/qwen_cpu_avx_vnni.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
    set_source_files_properties(src/qwen_cpu_q4_gemm_vnni512.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX512)
else()
    set_source_files_properties(
        src/qwen_cpu_avx_vnni.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni;-mf16c;-mfma"
    )
    set_source_files_properties(
        src/qwen_cpu_q4_gemm_vnni512.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mf16c;-mfma"
    )
endif()
