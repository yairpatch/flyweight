cmake_minimum_required(VERSION 3.24)
project(flyweight_v2 LANGUAGES CXX)

include(CTest)

option(
    FLYWEIGHT_BUILD_DEVELOPMENT_TARGETS
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
set(FLYWEIGHT_CPU_KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(FLYWEIGHT_CPU_KERNEL_BODY "${FLYWEIGHT_CPU_KERNEL_DIR}/flyweight_cpu_kernels.inc")
set(FLYWEIGHT_CPU_KERNEL_TABLE "${FLYWEIGHT_CPU_KERNEL_DIR}/flyweight_cpu_kernel_table.inc")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT ${FLYWEIGHT_CPU_KERNEL_BODY} ${FLYWEIGHT_CPU_KERNEL_TABLE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${FLYWEIGHT_CPU_KERNEL_DIR}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_kernels.py
        ${FLYWEIGHT_CPU_KERNEL_BODY}
        ${FLYWEIGHT_CPU_KERNEL_TABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/flyweight_v2_qwen_kernels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/flyweight_v2_native_kernels.hpp
        --preprocessor "${CMAKE_CXX_COMPILER}"
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_kernels.py
        ${CMAKE_CURRENT_SOURCE_DIR}/include/flyweight_v2_qwen_kernels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/flyweight_v2_native_kernels.hpp
    COMMENT "Generating host kernel corpus for the CPU backend"
    VERBATIM
)

add_custom_target(
    flyweight_cpu_kernel_sources
    DEPENDS ${FLYWEIGHT_CPU_KERNEL_BODY} ${FLYWEIGHT_CPU_KERNEL_TABLE}
)

if(FLYWEIGHT_BUILD_DEVELOPMENT_TARGETS)
add_executable(flyweight_v2_provider_contract tests/provider_contract.cpp)
target_compile_features(flyweight_v2_provider_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_v2_provider_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(NAME flyweight_v2_provider_contract COMMAND flyweight_v2_provider_contract)
endif()

add_executable(flyweight_bailing_router_contract tests/bailing_router_contract.cpp)
target_compile_features(flyweight_bailing_router_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_bailing_router_contract PRIVATE include src)
target_compile_definitions(flyweight_bailing_router_contract PRIVATE FLYWEIGHT_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_bailing_router_contract
        COMMAND flyweight_bailing_router_contract
    )
endif()

add_executable(flyweight_bailing_kda_contract tests/bailing_kda_contract.cpp)
target_compile_features(flyweight_bailing_kda_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_bailing_kda_contract PRIVATE include src)
target_compile_definitions(flyweight_bailing_kda_contract PRIVATE FLYWEIGHT_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_bailing_kda_contract
        COMMAND flyweight_bailing_kda_contract
    )
endif()

add_executable(flyweight_bailing_attention_contract tests/bailing_attention_contract.cpp)
target_compile_features(flyweight_bailing_attention_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_bailing_attention_contract PRIVATE include src)
target_compile_definitions(flyweight_bailing_attention_contract PRIVATE FLYWEIGHT_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_bailing_attention_contract
        COMMAND flyweight_bailing_attention_contract
    )
endif()

add_executable(flyweight_bailing_layer_contract tests/bailing_layer_contract.cpp)
target_compile_features(flyweight_bailing_layer_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_bailing_layer_contract PRIVATE include src)
target_compile_definitions(flyweight_bailing_layer_contract PRIVATE FLYWEIGHT_BAILING_NO_SIMD)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_bailing_layer_contract
        COMMAND flyweight_bailing_layer_contract
    )
endif()

add_executable(flyweight_qwen_workspace_contract tests/qwen_workspace_contract.cpp)
target_compile_features(flyweight_qwen_workspace_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_workspace_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(NAME flyweight_qwen_workspace_contract COMMAND flyweight_qwen_workspace_contract)
endif()

add_executable(
    flyweight_qwen_attention_policy_contract
    tests/qwen_attention_policy_contract.cpp
)
target_compile_features(
    flyweight_qwen_attention_policy_contract PRIVATE cxx_std_20
)
target_include_directories(
    flyweight_qwen_attention_policy_contract PRIVATE include
)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_qwen_attention_policy_contract
        COMMAND flyweight_qwen_attention_policy_contract
    )
endif()

add_executable(
    flyweight_qwen_moe_align_contract
    tests/qwen_moe_align_contract.cpp
)
target_compile_features(flyweight_qwen_moe_align_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_moe_align_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_qwen_moe_align_contract
        COMMAND flyweight_qwen_moe_align_contract
    )
endif()

add_executable(
    flyweight_qwen_expert_policy_contract
    tests/qwen_expert_policy_contract.cpp
)
target_compile_features(flyweight_qwen_expert_policy_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_expert_policy_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_qwen_expert_policy_contract
        COMMAND flyweight_qwen_expert_policy_contract
    )
endif()

add_executable(
    flyweight_qwen_format_dispatch_contract
    tests/qwen_format_dispatch_contract.cpp
)
target_compile_features(flyweight_qwen_format_dispatch_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_format_dispatch_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_qwen_format_dispatch_contract
        COMMAND flyweight_qwen_format_dispatch_contract
    )
endif()

add_executable(
    flyweight_qwen_expert_seed_contract
    tests/qwen_expert_seed_contract.cpp
)
target_compile_features(flyweight_qwen_expert_seed_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_expert_seed_contract PRIVATE include)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_qwen_expert_seed_contract
        COMMAND flyweight_qwen_expert_seed_contract
    )
endif()

add_executable(
    flyweight_qwen_kquant_contract
    tests/qwen_kquant_contract.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(flyweight_qwen_kquant_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_kquant_contract PRIVATE src)
if(BUILD_TESTING)
    add_test(NAME flyweight_qwen_kquant_contract COMMAND flyweight_qwen_kquant_contract)
endif()

add_executable(
    flyweight_qwen_iq_simd_contract
    tests/qwen_iq_simd_contract.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(flyweight_qwen_iq_simd_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_iq_simd_contract PRIVATE src include)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_qwen_iq_simd_contract
        COMMAND flyweight_qwen_iq_simd_contract
    )
endif()

add_executable(
    flyweight_turboquant_contract
    tests/turboquant_contract.cpp
)
target_compile_features(flyweight_turboquant_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_turboquant_contract PRIVATE src)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_turboquant_contract
        COMMAND flyweight_turboquant_contract
    )
endif()

add_executable(
    flyweight_qwen_avx2_contract
    tests/qwen_avx2_contract.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(flyweight_qwen_avx2_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_avx2_contract PRIVATE src)
if(NOT MSVC)
    set_source_files_properties(src/qwen_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
endif()
if(BUILD_TESTING)
    add_test(NAME flyweight_qwen_avx2_contract COMMAND flyweight_qwen_avx2_contract)
endif()

add_executable(
    flyweight_qwen_q4_gemm_contract
    tests/qwen_q4_gemm_contract.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_q4_gemm_vnni512.cpp
)
target_compile_features(flyweight_qwen_q4_gemm_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_q4_gemm_contract PRIVATE src)
if(NOT MSVC)
    set_source_files_properties(
        tests/qwen_q4_gemm_contract.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c"
    )
endif()
if(BUILD_TESTING)
    add_test(NAME flyweight_qwen_q4_gemm_contract COMMAND flyweight_qwen_q4_gemm_contract)
endif()

add_executable(
    flyweight_qwen_cpu_bench
    tools/bench_q8_cpu.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(flyweight_qwen_cpu_bench PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_cpu_bench PRIVATE src)

# The batched CPU MoE at production shape and threading, per weight type. It
# separates "the kernel is slow" from "the loop around it is slow", which is the
# distinction end-to-end prefill numbers cannot make.
add_executable(
    flyweight_qwen_moe_layer_bench
    tools/bench_moe_cpu_layer.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(flyweight_qwen_moe_layer_bench PRIVATE cxx_std_20)
target_include_directories(flyweight_qwen_moe_layer_bench PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(flyweight_qwen_moe_layer_bench PRIVATE Threads::Threads)

add_executable(
    flyweight_bailing_kernels_contract
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
target_compile_features(flyweight_bailing_kernels_contract PRIVATE cxx_std_20)
target_compile_definitions(flyweight_bailing_kernels_contract PRIVATE FLYWEIGHT_V2_STATIC)
target_include_directories(
    flyweight_bailing_kernels_contract PRIVATE include src ${FLYWEIGHT_CPU_KERNEL_DIR}
)
add_dependencies(flyweight_bailing_kernels_contract flyweight_cpu_kernel_sources)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flyweight_bailing_kernels_contract PRIVATE OpenMP::OpenMP_CXX)
endif()
if(BUILD_TESTING)
    add_test(
        NAME flyweight_bailing_kernels_contract
        COMMAND flyweight_bailing_kernels_contract
    )
endif()

add_executable(
    flyweight_hf_quantize_tiling_contract
    tests/hf_quantize_tiling_contract.cpp
    src/qwen_kquant_pack.cpp
    src/qwen_kquant_pack_avx2.cpp
)
target_compile_features(
    flyweight_hf_quantize_tiling_contract PRIVATE cxx_std_20
)
target_include_directories(
    flyweight_hf_quantize_tiling_contract PRIVATE include src
)
# The golden hashes cover this file's own sample generator too, which has a
# multiply-add in it.
if(NOT MSVC)
    target_compile_options(
        flyweight_hf_quantize_tiling_contract PRIVATE -ffp-contract=off
    )
endif()
if(BUILD_TESTING)
    add_test(
        NAME flyweight_hf_quantize_tiling_contract
        COMMAND flyweight_hf_quantize_tiling_contract
    )
endif()

add_executable(
    flyweight_tool_grammar_contract
    tests/tool_grammar_contract.cpp
)
target_compile_features(flyweight_tool_grammar_contract PRIVATE cxx_std_20)
target_include_directories(flyweight_tool_grammar_contract PRIVATE include src)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_tool_grammar_contract
        COMMAND flyweight_tool_grammar_contract
    )
endif()

add_executable(
    flyweight_cpu_parity_contract
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
target_compile_features(flyweight_cpu_parity_contract PRIVATE cxx_std_20)
target_compile_definitions(flyweight_cpu_parity_contract PRIVATE FLYWEIGHT_V2_STATIC)
target_include_directories(
    flyweight_cpu_parity_contract PRIVATE include src ${FLYWEIGHT_CPU_KERNEL_DIR}
)
add_dependencies(flyweight_cpu_parity_contract flyweight_cpu_kernel_sources)
if(NOT WIN32)
    target_link_libraries(flyweight_cpu_parity_contract PRIVATE ${CMAKE_DL_LIBS})
endif()
find_package(Threads REQUIRED)
target_link_libraries(flyweight_cpu_parity_contract PRIVATE Threads::Threads)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_cpu_parity_contract
        COMMAND flyweight_cpu_parity_contract
    )
endif()

# Shares the CPU backend sources with the parity contract: it runs the emulated
# corpus kernel, so it needs the shim that resolves a kernel by name.
add_executable(
    flyweight_qwen_expert_precision_contract
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
target_compile_features(flyweight_qwen_expert_precision_contract PRIVATE cxx_std_20)
target_compile_definitions(
    flyweight_qwen_expert_precision_contract PRIVATE FLYWEIGHT_V2_STATIC
)
target_include_directories(
    flyweight_qwen_expert_precision_contract
    PRIVATE include src ${FLYWEIGHT_CPU_KERNEL_DIR}
)
add_dependencies(
    flyweight_qwen_expert_precision_contract flyweight_cpu_kernel_sources
)
if(NOT WIN32)
    target_link_libraries(
        flyweight_qwen_expert_precision_contract PRIVATE ${CMAKE_DL_LIBS}
    )
endif()
target_link_libraries(
    flyweight_qwen_expert_precision_contract PRIVATE Threads::Threads
)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_qwen_expert_precision_contract
        COMMAND flyweight_qwen_expert_precision_contract
    )
endif()

add_executable(
    flyweight_iq_kernel_contract
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
target_compile_features(flyweight_iq_kernel_contract PRIVATE cxx_std_20)
target_compile_definitions(flyweight_iq_kernel_contract PRIVATE FLYWEIGHT_V2_STATIC)
target_include_directories(
    flyweight_iq_kernel_contract PRIVATE include src ${FLYWEIGHT_CPU_KERNEL_DIR}
)
add_dependencies(flyweight_iq_kernel_contract flyweight_cpu_kernel_sources)
if(NOT WIN32)
    target_link_libraries(flyweight_iq_kernel_contract PRIVATE ${CMAKE_DL_LIBS})
endif()
target_link_libraries(flyweight_iq_kernel_contract PRIVATE Threads::Threads)
if(BUILD_TESTING)
    add_test(
        NAME flyweight_iq_kernel_contract
        COMMAND flyweight_iq_kernel_contract
    )
endif()

add_executable(flyweight_turboquant_bench tools/bench_turboquant.cpp)
target_compile_features(flyweight_turboquant_bench PRIVATE cxx_std_20)
target_include_directories(flyweight_turboquant_bench PRIVATE src)
endif()

add_library(
    flyweight_v2
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
    ${FLYWEIGHT_CPU_KERNEL_BODY}
    ${FLYWEIGHT_CPU_KERNEL_TABLE}
)
add_dependencies(flyweight_v2 flyweight_cpu_kernel_sources)
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
    ${FLYWEIGHT_CPU_KERNEL_BODY} ${FLYWEIGHT_CPU_KERNEL_TABLE}
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)
target_include_directories(flyweight_v2 PRIVATE ${FLYWEIGHT_CPU_KERNEL_DIR})
target_compile_features(flyweight_v2 PRIVATE cxx_std_20)
target_include_directories(flyweight_v2 PRIVATE include src)
target_compile_definitions(flyweight_v2 PRIVATE FLYWEIGHT_V2_BUILD)
if(MSVC)
    # Match the -O3 optimization level used on GCC/Clang.  Without this the
    # default build type is Debug (/Od) and every kernel runs unoptimised.
    target_compile_options(flyweight_v2 PRIVATE /O2)
else()
    target_compile_options(flyweight_v2 PRIVATE -O3 -fvisibility=hidden)
endif()
set_target_properties(flyweight_v2 PROPERTIES PREFIX "" OUTPUT_NAME "flyweight_v2")

install(
    TARGETS flyweight_v2
    LIBRARY DESTINATION flyweight/_native
    RUNTIME DESTINATION flyweight/_native
)

if(NOT WIN32)
    target_link_libraries(flyweight_v2 PRIVATE ${CMAKE_DL_LIBS})
    target_compile_options(flyweight_v2 PRIVATE -O3 -fvisibility=hidden)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
    set_source_files_properties(src/cpu_q8_dot_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/cpu_q8_dot_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flyweight_v2 PRIVATE OpenMP::OpenMP_CXX)
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
