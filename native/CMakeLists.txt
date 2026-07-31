cmake_minimum_required(VERSION 3.24)
project(colibri_native LANGUAGES CXX)

include(CTest)

if(MSVC)
    # The whole public ABI is extern "C", and /EHsc (CMake's default) tells MSVC
    # those functions never throw -- so it drops the unwind state around calls
    # into them and a std::runtime_error unwinds straight past the guarded()
    # try/catch and out of the DLL as SEH 0xE06D7363, losing the message. /EHs
    # keeps the same C++ semantics but assumes extern "C" can throw.
    string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

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

add_executable(colibri_turboquant_bench tools/bench_turboquant.cpp)
target_compile_features(colibri_turboquant_bench PRIVATE cxx_std_20)
target_include_directories(colibri_turboquant_bench PRIVATE src)

add_library(
    colibri_v2
    SHARED
    src/v2_runtime.cpp
    src/gpu_driver.cpp
    src/q4_dispatch.cpp
    src/q4_scalar.cpp
    src/q4_avx2.cpp
    src/q4_avx512.cpp
    src/qwen_cpu_avx2.cpp
    src/qwen_cpu_avx_vnni.cpp
    src/qwen_cpu_avx512.cpp
)
target_compile_features(colibri_v2 PRIVATE cxx_std_20)
target_include_directories(colibri_v2 PRIVATE include src)
target_compile_definitions(colibri_v2 PRIVATE COLIBRI_NATIVE_BUILD)
if(NOT WIN32)
    target_compile_options(colibri_v2 PRIVATE -O3 -fvisibility=hidden)
endif()
set_target_properties(colibri_v2 PROPERTIES PREFIX "" OUTPUT_NAME "colibri_v2")

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
    target_link_libraries(colibri_v2 PRIVATE ${CMAKE_DL_LIBS})
    target_compile_options(colibri_v2 PRIVATE -O3 -fvisibility=hidden)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/qwen_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
    target_link_libraries(colibri_native PRIVATE ${CMAKE_DL_LIBS})
endif()

target_compile_features(colibri_native PRIVATE cxx_std_20)
target_include_directories(colibri_native PRIVATE include)
target_compile_definitions(colibri_native PRIVATE COLIBRI_NATIVE_BUILD)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(colibri_v2 PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(colibri_native PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(colibri_native PRIVATE /O2 /EHs /permissive-)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/qwen_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/qwen_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    target_compile_options(colibri_native PRIVATE -O3 -fvisibility=hidden)
    set_source_files_properties(src/q4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/q4_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma;-mf16c")
endif()

if(MSVC)
    set_source_files_properties(src/qwen_cpu_avx_vnni.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
else()
    set_source_files_properties(
        src/qwen_cpu_avx_vnni.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni;-mf16c"
    )
endif()

set_target_properties(
    colibri_native
    PROPERTIES
    PREFIX ""
    OUTPUT_NAME "colibri_native"
)
