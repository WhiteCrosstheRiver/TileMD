cmake_minimum_required(VERSION 3.24)

# User-facing architecture knob (PLAN.md D1). sm_80 = A100, sm_90 = H100.
set(TILEMD_ARCHS "80;90" CACHE STRING
    "Semicolon-separated CUDA architectures without the sm_ prefix")
set(CMAKE_CUDA_ARCHITECTURES "${TILEMD_ARCHS}" CACHE STRING
    "CUDA architectures; derived from TILEMD_ARCHS" FORCE)

project(TileMD VERSION 0.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(TILEMD_BUILD_TESTS "Build the GoogleTest suite" ON)
option(TILEMD_FAST_MATH
       "Pass --use_fast_math to nvcc. Changes numerics; leave OFF for validation." OFF)

if(TILEMD_FAST_MATH)
  message(WARNING
    "TILEMD_FAST_MATH=ON changes floating-point contraction and is not valid for "
    "correctness gates. Enable it only for a kernel that documents the effect.")
endif()

function(tilemd_apply_cuda_flags target)
  # -lineinfo stays on in every config so Nsight can map samples to source.
  # Release device code uses -O3. --use_fast_math is not added unless opted in.
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
  )
  # CCCL (CUB) rejects MSVC's traditional preprocessor.
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  endif()
  if(TILEMD_FAST_MATH)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
endfunction()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/core/version.hpp
  @ONLY)

add_library(tilemd_headers INTERFACE)
target_include_directories(tilemd_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/generated)

add_executable(tilemd src/main.cu)
target_link_libraries(tilemd PRIVATE tilemd_headers)
tilemd_apply_cuda_flags(tilemd)

add_library(tilemd_reference STATIC reference/lj_reference.cpp)
target_include_directories(tilemd_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/reference)
target_compile_features(tilemd_reference PUBLIC cxx_std_17)

add_executable(tilemd_make_fixtures reference/make_fixtures.cpp)
target_link_libraries(tilemd_make_fixtures PRIVATE tilemd_reference)

add_library(tilemd_engine STATIC src/md/engine.cu)
target_link_libraries(tilemd_engine PUBLIC tilemd_headers)
tilemd_apply_cuda_flags(tilemd_engine)

if(TILEMD_BUILD_TESTS)
  include(CTest)
  enable_testing()

  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()

  include(FetchContent)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)

  add_executable(tilemd_cpu_tests tests/unit/cpu_smoke.cpp tests/unit/lj_reference_test.cpp)
  target_link_libraries(tilemd_cpu_tests PRIVATE GTest::gtest_main tilemd_headers tilemd_reference)
  file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" TILEMD_SOURCE_DIR_FWD)
  target_compile_definitions(tilemd_cpu_tests PRIVATE TILEMD_SOURCE_DIR="${TILEMD_SOURCE_DIR_FWD}")

  add_executable(tilemd_gpu_tests tests/unit/gpu_smoke.cu tests/gpu/lj_gpu_test.cu)
  target_link_libraries(tilemd_gpu_tests PRIVATE GTest::gtest_main tilemd_headers tilemd_engine
                                                 tilemd_reference)
  target_compile_definitions(tilemd_gpu_tests PRIVATE TILEMD_SOURCE_DIR="${TILEMD_SOURCE_DIR_FWD}")
  tilemd_apply_cuda_flags(tilemd_gpu_tests)

  add_test(NAME cpu_smoke COMMAND tilemd_cpu_tests)
  add_test(NAME gpu_smoke COMMAND tilemd_gpu_tests)
  add_test(NAME tilemd_version COMMAND tilemd --version)
  set_tests_properties(cpu_smoke tilemd_version PROPERTIES LABELS cpu)
  set_tests_properties(gpu_smoke PROPERTIES LABELS gpu)
  set_tests_properties(tilemd_version PROPERTIES PASS_REGULAR_EXPRESSION "TileMD ")
endif()

message(STATUS "TileMD CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TileMD fast-math: ${TILEMD_FAST_MATH}")
