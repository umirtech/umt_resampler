cmake_minimum_required(VERSION 3.22.1)

project(umt_resampler_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(umt_resampler INTERFACE)

target_include_directories(umt_resampler INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark
    benchmarks/benchmark.cpp
)

target_link_libraries(benchmark
    umt_resampler
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(benchmark PRIVATE
        -O2
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
