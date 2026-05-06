cmake_minimum_required(VERSION 3.24)  # for CUDA architecture native
project(lab1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set(LAB_SOURCE_FILES
        src/ta_utilities.cpp
        src/ta_utilities.hpp
        src/transpose_device.cu
        src/transpose_device.cuh
        src/transpose_host.cpp
)

add_executable(transpose "${LAB_SOURCE_FILES}")
# --keep: save PTX files in build directory
# -src-in-ptx: Interleave source in PTX.
# --generate-line-info: Generate line-number information for device code.
#       - useful for profiling, and for showing location of errors in compute-sanitizer
target_compile_options(transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep;-src-in-ptx;--generate-line-info>)
