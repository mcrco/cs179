cmake_minimum_required(VERSION 3.24)  # for CUDA architecture native
project(lab1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE sndfile)

set(LAB_SOURCE_FILES
        src/fft_convolve.cpp
        src/fft_convolve.cu
        src/fft_convolve.cuh
        src/ta_utilities.cpp
        src/ta_utilities.hpp
        src/ErrorCheck.cuh
)

add_executable(noaudio-blur "${LAB_SOURCE_FILES}")
add_executable(audio-blur "${LAB_SOURCE_FILES}")

target_compile_definitions(noaudio-blur PRIVATE -DAUDIO_ON=0)
target_compile_definitions(audio-blur PRIVATE -DAUDIO_ON=1)

set(LAB_LINK_LIBRARIES
    "${SNDFILE_LIBRARIES}"
    CUDA::cufft
)
target_link_libraries(noaudio-blur "${LAB_LINK_LIBRARIES}")
target_link_libraries(audio-blur "${LAB_LINK_LIBRARIES}")

# --keep: save PTX files in build directory
# -src-in-ptx: Interleave source in PTX.
# --generate-line-info: Generate line-number information for device code.
#       - useful for profiling, and for showing location of errors in compute-sanitizer
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--keep;-src-in-ptx;--generate-line-info>)
