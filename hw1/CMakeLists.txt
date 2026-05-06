cmake_minimum_required(VERSION 3.24)  # for CUDA architecture native
project(lab1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE sndfile)

set(LAB_SOURCE_FILES
        src/blur.cpp
        src/blur.cu
        src/blur.cuh
        src/ErrorCheck.cuh
)

add_executable(noaudio-blur "${LAB_SOURCE_FILES}")
add_executable(audio-blur "${LAB_SOURCE_FILES}")

target_compile_definitions(noaudio-blur PRIVATE -DAUDIO_ON=0)
target_compile_definitions(audio-blur PRIVATE -DAUDIO_ON=1)

target_link_libraries(noaudio-blur "${SNDFILE_LIBRARIES}")
target_link_libraries(audio-blur "${SNDFILE_LIBRARIES}")
