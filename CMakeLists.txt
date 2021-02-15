cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Fusion)

SET(GCC_AVX_COMPILE_FLAGS "-mavx2 -mfma -O3 -march=native")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_AVX_COMPILE_FLAGS}")

find_package(Torch REQUIRED)

add_executable(torch_conv torch_driver.cpp direct_convolution.h kernel.h)
target_link_libraries(torch_conv "${TORCH_LIBRARIES}")
set_property(TARGET torch_conv PROPERTY CXX_STANDARD 14)
