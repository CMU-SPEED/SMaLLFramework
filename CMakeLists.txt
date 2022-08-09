#-------------------------------------------------------------------------------
# SMaLLFramework/CMakeLists.txt:  cmake script for SMaLL library
#-------------------------------------------------------------------------------

# LAGraph, (c) 2022 by The SMaLLFramework Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See additional acknowledgments in the LICENSE file

#-------------------------------------------------------------------------------

# CMakeLists.txt: instructions for cmake to build LAGraph.  An ANSI C11
# compiler is required.  First, install any GraphBLAS library.  Alternatively,
# use ../GraphBLAS (see comments below).
#
# To compile the LAGraph library and its tests and benchmarks, and run the
# tests:
#
#   cd build
#   cmake .. -DCMAKE_UARCH=<ukernel_arch>
#   make
#   make test
#

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

#set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
#set (CMAKE_MACOSX_RPATH TRUE)

# version of SMaLLFramework
set (SMaLLFramework_DATE "Aug 3, 2022" )
set (SMaLLFramework_VERSION_MAJOR 0 )
set (SMaLLFramework_VERSION_MINOR 1 )
set (SMaLLFramework_VERSION_SUB   0 )

project(SMaLLFramework
        VERSION "${SMaLLFramework_VERSION_MAJOR}.${SMaLLFramework_VERSION_MINOR}.${SMaLLFramework_VERSION_SUB}" )

# configure LAGraph.h with the LAGraph date and version
#configure_file (
#    "config/SMaLLFramework.h.in"
#    "${PROJECT_SOURCE_DIR}/include/SMaLLFramework.h" )

include(CTest)

set(CMAKE_VERBOSE_MAKEFILE off)
set(CMAKE_CXX_COMPILER g++) #<--- Specify C++ compiler
set(CMAKE_C_COMPILER gcc) #<--- Specify C compiler

#Compiler Flags

if(CMAKE_UARCH STREQUAL "ZEN2")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -mavx2 -mfma -O3 -fpermissive -march=native")
  message("Microarchitecture target: ZEN2")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/zen2 )
elseif(CMAKE_UARCH STREQUAL "ARM")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -fpermissive -mcpu=cortex-a57 ")
  message("Microarchitecture target: ARM-A57")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/arm )
else()
  SET(CMAKE_UARCH REF)
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3")
  message("Microarchitecture: REF")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/reference )
endif()

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall ${GCC_AVX_COMPILE_FLAGS}")
message("CXX_FLAGS:        ${CMAKE_CXX_FLAGS}")

# Build the SMaLL library binary (was interface.o)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(interface.o STATIC src/interface.cpp)
target_compile_definitions(interface.o PRIVATE uarch=${CMAKE_UARCH})

#add_executable(test_interface_relu.x generic_driver.cpp )
#target_link_libraries(test_interface_relu.x check_interface.o interface.o)
#target_compile_definitions(test_interface_relu.x PRIVATE LAYER=RELU)
#set_property(TARGET test_interface_relu.x PROPERTY CXX_STANDARD 14)

#add_executable(test_interface_conv.x generic_driver.cpp )
#target_link_libraries(test_interface_conv.x check_interface.o interface.o)
#target_compile_definitions(test_interface_conv.x PRIVATE LAYER=CONV)
#set_property(TARGET test_interface_conv.x PROPERTY CXX_STANDARD 14)


#add_executable(test_interface_pool.x generic_driver.cpp )
#target_link_libraries(test_interface_pool.x check_interface.o interface.o)
#target_compile_definitions(test_interface_pool.x PRIVATE LAYER=POOL)
#set_property(TARGET test_interface_pool.x PROPERTY CXX_STANDARD 14)

add_subdirectory(demo)


enable_testing()

add_subdirectory(test)

#[[REMOVE THE FOLLOWING;  Comment Lines out if you do not have libtorch
find_package(Torch REQUIRED)
add_executable(interface_relu.x torch_interface_driver.cpp )
target_link_libraries(interface_relu.x "${TORCH_LIBRARIES}")
target_link_libraries(interface_relu.x check_interface.o interface.o)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
target_compile_definitions(interface_relu.x PRIVATE LAYER=RELU)
set_property(TARGET interface_relu.x PROPERTY CXX_STANDARD 14)

add_executable(interface_conv.x torch_interface_driver.cpp )
target_link_libraries(interface_conv.x "${TORCH_LIBRARIES}")
target_link_libraries(interface_conv.x check_interface.o interface.o)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
target_compile_definitions(interface_conv.x PRIVATE LAYER=CONV)
set_property(TARGET interface_conv.x PROPERTY CXX_STANDARD 14)

add_executable(interface_pool.x torch_interface_driver.cpp )
target_link_libraries(interface_pool.x "${TORCH_LIBRARIES}")
target_link_libraries(interface_pool.x check_interface.o interface.o)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
target_compile_definitions(interface_pool.x PRIVATE LAYER=POOL)
set_property(TARGET interface_pool.x PROPERTY CXX_STANDARD 14)
]]
