#-------------------------------------------------------------------------------
# SMaLLFramework/CMakeLists.txt:  cmake script for SMaLL library
#-------------------------------------------------------------------------------

# SMaLLFramework, (c) 2022 by The SMaLLFramework Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See additional acknowledgments in the LICENSE file

#-------------------------------------------------------------------------------

# CMakeLists.txt: instructions for cmake to build LAGraph.  An ANSI C11
# compiler is required.  First, install any GraphBLAS library.  Alternatively,
# use ../GraphBLAS (see comments below).
#
# To compile the SMaLLFramework library and its tests and benchmarks, and run the
# tests:
#
# TODO: determine fate of CMAKE_SMALL_DIR from arm_support
#
#   cd build
#   cmake .. -DCMAKE_UARCH=<µkernel_arch>
#   cmake .. -DCMAKE_UARCH=<µkernal_arch> -DCMAKE_SMALL_DIR=<path-to/SMaLLFramework/src/>
#   make
#   make test
#

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

#set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
#set (CMAKE_MACOSX_RPATH TRUE)

# version of SMaLLFramework
set (SMaLLFramework_DATE "October 11, 2022" )
set (SMaLLFramework_VERSION_MAJOR 0 )
set (SMaLLFramework_VERSION_MINOR 2 )
set (SMaLLFramework_VERSION_SUB   0 )

project(SMaLLFramework
        VERSION "${SMaLLFramework_VERSION_MAJOR}.${SMaLLFramework_VERSION_MINOR}.${SMaLLFramework_VERSION_SUB}" )

# TODO: configure small.h with the project date and version
#configure_file (
#    "config/small.h.in"
#    "${PROJECT_SOURCE_DIR}/include/small.h" )

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
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -fpermissive -march=native -lrt ")
  # SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -fpermissive -mfpu=neon ")
  # SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -fpermissive -mcpu=cortex-a57 ")
  message("Microarchitecture target: ARM-A57")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/arm )
else()
  SET(CMAKE_UARCH REF)
  # TODO: why drop back to -O1 ?
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O1 -lrt")
  # SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3")
  message("Microarchitecture: REF")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/reference )
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w ${GCC_AVX_COMPILE_FLAGS}")

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall ${GCC_AVX_COMPILE_FLAGS}")
message("CXX_FLAGS:        ${CMAKE_CXX_FLAGS}")

#add_subdirectory(src)
add_subdirectory(demo)

enable_testing()
add_subdirectory(test)
