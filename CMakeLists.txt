#-------------------------------------------------------------------------------
# SMaLLFramework/CMakeLists.txt:  cmake script for SMaLL library
#-------------------------------------------------------------------------------

# SMaLL, Software for Machine Learning Libraries
# Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM23-0126

#-------------------------------------------------------------------------------

# To compile the SMaLLFramework library and its tests and benchmarks, and run the
# tests:
#
#   cd build
#   cmake .. -DCMAKE_UARCH=<µkernel_arch>
#   make
#   make test
#

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

#set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
#set (CMAKE_MACOSX_RPATH TRUE)

# version of SMaLLFramework
set (SMaLLFramework_DATE "March 2, 2023" )
set (SMaLLFramework_VERSION_MAJOR 0 )
set (SMaLLFramework_VERSION_MINOR 3 )
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
  message("Microarchitecture target: ARM-REF")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/arm )
elseif(CMAKE_UARCH STREQUAL "ARM-A72")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -fpermissive -march=native -lrt ")
  message("Microarchitecture target: ARM-A72")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/arm_a72 )
elseif(CMAKE_UARCH STREQUAL "ARM-A55")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp  -static -O3 -fpermissive -march=armv8.2-a -lrt ")
  message("Microarchitecture target: Qualcomm ARM-A55")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/arm_a55 )
elseif(CMAKE_UARCH STREQUAL "ARM-A78")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp  -static -O3 -fpermissive -march=armv8.2-a -lrt ")
  message("Microarchitecture target: Qualcomm ARM-A78")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/arm_a78 )
elseif(CMAKE_UARCH STREQUAL "ARM-X1")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp  -static -O3 -fpermissive -march=armv8.2-a -lrt ")
  message("Microarchitecture target: Qualcomm ARM-X1")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/arm_x1 )
elseif(CMAKE_UARCH STREQUAL "Q-ARM7E")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -lrt -DQUANTIZED")
  message("Microarchitecture target: Q-ARM-Mx (Q-ARM7E)")
  include_directories(${CMAKE_SOURCE_DIR}/src/kernels/quantized_arm7E)
elseif(CMAKE_UARCH STREQUAL "Q-REF")
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -lrt -DQUANTIZED")
  message("Microarchitecture target: Q-REF")
  include_directories(${CMAKE_SOURCE_DIR}/src/kernels/quantized_reference)
else()
  SET(CMAKE_UARCH REF)
  SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -O3 -lrt")
  message("Microarchitecture target: REF")
  include_directories ( ${CMAKE_SOURCE_DIR}/src/kernels/reference )
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${GCC_AVX_COMPILE_FLAGS}")

message("CXX_FLAGS:        ${CMAKE_CXX_FLAGS}")

add_subdirectory(demo)

enable_testing()
add_subdirectory(test)
