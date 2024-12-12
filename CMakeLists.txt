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

set(CMAKE_CXX_STANDARD 17)

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

# Set up build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -g -O1 -fsanitize=address -fpermissive")
# Release is setup to be default
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3 -fpermissive")
endif()

# Set up microarchitecture
set(CMAKE_PLATFORM_DIR ${CMAKE_SOURCE_DIR}/include/small/platforms/)

if(CMAKE_UARCH STREQUAL "ZEN2")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DZEN2 -DUARCH_ZEN2 -mavx2 -mfma -march=native")
  message("Microarchitecture target: ZEN2 (Epyc)")
  include_directories ( ${CMAKE_PLATFORM_DIR}/zen2 )

 elseif(CMAKE_UARCH STREQUAL "ARM")
   SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lrt -DUARCH_ARM")
   message("Microarchitecture target: ARM (vanilla)")
   include_directories ( ${CMAKE_PLATFORM_DIR}/arm )

elseif(CMAKE_UARCH STREQUAL "ARM-A72")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -lrt  -DUARCH_ARM")
  message("Microarchitecture target: ARM-A72")
  include_directories ( ${CMAKE_PLATFORM_DIR}/arm_a72 )

  elseif(CMAKE_UARCH STREQUAL "ARM-A76")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lrt  -DUARCH_ARM -march=armv8.2-a+fp16")
  message("Microarchitecture target: ARM-A76")
  include_directories ( ${CMAKE_PLATFORM_DIR}/arm_a76 )  

elseif(CMAKE_UARCH STREQUAL "ARM-A55")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static -march=armv8.2-a -lrt -DUARCH_ARM")
  message("Microarchitecture target: ARM-A55 (Qualcomm)")
  include_directories ( ${CMAKE_PLATFORM_DIR}/arm_a55 )

elseif(CMAKE_UARCH STREQUAL "ARM-A78")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static -march=armv8.2-a -lrt -DUARCH_ARM")
  message("Microarchitecture target: ARM-A78 (Qualcomm)")
  include_directories ( ${CMAKE_PLATFORM_DIR}/arm_a78 )

elseif(CMAKE_UARCH STREQUAL "ARM-X1")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static -march=armv8.2-a -lrt -DUARCH_ARM")
  message("Microarchitecture target: ARM-X1 (Qualcomm)")
  include_directories ( ${CMAKE_PLATFORM_DIR}/arm_x1 )

elseif(CMAKE_UARCH STREQUAL "Q-ARM7E")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lrt -DQUANTIZED -DUARCH_ARM")
  message("Microarchitecture target: quantized ARM7E (ARM-Mx)")
  include_directories(${CMAKE_PLATFORM_DIR}/quantized_arm7E)

else()
  SET(CMAKE_UARCH REF)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lrt -DUARCH_REF")
  message("Microarchitecture target: REF (combined)")
  include_directories ( ${CMAKE_PLATFORM_DIR}/reference )

endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${CMAKE_C_FLAGS}")

message("CXX_FLAGS:\t${CMAKE_CXX_FLAGS}")

add_subdirectory(demo)
add_subdirectory(benchmark)
enable_testing()
add_subdirectory(test)
