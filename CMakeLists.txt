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

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

#set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
#set (CMAKE_MACOSX_RPATH TRUE)

# version of SMaLLFramework
set (SMaLLFramework_DATE "August 16, 2023" )
set (SMaLLFramework_VERSION_MAJOR 0 )
set (SMaLLFramework_VERSION_MINOR 4 )
set (SMaLLFramework_VERSION_SUB   0 )

project(SMaLLFramework
        VERSION "${SMaLLFramework_VERSION_MAJOR}.${SMaLLFramework_VERSION_MINOR}.${SMaLLFramework_VERSION_SUB}" )

# TODO: configure small.h with the project date and version
#configure_file (
#    "config/small.h.in"
#    "${PROJECT_SOURCE_DIR}/include/small.h" )

include(CTest)

if(CMAKE_UARCH STREQUAL "ORIN")
  message("Configuring nvcc compiler")
  option(ENABLE_CUDA "Enable building CUDA components" ON)
  enable_language(CUDA)
  set(CMAKE_VERBOSE_MAKEFILE on)
  SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  #SET(CMAKE_CXX_COMPILER /usr/local/cuda/bin/nvcc)
  #SET(CMAKE_C_COMPILER /usr/local/cuda/bin/nvcc)
  SET(CMAKE_CUDA_ARCHITECTURES 87)
  #set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -arch=sm_87 -std=c++17")
  #set_target_properties(<target_name>  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # -arch=sm_30  for Kepler
else()
  message("Configuring gnu compilers")
  set(CMAKE_VERBOSE_MAKEFILE off)
  set(CMAKE_CXX_COMPILER g++) #<--- Specify C++ compiler
  set(CMAKE_C_COMPILER gcc) #<--- Specify C compiler
  set(CMAKE_CXX_STANDARD 17)

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  
  # Set up build types
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -g -O1 -fsanitize=address -fpermissive")
  # Release is setup to be default
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3 -fpermissive")
  endif()
endif()

# Set up microarchitecture
set(CMAKE_PLATFORM_DIR ${CMAKE_SOURCE_DIR}/include/small/platforms/)

if(CMAKE_UARCH STREQUAL "ORIN")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DORIN -DUARCH_ORIN")
  message("Microarchitecture target: ORIN (GPU)")
  include_directories ( ${CMAKE_PLATFORM_DIR}/cuda_orin )
  include_directories ( /usr/local/cuda/include )

elseif(CMAKE_UARCH STREQUAL "ZEN2")
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


if(CMAKE_UARCH STREQUAL "ORIN")
  message("CMAKE_CUDA_FLAGS:\t${CMAKE_CUDA_FLAGS}")
else()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}")
  message("CXX_FLAGS:\t${CMAKE_CXX_FLAGS}")
endif()


if(CMAKE_UARCH STREQUAL "ORIN")
  # Temporary until more complete integration is achieved
  add_subdirectory(cuda_demo)
  
else()
  add_subdirectory(demo)
  add_subdirectory(benchmark)
  enable_testing()
  add_subdirectory(test)
  
endif()
