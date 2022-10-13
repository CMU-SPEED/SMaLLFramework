#-------------------------------------------------------------------------------
# src/CMakeLists.txt:  cmake script for SMaLL library
#-------------------------------------------------------------------------------

# SMaLLFramework, (c) 2022 by The SMaLLFramework Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See additional acknowledgments in the LICENSE file

#-------------------------------------------------------------------------------
# Build a few object files from the src directory (temporary?)
#-------------------------------------------------------------------------------

include_directories ( ${CMAKE_SOURCE_DIR}/src )

# TODO: Deprecated??  Build the SMaLL library binary (was interface.o)
#add_library(interface.o STATIC interface.cpp)
#target_compile_definitions(interface.o PRIVATE uarch=${CMAKE_UARCH})

add_library(interface_abstract.o STATIC interface_abstract.cpp)
target_compile_definitions(interface_abstract.o PRIVATE uarch=${CMAKE_UARCH})
