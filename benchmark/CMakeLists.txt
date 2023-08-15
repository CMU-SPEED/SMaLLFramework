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

include_directories ( ${CMAKE_SOURCE_DIR}/benchmark )

set(SOURCES analyze_yolov3tiny_layers.cpp analyze_yolov3tiny_layers_accum.cpp)

message("Entering benchmark/CMakeLists.txt: ${SOURCES}")

foreach( source_file ${SOURCES} )
    get_filename_component(justname ${source_file} NAME)
    string( REPLACE ".cpp" ".exe" exename ${justname} )
    message("Adding: ${exename}")
    add_executable(${exename} ${source_file})
endforeach( source_file ${SOURCES} )

message("Leaving benchmark/CMakeLists.txt: ${SOURCES}")
