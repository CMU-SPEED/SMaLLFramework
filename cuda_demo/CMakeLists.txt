#-------------------------------------------------------------------------------
# demo/CMakeLists.txt:  cmake script for tests
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

include_directories ( ${CMAKE_SOURCE_DIR}/cuda_demo )

set(SOURCES cuda_relu.cu)
message("In cuda_demo/CMakeLists.txt: ${SOURCES}")

foreach( source_file ${SOURCES} )
    get_filename_component(justname ${source_file} NAME)
    string( REPLACE ".cu" ".x" exename ${justname} )
    set(exename "${exename}")
    message("Adding: ${exename}")
    add_executable(${exename} ${source_file})
    target_link_libraries(${exename} /usr/local/cuda/lib64/libcudart.so)

    # The following is only necessary if cuda code is making calls to other
    # kernel functions defined and compiled in different .cu files
    # It should be unnecessary with the current organization
    
    #set_target_properties(<target_name>  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach( source_file ${SOURCES} )

message("Leaving cuda_demo/CMakeLists.txt: ${SOURCES}")
