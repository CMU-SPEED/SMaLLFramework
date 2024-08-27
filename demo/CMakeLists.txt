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

include_directories ( ${CMAKE_SOURCE_DIR}/demo )

# TODO: Remove these from the demo + check_interface doesn't work
# # TODO: Fix name mismatch between source and object
add_library(check_interface.o STATIC check_interface_abstract.cpp)
target_compile_definitions(check_interface.o PRIVATE uarch=${CMAKE_UARCH})



# Test out CONV for different kernels
set(LAYER_NAMES CONV DW_CONV MAX_POOL)
set(KERNEL_W_OPTIONS 5 4)
foreach(layer_name ${LAYER_NAMES})
foreach( kernel_w ${KERNEL_W_OPTIONS} )
  set(demo_name "test_interface_${layer_name}_w${kernel_w}.exe")
  message("Adding executable: ${demo_name}")
  add_executable(${demo_name} layer_kernel_test_driver.cpp )
  target_link_libraries(${demo_name} check_interface.o)
  target_compile_definitions(${demo_name} PRIVATE LAYER=${layer_name})
  target_compile_definitions(${demo_name} PRIVATE KERNEL_W_ob=${kernel_w})
endforeach( kernel_w ${KERNEL_W_OPTIONS} )
endforeach(layer_name ${LAYER_NAMES})





set(LAYER_NAMES RELU CONV PARTIAL_CONV POOL DW_CONV)
message("In demo/CMakeLists.txt: ${LAYER_NAMES}")

# foreach( layer_name ${LAYER_NAMES} )
#   set(demo_name "test_interface_${layer_name}.exe")
#   message("Adding executable: ${demo_name}")
#   add_executable(${demo_name} generic_driver.cpp )
#   target_link_libraries(${demo_name} check_interface.o)
#   target_compile_definitions(${demo_name} PRIVATE LAYER=${layer_name})
# endforeach( layer_name ${LAYER_NAMES} )

set(MODEL_SOURCES mobilenet.cpp resnet.cpp dscnn.cpp dscnn_square.cpp autoencoder.cpp tinyyolov2.cpp)
message("In demo/CMakeLists.txt: ${MODEL_SOURCES}")

foreach( model_source_file ${MODEL_SOURCES} )
    get_filename_component(justname ${model_source_file} NAME)
    string( REPLACE ".cpp" ".exe" model_exename ${justname} )
  set(model_exename "model_${model_exename}")
    message("Adding: ${model_exename}")
    add_executable(${model_exename} ${model_source_file})
    # target_link_libraries( ${model_exename} check_interface.o)
endforeach( model_source_file ${MODEL_SOURCES} )

message("In demo/CMakeLists.txt: ${MODEL_SOURCES}")
