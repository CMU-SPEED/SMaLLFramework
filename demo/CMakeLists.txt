#-------------------------------------------------------------------------------
# demo/CMakeLists.txt:  cmake script for tests
#-------------------------------------------------------------------------------

# SMaLLFramework, (c) 2022 by The SMaLLFramework Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See additional acknowledgments in the LICENSE file

#-------------------------------------------------------------------------------
# This will only build tests from files with the name "test_*.c"
#-------------------------------------------------------------------------------

include_directories ( ${CMAKE_SOURCE_DIR}/demo )

# TODO: Fix name mismatch between source and object
add_library(check_interface.o STATIC check_interface_abstract.cpp)
target_compile_definitions(check_interface.o PRIVATE uarch=${CMAKE_UARCH})

set(LAYER_NAMES RELU CONV POOL DW)
message("In demo/CMakeLists.txt: ${LAYER_NAMES}")

foreach( layer_name ${LAYER_NAMES} )
  set(demo_name "test_interface_${layer_name}.exe")
  message("Adding executable: ${demo_name}")
  add_executable(${demo_name} generic_driver.cpp)
  target_link_libraries(${demo_name} check_interface.o interface_abstract.o)
  target_compile_definitions(${demo_name} PRIVATE LAYER=RELU)
  set_property(TARGET ${demo_name} PROPERTY CXX_STANDARD 14)
endforeach( layer_name ${LAYER_NAMES} )

set(MODEL_SOURCES mobilenet.cpp resnet.cpp dscnn.cpp dscnn_square.cpp autoencoder.cpp)
message("In demo/CMakeLists.txt: ${MODEL_SOURCES}")

foreach( model_source_file ${MODEL_SOURCES} )
    get_filename_component(justname ${model_source_file} NAME)
    string( REPLACE ".cpp" ".exe" model_exename ${justname} )
    message("Adding: ${model_exename}")
    add_executable(${model_exename} ${model_source_file})
    target_link_libraries( ${model_exename} check_interface.o interface_abstract.o )
    set_property(TARGET ${model_exename} PROPERTY CXX_STANDARD 14)
endforeach( model_source_file ${MODEL_SOURCES} )
