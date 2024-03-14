#-------------------------------------------------------------------------------
# test/CMakeLists.txt:  cmake script for tests
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
# This will only build tests from files with the name "test_*.c"
#-------------------------------------------------------------------------------

include_directories ( ${CMAKE_SOURCE_DIR}/test/include )

file( GLOB TEST_SOURCES LIST_DIRECTORIES false test_*.cpp )
message("In test/CMakeLists.txt: ${TEST_SOURCES}")

foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" ".exe" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile})
    target_include_directories(${testname} INTERFACE ${CMAKE_SOURCE_DIR}/test/include ${CMAKE_SOURCE_DIR}/test ${CMAKE_SOURCE_DIR})
    #target_link_libraries( ${testname} interface_abstract.o )
    string( REPLACE "test_" "ctest_" ctestname ${testname})
    add_test( NAME ${ctestname} COMMAND ${testname} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test )
endforeach( testsourcefile ${TEST_SOURCES} )

# kernel benchmarks
set(LAYER_NAMES MAX_POOL DW_CONV CONV)
foreach(layer_name ${LAYER_NAMES})
    set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(demo_name "test_ukernel_${layer_name}.exe")
    message("Adding executable: ${demo_name}")
    add_executable(${demo_name} kernel_benchmark_test.cpp)
    target_compile_definitions(${demo_name} PRIVATE LAYER=${layer_name})
    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
endforeach(layer_name ${LAYER_NAMES})