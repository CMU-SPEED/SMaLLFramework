#-------------------------------------------------------------------------------
# test/CMakeLists.txt:  cmake script for tests
#-------------------------------------------------------------------------------

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
    #target_link_libraries( ${testname} check_interface.o interface.o )
    string( REPLACE "test_" "ctest_" ctestname ${testname})
    add_test( ${ctestname} ${testname})
endforeach( testsourcefile ${TEST_SOURCES} )
