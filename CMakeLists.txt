cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ContextDependentFusion)
include(CTest)
set(CMAKE_VERBOSE_MAKEFILE off)
set(CMAKE_CXX_COMPILER g++) #<--- Specify C++ compiler
set(CMAKE_C_COMPILER gcc) #<--- Specify C compiler

#Compiler Flags 
SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -mavx2 -mfma -O3 -fpermissive -march=native")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_AVX_COMPILE_FLAGS}")



add_library(check_interface.o STATIC test/interface.cpp)
add_library(interface.o STATIC src/interface.cpp)

add_executable(test_interface_relu.x generic_driver.cpp )
target_link_libraries(test_interface_relu.x check_interface.o interface.o)
target_compile_definitions(test_interface_relu.x PRIVATE LAYER=RELU)
set_property(TARGET test_interface_relu.x PROPERTY CXX_STANDARD 14)

add_executable(test_interface_conv.x generic_driver.cpp )
target_link_libraries(test_interface_conv.x check_interface.o interface.o)
target_compile_definitions(test_interface_conv.x PRIVATE LAYER=CONV)
set_property(TARGET test_interface_conv.x PROPERTY CXX_STANDARD 14)


add_executable(test_interface_pool.x generic_driver.cpp )
target_link_libraries(test_interface_pool.x check_interface.o interface.o)
target_compile_definitions(test_interface_pool.x PRIVATE LAYER=POOL)
set_property(TARGET test_interface_pool.x PROPERTY CXX_STANDARD 14)



#[[Comment Lines out if you don not have libtorch
find_package(Torch REQUIRED)

# add_executable(ml_backend.o src/naive_direct_convolution.cpp)
# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ")
# set_property(TARGET ml_backend.o PROPERTY CXX_STANDARD 14)
add_executable(interface_relu.x torch_interface_driver.cpp )
target_link_libraries(interface_relu.x "${TORCH_LIBRARIES}")
target_link_libraries(interface_relu.x check_interface.o interface.o)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
target_compile_definitions(interface_relu.x PRIVATE LAYER=RELU)
set_property(TARGET interface_relu.x PROPERTY CXX_STANDARD 14)

add_executable(interface_conv.x torch_interface_driver.cpp )
target_link_libraries(interface_conv.x "${TORCH_LIBRARIES}")
target_link_libraries(interface_conv.x check_interface.o interface.o)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
target_compile_definitions(interface_conv.x PRIVATE LAYER=CONV)
set_property(TARGET interface_conv.x PROPERTY CXX_STANDARD 14)

add_executable(interface_pool.x torch_interface_driver.cpp )
target_link_libraries(interface_pool.x "${TORCH_LIBRARIES}")
target_link_libraries(interface_pool.x check_interface.o interface.o)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
target_compile_definitions(interface_pool.x PRIVATE LAYER=POOL)
set_property(TARGET interface_pool.x PROPERTY CXX_STANDARD 14)
]]

#libtorch free drivers




