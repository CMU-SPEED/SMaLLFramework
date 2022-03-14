
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ContextDependentFusion)
include(CTest)
set(CMAKE_VERBOSE_MAKEFILE off)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_COMPILER gcc)
SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -mavx2 -mfma -O3 -fpermissive -march=native")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_AVX_COMPILE_FLAGS}")


#[[Comment Lines 12-26 out if you don not have libtorch]]
find_package(Torch REQUIRED)
# add_executable(torch_conv.x torch_driver_scalar.cpp)
# target_link_libraries(torch_conv.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_conv.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_naive_conv.x torch_naive_driver.cpp )
# target_link_libraries(torch_naive_conv.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_naive_conv.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_1x1.x torch_driver_1x1_scalar.cpp )
# target_link_libraries(torch_1x1.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_1x1.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_dw.x torch_driver_dw_scalar.cpp )
# target_link_libraries(torch_dw.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_dw.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_group.x torch_driver_group_scalar.cpp )
# target_link_libraries(torch_group.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_group.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_pool.x torch_driver_pool_scalar_dw.cpp )
# target_link_libraries(torch_pool.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_pool.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_relu.x torch_driver_relu_scalar.cpp )
# target_link_libraries(torch_relu.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_relu.x PROPERTY CXX_STANDARD 14)

# add_executable(ml_backend.o src/naive_direct_convolution.cpp)
# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ")
# set_property(TARGET ml_backend.o PROPERTY CXX_STANDARD 14)

add_library(check_interface.o STATIC test/interface.cpp)
add_library(interface.o STATIC src/interface.cpp)

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

# add_executable(torch_general.x torch_driver_general.cpp )
# target_link_libraries(torch_general.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_general.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_pool.x torch_driver_pool_scalar_dw.cpp )
# target_link_libraries(torch_pool.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_pool.x PROPERTY CXX_STANDARD 14)

# add_executable(torch_gpool.x torch_driver_gpool_scalar.cpp )
# target_link_libraries(torch_gpool.x "${TORCH_LIBRARIES}")
# set_property(TARGET torch_gpool.x PROPERTY CXX_STANDARD 14)

#fused libtorch free drivers

# add_executable(conv driver.cpp )
# set_property(TARGET conv PROPERTY CXX_STANDARD 14)

# add_executable(conv_code_gen driver_code_gen.cpp )
# set_property(TARGET conv_code_gen PROPERTY CXX_STANDARD 14)

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

# add_executable(fused_dw driver_dw.cpp )
# set_property(TARGET fused_dw PROPERTY CXX_STANDARD 14)


# add_executable(fused_pool driver_pool.cpp )
# set_property(TARGET fused_pool PROPERTY CXX_STANDARD 14)

#[[BROKEN DO NOT UNCOMMENT
add_executable(fused_1x1 driver_1x1.cpp )
set_property(TARGET fused_1x1 PROPERTY CXX_STANDARD 14)
]]

