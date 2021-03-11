cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ContextDependentFusion)

SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -mavx2 -mfma -O3  -march=native")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_AVX_COMPILE_FLAGS}")

find_package(Torch REQUIRED)

add_executable(torch_conv torch_driver.cpp )
target_link_libraries(torch_conv "${TORCH_LIBRARIES}")
set_property(TARGET torch_conv PROPERTY CXX_STANDARD 14)

add_executable(torch_1x1 torch_driver_1x1.cpp )
target_link_libraries(torch_1x1 "${TORCH_LIBRARIES}")
set_property(TARGET torch_1x1 PROPERTY CXX_STANDARD 14)

add_executable(torch_dw torch_driver_dw.cpp )
target_link_libraries(torch_dw "${TORCH_LIBRARIES}")
set_property(TARGET torch_dw PROPERTY CXX_STANDARD 14)

add_executable(torch_pool torch_driver_pool.cpp )
target_link_libraries(torch_pool "${TORCH_LIBRARIES}")
set_property(TARGET torch_pool PROPERTY CXX_STANDARD 14)

add_executable(torch_pool_buffered torch_driver_pool.cpp )
target_link_libraries(torch_pool_buffered "${TORCH_LIBRARIES}")
set_property(TARGET torch_pool_buffered PROPERTY CXX_STANDARD 14)
target_compile_definitions(torch_pool_buffered PRIVATE BUFFER=1)
