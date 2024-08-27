include_directories ( ${CMAKE_SOURCE_DIR}/demo )




add_library(static_interface.o STATIC static_interface.cpp)
target_compile_definitions(static_interface.o PRIVATE uarch=${CMAKE_UARCH})