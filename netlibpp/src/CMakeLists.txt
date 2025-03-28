cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0057 NEW)
include(CMakePrintHelpers)
project(libnet_module)
# set(triple X86-64-linux-gnueabihf)

# set(triple X86-64-linux-gnu)
# set(CMAKE_C_COMPILER gcc)
# set(CMAKE_C_COMPILER_TARGET ${triple})
# set(CMAKE_CXX_COMPILER g++)
# set(CMAKE_CXX_COMPILER_TARGET ${triple})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Interpreter NumPy)
# 

add_subdirectory(extern/eigen)

# target_link_libraries(main PRIVATE Eigen3::Eigen)

SET(SOURCE_LIB graph_func.cpp)
SET(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ version selection")
SET(CMAKE_CXX_FLAGS_INIT "-lpthread -shared -fopenmp -std=c++20 -O4 -fPIC $(python3 -m pybind11 --includes)")

IF (APPLE)
    EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
    IF ((${ARCHITECTURE} STREQUAL "arm64") OR (${ARCHITECTURE} STREQUAL "arm64e"))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $(python3-config --embed --ldflags)")
    ENDIF()
ENDIF()

pybind11_add_module(netlibpp_cpy include/graph_func.cpp)
target_link_libraries(netlibpp_cpy PUBLIC Eigen3::Eigen)
set_target_properties(netlibpp_cpy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../")
set_target_properties(netlibpp_cpy PROPERTIES LIBRARY_OUTPUT_DIRECTORY "../")
