cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0057 NEW)
include(CMakePrintHelpers)
project(libnet_module)
# set(triple X86-64-linux-gnueabihf)

set(triple X86-64-linux-gnu)
set(CMAKE_C_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_COMPILER_TARGET ${triple})

# find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
# find_package(Python3 COMPONENTS Interpreter NumPy)
# 
# find_package(Eigen3 CONFIG REQUIRED)
# target_link_libraries(main PRIVATE Eigen3::Eigen)

SET(SOURCE_LIB graph_func.cpp)
SET(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ version selection")
SET(CMAKE_CXX_FLAGS_INIT "-lpthread -shared -std=c++20 -O4 -fPIC $(python3 -m pybind11 --includes)")

IF (APPLE)
    EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
    IF ((${ARCHITECTURE} STREQUAL "arm64") OR (${ARCHITECTURE} STREQUAL "arm64e"))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $(python3-config --embed --ldflags)")
    ENDIF()
ENDIF()

pybind11_add_module(graph_func include/graph_func.cpp)
# set_target_properties(graph_func PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../")
# set_target_properties(graph_func PROPERTIES LIBRARY_OUTPUT_DIRECTORY "../")
