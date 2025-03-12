cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0057 NEW)
include(CMakePrintHelpers)
project(libnet_cpp)

# Add the C++ module directory
add_subdirectory(netlibpp/src)
