cmake_minimum_required(VERSION 3.22)
project(neural_network)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(neural_network
        scratch-nn.cpp)
