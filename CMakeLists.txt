cmake_minimum_required(VERSION 3.28)
project(xor)

set(CMAKE_CXX_STANDARD 20)

add_executable(xor xor.cpp
        src/utils.cpp
        src/shorttypes.h
        src/utils.h
        src/activators.h
        src/neural_network.h
        src/weight_matrix.h
        src/neural_network_model.cpp
        src/neural_network.cpp)

project(figures)

set(CMAKE_CXX_STANDARD 20)

add_executable(figures figures.cpp
        src/utils.cpp
        src/shorttypes.h
        src/utils.h
        src/activators.h
        src/neural_network.h
        src/weight_matrix.h
        src/neural_network_model.cpp
        src/neural_network.cpp)