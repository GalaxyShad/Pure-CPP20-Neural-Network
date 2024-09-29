cmake_minimum_required(VERSION 3.28)
project(testya)

set(CMAKE_CXX_STANDARD 20)

add_executable(testya main.cpp
        utils.cpp
        shorttypes.h
        utils.h)
