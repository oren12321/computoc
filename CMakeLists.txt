cmake_minimum_required(VERSION 3.22)

project(la)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -Wall")
endif()

add_subdirectory(include)
add_subdirectory(test)
add_subdirectory(benchmark)

