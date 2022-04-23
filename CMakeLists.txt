cmake_minimum_required(VERSION 3.22)

project(math)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -Wall")
endif()

if (WIN32)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0)
    FetchContent_MakeAvailable(googletest)

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.6.1)
    FetchContent_MakeAvailable(benchmark)
endif()

add_subdirectory(include)
add_subdirectory(test)
add_subdirectory(benchmark)

