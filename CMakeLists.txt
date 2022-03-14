cmake_minimum_required(VERSION 3.22)

project(la)

add_library(la INTERFACE)
target_include_directories(la INTERFACE include/)
set_property(TARGET la PROPERTY CXX_STANDARD 20)
#install(DIRECTORY include/ DESTINATION /usr/local/include)

find_package(GTest REQUIRED)

add_executable(latest test/main.cpp)
target_link_libraries(latest GTest::gtest GTest::gtest_main la)
set_property(TARGET latest PROPERTY CXX_STANDARD 20)

find_package(benchmark REQUIRED)

add_executable(labenchmark benchmark/main.cpp)
target_link_libraries(labenchmark benchmark::benchmark la)
set_property(TARGET labenchmark PROPERTY CXX_STANDARD 20)

