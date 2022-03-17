find_package(benchmark REQUIRED)

add_executable(math_benchmark
    matrix.cpp
    main.cpp)
target_link_libraries(math_benchmark benchmark::benchmark math)
set_property(TARGET math_benchmark PROPERTY CXX_STANDARD 20)

