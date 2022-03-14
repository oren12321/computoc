find_package(benchmark REQUIRED)

add_executable(la_benchmark
    matrix.cpp
    main.cpp)
target_link_libraries(la_benchmark benchmark::benchmark la)
set_property(TARGET la_benchmark PROPERTY CXX_STANDARD 20)

