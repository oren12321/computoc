if (DEFINED IN_DOCKER)
    find_package(benchmark REQUIRED)
endif()

add_executable(math_benchmark
    allocators.cpp
    pointers.cpp
    main.cpp)
target_link_libraries(math_benchmark benchmark::benchmark computoc)
set_property(TARGET math_benchmark PROPERTY CXX_STANDARD 20)

