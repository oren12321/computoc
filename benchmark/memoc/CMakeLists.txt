if (DEFINED IN_DOCKER)
    find_package(benchmark REQUIRED)
endif()

add_executable(memoc_benchmark
    allocators.cpp
    pointers.cpp
    main.cpp)
target_link_libraries(memoc_benchmark benchmark::benchmark memoc)
set_property(TARGET memoc_benchmark PROPERTY CXX_STANDARD 20)

