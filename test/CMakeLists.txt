if (DEFINED IN_DOCKER)
    find_package(GTest REQUIRED)
endif()

add_executable(math_test
    memory.cpp
    allocators.cpp
    buffers.cpp
    utils.cpp
    matrix.cpp
    fraction.cpp
    complex.cpp
    algorithms.cpp
    pointers.cpp
    main.cpp)
target_link_libraries(math_test GTest::gtest GTest::gtest_main computoc)
set_property(TARGET math_test PROPERTY CXX_STANDARD 20)

add_subdirectory(memoc)

