find_package(GTest REQUIRED)

add_executable(math_test
    memory.cpp
    allocators.cpp
    main.cpp)
target_link_libraries(math_test GTest::gtest GTest::gtest_main math)
set_property(TARGET math_test PROPERTY CXX_STANDARD 20)

