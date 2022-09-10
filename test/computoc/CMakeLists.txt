add_executable(computoc_test
    matrix.cpp
    fraction.cpp
    complex.cpp
    algorithms.cpp
    linear_algebra.cpp
    computoc.cpp
    main.cpp)
target_link_libraries(computoc_test GTest::gtest GTest::gtest_main computoc)
set_property(TARGET computoc_test PROPERTY CXX_STANDARD 20)

