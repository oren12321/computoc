add_executable(enumoc_test
    enumoc.cpp
    main.cpp)
target_link_libraries(enumoc_test GTest::gtest GTest::gtest_main enumoc)
set_property(TARGET enumoc_test PROPERTY CXX_STANDARD 20)

