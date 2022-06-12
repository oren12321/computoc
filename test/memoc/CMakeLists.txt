add_executable(memoc_test
    blocks.cpp
    utils.cpp
    main.cpp)
target_link_libraries(memoc_test GTest::gtest GTest::gtest_main memoc)
set_property(TARGET memoc_test PROPERTY CXX_STANDARD 20)

