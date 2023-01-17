add_executable(erroc_test
    errors.cpp
    enumgen.cpp
    erroc.cpp
    main.cpp)
target_link_libraries(erroc_test GTest::gtest GTest::gtest_main erroc)
set_property(TARGET erroc_test PROPERTY CXX_STANDARD 20)

