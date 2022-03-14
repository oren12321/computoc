find_package(GTest REQUIRED)

add_executable(la_test main.cpp)
target_link_libraries(la_test GTest::gtest GTest::gtest_main la)
set_property(TARGET la_test PROPERTY CXX_STANDARD 20)

