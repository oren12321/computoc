#include <gtest/gtest.h>

#include <cstddef>

#include <memoc/accessors.h>


template <memoc::Read_only_memory_accessor<int> T>
static void dummy_initializer_list_processing_function(const T& roma) {}

TEST(Accessors_test, initializer_list)
{
    memoc::Initializer_list<int, 5> l{ 1, 2, 3, 4, 5 };

    dummy_initializer_list_processing_function(l);

    for (std::size_t i = 0; i < l.size(); ++i) {
        EXPECT_EQ(static_cast<int>(i + 1), l.data()[i]);
    }
}
