#include <gtest/gtest.h>

#include <type_traits>

#include <math/core/memory.h>

// Block tests

TEST(Block_test, is_empty_when_deafult_initalized)
{
    using namespace math::core::memory;

    Block b{};

    EXPECT_EQ(nullptr, b.p);
    EXPECT_EQ(0, b.s);
    EXPECT_TRUE(b.empty());
}

TEST(Block_test, can_be_of_specific_type)
{
    using namespace math::core::memory;

    Typed_block<int> b{};

    EXPECT_EQ(nullptr, b.p);
    EXPECT_EQ(0, b.s);
    EXPECT_TRUE(b.empty());

    bool valid_buffer_type = std::is_same<int, typename std::remove_pointer<decltype(b.p)>::type>();
    EXPECT_TRUE(valid_buffer_type);
}

