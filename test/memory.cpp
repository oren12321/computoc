#include <gtest/gtest.h>

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

