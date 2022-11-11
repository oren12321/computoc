#include <gtest/gtest.h>

#include <computoc/utils.h>

TEST(Algorithms_test, two_numbers_can_be_compared_with_specified_percision)
{
    using namespace computoc;

    EXPECT_TRUE(equal(0, 0));
    EXPECT_FALSE(equal(0, 1));

    EXPECT_TRUE(equal(0.0, 0.0));
    EXPECT_FALSE(equal(0.0, 1e-10));

    EXPECT_TRUE(equal(1, 1.0));
    EXPECT_FALSE(equal(0, 0.1));

    EXPECT_TRUE(close(1, 1));
    EXPECT_TRUE(close(1, 2, 2));
    EXPECT_FALSE(close(1, 2));
    EXPECT_FALSE(close(-1, 1, 1));

    EXPECT_TRUE(close(1e10, 1.00001e10));
    EXPECT_TRUE(close(1e-8, 1e-9));
    EXPECT_TRUE(close(1.0, 1.0));
    EXPECT_TRUE(close(1e-8, 0.0));
    EXPECT_TRUE(close(1e-10, 1e-20));
    EXPECT_TRUE(close(1e-10, 0.0));
    EXPECT_TRUE(close(1e-10, 0.999999e-10, 0.0));
    EXPECT_FALSE(close(1e-7, 1e-8));
    EXPECT_FALSE(close(1e10, 1.0001e10));
    EXPECT_FALSE(close(1e-7, 0.0));
    EXPECT_FALSE(close(1e-100, 0.0, 0.0));
    EXPECT_FALSE(close(1e-7, 0.0, 0.0));
    EXPECT_FALSE(close(1e-10, 1e-20, 0.0));
}
