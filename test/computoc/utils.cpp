#include <gtest/gtest.h>

#include <computoc/utils.h>

TEST(Algorithms_test, two_numbers_can_be_compared_with_specified_percision)
{
    using namespace computoc;

    int a = 0;
    int b = 1;
    EXPECT_FALSE(is_equal(a, b));
    EXPECT_TRUE(is_equal(a, b, 1));

    double c = 0.0;
    double d = 1.0e-10;
    EXPECT_TRUE(is_equal(c, d));
    EXPECT_FALSE(is_equal(c, d, 1.0e-15));

    int e = 1;
    double f = 1.0;
    EXPECT_TRUE(is_equal(e, f));

    int g = 0;
    double h = 0.1;
    EXPECT_FALSE(is_equal(g, h));
}
