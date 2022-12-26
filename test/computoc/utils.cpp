#include <gtest/gtest.h>

#include <computoc/utils.h>

TEST(Interval_test, initialization)
{
    computoc::Interval i1{};
    EXPECT_EQ(0, i1.start);
    EXPECT_EQ(0, i1.stop);
    EXPECT_EQ(1, i1.step);

    computoc::Interval i2{ 1 };
    EXPECT_EQ(1, i2.start);
    EXPECT_EQ(1, i2.stop);
    EXPECT_EQ(1, i2.step);

    computoc::Interval i3{ 1, 2 };
    EXPECT_EQ(1, i3.start);
    EXPECT_EQ(2, i3.stop);
    EXPECT_EQ(1, i3.step);

    computoc::Interval i4{ 1, 2, 3 };
    EXPECT_EQ(1, i4.start);
    EXPECT_EQ(2, i4.stop);
    EXPECT_EQ(3, i4.step);
}

TEST(Interval_test, reverse)
{
    computoc::Interval i{ computoc::reverse(computoc::Interval{ 1, 2, 3 }) };
    EXPECT_EQ(2, i.start);
    EXPECT_EQ(1, i.stop);
    EXPECT_EQ(-3, i.step);
}

TEST(Interval_test, modulo)
{
    computoc::Interval i{ computoc::modulo(computoc::Interval{-26, 26, -1}, 5) };
    EXPECT_EQ(4, i.start);
    EXPECT_EQ(1, i.stop);
    EXPECT_EQ(-1, i.step);
}

TEST(Interval_test, forward)
{
    computoc::Interval i1{ computoc::forward(computoc::Interval{ 1, 2, 3 }) };
    EXPECT_EQ(1, i1.start);
    EXPECT_EQ(2, i1.stop);
    EXPECT_EQ(3, i1.step);

    computoc::Interval i2{ computoc::forward(computoc::Interval{ 2, 1, -3 }) };
    EXPECT_EQ(1, i2.start);
    EXPECT_EQ(2, i2.stop);
    EXPECT_EQ(3, i2.step);
}
