#include <gtest/gtest.h>

#include <la/la.hpp>

TEST(Matrix, DefaultCtor)
{
    la::matrix<int, 1, 1> m;
    EXPECT_EQ(0, m(0, 0));
}

