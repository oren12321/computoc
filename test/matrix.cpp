#include <gtest/gtest.h>

#include <math/core/matrix.hpp>

TEST(Matrix, DefaultCtor)
{
    math::core::matrix m(1, 1, 0);
    EXPECT_EQ(0, m(0, 0));
}

