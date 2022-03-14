#include <gtest/gtest.h>

#include <la/la.hpp>

TEST(Matrix, DefaultCtor)
{
    la::matrix<int, 1, 1> m;
    EXPECT_EQ(0, m(0, 0));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

