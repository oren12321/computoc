#include <gtest/gtest.h>

#include <math/core/complex.h>

TEST(Complex_test, can_be_initialized_with_components_or_a_number)
{
    using namespace math::core::types;

    Complex c1 = 0;
    EXPECT_EQ(0, c1.r());
    EXPECT_EQ(0, c1.i());

    Complex c2 = 0.5;
    EXPECT_EQ(0.5, c2.r());
    EXPECT_EQ(0.0, c2.i());

    Complex c3{ 0, 1 };
    EXPECT_EQ(0, c3.r());
    EXPECT_EQ(1, c3.i());

    Complex c4 = { 1, 2 };
    EXPECT_EQ(1, c4.r());
    EXPECT_EQ(2, c4.i());
}

TEST(Complex_test, can_be_compared_with_other_number)
{
    using namespace math::core::types;

    EXPECT_EQ(0, (Complex{ 0 }));
    EXPECT_EQ((Complex{ 1, 2 }), (Complex{ 1, 2 }));
    EXPECT_EQ((Complex{ 0, 1 }), (Complex{ 0, 1 }));
    EXPECT_EQ(0.5, (Complex{ 0.5 }));
}

TEST(Complex_test, can_negate)
{
    using namespace math::core::types;

    EXPECT_EQ(0, (-Complex{ 0 }));
    EXPECT_EQ((Complex{ -1, -2 }), (-Complex{ 1, 2 }));
    EXPECT_EQ((Complex{ 0, -1 }), (-Complex{ 0, 1 }));
    EXPECT_EQ(-0.5, (-Complex{ 0.5 }));
}

TEST(Complex_test, have_conjugate)
{
    using namespace math::core::types;

    EXPECT_EQ(0, (Complex{ 0 }.conjugate()));
    EXPECT_EQ((Complex{ 1, -2 }), (Complex{ 1, 2 }.conjugate()));
    EXPECT_EQ((Complex{ 0, -1 }), (Complex{ 0, 1 }.conjugate()));
    EXPECT_EQ(0.5, (Complex{ 0.5 }.conjugate()));
}
