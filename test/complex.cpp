#include <gtest/gtest.h>

#include <stdexcept>
#include <numbers>

#include <math/core/complex.h>

TEST(Complex_test, can_be_initialized_with_components_or_a_number)
{
    using namespace math::core::types;

    Complex c1 = 0.0;
    EXPECT_EQ(0.0, c1.r());
    EXPECT_EQ(0.0, c1.i());

    Complex c2 = 0.5;
    EXPECT_EQ(0.5, c2.r());
    EXPECT_EQ(0.0, c2.i());

    Complex c3{ 0.0, 1.0 };
    EXPECT_EQ(0.0, c3.r());
    EXPECT_EQ(1.0, c3.i());

    Complex c4 = { 1.0, 2.0 };
    EXPECT_EQ(1.0, c4.r());
    EXPECT_EQ(2.0, c4.i());
}

TEST(Complex_test, can_be_compared_with_other_number)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, (Complex{ 0.0 }));
    EXPECT_EQ((Complex{ 1.0, 2.0 }), (Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), (Complex{ 0.0, 1.0 }));
    EXPECT_EQ(0.5, (Complex{ 0.5 }));
}

TEST(Complex_test, can_negate)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, (-Complex{ 0.0 }));
    EXPECT_EQ((Complex{ -1.0, -2.0 }), (-Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, -1.0 }), (-Complex{ 0.0, 1.0 }));
    EXPECT_EQ(-0.5, (-Complex{ 0.5 }));
}

TEST(Complex_test, have_conjugate)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, (Complex{ 0.0 }.conjugate()));
    EXPECT_EQ((Complex{ 1.0, -2.0 }), (Complex{ 1.0, 2.0 }.conjugate()));
    EXPECT_EQ((Complex{ 0.0, -1.0 }), (Complex{ 0.0, 1.0 }.conjugate()));
    EXPECT_EQ(0.5, (Complex{ 0.5 }.conjugate()));
}

TEST(Complex_test, can_be_added_to_other_number)
{
    using namespace math::core::types;

    Complex c{1.0, 2.0};
    EXPECT_EQ(c, c + 0.0);
    EXPECT_EQ((Complex{3.0, 2.0}), c + 2.0);
    EXPECT_EQ((Complex{1.5, 2.0}), c + 0.5);
    EXPECT_EQ((Complex{2.0, 1.0}), (c + Complex{1.0, -1.0}));

    c += Complex{1.0, -1.0};
    EXPECT_EQ((Complex{2.0, 1.0}), c);
}

TEST(Complex_test, can_be_subtracted_from_other_number)
{
    using namespace math::core::types;

    Complex c{1.0, 2.0};
    EXPECT_EQ(c, c - 0.0);
    EXPECT_EQ((Complex{-1.0, 2.0}), c - 2.0);
    EXPECT_EQ((Complex{0.5, 2.0}), c - 0.5);
    EXPECT_EQ((Complex{0.0, 3.0}), (c - Complex{1.0, -1.0}));

    c -= Complex{1.0, -1.0};
    EXPECT_EQ((Complex{0.0, 3.0}), c);
}

TEST(Complex_test, can_be_multiplied_with_other_number)
{
    using namespace math::core::types;

    Complex c{6.0, -2.0};
    EXPECT_EQ(0.0, c * 0.0);
    EXPECT_EQ(c, c * 1.0);
    EXPECT_EQ((Complex{1.5, -0.5}), c * 0.25);
    EXPECT_EQ((Complex{30.0, 10.0}), (c * Complex{4.0, 3.0}));

    c *= Complex{4.0, 3.0};
    EXPECT_EQ((Complex{30.0, 10.0}), c);
}

TEST(Complex_test, have_multiplicate_reciprocal)
{
    using namespace math::core::types;

    EXPECT_THROW((Complex{0.0}.multiplicative_inverse()), std::overflow_error);

    EXPECT_EQ((Complex{3.0 / 25.0, -4.0 / 25.0}), (Complex{3.0, 4.0}.multiplicative_inverse()));
}

TEST(Complex_test, can_be_divided_by_other_number)
{
    using namespace math::core::types;

    Complex c{2.0, 3.0};
    EXPECT_THROW((c / Complex{0.0}), std::overflow_error);
    EXPECT_EQ((Complex{1.0, 1.5}), c / 2);
    EXPECT_EQ((Complex{2.5, 0.5}), (c / Complex{1.0, 1.0}));

    c /= Complex{1.0, 1.0};
    EXPECT_EQ((Complex{2.5, 0.5}), c);
}

TEST(Complex_test, have_squared_magnitude)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, (Complex{ 0.0 }.squared_magnitude()));
    EXPECT_EQ(2.0, (Complex{ 1.0, 1.0 }.squared_magnitude()));
    EXPECT_EQ(5.0, (Complex{ 1.0, 2.0 }.squared_magnitude()));
    EXPECT_EQ(4.0, (Complex{ 0.0, 2.0 }.squared_magnitude()));
}

TEST(Complex_test, have_angle)
{
    using namespace math::core::types;

    EXPECT_THROW((Complex{ 0.0 }.theta()), std::overflow_error);

    EXPECT_EQ(0.0, (Complex{ 1.0, 0.0 }.theta()));
    EXPECT_EQ(std::numbers::pi / 4.0, (Complex{ 1.0, 1.0 }.theta()));
}
