#include <gtest/gtest.h>

#include <stdexcept>
#include <numbers>
#include <cmath>

#include <computoc/complex.h>

TEST(Complex_test, can_be_initialized_with_components_or_a_number)
{
    using namespace math::core::types;

    Complex c1 = 0.0;
    EXPECT_EQ(0.0, c1.real());
    EXPECT_EQ(0.0, c1.imag());

    Complex c2 = 0.5;
    EXPECT_EQ(0.5, c2.real());
    EXPECT_EQ(0.0, c2.imag());

    Complex c3{ 0.0, 1.0 };
    EXPECT_EQ(0.0, c3.real());
    EXPECT_EQ(1.0, c3.imag());

    Complex c4 = { 1.0, 2.0 };
    EXPECT_EQ(1.0, c4.real());
    EXPECT_EQ(2.0, c4.imag());
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

TEST(Complex_test, have_positive)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, (+Complex{ 0.0 }));
    EXPECT_EQ((Complex{ 1.0, 2.0 }), (+Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), (+Complex{ 0.0, 1.0 }));
    EXPECT_EQ(0.5, (+Complex{ 0.5 }));
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

TEST(Complex_test, can_be_divided_by_other_number)
{
    using namespace math::core::types;

    Complex c{2.0, 3.0};
    EXPECT_THROW((c / Complex{0.0}), std::overflow_error);
    EXPECT_EQ((Complex{1.0, 1.5}), c / 2.0);
    EXPECT_EQ((Complex{2.5, 0.5}), (c / Complex{1.0, 1.0}));

    c /= Complex{1.0, 1.0};
    EXPECT_EQ((Complex{2.5, 0.5}), c);
}

TEST(Complex_test, have_absolute_value)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, abs(Complex{ 0.0 }));
    EXPECT_EQ(5.0, abs(Complex{ 3.0, 4.0 }));
}

TEST(Complex_test, have_phase_angle)
{
    using namespace math::core::types;

    EXPECT_THROW(arg(Complex{ 0.0 }), std::overflow_error);

    EXPECT_EQ(0.0, arg(Complex{ 1.0, 0.0 }));
    EXPECT_EQ(std::numbers::pi / 4.0, arg(Complex{ 1.0, 1.0 }));
}

TEST(Complex_test, have_squared_magnitude)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, norm(Complex{ 0.0 }));
    EXPECT_EQ(2.0, norm(Complex{ 1.0, 1.0 }));
    EXPECT_EQ(5.0, norm(Complex{ 1.0, 2.0 }));
    EXPECT_EQ(4.0, norm(Complex{ 0.0, 2.0 }));
}

TEST(Complex_test, have_conjugate)
{
    using namespace math::core::types;

    EXPECT_EQ(0.0, conj(Complex{ 0.0 }));
    EXPECT_EQ((Complex{ 1.0, -2.0 }), conj(Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, -1.0 }), conj(Complex{ 0.0, 1.0 }));
    EXPECT_EQ(0.5, conj(Complex{ 0.5 }));
}

TEST(Comlex_test, have_projection)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 1.0, 2.0 }), proj(Complex{ 1.0, 2.0 }));
}

TEST(Complex_test, can_be_constructed_from_magnitude_and_phase_angle)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 0.0, 1.0 }), polar(1.0, std::numbers::pi / 2.0));
}

TEST(Complex_test, have_base_e_exponential)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ -1.0, 0.0 }), exp(Complex{0.0, std::numbers::pi}));
}

TEST(Complex_test, have_natural_logarithm)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 0.0, std::numbers::pi }), log(Complex{ -1.0, 0.0 }));
}

TEST(Complex_test, have_common_logarithm)
{
    using namespace math::core::types;

    Complex c{ -100.0, 0.0 };
    Complex r = log(c) / std::log(10.0);

    EXPECT_EQ(r, log10(c));
}

TEST(Complex_test, have_complex_power)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ -3.0, 4.0 }), pow(Complex{ 1.0, 2.0 }, 2.0));
    EXPECT_EQ(exp(Complex{ -std::numbers::pi / 2.0, 0.0 }), pow(Complex{ 0.0, 1.0 }, Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_square_root)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 0.0, 2.0 }), sqrt(Complex{ -4.0, 0.0 }));
}

TEST(Complex_test, have_sin)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ std::sin(1.0), 0.0 }), sin(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::sinh(1.0) }), sin(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_cos)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ std::cos(1.0), 0.0 }), cos(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ std::cosh(1.0), 0.0 }), cos(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_tan)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ std::tan(1.0), 0.0 }), tan(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::tanh(1.0) }), tan(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_asin)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), sin(asin(Complex{ 1.0, 0.0 })));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), sin(asin(Complex{ 0.0, 1.0 })));
}

TEST(Complex_test, have_acos)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), cos(acos(Complex{ 1.0, 0.0 })));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), cos(acos(Complex{ 0.0, 1.0 })));
}

TEST(Complex_test, have_atan)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), tan(atan(Complex{ 1.0, 0.0 })));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), tan(atan(Complex{ 0.0, 1.0 })));
}

TEST(Complex_test, have_sinh)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ std::sinh(1.0), 0.0 }), sinh(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::sin(1.0) }), sinh(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_cosh)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ std::cosh(1.0), 0.0 }), cosh(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ std::cos(1.0), 0.0 }), cosh(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_tanh)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ std::tanh(1.0), 0.0 }), tanh(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::tan(1.0) }), tanh(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_asinh)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), sinh(asinh(Complex{ 1.0, 0.0 })));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), sinh(asinh(Complex{ 0.0, 1.0 })));
}

TEST(Complex_test, have_acosh)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), cosh(acosh(Complex{ 1.0, 0.0 })));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), cosh(acosh(Complex{ 0.0, 1.0 })));
}

TEST(Complex_test, have_atanh)
{
    using namespace math::core::types;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), tanh(atanh(Complex{ 1.0, 0.0 })));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), tanh(atanh(Complex{ 0.0, 1.0 })));
}
