#include <gtest/gtest.h>

#include <stdexcept>
#include <numbers>
#include <cmath>

#include <computoc/complex.h>
#include <computoc/math.h>

TEST(Complex_test, can_be_initialized_with_components_or_a_number)
{
    using namespace computoc;

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
    using namespace computoc;

    EXPECT_EQ(0.0, (Complex{ 0.0 }));
    EXPECT_EQ((Complex{ 1.0, 2.0 }), (Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), (Complex{ 0.0, 1.0 }));
    EXPECT_EQ((Complex{ 0.5 }), 0.5);

    EXPECT_TRUE(close(0.0, Complex{ 0.0 }));
    EXPECT_TRUE(close(Complex{ 1.0, 2.0 }, Complex{ 1.0, 2.0 }));
    EXPECT_TRUE(close(Complex{ 0.0, 1.0 }, Complex{ 0.0, 1.0 }));
    EXPECT_TRUE(close(Complex{ 0.5 }, 0.5));

    EXPECT_EQ(0, (Complex{ 0.0 }));
    EXPECT_EQ(1, (Complex{ 1.0 }));
    EXPECT_EQ((Complex{ 1.0 }), 1);
    EXPECT_EQ((Complex{ 1 }), ((Complex{ 1.0 })));
}

TEST(Complex_test, can_negate)
{
    using namespace computoc;

    EXPECT_EQ(0.0, (-Complex{ 0.0 }));
    EXPECT_EQ((Complex{ -1.0, -2.0 }), (-Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, -1.0 }), (-Complex{ 0.0, 1.0 }));
    EXPECT_EQ(-0.5, (-Complex{ 0.5 }));
}

TEST(Complex_test, have_positive)
{
    using namespace computoc;

    EXPECT_EQ(0.0, (+Complex{ 0.0 }));
    EXPECT_EQ((Complex{ 1.0, 2.0 }), (+Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), (+Complex{ 0.0, 1.0 }));
    EXPECT_EQ(0.5, (+Complex{ 0.5 }));
}

TEST(Complex_test, can_be_added_to_other_number)
{
    using namespace computoc;

    Complex c{1.0, 2.0};
    EXPECT_EQ(c, c + 0.0);
    EXPECT_EQ(c, c + 0);
    EXPECT_EQ((Complex{3.0, 2.0}), c + 2.0);
    EXPECT_EQ((Complex{1.5, 2.0}), c + 0.5);
    EXPECT_EQ((Complex{2.0, 1.0}), (c + Complex{1.0, -1.0}));

    c += Complex{1.0, -1.0};
    EXPECT_EQ((Complex{2.0, 1.0}), c);
    c += Complex{ 0 };
    c += 0;
    EXPECT_EQ((Complex{ 2.0, 1.0 }), c);
}

TEST(Complex_test, can_be_subtracted_from_other_number)
{
    using namespace computoc;

    Complex c{1.0, 2.0};
    EXPECT_EQ(c, c - 0.0);
    EXPECT_EQ(c, c - 0);
    EXPECT_EQ((Complex{-1.0, 2.0}), c - 2.0);
    EXPECT_EQ((Complex{0.5, 2.0}), c - 0.5);
    EXPECT_EQ((Complex{0.0, 3.0}), (c - Complex{1.0, -1.0}));

    c -= Complex{1.0, -1.0};
    EXPECT_EQ((Complex{0.0, 3.0}), c);
    c -= Complex{ 0 };
    c -= 0;
    EXPECT_EQ((Complex{ 0.0, 3.0 }), c);
}

TEST(Complex_test, can_be_multiplied_with_other_number)
{
    using namespace computoc;

    Complex c{6.0, -2.0};
    EXPECT_EQ(0.0, c * 0.0);
    EXPECT_EQ(c, c * 1.0);
    EXPECT_EQ(c, c * 1);
    EXPECT_EQ((Complex{1.5, -0.5}), c * 0.25);
    EXPECT_EQ((Complex{30.0, 10.0}), (c * Complex{4.0, 3.0}));

    c *= Complex{4.0, 3.0};
    EXPECT_EQ((Complex{30.0, 10.0}), c);
    c *= Complex{ 1 };
    c *= 1;
    EXPECT_EQ((Complex{ 30.0, 10.0 }), c);
}

TEST(Complex_test, can_be_divided_by_other_number)
{
    using namespace computoc;

    Complex c{2.0, 3.0};
    EXPECT_TRUE(isnan(c / Complex{0.0}));
    EXPECT_EQ((Complex{1.0, 1.5}), c / 2.0);
    EXPECT_EQ((Complex{ 1.0, 1.5 }), c / 2);
    EXPECT_EQ((Complex{2.5, 0.5}), (c / Complex{1.0, 1.0}));

    c /= Complex{1.0, 1.0};
    EXPECT_EQ((Complex{2.5, 0.5}), c);
    c /= Complex{ 1 };
    c /= 1;
    EXPECT_EQ((Complex{ 2.5, 0.5 }), c);
}

TEST(Complex_test, have_reciprocal)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ -0.2, -0.4 }), reciprocal(Complex{ -1.0, 2.0 }));
}

TEST(Complex_test, have_absolute_value)
{
    using namespace computoc;

    EXPECT_EQ(0.0, abs(Complex{ 0.0 }));
    EXPECT_EQ(5.0, abs(Complex{ 3.0, 4.0 }));
}

TEST(Complex_test, have_phase_angle)
{
    using namespace computoc;

    EXPECT_TRUE(computoc::isnan(arg(Complex{ 0.0 })));

    EXPECT_EQ(0.0, arg(Complex{ 1.0, 0.0 }));
    EXPECT_EQ(std::numbers::pi / 4.0, arg(Complex{ 1.0, 1.0 }));
}

TEST(Complex_test, have_squared_magnitude)
{
    using namespace computoc;

    EXPECT_EQ(0.0, norm(Complex{ 0.0 }));
    EXPECT_EQ(2.0, norm(Complex{ 1.0, 1.0 }));
    EXPECT_EQ(5.0, norm(Complex{ 1.0, 2.0 }));
    EXPECT_EQ(4.0, norm(Complex{ 0.0, 2.0 }));
}

TEST(Complex_test, have_conjugate)
{
    using namespace computoc;

    EXPECT_EQ(0.0, conj(Complex{ 0.0 }));
    EXPECT_EQ((Complex{ 1.0, -2.0 }), conj(Complex{ 1.0, 2.0 }));
    EXPECT_EQ((Complex{ 0.0, -1.0 }), conj(Complex{ 0.0, 1.0 }));
    EXPECT_EQ(0.5, conj(Complex{ 0.5 }));
}

TEST(Comlex_test, have_projection)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ 1.0, 2.0 }), proj(Complex{ 1.0, 2.0 }));
}

TEST(Complex_test, can_be_constructed_from_magnitude_and_phase_angle)
{
    using namespace computoc;

    EXPECT_TRUE(close(Complex{ 0.0, 1.0 }, polar(1.0, std::numbers::pi / 2.0)));
}

TEST(Complex_test, have_base_e_exponential)
{
    using namespace computoc;

    EXPECT_TRUE(close(Complex{ -1.0, 0.0 }, exp(Complex{0.0, std::numbers::pi})));
}

TEST(Complex_test, have_natural_logarithm)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ 0.0, std::numbers::pi }), log(Complex{ -1.0, 0.0 }));
}

TEST(Complex_test, have_common_logarithm)
{
    using namespace computoc;

    Complex c{ -100.0, 0.0 };
    Complex r = log(c) / std::log(10.0);

    EXPECT_EQ(r, log10(c));
}

TEST(Complex_test, have_complex_power)
{
    using namespace computoc;

    EXPECT_TRUE(close(Complex{ -3.0, 4.0 }, pow(Complex{ 1.0, 2.0 }, 2.0)));
    EXPECT_EQ(exp(Complex{ -std::numbers::pi / 2.0, 0.0 }), pow(Complex{ 0.0, 1.0 }, Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_square_root)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ 0.0, 2.0 }), sqrt(Complex{ -4.0, 0.0 }));
}

TEST(Complex_test, have_sin)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ std::sin(1.0), 0.0 }), sin(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::sinh(1.0) }), sin(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_cos)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ std::cos(1.0), 0.0 }), cos(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ std::cosh(1.0), 0.0 }), cos(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_tan)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ std::tan(1.0), 0.0 }), tan(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::tanh(1.0) }), tan(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_asin)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), sin(asin(Complex{ 1.0, 0.0 })));
    EXPECT_TRUE(close(Complex{ 0.0, 1.0 }, sin(asin(Complex{ 0.0, 1.0 }))));
}

TEST(Complex_test, have_acos)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), cos(acos(Complex{ 1.0, 0.0 })));
    EXPECT_TRUE(close(Complex{ 0.0, 1.0 }, cos(acos(Complex{ 0.0, 1.0 }))));
}

TEST(Complex_test, have_atan)
{
    using namespace computoc;

    EXPECT_TRUE(close(Complex{ 1.0, 0.0 }, tan(atan(Complex{ 1.0, 0.0 }))));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), tan(atan(Complex{ 0.0, 1.0 })));
}

TEST(Complex_test, have_sinh)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ std::sinh(1.0), 0.0 }), sinh(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::sin(1.0) }), sinh(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_cosh)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ std::cosh(1.0), 0.0 }), cosh(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ std::cos(1.0), 0.0 }), cosh(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_tanh)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ std::tanh(1.0), 0.0 }), tanh(Complex{ 1.0, 0.0 }));
    EXPECT_EQ((Complex{ 0.0, std::tan(1.0) }), tanh(Complex{ 0.0, 1.0 }));
}

TEST(Complex_test, have_asinh)
{
    using namespace computoc;

    EXPECT_TRUE(close(Complex{ 1.0, 0.0 }, sinh(asinh(Complex{ 1.0, 0.0 }))));
    EXPECT_EQ((Complex{ 0.0, 1.0 }), sinh(asinh(Complex{ 0.0, 1.0 })));
}

TEST(Complex_test, have_acosh)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), cosh(acosh(Complex{ 1.0, 0.0 })));
    EXPECT_TRUE(close(Complex{ 0.0, 1.0 }, cosh(acosh(Complex{ 0.0, 1.0 }))));
}

TEST(Complex_test, have_atanh)
{
    using namespace computoc;

    EXPECT_EQ((Complex{ 1.0, 0.0 }), tanh(atanh(Complex{ 1.0, 0.0 })));
    EXPECT_TRUE(close(Complex{ 0.0, 1.0 }, tanh(atanh(Complex{ 0.0, 1.0 }))));
}

TEST(Complex_test, copy)
{
    using namespace computoc;

    {
        Complex<int> c1{ 1, 2 };

        Complex<int> c2{ c1 };
        EXPECT_EQ(c1, c2);

        Complex<int> c3{};
        c3 = c2;
        EXPECT_EQ(c2, c3);
    }

    {
        Complex<int> c1{ 1, 2 };

        Complex<double> c2{ c1 };
        EXPECT_EQ(c1, c2);

        Complex<float> c3{};
        c3 = c2;
        EXPECT_EQ(c2, c3);
    }
}

TEST(Complex_test, move)
{
    using namespace computoc;

    {
        Complex<int> c1{ 1, 2 };
        Complex c1c{ c1 };

        Complex<int> c2{ std::move(c1) };
        EXPECT_EQ(c1c, c2);

        Complex<int> c3{};
        c3 = std::move(c2);
        EXPECT_EQ(c1c, c3);
    }

    {
        Complex<int> c1{ 1, 2 };
        Complex c1c{ c1 };

        Complex<double> c2{ std::move(c1) };
        EXPECT_EQ(c1c, c2);

        Complex<float> c3{};
        c3 = std::move(c2);
        EXPECT_EQ(c1c, c3);
    }
}
