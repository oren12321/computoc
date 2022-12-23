#ifndef COMPUTOC_MATH_H
#define COMPUTOC_MATH_H

#include <cmath>
#include <limits>
#include <computoc/concepts.h>

namespace computoc {
    namespace details {
        template <typename T>
        T default_atol() noexcept
        {
            return T{};
        }

        template <Integral T>
        T default_atol() noexcept
        {
            return T{ 0 };
        }

        template <Decimal T>
        T default_atol() noexcept
        {
            return T{ 1e-8 };
        }

        template <typename T>
        T default_rtol() noexcept
        {
            return T{};
        }

        template <Integral T>
        T default_rtol() noexcept
        {
            return T{ 0 };
        }

        template <Decimal T>
        T default_rtol() noexcept
        {
            return T{ 1e-5 };
        }

        using std::abs;
        using std::acos;
        using std::acosh;
        using std::asin;
        using std::asinh;
        using std::atan;
        using std::atanh;
        using std::cos;
        using std::cosh;
        using std::exp;
        using std::log;
        using std::log10;
        using std::pow;
        using std::sin;
        using std::sinh;
        using std::sqrt;
        using std::tan;
        using std::tanh;
    }

    using details::default_atol;
    using details::default_rtol;

    using details::abs;
    using details::acos;
    using details::acosh;
    using details::asin;
    using details::asinh;
    using details::atan;
    using details::atanh;
    using details::cos;
    using details::cosh;
    using details::exp;
    using details::log;
    using details::log10;
    using details::pow;
    using details::sin;
    using details::sinh;
    using details::sqrt;
    using details::tan;
    using details::tanh;
}

#endif // COMPUTOC_MATH_H