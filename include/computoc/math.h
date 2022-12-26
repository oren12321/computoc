#ifndef COMPUTOC_MATH_H
#define COMPUTOC_MATH_H

#include <cmath>
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

        template <Integral T1, Integral T2>
        bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& tol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{}) & = default_rtol<decltype(T1{} - T2{}) > ())
        {
            return abs(a - b) <= tol;
        }

        template <Decimal T1, Decimal T2>
        bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ())
        {
            const decltype(a - b) reps{ rtol * (abs(a) > abs(b) ? abs(a) : abs(b)) };
            return abs(a - b) <= (atol > reps ? atol : reps);
        }

        template <Integral T1, Integral T2>
        auto modulo(const T1& value, const T2& modulus) -> decltype((value% modulus) + modulus)
        {
            return ((value % modulus) + modulus) % modulus;
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

    using details::close;
    using details::modulo;

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