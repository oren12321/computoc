#ifndef COMPUTOC_MATH_H
#define COMPUTOC_MATH_H

#include <cmath>
#include <numbers>
#include <limits>
#include <computoc/concepts.h>

namespace computoc {
    namespace details {
        template <typename T>
        T default_atol() noexcept
        {
            return T{};
        }

        template <Integer T>
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

        template <Integer T>
        T default_rtol() noexcept
        {
            return T{ 0 };
        }

        template <Decimal T>
        T default_rtol() noexcept
        {
            return T{ 1e-5 };
        }

        template <Number T1, Number T2>
        bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            const decltype(a - b) reps{ rtol * (abs(a) > abs(b) ? abs(a) : abs(b)) };
            return abs(a - b) <= (atol > reps ? atol : reps);
        }

        template <Integer T1, Integer T2>
        auto modulo(const T1& value, const T2& modulus) noexcept -> decltype((value% modulus) + modulus)
        {
            return ((value % modulus) + modulus) % modulus;
        }

        template <typename T>
        bool isnan(const T& value) noexcept
        {
            static_assert(std::numeric_limits<T>::has_quiet_NaN);
            return std::isnan(value);
        }

        template <typename T>
        T nan() noexcept
        {
            static_assert(std::numeric_limits<T>::has_quiet_NaN);
            return std::numeric_limits<T>::quiet_NaN();
        }

        template <typename T>
        bool isinf(const T& value) noexcept
        {
            static_assert(std::numeric_limits<T>::has_infinity);
            return std::isinf(value);
        }

        template <typename T>
        T inf() noexcept
        {
            static_assert(std::numeric_limits<T>::has_infinity);
            return std::numeric_limits<T>::infinity();
        }

        template <typename T>
        bool isfinite(const T& value) noexcept
        {
            return std::isfinite(value);
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

        using std::numbers::pi;
    }

    using details::default_atol;
    using details::default_rtol;

    using details::close;
    using details::modulo;

    using details::nan;
    using details::inf;
    using details::isnan;
    using details::isinf;
    using details::isfinite;

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

    using details::pi;
}

#endif // COMPUTOC_MATH_H