#ifndef COMPUTOC_MATH_H
#define COMPUTOC_MATH_H

#include <cmath>
#include <numbers>
#include <limits>
#include <computoc/concepts.h>

namespace computoc {
    namespace details {
        template <typename T>
        [[nodiscard]] inline constexpr T default_atol() noexcept
        {
            return T{};
        }

        template <Integer T>
        [[nodiscard]] inline constexpr T default_atol() noexcept
        {
            return T{ 0 };
        }

        template <Decimal T>
        [[nodiscard]] inline constexpr T default_atol() noexcept
        {
            return T{ 1e-8 };
        }

        template <typename T>
        [[nodiscard]] inline constexpr T default_rtol() noexcept
        {
            return T{};
        }

        template <Integer T>
        [[nodiscard]] inline constexpr T default_rtol() noexcept
        {
            return T{ 0 };
        }

        template <Decimal T>
        [[nodiscard]] inline constexpr T default_rtol() noexcept
        {
            return T{ 1e-5 };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline constexpr bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            const decltype(a - b) reps{ rtol * (abs(a) > abs(b) ? abs(a) : abs(b)) };
            return abs(a - b) <= (atol > reps ? atol : reps);
        }

        template <Integer T1, Integer T2>
        [[nodiscard]] inline constexpr auto modulo(const T1& value, const T2& modulus) noexcept -> decltype((value% modulus) + modulus)
        {
            decltype((value % modulus) + modulus) res{ value };
            while (res < 0) {
                res += modulus;
            }
            while (res >= modulus) {
                res -= modulus;
            }
            return res;
        }

        template <typename T>
        [[nodiscard]] inline constexpr bool isnan(const T& value) noexcept
        {
            static_assert(std::numeric_limits<T>::has_quiet_NaN);
            return std::isnan(value);
        }

        template <typename T>
        [[nodiscard]] inline constexpr T nan() noexcept
        {
            static_assert(std::numeric_limits<T>::has_quiet_NaN);
            return std::numeric_limits<T>::quiet_NaN();
        }

        template <typename T>
        [[nodiscard]] inline constexpr bool isinf(const T& value) noexcept
        {
            static_assert(std::numeric_limits<T>::has_infinity);
            return std::isinf(value);
        }

        template <typename T>
        [[nodiscard]] inline constexpr T inf() noexcept
        {
            static_assert(std::numeric_limits<T>::has_infinity);
            return std::numeric_limits<T>::infinity();
        }

        template <typename T>
        [[nodiscard]] inline constexpr bool isfinite(const T& value) noexcept
        {
            return std::isfinite(value);
        }

        template <typename T>
        [[nodiscard]] inline constexpr T max() noexcept
        {
            return std::numeric_limits<T>::max();
        }

        using std::abs;
        using std::acos;
        using std::acosh;
        using std::asin;
        using std::asinh;
        using std::atan;
        using std::atanh;
        using std::ceil;
        using std::cos;
        using std::cosh;
        using std::exp;
        using std::floor;
        using std::log;
        using std::log10;
        using std::pow;
        using std::round;
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
    using details::ceil;
    using details::cos;
    using details::cosh;
    using details::exp;
    using details::floor;
    using details::log;
    using details::log10;
    using details::pow;
    using details::round;
    using details::sin;
    using details::sinh;
    using details::sqrt;
    using details::tan;
    using details::tanh;

    using details::pi;
}

#endif // COMPUTOC_MATH_H