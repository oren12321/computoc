#ifndef COMPUTOC_TYPES_COMPLEX_H
#define COMPUTOC_TYPES_COMPLEX_H

#include <stdexcept>
#include <cmath>
#include <complex>

#include <computoc/math.h>
#include <computoc/concepts.h>

namespace computoc {
    namespace details {
        template <Number T = double>
        class Complex {
        public:
            Complex(T r = T{}, T i = T{}) noexcept
                : r_(r), i_(i) {}

            Complex(const Complex<T>& other) = default;
            Complex<T>& operator=(const Complex<T>& other) = default;

            Complex(Complex<T>&& other) = default;
            Complex<T>& operator=(Complex<T>&& other) = default;

            virtual ~Complex() = default;

            template <Number T_o>
            Complex(const Complex<T_o>& other) noexcept
                : r_(other.real()), i_(other.imag())
            {
            }
            template <Number T_o>
            Complex<T>& operator=(const Complex<T_o>& other) noexcept
            {
                r_ = other.real();
                i_ = other.imag();
                return *this;
            }

            template <Number T_o>
            Complex(Complex<T_o>&& other) noexcept
                : r_(other.real()), i_(other.imag())
            {
            }
            template <Number T_o>
            Complex<T>& operator=(Complex<T_o>&& other) noexcept
            {
                r_ = other.real();
                i_ = other.imag();

                return *this;
            }

            [[nodiscard]] T real() const noexcept
            {
                return r_;
            }

            [[nodiscard]] T imag() const noexcept
            {
                return i_;
            }

            [[nodiscard]] Complex<T> operator-() const noexcept
            {
                return { -r_, -i_ };
            }

            [[nodiscard]] Complex<T> operator+() const noexcept
            {
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator+=(const Complex<T_o>& other) noexcept
            {
                r_ += other.real();
                i_ += other.imag();
                return *this;
            }

            Complex<T>& operator+=(const Complex<T>& other) noexcept
            {
                r_ += other.r_;
                i_ += other.i_;
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator+=(T_o other) noexcept
            {
                r_ += other;
                return *this;
            }

            Complex<T>& operator+=(T other) noexcept
            {
                r_ += other;
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator-=(const Complex<T_o>& other) noexcept
            {
                r_ -= other.real();
                i_ -= other.imag();
                return *this;
            }

            Complex<T>& operator-=(const Complex<T>& other) noexcept
            {
                r_ -= other.r_;
                i_ -= other.i_;
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator-=(T_o other) noexcept
            {
                r_ -= other;
                return *this;
            }

            Complex<T>& operator-=(T other) noexcept
            {
                r_ -= other;
                return *this;
            }

            Complex<T>& operator*=(const Complex<T>& other) noexcept
            {
                T nr{ r_ * other.r_ - i_ * other.i_ };
                T ni{ r_ * other.i_ + other.r_ * i_ };
                r_ = nr;
                i_ = ni;
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator*=(const Complex<T_o>& other) noexcept
            {
                T nr{ r_ * other.real() - i_ * other.imag() };
                T ni{ r_ * other.imag() + other.real() * i_ };
                r_ = nr;
                i_ = ni;
                return *this;
            }

            Complex<T>& operator*=(T other) noexcept
            {
                r_ *= other;
                i_ *= other;
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator*=(T_o other) noexcept
            {
                r_ *= other;
                i_ *= other;
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator/=(const Complex<T_o>& other) noexcept
            {
                return operator*=(reciprocal(other));
            }

            template <Number T_o>
            Complex<T>& operator/=(T_o other) noexcept
            {
                r_ /= other;
                i_ /= other;
                return *this;
            }

            Complex<T>& operator/=(const Complex<T>& other) noexcept
            {
                return operator*=(reciprocal(other));
            }

            Complex<T>& operator/=(T other) noexcept
            {
                r_ /= other;
                i_ /= other;
                return *this;
            }

            [[nodiscard]] operator std::complex<T>() const noexcept
            {
                return std::complex<T>{ r_, i_ };
            }

        private:
            T r_{};
            T i_{};
        };

        template <Number T>
        [[nodiscard]] Complex<T> reciprocal(const Complex<T>& c) noexcept
        {
            return { c.real() / (c.real() * c.real() + c.imag() * c.imag()), -c.imag() / (c.real() * c.real() + c.imag() * c.imag()) };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline bool operator==(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline bool operator==(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return lhs.real() == rhs && lhs.imag() == T2{};
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline bool operator==(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return lhs == rhs.real() && T1{} == rhs.imag();
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline bool close(const Complex<T1>& lhs, const Complex<T2>& rhs, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            return close(lhs.real(), rhs.real(), atol, rtol) && close(lhs.imag(), rhs.imag(), atol, rtol);
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline bool close(const Complex<T1>& lhs, T2 rhs, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            return close(lhs.real(), rhs, atol, rtol) && close(lhs.imag(), T2{}, atol, rtol);
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline bool close(T1 lhs, const Complex<T2>& rhs, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            return close(lhs, rhs.real(), atol, rtol) && close(T1{}, rhs.imag(), atol, rtol);
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} + T2{}) > operator+(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs.real() + rhs.real(), lhs.imag() + rhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} + T2{}) > operator+(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return { lhs.real() + rhs, lhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} + T2{}) > operator+(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs + rhs.real(), rhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} - T2{}) > operator-(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs.real() - rhs.real(), lhs.imag() - rhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} - T2{}) > operator-(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return { lhs.real() - rhs, lhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} - T2{}) > operator-(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs - rhs.real(), -rhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} * T2{}) > operator*(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs.real() * rhs.real() - lhs.imag() * rhs.imag(), lhs.real() * rhs.imag() + rhs.real() * lhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} * T2{}) > operator*(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return { lhs.real() * rhs, lhs.imag() * rhs };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} * T2{}) > operator*(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs * rhs.real(), lhs * rhs.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} / T2{}) > operator/(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return operator*(lhs, reciprocal(rhs));
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} / T2{}) > operator/(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return { lhs.real() / rhs, lhs.imag() / rhs };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<decltype(T1{} / T2{}) > operator/(T1 lhs, const Complex<T1>& rhs) noexcept
        {
            return { lhs / rhs.real(), lhs / rhs.imag() };
        }

        template <Number T>
        [[nodiscard]] inline T abs(const Complex<T>& c) noexcept
        {
            return std::abs<T>(c);
        }

        template <Number T>
        [[nodiscard]] inline T arg(const Complex<T>& c) noexcept
        {
            return std::atan(c.imag() / c.real());
        }

        template <Number T>
        [[nodiscard]] inline T norm(const Complex<T>& c) noexcept
        {
            return c.real() * c.real() + c.imag() * c.imag();
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> conj(const Complex<T>& c) noexcept
        {
            return { c.real(), -c.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> proj(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::proj<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> polar(T r, T theta = T{}) noexcept
        {
            std::complex<T> rc = std::polar<T>(r, theta);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> exp(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::exp<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> log(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::log<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> log10(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::log10<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<T1> pow(const Complex<T1>& x, T2 y) noexcept
        {
            std::complex<T1> c = std::pow(std::complex<T1>{x.real(), x.imag()}, y);
            return { c.real(), c.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<T1> pow(T1 x, const Complex<T2>& y) noexcept
        {
            std::complex<T1> c = std::pow(x, std::complex<T2>{y.real(), y.imag()});
            return { c.real(), c.imag() };
        }

        template <Number T1, Number T2>
        [[nodiscard]] inline Complex<T1> pow(const Complex<T1>& x, const Complex<T2>& y) noexcept
        {
            std::complex<T1> c = std::pow(std::complex<T1>{x.real(), x.imag()}, std::complex<T2>{y.real(), y.imag()});
            return { c.real(), c.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> sqrt(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::sqrt<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> sin(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::sin<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> cos(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::cos<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> tan(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::tan<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> asin(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::asin<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> acos(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::acos<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> atan(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::atan<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> sinh(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::sinh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> cosh(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::cosh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> tanh(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::tanh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> asinh(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::asinh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> acosh(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::acosh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        [[nodiscard]] inline Complex<T> atanh(const Complex<T>& c) noexcept
        {
            std::complex<T> rc = std::atanh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T> 
        [[nodiscard]] bool isnan(const Complex<T>& c) noexcept
        {
            return isnan(c.real()) && isnan(c.imag());
        }

        template <Number T>
        [[nodiscard]] Complex<T> nan() noexcept
        {
            return Complex<T>{nan<T>(), nan<T>()};
        }

        template <Number T>
        [[nodiscard]] bool isinf(const Complex<T>& c) noexcept
        {
            return isinf(c.real()) && isinf(c.imag());
        }

        template <Number T>
        [[nodiscard]] Complex<T> inf() noexcept
        {
            return Complex<T>{nan<T>(), nan<T>()};
        }

        template <Number T>
        [[nodiscard]] bool isfinite(const Complex<T>& c) noexcept
        {
            return isfinite(c.real()) && isfinite(c.imag());
        }
    }

    using details::Complex;
    using details::close;
    using details::abs;
    using details::acos;
    using details::acosh;
    using details::arg;
    using details::asin;
    using details::asinh;
    using details::atan;
    using details::atanh;
    using details::conj;
    using details::cos;
    using details::cosh;
    using details::exp;
    using details::log;
    using details::log10;
    using details::norm;
    using details::polar;
    using details::pow;
    using details::proj;
    using details::reciprocal;
    using details::sin;
    using details::sinh;
    using details::sqrt;
    using details::tan;
    using details::tanh;

    using details::nan;
    using details::inf;
    using details::isnan;
    using details::isinf;
    using details::isfinite;
}

#endif // COMPUTOC_TYPES_COMPLEX_H
