#ifndef COMPUTOC_TYPES_COMPLEX_H
#define COMPUTOC_TYPES_COMPLEX_H

#include <stdexcept>
#include <cmath>
#include <complex>

#include <computoc/errors.h>
#include <computoc/math.h>
#include <computoc/concepts.h>

namespace computoc {
    namespace details {
        template <Number T>
        class Complex {
        public:
            Complex(T r = T{}, T i = T{})
                : r_(r), i_(i) {}

            Complex(const Complex<T>& other) = default;
            Complex<T>& operator=(const Complex<T>& other) = default;

            Complex(Complex<T>&& other)
                : r_(other.r_), i_(other.i_)
            {
                other.r_ = T{};
                other.i_ = T{};
            }
            Complex<T>& operator=(Complex<T>&& other)
            {
                if (&other == this) {
                    return *this;
                }

                r_ = other.r_;
                i_ = other.i_;

                other.r_ = T{};
                other.i_ = T{};

                return *this;
            }

            virtual ~Complex() = default;

            template <Number T_o>
            Complex(const Complex<T_o>& other)
                : r_(other.real()), i_(other.imag())
            {
            }
            template <Number T_o>
            Complex<T>& operator=(const Complex<T_o>& other)
            {
                r_ = other.real();
                i_ = other.imag();
                return *this;
            }

            template <Number T_o>
            friend class Complex;

            template <Number T_o>
            Complex(Complex<T_o>&& other)
                : r_(other.real()), i_(other.imag())
            {
                other.r_ = T{};
                other.i_ = T{};
            }
            template <Number T_o>
            Complex<T>& operator=(Complex<T_o>&& other)
            {
                r_ = other.real();
                i_ = other.imag();

                other.r_ = T{};
                other.i_ = T{};

                return *this;
            }

            T real() const noexcept
            {
                return r_;
            }

            T imag() const noexcept
            {
                return i_;
            }

            Complex<T> operator-() const noexcept
            {
                return { -r_, -i_ };
            }

            Complex<T> operator+() const noexcept
            {
                return *this;
            }

            template <Number T_o>
            Complex<T>& operator+=(const Complex<T_o>& other) noexcept
            {
                r_ += other.r_;
                i_ += other.i_;
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
                r_ -= other.r_;
                i_ -= other.i_;
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
                T nr{ r_ * other.r_ - i_ * other.i_ };
                T ni{ r_ * other.i_ + other.r_ * i_ };
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
            Complex<T>& operator/=(const Complex<T_o>& other)
            {
                return operator*=(other.multiplicative_inverse());
            }

            template <Number T_o>
            Complex<T>& operator/=(T_o other)
            {
                COMPUTOC_THROW_IF_FALSE(other != T_o{}, std::overflow_error, "division by zero");

                r_ /= other;
                i_ /= other;
                return *this;
            }

            Complex<T>& operator/=(const Complex<T>& other)
            {
                return operator*=(other.multiplicative_inverse());
            }

            Complex<T>& operator/=(T other)
            {
                COMPUTOC_THROW_IF_FALSE(other != T{}, std::overflow_error, "division by zero");

                r_ /= other;
                i_ /= other;
                return *this;
            }

            template <Number T1, Number T2>
            friend Complex<decltype(T1{} / T2{}) > operator/(const Complex<T1>& lhs, const Complex<T2>& rhs);

            operator std::complex<T>() const noexcept
            {
                return std::complex<T>{ r_, i_ };
            }

        private:
            Complex<T> multiplicative_inverse() const
            {
                COMPUTOC_THROW_IF_FALSE(r_ != T{} || i_ != T{}, std::overflow_error, "division by zero");

                return { r_ / (r_ * r_ + i_ * i_), -i_ / (r_ * r_ + i_ * i_) };
            }

            T r_{};
            T i_{};
        };

        template <Number T1, Number T2>
        inline bool operator==(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
        }

        template <Number T1, Number T2>
        inline bool operator==(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return lhs.real() == rhs && lhs.imag() == T2{};
        }

        template <Number T1, Number T2>
        inline bool operator==(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return lhs == rhs.real() && T1{} == rhs.imag();
        }

        template <Number T1, Number T2>
        inline bool close(const Complex<T1>& lhs, const Complex<T2>& rhs, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            return close(lhs.real(), rhs.real(), atol, rtol) && close(lhs.imag(), rhs.imag(), atol, rtol);
        }

        template <Number T1, Number T2>
        inline bool close(const Complex<T1>& lhs, T2 rhs, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            return close(lhs.real(), rhs, atol, rtol) && close(lhs.imag(), T2{}, atol, rtol);
        }

        template <Number T1, Number T2>
        inline bool close(T1 lhs, const Complex<T2>& rhs, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{}) > (), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{}) > ()) noexcept
        {
            return close(lhs, rhs.real(), atol, rtol) && close(T1{}, rhs.imag(), atol, rtol);
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} + T2{}) > operator+(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs.real() + rhs.real(), lhs.imag() + rhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} + T2{}) > operator+(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return { lhs.real() + rhs, lhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} + T2{}) > operator+(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs + rhs.real(), rhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} - T2{}) > operator-(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs.real() - rhs.real(), lhs.imag() - rhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} - T2{}) > operator-(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return { lhs.real() - rhs, lhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} - T2{}) > operator-(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs - rhs.real(), -rhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} * T2{}) > operator*(const Complex<T1>& lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs.real() * rhs.real() - lhs.imag() * rhs.imag(), lhs.real() * rhs.imag() + rhs.real() * lhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} * T2{}) > operator*(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            return { lhs.real() * rhs, lhs.imag() * rhs };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} * T2{}) > operator*(T1 lhs, const Complex<T2>& rhs) noexcept
        {
            return { lhs * rhs.real(), lhs * rhs.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} / T2{}) > operator/(const Complex<T1>& lhs, const Complex<T2>& rhs)
        {
            return operator*(lhs, rhs.multiplicative_inverse());
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} / T2{}) > operator/(const Complex<T1>& lhs, T2 rhs) noexcept
        {
            COMPUTOC_THROW_IF_FALSE(rhs != T2{}, std::overflow_error, "division by zero");

            return { lhs.real() / rhs, lhs.imag() / rhs };
        }

        template <Number T1, Number T2>
        inline Complex<decltype(T1{} / T2{}) > operator/(T1 lhs, const Complex<T1>& rhs) noexcept
        {
            return { lhs / rhs.real(), lhs / rhs.imag() };
        }

        template <Number T>
        inline T abs(const Complex<T>& c)
        {
            return std::abs<T>(c);
        }

        template <Number T>
        inline T arg(const Complex<T>& c)
        {
            COMPUTOC_THROW_IF_FALSE(c.real() != T{}, std::overflow_error, "division by zero");

            return std::atan(c.imag() / c.real());
        }

        template <Number T>
        inline T norm(const Complex<T>& c)
        {
            return c.real() * c.real() + c.imag() * c.imag();
        }

        template <Number T>
        inline Complex<T> conj(const Complex<T>& c)
        {
            return { c.real(), -c.imag() };
        }

        template <Number T>
        inline Complex<T> proj(const Complex<T>& c)
        {
            std::complex<T> rc = std::proj<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> polar(T r, T theta = T{})
        {
            std::complex<T> rc = std::polar<T>(r, theta);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> exp(const Complex<T>& c)
        {
            std::complex<T> rc = std::exp<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> log(const Complex<T>& c)
        {
            std::complex<T> rc = std::log<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> log10(const Complex<T>& c)
        {
            std::complex<T> rc = std::log10<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<T1> pow(const Complex<T1>& x, T2 y)
        {
            std::complex<T1> c = std::pow(std::complex<T1>{x.real(), x.imag()}, y);
            return { c.real(), c.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<T1> pow(T1 x, const Complex<T2>& y)
        {
            std::complex<T1> c = std::pow(x, std::complex<T2>{y.real(), y.imag()});
            return { c.real(), c.imag() };
        }

        template <Number T1, Number T2>
        inline Complex<T1> pow(const Complex<T1>& x, const Complex<T2>& y)
        {
            std::complex<T1> c = std::pow(std::complex<T1>{x.real(), x.imag()}, std::complex<T2>{y.real(), y.imag()});
            return { c.real(), c.imag() };
        }

        template <Number T>
        inline Complex<T> sqrt(const Complex<T>& c)
        {
            std::complex<T> rc = std::sqrt<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> sin(const Complex<T>& c)
        {
            std::complex<T> rc = std::sin<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> cos(const Complex<T>& c)
        {
            std::complex<T> rc = std::cos<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> tan(const Complex<T>& c)
        {
            std::complex<T> rc = std::tan<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> asin(const Complex<T>& c)
        {
            std::complex<T> rc = std::asin<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> acos(const Complex<T>& c)
        {
            std::complex<T> rc = std::acos<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> atan(const Complex<T>& c)
        {
            std::complex<T> rc = std::atan<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> sinh(const Complex<T>& c)
        {
            std::complex<T> rc = std::sinh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> cosh(const Complex<T>& c)
        {
            std::complex<T> rc = std::cosh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> tanh(const Complex<T>& c)
        {
            std::complex<T> rc = std::tanh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> asinh(const Complex<T>& c)
        {
            std::complex<T> rc = std::asinh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> acosh(const Complex<T>& c)
        {
            std::complex<T> rc = std::acosh<T>(c);
            return { rc.real(), rc.imag() };
        }

        template <Number T>
        inline Complex<T> atanh(const Complex<T>& c)
        {
            std::complex<T> rc = std::atanh<T>(c);
            return { rc.real(), rc.imag() };
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
    using details::sin;
    using details::sinh;
    using details::sqrt;
    using details::tan;
    using details::tanh;
}

#endif // COMPUTOC_TYPES_COMPLEX_H
